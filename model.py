import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Input, Conv1D, Conv2D, Reshape
from tensorflow.keras.layers import Lambda, SeparableConvolution1D, Add, PReLU
from tensorflow.keras.layers import Multiply, Concatenate, LayerNormalization
import tensorflow.keras.callbacks as cbs
from tensorflow.keras import backend as K
import librosa
import logging
import shutil

from dataloader import MetricNetDataLoader
from utils import mySetup
from evaluate_mp import evaluate_mp
from mylib import calcSDR
from scipy.stats import spearmanr, pearsonr

import warnings
# warnings.filterwarnings('ignore')

os.environ['CUDA_VISIBLE_DEVICES'] = '7'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def _myLoss(y_true, y_pred):
    upp = K.mean(y_true**2)
    low = K.mean((y_true-y_pred)**2)
    loss = -10*K.log(upp/low)/K.log(10.0)
    return loss

 
class MetricNet(object):
    def __init__(self, args):
        # general params
        self.max_epochs = args.max_epochs
        self.train_steps_per_epoch = args.train_steps_per_epoch
        self.valid_steps_per_epoch = args.valid_steps_per_epoch
        self.ckpt_file = args.ckpt_file
        self.train_log_csv_file = args.train_log_csv_file
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate

        # encoder params
        self.encode_channel_num = args.encode_channel_num
        self.encode_frame_len = args.encode_frame_len
        self.encode_hop_size = args.encode_hop_size
        self.encode_activation = args.encode_activation

        # tcn params
        self.tcn_channel_num = args.tcn_channel_num
        self.tcn_bottle_channel_num = args.tcn_bottle_channel_num
        self.tcn_block_num = args.tcn_block_num
        self.tcn_block_layer_num = args.tcn_block_layer_num
        self.tcn_kernerl_size = args.tcn_kernerl_size

        self.strategy = tf.distribute.MirroredStrategy()
        print('Number of devices: {}'.format(self.strategy.num_replicas_in_sync))

        # with self.strategy.scope():
        #     self._build_model()
        self._build_model()
        self.model_path = args.model_path

    def _build_model(self):
        # assistant functions
        def _stack(inputs, axis):
            return K.stack(inputs, axis)
        def _expand_dims(inputs, axis):
            return K.expand_dims(inputs, axis)
        def _reduce_mean(inputs, axis=None):
            if axis is None:
                return K.mean(inputs, -1, keepdims=True)
            else:
                return K.mean(inputs, axis, keepdims=True)
        def _element_multiply(inputs):
            in_1, in_2 = inputs
            return in_1 * in_2
        def _overlapp_add(inputs, frame_step):
            result = tf.signal.overlap_and_add(
                signal=inputs,
                frame_step=frame_step)
            return result
        def _getFrameNum(sig_len):
            sig_len = sig_len[0,0]
            sig_len = sig_len - (self.encode_frame_len-self.encode_hop_size)
            frame_num = sig_len // self.encode_hop_size
            return frame_num
        def _norm(inputs):
            [diff, x] = inputs
            xx = K.mean(x**2, [-1,-2], keepdims=True)**0.5
            diff = diff / (xx + 1e-6)
            return diff

        # inputs
        inputs_mix = Input(shape=(None,), name='inputs_mix')
        expand_dims_layer = Lambda(_expand_dims, arguments={'axis':-1})
        x_mix = expand_dims_layer(inputs_mix)  # [bs, sig_len, 1]

        # *************************************
        # enhance Net
        encoder_enh = Conv1D(filters=self.encode_channel_num,
                             kernel_size=self.encode_frame_len,
                             strides=self.encode_hop_size,
                             padding='valid',
                             activation=self.encode_activation)
        encoded_mix = encoder_enh(x_mix)  # [bs, frame_num, channel_num]
        encoded_input = encoded_mix
        x = encoded_input
        for i in range(self.tcn_block_num):
            for j in range(self.tcn_block_layer_num):
                cur_name = 'enh_tcn_{}_{}_'.format(i, j)
                y = x
                y = LayerNormalization(name=cur_name+'LN_0')(y)
                y = Conv1D(filters=self.tcn_channel_num,
                           kernel_size=1,
                           strides=1,
                           padding='same',
                           name=cur_name+'bottleneck')(y)
                y = PReLU(shared_axes=[1], name=cur_name+'PReLU_0')(y)
                y = LayerNormalization(name=cur_name+'LN_1')(y)
                dilation_rate = int(2 ** j)
                y = SeparableConvolution1D(filters=self.tcn_bottle_channel_num,
                                           kernel_size=self.tcn_kernerl_size,
                                           strides=1,
                                           padding='same',
                                           dilation_rate=dilation_rate,
                                           name=cur_name+'conv')(y)
                y = PReLU(shared_axes=[1], name=cur_name+'PReLU_1')(y)
                x = Add()([x, y])
        
        mask = Dense(self.encode_channel_num,
                     activation='tanh',
                     name='mask_gen')(x)

        # outputs
        diff = Lambda(_element_multiply, name='masking')([mask, encoded_input])
        x = tf.keras.layers.Add()([encoded_input, diff])
        diff = Lambda(_norm)([diff, x])
        x = Dense(self.encode_frame_len, name='decoder_to_time_domain')(x)
        x = Lambda(_overlapp_add,
                   arguments={'frame_step':self.encode_hop_size},
                   name='overlap_and_add')(x)
        outputs_enhan = x
        
        # *******************************
        # metric net
        # TCNs
        x = Conv1D(filters=self.tcn_bottle_channel_num,
                   kernel_size=1,
                   strides=1,
                   padding='same',
                   activation='elu',
                   name='tcn_bottleneck_layer')(diff)
        # x = encoded_input
        for i in range(1):
            for j in range(4):
                cur_name = 'tcn_{}_{}_'.format(i, j)
                y = x
                # y = LayerNormalization(name=cur_name+'LN_0')(y)
                y = Conv1D(filters=self.tcn_channel_num,
                           kernel_size=1,
                           strides=1,
                           padding='same',
                           name=cur_name+'bottleneck')(y)
                y = PReLU(shared_axes=[1], name=cur_name+'PReLU_0')(y)
                # y = LayerNormalization(name=cur_name+'LN_1')(y)
                dilation_rate = int(2 ** j)
                y = SeparableConvolution1D(filters=self.tcn_bottle_channel_num,
                                           kernel_size=self.tcn_kernerl_size,
                                           strides=1,
                                           padding='same',
                                           dilation_rate=dilation_rate,
                                           name=cur_name+'conv')(y)
                y = PReLU(shared_axes=[1], name=cur_name+'PReLU_1')(y)
                x = Add()([x, y])
        
        # outputs
        x = Dense(1, name="frame_score")(x)
        x = keras.layers.Flatten(name="reshape_to_bs_fn")(x)
        x = Lambda(_reduce_mean, name="utt_score")(x)
        
        outputs_metric = x
        
        self.model = keras.Model(inputs=inputs_mix,
                                 outputs=[outputs_metric, outputs_enhan])
        lr = self.learning_rate
        self.model.compile(optimizer=keras.optimizers.Adam(lr=lr),
                           loss={"utt_score":_myLoss,
                                 "overlap_and_add":_myLoss},
                           loss_weights={"utt_score":0.5,
                                         "overlap_and_add":0.5})
        # self.model = keras.Model(inputs=inputs_mix,
        #                          outputs=outputs_enhan)
        # lr = self.learning_rate
        # self.model.compile(optimizer=keras.optimizers.Adam(lr=lr),
        #                    loss=_myLoss)
 
    def _data_gen(self, inputs):
        for data in inputs:
            try:
                [mix, lbl, cln] = data
                yield mix, [lbl, cln]
            except GeneratorExit:
                break

    def train(self, args):
        os.mkdir(self.model_path)
        # init dataset
        train_data = MetricNetDataLoader('train', args)
        valid_data = MetricNetDataLoader('valid', args)
        inputs = train_data.get_next()
        inputs_v = valid_data.get_next()

        # iii = 0
        # for data in inputs:
        #     [mix, lbl, cln] = data
        #     print(mix.shape, lbl.shape, cln.shape)
        #     self.model.fit(x=mix, y=[lbl, cln],
        #                    steps_per_epoch=1)
        #     iii += 1
        #     if iii >=5:
        #         break

        # callbacks
        ckpt_callback = cbs.ModelCheckpoint(filepath=self.ckpt_file,
                                            monitor='val_loss',
                                            mode='min',
                                            verbose=0,
                                            save_best_only=True,
                                            save_weights_only=True)
        lr_callback = cbs.ReduceLROnPlateau(monitor='val_loss',
                                            factor=0.5,
                                            patience=3,
                                            min_delta=1e-7)
        csv_callback = cbs.CSVLogger(self.train_log_csv_file)
        estop_callback = cbs.EarlyStopping(monitor='val_loss',
                                           patience=10)

        # compile model
        # def _myloss(y_true, y_pred):
        #     # return K.mean((y_pred - y_true)**2)**0.5
        #     # return K.log(K.mean((y_pred - y_true)**2))
        #     low = K.mean((y_pred - y_true)**2)
        #     upp = K.mean(y_true**2)
        #     snr = 10*(K.log(upp/low)/K.log(10.0))
        #     return -1 * snr

        # lr = args.learning_rate
        # with self.strategy.scope():
        #     self.model.compile(optimizer=keras.optimizers.Adam(lr=lr),
        #                        loss=_myloss)
        # for cur_data in inputs:
        #     x, y = cur_data
        #     self.model.fit(x=x, y=y)

        # self.model.fit(inputs,
        #                steps_per_epoch=self.train_steps_per_epoch,
        #                epochs=self.max_epochs,
        #                verbose=1,
        #                callbacks=[ckpt_callback, lr_callback,
        #                           csv_callback, estop_callback],
        #                validation_data=inputs_v,
        #                validation_steps=self.valid_steps_per_epoch,
        #               )
        self.model.fit(self._data_gen(inputs),
                       steps_per_epoch=self.train_steps_per_epoch,
                       epochs=self.max_epochs,
                       verbose=1,
                       callbacks=[ckpt_callback, lr_callback,
                                  csv_callback, estop_callback],
                       validation_data=self._data_gen(inputs_v),
                       validation_steps=self.valid_steps_per_epoch,
                      )


    def test(self, args):
        args.batch_size = 1
        infer_data = MetricNetDataLoader('infer', args)
        inputs = infer_data.get_next()
        
        # self.model = keras.models.load_model(args.infer_ckpt_file,
        #                                      custom_objects={
        #                                          "_myloss":_myloss
        #                                      }
        #                                     )

        self.model.load_weights(args.infer_ckpt_file)

        res_file = args.result_file
        res_f = open(res_file, 'w')
        fs = args.sample_rate

        # self.model.evaluate(inputs, steps=5460, verbose=2)

        res = np.zeros(10000)
        lbls = np.zeros(10000)
        sdrs = np.zeros(10000)
        org_sdrs = np.zeros(10000)

        def _predict(in_sig):
            return self.model.predict_on_batch(in_sig)

        # max_len = 0  # 313520
        # for cur_sigs in inputs:
        #     cur_mix, cur_cln = cur_sigs
        #     print(cur_mix.shape)
        #     if max_len < cur_mix.shape[-1]:
        #         max_len = cur_mix.shape[-1]
        # print("max_len =", max_len)
        
        count = 0
        for cur_sigs in inputs:
            cur_data, cur_lbl, cur_cln = cur_sigs
            # cur_out = self.model.predict_on_batch(cur_mix)
            cur_out_metric, cur_out_sig = _predict(cur_data)
            # self.model.evaluate(cur_mix, cur_cln, verbose=2)
            cur_out_sig = cur_out_sig[0,:]
            cur_cln = cur_cln[0,:]
            cur_data = cur_data[0,:]
            cur_out_metric = cur_out_metric[0,0]
            res[count] = cur_out_metric
            lbls[count] = cur_lbl[0]
            cur_sdr = calcSDR(cur_cln, cur_out_sig)
            cur_org_sdr = calcSDR(cur_cln, cur_data)
            sdrs[count] = cur_sdr
            org_sdrs[count] = cur_org_sdr
            cur_str = "{}: pred = {:.5f}, label = {:.5f}".format(
                count, res[count], lbls[count])
            cur_str += ", org_sdr = {:.5f}dB".format(cur_org_sdr)
            cur_str += ", sdr = {:.5f}dB".format(cur_sdr)
            print(cur_str)
            res_f.write(cur_str + '\n')
            count += 1
        res = res[:count]
        lbls = lbls[:count]
        sdrs = sdrs[:count]
        srcc = spearmanr(res, lbls)[0]
        lcc = pearsonr(res, lbls)[0]
        rmse = np.mean((res - lbls)**2)**0.5
        cur_str = "\npred = {:.5f}({:.5f}), srcc = {:.5f}, lcc = {:.5f}".format(
            np.mean(res), np.mean(lbls), srcc, lcc)
        cur_str += ", rmse = {:.5f}, sdr = {:.5f}dB --> {:.5f}dB".format(
            rmse, np.mean(org_sdrs), np.mean(sdrs))
        print(cur_str)
        res_f.write(cur_str+'\n')

        res_f.close()

    def metricNet_dataGen(self, args):
        fs = 16000
        self.model.load_weights(args.mdg_ckpt_file)
        def _get_input_len(in_sig):
            sig_len = in_sig.shape[-1]
            out_len = 400000
            while sig_len > out_len:
                out_len = out_len*2
            out_sig = np.zeros([1,out_len])
            out_sig[0,:sig_len] = in_sig
            return out_sig, sig_len

        results = {}
        for cur_flag in args.proc_flags:
            results[cur_flag] = {}
            results[cur_flag]['mix'] = np.zeros(35000)
            results[cur_flag]['enh'] = np.zeros(35000)
            cur_cln_path = os.path.join(args.m_basepath, cur_flag, 'clean')
            cur_mix_path = os.path.join(args.m_basepath, cur_flag, 'noisy')
            cur_enh_path = os.path.join(args.m_basepath, cur_flag, 'enhan')
            record_file = os.path.join(args.m_basepath, cur_flag, 'record.txt')
            f = open(record_file, 'w')
            if os.path.isdir(cur_enh_path):
                shutil.rmtree(cur_enh_path)
            os.mkdir(cur_enh_path)
            filename_list = os.listdir(cur_cln_path)
            count = 0
            for cur_file in filename_list:
                cur_cln_file = os.path.join(cur_cln_path, cur_file)
                cur_mix_file = os.path.join(cur_mix_path, cur_file)
                cur_enh_file = os.path.join(cur_enh_path, cur_file)
                [cur_cln, fs] = librosa.load(cur_cln_file, fs)
                [cur_mix, fs] = librosa.load(cur_mix_file, fs)
                cur_inp, sig_len = _get_input_len(cur_mix)
                cur_out = self.model.predict_on_batch(cur_inp)
                cur_out = cur_out[0,:sig_len].numpy()
                librosa.output.write_wav(cur_enh_file, cur_out, fs)
                cur_mix_sdr = calcSDR(cur_cln, cur_mix)
                cur_enh_sdr = calcSDR(cur_cln, cur_out)
                results[cur_flag]['mix'][count] = cur_mix_sdr
                results[cur_flag]['enh'][count] = cur_enh_sdr
                cur_str = "{} {}: {} {:.5f} --> {:.5f} dB".format(
                    cur_flag, count, cur_file, cur_mix_sdr, cur_enh_sdr)
                print(cur_str)
                f.write(cur_str + '\n')
                count += 1
            f.close()
            results[cur_flag]['mix'] = results[cur_flag]['mix'][:count]
            results[cur_flag]['enh'] = results[cur_flag]['enh'][:count]
        print("\n")
        for cur_flag in args.proc_flags:
            print("{} set: {:.3f}dB --> {:.3f}dB".format(
                cur_flag,
                np.mean(results[cur_flag]['mix']),
                np.mean(results[cur_flag]['enh'])))



def evaluate(args):
    evaluate_mp(args.result_dir, args.result_file)



def test():
    args = mySetup()
    myNet = MetricNet(args)
    # myNet.model.summary()
    myNet.train(args)
    print("training finished!")

    myNet.test(args)
    print("testing finished!")


def test_infer_only():
    args_infer = mySetup()
    myNet_infer = MetricNet(args_infer)

    myNet_infer.test(args_infer)
    # evaluate(args_infer)
    print("testing finished!")

def MetricData():
    args = mySetup()
    myNet = enhanceNet(args)
    myNet.metricNet_dataGen(args)
    print("finished!")



if __name__ == '__main__':
    # test()
    test_infer_only()
    # MetricData()

