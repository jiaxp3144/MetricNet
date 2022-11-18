import tensorflow as tf
import os
import logging
import librosa
import numpy as np
from tqdm import tqdm
from mylib import calcSDR, calcSISDR, addNoise
from pypesq import pesq as calcPesq
from pystoi.stoi import stoi as calcStoi
from loizou import rev_llr as calcLLR


# from utils import create_dir


class MetricNetDataLoader():
    def __init__(self, mode, args):
        if mode != "train" and mode != "valid" and mode != "infer":
            raise ValueError("mode: {} while mode should be "
                             "'train', 'valid', or 'infer'".format(mode))

        if not os.path.isdir(args.data_path):
            raise ValueError("args.data_path not exist!")

        if mode == 'train':
            self.tfr = args.train_tfr
            self.wav_dir = args.train_data_path
            self.batch_size = args.batch_size
        elif mode == 'valid':
            self.tfr = args.valid_tfr
            self.wav_dir = args.valid_data_path
            self.batch_size = args.batch_size
        else:
            self.tfr = args.infer_tfr
            self.wav_dir = args.infer_data_path
            self.batch_size = 1

        self.mode = mode
        self.sample_rate = args.sample_rate
        self.time_len = args.time_len
        self.label_index = args.label_index

        if not os.path.isdir(os.path.dirname(self.tfr)):
            os.makedirs(os.path.dirname(self.tfr))

        if args.regenerate:
            if os.path.isfile(self.tfr):
                os.remove(self.tfr)

        if not os.path.isfile(self.tfr):
            raise ValueError("tfr file not found!")
            # self._encode()

    def _float_list_feature(self, value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    def get_next(self):
        logging.info("Loading data from {}".format(self.tfr))
        with tf.name_scope("input"):
            dataset = tf.data.TFRecordDataset(self.tfr).map(self._decode,
                                                           num_parallel_calls=4)
            if self.mode == "train":
                dataset = dataset.shuffle(2000 + 5 * self.batch_size)
            dataset = dataset.batch(self.batch_size,drop_remainder=True)
            dataset = dataset.prefetch(2000 + self.batch_size * 5)
            if self.mode != 'infer':
                dataset = dataset.repeat()
            return dataset

    def _getFileList(self, path, exc='clean.wav'):
        filenames = os.listdir(path)
        results = []
        for cur_file_name in filenames:
            if cur_file_name == exc:
                continue
            results.append(os.path.abspath(os.path.join(path, cur_file_name)))
        return results

    # # encode not needed here, tfr files are generated before.
    # def _encode(self):
    #     logging.info("Writing {}".format(self.tfr))
    #     # with tf.python_io.TFRecordWriter(self.tfr) as writer:
    #     with tf.io.TFRecordWriter(self.tfr) as writer:
    #         filedirs = os.listdir(self.wav_dir)
    #         for filedir in tqdm(filedirs):
    #             cur_dir = os.path.join(self.wav_dir, filedir)
    #             [cln_sig, fs] = librosa.load(os.path.join(cur_dir,
    #                                                       'clean.wav'),
    #                                          self.sample_rate)

    #             file_list = self._getFileList(cur_dir)
    #             for cur_mix_file in file_list:
    #                 [mix_sig, fs] = librosa.load(cur_mix_file,
    #                                              self.sample_rate)

    #                 def write(l, r):
    #                     cur_cln = cln_sig[l:r]
    #                     cur_mix = mix_sig[l:r]
    #                     example = tf.train.Example(
    #                         features=tf.train.Features(
    #                             feature={
    #                                 "mix": self._float_list_feature(cur_mix),
    #                                 "cln": self._float_list_feature(cur_cln)
    #                             }))
    #                     writer.write(example.SerializeToString())

    #                 now_length = mix_sig.shape[-1]

    #                 if self.mode == "train" or self.mode == 'valid':
    #                     target_length = int(self.time_len * self.sample_rate)
    #                     if now_length < target_length*0.5:
    #                         continue
    #                     if now_length % target_length >= 0.25 * target_length:
    #                         padding_len = target_length - (now_length % target_length)
    #                         
    #                         cln_sig = np.hstack((cln_sig, cln_sig[-1*padding_len:]))
    #                         mix_sig = np.hstack((mix_sig, mix_sig[-1*padding_len:]))
    #                         now_length = mix_sig.shape[-1]

    #                     # stride here controls the overlap of each slices
    #                     stride = int(self.time_len * self.sample_rate)
    #                     for i in range(0, now_length - target_length+1, stride):
    #                         write(i, i + target_length)
    #                 else:
    #                     now_length = now_length - (now_length%80)
    #                     write(0, now_length)

    def _decode(self, serialized_example):
        example = tf.io.parse_single_example(
            serialized_example,
            features={
                "mix": tf.io.VarLenFeature(tf.float32),
                "cln": tf.io.VarLenFeature(tf.float32),
                'labels': tf.io.VarLenFeature(tf.float32)
            },
        )
        mix_sig = tf.sparse.to_dense(example["mix"])
        cln_sig = tf.sparse.to_dense(example["cln"])
        labels = tf.sparse.to_dense(example["labels"])
        label = labels[self.label_index]
        return mix_sig, label, cln_sig
        # return mix_sig, cln_sig


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'   
    from utils import mySetup
    args = mySetup()

    # flags = ['train', 'valid', 'infer']
    flags = ['valid']
    for flag in flags:
        dataloader = MetricNetDataLoader(flag, args)
        data = dataloader.get_next()
        count = 0
        for c_d in data:
            count += 1
            [mix, lbl] = c_d
            if count == 1:
                print(mix.shape, lbl)
            if count % 50 == 0:
                print(count)
        print('count =', count)
    
    
