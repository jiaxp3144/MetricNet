import os
import time
import argparse


def getSavePath(cur_path='./model'):
    cur_path = './model'
    cur_time = time.strftime('%Y-%m-%d_%H:%M:%S',time.localtime(time.time()))
    cur_path = cur_path + '_' + cur_time
    return cur_path


def mySetup():
    # settings for enhanceNet train and test
    parser = argparse.ArgumentParser(description="net_train_test")
    args = parser.parse_args()

    # dataloader params
    args.regenerate = False  # regenerate the tfr files
    args.data_path = "/data/jiaxp/Projects/MetricNet/datasets/SingleNoiseTest/"
    args.train_tfr = os.path.join(args.data_path, "train.tfr")
    args.valid_tfr = os.path.join(args.data_path, "valid.tfr")
    # args.infer_tfr = os.path.join(args.data_path, "infer.tfr")
    args.infer_tfr = os.path.join(args.data_path, "volvo_NO.tfr")
    args.train_data_path = os.path.join(args.data_path, 'train')
    args.valid_data_path = os.path.join(args.data_path, 'valid')
    args.infer_data_path = os.path.join(args.data_path, 'infer')

    args.batch_size = 8 * 4 
    args.sample_rate = 16000
    args.time_len = 2
    args.label_index = 1  # 0: sdr, 1: pesq, 2: stoi

    # train params
    args.max_epochs = 100
    args.train_steps_per_epoch = 38271 // args.batch_size
    args.valid_steps_per_epoch = 6822 // args.batch_size
    args.model_path = getSavePath()
    args.ckpt_file = os.path.join(args.model_path, 'mymodel.ckpt')
    args.train_log_csv_file = os.path.join(args.model_path, 'train.log')
    args.learning_rate = 0.001

    # test params
    # args.infer_ckpt_file = args.ckpt_file
    args.infer_ckpt_file =\
    './model_2020-05-06_11:30:16/mymodel.ckpt'
    args.infer_model_path = os.path.dirname(args.infer_ckpt_file)
    args.result_file = os.path.join(args.infer_model_path, 'volvo_NO_results.txt')

    # net params
    # encoder
    # args.encode_channel_num = 129
    # args.encode_frame_len = 256
    # args.encode_hop_size = 128
    args.encode_channel_num = 128
    args.encode_frame_len = 20
    args.encode_hop_size = 10
    args.encode_activation = 'linear'
    # tcn
    args.tcn_channel_num = 256
    # args.tcn_bottle_channel_num = 129
    args.tcn_bottle_channel_num = 128
    args.tcn_block_layer_num = 8
    args.tcn_block_num = 2
    args.tcn_kernerl_size = 3

    # params for metricNet_dataGen
    args.m_basepath = '/share/datasets/MetricNet_data/'
    args.proc_flags = ['train', 'valid', 'infer']
    args.mdg_ckpt_file = './saved_model_2020-04-16_babble_0_9.217/mymodel.ckpt'


    return args


def dataSetup():
    # settings for dataprepare
    parser = argparse.ArgumentParser(description="data_gen")
    args = parser.parse_args()

    args.tar_path = '/data/jiaxp/datasets/MetricNet_enhanceNet/'
    args.train_cln_path = '/share/datasets/WSJ0/si_tr_s'
    args.train_num = 100  # 100 x 15 x 6 = 9000 train files
    args.valid_cln_path = '/share/datasets/WSJ0/si_dt_05'
    args.valid_num = 25
    args.infer_cln_path = '/share/datasets/WSJ0/si_et_05'
    args.infer_num = -1
    
    args.noise_path = '/share/datasets/NoiseX_16k/'
    # args.noise_types = ['babble','buccaneer1','buccaneer2','destroyerengine',
    #                     'destroyerops','f16','factory1','factory2','hfchannel',
    #                     'leopard']
    # args.noise_types = ['m109','machinegun','pink','volvo','white']
    args.noise_types = ['m109','machinegun','pink','volvo','white','babble',
                        'buccaneer1','buccaneer2','destroyerengine',
                        'destroyerops','f16','factory1','factory2','hfchannel',
                        'leopard']
    args.snrs = [-10, -5, 0, 5, 10, 15]

    return args



