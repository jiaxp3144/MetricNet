from mylib import listFile, addNoise
from scipy.io import wavfile
import os
from utils import dataSetup
import shutil
from tqdm import tqdm
import random


def generate(org_files, noise_files, snrs, tar_path):
    ''' add noise to org_files and save the mixed files to tar_path
        each file in org_files is processed with all noises and snrs.
        inputs:
            org_files: a list of file path
            noise_files: a list of noise path
            snrs: a list of target snrs
            tar_path: path to save the mixed files
        outputs:
            None
    '''
    if os.path.isdir(tar_path):
        shutil.rmtree(tar_path)
    os.makedirs(tar_path)

    # pre-load noise_files
    noise_sigs = {}
    noise_names = {}
    noise_num = len(noise_files)
    for i in range(noise_num):
        cur_noise_file = noise_files[i]
        [fs, sig] = wavfile.read(cur_noise_file)
        noise_sigs[i] = sig
        noise_names[i] = os.path.basename(cur_noise_file)[:-4]

    # proc each file
    for cur_org in tqdm(org_files):
        [fs, sig_cln] = wavfile.read(cur_org)
        basename = os.path.basename(cur_org)[:-4]
        cur_path = os.path.join(tar_path, basename)
        os.mkdir(cur_path)
        # write clean sig
        name_cln = os.path.join(cur_path, 'clean.wav')
        wavfile.write(name_cln, fs, sig_cln)
        # write noisy sigs
        for n_i in range(noise_num):
            for snr in snrs:
                sig_mix = addNoise(sig_cln, noise_sigs[n_i], snr)
                name_mix = '{}_{}dB.wav'.format(noise_names[n_i], snr)
                name_mix = os.path.join(cur_path, name_mix)
                wavfile.write(name_mix, fs, sig_mix)
    print("Done!")


def main():
    flag_train = False
    flag_valid = False
    flag_infer = True
    args = dataSetup()
    # get noise_files list
    noise_files = []
    for cur_type in args.noise_types:
        cur_noise_file = os.path.join(args.noise_path, cur_type+'.wav')
        print(cur_noise_file)
        noise_files.append(cur_noise_file)

    # train set
    if flag_train:
        print("processing train set...")
        tar_path = os.path.join(args.tar_path, 'train')
        file_list = listFile(args.train_cln_path, 'wav')
        file_list = random.sample(file_list, args.train_num)
        generate(file_list, noise_files, args.snrs, tar_path)
    
    # valid set
    if flag_valid:
        print("processing valid set...")
        tar_path = os.path.join(args.tar_path, 'valid')
        file_list = listFile(args.valid_cln_path, 'wav')
        file_list = random.sample(file_list, args.valid_num)
        generate(file_list, noise_files, args.snrs, tar_path)

    # infer set
    if flag_infer:
        print("processing infer set...")
        tar_path = os.path.join(args.tar_path, 'infer', 'babble_0dB')
        file_list = listFile(args.infer_cln_path, 'wav')
        noise_files = [os.path.join(args.noise_path, 'babble.wav')]
        snrs = [0]
        generate(file_list, noise_files, snrs, tar_path)
    

if __name__ == '__main__':
    main()

