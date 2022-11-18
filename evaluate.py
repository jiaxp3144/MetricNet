import os
import numpy as np
from scipy.io import wavfile
from mylib import calcSDR, listFile, calcSISDR
from pypesq import pesq as calcPesq
from pystoi.stoi import stoi as calcStoi
from loizou import rev_llr as calcLLR
from pysrmr.srmr import srmr as calcSRMR


# flag = 'infer'
# base_path = '/data/machao/dereverberation'
# reverb_path = os.path.join(base_path, flag, 'mix')
# direct_path = os.path.join(base_path, flag, 's2')
# 
# reverb_list = listFile(reverb_path, 'wav')
# file_num = len(reverb_list)
# 
# sdrs = np.zeros(file_num)
# i = 0
# for cur_reverb_file in reverb_list:
#     base_name = os.path.basename(cur_reverb_file)
#     cur_direct_file = os.path.join(direct_path, base_name)
# 
#     [fs, sig_r] = wavfile.read(cur_reverb_file)
#     [fs, sig_d] = wavfile.read(cur_direct_file)
#     cur_sdr = calcSDR(sig_d, sig_r)
#     print("{} / {} sdr = {:.3f}dB".format(i, file_num, cur_sdr))
#     sdrs[i] = cur_sdr
#     i += 1
# 
# print("\naverage sdr = {:.3f}dB".format(np.mean(sdrs)))

def evaluate(result_path='./log_2s_4GPUs_k40s10d20/test/',
             flag_sdr=True, flag_sisdr=True, flag_pesq=True,
             flag_stoi=True, flag_llr=True, flag_srmr=True):
    dir_list = os.listdir(result_path)
    file_num = len(dir_list)
    if flag_sdr:
        sdrs = np.zeros(file_num)
        sdrs_r = np.zeros(file_num)
    if flag_sisdr:
        sisdrs = np.zeros(file_num)
        sisdrs_r = np.zeros(file_num)
    if flag_pesq:
        pesqs = np.zeros(file_num)
        pesqs_r = np.zeros(file_num)
    if flag_pesq:
        stois = np.zeros(file_num)
        stois_r = np.zeros(file_num)
    if flag_llr:
        llrs = np.zeros(file_num)
        llrs_r = np.zeros(file_num)
        count_skip_llr = 0
    if flag_srmr:
        srmrs = np.zeros(file_num)
        srmrs_r = np.zeros(file_num)

    result_file = os.path.join(result_path, '../result_srmr.txt')
    f = open(result_file, 'w')
    i = 0
    for cur_dir in dir_list:
        cur_res_file = os.path.join(result_path, cur_dir, 's1.wav')
        cur_cln_file = os.path.join(result_path, cur_dir, 'true_s1.wav')
        cur_rvb_file = os.path.join(result_path, cur_dir, 'reverb_s1.wav')
        [fs, sig_res] = wavfile.read(cur_res_file)
        [fs, sig_cln] = wavfile.read(cur_cln_file)
        [fs, sig_rvb] = wavfile.read(cur_rvb_file)
        if len(sig_rvb) != len(sig_cln):
            sig_rvb = sig_rvb[:len(sig_cln)]

        cur_str = "{} / {}: ".format(i+1, file_num)

        if flag_sdr:
            cur_sdr = calcSDR(sig_cln, sig_res)
            cur_sdr_r = calcSDR(sig_cln, sig_rvb)
            cur_str += "sdr: {:.3f}dB --> {:.3f}dB ".format(cur_sdr_r, cur_sdr)
            sdrs[i] = cur_sdr
            sdrs_r[i] = cur_sdr_r
        if flag_sisdr:
            cur_sisdr = calcSISDR(sig_cln, sig_res)
            cur_sisdr_r = calcSISDR(sig_cln, sig_rvb)
            cur_str += "sisdr: {:.3f}dB --> {:.3f}dB ".format(cur_sisdr_r, cur_sisdr)
            sisdrs[i] = cur_sisdr
            sisdrs_r[i] = cur_sisdr_r
        if flag_pesq:
            cur_pesq = calcPesq(sig_cln, sig_res, fs)
            cur_pesq_r = calcPesq(sig_cln, sig_rvb, fs)
            cur_str += "pesq: {:.3f} --> {:.3f} ".format(cur_pesq_r, cur_pesq)
            pesqs[i] = cur_pesq
            pesqs_r[i] = cur_pesq_r
        if flag_stoi:
            cur_stoi = calcStoi(sig_cln, sig_res, fs)
            cur_stoi_r = calcStoi(sig_cln, sig_rvb, fs)
            cur_str += "stoi: {:.3f} --> {:.3f} ".format(cur_stoi_r, cur_stoi)
            stois[i] = cur_stoi
            stois_r[i] = cur_stoi_r
        if flag_llr:
            cur_llr = calcLLR(sig_cln, sig_res, fs)
            cur_llr_r = calcLLR(sig_cln, sig_rvb, fs)
            cur_str += "llr: {:.3f} --> {:.3f} ".format(cur_llr_r, cur_llr)
            if cur_llr == -1 or cur_llr_r == -1:
                count_skip_llr += 1
            llrs[i] = cur_llr
            llrs_r[i] = cur_llr_r
        if flag_srmr:
            cur_srmr, _ = calcSRMR(sig_res, fs, fast=True, norm=True)
            cur_srmr_r, _ = calcSRMR(sig_rvb, fs, fast=True, norm=True)
            cur_str += "srmr: {:.3f} --> {:.3f} ".format(cur_srmr_r, cur_srmr)
            srmrs[i] = cur_srmr
            srmrs_r[i] = cur_srmr_r
                  
        print(cur_str)
        f.write(cur_str + '\n')
        i += 1
    cur_str = '\nsummary: '
    if flag_sdr:
        cur_str += 'sdr = {:.3f}dB --> {:.3f}dB, '.format(np.mean(sdrs_r),
                                                          np.mean(sdrs))
    if flag_sisdr:
        cur_str += 'sisdr = {:.3f}dB --> {:.3f}dB, '.format(np.mean(sisdrs_r),
                                                           np.mean(sisdrs))
    if flag_pesq:
        cur_str += 'pesq = {:.3f} --> {:.3f}, '.format(np.mean(pesqs_r),
                                                      np.mean(pesqs))
    if flag_stoi:
        cur_str += 'stoi = {:.3f} --> {:.3f} '.format(np.mean(stois_r),
                                                    np.mean(stois))
    if flag_llr:
        cur_str += 'llr = {:.3f} --> {:.3f} '.format(np.mean(llrs_r),
                                                    np.mean(llrs))
        print("llr_skip num =", count_skip_llr)
    if flag_srmr:
        cur_str += 'srmr = {:.3f} --> {:.3f} '.format(np.mean(srmrs_r),
                                                      np.mean(srmrs))
    print(cur_str)
    f.write(cur_str + '\n')
    f.close()


def calcSISDR(sig_cln, sig_proc):
    def _norm(x):
        return np.sum(x ** 2)
    sig_proc = np.asarray(sig_proc, dtype = np.float32)
    sig_cln = np.asarray(sig_cln, dtype = np.float32)
    if np.max(sig_cln) > 2:
        sig_cln /= (2**15)
        sig_proc /= (2**15)
    sig_cln = sig_cln - np.mean(sig_cln)
    sig_proc = sig_proc - np.mean(sig_proc)
    sig_tar = np.sum(sig_cln * sig_proc) * sig_cln / _norm(sig_cln)
    upp = _norm(sig_tar)
    low = _norm(sig_proc - sig_tar)
    return 10 * np.log10(upp / low) 


def test_SDR(result_path='./log_4s_4GPUs/test/', flag='reverb', if_si=False):
    dir_list = os.listdir(result_path)
    file_num = len(dir_list)
    sdrs = np.zeros(file_num)
    i = 0
    for cur_dir in range(file_num):
        cur_path = os.path.join(result_path, str(cur_dir))
        cur_s1_true = os.path.join(cur_path, 'true_s1.wav')
        if flag == 'reverb':
            cur_rvb_file = os.path.join(cur_path, 'reverb_s1.wav')
        else:
            cur_rvb_file = os.path.join(cur_path, 's1.wav')
        [fs, sig_cln] = wavfile.read(cur_s1_true)
        [fs, sig_rvb] = wavfile.read(cur_rvb_file)
        if if_si:
            cur_sdr = calcSISDR(sig_cln, sig_rvb)
        else:
            cur_sdr = calcSDR(sig_cln, sig_rvb)
        print('{} / {}: sdr = {:.3f}dB'.format(i, file_num, cur_sdr))
        sdrs[i] = cur_sdr
        i += 1
    print('avg sdr = {:.3f}dB'.format(np.mean(sdrs)))
    return sdrs


def analyze(result_path='./log_4s_4GPUs/test/',
            index_file='/share/datasets/wsj0_reverb_v2/infer.csv'):
    # get sdrs and si-sdrs
    sdr_org = test_SDR(result_path=result_path, flag='reverb', if_si=False)
    sdr_proc = test_SDR(result_path=result_path, flag='proced', if_si=False)
    sisdr_org = test_SDR(result_path=result_path, flag='reverb', if_si=True)
    sisdr_proc = test_SDR(result_path=result_path, flag='proced', if_si=True)

    result_file = os.path.join(result_path, '../analyse.txt')
    res_f = open(result_file, 'w')
    
    file_list = os.listdir('/share/datasets/wsj0_reverb_v2/infer/direct/')
    file_num = len(file_list)
    
    sdr_org_dict = {}
    sdr_proc_dict = {}
    sisdr_org_dict = {}
    sisdr_proc_dict = {}
    keys = []
    rt60_dict = {} 

    import csv
    i = 0
    with open(index_file) as f:
        f_csv = csv.reader(f)
        headers = next(f_csv)
        for row in f_csv:
            cond_idx = row[-1]
            if cond_idx not in keys:
                sdr_org_dict[cond_idx] = []
                sdr_proc_dict[cond_idx] = []
                sisdr_org_dict[cond_idx] = []
                sisdr_proc_dict[cond_idx] = []
                keys.append(cond_idx)
                rt60_dict[cond_idx] = row[4]
            file_name = row[0]
            basename = os.path.basename(file_name)
            file_idx = file_list.index(basename)
            sdr_org_dict[cond_idx].append(sdr_org[file_idx])
            sdr_proc_dict[cond_idx].append(sdr_proc[file_idx])
            sisdr_org_dict[cond_idx].append(sisdr_org[file_idx])
            sisdr_proc_dict[cond_idx].append(sisdr_proc[file_idx])
            i += 1
            # print("{} / {}".format(i, file_num))

    for key in sorted(keys):
        cur_str = 'cond {} (rt60 = {}): '.format(key, rt60_dict[key]) +\
                  'sdr {:.3f}dB --> {:.3f}dB, '.format(
                      np.mean(np.asarray(sdr_org_dict[key])),
                      np.mean(np.asarray(sdr_proc_dict[key]))) +\
                  'sisdr {:.3f}dB --> {:.3f}dB, '.format(
                      np.mean(np.asarray(sisdr_org_dict[key])),
                      np.mean(np.asarray(sisdr_proc_dict[key]))) +\
                  'sisdr_improve {:.3f}dB'.format(
                      np.mean(np.asarray(sisdr_proc_dict[key])) -
                      np.mean(np.asarray(sisdr_org_dict[key])))

        print(cur_str)
        res_f.write(cur_str + '\n')

    cur_str = 'sisdr {:.3f}dB --> {:.3f}dB, improve {:.3f}dB'.format(
        np.mean(sisdr_org), np.mean(sisdr_proc),
        np.mean(sisdr_proc) - np.mean(sisdr_org))
    print(cur_str)
    res_f.write(cur_str + '\n')

    res_f.close()




if __name__ == '__main__':
    evaluate(flag_sdr=False, flag_sisdr=False, flag_pesq=False,
             flag_stoi=False, flag_llr=False, flag_srmr=True)
    # test_SDR()
    # analyze()
