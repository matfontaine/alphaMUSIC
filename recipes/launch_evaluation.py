#! /usr/bin/env python3
# coding: utf-8

import numpy as np
import random
import sys, os
import json
import librosa
import soundfile as sf
import time
import pickle as pic
import scipy as sc

import pyroomacoustics as pra
from alphaMusic.alphaMUSIC import AlphaMUSIC

if __name__ == "__main__":
    import argparse
    import librosa
    import soundfile as sf

    import glob as glob
    import os as os
    parser = argparse.ArgumentParser()
    parser.add_argument( '--gpu', type= int, default=   -1, help='GPU ID, no GPU = -1')
    parser.add_argument( '--n_fft', type= int, default= 1024, help='number of frequencies')
    parser.add_argument('--alpha', dest='alpha', type=float, default=1.2,  help='Gaussian case (alpha=2)')
    # parser.add_argument('--minF', dest='minF', type=float, default=2000,  help='minimum frequency to consider (in Hz)')
    # parser.add_argument('--maxF', dest='maxF', type=float, default=4000,  help='max frequency to consider (in Hz)')
    parser.add_argument('--minF', dest='minF', type=float, default=2000,  help='minimum frequency to consider (in Hz)')
    parser.add_argument('--maxF', dest='maxF', type=float, default=8000,  help='max frequency to consider (in Hz)')
    parser.add_argument('--seed', dest='seed', type=int, default=666,  help='initial seed')

    parser.add_argument('--n_srcs'       ,   dest='n_srcs', type=int, default=3,  help='Number of requested target sources')
    parser.add_argument('--pos_src',  dest='src_setting', type=str, default='circ',  help='source position setting')
    parser.add_argument('--dist_src',  dest='dist_src', type=float, default=1.5,  help='radius source distance')
    parser.add_argument('--RT60'       ,   dest='RT60', type=float, default=0.256,  help='Reverberation time in s')
    parser.add_argument('--snr'       ,   dest='snr', type=float, default=0.0,  help='Input SNR')
    parser.add_argument('--noise_type',   dest='noise_type', type=str, default='office',  help='cafet, office or living noise')

    parser.add_argument("--x_acc", type=float, default=0.5, help="azimuth/x angle accuracy")
    parser.add_argument("--y_acc", type=float, default=20, help="elevation/y angle accuracy")

    parser.add_argument('--n_mics',  dest='n_mics', type=int, default=    5, help='number of microphones')
    parser.add_argument('--array',  dest='array', type=str, default='hololens',  help='array type')
    parser.add_argument('--model',  dest='model', type=str, default='far',  help='acoustic model')


    args = parser.parse_args()

    if args.gpu < 0:
        import numpy as xp
    else:
        import cupy as xp
        print("Use GPU " + str(args.gpu))
        xp.cuda.Device(args.gpu).use()

    xp.random.seed(args.seed)
    DATA_DIR = "/media/mafontai/SSD_2/data/localization/RT_60={}/circ_r={}/{}_src/{}_noise/SNR={}/".format(args.RT60, args.dist_src, args.n_srcs, args.noise_type, args.snr)
    BASE_DIR_SAVE = "/home/mafontai/Documents/project/git_project/audio_localization/alphaMUSIC/results/"
    SAVE_PATH = os.path.join(BASE_DIR_SAVE, 'RT60={}/alpha={}/'.format(args.RT60, args.alpha))
    if args.gpu < 0:
        import numpy as xp
    else:
        import cupy as xp
        print("Use GPU " + str(args.gpu))
        xp.cuda.Device(args.gpu).use()
    # DATA_DIR = 'data/'
    FILE_PATH = glob.glob(os.path.join(DATA_DIR, "*.wav"))
    print ("alpha={}, RT_60={}s, N={}, M={}, noise_type={}, SNR={}dB".format(args.alpha,
                                                                             args.RT60,
                                                                             args.n_srcs,
                                                                             args.n_mics,
                                                                             args.noise_type,
                                                                             args.snr))
    for id_fp, file_path in enumerate(FILE_PATH):
        print("{}/{} localization done !".format(id_fp + 1, len(FILE_PATH)))
        file_id = file_path.split('/')[-1][:-4]
        file_split = file_path.split('/')[-1].split('_')
        ID = file_split[-1].split('.wav')[0][3:]
        prefix_id = "model={}_N={}_M={}_SNR={}dB_noise={}_ID={}".format(args.model,
                                                                        args.n_srcs,
                                                                        args.n_mics,
                                                                        args.snr,
                                                                        args.noise_type,
                                                                        ID)
        if (os.path.isfile(os.path.join(SAVE_PATH, "loc_res-" + prefix_id + ".json"))):
            print("File {} already treated for alpha={} and RT_60={}s".format(prefix_id, args.alpha, args.RT60))
        else:
            with open(os.path.join(DATA_DIR, file_id + ".json")) as json_file:
                data = json.load(json_file)
                mic_center = data['mic_center']
                mic_pos_x, mic_pos_y, mic_pos_z = (np.array(data['mic_pos'][0])[:, None],
                                                   np.array( data['mic_pos'][1])[:, None],
                                                   np.array(data['mic_pos'][2])[:, None])
                mic_pos_3M = np.concatenate((mic_pos_x, mic_pos_y, mic_pos_z), axis=1).T[:, :args.n_mics]


                source_pos_3N = np.array(data['source_pos']).T

                pos_N = np.array(data['angle_pos'])
            # for n in range(args.n_srcs):
            #     if args.model == 'far':
            #         vec1 = np.array([1, 0])
            #         vec2 = source_pos_3N[0:2, n] - mic_center[0:2]
            #         vec2 /= np.linalg.norm(vec2, keepdims=True)
            #         pos_N[n] = np.arccos(vec1[0]*vec2[0]+vec1[1]*vec2[1])
            tmp_wav, fs = sf.read(file_path)
            # import ipdb; ipdb.set_trace()
            for m in range(args.n_mics):
                tmp = librosa.core.stft(np.asfortranarray(tmp_wav[:, m]),
                                        n_fft=args.n_fft,
                                        hop_length=int(args.n_fft/4))
                tmp /= np.abs(tmp).max() * 1.2
                if m == 0:
                    obs_FTM = np.zeros([tmp.shape[0], tmp.shape[1], args.n_mics], dtype=xp.complex)
                obs_FTM[..., m] = tmp
            localizer = AlphaMUSIC(n_source=args.n_srcs,
                         seed=args.seed, alpha=args.alpha, ac_model=args.model, x_acc=args.x_acc, y_acc=args.y_acc,
                         mic_pos=mic_pos_3M, source_pos=pos_N, xp=xp)
            localizer.load_spectrogram(obs_FTM, minF=args.minF, maxF=args.maxF,
                                       nfft=args.n_fft, fs=fs)
            localizer.file_id = prefix_id


            if not os.path.exists(SAVE_PATH):
                os.makedirs(SAVE_PATH)
            localizer.localize(save_parameter=True, save_pos=True,
                               save_RMS=True, save_corr=False, SAVE_PATH=SAVE_PATH)
