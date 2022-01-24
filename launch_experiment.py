#! /usr/bin/env python3
# coding: utf-8

import numpy as np
import random
import sys, os
import librosa
import soundfile as sf
import time
import pickle as pic
import scipy as sc

import pyroomacoustics as pra
from alphaMUSIC import AlphaMUSIC
def make_filename_suffix(args):
    filename_suffix = "algo-%s_model-%s_abscoeff-%1.2f" % (
        args.algo, args.model, args.abs_coeff)
    return filename_suffix

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
    parser.add_argument('--minF', dest='minF', type=float, default=2000,  help='minimum frequency to consider (in Hz)')
    parser.add_argument('--maxF', dest='maxF', type=float, default=4000,  help='max frequency to consider (in Hz)')
    parser.add_argument('--seed', dest='seed', type=int, default=666,  help='initial seed')

    parser.add_argument('--n_srcs'       ,   dest='n_srcs', type=int, default=3,  help='Number of requested target sources')
    parser.add_argument('--RT60'       ,   dest='RT60', type=float, default=0.256,  help='Reverberation time in s')
    parser.add_argument('--snr'       ,   dest='snr', type=float, default=30,  help='Input SNR')

    parser.add_argument("--x_acc", type=float, default=5, help="azimuth/x angle accuracy")
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
    DATA_DIR = '/media/mafontai/SSD_2/data/speech_separation/wsj0/wsj0/sd_dt_05/001/'
    BASE_DIR_SAVE = "/home/mafontai/Documents/project/git_project/audio_localization/alphaMUSIC/results/"
    SAVE_PATH = os.path.join(BASE_DIR_SAVE, 'RT60={}/alpha={}/'.format(args.RT60, args.alpha))

    if args.gpu < 0:
        import numpy as xp
    else:
        import cupy as xp
        print("Use GPU " + str(args.gpu))
        xp.cuda.Device(args.gpu).use()
    # DATA_DIR = 'data/'
    FILE_PATH = glob.glob(os.path.join(DATA_DIR, "*.wav"))[:args.n_srcs]
    random.seed(args.seed)
    random.shuffle(FILE_PATH)
    max_file = np.argmax([os.path.getsize(FILE_PATH[n]) for n in range(args.n_srcs)])
    tmp_wav, fs = sf.read(FILE_PATH[max_file])
    max_length = len(tmp_wav)
    speech_dry_NT = np.zeros([args.n_srcs, max_length], dtype=xp.float32)
    obs_MT = np.zeros([args.n_mics, max_length], dtype=xp.float32)
    Name_file = ''
    for id_fp, file_path in enumerate(FILE_PATH):
        name_file = file_path[-12:-4]
        if id_fp == 0:
            Name_file += name_file
        else:
            Name_file += '_{}'.format(name_file)
        print('source {}: {}'.format(id_fp + 1, name_file))
        tmp_wav, fs = sf.read(file_path)
        tmp_wav /= np.std(tmp_wav)
        # tmp = librosa.core.stft(np.asfortranarray(tmp_wav)), n_fft=args.n_fft, hop_length=int(args.n_fft/4))
        speech_dry_NT[id_fp, :len(tmp_wav)] = tmp_wav

    rt60 = args.RT60  # seconds
    room_dim = [6.0, 5.0, 3.0]  # meters
    e_absorption, max_order = pra.inverse_sabine(rt60, room_dim)

    room = pra.ShoeBox(
        room_dim, fs=fs, materials=pra.Material(e_absorption), max_order=max_order
    )
    # spherical array
    mic_center = np.c_[[3.5, 2.5, 1.7]]
    mic_radius = 0.04
    # grid = pra.doa.GridSphere(args.n_mics)
    # R_3M = mic_center + mic_radius * grid.cartesian
    R_3M = pra.circular_2D_array(mic_center[:2, 0], args.n_mics, 0, mic_radius)
    # R_3M = pra.linear_2D_array(mic_center[:2, 0], args.n_mics, 0, mic_radius)

     # + 0.1 * np.random.rand(2, args.n_mics)
    R_3M = np.concatenate((R_3M, np.ones((1, args.n_mics)) * mic_center[2, 0]), axis=0)
    # import ipdb; ipdb.set_trace()
    # pos_N2 = np.zeros((args.n_srcs, 2 )).astype(xp.float32)
    pos_N = np.zeros((args.n_srcs)).astype(xp.float32)
    room.add_microphone_array(R_3M)
    if args.model == 'far':
        source_pos = np.array([[1.2, 3.3, 1.7], [3.4, 4.3, 1.7], [5.2, 3.3, 1.7]])
    elif args.model == 'near':
        source_pos = np.array([[2.3, 3.3, 1.8], [4, 3.3, 1.8], [1.6, 1.2, 1.8]])
    for n in range(args.n_srcs):
        room.add_source(source_pos[n], signal=speech_dry_NT[n])
        if args.model == 'far':
            vec1 = np.array([1, 0])
            vec2 = source_pos[n,0:2] - mic_center[0:2, 0]
            vec2 /= np.linalg.norm(vec2, keepdims=True)
            pos_N[n] = np.arccos(vec1[0]*vec2[0]+vec1[1]*vec2[1])

        elif args.model == 'near':
            source_pos[n] += mic_center[:, 0]
            pos_N2[n] = source_pos[n, :2]
    room.simulate()
    for m in range(args.n_mics):
        t = len(room.mic_array.signals[m, :])
        tmp = librosa.core.stft(np.asfortranarray(room.mic_array.signals[m, :]),
                                n_fft=args.n_fft,
                                hop_length=int(args.n_fft/4))
        if m == 0:
            obs_FTM = np.zeros([tmp.shape[0], tmp.shape[1], args.n_mics], dtype=xp.complex)
        obs_FTM[..., m] = tmp
    localizer = AlphaMUSIC(n_source=args.n_srcs,
                 seed=args.seed, alpha=args.alpha, ac_model=args.model, x_acc=args.x_acc, y_acc=args.y_acc,
                 mic_pos=R_3M, source_pos=pos_N2, P_prime=50, xp=xp)
    localizer.load_spectrogram(obs_FTM, minF=args.minF, maxF=args.maxF,
                               nfft=args.n_fft, fs=fs)
    localizer.file_id = Name_file
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
    localizer.localize(save_parameter=True, save_pos=True,
                       save_RMS=True, save_corr=False, SAVE_PATH=SAVE_PATH)
