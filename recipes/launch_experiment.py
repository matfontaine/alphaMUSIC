#! /usr/bin/env python3
# coding: utf-8

import os
from pathlib import Path 
import time

import random
import pickle as pic
import scipy as sc

import numpy as np
import librosa
import soundfile as sf
import pyroomacoustics as pra

import matplotlib.pyplot as plt

from alphaMusic.alphaMUSIC import AlphaMUSIC
from alphaMusic.utils.geo_utils import cart2sph, sph2cart

def make_filename_suffix(args):
    filename_suffix = "algo-%s_model-%s_abscoeff-%1.2f" % (
        args.algo, args.model, args.abs_coeff)
    return filename_suffix

if __name__ == "__main__":
    import argparse

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

    parser.add_argument('--algo',  dest='algo', type=str, default='music',  help='algorithm')

    parser.add_argument('--data_dir', dest='data_dir', type=str, default='./data/', help='path to the input raw data')
    parser.add_argument('--dataset',  dest="dataset", type=str, default='LibriSpeech', help="the dataset you want to use")
    parser.add_argument('--results_dir', dest='results_dir', type=str, default='./results/', help='path to the output results')

    args = parser.parse_args()


    # set up folders
    data_dir = Path(args.data_dir)
    results_dir = Path(args.results_dir, 'RT60={}/alpha={}/'.format(args.RT60, args.alpha))
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.gpu < 0:
        import numpy as xp
    else:
        import cupy as xp
        print("Use GPU " + str(args.gpu))
        xp.cuda.Device(args.gpu).use()

    # set up random seed
    xp.random.seed(args.seed)
    random.seed(args.seed)

    # DATA_DIR = '/media/mafontai/SSD_2/data/speech_separation/wsj0/wsj0/sd_dt_05/001/'
    # BASE_DIR_SAVE = "/home/mafontai/Documents/project/git_project/audio_localization/alphaMUSIC/results/"
    # SAVE_PATH = os.path.join(BASE_DIR_SAVE, 'RT60={}/alpha={}/'.format(args.RT60, args.alpha))
    
    # GET DATA
    data_dir = data_dir / Path("LibriSpeech", "dev-clean", "84", "121123")
    file_paths = list(data_dir.glob("*.flac"))[:args.n_srcs]

    random.shuffle(file_paths)

    # allocate memory
    id_biggest_file = np.argmax([os.path.getsize(file) for file in file_paths])
    tmp_wav, fs = sf.read(file_paths[id_biggest_file])
    max_length = len(tmp_wav)
    speech_dry_NT = np.zeros([args.n_srcs, max_length], dtype=xp.float32)
    
    # get anechoic speech
    obs_MT = np.zeros([args.n_mics, max_length], dtype=xp.float32)
    Name_file = ''
    for id_fp, file_path in enumerate(file_paths):
        name_file = file_path.name.split('.')[0]
        if id_fp == 0:
            Name_file += name_file
        else:
            Name_file += '_{}'.format(name_file)
        print('source {}: {}'.format(id_fp + 1, name_file))
        tmp_wav, fs = sf.read(file_path)
        tmp_wav /= np.std(tmp_wav)
        speech_dry_NT[id_fp, :len(tmp_wav)] = tmp_wav

    # simulate reverberation
    rt60 = args.RT60  # seconds
    room_dim = [6.0, 5.0, 3.0]  # meters
    e_absorption, max_order = pra.inverse_sabine(rt60, room_dim)

    room = pra.ShoeBox(
        room_dim, fs=fs, materials=pra.Material(e_absorption), max_order=max_order
    )

    # add spherical array
    mic_center = np.c_[[3.2, 1.5, 1.5]]
    mic_radius = 0.04
    R_3M = pra.circular_2D_array(mic_center[:2, 0], args.n_mics, 0, mic_radius)
    # R_3M = pra.linear_2D_array(mic_center[:2, 0], args.n_mics, 0, mic_radius)
    R_3M = np.concatenate((R_3M, np.ones((1, args.n_mics)) * mic_center[2, 0]), axis=0)
    pos_N = np.zeros((args.n_srcs)).astype(xp.float32)
    room.add_microphone_array(R_3M)
    
    # add sources
    source_pos = np.array([[2.3, 3.3, 1.8], [3.4, 4.3, 1.7], [5.2, 3.3, 1.7]])
    # if args.model == 'far':
    # elif args.model == 'near':
    #     source_pos = np.array([[2.3, 3.3, 1.8], [4, 3.3, 1.8], [1.6, 1.2, 1.8]])

    # compute DoA and relative position
    true_positions = {}
    for n in range(args.n_srcs):
        room.add_source(source_pos[n], signal=speech_dry_NT[n])
        
        true_positions['abs'] = source_pos[n,:]
        
        # relative position cartesian
        xyz = source_pos[n,:] - mic_center.squeeze()
        razel = cart2sph(xyz[:,None])
        
        if args.model == 'far':
            vec1 = np.array([1, 0])
            vec2 = source_pos[n,0:2] - mic_center[0:2, 0]
            vec2 /= np.linalg.norm(vec2, keepdims=True)
            pos_N[n] = np.arccos(vec1[0]*vec2[0]+vec1[1]*vec2[1])

        elif args.model == 'near':
            source_pos[n] += mic_center[:, 0]
            pos_N2[n] = source_pos[n, :2]

        print(razel[0], pos_N[n])

    1/0
    print(true_positions)
    1/0

    room.plot()
    plt.savefig(results_dir / Path("room_scene.png"))

    room.simulate()


    # to STFT
    for m in range(args.n_mics):
        t = len(room.mic_array.signals[m, :])
        tmp = librosa.core.stft(np.asfortranarray(room.mic_array.signals[m, :]),
                                n_fft=args.n_fft,
                                hop_length=int(args.n_fft/4))
        if m == 0:
            obs_FTM = np.zeros([tmp.shape[0], tmp.shape[1], args.n_mics], dtype=complex)
        obs_FTM[..., m] = tmp
        
    # localize with...
    spatial_resp = dict()
    if args.algo == 'alphaMUSIC':
        localizer = AlphaMUSIC(n_source=args.n_srcs,
                    seed=args.seed, alpha=args.alpha, ac_model=args.model, x_acc=args.x_acc, y_acc=args.y_acc,
                    mic_pos=R_3M, source_pos=pos_N, P_prime=50, xp=xp)
        localizer.load_spectrogram(obs_FTM, minF=args.minF, maxF=args.maxF,
                                nfft=args.n_fft, fs=fs)
        localizer.file_id = Name_file
        
        localizer.localize(save_parameter=True, save_pos=True,
                        save_RMS=True, save_corr=False, SAVE_PATH=results_dir)

    elif args.algo == "MUSIC":

        doa = pra.doa.algorithms["MUSIC"](R_3M, fs, args.n_fft, c=343, num_src=3)

        # this call here perform localization on the frames in X
        doa.locate_sources(obs_FTM.transpose([2,0,1]), freq_range=[args.minF, args.maxF])

        print(doa.azimuth_recon)
        print(pos_N)


        
        # # store spatial response
        # spatial_resp[algo_name] = doa.grid.values
            
        # # normalize   
        # min_val = spatial_resp[algo_name].min()
        # max_val = spatial_resp[algo_name].max()
        # spatial_resp[algo_name] = (spatial_resp[algo_name] - min_val) / (max_val - min_val)

