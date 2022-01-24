#! /usr/bin/env python3
# coding: utf-8

import numpy as np
import random
import json
import sys, os
import librosa
import soundfile as sf
import time
import pickle as pic
import scipy as sc
import uuid

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
    parser.add_argument('--seed', dest='seed', type=int, default=666,  help='initial seed')

    parser.add_argument('--n_srcs'       ,   dest='n_srcs', type=int, default=3,  help='Number of requested target sources')
    parser.add_argument('--n_mics',  dest='n_mics', type=int, default=    10, help='number of microphones (max)')
    parser.add_argument('--pos_src',  dest='src_setting', type=str, default='circ',  help='source position setting')
    parser.add_argument('--dist_src',  dest='dist_src', type=float, default=1.5,  help='radius source distance')
    parser.add_argument('--RT60'       ,   dest='RT60', type=float, default=0.256,  help='Reverberation time in s')
    parser.add_argument('--snr'       ,   dest='snr', type=float, default=30,  help='Input SNR')
    parser.add_argument('--noise_type',   dest='noise_type', type=str, default='office',  help='cafet, office or living noise')

    parser.add_argument('--model',  dest='model', type=str, default='far',  help='acoustic model')

    args = parser.parse_args()

    if args.gpu < 0:
        import numpy as xp
    else:
        import cupy as xp
        print("Use GPU " + str(args.gpu))
        xp.cuda.Device(args.gpu).use()

    xp.random.seed(args.seed)
    SPEECH_DIR = '/media/mafontai/SSD_2/data/speech_separation/wsj0/wsj0/sd_dt_05/001/'
    DIR_SAVE = "/media/mafontai/SSD_2/data/localization/RT_60={}/circ_r={}/{}_src/{}_noise/SNR={}/".format(args.RT60, args.dist_src, args.n_srcs, args.noise_type, args.snr)
    NOISE_FILE = "/media/mafontai/SSD_2/data/noise/{}_noise.wav".format(args.noise_type)

    if args.gpu < 0:
        import numpy as xp
    else:
        import cupy as xp
        print("Use GPU " + str(args.gpu))
        xp.cuda.Device(args.gpu).use()
    # DATA_DIR = 'data/'
    FILE_PATH = glob.glob(os.path.join(SPEECH_DIR, "*.wav"))[:args.n_srcs]
    random.seed(args.seed)
    random.shuffle(FILE_PATH)

    # detect the longest speech
    max_file = np.argmax([os.path.getsize(FILE_PATH[n]) for n in range(args.n_srcs)])
    tmp_wav, fs = sf.read(FILE_PATH[max_file])
    max_length = len(tmp_wav)
    speech_dry_NT = np.zeros([args.n_srcs, max_length], dtype=xp.float32)

    Name_file = ''

    org_angle = np.random.uniform(low=0., high=2. * np.pi, size=1)[0]
    angle_pos = [(org_angle + i * np.deg2rad(50)) % (2. * np.pi) for i in range(args.n_srcs)]

    for id_fp, file_path in enumerate(FILE_PATH):
        name_file = file_path[-12:-4]
        if id_fp == 0:
            Name_file += name_file
        else:
            Name_file += '_{}'.format(name_file)
        print('source {}: {}'.format(id_fp + 1, name_file))
        tmp_wav, fs = sf.read(file_path)
        # tmp_wav /= np.std(tmp_wav)
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
    pos_N2 = np.zeros((args.n_srcs)).astype(xp.float32)
    room.add_microphone_array(R_3M)
    if args.model == 'far':
        x_source_pos = mic_center[0, 0] + args.dist_src * np.cos(angle_pos)
        y_source_pos = mic_center[1, 0] + args.dist_src * np.sin(angle_pos)
        z_source_pos = np.ones((args.n_srcs, 1)) * mic_center[2, 0]
        source_pos = np.concatenate((x_source_pos[:, None],
                                     y_source_pos[:, None],
                                     z_source_pos), axis=1)
    elif args.model == 'near':
        source_pos = np.array([[2.3, 3.3, 1.8], [4, 3.3, 1.8], [1.6, 1.2, 1.8]])
    for n in range(args.n_srcs):
        room.add_source(source_pos[n], signal=speech_dry_NT[n])
        if args.model == 'far':
            vec1 = np.array([1, 0])
            vec2 = source_pos[n,0:2] - mic_center[0:2, 0]
            vec2 /= np.linalg.norm(vec2, keepdims=True)
            pos_N2[n] = np.arccos(vec1[0]*vec2[0]+vec1[1]*vec2[1])

        elif args.model == 'near':
            source_pos[n] += mic_center[:, 0]
            pos_N2[n] = source_pos[n, :2]
    room.simulate()

    obs_MT = np.zeros((args.n_mics, len(room.mic_array.signals[0, :]))).astype(xp.float32)
    tmp_noise_T, fs = sf.read(NOISE_FILE)
    noise_T = np.zeros(len(room.mic_array.signals[0, :]), dtype=xp.float32)
    min_length = min(len(room.mic_array.signals[0, :]), len(tmp_noise_T))
    noise_T[:min_length] = tmp_noise_T[:min_length, 0]
    snr = 10.0 ** ((args.snr - 12.0) / 10.0)
    for m in range(args.n_mics):
        obs_MT[m] = room.mic_array.signals[m, :]
        power = obs_MT[m].var()
        obs_MT[m] += noise_T * np.sqrt(power/snr)
    id_file = "{}_M={}_ID={}".format(Name_file, args.n_mics, uuid.uuid4().hex)
    if not os.path.exists(DIR_SAVE):
        os.makedirs(DIR_SAVE)

    sf.write(DIR_SAVE + "{}.wav".format(id_file), obs_MT.T, fs)
    dict_info = {
    'mic_center': mic_center[:, 0].tolist(),
    'mic_pos': R_3M.tolist(),
    'source_pos': source_pos.tolist(),
    'angle_pos': pos_N2,
    'org_angle': org_angle,
    'angle_pos': angle_pos
    }
    json.dump(dict_info, open(os.path.join(DIR_SAVE, "{}.json".format(id_file)), 'w' ))
    print("{} saved in {} ! ".format(id_file, DIR_SAVE))
