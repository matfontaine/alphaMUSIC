import os
from tqdm import tqdm
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import librosa 
import pyroomacoustics as pra
import soundfile as sf


import alphaMusic.utils.fls_utils as fls
import alphaMusic.utils.acu_utils as acu


from alphaMusic.alphaMUSIC import AlphaMUSIC
from alphaMusic.dataloaders import LibriSpeechDataset
import alphaMusic.evaluation as evl


path_to_librispeech = Path('data','LibriSpeech','dev-clean')
path_to_recipe = Path('recipes','EUSIPCO22')

assert path_to_librispeech.exists()
assert path_to_recipe.exists()


db = LibriSpeechDataset('Librispeech', path_to_librispeech, 16000, [6,8], path_to_recipe)

n_samples = 50

room_dim = [6, 5, 3]
RT60_range = [0]
SNR_range = [100]
DRR_range = [1]

array_setup = acu.linear_array_setup
# array_setup = acu.echo_array_setup
array_center = np.c_[[3.2, 2, 1.2]]

n_srcs = 3
noise_environment = 'cafet'

dataset_name = 'noRT_noSNR_3src'

path_to_output_pkl = path_to_recipe / Path(f'data/{dataset_name}_data.pkl')

# dataset = db.built_dataset(
#     n_samples,
#     room_dim, RT60_range, SNR_range, DRR_range, 
#     array_setup, array_center, 
#     n_srcs, noise_environment, do_plot=False, 
#     path_to_pkl=path_to_output_pkl)
dataset = db.load_dataset(path_to_output_pkl)

algos = ['myMUSIC', 'aMUSIC', 'MUSIC', 'NormMUSIC', 'SRP']

results = []

for n, (x, doas, acu_params) in enumerate(tqdm(dataset)):

    n_mics = x.shape[0]

    # to STFT
    for m in range(n_mics):
        t = len(x[m, :])
        tmp = librosa.core.stft(np.asfortranarray(x[m, :]), n_fft=1024, hop_length=int(1024/4))
        if m == 0:
            obs_FTM = np.zeros([tmp.shape[0], tmp.shape[1], n_mics], dtype=complex)
        obs_FTM[..., m] = tmp
        
    az_res = 1
    az = np.deg2rad(np.linspace(0, 180., 180 * az_res, endpoint=False))
    
    seed = 666
    minF = 1000
    maxF = 4000
    
    nfft = 1024
    fs = 16000
    ac_model = 'far'
    mic_pos = array_setup.mic_pos
    
    
    hparams = {
        'seed' : seed,
        'minF' : minF,
        'maxF' : maxF,
        'nfft' : nfft,
        'az_res' : az_res,
    }
    
    # OUR MUSIC
    for algo in algos:
        doa = None
        
        if algo == 'myMUSIC':
            doa = AlphaMUSIC(mic_pos, fs, nfft, c=343, num_src=n_srcs,
                             azimuth=az, mode='far', alpha=1, P_prime=50, 
                             xp=np)
            doa.locate_sources(obs_FTM, freq_range=[minF, maxF])
            est_doas = np.rad2deg(doa.azimuth_recon)
    
        elif algo == 'aMUSIC':
            doa = AlphaMUSIC(mic_pos, fs, nfft, c=343, num_src=n_srcs,
                     azimuth=az, mode='far', alpha=2, P_prime=50, 
                     xp=np)
            doa.locate_sources(obs_FTM, freq_range=[minF, maxF])
    
        else:
            import pyroomacoustics as pra
            doa = pra.doa.algorithms[algo](mic_pos, fs, nfft, c=343, mode='far', azimuth=az, num_src=n_srcs)
            doa.locate_sources(obs_FTM.transpose([2,0,1]), freq_range=[minF, maxF])
            
        est_doas = np.rad2deg(doa.azimuth_recon)
        
        print(algo, n_srcs, doa.num_src, len(doa.azimuth_recon))

        res_dict = {
            'id' : n,
            'algo' : algo,
            'nsrcs' : n_srcs,
            'doas_est' : est_doas.tolist(),
            'doas_ref' : doas,
            'acu_params' : acu_params,
            'hparams' : hparams,
        }
        results.append(res_dict)