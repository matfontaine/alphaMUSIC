import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyroomacoustics as pra
import time 

from tqdm import tqdm
from pathlib import Path
from scipy.signal import stft

import alphaMusic.utils.fls_utils as fls
import alphaMusic.utils.acu_utils as acu
import alphaMusic.evaluation as evl
from alphaMusic.dataloaders import LibriSpeechDataset

from pyroomacoustics import doa
from alphaMusic.alphaMUSIC import aMUSIC

Fs = 16000
speech_duration_range = [6,7]

nfft = 1024
minF = 500
maxF = 4000
nframes = int((3*16000)/(1024*0.5))

room_dim = [10, 12, 4]
# extra_name = 'auditorium_'
extra_name = ''

base_dir = os.getcwd()

def main(n_samples,RT60_range, SNR_range, DRR_range, n_srcs, n_mics, noise_type):

    # IMPORT/GENERATE DATASET
    path_to_librispeech = Path(base_dir,'data','LibriSpeech','dev-clean')
    path_to_recipe = Path(base_dir,'recipes','EUSIPCO22')

    assert path_to_librispeech.exists()
    assert path_to_recipe.exists()

    db = LibriSpeechDataset('Librispeech', path_to_librispeech, Fs, speech_duration_range, path_to_recipe)

    spacing = 0.08
    array_setup = acu.get_linear_array(n_mics, spacing)
    array_center = np.c_[[3.2, 2.1, 1.2]]

    path_to_noise = Path('.','data',f'{noise_type}_noise.wav')

    def range2str(rng):
        if len(rng) > 1:
            return f'sweep'
        else:
            return str(rng[0])

    rt60_name = range2str(RT60_range)
    snr_name = range2str(SNR_range)
    drr_name = range2str(DRR_range)
    exp_name = f'{extra_name}N:{n_samples}_RT60:{rt60_name}_SNR:{snr_name}_DRR:{drr_name}_nsrcs:{n_srcs}_nmics:{n_mics}_noise:{noise_type}'
    print(exp_name)

    path_to_output_pkl = path_to_recipe / Path(f'data/{exp_name}_data.pkl')

    if path_to_output_pkl.exists():
        print('Dataset already created.')
        dataset = db.load_dataset(path_to_output_pkl)
    
    else:
        print('Creating dataset...')
        dataset = db.built_dataset(
            n_samples,
            room_dim, RT60_range, SNR_range, DRR_range, 
            array_setup, array_center, 
            n_srcs, path_to_noise, do_plot=False, 
            path_to_pkl=path_to_output_pkl)


    # INITIALIZE ALGOs
    kwargs = {'L': array_setup.mic_pos,
            'fs': Fs, 
            'nfft': nfft,
            'azimuth': np.deg2rad(np.arange(180,step=1)),
            'num_src':n_srcs,
            
    }
    algorithms = [
        ('MUSIC', doa.music.MUSIC(**kwargs)),
        ('aMUSIC', aMUSIC(**kwargs,alpha=3, p=1., frequency_normalization=False)),
        ('aMUSIC', aMUSIC(**kwargs,alpha=1.5, p=1., frequency_normalization=False)),
        ('aMUSIC', aMUSIC(**kwargs,alpha=1.8, p=1., frequency_normalization=False)),
        ('aMUSIC', aMUSIC(**kwargs,alpha=3, p=1.5, frequency_normalization=False)),
        ('aMUSIC', aMUSIC(**kwargs,alpha=1.5, p=1.5, frequency_normalization=False)),
        ('aMUSIC', aMUSIC(**kwargs,alpha=1.8, p=1.5, frequency_normalization=False)),
        ('NormMUSIC', doa.normmusic.NormMUSIC(**kwargs)),
        ('aNormMUSIC', aMUSIC(**kwargs,alpha=3,p=1., frequency_normalization=True)),
        ('aNormMUSIC', aMUSIC(**kwargs,alpha=1.5,p=1., frequency_normalization=True)),
        ('aNormMUSIC', aMUSIC(**kwargs,alpha=1.8,p=1., frequency_normalization=True)),
        ('aNormMUSIC', aMUSIC(**kwargs,alpha=3,p=1.5, frequency_normalization=True)),
        ('aNormMUSIC', aMUSIC(**kwargs,alpha=1.5,p=1.5, frequency_normalization=True)),
        ('aNormMUSIC', aMUSIC(**kwargs,alpha=1.8,p=1.5, frequency_normalization=True)),
        ('SRP_PHAT', doa.srp.SRP(**kwargs))
    ]

    columns = ["DOAs", 'SNR', 'RT60','DRR', 'algo', "DOAs_est", "time", "alpha", "p"]
    predictions = {n:[] for n in columns}


    for n, (x, doas, acu_params) in enumerate(tqdm(dataset)):

        for start in [1, 3]:

            # STFT
            star = int(start * Fs)
            stft_signals = stft(x[:,start:start+nframes*nfft], fs=Fs, nperseg=nfft, noverlap=0.5)[2]
            
            # OUR MUSIC
            for algo_name, algo in algorithms:
                    
                predictions['DOAs'].append(doas)
                predictions['algo'].append(algo_name)
                
                start_time = time.time()
                algo.locate_sources(stft_signals, num_src=n_srcs, freq_range=[minF, maxF],mpd=10)
                time_elapsed = time.time() - start_time
                
                doa_est = np.rad2deg(algo.azimuth_recon)

                if algo_name in ['aMUSIC', 'aNormMUSIC']:
                    predictions['alpha'].append(algo.alpha)
                    predictions['p'].append(algo.p)
                else:
                    predictions['alpha'].append(np.NaN)
                    predictions['p'].append(np.NaN)
                predictions['DOAs_est'].append(doa_est)
                predictions['time'].append(time_elapsed)
                predictions['SNR'].append(acu_params['SNR'])
                predictions['RT60'].append(acu_params['RT60'])
                if drr_name == 'sweep':
                    predictions['DRR'].append(acu_params['DST'])
                else:
                    predictions['DRR'].append(acu_params['DRR'])

        df_predictions = pd.DataFrame.from_dict(predictions)
        
    fls.save_to_pkl(path_to_recipe / Path(f'results/{exp_name}_results.pkl'), df_predictions)

if __name__ == "__main__":

    n_samples = 30
    RT60_range = [0.5] # [0.25, 0.5, 0.75, 1., 1.25, 1.5]
    SNR_range = [10] # [-30, -20, -10, 0, 10, 20]
    DRR_range = [1] #[1., 1.5, 2., 2.5, 2.75]
    n_srcs = [1, 2, 3, 4]
    n_mics = [2, 4, 6]
    noise_type = 'cafet'
    
    for J in n_srcs:
        for I in n_mics:
            main(n_samples, RT60_range, SNR_range, DRR_range, J, I, noise_type)