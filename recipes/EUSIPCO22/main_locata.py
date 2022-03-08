from importlib.resources import path
from unittest import result
import numpy as np
import pandas as pd
import time

from tqdm import tqdm
from pathlib import Path
from scipy.signal import stft
from datetime import datetime, timedelta
from pyroomacoustics.datasets.locata import LOCATA, _find_ts


from pyroomacoustics.doa import MUSIC, NormMUSIC, SRP
from alphaMusic.alphaMUSIC import aMUSIC

base_dir = Path(__file__).parent.absolute()


def main(task_in, array_in):
    path_to_locata = Path('.','data','Locata','dev')
    print(path_to_locata.exists())

    tasks = [task_in]
    arrays = [array_in] # 'dicit']

    dB = LOCATA(path_to_locata, verbose=True, tasks=tasks, arrays=arrays)

    results = pd.DataFrame()

    # LOCATA PARAMS
    Fs = 48000  
    nfft = 1024

    frame_sec = 0.03
    frame_size = int(Fs * frame_sec)
    block_size = 10

    minF = 500
    maxF = 4000

    for recordings in dB.recordings:

        
        task = recordings.meta.task
        array = recordings.meta.array
        rec = recordings.meta.rec

        path_to_results = base_dir / Path(f'results_locata_task:{task}_array:{array}_rec:{rec}.csv')
        if path_to_results.exists():
            continue

        print(f'Processing: task: {task}\t array: {array} \t rec: #{rec}')
        
        data = recordings.data
        fs = recordings.fs
        timestamps = recordings.ts
        
        name_srcs = list(recordings.sources.keys())
        num_srcs = len(name_srcs)
        
        # Resample VAD
        vads = {}
        for src in name_srcs:
            v = recordings.sources[src]['vad']
            vads[src] = v
            
        # Resample 48kHz -> 16kHz
        # data = resample(data.T, orig_sr=fs, target_sr=Fs)
        data = data.T

        if array == 'dicit':
            sub_array = [2,3,4,5,6,8,9,10,11]
            data = data[sub_array,:]
        
        # STFT
        freqs, times, stft_signals = stft(data, fs=Fs, nfft=nfft, nperseg=frame_size//4, noverlap=0)
        
        M, F, T = stft_signals.shape
        
        # iterate over frames
        starting_timestamp = recordings.get_ts(0)

        row = 0
        for t in tqdm(range(block_size, T-block_size, 5)):
            
            ts = times[t]
            ts = starting_timestamp + timedelta(seconds=ts)
            
            results.at[row, 'timestamp'] = times[t]
            results.at[row, 'task'] = int(task)
            results.at[row, 'array'] = array
            results.at[row, 'rec'] = rec
            
            mic_pos = recordings.get_array(ts)
            
            if array == 'dicit':
                mic_pos = mic_pos[:,sub_array]
                
            doa_dict = recordings.get_doa(ts)

            doas = {}
            
            num_srcs = 0
            doas_true = []
            for s, src in enumerate(doa_dict.keys()):
                    
                azimuth = np.rad2deg(doa_dict[src]['azimuth'])

                doas[src] = azimuth
                doas_true.append(azimuth)
                
                idx = _find_ts(timestamps, ts)
                results.at[row, f'VAD_{s}'] = vads[src][idx]
                results.at[row, f'DOA_{s}'] = azimuth

                num_srcs += vads[src][idx]
            
            num_srcs = int(num_srcs)
            results.at[row, 'J'] = num_srcs

            if num_srcs == 0:
                continue

            if array == 'dicit':
                doa_grid = np.arange(180,step=1)
            elif array == 'dummy':
                doa_grid = np.arange(-180, 180,step=1)
            else:
                doa_grid = np.arange(360,step=1)

            kwargs = {'L': mic_pos,
                    'fs': Fs, 
                    'nfft': nfft,
                    'azimuth': np.deg2rad(doa_grid),
                    'num_src': num_srcs,
            }
            
            algorithms = {
                'MUSIC':         MUSIC(**kwargs),
                'aMUSIC' :      aMUSIC(**kwargs,alpha=3,frequency_normalization=False),
                'NormMUSIC': NormMUSIC(**kwargs),
                'aNormMUSIC' :  aMUSIC(**kwargs,alpha=3,frequency_normalization=True),
                'SRP_PHAT' :       SRP(**kwargs),
            }

            for algo_name, algo in algorithms.items():

                start = time.time()
                algo.locate_sources(stft_signals[:,:,t-block_size:t+block_size],
                                    num_src=num_srcs, 
                                    freq_range=[minF, maxF], 
                                    mpd=5)
                time_elapsed = time.time() - start

                doas_estm = np.rad2deg(algo.azimuth_recon)


                for d, doa in enumerate(doas_estm):
                    results.at[row, f'{algo_name}_{d}'] = doa

                results.at[row, f'{algo_name}_time'] = time_elapsed

            row += 1

        print(results)
        
        results.to_csv(path_to_results)

    return results


if __name__ == '__main__':
    print(base_dir)
    task = 4

    for array in ['benchmark2', 'dicit', 'dummy']:
        res = main(task, array)

    print(res)