from tabnanny import verbose
import numpy as np
import pandas as pd
from pathlib import Path

from tqdm import tqdm

import soundfile as sf

import alphaMusic.utils.fls_utils as fls
import alphaMusic.utils.mat_utils as mat
import alphaMusic.utils.geo_utils as geo
import alphaMusic.utils.acu_utils as acu

from pyroomacoustics.datasets.locata import LOCATA

import matplotlib.pyplot as plt

class LocataDataset:
    def __init__(self, data_dir, verbose, tasks, arrays):
        self.db = LOCATA(basedir=data_dir, verbose=verbose, tasks=tasks, arrays=arrays)
        size = len(self.db.samples)
        
    def iterdata(self):
        dataset = []
        for sample in self.db.samples:
            mix = sample.data
            data = []
            for ts in tqdm(range(len(sample.ts))):
                array = sample.get_array(ts)
                doa = sample.get_doa(ts)
                data.append((ts, array, doa))
            dataset.append((mix, data))
        return dataset

class LibriSpeechDataset:

    def _explore_corpus(self, data_dir):

        cached_csv_path = self.results_dir / Path('librispeech_db.csv')

        if not cached_csv_path.exists():
            # navigate the LibriSpeech dataset and create a df with all the wave info
            data_dir = Path(data_dir)
            print(data_dir.exists())
            
            file_list = fls.walk(data_dir)
            df = pd.DataFrame()
            # recursively traverse all files from current directory
            c = 0
            print('Navigating the folders...', end='')
            for file in file_list: 
                if not file.name.split('.')[-1] == 'flac':
                    continue
                
                wave, fs = sf.read(file)
                assert fs == self.fs

                duration = len(wave) / fs
                name = file.name
                
                df.at[c,'name'] = name
                df.at[c,'duration'] = duration
                df.at[c, 'fs'] = fs
                df.at[c, 'path'] = file
                c += 1
            print('done.')
            df.to_csv(cached_csv_path)
        else:
            df = pd.read_csv(cached_csv_path)

        return df


    def __init__(self, name, data_dir, fs, tlim, results_dir):
        
        self.name = name
        self.fs = fs
        self.tlim = tlim
        
        self.data_dir = Path(data_dir)
        self.results_dir = Path(results_dir)
        
        self.azimuths = [0, 30, -35, 55, -50]
        self.fs = fs

    def built_dataset(self, n_samples,
                            room_dim,
                            RT60s, SNRs, DRRs,
                            array_setup, array_center, 
                            n_srcs,
                            path_to_noise,
                            path_to_pkl=None,
                            do_plot=False):

        self.df = self._explore_corpus(self.data_dir)

        # select utterances withing the time limits
        df = self.df.loc[
                (self.df['duration'] > self.tlim[0]) 
            &   (self.df['duration'] < self.tlim[1])].reset_index(drop=True)

        # random RT60 or SNR or DRR
        RT60s = np.random.choice(RT60s, n_samples, replace=True)
        SNRs = np.random.choice(SNRs, n_samples, replace=True)
        DRRs = np.random.choice(DRRs, n_samples, replace=True)

        # random DoA (az)
        DOAs = np.arange(0, 180)[:n_samples]

        # random sources index
        srcs_idx = np.random.choice(np.arange(0,len(df)), [n_samples, n_srcs], replace=True)

        dataset = []
        for n in tqdm(range(n_samples)):

            # get the DOA for the 3 or 5 sources.
            # distance according to the DRR
            curr_signals = []
            curr_DOAs = []
            for j in range(n_srcs):
                
                idx = srcs_idx[n,j]

                path_to_signal = df.at[idx, 'path']
                signal, fs = sf.read(Path(path_to_signal))

                # first source
                if j == 0: 
                    doa = DOAs[n]

                else:
                    
                    doa_axis = np.arange(0,180)

                    possible_doas_for_j2 = []
                    for j2 in range(j):
                        doa_j2 = curr_DOAs[j2].azimuth
                        doa_j2_axis = np.arange(doa_j2-15, doa_j2+15)
                        possible_doas_for_j2.append(doa_j2_axis)
                    
                    possible_doas_for_j2 = np.concatenate(possible_doas_for_j2)
                    possible_doas_for_j2 = np.setdiff1d(doa_axis, possible_doas_for_j2)
                    
                    doa = np.random.choice(possible_doas_for_j2)

                curr_signals.append(signal)
                curr_DOAs.append(geo.DOASetup(
                                    distance=DRRs[n],
                                    azimuth=doa,
                                    elevation=0,
                                    deg=True))
                assert fs == self.fs
            
            scene = acu.AcousticScene(
                room_dim, RT60s[n], SNRs[n], 
                array_setup, array_center, 
                curr_DOAs, curr_signals, self.fs, path_to_noise
            )


            mix = scene.simulate()
            if do_plot: scene.plot()
            # return dataset

            doas = [curr_DOAs[j].azimuth for j in range(n_srcs)]

            if len(doas) > 1:
                assert not np.min(doas) < 0
                assert not np.min(np.diff(np.sort(doas))) < 15
                assert not np.max(doas) > 180

            acu_params = {
                'RT60': scene.acoustic_params['RT60'],
                'SNR':  scene.acoustic_params['SNR'],
                'DRR':  scene.acoustic_params['DRR'],
                'DER':  scene.acoustic_params['DER'],
                'DST':  DRRs[n],
            }
            dataset.append((mix, doas, acu_params))
        print('done')

        print('Saving', end='')
        if path_to_pkl is not None:
            fls.save_to_pkl(path_to_pkl, dataset)

        return dataset

    def load_dataset(self, path_to_pkl):
        return fls.load_from_pkl(path_to_pkl)