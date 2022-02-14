import numpy as np
import pandas as pd
from pathlib import Path

from tqdm import tqdm

import soundfile as sf

import alphaMusic.utils.fls_utils as fls
import alphaMusic.utils.mat_utils as mat
import alphaMusic.utils.geo_utils as geo
import alphaMusic.utils.acu_utils as acu

import matplotlib.pyplot as plt

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
        
        self.azimuths = [0, 30, -30, 50, -50]
        self.fs = fs

    def built_dataset(self, n_samples,
                            room_dim,
                            RT60_range, SNR_range, DRR_range,
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
        RT60s = mat.uniform_in_range(RT60_range, n_samples)
        SNRs = mat.uniform_in_range(SNR_range, n_samples)
        DRRs = mat.uniform_in_range(DRR_range, n_samples)
        
        # random DoA (az)
        DOAs = mat.uniform_in_range([31,180-31], n_samples)

        # random sources index
        srcs_idx = np.random.default_rng().choice(np.arange(0,len(df)), [n_samples, n_srcs], replace=False)

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
                
                curr_signals.append(signal)
                curr_DOAs.append(geo.DOASetup(
                                    distance=DRRs[n],
                                    azimuth=np.mod(self.azimuths[j] + DOAs[n], 360),
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

            doas = [curr_DOAs[j].azimuth for j in range(n_srcs)]

            acu_params = {
                'RT60': scene.acoustic_params['RT60'],
                'SNR':  scene.acoustic_params['SNR'],
                'DRR':  scene.acoustic_params['DRR'],
                'DER':  scene.acoustic_params['DER'],
            }
            dataset.append((mix, doas, acu_params))
        print('done')

        print('Saving', end='')
        if path_to_pkl is not None:
            fls.save_to_pkl(path_to_pkl, dataset)

        return dataset

    def load_dataset(self, path_to_pkl):
        return fls.load_from_pkl(path_to_pkl)