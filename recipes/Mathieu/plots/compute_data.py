import glob as glob
import os as os
import pandas as pd

import numpy as np
import json as json
SNR = [0.0, 5.0, 10.0]
RT60 = [0.256, 0.512, 1.024]
ALPHA = [2.0, 2.8, 0.0]
M = [5, 8, 10]
N = [2, 3]
NOISE_TYPE = ['office', 'cafet', 'living']

datas = pd.DataFrame(columns=['alpha', 'RT60', 'SNR', 'M', 'N', 'noise_type', 'angle', 'time'])


for alpha in ALPHA:
    print("alpha={}".format(alpha))
    for rt60 in RT60:
        SAVE_PATH = "/home/mafontai/Documents/project/git_project/audio_localization/alphaMUSIC/results/RT60={}/alpha={}/".format(rt60, alpha)
        for snr in SNR:
            for m in M:
                for n in N:
                    for noise_type in NOISE_TYPE:
                        file_type = "loc_res-model=far_N={}_M={}_SNR={}dB_noise={}*.json".format(n, m, snr, noise_type)
                        FILE_PATH = glob.glob(os.path.join(SAVE_PATH, file_type))
                        # print ("{} file detected".format(len(FILE_PATH)))
                        for file_path in FILE_PATH:
                            with open(file_path) as json_file:
                                data = json.load(json_file)
                                elapsed_time = data['time']
                                RMS = data['az_RMS']
                                if alpha == 0.0:
                                    alpha_name = r'TylerMUSIC'
                                elif alpha == 2.0:
                                    alpha_name = r'MUSIC'
                                elif alpha == ALPHA[1]:
                                    alpha_name = r'$\alpha$MUSIC'
                                else:
                                    alpha_name = r'$\alpha={}$MUSIC'.format(alpha)
                                data = {
                                'alpha': alpha_name,
                                'RT60': rt60,
                                'SNR': snr,
                                'M': m,
                                'N': n,
                                'noise_type': noise_type,
                                'angle': np.rad2deg(RMS),
                                'time': elapsed_time
                                }
                                datas = datas.append(data,
                                                     ignore_index=True)
datas.to_pickle('./results_loc.pic')
datas.to_csv('./results_loc.csv')

print('Complete !')
