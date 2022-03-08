import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.transforms import BlendedGenericTransform

from numpy import median
import math
import numpy as np

# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')
#
# mpl.rcParams['font.family'] = 'serif'
# mpl.rcParams['text.latex.unicode'] = 'True'

# sns.set()
sns.set_style("whitegrid")
cste = 12
params = {
    'backend': 'ps',
    'axes.labelsize': cste,
    'font.size': cste,
    'legend.fontsize': cste,
    'xtick.labelsize': cste,
    'ytick.labelsize': cste,
    'text.usetex': True,
    'font.family': 'serif',
    'font.serif': 'ptmrr8re',
}

sns.set_style("whitegrid", {
    'pgf.texsystem': 'xelatex',  # pdflatex, xelatex, lualatex
    'text.usetex': True,
    'font.family': 'serif',
    'axes.labelsize': cste,
    'legend.labelspacing':0,
    'legend.borderpad':0,
    'font.size': cste,
    'legend.fontsize': cste,
    'xtick.labelsize': cste,
    'ytick.labelsize': cste,
    'font.serif': [],
})
plt.rcParams.update(params)

fig_width_pt = 400.6937  # Get this from LaTeX using \showthe\columnwidth
inches_per_pt = 1.0 / 72.27               # Convert pt to inch
golden_mean = (math.sqrt(5) - 1.0) / 2.0         # Aesthetic ratio
fig_width = fig_width_pt * inches_per_pt  # width in inches
fig_height = fig_width * golden_mean      # height in inches
fig_size = np.array([2. * fig_width, 2. * fig_height])
fig_size2 = np.array([fig_width, fig_height + 0.042 * fig_height])
height = 5
capsize = 0.8
errwidth = 0
aspect = 1.

SNR = [0.0, 5.0, 10.0]
RT60 = [0.256, 0.512, 1.024]
ALPHA = [r"TylerMUSIC", r'MUSIC', r'$\alpha$MUSIC']
M = [5, 8, 10]
N = [2, 3]
NOISE_TYPE = ['office', 'cafet', 'living']



datas = pd.read_pickle("./results_loc.pic")

fig_size = np.array([len(RT60) * fig_width, len(N) * fig_height])
f, ax = plt.subplots(len(N), len(RT60), figsize=fig_size)
# ax = ax[None]
for i_n, n in enumerate(N):
    for i_rt, rt60 in enumerate(RT60):
        tmp_data = datas.loc[
        (datas['RT60'] == rt60) &
        (datas['N'] == n)
        #  &
        # (datas['SNR'] == SNR[1])
        ]
        tmp_data.drop(columns=['time', 'M', 'N', 'noise_type', 'RT60', 'SNR'])
        tmp_data = tmp_data.reset_index()
        tmp_data = tmp_data.drop(columns=['index'])
        g1 = sns.barplot(x="M", y="angle", hue="alpha", ax=ax[i_n, i_rt],
                         data=tmp_data, estimator=np.median)
        # g1 = sns.boxplot(x="M", y="angle", hue="alpha", ax=ax[i_n, i_rt],
        #                  data=tmp_data, showmeans=True, showfliers=False,
        #                  meanprops={"marker":"s","markerfacecolor":"white", "markeredgecolor":"black"})
        if i_n == 0 and i_rt == 1:
            lgd = ax[i_n, i_rt].legend(
            loc='upper center',
            bbox_to_anchor=(0.5, 1.32),
            labelspacing=0,
            borderpad=0.2,
            bbox_transform=BlendedGenericTransform(f.transFigure, ax[i_n, i_rt].transAxes),
            ncol=5)
        else:
            g1.legend_.remove()
        if i_n == 0:
            ax[i_n, i_rt].set_title("RT60={}".format(rt60))

plt.subplots_adjust(hspace=0.2, wspace=0.2)
plt.savefig("localization_results.pdf", bbox_inches='tight', dpi=300)
plt.savefig("localization_results.png", bbox_inches='tight', dpi=300)

for i_n, n in enumerate(N):
    for m in M:
        for alpha in ALPHA:
            tmp_data = datas.loc[
            (datas['N'] == n) &
            (datas['alpha'] == alpha) &
            (datas['M'] == m)
            ]
            value = tmp_data['time'].mean()
            print("N={}, M={}, method={}: {}s".format(n, m, alpha, value))
