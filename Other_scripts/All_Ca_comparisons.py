# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 10:52:37 2022

@author: Theo.ROSSI
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from sklearn.mixture import GaussianMixture

File = r'\\equipe2-nas1\Theo.ROSSI\BackupE\AAVDJ.GluSnFR-S72A\Tidy_files\GluSnFR_avg_clustering_variables_all_profiles_1.5vs2.5vs4mMCa_filtered_3sigma.xlsx'

data = [pd.read_excel(File, sheet_name=i) for i in range(8)]

amp1_15mM = pd.concat([data[0]['AMP1'], data[3]['AMP1']], axis=0, ignore_index=True)
amp1_25mM = pd.concat([data[1]['AMP1'], data[4]['AMP1']], axis=0, ignore_index=True)
amp1_4mM = pd.concat([data[2]['AMP1'], data[5]['AMP1']], axis=0, ignore_index=True)

ppr_20Hz = pd.concat([data[0]['PPR2/1'], data[1]['PPR2/1'], data[2]['PPR2/1']], axis=1, ignore_index=True)
ppr_50Hz = pd.concat([data[3]['PPR2/1'], data[4]['PPR2/1'], data[5]['PPR2/1']], axis=1, ignore_index=True)
ppr_20Hz.columns = ['1.5mM', '2.5mM', '4mM']
ppr_50Hz.columns = ['1.5mM', '2.5mM', '4mM']



amp1_noFail = pd.concat([data[-2]['AMP1'], data[-1]['AMP1']], axis=1)
amp1_noFail.columns = ['A1_1.5mM', 'A1_4mM']

df = pd.concat((amp1_15mM, amp1_25mM, amp1_4mM), axis=1)
df.columns = ['AMP1_1.5mM', 'AMP1_2.5mM', 'AMP1_4mM']
palette = ['k','g','r']
n_modes = [1,5]



def gauss_function(x, amp, x0, sigma):
    return amp * np.exp(-(x - x0) ** 2. / (2. * sigma ** 2.))


for item in range(amp1_noFail.shape[1]):
    fig, ax = plt.subplots(tight_layout=True)
    # sns.histplot(data=df.iloc[:,item], ax=ax, color=palette[item], binwidth=0.03, stat='proportion', alpha=0.4)
    sns.histplot(data=amp1_noFail.iloc[:,item].dropna(), ax=ax, color=palette[item], binwidth=0.07, stat='probability', alpha=0.4)
    
    gmm = GaussianMixture(n_components=n_modes[item], covariance_type="full", tol=0.001)
    gmm_x = np.linspace(0, 2.5, 5000)
    
    gmm = gmm.fit(X=np.expand_dims(amp1_noFail.iloc[:,item].dropna(), 1))
    
    print(f'{amp1_noFail.columns[item]} Modes:')
    print(gmm.means_)
    print('----------------')
    

    gmm_y = np.exp(gmm.score_samples(gmm_x.reshape(-1, 1)))

    # Construct function manually as sum of gaussians
    gmm_y_sum = np.full_like(gmm_x, fill_value=0, dtype=np.float32)
    for m, c, w in zip(gmm.means_.ravel(), gmm.covariances_.ravel(), gmm.weights_.ravel()):
        gmm_y_sum += gauss_function(x=gmm_x, amp=w, x0=m, sigma=np.sqrt(c))

    # Normalize so that integral is 1    
    gmm_y_sum /= np.trapz(gmm_y_sum, gmm_x)

    # ax.plot(gmm_x, gmm_y, color=palette[item], lw=2, label=amp1_noFail.columns[item])

# Annotate diagram
ax.set_xlim(0)
ax.set_ylabel("Probability density")
ax.set_xlabel("AMP1 (DF/F)")

# Draw legend
ax.legend()


#### BOXPLOT 1.5 VS 2.5 VS 4

fig_box, ax_box = plt.subplots(1,3,tight_layout=True)

sns.boxplot(data=[df['AMP1_1.5mM'].dropna(), df['AMP1_2.5mM'], df['AMP1_4mM'].dropna()], ax=ax_box[0], showmeans=True)
sns.boxplot(data=[df['AMP1_1.5mM'].dropna(), df['AMP1_2.5mM'], df['AMP1_4mM'].dropna()], ax=ax_box[0], showmeans=True)

sns.boxplot(data=[ppr_20Hz['1.5mM'].dropna(), ppr_20Hz['2.5mM'], ppr_20Hz['4mM'].dropna()], ax=ax_box[1], showmeans=True)
sns.boxplot(data=[ppr_50Hz['1.5mM'].dropna(), ppr_50Hz['2.5mM'], ppr_50Hz['4mM'].dropna()], ax=ax_box[2], showmeans=True)

ax_box[0].set_ylabel('Amp1 (DF/F)')
ax_box[1].set_ylabel('PPR 2/1 20Hz')
ax_box[2].set_ylabel('PPR 2/1 50Hz')