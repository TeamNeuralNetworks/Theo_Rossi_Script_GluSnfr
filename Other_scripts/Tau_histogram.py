# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 10:33:29 2022

@author: theo.rossi
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import itertools
from sklearn.mixture import GaussianMixture


Path = r'\\equipe2-nas1\Theo.ROSSI\BackupE\AAVDJ.GluSnFR-S72A\Tidy_files'
file = 'iGluSnFR_Tau_averages.xlsx'

TAU = []
for i in range(2):
    data = pd.read_excel(f'{Path}\{file}', sheet_name=i)
    tau1 = data['TAU1']*1000
    TAU.append(tau1.tolist())

TAU = list(itertools.chain.from_iterable(TAU))
x = np.array(TAU).reshape(-1,1)

x_mean = np.mean(x)
x_std = np.std(x)

gm = GaussianMixture(n_components=1).fit(x)
gm.means_
gmm_x = np.linspace(0, max(x), 5000)
gmm_y = np.exp(gm.score_samples(gmm_x.reshape(-1, 1)))


fig, ax = plt.subplots()
sns.histplot(data=TAU, ax=ax, binwidth=1, stat='probability')
# ax.plot(gmm_x, gmm_y)
ax.set_xlabel('A1 decay-time (ms)')
