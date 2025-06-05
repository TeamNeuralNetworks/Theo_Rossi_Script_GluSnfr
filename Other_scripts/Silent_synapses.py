# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 20:20:37 2022

@author: Theo.ROSSI
"""


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.mixture import GaussianMixture

Path = r'\\equipe2-nas1\Theo.ROSSI\BackupE\AAVDJ.GluSnFR-S72A\Silent_boutons'
file = 'Boutons_interval.xlsx'
file_active_boutons = 'Proba_active_boutons_at_least_4_boutons_per_PF.xlsx'
modes = 1
binwidth = 1


def gauss_function(x, amp, x0, sigma):
    return amp * np.exp(-(x - x0) ** 2. / (2. * sigma ** 2.))


file = pd.read_excel(f'{Path}/{file}')
file_active_boutons = pd.read_excel(f'{Path}/{file_active_boutons}')




distances = file['Distance (µm)'].values
mean = np.mean(distances)
std = np.std(distances)
print(f'Mean: {mean}')
print(f'Std: {std}')


boutons_state = file_active_boutons['State'].tolist()
silent = boutons_state.count(0)
active = boutons_state.count(1)
total = silent + active

p_active_active = (active/total) * ((active-1)/(total-1))
P_active_silent = (active/total) * (silent/(total-1))
p_silent_silent = (silent/total) * ((silent-1)/(total-1))

fig_proportion, ax_proportion = plt.subplots()
ax_proportion.bar(0, silent/total, color='k')
ax_proportion.bar(1, active/total, color='g')
ax_proportion.set_ylabel('Proportion')
ax_proportion.set_xticks([0,1], ['Silent', 'Active'])

fig_proba, ax_proba = plt.subplots()
ax_proba.bar(0, p_active_active, color='g')
ax_proba.bar(1, P_active_silent, color='b')
ax_proba.bar(2, p_silent_silent, color='k')
ax_proba.set_ylim(0,1)
ax_proba.set_ylabel('Probability')
ax_proba.set_xticks([0,1,2], ['Active/Active', 'Active/Silent', 'Silent/Silent'])



fig, ax = plt.subplots()
sns.histplot(data=distances, ax=ax, binwidth=binwidth, palette='k', stat='probability', alpha=0.5)

gmm = GaussianMixture(n_components=modes, covariance_type="full", tol=0.001)
gmm_x = np.linspace(0, np.max(distances), 5000)

gmm = gmm.fit(X=np.expand_dims(distances, 1))

print('Modes:')
print(gmm.means_)
print('----------------')


gmm_y = np.exp(gmm.score_samples(gmm_x.reshape(-1, 1)))

# Construct function manually as sum of gaussians
gmm_y_sum = np.full_like(gmm_x, fill_value=0, dtype=np.float32)
for m, c, w in zip(gmm.means_.ravel(), gmm.covariances_.ravel(), gmm.weights_.ravel()):
    gmm_y_sum += gauss_function(x=gmm_x, amp=w, x0=m, sigma=np.sqrt(c))

# Normalize so that integral is 1    
gmm_y_sum /= np.trapz(gmm_y_sum, gmm_x)

ax.plot(gmm_x, gmm_y, color='k', lw=2)
ax.set_xlabel('Distance (µm)')






pf_state = file_active_boutons.groupby(['PF_ID']).mean() #If mean State = 1, all boutons of the PF are active

proportion_active_pf = len([pf_state['State'][i] for i in range(pf_state.shape[0]) if pf_state['State'][i] == 1.])
proportion_at_least_1silent = len([pf_state['State'][i] for i in range(pf_state.shape[0]) if pf_state['State'][i] != 1.])
profiles = ['All active', 'At least 1 silent']


pf_silent_proportion = []
num = ['No silent', '1 silent', ]
for i in range(pf_state.shape[0]):
    active_silent_per_pf = []
    for j in range(file_active_boutons.shape[0]):
        if pf_state.index[i] == file_active_boutons['PF_ID'][j]:
            if file_active_boutons['State'][j] == 1:
                active_silent_per_pf.append(True)
            else:
                active_silent_per_pf.append(False)
    print(active_silent_per_pf)
    for k in range(7):
        if active_silent_per_pf.count(False) == k:
            pf_silent_proportion.append(active_silent_per_pf.count(False))

num = ['All active', '1 silent', '3 silent', '6 silent']
one_silent = len([pf_silent_proportion[i] for i in range(len(pf_silent_proportion)) if pf_silent_proportion[i] == 1])
three_silent = len([pf_silent_proportion[i] for i in range(len(pf_silent_proportion)) if pf_silent_proportion[i] == 3])
six_silent = len([pf_silent_proportion[i] for i in range(len(pf_silent_proportion)) if pf_silent_proportion[i] == 6])

fig_pie, ax_pie = plt.subplots(1,2,tight_layout=True)
ax_pie[0].pie([proportion_active_pf, proportion_at_least_1silent], labels=profiles, colors=['g', 'r'], startangle=90, autopct='%.1f%%')
ax_pie[1].pie([proportion_active_pf, one_silent, three_silent, six_silent], labels=num, startangle=90, autopct='%.1f%%')