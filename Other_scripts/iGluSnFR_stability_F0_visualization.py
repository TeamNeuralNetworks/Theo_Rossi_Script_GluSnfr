# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 18:30:58 2022

@author: theo.rossi
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
from sklearn.mixture import GaussianMixture


Path = r'\\equipe2-nas1\Theo.ROSSI\BackupE\AAVDJ.GluSnFR-S72A\Tidy_files'
file = 'iGluSnFR_measures_F0_averages.xlsx'
variable_against_F0 = 'Psyn2'

data = pd.read_excel(f'{Path}/{file}')
x = np.array(data['F0_AVG']).reshape(-1,1)
y = np.array(data[variable_against_F0]).reshape(-1,1)
clusters = data['Clusters'].tolist()

x_mean = np.mean(x)
y_mean = np.mean(y)

x_std = np.std(x)
y_std = np.std(y)

model = LinearRegression()
reg = model.fit(x,y)
pred = reg.predict(x)

plt.figure()
plt.scatter(x, y, color='g', alpha=0.8)
plt.plot(x, pred, color='g')
plt.scatter(x_mean, y_mean, color='r')
plt.errorbar(x_mean, y_mean, xerr=x_std, yerr=y_std, capsize=6, color='r')
plt.ylim(0)
plt.xlim(0)
plt.xlabel('F0')
plt.ylabel(variable_against_F0)


pearson_stat = stats.pearsonr(data['F0_AVG'].tolist(), data[variable_against_F0].tolist())
plt.title('r = {:.2f}; p = {:.2f}'.format(pearson_stat[0],pearson_stat[1]))
print(f'{variable_against_F0} vs F0_AVG')
print('Pearson correlation coeff and p_value')
print(pearson_stat)
print('-----------------------')
print('-----------------------')



n_clust=4
C = pd.DataFrame([[data['F0_AVG'][i] for i in range(len(clusters)) if clusters[i]==j] for j in range(n_clust)]).T
C.columns = ['C1','C2','C3','C4']

fig, ax = plt.subplots()
sns.boxplot(data=C, ax=ax, showmeans=True)
ax.set_ylabel('F0_avg')
ax.set_ylim(0)


kruskal = stats.kruskal(C['C1'], C['C2'].dropna(), C['C3'].dropna(), C['C4'].dropna())
print('F0 between clusters')
print('Kruskal-Wallis test')
print(kruskal)




gm = GaussianMixture(n_components=2).fit(x)
gm.means_
gmm_x = np.linspace(0, max(x), 5000)
gmm_y = np.exp(gm.score_samples(gmm_x.reshape(-1, 1)))

fig_hist_F0, ax_hist_F0 = plt.subplots()
sns.histplot(data=data['F0_AVG'], ax=ax_hist_F0, binwidth=1, stat='probability')
ax_hist_F0.plot(gmm_x, gmm_y)