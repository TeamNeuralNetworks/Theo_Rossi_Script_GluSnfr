# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 18:20:19 2022

@author: Theo.ROSSI
"""


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import FunctionTransformer
from statannot import add_stat_annotation
import scikit_posthocs as sp
import dabest




File = r'\\equipe2-nas1\Theo.ROSSI\BackupE\AAVDJ.GluSnFR-S72A\Tidy_files\GluSnFR_avg_clustering_variables_all_profiles_1.5vs2.5vs4mMCa_filtered_3sigma.xlsx'
save_path = r'\\equipe2-nas1\Theo.ROSSI\Paper_thesis\Figure1'
save = False



File_20Hz = pd.read_excel(File, sheet_name='20Hz_2,5mM')
File_50Hz = pd.read_excel(File, sheet_name='50Hz_2,5mM')


x = np.arange(1,11)
amps_20Hz = File_20Hz.iloc[:,1:11]
amps_50Hz = File_50Hz.iloc[:,1:11]
amp1 = amps_20Hz['AMP1'].tolist() + amps_50Hz['AMP1'].tolist()
amp1_mean = np.mean(amp1)
amp1_std = np.std(amp1)

PPR_20Hz = [[amps_20Hz.iloc[i,j]/amps_20Hz.iloc[i,0] for j in range(amps_20Hz.shape[1])] for i in range(amps_20Hz.shape[0])]
PPR_50Hz = [[amps_50Hz.iloc[i,j]/amps_50Hz.iloc[i,0] for j in range(amps_50Hz.shape[1])] for i in range(amps_50Hz.shape[0])]
MEAN_20Hz = np.mean(PPR_20Hz, axis=0)
STD_20Hz = np.std(PPR_20Hz, axis=0)
SEM_20Hz = stats.sem(PPR_20Hz, axis=0)
MEAN_50Hz = np.mean(PPR_50Hz, axis=0)
STD_50Hz = np.std(PPR_50Hz, axis=0)
SEM_50Hz = stats.sem(PPR_50Hz, axis=0)


PPR2_1_20Hz = [PPR_20Hz[i][1] for i in range(len(PPR_20Hz))]
PPR2_1_50Hz = [PPR_50Hz[i][1] for i in range(len(PPR_50Hz))]
df_ppr2_1 = pd.concat((pd.DataFrame(PPR2_1_20Hz), pd.DataFrame(PPR2_1_50Hz)), axis=1)
df_ppr2_1.columns = ['PPR2/1_20Hz', 'PPR2/1_50Hz']

Psyn_20Hz = 1 - (File_20Hz.iloc[:,20:23]/100)
Psyn_50Hz = 1 - (File_50Hz.iloc[:,20:23]/100)
all_fail1 = pd.concat((Psyn_20Hz['%Fail1'], Psyn_50Hz['%Fail1']), axis=0, ignore_index=True)
df_fail = pd.concat((all_fail1, Psyn_20Hz['%Fail2'], Psyn_50Hz['%Fail2']), axis=1)
df_fail.columns = ['Psyn1', 'Psyn2_20Hz', 'Psyn2_50Hz']


# TAU = File_20Hz['Tau1'].dropna().tolist() + File_50Hz['Tau1'].dropna().tolist()
# TAU_mean = np.mean(TAU)
# TAU_std = np.std(TAU)


##################################### FIGURES #################################################


### FIG POOLED BOUTONS PPR ###
fig_pool, ax_pool = plt.subplots(1,2, figsize=(14,6), sharey=True, tight_layout=True)
[ax_pool[0].plot(x, PPR_20Hz[i], 'k', alpha=0.05, lw=4) for i in range(len(PPR_20Hz))]
ax_pool[0].plot(x, np.mean(PPR_20Hz, axis=0), color='b', lw=3, marker='o')
ax_pool[0].fill_between(x, MEAN_20Hz+SEM_20Hz, MEAN_20Hz-SEM_20Hz, color='b', alpha=0.4)
ax_pool[0].axhline(1.0, color='k', ls='--', lw=2)
ax_pool[0].set_ylabel('An/A1')
ax_pool[0].set_xlabel('#Pulse')
ax_pool[0].set_title(f'PPR profiles 20Hz\nCa = 2.5mM\nn={len(PPR_20Hz)}')

[ax_pool[1].plot(x, PPR_50Hz[i], 'k', alpha=0.05, lw=4) for i in range(len(PPR_50Hz))]
ax_pool[1].plot(x, np.mean(PPR_50Hz, axis=0), color='orange', lw=3, marker='o')
ax_pool[1].fill_between(x, MEAN_50Hz+SEM_50Hz, MEAN_50Hz-SEM_50Hz, color='orange', alpha=0.4)
ax_pool[1].axhline(1.0, color='k', ls='--', lw=2)
ax_pool[1].set_ylabel('An/A1')
ax_pool[1].set_xlabel('#Pulse')
ax_pool[1].set_title(f'PPR profiles 50Hz\nCa = 2.5mM\nn={len(PPR_50Hz)}')

if save == True:
    plt.savefig(f'{save_path}\Pooled_boutons_profiles.pdf')




### FIG AMP1 ###
################

fig_amp1, ax_amp1 = plt.subplots()
sns.boxplot(data=[amp1, amps_20Hz['AMP2'], amps_50Hz['AMP2']], ax=ax_amp1, palette=['grey','b','darkorange'], showmeans=True)
ax_amp1.set_ylabel('Amplitude (dF/F)')
ax_amp1.set_ylim(0,2)

stats.kruskal(amp1, amps_20Hz['AMP2'], amps_50Hz['AMP2'])
sp.posthoc_dunn([amp1, amps_20Hz['AMP2'], amps_50Hz['AMP2']], p_adjust='bonferroni')

# sns.histplot(data=amp1, binwidth=0.04, ax=ax_amp1, color='k', stat='probability',  alpha=0.5)
# ax_amp1.set_xlabel('DF/F')

if save == True:
    plt.savefig(f'{save_path}\Amp1_histogram.pdf')




### FIG PPR2/1 ###
##################

fig_ppr, ax_ppr = plt.subplots(1,2,tight_layout=True)
sns.boxplot(data=df_ppr2_1, ax=ax_ppr[0], palette=['b', 'orange'], showmeans=True)

add_stat_annotation(ax_ppr[0], data=df_ppr2_1, box_pairs=[('PPR2/1_20Hz', 'PPR2/1_50Hz')],
                    test='Mann-Whitney', text_format='full', verbose=2)
ax_ppr[0].set_ylabel('A2/A1')
ax_ppr[0].set_xticklabels(['20Hz','50Hz'])
ax_ppr[0].set_ylim(0)

ppr_eff_size = dabest.load(df_ppr2_1, idx=('PPR2/1_20Hz', 'PPR2/1_50Hz'), resamples=5000)
ppr_eff_size.mean_diff.plot(ax=ax_ppr[1], custom_palette=['b', 'orange'])

if save == True:
    plt.savefig(f'{save_path}\PPR_comparison.pdf')


### FIG TAU ###
# fig_tau, ax_tau = plt.subplots()
# sns.histplot(data=TAU, bins=40, ax=ax_tau, color='k', stat='proportion', alpha=0.5)
# ax_tau.set_xlabel('Tau (ms)')
# ax_tau.set_title('µ = {:.2f} ms ; std = {:.2f}'.format(TAU_mean, TAU_std))

# if save == True:
#     plt.savefig(f'{save_path}\Tau1_histogram.pdf')




### FIG PPR2/1 VS AMP1 ###
##########################

fig_ppr_amp, ax_ppr_amp = plt.subplots(tight_layout=True)

transformer = FunctionTransformer(np.log, validate=True)
model = LinearRegression()

x_20Hz = np.array(amps_50Hz['AMP1']).reshape(-1,1)
y_20Hz = np.array(PPR2_1_50Hz).reshape(-1,1)

x_20Hz_trans = transformer.fit_transform(x_20Hz)
reg = model.fit(x_20Hz_trans, y_20Hz)
r = np.sqrt(reg.score(x_20Hz_trans, y_20Hz))
pred = reg.predict(x_20Hz_trans)

# PAIRS = [[x,y] for x, y in zip(amps_50Hz['AMP1'], PPR2_1_50Hz)]
# PAIRS_sorted = sorted(PAIRS)
# X = [PAIRS_sorted[pair][0] for pair in range(len(PAIRS_sorted))]
# X = np.array(X).reshape(len(X),1)
# Y = [PAIRS_sorted[pair][1] for pair in range(len(PAIRS_sorted))]

# model = SVR()
# model.fit(X, Y)
# model.score(X, Y)
# prediction = model.predict(X)

ax_ppr_amp.scatter(x_20Hz, y_20Hz, color='limegreen', alpha=0.8)
ax_ppr_amp.scatter(amps_50Hz['AMP1'], PPR2_1_50Hz, color='darkorange', alpha=0.8)
ax_ppr_amp.plot(sorted(x_20Hz), sorted(pred, reverse=True), color='darkorange', alpha=0.8)
ax_ppr_amp.axhline(y=1, color='k', ls='--')

ax_ppr_amp.set_ylabel('A2/A1')
ax_ppr_amp.set_xlabel('A1')
ax_ppr_amp.set_ylim(0)

if save == True:
    plt.savefig(f'{save_path}\PPR_against_Amp1.pdf')




### FIG Psyn AMP1 ###
#####################

fig_fail, ax_fail = plt.subplots(1,2,figsize=(12,8))
sns.boxplot(data=df_fail, ax=ax_fail[0], palette=['grey','limegreen','darkorange'], showmeans=True)
# add_stat_annotation(ax_fail[0], data=df_fail, box_pairs=[('Psyn1','Psyn2_20Hz'),
#                                                          ('Psyn1','Psyn2_50Hz'),
#                                                          ('Psyn2_20Hz','Psyn2_50Hz')], test='Mann-Whitney', text_format='full', verbose=2)
ax_fail[0].set_ylabel('Failures (%)')
ax_fail[0].set_ylim(0,1.3)

stats.kruskal(df_fail['Psyn1'], df_fail['Psyn2_20Hz'].dropna(), df_fail['Psyn2_50Hz'].dropna())
sp.posthoc_dunn([df_fail['Psyn1'], df_fail['Psyn2_20Hz'].dropna(), df_fail['Psyn2_50Hz'].dropna()], p_adjust='bonferroni')
# fail_eff_size = dabest.load(df_fail, idx=(('Psyn1','Psyn2_20Hz','Psyn2_50Hz')), resamples=5000)
# fail_eff_size.mean_diff.plot(ax=ax_fail[1], custom_palette=['grey','b','orange'])




### FIG FAIL1 AGAINST AMP1 ###
##############################

fig_fail1_amp1, ax_fail1_amp1 = plt.subplots()

x_fail = np.array(amp1).reshape(-1,1)
y_fail = np.array(all_fail1).reshape(-1,1)
x_trans = transformer.fit_transform(x_fail)
reg = model.fit(x_trans, y_fail)
r = np.sqrt(reg.score(x_trans, y_fail))
pred = reg.predict(x_trans)
ax_fail1_amp1.scatter(amp1, all_fail1, color='k', alpha=0.8)
ax_fail1_amp1.plot(sorted(x_fail), sorted(pred), color='r')
ax_fail1_amp1.set_title('A1 Psyn against A1')
ax_fail1_amp1.set_ylabel('A1 Psyn')
ax_fail1_amp1.set_xlabel('A1 (DF/F)')
ax_fail1_amp1.set_xlim(0)
ax_fail1_amp1.set_ylim(0)




### FIG FAIL1 AGAINST PPR ###
#############################

fig_fail1_ppr, ax_fail1_ppr = plt.subplots()
ax_fail1_ppr.scatter(df_ppr2_1['PPR2/1_20Hz'], Psyn_20Hz['%Fail1'], color='limegreen', alpha=0.8)
ax_fail1_ppr.set_title('A1 Psyn against PPR 20Hz')
ax_fail1_ppr.set_ylabel('A1 Psyn')
ax_fail1_ppr.set_xlabel('PPR2/1')

ax_fail1_ppr.scatter(df_ppr2_1['PPR2/1_50Hz'].dropna(), Psyn_50Hz['%Fail1'], color='darkorange', alpha=0.8)
# ax_fail1_ppr[1].set_title('A1 failures against PPR 50Hz')
# ax_fail1_ppr[1].set_ylabel('A1 failures (%)')
# ax_fail1_ppr[1].set_xlabel('PPR2/1 50Hz')



if save == True:
    plt.savefig(f'{save_path}\Failures_amp1.pdf')