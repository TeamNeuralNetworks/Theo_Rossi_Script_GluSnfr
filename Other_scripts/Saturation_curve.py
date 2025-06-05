# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 18:39:42 2020

@author: Theo.ROSSI
"""


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import seaborn as sns
from statannot import add_stat_annotation
from scipy import stats,optimize


def sigmoid(x):
    y = 1./(1.+np.exp(-1*(x)))
    return y

def find_x(y):
    x = np.log(1./((1./y)-1))
    return x

def affine(x,a,b):
    y = a*x+b
    return y

colors = ['tomato', 'turquoise', 'purple']

path = r'\\equipe2-nas1\Theo.ROSSI\BackupE\AAVDJ.GluSnFR-S72A\Controls\Probe_saturation\Saturation_profiles'
path2 = r'\\equipe2-nas1\Theo.ROSSI\BackupE\AAVDJ.GluSnFR-S72A\Controls\Probe_saturation\Saturation_Amps'

folder = sorted(os.listdir(path))
folder2 = sorted(os.listdir(path2))

AMP_DF_FILES, AMP_DF_F0_FILES = [],[]
AMP_dF_VALUES, AMP_dF_F0_VALUES = [],[]
AVERAGE_DF, AVERAGE_DF_F0 = [],[]
DF, DF_F0 = [],[]
PEAK1_DF_2_5, PEAK2_DF_2_5, PEAK3_DF_2_5, PEAK1_DF_F0_2_5, PEAK2_DF_F0_2_5, PEAK3_DF_F0_2_5 = [],[],[],[],[],[]
PEAK1_DF_4, PEAK2_DF_4, PEAK3_DF_4, PEAK1_DF_F0_4, PEAK2_DF_F0_4, PEAK3_DF_F0_4 = [],[],[],[],[],[]
FMAX_DF_2_5, FMAX_DF_4, FMAX_DF_F0_2_5, FMAX_DF_F0_4 = [],[],[],[]
F50_DF_2_5, F50_DF_4, F50_DF_F0_2_5, F50_DF_F0_4 = [],[],[],[]
NORM_F1_25, NORM_F2_25, NORM_F3_25, NORM_F1_F0_25, NORM_F2_F0_25, NORM_F3_F0_25 = [],[],[],[],[],[]
NORM_F1_4, NORM_F2_4, NORM_F3_4, NORM_F1_F0_4, NORM_F2_F0_4, NORM_F3_F0_4 = [],[],[],[],[],[]
PPR_DF_25, PPR_DF_F0_25 = [],[]
PPR_DF_4, PPR_DF_F0_4 = [],[]
PPR_DF_25_2_1, PPR_DF_25_3_1, PPR_DF_F0_25_2_1, PPR_DF_F0_25_3_1 = [],[],[],[]
PPR_DF_4_2_1, PPR_DF_4_3_1, PPR_DF_F0_4_2_1, PPR_DF_F0_4_3_1 = [],[],[],[]
TIME_DF, TIME_DF_F0 = [],[]

for file in range(len(folder)):
    if '200Hz' in folder[file]:
        if 'dF.xlsx' in folder[file]:
            DF.append(folder[file])
            data = pd.read_excel('{}/{}'.format(path, folder[file]))
            
            for col in range(len(data.columns)):
                if 'Average' == data.columns[col]:
                    average_df = data.iloc[:,col].values
                    AVERAGE_DF.append(average_df)
                
                if 'Time' == data.columns[col]:
                    timescale_df = data.iloc[:,col].values
                    TIME_DF.append(timescale_df)
            
        else:
            DF_F0.append(folder[file])
            data = pd.read_excel('{}/{}'.format(path, folder[file]))
            
            for col in range(len(data.columns)):
                if 'Average' == data.columns[col]:
                    average_df_f0 = data.iloc[:,col].values
                    AVERAGE_DF_F0.append(average_df_f0)
                
                if 'Time' == data.columns[col]:
                    timescale_df_f0 = data.iloc[:,col].values
                    TIME_DF_F0.append(timescale_df_f0)
            # if '4mM' in folder[file]:
            #     plt.figure(tight_layout=True)
            #     plt.plot(timescale_df_f0,average_df_f0,'orange')
            # else:
            #     plt.figure(tight_layout=True)
            #     plt.plot(timescale_df_f0,average_df_f0,'limegreen')

for row in range(len(folder2)):
    if 'dF_Amp' in folder2[row]:
        AMP_DF_FILES.append(folder2[row])
        data = pd.read_excel('{}/{}'.format(path2, folder2[row]))
        values = data.iloc[0,1:].values
        AMP_dF_VALUES.append(values)
    
    if 'dF_F0_Amp' in folder2[row]:
        AMP_DF_F0_FILES.append(folder2[row])
        data = pd.read_excel('{}/{}'.format(path2, folder2[row]))
        values = data.iloc[0,1:].values
        AMP_dF_F0_VALUES.append(values)


fig, ax = plt.subplots(5,12, figsize=(18,10), sharey=True)
fig2, ax2 = plt.subplots(5,12, figsize=(18,10), sharey=True)
fig3, ax3 = plt.subplots(5,12, figsize=(18,10), sharey=True)
fig4, ax4 = plt.subplots(5,12, figsize=(18,10), sharey=True)
fig5, ax5 = plt.subplots(5,12, figsize=(18,10), sharey=True)
fig6, ax6 = plt.subplots(5,12, figsize=(18,10), sharey=True)

fig7, ax7 = plt.subplots(2,3, figsize=(12,6))
fig8, ax8 = plt.subplots(2,3, figsize=(12,6))


i,j = 0,0

for item in range(len(AVERAGE_DF)):
    
    if '50-200Hz' not in AMP_DF_FILES[item]:
        x1 = np.ravel(np.where(TIME_DF[item] >= 0.55))[0]
        x2 = np.ravel(np.where(TIME_DF[item] <= 0.6))[-1]
    else:
        x1 = np.ravel(np.where(TIME_DF[item] >= 0.8))[0]
        x2 = np.ravel(np.where(TIME_DF[item] <= 0.85))[-1]
    
    
    Fmax_dF = np.nanmean(AVERAGE_DF[item][x1:x2])
    F50_df = Fmax_dF/2
    
    Fmax_dF_F0 = np.nanmean(AVERAGE_DF_F0[item][x1:x2])
    F50_dF_F0 = Fmax_dF_F0/2
    
    if '4mM' in AMP_DF_FILES[item]:
        FMAX_DF_4.append(Fmax_dF)
        F50_DF_4.append(F50_df)
        FMAX_DF_F0_4.append(Fmax_dF_F0)
        F50_DF_F0_4.append(F50_dF_F0)
    else:
        FMAX_DF_2_5.append(Fmax_dF)
        F50_DF_2_5.append(F50_df)
        FMAX_DF_F0_2_5.append(Fmax_dF_F0)
        F50_DF_F0_2_5.append(F50_dF_F0)
    
    norm_F1 = (F50_df-AMP_dF_VALUES[item][0])/F50_df
    norm_F2 = (F50_df-AMP_dF_VALUES[item][1])/F50_df
    norm_F3 = (F50_df-AMP_dF_VALUES[item][2])/F50_df
    
    norm_F1_F0 = (F50_dF_F0-AMP_dF_F0_VALUES[item][0])/F50_dF_F0
    norm_F2_F0 = (F50_dF_F0-AMP_dF_F0_VALUES[item][1])/F50_dF_F0
    norm_F3_F0 = (F50_dF_F0-AMP_dF_F0_VALUES[item][2])/F50_dF_F0

    PPR_VALUES_DF_4, PPR_VALUES_DF_F0_4 = [],[]
    PPR_VALUES_DF_25, PPR_VALUES_DF_F0_25 = [],[]
    

    if j == 12:
        i += 1
        j = 0
    
    
    if '4mM' in AMP_DF_FILES[item]:
        PEAK1_DF_4.append(AMP_dF_VALUES[item][0])
        PEAK2_DF_4.append(AMP_dF_VALUES[item][1])
        PEAK3_DF_4.append(AMP_dF_VALUES[item][2])
        
        PEAK1_DF_F0_4.append(AMP_dF_F0_VALUES[item][0])
        PEAK2_DF_F0_4.append(AMP_dF_F0_VALUES[item][1])
        PEAK3_DF_F0_4.append(AMP_dF_F0_VALUES[item][2])
        
        NORM_F1_4.append(norm_F1)
        NORM_F2_4.append(norm_F2)
        NORM_F3_4.append(norm_F3)
        
        NORM_F1_F0_4.append(norm_F1_F0)
        NORM_F2_F0_4.append(norm_F2_F0)
        NORM_F3_F0_4.append(norm_F3_F0)
        
        for value in range(len(AMP_dF_VALUES[item])):
            ppr_dF_4 = AMP_dF_VALUES[item][value]/AMP_dF_VALUES[item][0]
            ppr_dF_F0_4 = AMP_dF_F0_VALUES[item][value]/AMP_dF_F0_VALUES[item][0]
            PPR_VALUES_DF_4.append(ppr_dF_4)
            PPR_VALUES_DF_F0_4.append(ppr_dF_F0_4)
        PPR_DF_4.append(PPR_VALUES_DF_4)
        PPR_DF_F0_4.append(PPR_VALUES_DF_F0_4)   
        
        ax[i,j].plot(TIME_DF[item], AVERAGE_DF[item], 'darkorange', linewidth=0.5)
        ax2[i,j].plot(TIME_DF_F0[item], AVERAGE_DF_F0[item], 'darkorange', linewidth=0.5)
        ax3[i,j].scatter(np.zeros(3), AMP_dF_VALUES[item], color=colors, s=20)
        ax4[i,j].scatter(np.zeros(3), AMP_dF_F0_VALUES[item], color=colors, s=20)
        ax5[i,j].plot(PPR_VALUES_DF_4, 'darkorange'), ax5[i,j].scatter(np.arange(0,3,1), PPR_VALUES_DF_4, color=colors, s=20)
        ax6[i,j].plot(PPR_VALUES_DF_F0_4, 'darkorange'), ax6[i,j].scatter(np.arange(0,3,1), PPR_VALUES_DF_F0_4, color=colors, s=20)
        
        
    else:
        PEAK1_DF_2_5.append(AMP_dF_VALUES[item][0])
        PEAK2_DF_2_5.append(AMP_dF_VALUES[item][1])
        PEAK3_DF_2_5.append(AMP_dF_VALUES[item][2])
        
        PEAK1_DF_F0_2_5.append(AMP_dF_F0_VALUES[item][0])
        PEAK2_DF_F0_2_5.append(AMP_dF_F0_VALUES[item][1])
        PEAK3_DF_F0_2_5.append(AMP_dF_F0_VALUES[item][2])
        
        NORM_F1_25.append(norm_F1)
        NORM_F2_25.append(norm_F2)
        NORM_F3_25.append(norm_F3)
        
        NORM_F1_F0_25.append(norm_F1_F0)
        NORM_F2_F0_25.append(norm_F2_F0)
        NORM_F3_F0_25.append(norm_F3_F0)
        
        for value in range(len(AMP_dF_VALUES[item])):
            ppr_dF_25 = AMP_dF_VALUES[item][value]/AMP_dF_VALUES[item][0]
            ppr_dF_F0_25 = AMP_dF_F0_VALUES[item][value]/AMP_dF_F0_VALUES[item][0]
            PPR_VALUES_DF_25.append(ppr_dF_25)
            PPR_VALUES_DF_F0_25.append(ppr_dF_F0_25)
        PPR_DF_25.append(PPR_VALUES_DF_25)
        PPR_DF_F0_25.append(PPR_VALUES_DF_F0_25)  
        
        ax[i,j].plot(TIME_DF[item], AVERAGE_DF[item], 'cornflowerblue', linewidth=0.5)
        ax2[i,j].plot(TIME_DF_F0[item], AVERAGE_DF_F0[item], 'cornflowerblue', linewidth=0.5)
        ax3[i,j].scatter(np.zeros(3), AMP_dF_VALUES[item], color=colors, s=20)
        ax4[i,j].scatter(np.zeros(3), AMP_dF_F0_VALUES[item], color=colors, s=20)
        ax5[i,j].plot(PPR_VALUES_DF_25, 'cornflowerblue'), ax5[i,j].scatter(np.arange(0,3,1), PPR_VALUES_DF_25, color=colors, s=20)
        ax6[i,j].plot(PPR_VALUES_DF_F0_25, 'cornflowerblue'), ax6[i,j].scatter(np.arange(0,3,1), PPR_VALUES_DF_F0_25, color=colors, s=20)
        
    
    ax[i,j].plot([TIME_DF[item][0],TIME_DF[item][-1]], [Fmax_dF,Fmax_dF], 'darkred', linestyle='--')
    ax[i,j].set_title('{}\nFmax = {:.3f}'.format(DF[item], Fmax_dF), fontsize=4)
    ax[0,0].set_ylabel('dF')
    
    ax2[i,j].plot([TIME_DF[item][0],TIME_DF[item][-1]], [Fmax_dF_F0,Fmax_dF_F0], 'darkred', linestyle='--')
    ax2[i,j].set_title('{}\nFmax = {:.3f}'.format(DF_F0[item], Fmax_dF_F0), fontsize=4)
    ax2[0,0].set_ylabel('dF/F0')
    
    ax3[i,j].scatter(0, Fmax_dF, color='darkred', s=100, marker='_')
    ax3[i,j].scatter(0, F50_df, color='cornflowerblue', s=100, marker='_')
    ax3[0,0].set_ylabel('dF')
    
    ax4[i,j].scatter(0, Fmax_dF_F0, color='darkred', s=200, marker='_')
    ax4[i,j].scatter(0, F50_dF_F0, color='cornflowerblue', s=200, marker='_')
    ax4[0,0].set_ylabel('dF/F0')
    
    ax5[i,j].plot([TIME_DF[item][0],TIME_DF[item][-1]], [1.0,1.0], 'k', linestyle='--')
    ax6[i,j].plot([TIME_DF_F0[item][0],TIME_DF_F0[item][-1]], [1.0,1.0], 'k', linestyle='--')
    
    
    j += 1


for row in range(len(PPR_DF_25)):
    PPR_DF_25_2_1.append(PPR_DF_25[row][1])
    PPR_DF_F0_25_2_1.append(PPR_DF_F0_25[row][1])
    PPR_DF_25_3_1.append(PPR_DF_25[row][2])
    PPR_DF_F0_25_3_1.append(PPR_DF_F0_25[row][2])

for row in range(len(PPR_DF_4)):
    PPR_DF_4_2_1.append(PPR_DF_4[row][1])
    PPR_DF_F0_4_2_1.append(PPR_DF_F0_4[row][1])
    PPR_DF_4_3_1.append(PPR_DF_4[row][2])
    PPR_DF_F0_4_3_1.append(PPR_DF_F0_4[row][2])


sns.distplot(NORM_F1_25, bins=9, ax=ax7[0,0]), sns.distplot(NORM_F1_4, bins=9, ax=ax7[0,0])
sns.distplot(NORM_F2_25, bins=9, ax=ax7[0,1]), sns.distplot(NORM_F2_4, bins=9, ax=ax7[0,1])
sns.distplot(NORM_F3_25, bins=9, ax=ax7[0,2]), sns.distplot(NORM_F3_4, bins=9, ax=ax7[0,2])
ax7[0,0].set_title('Peak1', color=colors[0]), ax7[0,0].set_ylabel('Count')
ax7[0,1].set_title('Peak2', color=colors[1])
ax7[0,2].set_title('Peak3', color=colors[2])


sns.distplot(NORM_F1_25, bins=9, hist=False, hist_kws={'cumulative':True}, kde_kws={'cumulative':True}, ax=ax7[1,0]), sns.distplot(NORM_F1_4, bins=9, hist=False, hist_kws={'cumulative':True}, kde_kws={'cumulative':True}, ax=ax7[1,0])
sns.distplot(NORM_F2_25, bins=9, hist=False, hist_kws={'cumulative':True}, kde_kws={'cumulative':True}, ax=ax7[1,1]), sns.distplot(NORM_F2_4, bins=9, hist=False, hist_kws={'cumulative':True}, kde_kws={'cumulative':True}, ax=ax7[1,1])
sns.distplot(NORM_F3_25, bins=9, hist=False, hist_kws={'cumulative':True}, kde_kws={'cumulative':True}, ax=ax7[1,2]), sns.distplot(NORM_F3_4, bins=9, hist=False, hist_kws={'cumulative':True}, kde_kws={'cumulative':True}, ax=ax7[1,2])
ax7[1,0].set_ylabel('Cumulative frequencies')
ax7[1,0].set_xlabel('((Fmax/2)-F1)/(Fmax/2)')

ks_test, pvalue = stats.ks_2samp(NORM_F1_25, NORM_F1_4)
ks_test2, pvalue2 = stats.ks_2samp(NORM_F2_25, NORM_F2_4)
ks_test3, pvalue3 = stats.ks_2samp(NORM_F3_25, NORM_F3_4)
print('KS-test NORM_F1_25 vs NORM_F1_4:\nks_value = {:.2f} ; pvalue = {:.2f}'.format(ks_test, pvalue))
print('KS-test NORM_F2_25 vs NORM_F2_4:\nks_value = {:.2f} ; pvalue = {:.2f}'.format(ks_test2, pvalue2))
print('KS-test NORM_F3_25 vs NORM_F3_4:\nks_value = {:.2f} ; pvalue = {:.2f}'.format(ks_test3, pvalue3))

ax7[1,0].set_title('KS-test pvalue = {:.2f}'.format(pvalue))
ax7[1,1].set_title('KS-test pvalue = {:.2f}'.format(pvalue2))
ax7[1,2].set_title('KS-test pvalue = {:.2f}'.format(pvalue3))


sns.distplot(NORM_F1_F0_25, bins=9, ax=ax8[0,0], color='limegreen'), sns.distplot(NORM_F1_F0_4, bins=9, ax=ax8[0,0], color='orange')
sns.distplot(NORM_F2_F0_25, bins=9, ax=ax8[0,1], color='limegreen'), sns.distplot(NORM_F2_F0_4, bins=9, ax=ax8[0,1], color='orange')
sns.distplot(NORM_F3_F0_25, bins=9, ax=ax8[0,2], color='limegreen'), sns.distplot(NORM_F3_F0_4, bins=9, ax=ax8[0,2], color='orange')
ax8[0,0].set_title('DF/F0_Peak1', color=colors[0]), ax8[0,0].set_ylabel('Count')
ax8[0,1].set_title('DF/F0_Peak2', color=colors[1])
ax8[0,2].set_title('DF/F0_Peak3', color=colors[2])


ks_test4, pvalue4 = stats.ks_2samp(NORM_F1_F0_25, NORM_F1_F0_4)
ks_test5, pvalue5 = stats.ks_2samp(NORM_F2_F0_25, NORM_F2_F0_4)
ks_test6, pvalue6 = stats.ks_2samp(NORM_F3_F0_25, NORM_F3_F0_4)
print('------------------')
print('KS-test NORM_F1_F0_25 vs NORM_F1_F0_4:\nks_value = {:.2f} ; pvalue = {:.2f}'.format(ks_test4, pvalue4))
print('KS-test NORM_F2_F0_25 vs NORM_F2_F0_4:\nks_value = {:.2f} ; pvalue = {:.2f}'.format(ks_test5, pvalue5))
print('KS-test NORM_F3_F0_25 vs NORM_F3_F0_4:\nks_value = {:.2f} ; pvalue = {:.2f}'.format(ks_test6, pvalue6))

sns.distplot(NORM_F1_F0_25, bins=9, hist=False, hist_kws={'cumulative':True}, kde_kws={'cumulative':True}, ax=ax8[1,0], color='limegreen'), sns.distplot(NORM_F1_F0_4, bins=9, hist=False, hist_kws={'cumulative':True}, kde_kws={'cumulative':True}, ax=ax8[1,0], color='orange')
sns.distplot(NORM_F2_F0_25, bins=9, hist=False, hist_kws={'cumulative':True}, kde_kws={'cumulative':True}, ax=ax8[1,1], color='limegreen'), sns.distplot(NORM_F2_F0_4, bins=9, hist=False, hist_kws={'cumulative':True}, kde_kws={'cumulative':True}, ax=ax8[1,1], color='orange')
sns.distplot(NORM_F3_F0_25, bins=9, hist=False, hist_kws={'cumulative':True}, kde_kws={'cumulative':True}, ax=ax8[1,2], color='limegreen'), sns.distplot(NORM_F3_F0_4, bins=9, hist=False, hist_kws={'cumulative':True}, kde_kws={'cumulative':True}, ax=ax8[1,2], color='orange')
ax8[1,0].set_ylabel('Cumulative frequencies')
ax8[1,0].set_xlabel('((Fmax/2)-F1)/(Fmax/2)')

ax8[1,0].set_title('KS-test pvalue = {:.2f}'.format(pvalue4))
ax8[1,1].set_title('KS-test pvalue = {:.2f}'.format(pvalue5))
ax8[1,2].set_title('KS-test pvalue = {:.2f}'.format(pvalue6))


fig9, ax9 = plt.subplots(1,1, figsize=(6,6), tight_layout=True)

globalDf = pd.concat((pd.DataFrame(PEAK1_DF_2_5), pd.DataFrame(PEAK1_DF_4),
                     pd.DataFrame(PEAK2_DF_2_5), pd.DataFrame(PEAK2_DF_4),
                     pd.DataFrame(PEAK3_DF_2_5), pd.DataFrame(PEAK3_DF_4),
                     pd.DataFrame(F50_DF_2_5), pd.DataFrame(F50_DF_4),
                     pd.DataFrame(FMAX_DF_2_5), pd.DataFrame(FMAX_DF_4)),axis=1)

globalDf.columns=['P1\n2.5mM','P1\n4mM','P2\n2.5mM','P2\n4mM','P3\n2.5mM','P3\n4mM','F50\n2.5mM','F50\n4mM','Fmax\n2.5mM','Fmax\n4mM']

palette = ['limegreen','orange','limegreen','orange','limegreen','orange','limegreen','orange','limegreen','orange']
sns.boxplot(data=globalDf, showmeans=True, ax=ax9, palette=palette, meanprops={'markerfacecolor':'red',
                                                                              'markeredgecolor':'black',
                                                                              'markersize':'8'})
sns.swarmplot(data=globalDf, ax=ax9, color='0.25')
ax9.set_ylabel('DF')

add_stat_annotation(ax9, data=globalDf, box_pairs=[('P1\n2.5mM','P1\n4mM'), ('P2\n2.5mM','P2\n4mM'), ('P3\n2.5mM','P3\n4mM'),
                                                   ('P1\n2.5mM','Fmax\n2.5mM'), ('P2\n2.5mM','Fmax\n2.5mM'), ('P3\n2.5mM','Fmax\n2.5mM'),
                                                   ('P1\n4mM','Fmax\n4mM'), ('P2\n4mM','Fmax\n4mM'), ('P3\n4mM','Fmax\n4mM'), ('Fmax\n2.5mM','Fmax\n4mM')],
                    test='Mann-Whitney', text_format='star', verbose=2)


fig10, ax10 = plt.subplots(1,1, figsize=(6,6), tight_layout=True)

globalDf_F0 = pd.concat((pd.DataFrame(PEAK1_DF_F0_2_5), pd.DataFrame(PEAK1_DF_F0_4),
                     pd.DataFrame(PEAK2_DF_F0_2_5), pd.DataFrame(PEAK2_DF_F0_4),
                     pd.DataFrame(PEAK3_DF_F0_2_5), pd.DataFrame(PEAK3_DF_F0_4),
                     pd.DataFrame(F50_DF_F0_2_5), pd.DataFrame(F50_DF_F0_4),
                     pd.DataFrame(FMAX_DF_F0_2_5), pd.DataFrame(FMAX_DF_F0_4)),axis=1)

globalDf_F0.columns=['P1\n2.5mM','P1\n4mM','P2\n2.5mM','P2\n4mM','P3\n2.5mM','P3\n4mM','F50\n2.5mM','F50\n4mM','Fmax\n2.5mM','Fmax\n4mM']

sns.boxplot(data=globalDf_F0, showmeans=True, ax=ax10, palette=palette, meanprops={'markerfacecolor':'red',
                                                                                  'markeredgecolor':'black',
                                                                                  'markersize':'8'})
ax10.set_title('n_2.5mM: {} ; n_4mM: {}'.format(len(PEAK1_DF_F0_2_5),len(PEAK1_DF_F0_4)))
ax10.set_ylabel('DF/F0')

for i in range(globalDf_F0.shape[0]):
    data1 = globalDf_F0.iloc[i,:].values
    ax10.scatter([0,1,2,3,4,5,6,7,8,9],[data1[0],data1[1],data1[2],data1[3],data1[4],data1[5],data1[6],data1[7],data1[8],data1[9]],
                 color=[palette[0],palette[1],palette[2],palette[3],palette[4],palette[5],palette[6],palette[7],palette[8],palette[9]], edgecolors='k', s=20, alpha=0.6)

add_stat_annotation(ax10, data=globalDf_F0, box_pairs=[('P1\n2.5mM','P1\n4mM'), ('P2\n2.5mM','P2\n4mM'), ('P3\n2.5mM','P3\n4mM'),
                                                       ('P1\n2.5mM','Fmax\n2.5mM'), ('P2\n2.5mM','Fmax\n2.5mM'), ('P3\n2.5mM','Fmax\n2.5mM'),
                                                       ('P1\n4mM','Fmax\n4mM'), ('P2\n4mM','Fmax\n4mM'), ('P3\n4mM','Fmax\n4mM'), ('Fmax\n2.5mM','Fmax\n4mM')],
                    test='Mann-Whitney', text_format='star', verbose=2)


fig11, ax11 = plt.subplots(1,1, figsize=(4,5), tight_layout=True)

pprDf_F0 = pd.concat((pd.DataFrame(PPR_DF_F0_25_2_1), pd.DataFrame(PPR_DF_F0_4_2_1),
                   pd.DataFrame(PPR_DF_F0_25_3_1), pd.DataFrame(PPR_DF_F0_4_3_1)), axis=1)
pprDf_F0.columns = ['PPR2/1\n2.5mM', 'PPR2/1\n4mM', 'PPR3/1\n2.5mM', 'PPR3/1\n4mM']
palette2 = ['limegreen','orange','limegreen','orange']
sns.boxplot(data=pprDf_F0, showmeans=True, ax=ax11, palette=palette2, meanprops={'markerfacecolor':'red',
                                                                                'markeredgecolor':'black',
                                                                                'markersize':'8'})

for i in range(pprDf_F0.shape[0]):
    data1 = pprDf_F0.iloc[i,:].values
    ax11.scatter([0,1,2,3],[data1[0],data1[1],data1[2],data1[3]], color=[palette2[0],palette2[1],palette2[2],palette2[3]], edgecolors='k', s=20, alpha=0.6)
ax11.set_title('n_PPR2/1_2.5mM: {} ; n_PPR3/1_4mM: {}'.format(len(PPR_DF_F0_25_2_1),len(PPR_DF_F0_4_2_1)))




NORM_PEAK1_25, NORM_PEAK2_25, NORM_PEAK3_25 = [],[],[]
NORM_PEAK1_4, NORM_PEAK2_4, NORM_PEAK3_4 = [],[],[]

for peak in range(len(PEAK1_DF_F0_2_5)):
    norm_peak1_25 = PEAK1_DF_F0_2_5[peak]/FMAX_DF_F0_2_5[peak]
    norm_peak2_25 = PEAK2_DF_F0_2_5[peak]/FMAX_DF_F0_2_5[peak]
    norm_peak3_25 = PEAK3_DF_F0_2_5[peak]/FMAX_DF_F0_2_5[peak]
    NORM_PEAK1_25.append(norm_peak1_25)
    NORM_PEAK2_25.append(norm_peak2_25)
    NORM_PEAK3_25.append(norm_peak3_25)
    
for peak in range(len(PEAK1_DF_F0_4)):
    norm_peak1_4 = PEAK1_DF_F0_4[peak]/FMAX_DF_F0_4[peak]
    norm_peak2_4 = PEAK2_DF_F0_4[peak]/FMAX_DF_F0_4[peak]
    norm_peak3_4 = PEAK3_DF_F0_4[peak]/FMAX_DF_F0_4[peak]
    NORM_PEAK1_4.append(norm_peak1_4)
    NORM_PEAK2_4.append(norm_peak2_4)
    NORM_PEAK3_4.append(norm_peak3_4)


mean_peak1_25 = np.mean(NORM_PEAK1_25)
mean_peak2_25 = np.mean(NORM_PEAK2_25)
mean_peak3_25 = np.mean(NORM_PEAK3_25)

mean_peak1_4 = np.mean(NORM_PEAK1_4)
mean_peak2_4 = np.mean(NORM_PEAK2_4)
mean_peak3_4 = np.mean(NORM_PEAK3_4)

std_peak1_25 = np.std(NORM_PEAK1_25)
std_peak2_25 = np.std(NORM_PEAK2_25)
std_peak3_25 = np.std(NORM_PEAK3_25)

std_peak1_4 = np.std(NORM_PEAK1_4)
std_peak2_4 = np.std(NORM_PEAK2_4)
std_peak3_4 = np.std(NORM_PEAK3_4)

x = np.linspace(-5,5,1000,endpoint=True)
sigmoid_curve = sigmoid(x)

x_value_peak1_25 = find_x(mean_peak1_25)
x_value_peak2_25 = find_x(mean_peak2_25)
x_value_peak3_25 = find_x(mean_peak3_25)

x_value_peak1_4 = find_x(mean_peak1_4)
x_value_peak2_4 = find_x(mean_peak2_4)
x_value_peak3_4 = find_x(mean_peak3_4)

plt.figure()
plt.plot(x,sigmoid_curve,'k')
plt.scatter([x_value_peak1_25, x_value_peak2_25, x_value_peak3_25], [mean_peak1_25,mean_peak2_25,mean_peak3_25], color='b', alpha=0.5)
plt.errorbar([x_value_peak1_25, x_value_peak2_25, x_value_peak3_25], [mean_peak1_25,mean_peak2_25,mean_peak3_25],
             yerr=[std_peak1_25,std_peak2_25,std_peak3_25],fmt='o',capsize=4,color='b')

plt.scatter([x_value_peak1_4, x_value_peak2_4, x_value_peak3_4], [mean_peak1_4,mean_peak2_4,mean_peak3_4], color='r', alpha=0.5)
plt.errorbar([x_value_peak1_4, x_value_peak2_4, x_value_peak3_4], [mean_peak1_4,mean_peak2_4,mean_peak3_4],
             yerr=[std_peak1_4,std_peak2_4,std_peak3_4],fmt='o',capsize=4,color='r')

