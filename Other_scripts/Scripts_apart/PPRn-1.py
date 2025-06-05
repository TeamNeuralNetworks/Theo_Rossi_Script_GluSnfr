# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 10:12:01 2019

@author: Theo.ROSSI
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import seaborn as sns
import scipy.stats as stat
import pingouin as pg
from statannot import add_stat_annotation

#################################SETTINGS######################################
###############################################################################

Path = r'E:\AAVDJ.GluSnFR-S72A_Amp_Tau'
Path2 = r'E:\AAVDJ.GluSnFR-S72A\Profiles_extracted_stimML'
Freq = '20Hz'

###############################################################################
###############################################################################


files = sorted(os.listdir('{}'.format(Path)))

color = ['skyblue', 'limegreen']

Ca1_5_FILES, Ca2_5_FILES, Ca4_FILES = [], [], []
RAW_CA1_5, RAW_CA2_5, RAW_CA4 = [], [], []
NORM_CA1_5, NORM_CA2_5, NORM_CA4 = [], [], []

TAU = []
REAL_TAU =[]
REAL_TAUcA = []
TAU1_5, TAU2_5, TAU4 = [], [], []

def load_xls(file_xls):
    RAW_PEAK1, PPR2_1 = [],[]

    df=pd.read_excel (file_xls, header = 0)
    
    raw_profiles = df.iloc[0,1:].values
    RAW_PEAK1.append(raw_profiles[0])
    normalized_profil = raw_profiles[:]/raw_profiles[0]
    PPR2_1.append(normalized_profil[1])
    

    tau = df.iloc[0,1:].values

    # average_peak1 = np.nanmean(RAW_PEAK1)
    # sem_peak1 = np.nanstd(RAW_PEAK1)
    
    # average_ppr2_1 = np.nanmean(PPR2_1)
    # sem_ppr2_1 = np.nanstd(PPR2_1)

    return raw_profiles, normalized_profil, RAW_PEAK1, PPR2_1, tau

    
x = np.arange(0,10, step=1)
fig, ax = plt.subplots(2,3, figsize=(13,5), tight_layout=True)
fig2, ax2, = plt.subplots(1,2, tight_layout=True)
fig3, ax3 = plt.subplots(1,4, figsize=(15,5), tight_layout=True)

for file in range(len(files)):
    if Freq in files[file]:
        if 'Amp' in files[file]:
            if '1.5mMCa' in files[file]:
                Ca1_5_FILES.append(files[file])
                raw1, norm1, peak1, ppr2_1, tau = load_xls('{}/{}'.format(Path, files[file]))
                RAW_CA1_5.append(peak1)
                NORM_CA1_5.append(ppr2_1)
                
                ax[0,0].plot(raw1, 'skyblue', alpha=0.5), ax[0,0].grid(axis='y', linestyle='--')
                ax[0,0].scatter(x, raw1, color='skyblue', alpha=0.5)
                ax[1,0].plot(norm1, 'skyblue', alpha=0.5), ax[1,0].grid(axis='y', linestyle='--')
                ax[1,0].scatter(x, norm1, color='skyblue', alpha=0.5)
                
                ax[0,0].set_title('Ca: 1.5mM', color='skyblue'), ax[0,0].set_ylabel('dF/F0')
                ax[1,0].set_ylabel('Peakn/Peak1')
                
                
            elif '2.5mMCa' in files[file]:
                Ca2_5_FILES.append(files[file])
                raw2, norm2, peak1, ppr2_1, tau = load_xls('{}/{}'.format(Path, files[file]))
                RAW_CA2_5.append(peak1)
                NORM_CA2_5.append(ppr2_1)
    
                ax[0,1].plot(raw2, 'limegreen', alpha=0.5), ax[0,1].grid(axis='y', linestyle='--')
                ax[0,1].scatter(x, raw2, color='limegreen', alpha=0.5)
                ax[1,1].plot(norm2, 'limegreen', alpha=0.5), ax[1,1].grid(axis='y', linestyle='--')
                ax[1,1].scatter(x, norm2, color='limegreen', alpha=0.5)
                
                ax[0,1].set_title('Ca: 2.5mM', color='limegreen')
                
                
            elif '4mMCa' in files[file]:
                Ca4_FILES.append(files[file])
                raw3, norm3, peak1, ppr2_1, tau = load_xls('{}/{}'.format(Path, files[file]))
                RAW_CA4.append(peak1)
                NORM_CA4.append(ppr2_1)
                
                ax[0,2].plot(raw3, 'orange', alpha=0.5), ax[0,2].grid(axis='y', linestyle='--')
                ax[0,2].scatter(x, raw3, color='orange', alpha=0.5)
                ax[1,2].plot(norm3, 'orange', alpha=0.5), ax[1,2].grid(axis='y', linestyle='--')
                ax[1,2].scatter(x, norm3, color='orange', alpha=0.5)
    
                ax[0,2].set_title('Ca: 4mM', color='orange')
            
        if 'Tau' in files[file]:
            raw, norm, peak1, ppr2_1, tau = load_xls('{}/{}'.format(Path, files[file]))
            TAU.append(tau*1000)
            if '1.5mM' in files[file]:
                TAU1_5.append(tau*1000)
            if '2.5mM' in files[file]:
                TAU2_5.append(tau*1000)
            if '4mM' in files[file]:
                TAU4.append(tau*1000)

'''
to = np.ravel(TAU).tolist()

for value in to:
    if value <= 1000:
        REAL_TAU.append(value)
mean_tau = np.nanmean(REAL_TAU)
std_tau = np.nanstd(REAL_TAU)
ax3[0].hist(REAL_TAU, bins=80, facecolor='pink', alpha=0.8)
ax3[0].set_xlabel('Time(ms)'), ax3[0].set_ylabel('Count')
ax3[0].set_title('Tau\n µ: {:.2f}+/-{:.2f}'.format(mean_tau, std_tau))

TAUcA1_5, TAUcA2_5, TAUcA4 = [], [], []

tauCa = {'1.5mM': np.ravel(TAU1_5).tolist(),
         '2.5mM': np.ravel(TAU2_5).tolist(),
         '4mM': np.ravel(TAU4).tolist()}


for ca1_5, ca2_5, ca4 in zip(tauCa['1.5mM'], tauCa['2.5mM'], tauCa['4mM']):
    if ca1_5 <= 100:
        TAUcA1_5.append(ca1_5)
    if ca2_5 <= 100:
        TAUcA2_5.append(ca2_5)
    if ca4 <= 100:
        TAUcA4.append(ca4)


ax3[1].hist(TAUcA1_5, bins=40, facecolor='blue', alpha=0.8)
ax3[2].hist(TAUcA2_5, bins=40, facecolor='orange', alpha=0.8)
ax3[3].hist(TAUcA4, bins=40, facecolor='green', alpha=0.8)
'''

y_raw = pd.concat((pd.DataFrame(RAW_CA1_5),pd.DataFrame(RAW_CA2_5)),axis=1)
y_norm = pd.concat((pd.DataFrame(NORM_CA1_5),pd.DataFrame(NORM_CA2_5)),axis=1)
y_raw.columns = ['1.5mM', '2.5mM']
y_norm.columns = ['1.5mM', '2.5mM']


sns.boxplot(data=y_raw, ax=ax2[0], showmeans=True, palette=color, meanprops={'markerfacecolor':'red',
                                                                                  'markeredgecolor':'black',
                                                                                  'markersize':'8'})
sns.boxplot(data=y_norm, ax=ax2[1], showmeans=True, palette=color, meanprops={'markerfacecolor':'red',
                                                                                  'markeredgecolor':'black',
                                                                                  'markersize':'8'})

sns.swarmplot(data=y_raw, ax=ax2[0], color='0.25')
sns.swarmplot(data=y_norm, ax=ax2[1], color='0.25')


'''
average_files = sorted(os.listdir('{}'.format(Path2)))
plt.figure()

for file in range(len(average_files)):
    df = pd.read_excel('{}/{}'.format(Path2, average_files[file]))
    
    for column in range(len(df.columns)):
        if 'Average' == df.columns[column]:
            average=df.iloc[:,column].values
'''

# for i in range(len(df_raw.columns)):

#     raw_shapiro_test = pg.normality(df_raw)
#     norm_shapiro_test = pg.normality(df_norm)

    
#     for val in raw_shapiro_test['pval']:
        # if val <= 0.05:
            # KW_test = pg.kruskal(data=df_raw, dv=df_raw.columns[0], between=df_raw.columns[0])
            # print(KW_test)
            # ax2, MW_test = add_stat_annotation(ax2, data=df_raw,
            #                                    box_pairs=[('1.5mM','2.5mM'), ('1.5mM','4mM'), ('2.5mM','4mM')],
            #                                    test='Mann-Whitney',
            #                                    comparisons_correction='bonferroni',
            #                                    text_format='star',
            #                                    loc='inside')
    
#         if val > 0.05:
#             Anova = stat.f_oneway(df_raw.iloc[:,0].values, df_raw.iloc[:,1].values, df_raw.iloc[:,2].values)
#             ax2, T_test = add_stat_annotation(ax2, data=df_raw,
#                                               box_pairs=[('1.5mM','2.5mM'), ('1.5mM','4mM'), ('1.5mM','2.5mM')],
#                                               test='t-test_ind',
#                                               comparisons_correction='bonferroni',
#                                               text_format='star', loc='inside')
    
    # for val2 in norm_shapiro_test['pval']:
        # if norm_shapiro_test[1] <= 0.05:
        #     KW_test = stat.kruskal(y_norm[0], y_norm[1], y_norm[2])
        #     ax3, MW_test = add_stat_annotation(ax3, data=y_norm,
        #                                        box_pairs=[(y_norm[0],y_norm[1]), (y_norm[0],y_norm[2]), (y_norm[1],y_norm[2])],
        #                                        test='Mann-Whitney',
        #                                        comparisons_correction='bonferroni',
        #                                        text_format='star',
        #                                        loc='inside')
            
        # il val2 > 0.05:
        #     Anova = stat.f_oneway(y_norm[0], y_norm[1], y_norm[2])
        #     ax3, T_test = add_stat_annotation(ax3, data=y_norm,
        #                                       box_pairs=[(y_norm[0],y_norm[1]), (y_norm[0],y_norm[2]), (y_norm[1],y_norm[2])],
        #                                       test='t-test_ind',
        #                                       comparisons_correction='bonferroni',
        #                                       text_format='star', loc='inside')
    




#if Dataframe == True:
#    if Normalization == True:
#        df = pd.DataFrame(LIST_NORMALIZED_PROFILS)
#    else:
#        df = pd.DataFrame(LIST_RAW_PROFILES)
#        
#    print(df)
#    with pd.ExcelWriter('{}/{}/{}'.format(Path, Big_folder, Excel_normilized_amp)) as writer:
#        df.to_excel(writer, header=False)