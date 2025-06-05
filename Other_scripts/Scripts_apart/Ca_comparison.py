# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 16:21:42 2020

@author: Theo.ROSSI
"""

import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import pandas as pd
import numpy as np
import os
import seaborn as sns
from statannot import add_stat_annotation
from scipy import stats,signal,optimize
from sklearn.linear_model import LinearRegression
from scipy.signal import savgol_filter


#################################SETTINGS######################################
###############################################################################

Path = r'E:\AAV2.8.GluSnFR-S72A\Contrôle_Ca_1.5vs4mM_Paired'

Freq = '50Hz'
Peak1, Peak2 = 0,1
EP1 = False

###############################################################################
###############################################################################


def func_mono_exp(x, a, b, c, d):
    return a * np.exp(-(x-b)/c) + d


files = sorted(os.listdir('{}'.format(Path)))

FILES_20Hz_Ca15_Amp, FILES_20Hz_Ca25_Amp, FILES_20Hz_Ca4_Amp = [],[],[]
VALUES_20Hz_Ca15_Amp, VALUES_20Hz_Ca25_Amp, VALUES_20Hz_Ca4_Amp = [],[],[]

FILES_20Hz_Ca25_Amp_ep1, FILES_20Hz_Ca4_Amp_ep1 = [],[]
VALUES_20Hz_Ca25_Amp_ep1, VALUES_20Hz_Ca4_Amp_ep1 = [],[]

PEAK1_20Hz_Ca15_Amp, PEAK1_20Hz_Ca25_Amp, PEAK1_20Hz_Ca4_Amp = [],[],[]
PEAK2_20Hz_Ca15_Amp, PEAK2_20Hz_Ca25_Amp, PEAK2_20Hz_Ca4_Amp = [],[],[]

PEAK1_20Hz_Ca25_Amp_ep1, PEAK1_20Hz_Ca4_Amp_ep1 = [],[]
PEAK2_20Hz_Ca25_Amp_ep1, PEAK2_20Hz_Ca4_Amp_ep1 = [],[]

PPR2_1_20Hz_Ca15_Amp, PPR2_1_20Hz_Ca25_Amp, PPR2_1_20Hz_Ca4_Amp = [],[],[]
NORM_PROFILES_15, NORM_PROFILES_25, NORM_PROFILES_4 = [],[],[]

PPR2_1_20Hz_Ca25_Amp_ep1, PPR2_1_20Hz_Ca4_Amp_ep1 = [],[]
NORM_PROFILES_25_ep1, NORM_PROFILES_4_ep1 = [],[]

EP1_25, EP1_4, TIME_EP1_25, TIME_EP1_4 = [],[],[],[]
AVERAGE_15, AVERAGE_25, AVERAGE_4, TIME_15, TIME_25, TIME_4 = [],[],[],[],[],[]
TAU_15, TAU_25, TAU_4 = [],[],[]
AMP_15, AMP_4 = [],[]

        
for file in range(len(files)):
    if Freq in files[file]:
        if 'dF_F0_Profile.xlsx' in files[file]:
            data = pd.read_excel('{}/{}'.format(Path, files[file]), header=0)
            for i in range(len(data.columns)):
                ep1 = data.iloc[:,0].values
                
                if 'Time' == data.columns[i]:
                    time = data.iloc[:,i].values
                elif 'Average' == data.columns[i]:
                    average = data.iloc[:,i].values
                    
            if '1.5mM' in files[file]:
                AVERAGE_15.append(average)
                TIME_15.append(time)
            elif '2.5mM' in files[file]:
                AVERAGE_25.append(average)
                TIME_25.append(time)
                if 'Profile_ep1' in files[file]:
                    EP1_25.append(ep1)
                    TIME_EP1_25.append(time)
                
            else:
                AVERAGE_4.append(average)
                TIME_4.append(time)
                if 'Profile_ep1' in files[file]:
                    EP1_4.append(ep1)
                    TIME_EP1_4.append(time)
                
                
        if 'dF_F0_Amp.xlsx' in files[file]:
            data = pd.read_excel('{}/{}'.format(Path, files[file]), header=None, index=None)
            values = data.iloc[1,1:].values
            
            NORM = []
            for i in range(values.shape[0]):
                norm = values[i]/values[0]
                NORM.append(norm)
                
            if '1.5mM' in files[file]:
                FILES_20Hz_Ca15_Amp.append(files[file])
                VALUES_20Hz_Ca15_Amp.append(values)
                PEAK1_20Hz_Ca15_Amp.append(values[0])
                PEAK2_20Hz_Ca15_Amp.append(values[1])
                PPR2_1_20Hz_Ca15_Amp.append(values[Peak2]/values[Peak1])
                NORM_PROFILES_15.append(NORM)
                
            elif '2.5mM' in files[file]:
                FILES_20Hz_Ca25_Amp.append(files[file])
                VALUES_20Hz_Ca25_Amp.append(values)
                PEAK1_20Hz_Ca25_Amp.append(values[0])
                PEAK2_20Hz_Ca25_Amp.append(values[1])
                PPR2_1_20Hz_Ca25_Amp.append(values[Peak2]/values[Peak1])
                NORM_PROFILES_25.append(NORM)
                
            else:
                FILES_20Hz_Ca4_Amp.append(files[file])
                VALUES_20Hz_Ca4_Amp.append(values)
                PEAK1_20Hz_Ca4_Amp.append(values[0])
                PEAK2_20Hz_Ca4_Amp.append(values[1])
                PPR2_1_20Hz_Ca4_Amp.append(values[Peak2]/values[Peak1])
                NORM_PROFILES_4.append(NORM)
        
        if 'dF_F0_Amp_ep1.xlsx' in files[file]:
            data = pd.read_excel('{}/{}'.format(Path, files[file]), header=None, index=None)
            values = data.iloc[1,1:].values
            
            NORM = []
            for i in range(values.shape[0]):
                norm = values[i]/values[0]
                NORM.append(norm)
                
            if '2.5mM' in files[file]:
                FILES_20Hz_Ca25_Amp_ep1.append(files[file])
                VALUES_20Hz_Ca25_Amp_ep1.append(values)
                PEAK1_20Hz_Ca25_Amp_ep1.append(values[0])
                PEAK2_20Hz_Ca25_Amp_ep1.append(values[1])
                PPR2_1_20Hz_Ca25_Amp_ep1.append(values[Peak2]/values[Peak1])
                NORM_PROFILES_25_ep1.append(NORM)
                
            elif '4mM' in files[file]:
                FILES_20Hz_Ca4_Amp_ep1.append(files[file])
                VALUES_20Hz_Ca4_Amp_ep1.append(values)
                PEAK1_20Hz_Ca4_Amp_ep1.append(values[0])
                PEAK2_20Hz_Ca4_Amp_ep1.append(values[1])
                PPR2_1_20Hz_Ca4_Amp_ep1.append(values[Peak2]/values[Peak1])
                NORM_PROFILES_4_ep1.append(NORM)

        if 'dF_F0_Tau.xlsx' in files[file]:
            tau_data = pd.read_excel('{}/{}'.format(Path, files[file]), header=None, Index=None)
            tau_values = tau_data.iloc[1,1:].values*1000
            y = tau_values.reshape(-1,1)
            x = np.arange(1,len(tau_values)+1,1).reshape(-1,1)
            regr = LinearRegression().fit(x,y)
            prediction = regr.predict(x)
            DetCoeff = regr.score(x,y)
            
            if tau_values[0] >= 100:
                continue
            if '1.5mM' in files[file]:
                TAU_15.append(tau_values[0])
            elif '2.5mM' in files[file]:
                TAU_25.append(tau_values[0])
            else:
                TAU_4.append(tau_values[0])
                    
            # plt.figure()
            # if '1.5mM' in files[file]:
            #     plt.scatter(x,tau_values,color='skyblue')
            # elif '2.5mM' in files[file]:
            #     plt.scatter(x,tau_values,color='limegreen')
            # else:
            #     plt.scatter(x,tau_values,color='orange')
            # plt.plot(x,prediction,'k')
            
            # for i in range(tau_values.shape[0]):
            #     if tau_values[i] >= 100:
            #         continue
            #     if '1.5mM' in files[file]:
            #         TAU_15.append(tau_values[i])
            #     elif '2.5mM' in files[file]:
            #         TAU_25.append(tau_values[i])
            #     else:
            #         TAU_4.append(tau_values[i])

if EP1 == True:
    NEW_EP1_25, NEW_EP1_4 = [],[]
    fig_ep1, ax_ep1 = plt.subplots(2,1, figsize=(9,8), tight_layout=True)
    
    for ep1_25 in range(len(EP1_25)):    
        if '20190801' in FILES_20Hz_Ca25_Amp_ep1[ep1_25]:
            start25 = np.ravel(np.where(TIME_EP1_25[ep1_25] <= 0.95))[-1]
            end25 = np.ravel(np.where(TIME_EP1_25[ep1_25] <= 1.5))[-1]
            # continue        
        else:
            start25 = np.ravel(np.where(TIME_EP1_25[ep1_25] <= 0.45))[-1]
            end25 = np.ravel(np.where(TIME_EP1_25[ep1_25] <= 1.0))[-1]
            
        NewTrace_ep1_25 = EP1_25[ep1_25][start25:end25]
        resampling_ep1_25 = signal.resample(NewTrace_ep1_25, 505)
        NEW_EP1_25.append(resampling_ep1_25)
            
        start_ep1_25 = np.ravel(np.where(TIME_EP1_25[3] <= 0.45))[-1]
        end_ep1_average25 = np.ravel(np.where(TIME_EP1_25[3] <= 1.0))[-1]
        x_ep1_25 = np.linspace(TIME_EP1_25[3][start_ep1_25], TIME_EP1_25[3][end_ep1_average25], num=505, endpoint=True)
        
        ax_ep1[0].plot(x_ep1_25, resampling_ep1_25, 'k', alpha=0.2)
        
    for ep1_4 in range(len(TIME_EP1_4)):
        start4 = np.ravel(np.where(TIME_EP1_4[ep1_4] <= 0.45))[-1]
        end4 = np.ravel(np.where(TIME_EP1_4[ep1_4] <= 1.0))[-1]
        NewTrace_ep1_4 = EP1_4[ep1_4][start4:end4]
        resampling_ep1_4 = signal.resample(NewTrace_ep1_4, 489)
        NEW_EP1_4.append(resampling_ep1_4)
        
        start_ep1_4 = np.ravel(np.where(TIME_EP1_4[0] <= 0.45))[-1]
        end_ep1_4 = np.ravel(np.where(TIME_EP1_4[0] <= 1.0))[-1]
        x_ep1_4 = np.linspace(TIME_EP1_4[0][start_ep1_4], TIME_EP1_4[0][end_ep1_4], num=489, endpoint=True)
    
        ax_ep1[1].plot(x_ep1_4, resampling_ep1_4, 'k', alpha=0.2)
    
    
    avg_ep1_25 = np.average(NEW_EP1_25, axis=0)
    avg_ep1_4 = np.average(NEW_EP1_4, axis=0)
    ax_ep1[0].plot(x_ep1_25, avg_ep1_25, 'limegreen')
    ax_ep1[1].plot(x_ep1_4, avg_ep1_4, 'orange')
    ax_ep1[0].set_title('Profiles EP1')
    
    plt.figure()
    plt.plot(x_ep1_25, avg_ep1_25, 'limegreen')
    plt.plot(x_ep1_4, avg_ep1_4,'orange')
    
    
    fig_box_ep1, ax_box_ep1 = plt.subplots(1,2, figsize=(8,7), tight_layout=True)
    palette_ep1 = ['limegreen','orange','limegreen','orange']
    ep1Df = pd.concat((pd.DataFrame(PEAK1_20Hz_Ca25_Amp_ep1), pd.DataFrame(PEAK1_20Hz_Ca4_Amp_ep1),
                       pd.DataFrame(PEAK2_20Hz_Ca25_Amp_ep1), pd.DataFrame(PEAK2_20Hz_Ca4_Amp_ep1)), axis=1)
    ep1Df.columns=['P1\n2.5mM','P1\n4mM','P2\n2.5mM','P2\n4mM']
    sns.boxplot(data=ep1Df, showmeans=True, ax=ax_box_ep1[0], palette=palette_ep1, meanprops={'markerfacecolor':'red',
                                                                                           'markeredgecolor':'black',
                                                                                           'markersize':'8'})
    add_stat_annotation(ax_box_ep1[0], data=ep1Df, box_pairs=[('P1\n2.5mM','P1\n4mM'), ('P2\n2.5mM','P2\n4mM')],
                        test='Mann-Whitney', text_format='star', verbose=2)
    
    palette_ep1_2 = ['limegreen','orange']
    ep1PprDf = pd.concat((pd.DataFrame(PPR2_1_20Hz_Ca25_Amp_ep1), pd.DataFrame(PPR2_1_20Hz_Ca4_Amp_ep1)), axis=1)
    ep1PprDf.columns=['2.5mM','4mM']
    sns.boxplot(data=ep1PprDf, showmeans=True, ax=ax_box_ep1[1], palette=palette_ep1_2, meanprops={'markerfacecolor':'red',
                                                                                                   'markeredgecolor':'black',
                                                                                                   'markersize':'8'})
    add_stat_annotation(ax_box_ep1[1], data=ep1PprDf, box_pairs=[('2.5mM','4mM')], test='Mann-Whitney', text_format='star', verbose=2)
    
    ax_box_ep1[0].set_title('n_2.5mM: {}\nn_4mM: {}'.format(len(PEAK1_20Hz_Ca25_Amp_ep1),
                                                     len(PEAK1_20Hz_Ca4_Amp_ep1)))
    
    for i in range(ep1Df.shape[0]):
        data1 = ep1Df.iloc[i,:].values
        data2 = ep1PprDf.iloc[i,:].values
        ax_box_ep1[0].scatter([0,1,2,3],[data1[0],data1[1],data1[2],data1[3]], color=[palette_ep1[0],palette_ep1[1],palette_ep1[2],palette_ep1[3]], edgecolors='k', s=20, alpha=0.6)
        ax_box_ep1[1].scatter([0,1],[data2[0],data2[1]], color=[palette_ep1_2[0],palette_ep1_2[1]], edgecolors='k', s=20, alpha=0.6)
    
    fig_histo_ep1, ax_histo_ep1 = plt.subplots(2,2, figsize=(10,6), tight_layout=True)
    sns.distplot(PEAK1_20Hz_Ca25_Amp_ep1, bins=10, ax=ax_histo_ep1[0,0], color=palette_ep1_2[0])
    sns.distplot(PEAK2_20Hz_Ca25_Amp_ep1, bins=10, ax=ax_histo_ep1[0,1], color=palette_ep1_2[0])
    sns.distplot(PEAK1_20Hz_Ca4_Amp_ep1, bins=10, ax=ax_histo_ep1[1,0], color=palette_ep1_2[1])
    sns.distplot(PEAK2_20Hz_Ca4_Amp_ep1, bins=10, ax=ax_histo_ep1[1,1], color=palette_ep1_2[1])



NEW_AVERAGE_15, NEW_AVERAGE_25, NEW_AVERAGE_4 = [],[],[]
# INTERPOL_15, INTERPOL_25, INTERPOL_4 = [],[],[]
fig_average, ax_average = plt.subplots(3,1, figsize=(11,10), tight_layout=True)
if AVERAGE_15 != []:
    for trace15 in range(len(AVERAGE_15)):
        start15 = np.ravel(np.where(TIME_15[trace15] <= 0.45))[-1]
        end15 = np.ravel(np.where(TIME_15[trace15] <= 1.0))[-1]
        
        NewTrace_15 = AVERAGE_15[trace15][start15:end15]
        resampling15 = signal.resample(NewTrace_15, 407)
        NEW_AVERAGE_15.append(resampling15)
        
        # interpolation15 = interp1d(TIME_15[trace15][start15:end15], NewTrace_15, fill_value='extrapolate')
        
        start_average15 = np.ravel(np.where(TIME_15[0] <= 0.45))[-1]
        end_average15 = np.ravel(np.where(TIME_15[0] <= 1.0))[-1]
        x_average15 = np.linspace(TIME_15[0][start_average15], TIME_15[0][end_average15], num=407, endpoint=True)
        # int15 = interpolation15(x_average15)
        # INTERPOL_15.append(int15)
        
        ax_average[0].plot(x_average15, savgol_filter(resampling15, 7,2), alpha=0.2)
    
        # fig, ax = plt.subplots(figsize=(10,5), tight_layout=True)   
        # ax.plot(TIME_15[trace15], AVERAGE_15[trace15], 'skyblue')
        # ax.set_title('{}'.format(FILES_20Hz_Ca15_Amp[trace15]))
        # ax.set_ylabel('DF/F0'), ax.set_xlabel('Time (s)')
        # fig.savefig('{}.png'.format(FILES_20Hz_Ca15_Amp[trace15]))
    
    plt.figure()
    avg15 = np.average(NEW_AVERAGE_15, axis=0)
    ax_average[0].plot(x_average15, avg15, 'skyblue')
    plt.plot(x_average15, savgol_filter(avg15, 7, 2),'skyblue')

TAU_TAIL = []
if AVERAGE_25 != []:
    for trace25 in range(len(AVERAGE_25)):
        # param_bounds=([-np.inf,0.,0.,0.],[np.inf,1.,10.,1000.]) 
        # start = np.ravel(np.where(TIME_25[trace25] >= 1.03))[0]
        # end = np.ravel(np.where(TIME_25[trace25] <= TIME_25[trace25][-1]))[-1]
        # y = AVERAGE_25[trace25][start:end]
        # popt, pcov = optimize.curve_fit(func_mono_exp,TIME_25[trace25][start:end],y,bounds=param_bounds,maxfev=10000)
        # TAU_TAIL.append(popt[2]*1000)
        # curve = func_mono_exp(TIME_25[trace25][start:end],*popt)
        # plt.figure()
        # plt.plot(TIME_25[trace25],AVERAGE_25[trace25],'k')
        # plt.plot(TIME_25[trace25][start:end],curve,'r')
        # plt.title('{}\nTau = {:.2f} ms'.format(FILES_20Hz_Ca25_Amp[trace25],popt[2]*1000))
        # print('{}:\nTau = {:.2f} ms'.format(FILES_20Hz_Ca25_Amp[trace25],popt[2]*1000))
        # print('-------------')
        
        

        if '20190801' in FILES_20Hz_Ca25_Amp[trace25]:
            start25 = np.ravel(np.where(TIME_25[trace25] <= 0.95))[-1]
            end25 = np.ravel(np.where(TIME_25[trace25] <= 1.5))[-1]
            # continue        
        else:
            start25 = np.ravel(np.where(TIME_25[trace25] <= 0.45))[-1]
            end25 = np.ravel(np.where(TIME_25[trace25] <= 1.0))[-1]
            
        NewTrace_25 = AVERAGE_25[trace25][start25:end25]
        resampling25 = signal.resample(NewTrace_25, 505)
        NEW_AVERAGE_25.append(resampling25)
        # interpolation25 = interp1d(TIME_25[trace25][start25:end25], NewTrace, fill_value='extrapolate')
            
        start_average25 = np.ravel(np.where(TIME_25[3] <= 0.45))[-1]
        end_average25 = np.ravel(np.where(TIME_25[3] <= 1.0))[-1]
        x_average25 = np.linspace(TIME_25[3][start_average25], TIME_25[3][end_average25], num=505, endpoint=True)
        # int25 = interpolation25(x_average25)
        # INTERPOL_25.append(int25)
        
        ax_average[1].plot(x_average25, savgol_filter(resampling25, 7, 2), 'k', alpha=0.2)
        # figbis, axbis = plt.subplots(figsize=(10,5), tight_layout=True)   
        # axbis.plot(TIME_25[trace25], AVERAGE_25[trace25], 'limegreen')
        # axbis.set_title('{}'.format(FILES_20Hz_Ca25_Amp[trace25]))
        # axbis.set_ylabel('DF/F0'), axbis.set_xlabel('Time (s)')
        # figbis.savefig('{}.png'.format(FILES_20Hz_Ca25_Amp[trace25]))
        
    avg25 = np.average(NEW_AVERAGE_25, axis=0)
    ax_average[1].plot(x_average25, avg25, 'limegreen')
    plt.plot(x_average25, savgol_filter(avg25, 7,2),'limegreen')

if AVERAGE_4 != []:
    for trace4 in range(len(AVERAGE_4)):
        start4 = np.ravel(np.where(TIME_4[trace4] <= 0.45))[-1]
        end4 = np.ravel(np.where(TIME_4[trace4] <= 1.0))[-1]
        NewTrace_4 = AVERAGE_4[trace4][start4:end4]
        resampling4 = signal.resample(NewTrace_4, 489)
        NEW_AVERAGE_4.append(resampling4)
        # print(len(NewTrace_4))
        # interpolation4 = interp1d(TIME_4[trace4][start4:end4], NewTrace_4, fill_value='extrapolate')
        
        start_average4 = np.ravel(np.where(TIME_4[0] <= 0.45))[-1]
        end_average4 = np.ravel(np.where(TIME_4[0] <= 1.0))[-1]
        x_average4 = np.linspace(TIME_4[0][start_average4], TIME_4[0][end_average4], num=489, endpoint=True)
        # int4 = interpolation4(x_average4)
        # INTERPOL_4.append(int4)
        
        ax_average[2].plot(x_average4, savgol_filter(resampling4, 7, 2), alpha=0.2)
        
        # figbisbis, axbisbis = plt.subplots(figsize=(10,5), tight_layout=True) 
        # axbisbis.plot(TIME_4[trace4], AVERAGE_4[trace4], 'orange')
        # axbisbis.set_title('{}'.format(FILES_20Hz_Ca4_Amp[trace4]))
        # axbisbis.set_ylabel('DF/F0'), axbisbis.set_xlabel('Time (s)')
        # figbisbis.savefig('{}.png'.format(FILES_20Hz_Ca4_Amp[trace4]))
    
    avg4 = np.average(NEW_AVERAGE_4, axis=0)
    ax_average[2].plot(x_average4, avg4, 'orange')
    plt.plot(x_average4, savgol_filter(avg4, 7, 2),'orange')
    # plt.savefig('Profiles_amps.pdf')
    # fig_average.savefig('Profiles_traces.pdf')
ax_average[0].set_ylim(-0.5, 5)
ax_average[0].set_ylim(-0.5, 5)



fig1, ax1 = plt.subplots(1,2, figsize=(8,7), tight_layout=True)


if 'Paired' in Path:
    palette = ['skyblue','orange','skyblue','orange']
    globalDf = pd.concat((pd.DataFrame(PEAK1_20Hz_Ca15_Amp), pd.DataFrame(PEAK1_20Hz_Ca4_Amp),
                          pd.DataFrame(PEAK2_20Hz_Ca15_Amp), pd.DataFrame(PEAK2_20Hz_Ca4_Amp)), axis=1)
    globalDf.columns=['P1\n1.5mM','P1\n4mM','P2\n1.5mM','P2\n4mM']
    sns.boxplot(data=globalDf, showmeans=True, ax=ax1[0], palette=palette, meanprops={'markerfacecolor':'red',
                                                                                      'markeredgecolor':'black',
                                                                                      'markersize':'8'})

    
    palette2 = ['skyblue','orange']
    globalDf2 = pd.concat((pd.DataFrame(PPR2_1_20Hz_Ca15_Amp), pd.DataFrame(PPR2_1_20Hz_Ca4_Amp)), axis=1)
    globalDf2.columns=['1.5mM','4mM']
    sns.boxplot(data=globalDf2, showmeans=True, ax=ax1[1], palette=palette2, meanprops={'markerfacecolor':'red',
                                                                                      'markeredgecolor':'black',
                                                                                      'markersize':'8'})

    
    
    add_stat_annotation(ax1[0], data=globalDf, box_pairs=[('P1\n1.5mM','P1\n4mM'),
                                                          ('P2\n1.5mM','P2\n4mM')],
                        test='Wilcoxon', text_format='star', verbose=2)
    add_stat_annotation(ax1[1], data=globalDf2, box_pairs=[('1.5mM','4mM')],
                        test='Wilcoxon', text_format='star', verbose=2)
else:
    
    palette = ['skyblue','limegreen','orange','skyblue','limegreen','orange']
    globalDf = pd.concat((pd.DataFrame(PEAK1_20Hz_Ca15_Amp), pd.DataFrame(PEAK1_20Hz_Ca25_Amp), pd.DataFrame(PEAK1_20Hz_Ca4_Amp),
                          pd.DataFrame(PEAK2_20Hz_Ca15_Amp), pd.DataFrame(PEAK2_20Hz_Ca25_Amp), pd.DataFrame(PEAK2_20Hz_Ca4_Amp)), axis=1)
    globalDf.columns=['P1\n1.5mM','P1\n2.5mM','P1\n4mM','P2\n1.5mM','P2\n2.5mM','P2\n4mM']
    
    
    palette2 = ['skyblue','limegreen','orange']
    globalDf2 = pd.concat((pd.DataFrame(PPR2_1_20Hz_Ca15_Amp), pd.DataFrame(PPR2_1_20Hz_Ca25_Amp), pd.DataFrame(PPR2_1_20Hz_Ca4_Amp)), axis=1)
    globalDf2.columns=['1.5mM','2.5mM','4mM']
    sns.boxplot(data=globalDf2, showmeans=True, ax=ax1[1], palette=palette2, meanprops={'markerfacecolor':'red',
                                                                                      'markeredgecolor':'black',
                                                                                      'markersize':'8'})

    
    add_stat_annotation(ax1[0], data=globalDf, box_pairs=[('P1\n1.5mM','P1\n2.5mM'), ('P1\n1.5mM','P1\n4mM'),
                                                          ('P2\n1.5mM','P2\n2.5mM'), ('P2\n1.5mM','P2\n4mM'),
                                                          ('P1\n2.5mM','P1\n4mM'), ('P2\n2.5mM','P2\n4mM')],
                        test='Mann-Whitney', text_format='star', verbose=2)
    
    add_stat_annotation(ax1[1], data=globalDf2, box_pairs=[('1.5mM','2.5mM'),('1.5mM','4mM'),('2.5mM','4mM')],
                        test='Mann-Whitney', text_format='star', verbose=2)


for i in range(globalDf.shape[0]):
    data1 = globalDf.iloc[i,:].values
    data2 = globalDf2.iloc[i,:].values
    # ax1[0].scatter([0,1,2],[data1[0],data1[1],data1[2]], color=[palette2[0],palette2[1],palette2[2]], edgecolors='k', s=20, alpha=0.6)
    # ax1[0].scatter([3,4,5],[data1[3],data1[4],data1[5]], color=[palette2[0],palette2[1],palette2[2]], edgecolors='k', s=20, alpha=0.6)
    # ax1[1].scatter([0,1,2],[data2[0],data2[1],data2[2]], color=[palette2[0],palette2[1],palette2[2]], edgecolors='k', s=20, alpha=0.6)
    ax1[0].scatter([0,2],[data1[0],data1[2]], color=[palette2[0],palette2[0]], edgecolors='k', s=20, alpha=0.6)
    ax1[0].scatter([1,3],[data1[1],data1[3]], color=[palette2[1],palette2[1]], edgecolors='k', s=20, alpha=0.6)
    ax1[1].scatter([0,1],[data2[0],data2[1]], color=[palette2[0],palette2[1]], edgecolors='k', s=20, alpha=0.6)
    
    
    if 'Paired' in Path:
        ax1[0].plot([0,1],[data1[0],data1[1]], 'k', alpha=0.2)
        ax1[0].plot([2,3],[data1[2],data1[3]], 'k', alpha=0.2)
        ax1[1].plot([0,1],[data2[0],data2[1]], 'k', alpha=0.2)

ax1[0].set_title('n_1.5mM: {}\nn_2.5mM: {}\nn_4mM: {}'.format(len(PEAK1_20Hz_Ca15_Amp),len(PEAK1_20Hz_Ca25_Amp),len(PEAK1_20Hz_Ca4_Amp)))
ax1[1].set_title('n_1.5mM: {}\nn_2.5mM: {}\nn_4mM: {}'.format(len(PEAK2_20Hz_Ca15_Amp),len(PEAK2_20Hz_Ca25_Amp),len(PEAK2_20Hz_Ca4_Amp)))
ax1[0].set_ylabel('DF/F0'), ax1[1].set_ylabel('PPR2/1')
ax1[1].legend()
# fig1.savefig('Boxplot_Ca_comp.pdf')


fig_CV, ax_CV = plt.subplots(1,2, figsize=(8,7), tight_layout=True)
CV_peak1_15 = np.nanstd(PEAK1_20Hz_Ca15_Amp)/np.nanmean(PEAK1_20Hz_Ca15_Amp)
CV_peak1_25 = np.nanstd(PEAK1_20Hz_Ca25_Amp)/np.nanmean(PEAK1_20Hz_Ca25_Amp)
CV_peak1_4 = np.nanstd(PEAK1_20Hz_Ca4_Amp)/np.nanmean(PEAK1_20Hz_Ca4_Amp)
CV_peak2_15 = np.nanstd(PEAK2_20Hz_Ca15_Amp)/np.nanmean(PEAK2_20Hz_Ca15_Amp)
CV_peak2_25 = np.nanstd(PEAK2_20Hz_Ca25_Amp)/np.nanmean(PEAK2_20Hz_Ca25_Amp)
CV_peak2_4 = np.nanstd(PEAK2_20Hz_Ca4_Amp)/np.nanmean(PEAK2_20Hz_Ca4_Amp)
CV_PPR_15 = np.nanstd(PPR2_1_20Hz_Ca15_Amp)/np.nanmean(PPR2_1_20Hz_Ca15_Amp)
CV_PPR_25 = np.nanstd(PPR2_1_20Hz_Ca25_Amp)/np.nanmean(PPR2_1_20Hz_Ca25_Amp)
CV_PPR_4 = np.nanstd(PPR2_1_20Hz_Ca4_Amp)/np.nanmean(PPR2_1_20Hz_Ca4_Amp)

ax_CV[0].scatter([0,1,2], [CV_peak1_15, CV_peak1_25, CV_peak1_4], color=[palette[0],palette[1],palette[2]])
ax_CV[0].scatter([3,4,5], [CV_peak2_15, CV_peak2_25, CV_peak2_4], color=[palette[0],palette[1],palette[2]])
ax_CV[1].scatter([0,1,2], [CV_PPR_15, CV_PPR_25, CV_PPR_4], color=[palette[0],palette[1],palette[2]])
ax_CV[0].plot([0,3], [CV_peak1_15, CV_peak2_15], color='k', alpha=0.4)
ax_CV[0].plot([1,4], [CV_peak1_25, CV_peak2_25], color='k', alpha=0.4)
ax_CV[0].plot([2,5], [CV_peak1_4, CV_peak2_4], color='k', alpha=0.4)

ax_CV[0].set_ylim(0.,0.7)
ax_CV[1].set_ylim(0.,0.45)
ax_CV[0].set_title('CV of peak1 and peak2')
ax_CV[1].set_title('CV of PPR2/1')


fig_var_amp, ax_var_amp = plt.subplots(1,2,figsize=(6,5), tight_layout=True)
ax_var_amp[0].scatter(np.nanmean(PEAK1_20Hz_Ca15_Amp)**2,np.nanstd(PEAK1_20Hz_Ca15_Amp),color=palette[0])
ax_var_amp[0].scatter(np.nanmean(PEAK1_20Hz_Ca25_Amp)**2,np.nanstd(PEAK1_20Hz_Ca25_Amp),color=palette[1])
ax_var_amp[0].scatter(np.nanmean(PEAK1_20Hz_Ca4_Amp)**2,np.nanstd(PEAK1_20Hz_Ca4_Amp),color=palette[2])
ax_var_amp[1].scatter(np.nanmean(PEAK2_20Hz_Ca15_Amp)**2,np.nanstd(PEAK2_20Hz_Ca15_Amp),color=palette[0])
ax_var_amp[1].scatter(np.nanmean(PEAK2_20Hz_Ca25_Amp)**2,np.nanstd(PEAK2_20Hz_Ca25_Amp),color=palette[1])
ax_var_amp[1].scatter(np.nanmean(PEAK2_20Hz_Ca4_Amp)**2,np.nanstd(PEAK2_20Hz_Ca4_Amp),color=palette[2])

ax_var_amp[0].set_title('Variance/Mean for Peak1'), ax_var_amp[1].set_title('Variance/Mean for Peak2')
ax_var_amp[0].set_xlabel('Mean P1 (DF/F0)'), ax_var_amp[1].set_xlabel('Mean P2 (DF/F0)')
ax_var_amp[0].set_ylabel('Variance'), ax_var_amp[1].set_ylabel('Variance')


# fig_histo, ax_histo = plt.subplots(3,2, figsize=(10,6), tight_layout=True, sharex=True)
# ax_histo[0,0].hist(PEAK1_20Hz_Ca15_Amp, bins=3, color=palette2[0]), ax_histo[0,1].hist(PEAK2_20Hz_Ca15_Amp, bins=3, color=palette2[0])
# ax_histo[1,0].hist(PEAK1_20Hz_Ca25_Amp, bins=25, color=palette2[1]), ax_histo[1,1].hist(PEAK2_20Hz_Ca25_Amp, bins=25, color=palette2[1])
# ax_histo[2,0].hist(PEAK1_20Hz_Ca4_Amp, bins=7, color=palette2[2]), ax_histo[2,1].hist(PEAK2_20Hz_Ca4_Amp, bins=7, color=palette2[2])




fig2, ax2 = plt.subplots(1,3, figsize=(10,6), tight_layout=True, sharey=True)
x = np.arange(1,11,1)
for profile15 in range(len(NORM_PROFILES_15)):
    ax2[0].plot(x, NORM_PROFILES_15[profile15], 'skyblue', alpha=0.5), ax2[0].scatter(x, NORM_PROFILES_15[profile15], color='skyblue', alpha=0.5)
for profile25 in range(len(NORM_PROFILES_25)):
    ax2[1].plot(x, NORM_PROFILES_25[profile25], 'limegreen', alpha=0.5), ax2[1].scatter(x, NORM_PROFILES_25[profile25], color='limegreen', alpha=0.5)
for profile4 in range(len(NORM_PROFILES_4)):
    ax2[2].plot(x, NORM_PROFILES_4[profile4], 'orange', alpha=0.5), ax2[2].scatter(x, NORM_PROFILES_4[profile4], color='orange', alpha=0.5)
ax2[0].set_title('1.5mM'), ax2[1].set_title('2.5mM'), ax2[2].set_title('4mM')
ax2[0].set_ylabel('PEAKn/PEAK1'), ax2[0].set_xlabel('#Pulse')
ax2[0].plot([1,10],[1,1], 'k', linestyle='--'), ax2[1].plot([1,10],[1,1], 'k', linestyle='--'), ax2[2].plot([1,10],[1,1], 'k', linestyle='--')
# fig2.savefig('Averaged_traces.pdf') 


fig3, ax3 = plt.subplots(1,2, figsize=(10,6), tight_layout=True)
KS_value, pvalue = stats.ks_2samp(TAU_15, TAU_25)
KS_value2, pvalue2 = stats.ks_2samp(TAU_15, TAU_4)
KS_value3, pvalue3 = stats.ks_2samp(TAU_25, TAU_4)
print('----------')
print('KS-test 1.5Vs2.5:\nKS_value = {:.3f} ; pvalue = {:.3f}'.format(KS_value, pvalue))
print('KS-test 1.5Vs4:\nKS_value = {:.3f} ; pvalue = {:.3f}'.format(KS_value2, pvalue2))
print('KS-test 2.5Vs4:\nKS_value = {:.3f} ; pvalue = {:.3f}'.format(KS_value3, pvalue3))

tauDf = pd.concat((pd.DataFrame(TAU_15),pd.DataFrame(TAU_25), pd.DataFrame(TAU_4)), axis=1)
tauDf.columns = ['1.5mM', '2.5mM', '4mM']
binning = 4
sns.distplot(tauDf['1.5mM'], bins=binning, ax=ax3[0])
sns.distplot(tauDf['4mM'], bins=binning, ax=ax3[0])
sns.distplot(tauDf['2.5mM'], bins=binning, ax=ax3[0])
sns.distplot(tauDf['1.5mM'], bins=binning, hist=False, hist_kws={'cumulative':True}, kde_kws={'cumulative':True}, ax=ax3[1])
sns.distplot(tauDf['4mM'], bins=binning, hist=False, hist_kws={'cumulative':True}, kde_kws={'cumulative':True}, ax=ax3[1])
sns.distplot(tauDf['2.5mM'], bins=binning, hist=False, hist_kws={'cumulative':True}, kde_kws={'cumulative':True}, ax=ax3[1])
ax3[0].set_xlabel('Tau (ms)'), ax3[1].set_xlabel('Tau (ms)')
ax3[0].set_ylabel('Count'), ax3[1].set_ylabel('Cumulative frequencies')
ax3[0].set_title('1.5 vs 2.5: pvalue = {:.3f}\n1.5 vs 4: pvalue = {:.3f}\n2.5 vs 4: pvalue = {:.3f}'.format(pvalue,pvalue2,pvalue3))
# ax3[1,0].set_xlabel('Time (ms)'), ax3[1,1].set_xlabel('Time (ms)')
# ax3[1,0].set_ylabel('DF/F0')
# ax3[1,0].set_title('1.5mM'), ax3[1,1].set_title('4mM')

fig4, ax4 = plt.subplots(figsize=(8,8), tight_layout=True)
sns.boxplot(data=tauDf, showmeans=True, ax=ax4, palette=palette2, meanprops={'markerfacecolor':'red',
                                                                                  'markeredgecolor':'black',
                                                                                  'markersize':'8'})

data1 = tauDf.iloc[:5,0].values
data2 = tauDf.iloc[:,1].values
data3 = tauDf.iloc[:20,2].values
# ax4.scatter([0,1,2], [data1, data2, data3], color=[palette2[0],palette2[1],palette2[2]], edgecolors='k', s=20, alpha=0.6)
add_stat_annotation(ax4, data=tauDf, box_pairs=[('1.5mM','4mM'),('2.5mM','4mM'),('1.5mM','2.5mM')],test='Kruskal', text_format='star', verbose=2)


'''


for file in range(len(files_tau)):
    if 'Tau_alone' in files_tau[file]:
        continue
    
    # if 'Profile' in files_tau[file]:
    #     df = pd.read_excel('{}/{}'.format(Path_Tau, files_tau[file]), header=0)
        
    #     for i in range(len(df.columns)):
    #         if 'Time' == df.columns[i]:
    #             timescale = df.iloc[:,i].values
    #         if 'Average' == df.columns[i]:
    #             sweep = df.iloc[:,i].values
            
    #     if '1.5mM' in files_tau[file]:
    #         ax3[1,0].plot(timescale, sweep, alpha=0.5)
    #     else:
    #         ax3[1,1].plot(timescale, sweep, alpha=0.5)

                
    if 'Amp' in files_tau[file]:
        amp_data = pd.read_excel('{}/{}'.format(Path_Tau, files_tau[file]), header=None, Index=None)
        amp_values = amp_data.iloc[1,1:].values
        
    if 'Tau' in files_tau[file]:
        tau_data = pd.read_excel('{}/{}'.format(Path_Tau, files_tau[file]), header=None, Index=None)
        tau_values = tau_data.iloc[1,1:].values
        
        for i in range(tau_values.shape[0]):
            if tau_values[i] >= 1.0:
                continue
            if '1.5mM' in files_tau[file]:
                TAU_15.append(tau_values[i]*1000)
                AMP_15.append(amp_values[i])
            else:
                TAU_4.append(tau_values[i]*1000)
                AMP_4.append(amp_values[i])




# fig4, ax4 = plt.subplots(1,1, figsize=(7,5), tight_layout=True)
# AmpTauDf = pd.concat((pd.DataFrame(TAU_15), pd.DataFrame(AMP_15),
#                       pd.DataFrame(TAU_4), pd.DataFrame(AMP_4)), axis=1)
# AmpTauDf.columns = ['Amp1.5mM', 'Tau1.5mM', 'Amp4mM', 'Tau4mM']
# ax4.scatter(AmpTauDf['Amp1.5mM'], AmpTauDf['Tau1.5mM'], color=palette[0], s=20)
# ax4.scatter(AmpTauDf['Amp4mM'], AmpTauDf['Tau4mM'], color=palette[1], s=20)
# ax4.set_ylabel('Amp (DF/F0)'), ax4.set_xlabel('Tau (ms)')
'''