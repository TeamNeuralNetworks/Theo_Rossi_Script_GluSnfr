# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 13:04:11 2020

@author: Theo.ROSSI
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
import configparser as cp
import os
import numpy as np
import pandas as pd
import scipy.optimize as opt


##########################################################SETTINGS#############################################################
###############################################################################################################################

Path = r'\\equipe2-nas1\Theo.ROSSI\GluSnFR\Virus_AAV.hSynap.SF-iGluSnFR_S72A\AAVDJ.GluSnFR-S72A'
Big_folder = '2019_10_17'
Idx_excel_file = '20191017_linescan3_50Hz_idx.xlsx'
Linescan_folder = 'linescan3_50Hz_files'
Amp_excel_file = '20191017_linescan3_50Hz_amp.xlsx'


freq_Hz = 50
'''Stimulation frequency'''

f_back_start = 453
f_back_stop = 458
'''Take fback indexes from the image on the LinePy code'''

idx1_adjustment = 30
idx2_adjustment = 580
'''This will adjust the upper (idx1) and lower (idx2) boundaries of the pixels
window on the linescan for each button''' 

moving_average = 8
'''Min value=1. This gives the intensity of the smoothing filter. Change the number of points
to average in the np.convolve/np.ones functions.'''

Residual = False
'''Turn True when there is residual signal'''

Show_fit = False
'''Show decay exp fit for each peak'''

Photobleaching = False
'''Turn into True when photobleaching substraction is required'''

Dataframe = False
'''Turn True to import button's data in the excel file'''

###############################################################################################################################
###############################################################################################################################


def func(x, a, b, c):
    return a*np.exp(-b*x) + c


def gaus(x,a,x0,sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))



folders_files = sorted(os.listdir('{}/{}'.format(Path, Big_folder)))


LIST_INI_FILES = []
LIST_IDX1, LIST_IDX2 = [], []


cm = ['tomato', 'darkorange', 'turquoise', 'limegreen', 'cornflowerblue', 'purple', 'magenta', 'darkred']

for f in range(len(folders_files)):
    
    if Idx_excel_file in folders_files[f]:
        path2 = pd.read_excel('{}/{}/{}'.format(Path, Big_folder, folders_files[f]))
        
        for j in range(len(path2)):
            
            LIST_IDX1.append(path2.iloc[j,0])
            LIST_IDX2.append(path2.iloc[j,1])



LIST_FILES = []
BIG_LIST_AVERAGE_AMPS = []
LIST_CONVOLVED_dF_F0 = []

for i in range(len(path2)):
    LIST_AVERAGE_INTERVAL = []
    LIST_dF_F0 = []
    LIST_CORRECTED_FLUO = []
    LIST_MEAN_BSL = []
    LIST_PEAK_VALUE = []
    LIST_PEAK_TIME_INDEX = []
    LIST_NEG_PEAK_INDEX = []
    LIST_POS_PEAK_INDEX = []
    LIST_PEAK_INDEX = []
    LIST_AVERAGE_AMPS = []
    LIST_FITTED_dF_F0 = []

    
    fig = plt.figure(figsize=(15,7), tight_layout=True)
    gs = gridspec.GridSpec(2,2)
    ax1 = fig.add_subplot(gs[0,0])
    ax2 = fig.add_subplot(gs[1,:])
    ax2.set_ylim(-0.2, 1.3)
    
    for f in range(len(folders_files)):
        if Linescan_folder in folders_files[f]: 
    #        print('%s contains the linescans folders'%folders_files[f])
            path3 = '{}/{}/{}'.format(Path, Big_folder, folders_files[f])
            folders = os.listdir(path3)
            
    
            for file in folders:
                if '.ini' in file:
                    ini_file = '{}/{}/{}/{}'.format(Path, Big_folder, folders_files[f], file)
                    LIST_INI_FILES.append(ini_file)
                    
            config = cp.ConfigParser()
            config.read(LIST_INI_FILES[0])
            sampling_rate, px_dwell_time, px_size = float(config.get('_', 'lines.per.second')), float(config.get('_', 'pixel.dwell.time.in.sec'))*1000, float(config.get('_', 'x.pixel.sz'))*1000000
            px_100µm = 100/px_size
            #print('Sampling_rate: %s, px_dwell_time: %s, px_size: %s, px_100µm: %s'%(sampling_rate, px_dwell_time, px_size, px_100µm))
    
    
            for file in folders:
                if 'ch_1.tif' in file:
                    LIST_FILES.append('{}/{}/{}/{}'.format(Path, Big_folder, folders_files[f], file))
                    data = plt.imread('{}/{}/{}/{}'.format(Path, Big_folder, folders_files[f], file)).transpose(1,0,2)
                    data = data[:,:,0]
                    
                    f_back = np.mean(data[f_back_start : f_back_stop, :])
                    corrected_fluo = data-f_back
                    corrected_fluo = np.where(corrected_fluo < 0, 0, corrected_fluo)
                    LIST_CORRECTED_FLUO.append(corrected_fluo)
                    
                    time_vector = np.arange(0, 1.0/sampling_rate*len(corrected_fluo[1]), 1.0/sampling_rate)
                    
                    button = corrected_fluo[LIST_IDX1[i]-idx1_adjustment : LIST_IDX2[i]+idx2_adjustment, :]
                    average_interval = np.mean(button, axis=0)
                    LIST_AVERAGE_INTERVAL.append(average_interval)
                    
                    f0_start = np.ravel(np.where(time_vector >= 0.2))[0] 
                    f0_stop = np.ravel(np.where(time_vector >= 0.45))[0] 
                    f0 = np.mean(average_interval[f0_start:f0_stop])
                    
                    dF_F0 = (average_interval-f0)/f0
                    LIST_dF_F0.append(dF_F0)
                    
                    
                    y1 = np.ravel(np.where(time_vector >= time_vector[1]))[0]
                    y2 = np.ravel(np.where(time_vector <= 0.98))[-1]
                    curve1 = dF_F0[y1:y2]
                    popt, pcov = opt.curve_fit(func, time_vector[y1:y2], curve1, maxfev=2000)
                    
                    #print('Curve 1 coefficients:')
                    #print('a = %s, b = %s, c = %s' %(popt[0], popt[1], popt[2]))
                    '''Exponential curve made between 0 sec and 0.8 sec'''
                    
                    x1 = np.ravel(np.where(time_vector >= time_vector[1]))[0]
                    x2 = np.ravel(np.where(time_vector <= time_vector[-1]))[-1]
                    curve2 = func(time_vector[x1:x2],*popt)
                    '''Expand the exponential curve unitil the end of the plot'''
                        
                        
                    if Photobleaching == True:
                        fitted_dF_F0 = dF_F0[1:-1]-curve2
                        LIST_FITTED_dF_F0.append(fitted_dF_F0)
                        '''Photobleaching substraction'''
                        
                        
                        
    if freq_Hz == 20:
        stims = np.arange(0.5,1,0.05)
        stim_off = stims+0.05
        
    elif freq_Hz == 50:
        stims = np.arange(0.5,0.7,0.02)
        stim_off = stims+0.02
        
    elif freq_Hz == 100:
        stims = np.arange(0.5,0.6,0.01)
        stim_off = stims+0.01
    
    
    if Photobleaching == True:
        average_button = np.mean(LIST_FITTED_dF_F0, axis=0)
    else:
        average_button = np.mean(LIST_dF_F0, axis=0)
    
    convolved_dF_F0 = np.convolve(average_button, np.ones((moving_average,))/moving_average, mode='valid')
    LIST_CONVOLVED_dF_F0.append(convolved_dF_F0)
        

    for m,n,index in zip(stims, stim_off, range(len(stims))):
        
        if Photobleaching == True:
            where = time_vector[moving_average:-1]
        else:
            where = time_vector[moving_average-2:-1]
            
        bsl1 = np.ravel(np.where(where >= m))[0]
        bsl2 = np.ravel(np.where(where <= n))[-1]
        
        peak_time_index = time_vector[convolved_dF_F0[bsl1:bsl2].argmax()] + m
        peak_index = np.ravel(np.where(time_vector >= time_vector[convolved_dF_F0[bsl1:bsl2].argmax()] + time_vector[bsl1]))[0]
        LIST_PEAK_TIME_INDEX.append(peak_time_index)
        LIST_PEAK_INDEX.append(peak_index)
        
        
        for idx in range(peak_index):
            pos = idx+3
            neg = idx-1
        LIST_NEG_PEAK_INDEX.append(neg)
        LIST_POS_PEAK_INDEX.append(pos)
        
        for value1, value2 in zip(range(len(LIST_NEG_PEAK_INDEX)), range(len(LIST_POS_PEAK_INDEX))):
            peak_value = np.mean(convolved_dF_F0[LIST_NEG_PEAK_INDEX[value1]:LIST_POS_PEAK_INDEX[value2]])
        LIST_PEAK_VALUE.append(peak_value)

        
        if Residual == False:
            bsl3 = np.ravel(np.where(where >= 0.3))[0]
            bsl4 = np.ravel(np.where(where <= 0.5))[-1]
            
            mean_bsl = np.mean(convolved_dF_F0[bsl3:bsl4])
            LIST_MEAN_BSL.append(mean_bsl)


    if Residual == True:
        for a,b in zip(range(len(LIST_PEAK_INDEX)), stim_off):
            stim_off_index = np.ravel(np.where(where <= b))[-1]
            exp = convolved_dF_F0[LIST_PEAK_INDEX[a]:stim_off_index]
            popt, pcov = opt.curve_fit(func, where[LIST_PEAK_INDEX[a]:stim_off_index], exp, maxfev=40000)
            
            if LIST_PEAK_INDEX[a] == LIST_PEAK_INDEX[-1]:
                continue
                
            exp2 = func(where[LIST_PEAK_INDEX[a]:LIST_PEAK_INDEX[a+1]],*popt)
            
            LIST_MEAN_BSL.append(exp2[-1])
            
            if LIST_PEAK_INDEX[a] == LIST_PEAK_INDEX[0]:
                bsl3 = np.ravel(np.where(where >= 0.3))[0]
                bsl4 = np.ravel(np.where(where <= 0.5))[-1]
                mean_bsl = np.mean(convolved_dF_F0[bsl3:bsl4])
                LIST_MEAN_BSL.insert(0,mean_bsl)
            
            if Show_fit == True:
                ax2.plot(where[LIST_PEAK_INDEX[a]:LIST_PEAK_INDEX[a+1]], exp2, 'k', linewidth=2)
          

    for p,q in zip(range(len(LIST_MEAN_BSL)), range(len(LIST_PEAK_VALUE))):
        amps = LIST_PEAK_VALUE[q]-LIST_MEAN_BSL[p]
        LIST_AVERAGE_AMPS.append(amps)
    BIG_LIST_AVERAGE_AMPS.append(LIST_AVERAGE_AMPS)
    
    
    
    ax1.imshow(LIST_CORRECTED_FLUO[0]), ax1.set_xticks(np.arange(0, len(corrected_fluo[1]), 200)), ax1.set_title('First linescan')
    rectangle = patches.Rectangle((0, LIST_IDX1[i]-idx1_adjustment), len(corrected_fluo[1]), (LIST_IDX2[i]+idx2_adjustment)-(LIST_IDX1[i]-idx1_adjustment), linewidth=0.9, edgecolor='r', facecolor='none')
    ax1.add_patch(rectangle)
    
    ax1 = fig.add_subplot(gs[0,1])
    ax1.imshow(LIST_CORRECTED_FLUO[-1]), ax1.set_xticks(np.arange(0, len(corrected_fluo[1]), 200)), ax1.set_title('Last linescan')
    
    

    if Photobleaching == True:
        ax2.plot([time_vector[moving_average], time_vector[-1]], [0,0], 'r', linestyle='--', linewidth=2), ax2.set_xlabel('Time (s)'), ax2.set_ylabel('dF/F0'), ax2.grid(True, linestyle='--')
#        for z in range(len(LIST_FITTED_dF_F0)):
#            ax2.plot(time_vector[1:-1], LIST_FITTED_dF_F0[z], 'k', alpha=0.15)
        
    else:
        ax2.plot([time_vector[moving_average-1], time_vector[-1]], [0,0], 'r', linestyle='--', linewidth=2), ax2.set_xlabel('Time (s)'), ax2.set_ylabel('dF/F0'), ax2.grid(True, linestyle='--')
#        for z in range(len(LIST_dF_F0)):
#            ax2.plot(time_vector[1:-1], LIST_dF_F0[z][1:-1], 'k', alpha=0.15)
    
    
#    ax2.plot(time_vector[1:-1], curve2)
    ax2.plot(where, convolved_dF_F0, cm[i], linewidth=3)
#    ax2.scatter(LIST_PEAK_TIME_INDEX, LIST_PEAK_VALUE)
    
    for stim in stims:
        ax2.scatter(stim, 0, c='k', marker='*', linewidths=3)
        
        
    plt.title('BUTTON {} AT INDEXES {} TO {}'.format(i, LIST_IDX1[i]-idx1_adjustment, LIST_IDX2[i]+idx2_adjustment))
    
       

if Dataframe == True:
    df = pd.DataFrame(BIG_LIST_AVERAGE_AMPS, index=None, columns=None)
    print(df)
    
    with pd.ExcelWriter('{}/{}/{}'.format(Path, Big_folder, Amp_excel_file)) as writer:
        df.to_excel(writer, header=False, index=False)