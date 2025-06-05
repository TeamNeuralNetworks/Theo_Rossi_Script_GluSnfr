# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 11:29:11 2019

@author: Theo.ROSSI
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import configparser as cp
import peakutils

path = r'D:/Theo.ROSSI/GluSnFR/Virus_AAV.hSynap.SF-iGluSnFR_S72A/AAVDJ.GluSnFR-S72A/2019_10_10'

folders = os.listdir(path)

LIST_INI_FILES = []
LIST_FREQ_FILES = []

LIST_INTERVALS = []
LIST_X_POSITION = []

LIST_AVERAGE_INTERVAL = []
LIST_dF_F0 = []

for f in range(len(folders)):
   
    PIXEL_VALUE = []
    PIXEL_VALUE_WOUT_FBACK = []   
    PEAKS_VALUES = []
    
    
    if 'freq_coord' in folders[f]: #ADAPTER nom fichier excel
        new_path = pd.read_csv('%s/%s'%(path, folders[f]),sep="\t",header=1)
        
        X, Y = [],[]
        
        for i in range(len(new_path)): 
            
            basic_value = new_path.iloc[i,0]
            
            X.append(int(basic_value.split(',')[0])+1)
            Y.append(int(basic_value.split(',')[1])+1)
    

    if '20Hzbis' in folders[f]:
        continue
    
    if 'Hz' in folders[f]:
        path2 = '%s/%s'%(path, folders[f])
        the_folder = os.listdir(path2)
        
        for file in the_folder:
            if '.ini' in file:
                ini_file = '%s/%s/%s'%(path, folders[f], file)
                LIST_INI_FILES.append(ini_file)
                config = cp.ConfigParser()
                config.read(LIST_INI_FILES[0])
                sampling_rate, px_dwell_time, px_size = float(config.get('_', 'lines.per.second')), float(config.get('_', 'pixel.dwell.time.in.sec'))*1000, float(config.get('_', 'x.pixel.sz'))*1000000
                px_100µm = 100/px_size
                print('Sampling_rate: %s, px_dwell_time: %s, px_size: %s, px_100µm: %s'%(sampling_rate, px_dwell_time, px_size, px_100µm))
                
            
    if 'freq_roi' in folders[f]: #ADAPTER nom fichier .tif
        #print('%s this is the ROI file'%folders_files[f])
        ROI_file = plt.imread('%s/%s'%(path, folders[f]))
        ROI_file = ROI_file[:,:,0]
        
        for x, y in zip(X, Y):
            pixel = ROI_file[y,x]
            PIXEL_VALUE.append(pixel)
        
        distance_vector = np.arange(0, 100/px_100µm*len(X), 100/px_100µm)
    
        f_back_start = np.ravel(np.where(distance_vector >= 425))[0] #ADAPTER
        f_back_stop = np.ravel(np.where(distance_vector <= 450))[-1] #ADAPTER
        f_back = np.mean(PIXEL_VALUE[f_back_start:f_back_stop])
        ROI_file2 = ROI_file-f_back


        for z in range(len(PIXEL_VALUE)):
            px_wout_fback = PIXEL_VALUE[z]-f_back
            PIXEL_VALUE_WOUT_FBACK.append(px_wout_fback)
            
        slim_trace = np.convolve(PIXEL_VALUE_WOUT_FBACK, np.ones((4,))/4, mode='valid').tolist() #ADAPTER
        distance_vector2 = np.arange(0, 100/px_100µm*len(slim_trace), 100/px_100µm)
        
        ROI_sampling = (1.0*len(slim_trace))/(len(slim_trace)*px_dwell_time)
        time_vector_ROI = np.arange(0, 1.0/ROI_sampling*len(slim_trace), 1.0/ROI_sampling)
        
        threshold = 70 #ADAPTER
#        peaks_indexes = spsi.find_peaks(slim_trace, height=threshold) #Donne valeurs et indices pics > threshold
        indexes = peakutils.indexes(slim_trace, thres=0.7).tolist() #Donne uniquement indices des pics > threshold #ADAPTER
        peaks = np.where(np.where([(slim_trace - np.roll(slim_trace, 1) > 0) & (slim_trace - np.roll(slim_trace, -1) > 0)], slim_trace, 0) > threshold, slim_trace, np.nan)
        peaks = peaks[0,:]
#        print(indexes)
        
        
        fig, ax = plt.subplots(3,1, figsize=(15,10), tight_layout=True)
        ax[0].imshow(ROI_file2), ax[0].set_title('RAW LINESCAN', fontsize = 15), ax[0].set_ylabel('Y res'), ax[0].set_xlabel('X res')
        ax[0].plot(X, Y, 'r')
        ax[1].plot(distance_vector, PIXEL_VALUE_WOUT_FBACK, label='raw data'), ax[1].set_ylabel('Pixels values'), ax[1].grid(axis='y', linestyle='--')
        ax[2].plot(distance_vector2, slim_trace, 'r', label='Convolved data')
        ax[2].scatter(distance_vector2, peaks, s=100, marker='x')
        ax[2].set_xlabel('Line position (µm)'), ax[2].set_ylabel('Convolved pixels values'), ax[2].grid(axis='y', linestyle='--'), ax[2].grid(axis='x', linestyle='--')
        ax[2] = ax[2].twiny()
        ax[2].plot([0,1.0/ROI_sampling*len(slim_trace)], [threshold,threshold], 'grey', linestyle = '--', linewidth = 2, label='threshold')
        ax[2].set_xlabel('Time (ms)'), ax[2].tick_params(axis='x', colors='g'), ax[2].spines['top'].set_color('g'), ax[2].xaxis.label.set_color('g'), ax[2].grid(axis='x', linestyle='--', color='g')
        plt.legend()

    
    if 'Hz' in folders[f]:
        path2bis = '%s/%s'%(path, folders[f])
        the_folder2 = os.listdir(path2bis)
        
        for file in the_folder:    
            if 'ch_1' in file:
                data = plt.imread('%s/%s/%s'%(path, folders[f], file)).transpose(1,0,2)
                data = data[:,:,0]
                corrected_fluo = data-f_back
                LIST_FREQ_FILES.append(corrected_fluo)
                
                time_vector = np.arange(0, 1.0/sampling_rate*len(corrected_fluo[1]), 1.0/sampling_rate)
                selected_line = 268
                       
                button = corrected_fluo[selected_line-268 : selected_line+268, :]
                average_interval = np.mean(button, axis=0)
                LIST_AVERAGE_INTERVAL.append(average_interval)
                
                average_button = np.mean(LIST_AVERAGE_INTERVAL, axis=0)
                f0_start = np.ravel(np.where(time_vector >= 0.2))[0] 
                f0_stop = np.ravel(np.where(time_vector >= 0.45))[0] 
                f0 = np.mean(average_button[f0_start:f0_stop])
                
                dF_F0 = (average_button-f0)/f0
                LIST_dF_F0.append(dF_F0)
                
for linescan in range(len(LIST_FREQ_FILES)):
    fig, ax = plt.subplots(2,1, figsize=(15,10), tight_layout=True)
    ax[0].imshow(LIST_FREQ_FILES[linescan])
    ax[1].plot(time_vector[1:-1], LIST_dF_F0[linescan][1:-1])
    
                

    