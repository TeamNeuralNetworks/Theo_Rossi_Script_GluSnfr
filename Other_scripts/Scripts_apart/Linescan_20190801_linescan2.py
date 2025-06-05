# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 10:58:31 2019

@author: Theo.ROSSI
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
import configparser as cp
import os
import numpy as np
import scipy.signal as spsi
import scipy.optimize as opt
import pandas as pd
import peakutils
#matplotlib.rcParams['pdf.fonttype'] = 42


#Fonction exponentielle
def func(x, a, b, c):
    return a*np.exp(-b*x) + c

#Fonction polynome degre 3
#def func2(x, a, b, c, d):
#    return a*x**3 + b*x**2 + c*x + d

#Equation gaussienne
def gaus(x,a,x0,sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))


path = r'D:/Theo.ROSSI/GluSnFR/Virus_AAV.hSynap.SF-iGluSnFR_S72A/AAVDJ.GluSnFR-S72A/2019_08_01'

folders_files = sorted(os.listdir(path))

LIST_INI_FILES = []

X_COORD_1 = []
Y_COORD_1 = []
X_COORD_2 = []
Y_COORD_2 = []

LIST_INTERVALS = []
LIST_X_POSITION = []

NEGATIVE_2SIGMA_INDEXES = []
POSITIVE_2SIGMA_INDEXES = []

cm = ['tomato', 'darkorange', 'turquoise', 'limegreen', 'cornflowerblue']

for f in range(len(folders_files)):
    if 'linescan2.xlsx' in folders_files[f]: #ADAPTER nom fichier excel
        #print('%s are the coordinates of the line'%folders_files[f])
        the_file = pd.read_excel('%s/%s'%(path, folders_files[f]), sheet_name='Feuil1', header=0)
        x_data = the_file.iloc[:,0].values.astype(int).tolist()
        y_data = the_file.iloc[:,1].values.tolist()
    
    PIXEL_VALUE = []
    PIXEL_VALUE_WOUT_FBACK = []   
    PEAKS_VALUES = []
    
    if folders_files[f].startswith('20190801_linescan2'): #ADAPTER nom dossier
        #print('%s contains the linescans folders'%folders_files[f])
        path2 = '%s/%s'%(path, folders_files[f])
        folders = sorted(os.listdir(path2))
        
        for folder in range(len(folders)):
            path3 = '%s/%s/%s'%(path, folders_files[f], folders[folder])
            the_folder = os.listdir(path3)
              
            for file in the_folder:
                if '.ini' in file:
                    ini_file = '%s/%s/%s/%s'%(path, folders_files[f], folders[folder], file)
                    LIST_INI_FILES.append(ini_file)
                
        config = cp.ConfigParser()
        config.read(LIST_INI_FILES[0])
        sampling_rate, px_dwell_time, px_size = float(config.get('_', 'lines.per.second')), float(config.get('_', 'pixel.dwell.time.in.sec'))*1000, float(config.get('_', 'x.pixel.sz'))*1000000
        px_100µm = 100/px_size
        print('Sampling_rate: %s, px_dwell_time: %s, px_size: %s, px_100µm: %s'%(sampling_rate, px_dwell_time, px_size, px_100µm))
       
        
    if 'linescan2_[xy]T_ROI.tif' in folders_files[f]: #ADAPTER nom fichier .tif
        #print('%s this is the ROI file'%folders_files[f])
        ROI_file = plt.imread('%s/%s'%(path, folders_files[f]))
        ROI_file = ROI_file[:,:,0]
        
        for x1, y1 in zip(x_data, y_data):
            pos_x1, pos_y1 = x1-2, y1-2
            X_COORD_1.append(pos_x1)
            Y_COORD_1.append(pos_y1)
        
        for x, y in zip(X_COORD_1, Y_COORD_1):
            pixel = ROI_file[y,x]
            PIXEL_VALUE.append(pixel)
        
        distance_vector = np.arange(0, 100/px_100µm*len(x_data), 100/px_100µm)
        
        f_back_start = np.ravel(np.where(distance_vector >= 2))[0] #ADAPTER
        f_back_stop = np.ravel(np.where(distance_vector <= 7))[-1] #ADAPTER
        f_back = np.mean(PIXEL_VALUE[f_back_start:f_back_stop])
        ROI_file2 = ROI_file-f_back
        
        for z in range(len(PIXEL_VALUE)):
            px_wout_fback = PIXEL_VALUE[z]-f_back
            PIXEL_VALUE_WOUT_FBACK.append(px_wout_fback)
            
        slim_trace = np.convolve(PIXEL_VALUE_WOUT_FBACK, np.ones((4,))/4, mode='valid').tolist() #ADAPTER
        distance_vector2 = np.arange(0, 100/px_100µm*len(slim_trace), 100/px_100µm)
        
        ROI_sampling = (1.0*len(slim_trace))/(len(slim_trace)*px_dwell_time)
        time_vector_ROI = np.arange(0, 1.0/ROI_sampling*len(slim_trace), 1.0/ROI_sampling)
        
        threshold = 30 #ADAPTER
#        peaks_indexes = spsi.find_peaks(slim_trace, height=threshold) #Donne valeurs et indices pics > threshold
        indexes = peakutils.indexes(slim_trace, thres=0.75).tolist() #Donne uniquement indices des pics > threshold #ADAPTER
        peaks = np.where(np.where([(slim_trace - np.roll(slim_trace, 1) > 0) & (slim_trace - np.roll(slim_trace, -1) > 0)], slim_trace, 0) > threshold, slim_trace, np.nan)
        peaks = peaks[0,:]
        print(indexes)
#        fig = plt.figure(1)
#        gridspec.GridSpec(3,3)
#        plt.subplot2grid((3,3), (0,0), colspan=2, rowspan=3)
#        plt.imshow(ROI_file), plt.title('Raw linescan', fontsize = 15), plt.plot(X_COORD_2, Y_COORD_2, 'r')
#        
#        plt.subplot2grid((3,3), (0,2))
#        plt.plot(distance_vector, PIXEL_VALUE), plt.ylabel('Pixel value')
#        
#        plt.subplot2grid((3,3), (2,2))
#        plt.plot(distance_vector2, slim_trace, 'r'), plt.xlabel('Line position (µm)')
#        ax2 = plt.twiny()
#        ax2.plot(time_vector_ROI, slim_trace, 'r', label='convolved data'), ax2.set_xlabel(r'Time (ms)')
    
        plt.figure(figsize=(11,7))
        for j in range(len(indexes)):
            interval = slim_trace[indexes[j]-20 : indexes[j]+20]
            LIST_INTERVALS.append(interval)
            x_position = distance_vector2[indexes[j]-20 : indexes[j]+20]
            LIST_X_POSITION.append(x_position)
            plt.plot(distance_vector2[0:40], LIST_INTERVALS[j], cm[j], label='Button %s'%j), plt.grid(axis='y', linestyle='--')
        
        average_interval = np.mean(LIST_INTERVALS, axis=0)
        popt, pcov = opt.curve_fit(gaus, distance_vector2[0:40], average_interval)
        print('a = %s, x0 = %s, sigma = %s' %(popt[0], popt[1], popt[2]))
        for i in indexes:
            negative_2sigma = np.ravel(np.where(distance_vector2 <= distance_vector2[i]+2*popt[2]))[-1]
            positive_2sigma = np.ravel(np.where(distance_vector2 <= distance_vector2[i]-2*popt[2]))[-1]
            #print(negative_2sigma, positive_2sigma)
            NEGATIVE_2SIGMA_INDEXES.append(negative_2sigma)
            POSITIVE_2SIGMA_INDEXES.append(positive_2sigma)
#        
        x = np.arange(popt[1]-2*popt[2], popt[1]+2*popt[2], 0.001)
        
        plt.plot(distance_vector2[0:40], average_interval, color='crimson', linewidth=3, label='Average')
        plt.plot(distance_vector2[0:40], gaus(distance_vector2[0:40], *popt), color='k', linewidth=3, label='Gaussian fit on the average')
        plt.plot([popt[1], popt[1]], [0, popt[0]], 'k', linestyle='-.', linewidth=2, label='µ')
        plt.fill_between(x,  gaus(x, *popt), color='purple', alpha=0.2, label='2std')
#        plt.plot([popt[1]-2*popt[2], popt[1]-2*popt[2]], [0, popt[0]], 'k', linestyle=':', linewidth=2), plt.plot([popt[1]+2*popt[2], popt[1]+2*popt[2]], [0, popt[0]], 'k', linestyle=':', linewidth=2)
        plt.title('Buttons template'), plt.xlabel('Distance (µm)'), plt.ylabel('Convolved pixels values')
        plt.legend()
#        
#        
        fig, ax = plt.subplots(3,1, figsize=(15,10), tight_layout=True)
        ax[0].imshow(ROI_file2), ax[0].set_title('RAW LINESCAN', fontsize = 15), ax[0].set_ylabel('Y res'), ax[0].set_xlabel('X res')
        ax[0].plot(X_COORD_1, Y_COORD_1, 'r')
        ax[1].plot(distance_vector, PIXEL_VALUE_WOUT_FBACK, label='raw data'), ax[1].set_ylabel('Pixels values'), ax[1].grid(axis='y', linestyle='--')
        ax[2].plot(distance_vector2, slim_trace, 'r', label='Convolved data')
        for j in range(len(indexes)):
            ax[2].scatter(LIST_X_POSITION[j], LIST_INTERVALS[j], s=50, c=cm[j], marker='o')
            ax[2].plot(LIST_X_POSITION[j], gaus(distance_vector2[0:40], *popt), 'k', linewidth=2)
        ax[2].scatter(distance_vector2, peaks, s=100, marker='x')
        ax[2].set_xlabel('Line position (µm)'), ax[2].set_ylabel('Convolved pixels values'), ax[2].grid(axis='y', linestyle='--'), ax[2].grid(axis='x', linestyle='--')
        ax[2] = ax[2].twiny()
        ax[2].plot([0,1.0/ROI_sampling*len(slim_trace)], [threshold,threshold], 'grey', linestyle = '--', linewidth = 2, label='threshold')
        ax[2].set_xlabel('Time (ms)'), ax[2].tick_params(axis='x', colors='g'), ax[2].spines['top'].set_color('g'), ax[2].xaxis.label.set_color('g'), ax[2].grid(axis='x', linestyle='--', color='g')
        plt.legend()

  
LIST_FILES = []
LIST_FITTED_dF_F0 = []
LIST_CONVOLVED_dF_F0 = []
BIG_LIST_AMPS = []

for i in range(len(indexes)):
    LIST_AVERAGE_INTERVAL = []
    LIST_dF_F0 = []
    LIST_CORRECTED_FLUO = []
    LIST_MEAN_BSL = []
    LIST_PEAK_VALUE = []
    LIST_AMPS = []
    
    fig = plt.figure(figsize=(15,10), tight_layout=True)
    gs = gridspec.GridSpec(2,9) #ADAPTER donne nombre subplot linescan, ici 10
    
    for f in range(len(folders_files)):
        if folders_files[f].startswith('20190801_linescan2'): #ADAPTER nom dossier
            #print('%s contains the linescans folders'%folders_files[f])
            path2 = '%s/%s'%(path, folders_files[f])
            folders = sorted(os.listdir(path2))
            
            for folder in range(len(folders)):
                path3 = '%s/%s/%s'%(path, folders_files[f], folders[folder])
                the_folder = os.listdir(path3)
            
                for file in the_folder:
                    if 'ch_1.tif' in file:
                        LIST_FILES.append('%s/%s/%s/%s'%(path, folders_files[f], folders[folder], file))
                        data = plt.imread('%s/%s/%s/%s'%(path, folders_files[f], folders[folder], file)).transpose(1,0,2)
                        real_data = data[:,:,0]
                        
                        corrected_fluo = real_data
                        LIST_CORRECTED_FLUO.append(corrected_fluo)
                        
                        time_vector = np.arange(0, 1.0/sampling_rate*len(corrected_fluo[1]), 1.0/sampling_rate)
                       
                        button = corrected_fluo[POSITIVE_2SIGMA_INDEXES[i]-5 : NEGATIVE_2SIGMA_INDEXES[i]+5, :]
                        average_interval = np.mean(button, axis=0)
                        LIST_AVERAGE_INTERVAL.append(average_interval)
        
        
            #print('--------')      
            #print('Button %s at index %s'%(i, indexes[i]))

            average_button = np.mean(LIST_AVERAGE_INTERVAL, axis=0)
            
            #On determine les limites de l'intervalle de la baseline
            f0_start = np.ravel(np.where(time_vector >= 0.2))[0] 
            f0_stop = np.ravel(np.where(time_vector >= 0.45))[0] 
            f0 = np.mean(average_button[f0_start:f0_stop])
            
            #Determination du dF/F0
            dF_F0 = (average_button-f0)/f0
            LIST_dF_F0.append(dF_F0)
        
            #Intervalle [0:0.5] sur les raw data pour la courbe 1
            y1 = np.ravel(np.where(time_vector >= time_vector[1]))[0]
            y2 = np.ravel(np.where(time_vector <= 0.5))[-1]
            curve1 = dF_F0[y1:y2]
            popt, pcov = opt.curve_fit(func, time_vector[y1:y2], curve1)
            #print('Curve 1 coefficients:')
            #print('a = %s, b = %s, c = %s' %(popt[0], popt[1], popt[2]))
            
            #Courbe 1 etendue sur l'intervalle [0:1.9] en utilisant la fonction func
            x1 = np.ravel(np.where(time_vector >= time_vector[1]))[0]
            x2 = np.ravel(np.where(time_vector <= time_vector[-1]))[-1]
            curve_e = func(time_vector[x1:x2],*popt)
            
            #Suppression du photobleaching
            fitted_dF_F0 = dF_F0[1:-1]-curve_e
            LIST_FITTED_dF_F0.append(fitted_dF_F0)
            
            convolved_dF_F0 = np.convolve(fitted_dF_F0, np.ones((10,))/10, mode='valid') #ADAPTER
            LIST_CONVOLVED_dF_F0.append(convolved_dF_F0)

            #Artefacts de stimulation
            stims = np.arange(1,1.5,0.05) #ADAPTER SELON FREQ STIM
            stim_off = stims+0.05 #ADAPTER SELON FREQ STIM
            for m,n,index in zip(stims, stim_off, range(len(stims))):
#                print ('Stim #%s'%index)
#                print ('stim starts at : %s seconds'%l)
#                print ('stim stops at : %s seconds'%m)
#                print('------')
             
                bsl1 = np.ravel(np.where(time_vector[5:-1] <= m-0.01))[-1]
                bsl2 = np.ravel(np.where(time_vector[5:-1] <= m))[-1]
                bsl3 = np.ravel(np.where(time_vector[5:-1] >= m))[0]
                bsl4 = np.ravel(np.where(time_vector[5:-1] <= n))[-1]
                
                mean_bsl = np.mean(convolved_dF_F0[bsl1:bsl2])
                peak_value = np.max(convolved_dF_F0[bsl3:bsl4])
                LIST_MEAN_BSL.append(mean_bsl)
                LIST_PEAK_VALUE.append(peak_value)
                
            for p,q in zip(range(len(LIST_MEAN_BSL)), range(len(LIST_PEAK_VALUE))):
                amps = LIST_PEAK_VALUE[q]-LIST_MEAN_BSL[p]
                LIST_AMPS.append(amps)
    
        
            for j in range(len(LIST_CORRECTED_FLUO)):
                ax1 = fig.add_subplot(gs[0,j])
                ax1.imshow(LIST_CORRECTED_FLUO[j]), ax1.set_xticks(np.arange(0, len(corrected_fluo[1])+100, 300))
                rectangle = patches.Rectangle((0, POSITIVE_2SIGMA_INDEXES[i]), len(corrected_fluo[1]), (NEGATIVE_2SIGMA_INDEXES[i])-(POSITIVE_2SIGMA_INDEXES[i]), linewidth=0.9, edgecolor='r', facecolor='none')
                ax1.add_patch(rectangle)
            ax2 = fig.add_subplot(gs[1,:])
#            ax2.plot(time_vector[1:-1], dF_F0[0:-1], 'k', alpha=0.4), ax2.set_ylabel('dF/F0')
            ax2.plot([time_vector[0], time_vector[-1]], [0,0], 'r', linestyle='--', linewidth=2), ax2.set_xlabel('Time (s)'), ax2.set_ylabel('dF/F0'), ax2.grid(True, linestyle='--')
#            ax2.plot(time_vector[1:-1], fitted_dF_F0, color='c', alpha=0.4, label='Fitted curve')
            ax2.plot(time_vector[10:-1], convolved_dF_F0, cm[i], label='Convolved data')
            ax2.plot(time_vector[1:-1], curve_e)
#            print(f_back)
            for stim in stims:
                ax2.plot([stim,stim],[-0.3,0.3], 'k')
            plt.title('BUTTON %s AT INDEX %s'%(i, indexes[i]))
#    print(LIST_PEAK_VALUE)       
    BIG_LIST_AMPS.append(LIST_AMPS)


fig, ax = plt.subplots(len(LIST_CONVOLVED_dF_F0), 1, figsize=(15,10), sharex=True, sharey=True, tight_layout=True)
for k in range(len(LIST_CONVOLVED_dF_F0)):
    ax[k].plot([time_vector[0], time_vector[-1]], [0,0], 'r', linestyle='--', linewidth=2)
    ax[k].plot([1,1], [-0.3,0.5], 'k', linestyle='--', linewidth=2)
    ax[k].plot(time_vector[10:-1], LIST_CONVOLVED_dF_F0[k], cm[k]), ax[-1].set_xlabel('Time (s)', fontsize=15), ax[-1].set_ylabel('dF/F0', fontsize=15), ax[k].grid(True, linestyle='--')
    ax[k].set_title('BUTTON %s AT INDEX %s'%(k, indexes[k]))

plt.figure(figsize=(15,10))
global_convolved_average = np.mean([LIST_CONVOLVED_dF_F0[1], LIST_CONVOLVED_dF_F0[3]], axis=0) #ADAPTER PROFILS A MOYENNER
print('Button 3 is not a button', 'Button 2 has not been taken into account in the global average')
plt.plot([time_vector[0], time_vector[-1]], [0,0], 'r', linestyle='--', linewidth=2)
plt.plot(time_vector[10:-1], global_convolved_average, 'c'), plt.xlabel('Time(s)', fontsize=15), plt.ylabel('dF/F0', fontsize=15), plt.grid(True, linestyle='--')
plt.setp(plt.title('CONVOLVED AVERAGE OF ALL BUTTONS', fontsize=20), color='c')

#plt.savefig(r'D:\Theo.ROSSI\GluSnFR\Virus_AAV.hSynap.SF-iGluSnFR_S72A\AAVDJ.GluSnFR-S72A\2019_05_30\{}.pdf'.format(file))

df = pd.DataFrame([BIG_LIST_AMPS[0], BIG_LIST_AMPS[1]], index=None, columns=None)
print(df)
with pd.ExcelWriter('D:/Theo.ROSSI/GluSnFR/Virus_AAV.hSynap.SF-iGluSnFR_S72A/AAVDJ.GluSnFR-S72A/AAVDJ.GluSnFR-S72A_Amp_20190801_linescan2.xlsx') as writer:
    df.to_excel(writer, sheet_name='20190801_linescan2', header=False, index=False) #ADAPTER SHEET_NAME
