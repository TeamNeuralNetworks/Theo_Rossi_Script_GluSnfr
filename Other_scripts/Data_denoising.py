# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter




def threshold(time, trace, noise_start, noise_stop, peak_start, peak_stop):
    
    n_start = np.ravel(np.where(time >= noise_start))[0]
    n_stop = np.ravel(np.where(time <= noise_stop))[-1]
    
    mean_noise = np.mean(trace[n_start:n_stop], axis=0)
    std_noise = np.std(trace[n_start:n_stop], axis=0)
    
    p_start = np.ravel(np.where(time >= peak_start))[0]
    p_stop = np.ravel(np.where(time <= peak_stop))[-1]
    peak = np.max(trace[p_start:p_stop])
    
    thres = peak/std_noise
    
    return peak, thres




def func_mono_exp(x, a, b, c, d):
    return a * np.exp(-(x-b)/c) + d




def Fit_single_trace(time, trace, x_start, x_end):
    '''
    Computes an exponential fit on a window of a dataset values.
    The dataset has to be an excel file with variables as columns.
    
    time : time variable.
    trace : the trace the fit must be applied on.
    x_start : first limit of the window.
    x_end : second limit of the window.

    Returns
    -------
    popt : array
        optimal values for the parameters
    idx_start : int
        the first index of the window.
    idx_stop : int
        the second index of the window.

    '''
    
    idx_start = np.ravel(np.where(time >= x_start))[0]
    idx_stop = np.ravel(np.where(time <= x_end))[-1]
    
    x = time[idx_start:idx_stop]
    y = trace[idx_start:idx_stop]
    x2 = np.array(np.squeeze(x))
    y2 = np.array(np.squeeze(y))  
    
    try:
        param_bounds=([-np.inf,0.,0.,-1000.],[np.inf,1.,10.,1000.])      # be careful ok for seconds. If millisec change param 2 and 3
        popt, pcov = curve_fit(func_mono_exp, x2, y2, bounds=param_bounds, maxfev=10000) 
        return popt[2], popt, idx_start, idx_stop
    except:
        print ('Fit failed')
        popt[2]= float('nan')
        popt= float('nan')
        return popt[2], popt, idx_start, idx_stop
        pass
    
    
    

def bleaching_correction(time, trace, start, stop):
    '''
    Suppresses the exponential decay on a window of a dataset values
    
    time (array): time variable.
    trace (array or list): the trace or list of traces the fit must be applied on.
    start (int or float) : first limit of the window.
    end (int or float): second limit of the window.

    Returns
    -------
    no_bleach : list
        The list of traces corrected for the exp decay.

    '''
    
    x1 = float(start)
    x2 = float(stop)
    
    if len(trace) == 0:
        print('Bleaching failed: empty window')
        
    elif type(trace) == np.ndarray:
        tau, popt, xstart, xstop = Fit_single_trace(time, trace, x1, x2)
        bleach = func_mono_exp(time, *popt)
        no_bleach = trace-bleach

    elif type(trace) == list:
        param = [Fit_single_trace(time, trace[i], x1, x2) for i in range(len(trace))]
        bleach = [func_mono_exp(time, *param[i][1]) for i in range(len(trace))]
        no_bleach = [trace[i]-bleach[i] for i in range(len(trace))]
    return no_bleach






if __name__ == '__main__':
    # plt.close('all')  
    
    path = r'\\equipe2-nas1\Theo.ROSSI\BackupE\AAVDJ.GluSnFR-S72A\Controls\iGluSnFR.S72A_forskoline'
    freq = '20Hz'
    
    
    files = sorted(os.listdir(path))
    
    
    boutons_15 = {}
    boutons_25 = {}
    
    
    for file in tqdm(files):
        if freq in file:
            if 'converted.xlsx' in file:
                profile = pd.read_excel(f'{path}/{file}', sheet_name='Traces DF_F0')
                file = file.rsplit('_',1)[0]
                
                if '1.5mMCa' in file:
                    trace_15 = profile.iloc[:,-2:]
                    boutons_15[f'{file}'] = trace_15
                
                # if '2.5mM' in file:
                #     trace_25 = profile.iloc[:,-2:]
                #     boutons_25[f'{file}'] = trace_25
    
    
    bouton_ok = []
   
    if boutons_15 != {}:
        for name_15, data_15 in boutons_15.items():
            
            corrected_baseline_15 = bleaching_correction(data_15['Time'], savgol_filter(np.array(data_15['Average']), 9, 2), 0.01, 0.45)
            # corrected_baseline_15 = bleaching_correction(data_15['Time'], np.array(data_15['Average']), 0.01, 0.45)
            
            a1, t = threshold(data_15['Time'], corrected_baseline_15, 0.1, 0.4, 0.5, 0.51)
            
            plt.figure()
            if t <= 3.:     
                print(f'{name_15}')
                print(f'score = {t}\nNOT SIGNIFICANT')
                plt.plot(data_15['Time'], corrected_baseline_15, 'k', label='t-score = {:.2f}'.format(t))
            else:
                print(t, a1)
                print(f'{name_15}')
                print(f'score = {t}\nSIGNIFICANT')
                plt.plot(data_15['Time'], corrected_baseline_15, 'g', label='t-score = {:.2f}'.format(t))
                bouton_ok.append(corrected_baseline_15)
            # plt.plot(data_15['Time'], data_15['Average'], 'r')
            # plt.scatter(0.55, a1, color='b')
            plt.title(name_15)
            plt.legend()
            
            print('------------------')
            print('------------------')
    
    
    # if boutons_25 != {}:
    #     for name_25, data_25 in boutons_25.items():
            
    #         corrected_baseline_25 = bleaching_correction(data_25['Time'], savgol_filter(np.array(data_25['Average']), 9, 2), 0.01, 0.45)
    #         # corrected_baseline_25 = bleaching_correction(data_25['Time'], np.array(data_25['Average']), 0.01, 0.45)
            
    #         a1, t = threshold(data_25['Time'], corrected_baseline_25, 0.1, 0.4, 0.5, 0.51)
            
    #         plt.figure()
    #         if t <= 3:
    #             print(f'{name_25}')
    #             print(f'score = {t}\nNOT SIGNIFICANT')
    #             plt.figure()
    #             plt.plot(data_25['Time'], corrected_baseline_25, 'k', label='t-score = {:.2f}'.format(t))
    #         else:
    #             print(f'{name_25}')
    #             print(f'score = {t}\nSIGNIFICANT')
    #             plt.plot(data_25['Time'], corrected_baseline_25, 'g', label='t-score = {:.2f}'.format(t))
    #             bouton_ok.append(corrected_baseline_25)
    #         # plt.plot(data_15['Time'], data_15['Average'], 'r')
    #         # plt.scatter(0.55, a1, color='b')
    #         plt.title(name_25)
    #         plt.legend()
            
    #         print('------------------')
    #         print('------------------')

      