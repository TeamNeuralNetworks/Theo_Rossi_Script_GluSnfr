# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 18:19:32 2020

@author: Theo.ROSSI
"""


def gauss_func(mu,sigma,bins):
    y = ((1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * (1 / sigma * (bins - mu))**2)) 
    return y


def func_mono_exp(x, a, b, c, d):
    return a * np.exp(-(x-b)/c) + d



def MAD(a,axis=None):
     '''
     Computes median absolute deviation of an array along given axis
     '''
     #Median along given axis but keep reduced axis so that result can still broadcast along a

     med = np.nanmedian(a, axis=axis, keepdims=True)
     mad = np.nanmedian(np.abs(a-med),axis=axis) #MAD along the given axis

     return mad
 
  
    
def bootstrap_patterns(patterns, run=1000, N=12, input_method='average', output_method='average'):
    
    '''
    Bootstraps synaptic patterns and returns median or average pattern
    
    patterns (list of arrays) : the patterns (data)
    run (int) : the amount of runs for average/median
    N (int) : number of draws for each cycle
    input_method (str) : 'median' , 'average' stores median or average value for each run 
    output_method (str) : 'median' or 'average' : returns medianed or averaged pattern
    
    '''
    
    endCycle = []
    
    for i in range(run): 
        
        temp = []
        
        for j in range(N):
        
            randIndex = np.random.randint(0,len(patterns),size=1)[0]
                        
            temp.append(patterns[randIndex])
            
            if len(temp) == N:
                pass
            else:
                continue
                
        if input_method == 'median' : 
            endCycle.append(np.nanmedian(temp, axis=0))
        
        elif input_method == 'average' : 
            endCycle.append(np.nanmean(temp, axis=0))


    if output_method == 'median': 
        out_bootstrap = np.nanmedian(endCycle, axis=0) 
        out_deviation = MAD(endCycle, axis=0)
        
    elif output_method == 'average': 
        out_bootstrap = np.nanmean(endCycle, axis=0)
        out_deviation = np.nanstd(endCycle, axis=0)
        
    return np.asarray(out_bootstrap), np.asarray(out_deviation), endCycle



def load_xls(file_xls):  
    df=pd.read_excel (file_xls, header = 0)
    
    for i in range(len(df.columns)):
        if 'Time' == df.columns[i]:
            timescale=df.iloc[:,i].values
            TIME.append(timescale)
        
        elif 'Average' == df.columns[i]:
            avg = df.iloc[:,i].values
            AVG.append(avg)
        
        else:
            sweep=df.iloc[:,i].values
            SWEEP.append(sweep)
       
    # for i in range(len(SWEEP)-1):    # this to get the same size of matrix between REC and TIME 
    #     TIME.append(timescale)
     
    # global sampling
    # sampling=float(TIME[1][1])*1000



def Fit_single_trace(Trace, Time_trace, x_start,x_end):
    
    idx_start=np.ravel(np.where(Time_trace >= x_start))[0]
    idx_stop=np.ravel(np.where(Time_trace <= x_end))[-1]
    
    # print(idx_start,idx_stop)
    x = Time_trace[idx_start:idx_stop]
    y = Trace[idx_start:idx_stop]
    x2=np.array(np.squeeze(x))
    y2=np.array(np.squeeze(y))  
    
    try:
        param_bounds=([-np.inf,0.,0.,-1000.],[np.inf,1.,10.,1000.])      # be careful ok for seconds. If millisec change param 2 and 3
        popt, pcov = curve_fit(func_mono_exp, x2, y2, bounds=param_bounds) 
        # print ('tau decay =',popt[2]*1000, ' ms' )
        return popt[2], popt, idx_start, idx_stop
    except:
        print ('Fit failed')
        popt[2]= float('nan')
        popt= float('nan')
        return popt[2], popt, idx_start, idx_stop
        pass



def leak_substraction(start, stop):
    
    x1 = np.ravel(np.where(TIME[0] >= float(start)))[0]
    x2 = np.ravel(np.where(TIME[0] <= float(stop)))[-1]
    
    leak_avg = np.mean(AVG[0][x1:x2])
    avg_no_leak = AVG[0]-leak_avg
    AVG_NO_LEAK.append(avg_no_leak)
    
    for i in range(len(SWEEP)):
        leak = np.mean(SWEEP[i][x1:x2])
        sweep_no_leak = SWEEP[i]-leak
        SWEEP_NO_LEAK.append(sweep_no_leak)



def bleaching_correction(xstart, xstop):
    x1 = float(xstart)
    x2 = float(xstop)
    
    avg_tau, avg_popt, avg_start, avg_stop = Fit_single_trace(AVG_NO_LEAK[0], TIME[0], x1, x2)
    
    avg_bleaching = func_mono_exp(TIME[0], *avg_popt)
    avg_no_bleach = AVG_NO_LEAK[0] - avg_bleaching
    AVG_NO_BLEACH.append(avg_no_bleach)

    for i in range(len(SWEEP_NO_LEAK)):
        local_tau, local_popt, idxstart, idxstop = Fit_single_trace(SWEEP_NO_LEAK[i], TIME[0], x1, x2)
        
        bleaching = func_mono_exp(TIME[0], *local_popt)
        sweep_no_bleach = SWEEP_NO_LEAK[i] - bleaching
        SWEEP_NO_BLEACH.append(sweep_no_bleach)



def cursors(freq, start):
    start_stim = float(start)-0.001
    if '3pulses' in freq:
        if '20Hz' in freq:
            st = np.arange(start_stim, start_stim+0.04, 0.02)
        elif '50Hz' in freq:
            st = np.arange(start_stim, start_stim+0.04, 0.02)
        
    if '10pulses' in freq:
        if '20Hz' in freq:
            st = np.arange(start_stim, start_stim, 0.015)
        elif '50Hz' in freq:
            st = np.arange(start_stim, start_stim+0.04, 0.02)

    return st



def values_extraction(freq, start, time, trace):
    a = cursors(freq, start)
    
    x1 = np.ravel(np.where(time[0] >= a[0]))[0]
    x2 = np.ravel(np.where(time[0] <= a[1]))[-1]

    for tr in range(len(trace)):
        if start == value[5]:
            amp1 = trace[tr][x1:x2]
            AMPS1.append(amp1)

        if start == value[6]:
            noise = trace[tr][0:x2]
            NOISE.append(noise)
 


if __name__ == '__main__':
    
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit
    import PySimpleGUI as sg
    import numpy as np
    import pandas as pd
    from scipy import stats
    import math
    import random

    SWEEP, AVG, TIME = [], [], []
    SWEEP_NO_LEAK, AVG_NO_LEAK = [], []
    SWEEP_NO_BLEACH, AVG_NO_BLEACH = [], []
    
    NOISE, AMPS1 = [], []
    NOISE_MOM, AMP_MOM = [], []
  
    sg.theme('DarkBlue')

    layout = [[sg.Text('Excel File Name')],
              [sg.InputText(size=(35,1)), sg.FileBrowse()],
              [sg.Button('Start File')],
              [sg.Frame(layout=[
              [sg.Text('Leak'), sg.InputText(size=(4,1),default_text='0.4'), sg.Text('to'), sg.InputText(size=(4,1),default_text='0.45'), sg.Text('sec'), sg.Button('Substraction')],
              [sg.Text('Photobleaching'), sg.InputText(size=(4,1)), sg.Text('to'), sg.InputText(size=(4,1)), sg.Text('sec'), sg.Button('Correction')]], title='Correction settings', relief=sg.RELIEF_SUNKEN)],
              [sg.Frame(layout=[
              [sg.Text('Stim window')],
              [sg.InputText(size=(4,1),default_text='0.5'), sg.InputText(size=(4,1),default_text='0.015'), sg.Text('sec')],
              [sg.Text('Noise window')],
              [sg.InputText(size=(4,1),default_text='0.1'), sg.InputText(size=(4,1),default_text='0.4'), sg.Text('sec')],
              [sg.Text('Excel name'), sg.InputText(size=(30,1))],
              [sg.Button('To excel')]], title='Windows', relief=sg.RELIEF_SUNKEN), sg.Button('GO')],
              [sg.Button('Clear', button_color=('white','darkred'))]]
             
                        
    window = sg.Window('BOOTSTRAP', layout, location=(0,0))
    
    while True:
        event, value = window.read()
        
        try:
            if event in (None, 'Close'):
                
                plt.close('all')
                break
                 
            if event == 'Start File':
                
                load_xls(value[0])
                
                fig = plt.figure(figsize=(18,5), tight_layout=True)
                gs = fig.add_gridspec(1,3)
                ax1 = fig.add_subplot(gs[:,0])
                ax2 = fig.add_subplot(gs[:,1], sharey=ax1)
                ax3 = fig.add_subplot(gs[:,2], sharey=ax1)
                
                title = str(value[0]).split('/')
                ax1.set_title('{}'.format(title[-1]), fontsize=8)
                ax1.set_xlabel('Time(sec)')
                ax1.set_ylabel('dF/F0')
                ax2.set_title('Noise')
                ax3.set_title('Peak1')
                
                
                for trace in range(len(SWEEP)):
                    ax1.plot(TIME[0], SWEEP[trace], 'k', alpha=0.1)
                
                ax1.plot(TIME[0], AVG[0], 'b', linewidth=2, alpha=0.5)
                ax1.plot([TIME[0][0],TIME[0][-1]], [0,0], 'k', linestyle='--')
            
            
            if event == 'Substraction':
                
                ax1.clear()
                
                leak_substraction(value[1], value[2])
                
                for trace in range(len(SWEEP_NO_LEAK)):
                    ax1.plot(TIME[0], SWEEP_NO_LEAK[trace], 'k', alpha=0.1)
                
                ax1.plot(TIME[0], AVG_NO_LEAK[0], 'b', linewidth=2, alpha=0.5)
                ax1.plot([TIME[0][0],TIME[0][-1]], [0,0], 'k', linestyle='--')
                
                
            if event == 'Correction':
                
                ax1.clear()
                
                bleaching_correction(value[3], value[4])
                
                for trace in range(len(SWEEP_NO_BLEACH)):
                    ax1.plot(TIME[0], SWEEP_NO_BLEACH[trace], 'k', alpha=0.1)
                
                ax1.plot(TIME[0], AVG_NO_BLEACH[0], 'b', linewidth=2, alpha=0.5)
                ax1.plot([TIME[0][0],TIME[0][-1]], [0,0], 'k', linestyle='--')
            
            
            if event == 'GO':
                
                a = cursors(value[0], value[5])
                b = cursors(value[0], value[6])
                
                if SWEEP != [] and SWEEP_NO_LEAK == [] and SWEEP_NO_BLEACH == []:
                    for x,y in zip(a, b):
                        ax1.plot([x,x], [np.min(SWEEP), np.max(SWEEP)], c='cyan', linestyle='--', linewidth=1)
                        ax1.plot([y,y], [np.min(SWEEP), np.max(SWEEP)], c='k', linestyle='--', linewidth=1, alpha=0.5)
                    
                    values_extraction(value[0], value[5], TIME, SWEEP)
                    values_extraction(value[0], value[6], TIME, SWEEP)
                
                elif SWEEP != [] and SWEEP_NO_LEAK !=[] and SWEEP_NO_BLEACH == []:
                    for x,y in zip(a, b):
                        ax1.plot([x,x], [np.min(SWEEP_NO_LEAK), np.max(SWEEP_NO_LEAK)], c='cyan', linestyle='--', linewidth=1)
                        ax1.plot([y,y], [np.min(SWEEP_NO_LEAK), np.max(SWEEP_NO_LEAK)], c='k', linestyle='--', linewidth=1, alpha=0.5)
                    
                    values_extraction(value[0], value[5], TIME, SWEEP_NO_LEAK)
                    values_extraction(value[0], value[6], TIME, SWEEP_NO_LEAK)
                
                elif SWEEP != [] and SWEEP_NO_LEAK !=[] and SWEEP_NO_BLEACH !=[]:
                    for x,y in zip(a, b):
                        ax1.plot([x,x], [np.min(SWEEP_NO_BLEACH), np.max(SWEEP_NO_BLEACH)], c='cyan', linestyle='--', linewidth=1)
                        ax1.plot([y,y], [np.min(SWEEP_NO_BLEACH), np.max(SWEEP_NO_BLEACH)], c='k', linestyle='--', linewidth=1, alpha=0.5)
                    
                    values_extraction(value[0], value[5], TIME, SWEEP_NO_BLEACH)
                    values_extraction(value[0], value[6], TIME, SWEEP_NO_BLEACH)
                
                for ns in range(len(NOISE)):
                    
                    ax2.plot(NOISE[ns], 'k', alpha=0.1)
                    ax3.plot(AMPS1[ns], 'k', alpha=0.1)
                
                ax2.plot([0,len(NOISE[0])], [0,0], 'k', linestyle='--')
                ax3.plot([0,len(AMPS1[0])], [0,0], 'k', linestyle='--')
                
                
                fig2, ax = plt.subplots(3, math.ceil(len(NOISE)/3), figsize=(18,10), tight_layout=True)
                
                i = 0
                j = 0
                
                df_episodes = pd.DataFrame(index=None, columns=None)
                df_mom = pd.DataFrame(index=None, columns=None)
                
                NS, AMPS, FAIL, EP, AMP1 = [],[],[],[],[]
                for ns in range(len(NOISE)):
                    
                    ns_cycle = [np.mean(random.sample(NOISE[ns].tolist(),len(AMPS1[ns]))) for i in range(1000)]
                    ns_mean = np.mean(ns_cycle)
                    ns_std = np.std(ns_cycle)
                    # ns_mean, ns_std, ns_cycle = bootstrap_patterns(ns_random_points, N=len(ns_random_points))
                    amp_mean, amp_std, amp_cycle = bootstrap_patterns(AMPS1[ns], N=len(AMPS1[ns]))
                    NS.append(ns_mean)
                    
                    NOISE_MOM.append(ns_mean)
                    AMP_MOM.append(amp_mean)
                    
                    df_episodes['noise{}'.format(ns)] = ns_cycle
                    df_episodes['amp{}'.format(ns)] = amp_cycle                    
                        
                    ns_n, ns_norm_pvalue = stats.shapiro(ns_cycle)
                    amp_n, amp_norm_pvalue = stats.shapiro(amp_cycle)
                    
                    ns_l, ns_norm_l_pvalue = stats.levene(ns_cycle, amp_cycle)
                    
                    print('Episode {}'.format(ns))
                    print('Raw_mean: {:.3f} ; Bootstrap_mean: {:.3f}'.format(np.mean(NOISE[ns]), ns_mean))
                    print('Ns_mean: {:.3f} ; Ns_std: {:.3f}'.format(ns_mean, ns_std))
                    print('Amp_mean: {:.3f} ; Amp_std: {:.3f}'.format(amp_mean, amp_std))
                    print('Shapiro-Wilk test:\n noise_p = {:.3f} ; amp_p = {:.3f}'.format(ns_norm_pvalue, amp_norm_pvalue))
                    print('Levene test:\n p = {:.3f}'.format(ns_norm_l_pvalue))
                    
                    if j == math.ceil((len(NOISE)/3)):
                        i += 1
                        j = 0
                    elif j == math.ceil((2*(len(NOISE)/3))):
                        i += 1
                        j = 0
    
                    n_ns, bins_ns, patches_ns = ax[i,j].hist(ns_cycle, bins=int(np.sqrt(df_episodes.shape[0])/2), density=True, facecolor='k', alpha=0.2)
                    n_amp, bins_amp, patches_amp = ax[i,j].hist(amp_cycle, bins=int(np.sqrt(df_episodes.shape[0])/2), density=True, facecolor='cyan', alpha=0.4)
                    y_ns = gauss_func(ns_mean,ns_std,bins_ns)
                    y_amp = gauss_func(amp_mean,amp_std,bins_amp)
                    
                    ax[i,j].plot(bins_ns,y_ns,color='k',alpha=0.5)
                    ax[i,j].plot(bins_amp,y_amp,color='cyan',alpha=0.6)
                    
                    z_score = (amp_mean - ns_mean)/ns_std
                    print('Z_SCORE:\n{:.3f}'.format(z_score))
                    
                    amp = amp_mean - ns_mean
                    AMPS.append(amp)
                    
                    if z_score < 1.96:
                        FAIL.append(amp)
                        ax[i,j].text(0,0, 'zscore: {:.2f}'.format(z_score), fontsize=10)
                        print('Not statistically significant')
                    else:
                        ax[i,j].text(0,0, 'zscore: {:.2f}'.format(z_score), fontsize=10, color='red')
                        print('Statistically significant')
                        EP.append('Episode {}'.format(ns))
                        AMP1.append(amp)
                
                    
                        
                #     if ns_norm_pvalue <= 0.05 or amp_norm_pvalue <= 0.05:
                #         n, u_pvalue = stats.mannwhitneyu(ns_cycle, amp_cycle)
                #         print('Mann-Whitney U test:\n p = {:.3f}'.format(u_pvalue))
                #         ax[i,j].text(0,0, 'p = {:.3f}'.format(u_pvalue), fontsize=10)
                        
                #         if u_pvalue <= 0.05:
                #             print('Statistically significant')
                #         else:
                #             print('No statistically significant')
                    
                #     elif ns_norm_pvalue > 0.05 and amp_norm_pvalue > 0.05 and ns_norm_l_pvalue <= 0.05:
                #         t, t_pvalue = stats.ttest_ind(ns_cycle, amp_cycle, equal_var=False)
                #         print('Welch T test:\n p = {:.3f}'.format(t_pvalue))
                #         ax[i,j].text(0,0, 'p = {:.3f}'.format(t_pvalue), fontsize=10)
                    
                #         if t_pvalue <= 0.05:
                #             print('Statistically significant')
                #         else:
                #             print('No statistically significant')
                                         
                #     elif ns_norm_pvalue > 0.05 and amp_norm_pvalue > 0.05 and ns_norm_l_pvalue > 0.05:
                #         t, t_pvalue = stats.ttest_ind(ns_cycle, amp_cycle, equal_var=True)
                #         print('Student T test:\n p = {:.3f}'.format(t_pvalue))
                #         ax[i,j].text(0,0, 'p = {:.3f}'.format(t_pvalue), fontsize=10)
                    
                #         if t_pvalue <= 0.05:
                #             print('Statistically significant')
                #         else:
                #             print('No statistically significant')
                    
                    print('-------')
                    
                    ax[i,j].set_title('Episode {}\n NOISE µ:{:.2f}, σ:{:.2f}\n PEAK1 µ:{:.2f}, σ:{:.2f}'.format(ns, ns_mean, ns_std, amp_mean, amp_std))       
                    
                    j += 1
                
                percentage_failures = (len(FAIL)/len(NOISE))*100
                variance = np.var(AMPS) - np.var(NS)
                cv = np.sqrt(variance)/np.nanmean(AMPS)
                df_ep = pd.concat((pd.DataFrame(EP), pd.DataFrame(AMP1)),axis=1)
                df_ep.columns = ['Episode','AMP1']
                df_ep['CV'] = cv
                df_ep['PercFail'] = percentage_failures
                
                # plt.figure()
                # plt.hist(NOISE_MOM, bins=7, facecolor='k', alpha=0.2)
                # plt.hist(AMP_MOM, bins=7, facecolor='orange', alpha=0.4)
                # plt.title('NOISE µ:{:.2f}, σ:{:.2f}\nPEAK1 µ:{:.2f}, σ:{:.2f}'.format(np.nanmean(NOISE_MOM), np.nanstd(NOISE_MOM), np.nanmean(AMP_MOM), np.nanstd(AMP_MOM)))
                
                # z_score_mom = (np.nanmean(AMP_MOM) - np.nanmean(NOISE_MOM))/np.nanstd(NOISE_MOM)
                
                # if z_score_mom < 2.576:
                #     plt.text(0,0, 'zscore: {:.2f}'.format(z_score_mom), fontsize=10)
                # else:
                #     plt.text(0,0, 'zscore: {:.2f}'.format(z_score_mom), fontsize=10, color='red')
           
            if event == 'To excel':
                with pd.ExcelWriter('E:\AAVDJ.GluSnFR-S72A\AAVDJ.GluSnFR-S72A_Bootstrap\{}_histograms.xlsx'.format(value[7])) as writer:
                    df_episodes.to_excel(writer, header=True, index=False)
                    writer.save()
                with pd.ExcelWriter('E:\AAVDJ.GluSnFR-S72A\AAVDJ.GluSnFR-S72A_Bootstrap\{}_data.xlsx'.format(value[7])) as writer:
                    df_ep.to_excel(writer, header=True, index=False)
                    writer.save()
                    
                print('DONE')
                
            if event == 'Clear':
                plt.close()
                fig.close()
                
                SWEEP, AVG, TIME = [], [], []
                SWEEP_NO_LEAK, AVG_NO_LEAK = [], []
                SWEEP_NO_BLEACH, AVG_NO_BLEACH = [], []
                
                NOISE, AMPS1 = [], []
                NOISE_MOM, AMP_MOM = [], []
                
        except:
            pass
        
    window.close()