# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 17:01:43 2025

@author: Anthime.PERROT
"""

# -*- coding: utf-8 -*-
"""
Bootstrap Batch Processing Script
Modifié pour traitement automatique par lots

@author: Theo.ROSSI modified by Anthime Perrot
"""

import os
# Fix pour les erreurs MKL - DOIT ÊTRE EN PREMIER
os.environ['MKL_THREADING_LAYER'] = 'GNU'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
import PySimpleGUI as sg
import numpy as np
import pandas as pd
from scipy import stats
import math
import random
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
import glob
from pathlib import Path

filt = 5  #Smoothing value, before 9. Here 1 because with Antoine script smoothing is already done.



# Toutes vos fonctions existantes (copiées telles quelles)
def gauss_func(mu,sigma,bins):
    '''
    mu : mean
    sigma : standard deviation
    bins : histogram bins

    Returns
    -------
    y : gaussian function of a histogram dataset

    '''
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

def bootstrap_patterns(patterns, run=5000, N=12, input_method='average', output_method='average'):
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
    '''
    Version sécurisée du chargement Excel
    '''
    try:
        print(f"Ouverture du fichier: {file_xls}")
        
        # Essayer plusieurs moteurs Excel
        engines = ['openpyxl', 'xlrd']
        df = None
        
        for engine in engines:
            try:
                df = pd.read_excel(file_xls, sheet_name='Traces DF_F0', header=0, engine=engine)
                print(f"Fichier ouvert avec le moteur: {engine}")
                break
            except Exception as e:
                print(f"Échec avec {engine}: {e}")
                continue
        
        if df is None:
            # Dernier essai avec le premier sheet
            df = pd.read_excel(file_xls, sheet_name=0, header=0)
        
        if df is None or df.empty:
            raise ValueError("Impossible d'ouvrir le fichier Excel ou fichier vide")
        
        # Nettoyer les données
        df = df.dropna(how='all')
        df = df.replace([np.inf, -np.inf], np.nan)
        
        print(f"Colonnes trouvées: {list(df.columns)}")
        print(f"Forme des données: {df.shape}")
        
        polyorder = 2 #Polyorder must be less thant filt. Before 2.
        timescale = None
        average = None
        SWEEPS = []
        
        for col_name in df.columns:
            col_data = df[col_name].dropna().values
            
            if len(col_data) < 10:
                continue
                
            if 'Time' in str(col_name) or 'time' in str(col_name).lower():
                timescale = col_data
                
            elif 'Average' in str(col_name) or 'average' in str(col_name).lower():
                average = col_data
                if len(col_data) >= filt:
                    average = savgol_filter(col_data, filt, polyorder)
                else:
                    average = col_data
                    
            else:
                if len(col_data) >= filt:
                    sweep = savgol_filter(col_data, filt, polyorder)
                else:
                    sweep = col_data
                SWEEPS.append(sweep)
        
        if timescale is None or average is None or len(SWEEPS) == 0:
            raise ValueError("Colonnes Time, Average ou sweeps manquantes")
        
        # Ajuster les longueurs
        min_len = min(len(timescale), len(average), min(len(s) for s in SWEEPS))
        timescale = timescale[:min_len]
        average = average[:min_len]
        SWEEPS = [s[:min_len] for s in SWEEPS]
        
        print(f"Données chargées: {len(timescale)} points, {len(SWEEPS)} sweeps")
        return timescale, average, SWEEPS
        
    except Exception as e:
        print(f"ERREUR lors du chargement: {e}")
        return None, None, None

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
        popt = [float('nan')] * 4
        return float('nan'), popt, idx_start, idx_stop

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
        print('bleaching correction is done')
    return no_bleach

def leak_subtraction(time, trace, start, stop):
    ''' 
    Suppresses the offset on a window of a dataset values
    time (array): time variable.
    trace (array or list): the trace or list of traces.
    start (int or float) : first limit of the window.
    end (int or float): second limit of the window.

    Returns
    -------
    no_leak : list
        the list of traces without offset.

    '''
    
    x1 = np.ravel(np.where(time >= float(start)))[0]
    x2 = np.ravel(np.where(time <= float(stop)))[-1]
    
    if len(trace) == 0:
        print('Leak failed: empty window')
        
    elif type(trace) == np.ndarray:
        leak = np.mean(trace[x1:x2])
        no_leak = trace-leak
        
    elif type(trace) == list:
        leak = [np.mean(trace[i][x1:x2]) for i in range(len(trace))]
        no_leak = [trace[i]-leak[i] for i in range(len(trace))]
        
        print('substraction is done')
    return no_leak

def residual_sublimation(time, trace, start, stop, freq, n_peaks):
    ''' 
    Suppresses the residual preceding the onset of the peak from the peak
    time (array): time variable.
    trace (array or list): the trace or list of traces.
    start (int or float) : first limit of the window.
    end (int or float): second limit of the window.

    Returns
    -------
    no_res : list
        the list of traces containing peaks without res.

    '''
    
    if len(trace) == 0:
        print('Res sublimation failed: empty window')
        
    elif type(trace) == np.ndarray:
        x1 = np.ravel(np.where(time >= start))[0]
        x2 = np.ravel(np.where(time <= stop))[-1]
        x2_bis = np.ravel(np.where(time <= stop))[-1]
        bsl = trace[0:x2]
        PEAKS_NO_RES = []
        for i in range(n_peaks):
            res = np.mean(trace[x1:x2])
            if freq == '20':
                stop += 0.05
                x2_bis = np.ravel(np.where(time <= stop))[-1]
                peak_no_res = trace[x2:x2_bis]-res
                start += 0.05
            elif freq == '50':  
                stop += 0.02
                x2_bis = np.ravel(np.where(time <= stop))[-1]
                peak_no_res = trace[x2:x2_bis]-res
                start += 0.02
            elif freq == '100':  
                stop += 0.01
                x2_bis = np.ravel(np.where(time <= stop))[-1]
                peak_no_res = trace[x2:x2_bis]-res
                start += 0.01
                
            x1 = np.ravel(np.where(time >= start))[0]
            x2 = np.ravel(np.where(time <= stop))[-1]
            PEAKS_NO_RES.append(peak_no_res)
            
        end = np.append(np.hstack(PEAKS_NO_RES), trace[x2:])
        no_res = np.append(bsl, end)
        
    elif type(trace) == list:
        x2_bsl = np.ravel(np.where(time <= stop))[-1]
        no_res = []
        for i in range(len(trace)):
            alpha = start
            omega = stop
            x1 = np.ravel(np.where(time >= alpha))[0]
            x2 = np.ravel(np.where(time <= omega))[-1]
            x2_bis = np.ravel(np.where(time <= omega))[-1]
            bsl = trace[i][0:x2_bsl]
            PEAKS_NO_RES = []
            for j in range(n_peaks):
                res = np.mean(trace[i][x1:x2])
                if freq == '20':
                    omega += 0.05
                    x2_bis = np.ravel(np.where(time <= omega))[-1]
                    peak_no_res = trace[i][x2:x2_bis]-res
                    alpha += 0.05
                elif freq == '50':  
                    omega += 0.02
                    x2_bis = np.ravel(np.where(time <= omega))[-1]
                    peak_no_res = trace[i][x2:x2_bis]-res
                    alpha += 0.02
                elif freq == '100':
                    omega += 0.01
                    x2_bis = np.ravel(np.where(time <= omega))[-1]
                    peak_no_res = trace[i][x2:x2_bis]-res
                    alpha += 0.01
                
                x1 = np.ravel(np.where(time >= alpha))[0]
                x2 = np.ravel(np.where(time <= omega))[-1]
                PEAKS_NO_RES.append(peak_no_res)
 
            end = np.append(np.hstack(PEAKS_NO_RES), trace[i][x2:])
            no_res_trace = np.append(bsl, end)
            no_res.append(no_res_trace)
        print('sublimation is done')
    return no_res

def values_extraction(time, trace, start, stop):    
    '''
    Exctracts values on a window of a dataset values
    time (array): time variable.
    trace (array or list): the trace or list of traces.
    start (int or float) : first limit of the window.
    end (int or float): second limit of the window.

    Returns
    -------
    windows : list
        The list of values in the extracted window.

    '''
    
    x1 = np.ravel(np.where(time >= float(start)))[0]
    x2 = np.ravel(np.where(time <= float(stop)))[-1]
    
    if len(trace) == 0:
        print('Extraction failed: empty window')
        
    elif type(trace) == np.ndarray:
        windows = trace[x1:x2]
        
    elif type(trace) == list:
        windows = [trace[i][x1:x2] for i in range(len(trace))]
        
    return windows

def process_single_file(file_path, output_folder, parameters):
    """
    Traite un seul fichier avec les paramètres donnés
    """
    try:
        print(f"\n=== Traitement du fichier: {file_path} ===")
        
        # Chargement du fichier
        time, avg, sweeps = load_xls(file_path)
        
        if time is None or avg is None or sweeps is None:
            print(f"Erreur lors du chargement de {file_path}")
            return False
        
        # Application des corrections automatiques
        print("Application des corrections...")
        
        # 1. Correction du photobleaching
        avg_corrected = bleaching_correction(time, avg, parameters['photo_start'], parameters['photo_stop'])
        sweeps_corrected = bleaching_correction(time, sweeps, parameters['photo_start'], parameters['photo_stop'])
        
        # 2. Soustraction du leak
        avg_corrected = leak_subtraction(time, avg_corrected, parameters['leak_start'], parameters['leak_stop'])
        sweeps_corrected = leak_subtraction(time, sweeps_corrected, parameters['leak_start'], parameters['leak_stop'])
        
        # 3. Sublimation résiduelle
        avg_corrected = residual_sublimation(time, avg_corrected, parameters['res_start'], parameters['res_stop'], 
                                           parameters['frequency'], parameters['n_peaks'])
        sweeps_corrected = residual_sublimation(time, sweeps_corrected, parameters['res_start'], parameters['res_stop'], 
                                              parameters['frequency'], parameters['n_peaks'])
        
        # Extraction des fenêtres pour l'analyse
        file_name = Path(file_path).stem
        a = parameters['peak_start']
        b = parameters['peak_stop']
        
        WINDOWS_NS, WINDOWS_AMP = [], []
        
        for i in range(parameters['n_peaks']):
            windows_amp = values_extraction(time, sweeps_corrected, a, b)
            windows_ns = values_extraction(time, sweeps_corrected, parameters['noise_start'], parameters['noise_stop'])
            
            if parameters['frequency'] == '20':
                a += 0.05
                b += 0.05
            elif parameters['frequency'] == '50':  
                a += 0.02
                b += 0.02
            elif parameters['frequency'] == '100':  
                a += 0.01
                b += 0.01
                
            WINDOWS_NS.append(windows_ns)
            WINDOWS_AMP.append(windows_amp)
        
        # Analyse bootstrap
        print("Analyse bootstrap...")
        
        wb_hist = Workbook()
        wb_data = Workbook()
        
        ALL_FAILS = []
        
        for peak in range(parameters['n_peaks']):
            EP, PEAK, ZSCORE, AMP, NS_AMP, STD_NS, STD_AMP = [], [], [], [], [], [], []
            
            df_episodes = pd.DataFrame(index=None, columns=None)
            
            print(f'BOOTSTRAP PEAK{peak+1}')
            
            for item in range(len(WINDOWS_AMP[peak])):
                EP.append(f'Episode {item}')
            
                amp_mean, amp_std, amp_cycle = bootstrap_patterns(WINDOWS_AMP[peak][item], N=len(WINDOWS_AMP[peak][item]))
                
                ns_cycle = [np.mean(random.sample(windows_ns[item].tolist(), len(WINDOWS_AMP[peak][item]))) 
                           for i in range(len(amp_cycle))]
                ns_mean = np.mean(ns_cycle)
                ns_std = np.std(ns_cycle)
                
                NS_AMP.append(ns_mean)
                STD_AMP.append(amp_std)
                STD_NS.append(ns_std)
                
                amp = amp_mean - ns_mean
                AMP.append(amp)
                
                df_episodes[f'noise{item}'] = ns_cycle
                df_episodes[f'amp{item}'] = amp_cycle
                
                z_score = amp/ns_std
                ZSCORE.append(z_score)
                
                if z_score <= 3:
                    PEAK.append('Fail')
                else:
                    PEAK.append('Success')
            
            percentage_failures = (PEAK.count('Fail')/len(WINDOWS_AMP[peak]))*100
            ALL_FAILS.append(percentage_failures)
            
            df_ep = pd.concat((pd.DataFrame(EP), pd.DataFrame(PEAK), pd.DataFrame(AMP), pd.DataFrame(NS_AMP),
                               pd.DataFrame(STD_AMP), pd.DataFrame(STD_NS), pd.DataFrame(ZSCORE)), axis=1)
            df_ep.columns = ['Episode','PEAK','AMP','AMPns','Std_amp','Std_ns','Zscore']
            df_ep['PercFail'] = percentage_failures
            
            sheet_hist = wb_hist.create_sheet(f'PEAK{peak+1}')
            sheet_data = wb_data.create_sheet(f'PEAK{peak+1}')
            
            for r in dataframe_to_rows(df_episodes, index=False, header=True):
                sheet_hist.append(r)

            for r in dataframe_to_rows(df_ep, index=False, header=True):
                sheet_data.append(r)
        
        # Sauvegarde
        wb_hist.remove(wb_hist['Sheet'])
        wb_data.remove(wb_data['Sheet'])
        
        hist_path = os.path.join(output_folder, f'{file_name}_histograms_bootstrap.xlsx')
        data_path = os.path.join(output_folder, f'{file_name}_data_bootstrap.xlsx')
        
        wb_hist.save(hist_path)
        wb_data.save(data_path)
        
        print(f"Fichier traité avec succès: {file_name}")
        print(f"Fichiers sauvegardés: {hist_path}, {data_path}")
        
        return True
        
    except Exception as e:
        print(f"Erreur lors du traitement de {file_path}: {e}")
        return False

def batch_process():
    """
    Interface pour le traitement par lots
    """
    sg.theme('DarkBlue')

    layout = [
      [sg.Text('Dossier contenant les fichiers Excel:')],
      [sg.InputText(size=(50,1), key='input_folder'), sg.FolderBrowse()],
      [sg.Text('Dossier de sortie:')],
      [sg.InputText(size=(50,1), key='output_folder'), sg.FolderBrowse()],
      [sg.Frame('Paramètres de correction', [
          [sg.Text('Photobleaching:'), sg.InputText('0.01', size=(6,1), key='photo_start'), 
           sg.Text('à'), sg.InputText('0.45', size=(6,1), key='photo_stop'), sg.Text('sec')],
          [sg.Text('Leak:'), sg.InputText('0.35', size=(6,1), key='leak_start'), 
           sg.Text('à'), sg.InputText('0.45', size=(6,1), key='leak_stop'), sg.Text('sec')],
          [sg.Text('Résiduel:'), sg.InputText('0.49', size=(6,1), key='res_start'), 
           sg.Text('à'), sg.InputText('0.5', size=(6,1), key='res_stop'), sg.Text('sec')]
      ])],
      [sg.Frame('Paramètres d\'analyse', [
          [sg.Text('Fréquence (Hz):'), sg.InputText('20', size=(4,1), key='frequency'),
           sg.Text('Nombre de pics:'), sg.InputText('2', size=(4,1), key='n_peaks')],
          [sg.Text('Fenêtre pic:'), sg.InputText('0.498', size=(6,1), key='peak_start'), 
           sg.Text('à'), sg.InputText('0.51', size=(6,1), key='peak_stop')],
          [sg.Text('Fenêtre bruit:'), sg.InputText('0.1', size=(6,1), key='noise_start'), 
           sg.Text('à'), sg.InputText('0.4', size=(6,1), key='noise_stop')]
      ])],
      [sg.Button('Traiter tous les fichiers', size=(20,2)), sg.Button('Quitter')]
    ]


    window = sg.Window('Bootstrap Batch Processing', layout)


    while True:
        event, values = window.read()

        if event in (None, 'Quitter'):
            break

        if event == 'Traiter tous les fichiers':
            input_folder = values['input_folder']
            output_folder = values['output_folder']
            
            if not input_folder or not output_folder:
                sg.popup_error('Veuillez sélectionner les dossiers d\'entrée et de sortie')
                continue
            
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            
            # Recherche des fichiers Excel
            excel_files = []
            for ext in ['*.xlsx', '*.xls']:
                excel_files.extend(glob.glob(os.path.join(input_folder, ext)))
            
            if not excel_files:
                sg.popup_error('Aucun fichier Excel trouvé dans le dossier d\'entrée')
                continue
            
            # Paramètres
            parameters = {
                'photo_start': float(values['photo_start']),
                'photo_stop': float(values['photo_stop']),
                'leak_start': float(values['leak_start']), 
                'leak_stop': float(values['leak_stop']),
                'res_start': float(values['res_start']),
                'res_stop': float(values['res_stop']),
                'frequency': values['frequency'],
                'n_peaks': int(values['n_peaks']),
                'peak_start': float(values['peak_start']),
                'peak_stop': float(values['peak_stop']),
                'noise_start': float(values['noise_start']),
                'noise_stop': float(values['noise_stop'])
            }
            
            # Traitement des fichiers
            total_files = len(excel_files)
            successful = 0
            failed = 0
            
            progress_layout = [[sg.Text(f'Traitement en cours... 0/{total_files}', key='progress_text')],
                              [sg.ProgressBar(total_files, orientation='h', size=(50, 20), key='progress_bar')],
                              [sg.Button('Annuler', key='cancel')]]
            
            progress_window = sg.Window('Progression', progress_layout, finalize=True)
            
            for i, file_path in enumerate(excel_files):
                event_prog, values_prog = progress_window.read(timeout=10)
                if event_prog == 'cancel':
                    break
                
                progress_window['progress_text'].update(f'Traitement: {os.path.basename(file_path)} ({i+1}/{total_files})')
                progress_window['progress_bar'].update(i)
                
                if process_single_file(file_path, output_folder, parameters):
                    successful += 1
                else:
                    failed += 1
            
            progress_window.close()
            
            # Résumé
            sg.popup(f'Traitement terminé!\n\n'
                    f'Fichiers traités avec succès: {successful}\n'
                    f'Fichiers échoués: {failed}\n'
                    f'Total: {total_files}')

    window.close()

if __name__ == '__main__':
    batch_process()