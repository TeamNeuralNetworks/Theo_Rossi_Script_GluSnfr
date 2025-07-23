# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 00:01:39 2021

@author: Theo.ROSSI
"""


#global savedir
###############################
## DEF GET COORDINATE CURSORS #
###############################

def on_click(event):
    global startpos, endpos
    if event.button is MouseButton.LEFT:
        startpos=[event.xdata]
    elif event.button is MouseButton.RIGHT:
        endpos=[event.xdata]
     
 ###############################
## DEF FILTERS #
###############################       
 
def filter_signal(signal, order, sample_rate, freq_low, freq_high, axis=0):
    Wn = [freq_low / (sample_rate / 2), freq_high / (sample_rate / 2)]
    sos_coeff = scipy.signal.iirfilter(order, Wn, btype="band", ftype="butter", output="sos")
    filtered_signal = scipy.signal.sosfiltfilt(sos_coeff, signal, axis=axis)

    return filtered_signal


#def notch_filter(signal, order=8, sample_rate=20000, freq_low=48, freq_high=52, axis=0):
#    Wn = [freq_low / (sample_rate / 2), freq_high / (sample_rate / 2)]
#    notch_coeff = signal.iirfilter(order, Wn, btype="bandstop", ftype="butter", output="sos")
#    notch_signal = signal.sosfiltfilt(notch_coeff, signal, axis=axis)
#
#    return notch_signal
       
#############################
### FIT
#############################    
def Fit_single_trace(Trace, Time_trace, x_start,x_end):
    
    idx_start=np.ravel(np.where(Time_trace >=x_start))[0]
    idx_stop=np.ravel(np.where(Time_trace >= x_end))[0]
    x = Time_trace[idx_start:idx_stop]
    y = Trace[idx_start:idx_stop]
    x2=np.array(np.squeeze(x))
    y2=np.array(np.squeeze(y))  
        
    try:
        param_bounds=([-np.inf,0.,0.,0.],[np.inf,1.,10.,1000.])      # be careful ok for seconds. If millisec change param 2 and 3
        popt, pcov = curve_fit(func_mono_exp, x2, y2,bounds=param_bounds, max_nfev = 10000) 
        print ('tau decay =',popt[2]*1000, ' ms' )
        return popt[2], popt, idx_start, idx_stop
    except:
        print ('Fit failed')
        popt[2]= float('nan')
        popt= float('nan')
        return popt[2], popt, idx_start, idx_stop
        pass

def func_mono_exp(x, a, b, c, d):
    return a * np.exp(-(x-b)/c) + d

#def func_line(x, a, b):
#    return a * x + b    
#
#def func_dbl_exp(x, a, b, c, d, e):
#    return a * np.exp(-x/b) + c * np.exp(-x/d)+e
#
#def func_lognormal(x, a, b, c, d):    
#    pass
    
###############################
### DEF LOAD FILE  ############
###############################

def load_wcp(file_wcp):
    global REC
    global TIME
    my_file = neo.io.WinWcpIO(file_wcp)
    bl = my_file.read_block()
    for episode in bl.segments :
        time = episode.analogsignals[0].times #The time vector
        TIME.append(time)
        rec = episode.analogsignals[0].magnitude #The signal vector 
        REC.append(rec) 
    
    for i in range(len(REC)):
        REC[i] = np.array(np.squeeze(REC[i]))
        TIME[i] = np.array(np.squeeze(TIME[i]))
    global sampling
    sampling=float(TIME[1][1])*1000  #sampling rate in ms
        
def load_folder(path):
    list_file=glob.glob(os.path.join(path, '*.wcp'))
    print (list_file)
    for i in range(len(list_file)):
       load_wcp(list_file[i])
  
def load_xls(file_xls):
    global REC
    global TIME
    global sampling
    
    # Réinitialiser les listes
    REC = []
    TIME = []
    
    try:
        df = pd.read_excel(file_xls, sheet_name='Traces DF_F0', header=0)
        
        print(f"Colonnes trouvées: {df.columns.tolist()}")
        print(f"Dimensions du DataFrame: {df.shape}")
        
        # Trouver la colonne de temps
        time_column = None
        for col in df.columns:
            if 'Time' in str(col) or 'time' in str(col).lower():
                time_column = col
                break
        
        if time_column is None:
            print("Attention: Aucune colonne 'Time' trouvée, utilisation de la première colonne")
            time_column = df.columns[0]
        
        # Extraire le vecteur temps
        timescale = df[time_column].values
        print(f"Vecteur temps extrait: {len(timescale)} points")
        
        # Ajouter les traces (toutes les colonnes sauf celle du temps)
        for col in df.columns:
            if col != time_column:
                trace = df[col].values
                # Vérifier que la trace contient des données valides
                if not np.all(np.isnan(trace)):
                    REC.append(trace)
                    TIME.append(timescale.copy())  # Chaque trace a son propre vecteur temps
                    print(f"Trace ajoutée: {col}, {len(trace)} points")
                else:
                    print(f"Trace ignorée (données manquantes): {col}")
        
        # Calculer le taux d'échantillonnage
        if len(timescale) > 1:
            sampling = float(timescale[1] - timescale[0]) * 1000  # en ms
            print(f"Taux d'échantillonnage calculé: {sampling} ms")
        else:
            sampling = 1.0  # valeur par défaut
            print("Impossible de calculer le taux d'échantillonnage, utilisation de 1.0 ms")
        
        print(f"Chargement terminé: {len(REC)} traces chargées")
    
    except Exception as e:
        print(f"Erreur lors du chargement du fichier Excel: {e}")
        print("Vérifiez que:")
        print("1. Le fichier contient une feuille nommée 'Traces DF_F0'")
        print("2. La première ligne contient les en-têtes")
        print("3. Une colonne contient 'Time' dans son nom")
        
        # En cas d'erreur, essayer de charger avec des paramètres par défaut
        try:
            df = pd.read_excel(file_xls)  # Première feuille, sans spécifier le nom
            print(f"Tentative avec la première feuille: {df.shape}")
            
            # Utiliser la première colonne comme temps
            timescale = df.iloc[:, 0].values
            
            # Ajouter toutes les autres colonnes comme traces
            for i in range(1, len(df.columns)):
                trace = df.iloc[:, i].values
                if not np.all(np.isnan(trace)):
                    REC.append(trace)
                    TIME.append(timescale.copy())
            
            if len(timescale) > 1:
                sampling = float(timescale[1] - timescale[0]) * 1000
            else:
                sampling = 1.0
                
            print(f"Chargement de secours réussi: {len(REC)} traces")
            
        except Exception as e2:
            print(f"Échec du chargement de secours: {e2}")
            return
#    

    
###############################
### DEF AVERAGE  ############
###############################

def Make_average(REC,TIME,CursorOn):
    global AVERAGE
    global TAGREC

    
    for i in range(len(REC)):
        if TAG[i]==1:
            TAGREC.append(REC[i])
    AVERAGE = np.mean(TAGREC,axis = 0)
    if CursorOn == True:
        min_avg, _, _, _ = calc_min_in_trace(AVERAGE, startpos[0], endpos[0], 5)
        max_avg, _, _, _ = calc_max_in_trace(AVERAGE, startpos[0], endpos[0], 5)
    
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.plot(TIME[0], AVERAGE)
    plt.xlabel('Time (s)')
    plt.ylabel('Signal (Amp)')
    plt.draw()
    
    if CursorOn == True:
        plt.title('Average: min = '+str(min_avg)+'   max = '+str(max_avg)+'    (Close me)',fontweight="bold", fontsize=12, color="g")
    else:
        plt.title('Average (CLOSE ME)')
#

##########################
# DEF Mode superimposed ##
##########################

def superimposed(episode):
    global REC
    global TIME
    global startpos
    global endpos
    sg.theme('Black')	
 
    layout2 = [ [sg.Text('Superimposed Episodes')],
                [sg.Button('Start'), sg.Button('Close', button_color=('red', 'white'))],
                [sg.Button('Previous Trace'),sg.Button('Next Trace'),sg.Button('Clear')],
                [sg.Text('Go To (Push start)'), sg.InputText(default_text="0", size=(10, 1))]]
    
    window2 = sg.Window('Display Superimposed Traces', layout2, location=(0,110))
    episode = episode
    
    fig_super = plt.figure()
    ax_super = fig_super.add_subplot(111)
    ax_super.plot(TIME[episode], REC[episode])
    plt.xlabel('Time (s)')
    plt.ylabel('Signal (Amp)')
    plt.title('Episode '+str(episode)+'/'+str(len(REC)-1)) 
    
    while True:
        
        event2, values2 = window2.read()
        try:
           
            if event2 in (None, 'Close'):	# if user closes window or clicks cancel
                break
    
            if event2 == "Next Trace":
                episode+=1
                ax_super.plot(TIME[episode], REC[episode])
                try:
                    ax_super.axvline(startpos[0], color ='r')
                    ax_super.axvline(endpos[0], color ='g')
                    fig_super.canvas.draw()
                except:
                    pass
                plt.xlabel('Time (s)')
                plt.ylabel('Signal (Amp)')
                plt.title('Episode: '+str(episode))
                plt.draw()
     
            if event2 == "Previous Trace":
                episode-=1
                ax_super.plot(TIME[episode], REC[episode])
                try:
                    ax_super.axvline(startpos[0], color ='r')
                    ax_super.axvline(endpos[0], color ='g')
                    fig_super.canvas.draw()
                except:
                    pass
                plt.xlabel('Time (s)')
                plt.ylabel('Signal (Amp)')
                plt.title('Episode: '+str(episode))
                plt.draw()
     
            if event2 == "Start":
                ax_super.clear()
                try:
                    episode=int(values2[0])
                except:
                    episode=0
                ax_super.plot(TIME[episode], REC[episode])
                plt.xlabel('Time (s)')
                plt.ylabel('Signal (Amp)')
                plt.title('Episode: '+str(episode))
                plt.draw()
    
            if event2 == "Clear":
                ax_super.clear()
                               

        except:
            sg.popup_error('')
            pass
   
    window2.close()
    plt.close(fig_super)

    
##############################
## FUNCTIONS TO CALCULATE AMPLITUDES ##
##############################    
    
def calc_min_in_trace(trace, win1, win2, Win_for_extremum):
    start_idx = np.ravel(np.where(TIME[0] >= win1))[0]
    stop_idx = np.ravel(np.where(TIME[0] >= win2))[0]
    sub_trace=trace[start_idx:stop_idx]
    MIN_sub_trace_idx = np.argmin(sub_trace)
    MIN_idx=start_idx+MIN_sub_trace_idx
    
    if Win_for_extremum == 1:
        # Exact point measurement
        MIN = trace[MIN_idx]
        return MIN, MIN_idx, MIN_idx, MIN_idx  # value, center_idx, start_idx, end_idx
    else:
        # Averaged measurement over span
        span_start = max(0, MIN_idx - Win_for_extremum)
        span_end = min(len(trace), MIN_idx + Win_for_extremum + 1)
        MIN = np.mean(trace[span_start:span_end])
        return MIN, MIN_idx, span_start, span_end-1  # value, center_idx, start_idx, end_idx

def calc_max_in_trace(trace, win1, win2, Win_for_extremum):
    start_idx = np.ravel(np.where(TIME[0] >= win1))[0]
    stop_idx = np.ravel(np.where(TIME[0] >= win2))[0]
    sub_trace=trace[start_idx:stop_idx]
    MAX_sub_trace_idx = np.argmax(sub_trace)
    MAX_idx=start_idx+MAX_sub_trace_idx
    
    if Win_for_extremum == 1:
        # Exact point measurement
        MAX = trace[MAX_idx]
        return MAX, MAX_idx, MAX_idx, MAX_idx  # value, center_idx, start_idx, end_idx
    else:
        # Averaged measurement over span
        span_start = max(0, MAX_idx - Win_for_extremum)
        span_end = min(len(trace), MAX_idx + Win_for_extremum + 1)
        MAX = np.mean(trace[span_start:span_end])
        return MAX, MAX_idx, span_start, span_end-1  # value, center_idx, start_idx, end_idx


def Calculate_Amps():
    
    global amp_dict
    global amp_dict_idx
    global REC
    global TIME
    global TAG
    
    sg.theme('Black')	
 
    layout3 = [ [sg.Text('FIND PEAKS (Tagged)',font = ('Arial', 14, 'bold') )],
                [sg.Text('_'*30)], 
                [sg.Checkbox('Minimum', size=(12, 1), default=False)], 
                [sg.Text('_'*30)],
                [sg.Button('One Peak from Cursors')],
                [sg.Text('_'*30)],
                [sg.Button('All Peaks from Trains')],
                [sg.Text('ISI (ms)'), sg.InputText(default_text="50", size=(10, 1))],
                [sg.Text('Peak number'), sg.InputText(default_text="10", size=(10, 1))],
                [sg.Text('_'*30)],
                [sg.Text('Span for peaks (+/-)'), sg.InputText(default_text="1", size=(10, 1))],
                [sg.Text('_'*30)],
                [sg.Text('Save as'),sg.InputText(default_text="Amplitudes", size=(10, 1))],
                [sg.Text('_'*30)],
                [sg.Button('Some stats from Amplitudes')]]
#                [sg.Text('_'*30)],
#                [sg.Text(''*30)],
#                [sg.Button('Close_Amp', button_color=('black', 'white'))]]
    
    window3 = sg.Window('Amplitudes', layout3, location=(0,110), size=(250,500))
       
    while True:
        
        event3, values3 = window3.read()
        try:
           
            if event3 in (None, 'Close_Amp'):	# if user closes window or clicks cancel
                break
            
            if event3 == "One Peak from Cursors":
                amp_dict['AMP1'] = []
                amp_dict_idx['AMP1'] = []
                amp_dict_span = {'AMP1': []}  # Store span information
                locals().update(amp_dict)
                locals().update(amp_dict_idx)
                
                span_value = int(values3[3])
                
                for i in range(len(REC)):
                    if values3[0] == True:
                        if TAG[i] == 1:
                            local_amp, center_idx, span_start, span_end = calc_min_in_trace(REC[i], startpos[0], endpos[0], span_value)
                            amp_dict['AMP1'].append(local_amp)
                            amp_dict_idx['AMP1'].append(center_idx)
                            amp_dict_span['AMP1'].append((span_start, span_end))
                    else:
                        if TAG[i] == 1:
                            local_amp, center_idx, span_start, span_end = calc_max_in_trace(REC[i], startpos[0], endpos[0], span_value)
                            amp_dict['AMP1'].append(local_amp) 
                            amp_dict_idx['AMP1'].append(center_idx)
                            amp_dict_span['AMP1'].append((span_start, span_end))
                
                # Create a new figure to show measurement points on all tagged traces
                fig_measurements = plt.figure(figsize=(12, 8))
                ax_measurements = fig_measurements.add_subplot(111)
                
                # Plot all tagged traces with their measurement points
                tagged_indices = [i for i, tag in enumerate(TAG) if tag == 1]
                colors = plt.cm.tab10(np.linspace(0, 1, len(tagged_indices)))
                
                for idx, trace_idx in enumerate(tagged_indices):
                    if idx < len(amp_dict['AMP1']):
                        # Plot the trace
                        ax_measurements.plot(TIME[trace_idx], REC[trace_idx], 
                                           color=colors[idx], alpha=0.7, 
                                           label=f'Episode {trace_idx}')
                        
                        # Get measurement information
                        center_idx = amp_dict_idx['AMP1'][idx]
                        span_start, span_end = amp_dict_span['AMP1'][idx]
                        x_center = TIME[trace_idx][center_idx]
                        y_measurement = amp_dict['AMP1'][idx]
                        
                        if span_value == 1:
                            # Exact point - use the actual trace value at that point
                            y_exact = REC[trace_idx][center_idx]
                            ax_measurements.plot(x_center, y_exact, 'o', 
                                               color=colors[idx], markersize=8, 
                                               markeredgecolor='black', markeredgewidth=2,
                                               markerfacecolor='white', markerfacewidth=1)
                            print(f"Episode {trace_idx}: Exact measurement = {y_exact:.4f} at time = {x_center:.4f}s (index {center_idx})")
                        else:
                            # Averaged measurement - show as horizontal bar
                            x_start = TIME[trace_idx][span_start]
                            x_end = TIME[trace_idx][span_end]
                            ax_measurements.plot([x_start, x_end], [y_measurement, y_measurement], 
                                               color=colors[idx], linewidth=4, alpha=0.8)
                            ax_measurements.plot(x_center, y_measurement, 's', 
                                               color=colors[idx], markersize=6, 
                                               markeredgecolor='black', markeredgewidth=1)
                            print(f"Episode {trace_idx}: Averaged measurement = {y_measurement:.4f} from {x_start:.4f}s to {x_end:.4f}s (span={span_end-span_start+1} points)")
                
                # Draw cursors
                ax_measurements.axvline(startpos[0], color='r', linestyle='--', linewidth=2, 
                                      alpha=0.8, label='Start cursor')
                ax_measurements.axvline(endpos[0], color='g', linestyle='--', linewidth=2, 
                                      alpha=0.8, label='End cursor')
                
                ax_measurements.set_xlabel('Time (s)')
                ax_measurements.set_ylabel('Signal (Amp)')
                title_text = f'Amplitude Measurements on Tagged Traces (Span: {span_value})'
                if span_value == 1:
                    title_text += ' - Exact Points'
                else:
                    title_text += ' - Averaged Regions'
                ax_measurements.set_title(title_text, fontweight="bold", fontsize=14)
                ax_measurements.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.tight_layout()
                plt.show()
                
                fig4 = plt.figure()
                ax4 = fig4.add_subplot(111)
                ax4.plot(amp_dict['AMP1'])
                plt.xlabel('Number episodes')
                plt.ylabel('Signal (Amp)')
                plt.title('Amplitude Timecosurse')
                
                df = pd.DataFrame.from_dict(amp_dict)
                writer = pd.ExcelWriter('{}\{}.xlsx'.format(savedir,str(values3[4])))
                df.to_excel(writer)
                writer.save() 
                

            if event3 == "All Peaks from Trains":
                amp_dict = {}
                amp_dict_idx = {}
                amp_dict_span = {}
                locals().update(amp_dict) 
                locals().update(amp_dict_idx) 
                locals().update(amp_dict_span)
                   
                for i in range(int(values3[2])):
                    amp_dict['AMP' + str(i+1)] = []
                    amp_dict_idx['AMP' + str(i+1)] = []
                    amp_dict_span['AMP' + str(i+1)] = []
                
                span_value = int(values3[3])
                Start_for_trains=startpos[0]
                Stop_for_trains=endpos[0]
                
                for key in amp_dict.keys():                          
                    for i in range(len(REC)):
                        if values3[0] == True:
                            if TAG[i] == 1:
                                local_amp, center_idx, span_start, span_end = calc_min_in_trace(REC[i],float(Start_for_trains), float(Stop_for_trains), span_value)
                                amp_dict[key].append(local_amp)
                                amp_dict_idx[key].append(center_idx)
                                amp_dict_span[key].append((span_start, span_end))
                        else:
                            if TAG[i] == 1:
                                local_amp, center_idx, span_start, span_end = calc_max_in_trace(REC[i],float(Start_for_trains), float(Stop_for_trains), span_value)
                                amp_dict[key].append(local_amp)
                                amp_dict_idx[key].append(center_idx)
                                amp_dict_span[key].append((span_start, span_end))
                   
                    Start_for_trains+=float(values3[1])/1000
                    Stop_for_trains+=float(values3[1])/1000
                
                # Create visualization showing all measurement points
                fig_train_measurements = plt.figure(figsize=(15, 10))
                tagged_indices = [i for i, tag in enumerate(TAG) if tag == 1]
                
                # Show first few traces as examples (to avoid overcrowding)
                max_traces_to_show = min(5, len(tagged_indices))
                colors = plt.cm.tab10(np.linspace(0, 1, max_traces_to_show))
                
                for trace_idx in range(max_traces_to_show):
                    if trace_idx < len(tagged_indices):
                        real_trace_idx = tagged_indices[trace_idx]
                        ax_train = fig_train_measurements.add_subplot(max_traces_to_show, 1, trace_idx + 1)
                        ax_train.plot(TIME[real_trace_idx], REC[real_trace_idx], 
                                    color=colors[trace_idx], alpha=0.8, linewidth=1.5)
                        
                        # Plot measurement points for all peaks
                        for peak_idx, key in enumerate(amp_dict.keys()):
                            if trace_idx < len(amp_dict[key]):
                                center_idx = amp_dict_idx[key][trace_idx]
                                span_start, span_end = amp_dict_span[key][trace_idx]
                                x_center = TIME[real_trace_idx][center_idx]
                                y_measurement = amp_dict[key][trace_idx]
                                
                                if span_value == 1:
                                    # Exact point measurement
                                    y_exact = REC[real_trace_idx][center_idx]
                                    ax_train.plot(x_center, y_exact, 'o', 
                                                markersize=6, label=f'{key}: {y_exact:.4f}',
                                                markeredgecolor='black', markeredgewidth=0.5,
                                                color=f'C{peak_idx}')
                                else:
                                    # Averaged measurement - show as horizontal bar
                                    x_start = TIME[real_trace_idx][span_start]
                                    x_end = TIME[real_trace_idx][span_end]
                                    ax_train.plot([x_start, x_end], [y_measurement, y_measurement], 
                                                color=f'C{peak_idx}', linewidth=3, alpha=0.8)
                                    ax_train.plot(x_center, y_measurement, 's', 
                                                markersize=4, label=f'{key}: {y_measurement:.4f}',
                                                markeredgecolor='black', markeredgewidth=0.5,
                                                color=f'C{peak_idx}')
                        
                        # Draw measurement windows
                        temp_start = startpos[0]
                        temp_end = endpos[0]
                        for peak_idx in range(int(values3[2])):
                            ax_train.axvspan(temp_start, temp_end, alpha=0.1, color=f'C{peak_idx}')
                            temp_start += float(values3[1])/1000
                            temp_end += float(values3[1])/1000
                        
                        ax_train.set_xlabel('Time (s)')
                        ax_train.set_ylabel('Signal (Amp)')
                        title_text = f'Episode {real_trace_idx} - Measurement Points (Span: {span_value})'
                        if span_value == 1:
                            title_text += ' - Exact'
                        else:
                            title_text += ' - Averaged'
                        ax_train.set_title(title_text)
                        ax_train.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                
                plt.tight_layout()
                plt.show()
                
                # Print summary of measurements
                print("\nMeasurement Summary:")
                for key in amp_dict.keys():
                    print(f"{key}: {len(amp_dict[key])} measurements")
                    if len(amp_dict[key]) > 0:
                        print(f"  Range: {min(amp_dict[key]):.4f} to {max(amp_dict[key]):.4f}")
                        print(f"  Mean: {np.mean(amp_dict[key]):.4f}")
                
                fig4 = plt.figure()
                ax4 = fig4.add_subplot(111)
                for key in amp_dict.keys():
                    ax4.plot(amp_dict[key], label = key)
                ax4.legend()
                plt.xlabel('Number episodes')
                plt.ylabel('Signal (Amp)')
                plt.title('Amplitude Timecourse')    
             
                df = pd.DataFrame.from_dict(amp_dict)
                writer = pd.ExcelWriter('{}\{}.xlsx'.format(savedir,str(values3[4])))
                df.to_excel(writer)
                writer.save()
                
            if event3 == "Some stats from Amplitudes":
                df = pd.DataFrame.from_dict(amp_dict)  
                print ("\n")
                print('first values')
                print (df.head())
                print ("\n")
                print ('Basic stats')
                print (df.describe())
                print ("\n")
                print ('Pearson correlation')
                print (df.corr(method='pearson'))
                sns.pairplot(df)
                sns.jointplot(x=df.columns[0], y=df.columns[1], data=df, kind="reg")
                
        except:
            sg.popup_error('Tagged traces or cursors missing')
            pass
    window3.close()

   
###########################
##   lOAD FILE WINDOW  ####
###########################

def load_files():
    sg.theme('DarkBlue')	
    
    layout0= [  [sg.Text('Enter File name     '), sg.InputText(), sg.FileBrowse(),sg.Button('Start File')],
                [sg.Text('Enter Folder name '), sg.InputText(), sg.FolderBrowse(),sg.Button('Start Folder')],
                [sg.Text('Enter xls file          '), sg.InputText(), sg.FileBrowse(),sg.Button('Start xls')]]
    
    window0= sg.Window('Load files', layout0, location=(0,0))
    
    while True:
        event0, values0 = window0.read()
    
        try:
            if event0 in (None, 'Close'):	# if user closes window or clicks cancel
                break
                 
            if event0 == 'Start File':
                load_wcp(values0[0]) 
                break
            if event0 == 'Start Folder':
                load_folder(values0[1])
                break
            if event0 == 'Start xls':
                load_xls(values0[2])
                break

            
            
        except:
            pass
        
    window0.close()    


"""
###########################
##   MAIN WINDOW       ####
###########################
"""
def Main_window():
    
    sg.theme('SandyBeach')	
    
    layout1 = [ [sg.Frame(' Init ',[[sg.Button('Start'),sg.Button('Previous Trace'),sg.Button('Next Trace'),sg.Button('Clear')],
                                     [sg.Button('Superimposed'),sg.Text('Go To (Push start)'), sg.InputText(default_text="0", size=(10, 1))]],relief="ridge", border_width= 5)],
                [sg.Frame(' Adjust Traces ',[[sg.Button('Leak Subtraction'), sg.Button('Bleaching correction')],
                                     [sg.Text('Window (ms)'),sg.InputText(default_text="0", size=(10, 1)),sg.InputText(default_text="900", size=(11, 1)),sg.Button('Undo')]],relief="groove", border_width= 5)],
                [sg.Frame(' Filter Traces ',[[sg.Button('Smooth Traces'),sg.Text('Odd number'), sg.InputText(default_text="19", size=(12, 1))],
                  [sg.Button('Filter'),sg.Text('Band-Pass(Hz)'),sg.InputText(default_text="0.01", size=(10, 1)),sg.InputText(default_text="2000", size=(10, 1))]],relief="groove", border_width= 5)],
                [sg.Frame(' Select Traces ',[[sg.Button('Tag'), sg.Button('UnTag'), sg.Button('Tag All'), sg.Button('UnTag All'), sg.Button('Save Tags')]],relief="groove", border_width= 5)],
                [sg.Frame(' Average ',[[sg.Button('Averaged Tagged Traces'),sg.Checkbox('calc on cursors', size=(12, 1), default=False)],
                                        [sg.Button("Save Average"), sg.InputText(default_text="Avg", size=(10, 1))]],relief="groove", border_width= 5)],
                [sg.Frame(' Cursors and Amplitudes ',[[sg.Button('Select Cursors'),sg.Button('Draw Cursors'),sg.Text('Left/right click')],
                                        [sg.Button('Calculate Amps')]],relief="groove", border_width= 5)],
                [sg.Frame(' Fitting with cursors ',[[sg.Button('Fit current trace'),sg.Button('Fit all traces'), sg.Button('Fit average')],
                                        [sg.Button('Fit current train'),sg.Button('Fit all trains')],
                                        [sg.Text('Peak number'),sg.InputText(default_text="3", size=(10, 1)),sg.Text('ISI (ms)'), sg.InputText(default_text="50", size=(10, 1))],
                                        [sg.Button('Remove residuals in current train'), sg.Button('Remove all residuals')]],relief="groove", border_width= 5)]]
     
    window1 = sg.Window('Browse Traces', layout1, resizable=True, finalize=True)
    
    plt.ion()
    fig = plt.figure('Main Figure')
    ax = fig.add_subplot(111)
    
    global TAG
    global savedir
    global REC
    global TIME
    global Saved_REC
    global amp_dict
    global amp_dict_idx 
    global amp_dict_corr
    global FitPeaks_dict_tau
    global FitPeaks_dict_popt
    global AVERAGE
    global startpos
    global endpos
    global Fit_tau

        
    episode=0
    
    
    TAG = np.zeros(len(REC))
    
    while True:
        
        event, values = window1.read()
        
        try:
           
            if event in (None, 'Close'):	# if user closes window or clicks cancel
                break
            
            if event == "Superimposed":
                superimposed(episode)
    
            if event == "Leak Subtraction": 
                Saved_REC=[]
                locals().update(Saved_REC)
                
                for i in range(len(REC)):
                    Saved_REC.append(REC[i])
                for i in range(len(REC)):
                    rec = REC[i]
                    leak = np.mean(rec[int(float(values[1])/sampling):int(float(values[2])/sampling)])
                    REC[i]=REC[i]-leak 
            
            if event == "Bleaching correction": 
                xstart = float(values[1])/1000
                xstop = float(values[2])/1000
                Saved_REC=[]
                locals().update(Saved_REC)
                
                for i in range(len(REC)):
                    Saved_REC.append(REC[i])
                for i in range(len(REC)):
                    rec = REC[i]
                    time = TIME[i]
                    local_tau, local_popt, idxstart, idxstop = Fit_single_trace(rec, time, xstart,xstop)
                    print (local_popt)
                    try:
                        for j in range(len(REC[i])):
                            bleaching = func_mono_exp(TIME[i][j], *local_popt)
                            REC[i][j] -= bleaching
                    except:
                        pass
                    
            if event == "Smooth Traces": 
                Saved_REC=[]
                #locals().update(Saved_REC)
                
                for i in range(len(REC)):
                    Saved_REC.append(REC[i])
                for i in range(len(REC)):
                    REC[i][-1] = 0
                    REC[i] = savgol_filter(np.squeeze(REC[i]),int(values[3]),2)   #int(values[3]), 2) # Filter: window size 19, polynomial order 2   

                    
                    # def gauss_kernel(n=9, sigma=2.0):
                    #     assert n % 2 == 1, "n doit être impair"
                    #     r = n // 2
                    #     xs = np.arange(-r, r+1)
                    #     kern = np.exp(-xs**2 / (2 * sigma**2))
                    #     kern /= kern.sum()
                    #     return kern
                    
                    # kernel = gauss_kernel(n=9, sigma=2.0)
                    
                    # from scipy.signal import lfilter
                    
                    # x = np.squeeze(REC[i])
                    # REC[i] = lfilter(kernel, 1.0, x)
                    
                    # x = np.squeeze(REC[i])
                    # kernel = np.ones(int(values[3])) / int(values[3])
                    # # Utilise `lfilter` pour ne considérer que les points passés
                    # from scipy.signal import lfilter
                    # REC[i] = lfilter(kernel, 1.0, x) 
                          
            if event == "Undo": 
                for i in range(len(REC)):
                    REC[i]=Saved_REC[i]
            
            if event == "Filter": 
                Saved_REC=[]
                locals().update(Saved_REC)
                
                for i in range(len(REC)):
                    Saved_REC.append(REC[i])
                for i in range(len(REC)):
                    REC[i]=filter_signal(REC[i], 8, 1000*round(float(1/sampling)), float(values[4]), float(values[5]), axis=0)
                       
            if event == "Next Trace":
                ax.clear()
                episode+=1
                if episode>(len(REC)-1):
                    sg.popup_error('End of episodes')
                    pass
                else:
                    if TAG[episode] == 1:
                        ax.plot(TIME[episode], REC[episode], color ='r')
                    else:
                        ax.plot(TIME[episode], REC[episode])
                    try:
                        ax.axvline(startpos[0], color ='r')
                        ax.axvline(endpos[0], color ='g')
                        fig.canvas.draw()
                    except:
                        pass
                    try:
                        for key in amp_dict.keys():  
                            index=amp_dict_idx[key][episode]
                            x=TIME[episode][index]
                            y=amp_dict[key][episode]
                            ax.plot(x,y,'bo', linewidth = 3)
                    except:
                        pass
                    
                    plt.xlabel('Time (s)')
                    plt.ylabel('Signal (Amp)')
                    plt.title('Episode '+str(episode)+'/'+str(len(REC)-1), fontweight="bold", fontsize=16, color="g")
                    plt.draw()
     
            if event == "Previous Trace":
                ax.clear()
                episode-=1  
                if TAG[episode] == 1:
                    ax.plot(TIME[episode], REC[episode], color ='r')
                else:
                    ax.plot(TIME[episode], REC[episode])
                try:
                    ax.axvline(startpos[0], color ='r')
                    ax.axvline(endpos[0], color ='g')
                    fig.canvas.draw()
                except:
                    pass
                try:
                    for key in amp_dict.keys():  
                        index=amp_dict_idx[key][episode]
                        x=TIME[episode][index]
                        y=amp_dict[key][episode]
                        ax.plot(x,y,'bo', linewidth = 3)
                except:
                    pass
                plt.xlabel('Time (s)')
                plt.ylabel('Signal (Amp)')
                plt.title('Episode '+str(episode)+'/'+str(len(REC)-1), fontweight="bold", fontsize=16, color="g")
                plt.draw()
                           
            if event == "Start":
                ax.clear()
                try:
                    episode=int(values[0])
                except:
                    episode=0
                if TAG[episode] == 1:    
                    ax.plot(TIME[episode], REC[episode], color ='r')
                else:
                    ax.plot(TIME[episode], REC[episode])
                plt.xlabel('Time (s)')
                plt.ylabel('Signal (Amp)')
                plt.title('Episode '+str(episode)+'/'+str(len(REC)-1), fontweight="bold", fontsize=16, color="g")
                plt.draw()
    
            if event == "Clear":
                ax.clear()
                plt.draw()
                
            if event == "Tag":
                TAG[episode]=1
     
            if event == "UnTag":
                TAG[episode]=0
      
            if event == "Tag All":
                TAG= [1 for i in range(len(REC))]
           
            if event == "UnTag All":
                TAG= [0 for i in range(len(REC))]   
    
            if event == "Save Tags":
                df = pd.DataFrame(TAG, columns=['Tags'])
                writer = pd.ExcelWriter('{}\Tags.xlsx'.format(savedir))
                df.to_excel(writer, index = False)
                writer.save()
                            
            if event == "Averaged Tagged Traces":
                Make_average(REC,TIME,values[6])
            
            if event == "Save Average":
                df = pd.DataFrame(AVERAGE, columns=['Average'])
                writer = pd.ExcelWriter('{}\{}.xlsx'.format(savedir,str(values[7])))
                df.to_excel(writer, index = False)
                writer.save()
    
                    
            if event == 'Select Cursors': 
                cid=fig.canvas.mpl_connect('button_press_event', on_click)
                                  
            if event == 'Draw Cursors': 
                fig.canvas.mpl_disconnect(cid)
                try:
                    ax.axvline(startpos[0], color ='r')
                    ax.axvline(endpos[0], color ='g')
                    fig.canvas.draw()
                except:
                    pass
                
            if event == 'Calculate Amps':
                Calculate_Amps()
                
            if event == 'Fit current trace':    
                local_tau, local_popt, idxstart, idxstop = Fit_single_trace(REC[episode], TIME[episode],startpos[0],endpos[0])
                x=TIME[episode][idxstart:idxstop]
                x2=np.array(np.squeeze(x))
                fig_fit=plt.figure()
                ax_fit = fig_fit.add_subplot(111)
                ax_fit.plot(TIME[episode], REC[episode], color = 'green', alpha = 1, lw = '2')
                ax_fit.plot(x2, func_mono_exp(x2, *local_popt), color = 'black', alpha = 1, lw = '3')
                print ('tau decay episode'+str(episode)+' =',local_popt[2]*1000, ' ms' )
                
            if event == 'Fit all traces':    
                All_Tau=[]
                All_popt=[]
                for i in range(len(REC)): 
                    if TAG[i] == 1:
                        local_tau, local_popt, idxstart, idxstop = Fit_single_trace(REC[i], TIME[i],startpos[0],endpos[0])
                        All_Tau.append(local_tau)
                        All_popt.append(local_popt)
                    else:
                        pass
                data = {'Tau': np.array(All_Tau)}
                df = pd.DataFrame.from_dict(data)
                writer = pd.ExcelWriter('{}\All Tau AMP1.xlsx'.format(savedir))
                df.to_excel(writer)
                writer.save() 
                          
            if event == 'Fit average':    
                local_tau, local_popt, idxstart, idxstop = Fit_single_trace(AVERAGE, TIME[0],startpos[0],endpos[0])
                x=TIME[episode][idxstart:idxstop]
                x2=np.array(np.squeeze(x))
                fig_fit=plt.figure()
                ax_fit = fig_fit.add_subplot(111)
                ax_fit.plot(TIME[0], AVERAGE, color = 'green', alpha = 1, lw = '2')
                ax_fit.plot(x2, func_mono_exp(x2, *local_popt), color = 'black', alpha = 1, lw = '3')
                print ('tau decay average'+str(episode)+' =',local_popt[2]*1000, ' ms' )
                    
 
                            
            if event == 'Fit all trains':  
                FitPeaks_dict_tau = {}
                FitPeaks_dict_popt = {}
                locals().update(FitPeaks_dict_tau) 
                locals().update(FitPeaks_dict_popt)
                
                for i in range(int(values[8])):
                    FitPeaks_dict_tau['AMP' + str(i+1)] = []
                    FitPeaks_dict_popt['AMP' + str(i+1)] = []
                 
                Start_for_trains=startpos[0]
                Stop_for_trains=endpos[0]
                
                for key in FitPeaks_dict_popt.keys():
                    for i in range(len(REC)):
                        print(key,'  episode: ', i)
                        if TAG[i] == 1: 
                            local_tau, local_popt, idxstart, idxstop = Fit_single_trace(REC[i], TIME[i],Start_for_trains,Stop_for_trains)
                            FitPeaks_dict_tau[key].append(local_tau)
                            FitPeaks_dict_popt[key].append(local_popt) 

                        else:
                            pass
                    Start_for_trains+=float(values[9])/1000
                    Stop_for_trains+=float(values[9])/1000
    
            
                df = pd.DataFrame.from_dict(FitPeaks_dict_tau)
                writer = pd.ExcelWriter('{}\Fit peaks dict tau.xlsx'.format(savedir))
                df.to_excel(writer)
                writer.save() 
                df2 = pd.DataFrame.from_dict(FitPeaks_dict_popt)
                writer = pd.ExcelWriter('{}\Fit peaks dict popt.xlsx'.format(savedir))
                df2.to_excel(writer)
                writer.save() 
                
            if event == 'Fit current train':  
                
                FitPeaks_dict_tau = {}
                FitPeaks_dict_popt = {}
                locals().update(FitPeaks_dict_tau) 
                locals().update(FitPeaks_dict_popt)
                
                for i in range(int(values[8])):
                    FitPeaks_dict_tau['AMP' + str(i+1)] = []
                    FitPeaks_dict_popt['AMP' + str(i+1)] = []
                 
                
                Start_for_trains=startpos[0]
                Stop_for_trains=endpos[0]
                
                fig_fit=plt.figure()
                ax_fit = fig_fit.add_subplot(111)
                ax_fit.plot(TIME[episode], REC[episode], color = 'green', alpha = 1, lw = '2')
                
                for key in FitPeaks_dict_popt.keys(): 
                    tau, popt, idxstart, idxstop = Fit_single_trace(REC[episode], TIME[episode],Start_for_trains,Stop_for_trains)
                    FitPeaks_dict_tau[key].append(tau)
                    FitPeaks_dict_popt[key].append(popt) 
                    Start_for_trains+=float(values[9])/1000
                    Stop_for_trains+=float(values[9])/1000
                    x=TIME[episode][idxstart:idxstop]
                    x2=np.array(np.squeeze(x))
                    ax_fit.plot(x2, func_mono_exp(x2, *popt), color = 'black', alpha = 1, lw = '3')
                
                df = pd.DataFrame.from_dict(FitPeaks_dict_tau)
                writer = pd.ExcelWriter('{}\Fit peaks dict tau.xlsx'.format(savedir))
                df.to_excel(writer)
                writer.save() 
                df2 = pd.DataFrame.from_dict(FitPeaks_dict_popt)
                writer = pd.ExcelWriter('{}\Fit peaks dict popt.xlsx'.format(savedir))
                df2.to_excel(writer)
                writer.save() 
                
                print (FitPeaks_dict_tau)  
                
                
            if event == 'Remove residuals in current train':
                amp_dict_corr = {}
                print ('Episode', episode)
                locals().update(amp_dict_corr)
                for i in range(int(values[8])):
                    amp_dict_corr['AMP' + str(i+1)] = []
                     
                for key in amp_dict_corr.keys():
                    if key == 'AMP1':
                        amp_dict_corr[key].append(amp_dict[key][episode])
                        print ('AMP1 =', amp_dict[key][episode])
                    else:
                        key_for_residual='AMP' + str(int(key[3:])-1)
                        index_peak_key=amp_dict_idx[key][episode]
                        residual = func_mono_exp(float(TIME[episode][index_peak_key]), *FitPeaks_dict_popt[key_for_residual][0])
                        new_amp = amp_dict[key][episode]-residual
                        amp_dict_corr[key].append(new_amp)
                        print ('new',key,' = ',new_amp)
                        
                df = pd.DataFrame.from_dict(amp_dict_corr)
                writer = pd.ExcelWriter('{}\Amplitudes_corr.xlsx'.format(savedir))
                df.to_excel(writer)
                writer.save()
                
                
            if event == 'Remove all residuals':
                amp_dict_corr = {}
                locals().update(amp_dict_corr)  
                for i in range(int(values[8])):
                    amp_dict_corr['AMP' + str(i+1)] = []
               
                
                for key in amp_dict_corr.keys():
                     for i in range(len(REC)):
                        if TAG[i] == 1: 
                            if key == 'AMP1':
                                amp_dict_corr[key].append(amp_dict[key][i])
                                
                            else:
                                key_for_residual='AMP' + str(int(key[3:])-1)
                                index_peak_key=amp_dict_idx[key][i]
                                residual = func_mono_exp(float(TIME[i][index_peak_key]), *FitPeaks_dict_popt[key_for_residual][i])
                                new_amp=amp_dict[key][i]-residual
                                amp_dict_corr[key].append(new_amp)
                        else:
                            pass        #                                locals().update(amp_dict_corr)                            
                
                fig5 = plt.figure()
                ax5 = fig5.add_subplot(111)
                for key in amp_dict_corr.keys():
                    ax5.plot(amp_dict_corr[key], label = key)
                ax5.legend()
                plt.xlabel('Number episodes')
                plt.ylabel('Signal (Amp)')
                plt.title('Corrected Amplitude Timecourse')  
                
                df = pd.DataFrame.from_dict(amp_dict_corr)
                writer = pd.ExcelWriter('{}\Amplitudes_corr.xlsx'.format(savedir))
                df.to_excel(writer)
                writer.save()
      
        except:
            sg.popup_error('')
            pass
            
    window1.close()




    
    
if __name__ == '__main__' :
    
    from matplotlib.backend_bases import MouseButton
    import PySimpleGUI as sg
    import glob
    import os
    import neo
    import numpy as np
    from matplotlib import pyplot as plt
    from scipy.signal import savgol_filter
    import scipy.signal
    from scipy.optimize import curve_fit
    import pandas as pd
    import seaborn as sns
    
    
    savedir = r'E:\AAVDJ.GluSnFR-S72A\iGluSnFRvsEphy'

    
          
    sg.theme('DarkBlue')	
    
    layout0= [[sg.Button('GO'),sg.Button('STOP')],
              [sg.InputText(default_text='{}'.format(savedir)),sg.FolderBrowse()],
              [sg.Button('Select Save Folder')]]
    
    window_main= sg.Window('New Analysis', layout0, location=(0,0))
    
    while True:
        event_main, values_main = window_main.read()
                
        if event_main == 'GO':
       
            REC = []
            TAGREC=[]
            TAG = []
            Saved_REC = []
            TIME = []
            All_Tau=[]
            All_popt=[]
            amp_dict = {}
            amp_dict_idx = {}
            amp_dict_corr= {}
            FitPeaks_dict_tau = {}
            FitPeaks_dict_popt = {}
            amp_dict = []
          
            load_files() 
            TheMatrix=REC
            TheTIME=TIME
            Main_window()
            
        if event_main in (None, 'STOP'):	# if user closes window or clicks cancel
            break
        
        if event_main == 'Select Save Folder':
            savedir=str(values_main[0])
            print (savedir)
            
            
      
    window_main.close()    
    plt.close('all')