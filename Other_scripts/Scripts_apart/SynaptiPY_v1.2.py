# -*- coding: utf-8 -*-
"""
Created on Tue May 26 15:29:02 2020

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
        param_bounds=([-np.inf,0.,0.,-1000.],[np.inf,1.,1.,1000.])      # be careful ok for seconds. If millisec change param 2 and 3
        popt, pcov = curve_fit(func_mono_exp, x2, y2,bounds=param_bounds) 
    
    #    print ('a * np.exp(-(x-b)/c) + d', popt)
    #    print ('tau decay =',popt[2]*1000, ' ms' )
        return popt[2], popt, idx_start, idx_stop
    except:
        print ('Fit failed')
        popt[2]=float('nan')
        popt=float('nan')
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
    my_file = neo.io.WinWcpIO(file_wcp)
    bl = my_file.read_block()
    for episode in bl.segments :
        time = episode.analogsignals[0].times #The time vector
        TIME.append(time)
        rec = episode.analogsignals[0].magnitude #The signal vector 
        REC.append(rec) 
    global sampling
    sampling=float(TIME[1][1])*1000  #sampling rate in ms
        
def load_folder(path):
    list_file=glob.glob(os.path.join(path, '*.wcp'))
    print (list_file)
    for i in range(len(list_file)):
       load_wcp(list_file[i])
  
def load_xls(file_xls):  
    df=pd.read_excel (file_xls, header = 0)
    print(df)
    for i in range(len(df.columns)):
        if 'Time' == df.columns[i]:
            timescale=df.iloc[:,i].values
            TIME.append(timescale)
        else:
            sweep=df.iloc[:,i].values
            REC.append(sweep)
    print (REC)    
    for i in range(len(REC)-1):    # this to get the same size of matrix between REC and TIME 
        TIME.append(timescale)
     
    global sampling
    sampling=float(TIME[1][1])*1000
#    

    
###############################
### DEF AVERAGE  ############
###############################

def Make_average(REC,TIME,CursorOn):
            global AVERAGE
            TAGREC=[]
            for i in range(len(REC)):
                if TAG[i]==1:
                    TAGREC.append(REC[i])
            AVERAGE = np.mean(TAGREC,axis = 0)
            if CursorOn == True:
                min_avg=calc_min_in_trace(AVERAGE, startpos[0], endpos[0], 5)
                max_avg=calc_max_in_trace(AVERAGE, startpos[0], endpos[0], 5)
            
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
    
    sg.theme('Black')	
 
    layout2 = [ [sg.Text('Superimposed Episodes')],
                [sg.Button('Start'), sg.Button('Close', button_color=('red', 'white'))],
                [sg.Button('Previous Trace'),sg.Button('Next Trace'),sg.Button('Clear')],
                [sg.Text('Go To (Push start)'), sg.InputText(default_text="0", size=(10, 1))]]
    
    window2 = sg.Window('Display Superimposed Traces', layout2, location=(0,110))
    episode = episode
    
    while True:
        
        event2, values2 = window2.read()
        try:
           
            if event2 in (None, 'Close'):	# if user closes window or clicks cancel
                break
    
            if event2 == "Next Trace":
                episode+=1
                ax.plot(TIME[episode], REC[episode])
                try:
                    ax.axvline(startpos[0], color ='r')
                    ax.axvline(endpos[0], color ='g')
                    fig.canvas.draw()
                except:
                    pass
                plt.xlabel('Time (s)')
                plt.ylabel('Signal (Amp)')
                plt.title('Episode: '+str(episode))
                plt.draw()
     
            if event2 == "Previous Trace":
                episode-=1
                ax.plot(TIME[episode], REC[episode])
                try:
                    ax.axvline(startpos[0], color ='r')
                    ax.axvline(endpos[0], color ='g')
                    fig.canvas.draw()
                except:
                    pass
                plt.xlabel('Time (s)')
                plt.ylabel('Signal (Amp)')
                plt.title('Episode: '+str(episode))
                plt.draw()
     
            if event2 == "Start":
                ax.clear()
                try:
                    episode=int(values2[0])
                except:
                    episode=0
                ax.plot(TIME[episode], REC[episode])
                plt.xlabel('Time (s)')
                plt.ylabel('Signal (Amp)')
                plt.title('Episode: '+str(episode))
                plt.draw()
    
            if event2 == "Clear":
                ax.clear()
                               

        except:
            sg.popup_error('')
            pass
   
    window2.close()


    
##############################
## FUNCTIONS TO CALCULATE AMPLITUDES ##
##############################    
    
def calc_min_in_trace(trace, win1, win2, Win_for_extremum):
    start_idx = np.ravel(np.where(TIME[0] >= win1))[0]
    stop_idx = np.ravel(np.where(TIME[0] >= win2))[0]
    MIN = np.min(trace[start_idx:stop_idx])
    sub_trace=trace[start_idx:stop_idx]
    MIN_sub_trace_idx = np.argmin(sub_trace)
    MIN = np.mean(trace[start_idx+MIN_sub_trace_idx-Win_for_extremum:start_idx+MIN_sub_trace_idx+Win_for_extremum])
    MIN_idx=start_idx+MIN_sub_trace_idx
    return MIN, MIN_idx

def calc_max_in_trace(trace, win1, win2, Win_for_extremum):
    start_idx = np.ravel(np.where(TIME[0] >= win1))[0]
    stop_idx = np.ravel(np.where(TIME[0] >= win2))[0]
    MAX = np.max(trace[start_idx:stop_idx])
    sub_trace=trace[start_idx:stop_idx]
    MAX_sub_trace_idx = np.argmax(sub_trace)
    MAX = np.mean(trace[start_idx+MAX_sub_trace_idx-Win_for_extremum:start_idx+MAX_sub_trace_idx+Win_for_extremum])
    MAX_idx=start_idx+MAX_sub_trace_idx
    return MAX,MAX_idx


def Calculate_Amps():
    
    sg.theme('Black')	
 
    layout3 = [ [sg.Text('FIND PEAKS (Tagged traces)')],
                [sg.Text('_'*30)], 
                [sg.Checkbox('Minimum', size=(12, 1), default=True)], 
                [sg.Text('_'*30)],
                [sg.Button('One Peak from Cursors')],
                [sg.Text('_'*30)],
                [sg.Button('All Peaks from Trains')],
                [sg.Text('ISI (ms)'), sg.InputText(default_text="50", size=(10, 1))],
                [sg.Text('Peak number'), sg.InputText(default_text="3", size=(10, 1))],
                [sg.Text('_'*30)],
                [sg.Text('Span for peaks (+/-)'), sg.InputText(default_text="5", size=(10, 1))],
                [sg.Text('_'*30)],
                [sg.Text('Save as'),sg.InputText(default_text="Amplitudes", size=(10, 1))],
                [sg.Text('_'*30)],
                [sg.Text(''*30)],
                [sg.Button('Close_Amp', button_color=('black', 'white'))]]
    
    window3 = sg.Window('Amplitudes', layout3, location=(0,110), size=(400,700))
       
    while True:
        
        event3, values3 = window3.read()
        try:
           
            if event3 in (None, 'Close_Amp'):	# if user closes window or clicks cancel
                break
            
            if event3 == "One Peak from Cursors":
                global amp_dict, amp_dict_idx
                amp_dict['AMP1'] = []
                amp_dict_idx['AMP1'] = []
                locals().update(amp_dict)
                for i in range(len(REC)):
                    if values3[0] == True:
                        if TAG[i] == 1:
                            local_amp, local_amp_idx =calc_min_in_trace(REC[i], startpos[0], endpos[0], int(values3[3]))
                            amp_dict['AMP1'].append(local_amp)
                            amp_dict_idx['AMP1'].append(local_amp_idx)
                    else:
                        if TAG[i] == 1:
                            local_amp, local_amp_idx =calc_max_in_trace(REC[i], startpos[0], endpos[0], int(values3[3]))
                            amp_dict['AMP1'].append(local_amp) 
                            amp_dict_idx['AMP1'].append(local_amp_idx)
                
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
                
                for i in range(int(values3[2])):
                    amp_dict['AMP' + str(i+1)] = []
                    amp_dict_idx['AMP' + str(i+1)] = []
                locals().update(amp_dict)    
                Start_for_trains=startpos[0]
                Stop_for_trains=endpos[0]
                for key in amp_dict.keys():                          
                    for i in range(len(REC)):
                        if values3[0] == True:
                            if TAG[i] == 1:
                                local_amp, local_amp_idx =calc_min_in_trace(REC[i],float(Start_for_trains), float(Stop_for_trains),int(values3[3]))
                                amp_dict[key].append(local_amp)
                                amp_dict_idx[key].append(local_amp_idx)
                        else:
                            if TAG[i] == 1:
                                local_amp, local_amp_idx =calc_max_in_trace(REC[i],float(Start_for_trains), float(Stop_for_trains),int(values3[3]))
                                amp_dict[key].append(local_amp)
                                amp_dict_idx[key].append(local_amp_idx)
                   
                    Start_for_trains+=float(values3[1])/1000
                    Stop_for_trains+=float(values3[1])/1000
                
                print (amp_dict)
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
                
                   
        except:
            sg.popup_error('Tagged traces or cursors missing')
            pass
    window3.close()

""" 
###########################
##   MAIN SCRIPT       ####
###########################
   
###########################
##   lOAD FILE WINDOW  ####
###########################
"""
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
    
    layout1 = [ [sg.Button('Start'),sg.Button('Previous Trace'),sg.Button('Next Trace'),sg.Button('Clear'),sg.Button('Superimposed'),sg.Text('Go To (Push start)'), sg.InputText(default_text="0", size=(10, 1))],
                [sg.Text('_'*30)],
                [sg.Button('Leak Subtraction'), sg.Text('Window (ms)'),sg.InputText(default_text="0", size=(10, 1)),sg.InputText(default_text="10", size=(10, 1)),sg.Button('Undo leak subtraction')],
                [sg.Text('_'*30)],
                [sg.Button('Smooth Traces'),sg.Text('Odd number'), sg.InputText(default_text="19", size=(10, 1)),sg.Button('Undo Smoothing') ],
                [sg.Text('_'*30)],
                [sg.Button('Tag'), sg.Button('UnTag'), sg.Button('Tag All'), sg.Button('UnTag All'), sg.Button('Save Tags')],
                [sg.Button('Averaged Tagged Traces'),sg.Checkbox('calc on cursors', size=(12, 1), default=False), sg.Button("Save Average"), sg.InputText(default_text="Avg", size=(10, 1))],
                [sg.Text('_'*30)],
                [sg.Button('Select Cursors'),sg.Button('Draw Cursors'),sg.Text('Left/right click: Left/right cursor')],
                [sg.Text('_'*30)],
                [sg.Button('Calculate Amps')],
                [sg.Text('_'*30)],
                [sg.Button('Fit current trace'),sg.Button('Fit all traces'), sg.Button('Fit average')],
                [sg.Button('Fit current train'),sg.Button('Fit all trains'),sg.Text('Peak number'),sg.InputText(default_text="3", size=(10, 1)),sg.Text('ISI (ms)'), sg.InputText(default_text="50", size=(10, 1))],
                [sg.Button('Correct Amps in Trains')],
                [sg.Text('_'*30)],
                [sg.InputText(default_text=''), sg.FolderBrowse(),sg.Button('Select Save Folder')]]#sg.Button('Change save Folder'), sg.InputText(default_text="", size=(50, 1))]]
     
    window1 = sg.Window('Browse Traces', layout1, location=(0,110), resizable=True)
    
    episode=0
    
    plt.ion()
    fig = plt.figure('Main Figure')
    ax = fig.add_subplot(111)
    
    global TAG
    global savedir
    
    TAG = np.zeros(len(REC))
    
    while True:
        
        event, values = window1.read()
        
        try:
           
            if event in (None, 'Close'):	# if user closes window or clicks cancel
                break
            
            if event == "Superimposed":
                ax.plot(TIME[episode], REC[episode])
                plt.xlabel('Time (s)')
                plt.ylabel('Signal (Amp)')
                plt.title('Episode '+str(episode)+'/'+str(len(REC)-1))
                superimposed(episode)
    
            if event == "Leak Subtraction": 
                for i in range(len(REC)):
                    Saved_REC2.append(REC[i])
                for i in range(len(REC)):
                    rec = REC[i]
                    leak = np.mean(rec[int(float(values[1])/sampling):int(float(values[2])/sampling)])
                    REC[i]=REC[i]-leak 
               
            if event == "Smooth Traces": 
                for i in range(len(REC)):
                    Saved_REC.append(REC[i])
                for i in range(len(REC)):
                    REC[i] = savgol_filter(np.squeeze(REC[i]), int(values[3]), 2) # Filter: window size 19, polynomial order 2   
             
            if event == "Undo Smoothing": 
                for i in range(len(REC)):
                    REC[i]=Saved_REC[i]
                    
            if event == "Undo leak subtraction": 
                for i in range(len(REC)):
                    REC[i]=Saved_REC2[i]
               
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
                Make_average(REC,TIME,values[4])
            
            if event == "Save Average":
                df = pd.DataFrame(AVERAGE, columns=['Average'])
                writer = pd.ExcelWriter('{}\{}.xlsx'.format(savedir,str(values[5])))
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
                    
            if event == 'Fit current train':  
                FitPeaks_dict_tau = {}
                FitPeaks_dict_popt = {}
                Start_for_trains=startpos[0]
                Stop_for_trains=endpos[0]
                
                fig_fit=plt.figure()
                ax_fit = fig_fit.add_subplot(111)
                ax_fit.plot(TIME[episode], REC[episode], color = 'green', alpha = 1, lw = '2')
                
                for i in range(int(values[6])):
                    local_tau, local_popt, idxstart, idxstop = Fit_single_trace(REC[episode], TIME[episode],Start_for_trains,Stop_for_trains)
                    FitPeaks_dict_tau['FIT_ep'+str(episode)+'_'+str(i+1)] = local_tau
                    FitPeaks_dict_popt['FIT_ep' + str(episode)+'_pk'+str(i+1)] = local_popt
                    locals().update(FitPeaks_dict_tau)  
                    locals().update(FitPeaks_dict_popt) 
                    Start_for_trains+=float(values[7])/1000
                    Stop_for_trains+=float(values[7])/1000
                    
                    x=TIME[episode][idxstart:idxstop]
                    x2=np.array(np.squeeze(x))
                    ax_fit.plot(x2, func_mono_exp(x2, *local_popt), color = 'black', alpha = 1, lw = '3')
                
                print ('tau decay ep'+str(episode)+' =',FitPeaks_dict_tau)     
            
                
            if event == 'Fit all trains':  
                FitPeaks_dict_tau = {}
                FitPeaks_dict_popt = {}
                for i in range(int(values[6])):
                    FitPeaks_dict_tau['AMP' + str(i+1)] = []
                    FitPeaks_dict_popt['AMP' + str(i+1)] = []
                locals().update(amp_dict)    
                Start_for_trains=startpos[0]
                Stop_for_trains=endpos[0]
                for key in FitPeaks_dict_popt.keys():
                    for i in range(len(REC)):
                        if TAG[i] == 1: 
                            local_tau, local_popt, idxstart, idxstop = Fit_single_trace(REC[i], TIME[i],Start_for_trains,Stop_for_trains)
                            FitPeaks_dict_tau[key].append(local_tau)
                            FitPeaks_dict_popt[key].append(local_popt) 
                            locals().update(FitPeaks_dict_tau) 
                            locals().update(FitPeaks_dict_popt)
                        else:
                            pass
                    Start_for_trains+=float(values[7])/1000
                    Stop_for_trains+=float(values[7])/1000
    
                print (FitPeaks_dict_popt)    
                df = pd.DataFrame.from_dict(FitPeaks_dict_tau)
                writer = pd.ExcelWriter('{}\Fit peaks dict tau.xlsx'.format(savedir))
                df.to_excel(writer)
                writer.save() 
                df2 = pd.DataFrame.from_dict(FitPeaks_dict_popt)
                writer = pd.ExcelWriter('{}\Fit peaks dict popt.xlsx'.format(savedir))
                df2.to_excel(writer)
                writer.save() 
                
        
                
            if event == 'Correct Amps in Trains':
                amp_dict_corr= {}
                for i in range(int(values[6])):
                    amp_dict_corr['AMP' + str(i+1)] = []
                locals().update(amp_dict)    
                for key in amp_dict_corr.keys():
                     for i in range(len(REC)):
                        if TAG[i] == 1: 
                            residual = func_mono_exp(amp_dict_idx[key][i], *FitPeaks_dict_popt[key][i])
                            new_amp=amp_dict[key][i]-residual
                            amp_dict_corr[key].append(new_amp)   
                            
                print (amp_dict_corr)
                fig5 = plt.figure()
                ax5 = fig5.add_subplot(111)
                for key in amp_dict_corr.keys():
                    ax5.plot(amp_dict_corr[key], label = key)
                ax5.legend()
                plt.xlabel('Number episodes')
                plt.ylabel('Signal (Amp)')
                plt.title('Amplitude Timecourse')  
                
                df = pd.DataFrame.from_dict(amp_dict_corr)
                writer = pd.ExcelWriter('{}\Amplitudes_corr.xlsx'.format(savedir))
                df.to_excel(writer)
                writer.save()
                
                
            if event == 'Select Save Folder':  
                savedir=str(values[8])
                print (savedir)
                
                
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
    from scipy.optimize import curve_fit
    import pandas as pd
    

    savedir=r'E:\AAVDJ.GluSnFR-S72A_TripletAnalysis'
    
  
    sg.theme('DarkBlue')	
    
    layout0= [[sg.Button('START'),sg.Button('STOP')]]
    
    window_main= sg.Window('New Analysis', layout0, location=(0,0), size =(450,50))
    
    while True:
        event_main, values_main = window_main.read()
                
        if event_main == 'START':
#            global amp_dict, amp_dict_idx, TIME, REC, Saved_REC,Saved_REC2,Subtracted_REC
            TIME, REC, Saved_REC,Saved_REC2,Subtracted_REC = [],[],[],[],[]
            amp_dict = {}
            amp_dict_idx = {}
            locals().update(amp_dict)
            load_files() 
            Main_window()
            
        if event_main in (None, 'STOP'):	# if user closes window or clicks cancel
            break
            
        
    window_main.close()    