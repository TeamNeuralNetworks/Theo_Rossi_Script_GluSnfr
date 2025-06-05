# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 19:23:22 2019

@author: Theo.Rossi
"""

from neo import io
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy as sy
import pandas as pd


path = r'D:\Theo.ROSSI\Uncaging 2-P\GAMBP-glutamate\Antagonist properties\20190613'

files = os.listdir(path)

FILE1 = []
FILE2 = []
TIME1 = []
TIME2 = []
SIGNAL1 = []
SIGNAL2 = []
VALUE_MAX1 = []
VALUE_MAX2 = []
INDEX_MAX1 = []
INDEX_MAX2 = []
VALUES_PEAK1 = []
VALUES_PEAK2 = []
MEAN_PEAK1 = []
MEAN_PEAK2 = []

PRE = []
POST = []

#fig, ax = plt.subplots(1,1)

for file in files:
    if 'before' in file:
        new_path = io.WinWcpIO('%s\%s'%(path, file))
        FILE1.append(new_path)
        block = new_path.read_block()
    
        for sweep in range(len(block.segments)):
            time = block.segments[0].analogsignals[0].times
            TIME1.append(time)
            trace1 = block.segments[sweep].analogsignals[0].magnitude
            signal1 = trace1 - np.mean(trace1[0:17500])
            SIGNAL1.append(signal1)
        
            bl_start = np.ravel(np.where(time >= 0.405))[0]
            bl_stop = np.ravel(np.where(time <= 0.415))[-1]
            
            value_max1 = np.max(signal1[bl_start:bl_stop])
            VALUE_MAX1.append(value_max1)
            index_max1 = np.argmax(signal1[bl_start:bl_stop])+bl_start
            INDEX_MAX1.append(index_max1)
        
        for j in range(len(SIGNAL1)):
            bl_x1 = np.ravel(np.where(time >= 0.4070))[0]
            bl_x2 = np.ravel(np.where(time <= 0.4073))[-1]
            window1 = SIGNAL1[j][bl_x1:bl_x2]
            VALUES_PEAK1.append(window1)
            mean_window1 = np.mean(window1)
            MEAN_PEAK1.append(mean_window1)
            
            
        plt.figure()
        for i in range(len(SIGNAL1)):
            plt.plot(time, SIGNAL1[i], 'grey', alpha=0.4)
            

        average = np.mean(SIGNAL1, axis=0)
        
        plt.plot(time, average, 'c', linewidth=2)
        plt.title('%s'%(file))
        plt.xlabel('Time(sec)')
        plt.ylabel('Amplitude (pA)')
        plt.grid(True)


for file in files:
    if 'after' in file:
        new_path = io.WinWcpIO('%s\%s'%(path, file))
        FILE2.append(new_path)
        block = new_path.read_block()
    
        for sweep in range(len(block.segments)):
            time = block.segments[0].analogsignals[0].times
            TIME2.append(time)
            trace2 = block.segments[sweep].analogsignals[0].magnitude
            signal2 = trace2 - np.mean(trace2[0:17500])
            SIGNAL2.append(signal2)
        
            value_max2 = np.max(signal2[bl_start:bl_stop])
            VALUE_MAX2.append(value_max2)
            index_max2 = np.argmax(signal2[bl_start:bl_stop])+bl_start
            INDEX_MAX2.append(index_max2)
            
        for j in range(len(SIGNAL2)):
            bl_x3 = np.ravel(np.where(time >= 0.4067))[0]
            bl_x4 = np.ravel(np.where(time <= 0.4069))[-1]
            window2 = SIGNAL2[j][bl_x3:bl_x4]
            VALUES_PEAK2.append(window2)
            mean_window2 = np.mean(window2)
            MEAN_PEAK2.append(mean_window2)    
            
        plt.figure()
        for i in range(len(SIGNAL2)):
            plt.plot(time, SIGNAL2[i], 'grey', alpha=0.4)

        average = np.mean(SIGNAL2, axis=0)
        
        plt.plot(time, average, 'c', linewidth=2)
        plt.title('%s'%(file))
        plt.xlabel('Time(sec)')
        plt.ylabel('Amplitude (pA)')
        plt.grid(True)


#Create the violin pot
labels = ['Control', 'GAMBP-glu 300µM']
x = np.arange(len(labels))
y = [MEAN_PEAK1, MEAN_PEAK2]
z = [np.mean(MEAN_PEAK1), np.mean(MEAN_PEAK2)]
y_std = [np.std(MEAN_PEAK1), np.std(MEAN_PEAK2)]

fig, ax = plt.subplots(2,1, sharex=True)
violin = ax[0].violinplot(y, x, points=20, widths=0.3, showmeans=True, showextrema=True)
for part in violin['bodies']:
    part.set_color('#029386')
#    part.set_facecolor('#06c2ac')
#    part.set_edgecolor('#000000')
#ax.bar(x, z, yerr=y_std, width=0.3, align='center', alpha=0.5, ecolor='black', capsize=10)
ax[0].set_ylabel('Amplitude (pA)', fontsize=15)
ax[0].set_xticks(x)
ax[0].set_yticks(np.arange(0, 500, 100))
ax[0].yaxis.grid(True)

ratio = [z[0]/z[0], z[1]/z[0]]
print(ratio)
ax[1].bar(x, ratio, width=0.2, align='center', color='r', alpha=0.3)
ax[1].set_yticks(np.arange(0, 1.5, 0.2))
ax[1].set_xticklabels(labels, fontsize=15)
ax[1].set_ylabel('Amplitude (pA)', fontsize=15)
ax[1].yaxis.grid(True)


##Create a dataframe to excel
#ar = [MEAN_PEAK1, MEAN_PEAK2]
#df1 = pd.DataFrame(ar, index = ['PRE', 'POST'])
#df1.to_excel(r'D:\Theo.ROSSI\Uncaging 2-P\GAMBP-glutamate\Antagonist properties\20190608.xlsx')
#
#Statistical tests
k2, p = sy.stats.shapiro(MEAN_PEAK1)
alpha = 0.05
print('p = {:g}'.format(p))
if p<alpha:
    print('The null hypothesis can be rejected')
else:
    print('The null hypothesis cannot be rejected')
    
k3, p2 = sy.stats.shapiro(MEAN_PEAK2)
alpha2 = 0.05
print('p = {:g}'.format(p2))
if p<alpha2:
    print('The null hypothesis can be rejected')
else:
    print('The null hypothesis cannot be rejected')

s2, p3 = sy.stats.levene(MEAN_PEAK1, MEAN_PEAK2, center='mean')
alpha3 = 0.05
print('p2 = {:g}'.format(p3))
if p3<alpha3:
    print('The null hypothesis can be rejected')
else:
    print('The null hypothesis cannot be rejected')

Wcox, p_value = sy.stats.ranksums(MEAN_PEAK1, MEAN_PEAK2)
Wcox_alpha = 0.05
print('p_value = {:g}'.format(p_value))
if p_value < Wcox_alpha:
    print('The null hypothesis cannot be rejected')
else:
    print('The null hypothesis can be rejected')

#plt.figure()
#x = np.arange(0,2,1)
#y = [VALUE_MAX1, VALUE_MAX2]
#z = [[np.mean(VALUE_MAX1)], [np.mean(VALUE_MAX2)]]
#y_std = [np.std(VALUE_MAX1), np.std(VALUE_MAX2)]
#for xe, ye, ze in zip(x,y,z):
#    plt.scatter([xe]*len(ye), ye, s=40, alpha=0.6)
##    plt.scatter([xe]*len(ze), ze, s=50, c='r', marker='D', label='Mean')
#    plt.bar([xe]*len(ze), ze, width=0.4, align='center', color='r', alpha=0.3)
#plt.yaxis.grid(True)
#plt.ylabel('Amplitude (pA)')
#plt.xticks(x)
    




