# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 16:24:32 2020

@author: Theo.ROSSI
"""


import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
from sklearn import preprocessing


File = '20200403_linescan3_20Hz_10pulses_4mMCa_button1.xlsx'

data = pd.read_excel('E:/AAVDJ.GluSnFR-S72A/Profiles_extracted/{}'.format(File, header=0))

TIME, AVERAGE, PEAKS = [],[],[]

for i in range(len(data.columns)):
    
    if 'Time' == data.columns[i]:
        timescale=data.iloc[:,i].values
        TIME.append(timescale)
    
    if 'Average' == data.columns[i]:
        average = data.iloc[:,i].values
        AVERAGE.append(average)



if '20Hz' in File:
    if '3pulses' in File:
        stims = np.arange(0.5,0.64,0.05)
        stim_off = stims+0.045
    elif '10pulses' in File:
        stims = np.arange(0.5,1,0.05)
        stim_off = stims+0.05

elif '50Hz' in File:
    if '3pulses' in File:
        stims = np.arange(0.2,0.26,0.02)
        stim_off = stims+0.015
    if '10pulses' in File:
        stims = np.arange(0.5,0.7,0.02)
        stim_off = stims+0.015

elif '100Hz' in File:
    if '3pulses' in File:
        stims = np.arange(0.2,0.23,0.01)
        stim_off = stims+0.005
    if '10pulses' in File:
        stims = np.arange(0.5,0.6,0.01)
        stim_off = stims+0.005



fig, ax = plt.subplots(1,2, figsize=(12,5), tight_layout=True)
ax[0].plot(TIME[0], AVERAGE[0])

for stimon,stimoff in zip(stims,stim_off):
    index_on = np.ravel(np.where(timescale <= stimon))[-1]
    index_off = np.ravel(np.where(timescale <= stimoff))[-1]
    
    peaks = preprocessing.scale(average[index_on:index_off])
    PEAKS.append(peaks)

    ax[1].plot(timescale[0:len(peaks)], peaks, 'k', alpha=0.2)

peaks_average = np.mean(PEAKS, axis=0)
ax[1].plot(timescale[0:len(peaks)], peaks_average, 'b', linewidth=2)
# plt.scatter(stimon, np.max(AVERAGE[0]), color='k')
# plt.scatter(stimoff, np.max(AVERAGE[0]), color='r')