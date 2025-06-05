# -*- coding: utf-8 -*-
"""
Created on Sun Oct 10 14:07:00 2021

@author: Theo.ROSSI
"""


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

Path = r'E:\AAVDJ.GluSnFR-S72A\Controls\LFD'

bsl = pd.read_excel('{}/20210928_linescan2_bsl_50Hz_3pulses_2.5mMCa_button4_dF_F0_Amp.xlsx'.format(Path))
LFD = pd.read_excel('{}/20210928_linescan2_recovery_50Hz_LFDpeak_traces_2.5mMCa_button4_dF_F0_Amp.xlsx'.format(Path))
burst = pd.read_excel('{}/20210928_linescan2_recovery_50Hz_burst_button4_dF_F0_Amp.xlsx'.format(Path))

mean_bsl = bsl.iloc[-1,1]
bsl = bsl.iloc[:-1,1].values
mean_LFD = LFD.iloc[-1,1]
LFD = LFD.iloc[:-1,1].values
burst = burst.iloc[:,1:].values

plt.figure()
plt.scatter(np.arange(1,6), bsl, color = '#253494')
plt.scatter(3, mean_bsl, color='r', marker='*')
# plt.scatter([1,2], [np.mean(bsl['AMP1']), np.mean(LFD)], color = ['b', 'orange'])
plt.scatter(np.arange(6,16), LFD, color = '#1d91c0')
plt.scatter(11, mean_LFD, color='r', marker='*')
plt.scatter(np.arange(16,26), burst, color = '#7fcdbb')

plt.ylabel('DF/F')
plt.xlabel('#Peak')