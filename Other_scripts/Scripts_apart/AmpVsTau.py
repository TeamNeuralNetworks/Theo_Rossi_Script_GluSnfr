# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 18:01:09 2020

@author: Theo.ROSSI
"""


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os


file_amp = r'E:\AAVDJ.GluSnFR-S72A_Amp_Tau\20190530_linescan1_20Hz_10pulses_2.5mMCa_button1_Amp.xlsx'
file_tau = r'E:\AAVDJ.GluSnFR-S72A_Amp_Tau\20190530_linescan1_20Hz_10pulses_2.5mMCa_button1_Tau.xlsx'

amp = pd.read_excel(file_amp, header=0).iloc[0,1:].values
tau = pd.read_excel(file_tau, header=0).iloc[0,1:].values

plt.figure()
for i in range(len(amp)):
    plt.scatter(tau[i], amp[i])