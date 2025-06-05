# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 13:58:44 2022

@author: Theo.ROSSI
"""


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from scipy import optimize
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler



def gaussian(x, amplitude, mean, stddev):
    return amplitude * np.exp(-((x - mean) / 4 / stddev)**2)


Path = r'\\equipe2-nas1\Theo.ROSSI\Paper_thesis\Figure3'
file = '20210304_linescan1_px_intensity.xlsx'
number_of_boutons = 6

name = file.rsplit('_',2)

Boutons = [pd.read_excel(f'{Path}/{file}', sheet_name = f'Bouton{i+1}') for i in range(number_of_boutons)]

for item in range(len(Boutons)):
    distance = np.array(Boutons[item]['Distance'])
    green = savgol_filter(np.array(Boutons[item]['Green']), 9, 2)
    red = savgol_filter(np.array(Boutons[item]['Red']), 9, 2)
    
    bsl_red = np.mean(red[np.argmin(red)])
    bsl_green = np.mean(green[np.argmin(green)])
    
    dF_F_red = (red - bsl_red)/bsl_red
    dF_F_green = (green - bsl_green)/bsl_green
    
    plt.figure()
    plt.plot(distance, dF_F_red, 'r')
    plt.plot(distance, dF_F_green, 'g')
    plt.title(f'Bouton {item+1}')
    plt.ylabel('DF/F')
    plt.xlabel('Distance (µm)')
    
    plt.savefig(f'{Path}/{name[0]}_bouton{item+1}.pdf')