# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 15:56:12 2019

@author: Theo.ROSSI
"""

import matplotlib.pyplot as plt
import pandas as pd

new_path = 'D:/Theo.ROSSI/GluSnFR/Virus_AAV.hSynap.SF-iGluSnFR_S72A/AAVDJ.GluSnFR-S72A/2019_10_10/20191010_16_33_42_freq_coord_[xy]T_ROI.coord'

a = pd.read_csv(new_path,sep="\t",header=1)


x, y = [],[]
for i in range(len(a)): 
    
    basic_value = a.iloc[i,0]
    print (basic_value)
    
    x.append(int(basic_value.split(',')[0]))
    y.append(int(basic_value.split(',')[1]))
    

plt.figure()

plt.plot(x,y)