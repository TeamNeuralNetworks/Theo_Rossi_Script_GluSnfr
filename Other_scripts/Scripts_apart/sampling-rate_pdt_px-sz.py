# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 10:51:38 2019

@author: Theo.ROSSI
"""
import matplotlib.pyplot as plt
import numpy as np
import os
import configparser as cp

path = r'D:/Theo.ROSSI/GluSnFR/Virus_AAV.hSynap.SF-iGluSnFR_S72A/AAVDJ.GluSnFR-S72A/2019_05_30/20190530_linescan1/20190530_18_21_53_linescan1'

folder = os.listdir(path)
config = cp.ConfigParser()

for file in range(len(folder)):
    if '.ini' in folder[file]:
        ini_file = '%s/%s'%(path, folder[file])
        config.read(ini_file)
        sampling_rate, pdt, px_sz = config.get('_', 'lines.per.second'), config.get('_', 'pixel.dwell.time.in.sec'), config.get('_', 'x.pixel.sz')
        print(sampling_rate, pdt*1000, px_sz*1000000)