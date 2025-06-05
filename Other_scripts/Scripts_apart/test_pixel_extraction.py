# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 16:18:00 2019

@author: Theo.ROSSI
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

path = r'D:\Theo.ROSSI\GluSnFR\Virus_AAV.hSynap.SF-iGluSnFR_S72A\AAVDJ.GluSnFR-S72A\2019_05_30'

folders = os.listdir(path)
folders = sorted(folders)


for file in range(len(folders)):
    if 'ROI.tif' in folders[file]:
        ROI_file = Image.open('%s/%s'%(path, folders[file]))
        pix = ROI_file.load()
        print(ROI_file.size)
        print(int(pix[200,200]))
