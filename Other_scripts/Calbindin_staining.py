# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 12:10:21 2022

@author: theo.rossi
"""

import pandas as pd
import numpy as np
import scipy.stats as stats

file = r"\\equipe2-nas1\Theo.ROSSI\Controle_TdTomato_calbindin\20210629_confocal\Cells_counting.xlsx"

data = pd.read_excel(file)

sagittal = data.iloc[1,1:]
horizontal = data.iloc[-1,1:]

stats.chisquare(sagittal)
stats.chisquare(horizontal)
