# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 10:45:30 2019

@author: Theo.ROSSI
"""

x = [0,5,4,1]

def mean(x):
    _sum = np.sum(x)
    _mean = _sum/float(len(x))
    return _mean,_sum 
    
a = np.mean(x)
b,c = mean(x)

print (a,b)