# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 14:15:25 2021

@author: Theo.ROSSI
"""


def perm_t_test(set1, set2, permutation=1000, histogram=True):
    
    '''
    Set1: first dataset
    Set2: second dataset
    Permutation: number of iterations
    Histogram: histogram of distribution after the permutation
    
    return: p-value
    
    /!\ Set1 and Set2 must be equal size
    '''
    
    
    diff = np.mean(set2) - np.mean(set1)
    
    
    DIFF_ALL, DIFF_UP, DIFF_DOWN = [],[],[]
    
    for i in range(permutation):
        
        combination = np.concatenate((set1, set2), axis=None)
        shuffle = np.random.shuffle(combination)
        
        new_set1 = combination[0:int(len(combination)/2)]
        new_set2 = combination[int(len(combination)/2):]
        
        new_diff = np.mean(new_set2) - np.mean(new_set1)
        DIFF_ALL.append(new_diff)
        
        if new_diff >= diff:
            DIFF_UP.append(new_diff)
            
        elif new_diff <= -diff:
            DIFF_DOWN.append(new_diff)
        
        print(new_set1, new_set2, new_diff)
        
        
    p_value = (np.sum(DIFF_UP) + np.sum(DIFF_DOWN))/permutation
    
    if histogram == True:
        
        plt.figure()
        plt.hist(DIFF_ALL, bins=int(np.sqrt(permutation)))
        
    return p_value
        
        

if __name__ == '__main__':
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt 
    
    