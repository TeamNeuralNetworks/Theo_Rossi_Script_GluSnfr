# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 15:57:34 2021

@author: Theo.ROSSI
"""

import numpy as np
from scipy.optimize import curve_fit
import uncertainties.unumpy as unp
import uncertainties as unc
from uncertainties import ufloat
import matplotlib.pyplot as plt    
from scipy import stats
import pandas as pd


Path = r'E:\AAVDJ.GluSnFR-S72A\AAVDJ.GluSnFR-S72A_Good_CaMg_ratio\GluSnFR_avg_clustering_variables_all_profiles.xlsx'
# import data
data = pd.read_excel('{}'.format(Path), sheet_name='Sheet1')
x_name = data['AMP1'].name
y_name = data['%Fail1'].name
x = data['AMP1'].values
y = data['%Fail1'].values
y = 1-(y/100)
n = len(y)


def func_log(x, a, b):
    return a * np.log(x) + b

def func_mono_exp(x, a, b, c):
    return a * np.exp(b*x) + c

def func_reg(x, a, b):
    return a * x + b

popt, pcov = curve_fit(func_reg, x, y, maxfev=5000)


perr = np.sqrt(np.diag(pcov))

x_fit = np.linspace(np.min(x), np.max(x), 100)
bound_upper = func_reg(x_fit, *(popt + perr))
bound_lower = func_reg(x_fit, *(popt - perr))

a = ufloat(popt[0], perr[0])
b = ufloat(popt[1], perr[1])
text_res = "Best fit parameters:\na = {}\nb = {}".format(a, b)
print(text_res)


plt.scatter(x, y_proba, color='#95D840FF')
plt.plot(x_fit, func_reg(x_fit, *popt), 'k')
plt.plot(x_fit, bound_upper, 'r')
plt.plot(x_fit, bound_lower, 'r')




'''
# retrieve parameter values
a = popt[0]
b = popt[1]
print('Optimal Values')
print('a: ' + str(a))
print('b: ' + str(b))

# compute r^2
r2 = 1.0-(sum((y-func_reg(x,a,b))**2)/((n-1.0)*np.var(y,ddof=1)))
print('R^2: ' + str(r2))

# calculate parameter confidence interval
a,b = unc.correlated_values(popt, pcov)
print('Uncertainty')
print('a: ' + str(a))
print('b: ' + str(b))

# plot data
plt.scatter(x, y, s=10, color='green', label='Data')

# calculate regression confidence interval
px = np.linspace(np.min(x), np.max(x), len(x))
py = a*px+b
nom = unp.nominal_values(py)
std = unp.std_devs(py)

def predband(x, xd, yd, p, func, conf=0.95):
    # x = requested points
    # xd = x data
    # yd = y data
    # p = parameters
    # func = function name
    alpha = 1.0 - conf    # significance
    N = xd.size          # data sample size
    var_n = len(p)  # number of parameters
    # Quantile of Student's t distribution for p=(1-alpha/2)
    q = stats.t.ppf(1.0 - alpha / 2.0, N - var_n)
    # Stdev of an individual measurement
    se = np.sqrt(1. / (N - var_n) * \
                 np.sum((yd - func(xd, *p)) ** 2))
    # Auxiliary definitions
    sx = (x - xd.mean()) ** 2
    sxd = np.sum((xd - xd.mean()) ** 2)
    # Predicted values (best-fit model)
    yp = func(x, *p)
    # Prediction band
    dy = q * se * np.sqrt(1.0+ (1.0/N) + (sx/sxd))
    # Upper & lower prediction bands.
    lpb, upb = yp - dy, yp + dy
    return lpb, upb

lpb, upb = predband(px, x, y, popt, func_reg, conf=0.95)

# plot the regression
plt.plot(px, nom, c='black', label='y=a x + b')

# uncertainty lines (95% confidence)
plt.plot(px, nom - 1.96 * std, c='orange',\
         label='95% Confidence Region')
plt.plot(px, nom + 1.96 * std, c='orange')
# prediction band (95% confidence)
plt.plot(px, lpb, 'k--',label='95% Prediction Band')
plt.plot(px, upb, 'k--')
plt.ylabel('{}'.format(y_name))
plt.xlabel('{}'.format(x_name))
plt.legend(loc='best')
'''