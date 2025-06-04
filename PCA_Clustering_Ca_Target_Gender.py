# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 15:29:54 2022

@author: Theo.ROSSI
"""


import pandas as pd
import numpy as np
import os
import scipy.stats as stats
from math import atan2, degrees
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from scipy.spatial.distance import pdist, cdist
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
import seaborn as sns
from mlxtend.plotting import plot_pca_correlation_graph
from scipy.optimize import curve_fit
import uncertainties as unc
import uncertainties.unumpy as unp

Path = r'C:\Users\ruo4037\OneDrive - Northwestern University\Desktop\PhD paper'

file_traces = 'iGluSnFR_avg_traces_allCa_filtered_3sigma.xlsx'
file_all_Ca = 'GluSnFR_avg_variables_allCa_filtered_3sigma.xlsx'
file_15_4mM = 'GluSnFR_avg_variables_1.5_4mM_Ca_paired_filtered_3sigma.xlsx'

freq = '20Hz'
condition1 = '1.5mM'
condition2 = '4mM'
N_Clust = 6



def color_palette(num):
    clr = []
    cmap = cm.YlGnBu(np.linspace(0.2,0.9,num))
    for c in range(num):
        rgba = cmap[c]
        clr.append(colors.rgb2hex(rgba))
    return clr




def confidence_ellipse(x, y, ax, n_std=2.0, dataset_size=133, facecolor='none', **kwargs):       
    if x.size != y.size:
        raise ValueError("x and y must be the same size")
    
    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    
    cluster_size = x.size/dataset_size
    centroid_radius_x = np.sqrt(1 + pearson) * cluster_size
    centroid_radius_y = np.sqrt(1 - pearson) * cluster_size
        
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2, facecolor=facecolor, **kwargs)
    centroid = Ellipse((0, 0), width=centroid_radius_x * 2, height=centroid_radius_y * 2, facecolor=facecolor, **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    centroid.set_transform(transf + ax.transData)
    
    return ax.add_patch(ellipse), ax.add_patch(centroid) 





data_25mM = pd.read_excel(f'{Path}/{file_all_Ca}', sheet_name = f'{freq}_2,5mM').drop(['AMP3','AMP4','AMP5','AMP6','AMP7','AMP8','AMP9','AMP10',
                                                                                       '%Fail3', 'F0', 'q', 'Baseline_std'], axis=1).iloc[:,1:]
data_paired = pd.read_excel(f'{Path}/{file_15_4mM}', sheet_name = f'{freq}').drop(['AMP3','AMP4','AMP5','AMP6','AMP7','AMP8','AMP9','AMP10','%Fail3'], axis=1)


targets = data_25mM['Target']
gender = data_25mM['Sexe']
data_25mM = data_25mM.drop(['Target','Sexe'], axis=1)
labels = data_paired['ID']
data_paired = data_paired.iloc[:,1:-1]

scaler = StandardScaler()
scaling = scaler.fit(data_25mM)
data_25mM_scaled = scaling.transform(data_25mM)
data_paired_scaled = scaling.transform(data_paired)


clr = color_palette(N_Clust)
model_PCA = PCA(n_components=2)
HCPC = AgglomerativeClustering(n_clusters=N_Clust)


PCA_fit = model_PCA.fit(data_25mM_scaled)
X = PCA_fit.transform(data_25mM_scaled)
Y = PCA_fit.transform(data_paired_scaled)




Ca15 = [Y[i] for i in range(len(Y)) if condition1 in labels[i]]
Ca4 = [Y[i] for i in range(len(Y)) if condition2 in labels[i]]
Ca15 = np.array(Ca15).reshape(len(Ca15), -1)
Ca4 = np.array(Ca4).reshape(len(Ca4), -1)

HCPCPredict = HCPC.fit_predict(X)
IN, PC = [],[]



####### PCA PLOTS 1.5/4mM; PC vs nonPC; Male vs Female ##########
#################################################################

fig, ax = plt.subplots(1,4,figsize=(12, 3), sharex=True, sharey=True, tight_layout=True)
for i in range(len(X)):
    ax[0].scatter(X[:, 0][i], X[:, 1][i], c='k', alpha=0.2)
    ax[3].scatter(X[:, 0][i], X[:, 1][i], c='k', alpha=0.2)
    if 'UN' in targets[i]:
        ax[1].scatter(X[:, 0][i], X[:, 1][i], c='k', alpha=0.5)
    if 'IN' in targets[i]:
        IN.append(X[i])
        ax[1].scatter(X[:, 0][i], X[:, 1][i], c='b')
    elif 'PC' in targets[i]:
        PC.append(X[i])
        ax[1].scatter(X[:, 0][i], X[:, 1][i], c='r')
    
    if 'UN' in gender[i]:
        ax[2].scatter(X[:, 0][i], X[:, 1][i], c='k', alpha=0.5)
    elif 'F' in gender[i]:
        ax[2].scatter(X[:, 0][i], X[:, 1][i], c='r')
    if 'M' in gender[i]:
        ax[2].scatter(X[:, 0][i], X[:, 1][i], c='b')

# [ax[0].scatter(X[68:74, 0][i], X[68:74, 1][i]) for i in range(6)] #fiber 20210303_linescan3
ax[0].set_xlabel('PC1 '+str(model_PCA.explained_variance_ratio_[0]*100))
ax[0].set_ylabel('PC2 '+str(model_PCA.explained_variance_ratio_[1]*100))
ax[0].axvline(x=0.0, color='k', ls='--')
ax[0].axhline(y=0.0, color='k', ls='--')
ax[1].axvline(x=0.0, color='k', ls='--')
ax[1].axhline(y=0.0, color='k', ls='--')
ax[2].axvline(x=0.0, color='k', ls='--')
ax[2].axhline(y=0.0, color='k', ls='--')
ax[3].axvline(x=0.0, color='k', ls='--')
ax[3].axhline(y=0.0, color='k', ls='--')


for i in range(N_Clust):
    x = [X[j,0] for j in range(X.shape[0]) if HCPCPredict[j] == i]
    y = [X[j,1] for j in range(X.shape[0]) if HCPCPredict[j] == i]
    centroid = [np.mean(x), np.mean(y)]
    confidence_ellipse(np.array(x), np.array(y), dataset_size=X.shape[0], ax=ax[0], edgecolor=clr[i])
    confidence_ellipse(np.array(x), np.array(y), dataset_size=X.shape[0], ax=ax[1], edgecolor=clr[i])
    confidence_ellipse(np.array(x), np.array(y), dataset_size=X.shape[0], ax=ax[2], edgecolor=clr[i])
    confidence_ellipse(np.array(x), np.array(y), dataset_size=X.shape[0], ax=ax[3], edgecolor=clr[i])

### Add 1.5mM and 4mM 
for i in range(len(Y)):
    if condition1 in labels[i]:
        ax[0].scatter(Y[:,0][i], Y[:,1][i], c='k')
    if condition2 in labels[i]:
        ax[0].scatter(Y[:,0][i], Y[:,1][i], c='r')

[ax[0].plot((Ca15[i][0], Ca4[i][0]), (Ca15[i][1], Ca4[i][1]), 'k', lw=0.8, alpha=0.3) for i in range(len(Ca15))]

ax[0].set_title('1.5mM vs 4mM')
ax[1].set_title('PC vs nonPC')
ax[2].set_title('Males vs Females')
ax[3].set_title('Set1 vs Set2')

# df_pca = pd.DataFrame(data_25mM, columns=list(data_25mM.drop(['PPR2/1'],axis=1).columns))
# figure, correlation_matrix = plot_pca_correlation_graph(df_pca, df_pca.columns, dimensions=(1,2), figure_axis_size=5)
# amp_axis_dim1 = np.mean([correlation_matrix['Dim 1']['AMP1'], correlation_matrix['Dim 1']['AMP2']])
# amp_axis_dim2 = np.mean([correlation_matrix['Dim 2']['AMP1'], correlation_matrix['Dim 2']['AMP2']])
# fail_axis_dim1 = np.mean([correlation_matrix['Dim 1']['%Fail1'], correlation_matrix['Dim 1']['%Fail2']])
# fail_axis_dim2 = np.mean([correlation_matrix['Dim 2']['%Fail1'], correlation_matrix['Dim 2']['%Fail2']])
# ppr_axis_dim1 = np.mean(correlation_matrix.iloc[2:11,0])
# ppr_axis_dim2 = np.mean(correlation_matrix.iloc[2:11,1])

# plt.figure()
# plt.scatter([amp_axis_dim1, fail_axis_dim1, ppr_axis_dim1, -ppr_axis_dim1],
#             [amp_axis_dim2, fail_axis_dim2, ppr_axis_dim2, -ppr_axis_dim2])
# plt.plot([amp_axis_dim1, fail_axis_dim1],
#          [amp_axis_dim2, fail_axis_dim2])
# plt.plot([ppr_axis_dim1, -ppr_axis_dim1],
#          [ppr_axis_dim2, -ppr_axis_dim2])
# plt.axhline(y=0)
# plt.axvline(x=0)


####### PROPORTIONS Males vs Females ########
#############################################
plt.figure()
plt.pie([gender.value_counts()['F'], gender.value_counts()['M']], labels=['Females','Males'])
print(gender.value_counts())

############## VECTOR PLOT ##################
#############################################

dist = pdist(X, metric='euclidean')
max_dist = np.max(dist)
values_moved = Ca4-Ca15
euclidean_dist = cdist(Ca15, Ca4, metric='euclidean')
euclidean_pairs = np.diagonal(euclidean_dist)
norm_euclidean_pairs = euclidean_pairs/max_dist

middle_vector = (Ca15+Ca4)/2
Ca15_moved = Ca15-middle_vector
Ca4_moved = Ca4-middle_vector

mean_vector = np.mean(norm_euclidean_pairs)

# mean_Ca15 = np.mean(Ca15, axis=0)
# mean_Ca4 = np.mean(Ca4, axis=0)

ANGLE_RAD_15, ANGLE_RAD_4 = [],[]
for i in range(len(values_moved)):
    rad_15 = atan2(Ca15_moved[i][0], Ca15_moved[i][1])
    rad_4 = atan2(Ca4_moved[i][0], Ca4_moved[i][1])
    ANGLE_RAD_15.append(rad_15)
    ANGLE_RAD_4.append(rad_4)

LIST_ANGLES = [ANGLE_RAD_15, ANGLE_RAD_4]
palette = ['blue','red']

bins_number = 30
bins = np.linspace(-np.pi, np.pi, bins_number+1)
width = 2*np.pi/bins_number

fig_vector = plt.figure(figsize=(4,4))
ax_vector = fig_vector.add_subplot(111, projection='polar')
for i in range(len(LIST_ANGLES)):
    a,b = np.histogram(LIST_ANGLES[i], bins)
    [ax_vector.bar(bins[:bins_number], a, width=width, bottom=0.0, color=palette[i]) for j in range(bins_number)]
    # [ax_vector.plot([0,ANGLE_RAD[i]], [0,norm_euclidean_pairs[i]], color='k', alpha=0.2) for i in range(len(ANGLE_RAD))]
    # ax_vector.plot([0,np.mean(ANGLE_RAD)], [0,np.mean(norm_euclidean_pairs)], color='r')
    ax_vector.set_ylim(0,max(a)+1)
ax_vector.set_theta_zero_location("N")
ax_vector.set_theta_direction(-1)

# [ax_vector.plot([0, values_moved[i][0]], [0, values_moved[i][1]], color='k', alpha=0.2) for i in range(len(Ca15))]
# # ax_vector.plot([0,mean_values_moved[0]], [0,mean_values_moved[1]], color='r', lw=3)
# ax_vector.add_patch(Ellipse((0,0), width=2, height=2, facecolor='none', edgecolor='k'))
# [ax_vector.plot([Ca15[i][0], Ca4[i][0]], [Ca15[i][1], Ca4[i][1]], color='k', alpha=0.2) for i in range(len(Ca15))]
# # ax_vector.plot([mean_Ca15[0],mean_Ca4[0]], [mean_Ca15[1],mean_Ca4[1]], color='r', lw=3)

# ax_vector.axhline(y=0, color='k', ls='--')
# ax_vector.axvline(x=0, color='k', ls='--')
# ax_vector.set_xlim(-1, 1)
# ax_vector.set_ylim(-1, 1)



#############################################
####### STABILITY iGLUSNFR OVER TIME ########
#############################################

file = r'C:\Users\ruo4037\OneDrive - Northwestern University\Desktop\PhD paper\GluSnFR_avg_variables_profiles_20Hz_2.5mM_set1_set2.xlsx'
sheets = ['Set1_t0','Set2_t18','Set1_split','Set2_split']

Y_SETS = []
for sheet in sheets:
    data_set = pd.read_excel(file, sheet_name=sheet).drop(['ID', 'AMP3', 'AMP4', 'AMP5', 'AMP6', 'AMP7', 'AMP8', 'AMP9', 'AMP10', '%Fail3'], axis=1)
    data_set_scaled = scaling.transform(data_set)
    Y_set = PCA_fit.transform(data_set_scaled)
    Y_SETS.append(Y_set)

    for i in range(len(Y_set)):
        if sheet == sheets[0] or sheet == sheets[2]:
            ax[3].scatter(Y_set[:,0][i], Y_set[:,1][i], c='k')
        if sheet == sheets[1] or sheet == sheets[3]:
            ax[3].scatter(Y_set[:,0][i], Y_set[:,1][i], c='r')
    
[ax[3].plot((Y_SETS[0][i][0], Y_SETS[1][i][0]), (Y_SETS[0][i][1], Y_SETS[1][i][1]), 'k', lw=1) for i in range(len(Y_SETS[0]))]
[ax[3].plot((Y_SETS[2][i][0], Y_SETS[3][i][0]), (Y_SETS[2][i][1], Y_SETS[3][i][1]), 'k', lw=1) for i in range(len(Y_SETS[2]))]


#############################################
################# IN vs PC ##################
#############################################

# df_IN = pd.DataFrame(IN, columns=['PC1','PC2'])
# df_PC = pd.DataFrame(PC, columns=['PC1','PC2'])

# df = [df_IN, df_PC]
# for frame in df:
#     sns.jointplot(x='PC1', y='PC2', data=frame,
#                   height=5, ratio=5, space=0, color='#383C65').plot_joint(sns.kdeplot, zorder=0, shade=True, thresh=0,
#                                                                           levels=100, cmap='mako',
#                                                                           legend=True, cbar=False, cbar_kws={})


# data_25mM.insert(2, 'PPR1/1', np.ones(data_25mM.shape[0]), True)
# ppr_PC = [data_25mM.iloc[i,2:12] for i in range(data_25mM.shape[0]) if targets[i] == 'PC']
# ppr_IN = [data_25mM.iloc[i,2:12] for i in range(data_25mM.shape[0]) if targets[i] == 'IN']

# ppr_PC_mean = np.mean(ppr_PC, axis=0)
# ppr_IN_mean = np.mean(ppr_IN, axis=0)

# ppr_PC_sem = stats.sem(ppr_PC, axis=0)
# ppr_IN_sem = stats.sem(ppr_IN, axis=0)

# fig_targets_profiles, ax_targets_profiles = plt.subplots(2,2,figsize=(8,6), sharey='row',tight_layout=True)
# stim = np.arange(1,11)
# [ax_targets_profiles[0,0].plot(stim, ppr_PC[i], 'k', lw=4, alpha=0.1) for i in range(len(ppr_PC))]
# [ax_targets_profiles[0,1].plot(stim, ppr_IN[i], 'k', lw=4, alpha=0.1) for i in range(len(ppr_IN))]
# ax_targets_profiles[0,0].plot(stim, ppr_PC_mean, 'darkred', marker='o')
# ax_targets_profiles[0,0].fill_between(stim, ppr_PC_mean+ppr_PC_sem, ppr_PC_mean-ppr_PC_sem, color='darkred', alpha=0.4)
# ax_targets_profiles[0,1].plot(stim, ppr_IN_mean, 'b', marker='o')
# ax_targets_profiles[0,1].fill_between(stim, ppr_IN_mean+ppr_IN_sem, ppr_IN_mean-ppr_IN_sem, color='b', alpha=0.4)
# ax_targets_profiles[0,0].axhline(y=1.0, color='k', ls='--')
# ax_targets_profiles[0,1].axhline(y=1.0, color='k', ls='--')
# ax_targets_profiles[0,0].set_title('PC')
# ax_targets_profiles[0,1].set_title('IN')


# data_traces = pd.read_excel(f'{Path}/{file_traces}', sheet_name='20Hz_2,5mM')
# data_traces = data_traces.iloc[:,1:]
# # time = [data_traces.iloc[:,i].dropna() for i in range(data_traces.shape[1]) if 'time' in data_traces.columns[i]]
# # avg = [data_traces.iloc[:,i].dropna() for i in range(data_traces.shape[1]) if 'avg' in data_traces.columns[i]]


# if freq == '20Hz':
#     new_time = np.linspace(0., 0.62, num=620, endpoint=True)
# if freq == '50Hz':
#     new_time = np.linspace(0., 0.49, num=490, endpoint=True)
    
# AVG_TRACES, TIME_TRACES = [],[]
# for idx in range(data_traces.shape[1]):
#     if 'time' in data_traces.columns[idx]:
#         if '20190801' in data_traces.columns[idx]:
#             start = np.ravel(np.where(data_traces.iloc[:,idx].dropna() >= 0.9))[0]
#             end = np.ravel(np.where(data_traces.iloc[:,idx].dropna() <= 1.7))[-1]
#         else:
#             if freq == '20Hz':
#                 start = np.ravel(np.where(data_traces.iloc[:,idx].dropna() >= 0.4))[0] 
#                 end = np.ravel(np.where(data_traces.iloc[:,idx].dropna() <= 1.2))[-1] 
            
#             if freq == '50Hz':
#                 start = np.ravel(np.where(data_traces.iloc[:,idx].dropna() >= 0.4))[0]
#                 end = np.ravel(np.where(data_traces.iloc[:,idx].dropna() <= 0.9))[-1]
#         time = np.array(data_traces.iloc[start:end,idx])
#         time = time-time[0]
#         TIME_TRACES.append(time)
        
#     elif 'avg' in data_traces.columns[idx]:
#         trace = data_traces.iloc[start:end,idx]
#         leak = np.mean(trace[0:100])
#         trace = trace - leak
#         f = interp1d(time, trace)
#         y = f(new_time)
#         AVG_TRACES.append(y)


# traces_PC = [AVG_TRACES[i] for i in range(len(AVG_TRACES)) if targets[i] == 'PC']
# traces_IN = [AVG_TRACES[i] for i in range(len(AVG_TRACES)) if targets[i] == 'IN']

# mean_PC = np.mean(traces_PC, axis=0)
# mean_IN = np.mean(traces_PC, axis=0)
# sem_PC = stats.sem(traces_PC, axis=0)
# sem_IN = stats.sem(traces_IN, axis=0)
# ax_targets_profiles[1,0].plot(new_time, mean_PC, 'r')
# ax_targets_profiles[1,1].plot(new_time, mean_IN, 'b')
# ax_targets_profiles[1,0].fill_between(new_time, mean_PC+sem_PC, mean_PC-sem_PC, color='r', alpha=0.4)
# ax_targets_profiles[1,1].fill_between(new_time, mean_IN+sem_IN, mean_IN-sem_IN, color='b', alpha=0.4)


# clusters_PC = [HCPCPredict[i] for i in range(len(HCPCPredict)) if targets[i] == 'PC']
# clusters_IN = [HCPCPredict[i] for i in range(len(HCPCPredict)) if targets[i] == 'IN']

# for i in range(N_Clust):
#     mean_trace_PC = np.mean([traces_PC[j] for j in range(len(traces_PC)) if clusters_PC[j] == i], axis=0)
#     mean_trace_IN = np.mean([traces_IN[j] for j in range(len(traces_IN)) if clusters_IN[j] == i], axis=0)
#     sem_trace_PC = stats.sem([traces_PC[j] for j in range(len(traces_PC)) if clusters_PC[j] == i], axis=0)
#     sem_trace_IN = stats.sem([traces_IN[j] for j in range(len(traces_IN)) if clusters_IN[j] == i], axis=0)
    
#     ax_targets_profiles[2,0].plot(new_time, mean_trace_PC, c=clr[i])
#     ax_targets_profiles[2,1].plot(new_time, mean_trace_IN, c=clr[i])
    
#     ax_targets_profiles[2,0].fill_between(new_time, mean_trace_PC+sem_trace_PC, mean_trace_PC-sem_trace_PC, color=clr[i], alpha=0.4)
#     ax_targets_profiles[2,1].fill_between(new_time, mean_trace_IN+sem_trace_IN, mean_trace_IN-sem_trace_IN, color=clr[i], alpha=0.4)
    
    
    
    
    
    
    