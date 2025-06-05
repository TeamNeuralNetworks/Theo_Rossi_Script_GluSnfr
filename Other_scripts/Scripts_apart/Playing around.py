# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 11:16:30 2019

@author: Theo.ROSSI
"""
import numpy as np
from itertools import permutations

a=[[0,1,3],[4,2]]
a1=a[0]
a2=a[1]

a3=a1.copy()
a3.extend(a2)

perms=list(permutations(a3, 2))
print(perms)
# new_arr = []

# for i in perms:
#     if (i[0] in a1 and i[1] in a1) or (i[0] in a2 and i[1] in a2):
#         new_arr.append(1)
#     else:
#         new_arr.append(0)

# print(new_arr)
################ Distance matrix ##########################
###########################################################
# import numpy as np
# import pylab
# import scipy.cluster.hierarchy as sch
# from scipy.spatial.distance import squareform, pdist



# # Generate random features and distance matrix.
# np.random.seed(0)
# x = np.random.rand(40)
# y = np.random.rand(40)
# D = np.zeros([40,40])
# for i in range(40):
#     for j in range(40):
#         D[i,j] = np.sqrt(((x[j] - x[i])**2)+((y[j]-y[i])**2))

# condensedD = squareform(D)

# # Compute and plot first dendrogram.
# fig = pylab.figure(figsize=(8,8))
# ax1 = fig.add_axes([0.09,0.1,0.2,0.6])
# Y = sch.linkage(condensedD, method='ward', optimal_ordering=True)
# Z1 = sch.dendrogram(Y, orientation='left')
# ax1.set_xticks([])
# ax1.set_yticks([])

# # Compute and plot second dendrogram.
# ax2 = fig.add_axes([0.3,0.71,0.6,0.2])
# Y = sch.linkage(condensedD, method='ward', optimal_ordering=True)
# Z2 = sch.dendrogram(Y)
# ax2.set_xticks([])
# ax2.set_yticks([])

# # Plot distance matrix.
# axmatrix = fig.add_axes([0.3,0.1,0.6,0.6])
# idx1 = Z1['leaves']
# idx2 = Z2['leaves']
# D = D[idx1,:]
# D = D[:,idx2]
# im = axmatrix.matshow(D, aspect='auto', origin='lower', cmap=pylab.cm.YlGnBu)
# axmatrix.set_xticks([])
# axmatrix.set_yticks([])

# # Plot colorbar.
# axcolor = fig.add_axes([0.91,0.1,0.02,0.6])
# pylab.colorbar(im, cax=axcolor)








# Path = r'E:\AAVDJ.GluSnFR-S72A\iGluSnFR-S72A_LFD'

# files = sorted(os.listdir(Path))

# REC1, REC2 = [],[]
# for file in files:
#     if 'dF_F0.xlsx' in file:
#         if 'linescan1' in file:
#             if 'recovery1' in file:
#                 rec1 = pd.read_excel('{}/{}'.format(Path, file))
#                 trace = rec1['Average']
#                 time1 = rec1['Time']
#                 REC1.append(trace)
                    
#             elif 'recovery2' in file:
#                 rec2 = pd.read_excel('{}/{}'.format(Path, file))
#                 trace = rec2['Average']
#                 time2 = rec2['Time']
#                 REC2.append(trace)

# # time2 = time2 + time1.iloc[-1] + (time1.iloc[-1] - time1.iloc[-2])

# mean_rec1 = np.mean(REC1, axis=0)
# mean_rec2 = np.mean(REC2, axis=0)

# whole_trace = np.concatenate((mean_rec1, mean_rec2), axis=0)
# whole_time = np.concatenate((time1, time2), axis=0)

# df1 = pd.concat([pd.DataFrame(mean_rec1), time1], axis=1)
# df1.columns = ['Avg_trace', 'Time']

# df2 = pd.concat([pd.DataFrame(mean_rec2), time2], axis=1)
# df2.columns = ['Avg_trace', 'Time']

# with pd.ExcelWriter('{}/20210928_linescan1_recovery1AVG_50Hz_3pulses_2.5mMCa_dF_F0.xlsx'.format(Path)) as writer:
#     df1.to_excel(writer)

# with pd.ExcelWriter('{}/20210928_linescan1_recovery2AVG_50Hz_3pulses_2.5mMCa_dF_F0.xlsx'.format(Path)) as writer:
#     df2.to_excel(writer)
                





# import matplotlib.pyplot as plt
# import pandas as pd
# from sklearn.datasets import load_iris

# iris = load_iris()

# X = iris.data
# y = iris.target

# plt.scatter(X[:,0], X[:,1], c=y, alpha=0.8)

# df = pd.concat([pd.DataFrame(X, columns = ['Var1', 'Var2', 'Var3', 'Var4']), pd.DataFrame(y, columns = ['Target'])], axis=1)
# with pd.ExcelWriter('E:\Data_divers\Iris_dataset.xlsx') as writer:
#     df.to_excel(writer)

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import scipy.stats as stats

# file1 = r'E:\AAVDJ.GluSnFR-S72A\AAVDJ.GluSnFR-S72A_Good_CaMg_ratio\Controle_electrophy_Ca\Parameters_1.5vs4_20Hz.xlsx'
# file2 = r'E:\AAVDJ.GluSnFR-S72A\AAVDJ.GluSnFR-S72A_Good_CaMg_ratio\Controle_electrophy_Ca\Parameters_1.5vs4_50Hz.xlsx'

# data1_15 = pd.read_excel(file1).iloc[:,7].values
# data1_4 = pd.read_excel(file1).iloc[:,8].values

# data2_15 = pd.read_excel(file2).iloc[:,7].values
# data2_4 = pd.read_excel(file2).iloc[:,8].values

# df = pd.concat((pd.DataFrame(data2_15), pd.DataFrame(data1_15), pd.DataFrame(data2_4), pd.DataFrame(data1_4)), axis=1)
# df.columns = ['50Hz_1.5mM', '20Hz_1.5mM', '50Hz_4mM', '20Hz_4mM']


# plt.plot([0,1], [np.mean(df['50Hz_1.5mM']), np.mean(df['20Hz_1.5mM'])], marker='o', color='c', label='1.5mM')
# plt.plot([0,1], [np.mean(df['50Hz_4mM']), np.mean(df['20Hz_4mM'])], marker='o', color='purple', label='4mM')
# plt.errorbar(0,np.mean(df['50Hz_1.5mM']),yerr=stats.sem(data2_15), ecolor='c',capsize=2)
# plt.errorbar(0,np.mean(df['50Hz_4mM']),yerr=stats.sem(data2_4), ecolor='purple',capsize=2)
# plt.errorbar(1,np.mean(df['20Hz_1.5mM']),yerr=stats.sem(data1_15), ecolor='c',capsize=2)
# plt.errorbar(1,np.mean(df['20Hz_4mM']),yerr=stats.sem(data1_4), ecolor='purple',capsize=2)
# plt.ylim(0,2.5)
# plt.axhline(y=1.0, linestyle='--')
# plt.ylabel('Mean PPR (A2/A1)')
# plt.xlabel('Δt (ms)')
# plt.xticks([0,1],['20','50'])
# plt.legend()


# import numpy as np

# import mne
# from mne import io
# from mne.stats import permutation_t_test

# data_path = r'D:\mne_data\MNE-sample-data\MEG\sample'
# raw_fname = data_path + '\sample_audvis_filt-0-40_raw.fif'
# event_fname = data_path + '\sample_audvis_filt-0-40_raw-eve.fif'
# event_id = 1
# tmin = -0.2
# tmax = 0.5

# #   Setup for reading the raw data
# raw = io.read_raw_fif(raw_fname)
# events = mne.read_events(event_fname)

# # pick MEG Gradiometers
# picks = mne.pick_types(raw.info, meg='grad', eeg=False, stim=False, eog=True,
#                        exclude='bads')
# epochs = mne.Epochs(raw, events, event_id, tmin, tmax, picks=picks,
#                     baseline=(None, 0), reject=dict(grad=4000e-13, eog=150e-6))
# data = epochs.get_data()
# times = epochs.times

# temporal_mask = np.logical_and(0.04 <= times, times <= 0.06)
# data = np.mean(data[:, :, temporal_mask], axis=2)

# n_permutations = 50000
# T0, p_values, H0 = permutation_t_test(data, n_permutations, n_jobs=1)

# significant_sensors = picks[p_values <= 0.05]
# significant_sensors_names = [raw.ch_names[k] for k in significant_sensors]

# print("Number of significant sensors : %d" % len(significant_sensors))
# print("Sensors names : %s" % significant_sensors_names)
















# import neo
# import numpy as np
# import matplotlib.pyplot as plt

# my_file = neo.io.WinWcpIO(r'E:\AAVDJ.GluSnFR-S72A\AAVDJ.GluSnFR-S72A_Good_CaMg_ratio\Controle_electrophy_Ca\Electrophy_traces\20210616_PC6_3x50Hz_100µA_4mM.wcp')

# REC, TIME = [],[]
# bl = my_file.read_block()
# for episode in bl.segments :
#     time = episode.analogsignals[0].times #The time vector
#     TIME.append(time)
#     rec = episode.analogsignals[0].magnitude #The signal vector 
#     REC.append(rec) 

# for i in range(len(REC)):
#     REC[i] = np.array(np.squeeze(REC[i]))
#     TIME[i] = np.array(np.squeeze(TIME[i]))
# global sampling
# sampling=float(TIME[1][1])*1000  #sampling rate in ms

# plt.plot(TIME[17],REC[17])



# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd
# from sklearn.datasets import load_digits
# from sklearn.manifold import TSNE
# digits = load_digits()
# data_X = digits.data[:86]
# y = digits.target[:86]
# tsne = TSNE()
# tsne_obj= tsne.fit_transform(data_X)
# tsne_df = pd.DataFrame({'X':tsne_obj[:,0],
#                         'Y':tsne_obj[:,1],
#                         'digit':y})
# tsne_df.head()
# sns.scatterplot(x="X", y="Y",
#               hue="digit",
#               palette=['purple','red','orange','brown','blue',
#                        'dodgerblue','green','lightgreen','darkcyan', 'black'],
#               legend='full',
#               data=tsne_df)



# import matplotlib.pyplot as plt
# import pandas as pd
# import os
# import numpy as np
# from scipy import signal
# from scipy.signal import savgol_filter

# Path = r'E:\AAVDJ.GluSnFR-S72A\AAVDJ.GluSnFR-S72A_Good_CaMg_ratio\StimML\20210304_linescan2_20Hz_10pulses_2.5mMCa_button1_dF_F0_Profile.xlsx'

# data = pd.read_excel(Path,header=0)
# data = data.iloc[:,0]

# noise = data.iloc[:,0].values
# amp = data.iloc[:,1].values
# amp = savgol_filter(amp,5,2)
# plt.figure()
# plt.hist(noise,bins=15,color='k',alpha=0.3)
# plt.hist(amp,bins=15,color='r',alpha=0.3)
# plt.plot(amp,color='b')


#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------


# import matplotlib 
# import numpy as np 
# import matplotlib.pyplot as plt 
   
# np.random.seed(10**7) 
# mu = 121 
# sigma = 21
# x = mu + sigma * np.random.randn(1000) 
   
# num_bins = 100
   
# n, bins, patches = plt.hist(x, num_bins,  
#                             density = True,  
#                             color ='green', 
#                             alpha = 0.7) 
   
# y = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
#       np.exp(-0.5 * (1 / sigma * (bins - mu))**2)) 
  
# plt.plot(bins, y, '--', color ='black') 
  
# plt.xlabel('X-Axis') 
# plt.ylabel('Y-Axis') 
  
# plt.title('matplotlib.pyplot.hist() function Example\n\n', 
#           fontweight ="bold") 

#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------

# files = sorted(os.listdir(Path))
# plt.figure()
# for i in range(len(files)):
#     if '20210112_linescan1_20Hz_10pulses_2.5mMCa_button4_dF_F0_Profile' in files[i]:
#         data = pd.read_excel('{}/{}'.format(Path,files[i]))
#         for j in range(len(data.columns)):
#             if 'Time' == data.columns[j]:
#                 time = data.iloc[:,j].values
#             # elif 'Average' == data.columns[j]:
#             #     average = data.iloc[:,j].values
#             #     filtered_average = signal.medfilt(average)
#         ep1 = data.iloc[:,0].values
#         ep2 = data.iloc[:,1].values
#         average = (ep1+ep2)/2
#         filtered_average = signal.medfilt(average, kernel_size=3)
#         # plt.plot(time,average,'k', alpha=0.5)
#         plt.plot(time,filtered_average,'blue')
    # if '20210112_linescan1_20Hz_10pulses_2.5mMCa_button2_dF_F0_Profile' in files[i]:
    #     data2 = pd.read_excel('{}/{}'.format(Path,files[i]))
    #     for j in range(len(data2.columns)):
    #         if 'Time' == data2.columns[j]:
    #             time2 = data2.iloc[:,j].values
    #         elif 'Average' == data2.columns[j]:
    #             average2 = data2.iloc[:,j].values
    #             filtered_average2 = signal.medfilt(average2)
    #     # plt.plot(time2,average2,'k',alpha=0.5)
    #     plt.plot(time,filtered_average2,'orange')
    # if '20210112_linescan1_20Hz_10pulses_2.5mMCa_button3_dF_F0_Profile' in files[i]:
    #     data3 = pd.read_excel('{}/{}'.format(Path,files[i]))
    #     for j in range(len(data3.columns)):
    #         if 'Time' == data3.columns[j]:
    #             time3 = data3.iloc[:,j].values
    #         elif 'Average' == data3.columns[j]:
    #             average3 = data3.iloc[:,j].values
    #             filtered_average3 = signal.medfilt(average3,kernel_size=15)
    #     # plt.plot(time3,average3,'k',alpha=0.5)
    #     plt.plot(time,filtered_average3,'yellowgreen')
    # if '20210112_linescan1_20Hz_10pulses_2.5mMCa_button4_dF_F0_Profile' in files[i]:
    #     data4 = pd.read_excel('{}/{}'.format(Path,files[i]))
    #     for j in range(len(data4.columns)):
    #         if 'Time' == data4.columns[j]:
    #             time4 = data4.iloc[:,j].values
    #         elif 'Average' == data4.columns[j]:
    #             average4 = data4.iloc[:,j].values
    #             filtered_average4 = signal.medfilt(average4)
    #     # plt.plot(time4,average4,'k',alpha=0.5)
    #     plt.plot(time,filtered_average4,'red')




# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import os

# Path = r'E:\AAVDJ.GluSnFR-S72A_PFs_buttons\StimML\Figures'
# files = sorted(os.listdir(Path))

# plt.figure()
# x = np.arange(1,11,1)
# for file in range(len(files)):
#     if 'Clust1' in files[file]:
#         data = pd.read_excel('{}/{}'.format(Path,files[file]))
#         data_clust1 = data.iloc[:,1:11].values
#         for row in range(data_clust1.shape[0]):
#             plt.plot(x,data_clust1[row],'purple',marker='o',alpha=0.2)
#     if 'Clust2' in files[file]:
#         data = pd.read_excel('{}/{}'.format(Path,files[file]))
#         data_clust2 = data.iloc[:,1:11].values
#         for row in range(data_clust2.shape[0]):
#             plt.plot(x,data_clust2[row],'g',marker='o',alpha=0.2)

# avg_clust1 = np.mean(data_clust1,axis=0)
# avg_clust2 = np.mean(data_clust2,axis=0)
# plt.plot(x,avg_clust1,'purple',marker='o')
# plt.plot(x,avg_clust2,'g',marker='o')

#----------------------RAINBOW COLORS----------------------
#----------------------------------------------------------------------

# import matplotlib.cm as cm
# import matplotlib.patches as pa
# import matplotlib.pyplot as plt
# import numpy as np

# plt.figure()
# num_colors = 3;
# colors = cm.viridis(np.linspace(0.1,0.9,num_colors))
# indices = np.arange(0,len(colors))
# indices = indices.reshape((len(indices),1))
# plt.imshow(colors[indices])


#----------------------DENSITY CLUSTERING: OPTICS----------------------
#----------------------------------------------------------------------
# import os
# import matplotlib.pyplot as plt
# from matplotlib import cm
# import matplotlib.gridspec as gridspec
# import numpy as np
# import pandas as pd
# import seaborn as sns
# from statannot import add_stat_annotation
# from scipy import stats
# import scipy.cluster.hierarchy as hierarchy 
# import sklearn
# np.random.seed(0)
# n_points_per_cluster = 250

# C1 = [-5, -2] + .8 * np.random.randn(n_points_per_cluster, 2)
# C2 = [4, -1] + .1 * np.random.randn(n_points_per_cluster, 2)
# C3 = [1, -2] + .2 * np.random.randn(n_points_per_cluster, 2)
# C4 = [-2, 3] + .3 * np.random.randn(n_points_per_cluster, 2)
# C5 = [3, -2] + 1.6 * np.random.randn(n_points_per_cluster, 2)
# C6 = [5, 6] + 2 * np.random.randn(n_points_per_cluster, 2)
# X = np.vstack((C1, C2, C3, C4, C5,C6))
# clust = sklearn.cluster.OPTICS(min_samples=50, xi=.05, min_cluster_size=.05)

# # Run the fit
# clust.fit(X)

# labels_050 = sklearn.cluster.cluster_optics_dbscan(reachability=clust.reachability_,
#                                     core_distances=clust.core_distances_,
#                                     ordering=clust.ordering_, eps=0.5)
# labels_200 = sklearn.cluster.cluster_optics_dbscan(reachability=clust.reachability_,
#                                     core_distances=clust.core_distances_,
#                                     ordering=clust.ordering_, eps=2)

# space = np.arange(len(X))
# reachability = clust.reachability_[clust.ordering_]
# labels = clust.labels_[clust.ordering_]

# plt.figure(figsize=(10, 7))
# G = gridspec.GridSpec(2, 3)
# ax1 = plt.subplot(G[0, :])
# ax2 = plt.subplot(G[1, 0])
# ax3 = plt.subplot(G[1, 1])
# ax4 = plt.subplot(G[1, 2])

# # Reachability plot
# colors = ['g.', 'r.', 'b.', 'y.', 'c.']
# for klass, color in zip(range(0, 5), colors):
#     Xk = space[labels == klass]
#     Rk = reachability[labels == klass]
#     ax1.plot(Xk, Rk, color, alpha=0.3)
# ax1.plot(space[labels == -1], reachability[labels == -1], 'k.', alpha=0.3)
# ax1.plot(space, np.full_like(space, 2., dtype=float), 'k-', alpha=0.5)
# ax1.plot(space, np.full_like(space, 0.5, dtype=float), 'k-.', alpha=0.5)
# ax1.set_ylabel('Reachability (epsilon distance)')
# ax1.set_title('Reachability Plot')

# # OPTICS
# colors = ['g.', 'r.', 'b.', 'y.', 'c.']
# for klass, color in zip(range(0, 5), colors):
#     Xk = X[clust.labels_ == klass]
#     ax2.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3)
# ax2.plot(X[clust.labels_ == -1, 0], X[clust.labels_ == -1, 1], 'k+', alpha=0.1)
# ax2.set_title('Automatic Clustering\nOPTICS')

# # DBSCAN at 0.5
# colors = ['g', 'greenyellow', 'olive', 'r', 'b', 'c']
# for klass, color in zip(range(0, 6), colors):
#     Xk = X[labels_050 == klass]
#     ax3.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3, marker='.')
# ax3.plot(X[labels_050 == -1, 0], X[labels_050 == -1, 1], 'k+', alpha=0.1)
# ax3.set_title('Clustering at 0.5 epsilon cut\nDBSCAN')

# # DBSCAN at 2.
# colors = ['g.', 'm.', 'y.', 'c.']
# for klass, color in zip(range(0, 4), colors):
#     Xk = X[labels_200 == klass]
#     ax4.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3)
# ax4.plot(X[labels_200 == -1, 0], X[labels_200 == -1, 1], 'k+', alpha=0.1)
# ax4.set_title('Clustering at 2.0 epsilon cut\nDBSCAN')

# plt.tight_layout()
# plt.show()




#----------------------PCA:CHECK GOOD NUMB OF COMPONENTS---------------
#----------------------------------------------------------------------

# plt.figure()
# scaler = sklearn.preprocessing.MinMaxScaler()
# data_rescaled = scaler.fit_transform(PPR_TOTAL)
# pca = sklearn.decomposition.PCA().fit(data_rescaled)
# y = np.cumsum(pca.explained_variance_ratio_)
# plt.plot(x,y,marker='o', linestyle='--', color='b')
# plt.xlabel('Number of Components')
# plt.xticks(x)
# plt.ylabel('Cumulative variance (%)')
# plt.title('The number of components needed to explain variance')

# plt.axhline(y=0.95, color='r', linestyle='-')
# plt.text(0.5, 0.85, '95% cut-off threshold', color = 'red', fontsize=16)

# plt.grid(axis='x')

#----------------------------------------------------------------------
#----------------------------------------------------------------------

# import matplotlib.pyplot as plt
# import pandas as pd
# import os
# import numpy as np
# Path = r'E:\AAVDJ.GluSnFR-S72A_EletrophyVsGluSnFR\GluSnFR'
# file = '20201112_linescan2_50Hz_3pulses_2.5mMCa_button2_dF_F0_50microA_Profile'
# df=pd.read_excel('{}/{}.xlsx'.format(Path,file))
# PROFILE=[]
# for j in range(len(df.columns)):
#     if 'Average' == df.columns[j]:
#         average = df.iloc[:,j].values
#     elif 'Time' == df.columns[j]:
#         time = df.iloc[:,j].values
#     else:
#         profile = df.iloc[:,j].values
#         PROFILE.append(profile)

# for i in range(len(PROFILE)):
#     plt.figure(figsize=(12,4))
#     plt.plot(time,PROFILE[i],'k',alpha=0.6)
#     plt.savefig('{}/Figures/{}_sweep{}.pdf'.format(Path,file,i))
# plt.figure(figsize=(12,4))
# plt.plot(time,average,'limegreen')
# plt.savefig('{}/Figures/{}_average.pdf'.format(Path,file))

    
# amp=df.iloc[0,1:].values
# PPR=[]
# for i in range(amp.shape[0]):
#     ppr = amp[i]/amp[0]
#     PPR.append(ppr)
# x = np.arange(1,11,1)
# plt.figure()
# plt.plot(x,PPR,marker='o')
# files=sorted(os.listdir(Path))
# for file in range(len(files)):
#     if '20Hz' in files[file]:
#         if 'xlsx' in files[file]:
#             if 'Amp' in files[file]:
#                 data=pd.read_excel('{}/{}'.format(Path,files[file]))
#                 amp=data.iloc[0,1:].values
#                 PPR=[]
#                 for i in range(amp.shape[0]):
#                     ppr = amp[i]/amp[0]
#                     PPR.append(ppr)
#                 x = np.arange(1,11,1)
#                 plt.figure()
#                 plt.plot(x,PPR,marker='o')
#                 plt.title('{}'.format(files[file]))
# import pandas as pd
# import matplotlib.pyplot as plt

# file = r'E:\AAVDJ.GluSnFR-S72A_Ca2+_1.5vs4\20200909_linescan1_20Hz_10pulses_1.5mMCa_button3_dF.xlsx'

# df = pd.read_excel(file, header=0)
# average = df.iloc[1:,2].values
# plt.plot(average)

# from sklearn import preprocessing
# import numpy as np
# X_train = np.array([[ 1., -1.,  2.],
#                     [ 2.,  0.,  0.],
#                     [ 0.,  1., -1.]])
# X_scaled = preprocessing.scale(X_train)


# import matplotlib.pyplot as plt
# from scipy.optimize import curve_fit
# import numpy as np

# def func(x, a, b, c):
#     return a * np.exp(-b * x) + c

# xdata = np.linspace(0, 4, 50)
# y = func(xdata, 2.5, 1.3, 0.5)
# np.random.seed(1729)
# y_noise = 0.2 * np.random.normal(size=xdata.size)
# ydata = y + y_noise
# plt.plot(xdata, ydata, 'b-', label='data')

# popt, pcov = curve_fit(func, xdata, ydata)
# plt.plot(xdata, func(xdata, *popt), 'r-',
#          label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))

# popt, pcov = curve_fit(func, xdata, ydata, bounds=(0, [3., 1., 0.5]))
# plt.plot(xdata, func(xdata, *popt), 'g--',
#          label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))


# import numpy as np
# import pylab
# import scipy.optimize as opt

# def sigmoid(x, a, b):
#      return 1.0 / (1.0 + np.exp(-a*(x-b)))

# xdata = np.array([400, 600, 800, 1000, 1200, 1400, 1600])
# ydata = np.array([0, 0, 0.13, 0.35, 0.75, 0.89, 0.91])

# popt, pcov = opt.curve_fit(sigmoid, xdata, ydata, bounds=([0., 700.],[0.01, 800.]))
# print(popt)

# x = np.linspace(-1, 2000, 50)
# y = sigmoid(x, *popt)

# pylab.plot(xdata, ydata, 'o', label='data')
# pylab.plot(x,y, label='fit')
# pylab.ylim(0, 1.05)
# pylab.legend(loc='best')
# pylab.show()

# import numpy as np
# import pylab
# from scipy.optimize import curve_fit

# def sigmoid(x, x0, k):
#      y = 1 / (1 + np.exp(-k*(x-x0)))
#      return y

# xdata = np.array([0.0,   1.0,  3.0, 4.3, 7.0,   8.0,   8.5, 10.0, 12.0])
# ydata = np.array([0.01, 0.02, 0.04, 0.11, 0.43,  0.7, 0.89, 0.95, 0.99])

# popt, pcov = curve_fit(sigmoid, xdata, ydata, bounds=([0, 0],[10, 1]))
# print (popt)

# x = np.linspace(-1, 15, 50)
# y = sigmoid(x, *popt)

# pylab.plot(xdata, ydata, 'o', label='data')
# pylab.plot(x,y, label='fit')
# pylab.ylim(0, 1.05)
# pylab.legend(loc='best')
# pylab.show()


# import numpy as np
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# dictionary = {'Banner':['Type1']*10+['Type2']*10,
#               'Northen_californina':np.random.rand(20),
#               'Texas':np.random.rand(20)}
# df = pd.DataFrame(dictionary)
# df = pd.melt(df,id_vars=['Banner'],value_vars=['Northen_californina','Texas'], var_name='zone', value_name='amount')
# plt.figure(figsize=(9,9)) #for a bigger image
# sns.boxplot(x="Banner", y="amount", hue="zone", data=df, palette="Set1")


# fig, ax = plt.subplots(1)
# mean = [np.mean(x) for x in LIST_VALUES]
# sem = [np.std(x)/np.sqrt(len(x)) for x in LIST_VALUES]

# for average, error, distribution, index in zip(mean, sem, LIST_VALUES, np.arange(len(labels))):
# #    print ('------------------')
# #    print(index,average,distribution)*
    
#     width = 0.2
#     step = 5.
    
#     ax.bar(index, average, width=width, label=labels[index], color=colors[index], alpha=0.2)
#     ax.errorbar(index, average, yerr=error, capsize=10, color='k')
#     ax.legend(loc='best')
    
#     x = np.linspace(index-(width/step), index+(width/step), len(distribution))
#     ax.scatter(x, distribution, color=colors[index], alpha=0.5)
#     ax.plot([-0.3,1.3], [1.,1.], 'k', linestyle='--', linewidth=1, alpha=0.5)
    
#     ax.set_title(Excel_file)
#     ax.set_xlabel('Condition'), ax.set_ylabel('PPR2/1')


#--------------INTERPOLATION FOR AVERAGING SET OF DATA WITH DIFFERENT SAMPLING RATE---------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------------

# import numpy as np
# from scipy.interpolate import interp1d
# import matplotlib.pyplot as plt

# # make up three datasets for testing
# x1 = np.linspace(0, 10, num=11, endpoint=True)
# x2 = np.linspace(0, 10, num=13, endpoint=True)
# x3 = np.linspace(0, 10, num=23, endpoint=True)

# y1 = np.cos(-x1**2/9.0) + 0.2*np.random.rand((len(x1)))
# y2 = np.cos(-x2**2/9.0) + 0.2*np.random.rand((len(x2)))
# y3 = np.cos(-x3**2/9.0) + 0.2*np.random.rand((len(x3)))

# # interpolate data
# f1 = interp1d(x1, y1,'cubic')
# f2 = interp1d(x2, y2,'cubic')
# f3 = interp1d(x3, y3,'cubic')

# # define common carrier for calculation of average curve
# x_all = np.linspace(0, 10, num=101, endpoint=True)

# # evaluation of fits on common carrier
# f1_int = f1(x_all)
# f2_int = f2(x_all)
# f3_int = f3(x_all)

# # put all fits to one matrix for fast mean calculation
# data_collection = np.vstack((f1_int,f2_int,f3_int))

# # calculating mean value
# f_avg = np.average(data_collection, axis=0)

# # plot this example
# plt.figure()
# plt.plot(x1,y1,'ro',label='row1')
# plt.plot(x2,y2,'bo',label='row2')
# plt.plot(x3,y3,'go',label='row3')

# plt.plot(x_all,f1_int,'r-',label='fit1')
# plt.plot(x_all,f2_int,'b-',label='fit2')
# plt.plot(x_all,f3_int,'g-',label='fit3')

# plt.plot(x_all, f_avg,'k--',label='fit average')
# plt.legend(loc=3)


#--------------RESAMPLING SET OF DATA WITH DIFFERENT SAMPLING RATE---------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------------

# from scipy import signal

# x = np.linspace(0, 10, 20, endpoint=False)
# y = np.cos(-x**2/6.0)
# f = signal.resample(y, 100)
# xnew = np.linspace(0, 10, 100, endpoint=False)

# import matplotlib.pyplot as plt
# plt.plot(x, y, 'go-', xnew, f, '.-', 10, y[0], 'ro')
# plt.legend(['data', 'resampled'], loc='best')



#-----------------------------------PIE PLOT-----------------------------------
# import matplotlib.pyplot as plt

# proportions = [45, 41, 18]
# profiles = ['STP1', 'STP2', 'STP3']
# colors = ['purple', 'g', 'yellowgreen']

# plt.pie(proportions, labels=profiles, colors=colors, startangle=90, autopct='%.1f%%')


#------------------------------------LOOPS-------------------------------------

#start_game = True
#
#while start_game:
#    menu_choice = input(">")
#    
#    if menu_choice == "again":
#        continue
#        '''Permet de revenir au début de la boucle et de la continuer'''
#        
#    elif menu_choice == "quit":
#        start_game = False
#        '''Possibilité de remplacer par break. Opposition à la condition initiale
#        donc on sort de la boucle'''
#    
#    elif menu_choice == "hello":
#        print("Bonjour!")
#    
#    else:
#        print("Commande introuvable")
#
#print("A bientôt")


#---------------------------------FUNCTIONS------------------------------------

#def say(name, message):
#    print("{}:{}".format(name, message))
#
#say("Jak", "Fuck you asshole")
#say("Craptri", "Qu'est-ce que tu as dit???")



#def show_inventory(*list_items):
#    for item in list_items:
#        print(item)
#
#show_inventory("sword")
#show_inventory("sword", "shield", "bow", "mana potion")



#def poop(number1, number2):
#    if number1 < number2:
#        return "number1 < number2"
#    elif number1 > number2:
#        return "number1 > number2"
#    else:
#        return "number1 = number2"
#
#print(poop(6,6))



#TTC = lambda prixHT: prixHT + (prixHT*20/100)
#print(TTC(20))


#-------------------------------ERROR MANAGEMENT-------------------------------

#age_user = input("How old are you?")
#
#try:
#    age_user =int(age_user)
#except:
#    print("Age error")
#else:
#    print("I am", age_user, "old")
#finally:
#    print("END")
    


#number1 = 150
#
#number2 = input("Choose the number to divide : ")
#
#try:
#    number2 = int(number2)
#    print("Result = {}".format(number1/number2))
#except ZeroDivisionError:
#    print("You can't divide by 0")
#except ValueError:
#    print("You must write a number")
#except:
#    print("Value error")
#else:
#    print("Congrats you did it!")
#finally:
#    print("END")



#try:
#    age = input("How old are you? ")
#    age = int(age)
#    
#    if age < 25:
#        raise ZeroDivisionError("Did you see the condition nigga?")
#
#except ZeroDivisionError:
#    print("I caught your useless exception")



#try:
#    age = input("How old are you? ")
#    age = int(age)
#    
#    assert age < 25
#
#except AssertionError:
#    print("I caught the exception")


#-----------------------------CLASS AND ATTRIBUTS------------------------------

#class Human:
#    
#    humans_created = 0
#    
#    def __init__(self, c_first_name, c_age):
#        print("Creation of a human")
#        self.first_name = c_first_name
#        self.age = c_age
#        Human.humans_created += 1
#
#print("Starting program")
#
#h1 = Human("Theo", 24)
#print("Humans created : {}".format(Human.humans_created))
#
#
#h2 = Human("Ludo", 28)
#print("Humans created : {}".format(Human.humans_created))




#--------------------------- CREATING A CHARACTER -----------------------------

#class Personne:
#    '''Classe définissant un personnage D&D caractérisé par:
#        - son nom
#        - son prénom
#        - son sexe
#        - sa race
#        - sa classe
#        - son âge
#        - son lieu de résidence'''
#
#    def __init__(self, nom, prenom):
#        self.nom = nom
#        self.prenom = prenom
#        self.sex = 'Male'
#        self.race = 'Dragonborn'
#        self.classe = 'Sorcerer'
#        self.age = '15 ans'
#        self.lieu_residence = 'Strasbourg'


#--------------------------------- COMPTEUR -----------------------------------
        
#class Compteur:
#    '''Cette classe possède un attribut de classe qui s'incrémente à chaque fois que l'on crée un objet de ce type'''
#    objets_crees = 0 #Le compteur vaut 0 au départ
#    
#    def __init__(self):
#        '''A chaque fois qu'on créer un objet, on incrémente le compteur'''
#        Compteur.objets_crees += 1
#    
#    def combien(cls):
#        '''Méthode de classe affichant combien d'objets ont été créés'''
#        print('{} objets ont été créés.'.format(cls.objets_crees))
#    combien = classmethod(combien)


#--------------------------------- METHODS ------------------------------------

#class TableauNoir:
#    '''Classe définissant une surface sur laquelle on peut écrire,
#    que l'on peut lire et effacer, par jeu de méthodes.
#    L'attribut modifié est surface'''
#    
#    def __init__(self):
#        '''Par défaut, notre surface est vide'''
#        self.surface = ''
#    
#    def ecrire(self, message):
#        '''Méthode permettant d'écrire sur la surface du tableau.
#        Si la surface n'est pas vide, on saute une ligne avant de rajouter le message à écrire'''
#        if self.surface != '':
#            self.surface += '\n'
#        self.surface += message
#    
#    def lire(self):
#        '''Cette méthode se charge d'afficher le tableau'''
#        print(self.surface)
#    
#    def effacer(self):
#        '''Cette méthode se charge d'effacer le tableau'''
#        self.surface = ''


#class Human:
#    
#    living_place = "Earth"
#    
#    def __init__(self, name, age):
#        self.name = name
#        self.age = age
#    
#    def speak(self, message):
#        print("{} said : {}".format(self.name, message))
#    
#    def change_planet(cls, new_planet):
#        Human.living_place = new_planet
#    
#    change_planet = classmethod(change_planet)
#    
#    def define():
#        print("Humans are selfish")
#    
#    definition = staticmethod(define)
#
#h1 = Human("Theo", 24)
#print("Actual planet : {}".format(Human.living_place))
#
#
#Human.change_planet("Mars") 
#print("Actual planet : {}".format(Human.living_place))
#
#Human.definition() 
        
#-----------------------------------GUI----------------------------------------

#import tkinter as tk
#
#root = tk.Tk()
#root.title('Simple example')
#root.rowconfigure(0, weight=1)
#root.columnconfigure(0, weight=1)
#qb = tk.Button(root, text='Quitter', command=root.quit)
#qb.grid(row=0, column=0, sticky='nsew')
#root.mainloop()

    
    
#def calcul():
#    Text.set("Résultat : "+"15")
#
## Fenêtre principale
#MyWindow = tk.Tk()
#
#MyWindow.title("Titre de la fenetre")
#MyWindow.geometry("300x200")
#
## Création d'un bouton
#ButtonCalcul = tk.Button(MyWindow, text ="Calculer", command = calcul)
#ButtonCalcul.grid(row=1,column=0, padx = 5, pady = 5)
#
## Création d'un bouton Exit
#ButtonExit = tk.Button(MyWindow, text ="Quitter", command = MyWindow.destroy)
#ButtonExit.grid(row=2,column=0, padx = 5, pady = 5)
#
## Création d'un label pour afficher du texte
#Text = tk.StringVar()
#LabelResultat = tk.Label(MyWindow, textvariable = Text , bg ="grey")
#Text.set("Résultat : ")
#LabelResultat.grid(row=1,column=1, padx = 5, pady = 5)
#
#MyWindow.mainloop() 
    
    
#vals = ['A', 'B', 'C']
#etiqs = ['trop chaud', 'trop froid', 'correct']
#varGr = tk.StringVar()
#varGr.set(vals[1])
#for i in range(3):
#    b = tk.Radiobutton(root, variable=varGr, text=etiqs[i], value=vals[i])
#    b.pack(side='left', expand=1)
#root.mainloop()
    
    
# Simple enough, just import everything from tkinter.
#import tkinter as tk
#
#
##download and install pillow:
## http://www.lfd.uci.edu/~gohlke/pythonlibs/#pillow
#from PIL import Image, ImageTk
#
#
## Here, we are creating our class, Window, and inheriting from the Frame
## class. Frame is a class from the tkinter module. (see Lib/tkinter/__init__)
#class Window(tk.Frame):
#
#    # Define settings upon initialization. Here you can specify
#    def __init__(self, master=None):
#        
#        # parameters that you want to send through the Frame class. 
#        tk.Frame.__init__(self, master)   
#
#        #reference to the master widget, which is the tk window                 
#        self.master = master
#
#        #with that, we want to then run init_window, which doesn't yet exist
#        self.init_window()
#
#    #Creation of init_window
#    def init_window(self):
#
#        # changing the title of our master widget      
#        self.master.title("GUI")
#
#        # allowing the widget to take the full space of the root window
#        self.pack(fill='both', expand=1)
#
#        # creating a menu instance
#        menu = tk.Menu(self.master)
#        self.master.config(menu=menu)
#
#        # create the file object)
#        file = tk.Menu(menu)
#
#        # adds a command to the menu option, calling it exit, and the
#        # command it runs on event is client_exit
#        file.add_command(label="Exit", command=self.client_exit)
#
#        #added "file" to our menu
#        menu.add_cascade(label="File", menu=file)
#
#
#        # create the file object)
#        edit = tk.Menu(menu)
#
#        # adds a command to the menu option, calling it exit, and the
#        # command it runs on event is client_exit
#        edit.add_command(label="Show Img", command=self.showImg)
#        edit.add_command(label="Show Text", command=self.showText)
#
#        #added "file" to our menu
#        menu.add_cascade(label="Edit", menu=edit)
#
#    def showImg(self):
#        load = Image.open("D:/Theo.ROSSI/GluSnFR/Virus_AAV.hSynap.SF-iGluSnFR_S72A/AAVDJ.GluSnFR-S72A/2019_10_31/20191031_14_10_25_linescan1_50Hz_[xy]T_ROI.tif")
#        render = ImageTk.PhotoImage(load)
#
#        # labels can be text or images
#        img = tk.Label(self, image=render)
#        img.image = render
#        img.place(x=0, y=0)
#
#
#    def showText(self):
#        text = tk.Label(self, text="Hey there good lookin!")
#        text.pack()
#        
#
#    def client_exit(self):
#        exit()
#
#
## root window created. Here, that would be the only window, but
## you can later have windows within windows.
#root = tk.Tk()
#
#root.geometry("400x300")
#
##creation of an instance
#app = Window(root)
#
#
##mainloop 
#root.mainloop()    

