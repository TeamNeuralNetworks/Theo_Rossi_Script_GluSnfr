# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 17:55:29 2020

@author: Theo.ROSSI
"""


import os
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as hierarchy
from scipy import stats
from sklearn import decomposition
from sklearn import preprocessing
from sklearn import cluster
import neo

#----------------------------SETTINGS--------------------------------
#--------------------------------------------------------------------
Path = r'D:\Data Theo\Analyse GC-PC\Final_Amp_EPSC_min_proj1+2.xlsx'
Data = 'ppr'
PrincCompAn = True
HierAscClass = True
OPTICS=False
N_COMP,N_CLUST = 7,2
CAH_threshold = 0.6
OPTICS_MinPts = 10

#--------------------------------------------------------------------
#--------------------------------------------------------------------

file = pd.read_excel(Path)

AMP, PPR = [],[]
amp = file.iloc[:,1:11].values
x = np.arange(1,11,1)
fig,ax = plt.subplots(1,2,figsize=(8,4),tight_layout=True)
for profile in range(len(amp)):
    AMP.append(amp[profile])
    ppr = amp[profile][:]/amp[profile][0]
    PPR.append(ppr)
    ax[0].plot(x,amp[profile],'k',marker='o',alpha=0.2)
    ax[1].plot(x,ppr,'k',marker='o',alpha=0.2)
    
# files = sorted(os.listdir(Path))
# CSV_FILES = []
# XLSX_FILES = []
# AMP, PPR_TOTAL = [],[]

# for file in range(len(files)):
#     if 'Amplitude' in files[file]:
#         if 'COMP' in files[file]:
#             continue
#         else:
#             if '.csv' in files[file]:
#                 CSV_FILES.append(files[file])
#                 data_csv = pd.read_csv('{}/{}'.format(Path,files[file]))
#                 data_csv = data_csv.columns[0].split()[1:11]
#                 SPLITTED_VALUES = []
#                 for i in range(len(data_csv)):
#                     SPLITTED_VALUES.append(data_csv[i])
#                 df = pd.DataFrame(SPLITTED_VALUES).transpose()
#                 xlsx_filename = files[file].split('.')[0]
#                 with pd.ExcelWriter('{}\{}.xlsx'.format(Path,xlsx_filename)) as writer:
#                     excel = df.to_excel(writer)
#             elif '.xlsx' in files[file]:
#                 XLSX_FILES.append(files[file])
#                 data_xlsx = pd.read_excel('{}/{}'.format(Path,files[file]),header=0)
#                 amp = data_xlsx.iloc[0,1:].values
#                 AMP.append(amp)
#                 PPR=[]
#                 for value in range(amp.shape[0]):
#                     ppr = amp[value]/amp[0]
#                     PPR.append(ppr)
#                 PPR_TOTAL.append(PPR)
                
# x = np.arange(1,11,1)
# fig,ax = plt.subplots(1,2,figsize=(8,4),tight_layout=True)
# for profile in range(len(AMP)):
#     ax[0].plot(x,AMP[profile],'k',marker='o',alpha=0.2)
#     ax[1].plot(x,PPR_TOTAL[profile],'k',marker='o',alpha=0.2)


if PrincCompAn == True:

    #Elbow plot for n_components in the PCA and bar plot for explained variance
    PCA_n_comp = decomposition.pca.PCA()
    IMP = preprocessing.Imputer(strategy='median')
    if Data == 'raw':
        norm = preprocessing.normalize(IMP.fit_transform(AMP))
    else:
        norm = preprocessing.normalize(IMP.fit_transform(PPR))
    PCA_n_comp_fit = PCA_n_comp.fit(norm)
    y = np.cumsum(PCA_n_comp_fit.explained_variance_ratio_)
    
    fig0,ax0=plt.subplots(1,2,figsize=(10,4),tight_layout=True)
    ax0[0].plot(np.arange(1,11,1),y,marker='o', linestyle='--', color='b')
    ax0[0].set_xlabel('#Component'),ax0[0].set_ylabel('Cumulative variance (%)')
    ax0[0].set_xticks(x)
    ax0[0].set_title('Number of components needed to explain variance')
    ax0[0].axhline(y=0.95, color='r', linestyle='-')
    ax0[0].text(0.5, 0.85, '95% cut-off threshold', color = 'red', fontsize=16)
    ax0[0].grid(axis='x')
    
    ax0[1].bar(np.arange(len(PCA_n_comp_fit.explained_variance_ratio_)) + 0.5,
            PCA_n_comp_fit.explained_variance_ratio_)
    ax0[1].set_title("Explained variance")
    ax0[1].set_ylabel("Norm variance"),ax0[1].set_xlabel("#Component")
    # fig0.savefig('{}/Figures/PCs_explained_variance.pdf'.format(Path))
    
    #PCA
    PCA = decomposition.pca.PCA(n_components=N_COMP)
    IMPUTER = preprocessing.Imputer(strategy='median')
    if Data == 'raw':
        center_normed_matrix = preprocessing.normalize(IMPUTER.fit_transform(AMP))
    else:
        center_normed_matrix = preprocessing.normalize(IMPUTER.fit_transform(PPR))
    PCA_fit = PCA.fit(center_normed_matrix)
    eigenvalue = PCA_fit.transform(center_normed_matrix)
    percent_explained = (PCA_fit.explained_variance_ratio_)*100


if HierAscClass == True:
    clr = cm.viridis(np.linspace(0.05, 0.85, N_CLUST)).tolist()
    
    #HAC-PCA based with n_components and n_clusters settings
    HAC = cluster.AgglomerativeClustering(n_clusters=N_CLUST)
    HAC_clusters = HAC.fit_predict(eigenvalue)
    
    #HAC-PCA based scatter plot
    plt.figure(figsize=(3, 3),tight_layout=True)
    for i in range(len(eigenvalue)):
        plt.scatter(eigenvalue[i][0], eigenvalue[i][1], color=clr[HAC_clusters[i]], alpha=0.5)
        plt.xlabel('PC1 ({:.2f}%)'.format(percent_explained[0])),plt.ylabel('PC2 ({:.2f}%)'.format(percent_explained[1]))
    plt.axvline(x=0.0,color='k',linestyle='--'),plt.axhline(y=0.0,color='k',linestyle='--')
    # plt.savefig('{}/Figures/HAC-PCA_ScatterPlot.pdf'.format(Path))
    
    #HAC-PCA based dendrogram
    plt.figure(figsize=(4,3))
    Z=hierarchy.linkage(eigenvalue,method='ward',metric='euclidean')
    hierarchy.set_link_color_palette(['#95D840FF','#440154FF'])
    hierarchy.dendrogram(Z,orientation='top',color_threshold=CAH_threshold)
    plt.ylabel('Inertia'),plt.xlabel('#Button')
    # plt.savefig('{}/Figures/HAC-PCA_Dendrogram.pdf'.format(Path))
    
    #STP plot after HAC
    plt.figure(figsize=(4,3),tight_layout=True)
    for i in range(N_CLUST):
        if Data == 'raw':
            plt.plot(np.linspace(1, 10, 10), np.mean([AMP[j] for j in range(len(eigenvalue)) if HAC_clusters[j] == i], axis=0), color=clr[i])
            plt.scatter(np.linspace(1, 10, 10), np.mean([AMP[j] for j in range(len(eigenvalue)) if HAC_clusters[j] == i], axis=0), color=clr[i])
            MEAN = np.mean([AMP[j] for j in range(len(eigenvalue)) if HAC_clusters[j] == i], axis=0)
            SEM = stats.sem([AMP[j] for j in range(len(eigenvalue)) if HAC_clusters[j] == i], axis=0)
        else:
            plt.plot(np.linspace(1, 10, 10), np.mean([center_normed_matrix[j]/center_normed_matrix[j][0] for j in range(len(eigenvalue)) if HAC_clusters[j] == i], axis=0), color=clr[i])
            plt.scatter(np.linspace(1, 10, 10), np.mean([center_normed_matrix[j]/center_normed_matrix[j][0] for j in range(len(eigenvalue)) if HAC_clusters[j] == i], axis=0), color=clr[i])
            MEAN = np.mean([center_normed_matrix[j]/center_normed_matrix[j][0] for j in range(len(eigenvalue)) if HAC_clusters[j] == i], axis=0)
            SEM = stats.sem([center_normed_matrix[j]/center_normed_matrix[j][0] for j in range(len(eigenvalue)) if HAC_clusters[j] == i], axis=0)
        plt.fill_between(np.linspace(1, 10, 10), MEAN+SEM, MEAN-SEM, alpha=0.3, color=clr[i])
    plt.xlabel('eEPSC#'),plt.ylabel('eEPSCn/eEPSC1')
    plt.grid(axis='y',linestyle='--')
    # plt.savefig('{}/Figures/HAC-STP_clusters_plot.pdf'.format(Path))
    
    

if OPTICS == True:
    clust = cluster.OPTICS(min_samples=OPTICS_MinPts)
    clust_fit = clust.fit_predict(eigenvalue)
    space = np.arange(len(eigenvalue))
    reachability = clust.reachability_[clust.ordering_]
    labels = clust.labels_[clust.ordering_]
    
    fig4,ax4=plt.subplots(1,2,figsize=(6,3), tight_layout=True)
    
    # Reachability plot
    colors = ['g', 'r', 'b', 'y', 'c']
    for klass, color in zip(range(N_CLUST), colors):
        Xk = space[labels == klass]
        Rk = reachability[labels == klass]
        ax4[0].scatter(Xk, Rk, color=color, s=10, alpha=0.3)
    ax4[0].scatter(space[labels == -1], reachability[labels == -1], color='k', s=10, alpha=0.5)
    ax4[0].set_ylabel('Reachability (epsilon distance)')
    ax4[0].set_title('Reachability Plot')
    # fig4.savefig('{}/Figures/Reachability_plot.pdf'.format(Path))
    
    
    # PCA-OPTICS plot
    for klass, color in zip(range(N_CLUST), colors):
        Xk = eigenvalue[clust.labels_ == klass]
        ax4[1].scatter(Xk[:, 0], Xk[:, 1], color=color, s=10, alpha=0.3)
    ax4[1].scatter(eigenvalue[clust.labels_ == -1, 0], eigenvalue[clust.labels_ == -1, 1], color='k', marker='+', s=20, alpha=0.4)
    ax4[1].set_title('Automatic Clustering\nOPTICS')
    ax4[1].axvline(x=0.0,color='k',linestyle='--')
    ax4[1].axhline(y=0.0,color='k',linestyle='--')
    # fig4.savefig('{}/Figures/OPTICS-PCA_scatterPlot.pdf'.format(Path))


    #DataFrame of objects in each cluster. Cluster -1 is noise.
    CLUST0_N,CLUST1_N,CLUST2_N,CLUST_1_N = [],[],[],[]
    for item in range(len(space)):
        if clust_fit[item] == 0:
            CLUST0_N.append(space[item])
        elif clust_fit[item] == 1:
            CLUST1_N.append(space[item])
        elif clust_fit[item] == 2:
            CLUST2_N.append(space[item])
        elif clust_fit[item] == -1:
            CLUST_1_N.append(space[item])
    if CLUST1_N != []:
        ClustDf = pd.concat((pd.DataFrame(CLUST0_N),pd.DataFrame(CLUST1_N),pd.DataFrame(CLUST_1_N)),axis=1)
        ClustDf.columns = ['Clust0_N','Clust1_N','Clust_1_N']
    elif CLUST2_N != []:
        ClustDf = pd.concat((pd.DataFrame(CLUST0_N),pd.DataFrame(CLUST1_N),
                             pd.DataFrame(CLUST2_N),pd.DataFrame(CLUST_1_N)),axis=1)
        ClustDf.columns = ['Clust0_N','Clust1_N','Clust2_N','Clust_1_N']
    else:
        ClustDf = pd.concat((pd.DataFrame(CLUST0_N),pd.DataFrame(CLUST_1_N)),axis=1)
        ClustDf.columns = ['Clust0_N','Clust_1_N']
    print(ClustDf)
    
    #STP plot after OPTICS
    plt.figure(figsize=(4,3),tight_layout=True)
    for cl in range(ClustDf.shape[1]):
        Clust_values = ClustDf.iloc[:,cl].values[np.logical_not(np.isnan(ClustDf.iloc[:,cl].values))]
        if Data == 'raw':
            MEAN = np.mean([AMP[j] for j in range(len(Clust_values))], axis=0)
            SEM = stats.sem([AMP[j] for j in range(len(Clust_values))], axis=0)
        else:
            MEAN = np.mean([center_normed_matrix[j]/center_normed_matrix[j][0] for j in range(len(Clust_values))], axis=0)
            SEM = stats.sem([center_normed_matrix[j]/center_normed_matrix[j][0] for j in range(len(Clust_values))], axis=0)
        if cl == range(ClustDf.shape[1])[-1]:
            if Data == 'raw':
                plt.plot(x, np.mean([AMP[j] for j in range(len(Clust_values))], axis=0), color='k', marker='o', alpha=0.4)
            else:
                plt.plot(x, np.mean([center_normed_matrix[j]/center_normed_matrix[j][0] for j in range(len(Clust_values))], axis=0), color='k', marker='o', alpha=0.4)
            plt.fill_between(x, MEAN+SEM, MEAN-SEM, alpha=0.3, color='k')
        else:
            if Data == 'raw':
                plt.plot(x, np.mean([AMP[j] for j in range(len(Clust_values))], axis=0), color=colors[cl], marker='o', alpha=0.4)
            else:
                plt.plot(x, np.mean([center_normed_matrix[j]/center_normed_matrix[j][0] for j in range(len(Clust_values))], axis=0), color=colors[cl], marker='o', alpha=0.4)
            plt.fill_between(x, MEAN+SEM, MEAN-SEM, alpha=0.3, color=colors[cl])
    plt.xlabel('eEPSC#'),plt.ylabel('eEPSCn/eEPSC1')
    plt.grid(axis='y',linestyle='--')
    # plt.savefig('{}/Figures/OPTICS-STP_clusters_plot.pdf'.format(Path))
                