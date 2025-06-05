# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 15:24:49 2020

@author: Theo.ROSSI
"""

import neo
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors
import numpy as np
import pandas as pd
import seaborn as sns
from statannot import add_stat_annotation
from scipy import stats
import scipy.cluster.hierarchy as hierarchy
from scipy import stats,signal
from scipy.cluster.vq import kmeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.cluster import OPTICS
from sklearn.preprocessing import normalize, StandardScaler


#-------------------------SETTINGS------------------------------------------
#---------------------------------------------------------------------------
Path = r'E:\AAVDJ.GluSnFR-S72A\AAVDJ.GluSnFR-S72A_Good_CaMg_ratio\StimML'

TYPE = 'Img' #If electrophysiology:Ephy / if Imaging:Img. Change Path according to data TYPE.
Data = 'raw'  #raw or ppr

#Set if Img TYPE only
Freq='20Hz'
Calcium = '2.5mM'
PF = '20210301_linescan1'

smoothing_value = 7

PrincCompAn = False

t_SNE = False
n_components = 3
perplexity = 5

HierAscClass = False
Kmeans = False
N_CLUST = 3

Optics=False
OPTICS_MinPts = 3

Save = False
#---------------------------------------------------------------------------
#---------------------------------------------------------------------------


AMP_TOTAL, PPR_TOTAL = [],[]
# clr = []
# cmap = cm.get_cmap('viridis', N_CLUST)
# for c in range(cmap.N):
#     rgba = cmap(c)
#     clr.append(colors.rgb2hex(rgba))



if TYPE == 'Ephy':
    AMP1 = []
    data = pd.read_excel('{}'.format(Path))
    amp = data.iloc[:,1:].values
    fig,ax = plt.subplots(1,2,figsize=(8,4),tight_layout=True)
    for profile in range(len(amp)):
        AMP_TOTAL.append(amp[profile])
        AMP1.append(amp[profile][0])
        x = np.arange(1,len(amp[profile])+1,1)
        ppr = amp[profile][:]/amp[profile][0]
        # ppr = amp[profile][:]/np.mean(amp[profile])
        PPR_TOTAL.append(ppr)
        ax[0].plot(x,amp[profile],'k',marker='o',alpha=0.2)
        ax[1].plot(x,ppr,'k',marker='o',alpha=0.2)
    df_amp = pd.DataFrame(AMP_TOTAL)
    df_ppr = pd.DataFrame(PPR_TOTAL)
    
    # plt.figure()
    # plt.hist(AMP1,bins=20,facecolor='b',edgecolor='k',alpha=0.3)
    # plt.ylabel('Count'), plt.xlabel('A1 (pA)')
    


if TYPE == 'Img':
    files = sorted(os.listdir(Path))
    
    PROFILES = []
    NumPF,NumBUTTONS = [],[]
    AVERAGE_TOTAL,TIME_TOTAL,TIME_TOTAL_LENGTH = [],[],[]
    AVERAGE,TIME=[],[]
    AMP,PPR_AVG = [],[]
    RESAMPLED_TRACES, SMOOTHED_TRACES = [],[]
    
    fig,ax=plt.subplots(2,2,figsize=(10,5), tight_layout=True)
    # fig2,ax2=plt.subplots(1,3,figsize=(6,3), tight_layout=True)
    
    for file in range(len(files)):
        if Freq in files[file]:
            if Calcium in files[file]:
                if 'ROI.tif' in files[file]:
                    NumPF.append(files[file])
                    if PF in files[file]:
                        img = plt.imread('{}/{}'.format(Path,files[file]))
                        ax[0,0].imshow(img)
                
                elif 'xlsx' in files[file]:
                    if 'dF_F0' in files[file]:
                        data = pd.read_excel('{}/{}'.format(Path,files[file]))
                        if 'Amp.xlsx' in files[file]:
                            NumBUTTONS.append(files[file])
                            amp = data.iloc[0,1:].values
                            AMP_TOTAL.append(amp)
                            PPR=[]
                            for value in range(amp.shape[0]):
                                ppr = amp[value]/amp[0]
                                PPR.append(ppr)
                            PPR_TOTAL.append(PPR)
                            x = np.arange(1,len(PPR)+1,1)
                            if PF in files[file]:
                                AMP.append(amp)
                                PPR_AVG.append(PPR)
                                ax[1,0].plot(x,amp, marker='o')
                                ax[1,1].plot(x,PPR,marker='o')
                            # ax2[1].plot(x,amp,'k',marker='o',alpha=0.2)
                            # ax2[2].plot(x,PPR,'k',marker='o',alpha=0.2)
                
                        elif 'Profile.xlsx' in files[file]:
                            PROFILES.append(files[file])
                            for col in range(len(data.columns)):
                                if 'Average'==data.columns[col]:
                                    average = data.iloc[:,col].values
                                    AVERAGE_TOTAL.append(average)
                                    if PF in files[file]:
                                        AVERAGE.append(average)
                                elif 'Time'==data.columns[col]:
                                    time = data.iloc[:,col].values
                                    TIME_TOTAL.append(time)
                                    TIME_TOTAL_LENGTH.append(len(time))
                                    if PF in files[file]:
                                        TIME.append(time)
                            if PF in files[file]:        
                                ax[0,1].plot(time,signal.savgol_filter(average, smoothing_value, 2),alpha=0.5,label='{}'.format(files[file]))
                    
    
    df_amp = pd.DataFrame(AMP_TOTAL)
    df_ppr = pd.DataFrame(PPR_TOTAL)
    # with pd.ExcelWriter('{}/Figures/Amp_values_{}_10pulses_{}Ca.xlsx'.format(Path,Freq,Path.split('\\')[-1])) as writer:
    #     df_amp.to_excel(writer)
    # with pd.ExcelWriter('{}/Figures/PPR_values_{}_10pulses_{}Ca.xlsx'.format(Path,Freq,Path.split('\\')[-1])) as writer:
    #     df_ppr.to_excel(writer)
    
    print('Number of Parallel fibers: {}'.format(len(NumPF)))
    print('Number of buttons: {}'.format(len(NumBUTTONS)))
    # avg_traces = np.average(AVERAGE,axis=0)
    # avg_amp = np.average(AMP,axis=0)
    # avg_ppr = np.average(PPR_AVG,axis=0)
    # ax[0,1].plot(TIME[0],avg_traces,'brown',alpha=0.6)
    # ax[1,0].plot(x,avg_amp,'brown',marker='o',linewidth=3)
    # ax[1,1].plot(x,avg_ppr,'brown',marker='o',linewidth=3)
    # fig.legend(fontsize=6,loc='best')
    # fig2.legend(labels=['N={}'.format(len(AMP_TOTAL))])
    
    # if Save == True:
    #     fig.savefig('{}/Figures/{}.pdf'.format(Path,PF))
    #     fig2.savefig('{}/Figures/Plot_all_Profiles.pdf'.format(Path))
    
    # #Plot of all traces
    # plt.figure()
    # for trace in range(len(PROFILES)):
    #     if '20190801' in PROFILES[trace]:
    #         start = np.ravel(np.where(TIME_TOTAL[trace] <= 0.95))[-1]
    #         end = np.ravel(np.where(TIME_TOTAL[trace] <= 1.5))[-1]
    #     else:
    #         start = np.ravel(np.where(TIME_TOTAL[trace] <= 0.45))[-1]
    #         end = np.ravel(np.where(TIME_TOTAL[trace] <= 1.0))[-1]
    #     NewTrace = AVERAGE_TOTAL[trace][start:end]
    #     resampling = signal.resample(NewTrace, np.nanmin(TIME_TOTAL_LENGTH))
    #     RESAMPLED_TRACES.append(resampling)
    #     smoothing = signal.savgol_filter(resampling,7,2)
    #     SMOOTHED_TRACES.append(smoothing)
    #     plt.plot(resampling,alpha=0.4)
    #     for item in range(len(TIME_TOTAL_LENGTH)):
    #         if len(TIME_TOTAL[item]) == np.nanmin(TIME_TOTAL_LENGTH):
    #             start_average = np.ravel(np.where(TIME_TOTAL[item] <= 0.45))[-1]
    #             end_average = np.ravel(np.where(TIME_TOTAL[item] <= 1.0))[-1]
    #             x_average = np.linspace(TIME_TOTAL[item][start_average], TIME_TOTAL[item][end_average], num=len(TIME_TOTAL[item]), endpoint=True)
    # ResDf = pd.DataFrame(SMOOTHED_TRACES)
    # ax2[0].imshow(ResDf,extent=[0, 1, 0, 2])    #Colored image of all traces
    
    
    # #Plot PPR2/1 vs. Amp1
    # fig_amp_ppr, ax_amp_ppr = plt.subplots(1,2,tight_layout=True)
    # amplitudes1 = df_amp.iloc[:,0].values
    # amplitudes2 = df_amp.iloc[:,1].values
    # paired_pulse_ratio = df_ppr.iloc[:,1].values
    # ax_amp_ppr[0].scatter(amplitudes1,amplitudes2)
    # ax_amp_ppr[0].set_ylabel('Peak2')
    # ax_amp_ppr[0].set_xlabel('Peak1')
    # ax_amp_ppr[1].scatter(amplitudes1,paired_pulse_ratio)
    # ax_amp_ppr[1].set_ylabel('PPR2/1')
    # ax_amp_ppr[1].set_xlabel('Peak1')
    
    
    
if PrincCompAn == True:
    if Data == 'raw':
        center_norm_matrix = (df_amp - df_amp.mean(axis=0))/df_amp.std(axis=0)
    else:
        center_norm_matrix = (df_ppr - df_ppr.mean(axis=0))/df_ppr.std(axis=0)
    IMPUTER = Imputer(strategy='median').fit_transform(center_norm_matrix)
    
    pca = PCA()
    pca_fit = pca.fit(IMPUTER)
    VarComp = pd.DataFrame(pca_fit.components_[:,:2])
    ExpVar = pca_fit.explained_variance_ratio_
    cumsum = np.cumsum(ExpVar)
    d = np.argmax(cumsum >= 0.95)+1
    
    PCA = PCA(n_components=d)
    eigenvalue = PCA.fit_transform(IMPUTER)
        
    #PCA settings figure
    fig_PCA,ax_PCA=plt.subplots(1,5,figsize=(17,4),tight_layout=True)
    ax_PCA[0].plot(np.arange(1,len(IMPUTER[0])+1,1),cumsum,marker='o', linestyle='--', color='b')
    ax_PCA[0].set_xlabel('#PC')
    ax_PCA[0].set_ylabel('Cumulative variance (%)')
    ax_PCA[0].set_xticks(x)
    ax_PCA[0].set_title('Number of components\nneeded to explain variance')
    ax_PCA[0].axhline(y=0.95, color='r', linestyle='--')
    ax_PCA[0].text(0.5, 0.96, '95% cut-off threshold', color = 'red', fontsize=10)
    ax_PCA[0].grid(axis='x')
    ax_PCA[1].bar(np.arange(len(ExpVar)) + 0.5, ExpVar)
    ax_PCA[1].set_title("Explained variance")
    ax_PCA[1].set_ylabel("Norm variance")
    ax_PCA[1].set_xlabel("#Component")
    
    for eigen in range(len(eigenvalue)):    
        ax_PCA[2].scatter(eigenvalue[eigen][0],eigenvalue[eigen][1],color='k',alpha=0.4)
    ax_PCA[2].set_title('PCA on {} data'.format(Data))
    ax_PCA[2].set_xlabel('PC1 ({:.2f}%)'.format(ExpVar[0]*100))
    ax_PCA[2].set_ylabel('PC2 ({:.2f}%)'.format(ExpVar[1]*100))
    ax_PCA[2].axvline(x=0.0,color='k',linestyle='--')
    ax_PCA[2].axhline(y=0.0,color='k',linestyle='--')
    
    ax_PCA[3].bar(np.arange(1,len(VarComp)+1,1),VarComp[0])
    ax_PCA[3].set_title('PC1\n{} data'.format(Data))
    ax_PCA[3].set_xlabel('#PEAK'.format(ExpVar[0]*100))
    ax_PCA[3].set_ylabel('PEAK contribution'.format(ExpVar[1]*100))
    ax_PCA[3].axhline(y=0.0, color='k', linestyle='--')
    ax_PCA[4].bar(np.arange(1,len(VarComp)+1,1),VarComp[1])
    ax_PCA[4].set_title('PC2\n{} data'.format(Data))
    ax_PCA[4].set_xlabel('#PEAK'.format(ExpVar[0]*100))
    ax_PCA[4].set_ylabel('PEAK contribution'.format(ExpVar[1]*100))
    ax_PCA[4].axhline(y=0.0, color='k', linestyle='--')
    



if t_SNE == True:
    if Data == 'raw':
        IMPUTER = Imputer(strategy='median').fit_transform(df_amp)
    else:
        IMPUTER = Imputer(strategy='median').fit_transform(df_ppr)
    tsne = TSNE(n_components=n_components, perplexity=perplexity).fit_transform(df_amp)
    fig = plt.figure()
    if n_components == 2:
        ax = fig.add_subplot(111)
        ax.scatter(tsne[:,0],tsne[:,1])
    if n_components == 3:
        ax = fig.add_subplot(111,projection='3d')
        ax.scatter(tsne[:,0],tsne[:,1],tsne[:,2])
        ax.set_zlabel('Dim3')
    ax.set_xlabel('Dim1')
    ax.set_ylabel('Dim2')
    
    
    
     
if HierAscClass == True:
    
    #HAC-PCA based with n_components and n_clusters settings
    HCPC = AgglomerativeClustering(n_clusters=N_CLUST)
    HCPC_clusters = HCPC.fit_predict(eigenvalue)
    
    #HAC-PCA based scatter plot
    fig_HCPC,ax_HCPC = plt.subplots(1,4,figsize=(18,4),tight_layout=True)
    for i in range(len(eigenvalue)):
        ax_HCPC[0].scatter(eigenvalue[i][0], eigenvalue[i][1], color=clr[HCPC_clusters[i]], alpha=0.5)
        ax_HCPC[0].set_xlabel('PC1 ({:.2f}%)'.format(ExpVar[0]*100))
        ax_HCPC[0].set_ylabel('PC2 ({:.2f}%)'.format(ExpVar[1]*100))
    ax_HCPC[0].set_title('HCPC after PCA on {} data'.format(Data))
    ax_HCPC[0].axvline(x=0.0,color='k',linestyle='--')
    ax_HCPC[0].axhline(y=0.0,color='k',linestyle='--')
    
    #HAC-PCA based dendrogram
    Z=hierarchy.linkage(eigenvalue,method='ward',metric='euclidean',optimal_ordering=True)
    Inertia = sorted(Z[:,2],reverse=True)
    hierarchy.set_link_color_palette(clr)
    hierarchy.dendrogram(Z,orientation='top',ax=ax_HCPC[1],color_threshold=[(Inertia[i]-0.1) for i in range(N_CLUST-1)][-1])
    ax_HCPC[1].axhline(y=[(Inertia[i]-0.1) for i in range(N_CLUST-1)][-1],color='r',linestyle='--')
    ax_HCPC[1].set_title('HCPC dendrogram')
    ax_HCPC[1].set_xlabel('#Button')
    ax_HCPC[1].set_ylabel('Inertia gain')
    # ax_HCPC[2].bar(np.arange(len(Inertia)),Inertia,color='k',alpha=0.5)
    # ax_HCPC[2].set_title('Inertia gain')
    # ax_HCPC[2].set_xlabel('#Cluster')
    # ax_HCPC[2].axhline([(Inertia[i]-0.1) for i in range(N_CLUST-1)][-1],color='r',linestyle='--')
    
    #STP plot after HAC
    AMP1, PPR2_1 = [],[]
    for i in range(N_CLUST):
        if Data == 'raw':
            ax_HCPC[2].plot(np.linspace(1, df_amp.shape[1], df_amp.shape[1]), np.mean([df_amp.transpose()[j] for j in range(len(eigenvalue)) if HCPC_clusters[j] == i], axis=0), marker='o', color=clr[i])
            cluster = [df_amp.transpose()[j] for j in range(len(eigenvalue)) if HCPC_clusters[j] == i]
            df_clust = pd.DataFrame(cluster)
            AMP1.append(df_clust.iloc[:,0])
        else:
            ax_HCPC[2].plot(np.linspace(1, df_ppr.shape[1], df_ppr.shape[1]), np.mean([df_ppr.transpose()[j] for j in range(len(eigenvalue)) if HCPC_clusters[j] == i], axis=0), marker='o', color=clr[i])
            cluster = [df_ppr.transpose()[j] for j in range(len(eigenvalue)) if HCPC_clusters[j] == i] 
            df_clust = pd.DataFrame(cluster)
            PPR2_1.append(df_clust.iloc[:,1])
        MEAN = np.mean(cluster, axis=0)
        SEM = stats.sem(cluster, axis=0)
        
        if Data == 'raw':
            ax_HCPC[2].fill_between(np.linspace(1, df_amp.shape[1], df_amp.shape[1]), MEAN+SEM, MEAN-SEM, alpha=0.3, color=clr[i])
        else:
            ax_HCPC[2].fill_between(np.linspace(1, df_ppr.shape[1], df_ppr.shape[1]), MEAN+SEM, MEAN-SEM, alpha=0.3, color=clr[i])
    ax_HCPC[2].set_title('STP {} clusters'.format(Data))
    ax_HCPC[2].set_xlabel('#PEAK')
    ax_HCPC[2].set_ylabel('PEAKn/PEAK1')
    ax_HCPC[2].grid(axis='y',linestyle='--')
    
    if Data == 'raw':
        ClustDf = pd.concat(([AMP1[i] for i in range(N_CLUST)]),axis=1)
    else:
        ClustDf = pd.concat(([PPR2_1[i] for i in range(N_CLUST)]),axis=1)
        
    ClustDf.columns=['Cluster'+str(i) for i in range(N_CLUST)]
    sns.boxplot(data=ClustDf, showmeans=True, ax=ax_HCPC[3], palette = [clr[i] for i in range(N_CLUST)], meanprops={'markerfacecolor':'red','markeredgecolor':'black','markersize':'8'})


    if len(ClustDf.columns) == 2:
        add_stat_annotation(ax_HCPC[3], data=ClustDf, box_pairs=[(ClustDf.columns[0],ClustDf.columns[-1])], test='Mann-Whitney', text_format='star', verbose=2)
    if len(ClustDf.columns) == 3:
        add_stat_annotation(ax_HCPC[3], data=ClustDf, box_pairs=[(ClustDf.columns[0],ClustDf.columns[1]),
                                                                 (ClustDf.columns[0],ClustDf.columns[-1]),
                                                                 (ClustDf.columns[1],ClustDf.columns[-1])],
                            test='Mann-Whitney', text_format='star', verbose=2)
    if len(ClustDf.columns) == 4:
        add_stat_annotation(ax_HCPC[3], data=ClustDf, box_pairs=[(ClustDf.columns[0],ClustDf.columns[1]),
                                                                 (ClustDf.columns[0],ClustDf.columns[2]),
                                                                 (ClustDf.columns[0],ClustDf.columns[-1]),
                                                                 (ClustDf.columns[1],ClustDf.columns[2]),
                                                                 (ClustDf.columns[2],ClustDf.columns[-1])],
                            test='Mann-Whitney', text_format='star', verbose=2)
    if len(ClustDf.columns) == 5:
        add_stat_annotation(ax_HCPC[3], data=ClustDf, box_pairs=[(ClustDf.columns[0],ClustDf.columns[1]),
                                                                 (ClustDf.columns[0],ClustDf.columns[2]),
                                                                 (ClustDf.columns[0],ClustDf.columns[3]),
                                                                 (ClustDf.columns[0],ClustDf.columns[-1]),
                                                                 (ClustDf.columns[1],ClustDf.columns[2]),
                                                                 (ClustDf.columns[1],ClustDf.columns[3]),
                                                                 (ClustDf.columns[1],ClustDf.columns[-1]),
                                                                 (ClustDf.columns[2],ClustDf.columns[3]),
                                                                 (ClustDf.columns[2],ClustDf.columns[-1]),
                                                                 (ClustDf.columns[3],ClustDf.columns[-1])],
                            test='Mann-Whitney', text_format='star', verbose=2)
    
    if Save == True:            
        plt.savefig('{}/Figures/HAC-STP_clusters_plot.pdf'.format(Path))
        
    
    
    
if Kmeans == True:
    KM = KMeans(n_clusters=N_CLUST)
    KM_clusters = KM.fit_predict(eigenvalue)
  
    #HAC-PCA based scatter plot
    fig_KMeans,ax_KMeans = plt.subplots(1,3,figsize=(15,4),tight_layout=True)
    for i in range(len(eigenvalue)):
        ax_KMeans[0].scatter(eigenvalue[i][0], eigenvalue[i][1], color=clr[KM_clusters[i]], alpha=0.5)
        ax_KMeans[0].set_xlabel('PC1 ({:.2f}%)'.format(ExpVar[0]*100))
        ax_KMeans[0].set_ylabel('PC2 ({:.2f}%)'.format(ExpVar[1]*100))
    ax_KMeans[0].set_title('KMean after PCA on {} data'.format(Data))
    ax_KMeans[0].axvline(x=0.0,color='k',linestyle='--')
    ax_KMeans[0].axhline(y=0.0,color='k',linestyle='--')
    
    
    #STP plot after HAC
    AMP1, PPR2_1 = [],[]
    for i in range(N_CLUST):
        if Data == 'raw':
            ax_KMeans[1].plot(np.linspace(1, df_amp.shape[1], df_amp.shape[1]), np.mean([df_amp.transpose()[j] for j in range(len(eigenvalue)) if KM_clusters[j] == i], axis=0), marker='o', color=clr[i])
            cluster = [df_amp.transpose()[j] for j in range(len(eigenvalue)) if KM_clusters[j] == i]
            df_clust = pd.DataFrame(cluster)
            AMP1.append(df_clust.iloc[:,0])
        else:
            ax_KMeans[1].plot(np.linspace(1, df_ppr.shape[1], df_ppr.shape[1]), np.mean([df_ppr.transpose()[j] for j in range(len(eigenvalue)) if KM_clusters[j] == i], axis=0), marker='o', color=clr[i])
            cluster = [df_ppr.transpose()[j] for j in range(len(eigenvalue)) if KM_clusters[j] == i] 
            df_clust = pd.DataFrame(cluster)
            PPR2_1.append(df_clust.iloc[:,1])
        MEAN = np.mean(cluster, axis=0)
        SEM = stats.sem(cluster, axis=0)
        
        if Data == 'raw':
            ax_KMeans[1].fill_between(np.linspace(1, df_amp.shape[1], df_amp.shape[1]), MEAN+SEM, MEAN-SEM, alpha=0.3, color=clr[i])
        else:
            ax_KMeans[1].fill_between(np.linspace(1, df_ppr.shape[1], df_ppr.shape[1]), MEAN+SEM, MEAN-SEM, alpha=0.3, color=clr[i])
    ax_KMeans[1].set_title('STP {} clusters'.format(Data))
    ax_KMeans[1].set_xlabel('#PEAK')
    ax_KMeans[1].set_ylabel('PEAKn/PEAK1')
    ax_KMeans[1].grid(axis='y',linestyle='--')
    
    if Data == 'raw':
        ClustDf = pd.concat(([AMP1[i] for i in range(N_CLUST)]),axis=1)
    else:
        ClustDf = pd.concat(([PPR2_1[i] for i in range(N_CLUST)]),axis=1)
        
    ClustDf.columns=['Cluster'+str(i) for i in range(N_CLUST)]
    sns.boxplot(data=ClustDf, showmeans=True, ax=ax_KMeans[2], palette = [clr[i] for i in range(N_CLUST)], meanprops={'markerfacecolor':'red','markeredgecolor':'black','markersize':'8'})


    if len(ClustDf.columns) == 2:
        add_stat_annotation(ax_KMeans[2], data=ClustDf, box_pairs=[(ClustDf.columns[0],ClustDf.columns[-1])], test='Mann-Whitney', text_format='star', verbose=2)
    if len(ClustDf.columns) == 3:
        add_stat_annotation(ax_KMeans[2], data=ClustDf, box_pairs=[(ClustDf.columns[0],ClustDf.columns[1]),
                                                                 (ClustDf.columns[0],ClustDf.columns[-1]),
                                                                 (ClustDf.columns[1],ClustDf.columns[-1])],
                            test='Mann-Whitney', text_format='star', verbose=2)
    if len(ClustDf.columns) == 4:
        add_stat_annotation(ax_KMeans[2], data=ClustDf, box_pairs=[(ClustDf.columns[0],ClustDf.columns[1]),
                                                                 (ClustDf.columns[0],ClustDf.columns[2]),
                                                                 (ClustDf.columns[0],ClustDf.columns[-1]),
                                                                 (ClustDf.columns[1],ClustDf.columns[2]),
                                                                 (ClustDf.columns[2],ClustDf.columns[-1])],
                            test='Mann-Whitney', text_format='star', verbose=2)
    if len(ClustDf.columns) == 5:
        add_stat_annotation(ax_KMeans[2], data=ClustDf, box_pairs=[(ClustDf.columns[0],ClustDf.columns[1]),
                                                                 (ClustDf.columns[0],ClustDf.columns[2]),
                                                                 (ClustDf.columns[0],ClustDf.columns[3]),
                                                                 (ClustDf.columns[0],ClustDf.columns[-1]),
                                                                 (ClustDf.columns[1],ClustDf.columns[2]),
                                                                 (ClustDf.columns[1],ClustDf.columns[3]),
                                                                 (ClustDf.columns[1],ClustDf.columns[-1]),
                                                                 (ClustDf.columns[2],ClustDf.columns[3]),
                                                                 (ClustDf.columns[2],ClustDf.columns[-1]),
                                                                 (ClustDf.columns[3],ClustDf.columns[-1])],
                            test='Mann-Whitney', text_format='star', verbose=2)
    

        
    
if Optics == True:
    clust = OPTICS(min_samples=OPTICS_MinPts)
    clust.fit_predict(eigenvalue)
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
    fig4.savefig('{}/Figures/Reachability_plot.pdf'.format(Path))
    
    
    # PCA-OPTICS plot
    for klass, color in zip(range(N_CLUST), colors):
        Xk = eigenvalue[clust.labels_ == klass]
        ax4[1].scatter(Xk[:, 0], Xk[:, 1], color=color, s=10, alpha=0.3)
    ax4[1].scatter(eigenvalue[clust.labels_ == -1, 0], eigenvalue[clust.labels_ == -1, 1], color='k', marker='+', s=20, alpha=0.4)
    ax4[1].set_title('Automatic Clustering\nOPTICS')
    ax4[1].axvline(x=0.0,color='k',linestyle='--')
    ax4[1].axhline(y=0.0,color='k',linestyle='--')
    # fig4.savefig('{}/Figures/OPTICS-PCA_scatterPlot.pdf'.format(Path))


    # #DataFrame of objects in each cluster. Cluster -1 is noise.
    # CLUST0_N,CLUST1_N,CLUST2_N,CLUST_1_N = [],[],[],[]
    # for item in range(len(space)):
    #     if clust_fit[item] == 0:
    #         CLUST0_N.append(space[item])
    #     elif clust_fit[item] == 1:
    #         CLUST1_N.append(space[item])
    #     elif clust_fit[item] == 2:
    #         CLUST2_N.append(space[item])
    #     elif clust_fit[item] == -1:
    #         CLUST_1_N.append(space[item])
    # if CLUST1_N != []:
    #     ClustDf = pd.concat((pd.DataFrame(CLUST0_N),pd.DataFrame(CLUST1_N),pd.DataFrame(CLUST_1_N)),axis=1)
    #     ClustDf.columns = ['Clust0_N','Clust1_N','Clust_1_N']
    # elif CLUST2_N != []:
    #     ClustDf = pd.concat((pd.DataFrame(CLUST0_N),pd.DataFrame(CLUST1_N),
    #                          pd.DataFrame(CLUST2_N),pd.DataFrame(CLUST_1_N)),axis=1)
    #     ClustDf.columns = ['Clust0_N','Clust1_N','Clust2_N','Clust_1_N']
    # else:
    #     ClustDf = pd.concat((pd.DataFrame(CLUST0_N),pd.DataFrame(CLUST_1_N)),axis=1)
    #     ClustDf.columns = ['Clust0_N','Clust_1_N']
    # print(ClustDf)
    
    # #STP plot after OPTICS
    # plt.figure(figsize=(4,3),tight_layout=True)
    # for cl in range(ClustDf.shape[1]):
    #     Clust_values = ClustDf.iloc[:,cl].values[np.logical_not(np.isnan(ClustDf.iloc[:,cl].values))]
    #     if Data == 'raw':
    #         MEAN = np.mean([df_amp[j] for j in range(len(Clust_values))], axis=0)
    #         SEM = stats.sem([df_amp[j] for j in range(len(Clust_values))], axis=0)
    #     else:
    #         MEAN = np.mean([center_normed_matrix[j]/center_normed_matrix[j][0] for j in range(len(Clust_values))], axis=0)
    #         SEM = stats.sem([center_normed_matrix[j]/center_normed_matrix[j][0] for j in range(len(Clust_values))], axis=0)
    #     if cl == range(ClustDf.shape[1])[-1]:
    #         if Data == 'raw':
    #             plt.plot(x, np.mean([df_amp[j] for j in range(len(Clust_values))], axis=0), color='k', marker='o', alpha=0.4)
    #         else:
    #             plt.plot(x, np.mean([center_normed_matrix[j]/center_normed_matrix[j][0] for j in range(len(Clust_values))], axis=0), color='k', marker='o', alpha=0.4)
    #         plt.fill_between(x, MEAN+SEM, MEAN-SEM, alpha=0.3, color='k')
    #     else:
    #         if Data == 'raw':
    #             plt.plot(x, np.mean([df_amp[j] for j in range(len(Clust_values))], axis=0), color=colors[cl], marker='o', alpha=0.4)
    #         else:
    #             plt.plot(x, np.mean([center_normed_matrix[j]/center_normed_matrix[j][0] for j in range(len(Clust_values))], axis=0), color=colors[cl], marker='o', alpha=0.4)
    #         plt.fill_between(x, MEAN+SEM, MEAN-SEM, alpha=0.3, color=colors[cl])
    # plt.xlabel('eEPSC#'),plt.ylabel('eEPSCn/eEPSC1')
    # plt.grid(axis='y',linestyle='--')
    # plt.savefig('{}/Figures/OPTICS-STP_clusters_plot.pdf'.format(Path))