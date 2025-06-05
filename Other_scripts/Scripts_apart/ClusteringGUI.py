# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 12:23:31 2021

@author: Theo.ROSSI
"""


def viridis_colors(num):
    
    '''
    Creates a list of color values converted in color names
    
    num (int): number of returned colors
    '''
    
    clr = []
    num_colors = num
    cmap = cm.viridis(np.linspace(0.1,0.85,num_colors))
    for c in range(num_colors):
        rgba = cmap[c]
        clr.append(colors.rgb2hex(rgba))
    return clr



def load_ephy_xlsx(file, sheet):
    
    '''
    Load a single excel file and returns a dataframe from the excel file.
        /!\ COLUMN 0 ARE INDEXES
    
    file (str): file path
    '''
    
    data = pd.read_excel('{}'.format(file), sheet_name=sheet, header=0)
    amp = data.iloc[:].values
    df_amp = pd.DataFrame(amp)
    return df_amp



def load_img_xlsx(Folder,Freq,Calcium,PF):
    
    '''
    Load several excel files from imaging experiments and returns a dataframe from all the excel files
    
    Folder (str): folder path
    Freq (str): frequency of stim
    Calcium (str): calcium concentration
    PF (str): name of the parallel fiber to visualize
    '''
    
    files = sorted(os.listdir(Folder))
    
    AVERAGE_TOTAL,TIME_TOTAL,TIME_TOTAL_LENGTH = [],[],[]
    AVERAGE,TIME=[],[]
    AMP_TOTAL = []
    
    fig,ax=plt.subplots(2,2,figsize=(10,5), tight_layout=True)
    for file in range(len(files)):
        if Freq in files[file]:
            if Calcium in files[file]:
                if 'ROI.tif' in files[file]:
                    print(files[file])
                    if PF in files[file]:
                        img = plt.imread('{}/{}'.format(Folder,files[file]))
                        ax[0,0].imshow(img)
                
                elif 'xlsx' in files[file]:
                    if 'dF_F0' in files[file]:
                        data = pd.read_excel('{}/{}'.format(Folder,files[file]))
                        if 'Amp.xlsx' in files[file]:
                            amp = data.iloc[0,1:].values
                            AMP_TOTAL.append(amp)
                            ppr = [amp[i]/amp[0] for i in range(amp.shape[0])]
                            if PF in files[file]:
                                ax[1,0].plot(amp,marker='o')
                                ax[1,1].plot(ppr,marker='o')
                            
                        elif 'Profile.xlsx' in files[file]:
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
                                # ax[0,1].plot(time,average,alpha=0.5,label='{}'.format(files[file]))
                                ax[0,1].plot(time,savgol_filter(average,5,2),alpha=0.5,label='{}'.format(files[file]))
                                x1 = np.ravel(np.where(time <= 0.5))[-1]
                                x2 = np.ravel(np.where(time <= 0.55))[-1]
                                x = time[x1:x2]
                                a1 = average[x1:x2]
                                
                                b,a = signal.butter(3,0.4)
                                filtered = signal.filtfilt(b,a,average)
                                ax[0,1].set_ylim(-0.5,2.0)
                                
                                # ax[0,1].plot(time,filtered,alpha=0.5)
                                
                                
                                
    df_amp = pd.DataFrame(AMP_TOTAL)
    return df_amp



def PrinCompAn(dataframe,Reduction=True,graph=True):
    
    '''
    Computes a Pincipal Component Analysis on a dataframe and returns the explained variance
    of each principal component (PC) and the eigenvalues of each individual.
        
    dataframe: dataset after using pandas.DataFrame.
        /!\ INDIVIDUALS AS ROWS AND VARIABLES AS COLUMNS
    
    Reduction (bool): if True, each value in a row is divided by the standard deviation of the whole row.
        /!\ USE IT WHEN DIFFERENT UNITS BETWEEN VARIABLES
    
    graph (bool): if True, creates a figure with:
        - elbow plot (ax1): gives the number of PCs explaining at least 95% of the total variance.
        - bar plot (ax2): explained variance by each PC.
        - scatter plot (ax3): position of each individual in the new dimentional subspace.
        - bar plot (ax4 and ax5): contribution of each variable on the first (ax4) and second (ax5) PCs.
    '''
    
    variables = dataframe.iloc[:,4:]
    
    if Reduction == True:
        center_norm_matrix = (variables-variables.mean(axis=0))/variables.std(axis=0)
    else:
        center_norm_matrix = (variables-variables.mean(axis=0))

    IMPUTER = SimpleImputer(strategy='median').fit_transform(center_norm_matrix)
    
    pca = PCA()
    pca_fit = pca.fit(IMPUTER)
    VarComp = pd.DataFrame(pca_fit.components_[:,:2])
    ExpVar = pca_fit.explained_variance_ratio_
    cumsum = np.cumsum(ExpVar)
    d = np.argmax(cumsum >= 0.95)+1
    
    PrCoAn = PCA(n_components=d)
    eigenvalue = PrCoAn.fit_transform(IMPUTER)
    
    if graph == True:
        x = np.arange(1,variables.shape[1]+1,1)
        fig_PCA, ax_PCA=plt.subplots(1,5,figsize=(17,4),tight_layout=True)
        ax_PCA[0].plot(np.arange(1,len(IMPUTER[0])+1,1),cumsum,marker='o', linestyle='--', color='b')
        ax_PCA[0].set_xlabel('Number of Principal Components')
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
        
        # fig = plt.figure()
        # ax_pca3d = plt.axes(projection='3d')
        # for eigen in range(len(eigenvalue)):    
        #     ax_pca3d.scatter3D(eigenvalue[eigen][0],eigenvalue[eigen][1],eigenvalue[eigen][2],color='k',alpha=0.4)
        # ax_pca3d.set_title('PCA')
        # ax_pca3d.set_xlabel('PC1 ({:.2f}%)'.format(ExpVar[0]*100))
        # ax_pca3d.set_ylabel('PC2 ({:.2f}%)'.format(ExpVar[1]*100))
        # ax_pca3d.set_zlabel('PC3 ({:.2f}%)'.format(ExpVar[2]*100))

        for eigen in range(len(eigenvalue)):    
            ax_PCA[2].scatter(eigenvalue[eigen][0],eigenvalue[eigen][1],color='k',alpha=0.4)
        ax_PCA[2].set_title('PCA')
        ax_PCA[2].set_xlabel('PC1 ({:.2f}%)'.format(ExpVar[0]*100))
        ax_PCA[2].set_ylabel('PC2 ({:.2f}%)'.format(ExpVar[1]*100))
        ax_PCA[2].axvline(x=0.0,color='k',linestyle='--')
        ax_PCA[2].axhline(y=0.0,color='k',linestyle='--')
         
        ax_PCA[3].set_title('PC1 contributions')
        ax_PCA[3].bar(np.arange(1,len(VarComp)+1,1),VarComp[0])
        ax_PCA[3].set_xlabel('#Variable'.format(ExpVar[0]*100))
        ax_PCA[3].set_ylabel('Variable contribution'.format(ExpVar[1]*100))
        ax_PCA[3].axhline(y=0.0, color='k', linestyle='--')
        
        ax_PCA[4].set_title('PC2 contributions')
        ax_PCA[4].bar(np.arange(1,len(VarComp)+1,1),VarComp[1])
        ax_PCA[4].set_xlabel('#Variable'.format(ExpVar[0]*100))
        ax_PCA[4].set_ylabel('Variable contribution'.format(ExpVar[1]*100))
        ax_PCA[4].axhline(y=0.0, color='k', linestyle='--')
    return ExpVar,eigenvalue



def HierAscClass(dataframe,n_clust,pca_=True):
    
    '''
    Computes an Ascending Hierarchical Classification (AHC) using the Ward method on a dataset
    and returns the cluster's number each individual belongs to.
    
    dataframe: dataset after using pandas.DataFrame.
        /!\ INDIVIDUALS AS ROWS AND VARIABLES AS COLUMNS
        
    n_clust (int): number of arbitrary clusters.
    
    pca_ (bool): if True, computes the AHC on the dataset after PCA.
                 if False, computes the AHC on the raw dataset.
    '''
    
    indexes = dataframe.iloc[:,1]
    variables = dataframe.iloc[:,4:]
    
    colors = viridis_colors(n_clust)
    ExpVar,eigenvalue = PrinCompAn(variables)
    plt.close()
    HCPC = AgglomerativeClustering(n_clusters=n_clust)
    
    if pca_ == True:
        fig_clust,ax_clust = plt.subplots(1,2,figsize=(10,4),tight_layout=True)
        HCPC_clusters = HCPC.fit_predict(eigenvalue)
        Z=hierarchy.linkage(eigenvalue,method='ward',metric='euclidean',optimal_ordering=True)
    
        for i in range(len(HCPC_clusters)):
            ax_clust[1].scatter(eigenvalue[i][0], eigenvalue[i][1], color=colors[HCPC_clusters[i]], alpha=0.5)
        for j,txt in enumerate(indexes):
            ax_clust[1].annotate(j,(eigenvalue[j][0], eigenvalue[j][1]))
        ax_clust[1].set_xlabel('PC1 ({:.2f}%)'.format(ExpVar[0]*100))
        ax_clust[1].set_ylabel('PC2 ({:.2f}%)'.format(ExpVar[1]*100))
        ax_clust[1].set_title('HCPC after PCA')
        ax_clust[1].axvline(x=0.0,color='k',linestyle='--')
        ax_clust[1].axhline(y=0.0,color='k',linestyle='--')
        
    else:
        fig_clust,ax_clust = plt.subplots(1,2,figsize=(12,4),tight_layout=True)
        HCPC_clusters = HCPC.fit_predict(variables)
        Z=hierarchy.linkage(variables,method='ward',metric='euclidean',optimal_ordering=True)
    
    Inertia = sorted(Z[:,2],reverse=True)
    for i in range(len(HCPC_clusters)):
        hierarchy.set_link_color_palette(colors)
        hierarchy.dendrogram(Z,orientation='top',ax=ax_clust[0],color_threshold=[(Inertia[i]-0.1) for i in range(len(colors)-1)][-1])
        ax_clust[0].axhline(y=[(Inertia[i]-0.1) for i in range(len(colors)-1)][-1],color='r',linestyle='--')
    ax_clust[0].set_title('HCPC dendrogram')
    ax_clust[0].set_xlabel('#Button')
    ax_clust[0].set_ylabel('Inertia gain')
    
    
    # df = pd.concat((pd.DataFrame(indexes),pd.DataFrame(HCPC_clusters)),axis=1)
    # df.columns = ['Individuals','Cluster']
    # print('HCPC - clusters: {}'.format(n_clust))
    # print(df)
    
    return HCPC_clusters,df
    
               
        
def KMean(dataframe,n_clust,pca_=True):
    
    '''
    Computes a Kmeans on a dataset and returns the cluster's number each individual belongs to.
    
    dataframe: dataset after using pandas.DataFrame.
        /!\ INDIVIDUALS AS ROWS AND VARIABLES AS COLUMNS
        
    n_clust (int): number of arbitrary clusters.
    
    pca_ (bool): if True, computes the Kmeans on the dataset after PCA.
                 if False, computes the Kmeans on the raw dataset.
    '''
    
    indexes = dataframe.iloc[:,0]
    variables = dataframe.iloc[:,1:]
    
    colors = viridis_colors(n_clust)
    ExpVar,eigenvalue = PrinCompAn(variables)
    KM = KMeans(n_clusters=n_clust)
    plt.close()
    
    if pca_ == True:
        KM_clusters = KM.fit_predict(eigenvalue)
        fig_clust,ax_clust = plt.subplots(1,2,figsize=(10,4),tight_layout=True)
        for i in range(len(eigenvalue)):
            ax_clust[1].scatter(eigenvalue[i][0], eigenvalue[i][1], color=colors[KM_clusters[i]], alpha=0.5)
    else:
        fig_clust,ax_clust = plt.subplots(1,2)
        KM_clusters = KM.fit_predict(variables)
    
    K_MAX = int(8)
    KK = range(1,K_MAX+1)
    
    KM_ = [kmeans(eigenvalue, k) for k in KK]
    centroids = [cent for (cent, var) in KM_]
    D_k = [cdist(eigenvalue, cent, 'euclidean') for cent in centroids]
    dist = [np.min(eigenvalue, axis=1) for eigenvalue in D_k]

    tot_withinss = [sum(d**2) for d in dist]  # Total within-cluster sum of squares
    totss = sum(pdist(eigenvalue)**2)/eigenvalue.shape[0]       # The total sum of squares
    betweenss = totss - tot_withinss          # The between-cluster sum of squares

    #Elbow plot for KMeans
    ax_clust[0].plot(KK, betweenss/totss*100,marker='o')
    ax_clust[0].set_ylim((0, 100))
    ax_clust[0].scatter(KK[n_clust-1], np.array(betweenss/totss*100)[n_clust-1], color='red')
    ax_clust[0].plot((n_clust, n_clust), (0, 100), lw=1, c='red')
    ax_clust[0].set_xlabel('Number of clusters')
    ax_clust[0].set_ylabel('Percentage of variance explained (%)')
    ax_clust[0].set_title('Elbow for KMeans clustering')
    
    if pca_ == True:
        for i in range(n_clust):
            x = [eigenvalue[j][0] for j in range(len(eigenvalue)) if KM_clusters[j] == i]
            y = [eigenvalue[j][1] for j in range(len(eigenvalue)) if KM_clusters[j] == i]
            confidence_ellipse(np.array(x),np.array(y),ax=ax_clust[1],edgecolor=colors[i])
            ax_clust[1].scatter(centroids[n_clust-1][i][0], centroids[n_clust-1][i][1], color='red', s=30)
        
        ax_clust[1].set_xlabel('PC1 ({:.2f}%)'.format(ExpVar[0]*100))
        ax_clust[1].set_ylabel('PC2 ({:.2f}%)'.format(ExpVar[1]*100))
        ax_clust[1].set_title('KMeans clustering on PCA')
        ax_clust[1].axvline(x=0.0,color='k',linestyle='--')
        ax_clust[1].axhline(y=0.0,color='k',linestyle='--')
    
    df = pd.concat((pd.DataFrame(indexes),pd.DataFrame(KM_clusters)),axis=1)
    df.columns = ['Individuals','Cluster']
    print('Kmeans - clusters: {}'.format(n_clust))
    print(df)
    
    return KM_clusters,df

 
    

def Optics(dataframe,MinPts):
    
    '''
    Computes Ordering Points To Identify the Clustering Structure (OPTICS) on a dataset
    and returns the cluster's number each individual belongs to.
    
    dataframe: dataset after using pandas.DataFrame.
        /!\ INDIVIDUALS AS ROWS AND VARIABLES AS COLUMNS
        
    MinPts (int): number of individuals in a neighborhood for a point to be considered as a core point.
    '''
    
    ExpVar,eigenvalue = PrinCompAn(dataframe)
    plt.close()
    clust = OPTICS(min_samples=MinPts, xi=0.05, min_cluster_size=0.05)
    clust.fit_predict(eigenvalue)
    reachability = clust.reachability_[clust.ordering_]
    labels = clust.labels_[clust.ordering_]
    print(labels)
    colors = viridis_colors(np.max(labels)+1)
    space = np.arange(len(eigenvalue))
    
    fig_optics,ax_optics=plt.subplots(1,2,figsize=(8,3), tight_layout=True)
    for klass, color in zip(range(len(colors)), colors):
        Xk = space[labels == klass]
        Rk = reachability[labels == klass]
        Xk_eigen = eigenvalue[clust.labels_ == klass]
        ax_optics[0].scatter(Xk, Rk, color=color, s=10, alpha=0.3)
        ax_optics[1].scatter(Xk_eigen[:, 0], Xk_eigen[:, 1], color=color, s=10, alpha=0.3)
    ax_optics[0].scatter(space[labels == -1], reachability[labels == -1], color='k', s=10, alpha=0.5)
    ax_optics[0].set_ylabel('Reachability (epsilon distance)')
    ax_optics[0].set_title('Reachability Plot')
    ax_optics[1].scatter(eigenvalue[clust.labels_ == -1, 0], eigenvalue[clust.labels_ == -1, 1], color='k', marker='+', s=20, alpha=0.4)
    ax_optics[1].set_title('Automatic Clustering\nOPTICS')
    ax_optics[1].axvline(x=0.0,color='k',linestyle='--')
    ax_optics[1].axhline(y=0.0,color='k',linestyle='--')




def Tsne(dataframe,perplexity):
    labels = dataframe.iloc[:,0].values.astype(int)
    n_clusters = np.arange(np.min(labels), np.max(labels)+1, 1)
    clr = viridis_colors(len(n_clusters))
    
    variables = dataframe.iloc[:,1:]
    for i in range(5):
        tsne = TSNE(perplexity=perplexity, n_iter=500).fit_transform(variables)
        df_tsne = pd.concat((pd.DataFrame(labels, columns=['Labels']),
                            pd.DataFrame(tsne, columns=['Dim1','Dim2'])), axis=1)
        
        fig, ax = plt.subplots()
        for j,txt in enumerate(range(df_tsne.shape[0])):
            ax.scatter(df_tsne['Dim1'][j], df_tsne['Dim2'][j], color=clr[labels[j]])
            ax.annotate(j,(df_tsne['Dim1'][j], df_tsne['Dim2'][j]))
        ax.set_title('Perplexity: {}'.format(perplexity))
    
    return df_tsne, labels, n_clusters, clr




def confidence_ellipse(x, y, ax, n_std=2.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2, facecolor=facecolor, **kwargs)

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
    return ax.add_patch(ellipse)      






if __name__ == '__main__':
    import PySimpleGUI as sg
    import os
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib import colors
    from matplotlib.patches import Ellipse
    import matplotlib.transforms as transforms
    from mpl_toolkits import mplot3d
    from statannot import add_stat_annotation
    import seaborn as sns
    import numpy as np
    import pandas as pd
    from scipy import stats
    from scipy import signal
    from scipy.signal import savgol_filter
    import scipy.cluster.hierarchy as hierarchy
    from scipy.cluster.vq import kmeans
    from scipy.spatial.distance import cdist,pdist
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.impute import SimpleImputer
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.cluster import KMeans
    from sklearn.cluster import OPTICS
    
    
    sg.theme('DarkBlue')
    
    layout = [[sg.Text('Single Excelfile')],[sg.InputText(size=(35,1)), sg.FileBrowse()],
              [sg.Text('Sheet'), sg.InputText(default_text='Sheet1', size=(10,1))],
              [sg.Button('Start file')],
              [sg.Text('Multiple Excelfiles')],[sg.InputText(size=(35,1)), sg.FolderBrowse()],
              [sg.Frame(layout=[
              [sg.Text('Freq'), sg.InputText(default_text='20',size=(3,1))],
              [sg.Text('[Ca2+]'), sg.InputText(default_text='2.5',size=(4,1))],
              [sg.Text('Parallel fiber'), sg.InputText(default_text='20210112_linescan1',size=(9,1))]], title='Settings for imaging', relief=sg.RELIEF_SUNKEN)],
              [sg.Button('Start files')],
              [sg.Frame(layout=[
              [sg.Checkbox('Reduction', size=(12, 1), default=True)],
              [sg.Button('PCA')]], title='PCA', relief=sg.RELIEF_SUNKEN)],
              [sg.Frame(layout=[
              [sg.Checkbox('PCA on', size=(12, 1), default=True)],
              [sg.Text('n_clust'), sg.InputText(default_text='2',size=(3,1))],
              [sg.Checkbox('HCPC', size=(12, 1), default=False),sg.Checkbox('Kmeans', size=(12, 1), default=False)],
              [sg.Checkbox('tSNE', size=(12, 1), default=False),sg.Text('perplexity'),sg.InputText(default_text='10',size=(3,1))],
              [sg.Checkbox('OPTICS', size=(12, 1), default=False),sg.Text('MinPts'), sg.InputText(default_text='3',size=(3,1))],
              [sg.Button('Clust')]], title='Clustering', relief=sg.RELIEF_SUNKEN)],
              [sg.Button('Clear', button_color=('white','darkred'))]]
              
    window = sg.Window('CLUSTERING ANALYSIS', layout, location=(0,0))
    
    while True:
       event, value = window.read()
       try:
           if event in (None, 'Close'):
               plt.close('all')
               break
           
            
           if event == 'Start file':
               file_amps = r'E:\workingdataset_concat.xlsx'
               # df_amps = pd.DataFrame(pd.read_excel(file_amps, sheet_name=str(value[1]))).iloc[:,1:]
               df_var_ = pd.DataFrame(pd.read_excel(file_amps))
               print('Dataset loaded')
               # df_var_ = load_ephy_xlsx(value[0], str(value[1]))

               # df_ppr = pd.DataFrame([df_amps.iloc[:,i]/df_amps.iloc[:,0] for i in range(df_amps.shape[1])]).transpose()
               # df_ppr1 = df_ppr.iloc[:,0:10]
               
               # x = np.arange(1,11,1)
               # fig,ax = plt.subplots(1,2,figsize=(8,4),tight_layout=True)
               # [ax[0].plot(x, df_amps.iloc[i,:].values, 'k', lw=10, alpha=0.2) for i in range(df_amps.shape[0])]
               # [ax[1].plot(x, df_ppr.iloc[i,:].values, 'k', lw=10, alpha=0.2) for i in range(df_amps.shape[0])]
               # ax[0].set_ylabel('DF/F0')
               # ax[0].set_xlabel('#Pulse')
               # ax[1].set_ylabel('Norm. DF/F0')
               # ax[1].set_xlabel('#Pulse')
        
        
           if event == 'Start files':
               df_var = load_img_xlsx(value[2],value[3]+'Hz',value[4]+'mM',value[5])
               df_ppr = pd.DataFrame([df_var.iloc[:,i]/df_var.iloc[:,0] for i in range(df_var.shape[1])]).transpose()
               fig2,ax2=plt.subplots(1,2,figsize=(6,3), tight_layout=True)
               
               for i in range(df_var.shape[0]):
                   ax2[0].plot(df_var.iloc[i,:].values,'k',marker='o',alpha=0.2)
                   ax2[1].plot(df_ppr.iloc[i,:].values,'k',marker='o',alpha=0.2)
        
        
           if event == 'PCA':
               PrinCompAn(df_var_,value[6])
        
        
           if event == 'Clust':
               clr = viridis_colors(int(value[8]))
               x = np.linspace(1, 10, 10)
               
               fig,ax = plt.subplots(figsize=(3,3))
               fig2,ax2 = plt.subplots(figsize=(3,3))
               fig3,ax3 = plt.subplots(figsize=(3,3))
               fig4,ax4 = plt.subplots(figsize=(3,3))
               fig5,ax5 = plt.subplots(figsize=(3,3))
               [ax.plot(x, (df_amps.iloc[i,:10].values)/(df_amps.iloc[i,0]), 'k', lw=8, alpha=0.1) for i in range(df_amps.shape[0])]
               [ax2.plot(x, df_amps.iloc[i,:10].values,'k', lw=8, alpha=0.1) for i in range(df_amps.shape[0])]
               ax.set_xlabel('#PEAK')
               ax.set_ylabel('PPR')
               ax.grid(axis='y',linestyle='--')
               ax2.set_xlabel('#PEAK')
               ax2.set_ylabel('DF/F')
               ax2.grid(axis='y',linestyle='--')
                   
               
               if value[9] == True:
                   HCPC_clusters,df = HierAscClass(df_var_,int(value[8]),value[7])
                   
                   # for i in range(int(value[8])):
                   #     ax.plot(x, np.mean([(df_amps.iloc[j,:10].values)/(df_amps.iloc[j,0]) for j in range(df_amps.shape[0]) if HCPC_clusters[j] == i], axis=0), color=clr[i])
                   #     ax2.plot(x, np.mean([df_amps.iloc[j,:10].values for j in range(df_amps.shape[0]) if HCPC_clusters[j] == i], axis=0), color=clr[i])
                   #     cluster = [(df_amps.iloc[j,:10].values)/(df_amps.iloc[j,0]) for j in range(df_amps.shape[0]) if HCPC_clusters[j] == i]
                   #     cluster2 = [df_amps.iloc[j,:10].values for j in range(df_amps.shape[0]) if HCPC_clusters[j] == i]
                   #     MEAN = np.mean(cluster, axis=0)
                   #     MEAN2 = np.mean(cluster2, axis=0)
                   #     SEM = stats.sem(cluster, axis=0)
                   #     SEM2 = stats.sem(cluster2, axis=0)
                   #     ax.fill_between(x, MEAN+SEM, MEAN-SEM, alpha=0.7, color=clr[i])
                   #     ax2.fill_between(x, MEAN2+SEM2, MEAN2-SEM2, alpha=0.7, color=clr[i])
        
                   # ax.set_title('STP clusters after HCPC\n{} clusters'.format(int(value[8])))
                   # ax2.set_title('STP clusters after HCPC\n{} clusters'.format(int(value[8])))
                   
                   # prob_release_clusters = [[1-(df_var_.iloc[j,12]/100) for j in range(df_var_.shape[0]) if HCPC_clusters[j]==i] for i in range(int(value[8]))]
                   # df_prob = pd.concat((pd.DataFrame(prob_release_clusters[0]), pd.DataFrame(prob_release_clusters[1])),axis=1)
                   # df_prob.columns = ['Gr1','Gr2']
                   # sns.boxplot(data=df_prob,showmeans=True, ax=ax3, palette=[clr[0],clr[1]], meanprops={'markerfacecolor':'red','markeredgecolor':'black','markersize':'8'})
                   # add_stat_annotation(ax3, data=df_prob, box_pairs=[('Gr1','Gr2')],test='Mann-Whitney', text_format='star', verbose=2)
                   
                   # amp1_clusters = [[df_var_.iloc[j,1] for j in range(df_var_.shape[0]) if HCPC_clusters[j]==i] for i in range(int(value[8]))]
                   # dfAmp1 = pd.concat((pd.DataFrame(amp1_clusters[0]), pd.DataFrame(amp1_clusters[1])),axis=1)
                   # dfAmp1.columns = ['Gr1','Gr2']
                   # sns.boxplot(data=dfAmp1,showmeans=True, ax=ax4, palette=[clr[0],clr[1]], meanprops={'markerfacecolor':'red','markeredgecolor':'black','markersize':'8'})
                   # add_stat_annotation(ax4, data=dfAmp1, box_pairs=[('Gr1','Gr2')],test='Mann-Whitney', text_format='star', verbose=2)
                            
                   # ppr_clusters = [[df_var_.iloc[j,3] for j in range(df_var_.shape[0]) if HCPC_clusters[j]==i] for i in range(int(value[8]))]
                   # dfPpr = pd.concat((pd.DataFrame(ppr_clusters[0]), pd.DataFrame(ppr_clusters[1])),axis=1)
                   # dfPpr.columns = ['Gr1','Gr2']
                   # sns.boxplot(data=dfPpr,showmeans=True, ax=ax5, palette=[clr[0],clr[1]], meanprops={'markerfacecolor':'red','markeredgecolor':'black','markersize':'8'})
                   # add_stat_annotation(ax5, data=dfPpr, box_pairs=[('Gr1','Gr2')],test='Mann-Whitney', text_format='star', verbose=2)
                   
                   
               if value[10] == True:
                  KM_clusters,df = KMean(df_var_,int(value[8]),value[7])
                  
                  for i in range(int(value[8])):
                      ax.plot(x, np.mean([(df_amps.iloc[j,:10].values)/(df_amps.iloc[j,0]) for j in range(df_amps.shape[0]) if KM_clusters[j] == i], axis=0), color=clr[i])
                      ax2.plot(x, np.mean([df_amps.iloc[j,:10].values for j in range(df_amps.shape[0]) if KM_clusters[j] == i], axis=0), color=clr[i])
                      cluster = [(df_amps.iloc[j,:10].values)/(df_amps.iloc[j,0]) for j in range(df_amps.shape[0]) if KM_clusters[j] == i]
                      cluster2 = [df_amps.iloc[j,:10].values for j in range(df_amps.shape[0]) if KM_clusters[j] == i]
                      MEAN = np.mean(cluster, axis=0)
                      MEAN2 = np.mean(cluster2, axis=0)
                      SEM = stats.sem(cluster, axis=0)  
                      SEM2 = stats.sem(cluster2, axis=0)
                      ax.fill_between(x, MEAN+SEM, MEAN-SEM, alpha=0.7, color=clr[i])
                      ax2.fill_between(x, MEAN2+SEM2, MEAN2-SEM2, alpha=0.7, color=clr[i])
                      
                  ax.set_title('STP clusters after Kmeans\n{} clusters'.format(int(value[8])))
                  ax2.set_title('STP clusters after Kmeans\n{} clusters'.format(int(value[8])))
                  
                       
               if value[11] == True:
                  df_tsne, labels, n_clusters, clr = Tsne(df_var_, int(value[12]))
                  
                  # for i in n_clusters:
                  #     ax.plot(x, np.mean([(df_amps.iloc[j,:10].values)/(df_amps.iloc[j,0]) for j in range(df_amps.shape[0]) if labels[j] == i], axis=0), color=clr[i])
                  #     ax2.plot(x, np.mean([df_amps.iloc[j,:10].values for j in range(df_amps.shape[0]) if labels[j] == i], axis=0), color=clr[i])
                  #     cluster = [(df_amps.iloc[j,:10].values)/(df_amps.iloc[j,0]) for j in range(df_amps.shape[0]) if labels[j] == i]
                  #     cluster2 = [df_amps.iloc[j,:10].values for j in range(df_amps.shape[0]) if labels[j] == i]
                  #     MEAN = np.mean(cluster, axis=0)
                  #     MEAN2 = np.mean(cluster2, axis=0)
                  #     SEM = stats.sem(cluster, axis=0)
                  #     SEM2 = stats.sem(cluster2, axis=0)
                  #     ax.fill_between(x, MEAN+SEM, MEAN-SEM, alpha=0.7, color=clr[i])
                  #     ax2.fill_between(x, MEAN2+SEM2, MEAN2-SEM2, alpha=0.7, color=clr[i])
                   
                  # ax.set_title('STP clusters after t-SNE\n{} clusters'.format(len(n_clusters)))
                  # ax2.set_title('STP clusters after t-SNE\n{} clusters'.format(len(n_clusters)))

               
               if value[13] == True:
                  Optics(df_var_,int(value[14]))
                  
           
           if event == 'Clear':
                plt.close('all')
       except:
           pass
    window.close()