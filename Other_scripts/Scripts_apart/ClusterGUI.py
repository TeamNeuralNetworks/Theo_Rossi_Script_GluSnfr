# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 15:00:03 2021

@author: Theo.ROSSI
"""

def viridis_colors(num):
    
    '''
    Creates a list of color values converted in color names
    
    num (int): number of returned colors
    '''
    
    clr = []
    cmap = cm.viridis(np.linspace(0.1,0.85,num))
    for c in range(num):
        rgba = cmap[c]
        clr.append(colors.rgb2hex(rgba))
    return clr




def PrinCompAn(df_pca, threshold=0.95):
    
    '''
    Computes a Pincipal Component Analysis on a dataframe and returns the explained variance
    of each principal component (PC) and the eigenvalues of each individual.
        
    df_pca: dataset after using pandas.DataFrame.
        /!\ INDIVIDUALS AS ROWS AND VARIABLES AS COLUMNS
    
    threshold (float): set the number of principal components t. Default: 0.95.
    
    
    Returns a figure with:
        - elbow plot (ax1): gives the number of PCs explaining at least 95% of the total variance.
        - bar plot (ax2): explained variance by each PC.
        - scatter plot (ax3): position of each individual in the new dimentional subspace.
        - bar plot (ax4 and ax5): contribution of each variable on the first (ax4) and second (ax5) PCs.
    '''
            
    pca = PCA()
    pca_fit = pca.fit(df_pca)
    ExpVar = pca_fit.explained_variance_ratio_
    cumsum = np.cumsum(ExpVar)
    d = np.argmax(cumsum >= threshold) + 1
    VarComp = pd.DataFrame(pca_fit.components_[:, :d]) #Contribution of each variable on each PC

    PrCoAn = PCA(n_components = d)
    eigenvalue = PrCoAn.fit_transform(df_pca)
    df_eigen = pd.DataFrame(eigenvalue, columns = [f'PC{i+1}' for i in range(d)])
    
    x = np.arange(1, len(ExpVar) + 1)
    
    fig_PCA, ax_PCA = plt.subplots(1, 3, figsize = (17,4), tight_layout = True)
    
    ax_PCA[0].plot(x, cumsum, marker='o', linestyle='--', color='b')
    ax_PCA[0].set_xlabel('Number of Principal Components')
    ax_PCA[0].set_ylabel('Cumulative variance (%)')
    ax_PCA[0].set_xticks(x)
    ax_PCA[0].set_title('Number of components\nneeded to explain variance')
    ax_PCA[0].axhline(y = threshold, color = 'r', linestyle = '--')
    ax_PCA[0].text(0.5, threshold + 0.01, f'{threshold*100}% total variance', color = 'red', fontsize = 10)
    ax_PCA[0].grid(axis = 'x')
    ax_PCA[1].bar(x, ExpVar)
    ax_PCA[1].set_title("Explained variance")
    ax_PCA[1].set_ylabel("Norm variance")
    ax_PCA[1].set_xlabel("#Component")

        
    [ax_PCA[2].scatter(df_eigen['PC1'][eigen], df_eigen['PC2'][eigen], color='k', alpha=0.4) for eigen in range(df_eigen.shape[0])]
    [ax_PCA[2].annotate(i, (df_eigen['PC1'][i], df_eigen['PC2'][i])) for i, txt in enumerate(range(df_eigen.shape[0]))]
    ax_PCA[2].set_title('PCA')
    ax_PCA[2].set_xlabel('PC1 ({:.2f}%)'.format(ExpVar[0]*100))
    ax_PCA[2].set_ylabel('PC2 ({:.2f}%)'.format(ExpVar[1]*100))
    ax_PCA[2].axvline(x=0.0,color='k',linestyle='--')
    ax_PCA[2].axhline(y=0.0,color='k',linestyle='--')
    

    plt.figure()
    ax_pca3d = plt.axes(projection='3d')
    
    [ax_pca3d.scatter3D(df_eigen['PC1'][eigen], df_eigen['PC2'][eigen], df_eigen['PC3'][eigen], color='k', alpha=0.4) for eigen in range(df_eigen.shape[0])]
    [ax_pca3d.text(x, y, z, label) for x, y, z, label in zip(df_eigen['PC1'], df_eigen['PC2'], df_eigen['PC3'], range(df_eigen.shape[0]))] 
    
    ax_pca3d.set_title('PCA')
    ax_pca3d.set_xlabel('PC1 ({:.2f}%)'.format(ExpVar[0]*100))
    ax_pca3d.set_ylabel('PC2 ({:.2f}%)'.format(ExpVar[1]*100))
    ax_pca3d.set_zlabel('PC3 ({:.2f}%)'.format(ExpVar[2]*100))
    
    
    fig_contribution, ax_contribution = plt.subplots(1,d,figsize=(17,4),sharey=True,tight_layout=True)
    for plot in range(d):
        ax_contribution[plot].set_title(f'PC{plot+1} contributions')
        ax_contribution[plot].bar(x, VarComp[plot])
        ax_contribution[plot].set_xticks(x)
        ax_contribution[plot].set_xticklabels(df_pca.columns, rotation='vertical', fontsize=7)
        ax_contribution[plot].set_ylabel('Variable contribution')
        ax_contribution[plot].axhline(y=0.0, color='k', linestyle='--')

    return ExpVar, df_eigen




def Tsne(df_tSNE, perplexity, iteration, classes = 1, class_color = False):
    labels = range(df_tSNE.shape[0])
    clr = viridis_colors(np.max(classes)+1)

    for i in range(5):
        tsne = TSNE(perplexity=perplexity, n_iter=iteration).fit_transform(df_tSNE)
        df_tsne = pd.concat((pd.DataFrame(labels, columns=['Labels']),
                            pd.DataFrame(tsne, columns=['Dim1','Dim2'])), axis=1)

        fig, ax = plt.subplots(figsize=(4,3))
        
        for j, txt in enumerate(range(df_tsne.shape[0])):
            if class_color == True:
                ax.scatter(df_tsne['Dim1'][j], df_tsne['Dim2'][j], color=clr[classes[j]], alpha=0.5)
        
            else:
                ax.scatter(df_tsne['Dim1'][j], df_tsne['Dim2'][j], color='c')
            
            ax.annotate(j,(df_tsne['Dim1'][j], df_tsne['Dim2'][j]), color='k')
                
        ax.set_title('Perplexity: {}'.format(perplexity))



def outliers_detection(df_out, outliers_proportion = 0.01, outliers_off = False):
    
    model_isolation = IsolationForest(contamination = outliers_proportion)
    model_isolation.fit(df_out)
    outliers_ = model_isolation.predict(df_out) == -1

    df_outliers = df_out[outliers_]
    print('------------------')
    print('Outliers')
    print('------------------')
    print(df_outliers)
    print('------------------')

    if outliers_off == True:
        
        df_out['Labels_'] = df_out.index
        df = df_out.drop(df_outliers.index, axis = 0)
        print('------------------')
        print('Dataset without outliers')
        print('------------------')
    
    else:
        df = df_out
        print('------------------')
        print('Dataset')
        print('------------------')
    
    print(df)
    print('------------------')
    
    plt.figure()
    plt.scatter(df_out.iloc[:,0], df_out.iloc[:,1], color = 'k', alpha = 0.4)
    plt.scatter(df_out.iloc[:,0][outliers_], df_out.iloc[:,1][outliers_], color = 'r')
    [plt.annotate(i, (df_out.iloc[i,0], df_out.iloc[i,1])) for i, txt in enumerate(range(df_out.shape[0]))] 
    plt.xlabel('{}'.format(df_out.columns[0]))
    plt.ylabel('{}'.format(df_out.columns[1]))
    plt.title('Dataset and outliers')
    
    if df.columns[2] == 'Labels_':
        pass
    
    else:
        plt.figure()
        ax = plt.axes(projection = '3d')
        ax.scatter3D(df_out.iloc[:,0], df_out.iloc[:,1], df_out.iloc[:,2], color = 'k', alpha = 0.3)
        ax.scatter3D(df_out.iloc[:,0][outliers_], df_out.iloc[:,1][outliers_], df_out.iloc[:,2][outliers_], color = 'r')
        [ax.text(x, y, z, label) for x, y, z, label in zip(df_out.iloc[:,0], df.iloc[:,1], df_out.iloc[:,2], range(df_out.shape[0]))]
        ax.set_xlabel('{}'.format(df_out.columns[0]))
        ax.set_ylabel('{}'.format(df_out.columns[1]))
        ax.set_zlabel('{}'.format(df_out.columns[2]))
        ax.set_title('Dataset and outliers')
    
    return df_outliers, df

    
    

def KMean(df_km, n_clust, pca_=True, explained_variance=None, graph3D = True):
    
    '''
    Computes a Kmeans on a dataset and returns the cluster's number each individual belongs to.
    
    df_km: dataset after using pandas.DataFrame.
        /!\ INDIVIDUALS AS ROWS AND VARIABLES AS COLUMNS
        
    n_clust (int): number of arbitrary clusters.
    
    pca_ (bool): if True, computes the Kmeans on the dataset after PCA.
                 if False, computes the Kmeans on the raw dataset.
    
    outliers (float): percentage of outliers in the dataset. Range from 0.0 to 1.0.
    '''
    
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
    
    
    colors = viridis_colors(n_clust)

    df = df_km.dropna(axis = 0, how = 'all')
    
    if 'Labels_' in df_km.columns:
        df_km = df.drop(['Labels_'], axis = 1)
        labels = df['Labels_']


    #Model implementation
    model = KMeans(n_clusters = n_clust)
    KM_clusters = model.fit_predict(df_km)
    

    #Calculation of distances from centroids
    K_MAX = int(10)
    KK = range(1,K_MAX+1)
    KM_ = [kmeans(df_km, k) for k in KK]
    centroids = [cent for (cent, var) in KM_]
    D_k = [cdist(df_km, cent, 'euclidean') for cent in centroids]
    dist = [np.min(dist_value, axis=1) for dist_value in D_k]

    tot_withinss = [sum(d**2) for d in dist]  # Total within-cluster sum of squares
    totss = sum(pdist(df_km)**2)/df_km.shape[0]       # The total sum of squares
    betweenss = totss - tot_withinss          # The between-cluster sum of squares

    #Elbow plot for KMeans
    plt.figure(figsize=(5,4))
    plt.plot(KK, betweenss/totss*100, marker='o')
    plt.ylim((0, 100))
    plt.axvline(x=n_clust, color='red', linestyle='--', lw=1)
    plt.text(0.75, 95, '{:.2f}%'.format(np.array(betweenss/totss*100)[n_clust-1]), color='red')
    plt.xlabel('Number of clusters')
    plt.ylabel('Percentage of variance explained (%)')
    if pca_ == True:
        plt.title('Elbow for KMeans clustering with PCA')
    else:
        plt.title('Elbow for KMeans clustering without PCA')
    
    
    #PCA plots with clustering label colors
    fig, ax = plt.subplots(figsize=(5,4))
    
    [ax.scatter(df_km.iloc[i,0], df_km.iloc[i,1], color=colors[KM_clusters[i]], alpha=0.5) for i in range(df_km.shape[0])]
    
    if 'Labels_' in df.columns:
        [ax.annotate(txt, (df_km.iloc[i,0], df_km.iloc[i,1])) for i, txt in enumerate(labels)]
    
    else:
        [ax.annotate(txt, (df_km.iloc[i,0], df_km.iloc[i,1])) for i, txt in enumerate(range(df_km.shape[0]))]
        
        
    for i in range(n_clust):
        
        x = [df_km.iloc[j,0] for j in range(df_km.shape[0]) if KM_clusters[j] == i]
        y = [df_km.iloc[j,1] for j in range(df_km.shape[0]) if KM_clusters[j] == i]
        confidence_ellipse(np.array(x), np.array(y), ax=ax, edgecolor=colors[i])
        
    ax.scatter(model.cluster_centers_[:,0], model.cluster_centers_[:,1], c = 'r')
        
    if pca_ == True:
        ax.set_xlabel('{} ({:.2f}%)'.format(df_km.columns[0], explained_variance[0]*100))
        ax.set_ylabel('{} ({:.2f}%)'.format(df_km.columns[1], explained_variance[1]*100))
        ax.set_title('KMeans clustering on PCA')
        ax.axvline(x=0.,color='k',linestyle='--')
        ax.axhline(y=0.,color='k',linestyle='--')
    else:
        ax.set_xlabel('{}'.format(df_km.columns[0]))
        ax.set_ylabel('{}'.format(df_km.columns[1]))
        ax.set_title('KMeans clustering')
    
    if graph3D == True:
 
       plt.figure()
       ax = plt.axes(projection='3d')
       
       [ax.scatter3D(df_km.iloc[i,0], df_km.iloc[i,1], df_km.iloc[i,2], color=colors[KM_clusters[i]], alpha=0.5) for i in range(df_km.shape[0])]
       [ax.text(x, y, z, label) for x, y, z, label in zip(df_km.iloc[:,0], df_km.iloc[:,1], df_km.iloc[:,2], range(df_km.shape[0]))]
       
       if pca_ == True:
           ax.set_xlabel('{} ({:.2f}%)'.format(df_km.columns[0], explained_variance[0]*100))
           ax.set_ylabel('{} ({:.2f}%)'.format(df_km.columns[1], explained_variance[1]*100))
           ax.set_zlabel('{} ({:.2f}%)'.format(df_km.columns[2], explained_variance[2]*100))
           ax.set_title('KMeans clustering on PCA')
       
       else:
           ax.set_xlabel('{}'.format(df_km.columns[0]))
           ax.set_ylabel('{}'.format(df_km.columns[1]))
           ax.set_zlabel('{}'.format(df_km.columns[2]))
           ax.set_title('KMeans clustering')
   
    
    df_km['Cluster'] = KM_clusters
    
    print('------------------')
    print('Kmeans - clusters: {}'.format(n_clust))
    print('------------------')
    print(df_km)
    print('------------------')
    
    return KM_clusters, df_km




def HierAscClass(df_hc, n_clust, pca_=True, explained_variance=None, graph3D = True):
    
    '''
    Computes an Ascending Hierarchical Classification (AHC) using the Ward method on a dataset
    and returns the cluster's number each individual belongs to.
    
    df_hc: dataset after using pandas.DataFrame.
        /!\ INDIVIDUALS AS ROWS AND VARIABLES AS COLUMNS
        
    n_clust (int): number of arbitrary clusters.
    
    pca_ (bool): if True, computes the AHC on the dataset after PCA.
                 if False, computes the AHC on the raw dataset.
    '''


    colors = viridis_colors(n_clust)
    
    df = df_hc.dropna(axis = 0, how = 'all')
    
    if 'Labels_' in df.columns:
        df_hc = df.drop(['Labels_'], axis = 1)
        labels = df['Labels_']
        
    
    #Model implementation
    distance_matrix = pdist(df_hc)
    HCPC = AgglomerativeClustering(n_clusters = n_clust)
    HCPC_clusters = HCPC.fit_predict(df_hc)
    Z = linkage(distance_matrix, method='ward', metric='euclidean', optimal_ordering=True)
    Inertia = sorted(Z[:,2],reverse=True)
    
    
    #Elbow plot for inertia
    plt.figure()
    plt.plot(np.arange(1, len(Inertia) + 1), Inertia)
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.title('Elbow plot for HCPC inertia')
    
    
    #Dendrogram
    fig, ax = plt.subplots(figsize=(5,4))
    for i in range(len(HCPC_clusters)):
        set_link_color_palette(colors)
    dendrogram(Z, orientation='top', ax=ax, color_threshold=[(Inertia[i]-0.1) for i in range(len(colors)-1)][-1])
    ax.axhline(y = [(Inertia[i]-0.1) for i in range(len(colors)-1)][-1], color='r', linestyle='--')
    ax.set_ylabel('Inertia gain')
    
    if pca_ == True:
        ax.set_title('HCPC dendrogram on PCA')
    else:
        ax.set_title('HCPC dendrogram')
    

    plt.figure()
    
    [plt.scatter(df_hc.iloc[i,0], df_hc.iloc[i,1], color = colors[HCPC_clusters[i]], alpha = 0.5) for i in range(len(HCPC_clusters))]
    
    if 'Labels_' in df.columns:
        [plt.annotate(txt, (df_hc.iloc[j,0], df_hc.iloc[j,1])) for j, txt in enumerate(labels)]
    else:
        [plt.annotate(txt, (df_hc.iloc[j,0], df_hc.iloc[j,1])) for j, txt in enumerate(range(df_hc.shape[0]))]
        
    
    if pca_ == True:
        
        plt.xlabel('PC1 ({:.2f}%)'.format(explained_variance[0]*100))
        plt.ylabel('PC2 ({:.2f}%)'.format(explained_variance[1]*100))
        plt.title('HCPC on PCA')
        plt.axvline(x = 0.0, color = 'k', linestyle = '--')
        plt.axhline(y = 0.0, color = 'k', linestyle = '--')
    
    else:
        plt.xlabel('{}'.format(df_hc.columns[0]))
        plt.ylabel('{}'.format(df_hc.columns[1]))
        plt.title('HCPC')
        
        
    if graph3D == True:
        
        plt.figure()
        ax = plt.axes(projection='3d')
        
        [ax.scatter3D(df_hc.iloc[i,0], df_hc.iloc[i,1], df_hc.iloc[i,2], color=colors[HCPC_clusters[i]], alpha=0.5) for i in range(len(HCPC_clusters))]
        [ax.text(x, y, z, label) for x, y, z, label in zip(df_hc.iloc[:,0], df_hc.iloc[:,1], df_hc.iloc[:,2], range(df_hc.shape[0]))]
        
        if pca_ == True:
            
            ax.set_xlabel('PC1 ({:.2f}%)'.format(explained_variance[0]*100))
            ax.set_ylabel('PC2 ({:.2f}%)'.format(explained_variance[1]*100))
            ax.set_zlabel('PC3 ({:.2f}%)'.format(explained_variance[2]*100))
            ax.set_title('HCPC on PCA')
        
        else:
            
            ax.set_xlabel('{}'.format(df_hc.columns[0]))
            ax.set_ylabel('{}'.format(df_hc.columns[1]))
            ax.set_zlabel('{}'.format(df_hc.columns[2]))
            ax.set_title('HCPC')
            
    df_hc['Cluster'] = HCPC_clusters
    print('------------------')
    print('HCPC - clusters: {}'.format(n_clust))
    print('------------------')
    print(df_hc)
    print('------------------')
    
    return HCPC_clusters, df_hc




def fill_confusion_matrix(model, conf_mat_list, X_train, y_train, X_test, y_test):
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test) 
    confusion_mat = pd.crosstab(y_test, y_pred, rownames=["Actual"], colnames=["Predicted"]) 
    print(confusion_mat) 
    
    conf_mat_list.append(confusion_mat.div(confusion_mat.sum(axis=1), axis=0))
    
    
    

def analysis_window(dataframe):
    
    sg.theme('DarkBlue')
    
    tab_pca = [[sg.Checkbox('PCA on', key='PCA on')],
               [sg.InputText(default_text='0.95', size=(5,1), key='threshold'), sg.Text('Threshold')],
               [sg.Button('RUN PCA')]]
    
    tab_tsne = [[sg.Checkbox('t-SNE on', key='tSNE_on')],
                [sg.InputText(default_text='5', size=(3,1), key='perplexity'), sg.Text('Perplexity')],
                [sg.InputText(default_text='1000', size=(6,1), key='iterations'), sg.Text('Iterations')],
                [sg.Button('RUN t-SNE')]]
    
    tab_clustering = [[sg.Checkbox('Kmeans', key='Kmeans'), sg.InputText(default_text='2', size=(3,1), key='kmean_clusters'), sg.Text('Kmean clusters')],
                      [sg.Checkbox('Hierarchical clustering', key = 'HierAscClass'), sg.InputText(default_text='2', size=(3,1), key='Hcpc_clusters'), sg.Text('HCPC clusters')],
                      [sg.Button('RUN CLUSTERING')]]
    
    tab_learning = [[sg.Text('Cross-val method')],
                    [sg.InputText(default_text='20', size=(4,1), key='Split'), sg.Text('Splits')],
                    [sg.Checkbox('Train Test Split', key='TTS'), sg.InputText(default_text='0.2', size=(3,1), key='TTS_test_size'), sg.Text('TTS test size')],
                    [sg.Checkbox('Stratified KFold', key='SKF')],
                    [sg.Checkbox('Stratified Shuffle Split', key='SSS'), sg.InputText(default_text='0.2', size=(3,1), key='SSS_test_size'), sg.Text('SSS test size')], 
                    [sg.Text('Model')],
                    [sg.Checkbox('Random forest', key='RandForest')],
                    [sg.Button('RUN LEARNING')]]
    
    analysis_layout = [[sg.Checkbox('Supervised target', key='target'), sg.InputText(size=(7,1), key='target_text'), sg.Text('Target column name')],
                       [sg.Frame(layout=[
                       [sg.Text('Visualize dataset'), sg.Button('GRAPH')]], title='2D OR 3D DATASET', relief=sg.RELIEF_SUNKEN)],
                       [sg.Frame(layout=[
                       [sg.TabGroup([[sg.Tab('PCA', tab_pca), sg.Tab('t-SNE', tab_tsne)]])]], title='DIMENSIONAL REDUCTION', relief=sg.RELIEF_SUNKEN)],
                       [sg.Frame(layout=[
                       [sg.InputText(default_text='0.01', size=(4,1), key='contamination'), sg.Text('Proportion of outliers')],
                       [sg.Checkbox('Outliers Off', key = 'outliers_off')],
                       [sg.Button('RUN ISOLATION FOREST')]], title='ISOLATION FOREST', relief=sg.RELIEF_SUNKEN)],
                       [sg.Frame(layout=[
                       [sg.TabGroup([[sg.Tab('Clustering', tab_clustering), sg.Tab('Prediction', tab_learning)]])]], title='CLASSIFICATION', relief=sg.RELIEF_SUNKEN)]]
              
    
    analysis_window = sg.Window('CLUSTERING ANALYSIS', analysis_layout, location=(0,0))
    
    
    while True:
        
       analysis_event, analysis_value = analysis_window.read()
       
       try:
           if analysis_event in (None, 'Close'):
               break
           
            
           if analysis_value['target'] == True:
               
               X = dataframe.drop([analysis_value['target_text']], axis=1)
               y = dataframe[analysis_value['target_text']]
               
           
           else:
               
               X = dataframe
              
           
           if analysis_event == 'GRAPH':
               
               plt.figure()
               plt.scatter(X.iloc[:,0], X.iloc[:,1], color = 'k', alpha = 0.4)
               [plt.annotate(i, (X.iloc[i,0], X.iloc[i,1])) for i, txt in enumerate(range(X.shape[0]))]
               plt.xlabel('{}'.format(X.columns[0]))
               plt.ylabel('{}'.format(X.columns[1]))
               plt.title('Dataset')
               
               if dataframe.shape[1] == 3:
                   
                   plt.figure()
                   ax = plt.axes(projection='3d')
                   ax.scatter3D(X.iloc[:,0], X.iloc[:,1], X.iloc[:,2], color = 'k', alpha = 0.4)
                   [ax.text(x, y, z, label) for x, y, z, label in zip(X.iloc[:,0], X.iloc[:,1], X.iloc[:,2], range(X.shape[0]))]
                   ax.set_xlabel('{}'.format(X.columns[0]))
                   ax.set_ylabel('{}'.format(X.columns[1]))
                   ax.set_zlabel('{}'.format(X.columns[2]))
                   ax.set_title('Dataset')
                   
                   
           if analysis_event == 'RUN PCA':
               
               ExpVar, df_eigen = PrinCompAn(X, float(analysis_value['threshold']))

            
           if analysis_event == 'RUN t-SNE':
               
               Tsne(X, int(analysis_value['perplexity']), int(analysis_value['iterations']))
                     
           
           if analysis_event == 'RUN ISOLATION FOREST':
               
                if analysis_value['PCA on'] == True:
                    
                    if analysis_value['outliers_off'] == True:
                
                        df_outliers, df_eigen = outliers_detection(df_eigen, float(analysis_value['contamination']), analysis_value['outliers_off'])

                    else:
                        
                        outliers_detection(df_eigen, float(analysis_value['contamination']), analysis_value['outliers_off'])
                
                else:
                    
                    if analysis_value['outliers_off'] == True:
                        
                        df_outliers, df_iso = outliers_detection(X, float(analysis_value['contamination']), analysis_value['outliers_off'])
                        
                    else:
                        
                        outliers_detection(X, float(analysis_value['contamination']), analysis_value['outliers_off'])
                   
               
           if analysis_event == 'RUN CLUSTERING':

               if analysis_value['Kmeans'] == True:
                   
                   if analysis_value['PCA on'] == True:
                       
                       KM_clusters, df_clust = KMean(df_eigen, int(analysis_value['kmean_clusters']), analysis_value['PCA on'],
                                                     ExpVar, float(analysis_value['contamination']))
                       
                   else:
                       
                       if analysis_value['outliers_off'] == True:
                       
                           KM_clusters, df_clust = KMean(df_iso, int(analysis_value['kmean_clusters']), analysis_value['PCA on'],
                                                         float(analysis_value['contamination']))
                       
                       else:
                           
                           KM_clusters, df_clust = KMean(X, int(analysis_value['kmean_clusters']), analysis_value['PCA on'],
                                                         float(analysis_value['contamination']))
                   
                   if analysis_value['tSNE_on'] == True:
                       Tsne(X, int(analysis_value['perplexity']), int(analysis_value['iterations']), KM_clusters, class_color=True)
                           
                   
               if analysis_value['HierAscClass'] == True:
                    
                   if analysis_value['PCA on'] == True:
                       
                       HCPC_clusters, df_clust = HierAscClass(df_eigen, int(analysis_value['Hcpc_clusters']), analysis_value['PCA on'], ExpVar)
                        
                   else:
                       
                       if analysis_value['outliers_off'] == True:
                           
                           HCPC_clusters, df_clust = HierAscClass(df_iso, int(analysis_value['Hcpc_clusters']), analysis_value['PCA on'])
                       
                       else:
                           
                           HCPC_clusters, df_clust = HierAscClass(X, int(analysis_value['Hcpc_clusters']), analysis_value['PCA on'])
                        
                       
                   if analysis_value['tSNE_on'] == True:
                       Tsne(X, int(analysis_value['perplexity']), int(analysis_value['iterations']), HCPC_clusters, class_color=True)
               

               X['Clusters'] = df_clust['Cluster']
               
               if analysis_value['outliers_off'] == True:
                   
                   outliers = X.iloc[df_outliers.index, :]
                   X = X.drop(df_outliers.index, axis = 0)

                   print('------------------')
                   print('Outliers from original dataset')
                   print('------------------')
                   print(outliers)
                   print('------------------')
               
               print('------------------')
               print('Dataset clusterized')
               print('------------------')
               print(X)
               print('------------------')
               
               

           if analysis_event == 'RUN LEARNING':

               if analysis_value['target'] == False:
                   
                   X = X.drop(['Clusters'], axis=1)
                   y = X['Clusters']
                   

               if analysis_value['TTS'] == True:

                   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=float(analysis_value['TTS_test_size']), random_state=0)
                    
                   
               if analysis_value['SKF'] == True:
                   
                   splitting = StratifiedKFold(n_splits = int(analysis_value['Split']), random_state = 0)
                                         
                   
               if analysis_value['SSS'] == True:
                   
                   splitting = StratifiedShuffleSplit(n_splits = int(analysis_value['Split']), test_size = float(analysis_value['SSS_test_size']), random_state = 5)
                 
                    
               if analysis_value['RandForest'] == True:
                   
                   model = RandomForestClassifier()


               param_grid = {'n_estimators': np.arange(1, 20), 'max_depth': np.arange(1,20)}
               
               grid = GridSearchCV(RandomForestClassifier(), param_grid, cv = 5)
               grid.fit(X_train, y_train)
               print(grid.best_score_)
               print(grid.best_params_)
               
               model = grid.best_estimator_
               
               confusion_mats_all = []
               fill_confusion_matrix(model, confusion_mats_all, X_train, y_train, X_test, y_test)
    
    
    
               # confusion_mats_all = []
               # score_all = []
               
               # for (train, test), i in zip(splitting.split(XX,y), range(1, int(analysis_value['Split'])+1)):
                   
               #     X_train, X_test = XX.iloc[train], XX.iloc[test]
               #     y_train, y_test = y.iloc[train], y.iloc[test]
               #     fill_confusion_matrix(model, confusion_mats_all, X_train, y_train, X_test, y_test)
               #     score_all.append(model.score(X_test, y_test))
                  
               # confusion_mat_final = pd.concat(confusion_mats_all).groupby(level=0).mean()
               
               fig, ax = plt.subplots()
               sns.heatmap(confusion_mats_all[0], cmap='magma_r', vmin=0., vmax=1.,
                           xticklabels=[f'Clust {i}' for i in range(len(np.unique(y)))],
                           yticklabels=[f'Clust {i}' for i in range(len(np.unique(y)))],
                           ax=ax, annot=True, square=True)
               
               N, train_score, val_score = learning_curve(model, X_train, y_train, train_sizes = np.linspace(0.1, 1.0, 10), cv=5)
               plt.figure()
               plt.plot(N, train_score.mean(axis=1), label='train')
               plt.plot(N, val_score.mean(axis=1), label='validation')
               plt.legend()
              
       except:
           pass
       
    analysis_window.close()


DF = []

def variables(tab):
    
    '''
    /!\ Tab must be a dictionary
    '''
    
    tabs_checkbox = {}
    
    for i in tab:
        tabs_checkbox[i] = []
    
        for j in range(len(tab[i].columns)):
            tabs_checkbox[i].append([sg.Checkbox(f'{tab[i].columns[j]}', key=f'{i}_{tab[i].columns[j]}')])
            
    sg.theme('DarkBlue')
    
    tab_layout = [[sg.Frame(layout=[
                  [sg.TabGroup([[sg.Tab(sheet, checkbox_list) for sheet, checkbox_list in tabs_checkbox.items()]])]], title='VARIABLES SELECTION', relief=sg.RELIEF_SUNKEN)],
                  [sg.Frame(layout=[
                  [sg.InputText(default_text = 'median', size=(7,1), key='Imputer'), sg.Text('Imputer strategy')],
                  [sg.Checkbox('Pre-processing', key='preprocessing', default=True)]], title='PREPROCESSING', relief=sg.RELIEF_SUNKEN)],
                  [sg.Button('ANALYSE')]]
    
    window = sg.Window('VARIABLES').Layout([[sg.Column(tab_layout, size=(300,900), scrollable=True)]])
    
    while True:
        
        event, value = window.read()
        
        try:
            if event in (None, 'Close'):
                break
                
            
            if event == 'ANALYSE':
                
                df_variables = pd.DataFrame(index=None, columns=None)
                
                for sheet, checkbox_list in tabs_checkbox.items():
                    
                    for item in range(len(checkbox_list)):
                        
                        if value[f'{sheet}_{tab[sheet].columns[item]}'] == True:

                            if tab[sheet][tab[sheet].columns[item]].dtype == 'object':
                                
                                if value['preprocessing'] == True:
                                    
                                    encoder = LabelEncoder()
                                    targets = encoder.fit_transform(tab[sheet][tab[sheet].columns[item]])
                                
                                else:
                                    
                                    targets = tab[sheet][tab[sheet].columns[item]]
                                    
                                df_variables[f'{tab[sheet].columns[item]}'] = targets
                            
                            else:
                                
                                if value['preprocessing'] == True:
                                    
                                    var = np.array(tab[sheet][tab[sheet].columns[item]]).reshape(-1,1)
                                    
                                    imputer = SimpleImputer(strategy=value['Imputer']).fit_transform(var)
                                    
                                    scaler = StandardScaler()
                                    var_ = scaler.fit_transform(imputer).reshape(-1,)
                                
                                else:
                                    
                                    var_ = tab[sheet][tab[sheet].columns[item]]
                                    
                                df_variables[f'{tab[sheet].columns[item]}'] = var_
                

                
                DF.append(df_variables)
                print('Dataframe used for analyses')
                print('------------------')
                print(df_variables)
                 
                analysis_window(df_variables)
             
        except:
            pass
    window.close()




if __name__ == '__main__':
    
    import PySimpleGUI as sg
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    from matplotlib import cm, colors
    from matplotlib.patches import Ellipse
    import matplotlib.transforms as transforms
    from mpl_toolkits import mplot3d
    from scipy.cluster.hierarchy import linkage, dendrogram, set_link_color_palette
    from scipy.cluster.vq import kmeans
    from scipy.spatial.distance import cdist, pdist
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.impute import SimpleImputer
    from sklearn.cluster import KMeans, AgglomerativeClustering
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold, GridSearchCV, train_test_split, learning_curve
    from sklearn.ensemble import IsolationForest
    from sklearn.metrics import confusion_matrix

    sg.theme('DarkBlue')
        
    main_layout = [[sg.Text('File path')],
                   [sg.InputText(size=(35,1)), sg.FileBrowse()],
                   [sg.Button('GO')]]
              
    main_window = sg.Window('ClusterGUI', main_layout, location=(0,0))
    
    while True:
       main_event, main_value = main_window.read()
       try:
           if main_event in (None, 'Close'):
               plt.close('all')
               break
           
           if main_event == 'GO':
               Dataset = {}
               
               file_sheets = pd.ExcelFile(str(main_value[0])).sheet_names
               for i in file_sheets:
                   data = pd.read_excel(f'{main_value[0]}', sheet_name=i, header=0)
                   Dataset[f'{i}'] = data

               variables(Dataset)
               
       except:
           pass
    main_window.close()