# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 20:12:13 2022

@author: Theo.ROSSI
"""


import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection
import numpy as np
import pandas as pd
import os
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
from matplotlib import cm, colors


Path = r'\\equipe2-nas1\Theo.ROSSI\BackupE\AAVDJ.GluSnFR-S72A\Boutons_analysis'
file_data = r'\\equipe2-nas1\Theo.ROSSI\BackupE\AAVDJ.GluSnFR-S72A\Tidy_files\GluSnFR_avg_clustering_variables_all_profiles_20Hz_forskoline_filtered_3sigma.xlsx'
# file_first_linescan = r'\\equipe2-nas1\Theo.ROSSI\Paper_thesis\Figure1\Second analysis\20210304_linescan3.tif'
linescan = '20220420_linescan1'
condition = '20Hz'
bouton = 'bouton3'
filter_value = 9




def viridis_colors(num):
    clr = []
    cmap = cm.viridis(np.linspace(0.1,0.85,num))
    for c in range(num):
        rgba = cmap[c]
        clr.append(colors.rgb2hex(rgba))
    return clr



# ls = plt.imread(file_first_linescan).transpose(1,0,2)[:,:,0]
# ls_gaussian_filtered = gaussian_filter1d(ls, sigma=1, axis=1)
# ls_savgol_filtered = savgol_filter(ls, filter_value, 2)

# fig1, ax1 = plt.subplots(3,2, tight_layout=True)
# ax1[0,0].imshow(ls, cmap='turbo')
# ax1[1,0].imshow(ls_gaussian_filtered, cmap='turbo')
# ax1[2,0].imshow(ls_savgol_filtered, cmap='turbo')

# ax1[0,1].plot(range(0, len(ls[44])), np.mean(ls[39:49], axis=0))
# ax1[1,1].plot(range(0, len(ls_gaussian_filtered[44])), np.mean(ls_gaussian_filtered[39:49], axis=0))
# ax1[2,1].plot(range(0, len(ls_savgol_filtered[44])), np.mean(ls_savgol_filtered[39:49], axis=0))

# ax1[0,0].set_title('Original')
# ax1[1,0].set_title('Gaussian_filter')
# ax1[2,0].set_title('Savgol_filter')




files = sorted(os.listdir(Path))


AVERAGES = []
for file in range(len(files)):
    if linescan in files[file]:
        if condition in files[file]:
            if 'traces_converted.xlsx' in files[file]:
                print(files[file])
                data = pd.read_excel(f'{Path}\{files[file]}', sheet_name='Traces DF_F0')
                avg = data['Average'].values
                time = data['Time'].values
                AVERAGES.append(avg)
                
                if bouton in files[file]:
                    episodes = data.drop(['Time'], axis=1)
                    move = episodes.pop("Average")
                    episodes.insert(0, "Average", move)
                    # episodes = episodes.rename(columns={"Average": f'{episodes.shape[1]-1}'})
                    episodes = episodes.transpose()



df_avg = pd.DataFrame(AVERAGES)
    

def offset(myFig,myAx,n=1,yOff=60):
    dx, dy = 0., yOff/myFig.dpi 
    return myAx.transData + mpl.transforms.ScaledTranslation(dx,n*dy,myFig.dpi_scale_trans)


def plot_2D_a():
    """ a 2D plot which uses color to indicate the angle"""
    fig,ax=plt.subplots()
    sampling=20
    thetas=range(0,df_avg.shape[0])

    cmap = mpl.cm.get_cmap('viridis')
    norm = mpl.colors.Normalize(vmin=0,vmax=df_avg.shape[0])

    for idx in thetas:
        z_ind=df_avg.shape[0]-idx ## to ensure each plot is "behind" the previous plot
        trans=offset(fig,ax,idx,yOff=sampling)
        # xs=df.loc[0]
        ys=df_avg.iloc[idx,:]
        ax.plot(time,savgol_filter(ys,filter_value,2),color='k',linewidth=2, transform=trans,zorder=z_ind)
        ax.fill_between(time,savgol_filter(ys,filter_value,2),-0.5,facecolor='w', edgecolor="None",transform=trans,zorder=z_ind)

    cbax = fig.add_axes([0.9, 0.15, 0.02, 0.7]) # x-position, y-position, x-width, y-height
    cb1 = mpl.colorbar.ColorbarBase(cbax, cmap=cmap, norm=norm, orientation='vertical')
    cb1.set_label('Angle')

    # ax.set_xlim(-0.2,2.2)
    ax.set_ylim(-0.5,5)
    ax.set_xlabel('time [ms]')
    
plot_2D_a()


fig_norm_profiles, ax_norm_profiles = plt.subplots()
data = pd.read_excel(f'{file_data}', sheet_name='20Hz_1.5mMCa')
x = np.arange(1,11)
for i in range(data.shape[0]):
    if linescan in data['ID'][i]:
        norm_profile = [data.iloc[i,j+1]/data.iloc[i,1] for j in range(len(data.iloc[i,1:11]))]
        ax_norm_profiles.plot(x, norm_profile, marker='o')
ax_norm_profiles.set_ylim(0)


  
'''
def plot_2D_b():
    """ a 2D plot which removes the y-axis and replaces it with text labels to indicate angles """
    fig,ax=plt.subplots(figsize=(5,6))
    sampling=2
    thetas=range(0,360)[::sampling]

    for idx,i in enumerate(thetas):
        z_ind=360-idx ## to ensure each plot is "behind" the previous plot
        trans=offset(fig,ax,idx,yOff=sampling)

        xs=df.loc[0]
        ys=df.loc[i+1]

        ## note that I am using both .plot() and .fill_between(.. edgecolor="None" ..) 
        #  in order to circumvent showing the "edges" of the fill_between 
        ax.plot(xs,ys,color="k",linewidth=0.5, transform=trans,zorder=z_ind)
        ax.fill_between(xs,ys,-0.5,facecolor="w", edgecolor="None",transform=trans,zorder=z_ind)

        ## for every 10th line plot, add a text denoting the angle. 
        #  There is probably a better way to do this.
        if idx%10==0:
            textTrans=mpl.transforms.blended_transform_factory(ax.transAxes, trans)
            ax.text(-0.05,0,u'{0}º'.format(i),ha="center",va="center",transform=textTrans,clip_on=False)

    ## use some sensible viewing limits
    ax.set_xlim(df.loc[0].min(),df.loc[0].max())
    ax.set_ylim(-0.5,5)

    ## turn off the spines
    for side in ["top","right","left"]:
        ax.spines[side].set_visible(False)
    ## and turn off the y axis
    ax.set_yticks([])

    ax.set_xlabel('time [ms]')

#--------------------------------------------------------------------------------
def plot_3D():
    """ a 3D plot of the data, with differently scaled axes"""
    fig=plt.figure(figsize=(5,6))
    ax= fig.gca(projection='3d')

    """                                                                                                                                                    
    adjust the axes3d scaling, taken from https://stackoverflow.com/a/30419243/565489
    """
    # OUR ONE LINER ADDED HERE:                to scale the    x, y, z   axes
    ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1, 2, 1, 1]))

    sampling=2
    thetas=range(0,episodes.shape[0])
    verts = []
    count = episodes.shape[0]

    for idx,i in enumerate(thetas):
        z_ind=episodes.shape[0]-idx

        # xs=episodes.loc[0].values
        ys=episodes.loc[i].values

        ## To have the polygons stretch to the bottom, 
        #  you either have to change the outermost ydata here, 
        #  or append one "x" pixel on each side and then run this.
        ys[0] = -0.5 
        ys[-1]= -0.5

        verts.append(list(zip(time, ys)))        

    zs=thetas

    poly = PolyCollection(verts, facecolors = "w", edgecolors="k",linewidth=0.5 )
    ax.add_collection3d(poly, zs=zs, zdir='y')

    ax.set_ylim(0,episodes.shape[0])
    ax.set_xlim(time.min(),time.max())
    ax.set_zlim(-0.5,1)

    ax.set_xlabel('time [ms]')
'''

# plot_2D_b()
# plot_3D()
