# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 15:14:52 2021

@author: Theo.ROSSI
"""


import matplotlib.pyplot as plt
from matplotlib import cm, colors
import seaborn as sns
import pandas as pd
import numpy as np
import os
from scipy.signal import savgol_filter
from scipy.stats import binned_statistic
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from collections import Counter


Path = r'E:\AAVDJ.GluSnFR-S72A\Boutons_analysis'
file_amplitudes = r'E:\AAVDJ.GluSnFR-S72A\GluSnFR_avg_clustering_variables_all_profiles_unsupervised.xlsx'
save_folder = r'\\equipe2-nas1\Theo.ROSSI\Paper_thesis\Figure1'

Calcium = '2.5mM'
Freq = '20Hz'
save = False

check_single_fibre = '20210303_linescan3'





def palette_colors(num):
    clr = []
    cmap = cm.magma_r(np.linspace(0.2,0.85,num))
    for c in range(num):
        rgba = cmap[c]
        clr.append(colors.rgb2hex(rgba))
    return clr


files = sorted(os.listdir(Path))
file_amplitudes = pd.read_excel(file_amplitudes, sheet_name=f'{Freq}')
amplitudes = file_amplitudes.iloc[:, 1:11]


ROI_NAMES, ROI_IMAGES = [],[]
TRACE_NAMES, TRACE, TIME = [],[],[]
TIME_SINGLE_FIBRE, TRACES_SINGLE_FIBRE = [],[]
ALL_AMP1, ALL_AMP2 = [],[]
AMP1, AMP2 = [],[]
FAIL1, FAIL2 = [],[]
PPR_SINGLE_FIBRE = []
F0 = []

for file in tqdm(range(len(files))):
    if Calcium in files[file]:
        if Freq in files[file]:
            if 'ROI.tif' in files[file]:
                ROI_NAMES.append(files[file])
                roi = plt.imread(f'{Path}/{files[file]}')[:,:,0]
                ROI_IMAGES.append(roi)

                
            elif 'traces_converted.xlsx' in files[file]:
                TRACE_NAMES.append(files[file])
                data = pd.read_excel(f'{Path}/{files[file]}', sheet_name='Traces DF_F0')
                mean_trace = data['Average']
                time = data['Time']
                TRACE.append(mean_trace)
                TIME.append(time)
                
                f0_data = pd.read_excel(f'{Path}/{files[file]}', sheet_name='F0')
                f0 = f0_data.iloc[0,-1]
                F0.append(f0)
                
                if check_single_fibre in files[file]:
                    TRACES_SINGLE_FIBRE.append(mean_trace)
                    TIME_SINGLE_FIBRE.append(time)
                    
            
            
            elif 'Amp.xlsx' in files[file]:
                amps = pd.read_excel(f'{Path}/{files[file]}')
                amps = amps.iloc[0,1:]
                ppr = [amps.iloc[i]/amps.iloc[0] for i in range(len(amps))]
                ALL_AMP1.append(amps[0])
                ALL_AMP2.append(amps[1])
                if check_single_fibre in files[file]:
                    AMP1.append(amps[0])
                    AMP2.append(amps[1])
                    PPR_SINGLE_FIBRE.append(ppr)
            
            elif 'data_bootstrap.xlsx' in files[file]:
                bootstrap = [pd.read_excel(f'{Path}/{files[file]}', sheet_name=f'PEAK{i+1}') for i in range(3)]
                if check_single_fibre in files[file]:
                    FAIL1.append(bootstrap[0]['PercFail'][0])
                    FAIL2.append(bootstrap[1]['PercFail'][0])
                    
                    
fig_F0, ax_F0 = plt.subplots(1,2,figsize=(12,4),tight_layout=True)
df = pd.concat((pd.DataFrame(F0), pd.DataFrame(ALL_AMP2)), axis=1)
df.columns = ['F0', 'AMP2']

mean = np.mean(df['F0'])
std = np.std(df['F0'])
sns.histplot(data=df['F0'], binwidth=2, stat='proportion', ax=ax_F0[0], color='darkgreen')
ax_F0[0].set_title('µ = {:.3f} ; std = {:.3f}'.format(mean, std))


sns.regplot(x='F0', y='AMP2', data=df, ci=95, ax=ax_F0[1], color='darkgreen', scatter_kws={'alpha':0.2}, line_kws={'color':'g'})
ax_F0[1].set_ylim(0)
    
    
ax_F0[1].errorbar(np.mean(F0), np.mean(ALL_AMP2), xerr=np.std(F0), yerr=np.std(ALL_AMP2), fmt='o', color='r', label='AMP1')
ax_F0[1].set_ylim(0)

x = np.array(F0).reshape(-1,1)
y = np.array(ALL_AMP2).reshape(-1,1)
reg = LinearRegression().fit(x,y)
r2 = reg.score(x,y)
ax_F0[1].set_title('R2 = {:.3f}'.format(r2))


# bins=15
# s_mean, edges_mean, binnumber = binned_statistic(df['F0'], df['AMP2'], statistic='mean', bins=bins)
# s_std, edges_std, binnumber = binned_statistic(df['F0'], df['AMP2'], statistic='std', bins=bins)
# cv = s_std/s_mean
# ax_F0[2].hlines(cv,edges_mean[:-1],edges_mean[1:], color="crimson", )
# [ax_F0[2].axvline(e, color="grey", linestyle="--") for e in edges_mean]
# ax_F0[2].scatter(edges_mean[:-1]+np.diff(edges_mean)/2, cv, c="limegreen", zorder=3, label='CV')
# ax_F0[2].legend(loc='upper right')

                

fig_single_fibre, ax_single_fibre = plt.subplots(1, 5, figsize=(12, 4), tight_layout=True)
for i in range(len(PPR_SINGLE_FIBRE)):
    clr = palette_colors(len(PPR_SINGLE_FIBRE))[i]
    ax_single_fibre[0].plot(TIME_SINGLE_FIBRE[i], savgol_filter(TRACES_SINGLE_FIBRE[i], 9, 2), color=clr, alpha=0.5)
    # ax_single_fibre[1].plot(np.arange(1,11), PPR_SINGLE_FIBRE[i], color=clr, marker='o')
    ax_single_fibre[2].scatter([0,1], [AMP1[i], AMP2[i]], color=palette_colors(len(PPR_SINGLE_FIBRE))[i])
    ax_single_fibre[3].scatter([0,1], [FAIL1[i], FAIL2[i]], color=palette_colors(len(PPR_SINGLE_FIBRE))[i])
    ax_single_fibre[4].scatter(1, PPR_SINGLE_FIBRE[i][1], color=palette_colors(len(PPR_SINGLE_FIBRE))[i])

# ax_single_fibre[1].plot(np.arange(1,11), np.mean([PPR_SINGLE_FIBRE[0], PPR_SINGLE_FIBRE[2], PPR_SINGLE_FIBRE[4]], axis=0), marker='o')
# ax_single_fibre[1].plot(np.arange(1,11), np.mean([PPR_SINGLE_FIBRE[3], PPR_SINGLE_FIBRE[5]], axis=0), marker='o')
# ax_single_fibre[1].plot(np.arange(1,11), PPR_SINGLE_FIBRE[1], marker='o')
[ax_single_fibre[1].plot(np.arange(1,11), PPR_SINGLE_FIBRE[i], marker='o', label=f'{i}') for i in range(len(PPR_SINGLE_FIBRE))]

ax_single_fibre[0].set_ylabel('DF/F')
ax_single_fibre[0].set_xlabel('Time (sec)')
ax_single_fibre[1].axhline(y=1, color='k', ls='--')
ax_single_fibre[1].set_ylabel('An/A1')
ax_single_fibre[1].set_xlabel('Stim number')
ax_single_fibre[1].set_ylim(0)
ax_single_fibre[2].set_ylim(0)
ax_single_fibre[3].set_ylim(0,100)
ax_single_fibre[4].set_ylim(0.5)
ax_single_fibre[1].legend()


##########################################
######## POLAR PLOT SINGLE FIBER #########
##########################################

def _invert(x, limits):
    """inverts a value x on a scale from
    limits[0] to limits[1]"""
    return limits[1] - (x - limits[0])

def _scale_data(data, ranges):
    """scales data[1:] to ranges[0],
    inverts if the scale is reversed"""
    for d, (y1, y2) in zip(data[1:], ranges[1:]):
        assert (y1 <= d <= y2) or (y2 <= d <= y1)
    x1, x2 = ranges[0]
    d = data[0]
    if x1 > x2:
        d = _invert(d, (x1, x2))
        x1, x2 = x2, x1
    sdata = [d]
    for d, (y1, y2) in zip(data[1:], ranges[1:]):
        if y1 > y2:
            d = _invert(d, (y1, y2))
            y1, y2 = y2, y1
        sdata.append((d-y1) / (y2-y1) 
                     * (x2 - x1) + x1)
    return sdata

class ComplexRadar():
    def __init__(self, fig, variables, ranges,
                 n_ordinate_levels=6):
        angles = np.arange(0, 360, 360./len(variables))

        axes = [fig.add_axes([0.1,0.1,0.8,0.8],polar=True,
                label = "axes{}".format(i)) 
                for i in range(len(variables))]
        l, text = axes[0].set_thetagrids(angles, 
                                         labels=variables)
        [txt.set_rotation(angle-90) for txt, angle 
             in zip(text, angles)]
        for ax in axes[1:]:
            ax.patch.set_visible(False)
            ax.grid("off")
            ax.xaxis.set_visible(False)
        for i, ax in enumerate(axes):
            grid = np.linspace(*ranges[i], 
                               num=n_ordinate_levels)
            gridlabel = ["{}".format(round(x,2)) 
                         for x in grid]
            if ranges[i][0] > ranges[i][1]:
                grid = grid[::-1] # hack to invert grid
                          # gridlabels aren't reversed
            gridlabel[0] = "" # clean up origin
            ax.set_rgrids(grid, labels=gridlabel,
                         angle=angles[i])
            #ax.spines["polar"].set_visible(False)
            ax.set_ylim(*ranges[i])
        # variables for plotting
        self.angle = np.deg2rad(np.r_[angles, angles[0]])
        self.ranges = ranges
        self.ax = axes[0]
    def plot(self, data, *args, **kw):
        sdata = _scale_data(data, self.ranges)
        self.ax.plot(self.angle, np.r_[sdata, sdata[0]], *args, **kw)
    def fill(self, data, *args, **kw):
        sdata = _scale_data(data, self.ranges)
        self.ax.fill(self.angle, np.r_[sdata, sdata[0]], *args, **kw)


variables = ('A1(DF/F)', 'F1(%)', 'PPR')
ranges = [(0., 1.), (0., 100), (0., 4.)]            

fig1 = plt.figure(figsize=(6, 6))
radar = ComplexRadar(fig1, variables, ranges)

data_c1 = (np.mean([AMP1[1], AMP1[2], AMP1[4], AMP1[5]]), np.mean([FAIL1[1], FAIL1[2], FAIL1[4], FAIL1[5]]), np.mean([PPR_SINGLE_FIBRE[1], PPR_SINGLE_FIBRE[2], PPR_SINGLE_FIBRE[4], PPR_SINGLE_FIBRE[5]]))
data_c2 = (np.mean([AMP1[0], AMP1[3]]), np.mean([FAIL1[0], FAIL1[3]]), np.mean([PPR_SINGLE_FIBRE[0], PPR_SINGLE_FIBRE[3]]))
# data_c3 = (AMP1[1], FAIL1[1], PPR_SINGLE_FIBRE[1][1])
for i in [data_c1, data_c2]:
    radar.plot(i)
    radar.fill(i, alpha=0.2)

##########################################
##########################################





'''
fig_pooled_boutons, ax_pooled_boutons = plt.subplots()
df_avg_fibre = pd.DataFrame(index=None, columns=None)
NAME = []
MEAN_AMPS, MEAN_PPR = [],[]

for i in range(len(ROI_NAMES)):
    name_roi = ROI_NAMES[i].rsplit('_',4)
    
    N = []
    for num in range(len(TRACE_NAMES)):
        name_trace = TRACE_NAMES[num].rsplit('_',6)
        if name_roi[0] == name_trace[0]:
            N.append(num)
    NAME.append(name_roi[0])
            
    fig, ax = plt.subplots(2,len(N)+1, figsize=(20,10), sharey='row', tight_layout=True)
    ax[0,0].imshow(ROI_IMAGES[i], cmap='magma')
    
    TEMPUS, TRACES = [],[]
    AMPS, PPR = [],[]
    
    k=1
    for j in range(len(TRACE_NAMES)):
        name_trace = TRACE_NAMES[j].rsplit('_',6)
        name_bouton = file_amplitudes.iloc[j,0].rsplit('_',4)[0]
        
        if name_roi[0] == name_bouton:
            amps = amplitudes.iloc[j,:]
            ppr = [amplitudes.iloc[j,n]/amplitudes.iloc[j,0] for n in range(amplitudes.shape[1])]
            AMPS.append(amps)
            PPR.append(ppr)
            
        if name_roi[0] == name_trace[0]:
            TRACES.append(TRACE[j])
            TEMPUS.append(TIME[j])
            ax[1,k-1].plot(TIME[j], savgol_filter(TRACE[j], 9,2))
            ax[1,k-1].set_title(f'{name_trace[4]}')
            
            k += 1
    
    mean_amps = np.mean(AMPS, axis=0)
    mean_ppr = np.mean(PPR, axis=0)
    MEAN_AMPS.append(mean_amps)
    MEAN_PPR.append(mean_ppr)
    
    ax_pooled_boutons.plot(np.arange(1,11), mean_ppr, 'k', lw='4', alpha=0.2)
    
    avg = np.mean(TRACES, axis=0)
    df_avg_fibre[f'{name_roi[0]}_Time'] = TEMPUS[0]
    df_avg_fibre[f'{name_roi[0]}_Avg'] = pd.Series(avg)
    ax[1,-1].plot(TEMPUS[0], savgol_filter(avg, 9, 2), color='r')
    ax[1,-1].set_title('PF profile')
    ax[0,0].set_title(f'{ROI_NAMES[i]}', fontsize=10)
    
    if save == True:
        plt.savefig(f'E:\AAVDJ.GluSnFR-S72A\Boutons_analysis\Figures_mean_traces\{name_roi[0]}_{name_roi[1]}_{name_roi[2]}_{name_roi[3]}Ca.pdf')

ax_pooled_boutons.axhline(y=1.0, color='k', ls='--')
ax_pooled_boutons.set_title('Single PFs profiles')

if save == True:
    with pd.ExcelWriter(f'{save_folder}\Avg_trace_per_fibre_{Calcium}_{Freq}.xlsx') as writer:
        df_avg_fibre.to_excel(writer)




# df_mean_PF = pd.concat((pd.DataFrame(NAME),
#                         pd.DataFrame([MEAN_AMPS[i][0] for i in range(len(MEAN_AMPS))]),
#                         pd.DataFrame([MEAN_AMPS[i][1] for i in range(len(MEAN_AMPS))]),
#                         pd.DataFrame([MEAN_PPR[i][1] for i in range(len(MEAN_PPR))])), axis=1)
# df_mean_PF.columns = ['PF_ID', 'AMP1', 'AMP2', 'PPR']

# fig_boutons_vs_PF, ax_boutons_vs_PF = plt.subplots(1,2)
# sns.violinplot(data=[amplitudes['AMP1'], amplitudes['AMP2'], df_mean_PF['AMP1'], df_mean_PF['AMP2']],
#                      ax=ax_boutons_vs_PF[0], palette = palette_colors(4))

# sns.violinplot(data=[file_amplitudes['PPR2/1'], df_mean_PF['PPR']],
#                      ax=ax_boutons_vs_PF[1], palette = palette_colors(2))
'''