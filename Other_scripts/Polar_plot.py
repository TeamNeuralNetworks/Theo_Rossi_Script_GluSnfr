# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 15:59:53 2022

@author: Theo.ROSSI
"""


import matplotlib.pyplot as plt
from matplotlib import cm, colors
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats


file = r'\\equipe2-nas1\Theo.ROSSI\Paper_thesis\Figure3\Fourth_analysis\Targets distribution for each cluster.xlsx'
# file2 = r'\\equipe2-nas1\Theo.ROSSI\Paper_thesis\Figure5_PCvsMLI\Third_analysis\Targets distribution for each cluster - Copie.xlsx'
file2 = r'\\equipe2-nas1\Theo.ROSSI\Paper_thesis\Figure3\Fourth_analysis\Proportion_PF_clusters.xlsx'
# group = 'Without targets'



def palette_colors(num):
    clr = []
    cmap = cm.viridis(np.linspace(0.1,0.85,num))
    for c in range(num):
        rgba = cmap[c]
        clr.append(colors.rgb2hex(rgba))
    return clr



file = pd.read_excel(file)
labels = file.iloc[:-1,0]
num_vars = len(labels)
clr = palette_colors(num_vars)

max_IN = np.max(file.iloc[:-1,1])
max_PC = np.max(file.iloc[:-1,2])
max_value = np.max([max_IN, max_PC])

file = file.drop('Total', axis=1).iloc[:-1,1:]
# fiber_name = file.iloc[:-1,0].tolist()
# fibers = file.iloc[:-1,1:]
# C1_PC = fibers['C1_PC'].values
# C2_PC = fibers['C2_PC'].values
# C3_PC = fibers['C3_PC'].values
# C4_PC = fibers['C4_PC'].values

# C1_IN = -fibers['C1_IN'].values
# C2_IN = -fibers['C2_IN'].values
# C3_IN = -fibers['C3_IN'].values
# C4_IN = -fibers['C4_IN'].values


# plt.figure()
# plt.bar(fiber_name, C1_PC, color=clr[0], label='C1')
# plt.bar(fiber_name, C2_PC, color=clr[1], bottom=C1_PC, label='C2')
# plt.bar(fiber_name, C3_PC, color=clr[2], bottom=C1_PC+C2_PC, label='C3')
# plt.bar(fiber_name, C4_PC, color=clr[3], bottom=C1_PC+C2_PC+C3_PC, label='C4')

# plt.bar(fiber_name, C1_IN, color=clr[0], label='C1')
# plt.bar(fiber_name, C2_IN, color=clr[1], bottom=C1_IN, label='C2')
# plt.bar(fiber_name, C3_IN, color=clr[2], bottom=C1_IN+C2_IN, label='C3')
# plt.bar(fiber_name, C4_IN, color=clr[3], bottom=C1_IN+C2_PC+C3_IN, label='C4')

# C1 = fibers['C1'].values
# C2 = fibers['C2'].values
# C3 = fibers['C3'].values
# C4 = fibers['C4'].values

# plt.figure()
# plt.bar(fiber_name, C1, color=clr[0], label='C1')
# plt.bar(fiber_name, C2, color=clr[1], bottom=C1, label='C2')
# plt.bar(fiber_name, C3, color=clr[2], bottom=C1+C2, label='C3')
# plt.bar(fiber_name, C4, color=clr[3], bottom=C1+C2+C3, label='C4')
# plt.axhline(y=0.0, color='k')
# plt.xticks(rotation=45)
# plt.ylabel('Count')



#########################
###### RADAR PLOT #######
#########################

# fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

# for parallel_fiber in range(file.shape[0]):
#     if file.iloc[parallel_fiber,0] == 'Total':
#         continue
#     else:
#         values = file.iloc[parallel_fiber,1:].tolist()
#         angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        
#         VALUES, ANGLES = [],[]
#         for i,j in zip(range(len(values)), range(len(angles))):
#             if values[i] == 0:
#                 continue
#             else:
#                 VALUES.append(values[i])
#                 ANGLES.append(angles[j])
        
#         VALUES.insert(len(VALUES), 0)
#         VALUES.insert(0, 0)
#         ANGLES.insert(len(ANGLES), 0.0)
#         ANGLES.insert(0, 0.0)

#         ax.plot(ANGLES, VALUES, linewidth=1)
#         ax.fill(ANGLES, VALUES, alpha=0.1)
        
                
# ax.set_theta_offset(np.pi / 2)
# ax.set_theta_direction(-1)
# ax.set_thetagrids(np.degrees(angles), labels)
# ax.set_ylim(0, 5.0)




fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))

for target in range(file.shape[1]):
    values = file.iloc[:,target].tolist()
    print(values)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    
    VALUES, ANGLES = [],[]
    for i,j in zip(range(len(values)), range(len(angles))):
        if values[i] == 0:
            continue
        else:
            VALUES.append(values[i])
            ANGLES.append(angles[j])
    
    VALUES.insert(len(VALUES), 0)
    VALUES.insert(0, 0)
    ANGLES.insert(len(ANGLES), 0.0)
    ANGLES.insert(0, 0.0)

    ax.plot(ANGLES, VALUES, linewidth=1, label=f'{file.columns[target]}')
    ax.fill(ANGLES, VALUES, alpha=0.1)


ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_thetagrids(np.degrees(angles), labels)
ax.set_ylim(0, 45)
ax.legend()




fig2, ax2 = plt.subplots()
# data2 = pd.read_excel(file2, sheet_name = excel_sheet)
# sns.countplot(x='Cluster', hue='Target', data=data2, ax=ax2)
IN_proportion = file['IN']/np.sum(file['IN'])
PC_proportion = file['PC']/np.sum(file['PC'])
UN_proportion = file['UN']/np.sum(file['UN'])

# chi2, p, dof, expected = stats.chi2_contingency(file)

barWidth = 0.15
r1 = np.arange(len(IN_proportion)).tolist()
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]

ax2.bar(r1, IN_proportion, color='b', width=barWidth, edgecolor='white', label='IN')
ax2.bar(r2, PC_proportion, color='r', width=barWidth, edgecolor='white', label='PC')
ax2.bar(r3, UN_proportion, color='grey', width=barWidth, edgecolor='white', label='UN')

# ax2.set_title(f'p = {p}')
ax2.set_ylabel('Proportion')
ax2.set_xticks([r + barWidth for r in range(len(IN_proportion))])
ax2.set_xticklabels(['C1', 'C2', 'C3', 'C4'])
ax2.set_ylim(0,1)
ax2.legend()


fig3, ax3 = plt.subplots(tight_layout=True)
data2 = pd.read_excel(file2)
total_PF = data2.iloc[-1,1]
total_clusters = data2.iloc[-1,-1]
PF = data2.iloc[:-1,1]
PF_proportion = PF/total_PF

clusters = data2.iloc[:-2,2:-1]
clusters_proportion = [clusters[col]/total_clusters for col in clusters.columns]
ax3.bar(r1[:-1], clusters_proportion[0], color=clr[0], label=f'{clusters.columns[0]}')
ax3.bar(r1[:-1], clusters_proportion[1], color=clr[1], bottom=clusters_proportion[0], label=f'{clusters.columns[1]}')
ax3.bar(r1[:-1], clusters_proportion[2], color=clr[2], bottom=clusters_proportion[0]+clusters_proportion[1], label=f'{clusters.columns[2]}')
ax3.bar(r1[:-1], clusters_proportion[3], color=clr[3], bottom=clusters_proportion[0]+clusters_proportion[1]+clusters_proportion[2], label=f'{clusters.columns[3]}')
ax3.set_ylabel('Proportion')
ax3.set_xticks([0,1,2])
ax3.set_xticklabels(['1 Profile', '2 Profile', '3 Profile'])
ax3.set_ylim(0,1)
ax3.legend()


'''
from vega_datasets import data

# Load cars dataset so we can compare cars across
# a few dimensions in the radar plot.
df = data.cars()
df.head()

# The attributes we want to use in our radar plot.
factors = ['Acceleration', 'Displacement', 'Horsepower',
           'Miles_per_Gallon', 'Weight_in_lbs']

# New scale should be from 0 to 100.
new_max = 100
new_min = 0
new_range = new_max - new_min

# Do a linear transformation on each variable to change value
# to [0, 100].
for factor in factors:
  max_val = df[factor].max()
  min_val = df[factor].min()
  val_range = max_val - min_val
  df[factor + '_Adj'] = df[factor].apply(
      lambda x: (((x - min_val) * new_range) / val_range) + new_min)


# Add the year to the name of the car to differentiate between
# the same model.
df['Car Model'] = df.apply(lambda row: '{} {}'.format(row.Name, row.Year.year), axis=1)

# Trim down to cols we want and rename to be nicer.
dft = df.loc[:, ['Car Model', 'Acceleration_Adj', 'Displacement_Adj',
                 'Horsepower_Adj', 'Miles_per_Gallon_Adj',
                 'Weight_in_lbs_Adj']]

dft.rename(columns={
    'Acceleration_Adj': 'Acceleration',
    'Displacement_Adj': 'Displacement',
    'Horsepower_Adj': 'Horsepower',
    'Miles_per_Gallon_Adj': 'MPG',
    'Weight_in_lbs_Adj': 'Weight'
}, inplace=True)

dft.set_index('Car Model', inplace=True)

dft.head()



# Each attribute we'll plot in the radar chart.
labels = ['Acceleration', 'Displacement', 'Horsepower', 'MPG', 'Weight']

# Let's look at the 1970 Chevy Impala and plot it.
car = 'chevrolet impala 1970'
values = dft.loc[car].tolist()

# Number of variables we're plotting.
num_vars = len(labels)

# Split the circle into even parts and save the angles
# so we know where to put each axis.
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

# The plot is a circle, so we need to "complete the loop"
# and append the start value to the end.
values += values[:1]
angles += angles[:1]

# ax = plt.subplot(polar=True)
fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

# Draw the outline of our data.
ax.plot(angles, values, color='#1aaf6c', linewidth=1)
# Fill it in.
ax.fill(angles, values, color='#1aaf6c', alpha=0.25)

# Fix axis to go in the right order and start at 12 o'clock.
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)

# Draw axis lines for each angle and label.
ax.set_thetagrids(np.degrees(angles), labels)

# Go through labels and adjust alignment based on where
# it is in the circle.
for label, angle in zip(ax.get_xticklabels(), angles):
  if angle in (0, np.pi):
    label.set_horizontalalignment('center')
  elif 0 < angle < np.pi:
    label.set_horizontalalignment('left')
  else:
    label.set_horizontalalignment('right')

# Ensure radar goes from 0 to 100.
ax.set_ylim(0, 100)
# You can also set gridlines manually like this:
# ax.set_rgrids([20, 40, 60, 80, 100])

# Set position of y-labels (0-100) to be in the middle
# of the first two axes.
ax.set_rlabel_position(180 / num_vars)

# Add some custom styling.
# Change the color of the tick labels.
ax.tick_params(colors='#222222')
# Make the y-axis (0-100) labels smaller.
ax.tick_params(axis='y', labelsize=8)
# Change the color of the circular gridlines.
ax.grid(color='#AAAAAA')
# Change the color of the outermost gridline (the spine).
ax.spines['polar'].set_color('#222222')
# Change the background color inside the circle itself.
ax.set_facecolor('#FAFAFA')

# Lastly, give the chart a title and give it some
# padding above the "Acceleration" label.
ax.set_title('1970 Chevy Impala Specs', y=1.08)
'''