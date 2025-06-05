# -*- coding: utf-8 -*-
"""
Created on Fri Dec 28 14:25:00 2018

@author: Theo.Rossi
"""



import matplotlib.pyplot as plt
import numpy as np

#Importer le fichier via pyplot
line_scan = plt.imread(r"D:\Theo.ROSSI\Virus_AAV-DJ.hSynapto_170915\20180727_13_12_39_20180727\20180727_13_12_39_20180727_[xy]T_ch_1.tif")

#On fait une rotation de l'image de 90 degres sens anti horraire
line_scan = np.rot90(line_scan.astype(float))

#Determine la periode d'echantillonage
sampling_period = 1.0/line_scan.shape[1]#Periode echantillonage


#Normaliser le fichier par rapport au bruit 
for line in range(line_scan.shape[0]): #Pour chaque ligne (line) dans le nombre total de ligne du scan (range(line_scan.shape[0]))

    #Calcul la moyenne du bruit au début de chaque ligne
    noise = float(np.mean(line_scan[line,0:16])) #16 pts egal grosso merdo 100ms

    #Cree un nouvel array (matrice) qui contient n fois la valeur du bruit 
    noise_array = np.zeros(line_scan.shape[1])
    noise_array.fill(noise)

    #Calcul du DF/F
    line_scan[line] = (line_scan[line]-noise_array)/noise


#Maintenant les plots

#On fait une premiere figure
plt.figure()

#On affiche le linescan normalise complet
plt.imshow(line_scan,cmap='hot')
plt.show()

#On selectionne la ligne qu'on veut regarder de plus pres
select_line = 1500

#On cree un nouveau tableau, composé des 10 lignes avant la ligne d'interet + les 10 lignes apres
sub_frame = line_scan[select_line-10 : select_line+10 , :]

#Nouvelle figure
#Fig = fenetre, ax = subplot dans la fenetre, ici on en a 2
fig, ax = plt.subplots(2,1,sharex = True) #Sharex = meme abscisse pour les 2 plots
ax[0].set_title('This is the subframe')
ax[0].imshow(sub_frame,cmap='hot')

ax[1].set_title('Average signal +/-10 DF/F at line %s'%select_line)

#for line in range(sub_frame.shape[0]): #plot chacun des line scans utilses pour la moyenne
#    ax[1].plot(sub_frame[line],color='0.5',alpha=0.4)

ax[1].plot(np.mean(sub_frame,axis=0),color='blue',linewidth=2) #Moyenne des 20 line scan consecutifs
plt.show()