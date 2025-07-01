# -*- coding: utf-8 -*-
"""
Créé le Samedi 20 Novembre 2021

@author: Theo.ROSSI

Programme d'analyse de traces de fluorescence avec calcul du ΔF/F et interface graphique
pour la sélection des épisodes à analyser.
MODIFIÉ: Ajout de la correction de baseline pour tous les événements individuels
"""


def baseline_correction(dataframe, baseline_window=None, method='mean'):
    """
    Applique une correction de baseline à tous les événements individuels
    
    Paramètres:
    -----------
    dataframe : DataFrame
        DataFrame contenant les données avec colonnes de temps et d'événements
    baseline_window : tuple ou None
        Fenêtre de temps (début, fin) en secondes pour calculer la baseline
        Si None, utilise les 10% premiers points
    method : str
        Méthode de calcul de baseline ('mean' ou 'median')
    
    Retourne:
    ---------
    DataFrame avec les données corrigées
    """
    
    df_corrected = dataframe.copy()
    time = df_corrected['Time']
    
    # Définition de la fenêtre de baseline
    if baseline_window is None:
        # Utilise les 10% premiers points par défaut
        baseline_end_idx = int(len(time) * 0.1)
        baseline_indices = range(0, baseline_end_idx)
    else:
        # Utilise la fenêtre de temps spécifiée
        baseline_start, baseline_end = baseline_window
        baseline_indices = np.where((time >= baseline_start) & (time <= baseline_end))[0]
    
    print(f"Correction de baseline: utilisation de {len(baseline_indices)} points")
    print(f"Fenêtre de baseline: {time.iloc[baseline_indices[0]]:.2f} - {time.iloc[baseline_indices[-1]]:.2f} sec")
    
    # Application de la correction pour chaque colonne (sauf Time)
    for col in df_corrected.columns:
        if col != 'Time':
            # Calcul de la baseline
            if method == 'mean':
                baseline_value = np.mean(df_corrected[col].iloc[baseline_indices])
            elif method == 'median':
                baseline_value = np.median(df_corrected[col].iloc[baseline_indices])
            
            # Soustraction de la baseline
            df_corrected[col] = df_corrected[col] - baseline_value
            
            print(f"Épisode {col}: baseline = {baseline_value:.4f}")
    
    return df_corrected


def visualization(name, dataframe, input_file_path, baseline_corrected=False):
    """
    Fonction de visualisation et d'exportation des données ΔF/F déjà normalisées
    
    Paramètres:
    -----------
    name : str
        Nom du fichier pour la sauvegarde
    dataframe : DataFrame
        DataFrame contenant les données ΔF/F normalisées avec colonnes de temps
    input_file_path : str
        Chemin complet du fichier d'entrée pour déterminer le dossier de sortie
    baseline_corrected : bool
        Indique si une correction de baseline a été appliquée
    """
    
    # Extraction du temps et des épisodes (données non smoothées, non normalisées.)
    time = dataframe['Time']
    df_f_episodes = dataframe.drop('Time', axis=1)  # Supprime la colonne temps
    df_f_average = np.mean(df_f_episodes, axis=1)   # Calcule la moyenne de tous les épisodes
    
    # Smoothing des données.
    df_f_episodes = savgol_filter(df_f_episodes, 9, 2)
    df_f_average = savgol_filter(df_f_average, 9, 2)
    
    # Normalisation DF.
    
    
    # Création des graphiques (3 graphiques si correction de baseline)
    if baseline_corrected:
        fig, ax = plt.subplots(1, 3, figsize=(18, 4), tight_layout=True)
        
        # Graphique 1: Traces ΔF/F individuelles avec moyenne
        ax[0].plot(time, df_f_episodes, 'k', alpha=0.3)  # Traces individuelles en noir transparent
        ax[0].plot(time, df_f_average, 'r', linewidth=2, label='Moyenne')  # Trace moyenne en rouge
        ax[0].axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='Baseline')
        ax[0].set_xlabel('Time (sec)')
        ax[0].set_ylabel('ΔF/F (corrigé)')
        ax[0].legend()
        ax[0].set_title('Traces ΔF/F avec correction de baseline')
        
        # Graphique 2: Moyenne ΔF/F avec filtre de lissage Savitzky-Golay
        smoothed_average = savgol_filter(df_f_average, 9, 2)  # Filtre avec fenêtre=9, ordre=2
        ax[1].plot(time, smoothed_average, 'b', linewidth=2)
        ax[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='Baseline')
        ax[1].set_xlabel('Time (sec)')
        ax[1].set_ylabel('ΔF/F (lissé, corrigé)')
        ax[1].set_title('Moyenne ΔF/F lissée avec baseline')
        ax[1].legend()
        
        # Graphique 3: Comparaison des baselines avant/après correction
        baseline_values = []
        for col in df_f_episodes.columns:
            # Calcul de la baseline sur les 10% premiers points
            baseline_end_idx = int(len(time) * 0.1)
            baseline_val = np.mean(df_f_episodes[col].iloc[:baseline_end_idx])
            baseline_values.append(baseline_val)
        
        ax[2].bar(range(len(baseline_values)), baseline_values, alpha=0.7)
        ax[2].axhline(y=0, color='r', linestyle='--', label='Baseline cible')
        ax[2].set_xlabel('Numéro d\'épisode')
        ax[2].set_ylabel('Valeur de baseline')
        ax[2].set_title('Baselines après correction')
        ax[2].legend()
        
    else:
        fig, ax = plt.subplots(1, 2, figsize=(12, 4), tight_layout=True)
        
        # Graphique 1: Traces ΔF/F individuelles avec moyenne
        ax[0].plot(time, df_f_episodes, 'k', alpha=0.3)  # Traces individuelles en noir transparent
        ax[0].plot(time, df_f_average, 'r', linewidth=2, label='Moyenne')  # Trace moyenne en rouge
        ax[0].set_xlabel('Time (sec)')
        ax[0].set_ylabel('ΔF/F')
        ax[0].legend()
        ax[0].set_title('Traces ΔF/F brutes')
        
        # Graphique 2: Moyenne ΔF/F avec filtre de lissage Savitzky-Golay
        smoothed_average = savgol_filter(df_f_average, 9, 2)  # Filtre avec fenêtre=9, ordre=2
        ax[1].plot(time, smoothed_average, 'b', linewidth=2)
        ax[1].set_xlabel('Time (sec)')
        ax[1].set_ylabel('ΔF/F (lissé)')
        ax[1].set_title('Moyenne ΔF/F lissée')
    
    # Détermination des chemins de sortie
    input_dir = os.path.dirname(input_file_path)  # Dossier du fichier d'entrée
    suffix = "_baseline_corrected" if baseline_corrected else "_selected"
    output_name = f"{name}{suffix}"  # Nom de sortie avec suffixe
    
    # Création du dossier pour les figures
    figures_dir = os.path.join(input_dir, output_name)
    os.makedirs(figures_dir, exist_ok=True)
    
    # Sauvegarde de la figure
    fig.savefig(os.path.join(figures_dir, f"{output_name}_visualization.png"), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(figures_dir, f"{output_name}_visualization.pdf"), bbox_inches='tight')
        
    # Préparation des DataFrames pour l'exportation Excel
    # Feuille 1: Traces ΔF/F lissées avec moyenne lissée et temps
    df_smoothed = df_f_episodes.copy()
    for col in df_smoothed.columns:
        df_smoothed[col] = savgol_filter(df_smoothed[col], 9, 2)
    df_smoothed = pd.concat((df_smoothed, pd.DataFrame(smoothed_average, columns=['Average']), time), axis=1)
    

    # Feuille 2: Traces ΔF/F brutes avec moyenne et temps
    df_raw = pd.concat((df_f_episodes, pd.DataFrame(df_f_average, columns=['Average']), time), axis=1)
    
    # Création du fichier Excel avec 2 feuilles
    wb = Workbook()
    
    # Création des feuilles Excel
    sheet_name_1 = 'DF_F_Smoothed_Corrected' if baseline_corrected else 'DF_F_Smoothed'
    sheet_name_2 = 'DF_F_Baseline_Corrected' if baseline_corrected else 'DF_F_Raw'

    sheet1 = wb.create_sheet(sheet_name_1)      # ΔF/F lissées (ou corrigées)
    sheet2 = wb.create_sheet(sheet_name_2)      # ΔF/F brutes (ou corrigées)

        
    # Écriture des données dans chaque feuille
    [sheet1.append(i) for i in dataframe_to_rows(df_smoothed, index=False, header=True)]
    [sheet2.append(i) for i in dataframe_to_rows(df_raw, index=False, header=True)]


    print('----------------------')
    print(f'Données exportées: {len(df_f_episodes.columns)} épisodes analysés')
    if baseline_corrected:
        print('Correction de baseline appliquée')

    # Suppression de la feuille par défaut créée automatiquement
    wb.remove(wb['Sheet'])

    # Sauvegarde du fichier Excel dans le même dossier que le fichier d'entrée
    excel_path = os.path.join(input_dir, f"{output_name}.xlsx")
    wb.save(excel_path)
    
    print(f'Fichier Excel sauvegardé: {excel_path}')
    print(f'Figures sauvegardées dans: {figures_dir}')
    
    plt.show()  # Affichage des graphiques


def traces_selection(name, tab, input_file_path):
    """
    Interface graphique pour la sélection des épisodes à analyser
    
    Paramètres:
    -----------
    name : str
        Nom du fichier
    tab : dict
        Dictionnaire avec les noms des feuilles Excel comme clés et les DataFrames comme valeurs
    input_file_path : str
        Chemin complet du fichier d'entrée
    
    Retourne:
    ---------
    df_variables : DataFrame
        DataFrame des variables sélectionnées (forme: n_échantillons, n_variables)
    """
    
    # ÉTAPE 1: APPLICATION DE LA CORRECTION DE BASELINE DÈS LE DÉBUT
    # Interface pour définir les paramètres de baseline
    sg.theme('DarkBlue')
    
    baseline_layout = [
        [sg.Text('PARAMÈTRES DE CORRECTION DE BASELINE', font=('Arial', 12, 'bold'))],
        [sg.Checkbox('Appliquer correction de baseline', key='apply_baseline', default=True)],
        [sg.Text('Fenêtre baseline (sec):'), sg.InputText(default_text='0', size=(5,1), key='baseline_start'), 
         sg.Text('à'), sg.InputText(default_text='0.8', size=(5,1), key='baseline_end')],
        [sg.Text('Méthode:'), sg.Combo(['mean', 'median'], default_value='mean', key='baseline_method', readonly=True)],
        [sg.Button('APPLIQUER BASELINE'), sg.Button('PASSER SANS CORRECTION')]
    ]
    
    baseline_window = sg.Window('Configuration Baseline', baseline_layout, location=(100,100))
    
    # Variables pour stocker les données avec/sans correction
    corrected_data = None
    baseline_applied = False
    
    while True:
        baseline_event, baseline_value = baseline_window.read()
        
        if baseline_event in (None, 'Close'):
            baseline_window.close()
            return None
            
        if baseline_event == 'PASSER SANS CORRECTION':
            corrected_data = tab['Traces'].copy()
            baseline_applied = False
            break
            
        if baseline_event == 'APPLIQUER BASELINE':
            if baseline_value['apply_baseline']:
                print('Application de la correction de baseline...')
                
                # Définition de la fenêtre de baseline
                baseline_start = float(baseline_value['baseline_start'])
                if baseline_value['baseline_end'] == 'auto':
                    baseline_window_range = None  # Utilise les 10% premiers points
                else:
                    baseline_end = float(baseline_value['baseline_end'])
                    baseline_window_range = (baseline_start, baseline_end)
                
                baseline_method = baseline_value['baseline_method']
                
                # Application de la correction
                corrected_data = baseline_correction(tab['Traces'], baseline_window_range, baseline_method)
                baseline_applied = True
                
                print('Correction de baseline terminée !')
            else:
                corrected_data = tab['Traces'].copy()
                baseline_applied = False
            break
    
    baseline_window.close()
    
    # Mise à jour des données dans le dictionnaire tab
    tab['Traces'] = corrected_data
    
    # Initialisation du dictionnaire pour les cases à cocher
    Checkboxes = {}
    Checkboxes['Traces'] = []
    
    # Création des cases à cocher pour chaque épisode (colonne) sauf 'Time'
    for i in range(len(tab['Traces'].columns)):
        if tab['Traces'].columns[i] == 'Time':
            pass  # Ignore la colonne temps
        else:
            # Crée une case à cocher pour chaque épisode
            Checkboxes['Traces'].append([sg.Checkbox('Ep {}'.format(tab['Traces'].columns[i]), key='{}'.format(tab['Traces'].columns[i]))])
    
    # Configuration du thème de l'interface
    sg.theme('DarkBlue')
    
    # Définition du layout de l'interface graphique (simplifié, sans options de baseline)
    baseline_status = "✓ Baseline corrigée" if baseline_applied else "○ Données originales"
    
    tab_layout = [
        # Information sur le statut de la baseline
        [sg.Text(f'Statut: {baseline_status}', font=('Arial', 10, 'bold'))],
        
        # Ligne 1: Seuil pour la détection d'outliers et bouton de vérification
        [sg.InputText(default_text='1', size=(2,1), key='threshold'), sg.Text('Threshold'), sg.Button('Check_traces')],
        
        # Ligne 2: Frame avec les cases à cocher pour la sélection des épisodes
        [sg.Frame(layout=[
            [sg.Checkbox('ALL', key='all_traces')],  # Case "Tout sélectionner"
            [sg.TabGroup([[sg.Tab(sheet, checkbox_list) for sheet, checkbox_list in Checkboxes.items()]])]], 
            title='EPISODES SELECTION', relief=sg.RELIEF_SUNKEN)],
        # Ligne 3: Bouton d'analyse
        [sg.Button('ANALYSE')]
    ]
    
    # Création de la fenêtre principale
    window = sg.Window('VARIABLES').Layout([[sg.Column(tab_layout, size=(300,900), scrollable=True)]])
    
    # Boucle principale de l'interface
    while True:
        
        event, value = window.read()  # Lecture des événements
        
        try:
            # Gestion de la fermeture de la fenêtre
            if event in (None, 'Close'):
                break
                
            # Événement: Vérification des traces (détection d'outliers)
            if event == 'Check_traces':
                
                df_variables = pd.DataFrame(index=None)
                
                # Extraction du temps et des traces (DÉJÀ CORRIGÉES pour la baseline si applicable)
                time = tab['Traces']['Time']
                traces_only = tab['Traces'].drop('Time', axis=1)
                
                # Les données sont déjà normalisées ET potentiellement corrigées pour la baseline
                for col in range(len(traces_only.columns)):
                    df_variables[f'{traces_only.columns[col]}'] = traces_only.iloc[:,col]
                
                
                DF_start = np.ravel(np.where(time >= 0.38))[0]
                DF_stop = np.ravel(np.where(time <= 0.48))[-1]
                
                # Calcul des statistiques sur les données CORRIGÉES
                trace_means = [np.mean(df_variables.iloc[:,i][DF_start : DF_stop]) for i in range(len(df_variables.columns))]
                overall_mean = np.mean(trace_means)  # Moyenne globale
                overall_std = np.std(trace_means)    # Écart-type global
                
                print(f"Statistiques sur les données {'corrigées' if baseline_applied else 'originales'}:")
                print(f"Moyenne globale des moyennes: {overall_mean:.4f}")
                print(f"Écart-type des moyennes: {overall_std:.4f}")
            
                # Création des graphiques de vérification
                fig, ax = plt.subplots(1,3,figsize=(14,4),tight_layout=True)
                
                # Graphique 1: Toutes les traces ΔF/F
                [ax[0].plot(time, df_variables.iloc[:,i], alpha = 0.3, label=f'Ep{i}') for i in range(len(traces_only.columns))]
                if baseline_applied:
                    ax[0].axhline(y=0, color='gray', linestyle='--', alpha=0.7, label='Baseline')
                ax[0].legend()
                ylabel = 'ΔF/F (corrigé)' if baseline_applied else 'ΔF/F'
                ax[0].set_ylabel(ylabel)
                ax[0].set_xlabel('Time (sec)')
                title = 'Traces ΔF/F avec baseline corrigée' if baseline_applied else 'Traces ΔF/F'
                ax[0].set_title(title)
                
                # Graphique 2: Distribution des moyennes de traces (violin plot)
                sns.violinplot(data=trace_means, ax=ax[1])
                ax[1].set_ylabel('Moyenne ΔF/F par trace')
                title = 'Distribution des moyennes (baseline corrigée)' if baseline_applied else 'Distribution des moyennes'
                ax[1].set_title(title)
                
                # Détection des outliers basée sur le z-score des moyennes (calculé sur données corrigées)
                threshold = int(value['threshold'])  # Seuil défini par l'utilisateur
                for i in range(len(trace_means)):
                    z_score = (trace_means[i] - overall_mean)/overall_std if overall_std > 0 else 0  # Calcul du z-score
                    print(f"Trace {i}: moyenne={trace_means[i]:.4f}, z-score={z_score:.3f}")
                    
                    # Classification: outlier si |z-score| > threshold
                    if abs(z_score) > threshold:
                        # Outlier: marquage en rouge
                        ax[1].scatter(0, trace_means[i], color = 'r', s=50)
                        ax[1].text(0.1, trace_means[i], f'Ep{i}', color = 'r', fontsize = 12)
                        ax[2].scatter(i+1, z_score, color='r', s=50)
                        ax[2].text(i+1, z_score+0.1, f'Ep{i}', color='r', fontsize=10)
                    else:
                        # Normal: marquage en noir
                        ax[1].scatter(0, trace_means[i], color = 'k', s=50)
                        ax[1].text(0.1, trace_means[i], f'Ep{i}', color = 'k', fontsize = 12)
                        ax[2].scatter(i+1, z_score, color='k', s=50)
                
                print('-----------')
                
                # Ajout des lignes de référence sur les graphiques
                # Graphique 2: Lignes horizontales pour la moyenne et les seuils
                ax[1].axhline(y=overall_mean, color='k', ls='--', label='Moyenne')
                if overall_std > 0:
                    ax[1].axhline(y=overall_mean+(threshold*overall_std), color='r', ls='--', label=f'Seuil ±{threshold}σ')
                    ax[1].axhline(y=overall_mean-(threshold*overall_std), color='r', ls='--')
                ax[1].legend()
                
                # Graphique 3: Z-scores avec seuils
                ax[2].axhline(y=0, color='k', ls='--', label='Moyenne')
                ax[2].axhline(y=threshold, color='r', ls='--', label=f'Seuil ±{threshold}')
                ax[2].axhline(y=-threshold, color='r', ls='--')
                ax[2].set_xlabel('Numéro de trace')
                ax[2].set_ylabel('Z-score')
                title = 'Z-scores (baseline corrigée)' if baseline_applied else 'Z-scores des moyennes'
                ax[2].set_title(title)
                ax[2].legend()
                
                # Détermination des chemins de sortie pour sauvegarder les figures de vérification
                input_dir = os.path.dirname(input_file_path)
                suffix = "_baseline_corrected" if baseline_applied else "_selected"
                output_name = f"{name}{suffix}"
                figures_dir = os.path.join(input_dir, output_name)
                os.makedirs(figures_dir, exist_ok=True)
                
                plt.tight_layout()
                # Sauvegarde de la figure de vérification
                fig.savefig(os.path.join(figures_dir, f"{output_name}_outlier_check.png"), dpi=300, bbox_inches='tight')
                fig.savefig(os.path.join(figures_dir, f"{output_name}_outlier_check.pdf"), bbox_inches='tight')
                plt.show()
                
                
            # Événement: Analyse des traces sélectionnées
            if event == 'ANALYSE':
                
                df_variables = pd.DataFrame(index=None)
                
                # Si "ALL" est sélectionné, prendre tous les épisodes
                if value['all_traces'] == True:
                    
                    ep = tab['Traces'].drop('Time', axis=1)
                    
                    # Les données sont déjà normalisées ET potentiellement corrigées
                    for col in range(len(ep.columns)):
                        df_variables[f'{ep.columns[col]}'] = ep.iloc[:,col]
                
                # Sinon, traiter les épisodes sélectionnés individuellement
                for sheet, checkbox_list in Checkboxes.items():
                    
                    for item in range(len(checkbox_list)):
                        
                        # Si la case est cochée, ajouter cet épisode (déjà traité)
                        if value[f'{tab[sheet].columns[item]}'] == True:
                            df_variables[f'{tab[sheet].columns[item]}'] = tab['Traces'].iloc[:,item]                        
                            
                # Ajout de la colonne temps
                df_variables['Time'] = tab['Traces']['Time']
                
                # Affichage des informations
                if baseline_applied:
                    print('Données ΔF/F sélectionnées (baseline corrigée)')
                else:
                    print('Données ΔF/F sélectionnées (données originales)')
                print('---------------------------')
                print(df_variables)
              
                # Appel de la fonction de visualisation
                visualization(name, df_variables, input_file_path, baseline_applied)
             
        except Exception as e:
            print(f"Erreur: {e}")
            pass  # Ignore les erreurs pour éviter les crashes
    
    # Fermeture de la fenêtre
    window.close()
    
    return(df_variables)




# Point d'entrée principal du programme
if __name__ == '__main__':
    
    # Importation des bibliothèques nécessaires
    import PySimpleGUI as sg        # Interface graphique
    import pandas as pd             # Manipulation de données
    import numpy as np              # Calculs numériques
    import seaborn as sns           # Visualisation statistique
    import matplotlib.pyplot as plt # Graphiques
    import os                       # Manipulation des chemins de fichiers
    from scipy.signal import savgol_filter    # Filtrage de signal
    from openpyxl import Workbook              # Création de fichiers Excel
    from openpyxl.utils.dataframe import dataframe_to_rows  # Conversion DataFrame vers Excel

    # Configuration du thème de l'interface
    sg.theme('DarkBlue')
        
    # Layout de la fenêtre principale
    main_layout = [
        [sg.Text('File path')],                           # Texte d'instruction
        [sg.InputText(size=(35,1)), sg.FileBrowse()],    # Champ de saisie + bouton parcourir
        [sg.Button('GO')]                                 # Bouton de lancement
    ]
              
    # Création de la fenêtre principale
    main_window = sg.Window('DF_F_conversion', main_layout, location=(0,0))
                   
    # Boucle principale du programme
    while True:
       main_event, main_value = main_window.read()
       
       try:
           # Gestion de la fermeture
           if main_event in (None, 'Close'):
               plt.close('all')  # Ferme tous les graphiques ouverts
               break
           
           # Événement: Bouton GO pressé
           if main_event == 'GO':
               # Extraction du nom de fichier sans extension
               name = main_value[0].rsplit('/', 1)[-1].rsplit('.', 1)[0]
               
               # Lecture du fichier Excel (données ΔF/F déjà normalisées)
               # Structure attendue: 10 colonnes d'événements individuels + 1 colonne moyenne + 1 colonne temps
               df_data = pd.read_excel(f'{main_value[0]}', header=0)
               
               print(f"Fichier chargé: {name}")
               print(f"Dimensions des données: {df_data.shape}")
               print(f"Colonnes disponibles: {list(df_data.columns)}")
               
               # Initialisation du dictionnaire de données pour compatibilité avec traces_selection
               Dataset = {}
               Dataset['Traces'] = df_data
               Dataset['Fback'] = pd.Series([0] * (len(df_data.columns)-1))  # Pas de soustraction de fond nécessaire
               
               # Lancement de l'interface de sélection des traces avec le chemin du fichier
               traces_selection(name, Dataset, main_value[0])
               
       except Exception as e:
           print(f"Erreur lors du chargement: {e}")
           pass  # Ignore les erreurs pour éviter les crashes
    
    # Fermeture de la fenêtre principale
    main_window.close()