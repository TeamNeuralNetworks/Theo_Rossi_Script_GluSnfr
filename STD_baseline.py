import pandas as pd
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

start = 0.38 #0.8
stop = 0.48 #0.9
filt = 4

def analyze_button_std(file_path, return_traces=False):
    """
    Analyse un fichier Excel de bouton et calcule la std moyenne de la baseline.
    
    Args:
        file_path: Chemin vers le fichier Excel
        return_traces: Si True, retourne aussi les traces pour la visualisation
        
    Returns:
        float ou tuple: Std moyenne, ou (std_moyenne, traces_data, time_data) si return_traces=True
    """
    try:
        # Lire le fichier Excel
        df = pd.read_excel(file_path, sheet_name = 1)
        
        # Trouver la colonne de temps (dernière colonne)
        time_column = df.columns[-1]
        
        # Identifier les colonnes de signaux (toutes sauf "Average" et la colonne de temps)
        signal_columns = []
        for col in df.columns[:-1]:  # Exclure la colonne de temps
            if 'Average' not in str(col):
                signal_columns.append(col)
        

        # Filtrer les données pour la période de baseline (0.80s à 0.90s)
        # On prend 100ms avant 0.90s, donc de 0.80s à 0.90s pour avoir une fenêtre
        baseline_mask = (df[time_column] >= start) & (df[time_column] <= stop)
        baseline_data = df[baseline_mask]
        
        if baseline_data.empty:
            print(f"Attention: Aucune donnée trouvée dans la période de baseline pour {file_path}")
            if return_traces:
                return np.nan, None, None
            return np.nan
        
        # Calculer la std pour chaque signal dans la baseline
        std_values = []
        baseline_data = baseline_data.copy()
        
        for col in signal_columns:
            if col in baseline_data.columns:
                filtered = savgol_filter(baseline_data[col], filt, 2)
                baseline_data[col] = pd.Series(filtered, index=baseline_data.index)
                signal_std = baseline_data[col].std()
                if not np.isnan(signal_std):
                    std_values.append(signal_std)
        
        if not std_values:
            print(f"Attention: Aucune std valide calculée pour {file_path}")
            if return_traces:
                return np.nan, None, None
            return np.nan
        
        # Calculer la moyenne des std
        mean_std = np.mean(std_values)
        
        if return_traces:
            # Retourner aussi les données pour la visualisation
            traces_data = df[signal_columns]
            time_data = df[time_column]
            return mean_std, traces_data, time_data
        
        return mean_std
        
    except Exception as e:
        print(f"Erreur lors du traitement de {file_path}: {e}")
        if return_traces:
            return np.nan, None, None
        return np.nan

def process_all_buttons(input_folder, output_filename="RESULT.xlsx"):
    """
    Traite tous les fichiers Excel dans un dossier et sauvegarde les résultats.
    
    Args:
        input_folder: Dossier contenant les fichiers Excel
        output_filename: Nom du fichier de sortie (sera créé dans le même dossier)
    """
    
    # Créer le chemin du dossier d'entrée
    input_path = Path(input_folder)
    if not input_path.exists():
        print(f"Erreur: Le dossier {input_folder} n'existe pas")
        return
    
    # Créer le chemin complet du fichier de sortie dans le même dossier
    output_file = input_path / output_filename
    
    # Trouver tous les fichiers Excel
    excel_files = list(input_path.glob("*.xlsx")) + list(input_path.glob("*.xls"))
    
    if not excel_files:
        print(f"Aucun fichier Excel trouvé dans {input_folder}")
        return
    
    print(f"Traitement de {len(excel_files)} fichiers...")
    
    # Traiter chaque fichier
    results = []
    for file_path in excel_files:
        print(f"Traitement de: {file_path.name}")
        
        mean_std = analyze_button_std(file_path)
        
        results.append({
            'Fichier': file_path.name,
            'STD_Moyenne': mean_std
        })
    
    # Créer le DataFrame des résultats
    results_df = pd.DataFrame(results)
    
    # Sauvegarder dans un fichier Excel
    results_df.to_excel(output_file, index=False)
    
    print(f"\nRésultats sauvegardés dans: {output_file}")
    print(f"Nombre de fichiers traités: {len(results)}")
    
    # Afficher un aperçu des résultats
    print("\nAperçu des résultats:")
    print(results_df.head(10))
    
    return results_df

def create_visualization(input_folder, output_path):
    """
    Crée une figure montrant la trace moyenne du premier bouton avec la fenêtre de baseline.
    
    Args:
        input_folder: Dossier contenant les fichiers Excel
        output_path: Chemin de sortie pour la figure
    """
    input_path = Path(input_folder)
    excel_files = list(input_path.glob("*.xlsx")) + list(input_path.glob("*.xls"))
    
    if not excel_files:
        print("Aucun fichier Excel trouvé pour la visualisation")
        return
    
    # Analyse du premier bouton pour l'exemple
    first_file = excel_files[0]
    print(f"Analyse du premier bouton pour visualisation: {first_file.name}")
    _, first_traces, first_time = analyze_button_std(first_file, return_traces=True)
    
    if first_traces is None:
        print("Impossible de charger les données du premier bouton")
        return
    
    # Calcul de la trace moyenne du premier bouton
    first_mean_trace = first_traces.mean(axis=1)
    
    # Création de la figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    # Plot de la trace moyenne
    ax.plot(first_time, first_mean_trace, 'b-', linewidth=2, label='Trace moyenne')
    ax.axvline(x=start, color='r', linestyle='--', linewidth=2, label='Début baseline')
    ax.axvline(x=stop, color='r', linestyle='--', linewidth=2, label='Fin baseline')
    ax.axvspan(start, stop, alpha=0.2, color='red', label='Fenêtre baseline')
    ax.axvline(x=0.98, color='g', linestyle='-', linewidth=2, label='Début stimulation (légèrement avant 1 scd)') #légèrement avant une seconde car doit manquer 9 points à cause du filtre appliqué.
    
    ax.set_title('Exemple de trace moyenne d\'un bouton avec fenêtre de sélection', fontsize=14, fontweight='bold')
    ax.set_xlabel('Temps (secondes)', fontsize=12)
    ax.set_ylabel('ΔF/F₀', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    
    # Sauvegarde de la figure
    figure_path = output_path / "traces_analysis.png"
    plt.savefig(figure_path, dpi=300, bbox_inches='tight')
    print(f"Figure sauvegardée: {figure_path}")
    
    # Affichage optionnel
    plt.show()
    
    return fig

# Exemple d'utilisation
if __name__ == "__main__":
    # Paramètres à modifier selon vos besoins
    DOSSIER_ENTREE = r"C:\Anthime.PERROT\1_Thèse\1_Manip\5_Glusnf_Théo\2_Revision\All_boutons\All_Théo"  # Remplacez par votre dossier
    
    # Lancer l'analyse (le fichier RESULT.xlsx sera créé dans le même dossier)
    results = process_all_buttons(DOSSIER_ENTREE)
    
    # Créer la visualisation
    input_path = Path(DOSSIER_ENTREE)
    create_visualization(DOSSIER_ENTREE, input_path)
    
    # Statistiques supplémentaires
    if results is not None and not results.empty:
        print(f"\nStatistiques:")
        print(f"STD moyenne globale: {results['STD_Moyenne'].mean():.4f}")
        print(f"STD médiane: {results['STD_Moyenne'].median():.4f}")
        print(f"Min: {results['STD_Moyenne'].min():.4f}")
        print(f"Max: {results['STD_Moyenne'].max():.4f}")