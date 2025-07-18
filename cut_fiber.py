import pandas as pd
import numpy as np
from pathlib import Path
from scipy.interpolate import interp1d

# ==============================================
# PARAMÈTRES À MODIFIER FACILEMENT
# ==============================================
TEMPS_DEBUT = 0.4     # Temps de début (X)
TEMPS_FIN = 1.0      # Temps de fin (Y)
NB_POINTS = 1500        # Nombre de points pour l'interpolation (Z)
# ==============================================

def interpolate_trace(time_values, trace_values, start_time, end_time, num_points):
    """
    Interpole une trace entre deux temps donnés sur un nombre de points spécifié
    
    Args:
        time_values: Array des valeurs de temps
        trace_values: Array des valeurs de la trace
        start_time: Temps de début
        end_time: Temps de fin
        num_points: Nombre de points pour l'interpolation
    
    Returns:
        tuple: (temps_interpolés, valeurs_interpolées)
    """
    # Filtrer les données dans la plage de temps
    mask = (time_values >= start_time) & (time_values <= end_time)
    time_filtered = time_values[mask]
    trace_filtered = trace_values[mask]
    
    if len(time_filtered) < 2:
        print(f"Attention: Pas assez de points dans la plage [{start_time}, {end_time}]")
        return None, None
    
    # Créer la fonction d'interpolation
    try:
        interp_func = interp1d(time_filtered, trace_filtered, kind='linear', 
                              bounds_error=False, fill_value='extrapolate')
        
        # Créer les nouveaux points de temps
        new_time = np.linspace(start_time, end_time, num_points)
        
        # Interpoler les valeurs
        new_trace = interp_func(new_time)
        
        return new_time, new_trace
    
    except Exception as e:
        print(f"Erreur lors de l'interpolation: {e}")
        return None, None

def process_excel_files(folder_path, start_time=TEMPS_DEBUT, end_time=TEMPS_FIN, num_points=NB_POINTS):
    """
    Traite tous les fichiers Excel d'un dossier et extrait/interpole la colonne 'Average'
    
    Args:
        folder_path (str): Chemin vers le dossier contenant les fichiers Excel
        start_time (float): Temps de début pour l'extraction
        end_time (float): Temps de fin pour l'extraction
        num_points (int): Nombre de points pour l'interpolation
    """
    
    # Convertir en objet Path pour une manipulation plus facile
    folder = Path(folder_path)
    
    # Vérifier que le dossier existe
    if not folder.exists():
        print(f"Erreur: Le dossier '{folder_path}' n'existe pas.")
        return
    
    # Créer des DataFrames pour stocker les résultats
    result_df = pd.DataFrame()
    interpolated_df = pd.DataFrame()
    
    # Parcourir tous les fichiers Excel dans le dossier
    excel_files = list(folder.glob("*.xlsx")) + list(folder.glob("*.xls"))
    
    if not excel_files:
        print("Aucun fichier Excel trouvé dans le dossier.")
        return
    
    print(f"Traitement de {len(excel_files)} fichier(s) Excel...")
    print(f"Paramètres: Temps [{start_time} - {end_time}], {num_points} points d'interpolation")
    
    # Créer le vecteur temps interpolé (commun à toutes les traces)
    time_interpolated = np.linspace(start_time, end_time, num_points)
    interpolated_df['Time'] = time_interpolated
    
    for file_path in excel_files:
        try:
            # Lire le fichier Excel depuis la feuille 'Traces DF_F0'
            df = pd.read_excel(file_path, sheet_name='Traces DF_F0')
            
            # Vérifier si les colonnes nécessaires existent
            if 'Average' not in df.columns:
                print(f"Attention: La colonne 'Average' n'existe pas dans {file_path.name}")
                continue
            
            # Chercher la colonne de temps (plusieurs noms possibles)
            time_column = None
            possible_time_columns = ['Time', 'time', 'Temps', 'temps', 'T', 't']
            for col in possible_time_columns:
                if col in df.columns:
                    time_column = col
                    break
            
            if time_column is None:
                print(f"Attention: Aucune colonne de temps trouvée dans {file_path.name}")
                # Utiliser l'index comme temps si pas de colonne temps
                time_values = df.index.values
                print(f"  -> Utilisation de l'index comme temps")
            else:
                time_values = df[time_column].values
            
            # Extraire la colonne 'Average'
            average_column = df['Average'].values
            
            # Créer le nom de colonne en supprimant '_converted_NoFail'
            column_name = file_path.stem  # Nom du fichier sans extension
            column_name = column_name.replace('_traces_converted', '')
            
            # Ajouter la colonne complète au DataFrame résultat
            result_df[column_name] = df['Average']
            
            # Interpoler la trace
            _, interpolated_trace = interpolate_trace(time_values, average_column, 
                                                    start_time, end_time, num_points)
            
            if interpolated_trace is not None:
                interpolated_df[column_name] = interpolated_trace
                print(f"✓ Traité et interpolé: {file_path.name} -> Colonne: {column_name}")
            else:
                print(f"⚠ Traité mais non interpolé: {file_path.name}")
            
        except Exception as e:
            print(f"Erreur lors du traitement de {file_path.name}: {str(e)}")
            print(f"  -> Vérifiez que la feuille 'Traces DF_F0' existe dans ce fichier")
    
    # Sauvegarder les résultats
    if not result_df.empty:
        # Fichier avec toutes les données originales
        output_path_full = folder / "ICI_donnees_completes.xlsx"
        result_df.to_excel(output_path_full, index=False)
        print(f"\n✓ Fichier de données complètes créé: {output_path_full}")
        
        # Fichier avec les données interpolées
        if not interpolated_df.empty:
            output_path_interp = folder / "ICI_donnees_interpolees.xlsx"
            interpolated_df.to_excel(output_path_interp, index=False)
            print(f"✓ Fichier de données interpolées créé: {output_path_interp}")
            print(f"  -> Plage de temps: [{start_time} - {end_time}]")
            print(f"  -> Nombre de points: {num_points}")
        
        print(f"Colonnes créées: {list(result_df.columns)}")
    else:
        print("Aucune donnée à sauvegarder.")

def main():
    """
    Fonction principale pour demander le dossier et lancer le traitement
    """
    print("=== Traitement et interpolation des fichiers Excel ===")
    print("Ce script extrait la colonne 'Average' de la feuille 'Traces DF_F0'")
    print("de tous les fichiers Excel d'un dossier et les regroupe dans de nouveaux fichiers\n")
    
    print(f"Paramètres actuels:")
    print(f"  - Temps de début: {TEMPS_DEBUT}")
    print(f"  - Temps de fin: {TEMPS_FIN}")
    print(f"  - Nombre de points d'interpolation: {NB_POINTS}")
    print()
    
    # Demander s'il faut modifier les paramètres
    modify = input("Voulez-vous modifier ces paramètres? (o/N): ").strip().lower()
    
    start_time = TEMPS_DEBUT
    end_time = TEMPS_FIN
    num_points = NB_POINTS
    
    if modify in ['o', 'oui', 'y', 'yes']:
        try:
            start_time = float(input(f"Temps de début (actuel: {TEMPS_DEBUT}): ") or TEMPS_DEBUT)
            end_time = float(input(f"Temps de fin (actuel: {TEMPS_FIN}): ") or TEMPS_FIN)
            num_points = int(input(f"Nombre de points (actuel: {NB_POINTS}): ") or NB_POINTS)
        except ValueError:
            print("Valeurs invalides, utilisation des paramètres par défaut.")
            start_time = TEMPS_DEBUT
            end_time = TEMPS_FIN
            num_points = NB_POINTS
    
    # Demander le chemin du dossier
    folder_path = input("Entrez le chemin du dossier contenant les fichiers Excel: ").strip()
    
    # Supprimer les guillemets si présents
    folder_path = folder_path.strip('"\'')
    
    # Lancer le traitement
    process_excel_files(folder_path, start_time, end_time, num_points)

if __name__ == "__main__":
    main()