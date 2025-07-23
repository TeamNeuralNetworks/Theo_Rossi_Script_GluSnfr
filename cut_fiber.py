import pandas as pd
import numpy as np
from pathlib import Path
from scipy.interpolate import interp1d

# ==============================================
# PARAMÈTRES À MODIFIER FACILEMENT
# ==============================================
TEMPS_DEBUT = 0.5       # Temps de début (X) #0.9
TEMPS_FIN = 1.5         # Temps de fin (Y) #1.5
ACTIVER_INTERPOLATION = True  # Activer/désactiver l'interpolation
NB_POINTS = 300        # Nombre de points pour l'interpolation (Z) - utilisé seulement si interpolation activée
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

def extract_data_in_range(time_values, trace_values, start_time, end_time):
    """
    Extrait les données dans une plage de temps sans interpolation
    
    Args:
        time_values: Array des valeurs de temps
        trace_values: Array des valeurs de la trace
        start_time: Temps de début
        end_time: Temps de fin
    
    Returns:
        tuple: (temps_extraits, valeurs_extraites)
    """
    # Filtrer les données dans la plage de temps
    mask = (time_values >= start_time) & (time_values <= end_time)
    time_filtered = time_values[mask]
    trace_filtered = trace_values[mask]
    
    if len(time_filtered) == 0:
        print(f"Attention: Aucun point dans la plage [{start_time}, {end_time}]")
        return None, None
    
    return time_filtered, trace_filtered

def process_excel_files(folder_path, start_time=TEMPS_DEBUT, end_time=TEMPS_FIN, 
                       enable_interpolation=ACTIVER_INTERPOLATION, num_points=NB_POINTS):
    """
    Traite tous les fichiers Excel d'un dossier et extrait/interpole la colonne 'Average'
    
    Args:
        folder_path (str): Chemin vers le dossier contenant les fichiers Excel
        start_time (float): Temps de début pour l'extraction
        end_time (float): Temps de fin pour l'extraction
        enable_interpolation (bool): Activer l'interpolation
        num_points (int): Nombre de points pour l'interpolation (si activée)
    """
    
    # Convertir en objet Path pour une manipulation plus facile
    folder = Path(folder_path)
    
    # Vérifier que le dossier existe
    if not folder.exists():
        print(f"Erreur: Le dossier '{folder_path}' n'existe pas.")
        return
    
    # Créer des DataFrames pour stocker les résultats
    result_df = pd.DataFrame()
    processed_df = pd.DataFrame()  # Pour les données traitées (interpolées ou extraites)
    
    # Parcourir tous les fichiers Excel dans le dossier
    excel_files = list(folder.glob("*.xlsx")) + list(folder.glob("*.xls"))
    
    if not excel_files:
        print("Aucun fichier Excel trouvé dans le dossier.")
        return
    
    print(f"Traitement de {len(excel_files)} fichier(s) Excel...")
    print(f"Paramètres: Temps [{start_time} - {end_time}]")
    if enable_interpolation:
        print(f"Interpolation activée: {num_points} points")
    else:
        print("Interpolation désactivée: extraction des données originales dans la plage")
    
    # Variables pour stocker les temps (différents selon interpolation ou non)
    common_time = None
    
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
            
            # Traiter selon le mode (interpolation ou extraction)
            if enable_interpolation:
                # Interpoler la trace
                processed_time, processed_trace = interpolate_trace(time_values, average_column, 
                                                                  start_time, end_time, num_points)
                
                if processed_trace is not None:
                    # Créer le vecteur temps commun pour l'interpolation (si pas encore fait)
                    if common_time is None:
                        common_time = processed_time
                        processed_df['Time'] = common_time
                    
                    processed_df[column_name] = processed_trace
                    
                    # Sauvegarder aussi chaque trace interpolée individuellement
                    temp_df = pd.DataFrame({
                        'Time': processed_time,
                        column_name: processed_trace
                    })
                    
                    output_path_individual = folder / f"ICI_interpole_{column_name}.xlsx"
                    temp_df.to_excel(output_path_individual, index=False)
                    
                    print(f"✓ Traité et interpolé: {file_path.name} -> Colonne: {column_name}")
                    print(f"  -> Fichier individuel: ICI_interpole_{column_name}.xlsx")
                else:
                    print(f"⚠ Traité mais non interpolé: {file_path.name}")
            else:
                # Extraire les données dans la plage sans interpolation
                processed_time, processed_trace = extract_data_in_range(time_values, average_column,
                                                                       start_time, end_time)
                
                if processed_trace is not None:
                    # Pour l'extraction sans interpolation, chaque fichier peut avoir des longueurs différentes
                    # On créera un fichier séparé pour chaque trace extraite
                    temp_df = pd.DataFrame({
                        'Time': processed_time,
                        column_name: processed_trace
                    })
                    
                    # Sauvegarder chaque trace extraite individuellement
                    output_path_individual = folder / f"ICI_extrait_{column_name}.xlsx"
                    temp_df.to_excel(output_path_individual, index=False)
                    
                    print(f"✓ Traité et extrait: {file_path.name} -> Fichier: ICI_extrait_{column_name}.xlsx")
                else:
                    print(f"⚠ Aucune donnée dans la plage pour: {file_path.name}")
            
        except Exception as e:
            print(f"Erreur lors du traitement de {file_path.name}: {str(e)}")
            print(f"  -> Vérifiez que la feuille 'Traces DF_F0' existe dans ce fichier")
    
    # Sauvegarder les résultats
    if not result_df.empty:
        # Fichier avec toutes les données originales
        output_path_full = folder / "ICI_donnees_completes.xlsx"
        result_df.to_excel(output_path_full, index=False)
        print(f"\n✓ Fichier de données complètes créé: {output_path_full}")
        
        # Fichier avec les données traitées (seulement si interpolation activée)
        if enable_interpolation and not processed_df.empty:
            output_path_processed = folder / "ICI_donnees_interpolees.xlsx"
            processed_df.to_excel(output_path_processed, index=False)
            print(f"✓ Fichier de données interpolées créé: {output_path_processed}")
            print(f"  -> Plage de temps: [{start_time} - {end_time}]")
            print(f"  -> Nombre de points: {num_points}")
            print(f"✓ Fichiers individuels interpolés également créés (ICI_interpole_*.xlsx)")
        elif not enable_interpolation:
            print(f"✓ Données extraites sauvegardées individuellement dans des fichiers séparés (ICI_extrait_*.xlsx)")
            print(f"  -> Plage de temps: [{start_time} - {end_time}]")
        
        if enable_interpolation:
            print(f"Colonnes interpolées créées: {list(processed_df.columns) if not processed_df.empty else 'Aucune'}")
        else:
            print(f"Colonnes originales: {list(result_df.columns)}")
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
    print(f"  - Interpolation activée: {'Oui' if ACTIVER_INTERPOLATION else 'Non'}")
    if ACTIVER_INTERPOLATION:
        print(f"  - Nombre de points d'interpolation: {NB_POINTS}")
    print()
    
    # Demander s'il faut modifier les paramètres
    modify = input("Voulez-vous modifier ces paramètres? (o/N): ").strip().lower()
    
    start_time = TEMPS_DEBUT
    end_time = TEMPS_FIN
    enable_interpolation = ACTIVER_INTERPOLATION
    num_points = NB_POINTS
    
    if modify in ['o', 'oui', 'y', 'yes']:
        try:
            start_time = float(input(f"Temps de début (actuel: {TEMPS_DEBUT}): ") or TEMPS_DEBUT)
            end_time = float(input(f"Temps de fin (actuel: {TEMPS_FIN}): ") or TEMPS_FIN)
            
            # Demander si on veut activer l'interpolation
            interp_choice = input(f"Activer l'interpolation? (O/n, actuel: {'O' if ACTIVER_INTERPOLATION else 'n'}): ").strip().lower()
            if interp_choice in ['o', 'oui', 'y', 'yes', '']:
                enable_interpolation = True
                num_points = int(input(f"Nombre de points d'interpolation (actuel: {NB_POINTS}): ") or NB_POINTS)
            elif interp_choice in ['n', 'non', 'no']:
                enable_interpolation = False
                print("Interpolation désactivée - les données seront extraites dans la plage de temps spécifiée")
            else:
                # Garder la valeur par défaut
                enable_interpolation = ACTIVER_INTERPOLATION
                if enable_interpolation:
                    num_points = int(input(f"Nombre de points d'interpolation (actuel: {NB_POINTS}): ") or NB_POINTS)
            
        except ValueError:
            print("Valeurs invalides, utilisation des paramètres par défaut.")
            start_time = TEMPS_DEBUT
            end_time = TEMPS_FIN
            enable_interpolation = ACTIVER_INTERPOLATION
            num_points = NB_POINTS
    
    # Demander le chemin du dossier
    folder_path = input("Entrez le chemin du dossier contenant les fichiers Excel: ").strip()
    
    # Supprimer les guillemets si présents
    folder_path = folder_path.strip('"\'')
    
    # Lancer le traitement
    process_excel_files(folder_path, start_time, end_time, enable_interpolation, num_points)

if __name__ == "__main__":
    main()