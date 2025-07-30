import pandas as pd
import numpy as np
from pathlib import Path
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import random

# ==============================================
# PARAMÈTRES À MODIFIER FACILEMENT
# ==============================================
TEMPS_DEBUT = 0.5    # Temps de début (X) #0.9
TEMPS_FIN = 1.5     # Temps de fin (Y) #1.5
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
    
    # Ajuster les bornes aux données réellement disponibles
    actual_start = time_filtered.min()
    actual_end = time_filtered.max()
    
    # Informer si on ajuste les bornes
    if actual_start > start_time or actual_end < end_time:
        print(f"  -> Ajustement: plage demandée [{start_time}, {end_time}] -> disponible [{actual_start:.3f}, {actual_end:.3f}]")
    
    # Créer la fonction d'interpolation sur l'échelle de temps originale
    try:
        interp_func = interp1d(time_filtered, trace_filtered, kind='linear', 
                              bounds_error=True)  # Pas d'extrapolation
        
        # Créer les nouveaux points de temps sur l'échelle originale ajustée
        original_time_interpolated = np.linspace(actual_start, actual_end, num_points)
        
        # Interpoler les valeurs sur l'échelle originale
        new_trace = interp_func(original_time_interpolated)
        
        # Créer l'échelle de temps normalisée (0 à durée réelle)
        actual_duration = actual_end - actual_start
        new_time = np.linspace(0, actual_duration, num_points)
        
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
        tuple: (temps_extraits_normalisés, valeurs_extraites)
    """
    # Filtrer les données dans la plage de temps
    mask = (time_values >= start_time) & (time_values <= end_time)
    time_filtered = time_values[mask]
    trace_filtered = trace_values[mask]
    
    if len(time_filtered) == 0:
        print(f"Attention: Aucun point dans la plage [{start_time}, {end_time}]")
        return None, None
    
    # Utiliser les bornes réelles des données filtrées
    actual_start = time_filtered.min()
    actual_end = time_filtered.max()
    
    # Informer si on ajuste les bornes
    if actual_start > start_time or actual_end < end_time:
        print(f"  -> Ajustement: plage demandée [{start_time}, {end_time}] -> disponible [{actual_start:.3f}, {actual_end:.3f}]")
    
    # Normaliser les temps pour commencer à 0
    time_normalized = time_filtered - actual_start
    
    return time_normalized, trace_filtered

def preview_traces(folder_path, start_time, end_time, num_samples=5):
    """
    Affiche un aperçu de traces aléatoires avec les barres de début/fin
    
    Args:
        folder_path (str): Chemin vers le dossier contenant les fichiers Excel
        start_time (float): Temps de début
        end_time (float): Temps de fin
        num_samples (int): Nombre de traces à afficher
    """
    folder = Path(folder_path)
    
    if not folder.exists():
        print(f"Erreur: Le dossier '{folder_path}' n'existe pas.")
        return False
    
    # Parcourir tous les fichiers Excel dans le dossier
    excel_files = list(folder.glob("*.xlsx")) + list(folder.glob("*.xls"))
    
    if not excel_files:
        print("Aucun fichier Excel trouvé dans le dossier.")
        return False
    
    # Sélectionner aléatoirement des fichiers
    num_files_to_show = min(num_samples, len(excel_files))
    selected_files = random.sample(excel_files, num_files_to_show)
    
    # Créer la figure avec des subplots
    fig, axes = plt.subplots(num_files_to_show, 1, figsize=(12, 3*num_files_to_show))
    if num_files_to_show == 1:
        axes = [axes]  # Pour cohérence si une seule trace
    
    fig.suptitle(f'Aperçu de {num_files_to_show} traces aléatoires\n'
                 f'Barres verticales: début={start_time}s, fin={end_time}s', 
                 fontsize=14, fontweight='bold')
    
    successful_plots = 0
    
    for i, file_path in enumerate(selected_files):
        try:
            # Lire le fichier Excel
            df = pd.read_excel(file_path, sheet_name='Traces DF_F0')
            
            if 'Average' not in df.columns:
                print(f"Attention: Colonne 'Average' manquante dans {file_path.name}")
                continue
            
            # Chercher la colonne de temps
            time_column = None
            possible_time_columns = ['Time', 'time', 'Temps', 'temps', 'T', 't']
            for col in possible_time_columns:
                if col in df.columns:
                    time_column = col
                    break
            
            if time_column is None:
                time_values = df.index.values
                time_label = "Index"
            else:
                time_values = df[time_column].values
                time_label = time_column
            
            # Créer le nom court du fichier
            short_name = file_path.stem.replace('_traces_converted', '')
            
            # Tracer la trace complète
            axes[successful_plots].plot(time_values, df['Average'], 'b-', linewidth=1, alpha=0.8)
            
            # Ajouter les barres verticales
            axes[successful_plots].axvline(x=start_time, color='green', linestyle='--', linewidth=2, 
                                         label=f'Début ({start_time}s)')
            axes[successful_plots].axvline(x=end_time, color='red', linestyle='--', linewidth=2, 
                                         label=f'Fin ({end_time}s)')
            
            # Zone d'extraction en surbrillance
            axes[successful_plots].axvspan(start_time, end_time, alpha=0.2, color='yellow', 
                                         label='Zone d\'extraction')
            
            # Configuration de l'axe
            axes[successful_plots].set_title(f'{short_name}', fontweight='bold')
            axes[successful_plots].set_xlabel(f'{time_label}')
            axes[successful_plots].set_ylabel('Average')
            axes[successful_plots].grid(True, alpha=0.3)
            axes[successful_plots].legend(loc='upper right')
            
            # Ajuster les limites pour bien voir la zone d'intérêt
            time_range = end_time - start_time
            margin = time_range * 0.5  # 50% de marge de chaque côté
            axes[successful_plots].set_xlim(start_time - margin, end_time + margin)
            
            successful_plots += 1
            
        except Exception as e:
            print(f"Erreur lors de la lecture de {file_path.name}: {str(e)}")
            continue
    
    # Masquer les axes inutilisés si il y a eu des erreurs
    for i in range(successful_plots, len(axes)):
        axes[i].set_visible(False)
    
    if successful_plots > 0:
        plt.tight_layout()
        
        # Utiliser un affichage non-bloquant
        plt.ion()  # Mode interactif
        plt.show()
        plt.draw()
        plt.pause(0.1)  # Petite pause pour s'assurer que le plot s'affiche
        
        print(f"\n✓ Aperçu affiché pour {successful_plots} trace(s)")
        print("📊 Graphique affiché - vérifiez la fenêtre matplotlib")
        return True
    else:
        print("Aucune trace n'a pu être affichée.")
        plt.close(fig)
        return False

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
    print(f"Paramètres: Temps [{start_time} - {end_time}] -> Normalisé [0 - {end_time - start_time}]")
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

            # Identifier la colonne de temps (dernière colonne)
            time_column = df.columns[-1]
            time_values = df[time_column].values

            # Identifier la colonne 'Average' (avant-dernière colonne)
            average_column_name = df.columns[-2]

            # Colonnes individuelles = toutes sauf 'Average' et colonne de temps
            individual_columns = [col for col in df.columns[:-2]]

            # Créer le nom de colonne de base (nom du fichier sans extension)
            base_name = file_path.stem.replace('_traces_converted', '')

            # Pour chaque trace individuelle
            for col in individual_columns + [average_column_name]:
                column_name = f"{base_name}_{col}"
                trace_values = df[col].values

                # Ajouter la colonne complète au DataFrame résultat (pour sauvegarde complète)
                result_df[column_name] = trace_values

                # Traiter selon le mode (interpolation ou extraction)
                if enable_interpolation:
                    processed_time, processed_trace = interpolate_trace(time_values, trace_values, 
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
                        temp_df.to_excel(output_path_individual, index=False, sheet_name='Traces DF_F0')
                        print(f"✓ Traité et interpolé: {file_path.name} -> Colonne: {column_name}")
                        print(f"  -> Fichier individuel: ICI_interpole_{column_name}.xlsx")
                    else:
                        print(f"⚠ Traité mais non interpolé: {file_path.name} - {col}")
                else:
                    processed_time, processed_trace = extract_data_in_range(time_values, trace_values,
                                                                           start_time, end_time)
                    if processed_trace is not None:
                        temp_df = pd.DataFrame({
                            'Time': processed_time,
                            column_name: processed_trace
                        })
                        output_path_individual = folder / f"ICI_extrait_{column_name}.xlsx"
                        temp_df.to_excel(output_path_individual, index=False, sheet_name='Traces DF_F0')
                        print(f"✓ Traité et extrait: {file_path.name} -> Fichier: ICI_extrait_{column_name}.xlsx")
                    else:
                        print(f"⚠ Aucune donnée dans la plage pour: {file_path.name} - {col}")

        except Exception as e:
            print(f"Erreur lors du traitement de {file_path.name}: {str(e)}")
            print(f"  -> Vérifiez que la feuille 'Traces DF_F0' existe dans ce fichier")

    # Sauvegarder les résultats
    if not result_df.empty:
        # Réordonner les colonnes pour respecter l'ordre original (toutes traces individuelles, Average, Time)
        ordered_cols = []
        if not excel_files:
            ordered_cols = list(result_df.columns)
        else:
            # Prendre l'ordre du premier fichier traité
            df_first = pd.read_excel(excel_files[0], sheet_name='Traces DF_F0')
            base_name = excel_files[0].stem.replace('_traces_converted', '')
            individual_columns = [col for col in df_first.columns[:-2]]
            average_column_name = df_first.columns[-2]
            for col in individual_columns:
                ordered_cols.append(f"{base_name}_{col}")
            ordered_cols.append(f"{base_name}_{average_column_name}")
        # Appliquer l'ordre si possible
        result_df = result_df[[col for col in ordered_cols if col in result_df.columns]]
        output_path_full = folder / "ICI_donnees_completes.xlsx"
        result_df.to_excel(output_path_full, index=False, sheet_name='Traces DF_F0')
        print(f"\n✓ Fichier de données complètes créé: {output_path_full}")

        # Fichier avec les données traitées (seulement si interpolation activée)
        if enable_interpolation and not processed_df.empty:
            processed_cols = [col for col in result_df.columns if col in processed_df.columns]
            processed_df = processed_df[['Time'] + processed_cols]
            output_path_processed = folder / "ICI_donnees_interpolees.xlsx"
            processed_df.to_excel(output_path_processed, index=False, sheet_name='Traces DF_F0')
            print(f"✓ Fichier de données interpolées créé: {output_path_processed}")
            print(f"  -> Plage de temps originale: [{start_time} - {end_time}]")
            print(f"  -> Plage de temps normalisée: [0 - {end_time - start_time}]")
            print(f"  -> Nombre de points: {num_points}")
            print(f"✓ Fichiers individuels interpolés également créés (ICI_interpole_*.xlsx)")
        elif not enable_interpolation:
            print(f"✓ Données extraites sauvegardées individuellement dans des fichiers séparés (ICI_extrait_*.xlsx)")
            print(f"  -> Plage de temps originale: [{start_time} - {end_time}]")
            print(f"  -> Plage de temps normalisée: [0 - {end_time - start_time}]")

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
    
    # Afficher l'aperçu des traces avant traitement
    print("\n" + "="*60)
    print("APERÇU DES TRACES")
    print("="*60)
    
    preview_success = preview_traces(folder_path, start_time, end_time)
    
    if preview_success:
        print("\n" + "="*60)
        confirm = input("Voulez-vous continuer avec le traitement? (O/n): ").strip().lower()
        if confirm in ['n', 'non', 'no']:
            print("Traitement annulé.")
            plt.ioff()  # Désactiver le mode interactif
            return
        print("="*60)
        plt.ioff()  # Désactiver le mode interactif avant de continuer
    
    # Lancer le traitement
    process_excel_files(folder_path, start_time, end_time, enable_interpolation, num_points)

if __name__ == "__main__":
    main()