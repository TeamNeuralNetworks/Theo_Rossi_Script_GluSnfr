import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import numpy as np
from scipy import interpolate

# Dossier contenant les fichiers Excel
folder_path = r"C:\Anthime.PERROT\1_Thèse\1_Manip\5_Glusnf_Théo\2_Revision\All_boutons\All_Théo"

# Trouver tous les fichiers Excel dans le dossier
excel_files = glob.glob(os.path.join(folder_path, "*.xlsx"))

# Liste pour stocker toutes les traces "Average" et les temps
all_traces = []
all_times = []  # Stocker les temps correspondants à chaque trace
trace_names = []  # Stocker les noms des traces pour les titres

# Lire chaque fichier Excel
for file_path in excel_files:
    try:
        # Lire la feuille "Traces DF/F0"
        df = pd.read_excel(file_path, sheet_name="Traces DF_F0")
        
        # Chercher les colonnes qui contiennent "Average"
        average_columns = [col for col in df.columns if "Average" in str(col)]
        
        # Pour chaque trace "Average", récupérer sa colonne Time associée
        for col in average_columns:
            trace = df[col].dropna()  # Enlever les valeurs NaN
            if len(trace) > 0:
                all_traces.append(trace.values)
                
                # Récupérer la colonne Time correspondante
                if "Time" in df.columns:
                    time_col = df["Time"].dropna()
                    # S'assurer que la longueur du temps correspond à la trace
                    if len(time_col) >= len(trace):
                        all_times.append(time_col.values[:len(trace)])
                    else:
                        # Si Time est plus court, créer un axe temporel par défaut
                        all_times.append(np.arange(len(trace)))
                        print(f"Attention: Axe temporel par défaut pour {col} dans {os.path.basename(file_path)}")
                else:
                    # Pas de colonne Time, créer un axe temporel par défaut
                    all_times.append(np.arange(len(trace)))
                    print(f"Attention: Pas de colonne Time dans {os.path.basename(file_path)}")
                
                # Créer un nom pour la trace (nom du fichier + nom de la colonne)
                file_name = os.path.splitext(os.path.basename(file_path))[0]
                trace_names.append(f"{file_name}\n{col}")
        
        print(f"Fichier traité: {os.path.basename(file_path)} - {len(average_columns)} traces trouvées")
        
    except Exception as e:
        print(f"Erreur avec le fichier {os.path.basename(file_path)}: {e}")

# Visualiser les traces individuelles
if all_traces:
    # Trouver la longueur minimale pour aligner toutes les traces
    min_length = min(len(trace) for trace in all_traces)
    
    # Ajuster toutes les traces et leurs temps correspondants à la longueur minimale
    aligned_traces = []
    aligned_times = []
    
    for i, (trace, time) in enumerate(zip(all_traces, all_times)):
        aligned_traces.append(trace[:min_length])
        aligned_times.append(time[:min_length])
    
    # Créer un axe temporel commun pour la moyenne
    # Utiliser l'axe temporel qui a la plus grande plage temporelle
    max_time_range = 0
    best_time_axis = None
    for time_axis in aligned_times:
        time_range = time_axis[-1] - time_axis[0] if len(time_axis) > 1 else 0
        if time_range > max_time_range:
            max_time_range = time_range
            best_time_axis = time_axis
    
    # Si pas d'axe temporel valide, créer un axe par défaut
    if best_time_axis is None:
        common_time = np.arange(min_length)
        print("Attention: Axe temporel commun créé par défaut")
    else:
        common_time = best_time_axis
    
    # Interpoler toutes les traces sur l'axe temporel commun pour la moyenne
    interpolated_traces = []
    
    for trace, time_axis in zip(aligned_traces, aligned_times):
        if len(time_axis) > 1 and len(np.unique(time_axis)) > 1:
            # Interpolation seulement si l'axe temporel est valide
            try:
                f = interpolate.interp1d(time_axis, trace, kind='linear', 
                                       bounds_error=False, fill_value='extrapolate')
                interpolated_trace = f(common_time)
                interpolated_traces.append(interpolated_trace)
            except:
                # En cas d'erreur d'interpolation, utiliser la trace originale
                interpolated_traces.append(trace)
        else:
            # Si l'axe temporel n'est pas valide, utiliser la trace originale
            interpolated_traces.append(trace)
    
    # Calculer et afficher la trace moyenne globale EN PREMIER
    mean_trace = np.mean(interpolated_traces, axis=0)
    std_trace = np.std(interpolated_traces, axis=0)
    
    # Créer le graphique de la moyenne globale
    plt.figure(figsize=(12, 8))
    
    # Tracer les traces individuelles interpolées en arrière-plan (optionnel)
    for i, trace in enumerate(interpolated_traces):
        plt.plot(common_time, trace, alpha=0.3, color='gray', linewidth=0.8)
    
    # Tracer la moyenne avec l'écart-type
    plt.plot(common_time, mean_trace, 'b-', linewidth=2, label=f'Moyenne (n={len(all_traces)})')
    plt.fill_between(common_time, 
                     mean_trace - std_trace, 
                     mean_trace + std_trace, 
                     alpha=0.3, color='blue', label='±1 SD')
    
    plt.xlabel('Temps (s)')
    plt.ylabel('ΔF/F0')
    plt.title(f'Trace moyenne globale de toutes les données\n({len(all_traces)} traces de {len(excel_files)} fichiers)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Classe pour la navigation interactive des traces
    class TraceNavigator:
        def __init__(self, traces, times, names):
            self.traces = traces
            self.times = times
            self.names = names
            self.current_figure = 0
            self.cols = 3
            self.rows = 3
            self.traces_per_figure = self.cols * self.rows
            self.num_figures = (len(traces) + self.traces_per_figure - 1) // self.traces_per_figure
            
            # Créer la figure
            self.fig = plt.figure(figsize=(12, 12))
            self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
            
            # Afficher la première figure
            self.update_plot()
            plt.show()
        
        def on_key_press(self, event):
            if event.key == 'right' or event.key == 'left':
                if event.key == 'right' and self.current_figure < self.num_figures - 1:
                    self.current_figure += 1
                elif event.key == 'left' and self.current_figure > 0:
                    self.current_figure -= 1
                
                self.update_plot()
                self.fig.canvas.draw()
        
        def update_plot(self):
            self.fig.clear()
            
            start_idx = self.current_figure * self.traces_per_figure
            end_idx = min(start_idx + self.traces_per_figure, len(self.traces))
            
            for i, trace_idx in enumerate(range(start_idx, end_idx)):
                ax = self.fig.add_subplot(self.rows, self.cols, i + 1)
                
                # Tracer la trace individuelle avec son axe temporel spécifique
                ax.plot(self.times[trace_idx], self.traces[trace_idx], 'b-', linewidth=1)
                
                # Titre avec le nom de la trace
                title = self.names[trace_idx] if trace_idx < len(self.names) else f"Trace {trace_idx+1}"
                ax.set_title(title, fontsize=10, pad=5)
                
                # Labels
                ax.set_xlabel('Temps (s)', fontsize=9)
                ax.set_ylabel('ΔF/F0', fontsize=9)
                
                # Grille légère
                ax.grid(True, alpha=0.3)
                
                # Ajuster les ticks
                ax.tick_params(axis='both', which='major', labelsize=8)
            
            self.fig.tight_layout(pad=2.0)
            self.fig.suptitle(f'Traces individuelles - Figure {self.current_figure + 1}/{self.num_figures}\n(Utilisez les flèches ← → pour naviguer)', 
                             fontsize=14, y=0.98)
    
    # Créer le navigateur interactif
    navigator = TraceNavigator(aligned_traces, aligned_times, trace_names)
    
    print(f"\nRésumé:")
    print(f"- {len(excel_files)} fichiers Excel trouvés")
    print(f"- {len(all_traces)} traces 'Average' analysées")
    print(f"- {navigator.num_figures} figure(s) de traces individuelles créée(s)")
    print(f"- Longueur des traces: {min_length} points")
    
else:
    print("Aucune trace 'Average' trouvée dans les fichiers Excel.")

