
import pandas as pd
import random
import re

# Charger le fichier Excel
df = pd.read_excel(r'C:\Anthime.PERROT\1_Thèse\1_Manip\5_Glusnf_Théo\2_Revision\DATA_WT_Théo_smo4.xlsx')

# Extraire la première colonne (noms de fichiers)
fichiers = df.iloc[:, 0]

# Regrouper les lignes par (date, linescanX)
group_dict = {}
for idx, nom in enumerate(fichiers):
    match = re.search(r'^(\d{8})_linescan(\d)', str(nom))
    if match:
        date = match.group(1)
        scan = match.group(2)
        key = f'{date}_linescan{scan}'
        group_dict.setdefault(key, []).append(idx)

# Liste de tous les couples (date + linescanX)
all_groups = list(group_dict.keys())

# Vérifier qu'on a au moins 10 groupes
if len(all_groups) < 10:
    raise ValueError("Moins de 10 groupes 'date + linescanX' trouvés.")

# Tirage aléatoire jusqu'à ce que la sélection ait au moins 40 lignes
while True:
    selection = random.sample(all_groups, 10)
    indices_selectionnes = [i for key in selection for i in group_dict[key]]
    if len(indices_selectionnes) >= 50:
        break

# Extraire les lignes correspondantes (colonnes A à X → index 0 à 23)
df_selection = df.iloc[indices_selectionnes, :24]

# Exporter vers Excel
df_selection.to_excel('linescan_selection.xlsx', index=False)

