% Dossier de sauvegarde
output_folder = 'C:\Anthime.PERROT\1_Thèse\1_Manip\5_Glusnf_Théo\2_Revision\Longue_fibre\241212_theo_9\fibre_3\Excel_15';
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

% Dimensions
[num_boutons, num_points, num_essais] = size(result_Trials);

for bouton = 1:num_boutons
    % Initialiser matrice de 1500 x 12 avec NaN
    data = NaN(num_points, 12);

    % Essais (colonnes 1 à 10)
    for essai = 1:min(num_essais, 10)
        data(:, essai) = result_Trials(bouton, :, essai)';
    end

    % Résultat moyen (colonne 11)
    data(:, 11) = result_Average(bouton, :)';

    % Temps (colonne 12)
    data(:, 12) = t_ax';

    % Nom du fichier
    filename = fullfile(output_folder, sprintf('Bouton_%d.xlsx', bouton));

    % Écriture des en-têtes numériques (0 à 9, puis Moyenne et Temps)
    headers = [0:9, NaN, NaN];
    writematrix(headers, filename, 'Sheet', 'Traces DF_F0', 'Range', 'A1');

    % Étiquettes texte pour les colonnes 11 et 12
    writecell({'Average'}, filename, 'Sheet', 'Traces DF_F0', 'Range', 'K1');
    writecell({'Time'}, filename, 'Sheet', 'Traces DF_F0', 'Range', 'L1');

    % Écriture des données
    writematrix(data, filename, 'Sheet', 'Traces DF_F0', 'Range', 'A2');
end