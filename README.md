Ce dossier contient tout les scripts que Théo Rossi a utilisé pour ses travaux et la rédaction de son papier (Rossi et al 2025).

Organisation :
	-'Other_scripts' contient tout les autres scritps de Théo que je (Anthime Perrot) n'ai pas eu à utiliser pour poursuivre son travail*.
	-'Dummy_Data' est un jeu de donné complet de Théo, des données brutes à celles analysées. A titre d'exemple pour l'organisation de ses données et l'utilisation de ses scripts. Toutes les données sont disponibles sur notre NAS.
	-'PCA_Notebooks_Data' est un dossier avec les différentes étapes (en jupiter notebook) pour la construction des différents clusters.
	- Les scripts présents dans le dossier initial sont ceux que j'ai utilisé pour poursuivre son travail (hors extraction des données*).

Ordre d'utilisation :
	- Extraction des traces et du F_background avec le script 'Extractor.py'. [File_traces.xlsx].
	- Normalisation DF/F et création d'un excel avec le script 'Converter.py'. [File_traces_converted.xlsx].
	- Extraction des amplitudes moyennes des 10 évènements avec le script 'SynaptiPy.py'. [File_traces_converted_Amp.xlsx].
	- Bootstrap pour détection du Failure rate avec le script 'Bootstrap_failures.py'. [File_traces_converted_data_boostrap.xlsx] et [File_traces_converted_histograms_bootstrap.xlsx].
	- PCA avec le script 'PCA_Clustering_Ca_Target_Gender.py'. Le détail est dans le dossier PCA_Notebooks_Data.


Le script 'Bootstrap_failures' a été légèrement corrigé par moi pour faciliter l'ouverture des excels, et un correction permettant au bouton 'Subtraction' d'être fonctionnel. Le script original est disponible en V1, le premier push est ma version corrigée.

* L'extraction de mes données a été effectué via le script Matlab 'ScanImage_Linscan_Analysis.m' d'Antoine Valera, et présent dans mon repository 'GUI_GluSnFR_Anthime'. Le script est différent car adaptés aux spécificités du logiciel scanimage que j'ai pu utiliser (sciscan pour Théo).


=========Readme initial de Théo Rossi===========

######## iGluSnFR.S72A/Data extraction and conversion #############

- "Boutons_raw_data": contains raw data. "GroupLinescan" folders are set for each parallel fiber and contain corresponding raw files.
	REQUIREMENT: 
		1. THESE FOLDERS ARE USED IN THE SCRIPT "Extractor.py" TO EXTRACT FLUORESCENCE TRACES AND FBACK.
		2. THE "traces.xlsx" FILE CREATED IS USED IN THE SCRIPT "Converter.py" FOR DF/F CONVERSION.

- "Boutons_analysis": contains extracted and converted data:
	- Date: animal
	- linescan#: parallel fiber ID
	- 20Hz: frequency
	- 10pulses: electrical stimulation train.
	- 2.5mMCa: [Ca2+]
	- bouton#: bouton ID
	- bootstrap: failures and success determined by bootstrap


Python scripts:
	- "Extractor.py"
	- "Converter.py"
	- "Bootstrap_failure_percentage.py"
