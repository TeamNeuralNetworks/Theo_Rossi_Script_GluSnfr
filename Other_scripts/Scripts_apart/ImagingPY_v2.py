# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 16:53:47 2021

@author: Theo.ROSSI
"""





























if __name__ == '__main__':
    
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import PySimpleGUI as sg
    import configparser as cp
    import numpy as np
    import pandas as pd
    from openpyxl import Workbook
    from openpyxl import load_workbook
    from openpyxl.utils.dataframe import dataframe_to_rows
    import os


    sg.theme('DarkBlue')
        
    main_layout = [[sg.Text('File path')],
                   [sg.InputText(size=(35,1)), sg.FileBrowse()],
                   [sg.Button('GO')]]
              
    main_window = sg.Window('ImagingPY', main_layout, location=(0,0))
    
    while True:
       main_event, main_value = main_window.read()
       try:
           if main_event in (None, 'Close'):
               plt.close('all')
               break
           
           if main_event == 'GO':
               Dataset = {}
               
               file_sheets = pd.ExcelFile(str(main_value[0])).sheet_names
               for i in file_sheets:
                   data = pd.read_excel(f'{main_value[0]}', sheet_name=i, header=0)
                   Dataset[f'{i}'] = data

               variables(Dataset)
               
       except:
           pass
    main_window.close()