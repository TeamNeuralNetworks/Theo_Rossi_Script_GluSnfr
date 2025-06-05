# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 18:09:27 2019

@author: Theo.ROSSI
"""

import pandas as pd
import tkinter as tk
from PIL import Image, ImageTk



####################################SETTINGS###################################
###############################################################################

Path = r'\\equipe2-nas1\Theo.ROSSI\GluSnFR\Virus_AAV.hSynap.SF-iGluSnFR_S72A\AAVDJ.GluSnFR-S72A'
Big_folder = '2019_12_20'
Coord_file = '20191220_18_44_20_linescan8_50Hz_[xy]T_ROI.coord'
ROI_file = '20191220_18_44_20_linescan8_50Hz_[xy]T_ROI.tif'

###############################################################################
###############################################################################


X, Y = [],[]

new_path = pd.read_csv("{}/{}/{}".format(Path, Big_folder, Coord_file),sep="\t",header=1)

for i in range(len(new_path)): 
    
    basic_value = new_path.iloc[i,0]
    
    X.append(int(basic_value.split(',')[0]))
    Y.append(int(basic_value.split(',')[1]))

data = {'X':X, 'Y':Y}
df = pd.DataFrame(data)


class LoadImage:
    def __init__(self, root):
        
        frame1 = tk.Frame(root)
        frame1.grid(row=0, column=0)
        
        self.vals = ['A', 'B', 'C', 'D', 'E', 'F']
        self.labels = ['Button1', 'Button2', 'Button3', 'Button4', 'Button5', 'Button6']
        self.varGr = tk.StringVar()
        self.varGr.set(self.vals[0])
        
        for i in range(len(self.labels)):
            self.buttons = tk.Radiobutton(frame1, variable=self.varGr, text=self.labels[i], value=self.vals[i], command=self.enable_data)
            self.buttons.grid(row=i+1)
            
            self.entr1 = tk.Entry(frame1, width=6)
            self.entr1.configure(state='disabled')
            self.entr1.grid(row=i+1, column=1)
            self.entr2 = tk.Entry(frame1, width=6)
            self.entr2.configure(state='disabled')
            self.entr2.grid(row=i+1, column=2)
            
            idx1 = tk.Label(frame1, text='Index1')
            idx2 = tk.Label(frame1, text='Index2')
            idx1.grid(row=0, column=1)
            idx2.grid(row=0, column=2)
        
        self.button_delete = tk.Button(frame1, text='Clear', width=7, command=self.clear_all_data)
        self.button_delete.grid(row=7, column=1)
        
        
        frame2 = tk.Frame(root)
        frame2.grid(row=0, column=2)
        
        self.canvas = tk.Canvas(frame2, width=256, height=256)
        self.canvas.grid(row=3, column=4)
        
        File = "{}/{}/{}".format(Path, Big_folder, ROI_file)
        self.orig_img = Image.open(File)
        self.img = ImageTk.PhotoImage(self.orig_img)
        self.canvas.create_image(130,130,image=self.img)
        
    
        self.zoomcycle = 0
        '''Start with the image not zoomed'''
        self.zimg_id = None
        '''Start with the image without the zoom crop'''
        
        self.canvas.bind("<MouseWheel>",self.zoomer)
        self.canvas.bind("<Motion>",self.crop)
        self.canvas.bind("<Button-1>",self.pixel_data)
        
        
        
    def zoomer(self, event):
        if event.delta > 0:
            if self.zoomcycle != 4: self.zoomcycle += 1
        elif (event.delta < 0):
            if self.zoomcycle != 0: self.zoomcycle -= 1
        self.crop(event)
        

    def crop(self, event):
        if self.zimg_id: self.canvas.delete(self.zimg_id)
        '''Delete the crop when zoomcycle=0'''
        
        if self.zoomcycle != 0:
            x,y = event.x, event.y
            
            if self.zoomcycle == 1:
                tmp = self.orig_img.crop((x-45,y-30,x+45,y+30))
                '''The Image.crop method is used to crop a rectangular portion of the image'''
                
            elif self.zoomcycle == 2:
                tmp = self.orig_img.crop((x-30,y-20,x+30,y+20))
                
            elif self.zoomcycle == 3:
                tmp = self.orig_img.crop((x-15,y-10,x+15,y+10))
                
            elif self.zoomcycle == 4:
                tmp = self.orig_img.crop((x-6,y-4,x+6,y+4))
                
            size = 300,200
            self.zimg = ImageTk.PhotoImage(tmp.resize(size))
            '''The Image.resize method returns a resized copy of the image'''
            self.zimg_id = self.canvas.create_image(event.x,event.y,image=self.zimg)
            '''The crop is created'''

    
    def pixel_data(self, event):
        for idx in range(len(df)):
            if event.x == df.loc[idx][0] and event.y == df.loc[idx][1]:
                print(idx)
                self.data = str(idx) + "\t" 
                self.entr1.insert(0, self.data)
                self.entr2.insert(0, self.data)
                self.pixel_color = self.canvas.create_rectangle(event.x, event.y-1, event.x+1, event.y, fill = 'red')

        
    def clear_all_data(self):
        self.entr1.delete(0, 'end')
        self.entr2.delete(0, 'end')
        self.canvas.delete(self.pixel_color)

    
    def enable_data(self):
        
        self.entr1.configure(state='normal')
        
        self.entr2.configure(state='normal')
            
    
#    def excel_file(self):
#        self.values = self.txt.get(1.0, 'end')
#        print(self.values)
#        self.df2 = pd.DataFrame(self.values, index=None, columns=None)
#        with pd.ExcelWriter("D:/Theo.ROSSI/GluSnFR/Virus_AAV.hSynap.SF-iGluSnFR_S72A/AAVDJ.GluSnFR-S72A/{}/{}.".format(Big_folder, Xlsx_file)) as writer:
#            self.df2.to_excel(writer, header=False, index=False) 


if __name__ == '__main__':
    root = tk.Tk()
    root.title("Linescan nb: {}".format(ROI_file))
    App = LoadImage(root)
    root.mainloop()
