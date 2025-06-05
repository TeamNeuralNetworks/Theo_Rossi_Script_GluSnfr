# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 14:53:30 2019

@author: Theo.ROSSI
"""

for f in range(len(protocols)):    
    for file in range(len(names)):
        if protocols[f] in names[file]:
            print(names[file],'stop', f,file)
            
            file_path = 'D:/F.LARENO.FACCINI/Preliminary Results/Ephy/Power values/Second Cue/{}.xlsx'.format(names[file])
            df1 = pd.read_excel(file_path, sheet_name='Before (500 ms)')
            df2 = pd.read_excel(file_path, sheet_name='During (500 ms)')
            df3 = pd.read_excel(file_path, sheet_name='After (500 ms)')
 
            m1 = df1.mean(axis=1)
            sd1 = df1.std(axis=1)
            m2 = df2.mean(axis=1)
            sd2 = df2.std(axis=1)
            m3 = df3.mean(axis=1)
            sd3 = df3.std(axis=1)
    
            if f+file==0:
                mf1 = pd.DataFrame(m1, index=None,columns=[names[f]])
                sdf1 = pd.DataFrame(sd1, index=None,columns=[names[f]])
                mf2 = pd.DataFrame(m2, index=None,columns=[names[f]])
                sdf2 = pd.DataFrame(sd2, index=None,columns=[names[f]])
                mf3 = pd.DataFrame(m3, index=None,columns=[names[f]])
                sdf3 = pd.DataFrame(sd3, index=None,columns=[names[f]])
                
            else:
                mf1[names[file]] = m1
                sdf1[names[file]] = sd1
                mf2[names[file]] = m2
                sdf2[names[file]] = sd2
                mf3[names[file]] = m3
                sdf3[names[file]] = sd3
                            
            
            with pd.ExcelWriter('{}/tot.xlsx'.format(powerdir), engine='openpyxl') as writer:
                                   
                mf1.to_excel(writer, sheet_name='Before (500 ms)', index=False, header=True)
                sdf1.to_excel(writer, sheet_name='SD Before', index=False, header=True)
                mf2.to_excel(writer, sheet_name='During (500 ms)', index=False, header=True)
                sdf2.to_excel(writer, sheet_name='SD During', index=False, header=True)
                mf3.to_excel(writer, sheet_name='After (500 ms)', index=False, header=True)
                sdf3.to_excel(writer, sheet_name='SD After', index=False, header=True)