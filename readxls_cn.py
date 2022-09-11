# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 20:08:02 2021

@author: j
"""

import xlrd 
import pandas as pd
import numpy as np

dfcp = pd.read_pickle("D:/PhD/Studies/Papers/MICP/Dataset/DS_1/micp.pk1")
dnuis=len(dfcp)

datano=172



for x in range(0+dnuis,datano+dnuis):
    xz=x-dnuis+1
    if xz<10:
     filename= '100'+str(xz)+'.xlsx'
    elif xz<100:
     filename= '10'+str(xz)+'.xlsx'
    else:
     filename= '1'+str(xz)+'.xlsx'  
    
    print(x)
    loc = ("D:/PhD/Studies/Papers/MICP/Dataset/DS_2/"+filename)
#    print(loc)
    # To open Workbook
    wb    = xlrd.open_workbook(loc)
    sheet = wb.sheet_by_index(0)
    
    dfcp.loc[x,'log']= "Log \n"  

    #Test Number    
    dfcp.loc[x,'testNo']=x

    #File Name    

    dfcp.loc[x,'filename']=filename

    
    #Porosity
    if sheet.cell_value(2,7)=="prct":
        dfcp.loc[x,'Porosity']=float(sheet.cell_value(3,7))/100
    else:
        dfcp.loc[x,'Porosity']=float(sheet.cell_value(3,7))
    if dfcp.loc[x,'Porosity']>1:
        errorlog = "Wrong Porosity Scale \n"
        dfcp.loc[x,'log']= str(dfcp.loc[x,'log']) + errorlog      
    
    #Permeability
    if sheet.cell_value(2,6)=="mD":
        dfcp.loc[x,'Permeability']=float(sheet.cell_value(3,6))*(0.986923e-15)   
    else:
        dfcp.loc[x,'Permeability']=float(sheet.cell_value(3,6))     
    
    if dfcp.loc[x,'Permeability']>1e-5:
        errorlog = "Wrong Permeability Scale \n"
        dfcp.loc[x,'log']= str(dfcp.loc[x,'log'])+ errorlog    
    
    
    ## Pc Curve
    #Saturation
    if sheet.cell_value(3,3)=="prct":
        pcc_sat=(np.array(sheet.col_values(3, start_rowx=5, end_rowx=150)))/100
    else:
        pcc_sat=np.array(sheet.col_values(3, start_rowx=5, end_rowx=150))
 
    if (min(pcc_sat)<-60) or (max(pcc_sat)>10):
        errorlog = "Wrong Sat Scale \n"
        dfcp.loc[x,'log']= str(dfcp.loc[x,'log'])+ errorlog    


  
    if sheet.cell_value(4,3)=="Snw":
        pcc_sat_max=np.max(pcc_sat)          
        pcc_sat=1-pcc_sat
        
    
        # Source of Data, Norway: no, China: cn
    dfcp.loc[x,'Miner']="china"  
    
    #Pc
    if   sheet.cell_value(3,4)=="Mpa" or sheet.cell_value(3,4)=="MPa":
         pcc_pc= np.array(sheet.col_values(4, start_rowx=5, end_rowx=150))*1e6
    elif sheet.cell_value(3,4)=="psi":
         pcc_pc= np.array(sheet.col_values(4, start_rowx=5, end_rowx=150))*6894.76      
    else:
        pcc_pc= np.array(sheet.col_values(4, start_rowx=5, end_rowx=150))
    
    if  (max(pcc_pc)<1e7):
        errorlog = "Wrong Pc Scale \n"
        dfcp.loc[x,'log']= str(dfcp.loc[x,'log'])+ errorlog    

    satfirst = np.asarray(np.where(pcc_sat==min(pcc_sat)))
    satlast = np.asarray(np.where(pcc_sat==max(pcc_sat)))
    if (satfirst.any()>satlast.any()):
     pcc_sat=np.flip(pcc_sat, 0)   
     pcc_pc =np.flip(pcc_pc , 0)   
   
    
    
    pcc_sat_max=np.max(pcc_sat)    
    pcc_sat_min=np.min(pcc_sat)
    pcc_pc_max=np.max(pcc_pc)    
    pcc_pc_min=np.min(pcc_pc)
    
    pcc_sat_norm=(pcc_sat-pcc_sat_min)/(pcc_sat_max-pcc_sat_min)
    pcc_pc_norm =(pcc_pc-pcc_pc_min)/(pcc_pc_max-pcc_pc_min)    
    dfcp.loc[x,'Pccurve']=np.array([pcc_sat,pcc_pc]).T    
    dfcp.loc[x,'Pccurve_norm']=np.array([pcc_sat_norm,pcc_pc]).T
    dfcp.loc[x,'Pccurve_J']=np.array([pcc_sat_norm,pcc_pc_norm]).T

    #Lithology
    dfcp.loc[x,'Lithology']=(sheet.cell_value(2,8))
        
    #Source
    dfcp.loc[x,'Source']=(sheet.cell_value(11,7))    


  

    resultMin = pcc_pc[np.where(pcc_sat==min(pcc_sat))]
    resultMax = pcc_pc[np.where(pcc_sat==max(pcc_sat))]
    if (resultMax[0]>resultMin[0]):
        errorlog = "Wrong Pc Curve \n"
        dfcp.loc[x,'log']= str(dfcp.loc[x,'log'])+ errorlog   

#SAVE IT
dfcp.to_pickle("D:/PhD/Studies/Papers/MICP/Dataset/DS_1/micp.pk1")
#dfcpload = pd.read_pickle("D:/PhD/Studies/Papers/MICP/Dataset/DS_1/micp.pk1")

