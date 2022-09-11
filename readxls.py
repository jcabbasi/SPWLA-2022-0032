# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 20:08:02 2021

@author: j
"""
# from IPython import get_ipython
# get_ipython().magic('reset -sf')


import xlrd # pip install xlrd==1.2.0 : THE LATEST VERSION OF XLRD DOESNOT WORK
import pandas as pd
import numpy as np


datano=101

dfcp = pd.DataFrame(index=range(datano), columns=['testNo','filename','Source','Miner','Outlier','log', 'Porosity', 'Permeability', 'Lithology', 'Pccurve','Pccurve_norm', 'Pccurve_J', 'radiusAvg','Purcell','Swanson','fractal' , 'Winland','SVMPerm',"Parachor",'NNPerm','SavedPlace1','SavedPlace2', 'PSD','PCparameters' ])


for x in range(0,datano):
  
    if x<10:
     filename= '000'+str(x)+'.xlsx'
    elif x<100:
     filename= '00' +str(x)+'.xlsx'
    else:
     filename= '0'  +str(x)+'.xlsx'  
    

    loc = ("D:/PhD/Studies/Papers/MICP/Dataset/DS_1/"+filename)
    print(loc)
    # To open Workbook
    wb = xlrd.open_workbook(loc)
    sheet = wb.sheet_by_index(0)
    dfcp.loc[x,'log']= "Log \n"  

    #Test Number    
    dfcp.loc[x,'testNo']=x

    #File Name    

    dfcp.loc[x,'filename']=filename

    
    #Porosity
    if sheet.cell_value(2,6)=="prct":
        dfcp.loc[x,'Porosity']=float(sheet.cell_value(3,6))/100
    else:
        dfcp.loc[x,'Porosity']=float(sheet.cell_value(3,6))
    if dfcp.loc[x,'Porosity']>1:
        errorlog = "Wrong Porosity Scale \n"
        dfcp.loc[x,'log']= str(dfcp.loc[x,'log']) + errorlog      
    
    #Permeability
    if sheet.cell_value(2,5)=="mD":
        dfcp.loc[x,'Permeability']=float(sheet.cell_value(3,5))*(0.986923e-15)   
    else:
        dfcp.loc[x,'Permeability']=float(sheet.cell_value(3,5))     
    
    if dfcp.loc[x,'Permeability']>1e-5:
        errorlog = "Wrong Permeability Scale \n"
        dfcp.loc[x,'log']= str(dfcp.loc[x,'log'])+ errorlog    
    
    
    ## Pc Curve
    #Saturation
    if sheet.cell_value(3,2)=="prct":
        pcc_sat=np.array(sheet.col_values(2, start_rowx=5, end_rowx=150))/100
    else:
        pcc_sat=np.array(sheet.col_values(2, start_rowx=5, end_rowx=150))
 
    if (min(pcc_sat)<-60) or (max(pcc_sat)>10):
        errorlog = "Wrong Sat Scale \n"
        dfcp.loc[x,'log']= str(dfcp.loc[x,'log'])+ errorlog    


  
    if sheet.cell_value(4,2)=="Snw":
        pcc_sat_max=np.max(pcc_sat)          
        pcc_sat=1-pcc_sat
        
    
    
    
    #Pc
    if   sheet.cell_value(3,3)=="Mpa" or sheet.cell_value(3,3)=="MPa":
         pcc_pc= np.array(sheet.col_values(3, start_rowx=5, end_rowx=150))*1e6
    elif sheet.cell_value(3,3)=="psi":
         pcc_pc= np.array(sheet.col_values(3, start_rowx=5, end_rowx=150))*6894.76      
    else:
        pcc_pc= np.array(sheet.col_values(3, start_rowx=5, end_rowx=150))
    
    if  (max(pcc_pc)<1e6):
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
    dfcp.loc[x,'Lithology']=(sheet.cell_value(2,7))
        
    #Source
    dfcp.loc[x,'Source']=(sheet.cell_value(11,6))    
    
    # Source of Data, Norway: norway, China: china
    dfcp.loc[x,'Miner']="norway"  

  

    resultMin = pcc_pc[np.where(pcc_sat==min(pcc_sat))]
    resultMax = pcc_pc[np.where(pcc_sat==max(pcc_sat))]
    if (resultMax[0]>resultMin[0]):
        errorlog = "Wrong Pc Curve \n"
        dfcp.loc[x,'log']= str(dfcp.loc[x,'log'])+ errorlog   

#SAVE IT
dfcp.to_pickle("D:/PhD/Studies/Papers/MICP/Dataset/DS_1/micp.pk1")
#dfcpload = pd.read_pickle("D:/PhD/Studies/Papers/MICP/Dataset/DS_1/micp.pk1")

