# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 14:28:15 2021

@author: j
"""
import pandas as pd
import numpy as np
import math  
import matplotlib as mpl
import matplotlib.pyplot as plt

dfcp = pd.read_pickle("D:/PhD/Studies/Papers/MICP/Dataset/DS_1/micp.pk1")


fig, ax=plt.subplots(1)
for coreindx in dfcp.index: 
    pccurve=dfcp.loc[coreindx,'Pccurve']
    ax.plot(pccurve[:,0],pccurve[:,1], marker="o", ls="")
    excelfile=dfcp.loc[coreindx,'filename']
    ax.text(pccurve[8,0], pccurve[8,1], excelfile)

ax.set_title("Capillary Pressure")
ax.set_xlabel("Saturation"), ax.set_ylabel("Pc (pa)")


fig, axn=plt.subplots()
   
for coreindx in dfcpn.index: 
    pccurve=dfcpn.loc[coreindx,'Pccurve_J']
    axn.plot(pccurve[:,0],pccurve[:,1], marker="o", ls="")
axn.set_title("Normalized Capillary Pressure")
axn.set_xlabel("Saturation"), axn.set_ylabel("Dimensionless Pc")
   
fig, ax1=plt.subplots()
for coreindx in dfcp.index: 
    pccurve=dfcp.loc[coreindx,'PSD']
    ax1.plot(pccurve[:,0],pccurve[:,1], marker="o", ls="")    
ax1.set_title("Pore Size Distribution")
    
#Poro Perm    
fig, ax2=plt.subplots(1,1)
ax2.set_xlabel("Porosity"), ax2.set_ylabel("Permeability")
for coreindx in dfcp.index: 
    poro=dfcp.loc[coreindx,'Porosity']
    perm=dfcp.loc[coreindx,'Permeability']
    lith=dfcp.loc[coreindx,'Lithology']
    if lith=="SS":
        ax2.plot(poro,perm, marker="o", ls="", color="black")  
    elif lith=="SH":
        ax2.plot(poro,perm, marker="o", ls="", color="red")  
    elif lith=="C":
        ax2.plot(poro,perm, marker="o", ls="", color="blue")  
    ax2.set_yscale('log')
ax2.set_title("Permeability, black: SS, red: SH, blue: C")

   