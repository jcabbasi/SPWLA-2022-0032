# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 08:17:33 2021

@author: j
"""

import pandas as pd
import numpy as np
import math  
import matplotlib as mpl
import matplotlib.pyplot as plt
import sklearn.metrics 

# %% Pc

dfcp = pd.read_pickle("D:/PhD/Studies/Papers/MICP/Dataset/DS_1/micpSS.pk1")
dfcp=dfcp.reset_index()
fig, ax=plt.subplots(1)
ax.set_yscale('Log')
for coreindx in dfcp.index: 
    pccurve=dfcp.loc[coreindx,'Pccurve_norm']
    ax.plot(pccurve[:,0],pccurve[:,1], marker="o", ls="")
    excelfile=dfcp.loc[coreindx,'filename']
    ax.text(pccurve[5,0], pccurve[5,1], excelfile)
ax.set_xlabel("Saturation"), ax.set_ylabel("Pc (pa)")


# %%
fig, ax2=plt.subplots(figsize=(9, 8.5))
# ax.set_title("Capillary Pressure")
plt.rcParams.update({'font.size': 17})
ax2.set_yscale('Log')
for coreindx in dfcp.index: 
    pccurve=dfcp.loc[coreindx,'Pccurve_norm']
    ax2.plot(pccurve[:,0],pccurve[:,1]*0.000145038, marker="o", ls="")
    excelfile=dfcp.loc[coreindx,'filename']
    ax.text(pccurve[5,0], pccurve[5,1], excelfile)
ax2.set_xlabel("Saturation"), ax2.set_ylabel("Pc (psi)")

# %% 

#
#fig, axn=plt.subplots()
#   
#for coreindx in dfcp.index: 
#    pccurve=dfcp.loc[coreindx,'Pccurve_J']
#    axn.plot(pccurve[:,0],pccurve[:,1], marker="o", ls="")
#axn.set_title("Normalized Capillary Pressure")
#axn.set_xlabel("Saturation"), axn.set_ylabel("Dimensionless Pc")
# 
  
Xso=dfcp.Miner
Yso='china'
L1='Norway'
L2='China'
fig, ax1 = plt.subplots(ncols=2,nrows=3, sharey=False)
ax1[2,0].set_yscale('Log')
ax1[2,0].set_xscale('Log')
ax1[2,0].set_xlabel("Permeability (mD)"), ax1[2,0].set_ylabel("Fractal Permeability (mD)")
ax1[2,0].scatter(dfcp.Permeability/9.869233e-15,dfcp.fractal/9.869233e-15,color="blue",s=150,alpha=0.7,label=L1)    
# ax1[2,0].scatter(dfcp.Permeability[ np.where(Xso==Yso)[0]],dfcp.fractal[ np.where(Xso==Yso)[0]], c='b',label=L2,alpha=1)
ax1[2,0].plot(dfcp.Permeability/9.869233e-15,dfcp.Permeability/9.869233e-15,ls="-",c='red',alpha=0.5)    

# ax1[0,0].legend(loc=0)


ax1[0,1].set_yscale('Log')
ax1[0,1].set_xscale('Log')
ax1[0,1].set_xlabel("Permeability (mD)"), ax1[0,1].set_ylabel("Permeability Winland (mD)")
ax1[0,1].scatter(dfcp.Permeability/9.869233e-15,dfcp.Winland/9.869233e-15,color="blue",s=150,alpha=0.7,label=L1)    
# ax1[0,1].scatter(dfcp.Permeability[ np.where(Xso==Yso)[0]],dfcp.Winland[ np.where(Xso==Yso)[0]], c='b',label=L2,alpha=1)
ax1[0,1].plot(dfcp.Permeability/9.869233e-15,dfcp.Permeability/9.869233e-15,ls="-",c='red',alpha=0.5)    
# ax1[0,1].legend(loc=0)

ax1[1,1].set_yscale('Log')
ax1[1,1].set_xscale('Log')
ax1[1,1].set_xlabel("Permeability (mD)"), ax1[1,1].set_ylabel("Permeability Purcell (mD)") 
ax1[1,1].scatter(dfcp.Permeability/9.869233e-15,dfcp.Purcell/9.869233e-15,color="blue",s=150,alpha=0.7,label=L1)   
# ax1[1,1].scatter(dfcp.Permeability[ np.where(Xso==Yso)[0]],dfcp.Purcell[ np.where(Xso==Yso)[0]], c='b',label=L2,alpha=1)
ax1[1,1].plot(dfcp.Permeability/9.869233e-15,dfcp.Permeability/9.869233e-15,ls="-",c='red',alpha=0.5)    

# ax1[1,1].legend(loc=0)

ax1[1,0].set_yscale('Log')
ax1[1,0].set_xscale('Log')
ax1[1,0].set_xlabel("Permeability (mD)"), ax1[1,0].set_ylabel("Permeability Swanson (mD)") 
ax1[1,0].scatter(dfcp.Permeability/9.869233e-15,dfcp.Swanson/9.869233e-15,color="blue",s=150,alpha=0.7,label=L1)    
# ax1[1,0].scatter(dfcp.Permeability[ np.where(Xso==Yso)[0]],dfcp.Swanson[ np.where(Xso==Yso)[0]], c='b',label=L2,alpha=1)
ax1[1,0].plot(dfcp.Permeability/9.869233e-15,dfcp.Permeability/9.869233e-15,ls="-",c='red',alpha=0.5)    


xphi=dfcp.Porosity.to_numpy(dtype ='float32')
yper=dfcp.Permeability.to_numpy(dtype ='float32')/9.869233e-15
m,b=np.polyfit(xphi,np.log10(yper),1)
ynew=m*xphi+b
yperlog=np.log10(yper)
ynewlog=np.log10(ynew)
ax1[0,0].set_yscale('log')
ax1[0,0].set_xlabel("Porosity"), ax1[0,0].set_ylabel("Permeability (mD)") 
ax1[0,0].scatter(dfcp.Porosity,dfcp.Permeability/9.869233e-15,color="blue",s=150,alpha=0.7)
ax1[0,0].plot(dfcp.Porosity,np.power(10,ynew),ls="-",c='red',alpha=0.7)    


ax1[2,1].set_yscale('Log')
ax1[2,1].set_xscale('Log')
ax1[2,1].set_xlabel("Permeability (mD)"), ax1[2,1].set_ylabel("Permeability Parachor (mD)") 
ax1[2,1].scatter(dfcp.Permeability/9.869233e-15,dfcp.Parachor/9.869233e-15,color="blue",s=150,alpha=0.7,label=L1)    
# ax1[1,0].scatter(dfcp.Permeability[ np.where(Xso==Yso)[0]],dfcp.Swanson[ np.where(Xso==Yso)[0]], c='b',label=L2,alpha=1)
ax1[2,1].plot(dfcp.Permeability/9.869233e-15,dfcp.Permeability/9.869233e-15,ls="-",c='red',alpha=0.5)    


# ax1[1,0].legend(loc=0)

#############
fig, ax2 = plt.subplots(ncols=1,nrows=1, sharey=True)
ax2.set_yscale('Log')
ax2.set_xscale('Log')
ax2.set_xlabel("Permeability (m2)"), ax2.set_ylabel("Fractal Permeability (m2)")
ax2.scatter(dfcp.Permeability,dfcp.fractal,c='b')    
ax2.scatter(dfcp.Permeability[dfcp.Outlier==True],dfcp.fractal[dfcp.Outlier==True], c='r',alpha=1)


mses=np.zeros([2,4])
mses[0,0]=sklearn.metrics.mean_squared_error(np.log10(dfcp.Permeability.to_numpy(dtype ='float32')), np.log10(dfcp.fractal.to_numpy(dtype ='float32')))
mses[0,1]=sklearn.metrics.mean_squared_error(np.log10(dfcp.Permeability.to_numpy(dtype ='float32')), np.log10(dfcp.Winland.to_numpy(dtype ='float32')))
mses[0,2]=sklearn.metrics.mean_squared_error(np.log10(dfcp.Permeability.to_numpy(dtype ='float32')), np.log10(dfcp.Purcell.to_numpy(dtype ='float32')))
mses[0,3]=sklearn.metrics.mean_squared_error(np.log10(dfcp.Permeability.to_numpy(dtype ='float32')), np.log10(dfcp.Swanson.to_numpy(dtype ='float32')))

mses[1,0]=sklearn.metrics.r2_score(np.log10(dfcp.Permeability.to_numpy(dtype ='float32')), np.log10(dfcp.fractal.to_numpy(dtype ='float32')))
mses[1,1]=sklearn.metrics.r2_score(np.log10(dfcp.Permeability.to_numpy(dtype ='float32')), np.log10(dfcp.Winland.to_numpy(dtype ='float32')))
mses[1,2]=sklearn.metrics.r2_score(np.log10(dfcp.Permeability.to_numpy(dtype ='float32')), np.log10(dfcp.Purcell.to_numpy(dtype ='float32')))
mses[1,3]=sklearn.metrics.r2_score(np.log10(dfcp.Permeability.to_numpy(dtype ='float32')), np.log10(dfcp.Swanson.to_numpy(dtype ='float32')))
fig, ax3 = plt.subplots()
# ax3.bar(['a','b','c','d'],np.log10(mses[0,0:4]), width = 0.25)
ax3.bar(['a','b','c','d'],np.log10(mses[1,0:4]), width = 0.25)


#fig, ax1=plt.subplots()
#for coreindx in dfcp.index: 
#    pccurve=dfcp.loc[coreindx,'PSD']
#    ax1.plot(pccurve[:,0],pccurve[:,1], marker="o", ls="")    
#ax1.set_title("Pore Size Distribution")
#    
##Poro Perm    
#fig, ax2=plt.subplots(1,1)
#ax2.set_xlabel("Porosity"), ax2.set_ylabel("Permeability")
#for coreindx in dfcp.index: 
#    poro=dfcp.loc[coreindx,'Porosity']
#    perm=dfcp.loc[coreindx,'Permeability']
#    lith=dfcp.loc[coreindx,'Lithology']
#    if lith=="SS":
#        ax2.plot(poro,perm, marker="o", ls="", color="black")  
#    elif lith=="SH":
#        ax2.plot(poro,perm, marker="o", ls="", color="red")  
#    elif lith=="C":
#        ax2.plot(poro,perm, marker="o", ls="", color="blue")  
#    ax2.set_yscale('log')
#ax2.set_title("Permeability, black: SS, red: SH, blue: C")