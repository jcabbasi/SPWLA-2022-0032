# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 23:14:47 2021

@author: j
"""

#Outlier Detection

import pandas as pd
import numpy as np
import math  
import matplotlib as mpl
import matplotlib.pyplot as plt


dfcp = pd.read_pickle("D:/PhD/Studies/Papers/MICP/Dataset/DS_1/micp.pk1")
xdir2=np.empty(0)
xdir=dfcp.Porosity.to_numpy(dtype ='float32')
ydir=dfcp.Permeability.to_numpy(dtype ='float32')
m,b=np.polyfit(xdir,ydir,1)
ynew=m*xdir+b
fig, ax0=plt.subplots()
#ax0.plot(xdir,ynew)    
ax0.plot(xdir,ydir, marker="o", ls="")    
#ax0.set_yscale('log')


diff= ((ynew)-(ydir))

#fig, axhist=plt.hist((diff), bins = 5)

fig, ax1=plt.subplots()
ax1.plot(xdir,diff, marker="o", ls="")    
ax1.set_title("Diff")
Q1=np.quantile(diff,0.25)
Q3=np.quantile(diff,0.75)
IQR=Q3-Q1
lowbound=Q1-1.5*IQR
upbound=Q3+2.5*IQR
outindd=np.where(diff<lowbound)
outindd2=np.where(diff>upbound)
outindd=np.asarray(outindd,dtype='int')
outindd2=np.asarray(outindd2,dtype='int')
outindd=np.append(outindd,outindd2)
outindd.sort()

ax1.plot(xdir[outindd],diff[outindd], marker="o", ls="") 
ax0.plot(xdir[outindd],ydir[outindd], marker="*", ls="") 


fig, ax2=plt.subplots()
ax2.plot(xdir,diff, marker="o", ls="")    
ax2.plot(xdir[outindd],diff[outindd], marker="*",s=10,c="red", ls="") 

ax0.set_yscale('log')

fig, axsc=plt.subplots()
axsc.scatter(xdir,np.log(np.abs(diff)) )   
axsc.set_yscale('log')
axsc.set_ylim(-0.00000000003,0.000000000003)
axsc.set_xlim(-2,2)


#
outindd=np.append(outindd,[104]) #Observation of Fractal values
for coreindx in dfcp.index: 
   print(coreindx)
   if coreindx in outindd:
       dfcp.loc[coreindx,'Outlier']=bool(True)
   else:
       dfcp.loc[coreindx,'Outlier']=bool(False)
       
       


dfcp.to_pickle("D:/PhD/Studies/Papers/MICP/Dataset/DS_1/micp.pk1")

       
       






