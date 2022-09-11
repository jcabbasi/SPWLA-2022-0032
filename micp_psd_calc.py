# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 20:08:02 2021

@author: j
"""


import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d




sigmaift=485/1000
tethaCA=140/180

dfcp = pd.read_pickle("D:/PhD/Studies/Papers/MICP/Dataset/DS_1/micpSS.pk1")

row_count = 0
# iterating over indices 
for col in dfcp.index: 
    row_count += 1
    
#fig, axr=plt.subplots()
for coreindx in dfcp.index: 
      print(coreindx)
      pccurve=dfcp.loc[coreindx,'Pccurve']
      psdcurve = np.empty([len(pccurve)-2,2])
      psdcurveR = np.empty([len(pccurve)-2,1])
      psdcurveF = np.empty([len(pccurve)-2,1])
      ravgc=0
      rnum=0
      for row in range(len(pccurve)-2):
          r1=2*sigmaift*math.cos(tethaCA)/pccurve[row,1]
          r2=2*sigmaift*math.cos(tethaCA)/pccurve[row+1,1]
          if(np.isinf(r1)):
              r1=0.1
          if(np.isinf(r2)):
              r2=0.1        
          ravg=(r1+r2)/2
          rnum=rnum+1
          ravgc=ravgc+ravg
          psdD=(pccurve[row,0]-pccurve[row+1,0])/(math.log10(r2/r1))
          psdcurveR[row,0]=ravg          
          psdcurveF[row,0]=psdD
#          axr.scatter(row,r1)
#      axr.set_yscale('log')    
      psdcurveN=psdcurveF/np.max(psdcurveF)
      dfcp.loc[coreindx,'PSD']=np.array([psdcurveR,psdcurveN])
      dfcp.loc[coreindx,'radiusAvg']=ravgc/rnum
# Swanson & Parachor
      Swansonarray=((max(pccurve[:,0])-pccurve[:,0])*100/(pccurve[:,1]/1000))
      Parachorarray=((max(pccurve[:,0])-pccurve[:,0])*1/(pccurve[:,1]/1e5)/(pccurve[:,1]/1e5))     
      Swansonarray=Swansonarray[np.isfinite(Swansonarray)]
      Parachorarray=Parachorarray[np.isfinite(Parachorarray)]
      SwansonPerm=431*math.pow(Swansonarray.max(),2.109)
      ParachorPerm=0.054*math.pow(Parachorarray.max(),1)
      # ParachorPerm=0.000279*math.pow(Parachorarray.max(),1.448)
      dfcp.loc[coreindx,'Swanson']=SwansonPerm*9.869233e-15
      dfcp.loc[coreindx,'Parachor']=ParachorPerm*9.869233e-15

# r35 Winland    
      pccurveN=dfcp.loc[coreindx,'Pccurve_norm']      
      fnc = interp1d(pccurveN[:,0], pccurveN[:,1])
      pc35=fnc(0.35)
      r35=2*sigmaift*math.cos(tethaCA)/pc35*1e6     #micro meter 
      kwinland=(0.588*math.log10(r35)-0.732+0.864*math.log10(dfcp.loc[coreindx,'Porosity']*100))/0.588#Porosity: %, K: m2
      dfcp.loc[coreindx,'Winland']=math.pow(10,kwinland)*9.869233e-15
      
# Purcell
      satcurvepurcel=pccurve[:,0]*100
      pccurvepurcel=pccurve[:,1]/1e5 
      indp=np.asarray(np.where(pccurve[:,1]<=0))
      satcurvepurcel=np.delete(satcurvepurcel,indp)
      pccurvepurcel=np.delete(pccurvepurcel,indp)
      purcellcoeff= satcurvepurcel/pccurvepurcel/pccurvepurcel
      summ=0
      for ii in range(np.size(purcellcoeff)-1):
          summ=purcellcoeff[ii]-purcellcoeff[ii+1]+summ
      dfcp.loc[coreindx,'Purcell']=0.66*0.216*dfcp.loc[coreindx,'Porosity']*summ*9.869233e-15

      
## Calculating Fractal Properties
for coreindx in dfcp.index:  
    pccurve=dfcp.loc[coreindx,'Pccurve']
    xx=np.log10(1-pccurve[:,0])
    yy=np.log10(pccurve[:,1])
    xxfa=np.zeros(0)
    for iii in range(0,np.size(xx)):
        if  np.isnan(xx[iii]) or  np.isnan(yy[iii]) or  np.isinf(xx[iii]) or  np.isinf(xx[iii]):
            xxfa=np.append(xxfa,iii)
    xx2=np.delete(xx,np.asarray(xxfa,dtype='int'))
    yy2=np.delete(yy,np.asarray(xxfa,dtype='int'))            
    m,b=np.polyfit(xx2,yy2,1)
    phi=dfcp.loc[coreindx,'Porosity']
    Sp=3/dfcp.loc[coreindx,'radiusAvg']*(1-phi)/phi
    kl=1.607*math.pow(((1-phi)/Sp), 2)*math.pow(0.952*phi*phi/(1-phi), 2/(m+2-1))
    dfcp.loc[coreindx,'fractal']=kl

    
    
    
    
     

fig, axswa=plt.subplots()
for coreindx in dfcp.index: 
    swaarr=dfcp.loc[coreindx,'Swanson']
    porosityarray=dfcp.loc[coreindx,'Porosity']
    axswa.plot(porosityarray,swaarr, marker="o", ls="")    
axswa.set_title("Swanson Parameter-Poro")
axswa.set_yscale('log') 


fig, axswap=plt.subplots()
for coreindx in dfcp.index: 
    swaarr=dfcp.loc[coreindx,'Swanson']
    permarray=dfcp.loc[coreindx,'Permeability']
    axswap.plot(permarray,swaarr, marker="o", ls="")    
axswap.set_title("Swanson Parameter-Perm")
axswap.set_yscale('log') 
axswap.set_xscale('log') 
 

fig, axwin=plt.subplots()
for coreindx in dfcp.index: 
    swaarr=dfcp.loc[coreindx,'Winland']
    permarray=dfcp.loc[coreindx,'Permeability']
    axwin.plot(permarray,swaarr, marker="o", ls="")    
axwin.set_title("Winland Perm")
axwin.set_yscale('log') 
axwin.set_xscale('log') 

dfcp.to_pickle("D:/PhD/Studies/Papers/MICP/Dataset/DS_1/micpSS.pk1")


    
          


   



    
    
    
    