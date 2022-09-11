# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 17:18:17 2021

@author: j

"""

import numpy as np
import scipy

def averagePc(dfcp):
    import numpy as np
    pcavg=np.zeros([dfcp.shape[0],1])
    indxno=0
    for coreindx in dfcp.index: 
       indxno=indxno+1
       popt=dfcp.loc[coreindx,'PCparameters']
       pcavg[indxno-1]=np.average(popt[2,:])
    return pcavg

def medianPc(dfcp):
    import numpy as np
    pcavg=np.zeros([dfcp.shape[0],1])
    indxno=0
    for coreindx in dfcp.index: 
       indxno=indxno+1
       popt=dfcp.loc[coreindx,'PCparameters']
       pcavg[indxno-1]=np.median(popt[2,:])
    return pcavg


def modePc(dfcp):
    import numpy as np
    pcavg=np.zeros([dfcp.shape[0],1])
    indxno=0
    for coreindx in dfcp.index: 
       indxno=indxno+1
       popt=dfcp.loc[coreindx,'Pccurve']  
       pcc=popt[:,1]
       pcc=np.log10(pcc[pcc>0])
       his,binn=np.histogram(pcc, bins=10)
       xx=binn[np.where(his==his.max())[0][0]]+binn[np.where(his==his.max())[0][0]+1]/2       
       pcavg[indxno-1]=xx
    return pcavg

def pcmax(dfcp):  
    import numpy as np
    indxno=0
    pcmax=np.zeros([dfcp.shape[0],1])
    for coreindx in dfcp.index: 
      indxno=indxno+1
      pccurveN=dfcp.loc[coreindx,'Pccurve']    
      pcmax[indxno-1]=np.max(pccurveN[:,1])
    return pcmax


def rsaturation(dfcp,fraction):
    
    from scipy.interpolate import interp1d
    import math
    sigmaift=485/1000
    tethaCA=140/180
    indxno=0
    r35=np.zeros([dfcp.shape[0],1])

    for coreindx in dfcp.index: 
      indxno=indxno+1
      pccurveN=dfcp.loc[coreindx,'Pccurve_norm']      
      fnc = interp1d(pccurveN[:,0], pccurveN[:,1])
      pc35=fnc(fraction)
      r35[indxno-1]=2*sigmaift*math.cos(tethaCA)/pc35*1e6     #micro meter 
    return r35

def satrange(dfcp):  
    import numpy as np
    indxno=0
    srange=np.zeros([dfcp.shape[0],1])
    for coreindx in dfcp.index: 
      indxno=indxno+1
      pccurveN=dfcp.loc[coreindx,'Pccurve']    
      srange[indxno-1]=np.max(pccurveN[:,0])-np.min(pccurveN[:,0])
    return srange

def satmin(dfcp):  
    import numpy as np
    indxno=0
    satmin=np.zeros([dfcp.shape[0],1])
    for coreindx in dfcp.index: 
      indxno=indxno+1
      pccurveN=dfcp.loc[coreindx,'Pccurve']    
      satmin[indxno-1]=np.min(pccurveN[:,0])
    return satmin