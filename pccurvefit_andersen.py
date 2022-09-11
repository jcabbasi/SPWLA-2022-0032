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
import sklearn
from scipy.optimize import curve_fit

dfcp = pd.read_pickle("D:/PhD/Studies/Papers/MICP/Dataset/DS_1/micp.pk1")


# objective function
def func(x, a, b, c):
    return a * x + b * x * x + c
def cappfunc(x, a1,a2,a3,k1,k2,n1,n2):
    return   ( a1/(1 + k1 * np.power(x,n1) ) ) - ( a2/( 1 + k2 *( 1 - np.power(x,n2) ) ) )+ a3

def cappfunc2(x, a1,a2,a3,k1,k2,n1,n2):
    return   ( a1/(1 + k1 * (x*n1) ) ) - ( a2/( 1 + k2 *( 1 - (x*n2) ) ) )+ a3

def cappfunccorey(x, pct,la,c):
    return pct*np.power(x,la)+0*c

def cappfuncbentsen(x, pct,pcs,la):
    return pct*np.power(x,la)+c  
def cappfuncmultsent(x, a,b,c,d):
    return a*x*x*x+b*x*x+c*x+d                                



for coreindx in dfcp.index: 
    pccurve=dfcp.loc[coreindx,'Pccurve_norm']
    xvalues=pccurve[:,0]
    yvalues=pccurve[:,1]
    try:
        popt, peptt = curve_fit(cappfuncmultsent, xvalues, yvalues)
        a,b,c,d = popt
        y_new = cappfuncmultsent(xvalues, a,b,c,d)
        plt.scatter(xvalues, yvalues)
        plt.plot(xvalues, y_new)
        dfcp.loc[coreindx,'PCparameters']=popt
    except:
        dfcp.loc[coreindx,'PCparameters']="ERROR"
        
        
        
        
dfcp.to_pickle("D:/PhD/Studies/Papers/MICP/Dataset/DS_1/micp.pk1")
         