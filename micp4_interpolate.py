# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 12:47:34 2021

@author: j
"""

from scipy import interpolate
from scipy.interpolate import interp1d
from scipy.interpolate import Rbf, InterpolatedUnivariateSpline
import sklearn as svm
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

dfcp = pd.read_pickle("D:/PhD/Studies/Papers/MICP/Dataset/DS_1/micpSS.pk1")



for coreindx in dfcp.index:
   intvals=np.zeros([3,21])
   print(coreindx)
   pccurveN=dfcp.loc[coreindx,'Pccurve_norm']
   pccurve =dfcp.loc[coreindx,'Pccurve']
#   tck = interpolate.splrep(xvalues, yvalues, s=0)
   fnc = interp1d(pccurve[:,0], pccurve[:,1])
#   fncrbf=Rbf(xvalues,yvalues)
   xnew = np.linspace(pccurve[:,0].min(),pccurve[:,0].max(),21) 
   xnorm = np.linspace(0,1,21) 
   ynew1d=fnc(xnew)
   intvals[0,:]=xnorm
   intvals[1,:]=xnew
   intvals[2,:]=ynew1d/1e5
   dfcp.loc[coreindx,'PCparameters']=intvals


   
   
dfcp.to_pickle("D:/PhD/Studies/Papers/MICP/Dataset/DS_1/micpSS.pk1")
