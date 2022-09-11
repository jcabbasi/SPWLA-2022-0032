# -*- coding: utf-8 -*-
"""
Created on Sat Jul 17 12:36:39 2021

@author: j
"""
from sklearn import svm
import pandas as pd
import numpy as np
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize
#import pyswarms as pso
#from pyswarms.utils.functions import single_obj as fx
import pyswarms as ps
from pyswarms.utils.plotters import (plot_cost_history, plot_contour, plot_surface)
import sklearn.metrics 
from averagePc import averagePc
from averagePc import rsaturation
from averagePc import satrange
from averagePc import satmin
from averagePc import pcmax
from averagePc import medianPc
from scipy import stats



dfcp = pd.read_pickle("D:/PhD/Studies/Papers/MICP/Dataset/DS_1/micpSS.pk1")


permMeasured=dfcp.Permeability.to_numpy(dtype='float64')/9.869233e-15
srange=satrange(dfcp)
smin=satmin(dfcp)
avgpcl=(averagePc(dfcp))/10 #MPa
pcmax=pcmax(dfcp)/1e6 #MPa
medianPc=medianPc(dfcp)/10 #MPa

tau, p_value = stats.kendalltau(np.log10(permMeasured), medianPc); print([tau,p_value])
tau, p_value = stats.kendalltau(np.log10(permMeasured), avgpcl); print([tau,p_value])
tau, p_value = stats.kendalltau(np.log10(permMeasured), pcmax); print([tau,p_value])
tau, p_value = stats.kendalltau(np.log10(permMeasured), srange); print([tau,p_value])



####


plt.rcParams.update({'font.size': 19})
plt.rcParams['font.family'] = 'arial'


fig, axall=plt.subplots(2,2)

# axall[0,0].scatter((y_test), (y_pred), color="red")
axall[0,0].scatter((permMeasured), (srange), color="blue",s=150,alpha=0.5,edgecolors="blue")
axall[0,0].set_xlabel("K Measured (mD)"), axall[0,0].set_ylabel("Saturation Range")
axall[0,0].set_xscale('log')

axall[1,0].scatter((permMeasured), (medianPc), color="blue",s=150,alpha=0.5,edgecolors="blue")
axall[1,0].set_xlabel("K Measured (mD)"), axall[1,0].set_ylabel("Median Pc (MPa)")
axall[1,0].set_xscale('log'), axall[1,0].set_yscale('log')

axall[0,1].scatter((permMeasured), (avgpcl), color="blue",s=150,alpha=0.5,edgecolors="blue")
axall[0,1].set_xlabel("K Measured (mD)"), axall[0,1].set_ylabel("Average Pc (MPa)")
axall[0,1].set_xscale('log'), axall[0,1].set_yscale('log')

axall[1,1].scatter((permMeasured), (pcmax), color="blue",s=150,alpha=0.5,edgecolors="blue")
axall[1,1].set_xlabel("K Measured (mD)"), axall[1,1].set_ylabel("Maximum Pc (MPa)")
axall[1,1].set_xscale('log'),axall[1,1].set_yscale('log')


