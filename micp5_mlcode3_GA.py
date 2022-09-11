# -*- coding: utf-8 -*-
"""
Created on Sat Apr 10 13:22:30 2021

@author: j
"""
from sklearn import svm
import pandas as pd
import numpy as np
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize
import tensorflow as tf
import tensorflow_probability as tfp

#import pyswarms as pso
#from pyswarms.utils.functions import single_obj as fx
import pyswarms as ps
import sklearn.metrics 
from averagePc import averagePc
from averagePc import rsaturation


dfcp = pd.read_pickle("D:/PhD/Studies/Papers/MICP/Dataset/DS_1/micpf.pk1")
dfcpsize=dfcp.shape

dfcp.to_csv("D:/PhD/Studies/Papers/MICP/Dataset/DS_1/micp.csv")

irangec=np.linspace(0,20,0,dtype='int')
idatacNUM=len(irangec)

# idatacor=["Porosity","fractal","Purcell","Swanson","Winland"]
idatacor=["Porosity","Winland","fractal","Swanson"]

dfcp.Winland=np.log10(dfcp.Winland.to_numpy(dtype='float64')/9.869233e-15)
dfcp.fractal=np.log10(dfcp.fractal.to_numpy(dtype='float64')/9.869233e-15)
dfcp.Swanson=np.log10(dfcp.Swanson.to_numpy(dtype='float64')/9.869233e-15)
dfcp.Purcell=np.log10(dfcp.Purcell.to_numpy(dtype='float64')/9.869233e-15)


idata=np.zeros((dfcpsize[0],5))
rdata=np.zeros((dfcpsize[0],1))
idatacorlist=np.zeros([dfcp.shape[0],len(idatacor)],dtype='float64')
idatapclist=np.zeros([dfcp.shape[0],idatacNUM*1],dtype='float64')

rdata= np.log10(dfcp.Permeability.to_numpy(dtype='float32')/9.869233e-15)
avgpcl=(averagePc(dfcp))/100
rsaturation20=np.log10(rsaturation(dfcp,0.20)/9.869233e-15)
rsaturation80=np.log10(rsaturation(dfcp,0.80)/9.869233e-15)
rsaturation50=np.log10(rsaturation(dfcp,0.40)/9.869233e-15)

cornom=-1
for coreindx in dfcp.index: 
   cornom = cornom+1 
   popt=dfcp.loc[coreindx,'PCparameters']
   datapc=  ((popt[2,irangec].flatten()) )
   # datapc=  np.concatenate((popt[1,irangec].flatten() ,popt[2,irangec].flatten()) )
   idatapclist[cornom,:]=datapc
   idatacorlist[cornom,:]=dfcp.loc[coreindx,idatacor].to_numpy(dtype='float64')
   if popt=="ERROR":
       print(coreindx)
       continue
   # a,b,c=np.log10(popt)
   # idata[coreindx,0]=dfcp.loc[coreindx,'Porosity'] 
   # idata[coreindx,1]=dfcp.loc[coreindx,'Winland']                 
   # idata[coreindx,[2,3,4]]=a,b,c
   # rdata[coreindx,0]=dfcp.loc[coreindx,'Permeability']

idata=np.concatenate((idatapclist,idatacorlist),axis=1)
idata=np.concatenate((idata,avgpcl),axis=1)
idata=np.concatenate((idata,rsaturation20),axis=1)
idata=np.concatenate((idata,rsaturation80),axis=1)
# idata=np.concatenate((idata,rsaturation50),axis=1)

# Data Comparison:
x_train, x_test, y_train, y_test = train_test_split((idata), (rdata),test_size=0.2, random_state=10)

# Neural Network:
regrNN = MLPRegressor(hidden_layer_sizes=(70,80,55),random_state=10,solver="lbfgs",learning_rate="adaptive",alpha=0.0001,verbose=True, activation= "identity", max_iter=3000).fit(x_train, y_train)
y_predNN=regrNN.predict(x_test)
scoreNN=regrNN.score(x_test, y_test)
print(scoreNN)
mseNNreg=sklearn.metrics.mean_absolute_percentage_error((y_predNN),(y_test))
print("MSE NN:"+ str(mseNNreg))
fig, ax1=plt.subplots(1,1)
ax1.scatter(y_test, y_predNN, color="blue")



# PSO Algorithm:
maxbound=np.ones(3)
#[20,1000,5]
minbound=np.zeros(3)
bounds=(minbound,maxbound)
# Initialize swarm
options = {'c1': 0.9, 'c2': 0.8, 'w':0.9}

optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=3, options=options, bounds=bounds)






population_size = 40
 # With an initial population and a multi-part state.
initial_population = (tf.random.normal([population_size]),
                      tf.random.normal([population_size]))


def f1ga(c,ga,eps):
    xc=c
    xga=ga
    xeps=eps
    regr = svm.SVR(kernel='rbf', C=xc, gamma=xga, epsilon=xeps)
    regr.fit(x_train, y_train)
    y_pred=regr.predict(x_test)
    scoreReg=1 -regr.score(x_test, y_test)
    return scoreReg



optim_results = tfp.optimizer.differential_evolution_minimize(
    f1ga,
    initial_population=initial_population,
    seed=43210)

print(optim_results.converged)
print(optim_results.position)  # Should be (close to) [pi, pi].
print(optim_results.objective_value)    # Should be -1.


# With a single starting point
initial_position = (tf.constant(1.0), tf.constant(1.0))

optim_results = tfp.optimizer.differential_evolution_minimize(
    easom_fn,
    initial_position=initial_position,
    population_size=40,
    population_stddev=2.0,
    seed=43210)










# SVM Method:

#Optimization:
x0=[12,0.01,9.5]


def fpso(c,ga=10,ep=1):
    scoreReg=np.zeros(10)    
    for xx in  range(10):
        xc=c[xx,0]
        xga=c[xx,1]
        xeps=c[xx,2]
        regr = svm.SVR(kernel='rbf', C=xc, gamma=xga, epsilon=xeps)
        regr.fit(x_train, y_train)
        scoreReg[xx]=1 -regr.score(x_test, y_test)
    
    return scoreReg


kwargs={"ga":x0[1],"ep":x0[2]}
cost, pos = optimizer.optimize(fpso, 700)

regrPSO = svm.SVR(kernel='rbf', C=pos[0], gamma=pos[1], epsilon=pos[2])
regrPSO.fit(x_train, y_train)
ScoreTrainPSO=regrPSO.score(x_train, y_train)
y_predPSO=regrPSO.predict(x_test)
scoreRegPSO=regrPSO.score(x_test, y_test)


msePSOSVM=sklearn.metrics.mean_absolute_percentage_error(y_predPSO,y_test)
print("MSE SVM:"+ str(msePSOSVM))

def f1(c):
    xc=c[0]
    xga=1
    xeps=1
    regr = svm.SVR(kernel='rbf', C=xc, gamma=xga, epsilon=xeps)
    regr.fit(x_train, y_train)
    y_pred=regr.predict(x_test)
    scoreReg=1 -regr.score(x_test, y_test)
    return scoreReg

res = minimize(f1, x0,  method='nelder-mead', options={'xatol': 1e-8, 'disp': True})
print(res.x)

regr = svm.SVR(kernel='rbf', C=res.x[0], gamma=res.x[1], epsilon=res.x[2])
regr.fit(x_train, y_train)
y_pred=regr.predict(x_test)
scoreRegT=regr.score(x_train, y_train)
scoreRegTest=regr.score(x_test, y_test)

print([scoreRegT,scoreRegTest])




# xideal=[min(rdata),max(rdata)]
# yideal= [min(rdata),max(rdata)]

fig, ax2=plt.subplots(1,1)
ax2.scatter((y_test), (y_pred), color="red")
ax2.scatter((y_test), (y_predPSO), color="green")

#ax2.scatter((y_test), (y_predNN), color="green")
# ax2.plot((xideal), (yideal), color="black")
ax2.set_xlabel("K Measured"), ax2.set_ylabel("K Prediction")
#ax2.set_xscale('log'), ax2.set_yscale('log')

ax2.set_title("R2 SVM="  + str(scoreRegT) + ";  R2 SVM PSO:"+ str(scoreRegPSO) + ";  R2 NN=" +str(scoreNN))
#ax2.set_yscale('log')
#ax2.set_xscale('log')



