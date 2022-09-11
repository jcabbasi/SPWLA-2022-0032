# -*- coding: utf-8 -*-
"""
Created on Sat Jul  3 12:53:47 2021

@author: j
"""
# ERROR CALCULATION

import pandas as pd
import numpy as np
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import sklearn.metrics 
from scipy import stats




dfcp = pd.read_pickle("D:/PhD/Studies/Papers/MICP/Dataset/DS_1/micpSS.pk1")


#
dferr = pd.DataFrame(index=range(20), columns=['Algorithm','R2train','R2test',"MAPEtrain","MAPEtest","MAEtrain","MAEtest" ])
dferr.at[0, 'Algorithm']="PoroPerm"
dferr.at[1, 'Algorithm']="Winland"
dferr.at[2, 'Algorithm']="fractal"
dferr.at[3, 'Algorithm']="Swanson"
dferr.at[4, 'Algorithm']="Purcell"
dferr.at[5, 'Algorithm']="Parachor"
dferr.at[6, 'Algorithm']="ANN"
dferr.at[7, 'Algorithm']="SVMPSO"
dferr.at[8, 'Algorithm']="SVMReg"






## POROSITY PERMEABILITY CURVE 

xdir2=np.empty(0)
xphi=dfcp.Porosity.to_numpy(dtype ='float32')
yper=dfcp.Permeability.to_numpy(dtype ='float32')/9.869233e-15
m,b=np.polyfit(xphi,np.log10(yper),1)
ynew=m*xphi+b

yperlog=np.log10(yper)
ynewlog=np.log10(ynew)

fig, axporo=plt.subplots(nrows=1,ncols=1)
axporo.set_xscale('log')
axporo.set_yscale('log')
axporo.scatter(yper,np.power(10,ynew),color="midnightblue",s=250,alpha=0.6)



# Perms: Correlation
yWinland= ( dfcp.Winland.to_numpy(dtype ='float32')/9.869233e-15 )
yfractal= ( dfcp.fractal.to_numpy(dtype ='float32')/9.869233e-15 )
ySwanson= ( dfcp.Swanson.to_numpy(dtype ='float32')/9.869233e-15 )
yPurcell= ( dfcp.Purcell.to_numpy(dtype ='float32')/9.869233e-15 )
yParachor=( dfcp.Parachor.to_numpy(dtype ='float32')/9.869233e-15 )






sklearn.metrics.r2_score(yper,ynew)

mses=np.zeros([1,6])
mses[0,0]=np.log10(sklearn.metrics.mean_absolute_percentage_error(yper,np.power(10,ynew)))
mses[0,1]=(sklearn.metrics.mean_absolute_percentage_error(yperlog,np.log10(yWinland)))
mses[0,2]=(sklearn.metrics.mean_absolute_percentage_error(yperlog,np.log10(yfractal)))
mses[0,3]=(sklearn.metrics.mean_absolute_percentage_error(yperlog,np.log10(ySwanson)))
mses[0,4]=(sklearn.metrics.mean_absolute_percentage_error(yperlog,np.log10(yPurcell)))
mses[0,5]=(sklearn.metrics.mean_absolute_percentage_error(yperlog,np.log10(yParachor)))

dferr.at[dferr.Algorithm=="PoroPerm", 'MAPE']=mses[0,0]
dferr.at[dferr.Algorithm=="Winland", 'MAPE']=mses[0,1]
dferr.at[dferr.Algorithm=="fractal", 'MAPE']=mses[0,2]
dferr.at[dferr.Algorithm=="Swanson", 'MAPE']=mses[0,3]
dferr.at[dferr.Algorithm=="Purcell", 'MAPE']=mses[0,4]
dferr.at[dferr.Algorithm=="Parachor", 'MAPE']=mses[0,5]

tau, p_value = stats.kendalltau(np.log10(yper), xphi); print([tau,p_value])
tau, p_value = stats.kendalltau(np.log10(yper), yWinland); print([tau,p_value])
tau, p_value = stats.kendalltau(np.log10(yper), yfractal); print([tau,p_value])
tau, p_value = stats.kendalltau(np.log10(yper), ySwanson); print([tau,p_value])
tau, p_value = stats.kendalltau(np.log10(yper), yPurcell); print([tau,p_value])
tau, p_value = stats.kendalltau(np.log10(yper), yParachor); print([tau,p_value])


mserr=np.zeros([1,6])
mserr[0,0]=np.log10(sklearn.metrics.mean_squared_error(yper,np.power(10,ynew)))
mserr[0,1]=(sklearn.metrics.mean_squared_error(yperlog,np.log10(yWinland)))
mserr[0,2]=(sklearn.metrics.mean_squared_error(yperlog,np.log10(yfractal)))
mserr[0,3]=(sklearn.metrics.mean_squared_error(yperlog,np.log10(ySwanson)))
mserr[0,4]=(sklearn.metrics.mean_squared_error(yperlog,np.log10(yPurcell)))
mserr[0,5]=(sklearn.metrics.mean_squared_error(yperlog,np.log10(yParachor)))


rsqs=np.zeros([1,6])

rsqs[0,0]=(sklearn.metrics.mean_absolute_error(yper,np.power(10,ynew)))
rsqs[0,1]=(sklearn.metrics.mean_absolute_error(yperlog,np.log10(yWinland)))
rsqs[0,2]=(sklearn.metrics.mean_absolute_error(yperlog,np.log10(yfractal)))
rsqs[0,3]=(sklearn.metrics.mean_absolute_error(yperlog,np.log10(ySwanson)))
rsqs[0,4]=(sklearn.metrics.mean_absolute_error(yperlog,np.log10(yPurcell)))
rsqs[0,5]=(sklearn.metrics.mean_absolute_error(yperlog,np.log10(yParachor)))


dferr.at[dferr.Algorithm=="PoroPerm", 'MAE']=rsqs[0,0]
dferr.at[dferr.Algorithm=="Winland", 'MAE']=rsqs[0,1]
dferr.at[dferr.Algorithm=="fractal", 'MAE']=rsqs[0,2]
dferr.at[dferr.Algorithm=="Swanson", 'MAE']=rsqs[0,3]
dferr.at[dferr.Algorithm=="Purcell", 'MAE']=rsqs[0,4]
dferr.at[dferr.Algorithm=="Parachor", 'MAE']=rsqs[0,5]

dferr.at[dferr.Algorithm=="PoroPerm", 'R2train']=(sklearn.metrics.r2_score(yper,ynew))
dferr.at[dferr.Algorithm=="Winland", 'R2train']=(sklearn.metrics.r2_score(yper,yWinland))
dferr.at[dferr.Algorithm=="fractal", 'R2train']=(sklearn.metrics.r2_score(yper,yfractal))
dferr.at[dferr.Algorithm=="Swanson", 'R2train']=(sklearn.metrics.r2_score(yper,ySwanson))
dferr.at[dferr.Algorithm=="Purcell", 'R2train']=(sklearn.metrics.r2_score(yper,yPurcell))
dferr.at[dferr.Algorithm=="Parachor", 'R2train']=(sklearn.metrics.r2_score(yper,yPurcell))




#mean_absolute_percentage_error
plt.rcParams.update({'font.size': 20})
plt.rcParams['font.family'] = 'arial'

fig, ax3 = plt.subplots(nrows=1,ncols=2)
ax3[0].bar(['Phi-Perm','Winland','Fractal','Swanson','Purcell','Parachor'],(rsqs[0,0:6]), width = 0.8,color='blue',alpha=0.7)
ax3[0].set_xlabel("Correlation"), ax3[0].set_ylabel("Mean Absolute Error (mD)")
# ax3[0].set_yscale('log')
ax3[1].bar(['Phi-Perm','Winland','Fractal','Swanson','Purcell','Parachor'],((mserr[0,0:6])), width = 0.8,color='blue',alpha=0.7)
ax3[1].set_xlabel("Correlation"), ax3[1].set_ylabel("Mean Squared Error (mD) ")
# ax3[1].set_yscale('log')



fig, ax4 = plt.subplots(nrows=1,ncols=2)
ax4[0].bar(['Phi-Perm','Winland',"PSO-SVM","SVM","ANN"],([1.27,10.1,0.34,1.23,0.4]), width = 0.8,color='blue',alpha=0.7)
ax4[0].set_xlabel("Algorithm"), ax4[0].set_ylabel("MAPE ")
ax4[0].set_yscale('log')
ax4[1].bar(['Phi-Perm','Winland','Fractal','Swanson','Purcell','Parachor'],((rsqs[0,0:6])), width = 0.8,color='blue',alpha=0.7)
ax4[1].set_xlabel("Algorithm"), ax4[1].set_ylabel("Mean Absolute Error ")
ax4[1].set_yscale('log')







