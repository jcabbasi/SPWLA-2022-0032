# -*- coding: utf-8 -*-
"""
Created on Sun Jun 20 09:38:46 2021

@author: j
"""

import seaborn as sns 
import pandas as pd
import numpy as np
sns.set(style="ticks", color_codes=True)

dfcpn = pd.read_pickle("D:/PhD/Studies/Papers/MICP/Dataset/DS_1/micpf.pk1")

huex="Miner"
dfcpnew=dfcpn[[ "Porosity", "Permeability", "fractal","Purcell","Winland"]]
dfcpnew.Permeability= np.log10(dfcpnew.Permeability.to_numpy(dtype ='float32'))
dfcpnew.fractal= np.log10(dfcpnew.fractal.to_numpy(dtype ='float32'))
dfcpnew.Winland= np.log10(dfcpnew.Winland.to_numpy(dtype ='float32'))
dfcpnew.Purcell= np.log10(dfcpnew.Purcell.to_numpy(dtype ='float32'))

g = sns.pairplot(dfcpnew,kind="scatter")
g.map_lower(sns.kdeplot, levels=4, color=".6")
