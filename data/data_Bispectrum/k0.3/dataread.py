import numpy as np
import matplotlib.pyplot as plt

Bk = np.loadtxt('bk_norm03')
params = np.loadtxt('params03')
nbins = np.loadtxt('nbins03')
Bk =  np.around(Bk,1)
ind= np.loadtxt('index')
index = np.zeros(len(ind),dtype=int)
for i in range(len(ind)):
    index[i] = ind[i]
nbins_test = nbins[index]

Bk_03_test = Bk[index]

Bk_03_train = np.delete(Bk,index,axis=0)

Bk_03_train = Bk_03_train

params_test = params[index]

params_train = np.delete(params,index,axis=0)

params_train = params_train


