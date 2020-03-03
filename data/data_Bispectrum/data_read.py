import numpy as np

path = '/home/ht/EmuPBk/data/data_Bispectrum/k0.2/'

N_bins = np.loadtxt('nbins02')

Bk = np.loadtxt('bk_norm02')

params = np.loadtxt('params02')

sigma = Bk/np.sqrt(N_bins)
cov_inv = 1./sigma

cov = np.zeros(shape=(1058,550,550))
for i in range(1058):
    for j in range(550):
        cov[i][j][j] = cov_inv[i][j]

ind = np.loadtxt('index')
index = np.zeros(len(ind),dtype=int)
for i in range(len(ind)):
    index[i] = int(ind[i])

Bk_test = Bk[index]
params_test = params[index]
cov_inv_test = cov[index]






'''The covarience matrix:
* It will contain the sample varience + Instrumental noise from SKA low telescope array.
'''
cov = np.zeros(shape=(Bk_02.shape))
cov_inv = 1./cov
