import numpy as np

path = '/home/ht/EmuPBk/data/data_Bispectrum/k0.2/'

N_bins = np.loadtxt(path+'nbins02')

Bk = np.loadtxt(path+'bk_norm02')

params = np.loadtxt(path+'params02')


ind = np.loadtxt(path+'index')
index = np.zeros(len(ind),dtype=int)
for i in range(len(ind)):
    index[i] = int(ind[i])

Bk_test = Bk[index]
params_test = params[index]
N_bins_test = N_bins[index]

cov_inv_test = np.zeros(shape=(len(N_bins_test),550,550))
for i in range(len(N_bins_test)):
    for j in range(550):
        cov_inv_test[i][j][j] = Bk_test[i][j]/np.sqrt(N_bins_test[i][j])

np.savetxt(path+'Bk_test',Bk_test)
np.savetxt(path+'params_test',params_test)
np.savetxt(path+'cov_inv_test',cov_inv_test[0])





