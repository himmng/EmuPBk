import numpy as np
path = '/home/ht/EmuPBk/data/data_Powerspectrum/'
pk = np.loadtxt(path+'pk_test')
params = np.loadtxt(path+'params_test')
cov_inv = np.loadtxt(path+'cov_inv_test')

print(cov_inv)