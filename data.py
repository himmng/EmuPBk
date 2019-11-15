import numpy as np
from sklearn.model_selection import train_test_split
path = '/home/ht/PycharmProjects/THANN/data/'                       #============Directory for the data files ===================#

#================================================================================================================#
z = 9.210  #================ redshift used for the Emulation=====================================================#
#================================================================================================================#

pk = np.loadtxt(path+'powerspectrum.txt', unpack=True)
pk = pk.T                                                   #====loading the powerspectrum=======================#
pk = np.delete(pk, [0, 1, 9], 1)
pk = pk

#================================================================================================================#

Nk_bins = np.loadtxt(path+'no_of_bins.txt', unpack=True)
Nk_bins = Nk_bins.T                                         #=========K-Bins corresponding to each powerspectrum=#
Nk_bins = np.delete(Nk_bins, [0, 1, 9], 1)

#================================================================================================================#

k = np.loadtxt(path+'k.txt', unpack=True)
k = np.delete(k, [0, 1, 9])                          #============K values (identical for each powerspectrum)====#
k = k

#================================================================================================================#

params = np.loadtxt(path+'params.txt', unpack=True)
params = params.T                                       #============parameters saved as (n_ion,R_mfp,NoH)=======#

#================================================================================================================#

x_HI = np.loadtxt(path+'neutral_frac.txt', unpack=True)
x_HI = x_HI/1000.                                           #======================== x_HI (neutral fraction)====#
x_HI = x_HI.reshape(len(x_HI), 1)

#================================================================================================================#

index = [800,854,908,953,962]                       #=====cleaning data (removing NaN values)=====================#
params = np.delete(params,index,axis = 0)
params = params
pk = np.delete(pk,index,axis= 0)
pk = pk
x_HI = np.delete(x_HI,index,axis = 0)
x_HI =x_HI
Nk_bins = np.delete(Nk_bins,index,axis = 0)
Nk_bins = Nk_bins

#================================================================================================================#

Omega_bh2 = 0.0224
Omega_mh2 = 0.1424                                      #==============present day values Omega_mh2,Omega_bh2====#

#================================================================================================================#

dT_b = 27.0 * x_HI * (Omega_bh2 / 0.023) * np.power(((0.15 * (1. + z)) / (10. * Omega_mh2)), 0.50)
                                                                #==========brightness temperature constrast======#
#================================================================================================================#

PK = np.zeros(shape=pk.shape)
for i in range(len(pk[:])):
    PK[i] = (k ** 3 * pk[i]) / (2 * (np.pi ** 2))  #======Powespectrum==================================#
    PK[i] = PK[i] * (dT_b[i] ** 2)

#================================================================================================================#

cov = PK/(Nk_bins**(1./2.))      #=====Error in powerspectrum(for the covarience matrix)=========#


index =[14, 15, 19, 20, 21, 22, 23, 24, 25, 26]#int(input('give the value of pk_index(between 0 to 10):',)))
#index = np.random.randint(0,1000,50)
#================================================================================================================#
cov_inv = np.zeros(shape=(7,7))
ind = int(input('invex:',))
for i in range(7):
    cov_inv[i,i] = 1./cov[index[ind]][i]
#cov_inv =
#print(np.where(params[:,0]>180))