import numpy as np
#================================================================================================================#
z = 9.210  #================ redshift used in the simulation============#
#================================================================================================================#

pk = np.loadtxt('powerspectrum.txt')     #====loading the powerspectrum=======================#

pk = np.delete(pk, [0, 1, 9], axis = 1)
pk = pk



#================================================================================================================#

Nk_bins = np.loadtxt('no_of_bins.txt',)                                     #=========K-Bins corresponding to each powerspectrum=#
Nk_bins = np.delete(Nk_bins, [0, 1, 9], axis = 1)

#================================================================================================================#

k = np.loadtxt('k.txt',)
k = np.delete(k, [0, 1, 9],)                          #============K values (identical for each powerspectrum)====#
k = k

#================================================================================================================#

params = np.loadtxt('params.txt',)                                     #============parameters saved as (n_ion,R_mfp,NoH)=======#

#================================================================================================================#

x_HI = np.loadtxt('neutral_frac.txt',)              #=====values are 1000 times scaled need to be rescaled ======#
x_HI = x_HI/1000.                                           #======================== x_HI (neutral fraction)====#
x_HI = x_HI.reshape(len(x_HI), 1)

#================================================================================================================#

nan = [800,854,908,953,962]                       #=====cleaning data (removing NaN values)=====================#
params = np.delete(params,nan,axis = 0)
params = params
pk = np.delete(pk,nan,axis= 0)
pk = pk
x_HI = np.delete(x_HI,nan,axis = 0)
x_HI =x_HI
Nk_bins = np.delete(Nk_bins,nan,axis = 0)
Nk_bins = Nk_bins

#================================================================================================================#

Omega_bh2 = 0.0224
Omega_mh2 = 0.1424                                      #==============present day values Omega_mh2,Omega_bh2====#

#================================================================================================================#

dT_b = 27.0 * x_HI * (Omega_bh2 / 0.023) * np.power(((0.15 * (1. + z)) / (10. * Omega_mh2)), 0.50)
                                                                #==========brightness temperature constrast======#
#================================================================================================================#

pk = (k ** 3 * pk) / (2 * (np.pi ** 2))  #======Powespectrum==================================#
pk = pk * (dT_b ** 2)


#================================================================================================================#

cov = pk/(Nk_bins**(1./2.))      #=====Error in powerspectrum(for the covarience matrix)=========#

cov_inv = 1./cov
np.savetxt('cov_inverse.txt',cov_inv)
#================================================================================================================#

np.savetxt('pk_dTb2.txt',pk)
np.savetxt('x_H1_updated.txt',x_HI)
np.savetxt('params_updated.txt',params)
np.savetxt('k_updated.txt',k)
np.savetxt('Nk_bins_updated.txt',Nk_bins)