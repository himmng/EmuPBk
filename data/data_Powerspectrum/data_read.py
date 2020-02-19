import numpy as np
#================================================================================================================#
z = 9.210                                               #================ redshift used in the simulation============#
h = 0.6704
Omega_b = 0.049
Omega_m = 0.3183                                     #==============matter density parameter values used in simulation====#
Omega_bh2 = Omega_b*(h**2)
Omega_mh2 = Omega_m*(h**2)
#================================================================================================================#

pk = np.loadtxt('norm_pk')                        #====loading the normalized powerspectrum(units = mk^2)=======================#

Nk_bins = np.loadtxt('nkbins',)                  #=========K-Bins corresponding to each powerspectrum=#

k = np.loadtxt('k.txt')                          #============K values (identical for each powerspectrum)====#

params = np.loadtxt('params',)                   #============parameters saved as (n_ion,R_mfp,NoH)=======#

x_HI = np.loadtxt('neutral_frac',)            #======================== x_HI (neutral fraction)====#

'''
dT_b = 27.0 * x_HI * (Omega_bh2 / 0.023) * np.power(((0.15 * (1. + z)) / (10. * Omega_mh2)), 0.50)
                                                             '''#==========brightness temperature constrast======#

#================================================================================================================#
'''The covarience matrix only contains diagonal terms, flattening it will make no difference in the dot-product 
during likelihood evaluation'''

cov = pk/(Nk_bins**(1./2.))      #=====Error in powerspectrum(for the covarience matrix)=========#

cov_inv = 1./cov
np.savetxt('cov_inverse.txt',cov_inv)
#================================================================================================================#

