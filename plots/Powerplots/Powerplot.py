import numpy as np
import corner
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

#======================================================================================================================#

path = '/MCMC/Power_sampler/'
n_ion , R_mfp , NoH = np.loadtxt(path+"Powerspectrum_THANN_.out", usecols = (0,1,2), unpack = True)
#======= n_ion,mean free path and no- of halos =====#
Mh = 1.087*1.e8*NoH
logL = np.loadtxt(path+"Powerspectrum_THANN_prob.out", usecols = (0), unpack = True)     #=== loglikelyhood =====#

#======================================================================================================================#


a=[]
a = np.c_[n_ion,R_mfp,NoH]
#sns.set_style('darkgrid')
#fig = corner.corner(a,bins=100,weights = -logL,color='purple',truth_color='blue',labels = ['$\zeta$','$R_{mfp}$','$Mh_{min}$'])
#fig.suptitle("$\zeta$ $h$,$R_{mfp}$,$Mh_{min}$  ")
#fig.savefig('corner_plot.png')



#======================================================================================================================#
plt.figure(figsize=(15,5))
plt.subplot(131)
plt.ylabel('freq.')
plt.xlabel('$\zeta$')
plt.hist(n_ion,bins=50,color='pink')

plt.subplot(132)
plt.ylabel('freq.')
plt.xlabel('$R_{mfp}$')
plt.hist(R_mfp,bins=50,color='orange')
plt.subplot(133)
plt.xlabel('$Mh_{min}$')
plt.ylabel('$freq$')
plt.hist(NoH,bins = 50,color='blue')

plt.savefig('hist_params.png')

