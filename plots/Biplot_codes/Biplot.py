import corner
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

#======================================================================================================================#
id = 17
path = '/home/ht/PycharmProjects/EmuPBk/plots/results & plots/Bk_results/K0.3/%d/'%id
#======================================================================================================================#

n_ion , R_mfp , NoH = np.loadtxt(path+"Bk.out", usecols = (0,1,2), unpack = True)
#======= n_ion,mean free path and no- of halos =====#
Mh = 1.087*1.e8*NoH
logL = np.loadtxt(path+"Bkprob.out", usecols = (0), unpack = True)     #=== loglikelyhood =====#

#======================================================================================================================#


a=[]
a = np.c_[n_ion,R_mfp,NoH]
sns.set_style('darkgrid')
fig = corner.corner(a,bins=100,weights = -logL,color='purple',truth_color='blue',labels = ['$\zeta$','$R_{mfp}$','$Mh_{min}$'])
fig.suptitle("$\zeta$ $h$,$R_{mfp}$,$Mh_{min}$  ")
fig.savefig('corner_plot.png')



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

