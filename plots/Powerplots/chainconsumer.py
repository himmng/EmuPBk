from chainconsumer import ChainConsumer
import numpy as np
import pandas as pd

path = '/MCMC/Power_sampler/'
n_ion , R_mfp , NoH = np.loadtxt(path+"Powerspectrum_THANN_.out",
                                 usecols = (0,1,2), unpack = True) #======= n_ion,mean free path and no- of halos =====#

Mh = 1.087*NoH #====================Mh = 1.087*1.e8*NoH

logL = np.loadtxt(path+"Powerspectrum_THANN_prob.out", usecols = (0), unpack = True)     #=== loglikelyhood =====#

#df = pd.DataFrame({'$\zeta$':n_ion,'$Rmfp$':R_mfp,'$Mhalo_{min}(10^8$ $M_\odot)$':NoH})

data = [n_ion , R_mfp ,NoH ]

fig = ChainConsumer().add_chain(data,parameters=["$\zeta$", "$R_{mfp}$", "$Mh_{min}$"]).plotter.plot()

fig.set_size_inches(3 + fig.get_size_inches())  # Resize fig for doco. You don't need
fig.savefig('plot_nion_{0:.3f}_R_mfp_{1:.3f}_Mhalo_{2:.3f}.png'.format(n_ion.mean(),R_mfp.mean(),Mh.mean()),dpi=400)
print('plot_nion_{0:.3f}_R_mfp_{1:.3f}_Mhalo_{2:.3f}.png'.format(n_ion.mean(),R_mfp.mean(),Mh.mean()))

'''
plt.subplot(312)
plt.ylabel('freq.')
plt.xlabel('$R_{mfp}$')
plt.hist(R_mfp,bins=100,color='orange')
plt.axvline(R_mfp.mean(),color='green',lw=2)
plt.subplot(313)
plt.xlabel('$Mh_{min}$')
plt.ylabel('$freq$')

plt.hist(Mh,bins = 100,color='blue',alpha=0.5)
a=[]
a = np.c_[n_ion,R_mfp,Mh]
fig = corner.corner(a,bins=55,color='b',weights = -logL,labels = ['$\zeta$','$R_{mfp}$','$Mh_{min}$'])
fig.suptitle("$\zeta$ $h$,$R_{mfp}$,$Mh_{min}$  ")
fig.savefig('corner_plot.png')

data_Powerspectrum = pd.DataFrame({'$\zeta$':n_ion,'$R_{mfp}$':R_mfp,'$Mh_{min}$':Mh})
#print(data_Powerspectrum['$h$'].describe(),data_Powerspectrum['$\Omega_m$'].describe())

'''