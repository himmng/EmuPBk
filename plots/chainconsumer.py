from chainconsumer import ChainConsumer
import numpy as np
import pandas as pd
np.random.seed(0)
#data = np.random.multivariate_normal([0.0, 4.0], [[1.0, 0.7], [0.7, 1.5]], size=1000000)
path = '/home/ht/PycharmProjects/THANN/'
n_ion , R_mfp , NoH = np.loadtxt(path+"Powerspectrum_THANN_.out", usecols = (0,1,2), unpack = True)
#======= n_ion,mean free path and no- of halos =====#
#Mh = 1.087*1.e8*NoH
Mh = 1.087*NoH
logL = np.loadtxt(path+"Powerspectrum_THANN_prob.out", usecols = (0), unpack = True)     #=== loglikelyhood =====#
#df = pd.DataFrame({'$\zeta$':n_ion,'$Rmfp$':R_mfp,'$Mhalo_{min}(10^8$ $M_\odot)$':NoH})
data = [n_ion , R_mfp ,NoH ]

fig = ChainConsumer().add_chain(data,parameters=["$\zeta$", "$R_{mfp}$", "$Mh_{min}$"]).plotter.plot()

fig.set_size_inches(3 + fig.get_size_inches())  # Resize fig for doco. You don't need
fig.savefig('plot_nion_{0:.3f}_R_mfp_{1:.3f}_Mhalo_{2:.3f}.png'.format(n_ion.mean(),R_mfp.mean(),Mh.mean()),dpi=400)
print('plot_nion_{0:.3f}_R_mfp_{1:.3f}_Mhalo_{2:.3f}.png'.format(n_ion.mean(),R_mfp.mean(),Mh.mean()))