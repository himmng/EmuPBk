import numpy as np
from chainconsumer import ChainConsumer

id = 0
path = '/home/ht/PycharmProjects/EmuPBk/plots/results & plots/Pk_results/%d/'%id
n_ion , R_mfp , NoH = np.loadtxt(path+"Pk_.out", usecols = (0,1,2), unpack = True)
Mh = 1.087*NoH

logL = np.loadtxt(path+"Pk_prob.out", usecols = (0), unpack = True)




data = [n_ion , R_mfp ,NoH ]

fig = ChainConsumer().add_chain(data,parameters=["$\zeta$", "$R_{mfp}$", "$Mh_{min}$"]).plotter.plot()

fig.set_size_inches(3 + fig.get_size_inches())  # Resize fig for doco. You don't need
fig.savefig('plot_nion_{0:.3f}_R_mfp_{1:.3f}_Mhalo_{2:.3f}.png'.format(n_ion.mean(),R_mfp.mean(),Mh.mean()),dpi=100)
print('plot_nion_{0:.3f}_R_mfp_{1:.3f}_Mhalo_{2:.3f}.png'.format(n_ion.mean(),R_mfp.mean(),Mh.mean()))

