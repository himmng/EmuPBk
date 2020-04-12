import chainconsumer as cs
import numpy as np
k = 0.2
id = 15
path = '/home/ht/PycharmProjects/EmuPBk/plots/results & plots/Bk_results/K{0:.1f}/{1:d}/'.format(k,id)

n_ion , R_mfp , NoH = np.loadtxt(path+"Bk.out", usecols = (0,1,2), unpack = True)
Mh = 1.087*NoH

logL = np.loadtxt(path+"Bkprob.out", usecols = (0), unpack = True)

data = [n_ion, R_mfp,NoH,logL]



c = cs.ChainConsumer().add_chain(data,parameters=["$\zeta$", "$R_{mfp}$", "$Mh_{min}$","logL"])
c.configure(color_params="logL",cmap='Blues_r',cmaps=['Blues'])
fig = c.plotter.plot(figsize=1.0)

fig.set_size_inches(3 + fig.get_size_inches())  # Resize fig for doco. You don't need this.

fig.savefig(path+'plot_nion_{0:.3f}_R_mfp_{1:.3f}_Mhalo_{2:.3f}.png'.format(n_ion.mean(),R_mfp.mean(),Mh.mean()),dpi=100)
print('plot_nion_{0:.3f}_R_mfp_{1:.3f}_Mhalo_{2:.3f}.png'.format(n_ion.mean(),R_mfp.mean(),Mh.mean()))

