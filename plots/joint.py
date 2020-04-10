from chainconsumer import ChainConsumer
import numpy  as np
id = 12
path1 = '/home/ht/PycharmProjects/EmuPBk/plots/results/Pk_results/%d/'%id
path2 = '/home/ht/PycharmProjects/EmuPBk/plots/results/Bk_results/%d/'%id

n_ionp , R_mfpp , NoHp = np.loadtxt(path1+"Pk_.out", usecols = (0,1,2), unpack = True)
logLp = np.loadtxt(path1+"Pk_prob.out", usecols = (0), unpack = True)

n_ionb , R_mfpb , NoHb = np.loadtxt(path2+"Bk.out", usecols = (0,1,2), unpack = True)
logLb = np.loadtxt(path2+"Bkprob.out", usecols = (0), unpack = True)

Mhp = 1.087*NoHp
Mhb = 1.087*NoHb

datap = [n_ionp,R_mfpp,Mhp]

datab = [n_ionb,R_mfpb,Mhb]

fig = ChainConsumer().add_chain(datap, parameters=
["$\zeta$", "$R_{mfp}$", r"$Mhalo_{min}(10^8$ $M_\odot)$"], name="Powerspectrum",color='blue')\
    .add_chain(datab, name="Bispectrum",color='red').plotter.plot()

fig.set_size_inches(3 + fig.get_size_inches())
fig.savefig('plot_nion_{0:.3f}_R_mfp_{1:.3f}_Mhalo_{2:.3f}.png'.format(n_ionb.mean(),R_mfpb.mean(),Mhb.mean()),dpi=100)
