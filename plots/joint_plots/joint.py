from chainconsumer import ChainConsumer
import numpy  as np
id = 17
path1 = '/home/ht/PycharmProjects/EmuPBk/plots/results & plots/Pk_results/%d/'%id
path2 = '/home/ht/PycharmProjects/EmuPBk/plots/results & plots/Bk_results/K0.2/%d/'%id
path3 = '/home/ht/PycharmProjects/EmuPBk/plots/results & plots/Bk_results/K0.3/%d/'%id
#path4 = '/home/ht/PycharmProjects/EmuPBk/plots/results & plots/Bk_results/K1.5/%d/'%id

n_ionp , R_mfpp , NoHp = np.loadtxt(path1+"Pk_.out", usecols = (0,1,2), unpack = True)
logLp = np.loadtxt(path1+"Pk_prob.out", usecols = (0), unpack = True)
Mhp = 1.087*NoHp

n_ionb02 , R_mfpb02 , NoHb02 = np.loadtxt(path2+"Bk.out", usecols = (0,1,2), unpack = True)
logLb02 = np.loadtxt(path2+"Bkprob.out", usecols = (0), unpack = True)
Mhb02 = 1.087*NoHb02

n_ionb03 , R_mfpb03 , NoHb03 = np.loadtxt(path3+"Bk.out", usecols = (0,1,2), unpack = True)
logLb03 = np.loadtxt(path3+"Bkprob.out", usecols = (0), unpack = True)
Mhb03 = 1.087*NoHb03
'''
n_ionb15 , R_mfpb15 , NoHb15 = np.loadtxt(path4+"Bk.out", usecols = (0,1,2), unpack = True)
logLb15 = np.loadtxt(path4+"Bkprob.out", usecols = (0), unpack = True)
Mhb15 = 1.087*NoHb15
'''
datap = [n_ionp,R_mfpp,Mhp]

datab02 = [n_ionb02,R_mfpb02,Mhb02]

datab03 = [n_ionb03,R_mfpb03,Mhb03]
'''
datab15 = [n_ionb15,R_mfpb15,Mhb15]

fig = ChainConsumer().add_chain(datap, parameters=
["$\zeta$", "$R_{mfp}$", r"$Mhalo_{min}(10^8$ $M_\odot)$"], name="Powerspectrum",color='blue')\
    .add_chain(datab, name="Bispectrum",color='red').plotter.plot()
fig.set_size_inches(3 + fig.get_size_inches())
fig.savefig('jointplot_index{0:}.png'.format(id),dpi=100)
'''
fig = ChainConsumer().add_chain(datab02, parameters=
["$\zeta$", "$R_{mfp}$", r"$Mhalo_{min}(10^8$ $M_\odot)$"], name="Bispectrum for $K_1$ = 0.19$Mpc^{-1}$",color='blue')\
    .add_chain(datab03, name="Bispectrum for $K_1$ = 0.3$Mpc^{-1}$",color='red',).plotter.plot()

fig.set_size_inches(3 + fig.get_size_inches())
fig.savefig('jointplot_B0.2_vs._B1.5_index{0:}.png'.format(id),dpi=100)
