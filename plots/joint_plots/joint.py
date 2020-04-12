from chainconsumer import ChainConsumer
import numpy  as np
id = 16
path1 = '/home/ht/PycharmProjects/EmuPBk/plots/results & plots/Pk_results/%d/'%id
path2 = '/home/ht/PycharmProjects/EmuPBk/plots/results & plots/Bk_results/K0.2/%d/'%id
path3 = '/home/ht/PycharmProjects/EmuPBk/plots/results & plots/Bk_results/K0.3/%d/'%id
path4 = '/home/ht/PycharmProjects/EmuPBk/plots/results & plots/Bk_results/K1.5/%d/'%id

n_ionp , R_mfpp , NoHp = np.loadtxt(path1+"Pk_.out", usecols = (0,1,2), unpack = True)
logLp = np.loadtxt(path1+"Pk_prob.out", usecols = (0), unpack = True)
Mhp = 1.087*NoHp

n_ionb03 , R_mfpb03 , NoHb03 = np.loadtxt(path3+"Bk.out", usecols = (0,1,2), unpack = True)
logLb03 = np.loadtxt(path3+"Bkprob.out", usecols = (0), unpack = True)
Mhb03 = 1.087*NoHb03


n_ionb02 , R_mfpb02 , NoHb02 = np.loadtxt(path2+"Bk.out", usecols = (0,1,2), unpack = True)
logLb02 = np.loadtxt(path2+"Bkprob.out", usecols = (0), unpack = True)
Mhb02 = 1.087*NoHb02


n_ionb15 , R_mfpb15 , NoHb15 = np.loadtxt(path4+"Bk.out", usecols = (0,1,2), unpack = True)
logLb15 = np.loadtxt(path4+"Bkprob.out", usecols = (0), unpack = True)
Mhb15 = 1.087*NoHb15
'''
datap = [n_ionp,R_mfpp,Mhp,logLp]
datab02 = [n_ionb02,R_mfpb02,Mhb02,logLb02]

datab03 = [n_ionb03,R_mfpb03,Mhb03,logLb03]

c = ChainConsumer().add_chain(datap,parameters=["$\zeta$", "$R_{mfp}$", "$Mh_{min}$","logLp"],
                                 name="Powerspectrum")
c.add_chain(datab02,parameters=["$\zeta$", "$R_{mfp}$", "$Mh_{min}$","logLb0.2"],
            name="Bispectrum_for_K=0.2")
c.add_chain(datab03,parameters=["$\zeta$", "$R_{mfp}$", "$Mh_{min}$","logLb0.3"],
            name="Bispectrum_for_K=0.3")


c.configure(color_params=["logLp","logLb0.2","logLb0.3"],cmaps=['Blues','Reds','cool']
            ,cmap='Blues_r')
fig = c.plotter.plot(figsize=1.0)

fig.set_size_inches(3 + fig.get_size_inches())

fig.savefig('Pk_vs_Bk2',dpi=100)


