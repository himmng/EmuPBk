from chainconsumer import ChainConsumer
import numpy  as np
id = 4
path1 = '/home/ht/PycharmProjects/EmuPBk/plots/results & plots/Pk_results/%d/'%id
path2 = '/home/ht/PycharmProjects/EmuPBk/plots/results & plots/Bk_results/K0.2/%d/'%id
path3 = '/home/ht/PycharmProjects/EmuPBk/plots/results & plots/Bk_results/K0.3/%d/'%id
#path4 = '/home/ht/PycharmProjects/EmuPBk/plots/results & plots/Bk_results/K1.5/%d/'%id

n_ionp , R_mfpp , NoHp = np.loadtxt(path1+"Pk_.out", usecols = (0,1,2), unpack = True)
logLp = np.loadtxt(path1+"Pk_prob.out", usecols = (0), unpack = True)
Mhp = 1.087*NoHp
datap = [n_ionp,R_mfpp,Mhp,logLp]



n_ionb02 , R_mfpb02 , NoHb02 = np.loadtxt(path2+"Bk.out", usecols = (0,1,2), unpack = True)
logLb02 = np.loadtxt(path2+"Bkprob.out", usecols = (0), unpack = True)
Mhb02 = 1.087*NoHb02
datab02 = [n_ionb02,R_mfpb02,Mhb02,logLb02]

n_ionb03 , R_mfpb03 , NoHb03 = np.loadtxt(path3+"Bk.out", usecols = (0,1,2), unpack = True)
logLb03 = np.loadtxt(path3+"Bkprob.out", usecols = (0), unpack = True)
Mhb03 = 1.087*NoHb03
datab03 = [n_ionb03,R_mfpb03,Mhb03,logLb03]
'''
n_ionb15 , R_mfpb15 , NoHb15 = np.loadtxt(path4+"Bk.out", usecols = (0,1,2), unpack = True)
logLb15 = np.loadtxt(path4+"Bkprob.out", usecols = (0), unpack = True)
Mhb15 = 1.087*NoHb15
datab15 = [n_ionb15,R_mfpb15,Mhb15,logLb15]
'''



c = ChainConsumer().add_chain(datap,parameters=["$\zeta$", "$R_{mfp}$", "$Mh_{min}$",'logLp'],
                                 name="$Powerspectrum$",color='green')

c.add_chain(datab02,parameters=["$\zeta$", "$R_{mfp}$", "$Mh_{min}$",'logLB02'],
                                 name="$Bispectrum(K_1=0.2Mpc^{-1},K_2,k_3)$",color='blue')

c.add_chain(datab03,parameters=["$\zeta$", "$R_{mfp}$", "$Mh_{min}$",'logLB03'],
            name="$Bispectrum(K_1=0.3Mpc^{-1},K_2,K_3)$",color='red')

c.configure(shade_alpha=0.2,color_params=['logLp','logLB02','logLB03'],cmaps=['Reds','GnBu','cool_r'])
fig = c.plotter.plot(figsize=1.0,)

fig.set_size_inches(3 + fig.get_size_inches(),)

fig.savefig('Pk_Bk02_vs_Bk03_%d_'%id,dpi=100)
'''
c.add_chain(datab03,parameters=["$\zeta$", "$R_{mfp}$", "$Mh_{min}$","logLb0.3"],
            name="Bispectrum_for_K=0.3")

'''



