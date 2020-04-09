import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from MCMC.Bi_sampler.Bi_like import params,id


path = '/hom/ht/PycharmProjects/EmuPBk/MCMC/Bi_sampler/'

n_ion , R_mfp , NoH = np.loadtxt(path+"Bipectrum_.out", usecols = (0,1,2), unpack = True)
Mh = 1.087*NoH

logL = np.loadtxt(path+"Bipectrum_prob.out", usecols = (0), unpack = True)

df = pd.DataFrame({'$\zeta$':n_ion,'$Rmfp$':R_mfp,'$Mhalo_{min}(10^8$ $M_\odot)$':NoH})


plt.figure(figsize=(8,8))
plt.title('Parameter Esitmation of EoR')
sns.set_style("ticks")
sns.despine()
g = sns.PairGrid(df, diag_sharey=False)
g.map_lower(sns.kdeplot,cmap='Blues', shade=True,shade_lowest=False)
g.map_diag(sns.distplot, color="g", kde_kws={"shade": True})
for i, j in zip(*np.triu_indices_from(g.axes, 1)):
    g.axes[i, j].set_visible(False)

g.savefig('plot_nion_{3:.3f}_R_mfp_{4:.3f}_Mhalo_{5:.3f}_means{1:},{2:},{3:}.png'
          .format(n_ion.mean(),R_mfp.mean(),Mh.mean()
    ,int(params[id,0]),int(params[id,0]),int(params[id,2])),dpi=400)
