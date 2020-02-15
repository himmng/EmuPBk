import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd

#======================================================================================================================#

path = '/MCMC/Power_sampler/'
n_ion , R_mfp , NoH = np.loadtxt(path+"Powerspectrum_THANN_.out", usecols = (0,1,2), unpack = True)
#======= n_ion,mean free path and no- of halos =====#
#Mh = 1.087*1.e8*NoH
Mh = 1.087*NoH
logL = np.loadtxt(path+"Powerspectrum_THANN_prob.out", usecols = (0), unpack = True)     #=== loglikelyhood =====#
df = pd.DataFrame({'$\zeta$':n_ion,'$Rmfp$':R_mfp,'$Mhalo_{min}(10^8$ $M_\odot)$':NoH})

#======================================================================================================================#





#mean = R_mfp.mean()

plt.figure(figsize=(8,8))
plt.title('Parameter Constraints for EoR')
sns.set_style("ticks")
#sns.set(style="white")
sns.despine()
g = sns.PairGrid(df, diag_sharey=False)
#cmap = sns.cubehelix_palette(8, start=.2, rot=-.95,as_cmap=True)
g.map_lower(sns.kdeplot,cmap='Blues', shade=True,shade_lowest=False)
#g.map_upper(sns.scatterplot)
g.map_diag(sns.distplot, color="g", kde_kws={"shade": True})
for i, j in zip(*np.triu_indices_from(g.axes, 1)):
    g.axes[i, j].set_visible(False)

g.savefig('plot_nion_{0:.3f}_R_mfp_{1:.3f}_Mhalo_{2:.3f}.png'.format(n_ion.mean(),R_mfp.mean(),Mh.mean()),dpi=400)
print('plot_nion_{0:.3f}_R_mfp_{1:.3f}_Mhalo_{2:.3f}.png'.format(n_ion.mean(),R_mfp.mean(),Mh.mean()))