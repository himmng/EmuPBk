import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
id = 0
path = '/home/ht/PycharmProjects/EmuPBk/plots/results/Pk_results/%d/'%id
n_ion , R_mfp , NoH = np.loadtxt(path+"Pk_.out", usecols = (0,1,2), unpack = True)
Mh = 1.087*NoH

logL = np.loadtxt(path+"Pk_prob.out", usecols = (0), unpack = True)
df = pd.DataFrame({'$\zeta$':n_ion,'$Rmfp$':R_mfp,'$Mhalo_{min}(10^8$ $M_\odot)$':NoH})


plt.figure(figsize=(8,8))
plt.title('Parameter Constraints for EoR')
sns.set_style("ticks")
sns.despine()
g = sns.PairGrid(df, diag_sharey=False)
g.map_lower(sns.kdeplot,cmap='Blues', shade=True,shade_lowest=False)
g.map_diag(sns.distplot, color="g", kde_kws={"shade": True})
for i, j in zip(*np.triu_indices_from(g.axes, 1)):
    g.axes[i, j].set_visible(False)

g.savefig(path+'Pkplot_nion_{0:.2f}_R_mfp_{1:.2f}_Mhalo_{2:.2f}.png'
          .format(n_ion.mean(),R_mfp.mean(),Mh.mean()),dpi=100)