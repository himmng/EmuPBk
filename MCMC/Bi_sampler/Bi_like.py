import numpy as np

id = 1 #========= Observed Powerspectrum index

datapath = '/home/ht/PycharmProjects/EmuPBk/data/data_Bispectrum/k0.2/'

Bk = np.loadtxt(datapath+'Bk_test')
params = np.loadtxt(datapath+'params_test')
nbins = np.loadtxt(datapath+'N_bins_test')

def cov_inv(ind):
    '''Covariance matrix calculated by using sample variance'''

    cov = np.zeros(shape=(len(nbins),550,550))
    for i in range(len(nbins)):
        for j in range(550):
            cov[i,j,j] = abs(Bk[i,j])/np.sqrt(nbins[i][j])
        cov[i]=np.linalg.inv(cov[i])
    return cov[ind]

Bk_ob = Bk[id]
cov = cov_inv(id)
#======================================================================================================================#
class BklikeModule(object):
    '''Likelihood function class'''

    def __init__(self):
        pass

    def computeLikelihood(self,ctx):
        Bk_th = ctx.get("key_data")
        diff = np.subtract(Bk_th,Bk_ob)
        logl = -np.dot(np.dot(diff,cov),diff.T)/2.
        return logl

    def setup(self):

        print("Bispectrum logLikelihood setup is done")

print(params[id])
