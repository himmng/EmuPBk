import numpy as np
from ANN.BkEmu.Bk_build import index     # index array contains nearly 10 values
from import data.data_Bispectrum.data_read import Bk_02,cov_inv,params_02


id = 0 # using for first run
#======================================================================================================================#
class BklikeModule(object):
    def __init__(self):
        pass

    def computeLikelihood(self,ctx):                 #================computed liklihood=========#
        Bk_th = ctx.get("key_data")

        Bk_ob = Bk[index[id]]               #====='id' can vary from 0 to 9 for 10 samples  ===========#
       # Bk_ob = Bk_ob.reshape(1,7)
        diff = Bk_th - Bk_ob
        diff = diff.reshape(1,7)
        cov_inv = cov_inv[index[id]]
        logl = -np.dot(diff,np.dot(cov_inv,diff.T))/2.
        return logl

    def setup(self):

        print("Bispectrum logLikelihood setup is done")

#======================================================================================================================#