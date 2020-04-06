from Bkemu import cov
from dataread import nbins_test , Bk_03_test
import numpy as np
#from Bkemu import Bk_pred
#======================================================================================================================#
class PSlikeModule(object):
    def __init__(self):
        pass

    def computeLikelihood(self,ctx):                 #================computed liklihood=========#
        Bk_th = ctx.get("key_data")

        ind = 9
        Bk = np.reshape(Bk_03_test[ind],(1,550))
        diff = np.subtract(Bk_th,Bk)
        #diff = diff/Bk
        cov_inv = cov(Bk_03_test[ind],nbins_test[ind])
        

        logl = -0.5*np.dot(np.dot(diff,cov_inv),diff.T)
        logl = logl[0]
        

        return logl

    def setup(self):

        print("Powerspectrum logLikelihood setup is done")

#======================================================================================================================#
