import numpy as np

from data.data_Bispectrum.data_read2 import Bk_test, cov_inv_test

id = 0 # using for first run
#======================================================================================================================#
class BklikeModule(object):
    def __init__(self):
        pass

    def computeLikelihood(self,ctx):                 #================computed liklihood=========#
        Bk_th = ctx.get("key_data")

        Bk_ob = Bk_test[id]               #====='id' can vary from 0 to 9 for 10 samples  ===========#
        diff = Bk_th - Bk_ob
        diff = diff.reshape(1,7)
        cov_inv = cov_inv_test
        logl = -np.dot(np.dot(cov_inv,diff),diff.T)/2.
        return logl

    def setup(self):

        print("Bispectrum logLikelihood setup is done")

#======================================================================================================================#