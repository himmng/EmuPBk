from ANN.ANN_build import index     # index array contains nearly 10 values
import numpy as np


#======================================================================================================================#
class PSlikeModule(object):
    def __init__(self):
        pass

    def computeLikelihood(self,ctx):                 #================computed liklihood=========#
        pk_th = ctx.get("key_data")

        pk_ob = d.PK[index[id]]               #====='id' can vary from 0 to 9 for 10 samples  ===========#
        pk_ob = pk_ob.reshape(1,7)
        diff = pk_th - pk_ob
        diff = diff.reshape(1,7)
        cov_inv = d.cov_inv
        logl = -np.dot(diff,np.dot(cov_inv,diff.T))/2.
        return logl

    def setup(self):

        print("Powerspectrum logLikelihood setup is done")

#======================================================================================================================#