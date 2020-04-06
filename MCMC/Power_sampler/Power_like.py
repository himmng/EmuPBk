
import numpy as np

id = 0 # using for first run
#======================================================================================================================#
class PSlikeModule(object):
    def __init__(self):
        pass

    def computeLikelihood(self,ctx):                 #================computed liklihood=========#
        pk_th = ctx.get("key_data")

        pk_ob = pk[id]               #====='id' can vary from 0 to 9 for 10 samples  ===========#
       # pk_ob = pk_ob.reshape(1,7)
        diff = pk_ob
        #diff = diff.reshape(1,7)
        cov = cov_inv()[id]
        logl = -np.dot(np.dot(diff,cov),diff.T)/2.

        return logl

    def setup(self):

        print("Powerspectrum logLikelihood setup is done")

#======================================================================================================================#