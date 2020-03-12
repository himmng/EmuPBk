import numpy as np
#from ANN.ANN_build import index  # index array contains nearly 10 values
from data_read2 import pk,cov_inv

id = 0 # using for first run
#======================================================================================================================#
class PSlikeModule(object):
    def __init__(self):
        pass

    def computeLikelihood(self,ctx):                 #================computed liklihood=========#
        pk_th = ctx.get("key_data")

        pk_ob = pk[id]               #====='id' can vary from 0 to 9 for 10 samples  ===========#
        pk_ob = pk_ob.reshape(1,7)
        diff = pk_th - pk_ob
        diff = diff.reshape(1,7)


        logl = -np.dot(diff,np.dot(cov_inv,diff.T))/2.
        return logl

    def setup(self):

        print("Powerspectrum logLikelihood setup is done")

#======================================================================================================================#