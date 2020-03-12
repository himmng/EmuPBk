import numpy as np
from Bipectrum_Emulator import Bk_pred

#======================================================================================================================#

class PScore(object):

    def __init__(self):
        pass                                    #=====================CosmoHammer core module for THANN================#
    def __call__(self, ctx):
        params = ctx.getParams()

        ctx.add('params_pk',params)
        n_ion, R_mfp, NoH = params
        prm = np.array([[n_ion, R_mfp, NoH]])
        prm.reshape(1,3)

        Bk_th = Bk_pred(prm)

        ctx.add("key_data",Bk_th)

    def setup(self):

        print("Bispectrum core setup is done")

#======================================================================================================================#



