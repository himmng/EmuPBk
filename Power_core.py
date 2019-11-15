import data as d
import numpy as np
from Powerspectrum_Emulator import pk_pred

#======================================================================================================================#

class PScore(object):

    def __init__(self):
        pass                                    #=====================CosmoHammer core module for THANN================#
    def __call__(self, ctx):
        params = ctx.getParams()

        ctx.add('params_pk',params)
        n_ion ,R_mfp, NoH = params

        prm = [[n_ion,R_mfp,NoH]]
        prm = np.array(prm)
        prm.reshape(1,3)

        pk_th = pk_pred(prm)

        ctx.add("key_data",pk_th)

    def setup(self):

        print("Powerspectrum core setup is done")

#======================================================================================================================#



