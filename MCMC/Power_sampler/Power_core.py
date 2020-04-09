import numpy as np
from tensorflow import keras as ks

modelpath = '/home/ht/PycharmProjects/EmuPBk/ANN/PkEmu/Pk.h5'
model = ks.models.load_model(modelpath)

def Pk_pred(params):

    '''21-cm Powerspectrum emulator, it can be able to give power spectrum given
    the 3 EoR paramters'''

    params = np.reshape(params,(1,3))
    Pk_th = model.predict(params)

    return Pk_th

class PScore(object):

    '''Core module for CosmoHammer'''

    def __init__(self):

        pass

    def __call__(self, ctx):

        params = ctx.getParams()

        ctx.add('params_pk',params)

        n_ion, R_mfp, NoH = params

        prm = np.array([[n_ion, R_mfp, NoH]])

        prm.reshape(1,3)

        Pk_th = Pk_pred(prm)

        ctx.add("key_data",Pk_th)

    def setup(self):

        print("Powerspectrum core setup is done")

#======================================================================================================================#