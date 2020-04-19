import numpy as np
import tensorflow as tf
tf.config.optimizer.set_jit(True)



modelpath1 = '/home/ht/PycharmProjects/EmuPBk/ANN/BkEmu/BkANN02.h5'
modelpath2 = '/home/ht/PycharmProjects/EmuPBk/ANN/BkEmu/BkANN03.h5'
modelpath3 = '/home/ht/PycharmProjects/EmuPBk/ANN/BkEmu/BkANN15.h5'
model = tf.keras.models.load_model(modelpath2)

rescale1 =100. #======The rescaling factor used for training the data
rescale2 =10000.
rescale3 =1000000.


def Bk_pred(params):

    '''The funtion will be able to predict the 21-cm Bispectrum given the parameters,
    The parameters that have used in the training are (Ionizing efficiency, RMS distance, Mininum Halo Mass)
     '''
    params = np.reshape(params,(1,3))
    Bk_th = model.predict(params)
    Bk_th = Bk_th*rescale2
    Bk_th =np.around(Bk_th,1)

    return Bk_th

class Bkcore(object):

    '''The core module of cosmoHammer'''

    def __init__(self):

        pass


    def __call__(self, ctx):
        params = ctx.getParams()

        ctx.add('params_pk',params)
        n_ion, R_mfp, NoH = params
        prm = np.array([n_ion, R_mfp, NoH])

        Bk_th = Bk_pred(prm)

        ctx.add("key_data",Bk_th)




    def setup(self):

        print("Bispectrum core setup is done")