'''
import tensorflow as tf
import numpy as np


class EANN():

This will calculate the emulated statistics given the parameters
    def __int__(self,params):
        self.params = params
        self.params = np.reshape(self.params, (1, 3))
    def model(self,md):

        model = tf.keras.models.load_model(md)

        model = model.predict(self.params)

        return model


def PANN():
    params = np.array([30,20,50])
    model = '/PkEMu/Pk.h5'
    pk = EANN.model(model)

    return Pk

print(PANN()
      
'''
from cosmoHammer import MpiCosmoHammerSampler
help(MpiCosmoHammerSampler())