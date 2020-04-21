import numpy as np
import os
from tensorflow import keras as ks

path = os.path.abspath(os.path.join(__file__, os.pardir))
path = path+'/tests/existing_models/'

pathpk = path+'pk99.h5'
pathbk02 = path+'BkANN02.h5'
pathbk03 = path+'BkANN03.h5'
pathbk15 = path+'BkANN15.h5'

class EMUPBK:
    '''It is able to give the Epoch of Reionization(EoR) 21-cm power spectrum
            and bispectrum given the 3 astrophysical
            parameter (Nion,Rmfp,NoH)
    '''
    def __init__(self,):
        self.model = ks.models.load_model



    def PK(self,params, path = pathpk):
        '''The 21-cm powerspectrum EmuPBk based on Artificial Neural Networks(ANN),
            It is 99% accurate in the prediction of Powerspectrum, given the three parameter array
            Input: array([Nion,Rmfp,NoH])

            Output: P(k)
        '''
        params = np.reshape(params, (1, 3))
        model = self.model(path)
        pk = model.predict(params)

        return pk


    def BK02(self,params,rescale02=100,path=pathbk02):

        params = np.reshape(params, (1, 3))
        model = self.model(path)
        Bk02 = model.predict(params)
        Bk02 = Bk02*rescale02

        return Bk02


    def BK03(self, params,rescale03=10000,path=pathbk03):


        params = np.reshape(params, (1, 3))
        model = self.model(path)
        Bk03 = model.predict(params)
        Bk03 = Bk03*rescale03

        return Bk03


    def BK15(self, params,rescale15=10000000,path=pathbk15):

        params = np.reshape(params, (1, 3))
        model = self.model(path)
        Bk15 = model.predict(params)
        Bk15 = Bk15*rescale15


        return Bk15





