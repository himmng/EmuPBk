import numpy as np
import os
from tensorflow import keras as ks

path = os.path.abspath(os.path.join(__file__, os.pardir))
path = path+'/existing_models/'

print(path)

modelp = ks.models.load_model(path+'pk99.h5')
model02 = ks.models.load_model(path+'BkANN02.h5')
model03 = ks.models.load_model(path+'BkANN03.h5')
model15 = ks.models.load_model(path+'BkANN15.h5')

class EMUPBK:
    '''It is able to give the Epoch of Reionization(EoR) 21-cm power spectrum
            and bispectrum given the 3 astrophysical
            parameter (Nion,Rmfp,NoH)
            '''

    def __init__(self):

        pass
    
    def PK(self,params):
        '''The 21-cm powerspectrum EmuPBk based on Artificial Neural Networks(ANN),
            It is 99% accurate in the prediction of Powerspectrum, given the three parameter array
            Input: array([Nion,Rmfp,NoH])

            Output: P(k)
            '''
        params = np.reshape(params, (1, 3))
        pk = modelp.predict(params)

        return pk


    def BK02(self,params):

        params = np.reshape(params, (1, 3))

        Bk02 = model02.predict(params)

        return Bk02


    def BK03(self, params):


        params = np.reshape(params, (1, 3))

        Bk03 = model03.predict(params)

        return Bk03


    def BK15(self, params):

        params = np.reshape(params, (1, 3))

        Bk15 = model15.predict(params)


        return Bk15





