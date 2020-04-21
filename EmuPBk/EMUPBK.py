import numpy as np
import os
from tensorflow import keras as ks
#==========These are default existing models=======================#
path = os.path.abspath(os.path.join(__file__, os.pardir))
path = path+'/tests/existing_models/'
pathpk = path+'pk99.h5'
pathbk02 = path+'BkANN02.h5'
pathbk03 = path+'BkANN03.h5'
pathbk15 = path+'BkANN15.h5'

class EMUPBK:
    ''' gives the Epoch of Reionization(EoR) 21-cm power spectrum
            and bispectrum given the 3 astrophysical
            parameter (Nion,Rmfp,NoH)
    :default: It will use existing trained models, but you can use it for newly trained
    models as well, (both for Powerspctrum and Bispectrum)
    '''
    def __init__(self,):


        self.model = ks.models.load_model

    def PK(self,params, load_model = pathpk):
        '''The 21-cm powerspectrum EmuPBk based on Artificial Neural Networks(ANN),
            It is 99% accurate in the prediction of Power spectrum, given the 3 parameter array
            Input:
            :param params: array([Nion,Rmfp,NoH])
            :param path: default it use exisiting trained model, can be used for new
            models, e.g.: path = './' or 'current working directory'

            Output: P(k)
        '''
        params = np.reshape(params, (1, 3))
        model = self.model(load_model)
        pk = model.predict(params)

        return pk


    def BK02(self,params,rescale02=100,load_model=pathbk02):
        '''
        21-cm Bispectrum emulator for k1 = 0.2 mpc^-1
        :param params: array([Nion, Rmfp, NoH])
        :param path: default it use exisiting trained model, can be
        used for new models, e.g.: path = './' or 'current working directory'
        :return: Bk

        '''
        params = np.reshape(params, (1, 3))
        model = self.model(load_model)
        Bk02 = model.predict(params)
        Bk02 = Bk02*rescale02

        return Bk02


    def BK03(self, params,rescale03=10000,load_model=pathbk03):
        '''
        21-cm Bispectrum emulator for k1 = 0.3 mpc^-1
        :param params: array([Nion, Rmfp, NoH])
        :param path: default it use exisiting trained model, can be
        used for new models, e.g.: path = './' or 'current working directory'
        :return: Bk
        '''
        params = np.reshape(params, (1, 3))
        model = self.model(load_model)
        Bk03 = model.predict(params)
        Bk03 = Bk03*rescale03

        return Bk03


    def BK15(self, params,rescale15=10000000,load_model=pathbk15):
        '''
        21-cm Bispectrum emulator for k1 = 1.5 mpc^-1
        :param params: array([Nion, Rmfp, NoH])
        :param path: default it use exisiting trained model, can be
        used for new models, e.g.: path = './' or 'current working directory'
        :return: Bk
        '''

        params = np.reshape(params, (1, 3))
        model = self.model(load_model)
        Bk15 = model.predict(params)
        Bk15 = Bk15*rescale15


        return Bk15





