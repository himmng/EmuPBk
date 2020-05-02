import os

from tensorflow import keras as ks

#==========These are default existing models=======================#

path = os.path.abspath(os.path.join(__file__, os.pardir))
path = path+'/MCMC/tests/existing_models/'

class Predict:
    ''' gives the Epoch of Reionization(EoR) 21-cm power spectrum
            and bispectrum given the 3 astrophysical
            parameter (Nion,Rmfp,NoH)
    :default: It will use existing trained models, but you can use it for newly trained
    models as well, (both for Powerspctrum and Bispectrum)
    '''

    def __init__(self,params):


        self.model = ks.models.load_model
        self.params = params

    def PK(self,load_model = path+'Pk.h5'):

        '''The 21-cm powerspectrum EmuPBk based on Artificial Neural Networks(ANN),
            It is 99% accurate in the prediction of Power spectrum, given the 3 parameter array
            Input:
            :param params: array([Nion,Rmfp,NoH])
            :param path: default it use exisiting trained model, can be used for new
            models, e.g.: path = './' or 'current working directory'

            Output: P(k)
        '''
        model = self.model(load_model)
        pk = model.predict(self.params)

        return pk


    def BK02(self,rescale02=100,load_model=path+'Bk02.h5'):
        '''
        21-cm Bispectrum emulator for k1 = 0.2 mpc^-1
        :param params: array([Nion, Rmfp, NoH])
        :param path: default it use exisiting trained model, can be
        used for new models, e.g.: path = './' or 'current working directory'
        :return: Bk

        '''

        model = self.model(load_model)
        Bk02 = model.predict(self.params)
        Bk02 = Bk02*rescale02

        return Bk02


    def BK03(self,rescale03=10000,load_model=path+'Bk03.h5'):
        '''
        21-cm Bispectrum emulator for k1 = 0.3 mpc^-1
        :param params: array([Nion, Rmfp, NoH])
        :param path: default it use exisiting trained model, can be
        used for new models, e.g.: path = './' or 'current working directory'
        :return: Bk
        '''
        model = self.model(load_model)
        Bk03 = model.predict(self.params)
        Bk03 = Bk03*rescale03

        return Bk03


    def BK15(self,rescale15=10000000,load_model=path+'Bk15.h5'):
        '''
        21-cm Bispectrum emulator for k1 = 1.5 mpc^-1
        :param params: array([Nion, Rmfp, NoH])
        :param path: default it use exisiting trained model, can be
        used for new models, e.g.: path = './' or 'current working directory'
        :return: Bk
        '''

        model = self.model(load_model)
        Bk15 = model.predict(self.params)
        Bk15 = Bk15*rescale15


        return Bk15
