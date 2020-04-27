import numpy as np
from tensorflow import keras as ks



class Core(object):

    '''Core module for CosmoHammer'''

    def __init__(self,load_model,rescale):
        '''
        :param load_model: load the model(can be a function of parameters  model=f(a,b,c)),
        :param rescale: if model is ANN based, the rescaling used in training.
        '''
        self.model = ks.models.load_model(load_model)
        self.rescale = rescale

    def __call__(self, ctx):
        '''
        :param ctx: contexts saves proposed step in (parameter, data) space
        '''

        params = ctx.getParams()

        ctx.add('params_pk',params)

        n_ion, R_mfp, NoH = params

        params = np.array([[n_ion, R_mfp, NoH]])

        params = np.reshape(params,(1,3))

        Pk_th = self.model.predict(params)
        Pk_th = Pk_th*self.rescale
        ctx.add("key_data",Pk_th)

    def setup(self):

        print("Core setup is done")

#======================================================================================================================#