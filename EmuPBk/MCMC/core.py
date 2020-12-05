import numpy as np
from tensorflow import keras as ks

class Core(object):

    '''Core module (CosmoHammer)'''

    def __init__(self,load_model,norm):
        '''
        :param load_model: load the model(can be a function of parameters model=f(a,b,c)),
        :param rescale: if model is ANN based, the rescaling used in training.
        '''
        self.model = ks.models.load_model(load_model)
        self.norm = norm

    def __call__(self, ctx):
        '''
        :param ctx: contexts saves proposed step in (parameter, data) space
        '''

        params = ctx.getParams()

        ctx.add('params_pk',params)

        Mhmin, Nion, Rmfp = params

        params = np.array([[Mhmin, Nion, Rmfp]])

        params = np.reshape(params,(1,3))

        model_th = self.model.predict(params)
        model_th = model_th * self.norm

        ctx.add("model_th",model_th)

    def setup(self):

        print("Core setup done!")

#======================================================================================================================#