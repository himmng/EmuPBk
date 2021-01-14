import numpy as np
from tensorflow import keras as ks


def setup():

    print("Core setup done!")


class Core(object):

    """Core module (CosmoHammer)"""

    def __init__(self, load_model, norm=1.0):
        """
        :param load_model: load the model(can be a function of parameters model=f(a,b,c)),
        :param norm: the rescaling used in training, if any, default = 1.0.
        """
        self.model = ks.models.load_model(load_model)
        self.norm = norm

    def __call__(self, ctx):
        """
        :param ctx: contexts saves proposed step in (parameter, data) space
        """

        params = ctx.getParams()

        ctx.add('params_pk', params)

        mhmin, nion, rmfp = params

        params = np.array([[mhmin, nion, rmfp]])

        params = np.reshape(params, (1, 3))

        model_th = self.model.predict(params)
        model_th = model_th * self.norm

        ctx.add("model_th", model_th)

    @staticmethod
    def setup():
        print('core is done!')