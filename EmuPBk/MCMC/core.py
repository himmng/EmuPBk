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

        par = ctx.getParams()

        ctx.add('params', par)

        par_count = len(par.keys)

        params = np.array([i for i in par])

        params = np.reshape(params, (1, par_count))

        model_th = self.model.predict(params)
        model_th = model_th * self.norm

        ctx.add("model_th", model_th)

    @staticmethod
    def setup():
        print('core is done!')