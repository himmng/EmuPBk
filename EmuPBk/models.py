import os
from tensorflow import keras as ks

path = os.path.abspath(os.path.join(__file__, os.pardir))
path = path+'/MCMC/existing_models/'
ex_models = ['pk.h5', 'bk.h5']


class Predict:
    """
        Epoch of Reionization(EoR) 21-cm powerspectrum and Bispectrum,
        provided astrophysical parameters (Mhmin,Nion,Rmfp)
    """
    def __init__(self,):
        pass

    def existing_models(self, params, stats,):
        """
        stats: int(0, 1)
        choose 0 for powerspectrum, and 1 for bispectrum
        """
        load_model = path+'%s' % ex_models[stats]
        model = ks.models.load_model(load_model)
        model = model.predict(params)
        return model

    def model(self, load_model, params):
        """
        The 21-cm powerspectrum or 21-cm Bispectrum from k1 range (0.2-1.5) mpc^-1
        based on Artificial Neural Networks(ANN),
            Input:
            :param: params: array([Mhmin,Nion,Rmfp])
            :returns: emulated P(k) or B(k1,k2,k3)
        """

        model = ks.models.load_model(load_model)
        pred = model.predict(params)
        return pred