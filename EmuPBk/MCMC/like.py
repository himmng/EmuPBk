import numpy as np


class LikeModule(object):
    """Likelihood module (CosmoHammer)"""

    def __init__(self, data, nbins, noise, div):
        """
		:param data: load the observational (test powerspectrum, bispectrum)data, or (test_data, dtype=array,list)
		:param nbins: numbers of k-modes in powerspectrum OR number of triangles for bispectrum(dtype = array, list)
		:param noise: instrumental noise, (dtype = array, list)
		:div: normalizing constant for likelihood.
		"""
        self.data = data
        eye = np.eye(len(data))
        if np.sum(nbins) != 0.:
            cov = abs(data ** 2) / nbins
            cov = cov + np.abs(noise)
            cov = eye * cov
            cov_inv = np.linalg.inv(cov)

        else:
            cov = np.abs(noise)
            cov = eye * cov
            cov_inv = np.linalg.inv(cov)

        self.div = div
        self.cov = cov
        self.cov_inv = cov_inv

    def computeLikelihood(self, ctx):
        """
		:param ctx: context (load proposed steps from context)
		:return: loglikelihood
		"""
        model_th = ctx.get("model_th")
        diff = np.subtract(model_th, self.data)
        diff = diff.reshape(1, len(self.data))
        logl = -np.dot(diff, np.dot(self.cov_inv, diff.T)) / 2.
        logl = logl / self.div
        return logl

    @staticmethod
    def setup():
        print("Likelihood setup done!")


class ComplexLikeModule(object):
    """Likelihood module (CosmoHammer)"""

    def __init__(self, data, ntri, noise):

        """
		simulation specific function, will not be so useful on more than 3 EoR parameters.
		It will give the animation of real bispectrum vs ANN predictions
		for only Unique triangle space configuration.
		:parameter:
		test_data : test_data(Bispectrum)
		test_params : test_parameters of EoR
		ntri: number of triangles contributes in each bisepctrum.
		norm: normalizing factor, (default: 100.)
		"""

        self.costheta = np.array([0.52, 0.57, 0.62, 0.67, 0.72, 0.77, 0.82, 0.87, 0.92, 0.97])
        self.non_na_index = np.array([18, 19, 27, 28, 29, 35, 36, 37, 38, 39, 44, 45, 46, 47,
                                      48, 49, 53, 54, 55, 56, 57, 58, 59, 63, 64, 65, 66, 67,
                                      68, 69, 72, 73, 74, 75, 76, 77, 78, 79, 81, 82, 83, 84,
                                      85, 86, 87, 88, 89, 91, 92, 93, 94, 95, 96, 97, 98, 99,
                                      100, 101, 102, 103, 104, 105, 106, 107, 108, 109])
        self.k2byk1 = np.array([0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.])
        self.k1 = np.array([0.1903934, 0.3220935, 0.5448941, 0.9218117, 1.559453])
        self.data = np.around(data, 2)
        self.ntri = ntri
        self.noise = noise

    def computeLikelihood(self, ctx, region='unique'):
        """
		:parameters
		 ctx: context loads the proposed steps Core module
		 region: you can choose the region of unique triangle space of bispectrum
		  to specifically see its senstivity on parameter posterior PDFs.
		  e.g. use 'unique', 'stretched', 'squeezed', 'linear', 'l-isosceles', 'equilateral'.
		:return: loglikelihood
		"""
        if np.sum(self.ntri) != 0.:
            cov = abs(self.data ** 2) / self.ntri
            cov = cov + np.abs(self.noise)

        else:
            cov = np.abs(self.noise)

        cov = np.reshape(cov, (len(self.k1), len(self.non_na_index)))
        cov = cov.T
        data_ob = np.reshape(self.data, np.shape(cov))
        data_pd = ctx.get("model_th").reshape(np.shape(cov))
        diff = np.subtract(data_pd, data_ob)
        diff = diff.T
        like = np.zeros(shape=len(self.non_na_index))
        for i in range(len(self.non_na_index)):
            like[i] = np.log(1 / np.sqrt(6.28 * np.linalg.det(np.eye(len(self.k1)) * cov[i])))
            -0.5 * np.dot((diff[i] * 1 / (cov[i])), diff[i].T)
        logl = np.zeros(shape=110)
        logl[self.non_na_index] = like
        logl = logl.reshape((11, 10))
        if region == 'unique':
            return np.average(logl)
        elif region == 'linear':
            return np.sum(logl.T[-1])
        elif region == 'equilateral':
            return logl[-1][0]
        elif region == 'stretched':
            return logl[1][1]
        elif region == 'squeezed':
            return logl[-1][-1]
        else:
            return np.average(logl[-1])

    @staticmethod
    def setup():
        print("Likelihood setup done!")
