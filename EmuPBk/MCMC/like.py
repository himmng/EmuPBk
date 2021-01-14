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
		cov = abs(data)/np.sqrt(nbins)
		cov = cov + abs(noise)
		cov = eye*cov
		self.cov_inv = np.linalg.inv(cov)
		self.div = div

	def computelikelihood(self, ctx):
		"""
		:param ctx: context (load proposed steps from context)
		:return: loglikelihood
		"""
		model_th = ctx.get("model_th")
		diff = np.subtract(model_th, self.data)
		diff = diff.reshape(1, len(self.data))
		logl = -np.dot(diff, np.dot(self.cov_inv, diff.T))/2.
		logl = logl/self.div
		return logl

	@staticmethod
	def setup():

		print("Likelihood setup done!")
