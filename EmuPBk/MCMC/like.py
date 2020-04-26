import numpy as np


class LikeModule(object):

	'''Likelihood module for cosmoHammer sampling'''

	def __init__(self,data,nbins):

		'''
		:param data: load the observational data, or (test_data)
		:param nbins: numbers of k-bins at each data-set
		'''

		self.data = data
		eye = np.eye(len(data))
		cov = abs(data)/np.sqrt(nbins)
		cov = eye*cov
		self.cov_inv = np.linalg.inv(cov)

	def computeLikelihood(self,ctx):
		'''

		:param ctx: context (loads proposed parameters, and proposed data)
		:return: loglikelihood
		'''
		pk_th = ctx.get("key_data")
		diff = np.subtract(pk_th, self.data)
		diff = diff.reshape(1,len(self.data))
		diff = diff/100.
		logl = -np.dot(diff,np.dot(self.cov_inv,diff.T))/2.

		return logl

	def setup(self):

		print("logLikelihood setup is done")
