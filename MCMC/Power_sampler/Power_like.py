import numpy as np

id = 17 #===using for first run

datapath = '/home/ht/PycharmProjects/EmuPBk/data/data_Powerspectrum/'
pk = np.loadtxt(datapath+'pk_test')
params = np.loadtxt(datapath+'params_test')
nbins = np.loadtxt(datapath+'nbins_test')

def cov_inv(ind):
	'''The covariance matrix calculated using sample variance'''

	cov = np.zeros(shape=(len(nbins),7,7))
	for i in range(len(nbins)):
		for j in range(7):
			cov[i,j,j] = pk[i,j]/np.sqrt(nbins[i][j])
		cov[i]=np.linalg.inv(cov[i])

	return cov[ind]

cov = cov_inv(id)
pk_ob = pk[id]
pk_ob = pk_ob.reshape(1,7)

class PSlikeModule(object):
	'''Likelihood module for cosmoHammer sampling'''
	def __init__(self):

		pass

	def computeLikelihood(self,ctx):

		pk_th = ctx.get("key_data")
		diff = np.subtract(pk_th, pk_ob)
		diff = diff.reshape(1,7)
		logl = -np.dot(diff,np.dot(cov,diff.T))/2.

		return logl

	def setup(self):

		print("Powerspectrum logLikelihood setup is done")

print(params[id])