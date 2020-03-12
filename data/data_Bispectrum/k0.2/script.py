
import sys
import time
import emcee
import numpy as np
from schwimmbad import MPIPool
import keras as ks


N_bins_test = np.loadtxt('N_bins_test')
Bk_test = np.loadtxt('Bk_test')

def Bk_model(params):
    
    model = ks.models.load_model('98bk02.h5')

    Bk_model = model.predict([[params]])
    Bk_model = Bk_model*100.
    
    return Bk_model[0]

def cov(Bk,Nbins):
    a = np.eye(550,550)
    b = Bk/np.sqrt(Nbins)
    b = np.reshape(b,(550,1))
    b = b*a
    cov_inv = np.linalg.inv(b)
    return cov_inv
    
    
def lL(params,Bk,N_bins):
    if params[0]<=0 or params[0] > 200 or params[1]<=0 or params[1] > 150 or params[2]<=0 or params[2] > 1500 :
        l=-np.inf
    else:

        model=Bk_model(params)
        DMU=Bk-model
        DMU = np.reshape(DMU, (1,550))
        cov_inv = cov(Bk,N_bins)
        
        l=-0.5*np.dot(np.dot(DMU,cov_inv),DMU.T)
        l = l[0][0]
   
    return l
    
    
with MPIPool(8) as pool:
    if not pool.is_master():
        pool.wait()
        sys.exit(0)

    np.random.seed(42)
    ndim=3            #===== no. of parameters =======#

    nwalkers=6      #========= no. of random walkers ========#

    Nion = np.random.randint(20,200,6).T
    Rmfp = np.random.randint(20,120,6).T
    NoH = np.random.randint(50,1100,6).T
    p = np.zeros(shape=(6,3))
    p[:,0] = Nion
    p[:,1] = Rmfp
    p[:,2] = NoH
    nsteps = 8
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lL, args=[Bk_test[0],N_bins_test[0]],pool=pool)
    start = time.time()
    sampler.run_mcmc(p, nsteps)
    end = time.time()
    print(end - start)
