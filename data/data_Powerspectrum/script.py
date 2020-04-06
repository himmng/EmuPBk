
import sys
import time
import emcee
import numpy as np
from schwimmbad import MPIPool

pk = np.loadtxt('pk_test')
cov = np.loadtxt('cov_inv_test')
params = np.loadtxt('params_test')

def logL(params):
    
    nion , rmfp , noh = params
    
    if nion < 0 or nion > 200 or rmfp < 0 or rmfp > 150 or noh < 30 or noh > 1500:
        logl = -np.inf
        
    else:
        diff = pk[0]-pk_th(params)
        
        logl = -0.5*np.dot(np.dot(diff,cov),diff.T)
        
    return logl
        

with MPIPool() as pool:
    if not pool.is_master():
        pool.wait()
        sys.exit(0)

    nwalkers = 6
    ndim = 3

    p  = 100*np.random.rand(nwalkers,ndim)
    
    nsteps = 1000

    sampler = emcee.EnsembleSampler(nwalkers, ndim, logL, pool=pool)
    start = time.time()
    sampler.run_mcmc(p, nsteps)
    s_flat = sampler.flatchain
    logp = sampler.lnprobability
    end = time.time()
    print(end - start)
    np.savetxt('sflat',s_flat)
    np.savetxt('logp',logp)
    
