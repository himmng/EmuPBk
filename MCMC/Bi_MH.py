import numpy as np
from data.data_Bispectrum.data_read import Bk_02,params_02,cov_inv
from ANN.BkEmu.Bispectrum_Emulator import Bk_pred as model

'''Initializing the no. of sampling steps, log_likelihood and parameter space'''
N = 20000
logL = np.zeros(shape=(N,1))
par = np.zeros(shape=(N,params[:,1].shape))

'''A rough region from where the initial parameter sampling should start, It basically helps the sampler to probe the parameter space in a 
guided manner, although this can not be considered as prior for the parameters.'''


def logL(model,data,cov_inv):
    diff = model - data
    logL = -np.dot(diff,np.dot(cov_inv,diff.T))/2.
    return logL

nion0 = np.random.randint(20.,200.,0.1)
rmfp0 = np.random.randint(20.,100.,0.1)
NoH0  = np.random.randint(50, 1500, 10.)
params[0] = [nion0,rmfp0,NoH0]
model0 = model([params[0]])
lLk0  = logL(model0,data,cov_inv)

for i in range(1,N):
    nion = np.random.normal(params[i-1,0],0.1)
    rmfp = np.random.normal(params[i-1,1],0.1)
    NoH  = np.random.normal(params[i-1,2],10)

    p = [[nion,rmfp,NoH]]

    model= model(p)
    lLk  = logL(model,data,cov_inv)

    if lLk >= lLk0 | lLk/lLk0 > 0.50 :
        params[i, 0] = nion
        params[i, 1] = rmfp
        params[i, 2] = NoH
        logL[i]      = lLk

    else:
        params[i, 0] = params[i-1,0]
        params[i, 1] = params[i-1,1]
        params[i, 2] = params[i-1,2]
        logL[i]      = logL[i-1]



np.savetxt('log_likelihoodBK',logL)
np.savetxt('posterior_space'params)


