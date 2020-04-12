import time
import numpy as np
from Power_core import PScore
from Power_like import PSlikeModule as slk
from cosmoHammer import LikelihoodComputationChain
from cosmoHammer import MpiCosmoHammerSampler
from cosmoHammer.pso.MpiParticleSwarmOptimizer import MpiParticleSwarmOptimizer
from cosmoHammer.util import Params
from Power_like import id
'''     The parameter space is defined
paramters = [peak, min., max., sigma] ===> A rough idea about the parameters
'''
path = '/home/ht/PycharmProjects/EmuPBk/plots/results & plots/Pk_results'
params = Params(("n_ion",[105,10,220,2]),("R_mfp", [62,5,130,1]),("NoH",[750,10,1510,5]))


chain = LikelihoodComputationChain(min = params[:,1] , max = params[:,2])
chain.params = params
chain.addCoreModule(PScore())

chain.addLikelihoodModule(slk())

chain.setup()
print("find best fit point")
pso = MpiParticleSwarmOptimizer(chain, params[:,1], params[:,2])
psoTrace = np.array([pso.gbest.position.copy() for _ in pso.sample()])
params[:, 0] = pso.gbest.position



sampler = MpiCosmoHammerSampler(
            params= params,
            likelihoodComputationChain=chain,
            filePrefix=path+"/%d/Pk_"%id,
            walkersRatio=20,
            burninIterations=250,
            sampleIterations=250,threadCount=16)

print("start sampling: Here we go with the power of zeus.")
sampler.startSampling()
start = time.time()
sampler.startSampling()
end = time.time()
tic = end-start
#print("The time taken %.2f sec. done!"%tic)
print('Done!')
