import time
import numpy as np
from Bi_core import Bkcore as Bcr
from Bi_like import BklikeModule as Blk
from cosmoHammer import LikelihoodComputationChain
from cosmoHammer import MpiCosmoHammerSampler
from cosmoHammer.pso.MpiParticleSwarmOptimizer import MpiParticleSwarmOptimizer
from cosmoHammer.util import Params
from Bi_like import id,k



'''     The parameter space is defined
paramters = [peak, min., max., sigma] ===> A rough idea about the parameters
'''

path = '/home/ht/PycharmProjects/EmuPBk/plots/results & plots/Bk_results/K%.1f'%k
params = Params(("n_ion",[125,10,250,1]),("R_mfp", [75,5,150,1]),("NoH",[750,10,1510,1]))


chain = LikelihoodComputationChain(min = params[:,1] , max = params[:,2])
chain.params = params
chain.addCoreModule(Bcr())

chain.addLikelihoodModule(Blk())

chain.setup()
print("find best fit point")
pso = MpiParticleSwarmOptimizer(chain, params[:,1], params[:,2])
psoTrace = np.array([pso.gbest.position.copy() for _ in pso.sample()])
params[:, 0] = pso.gbest.position

sampler = MpiCosmoHammerSampler(
            params= params,
            likelihoodComputationChain=chain,
            filePrefix=path+"/%d/Bk"%id,
            burninIterations=250,
            walkersRatio=20,
            sampleIterations=500,)


print("start sampling: Here.")
start = time.time()
sampler.startSampling()
end = time.time()
t = end-start
print('The time taken %.2f sec. done!'%t)