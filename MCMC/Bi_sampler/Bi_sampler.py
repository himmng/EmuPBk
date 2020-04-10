import time
from Bi_core import Bkcore as Bcr
from Bi_like import BklikeModule as Blk
from cosmoHammer import LikelihoodComputationChain
from cosmoHammer import MpiCosmoHammerSampler
from cosmoHammer.util import Params
from Bi_like import id,k

'''     The parameter space is defined
paramters = [peak, min., max., sigma] ===> A rough idea about the parameters
'''

path = '/home/ht/PycharmProjects/EmuPBk/plots/results & plots/Bk_results/K%.1f'%k[0]
params = Params(("n_ion",[250.,10.0,510.0,1.0]),("R_mfp", [65.0,5.0,130.0,0.5]),("NoH",[750.0,10.0,1510.0,1.0]))


chain = LikelihoodComputationChain(min = params[:,1] , max = params[:,2])

chain.addCoreModule(Bcr())

chain.addLikelihoodModule(Blk())

chain.setup()

sampler = MpiCosmoHammerSampler(
            params= params,
            likelihoodComputationChain=chain,
            filePrefix=path+"/%d/Bk"%id,
            burninIterations=250,
            walkersRatio=20,
            sampleIterations=500,threadCount=16)


print("start sampling: Here.")
start = time.time()
sampler.startSampling()
end = time.time()
t = end-start
print('The time taken %.2f sec. done!'%t)

