import time
from cosmoHammer import LikelihoodComputationChain
from cosmoHammer import MpiCosmoHammerSampler
from cosmoHammer.util import Params
from Bi_core import Bkcore as Bcr
from Bi_like import BklikeModule as Blk

'''     The parameter space is defined
paramters = [peak, min., max., sigma] ===> A rough idea about the parameters
'''
params = Params(("n_ion",[250.,10.0,510.0,1.0]),("R_mfp", [38.0,5.0,80.0,.5]),("NoH",[750.0,10.0,1510.0,1.0]))


chain = LikelihoodComputationChain(min = params[:,1] , max = params[:,2])

chain.addCoreModule(Bcr())

chain.addLikelihoodModule(Blk())

chain.setup()

sampler = MpiCosmoHammerSampler(
            params= params,
            likelihoodComputationChain=chain,
            filePrefix="Bipectrum_",
            burninIterations=250,
            walkersRatio=10,
            sampleIterations=250,threadCount=16)


print("start sampling: Here.")
start = time.time()
sampler.startSampling()
end = time.time()
print(f"The time taken {end-start}sec. done!")

