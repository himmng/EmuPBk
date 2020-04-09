import time
from cosmoHammer import MpiCosmoHammerSampler
from cosmoHammer import LikelihoodComputationChain
from cosmoHammer.util import Params
from Power_like import PSlikeModule as slk
from Power_core import PScore

'''     The parameter space is defined
paramters = [peak, min., max., sigma] ===> A rough idea about the parameters
'''
params = Params(("n_ion",[250.,10.0,510.0,1.0]),("R_mfp", [65.0,5.0,130.0,.5]),("NoH",[750.0,10.0,1510.0,1.0]))


chain = LikelihoodComputationChain(min = params[:,1] , max = params[:,2])

chain.addCoreModule(PScore())

chain.addLikelihoodModule(slk())

chain.setup()

sampler = MpiCosmoHammerSampler(
            params= params,
            likelihoodComputationChain=chain,
            filePrefix="PkANN_",
            walkersRatio=10,
            burninIterations=250,
            sampleIterations=250,threadCount=16)

print("start sampling: Here we go with the power of zeus.")
sampler.startSampling()
start = time.time()
sampler.startSampling()
end = time.time()
print(f"The time taken {end-start}sec. done!")

