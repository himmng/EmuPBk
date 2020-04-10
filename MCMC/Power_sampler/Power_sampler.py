import time

from Power_core import PScore
from Power_like import PSlikeModule as slk
from cosmoHammer import LikelihoodComputationChain
from cosmoHammer import MpiCosmoHammerSampler
from cosmoHammer.util import Params
from Power_like import id
'''     The parameter space is defined
paramters = [peak, min., max., sigma] ===> A rough idea about the parameters
'''
path = '/home/ht/PycharmProjects/EmuPBk/plots/results & plots/Pk_results'
params = Params(("n_ion",[250.,10.0,510.0,1.0]),("R_mfp", [65.0,5.0,130.0,.5]),("NoH",[750.0,10.0,1510.0,1.0]))


chain = LikelihoodComputationChain(min = params[:,1] , max = params[:,2])

chain.addCoreModule(PScore())

chain.addLikelihoodModule(slk())

chain.setup()

sampler = MpiCosmoHammerSampler(
            params= params,
            likelihoodComputationChain=chain,
            filePrefix=path+"/%d/Pk_"%id,
            walkersRatio=20,
            burninIterations=250,
            sampleIterations=500,threadCount=16)

print("start sampling: Here we go with the power of zeus.")
sampler.startSampling()
start = time.time()
sampler.startSampling()
end = time.time()
tic = end-start
#print("The time taken %.2f sec. done!"%tic)
print('Done!')
