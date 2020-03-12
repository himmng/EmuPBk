from cosmoHammer import LikelihoodComputationChain
from cosmoHammer.CosmoHammerSampler import CosmoHammerSampler
from cosmoHammer.util import Params
from Power_core import PScore
from Power_like import PSlikeModule as slk

#======================================================================================================================#

params = Params(("n_ion",[250.,10.0,510.0,1.0]),("R_mfp", [38.0,5.0,70.0,.5]),("NoH",[750.0,10.0,1510.0,1.0]))
                            #==========The parameter space is defined================================#
#======================================================================================================================#

chain = LikelihoodComputationChain(min = params[:,1] , max = params[:,2])

chain.addCoreModule(PScore())                                       #=========setting up the modules===================#

chain.addLikelihoodModule(slk())

chain.setup()
#======================================================================================================================#

sampler = CosmoHammerSampler(
            params= params,
            likelihoodComputationChain=chain,                  #=============mpi sampler===============================#
            filePrefix="Powerspectrum_THANN_",
            walkersRatio=4,
            burninIterations=250,
            sampleIterations=250, threadCount=8)

#======================================================================================================================#

print("start sampling: Here we go with the power of zeus.")
sampler.startSampling()
print("done!")

#======================================================================================================================#
