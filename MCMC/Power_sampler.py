from cosmoHammer import MpiCosmoHammerSampler
from cosmoHammer import LikelihoodComputationChain
from cosmoHammer.util import Params
from MCMC.Power_like import PSlikeModule as slk
from MCMC.Power_core import PScore

#======================================================================================================================#

params = Params(("n_ion",[250.,10.0,510.0,1.0]),("R_mfp", [38.0,5.0,70.0,.5]),("NoH",[750.0,10.0,1510.0,1.0]))
                            #==========The parameter space is defined================================#
#======================================================================================================================#

chain = LikelihoodComputationChain(min = params[:,1] , max = params[:,2])

chain.addCoreModule(PScore())                                       #=========setting up the modules===================#

chain.addLikelihoodModule(slk())

chain.setup()
#======================================================================================================================#

sampler = MpiCosmoHammerSampler(
            params= params,
            likelihoodComputationChain=chain,                  #=============mpi sampler===============================#
            filePrefix="Powerspectrum_THANN_",
            walkersRatio=10,
            burninIterations=250,
            sampleIterations=250)

#======================================================================================================================#

print("start sampling: Here we go with the power of zeus.")
sampler.startSampling()
print("done!")

#======================================================================================================================#
