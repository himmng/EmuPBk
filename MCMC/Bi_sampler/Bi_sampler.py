from Bi_core import PScore as Bcr
from Bi_like import BklikeModule as Blk
from cosmoHammer import LikelihoodComputationChain
from cosmoHammer import MpiCosmoHammerSampler
from cosmoHammer.util import Params



#======================================================================================================================#

params = Params(("n_ion",[250.,10.0,510.0,1.0]),("R_mfp", [38.0,5.0,70.0,.5]),("NoH",[750.0,10.0,1510.0,1.0]))
                            #==========The parameter space is defined================================#
#======================================================================================================================#

chain = LikelihoodComputationChain(min = params[:,1] , max = params[:,2])

chain.addCoreModule(Bcr())                                       #=========setting up the modules===================#

chain.addLikelihoodModule(Blk())

chain.setup()
#======================================================================================================================#

sampler = MpiCosmoHammerSampler(
            params= params,
            likelihoodComputationChain=chain,                  #=============mpi sampler===============================#
            filePrefix="Bipectrum_",
            walkersRatio=4,
            burninIterations=250,
            sampleIterations=250)

#======================================================================================================================#

print("start sampling: Here we go with the power of zeus.")
sampler.startSampling()
print("done!")

#======================================================================================================================#
