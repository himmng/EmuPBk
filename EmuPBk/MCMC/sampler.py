import os
import time
import numpy as np
from EmuPBk.MCMC.core import Core
from EmuPBk.MCMC.like import LikeModule
from cosmoHammer.util import Params
from cosmoHammer import MpiCosmoHammerSampler
from cosmoHammer import LikelihoodComputationChain
from cosmoHammer.pso.MpiParticleSwarmOptimizer import MpiParticleSwarmOptimizer


#The parameter space is defined
#paramters = [peak, min., max., sigma] ===> A rough idea about the prior

path = os.path.abspath(os.path.join(__file__, os.pardir))
path = path+'/tests/existing_models/'

params = Params(("n_ion",[105,10,220,2]),("R_mfp", [62,5,130,1]),("NoH",[750,10,1510,5]))

class Run_MCMC:

    '''
    CosmoHammer based MCMC sampler
    Uses MPI sampler class
    '''

    def __init__(self,data,nbins,noise=None,div=1):

        '''
        :param data: load your data
        :param nbins: no. of bins in the data (for covariance matrix )
        '''

        chain = LikelihoodComputationChain(min=params[:, 1], max=params[:, 2])
        chain.params = params
        chain.addLikelihoodModule(LikeModule(data,nbins,noise,div))
        self.chain = chain

    def load_existing_model(self,name='Pk'):
        '''
        Use the existing ANN models for MCMC analysis
        :param name: use ('Pk','Bk02','Bk03','Bk15')==>for powerspectrum, Bispectrum02, Bispectrum03, Bispectrum15
        '''

        self.name = name
        if name == 'Pk':
            self.chain.addCoreModule(Core(load_model=path+'Pk.h5',rescale=1))

        elif name == 'Bk02':
            self.chain.addCoreModule(Core(load_model=path+'Bk02.h5',rescale=100))

        elif name == 'Bk03':
            self.chain.addCoreModule(Core(load_model=path+'Bk03.h5',rescale=10000))

        else:
            self.chain.addCoreModule(Core(load_model=path+'Bk15.h5',rescale=10000000))
        self.chain.setup()


    def load_model(self,load_model,name='Pk',rescale=1):

        '''
        :param load_model: load your own model, (give the path)
        :param name: name of data, ('Pk','Bk02','Bk03','Bk15')==>for powerspectrum, Bispectrum02, Bispectrum03, Bispectrum15
        :param rescale: rescale used in the training
        '''
        self.name = name
        self.chain.addCoreModule(Core(load_model,rescale))

        self.chain.setup()


    def sampler(self,walker_ratio, burnin, samples,num, threads=-1):
        '''

        :param walker_ratio:  the ratio of walkers and the count of sampled parameters
        :param burnin: burin iterations
        :param samples: no. of sample iterations
        :param num: number to put in output files e.g: string(name+num)=Pk_1,Bk_1
        :param threads: no. of cpu threads
        '''
        print("find best fit point")
        pso = MpiParticleSwarmOptimizer(self.chain, params[:, 1], params[:, 2])
        psoTrace = np.array([pso.gbest.position.copy() for _ in pso.sample()])
        params[:, 0] = pso.gbest.position

        sampler = MpiCosmoHammerSampler(
                    params= params,
                    likelihoodComputationChain=self.chain,
                    filePrefix='%s'%self.name+'%d'%num,
                    walkersRatio=walker_ratio,
                    burninIterations=burnin,
                    sampleIterations=samples,threadCount=threads)

        print("start sampling:.")
        start = time.time()
        sampler.startSampling()
        end = time.time()
        tics = end-start
        print("The time taken %.2f sec. done!"%tics)
        print('Done!')
