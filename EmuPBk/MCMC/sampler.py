import os
import time
from EmuPBk.MCMC.core import Core
from EmuPBk.MCMC.like import LikeModule
from cosmoHammer.util import Params
from cosmoHammer import MpiCosmoHammerSampler
from cosmoHammer import CosmoHammerSampler
from cosmoHammer import LikelihoodComputationChain
#  from cosmoHammer.pso.MpiParticleSwarmOptimizer import MpiParticleSwarmOptimizer

# The parameter space is defined
# paramters = [peak, min., max., jump] ===> A rough idea about the prior

path = os.path.abspath(os.path.join(__file__, os.pardir))
path = path + '/existing_models/'
params = Params(("NoH", [275, 10, 550, 3]),
                ("n_ion", [90.00, 10.00, 180.00, 1]),
                ("R_mfp", [30.00, 5.00, 60.00, 0.5]))


class RunMCMC:

    """ sampler & MPI sampler class """
    def __init__(self, data, nbins, noise=0., div=1.0, name='pk'):
        """
        :param data: load your data
        :param nbins: number of k-modes in powerspectrum OR
         number of triangle contributions in bispectrum (for covariance matrix)
        :param noise: system noise, e.g. SKA, MWA noise response (if any), default 0.0,
        :param div: likelihood normalization factor, default 1.0,
        :param name: use ('pk','bk') for powerspectrum, bispectrum
        """
        chain = LikelihoodComputationChain(min=params[:, 1], max=params[:, 2])
        chain.params = params
        chain.addLikelihoodModule(LikeModule(data, nbins, noise, div))
        self.chain = chain
        self.name = name

    def load_existing_model(self):
        """
        Use the existing ANN models for MCMC analysis
        """

        if self.name == 'pk':
            self.chain.addCoreModule(Core(load_model=path+'pk.h5', norm=1.))

        elif self.name == 'bk':
            self.chain.addCoreModule(Core(load_model=path+'bk.h5', norm=100.))

        self.chain.setup()

    def load_model(self, load_model, name='pk', norm=1.):

        """
        :param load_model: load your own model, (give the path)
        :param name: name for data, ('pk','bk')==>for powerspectrum, bispectrum
        :param norm: rescale used in the training
        """
        self.name = name
        self.chain.addCoreModule(Core(load_model, norm))
        self.chain.setup()

    def mpi_sampler(self, walker_ratio, burnin, samples, num, threads=-1):
        """
        :param walker_ratio:  the ratio of walkers and the count of sampled parameters
        :param burnin: burin iterations
        :param samples: no. of sample iterations
        :param num: number to put in output files e.g: string(name+num)=Pk_1,Bk_1
        :param threads: no. of cpu threads
        """
        #   self.chain.setup()
        #   print("find best fit point")
        #   pso = MpiParticleSwarmOptimizer(self.chain, params[:, 1], params[:, 2])
        #   psoTrace = np.array([pso.gbest.position.copy() for _ in pso.sample()])
        #   params[:, 0] = pso.gbest.position
        mpi_sampler = MpiCosmoHammerSampler(params=params,
                                            likelihoodComputationChain=self.chain,
                                            filePrefix='%s' % self.name+'%d' % num,
                                            walkersRatio=walker_ratio,
                                            burninIterations=burnin,
                                            sampleIterations=samples, threadCount=threads)
        print("started sampling:")
        start = time.time()
        mpi_sampler.startSampling()
        end = time.time()
        tics = end - start
        print("The time taken %.2f sec. done!" % tics)
        print('Done!')

    def sampler(self, walker_ratio, burnin, samples, num, threads=-1):
        """
            :param walker_ratio:  the ratio of walkers and the count of sampled parameters
            :param burnin: burin iterations
            :param samples: no. of sample iterations
            :param num: number to put in output files e.g: string(name+num)=Pk_1,Bk_1
            :param threads: no. of cpu threads

            self.chain.setup()
            print("find best fit point")
            pso = MpiParticleSwarmOptimizer(self.chain, params[:, 1], params[:, 2])
            psoTrace = np.array([pso.gbest.position.copy() for _ in pso.sample()])
            params[:, 0] = pso.gbest.position
        """

        sampler = CosmoHammerSampler(
                params=params,
                likelihoodComputationChain=self.chain,
                filePrefix='%s' % self.name + '%d' % num,
                walkersRatio=walker_ratio,
                burninIterations=burnin,
                sampleIterations=samples, threadCount=threads)

        print("started sampling:")
        start = time.time()
        sampler.startSampling()
        end = time.time()
        tics = end-start
        print("The time taken %.2f sec. done!" % tics)
        print('Done!')
