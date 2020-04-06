import cosmoHammer
from Power_like import PSlikeModule as slk

chain = cosmoHammer.LikelihoodComputationChain()
chain.addLikelihoodModule(slk())

pso = cosmoHammer.ParticleSwarmOptimizer(chain, low=[-5,-5], high=[5,5], particleCount=40)

#returns all swarms and the best particle for all iterations
swarms, gbests = pso.optimize()

print(gbests[-1])