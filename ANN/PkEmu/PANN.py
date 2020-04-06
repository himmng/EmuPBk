from ANN.Emu import EANN
import numpy as np


def PANN() -> object:
    params = np.array([30,20,50])
    model = 'Pk.h5'
    Pk = EANN.model(params,model)
    return Pk

print(PANN())