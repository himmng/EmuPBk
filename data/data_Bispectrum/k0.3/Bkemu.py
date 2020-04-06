from tensorflow import keras as ks
import numpy as np

model = ks.models.load_model('bkkkk0.3.h5')

#======================================================================================================================#

def Bk_pred(params):
    
    
    Bk_pdct = model.predict(params)
    Bk_pdct = Bk_pdct*10000.
    
    return Bk_pdct


def cov(Bk,Nbins):
    a = np.eye(550,550)
    b = Bk/np.sqrt(Nbins)
    b = np.reshape(b,(550,1))
    b = b*a
    cov_inv = np.linalg.inv(b)
    return cov_inv

