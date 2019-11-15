import keras as ks
import numpy as np

#======================================================================================================================#

path = '/home/ht/PycharmProject/THANN/data'                   #========Loading the ANN from the directory==================#
model = ks.models.load_model(path+'EMuPk222.h5')   #========making the model ===============#

#======================================================================================================================#

def pk_pred(params):
    pk_pred = model.predict(params)             #======function will return the powerspectrum,given the parameters=====#

    return pk_pred

#======================================================================================================================#