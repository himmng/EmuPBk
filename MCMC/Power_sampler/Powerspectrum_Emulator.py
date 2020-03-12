import keras as ks

#======================================================================================================================#
path = '/home/ht/EmuPBk/data/data_Powerspectrum/'
                 #========Loading the ANN model=======#
model = ks.models.load_model(path+'pk99.h5')

#======================================================================================================================#

def pk_pred(params):
    pk_pred = model.predict(params)             #======function will return the powerspectrum,given the parameters=====#

    return pk_pred

#======================================================================================================================#