import keras as ks

#======================================================================================================================#

path = '/home/ht/Desktop/NN'                   #========Loading the ANN model from the directory=======#
model = ks.models.load_model(path+'/EMuPk222.h5')

#======================================================================================================================#

def pk_pred(params):
    pk_pred = model.predict(params)             #======function will return the powerspectrum,given the parameters=====#

    return pk_pred

#======================================================================================================================#