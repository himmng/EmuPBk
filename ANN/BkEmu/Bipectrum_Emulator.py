import keras as ks

#======================================================================================================================#
path ='/home/ht/EmuPBk/data/data_Bispectrum/k0.2/'
                 #========Loading the ANN model=======#
model = ks.models.load_model(path+'98bk02.h5')

#======================================================================================================================#

def Bk_pred(params):
    Bk_pred = model.predict(params)  #======function will return the powerspectrum,given the parameters=====#
    Bk_pred = Bk_pred*100.

    return Bk_pred

#======================================================================================================================#