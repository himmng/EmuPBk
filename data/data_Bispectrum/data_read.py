import numpy as np
from sklearn.model_selection import train_test_split



Bk_02 = np.loadtxt('Bk_02f')
params_02 = np.loadtxt('params_02f')


Ptr, Ptt, Btr, Btt = train_test_split(params_02,Bk_02,test_size=0.05)


'''The covarience matrix:
* It will contain the sample varience + Instrumental noise from SKA low telescope array.
'''
cov = np.zeros(shape=(Bk_02.shape))
cov_inv = 1./cov