import numpy as np






'''The covarience matrix:
* It will contain the sample varience + Instrumental noise from SKA low telescope array.
'''
cov = np.zeros(shape=(Bk_02.shape))
cov_inv = 1./cov
