# EmuPBk
 The Epoch of Reionization(EoR) 21-cm Powerspectrum and Bispectrum emulators based on **Supervised machine learning**(Artificial Neural Network). We have used the semi-numarical 
EoR simulation **[Reion-Yuga](https://github.com/rajeshmondal18/ReionYuga)** to build the datasets for training and testing of the Artificial Neural Network(ANN). To build the ANN model, we have used 
**[Keras](https://keras.io/)**, a python based deeplearning library.

Powerspectrum and Bispectrum are two different statistics used to probe the early Universe. So, we have developed two different
ANN based emulators for each statistics.

These ANNs are trained over **1065** such simulated Powerspectrum and Bispectrum for given 3 main parameters of EoR
**(Nion,Rmfp,MHmin)**- The Ionizing efficiency, Mean free path of ionizing photons & Minimum halo mass 
that produces those ionizing photons in the same order shown here.

* The ANN is trained for a specific redshift **z = 9.210**. So, the observational data should be at same redshift, the
 corresponding wavelength of H1-line will be 21*(1+z)cm = 2.1441 meters, which is consistant with the future 
 observations of EoR using telescopes, such as **[SKA](https://www.skatelescope.org/)**. 
 
* These emulated 21-cm Powerspectrum & Bispectrum signals then can be used as a theoretical models inplace of simulations.

* We can further use these Emulators for a **MCMC Bayesian analysis**. This remarkably speeds up the parameter estimation task.


**Note: The ANN is trained for a specific redshift z = 9.210. So, It will not be able to work well for other redshifts.**


 

 

 
