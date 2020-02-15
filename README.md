# EmuPBk
21-cm powerspectrum emulator based on Artificial Neural Network, It is trained over 1065 simulated powerspectrums for a 
wide EoR parameter ranges at a redshift (z) of 9.210. It capable of reproducing the 21-cm powerspectrums with more than 
98% accuracy compared to simulated 21-cm powerspectrum.

* These emulated 21-cm powerspectrum signals then can be used as a theoretical model for MCMC analysis, replacing the
 time consuming EoR simulations.
==================================================



### How one can use the Trained ANN

##### Note: The ANN is trained for a specific redshift z = 9.210. So, It will not be able to work well for other redshifts.

* The ANN is trained for a specific redshift z = 9.210. So, the observational data should be at same redshift
 (corresponding wavelength of H1-line will be 21*(1+z)cm = 2.1441 meters). It will not be able to work well for other redshifts.

* For paramete-estimation and Baysian MCMC, one has to provide the only 3 input parameters fo Epoch of Reionization,
 (Nion,Rmfp,MHmin) - which are Ionizing efficiency, Mean free path of ionizing photons & Minimum halo mass 
that produces those ionizing photons in the same order shown here.

 
