======
EmuPBk
======



.. image:: https://badge.fury.io/py/EmuPBk.png
    :target: http://badge.fury.io/py/EmuPBk


The Epoch of Reionization(EoR) is the one of the least and yet important period
in the history of Universe. It is supposed, that the first structures were
formed during this era. The 21-cm Powerspectrum and Bispectrum statistics
puts great possibility to probe the EoR. So, we have developed two different ANN based emulators
for each statistics. The 21-cm Powerspectrum and Bispectrum emulator are based on Supervised machine learning
(Artificial Neural Network).
The already existing model ANNs are trained over 1000 such simulated Powerspectrum and Bispectrum
for given 3 main parameters of EoR(Nion,Rmfp,Mhmin)- The ionizing efficiency, Mean free path of
ionizing photons(The size of the ionized region) & Minimum halo mass of the ionizing region.
We used semi-numarical simulation `ReionYuga <https://github.com/rajeshmondal18/ReionYuga>`_
to build the data-sets for training and testing of Artificial Neural Network(ANN).
To build the ANN model, we have used `tensorflow-keras <https://keras.io/>`_, a python based deeplearning library.



 

 

 