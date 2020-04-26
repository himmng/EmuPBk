======
EmuPBk
======



.. image:: https://badge.fury.io/py/EmuPBk.png
    :target: http://badge.fury.io/py/EmuPBk
    
.. image:: https://readthedocs.org/projects/emupbk/badge/?version=latest
            :target: https://emupbk.readthedocs.io/en/latest/?badge=latest


The Epoch of Reionization(EoR) is the one of the least known and yet important period
in the history of Universe. It is supposed, that the first structures were
formed during this era. The 21-cm Powerspectrum and Bispectrum statistics
puts great possibility to probe the EoR. So, we have developed two different ANN based emulators
for each statistics. The 21-cm Powerspectrum and Bispectrum emulator are based on Supervised machine learning
(Artificial Neural Network).
This module contains some already existing ANN models are trained over 1000 such simulated Powerspectrum and Bispectrum
for given 3 main parameters of EoR(Nion,Rmfp,Mhmin)- The ionizing efficiency, Mean free path of
ionizing photons(The size of the ionized region) & Minimum halo mass of the ionizing region.
And also, one can make their own models according to their datasets.
We used semi-numarical simulation `ReionYuga <https://github.com/rajeshmondal18/ReionYuga>`_
to build the data-sets for training and testing of Artificial Neural Network(ANN).
To build the ANN model, we have used `tensorflow-keras <https://keras.io/>`_, a python based deeplearning library.



 

 

 
