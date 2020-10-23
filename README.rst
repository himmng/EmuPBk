======
EmuPBk
======


.. image:: https://badge.fury.io/py/EmuPBk.png
    :target: http://badge.fury.io/py/EmuPBk
    
.. image:: https://readthedocs.org/projects/emupbk/badge/?version=latest
            :target: https://emupbk.readthedocs.io/en/latest/?badge=latest



Epoch of Reionization(EoR) is one of the least known and yet important period
in the history of the Universe. The very first structures (Stars, Galaxies) were formed during this era. The 21-cm Powerspectrum and Bispectrum statistics
puts great possibility to probe this period.
We have developed two different ANN based 21-cm statistical signal emulators (21-cm Powerspectrum and Bispectrum emulator) based on Supervised machine learning
(Artificial Neural Network) to infer the upcoming future observations.
This module contains some trained ANN models from over 1000 simulated Powerspectrum and Bispectrum given the 3 EoR model parameters (Nion,Rmfp,Mhmin)- The ionizing efficiency, Mean free path of
ionizing photons(A proxy to the size of the ionized region) & Minimum halo mass of the host source.
We used a C-based semi-numarical code `ReionYuga <https://github.com/rajeshmondal18/ReionYuga>`_
to build the data-sets for training and testing our ANNs.



The structure of ANNs' are such, that anyone can build their own ANN model from their own datasets.
This module also simplifies MCMC analysis and posterior visualization.

All ANNs' related tasks were done using `keras <https://keras.io/>`_, a python based deeplearning library,
For MCMC analysis we used python `cosmoHammer <http://cosmo-docs.phys.ethz.ch/cosmoHammer/>`_, which uses 
`emcee <https://emcee.readthedocs.io/en/stable/>`_
and for plotting and visualization we used `chainconsumer <https://samreay.github.io/ChainConsumer/>`_


pip install::

    $ pip install EmuPBk


The documentation of EmuPBk is available at  `readthedocs.org <https://emupbk.readthedocs.io/en/latest/>`_
and the package is distributed over `PyPI <https://pypi.org/project/EmuPBk/>`_.
Help Contact: `himanshuhimang@gmail.com <himanshuhimang@gmail.com>`_



 

 

 
