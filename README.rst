======
EmuPBk
======



.. image:: https://badge.fury.io/py/EmuPBk.png/
    :target: http://badge.fury.io/py/EmuPBk/

.. image:: https://img.shields.io/pypi/l/ansicolortags.svg
   :target: https://pypi.python.org/pypi/ansicolortags/

.. image:: https://readthedocs.org/projects/emupbk/badge/?version=latest&style=for-the-badge
            :target: https://emupbk.readthedocs.io/en/latest/?badge=latest
.. image:: http://ForTheBadge.com/images/badges/made-with-python.svg
   :target: https://www.python.org/



Epoch of Reionization(EoR) is one of the least known and yet important period
in the history of the Universe. The very first structures (stars & galaxies) are supposed to
form during this era. The 21-cm powerspectrum and bispectrum puts great possibility to probe the EoR.
Therefore, We have developed ANN based 21-cm statistical signal emulators (Powerspectrum and Bispectrum) based on
Supervised machine learning (Artificial Neural Network).
This module contains trained ANN models with 550 LH (Latin-Hypercube sample) simulated
Powerspectrum and Bispectrum given the 3 EoR parameters (ζ, Rmfp, Mhmin) - The ionizing efficiency,
Mean free path of ionizing photons (The size of the ionized region) & Minimum halo mass of the ionized region.
We used a C-based semi-numarical code `ReionYuga <https://github.com/rajeshmondal18/ReionYuga>`_
to build the data-sets for training and testing our ANNs.

emulation of EoR simulations
----------------------------




..  image:: /docs/source/Pk_pred.gif

This package is limited to one re-ionization model and it is under development.
-------------------------------------------------------------------------------
The documentation of the project can be found here `ReadtheDocs <https://emupbk.readthedocs.io/en/latest/>`_.
This package is limited to one re-ionization model and it is under development.
However, It can be installed using pip:

::


    $ pip install EmuPBk

::

 
All ANNs' related tasks are done using `keras <https://keras.io/>`_, a python based deep-learning library,
For MCMC analysis we used python `cosmoHammer <http://cosmo-docs.phys.ethz.ch/cosmoHammer/>`_, which uses 
`emcee <https://emcee.readthedocs.io/en/stable/>`_
and for plotting and visualization we used `matplotlib <https://matplotlib.org>`_ and `chainconsumer <https://samreay.github.io/ChainConsumer/>`_.
 

 
