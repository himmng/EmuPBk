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


Epoch of Reionization(EoR) is one of the crucial periods in the history of our Universe. The origin of the very first
stars & galaxies formed during this era are unknown mainly due to various observational challenges that prohibit the
detection of preferred H1 21-cm signal (hyperfine transition line from the neutral Hydrogen) coming from the EoR.
The Fourier-based signal statistics (e.g. 21-cm power spectrum and Bispectrum) thus provide a much-refined way to
probe the EoR. One way of characterizing EoR is constraining the reionization parameters using these statistical models
via methods like MCMC and Bayesian Inference.

ANN emulation of EoR simulations
---------------------------------
Simulation-based statistical models cast expansive computational overhead while performing the Bayesian
Inference. Thus, we used Artificial Neural Networks (ANN)-based emulation models for the Power spectrum and
Bispectrum to replace the simulation models.
We generated 550 samples of Power spectrum and Bispectrum by varying 3 parameters (Mhmin, Rmfp, Nion) to train the
networks. The parameters are minimum mass of the host dark matter halo, mean free path of ionizing photons
(i.e. also the relative size of the ionized region) and the ionizing efficiency of the ionizing photons, respectively.
We used semi-numarical code `ReionYuga <https://github.com/rajeshmondal18/ReionYuga>`_
to build the data sets for training and testing our ANNs. The documentation of the project can be found here
`ReadtheDocs <https://emupbk.readthedocs.io/en/latest/>`_.
ANN model evaluation of the unseen test-data,
The ANN models shows more than 90% accuracy in the
predictions.


..  image:: /docs/source/test_emulations/pk.gif
    :width: 10%

.. image:: /docs/source/test_emulations/Bk.gif
    :width: 10%

Parameter estimation
---------------------
We assume the situation where the foregrounds, RFIs and other artifacts are completely removed from
the signal. The signal thus only contributed by the sample variance and system noise of the observing instrument
(SKA1-low in this case). We simulated system noise of SKA1-low correspond to 1000 hours of observation time.

..  image:: /docs/source/joint.jpg
    :width: 5%



This package is limited to one re-ionization model and it is under  development.
-------------------------------------------------------------------------------
All ANNs' related tasks are done using `keras <https://keras.io/>`_, a python based deep-learning library,
For MCMC analysis we used python `cosmoHammer <http://cosmo-docs.phys.ethz.ch/cosmoHammer/>`_, which uses 
`emcee <https://emcee.readthedocs.io/en/stable/>`_
and for plotting and visualization we used `matplotlib <https://matplotlib.org>`_ and `chainconsumer <https://samreay.github.io/ChainConsumer/>`_.
 

 
