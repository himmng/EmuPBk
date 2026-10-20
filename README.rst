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

**Neural network emulators and a Bayesian inference pipeline for estimating
Epoch of Reionization parameters from 21-cm signal statistics.**

EmuPBk replaces an expensive forward simulation with a trained regression model,
which makes full Markov Chain Monte Carlo parameter estimation practical. It
emulates both the 21-cm power spectrum (a second-order statistic) and the 21-cm
bispectrum (a third-order statistic) as a function of three reionization
parameters, and uses those emulators inside a Bayesian inference pipeline to
recover the parameters with quantified uncertainty.

Documentation: https://emupbk.readthedocs.io/en/latest/

Why emulation
-------------

The Epoch of Reionization (EoR) is one of the least understood periods in the
history of the Universe. The redshifted 21-cm line from neutral hydrogen is the
most direct probe of it, but the signal is several orders of magnitude fainter
than the foregrounds, so it is studied through Fourier statistics rather than
images.

Constraining the reionization parameters from those statistics is a Bayesian
inference problem. The obstacle is cost: each likelihood evaluation requires a
simulated model statistic, and a semi-numerical EoR simulation takes a large
amount of memory and CPU time to produce a single forward model. An MCMC chain
needs hundreds of thousands of such evaluations, which is not feasible directly.

EmuPBk solves this by training an Artificial Neural Network to predict the
signal statistics directly from the parameters, reducing each model evaluation
to a fraction of a second while preserving accuracy.

Method
------

**Parameters.** The reionization model has three parameters:

- ``Mh_min``: minimum mass of the dark matter halo hosting ionizing sources
- ``R_mfp``: mean free path of ionizing photons, which sets the typical size of
  ionized regions
- ``N_ion``: ionizing efficiency of the sources

**Experimental design.** The three-dimensional parameter space is sampled using
Latin Hypercube sampling, which covers the space more efficiently than a regular
grid. 550 parameter sets were simulated, with 535 used for training and
validation and 15 held out as a test set.

**Training data.** Power spectra and bispectra were simulated with the
semi-numerical code `ReionYuga <https://github.com/rajeshmondal18/ReionYuga>`_
in a comoving volume of (215 Mpc)^3 at z = 8. Bispectra are estimated over all
unique triangle configurations in Fourier space, parameterised by a triangle
size ``k1`` and two shape parameters ``n`` and ``cos(theta)``.

**Emulators.** Two feed-forward networks, built in `keras <https://keras.io/>`_,
with ELU activations and the Adam optimiser, tuned with Keras Tuner. The power
spectrum emulator reaches approximately 98 per cent accuracy on both the
validation and held-out test sets; the bispectrum emulator reaches
approximately 93 per cent on the test set.

.. image:: /docs/source/test_emulations/pk.gif
    :width: 10%

.. image:: /docs/source/test_emulations/Bk.gif
    :width: 10%

**Error covariance.** The inference uses a realistic error budget rather than
signal alone. Two contributions are combined in quadrature: the sample variance
of the estimator, and the system noise expected from 1000 hours of observation
with SKA1-Low, computed from the proposed array configuration. Foregrounds, RFI
and calibration errors are assumed to have been perfectly removed.

**Inference.** Uniform priors over the sampled parameter ranges, a multivariate
Gaussian likelihood, and posterior sampling by MCMC using
`cosmoHammer <http://cosmo-docs.phys.ethz.ch/cosmoHammer/>`_, an affine-invariant
ensemble sampler built on `emcee <https://emcee.readthedocs.io/en/stable/>`_,
with 20 walkers taking 20,000 steps each.

.. image:: /docs/source/joint.jpg
    :width: 5%

Result
------

Using bispectra over all unique triangle shapes gives substantially tighter
constraints than the power spectrum alone. Relative uncertainty on ``Mh_min``
falls from 10 per cent to 4.5 per cent and on ``N_ion`` from 8.2 per cent to
4.3 per cent, an improvement of roughly a factor of 2 to more than 4 depending
on the stage of reionization. ``R_mfp``, which the power spectrum cannot
constrain at all, is recovered to within 32.4 per cent.

Bispectra restricted to a limited subset of triangle shapes perform worse than
the power spectrum, which is why all unique shapes are used.

Dependencies
------------

::

    numpy
    scipy
    keras / tensorflow
    cosmoHammer
    emcee
    matplotlib
    chainconsumer

Install
-------

::

    $ pip install EmuPBk

or from source::

    $ git clone https://github.com/himmng/EmuPBk.git
    $ cd EmuPBk
    $ python setup.py install --user

Citation
--------

If you use EmuPBk, please cite:

    H. Tiwari, A. K. Shaw, S. Majumdar, M. Kamran and M. Choudhury,
    *Improving constraints on the reionization parameters using 21-cm
    bispectrum*, Journal of Cosmology and Astroparticle Physics, 2022(04), 045.

- DOI: https://doi.org/10.1088/1475-7516/2022/04/045
- arXiv: https://arxiv.org/abs/2108.07279

Status and scope
----------------

EmuPBk currently supports a single reionization model and is under active
development. Known limitations, both documented in the paper:

- emulator performance degrades for models with a mean neutral fraction below
  about 0.5, because the Latin Hypercube design under-populates that region
- the analysis is carried out at a single redshift; combining multiple redshifts
  is expected to tighten the constraints further

Later work in the group has extended this approach to uncertainty-aware Bayesian
neural network emulators.
