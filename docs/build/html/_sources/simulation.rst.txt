==============================
Populating the Parameter Space
==============================

Our primary task is the development of data-set to train the ANN models.
For this, one needs to populate the parameter space in such a way
that it covers the space effectively. One better approach is to use
Latin-Hypercube sampling (LH) method to generate sample points in the
relevant section of parameter space.
We have produced 550 LH sample points in this way to run simulations over them.
Our simulation model intakes three parameters (ζ, Rmfp, Mhmin) to generate
re-ionization scenario.

::

Out of 550 samples, 15 samples are used as test-set (lower triangular region)
and rest were used for the training-set(upper triangular region), see bottom-left figure.

.. image:: LH.jpg
   :width: 48%

.. image:: LH0.png
   :width: 48%
