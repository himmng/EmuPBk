======
EmuPBk
======


.. image:: https://badge.fury.io/py/EmuPBk.png
    :target: http://badge.fury.io/py/EmuPBk
    
.. image:: https://readthedocs.org/projects/emupbk/badge/?version=latest
            :target: https://emupbk.readthedocs.io/en/latest/?badge=latest


The Epoch of Reionization(EoR) 21-cm Powerspectrum and Bispectrum emulators based on Supervised machine learning(Artificial Neural Network). We have used the semi-numarical
simulation `ReionYuga <https://github.com/rajeshmondal18/ReionYuga>`_ to build the data-sets for training and testing of Artificial Neural Network(ANN). To build the ANN model, we have used
`tensorflow-keras <https://keras.io/>`_, a python based deeplearning library.

The 21-cm Powerspectrum and Bispectrum statistics put great possibility to probe the EoR.
So, we have developed two different ANN based emulators for each statistics.

Package includes 4 already trained ANN models 1 for Powerspectrum & 3 for Bispectrum
for different k-modes. These ANNs are trained over 1058 such simulated Powerspectrum and Bispectrum for given 3 main parameters of EoR
(Nion,Rmfp,Mhmin)- The ionizing efficiency, Mean free path of ionizing photons(The size of the ionized region) & Minimum dark matter halo mass that hosts these ionizing source.

But, anyone can train their own model, and get the accuracy and loss details.

Easy install::

    $ pip install EmuPBk

Fast train your own model::

    from EmuPBk import ANN

    '''
    Ready with your (data,params)
    data   => (powerspectrum or Bispectrum)
    params =>  EoR parameters
    '''

    model = ANN.Model(data,params)

    Pk = model.train_Pk()            # if training powerspectrum

    Bk = model.train_BK_model_01()   # if training Bispectrum


The documentation of EmuPBk is available at  `readthedocs.org <https://emupbk.readthedocs.io/en/latest/>`_
and the package is distributed over `PyPI <https://pypi.org/project/EmuPBk/>`_.
Help Contact: `himanshuhimang@gmail.com <himanshuhimang@gmail.com>`_



 

 

 
