import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras as ks

model = ks.models.Sequential()
Dense = ks.layers.Dense


class ANN:

    def __init__(self, data, parameters, name='powerspectrum',
                 epochs=100, batch=10, optimizer='adam',
                 kernel_init='uniform', validation=0.038):
        """
                : data: must be an array (N*k_bins)
                : params: must be an array (N*n_parameters)
                : epochs: number of epochs for the training
                : batch: batch size
                : optimizer: choose the optimizer ('adam','adamax', 'graident_descent'),
                default: 'adam'
                :kernel_init: kernel_initilizier: default is 'uniform'
                :validation: The ration of validation to the training set, default:0.010,
                which is 10 out of 1000 training data, (10 for validation, 9990 for training)

                :return: A trained model

                e.g.: data =array([10,20,40,80,130],....N)
                params = array([10,20,50],...N)
        """
        self.name = name
        self.data = data
        self.parameters = parameters
        self.epochs = epochs
        self.batch = batch
        self.kernel_init = kernel_init
        self.optimizer = optimizer
        self.validation = validation

    def train_pk(self):
        """
        :return: For training ANN based 21-cm powerspectrum
        """

        model.add(Dense(3, input_shape=np.shape(self.parameters[0]),
                        activation='elu', kernel_initializer=self.kernel_init,))
        model.add(Dense(48, activation='elu', kernel_initializer=self.kernel_init))
        model.add(Dense(28, activation='elu', kernel_initializer=self.kernel_init))
        model.add(Dense(14, activation='elu', kernel_initializer=self.kernel_init))
        model.add(Dense(len(self.data[0]), activation='linear'))
        model.compile(loss='mse', optimizer=self.optimizer, metrics=['acc'])
        history = model.fit(self.parameters, self.data,
                            validation_split=self.validation,
                            epochs=self.epochs,
                            batch_size=self.batch)

        model.save('pk.h5')
        np.save('pk_history', history.history)
        print('The model pk.h5 saved!')

    def train_bk(self):
        """
        :return: For training ANN based 21cm Bispectrum
        """
        model.add(Dense(656, input_shape=np.shape(self.parameters[0]), activation='elu'))
        model.add(Dense(328, activation='elu'))
        model.add(Dense(len(self.data[0])))
        optimizer = ks.optimizers.Adam(lr=0.0001, )
        model.compile(loss='mse', optimizer=optimizer, metrics=['acc'], )
        history = model.fit(self.parameters, self.data,
                            validation_split=self.validation,
                            epochs=self.epochs,
                            batch_size=self.batch, )

        model.save('bk.h5')
        np.save('bk_history', history.history)
        print('bk.h5 model saved!')

    def train_bk_model_01(self):    # old version
        """
        :return: for training ANN based 21-cm Bispectrum
        """
        model.add(Dense(80, input_shape=np.shape(self.parameters[0]), activation='elu'))
        model.add(Dense(320, activation='elu'))
        model.add(Dense(460, activation='elu'))
        model.add(Dense(560, activation='elu'))
        model.add(Dense(260, activation='elu'))
        model.add(Dense(100, activation='elu'))
        model.add(Dense(len(self.data[0])))

        optimizer = ks.optimizers.Adamax(lr=0.0001, beta_1=0.2, beta_2=0.088)
        model.compile(loss='mse', optimizer=optimizer, metrics=['acc'], )
        history = model.fit(self.parameters,
                            self.data,
                            validation_split=self.validation,
                            epochs=self.epochs,
                            batch_size=self.batch, )

        model.save('Bk_model_01.h5')
        np.save('Bk_model_01_history', history.history)
        print('Bk_model_01.h5 model saved at current directory')

    def train_bk_model_02(self):    # old version
        """
        Train your Bispectrum data given the parameters
        Please normalize the data before training
        :return: A trained ANN based 21-cm Bispectrum EmuPBk
        """
        model.add(Dense(80, input_shape=np.shape(self.parameters[0]), activation='elu'))
        model.add(Dense(320, activation='elu'))
        model.add(Dense(460, activation='relu'))
        model.add(Dense(560, activation='elu'))
        model.add(Dense(260, activation='elu'))
        model.add(Dense(100, activation='elu'))
        model.add(Dense(len(self.data[0])))
        optimizer = ks.optimizers.Adam(lr=0.0001)
        model.compile(loss='mse',
                      optimizer=optimizer,
                      metrics=['acc'])
        history = model.fit(self.parameters,
                            self.data,
                            validation_split=self.validation,
                            epochs=self.epochs,
                            batch_size=self.batch, )

        model.save('bk_model_02.h5')
        np.save('bk_model_02_history', history.history)
        print('bk_model_02.h5 model saved at current location')

    def get_plot(self, history):
        """
        Please normalize the data before training
        e.g.: for Bispectrum at k1 = 0.2, divide data/100. is a good choice,

        return: gives the plot of model accuracy and model loss for the training set and the test set
        """

        plt.figure(figsize=(16, 6))

        plt.suptitle(r'Accuracy and Loss on the validation set for %s' % self.name)
        plt.subplot(1, 2, 1)
        c = list(np.arange(0, 1, 0.2))
        c.append(0.96)
        plt.yticks(c)
        plt.plot(history.history['acc'], lw=3, color='purple')
        plt.plot(history.history['val_acc'], lw=1, color='yellow', alpha=0.7)
        plt.hlines(y=0.96, xmin=0, xmax=self.epochs, linestyles='dashdot', lw=2)

        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['training', 'validation'], loc='lower right')

        plt.subplot(1, 2, 2)

        plt.plot(history.history['loss'], color='purple')
        plt.plot(history.history['val_loss'], color='orange')
        plt.hlines(y=0, xmin=0, xmax=self.epochs, linestyles='dashed', lw=1)
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['training', 'validation'], loc='upper right')
        plt.savefig('%s_acc._vs._loss.png' % self.name)
        print('Successfully saved the figure at current location')