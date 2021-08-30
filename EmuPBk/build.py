import numpy as np
from tensorflow import keras as ks
import matplotlib as mpl
import matplotlib.pyplot as plt

model = ks.models.Sequential()
Dense = ks.layers.Dense


class ANN:
    '''
    class for building the simple ANN emulators
    '''
    def __init__(self, data, parameters, name='powerspectrum',
                 epochs=100, batch=10, optimizer='adam',
                 kernel_init='uniform', validation=0.08):
        """
                : data: training data (output features) (i.e. powerspectrum, bispectrum); dtype-:ndarray
                : parameters: training parameters (input features)
                (e.g. reionization parameters, k-modes); dtype-:ndarray
                : name: name of statistics (e.g. powerspectrum, bispectrum); type-: str
                : epochs: number of epochs for the training; type-: int
                : batch: batch size, type-: int
                : optimizer: choose the optimizer ('adam','adamax', 'graident_descent'),
                default: 'adam'; type-: str
                :kernel_init: kernel_initilizier: default is 'uniform'; type-: str
                :validation: The ration of validation to the training set; type-: float(range 0-1)
                default:0.1, which is 100 for 1000 sized training data,
                (100 for validation, 9900 for training)

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

    def customized_build(self):
        """
        (Coming soon) Build your customized neural network from scratch
        :return: model architecture
        """
        pass

    def hyperoptimize(self):
        """
        (Coming soon)
        :return: Hyperparameter optimized network (using scikit learn, keras-tuner)
        """
        pass

    def train_pk(self, early_stopping=True, thres_acc=.95):
        """
        : early_stopping: stop training if sufficiently trained; type-: bool
        : thres_acc: model accuracy upto which early stopping should run; type-: float (range 0-1),
        default value=.95, equivalant to 95% of threshold accuracy
        :return: Trained 21-cm powerspectrum emulator model and training history
        """
        callback = Early_stop(acc=thres_acc)
        model.add(Dense(28, input_shape=np.shape(self.parameters[0]),
                        activation='elu', ))
        model.add(Dense(14, activation='elu', ))
        model.add(Dense(len(self.data[0]), activation='linear'))
        model.compile(loss='mse', optimizer=self.optimizer, metrics=['acc'])
        if early_stopping==True:
            history = model.fit(self.parameters, self.data,
                                validation_split=self.validation,
                                epochs=self.epochs,
                                batch_size=self.batch,
                                callbacks=[callback] )
        else:
            history = model.fit(self.parameters, self.data,
                                validation_split=self.validation,
                                epochs=self.epochs,
                                batch_size=self.batch,)
        model.save('pk.h5')
        np.save('pk_history', history.history)
        print('The model pk.h5 and history saved!')

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

    def get_plot(self, history):
        """
        return: model accuracy and model loss of the training and validation set
        """
        plt.style.use('classic')
        mpl.rcParams['text.usetex'] = True
        mpl.rcParams['xtick.major.size'] = 8
        mpl.rcParams['xtick.minor.size'] = 4
        mpl.rcParams['xtick.major.width'] = 1.5
        mpl.rcParams['xtick.minor.width'] = 1.2
        mpl.rcParams['ytick.major.size'] = 8
        mpl.rcParams['ytick.minor.size'] = 4
        mpl.rcParams['ytick.major.width'] = 1.5
        mpl.rcParams['ytick.minor.width'] = 1.2
        mpl.rcParams['patch.linewidth'] = 1.8
        mpl.rcParams['axes.linewidth'] = 1.8

        plt.figure(figsize=(13, 4), facecolor='w', edgecolor='w')
        plt.suptitle(r'$\rm Accuracy~\&~Loss$', size=23)
        plt.subplot(1, 2, 1)
        plt.grid(True, lw=1)
        plt.ylim(0,1.2)
        plt.xlim(-10, len(history.item().get('val_loss'))+10)
        plt.plot(history.item().get('acc'), color='purple', lw=2)
        plt.plot(history.item().get('val_acc'), color='orange',lw=2)
        plt.xticks(size=16)
        plt.yticks(size=16)
        plt.ylabel(r'$\rm Accuracy$', size=18, )
        plt.xlabel(r'$\rm Epochs$', size=18, )

        plt.subplot(1, 2, 2)
        plt.grid(True, lw=1)
        plt.plot(history.item().get('loss'), color='purple', lw=2)
        plt.plot(history.item().get('val_loss'), color='orange', lw=2)
        plt.ylabel(r'$\rm Loss$', size=18, color='k', )
        plt.xticks(size=16, color='k', )
        plt.yticks(size=16, color='k')
        plt.ylim(-10, np.max(history.item().get('val_loss'))+10)
        plt.xlim(-10,len(history.item().get('val_loss'))+10)
        plt.xlabel(r'$\rm Epochs$', size=18, color='k')
        plt.legend([r'$\rm Training$', r'$\rm Validation$'], loc='best', prop={'size': 20}, labelcolor='k')
        plt.savefig('%s_acc._vs._loss.png' % self.name)
        print('Successfully saved the figure at current location')

class Early_stop(ks.callbacks.Callback):
    def __init__(self, acc):
        self.acc = acc
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('acc')>self.acc):
            print("\nReached d% accuracy so cancelling training!"%int(100*self.acc))
            self.model.stop_training = True


def test_train_split(data, parameters, xh, test_size):
    """
        : data: enter the data
        : parameters: enter the parameters
        : test_size: enter the size of your test set
        :return: split the data in test and train sets
    """
    ind = np.random.randint(low=0, high=len(data), size=test_size)
    params_test = parameters[ind]
    data_test = data[ind]
    xh_test = xh[ind]

    params_train = np.delete(parameters, ind, axis=0)
    data_train = np.delete(data, ind, axis=0)
    xh_train = np.delete(xh, ind, axis=0)

    return params_test, params_train, data_test, data_train, xh_test, xh_train