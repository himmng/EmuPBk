import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow import keras as ks

#==========These are default existing models=======================#
path = os.path.abspath(os.path.join(__file__, os.pardir))
path = path+'/tests/existing_models/'
pathpk = path+'pk99.h5'
pathbk02 = path+'BkANN02.h5'
pathbk03 = path+'BkANN03.h5'
pathbk15 = path+'BkANN15.h5'

class EMUPBK:
    ''' gives the Epoch of Reionization(EoR) 21-cm power spectrum
            and bispectrum given the 3 astrophysical
            parameter (Nion,Rmfp,NoH)
    :default: It will use existing trained models, but you can use it for newly trained
    models as well, (both for Powerspctrum and Bispectrum)
    '''
    def __init__(self,params):


        self.model = ks.models.load_model
        self.params = np.reshape(params,(len(params),len(params[0])))

    def PK(self, load_model = pathpk):
        '''The 21-cm powerspectrum EmuPBk based on Artificial Neural Networks(ANN),
            It is 99% accurate in the prediction of Power spectrum, given the 3 parameter array
            Input:
            :param params: array([Nion,Rmfp,NoH])
            :param path: default it use exisiting trained model, can be used for new
            models, e.g.: path = './' or 'current working directory'

            Output: P(k)
        '''
        model = self.model(load_model)
        pk = model.predict(self.params)

        return pk


    def BK02(self,rescale02=100,load_model=pathbk02):
        '''
        21-cm Bispectrum emulator for k1 = 0.2 mpc^-1
        :param params: array([Nion, Rmfp, NoH])
        :param path: default it use exisiting trained model, can be
        used for new models, e.g.: path = './' or 'current working directory'
        :return: Bk

        '''

        model = self.model(load_model)
        Bk02 = model.predict(self.params)
        Bk02 = Bk02*rescale02

        return Bk02


    def BK03(self,rescale03=10000,load_model=pathbk03):
        '''
        21-cm Bispectrum emulator for k1 = 0.3 mpc^-1
        :param params: array([Nion, Rmfp, NoH])
        :param path: default it use exisiting trained model, can be
        used for new models, e.g.: path = './' or 'current working directory'
        :return: Bk
        '''
        model = self.model(load_model)
        Bk03 = model.predict(self.params)
        Bk03 = Bk03*rescale03

        return Bk03


    def BK15(self,rescale15=10000000,load_model=pathbk15):
        '''
        21-cm Bispectrum emulator for k1 = 1.5 mpc^-1
        :param params: array([Nion, Rmfp, NoH])
        :param path: default it use exisiting trained model, can be
        used for new models, e.g.: path = './' or 'current working directory'
        :return: Bk
        '''

        model = self.model(load_model)
        Bk15 = model.predict(self.params)
        Bk15 = Bk15*rescale15


        return Bk15

class Animate_Pk:

    def __init__(self,test_data,test_params,k,load_model,rescale=1):
        '''
         :param: test_data : test_data(powerspectum)
         :param: test_params : test_parameters of EoR
         :param: load_model: load_your own model from a directory path
         :param: rescale: the rescaling you did during your model training, default is no rescaling =1
         :return a comparision plot animation between real test_data and predicted data.

         '''
        self.test_data = test_data
        self.test_params = test_params
        self.k = k
        model = ks.models.load_model(load_model)
        pk_pred = model.predict(test_params)
        self.pk_pred = pk_pred * rescale


    def get_animation_PK(self):

        for i in range(len(self.test_params)):
            plt.style.use('seaborn-pastel')

            plt.figure(figsize=(8, 6))
            plt.axes(xlim=(0, 3), ylim=(0, 105))
            plt.xlabel('k')
            plt.ylabel('P(k)')
            plt.suptitle('Simulated Powerspectrum vs. ANN predictions on test data-%d'%i)
            plt.plot(self.k,self.test_data[i], label='Simulated Powerspectrum', color='black', marker='o', )
            plt.plot(self.k,self.pk_pred[i], label='ANN predicted', color='blue')
            plt.legend(loc='lower right')
            filename = 'Pk_pred' + str(i) + '.jpg'
            plt.savefig(filename)
            plt.gca()

        os.system('convert -delay 150 Pk*.jpg Pk_pred.gif')
        os.system('rm Pk*.jpg')
        print('Pk_pred.gif file has been successfully saved at your current directory')

class Animate_Bk:

    def __init__(self,test_data,test_params,load_model,xHI,
                         k1= 0.19,cos_min=0.50,cos_max=0.99, cos_step=0.01,
                         k2byk1_min=0.50,k2byk1_max=1.00,k2byk1_step=0.05,rescale=1):

        '''
         It will give the animation of real bispectrum vs ANN predictions for only Unique triangle space configuration.
         :param: test_data : test_data(Bispectrum)
         :param: test_params : test_parameters of EoR
         :param: load_model: load_your own model from a directory path
         :param: k1: provide the value of k1
         :param: cosalpha: provide cos_min and cos_max and its step
         :param: k2byk1: provide the range of k2byk1 and its step

         :return a comparision plot animation between real test_data and predicted data.

         '''


        self.test_data = test_data
        self.test_params = test_params
        self.xHI = xHI
        self.k1 = k1


        k2byk1 = np.arange(k2byk1_min, k2byk1_max+k2byk1_step, k2byk1_step)
        cosalpha = np.arange(cos_min, cos_max+cos_step, cos_step)
        self.cosalpha = cosalpha
        k2byk1 = np.reshape(k2byk1,(len(k2byk1),1))
        self.k2byk1 = k2byk1
        self.condition = k2byk1*cosalpha  #condition for unique triangle

        model = ks.models.load_model(load_model)
        Bk = model.predict(self.test_params)
        Bk = Bk*rescale

        Bk_test = np.zeros(shape=(len(Bk), len(k2byk1), len(cosalpha)))
        Bk_pdct = np.zeros(shape=Bk_test.shape)
        for i in range(len(self.test_params)):
            Bk_pdct[i] = Bk[i].reshape(self.condition.shape)
            Bk_test[i] = test_data[i].reshape(self.condition.shape)

        for i in range(len(self.test_params)):
            for j in range(len(self.k2byk1)):
                for k in range(len(self.cosalpha)):
                    if self.condition[j][k] < 0.5:
                        Bk_pdct[i][j][k] = np.nan
                        Bk_test[i][j][k] = np.nan

        self.Bk_pdct = np.ma.masked_invalid(Bk_pdct)
        self.Bk_test = np.ma.masked_invalid(Bk_test)



    def get_animation_Bk(self):


        for index in range(len(self.test_params)):
            plt.figure(figsize=(16, 6))

            plt.tick_params(labelsize=10)

            plt.suptitle(
                r'At $k_1$ = {0:f},$x_H$ = {1:.3f}, $\zeta$ = {2:.2f}, $Rmfp$= {3:.2f}, $M_h$= {4:.2f}$\times 10^8 M_\odot$'
                .format(self.k1, self.xHI[index], self.test_params[index][0], self.test_params[index][1],
                        1.087 * self.test_params[index][2]), size=15)

            plt.subplot(1, 2, 1)
            plt.title('Bispectrum')
            plt.imshow(self.Bk_test[index], origin='lower', extent=[0.50, 0.99, 0.50, 1.00],
                       cmap="Spectral", )  # norm=colors.SymLogNorm(linthresh=1, linscale=1,
            # vmin=-10000.0, vmax=10000.0))
            # plt.plot(cosalpha,Bk_test[index][-1],'.-')
            cbar = plt.colorbar(label=r"$\frac{k_1^3.k_2^3.B(k_1,k_2,k_3).\delta T_b^3}{(2\pi^2)^2}$", )
            cbar.ax.yaxis.label.set_size(20, )
            plt.xlabel(r'$cos(\alpha)$')
            plt.ylabel(r'$k_2/k_1$')

            plt.subplot(1, 2, 2)
            plt.title('ANN prediction')
            plt.imshow(self.Bk_pdct[index], origin='lower', extent=[0.50, 0.99, 0.50, 1.00],
                       cmap="Spectral", )  # norm=colors.SymLogNorm(linthresh=1, linscale=1,
            # vmin=-10000.0, vmax=10000.0))
            # plt.plot(cosalpha,Bk_pdct[index][-1],'.-')
            cbar = plt.colorbar(label=r"$\frac{k_1^3.k_2^3.B(k_1,k_2,k_3).\delta T_b^3}{(2\pi^2)^2}$", )
            cbar.ax.yaxis.label.set_size(20, )
            plt.ylabel(r'$k_2/k_1$')
            plt.xlabel(r'$cos(\alpha)$')
            filename = 'Bk_pred' + str(index) + '.jpg'
            plt.savefig(filename)

            plt.gca()

        os.system('convert -delay 150 Bk*.jpg Bk_pred.gif')
        os.system('rm Bk*.jpg')
        print('Bk_pred.gif successfully saved at your current directory')


    def get_Bk_vs_cos(self,):

        for j in range(len(self.test_params)):
            plt.figure(figsize=(16, 12))
            plt.suptitle('Masked Simulated Bispectrum vs. ANN predictions at different k2/k1 ratios', fontsize=15)

            for i in range(1, len(self.k2byk1), 1):
                plt.title('k2/k1 = %.2f' % self.k2byk1[i - 1])
                plt.subplot(3, 4, i)
                plt.xticks([False])
                plt.yticks([False])
                BB = self.Bk_test[j][i]
                BP = self.Bk_pdct[j][i]
                plt.xlabel(r'cos($\alpha$)')
                plt.ylabel(r'$B(k_1,k_2,k_3)$')

                plt.plot(self.cosalpha, BB, '.-', label='real')
                plt.plot(self.cosalpha, BP, '.-', label='pred')
                BD = np.subtract(BB, BP)
                plt.plot(self.cosalpha, BD, '.-', label='diff', )
                plt.legend(loc='upper left')
            plt.title('k2/k1 = %.2f' % self.k2byk1[i])
            filename = 'Bk_vs_cos' + str(j) + '.png'
            plt.savefig(filename)
            plt.gca()
        os.system('convert -delay 150 Bk_vs_cos*.png Bk_vs_cos.gif')
        os.system('rm Bk_vs_cos*.png')

        print('Bk_vs_cos.gif animation saved at current directory')