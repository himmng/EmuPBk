import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras as ks
from chainconsumer import ChainConsumer
from celluloid import Camera
import matplotlib.ticker as ticker
import imageio

path = os.path.abspath(os.path.join(__file__, os.pardir))
path = path+'/MCMC/existing_models/'
ex_models = ['pk.h5', 'bk.h5']
dr = './'
images = []


class AnimatePk:
    """
    Generelized function for animating comparison between
    simulated data(powerspectrum) vs ANN predictions
    """

    def __init__(self, test_data, test_params, k, xh_test, load_model=path+ex_models[0], norm=1.0):
        """
         :param: test_data : test_data(powerspectum)
         :param: test_params : test_parameters of EoR
         :param: k: corresponding k-values
         :param: load_model: load model from a directory path, default: uses existing models.
         :param: norm: normalizing factor, default is no rescaling = 1
         :returns: animation between test_data and ANN predictions

        """

        self.test_data = test_data
        self.test_params = test_params
        self.k = k
        self.xh_test = xh_test
        model = ks.models.load_model(load_model)
        pk_pred = model.predict(test_params)
        self.pk_pred = pk_pred * norm

    def get_animation_pk(self):
        fig = plt.figure(figsize=(10, 7))
        plt.style.use('seaborn')
        plt.grid(color='white', lw=1.5)
        camera = Camera(fig)
        for i in range(15):
            plt.suptitle('EoR 21-cm Powerspectrum, Simulation vs. ANN Prediction',
                         size=20, color='Darkblue')
            plt.xlabel(r'$\mathrm{k(Mpc^{-1})}$', size=20, color='darkblue')
            plt.ylabel(r'$\rm k^3P(k)/2\pi^2(mK^2)$', size=25, color='Darkblue')
            plt.xscale('log')
            plt.ylim(0, int(np.max(self.test_data)+80))
            plt.yscale('symlog')
            plt.plot(self.k, self.pk_pred[i], marker='o', ls='-', color='blue', lw=3)
            plt.plot(self.k, self.test_data[i], ls='--', lw=2, color='red')
            plt.legend(['ANN Prediction',
                        r' Powerspectrum for $\rm \zeta = {0:.2f}, '
                        r'Rmfp = {1:.2f}, Mhmin(10^8 M\odot) = {2:.1f}$'
                       .format(self.test_params[i, 1], self.test_params[i, 2],
                               self.test_params[i, 0], self.xh_test[i])],
                       loc='lower right', prop={'size': 13})
            camera.snap()
        animation = camera.animate()
        animation.save('pk.gif', fps=1, dpi=160)
        print('pk.gif saved!')


class AnimateBk:

    def __init__(self, bk_test, params_test, ntri, xh, load_model=path+ex_models[1],
                 cos_min=0.50, cos_max=0.99, cos_step=0.1,
                 k2byk1_min=0.50, k2byk1_max=1.00, k2byk1_step=0.05, norm=100.):
        """
        simulation specific function, will not be so useful on more than 3 EoR parameters.
        It will give the animation of real bispectrum vs ANN predictions
         for only Unique triangle space configuration.
         :param: test_data : test_data(Bispectrum)
         :param: test_params : test_parameters of EoR
         :param: ntri: number of triangles contributes in each bisepctrum.
         :param: xh : neutral fraction
         :param: load_model: load_your own model from a directory path, (default: uses existing model)
         :param: costheta: provide cos_min and cos_max and its step
         :param: k2byk1: provide the range of k2byk1 and its step
         :param: norm: normalizing factor, (default: 100.)
        """
        costheta = np.array([0.52, 0.57, 0.62, 0.67, 0.72, 0.77, 0.82, 0.87, 0.92, 0.97])
        self.costheta = costheta
        non_na_index = np.array([18, 19, 27, 28, 29, 35, 36, 37, 38, 39, 44, 45, 46,
                                 47, 48, 49, 53, 54, 55, 56, 57, 58, 59, 63, 64, 65,
                                 66, 67, 68, 69, 72, 73, 74, 75, 76, 77, 78, 79, 81,
                                 82, 83, 84, 85, 86, 87, 88, 89, 91, 92, 93, 94, 95,
                                 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109])
        self.k1 = np.array([0.1903934, 0.3220935, 0.5448941, 0.9218117, 1.559453])
        self.params_test = params_test
        self.xh = xh
        k2byk1 = np.arange(k2byk1_min, k2byk1_max + k2byk1_step, k2byk1_step)
        costheta = np.arange(cos_min, cos_max + cos_step, cos_step)
        self.costheta = costheta
        self.k2byk1 = np.reshape(k2byk1, (len(k2byk1), 1))
        # cond = costheta.reshape(len(costheta), 1) * k2byk1.reshape(1, len(k2byk1))
        # bk_test_dummy = np.reshape(bk_test, shape = (len(self.params_test), len(k2byk1) * len(costheta)))
        # ntri_dummy = np.reshape(ntri, shape = (len(k2byk1 * len(costheta))))
        model = ks.models.load_model(load_model)
        # costheta = np.arange(cos_min, cos_max + cos_step, cos_step)
        # self.costheta = costheta
        self.k2byk1 = np.reshape(k2byk1, (len(k2byk1), 1))
        cond = costheta.reshape(len(costheta), 1) * k2byk1.reshape(1, len(k2byk1))
        ntri_dummy = np.reshape(ntri, (len(self.k1), len(non_na_index)))
        bk_test_dummy = np.reshape(bk_test, (len(self.params_test), len(self.k1), len(non_na_index)))
        bk_pred_dummy = model.predict(self.params_test)
        bk_pred_dummy = bk_pred_dummy * norm
        bk_pred_dummy = np.reshape(bk_pred_dummy, np.shape(bk_test_dummy))
        bk_test = np.zeros(shape=(len(self.params_test), len(k2byk1) * len(costheta)))
        bk_pdct = np.zeros(shape=bk_test.shape)
        ntri = np.zeros(shape=len(k2byk1) * len(costheta))
        ntri[non_na_index] = ntri_dummy
        for i in range(len(self.xh)):
            bk_pdct[i][non_na_index] = bk_pred_dummy[i]
            bk_test[i][non_na_index] = bk_test_dummy[i]
        bk_test = np.zeros(shape=(len(self.params_test),
                                  len(self.k1), len(k2byk1) * len(costheta)))
        bk_pdct = np.zeros(shape=bk_test.shape)
        ntri = np.zeros(shape=(len(self.k1), len(costheta)*len(k2byk1)))
        for i in range(len(self.k1)):
            ntri[i][non_na_index] = ntri_dummy[i]

        for i in range(len(self.xh)):
            for j in range(len(self.k1)):
                bk_pdct[i][j][non_na_index] = bk_pred_dummy[i][j]
                bk_test[i][j][non_na_index] = bk_test_dummy[i][j]
        bk_test = np.reshape(bk_test, (len(xh), len(self.k1), len(k2byk1), len(costheta)))
        bk_pdct = np.reshape(bk_pdct, np.shape(bk_test))
        ntri = np.reshape(ntri, (len(self.k1), len(k2byk1), len(costheta)))

        for i in range(len(xh)):
            for j in range(len(self.k1)):
                for k in range(len(k2byk1)):
                    for l in range(len(costheta)):
                        if cond.T[k][l] < 0.50:
                            bk_test[i][j][k][l] = np.nan
                            bk_pdct[i][j][k][l] = np.nan
                            ntri[j][k][l] = np.nan

        self.bk_pdct = np.ma.masked_invalid(bk_pdct)
        self.bk_test = np.ma.masked_invalid(bk_test)
        self.ntri = np.ma.masked_invalid(ntri)

    def get_animation_bk(self):
        """
        :returns: Animation between simulated Bispectrum and ANN predictions.
        """
        for i in range(len(self.xh)):
            fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(26, 8), sharex='True', sharey='True')
            b = np.array([self.bk_test[i][0], self.bk_test[i][1], self.bk_test[i][2],
                          self.bk_test[i][3], self.bk_test[i][4],
                          self.bk_pdct[i][0], self.bk_pdct[i][1], self.bk_pdct[i][2],
                          self.bk_pdct[i][3], self.bk_pdct[i][4]])
            fig.text(0.435, 0.96,
                     r' Bispectrum, $\rm \zeta$ = {0:.2f}, Rmfp(Mpc) = {1:.2f},'
                     r' $\rm Mhmin(10^8 M_\odot)$ = {2:.1f}, $\rm xH1$ ={3:.3f}'
                     .format(np.float(self.params_test[i, 1]), np.float(self.params_test[i, 2]),
                             np.float(self.params_test[i, 0]), self.xh[i]),
                     ha='center', size=25, fontstyle='normal')
            for j, ax in zip(b, axes.flat):
                bounds = np.array([-5e3, -2e3, -1e3, 0, 1e3, 2e3, 3e3, 4e3, 5e3, 6e3, 7e3, 8e3, 9e3])
                im = ax.contourf(j, origin='lower', extent=[0.52, .97, 0.50, 1.00],
                                 cmap='seismic', extend='both',
                                 corner_mask=False, levels=bounds)
                fig.text(0.134, 0.9,
                         r'$\mathrm{k_1(Mpc^{-1})}$ = 0.19          '
                         r'$\mathrm{k_1(Mpc^{-1})}$ = 0.32          '
                         r'$\mathrm{k_1(Mpc^{-1})}$ = 0.54           '
                         r'$\mathrm{k_1(Mpc^{-1})}$ = 0.92           '
                         r'$\mathrm{k_1(Mpc^{-1})}$ = 1.55',
                         size=20)
                fig.text(0.435, 0.04, r'$\mathrm{cos(\theta)}$', ha='center', size=30)
                fig.text(0.05, 0.5, r'$\mathrm{k_2/k_1}$', va='center', rotation='vertical', size=30)
                fig.text(0.09, 0.71, r'Simulation', va='center', rotation='vertical', size=25)
                fig.text(0.09, 0.31, r'Emulation', va='center', rotation='vertical', size=25)
                fmt = ticker.ScalarFormatter(useMathText=True)
                fmt.set_powerlimits((0, 0))
                cbar = fig.colorbar(im, label=r"$(k_1^3.k_2^3.B(k_1,k_2,k_3)/(2\pi^2)^2)[mK^3]$",
                                    ax=axes.ravel().tolist(), format=fmt)
                cbar.ax.yaxis.label.set_size(25.)
                plt.savefig('Bk_%d.jpg' % i, dpi=120, bbox_inches='tight')
            fmt = ticker.ScalarFormatter(useMathText=True)
            fmt.set_powerlimits((0, 0))
            cbar = fig.colorbar(im, label=r"$\rm (k_1^3.k_2^3.B(k_1,k_2,k_3)/(2\pi^2)^2)[mK^3]$",
                                ax=axes.ravel().tolist(), format=fmt)
            cbar.ax.yaxis.label.set_size(25.)
            fig.text(0.435, 0.04, r'$\mathrm{cos(\theta)}$', ha='center', size=30)
            fig.text(0.05, 0.5, r'$\mathrm{k_2/k_1}$', va='center', rotation='vertical', size=30)
            fig.text(0.09, 0.71, r'Simulation', va='center', rotation='vertical', size=25)
            fig.text(0.09, 0.31, r'Emulation', va='center', rotation='vertical', size=25)
            plt.savefig('Bk_%d.jpg' % i, dpi=120, bbox_inches='tight')
        for file_name in os.listdir(dr):
            if file_name.endswith('.jpg'):
                file_path = os.path.join(dr, file_name)
                images.append(imageio.imread(file_path))
        imageio.mimsave('Bk.gif', images, fps=1)
        os.system('rm Bk_*.jpg')
    print('Bk.gif saved!')

    def get_bk_vs_cos(self,):
        """
        :return: gives a broad idea of ANN predictions over real
        bispectrum values, at different k2/k1 ratios.
        """
        for j in range(len(self.xh)):
            plt.figure(figsize=(16, 12))
            plt.suptitle(r'$\rm Bispectrum vs. ANN predictions at different k_2/k_1 ratios$',
                         fontsize=15)

            for i in range(1, len(self.k2byk1), 1):
                plt.title(r'$\rm k_2/k_1$ = %.2f' % self.k2byk1[i - 1])
                plt.subplot(3, 4, i)
                plt.xticks([False])
                plt.yticks([False])
                bb = self.bk_test[j][i]
                bp = self.bk_pdct[j][i]
                plt.xlabel(r'cos($\theta$)')
                plt.ylabel(r'$B(k_1,k_2,k_3)$')

                plt.plot(self.costheta, bb, '.-', label='real')
                plt.plot(self.costheta, bp, '.-', label='pred')
                bd = np.subtract(bb, bp)
                plt.plot(self.costheta, bd, '.-', label='diff', )
                plt.legend(loc='upper left')
                plt.title('k2/k1 = %.2f' % self.k2byk1[i])
                filename = 'Bk_vs_cos' + str(j) + '.jpg'
                plt.savefig(filename)
            for file_name in os.listdir(dr):
                if file_name.endswith('.jpg'):
                    file_path = os.path.join(dr, file_name)
                    images.append(imageio.imread(file_path))
        imageio.mimsave('Bk_vs_cos.gif', images, fps=1)
        os.system('rm Bk_vs_cos*.jpg')
        print('Bk_vs_cos.gif animation saved.')


class GetPosterior:
    """
    Returns the posterior distribution
    :params : the posterior parameter array
    walks in the paramter space: type array( N * (no. of parameters) )
    :logl: array contains the values of
    loglikelihood : type array(N*1)
    :name: Powerpsectrum, Bispectrum
    """

    def __init__(self, data):
        self.c = ChainConsumer()
        parameters = [r"$\zeta$", "$R_{mfp}$", r"$Mh_{min} \times 10^8 M\odot$"]
        self.parameters = parameters
        self.data = np.array([data[:, 1], data[:, 2], data[:, 0]])

    def corner(self, index, name='bk', color='blue', dpi=120):
        """
        :param index: index for the plot, type int
        :param name: statistics used, e.g. Powerspectrum, Bispectrum
        :param color: color
        :param dpi: dpi of the plot
        """
        self.c.add_chain(self.data, parameters=self.parameters, color=color)
        self.c.add_chain(self.data, parameters=['zeta', 'rmfp', 'noh'], color=color)

        fig = self.c.plotter.plot(figsize=1.0)

        fig.set_size_inches(3 + fig.get_size_inches())

        fig.savefig('{0:s}_{1:d}.jpg'.format(name, index), dpi=dpi)
        print('{0:s}_{1:d}.jpg saved'.format(name, index))