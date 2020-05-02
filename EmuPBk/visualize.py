import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow import keras as ks
from chainconsumer import ChainConsumer
from numpy.random import normal
from mpl_toolkits import mplot3d


class Animate_Pk:
    '''
    Generelized function for animating comparison between simulated data(powerspectrum) vs ANN predictions (line plots)
    '''

    def __init__(self, test_data, test_params, k, load_model, rescale=1):
        '''
         :param: test_data : test_data(powerspectum)
         :param: test_params : test_parameters of EoR
         :param: k: corresponding k-values
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
            plt.suptitle('Simulated Powerspectrum vs. ANN predictions on test data-%d' % i)
            plt.plot(self.k, self.test_data[i], label='Simulated Powerspectrum', color='black', marker='o', )
            plt.plot(self.k, self.pk_pred[i], label='ANN predicted', color='blue')
            plt.legend(loc='lower right')
            filename = 'Pk_pred' + str(i) + '.jpg'
            plt.savefig(filename)
            plt.gca()

        os.system('convert -delay 150 Pk*.jpg Pk_pred.gif')
        os.system('rm Pk*.jpg')
        print('Pk_pred.gif file has been successfully saved at your current directory')


class Animate_Bk:
    '''
    simulation specific class, will not be so useful for more than 3 EoR parameters, with unique triangle parameter space
    for k1,k2,k3
    '''

    def __init__(self, test_data, test_params, load_model, xHI,
                 k1=0.19, cos_min=0.50, cos_max=0.99, cos_step=0.01,
                 k2byk1_min=0.50, k2byk1_max=1.00, k2byk1_step=0.05, rescale=1):

        '''
         It will give the animation of real bispectrum vs ANN predictions for only Unique triangle space configuration.
         :param: test_data : test_data(Bispectrum)
         :param: test_params : test_parameters of EoR
         :param: load_model: load_your own model from a directory path
         :param: k1: provide the value of k1
         :param: cosalpha: provide cos_min and cos_max and its step
         :param: k2byk1: provide the range of k2byk1 and its step

         '''

        self.test_data = test_data
        self.test_params = test_params
        self.xHI = xHI
        self.k1 = k1

        k2byk1 = np.arange(k2byk1_min, k2byk1_max + k2byk1_step, k2byk1_step)
        cosalpha = np.arange(cos_min, cos_max + cos_step, cos_step)
        self.cosalpha = cosalpha
        k2byk1 = np.reshape(k2byk1, (len(k2byk1), 1))
        self.k2byk1 = k2byk1
        self.condition = k2byk1 * cosalpha  # condition for unique triangle

        model = ks.models.load_model(load_model)
        Bk = model.predict(self.test_params)
        Bk = Bk * rescale

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

        '''
        :return: Animates real Bispectrum vs ANN predictions.
        '''

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

    def get_Bk_vs_cos(self, ):
        '''
        :return: gives a broad idea of ANN predictions over real bispectrum values, at different k2/k1 ratios.
        '''

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


class Get_Posterior:
    '''
    Gives the posterior distribution from MCMC analysis
    Some functions are generalized and some are specific to certain type of data (for now)
    :params : the parameter array contains the samplers walks in the paramter space:array( N*(no. of parameters) )
    :logl: the loglikelihood array contains the values of loglikelihood at each step done by sampler:array(N*1)
    '''

    def __init__(self,nop):
        '''
        :param nop: number of parameters
        '''

        self.c = ChainConsumer()
        self.params = params
        self.logL = logL
        self.nop = nop
        self.parameters = []

        for j in range(nop):
            self.parameters.append(str(input('name of param_%d: ' % (j + 1), )))



    def corner(self,index,dpi=120):
        '''
        :param index: index for plot name
        '''

        data = str(input('enter data_%d file name: ', ))
        params = np.loadtxt(data)
        self.c.add_chain(params, parameters=self.parameters,)

        fig = self.c.plotter.plot(figsize=1.0)

        fig.set_size_inches(3 + fig.get_size_inches())

        fig.savefig('corner_plot_%d.png'%index, dpi=dpi)


    def jointplot(self,index,dpi=120,):

        '''
        Generelized fuction can work with any data,with any no. of parameters
        index: data index name
        dpi: dots per inches for you plot (low value : low resolution,fast processing, high value: high. resolution, slow speed)
        :return: corner plot between the parameters. (default:120)

        Do in following manner:

        nod: the no. of different datasets you are using in joint plot
        name: name the data-set, you can write name in the 'tex-styling' as well.
        e.g.:$B(k_1,k_2,k_3)$ at $k_1=0.2 Mpc^{-1}$

        parameter: name of the individual parameters

        data: corresponding file name of the data,(or give the path to the data-set)

        :return: joint plot between all the given data

        '''
        truth = normal(size=3)
        color = ['red', 'blue', 'green', 'orange', 'pink', 'yellow']
        parameters = []

        nod = int(input('no. of different data-sets you are using: ', ))

        for i in range(nod):
            name = str(input('name of dataset_%d: ' % (i + 1), ))
            data = str(input('enter data_%d file name: ' % (i + 1), ))
            params = np.loadtxt(data)
            self.c.add_chain(params, parameters=self.parameters,
                        name="%s" % name, color=color[i])

        fig = self.c.plotter.plot(truth=truth)

        fig.set_size_inches(3 + fig.get_size_inches())

        fig.savefig('joint_plot_%d.png'%index, dpi=dpi)

        print('joint_plot.png successfully saved at your current directory')


    def animate_3D(self, a=0.5, b=0.2, alpha=0.7, cmap1='PuBuGn', cmap2='Blues', linecolor='lightgray', dpi=100,
                   delay=60):

        '''
        Note: Simulation Specific for EoR with 3 parameter (Nion,Rmfp,Mh_min),
         will not work if more than 3 parameters are there.

        :param a,b: adjust plots for better view (bispectrum is sensitive so most of parameters have very short sigma,
        for it use 0<(a,b)<2, for powerspectrum use 0<(a,b)<10) , (defaults a=0.5,b=0.2)
        :param alpha: transparency of horizontal and vertical lines showing the mean values.(default 0.7)
        :param cmap1: colormap for 3D plots (default:'PuBuGn')
        :param cmap2: colormap for KDE plots (default:'Blues')
        :param linecolor: color of lines (defaul:'lightgray')
        :param dpi: dots per inches for you plot (low value : low resolution,fast processing, high value: high. resolution, slow speed)
        :return: corner plot between the parameters. (default:100)
        :param delay: rotation time of the animation. (default:60ms)

        :return: Corner plots between parameters with 3D surface plots, in place of traditional histogram.

        '''

        nion, rmfp, noh = self.params[:, 0], self.params[:, 1], self.params[:, 2]
        mh = 1.087 * noh
        logl = self.logL
        nion_mean = np.mean(nion)
        nion_mean = np.around(nion_mean, 2)

        rmfp_mean = np.mean(rmfp)
        rmfp_mean = np.around(rmfp_mean, 2)

        mh_mean = np.mean(mh)
        mh_mean = np.around(mh_mean, 2)

        nion_min = min(nion)
        nion_max = max(nion)

        rmfp_min = min(rmfp)
        # rmfp_max = max(rmfp)

        mh_min = min(mh)
        mh_max = max(mh)

        for angle in range(70, 440, 10):
            fig = plt.figure(figsize=(15, 15))

            ax = plt.subplot(3, 3, 1, projection='3d', xticks=[False], yticks=[False], )

            ax.set_zticks([100])
            ax.set_xlabel(r'$\zeta$')
            ax.set_ylabel(r'$M_{halo}$')
            ax.plot_trisurf(nion, mh, logl, lw=1, cmap=cmap1)
            ax.view_init(30, angle)

            ax1 = plt.subplot(3, 3, 5, projection='3d', xticks=[False], yticks=[False])

            ax1.set_zticks([100])
            ax1.set_xlabel(r'$\zeta$')
            ax1.set_ylabel(r'$R_{mfp}$')
            ax1.plot_trisurf(nion, rmfp, logl, lw=1, cmap=cmap1)
            ax1.view_init(30, angle)

            ax2 = plt.subplot(3, 3, 9, projection='3d', xticks=[False], yticks=[False])

            ax2.set_zticks([100])
            ax2.set_xlabel(r'$R_{mfp}$')
            ax2.set_ylabel(r'$M_{halo}$')
            ax2.plot_trisurf(rmfp, mh, logl, lw=1, cmap=cmap1)
            ax2.view_init(30, angle)

            plt.subplot(3, 3, 4)

            plt.xticks([False])
            plt.ylabel(r'$R_{mfp}$', size=15)
            sns.kdeplot(nion, rmfp, cmap=cmap2, shade=True, shade_lowest=False)
            plt.hlines(y=rmfp_mean, xmin=(nion_min - a), xmax=nion_mean, lw=1.5, colors=linecolor, alpha=alpha)
            plt.vlines(x=nion_mean, ymin=(rmfp_min - b), ymax=rmfp_mean, lw=1.5, colors=linecolor, alpha=alpha)

            plt.subplot(3, 3, 7)

            plt.xlabel(r'$\zeta$', size=15)
            plt.ylabel(r'$M_{halo}(\times 10^8 M_\odot)$', size=15)
            sns.kdeplot(nion, mh, cmap=cmap2, shade=True, shade_lowest=False)
            plt.hlines(y=mh_mean, xmin=(nion_min), xmax=(nion_max + a), lw=1.5, colors=linecolor, alpha=alpha)
            plt.vlines(x=nion_mean, ymin=(mh_min - a), ymax=(mh_max + a), lw=1.5, colors=linecolor, alpha=alpha)

            plt.subplot(3, 3, 8)

            plt.yticks([False])
            plt.xlabel(r'$R_{mfp}$', size=15)
            sns.kdeplot(rmfp, mh, cmap=cmap2, shade=True, shade_lowest=False)
            plt.hlines(y=mh_mean, xmin=(rmfp_min - b), xmax=rmfp_mean, lw=1.5, colors=linecolor, alpha=alpha)
            plt.vlines(x=rmfp_mean, ymin=(mh_min - a), ymax=mh_mean, lw=1.5, colors=linecolor, alpha=alpha)

            filename = '3D-Posterior' + str(angle) + '.png'
            plt.savefig(filename, dpi=dpi)
        os.system('convert -delay %d 3D-Posterior*.png animated_3D_post.gif' % delay)
        os.system('rm 3D-Posterior*.png')

        print('animate_3D_post.gif successfully saved at your current directory')
