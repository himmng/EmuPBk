


Joint Plot
----------------------


::

        from EmuPBk import visualize
        import numpy as np

        # This fuction will take different date-sets from current directory, and gives a jointplot,

        plot = visualize.Get_Posterior() # no parameters should go inside this, leave it blank in this case
        plot.jointplot()

.. image:: joint_plot.png
   :alt: joint plot
   :align: center


Posterior Surface Plot
----------------------

::



        params = np.loadtxt('path/to/file')
        plot = visualize.Get_Posterior(params=params)
        plot.animate_3D(a, b, alpha, cmap1, cmap2, linecolor, dpi, delay)


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


.. image:: animated_3D_post.gif
   :alt: posterior animation plot in 3d.
   :align: center
