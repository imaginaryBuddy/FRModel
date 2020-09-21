############
Frame KMeans
############

This is an in-built k-means using ``sklearn.cluster.KMeans`` and ``seaborn`` for plotting.

The purpose of this is to quickly generate a kmeans run and plot the results.

*It will be limiting on how you can modify, if you need more flexibility you can just copy the source code and modify.*

========
Clusters
========

The number of kmeans clusters to start off with

===========
Fit Indexes
===========

The indexes to fit the kmeans to.

For example

.. code-block:: python

    frame_xy = f.get_chns(xy=True, hsv=True)
    frame_xy.kmeans(fit_indexes=[1,2,3], ...)

Will use the 2nd, 3rd, 4th channels to perform kmeans on.

That is, the **Y-Axis, Hue and Saturation**.

=============
Sample Weight
=============

A numpy array to set the weight for each record

======
Scaler
======

Scales the data before running kmeans

===============
Figure Plotting
===============

Set ``plot_figure == True`` to automatically let seaborn plot into the current figure.

Grab the figure with ``plt.gcf``.

----------
XY Indexes
----------

If your plot doesn't look right, it's likely due to the ``xy_indexes`` being incorrect.

``seaborn`` will generate the image based on these coordinates.

.. code-block:: python

    frame_xy = f.get_chns(xy=True, hsv=True)
    frame_xy.kmeans(fit_indexes=[1,2,3], xy_indexes=(0,1), ...)

In this case, we need to set it to 0 and 1 as they are the first 2 channels.

It is defaulted to 3 and 4 because if you include the rgb channels, it will be correct.

.. code-block:: python

    frame_xy = f.get_chns(self_=True, xy=True, hsv=True)
    frame_xy.kmeans(fit_indexes=[1,2,3], xy_indexes=(3,4), ...)

------------
Scatter Size
------------

This sets the scatter plot size.

-------------------------------
Y-Axis Inversion & Aspect Ratio
-------------------------------

Implicitly, these will be called::

    plt.gca().set_aspect('equal')
    plt.gca().invert_yaxis()

So the Y Axis may seem inverted, because the coordinate system starts from the top left.

=======
Example
=======

.. code-block:: python

    import matplotlib.pyplot as plt
    from sklearn.preprocessing import minmax_scale
    import seaborn as sns

    from frmodel.base.D2.frame2D import Frame2D
    f = Frame2D.from_image("rsc/imgs/test_img.jpg")

    sns.set_palette(sns.color_palette("Blues"))

    frame_xy = f.get_chns(xy=True, glcm_con=True, glcm_cor=True, glcm_ent=True,
                          glcm_by_x=1, glcm_by_y=1, glcm_radius=25, glcm_verbose=True,
                          glcm_entropy_mp=False, glcm_entropy_mp_procs=None)


    frame_xy.kmeans(clusters=5,
                    fit_indexes=[2,3,4,5,6,7,8,9,10],
                    plot_figure=True,
                    xy_indexes=(0, 1),
                    verbose=True,
                    scatter_size=0.4,
                    scaler=minmax_scale)

    plt.gcf().set_size_inches(f.width() / 96 * 2, f.height() / 96 * 2)
    plt.gcf().savefig('cluster.jpg', dpi=96)
