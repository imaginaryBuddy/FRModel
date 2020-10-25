##############
Frame Plotting
##############

This provides a simple high-level interface of plotting a frame.

To start

.. code-block:: python

    f = Frame2D.from_image("path/to/file.png")
    fpl = f.plot()

This gives you a ``Frame2DPlot`` object to call plotting functions with.

If you only want to plot a few indexes, you can call it with arguments

.. code-block:: python

    from sklearn.preprocessing import robust_scale,

    f = Frame2D.from_image("path/to/file.png")
    fpl = f.plot([0, 1])

If the file is an RGB image, it'll just plot with R and G only.

There are multiple ways to generate the plots, listed in the Module Info.

=====================
Setting Subplot Shape
=====================

By default, if ``Frame2DPlot.subplot_shape`` is None, it'll find the best shape to fit the plot.

However, if you want to set a fixed shape, you can do so

.. code-block:: python

    ROWS = 3
    COLS = 2
    fpl.subplot_shape = (ROWS, COLS)

==============
Setting Titles
==============

By default, if ``Frame2DPlot.titles`` is None, it'll generate titles of **Index 0**, **Index 1**, ... and so on.

If you do set the title, take note that the number of titles must match the number of channels, else an
``AssertionError`` will be thrown.

=======
Example
=======

Here, we load in an image on 25% scale then plot the images with 22 rows and 1 column.

.. code-block:: python

    SCALE = 0.25
    f = Frame2D.from_image("path/to/file.png", scale=SCALE)
    fc = f.get_all_chns()

    fpl = fc.plot()
    ROWS = 22
    COLS = 1
    PLT_SCALE = 1.1
    fpl.subplot_shape = (ROWS, COLS)
    fpl.image(scale=PLT_SCALE)

    plt.show()

===========
Module Info
===========

.. automodule:: frmodel.base.D2.frame._frame_plot
