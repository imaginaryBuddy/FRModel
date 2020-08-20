###############################
Gray Level Co-occurrence Matrix
###############################

=========================
Underlying Representation
=========================

The data stored isn't actually a co-occurrence matrix, instead 2 ``np.ndarray``s.

These 2 arrays can be created manually or from :doc:`Channel2D <channel2D>`.

-----------------------
Creation from Channel2D
-----------------------

Consider the following. ::

    +-------+                     +-------+  data 1
    | 1 0 1 |                     | 1 0 1 | +-------+
    | 1 1 1 |        .glcm        | 1 1 1 | | 1 1 1 |
    | 0 0 0 | --(by=1, axis=Y)--> | 0 0 0 | | 0 0 0 |
    | 2 1 2 |                     | 2 1 2 | | 2 1 2 |
    | 0 1 1 |                     +-------+ | 0 1 1 |
    +-------+                       data0   +-------+
     Frame2D

Notice that they have the same size, this allows us to use common ``numpy`` operations on it.

==========
Statistics
==========

--------
Contrast
--------

Equation:

.. math::

    Con = \sum_{i,j=1} (i - j)^2 * GLCM_{(i,j)}

=======
Example
=======

1) In this, we firstly split it by 100x100 windows. (Note that uneven slices will be dropped by default)

2) Then loop through all windows.

3) Grab the red channel

4) Create the GLCM

5) Calculate all available statistics

.. code-block:: python

    from frmodel.base.consts import CONSTS
    from frmodel.base.D2.frame2D import Frame2D

    frame = Frame2D.from_image("path/to/file.jpg")

    frames = frame.split_xy(100)

    for xframes in frames:
        for frame in xframes:
            frame_red = frame.channel(CONSTS.CHANNEL.RED)
            glcm = frame_red.glcm(by=1, axis=CONSTS.AXIS.Y)
            glcm.contrast()
            glcm.correlation()
            glcm.entropy()

===========
Module Info
===========
