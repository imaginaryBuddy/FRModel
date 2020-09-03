###############################
Gray Level Co-occurrence Matrix
###############################

**This is deprecated from 0.0.3 onwards, however, the knowledge is still relevant**

=========================
Underlying Representation
=========================

The data stored isn't actually a co-occurrence matrix, instead 2 ``np.ndarray``

These 2 arrays can be created manually or from :doc:`Channel2D <channel2D>`.

***********************
Creation from Channel2D
***********************

Consider the following. ::

    +-------+                     +-------+   data1
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

Because of how GLCM holds data, some equations may be slightly different.

This is to allow efficient processing of arrays.

- *a* is data0 (array 0)
- *b* is data1 (array 1)

.. math::

    \sum_{x=0}^{x_n} \sum_{y=0}^{y_n} a_{xy} * b_{xy}

Means to loop through both arrays, for each index, multiply with each other and add to the sum.

********
Contrast
********

.. math::

    Con = \sum_{x=0}^{x_n} \sum_{y=0}^{y_n} (a_{x,y} - b_{x,y})^2

***********
Correlation
***********

.. math::

    mean_{a,b} &= mean(a) - mean(b) \\
    std_{a,b} &= std(a) * std(b) \\
    Corr &= \sum_{x=0}^{x_n} \sum_{y=0}^{y_n}
            \frac{a_{x,y} * b_{x,y} - {mean_{a,b}}}
                 {std_{a,b}}

*******
Entropy
*******

Note that we cannot directly use element wise operations here.

The algorithm has to count the pairs and square them.

**i** and **j** represent the co-occurrence matrix indexes

.. math::

    Con = \sum_{i=0}^{i_n} \sum_{j=0}^{j_n} GLCM_{(i,j)}^2

=======
Example
=======

Create the GLCM from a channel then calculate all available statistics

``frame`` is a ``Frame2D`` object instance.

.. code-block:: python

    from frmodel.base.consts import CONSTS\

    frame_red = frame.channel(CONSTS.CHANNEL.RED)
    glcm = frame_red.glcm(by=1, axis=CONSTS.AXIS.Y)
    glcm.contrast()
    glcm.correlation()
    glcm.entropy()
