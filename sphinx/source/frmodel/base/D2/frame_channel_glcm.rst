#####################################
Frame Gray Level Co-occurrence Matrix
#####################################

*The GLCM2D Class is deprecated from 0.0.3 onwards, replaced by the current in-built methods*

=========================
Underlying Representation
=========================

The data stored isn't actually a co-occurrence matrix, instead :math:`2 * channels` ``np.ndarray``

=========
Algorithm
=========

-------------
Offset Arrays
-------------

1 Channel
#########

Consider the following. ::

    o o o o o                      o o o o .   . . . . .          o o o o
    o o o o o      .get_glcm()     o o o o .   . o o o o  f(a,b)  o o o o
    o o o o o  --(by=1, axis=Y)--> o o o o . + . o o o o  ----->  o o o o
    o o o o o                      o o o o .   . o o o o          o o o o
    o o o o o                      . . . . .   . o o o o
     Channel                        Array A     Array B           Array C
     [5 x 5]                        [4 x 4]     [4 x 4]           [4 x 4]

Notice that they have the same size, this allows us to use common ``numpy`` operations on it.

n Channels
##########

Imagine the above, but we do that to multiple channels at once in parallel. This helps save a lot of time because
``numpy`` favors vectorized operations. However, it may limit some calculations.

---------------------
Neighbour Convolution
---------------------

In order to map these values to channels that just rely on a single cell, we have to rely on neighbour's values.

Consider this::

    .              o: Centre, +: Neighbour
    o o o o        + + + .  ,  . + + +  ,  . . . .  ,  . . . . 
    o o o o        + o + .  ,  . + o +  ,  . + + +  ,  + + + . 
    o o o o   -->  + + + .  ,  . + + +  ,  . + o +  ,  + o + . 
    o o o o        . . . .  ,  . . . .  ,  . + + +  ,  + + + . 
    Array C
                   . . . .  +  . . . .  +  . . . .  +  . . . .
      x = g(o, +)  . x . .  +  . . x .  +  . . . .  +  . . . .
              -->  . . . .  +  . . . .  +  . x . .  +  . x . .
                   . . . .  +  . . . .  +  . . . .  +  . . . .

                   . . . .
                   . x x .
              -->  . x x .
                   . . . .

With a radius 1, we sum all of the neighbour values together to calculate a GLCM at a specific cell.

To further clarify, we do not repeatedly repeat GLCM creation, we create the 2 offset arrays, multiplied by the number
of GLCM statistics, then do a convolution at each feasible point.

-------------
Edge Cropping
-------------

We will not resort to padding due to difficulty and minimal impact on result (a small amount of pixels removed).

Note that the larger the GLCM stride, the more edge pixels will be removed.

There will be edge cropping here, so take note of the following:

1) Edge will be cropped on GLCM Making (That is shifting the frame with by_x and by_y).
2) Edge will be further cropped by GLCM Neighbour convolution.

Going through the example again::

    1) GLCM Making, by_x = 1, by_y = 1
    o o o o o                      o o o o .   . . . . .          o o o o
    o o o o o      .get_glcm()     o o o o .   . o o o o  f(a,b)  o o o o
    o o o o o  --(by=1, axis=Y)--> o o o o . + . o o o o  ----->  o o o o
    o o o o o                      o o o o .   . o o o o          o o o o
    o o o o o                      . . . . .   . o o o o
     Channel                        Array A     Array B           Array C
     [5 x 5]                        [4 x 4]     [4 x 4]           [4 x 4]

    2) GLCM Neighbour Summation, radius = 1
    .              o: Centre, +: Neighbour
    o o o o        + + + .  ,  . + + +  ,  . . . .  ,  . . . .
    o o o o        + o + .  ,  . + o +  ,  . + + +  ,  + + + .
    o o o o   -->  + + + .  ,  . + + +  ,  . + o +  ,  + o + .
    o o o o        . . . .  ,  . . . .  ,  . + + +  ,  + + + .
    Array C
                   . . . .  +  . . . .  +  . . . .  +  . . . .
      x = g(o, +)  . x . .  +  . . x .  +  . . . .  +  . . . .
              -->  . . . .  +  . . . .  +  . x . .  +  . x . .
                   . . . .  +  . . . .  +  . . . .  +  . . . .

                   . . . .
                   . x x .         x x
              -->  . x x .  -->    x x
                   . . . .
                                 Array D
                                 [2 x 2]

The resultant size, if by_x = by_y, is :math:`frame.size - by - radius * 2`.

------------------------
Non-GLCM Channel Fitting
------------------------

Because GLCM slightly compresses the frame size, we need to somehow fit the other channels into a new shape.

That's where Gaussian Convolution comes in.

By creating a Gaussian kernel with ``scipy.signal.windows.gaussian``,
then using a Fast Fourier Transform Convolution ``scipy.signal.fftconvolve``,
we apply convolution to the channel axis for all other channels.

This is an improvement from the previous convolution, where it just attempts to average overlapping values.

**Note:** Gaussian standard deviation is controlled by ``conv_gauss_stdev``.

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

--------
Contrast
--------

.. math::

    Con = \sum_{x=0}^{x_n} \sum_{y=0}^{y_n} (a_{x,y} - b_{x,y})^2

Implementation
##############

Assume ``rgb_a`` and ``rgb_b`` are the offset arrays.

.. code-block:: python

    ar = (rgb_a - rgb_b) ** 2
    return fftconvolve(ar, np.ones(shape=[radius * 2 + 1, radius * 2 + 1, 1]), mode='valid')

This one is pretty simple, we create a calculated array for every cell, then convolve over it with
a specified ``np.ones`` kernel the size of the window.

-----------
Correlation
-----------

Note: If :math:`std_{a,b} = 0`, then value is 1 or -1 depending on the sign.

.. math::

    mean_{a,b} &= mean(a) - mean(b) \\
    std_{a,b} &= std(a) * std(b) \\
    Corr &= \sum_{x=0}^{x_n} \sum_{y=0}^{y_n}
            \frac{a_{x,y} * b_{x,y} - {mean_{a,b}}}
                 {std_{a,b}}

Implementation
##############

Assume ``rgb_a`` and ``rgb_b`` are the offset arrays.

This is pretty complicated, so let's break it down into smaller parts.

Variance Formula
================

Variance can be alternatively expressed as :math:`var = E(X^2) - E(X)^2`, from here we just ``sqrt`` to get
the stdev.

This alternative formula allows us to use vectorization and convolution as a solution to the problem.

This can be achieved by the following code only for offset array A

.. code-block:: python

    conv_a = fftconvolve(rgb_a, kernel, mode='valid')
    conv_ae = conv_a / kernel.size  # E(A)

    conv_a2 = fftconvolve(rgb_a ** 2, kernel, mode='valid')
    conv_ae2 = conv_a2 / kernel.size  # E(A^2)

    conv_ae_2 = conv_ae ** 2  # E(A)^2
    conv_stda = np.sqrt(np.abs(conv_ae2 - conv_ae_2))

Correlation
===========

Correlation is then calculated like so, note that :math:`E(x)` is just the mean.

:math:`Corr = (a * b - (E(a) - E(b))) / std(a) * std(b)`

Error Correction
================

If either stdev are 0, we cap it to -1 or 1 depending on the numerator.

This only happens if we happen to convolute a perfectly monotonous window.

.. code-block:: python

    with np.errstate(divide='ignore', invalid='ignore'):
        cor = (conv_ab - (conv_ae - conv_be)) / conv_stda * conv_stdb
        return np.nan_to_num(cor, copy=False, nan=0, neginf=-1, posinf=1)

-------
Entropy
-------

Note that we cannot directly use element wise operations here.

The algorithm has to count the pairs and square them.

**i** and **j** represent the co-occurrence matrix indexes

.. math::

    Con = \sum_{i=0}^{i_n} \sum_{j=0}^{j_n} GLCM_{(i,j)}^2

Implementation
##############

Entropy doesn't use convolution, it receives a prepared window the runs ``bin_count`` on it.

For the ``bin_count``, it's then squared then summed to get entropy for the channel.

**Note:** As of v0.0.4, ``bin_count`` replaces ``unique`` for performance.

Multiprocessing
###############

As Entropy uses a for loop, we can deploy multiple processes to parallelise it.

It simply uses the in-built ``multiprocessing.Pool`` class to execute it.
