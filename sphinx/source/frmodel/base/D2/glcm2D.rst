###############################
Gray Level Co-occurrence Matrix
###############################

**The ``GLCM2D`` Class is deprecated from 0.0.3 onwards. However, the knowledge is still up-to-date**

=========================
Underlying Representation
=========================

The data stored isn't actually a co-occurrence matrix, instead :math:`2 * channels` ``np.ndarray``

========
Creation
========

---------
1 Channel
---------

Consider the following. ::

    o o o o o                      | o o o o |       <B>            | o o o o |
    o o o o o      .get_glcm()     | o o o o |   | o o o o |  func  | o o o o |
    o o o o o  --(by=1, axis=Y)--> | o o o o | + | o o o o |  --->  | o o o o |
    o o o o o                      | o o o o |   | o o o o |        | o o o o |
    o o o o o                          <A>       | o o o o |            <C>
     Channel

Notice that they have the same size, this allows us to use common ``numpy`` operations on it.

---------
n Channels
---------

Imagine the above, but we do that to multiple channels at once in parallel. This helps save a lot of time because
``numpy`` favors vectorized operations. However, it may limit some calculations.

=====================
Neighbour Convolution
=====================

In order to map these values to channels that just rely on a single cell, we have to rely on neighbour's values.

Consider this::

    .                 o: Centre, +: Neighbour
    | o o o o |       | + + + x | , | x + + + | , | x x x x | , | x x x x |
    | o o o o |       | + o + x | , | x + o + | , | x + + + | , | + + + x |
    | o o o o |  -->  | + + + x | , | x + + + | , | x + o + | , | + o + x |
    | o o o o |       | x x x x | , | x x x x | , | x + + + | , | + + + x |
        <C>
    x x x x x  =>                 Note that it's slightly off centre because of (1)
    x o o x x  =>  | o o |
    x o o x x  =>  | o o |
    x x x x x  =>
    x x x x x  =>
    Original       Transformed


=============
Edge Cropping
=============

Note that the larger the GLCM stride, the more edge pixels will be removed.

There will be edge cropping here, so take note of the following:

1) Edge will be cropped on GLCM Making (That is shifting the frame with by_x and by_y).
2) Edge will be further cropped by GLCM Neighbour convolution.

Going through the example again::

    1) GLCM Making, by_x = 1, by_y = 1
    o o o o o       | o o o o |       <B>            | o o o o |
    o o o o o       | o o o o |   | o o o o |  func  | o o o o |
    o o o o o  -->  | o o o o | + | o o o o |  --->  | o o o o |
    o o o o o       | o o o o |   | o o o o |        | o o o o |
    o o o o o           <A>       | o o o o |            <C>

    2) GLCM Neighbour Summation, radius = 1
                      o: Centre, +: Neighbour
    | o o o o |       | + + + x | , | x + + + | , | x x x x | , | x x x x |
    | o o o o |       | + o + x | , | x + o + | , | x + + + | , | + + + x |
    | o o o o |  -->  | + + + x | , | x + + + | , | x + o + | , | + o + x |
    | o o o o |       | x x x x | , | x x x x | , | x + + + | , | + + + x |
        <C>
    x x x x x  =>                 Note that it's slightly off centre because of (1)
    x o o x x  =>  | o o |
    x o o x x  =>  | o o |
    x x x x x  =>
    x x x x x  =>

The resultant size, if by_x = by_y, is :math:`frame.size - by - radius * 2`.

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

*Note: If :math:`std_{a,b} = 0`, then value is 1 or -1 depending on the sign.

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

