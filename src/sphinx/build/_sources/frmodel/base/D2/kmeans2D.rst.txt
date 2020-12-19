############
Frame KMeans
############

As of ``0.0.4``, the API has been loosened to introduce more features.

========
Creation
========

Grab this class ``from frmodel.base.D2.kmeans2D import KMeans2D``.

``KMeans2D`` would have the signature of

.. code-block:: python

    def __init__(self,
                 frame: Frame2D,
                 model: KMeans,
                 fit_indexes,
                 frame_1dmask: np.ndarray = None,
                 scaler=None):

The ``frame`` argument is for a Frame2D.

----------------
KMeans Modelling
----------------

The API allows you to set up your own ``KMeans`` model object, hence most
parameters such as ``clusters`` can be set-up outside ``frmodel`` pre-fitting.

-----------
Fit Indexes
-----------

The indexes to fit the kmeans to.

For example

.. code-block:: python

    frame_xy = f.get_chns(xy=True, hsv=True)
    frame_xy.kmeans(fit_indexes=[1,2,3], ...)

Will use the 2nd, 3rd, 4th channels to perform kmeans on.

That is, the **Y-Axis, Hue and Saturation**.

--------------
Frame 1-D Mask
--------------

This mask is to remove certain data points from the ``Frame2D``. This is useful if you don't want to run K-Means on all
data.

The mask must be:
- an ``np.ndarray`` of boolean or truthy values.
- in 1-Dimension, hence, call ``flatten()`` before passing it as an arugment.
- the same size as the ``Frame2D`` passed as argument.

------
Scaler
------

Scales the data before running kmeans, must be a **Callable**

If ``None``, no scaling is done!

===============
Figure Plotting
===============

Plotting is done with :doc:`Frame2D Plotting <frame_plot>`.

=====
Score
=====

--------------
Custom Scoring
--------------

To test out how well the clustering works, we can mimic **supervised learning**.

We can have another image (Score Image) that shows the expected grouping of clusters,
by simply filling another image with same dimensions with different gray-scales.

We then pair the labelled KMeans and Score gray-scale labels to find out the
maximum score attainable.

For example::

    [ORIGINAL]
        KMEANS      SCORE    COUNT
    [1] LABEL A <-> LABEL A  1000
    [2] LABEL A <-> LABEL B  500
    [3] LABEL B <-> LABEL A  2000
    [4] LABEL B <-> LABEL B  4000

If we wanted the highest score attainable, we look at the top values::

    [SORTED BY COUNT]
        KMEANS      SCORE    COUNT
    [4] LABEL B <-> LABEL B  4000
    [3] LABEL B <-> LABEL A  2000
    [1] LABEL A <-> LABEL A  1000
    [2] LABEL A <-> LABEL B  500

If we picked ``[4]`` here, we cannot pick ``[3]`` to attain a maximum score,
this is because ``KMEANS B`` is connected to ``SCORE B`` already, we need to find another.

The only other connection available is ``A <-> A``::

    .   KMEANS      SCORE    COUNT
    [4] LABEL B <-> LABEL B  4000  [Accept]
    [3] LABEL B <-> LABEL A  2000  [Visited KMEANS Label]
    [1] LABEL A <-> LABEL A  1000  [Accept]
    [2] LABEL A <-> LABEL B  500   [Loop Ended]

Hence, the follow **pseudo-code** is used::

    for kmeans_label, score_label, count in array:
        if kmeans_label or score_label in visited:
            continue
        else:
            visited.append(kmeans_label)
            visited.append(score_label)
            counts.append(count)

----------
Score File
----------

A Score File is any image with a deliberate discrete amount of gray-scale values that mark clusters.

Any area which has the same gray-scale value is deemed to be one cluster.

- Note that **anti-aliasing** can cause multiple unnecessary grayscale interpolated values.
- Note to save it in a **lossless** format like ``png`` to avoid artifacts.

=======
Example
=======

.. code-block:: python

    from sklearn.cluster import KMeans
    from sklearn.preprocessing import minmax_scale

    from frmodel.base import CONSTS
    from frmodel.base.D2 import Frame2D
    from frmodel.base.D2.kmeans2D import KMeans2D
    from tests.base.D2.test_d2 import TestD2

        f = Frame2D.from_image("path/to/file.png")

        C = f.CHN
        frame_xy = f.get_chns(self_=False,
                              chns=[C.MEX_G, C.EX_GR, C.NDI])

        km = KMeans2D(frame_xy,
                      KMeans(n_clusters=3, verbose=False),
                      fit_to=[C.MEX_G, C.EX_GR, C.NDI],
                      scaler=minmax_scale)

        kmf = km.as_frame()
        score = kmf.score(f)

        self.assertAlmostEqual(score['Custom'], 1)
        self.assertAlmostEqual(score['Homogeneity'], 1)
        self.assertAlmostEqual(score['Completeness'], 1)
        self.assertAlmostEqual(score['V Measure'], 1)

- Here, we grab MEX_G, EX_GR, NDI to use for KMeans
- Then fit using them in ``fit_to``. By default if we don't specify these channels, all will be used anyways.
- We also pre-scale them with ``minmax-scale``. Note it's passed as a function without brackets
- We can convert the clustering as a frame to view its clusters or ``score`` it against something else.
- Note that because we scored it against itself, it should, ideally, converge to a perfect score.

``Custom`` is the custom scoring algorithm mentioned above.

===========
Module Info
===========

.. automodule:: frmodel.base.D2.kmeans2D
