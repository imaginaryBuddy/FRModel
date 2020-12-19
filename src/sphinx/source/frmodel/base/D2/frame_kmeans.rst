############
Frame KMeans
############

As of ``0.0.4``, the API has been loosened to introduce more features.

========
Creation
========

This class can and should be created from ``Frame2D``, otherwise it might not work correctly.

However, if it's absolutely needed,

- ``model`` is a **fitted** ``KMeans`` instance.
- ``data`` is a ``np.ndarray`` where it follows the **Channel Dimension** convention used.
  This is to facilitate ``plot()`` to use the right axes.

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

-------------
Sample Weight
-------------

A numpy array to set the weight for each record. Refer to ``KMeans.fit(sample_weight=)`` argument

------
Scaler
------

Scales the data before running kmeans, must be a **Callable**

===============
Figure Plotting
===============

``plot`` plots using ``seaborn.lmplot()``

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

=====
Score
=====

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

    from frmodel.base.D2 import Frame2D


    f = Frame2D.from_image("path/to/file.png")
    frame_xy = f.get_chns(xy=True, hsv=True, mex_g=True, ex_gr=True, ndi=True)

    km = frame_xy.kmeans(KMeans(n_clusters=3, verbose=False),
                         fit_indexes=[2, 3, 4, 5, 6, 7],
                         scaler=minmax_scale)

    counts, score = km.score("path/to/score_file.png")

===========
Module Info
===========

.. automodule:: frmodel.base.D2.kmeans2D
