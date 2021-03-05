#############
Frame Scaling
#############

Note that scaling will break ``.save`` unless the data is denormalized manually!

=========
Normalize
=========

Normalize everything on the last axis using ``sklearn.preprocessing.normalize``.

===============
Min Max Scaling
===============

Scales everything on the last axis using ``sklearn.preprocessing.minmax_scale``.

==============
Custom Scaling
==============

In case you have a custom scaler, you can scale it by passing the scale ``Callable`` as an argument

.. code-block:: python

    from sklearn.preprocessing import robust_scale

    f = Frame2D.from_image("../rsc/imgs/test_kmeans/test_chestnut.png", scale=SCALE)
    fc = f.get_all_chns()
    fcs = fc.scale(robust_scale)

===========
Module Info
===========

.. automodule:: frmodel.base.D2.frame._frame_scaling
