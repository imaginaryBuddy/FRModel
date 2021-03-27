#########################
Cloud To Frame Conversion
#########################

==========
Motivation
==========

-------------
Interpolation
-------------

Note that since Cloud3D X and Y doesn't always fall on integer X, Y values,
it's not possible to directly map Cloud3D onto Frame2D by removing the Z Axis.

Instead, we need to estimate the values at every integer through interpolation.

Note that R, G, B are also interpolated, hence values may be invalid on certain
interpolations.

-------
Scaling
-------

It is also important to determine the size of the Frame2D, note that Cloud3D
usually uses units differently from Frame2D. As an example, a pixel in Frame2D
can represent 300 units in Cloud3D.

--------
Sampling
--------

Interpolation may be slow with large amount of samples, and it may be inaccurate
with low amount of it. A balance must be struck to find the appropriate value
for interpolation.

=======
Example
=======

Here we demonstrate a simple Cloud to Frame conversion with the Cubic Interpolation.

.. code-block:: python

    c = Cloud3D.from_las("path/to/file.las")
    f = c.to_frame(10000, height=1000, method=CONSTS.INTERP3D.CUBIC)

We limit the points to sample to 10,000 and the height to 1000.

Note that the return Frame2D has R, G, B, Z as the channels. Where Z is the height.

==============
Interpolations
==============

Interpolation is done using ``scipy.interpolate.griddata``.

Currently, it only supports Nearest, Linear and Cubic. To access these interpolation for the ``method`` argument, use
``CONSTS.INTERP3D``.

--------------
Cubic Clamping
--------------

Cubic interpolation is prone to producing erroneous outliers due to overfitting. There is a default formula to clamp
all RGB results to [0, 255]

.. math::

    \frac{a}{ 1 + e^{[ - ( x - \alpha / 2 ) / \beta ]} }\\
    \alpha = \text{Upper Bound}\\
    \beta = \text{Curve Bend}

Alpha is 255, Beta is 50 by default.

===========
Module Info
===========

.. automodule:: frmodel.base.D3.cloud._cloud_frame
   :inherited-members:
