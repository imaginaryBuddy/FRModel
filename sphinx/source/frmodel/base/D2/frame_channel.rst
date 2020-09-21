#############
Frame Channel
#############

*Channel2D Class is deprecated from 0.0.3 onwards, replaced by the current in-built methods*

================
Getting Channels
================

As to cater to this research project, index calculations are readily available with ``get_xx`` named functions.

Currently, ``frmodel`` supports the following:

``index_count`` represents how many "layers" it will take, similar to the cake analogy.

+---------------------+---------------------------------+-------------+
| function            | description                     | index_count |
+=====================+=================================+=============+
| ``get_self``        | Gets channels including current | n           |
+---------------------+---------------------------------+-------------+
| ``get_xy``          | X, Y Coordinates                | 2           |
+---------------------+---------------------------------+-------------+
| ``get_hsv``         | Hue Saturation and Value        | 3           |
+---------------------+---------------------------------+-------------+
| ``get_ex_g(False)`` | Excess Green                    | 1           |
+---------------------+---------------------------------+-------------+
| ``get_ex_g(True)``  | (Modified) Excess Green         | 1           |
+---------------------+---------------------------------+-------------+
| ``get_ex_gr``       | Excess Green Minus Red          | 1           |
+---------------------+---------------------------------+-------------+
| ``get_ndi``         | Normalized Difference Index     | 1           |
+---------------------+---------------------------------+-------------+
| ``get_veg``         | Vegetative Index                | 1           |
+---------------------+---------------------------------+-------------+
| ``get_glcm``        | GLCM (All Statistics)           | 9           |
+---------------------+---------------------------------+-------------+

Note that ``get_all_chn`` and ``get_chn`` gets all channels above, the order is as shown above too.

The difference between those 2 is the default values,
depending on your choices, one may be more succinct than the other.

====
GLCM
====

GLCM is detailed in the :doc:`GLCM Implementation Page <frame_channel_glcm>`.

==
XY
==

The XY coordinate for every Pixel

===
HSV
===

HSV (Hue, Saturation, Value)

====
EX_G
====

Excess Green, defined as

.. math::

    2G - 1R - 1B

=====
MEX_G
=====

Modified Excess Green, defined as

.. math::

    1.262G - 0.884R - 0.331B

=====
EX_GR
=====

Excess Green Minus Red, defined as

.. math::

    3G - 2.4R - B

===
NDI
===

Normalized Difference Index, defined as

.. math::

    \frac{G - R}{G + R}

===
VEG
===

Vegetative Index, defined as

.. math::

    \frac{g}{r^{a} * b^{1-a}}