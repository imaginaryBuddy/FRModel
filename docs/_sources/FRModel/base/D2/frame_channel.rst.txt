#############
Frame Channel
#############

*Channel2D Class is deprecated from 0.0.3 onwards, replaced by the current in-built methods*

================
Getting Channels
================

**For Versions v0.0.5 and beyond**

Due to the ever expanding number of **vegetation indexes**, **0.0.5** introduces a new methodology, detailed here
`Pull Request #67 <https://github.com/Eve-ning/FRModel/pull/67>`_.

As to cater to this research project, index calculations are readily available through ``get_chns`` or ``get_all_chns``.

``index_count`` represents how many "layers" it has, similar to the cake analogy.

While the prior functions of ``get_...`` is still available, it's recommended to use ``get_chns`` or
``get_all_chns`` because it rids of boilerplate code. However, it's not strictly forbidden.

The new format, uses the following calling template

.. code-block:: python

    f: Frame2D
    f.get_chns(self_:bool, chns:List[CHN], glcm: GLCM)

For example

.. code-block:: python

    f: Frame2D
    f.get_chns(self_=True, chns=[f.CHN.HSV, f.CHN.EX_G, f.CHN.VEG])

The ``CHN`` argument can be grabbed from ``Frame2D`` class or any instance, or ``from consts import CONSTS``, though
the latter method is not recommended as it's verbose.

These ``CHN`` are actually strings, hence it's perfectly valid to call as

.. code-block:: python

    f: Frame2D
    f.get_chns(self_=True, chns=['H', 'S']) # Other arguments omitted

However, it's not recommended as it's prone to capitalization errors and typos.

+--------------+------------------------------------------------------+--------+
| CHN Constant | Description                                          | Layers |
+==============+======================================================+========+
| RED          | Red Channel                                          | 1      |
+--------------+------------------------------------------------------+--------+
| GREEN        | Green Channel                                        | 1      |
+--------------+------------------------------------------------------+--------+
| BLUE         | Blue Channel                                         | 1      |
+--------------+------------------------------------------------------+--------+
| RGB          | A Tuple of all Red, Green, Blue Channels             | 3      |
+--------------+------------------------------------------------------+--------+
| X            | X Position                                           | 1      |
+--------------+------------------------------------------------------+--------+
| Y            | Y Position                                           | 1      |
+--------------+------------------------------------------------------+--------+
| XY           | A Tuple of X, Y Channels                             | 2      |
+--------------+------------------------------------------------------+--------+
| HUE          | Hue Channel                                          | 1      |
+--------------+------------------------------------------------------+--------+
| SATURATION   | Saturation Channel                                   | 1      |
+--------------+------------------------------------------------------+--------+
| VALUE        | Value Channel                                        | 1      |
+--------------+------------------------------------------------------+--------+
| HSV          | A Tuple of Hue, Saturation, Value Channels           | 3      |
+--------------+------------------------------------------------------+--------+
| NDI          | Normalized Difference Index Channel                  | 1      |
+--------------+------------------------------------------------------+--------+
| EX_G         | Excess Green Channel                                 | 1      |
+--------------+------------------------------------------------------+--------+
| MEX_G        | Modified Excess Green Channel                        | 1      |
+--------------+------------------------------------------------------+--------+
| EX_GR        | Excess Green Minus Red Channel                       | 1      |
+--------------+------------------------------------------------------+--------+
| VEG          | Vegetation Channel                                   | 1      |
+--------------+------------------------------------------------------+--------+
| NDVI         | Normalized Difference Vegetation Index Channel       | 1      |
+--------------+------------------------------------------------------+--------+
| GNDVI        | Green Normalized Difference Vegetation Index Channel | 1      |
+--------------+------------------------------------------------------+--------+
| OSAVI        | Optimized Soil Adjusted Vegetation Index Channel     | 1      |
+--------------+------------------------------------------------------+--------+
| NDRE         | Normalized Difference Red Edge Channel               | 1      |
+--------------+------------------------------------------------------+--------+
| LCI          | Leaf Chrolophyll Index Channel                       | 1      |
+--------------+------------------------------------------------------+--------+

Note that ``get_all_chns`` and ``get_chns`` gets all channels above.

The difference between those 2 is that, with ``get_all_chns`` you exclude channels you don't need,
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

===========
Module Info
===========

.. automodule:: frmodel.base.D2.frame._frame_channel