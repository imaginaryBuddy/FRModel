#####
Cloud
#####

.. toctree::
   :maxdepth: 2

    Frame Conversion <cloud_to_frame>

=========
Intuition
=========

Cloud3D is **just** a additive wrapper for a laspy File object.

=======
Example
=======

We can load a cloud from a LAS like so.
We can then index the np.ndarray using ``.data``. Cloud3D will query the laspy object for its XYZRGB channels to form
a tabular np.ndarray.

*Note that most classes have a data property to grab the underlying data representation.*

.. code-block:: python

    cloud = Cloud3D.from_las("../rsc/las/chestnut_0/cloud.las")
    d = cloud.data

d is represented as a simple X, Y, Z, R, G, B tabular format.

+----+----+----+----+----+----+
| x0 | y0 | z0 | r0 | g0 | b0 |
+----+----+----+----+----+----+
| x1 | y1 | z1 | r1 | g1 | b1 |
+----+----+----+----+----+----+
| x2 | y2 | z2 | r2 | g2 | b2 |
+----+----+----+----+----+----+

and so on...

===========
Module Info
===========

.. automodule:: frmodel.base.D3.cloud3D
   :inherited-members:
