#########################
Cloud To Frame Conversion
#########################

==========
Motivation
==========

We would like to have a "Height channel" in our ``Frame2D``. This can likely
improve our meaningless segmentation/supervised learning. This gives a robust
Canopy Height Estimation on the 2D plane.

Note that this is still an estimation as we have to interpolate points.

-------------------
Matching Dimensions
-------------------

For our 3D data, we use `.las`, 2D using a geo-referenced `.tiff`.

These 2 formats contains enough information for us to detect the correct
latitude longitude position on earth.

Using `osgeo` to decode `.tiff`, we can get the lat-long ranges, this gives us
a rectangular bound in which we would want the height data. The area mapped for
2D and 3D are different. Hence it's important to make this distinction.

Using `laspy` to decode `.las`, we can get the UTM data, I'll refrain from
describing it here since I'm not well versed in geo-referencing, it's a
miracle I made it work anyways.

By converting UTM to lat-long using the package `utm`, I can thus fit it onto
the 2D data.

-------------
Interpolation
-------------

Note that since Cloud3D X and Y doesn't always fall on integer X, Y values,
it's not possible to directly map Cloud3D onto Frame2D, thus we estimate the
values at every integer through interpolation.

In this project, we do receive the spectral bands from a 2D image, hence there
is no need for interpolation of the RGB bands in the `.las`.

The interpolation used is the `scipy.interpolate.CloughTocher2DInterpolator`.

Usually the cloud points is in the millions, and interpolation may be slow
with large amount of samples; it may be inaccurate with low amount of it.
A balance must be struck to find the appropriate value for interpolation.

=======
Example
=======

Here we demonstrate a simple Cloud to Frame conversion with the Cubic Interpolation.

.. code-block:: python

    c = Cloud3D.from_las("path/to/las.las")
    f = c.to_frame(geotiff_path="path/to/geotiff.tif",
                   shape=(a, b),
                   samples=samples)

This grabs the `.las` and maps it to the `.tif` lat long provided.

The shape defines the resolution and the samples defines how many random points
to take for interpolation.

===========
Module Info
===========

.. automodule:: frmodel.base.D3.cloud._cloud_frame
   :inherited-members:
