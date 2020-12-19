###############
Getting Started
###############

Most classes rely on np.ndarray representation, so most operations are similar and familiar.

If you require the underlying np.ndarray, you just need to call ``f.data`` to grab it.

=======
Example
=======

This short example illustrates how to load in an image (stored as a np.ndarray)

.. code-block:: python

    from frmodel.base.consts import CONSTS
    from frmodel.base.D2.frame2D import Frame2D

    frame = Frame2D.from_image("path/to/file.jpg")
    frame = frame.get_chns(self_=True, chns=[frame.CHN.HSV])
    data: np.ndarray = frame.data

This simple example illustrates how to load a Frame2D and grab the HSV channel, then convert to np.ndarray.

========
Wrapping
========

Most classes here are just wrappers of existing library classes.
If you would like to extract the underlying data, e.g. ``np.ndarray``, just call ``.data``.

This is useful if you would like to utilize functions that the library doesn't provide.
However, it's recommended to use the library functions as a lot of boilercode is hidden.

Open up a **Issue** on GitHub to notify me to create a wrapper for a specific repeated function.

