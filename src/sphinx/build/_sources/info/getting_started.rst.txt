###############
Getting Started
###############

This document is directed to the users of the library.

=======
Example
=======

This short example illustrates how to load in an image (stored as a np.ndarray)

.. code-block:: python

    from frmodel.base.consts import CONSTS
    from frmodel.base.D2.frame2D import Frame2D

    frame = Frame2D.from_image("path/to/file.jpg")
    frames =\
        frame.split(200, axis=CONSTS.AXIS.X, method=Frame2D.SplitMethod.DROP)
    for e, f in enumerate(frames): f.save(f"path/to/save{e}.jpg")

If the image is **Width x Height** large,
We then split it into **200px x Height** chunks using ``split()``

- Note that it slices vertically because we specified ``CONSTS.AXIS.X``.
- It will also ``DROP`` any pieces that cannot be perfectly cut. E.g. 1920 will be sliced into 200 x 9 pieces.

========
Wrapping
========

Most classes here are just wrappers of existing library classes.
If you would like to extract the underlying data, e.g. ``np.ndarray``, just call .data.

This is useful if you would like to utilize functions that the library doesn't provide.
However, it's recommended to use the library functions as a lot of boilercode is hidden.

Open up a **Issue** on GitHub to notify me to create a wrapper for a specific repeated function.

