#####
Frame
#####

=======
Example
=======

We can load a frame from an image like so.
We can then grab the np.ndarray using ``.data``.

*Note that all classes have a data property to grab the underlying data representation.*

.. code-block:: python

    frame = Frame2D.from_image("path/to/file.jpg")
    d = frame.data

***********
Module Info
***********

.. automodule:: FRModel.base.D2.frame2D