#######
Channel
#######

**This is deprecated from 0.0.3 onwards, however, the knowledge is still relevant**

This is simply a *layer* of **Frame2D**.

=======
Example
=======

In this example, we load a frame from an image, then get the red channel.
For every channel, we can save it as an image.

.. code-block:: python

    frame = Frame2D.from_image("path/to/file.jpg")
    channel_red = frame.channel(CONSTS.CHANNEL.RED)
    channel_red.save("path/to/file_red.jpg")

====
GLCM
====

**It's recommended to use Frame2D to calculate glcm to get updated.**

The GLCM2D instance can be grabbed by just calling the ``.glcm()`` method.

More details on :doc:`GLCM <glcm2D>`'s documentation