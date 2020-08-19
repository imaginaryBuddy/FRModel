#######
Channel
#######

=======
Example
=======

In this example, we load a frame from an image, then get the red channel.
For every channel, we can save it as an image.

.. code-block:: python

    frame = Frame2D.from_image("path/to/file.jpg")
    channel_red = frame.channel(CONSTS.CHANNEL.RED)
    channel_red.save("path/to/file_red.jpg")

***********
Module Info
***********

.. automodule:: frmodel.base.D2.channel2D