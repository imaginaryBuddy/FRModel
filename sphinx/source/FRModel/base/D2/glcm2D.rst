###############################
Gray Level Co-occurrence Matrix
###############################

=======
Example
=======

1) In this, we firstly split it by 100x100 windows. (Note that uneven slices will be dropped by default)

2) Then loop through all windows.

3) Grab the red channel

4) Create the GLCM

5) Calculate all available statistics

.. code-block:: python

    from FRModel.base.consts import CONSTS
    from FRModel.base.D2.frame2D import Frame2D

    frame = Frame2D.from_image("path/to/file.jpg")

    frames = frame.split_xy(100)

    for xframes in frames:
        for frame in xframes:
            frame_red = frame.channel(CONSTS.CHANNEL.RED)
            glcm = frame_red.glcm(by=1, axis=CONSTS.AXIS.Y)
            glcm.contrast()
            glcm.correlation()
            glcm.entropy()

***********
Module Info
***********

.. automodule:: FRModel.base.D2.glcm2D