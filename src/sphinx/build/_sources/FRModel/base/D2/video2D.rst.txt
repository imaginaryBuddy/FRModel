#####
Video
#####

This class is mainly used to extract images from videos.

The project isn't pivoting off of video data yet, so it lacks functions.

===============
Grabbing Frames
===============

We can grab :doc:`frames <frame2D>` from a video using anything iterable, like ``np.linspace``.

This grabs the 0s, 50s, 100s frames

.. code-block:: python

    from frmodel.base.D2.video2D import Video2D
    import numpy as np
    from frmodel.base.consts import CONSTS
    vid = Video2D.from_video("sample.mp4")

    frames = vid.to_frames(np.linspace(0, 100000, 3))

=======
Caveats
=======

The class doesn't provide a way to calculate the length of the video.

===============
Grabbing Frames
===============

.. automodule:: frmodel.base.D2.video2D