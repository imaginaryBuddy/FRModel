#####
Frame
#####

=========
Intuition
=========

For an intuition on how this works, look at :doc:`2 Dimensional Classes <D2>` documentation.

=======
Example
=======

We can load a frame from an image like so.
We can then grab the np.ndarray using ``.data``.

*Note that most classes have a data property to grab the underlying data representation.*

.. code-block:: python

    frame = Frame2D.from_image("path/to/file.jpg")
    d = frame.data

=========
Splitting
=========

``Frame2D`` provides a quick way to create windows

.. code-block:: python

    frame.split(
        by = 50,
        axis = CONSTS.AXIS.X,
        method = Frame2D.SplitMethod.DROP
    )

This splits the current frame into 50px-wide frames. If you specify ``CONSTS.AXIS.Y`` it'll split into 50px-tall ones.

``split_xy`` is a shortcut for splitting on both axes with the same ``by`` and ``method`` arguments.

-------
Methods
-------

In order to deal with irregular splitting on the edges (e.g. 1920 can't be split perfectly into 50s),
we can specify to either ``DROP`` or ``CROP``.

- **DROP** just omits that split from the result, creating a ``list`` of consistently sized **Frames**.
- **CROP** crops out the edge, preserving all pixels but with irregular edge ``list`` elements.

========
Channels
========

In each frame, there will be **channel layers**.

.. code-block:: python

    from frmodel.base.consts import CONSTS
    red_channel = frame.channel(CONSTS.CHANNEL.RED)
    green_channel = frame.channel(CONSTS.CHANNEL.GREEN)
    blue_channel = frame.channel(CONSTS.CHANNEL.BLUE)

You can grab the channels like so. Each of these will create a separate :doc:`Channel2D <channel2D>` class instance.