##################
Frame Partitioning
##################

``Frame2D`` provides a quick way to create windows

=======
View As
=======

There are a few methods to split the frames. This is the recommended way as splitting doesn't use memory referencing.

- ``view_windows``
- ``view_windows_as_frames``
- ``view_blocks``
- ``view_blocks_as_frames``

--------------------------------
Windows vs. Blocks vs. As Frames
--------------------------------

Windows can overlap, blocks cannot.

Note that the functions without ``as_frames`` returns a ``np.ndarray``.

=========
Splitting
=========

*This is not the recommended way of partitioning from 0.0.5 onwards*

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

=======
Sliding
=======

*This is not the recommended way of partitioning from 0.0.5 onwards*

``Frame2D``

.. code-block:: python

    frame.slide(
        by = 50,
        stride = 10,
        axis = CONSTS.AXIS.X
    )

Sliding is different from splitting as it allows for overlapping regions.

The ``stride`` argument specifies how far the window to shift on that axis.

``slide_xy`` is a shortcut for splitting on both axes with the same ``by`` and ``stride`` arguments.

===========
Module Info
===========

.. automodule:: frmodel.base.D2.frame._frame_partition
