##################
Frame Partitioning
##################

``Frame2D`` provides a quick way to create windows

=========
Splitting
=========

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
