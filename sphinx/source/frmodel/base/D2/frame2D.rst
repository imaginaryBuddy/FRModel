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

==============
Indexes / Gets
==============

As to cater to this research project, index calculations are readily available with ``get_xx`` named functions.

Currently, ``frmodel`` supports the following:

``index_count`` represents how many "layers" it will take, similar to the cake analogy.

+---------------------+-----------------------------+-------------+
| function            | description                 | index_count |
+=====================+=============================+=============+
| ``get_xy``          | X, Y Coordinates            | 2           |
+---------------------+-----------------------------+-------------+
| ``get_hsv``         | Hue Saturation and Value    | 3           |
+---------------------+-----------------------------+-------------+
| ``get_ex_g(False)`` | Excess Green                | 1           |
+---------------------+-----------------------------+-------------+
| ``get_ex_g(True)``  | (Modified) Excess Green     | 1           |
+---------------------+-----------------------------+-------------+
| ``get_ex_gr``       | Excess Green Minus Red      | 1           |
+---------------------+-----------------------------+-------------+
| ``get_ndi``         | Normalized Difference Index | 1           |
+---------------------+-----------------------------+-------------+
| ``get_veg``         | Vegetative Index            | 1           |
+---------------------+-----------------------------+-------------+
| ``get_chn``         | All Indexes                 | 13          |
+---------------------+-----------------------------+-------------+

Note that ``get_all_chn`` and ``get_chn`` gets all channels above, the order is as shown above too,
i.e. xy will be the first 2 indexes.

====
GLCM
====

GLCM Calculation is similar to the deprecated :doc:`GLCM2D <glcm2D>` class.

This is moved to ``Frame2D`` for efficiency in code.

=========
Normalize
=========

Calling ``normalize`` will normalize everything on the last axis using ``sklearn.preprocessing.normalize``.

Note that normalizing will break ``.save`` unless the data is denormalized manually!

========
Channels
========

**Refer to** :doc:`2 Dimensional Classes <D2>` for information on "Channels".

``Channel2D`` **Deprecated since 0.0.3**

In each frame, there will be **channel layers**.

.. code-block:: python

    from frmodel.base.consts import CONSTS
    red_channel = frame.channel(CONSTS.CHANNEL.RED)
    green_channel = frame.channel(CONSTS.CHANNEL.GREEN)
    blue_channel = frame.channel(CONSTS.CHANNEL.BLUE)

You can grab the channels like so. Each of these will create a separate :doc:`Channel2D <channel2D>` class instance.