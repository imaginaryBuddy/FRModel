#################
NumPy Integration
#################

``Frame2D`` uses a NumPy array to store data. The array can be accessed with ``.data``.

===========
Get Slicing
===========

For ``>=0.0.6``

Note that this is only for **get**, that is, you can't set frames via this interface yet.

This slicing uses the same method as NumPy, for example, ``ar[0,1]``, ``ar[0:40,20:30]``.

.. code-block:: python

    frame = Frame2D.from_image("path/to/file.jpg")
    sliced_frame = frame[0:400, 200:300]

However, note that to get the channel dimension, you have to specify with its appropriate enums/strings.

.. code-block:: python

    frame = Frame2D.from_image("path/to/file.jpg")
    sliced_frame = frame[0:400, 200:300, frame.CHN.RED]

You can also specify multiple channels, just no duplicates.

.. code-block:: python

    frame = Frame2D.from_image("path/to/file.jpg")
    sliced_frame = frame[0:400, 200:300, [frame.CHN.RED, frame.CHN.GREEN]]

--------
Ellipsis
--------

Similar to NumPy, you can specify an Ellipsis to auto fill XY slicing.

.. code-block:: python

    frame = Frame2D.from_image("path/to/file.jpg")
    sliced_frame_1 = frame[:, :, frame.CHN.RED]
    sliced_frame_2 = frame[..., frame.CHN.RED]

``sliced_frame_1`` and ``sliced_frame_2`` are equivalent.

---------
Step Size
---------

You can also specify a Step Size

.. code-block:: python

    frame = Frame2D.from_image("path/to/file.jpg")
    sliced_frame = frame[::2, ::2, frame.CHN.RED]

This will then half the size of each dimension.

-------
Slicing
-------

If you are using a ``slice`` object, you can supply it directly

.. code-block:: python

    frame = Frame2D.from_image("path/to/file.jpg")
    s1 = slice(10, 20, None)
    s2 = slice(15, 25, None)
    sliced_frame = frame[s1, s2, frame.CHN.RED]
