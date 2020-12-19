#####
Frame
#####

.. toctree::
    :maxdepth: 1

    Partition <frame_partition>
    Channel <frame_channel>
    GLCM <frame_channel_glcm>
    Scaling <frame_scaling>
    Plotting <frame_plot>

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

===========
Module Info
===========

.. automodule:: frmodel.base.D2.frame2D
   :inherited-members:
