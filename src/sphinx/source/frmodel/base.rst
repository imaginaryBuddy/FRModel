############
Base Classes
############

.. toctree::
   :maxdepth: 2

    2D Classes <base/D2/D2>

=======
Purpose
=======

Usually Python packages provides a generic interface for easy user interaction.

However, we can make it better by making it specifically for a project to use.

``from PIL import Image``

This provides a generic package interface for us to use, in order to just extract the Red Channel, we likely need
to go through multiple boilerplate code to extract it.

.. code-block:: python

    img = Image.load("img.png")
    ... <boilerplate code to convert to red channel>
    img_red

If we were to wrap it in a class like

``from frmodel.base import Frame2D``

We create an interface that's just.::

.. code-block:: python

    img = Frame2D.load("img.png")
    img_red = img.data_chn(img.CHN.RED)

We cover a lot of these ugly code behind the interface.

