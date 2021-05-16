#####################################
Frame Gray Level Co-occurrence Matrix
#####################################

*The GLCM2D Class is deprecated from 0.0.3 onwards, replaced by the current in-built methods*
*Selective GLCM deprecated from 0.0.6 onwards, the GLCM arguments are replaced with channel.*

==========
GLCM Class
==========

*New in 0.0.5*

Due to the ever expanding number of **vegetation indexes**, **0.0.5** introduces a new methodology, detailed here
`Pull Request #67 <https://github.com/Eve-ning/FRModel/pull/67>`_.

Retrieved using ``Frame2D.GLCM``.

The purpose of getting this class is to pass as an argument for GLCM generation.

*From 0.0.6 onwards*

For every recieved ``channel``, all 5 features are now all generated regardless.

=======
Binning
=======

*New in 0.0.6*

Using the argument ``bins`` in the GLCM Class, data is binned before being processed.
This makes GLCM extremely fast.

Recommended to use 4, 8, 16.

=======
Example
=======

``<0.0.5`` code
.. code-block:: python

    out = f.get_chns(glcm_con=True, glcm_cor=True, glcm_ent=True,
                     glcm_by_x=1, glcm_by_y=1, glcm_radius=25, glcm_verbose=True,
                     glcm_entropy_mp=True, glcm_entropy_mp_procs=2)

``>=0.0.5`` code
.. code-block:: python

    glcm = f.GLCM(by=1, radius=25, verbose=True,
                  contrast=[f.CHN.RGB],
                  correlation=[f.CHN.RGB],
                  entropy=[f.CHN.RGB])

    frame = f.get_chns(glcm=glcm)

``>=0.0.6`` code
.. code-block:: python

    glcm = f.GLCM(by=1, radius=2, verbose=True, bins=8
                  channel=[f.CHN.RGB])

    frame = f.get_chns(glcm=glcm)

Assuming f is a ``Frame2D``.

- This grabs the GLCM features.
- The GLCM is offset by 1 x 1.
- The Neighbour Convolution radius is 25.
- The function will output its progress with a progress bar.
- GLCM Entropy will use multiprocessing to speed up the entropy loop
- It will use 2 processes to loop.

Note that for ``0.0.5`` GLCM is not strictly for RGB, however, entropy must be a combination of RGB.

=========
Algorithm
=========

The algorithm information has been redacted to avoid versioning issues with the research journal. An explanation of how
it works can be found in the research journal.

It references this tutorial `GLCM Texture: A Tutorial v. 3.0 March 2017 <https://prism.ucalgary.ca/handle/1880/51900>`_

===========
Module Info
===========

.. automodule:: frmodel.base.D2.frame._frame_channel_glcm