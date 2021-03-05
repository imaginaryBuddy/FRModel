#####################################
Frame Gray Level Co-occurrence Matrix
#####################################

*The GLCM2D Class is deprecated from 0.0.3 onwards, replaced by the current in-built methods*

==========
GLCM Class
==========

*New in 0.0.5*

Due to the ever expanding number of **vegetation indexes**, **0.0.5** introduces a new methodology, detailed here
`Pull Request #67 <https://github.com/Eve-ning/FRModel/pull/67>`_.

Retrieved using ``Frame2D.GLCM``.

The purpose of getting this class is to pass as an argument for GLCM generation.

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

    glcm = f.GLCM(by_x=1, by_y=1, radius=25, verbose=True,
                  contrast=[f.CHN.RGB],
                  correlation=[f.CHN.RGB],
                  entropy=[f.CHN.RGB])

    frame = f.get_chns(glcm=glcm)

Assuming f is a ``Frame2D``.

- This grabs the GLCM Contrast, Correlation, Entropy.
- The GLCM is offset by 1 x 1.
- The Neighbour Convolution radius is 25.
- The function will output its progress with a progress bar.
- GLCM Entropy will use multiprocessing to speed up the entropy loop
- It will use 2 processes to loop.

``frame`` will have the channel dimension length of 9, as contrast, correlation, entropy all act on RGB.

Note that for ``0.0.5`` GLCM is not strictly for RGB, however, entropy must be a combination of RGB.

=========
Algorithm
=========

The algorithm information has been redacted to avoid versioning issues with the research journal. An explanation of how
it works can be found in the research journal.

.. math::

    GLCM_{Contrast} = (i - j)^2 * P(i,j)\\
    GLCM_{Correlation} = \frac{\Delta(i)\Delta(j)}{\sigma(i)\sigma(j)} * P(i,j)\\
    GLCM_{Entropy} = \frac{GLCM(i,j)^2}{Size}

===========
Module Info
===========

.. automodule:: frmodel.base.D2.frame._frame_channel_glcm