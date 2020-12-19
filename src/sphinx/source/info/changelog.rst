#########
ChangeLog
#########

-----
0.0.5
-----

- Cythonize Entropy
- Integrate Cython Pipeline into current Package
- Implement Homogeneity, Completeness, V Measure Scoring
- Add Image Scaling
- Add Image-like methods for ``Frame2D``
- Add Test for Scoring
- Add ``express`` for quick running of long scripts for quick replicaiton purposes
    - While this is useful to backtrack on, most scripts get deprecated very quickly due to constant updates
- Add Convolution method for integrating non-GLCM with GLCM channels
- Add Plotting directly for ``Frame2D``
    - 2D Per Channel Plotting
    - 3D Channel Plotting
- Add Scaling API to use lambda calls, e.g. ``minmax_scaling``
- Rename ``from_rgbxy`` to ``from_nxy`` for short verbosity
- Detach ``KMeans`` operation into separate ``KMeans2D`` to decrease bulkiness
- Fix Correlation Algorithm
    - Correlation was using a naive algorithm that was inaccurate to standards
    - Reimplemented Correlation on Cython
- Fix Entropy Algorithm
    - Entropy had ``glcm_view`` that was reset on the wrong line
- Force Scale Correlation and Entropy to standards. [-1, +1] and [0, 1] respectively for interpretability
- Invert Entropy Result to align with defined meaning
- Update Channel getting and setting to use new const string indexing convention
    - See #67 PR for details on updating

-----
0.0.4
-----

- Separated implementation for ``Frame2D``
- Improved performance for GLCM statistics
- Use Gaussian + FFTConvolution for Non-GLCM Channel fitting
- Implement GLCM statistics with FFTConvolution
- Add Multiprocessing capability for GLCM Entropy
- Split the ``Frame2D`` page into separate implementation pages
- Add basic LAS w/ XML Metadata support.
- Fix autodoc issue
- Add ``Draw2D`` for marking
- Deprecate ``Channel2D`` and ``GLCM2D``
- Remove files in ``rsc``
- Implement ``KMeans2D``
- Implement ``score`` -ing system to evaluate ``KMeans2D`` clustering performance.


-----
0.0.3
-----
- Implement Index Grabbing with ``get_xx`` ops.
- Added shorthand for multiple Index Grabbing with ``get_idxs`` and ``get_all_idxs``.
- **Index** is now a term to represent a generated "channel". That is it's calculated from source data.
- Add from ``from_rgbxy_`` function to enable generation from RGBXY+ arrays. RGB is optional but recommended to include.
- Added GLCM calculation within ``Frame2D``
- Fix issue with ``X`` and ``Y`` consts being flipped
- Added simple wrapper for ``sklearn.neighbours.KDTree`` generation from ``Frame2D``.
- Stage ``Channel2D`` and ``GLCM2D`` for deprecation.
- Force rename **index** to **channel** for differentiation.

-----
0.0.2
-----
- Replace structured array with general ``dtyping`` for efficient coding
- Implement indexes with new data structure

-----
0.0.1
-----
**Initial Commit**
- No changes if there's nothing to change!