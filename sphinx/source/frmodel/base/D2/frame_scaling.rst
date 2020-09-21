#############
Frame Scaling
#############

Note that scaling will break ``.save`` unless the data is denormalized manually!

=========
Normalize
=========

Normalize everything on the last axis using ``sklearn.preprocessing.normalize``.

===============
Min Max Scaling
===============

Scales everything on the last axis using ``sklearn.preprocessing.minmax_scale``.