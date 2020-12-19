# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# sys.path.insert(0, os.path.abspath('.'))

import os
import sys
import sphinx_rtd_theme
sys.path.insert(0, os.path.abspath('../../'))
print(os.path.abspath('../../'))

# -- Project information -----------------------------------------------------

project = 'frmodel'
copyright = '2020, Eve-ning'
author = 'Eve-ning'

# The full version, including alpha/beta/rc tags
release = '0.0.5'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx_rtd_theme',
              'sphinx.ext.autodoc',
              'sphinx.ext.coverage',
              'sphinx.ext.napoleon']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

autodoc_default_options = {
    'members': True,
    'show-inheritance': True,
    'special-members': '__init__',
    'member-order': 'groupwise',
    'private-members': True
}

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

html_theme_options = {
    'navigation_depth': -1
}

