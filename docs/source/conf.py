# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
from datetime import datetime
import sys, os
#sys.path.insert(0, os.path.abspath('../..'))
sys.path.insert(0, os.path.abspath('../..'))


# -- Project information -----------------------------------------------------

project = "Quantus"
copyright = f"{str(datetime.utcnow().year)}, Anna Hedström"
author = 'Anna Hedström'
release = 'v0.2.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['myst_parser',
              "sphinx.ext.autodoc",
              #'myst_nb',
              #'myst_parser',
              #'sphinx.ext.coverage',
              #'sphinx.ext.napoleon',
              #'autoapi.extension'
              "numpydoc"
              ]

# Napolean settings.
#napoleon_numpy_docstring = True

source_suffix = ['.rst', '.md']

templates_path = ['_templates']
exclude_patterns = []

language = "en"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    "logo_only": True,
}
html_logo = "assets/quantus_logo 2.png"
html_static_path = ["", "_static"]

#autoapi_dirs = ['../../quantus']