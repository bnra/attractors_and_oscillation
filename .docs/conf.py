# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath(".."))


# -- Project information -----------------------------------------------------

project = "Attractors and Oscillation"
copyright = "2022, Benedikt Rank"
author = "Benedikt Rank"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ["autoapi.extension", "sphinx.ext.autodoc", "sphinx.ext.doctest"]
autodoc_typehints = "description"

autoapi_type = "python"
autoapi_dirs = ["..", "."]
autoapi_ignore = ['**/.docs/*', '**/test/*', '**/nb_tests/*', '**/data/*', '**/notes/*', '**/notebooks/*', '**/scripts/*', '*migrations*']

modules_skipped = ['run_speed_test']

#autodoc_default_options = {
#    'members': True,
#    'member-order': 'bysource',
#    'special-members': True,
#    'undoc-members': True,
#    'exclude-members': '__weakref__'
#}

#autoclass_content = 'class'
#autodoc_class_signature = 'separated'

def do_not_skip_special_members(app, what, name, obj, skip, options):
    if what == "method" and "__" in name:
        skip = False
    return skip

# unfortunately it only prevents from showing in the api reference
#  shows in index, module index and search page
def skip_specific_models(app, what, name, obj, skip, options):
    if what == "module" and name in modules_skipped:
        skip=True
    return skip


def setup(sphinx):
    sphinx.connect("autoapi-skip-member", skip_specific_models)
    sphinx.connect("autoapi-skip-member", do_not_skip_special_members)


# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "alabaster"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
