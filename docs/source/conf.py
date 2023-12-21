# Configuration file for the Sphinx documentation builder.

# -- Project information

import appletree as apt

project = "Appletree"
copyright = "2023, Appletree contributors and the XENON collaboration"
author = "Zihao Xu, Dacheng Xu"

release = apt.__version__
version = apt.__version__

# -- General configuration

extensions = [
    "nbsphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
}
intersphinx_disabled_domains = ["std"]

templates_path = ["_templates"]

# -- Options for NAPOLEON output

napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = True
napoleon_include_special_with_doc = True

# -- Options for HTML output

html_theme = "sphinx_rtd_theme"

# -- Options for EPUB output
epub_show_urls = "footnote"

# -- Options for TODO output

todo_include_todos = True

# These folders are copied to the documentation's HTML output
html_static_path = ["_static"]

# These paths are either relative to html_static_path
# or fully qualified paths (eg. https://...)
html_css_files = [
    "css/custom.css",
]


def setup(app):
    # app.add_css_file('css/custom.css')
    # Hack to import something from this dir. Apparently we're in a weird
    # situation where you get a __name__  is not in globals KeyError
    # if you just try to do a relative import...
    import os
    import sys

    sys.path.append(os.path.dirname(os.path.realpath(__file__)))
    from build_release_notes import convert_release_notes

    convert_release_notes()
