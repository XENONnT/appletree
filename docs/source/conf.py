# Configuration file for the Sphinx documentation builder.

# -- Project information

project = 'Appletree'
copyright = '2022, appletree contributors and the XENON collaboration'
author = 'Zihao Xu, Dacheng Xu'

release = '0.0'
version = '0.0.0'

# -- General configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
    'sphinx.ext.viewcode',
    'nbsphinx',
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'

# -- Options for EPUB output
epub_show_urls = 'footnote'
