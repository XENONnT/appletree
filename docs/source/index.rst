Appletree
=========================

Github page: https://github.com/XENONnT/appletree

Appletree stands for "a high-performance program simulates and fits response of xenon".
It's a software framework of fitting response of liquid xenon detector, based on 
`jax <https://jax.readthedocs.io/en/latest/>`_ and affine invariant MCMC ensemble sampler 
using `emcee <https://emcee.readthedocs.io/en/stable/>`_.

.. toctree::
    :maxdepth: 1
    :caption: Installation

    installation.rst

.. toctree::
    :maxdepth: 1
    :caption: Basics

    basics/general.rst
    basics/plugin.rst
    basics/component.rst
    basics/likelihood.rst
    basics/context.rst

.. toctree::
    :maxdepth: 1
    :caption: Advanced techniques

    advance/parameter.rst
    advance/instruct.rst
    advance/map.rst
    advance/file.rst

.. toctree::
    :maxdepth: 1
    :caption: Example notebooks

    notebooks/1%2C%20component.ipynb
    notebooks/2%2C%20likelihood.ipynb
    notebooks/3%2C%20context.ipynb
    notebooks/4%2C%20benchmark.ipynb
    notebooks/5%2C%20datastructure.ipynb
    notebooks/5%2C%20datastructure.ipynb
