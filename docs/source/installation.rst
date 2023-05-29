:orphan:

Installation
=====================================================

Option 1: install via pypi
-----------------------------------------------------

Appletree can be found on `pypi <https://pypi.org/project/appletree/>`_ now. To install it via regular pip, simply run

.. code-block:: console

    pip install --upgrade pip

With cpu support:

.. code-block:: console

    pip install appletree[cpu]

With CUDA Toolkit 11.2 support:

.. code-block:: console

    pip install appletree[cuda112] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

With CUDA Toolkit 12.1 support:

.. code-block:: console

    pip install appletree[cuda121] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

Option 2: install from source code
-----------------------------------------------------

.. code-block:: console

    git clone https://github.com/XENONnT/appletree
    cd appletree

To install the package and requirements in your environment, replace `pip install appletree[*]` to `python3 -m pip install .[*] --user` in the above `pip` commands.

To install appletree in editable mode, insert `--editable` argument after `install` in the above `pip install` or `python3 -m pip install` commands.

For example, to install in your environment and in editable mode with CUDA Toolkit 12.1 support:

.. code-block:: console

    python3 -m pip install --editable .[cuda121] --user -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
