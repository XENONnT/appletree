# Appletree

A high-Performance Program simuLatEs and fiTs REsponse of xEnon.

[![DOI](https://zenodo.org/badge/534803881.svg)](https://zenodo.org/badge/latestdoi/534803881)
[![Test package](https://github.com/XENONnT/appletree/actions/workflows/pytest.yml/badge.svg?branch=master)](https://github.com/XENONnT/appletree/actions/workflows/pytest.yml)
[![Coverage Status](https://coveralls.io/repos/github/XENONnT/appletree/badge.svg?branch=master&kill_cache=1)](https://coveralls.io/github/XENONnT/appletree?branch=master&kill_cache=1)
[![PyPI version shields.io](https://img.shields.io/pypi/v/appletree.svg)](https://pypi.python.org/pypi/appletree/)
[![Readthedocs Badge](https://readthedocs.org/projects/appletree/badge/?version=latest)](https://appletree.readthedocs.io/en/latest/?badge=latest)
[![CodeFactor](https://www.codefactor.io/repository/github/xenonnt/appletree/badge)](https://www.codefactor.io/repository/github/xenonnt/appletree)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/XENONnT/appletree/master.svg)](https://results.pre-commit.ci/latest/github/XENONnT/appletree/master)

## Installation and Set-Up

### Regular installation:

With cpu support:

```
pip install appletree[cpu]
```

With CUDA Toolkit 11.2 support:

```
pip install appletree[cuda112] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

With CUDA Toolkit 12.1 support:

```
pip install appletree[cuda121] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### Developer setup:

Clone the repository:

```
git clone https://github.com/XENONnT/appletree
cd appletree
```

To install the package and requirements in your environment, replace `pip install appletree[*]` to `python3 -m pip install .[*] --user` in the above `pip` commands.

To install appletree in editable mode, insert `--editable` argument after `install` in the above `pip install` or `python3 -m pip install` commands.

For example, to install in your environment and in editable mode with CUDA Toolkit 12.1 support:

```
python3 -m pip install --editable .[cuda121] --user -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

Then you are now good to go!

## Usage

The best way to start with the `appletree` package is to have a look at the tutorial `notebooks`.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.
