:orphan:

.. _head:

Parameters in appletree
=======================

In general, all parameters including fixed parameters and fit
parameters are handled by `appletree.Parameter`, which has many useful attributes

.. autoclass:: appletree.Parameter
    :members:
    :undoc-members:
    :show-inheritance:
    :special-members: __init__

We recommend to initialize `Parameter` from a json file. An example of parameter config file is
`the ER parameter file <https://github.com/XENONnT/appletree/blob/master/appletree/parameters/er_sr0.json>`_, in
which the dictionary is structured as

.. code-block:: python

    {
        param_name_0: config_of_param_0,
        param_name_1: config_of_param_1,
        ...
    }

For each `config_of_param`, it should be a dictionary containing the following items:

* **"prior_type"**: can be either

  * "norm": normal prior, `prior_args` are `mean` and `std`.
  * "uniform": uniform prior, `prior_args` are `lower` and `upper`.
  * "free": free prior, no `prior_args` required.
  * "fixed": it won't be considered as a fit parameter, `prior_args` is `val`.

* **"prior_args"**: a dictionary like `{arg_name : arg_value}` which goes into prior.
* **"allowed_range"**: a list like `[lower_boundary, upper_boundary]`, above which parameters will be clipped and have `-np.inf` log prior.
* **"init_mean"**: the gaussian mean as the initial guess of the MCMC walkers. The random initialzation of MCMC will be clipped by "allowed_range".
* **"init_std"**: the gaussian std as the initial guess of the MCMC walkers.
* **"unit"**: the unit of the parameter, only for documentation purpose.
* **"doc"**: the addtional docstring for the parameter.

For example,

.. code-block:: python

    {
        "w": {
            "prior_type": "norm",
            "prior_args": {
                "mean": 0.0137,
                "std": 0.0002
            },
            "allowed_range": [
                0,
                1.0
            ],
            "init_mean": 0.0137,
            "init_std": 0.0002,
            "unit": "keV",
            "doc": "Mean energy to generate a quanta in liquid xenon"
        },
        "fano": {
            "prior_type": "fixed",
            "prior_args": {
                "val": 0.059
            },
            "allowed_range": null,
            "init_mean": null,
            "init_std": null,
            "unit": "1",
            "doc": "Fano factor which describes the fluctuation of num of quanta"
        },
    }

All non-fixed parameters are called fit parameters in appletree,
and will be the parameters that MCMC samples.
`appletree.Plugin` needs a dictionary of parameters. With the `Parameter` class,
it can be simply obtained by `Parameter.get_all_parameter()`.
