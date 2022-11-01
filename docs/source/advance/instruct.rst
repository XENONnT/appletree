:orphan:

.. _head:

Instruct file for context
===============================

In appletree, we use an instruct json file to initialize `Context`. Some examples can be found in
`instructs folder <https://github.com/XENONnT/appletree/tree/master/appletree/instructs>`_. In the json file,
the dictionary must contain three items: **"configs"**, **"par_config"** and **"likelihoods"**.

The value of "configs" is a dictionary that overwrites the default plugins' config, like

.. code-block:: python

    {
        name_of_plugin_config: new_value_of_plugin_config,
        ...
    }

We will discuss more about how appletree deals with them in :ref:`map <head>` section.

The value of "par_config" should be the file name of parameter file described in :ref:`parameter <head>`.

The value of "likelihoods" is a dictionary that gives the definition of likelihood. If the fitting is a single dataset fitting,
the dictionary only has one item; if the fitting is a combined fitting, it could have multiple items. In each item, it must contain

* **"components"**: all components that define the model.
* **"data_file_name"**: data file name, must be .pkl or .csv.
* **"bins_type"**: can be "meshgrid" or "equiprob". For most of the case, we only recommend "equiprob", meaning a equi-probable binning defined by data.
* **"bins_on"**: names of variables that the histogram is applied on.
* **"bins"**: number of bins.
* **"x_clip"**: range of the first dimension.
* **"y_clip"**: range of the second dimension.

For example,

.. code-block:: python

    {
        "components": {
            "rn220_er": "ERBand",
            "rn220_ac": {"component_cls": "AC", "file_name": "AC_Rn220.pkl"}
        },
        "data_file_name": "data_Rn220.csv",
        "bins_type": "equiprob",
        "bins_on": ["cs1", "cs2"],
        "bins": [15, 15],
        "x_clip": [0, 100],
        "y_clip": [2e2, 1e4]
    }

In this likelihood config, the data has two components: ER and AC, where ER component is `ERBand`, and AC component is `AC` with input file "AC_Rn220.pkl".
And the binned Poisson likelihood is defined in the equi-probable binning of cS1-cS2 space, with 15 bins on each dimension. To understand how appletree locates
the files and classes, like "data_Rn220.csv" etc., check :ref:`file <head>` section.
