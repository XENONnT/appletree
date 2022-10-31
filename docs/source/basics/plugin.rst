:orphan:

Plugin
===============

Plugin is the smallest simulation unit in appletree. All plugins must inherit from the `appletree.Plugin`. 

.. autoclass:: appletree.Plugin
    :members:
    :undoc-members:

There are many default plugins under `appletree.plugins`.

.. toctree::
    :maxdepth: 1

    appletree.plugins.common
    appletree.plugins.er_microphysics
    appletree.plugins.nr_microphysics
    appletree.plugins.detector
    appletree.plugins.reconstruction
    appletree.plugins.efficiency

Here is an example how a plugin works:

.. code-block:: python
    
    import appletree as apt

    # generate a key for pseudorandom generator
    key = apt.randgen.get_key()

    energy_sampler = apt.plugins.common.UniformEnergySpectra()
    key, energy = energy_sampler(
        key,       # key is always the first argument
        {},        # this plugin does not need any parameter so we can send an empty dict
        int(1e6),  # this is the batch_size, the only element in self.depends_on
    )
