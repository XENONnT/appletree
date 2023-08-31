:orphan:

Plugin
======

Plugin is the smallest simulation unit in appletree.
All plugins must inherit from the `appletree.Plugin`.

.. autoclass:: appletree.Plugin
    :members:
    :undoc-members:
    :show-inheritance:

There are many default plugins under `appletree.plugins`.

.. toctree::
    :maxdepth: 1

    appletree.plugins.common
    appletree.plugins.detector
    appletree.plugins.efficiency
    appletree.plugins.er_microphysics
    appletree.plugins.lyqy
    appletree.plugins.nestv2
    appletree.plugins.reconstruction

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

Note that whatever `key` or `parameters` will be used in `Plugin.simulate` or not,
they must be the first and second arguments, and `key` is always the first in the returned tuple.
