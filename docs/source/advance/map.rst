:orphan:

.. _head:

Register maps and constants to plugins
======================================

Sometimes, a plugin could depend on some constants and maps,
for example the energy range in an energy sampler, or a curve that gives
the energy spectrum. In appletree, we recommend to use `appletree.takes_config`
to systematically manage them.

.. autofunction:: appletree.takes_config

`appletree.takes_config` takes `Config` as arguments, which will be discussed short later,
and returns a decorator for the plugin. For example,

.. code-block:: python

    @takes_config(
        config0,
        config1,
        ...
    )
    class TestPlugin(appletree.plugin):
        ...

Currently, appletree supports two kinds of configs, `appletree.Constant` and `appletree.Map`.
Both inherit from `appletree.Config`

.. autoclass:: appletree.Map
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: appletree.Constant
    :members:
    :undoc-members:
    :show-inheritance:

Here is an example of using `Constant`

.. code-block:: python

    @takes_config(
        Constant(
            name='a',
            type=float,
            default=137.,
            help='A meaningless scaler.',
        ),
    )
    class Scale(Plugin):
        depends_on = ['x']
        provides = ['y']

        @partial(jit, static_argnums=(0, ))
        def simulate(self, key, parameters, x):
            y = self.a.value * x
            return key, y

and an example of using `Map`

.. code-block:: python

    @takes_config(
        Map(
            name='b',
            default='test_file.json',
            help='A meaningless shift.',
        ),
    )
    class Shift(Plugin):
        depends_on = ['y']
        provides = ['z']

        @partial(jit, static_argnums=(0, ))
        def simulate(self, key, parameters, y):
            shift = appletree.interpolation.curve_interpolator(
                y,
                self.b.coordinate_system,
                self.s2_bias.map,
            )
            z = y + shift
            return key, z

As mentioned in :ref:`instruct <head>`,
the instruct file for `Context` can overwrite the default value of plugins' config. For example,

.. code-block:: python

    {
        "configs": {
            "a": 137.036,
            "b": "alt_test_file.json",
        },
        ...
    }

which changes the value of `a` in `Scale` plugin to 137.036 and json file of `b` in `Shift`
Plugin to "alt_test_file.json".
