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

ER light-yield morphing
-----------------------

The NESTv2 ER components use ``ERYieldMorpher`` to vary the light yield while
keeping the total quanta yield fixed.  If ``T = 1 / w`` and ``QY_0`` is the
nominal output of ``QyER``, the morpher evaluates the relative light-yield
uncertainty map ``r(E)`` and computes

.. math::

    LY_0 = T - QY_0,
    \qquad LY(t) = \operatorname{clip}\left[LY_0(1 + t_{er\_yield} r(E)), 0, T\right],
    \qquad QY_{morphed} = T - LY(t).

``t_er_yield = +1`` therefore means a +1-sigma light-yield shift; it is not a
100% scale change.  It has a ``norm(0, 1)`` prior with bounds ``[-3, 3]``,
matching the ER sigma nuisances.  The NEST work-function parameter ``w`` is
not shifted by this nuisance.  ``LyER`` and the ER expected-electron consumer use
``QY_morphed``, so the morphed light and charge yields sum to ``1 / w``.
The downstream ER recombination model and its electron bound are otherwise
unchanged.

Here ``r(E) = sigma_LY(E) / LY_nominal(E)`` is an externally derived relative
uncertainty from measurements compared with the nominal NESTv2/Appletree
prediction; it is not a NEST coefficient.

The default ``_er_ly_rel_uncertainty.json`` is an explicit zero point map, not
a measurement.  A replacement map must use ``coordinate_type: "point"`` with
sorted energy coordinates in keV and finite, nonnegative, unitless relative
uncertainties.  The ``LERP`` implementation holds the first/last map value at
energies outside the coordinate range (it does not extrapolate).  The morpher
clips the resulting light yield to ``[0, 1 / w]``; shifts beyond either bound
therefore saturate and remain physically valid.

For example, an instruct can select a measured uncertainty map with:

.. code-block:: json

    {
        "configs": {
            "er_ly_rel_uncertainty": "my_er_ly_rel_uncertainty.json"
        }
    }

At ``t_er_yield = 0``, the output agrees with the old nominal QY wherever the
old QyER result is physical.  QyER historically clips only its lower bound,
so at sufficiently low energy it can exceed ``1 / w``; the conserved morpher
clips that case to ``QY_morphed = 1 / w`` and ``LY = 0`` even at nominal.
The built-in ER components register the morpher automatically.  A custom
component that manually registers a subset of plugins must also register
``ERYieldMorpher`` and retain its dependency chain, and custom parameter
dictionaries must include ``t_er_yield``.
