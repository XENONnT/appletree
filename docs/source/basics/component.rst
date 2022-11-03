:orphan:

.. _head:

Component
================

Component is an ensemble of plugins. In general, users can register `Plugin`s to `Component`,
and tell it the varibales you wanna get. Component will auto-deduce the workflow based
on the registered plugins' `depends_on` and `provides`.

In addition, if some component is fixed, i.e. the distribution of variables is independent on
the parameters, component will not do the deduction and simulation.

The base class is `Component`.

.. autoclass:: appletree.Component
    :members:
    :undoc-members:
    :show-inheritance:

For the component that needs simulation, we have a child class `ComponentSim`.

.. autoclass:: appletree.ComponentSim
    :members:
    :undoc-members:
    :show-inheritance:

For the component that is fixed, `ComponentFixed` should be used instead.

.. autoclass:: appletree.ComponentFixed
    :members:
    :undoc-members:
    :show-inheritance:

There are some default plugins with registrations in `appletree.components`. Check them
as a quick start on how to build your own component. See also

.. toctree::
    :maxdepth: 1

    ../notebooks/component.ipynb
