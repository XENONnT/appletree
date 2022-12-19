from jax import jit
from functools import partial

from jax import numpy as jnp

import appletree
from appletree import randgen
from appletree import interpolation
from appletree.config import Map
from appletree.plugin import Plugin
from appletree.utils import exporter

export, __all__ = exporter(export_self=False)


@export
@appletree.takes_config(
    Map(name='posrec_reso',
        default='posrec_reso.json',
        help='Position reconstruction resolution'),
)
class PositionRecon(Plugin):
    depends_on = ['x', 'y', 'z', 'num_electron_drifted']
    provides = ['rec_x', 'rec_y', 'rec_z', 'rec_r']

    @partial(jit, static_argnums=(0, ))
    def simulate(self, key, parameters, x, y, z, num_electron_drifted):
        std = interpolation.curve_interpolator(
            num_electron_drifted,
            self.posrec_reso.coordinate_system,
            self.posrec_reso.map,
        )
        std /= jnp.sqrt(2)
        mean = jnp.zeros_like(num_electron_drifted)
        key, delta_x = randgen.normal(key, mean, std)
        key, delta_y = randgen.normal(key, mean, std)
        rec_x = x + delta_x
        rec_y = y + delta_y
        rec_z = z
        rec_r = jnp.sqrt(x_rec**2 + y_rec**2)
        return key, rec_x, rec_y, rec_z, rec_r


@export
@appletree.takes_config(
    Map(name='s1_bias',
        default='s1_bias.json',
        help='S1 reconstruction bias'),
    Map(name='s1_smear',
        default='s1_smearing.json',
        help='S1 reconstruction smearing'),
)
class S1(Plugin):
    depends_on = ['num_s1_phd', 'num_s1_pe']
    provides = ['s1']

    @partial(jit, static_argnums=(0, ))
    def simulate(self, key, parameters, num_s1_phd, num_s1_pe):
        mean = interpolation.curve_interpolator(num_s1_phd,
                                                self.s1_bias.coordinate_system,
                                                self.s1_bias.map)
        std = interpolation.curve_interpolator(num_s1_phd,
                                               self.s1_smear.coordinate_system,
                                               self.s1_smear.map)
        key, bias = randgen.normal(key, mean, std)
        s1 = num_s1_pe * (1. + bias)
        return key, s1


@export
@appletree.takes_config(
    Map(name='s2_bias',
        default='s2_bias.json',
        help='S2 reconstruction bias'),
    Map(name='s2_smear',
        default='s2_smearing.json',
        help='S2 reconstruction smearing'),
)
class S2(Plugin):
    depends_on = ['num_s2_pe']
    provides = ['s2']

    @partial(jit, static_argnums=(0, ))
    def simulate(self, key, parameters, num_s2_pe):
        mean = interpolation.curve_interpolator(num_s2_pe,
                                                self.s2_bias.coordinate_system,
                                                self.s2_bias.map)
        std = interpolation.curve_interpolator(num_s2_pe,
                                               self.s2_smear.coordinate_system,
                                               self.s2_smear.map)
        key, bias = randgen.normal(key, mean, std)
        s2 = num_s2_pe * (1. + bias)
        return key, s2


@export
class cS1(Plugin):
    depends_on = ['s1', 's1_correction']
    provides = ['cs1']

    @partial(jit, static_argnums=(0, ))
    def simulate(self, key, parameters, s1, s1_correction):
        cs1 = s1 / s1_correction
        return key, cs1


@export
class cS2(Plugin):
    depends_on = ['s2', 's2_correction', 'drift_survive_prob']
    provides = ['cs2']

    @partial(jit, static_argnums=(0, ))
    def simulate(self, key, parameters, s2, s2_correction, drift_survive_prob):
        cs2 = s2 / s2_correction / drift_survive_prob
        return key, cs2
