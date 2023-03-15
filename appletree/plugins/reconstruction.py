from jax import jit
from functools import partial

from jax import numpy as jnp

import appletree
from appletree import randgen
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
        std = self.posrec_reso.apply(num_electron_drifted)
        std /= jnp.sqrt(2)
        mean = jnp.zeros_like(num_electron_drifted)
        key, delta_x = randgen.normal(key, mean, std)
        key, delta_y = randgen.normal(key, mean, std)
        rec_x = x + delta_x
        rec_y = y + delta_y
        rec_z = z
        rec_r = jnp.sqrt(rec_x**2 + rec_y**2)
        return key, rec_x, rec_y, rec_z, rec_r


@export
@appletree.takes_config(
    Map(name='s1_bias_3f',
        default='s1_bias_3f.json',
        help='3fold S1 reconstruction bias'),
    Map(name='s1_smear_3f',
        default='s1_smearing_3f.json',
        help='3fold S1 reconstruction smearing'),
)
class S1(Plugin):
    depends_on = ['num_s1_phd', 'num_s1_pe']
    provides = ['s1_area']

    @partial(jit, static_argnums=(0, ))
    def simulate(self, key, parameters, num_s1_phd, num_s1_pe):
        mean = self.s1_bias_3f.apply(num_s1_phd)
        std = self.s1_smear_3f.apply(num_s1_phd)
        key, bias = randgen.normal(key, mean, std)
        s1_area = num_s1_pe * (1. + bias)
        return key, s1_area


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
    provides = ['s2_area']

    @partial(jit, static_argnums=(0, ))
    def simulate(self, key, parameters, num_s2_pe):
        mean = self.s2_bias.apply(num_s2_pe)
        std = self.s2_smear.apply(num_s2_pe)
        key, bias = randgen.normal(key, mean, std)
        s2_area = num_s2_pe * (1. + bias)
        return key, s2_area


@export
class cS1(Plugin):
    depends_on = ['s1_area', 's1_correction']
    provides = ['cs1']

    @partial(jit, static_argnums=(0, ))
    def simulate(self, key, parameters, s1_area, s1_correction):
        cs1 = s1_area / s1_correction
        return key, cs1


@export
class cS2(Plugin):
    depends_on = ['s2_area', 's2_correction', 'drift_survive_prob']
    provides = ['cs2']

    @partial(jit, static_argnums=(0, ))
    def simulate(self, key, parameters, s2_area, s2_correction, drift_survive_prob):
        cs2 = s2_area / s2_correction / drift_survive_prob
        return key, cs2
