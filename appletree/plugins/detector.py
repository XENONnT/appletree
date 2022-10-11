from jax import numpy as jnp
from jax import jit
from functools import partial

import appletree
from appletree import randgen
from appletree import interpolation
from appletree.plugin import Plugin
from appletree.config import Map
from appletree.utils import exporter

export, __all__ = exporter(export_self=False)


@export
@appletree.takes_config(
    Map(name='s1_lce',
        default='s1_correction_map.json',
        help='S1 light collation efficiency correction'),
)
class S1Correction(Plugin):
    depends_on = ['rec_x', 'rec_y', 'rec_z']
    provides = ['s1_correction']

    @partial(jit, static_argnums=(0, ))
    def simulate(self, key, parameters, x, y, z):
        pos = jnp.stack([x, y, z]).T
        s1_correction = interpolation.map_interpolator_regular_binning_3d(
            pos,
            self.s1_lce.coordinate_lowers,
            self.s1_lce.coordinate_uppers,
            self.s1_lce.map,
        )
        return key, s1_correction


@export
@appletree.takes_config(
    Map(name='s2_lce',
        default='s2_correction_map.json',
        help='S2 light collation efficiency correction'),
)
class S2Correction(Plugin):
    depends_on = ['rec_x', 'rec_y']
    provides = ['s2_correction']

    @partial(jit, static_argnums=(0, ))
    def simulate(self, key, parameters, x, y):
        pos = jnp.stack([x, y]).T
        s2_correction = interpolation.map_interpolator_regular_binning_2d(
            pos,
            self.s2_lce.coordinate_lowers,
            self.s2_lce.coordinate_uppers,
            self.s2_lce.map,
        )
        return key, s2_correction


@export
class PhotonDetection(Plugin):
    depends_on = ['num_photon', 's1_correction']
    provides = ['num_s1_phd']
    parameters = ('g1', 'p_dpe')

    @partial(jit, static_argnums=(0, ))
    def simulate(self, key, parameters, num_photon, s1_correction):
        g1_true_no_dpe = jnp.clip(parameters['g1'] * s1_correction / (1. + parameters['p_dpe']), 0, 1.)
        key, num_s1_phd = randgen.binomial(key, g1_true_no_dpe, num_photon)
        return key, num_s1_phd


@export
class S1PE(Plugin):
    depends_on = ['num_s1_phd']
    provides = ['num_s1_pe']
    parameters = ('p_dpe',)

    @partial(jit, static_argnums=(0, ))
    def simulate(self, key, parameters, num_s1_phd):
        key, num_s1_dpe = randgen.binomial(key, parameters['p_dpe'], num_s1_phd)
        num_s1_pe = num_s1_dpe + num_s1_phd
        return key, num_s1_pe


@export
@appletree.takes_config(
    Map(name='elife',
        default='elife.json',
        help='Electron lifetime correction'),
)
class DriftLoss(Plugin):
    depends_on = ['z']
    provides = ['drift_survive_prob']
    parameters = ('drift_velocity',)

    @partial(jit, static_argnums=(0, ))
    def simulate(self, key, parameters, z):
        key, p = randgen.uniform(key, 0, 1., shape=jnp.shape(z))
        lifetime = interpolation.curve_interpolator(p, self.elife.coordinate_system, self.elife.map)
        drift_survive_prob = jnp.exp(- jnp.abs(z) / parameters['drift_velocity'] / lifetime)
        return key, drift_survive_prob


@export
class ElectronDrifted(Plugin):
    depends_on = ['num_electron', 'drift_survive_prob']
    provides = ['num_electron_drifted']

    @partial(jit, static_argnums=(0, ))
    def simulate(self, key, parameters, num_electron, drift_survive_prob):
        key, num_electron_drifted = randgen.binomial(key, drift_survive_prob, num_electron)
        return key, num_electron_drifted


@export
class S2PE(Plugin):
    depends_on = ['num_electron_drifted', 's2_correction']
    provides = ['num_s2_pe']
    parameters = ('g2', 'gas_gain')

    @partial(jit, static_argnums=(0, ))
    def simulate(self, key, parameters, num_electron_drifted, s2_correction):
        extraction_eff = parameters['g2'] / parameters['gas_gain']
        g2_true = parameters['g2'] * s2_correction
        gas_gain_true = g2_true / extraction_eff

        key, num_electron_extracted = randgen.binomial(key, extraction_eff, num_electron_drifted)

        mean_s2_pe = num_electron_extracted * gas_gain_true
        key, num_s2_pe = randgen.truncate_normal(key, mean_s2_pe, jnp.sqrt(mean_s2_pe), vmin=0)

        return key, num_s2_pe
