from jax import jit
from functools import partial

from jax import numpy as jnp

from appletree import randgen
from appletree.config import takes_config, Map
from appletree.plugin import Plugin
from appletree.utils import exporter

export, __all__ = exporter(export_self=False)


@export
@takes_config(
    Map(
        name="posrec_reso",
        method="LERP",
        default="_posrec_reso.json",
        help="Position reconstruction resolution",
    ),
)
class PositionRecon(Plugin):
    depends_on = ["x", "y", "z", "num_electron_drifted"]
    provides = ["rec_x", "rec_y", "rec_z", "rec_r"]

    @partial(jit, static_argnums=(0,))
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
@takes_config(
    Map(
        name="s1_bias_3f",
        method="LERP",
        default="_s1_bias.json",
        help="3fold S1 reconstruction bias",
    ),
    Map(
        name="s1_smear_3f",
        method="LERP",
        default="_s1_smearing.json",
        help="3fold S1 reconstruction smearing",
    ),
)
class S1(Plugin):
    depends_on = ["num_s1_pe"]
    provides = ["s1_area"]

    p_bernoulli = (0.3720687858259511, 0.29198438734498294, 0.24134987719993664) #from sprinkling
    mu = 1.042 #in PE
    sigma = 0.336 #in PE

    @partial(jit, static_argnums=(0,))
    def simulate(self, key, parameters, num_s1_pe):
        mean = self.s1_bias_3f.apply(num_s1_pe)
        std = self.s1_smear_3f.apply(num_s1_pe)
        key, bias = randgen.normal(key, mean, std)
        _s1_area = num_s1_pe * (1.0 + bias)
        
        key, bernoulli_0 = randgen.bernoulli(key,self.p_bernoulli[0],shape = (len(_s1_area),))
        key, bernoulli_1 = randgen.bernoulli(key,self.p_bernoulli[1],shape = (len(_s1_area),))
        key, bernoulli_2 = randgen.bernoulli(key,self.p_bernoulli[2],shape = (len(_s1_area),))
        add_PE = bernoulli_0
        add_PE = jnp.where(add_PE==1,add_PE+bernoulli_1,add_PE)
        add_PE = jnp.where(add_PE==2, add_PE+bernoulli_2, add_PE)
        key, bias_ambience = truncate_normal(key, self.mu*add_PE, self.sigma*jnp.sqrt(add_PE), vmin=0.2, shape=(len(_s1_area),))
        s1_area = _s1_area + bias_ambience
        return key, s1_area



@export
@takes_config(
    Map(
        name="s2_bias",
        method="LERP",
        default="_s2_bias.json",
        help="S2 reconstruction bias",
    ),
    Map(
        name="s2_smear",
        method="LERP",
        default="_s2_smearing.json",
        help="S2 reconstruction smearing",
    ),
)
class S2(Plugin):
    depends_on = ["num_s2_pe"]
    provides = ["s2_area"]
    
    p_binom_value=0.248
    mu=5.56 # in PE
    mu2=26.23
    sigma=5.59
    sigma2=26.48
    
    @partial(jit, static_argnums=(0,))
    def simulate(self, key, parameters, num_s2_pe):
        mean = self.s2_bias.apply(num_s2_pe)
        std = self.s2_smear.apply(num_s2_pe)
        key, bias = randgen.normal(key, mean, std)
        _s2_area = num_s2_pe * (1.0 + bias)

        ones = jnp.ones(len(_s2_area))
        key, p_binom = randgen.binomial(key, self.p_binom_value, ones)
        mean = jnp.where(p_binom==0, ones*self.mu, ones*self.mu2)
        std = jnp.where(p_binom==0, ones*self.sigma, ones*self.sigma2)
        key, bias_ambience = truncate_normal(key, mean, std, vmin=0,shape=(len(_s2_area),))
        s2_area = _s2_area + bias_ambience
        return key, s2_area


import numpy as np
from jax import random
import jax
if jax.config.x64_enabled:
    INT = np.int64
    FLOAT = np.float64
else:
    INT = np.int32
    FLOAT = np.float32
    
def truncate_normal(key, mean, std, vmin=None, vmax=None, shape=()):
    """Truncated normal distribution random sampler.
    Args:
        key: seed for random generator.
        mean: <jnp.array>-like mean in normal distribution.
        std: <jnp.array>-like std in normal distribution.
        vmin: <jnp.array>-like min value to clip. By default it's None.
            vmin and vmax cannot be both None.
        vmax: <jnp.array>-like max value to clip. By default it's None.
            vmin and vmax cannot be both None.
        shape: parameter passed to normal(..., shape=shape)
    Returns:
        an updated seed, random variables.
    """
    # Assume that vmin and vmax cannot both be None, and vmax > vmin
    if vmin is None:
        lower_norm = -jnp.inf
        upper_norm = (vmax - mean) / std
    elif vmax is None:
        lower_norm = (vmin - mean) / std
        upper_norm = jnp.inf
    else:
        lower_norm, upper_norm = (vmin - mean) / std, (vmax - mean) / std
    shape = shape or jnp.broadcast_shapes(
        jnp.shape(mean), jnp.shape(std), jnp.shape(lower_norm), jnp.shape(upper_norm)
    )
    rvs = random.truncated_normal(key, lower_norm, upper_norm, shape=shape)
    rvs = rvs * std + mean
    return key, rvs.astype(FLOAT)





@export
@takes_config(
    Map(
        name="s1_correction",
        default="_s1_correction.json",
        help="S1 xyz correction on reconstructed positions",
    ),
)
class S1Correction(Plugin):
    depends_on = ["rec_x", "rec_y", "rec_z"]
    provides = ["s1_correction"]

    @partial(jit, static_argnums=(0,))
    def simulate(self, key, parameters, rec_x, rec_y, rec_z):
        pos_rec = jnp.stack([rec_x, rec_y, rec_z]).T
        s1_correction = self.s1_correction.apply(pos_rec)
        return key, s1_correction


@export
@takes_config(
    Map(
        name="s2_correction",
        default="_s2_correction.json",
        help="S2 xy correction on constructed positions",
    ),
)
class S2Correction(Plugin):
    depends_on = ["rec_x", "rec_y"]
    provides = ["s2_correction"]

    @partial(jit, static_argnums=(0,))
    def simulate(self, key, parameters, rec_x, rec_y):
        pos_rec = jnp.stack([rec_x, rec_y]).T
        s2_correction = self.s2_correction.apply(pos_rec)
        return key, s2_correction


@export
class cS1(Plugin):
    depends_on = ["s1_area", "s1_correction"]
    provides = ["cs1"]

    @partial(jit, static_argnums=(0,))
    def simulate(self, key, parameters, s1_area, s1_correction):
        cs1 = s1_area / s1_correction
        return key, cs1


@export
class cS2(Plugin):
    depends_on = ["s2_area", "s2_correction", "drift_survive_prob"]
    provides = ["cs2"]

    @partial(jit, static_argnums=(0,))
    def simulate(self, key, parameters, s2_area, s2_correction, drift_survive_prob):
        cs2 = s2_area / s2_correction / drift_survive_prob
        return key, cs2
