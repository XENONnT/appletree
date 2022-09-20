import jax.numpy as jnp

from jax import jit, lax
from functools import partial

from appletree import exporter

export, __all__ = exporter(export_self=False)


@jit
def _L2_dist2(pos1, pos2):
    """
    pos1 : <jnp.array>-like with shape (N, D)
    pos2 : <jnp.array>-like with shape (M, D)

    return : L2 distance squared with shape (N, M)
    """
    dr = jnp.expand_dims(pos1, axis=1) - jnp.expand_dims(pos2, axis=0)
    return jnp.sum(dr * dr, axis=-1)


@export
@partial(jit, static_argnums=(3,))
def map_interpolator_knn(pos, ref_pos, ref_val, k=3):
    """
    pos : <jnp.array> with shape (N, D), as the points to be interpolated.
    ref_pos : <jnp.array> with shape (M, D), as the reference points.
    ref_val : <jnp.array> with shape (M, ), as the reference values.

    return : interpolated values with shape (N, ), weighted by the inverse of distance to k nearest neighbors. 
    """
    pos = jnp.asarray(pos)
    ref_pos = jnp.asarray(ref_pos)
    ref_val = jnp.asarray(ref_val)
    
    dr2 = -_L2_dist2(pos, ref_pos)
    dr2, ind = lax.top_k(dr2, k)
    weights = 1.0 / jnp.clip(jnp.sqrt(-dr2), 1e-6, float('inf'))
    val = jnp.take(ref_val, ind)
    val = jnp.sum(val * weights, axis=1) / jnp.sum(weights, axis=1)
    
    return val


@export
@jit
def curve_interpolator(pos, ref_pos, ref_val):
    """
    pos : <jnp.array> with shape (N, ), as the points to be interpolated.
    ref_pos : <jnp.array> with shape (M, ), as the reference points.
    ref_val : <jnp.array> with shape (M, ), as the reference values.

    return : interpolated values with shape (N, ), weighted by the inverse of distance to k nearest neighbors. 
    """
    right = jnp.searchsorted(ref_pos, pos)
    left = right - 1

    right = jnp.clip(right, 0, len(ref_pos)-1)
    left = jnp.clip(left, 0, len(ref_pos)-1)

    val_right = ref_val[right]
    val_left = ref_val[left]


    dist_right = jnp.abs(pos - ref_pos[right])
    dist_left = jnp.abs(pos - ref_pos[left])

    val = (val_right*dist_left+val_left*dist_right)/(dist_right+dist_left)
    
    return val


@export
@jit
def map_interpolator_regular_binning_2d(pos, ref_pos_lowers, ref_pos_uppers, ref_val):
    num_bins = jnp.asarray(jnp.shape(ref_val))
    bin_sizes = (ref_pos_uppers - ref_pos_lowers) / (num_bins - 1)
    num_bins = num_bins[jnp.newaxis, :]
    bin_sizes = bin_sizes[jnp.newaxis, :]
    
    ind1 = jnp.asarray(jnp.clip(jnp.floor((pos - ref_pos_lowers) / bin_sizes), a_min=0, a_max=num_bins-1), dtype=int)
    ind2 = ind1.at[:, 0].add(1)
    ind3 = ind1.at[:, 1].add(1)
    ind4 = ind2.at[:, 1].add(1)

    val1 = ref_val[ind1[:,0], ind1[:,1]]
    val2 = ref_val[ind2[:,0], ind2[:,1]]
    val3 = ref_val[ind3[:,0], ind3[:,1]]
    val4 = ref_val[ind4[:,0], ind4[:,1]]

    ref_pos1 = ref_pos_lowers + bin_sizes * ind1
    ref_pos2 = ref_pos_lowers + bin_sizes * ind2
    ref_pos3 = ref_pos_lowers + bin_sizes * ind3
    ref_pos4 = ref_pos_lowers + bin_sizes * ind4

    dr1 = jnp.clip(jnp.sqrt(jnp.sum((ref_pos1 - pos)**2, axis=-1)), a_min=1e-10)
    dr2 = jnp.clip(jnp.sqrt(jnp.sum((ref_pos2 - pos)**2, axis=-1)), a_min=1e-10)
    dr3 = jnp.clip(jnp.sqrt(jnp.sum((ref_pos3 - pos)**2, axis=-1)), a_min=1e-10)
    dr4 = jnp.clip(jnp.sqrt(jnp.sum((ref_pos4 - pos)**2, axis=-1)), a_min=1e-10)

    val = (val1/dr1+val2/dr2+val3/dr3+val4/dr4) / (1./dr1+1./dr2+1./dr3+1./dr4)
    
    return val


@export
@jit
def map_interpolator_regular_binning_3d(pos, ref_pos_lowers, ref_pos_uppers, ref_val):
    num_bins = jnp.asarray(jnp.shape(ref_val))
    bin_sizes = (ref_pos_uppers - ref_pos_lowers) / (num_bins - 1)
    num_bins = num_bins[jnp.newaxis, :]
    bin_sizes = bin_sizes[jnp.newaxis, :]
    
    ind1 = jnp.asarray(jnp.clip(jnp.floor((pos - ref_pos_lowers) / bin_sizes), a_min=0, a_max=num_bins-1), dtype=int)
    ind2 = ind1.at[:, 0].add(1)
    ind3 = ind1.at[:, 1].add(1)
    ind4 = ind1.at[:, 2].add(1)
    ind5 = ind2.at[:, 1].add(1)
    ind6 = ind3.at[:, 2].add(1)
    ind7 = ind4.at[:, 0].add(1)
    ind8 = ind7.at[:, 1].add(1)

    val1 = ref_val[ind1[:,0], ind1[:,1], ind1[:,2]]
    val2 = ref_val[ind2[:,0], ind2[:,1], ind2[:,2]]
    val3 = ref_val[ind3[:,0], ind3[:,1], ind3[:,2]]
    val4 = ref_val[ind4[:,0], ind4[:,1], ind4[:,2]]
    val5 = ref_val[ind5[:,0], ind5[:,1], ind5[:,2]]
    val6 = ref_val[ind6[:,0], ind6[:,1], ind6[:,2]]
    val7 = ref_val[ind7[:,0], ind7[:,1], ind7[:,2]]
    val8 = ref_val[ind8[:,0], ind8[:,1], ind8[:,2]]

    ref_pos1 = ref_pos_lowers + bin_sizes * ind1
    ref_pos2 = ref_pos_lowers + bin_sizes * ind2
    ref_pos3 = ref_pos_lowers + bin_sizes * ind3
    ref_pos4 = ref_pos_lowers + bin_sizes * ind4
    ref_pos5 = ref_pos_lowers + bin_sizes * ind5
    ref_pos6 = ref_pos_lowers + bin_sizes * ind6
    ref_pos7 = ref_pos_lowers + bin_sizes * ind7
    ref_pos8 = ref_pos_lowers + bin_sizes * ind8

    dr1 = jnp.clip(jnp.sqrt(jnp.sum((ref_pos1 - pos)**2, axis=-1)), a_min=1e-10)
    dr2 = jnp.clip(jnp.sqrt(jnp.sum((ref_pos2 - pos)**2, axis=-1)), a_min=1e-10)
    dr3 = jnp.clip(jnp.sqrt(jnp.sum((ref_pos3 - pos)**2, axis=-1)), a_min=1e-10)
    dr4 = jnp.clip(jnp.sqrt(jnp.sum((ref_pos4 - pos)**2, axis=-1)), a_min=1e-10)
    dr5 = jnp.clip(jnp.sqrt(jnp.sum((ref_pos5 - pos)**2, axis=-1)), a_min=1e-10)
    dr6 = jnp.clip(jnp.sqrt(jnp.sum((ref_pos6 - pos)**2, axis=-1)), a_min=1e-10)
    dr7 = jnp.clip(jnp.sqrt(jnp.sum((ref_pos7 - pos)**2, axis=-1)), a_min=1e-10)
    dr8 = jnp.clip(jnp.sqrt(jnp.sum((ref_pos8 - pos)**2, axis=-1)), a_min=1e-10)

    val = (val1/dr1+val2/dr2+val3/dr3+val4/dr4+val5/dr5+val6/dr6+val7/dr7+val8/dr8) / (1./dr1+1./dr2+1./dr3+1./dr4+1./dr5+1./dr6+1./dr7+1./dr8)
    
    return val