import numpy as np
from jax import numpy as jnp
from jax import jit, random
from functools import partial

from appletree import randgen
from appletree import interpolation
from appletree.plugin import Plugin
from appletree.plugins import detector, reconstruction, efficiency


# Plugins that handle merging
class MSPhotonMerge(Plugin):
    """For MS, photon will always be merged."""
    depends_on = ['num_photon', 'event_id']
    provides = ['num_photon_sum']

    @partial(jit, static_argnums=(0, ))
    def simulate(self, key, parameters, num_photon, event_id):
        num_photon_sum = jnp.zeros_like(num_photon)
        num_photon_sum = num_photon_sum.at[event_id].add(num_photon,
                                                         indices_are_sorted = True)
        return key, num_photon_sum


class MSDriftLoss(Plugin):
    """Different from SS-version, since we want same event has same surival probability"""
    depends_on = ['z', 'event_id']
    provides = ['drift_survive_prob']
    parameters = ('drift_velocity', )

    @partial(jit, static_argnums=(0, ))
    def simulate(self, key, parameters, z, event_id):
        keys = key[jnp.newaxis, :] + event_id[:, jnp.newaxis]
        keys = jnp.asarray(keys, dtype=np.uint32)
        p = randgen.uniform_key_vectorized(keys)
        key = random.split(key)
        lifetime = interpolation.curve_interpolator(p, self.elife.coordinate_system, self.elife.map)
        drift_survive_prob = jnp.exp(- jnp.abs(z) / parameters['drift_velocity'] / lifetime)
        return key, drift_survive_prob


class MSElectronDrifted(detector.ElectronDrifted):
    """Same as SS, ['num_electron', 'drift_survive_prob'] -> ['num_electron_drifted']"""
    pass


class MSElectronMerge(Plugin):
    """This plugin will merge num_electron_drifted based on
    s2_tag: 0 means main S2, 1 means alt S2
    event_id: different id's mean different events, max of which should be < batch_size
    The reconstructed xyz are modeled as the weighted averge of true xyz
    """
    depends_on = [
        'num_electron_drifted',
        'x', 'y', 'z', 'drift_survive_prob',
        's2_tag', 'event_id',
    ]
    provides = [
        'num_electron_drifted_main',
        'num_electron_drifted_alt',
        'rec_x_main', 'rec_x_alt',
        'rec_y_main', 'rec_y_alt',
        'rec_z_main', 'rec_z_alt',
        'rec_drift_survive_prob_main',
        'rec_drift_survive_prob_alt',
    ]

    @partial(jit, static_argnums=(0, ))
    def simulate(self, key, parameters, num_electron_drifted, x, y, z, drift_survive_prob, s2_tag, event_id):
        num_scatter = len(num_electron_drifted)
        ind = jnp.stack((s2_tag, event_id))

        num_electron_drifted_merged = jnp.zeros(shape=(2, num_scatter))
        num_electron_drifted_merged = num_electron_drifted_merged.at[tuple(ind)].add(num_electron_drifted)
        num_electron_drifted_main, num_electron_drifted_alt = num_electron_drifted_merged

        rec_x = jnp.zeros(shape=(2, num_scatter))
        rec_x = rec_x.at[tuple(ind)].add(num_electron_drifted * x)
        rec_x /= jnp.where(num_electron_drifted_merged > 0, num_electron_drifted_merged, 1)
        rec_x_main, rec_x_alt = rec_x

        rec_y = jnp.zeros(shape=(2, num_scatter))
        rec_y = rec_y.at[tuple(ind)].add(num_electron_drifted * y)
        rec_y /= jnp.where(num_electron_drifted_merged > 0, num_electron_drifted_merged, 1)
        rec_y_main, rec_y_alt = rec_y

        rec_z = jnp.zeros(shape=(2, num_scatter))
        rec_z = rec_z.at[tuple(ind)].add(num_electron_drifted * z)
        rec_z /= jnp.where(num_electron_drifted_merged > 0, num_electron_drifted_merged, 1)
        rec_z_main, rec_z_alt = rec_z

        # We assume in MS they are close enough
        # geometric average (average in z) ~ arithmetic mean
        rec_drift_survive_prob = jnp.ones(shape=(2, num_scatter))
        rec_drift_survive_prob = rec_drift_survive_prob.at[tuple(ind)].add(num_electron_drifted * rec_drift_survive_prob)
        rec_drift_survive_prob /= jnp.where(num_electron_drifted_merged > 0, num_electron_drifted_merged, 1)
        rec_drift_survive_prob_main, rec_drift_survive_prob_alt = rec_drift_survive_prob

        return (
            key,
            num_electron_drifted_main,
            num_electron_drifted_alt,
            rec_x_main, rec_x_alt,
            rec_y_main, rec_y_alt,
            rec_z_main, rec_z_alt,
            rec_drift_survive_prob_main,
            rec_drift_survive_prob_alt,
        )


class MSS1Correction(detector.S1Correction):
    depends_on = ['rec_x_main', 'rec_y_main', 'rec_z_main']
    provides = ['s1_correction']


class MSS2CorrectionMain(detector.S2Correction):
    depends_on = ['rec_x_main', 'rec_y_main']
    provides = ['s2_correction_main']


class MSS2CorrectionAlt(detector.S2Correction):
    depends_on = ['rec_x_alt', 'rec_y_alt']
    provides = ['s2_correction_alt']


# S1 workflow
class MSPhotonDetection(detector.PhotonDetection):
    depends_on = ['num_photon_sum', 's1_correction']
    provides = ['num_s1_phd']


class MSS1PE(detector.S1PE):
    pass


class MSS1(reconstruction.S1):
    pass


class MScS1(reconstruction.cS1):
    pass


# S2 workflow
class MSS2PEMain(detector.S2PE):
    depends_on = ['num_electron_drifted_main', 's2_correction_main']
    provides = ['num_s2_pe_main']


class MSS2PEAlt(detector.S2PE):
    depends_on = ['num_electron_drifted_alt', 's2_correction_alt']
    provides = ['num_s2_pe_alt']


class MSS2Main(reconstruction.S2):
    depends_on = ['num_s2_pe_main']
    provides = ['s2_main']


class MSS2Alt(reconstruction.S2):
    depends_on = ['num_s2_pe_alt']
    provides = ['s2_alt']


class MScS2Main(reconstruction.cS2):
    depends_on = ['s2_main', 's2_correction_main', 'rec_drift_survive_prob_main']
    provides = ['cs2_main']


class MScS2Alt(reconstruction.cS2):
    depends_on = ['s2_alt', 's2_correction_alt', 'rec_drift_survive_prob_alt']
    provides = ['cs2_alt']


class MSS2Threshold(efficiency.S2Threshold):
    depends_on = ['s2_main']


class MSS1ReconEff(efficiency.S1ReconEff):
    pass


class MSS1CutAccept(efficiency.S1CutAccept):
    pass


class MSS2CutAccept(efficiency.S2CutAccept):
    depends_on = ['s2_main']


class MSSSCutAccept(Plugin):
    depends_on = ['s2_main', 's2_alt']
    provides = ['acc_ss_cut']

    @partial(jit, static_argnums=(0, ))
    def simulate():
        raise NotImplementedError


class MSEff(Plugin):
    depends_on = ['acc_s2_threshold', 'acc_s1_recon_eff', 'cut_acc_s1', 'cut_acc_s2', 'acc_ss_cut']
    provides = ['eff']

    @partial(jit, static_argnums=(0, ))
    def simulate(self, key, parameters, acc_s2_threshold, acc_s1_recon_eff, cut_acc_s1, cut_acc_s2, acc_ss_cut):
        eff = acc_s2_threshold * acc_s1_recon_eff * cut_acc_s1 * cut_acc_s2 * acc_ss_cut
        return key, eff


