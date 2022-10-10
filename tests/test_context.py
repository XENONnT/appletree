import os
import appletree as apt


def test_rn220_ar37_context():
    """Test Context of Rn220 & Ar37 combine fitting"""
    config = os.path.join(apt.share.CONFPATH, 'apt_config_rn220_ar37_sr0.json')
    context = apt.Context(config)

    context.print_context_summary()
    context.fitting(nwalkers=100, iteration=2, batch_size=int(1e4))


def test_rn220_context():
    """Test Context of Rn220 combine fitting"""
    config = os.path.join(apt.share.CONFPATH, 'apt_config_rn220_sr0.json')
    context = apt.Context(config)

    context.print_context_summary()
    context.fitting(nwalkers=100, iteration=2, batch_size=int(1e4))
