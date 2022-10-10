import os
import appletree as apt


def test_context():
    """Test Context"""
    config = os.path.join(apt.share.CONFPATH, 'apt_config_rn220_ar37_sr0.json')
    context = apt.Context(config)

    context.print_context_summary()
    context.fitting(nwalkers=100, iteration=2, batch_size=int(1e4))
