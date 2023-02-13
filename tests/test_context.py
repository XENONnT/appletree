import appletree as apt
from appletree.utils import get_file_path


def test_rn220_context():
    """Test Context of Rn220 combine fitting"""
    config = get_file_path('rn220.json')
    context = apt.Context(config)

    context.print_context_summary()
    context.fitting(nwalkers=100, iteration=2, batch_size=int(1e4))
    context.dump_post_parameters('_temp.json')


def test_rn220_ar37_context():
    """Test Context of Rn220 & Ar37 combine fitting"""
    config = get_file_path('rn220_ar37.json')
    context = apt.Context(config)

    context.print_context_summary()
    context.fitting(nwalkers=100, iteration=2, batch_size=int(1e4))


def test_neutron_context():
    """Test Context of neutron combine fitting"""
    config = get_file_path('neutron_low.json')
    context = apt.Context(config)

    context.print_context_summary()
    context.fitting(nwalkers=100, iteration=2, batch_size=int(1e4))


def test_literature_context():
    """Test Context of neutron combine fitting"""
    config = get_file_path('literature_lyqy.json')
    context = apt.Context(config)

    context.print_context_summary()
    context.fitting(nwalkers=100, iteration=2, batch_size=int(1))
