import appletree as apt
from appletree.utils import get_file_path


def test_rn220_context():
    """Test Context of Rn220 combine fitting"""
    context = apt.ContextRn220()

    context.print_context_summary()
    context.fitting(nwalkers=100, iteration=2, batch_size=int(1e4))
    context.dump_post_parameters('_temp.json')

    context['rn220_llh']['rn220_er'].new_component()


def test_rn220_ar37_context():
    """Test Context of Rn220 & Ar37 combine fitting"""
    context = apt.ContextRn220Ar37()

    context.print_context_summary()

    batch_size = int(1e4)
    context.fitting(nwalkers=100, iteration=2, batch_size=batch_size)

    parameters = context.get_post_parameters()
    context.get_n_events_in_hist(parameters, batch_size=batch_size)


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

    batch_size = int(1)
    context.fitting(nwalkers=100, iteration=2, batch_size=batch_size)

    parameters = context.get_post_parameters()
    context.get_n_events_in_hist(parameters, batch_size=batch_size)
