import appletree as apt
from appletree.share import _cached_configs
from appletree.utils import get_file_path, check_unused_configs


def test_rn220_context():
    """Test Context of Rn220 combine fitting."""
    _cached_configs.clear()
    context = apt.ContextRn220()

    context.print_context_summary()
    context.fitting(nwalkers=100, iteration=2, batch_size=int(1e4))
    context.dump_post_parameters("_temp.json")

    context["rn220_llh"]["rn220_er"].new_component()
    check_unused_configs()


def test_rn220_ar37_context():
    """Test Context of Rn220 & Ar37 combine fitting."""
    _cached_configs.clear()
    context = apt.ContextRn220Ar37()

    context.print_context_summary()

    batch_size = int(1e4)
    context.fitting(nwalkers=100, iteration=2, batch_size=batch_size)

    parameters = context.get_post_parameters()
    context.get_num_events_accepted(parameters, batch_size=batch_size)
    check_unused_configs()


def test_neutron_context():
    """Test Context of neutron combine fitting."""
    _cached_configs.clear()
    instruct = get_file_path("neutron_low.json")
    context = apt.Context(instruct)

    context.print_context_summary()
    context.fitting(nwalkers=100, iteration=2, batch_size=int(1e4))
    check_unused_configs()


def test_literature_context():
    """Test Context of neutron combine fitting."""
    _cached_configs.clear()
    instruct = get_file_path("literature_lyqy.json")
    context = apt.Context(instruct)

    context.print_context_summary()

    batch_size = int(1)
    context.fitting(nwalkers=100, iteration=2, batch_size=batch_size)

    parameters = context.get_post_parameters()
    context.get_num_events_accepted(parameters, batch_size=batch_size)
    check_unused_configs()
