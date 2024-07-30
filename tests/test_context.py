import appletree as apt
from appletree.share import _cached_configs, _cached_functions
from appletree.utils import get_file_path, load_json, check_unused_configs


def test_rn220_context():
    """Test Context of Rn220 fitting."""
    _cached_functions.clear()
    _cached_configs.clear()
    context = apt.ContextRn220()
    context.lineage_hash

    context.print_context_summary(short=False)
    context.fitting(nwalkers=100, iteration=2, batch_size=int(1e4))
    context.dump_post_parameters("_temp.json")

    context["rn220_llh"]["rn220_er"].new_component()
    check_unused_configs()


def test_rn220_context_1d():
    """Test 1D Context of Rn220 fitting."""
    instruction = load_json("rn220.json")

    bins = instruction["likelihoods"]["rn220_llh"]["bins"][1]
    instruction["likelihoods"]["rn220_llh"]["bins_on"] = "cs2"
    instruction["likelihoods"]["rn220_llh"]["clip"] = instruction["likelihoods"]["rn220_llh"][
        "y_clip"
    ]
    instruction["likelihoods"]["rn220_llh"].pop("x_clip", None)
    instruction["likelihoods"]["rn220_llh"].pop("y_clip", None)

    for bins_type, bins in zip(
        ["equiprob", "meshgrid", "irreg"],
        [bins, bins, [instruction["likelihoods"]["rn220_llh"]["clip"]]],
    ):
        _cached_functions.clear()
        _cached_configs.clear()
        instruction["likelihoods"]["rn220_llh"]["bins_type"] = bins_type
        instruction["likelihoods"]["rn220_llh"]["bins"] = bins
        context = apt.Context(instruction)
        context.lineage_hash
        context.print_context_summary(short=False)


def test_rn220_ar37_context():
    """Test Context of Rn220 & Ar37 combine fitting."""
    _cached_functions.clear()
    _cached_configs.clear()
    context = apt.ContextRn220Ar37()
    context.lineage_hash

    context.print_context_summary(short=False)

    batch_size = int(1e4)
    context.fitting(nwalkers=100, iteration=2, batch_size=batch_size)

    parameters = context.get_post_parameters()
    context.get_num_events_accepted(parameters, batch_size=batch_size)
    check_unused_configs()


def test_neutron_context():
    """Test Context of neutron combine fitting."""
    _cached_functions.clear()
    _cached_configs.clear()
    instruct = get_file_path("neutron_low.json")
    context = apt.Context(instruct)
    context.lineage_hash

    context.print_context_summary(short=False)
    context.fitting(nwalkers=100, iteration=2, batch_size=int(1e4))
    check_unused_configs()


def test_literature_context():
    """Test Context of neutron combine fitting."""
    _cached_functions.clear()
    _cached_configs.clear()
    instruct = get_file_path("literature_lyqy.json")
    context = apt.Context(instruct)
    context.lineage_hash

    context.print_context_summary(short=False)

    batch_size = int(1)
    context.fitting(nwalkers=100, iteration=2, batch_size=batch_size)

    parameters = context.get_post_parameters()
    context.get_num_events_accepted(parameters, batch_size=batch_size)
    check_unused_configs()


def test_backend():
    """Test backend, initialize from backend and continue fitting."""
    _cached_functions.clear()
    _cached_configs.clear()
    instruct = apt.utils.load_json("rn220.json")
    instruct["backend_h5"] = "test_backend.h5"
    context = apt.Context(instruct)
    context.lineage_hash
    context.fitting(nwalkers=100, iteration=2, batch_size=int(1e4))

    _cached_functions.clear()
    _cached_configs.clear()
    context = apt.Context.from_backend("test_backend.h5")
    context.continue_fitting(iteration=2, batch_size=int(1e4))
    assert context.sampler.get_chain().shape[0] == 4
