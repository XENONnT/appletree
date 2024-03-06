import appletree as apt
from appletree import Plotter
from appletree.utils import load_json


def test_plot():
    """Test plot of Rn220 fitting."""
    instruction = load_json("rn220.json")

    filename = "rn220.h5"
    instruction["backend_h5"] = filename

    context = apt.Context(instruction)

    context.print_context_summary(short=False)
    context.fitting(nwalkers=100, iteration=2, batch_size=int(1e4))

    plotter = Plotter(filename)
    plotter.make_all_plots()
