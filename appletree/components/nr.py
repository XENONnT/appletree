import appletree as apt

from appletree.plugins import FixedEnergySpectra
from appletree import ComponentSim


class NRBand(ComponentSim):
    norm_type = 'on_pdf'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register_all(apt.plugins)
        self.register(FixedEnergySpectra)
        self.register_all(apt.plugins.nr_microphys)
