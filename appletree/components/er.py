import appletree as apt

from appletree.plugins import UniformEnergySpectra, MonoEnergySpectra
from appletree import ComponentSim


class ERBand(ComponentSim):
    norm_type = 'on_pdf'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register_all(apt.plugins)
        self.register(UniformEnergySpectra)
        self.register_all(apt.plugins.microphys)


class ERPeak(ComponentSim):
    norm_type = 'on_pdf'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register_all(apt.plugins)
        self.register(MonoEnergySpectra)
        self.register_all(apt.plugins.microphys)
