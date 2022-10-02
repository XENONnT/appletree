from appletree import plugins
from appletree.plugins import UniformEnergySpectra, MonoEnergySpectra
from appletree import ComponentSim


class ERBand(ComponentSim):
    rate_name = 'er_band_rate'
    norm_type = 'on_pdf'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register_all(plugins)
        self.register(UniformEnergySpectra)


class ERPeak(ComponentSim):
    rate_name = 'er_peak_rate'
    norm_type = 'on_pdf'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register_all(plugins)
        self.register(MonoEnergySpectra)
