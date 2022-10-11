import appletree as apt

from appletree import ComponentSim


class ERBand(ComponentSim):
    rate_name = 'er_band_rate'
    norm_type = 'on_pdf'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register(apt.plugins.common.UniformEnergySpectra)
        self.register(apt.plugins.common.PositionSpectra)
        self.register_all(apt.plugins.er_microphys)
        self.register_all(apt.plugins.detector)
        self.register_all(apt.plugins.reconstruction)
        self.register_all(apt.plugins.efficiency)


class ERPeak(ComponentSim):
    rate_name = 'er_peak_rate'
    norm_type = 'on_pdf'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register(apt.plugins.common.MonoEnergySpectra)
        self.register(apt.plugins.common.PositionSpectra)
        self.register_all(apt.plugins.er_microphys)
        self.register_all(apt.plugins.detector)
        self.register_all(apt.plugins.reconstruction)
        self.register_all(apt.plugins.efficiency)
