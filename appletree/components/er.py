import appletree as apt

from appletree.component import ComponentSim
from applefiles import aptext

class ERBand(ComponentSim):
    norm_type = "on_pdf"
    add_eps_to_hist = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register(apt.plugins.common.UniformEnergySpectra)
        self.register(apt.plugins.common.PositionSpectra)
        self.register_all(apt.plugins.er_microphys)
        self.register_all(apt.plugins.new_detector)
        self.register_all(apt.plugins.reconstruction)
        self.register_all(apt.plugins.new_efficiency)


class mERBand(ComponentSim):
    norm_type = "on_pdf"
    add_eps_to_hist = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register(apt.plugins.common.UniformEnergySpectra)
        self.register(apt.plugins.common.PositionSpectra)
        self.register_all(apt.plugins.er_microphys)
        self.register_all(apt.plugins.new_detector)
        self.register_all(aptext.field.s1_n_hits)
        self.register_all(aptext.field.s1_max_pmt)
        self.register_all(apt.plugins.reconstruction)
        self.register_all(apt.plugins.efficiency)


class ERPeak(ComponentSim):
    norm_type = "on_pdf"
    add_eps_to_hist = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register(apt.plugins.common.MonoEnergySpectra)
        self.register(apt.plugins.common.PositionSpectra)
        self.register_all(apt.plugins.er_microphys)
        self.register_all(apt.plugins.new_detector)
        self.register_all(apt.plugins.reconstruction)
        self.register_all(apt.plugins.new_efficiency)
