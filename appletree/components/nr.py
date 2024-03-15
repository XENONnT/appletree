import appletree as apt

from appletree.component import ComponentSim


class NR(ComponentSim):
    norm_type = "on_pdf"
    add_eps_to_hist = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register(apt.plugins.common.FixedEnergySpectra)
        self.register(apt.plugins.common.PositionSpectra)
        self.register_all(apt.plugins.lyqy)
        self.register_all(apt.plugins.detector)
        self.register_all(apt.plugins.reconstruction)
        self.register_all(apt.plugins.efficiency)
