import appletree as apt

from appletree import ComponentSim


class NRband(ComponentSim):
    rate_name = 'nr_band_rate'
    norm_type = 'on_pdf'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register_all(apt.plugins)
        self.register_all(apt.plugins.nr_microphys)