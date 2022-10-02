from appletree import ComponentSim
from appletree import plugins

class ERBand(ComponentSim):
    rate_name = 'er_rate'
    norm_type = 'on_pdf'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register_all(plugins)
