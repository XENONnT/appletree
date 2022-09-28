from appletree import ComponentSim
from appletree import plugins

class ERBand(ComponentSim):
    rate_par_name = 'er_rate'
    norm_type = 'on_pdf'

    def __init__(self):
        super().__init__()

        self.register_all(plugins)
