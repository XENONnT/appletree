from appletree import ComponentSim
from appletree import plugins

class ERBand(ComponentSim):
    def __init__(self):
        super().__init__()

        self.register_all(plugins)
