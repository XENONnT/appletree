from appletree import Context
from appletree import plugins

@export
class ERBand(Context):
    def __init__(self):
        super().__init__()

        self.register_all(plugins)
