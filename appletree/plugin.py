from appletree import exporter

export, __all__ = exporter()

@export
class Plugin():
    depends_on = tuple()
    provides = tuple()

    def __init__(self):
        self.par_names = []

    def __call__(self, *args, **kwargs):
        return self.simulate(*args, **kwargs)

    def simulate(self, *args, **kwargs):
        raise NotImplementedError
