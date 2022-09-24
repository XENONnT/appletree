from appletree import exporter

export, __all__ = exporter()

@export
class Plugin():
    depends_on = []
    provides = []

    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        return self.simulate(*args, **kwargs)

    def simulate(self, *args, **kwargs):
        raise NotImplementedError

    def sanity_check(self):
        """
        Check the consistency between `depends_on`, `provides` and in(out)put of `simulation`
        """
        pass
