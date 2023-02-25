import appletree as apt

from appletree.component import ComponentSim


class MonoEnergiesYields(ComponentSim):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register(apt.plugins.MonoEnergiesSpectra)
        self.register(apt.plugins.TotalQuanta)
        self.register(apt.plugins.TIB)
        self.register(apt.plugins.Qy)
        self.register(apt.plugins.Ly)
        self.register(apt.plugins.MonoEnergiesClipEff)


class BandEnergiesYields(ComponentSim):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register(apt.plugins.UniformEnergiesSpectra)
        self.register(apt.plugins.TotalQuanta)
        self.register(apt.plugins.TIB)
        self.register(apt.plugins.Qy)
        self.register(apt.plugins.Ly)
        self.register(apt.plugins.BandEnergiesClipEff)
