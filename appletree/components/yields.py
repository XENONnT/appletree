import appletree as apt

from appletree.component import ComponentSim


class MonoEnergyYields(ComponentSim):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register(apt.plugins.MonoEnergySpectra)
        self.register(apt.plugins.TotalQuanta)
        self.register(apt.plugins.TIB)
        self.register(apt.plugins.Qy)
        self.register(apt.plugins.Ly)
        self.register(apt.plugins.MonoEnergyClipEff)


class BandEnergyYields(ComponentSim):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register(apt.plugins.BandEnergySpectra)
        self.register(apt.plugins.TotalQuanta)
        self.register(apt.plugins.TIB)
        self.register(apt.plugins.Qy)
        self.register(apt.plugins.Ly)
        self.register(apt.plugins.BandEnergyClipEff)
