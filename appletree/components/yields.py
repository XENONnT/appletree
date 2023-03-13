import appletree as apt

from appletree.component import ComponentSim


class MonoEnergiesYields(ComponentSim):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register(apt.plugins.MonoEnergiesSpectra)
        self.register(apt.plugins.TotalQuanta)
        self.register(apt.plugins.ThomasImelBox)
        self.register(apt.plugins.QyNR)
        self.register(apt.plugins.LyNR)
        self.register(apt.plugins.MonoEnergiesClipEff)


class BandEnergiesYields(ComponentSim):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register(apt.plugins.UniformEnergiesSpectra)
        self.register(apt.plugins.TotalQuanta)
        self.register(apt.plugins.ThomasImelBox)
        self.register(apt.plugins.QyNR)
        self.register(apt.plugins.LyNR)
        self.register(apt.plugins.BandEnergiesClipEff)
