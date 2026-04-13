import aptext

import appletree as apt
from appletree.component import ComponentSim
from appletree.utils import exporter

export, __all__ = exporter(export_self=False)


@export
class AmBeSR0(ComponentSim):
    # This component is deprecated, only to reproduce SR0 results
    norm_type = "on_sim"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register(apt.plugins.er_microphys.Quanta)
        self.register_all(apt.plugins.detector)
        self.register_all(apt.plugins.reconstruction)
        self.register_all(apt.plugins.efficiency)

        self.register_all(aptext.multiscatter)
        self.register(aptext.s1_n_hits.MSS1NHits)
        self.register(aptext.efficiency_2f.AntiCorrelationEfficiency)
        self.register(aptext.multiscatter.MSFullEff)

        self.register_all(aptext.nr_nest_v1)

        self.register(aptext.cut_fv.AmBeFiducialVolumeAccept)


@export
class Rn220(ComponentSim):
    norm_type = "on_pdf"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Simulations
        self.register(apt.plugins.common.UniformEnergySpectra)
        self.register(apt.plugins.common.PositionSpectra)
        self.register_all(apt.plugins.er_microphys)
        self.register_all(apt.plugins.detector)
        self.register_all(apt.plugins.reconstruction)
        self.register(aptext.field.S1NHits)

        # Efficiencies
        self.register(apt.plugins.efficiency.S1CutAccept)
        self.register(apt.plugins.efficiency.S2CutAccept)
        self.register(apt.plugins.efficiency.S2Threshold)
        self.register(aptext.acceptance.cut_fv.FiducialVolumeCylinderAccept)
        self.register(aptext.acceptance.efficiency_3f.S1ReconEffNHits)
        self.register(aptext.efficiency_2f.AntiCorrelationEfficiency)
        self.register(aptext.acceptance.efficiency_total.FullEffSS)
        
@export
class Rn220SR2_Extension(ComponentSim):
    norm_type = "on_pdf"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Simulations
        self.register(apt.plugins.common.UniformEnergySpectra)
        self.register(apt.plugins.common.PositionSpectra)
        self.register_all(apt.plugins.nestv2_er_extension)
        self.register_all(apt.plugins.detector)
        self.register(apt.plugins.PositionRecon)
        self.register(apt.plugins.S1)
        self.register(apt.plugins.S2)
        self.register(aptext.field.S1CorrectionWithBias)
        self.register(aptext.field.S2CorrectionWithBias)
        self.register(apt.plugins.cS1)
        self.register(apt.plugins.cS2)
        self.register(aptext.field.S1NHits)

        # Efficiencies
        self.register(apt.plugins.efficiency.S1CutAccept)
        self.register(apt.plugins.efficiency.S2CutAccept)
        self.register(apt.plugins.efficiency.S2Threshold)
        self.register(aptext.acceptance.cut_fv.FiducialVolumeCylinderAccept)
        self.register(aptext.acceptance.efficiency_3f.S1ReconEffNHits)
        self.register(aptext.efficiency_2f.AntiCorrelationEfficiency)
        self.register(aptext.acceptance.efficiency_total.FullEffSS)


@export
class Rn220SR2(ComponentSim):
    norm_type = "on_pdf"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Simulations
        self.register(apt.plugins.common.UniformEnergySpectra)
        self.register(apt.plugins.common.PositionSpectra)
        self.register_all(apt.plugins.er_nestv2)
        self.register_all(apt.plugins.detector)
        self.register(apt.plugins.PositionRecon)
        self.register(apt.plugins.S1)
        self.register(apt.plugins.S2)
        self.register(aptext.field.S1CorrectionWithBias)
        self.register(aptext.field.S2CorrectionWithBias)
        self.register(apt.plugins.cS1)
        self.register(apt.plugins.cS2)
        self.register(aptext.field.S1NHits)

        # Efficiencies
        self.register(apt.plugins.efficiency.S1CutAccept)
        self.register(apt.plugins.efficiency.S2CutAccept)
        self.register(apt.plugins.efficiency.S2Threshold)
        self.register(aptext.acceptance.cut_fv.FiducialVolumeCylinderAccept)
        self.register(aptext.acceptance.efficiency_3f.S1ReconEffNHits)
        self.register(aptext.efficiency_2f.AntiCorrelationEfficiency)
        self.register(aptext.acceptance.efficiency_total.FullEffSS)


@export
class Ar37SR2(ComponentSim):
    norm_type = "on_pdf"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Simulations
        self.register(apt.plugins.common.MonoEnergySpectra)
        self.register(apt.plugins.common.PositionSpectra)
        self.register_all(apt.plugins.er_nestv2)
        self.register_all(apt.plugins.detector)
        self.register(apt.plugins.PositionRecon)
        self.register(apt.plugins.S1)
        self.register(apt.plugins.S2)
        self.register(aptext.field.S1CorrectionWithBias)
        self.register(aptext.field.S2CorrectionWithBias)
        self.register(apt.plugins.cS1)
        self.register(apt.plugins.cS2)
        self.register(aptext.field.S1NHits)

        # Efficiencies
        self.register(apt.plugins.efficiency.S1CutAccept)
        self.register(apt.plugins.efficiency.S2CutAccept)
        self.register(apt.plugins.efficiency.S2Threshold)
        self.register(aptext.acceptance.cut_fv.FiducialVolumeCylinderAccept)
        self.register(aptext.acceptance.efficiency_3f.S1ReconEffNHits)
        self.register(aptext.efficiency_2f.AntiCorrelationEfficiency)
        self.register(aptext.acceptance.efficiency_total.FullEffSS)


@export
class AmBe(ComponentSim):
    norm_type = "on_pdf"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Simulations
        self.register(apt.plugins.er_microphys.Quanta)
        self.register_all(apt.plugins.detector)
        self.register_all(apt.plugins.reconstruction)
        self.register_all(aptext.multiscatter.ms)
        self.register(aptext.field.MSS1NHits)

        # Emission model
        self.register(apt.plugins.nestv2.TotalQuanta)
        self.register(apt.plugins.nestv2.ThomasImelBox)
        self.register(apt.plugins.nestv2.QyNR)
        self.register(apt.plugins.nestv2.LyNR)
        self.register(apt.plugins.nestv2.MeanNphNe)
        self.register(apt.plugins.nestv2.MeanExcitonIon)
        self.register(apt.plugins.nestv2.TrueExcitonIonNR)
        self.register(apt.plugins.nestv2.OmegaNR)
        self.register(apt.plugins.nestv2.TruePhotonElectronNR)

        # Efficiencies
        self.register(apt.plugins.efficiency.S1CutAccept)
        self.register(apt.plugins.efficiency.S2CutAccept)
        self.register(apt.plugins.efficiency.S2Threshold)
        self.register(aptext.acceptance.cut_fv.AmBeFiducialVolumeAccept)
        self.register(aptext.acceptance.efficiency_3f.S1ReconEffNHits)
        self.register(aptext.efficiency_2f.AntiCorrelationEfficiency)
        self.register(aptext.acceptance.efficiency_total.FullEffMS)


@export
class ERBackground(Rn220):
    norm_type = "on_pdf"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register(aptext.acceptance.cut_fv.WIMPFiducialVolumeAccept)
        # Masks for wire region
        self.register(aptext.acceptance.cut_fv.FarTransverseWiresAccept)
        self.register(aptext.acceptance.cut_fv.NearTransverseWiresAccept)


@export
class TritiumBackground(Rn220):
    norm_type = "on_pdf"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register(apt.plugins.FixedEnergySpectra)

        self.register(aptext.acceptance.cut_fv.WIMPFiducialVolumeAccept)
        # Masks for wire region
        self.register(aptext.acceptance.cut_fv.FarTransverseWiresAccept)
        self.register(aptext.acceptance.cut_fv.NearTransverseWiresAccept)


@export
class RGBackground(AmBe):
    norm_type = "on_sim"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register(aptext.acceptance.cut_fv.WIMPFiducialVolumeAccept)
        # Masks for wire region
        self.register(aptext.acceptance.cut_fv.FarTransverseWiresAccept)
        self.register(aptext.acceptance.cut_fv.NearTransverseWiresAccept)


@export
class CEvNSBackground(Rn220):
    norm_type = "on_sim"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register(apt.plugins.FixedEnergySpectra)

        # Overwrite emission model
        self.register(apt.plugins.nestv2.TotalQuanta)
        self.register(apt.plugins.nestv2.ThomasImelBox)
        self.register(apt.plugins.nestv2.QyNR)
        self.register(apt.plugins.nestv2.LyNR)
        self.register(apt.plugins.nestv2.MeanNphNe)
        self.register(apt.plugins.nestv2.MeanExcitonIon)
        self.register(apt.plugins.nestv2.TrueExcitonIonNR)
        self.register(apt.plugins.nestv2.OmegaNR)
        self.register(apt.plugins.nestv2.TruePhotonElectronNR)

        self.register(aptext.acceptance.cut_fv.WIMPFiducialVolumeAccept)
        # Masks for wire region
        self.register(aptext.acceptance.cut_fv.FarTransverseWiresAccept)
        self.register(aptext.acceptance.cut_fv.NearTransverseWiresAccept)


@export
class WIMP(CEvNSBackground):
    norm_type = "on_sim"

    pass


@export
class FlatNR(CEvNSBackground):
    norm_type = "on_pdf"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register(apt.plugins.common.UniformEnergySpectra)
