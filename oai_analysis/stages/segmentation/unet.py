from dataclasses import dataclass
from typing import ClassVar

import itk

from oai_analysis.data import models_dir, models_sha256
from oai_analysis.registry import register
from oai_analysis.stages.segmentation.checks import segmentation_findings
from oai_analysis.stages.segmentation.segmenter import Segmenter3DInPatchClassWise
from oai_analysis.stages.base import CaseArtifacts, CaseInfo, QCFinding, Segmenter


@register
@dataclass
class UNetSegmenter(Segmenter):
    """The 3D U-Net trained on OAI knees, applied in overlapping patches."""

    name: ClassVar[str] = "unet3d_oai"
    performance_params: ClassVar[tuple[str, ...]] = ("batch_size",)
    check_params: ClassVar[tuple[str, ...]] = ("fc_volume_ml", "tc_volume_ml", "max_outside_main_pieces",
                                               "max_uncertainty")
    batch_size: int = 4  # patches per pass through the network
    overlap_size: tuple[int, int, int] = (16, 16, 8)  # voxels shared by neighbouring patches
    # checks, tuned on the benchmark cases and the Dec 18, 2024 triage
    fc_volume_ml: tuple[float, float] = (4.0, 20.0)  # usual range of the femoral cartilage volume
    tc_volume_ml: tuple[float, float] = (1.5, 8.0)  # usual range of the tibial cartilage volume
    max_outside_main_pieces: float = 0.05  # share of the volume outside the largest piece (FC) or two (TC)
    max_uncertainty: float = 0.3  # mean of 1 - |2p - 1| over the cartilage

    def setup(self, device):
        self._segmenter = Segmenter3DInPatchClassWise(mode="pred", config=dict(
            ckpoint_path=str(models_dir() / "segmentation_model.pth.tar"),
            training_config_file=str(models_dir() / "segmentation_train_config.pth.tar"),
            device=device,
            batch_size=self.batch_size,
            overlap_size=self.overlap_size,
            output_prob=True,
            output_itk=True,
        ))
        self._segmenter.pred_setup()

    def weights(self):
        return {"oai-analysis-models.tar.gz": f"sha256:{models_sha256}"}

    def run(self, image: itk.Image, case: CaseInfo | None = None) -> dict[str, itk.Image]:
        FC_prob, TC_prob = self._segmenter.segment(image, if_output_prob_map=True, if_output_itk=True)
        return {"FC": FC_prob, "TC": TC_prob}

    def check(self, case: CaseArtifacts) -> list[QCFinding]:
        return segmentation_findings(case, self.fc_volume_ml, self.tc_volume_ml, self.max_outside_main_pieces,
                                     self.max_uncertainty)
