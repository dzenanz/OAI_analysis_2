from dataclasses import dataclass
from typing import ClassVar

import itk

from oai_analysis.data import models_dir
from oai_analysis.segmentation.segmenter import Segmenter3DInPatchClassWise
from oai_analysis.stages.base import Segmenter


@dataclass
class UNetSegmenter(Segmenter):
    """The 3D U-Net trained on OAI knees, applied in overlapping patches."""

    name: ClassVar[str] = "unet3d_oai"
    batch_size: int = 4
    overlap_size: tuple[int, int, int] = (16, 16, 8)

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

    def run(self, image: itk.Image) -> dict[str, itk.Image]:
        FC_prob, TC_prob = self._segmenter.segment(image, if_output_prob_map=True, if_output_itk=True)
        return {"FC": FC_prob, "TC": TC_prob}
