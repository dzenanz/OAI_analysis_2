import copy
from dataclasses import dataclass
from typing import ClassVar

import itk
import torch

from oai_analysis.data import models_dir, models_sha256
from oai_analysis.registry import register
from oai_analysis.stages.segmentation.segmenter import Segmenter3DInPatchClassWise
from oai_analysis.stages.base import Segmenter


@register
@dataclass
class UNetSegmenter(Segmenter):
    """The 3D U-Net trained on OAI knees, applied in overlapping patches."""

    name: ClassVar[str] = "unet3d_oai"
    performance_params: ClassVar[tuple[str, ...]] = ("batch_size",)
    batch_size: int = 4  # patches per pass through the network
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

    def weights(self):
        return {"oai-analysis-models.tar.gz": f"sha256:{models_sha256}"}

    def run(self, image: itk.Image) -> dict[str, itk.Image]:
        FC_prob, TC_prob = self._segmenter.segment(image, if_output_prob_map=True, if_output_itk=True)
        return {"FC": FC_prob, "TC": TC_prob}

    def run_batch(self, images: list[itk.Image]) -> list[dict[str, itk.Image]]:
        """The patches of all images go through the network batch_size at a time, as if from one image."""
        segmenter = self._segmenter
        partitions, tiles = [], []
        for image in images:
            partition = copy.copy(segmenter.partition)  # it remembers the geometry of the image it split
            tiles.append(partition({"image": image, "name": ""})["image"])
            partitions.append(partition)
        counts = [len(t) for t in tiles]
        results, outputs = [], []  # outputs: network outputs of the patches of images not assembled yet
        with torch.no_grad():
            for chunk in torch.split(torch.cat(tiles), self.batch_size):
                outputs.append(segmenter.model(chunk.to(segmenter.device)).cpu())
                # assemble each image as soon as all its patches are done, to keep only a few in memory
                while len(results) < len(images) and sum(len(o) for o in outputs) >= counts[len(results)]:
                    predictions = torch.cat(outputs)
                    count = counts[len(results)]
                    outputs = [predictions[count:]]
                    probability = torch.sigmoid(predictions[:count])
                    partition = partitions[len(results)]
                    results.append({s: partition.assemble(probability[:, i], if_itk=True, crop_size=self.overlap_size)
                                    for i, s in enumerate(("FC", "TC"))})
        return results
