from dataclasses import dataclass
from typing import ClassVar

import icon_registration.itk_wrapper as itk_wrapper
import itk
import numpy as np
import torch
import torch.nn.functional as F
from icon_registration import config as icon_config
from unigradicon import get_unigradicon

from oai_analysis import data
from oai_analysis.registry import register
from oai_analysis.stages.base import RegistrationResult, Registerer


@register
@dataclass
class UniGradICONRegisterer(Registerer):
    """uniGradICON, fine-tuned on each image pair (finetune_steps: 0 uses the network as trained)."""

    name: ClassVar[str] = "unigradicon"
    performance_params: ClassVar[tuple[str, ...]] = ("batch_size",)
    finetune_steps: int = 2
    batch_size: int = 4  # image pairs per pass through the network, without fine-tuning

    def setup(self, device):
        # icon_registration places the network on its own configured device
        self._model = get_unigradicon(weights_location=str(data.unigradicon_weights()))

    def weights(self):
        return {"unigradicon1.0/Step_2_final.trch": f"sha256:{data.unigradicon_sha256}"}

    def run(self, moving: itk.Image, fixed: itk.Image) -> RegistrationResult:
        phi_AB, phi_BA = itk_wrapper.register_pair(self._model, moving.astype(itk.D), fixed.astype(itk.D),
                                                   finetune_steps=self.finetune_steps or None)
        return RegistrationResult(phi_AB, phi_BA)

    def run_batch(self, movings: list[itk.Image], fixed: itk.Image) -> list[RegistrationResult]:
        """
        Without fine-tuning, batch_size pairs go through the network together. Fine-tuning tunes a copy of the
        network for each pair, so those pairs are registered one by one.
        """
        if self.finetune_steps:
            return [self.run(moving, fixed) for moving in movings]
        model = self._model.to(icon_config.device)
        identity = model.identity_map
        fixed = fixed.astype(itk.D)
        image_B = _network_input(fixed, identity.shape[2:])
        results = []
        with torch.no_grad():
            for start in range(0, len(movings), self.batch_size):
                chunk = [moving.astype(itk.D) for moving in movings[start:start + self.batch_size]]
                # what itk_wrapper.register_pair() does for one pair, without computing the loss
                image_A = torch.cat([_network_input(moving, identity.shape[2:]) for moving in chunk])
                images_B = image_B.expand(len(chunk), *image_B.shape[1:])
                identities = identity.expand(len(chunk), *identity.shape[1:])
                phi_AB = model.regis_net(image_A, images_B)(identities)
                phi_BA = model.regis_net(images_B, image_A)(identities)
                for i, moving in enumerate(chunk):
                    results.append(RegistrationResult(
                        itk_wrapper.create_itk_transform(phi_AB[i:i + 1], identity, moving, fixed),
                        itk_wrapper.create_itk_transform(phi_BA[i:i + 1], identity, fixed, moving)))
        return results


def _network_input(image, shape):
    """The image as a [1, 1, *shape] tensor on the network's device, as itk_wrapper.register_pair() makes it."""
    tensor = torch.Tensor(np.array(image)).to(icon_config.device)[None, None]
    return F.interpolate(tensor, size=shape, mode="trilinear", align_corners=False)
