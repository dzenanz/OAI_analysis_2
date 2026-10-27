from dataclasses import dataclass
from typing import ClassVar

import icon_registration.itk_wrapper as itk_wrapper
import itk
from unigradicon import get_unigradicon

from oai_analysis import data
from oai_analysis.stages.base import RegistrationResult, Registerer


@dataclass
class UniGradICONRegisterer(Registerer):
    """uniGradICON, fine-tuned on each image pair."""

    name: ClassVar[str] = "unigradicon"
    finetune_steps: int = 2

    def setup(self, device):
        # icon_registration places the network on its own configured device
        self._model = get_unigradicon(weights_location=str(data.unigradicon_weights()))

    def run(self, moving: itk.Image, fixed: itk.Image) -> RegistrationResult:
        phi_AB, phi_BA = itk_wrapper.register_pair(self._model, moving.astype(itk.D), fixed.astype(itk.D),
                                                   finetune_steps=self.finetune_steps)
        return RegistrationResult(phi_AB, phi_BA)
