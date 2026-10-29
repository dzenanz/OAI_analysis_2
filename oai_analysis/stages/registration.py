from dataclasses import dataclass
from typing import ClassVar

import icon_registration.itk_wrapper as itk_wrapper
import itk
from unigradicon import get_unigradicon

from oai_analysis import data, qc
from oai_analysis.registry import register
from oai_analysis.stages.base import CaseArtifacts, QCFinding, RegistrationResult, Registerer


@register
@dataclass
class UniGradICONRegisterer(Registerer):
    """uniGradICON, fine-tuned on each image pair."""

    name: ClassVar[str] = "unigradicon"
    check_params: ClassVar[tuple[str, ...]] = ("max_folding", "min_similarity")
    finetune_steps: int = 2
    max_folding: float = 0.01  # fraction of the displacement field with a Jacobian determinant <= 0
    min_similarity: float = 0.75  # correlation of the warped image with the atlas image

    def setup(self, device):
        # icon_registration places the network on its own configured device
        self._model = get_unigradicon(weights_location=str(data.unigradicon_weights()))

    def weights(self):
        return {"unigradicon1.0/Step_2_final.trch": f"sha256:{data.unigradicon_sha256}"}

    def run(self, moving: itk.Image, fixed: itk.Image) -> RegistrationResult:
        phi_AB, phi_BA = itk_wrapper.register_pair(self._model, moving.astype(itk.D), fixed.astype(itk.D),
                                                   finetune_steps=self.finetune_steps)
        return RegistrationResult(phi_AB, phi_BA)

    def check(self, case: CaseArtifacts) -> list[QCFinding]:
        folding = qc.folding_fraction(case.registration.phi_AB)
        similarity = qc.similarity_to_atlas(case.image, case.atlas.image, case.registration.phi_AB)
        return [
            qc.finding(folding > self.max_folding, "register.folding",
                       f"the registration folds space in {qc.percent(folding)} of its field "
                       f"(review above {qc.percent(self.max_folding)})", folding),
            qc.finding(similarity < self.min_similarity, "register.similarity",
                       f"after registration, the image correlates with the atlas at {similarity:.3f} "
                       f"(review below {self.min_similarity})", similarity),
        ]
