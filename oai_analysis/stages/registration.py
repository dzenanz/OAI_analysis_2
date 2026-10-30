from dataclasses import dataclass
from typing import ClassVar

import icon_registration.itk_wrapper as itk_wrapper
import itk
from unigradicon import get_unigradicon

from oai_analysis import data, qc
from oai_analysis.registration import gradicon_knee_model
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
        return registration_findings(case, self.max_folding, self.min_similarity)


@register
@dataclass
class GradICONKneeRegisterer(Registerer):
    """
    GradICON trained on OAI knees (icon_registration's OAI_knees_gradICON_model), the registration the pipeline
    used before uniGradICON. finetune_steps: 0 uses the network as trained, as that pipeline did.
    """

    name: ClassVar[str] = "gradicon_knee"
    check_params: ClassVar[tuple[str, ...]] = ("max_folding", "min_similarity")
    finetune_steps: int = 0
    max_folding: float = 0.01
    min_similarity: float = 0.75

    def setup(self, device):
        # icon_registration places the network on its own configured device
        self._model = gradicon_knee_model()

    def weights(self):
        return {"gradicon_knee/knee_model_weights_step_2.trch": f"sha256:{data.gradicon_knee_sha256}"}

    def run(self, moving: itk.Image, fixed: itk.Image) -> RegistrationResult:
        phi_AB, phi_BA = itk_wrapper.register_pair(self._model, moving.astype(itk.D), fixed.astype(itk.D),
                                                   finetune_steps=self.finetune_steps or None)
        return RegistrationResult(phi_AB, phi_BA)

    def check(self, case: CaseArtifacts) -> list[QCFinding]:
        return registration_findings(case, self.max_folding, self.min_similarity)


def registration_findings(case: CaseArtifacts, max_folding, min_similarity) -> list[QCFinding]:
    """The checks of any registerer: the transform does not fold space, and the warped image matches the atlas."""
    folding = qc.folding_fraction(case.registration.phi_AB)
    similarity = qc.similarity_to_atlas(case.image, case.atlas.image, case.registration.phi_AB)
    return [
        qc.finding(folding > max_folding, "register.folding",
                   f"the registration folds space in {qc.percent(folding)} of its field "
                   f"(review above {qc.percent(max_folding)})", folding),
        qc.finding(similarity < min_similarity, "register.similarity",
                   f"after registration, the image correlates with the atlas at {similarity:.3f} "
                   f"(review below {min_similarity})", similarity),
    ]
