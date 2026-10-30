from dataclasses import dataclass
from typing import ClassVar

import itk

from oai_analysis.registry import register
from oai_analysis.stages.base import CaseArtifacts, CaseInfo, QCFinding, Segmenter
from oai_analysis.stages.segmentation.checks import segmentation_findings


@register
@dataclass
class PrecomputedSegmenter(Segmenter):
    """
    Probability maps made earlier, e.g. by an earlier run or another program, read from files.

    path is a template for each map's file. These fill it in: {structure} (FC or TC), {case_id}, {laterality},
    {LATERALITY} (upper case) and the columns of a project's cases.csv, e.g. {patient_id} and {month}. The maps are
    resampled onto the preprocessed image, so they have to be in its physical space, as a run's FC_prob.nrrd is.
    The files are not part of the fingerprint: after they change, redo the step with `run --from segment`.
    """

    name: ClassVar[str] = "precomputed"
    check_params: ClassVar[tuple[str, ...]] = ("fc_volume_ml", "tc_volume_ml", "max_outside_main_pieces",
                                               "max_uncertainty")
    path: str = "{case_id}/{structure}_prob.nrrd"  # template of each map's file, see above
    fc_volume_ml: tuple[float, float] = (4.0, 20.0)  # usual range of the femoral cartilage volume
    tc_volume_ml: tuple[float, float] = (1.5, 8.0)  # usual range of the tibial cartilage volume
    max_outside_main_pieces: float = 0.05  # share of the volume outside the largest piece (FC) or two (TC)
    max_uncertainty: float = 0.3  # mean of 1 - |2p - 1| over the cartilage

    def files(self, case: CaseInfo) -> dict[str, str]:
        """The file of each structure's map."""
        fields = {**case.details, "case_id": case.case_id, "laterality": case.laterality,
                  "LATERALITY": case.laterality.upper()}
        return {s: self.path.format(structure=s, **fields) for s in ("FC", "TC")}

    def run(self, image: itk.Image, case: CaseInfo | None = None) -> dict[str, itk.Image]:
        if case is None:
            raise ValueError("PrecomputedSegmenter needs the case, to know which files to read")
        return {structure: itk.resample_image_filter(itk.imread(path, itk.F), use_reference_image=True,
                                                     reference_image=image)
                for structure, path in self.files(case).items()}

    def check(self, case: CaseArtifacts) -> list[QCFinding]:
        return segmentation_findings(case, self.fc_volume_ml, self.tc_volume_ml, self.max_outside_main_pieces,
                                     self.max_uncertainty)
