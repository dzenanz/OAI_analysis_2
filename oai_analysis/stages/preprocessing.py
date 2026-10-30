from dataclasses import dataclass
from typing import ClassVar

import itk
import numpy as np

from oai_analysis import qc
from oai_analysis.registry import register
from oai_analysis.stages.base import CaseArtifacts, CaseInfo, Preprocessor, QCFinding


def into_canonical_orientation(image, flip_left_right):
    """
    Reorient the given image into the canonical orientation.

    :param image: input image
    :param flip_left_right: if True, flips the image left-right
    :return: reoriented image
    """
    dicom_pir = itk.SpatialOrientationEnums.ValidCoordinateOrientations_ITK_COORDINATE_ORIENTATION_ASL
    if flip_left_right:
        image_dir = itk.array_from_matrix(image.GetDirection())
        image_dir[0, :] *= -1  # flip left-right
        image.SetDirection(itk.matrix_from_array(image_dir))
    oriented_image = itk.orient_image_filter(
        image,
        use_image_direction=True,
        desired_coordinate_orientation=dicom_pir,  # atlas' orientation
    )
    return oriented_image


def window_intensities(image, window_min_percentile=0.1, window_max_percentile=99.9, output_min=0.0, output_max=1.0):
    image_array = itk.GetArrayViewFromImage(image)
    win_min = float(np.percentile(image_array, window_min_percentile))
    win_max = float(np.percentile(image_array, window_max_percentile))
    result = itk.intensity_windowing_image_filter(image,
                                                  window_minimum=win_min,
                                                  window_maximum=win_max,
                                                  output_minimum=output_min,
                                                  output_maximum=output_max)
    return result


@register
@dataclass
class OAIPreprocessor(Preprocessor):
    """Mirror right knees to look like left ones, reorient to the atlas orientation, and window the intensities."""

    name: ClassVar[str] = "oai_default"
    check_params: ClassVar[tuple[str, ...]] = ("expected_spacing_mm", "expected_size")
    window_min_percentile: float = 0.1  # intensities at or below this percentile become 0
    window_max_percentile: float = 99.9  # intensities at or above this percentile become 1
    expected_spacing_mm: tuple[float, float, float] = (0.365, 0.365, 0.7)  # OAI sagittal 3D DESS; review if 10% off
    expected_size: tuple[int, int, int] = (384, 384, 160)  # voxels, OAI sagittal 3D DESS; review if different

    def run(self, image: itk.Image, case: CaseInfo) -> itk.Image:
        image = into_canonical_orientation(image, case.laterality == "right")  # simplifies mesh processing
        return window_intensities(image, self.window_min_percentile, self.window_max_percentile)

    def check(self, case: CaseArtifacts) -> list[QCFinding]:
        image = case.image
        findings = []
        axes, atlas_axes = qc.axis_directions(image), qc.axis_directions(case.atlas.image)
        if axes != atlas_axes:
            findings.append(QCFinding("fail", "preprocess.orientation",
                                      f"after reorientation, the image axes point along {axes}, but the atlas "
                                      f"axes along {atlas_axes}"))
        spacing = tuple(round(s, 3) for s in image.GetSpacing())
        unusual = any(abs(s - e) > 0.1 * e for s, e in zip(spacing, self.expected_spacing_mm))
        findings.append(qc.finding(unusual, "preprocess.spacing",
                                   f"the voxel spacing is {' x '.join(map(str, spacing))} mm (expected: "
                                   f"{' x '.join(map(str, self.expected_spacing_mm))})", max(spacing)))
        size = tuple(int(s) for s in image.GetLargestPossibleRegion().GetSize())
        findings.append(qc.finding(size != tuple(self.expected_size), "preprocess.size",
                                   f"the image has {' x '.join(map(str, size))} voxels (expected: "
                                   f"{' x '.join(map(str, self.expected_size))})", np.prod(size)))
        median = float(np.median(itk.array_view_from_image(image)))
        findings.append(QCFinding("info", "preprocess.median_intensity",
                                  f"the median intensity after windowing is {median:.3f}", round(median, 6)))
        return findings
