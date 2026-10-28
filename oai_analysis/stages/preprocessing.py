from dataclasses import dataclass
from typing import ClassVar

import itk
import numpy as np

from oai_analysis.registry import register
from oai_analysis.stages.base import CaseInfo, Preprocessor


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
    window_min_percentile: float = 0.1
    window_max_percentile: float = 99.9

    def run(self, image: itk.Image, case: CaseInfo) -> itk.Image:
        image = into_canonical_orientation(image, case.laterality == "right")  # simplifies mesh processing
        return window_intensities(image, self.window_min_percentile, self.window_max_percentile)
