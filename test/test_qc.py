import itk
import numpy as np
import pytest

from oai_analysis import qc
from oai_analysis.stages import CaseArtifacts, CaseInfo, OAIPreprocessor, UNetSegmenter
from oai_analysis.atlas import Atlas
from test.test_stages import ball_image


def two_balls():
    """A probability map with a big ball (radius 6 mm) and a small one (radius 3 mm), 0.5 mm voxels."""
    z, y, x = np.mgrid[:48, :32, :32] * 0.5
    big = (x - 8) ** 2 + (y - 8) ** 2 + (z - 8) ** 2 < 6 ** 2
    small = (x - 8) ** 2 + (y - 8) ** 2 + (z - 20) ** 2 < 3 ** 2
    image = itk.image_from_array((big | small).astype(np.float32))
    image.SetSpacing([0.5] * 3)
    return image, big.sum(), small.sum()


def test_segmentation_measures():
    image, big, small = two_balls()
    femoral = qc.segmentation_measures(image, "FC")
    assert femoral["volume_ml"] == pytest.approx((big + small) * 0.125 / 1000)
    assert femoral["outside_main_pieces"] == pytest.approx(small / (big + small))
    assert femoral["uncertainty"] == 0  # probabilities of exactly 1
    assert qc.segmentation_measures(image, "TC")["outside_main_pieces"] == 0  # two pieces are expected


def displacement_field(u):
    """A displacement field transform from an array of displacements (z, y, x, component)."""
    field = itk.image_from_array(u.astype(np.float64), is_vector=True)
    field.SetSpacing([2.0] * 3)
    transform = itk.DisplacementFieldTransform[itk.D, 3].New()
    transform.SetDisplacementField(field)
    return transform, field


def test_folding():
    z, y, x = np.mgrid[:8, :8, :8] * 2.0
    stretch = np.stack([0.5 * x, 0 * y, 0 * z], axis=-1)  # stretches x by 1.5
    _, field = displacement_field(stretch)
    np.testing.assert_allclose(qc.jacobian_determinant(field)[2:-2, 2:-2, 2:-2], 1.5)
    flip = np.stack([-3 * x, 0 * y, 0 * z], axis=-1)  # turns x around: folds everywhere
    transform, _ = displacement_field(flip)
    composite = itk.CompositeTransform[itk.D, 3].New()
    composite.AddTransform(itk.AffineTransform[itk.D, 3].New())
    composite.AddTransform(transform)
    assert qc.folding_fraction(composite) == 1.0
    assert qc.folding_fraction(itk.CompositeTransform[itk.D, 3].New()) == 0.0  # no displacement field


def test_axis_directions():
    image = ball_image()
    assert qc.axis_directions(image) == (1, 2, 3)
    image.SetDirection(itk.matrix_from_array(np.array([[0.0, 0, -1], [1, 0, 0], [0, 1, 0]])))
    assert qc.axis_directions(image) == (2, 3, -1)


def test_segmenter_check():
    image, _, _ = two_balls()
    empty = itk.image_from_array(np.zeros_like(itk.array_from_image(image)))
    empty.CopyInformation(image)
    case = CaseArtifacts(case=CaseInfo("knee.nrrd", "left"), atlas=None, probability={"FC": image, "TC": empty})
    findings = {(f.code, f.severity) for f in UNetSegmenter(fc_volume_ml=(4, 20)).check(case)}
    assert findings == {("segment.volume", "review"),  # 1 ml of femoral cartilage
                        ("segment.pieces", "review"), ("segment.uncertainty", "info"),
                        ("segment.empty", "fail")}  # no tibial cartilage


def test_preprocessor_check():
    image = ball_image()
    atlas = Atlas(name="tiny", image=ball_image(), inner_mesh={}, embedding_2d={})
    case = CaseArtifacts(case=CaseInfo("knee.nrrd", "left"), atlas=atlas, image=image)
    findings = {f.code: f.severity for f in OAIPreprocessor().check(case)}
    assert findings == {"preprocess.spacing": "review", "preprocess.size": "review",
                        "preprocess.median_intensity": "info"}
    image.SetDirection(itk.matrix_from_array(np.array([[0.0, 0, -1], [1, 0, 0], [0, 1, 0]])))
    assert {f.code: f.severity for f in OAIPreprocessor().check(case)}["preprocess.orientation"] == "fail"
