from dataclasses import dataclass
from typing import ClassVar

import itk
import numpy as np
import pytest
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

from oai_analysis import OAPipeline
from oai_analysis.algorithms import mesh_processing as mp
from oai_analysis.atlas import Atlas
from oai_analysis.stages import (
    AtlasMapper,
    BiomarkerExtractor,
    CaseInfo,
    CuberilleMeshGenerator,
    MeshGenerator,
    MeshResult,
    OAIPreprocessor,
    Preprocessor,
    RegistrationResult,
    Registerer,
    Segmenter,
    ThicknessEstimator,
    ThicknessMap2D,
)
from oai_analysis.stages.base import CaseArtifacts


def ball_image(radius_mm=6.0, size=32, spacing=0.5):
    """A float image with value 1 inside a ball centred in the image and 0 outside."""
    z, y, x = np.mgrid[:size, :size, :size] * spacing
    centre = (size - 1) * spacing / 2
    inside = (x - centre) ** 2 + (y - centre) ** 2 + (z - centre) ** 2 < radius_mm ** 2
    image = itk.image_from_array(inside.astype(np.float32))
    image.SetSpacing([spacing] * 3)
    return image


def with_thickness(mesh, value):
    """Copy of the mesh with a constant per-vertex thickness array."""
    result = vtk.vtkPolyData()
    result.DeepCopy(mesh)
    thickness = numpy_to_vtk(np.full(mesh.GetNumberOfPoints(), value, dtype=np.float32), deep=True)
    thickness.SetName("vertex_thickness")
    result.GetPointData().AddArray(thickness)
    return result


def test_case_info():
    case = CaseInfo("some/dir/image.nii.gz", "left")
    assert case.case_id == "image.nii.gz"
    with pytest.raises(ValueError):
        CaseInfo("image.nii.gz", "Left")


@pytest.mark.parametrize("laterality", ["left", "right"])
def test_preprocessor(laterality):
    image = ball_image()
    image.SetOrigin([1.0, 2.0, 3.0])
    preprocessor = OAIPreprocessor()
    assert preprocessor.result_params() == {"window_min_percentile": 0.1, "window_max_percentile": 99.9}

    result = preprocessor.run(image, CaseInfo("x.nii.gz", laterality))

    asl = itk.SpatialOrientationEnums.ValidCoordinateOrientations_ITK_COORDINATE_ORIENTATION_ASL
    expected = itk.orient_image_filter(ball_image(), use_image_direction=True, desired_coordinate_orientation=asl)
    np.testing.assert_array_equal(itk.array_from_matrix(result.GetDirection()),
                                  itk.array_from_matrix(expected.GetDirection()))
    values = itk.array_view_from_image(result)
    assert values.min() == 0.0 and values.max() == 1.0


def test_cuberille_mesh_generator():
    mesh = CuberilleMeshGenerator().run(ball_image(radius_mm=6.0))
    assert mesh.GetNumberOfPoints() > 100
    points = vtk_to_numpy(mesh.GetPoints().GetData())
    radii = np.linalg.norm(points - points.mean(axis=0), axis=1)
    np.testing.assert_allclose(radii, 6.0, atol=0.5)


def test_transform_mesh_keeps_thickness():
    mesh = with_thickness(CuberilleMeshGenerator().run(ball_image()), 2.5)
    translation = itk.TranslationTransform[itk.D, 3].New()
    translation.SetOffset([1.0, -2.0, 3.0])

    moved = mp.transform_mesh(mesh, translation)

    np.testing.assert_allclose(vtk_to_numpy(moved.GetPoints().GetData()),
                               vtk_to_numpy(mesh.GetPoints().GetData()) + [1.0, -2.0, 3.0], atol=1e-4)
    np.testing.assert_array_equal(vtk_to_numpy(moved.GetPointData().GetArray(0)), 2.5)


def test_thickness_map_2d():
    atlas = Atlas.default()
    case = CaseArtifacts(case=CaseInfo("x.nii.gz", "left"), atlas=atlas)
    for s in ("FC", "TC"):
        case.atlas_mapped[s] = MeshResult(with_thickness(atlas.inner_mesh[s], 2.0))

    outputs = ThicknessMap2D(grid_resolution=50).run(case)

    assert sorted(outputs) == ["FC_thickness.npy", "FC_thickness.png", "TC_thickness.npy", "TC_thickness.png"]
    grid = outputs["FC_thickness.npy"]
    assert grid.shape[0] >= 50 and np.isnan(grid).any()
    np.testing.assert_allclose(grid[~np.isnan(grid)], 2.0, rtol=1e-6)
    assert outputs["FC_thickness.png"].startswith(b"\x89PNG")


# Lightweight stand-ins for the default stages, to test how the pipeline wires stages together.

@dataclass
class FakePreprocessor(Preprocessor):
    name: ClassVar[str] = "fake"
    setup_calls: int = 0

    def setup(self, device):
        self.setup_calls += 1

    def run(self, image, case):
        return image


@dataclass
class FakeSegmenter(Segmenter):
    name: ClassVar[str] = "fake"

    def run(self, image):
        return {"FC": image, "TC": image}


@dataclass
class IdentityRegisterer(Registerer):
    name: ClassVar[str] = "identity"

    def run(self, moving, fixed):
        identity = itk.IdentityTransform[itk.D, 3].New()
        return RegistrationResult(identity, identity)


@dataclass
class FakeMeshGenerator(MeshGenerator):
    name: ClassVar[str] = "fake"

    def run(self, probability):
        sphere = vtk.vtkSphereSource()
        sphere.Update()
        mesh = sphere.GetOutput()
        mesh.GetPointData().RemoveArray("Normals")  # the thickness has to be the first point data array
        return mesh


@dataclass
class ConstantThickness(ThicknessEstimator):
    name: ClassVar[str] = "constant"
    thickness: float = 2.0

    def run(self, mesh, probability, registration, atlas, structure):
        return MeshResult(with_thickness(mesh, self.thickness), intermediates={"inner": mesh})


@dataclass
class IdentityMapper(AtlasMapper):
    name: ClassVar[str] = "identity"

    def run(self, thickness_mesh, registration, atlas, structure):
        return MeshResult(thickness_mesh)


@dataclass
class MeanThickness(BiomarkerExtractor):
    name: ClassVar[str] = "mean_thickness"

    def run(self, case):
        return {"mean_thickness.npy": np.array([case.atlas_thickness(s).mean() for s in ("FC", "TC")])}


@pytest.fixture
def fake_pipeline():
    tiny_atlas = Atlas(name="tiny", image=ball_image(), inner_mesh={}, embedding_2d={})
    return OAPipeline(preprocessor=FakePreprocessor(), segmenter=FakeSegmenter(), registerer=IdentityRegisterer(),
                      mesh_generator=FakeMeshGenerator(), thickness_estimator=ConstantThickness(),
                      atlas_mapper=IdentityMapper(), biomarker_extractors=[MeanThickness()],
                      atlas=tiny_atlas, device="cpu")


def test_pipeline_flat_layout(fake_pipeline, tmp_path):
    input_path = tmp_path / "input.nrrd"
    itk.imwrite(ball_image(), str(input_path))

    for i in range(2):
        result = fake_pipeline.run_case(CaseInfo(input_path, "left"), tmp_path / f"case{i}")
    assert fake_pipeline.preprocessor.setup_calls == 1  # models are loaded once for all cases

    np.testing.assert_allclose(result.biomarkers["mean_thickness.npy"], [2.0, 2.0])
    written = sorted(p.name for p in (tmp_path / "case1").iterdir())
    assert written == sorted([
        "in_image.nrrd", "resampling.tfm", "modelling.tfm", "mean_thickness.npy",
        *(f"{s}_{name}" for s in ("FC", "TC") for name in
          ("prob.nrrd", "mesh_patient.vtk", "outer.vtk", "inner.vtk", "mapped_mesh.vtk")),
    ])


def test_pipeline_without_intermediates(fake_pipeline, tmp_path):
    input_path = tmp_path / "input.nrrd"
    itk.imwrite(ball_image(), str(input_path))

    fake_pipeline.run_case(CaseInfo(input_path, "right"), tmp_path / "out", keep_intermediate_outputs=False)

    assert [p.name for p in (tmp_path / "out").iterdir()] == ["mean_thickness.npy"]
