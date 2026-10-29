from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk

from oai_analysis import qc
from oai_analysis.algorithms import mesh_processing as mp
from oai_analysis.algorithms.thickness_computation import DistanceMapMethod, compute_thickness
from oai_analysis.registry import register
from oai_analysis.stages.base import CaseArtifacts, MeshResult, QCFinding, ThicknessEstimator


def thickness_findings(case, max_thickness_mm, max_too_thick) -> list[QCFinding]:
    """Flag a structure thicker than max_thickness_mm on more than max_too_thick of its vertices."""
    findings = []
    for structure, result in case.thickness.items():
        thickness = qc.point_values(result.mesh)
        too_thick = float(np.mean(thickness > max_thickness_mm)) if len(thickness) else 0.0
        findings.append(qc.finding(too_thick > max_too_thick, "thickness.too_thick",
                                   f"the {qc.STRUCTURE_NAMES[structure]} is thicker than {max_thickness_mm} mm "
                                   f"on {qc.percent(too_thick)} of its surface "
                                   f"(review above {qc.percent(max_too_thick)})", too_thick))
    return findings


@register
@dataclass
class MeshSplittingThickness(ThicknessEstimator):
    """
    Split the surface into inner/outer in atlas space; measure inner-outer distance in patient space.

    back_mapping chooses how the split gets back to patient space:
    "cell_ids" cuts the halves out of the original patient mesh, by the ids of the cells in each half;
    "transform" (version 1) transforms the halves of the smoothed atlas-space mesh back.
    """

    name: ClassVar[str] = "mesh_splitting"
    version: ClassVar[str] = "2"
    check_params: ClassVar[tuple[str, ...]] = ("max_thickness_mm", "max_too_thick")
    smoothing_iterations: int = 150
    island_distance_threshold_mm: float = 10.0
    back_mapping: str = "cell_ids"
    max_thickness_mm: float = 6.0
    max_too_thick: float = 0.01  # fraction of the vertices thicker than max_thickness_mm

    def __post_init__(self):
        if self.back_mapping not in ("cell_ids", "transform"):
            raise ValueError(f"back_mapping must be 'cell_ids' or 'transform', not {self.back_mapping!r}")

    def run(self, mesh, probability, registration, atlas, structure):
        mesh_atlas = mp.transform_mesh(mesh, registration.phi_BA)  # keeps the order of the cells
        split_args = dict(atlas_mesh=atlas.inner_mesh[structure], mesh_type=structure,
                          num_iterations=self.smoothing_iterations,
                          distance_threshold_mm=self.island_distance_threshold_mm)
        if self.back_mapping == "cell_ids":
            inner_ids, outer_ids = mp.get_split_cell_ids(mesh_atlas, **split_args)
            inner = mp.get_vtk_sub_mesh(mesh, inner_ids)
            outer = mp.get_vtk_sub_mesh(mesh, outer_ids)
        else:
            inner_atlas, outer_atlas = mp.get_split_mesh(mesh_atlas, **split_args)
            inner = mp.transform_mesh(inner_atlas, registration.phi_AB)
            outer = mp.transform_mesh(outer_atlas, registration.phi_AB)
        inner_with_thickness, outer_with_thickness = mp.get_distance(inner, outer)
        return MeshResult(outer_with_thickness, intermediates={"inner": inner_with_thickness})

    def check(self, case: CaseArtifacts) -> list[QCFinding]:
        return thickness_findings(case, self.max_thickness_mm, self.max_too_thick)


def sample_distance_from_image(thickness_image, mesh):
    """Return a shallow copy of the mesh with a vertex_thickness array sampled from the image."""
    num_points = mesh.GetNumberOfPoints()
    thickness = np.zeros(num_points, dtype=np.float32)
    for i in range(num_points):
        point = mesh.GetPoint(i)
        image_index = thickness_image.TransformPhysicalPointToIndex(point)
        thickness[i] = thickness_image.GetPixel(image_index)

    vtk_array = numpy_to_vtk(thickness, deep=True, array_type=vtk.VTK_FLOAT)
    vtk_array.SetName("vertex_thickness")
    result = vtk.vtkPolyData()
    result.ShallowCopy(mesh)
    result.GetPointData().AddArray(vtk_array)
    return result


@register
@dataclass
class DistanceTransformThickness(ThicknessEstimator):
    """Thickness from a distance transform of the segmentation mask, sampled at the mesh vertices."""

    name: ClassVar[str] = "distance_transform"
    check_params: ClassVar[tuple[str, ...]] = ("max_thickness_mm", "max_too_thick")
    distance_map_method: str = DistanceMapMethod.parabolic_morphology.value
    max_thickness_mm: float = 6.0
    max_too_thick: float = 0.01

    def run(self, mesh, probability, registration, atlas, structure):
        thickness_image, _, _ = compute_thickness(probability, method=DistanceMapMethod(self.distance_map_method))
        return MeshResult(sample_distance_from_image(thickness_image, mesh),
                          intermediates={"thickness_image": thickness_image})

    def check(self, case: CaseArtifacts) -> list[QCFinding]:
        return thickness_findings(case, self.max_thickness_mm, self.max_too_thick)
