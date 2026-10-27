from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk

from oai_analysis import mesh_processing as mp
from oai_analysis.stages.base import MeshResult, ThicknessEstimator
from oai_analysis.thickness_computation import DistanceMapMethod, compute_thickness


@dataclass
class MeshSplittingThickness(ThicknessEstimator):
    """Split the surface into inner/outer in atlas space; measure inner-outer distance in patient space."""

    name: ClassVar[str] = "mesh_splitting"
    smoothing_iterations: int = 150
    island_distance_threshold_mm: float = 10.0

    def run(self, mesh, probability, registration, atlas, structure):
        mesh_atlas = mp.transform_mesh(mesh, registration.phi_BA)
        inner_atlas, outer_atlas = mp.get_split_mesh(
            mesh_atlas, atlas.inner_mesh[structure], mesh_type=structure,
            num_iterations=self.smoothing_iterations,
            distance_threshold_mm=self.island_distance_threshold_mm,
        )
        inner = mp.transform_mesh(inner_atlas, registration.phi_AB)
        outer = mp.transform_mesh(outer_atlas, registration.phi_AB)
        inner_with_thickness, outer_with_thickness = mp.get_distance(inner, outer)
        return MeshResult(outer_with_thickness, intermediates={"inner": inner_with_thickness})


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


@dataclass
class DistanceTransformThickness(ThicknessEstimator):
    """Thickness from a distance transform of the segmentation mask, sampled at the mesh vertices."""

    name: ClassVar[str] = "distance_transform"
    distance_map_method: str = DistanceMapMethod.parabolic_morphology.value

    def run(self, mesh, probability, registration, atlas, structure):
        thickness_image, _, _ = compute_thickness(probability, method=DistanceMapMethod(self.distance_map_method))
        return MeshResult(sample_distance_from_image(thickness_image, mesh),
                          intermediates={"thickness_image": thickness_image})
