from dataclasses import dataclass
from typing import ClassVar

import itk
import vtk

from oai_analysis.algorithms import mesh_processing as mp
from oai_analysis.registry import register
from oai_analysis.stages.base import MeshGenerator


@register
@dataclass
class CuberilleMeshGenerator(MeshGenerator):
    """Cuberille iso-surface of the probability map, with vertices projected onto the iso-surface."""

    name: ClassVar[str] = "cuberille"
    iso_surface_value: float = 0.5
    project_vertex_surface_distance_threshold: float = 0.05

    def run(self, probability: itk.Image) -> vtk.vtkPolyData:
        itk_mesh = mp.get_mesh_from_probability_map(
            probability, self.iso_surface_value, self.project_vertex_surface_distance_threshold)
        return mp.itk_mesh_to_vtk_mesh(itk_mesh)
