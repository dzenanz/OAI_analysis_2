from dataclasses import dataclass
from typing import ClassVar

import itk
import vtk

from oai_analysis import qc
from oai_analysis.algorithms import mesh_processing as mp
from oai_analysis.registry import register
from oai_analysis.stages.base import CaseArtifacts, MeshGenerator, QCFinding


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

    def check(self, case: CaseArtifacts) -> list[QCFinding]:
        return [QCFinding("fail", "mesh.empty", f"the {qc.STRUCTURE_NAMES[s]} surface is empty", 0.0)
                for s, mesh in case.mesh.items() if qc.mesh_is_empty(mesh)]
