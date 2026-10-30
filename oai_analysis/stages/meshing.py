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
    iso_surface_value: float = 0.5  # the probability where the surface is
    project_vertex_surface_distance_threshold: float = 0.05  # how close (mm) a projected vertex gets to the surface

    def run(self, probability: itk.Image) -> vtk.vtkPolyData:
        itk_mesh = mp.get_mesh_from_probability_map(
            probability, self.iso_surface_value, self.project_vertex_surface_distance_threshold)
        return mp.itk_mesh_to_vtk_mesh(itk_mesh)

    def check(self, case: CaseArtifacts) -> list[QCFinding]:
        return mesh_findings(case)


@register
@dataclass
class MarchingCubesMeshGenerator(MeshGenerator):
    """Marching cubes iso-surface of the probability map (scikit-image), optionally smoothed."""

    name: ClassVar[str] = "marching_cubes"
    iso_surface_value: float = 0.5  # the probability where the surface is
    smoothing_iterations: int = 0  # smoothing passes over the surface; 0: none

    def run(self, probability: itk.Image) -> vtk.vtkPolyData:
        mesh = mp.marching_cubes_mesh(probability, self.iso_surface_value)
        if self.smoothing_iterations:
            mesh = mp.smooth_mesh(mesh, num_iterations=self.smoothing_iterations)
        return mesh

    def check(self, case: CaseArtifacts) -> list[QCFinding]:
        return mesh_findings(case)


def mesh_findings(case: CaseArtifacts) -> list[QCFinding]:
    """The check of any mesh generator: the surfaces are not empty."""
    return [QCFinding("fail", "mesh.empty", f"the {qc.STRUCTURE_NAMES[s]} surface is empty", 0.0)
            for s, mesh in case.mesh.items() if qc.mesh_is_empty(mesh)]
