from dataclasses import dataclass
from typing import ClassVar

from oai_analysis.algorithms import mesh_processing as mp
from oai_analysis.registry import register
from oai_analysis.stages.base import AtlasMapper, MeshResult


@register
@dataclass
class ClosestPointAtlasMapper(AtlasMapper):
    """Move the patient mesh into atlas space and give each atlas vertex the thickness of the closest patient point."""

    name: ClassVar[str] = "closest_point"

    def run(self, thickness_mesh, registration, atlas, structure):
        mesh_atlas = mp.transform_mesh(thickness_mesh, registration.phi_BA)
        mapped = mp.map_attributes(mesh_atlas, atlas.inner_mesh[structure])
        return MeshResult(mapped, intermediates={"mesh_atlas": mesh_atlas})
