from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from oai_analysis import qc
from oai_analysis.algorithms import mesh_processing as mp
from oai_analysis.registry import register
from oai_analysis.stages.base import AtlasMapper, CaseArtifacts, MeshResult, QCFinding

NO_THICKNESS_MM = 0.1  # thinner than this counts as no cartilage


@register
@dataclass
class ClosestPointAtlasMapper(AtlasMapper):
    """Move the patient mesh into atlas space and give each atlas vertex the thickness of the closest patient point."""

    name: ClassVar[str] = "closest_point"
    check_params: ClassVar[tuple[str, ...]] = ("max_without_thickness",)
    # fraction of the atlas surface without thickness; above it, the case needs review. In the benchmark cases
    # and the Dec 18, 2024 triage, the clean cases stay below 0.15 and most problem cases are above it
    max_without_thickness: float = 0.15

    def run(self, thickness_mesh, registration, atlas, structure):
        mesh_atlas = mp.transform_mesh(thickness_mesh, registration.phi_BA)
        mapped = mp.map_attributes(mesh_atlas, atlas.inner_mesh[structure])
        return MeshResult(mapped, intermediates={"mesh_atlas": mesh_atlas})

    def check(self, case: CaseArtifacts) -> list[QCFinding]:
        findings = []
        for structure in case.atlas_mapped:
            without = float(np.mean(case.atlas_thickness(structure) < NO_THICKNESS_MM))
            findings.append(qc.finding(without > self.max_without_thickness, "atlas_map.no_thickness",
                                       f"the {qc.STRUCTURE_NAMES[structure]} has no thickness on "
                                       f"{qc.percent(without)} of the atlas surface "
                                       f"(review above {qc.percent(self.max_without_thickness)})", without))
        return findings
