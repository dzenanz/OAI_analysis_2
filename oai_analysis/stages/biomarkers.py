from dataclasses import dataclass
from typing import ClassVar

from oai_analysis.cartilage_shape_processing import thickness_grid, thickness_grid_png
from oai_analysis.stages.base import BiomarkerExtractor, CaseArtifacts


@dataclass
class ThicknessMap2D(BiomarkerExtractor):
    """Thickness on the atlas, interpolated over the atlas' 2D embedding: an array and a picture per structure."""

    name: ClassVar[str] = "thickness_map_2d"
    grid_resolution: int = 300

    def run(self, case: CaseArtifacts) -> dict[str, object]:
        outputs = {}
        for structure in case.atlas_mapped:
            grid, extent = thickness_grid(case.atlas.embedding_2d[structure], case.atlas_thickness(structure),
                                          ninter=self.grid_resolution)
            outputs[f"{structure}_thickness.npy"] = grid
            outputs[f"{structure}_thickness.png"] = thickness_grid_png(grid, extent, name=f"{structure}_2D_map")
        return outputs
