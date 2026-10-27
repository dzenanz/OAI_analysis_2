"""
The atlas that patient images are registered to and thickness maps are reported in.
The atlas files ship with the package.
"""
import functools
from dataclasses import dataclass
from pathlib import Path

import itk
import numpy as np
import vtk

from oai_analysis import mesh_processing as mp

RESOURCES_DIR = Path(__file__).resolve().parent / "resources"
DEFAULT_ATLAS_NAME = "atlas_60_LEFT_baseline_NMI"
STRUCTURES = ("FC", "TC")  # femoral and tibial cartilage


@dataclass(frozen=True)
class Atlas:
    name: str
    image: itk.Image  # float pixels, ASL orientation
    inner_mesh: dict[str, vtk.vtkPolyData]  # inner (bone-facing) cartilage surface, keyed by structure
    embedding_2d: dict[str, np.ndarray]  # 2D coordinates of each inner_mesh vertex, keyed by structure

    @classmethod
    def load(cls, atlas_dir):
        atlas_dir = Path(atlas_dir)
        return cls(
            name=atlas_dir.name,
            image=itk.imread(atlas_dir / "atlas_image.nii.gz", itk.F),
            inner_mesh={s: mp.read_vtk_mesh(atlas_dir / f"atlas_{s}_inner_mesh_LPS.ply") for s in STRUCTURES},
            embedding_2d={s: np.load(atlas_dir / f"{s}_inner_embedded.npy") for s in STRUCTURES},
        )

    @classmethod
    @functools.cache
    def default(cls):
        """The atlas used by the pipeline, loaded once per process."""
        return cls.load(RESOURCES_DIR / DEFAULT_ATLAS_NAME)
