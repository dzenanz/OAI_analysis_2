"""
The stage interfaces. The pipeline runs one implementation of each, in this order:

    Preprocessor -> Segmenter -> Registerer -> MeshGenerator -> ThicknessEstimator -> AtlasMapper -> BiomarkerExtractors

Implementations are dataclasses whose fields are their parameters.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import itk
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy

if TYPE_CHECKING:
    from oai_analysis.atlas import Atlas

LATERALITIES = ("left", "right")


@dataclass
class CaseInfo:
    """One knee image to process."""
    input: str | Path  # image file, or a directory holding a DICOM series
    laterality: str  # "left" or "right"
    case_id: str | None = None  # defaults to the name of the input

    def __post_init__(self):
        if self.laterality not in LATERALITIES:
            raise ValueError(f"laterality must be one of {LATERALITIES}, not {self.laterality!r}")
        if self.case_id is None:
            self.case_id = Path(self.input).name


class Stage(ABC):
    """Base for every pipeline stage."""

    kind: ClassVar[str]  # which pipeline slot the stage fills, e.g. "registerer"; set by each interface below
    name: ClassVar[str]  # short identifier, unique within its kind, e.g. "unigradicon"
    version: ClassVar[str] = "1"  # bump whenever the implementation's output changes
    # parameters that change how the stage runs but not what it computes, e.g. how many images share a pass
    # through the network; changing them does not make earlier outputs outdated
    performance_params: ClassVar[tuple[str, ...]] = ()

    def params(self) -> dict:
        """The parameters of this stage."""
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def result_params(self) -> dict:
        """The parameters that can change the stage's outputs: all but the performance_params."""
        return {name: value for name, value in self.params().items() if name not in self.performance_params}

    def setup(self, device: str) -> None:
        """Load heavy resources (networks, weights). Called once, before the first case. Default: nothing."""

    def weights(self) -> dict[str, str]:
        """The files of trained weights the stage uses, with their hashes, e.g. {"model.pth": "sha256:..."}."""
        return {}


@dataclass
class RegistrationResult:
    phi_AB: itk.Transform  # resampling transform (patient -> atlas image)
    phi_BA: itk.Transform  # modelling transform (moves patient meshes into atlas space)


@dataclass
class MeshResult:
    mesh: vtk.vtkPolyData
    intermediates: dict[str, object] = field(default_factory=dict)  # other outputs worth saving, by name


class Preprocessor(Stage):
    kind: ClassVar[str] = "preprocessor"

    @abstractmethod
    def run(self, image: itk.Image, case: CaseInfo) -> itk.Image:
        """Return the image in the atlas orientation, ready for segmentation and registration."""


class Segmenter(Stage):
    kind: ClassVar[str] = "segmenter"

    @abstractmethod
    def run(self, image: itk.Image) -> dict[str, itk.Image]:
        """Return probability maps keyed by structure: {"FC": ..., "TC": ...}."""

    def run_batch(self, images: list[itk.Image]) -> list[dict[str, itk.Image]]:
        """Segment several images. Override to pass them through the network together. Default: one by one."""
        return [self.run(image) for image in images]


class Registerer(Stage):
    kind: ClassVar[str] = "registerer"

    @abstractmethod
    def run(self, moving: itk.Image, fixed: itk.Image) -> RegistrationResult: ...

    def run_batch(self, movings: list[itk.Image], fixed: itk.Image) -> list[RegistrationResult]:
        """Register several images to fixed. Override to pass them through the network together. Default: one by one."""
        return [self.run(moving, fixed) for moving in movings]


class MeshGenerator(Stage):
    kind: ClassVar[str] = "mesh_generator"

    @abstractmethod
    def run(self, probability: itk.Image) -> vtk.vtkPolyData:
        """Return the surface of the structure, in patient space."""


class ThicknessEstimator(Stage):
    kind: ClassVar[str] = "thickness_estimator"

    @abstractmethod
    def run(self, mesh: vtk.vtkPolyData, probability: itk.Image, registration: RegistrationResult,
            atlas: Atlas, structure: str) -> MeshResult:
        """Return a patient-space mesh carrying a per-vertex thickness array."""


class AtlasMapper(Stage):
    kind: ClassVar[str] = "atlas_mapper"

    @abstractmethod
    def run(self, thickness_mesh: vtk.vtkPolyData, registration: RegistrationResult,
            atlas: Atlas, structure: str) -> MeshResult:
        """Return the atlas mesh carrying the patient's thickness."""


@dataclass
class CaseArtifacts:
    """Everything produced for one case so far, keyed by structure where it applies."""
    case: CaseInfo
    atlas: Atlas
    image: itk.Image | None = None  # preprocessed
    probability: dict[str, itk.Image] = field(default_factory=dict)
    registration: RegistrationResult | None = None
    mesh: dict[str, vtk.vtkPolyData] = field(default_factory=dict)  # patient space
    thickness: dict[str, MeshResult] = field(default_factory=dict)  # patient space
    atlas_mapped: dict[str, MeshResult] = field(default_factory=dict)  # atlas inner mesh with the patient's thickness
    biomarkers: dict[str, object] = field(default_factory=dict)  # outputs of the biomarker extractors, by file name
    seconds: dict[str, float] = field(default_factory=dict)  # wall time per stage, including writing its outputs

    def atlas_thickness(self, structure: str) -> np.ndarray:
        """Thickness at each vertex of the atlas inner mesh."""
        return vtk_to_numpy(self.atlas_mapped[structure].mesh.GetPointData().GetArray(0))


class BiomarkerExtractor(Stage):
    kind: ClassVar[str] = "biomarker_extractor"
    # the steps whose outputs run() reads; a project loads or reruns only these before rerunning the extractor
    requires: ClassVar[tuple[str, ...]] = ("preprocess", "segment", "register", "mesh", "thickness", "atlas_map")

    @abstractmethod
    def run(self, case: CaseArtifacts) -> dict[str, object]:
        """
        Read anything produced so far (read-only); return named outputs to be saved.
        The names are file names. Values can be bytes, numpy arrays (.npy), images or meshes.
        """
