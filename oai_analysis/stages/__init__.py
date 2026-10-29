"""
Pipeline stages: the interfaces (base.py) and their implementations.
"""
from oai_analysis.stages.base import (
    AtlasMapper,
    BiomarkerExtractor,
    CaseArtifacts,
    CaseInfo,
    MeshGenerator,
    MeshResult,
    Preprocessor,
    QCFinding,
    RegistrationResult,
    Registerer,
    Segmenter,
    Stage,
    ThicknessEstimator,
)
from oai_analysis.stages.atlas_mapping import ClosestPointAtlasMapper
from oai_analysis.stages.biomarkers import ThicknessMap2D
from oai_analysis.stages.meshing import CuberilleMeshGenerator
from oai_analysis.stages.preprocessing import OAIPreprocessor
from oai_analysis.stages.registration import UniGradICONRegisterer
from oai_analysis.stages.segmentation import UNetSegmenter
from oai_analysis.stages.thickness import DistanceTransformThickness, MeshSplittingThickness
