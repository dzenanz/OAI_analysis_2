"""
OAPipeline wires the stages together in their fixed order. It holds no algorithm code.

analysis_pipeline() is the original entry point, kept with the same signature and output files.
"""
import contextlib
import functools
import os
import sys
import time
from dataclasses import dataclass, field

import torch

from oai_analysis import config, data, io
from oai_analysis.atlas import DEFAULT_ATLAS_NAME, STRUCTURES, Atlas
from oai_analysis.stages import (
    AtlasMapper,
    BiomarkerExtractor,
    CaseArtifacts,
    CaseInfo,
    ClosestPointAtlasMapper,
    CuberilleMeshGenerator,
    MeshGenerator,
    MeshResult,
    MeshSplittingThickness,
    OAIPreprocessor,
    Preprocessor,
    RegistrationResult,
    Registerer,
    Segmenter,
    Stage,
    ThicknessEstimator,
    ThicknessMap2D,
    UNetSegmenter,
    UniGradICONRegisterer,
)


# the stages OAPipeline uses unless told otherwise
DEFAULT_STAGES = {
    "preprocessor": OAIPreprocessor,
    "segmenter": UNetSegmenter,
    "registerer": UniGradICONRegisterer,
    "mesh_generator": CuberilleMeshGenerator,
    "thickness_estimator": MeshSplittingThickness,
    "atlas_mapper": ClosestPointAtlasMapper,
}
DEFAULT_BIOMARKER_EXTRACTORS = (ThicknessMap2D,)


def default_device():
    if torch.cuda.is_available():
        return "cuda"
    print("WARNING: CUDA NOT AVAILABLE, FALLING BACK TO CPU")
    return "cpu"


class OAPipeline:
    """
    Computes cartilage thickness for femur and tibia from knee MRI.

    Every stage has a default; pass an instance to replace it. Models are loaded once, before the first case,
    and reused for every following case.
    """

    def __init__(self,
                 preprocessor: Preprocessor | None = None,
                 segmenter: Segmenter | None = None,
                 registerer: Registerer | None = None,
                 mesh_generator: MeshGenerator | None = None,
                 thickness_estimator: ThicknessEstimator | None = None,
                 atlas_mapper: AtlasMapper | None = None,
                 biomarker_extractors: list[BiomarkerExtractor] | None = None,
                 atlas: Atlas | None = None,
                 device: str | None = None):
        self.preprocessor = preprocessor or DEFAULT_STAGES["preprocessor"]()
        self.segmenter = segmenter or DEFAULT_STAGES["segmenter"]()
        self.registerer = registerer or DEFAULT_STAGES["registerer"]()
        self.mesh_generator = mesh_generator or DEFAULT_STAGES["mesh_generator"]()
        self.thickness_estimator = thickness_estimator or DEFAULT_STAGES["thickness_estimator"]()
        self.atlas_mapper = atlas_mapper or DEFAULT_STAGES["atlas_mapper"]()
        self.biomarker_extractors = (biomarker_extractors if biomarker_extractors is not None
                                     else [cls() for cls in DEFAULT_BIOMARKER_EXTRACTORS])
        self._atlas = atlas
        self.device = device
        self._is_set_up = False

    @classmethod
    def from_config(cls, source, atlas: Atlas | None = None) -> "OAPipeline":
        """Create a pipeline from a config: a dict, or the path of a YAML file. See config.py for the format."""
        cfg = config.load(source)
        config.check_keys(cfg)
        section = cfg.get("pipeline") or {}
        stages = {kind: config.stage_from_dict(kind, section.get(kind), DEFAULT_STAGES[kind])
                  for kind in config.SINGLE_STAGES}
        if config.BIOMARKER_EXTRACTORS in section:
            stages[config.BIOMARKER_EXTRACTORS] = [config.stage_from_dict("biomarker_extractor", entry)
                                                   for entry in section[config.BIOMARKER_EXTRACTORS] or []]
        device = cfg.get("device", "auto")
        return cls(**stages, atlas=atlas, device=None if device == "auto" else device)

    def config(self) -> dict:
        """The full config of this pipeline; from_config() of it creates an identical pipeline."""
        return {
            "pipeline": {
                **{kind: config.stage_to_dict(getattr(self, kind)) for kind in config.SINGLE_STAGES},
                config.BIOMARKER_EXTRACTORS: [config.stage_to_dict(e) for e in self.biomarker_extractors],
            },
            "device": self.device or "auto",
        }

    @property
    def atlas_name(self) -> str:
        """The name of the atlas, without loading it."""
        return self._atlas.name if self._atlas is not None else DEFAULT_ATLAS_NAME

    @property
    def atlas(self) -> Atlas:
        if self._atlas is None:  # loaded on first use, so that creating a pipeline is cheap
            self._atlas = Atlas.default()
        return self._atlas

    @property
    def stages(self):
        """All stages, in the order they run."""
        return [step.stage for step in self.steps()]

    def steps(self) -> list["Step"]:
        """The steps of a case, in the order they run: one per stage, and one per biomarker extractor."""
        steps = [Step(name, getattr(self, attribute), inputs, f"{i:02d}_{name}", description)
                 for i, (name, (attribute, inputs, description)) in enumerate(STEPS.items(), start=1)]
        biomarkers_folder = f"{len(STEPS) + 1:02d}_biomarkers"
        steps += [Step(f"biomarkers/{e.name}", e, tuple(e.requires), f"{biomarkers_folder}/{e.name}",
                       f'biomarker "{e.name}"') for e in self.biomarker_extractors]
        return steps

    def setup(self):
        """Load models. run_case() calls this when needed, so calling it directly is optional."""
        if self._is_set_up:
            return
        self.device = self.device or default_device()
        for stage in self.stages:
            stage.setup(self.device)
        self._is_set_up = True

    def run_case(self, case: CaseInfo, output_dir=None, keep_intermediate_outputs=True) -> CaseArtifacts:
        """
        Process one case and return everything it produced.

        If output_dir is given, the biomarker outputs are written to it, along with the intermediate outputs
        if keep_intermediate_outputs is set. Files are written as soon as each stage finishes.
        """
        self.setup()
        writer = _FlatLayoutWriter(output_dir, keep_intermediate_outputs)
        result = CaseArtifacts(case=case, atlas=self.atlas)
        for step in self.steps():
            with _timed(result, step.name):
                writer.write(step, self.run_step(step, result))
            if step.name in BATCHED_STEPS:
                release_gpu_memory()
        return result

    def run_step(self, step: "Step", result: CaseArtifacts) -> "StepOutputs":
        """Run one step, reading its inputs from result and storing its outputs in it."""
        case = result.case
        match step.name:
            case "preprocess":
                image = io.read_image(case.input)
                description = io.series_description(case.input, image)
                if description is not None:
                    print(f"Laterality: {case.laterality}, Series description: {description}")
                result.image = self.preprocessor.run(image, case)
                return StepOutputs({"image": result.image})
            case "segment":
                print("Segmenting the cartilage")
                result.probability = self.segmenter.run(result.image, case)
                return StepOutputs({f"{s}_prob": result.probability[s] for s in STRUCTURES})
            case "register":
                print("Registering the input image to the atlas")
                result.registration = self.registerer.run(result.image, self.atlas.image)
                return StepOutputs({"phi_AB": result.registration.phi_AB, "phi_BA": result.registration.phi_BA})
            case "mesh":
                print("Converting segmentation probability images to meshes")
                for s in STRUCTURES:
                    result.mesh[s] = self.mesh_generator.run(result.probability[s])
                return StepOutputs({f"{s}_mesh": result.mesh[s] for s in STRUCTURES})
            case "thickness":
                print(f"Computing the thickness with {self.thickness_estimator.name}")
                for s in STRUCTURES:
                    result.thickness[s] = self.thickness_estimator.run(
                        result.mesh[s], result.probability[s], result.registration, self.atlas, s)
                return _mesh_results("thickness", result.thickness)
            case "atlas_map":
                print("Mapping thickness from patient meshes to the atlas")
                for s in STRUCTURES:
                    result.atlas_mapped[s] = self.atlas_mapper.run(
                        result.thickness[s].mesh, result.registration, self.atlas, s)
                return _mesh_results("atlas", result.atlas_mapped)
            case _:
                print(f"Computing the biomarker {step.stage.name}")
                outputs = step.stage.run(result)
                result.biomarkers.update(outputs)
                return StepOutputs(outputs)

    def load_step(self, step: "Step", result: CaseArtifacts, files: dict):
        """Store the outputs of an earlier run of a step in result: the reverse of run_step(), from files by name."""
        values = {name: io.read(path) for name, path in files.items()}
        match step.name:
            case "preprocess":
                result.image = values["image"]
            case "segment":
                result.probability = {s: values[f"{s}_prob"] for s in STRUCTURES}
            case "register":
                result.registration = RegistrationResult(values["phi_AB"], values["phi_BA"])
            case "mesh":
                result.mesh = {s: values[f"{s}_mesh"] for s in STRUCTURES}
            case "thickness":
                result.thickness = {s: MeshResult(values[f"{s}_thickness"]) for s in STRUCTURES}
            case "atlas_map":
                result.atlas_mapped = {s: MeshResult(values[f"{s}_atlas"]) for s in STRUCTURES}
            case _:
                result.biomarkers.update(values)


# step name: (OAPipeline attribute holding its stage, the steps whose outputs it reads, plain-English description)
STEPS = {
    "preprocess": ("preprocessor", (), "preprocessing"),
    "segment": ("segmenter", ("preprocess",), "segmentation"),
    "register": ("registerer", ("preprocess",), "registration"),
    "mesh": ("mesh_generator", ("segment",), "mesh generation"),
    "thickness": ("thickness_estimator", ("mesh", "segment", "register"), "thickness measurement"),
    "atlas_map": ("atlas_mapper", ("thickness", "register"), "mapping to the atlas"),
}
# the steps that use a neural network; a project runs each of them for all the cases of a batch, one after
# another, while the network is loaded and warm, then finishes each case on its own
BATCHED_STEPS = ("segment", "register")


@dataclass(frozen=True)
class Step:
    """One step of a case: a stage applied to every structure, or one biomarker extractor."""
    name: str  # e.g. "segment" or "biomarkers/thickness_map_2d"
    stage: Stage
    inputs: tuple[str, ...]  # the steps whose outputs it reads
    folder: str  # where a project keeps its outputs, relative to the case folder
    description: str  # plain English, for statuses and summaries


@dataclass
class StepOutputs:
    outputs: dict[str, object]  # by name, e.g. "FC_prob"; later steps read these. Biomarker names are file names.
    intermediates: dict[str, object] = field(default_factory=dict)  # other outputs worth looking at


def release_gpu_memory():
    """
    Give the GPU memory PyTorch keeps cached back, once a network is done with all the cases it gets. Otherwise
    segmentation and registration together reserve more than a 16 GB GPU has, and Windows makes up the difference
    with much slower system memory: registration took 180 s instead of 17 s.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _mesh_results(name, results):
    """The outputs of a step that produces a MeshResult per structure, e.g. FC_thickness and FC_inner."""
    return StepOutputs({f"{s}_{name}": r.mesh for s, r in results.items()},
                       {f"{s}_{key}": value for s, r in results.items() for key, value in r.intermediates.items()})


@contextlib.contextmanager
def _timed(result, stage):
    start = time.perf_counter()
    yield
    result.seconds[stage] = time.perf_counter() - start


class _FlatLayoutWriter:
    """Writes outputs into one directory, with the file names analysis_pipeline() has always used."""

    NAMES = {
        "image": "in_image.nrrd", "phi_AB": "resampling.tfm", "phi_BA": "modelling.tfm",
        **{f"{s}_{name}": f"{s}_{file_name}" for s in STRUCTURES for name, file_name in (
            ("prob", "prob.nrrd"), ("mesh", "mesh_patient.vtk"), ("thickness", "outer.vtk"),
            ("atlas", "mapped_mesh.vtk"))},
    }

    def __init__(self, output_dir, keep_intermediate_outputs):
        self.output_dir = output_dir
        self.keep_intermediate_outputs = keep_intermediate_outputs
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)

    def write(self, step, outputs):
        if step.stage.kind == "biomarker_extractor":  # the final results, always written
            for filename, value in outputs.outputs.items():
                self._write(filename, value)
            return
        if not self.keep_intermediate_outputs:
            return
        for name, value in outputs.outputs.items():
            self._write(self.NAMES[name], value)
        for name, value in outputs.intermediates.items():
            # stage intermediates such as thickness images are compressed, as they always have been
            self._write(f"{name}{io.default_extension(value)}", value, compression=True)

    def _write(self, filename, value, compression=False):
        if self.output_dir is not None:
            io.write(value, os.path.join(self.output_dir, filename), compression=compression)


@functools.cache
def default_pipeline():
    """The pipeline with all default stages, created once per process."""
    return OAPipeline()


def analysis_pipeline(input_path, output_path, laterality, keep_intermediate_outputs):
    """
    Computes cartilage thickness for femur and tibia from knee MRI.

    :param input_path: path to the input image file, or path to input directory containing DICOM image series.
    :param output_path: path to the desired directory for outputs.
    :param laterality: "left" or "right"
    :param keep_intermediate_outputs: also write images, transforms and meshes from intermediate steps
    """
    default_pipeline().run_case(CaseInfo(input_path, laterality), output_path, keep_intermediate_outputs)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        oai_dir = sys.argv[1]
        test_cases = {
            f"{oai_dir}/OAIBaselineImages/results/0.C.2/9016304/20040917/10242106": {
                "laterality": "left", "name": "9016304_0M"},
            f"{oai_dir}/OAIBaselineImages/results/0.C.2/9016304/20040917/10242112": {
                "laterality": "right", "name": "9016304_0M"},
            f"{oai_dir}/OAIBaselineImages/results/0.C.2/9021791/20040922/10250903": {
                "laterality": "left", "name": "9021791_0M"},
            f"{oai_dir}/OAIBaselineImages/results/0.C.2/9021791/20040922/10250909": {
                "laterality": "right", "name": "9021791_0M"},
            f"{oai_dir}/OAIBaselineImages/results/0.C.2/9094865/20050204/10165606": {
                "laterality": "left", "name": "9094865_0M"},
            f"{oai_dir}/OAIBaselineImages/results/0.C.2/9094865/20050204/10165612": {
                "laterality": "right", "name": "9094865_0M"},
            f"{oai_dir}/OAI96MonthImages/results/10.C.1/9016304/20120831/13486606": {
                "laterality": "left", "name": "9016304_96M"},
            f"{oai_dir}/OAI96MonthImages/results/10.C.1/9016304/20120831/13486612": {
                "laterality": "right", "name": "9016304_96M"},
            f"{oai_dir}/OAI96MonthImages/results/10.C.1/9021791/20121016/13466523": {
                "laterality": "left", "name": "9021791_96M"},
            f"{oai_dir}/OAI96MonthImages/results/10.C.1/9021791/20121016/13466530": {
                "laterality": "right", "name": "9021791_96M"},
            f"{oai_dir}/OAI96MonthImages/results/10.C.1/9040390/20121219/13439303/": {
                "laterality": "left", "name": "9040390_96M"},
            f"{oai_dir}/OAI96MonthImages/results/10.C.1/9040390/20121219/13439309/": {
                "laterality": "right", "name": "9040390_96M"},
        }
    else:  # only run on the test case bundled with the code
        test_cases = {
            f"{data.test_data_dir()}/colab_case/image_preprocessed.nii.gz": {
                "laterality": "left", "name": "test_case"},
        }

    for case, case_info in test_cases.items():
        print(f"\nProcessing {case}")
        output_path = f"./OAI_results_2/{case_info['name']}_{case_info['laterality']}"
        analysis_pipeline(case, output_path, case_info['laterality'], keep_intermediate_outputs=True)
