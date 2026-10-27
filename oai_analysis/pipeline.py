"""
OAPipeline wires the stages together in their fixed order. It holds no algorithm code.

analysis_pipeline() is the original entry point, kept with the same signature and output files.
"""
import contextlib
import functools
import os
import sys
import time

import torch

from oai_analysis import data, io
from oai_analysis.atlas import STRUCTURES, Atlas
from oai_analysis.stages import (
    AtlasMapper,
    BiomarkerExtractor,
    CaseArtifacts,
    CaseInfo,
    ClosestPointAtlasMapper,
    CuberilleMeshGenerator,
    MeshGenerator,
    MeshSplittingThickness,
    OAIPreprocessor,
    Preprocessor,
    Registerer,
    Segmenter,
    ThicknessEstimator,
    ThicknessMap2D,
    UNetSegmenter,
    UniGradICONRegisterer,
)


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
        self.preprocessor = preprocessor or OAIPreprocessor()
        self.segmenter = segmenter or UNetSegmenter()
        self.registerer = registerer or UniGradICONRegisterer()
        self.mesh_generator = mesh_generator or CuberilleMeshGenerator()
        self.thickness_estimator = thickness_estimator or MeshSplittingThickness()
        self.atlas_mapper = atlas_mapper or ClosestPointAtlasMapper()
        self.biomarker_extractors = biomarker_extractors if biomarker_extractors is not None else [ThicknessMap2D()]
        self.atlas = atlas or Atlas.default()
        self.device = device
        self._is_set_up = False

    @property
    def stages(self):
        """All stages, in the order they run."""
        return [self.preprocessor, self.segmenter, self.registerer, self.mesh_generator,
                self.thickness_estimator, self.atlas_mapper, *self.biomarker_extractors]

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

        image = io.read_image(case.input)
        description = io.series_description(case.input, image)
        if description is not None:
            print(f"Laterality: {case.laterality}, Series description: {description}")

        with _timed(result, "preprocessor"):
            result.image = self.preprocessor.run(image, case)
            writer.intermediate("in_image.nrrd", result.image)

        print("Segmenting the cartilage")
        with _timed(result, "segmenter"):
            result.probability = self.segmenter.run(result.image)
            for s in STRUCTURES:
                writer.intermediate(f"{s}_prob.nrrd", result.probability[s])

        print("Registering the input image to the atlas")
        with _timed(result, "registerer"):
            result.registration = self.registerer.run(result.image, self.atlas.image)
            writer.intermediate("resampling.tfm", result.registration.phi_AB)
            writer.intermediate("modelling.tfm", result.registration.phi_BA)

        print("Converting segmentation probability images to meshes")
        with _timed(result, "mesh_generator"):
            for s in STRUCTURES:
                result.mesh[s] = self.mesh_generator.run(result.probability[s])
                writer.intermediate(f"{s}_mesh_patient.vtk", result.mesh[s])

        print(f"Computing the thickness with {self.thickness_estimator.name}")
        with _timed(result, "thickness_estimator"):
            for s in STRUCTURES:
                result.thickness[s] = self.thickness_estimator.run(
                    result.mesh[s], result.probability[s], result.registration, self.atlas, s)
                writer.mesh_result(s, "outer", result.thickness[s])

        print("Mapping thickness from patient meshes to the atlas")
        with _timed(result, "atlas_mapper"):
            for s in STRUCTURES:
                result.atlas_mapped[s] = self.atlas_mapper.run(
                    result.thickness[s].mesh, result.registration, self.atlas, s)
                writer.mesh_result(s, "mapped_mesh", result.atlas_mapped[s])

        print("Computing biomarkers")
        for extractor in self.biomarker_extractors:
            with _timed(result, extractor.name):
                outputs = extractor.run(result)
                result.biomarkers.update(outputs)
                for filename, value in outputs.items():
                    writer.final(filename, value)

        return result


@contextlib.contextmanager
def _timed(result, stage):
    start = time.perf_counter()
    yield
    result.seconds[stage] = time.perf_counter() - start


class _FlatLayoutWriter:
    """Writes outputs into one directory, with the file names analysis_pipeline() has always used."""

    def __init__(self, output_dir, keep_intermediate_outputs):
        self.output_dir = output_dir
        self.keep_intermediate_outputs = keep_intermediate_outputs
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)

    def final(self, filename, value, compression=False):
        if self.output_dir is not None:
            io.write(value, os.path.join(self.output_dir, filename), compression=compression)

    def intermediate(self, filename, value, compression=False):
        if self.keep_intermediate_outputs:
            self.final(filename, value, compression)

    def mesh_result(self, structure, mesh_name, mesh_result):
        """Write e.g. FC_outer.vtk, then the stage's other outputs as e.g. FC_inner.vtk."""
        self.intermediate(f"{structure}_{mesh_name}.vtk", mesh_result.mesh)
        for name, value in mesh_result.intermediates.items():
            # stage intermediates such as thickness images are compressed, as they always have been
            self.intermediate(f"{structure}_{name}{io.default_extension(value)}", value, compression=True)


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
