import json
import textwrap
from dataclasses import dataclass
from typing import ClassVar

import itk
import numpy as np
import pytest

from oai_analysis import OAPipeline
from oai_analysis.atlas import Atlas
from oai_analysis.project import Project, read_manifest
from oai_analysis.stages import Preprocessor
from test.test_stages import (
    ConstantThickness,
    FakeMeshGenerator,
    FakeSegmenter,
    IdentityMapper,
    IdentityRegisterer,
    MeanThickness,
    ball_image,
)

STEP_NAMES = ["preprocess", "segment", "register", "mesh", "thickness", "atlas_map", "biomarkers/mean_thickness"]


@dataclass
class FakePreprocessor(Preprocessor):
    name: ClassVar[str] = "fake"

    def run(self, image, case):
        return image


class FailingMeshGenerator(FakeMeshGenerator):
    def run(self, probability):
        if probability.GetSpacing()[0] != 0.5:  # the case made with a different spacing
            raise RuntimeError("cuberille went wrong")
        return super().run(probability)


@dataclass
class AtlasOnlyMeanThickness(MeanThickness):
    requires: ClassVar[tuple[str, ...]] = ("atlas_map",)


def make_pipeline(thickness=2.0, mesh_generator=None, extractor=None):
    tiny_atlas = Atlas(name="tiny", image=ball_image(), inner_mesh={}, embedding_2d={})
    return OAPipeline(preprocessor=FakePreprocessor(), segmenter=FakeSegmenter(), registerer=IdentityRegisterer(),
                      mesh_generator=mesh_generator or FakeMeshGenerator(),
                      thickness_estimator=ConstantThickness(thickness=thickness), atlas_mapper=IdentityMapper(),
                      biomarker_extractors=[extractor or MeanThickness()], atlas=tiny_atlas, device="cpu")


@pytest.fixture
def project(tmp_path):
    for name, spacing in (("a.nrrd", 0.5), ("b.nrrd", 0.6)):
        image = ball_image()
        image.SetSpacing([spacing] * 3)
        itk.imwrite(image, str(tmp_path / name))
    cases = [{"case_id": "1_00M_LEFT", "patient_id": "1", "month": 0, "laterality": "left", "input": "a.nrrd"},
             {"case_id": "2_00M_RIGHT", "patient_id": "2", "month": 0, "laterality": "right", "input": "b.nrrd"}]
    return Project.create(tmp_path / "project", cases, data_root=tmp_path)


def started_times(project, case_id):
    """When each step's provenance was written, to tell which steps ran."""
    folder = project.case_folder(case_id)
    return {step.name: (folder / step.folder / "provenance.json").stat().st_mtime_ns
            for step in make_pipeline().steps()}


def statuses(project, pipeline):
    return {s.case_id: s.status for s in project.status(pipeline)}


def test_create(project):
    assert (project.path / "project.yaml").is_file()
    assert project.case_folder("1_00M_LEFT") == project.path / "cases" / "1" / "00M_LEFT"
    assert project.storage == {"image_format": "nrrd", "transform_format": "h5"}
    assert statuses(project, make_pipeline()) == {"1_00M_LEFT": "Not started", "2_00M_RIGHT": "Not started"}


def test_run_and_resume(project):
    pipeline = make_pipeline()
    result = project.run(pipeline=pipeline, in_process=True, echo=lambda text: None)
    assert [s.status for s in result] == ["Succeeded", "Succeeded"]

    folder = project.case_folder("1_00M_LEFT")
    assert sorted(p.name for p in (folder / "03_register").iterdir()) == ["phi_AB.h5", "phi_BA.h5", "provenance.json"]
    assert not (folder / "05_thickness" / "FC_inner.vtk").exists()  # intermediates only with retention: all
    record = json.loads((folder / "05_thickness" / "provenance.json").read_text())
    assert record["status"] == "complete" and record["params"] == {"name": "constant", "thickness": 2.0}
    assert set(record["inputs"]) >= {"atlas", "mesh/FC_mesh", "register/phi_AB"}
    assert record["inputs"]["mesh/FC_mesh"]["path"] == "04_mesh/FC_mesh.vtk"
    mean = np.load(folder / "07_biomarkers" / "mean_thickness" / "mean_thickness.npy")
    np.testing.assert_allclose(mean, [2.0, 2.0])
    assert (project.path / "index.parquet").is_file() and (project.path / "runs.jsonl").is_file()

    before = started_times(project, "1_00M_LEFT")
    project.run(pipeline=pipeline, in_process=True, echo=lambda text: None)
    assert started_times(project, "1_00M_LEFT") == before  # nothing to redo


def test_changed_parameter_reruns_only_affected_steps(project):
    project.run(pipeline=make_pipeline(), in_process=True, echo=lambda text: None)
    before = started_times(project, "1_00M_LEFT")

    pipeline = make_pipeline(thickness=3.0)
    status = project.status(pipeline)[0]
    assert status.status == "Outdated"
    assert status.note == "thickness_estimator.thickness changed 2.0 -> 3.0"
    assert status.steps["mesh"] == "done" and status.steps["atlas_map"] == "outdated"

    project.run(pipeline=pipeline, in_process=True, echo=lambda text: None)
    after = started_times(project, "1_00M_LEFT")
    rerun = [name for name in STEP_NAMES if after[name] != before[name]]
    assert rerun == ["thickness", "atlas_map", "biomarkers/mean_thickness"]
    mean = np.load(project.case_folder("1_00M_LEFT") / "07_biomarkers/mean_thickness/mean_thickness.npy")
    np.testing.assert_allclose(mean, [3.0, 3.0])


def test_failure_does_not_stop_the_batch(project):
    pipeline = make_pipeline(mesh_generator=FailingMeshGenerator())
    result = project.run(pipeline=pipeline, in_process=True, echo=lambda text: None)
    assert [s.status for s in result] == ["Succeeded", "Failed"]
    failed = result[1]
    assert failed.failed_step == "mesh"
    assert failed.note == 'Failed at "mesh generation": RuntimeError: cuberille went wrong'
    assert project.select(project.status(pipeline), failed_at="mesh")[0].case_id == "2_00M_RIGHT"
    record = json.loads((project.case_folder("2_00M_RIGHT") / "04_mesh" / "provenance.json").read_text())
    assert "cuberille went wrong" in record["traceback"]

    # once fixed, a rerun redoes the failed case from the failed step
    before = started_times(project, "1_00M_LEFT")
    segment_before = (project.case_folder("2_00M_RIGHT") / "02_segment/provenance.json").stat().st_mtime_ns
    result = project.run(pipeline=make_pipeline(), in_process=True, echo=lambda text: None)
    assert [s.status for s in result] == ["Succeeded", "Succeeded"]
    assert started_times(project, "1_00M_LEFT") == before
    segment_after = (project.case_folder("2_00M_RIGHT") / "02_segment/provenance.json").stat().st_mtime_ns
    assert segment_after == segment_before


def test_interrupted_case_resumes(project):
    pipeline = make_pipeline()
    project.run(["1_00M_LEFT"], pipeline=pipeline, in_process=True, echo=lambda text: None)
    folder = project.case_folder("1_00M_LEFT")
    for step in ("06_atlas_map", "07_biomarkers"):  # as if the run was killed during atlas mapping
        for record in folder.glob(f"{step}/**/provenance.json"):
            record.unlink()
    (folder / "06_atlas_map" / "FC_atlas.vtk").write_bytes(b"half a file")
    before = started_times_partial(project, "1_00M_LEFT")
    status = project.status(pipeline)[0]
    assert status.status == "Incomplete" and status.note.startswith('Stopped before "mapping to the atlas"')

    project.run(["1_00M_LEFT"], pipeline=pipeline, in_process=True, echo=lambda text: None)
    after = started_times(project, "1_00M_LEFT")
    assert {name: after[name] for name in before} == before
    assert project.status(pipeline)[0].status == "Succeeded"


def started_times_partial(project, case_id):
    folder = project.case_folder(case_id)
    times = {}
    for step in make_pipeline().steps():
        path = folder / step.folder / "provenance.json"
        if path.exists():
            times[step.name] = path.stat().st_mtime_ns
    return times


def test_from_step(project):
    project.run(pipeline=make_pipeline(), in_process=True, echo=lambda text: None)
    before = started_times(project, "1_00M_LEFT")
    project.run(["1_00M_LEFT"], from_step="mesh", pipeline=make_pipeline(), in_process=True, echo=lambda text: None)
    after = started_times(project, "1_00M_LEFT")
    assert [name for name in STEP_NAMES if after[name] != before[name]] == [
        "mesh", "thickness", "atlas_map", "biomarkers/mean_thickness"]


def test_retention_final(project):
    project.settings["retention"] = "final"
    pipeline = make_pipeline(extractor=AtlasOnlyMeanThickness())
    project.run(["1_00M_LEFT"], pipeline=pipeline, in_process=True, echo=lambda text: None)
    folder = project.case_folder("1_00M_LEFT")
    assert [p.name for p in (folder / "02_segment").iterdir()] == ["provenance.json"]
    assert (folder / "06_atlas_map" / "FC_atlas.vtk").is_file()
    assert project.status(pipeline)[0].status == "Succeeded"

    # the biomarker reads only the atlas-mapped meshes, which are kept, so it reruns alone
    before = started_times(project, "1_00M_LEFT")
    project.run(["1_00M_LEFT"], from_step="biomarkers/mean_thickness", pipeline=pipeline, in_process=True,
                echo=lambda text: None)
    after = started_times(project, "1_00M_LEFT")
    assert [name for name in STEP_NAMES if after[name] != before[name]] == ["biomarkers/mean_thickness"]

    # a step whose inputs were deleted gets them by running the earlier steps again
    project.run(["1_00M_LEFT"], from_step="mesh", pipeline=pipeline, in_process=True, echo=lambda text: None)
    assert project.status(pipeline)[0].status == "Succeeded"


def test_read_manifest(tmp_path):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(textwrap.dedent("""\
        patient_id,study_uid,vol_path,month,study_description
        9016304,1.2,OAIBaselineImages/results/0.C.2/9016304/20040917/10242106,0,OAI^MR^ENROLLMENT^LEFT
        9016304,1.3,OAIBaselineImages/results/0.C.2/9016304/20040917/10242112,0,OAI^MR^ENROLLMENT^RIGHT
        9016304,1.4,OAI96MonthImages/results/10.C.1/9016304/20120831/13486606,96,OAI^MR^96 MONTH^LEFT
        9021791,1.5,OAIBaselineImages/results/0.C.2/9021791/20040922/10250903,0,OAI^MR^ENROLLMENT^LEFT
    """))
    cases = read_manifest(manifest, patients=["9016304"], laterality="left")
    assert [c["case_id"] for c in cases] == ["9016304_00M_LEFT", "9016304_96M_LEFT"]
    assert cases[1]["input"] == "OAI96MonthImages/results/10.C.1/9016304/20120831/13486606"
    assert len(read_manifest(manifest, months=[0])) == 3


PLUGIN = '''
import os
from dataclasses import dataclass
from typing import ClassVar

import itk
import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk

from oai_analysis.registry import register
from oai_analysis.stages import (AtlasMapper, BiomarkerExtractor, MeshGenerator, MeshResult, Preprocessor,
                                 RegistrationResult, Registerer, Segmenter, ThicknessEstimator)


@register
@dataclass
class CrashingPreprocessor(Preprocessor):
    name: ClassVar[str] = "test_crashing"

    def run(self, image, case):
        if "CRASH" in case.case_id:
            os._exit(7)  # like a segmentation fault: the process ends without raising
        return image


@register
@dataclass
class TestSegmenter(Segmenter):
    name: ClassVar[str] = "test_identity"

    def run(self, image):
        return {"FC": image, "TC": image}


@register
@dataclass
class TestRegisterer(Registerer):
    name: ClassVar[str] = "test_identity"

    def run(self, moving, fixed):
        identity = itk.IdentityTransform[itk.D, 3].New()
        return RegistrationResult(identity, identity)


@register
@dataclass
class TestMeshGenerator(MeshGenerator):
    name: ClassVar[str] = "test_sphere"

    def run(self, probability):
        sphere = vtk.vtkSphereSource()
        sphere.Update()
        mesh = sphere.GetOutput()
        mesh.GetPointData().RemoveArray("Normals")
        return mesh


@register
@dataclass
class TestThickness(ThicknessEstimator):
    name: ClassVar[str] = "test_constant"

    def run(self, mesh, probability, registration, atlas, structure):
        result = vtk.vtkPolyData()
        result.DeepCopy(mesh)
        array = numpy_to_vtk(np.full(mesh.GetNumberOfPoints(), 2.0, dtype=np.float32), deep=True)
        array.SetName("vertex_thickness")
        result.GetPointData().AddArray(array)
        return MeshResult(result)


@register
@dataclass
class TestMapper(AtlasMapper):
    name: ClassVar[str] = "test_identity"

    def run(self, thickness_mesh, registration, atlas, structure):
        return MeshResult(thickness_mesh)


@register
@dataclass
class TestMean(BiomarkerExtractor):
    name: ClassVar[str] = "test_mean"
    requires: ClassVar[tuple] = ("atlas_map",)

    def run(self, case):
        return {"mean.npy": np.array([case.atlas_thickness(s).mean() for s in ("FC", "TC")])}
'''


def test_worker_crash_fails_one_case(tmp_path, monkeypatch):
    """A crash that ends the worker process fails that case; a new worker processes the rest."""
    (tmp_path / "plugin").mkdir()
    (tmp_path / "plugin" / "oai_test_stages.py").write_text(PLUGIN)
    dist_info = tmp_path / "plugin" / "oai_test_stages-0.1.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text("Metadata-Version: 2.1\nName: oai-test-stages\nVersion: 0.1\n")
    (dist_info / "entry_points.txt").write_text("[oai_analysis.plugins]\ntest_stages = oai_test_stages\n")
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "plugin"))

    itk.imwrite(ball_image(), str(tmp_path / "a.nrrd"))
    cases = [{"case_id": case_id, "laterality": "left", "input": "a.nrrd"}
             for case_id in ("A_OK", "B_CRASH", "C_OK")]
    config = {"pipeline": {"preprocessor": {"name": "test_crashing"}, "segmenter": {"name": "test_identity"},
                           "registerer": {"name": "test_identity"}, "mesh_generator": {"name": "test_sphere"},
                           "thickness_estimator": {"name": "test_constant"},
                           "atlas_mapper": {"name": "test_identity"},
                           "biomarker_extractors": [{"name": "test_mean"}]},
              "device": "cpu"}
    monkeypatch.syspath_prepend(str(tmp_path / "plugin"))
    project = Project.create(tmp_path / "project", cases, config, data_root=tmp_path)
    lines = []

    result = project.run(echo=lines.append)

    assert [s.status for s in result] == ["Succeeded", "Failed", "Succeeded"]
    assert result[1].failed_step == "preprocess"
    assert "The process crashed (exit code 7)" in result[1].note
    assert any("Starting a new worker process for the remaining 1 case." in line for line in lines)
    summary = next((project.path / "logs").glob("*.summary.txt")).read_text()
    assert "2 Succeeded, 1 Failed" in summary
