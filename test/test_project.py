import json
import textwrap
from dataclasses import dataclass, field
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
    result = project.run(pipeline=pipeline, echo=lambda text: None)
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
    project.run(pipeline=pipeline, echo=lambda text: None)
    assert started_times(project, "1_00M_LEFT") == before  # nothing to redo


def test_changed_parameter_reruns_only_affected_steps(project):
    project.run(pipeline=make_pipeline(), echo=lambda text: None)
    before = started_times(project, "1_00M_LEFT")

    pipeline = make_pipeline(thickness=3.0)
    status = project.status(pipeline)[0]
    assert status.status == "Outdated"
    assert status.note == "thickness_estimator.thickness changed 2.0 -> 3.0"
    assert status.steps["mesh"] == "done" and status.steps["atlas_map"] == "outdated"

    project.run(pipeline=pipeline, echo=lambda text: None)
    after = started_times(project, "1_00M_LEFT")
    rerun = [name for name in STEP_NAMES if after[name] != before[name]]
    assert rerun == ["thickness", "atlas_map", "biomarkers/mean_thickness"]
    mean = np.load(project.case_folder("1_00M_LEFT") / "07_biomarkers/mean_thickness/mean_thickness.npy")
    np.testing.assert_allclose(mean, [3.0, 3.0])


def test_failure_does_not_stop_the_batch(project):
    pipeline = make_pipeline(mesh_generator=FailingMeshGenerator())
    result = project.run(pipeline=pipeline, echo=lambda text: None)
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
    result = project.run(pipeline=make_pipeline(), echo=lambda text: None)
    assert [s.status for s in result] == ["Succeeded", "Succeeded"]
    assert started_times(project, "1_00M_LEFT") == before
    segment_after = (project.case_folder("2_00M_RIGHT") / "02_segment/provenance.json").stat().st_mtime_ns
    assert segment_after == segment_before


def test_interrupted_case_resumes(project):
    pipeline = make_pipeline()
    project.run(["1_00M_LEFT"], pipeline=pipeline, echo=lambda text: None)
    folder = project.case_folder("1_00M_LEFT")
    for step in ("06_atlas_map", "07_biomarkers"):  # as if the run was killed during atlas mapping
        for record in folder.glob(f"{step}/**/provenance.json"):
            record.unlink()
    (folder / "06_atlas_map" / "FC_atlas.vtk").write_bytes(b"half a file")
    before = started_times_partial(project, "1_00M_LEFT")
    status = project.status(pipeline)[0]
    assert status.status == "Incomplete" and status.note.startswith('Stopped before "mapping to the atlas"')

    project.run(["1_00M_LEFT"], pipeline=pipeline, echo=lambda text: None)
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
    project.run(pipeline=make_pipeline(), echo=lambda text: None)
    before = started_times(project, "1_00M_LEFT")
    project.run(["1_00M_LEFT"], from_step="mesh", pipeline=make_pipeline(), echo=lambda text: None)
    after = started_times(project, "1_00M_LEFT")
    assert [name for name in STEP_NAMES if after[name] != before[name]] == [
        "mesh", "thickness", "atlas_map", "biomarkers/mean_thickness"]


def test_retention_final(project):
    project.settings["retention"] = "final"
    pipeline = make_pipeline(extractor=AtlasOnlyMeanThickness())
    project.run(["1_00M_LEFT"], pipeline=pipeline, echo=lambda text: None)
    folder = project.case_folder("1_00M_LEFT")
    assert [p.name for p in (folder / "02_segment").iterdir()] == ["provenance.json"]
    assert (folder / "06_atlas_map" / "FC_atlas.vtk").is_file()
    assert project.status(pipeline)[0].status == "Succeeded"

    # the biomarker reads only the atlas-mapped meshes, which are kept, so it reruns alone
    before = started_times(project, "1_00M_LEFT")
    project.run(["1_00M_LEFT"], from_step="biomarkers/mean_thickness", pipeline=pipeline,
                echo=lambda text: None)
    after = started_times(project, "1_00M_LEFT")
    assert [name for name in STEP_NAMES if after[name] != before[name]] == ["biomarkers/mean_thickness"]

    # a step whose inputs were deleted gets them by running the earlier steps again
    project.run(["1_00M_LEFT"], from_step="mesh", pipeline=pipeline, echo=lambda text: None)
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


@dataclass
class BatchSegmenter(FakeSegmenter):
    """Records how many images each call gets; a batch fails if it holds the case with spacing 0.6 and fail is set."""
    performance_params: ClassVar[tuple[str, ...]] = ("calls",)  # so that recording calls does not outdate outputs
    calls: list = field(default_factory=list)
    fail: bool = False

    def run(self, image):
        self.calls.append(1)
        if self.fail and image.GetSpacing()[0] != 0.5:
            raise RuntimeError("bad image")
        return super().run(image)

    def run_batch(self, images):
        self.calls.append(len(images))
        if self.fail and any(image.GetSpacing()[0] != 0.5 for image in images):
            raise RuntimeError("bad image in the batch")
        return [super(BatchSegmenter, self).run(image) for image in images]


@dataclass
class BatchRegisterer(IdentityRegisterer):
    performance_params: ClassVar[tuple[str, ...]] = ("calls",)
    calls: list = field(default_factory=list)

    def run(self, moving, fixed):
        self.calls.append(1)
        return super().run(moving, fixed)

    def run_batch(self, movings, fixed):
        self.calls.append(len(movings))
        return [super(BatchRegisterer, self).run(moving, fixed) for moving in movings]


def batch_pipeline(fail=False):
    pipeline = make_pipeline()
    pipeline.segmenter, pipeline.registerer = BatchSegmenter(fail=fail), BatchRegisterer()
    return pipeline


def test_batches(project):
    """The cases of a batch go through segmentation and registration together, each getting its own outputs."""
    assert project.settings["batch_size"] == 8
    pipeline = batch_pipeline()
    lines = []
    result = project.run(pipeline=pipeline, echo=lines.append)
    assert [s.status for s in result] == ["Succeeded", "Succeeded"]
    assert pipeline.segmenter.calls == [2] and pipeline.registerer.calls == [2]
    assert any("segmentation of 2 cases" in line for line in lines)
    for case_id, spacing in (("1_00M_LEFT", 0.5), ("2_00M_RIGHT", 0.6)):
        folder = project.case_folder(case_id)
        assert itk.imread(str(folder / "02_segment" / "FC_prob.nrrd")).GetSpacing()[0] == pytest.approx(spacing)
        record = json.loads((folder / "02_segment" / "provenance.json").read_text())
        assert record["batch_size"] == 2 and record["status"] == "complete"
        assert not (folder / "running.json").exists()

    # batch_size: 1 runs the cases one after another
    pipeline = batch_pipeline()
    project.run(from_step="preprocess", pipeline=pipeline, batch_size=1, echo=lambda text: None)
    assert pipeline.segmenter.calls == [1, 1] and pipeline.registerer.calls == [1, 1]


def test_failed_batch_fails_only_its_bad_case(project):
    pipeline = batch_pipeline(fail=True)
    result = project.run(pipeline=pipeline, echo=lambda text: None)
    assert [s.status for s in result] == ["Succeeded", "Failed"]
    assert result[1].failed_step == "segment" and result[1].note.endswith("RuntimeError: bad image")
    assert pipeline.segmenter.calls == [2, 1, 1]  # the batch, then each case on its own
    assert pipeline.registerer.calls == [1]  # the failed case goes no further


def test_performance_params_do_not_make_outputs_outdated(project):
    project.run(pipeline=make_pipeline(), echo=lambda text: None)
    pipeline = make_pipeline(extractor=FastMeanThickness(chunk=1))
    project.run(pipeline=pipeline, echo=lambda text: None)
    pipeline = make_pipeline(extractor=FastMeanThickness(chunk=2))
    assert statuses(project, pipeline) == {"1_00M_LEFT": "Succeeded", "2_00M_RIGHT": "Succeeded"}


@dataclass
class FastMeanThickness(MeanThickness):
    performance_params: ClassVar[tuple[str, ...]] = ("chunk",)
    chunk: int = 1
