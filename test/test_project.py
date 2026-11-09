import json
import textwrap
from pathlib import Path
from dataclasses import dataclass, field
from typing import ClassVar

import itk
import numpy as np
import pyarrow.parquet as pq
import pytest

from oai_analysis import OAPipeline, io, parallel, provenance, qc
from oai_analysis.atlas import Atlas
from oai_analysis.project import Project, choose_series, read_manifest
from oai_analysis.stages import Preprocessor, QCFinding
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
    return Project.create(tmp_path / "project", cases, {"workers": 0}, data_root=tmp_path)


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


def test_failure_does_not_stop_the_others(project):
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


def write_series(folder, slices, acquired, description="SAG_3D_DESS_LEFT"):
    """A DICOM series of 4 x 5 slices, acquired at (date, time)."""
    folder.mkdir(parents=True)
    for i in range(slices):
        image = itk.image_from_array(np.zeros((4, 5), np.int16))
        metadata = image.GetMetaDataDictionary()
        metadata["0008|0022"], metadata["0008|0032"] = acquired
        metadata["0008|103e"] = description
        itk.imwrite(image, folder / f"{i}.dcm", imageio=itk.GDCMImageIO.New())
    return folder.name


def test_series_description(tmp_path):
    folder = tmp_path / write_series(tmp_path / "series", 2, ("20121214", "090000"), "SAG_3D_DESS_RIGHT")
    for _ in range(50):  # it used to read a freed dictionary, which returned None and then crashed
        churn = [np.ones(10_000 * (i + 1)) for i in range(50)]
        assert io.series_description(folder).strip() == "SAG_3D_DESS_RIGHT"
    del churn
    image = itk.imread(folder / "0.dcm")
    assert io.series_description(folder / "0.dcm", image).strip() == "SAG_3D_DESS_RIGHT"
    assert io.series_description(folder / "0.dcm").strip() == "SAG_3D_DESS_RIGHT"


def test_choose_series(tmp_path):
    def case(series):
        return {"case_id": "1_00M_LEFT", "patient_id": "1", "month": 0, "laterality": "left", "input": series}

    full = write_series(tmp_path / "full", 3, ("20121214", "090000"))
    stopped = write_series(tmp_path / "stopped", 2, ("20121214", "100000"))
    early = write_series(tmp_path / "early", 3, ("20121214", "083000.5"))
    with pytest.warns(UserWarning, match=r"using full \(most voxels: 60\), skipping stopped"):
        assert choose_series([case(full), case(stopped)], tmp_path) == [case(full)]
    with pytest.warns(UserWarning, match=r"using full \(acquired last\), skipping early"):
        assert choose_series([case(full), case(early)], tmp_path) == [case(full)]
    with pytest.warns(UserWarning, match=r"using early \(could not read missing\)"):  # a row that cannot be read
        assert choose_series([case(early), case("missing")], tmp_path) == [case(early)]
    other = {**case(full), "case_id": "1_00M_RIGHT"}
    assert choose_series([case(full), other], tmp_path) == [case(full), other]  # no duplicate, no warning


def test_case_listed_twice(tmp_path):
    full = write_series(tmp_path / "data" / "full", 3, ("20121214", "090000"))
    stopped = write_series(tmp_path / "data" / "stopped", 2, ("20121214", "100000"))
    rows = [{"case_id": "1_00M_LEFT", "laterality": "left", "input": stopped},
            {"case_id": "1_00M_LEFT", "laterality": "left", "input": full}]
    with pytest.warns(UserWarning, match="1_00M_LEFT has 2 series; using full"):
        project = Project.create(tmp_path / "project", rows, data_root=tmp_path / "data")
    assert project.cases["1_00M_LEFT"]["input"] == "full"
    assert (tmp_path / "project" / "cases.csv").read_text().count("1_00M_LEFT") == 1

    with open(tmp_path / "project" / "cases.csv", "a", encoding="utf-8") as f:  # listed twice by hand
        f.write(f"1_00M_LEFT,,,left,{stopped}\n")
    with pytest.warns(UserWarning, match="1_00M_LEFT has 2 series; using full"):
        assert Project(tmp_path / "project").cases["1_00M_LEFT"]["input"] == "full"


def case_of(image):
    return "1_00M_LEFT" if image.GetSpacing()[0] == 0.5 else "2_00M_RIGHT"


@dataclass
class RecordingSegmenter(FakeSegmenter):
    """Records which case it segments, in order; fails on the second case if fail is set."""
    performance_params: ClassVar[tuple[str, ...]] = ("calls", "fail")  # so that they do not outdate outputs
    calls: list = field(default_factory=list)
    fail: bool = False

    def run(self, image, case=None):
        self.calls.append(("segment", case_of(image)))
        if self.fail and case_of(image) == "2_00M_RIGHT":
            raise RuntimeError("bad image")
        return super().run(image, case)


@dataclass
class RecordingRegisterer(IdentityRegisterer):
    performance_params: ClassVar[tuple[str, ...]] = ("calls",)
    calls: list = field(default_factory=list)

    def run(self, moving, fixed):
        self.calls.append(("register", case_of(moving)))
        return super().run(moving, fixed)


@dataclass
class RecordingMeshGenerator(FakeMeshGenerator):
    performance_params: ClassVar[tuple[str, ...]] = ("calls",)
    calls: list = field(default_factory=list)

    def run(self, probability):
        if not self.calls or self.calls[-1] != ("mesh", case_of(probability)):  # once per case, not per structure
            self.calls.append(("mesh", case_of(probability)))
        return super().run(probability)


def recording_pipeline(fail=False):
    """A pipeline whose segmenter, registerer and mesh generator record their calls in one list."""
    calls = []
    pipeline = make_pipeline(mesh_generator=RecordingMeshGenerator(calls=calls))
    pipeline.segmenter = RecordingSegmenter(calls=calls, fail=fail)
    pipeline.registerer = RecordingRegisterer(calls=calls)
    return pipeline, calls


def test_parallel_run(project):
    """Worker processes give the results of a run in this process; a failing case stops only itself."""
    pipeline, _ = recording_pipeline(fail=True)
    result = project.run(pipeline=pipeline, workers=2, echo=lambda text: None)
    assert [s.status for s in result] == ["Succeeded", "Failed"]
    assert result[1].failed_step == "segment" and result[1].note.endswith("RuntimeError: bad image")
    assert not (project.case_folder("2_00M_RIGHT") / "03_register").exists()  # the failed case goes no further
    for case_id in ("1_00M_LEFT", "2_00M_RIGHT"):
        assert not (project.case_folder(case_id) / "running.json").exists()
    log = next((project.path / "logs").glob("*[0-9].log")).read_text()
    assert "1_00M_LEFT: Segmenting the cartilage" in log  # what the workers print, with the case it is about
    mean = np.load(project.case_folder("1_00M_LEFT") / "07_biomarkers/mean_thickness/mean_thickness.npy")
    np.testing.assert_allclose(mean, [2.0, 2.0])
    record = provenance_of(project, "1_00M_LEFT", "04_mesh")
    assert record["environment"]["device"] == "cpu"
    assert "check_params" in provenance_of(project, "1_00M_LEFT", "02_segment")  # checked by a CPU worker

    result = project.run(pipeline=make_pipeline(), workers=2, echo=lambda text: None)  # fixed: resumes
    assert [s.status for s in result] == ["Succeeded", "Succeeded"]
    assert provenance_of(project, "1_00M_LEFT", "04_mesh")["started"] == record["started"]


def test_parts_and_resources(monkeypatch):
    assert parallel.parts(make_pipeline()) == (["preprocess"], ["segment", "register"],
                                               ["mesh", "thickness", "atlas_map", "biomarkers/mean_thickness"])
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,5")  # e.g. the GPUs SLURM gave the job
    assert parallel.gpu_devices("cuda") == ["3", "5"]
    assert parallel.gpu_devices("cuda:1") == ["5"]
    assert parallel.gpu_devices("cpu") == [None]
    assert parallel.default_workers(24, 4) == 8 and parallel.default_workers(2, 1) == 1


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


@dataclass
class CheckedThickness(ConstantThickness):
    """Flags femoral cartilage thicker than max_mm for review."""
    name: ClassVar[str] = "checked"
    check_params: ClassVar[tuple[str, ...]] = ("max_mm",)
    max_mm: float = 3.0

    def check(self, case):
        thickness = float(qc.point_values(case.thickness["FC"].mesh).max())
        return [qc.finding(thickness > self.max_mm, "thickness.too_thick", f"FC is {thickness} mm thick", thickness)]


@dataclass
class CheckedSegmenter(FakeSegmenter):
    """Fails the case whose image has spacing 0.6."""
    name: ClassVar[str] = "checked"

    def check(self, case):
        if case.image.GetSpacing()[0] != 0.5:
            return [QCFinding("fail", "segment.empty", "no femoral cartilage was found in the image")]
        return []


def provenance_of(project, case_id, folder):
    return json.loads((project.case_folder(case_id) / folder / "provenance.json").read_text())


def test_review_flags(project):
    pipeline = make_pipeline()
    pipeline.thickness_estimator = CheckedThickness(thickness=4.0)
    result = project.run(pipeline=pipeline, echo=lambda text: None)
    assert [s.status for s in result] == ["Needs Review", "Needs Review"]
    assert result[0].note == "FC is 4.0 mm thick" and result[0].review == {"thickness": ["FC is 4.0 mm thick"]}
    record = provenance_of(project, "1_00M_LEFT", "05_thickness")
    assert record["qc"] == [{"severity": "review", "code": "thickness.too_thick", "message": "FC is 4.0 mm thick",
                             "value": 4.0}]
    assert record["check_params"] == {"max_mm": 3.0}
    assert len(project.select(project.status(pipeline), needs_review=True)) == 2
    before = started_times(project, "1_00M_LEFT")
    project.run(pipeline=pipeline, echo=lambda text: None)
    assert started_times(project, "1_00M_LEFT") == before  # a case that needs review is finished

    # a new threshold rechecks the outputs without redoing the step
    pipeline.thickness_estimator.max_mm = 5.0
    status = project.status(pipeline)[0]
    assert status.status == "Outdated"
    assert status.note == "thickness_estimator.max_mm changed 3.0 -> 5.0; run again to recheck"
    result = project.run(pipeline=pipeline, echo=lambda text: None)
    assert [s.status for s in result] == ["Succeeded", "Succeeded"]
    after = provenance_of(project, "1_00M_LEFT", "05_thickness")
    assert after["started"] == record["started"] and after["qc"][0]["severity"] == "info"


def test_visits_of_a_knee_are_compared(tmp_path):
    itk.imwrite(ball_image(), str(tmp_path / "a.nrrd"))
    cases = [{"case_id": f"1_{m:02d}M_LEFT", "patient_id": "1", "month": m, "laterality": "left", "input": "a.nrrd"}
             for m in (0, 12, 24, 36)] + [{"case_id": "2_00M_LEFT", "laterality": "left", "input": "a.nrrd"}]
    project = Project.create(tmp_path / "project", cases, {"workers": 0}, data_root=tmp_path)
    pipeline = make_pipeline()
    project.run(pipeline=pipeline, echo=lambda text: None)

    def set_no_thickness(case_id, fc, tc):  # as if the atlas map check had found these
        path = project.case_folder(case_id) / "06_atlas_map" / "provenance.json"
        record = json.loads(path.read_text())
        record["qc"] = [{"severity": "review" if v > 0.15 else "info", "code": "atlas_map.no_thickness",
                         "message": f"{s} {v}", "value": v} for s, v in (("FC", fc), ("TC", tc))]
        path.write_text(json.dumps(record))

    for case_id, fc, tc in (("1_00M_LEFT", 0.20, 0.02), ("1_12M_LEFT", 0.22, 0.03), ("1_24M_LEFT", 0.21, 0.02),
                            ("1_36M_LEFT", 0.21, 0.13), ("2_00M_LEFT", 0.20, 0.02)):
        set_no_thickness(case_id, fc, tc)
    result = {s.case_id: s for s in project.status(pipeline)}
    assert result["1_00M_LEFT"].status == "Succeeded"  # FC flagged at every visit: usual for this knee
    assert result["1_00M_LEFT"].knee_typical[0].startswith("FC 0.2; most of this knee's other visits are the same")
    assert result["1_36M_LEFT"].status == "Needs Review"  # TC far from the knee's other visits
    assert result["1_36M_LEFT"].review == {"atlas_map": ["TC 0.13; this knee's other visits have 2.0% (median), so "
                                                         "this visit either has a processing problem or shows a "
                                                         "real change"]}
    assert result["2_00M_LEFT"].status == "Needs Review"  # no patient_id, so nothing to compare with
    assert project.status(pipeline, case_ids=["1_00M_LEFT"])[0].status == "Succeeded"  # compared with the others
    project.write_index(result.values())
    index = {row["case_id"]: row for row in pq.read_table(project.path / "index.parquet").to_pylist()}
    assert len(index["1_12M_LEFT"]["knee_typical"]) == 1 and len(index["1_36M_LEFT"]["review"]) == 1


def test_fail_finding_fails_the_case(project):
    pipeline = make_pipeline()
    pipeline.segmenter = CheckedSegmenter()
    result = project.run(pipeline=pipeline, echo=lambda text: None)
    assert [s.status for s in result] == ["Succeeded", "Failed"]
    assert result[1].failed_step == "segment"
    assert result[1].note == 'Failed at "segmentation": no femoral cartilage was found in the image'
    assert not (project.case_folder("2_00M_RIGHT") / "03_register").exists()  # the case went no further


def test_older_outputs_get_checked(project):
    project.run(pipeline=make_pipeline(), echo=lambda text: None)
    for path in project.path.glob("cases/**/provenance.json"):  # as if made before the checks existed
        record = json.loads(path.read_text())
        del record["check_params"]
        path.write_text(json.dumps(record))
    status = project.status(make_pipeline())[0]
    assert status.status == "Outdated" and status.note == "not checked yet; run again to check"
    project.run(pipeline=make_pipeline(), echo=lambda text: None)
    assert statuses(project, make_pipeline()) == {"1_00M_LEFT": "Succeeded", "2_00M_RIGHT": "Succeeded"}


def test_thumbnails(project):
    project.run(["1_00M_LEFT"], pipeline=make_pipeline(), echo=lambda text: None)
    folder = project.case_folder("1_00M_LEFT") / "thumbnails"
    assert (folder / "segment_overlay.png").read_bytes().startswith(b"\x89PNG")
    # the tiny atlas has no 2D embedding, so there are no thickness maps; that does not fail the case
    assert not list(folder.glob("atlas_map_*"))
    assert project.status(make_pipeline())[0].status == "Succeeded"


def test_container_provenance(monkeypatch):
    monkeypatch.setenv("OAI_ANALYSIS_IMAGE", "oai-analysis:1.0")
    monkeypatch.setenv("OAI_ANALYSIS_IMAGE_DIGEST", "sha256:abc")
    assert provenance.container() == {"image": "oai-analysis:1.0", "digest": "sha256:abc"}
    monkeypatch.delenv("OAI_ANALYSIS_IMAGE")
    monkeypatch.delenv("OAI_ANALYSIS_IMAGE_DIGEST")
    if not Path("/.dockerenv").exists() and not Path("/run/.containerenv").exists():
        assert provenance.container() is None
