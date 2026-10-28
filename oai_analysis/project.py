"""
A project: a folder with the cases to process, the pipeline config, and every case's outputs and provenance.

    PROJECT/
      project.yaml     pipeline config, data root, retention and storage formats
      cases.csv        one row per case: case_id, patient_id, month, laterality, input
      runs.jsonl       one line per `oai-analysis run`
      index.parquet    one row per case: status, the step that failed, a note
      logs/            the full log and a plain-English summary of each run
      cases/9016304/96M_LEFT/
        case.json                 the input and its hash
        01_preprocess/            image.nrrd, provenance.json
        ...
        07_biomarkers/thickness_map_2d/

A step is done when its provenance.json says so and its fingerprint (see provenance.py) still matches the config
and the outputs of the steps it reads. `run` redoes everything else, so it resumes an interrupted run and redoes
only the steps a config change affects. Cases run in a separate worker process, so a crash in compiled code
fails one case instead of stopping the batch.
"""
import csv
import datetime
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

from oai_analysis import config, io, provenance
from oai_analysis.atlas import Atlas

CASE_COLUMNS = ("case_id", "patient_id", "month", "laterality", "input")
RETENTION = ("checkpoints", "all", "final")
IMAGE_FORMATS = ("nrrd", "nii.gz")
TRANSFORM_FORMATS = ("h5", "tfm")
DEFAULT_SETTINGS = {"data_root": None, "retention": "checkpoints",
                    "storage": {"image_format": "nrrd", "transform_format": "h5"}}
KEPT_BY_FINAL = ("atlas_map", "biomarkers/")  # steps whose files retention: final keeps

# case statuses, in plain language
SUCCEEDED, FAILED, OUTDATED, INCOMPLETE, NOT_STARTED, RUNNING = (
    "Succeeded", "Failed", "Outdated", "Incomplete", "Not started", "Running")
# step states
DONE, STEP_FAILED, STEP_OUTDATED, MISSING = "done", "failed", "outdated", "missing"

HEARTBEAT_SECONDS = 60  # how often a running case updates its running.json
RUNNING_TIMEOUT_SECONDS = 5 * 60  # a running.json older than this was left by a run that stopped
PROGRESS_EVERY_SECONDS = 5 * 60  # how often a long step logs that it is still working
WORKER_PREFIX = "@oai-analysis "  # marks the worker's messages to the supervisor in its output


class DeviceError(RuntimeError):
    pass


@dataclass
class CaseStatus:
    case_id: str
    status: str
    steps: dict[str, str]  # step name -> state
    note: str = ""
    failed_step: str | None = None
    duration_s: float = 0.0  # of the steps that are done
    last_run_s: float = 0.0  # of the steps that are done, run by the latest run that worked on the case
    finished: str | None = None  # when the last step finished


@dataclass
class _CaseRun:
    """What the runner knows about a case while running it."""
    case_id: str
    folder: Path
    input_sha256: str
    records: dict[str, dict] = field(default_factory=dict)  # step name -> provenance of a finished step
    loaded: set[str] = field(default_factory=set)  # steps whose outputs are in memory


class Project:
    def __init__(self, path):
        self.path = Path(path)
        if not (self.path / "project.yaml").is_file():
            raise FileNotFoundError(f"{self.path} is not a project: it has no project.yaml (see `oai-analysis init`)")
        self.settings = {**DEFAULT_SETTINGS, **config.load(self.path / "project.yaml")}
        config.check_keys(self.settings)
        if self.settings["retention"] not in RETENTION:
            raise ValueError(f"retention must be one of {RETENTION}, not {self.settings['retention']!r}")
        storage = {**DEFAULT_SETTINGS["storage"], **(self.settings.get("storage") or {})}
        if storage["image_format"] not in IMAGE_FORMATS or storage["transform_format"] not in TRANSFORM_FORMATS:
            raise ValueError(f"storage formats must be image_format: {IMAGE_FORMATS}, "
                             f"transform_format: {TRANSFORM_FORMATS}, not {storage}")
        self.storage = storage
        self.cases = _read_cases(self.path / "cases.csv")

    # creating a project

    @classmethod
    def create(cls, path, cases=(), pipeline_config=None, data_root=None, retention="checkpoints") -> "Project":
        """
        Create a project folder. cases are dicts with the CASE_COLUMNS; inputs are relative to data_root
        (or to the project folder) unless absolute. pipeline_config is a config dict or YAML file (see config.py).
        """
        from oai_analysis.pipeline import OAPipeline

        path = Path(path)
        if (path / "project.yaml").exists():
            raise FileExistsError(f"{path} is already a project")
        cfg = config.load(pipeline_config) if pipeline_config is not None else {}
        settings = {
            **OAPipeline.from_config(cfg).config(),  # the full config, so every default is written down
            "data_root": str(data_root) if data_root is not None else cfg.get("data_root"),
            "retention": cfg.get("retention", retention),
            "storage": {**DEFAULT_SETTINGS["storage"], **(cfg.get("storage") or {})},
        }
        path.mkdir(parents=True, exist_ok=True)
        (path / "project.yaml").write_text(config.dump(settings), encoding="utf-8")
        _write_cases(path / "cases.csv", list(cases))
        return cls(path)

    @classmethod
    def from_manifest(cls, path, manifest, data_root, patients=None, months=None, laterality=None,
                      pipeline_config=None) -> "Project":
        """Create a project from a knee-sarg manifest (patient_id, study_uid, vol_path, month, study_description)."""
        return cls.create(path, read_manifest(manifest, patients, months, laterality), pipeline_config, data_root)

    # cases and their folders

    def case_folder(self, case_id) -> Path:
        row = self.cases[case_id]
        patient = row.get("patient_id")
        if patient and case_id.startswith(f"{patient}_"):
            return self.path / "cases" / patient / case_id[len(patient) + 1:]
        return self.path / "cases" / case_id

    def case_info(self, case_id):
        from oai_analysis.stages import CaseInfo

        row = self.cases[case_id]
        return CaseInfo(self.input_path(case_id), row["laterality"], case_id)

    def input_path(self, case_id) -> Path:
        path = Path(self.cases[case_id]["input"])
        if path.is_absolute():
            return path
        return Path(self.settings["data_root"] or self.path) / path

    def pipeline(self):
        from oai_analysis.pipeline import OAPipeline

        return OAPipeline.from_config(self.settings)

    # status

    def status(self, pipeline=None, case_ids=None) -> list[CaseStatus]:
        pipeline = pipeline or self.pipeline()
        return [self.case_status(pipeline, case_id) for case_id in (case_ids or self.cases)]

    def case_status(self, pipeline, case_id) -> CaseStatus:
        folder = self.case_folder(case_id)
        steps = pipeline.steps()
        case_record = _read_json(folder / "case.json") or {}
        records, states, notes = {}, {}, {}
        for step in steps:
            record = _read_json(folder / step.folder / "provenance.json")
            if record is None:
                states[step.name] = MISSING
                continue
            records[step.name] = record
            if record["status"] != "complete":
                states[step.name] = STEP_FAILED
                continue
            inputs = _input_hashes(step, case_record.get("input_sha256"), records, pipeline.atlas_name)
            upstream_done = all(states.get(name) == DONE for name in step.inputs)
            if (upstream_done and inputs is not None
                    and provenance.fingerprint(step.stage, inputs) == record["fingerprint"]):
                states[step.name] = DONE
            else:
                states[step.name] = STEP_OUTDATED
                notes[step.name] = _what_changed(step, record, upstream_done)

        done_records = [records[name] for name, state in states.items() if state == DONE]
        latest = max(done_records, key=lambda r: r["finished"], default={})
        result = CaseStatus(case_id, NOT_STARTED, states,
                            duration_s=round(sum(r.get("duration_s", 0) for r in done_records), 1),
                            last_run_s=round(sum(r.get("duration_s", 0) for r in done_records
                                                 if r.get("run_id") == latest.get("run_id")), 1),
                            finished=latest.get("finished"))
        described = {s.name: s.description for s in steps}
        if _is_running(folder):
            marker = _read_json(folder / "running.json")
            result.status, result.note = RUNNING, f"Running {marker.get('step') or ''}".strip()
            return result
        failed = [name for name, state in states.items() if state == STEP_FAILED]
        outdated = [name for name, state in states.items() if state == STEP_OUTDATED]
        if failed:
            result.status, result.failed_step = FAILED, failed[0]
            result.note = f'Failed at "{described[failed[0]]}": {records[failed[0]].get("error", "unknown error")}'
        elif outdated:
            result.status, result.note = OUTDATED, notes[outdated[0]]
        elif all(state == DONE for state in states.values()):
            result.status = SUCCEEDED
        elif any(state == DONE for state in states.values()):
            missing = next(name for name, state in states.items() if state == MISSING)
            result.status, result.note = INCOMPLETE, f'Stopped before "{described[missing]}"; run again to finish'
        return result

    def select(self, statuses, failed=False, stale=False, failed_at=None) -> list[CaseStatus]:
        """The statuses matching the selectors; with none given, all of them."""
        wanted = set()
        if failed:
            wanted.add(FAILED)
        if stale:
            wanted.add(OUTDATED)
        selected = [s for s in statuses if not wanted or s.status in wanted]
        if failed_at:
            selected = [s for s in selected if s.failed_step == failed_at]
        return selected

    def write_index(self, statuses):
        import pyarrow as pa
        import pyarrow.parquet as pq

        rows = []
        for s in statuses:
            row = self.cases[s.case_id]
            rows.append({"case_id": s.case_id, "patient_id": row.get("patient_id") or None,
                         "month": int(row["month"]) if row.get("month") else None,
                         "laterality": row["laterality"], "status": s.status, "failed_step": s.failed_step,
                         "note": s.note, **{f"step_{name}": state for name, state in s.steps.items()},
                         "duration_s": s.duration_s, "finished": s.finished})
        temporary = self.path / "index.parquet.tmp"
        pq.write_table(pa.Table.from_pylist(rows), temporary)
        os.replace(temporary, self.path / "index.parquet")

    # running

    def run(self, case_ids=None, from_step=None, pipeline=None, in_process=False, echo=None) -> list[CaseStatus]:
        """
        Process the cases (default: all) that are not up to date, skipping the steps that are.
        from_step reruns that step and every step that depends on it. A separate worker process runs the cases
        unless in_process is set; a pipeline object (rather than project.yaml) can only be used in process.
        """
        echo = echo or (lambda text: print(text, flush=True))
        if pipeline is not None and not in_process:
            raise ValueError("A pipeline object can only be used with in_process=True")
        pipeline = pipeline or self.pipeline()
        case_ids = list(case_ids or self.cases)
        unknown = [c for c in case_ids if c not in self.cases]
        if unknown:
            raise KeyError(f"Not in cases.csv: {', '.join(unknown)}")
        if from_step is not None:
            _steps_from(pipeline, from_step)  # check the name

        statuses = {s.case_id: s for s in self.status(pipeline)}
        todo = [c for c in case_ids if statuses[c].status != RUNNING
                and (from_step is not None or statuses[c].status != SUCCEEDED)]
        running = [c for c in case_ids if statuses[c].status == RUNNING]

        run_id = datetime.datetime.now().strftime("%Y-%m-%dT%H%M%S")
        log = _RunLog(self.path / "logs", run_id, echo)
        log.summary(f"Run started {datetime.datetime.now():%Y-%m-%d %H:%M} on {_host()}, "
                    f"oai-analysis {provenance.software()['oai-analysis']}, config project.yaml")
        log.summary(f"{_cases(len(case_ids))} selected ({len(todo)} to do, "
                    f"{len(case_ids) - len(todo) - len(running)} already done"
                    + (f", {len(running)} running in another process" if running else "") + ").")
        with open(self.path / "runs.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"run_id": run_id, "started": provenance.now(), "command": " ".join(sys.argv),
                                "cases": len(case_ids), "to_do": len(todo), "from_step": from_step,
                                "software": provenance.software()}) + "\n")

        def case_finished(case_id):
            statuses[case_id] = status = self.case_status(pipeline, case_id)
            log.summary(_summary_line(status))
            self.write_index(statuses.values())

        try:
            if in_process:
                self.work(pipeline, todo, from_step, run_id, log.progress, case_finished)
            else:
                self._supervise(todo, from_step, run_id, log, case_finished)
        except KeyboardInterrupt:
            log.summary("Run stopped by the user. Run again to continue where it stopped.")
            raise
        finally:
            counts = {}
            for case_id in todo:
                counts[statuses[case_id].status] = counts.get(statuses[case_id].status, 0) + 1
            log.summary(f"Run finished {datetime.datetime.now():%H:%M}: "
                        + (", ".join(f"{n} {status}" for status, n in counts.items()) or "nothing to do")
                        + f". Details: logs/{run_id}.log")
            self.write_index(statuses.values())
            log.close()
        return [statuses[c] for c in case_ids]

    def work(self, pipeline, case_ids, from_step, run_id, progress, case_finished):
        """Run the cases one after another, in this process. A failing step fails its case, not the batch."""
        pipeline.device = check_device(pipeline.device)
        pipeline.setup()
        rerun = _steps_from(pipeline, from_step) if from_step is not None else set()
        heartbeat = _Heartbeat(progress)
        heartbeat.start()
        try:
            for case_id in case_ids:
                self._run_case(pipeline, case_id, rerun, run_id, progress, heartbeat)
                case_finished(case_id)
        finally:
            heartbeat.stop()

    def _run_case(self, pipeline, case_id, rerun, run_id, progress, heartbeat):
        from oai_analysis.stages import CaseArtifacts

        folder = self.case_folder(case_id)
        folder.mkdir(parents=True, exist_ok=True)
        case = self.case_info(case_id)
        case_record = _read_json(folder / "case.json") or {}
        if case_record.get("input") != str(case.input) or not case_record.get("input_sha256"):
            progress(f"{case_id}: hashing the input")
            case_record = {**self.cases[case_id], "input": str(case.input),
                           "input_sha256": provenance.sha256_path(case.input)}
            _write_json(folder / "case.json", case_record)
        state = _CaseRun(case_id, folder, case_record["input_sha256"])
        heartbeat.watch(folder / "running.json", case_id, run_id)
        result = CaseArtifacts(case=case, atlas=pipeline.atlas)
        steps = {s.name: s for s in pipeline.steps()}

        def ensure_loaded(name) -> bool:
            """Put a step's outputs in memory: read them, or run the step again if they were deleted."""
            if name in state.loaded:
                return True
            step = steps[name]
            files = {out: folder / step.folder / o["path"] for out, o in state.records[name]["outputs"].items()}
            if all(path.is_file() for path in files.values()):
                pipeline.load_step(step, result, files)
                state.loaded.add(name)
                return True
            return run_step(step)

        def run_step(step) -> bool:
            if not all(ensure_loaded(name) for name in step.inputs):
                return False
            inputs = _input_hashes(step, state.input_sha256, state.records, pipeline.atlas_name)
            record = self._new_record(step, pipeline, inputs, state.records, run_id)
            step_folder = folder / step.folder
            shutil.rmtree(step_folder, ignore_errors=True)
            step_folder.mkdir(parents=True)
            progress(f"{case_id}: {step.description}")
            heartbeat.set_step(step)
            start = time.perf_counter()
            try:
                outputs = pipeline.run_step(step, result)
                record["outputs"] = self._save(step, outputs.outputs, step_folder)
                if self.settings["retention"] == "all":
                    record["intermediates"] = self._save(step, outputs.intermediates, step_folder)
                record["status"] = "complete"
            except Exception as e:
                record.update(status="failed", error=f"{type(e).__name__}: {e}".strip(),
                              traceback=traceback.format_exc())
                print(record["traceback"], flush=True)
            record["finished"] = provenance.now()
            record["duration_s"] = round(time.perf_counter() - start, 1)
            _write_json(step_folder / "provenance.json", record)
            if record["status"] != "complete":
                return False
            state.records[step.name] = record
            state.loaded.add(step.name)
            return True

        succeeded = True
        for step in steps.values():
            record = _read_json(folder / step.folder / "provenance.json")
            inputs = _input_hashes(step, state.input_sha256, state.records, pipeline.atlas_name)
            if (step.name not in rerun and record is not None and record["status"] == "complete"
                    and inputs is not None and provenance.fingerprint(step.stage, inputs) == record["fingerprint"]):
                state.records[step.name] = record  # up to date
                continue
            if not run_step(step):
                succeeded = False
                break
        if succeeded and self.settings["retention"] == "final":
            self._prune(steps.values(), state)
        heartbeat.watch(None)
        (folder / "running.json").unlink(missing_ok=True)

    def _new_record(self, step, pipeline, inputs, records, run_id) -> dict:
        input_paths = {f"{name}/{out}": f"{records[name]['folder']}/{o['path']}"  # relative to the case folder
                       for name in step.inputs for out, o in records[name]["outputs"].items()}
        return {
            "step": step.name,
            "folder": step.folder,
            "implementation": step.stage.name,
            "implementation_version": step.stage.version,
            "status": "running",
            "params": config.stage_to_dict(step.stage),
            "weights": step.stage.weights(),
            "inputs": {key: {"sha256": sha, **({"path": input_paths[key]} if key in input_paths else {})}
                       for key, sha in inputs.items()},
            "fingerprint": provenance.fingerprint(step.stage, inputs),
            "software": provenance.software(),
            "environment": provenance.environment(pipeline.device),
            "run_id": run_id,
            "qc": [],
            "started": provenance.now(),
        }

    def _save(self, step, values, step_folder) -> dict:
        """Write outputs into the step folder; return their file names and hashes."""
        saved = {}
        for name, value in values.items():
            if step.stage.kind == "biomarker_extractor":
                filename = name  # biomarker outputs are named with their file names
            else:
                filename = name + io.extension(value, "." + self.storage["image_format"],
                                               "." + self.storage["transform_format"])
            path = step_folder / filename
            io.write(value, path, compression=True)
            saved[name] = {"path": filename, "sha256": provenance.sha256_file(path)}
        return saved

    def _prune(self, steps, state):
        """retention: final keeps only the atlas-mapped meshes, the biomarkers and the provenance."""
        for step in steps:
            if step.name.startswith(KEPT_BY_FINAL):
                continue
            for path in (state.folder / step.folder).iterdir():
                if path.name != "provenance.json":
                    path.unlink()

    def _supervise(self, case_ids, from_step, run_id, log, case_finished):
        """Run the cases in a worker process; after a crash, fail the case it was on and start a new worker."""
        remaining = list(case_ids)
        cases_file = self.path / "logs" / f"{run_id}.cases.txt"
        while remaining:
            cases_file.write_text("\n".join(remaining), encoding="utf-8")
            command = [sys.executable, "-u", "-m", "oai_analysis.project", "worker", str(self.path),
                       "--run-id", run_id, "--cases-file", str(cases_file)]
            if from_step is not None:
                command += ["--from", from_step]
            worker = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                      encoding="utf-8", errors="replace",
                                      env={**os.environ, "MPLBACKEND": "Agg"})  # no GUI toolkit in the worker
            try:
                for line in worker.stdout:
                    if line.startswith(WORKER_PREFIX):
                        kind, _, text = line[len(WORKER_PREFIX):].rstrip("\n").partition(" ")
                        if kind == "progress":
                            log.progress(text)
                        elif kind == "done":
                            remaining.remove(text)
                            case_finished(text)
                        elif kind == "error":
                            log.summary(text)
                    else:
                        log.write(line)
                exit_code = worker.wait()
            except BaseException:
                worker.kill()
                raise
            if not remaining or exit_code == 0:
                break
            crashed = remaining[0]
            marker = _read_json(self.case_folder(crashed) / "running.json")
            if marker is None or marker.get("run_id") != run_id:
                log.summary(f"The worker process stopped (exit code {_exit_code(exit_code)}) before processing "
                            f"{crashed}. See logs/{run_id}.log.")
                break
            self._record_crash(crashed, marker, exit_code)
            remaining.pop(0)
            case_finished(crashed)
            if remaining:
                log.summary(f"Starting a new worker process for the remaining {_cases(len(remaining))}.")
        cases_file.unlink(missing_ok=True)

    def _record_crash(self, case_id, marker, exit_code):
        """Write a failed provenance for the step a crashed worker was running."""
        folder = self.case_folder(case_id)
        step_folder = folder / marker["step_folder"] if marker.get("step_folder") else None
        message = f"The process crashed (exit code {_exit_code(exit_code)}), e.g. inside ITK or VTK code"
        if step_folder is not None:
            record = _read_json(step_folder / "provenance.json")
            if record is None or record["status"] == "running":
                record = {**(record or {}), "step": marker.get("step_name"), "status": "failed", "error": message,
                          "finished": provenance.now()}
                step_folder.mkdir(parents=True, exist_ok=True)
                _write_json(step_folder / "provenance.json", record)
        (folder / "running.json").unlink(missing_ok=True)


def check_device(device) -> str:
    """The device to run on. Refuses to fall back to the CPU when a GPU was asked for, or none was named."""
    import torch

    if device in (None, "auto"):
        if torch.cuda.is_available():
            return "cuda"
        raise DeviceError("No CUDA GPU is available. To run on the CPU (much slower), set `device: cpu` "
                          "in project.yaml.")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise DeviceError(f"project.yaml asks for device {device}, but no CUDA GPU is available.")
    return device


def read_manifest(manifest, patients=None, months=None, laterality=None) -> list[dict]:
    """Cases from a knee-sarg manifest CSV (patient_id, study_uid, vol_path, month, study_description)."""
    patients = {str(p) for p in patients} if patients else None
    months = {int(m) for m in months} if months else None
    cases = []
    with open(manifest, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            side = row["study_description"].rsplit("^", 1)[-1].lower()
            month = int(row["month"])
            if ((patients and row["patient_id"] not in patients) or (months is not None and month not in months)
                    or (laterality and side != laterality) or side not in ("left", "right")):
                continue
            cases.append({"case_id": f"{row['patient_id']}_{month:02d}M_{side.upper()}",
                          "patient_id": row["patient_id"], "month": month, "laterality": side,
                          "input": row["vol_path"]})
    return cases


def _steps_from(pipeline, from_step) -> set[str]:
    """The step and every step that reads its outputs, directly or not."""
    steps = pipeline.steps()
    names = [s.name for s in steps]
    if from_step not in names:
        raise ValueError(f"No step named {from_step!r}. Steps: {', '.join(names)}")
    rerun = {from_step}
    for step in steps:  # steps are in order, so one pass finds everything downstream
        if rerun & set(step.inputs):
            rerun.add(step.name)
    return rerun


def _input_hashes(step, input_sha256, records, atlas_name) -> dict[str, str] | None:
    """The hashes a step's fingerprint covers; None while a step it reads has no outputs."""
    inputs = {"atlas": f"{atlas_name}:{provenance.atlas_sha256(atlas_name)}"}
    if not step.inputs:
        if input_sha256 is None:
            return None
        inputs["input"] = input_sha256
    for name in step.inputs:
        record = records.get(name)
        if record is None or record["status"] != "complete":
            return None
        inputs.update({f"{name}/{out}": o["sha256"] for out, o in record["outputs"].items()})
    return inputs


def _what_changed(step, record, upstream_done) -> str:
    """Why a finished step is no longer up to date, in plain words."""
    stage = step.stage
    if record["implementation"] != stage.name:
        return f"{stage.kind} changed {record['implementation']} -> {stage.name}"
    if record["implementation_version"] != stage.version:
        return f"{stage.name} was updated to version {stage.version}"
    old_params = {k: v for k, v in record["params"].items() if k != "name"}
    new_params = config.stage_to_dict(stage)
    changed = [f"{stage.kind}.{k} changed {old_params.get(k)} -> {v}" for k, v in new_params.items()
               if k != "name" and old_params.get(k) != v]
    if changed:
        return "; ".join(changed)
    if record.get("weights") != stage.weights():
        return f"{stage.name} uses different weights"
    if not upstream_done:
        return f'redo after an earlier step ("{step.description}" reads its outputs)'
    return "its inputs changed"


def _summary_line(status) -> str:
    time_text = datetime.datetime.now().strftime("%H:%M")
    if status.status == SUCCEEDED:
        if status.last_run_s == status.duration_s:
            return f"[{time_text}] {status.case_id:<20} Succeeded in {_duration(status.duration_s)}."
        return (f"[{time_text}] {status.case_id:<20} Succeeded in {_duration(status.last_run_s)} "
                f"({_duration(status.duration_s)} including earlier runs).")
    if status.status == FAILED:
        return (f"[{time_text}] {status.case_id:<20} {status.note}. Earlier steps are kept; rerun with "
                f"--cases {status.case_id} --from {status.failed_step}")
    return f"[{time_text}] {status.case_id:<20} {status.status}. {status.note}"


def _cases(count) -> str:
    return f"{count} case" if count == 1 else f"{count} cases"


def _duration(seconds) -> str:
    minutes, seconds = divmod(int(seconds), 60)
    return f"{minutes}m{seconds:02d}s" if minutes else f"{seconds}s"


def _exit_code(code) -> str:
    return f"0x{code & 0xFFFFFFFF:08X}" if code < 0 or code > 255 else str(code)


def _host():
    import platform

    return platform.node()


def _is_running(folder) -> bool:
    """Whether a worker is processing the case: its marker is fresh, and its process is alive if it is on this host."""
    marker = _read_json(folder / "running.json")
    if marker is None or time.time() - marker.get("heartbeat", 0) >= RUNNING_TIMEOUT_SECONDS:
        return False
    return marker.get("host") != _host() or _process_alive(marker.get("pid"))


def _process_alive(pid) -> bool:
    if not pid:
        return False
    if sys.platform == "win32":
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        exit_code = ctypes.c_ulong()
        kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
        kernel32.CloseHandle(handle)
        return exit_code.value == 259  # STILL_ACTIVE
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _read_json(path) -> dict | None:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _write_json(path, value):
    """Write the file completely or not at all, so a killed run never leaves half a provenance record."""
    temporary = Path(f"{path}.tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")
    os.replace(temporary, path)


def _read_cases(path) -> dict[str, dict]:
    cases = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row = {k: (v or "").strip() for k, v in row.items()}
            if row["case_id"] in cases:
                raise ValueError(f"{path}: case_id {row['case_id']} appears twice")
            if row["laterality"] not in ("left", "right"):
                raise ValueError(f"{path}: case {row['case_id']} has laterality {row['laterality']!r}")
            cases[row["case_id"]] = row
    return cases


def _write_cases(path, cases):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CASE_COLUMNS)
        writer.writeheader()
        for case in cases:
            writer.writerow({column: case.get(column, "") for column in CASE_COLUMNS})


class _RunLog:
    """logs/RUN_ID.log gets everything; logs/RUN_ID.summary.txt and the console get the plain-English lines."""

    def __init__(self, folder, run_id, echo):
        folder.mkdir(parents=True, exist_ok=True)
        self.echo = echo
        self.log = open(folder / f"{run_id}.log", "a", encoding="utf-8")
        self.summary_file = open(folder / f"{run_id}.summary.txt", "a", encoding="utf-8")

    def write(self, text):
        self.log.write(text)
        self.log.flush()

    def progress(self, text):
        self.write(f"[{datetime.datetime.now():%H:%M:%S}] {text}\n")
        self.echo(f"  {text}")

    def summary(self, text):
        self.write(text + "\n")
        self.summary_file.write(text + "\n")
        self.summary_file.flush()
        self.echo(text)

    def close(self):
        self.log.close()
        self.summary_file.close()


class _Heartbeat(threading.Thread):
    """Keeps the running case's running.json fresh, and says when a step takes long."""

    def __init__(self, progress):
        super().__init__(daemon=True)
        self.progress = progress
        self.marker = None
        self.step, self.step_details = None, {}
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

    def watch(self, marker, case_id=None, run_id=None):
        with self._lock:
            self.marker, self.case_id, self.run_id = marker, case_id, run_id
            self.step, self.step_details = None, {}
            self.step_started = self.last_progress = time.time()
            self._beat()

    def _beat(self):
        if self.marker is not None:
            _write_json(self.marker, {"pid": os.getpid(), "host": _host(), "run_id": self.run_id,
                                      "case_id": self.case_id, "step": self.step, "heartbeat": time.time(),
                                      **self.step_details})

    def set_step(self, step):
        with self._lock:
            self.step = step.description
            self.step_details = {"step_name": step.name, "step_folder": step.folder}
            self.step_started = self.last_progress = time.time()
            self._beat()

    def run(self):
        while not self._stop_event.wait(HEARTBEAT_SECONDS):
            with self._lock:
                self._beat()
                if self.marker is not None and time.time() - self.last_progress >= PROGRESS_EVERY_SECONDS:
                    minutes = (time.time() - self.step_started) / 60
                    self.progress(f"{self.case_id}: still working on {self.step} ({minutes:.0f} min)")
                    self.last_progress = time.time()

    def stop(self):
        self._stop_event.set()


def _worker_main(argv):
    """The worker process started by Project.run(): `python -m oai_analysis.project worker PROJECT ...`."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["worker"])
    parser.add_argument("project")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--cases-file", required=True)
    parser.add_argument("--from", dest="from_step")
    args = parser.parse_args(argv)

    def say(kind, text):
        print(f"{WORKER_PREFIX}{kind} {text}", flush=True)

    project = Project(args.project)
    case_ids = Path(args.cases_file).read_text(encoding="utf-8").split()
    try:
        project.work(project.pipeline(), case_ids, args.from_step, args.run_id,
                     lambda text: say("progress", text), lambda case_id: say("done", case_id))
    except DeviceError as e:
        say("error", str(e))
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(_worker_main(sys.argv[1:]))
