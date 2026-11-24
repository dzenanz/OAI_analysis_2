"""
Add the measures of the checks to the provenance of a project processed before they were recorded, so that new
thresholds recheck its cases from provenance.json alone instead of redoing the steps whose outputs retention:
final deleted. The measures are read back from the QC findings recorded with each step. A record is changed only
when judging its new measures with the project's thresholds gives exactly its recorded findings.

    python tools/record_measures.py PROJECT

Run it while no `oai-analysis run` works on the project, before changing any threshold in project.yaml.
"""
import json
import re
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from oai_analysis.project import Project, _check_params, _write_json

STRUCTURES = {"femoral cartilage": "FC", "tibial cartilage": "TC"}


def structure(message):
    return next(code for name, code in STRUCTURES.items() if name in message)


def measures_from_qc(step, record):
    """The measures of the built-in stages' checks, from the findings recorded with them."""
    qc = record["qc"]
    if step == "preprocess":
        spacing = next(f["message"] for f in qc if f["code"] == "preprocess.spacing")
        size = next(f["message"] for f in qc if f["code"] == "preprocess.size")
        return {"axes": None, "atlas_axes": None,  # not recorded; a finished case had the atlas orientation
                "spacing_mm": [float(s) for s in re.search(r"spacing is ([\d. x]+) mm", spacing)[1].split(" x ")],
                "size": [int(s) for s in re.search(r"has ([\d x]+) voxels", size)[1].split(" x ")],
                "median_intensity": next(f["value"] for f in qc if f["code"] == "preprocess.median_intensity")}
    if step == "segment":
        names = {"segment.volume": "volume_ml", "segment.pieces": "outside_main_pieces",
                 "segment.uncertainty": "uncertainty"}
        measures = {}
        for f in qc:
            measures.setdefault(structure(f["message"]), {})[names[f["code"]]] = f["value"]
        return measures
    if step == "register":
        return {f["code"].split(".")[1]: f["value"] for f in qc}
    if step == "thickness":
        return {"max_thickness_mm": record["check_params"]["max_thickness_mm"],
                "too_thick": {structure(f["message"]): f["value"] for f in qc}}
    if step == "atlas_map":
        return {"no_thickness": {structure(f["message"]): f["value"] for f in qc}}
    return None


def main(path):
    project = Project(path)
    steps = {step.name: step for step in project.pipeline().steps()}
    outcome = Counter()
    for file in sorted(project.path.glob("cases/*/*/*/provenance.json")):
        record = json.loads(file.read_text(encoding="utf-8"))
        step = steps.get(record.get("step"))
        if step is None or record.get("status") != "complete" or "measures" in record or "check_params" not in record:
            continue
        if record["check_params"] != _check_params(step.stage):
            outcome["skipped: thresholds differ from project.yaml"] += 1
            continue
        try:
            measures = measures_from_qc(step.name, record)
        except (StopIteration, KeyError, TypeError) as e:
            outcome[f"skipped: findings not understood ({type(e).__name__})"] += 1
            continue
        if measures is None:
            continue
        findings = step.stage.judge(measures)
        if findings is None or [asdict(f) for f in findings] != record["qc"]:
            outcome[f"skipped {step.name}: judging the measures does not give the recorded findings"] += 1
            print(f"  {file.parent}: recorded {record['qc']}, judged {findings}")
            continue
        record["measures"] = measures
        _write_json(file, record)
        outcome[f"recorded {step.name}"] += 1
    for what, n in sorted(outcome.items()):
        print(f"{n:7d}  {what}")


if __name__ == "__main__":
    main(sys.argv[1])
