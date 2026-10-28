"""
Link a project's outputs into the flat layout of analysis_pipeline(), one directory per case,
so tools/compare_outputs.py can compare them with a flat reference such as the Step 0 golden outputs.

    python tools/export_flat.py PROJECT OUT_DIR

Case directories are named like the benchmark cases (9016304_96M_left). Files are hard-linked when possible.
Transforms are not exported; compare_outputs skips them anyway.
"""
import argparse
import os
import shutil
import sys
from pathlib import Path

from oai_analysis.project import Project

# (step folder, project file name) -> flat file name; {s} is the structure
FLAT_NAMES = {
    ("01_preprocess", "image"): "in_image",
    ("02_segment", "{s}_prob"): "{s}_prob",
    ("04_mesh", "{s}_mesh"): "{s}_mesh_patient",
    ("05_thickness", "{s}_thickness"): "{s}_outer",
    ("05_thickness", "{s}_inner"): "{s}_inner",
    ("06_atlas_map", "{s}_atlas"): "{s}_mapped_mesh",
    ("06_atlas_map", "{s}_mesh_atlas"): "{s}_mesh_atlas",
}


def flat_case_name(row):
    if row.get("patient_id") and row.get("month"):
        return f"{row['patient_id']}_{int(row['month'])}M_{row['laterality']}"
    return row["case_id"]


def link(source, target):
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("project", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    project = Project(args.project)
    for case_id, row in project.cases.items():
        folder = project.case_folder(case_id)
        if not folder.is_dir():
            continue
        out = args.output_dir / flat_case_name(row)
        out.mkdir(parents=True, exist_ok=True)
        for (step_folder, name), flat_name in FLAT_NAMES.items():
            for s in ("FC", "TC"):
                for source in (folder / step_folder).glob(name.format(s=s) + ".*"):
                    suffix = source.name[len(name.format(s=s)):]
                    target = out / (flat_name.format(s=s) + suffix)
                    if not target.exists():
                        link(source, target)
        for source in (folder / "07_biomarkers").glob("*/*"):
            if source.name != "provenance.json" and not (out / source.name).exists():
                link(source, out / source.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
