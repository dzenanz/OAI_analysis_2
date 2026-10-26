"""
Compare two pipeline output directories case by case, and report pass/fail against tolerances.

Usage:
    python tools/compare_outputs.py REFERENCE_DIR TEST_DIR [--tolerances FILE] [--report FILE.md]

Both directories hold one subdirectory per case (e.g. 9016304_0M_left), in the flat layout written by
pipeline.py / tools/run_baseline.py. Compared files:
    images (.nrrd, .nii.gz): geometry, max/mean absolute difference, Dice at 0.5 for probability maps
    meshes (.vtk): vertex count, closest-vertex distance between surfaces, difference of the point scalars
    2D thickness maps (.npy): NaN-mask agreement, mean/95th-percentile/max absolute difference
Transforms (.tfm) and images (.png) are skipped: registration differences show up in the atlas-space meshes.

Tolerances are a JSON object mapping a file name pattern to metric limits; the first matching pattern applies.
The exit code is 1 if any case fails, so the tool can gate CI.
"""
import argparse
import fnmatch
import json
import pathlib
import sys

import itk
import numpy as np
import vtk
from scipy.spatial import cKDTree
from vtk.util.numpy_support import vtk_to_numpy

DEFAULT_TOLERANCES_FILE = pathlib.Path(__file__).resolve().parent / "regression_tolerances.json"
IMAGE_SUFFIXES = (".nrrd", ".nii.gz", ".nii")
SKIPPED_SUFFIXES = (".tfm", ".png")


def file_kind(name):
    if name.endswith(IMAGE_SUFFIXES):
        return "image"
    if name.endswith(".vtk"):
        return "mesh"
    if name.endswith(".npy"):
        return "map2d"
    return None


def compare_images(ref_path, test_path):
    ref = itk.imread(str(ref_path), itk.F)
    test = itk.imread(str(test_path), itk.F)
    geometry_match = (
        tuple(ref.GetLargestPossibleRegion().GetSize()) == tuple(test.GetLargestPossibleRegion().GetSize())
        and np.allclose(ref.GetSpacing(), test.GetSpacing())
        and np.allclose(ref.GetOrigin(), test.GetOrigin(), atol=1e-4)
        and np.allclose(itk.array_from_matrix(ref.GetDirection()), itk.array_from_matrix(test.GetDirection()))
    )
    metrics = {"geometry_match": geometry_match}
    if not geometry_match:
        return metrics
    ref_array = itk.array_view_from_image(ref)
    test_array = itk.array_view_from_image(test)
    diff = np.abs(ref_array - test_array)
    metrics["max_abs_diff"] = float(diff.max())
    metrics["mean_abs_diff"] = float(diff.mean())
    if "prob" in ref_path.name:
        ref_mask = ref_array >= 0.5
        test_mask = test_array >= 0.5
        total = ref_mask.sum() + test_mask.sum()
        metrics["dice"] = float(2.0 * np.logical_and(ref_mask, test_mask).sum() / total) if total else 1.0
    return metrics


def read_mesh(path):
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    mesh = reader.GetOutput()
    points = vtk_to_numpy(mesh.GetPoints().GetData()) if mesh.GetNumberOfPoints() else np.zeros((0, 3))
    scalars = mesh.GetPointData().GetArray(0)
    scalars = vtk_to_numpy(scalars).astype(np.float64) if scalars is not None else None
    if scalars is not None and scalars.ndim > 1:
        scalars = None  # only scalar arrays (thickness, distance) are compared
    return points, scalars


def compare_meshes(ref_path, test_path):
    ref_points, ref_scalars = read_mesh(ref_path)
    test_points, test_scalars = read_mesh(test_path)
    metrics = {
        "points_ref": len(ref_points),
        "points_test": len(test_points),
        "points_rel_diff": abs(len(test_points) - len(ref_points)) / max(len(ref_points), 1),
    }
    if len(ref_points) == 0 or len(test_points) == 0:
        metrics["surface_max_dist"] = 0.0 if len(ref_points) == len(test_points) else float("inf")
        return metrics

    test_to_ref, nearest_ref = cKDTree(ref_points).query(test_points)
    ref_to_test, _ = cKDTree(test_points).query(ref_points)
    metrics["surface_mean_dist"] = float((test_to_ref.mean() + ref_to_test.mean()) / 2)
    metrics["surface_max_dist"] = float(max(test_to_ref.max(), ref_to_test.max()))

    if ref_scalars is not None and test_scalars is not None:
        if len(ref_points) == len(test_points) and np.allclose(ref_points, test_points):
            # same vertices (e.g. thickness mapped onto the atlas mesh): compare vertex by vertex
            scalar_diff = np.abs(ref_scalars - test_scalars)
        else:
            scalar_diff = np.abs(ref_scalars[nearest_ref] - test_scalars)
        metrics["scalar_mean_abs_diff"] = float(scalar_diff.mean())
        metrics["scalar_p95_abs_diff"] = float(np.percentile(scalar_diff, 95))
    return metrics


def compare_maps(ref_path, test_path):
    ref = np.load(ref_path)
    test = np.load(test_path)
    if ref.shape != test.shape:
        return {"shape_match": False}
    ref_nan = np.isnan(ref)
    test_nan = np.isnan(test)
    valid = ~ref_nan & ~test_nan
    metrics = {
        "shape_match": True,
        "nan_mismatch_frac": float(np.logical_xor(ref_nan, test_nan).sum() / max((~ref_nan).sum(), 1)),
    }
    if valid.any():
        diff = np.abs(ref[valid] - test[valid])
        metrics["mean_abs_diff"] = float(diff.mean())
        metrics["p95_abs_diff"] = float(np.percentile(diff, 95))
        metrics["max_abs_diff"] = float(diff.max())
    return metrics


COMPARE = {"image": compare_images, "mesh": compare_meshes, "map2d": compare_maps}


def check_limits(metrics, limits):
    """Return the list of violated limits. Booleans must be True; numbers must be within [min, max]."""
    violations = []
    for key, value in metrics.items():
        if isinstance(value, bool):
            if not value:
                violations.append(f"{key} is false")
            continue
        if f"{key}_max" in limits and value > limits[f"{key}_max"]:
            violations.append(f"{key}={value:.4g} > {limits[f'{key}_max']}")
        if f"{key}_min" in limits and value < limits[f"{key}_min"]:
            violations.append(f"{key}={value:.4g} < {limits[f'{key}_min']}")
    return violations


def limits_for(name, tolerances):
    for pattern, limits in tolerances.items():
        if fnmatch.fnmatch(name, pattern):
            return limits
    return {}


def format_metrics(metrics):
    parts = []
    for key, value in metrics.items():
        parts.append(f"{key}={value:.4g}" if isinstance(value, float) else f"{key}={value}")
    return ", ".join(parts)


def compare_dirs(ref_dir, test_dir, tolerances, only_common=False):
    rows = []
    for ref_case in sorted(p for p in ref_dir.iterdir() if p.is_dir()):
        test_case = test_dir / ref_case.name
        if not test_case.is_dir():
            if not only_common:
                rows.append((ref_case.name, "", "FAIL", "case missing in test directory", {}))
            continue
        ref_files = {p.name for p in ref_case.iterdir() if p.is_file()}
        test_files = {p.name for p in test_case.iterdir() if p.is_file()}
        for name in sorted(ref_files - test_files):
            if not name.endswith(SKIPPED_SUFFIXES):
                rows.append((ref_case.name, name, "FAIL", "file missing in test directory", {}))
        for name in sorted(ref_files & test_files):
            kind = file_kind(name)
            if kind is None:
                continue
            print(f"  {ref_case.name}/{name}", file=sys.stderr, flush=True)
            metrics = COMPARE[kind](ref_case / name, test_case / name)
            violations = check_limits(metrics, limits_for(name, tolerances))
            rows.append((ref_case.name, name, "FAIL" if violations else "pass", "; ".join(violations), metrics))
    return rows


def write_report(rows, ref_dir, test_dir, out):
    failed_cases = sorted({case for case, _, status, _, _ in rows if status == "FAIL"})
    all_cases = sorted({case for case, *_ in rows})
    out.write(f"# Output comparison\n\nReference: `{ref_dir}`  \nTest: `{test_dir}`\n\n")
    out.write(f"**{len(all_cases) - len(failed_cases)} of {len(all_cases)} cases pass.**")
    out.write(f" Failing: {', '.join(failed_cases)}\n\n" if failed_cases else "\n\n")
    out.write("| case | file | result | metrics | violations |\n|---|---|---|---|---|\n")
    for case, name, status, violations, metrics in rows:
        out.write(f"| {case} | {name} | {status} | {format_metrics(metrics)} | {violations} |\n")
    return failed_cases


def main():
    parser = argparse.ArgumentParser(description="Compare two pipeline output directories")
    parser.add_argument("reference_dir", type=pathlib.Path)
    parser.add_argument("test_dir", type=pathlib.Path)
    parser.add_argument("--tolerances", type=pathlib.Path, default=DEFAULT_TOLERANCES_FILE)
    parser.add_argument("--report", type=pathlib.Path, help="Write the Markdown report here instead of stdout")
    parser.add_argument("--metrics-json", type=pathlib.Path, help="Also dump all raw metrics as JSON")
    parser.add_argument("--only-common", action="store_true", help="Ignore cases missing from the test directory")
    args = parser.parse_args()

    tolerances = {}
    if args.tolerances.exists():
        with open(args.tolerances) as f:
            tolerances = json.load(f)
        tolerances.pop("_comment", None)

    rows = compare_dirs(args.reference_dir, args.test_dir, tolerances, args.only_common)
    if args.metrics_json:
        with open(args.metrics_json, "w") as f:
            json.dump([{"case": c, "file": n, "status": s, "metrics": m} for c, n, s, _, m in rows], f, indent=1)
    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            failed_cases = write_report(rows, args.reference_dir, args.test_dir, f)
        print(f"Report written to {args.report}")
    else:
        failed_cases = write_report(rows, args.reference_dir, args.test_dir, sys.stdout)
    return 1 if failed_cases else 0


if __name__ == "__main__":
    sys.exit(main())
