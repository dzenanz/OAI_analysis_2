"""
Run the pipeline on the regression benchmark cases and record how the outputs were produced.

Usage:
    python tools/run_baseline.py OUTPUT_DIR --oai-dir O:/OAI
    python tools/run_baseline.py OUTPUT_DIR --bundled-only

Each case is written to OUTPUT_DIR/<name>_<laterality>, the same layout as pipeline.py's __main__.
OUTPUT_DIR/run_info.json records the git commit, installed packages, GPU, per-case timings and failures.
Compare two such directories with tools/compare_outputs.py.
"""
import argparse
import datetime
import json
import os
import pathlib
import platform
import subprocess
import sys
import time
import traceback

REPO_DIR = pathlib.Path(__file__).resolve().parent.parent
PACKAGE_DIR = REPO_DIR / "oai_analysis"
CASES_FILE = pathlib.Path(__file__).resolve().parent / "benchmark_cases.json"


def import_analysis_pipeline():
    """Return analysis_pipeline() and the directory that bundled case paths are relative to."""
    try:
        from oai_analysis.data import test_data_dir
        from oai_analysis.pipeline import analysis_pipeline

        return analysis_pipeline, test_data_dir().parent
    except ImportError:
        # pipeline.py up to f198a2f only works with oai_analysis/ on sys.path,
        # and uniGradICON loads its weights from ./network_weights, relative to the working directory
        sys.path.insert(0, str(PACKAGE_DIR))
        os.chdir(PACKAGE_DIR)
        from pipeline import analysis_pipeline

        return analysis_pipeline, PACKAGE_DIR / "data"


def git_info():
    def git(*args):
        return subprocess.run(["git", *args], cwd=REPO_DIR, capture_output=True, text=True).stdout.strip()

    return {
        "commit": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty_files": git("status", "--porcelain", "--untracked-files=no").splitlines(),
    }


def environment_info():
    import torch

    packages = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "host": platform.node(),
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "packages": packages.splitlines(),
    }


def select_cases(args, bundled_data_dir):
    with open(CASES_FILE) as f:
        cases_spec = json.load(f)
    cases = []
    for case in cases_spec["bundled"]:
        cases.append({**case, "input": str(bundled_data_dir / case["path"])})
    if not args.bundled_only:
        for case in cases_spec["oai"]:
            cases.append({**case, "input": str(pathlib.Path(args.oai_dir) / case["path"])})
    if args.cases:
        wanted = set(args.cases.split(","))
        cases = [c for c in cases if f"{c['name']}_{c['laterality']}" in wanted]
    return cases


def main():
    parser = argparse.ArgumentParser(description="Run the pipeline on the regression benchmark cases")
    parser.add_argument("output_dir", type=pathlib.Path)
    parser.add_argument("--oai-dir", default="O:/OAI", help="Root of the OAI dataset")
    parser.add_argument("--bundled-only", action="store_true", help="Only run the case bundled with the code")
    parser.add_argument("--cases", help="Comma-separated subset of case directory names, e.g. 9016304_0M_left")
    parser.add_argument("--compare-to", type=pathlib.Path,
                        help="Reference output directory; run tools/compare_outputs.py against it afterwards")
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    analysis_pipeline, bundled_data_dir = import_analysis_pipeline()
    cases = select_cases(args, bundled_data_dir)

    run_info = {
        "started": datetime.datetime.now().astimezone().isoformat(),
        "git": git_info(),
        "environment": environment_info(),
        "cases": {},
    }
    info_file = output_dir / "run_info.json"

    for case in cases:
        case_dir_name = f"{case['name']}_{case['laterality']}"
        print(f"\nProcessing {case_dir_name}: {case['input']}", flush=True)
        start = time.perf_counter()
        try:
            analysis_pipeline(case["input"], str(output_dir / case_dir_name), case["laterality"],
                              keep_intermediate_outputs=True)
            status = "succeeded"
            error = None
        except Exception:
            status = "failed"
            error = traceback.format_exc()
            print(error, flush=True)
        run_info["cases"][case_dir_name] = {
            "input": case["input"],
            "laterality": case["laterality"],
            "status": status,
            "duration_s": round(time.perf_counter() - start, 1),
            "error": error,
        }
        run_info["finished"] = datetime.datetime.now().astimezone().isoformat()
        with open(info_file, "w") as f:
            json.dump(run_info, f, indent=2)

    failed = [name for name, c in run_info["cases"].items() if c["status"] == "failed"]
    print(f"\nDone: {len(cases) - len(failed)} succeeded, {len(failed)} failed {failed}")

    if args.compare_to:
        compare_script = pathlib.Path(__file__).resolve().parent / "compare_outputs.py"
        command = [sys.executable, str(compare_script), str(args.compare_to.resolve()), str(output_dir),
                   "--report", str(output_dir / "comparison.md")]
        if args.cases or args.bundled_only:
            command.append("--only-common")
        if subprocess.run(command).returncode != 0:
            return 1
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
