"""
Provenance: which code, parameters, inputs and machine produced each output.

A project writes one provenance.json per step of each case. Its fingerprint, a hash of the stage's name, version,
parameters, weights and inputs, decides whether the step is still up to date.
"""
import datetime
import functools
import hashlib
import json
import platform
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
TRACKED_PACKAGES = ("oai-analysis", "torch", "itk", "vtk", "icon_registration", "unigradicon", "numpy", "scipy",
                    "scikit-learn", "trimesh")


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_path(path) -> str:
    """Hash of a file, or of every file in a directory (e.g. a DICOM series) with its relative name."""
    path = Path(path)
    if path.is_file():
        return sha256_file(path)
    digest = hashlib.sha256()
    for file in sorted(p for p in path.rglob("*") if p.is_file()):
        digest.update(file.relative_to(path).as_posix().encode())
        digest.update(sha256_file(file).encode())
    return digest.hexdigest()


def sha256_json(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def fingerprint(stage, inputs: dict[str, str]) -> str:
    """
    What decides whether a step's outputs are up to date: the stage and the hashes of its inputs, by name.
    Parameters that only change how fast the stage runs (its performance_params) are left out.
    """
    return sha256_json({
        "implementation": stage.name,
        "version": stage.version,
        "params": stage.result_params(),
        "weights": stage.weights(),
        "inputs": inputs,
    })


@functools.cache
def atlas_sha256(atlas_name: str) -> str | None:
    """Hash of the files of an atlas shipped with the package; None for other atlases."""
    atlas_dir = PACKAGE_DIR / "resources" / atlas_name
    return sha256_path(atlas_dir) if atlas_dir.is_dir() else None


@functools.cache
def software() -> dict:
    versions = {}
    for package in TRACKED_PACKAGES:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    versions["python"] = platform.python_version()
    versions.update(_git())
    return versions


def _git() -> dict:
    """The commit and whether there are uncommitted changes, when running from a git checkout."""
    if not (PACKAGE_DIR.parent / ".git").exists():
        return {}
    def git(*args):
        return subprocess.run(["git", *args], cwd=PACKAGE_DIR.parent, capture_output=True, text=True).stdout.strip()
    try:
        return {"git_commit": git("rev-parse", "HEAD"),
                "git_dirty": bool(git("status", "--porcelain", "--untracked-files=no"))}
    except OSError:  # git is not installed
        return {}


@functools.cache
def environment(device: str) -> dict:
    import torch

    gpu = None
    if device.startswith("cuda") and torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(torch.device(device))
    return {
        "host": platform.node(),
        "platform": platform.platform(),
        "device": device,
        "gpu": gpu,
        "torch_cuda": torch.version.cuda,
        "command": " ".join(sys.argv),
    }


def now() -> str:
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")
