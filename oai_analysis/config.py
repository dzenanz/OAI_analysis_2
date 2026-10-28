"""
Pipeline configuration as plain data, read from and written to YAML.

    pipeline:
      preprocessor: {name: oai_default, window_min_percentile: 0.1, window_max_percentile: 99.9}
      registerer: {name: unigradicon, finetune_steps: 2}
      ...
      biomarker_extractors:
        - {name: thickness_map_2d, grid_resolution: 300}
    device: auto

Each stage is chosen by its registered name (see registry.py), and the other keys are its parameters.
Anything left out takes its default, and a stage given without a name keeps the default implementation,
so `registerer: {finetune_steps: 5}` changes one parameter. `oai-analysis config dump` prints the full config.
"""
from dataclasses import fields
from pathlib import Path

import yaml

from oai_analysis import registry

SINGLE_STAGES = ("preprocessor", "segmenter", "registerer", "mesh_generator", "thickness_estimator", "atlas_mapper")
BIOMARKER_EXTRACTORS = "biomarker_extractors"
TOP_LEVEL_KEYS = ("pipeline", "device")


def load(source) -> dict:
    """Read a config: a dict is returned as it is, anything else is the path of a YAML file."""
    if isinstance(source, dict):
        return source
    return yaml.safe_load(Path(source).read_text(encoding="utf-8")) or {}


class _Dumper(yaml.SafeDumper):
    """Writes lists of plain values on one line, e.g. overlap_size: [16, 16, 8]."""

    def represent_list(self, data):
        flow = all(not isinstance(v, dict | list) for v in data)
        return self.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=flow)


_Dumper.add_representer(list, _Dumper.represent_list)


def dump(cfg: dict) -> str:
    return yaml.dump(cfg, Dumper=_Dumper, sort_keys=False)


def check_keys(cfg: dict):
    unknown = set(cfg) - set(TOP_LEVEL_KEYS)
    if unknown:
        raise ValueError(f"Unknown config keys {sorted(unknown)}. Known: {list(TOP_LEVEL_KEYS)}")
    unknown = set(cfg.get("pipeline") or {}) - {*SINGLE_STAGES, BIOMARKER_EXTRACTORS}
    if unknown:
        raise ValueError(f"Unknown pipeline stages {sorted(unknown)}. Known: {[*SINGLE_STAGES, BIOMARKER_EXTRACTORS]}")


def stage_to_dict(stage) -> dict:
    return {"name": stage.name, **{k: _plain(v) for k, v in stage.params().items()}}


def stage_from_dict(kind: str, entry: dict | None, default_class: type | None = None):
    """Create the stage an entry names, with its parameters; with no name, an instance of default_class."""
    params = dict(entry or {})
    name = params.pop("name", None)
    if name is None and default_class is None:
        raise ValueError(f"A {kind} needs a name. Known: {', '.join(registry.available(kind))}")
    cls = default_class if name is None else registry.get(kind, name)
    known = {f.name: f for f in fields(cls) if f.init}
    unknown = set(params) - set(known)
    if unknown:
        raise ValueError(f"{kind} {cls.name!r} has no parameter {', '.join(sorted(unknown))}. "
                         f"Parameters: {', '.join(known) or 'none'}")
    for key, value in params.items():
        if isinstance(value, list) and isinstance(known[key].default, tuple):  # YAML has lists, not tuples
            params[key] = tuple(value)
    return cls(**params)


def _plain(value):
    """The value as YAML can write it safely."""
    if isinstance(value, tuple | list):
        return [_plain(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value

