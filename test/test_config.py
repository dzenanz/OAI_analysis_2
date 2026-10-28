import textwrap

import pytest
import yaml

from oai_analysis import OAPipeline, config, registry
from oai_analysis.cli import main as cli_main
from oai_analysis.stages import (
    DistanceTransformThickness,
    MeshSplittingThickness,
    ThicknessMap2D,
    UNetSegmenter,
    UniGradICONRegisterer,
)


def stage_types_and_params(pipeline):
    return [(type(stage), stage.params()) for stage in pipeline.stages]


def test_default_config_round_trip():
    default = OAPipeline()
    text = config.dump(default.config())
    rebuilt = OAPipeline.from_config(yaml.safe_load(text))
    assert stage_types_and_params(rebuilt) == stage_types_and_params(default)
    assert rebuilt.config() == default.config()
    assert isinstance(rebuilt.segmenter.overlap_size, tuple)  # YAML lists become tuples again


def test_config_file(tmp_path):
    path = tmp_path / "project.yaml"
    path.write_text(textwrap.dedent("""
        pipeline:
          registerer: {finetune_steps: 5}
          thickness_estimator: {name: distance_transform}
          biomarker_extractors:
            - {name: thickness_map_2d, grid_resolution: 100}
        device: cpu
    """))
    pipeline = OAPipeline.from_config(path)
    assert pipeline.registerer == UniGradICONRegisterer(finetune_steps=5)  # no name: the default implementation
    assert pipeline.thickness_estimator == DistanceTransformThickness()
    assert pipeline.biomarker_extractors == [ThicknessMap2D(grid_resolution=100)]
    assert pipeline.segmenter == UNetSegmenter()  # left out: the default
    assert pipeline.device == "cpu"


def test_empty_config():
    pipeline = OAPipeline.from_config({})
    assert stage_types_and_params(pipeline) == stage_types_and_params(OAPipeline())
    assert pipeline.device is None


@pytest.mark.parametrize("cfg, message", [
    ({"pipelin": {}}, "Unknown config keys"),
    ({"pipeline": {"registration": {}}}, "Unknown pipeline stages"),
    ({"pipeline": {"registerer": {"name": "nope"}}}, "No registerer named 'nope'"),
    ({"pipeline": {"registerer": {"finetune_step": 5}}}, "has no parameter finetune_step"),
    ({"pipeline": {"thickness_estimator": {"back_mapping": "nope"}}}, "back_mapping must be"),
    ({"pipeline": {"biomarker_extractors": [{"grid_resolution": 100}]}}, "needs a name"),
])
def test_config_errors(cfg, message):
    with pytest.raises((KeyError, ValueError), match=message):
        OAPipeline.from_config(cfg)


def test_registry_has_built_in_stages():
    assert registry.get("thickness_estimator", "mesh_splitting") is MeshSplittingThickness
    assert registry.available("thickness_estimator") == ["distance_transform", "mesh_splitting"]
    with pytest.raises(ValueError, match="already registered"):
        @registry.register
        class Impostor(MeshSplittingThickness):
            pass


def test_plugin_entry_point(tmp_path, monkeypatch):
    """A stage in another installed package is found through its oai_analysis.plugins entry point."""
    (tmp_path / "oai_test_plugin.py").write_text(textwrap.dedent("""
        from dataclasses import dataclass
        from typing import ClassVar

        import numpy as np

        from oai_analysis.registry import register
        from oai_analysis.stages import BiomarkerExtractor

        @register
        @dataclass
        class MeanThickness(BiomarkerExtractor):
            name: ClassVar[str] = "test_mean_thickness"
            percentile: float = 50.0

            def run(self, case):
                return {"mean_thickness.npy": np.array([np.nanmean(case.atlas_thickness(s)) for s in ("FC", "TC")])}
    """))
    dist_info = tmp_path / "oai_test_plugin-0.1.dist-info"  # what `pip install` would create
    dist_info.mkdir()
    (dist_info / "METADATA").write_text("Metadata-Version: 2.1\nName: oai-test-plugin\nVersion: 0.1\n")
    (dist_info / "entry_points.txt").write_text("[oai_analysis.plugins]\ntest = oai_test_plugin\n")
    monkeypatch.syspath_prepend(str(tmp_path))

    pipeline = OAPipeline.from_config({"pipeline": {"biomarker_extractors": [
        {"name": "thickness_map_2d"}, {"name": "test_mean_thickness", "percentile": 90}]}})
    extractor = pipeline.biomarker_extractors[1]
    assert type(extractor).__module__ == "oai_test_plugin"
    assert extractor.params() == {"percentile": 90}
    assert "test_mean_thickness" in registry.available("biomarker_extractor")
    assert pipeline.config()["pipeline"]["biomarker_extractors"][1] == {"name": "test_mean_thickness",
                                                                        "percentile": 90}


def test_cli_config_dump(tmp_path, capsys):
    cli_main(["config", "dump"])
    assert yaml.safe_load(capsys.readouterr().out) == OAPipeline().config()

    path = tmp_path / "changes.yaml"
    path.write_text("pipeline:\n  registerer: {finetune_steps: 5}\n")
    cli_main(["config", "dump", "--config", str(path)])
    assert yaml.safe_load(capsys.readouterr().out)["pipeline"]["registerer"] == {"name": "unigradicon",
                                                                                  "finetune_steps": 5}
