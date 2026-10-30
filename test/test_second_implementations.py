"""
The plugin design's acceptance test (Phase 1, M1.2): the colab case runs with the default stages and with their
second implementations, chosen only by the config, and provenance records which ones ran. Needs a CUDA GPU.
"""
import json

import pytest
import torch

from oai_analysis import data
from oai_analysis.project import FINISHED, Project

DEFAULTS = {"segment": "unet3d_oai", "register": "unigradicon", "mesh": "cuberille", "thickness": "mesh_splitting"}
SECOND = {
    "segmenter": {"name": "precomputed"},  # its path is set to the colab case's maps below
    "registerer": {"name": "gradicon_knee"},
    "mesh_generator": {"name": "marching_cubes"},
    "thickness_estimator": {"name": "distance_transform"},
}
SECOND_NAMES = {"segment": "precomputed", "register": "gradicon_knee", "mesh": "marching_cubes",
                "thickness": "distance_transform"}

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


@pytest.mark.parametrize("stages, expected", [({}, DEFAULTS), (SECOND, SECOND_NAMES)], ids=["defaults", "second"])
def test_switching_stages_by_config(tmp_path, stages, expected):
    colab = data.test_data_dir() / "colab_case"
    if "segmenter" in stages:
        stages = {**stages, "segmenter": {**stages["segmenter"], "path": str(colab / "{structure}_probmap.nii.gz")}}
    case = {"case_id": "colab", "patient_id": "", "month": "", "laterality": "left",
            "input": "image_preprocessed.nii.gz"}
    project = Project.create(tmp_path / "project", [case], pipeline_config={"pipeline": stages}, data_root=colab)
    result = project.run(echo=lambda text: None)
    assert result[0].status in FINISHED, result[0].note
    folder = project.case_folder("colab")
    for step, implementation in expected.items():
        record = json.loads(next(folder.glob(f"*_{step}/provenance.json")).read_text())
        assert record["implementation"] == implementation
    assert (folder / "07_biomarkers" / "thickness_map_2d" / "FC_thickness.png").is_file()
