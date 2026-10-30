"""
A smoke test of the whole pipeline on the colab case that needs no GPU: the probability maps are read from files
(PrecomputedSegmenter), so the segmentation network does not run, and the other stages are the defaults.
"""
import json
import os

import numpy as np
import torch

from oai_analysis import data
from oai_analysis.project import FINISHED, Project

# Mean thickness (mm) of the 2D maps of the colab case. uniGradICON's fine-tuning varies from run to run; over 5 runs
# on a GPU and 2 on a CPU the means were 2.2847-2.2867 (FC) and 2.2565-2.2586 (TC).
EXPECTED_MEAN_THICKNESS = {"FC": 2.286, "TC": 2.258}


def test_colab_case(tmp_path):
    colab = data.test_data_dir() / "colab_case"
    stages = {"segmenter": {"name": "precomputed", "path": str(colab / "{structure}_probmap.nii.gz")}}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    case = {"case_id": "colab", "patient_id": "", "month": "", "laterality": "left",
            "input": "image_preprocessed.nii.gz"}
    project = Project.create(tmp_path / "project", [case], pipeline_config={"pipeline": stages, "device": device},
                             data_root=colab)

    result = project.run(echo=lambda text: None)

    assert result[0].status in FINISHED, result[0].note
    folder = project.case_folder("colab")
    record = json.loads(next(folder.glob("*_segment/provenance.json")).read_text())
    assert record["implementation"] == "precomputed"
    assert record["environment"]["seed"] == 0
    if os.environ.get("OAI_ANALYSIS_IMAGE"):  # in the Docker image, provenance names the image and the commit
        assert record["environment"]["container"] == {"image": os.environ["OAI_ANALYSIS_IMAGE"],
                                                      "digest": os.environ.get("OAI_ANALYSIS_IMAGE_DIGEST")}
        if os.environ.get("OAI_ANALYSIS_GIT_COMMIT"):
            assert record["software"]["git_commit"] == os.environ["OAI_ANALYSIS_GIT_COMMIT"]
    for s, expected in EXPECTED_MEAN_THICKNESS.items():
        grid = np.load(folder / "07_biomarkers" / "thickness_map_2d" / f"{s}_thickness.npy")
        assert abs(np.nanmean(grid) - expected) < 0.02, s
    assert (folder / "thumbnails" / "atlas_map_FC.png").is_file()
