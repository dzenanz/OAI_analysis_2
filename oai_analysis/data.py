"""
Downloadable data: model weights and test data. The atlas ships with the package (see atlas.py).

Everything is fetched with pooch into a per-user cache directory and checked against its sha256.
Set the OAI_ANALYSIS_DATA environment variable to use a different directory, e.g. a shared one on a cluster.
Nothing is written into the package directory or the current working directory.
"""
from pathlib import Path

import pooch

github_release_tag = 'v2.0.0'
test_data_sha256 = "bfb5d5f17ff0886f5815c79fba119fc4294c35b8f9fcd586a171d9310cd90cdf"
models_sha256 = "ffd081ba26f9908f17790ecdb8b1d025bff32ccbdced9bf55158166561064458"
unigradicon_sha256 = "1fcfa02214540915cad7b768570add2b4247c22c516d75a21e0e8258f50254a6"
gradicon_knee_sha256 = "8696b2cb072792e75074d7d3ac9975677d6bdc7659fe3901ca20aa42b0b43b99"

data_fetcher = pooch.create(
    path=pooch.os_cache("oai-analysis"),
    base_url=f"https://github.com/uncbiag/OAI_analysis_2/releases/download/{github_release_tag}/",
    env="OAI_ANALYSIS_DATA",
    registry={
        "oai-analysis-test-data.tar.gz": f"sha256:{test_data_sha256}",
        "oai-analysis-models.tar.gz": f"sha256:{models_sha256}",
        "unigradicon1.0/Step_2_final.trch": f"sha256:{unigradicon_sha256}",
        "gradicon_knee/knee_model_weights_step_2.trch": f"sha256:{gradicon_knee_sha256}",
    },
    urls={
        "unigradicon1.0/Step_2_final.trch":
            "https://github.com/uncbiag/uniGradICON/releases/download/unigradicon_weights/Step_2_final.trch",
        "gradicon_knee/knee_model_weights_step_2.trch":
            "https://github.com/uncbiag/ICON/releases/download/pretrained_models_v1.0.0/knee_model_weights_step_2.trch",
    },
    retry_if_failed=5,
)


def data_dir():
    """Directory holding the downloaded data."""
    return Path(data_fetcher.abspath)


def _fetch_and_extract(archive, extract_dir):
    data_fetcher.fetch(archive, processor=pooch.Untar(extract_dir=extract_dir))
    return data_dir() / extract_dir


def test_data_dir():
    return _fetch_and_extract("oai-analysis-test-data.tar.gz", "test_data")


def models_dir():
    return _fetch_and_extract("oai-analysis-models.tar.gz", "models")


def unigradicon_weights():
    """Path to the uniGradICON network weights."""
    return Path(data_fetcher.fetch("unigradicon1.0/Step_2_final.trch"))


def gradicon_knee_weights():
    """Path to the GradICON network weights trained on OAI knees."""
    return Path(data_fetcher.fetch("gradicon_knee/knee_model_weights_step_2.trch"))
