# Install

You need Python 3.11 or newer. A CUDA GPU is strongly recommended: a case takes about a minute on a GPU with 16 GB,
and many times longer on a CPU. Plan for about 16 GB of GPU memory; uniGradICON's fine-tuning peaks at 10.5 GB.

## 1. A virtual environment

```
python -m venv oai-env
oai-env\Scripts\activate        # Windows
source oai-env/bin/activate     # Linux
```

## 2. PyTorch

Install PyTorch **before** the package, choosing the build for your CUDA version on
https://pytorch.org/get-started/locally/. For example, for CUDA 12.8:

```
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

Otherwise pip installs a default build, which on Windows has no CUDA support. Check with:

```
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 3. The package

From PyPI:

```
pip install oai-analysis
```

or from a clone, for development (adds pytest and the linters):

```
git clone https://github.com/uncbiag/OAI_analysis_2
cd OAI_analysis_2
pip install -e ".[dev]"
```

With ITK 5.x (the version on PyPI), also install two ITK modules that ITK 6 includes:

```
pip install itk-cuberille itk-parabolicmorphology
```

## 4. Data and network weights

The network weights (segmentation U-Net, uniGradICON, the GradICON knee model) and the test data are downloaded on
first use, with checksums, into a per-user cache:

| System | Folder |
|---|---|
| Windows | `%LOCALAPPDATA%\oai-analysis\oai-analysis\Cache` |
| Linux | `~/.cache/oai-analysis` |

Set the `OAI_ANALYSIS_DATA` environment variable to use another folder, e.g. a shared one on a cluster. The first
case therefore needs internet access; after that, none. The atlas is part of the package.

## 5. Check the installation

```
oai-analysis config dump
pytest            # in a clone; about 12 min on a CPU, the GPU-only tests are skipped there
```

`pytest` runs the whole pipeline on the bundled test case (`test/test_smoke.py`), so a pass means the installation
works end to end.
