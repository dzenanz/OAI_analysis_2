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

## Docker

The `Dockerfile` builds an image with CUDA PyTorch, the package, the ITK 5 modules and the network weights, so it
runs without internet access. The host needs the NVIDIA driver (570 or newer for the default CUDA 12.8 build) and
the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/) for `--gpus`.

Build it from the repository root. The build has no `.git`, so it is given the version and the commit:

```
pip install setuptools-scm
docker build -t oai-analysis:$(git rev-parse --short HEAD) --build-arg VERSION=$(python -m setuptools_scm) --build-arg GIT_COMMIT=$(git rev-parse HEAD) .
```

`--build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126` picks another CUDA build of PyTorch, for older
drivers.

The image's command is `oai-analysis`, and its working folder is `/data`. Mount the images and the projects:

```
docker run --gpus all -v /data:/data oai-analysis:TAG /data/IMAGE.nii.gz /data/out left
docker run --gpus all -v /data:/data oai-analysis:TAG init /data/oai_projects/pilot --manifest /data/manifest.csv --data-root /data/OAI
docker run --gpus all -v /data:/data oai-analysis:TAG run /data/oai_projects/pilot
```

Each step's `provenance.json` then names the image and the git commit it was built from. An image cannot know its
own digest, so pass it in to have it recorded too:

```
docker run --gpus all -e OAI_ANALYSIS_IMAGE_DIGEST=$(docker inspect --format "{{.Id}}" oai-analysis:TAG) -v /data:/data oai-analysis:TAG run /data/oai_projects/pilot
```

On clusters that run Apptainer (Singularity) rather than Docker, convert the image with
`apptainer build oai-analysis.sif docker-daemon://oai-analysis:TAG` and run it with `apptainer run --nv` (not
tested yet). The network weights are in the image, so it needs no write access outside the mounted folders.
