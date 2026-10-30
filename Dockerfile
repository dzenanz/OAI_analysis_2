# oai-analysis with CUDA PyTorch, the ITK 5 modules and the network weights, ready to run offline.
#
# Build, from the repository root (setuptools-scm gives the version from git, as pip install would):
#   pip install setuptools-scm
#   docker build -t oai-analysis:$(git rev-parse --short HEAD) \
#     --build-arg VERSION=$(python -m setuptools_scm) --build-arg GIT_COMMIT=$(git rev-parse HEAD) .
# Run one image (the NVIDIA Container Toolkit is needed for --gpus):
#   docker run --gpus all -v /data:/data oai-analysis:TAG /data/IMAGE /data/out left
# Run a project:
#   docker run --gpus all -v /data:/data oai-analysis:TAG run /data/oai_projects/pilot
# See docs/install.md#docker for recording the image digest in provenance.

FROM python:3.11-slim-bookworm

# VTK needs the X11 and OpenGL client libraries even without a display
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libx11-6 libxext6 libxrender1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# The CUDA build of PyTorch; its wheels carry the CUDA libraries, so the host only needs the NVIDIA driver.
# cu128 runs on drivers 570 and newer and supports GPUs up to the RTX 50 series.
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
RUN pip install --no-cache-dir torch --index-url ${TORCH_INDEX_URL}

# The version comes from git tags, and the build context has no .git, so the build passes it in (PEP 440)
ARG VERSION=0.0.0+unknown
ARG GIT_COMMIT=
COPY pyproject.toml README.md LICENSE /src/
COPY oai_analysis /src/oai_analysis
RUN SETUPTOOLS_SCM_PRETEND_VERSION=${VERSION} pip install --no-cache-dir /src \
        itk-cuberille itk-parabolicmorphology \
    && pip freeze > /opt/pip-freeze.txt \
    && rm -rf /src

# Network weights in the image, so it runs without internet access
ENV OAI_ANALYSIS_DATA=/opt/oai-analysis-data
RUN python -c "from oai_analysis import data; data.models_dir(); data.unigradicon_weights(); data.gradicon_knee_weights()" \
    && chmod -R a+rwX ${OAI_ANALYSIS_DATA}

# Recorded in each step's provenance.json (provenance.software() and provenance.container())
ENV OAI_ANALYSIS_IMAGE=oai-analysis:${VERSION} \
    OAI_ANALYSIS_GIT_COMMIT=${GIT_COMMIT} \
    MPLBACKEND=Agg

RUN useradd --create-home oai
USER oai
WORKDIR /data
ENTRYPOINT ["oai-analysis"]
CMD ["--help"]
