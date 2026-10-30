# Changelog

Each release has two parts: **What might look different**, in plain language for people who use the results, and
**Engineering changes**, for people who run or develop the code. Numbers come from the regression benchmark: 12 OAI
knee images and the bundled test case, compared with `tools/compare_outputs.py` against a reference run of the
previous version (`f198a2f`).

## Unreleased: Phase 1 refactor

### What might look different

- **Femoral cartilage is slightly thicker.** Thickness is now measured on the cartilage surface as found in the
  image. Before, it was measured on a smoothed copy, about 0.3 mm away from the real surface on average. On the
  benchmark cases:
  - femoral cartilage: +0.03 to +0.16 mm in the average over the map (about +6%), in every case;
  - tibial cartilage: −0.09 to +0.10 mm, with no consistent direction;
  - per point, maps differ by 0.12–0.32 mm on average, and by up to 3.8 mm at single points.

  Values may also be a little noisier, as they are no longer smoothed. Do not mix results from before and after
  this change in one analysis. The old method is still available (`thickness_estimator: {back_mapping:
  transform}`), and with it, results match the previous version within the run-to-run noise.
- **Nothing else changes in the results.** The image preparation, the cartilage segmentation and the surfaces in the
  image are identical to the previous version. Registration to the atlas varies a little from run to run, as before:
  maps of the same case differ by 0.035 mm on average between two runs.
- **Cases can be marked *Needs Review*.** Automatic checks now flag results that look unusual, with a note in plain
  English, e.g. "37.6% of the femoral cartilage is outside its largest piece". On the benchmark cases they flag all
  the cases a Dec 2024 manual review found problems in, and none of the clean ones. See the
  [reviewer guide](docs/reviewer_guide.md).
- **Each case has pictures to look at**: the cartilage found on an MRI slice, and the thickness maps on a fixed
  0–5 mm colour scale.
- **Results come much faster**: about 1 minute per case instead of 4.5 on the same GPU (RTX 5060 Ti), mainly because
  the networks are loaded once for many cases and GPU memory is freed between them.
- **One case that used to stop the whole batch** (9135902, left knee, 0M) now fails on its own, and the others
  finish.

### Engineering changes

- **Package.** Installable with `pip` from any directory; Python 3.11 or newer; Apache-2.0 license in
  `pyproject.toml` too; dependency ranges instead of exact pins; version from git tags. Network weights and test
  data download, with checksums, into a per-user cache (`OAI_ANALYSIS_DATA` overrides it) instead of the working
  directory. The atlas is part of the package. `itk-cuberille` and `itk-parabolicmorphology` are no longer
  dependencies, because ITK 6 includes them; with ITK 5.x, install them separately.
- **Stages.** The pipeline is seven steps, each done by a replaceable stage class with its parameters as fields
  (`oai_analysis/stages/`). `OAPipeline` runs them in a fixed order and loads the networks once. See
  [the pipeline](docs/pipeline.md) and the [Python API](docs/api.md).
- **Config.** Stages and parameters can be chosen in a YAML file (`--config`, `project.yaml`);
  `oai-analysis config dump` prints the defaults. Other packages can add stages through the `oai_analysis.plugins`
  entry point. See the [config reference](docs/config.md) and [writing a plugin](docs/plugins.md).
- **Second implementations**, chosen by config: `gradicon_knee` registration, `precomputed` segmentation (probability
  maps from files), `marching_cubes` surfaces, `distance_transform` thickness.
- **Projects** (`oai-analysis init | run | status`): a folder per project with every step's outputs and a
  `provenance.json` (parameters, software versions, git commit, GPU, input and output hashes). `run` resumes,
  retries failures, and after a config change redoes only what it affects. Statuses Succeeded, Needs Review, Failed,
  Outdated, Incomplete, Running, Not started; `index.parquet`; plain-English run summaries; cases in batches; a
  failing case does not stop the others. See [projects](docs/projects.md).
- **Storage.** A project case takes about 330 MB (340 MB with every intermediate output) instead of 1050 MB:
  compressed NRRD images and HDF5 transforms instead of uncompressed NRRD and 300 MB text transforms.
  `retention: final` keeps about 10 MB per case. The single-image command writes the same files as before.
- **Sanity checks** on every default stage, with thresholds as stage parameters; thumbnails per case.
- **Kept working:** `oai-analysis IMAGE OUT left|right` and `pipeline_cli.py` (what knee-sarg runs), and
  `analysis_pipeline()`, with the same output file names.
- **Moved:** `mesh_processing`, `thickness_computation` and `cartilage_shape_processing` are now in
  `oai_analysis.algorithms`; the segmentation network code is in `oai_analysis.stages.segmentation`.
- **Removed:** `AVSM_Registration`, `dask_processing.py` and its notebooks. `AnalysisObject` is deprecated, kept
  for the notebooks.
- **Tests and CI.** Unit tests per stage, a CPU smoke test of the whole pipeline, restored asserts, a CPU workflow
  with cached downloads, and a GPU workflow with an optional regression job (`tools/run_baseline.py`).
- **Documentation** in `docs/`.
- **Reproducibility.** `seed` (default 0) seeds the random number generators at every step; `deterministic: true`
  uses torch's deterministic algorithms, which cuts the run-to-run differences of registration about 70-fold and
  makes it about 5 times slower. Both are recorded in provenance.
- **Docker.** A `Dockerfile` for an image with CUDA PyTorch and the network weights, which runs offline. Provenance
  records the image, the git commit it was built from, and its digest when given (`OAI_ANALYSIS_IMAGE_DIGEST`). The
  `docker` workflow builds it and runs the smoke test in it.
