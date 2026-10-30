# Handoff guide

For a team taking over the pipeline on their own hardware: how to set it up, run a pilot batch, read the results,
and what to watch out for.

## 1. Set up the environment

Follow [install](install.md), or use the [Docker image](install.md#docker), which has the weights in it. On a Linux
GPU server or cluster node:

```
python3.11 -m venv ~/oai-env && source ~/oai-env/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu128   # match the node's CUDA driver
pip install oai-analysis itk-cuberille itk-parabolicmorphology
export OAI_ANALYSIS_DATA=/shared/oai-analysis-data                       # optional: one shared download
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

- **GPU memory:** about 16 GB is comfortable. uniGradICON's fine-tuning peaks at about 10.5 GB.
- **Which GPU:** set `CUDA_VISIBLE_DEVICES=1` to use the second GPU. The registration package picks its own
  device, so this is more reliable than `device: cuda:1` in `project.yaml`.
- **Compute nodes without internet:** run one case on a node with internet first (or with `OAI_ANALYSIS_DATA`
  on shared storage), so the weights are downloaded once.
- **Check the install** with `pytest` in a clone of the repository: it runs the whole pipeline on the bundled
  case, on the CPU if there is no GPU.

## 2. The OAI images and the case list

The pipeline reads the OAI's sagittal 3D DESS images as DICOM folders (or any image file). Projects are created from
a *manifest*: a CSV with `patient_id, study_uid, vol_path, month, study_description`. The one for the whole cohort
(48,067 images) is `data/oai-sampler/study_uid_to_vol_path.csv` in
[knee-sarg](https://github.com/PaulHax/knee-sarg). `vol_path` is relative to the folder holding the OAI download
(`OAIBaselineImages/`, `OAI96MonthImages/`, ...), which is `--data-root`. The knee is read from
`study_description`.

## 3. A pilot batch

Start with a few patients, check the results, then grow:

```
oai-analysis init /data/oai_projects/pilot --manifest study_uid_to_vol_path.csv --data-root /data/OAI \
    --patients 9016304,9021791,9040390,9094865 --months 0,96
oai-analysis run /data/oai_projects/pilot
oai-analysis status /data/oai_projects/pilot
```

On an RTX 5060 Ti (16 GB) a case takes about a minute: preprocessing 6 s, segmentation 18 s, registration 11–20 s,
thickness 22 s, the rest a few seconds. A long run keeps going if you log out when started with `nohup` or in
`tmux`/`screen`; if it stops, the same `run` command resumes it.

On a cluster, a job can run one project. Only one run should work on a case at a time: two runs on separate cases
(`--cases`) of the same project have not been tested, so for several GPUs the safe way is one project per GPU (e.g.
by patient range), each with its own `run`.

## 4. Reading the results

- **`oai-analysis status PROJECT`**: one line per case. Statuses and selectors are on the
  [projects page](projects.md#statuses), and the [reviewer guide](reviewer_guide.md) explains them for
  non-engineers.
- **`logs/`**: per run, a plain-English summary (`*.summary.txt`) and the full log (`*.log`), which has the stages'
  output and the tracebacks.
- **The results** are in `cases/PATIENT/VISIT_KNEE/`: the 2D thickness maps in
  `07_biomarkers/thickness_map_2d/*.npy`, the atlas surfaces with thickness in `06_atlas_map/*_atlas.vtk`, pictures
  in `thumbnails/`.
- **How a result was made** is in each step's `provenance.json`: the stage, its parameters, the software versions
  and git commit, the GPU, the input and output hashes, the sanity checks. See
  [provenance](projects.md#provenance).
- **For analysis**, `index.parquet` has one row per case (status, flags, timing).

## 5. Known problems

| Problem | What happens | What to do |
|---|---|---|
| **Cuberille crash.** On some cases ITK's cuberille filter fails, e.g. 9135902, left knee, 0M. | The case fails at mesh generation with `IndexError: invalid vector subscript`, or the whole process crashes (segmentation fault). | The case shows Failed, or after a crash Incomplete, and `run` resumes the rest. A newer ITKCuberille may fix it. Meanwhile, the `marching_cubes` mesh generator gets through this case and agrees with cuberille to within 0.03–0.08 mm. Use it for the failed cases in a separate project, or for all cases. |
| **Smoothing and thickness.** The original pipeline measured thickness on smoothed surfaces moved back from atlas space, about 0.3 mm from the real surface on average. | Since this refactor the thickness is measured on the original, unsmoothed surface (`back_mapping: cell_ids`). Femoral cartilage comes out about 6% thicker (0.03–0.16 mm) than in older results. | Do not mix results from before and after this change without accounting for it. `thickness_estimator: {back_mapping: transform}` reproduces the old method. See [CHANGELOG.md](../CHANGELOG.md). |
| **Registration is not deterministic.** uniGradICON's fine-tuning varies from run to run. | Thickness maps of the same case differ by 0.035 mm on average between runs (0.12 mm at the 95th percentile, up to 2.6 mm at single points). | Expected. Everything before registration is exactly reproducible. `deterministic: true` cuts the difference about 70-fold, but makes registration about 5 times slower; see [reproducibility](projects.md#reproducibility). |
| **GPU and CPU differ slightly.** | The segmentation probabilities differ by up to 0.0025 between GPU and CPU. | Expected, and too small to matter. |
| **Cases flagged Needs Review.** In the benchmark cases and a Dec 2024 manual review, flagged cases had segmentation pieces outside the cartilage, or holes in the thickness. | The case finishes, with a note. | Look at its thumbnails; see the [reviewer guide](reviewer_guide.md). Thresholds can be changed per project. |
| **A run on a machine without a GPU.** | `run` stops: "No CUDA GPU is available". | Check `torch.cuda.is_available()`. To run on the CPU on purpose, set `device: cpu` in `project.yaml`. |
| **The package version.** A clone that is not at a release tag reports a version like `0.1.dev230+g…`. | Only cosmetic: provenance also records the git commit. | Tag releases on `main`. |

## 6. Checking a new installation or code change against known results

`tools/run_baseline.py` runs the regression benchmark cases (`tools/benchmark_cases.json`: 12 OAI images and the
bundled case) and compares them with a reference run:

```
python tools/run_baseline.py run1 --oai-dir /data/OAI
python tools/run_baseline.py run2 --oai-dir /data/OAI --compare-to run1
```

The comparison is in `run2/comparison.md`. Its tolerances (`tools/regression_tolerances.json`) allow for the
run-to-run differences of registration and require everything before it to match exactly. Make the first reference
on your own hardware, since GPUs and library versions change results slightly. The `gpu-tests` GitHub workflow runs
the same check on a self-hosted runner once the `OAI_REFERENCE_DIR` repository variable points to a reference run on
it.

## 7. Where to change things

| To | Change |
|---|---|
| use another stage or parameter | `project.yaml`; see the [config reference](config.md) |
| add a result computed for each case | a biomarker extractor; see [writing a plugin](plugins.md) |
| change a sanity-check threshold | the stage's parameter in `project.yaml`; `run` rechecks without recomputing |
| keep less on disk | `retention: final` in `project.yaml` |

Orchestrators (knee-sarg's Dagster, SLURM scripts) should call `oai-analysis run` or `Project.run()` on a list of
cases, rather than one process per image, so the networks are loaded once per batch.
