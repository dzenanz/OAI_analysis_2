# Projects

A project is a folder with the cases to process, the settings, and every result with a record of how it was made.
Everything `oai-analysis` needs to know about a project is in that folder, so it can be copied, resumed on another
machine, or read by other programs.

## The project folder

```
M:/OAI_projects/pilot/
  project.yaml          settings: the pipeline config, data root, batch size, retention, storage formats
  cases.csv             one row per case: case_id, patient_id, month, laterality, input
  index.parquet         one row per case: status, the step that failed, a note, the review flags, ...
  runs.jsonl            one line per `oai-analysis run`: when, the command, how many cases, the software versions
  logs/
    2026-10-29T130729.log           everything, including what the stages print
    2026-10-29T130729.summary.txt   one plain-English line per finished case
  cases/
    9016304/
      96M_LEFT/         one case: patient 9016304, 96-month visit, left knee
```

- `project.yaml` takes the settings of the [config reference](config.md). `oai-analysis init --config FILE` starts
  from a config file; otherwise it holds the full default pipeline config.
- `cases.csv` can be edited, e.g. to add cases. `input` is a file or a DICOM folder, relative to `data_root` (or
  absolute). Case IDs from a manifest look like `9016304_96M_LEFT`; any unique ID works.
- `index.parquet` is rewritten after every case. Read it with `pandas.read_parquet` or DuckDB.

### A case's folder

```
cases/9016304/96M_LEFT/
  case.json                  the input path, its sha256, the patient, visit and knee
  thumbnails/
    segment_overlay.png      the sagittal slice with the most cartilage, FC in red, TC in blue
    atlas_map_FC.png         the 2D thickness maps, on a fixed 0–5 mm scale so cases compare
    atlas_map_TC.png
  01_preprocess/   image.nrrd                                     provenance.json
  02_segment/      FC_prob.nrrd  TC_prob.nrrd                     provenance.json
  03_register/     phi_AB.h5  phi_BA.h5                           provenance.json
  04_mesh/         FC_mesh.vtk  TC_mesh.vtk                       provenance.json
  05_thickness/    FC_thickness.vtk  TC_thickness.vtk             provenance.json
  06_atlas_map/    FC_atlas.vtk  TC_atlas.vtk                     provenance.json
  07_biomarkers/
    thickness_map_2d/   FC_thickness.npy  FC_thickness.png  TC_...   provenance.json
```

[The pipeline page](pipeline.md) says what each step does. The results most people want are the 2D maps in
`07_biomarkers/thickness_map_2d/` and the atlas surfaces with thickness in `06_atlas_map/` (VTK files; the
`vertex_thickness` array has the thickness in mm).

## Statuses

`oai-analysis status PROJECT` shows each case's status, a mark per step, and a note.

| Status | Meaning |
|---|---|
| **Succeeded** | Every step is done and no sanity check flagged anything. |
| **Needs Review** | Every step is done, but a sanity check flagged something; the note says what. See the [reviewer guide](reviewer_guide.md). |
| **Failed** | A step failed; the note says which and why. The steps before it are kept. |
| **Outdated** | A setting changed since the case ran; the note says which (e.g. `registerer.finetune_steps changed 2 -> 5`). |
| **Incomplete** | Some steps are done, and the run stopped before the rest (e.g. it was interrupted). |
| **Running** | A run is working on it now. |
| **Not started** | Nothing done yet. |

Step marks: `✓` done, `!` done but flagged for review, `✗` failed, `~` outdated, blank not done.

Selectors, for both `status` and `run`:

| Option | Cases |
|---|---|
| `--cases ID,ID` | these |
| `--failed` | Failed |
| `--failed-at STEP` | failed at this step: `preprocess`, `segment`, `register`, `mesh`, `thickness`, `atlas_map`, `biomarkers/thickness_map_2d` |
| `--stale` | Outdated |
| `--needs-review` | Needs Review |

`status --csv FILE` also writes the table to a CSV file, with all the review messages.

## Running, resuming and redoing

`oai-analysis run PROJECT` does every step that is not done and up to date, for the selected cases (all by default).
So the same command:
- **resumes** an interrupted run: finished steps are kept, and an Incomplete case continues where it stopped;
- **retries** Failed cases from the step that failed;
- **redoes outdated work**: after a setting changes, only the steps it affects and the steps after them.

It leaves Succeeded and Needs Review cases alone. `--from STEP` redoes a step and everything after it even if it is
up to date, e.g. after replacing the files a `precomputed` segmenter reads.

**Batches.** Cases run in batches of `batch_size` (default 8, or `run --batch-size N`). A batch's cases are
preprocessed, then segmented one after another, then registered one after another, so each network is loaded once
and stays in GPU memory for the whole batch; GPU memory is released after each network is done. Then each case
finishes on its own. A batch holds about 0.5 GB per case in memory.

**Failures.** A step that raises an error fails its case, and the run goes on with the others. Its
`provenance.json` records the error and the traceback. A crash of the whole process (e.g. a segmentation fault in
compiled code) stops the run; the cases it was working on show as Incomplete, and the next `run` resumes them.

**The device.** With `device: auto` (the default) a project runs on the GPU, and stops with a message if there is
none, rather than run on the CPU without being asked. Set `device: cpu` in `project.yaml` to run on the CPU.

**Watching a run.** A case's `running.json` gets a heartbeat every minute, and steps that take more than 5 minutes
log that they are still working, so a stalled run shows in the log. `status` from another window shows which cases
are Running.

## When is a step up to date?

Each step's `provenance.json` holds a *fingerprint*: a hash of the stage's name, version and parameters, its network
weights, the atlas, and the hashes of the files it read. A step is done if its fingerprint still matches. So it is
redone when:
- its stage or any parameter that affects its results changes in `project.yaml`;
- a step it reads was redone and gave different results (the input changed too: `preprocess` hashes the input
  image).

Two kinds of parameters do not count, and the [config reference](config.md) marks them:
- *speed* parameters (e.g. the U-Net's `batch_size`) change only how fast a stage runs;
- *check* parameters are sanity-check thresholds. Changing one makes the step Outdated, and `run` rechecks the saved
  outputs (a few seconds per case) without redoing the step.

## Provenance

`provenance.json` in each step folder records how its outputs were made:

| Field | Content |
|---|---|
| `step`, `implementation`, `implementation_version` | e.g. `segment`, `unet3d_oai`, `1` |
| `status` | `complete` or `failed`, with `error` and `traceback` when failed |
| `params`, `check_params` | the stage's parameters and sanity-check thresholds |
| `weights` | the sha256 of the network weights used |
| `inputs`, `outputs` | the files read and written, with their sha256 |
| `fingerprint` | see above |
| `software` | versions of oai-analysis, Python, torch, ITK, VTK, ..., the git commit, and whether it had uncommitted changes |
| `environment` | host, platform, device, GPU, CUDA version, the command |
| `run_id`, `started`, `finished`, `duration_s` | when, and how long; `run_id` names the run's log in `logs/` |
| `qc` | the sanity-check findings: severity (`info`, `review`, `fail`), code, plain-English message, measured value |

The hashes are kept even when `retention` deletes the files, so the record of what was computed survives.

## Sanity checks

After each step, its stage checks the outputs. A *fail* finding fails the case. A *review* finding lets the case
finish and marks it Needs Review. An *info* finding only records a measured value. All findings are in the step's
`provenance.json`.

| Step | Check | Default threshold |
|---|---|---|
| Preprocess | axes point the same way as the atlas' | fail otherwise |
| | spacing and size are the OAI DESS ones | review if spacing is 10% off, or size differs |
| | median intensity after windowing | info |
| Segment | each structure is found | fail otherwise |
| | volume | review outside 4–20 ml (FC), 1.5–8 ml (TC) |
| | share of the volume outside the largest piece (FC) or two pieces (TC) | review above 5% |
| | the network's mean uncertainty over the cartilage | review above 0.3 |
| Register | share of the transform that folds space | review above 1% |
| | correlation of the warped image with the atlas | review below 0.75 |
| Mesh | the surface is not empty | fail otherwise |
| Thickness | share of the surface thicker than 6 mm | review above 1% |
| Atlas map | share of the atlas surface without thickness | review above 15% |

The thresholds were tuned on the regression benchmark cases and the cases a Dec 2024 manual review found to have
problems: those problem cases are all flagged, and the clean ones are not. They are stage parameters, so each
project can change them in `project.yaml`.

## Retention and storage

`retention` in `project.yaml` sets what is kept:

| Value | Kept | Size per case |
|---|---|---|
| `checkpoints` (default) | each step's outputs: enough to resume or redo from any step | about 330 MB |
| `all` | also the stages' intermediate outputs (e.g. `FC_inner.vtk`, the surfaces moved into atlas space) | about 340 MB |
| `final` | once a case finishes: the atlas-mapped surfaces, the biomarkers, the thumbnails and every `provenance.json` | about 10 MB |

With `final`, a step that needs deleted files gets them by redoing the earlier steps.

`storage.image_format` (`nrrd` or `nii.gz`) and `storage.transform_format` (`h5` or `tfm`) set the file formats.
The defaults, compressed NRRD and HDF5, keep the values exactly and are small: the old single-folder layout took about 1 GB per
case, mostly in 300 MB text transforms.

## From Python

```python
from oai_analysis.project import Project

project = Project("M:/OAI_projects/pilot")
for s in project.status():
    print(s.case_id, s.status, s.note)
project.run(case_ids=["9016304_96M_LEFT"], from_step="mesh")
```

See the [Python API](api.md).
