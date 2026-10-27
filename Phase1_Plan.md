# Phase 1 plan: library-ification + modular architecture

Status: **approved**, all open questions decided (see section 8). Step 0 in progress.

Sources reviewed:
- *OAI Roadmap 2026-29* (Google Doc): Phase 1 goals and milestones M1.1–M1.4.
- *Notes - OAI pipeline.docx*: meetings from Nov 6 to Dec 18, 2024, including the open action items.
- *Bea's Notes - proposal main points.docx*: the grant's Aim 4 (DataLad repository, Parquet/OME-NGFF formats, web app), a stakeholder wish list (accessibility, logging, QC, scale, reproducibility, reporting, trust), and a barrier analysis.
- `OAI_analysis_2` at `f198a2f` (local `main`, 2 commits ahead of `origin/main`).
- `OAI_analysis` (the older pipeline), `knee-sarg` (the Dagster orchestration), `OAI_results_Marc`, the `OAI_results_2*` and `OAI_results_old` folders, `NAS`, and the top level of `O:\OAI`.

---

## 1. What Phase 1 asks for (from the GDoc)

> Turn OAI_analysis_2 into an installable Python package where each pipeline stage is a standalone module rather than a hardcoded script sequence. This is a refactor, not a rewrite.

| Milestone | Deliverable |
|---|---|
| **M1.1** | A package skeleton with defined interfaces that installs with pip. |
| **M1.2** | At least 2 interchangeable implementations for one stage, which shows the plugin architecture actually works. |
| **M1.3** | Provenance metadata written for every run in human-readable form. A project folder structure the library can parse to tell where a computation stopped. |
| **M1.4** | Documentation for the Kitware → UCSD handoff. |

Dagster (knee-sarg) is explicitly *above* this layer. It stays optional, and Phase 1 must not break it.

### 1.1 What Bea's notes add

Bea's notes list what stakeholders need from the whole project. Most of it lands in Phases 2–3, but several items are cheap *only if the Phase 1 architecture is designed for them*: status vocabulary, sanity-check hooks, data formats and lineage. Phase 1 therefore takes the items below. Where they show up in the plan is in the last column.

| Bea's item | Phase 1 takes | Plan section |
|---|---|---|
| **Aim 4.1:** DataLad provenance-tracked repository | **Not Phase 1** (Q7). The pipeline's own provenance and the raw-input references in `case.json` stand on their own; DataLad can be layered on later. | 7 |
| **Aim 4.2:** Parquet + OME-NGFF formats | A project-level **Parquet** index of case status, QC metrics and scalar biomarkers. An IO layer where an OME-NGFF writer can be added; OME-Zarr output is optional. | 4.6, 4.7 |
| **Aim 4.3:** web visualization app | Nothing to build yet. Outputs are kept web-friendly (Parquet, PNG thumbnails, `.vtk`/`.vtp` meshes) so the app can consume them later. | 7 |
| Plain-language status: Running / Succeeded / Failed / Needs Review | Adopted as the status vocabulary of `oai-analysis status` and the Parquet index. | 4.6 |
| Human-readable logs, per-case status, searchable failures | A plain-English run summary, per-stage status, and `status --failed-at STAGE` filters. | 4.6 |
| Provenance (code version, weights, params) and data lineage to raw inputs | Already in the plan. Adds OAI identifiers (patient, study UID, series UID) and a raw-input hash, so lineage survives if access to the source data is later restricted. | 4.6 |
| Automated sanity checks at each stage; "needs human review" flags | A `check()` hook on every stage, with basic checks shipped for the default stages. | 4.9 |
| Uncertainty / confidence indicators | One cheap metric from the segmentation probabilities (mean uncertainty inside the cartilage), recorded as a QC value. | 4.9 |
| Regression comparison against known-good runs | Step 0's comparison tool becomes `oai-analysis compare` and works on any two projects. | Step 0, 6 |
| Visual outputs by default; per-run/per-subject exportable summary | Per-case thumbnails, plus a CSV export of `status`. The static HTML report is **Phase 2** (Q8). | 4.6, 4.9 |
| Audit trail of who approved/rejected cases; escalation path | **Phase 2** (Q8), together with the report and dashboard. | 7 |
| Small-batch / rerun only failed or changed cases; checkpointing | Already in the plan (resume, staleness). Adds `--failed` and `--stale` selectors. | 4.6 |
| Resource-aware scheduling; runs must not stall silently | A device check at start, per-stage timing in provenance, and a heartbeat in the run log. Real scheduling stays in the orchestrator. | 4.6 |
| Determinism, environment pinning, containers | Seeds set and recorded, an optional deterministic mode, a lock file, and a container recipe. | 4.8 |
| Plain-language change log, "what might look different" | A `CHANGELOG.md` with a reviewer-facing section per release, backed by `compare` results. | Step 7 |
| Batching is hard inside Dagster jobs; risk of Dagster lock-in | Batching lives in the library (`run_cases`), so an orchestrator only hands it a list of cases. There is no orchestrator logic in the package. | 3, 4.4 |
| No-code dashboard, overlay viewers, comparison sliders, aggregate trend dashboards | **Not Phase 1.** These are Phase 2/3, and they build on the Parquet index and the review records. | 7 |
| Harmonization across cohorts (MOST, JoCo, UK Biobank; ComBat, cycle-GANs) | **Not Phase 1.** The `Preprocessor` interface is where harmonizers will plug in. Its checks catch the single-dataset problems already seen: intensity normalization (Dec 4) and orientation/origin metadata (Nov 20). | 4.9, 7 |

---

## 2. Where the code is today

The pipeline works: the Dec 2024 fixes are in, and a 1-case sanity run with updated dependencies was done on Sep 29, 2026 (`OAI_results_2/9016304_0M_left`). Structurally, though, it is a script.

### 2.1 Problems the refactor must fix

| # | Finding | Where |
|---|---|---|
| 1 | **It is not importable as a library.** `pipeline.py` uses top-level imports (`import mesh_processing as mp`, `from analysis_object import ...`), so it only runs as a script from inside `oai_analysis/`. `pipeline_cli.py` uses `from pipeline import ...`. `analysis_object.py` uses `oai_analysis.` imports, so the two styles are mixed. | `pipeline.py:12-15`, `pipeline_cli.py:5` |
| 2 | **The step order is hardcoded** in a 110-line function. Algorithm switches are local variables (`thickness_via_mesh_splitting = True`). This is the Matt "swapped two stages" risk the GDoc mentions. | `pipeline.py:97-206` |
| 3 | **Unused model loading.** `AnalysisObject()` builds an `ICON_Registration` (downloads the gradICON knee model and takes GPU memory) that the pipeline never uses, because `pipeline.py` calls `get_unigradicon()` directly. The atlas image is read twice. | `analysis_object.py:38-41`, `pipeline.py:129-141` |
| 4 | **Models are reloaded for every case.** The segmenter and uniGradICON are rebuilt for each case. The Dec 18 action item "add small batch processing" is still open. | `pipeline.py:123,138` |
| 5 | **Magic numbers are spread across modules**, are not configurable, and are not recorded: percentiles 0.1/99.9, `finetune_steps=2`, iso-value 0.5, 150 smoothing iterations, 10 mm island threshold, a 10k region size, the TC split at x = −50, `ninter=300`, batch 4, overlap (16, 16, 8), and others. | `pipeline.py`, `mesh_processing.py`, `cartilage_shape_processing.py` |
| 6 | **Data and weights depend on the working directory.** `data.py` creates folders inside the package at import time. `pipeline.py` reads atlas files through `DATA_DIR` without calling `atlases_dir()`, so it only works because `AnalysisObject()` happens to trigger the download first. uniGradICON downloads its weights into `./network_weights/` relative to the CWD, which is why several `network_weights` copies exist. | `data.py:5-6`, `pipeline.py:17,129-135`, `unigradicon/__init__.py:267-274` |
| 7 | **There are no provenance records**: no recorded parameters, versions, input hashes or weight hashes. The output folder name is the only record of what produced a result (for example `OAI_results_2_icon_1.1.5` vs `_uniGradICON`). | n/a |
| 8 | **There is no resume or per-case error isolation.** One crash, such as the cuberille crash on 9135902 left 0M, kills the batch loop. | `pipeline.py:244-247` |
| 9 | **Storage is large: about 1.07 GB per case** with intermediates. Each `.tfm` text transform is about 300 MB, and the prob maps are uncompressed 189 MB `.nrrd` files. For about 48k cases that comes to roughly 50 TB. This is the "TBs of data" problem from Phase 2. | `OAI_results_2/9016304_0M_left` |
| 10 | **Dead or broken code.** `AVSM_Registration` references undefined names. `dask_processing.py` imports `oai_analysis.utils`, which doesn't exist. `cartilage_shape_processing.map_thickness_to_2D_projection` uses `mp` without importing it on the `str` path. | `registration.py:7-16`, `dask_processing.py:133`, `cartilage_shape_processing.py:94` |
| 11 | **The tests are weak.** `test_MeshThicknessCPU` and `test_RegistrationCPU` have their asserts commented out, there is no pipeline-level test, and the hatch scripts point at `tests/` while the folder is `test/`. | `test/test_all.py` |
| 12 | **Packaging inconsistencies.** The license is MIT in `pyproject.toml` but Apache-2.0 in `LICENSE`, the README and the SPDX header. `requires-python >=3.8`, but the code is used on 3.11. Exact `==` pins sit in the library's dependencies. `oai_analysis/data/` is gitignored, so hatch probably leaves the atlas `.ply` files out of the wheel (needs checking). | `pyproject.toml` |

### 2.2 Assets we can build on

- The algorithms are validated and were reviewed against Marc's old results in Nov–Dec 2024. Phase 1 keeps them **numerically unchanged** by default.
- **Second implementations already exist in the code.** These make M1.2 cheap and meaningful:
  - **Thickness:** mesh-splitting (current default) vs distance-transform (`thickness_computation.py`, selectable only by editing a variable).
  - **Registration:** uniGradICON (default) vs the gradICON OAI-knee model (`registration.ICON_Registration`). Both were run before, in `OAI_results_2_uniGradICON` vs `OAI_results_2_icon_1.1.5`.
  - **Meshing:** cuberille (default) vs marching cubes (`mesh_processing.get_mesh`).
- **A case manifest already exists.** `knee-sarg/data/oai-sampler/study_uid_to_vol_path.csv` has 48,067 rows (`patient_id, study_uid, vol_path, month, study_description`), and laterality can be derived from the study description. This is the full cohort the GDoc's "48k cases" refers to.
- **Reference data:**
  - The 12-case list in `pipeline.py:__main__`, read from `O:\OAI` (I spot-checked one path).
  - `OAI_results_2_2025` (main as of Feb 2025, older dependencies).
  - `OAI_results_Marc`: 11 patients with all time points and both knees (64 GB).
  - The bundled `colab_case`.
- **knee-sarg's contract**, which Phase 1 must keep working: it runs `python ./oai_analysis/pipeline_cli.py IMAGE OUTDIR left|right` and then checks that `FC_thickness.png` and `TC_thickness.png` exist in `OUTDIR`.

---

## 3. Guiding principles

1. **Refactor, not rewrite.** Algorithm code moves behind interfaces with its numbers unchanged. Any intentional numerical change, such as the `vtkWindowedSincPolyDataFilter` switch from the Dec 18 notes, becomes a *config option that is off by default*. It is evaluated separately.
2. **Every structural PR passes the regression check** (Step 0) on the benchmark cases before it merges.
3. **Keep the legacy entry points working.** `pipeline_cli.py IMAGE OUT laterality` and `analysis_pipeline(...)` keep their behavior and flat output layout, so knee-sarg needs no changes.
4. **Headless and scriptable first**, per the GDoc: the user is a data scientist, not a clinician.
5. **Provenance is part of the design.** The runner writes it, not each stage, so plugins get it for free.
6. **No orchestrator inside the library.** Dagster, SLURM or a plain loop all call the same `OAPipeline.run_cases()`. Batching, resume and provenance live in the library. That removes the Dagster batching problem and the lock-in risk Bea's notes point out.
7. **Statuses a reviewer can read.** Every status, QC flag and log summary is written so a non-engineer can act on it.

---

## 4. Target architecture

### 4.1 Package layout

```
oai_analysis/
  __init__.py              # exports OAPipeline, Project, CaseInfo, __version__
  __main__.py              # python -m oai_analysis ...
  cli.py                   # `oai-analysis` console script (run, init, status, compare, config)
  pipeline.py              # OAPipeline: wires the stages; no algorithm code
                           #   + analysis_pipeline(): legacy wrapper (same signature as today)
  pipeline_cli.py          # legacy CLI shim kept for knee-sarg (3 positional args)
  config.py                # dataclass configs, YAML load/dump
  registry.py              # name -> class lookup, entry-point plugin discovery
  provenance.py            # provenance records, hashing, environment capture
  project.py               # project folder layout, manifest, status/resume/staleness
  io.py                    # image (DICOM dir / NIfTI / NRRD), mesh, transform IO
  data.py                  # pooch-managed atlases + model weights in a user cache dir
  atlas.py                 # Atlas object (image, inner meshes, 2D embeddings), loaded once
  stages/
    base.py                # the abstract stage interfaces
    preprocessing.py       # OAIPreprocessor
    segmentation/          # UNetSegmenter (+ existing networks.py, image_transforms.py, ...)
    registration.py        # UniGradICONRegisterer, GradICONKneeRegisterer
    meshing.py             # CuberilleMeshGenerator, MarchingCubesMeshGenerator
    thickness.py           # MeshSplittingThickness, DistanceTransformThickness
    atlas_mapping.py       # AtlasMapper (patient mesh -> atlas mesh attributes)
    biomarkers.py          # ThicknessMap2D (current 2D projection), ...
  algorithms/              # the existing modules, moved unchanged apart from imports:
    mesh_processing.py
    thickness_computation.py
    cartilage_shape_processing.py
```

`dask_processing.py`, `analysis_object.py` and `AVSM_Registration` are removed. `AnalysisObject` is replaced by `OAPipeline`, and a deprecation alias is kept for one release if the notebooks need it.

### 4.2 Stages and their interfaces

The GDoc names four interfaces: Preprocessor, Segmenter, MeshGenerator and BiomarkerExtractor. The actual pipeline has three more real steps, so Phase 1 uses **seven stage types**. Their order is fixed by the pipeline, and each declares what it consumes.

```
            ┌────────────┐   ┌───────────┐   ┌────────────┐
 input ───▶ │ Preprocess │──▶│  Segment  │──▶│    Mesh    │──┐
            └────────────┘   └───────────┘   └────────────┘  │
                  │                                           ▼
                  │          ┌───────────┐           ┌──────────────┐   ┌─────────────┐   ┌──────────────┐
                  └────────▶ │ Register  │──────────▶│  Thickness   │──▶│ Atlas map   │──▶│ Biomarkers[] │
                             └───────────┘  (phi)    └──────────────┘   └─────────────┘   └──────────────┘
```

**How today's code maps onto the stages:**

| Today | Stage interface | Default implementation (Phase 1) | Second implementation |
|---|---|---|---|
| `into_canonical_orientation` + `preprocess` (`pipeline.py:46-80`) | `Preprocessor` | `OAIPreprocessor` | none |
| `AnalysisObject.segment` → `Segmenter3DInPatchClassWise` | `Segmenter` | `UNetSegmenter` | `PrecomputedSegmenter` (reads prob maps from disk, e.g. Marc's `FC_probmap.nii.gz`) |
| `get_unigradicon()` + `register_pair(finetune_steps=2)` (`pipeline.py:138-141`) | `Registerer` | `UniGradICONRegisterer` | `GradICONKneeRegisterer` (from `registration.ICON_Registration`) |
| `get_mesh_from_probability_map` + `itk_mesh_to_vtk_mesh` | `MeshGenerator` | `CuberilleMeshGenerator` | `MarchingCubesMeshGenerator` (from `mp.get_mesh`) |
| `pipeline.py:157-175` (to atlas → split → back → distance) | `ThicknessEstimator` | `MeshSplittingThickness` | `DistanceTransformThickness` (from `thickness_computation.py`, `pipeline.py:176-184`) |
| `pipeline.py:190-202` (transform to atlas + `map_attributes`) | `AtlasMapper` | `ClosestPointAtlasMapper` | none |
| `thickness_3d_to_2d` (`pipeline.py:204-206`) | `BiomarkerExtractor` (a list) | `ThicknessMap2D` | user code (see 4.5) |

**Interface definitions** (`stages/base.py`, abridged):

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from typing import ClassVar

import itk
import vtk


class Stage(ABC):
    """Base for every pipeline stage. Implementations are dataclasses whose fields are their parameters."""

    name: ClassVar[str]          # registry key, e.g. "unigradicon"
    version: ClassVar[str] = "1"  # bump whenever the implementation's output changes

    def params(self) -> dict:
        """Parameters recorded in provenance and used for staleness checks."""
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def setup(self, device: str) -> None:
        """Load heavy resources (networks, weights) once per process. Default: nothing."""

    def check(self, outputs: dict, case: "CaseInfo") -> list["QCFinding"]:
        """Sanity-check this stage's outputs (see 4.9). Default: no findings."""
        return []


class Preprocessor(Stage):
    @abstractmethod
    def run(self, image: itk.Image, case: "CaseInfo") -> itk.Image: ...


class Segmenter(Stage):
    @abstractmethod
    def run(self, image: itk.Image) -> dict[str, itk.Image]:
        """Return probability maps keyed by structure: {"FC": ..., "TC": ...}."""


@dataclass
class RegistrationResult:
    phi_AB: itk.Transform  # resampling transform (patient -> atlas image)
    phi_BA: itk.Transform  # modelling transform (moves patient meshes into atlas space)


class Registerer(Stage):
    @abstractmethod
    def run(self, moving: itk.Image, fixed: itk.Image) -> RegistrationResult: ...


class MeshGenerator(Stage):
    @abstractmethod
    def run(self, probability: itk.Image) -> vtk.vtkPolyData: ...


class ThicknessEstimator(Stage):
    @abstractmethod
    def run(self, mesh: vtk.vtkPolyData, probability: itk.Image, registration: RegistrationResult,
            atlas: "Atlas", structure: str) -> vtk.vtkPolyData:
        """Return a patient-space mesh carrying a per-vertex thickness array."""


class AtlasMapper(Stage):
    @abstractmethod
    def run(self, thickness_mesh: vtk.vtkPolyData, registration: RegistrationResult,
            atlas: "Atlas", structure: str) -> vtk.vtkPolyData:
        """Return the atlas mesh carrying the patient's thickness."""


class BiomarkerExtractor(Stage):
    @abstractmethod
    def run(self, case: "CaseArtifacts") -> dict[str, object]:
        """Read anything produced so far (read-only); return named outputs to be saved."""
```

### 4.3 Concrete example: today's block → a stage

**Today** (`pipeline.py:157-172`, inline, with output paths threaded through):

```python
thickness_via_mesh_splitting = True
if thickness_via_mesh_splitting:
    fc_mesh_atlas = transform_mesh(fc_mesh, phi_BA, output_path + "/FC_mesh", False)
    tc_mesh_atlas = transform_mesh(tc_mesh, phi_BA, output_path + "/TC_mesh", False)
    fc_inner_atlas, fc_outer_atlas = mp.get_split_mesh(fc_mesh_atlas, inner_mesh_fc_atlas, mesh_type='FC')
    ...
    fc_inner, fc_mesh = mp.get_distance(fc_inner_patient, fc_outer_patient)
```

**After** (`stages/thickness.py`). The algorithm calls are identical, there is one structure per call, and the hidden constants become fields:

```python
@register("thickness_estimator")
@dataclass
class MeshSplittingThickness(ThicknessEstimator):
    """Split the surface into inner/outer in atlas space; measure inner-outer distance in patient space."""

    name: ClassVar[str] = "mesh_splitting"
    version: ClassVar[str] = "1"
    smoothing_iterations: int = 150          # was the default of mp.get_split_mesh
    island_distance_threshold_mm: float = 10.0

    def run(self, mesh, probability, registration, atlas, structure):
        mesh_atlas = transform_mesh(mesh, registration.phi_BA)
        inner_atlas, outer_atlas = mp.get_split_mesh(
            mesh_atlas, atlas.inner_mesh[structure], mesh_type=structure,
            num_iterations=self.smoothing_iterations,
            distance_threshold_mm=self.island_distance_threshold_mm,
        )
        inner = transform_mesh(inner_atlas, registration.phi_AB)
        outer = transform_mesh(outer_atlas, registration.phi_AB)
        _, outer_with_thickness = mp.get_distance(inner, outer)
        return outer_with_thickness
```

`transform_mesh` loses its `filename_prefix` and `keep_intermediate_outputs` arguments. It uses a temp file internally, the same round trip as today, so results stay bit-compatible. Intermediate outputs are saved by the runner instead.

**Today** (`pipeline.py:137-141`) vs **after**. The model is loaded once in `setup()` and reused for every case, and the weights come from the managed cache instead of the CWD:

```python
@register("registerer")
@dataclass
class UniGradICONRegisterer(Registerer):
    name: ClassVar[str] = "unigradicon"
    finetune_steps: int = 2

    def setup(self, device):
        self._model = get_unigradicon(weights_location=str(data.weights("unigradicon1.0")))

    def run(self, moving, fixed):
        phi_AB, phi_BA = itk_wrapper.register_pair(
            self._model, moving.astype(itk.D), fixed.astype(itk.D), finetune_steps=self.finetune_steps)
        return RegistrationResult(phi_AB, phi_BA)
```

### 4.4 The pipeline object

```python
from oai_analysis import OAPipeline, CaseInfo

pipeline = OAPipeline()  # all defaults: identical to today's pipeline.py
pipeline.run_case(CaseInfo(input="O:/OAI/OAI96MonthImages/results/10.C.1/9016304/20120831/13486606",
                           laterality="left", case_id="9016304_96M_LEFT"),
                  output_dir="M:/OAI_projects/scratch/9016304_96M_LEFT")
```

You can swap stages by passing instances:

```python
from oai_analysis.stages.registration import GradICONKneeRegisterer
from oai_analysis.stages.thickness import DistanceTransformThickness

pipeline = OAPipeline(
    registerer=GradICONKneeRegisterer(),
    thickness_estimator=DistanceTransformThickness(),
)
```

You can also build the pipeline from a config file, MONAI-bundle style, with a registry name plus kwargs:

```yaml
# project.yaml (excerpt). `oai-analysis config dump` prints the full default config
pipeline:
  preprocessor:        {name: oai_default, window_min_percentile: 0.1, window_max_percentile: 99.9}
  segmenter:           {name: unet3d_oai, batch_size: 4, overlap_size: [16, 16, 8]}
  registerer:          {name: unigradicon, finetune_steps: 2}
  mesh_generator:      {name: cuberille, iso_surface_value: 0.5}
  thickness_estimator: {name: mesh_splitting, smoothing_iterations: 150}
  atlas_mapper:        {name: closest_point}
  biomarker_extractors:
    - {name: thickness_map_2d, grid_resolution: 300}
device: cuda
retention: checkpoints        # final | checkpoints | all  (see 4.7)
```

```python
pipeline = OAPipeline.from_config("project.yaml")
```

`OAPipeline` holds the fixed stage order, so stages cannot be accidentally swapped. It calls `setup()` once on every stage before the first case, and it delegates saving, provenance and resume to `Project` (4.6).

To process several cases, `pipeline.run_cases(project, case_ids)` loads the models once and works through the list. This is the single entry point for every orchestrator: the CLI, a notebook loop, a Dagster asset that receives a batch of partition keys, or a SLURM array job that receives a slice of `cases.csv`.

### 4.5 Plugins and user code (the GDoc's `OAPipeline(biomarker_extractor=statisticalAnalysis)`)

There are three ways to plug in, from least to most formal:

1. **Pass an instance.** Nothing to register, which suits notebooks:

   ```python
   import numpy as np
   from dataclasses import dataclass
   from typing import ClassVar
   from oai_analysis import OAPipeline
   from oai_analysis.stages import BiomarkerExtractor, ThicknessMap2D

   @dataclass
   class MeanThickness(BiomarkerExtractor):
       name: ClassVar[str] = "mean_thickness"
       version: ClassVar[str] = "0.1"

       def run(self, case):
           return {"mean_thickness.json": {
               s: float(np.nanmean(case.atlas_thickness(s))) for s in ("FC", "TC")}}

   pipeline = OAPipeline(biomarker_extractors=[ThicknessMap2D(), MeanThickness()])
   ```

2. **Use `@register` in an imported module.** The class is then usable by name in `project.yaml`.
3. **Declare an entry point in another package.** For example, Hongtu's group could ship `oai-gwas` that declares:

   ```toml
   [project.entry-points."oai_analysis.plugins"]
   gwas = "oai_gwas.stages"
   ```

   After `pip install oai-gwas`, `{name: gwas_analysis}` works in `project.yaml` with no change to this repo. This is the Phase 3 payoff the GDoc describes.

### 4.6 Project folder structure, provenance and resume (M1.3)

**Layout.** A project is a folder. Everything the library needs to know is in it:

```
M:/OAI_projects/pilot/
  project.yaml                 # pipeline config + data roots + retention policy
  cases.csv                    # manifest: case_id, patient_id, month, laterality, input
  runs.jsonl                   # append-only: one line per `oai-analysis run` invocation
  index.parquet                # one row per case: status, failed stage, QC flags/metrics, scalar biomarkers
  logs/2026-11-03T140311.log          # engineering log (DEBUG)
  logs/2026-11-03T140311.summary.txt  # plain-English run summary
  cases/
    9016304/
      96M_LEFT/
        case.json              # CaseInfo: input path + raw-input hash, laterality, OAI patient/study/series UIDs, ...
        thumbnails/            # segmentation overlay + 2D maps, small PNGs
        01_preprocess/   image.nii.gz                         provenance.json
        02_segment/      FC_prob.nii.gz  TC_prob.nii.gz        provenance.json
        03_register/     phi_AB.h5  phi_BA.h5                  provenance.json
        04_mesh/         FC_mesh.vtk  TC_mesh.vtk              provenance.json
        05_thickness/    FC_thickness.vtk  TC_thickness.vtk    provenance.json
        06_atlas_map/    FC_atlas.vtk  TC_atlas.vtk            provenance.json
        07_biomarkers/
          thickness_map_2d/  FC_thickness.npy  FC_thickness.png  TC_...  provenance.json
          mean_thickness/    mean_thickness.json               provenance.json
```

**Creating a project from the existing knee-sarg manifest:**

```
oai-analysis init M:/OAI_projects/pilot --data-root O:/OAI ^
    --manifest M:/Dev/Osteoarthritis/knee-sarg/data/oai-sampler/study_uid_to_vol_path.csv ^
    --patients 9016304,9021791,9040390
```

**Provenance record** (`03_register/provenance.json`). The runner writes it after a stage succeeds, and a stage counts as complete only if this file exists:

```json
{
  "stage": "register",
  "implementation": "unigradicon",
  "implementation_version": "1",
  "status": "complete",
  "params": {"finetune_steps": 2},
  "inputs": {
    "moving": {"path": "../01_preprocess/image.nii.gz", "sha256": "4f1c…"},
    "fixed":  {"path": "atlas:atlas_60_LEFT_baseline_NMI/atlas_image.nii.gz", "sha256": "9a0b…"}
  },
  "outputs": {
    "phi_AB": {"path": "phi_AB.h5", "sha256": "…"},
    "phi_BA": {"path": "phi_BA.h5", "sha256": "…"}
  },
  "weights": {"unigradicon1.0/Step_2_final.trch": "sha256:…"},
  "fingerprint": "c81e…",
  "software": {"oai_analysis": "2.1.0.dev14+gf198a2f", "git_dirty": false,
               "python": "3.11.x", "torch": "2.14.0+cu132", "itk": "5.4.7", "vtk": "9.7.1",
               "icon_registration": "1.1.8", "unigradicon": "1.0.5"},
  "environment": {"host": "…", "device": "cuda:0", "gpu": "…", "seed": 0, "deterministic": false},
  "qc": [],
  "started": "2026-11-03T14:03:11-05:00",
  "finished": "2026-11-03T14:08:02-05:00",
  "duration_s": 291.4
}
```

`fingerprint` is a hash of the implementation name, version and params, plus the input hashes. It drives the **staleness** check: when the config changes, that stage and everything downstream of it is marked stale and rerun.

**Status and resume:**

Statuses use the plain-language words from Bea's notes: **Succeeded**, **Needs Review** (succeeded, but a QC check flagged it), **Failed**, **Running**, **Outdated** (the config changed since the run) and **Not started**.

```
> oai-analysis status M:/OAI_projects/pilot
12 cases: 8 Succeeded, 1 Needs Review, 1 Failed, 1 Outdated, 1 Not started

case                 status         prep seg reg mesh thick atlas bio  note
9016304_00M_LEFT     Succeeded       ✓    ✓   ✓   ✓    ✓     ✓     ✓
9103365_96M_RIGHT    Needs Review    ✓    ✓   ✓   ✓    !     ✓     ✓   TC thickness > 6 mm on 3% of vertices
9135902_00M_LEFT     Failed          ✓    ✓   ✓   ✗                     Mesh generation crashed (CuberilleError)
9040390_96M_LEFT     Outdated        ✓    ✓   ~                         registerer.finetune_steps changed 2 -> 5
9094865_96M_RIGHT    Not started

> oai-analysis status M:/OAI_projects/pilot --failed-at mesh       # "all cases that failed at step X"
> oai-analysis status M:/OAI_projects/pilot --needs-review --csv flagged.csv
> oai-analysis run M:/OAI_projects/pilot                          # resume: skips Succeeded, redoes the rest
> oai-analysis run M:/OAI_projects/pilot --failed                 # small batch: only Failed cases
> oai-analysis run M:/OAI_projects/pilot --cases 9135902_00M_LEFT --from mesh
```

The same table is written to `index.parquet` after every case, so pandas, DuckDB or a later dashboard can filter it without parsing logs.

**Plain-English run summary** (`logs/*.summary.txt`), written as the run progresses:

```
Run started 2026-11-03 14:03 on <host> (GPU: <name>), oai-analysis 2.1.0.dev14+gf198a2f, config project.yaml
12 cases selected (11 to do, 1 already done).
[14:09] 9016304_00M_LEFT   Succeeded in 5m51s.
[14:15] 9135902_00M_LEFT   Failed at "mesh generation": the cuberille filter crashed. Earlier steps are kept; rerun with --cases 9135902_00M_LEFT --from mesh.
[14:21] 9103365_96M_RIGHT  Needs Review: tibial cartilage thickness above 6 mm on 3% of the surface.
...
Run finished 15:12: 9 Succeeded, 1 Needs Review, 1 Failed. Details: logs/2026-11-03T140311.log
```

The runner also checks at startup that the requested device is available. It refuses to fall back to CPU silently, unlike today's warning in `AnalysisObject`. It writes a heartbeat line every few minutes during long GPU stages, so a stalled run is visible.

`Project.status()` and `Project.case(case_id).artifacts` are the same thing as a Python API, for notebooks.

**Batch behavior.** Models are loaded once per process. A failure in one case is logged with a traceback in that stage's `provenance.json` (`"status": "failed"`) and the batch moves on. This closes the Dec 18 "small batch processing" action item.

### 4.7 Storage and retention

File formats are **configurable per artifact type**. The defaults favor size and speed, and the alternatives favor debugging:

| Artifact | Today | Options | Default |
|---|---|---|---|
| Transforms | `.tfm` text, about 300 MB each | `.h5` (ITK HDF5, compressed) or `.tfm` (text, easy to inspect) | `.h5`. Lossless, expected several times smaller (to be measured). |
| Images (prob maps, preprocessed image, thickness image) | uncompressed `.nrrd`, 94–189 MB | `.nii.gz` or `.nrrd`, both compressed at the library's default level | `.nii.gz` |
| Meshes | binary legacy `.vtk` | unchanged | `.vtk` |

Final outputs that are kept for a long time, such as the `ThicknessMap2D` arrays and the atlas-mapped meshes, can use a **higher compression level** than intermediates. It is a separate setting, so intermediates stay fast to write and read:

```yaml
storage:
  transform_format: h5            # h5 | tfm
  image_format: nii.gz            # nii.gz | nrrd
  compression_level: default      # intermediates
  final_compression_level: 9      # long-lived outputs (biomarkers, atlas meshes)
```

ITK's `imwrite` takes a `compression_level`, `.npy` arrays become compressed `.npz` when a level is set, and VTK writers support compression for XML formats. Step 4 measures the real size and time effect of each level before the defaults are fixed.

`retention` decides what is kept after a case **completes**:
- `all` keeps every intermediate, like `keep_intermediate_outputs=True` today.
- `checkpoints` (the default) keeps whatever is needed to resume from any stage.
- `final` keeps only the atlas-mapped meshes, the biomarkers and all `provenance.json` files.

Provenance records the hashes of deleted intermediates, so lineage survives pruning. The legacy CLI keeps its exact current filenames and formats.

All file writing goes through `io.py` writers chosen by artifact type. That keeps Aim 4.2 open: an **OME-NGFF (OME-Zarr)** image writer can be added as another `image_format` option (`zarr` gets added back as a dependency then; Step 1 removed it as unused) without touching any stage. Scalar results (status, QC metrics, per-case biomarker summaries) go to **Parquet** from the start (adds a `pyarrow` dependency).

### 4.8 Packaging and data (M1.1)

- **Distribution name.** Keep `oai-analysis` / `import oai_analysis`; it already exists on PyPI and knee-sarg depends on it (see Q1).
- **Version from git** via `hatch-vcs` (for example `2.1.0.dev14+gf198a2f`), so provenance always names the commit.
- **Python and dependencies.** `requires-python >= 3.11` (older versions are deprecated). Library dependencies become ranges. Exact versions for reproducible environments move to a lock or constraints file, generated from venv11. torch stays unpinned in the library, since its CUDA builds have to be installed separately (as `repro.cmd` already does).
- **Data and weights.** All data and weights come from a pooch-managed **user cache** (`OAI_ANALYSIS_DATA` env var override) with sha256 checks. Nothing is written into the package or the CWD. Atlas files are loaded once through `Atlas.default()`.
- **Console script.** `oai-analysis` (`run`, `init`, `status`, `compare`, `config dump`).
- **License.** Fix `pyproject.toml` to Apache-2.0 to match `LICENSE`.
- **Reproducible environment.** A lock file for the full stack, and a container recipe (Dockerfile with CUDA torch; knee-sarg already has one to start from). Provenance records the container image digest when it runs in one.
- **Determinism.** The runner sets and records seeds (torch, numpy; KMeans already uses `random_state=5`). An optional `deterministic: true` setting turns on deterministic torch/cuDNN algorithms. It will be slower, and Step 0 measures how much noise remains (uniGradICON fine-tuning).
- **Check:** `pip install .` (non-editable) into a fresh venv, then run the bundled case from a different directory.

### 4.9 Sanity checks and review flags

**Stage checks.** After a stage succeeds, the runner calls its `check()` method. A finding is either `fail` (stop this case) or `review` (continue, mark the case *Needs Review*). Findings are stored in `provenance.json` and `index.parquet`. The default stages ship these checks, with thresholds in the config:

| Stage | Check | Why (from the notes) |
|---|---|---|
| Preprocess | orientation is the expected one after reorientation; spacing and size are in the normal OAI DESS range; intensity percentiles are recorded | Nov 20 orientation/origin question, Dec 4 normalization bug |
| Segment | each structure is non-empty; its volume is within a plausible range; the number of connected components; **mean uncertainty** inside the cartilage (from the probabilities) | Nov 6 segmentation failures; Bea's "confidence indicator" |
| Register | Jacobian determinant > 0 everywhere (no folding); similarity to the atlas after warping | 9040390 registration imperfection |
| Mesh | non-empty; island count after filtering; self-intersections (VTK collision check) | cuberille crash, self-intersection in 9021791 (PR80) |
| Thickness | thickness range, and the fraction of vertices above N mm or at exactly 0 | "zeroes around edges" |
| Biomarkers | NaN fraction of the 2D map | holes in the 2D projection |

```python
@dataclass
class QCFinding:
    severity: Literal["review", "fail"]
    code: str          # machine-readable, e.g. "thickness.too_thick"
    message: str       # plain English, shown in status, summary and report
    value: float | None = None
```

These are deliberately simple per-case checks. Statistical outlier detection across the cohort (z-scores over descriptors) is Phase 2 and will reuse the same `QCFinding` records and Parquet index.

**Thumbnails.** Each case gets small PNGs in `thumbnails/`: a segmentation overlay on the middle slice, plus the FC/TC 2D thickness maps. They are cheap to make while the data is in memory, and the Phase 2 report and dashboard will use them.

**Moved to Phase 2 (Q8):** the static HTML report (`oai-analysis report`) and the review records (`review.jsonl`, `oai-analysis review CASE --accept/--reject/--flag`). Phase 1 leaves room for both: `index.parquet` and the `QCFinding` records are what they will read.

---

## 5. Second implementations (M1.2)

The GDoc's example is "2 segmentation methods". The repo has only one segmentation network, while real second implementations already exist for three other stages. The proposal:

| Stage | Second implementation | Why it is a meaningful test of the plugin design |
|---|---|---|
| **Registerer** | `GradICONKneeRegisterer` | It is a different model with different weights and the same interface. A comparison already exists (`OAI_results_2_icon_1.1.5` vs `_uniGradICON`). |
| **ThicknessEstimator** | `DistanceTransformThickness` | It is a different algorithm family (image-based vs mesh-based) and needs different inputs (it does not use the registration). |
| **Segmenter** | `PrecomputedSegmenter` | It reads existing prob maps (Marc's `FC_probmap.nii.gz`/`TC_probmap.nii.gz`). It is small, it makes old-vs-new comparisons easy, and it lets the CPU tests skip the network. |

A **second segmentation network**, such as a MONAI or nnU-Net model, would need training or a pretrained OAI model that isn't here, so it is treated as optional (Q2).

Acceptance: a test runs the colab case with each combination selected **only by config**, and provenance records which one ran.

---

## 6. Work breakdown

Each step is one or more PRs to `uncbiag/OAI_analysis_2` through the `dzenanz` fork. Each PR must pass the Step 0 regression check.

### Step 0: Regression safety net (a lean slice of the GDoc's Phase 0, done first)

The refactor needs a reference before anything moves.

- `tools/compare_outputs.py OLD_DIR NEW_DIR` compares two outputs in today's flat layout:
  - prob maps: max abs diff and Dice at 0.5
  - meshes: point count and mean/max closest-point distance
  - 2D thickness maps: agreement of the NaN masks and mean/95th-percentile abs diff, using ITK image comparison as the Dec 18 notes suggest

  It prints a pass/fail table with tolerances.
- Run current `main` (`f198a2f`, venv11) on the **12 cases** in `pipeline.py:__main__` from `O:\OAI` plus the `colab_case`, **twice**. The two runs measure GPU run-to-run noise (uniGradICON fine-tuning is not deterministic), which sets the tolerances. Store the result as `M:\Dev\Osteoarthritis\OAI_golden_f198a2f\` (outside git) along with timings.
- Done when: one command regenerates outputs and prints a comparison against the golden set.

**Status: done** (2026-10-26, branch `regression-baseline`).
- `tools/run_baseline.py OUT --oai-dir O:/OAI [--cases …] [--compare-to REF]` runs the cases from `tools/benchmark_cases.json`. It records the commit, packages, GPU, per-case timings and failures in `OUT/run_info.json`. With `--compare-to`, it then writes `OUT/comparison.md` and exits non-zero on a failure.
- `tools/compare_outputs.py REF TEST` compares images, meshes and 2D maps. The 2D maps are compared as numpy arrays rather than with ITK image comparison. Limits come from `tools/regression_tolerances.json`.
- The golden set is `OAI_golden_f198a2f/run1` and `run2` (27 GB): 13/13 cases succeeded in each run, at about 4.6 min per case on the RTX 5060 Ti (about 1 h per run).
- **Measured noise.** Preprocessing, segmentation and the patient-space meshes are **bit-identical** between the runs, so they must match exactly. Run-to-run variation starts at uniGradICON registration. Downstream of it, the largest differences between runs were:

  | Output | Largest run-to-run difference (mm) |
  |---|---|
  | 2D thickness maps | 0.035 mean, 0.12 at the 95th percentile, 2.6 at one pixel |
  | Thickness on the atlas mesh | 0.038 mean, 0.14 at the 95th percentile |

  These differences set the tolerances at about 2× the largest value seen.
- **Check against older outputs.** Comparing run1 with the Feb 2025 outputs (`OAI_results_2_2025`, older dependencies) correctly fails the exact-match stages. Probability maps differ by up to 0.0015 and patient-space meshes by up to 0.2 mm, from the Sep 2026 dependency update. All post-registration outputs are within the noise.
- If tighter checks on the post-registration stages are wanted later, deterministic registration (4.8, `deterministic: true`) would allow them.
- In Step 4 this tool becomes `oai-analysis compare PROJECT_A PROJECT_B`, which also works on the new layout. That is the reusable "compare to prior known-good runs" from Bea's notes, and the source for the reviewer-facing change log.

### Step 1: Package hygiene (M1.1a)

- Switch to absolute imports everywhere. `pipeline.py`, `pipeline_cli.py` and `python -m oai_analysis` all work from any CWD.
- Move `data.py` to the user cache. Route uniGradICON and gradICON weights through it. Add `Atlas.default()`.
- Delete the dead code listed in finding #10, and fix the `mp` import in `cartilage_shape_processing.py`.
- Update `pyproject.toml`: license, Python floor, dependency ranges, `hatch-vcs`, the console script, a `[dev]` extra, and the hatch test path.
- Done when: a clean-venv non-editable install runs the colab case, and Step 0 passes on all 12 cases. Nothing should change numerically here.

**Status: done** (2026-10-27, branch `refactor`).
- Absolute imports throughout (including `segmentation/segmenter.py`). Added `oai_analysis/__main__.py` and the `oai-analysis` console script, which keeps the `pipeline_cli.py` arguments knee-sarg uses.
- `data.py` fetches into `pooch.os_cache("oai-analysis")`, which `OAI_ANALYSIS_DATA` overrides. It creates nothing at import time. The uniGradICON and GradICON-knee weights are now in its registry with sha256 checks, and `pipeline.py` passes `weights_location`, so `./network_weights` is never created.
- `atlas.py`: `Atlas.default()` loads the atlas image, `inner_mesh[...]` and `embedding_2d[...]` once per process. The atlas files tracked in git moved to package data in `oai_analysis/resources/`: `atlas_image.nii.gz`, the LPS/smooth `.ply` meshes, the 2D embeddings, and `atlas_fc`/`atlas_tc`. The atlas therefore needs no download, and the code stays simpler. The tests use it too, so `data.py` no longer fetches the release-tarball atlases.
- Deleted `AVSM_Registration`, `dask_processing.py`, and the Dask notebooks that only used it. `AnalysisObject` creates the GradICON registerer only when `register()` is called, and it gets its atlas from `Atlas.default()`. Fixed the missing `mp` import in `cartilage_shape_processing.py`.
- `pyproject.toml`: Apache-2.0, Python >=3.11, dependency ranges (floor = baseline version, below the next major), and the missing direct dependencies added (numpy, scipy, matplotlib). `zarr` was dropped as unused. Also: hatch-vcs, the console script, the `[dev]` extra, `test/` paths, and an explicit wheel/sdist content list. The wheel is 81 MB, mostly the atlas image, and stays under PyPI's 100 MB default file limit.
- **The version needs a tag on `main`.** `v2.0.0` sits on the old `master` branch, so hatch-vcs currently reports `0.1.dev221+g…`. Tagging `main` (for example `v2.0.1` once these PRs land) fixes that.
- Verification:
  - All 13 cases pass `compare_outputs` against `OAI_golden_f198a2f/run1`, 195/195 files, run from an unrelated CWD.
  - A clean Python 3.11 venv with a non-editable wheel install (which resolved the newer `icon_registration` 1.1.9 and `unigradicon` 1.0.6) ran the colab case through the `oai-analysis` console script from an unrelated CWD and passes against the golden set.
  - `test_SegmentationCPU` fails at 12.12 vs a limit of 12. The f198a2f golden prob map gives the same 12.12, so this failure predates Step 1 and comes from the Sep 2026 dependency update. It is left for Step 6.

### Step 2: Stage interfaces + default implementations + `OAPipeline` (M1.1b)

- Add `stages/base.py` (4.2) and move each block from the table in 4.2 behind its interface, with the algorithm calls unchanged.
- `OAPipeline` with in-memory execution, which keeps today's behavior and calls `setup()` once.
- `analysis_pipeline()` and `pipeline_cli.py` become thin wrappers that write today's flat filenames.
- Done when: Step 0 passes, knee-sarg's command line produces `FC_thickness.png`/`TC_thickness.png`, and processing N cases loads each model once.

**Status: done** (2026-10-27, branch `refactor`).
- `oai_analysis/stages/`: `base.py` has the seven interfaces from 4.2, plus `CaseInfo`, `RegistrationResult`, `CaseArtifacts`, and `MeshResult` (a mesh plus named intermediate outputs, such as the inner surface, for the runner to save). The default implementations call the algorithm functions unchanged: `OAIPreprocessor`, `UNetSegmenter`, `UniGradICONRegisterer`, `CuberilleMeshGenerator`, `MeshSplittingThickness`, `ClosestPointAtlasMapper` and `ThicknessMap2D`. The hidden constants became fields (window percentiles, batch and overlap size, fine-tuning steps, iso-surface value, smoothing iterations, island distance, grid resolution).
- `DistanceTransformThickness` also moved now, because its code was the unused `else` branch in `pipeline.py`. `GradICONKneeRegisterer` waits for Step 5; `registration.ICON_Registration` is unchanged.
- `pipeline.py` holds `OAPipeline` and no algorithm code. `setup()` runs once per pipeline object, before the first case. `run_case()` returns a `CaseArtifacts` with everything in memory and, given an output directory, writes today's flat file names as each stage finishes. `analysis_pipeline()` keeps its signature and reuses one cached default pipeline, so a loop over cases loads the models once. `oai_analysis.OAPipeline` and `oai_analysis.CaseInfo` are imported lazily, so `import oai_analysis.data` stays light.
- `mesh_processing.transform_mesh(mesh, transform)` replaces the pipeline's version. It keeps the same ITK write/read round trip, but through a temp directory instead of the output directory.
- `ThicknessMap2D` returns the 2D grid and the PNG bytes instead of writing files. `cartilage_shape_processing` gained `thickness_grid()` and `thickness_grid_png()` for that; `thickness_3d_to_2d()` still works for the notebooks.
- `io.py`: reading the input, the DICOM series description, and one `write()` that picks the writer by type.
- `CaseInfo` accepts only `left` or `right`. Before, anything other than `right` was treated as left.
- `AnalysisObject` is a deprecated shim over `UNetSegmenter`, kept for the notebooks. The tests use the stages directly.
- Not in this step: `check()`/`QCFinding` (Step 4b) and `@register` (Step 3).
- A follow-up commit moved the files into the 4.1 layout, with no code changes beyond imports: `mesh_processing.py`, `thickness_computation.py` and `cartilage_shape_processing.py` are now in `oai_analysis/algorithms/`, and the `segmentation/` package is now `oai_analysis/stages/segmentation/`, next to `UNetSegmenter` (`unet.py`). Code and notebooks that import the old paths must switch to `oai_analysis.algorithms.mesh_processing` and so on.
- Verification:
  - All 13 cases pass `compare_outputs` against `OAI_golden_f198a2f/run1`, 195/195 files, run through `analysis_pipeline()` from an unrelated CWD. The segmentation model was loaded once for all 13 cases. Cases took 226–300 s, 57 min in total.
  - knee-sarg's command line (`oai-analysis IMAGE OUT left`) on the colab case writes `FC_thickness.png`/`TC_thickness.png` and passes against the golden set.
  - `pytest`: 15 passed (8 of them the new `test/test_stages.py`, which checks the stage wiring, the flat file names and the one-time `setup()` with lightweight fake stages). `test_SegmentationCPU` still fails at 12.12 vs 12, as before Step 1.

### Step 2b: Split by cell IDs instead of transforming the halves back (separate commit)

This change is mainly about performance. Today `MeshSplittingThickness` transforms the patient mesh into atlas space and splits it there. It then transforms **both halves back** to patient space (`pipeline.py:167-170`: 4 extra `transform_mesh` calls per case, each a disk round trip through ITK) before measuring distances.

The new version tags the patient mesh with an `original_cell_id` cell-data array before the atlas-space transform:

- `vtkPolyDataConnectivityFilter` (island filtering) and `vtkSmoothPolyDataFilter` both pass cell data through.
- `split_femoral_cartilage_surface` and `split_tibial_cartilage_surface` already compute `inner_face_list` and `outer_face_list`, but `split_mesh()` throws them away. They get returned.
- The face lists are mapped through `original_cell_id`, and the halves are cut out of the **original patient-space mesh** with `get_vtk_sub_mesh`. No inverse transform is needed.

**This is an intentional numerical change, not just a speedup.** Distances are then measured on the unsmoothed patient mesh, not on the smoothed mesh moved back from atlas space. This is the Dec 18 action item "transfer splitting IDs to the original mesh". It also removes the ~2 mm smoothing migration from the thickness measurement. As the notes say, values may get somewhat noisier, because today's distances come from smoothed meshes.

Proposed placement: **right after Step 2**, as one commit on `MeshSplittingThickness`:
- `version` goes from "1" to "2".
- A `back_mapping: "cell_ids" | "transform"` parameter, defaulting to `"cell_ids"`, keeps the old path available for comparison.
- Doing it after Step 2 keeps the main refactor bit-compatible, and the change is confined to one stage class.
- Done when: `compare_outputs` against the Step 0 baseline is written up (timing saved per case, and how the thickness maps differ), and the old path with `back_mapping: transform` still passes Step 0.

### Step 3: Config + registry + plugins

- Add `config.py` (dataclasses ↔ YAML), `registry.py` with `@register`, and the `oai_analysis.plugins` entry-point discovery.
- Add `oai-analysis config dump`, and an `OAPipeline.from_config()` round-trip test.
- Done when: the default config reproduces Step 0, and a plugin in a throwaway test package is discovered through its entry point.

### Step 4: Project structure + provenance + resume + batch (M1.3)

- Add `project.py`: init from a manifest (knee-sarg CSV format), the folder layout, `status`, resume, staleness and `--from STAGE`.
- Add `provenance.py`: hashing, environment and version capture, weight hashes and failure records.
- Storage formats and retention (4.7).
- Plain-language statuses, `index.parquet`, the run summary, the device check and heartbeat, and the `--failed`/`--stale`/`--failed-at` selectors.
- Done when: on a 12-case project, killing the run midway and rerunning resumes where it stopped. Changing one parameter reruns only the affected stages. The cuberille-crash case ends as *Failed* without stopping the batch. `oai-analysis status` matches the folder contents.

### Step 4b: Sanity checks and thumbnails

- `check()` on the default stages (4.9), `QCFinding`, the *Needs Review* status, and per-case thumbnails.
- Tune the thresholds on the benchmark cases, using the Dec 18 triage as a reference: the Group 3 cases should come out *Needs Review* and the clean cases should not.
- Done when: `oai-analysis status` for the 12-case project shows the expected Succeeded / Needs Review / Failed split, and every flag has a plain-English message.

### Step 5: Second implementations (M1.2)

- `GradICONKneeRegisterer`, `DistanceTransformThickness`, `PrecomputedSegmenter`, and optionally `MarchingCubesMeshGenerator`.
- Done when: the config-only switching test from section 5 passes, and there is a short comparison write-up on the benchmark cases.

### Step 6: Tests and CI (ongoing from Step 1)

- Unit tests per stage on small data.
- A CPU smoke test of the full pipeline on the colab case using `PrecomputedSegmenter`, so the GitHub-hosted runners stay fast.
- Restore the commented-out asserts.
- A GPU job on the self-hosted runner that runs Step 0 on 1–2 cases.

### Step 7: Documentation and handoff (M1.4)

- Plain Markdown in `docs/`, readable on GitHub with no build step (Q5), with these pages:
  - install (including the CUDA torch step)
  - quick start (one case, then a project)
  - the pipeline steps and their order (closes the Dec 18 "update readme pipeline-steps" item)
  - the config reference, generated from the dataclasses
  - writing a plugin (the 4.5 example)
  - project layout and provenance
  - an API overview (details stay in docstrings)
- **Handoff guide** for UCSD: environment setup on their hardware, running a pilot batch, reading `status`, where results and provenance live, and known failure modes (cuberille crash, smoothing migration).
- Update the notebooks to the new API, or mark them as legacy.
- `CHANGELOG.md` with two parts per release: the engineering changes, and a plain-language **"What might look different"** section for reviewers, backed by `oai-analysis compare` numbers on the benchmark cases.
- A one-page reviewer guide for non-engineers: reading `status` and its CSV export, what *Needs Review* means, and what to send engineers when flagging a case (case ID, stage, note, screenshot).

---

## 7. Deliberately *not* in Phase 1

- **Algorithm changes.** Smoothing migration, the windowed-sinc filter, the cuberille crash fix and the 0-thickness edges stay as they are. They become config options or tracked issues and are evaluated with the Step 0 harness once the refactor lands.
- **Dagster / Dask / SLURM orchestration.** knee-sarg keeps working through the legacy CLI. Running the new `oai-analysis run` from Dagster is a later step (the GDoc's Phase 4 backlog).
- **1D descriptors, outlier flagging and the QC dashboard** belong to Phase 2. The `BiomarkerExtractor` slot and the project layout are designed so they plug in.
- **Interactive UI** (Bea's no-code dashboard, triggering reruns from a browser, overlay viewers, comparison sliders, aggregate trend dashboards) and the **Aim 4.3 web app** (BIG-KP, itk-wasm, vtk.js) belong to Phases 2–3. Phase 1 supplies their data: `index.parquet`, the QC findings, thumbnails and atlas-space meshes.
- **The static HTML report and the review records** (audit trail, accept/reject/flag) move to Phase 2 (Q8).
- **DataLad (Aim 4.1)** in any form (Q7). Nothing in Phase 1 creates, saves to or tests against a DataLad dataset.
- **OME-NGFF as the default image format.** The writer slot exists, but switching formats waits until a viewer needs it.
- **Cross-cohort harmonization** (MOST, JoCo, UK Biobank; ComBat, cycle-GANs, mm-DeepAtlas).
- **Batching inside Dagster itself.** Phase 1 gives Dagster `run_cases()`; changing knee-sarg to use it is a later step.
- **A full 48k-case run.**

---

## 8. Open questions

| # | Question | My recommendation |
|---|---|---|
| Q1 | Keep the `oai-analysis` / `oai_analysis` name, or rename (the GDoc says "oa-toolbox or similar")? | **Keep.** It is already on PyPI and knee-sarg uses it. A rename can be done at any point with a shim. |
| Q2 | Is M1.2 satisfied by the registration + thickness implementations plus `PrecomputedSegmenter`, or do you want a second *segmentation network*? | Registration + thickness. A second network is a separate, data-dependent effort. |
| Q3 | Where should the work land: feature branches on `dzenanz/OAI_analysis_2` with PRs to `uncbiag` main, one PR per step? There are also 2 unpushed commits on local `main` to deal with first. | Push or PR the 2 commits first, then one PR per step. |
| Q4 | Is `requires-python >= 3.10` acceptable? knee-sarg's README still says 3.8, but venv11 is 3.11. | **Decided: 3.11 or newer.** Older versions are deprecated. |
| Q5 | Docs tooling: MkDocs (Material + mkdocstrings), or plain Markdown in `docs/`? | **Decided: plain Markdown**, the simpler choice. |
| Q6 | Is it OK for the *new* project layout to use different file formats (`.h5` transforms, `.nii.gz` images), while the legacy CLI keeps today's? | Yes. That is the biggest storage saving. |
| Q7 | Should DataLad be optional (`--datalad`) or required in Phase 1? | **Decided: neither.** DataLad is not part of Phase 1 at all. |
| Q8 | Do the static HTML report and the review records (4.9) belong in Phase 1, or should they move to Phase 2 with the dashboard? | **Decided: Phase 2.** Phase 1 keeps the per-stage checks and thumbnails. |
| Q9 | Container recipe: Docker (from knee-sarg's Dockerfile), or a pixi environment like knee-sarg uses? | **Decided: Docker.** |
