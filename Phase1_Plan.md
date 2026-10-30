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
- Run current `main` (`f198a2f`, venv11) on the **12 cases** in `pipeline.py:__main__` from `O:\OAI` plus the `colab_case`, **twice**. The two runs measure GPU run-to-run noise (uniGradICON fine-tuning is not deterministic), which sets the tolerances. Store the result as `M:\Dev\Osteoarthritis\Refactoring\OAI_golden_f198a2f\` (outside git) along with timings.
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

**Status: done** (2026-10-27, branch `refactor`, commit `b8bc907`).
- `mesh_processing.get_split_cell_ids()` returns the inner and outer cell ids of the patient mesh. It tags `original_cell_id` on a shallow copy, so the input is not modified. `split_mesh()` is now built on `split_mesh_face_lists()`, which returns the face lists that used to be thrown away. `MeshSplittingThickness` is version "2", with `back_mapping` defaulting to `"cell_ids"`.
- `transform_mesh` keeps the point and cell order exactly, which is what makes the cell ids valid in patient space. The cell_ids path makes 1 `transform_mesh` call per structure instead of 3.
- `CaseArtifacts.seconds` records the wall time of each stage, and `tools/run_baseline.py` saves it per case in `run_info.json` (`--back-mapping` picks the mode).
- New test: `test_split_cell_ids_match_split_mesh`. With no smoothing, the halves cut by cell id are identical to the halves from `split_mesh()`.
- Verification (13 cases, against `OAI_golden_f198a2f/run1`):
  - `back_mapping: transform`: 13/13 cases pass (`M:/Dev/Osteoarthritis/Refactoring/OAI_step2b_transform`).
  - `back_mapping: cell_ids` (`M:/Dev/Osteoarthritis/Refactoring/OAI_step2b_cell_ids`): as expected, it does not pass the bit-compatibility tolerances. The files that differ are the thickness-related ones: `*_inner.vtk`, `*_outer.vtk`, `*_mesh_atlas.vtk`, `*_mapped_mesh.vtk` and `*_thickness.npy`. The probability maps, transforms and patient meshes are unchanged.
  - Thickness maps: mean absolute difference 0.12–0.32 mm per map, 95th percentile 0.33–0.93 mm, maximum up to 3.8 mm. The halves are now the unsmoothed patient surface, about 0.3 mm from the smoothed surfaces used before.
  - Bias: femoral cartilage comes out thicker in every case, by +0.03 to +0.16 mm in the map mean (about +6%). Tibial cartilage has no consistent direction, from -0.09 to +0.10 mm. Over all 26 maps, the mean shift is +0.04 mm (+2.4%).
  - Time: the thickness stage averages 22.6 s per case with cell ids, against 24.3 s with the transform path (range 16–39 s against 19–40 s), about 7% faster. Whole cases average 214 s against 233 s. Registration takes about 200 s of each case, so that gap is mostly run-to-run noise in the registration stage rather than this change.
  - Before relying on the new values, someone should look at the FC bias. The shift is the expected effect of measuring on unsmoothed meshes, but it is systematic, not noise.

### Step 3: Config + registry + plugins

- Add `config.py` (dataclasses ↔ YAML), `registry.py` with `@register`, and the `oai_analysis.plugins` entry-point discovery.
- Add `oai-analysis config dump`, and an `OAPipeline.from_config()` round-trip test.
- Done when: the default config reproduces Step 0, and a plugin in a throwaway test package is discovered through its entry point.

**Status: done** (2026-10-28, branch `refactor`).
- `registry.py`: `@register` makes a stage class available by its kind and name. The kind comes from the interface (a `kind` class attribute on each interface in `base.py`), so it is a bare `@register` rather than `@register("thickness_estimator")`. Registering a different class under a name that is taken is an error. Plugins are the modules named by `oai_analysis.plugins` entry points. They are imported the first time a name is not found, and a plugin that fails to import gives a warning instead of stopping the others.
- `config.py` reads and writes the YAML format from 4.4: one entry per stage, `name` plus parameters, and `device` (`auto` picks CUDA when available). Settings that are left out keep their defaults, and an entry with no `name` keeps the default implementation, so `registerer: {finetune_steps: 5}` changes one parameter. Unknown keys, stage names and parameters stop with a message listing the valid ones. YAML lists become tuples again where the parameter's default is a tuple.
- `OAPipeline.from_config(path_or_dict)` and `OAPipeline.config()`. The atlas is now loaded on first use, so creating a pipeline (e.g. to print its config) is cheap.
- `cli.py` is the `oai-analysis` entry point: `oai-analysis config dump [--config FILE]` prints the full config, and the original `oai-analysis IMAGE OUT left|right` still works, now with `--config FILE`. `pipeline_cli.py` keeps the original command.
- New dependency: `pyyaml`. `tools/run_baseline.py` gained `--config` and records the config in `run_info.json`.
- The project-level keys of 4.4 (`retention`, `storage`, data roots) are not accepted yet. They arrive with Step 4, which also decides how `project.yaml` holds them.
- Verification:
  - Since Step 2b the default `back_mapping` is `cell_ids`, which intentionally differs from the golden set. So the check used the full dumped default config with only `back_mapping: transform` changed (`M:/Dev/Osteoarthritis/Refactoring/OAI_step3_transform.yaml`): 13/13 cases and 195/195 files pass against `OAI_golden_f198a2f/run1` (`M:/Dev/Osteoarthritis/Refactoring/OAI_step3_config_transform`), at 234–314 s per case.
  - `oai-analysis IMAGE OUT left --config` with a two-line partial config, on the colab case from an unrelated CWD, passes against the golden set.
  - `test/test_config.py` (12 tests): dump/load round trip, partial configs, error messages, the built-in registry, `config dump`, and a throwaway plugin package (a module plus a `.dist-info` with `entry_points.txt` on `sys.path`) whose stage is found through its entry point and named in a config. `pytest`: 29 passed; `test_SegmentationCPU` still fails as before.

### Step 4: Project structure + provenance + resume + batch (M1.3)

- Add `project.py`: init from a manifest (knee-sarg CSV format), the folder layout, `status`, resume, staleness and `--from STAGE`.
- Add `provenance.py`: hashing, environment and version capture, weight hashes and failure records.
- Storage formats and retention (4.7).
- Plain-language statuses, `index.parquet`, the run summary, the device check and heartbeat, and the `--failed`/`--stale`/`--failed-at` selectors.
- Done when: on a 12-case project, killing the run midway and rerunning resumes where it stopped. Changing one parameter reruns only the affected stages. The cuberille-crash case ends as *Failed* without stopping the batch. `oai-analysis status` matches the folder contents.

**Status: done** (2026-10-28, branch `refactor`).
- `pipeline.py`: a case is now a list of named steps (`preprocess`, `segment`, `register`, `mesh`, `thickness`, `atlas_map`, and one `biomarkers/NAME` per extractor), each with the steps whose outputs it reads. `OAPipeline.run_step()` and `load_step()` run a step or read back its saved outputs. `run_case()` and the flat layout use the same steps, and the bundled case still passes Step 0 through `analysis_pipeline()`.
- `provenance.py`: file and folder hashing, software versions and git commit, host/device/GPU, and the fingerprint. A fingerprint covers the stage name, version, parameters, `weights()` (new on `Stage`: the pooch sha256 of the U-Net and uniGradICON weights), the atlas files, and the hashes of the input files it reads (the raw input for `preprocess`, otherwise the outputs of earlier steps, read from their provenance). So a step is redone when its config changes or when anything it reads was redone with different results.
- `project.py`: `Project.create()` / `from_manifest()` (knee-sarg CSV; case IDs like `9016304_00M_LEFT`, folders `cases/9016304/00M_LEFT/`), `status()`, `run()`, `index.parquet`, `runs.jsonl`, and `logs/RUN.log` + `logs/RUN.summary.txt`.
  - Statuses: Succeeded, Failed, Outdated, Running, Not started, and **Incomplete**, which is new: some steps are done and the run stopped before the rest. Outdated says what changed, e.g. `thickness_estimator.back_mapping changed transform -> cell_ids`.
  - **Cases run in a worker process.** The 9135902 left 0M case crashed the whole process (a segmentation fault) on one run and raised `IndexError: invalid vector subscript` in cuberille on another, so an in-process `try/except` is not enough. The supervisor starts a worker that loads the models once and works through the cases. If the worker dies, the supervisor marks the step it was on (from the case's `running.json`) as failed with the exit code, and starts a new worker for the remaining cases. `--in-process` skips this, e.g. for tests and debugging.
  - `running.json` in the case folder has a heartbeat, updated every minute. A case is Running only while that is fresh and, on the same host, the worker process is alive, so a killed run shows as Incomplete at once. Steps that take longer than 5 minutes log that they are still working.
  - Writing a step clears its folder first, and `provenance.json` is written last and atomically, so a killed run never leaves a step that looks complete.
  - The device check: `device: auto` with no GPU stops with a message instead of falling back to the CPU; `device: cpu` must be asked for.
  - `BiomarkerExtractor.requires` names the steps an extractor reads (default: all). `ThicknessMap2D` reads only `atlas_map`, so with `retention: final` adding or changing it does not redo the case.
- Storage (4.7), measured on 9016304 96M left:
  - Transforms: compressed `.h5` is 67 MB and reads in 0.4 s; `.tfm` is 304 MB and reads in 13 s. Same transform after the round trip. Default `h5`.
  - Images: compressed NRRD and `.nii.gz` are about the same size (prob map 83 vs 80 MB, preprocessed image 33 vs 30 MB), but `.nii.gz` changes the origin by 1.6e-6 mm and the direction by 3e-8 (NIfTI stores them in single precision), while NRRD is exact. **The default is `nrrd`, not `nii.gz`**, so that resuming from saved files gives the same numbers as an uninterrupted run. `nii.gz` is an option.
  - Compression level made no difference to NIfTI sizes (levels 1, 6 and 9 gave the same files), so `compression_level` and `final_compression_level` were left out.
  - A case with `retention: all` takes 339 MB, against 1051 MB in the flat layout.
  - ITK's HDF5 writer cannot compress a transform with no parameters (e.g. identity), so those are written uncompressed.
  - HDF5 transform files are not byte-identical when rewritten (the image formats are), so if a pruned registration is rerun, the steps after it are rerun too. Registration is not deterministic anyway.
- Retention: `checkpoints` (default) writes each step's outputs but not the stage intermediates (`FC_inner`, `FC_mesh_atlas`, thickness images); `all` writes those too; `final` deletes everything but the atlas-mapped meshes, biomarkers and provenance once a case succeeds. A step whose inputs were deleted gets them by rerunning the earlier steps.
- `cli.py`: `oai-analysis init PROJECT --manifest CSV --data-root DIR [--patients] [--months] [--laterality] [--config]`, `oai-analysis run PROJECT [--cases] [--failed] [--stale] [--failed-at STEP] [--from STEP] [--in-process]`, `oai-analysis status PROJECT [same selectors] [--csv FILE]`.
- `tools/export_flat.py` links a project's outputs into the flat layout, so `compare_outputs.py` can compare them with the golden set.
- Found while testing: after a case wrote its thickness PNG with `pyplot`, Tk objects were garbage-collected on the heartbeat thread and Tk aborted the process (`Tcl_AsyncDelete: async handler deleted by the wrong thread`), once after the segmentation of the next case had hung for 11 minutes. `thickness_grid_png()` now draws on a plain matplotlib `Figure`, which needs no GUI toolkit, and the worker runs with `MPLBACKEND=Agg`.
- New dependency: `pyarrow`.
- Not done: recording the container image digest and setting/recording seeds (4.8). They fit with the Docker work.
- Verification, on a 13-case project (`M:/Dev/Osteoarthritis/Refactoring/OAI_step4_project`: the 12 benchmark cases plus 9135902 left 0M, `back_mapping: transform`, `retention: all`):
  - Killed with `taskkill /F /T` during the registration of the 4th case. `status`: 3 Succeeded, 1 Incomplete (preprocessing and segmentation done), 9 Not started. The next `run` kept every finished step (provenance files unchanged) and finished the 4th case from registration.
  - In that run, 9021791 left 0M hit the Tk abort above (exit code 0x80000003): it was marked Failed at segmentation, a new worker processed the remaining 8 cases, and `run --failed` then finished it from segmentation. 9135902 left 0M failed at mesh generation (`IndexError`) both times without stopping the batch; the retry redid only that step.
  - `export_flat.py` + `compare_outputs.py` against `OAI_golden_f198a2f/run1`: 12/12 cases, 180/180 files pass.
  - Changing `back_mapping` to `cell_ids` in `project.yaml`: `status` showed 12 Outdated, with the reason. `run --stale` redid thickness, atlas mapping and the 2D maps of the 12 cases in 7 minutes. The provenance of the first four steps was unchanged, and the results pass against the Step 2b `cell_ids` run (12/12, 180/180).
  - `status` matched the folders at every point. `pytest`: 38 passed (9 in `test/test_project.py`, including a plugin whose stage kills the worker process with `os._exit`), plus the known `test_SegmentationCPU` failure.

**Follow-up: batches instead of a worker process** (2026-10-28, branch `refactor`). A separate process per run cost performance and prevented batching, and the crashes should mostly go away with a newer ITKCuberille, so the supervisor/worker is gone (`--in-process` too).
- Cases now run in batches, all in one process. `batch_size` in `project.yaml` (default 8) or `run --batch-size N` sets the batch size. A batch runs preprocessing case by case, then segmentation for all its cases, then registration for all of them. After that, each case finishes on its own.
  - A batch keeps each case's image, probability maps and transforms in memory, about 0.5 GB per case.
  - If a batched step raises, its cases are rerun one by one, so only the bad case fails.
  - A crash in compiled code stops the run. The cases show as Incomplete, and the next `run` resumes them.
- The stage interfaces gained `Segmenter.run_batch(images)` and `Registerer.run_batch(movings, fixed)`. By default they loop over `run()`. The pipeline gained `OAPipeline.run_step_batch()` and `BATCHED_STEPS = ("segment", "register")`. Each provenance record notes the `batch_size` it ran with, and `duration_s` is the case's share of the batch.
- `UNetSegmenter.run_batch` sends the patches of all images through the network `batch_size` patches at a time. On 4 cases the results were bit-identical to `run()`, but no faster (56 s either way): the network is GPU-bound already. 8 patches per pass reached 8.1 GB and spilled out of GPU memory (1227 s for 4 images), so the default stays 4.
- `UniGradICONRegisterer.run_batch`:
  - **With fine-tuning (the default, `finetune_steps: 2`), pairs are registered one by one.** Fine-tuning tunes a copy of the network on each pair and peaks at 10.5 GB. Tuning one copy on several pairs would change the results, and two pairs would not fit in memory anyway.
  - With `finetune_steps: 0` (now accepted; it means no fine-tuning), `batch_size` pairs (new parameter, default 4) go through the network together. On 8 cases: 8.1 s against 39.4 s one by one, peak 6.6 GB. Transformed points differ by at most 0.004 mm.
- **The real speed-up: releasing GPU memory between networks.** Segmentation left 13 GB in PyTorch's cache, so registration reserved 26 GB on the 16 GB card. Windows made up the difference with system memory, and registration took 180–190 s instead of 17 s. The pipeline now calls `torch.cuda.empty_cache()` after segmentation and registration, which also helps `analysis_pipeline()`.
- `Stage.performance_params` lists the parameters that change how fast a stage runs but not what it computes (`batch_size` of both networks). They are left out of the fingerprint, so changing them does not make outputs outdated.
  - Removing the U-Net's `batch_size` from its fingerprint makes earlier projects' segmentations Outdated once. `status` says so, rather than "its inputs changed".
- What the stages print, warnings included, goes to the run's full log (`logs/RUN.log`), as it did with the worker. The console gets the progress and summary lines.
- Verification, on a fresh copy of the 13-case project (`M:/Dev/Osteoarthritis/Refactoring/OAI_batch_project`, same `project.yaml`, which has `back_mapping: cell_ids`, batches of 8):
  - 12 cases succeeded. 9135902 left 0M failed at mesh generation (`IndexError`) without stopping the run.
  - The results pass against the Step 2b `cell_ids` run (12/12 cases, 180/180 files).
  - Each case took about 65 s, against about 4.5 min in Step 4. Average per step: preprocessing 6.5 s, segmentation 17.6 s (was 23.5), registration 11.1 s (was 180), mesh 1.9 s, thickness 21.8 s, atlas mapping 1.0 s.
  - The first run crashed (segmentation fault) while preprocessing 9094865 right 0M, in batch 2. `status` showed 8 Succeeded, 3 Incomplete, 2 Not started. The next `run` did only the 5 remaining cases, and 9094865 right 0M succeeded this time.
  - `pytest`: 40 passed, plus the known `test_SegmentationCPU` failure. New tests cover batches (one `run_batch` call per batch, each case getting its own outputs), a failing batch falling back to one by one, and `performance_params`.

**Follow-up: batches without GPU batching** (2026-10-29). A batch was meant to run each network over its cases one after another, with the weights staying in GPU memory, not to pass several images through a network at once. GPU batching gained nothing with fine-tuning (the default), so it is gone:
- Removed `Segmenter.run_batch`, `Registerer.run_batch`, their U-Net and uniGradICON overrides, the uniGradICON `batch_size` parameter, `OAPipeline.run_step_batch` and the fall-back to one by one. Provenance records no longer carry `batch_size`; `runs.jsonl` still does.
- A batch now runs each step up to registration for each of its cases in turn (preprocess all, segment all, register all), then finishes each case on its own. A failing step fails only its case, as for any other step.
- Kept: `torch.cuda.empty_cache()` once segmentation, then registration, is done with the batch's cases (`release_gpu_memory()`; the 180 s → 11 s gain; `OAPipeline.run()` calls it after each network), `batch_size` in `project.yaml` and `run --batch-size N`, and `performance_params` (for the U-Net's patches-per-pass `batch_size`).
- Verification: 3 cases of `OAI_batch_project` rerun from preprocessing in one batch ran preprocess ×3, segment ×3, register ×3, then each case to the end; all succeeded. Segmentation took 18 s per case and registration 15–20 s. Registration is slower than the 11 s of 2026-10-28 on this machine today with the previous commit too (24 s), so the change costs nothing. `pytest`: 40 passed, plus the known `test_SegmentationCPU` failure.

### Step 4b: Sanity checks and thumbnails

- `check()` on the default stages (4.9), `QCFinding`, the *Needs Review* status, and per-case thumbnails.
- Tune the thresholds on the benchmark cases, using the Dec 18 triage as a reference: the Group 3 cases should come out *Needs Review* and the clean cases should not.
- Done when: `oai-analysis status` for the 12-case project shows the expected Succeeded / Needs Review / Failed split, and every flag has a plain-English message.

**Done** (2026-10-29, branch `refactor`). What was built, and where it departs from 4.9:
- `QCFinding(severity, code, message, value)` with a third severity, `"info"`: a value that is only recorded (volumes, uncertainty, correlation, ...). `Stage.check(case: CaseArtifacts)` gets everything produced so far, not just the stage's outputs, because some checks need more (registration: the image and the atlas). Shared measurements are in `oai_analysis/qc.py`.
- Thresholds are stage parameters listed in the new `Stage.check_params`. Like `performance_params` they stay out of the fingerprint. A project records them in `provenance.json` (`check_params`) next to the findings (`qc`). When they change, `status` says Outdated ("… changed 0.05 -> 0.1; run again to recheck") and `run` rechecks the saved outputs without redoing the step. Outputs from before Step 4b are Outdated once ("not checked yet"); rechecking takes about 10 s per case.
- A "fail" finding fails the step (its outputs are kept), and `status` shows the message. "review" findings make a finished case **Needs Review**, mark their step `!` in `status`, and appear in `index.parquet` (`review` column) and the run summary. `status --needs-review` selects those cases, and `run` leaves them alone like Succeeded ones.
- Checks of the default stages:

| Stage | Check | Threshold (parameter) |
|---|---|---|
| Preprocess | axes point the same way as the atlas' | fail otherwise |
| | spacing and size are the OAI DESS ones | review (`expected_spacing_mm` ±10%, `expected_size`) |
| | median intensity after windowing | info |
| Segment | the structure is not empty | fail |
| | volume | review outside 4–20 ml FC, 1.5–8 ml TC (`fc_volume_ml`, `tc_volume_ml`) |
| | share of the volume outside the largest piece (FC) or two (TC) | review above 5% (`max_outside_main_pieces`) |
| | mean uncertainty 1 − \|2p − 1\| over the cartilage | review above 0.3 (`max_uncertainty`) |
| Register | share of the displacement field that folds (Jacobian ≤ 0) | review above 1% (`max_folding`) |
| | correlation of the warped image with the atlas | review below 0.75 (`min_similarity`) |
| Mesh | the surface is not empty | fail |
| Thickness | share of vertices thicker than 6 mm | review above 1% (`max_thickness_mm`, `max_too_thick`) |
| Atlas mapping | share of the atlas surface without thickness (< 0.1 mm) | review above 15% (`max_without_thickness`) |

- Left out: mesh self-intersection (VTK has no cheap self-intersection test; the 9021791 PR80 case is fine by every other measure), mesh island counts (they mirror the segmentation pieces), and the NaN fraction of the 2D map, which the atlas' shape fixes: 40.0% (FC) and 41.0% (TC) on every case.
- Thumbnails (`oai_analysis/thumbnails.py`), in each case's `thumbnails/` folder: `segment_overlay.png` (the sagittal slice with the most cartilage, FC red, TC blue) and `atlas_map_FC.png` / `atlas_map_TC.png` (the 2D thickness maps on a fixed 0–5 mm scale, so cases compare). A failure to draw them is only logged.
- Tuning, on `M:/Dev/Osteoarthritis/Refactoring/OAI_qc_project`: the 13 benchmark cases plus the 6 Dec 18 Group 3 cases they lacked (9094865 right 96M, 9103365 right 96M, 9135902 right 96M, 9530445 left 0M, left 36M, right 72M). 9103365 right 96M has two DESS series in the manifest; both are in, the second as `9103365_96M_RIGHT_B`.
  - The two checks that separate the Group 3 cases are segmentation pieces and the atlas surface without thickness. Clean cases stay at or below 0.7% outside the main pieces and 11.3% without thickness. Every Group 3 case is above 5% or 15% on at least one of them.
  - The other thresholds catch gross failures and flag none of the clean cases: uncertainty there is at most 0.23, correlation at least 0.83, folding at most 0.18%, thickness never above 5.2 mm.
- Result (`status` after `run` rechecked the outputs): 10 Succeeded, 9 Needs Review, 1 Failed. All 9 Group 3 cases (counting 9103365's two series) need review. 9135902 left 0M fails at mesh generation as before. Group 2's 9016304 right 96M (the dent the old pipeline smoothed) needs review too, with 22.9% of its femoral cartilage outside the largest piece. The other 10 benchmark cases succeed. The 12-case project's own split is the same, since it holds those same outputs: 10 Succeeded, 2 Needs Review (9016304 left and right 96M), 1 Failed.

### Step 5: Second implementations (M1.2)

- `GradICONKneeRegisterer`, `DistanceTransformThickness`, `PrecomputedSegmenter`, and optionally `MarchingCubesMeshGenerator`.
- Done when: the config-only switching test from section 5 passes, and there is a short comparison write-up on the benchmark cases.

**Done** (2026-10-29, branch `refactor`).
- New stages, each chosen by name in a config:
  - `gradicon_knee` (`GradICONKneeRegisterer`): the GradICON OAI-knee model from `registration.gradicon_knee_model()`. `finetune_steps: 0` (the default) uses the network as trained, as the pipeline did before uniGradICON.
  - `precomputed` (`PrecomputedSegmenter`): reads each structure's probability map from `path`, a template filled in with `{structure}`, `{case_id}`, `{laterality}`, `{LATERALITY}` and the columns of `cases.csv`, e.g. `{patient_id}` and `{month}`. It resamples the maps onto the preprocessed image. The files are not part of the fingerprint, so after they change, use `run --from segment`.
  - `marching_cubes` (`MarchingCubesMeshGenerator`): scikit-image marching cubes, written anew in `mp.marching_cubes_mesh()`. The old `mp.get_mesh()` ignored the image's origin and direction, smoothed the surface and dropped pieces under 3000 cells. The new function places vertices in physical space and turns the faces outward, as cuberille's are (the inner/outer split relies on the normals). Smoothing is an option, off by default.
  - `distance_transform` already existed (Step 1).
- Interface changes:
  - `Segmenter.run(image, case=None)` now receives the case, which `PrecomputedSegmenter` needs to find its files. Segmenters that ignore it work as before.
  - `CaseInfo.details` carries a project's `cases.csv` row.
  - The segmentation, registration and mesh checks moved into shared functions, so every implementation of a stage checks the same things.
- `test/test_second_implementations.py` runs the colab case through a project twice, once with the defaults and once with all four second implementations, chosen only by `project.yaml`. Each time it checks the implementation recorded in each step's `provenance.json`. It needs a CUDA GPU and is skipped without one.

**Comparison on the 13 benchmark cases.** Each variant is a copy of the Step 4 project (`M:/Dev/Osteoarthritis/Refactoring/OAI_step5_*`) with one stage changed in `project.yaml`. `run` redid only the changed step and the ones after it; for example, `gradicon_knee` left preprocessing, segmentation and meshing alone. Differences are on the 2D thickness maps, against the default project:

| Variant | Succeeded / Needs Review / Failed (default: 10 / 2 / 1) | Step time, default → variant | Map abs diff, FC / TC (mean; mean 95th pct) | Mean thickness FC, TC (default 1.66, 1.63 mm) |
|---|---|---|---|---|
| `gradicon_knee` | 9 / 3 / 1 | 14.4 s → 8.2 s | 0.23 mm / 0.26 mm; 0.81 / 0.84 | 1.66, 1.57 mm |
| `distance_transform` | 10 / 2 / 1 | 20.6 s → 12.8 s | 0.87 mm / 1.00 mm; 1.96 / 2.08 | 2.53, 2.63 mm |
| `marching_cubes` | 11 / 2 / 0 | 2.0 s → 1.2 s | 0.04 mm / 0.05 mm; 0.13 / 0.16 | 1.66, 1.62 mm |
| `precomputed` (Step 0 golden maps) | 10 / 2 / 1 | 17.8 s → 7.4 s | 0.00 / 0.00 | 1.66, 1.63 mm |

- **`precomputed`** reproduces the default results to the hundredth of a millimetre from the maps of the original pipeline (`OAI_golden_f198a2f/run1`). 9135902 left 0M fails because that run has no maps for it.
- **`marching_cubes`** agrees with cuberille to 0.03–0.08 mm per case, and it gets through 9135902 left 0M, the case where cuberille fails (IndexError).
- **`gradicon_knee`** stays close on most cases (0.1–0.2 mm). The largest differences are in 9040390, the case the Dec 18 triage put down to registration (uniGradICON vs ICON), and in the tibial cartilage of Group 3's 9016304 left 96M (1.43 → 1.00 mm). It flags 9040390 right 96M: no tibial thickness on 17.9% of the atlas surface. Its correlation with the atlas is lower on 9016304 left 96M (0.708, against 0.778 with uniGradICON).
- **`distance_transform`** measures a different quantity: about 0.9 mm (around 55%) thicker on every case. It is consistent between cases but not comparable with the mesh-based thickness. Its QC flags match the default's.

### Step 6: Tests and CI (ongoing from Step 1)

- Unit tests per stage on small data.
- A CPU smoke test of the full pipeline on the colab case using `PrecomputedSegmenter`, so the GitHub-hosted runners stay fast.
- Restore the commented-out asserts.
- A GPU job on the self-hosted runner that runs Step 0 on 1–2 cases.

**Done** (2026-10-30, branch `refactor`).
- Unit tests on small synthetic images for the stages that had none:
  - `marching_cubes`: the surface of a ball sits where the ball is in physical space, and its faces point outward.
  - `distance_transform`: a slab reads its thickness.
  - `precomputed`: fills the path template from `cases.csv` columns, resamples a coarser map onto the image, and refuses to run without the case.

  The registerers and `mesh_splitting` need the atlas and real data, so the tests below cover them.
- `test/test_smoke.py` runs the whole pipeline as a project on the colab case with `precomputed` and the default other stages. It checks the 2D maps' mean thickness within 0.02 mm of the expected values: 2.286 (FC) and 2.258 (TC). Over 5 GPU and 2 CPU runs the spread was 0.002 mm. On a CPU it takes about 3.5 min here, so it needs no GPU. A project refuses to run on a CPU unless asked, so the test sets `device` by what the machine has.
- Restored asserts in `test/test_all.py`:
  - `test_SegmentationCPU` compares by max abs diff (< 0.005) and Dice (> 0.999), instead of the sum of the ITK comparison image (< 12) that failed. The reference maps were made on a CPU: a CPU reproduces them to 3e-6, a GPU differs by up to 0.0023. So the old failure was the GPU, not a regression.
  - The mesh point counts are restored with today's values: 55366/19329 on a GPU, 55406/19323 on a CPU. The old 64800–65000 and 20460–20480 predate the Sep 2026 dependency update.
  - The registration test checks folding (< 1%) and similarity to the atlas (> 0.75). `avsm/inv_transform_to_atlas.nii.gz` is from a different method, so it can't be a reference.
  - Removed the duplicate `TestICONRegistration`.
- CI:
  - `cpu-test` (GitHub-hosted) installs the CPU build of torch (no CUDA wheels to download) and caches pip and the network weights and test data (`~/.cache/oai-analysis`, keyed on `data.py`).
  - Both workflows install `itk-cuberille` and `itk-parabolicmorphology` explicitly, since 9d843c1 took them out of the dependencies for ITK 6.
  - `gpu-tests` uses the cu128 build of torch (cu118 has no wheels for recent GPUs) and gains a `regression` job. That job runs `tools/run_baseline.py` with `--back-mapping transform` on the cases in the repository variable `OAI_REGRESSION_CASES` (default: the colab case) and compares them with the reference run in `OAI_REFERENCE_DIR`. It runs only when that variable is set, and uploads `comparison.md`.
- Verification:
  - Full suite: 55 passed on the GPU, in 21 min (shared with the regression run). On a CPU only: 53 passed and the 2 `test_second_implementations` tests skipped, in 11.5 min. The CPU UNet test (6 min) and the smoke test (3.3 min) take most of that.
  - The regression job's command on `test_case_left,9016304_0M_left` against `OAI_golden_f198a2f/run1`: 2/2 cases and 30/30 files pass (`M:/Dev/Osteoarthritis/Refactoring/OAI_step6_regression`).
  - The workflows have not run on GitHub yet: that needs a push, and for `regression`, the variables and a reference run on the self-hosted runner.

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

**Done** (2026-10-30, branch `refactor`).
- `docs/`, plain Markdown:
  - `README.md` (index)
  - `install.md` (including the CUDA torch step, and the ITK 5 modules)
  - `quickstart.md` (one image, then a project)
  - `pipeline.md` (the steps in order, what each does and checks, the single-image command's files). The repository README's pipeline steps are rewritten to match and link to it, which closes the Dec 18 "update readme pipeline-steps" item.
  - `projects.md` (layout, statuses, resume, fingerprints, provenance, sanity checks, retention)
  - `config.md` (see below)
  - `plugins.md` (the 4.5 example, which ran in a project, chosen by name in its config)
  - `api.md`
  - `handoff.md`: setup on Linux GPU servers, the manifest, a pilot batch, reading results, known problems (cuberille crash, the thickness change from Step 2b, registration noise, GPU/CPU differences, the version without a tag), the regression check, and where to change things.
  - `reviewer_guide.md`: the statuses, what each kind of *Needs Review* note usually means, the thumbnails, and what to send engineers.
- **Config reference.** `tools/config_reference.py` writes `docs/config.md` from the built-in stage classes: each stage's docstring, and each parameter's default and description, taken from the comment on its field. Parameters are marked *check* or *speed*. Every stage parameter now has such a comment. `test_config_reference_is_current` fails when the file is out of date (`--check`).
- **Notebooks** are marked legacy in `notebooks/README.md`, which says what each shows and its state: `SegmentationDemo` is broken (it imports names `test_all.py` no longer has), `FullDemo` uses the deprecated `AnalysisObject`, and the other two do not use the package. Updating them to the new API is left for later.
- **`CHANGELOG.md`**: "What might look different" in plain language (the Step 2b femoral thickness shift, with its numbers; nothing else changes; Needs Review; thumbnails; speed), then the engineering changes. The numbers come from `tools/compare_outputs.py`, since there is no `oai-analysis compare` command (4.6 planned one; `export_flat.py` + `compare_outputs.py` do the job).
- Not done: the Docker recipe (Q9) and recording seeds and the container digest (4.8), which were left for the Docker work in Step 4.

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
