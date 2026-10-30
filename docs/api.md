# Python API

An overview of the classes to use from Python and notebooks. Their docstrings have the details.

## One case: `OAPipeline`

```python
from oai_analysis import OAPipeline, CaseInfo

pipeline = OAPipeline()                       # the default stages
result = pipeline.run_case(CaseInfo("IMAGE.nii.gz", "left"))
fc_map = result.biomarkers["FC_thickness.npy"]
```

- `OAPipeline(preprocessor=..., segmenter=..., registerer=..., mesh_generator=..., thickness_estimator=...,
  atlas_mapper=..., biomarker_extractors=[...])`: any stage left out is the default.
- `OAPipeline.from_config("config.yaml")` (or a dict) builds it from a [config](config.md); `pipeline.config()` is
  the config of a pipeline, as a dict.
- `run_case(case, output_dir=None, keep_intermediate_outputs=True)` runs every step and returns a `CaseArtifacts`.
  With `output_dir`, it also writes the files of the [single-image command](pipeline.md#files-of-the-single-image-command).
- The networks are loaded on the first case and kept, so one pipeline object should process many cases.
- `OAPipeline(device="cpu")` chooses the device. By default it is `"cuda"` when a GPU is available, otherwise
  `"cpu"` with a warning.

`analysis_pipeline(input_path, output_path, laterality, keep_intermediate_outputs)` in `oai_analysis.pipeline` is the
original function, kept for older code: it runs the default pipeline and writes the same files.

## The case and its results

- `CaseInfo(input, laterality, case_id=None, details={})`: `input` is an image file or a DICOM folder,
  `laterality` is `"left"` or `"right"`. `case_id` defaults to the input's name; `details` holds anything else
  known about the case (a project puts its `cases.csv` row there).
- `CaseArtifacts`, returned by `run_case`, holds everything produced, keyed by structure (`"FC"`, `"TC"`) where it
  applies:

| Attribute | Content |
|---|---|
| `image` | the preprocessed image (ITK) |
| `probability[s]` | probability maps (ITK) |
| `registration` | `RegistrationResult(phi_AB, phi_BA)`: ITK transforms, image → atlas resampling and mesh moving |
| `mesh[s]` | surfaces in the image's space (VTK `vtkPolyData`) |
| `thickness[s].mesh` | the surface with a `vertex_thickness` array |
| `atlas_mapped[s].mesh` | the atlas surface with the case's thickness; `atlas_thickness(s)` gives it as a numpy array |
| `biomarkers` | the biomarker outputs, by file name, e.g. `"FC_thickness.npy"` |
| `seconds` | wall time of each step |

## Many cases: `Project`

```python
from oai_analysis.project import Project

project = Project.from_manifest("M:/OAI_projects/pilot", "study_uid_to_vol_path.csv", "O:/OAI",
                                patients=["9016304"], months=[0, 96])
project = Project("M:/OAI_projects/pilot")    # an existing project
statuses = project.run()                      # like `oai-analysis run`
for s in project.status():                    # like `oai-analysis status`
    print(s.case_id, s.status, s.failed_step, s.note, s.review)
flagged = project.select(project.status(), needs_review=True)
```

- `Project.create(path, cases, pipeline_config=None, data_root=None, retention="checkpoints")` starts a project
  from a list of `cases.csv` rows (dicts with `case_id, patient_id, month, laterality, input`).
- `run(case_ids=None, from_step=None, pipeline=None, batch_size=None)`: `pipeline` replaces the one built from
  `project.yaml`, e.g. to add a stage that is not registered.
- `status()` returns a `CaseStatus` per case: `status`, `steps` (step name → `done`, `failed`, `outdated` or
  `missing`), `note`, `failed_step`, `review` (step → messages), `duration_s`, `finished`.
- `case_folder(case_id)` is where a case's files are; [projects](projects.md#a-cases-folder) describes them.
- `index.parquet` has the same information for other programs: `pandas.read_parquet(project.path / "index.parquet")`.

## Stages

The stage interfaces and the built-in implementations are in `oai_analysis.stages`; [writing a plugin](plugins.md)
describes the interfaces, and the [config reference](config.md) lists the implementations and their parameters.
`oai_analysis.registry` finds stages by name: `registry.available("registerer")` lists the registerers, including
those of installed plugins.

## Data

- `oai_analysis.atlas.Atlas.default()`: the atlas image, its inner surfaces (`inner_mesh[s]`) and their 2D layouts
  (`embedding_2d[s]`), loaded once.
- `oai_analysis.data`: the downloaded files. `test_data_dir()` is the bundled test case's folder.
- `oai_analysis.io.read_image(path)` and `io.write(obj, path)` read and write images, meshes, transforms and arrays.
