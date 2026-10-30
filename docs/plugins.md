# Writing a plugin

Every step of the [pipeline](pipeline.md) is done by a *stage*, and any stage can be replaced by your own. A stage is
a dataclass that subclasses one of the seven interfaces in `oai_analysis.stages`:

| Interface | Config key | `run(...)` takes | and returns |
|---|---|---|---|
| `Preprocessor` | `preprocessor` | `image, case: CaseInfo` | the prepared image |
| `Segmenter` | `segmenter` | `image, case: CaseInfo \| None` | `{"FC": probability_image, "TC": ...}` |
| `Registerer` | `registerer` | `moving, fixed` (the image, the atlas image) | `RegistrationResult(phi_AB, phi_BA)` |
| `MeshGenerator` | `mesh_generator` | `probability` (one structure) | a `vtkPolyData` in the image's physical space |
| `ThicknessEstimator` | `thickness_estimator` | `mesh, probability, registration, atlas, structure` | `MeshResult(mesh with a thickness array)` |
| `AtlasMapper` | `atlas_mapper` | `thickness_mesh, registration, atlas, structure` | `MeshResult(atlas inner mesh with the thickness)` |
| `BiomarkerExtractor` | `biomarker_extractors` (a list) | `case: CaseArtifacts`, everything so far | `{file_name: value}` to save |

The most common plugin is a new biomarker extractor: a result computed from the thickness, saved with each case.

## A biomarker extractor

```python
import json
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from oai_analysis.registry import register
from oai_analysis.stages import BiomarkerExtractor


@register
@dataclass
class MeanThickness(BiomarkerExtractor):
    """Mean cartilage thickness over the atlas surface, per structure."""

    name: ClassVar[str] = "mean_thickness"
    version: ClassVar[str] = "1"
    requires: ClassVar[tuple[str, ...]] = ("atlas_map",)  # the steps run() reads
    min_thickness_mm: float = 0.1  # thinner than this counts as no cartilage

    def run(self, case):
        means = {}
        for structure in case.atlas_mapped:
            thickness = case.atlas_thickness(structure)
            means[structure] = float(np.mean(thickness[thickness >= self.min_thickness_mm]))
        return {"mean_thickness.json": json.dumps(means).encode()}
```

- The class's fields are its parameters. They appear in `oai-analysis config dump`, are recorded in provenance,
  and a change to one makes the outputs outdated.
- `name` is how a config chooses it. `version` should change whenever the code changes its results, so that
  existing outputs are recomputed.
- `run()` returns file names and values. Values can be bytes, numpy arrays (`.npy`), ITK images or VTK meshes. In a
  project they go to `07_biomarkers/mean_thickness/`, with a `provenance.json`.
- `requires` lists the steps whose outputs `run()` reads. With `retention: final`, adding the extractor then reruns
  only what it needs, not the whole case.

`CaseArtifacts` holds everything produced for the case: `case` (the `CaseInfo`), `atlas`, `image`,
`probability[s]`, `registration`, `mesh[s]`, `thickness[s]`, `atlas_mapped[s]` and `atlas_thickness(s)`, with `s`
being `"FC"` or `"TC"`. Treat it as read-only.

## Using it

There are three ways, from least to most formal.

**1. Pass an instance**, e.g. in a notebook. Nothing needs registering:

```python
from oai_analysis import OAPipeline, CaseInfo
from oai_analysis.stages import ThicknessMap2D

pipeline = OAPipeline(biomarker_extractors=[ThicknessMap2D(), MeanThickness()])
result = pipeline.run_case(CaseInfo("IMAGE.nii.gz", "left"))
print(result.biomarkers["mean_thickness.json"])
```

A project can take one too: `Project(path).run(pipeline=pipeline)`.

**2. Name it in a config.** With `@register`, the class can be chosen by name once its module is imported:

```yaml
pipeline:
  biomarker_extractors:
    - {name: thickness_map_2d}
    - {name: mean_thickness, min_thickness_mm: 0.2}
```

**3. Ship it in a package.** So that the command line finds it without importing anything by hand, declare the
module in an entry point of your package's `pyproject.toml`:

```toml
[project.entry-points."oai_analysis.plugins"]
my_biomarkers = "my_package.biomarkers"
```

After `pip install my-package`, `{name: mean_thickness}` works in any `project.yaml`, with no change to
oai-analysis. A plugin that fails to import gives a warning and does not stop the others.

## Other stages

The same pattern works for the other interfaces. What to know:

- **Heavy resources** (networks, weights) are loaded in `setup(self, device)`, which is called once before the first
  case, not in `__init__` or `run()`. Store them in an attribute whose name starts with `_`, so it is not taken for a
  parameter.
- **Weights.** `weights()` should return the files of trained weights with their hashes,
  `{"model.pth": "sha256:..."}`, so they are recorded in provenance and new weights make old outputs outdated.
- **Sanity checks.** `check(self, case)` returns a list of `QCFinding(severity, code, message, value)`. `severity` is
  `"fail"` (fails the case), `"review"` (marks it Needs Review) or `"info"` (only records the value). Write
  `message` for a person, e.g. `"the femoral cartilage volume is 2.1 ml (usual: 4.0-20.0 ml)"`. List the parameters
  that only `check()` reads in `check_params`, so changing them rechecks rather than recomputes.
- **Speed parameters** that do not change the results (e.g. how many patches go through a network at once) belong
  in `performance_params`, so changing them does not make outputs outdated.
- **Intermediate outputs** worth keeping go in `MeshResult(mesh, intermediates={"name": value})`. A project saves
  them with `retention: all`.
- A **segmenter** gets the case as well as the image, e.g. to find files by `case.case_id` or by the `cases.csv`
  columns in `case.details` (see `PrecomputedSegmenter`).

The built-in stages in `oai_analysis/stages/` are working examples of each interface, and
[`test/test_config.py`](../test/test_config.py) has a test that installs a throwaway plugin package and finds its
stage through the entry point.
