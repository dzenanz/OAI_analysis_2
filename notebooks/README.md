# Notebooks

These notebooks are **legacy**: they predate the Phase 1 refactor and use the old internals, not the current
[Python API](../docs/api.md). They are kept as illustrations of the algorithms. For new work, start from the
[quick start](../docs/quickstart.md) and the [Python API](../docs/api.md):

```python
from oai_analysis import OAPipeline, CaseInfo
result = OAPipeline().run_case(CaseInfo("IMAGE.nii.gz", "left"))
```

| Notebook | What it shows | State |
|---|---|---|
| `FullDemo.ipynb` | The whole pipeline step by step, on Colab | Uses the deprecated `AnalysisObject` and `icon_registration`'s pretrained models directly. |
| `SegmentationDemo.ipynb` | The segmentation network on the test case | Broken: imports `TEST_DATA_DIR` and `download_test_data` from `test/test_all.py`, which no longer has them; use `oai_analysis.data.test_data_dir()` instead. |
| `ComputeThicknessDemo.ipynb` | Splitting a surface into inner and outer halves and measuring thickness, on Colab | Standalone: does not use `oai_analysis`. |
| `ProbabilityMapThicknessORMIR.ipynb` | Thickness from probability maps by parabolic morphology (the `distance_transform` stage's method), from the ORMIR community | Standalone: does not use `oai_analysis`. |

`parent.py` puts the repository on `sys.path` for `SegmentationDemo.ipynb`.
