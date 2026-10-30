# Quick start

## One image

```
oai-analysis IMAGE OUTPUT_DIR left
```

`IMAGE` is an image file (`.nii.gz`, `.nrrd`, ...) or a folder holding one DICOM series, and the last argument is the
knee: `left` or `right`. `OUTPUT_DIR` gets the 2D thickness maps (`FC_thickness.png`, `TC_thickness.png` and the
arrays as `.npy`) and, unless `--no_intermediates` is given, the intermediate images, transforms and meshes. The
[pipeline page](pipeline.md#files-of-the-single-image-command) lists the files.

To try it on the bundled test case:

```
python -c "from oai_analysis.data import test_data_dir; print(test_data_dir() / 'colab_case' / 'image_preprocessed.nii.gz')"
oai-analysis PATH_PRINTED_ABOVE out_test left
```

`--config FILE` chooses stages and parameters (see the [config reference](config.md)), e.g. a file with

```yaml
pipeline:
  registerer: {name: gradicon_knee}
```

This command is the one [knee-sarg](https://github.com/PaulHax/knee-sarg) runs, and `python
oai_analysis/pipeline_cli.py IMAGE OUT left` still works the same way.

## Many images: a project

A project is a folder that holds the list of cases, the settings, and every result with a record of how it was
made. Running it again skips what is done, so it resumes after a crash and retries failed cases.

**Create it** from a knee-sarg manifest (`patient_id, study_uid, vol_path, month, study_description`), here two
patients, both knees, every visit:

```
oai-analysis init M:/OAI_projects/pilot --manifest study_uid_to_vol_path.csv --data-root O:/OAI --patients 9016304,9021791
```

`--months 0,96` and `--laterality left` narrow it further. Without `--manifest`, the project starts with an empty
`cases.csv` that you fill in yourself (`case_id, patient_id, month, laterality, input`, with `input` relative to
`data_root`).

**Run it:**

```
oai-analysis run M:/OAI_projects/pilot
```

The console shows progress and a line per finished case. The same lines are kept in `logs/*.summary.txt`, and the
full log, with everything the stages print, in `logs/*.log`.

**See where it stands:**

```
oai-analysis status M:/OAI_projects/pilot
```

```
4 cases: 3 Succeeded, 1 Needs Review

case                 status        prep  seg  reg  mesh  thick  atlas  bio  note
9016304_00M_LEFT     Succeeded     ✓     ✓    ✓    ✓     ✓      ✓      ✓
9016304_96M_LEFT     Needs Review  ✓     !    ✓    ✓     ✓      ✓      ✓    37.6% of the femoral cartilage is outside its largest piece (review above 5.0%)
...
```

**The results** of each case are in `cases/PATIENT/VISIT_KNEE/`, e.g. the 2D thickness maps in
`cases/9016304/96M_LEFT/07_biomarkers/thickness_map_2d/`, and small pictures to look at in its `thumbnails/`
folder. `index.parquet` has one row per case, for pandas or DuckDB.

**Common next commands:**

```
oai-analysis run M:/OAI_projects/pilot --failed                        # retry the cases that failed
oai-analysis run M:/OAI_projects/pilot --cases 9016304_96M_LEFT --from mesh   # redo one case from a step on
oai-analysis status M:/OAI_projects/pilot --needs-review --csv flagged.csv    # the flagged cases, as a spreadsheet
```

To change a setting, edit `project.yaml` (e.g. `registerer: {finetune_steps: 5}`). `status` then shows the affected
cases as *Outdated*, and `run` redoes only the steps the change affects. The [projects page](projects.md) has the
details.
