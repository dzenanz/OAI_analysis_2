# OAI Analysis 2

[<img src="https://github.com/uncbiag/OAI_analysis_2/actions/workflows/github-hosted-action.yml/badge.svg">](https://github.com/uncbiag/OAI_analysis_2/actions)
[![PyPI - Version](https://img.shields.io/pypi/v/oai-analysis.svg)](https://pypi.org/project/oai-analysis)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/oai-analysis.svg)](https://pypi.org/project/oai-analysis)

**Table of Contents**

- [Installation](#installation)
- [Introduction](#introduction)
- [Development](#development)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## Installation

```console
pip install oai-analysis
```

Python 3.11 or newer is required.

## Introduction

This repository contains open-source analysis approaches for the [Osteoarthritis Initiative (OAI)](https://nda.nih.gov/oai/) magnetic resonance image (MRI) data.
The analysis code is largely written in Python with the help of [ITK](http://itk.org) and [VTK](http://vtk.org) for data I/O and mesh processing
as well as [PyTorch](http://pytorch.org) for the deep learning approaches for segmentation and registration. The initial development of this work
was led by [UNC Chapel Hill](http://biag.cs.unc.edu) as well as [Kitware](http://kitware.com). This work is also an outgrowth of conversations within the
[Community for Open and Reproducible Musculoskeletal Imaging Research (JC|MSK)](https://jcmsk.github.io/). Going forward, contributions by the 
broader community are, of course, not only welcome but encouraged.

### Main functionality
1. **Deep Learning Segmentation**: Automatic cartilage segmentation (femoral and tibial cartilage) using a 3D UNet.
2. **Deep Learning Atlas registration**: Registration of the cartilage meshes with associated cartilage thickness to a knee atlas space via a deep registration network for 3D images.
3. **Cartilage thickness**: Extraction of femoral and tibial cartilage meshes and measuring of cartilage thickness based on a closest-point thickness estimation.
4. **2D thickness mapping**: Mapping of the thickness maps to a common 2D atlas space which provides full spatial correspondence. This is achieved via unrolling (based on a cylindrical coordinate system) for the femoral cartilage and a planar projection for the tibial cartilage.
5. **Statistical analysis**: [Longitudinal statistical analysis approaches are planned]

### Pipeline steps:
1. Read image
 a. Read
 b. Put into canonical orientation
 c. Intensity preprocess (rescale to 0-1)
2. Segment cartilage using deep learning  
3. Register patient image to atlas image.  
4. Transform patient mesh into atlas space using registration transform  
5. Split the mesh into inner and outer surfaces in atlas space
  a. Island filtering (keep largest + islands close to atlas mesh)
  b. Mesh smoothing
  c. Clustering into inner and outer surface
6. Transform the inner and outer meshes back to patient space  
7. Measuring the thickness by computing distances between inner and outer surfaces
8. Transfer thicknesses from patient mesh to atlas mesh by iterating over atlas vertices and finding the closest patient vertex
9. Project atlas mesh to 2D using precomputed mapping

![OAI analysis workflow](doc_imgs/OAI_workflow.png)

### Difference between OAI Analysis and OAI Analysis 2

The analysis approaches in this repository are based on our initial [OAI Analysis](https://github.com/uncbiag/OAI_analysis) work.
Much of the functionality of the original code-base has been ported to the *OAI Analysis 2* repository. The main differences are
1. **Refactoring**: A significant refactoring of the code so that it makes better use of [ITK](http://itk.org) conventions as well as [VTK](http://vtk.org) for all the mesh processing needs.
2. **uniGradICON registration**: We switched to the new [uniGradICON](https://github.com/uncbiag/uniGradICON) registration approach (see manuscripts below).
3. **Data processing**: We improved the data processing by better handling of data objects. Whereas the previous *OAI Analysis* pipeline largely depended on reading and writing various different files, the *OAI Aanalysis 2* refactoring uses ITK and VTK objects.
4. **Jupyter notebooks**: Better support of analysis in Jupyter notebooks.

We are currently working on the following features which should be available in the near future:
1. **Distributed processing**: Whereas *OAI Analysis* was set up for cluster computing via a simple SLURM script OAI Analysis 2 is moving toward using [Dask](https://dask.org/) to allow for parallel processing on clusters and the cloud.
2. **Workflow management**: Whereas *OAI Analysis* used custom code to avoid recomputing results, we are switching to [Dagster](https://dagster.io/) to manage data dependencies in *OAI Analysis 2*. Current progress is [here](https://github.com/PaulHax/knee-sarg).
3. **Distribution of analysis results**: As we are planning on not only distributing code, but also analysis results (such as segmentations, meshes, thickness maps) we are planning on supporting data access via the [Interplanetary File System (IPFS)](https://ipfs.io/).

## Development

Contributions are appreciated and welcome.

```
git clone https://github.com/uncbiag/OAI_analysis_2
cd OAI_analysis_2
pip install -e ".[dev]"
pytest
```

Python 3.11 or newer is required. For GPU processing, install the CUDA build of PyTorch first,
following https://pytorch.org/get-started/locally/.

Network weights and test data are downloaded on first use into a per-user cache directory
(on Windows `%LOCALAPPDATA%\oai-analysis\oai-analysis\Cache`, on Linux `~/.cache/oai-analysis`).
Set the `OAI_ANALYSIS_DATA` environment variable to use a different directory.

To process one knee image (a file, or a directory with a DICOM series):
```
oai-analysis IMAGE OUTPUT_DIR left
```

From Python, `OAPipeline` loads the models once and can process many cases. Any stage can be replaced
by passing an instance (see `oai_analysis/stages/`):
```python
from oai_analysis import OAPipeline, CaseInfo

pipeline = OAPipeline()
result = pipeline.run_case(CaseInfo("IMAGE", "left"), "OUTPUT_DIR")
fc_thickness_map = result.biomarkers["FC_thickness.npy"]
```

Stages and their parameters can also be chosen in a YAML config file. Only the settings that differ
from the defaults are needed; `oai-analysis config dump` prints the full config with every default.
```yaml
pipeline:
  registerer: {finetune_steps: 5}
  thickness_estimator: {name: distance_transform}
```
```
oai-analysis IMAGE OUTPUT_DIR left --config my_config.yaml
```
or `OAPipeline.from_config("my_config.yaml")` in Python.

Other packages can add stages that a config can then name: decorate the stage class with
`@oai_analysis.registry.register`, and declare the module in an entry point:
```toml
[project.entry-points."oai_analysis.plugins"]
my_stages = "my_package.stages"
```

For many cases, use a project: a folder holding the cases, the config, and every output with its provenance.
`run` skips what is already up to date, so it resumes an interrupted run, retries failed cases, and after a
config change redoes only the steps that change affects. A case whose step fails is marked Failed and the run
moves on. Cases run in batches of `batch_size` (default 8): a batch's cases are preprocessed, then segmented one
after another, then registered one after another, so each network is loaded and warm for the whole batch; then
each case finishes on its own. A batch holds its cases' images, probability maps and transforms in memory, about
0.5 GB per case.
```
oai-analysis init M:/OAI_projects/pilot --data-root O:/OAI --manifest study_uid_to_vol_path.csv --patients 9016304,9021791
oai-analysis run M:/OAI_projects/pilot
oai-analysis status M:/OAI_projects/pilot
oai-analysis run M:/OAI_projects/pilot --failed                 # only the cases that failed
oai-analysis run M:/OAI_projects/pilot --cases 9016304_00M_LEFT --from mesh
oai-analysis run M:/OAI_projects/pilot --batch-size 20
```
Settings are in `project.yaml` (the pipeline config, plus `batch_size`, `retention: checkpoints|all|final` and
the storage formats), and the cases are in `cases.csv`, which can be edited. Each run writes a full log and a plain-English
summary to `logs/`, and a one-row-per-case table to `index.parquet`.

To view the demo notebooks:
```
cd notebooks
jupyter notebook
```

upload test data to https://data.kitware.com/#collection/6205586c4acac99f42957ac3/folder/620559344acac99f42957d63

## Citation

We now use registration approach available in [uniGradICON](https://github.com/uncbiag/uniGradICON) for *OAI Analysis*
[paper](https://arxiv.org/abs/2403.05780)
```
@article{tian2024unigradicon,
  title={uniGradICON: A Foundation Model for Medical Image Registration},
  author={Tian, Lin and Greer, Hastings and Kwitt, Roland and Vialard, Francois-Xavier and Estepar, Raul San Jose and Bouix, Sylvain and Rushmore, Richard and Niethammer, Marc},
  journal={arXiv preprint arXiv:2403.05780},
  year={2024}
}
```

An overview of the analysis framework is available as a [QMSKI](https://qmski.org/) conference [abstract](https://www.researchgate.net/publication/361754685_Reproducible_Workflow_for_Visualization_and_Analysis_of_OsteoArthritis_Abnormality_Progression)
```
@InProceedings{Sahu_2022_QMSKI,
author    = {Sahu, Pranjal and Greer, Hastings and Xu, Zhenlin and Shen, Zhengyang and Bonaretti, Serena and McCormick, Matt and Niethammer, Marc},
title     = {Reproducible Workflow for Visualization and Analysis of OsteoArthritis Abnormality Progression},
booktitle = {Proceedings of the International Workshop on Quantitative Musculoskeletal Imaging (QMSKI)},
year      = {2022}
}
```

Results obtained by the *OAI Analysis* pipeline can be found in this manuscript
[[paper]](https://www.sciencedirect.com/science/article/pii/S1361841521003881?casa_token=Pgx9BQD4H_4AAAAA:P91yIWLkxXp_ZzLURL7LGSyrdqFJ-QARRTKSq5IoyQ8uGHUc221-hWNoL8ObpvQuls1JBZ4)
```
@article{huang2022dadp,
  title={DADP: Dynamic abnormality detection and progression for longitudinal knee magnetic resonance images from the Osteoarthritis Initiative},
  author={Huang, Chao and Xu, Zhenlin and Shen, Zhengyang and Luo, Tianyou and Li, Tengfei and Nissman, Daniel and Nelson, Amanda and Golightly, Yvonne and Niethammer, Marc and Zhu, Hongtu},
  journal={Medical Image Analysis},
  pages={102343},
  year={2022},
  publisher={Elsevier}
}
```

## Acknowledgements

This work was developed with support in part from the National Institute of Arthritis and Musculoskeletal and Skin Diseases (NIAMS) 
under award numbers [1R44AR074375](https://reporter.nih.gov/search/Naf5qSR3eUStFkMfGm6KpQ/project-details/9777582), [1R01AR072013](https://reporter.nih.gov/search/eE7eB34dVUGoY1nLF3kZNA/project-details/9368542), and [1R01AR082684](https://reporter.nih.gov/search/TdbftRplwU-HPeF3spFejw/project-details/10822142).

## License

This is distributed under the terms of the [Apache-2.0](https://spdx.org/licenses/Apache-2.0.html) license.
