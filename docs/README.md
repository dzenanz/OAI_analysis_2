# OAI Analysis 2 documentation

OAI Analysis 2 measures femoral and tibial cartilage thickness in knee MRIs from the
[Osteoarthritis Initiative](https://nda.nih.gov/oai/) and maps it onto a common atlas, so knees can be compared.
It is a Python package, `oai-analysis`, with a command line for single images and for projects of many cases.

| Page | For |
|---|---|
| [Install](install.md) | Setting up Python, PyTorch with CUDA, and the package |
| [Quick start](quickstart.md) | One image, then a project of many cases |
| [The pipeline](pipeline.md) | What each step does, in order, and what it writes |
| [Projects](projects.md) | Project folders, statuses, resuming, provenance, sanity checks, storage |
| [Config reference](config.md) | Every setting and stage parameter (generated from the code) |
| [Writing a plugin](plugins.md) | Adding your own stage: a biomarker, a segmenter, ... |
| [Python API](api.md) | The classes to use from Python and notebooks |
| [Handoff guide](handoff.md) | Running the pipeline on new hardware: setup, a pilot batch, known problems |
| [Reviewer guide](reviewer_guide.md) | For non-engineers: reading `status`, what *Needs Review* means, reporting a case |

What changed between versions, including what might look different in the results, is in
[CHANGELOG.md](../CHANGELOG.md).
