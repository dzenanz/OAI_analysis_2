# SPDX-FileCopyrightText: 2023-present OAI Analysis Development Team
#
# SPDX-License-Identifier: Apache-2.0
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("oai-analysis")  # from git tags, via hatch-vcs
except PackageNotFoundError:  # not installed
    __version__ = "unknown"


__all__ = ["OAPipeline", "CaseInfo", "__version__"]


def __getattr__(name):
    # imported on first use, so that e.g. `import oai_analysis.data` does not load torch
    if name == "OAPipeline":
        from oai_analysis.pipeline import OAPipeline
        return OAPipeline
    if name == "CaseInfo":
        from oai_analysis.stages.base import CaseInfo
        return CaseInfo
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
