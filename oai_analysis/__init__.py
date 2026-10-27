# SPDX-FileCopyrightText: 2023-present OAI Analysis Development Team
#
# SPDX-License-Identifier: Apache-2.0
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("oai-analysis")  # from git tags, via hatch-vcs
except PackageNotFoundError:  # not installed
    __version__ = "unknown"

