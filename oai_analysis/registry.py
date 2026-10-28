"""
Stage implementations by name, so a config file can choose them.

Built-in stages register themselves with @register. Other packages add theirs through an entry point in the
"oai_analysis.plugins" group, naming a module that registers its stages when imported:

    [project.entry-points."oai_analysis.plugins"]
    gwas = "oai_gwas.stages"
"""
import importlib
import warnings
from importlib.metadata import entry_points

ENTRY_POINT_GROUP = "oai_analysis.plugins"

_stages: dict[str, dict[str, type]] = {}  # kind -> name -> class
_loaded_plugins: set[str] = set()


def register(cls):
    """Class decorator: make a stage implementation available by its kind and name."""
    kind, name = cls.kind, cls.name
    existing = _stages.setdefault(kind, {}).get(name)
    if existing is not None and existing.__qualname__ != cls.__qualname__:  # a module imported twice is fine
        raise ValueError(f"{kind} {name!r} is already registered by {existing.__module__}.{existing.__qualname__}")
    _stages[kind][name] = cls
    return cls


def load_plugins():
    """Import the modules named by the oai_analysis.plugins entry points that are not imported yet."""
    importlib.import_module("oai_analysis.stages")  # the built-in stages
    for entry_point in entry_points(group=ENTRY_POINT_GROUP):
        if entry_point.name in _loaded_plugins:
            continue
        _loaded_plugins.add(entry_point.name)
        try:
            entry_point.load()
        except Exception as e:  # one broken plugin should not stop the others
            warnings.warn(f"Could not load plugin {entry_point.name!r} ({entry_point.value}): {e!r}", stacklevel=2)


def get(kind: str, name: str) -> type:
    """The stage class registered as this kind and name."""
    if name not in _stages.get(kind, {}):
        load_plugins()
    try:
        return _stages[kind][name]
    except KeyError:
        known = ", ".join(sorted(_stages.get(kind, {}))) or "none"
        raise KeyError(f"No {kind} named {name!r}. Known: {known}") from None


def available(kind: str) -> list[str]:
    """Names of every registered stage of this kind, including plugins."""
    load_plugins()
    return sorted(_stages.get(kind, {}))
