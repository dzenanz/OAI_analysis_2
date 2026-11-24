"""
Comparing a case with the other visits of the same knee (same patient_id and laterality in cases.csv).

The stages' checks look at one case against fixed thresholds, which cannot tell a processing problem from the knee
itself: a knee that has lost cartilage has little of it at every visit. A project therefore also compares each
finished case with the knee's other finished visits:
- A value that departs from the median of the knee's other visits by more than its departure limit, in the worse
  direction, marks the case Needs Review, even within the fixed thresholds: either the processing went wrong or the
  knee changed.
- A review flag that most of the knee's other visits also have can be taken as usual for this knee (e.g. cartilage
  loss): for the values in knee_typical it no longer marks the case Needs Review, and is kept as a note instead.
  None are by default: in the full OAI run, such flags were mostly a processing failure repeated at every visit.
Both need at least min_other_visits other finished visits. The settings are `longitudinal` in project.yaml.
"""
import statistics

from oai_analysis import qc

# for each compared value, the direction that is worse ("higher", "lower", or "lower_relative": lower, as a share of
# the knee's median), and how to show it. A volume above the knee's median was not a problem in the full OAI run's
# visual review; nearly all problems were below it
MEASURES = {
    "segment.volume": ("lower_relative", lambda v: f"{v:.1f} ml"),
    "segment.pieces": ("higher", qc.percent),
    "segment.uncertainty": ("higher", lambda v: f"{v:.3f}"),
    "register.similarity": ("lower", lambda v: f"{v:.3f}"),
    "atlas_map.no_thickness": ("higher", qc.percent),
}
DEFAULTS = {
    "min_other_visits": 2,
    # how far a value may be from the median of the knee's other visits; set by a run of 148 knees (11 patients,
    # all visits), where about 95% of the visits stay within these. null switches one off
    "departure": {"segment.volume": 0.25, "segment.pieces": 0.05, "segment.uncertainty": 0.08,
                  "register.similarity": 0.08, "atlas_map.no_thickness": 0.1},
    # values whose review flag is dropped when usual for the knee, because a knee's anatomy or disease explains them.
    # None by default: in the full OAI run (47857 cases), 40 of the 48 graded failures that ended up Succeeded had a
    # flag dropped this way, nearly all a large area without thickness repeated at every visit
    "knee_typical": [],
}


def settings(value) -> dict:
    """The longitudinal settings of a project.yaml, with the defaults for what it leaves out."""
    value = value or {}
    unknown = set(value) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown longitudinal settings {sorted(unknown)}. Known: {list(DEFAULTS)}")
    result = {**DEFAULTS, **value, "departure": {**DEFAULTS["departure"], **(value.get("departure") or {})}}
    unknown = (set(result["departure"]) | set(result["knee_typical"])) - set(MEASURES)
    if unknown:
        raise ValueError(f"Unknown values {sorted(unknown)} in the longitudinal settings. Known: {list(MEASURES)}")
    return result


def _by_measure(case_qc) -> dict[tuple[str, str, int], dict]:
    """A case's findings keyed by (step, code, n), n counting the findings with that code: FC, then TC."""
    keyed = {}
    for step, findings in case_qc.items():
        counts = {}
        for finding in findings:
            n = counts[finding["code"]] = counts.get(finding["code"], -1) + 1
            keyed[(step, finding["code"], n)] = finding
    return keyed


def review(case_qc, other_visits_qc, cfg) -> tuple[dict[str, list[str]], list[str]]:
    """
    The review messages of a finished case by step, given its findings and those of the knee's other finished
    visits ({step: [finding dicts]} each), and the notes of the flags dropped as usual for the knee.
    """
    others = [_by_measure(o) for o in other_visits_qc]
    messages, typical = {}, []
    for (step, code, n), finding in _by_measure(case_qc).items():
        same = [o[(step, code, n)] for o in others if (step, code, n) in o]
        enough = len(same) >= cfg["min_other_visits"]
        if finding["severity"] == "review":
            if (code in cfg["knee_typical"] and enough
                    and sum(f["severity"] == "review" for f in same) >= len(same) / 2):
                typical.append(f"{finding['message']}; most of this knee's other visits are the same, so it is "
                               f"likely the knee itself (e.g. cartilage loss), not a processing problem")
            else:
                messages.setdefault(step, []).append(finding["message"])
            continue
        limit = cfg["departure"].get(code)
        values = [f["value"] for f in same if f.get("value") is not None]
        if limit is None or not enough or finding.get("value") is None or len(values) < cfg["min_other_visits"]:
            continue
        kind, show = MEASURES[code]
        median = statistics.median(values)
        change = finding["value"] - median
        worse = {"higher": change, "lower": -change, "lower_relative": -change / median if median else 0}[kind]
        if worse > limit:
            messages.setdefault(step, []).append(
                f"{finding['message']}; this knee's other visits have {show(median)} (median), so this visit "
                f"either has a processing problem or shows a real change")
    return messages, typical
