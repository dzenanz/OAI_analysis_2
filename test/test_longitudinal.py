import pytest

from oai_analysis import longitudinal

CFG = longitudinal.settings(None)


def visit(no_thickness_tc, similarity=0.9, threshold=0.15):
    """A finished visit's findings: the atlas map of FC (always fine) and TC, and the registration."""
    def no_thickness(value):
        return {"severity": "review" if value > threshold else "info", "code": "atlas_map.no_thickness",
                "message": f"no thickness on {value:.0%}", "value": value}
    return {"register": [{"severity": "review" if similarity < 0.75 else "info", "code": "register.similarity",
                          "message": f"similarity {similarity}", "value": similarity}],
            "atlas_map": [no_thickness(0.02), no_thickness(no_thickness_tc)]}


def test_flag_usual_for_the_knee_is_dropped():
    review, typical = longitudinal.review(visit(0.30), [visit(0.32), visit(0.28), visit(0.05)], CFG)
    assert review == {}
    assert typical == ["no thickness on 30%; most of this knee's other visits are the same, so it is likely the knee "
                       "itself (e.g. cartilage loss), not a processing problem"]


def test_flag_unusual_for_the_knee_is_kept():
    review, typical = longitudinal.review(visit(0.30), [visit(0.05), visit(0.04), visit(0.32)], CFG)
    assert review == {"atlas_map": ["no thickness on 30%"]} and typical == []


def test_flag_that_is_not_anatomy_is_kept():  # a registration that is poor at every visit still needs a look
    review, _ = longitudinal.review(visit(0.05, similarity=0.7), [visit(0.05, similarity=0.7)] * 3, CFG)
    assert review == {"register": ["similarity 0.7"]}


def test_departure_from_the_knee():
    review, _ = longitudinal.review(visit(0.13), [visit(0.02), visit(0.03), visit(0.02)], CFG)
    assert review == {"atlas_map": ["no thickness on 13%; this knee's other visits have 2.0% (median), so this "
                                    "visit either has a processing problem or shows a real change"]}
    assert longitudinal.review(visit(0.11), [visit(0.02), visit(0.03), visit(0.02)], CFG) == ({}, [])  # within 0.1
    assert longitudinal.review(visit(0.02), [visit(0.13), visit(0.13)], CFG) == ({}, [])  # better is fine
    review, _ = longitudinal.review(visit(0.05, similarity=0.80), [visit(0.05)] * 2, CFG)  # lower is worse
    assert list(review) == ["register"]


def test_needs_enough_other_visits():
    assert longitudinal.review(visit(0.30), [visit(0.32)], CFG) == ({"atlas_map": ["no thickness on 30%"]}, [])
    assert longitudinal.review(visit(0.13), [visit(0.02)], CFG) == ({}, [])


def test_settings():
    cfg = longitudinal.settings({"departure": {"segment.volume": None}, "min_other_visits": 3})
    assert cfg["departure"]["segment.volume"] is None and cfg["departure"]["segment.pieces"] == 0.05
    assert cfg["min_other_visits"] == 3 and cfg["knee_typical"] == longitudinal.DEFAULTS["knee_typical"]
    with pytest.raises(ValueError, match="Unknown longitudinal settings"):
        longitudinal.settings({"min_visits": 3})
    with pytest.raises(ValueError, match="Unknown values"):
        longitudinal.settings({"departure": {"segment.volumes": 0.2}})
