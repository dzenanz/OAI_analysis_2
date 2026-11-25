import pytest

from oai_analysis import longitudinal

CFG = longitudinal.settings({"knee_typical": ["atlas_map.no_thickness"],  # none by default
                             "departure": {"register.similarity": 0.08}})  # off by default


def review_messages(case_qc, other_visits_qc, cfg=CFG):
    """longitudinal.review with only the messages of the flags."""
    flags, typical = longitudinal.review(case_qc, other_visits_qc, cfg)
    return {step: [f["message"] for f in findings] for step, findings in flags.items()}, typical


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
    review, typical = review_messages(visit(0.30), [visit(0.05), visit(0.04), visit(0.32)])
    assert review == {"atlas_map": ["no thickness on 30%"]} and typical == []


def test_flag_that_is_not_anatomy_is_kept():  # a registration that is poor at every visit still needs a look
    review, _ = review_messages(visit(0.05, similarity=0.7), [visit(0.05, similarity=0.7)] * 3)
    assert review == {"register": ["similarity 0.7"]}


def test_departure_from_the_knee():
    review, _ = review_messages(visit(0.13), [visit(0.02), visit(0.03), visit(0.02)])
    assert review == {"atlas_map": ["no thickness on 13%; this knee's other visits have 2.0% (median), so this "
                                    "visit either has a processing problem or shows a real change"]}
    flag = longitudinal.review(visit(0.13), [visit(0.02), visit(0.03), visit(0.02)], CFG)[0]["atlas_map"][0]
    assert flag["value"] == pytest.approx(0.11) and flag["limit"] == 0.1  # the departure, and its limit
    assert longitudinal.review(visit(0.11), [visit(0.02), visit(0.03), visit(0.02)], CFG) == ({}, [])  # within 0.1
    assert longitudinal.review(visit(0.02), [visit(0.13), visit(0.13)], CFG) == ({}, [])  # better is fine
    review, _ = longitudinal.review(visit(0.05, similarity=0.80), [visit(0.05)] * 2, CFG)  # lower is worse
    assert list(review) == ["register"]


def test_volume_departs_only_downward():
    def volume(ml):
        return {"segment": [{"severity": "info", "code": "segment.volume", "message": f"{ml} ml", "value": ml}]}

    review, _ = longitudinal.review(volume(6.0), [volume(10.0), volume(10.5), volume(9.8)], CFG)
    assert list(review) == ["segment"]  # 40% less than the knee's median
    assert longitudinal.review(volume(14.0), [volume(10.0), volume(10.5), volume(9.8)], CFG) == ({}, [])  # 40% more


def test_no_flag_is_usual_for_the_knee_by_default():
    review, typical = review_messages(visit(0.30), [visit(0.32), visit(0.28), visit(0.05)],
                                      longitudinal.settings(None))
    assert review == {"atlas_map": ["no thickness on 30%"]} and typical == []


def test_needs_enough_other_visits():
    assert review_messages(visit(0.30), [visit(0.32)]) == ({"atlas_map": ["no thickness on 30%"]}, [])
    assert longitudinal.review(visit(0.13), [visit(0.02)], CFG) == ({}, [])


def test_settings():
    cfg = longitudinal.settings({"departure": {"segment.volume": None}, "min_other_visits": 3})
    assert cfg["departure"]["segment.volume"] is None and cfg["departure"]["segment.pieces"] == 0.05
    assert cfg["min_other_visits"] == 3 and cfg["knee_typical"] == longitudinal.DEFAULTS["knee_typical"]
    with pytest.raises(ValueError, match="Unknown longitudinal settings"):
        longitudinal.settings({"min_visits": 3})
    with pytest.raises(ValueError, match="Unknown values"):
        longitudinal.settings({"departure": {"segment.volumes": 0.2}})
