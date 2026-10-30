from oai_analysis import qc
from oai_analysis.stages.base import CaseArtifacts, QCFinding


def segmentation_findings(case: CaseArtifacts, fc_volume_ml, tc_volume_ml, max_outside_main_pieces,
                          max_uncertainty) -> list[QCFinding]:
    """The checks of any segmenter: not empty, a usual volume, in the expected pieces, and a confident network."""
    findings = []
    for structure, probability in case.probability.items():
        name = qc.STRUCTURE_NAMES[structure]
        measures = qc.segmentation_measures(probability, structure)
        volume = measures["volume_ml"]
        if volume == 0:
            findings.append(QCFinding("fail", "segment.empty", f"no {name} was found in the image", 0.0))
            continue
        low, high = fc_volume_ml if structure == "FC" else tc_volume_ml
        findings.append(qc.finding(not low <= volume <= high, "segment.volume",
                                   f"the {name} volume is {volume:.1f} ml (usual: {low}-{high} ml)", volume))
        outside = measures["outside_main_pieces"]
        pieces = "largest piece" if qc.EXPECTED_PIECES[structure] == 1 else "two largest pieces"
        findings.append(qc.finding(outside > max_outside_main_pieces, "segment.pieces",
                                   f"{qc.percent(outside)} of the {name} is outside its {pieces} "
                                   f"(review above {qc.percent(max_outside_main_pieces)})", outside))
        uncertainty = measures["uncertainty"]
        findings.append(qc.finding(uncertainty > max_uncertainty, "segment.uncertainty",
                                   f"the uncertainty about the {name} is {uncertainty:.3f} "
                                   f"(review above {max_uncertainty})", uncertainty))
    return findings
