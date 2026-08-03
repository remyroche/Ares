from __future__ import annotations

import pandas as pd

from extreme_price_movements.base_candidate_population import BaseCandidatePopulationContract, select_base_candidate_population
from scripts.audit_febapr_canonical_top40_residual_readiness import _ranking_audit


def test_audit_requires_exact_timestamp_side_top40_ceiling() -> None:
    rows = []
    for side in ("long", "short"):
        for symbol, score in zip(("C", "A", "B", "D", "E"), (0.5, 0.8, 0.8, 0.2, 0.1), strict=True):
            rows.append({"__ts__": "2025-02-01T00:00:00Z", "__symbol__": symbol, "side_name": side, "score": score})
    source = pd.DataFrame(rows)
    selected = select_base_candidate_population(source, BaseCandidatePopulationContract(score_col="score", top_fraction=0.40))
    audit = _ranking_audit(source, selected)
    assert audit["all_groups_exact_ceiling_fraction"]
    assert audit["all_selected_ranks_within_ceiling"]
    assert selected.groupby("side_name").size().to_dict() == {"long": 2, "short": 2}
