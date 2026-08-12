import json

import numpy as np
import pandas as pd

from extreme_price_movements.structural_archetypes import (
    build_recurrent_archetypes,
    match_catalog_to_archetypes,
    materialize_row_archetype_exposures,
)


def _catalog() -> pd.DataFrame:
    rows = []
    path = json.dumps([{
        "feature": "f0",
        "branch": "left",
        "threshold_band_index": 0,
        "threshold_band_count": 2,
    }])
    for fold in ("2024-01", "2024-03", "2024-06"):
        rows.append({
            "fold_id": fold,
            "tree_index": 0,
            "leaf_token": f"leaf_{fold}",
            "rule_signature": "sig_recurrent",
            "rule_structural_path_json": path,
            "ensemble_tree_contribution": 0.02,
            "train_leaf_frequency": 0.10,
        })
    rows.append({
        "fold_id": "2024-06",
        "tree_index": 1,
        "leaf_token": "leaf_unmatched",
        "rule_signature": "sig_other",
        "rule_structural_path_json": json.dumps([{
            "feature": "different",
            "branch": "right",
            "threshold_band_index": 1,
            "threshold_band_count": 2,
        }]),
        "ensemble_tree_contribution": -0.02,
        "train_leaf_frequency": 0.10,
    })
    return pd.DataFrame(rows)


def test_recurrence_first_and_explicit_unmatched_mass():
    catalog = _catalog()
    archetypes, audit = build_recurrent_archetypes(catalog, min_folds=3, min_train_frequency=0.01)
    assert len(archetypes) == 1
    assert bool(audit.loc[audit.rule_signature.eq("sig_recurrent"), "selected"].iloc[0])

    matches, _ = match_catalog_to_archetypes(catalog, archetypes, top_n=2)
    assert matches.top1_archetype.notna().all()
    assert matches.unmatched_probability.between(0.0, 1.0).all()
    assert (matches[["top1_probability", "top2_probability"]].sum(axis=1) + matches.unmatched_probability <= 1.0 + 1e-8).all()


def test_row_exposures_are_normalized_and_retain_unmatched_mass():
    catalog = _catalog()
    archetypes, _ = build_recurrent_archetypes(catalog, min_folds=3, min_train_frequency=0.01)
    matches, _ = match_catalog_to_archetypes(catalog, archetypes, top_n=2)
    leaves = pd.DataFrame({
        "candidate_id": ["c0"],
        "__ts__": [pd.Timestamp("2024-06-01", tz="UTC")],
        "leaf_assignment__0": ["leaf_2024-06"],
        "leaf_assignment__1": ["leaf_unmatched"],
    })
    features, _ = materialize_row_archetype_exposures(leaves, catalog[catalog.fold_id.eq("2024-06")], matches[matches.fold_id.eq("2024-06")], archetypes)
    exposure = features.filter(like="__abs_contribution").sum(axis=1).iloc[0]
    assert 0.0 <= exposure <= 1.0
    assert np.isclose(features.archetype_matched_mass.iloc[0] + features.archetype_unmatched_mass.iloc[0], 1.0)
