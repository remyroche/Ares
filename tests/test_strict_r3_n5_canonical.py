from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.n5_forest_support_sizing import (
    CURRENT_CANONICAL_LDF_PARAMS,
    CURRENT_CANONICAL_SCHEMA,
    MODEL_DISPLAY_NAME,
)
from extreme_price_movements.strict_r3_n5_canonical import (
    load_canonical_n5_bundle,
    load_n5_contract,
    persist_canonical_n5_bundle,
    score_canonical_n5_bundle,
    train_canonical_n5_bundle,
)


def _frame(rows: int = 1_000) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    score = rng.uniform(size=rows)
    support = rng.normal(size=rows)
    ood = rng.normal(size=rows)
    raw = 110.0 * score + 20.0 * support
    net = raw + 40.0 * support - 60.0 * ood + rng.normal(0.0, 80.0, rows)
    return pd.DataFrame(
        {
            "candidate_id": [f"c-{index}" for index in range(rows)],
            "__decision_ts__": pd.date_range("2025-01-01", periods=rows, freq="h", tz="UTC"),
            "final_score": score,
            "policy_label_available_ts": pd.date_range("2025-01-01", periods=rows, freq="h", tz="UTC") + pd.Timedelta(hours=12),
            "policy_path_valid": True,
            "raw_expected_bps": raw,
            "policy_net_bps": net,
            "mapped_ev_available": True,
            "support_feature": support,
            "ood_feature": ood,
            "geometry_bundle_sha256": "stable-summary-only",
        }
    )


def test_checked_in_n5_contract_is_exact_two_forest_bundle() -> None:
    contract = load_n5_contract()
    assert contract["schema"] == CURRENT_CANONICAL_SCHEMA
    assert contract["model_display_name"] == MODEL_DISPLAY_NAME
    assert contract["canonical_arm"] == "compact12_two_forest_meanrisk"
    assert contract["model"]["params"]["n_estimators"] == 96
    assert contract["model"]["params"]["max_depth"] == 9
    assert contract["model"]["params"]["min_samples_leaf"] == 100
    assert contract["model"]["params"] == CURRENT_CANONICAL_LDF_PARAMS.__dict__
    assert len(contract["features"]) == 12
    assert not any(field.startswith("k09__cluster_") for field in contract["features"])
    assert contract["ranking_changes"] is False
    assert contract["admission_changes"] is False


def test_canonical_n5_scores_target_free_and_bounds_size() -> None:
    frame = _frame(3_200)
    bundle = train_canonical_n5_bundle(
        frame,
        cutoff="2025-05-10",
        fields=["support_feature", "ood_feature"],
    )
    target_free = frame.iloc[-200:].drop(columns=["policy_net_bps", "policy_label_available_ts"])
    scored = score_canonical_n5_bundle(bundle, target_free)
    assert len(scored) == len(target_free)
    assert scored["portfolio_size_multiplier"].between(0.25, 1.75).all()
    assert scored["n5_schema"].eq(CURRENT_CANONICAL_SCHEMA).all()
    with pytest.raises(ValueError, match="outcomes/labels"):
        score_canonical_n5_bundle(
            bundle,
            target_free.assign(policy_net_bps=0.0),
        )


def test_original_n5_ldf_bundle_round_trip_is_canonical(tmp_path) -> None:
    frame = _frame(3_200)
    fields = load_n5_contract()["features"]
    rng = np.random.default_rng(9)
    for field in fields:
        if field not in frame:
            frame[field] = rng.normal(size=len(frame))
    bundle = train_canonical_n5_bundle(
        frame,
        cutoff="2025-05-10",
    )
    directory = tmp_path / "ldf"
    manifest = persist_canonical_n5_bundle(bundle, directory)
    assert manifest["canonical_arm"] == "compact12_two_forest_meanrisk"
    loaded = load_canonical_n5_bundle(directory)
    assert loaded.params.n_estimators == 96
    assert loaded.params.support_prior == 300.0
