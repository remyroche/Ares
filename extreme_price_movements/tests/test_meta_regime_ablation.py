from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.meta_regime_ablation import (
    OUTCOME_COLUMNS,
    apply_regime_builder_fold,
    drop_oos_outcome_columns,
    make_regime_builder,
)


ARMS = [
    "current_archetype_meta_regimes",
    "meta_feature_only_regimes",
    "base_error_signature_regimes",
    "joint_feature_error_regimes",
    "side_archetype_local_regimes",
    "temporal_reliability_regimes",
    "supervised_embedding_regimes",
]


def _frame(n: int, *, start: str = "2026-01-01") -> pd.DataFrame:
    rng = np.random.default_rng(42)
    ts = pd.date_range(start, periods=n, freq="4h", tz="UTC")
    score = rng.normal(size=n)
    clean = (score + rng.normal(scale=0.7, size=n) > 0.4).astype(float)
    bad = (score + rng.normal(scale=1.0, size=n) < -0.2).astype(float)
    timeout = (rng.random(n) < 0.08).astype(float)
    dirty = ((score > 0.2) & ((bad > 0) | (timeout > 0))).astype(float)
    return pd.DataFrame(
        {
            "__ts__": ts,
            "__symbol__": np.where(np.arange(n) % 2 == 0, "BTC", "ETH"),
            "month": ts.strftime("%Y-%m"),
            "side_name": np.where(np.arange(n) % 3 == 0, "short", "long"),
            "score": score,
            "base_score_rank_pct_train_prior": pd.Series(score).rank(pct=True).to_numpy(),
            "bb_pos_12": rng.normal(size=n),
            "eth_ret_24h": rng.normal(size=n),
            "cs_rank_oi_value_z_30d": rng.normal(size=n),
            "support_min_frequency": rng.random(n),
            "support_rare_bucket_share": rng.random(n),
            "base_arch_hit_surprise_hl3d": rng.normal(size=n),
            "base_arch_hit_surprise_z_hl7d": rng.normal(size=n),
            "regime_lgbm_leaf_bad_mae_k4": np.where(np.arange(n) % 4 == 0, "leaf_a", "leaf_b"),
            "aegmm_cluster": np.arange(n) % 4,
            "side_aegmm_cluster": np.arange(n) % 3,
            "source_semantic_family": np.where(np.arange(n) % 2 == 0, "mean_reversion", "compression"),
            "archetype_label_family": np.where(np.arange(n) % 2 == 0, "family_a", "family_b"),
            "clean_exec_label": clean,
            "clean_exec": clean,
            "dirty_positive": dirty,
            "first_touch_bad_mae_1r": bad,
            "full_path_bad_mae_1r": bad,
            "timeout": timeout,
            "exec_margin": 0.004 * score - 0.003 * bad - 0.001 * timeout,
            "ev_after_1pct": 0.004 * score - 0.003 * bad - 0.001 * timeout,
        }
    )


@pytest.mark.parametrize("arm", ARMS)
def test_regime_builder_oos_assignment_is_outcome_stripped(arm: str) -> None:
    train = _frame(180, start="2026-01-01")
    valid = _frame(50, start="2026-04-01")
    train_aug, valid_aug, features, meta = apply_regime_builder_fold(
        arm,
        train=train,
        valid=valid,
        seed=123,
    )
    assert features
    assert meta["name"] == arm
    assert all(col in train_aug.columns for col in features)
    assert all(col in valid_aug.columns for col in features)
    assert not set(features).intersection(OUTCOME_COLUMNS)
    assert valid_aug[features].notna().all().all()


def test_transform_oos_rejects_realized_outcome_columns() -> None:
    train = _frame(180)
    valid = _frame(30, start="2026-04-01")
    builder = make_regime_builder("meta_feature_only_regimes", seed=123)
    assert builder is not None
    builder.fit(train)
    with pytest.raises(ValueError):
        builder.transform_oos(valid)
    safe = drop_oos_outcome_columns(valid)
    out = builder.transform_oos(safe)
    assert all(col in out.columns for col in builder.feature_names())
