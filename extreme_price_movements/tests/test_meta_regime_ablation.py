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
    _failure_shock_labels,
    _residual_state_pre_entry_columns,
    apply_regime_builder_fold,
    drop_oos_outcome_columns,
    make_regime_builder,
)
from scripts.report_train_meta_extended_pool_ablation_metrics import build_report


ARMS = [
    "current_archetype_meta_regimes",
    "meta_feature_only_regimes",
    "base_error_signature_regimes",
    "joint_feature_error_regimes",
    "hit_surprise_failure_shock_regimes",
    "mlp_failure_shock_regimes",
    "residual_event_aegmm_local",
    "residual_event_aegmm_local_market",
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


def test_failure_shocks_use_global_ev_tail_and_exclude_recent_performance() -> None:
    frame = _frame(2_400)
    # A single well-supported local stream keeps the tail contract test focused
    # on global-EV target construction rather than partition sparsity.
    frame["side_name"] = "long"
    frame["archetype_label_family"] = "arch_a"
    frame["base_arch_hit_surprise_hl3d"] = np.linspace(-3.0, 3.0, len(frame))
    labels, metadata = _failure_shock_labels(frame)
    contract = metadata["tail_contract"]
    assert contract["status"] == "ok"
    assert contract["contract"] == "global_topk_ev_target_with_side_x_archetype_train_thresholds"
    assert metadata["top20_rows"] >= metadata["top10_rows"] > 0
    assert len(labels) == len(frame)
    numeric, categorical = _residual_state_pre_entry_columns(frame)
    assert "base_arch_hit_surprise_hl3d" not in {*numeric, *categorical}


def test_ablation_report_writes_high_surprise_event_deltas(tmp_path: Path) -> None:
    root = tmp_path / "matrix"
    out = tmp_path / "report"
    rng = np.random.default_rng(7)
    ts = pd.date_range("2026-04-01", periods=80, freq="6h", tz="UTC")
    base = pd.DataFrame(
        {
            "__ts__": ts,
            "__symbol__": np.where(np.arange(len(ts)) % 2 == 0, "BTC", "ETH"),
            "side_name": np.where(np.arange(len(ts)) % 2 == 0, "long", "short"),
            "archetype_policy_key": np.where(
                np.arange(len(ts)) % 3 == 0, "arch_a", "arch_b"
            ),
            "score_meta_base_soft_label": np.linspace(0.99, 0.20, len(ts)),
            "score_base": np.linspace(0.95, 0.10, len(ts)),
            "clean_exec": 1.0,
            "dirty_positive": 0.0,
            "full_path_bad_mae_1r": 0.0,
            "first_touch_bad_mae_1r": 0.0,
            "timeout": 0.0,
            "exec_margin": rng.normal(0.01, 0.001, len(ts)),
            "ev_after_1pct": rng.normal(0.01, 0.001, len(ts)),
        }
    )
    # Make the baseline top tail contain a clear negative-surprise cell.
    base.loc[:7, "side_name"] = "long"
    base.loc[:7, "archetype_policy_key"] = "arch_a"
    base.loc[:7, "clean_exec"] = 0.0
    base.loc[:7, "dirty_positive"] = 1.0
    base.loc[:7, "full_path_bad_mae_1r"] = 1.0
    base.loc[:7, "ev_after_1pct"] = -0.01
    challenger = base.copy()
    challenger.loc[:7, "score_meta_base_soft_label"] = 0.40
    challenger.loc[8:15, "score_meta_base_soft_label"] = 0.99
    for arm, frame in (
        ("baseline_current_full_context", base),
        ("hit_surprise_failure_shock_regimes", challenger),
    ):
        arm_dir = root / arm
        arm_dir.mkdir(parents=True)
        frame.to_parquet(
            arm_dir / "s52_train_meta_regime_handoff_smoke_predictions.parquet",
            index=False,
        )
    manifest = build_report(root_dir=root, out_dir=out, min_group_rows=1)
    delta_path = Path(manifest["outputs"]["high_surprise_event_deltas"])
    assert delta_path.exists()
    deltas = pd.read_csv(delta_path)
    assert "high_surprise_significantly_improved" in deltas.columns
    assert deltas["baseline_high_surprise_event"].fillna(False).any()
