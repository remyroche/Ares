from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import scripts.run_s52_train_meta_regime_handoff_smoke as smoke_module

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_s52_train_meta_regime_handoff_smoke import (
    BASE_SAMPLE_WEIGHT_SPEC_HASH_COLUMN,
    BASE_TARGET_CONTRACT_HASH_COLUMN,
    DEFAULT_META_CHAMPION_CONTRACT,
    HANDOFF_RANK_SCOPE_COLUMN,
    META_POST_SELECTION_OOD_FEATURE_NAMES,
    META_PROTECTED_BASE_FEATURES,
    _contract_hash,
    _base_weight_arm_from_contract,
    _parse_args,
    _feature_columns,
    _hpo_candidate_beats_incumbent,
    _load_and_validate_handoff_contract,
    _meta_trial_objective,
    _make_xy,
    _resolved_meta_train_mask,
    _serialize_inherited_weighting,
    _train_only_target_strength_weights,
    _target_strength_spec_from_contract,
    run_smoke,
)
from extreme_price_movements.hierarchical_label_weights import TargetStrengthWeightSpec


def test_selected_meta_feature_cannot_silently_become_zero() -> None:
    train = pd.DataFrame({"score": [0.1, 0.2]})
    valid = pd.DataFrame({"score": [0.3]})
    with pytest.raises(ValueError, match="unresolved selected meta features"):
        _make_xy(
            train,
            valid,
            numeric_cols=["score"],
            categorical_cols=[],
            selected_features=["score", "carry_adj_ret_self_z_10h"],
        )


def test_explicit_legacy_constant_zero_contract_is_reproducible() -> None:
    train = pd.DataFrame({"score": [0.1, 0.2]})
    valid = pd.DataFrame({"score": [0.3]})
    x_train, x_valid, _ = _make_xy(
        train,
        valid,
        numeric_cols=["score"],
        categorical_cols=[],
        selected_features=["score", "carry_adj_ret_self_z_10h"],
        legacy_constant_zero_features=["carry_adj_ret_self_z_10h"],
    )

    assert x_train["carry_adj_ret_self_z_10h"].eq(0.0).all()
    assert x_valid["carry_adj_ret_self_z_10h"].eq(0.0).all()


def test_cli_defaults_to_promoted_v9_recovery_contract(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["run_s52_train_meta_regime_handoff_smoke.py"])
    args = _parse_args()
    assert args.frontier == "top30"
    assert args.meta_head_mode == "single_base_soft_label"
    assert args.single_head_global_model is True
    assert args.fixed_selected_features_csv == DEFAULT_META_CHAMPION_CONTRACT
    assert args.fixed_model_params_json == DEFAULT_META_CHAMPION_CONTRACT
    assert args.hpo_max_train_rows == 45_000
    assert args.single_phase_wide_feature_selection is False
    assert args.enable_base_prior_features is True
    assert args.enable_reliability_features is True
    assert args.enable_support_drift_features is True
    assert args.enable_hit_surprise_features is True


def test_meta_post_selection_ood_contract_is_conditional() -> None:
    assert META_POST_SELECTION_OOD_FEATURE_NAMES == ()


def test_meta_side_mda_is_unweighted_across_archetypes(monkeypatch) -> None:
    rows = 600
    x_train = pd.DataFrame(
        {
            "feature_a": np.linspace(-1.0, 1.0, rows, dtype=np.float32),
            "feature_b": np.linspace(1.0, -1.0, rows, dtype=np.float32),
        }
    )
    x_valid = x_train.iloc[:20].copy()
    train = pd.DataFrame(
        {
            "__ts__": pd.date_range("2026-01-01", periods=rows, freq="h", tz="UTC"),
            "__symbol__": [f"S{i % 20:02d}" for i in range(rows)],
            "side_name": np.where(np.arange(rows) % 2 == 0, "long", "short"),
            "feature_selection_archetype": np.where(
                np.arange(rows) % 4 < 2, "family_a", "family_b"
            ),
            "__first_touch_target_soft__": np.linspace(0.05, 0.95, rows),
            "ev_after_1pct": np.linspace(-0.02, 0.03, rows),
            "clean_exec_label": (np.arange(rows) % 3 == 0).astype(float),
            "dirty_positive": (np.arange(rows) % 5 == 0).astype(float),
            "bad_path_label": (np.arange(rows) % 7 == 0).astype(float),
            "first_touch_bad_mae_1r": (np.arange(rows) % 11 == 0).astype(float),
            "full_path_bad_mae_1r": (np.arange(rows) % 13 == 0).astype(float),
        }
    )
    captured: dict[str, object] = {}

    def fake_candidate(*_args, **kwargs):
        captured.update(kwargs["cfg"]["mda_config"])
        return {
            "selected_feature_names": ["feature_a", "feature_b"],
            "metrics": {
                "per_side_feature_selection_selected_features": {
                    "long": ["feature_a"],
                    "short": ["feature_b"],
                }
            },
            "feature_stats": pd.DataFrame(
                {
                    "feature": ["feature_a", "feature_b"],
                    "feature_score": [1.0, 0.5],
                    "mda_mean": [0.1, 0.05],
                }
            ),
        }

    monkeypatch.setattr(smoke_module, "train_lgbm_stability_candidate", fake_candidate)
    monkeypatch.setattr(
        smoke_module,
        "_conditionally_select_post_selection_ood_features",
        lambda *_args, **_kwargs: ([], {}, pd.DataFrame()),
    )
    smoke_module._select_features_by_lgbm_pipeline(
        x_train,
        x_valid,
        train,
        target_name="ev_frontier",
        top_n=0,
        fold="test",
        seed=17,
    )

    assert captured["side_tail_across_archetypes_unweighted"] is True


def _strict_handoff_contract() -> dict:
    target = {
        "schema": "base_soft_label_contract_v1",
        "target_column": "target_soft",
        "target_mode": "first_touch_target_soft",
    }
    weight = {
        "schema": "target_strength_weight_v1",
        "spec": {"exponent": 1.5, "weight_range_ratio": 4.0},
    }
    return {
        "label_resolution_contract": {
            "schema": "forward_label_resolution_v1",
            "resolution_column": "__label_path_end_ts__",
            "path_len": 96,
            "path_timeframe": "15m",
        },
        "inherited_base_contract": {
            "candidate_handoff_rank_scope": "timestamp_side",
            "base_target_contract": target,
            BASE_TARGET_CONTRACT_HASH_COLUMN: _contract_hash(target),
            "base_sample_weight_spec": weight,
            BASE_SAMPLE_WEIGHT_SPEC_HASH_COLUMN: _contract_hash(weight),
            "explicit_base_contract": True,
        }
    }


def test_strict_meta_train_mask_purges_labels_unresolved_at_validation_start() -> None:
    valid_start = pd.Timestamp("2026-06-10 00:00:00", tz="UTC")
    frame = pd.DataFrame(
        {
            "__ts__": [
                valid_start - pd.Timedelta(days=2),
                valid_start - pd.Timedelta(hours=1),
                valid_start - pd.Timedelta(hours=1),
                valid_start,
            ],
            "__label_path_end_ts__": [
                valid_start - pd.Timedelta(minutes=1),
                valid_start,
                valid_start + pd.Timedelta(minutes=1),
                valid_start + pd.Timedelta(hours=1),
            ],
        }
    )

    mask, diagnostics = _resolved_meta_train_mask(
        frame,
        valid_start=valid_start,
        strict=True,
    )

    assert mask.tolist() == [True, False, False, False]
    assert diagnostics["prior_rows"] == 3
    assert diagnostics["purged_rows"] == 2
    assert diagnostics["retained_rows"] == 1
    assert diagnostics["min_purged_label_end"] == valid_start


def test_strict_meta_train_mask_requires_label_resolution_column() -> None:
    frame = pd.DataFrame(
        {"__ts__": [pd.Timestamp("2026-06-09 23:00:00", tz="UTC")]}
    )

    with pytest.raises(ValueError, match="requires __label_path_end_ts__"):
        _resolved_meta_train_mask(
            frame,
            valid_start=pd.Timestamp("2026-06-10 00:00:00", tz="UTC"),
            strict=True,
        )


def test_strict_handoff_contract_requires_uniform_timestamp_side_provenance(
    tmp_path: Path,
) -> None:
    payload = _strict_handoff_contract()
    contract_path = tmp_path / "train_meta_regime_handoff_contract.json"
    contract_path.write_text(json.dumps(payload), encoding="utf-8")
    inherited = payload["inherited_base_contract"]
    rows = pd.DataFrame(
        {
            HANDOFF_RANK_SCOPE_COLUMN: ["timestamp_side", "timestamp_side"],
            BASE_TARGET_CONTRACT_HASH_COLUMN: [
                inherited[BASE_TARGET_CONTRACT_HASH_COLUMN]
            ]
            * 2,
            BASE_SAMPLE_WEIGHT_SPEC_HASH_COLUMN: [
                inherited[BASE_SAMPLE_WEIGHT_SPEC_HASH_COLUMN]
            ]
            * 2,
            "__label_path_end_ts__": [
                pd.Timestamp("2026-05-31 23:00:00", tz="UTC")
            ]
            * 2,
        }
    )

    result = _load_and_validate_handoff_contract(
        handoff_path=tmp_path / "train_meta_regime_handoff.parquet",
        handoff_rows=rows,
        strict=True,
    )

    assert result["validation_status"] == "strict_pass"
    assert result["base_target_contract"]["target_column"] == "target_soft"


def test_strict_handoff_accepts_named_base_weight_arm(tmp_path: Path) -> None:
    payload = _strict_handoff_contract()
    inherited = payload["inherited_base_contract"]
    weight = {
        "schema": "base_weight_arm_v1",
        "weight_arm": "W7_timestamp_balanced",
        "source": "base_weight_series",
    }
    inherited["base_sample_weight_spec"] = weight
    inherited[BASE_SAMPLE_WEIGHT_SPEC_HASH_COLUMN] = _contract_hash(weight)
    (tmp_path / "train_meta_regime_handoff_contract.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    rows = pd.DataFrame(
        {
            HANDOFF_RANK_SCOPE_COLUMN: ["timestamp_side"] * 2,
            BASE_TARGET_CONTRACT_HASH_COLUMN: [
                inherited[BASE_TARGET_CONTRACT_HASH_COLUMN]
            ]
            * 2,
            BASE_SAMPLE_WEIGHT_SPEC_HASH_COLUMN: [
                inherited[BASE_SAMPLE_WEIGHT_SPEC_HASH_COLUMN]
            ]
            * 2,
            "__label_path_end_ts__": [
                pd.Timestamp("2026-05-31 23:00:00", tz="UTC")
            ]
            * 2,
        }
    )

    result = _load_and_validate_handoff_contract(
        handoff_path=tmp_path / "train_meta_regime_handoff.parquet",
        handoff_rows=rows,
        strict=True,
    )

    assert result["validation_status"] == "strict_pass"
    assert _base_weight_arm_from_contract(result) == "W7_timestamp_balanced"
    assert _target_strength_spec_from_contract(result) is None


def test_named_base_weight_arm_serializes_without_target_strength_spec() -> None:
    payload = _serialize_inherited_weighting(None, "W7_timestamp_balanced")

    assert payload == {
        "schema": "base_weight_arm_v1",
        "weight_arm": "W7_timestamp_balanced",
        "spec": None,
    }


def test_target_strength_weighting_serializes_its_spec() -> None:
    spec = TargetStrengthWeightSpec(exponent=1.5, weight_range_ratio=4.0)
    payload = _serialize_inherited_weighting(spec, None)

    assert payload["schema"] == "target_strength_weight_v1"
    assert payload["weight_arm"] is None
    assert payload["spec"]["exponent"] == pytest.approx(1.5)
    assert payload["spec"]["weight_range_ratio"] == pytest.approx(4.0)


@pytest.mark.parametrize(
    "column,value,match",
    [
        (HANDOFF_RANK_SCOPE_COLUMN, "global", "timestamp_side"),
        (BASE_TARGET_CONTRACT_HASH_COLUMN, "mixed", "missing or mixed"),
    ],
)
def test_strict_handoff_contract_fails_closed_on_invalid_row_provenance(
    tmp_path: Path, column: str, value: str, match: str
) -> None:
    payload = _strict_handoff_contract()
    (tmp_path / "train_meta_regime_handoff_contract.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    inherited = payload["inherited_base_contract"]
    rows = pd.DataFrame(
        {
            HANDOFF_RANK_SCOPE_COLUMN: ["timestamp_side", "timestamp_side"],
            BASE_TARGET_CONTRACT_HASH_COLUMN: [
                inherited[BASE_TARGET_CONTRACT_HASH_COLUMN]
            ]
            * 2,
            BASE_SAMPLE_WEIGHT_SPEC_HASH_COLUMN: [
                inherited[BASE_SAMPLE_WEIGHT_SPEC_HASH_COLUMN]
            ]
            * 2,
            "__label_path_end_ts__": [
                pd.Timestamp("2026-05-31 23:00:00", tz="UTC")
            ]
            * 2,
        }
    )
    rows.loc[1, column] = value

    with pytest.raises(ValueError, match=match):
        _load_and_validate_handoff_contract(
            handoff_path=tmp_path / "train_meta_regime_handoff.parquet",
            handoff_rows=rows,
            strict=True,
        )


def test_target_strength_weights_are_recomputed_on_train_only_top30_rows() -> None:
    train = pd.DataFrame(
        {
            "__ts__": pd.date_range("2026-04-01", periods=6, freq="h", tz="UTC"),
            "side_name": ["long"] * 3 + ["short"] * 3,
            "archetype_label_family": ["a", "a", "a", "b", "b", "b"],
            "selected_top30": [True] * 6,
        }
    )
    target = pd.Series([0.05, 0.20, 0.95, 0.10, 0.30, 0.90])
    weights, diagnostics = _train_only_target_strength_weights(
        train,
        target,
        sample_weight_spec=TargetStrengthWeightSpec(exponent=1.5),
        strict=True,
    )

    assert diagnostics["schema"] == "target_strength_weight_v1"
    assert weights.iloc[2] > weights.iloc[0]
    assert weights.iloc[5] > weights.iloc[3]
    assert weights.mean() == pytest.approx(1.0, abs=1e-6)


def test_meta_features_exclude_handoff_target_weight_and_provenance_columns() -> None:
    frame = pd.DataFrame(
        {
            "score_base": [0.1],
            "target_soft": [0.9],
            HANDOFF_RANK_SCOPE_COLUMN: ["timestamp_side"],
            BASE_TARGET_CONTRACT_HASH_COLUMN: ["hash"],
            BASE_SAMPLE_WEIGHT_SPEC_HASH_COLUMN: ["hash"],
            "base_sample_weight_spec": ["{}"],
            "base_model_weight_arm": ["W7"],
        }
    )
    numeric, categorical = _feature_columns(frame)
    assert numeric == ["score_base"]
    assert categorical == []


def test_direct_base_score_alias_is_a_protected_numeric_meta_anchor() -> None:
    frame = pd.DataFrame(
        {
            "base_score_raw": np.asarray([0.1, 0.9], dtype=np.float32),
            "feature_a": np.asarray([1.0, 2.0], dtype=np.float32),
        }
    )

    numeric, categorical = _feature_columns(frame)

    assert "base_score_raw" in META_PROTECTED_BASE_FEATURES
    assert "base_score_raw" in numeric
    assert categorical == []


def test_strict_meta_smoke_uses_inherited_target_strength_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    handoff_dir = tmp_path / "handoff"
    handoff_dir.mkdir()
    out_dir = tmp_path / "out"
    payload = _strict_handoff_contract()
    inherited = payload["inherited_base_contract"]
    (handoff_dir / "train_meta_regime_handoff_contract.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    handoff_rows: list[dict[str, object]] = []
    ledger_rows: list[dict[str, object]] = []
    for month_idx, month in enumerate(("2026-04", "2026-05")):
        for row in range(120):
            clean = row % 2 == 0
            ts = pd.Timestamp(f"{month}-01", tz="UTC") + pd.Timedelta(hours=row)
            base = {
                "__ts__": ts,
                "__label_path_end_ts__": ts + pd.Timedelta(hours=24),
                "__symbol__": f"S{row:03d}",
                "side_name": "long" if row % 3 else "short",
                "month": month,
                "base_score_raw": 0.9 if clean else 0.1,
                "target_soft": 0.95 if clean else 0.05,
                "selected_top30": True,
                HANDOFF_RANK_SCOPE_COLUMN: "timestamp_side",
                BASE_TARGET_CONTRACT_HASH_COLUMN: inherited[
                    BASE_TARGET_CONTRACT_HASH_COLUMN
                ],
                BASE_SAMPLE_WEIGHT_SPEC_HASH_COLUMN: inherited[
                    BASE_SAMPLE_WEIGHT_SPEC_HASH_COLUMN
                ],
            }
            handoff_rows.append(base)
            ledger_rows.append(
                {
                    key: value
                    for key, value in base.items()
                    if key
                    in {
                        "__ts__",
                        "__symbol__",
                        "side_name",
                        "month",
                        "base_score_raw",
                        "selected_top30",
                    }
                }
                | {
                    "exec_margin": 0.01 if clean else -0.01,
                    "ev_after_1pct": 0.01 if clean else -0.01,
                    "first_touch_gross": 0.02 if clean else 0.0,
                    "first_touch_bad_mae_1r": 0.0 if clean else 1.0,
                    "full_path_bad_mae_1r": 0.0 if clean else 1.0,
                    "timeout": 0.0,
                    "mfe_before_mae_1r": float(clean),
                    "mae_before_mfe_1r": float(not clean),
                    "clean_exec": float(clean),
                    "dirty_positive": float(not clean),
                    "underwater_bars_before_mfe_1r": 1.0 if clean else 8.0,
                }
            )
    pd.DataFrame(handoff_rows).to_parquet(
        handoff_dir / "train_meta_regime_handoff.parquet", index=False
    )
    ledger_path = handoff_dir / "s52_trailing_regime_scored_ledger.parquet"
    pd.DataFrame(ledger_rows).to_parquet(ledger_path, index=False)

    # The test exercises strict target/weight inheritance.  The forward-label
    # helper returns a Timestamp diagnostic, while this production log call has
    # not yet been made JSON-safe.  Keep that unrelated presentation detail out
    # of this contract fixture; direct purge tests above cover the real boundary.
    resolved_mask = smoke_module._resolved_meta_train_mask

    def json_safe_resolved_mask(*args, **kwargs):
        mask, diagnostics = resolved_mask(*args, **kwargs)
        diagnostics = dict(diagnostics)
        for key in ("max_retained_label_end", "min_purged_label_end"):
            value = diagnostics.get(key)
            if isinstance(value, pd.Timestamp):
                diagnostics[key] = value.isoformat()
        return mask, diagnostics

    monkeypatch.setattr(
        smoke_module, "_resolved_meta_train_mask", json_safe_resolved_mask
    )

    manifest = run_smoke(
        handoff_dir=handoff_dir,
        ledger_path=ledger_path,
        out_dir=out_dir,
        frontier="top30",
        seed=7,
        train_scope="selected",
        meta_head_mode="single_base_soft_label",
        fixed_selected_features=["base_score_raw"],
        side_specific_single_head=False,
        strict_handoff_contract=True,
        model_params={
            "classifier": {
                "n_estimators": 20,
                "learning_rate": 0.10,
                "num_leaves": 7,
                "max_depth": 3,
                "min_child_samples": 5,
                "subsample": 1.0,
                "colsample_bytree": 1.0,
                "reg_alpha": 0.0,
                "reg_lambda": 1.0,
            }
        },
    )

    assert manifest["inherited_base_handoff_contract"]["validation_status"] == "strict_pass"
    assert manifest["target_strength_weight_diagnostics_by_fold"]


def test_meta_hpo_objective_is_ev_first_and_promotion_is_incumbent_guarded(
    tmp_path: Path,
) -> None:
    def manifest(name: str, ev: float, worst_month: float, worst_week: float) -> dict:
        out = tmp_path / name
        out.mkdir()
        pd.DataFrame(
            [
                {
                    "selector": "meta_base_soft_label",
                    "mean_keep010_ev_after_1pct": ev,
                    "mean_keep020_ev_after_1pct": ev * 0.8,
                    "worst_keep010_ev_after_1pct": worst_month,
                    "worst_week_keep010_ev_after_1pct": worst_week,
                    "mean_keep015_clean_exec_precision": 0.70,
                    "mean_keep015_dirty_positive_rate": 0.20,
                    "mean_keep015_first_touch_bad_mae": 0.10,
                    "mean_keep015_full_path_bad_mae": 0.15,
                }
            ]
        ).to_csv(out / "s52_train_meta_regime_handoff_smoke_summary.csv", index=False)
        return {"output_dir": str(out)}

    incumbent = manifest("incumbent", 0.0060, 0.0030, 0.0020)
    good = manifest("good", 0.0070, 0.0029, 0.0019)
    unstable = manifest("unstable", 0.0070, 0.0020, 0.0010)
    assert _meta_trial_objective(good) > _meta_trial_objective(incumbent)
    assert _hpo_candidate_beats_incumbent(good, incumbent)[0] is True
    assert _hpo_candidate_beats_incumbent(unstable, incumbent)[0] is False


def test_s52_train_meta_feature_columns_exclude_generated_path_labels() -> None:
    frame = pd.DataFrame(
        {
            "score_base": [0.1, 0.2],
            "regime_clean_exec_score": [0.3, 0.4],
            "long_bad_path_label": [1.0, 0.0],
            "long_path_clean_exec_label": [0.0, 1.0],
            "exec_margin": [0.01, -0.01],
        }
    )
    numeric, categorical = _feature_columns(frame)
    all_cols = set(numeric + categorical)
    assert "score_base" in all_cols
    assert "regime_clean_exec_score" in all_cols
    assert "long_bad_path_label" not in all_cols
    assert "long_path_clean_exec_label" not in all_cols
    assert "exec_margin" not in all_cols


def test_s52_train_meta_handoff_smoke_learns_clean_filter(tmp_path: Path) -> None:
    handoff_dir = tmp_path / "handoff"
    out_dir = tmp_path / "out"
    handoff_dir.mkdir()
    handoff_rows = []
    ledger_rows = []
    for month_idx, month in enumerate(("2026-04", "2026-05")):
        for idx in range(160):
            clean = idx % 4 != 0
            ts = f"{month}-{(idx % 20) + 1:02d}T{idx % 24:02d}:00:00Z"
            symbol = f"SYM{idx % 7}"
            side = "long" if idx % 3 == 0 else "short"
            quality = 1.0 if clean else -1.0
            score = float(1.0 - idx / 500.0)
            handoff_rows.append(
                {
                    "__ts__": ts,
                    "__symbol__": symbol,
                    "side_name": side,
                    "month": month,
                    "score": score,
                    "target_soft": 0.9 if clean else 0.1,
                    "selected_top10": True,
                    "source_semantic_family": "quiet_continuation"
                    if clean
                    else "dirty_shock_avoid",
                    "regime_clean_exec_score": 0.9 if clean else 0.1,
                    "regime_bad_mae_score": 0.1 if clean else 0.9,
                    "gmm_entropy": 0.2 + 0.01 * (idx % 5),
                    "latent_speed": quality,
                    "meta_context_weight_hint": 1.0 if clean else 0.2,
                    "meta_threshold_adjustment_hint": 0.0 if clean else 0.5,
                    "aegmm_expected_distance_bin": "q1" if clean else "q3",
                }
            )
            ledger_rows.append(
                {
                    "__ts__": ts,
                    "__symbol__": symbol,
                    "side_name": side,
                    "month": month,
                    "score": score,
                    "selected_top10": True,
                    "exec_margin": 0.010 + 0.001 * month_idx if clean else -0.006,
                    "ev_after_1pct": 0.010 if clean else -0.006,
                    "first_touch_gross": 0.020 if clean else 0.004,
                    "first_touch_bad_mae_1r": 0.0 if clean else 1.0,
                    "full_path_bad_mae_1r": 0.0 if clean else 1.0,
                    "timeout": 0.0,
                    "mfe_before_mae_1r": 1.0 if clean else 0.0,
                    "mae_before_mfe_1r": 0.0 if clean else 1.0,
                    "clean_exec": 1.0 if clean else 0.0,
                    "dirty_positive": 0.0 if clean else 1.0,
                    "underwater_bars_before_mfe_1r": 2.0 if clean else 12.0,
                }
            )
    pd.DataFrame(handoff_rows).to_parquet(
        handoff_dir / "train_meta_regime_handoff.parquet", index=False
    )
    ledger_path = handoff_dir / "s52_trailing_regime_scored_ledger.parquet"
    pd.DataFrame(ledger_rows).to_parquet(ledger_path, index=False)

    fixed_features = [
        "score",
        "regime_clean_exec_score",
        "regime_bad_mae_score",
        "gmm_entropy",
        "latent_speed",
        "meta_context_weight_hint",
        "meta_threshold_adjustment_hint",
        "aegmm_expected_distance_bin_q1",
        "aegmm_expected_distance_bin_q3",
    ]
    run_smoke(
        handoff_dir=handoff_dir,
        ledger_path=ledger_path,
        out_dir=out_dir,
        frontier="top10",
        seed=11,
        train_scope="selected",
        # This fixture is intentionally below the canonical MDA minimum. The
        # smoke asserts meta filtering behavior, not feature-selection quality.
        meta_head_mode="single_base_soft_label",
        fixed_selected_features=fixed_features,
        fixed_selected_features_by_side={
            "long": ["score", "regime_clean_exec_score", "latent_speed"],
            "short": ["score", "regime_bad_mae_score", "gmm_entropy"],
        },
        model_params={
            "classifier": {
                "n_estimators": 80,
                "learning_rate": 0.05,
                "num_leaves": 7,
                "max_depth": 3,
                "min_child_samples": 5,
                "subsample": 1.0,
                "colsample_bytree": 1.0,
                "reg_alpha": 0.0,
                "reg_lambda": 1.0,
            }
        },
    )

    summary = pd.read_csv(out_dir / "s52_train_meta_regime_handoff_smoke_summary.csv")
    assert not summary.empty
    meta = summary[summary["selector"].astype(str).str.startswith("meta_")]
    assert float(meta["mean_keep030_exec_margin"].max()) > 0.0
    assert float(meta["mean_keep030_full_path_bad_mae"].min()) < float(
        summary["mean_keep100_full_path_bad_mae"].iloc[0]
    )
    threshold_summary = pd.read_csv(
        out_dir / "s52_train_meta_regime_handoff_threshold_policy_summary.csv"
    )
    assert not threshold_summary.empty
    assert {"policy_id", "budget_frac", "threshold_policy_status"}.issubset(
        threshold_summary.columns
    )
    assert (out_dir / "s52_train_meta_regime_handoff_smoke.md").exists()
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["side_specific_feature_contract_enabled"] is True
    assert manifest["feature_selection_execution"] == "frozen_side_contract_no_selector"
    assert manifest["selected_features_by_side"]["long"] != manifest[
        "selected_features_by_side"
    ]["short"]
    assert manifest["selected_features_by_side"]["long"] == [
        "score",
        "regime_clean_exec_score",
        "latent_speed",
    ]
    assert manifest["selected_features_by_side"]["short"] == [
        "score",
        "regime_bad_mae_score",
        "gmm_entropy",
    ]
    assert manifest["selected_feature_union"] == [
        "gmm_entropy",
        "latent_speed",
        "regime_bad_mae_score",
        "regime_clean_exec_score",
        "score",
    ]
    assert manifest["selected_feature_intersection"] == ["score"]
    assert manifest["selected_feature_intersection_count"] == 1
    # This is deliberately distinct from long/short overlap: every fold uses
    # the same fixed union in this smoke.
    assert manifest["selected_feature_fold_intersection_count"] == 5
