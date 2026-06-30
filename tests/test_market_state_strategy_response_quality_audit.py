from pathlib import Path

import numpy as np
import pandas as pd

from scripts.audit_market_state_strategy_response_quality import (
    DEFAULT_POLICY,
    _compute_quality_reasons,
    audit_strategy_response_quality,
)


def _response_rows(*, bad: bool = False) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(17)
    heads = ["short_asset", "short_boll"]
    for fold in [1, 2]:
        for head_idx, head in enumerate(heads):
            for i in range(40):
                ts = pd.Timestamp("2026-05-01", tz="UTC") + pd.Timedelta(hours=i)
                signal = (i - 20) / 20.0 + 0.15 * head_idx
                actual_utility = signal * 0.02 + rng.normal(0.0, 0.001)
                pred_utility = (-signal if bad else signal) * 0.02
                full_sl = float(signal < -0.2)
                timeout = float(abs(signal) < 0.1)
                pred_full_sl = 0.85 if full_sl else 0.05
                pred_timeout = 0.80 if timeout else 0.05
                if bad:
                    pred_full_sl = 0.95 - pred_full_sl
                    pred_timeout = 0.95 - pred_timeout
                rows.append(
                    {
                        "timestamp": ts,
                        "strategy_id": f"{head}_strategy",
                        "head": head,
                        "side": "short",
                        "symbol": f"SYM{i % 6}/USD:USD",
                        "_rank": 0.55 + 0.45 * (i / 39.0),
                        "_threshold": 0.70,
                        "_net_return": actual_utility,
                        "_is_full_sl": full_sl,
                        "_is_timeout": timeout,
                        "state_feature_coverage": 0.60 if bad else 1.0,
                        "response_feature_coverage": 0.60 if bad else 1.0,
                        "state_input_coverage": 1.0,
                        "state_low_input_coverage": 0.0,
                        "state_ood_score": 5.0 if bad else 0.1,
                        "state_ood_cutoff": 1.0,
                        "state_ood_flag": bad,
                        "base_mu": 0.0,
                        "base_psl": 0.3,
                        "base_pto": 0.1,
                        "pred_eu_mean": pred_utility,
                        "pred_eu_q10": pred_utility - 0.001,
                        "pred_excess_full_sl": pred_full_sl - 0.3,
                        "pred_excess_timeout": pred_timeout - 0.1,
                        "pred_mean_utility": pred_utility,
                        "pred_lcb_utility": pred_utility - 0.001,
                        "pred_full_sl": pred_full_sl,
                        "pred_timeout": pred_timeout,
                        "fold": fold,
                        "arm": "S1_observed_axes_shared_response",
                        "state_prediction_contract": "outer_fold_validation_state_scores",
                        "actual_resid_utility": actual_utility,
                        "actual_resid_full_sl": full_sl - 0.3,
                        "actual_resid_timeout": timeout - 0.1,
                        "pred_resid_utility": pred_utility,
                        "pred_resid_utility_lcb": pred_utility - 0.001,
                        "pred_resid_full_sl": pred_full_sl - 0.3,
                        "pred_resid_timeout": pred_timeout - 0.1,
                    }
                )
    return pd.DataFrame(rows)


def _write_bundle(root: Path, response: pd.DataFrame) -> None:
    root.mkdir(parents=True, exist_ok=True)
    response.to_parquet(root / "strategy_response_oof_predictions.parquet", index=False)
    effects = pd.DataFrame(
        [
            {
                "fold": 1,
                "arm": "S1_observed_axes_shared_response",
                "scope": "head",
                "scope_value": "short_asset",
                "state_feature": "state_shock",
                "target": "pred_resid_utility",
                "rows": 40,
                "state_q10": -1.0,
                "state_q90": 1.0,
                "target_mean_state_q10": -0.01,
                "target_mean_state_q90": 0.01,
                "target_q90_minus_q10": 0.02,
                "pearson": 0.5,
                "spearman": 0.5,
            }
        ]
    )
    effects.to_csv(root / "strategy_state_effect_matrix.csv", index=False)


def test_strategy_response_quality_audit_passes_learnable_response(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifact"
    output_dir = tmp_path / "out"
    _write_bundle(artifact_dir, _response_rows(bad=False))

    payload = audit_strategy_response_quality(
        artifact_dir,
        output_dir,
        policy={
            "min_total_rows": 50,
            "min_fold_rows": 20,
            "min_timestamp_count": 10,
            "max_state_ood_share": 0.1,
        },
    )

    by_head = pd.read_csv(output_dir / "market_state_strategy_response_quality_by_head.csv")
    by_arm = pd.read_csv(output_dir / "market_state_strategy_response_quality_by_arm.csv")
    blockers = pd.read_csv(output_dir / "market_state_strategy_response_gate_blockers.csv")

    assert payload["passed"] is True
    assert payload["structural_passed"] is True
    assert payload["quality_gate_passed"] is True
    assert payload["controller_activation_allowed"] is True
    assert payload["quality_passing_arm_count"] == 1
    assert payload["quality_passing_head_count"] == 2
    assert payload["support_blocked_heads"] == []
    assert payload["signal_passing_but_support_blocked_heads"] == []
    assert payload["response_gate_blocker_counts"] == {"passed": 2}
    assert payload["support_only_blocked_candidates"] == 0
    assert payload["min_required_extra_rows_to_clear_support"] == 0
    assert set(blockers["blocker_type"]) == {"passed"}
    assert by_arm.loc[0, "all_heads_passed_response_quality"] in {True, "True", np.bool_(True)}
    assert bool(by_head["response_quality_passed"].all())
    assert bool(by_head["response_support_passed"].all())
    assert bool(by_head["response_signal_passed"].all())
    assert float(by_head["median_utility_spearman"].min()) > 0.9
    assert (output_dir / "market_state_strategy_response_top_state_effects.csv").exists()
    assert (output_dir / "market_state_strategy_response_quality_report.md").exists()


def test_strategy_response_quality_audit_explains_bad_response(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifact"
    output_dir = tmp_path / "out"
    _write_bundle(artifact_dir, _response_rows(bad=True))

    payload = audit_strategy_response_quality(
        artifact_dir,
        output_dir,
        policy={
            "min_total_rows": 50,
            "min_fold_rows": 20,
            "min_timestamp_count": 10,
            "max_state_ood_share": 0.1,
        },
    )

    by_head = pd.read_csv(output_dir / "market_state_strategy_response_quality_by_head.csv")
    blockers = pd.read_csv(output_dir / "market_state_strategy_response_gate_blockers.csv")
    reasons = ";".join(by_head["response_quality_fail_reasons"].fillna("").astype(str).tolist())

    assert payload["passed"] is True
    assert payload["structural_passed"] is True
    assert payload["quality_gate_passed"] is False
    assert payload["controller_activation_allowed"] is False
    assert payload["quality_passing_arm_count"] == 0
    assert payload["quality_passing_head_count"] == 0
    assert payload["support_blocked_heads"] == []
    assert payload["response_gate_blocker_counts"] == {"signal_quality": 2}
    assert not bool(by_head["response_quality_passed"].any())
    assert bool(by_head["response_support_passed"].all())
    assert not bool(by_head["response_signal_passed"].any())
    assert set(blockers["blocker_type"]) == {"signal_quality"}
    assert bool(blockers["promotion_waiver_allowed"].any()) is False
    assert "low_response_feature_coverage" in reasons
    assert "state_ood_share_too_high" in reasons
    assert "median_utility_ic_not_positive" in reasons


def test_strategy_response_quality_audit_separates_support_from_signal(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifact"
    output_dir = tmp_path / "out"
    response = _response_rows(bad=False)
    response = response.loc[
        ~(
            response["head"].eq("short_asset")
            & response["fold"].eq(2)
            & response.groupby(["head", "fold"]).cumcount().ge(10)
        )
    ].copy()
    _write_bundle(artifact_dir, response)

    payload = audit_strategy_response_quality(
        artifact_dir,
        output_dir,
        policy={
            "min_total_rows": 20,
            "min_fold_rows": 20,
            "min_timestamp_count": 5,
            "max_state_ood_share": 0.1,
        },
    )

    by_head = pd.read_csv(output_dir / "market_state_strategy_response_quality_by_head.csv")
    blockers = pd.read_csv(output_dir / "market_state_strategy_response_gate_blockers.csv")
    blocked = by_head.loc[by_head["head"].eq("short_asset")].iloc[0]
    blocker = blockers.loc[blockers["head"].eq("short_asset")].iloc[0]

    assert payload["quality_gate_passed"] is False
    assert payload["support_blocked_heads"] == ["short_asset"]
    assert payload["signal_passing_but_support_blocked_heads"] == ["short_asset"]
    assert payload["response_gate_blocker_counts"] == {"passed": 1, "support_only": 1}
    assert payload["support_only_blocked_candidates"] == 1
    assert payload["min_required_extra_rows_to_clear_support"] == 10
    assert bool(blocked["response_support_passed"]) is False
    assert bool(blocked["response_signal_passed"]) is True
    assert blocked["under_supported_folds"] == "2:10"
    assert blocker["blocker_type"] == "support_only"
    assert blocker["required_extra_rows_by_fold"] == "2:+10"
    assert int(blocker["required_extra_rows_total_to_clear_support"]) == 10
    assert "do_not_relax_quality_gate" in blocker["next_action"]
    assert bool(blocker["promotion_waiver_allowed"]) is False


def test_strategy_response_quality_gate_tolerates_float_boundary_coverage() -> None:
    row = pd.Series(
        {
            "rows_total": 100,
            "min_fold_rows": 30,
            "timestamp_count_total": 10,
            "mean_response_feature_coverage": 0.8 - 1e-16,
            "mean_state_feature_coverage": 0.8 - 1e-16,
            "mean_state_ood_share": 0.1 + 1e-16,
            "median_utility_spearman": 0.1,
            "q25_utility_spearman": -0.02 - 1e-16,
            "positive_utility_ic_share": 0.5 - 1e-16,
            "median_utility_decile_spread": 0.1,
            "q25_utility_decile_spread": -0.005 - 1e-16,
            "median_full_sl_calibration_error": 0.2 + 1e-16,
            "median_timeout_calibration_error": 0.2 + 1e-16,
        }
    )

    assert _compute_quality_reasons(row, DEFAULT_POLICY) == []
