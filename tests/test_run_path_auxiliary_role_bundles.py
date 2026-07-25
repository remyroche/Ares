from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.side_aware import candidate_id_series
from scripts.run_path_auxiliary_role_bundles import (
    MEANINGFUL_EVENT_ROLE,
    _load_checkpoint,
    _persist_head,
    _promotion_contract,
    _representation_role_metrics,
    _timing_role_results,
)


def _labels() -> pd.DataFrame:
    timestamp = pd.date_range("2026-05-01", periods=8, freq="h", tz="UTC")
    side = np.repeat(["long", "short"], 4)
    symbol = np.resize(["AAA/USD:USD", "BBB/USD:USD"], 8)
    hit = np.resize([1.0, 0.0, 1.0, 0.0], 8)
    return pd.DataFrame(
        {
            "__ts__": timestamp,
            "__symbol__": symbol,
            "side": side,
            "candidate_id": candidate_id_series(
                timestamp, pd.Series(symbol), "1h", pd.Series(side)
            ),
            "archetype": "base",
            "gmm_representation_available": np.resize(
                [1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0], 8
            ),
            "__path_auxiliary_target_valid__": 1.0,
            "__meaningful_mfe_reached_12h__": hit,
            "__time_to_first_meaningful_mfe_hours_12h__": np.where(
                hit > 0.5, 2.0, 12.0
            ),
            "__peak_mfe_atr_12h__": np.where(hit > 0.5, 3.0, 0.0),
            "__mae_before_meaningful_mfe_atr_12h__": np.where(hit > 0.5, 0.5, 3.0),
            "__bars_to_adverse_extreme_before_mfe_12h__": np.resize(
                [1.0, 4.0, 2.0, 5.0], 8
            ),
            "__bars_to_confirmed_adverse_trough__": np.resize(
                [4.0, np.nan, 5.0, np.nan], 8
            ),
            "__future_slope_atr_per_hour_12h__": np.resize([1.0, 0.0, 0.5, 0.1], 8),
            "__label_end_ts__": timestamp + pd.Timedelta(hours=13),
        }
    )


def _side_state(side: str) -> dict[str, object]:
    return {
        "oof_fold_ids": np.zeros(4, dtype=np.int16),
        "fold_provenance": [
            {
                "fold": 0,
                "fold_month": "2026-05",
                "valid_start": "2026-05-01T00:00:00+00:00",
                "valid_end": "2026-05-31T23:00:00+00:00",
                "training_label_resolved_max": "2026-04-30T23:00:00+00:00",
                "side": side,
            }
        ],
    }


def _role_result(
    role_name: str,
    values: list[float],
    *,
    task: str = "regression",
) -> dict[str, object]:
    prediction = np.asarray(values, dtype=np.float32)
    target = (
        np.resize(np.array([0.0, 1.0]), len(prediction))
        if task == "binary"
        else np.linspace(0.0, 1.0, len(prediction))
    )
    return {
        "role_name": role_name,
        "task_kind": task,
        "target": target,
        "role_train_mask": np.ones(len(prediction), dtype=bool),
        "valid_mask": np.ones(len(prediction), dtype=bool),
        "oof_predictions": prediction,
        "oof_prediction_mask": np.isfinite(prediction),
        "oof_fold_ids": np.zeros(len(prediction), dtype=np.int16),
        "side_results": {
            "long": _side_state("long"),
            "short": _side_state("short"),
        },
    }


def test_checkpoint_ignores_preflight_telemetry_and_rejects_corruption(
    tmp_path: Path,
) -> None:
    output = tmp_path / "run"
    output.mkdir()
    (output / "training_resource_telemetry.jsonl").write_text("{}\n", encoding="utf-8")
    fingerprint = {"sha256": "abc", "payload": {"x": 1}}

    checkpoint = _load_checkpoint(output, run_fingerprint=fingerprint, overwrite=False)
    assert checkpoint["run_fingerprint"]["sha256"] == "abc"

    path = output / "checkpoint.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["run_fingerprint"]["sha256"] = "different"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="mismatched contract"):
        _load_checkpoint(output, run_fingerprint=fingerprint, overwrite=False)


def test_peak_bundle_retains_all_rows_and_shared_event_prediction(
    tmp_path: Path,
) -> None:
    labels = _labels()
    event = _role_result(
        MEANINGFUL_EVENT_ROLE,
        [0.2, 0.4, 0.6, 0.8, 0.3, 0.5, 0.7, 0.9],
        task="binary",
    )
    roles = {
        MEANINGFUL_EVENT_ROLE: event,
        "peak_mfe_12h_atr.conditional_mean": _role_result(
            "peak_mfe_12h_atr.conditional_mean", [2.0] * 8
        ),
        "peak_mfe_12h_atr.conditional_q80": _role_result(
            "peak_mfe_12h_atr.conditional_q80", [3.0] * 8
        ),
    }
    manifest = _persist_head(
        labels,
        head_name="peak_mfe_12h_atr",
        role_results=roles,
        reference_end=pd.Timestamp("2026-05-01T00:00:00Z"),
        output_dir=tmp_path,
    )
    bundle = pd.read_parquet(manifest["oof_bundle"]["path"])

    assert len(bundle) == len(labels)
    assert bundle["oof_available"].all()
    np.testing.assert_array_equal(
        bundle["pred_p_meaningful_mfe_12h"].to_numpy(),
        event["oof_predictions"],
    )
    np.testing.assert_allclose(
        bundle["pred_expected_peak_mfe_atr"],
        np.asarray(event["oof_predictions"]) * 2.0,
    )
    assert manifest["final_refit_excluded_from_oof"] is True
    gate = json.loads(Path(manifest["promotion_gate"]["path"]).read_text())
    assert gate["research_ablation"]["status"] == (
        "ELIGIBLE_FOR_EXECUTION_EV_RESEARCH_ABLATION"
    )
    assert gate["production_promotion"]["production_ready"] is False
    assert (
        "pred_peak_mfe_if_hit_q80_atr"
        not in gate["research_ablation"]["prediction_columns"]
    )
    assert manifest["production_ready"] is False


def test_peak_q80_quality_gate_withholds_materially_miscalibrated_slice() -> None:
    def report(coverage: float) -> dict[str, object]:
        return {
            side: {
                availability: {
                    "metric_rows": 1_000
                    if (side, availability) == ("long", "available")
                    else 0,
                    "status": "evaluated"
                    if (side, availability) == ("long", "available")
                    else "not_evaluable_zero_missing_support",
                    "metrics": {
                        "empirical_coverage_alpha_0_8": coverage
                        if (side, availability) == ("long", "available")
                        else np.nan,
                        "pinball_loss_alpha_0_8": 1.0,
                        "bias": 0.0,
                        "spearman_ic": 0.1,
                    },
                }
                for availability in ("available", "missing")
            }
            for side in ("long", "short")
        }

    generic = report(0.80)
    gate = _promotion_contract(
        "peak_mfe_12h_atr",
        {
            MEANINGFUL_EVENT_ROLE: generic,
            "peak_mfe_12h_atr.conditional_mean": generic,
            "peak_mfe_12h_atr.conditional_q80": report(0.989),
        },
    )

    q80 = gate["component_quality"]["conditional_q80"]
    assert q80["status"] == "WITHHELD_MISCALIBRATED_Q80"
    assert q80["failing_slices"][0]["coverage"] == pytest.approx(0.989)
    assert (
        "pred_peak_mfe_if_hit_q80_atr"
        not in gate["research_ablation"]["prediction_columns"]
    )
    assert gate["production_promotion"]["production_ready"] is False


def test_representation_report_emits_zero_support_slice_explicitly() -> None:
    labels = _labels()
    labels.loc[labels["side"].eq("long"), "gmm_representation_available"] = 1.0
    result = _role_result(
        MEANINGFUL_EVENT_ROLE,
        [0.2, 0.4, 0.6, 0.8, 0.3, 0.5, 0.7, 0.9],
        task="binary",
    )
    report = _representation_role_metrics(labels, result)

    assert report["long"]["missing"]["canonical_rows"] == 0
    assert report["long"]["missing"]["metric_rows"] == 0
    assert report["long"]["missing"]["status"] == "not_evaluable_zero_missing_support"
    assert report["short"]["missing"]["metric_rows"] == 2


def test_quantile_representation_report_computes_coverage_with_support() -> None:
    labels = _labels()
    labels.loc[labels["side"].eq("long"), "gmm_representation_available"] = 1.0
    result = _role_result(
        "peak_mfe_12h_atr.conditional_q80",
        [0.2, 0.4, 0.6, 0.8, 0.3, 0.5, 0.7, 0.9],
        task="quantile",
    )
    report = _representation_role_metrics(labels, result)

    assert report["long"]["available"]["metric_rows"] == 4
    assert "empirical_coverage_alpha_0_8" in report["long"]["available"]["metrics"]
    assert report["long"]["missing"]["metric_rows"] == 0


def test_timing_family_owns_the_bitwise_shared_12h_event_prediction() -> None:
    labels = _labels()
    families = {}
    for side_index, side in enumerate(("long", "short")):
        p12 = np.asarray([0.2, 0.4, 0.6, 0.8], dtype=np.float32) + 0.05 * side_index
        by_horizon = {
            2: p12 * 0.25,
            4: p12 * 0.50,
            8: p12 * 0.75,
            12: p12,
        }
        folds = [
            {
                **_side_state(side)["fold_provenance"][0],
                "model_sha256_by_horizon": {
                    str(hours): f"{side}-{hours}" for hours in (2, 4, 8, 12)
                },
            }
        ]
        families[side] = {
            "oof_predictions_by_horizon": by_horizon,
            "oof_fold_ids": np.zeros(4, dtype=np.int16),
            "oof_prediction_mask": np.ones(4, dtype=bool),
            "oof_metrics_by_horizon": {
                hours: {"metric_support": 4} for hours in (2, 4, 8, 12)
            },
            "fold_provenance": folds,
            "side_models": {
                side: {
                    "best_params": {"objective": "binary"},
                    "hpo": {"trial_count": 1},
                    "selected_features_by_horizon": {
                        hours: [f"f_{hours}"] for hours in (2, 4, 8, 12)
                    },
                    "final_refit_contract": {
                        "model_sha256_by_horizon": {
                            str(hours): f"final-{side}-{hours}"
                            for hours in (2, 4, 8, 12)
                        }
                    },
                }
            },
        }

    results = _timing_role_results(labels, families_by_side=families)
    event = results[MEANINGFUL_EVENT_ROLE]["oof_predictions"]
    for side in ("long", "short"):
        rows = labels["side"].eq(side).to_numpy()
        np.testing.assert_array_equal(
            event[rows], families[side]["oof_predictions_by_horizon"][12]
        )
    assert (
        results[MEANINGFUL_EVENT_ROLE]["hpo_group_id"] == "timing_cdf_shared_2_4_8_12"
    )
