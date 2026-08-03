from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.run_canonical_economic_conversion_transition_head_ablation import (
    TARGETS,
    _classification_metrics,
    build_expanding_folds,
    fit_target_oof,
    prepare_population,
)


FEATURES = ("context__base_oof_score__mean", "context__range_24h_pct__mean")


def _inputs(days: int = 18) -> tuple[pd.DataFrame, pd.DataFrame]:
    context_rows: list[dict[str, object]] = []
    label_rows: list[dict[str, object]] = []
    origin = pd.Timestamp("2025-02-01T00:00:00Z")
    for day in range(days):
        anchor = origin + pd.Timedelta(days=day)
        for side_index, side in enumerate(("long", "short")):
            for decile in range(10):
                value = float(day + decile - 5 + side_index)
                context_rows.append(
                    {
                        "cohort_anchor_utc": anchor,
                        "side_name": side,
                        "frozen_base_score_decile": decile,
                        FEATURES[0]: value,
                        FEATURES[1]: value * 0.1,
                    }
                )
                for horizon in (12, 3):
                    row: dict[str, object] = {
                        "cohort_anchor_utc": anchor,
                        "side_name": side,
                        "frozen_base_score_decile": decile,
                        "horizon_hours": horizon,
                        "horizon_role": "primary" if horizon == 12 else "auxiliary",
                        "before_global_hour_complete_flag": True,
                        "after_global_hour_complete_flag": True,
                        "before_candidate_support": 8,
                        "after_candidate_support": 8,
                        "before_target_available_utc": anchor + pd.Timedelta(hours=12),
                        "after_target_available_utc": anchor + pd.Timedelta(hours=24),
                        "before_favorable_net_missing_support_flag": False,
                        "after_favorable_net_missing_support_flag": False,
                        "before_adverse_loss_missing_support_flag": False,
                        "after_adverse_loss_missing_support_flag": False,
                        "delta_opportunity_probability_0bps": value / 100.0,
                        "delta_conditional_favorable_net_robust_mean": value / 200.0,
                        "delta_conditional_adverse_loss_robust_mean": -value / 300.0,
                        "delta_exit_mixture_expected_net": value / 400.0,
                        "delta_direct_mean_net": value / 500.0,
                    }
                    label_rows.append(row)
    labels = pd.DataFrame.from_records(label_rows)
    # This label must stay absent for the favourable-payoff component rather
    # than becoming a zero target or a model prediction.
    labels.loc[0, "after_favorable_net_missing_support_flag"] = True
    labels.loc[0, "delta_conditional_favorable_net_robust_mean"] = np.nan
    return pd.DataFrame.from_records(context_rows), labels


def test_expanding_folds_purge_on_actual_target_availability_and_preserve_missing_conditionals() -> None:
    context, labels = _inputs()
    population = prepare_population(context, labels, FEATURES)
    missing = population.loc[
        population["favorable_payoff_robust_mean__target_status"].eq("missing_conditional_support")
    ]
    assert len(missing) == 1
    assert not missing["favorable_payoff_robust_mean__target_valid"].iloc[0]
    assert np.isnan(missing["delta_conditional_favorable_net_robust_mean"].iloc[0])

    folds = build_expanding_folds(population, min_train_days=5, validation_days=4)
    direct = population.loc[population["horizon_hours"].eq(12)].copy()
    predictions, per_fold = fit_target_oof(
        direct,
        target=next(item for item in TARGETS if item.name == "direct_mean_net"),
        features=FEATURES,
        folds=folds,
        min_train_rows=100_000,  # test the deterministic constant fallback only
        fit_budget_rows=100_000,
        random_state=7,
        threads=1,
    )
    assert not predictions.empty
    scored = predictions.loc[predictions["target_valid"].astype(bool)]
    assert scored["fit_status"].str.startswith("constant_fallback").all()
    assert (scored["training_max_after_target_available_utc"] < scored["validation_start_utc"]).all()
    assert {"model_regression_mae", "model_regression_rank_ic", "model_sign_auc", "model_sign_ap", "model_sign_brier", "model_sign_calibration_ece_10"}.issubset(per_fold.columns)


def test_sign_metrics_report_probability_quality_and_calibration() -> None:
    metrics = _classification_metrics(
        np.asarray([0, 0, 1, 1], dtype=np.int8),
        np.asarray([0.1, 0.2, 0.8, 0.9], dtype=float),
    )
    assert metrics["auc"] == 1.0
    assert metrics["ap"] == 1.0
    assert metrics["brier"] < 0.03
    assert metrics["calibration_ece_10"] >= 0.0
