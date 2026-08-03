from __future__ import annotations

import numpy as np
import pandas as pd

from extreme_price_movements.execution_ev_regime_diagnosis import (
    RegimeDiagnosisConfig,
    build_regime_diagnosis_splits,
    evaluate_regime_diagnosis,
    feature_regime_diagnostics,
    split_audit_frame,
    validate_regime_diagnosis_input,
)


def _frame(rows_per_month: int = 12) -> pd.DataFrame:
    blocks: list[pd.DataFrame] = []
    for month in pd.period_range("2026-01", "2026-06", freq="M"):
        start = month.start_time.tz_localize("UTC")
        times = start + pd.to_timedelta(np.arange(rows_per_month), unit="h")
        signal = np.linspace(-1.0, 1.0, rows_per_month)
        blocks.append(
            pd.DataFrame(
                {
                    "execution_decision_utc": times,
                    "execution_label_end_utc": times + pd.Timedelta(hours=6),
                    "execution_net_ev_12h": signal * 0.01,
                    "score": signal,
                    "weight": np.arange(1, rows_per_month + 1, dtype=float),
                }
            )
        )
    return pd.concat(blocks, ignore_index=True)


def _config(**overrides: object) -> RegimeDiagnosisConfig:
    values: dict[str, object] = {
        "train_window_months": 2,
        "purge_hours": 12.0,
        "min_train_rows": 8,
        "top_k_fraction": 0.25,
        "max_periods": None,
    }
    values.update(overrides)
    return RegimeDiagnosisConfig(**values)


def test_forward_controls_only_use_resolved_past_and_reverse_is_non_oos() -> None:
    config = _config()
    work = validate_regime_diagnosis_input(_frame(), ["score"], config=config)
    splits = build_regime_diagnosis_splits(work, config=config)
    forward = next(
        split
        for split in splits
        if split.mode == "forward_rolling" and split.evaluation_month == "2026-03"
    )
    reverse = next(
        split
        for split in splits
        if (
            split.mode == "reversed_month_diagnostic"
            and split.evaluation_month == "2026-03"
        )
    )
    forward_train = work.iloc[forward.train_positions]
    reverse_train = work.iloc[reverse.train_positions]
    assert (
        forward_train["execution_decision_utc"]
        < forward.evaluation_start - pd.Timedelta(hours=12)
    ).all()
    assert (forward_train["execution_label_end_utc"] < forward.evaluation_start).all()
    assert (reverse_train["execution_decision_utc"] >= reverse.evaluation_end).all()
    assert len(reverse_train) == len(forward_train)
    assert reverse.is_oos is False
    assert reverse.evaluation_status == "diagnostic_non_oos_reversed_training"


def test_evaluation_uses_global_top_k_and_training_only_weight_hook() -> None:
    config = _config(top_k_fraction=0.25)
    seen_weight_months: list[str] = []

    def weights(train: pd.DataFrame) -> np.ndarray:
        seen_weight_months.extend(
            train["execution_decision_utc"]
            .dt.tz_localize(None)
            .dt.to_period("M")
            .astype(str)
            .unique()
            .tolist()
        )
        return train["weight"].to_numpy(dtype=float)

    def predict(
        train_x: pd.DataFrame,
        train_y: np.ndarray,
        evaluation_x: pd.DataFrame,
        sample_weight: np.ndarray | None,
    ) -> np.ndarray:
        assert sample_weight is not None
        assert len(sample_weight) == len(train_x)
        return evaluation_x["score"].to_numpy(dtype=float)

    result = evaluate_regime_diagnosis(
        _frame(rows_per_month=12),
        ["score"],
        predict,
        config=config,
        sample_weight_hook=weights,
    )
    row = result.metrics.loc[
        (result.metrics["mode"] == "forward_rolling")
        & (result.metrics["evaluation_month"] == "2026-03")
    ].iloc[0]
    # Twelve rows in the whole month at 25% means exactly three global trades,
    # not three rows per timestamp or per side.
    assert row["top_k_rows"] == 3
    assert row["selection_basis"] == "global_topk_across_full_evaluation_period"
    assert row["sample_weighting"] == "training_only_hook"
    forward_metrics = result.metrics.loc[
        result.metrics["mode"].eq("forward_rolling")
    ]
    assert (
        pd.to_datetime(forward_metrics["max_train_label_resolution_utc"], utc=True)
        < pd.to_datetime(forward_metrics["evaluation_start_utc"], utc=True)
    ).all()
    assert set(seen_weight_months) <= {
        "2026-01",
        "2026-02",
        "2026-03",
        "2026-04",
        "2026-05",
        "2026-06",
    }
    audit = split_audit_frame(
        [
            split
            for split in build_regime_diagnosis_splits(
                validate_regime_diagnosis_input(_frame(), ["score"], config=config),
                config=config,
            )
            if split.evaluation_month == "2026-03"
        ]
    )
    assert set(audit["selection_basis"]) == {
        "global_topk_across_full_evaluation_period"
    }
    assert audit.loc[
        audit["mode"] == "reversed_month_diagnostic", "promotion_eligible"
    ].eq(False).all()


def test_forward_control_rejects_unresolved_boundary_rows() -> None:
    frame = _frame()
    boundary = (
        frame["execution_decision_utc"]
        .dt.tz_localize(None)
        .dt.to_period("M")
        .astype(str)
        .eq("2026-02")
    )
    frame.loc[boundary, "execution_label_end_utc"] = pd.Timestamp(
        "2026-03-02", tz="UTC"
    )
    config = _config()
    work = validate_regime_diagnosis_input(frame, ["score"], config=config)
    forward = next(
        split
        for split in build_regime_diagnosis_splits(work, config=config)
        if split.mode == "forward_rolling" and split.evaluation_month == "2026-03"
    )
    train = work.iloc[forward.train_positions]
    assert not (
        train["execution_decision_utc"]
        .dt.tz_localize(None)
        .dt.to_period("M")
        .astype(str)
        .eq("2026-02")
        .any()
    )


def test_feature_diagnostics_exposes_monthly_shift_and_sign_flip() -> None:
    frame = _frame(rows_per_month=12)
    month = (
        frame["execution_decision_utc"]
        .dt.tz_localize(None)
        .dt.to_period("M")
        .astype(str)
    )
    frame.loc[month.eq("2026-06"), "score"] *= -1.0
    diagnostics = feature_regime_diagnostics(
        frame, ["score"], config=_config()
    )
    june = diagnostics.loc[diagnostics["month"].eq("2026-06")].iloc[0]
    assert bool(june["target_spearman_sign_flip"])
    assert june["target_spearman"] < 0.0
