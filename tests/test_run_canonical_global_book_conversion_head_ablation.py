from __future__ import annotations

import pandas as pd
import pytest

from scripts.run_canonical_global_book_conversion_head_ablation import (
    AUDIT_COLUMNS,
    TARGETS,
    _eligible_training_rows,
    _features,
    _prepare_population,
)


def test_feature_contract_allows_causal_execution_regime_names_only() -> None:
    manifest = {
        "global_book_feature_columns": [
            "context__regime_source_execution_quality_score__mean"
        ],
        "global_band_feature_columns": [
            "context__regime_source_execution_risk_score__mean"
        ],
    }
    assert _features(manifest, "book")
    assert _features(manifest, "band")
    manifest["global_book_feature_columns"] = [
        "context__execution_net_ev_12h"
    ]
    with pytest.raises(ValueError, match="prohibited"):
        _features(manifest, "book")


def test_book_population_parity_and_strict_training_availability() -> None:
    anchor = pd.Timestamp("2025-02-01T00:00:00Z")
    available = anchor + pd.Timedelta(hours=13)
    feature = "context__causal_feature"
    context_rows = []
    label_rows = []
    for index in range(2):
        row_anchor = anchor + pd.Timedelta(days=index)
        audit = {
            "before_global_hour_complete_flag": True,
            "after_global_hour_complete_flag": True,
            "before_target_available_utc": row_anchor,
            "after_target_available_utc": available
            + pd.Timedelta(days=index),
        }
        context_rows.append(
            {
                "cohort_anchor_utc": row_anchor,
                "horizon_hours": 12,
                "book_fraction": 0.1,
                **{
                    f"label_audit__{column}": audit[column]
                    for column in AUDIT_COLUMNS
                },
                feature: float(index),
            }
        )
        labels = {
            "cohort_anchor_utc": row_anchor,
            "horizon_hours": 12,
            "book_fraction": 0.1,
            "horizon_role": "primary",
            **audit,
            "before_selected_candidate_support": 10,
            "after_selected_candidate_support": 10,
        }
        for target in TARGETS:
            labels[target.book_column] = 0.1 + index
        label_rows.append(labels)
    population = _prepare_population(
        architecture="book",
        context=pd.DataFrame(context_rows),
        labels=pd.DataFrame(label_rows),
        features=[feature],
    )
    assert population[
        "conversion_residual__target_valid"
    ].all()
    train = _eligible_training_rows(
        population,
        valid_column="conversion_residual__target_valid",
        validation_start_utc=anchor + pd.Timedelta(days=1, hours=12),
    )
    assert len(train) == 1
    assert (
        train["after_target_available_utc"]
        < anchor + pd.Timedelta(days=1, hours=12)
    ).all()


def test_context_label_availability_mismatch_fails_closed() -> None:
    anchor = pd.Timestamp("2025-02-01T00:00:00Z")
    audit = {
        "before_global_hour_complete_flag": True,
        "after_global_hour_complete_flag": True,
        "before_target_available_utc": anchor,
        "after_target_available_utc": anchor + pd.Timedelta(hours=13),
    }
    context = pd.DataFrame(
        [
            {
                "cohort_anchor_utc": anchor,
                "horizon_hours": 12,
                "book_fraction": 0.1,
                **{
                    f"label_audit__{column}": audit[column]
                    for column in AUDIT_COLUMNS
                },
                "context__x": 1.0,
            }
        ]
    )
    labels = {
        "cohort_anchor_utc": anchor,
        "horizon_hours": 12,
        "book_fraction": 0.1,
        "horizon_role": "primary",
        **audit,
        "before_selected_candidate_support": 10,
        "after_selected_candidate_support": 10,
    }
    for target in TARGETS:
        labels[target.book_column] = 0.1
    labels["after_target_available_utc"] = anchor + pd.Timedelta(hours=14)
    with pytest.raises(ValueError, match="audit parity"):
        _prepare_population(
            architecture="book",
            context=context,
            labels=pd.DataFrame([labels]),
            features=["context__x"],
        )
