from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_meta_handoff_parity import (
    audit_feature_contract,
    audit_prior_provenance,
    compare_meta_handoff,
    rescore_observed_meta_matrix,
)

FEATURES = ["rel_rankband_edge", "rel_marginband_timeout_rate"]


def _frame(*, base: float, meta: float, first: float = 0.5, second: float = 0.1) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "__ts__": [pd.Timestamp("2026-07-10 09:00:00", tz="UTC")],
            "__symbol__": ["AAVE/USD:USD"],
            "side_name": ["long"],
            "score_base": [base],
            "score_meta_base_soft_label": [meta],
            "rel_rankband_edge": [first],
            "rel_marginband_timeout_rate": [second],
        }
    )


def _observed(*, base: float, meta: float, first: float = 0.5, second: float = 0.1) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "signal_bar_ts": [pd.Timestamp("2026-07-10 09:00:00", tz="UTC")],
            "symbol": ["AAVE/USD:USD"],
            "side": ["long"],
            "base_pred": [base],
            "meta_pred": [meta],
            "meta_model_feature_values_json": [
                json.dumps(
                    {
                        "rel_rankband_edge": first,
                        "rel_marginband_timeout_rate": second,
                    }
                )
            ],
        }
    )


def test_exact_historical_anchor_has_equal_meta_inputs_and_score() -> None:
    detail, summary = compare_meta_handoff(
        _frame(base=0.56, meta=0.82),
        _observed(base=0.56, meta=0.82),
        feature_names=FEATURES,
    )

    assert summary["base_parity_rows"] == 1
    assert summary["post_base_meta_input_or_score_mismatch_rows"] == 0
    assert summary["first_post_base_divergence"] == "none"
    assert detail.loc[0, "shared_feature_count"] == 2
    assert detail.loc[0, "feature_max_abs_delta"] == 0.0


def test_upstream_base_drift_is_not_reported_as_meta_handoff_failure() -> None:
    _, summary = compare_meta_handoff(
        _frame(base=0.56, meta=0.82),
        _observed(base=0.57, meta=0.30),
        feature_names=FEATURES,
    )

    assert summary["base_parity_rows"] == 0
    assert summary["base_mismatch_rows"] == 1
    assert summary["post_base_meta_input_or_score_mismatch_rows"] == 0


def test_first_meta_input_divergence_is_named_on_base_equal_row() -> None:
    detail, summary = compare_meta_handoff(
        _frame(base=0.56, meta=0.82),
        _observed(base=0.56, meta=0.30, first=0.7),
        feature_names=FEATURES,
    )

    assert summary["post_base_meta_input_or_score_mismatch_rows"] == 1
    assert summary["first_post_base_divergence"] == "meta_input:rel_rankband_edge"
    assert detail.loc[0, "feature_mismatch_count"] == 1


def test_missing_reference_feature_is_a_meta_contract_failure() -> None:
    reference = _frame(base=0.56, meta=0.82).drop(columns=[FEATURES[1]])
    detail, summary = compare_meta_handoff(
        reference,
        _observed(base=0.56, meta=0.82),
        feature_names=FEATURES,
    )

    assert summary["post_base_meta_input_or_score_mismatch_rows"] == 1
    assert summary["first_post_base_divergence"] == (
        "meta_input:rel_marginband_timeout_rate"
    )
    assert detail.loc[0, "reference_contract_missing_count"] == 1
    assert detail.loc[0, "feature_contract_mismatch_count"] == 1


def test_prior_provenance_proves_exact_causal_cutoff(tmp_path: Path) -> None:
    source = tmp_path / "scored.parquet"
    pd.DataFrame(
        {
            "__ts__": pd.to_datetime(
                ["2026-06-30 22:00:00Z", "2026-06-30 23:00:00Z", "2026-07-01 00:00:00Z"]
            ),
            "__symbol__": ["A", "B", "C"],
            "side_name": ["long", "short", "long"],
            "selected_top30": [True, True, True],
        }
    ).to_parquet(source, index=False)
    prior = tmp_path / "prior.json"
    prior.write_text(
        json.dumps(
            {
                "schema": "s52_meta_reliability_priors_v1",
                "rows": 2,
                "selected_col": "selected_top30",
                "groups": {},
                "side_arch_priors": {},
                "source": {
                    "scored_ledger": str(source),
                    "train_end_exclusive": "2026-07-01T00:00:00Z",
                },
            }
        ),
        encoding="utf-8",
    )

    result = audit_prior_provenance(prior, repo_root=tmp_path)

    assert result["fit_rows"] == 2
    assert result["fit_unique_rows"] == 2
    assert result["excluded_future_rows"] == 1
    assert result["payload_row_count_matches"]
    assert result["canonical_keys_unique"]
    assert result["causal_cutoff_pass"]


def test_feature_contract_audit_separates_handoff_context_from_selected_inputs() -> None:
    audit = audit_feature_contract(
        ["rel_rankband_edge", "meta_sel_ood_abs_z_max", "gmm_prob_0"],
        reference_columns=["score_base", "base_margin_to_cutoff", "rel_rankband_edge"],
    )

    assert audit["base_score_rank_margin_available_in_handoff"] == [
        "score_base",
        "base_margin_to_cutoff",
    ]
    assert audit["base_score_rank_margin_selected_by_meta_model"] == []
    assert audit["reliability_prior_features"] == ["rel_rankband_edge"]
    assert audit["ood_features"] == ["meta_sel_ood_abs_z_max"]
    assert audit["ae_gmm_latent_features"] == ["gmm_prob_0"]


def test_captured_full_matrix_reproduces_frozen_meta_score() -> None:
    class SumModel:
        def predict(self, matrix: np.ndarray) -> np.ndarray:
            return matrix.sum(axis=1)

    observed = _observed(base=0.56, meta=0.6, first=0.5, second=0.1)
    detail, summary = rescore_observed_meta_matrix(
        observed,
        feature_names=FEATURES,
        model=SumModel(),
    )

    assert summary["complete_matrix_rows"] == 1
    assert summary["mismatch_rows"] == 0
    assert summary["max_abs_delta"] < 1e-12
    assert detail.loc[0, "meta_score_rescored"] == 0.6
