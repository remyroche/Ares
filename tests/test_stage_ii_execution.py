from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.stage_ii_execution import (
    StageIIExecutionError,
    build_stage_ii_ledger,
    make_side_local_strict_meta_predictor,
    make_locked_stage_ii_scorer,
    validate_enriched_ledger_manifest,
)
from extreme_price_movements.stage_ii_meta_archetypes import StageIIMetaArchetypeConfig


def _source(rows: int = 24) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    decision = pd.date_range("2024-02-02", periods=rows, freq="h", tz="UTC")
    side = np.where(np.arange(rows) % 2, "long", "short")
    source = pd.DataFrame({
        "candidate_id": [f"c-{value}" for value in range(rows)],
        "symbol": np.where(np.arange(rows) % 3, "BTC", "ETH"),
        "side_name": side, "decision_ts": decision,
        "label_available_ts": decision + pd.Timedelta(hours=13),
        "exact_gross_bps": np.full(rows, 130.0), "exact_net_bps": np.full(rows, 30.0),
        "base_strict_oof_available": True, "base_fold_id": 0,
        "r3_p_adverse": .2, "r3_p_weak": .3, "r3_p_clear": .5,
        "prequential_base_expected_net_bps": 10.0,
        "value_map__value_map_max_label_available_ts": decision - pd.Timedelta(hours=1),
    })
    provenance = pd.DataFrame([
        {"side": name, "layer": "base_r3", "fold_id": 0,
         "train_max_label_available_ts": "2024-02-01T00:00:00Z",
         "validation_start_ts": "2024-02-02T00:00:00Z",
         "validation_end_ts": "2024-02-03T00:00:00Z", "strict_prior_resolved": True}
        for name in ("long", "short")
    ])
    enriched = source.loc[:, ["candidate_id", "side_name", "decision_ts"]].copy()
    enriched["meta_context"] = np.linspace(-1, 1, rows)
    enriched["causal_context"] = np.linspace(2, 3, rows)
    enriched["realised_path"] = np.linspace(-.5, .5, rows)
    return source, provenance, enriched


def test_stage_ii_ledger_uses_direct_strict_base_handoff_and_namespaces_side_folds() -> None:
    source, provenance, enriched = _source()
    ledger, catalogue = build_stage_ii_ledger(
        stage_i_predictions=source, stage_i_fold_provenance=provenance,
        enriched_ledger=enriched,
        required_enriched_columns=("meta_context", "causal_context", "realised_path"),
    )
    assert len(ledger) == len(source)
    assert ledger.r3_is_strict_oof.all()
    assert ledger.r3_source_side.eq(ledger.side_name).all()
    assert ledger.base_map_source_side.eq(ledger.side_name).all()
    assert (ledger.signal_close_ts + pd.Timedelta(hours=1)).eq(ledger.decision_ts).all()
    assert len(catalogue) == 2
    assert ledger.r3_oof_fold_id.nunique() == 2
    assert all(item["train_max_label_available_ts"] < item["validation_start_ts"] for item in catalogue)


def test_stage_ii_ledger_rejects_missing_enriched_path_or_context_coverage() -> None:
    source, provenance, enriched = _source()
    with pytest.raises(StageIIExecutionError, match="lacks required"):
        build_stage_ii_ledger(
            stage_i_predictions=source, stage_i_fold_provenance=provenance,
            enriched_ledger=enriched.drop(columns="realised_path"),
            required_enriched_columns=("meta_context", "realised_path"),
        )
    enriched.loc[0, "realised_path"] = np.nan
    with pytest.raises(StageIIExecutionError, match="must be finite"):
        build_stage_ii_ledger(
            stage_i_predictions=source, stage_i_fold_provenance=provenance,
            enriched_ledger=enriched,
            required_enriched_columns=("meta_context", "causal_context", "realised_path"),
        )


def test_stage_ii_meta_predictor_refuses_non_huber_or_noncanonical_side_contracts() -> None:
    with pytest.raises(StageIIExecutionError, match="Huber"):
        make_side_local_strict_meta_predictor(
            side_by_candidate_id={"c": "long"}, params={"objective": "binary"},
        )
    with pytest.raises(StageIIExecutionError, match="positive folds"):
        make_side_local_strict_meta_predictor(
            side_by_candidate_id={"c": "long"}, params={"objective": "huber"}, n_validation_folds=0,
        )


def test_enriched_manifest_binds_bytes_and_canonical_path_context_lineage(tmp_path) -> None:
    ledger = tmp_path / "enriched.parquet"
    _source()[2].to_parquet(ledger, index=False)
    from extreme_price_movements.stage_ii_execution import file_sha256
    manifest = {
        "schema": "stage_ii_enriched_path_context_ledger_v1",
        "ledger_sha256": file_sha256(ledger),
        "identity_columns": ["candidate_id", "symbol", "side_name", "signal_close_ts", "decision_ts", "label_available_ts"],
        "causal_columns": ["meta_context", "causal_context"],
        "path_descriptor_columns": ["realised_path"],
        "label_lineage": {"artifact_path": "canonical_labels", "artifact_sha256": "a", "identity_sha256": "b"},
        "context_lineage": {"artifact_path": "canonical_context", "artifact_sha256": "c", "identity_sha256": "d"},
    }
    validate_enriched_ledger_manifest(
        manifest, ledger_path=ledger, required_causal_columns=("meta_context",),
        required_path_columns=("realised_path",),
    )
    manifest["path_descriptor_columns"] = []
    with pytest.raises(StageIIExecutionError, match="omits"):
        validate_enriched_ledger_manifest(
            manifest, ledger_path=ledger, required_causal_columns=("meta_context",),
            required_path_columns=("realised_path",),
        )


def test_locked_scorer_fits_only_prior_windows_and_emits_raw_and_admission_fields() -> None:
    from types import SimpleNamespace

    rows = 160
    rng = np.random.default_rng(71)
    decision = pd.date_range("2024-01-01", periods=rows, freq="h", tz="UTC")
    side = np.where(np.arange(rows) % 2, "long", "short")
    context = rng.normal(size=rows).astype("float32")
    base = (context * 20).astype("float32")
    path = (context + rng.normal(0, .1, rows)).astype("float32")
    net = (base + path * 50 + rng.normal(0, 3, rows)).astype("float32")
    ledger = pd.DataFrame({
        "candidate_id": [f"locked-{i}" for i in range(rows)], "symbol": "BTC", "side_name": side,
        "signal_close_ts": decision - pd.Timedelta(hours=1), "decision_ts": decision,
        "label_available_ts": decision + pd.Timedelta(hours=12), "exact_net_bps": net,
        "exact_gross_bps": net + 100., "total_cost_bps": 100.,
        "prequential_base_expected_net_bps": base, "r3_p_adverse": .2,
        "r3_p_weak": .3, "r3_p_clear": .5, "r3_oof_fold_id": 0,
        "r3_fit_end_ts": decision - pd.Timedelta(days=1),
        "base_map_is_prequential": True, "base_map_source_side": side,
        "base_map_max_label_available_ts": decision - pd.Timedelta(hours=1),
        "context": context, "realised_path": path,
    })
    config = StageIIMetaArchetypeConfig(
        path_descriptor_cols=("realised_path",), components=3, min_side_rows=10,
        min_component_rows=4, min_train_rows=20, oof_folds=3,
    )
    scorer = make_locked_stage_ii_scorer(
        full_ledger=ledger, candidate_config=config,
        causal_feature_cols=("context", "prequential_base_expected_net_bps"),
        meta_feature_cols=("context",), selected_control_arm="both",
        meta_params={"objective": "huber", "n_estimators": 10, "learning_rate": .1, "verbosity": -1},
    )
    # The explicit 20-hour gap ensures every development label is resolved
    # before the locked period begins; a contiguous split must fail instead.
    history, development, evaluation = ledger.iloc[:60], ledger.iloc[60:100], ledger.iloc[120:]
    result = scorer({
        "history": history, "development": development,
        "evaluation_identity": evaluation.loc[:, ["candidate_id", "symbol", "signal_close_ts", "decision_ts", "side_name"]],
        "winner_manifest": SimpleNamespace(selected_discovery_candidate_id="path-k3", selected_control_arm="both"),
    })
    output = result.ledger
    assert len(output) == len(evaluation)
    assert output.meta_raw_predicted_residual_bps.notna().all()
    assert output.meta_reconstructed_expected_net_bps.notna().all()
    assert output.meta_causal_21d_side_admitted_ge_50bps.eq(
        output.meta_causal_21d_side_expected_net_bps.ge(50.).fillna(False)
    ).all()
