from __future__ import annotations

import hashlib
import json
from argparse import Namespace
from pathlib import Path

import pandas as pd
import pytest

from scripts.diagnose_historical_exact1m_recurrence import (
    _join_transition_dataset,
    _stable_top_k,
    run,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_bundle(tmp_path: Path, *, complete: bool = True) -> tuple[Path, Path, Path]:
    source_root = tmp_path / "source"
    source_root.mkdir()
    rows = []
    for index, (ts, side, score, candidate_id) in enumerate(
        [
            ("2022-12-01T00:00:00Z", "long", 0.80, "d"),
            ("2022-12-01T00:00:00Z", "short", 0.80, "c"),
            ("2022-12-01T00:00:00Z", "long", 0.30, "b"),
            ("2022-12-01T00:00:00Z", "short", 0.10, "a"),
            ("2023-01-01T00:00:00Z", "long", 0.90, "h"),
            ("2023-01-01T00:00:00Z", "short", 0.90, "g"),
            ("2023-01-01T00:00:00Z", "long", 0.40, "f"),
            ("2023-01-01T00:00:00Z", "short", 0.20, "e"),
        ]
    ):
        rows.append(
            {
                "__ts__": pd.Timestamp(ts),
                "__symbol__": f"S{index}/USD:USD",
                "side_name": side,
                "base_score": score,
                "preentry_transition_score": float(index),
                # This deliberately forbidden source outcome must not be
                # included by the pre-entry screen.
                "clean_exec": float(index % 2),
                "selected_for_monitor": True,
                "evidence_scope": "frozen_backcast_diagnostic",
            }
        )
    source = source_root / "candidates_2022_2023.parquet"
    pd.DataFrame(rows).to_parquet(source, index=False)

    stage = tmp_path / "stage"
    stage.mkdir()
    staged = []
    for index, row in enumerate(rows):
        signal = pd.Timestamp(row["__ts__"])
        staged.append(
            {
                "candidate_id": ["d", "c", "b", "a", "h", "g", "f", "e"][index],
                "source_shard_path": str(source.resolve()),
                "source_shard_sha256": _sha(source),
                "source_row_number": index,
                "signal_timestamp": signal,
                "decision_timestamp": signal + pd.Timedelta(hours=1),
                "symbol": row["__symbol__"],
                "side_name": row["side_name"],
                "base_score": row["base_score"],
            }
        )
    staged_path = stage / "staged_candidates.parquet"
    pd.DataFrame(staged).to_parquet(staged_path, index=False)
    stage_manifest = {
        "schema": "historical_backcast_exact1m_request_stage_v2",
        "evidence_scope": "frozen_backcast_diagnostic_not_oof",
        "promotion_eligible": False,
        "execution_parity_claim": False,
        "path_horizon_minutes": 720,
        "selected_rows": len(staged),
        "sources": [{"path": str(source.resolve()), "sha256": _sha(source)}],
        "outputs": {
            "staged_candidates": {
                "path": str(staged_path.resolve()), "sha256": _sha(staged_path)
            }
        },
    }
    (stage / "manifest.json").write_text(json.dumps(stage_manifest), encoding="utf-8")

    labels_root = tmp_path / "labels"
    labels_root.mkdir()
    labels = []
    for index, row in enumerate(staged):
        decision = row["decision_timestamp"]
        labels.append(
            {
                "__ts__": row["signal_timestamp"], "__symbol__": row["symbol"],
                "side_name": row["side_name"], "candidate_id": row["candidate_id"],
                "__decision_ts__": decision,
                "__opportunity_occurred_12h__": int(index % 2 == 0),
                "__peak_mfe_atr_12h__": 1.0 + index,
                "__favorable_payoff_return_12h__": 0.01 * (index + 1),
                "__adverse_competing_risk_12h__": int(index % 3 == 0),
                "__timeout_outcome_12h__": int(index % 4 == 0),
                "__exit_conversion_loss_return_12h__": 0.001 * index,
                "__opportunity_scarcity_proxy_12h__": int(index % 2 != 0),
                "__exit_conversion_failure_proxy_12h__": int(index % 3 == 1),
                "__timeout_degradation_proxy_12h__": int(index % 4 == 0),
                "__adverse_payoff_expansion_proxy_12h__": int(index % 3 == 0),
                # Policy economics deliberately differ from physical path targets.
                "execution_gross_ev_12h": 0.02 - index * 0.002,
                "execution_cost_return": 0.005,
                "execution_net_ev_12h": 0.015 - index * 0.002,
                "execution_exit_reason": "timeout" if index % 2 else "trailing",
                "execution_exit_hour": 12.0 if index % 2 else 4.0,
            }
        )
    joined = labels_root / "joined_multitask_labels.parquet"
    pd.DataFrame(labels).to_parquet(joined, index=False)
    coverage = tmp_path / "coverage.json"
    coverage.write_text(
        json.dumps(
            {
                "schema": "historical_exact1m_candidate_coverage_v1",
                "status": "complete" if complete else "incomplete",
                "candidate_coverage_fraction": 1.0 if complete else 0.99,
                "complete_candidates": len(staged) if complete else len(staged) - 1,
            }
        ),
        encoding="utf-8",
    )
    labels_manifest = {
        "schema": "historical_backcast_exact1m_execution_path_labels_v1",
        "status": "materialized",
        "oof_status": "not_oof",
        "promotion_eligible": False,
        "execution_parity_claim": False,
        "rows": len(labels),
        "label_timing": {"path": "[decision, decision+12h)"},
        "source_separation": {"physical_path_labels": "physical", "policy_economics": "policy"},
        "outputs": {
            "joined_multitask_labels": {
                "path": str(joined.resolve()), "sha256": _sha(joined), "rows": len(labels)
            }
        },
        "sources": {"candidate_coverage_manifest": {"path": str(coverage.resolve()), "sha256": _sha(coverage)}},
    }
    (labels_root / "manifest.json").write_text(json.dumps(labels_manifest), encoding="utf-8")

    transition = tmp_path / "transition.parquet"
    pd.DataFrame(
        {
            "execution_decision_utc": pd.to_datetime(["2023-01-01T01:00:00Z"]),
            "transition_new__score": [0.7],
            "model_health_compact": [0.2],
            "target__transition_active": [1],
            "target__destination_state": ["deleveraging"],
            "target__phase": ["active"],
        }
    ).to_parquet(transition, index=False)
    return stage, labels_root, transition


def _args(stage: Path, labels: Path, transition: Path, output: Path) -> Namespace:
    return Namespace(
        stage_dir=stage, labels_root=labels, transition_dataset=transition,
        no_transition_dataset=False, context_columns="preentry_transition_score", output_dir=output,
    )


def test_stable_top_k_is_global_and_candidate_tie_broken() -> None:
    frame = pd.DataFrame(
        {"candidate_id": ["z", "a", "b", "c"], "base_score": [1.0, 1.0, 0.9, 0.8], "side_name": ["long", "short", "long", "short"]}
    )
    selected = _stable_top_k(frame, 0.25)
    assert selected["candidate_id"].tolist() == ["a"]


def test_runner_preserves_identity_source_separation_and_global_month_selection(tmp_path) -> None:
    stage, labels, transition = _write_bundle(tmp_path)
    output = tmp_path / "report"

    manifest = run(_args(stage, labels, transition, output))

    assert manifest["evidence_scope"] == "frozen_backcast_diagnostic_not_oof"
    assert manifest["promotion_eligible"] is False
    assert manifest["runner"]["sha256"] == _sha(
        Path("scripts/diagnose_historical_exact1m_recurrence.py")
    )
    assert manifest["transition_join"]["coverage_by_year"] == {"2022": 0.0, "2023": 1.0}
    assert (
        manifest["transition_join"]["2022_interpretation"]
        == "no_transition_dataset_coverage_explicitly_unobserved"
    )
    top = pd.read_csv(output / "monthly_global_topk_economics.csv")
    january = top.loc[(top["month"] == "2023-01") & (top["top_fraction"] == 0.10)].iloc[0]
    assert january["selected_rows"] == 1
    assert january["short_selected_rows"] == 1  # g wins the global, tied-score tail.
    bridge = pd.read_csv(output / "monthly_side_score_target_bridge.csv")
    assert "base_score_rank_ic__physical_opportunity" in bridge
    assert "base_score_rank_ic__policy_net" in bridge
    assert (output / "monthly_global_top10_decomposition.csv").exists()
    associations = pd.read_csv(output / "preentry_context_associations.csv")
    assert list(associations.columns)  # retained even when bounded support is too small.
    assert manifest["preentry_context"]["source_columns"] == ["preentry_transition_score"]
    assert "clean_exec" not in manifest["preentry_context"]["source_columns"]
    interactions = pd.read_csv(output / "transition_execution_component_interactions.csv")
    assert set(interactions["transition_active"].astype(str)) == {"1.0"}


def test_runner_refuses_incomplete_exact_labels_before_writing_output(tmp_path) -> None:
    stage, labels, transition = _write_bundle(tmp_path, complete=False)
    output = tmp_path / "report"

    with pytest.raises(ValueError, match="incomplete"):
        run(_args(stage, labels, transition, output))
    assert not output.exists()


def test_runner_rejects_tampered_frozen_source_manifest_hash(tmp_path) -> None:
    stage, labels, transition = _write_bundle(tmp_path)
    stage_manifest_path = stage / "manifest.json"
    payload = json.loads(stage_manifest_path.read_text())
    payload["sources"][0]["sha256"] = "0" * 64
    stage_manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="frozen candidate source shard"):
        run(_args(stage, labels, transition, tmp_path / "report"))


def test_runner_rejects_tampered_exact_label_manifest_hash(tmp_path) -> None:
    stage, labels, transition = _write_bundle(tmp_path)
    manifest_path = labels / "manifest.json"
    payload = json.loads(manifest_path.read_text())
    payload["outputs"]["joined_multitask_labels"]["sha256"] = "f" * 64
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="joined exact multitask labels"):
        run(_args(stage, labels, transition, tmp_path / "report"))


def test_runner_rejects_outcome_derived_requested_context(tmp_path) -> None:
    stage, labels, transition = _write_bundle(tmp_path)
    args = _args(stage, labels, transition, tmp_path / "report")
    args.context_columns = "clean_exec"

    with pytest.raises(ValueError, match="outcome-derived"):
        run(args)


def test_transition_panels_are_schema_identical_and_non_overlapping(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.parquet"
    second = tmp_path / "second.parquet"
    base = pd.DataFrame(
        {
            "execution_decision_utc": pd.to_datetime(
                ["2022-12-31 23:00:00+00:00"]
            ),
            "state_context__current_state": [1.0],
            "target__transition_active": [0],
            "target__destination_state": [1],
            "target__phase": ["stable"],
        }
    )
    base.to_parquet(first, index=False)
    base.assign(
        execution_decision_utc=pd.to_datetime(["2023-01-01 00:00:00+00:00"])
    ).to_parquet(second, index=False)
    frame = pd.DataFrame(
        {
            "execution_decision_utc": pd.to_datetime(
                [
                    "2022-12-31 23:00:00+00:00",
                    "2023-01-01 00:00:00+00:00",
                ]
            )
        }
    )

    joined, report, context = _join_transition_dataset(frame, [first, second])
    assert report["coverage"] == 1.0
    assert len(report["sources"]) == 2
    assert (
        report["2022_interpretation"]
        == "fully_covered_by_supplied_transition_panel"
    )
    assert context == ["transition_ctx__state_context__current_state"]
    assert joined["transition_target_diagnostic__transition_active"].notna().all()

    base.to_parquet(second, index=False)
    with pytest.raises(ValueError, match="duplicate decision timestamps"):
        _join_transition_dataset(frame, [first, second])

    base.assign(extra=[1.0]).to_parquet(second, index=False)
    with pytest.raises(ValueError, match="exact column contract"):
        _join_transition_dataset(frame, [first, second])
