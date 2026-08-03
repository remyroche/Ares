from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

SCRIPT = Path(__file__).parents[1] / "scripts/materialize_packb_outer_oof_population.py"
SPEC = importlib.util.spec_from_file_location("packb_outer_population", SCRIPT)
assert SPEC and SPEC.loader
materializer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(materializer)


def test_final_outer_fold_can_extend_without_changing_prior_folds() -> None:
    extended = materializer.resolved_outer_folds("2026-07-21T00:00:00Z")
    assert extended[:-1] == materializer.OUTER_FOLDS[:-1]
    assert extended[-1][:2] == materializer.OUTER_FOLDS[-1][:2]
    assert extended[-1][2] == pd.Timestamp("2026-07-21T00:00:00Z")


def test_final_outer_fold_extension_cannot_shorten_calendar() -> None:
    with pytest.raises(
        materializer.OuterPopulationMaterializationError,
        match="cannot shorten",
    ):
        materializer.resolved_outer_folds("2026-07-10T00:00:00Z")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _decisions(path: Path) -> None:
    _write_json(
        path,
        {
            "schema_version": "full_pipeline_decisions_v1",
            "status": "LOCKED_BEFORE_NEW_TRAINING",
            "decisions": {
                "DEC-09": {
                    "feature_selection_hpo_resolution_cutoff_utc": (
                        "2026-03-01T00:00:00Z"
                    ),
                    "decision_timestamp": "signal_timestamp + 1 hour",
                    "outer_folds": [
                        list(item) for item in materializer.inner_population.OUTER_FOLDS
                    ],
                    "packb_pre_march_inner_calendar": {
                        "ae_gmm_reference_signal_interval": [
                            "2025-01-01T00:00:00Z",
                            "2025-11-01T00:00:00Z",
                        ],
                        "feature_selection_validation_interval": [
                            "2025-11-01T00:00:00Z",
                            "2025-12-01T00:00:00Z",
                        ],
                        "hpo_validation_intervals": [
                            ["2025-12-01T00:00:00Z", "2026-01-01T00:00:00Z"],
                            ["2026-01-01T00:00:00Z", "2026-02-01T00:00:00Z"],
                            ["2026-02-01T00:00:00Z", "2026-03-01T00:00:00Z"],
                        ],
                    },
                }
            },
        },
    )


def _source(tmp_path: Path) -> tuple[Path, Path]:
    labels = tmp_path / "labels"
    labels.mkdir(parents=True)
    signals = pd.to_datetime(
        [
            "2026-03-30T22:00:00Z",
            "2026-03-30T23:00:00Z",
            "2026-04-01T00:00:00Z",
            "2026-04-30T23:00:00Z",
            "2026-05-01T00:00:00Z",
            "2026-06-01T00:00:00Z",
            "2026-07-01T00:00:00Z",
            "2026-07-10T23:00:00Z",
            "2026-07-11T00:00:00Z",
        ],
        utc=True,
    )
    inventory = []
    for side in materializer.CANONICAL_SIDES:
        frame = pd.DataFrame(
            {
                "candidate_id": [f"{side}-{index}" for index in range(len(signals))],
                "side_name": side,
                "__ts__": signals,
                "__signal_ts__": signals,
                "__decision_ts__": signals + pd.Timedelta(hours=1),
                "__symbol__": "BTCUSDT",
            }
        )
        name = f"{side}.parquet"
        frame.to_parquet(labels / name, index=False)
        inventory.append(
            {
                "file": name,
                "rows": len(frame),
                "expected_current_rows": len(frame),
            }
        )
    pd.DataFrame(
        {
            "candidate_id": ["excluded"],
            "side_name": ["short"],
            "__ts__": [pd.Timestamp("2025-01-01T00:00:00Z")],
            "__decision_ts__": [pd.Timestamp("2025-01-01T01:00:00Z")],
        }
    ).to_parquet(labels / "train_global_short_7.parquet", index=False)
    audit = labels / "audit.json"
    total = sum(item["rows"] for item in inventory)
    _write_json(
        audit,
        {
            "schema": "packb_current_canonical_label_inventory_audit_v1",
            "status": "PASS",
            "per_file": inventory,
            "inventory": {
                "canonical_monthly_files": len(inventory),
                "excluded_unlisted_monolithic_files": ["train_global_short_7.parquet"],
                "expected_current_rows": total,
            },
            "totals": {"rows": total},
        },
    )
    return labels, audit


def test_outer_boundaries_are_strict_and_half_open() -> None:
    start = pd.Timestamp("2026-04-01T00:00:00Z")
    end = pd.Timestamp("2026-05-01T00:00:00Z")
    signal = pd.Series(
        pd.to_datetime(
            [
                "2026-03-30T22:00:00Z",
                "2026-03-30T23:00:00Z",
                "2026-04-01T00:00:00Z",
                "2026-04-30T23:00:00Z",
                "2026-05-01T00:00:00Z",
            ],
            utc=True,
        )
    )
    decision = signal + pd.Timedelta(hours=1)
    assert materializer.strict_outer_train_mask(signal, decision, start).tolist() == [
        True,
        False,
        False,
        False,
        False,
    ]
    assert materializer.strict_outer_validation_mask(
        signal, decision, start, end
    ).tolist() == [False, False, True, True, False]


def test_materializes_expanding_side_local_disjoint_folds(tmp_path: Path) -> None:
    decisions = tmp_path / "decisions.json"
    _decisions(decisions)
    labels, audit = _source(tmp_path)
    output = tmp_path / "outer"
    manifest = materializer.materialize(
        decisions_path=decisions,
        labels_dir=labels,
        causal_audit_path=audit,
        output_dir=output,
        batch_rows=2,
    )
    assert manifest["status"] == "MATERIALIZED_IMMUTABLE"
    for side in materializer.CANONICAL_SIDES:
        validation_ids: set[str] = set()
        train_counts = []
        for name, _, _ in materializer.OUTER_FOLDS:
            train = pd.read_parquet(output / "folds" / name / side / "train.parquet")
            valid = pd.read_parquet(
                output / "folds" / name / side / "validation.parquet"
            )
            assert set(valid["side_name"]) == {side}
            assert validation_ids.isdisjoint(set(valid["candidate_id"]))
            validation_ids.update(valid["candidate_id"])
            train_counts.append(len(train))
        assert train_counts == sorted(train_counts)
        assert len(set(train_counts)) == len(train_counts)
        assert len(validation_ids) == 6


def test_contract_only_and_existing_output_fail_closed(tmp_path: Path) -> None:
    decisions = tmp_path / "decisions.json"
    _decisions(decisions)
    labels, audit = _source(tmp_path)
    output = tmp_path / "contract"
    result = materializer.materialize(
        decisions_path=decisions,
        labels_dir=labels,
        causal_audit_path=audit,
        output_dir=output,
        contract_only=True,
    )
    assert result["status"] == "CONTRACT_ONLY"
    assert not list(output.rglob("*.parquet"))
    with pytest.raises(FileExistsError):
        materializer.materialize(
            decisions_path=decisions,
            labels_dir=labels,
            causal_audit_path=audit,
            output_dir=output,
            contract_only=True,
        )
