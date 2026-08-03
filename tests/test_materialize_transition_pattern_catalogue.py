import json
from pathlib import Path

import numpy as np
import pandas as pd

from extreme_price_movements.transition_pattern_catalogue import TransitionPatternConfig
from extreme_price_movements.transition_pattern_models import TransitionMorphologyConfig
from scripts.materialize_transition_pattern_catalogue import (
    _event_grouped_purged_folds,
    materialize_transition_pattern_catalogue,
)


def _ledger(root: Path) -> Path:
    ledger = root / "ledger"
    ledger.mkdir()
    source = pd.date_range("2024-01-01", periods=360, freq="h", tz="UTC")
    state = np.zeros(len(source), dtype=np.int16)
    anchors = [80, 120, 160, 200, 240, 280]
    records = []
    current = 0
    for number, anchor_index in enumerate(anchors):
        destination = 1 - current
        state[anchor_index:] = destination
        anchor = source[anchor_index]
        records.append(
            {
                "event_id": f"event_{number}",
                "source_segment_id": 1,
                "anchor_source_utc": anchor,
                "anchor_decision_utc": anchor + pd.Timedelta(hours=1),
                "transition_start_utc": anchor,
                "transition_end_utc": anchor + pd.Timedelta(hours=3),
                "target_available_utc": anchor + pd.Timedelta(hours=13),
                "source_state": current,
                "destination_state": destination,
                "transition_archetype": f"state_{current}_to_state_{destination}",
                "label_contract": "synthetic",
            }
        )
        current = destination
    hourly = pd.DataFrame(
        {
            "source_utc": source,
            "execution_decision_utc": source + pd.Timedelta(hours=1),
            "calendar_segment_id": 0,
            "source_segment_id": 1,
            "target__pooled_state": state,
            "state_context__current_state": state,
            "target__phase": "stable",
            "target__event_id": None,
            "target__transition_active": 0,
            "observable_breadth": np.sin(np.arange(len(source)) / 11.0),
            "observable_volatility": np.cos(np.arange(len(source)) / 9.0),
            "target__forbidden_future": 1.0,
        }
    )
    hourly.to_parquet(ledger / "hourly_state_calendar.parquet", index=False)
    pd.DataFrame(records).to_parquet(ledger / "transition_episode_ledger.parquet", index=False)
    (ledger / "manifest.json").write_text(json.dumps({"schema": "synthetic_ledger"}), encoding="utf-8")
    return ledger


def test_event_grouped_purge_keeps_event_identity_disjoint() -> None:
    frame = pd.DataFrame(
        {
            "event_id": ["a", "b", "c", "d"],
            "anchor_source_utc": pd.date_range("2025-01-01", periods=4, freq="2D", tz="UTC"),
        }
    )
    folds, plan = _event_grouped_purged_folds(frame, n_splits=2, purge_hours=24)
    assert len(folds) == 2
    for fold, (train, validation) in enumerate(folds):
        assert not set(frame.iloc[train]["event_id"]).intersection(frame.iloc[validation]["event_id"])
        assert set(plan.loc[(plan["fold"] == fold) & (plan["role"] == "validation"), "event_id"]) == set(frame.iloc[validation]["event_id"])


def test_materializes_separate_causal_and_descriptive_transition_pattern_outputs(tmp_path: Path) -> None:
    output = tmp_path / "output"
    report = materialize_transition_pattern_catalogue(
        ledger_dir=_ledger(tmp_path),
        output_dir=output,
        pattern_config=TransitionPatternConfig(precondition_hours=48, sequence_horizons_hours=(3, 6, 12, 24, 48)),
        morphology_config=TransitionMorphologyConfig(n_components=2, embedding_components=2, min_component_events=2),
        n_splits=3,
        purge_hours=12,
    )
    labels = pd.read_parquet(output / "adaptive_phase_labels.parquet")
    sequence = pd.read_parquet(output / "event_preonset_sequences.parquet")
    morphology = pd.read_parquet(output / "morphology_oof.parquet")
    stable = pd.read_parquet(output / "stable_transition_oof.parquet")
    brl = pd.read_parquet(output / "stable_transition_brl_oof.parquet")
    metrics = pd.read_csv(output / "oof_diagnostic_metrics.csv")
    manifest = json.loads((output / "manifest.json").read_text())
    assert "target__pattern_phase" in labels
    assert "target__forbidden_future" not in report["field_contract"]["causal_sequence_fields"]
    assert all(name.startswith("sequence__") for name in report["field_contract"]["causal_sequence_fields"])
    assert {"source_state", "destination_state"}.issubset(sequence.columns)
    assert "source_state" not in report["field_contract"]["causal_sequence_fields"]
    assert manifest["research_only"] and not manifest["promotion_eligible"]
    assert (output / "morphology_fold_plan.parquet").exists()
    assert "stable_vs_transition" in set(metrics["task"])
    assert "stable_vs_transition_brl" in set(metrics["task"])
    assert (output / "stable_transition_brl_rule_lists.json").exists()
    assert stable.loc[:, ["event_id", "oof_fold"]].sort_values(["event_id", "oof_fold"]).reset_index(drop=True).equals(
        brl.loc[:, ["event_id", "oof_fold"]].sort_values(["event_id", "oof_fold"]).reset_index(drop=True)
    )
    assert brl["brl__p_transition"].between(0.0, 1.0).all()
    if len(morphology):
        posterior = morphology.filter(like="morphology__posterior_")
        assert np.allclose(posterior.sum(axis=1), 1.0)
        assert morphology["morphology__abstained"].isin([0, 1]).all()
