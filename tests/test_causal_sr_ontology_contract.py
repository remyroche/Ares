from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from extreme_price_movements.causal_sr_engine import CausalSREngine, PendingInteraction, SREngineConfig, Zone
from extreme_price_movements.inference.causal_sr_c1_state import (
    CausalSRC1AppendState,
    materialize_lva_zone_rows,
    score_c1_lva_target_free,
)


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("sr_ontology", ROOT / "scripts/run_causal_sr_ontology_ablation.py")
assert SPEC is not None and SPEC.loader is not None
SUBJECT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUBJECT)


def test_default_ontology_preserves_v1_contract():
    config = SREngineConfig()
    assert config.merge_radius_atr == 0.20
    assert config.touch_radius_atr == 0.15
    assert config.reset_distance_atr == 0.75
    assert config.reset_bars == 2
    assert config.reset_mode == "and"
    assert config.horizon_bars == 32
    assert config.speed_tau_bars == 8.0


def test_bounded_variants_are_ontology_only_and_valid():
    assert set(SUBJECT.VARIANTS) == {"S1_precise_levels", "S2_independent_retests", "S3_barrier_12h"}
    for variant in SUBJECT.VARIANTS.values():
        SREngineConfig(**{
            key: tuple(value) if key in {"reaction_barriers", "penetration_barriers"} else value
            for key, value in variant.items()
        })


def _bars(rows: int = 3_400) -> pd.DataFrame:
    index = pd.date_range("2026-01-01", periods=rows, freq="15min", tz="UTC")
    rng = np.random.default_rng(1729)
    close = 100.0 + np.cumsum(rng.normal(0.0, .20, rows))
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) + rng.uniform(.02, .25, rows)
    low = np.minimum(open_, close) - rng.uniform(.02, .25, rows)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close,
         "volume": rng.uniform(10.0, 100.0, rows)},
        index=index,
    )


def _snapshot_frame(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "__symbol__", "snapshot_ts", "target_kind", "target_id", "candidate_id",
        "support_available", "support_distance_atr", "support__shrunk_historical_strength",
        "resistance_available", "resistance_distance_atr", "resistance__shrunk_historical_strength",
    ]
    available = [column for column in columns if column in frame]
    return frame.loc[:, available].sort_values(["snapshot_ts", "candidate_id"], kind="stable").reset_index(drop=True)


def test_sr_checkpoint_append_matches_single_causal_replay(tmp_path: Path):
    """A checkpoint may not change a later target-free structural snapshot."""
    bars = _bars()
    origin, split, target_ts = bars.index[0], bars.index[2_600], bars.index[3_100]
    target = {target_ts: [{"target_kind": "entry", "target_id": "candidate", "candidate_id": "candidate"}]}
    full = CausalSREngine(
        "A/USD:USD", bars, output_start=origin, output_end=target_ts,
        snapshot_targets=target, record_tape=False,
    )
    _, _, _, expected = full.run()
    assert len(expected) == 1
    seed = CausalSREngine(
        "A/USD:USD", bars.loc[:split], output_start=origin, output_end=split,
        record_tape=False,
    )
    seed.bootstrap_append_state()
    checkpoint = seed.save_checkpoint(tmp_path / "causal_sr_state.json")
    json.loads(checkpoint.read_text())
    restored = CausalSREngine.load_checkpoint(checkpoint, record_tape=False)
    _, _, _, actual = restored.advance(bars.loc[bars.index > split], snapshot_targets=target)
    assert len(actual) == 1
    pdt.assert_frame_equal(_snapshot_frame(actual), _snapshot_frame(expected), check_exact=True)


def test_sr_checkpoint_pending_resolution_uses_timestamp_not_stale_archive_index(tmp_path: Path):
    """A bounded checkpoint must resolve its pending event by timestamp.

    Checkpoints retain 45 days of raw bars, while a pending event's original
    integer position may refer to a much longer source archive.  Reusing that
    position after restore used to fail with an IndexError even though the
    exact resolution bar was present in the appended suffix.
    """
    bars = _bars(5_100)
    split, resolve = bars.index[5_000], bars.index[5_001]
    engine = CausalSREngine(
        "A/USD:USD", bars.loc[:split], output_start=bars.index[0], output_end=split,
        config=SREngineConfig(horizon_bars=1), record_tape=False,
    )
    engine.bootstrap_append_state()
    zone = Zone(
        zone_id="stale-index-zone", side="support", center=float(bars.close.iloc[5_000]),
        lower=float(bars.close.iloc[5_000] - 0.1), upper=float(bars.close.iloc[5_000] + 0.1),
        created_ts=split, available_ts=split, last_seen_ts=split, pending=True,
    )
    engine.zones[zone.zone_id] = zone
    engine.pending[resolve].append(PendingInteraction(
        zone_id=zone.zone_id, event_ts=split,
        # Deliberately nonsensical after the bounded-tail checkpoint restore.
        event_index=90_000, resolve_index=90_001, resolve_ts=resolve,
        atr=1.0, side="support", center=zone.center, lower=zone.lower, upper=zone.upper,
        row={"event_marker": "checkpoint-regression"},
    ))
    restored = CausalSREngine.load_checkpoint(
        engine.save_checkpoint(tmp_path / "causal_sr_pending_state.json"), record_tape=False,
    )
    _, _, interactions, _ = restored.advance(bars.loc[bars.index > split], snapshot_targets={})
    resolved = interactions.loc[interactions["event_marker"].eq("checkpoint-regression")].iloc[0]
    assert pd.Timestamp(resolved["label_available_ts"]) == resolve


def test_sr_checkpoint_rejects_rewrite_of_processed_source(tmp_path: Path):
    # The rewritten bar lies outside the persisted 45-day recurrence tail;
    # this exercises the cumulative source-identity chain, not merely the
    # local tail equality check.
    bars = _bars(4_500)
    split = bars.index[4_400]
    engine = CausalSREngine(
        "A/USD:USD", bars.loc[:split], output_start=bars.index[0], output_end=split,
        record_tape=False,
    )
    engine.bootstrap_append_state()
    checkpoint = engine.save_checkpoint(tmp_path / "causal_sr_state.json")
    restored = CausalSREngine.load_checkpoint(checkpoint, record_tape=False)
    rewritten = bars.loc[[bars.index[10]]].copy()
    rewritten.loc[:, "close"] += 1.0
    with pytest.raises(ValueError, match="unverifiable off-tail overlap"):
        restored.advance(rewritten)


def test_sr_checkpoint_rejects_rewrite_inside_retained_tail(tmp_path: Path):
    """A supplied retained overlap is checked against the checkpoint bars."""
    bars = _bars(3_000)
    split = bars.index[2_800]
    engine = CausalSREngine(
        "A/USD:USD", bars.loc[:split], output_start=bars.index[0], output_end=split,
        record_tape=False,
    )
    engine.bootstrap_append_state()
    restored = CausalSREngine.load_checkpoint(
        engine.save_checkpoint(tmp_path / "causal_sr_tail_state.json"), record_tape=False,
    )
    rewritten = bars.loc[[bars.index[2_790]]].copy()
    rewritten.loc[:, "close"] += 1.0
    with pytest.raises(ValueError, match="rewrites an already processed bar"):
        restored.advance(rewritten)


def test_c1_append_store_is_target_free_and_matches_engine_snapshot(tmp_path: Path):
    bars = _bars()
    origin, decision = bars.index[0], bars.index[3_100]
    targets = [{"target_kind": "entry", "target_id": "candidate", "candidate_id": "candidate"}]
    direct = CausalSREngine(
        "A/USD:USD", bars, output_start=origin, output_end=decision,
        snapshot_targets={decision: targets}, record_tape=False,
    )
    _, _, _, expected = direct.run()
    store = CausalSRC1AppendState(tmp_path / "state", source_origin=origin)
    actual = store.materialize(symbol="A/USD:USD", bars=bars.loc[:decision], decision_ts=decision, targets=targets)
    assert len(actual) == len(expected) == 1
    pdt.assert_frame_equal(_snapshot_frame(actual), _snapshot_frame(expected), check_exact=True)
    assert (tmp_path / "state" / "state_manifest.json").is_file()
    assert "y_reaction_strength" not in actual.columns


def test_c1_append_store_later_snapshot_matches_without_reset(tmp_path: Path):
    bars = _bars()
    origin, first, second = bars.index[0], bars.index[2_900], bars.index[3_100]
    target_1 = [{"target_kind": "entry", "target_id": "one", "candidate_id": "one"}]
    target_2 = [{"target_kind": "entry", "target_id": "two", "candidate_id": "two"}]
    direct = CausalSREngine(
        "A/USD:USD", bars, output_start=origin, output_end=second,
        snapshot_targets={first: target_1, second: target_2}, record_tape=False,
    )
    _, _, _, expected = direct.run()
    store = CausalSRC1AppendState(tmp_path / "state", source_origin=origin)
    store.materialize(symbol="A/USD:USD", bars=bars.loc[:first], decision_ts=first, targets=target_1)
    actual = store.materialize(symbol="A/USD:USD", bars=bars.loc[:second], decision_ts=second, targets=target_2)
    expected_second = expected.loc[expected["candidate_id"].eq("two")].copy()
    assert len(actual) == len(expected_second) == 1
    pdt.assert_frame_equal(_snapshot_frame(actual), _snapshot_frame(expected_second), check_exact=True)


def test_c1_lva_zone_rows_keep_target_free_identity_and_completed_profile_context(tmp_path: Path):
    bars = _bars()
    origin, decision = bars.index[0], bars.index[3_100]
    target = [{"target_kind": "entry", "target_id": "candidate", "candidate_id": "candidate"}]
    store = CausalSRC1AppendState(tmp_path / "state", source_origin=origin)
    snapshots = store.materialize(symbol="A/USD:USD", bars=bars.loc[:decision], decision_ts=decision, targets=target)
    rows = materialize_lva_zone_rows(snapshots=snapshots, bars=bars.loc[:decision], decision_ts=decision)
    assert not rows.empty
    assert set(rows["candidate_id"]) == {"candidate"}
    assert pd.to_datetime(rows["snapshot_ts"], utc=True).le(decision).all()
    assert "profile_poc_distance_atr" in rows
    assert "y_reaction_strength" not in rows


def test_c1_lva_runtime_scores_only_target_free_source_rows(tmp_path: Path):
    class Bundle:
        def score_zone_rows(self, rows: pd.DataFrame) -> pd.DataFrame:
            assert "profile_poc_distance_atr" in rows
            assert "y_reaction_strength" not in rows
            return rows.loc[:, ["candidate_id", "snapshot_ts"]].drop_duplicates().assign(sr_score=1.0)

    bars = _bars()
    origin, decision = bars.index[0], bars.index[3_100]
    state = CausalSRC1AppendState(tmp_path / "state", source_origin=origin)
    result = score_c1_lva_target_free(
        state=state, bundle=Bundle(), symbol="A/USD:USD", bars=bars.loc[:decision],
        decision_ts=decision,
        targets=[{"target_kind": "entry", "target_id": "candidate", "candidate_id": "candidate"}],
    )
    assert result.candidate_id.tolist() == ["candidate"]
    assert result.c1_lva_source_state.tolist() == ["append_only_completed_bar"]
