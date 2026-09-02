from __future__ import annotations

import json

import pandas as pd
import pytest
from pyarrow.lib import ArrowInvalid

from scripts import run_causal_sr_mc1_residual_ablation as subject


def test_causal_sr_root_rejects_oracle_path(tmp_path):
    root = tmp_path / "oracle_NONCAUSAL"
    root.mkdir()
    with pytest.raises(ValueError, match="causal OOF"):
        subject._assert_causal_sr_root(root)


def test_causal_sr_merge_preserves_identity_and_marks_missing(tmp_path):
    root = tmp_path / "causal"
    root.mkdir()
    (root / "run_manifest.json").write_text(
        json.dumps({"schema": "causal-sr-heads-oof-test", "causality": "snapshot rows precede held decisions"}), encoding="utf-8"
    )
    timestamp = pd.Timestamp("2026-06-01T00:00:00Z")
    values = {field: [0.25] for field in subject.SR_FEATURES}
    pd.DataFrame({"candidate_id": ["a"], "snapshot_ts": [timestamp], **values}).to_parquet(
        root / subject.CAUSAL_SNAPSHOT_FILE, index=False
    )
    panel = pd.DataFrame({
        "candidate_id": ["a", "b"], "__decision_ts__": [timestamp, timestamp],
    })
    merged, coverage = subject._merge_causal_sr(panel, root)
    assert merged.candidate_id.tolist() == ["a", "b"]
    assert merged.sr_snapshot_available.tolist() == [1, 0]
    assert coverage.causal_sr_available.tolist() == [1]


def test_source_aligned_labels_exclude_corrupt_symbol_from_every_arm(tmp_path, monkeypatch):
    good = tmp_path / "policy_parts" / "symbol=GOOD_USD:USD"
    bad = tmp_path / "policy_parts" / "symbol=BAD_USD:USD"
    good.mkdir(parents=True)
    bad.mkdir(parents=True)
    pd.DataFrame({
        "candidate_id": ["good"], "policy_path_valid": [True],
        "policy_gross_bps": [125.0], "policy_net_bps": [25.0],
        "policy_exit_bar_15m": [1], "policy_entry_price": [1.0], "policy_exit_price": [1.01],
        "policy_exit_reason": ["timeout"], "policy_label_available_ts": [pd.Timestamp("2026-06-01T01:00:00Z")],
        "policy_cost_bps": [100.0],
    }).to_parquet(good / "policy_labels.parquet", index=False)
    bad_file = bad / "policy_labels.parquet"
    bad_file.write_text("fixture marker", encoding="utf-8")
    original_read = subject.pd.read_parquet

    def corrupt_only_bad(path, *args, **kwargs):
        if str(path) == str(bad_file):
            raise ArrowInvalid("test corrupt source part")
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(subject.pd, "read_parquet", corrupt_only_bad)
    labels, unavailable = subject._source_aligned_labels(tmp_path)
    assert labels.candidate_id.tolist() == ["good"]
    assert unavailable == [{"symbol": "BAD_USD:USD", "reason": "unreadable_parquet:ArrowInvalid"}]
