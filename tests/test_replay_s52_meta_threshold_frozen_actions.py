from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.replay_s52_meta_threshold_frozen_actions import run_replay


def _write_variant(root: Path, name: str) -> None:
    d = root / name
    d.mkdir(parents=True)
    clean = pd.DataFrame(
        {
            "handoff_row_id": ["h0", "h1", "h2"],
            "timestamp": ["2026-05-01 00:00:00", "2026-05-02 00:00:00", "2026-06-01 00:00:00"],
            "symbol": ["BTC/USD:USD", "ETH/USD:USD", "SOL/USD:USD"],
            "side_name": ["long", "short", "short"],
            "month": ["2026-05", "2026-05", "2026-06"],
            "source_semantic_family": ["trend", "mean_reversion", "mean_reversion"],
            "aegmm_cluster": ["c0", "c1", "c1"],
            "meta_score_oof": [0.8, 0.7, 0.9],
        }
    )
    offline = pd.DataFrame(
        {
            "handoff_row_id": ["h0", "h1", "h2"],
            "__ts__": ["2026-05-01 00:00:00", "2026-05-02 00:00:00", "2026-06-01 00:00:00"],
            "__symbol__": ["BTC/USD:USD", "ETH/USD:USD", "SOL/USD:USD"],
            "side_name": ["long", "short", "short"],
            "month": ["2026-05", "2026-05", "2026-06"],
            "ret_net": [0.02, -0.01, 0.03],
            "exec_margin": [0.016, -0.014, 0.026],
            "full_path_bad_mae_1r": [0.0, 1.0, 0.0],
            "timeout": [0.0, 0.0, 1.0],
        }
    )
    clean.to_parquet(d / "s52_meta_threshold_guarded_candidates.parquet", index=False)
    offline.to_parquet(d / "s52_meta_threshold_guarded_offline_eval_candidates.parquet", index=False)
    (d / "s52_meta_threshold_guarded_leakage_audit.json").write_text(
        '{"clean_handoff_has_no_realized_outcomes": true, "duplicate_decision_key_rows": 0}'
    )


def test_frozen_action_replay_joins_clean_actions_to_scored_ledger(tmp_path: Path) -> None:
    handoff_root = tmp_path / "handoff"
    _write_variant(handoff_root, "candidate")
    ledger = pd.DataFrame(
        {
            "__ts__": ["2026-05-01 00:00:00", "2026-05-02 00:00:00", "2026-06-01 00:00:00"],
            "__symbol__": ["BTC/USD:USD", "ETH/USD:USD", "SOL/USD:USD"],
            "side_name": ["long", "short", "short"],
            "ret_net": [0.02, -0.01, 0.03],
            "exec_margin": [0.016, -0.014, 0.026],
            "first_touch_net": [0.02, -0.01, 0.03],
            "full_path_bad_mae_1r": [0.0, 1.0, 0.0],
            "timeout": [0.0, 0.0, 1.0],
            "clean_exec": [1.0, 0.0, 1.0],
            "dirty_positive": [0.0, 0.0, 0.0],
            "first_touch_mae_norm": [0.2, 0.6, 0.1],
            "first_touch_full_path_mae_norm": [0.4, 0.7, 0.2],
        }
    )
    ledger_path = tmp_path / "ledger.parquet"
    ledger.to_parquet(ledger_path, index=False)

    manifest = run_replay(
        handoff_root=handoff_root,
        variants=("candidate",),
        scored_ledger_path=ledger_path,
        out_dir=tmp_path / "out",
        notional_per_trade=100.0,
    )
    summary = pd.read_csv(manifest["outputs"]["summary"])
    replay = pd.read_parquet(manifest["replay_candidates"]["candidate"])

    assert summary.loc[0, "matched_rows"] == 3
    assert summary.loc[0, "unmatched_rows"] == 0
    assert summary.loc[0, "clean_handoff_has_no_realized_outcomes"]
    assert summary.loc[0, "offline_parity_key_set_match"]
    assert abs(summary.loc[0, "sum_net_pnl"] - 4.0) < 1e-9
    assert abs(summary.loc[0, "mean_ret_net"] - ((0.02 - 0.01 + 0.03) / 3.0)) < 1e-9
    assert "ret_net" in replay.columns
    assert "frozen_action_net_pnl" in replay.columns


def test_frozen_action_replay_flags_leaky_clean_handoff(tmp_path: Path) -> None:
    handoff_root = tmp_path / "handoff"
    _write_variant(handoff_root, "candidate")
    clean_path = handoff_root / "candidate" / "s52_meta_threshold_guarded_candidates.parquet"
    clean = pd.read_parquet(clean_path)
    clean["ret_net"] = 0.01
    clean.to_parquet(clean_path, index=False)
    ledger = pd.DataFrame(
        {
            "__ts__": ["2026-05-01 00:00:00", "2026-05-02 00:00:00", "2026-06-01 00:00:00"],
            "__symbol__": ["BTC/USD:USD", "ETH/USD:USD", "SOL/USD:USD"],
            "side_name": ["long", "short", "short"],
            "ret_net": [0.02, -0.01, 0.03],
            "exec_margin": [0.016, -0.014, 0.026],
        }
    )
    ledger_path = tmp_path / "ledger.parquet"
    ledger.to_parquet(ledger_path, index=False)

    manifest = run_replay(
        handoff_root=handoff_root,
        variants=("candidate",),
        scored_ledger_path=ledger_path,
        out_dir=tmp_path / "out",
    )
    summary = pd.read_csv(manifest["outputs"]["summary"])

    assert not bool(summary.loc[0, "clean_handoff_has_no_realized_outcomes"])
    assert "ret_net" in summary.loc[0, "clean_handoff_forbidden_columns"]
