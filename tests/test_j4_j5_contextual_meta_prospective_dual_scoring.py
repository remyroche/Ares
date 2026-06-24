from pathlib import Path

import pandas as pd

from scripts.run_j4_j5_contextual_meta_prospective_dual_scoring import run


def _write_freeze(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"head": "short_asset", "effective_fresh_oos_after": "2026-06-15T04:00:00+00:00"},
            {"head": "long_dist", "effective_fresh_oos_after": "2026-06-15T04:00:00+00:00"},
        ]
    ).to_csv(path, index=False)


def _write_ledger(root: Path) -> None:
    ledger_dir = root / "prediction_ledgers" / "run_a"
    ledger_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "signal_bar_ts": [
                "2026-06-17T14:00:00Z",
                "2026-06-17T14:00:00Z",
                "2026-06-18T14:00:00Z",
            ],
            "decision_ts": [
                "2026-06-17T15:01:00Z",
                "2026-06-17T15:02:00Z",
                "2026-06-18T15:01:00Z",
            ],
            "symbol": ["AAVE/USD:USD", "SOL/USD:USD", "ETH/USD:USD"],
            "strategy_id": [
                "short_asset_minus_mkt_oi_1d_peer_resid_foo",
                "long_dist_ema20_atr_foo",
                "short_asset_minus_mkt_oi_1d_peer_resid_foo",
            ],
            "calibrated_score": [0.7, 0.6, 0.8],
            "meta_pred": [0.65, 0.55, 0.75],
        }
    ).to_parquet(ledger_dir / "prediction_ledger.parquet", index=False)


def test_prospective_dual_scoring_reports_baseline_only_without_candidates(tmp_path: Path) -> None:
    freeze = tmp_path / "freeze.csv"
    ledger_root = tmp_path / "live_state"
    out_dir = tmp_path / "out"
    _write_freeze(freeze)
    _write_ledger(ledger_root)

    audit = run(
        freeze_manifest=freeze,
        ledger_root=ledger_root,
        score_dirs=[],
        score_files=[],
        output_dir=out_dir,
        start="2026-06-16",
        end="2026-06-22",
    )

    assert audit["status"] == "baseline_only_missing_candidate_scores"
    summary = pd.read_csv(out_dir / "prospective_dual_scoring_summary.csv")
    assert int(summary["baseline_rows"].sum()) == 3
    assert int(summary["candidate_rows_matched"].sum()) == 0
    daily = pd.read_csv(out_dir / "prospective_dual_scoring_daily_summary.csv")
    assert set(daily["date"]) == {
        "2026-06-16",
        "2026-06-17",
        "2026-06-18",
        "2026-06-19",
        "2026-06-20",
        "2026-06-21",
        "2026-06-22",
    }


def test_prospective_dual_scoring_merges_candidate_scores(tmp_path: Path) -> None:
    freeze = tmp_path / "freeze.csv"
    ledger_root = tmp_path / "live_state"
    score_dir = tmp_path / "scores"
    out_dir = tmp_path / "out"
    _write_freeze(freeze)
    _write_ledger(ledger_root)
    score_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "timestamp": ["2026-06-17T14:00:00Z", "2026-06-18T14:00:00Z"],
            "symbol": ["AAVE/USD:USD", "ETH/USD:USD"],
            "candidate_score": [0.9, 0.85],
        }
    ).to_parquet(score_dir / "candidate_short_asset.parquet", index=False)
    pd.DataFrame(
        {
            "timestamp": ["2026-06-17T14:00:00Z"],
            "symbol": ["SOL/USD:USD"],
            "candidate_score": [0.61],
        }
    ).to_parquet(score_dir / "candidate_long_dist.parquet", index=False)

    audit = run(
        freeze_manifest=freeze,
        ledger_root=ledger_root,
        score_dirs=[score_dir],
        score_files=[],
        output_dir=out_dir,
        start="2026-06-16",
        end="2026-06-22",
    )

    assert audit["status"] == "dual_scores_ready"
    scored = pd.read_parquet(out_dir / "prospective_dual_scores.parquet")
    assert scored["candidate_score"].notna().all()
    assert set(scored["head"]) == {"short_asset", "long_dist"}
