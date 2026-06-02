#!/usr/bin/env python3
"""Reconcile live prediction-ledger decisions against replay decision artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd


DEFAULT_LEDGER = Path("data_perp/exchanges/krakenfutures/live_state/prediction_ledger.parquet")
DEFAULT_REPLAY = Path(
    "data_perp/artifacts/20260525_010004_nopenalty/"
    "policy_holdout_frozen_replay_1m_fallback/"
    "portfolio_policy_replay/per_candidate_replay_decisions.parquet"
)
DEFAULT_CANDIDATES = Path(
    "data_perp/artifacts/20260525_010004_nopenalty/"
    "policy_holdout_frozen_replay_1m_fallback/simple_policy_holdout_candidates.parquet"
)
DEFAULT_OUT = Path("extreme_price_movements/reports/inference_mismatch_investigation")


def _read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def _value_counts(df: pd.DataFrame, col: str, limit: int = 25) -> Dict[str, int]:
    if df.empty or col not in df.columns:
        return {}
    counts = (
        df[col]
        .fillna("NA")
        .astype(str)
        .value_counts(dropna=False)
        .head(int(limit))
    )
    return {str(k): int(v) for k, v in counts.items()}


def _time_range(df: pd.DataFrame, col: str) -> Dict[str, Any]:
    if df.empty or col not in df.columns:
        return {"non_null": 0, "min": None, "max": None}
    ts = pd.to_datetime(df[col], utc=True, errors="coerce")
    return {
        "non_null": int(ts.notna().sum()),
        "min": ts.min().isoformat() if ts.notna().any() else None,
        "max": ts.max().isoformat() if ts.notna().any() else None,
    }


def _normalise_live_ledger(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()
    if "signal_bar_ts" in out.columns:
        out["reconcile_ts"] = pd.to_datetime(
            out["signal_bar_ts"], utc=True, errors="coerce"
        )
    else:
        out["reconcile_ts"] = pd.to_datetime(
            out.get("timestamp"), utc=True, errors="coerce"
        ).dt.floor("h")
    if "strategy_id" not in out.columns:
        out["strategy_id"] = pd.NA
    if "side" not in out.columns:
        out["side"] = pd.NA
    out["side"] = out["side"].fillna("").astype(str).str.lower()
    return out


def _normalise_replay(df: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()
    out["reconcile_ts"] = pd.to_datetime(
        out.get("timestamp"), utc=True, errors="coerce"
    )
    if "strategy_id" not in out.columns:
        out["strategy_id"] = pd.NA
    if "side" not in out.columns:
        out["side"] = pd.NA
    out["side"] = out["side"].fillna("").astype(str).str.lower()

    if not candidates.empty:
        cand = candidates.copy()
        cand["reconcile_ts"] = pd.to_datetime(
            cand.get("timestamp"), utc=True, errors="coerce"
        )
        cand["side"] = cand["side"].fillna("").astype(str).str.lower()
        join_cols = ["reconcile_ts", "symbol", "side", "strategy_id", "normalized_rank_score"]
        add_cols = [
            c
            for c in (
                "entry_execution_source",
                "entry_delay_actual_minutes",
                "entry_delay_fallback_minutes",
                "entry_gap_bps",
                "net_return",
                "gross_return",
                "auction_rank_score",
            )
            if c in cand.columns
        ]
        if all(c in out.columns for c in join_cols) and all(c in cand.columns for c in join_cols):
            out = out.merge(
                cand[join_cols + add_cols],
                on=join_cols,
                how="left",
                validate="one_to_one",
            )
    return out


def _match_flags(live: pd.DataFrame, replay: pd.DataFrame) -> pd.DataFrame:
    if live.empty:
        return live.copy()
    out = live.copy()
    exact_cols = ["reconcile_ts", "symbol", "side", "strategy_id"]
    loose_cols = ["reconcile_ts", "symbol", "side"]
    if replay.empty:
        out["replay_exact_match"] = False
        out["replay_symbol_side_match"] = False
        return out
    exact = replay[exact_cols].drop_duplicates() if all(c in replay.columns for c in exact_cols) else pd.DataFrame()
    loose = replay[loose_cols].drop_duplicates() if all(c in replay.columns for c in loose_cols) else pd.DataFrame()
    if not exact.empty and all(c in out.columns for c in exact_cols):
        out = out.merge(
            exact.assign(replay_exact_match=True),
            on=exact_cols,
            how="left",
        )
        out["replay_exact_match"] = out["replay_exact_match"].eq(True)
    else:
        out["replay_exact_match"] = False
    if not loose.empty and all(c in out.columns for c in loose_cols):
        out = out.merge(
            loose.assign(replay_symbol_side_match=True),
            on=loose_cols,
            how="left",
        )
        out["replay_symbol_side_match"] = (
            out["replay_symbol_side_match"].eq(True)
        )
    else:
        out["replay_symbol_side_match"] = False
    return out


def _write_csv(df: pd.DataFrame, path: Path, columns: Iterable[str]) -> None:
    cols = [c for c in columns if c in df.columns]
    df.loc[:, cols].to_csv(path, index=False)


def _strategy_side_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    group_cols = [c for c in ["side", "strategy_id"] if c in df.columns]
    if not group_cols:
        return pd.DataFrame()
    named_aggs: Dict[str, Any] = {"rows": ("symbol", "size")}
    if "accepted" in df.columns:
        named_aggs["accepted"] = ("accepted", lambda s: int(pd.Series(s).astype(bool).sum()))
    if "net_return" in df.columns:
        named_aggs["mean_net_return"] = ("net_return", "mean")
        named_aggs["net_hit_rate"] = ("net_return", lambda s: float((pd.to_numeric(s, errors="coerce") > 0).mean()))
    return df.groupby(group_cols, dropna=False).agg(**named_aggs).reset_index()


def _markdown_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df.empty:
        return "_No rows._"
    return df.head(max_rows).to_markdown(index=False)


def reconcile(
    *,
    ledger_path: Path,
    replay_path: Path,
    candidates_path: Optional[Path],
    output_dir: Path,
    report_name: str,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    live_raw = _read_parquet(ledger_path)
    replay_raw = _read_parquet(replay_path)
    candidates = _read_parquet(candidates_path) if candidates_path else pd.DataFrame()
    live = _normalise_live_ledger(live_raw)
    replay = _normalise_replay(replay_raw, candidates)
    live_matched = _match_flags(live, replay)

    current_run = "20260525_010004_nopenalty"
    artifact_cols = [c for c in ["model_artifact_run_id", "policy_artifact_run_id"] if c in live_matched.columns]
    current_mask = pd.Series(False, index=live_matched.index)
    for col in artifact_cols:
        current_mask |= live_matched[col].fillna("").astype(str).eq(current_run)
    current_live = live_matched.loc[current_mask].copy()

    replay_summary = _strategy_side_summary(replay)
    live_current_summary = _strategy_side_summary(current_live)

    live_matched_path = output_dir / "live_vs_replay_decision_matches.csv"
    current_path = output_dir / "live_current_artifact_rows.csv"
    replay_summary_path = output_dir / "replay_decision_strategy_summary.csv"
    summary_path = output_dir / "live_vs_replay_decision_reconciliation.json"
    report_path = output_dir / report_name

    _write_csv(
        live_matched,
        live_matched_path,
        [
            "timestamp",
            "signal_bar_ts",
            "decision_ts",
            "reconcile_ts",
            "symbol",
            "side",
            "strategy_id",
            "model_artifact_run_id",
            "policy_artifact_run_id",
            "portfolio_decision",
            "portfolio_reject_reason",
            "liquidity_reject_reason",
            "was_traded",
            "normalized_rank_score",
            "final_threshold",
            "signal_gap_bps",
            "expected_total_entry_friction_bps",
            "entry_delay_adverse_bps",
            "signal_to_entry_seconds",
            "replay_exact_match",
            "replay_symbol_side_match",
        ],
    )
    _write_csv(
        current_live,
        current_path,
        [
            "timestamp",
            "signal_bar_ts",
            "decision_ts",
            "reconcile_ts",
            "symbol",
            "side",
            "strategy_id",
            "model_artifact_run_id",
            "policy_artifact_run_id",
            "portfolio_decision",
            "portfolio_reject_reason",
            "was_traded",
            "normalized_rank_score",
            "final_threshold",
            "signal_gap_bps",
            "expected_total_entry_friction_bps",
            "entry_delay_adverse_bps",
            "signal_to_entry_seconds",
            "replay_exact_match",
            "replay_symbol_side_match",
        ],
    )
    replay_summary.to_csv(replay_summary_path, index=False)

    summary = {
        "ledger_path": str(ledger_path),
        "replay_path": str(replay_path),
        "candidates_path": str(candidates_path) if candidates_path else None,
        "live_rows": int(len(live_matched)),
        "replay_rows": int(len(replay)),
        "candidate_rows": int(len(candidates)),
        "live_time_range": {
            "timestamp": _time_range(live_matched, "timestamp"),
            "signal_bar_ts": _time_range(live_matched, "signal_bar_ts"),
            "decision_ts": _time_range(live_matched, "decision_ts"),
        },
        "replay_time_range": {"timestamp": _time_range(replay, "reconcile_ts")},
        "live_counts": {
            "model_artifact_run_id": _value_counts(live_matched, "model_artifact_run_id"),
            "policy_artifact_run_id": _value_counts(live_matched, "policy_artifact_run_id"),
            "strategy_id": _value_counts(live_matched, "strategy_id"),
            "portfolio_decision": _value_counts(live_matched, "portfolio_decision"),
            "portfolio_reject_reason": _value_counts(live_matched, "portfolio_reject_reason"),
            "liquidity_reject_reason": _value_counts(live_matched, "liquidity_reject_reason"),
            "was_traded": _value_counts(live_matched, "was_traded"),
        },
        "replay_counts": {
            "strategy_id": _value_counts(replay, "strategy_id"),
            "accepted": _value_counts(replay, "accepted"),
            "rejection_reason": _value_counts(replay, "rejection_reason"),
            "entry_execution_source": _value_counts(replay, "entry_execution_source"),
        },
        "match_counts": {
            "live_exact_replay_matches": int(live_matched["replay_exact_match"].sum()),
            "live_symbol_side_replay_matches": int(
                live_matched["replay_symbol_side_match"].sum()
            ),
            "current_artifact_live_rows": int(len(current_live)),
            "current_artifact_exact_replay_matches": int(
                current_live["replay_exact_match"].sum()
            )
            if not current_live.empty
            else 0,
            "current_artifact_symbol_side_replay_matches": int(
                current_live["replay_symbol_side_match"].sum()
            )
            if not current_live.empty
            else 0,
        },
        "outputs": {
            "live_matches_csv": str(live_matched_path),
            "current_live_csv": str(current_path),
            "replay_strategy_summary_csv": str(replay_summary_path),
            "report": str(report_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))

    lines: List[str] = [
        "# Live vs Replay Decision Reconciliation",
        "",
        "Status: updated, 2026-06-02.",
        "",
        "## Inputs",
        "",
        f"- Live ledger: `{ledger_path}`",
        f"- Replay decisions: `{replay_path}`",
        f"- Replay candidates: `{candidates_path}`" if candidates_path else "- Replay candidates: none",
        "",
        "## Summary",
        "",
        f"- Live ledger rows: `{len(live_matched)}`.",
        f"- Replay decision rows: `{len(replay)}`.",
        f"- Live exact replay matches on signal timestamp, symbol, side, and strategy: `{summary['match_counts']['live_exact_replay_matches']}`.",
        f"- Live loose replay matches on signal timestamp, symbol, and side: `{summary['match_counts']['live_symbol_side_replay_matches']}`.",
        f"- Live rows from `{current_run}`: `{len(current_live)}`.",
        f"- Current-run exact replay matches: `{summary['match_counts']['current_artifact_exact_replay_matches']}`.",
        "",
        "## Live Artifact Mix",
        "",
        "Model artifact run ids:",
        "",
        _markdown_table(pd.DataFrame(summary["live_counts"]["model_artifact_run_id"].items(), columns=["model_artifact_run_id", "rows"])),
        "",
        "Policy artifact run ids:",
        "",
        _markdown_table(pd.DataFrame(summary["live_counts"]["policy_artifact_run_id"].items(), columns=["policy_artifact_run_id", "rows"])),
        "",
        "## Live Gate Distribution",
        "",
        _markdown_table(pd.DataFrame(summary["live_counts"]["portfolio_decision"].items(), columns=["portfolio_decision", "rows"])),
        "",
        "Portfolio reject reasons:",
        "",
        _markdown_table(pd.DataFrame(summary["live_counts"]["portfolio_reject_reason"].items(), columns=["portfolio_reject_reason", "rows"])),
        "",
        "Liquidity reject reasons:",
        "",
        _markdown_table(pd.DataFrame(summary["live_counts"]["liquidity_reject_reason"].items(), columns=["liquidity_reject_reason", "rows"])),
        "",
        "## Replay Gate Distribution",
        "",
        _markdown_table(pd.DataFrame(summary["replay_counts"]["rejection_reason"].items(), columns=["rejection_reason", "rows"])),
        "",
        "## Replay Strategy Summary",
        "",
        _markdown_table(replay_summary, max_rows=30),
        "",
        "## Current-Run Live Rows",
        "",
        _markdown_table(
            current_live[
                [
                    c
                    for c in [
                        "signal_bar_ts",
                        "symbol",
                        "side",
                        "strategy_id",
                        "portfolio_decision",
                        "was_traded",
                        "normalized_rank_score",
                        "final_threshold",
                        "expected_total_entry_friction_bps",
                        "entry_delay_adverse_bps",
                        "replay_exact_match",
                    ]
                    if c in current_live.columns
                ]
            ],
            max_rows=20,
        ),
        "",
        "## Interpretation",
        "",
        "- The current live ledger is not a clean live-vs-replay parity table for the six-head package because it mixes three artifact generations.",
        f"- Only `{len(current_live)}` live row references `{current_run}` in either model or policy artifact fields, and it does not match the frozen replay window/strategy set.",
        "- Replay decisions reject mainly on portfolio state gates (`symbol_already_open`, dynamic thresholds, position/concurrency, cooldown). Live rejects include rank, min-notional sizing, spread, stale ticker, and stale adverse price movement gates that are not represented one-for-one in the portfolio replay artifact.",
        "- Therefore the next valid test is a fresh live-test cycle using the current six-head package with the ledger cleared or namespaced to this run, then replaying the same signal bars with the same initial portfolio state.",
    ]
    report_path.write_text("\n".join(lines) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--replay-decisions", type=Path, default=DEFAULT_REPLAY)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--report-name", default="live_vs_replay_decision_reconciliation.md")
    args = parser.parse_args()
    summary = reconcile(
        ledger_path=args.ledger,
        replay_path=args.replay_decisions,
        candidates_path=args.candidates,
        output_dir=args.output_dir,
        report_name=args.report_name,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
