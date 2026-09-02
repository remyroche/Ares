#!/usr/bin/env python3
"""Compare fill- and decision-close-anchored rich-exit state on one panel.

This is a research-only, causally valid comparison for the frozen dual BCF /
current-v5 admission panel used by the entry-delay study.  It holds fixed:

* the sealed target-free candidates and BCF MC1 auction priority;
* the actual entry timestamp and Kraken one-minute path (default: +5 minutes);
* frozen rich-exit parameters, H12 horizon, policy cost, and portfolio rules.

``fill_anchor`` is the executable control: MFE, MAE, geometry, stops and
trailing state are all referenced to the observed fill.  ``decision_close``
replays the *same post-fill path* but references state/geometry to the close
of the completed hourly signal candle.  Crucially, its realised PnL is still
computed from the observed fill to the generated exit price.  It therefore
does not credit pre-fill price movement as PnL.

The decision-close arm is a causal policy counterfactual, not a live-policy
change.  It exposes whether signal-time geometric state would manage a late
fill better than actual-fill state.  The manifest records the rate of
reference-derived exits whose threshold sits above the realised long fill;
those need an explicit executable-order treatment before any promotion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.data_store import PartitionedOHLCVStore  # noqa: E402
from extreme_price_movements.exact_1m_rich_policy_contract import (  # noqa: E402
    Exact1mRichV2ExecutionContract,
    RichExitExtensions,
    replay_exact_1m_rich_policy_v2,
)
from scripts.materialize_strict_r3_exact_1m_policy_hpo_dataset import (  # noqa: E402
    ATR_SOURCE_LOOKBACK_HOURS,
    _causal_atr,
    _clean_minute,
)
from scripts.run_strict_r3_exact_1m_rich_entry_delay_ladder import _complete_mask  # noqa: E402
from scripts.run_strict_r3_exact_1m_rich_matched_attribution import (  # noqa: E402
    DEFAULT_CANDIDATE_DIR,
    DEFAULT_FROZEN_POLICY,
    _candidate_panel,
    _json_safe,
    _load_frozen_policy,
    _portfolio_candidates,
    _sha256,
    _write_arm,
)


DEFAULT_MINUTE_ROOT = ROOT / "data_perp/exchanges/krakenfutures/execution_1m"
DEFAULT_OUT = ROOT / "data_perp/artifacts/strict_r3_exact1m_state_anchor_20260817_v1"
HORIZON_MINUTES = 12 * 60


def _assert_new(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"immutable output already exists: {path}")
    path.mkdir(parents=True, exist_ok=False)


def _utc(value: Any) -> pd.DatetimeIndex | pd.Series:
    return pd.to_datetime(value, utc=True, errors="raise")


def _replay(
    *,
    state_anchor: np.ndarray,
    fill: np.ndarray,
    atr: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    entry_timestamps: pd.Series,
    params: Any,
    median_atr_fraction: float,
    delay: int,
) -> dict[str, np.ndarray]:
    """Replay policy state from anchor; debit returns from actual fill."""
    result = replay_exact_1m_rich_policy_v2(
        entry=state_anchor,
        atr=atr,
        highs=highs,
        lows=lows,
        closes=closes,
        entry_timestamps=_utc(entry_timestamps),
        params=params,
        median_atr_fraction=float(median_atr_fraction),
        extensions=RichExitExtensions(),
        contract=Exact1mRichV2ExecutionContract(entry_delay_minutes=int(delay)),
    )
    # State geometry may be decision-close anchored, but realised economics
    # must always start at the price at which the trade could actually exist.
    gross_fill = (np.asarray(result["exit_price"], dtype=float) / fill - 1.0) * 10_000.0
    result["gross_bps"] = gross_fill
    result["net_bps"] = gross_fill - 100.0
    return result


def _outcomes(
    *,
    rows: pd.DataFrame,
    fill: np.ndarray,
    state_anchor: np.ndarray,
    result: dict[str, np.ndarray],
    arm: str,
) -> pd.DataFrame:
    return pd.DataFrame({
        "candidate_id": rows["candidate_id"].astype(str).to_numpy(),
        "decision_timestamp": _utc(rows["decision_timestamp"]),
        "entry_timestamp": _utc(rows["entry_timestamp"]),
        "entry_price": fill,
        "state_anchor_price": state_anchor,
        "anchor_gap_bps": (fill / state_anchor - 1.0) * 10_000.0,
        "exit_timestamp": pd.to_datetime(result["exit_timestamp"], utc=True),
        "exit_price": np.asarray(result["exit_price"], dtype=float),
        "gross_bps": np.asarray(result["gross_bps"], dtype=float),
        "net_bps": np.asarray(result["net_bps"], dtype=float),
        "exit_reason": np.asarray(result["exit_reason"], dtype=object),
        "exit_minute": np.asarray(result["exit_minute"], dtype=np.int16),
        "state_mfe_bps": np.asarray(result["final_mfe"], dtype=float) / state_anchor * 10_000.0,
        "state_mae_bps": np.asarray(result["final_mae"], dtype=float) / state_anchor * 10_000.0,
        "outcome_available": np.asarray(result["path_valid"], dtype=bool),
        "outcome_invalid_reason": "",
        "outcome_source": f"exact_1m_state_anchor_{arm}_v1",
    })


def run(args: argparse.Namespace) -> Path:
    output = Path(args.out_dir).resolve()
    _assert_new(output)
    delay = int(args.entry_delay_minutes)
    if delay < 0 or delay > 60:
        raise ValueError("entry delay must be between zero and 60 minutes")
    routed, route_audit = _candidate_panel(Path(args.candidate_dir))
    params, median_atr_fraction, frozen_audit = _load_frozen_policy(Path(args.frozen_policy))
    store = PartitionedOHLCVStore(str(Path(args.minute_root).resolve()), timeframe="1m")

    fill_parts: list[pd.DataFrame] = []
    decision_parts: list[pd.DataFrame] = []
    coverage: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    pending_rows = 0

    def flush() -> None:
        nonlocal pending_rows
        if not pending_rows:
            return
        rows = pd.concat([part["rows"] for part in pending], ignore_index=True)
        fill = np.concatenate([part["fill"] for part in pending]).astype(float, copy=False)
        anchor = np.concatenate([part["anchor"] for part in pending]).astype(float, copy=False)
        atr = np.concatenate([part["atr"] for part in pending]).astype(float, copy=False)
        highs = np.concatenate([part["high"] for part in pending]).astype(np.float32, copy=False)
        lows = np.concatenate([part["low"] for part in pending]).astype(np.float32, copy=False)
        closes = np.concatenate([part["close"] for part in pending]).astype(np.float32, copy=False)
        fill_result = _replay(
            state_anchor=fill, fill=fill, atr=atr, highs=highs, lows=lows, closes=closes,
            entry_timestamps=rows["entry_timestamp"], params=params,
            median_atr_fraction=median_atr_fraction, delay=delay,
        )
        decision_result = _replay(
            state_anchor=anchor, fill=fill, atr=atr, highs=highs, lows=lows, closes=closes,
            entry_timestamps=rows["entry_timestamp"], params=params,
            median_atr_fraction=median_atr_fraction, delay=delay,
        )
        if not fill_result["path_valid"].all() or not decision_result["path_valid"].all():
            raise AssertionError("complete exact path became invalid during anchor replay")
        fill_parts.append(_outcomes(rows=rows, fill=fill, state_anchor=fill, result=fill_result, arm="fill"))
        decision_parts.append(_outcomes(rows=rows, fill=fill, state_anchor=anchor, result=decision_result, arm="decision_close"))
        pending.clear()
        pending_rows = 0

    for symbol, raw in routed.groupby("symbol", sort=True):
        group = raw.reset_index(drop=True).copy()
        decision = _utc(group["timestamp"])
        entries = pd.DatetimeIndex(decision + pd.Timedelta(minutes=delay))
        earliest = decision.min() - pd.Timedelta(minutes=1, hours=ATR_SOURCE_LOOKBACK_HOURS)
        latest = entries.max() + pd.Timedelta(minutes=HORIZON_MINUTES - 1)
        minute = _clean_minute(store.load(
            str(symbol), columns=["ts", "open", "high", "low", "close"],
            start_ts=earliest, end_ts=latest,
        ))
        if minute.empty:
            coverage.append({"symbol": str(symbol), "routed_rows": len(group), "complete_rows": 0, "reason": "missing_minute_source"})
            continue
        atr = _causal_atr(minute)
        valid, locations, atr_values, reasons = _complete_mask(minute, atr, entries)
        signal_close = minute["close"].reindex(pd.DatetimeIndex(decision - pd.Timedelta(minutes=1))).to_numpy(float)
        reference_valid = np.isfinite(signal_close) & (signal_close > 0.0)
        common = valid & reference_valid
        coverage.append({
            "symbol": str(symbol), "routed_rows": len(group), "complete_rows": int(common.sum()),
            "missing_path_rows": int((~valid).sum()), "missing_signal_close_rows": int((valid & ~reference_valid).sum()),
            "reason": "ok",
        })
        if not common.any():
            continue
        offsets = np.arange(HORIZON_MINUTES, dtype=np.int64)
        selected = locations[common, None] + offsets[None, :]
        source = {name: minute[name].to_numpy(float) for name in ("open", "high", "low", "close")}
        rows = group.loc[common, ["candidate_id", "timestamp"]].copy().rename(columns={"timestamp": "decision_timestamp"})
        rows["entry_timestamp"] = _utc(rows["decision_timestamp"]) + pd.Timedelta(minutes=delay)
        pending.append({
            "rows": rows,
            "fill": source["open"][selected[:, 0]],
            "anchor": signal_close[common],
            "atr": atr_values[common],
            "high": source["high"][selected].astype(np.float32, copy=False),
            "low": source["low"][selected].astype(np.float32, copy=False),
            "close": source["close"][selected].astype(np.float32, copy=False),
        })
        pending_rows += int(common.sum())
        if pending_rows >= int(args.batch_rows):
            flush()
    flush()

    if not fill_parts or not decision_parts:
        raise RuntimeError("no complete exact paths with signal-close references")
    outcomes = {
        "fill_anchor": pd.concat(fill_parts, ignore_index=True),
        "decision_close_anchor": pd.concat(decision_parts, ignore_index=True),
    }
    if outcomes["fill_anchor"]["candidate_id"].duplicated().any():
        raise AssertionError("duplicate candidate outcomes")
    if not outcomes["fill_anchor"]["candidate_id"].equals(outcomes["decision_close_anchor"]["candidate_id"]):
        raise AssertionError("anchor arms do not share candidate identities")
    # Match the entry-delay ladder's evaluation convention: frozen routing is
    # target-free, then both counterfactuals are assessed on their shared,
    # complete exact-path intersection.  The dropped remainder is retained in
    # source_coverage, rather than being misrepresented as a zero-return trade.
    common_ids = set(outcomes["fill_anchor"]["candidate_id"].astype(str))
    comparison_id_audit: dict[str, Any] | None = None
    if args.eligible_id_audit is not None:
        audit_path = Path(args.eligible_id_audit).resolve()
        eligible = pd.read_parquet(audit_path)
        required = {"candidate_id", "complete_all_requested_delays"}
        if not required.issubset(eligible.columns):
            raise ValueError(f"eligible-id audit missing {sorted(required - set(eligible.columns))}")
        frozen_ids = set(eligible.loc[
            eligible["complete_all_requested_delays"].fillna(False).astype(bool), "candidate_id"
        ].astype(str))
        if not frozen_ids.issubset(common_ids):
            raise AssertionError("eligible-id audit includes a row without an exact anchor path")
        common_ids &= frozen_ids
        comparison_id_audit = {
            "path": str(audit_path), "sha256": _sha256(audit_path),
            "eligible_rows": int(len(frozen_ids)), "purpose": "match predeclared delay-ladder common exact-path population",
        }
    common_routed = routed.loc[routed["candidate_id"].astype(str).isin(common_ids)].copy()
    if len(common_routed) != len(common_ids):
        raise AssertionError("complete anchor population is not a routed subset")
    # Keep every downstream metric and diagnostic on precisely the same
    # predeclared common population.  In particular, do not let the paired
    # path audit retain extra rows merely because +5m happened to be complete
    # when another delay in the supplied ladder was not.
    outcomes = {
        arm: frame.loc[frame["candidate_id"].astype(str).isin(common_ids)].copy()
        for arm, frame in outcomes.items()
    }
    for arm, frame in outcomes.items():
        if len(frame) != len(common_ids) or frame["candidate_id"].duplicated().any():
            raise AssertionError(f"{arm}: exact outcome population does not match common IDs")
        outcomes[arm] = frame.sort_values("candidate_id", kind="stable").reset_index(drop=True)
    if not outcomes["fill_anchor"]["candidate_id"].equals(outcomes["decision_close_anchor"]["candidate_id"]):
        raise AssertionError("restricted anchor arms do not share candidate identities")

    metrics_rows: list[dict[str, Any]] = []
    monthly: list[pd.DataFrame] = []
    exits: list[pd.DataFrame] = []
    accepted: dict[str, pd.DataFrame] = {}
    for arm, outcome in outcomes.items():
        candidate, population = _portfolio_candidates(common_routed, outcome, arm=arm)
        metrics, month, reason, picked = _write_arm(output, arm, candidate, population)
        metrics_rows.append(metrics)
        monthly.append(month)
        exits.append(reason)
        accepted[arm] = picked
        outcome.to_parquet(output / f"{arm}_outcomes.parquet", index=False, compression="zstd")
    summary = pd.DataFrame(metrics_rows).set_index("arm")
    fill_metric, decision_metric = summary.loc["fill_anchor"], summary.loc["decision_close_anchor"]
    delta_fields = [
        "portfolio_accepted_trades", "net_ev_bps_per_trade", "net_sum_bps",
        "portfolio_net_pnl_quote", "portfolio_max_drawdown", "portfolio_sortino",
        "portfolio_worst_week_return",
    ]
    delta = {field: float(decision_metric[field] - fill_metric[field]) for field in delta_fields}
    paired = outcomes["fill_anchor"].merge(
        outcomes["decision_close_anchor"], on="candidate_id", suffixes=("_fill", "_decision"), validate="one_to_one",
    )
    same_exit = paired["exit_reason_fill"].eq(paired["exit_reason_decision"]) & paired["exit_minute_fill"].eq(paired["exit_minute_decision"])
    # A decision reference can lie above a delayed long fill.  This identifies
    # reference-derived threshold fills that would need explicit live order
    # semantics before any production use.
    decision_threshold_above_fill = (
        paired["exit_price_decision"].gt(paired["entry_price_fill"])
        & paired["exit_reason_decision"].isin(["stop_loss", "capital_protect", "smooth_capital_protect", "stepped_mfe_protect", "time_stop"])
    )
    state_audit = pd.DataFrame([{
        "rows": len(paired),
        "same_exit_identity_fraction": float(same_exit.mean()),
        "changed_exit_identity_fraction": float((~same_exit).mean()),
        "mean_anchor_gap_bps": float(paired["anchor_gap_bps_decision"].mean()),
        "p05_anchor_gap_bps": float(paired["anchor_gap_bps_decision"].quantile(0.05)),
        "p95_anchor_gap_bps": float(paired["anchor_gap_bps_decision"].quantile(0.95)),
        "decision_threshold_above_fill_rows": int(decision_threshold_above_fill.sum()),
        "decision_threshold_above_fill_fraction": float(decision_threshold_above_fill.mean()),
    }])
    summary.reset_index().to_parquet(output / "anchor_summary.parquet", index=False, compression="zstd")
    summary.reset_index().to_csv(output / "anchor_summary.csv", index=False)
    pd.DataFrame([delta]).to_parquet(output / "decision_minus_fill_deltas.parquet", index=False, compression="zstd")
    pd.concat(monthly, ignore_index=True).to_parquet(output / "monthly_portfolio_metrics.parquet", index=False, compression="zstd")
    pd.concat(exits, ignore_index=True).to_parquet(output / "exit_reason_metrics.parquet", index=False, compression="zstd")
    state_audit.to_parquet(output / "state_anchor_audit.parquet", index=False, compression="zstd")
    paired.loc[:, [
        "candidate_id", "entry_price_fill", "state_anchor_price_decision", "anchor_gap_bps_decision",
        "exit_reason_fill", "exit_minute_fill", "net_bps_fill", "state_mfe_bps_fill", "state_mae_bps_fill",
        "exit_reason_decision", "exit_minute_decision", "net_bps_decision", "state_mfe_bps_decision", "state_mae_bps_decision",
    ]].to_parquet(output / "paired_path_state_comparison.parquet", index=False, compression="zstd")
    pd.DataFrame(coverage).to_parquet(output / "source_coverage.parquet", index=False, compression="zstd")
    (output / "run_manifest.json").write_text(json.dumps(_json_safe({
        "schema": "strict_r3_exact1m_state_anchor_ablation_v1",
        "status": "complete",
        "purpose": "causal fill-versus-decision-close state anchor ablation; no live contract change",
        "candidate_route": route_audit,
        "target_free_routed_rows": int(len(routed)),
        "shared_complete_exact_path_rows": int(len(common_routed)),
        "comparison_id_audit": comparison_id_audit,
        "frozen_policy": frozen_audit,
        "entry_delay_minutes": delay,
        "entry": "observed Kraken one-minute open at decision plus delay",
        "state_arms": {
            "fill_anchor": "MFE/MAE, geometry and exit-state anchor = actual fill; realised PnL = fill to exit",
            "decision_close_anchor": "MFE/MAE, geometry and exit-state anchor = completed signal-hour close; realised PnL = actual fill to exit",
        },
        "path": "720 complete post-fill exchange-observed one-minute bars; no interpolation",
        "cost": "100 bps exactly once",
        "portfolio": "same BCF-MC1-priority global auction: 7x, 80% margin, 10% slots, two entries per decision",
        "state_audit": state_audit.iloc[0].to_dict(),
        "code_sha256": _sha256(Path(__file__).resolve()),
    }), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", type=Path, default=DEFAULT_CANDIDATE_DIR)
    parser.add_argument("--frozen-policy", type=Path, default=DEFAULT_FROZEN_POLICY)
    parser.add_argument("--minute-root", type=Path, default=DEFAULT_MINUTE_ROOT)
    parser.add_argument("--entry-delay-minutes", type=int, default=5)
    parser.add_argument(
        "--eligible-id-audit", type=Path, default=None,
        help="optional predeclared common-path audit; restricts both arms to its eligible candidate IDs",
    )
    parser.add_argument("--batch-rows", type=int, default=2500)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    print(run(args))


if __name__ == "__main__":
    main()
