#!/usr/bin/env python3
"""Matched entry-delay ladder for the frozen Strict-R3 rich exit policy.

This is intentionally an *offline research* producer.  It holds fixed:

* the target-free dual-MC1 >=30 routed population;
* frozen BCF-MC1 priority and the normal global portfolio auction;
* the frozen rich policy parameters, H12 horizon, and exactly-one 100-bps
  policy cost; and
* complete exchange-observed one-minute OHLC paths only.

It varies only uniform entry offset from the decision timestamp.  To make the
offset comparison fair, every arm is evaluated on the intersection of rows
with a complete causal ATR and 720 complete one-minute bars for *all* tested
offsets.  That post-routing label-coverage intersection is evaluation-only;
it never changes the sealed score/admission route or reserves capacity.

The implementation loads each symbol's minute history once, constructs all
offset paths in bounded batches, and writes only outcomes/audits rather than
duplicating dense raw paths on disk.
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
DEFAULT_OUT = ROOT / (
    "data_perp/artifacts/strict_r3_exact_1m_rich_entry_delay_ladder_"
    "2025_2026_20260817_v1"
)
HORIZON_MINUTES = 12 * 60
DEFAULT_DELAYS = tuple(range(1, 11))


def _utc(value: Any) -> pd.DatetimeIndex | pd.Series:
    return pd.to_datetime(value, utc=True, errors="raise")


def _assert_new_output(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"refusing to overwrite immutable output: {path}")
    path.mkdir(parents=True, exist_ok=False)


def _complete_mask(
    minute: pd.DataFrame,
    atr: pd.Series,
    entries: pd.DatetimeIndex,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return coverage mask, locations, causal ATR, and reason codes."""
    locations = minute.index.get_indexer(entries)
    offsets = np.arange(HORIZON_MINUTES, dtype=np.int64)
    locations_2d = locations[:, None] + offsets[None, :]
    in_range = (locations >= 0) & (locations_2d[:, -1] < len(minute))
    atr_values = atr.reindex(
        entries, method="ffill", tolerance=pd.Timedelta(hours=1)
    ).to_numpy(float)
    valid = np.zeros(len(entries), dtype=bool)
    if in_range.any():
        idx = np.flatnonzero(in_range)
        selected = locations_2d[idx]
        finite = np.isfinite(atr_values[idx]) & (atr_values[idx] > 0.0)
        for column in ("open", "high", "low", "close"):
            finite &= np.isfinite(minute[column].to_numpy(float)[selected]).all(axis=1)
        valid[idx] = finite
    reason = np.full(len(entries), "nonfinite_minute_path", dtype=object)
    reason[locations < 0] = "missing_entry_minute"
    reason[(locations >= 0) & ~in_range] = "incomplete_h12_minute_path"
    no_atr = ~np.isfinite(atr_values) | (atr_values <= 0.0)
    reason[no_atr] = "missing_causal_atr"
    reason[valid] = ""
    return valid, locations, atr_values, reason


def _replay_batch(
    *,
    delay: int,
    parts: list[dict[str, Any]],
    params: Any,
    median_atr_fraction: float,
) -> pd.DataFrame:
    """Run one delay on a bounded collection of fully observed paths."""
    if not parts:
        return pd.DataFrame()
    rows = pd.concat([part["rows"] for part in parts], ignore_index=True)
    entry = np.concatenate([part["entry"] for part in parts]).astype(float, copy=False)
    atr = np.concatenate([part["atr"] for part in parts]).astype(float, copy=False)
    highs = np.concatenate([part["high"] for part in parts]).astype(np.float32, copy=False)
    lows = np.concatenate([part["low"] for part in parts]).astype(np.float32, copy=False)
    closes = np.concatenate([part["close"] for part in parts]).astype(np.float32, copy=False)
    if len(rows) != len(entry) or len(rows) != len(highs):
        raise AssertionError("delay batch path identity is misaligned")
    result = replay_exact_1m_rich_policy_v2(
        entry=entry,
        atr=atr,
        highs=highs,
        lows=lows,
        closes=closes,
        entry_timestamps=_utc(rows["entry_timestamp"]),
        params=params,
        median_atr_fraction=float(median_atr_fraction),
        extensions=RichExitExtensions(),
        contract=Exact1mRichV2ExecutionContract(entry_delay_minutes=int(delay)),
    )
    if not result["path_valid"].all():
        raise AssertionError("complete exact one-minute path failed rich replay")
    return pd.DataFrame(
        {
            "candidate_id": rows["candidate_id"].astype(str).to_numpy(),
            "decision_timestamp": _utc(rows["decision_timestamp"]),
            "entry_timestamp": _utc(rows["entry_timestamp"]),
            "entry_price": entry,
            "exit_timestamp": pd.to_datetime(result["exit_timestamp"], utc=True),
            "exit_price": np.asarray(result["exit_price"], dtype=float),
            "gross_bps": np.asarray(result["gross_bps"], dtype=float),
            "net_bps": np.asarray(result["net_bps"], dtype=float),
            "exit_reason": np.asarray(result["exit_reason"], dtype=object),
            "exit_minute": np.asarray(result["exit_minute"], dtype=np.int16),
            "outcome_available": True,
            "outcome_invalid_reason": "",
            "outcome_source": "exact_1m_frozen_rich_delay_ladder_v1",
        }
    )


def _append_paths(
    pending: dict[int, list[dict[str, Any]]],
    group: pd.DataFrame,
    minute: pd.DataFrame,
    atr_values: dict[int, np.ndarray],
    locations: dict[int, np.ndarray],
    common: np.ndarray,
    delays: tuple[int, ...],
) -> None:
    offsets = np.arange(HORIZON_MINUTES, dtype=np.int64)
    source = {key: minute[key].to_numpy(float) for key in ("open", "high", "low", "close")}
    for delay in delays:
        loc = locations[delay][common]
        selected = loc[:, None] + offsets[None, :]
        rows = group.loc[common, ["candidate_id", "timestamp"]].copy()
        rows = rows.rename(columns={"timestamp": "decision_timestamp"})
        rows["entry_timestamp"] = rows["decision_timestamp"] + pd.Timedelta(minutes=int(delay))
        pending[delay].append(
            {
                "rows": rows,
                "entry": source["open"][selected[:, 0]],
                "atr": atr_values[delay][common],
                "high": source["high"][selected].astype(np.float32, copy=False),
                "low": source["low"][selected].astype(np.float32, copy=False),
                "close": source["close"][selected].astype(np.float32, copy=False),
            }
        )


def run(args: argparse.Namespace) -> Path:
    output = Path(args.out_dir).resolve()
    _assert_new_output(output)
    delays = tuple(sorted({int(value) for value in args.delays}))
    if not delays or min(delays) < 0 or max(delays) > 60:
        raise ValueError("delays must be unique non-negative minute offsets no larger than 60")
    routed, route_audit = _candidate_panel(Path(args.candidate_dir))
    params, median_atr_fraction, frozen_audit = _load_frozen_policy(Path(args.frozen_policy))
    store = PartitionedOHLCVStore(str(Path(args.minute_root).resolve()), timeframe="1m")

    # Outcome labels are generated only after the target-free route above is
    # sealed.  We batch path arrays to cap memory while reusing every symbol's
    # source frame for all offsets.
    pending: dict[int, list[dict[str, Any]]] = {delay: [] for delay in delays}
    pending_rows = 0
    outcome_parts: dict[int, list[pd.DataFrame]] = {delay: [] for delay in delays}
    coverage: list[dict[str, Any]] = []
    common_ids: list[str] = []

    def flush() -> None:
        nonlocal pending_rows
        if pending_rows == 0:
            return
        for delay in delays:
            outcome_parts[delay].append(
                _replay_batch(
                    delay=delay,
                    parts=pending[delay],
                    params=params,
                    median_atr_fraction=median_atr_fraction,
                )
            )
            pending[delay].clear()
        pending_rows = 0

    for symbol, raw_group in routed.groupby("symbol", sort=True):
        group = raw_group.reset_index(drop=True).copy()
        earliest = group["timestamp"].min() + pd.Timedelta(minutes=min(delays)) - pd.Timedelta(hours=ATR_SOURCE_LOOKBACK_HOURS)
        latest = group["timestamp"].max() + pd.Timedelta(minutes=max(delays) + HORIZON_MINUTES - 1)
        minute = _clean_minute(
            store.load(
                str(symbol),
                columns=["ts", "open", "high", "low", "close"],
                start_ts=earliest,
                end_ts=latest,
            )
        )
        if minute.empty:
            coverage.append(
                {
                    "symbol": str(symbol), "routed_rows": int(len(group)),
                    "common_complete_rows": 0, "source_status": "missing_minute_source",
                    **{f"delay_{delay}_complete_rows": 0 for delay in delays},
                }
            )
            continue
        atr = _causal_atr(minute)
        valid: dict[int, np.ndarray] = {}
        loc: dict[int, np.ndarray] = {}
        atr_values: dict[int, np.ndarray] = {}
        reasons: dict[int, np.ndarray] = {}
        for delay in delays:
            entries = pd.DatetimeIndex(group["timestamp"] + pd.Timedelta(minutes=delay))
            valid[delay], loc[delay], atr_values[delay], reasons[delay] = _complete_mask(minute, atr, entries)
        common = np.logical_and.reduce([valid[delay] for delay in delays])
        common_ids.extend(group.loc[common, "candidate_id"].astype(str).tolist())
        record: dict[str, Any] = {
            "symbol": str(symbol), "routed_rows": int(len(group)),
            "common_complete_rows": int(common.sum()), "source_status": "ok",
        }
        for delay in delays:
            record[f"delay_{delay}_complete_rows"] = int(valid[delay].sum())
            for reason, count in pd.Series(reasons[delay][~valid[delay]]).value_counts().items():
                record[f"delay_{delay}_{reason}"] = int(count)
        coverage.append(record)
        if common.any():
            _append_paths(pending, group, minute, atr_values, loc, common, delays)
            pending_rows += int(common.sum())
            if pending_rows >= int(args.batch_rows):
                flush()
    flush()

    if not common_ids:
        raise RuntimeError("no candidate has complete exchange-observed paths at every requested entry offset")
    common_id_set = set(common_ids)
    if len(common_id_set) != len(common_ids):
        raise AssertionError("common path population has duplicate candidate IDs")
    common_routed = routed.loc[routed["candidate_id"].astype(str).isin(common_id_set)].copy()
    if len(common_routed) != len(common_id_set):
        raise AssertionError("common path population is not a subset of the sealed route")
    common_routed = common_routed.sort_values(["timestamp", "candidate_id"], kind="stable").reset_index(drop=True)

    summary_rows: list[dict[str, Any]] = []
    monthly_frames: list[pd.DataFrame] = []
    accepted_frames: list[pd.DataFrame] = []
    for delay in delays:
        outcome = pd.concat(outcome_parts[delay], ignore_index=True)
        if outcome["candidate_id"].duplicated().any() or len(outcome) != len(common_routed):
            raise AssertionError(f"delay {delay}: replay outcome identity is incomplete")
        candidate, population = _portfolio_candidates(
            common_routed, outcome, arm=f"exact_1m_rich_delay_{delay}m"
        )
        metrics, monthly, _, accepted = _write_arm(
            output, f"exact_1m_rich_delay_{delay}m", candidate, population
        )
        metrics["entry_delay_minutes"] = int(delay)
        metrics["common_complete_candidates"] = int(len(common_routed))
        summary_rows.append(metrics)
        monthly_frames.append(monthly.assign(entry_delay_minutes=int(delay)))
        accepted_frames.append(accepted.assign(entry_delay_minutes=int(delay)))
        outcome.to_parquet(output / f"delay_{delay}m_exact_outcomes.parquet", index=False, compression="zstd")

    summary = pd.DataFrame(summary_rows).sort_values("entry_delay_minutes").reset_index(drop=True)
    baseline = summary.iloc[0]
    for column in (
        "portfolio_accepted_trades", "net_ev_bps_per_trade", "net_sum_bps",
        "portfolio_max_drawdown", "portfolio_sortino", "portfolio_worst_week_return",
    ):
        summary[f"delta_vs_{int(baseline.entry_delay_minutes)}m_{column}"] = (
            pd.to_numeric(summary[column], errors="coerce") - float(baseline[column])
        )
    summary.to_parquet(output / "delay_ladder_summary.parquet", index=False, compression="zstd")
    summary.to_csv(output / "delay_ladder_summary.csv", index=False)
    pd.concat(monthly_frames, ignore_index=True).to_parquet(
        output / "delay_ladder_monthly_metrics.parquet", index=False, compression="zstd"
    )
    pd.concat(accepted_frames, ignore_index=True).to_parquet(
        output / "delay_ladder_accepted_trades.parquet", index=False, compression="zstd"
    )
    pd.DataFrame(coverage).to_parquet(output / "delay_ladder_source_coverage.parquet", index=False, compression="zstd")
    pd.DataFrame(
        {
            "candidate_id": routed["candidate_id"].astype(str),
            "complete_all_requested_delays": routed["candidate_id"].astype(str).isin(common_id_set),
        }
    ).to_parquet(output / "delay_ladder_common_coverage_audit.parquet", index=False, compression="zstd")

    manifest = {
        "schema": "strict_r3_exact_1m_rich_entry_delay_ladder_v1",
        "target_free_route": route_audit,
        "frozen_policy": frozen_audit,
        "source": {
            "minute_root": str(Path(args.minute_root).resolve()),
            "exchange_observed_only": True,
            "causal_atr": "Wilder-14 after 100 complete prior hourly windows",
            "horizon_minutes": HORIZON_MINUTES,
        },
        "evaluation": {
            "entry_delays_minutes": list(delays),
            "common_complete_population": int(len(common_routed)),
            "target_free_routed_population": int(len(routed)),
            "coverage_rule": "intersection complete causal ATR + exact H12 paths across every tested delay; joined after routing",
            "portfolio": "canonical global auction; 7x, 80% margin, 10% slots, two entries/decision; BCF-MC1 priority only",
            "cost": "100 bps exactly once in frozen exact-1m rich engine",
        },
        "code_sha256": _sha256(Path(__file__).resolve()),
    }
    (output / "run_manifest.json").write_text(
        json.dumps(_json_safe(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", type=Path, default=DEFAULT_CANDIDATE_DIR)
    parser.add_argument("--frozen-policy", type=Path, default=DEFAULT_FROZEN_POLICY)
    parser.add_argument("--minute-root", type=Path, default=DEFAULT_MINUTE_ROOT)
    parser.add_argument("--delays", type=int, nargs="+", default=list(DEFAULT_DELAYS))
    parser.add_argument("--batch-rows", type=int, default=2500)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> None:
    print(run(parse_args()))


if __name__ == "__main__":
    main()
