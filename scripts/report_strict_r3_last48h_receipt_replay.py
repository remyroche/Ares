#!/usr/bin/env python3
"""Report a bounded no-order replay from immutable dual-admission receipts.

The script deliberately reuses the historically persisted, target-free score,
admission and portfolio decisions.  It joins frozen parent-policy labels only
after those decisions are fixed.  Consequently, it is safe to use as an
operational replay report and does not re-run model scoring, access Kraken, or
write to any live state.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.strict_r3_rich_policy import RichPolicyParams, simulate_rich_policy
KEEP = [
    "candidate_id", "__decision_ts__", "__symbol__", "final_score",
    "frozen_base_contract_complete", "base_route_timestamp_top30",
    "mc1_d2_expected_net_bps", "bcf_mc1_expected_net_bps",
    "bcf_mc1_available", "current_mc1_admitted_ge_30bps",
    "bcf_mc1_admitted_ge_30bps", "dual_bcf_current_admitted_ge_30bps",
    "dual_auction_priority_bps", "decision_open", "signal_atr",
    "policy_sl_atr", "policy_trailing_activation_atr",
    "policy_trailing_giveback_atr", "policy_timeout_hours",
    "policy_cost_bps_once", "portfolio_accepted",
    "portfolio_rejection_reason", "portfolio_priority_rank",
    "portfolio_open_positions_before", "portfolio_committed_margin_before",
    "portfolio_margin_cap", "shadow_action", "dual_admission_rejection_reason",
]
LABELS = [
    "candidate_id", "__decision_ts__", "__symbol__", "policy_path_valid",
    "policy_gross_bps", "policy_net_bps", "policy_exit_bar_15m",
    "policy_exit_reason", "policy_entry_price", "policy_exit_price",
    "policy_label_available_ts", "policy_cost_bps",
]


def _symbol_15m_path(symbol: str) -> Path:
    """Return the canonical local 15-minute source for a Kraken symbol."""
    return ROOT / "15m_ohlcv_perp" / f"{str(symbol).lower().replace('/', '')}_15m.parquet"


def _receipt_parent_policy_15m_labels(
    requested: pd.DataFrame,
    *,
    as_of: pd.Timestamp,
) -> pd.DataFrame:
    """Materialize only already-resolved parent-policy paths from receipts.

    The target-free decision/portfolio receipt is fixed *before* this function
    loads a future path.  The parent geometry and causal signal ATR are the
    values persisted on that same receipt.  This is deliberately a compact
    15-minute threshold-fill proxy for the stored SimplePolicyOptimiser parent
    (hard stop -> trailing -> H12 timeout), not a replacement for the live
    minute/VWAP monitor or an Adaptive Exit model.
    """
    columns = [
        "candidate_id", "__decision_ts__", "__symbol__", "signal_atr",
        "policy_sl_atr", "policy_trailing_activation_atr",
        "policy_trailing_giveback_atr", "policy_timeout_hours",
        "policy_cost_bps_once",
    ]
    frame = requested.loc[:, columns].copy()
    frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True)
    horizon = pd.to_timedelta(pd.to_numeric(frame["policy_timeout_hours"], errors="coerce"), unit="h")
    eligible = frame.loc[(frame["__decision_ts__"] + horizon).lt(as_of)].copy()
    out_columns = [
        "candidate_id", "__decision_ts__", "__symbol__", "policy_path_valid",
        "policy_gross_bps", "policy_net_bps", "policy_exit_bar_15m",
        "policy_exit_reason", "policy_entry_price", "policy_exit_price",
        "policy_label_available_ts", "policy_cost_bps", "policy_outcome_source",
    ]
    if eligible.empty:
        return pd.DataFrame(columns=out_columns)
    results: list[dict[str, object]] = []
    for symbol, group in eligible.groupby("__symbol__", sort=True):
        path = _symbol_15m_path(str(symbol))
        bars: pd.DataFrame | None = None
        if path.exists():
            try:
                bars = pd.read_parquet(path, columns=["open", "high", "low", "close"])
                bars.index = pd.to_datetime(bars.index, utc=True, errors="coerce")
                bars = bars.loc[bars.index.notna() & ~bars.index.duplicated(keep="last")].sort_index()
                bars = bars.apply(pd.to_numeric, errors="coerce")
            except Exception:
                bars = None
        for _, row in group.iterrows():
            common = {
                "candidate_id": str(row["candidate_id"]),
                "__decision_ts__": row["__decision_ts__"],
                "__symbol__": str(row["__symbol__"]),
                "policy_path_valid": False,
                "policy_gross_bps": np.nan,
                "policy_net_bps": np.nan,
                "policy_exit_bar_15m": -1,
                "policy_exit_reason": "incomplete_15m_h12_path_or_receipt_geometry",
                "policy_entry_price": np.nan,
                "policy_exit_price": np.nan,
                "policy_label_available_ts": row["__decision_ts__"] + pd.Timedelta(hours=float(row["policy_timeout_hours"])),
                "policy_cost_bps": np.nan,
                "policy_outcome_source": "receipt_parent_simple_policy_15m_threshold_fill",
            }
            values = [
                row["signal_atr"], row["policy_sl_atr"],
                row["policy_trailing_activation_atr"], row["policy_trailing_giveback_atr"],
                row["policy_timeout_hours"], row["policy_cost_bps_once"],
            ]
            if bars is None or not all(np.isfinite(pd.to_numeric(v, errors="coerce")) for v in values):
                results.append(common)
                continue
            timeout = int(round(float(row["policy_timeout_hours"]) * 4.0))
            decision = row["__decision_ts__"]
            index = bars.index.get_indexer([decision])[0]
            if index < 0 or index + timeout > len(bars):
                results.append(common)
                continue
            window = bars.iloc[index:index + timeout]
            array = window[["open", "high", "low", "close"]].to_numpy(dtype=float)
            if array.shape[0] != timeout or not np.isfinite(array).all() or np.any(array <= 0.0):
                results.append(common)
                continue
            entry = float(array[0, 0])
            atr = float(row["signal_atr"])
            stop = entry - float(row["policy_sl_atr"]) * atr
            activation = float(row["policy_trailing_activation_atr"]) * atr
            giveback = float(row["policy_trailing_giveback_atr"]) * atr
            if not np.isfinite(entry) or entry <= 0.0 or stop <= 0.0:
                results.append(common)
                continue
            max_favorable = 0.0
            armed = False
            exit_bar = timeout - 1
            exit_price = float(array[-1, 3])
            reason = "timeout_h12"
            # The frozen parent policy evaluates the prior bar's MFE to arm
            # trailing, then uses the current bar's low.  This cannot arm and
            # exit on the same 15-minute bar.
            for bar in range(1, timeout):
                high = float(array[bar, 1])
                low = float(array[bar, 2])
                if low <= stop:
                    exit_bar, exit_price, reason = bar, stop, "stop_loss"
                    break
                if max_favorable > activation:
                    armed = True
                if armed:
                    trail = entry + max(max_favorable - giveback, 0.0)
                    if low <= trail:
                        exit_bar, exit_price, reason = bar, trail, "trailing"
                        break
                max_favorable = max(max_favorable, high - entry)
            gross_bps = (exit_price / entry - 1.0) * 10_000.0
            cost_bps = float(row["policy_cost_bps_once"])
            common.update({
                "policy_path_valid": True,
                "policy_gross_bps": gross_bps,
                "policy_net_bps": gross_bps - cost_bps,
                "policy_exit_bar_15m": int(exit_bar),
                "policy_exit_reason": reason,
                "policy_entry_price": entry,
                "policy_exit_price": exit_price,
                "policy_cost_bps": cost_bps,
            })
            results.append(common)
    output = pd.DataFrame(results, columns=out_columns)
    if output["candidate_id"].duplicated().any() or len(output) != len(eligible):
        raise AssertionError("receipt parent-policy materialization changed candidate identity")
    good = output["policy_path_valid"].fillna(False).astype(bool)
    if good.any() and not np.allclose(
        output.loc[good, "policy_gross_bps"].to_numpy(float) - output.loc[good, "policy_net_bps"].to_numpy(float),
        output.loc[good, "policy_cost_bps"].to_numpy(float),
        rtol=0.0, atol=1e-8,
    ):
        raise AssertionError("receipt parent-policy cost was not applied exactly once")
    return output


def _receipt_rich_policy_15m_labels(
    requested: pd.DataFrame,
    *,
    as_of: pd.Timestamp,
    frozen_policy: Path,
) -> pd.DataFrame:
    """Materialize receipt-selected, H12-resolved rich-policy outcomes.

    This is deliberately outcome-only: the candidate identities, dual
    admission and common portfolio decisions have already been fixed in the
    immutable receipts.  Unlike the compact parent-policy diagnostic, it uses
    the sealed rich policy's hard stop, prior-bar smooth protection, trailing,
    fast-adverse rule and H12 timeout over the local 15-minute path.  It is
    still a 15-minute historical execution proxy, not the live one-minute
    VWAP/native-stop implementation.
    """
    payload = json.loads(frozen_policy.read_text(encoding="utf-8"))
    if not np.isclose(float(payload.get("cost_bps", np.nan)), 100.0):
        raise AssertionError("frozen rich policy must carry exactly one 100-bps cost")
    params = RichPolicyParams.from_mapping(dict(payload.get("params") or {}))
    if not bool(params.smooth_capital_protection_enabled):
        raise AssertionError("frozen rich policy is missing smooth capital protection")
    if not bool(params.adverse_exit_enabled):
        raise AssertionError("frozen rich policy is missing fast-adverse protection")
    median_atr = float(payload.get("median_atr_fraction_fitted_on_complete_2024_development", np.nan))
    if not np.isfinite(median_atr) or median_atr <= 0.0:
        raise AssertionError("frozen rich policy has invalid development ATR reference")
    columns = ["candidate_id", "__decision_ts__", "__symbol__", "signal_atr", "policy_timeout_hours"]
    frame = requested.loc[:, columns].copy()
    frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True)
    horizon = pd.to_timedelta(pd.to_numeric(frame["policy_timeout_hours"], errors="coerce"), unit="h")
    eligible = frame.loc[(frame["__decision_ts__"] + horizon).lt(as_of)].copy()
    out_columns = [
        "candidate_id", "__decision_ts__", "__symbol__", "policy_path_valid",
        "policy_gross_bps", "policy_net_bps", "policy_exit_bar_15m",
        "policy_exit_reason", "policy_entry_price", "policy_exit_price",
        "policy_label_available_ts", "policy_cost_bps", "policy_outcome_source",
    ]
    if eligible.empty:
        return pd.DataFrame(columns=out_columns)
    results: list[pd.DataFrame] = []
    for symbol, group in eligible.groupby("__symbol__", sort=True):
        path = _symbol_15m_path(str(symbol))
        bars: pd.DataFrame | None = None
        if path.exists():
            try:
                bars = pd.read_parquet(path, columns=["open", "high", "low", "close"])
                bars.index = pd.to_datetime(bars.index, utc=True, errors="coerce")
                bars = bars.loc[bars.index.notna() & ~bars.index.duplicated(keep="last")].sort_index()
                bars = bars.apply(pd.to_numeric, errors="coerce")
            except Exception:
                bars = None
        group = group.reset_index(drop=True)
        output = pd.DataFrame({
            "candidate_id": group["candidate_id"].astype(str),
            "__decision_ts__": group["__decision_ts__"],
            "__symbol__": group["__symbol__"].astype(str),
            "policy_path_valid": False,
            "policy_gross_bps": np.nan,
            "policy_net_bps": np.nan,
            "policy_exit_bar_15m": np.full(len(group), -1, dtype=np.int16),
            "policy_exit_reason": "incomplete_15m_h12_path_or_receipt_signal_atr",
            "policy_entry_price": np.nan,
            "policy_exit_price": np.nan,
            "policy_label_available_ts": group["__decision_ts__"] + pd.Timedelta(hours=12),
            "policy_cost_bps": np.nan,
            "policy_outcome_source": "receipt_frozen_rich_policy_15m_proxy",
        })
        if bars is None:
            results.append(output)
            continue
        timeout_values = pd.to_numeric(group["policy_timeout_hours"], errors="coerce")
        if not np.all(np.isfinite(timeout_values)) or not np.allclose(timeout_values, 12.0):
            raise AssertionError("receipt rich replay requires the frozen H12 horizon")
        positions = bars.index.get_indexer(pd.DatetimeIndex(group["__decision_ts__"]))
        offset = np.arange(48, dtype=np.int64)
        locations = positions[:, None] + offset[None, :]
        in_range = (positions >= 0) & (locations[:, -1] < len(bars))
        atr = pd.to_numeric(group["signal_atr"], errors="coerce").to_numpy(float)
        valid = in_range & np.isfinite(atr) & (atr > 0.0)
        for column in ("open", "high", "low", "close"):
            source = bars[column].to_numpy(float)
            present = np.flatnonzero(in_range)
            if len(present):
                valid[present] &= np.isfinite(source[locations[present]]).all(axis=1)
        if valid.any():
            selected = np.flatnonzero(valid)
            entry = bars["open"].to_numpy(float)[locations[selected, 0]]
            simulated = simulate_rich_policy(
                entry=entry,
                atr=atr[selected],
                highs=bars["high"].to_numpy(float)[locations[selected]],
                lows=bars["low"].to_numpy(float)[locations[selected]],
                closes=bars["close"].to_numpy(float)[locations[selected]],
                params=params,
                median_atr_fraction=median_atr,
                side="long",
            )
            if not np.asarray(simulated["path_valid"], dtype=bool).all():
                raise AssertionError(f"{symbol}: rich simulator rejected prevalidated paths")
            gross = np.asarray(simulated["gross_bps"], dtype=float)
            net = np.asarray(simulated["net_bps"], dtype=float)
            if not np.allclose(gross - net, 100.0, rtol=0.0, atol=1e-8):
                raise AssertionError(f"{symbol}: rich replay cost was not applied exactly once")
            output.loc[selected, "policy_path_valid"] = True
            output.loc[selected, "policy_gross_bps"] = gross
            output.loc[selected, "policy_net_bps"] = net
            output.loc[selected, "policy_exit_bar_15m"] = np.asarray(simulated["exit_bar"], dtype=np.int16)
            output.loc[selected, "policy_exit_reason"] = np.asarray(simulated["exit_reason"], dtype=object)
            output.loc[selected, "policy_entry_price"] = entry
            output.loc[selected, "policy_exit_price"] = entry * (1.0 + gross / 10_000.0)
            output.loc[selected, "policy_cost_bps"] = 100.0
        results.append(output)
    result = pd.concat(results, ignore_index=True).loc[:, out_columns]
    if len(result) != len(eligible) or result["candidate_id"].duplicated().any():
        raise AssertionError("receipt rich-policy materialization changed candidate identity")
    return result


def utc(value: str) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def source_rank(path: Path, rows: int) -> tuple[int, int, str]:
    value = str(path)
    if "successor_" in value and "_live_" in value:
        kind = 0
    elif "stateful_recovery" in value:
        kind = 1
    elif "feature_runtime_equivalence" in value:
        kind = 2
    elif "backfill" in value:
        kind = 3
    else:
        kind = 4
    return kind, -rows, value


def load_receipts(start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    chosen: dict[pd.Timestamp, tuple[tuple[int, int, str], Path]] = {}
    for path in ROOT.glob("data_perp/artifacts/**/cycle/shadow_decisions.parquet"):
        if "terminal" in str(path) or "parity" in str(path):
            continue
        # Avoid opening every historical parquet (many old artifacts are cloud
        # placeholders): derive the target decision hour from its immutable
        # receipt path and only then read the bounded report interval.
        text = str(path)
        hour_match = re.search(r"/hour_(\d{8}T\d{6})Z?/", text)
        stamps = re.findall(r"\d{8}T\d{6}", text)
        encoded = hour_match.group(1) if hour_match else (stamps[-1] if stamps else None)
        if encoded is None:
            continue
        timestamp = pd.to_datetime(encoded, format="%Y%m%dT%H%M%S", utc=True)
        if not start <= timestamp < end:
            continue
        try:
            timestamp_frame = pd.read_parquet(path, columns=["__decision_ts__"])
            observed = pd.to_datetime(timestamp_frame["__decision_ts__"].iloc[0], utc=True)
        except Exception:
            continue
        if observed != timestamp:
            continue
        proposal = (source_rank(path, len(timestamp_frame)), path)
        if timestamp not in chosen or proposal[0] < chosen[timestamp][0]:
            chosen[timestamp] = proposal
    expected = pd.date_range(start, end - pd.Timedelta(hours=1), freq="1h", tz="UTC")
    missing = [stamp.isoformat() for stamp in expected if stamp not in chosen]
    data: list[pd.DataFrame] = []
    audit: list[dict[str, object]] = []
    for timestamp, (_, path) in sorted(chosen.items()):
        frame = pd.read_parquet(path)
        absent = sorted(set(KEEP).difference(frame.columns))
        if absent:
            raise ValueError(f"{path} lacks required fields: {absent}")
        frame = frame.loc[:, KEEP].copy()
        frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True)
        if frame["__decision_ts__"].nunique() != 1 or frame["__decision_ts__"].iloc[0] != timestamp:
            raise ValueError(f"receipt timestamp mismatch: {path}")
        if frame["candidate_id"].duplicated().any():
            raise ValueError(f"duplicate candidate IDs: {path}")
        frame["source_receipt"] = str(path.relative_to(ROOT))
        data.append(frame)
        audit.append({"decision_ts": timestamp, "rows": len(frame), "source_receipt": str(path.relative_to(ROOT))})
    return pd.concat(data, ignore_index=True), pd.DataFrame(audit), missing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True, help="exclusive decision time")
    parser.add_argument("--as-of", required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument(
        "--materialize-receipt-parent-policy-15m", action="store_true",
        help="materialize already-resolved stored SimplePolicy parent paths from local 15m bars",
    )
    parser.add_argument(
        "--materialize-receipt-rich-policy-15m", action="store_true",
        help="materialize the exact frozen rich/smooth-policy 15m outcome proxy from receipt-selected paths",
    )
    parser.add_argument(
        "--frozen-rich-policy", type=Path,
        default=ROOT / "data_perp/artifacts/strict_r3_rich_policy_smooth_protection_long_20260817_v1/frozen_policy.json",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    start, end, as_of = utc(args.start), utc(args.end), utc(args.as_of)
    if args.out_dir.exists():
        raise FileExistsError(f"immutable output exists: {args.out_dir}")
    scores, coverage, missing = load_receipts(start, end)
    for column in ["base_route_timestamp_top30", "dual_bcf_current_admitted_ge_30bps", "portfolio_accepted"]:
        scores[column] = scores[column].fillna(False).astype(bool)
    labels = pd.read_parquet(args.labels, columns=LABELS).copy()
    labels["candidate_id"] = labels["candidate_id"].astype(str)
    labels["__decision_ts__"] = pd.to_datetime(labels["__decision_ts__"], utc=True)
    labels["policy_label_available_ts"] = pd.to_datetime(labels["policy_label_available_ts"], utc=True)
    labels = labels.loc[labels["policy_label_available_ts"].lt(as_of)].copy()
    if labels["candidate_id"].duplicated().any():
        labels = labels.sort_values("policy_label_available_ts", kind="stable").drop_duplicates("candidate_id", keep="last")
    scores["dual_admitted"] = scores["dual_bcf_current_admitted_ge_30bps"]
    accepted_receipts = scores.loc[scores["dual_admitted"] & scores["portfolio_accepted"]].copy()
    if args.materialize_receipt_parent_policy_15m and args.materialize_receipt_rich_policy_15m:
        raise ValueError("choose either the parent or rich receipt outcome materializer")
    if args.materialize_receipt_parent_policy_15m:
        parent_labels = _receipt_parent_policy_15m_labels(accepted_receipts, as_of=as_of)
        if not parent_labels.empty:
            labels = pd.concat([labels, parent_labels], ignore_index=True)
            # The receipt-derived outcome has exact match provenance for the
            # current period and therefore supersedes an older generic label.
            labels["_receipt_parent"] = labels["policy_outcome_source"].eq("receipt_parent_simple_policy_15m_threshold_fill")
            labels = labels.sort_values(["_receipt_parent", "policy_label_available_ts"], kind="stable").drop_duplicates("candidate_id", keep="last").drop(columns="_receipt_parent")
    if args.materialize_receipt_rich_policy_15m:
        rich_labels = _receipt_rich_policy_15m_labels(
            accepted_receipts, as_of=as_of, frozen_policy=args.frozen_rich_policy.resolve(),
        )
        if not rich_labels.empty:
            labels = pd.concat([labels, rich_labels], ignore_index=True)
            labels["_receipt_rich"] = labels["policy_outcome_source"].eq("receipt_frozen_rich_policy_15m_proxy")
            labels = labels.sort_values(["_receipt_rich", "policy_label_available_ts"], kind="stable").drop_duplicates("candidate_id", keep="last").drop(columns="_receipt_rich")
    scored = scores.merge(labels, on=["candidate_id", "__decision_ts__", "__symbol__"], how="left", validate="one_to_one")
    expected_available = scores["__decision_ts__"] + pd.to_timedelta(
        pd.to_numeric(scores["policy_timeout_hours"], errors="coerce").fillna(12.0), unit="h"
    )
    has_valid_path = scored["policy_path_valid"].fillna(False).astype(bool)
    path_should_be_resolved = expected_available.lt(as_of)
    scored["outcome_status"] = np.select(
        [
            ~scored["dual_admitted"],
            has_valid_path,
            path_should_be_resolved,
        ],
        ["not_dual_admitted", "resolved", "invalid_policy_path",],
        default="pending_h12",
    )
    accepted = scored.loc[scored["dual_admitted"] & scored["portfolio_accepted"]].copy()
    resolved = accepted.loc[accepted["outcome_status"].eq("resolved")].copy()
    pending = accepted.loc[~accepted["outcome_status"].eq("resolved")].copy()
    resolved["policy_exit_timestamp"] = resolved["__decision_ts__"] + pd.to_timedelta(
        (pd.to_numeric(resolved["policy_exit_bar_15m"], errors="coerce") + 1.0) * 15.0,
        unit="min",
    )
    per_trade = resolved.loc[:, [
        "candidate_id", "__decision_ts__", "__symbol__", "final_score", "bcf_mc1_expected_net_bps",
        "mc1_d2_expected_net_bps", "dual_auction_priority_bps", "portfolio_priority_rank",
        "decision_open", "policy_entry_price", "policy_exit_timestamp", "policy_exit_price",
        "policy_exit_reason", "policy_gross_bps", "policy_cost_bps", "policy_net_bps",
        "signal_atr", "policy_sl_atr", "policy_trailing_activation_atr", "policy_trailing_giveback_atr",
        "source_receipt",
    ]].copy()
    per_trade = per_trade.sort_values("__decision_ts__", kind="stable")
    if len(per_trade):
        net = pd.to_numeric(per_trade["policy_net_bps"], errors="coerce")
        aggregate = {
            "trades": int(len(per_trade)), "net_bps_sum": float(net.sum()), "net_bps_mean": float(net.mean()),
            "net_bps_median": float(net.median()), "hit_rate": float((net > 0.0).mean()),
            "wins": int((net > 0.0).sum()), "losses": int((net <= 0.0).sum()),
            "best_net_bps": float(net.max()), "worst_net_bps": float(net.min()),
        }
    else:
        aggregate = {"trades": 0}
    hourly = scored.groupby("__decision_ts__", as_index=False).agg(
        receipt_rows=("candidate_id", "size"),
        base_routed=("base_route_timestamp_top30", "sum"),
        dual_admitted=("dual_admitted", "sum"),
        portfolio_accepted=("portfolio_accepted", lambda x: int((x & scored.loc[x.index, "dual_admitted"]).sum())),
    )
    if len(per_trade):
        daily = (
            per_trade.assign(date=pd.to_datetime(per_trade["__decision_ts__"], utc=True).dt.strftime("%Y-%m-%d"))
            .groupby("date", as_index=False)
            .agg(
                resolved_trades=("candidate_id", "size"),
                net_bps_sum=("policy_net_bps", "sum"),
                net_bps_mean=("policy_net_bps", "mean"),
                net_bps_median=("policy_net_bps", "median"),
                wins=("policy_net_bps", lambda values: int((values > 0.0).sum())),
                best_net_bps=("policy_net_bps", "max"),
                worst_net_bps=("policy_net_bps", "min"),
            )
        )
        daily["hit_rate"] = daily["wins"] / daily["resolved_trades"]
    else:
        daily = pd.DataFrame(columns=["date", "resolved_trades", "net_bps_sum", "net_bps_mean", "net_bps_median", "wins", "best_net_bps", "worst_net_bps", "hit_rate"])
    args.out_dir.mkdir(parents=True)
    scores.to_parquet(args.out_dir / "receipt_backed_scores.parquet", index=False, compression="zstd")
    coverage.to_parquet(args.out_dir / "receipt_coverage.parquet", index=False, compression="zstd")
    scored.to_parquet(args.out_dir / "receipt_backed_score_outcome_ledger.parquet", index=False, compression="zstd")
    per_trade.to_parquet(args.out_dir / "per_trade_metrics.parquet", index=False, compression="zstd")
    per_trade.to_csv(args.out_dir / "per_trade_metrics.csv", index=False)
    pending.to_parquet(args.out_dir / "pending_portfolio_accepted.parquet", index=False, compression="zstd")
    pending.to_csv(args.out_dir / "pending_or_invalid_portfolio_accepted.csv", index=False)
    hourly.to_parquet(args.out_dir / "hourly_funnel.parquet", index=False, compression="zstd")
    hourly.to_csv(args.out_dir / "hourly_funnel.csv", index=False)
    daily.to_parquet(args.out_dir / "daily_resolved_trade_metrics.parquet", index=False, compression="zstd")
    daily.to_csv(args.out_dir / "daily_resolved_trade_metrics.csv", index=False)
    manifest = {
        "schema": "strict_r3_last48h_receipt_backed_replay_v1",
        "purpose": "offline replay report from immutable target-free dual admission and portfolio receipts",
        "exchange_calls": 0,
        "order_submission_enabled": False,
        "range": {"start": start.isoformat(), "end_exclusive": end.isoformat(), "as_of": as_of.isoformat()},
        "coverage": {"expected_hours": int(len(pd.date_range(start, end - pd.Timedelta(hours=1), freq="1h"))), "covered_hours": int(len(coverage)), "missing_hours": missing},
        "contract": {
            "admission": "BCF MC1 >= +30 bps AND current-v5 MC1 >= +30 bps",
            "priority": "BCF MC1 expected bps",
            "portfolio": "historically persisted common portfolio auction state and constraints",
            "outcome": (
                "frozen rich/smooth SimplePolicyOptimiser extension replayed on local 15m paths for receipt-selected, H12-resolved rows; "
                "hard stop, prior-bar smooth capital protection, trailing, fast-adverse, H12 timeout and cost once; "
                "no live one-minute VWAP/native-stop or Adaptive Exit V1 overlay"
                if args.materialize_receipt_rich_policy_15m else
                "stored SimplePolicyOptimiser parent geometry replayed on local 15m paths for receipt-selected, H12-resolved rows; cost once; no live VWAP, execution-delay or Adaptive Exit V1 overlay"
            ),
        },
        "funnel": {
            "target_free_rows": int(len(scored)), "base_routed": int(scored["base_route_timestamp_top30"].sum()),
            "dual_admitted": int(scored["dual_admitted"].sum()), "portfolio_accepted": int(len(accepted)),
            "resolved_accepted": int(len(resolved)),
            "pending_h12_accepted": int((pending["outcome_status"] == "pending_h12").sum()),
            "invalid_policy_path_accepted": int((pending["outcome_status"] == "invalid_policy_path").sum()),
        },
        "aggregate": aggregate,
    }
    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps({"event": "complete", "out_dir": str(args.out_dir), **manifest["funnel"], **aggregate}, sort_keys=True))


if __name__ == "__main__":
    main()
