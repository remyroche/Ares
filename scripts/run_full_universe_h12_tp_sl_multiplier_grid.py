#!/usr/bin/env python3
"""Replay the frozen full-universe score through a 7x7 H12 barrier grid.

This is an *execution-label* ablation.  It never refits, recalibrates, or
reranks the base/meta stack: one deterministic global top-20% score prefix is
fixed first, then each selected path is replayed under every TP/SL pair.
The four reported top-k prefixes are therefore identical candidate sets for
all 49 geometries.  TP/SL are independent multipliers of the training label
contract (TP=3 ATR, SL=2 ATR); timeout remains exactly 720 one-minute bars.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "data_perp/artifacts/full_universe_t2_t4_panel_20260801_v3"
SCORES = ROOT / "data_perp/artifacts/full_universe_stage5_2_conditional_overlay_20260804_v1/oos_integrated_predictions.parquet"
MINUTE_ROOT = ROOT / "data_perp/exchanges/krakenfutures/execution_1m/ohlcv"
MULTIPLIERS = (0.50, 0.66, 0.75, 1.00, 1.25, 1.50, 2.00)
TOP_FRACTIONS = (0.01, 0.05, 0.10, 0.20)
HORIZON_MINUTES = 720
ROUND_TRIP_COST_BPS = 100.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _load_selection(scores_path: Path, top_fraction: float) -> tuple[pd.DataFrame, int]:
    score = pd.read_parquet(scores_path, columns=["candidate_id", "__ts__", "side_name", "selected_score"])
    score["__ts__"] = pd.to_datetime(score["__ts__"], utc=True, errors="raise")
    score["selected_score"] = pd.to_numeric(score["selected_score"], errors="coerce")
    if score["candidate_id"].duplicated().any() or not np.isfinite(score["selected_score"]).all():
        raise ValueError("frozen score population must have one finite score per candidate")
    score = score.sort_values(["selected_score", "candidate_id"], ascending=[False, True], kind="stable").reset_index(drop=True)
    score["global_rank"] = np.arange(1, len(score) + 1, dtype=np.int64)
    selected_rows = int(np.ceil(top_fraction * len(score)))
    return score.iloc[:selected_rows].copy(), int(len(score))


def _load_panel_rows(panel: Path, candidate_ids: set[str]) -> pd.DataFrame:
    columns = [
        "candidate_id", "__ts__", "__symbol__", "side_name", "__decision_ts__",
        "atr_1h", "decision_price", "assumed_round_trip_cost_bps",
        "t4_tp3_sl2_gross_bps", "t4_tp3_sl2_net_bps",
    ]
    frames: list[pd.DataFrame] = []
    for path in sorted((panel / "parts").glob("*.parquet")):
        frame = pd.read_parquet(path, columns=columns)
        frame = frame.loc[frame["candidate_id"].isin(candidate_ids)]
        if not frame.empty:
            frames.append(frame)
    if not frames:
        raise ValueError("none of the frozen-score candidates joined to the label panel")
    result = pd.concat(frames, ignore_index=True)
    if result["candidate_id"].duplicated().any():
        raise ValueError("panel candidate identity is not unique")
    result["__decision_ts__"] = pd.to_datetime(result["__decision_ts__"], utc=True, errors="raise")
    numeric = ["atr_1h", "decision_price", "assumed_round_trip_cost_bps", "t4_tp3_sl2_gross_bps", "t4_tp3_sl2_net_bps"]
    for column in numeric:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    if not np.isfinite(result[numeric].to_numpy(float)).all() or (result["atr_1h"] <= 0).any() or (result["decision_price"] <= 0).any():
        raise ValueError("selected panel paths have invalid entry or ATR inputs")
    if not np.allclose(result["assumed_round_trip_cost_bps"], ROUND_TRIP_COST_BPS, rtol=0.0, atol=1e-6):
        raise ValueError("this ablation is bound to the panel's 100-bps cost contract")
    return result


def _load_minute(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    root = MINUTE_ROOT / f"symbol={symbol}"
    if not root.exists():
        raise FileNotFoundError(root)
    years = list(range(start.year, (end - pd.Timedelta(minutes=1)).year + 1))
    table = ds.dataset(root, format="parquet", partitioning="hive").to_table(
        filter=(ds.field("year").isin(years)) & (ds.field("ts") >= start) & (ds.field("ts") < end),
        columns=["ts", "high", "low", "close"],
    )
    frame = table.to_pandas()
    frame["ts"] = pd.to_datetime(frame["ts"], utc=True, errors="raise")
    return frame.drop_duplicates("ts", keep="last").set_index("ts").sort_index()


def _replay_batch(rows: pd.DataFrame, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return gross/net bps and event codes for every row x independent grid.

    The implementation is vectorised across a batch.  It computes first-hit
    times only for the seven TP and seven SL levels (not 49 full simulations),
    then combines those times with the panel's adverse-tie rule.
    """
    entry = rows["decision_price"].to_numpy(float)
    atr = rows["atr_1h"].to_numpy(float)
    side = np.where(rows["side_name"].astype(str).eq("long"), 1.0, -1.0)
    if (side == 0).any():
        raise ValueError("invalid side")
    favorable = np.where(side[:, None] > 0.0, (high - entry[:, None]) / atr[:, None], (entry[:, None] - low) / atr[:, None])
    adverse = np.where(side[:, None] > 0.0, (entry[:, None] - low) / atr[:, None], (high - entry[:, None]) / atr[:, None])
    step = np.arange(HORIZON_MINUTES, dtype=np.int16)[None, :, None]
    tp_levels = np.asarray(MULTIPLIERS, dtype=float) * 3.0
    sl_levels = np.asarray(MULTIPLIERS, dtype=float) * 2.0
    tp_time = np.where(favorable[:, :, None] >= tp_levels[None, None, :], step, HORIZON_MINUTES).min(axis=1)
    sl_time = np.where(adverse[:, :, None] >= sl_levels[None, None, :], step, HORIZON_MINUTES).min(axis=1)
    tp_grid = tp_time[:, :, None]
    sl_grid = sl_time[:, None, :]
    stop = (sl_grid <= tp_grid) & (sl_grid < HORIZON_MINUTES)
    take = (tp_grid < sl_grid) & (tp_grid < HORIZON_MINUTES)
    timeout = ~(stop | take)
    timeout_pnl = side[:, None, None] * (
        close[:, -1][:, None, None] - entry[:, None, None]
    ) / atr[:, None, None]
    pnl_atr = np.where(
        stop,
        -sl_levels[None, None, :],
        np.where(take, tp_levels[None, :, None], timeout_pnl),
    )
    event = np.where(stop, 1, np.where(take, 0, 2)).astype(np.int8)
    gross = pnl_atr * (atr / entry)[:, None, None] * 10_000.0
    net = gross - ROUND_TRIP_COST_BPS
    if not np.isfinite(gross).all() or not np.isfinite(net).all() or timeout.shape != event.shape:
        raise ValueError("non-finite vectorised replay output")
    return gross.astype(np.float32), net.astype(np.float32), event


def _replay(rows: pd.DataFrame, batch_rows: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    shape = (len(rows), len(MULTIPLIERS), len(MULTIPLIERS))
    gross = np.full(shape, np.nan, dtype=np.float32)
    net = np.full(shape, np.nan, dtype=np.float32)
    event = np.full(shape, -1, dtype=np.int8)
    for count, (symbol, positions) in enumerate(rows.groupby("__symbol__", sort=True).groups.items(), start=1):
        loc = np.asarray(list(positions), dtype=np.int64)
        symbol_rows = rows.loc[loc].copy().reset_index(drop=True)
        start = symbol_rows["__decision_ts__"].min()
        end = symbol_rows["__decision_ts__"].max() + pd.Timedelta(minutes=HORIZON_MINUTES)
        bars = _load_minute(str(symbol), start, end)
        grid = pd.date_range(start, end, freq="min", inclusive="left", tz="UTC")
        values = bars.reindex(grid)[["high", "low", "close"]].to_numpy(dtype=np.float64)
        offsets = ((symbol_rows["__decision_ts__"] - start) / pd.Timedelta(minutes=1)).astype(np.int64).to_numpy()
        for begin in range(0, len(symbol_rows), batch_rows):
            end_batch = min(begin + batch_rows, len(symbol_rows))
            path_offsets = offsets[begin:end_batch]
            paths = np.stack([values[offset: offset + HORIZON_MINUTES] for offset in path_offsets])
            if paths.shape[1:] != (HORIZON_MINUTES, 3):
                raise ValueError(f"invalid one-minute path shape for {symbol}")
            complete = np.isfinite(paths).all(axis=(1, 2)) & (paths > 0.0).all(axis=(1, 2))
            if not complete.any():
                continue
            g, n, e = _replay_batch(symbol_rows.iloc[begin:end_batch].iloc[np.flatnonzero(complete)], paths[complete, :, 0], paths[complete, :, 1], paths[complete, :, 2])
            target = loc[begin:end_batch][complete]
            gross[target], net[target], event[target] = g, n, e
        if count == 1 or count % 20 == 0:
            print(json.dumps({"event": "symbol_complete", "symbols": count, "rows": int(np.isfinite(net[:, 0, 0]).sum())}), flush=True)
    complete = np.isfinite(net).all(axis=(1, 2)) & np.isfinite(gross).all(axis=(1, 2)) & (event >= 0).all(axis=(1, 2))
    if not complete.any():
        raise ValueError("no selected paths have a complete exact 12-hour window")
    return gross, net, event, complete


def _metrics(rows: pd.DataFrame, gross: np.ndarray, net: np.ndarray, event: np.ndarray, path_complete: np.ndarray, total_scored_rows: int, tp_index: int, sl_index: int, fraction: float) -> list[dict[str, Any]]:
    # `rows` is deliberately only the outer top-20% replay prefix.  Top-k is
    # nevertheless defined against the *full* frozen global score population.
    limit = int(np.ceil(fraction * total_scored_rows))
    selected = np.flatnonzero(path_complete & (rows["global_rank"].to_numpy() <= limit))
    out: list[dict[str, Any]] = []
    for name, position in (("all", selected), ("long", selected[rows.iloc[selected]["side_name"].eq("long").to_numpy()]), ("short", selected[rows.iloc[selected]["side_name"].eq("short").to_numpy()])):
        values = net[position, tp_index, sl_index]
        exits = event[position, tp_index, sl_index]
        out.append({"top_fraction": fraction, "side": name, "n": int(len(position)), "gross_bps": float(gross[position, tp_index, sl_index].mean()) if len(position) else None, "net_bps": float(values.mean()) if len(position) else None, "sum_net_bps": float(values.sum()) if len(position) else None, "positive_rate": float((values > 0.0).mean()) if len(position) else None, "take_profit_rate": float((exits == 0).mean()) if len(position) else None, "stop_loss_rate": float((exits == 1).mean()) if len(position) else None, "timeout_rate": float((exits == 2).mean()) if len(position) else None})
    return out


def main() -> None:
    global MULTIPLIERS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, default=SCORES)
    parser.add_argument("--panel", type=Path, default=PANEL)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--top-fraction", type=float, default=0.20)
    parser.add_argument("--batch-rows", type=int, default=256)
    parser.add_argument("--multipliers", default=",".join(str(x) for x in MULTIPLIERS), help="comma-separated independent TP/SL multipliers")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite {args.out}")
    if not 0.0 < args.top_fraction <= 1.0:
        raise ValueError("top fraction must be in (0, 1]")
    MULTIPLIERS = tuple(float(value) for value in args.multipliers.split(","))
    if not MULTIPLIERS or any(value <= 0.0 for value in MULTIPLIERS):
        raise ValueError("multipliers must be positive")
    args.out.mkdir(parents=True)
    selected, all_rows = _load_selection(args.scores, args.top_fraction)
    panel = _load_panel_rows(args.panel, set(selected["candidate_id"]))
    rows = selected.merge(panel, on=["candidate_id", "__ts__", "side_name"], how="left", validate="one_to_one")
    if len(rows) != len(selected) or rows["__symbol__"].isna().any():
        raise ValueError("frozen global selection did not completely join the exact H12 panel")
    gross, net, event, path_complete = _replay(rows, int(args.batch_rows))
    if 1.00 in MULTIPLIERS:
        parent = MULTIPLIERS.index(1.00)
        parity = net[path_complete, parent, parent].astype(float) - rows.loc[path_complete, "t4_tp3_sl2_net_bps"].to_numpy(float)
        max_abs = float(np.max(np.abs(parity)))
        if max_abs > 2e-3:
            raise ValueError(f"TP=1x / SL=1x parity failed against frozen TP3/SL2 labels: {max_abs}")
        parity_report: dict[str, float | None] = {"max_abs_net_bps_delta": max_abs, "mean_abs_net_bps_delta": float(np.mean(np.abs(parity)))}
    else:
        parity_report = {"max_abs_net_bps_delta": None, "mean_abs_net_bps_delta": None}
    metrics: list[dict[str, Any]] = []
    for tp_i, tp in enumerate(MULTIPLIERS):
        for sl_i, sl in enumerate(MULTIPLIERS):
            for fraction in TOP_FRACTIONS:
                for item in _metrics(rows, gross, net, event, path_complete, all_rows, tp_i, sl_i, fraction):
                    item.update({"tp_multiplier": tp, "sl_multiplier": sl, "tp_atr": 3.0 * tp, "sl_atr": 2.0 * sl})
                    metrics.append(item)
    metric_frame = pd.DataFrame(metrics)
    metric_frame.to_csv(args.out / "grid_metrics.csv", index=False)
    compact = rows[["candidate_id", "__ts__", "__symbol__", "side_name", "selected_score", "global_rank"]].copy()
    compact["exact_h12_path_complete"] = path_complete
    for tp_i, tp in enumerate(MULTIPLIERS):
        for sl_i, sl in enumerate(MULTIPLIERS):
            key = f"tp{tp:g}_sl{sl:g}".replace(".", "p")
            compact[f"net_bps__{key}"] = net[:, tp_i, sl_i]
    compact.to_parquet(args.out / "selected_paths_all_49_geometries.parquet", index=False, compression="zstd")
    selected_top10 = metric_frame[(metric_frame.top_fraction == 0.10) & (metric_frame.side == "all")].sort_values(["net_bps", "gross_bps"], ascending=False, kind="stable")
    winners_by_tail = {}
    for fraction in TOP_FRACTIONS:
        ranked = metric_frame[(metric_frame.top_fraction == fraction) & (metric_frame.side == "all")].sort_values(["net_bps", "gross_bps"], ascending=False, kind="stable")
        winners_by_tail[str(fraction)] = ranked.iloc[0].to_dict()
    summary = {"schema": "full_universe_h12_tp_sl_multiplier_grid_v1", "status": "COMPLETED_FROZEN_SCORE_EXECUTION_ABLATION", "contract": {"scores": "frozen Stage5.2 conditional-value + residual + reliability rank blend; no refit/recalibration/reranking", "candidate_selection": f"one deterministic global score prefix, top {args.top_fraction:.0%}; smaller top-k metrics are nested prefixes", "entry": "panel decision timestamp = signal close + one hour; exact next one-minute open", "barriers": "TP=3 ATR x tp_multiplier; SL=2 ATR x sl_multiplier; independent grid", "multipliers": list(MULTIPLIERS), "timeout_minutes": HORIZON_MINUTES, "same_minute_conflict": "adverse SL precedence", "cost": "fixed 100 bps round trip; no synthetic spread", "portfolio_constraints": "not applied; candidate-local label replay"}, "population": {"all_scored_rows": all_rows, "global_prefix_rows": len(rows), "exact_h12_complete_rows": int(path_complete.sum()), "exact_h12_coverage": float(path_complete.mean()), "first_ts": rows.__ts__.min(), "last_ts": rows.__ts__.max()}, "parity_tp1_sl1_complete_rows_only": parity_report, "winning_geometry_by_global_tail": winners_by_tail, "best_top10_by_net": selected_top10.head(10).to_dict(orient="records"), "sources": {"scores": {"path": args.scores, "sha256": _sha256(args.scores)}, "panel_manifest": {"path": args.panel / "manifest.json", "sha256": _sha256(args.panel / "manifest.json")}}}
    _write_json(args.out / "summary.json", summary)
    print(json.dumps({"parity": summary["parity_tp1_sl1_complete_rows_only"], "best_top10_by_net": summary["best_top10_by_net"][:3]}, indent=2))


if __name__ == "__main__":
    main()
