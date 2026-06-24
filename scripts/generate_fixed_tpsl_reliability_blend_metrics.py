#!/usr/bin/env python3
"""Generate fixed TP/SL metrics for reliability blend scores.

This audit recomputes first-touch triple-barrier outcomes from hourly OHLCV
using a fixed TP/SL pair, then evaluates baseline anchor scores versus the
selected reliability blend on recent OOF windows.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from sklearn.metrics import roc_auc_score
except Exception:  # pragma: no cover - sklearn is available in the project env.
    roc_auc_score = None


OUT_SL = 0
OUT_TO = 1
OUT_TP = 2

DEFAULT_REPORT_DIR = Path("data_perp/reports/reliability_blend_optuna_20260623_native_lgbm_only_50k")
DEFAULT_CONFIG_PATH = Path("config/reliability_blend_default_configs.json")
DEFAULT_OHLCV_ROOT = Path("data_perp/exchanges/krakenfutures/ohlcv")


@dataclass(frozen=True)
class FixedBarrierConfig:
    tp: float
    sl: float
    horizon_hours: float


def _safe_auc(y: np.ndarray, score: np.ndarray) -> float:
    mask = np.isfinite(y) & np.isfinite(score)
    if int(mask.sum()) < 2:
        return float("nan")
    yy = y[mask].astype(int)
    if len(np.unique(yy)) < 2:
        return float("nan")
    if roc_auc_score is None:
        return float("nan")
    return float(roc_auc_score(yy, score[mask]))


def _symbol_to_kraken_key(symbol: str) -> str:
    return str(symbol).replace("/", "_")


def _load_ohlcv_for_symbol(
    symbol: str,
    *,
    ohlcv_root: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame | None:
    key = _symbol_to_kraken_key(symbol)
    symbol_dir = ohlcv_root / f"symbol={key}"
    if not symbol_dir.exists():
        return None

    years = range(int(start.year), int(end.year) + 1)
    frames: list[pd.DataFrame] = []
    cols = ["ts", "open", "high", "low", "close"]
    for year in years:
        year_path = symbol_dir / f"year={year}"
        if not year_path.exists():
            continue
        try:
            frame = pd.read_parquet(year_path, columns=cols)
        except Exception:
            try:
                frame = pd.read_parquet(year_path)
                frame = frame[[c for c in cols if c in frame.columns]]
            except Exception:
                continue
        if not {"ts", "high", "low", "close"}.issubset(frame.columns):
            continue
        frames.append(frame)

    if not frames:
        return None

    out = pd.concat(frames, ignore_index=True)
    out["ts"] = pd.to_datetime(out["ts"], utc=True, errors="coerce")
    out = out.dropna(subset=["ts", "high", "low", "close"])
    out = out[(out["ts"] >= start - pd.Timedelta(hours=1)) & (out["ts"] <= end + pd.Timedelta(hours=1))]
    if out.empty:
        return None
    out = out.sort_values("ts").drop_duplicates("ts", keep="last").reset_index(drop=True)
    for col in ("open", "high", "low", "close"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")
    return out


def _label_symbol_rows(
    ohlcv: pd.DataFrame,
    rows: pd.DataFrame,
    *,
    side: str,
    cfg: FixedBarrierConfig,
) -> pd.DataFrame:
    times_ns = ohlcv["ts"].to_numpy(dtype="datetime64[ns]").astype("int64")
    highs = ohlcv["high"].to_numpy(dtype="float64", copy=False)
    lows = ohlcv["low"].to_numpy(dtype="float64", copy=False)
    closes = ohlcv["close"].to_numpy(dtype="float64", copy=False)
    ts_to_idx = {int(v): i for i, v in enumerate(times_ns)}
    horizon_ns = int(float(cfg.horizon_hours) * 3600 * 1_000_000_000)
    is_long = side == "long"

    out_rows: list[dict[str, Any]] = []
    for row in rows.itertuples(index=False):
        row_ts = pd.Timestamp(row.timestamp).tz_convert("UTC")
        row_ns = int(row_ts.to_datetime64().astype("datetime64[ns]").astype("int64"))
        i = ts_to_idx.get(row_ns)
        if i is None:
            out_rows.append(
                {
                    "head": row.head,
                    "row_id": int(row.row_id),
                    "timestamp": row.timestamp,
                    "symbol": row.symbol,
                    "fixed_outcome": np.nan,
                    "fixed_y_tp": np.nan,
                    "fixed_return": np.nan,
                    "fixed_conflict_same_bar": False,
                    "fixed_missing_reason": "timestamp_missing",
                }
            )
            continue

        entry = closes[i]
        if not np.isfinite(entry) or entry <= 0:
            out_rows.append(
                {
                    "head": row.head,
                    "row_id": int(row.row_id),
                    "timestamp": row.timestamp,
                    "symbol": row.symbol,
                    "fixed_outcome": np.nan,
                    "fixed_y_tp": np.nan,
                    "fixed_return": np.nan,
                    "fixed_conflict_same_bar": False,
                    "fixed_missing_reason": "bad_entry",
                }
            )
            continue

        cutoff = row_ns + horizon_ns
        j_start = i + 1
        j_end = int(np.searchsorted(times_ns, cutoff, side="right"))
        if j_end <= j_start:
            out_rows.append(
                {
                    "head": row.head,
                    "row_id": int(row.row_id),
                    "timestamp": row.timestamp,
                    "symbol": row.symbol,
                    "fixed_outcome": OUT_TO,
                    "fixed_y_tp": 0,
                    "fixed_return": 0.0,
                    "fixed_conflict_same_bar": False,
                    "fixed_missing_reason": "",
                }
            )
            continue

        if is_long:
            tp_price = entry * (1.0 + cfg.tp)
            sl_price = entry * (1.0 - cfg.sl)
        else:
            tp_price = entry * (1.0 - cfg.tp)
            sl_price = entry * (1.0 + cfg.sl)

        outcome = OUT_TO
        fixed_ret = 0.0
        conflict = False
        last_close = closes[min(j_end - 1, len(closes) - 1)]

        for j in range(j_start, min(j_end, len(closes))):
            hh = highs[j]
            ll = lows[j]
            cc = closes[j]
            if not (np.isfinite(hh) and np.isfinite(ll)):
                if np.isfinite(cc):
                    last_close = cc
                continue
            if is_long:
                hit_tp = hh >= tp_price
                hit_sl = ll <= sl_price
            else:
                hit_tp = ll <= tp_price
                hit_sl = hh >= sl_price

            # Match the conservative convention used by the project labeler for same-bar conflicts.
            if hit_tp and hit_sl:
                outcome = OUT_SL
                fixed_ret = -float(cfg.sl)
                conflict = True
                break
            if hit_sl:
                outcome = OUT_SL
                fixed_ret = -float(cfg.sl)
                break
            if hit_tp:
                outcome = OUT_TP
                fixed_ret = float(cfg.tp)
                break
            if np.isfinite(cc):
                last_close = cc
        else:
            if np.isfinite(last_close) and last_close > 0:
                fixed_ret = (last_close / entry - 1.0) if is_long else (entry / last_close - 1.0)

        out_rows.append(
            {
                "head": row.head,
                "row_id": int(row.row_id),
                "timestamp": row.timestamp,
                "symbol": row.symbol,
                "fixed_outcome": int(outcome),
                "fixed_y_tp": int(outcome == OUT_TP),
                "fixed_return": float(fixed_ret),
                "fixed_conflict_same_bar": bool(conflict),
                "fixed_missing_reason": "",
            }
        )

    return pd.DataFrame(out_rows)


def _load_default_variants(config_path: Path) -> dict[str, str]:
    payload = json.loads(config_path.read_text())
    configs = payload.get("configs", {})
    return {str(head): str(cfg["variant"]) for head, cfg in configs.items()}


def _head_side(head: str) -> str:
    return "short" if str(head).startswith("short") else "long"


def _top_fraction_metrics(frame: pd.DataFrame, score_col: str, y_col: str, ret_col: str, frac: float) -> dict[str, float]:
    scored = frame[np.isfinite(frame[score_col].to_numpy(dtype="float64"))].copy()
    if scored.empty:
        return {"hr": float("nan"), "n": 0, "mean_return": float("nan"), "sl_rate": float("nan"), "timeout_rate": float("nan")}
    scored = scored.sort_values(score_col, ascending=False)
    k = max(1, int(math.ceil(float(frac) * len(scored))))
    top = scored.head(k)
    return {
        "hr": float(top[y_col].mean()),
        "n": int(len(top)),
        "mean_return": float(top[ret_col].mean()),
        "sl_rate": float((top["fixed_outcome"] == OUT_SL).mean()),
        "timeout_rate": float((top["fixed_outcome"] == OUT_TO).mean()),
    }


def _compute_window_metrics(
    scores: pd.DataFrame,
    variants: dict[str, str],
    cfg: FixedBarrierConfig,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for head, group in scores.groupby("head", sort=True):
        variant = variants.get(str(head))
        if not variant:
            continue
        current_col = f"blend_{variant}_score"
        if current_col not in group.columns:
            continue
        max_ts = group["timestamp"].max()
        for days in (7, 14, 28):
            start = max_ts - pd.Timedelta(days=days)
            w = group[(group["timestamp"] >= start) & (group["timestamp"] <= max_ts)].copy()
            total_rows = int(len(w))
            w = w[np.isfinite(w["fixed_y_tp"].to_numpy(dtype="float64"))].copy()
            labeled_rows = int(len(w))
            row: dict[str, Any] = {
                "head": head,
                "window": f"last_{days // 7}w",
                "start": start,
                "end": max_ts,
                "rows_total": total_rows,
                "rows_labeled_exact": labeled_rows,
                "coverage": float(labeled_rows / total_rows) if total_rows else float("nan"),
                "current_variant": variant,
                "fixed_tp": float(cfg.tp),
                "fixed_sl": float(cfg.sl),
                "fixed_horizon_hours": float(cfg.horizon_hours),
            }
            if w.empty:
                rows.append(row)
                continue
            y = w["fixed_y_tp"].to_numpy(dtype="float64")
            row["base_auc"] = _safe_auc(y, w["anchor_score"].to_numpy(dtype="float64"))
            row["blend_auc"] = _safe_auc(y, w[current_col].to_numpy(dtype="float64"))
            row["delta_auc"] = row["blend_auc"] - row["base_auc"] if np.isfinite(row["base_auc"]) and np.isfinite(row["blend_auc"]) else float("nan")
            row["all_tp_rate"] = float((w["fixed_outcome"] == OUT_TP).mean())
            row["all_sl_rate"] = float((w["fixed_outcome"] == OUT_SL).mean())
            row["all_timeout_rate"] = float((w["fixed_outcome"] == OUT_TO).mean())
            row["all_conflict_rate"] = float(w["fixed_conflict_same_bar"].mean())
            row["all_mean_return"] = float(w["fixed_return"].mean())
            for frac, label in ((0.10, "10"), (0.20, "20"), (0.30, "30")):
                b = _top_fraction_metrics(w, "anchor_score", "fixed_y_tp", "fixed_return", frac)
                c = _top_fraction_metrics(w, current_col, "fixed_y_tp", "fixed_return", frac)
                row[f"base_hr{label}"] = b["hr"]
                row[f"blend_hr{label}"] = c["hr"]
                row[f"delta_hr{label}"] = c["hr"] - b["hr"] if np.isfinite(b["hr"]) and np.isfinite(c["hr"]) else float("nan")
                row[f"base_top{label}_n"] = b["n"]
                row[f"blend_top{label}_n"] = c["n"]
                row[f"base_top{label}_mean_return"] = b["mean_return"]
                row[f"blend_top{label}_mean_return"] = c["mean_return"]
                row[f"delta_top{label}_mean_return"] = (
                    c["mean_return"] - b["mean_return"] if np.isfinite(b["mean_return"]) and np.isfinite(c["mean_return"]) else float("nan")
                )
                row[f"base_top{label}_sl_rate"] = b["sl_rate"]
                row[f"blend_top{label}_sl_rate"] = c["sl_rate"]
            rows.append(row)
    return pd.DataFrame(rows)


def _format_metric_table(metrics: pd.DataFrame) -> str:
    cols = [
        "head",
        "window",
        "rows_labeled_exact",
        "coverage",
        "base_auc",
        "blend_auc",
        "delta_auc",
        "base_hr10",
        "blend_hr10",
        "delta_hr10",
        "base_hr20",
        "blend_hr20",
        "delta_hr20",
        "base_hr30",
        "blend_hr30",
        "delta_hr30",
        "all_tp_rate",
        "all_sl_rate",
        "all_timeout_rate",
        "all_conflict_rate",
    ]
    view = metrics[cols].copy()
    for col in view.columns:
        if col not in {"head", "window", "rows_labeled_exact"}:
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.6f}")
    return view.to_markdown(index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
    ap.add_argument("--default-config", default=str(DEFAULT_CONFIG_PATH))
    ap.add_argument("--ohlcv-root", default=str(DEFAULT_OHLCV_ROOT))
    ap.add_argument("--tp", type=float, default=0.03)
    ap.add_argument("--sl", type=float, default=0.02)
    ap.add_argument("--horizon-hours", type=float, default=5.0)
    args = ap.parse_args()

    report_dir = Path(args.report_dir)
    variants = _load_default_variants(Path(args.default_config))
    score_path = report_dir / "reliability_blend_component_scores.parquet"
    scores = pd.read_parquet(score_path)
    scores["timestamp"] = pd.to_datetime(scores["timestamp"], utc=True, errors="coerce")
    cfg = FixedBarrierConfig(tp=float(args.tp), sl=float(args.sl), horizon_hours=float(args.horizon_hours))

    needed_parts: list[pd.DataFrame] = []
    for head, group in scores.groupby("head", sort=True):
        variant = variants.get(str(head))
        col = f"blend_{variant}_score" if variant else ""
        if col not in group.columns:
            continue
        group = group[np.isfinite(group["anchor_score"].to_numpy(dtype="float64")) & np.isfinite(group[col].to_numpy(dtype="float64"))].copy()
        if group.empty:
            continue
        max_ts = group["timestamp"].max()
        needed_parts.append(group[group["timestamp"] >= max_ts - pd.Timedelta(days=28)])
    needed = pd.concat(needed_parts, ignore_index=True) if needed_parts else pd.DataFrame()

    if needed.empty:
        raise SystemExit("No rows available for fixed TP/SL audit")

    global_start = needed["timestamp"].min()
    global_end = needed["timestamp"].max() + pd.Timedelta(hours=float(args.horizon_hours) + 1)
    labeled_parts: list[pd.DataFrame] = []
    missing_symbol_rows: list[dict[str, Any]] = []

    grouped = needed.groupby(["head", "symbol"], sort=True)
    for (head, symbol), rows in grouped:
        symbol_start = rows["timestamp"].min()
        symbol_end = rows["timestamp"].max() + pd.Timedelta(hours=float(args.horizon_hours) + 1)
        ohlcv = _load_ohlcv_for_symbol(
            str(symbol),
            ohlcv_root=Path(args.ohlcv_root),
            start=min(global_start, symbol_start),
            end=max(global_end, symbol_end),
        )
        if ohlcv is None or ohlcv.empty:
            for row in rows.itertuples(index=False):
                missing_symbol_rows.append(
                    {
                        "head": row.head,
                        "row_id": int(row.row_id),
                        "timestamp": row.timestamp,
                        "symbol": row.symbol,
                        "fixed_outcome": np.nan,
                        "fixed_y_tp": np.nan,
                        "fixed_return": np.nan,
                        "fixed_conflict_same_bar": False,
                        "fixed_missing_reason": "symbol_ohlcv_missing",
                    }
                )
            continue
        labeled_parts.append(_label_symbol_rows(ohlcv, rows, side=_head_side(str(head)), cfg=cfg))

    outcome_rows = labeled_parts
    if missing_symbol_rows:
        outcome_rows.append(pd.DataFrame(missing_symbol_rows))
    outcomes = pd.concat(outcome_rows, ignore_index=True) if outcome_rows else pd.DataFrame()

    enriched = needed.merge(
        outcomes,
        on=["head", "row_id", "timestamp", "symbol"],
        how="left",
        validate="one_to_one",
    )
    metrics = _compute_window_metrics(enriched, variants, cfg)

    tag = f"fixed_tpsl_{int(round(cfg.tp * 100))}_{int(round(cfg.sl * 100))}_h{int(round(cfg.horizon_hours))}"
    rows_path = report_dir / f"reliability_blend_{tag}_row_outcomes.parquet"
    metrics_path = report_dir / f"reliability_blend_{tag}_last_1_2_4w_metrics.csv"
    report_path = report_dir / f"reliability_blend_{tag}_metrics.md"
    enriched.to_parquet(rows_path, index=False)
    metrics.to_csv(metrics_path, index=False)

    exact_count = int(np.isfinite(enriched["fixed_y_tp"].to_numpy(dtype="float64")).sum())
    lines = [
        f"# Reliability Blend Fixed TP/SL Metrics ({cfg.tp:.2%}/{cfg.sl:.2%}, H={cfg.horizon_hours:g}h)",
        "",
        "Metric type: OOF sampled blend rows with exact first-touch labels recomputed from Kraken futures hourly OHLCV.",
        "Same-bar TP/SL conflicts are treated conservatively as SL, matching the project labeler convention.",
        "",
        f"- rows evaluated: {len(enriched)}",
        f"- exact labeled rows: {exact_count}",
        f"- exact coverage: {exact_count / max(1, len(enriched)):.2%}",
        f"- source scores: `{score_path}`",
        f"- row outcomes: `{rows_path}`",
        f"- metrics csv: `{metrics_path}`",
        "",
        _format_metric_table(metrics),
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {metrics_path}")
    print(f"Wrote {rows_path}")
    print(f"Wrote {report_path}")
    print(_format_metric_table(metrics))


if __name__ == "__main__":
    main()
