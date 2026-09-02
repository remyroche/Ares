#!/usr/bin/env python3
"""Assess low-performance-period specialist meta models against baseline.

The companion training script writes per-head slice plans and specialist run ids.
This script waits for the specialist meta OOF files, aligns them to the source
meta OOF rows, and evaluates only the selected low-performance periods.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_summary(plan_dir: Path) -> pd.DataFrame:
    path = plan_dir / "low_period_specialist_summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing summary: {path}")
    return pd.read_csv(path)


def _meta_oof_path(data_root: Path, run_id: str, strategy_id: str) -> Path:
    return data_root / "artifacts" / run_id / "meta_oof" / f"meta_oof_{strategy_id}_tbm_clf.parquet"


def _coerce_oof(path: Path, score_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(str(path))
    cols = ["timestamp", "symbol", "y_bin", "oof_pred", "oof_p_move"]
    raw = pd.read_parquet(path)
    keep = [c for c in cols if c in raw.columns]
    df = raw[keep].copy()
    if "oof_pred" not in df.columns and "oof_p_move" in df.columns:
        df["oof_pred"] = df["oof_p_move"]
    missing = sorted({"timestamp", "symbol", "y_bin", "oof_pred"} - set(df.columns))
    if missing:
        raise RuntimeError(f"{path} missing required columns: {missing}")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df["symbol"] = df["symbol"].astype(str)
    df["y_bin"] = pd.to_numeric(df["y_bin"], errors="coerce")
    df[score_name] = pd.to_numeric(df["oof_pred"], errors="coerce")
    out = df[["timestamp", "symbol", "y_bin", score_name]].dropna().copy()
    out["y_bin"] = out["y_bin"].astype(np.float32)
    out[score_name] = out[score_name].astype(np.float32)
    return out


def _load_periods(plan_dir: Path, head: str) -> list[dict[str, str]]:
    path = plan_dir / f"{head}_slice_plan.json"
    payload = _read_json(path)
    view = payload.get("materialized_views", {}).get("train_meta")
    if not isinstance(view, dict):
        view = payload.get("materialized_views", {}).get("train_base", {})
    periods = view.get("allowed_periods", [])
    if not isinstance(periods, list):
        return []
    return [p for p in periods if isinstance(p, dict) and p.get("start_ts") and p.get("end_ts")]


def _period_mask(timestamps: pd.Series, periods: list[dict[str, str]]) -> np.ndarray:
    ts = pd.to_datetime(timestamps, utc=True, errors="coerce")
    out = np.zeros(len(ts), dtype=bool)
    for period in periods:
        start = pd.to_datetime(period.get("start_ts"), utc=True, errors="coerce")
        end = pd.to_datetime(period.get("end_ts"), utc=True, errors="coerce")
        if pd.isna(start) or pd.isna(end):
            continue
        out |= ((ts >= start) & (ts < end)).to_numpy(dtype=bool)
    return out


def _auc(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float64)
    score = np.asarray(score, dtype=np.float64)
    mask = np.isfinite(y) & np.isfinite(score)
    y = y[mask]
    score = score[mask]
    n_pos = int(np.sum(y > 0.5))
    n_neg = int(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=np.float64)
    ranks[order] = np.arange(1, len(score) + 1, dtype=np.float64)
    return float((np.sum(ranks[y > 0.5]) - n_pos * (n_pos + 1) / 2.0) / max(n_pos * n_neg, 1))


def _top_indices(score: np.ndarray, frac: float) -> np.ndarray:
    n = len(score)
    if n <= 0:
        return np.asarray([], dtype=np.int64)
    k = max(1, int(math.ceil(float(frac) * n)))
    order = np.argsort(np.asarray(score, dtype=np.float64), kind="mergesort")
    return order[-k:]


def _ndcg_at_frac(y: np.ndarray, score: np.ndarray, frac: float) -> float:
    n = len(y)
    if n <= 0:
        return float("nan")
    k = max(1, int(math.ceil(float(frac) * n)))
    order = np.argsort(np.asarray(score, dtype=np.float64), kind="mergesort")[::-1][:k]
    gains = np.asarray(y, dtype=np.float64)[order]
    discounts = 1.0 / np.log2(np.arange(2, k + 2, dtype=np.float64))
    dcg = float(np.sum(gains * discounts))
    ideal = np.sort(np.asarray(y, dtype=np.float64))[::-1][:k]
    idcg = float(np.sum(ideal * discounts))
    return float(dcg / idcg) if idcg > 0 else float("nan")


def _timestamp_metrics(df: pd.DataFrame, score_col: str, min_rows: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for ts, g in df.groupby("timestamp", sort=True):
        if len(g) < int(min_rows):
            continue
        y = g["y_bin"].to_numpy(dtype=np.float32)
        score = g[score_col].to_numpy(dtype=np.float32)
        rec: dict[str, Any] = {"timestamp": ts, "row_count": int(len(g))}
        for frac, name in [(0.10, "hr10"), (0.20, "hr20"), (0.30, "hr30")]:
            idx = _top_indices(score, frac)
            rec[name] = float(np.mean(y[idx])) if len(idx) else float("nan")
        rec["top_hr"] = (rec["hr10"] + 0.33 * rec["hr20"] + 0.25 * rec["hr30"]) / 1.58
        rec["ndcg30"] = _ndcg_at_frac(y, score, 0.30)
        rows.append(rec)
    return pd.DataFrame(rows)


def _swap_metrics(df: pd.DataFrame, min_rows: int) -> dict[str, float]:
    entrants: list[float] = []
    removed: list[float] = []
    net_correct = 0.0
    overlap = []
    for _, g in df.groupby("timestamp", sort=True):
        if len(g) < int(min_rows):
            continue
        y = g["y_bin"].to_numpy(dtype=np.float32)
        base = g["baseline_score"].to_numpy(dtype=np.float32)
        cand = g["candidate_score"].to_numpy(dtype=np.float32)
        base_idx = set(_top_indices(base, 0.30).tolist())
        cand_idx = set(_top_indices(cand, 0.30).tolist())
        ent = sorted(cand_idx - base_idx)
        rem = sorted(base_idx - cand_idx)
        if base_idx or cand_idx:
            overlap.append(len(base_idx & cand_idx) / max(len(base_idx | cand_idx), 1))
        if ent:
            entrants.extend(y[ent].astype(float).tolist())
        if rem:
            removed.extend(y[rem].astype(float).tolist())
        net_correct += float(np.sum(y[ent]) - np.sum(y[rem])) if ent or rem else 0.0
    return {
        "top30_jaccard": float(np.mean(overlap)) if overlap else float("nan"),
        "entrant_hr": float(np.mean(entrants)) if entrants else float("nan"),
        "removed_hr": float(np.mean(removed)) if removed else float("nan"),
        "net_correct_trades_gained": float(net_correct),
        "entrant_count": float(len(entrants)),
        "removed_count": float(len(removed)),
    }


def _summarize_pair(df: pd.DataFrame, *, min_rows: int) -> dict[str, Any]:
    base_ts = _timestamp_metrics(df, "baseline_score", min_rows)
    cand_ts = _timestamp_metrics(df, "candidate_score", min_rows)
    joined_ts = base_ts.merge(cand_ts, on=["timestamp", "row_count"], suffixes=("_baseline", "_candidate"))
    out: dict[str, Any] = {
        "aligned_rows": int(len(df)),
        "aligned_timestamps": int(df["timestamp"].nunique()),
        "evaluable_timestamps": int(len(joined_ts)),
        "baseline_auc": _auc(df["y_bin"].to_numpy(), df["baseline_score"].to_numpy()),
        "candidate_auc": _auc(df["y_bin"].to_numpy(), df["candidate_score"].to_numpy()),
    }
    out["delta_auc"] = out["candidate_auc"] - out["baseline_auc"] if np.isfinite(out["candidate_auc"]) and np.isfinite(out["baseline_auc"]) else float("nan")
    for metric in ["hr10", "hr20", "hr30", "top_hr", "ndcg30"]:
        b = f"{metric}_baseline"
        c = f"{metric}_candidate"
        if b in joined_ts.columns and c in joined_ts.columns:
            out[f"baseline_{metric}_timestamp_mean"] = float(joined_ts[b].mean())
            out[f"candidate_{metric}_timestamp_mean"] = float(joined_ts[c].mean())
            out[f"delta_{metric}_timestamp_mean"] = float((joined_ts[c] - joined_ts[b]).mean())
            out[f"delta_{metric}_q25_timestamp"] = float((joined_ts[c] - joined_ts[b]).quantile(0.25))
    out.update(_swap_metrics(df, min_rows))
    return out


def assess(plan_dir: Path, data_root: Path, source_run_id: str, min_rows_per_timestamp: int) -> tuple[pd.DataFrame, str]:
    summary = _load_summary(plan_dir)
    rows: list[dict[str, Any]] = []
    for rec in summary.to_dict("records"):
        head = str(rec.get("head", ""))
        strategy_id = str(rec.get("strategy_id", ""))
        run_id = str(rec.get("run_id", ""))
        if not head or not strategy_id or not run_id:
            continue
        base_path = _meta_oof_path(data_root, source_run_id, strategy_id)
        cand_path = _meta_oof_path(data_root, run_id, strategy_id)
        row: dict[str, Any] = {
            "head": head,
            "strategy_id": strategy_id,
            "run_id": run_id,
            "baseline_meta_oof_path": str(base_path),
            "candidate_meta_oof_path": str(cand_path),
        }
        try:
            periods = _load_periods(plan_dir, head)
            baseline = _coerce_oof(base_path, "baseline_score")
            candidate = _coerce_oof(cand_path, "candidate_score")
            baseline = baseline.loc[_period_mask(baseline["timestamp"], periods)].copy()
            key = ["timestamp", "symbol"]
            joined = baseline.merge(
                candidate[key + ["candidate_score"]],
                on=key,
                how="inner",
                validate="one_to_one",
            )
            row.update(_summarize_pair(joined, min_rows=min_rows_per_timestamp))
            row["status"] = "ok"
        except Exception as exc:  # noqa: BLE001
            row["status"] = "missing_or_failed"
            row["error"] = str(exc)
        rows.append(row)
    df = pd.DataFrame(rows)
    csv_path = plan_dir / "low_period_specialist_assessment.csv"
    md_path = plan_dir / "low_period_specialist_assessment.md"
    df.to_csv(csv_path, index=False)
    display_cols = [
        "head",
        "status",
        "aligned_rows",
        "evaluable_timestamps",
        "baseline_auc",
        "candidate_auc",
        "delta_auc",
        "baseline_hr30_timestamp_mean",
        "candidate_hr30_timestamp_mean",
        "delta_hr30_timestamp_mean",
        "baseline_top_hr_timestamp_mean",
        "candidate_top_hr_timestamp_mean",
        "delta_top_hr_timestamp_mean",
        "baseline_ndcg30_timestamp_mean",
        "candidate_ndcg30_timestamp_mean",
        "delta_ndcg30_timestamp_mean",
        "entrant_hr",
        "removed_hr",
        "net_correct_trades_gained",
    ]
    existing = [c for c in display_cols if c in df.columns]
    lines = [
        "# Low-Performance Specialist Assessment",
        "",
        f"Plan dir: `{plan_dir}`",
        f"Source run: `{source_run_id}`",
        "",
    ]
    if existing:
        lines.append(df[existing].to_markdown(index=False))
    else:
        lines.append("No assessment rows.")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return df, str(md_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-dir", default="data_perp/reports/low_performance_period_specialist_latest")
    parser.add_argument("--data-root", default="data_perp")
    parser.add_argument("--source-run-id", default="20260617_090000_no_mkt4_labelhpo_final_fit")
    parser.add_argument("--min-rows-per-timestamp", type=int, default=4)
    args = parser.parse_args()
    df, md_path = assess(
        Path(args.plan_dir),
        Path(args.data_root),
        str(args.source_run_id),
        int(args.min_rows_per_timestamp),
    )
    print(f"Wrote {md_path}")
    print(df[["head", "status"]].to_string(index=False) if not df.empty else "No rows")


if __name__ == "__main__":
    main()
