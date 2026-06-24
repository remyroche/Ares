"""One-shot fresh-OOS evaluator for frozen J4/J5 contextual meta decisions.

This evaluator is designed to be run after the readiness gate has passed.  It
does not select features, thresholds, heads, or hyperparameters.  It applies the
frozen top-fraction rule to scored fresh rows and reports directional metrics.
"""

from __future__ import annotations

import argparse
import json
import sys
from math import ceil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.check_j4_j5_contextual_meta_fresh_oos_readiness import DEFAULT_LABEL_DIR, HEAD_LABEL_PREFIX


DEFAULT_FREEZE_MANIFEST = Path(
    "data_perp/reports/j4_j5_contextual_meta_all_head_freeze_20260623/"
    "j4_j5_contextual_meta_all_head_freeze_manifest.csv"
)
DEFAULT_READINESS_AUDIT = Path(
    "data_perp/reports/j4_j5_contextual_meta_fresh_oos_readiness_20260623/"
    "j4_j5_contextual_meta_fresh_oos_readiness_audit.json"
)
DEFAULT_OUTPUT_DIR = Path("data_perp/reports/j4_j5_contextual_meta_fresh_oos_eval_20260623")

TIME_COLUMNS = ("timestamp", "__ts__", "ts", "entry_time", "bar_time")
SYMBOL_COLUMNS = ("symbol", "__symbol__", "asset", "ticker")
LABEL_COLUMNS = ("y_bin", "__y_bin__", "label", "target")
RETURN_COLUMNS = ("return", "ret", "net_return", "y_ret", "__y_ret__")
BASELINE_SCORE_COLUMNS = ("baseline_score", "baseline_pred", "baseline_probability", "score_baseline", "base_score")
CANDIDATE_SCORE_COLUMNS = ("candidate_score", "candidate_pred", "contextual_score", "frozen_score", "score", "pred")


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        val = float(obj)
        return None if not np.isfinite(val) else val
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def _first_present(columns: list[str] | pd.Index, candidates: tuple[str, ...]) -> str | None:
    present = set(str(c) for c in columns)
    for col in candidates:
        if col in present:
            return col
    return None


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".csv", ".gz"} or path.name.endswith(".csv.gz"):
        return pd.read_csv(path)
    raise ValueError(f"Unsupported score file type: {path}")


def _discover_score_files(score_dirs: list[Path], score_files: list[Path], head: str) -> list[Path]:
    out = [p for p in score_files if p.exists()]
    for score_dir in score_dirs:
        if not score_dir.exists():
            continue
        out.extend(sorted(score_dir.glob(f"*{head}*.parquet")))
        out.extend(sorted(score_dir.glob(f"*{head}*.csv")))
        out.extend(sorted(score_dir.glob(f"*{head}*.csv.gz")))
    return sorted(set(out))


def _head_label_file(label_dir: Path, head: str) -> Path | None:
    prefix = HEAD_LABEL_PREFIX.get(head, f"train_{head}_")
    matches = sorted(label_dir.glob(f"{prefix}*.parquet")) if label_dir.exists() else []
    return matches[0] if matches else None


def _normalise_scores(df: pd.DataFrame, *, head: str, source_path: Path) -> pd.DataFrame:
    if "head" in df.columns:
        df = df.loc[df["head"].astype(str).eq(str(head))].copy()
    time_col = _first_present(df.columns, TIME_COLUMNS)
    baseline_col = _first_present(df.columns, BASELINE_SCORE_COLUMNS)
    candidate_col = _first_present(df.columns, CANDIDATE_SCORE_COLUMNS)
    label_col = _first_present(df.columns, LABEL_COLUMNS)
    symbol_col = _first_present(df.columns, SYMBOL_COLUMNS)
    ret_col = _first_present(df.columns, RETURN_COLUMNS)
    missing = [
        name
        for name, col in {
            "timestamp": time_col,
            "baseline_score": baseline_col,
            "candidate_score": candidate_col,
        }.items()
        if col is None
    ]
    if missing:
        raise ValueError(f"{source_path} missing required columns: {missing}")
    out = pd.DataFrame(
        {
            "head": head,
            "timestamp": pd.to_datetime(df[time_col], utc=True, errors="coerce"),
            "baseline_score": pd.to_numeric(df[baseline_col], errors="coerce"),
            "candidate_score": pd.to_numeric(df[candidate_col], errors="coerce"),
            "score_source_path": str(source_path),
        }
    )
    if label_col is not None:
        out["y_bin"] = pd.to_numeric(df[label_col], errors="coerce")
    if symbol_col is not None:
        out["symbol"] = df[symbol_col].astype(str).to_numpy()
    if ret_col is not None:
        out["return"] = pd.to_numeric(df[ret_col], errors="coerce")
    return out


def _merge_labels_if_needed(scores: pd.DataFrame, label_dir: Path, head: str) -> pd.DataFrame:
    if "y_bin" in scores.columns and scores["y_bin"].notna().any():
        return scores
    if "symbol" not in scores.columns:
        raise ValueError(f"{head} score rows need y_bin or symbol column for label merge")
    label_path = _head_label_file(label_dir, head)
    if label_path is None:
        raise ValueError(f"No label file found for {head} in {label_dir}")
    labels = pd.read_parquet(label_path)
    time_col = _first_present(labels.columns, TIME_COLUMNS)
    symbol_col = _first_present(labels.columns, SYMBOL_COLUMNS)
    label_col = _first_present(labels.columns, LABEL_COLUMNS)
    ret_col = _first_present(labels.columns, RETURN_COLUMNS)
    if time_col is None or symbol_col is None or label_col is None:
        raise ValueError(f"{label_path} cannot provide timestamp/symbol/y_bin labels")
    keep = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(labels[time_col], utc=True, errors="coerce"),
            "symbol": labels[symbol_col].astype(str),
            "y_bin": pd.to_numeric(labels[label_col], errors="coerce"),
        }
    )
    if ret_col is not None and "return" not in scores.columns:
        keep["return"] = pd.to_numeric(labels[ret_col], errors="coerce")
    merged = scores.merge(keep.drop_duplicates(["timestamp", "symbol"]), on=["timestamp", "symbol"], how="left")
    return merged


def _dcg(labels: np.ndarray) -> float:
    if labels.size == 0:
        return np.nan
    discounts = 1.0 / np.log2(np.arange(2, labels.size + 2, dtype=np.float64))
    return float(np.sum(labels.astype(np.float64) * discounts))


def _ndcg_at_k(selected_labels: np.ndarray, all_labels: np.ndarray) -> float:
    if selected_labels.size == 0:
        return np.nan
    ideal = np.sort(all_labels.astype(np.float64))[::-1][: selected_labels.size]
    denom = _dcg(ideal)
    return np.nan if denom <= 0 else float(_dcg(selected_labels) / denom)


def _top_indices(values: np.ndarray, k: int) -> np.ndarray:
    order = np.argsort(-values, kind="mergesort")
    return order[:k]


def _top_k(n: int, fraction: float) -> int:
    return max(1, int(ceil(round(float(fraction) * int(n), 12))))


def timestamp_metrics(
    panel: pd.DataFrame,
    *,
    rank_threshold: float,
    min_timestamp_rows: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    top_fraction = 1.0 - float(rank_threshold)
    for ts, group in panel.sort_values("timestamp").groupby("timestamp", sort=True):
        g = group.reset_index(drop=True)
        n = int(len(g))
        if n < int(min_timestamp_rows):
            continue
        k = _top_k(n, top_fraction)
        y = pd.to_numeric(g["y_bin"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        cand = pd.to_numeric(g["candidate_score"], errors="coerce").to_numpy(dtype=np.float64)
        base = pd.to_numeric(g["baseline_score"], errors="coerce").to_numpy(dtype=np.float64)
        valid = np.isfinite(cand) & np.isfinite(base) & np.isfinite(y)
        if int(valid.sum()) < int(min_timestamp_rows):
            continue
        valid_idx = np.flatnonzero(valid)
        yv = y[valid]
        cand_v = cand[valid]
        base_v = base[valid]
        k = max(1, min(k, int(valid.sum())))
        cand_top = _top_indices(cand_v, k)
        base_top = _top_indices(base_v, k)
        cand_set = set(valid_idx[cand_top].tolist())
        base_set = set(valid_idx[base_top].tolist())
        entrants = sorted(cand_set - base_set)
        removed = sorted(base_set - cand_set)
        overlap = len(cand_set & base_set)
        union = len(cand_set | base_set)
        cand_labels = yv[cand_top]
        base_labels = yv[base_top]
        row = {
            "timestamp": pd.Timestamp(ts).isoformat(),
            "eligible_rows": int(valid.sum()),
            "selected_count_top30": int(k),
            "hr_top30": float(np.mean(cand_labels)) if cand_labels.size else np.nan,
            "baseline_hr_top30": float(np.mean(base_labels)) if base_labels.size else np.nan,
            "delta_hr_top30": float(np.mean(cand_labels) - np.mean(base_labels)) if cand_labels.size else np.nan,
            "ndcg_top30": _ndcg_at_k(cand_labels, yv),
            "baseline_ndcg_top30": _ndcg_at_k(base_labels, yv),
            "top30_jaccard": float(overlap / union) if union else np.nan,
            "top30_entrant_count": int(len(entrants)),
            "top30_removed_count": int(len(removed)),
            "top30_entrant_hit_rate": float(np.mean(y[entrants])) if entrants else np.nan,
            "top30_removed_hit_rate": float(np.mean(y[removed])) if removed else np.nan,
            "net_correct_trades_gained": float(np.sum(y[entrants]) - np.sum(y[removed])),
        }
        row["delta_ndcg_top30"] = (
            row["ndcg_top30"] - row["baseline_ndcg_top30"]
            if np.isfinite(row["ndcg_top30"]) and np.isfinite(row["baseline_ndcg_top30"])
            else np.nan
        )
        for frac in (0.10, 0.20):
            kk = _top_k(int(valid.sum()), frac)
            ctop = _top_indices(cand_v, kk)
            btop = _top_indices(base_v, kk)
            row[f"hr_top{int(frac * 100)}"] = float(np.mean(yv[ctop]))
            row[f"baseline_hr_top{int(frac * 100)}"] = float(np.mean(yv[btop]))
            row[f"delta_hr_top{int(frac * 100)}"] = row[f"hr_top{int(frac * 100)}"] - row[f"baseline_hr_top{int(frac * 100)}"]
        if "return" in g.columns:
            ret = pd.to_numeric(g["return"], errors="coerce").to_numpy(dtype=np.float64)
            row["top30_mean_return"] = float(np.nanmean(ret[valid_idx[cand_top]]))
            row["baseline_top30_mean_return"] = float(np.nanmean(ret[valid_idx[base_top]]))
            row["delta_top30_mean_return"] = row["top30_mean_return"] - row["baseline_top30_mean_return"]
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_metrics(ts_metrics: pd.DataFrame, head: str) -> dict[str, Any]:
    if ts_metrics.empty:
        return {"head": head, "timestamp_count": 0}
    out = {
        "head": head,
        "timestamp_count": int(len(ts_metrics)),
        "eligible_rows": int(ts_metrics["eligible_rows"].sum()),
    }
    for col in [
        "hr_top10",
        "baseline_hr_top10",
        "delta_hr_top10",
        "hr_top20",
        "baseline_hr_top20",
        "delta_hr_top20",
        "hr_top30",
        "baseline_hr_top30",
        "delta_hr_top30",
        "ndcg_top30",
        "baseline_ndcg_top30",
        "delta_ndcg_top30",
        "net_correct_trades_gained",
        "delta_top30_mean_return",
    ]:
        if col in ts_metrics.columns:
            out[f"timestamp_weighted_{col}"] = float(pd.to_numeric(ts_metrics[col], errors="coerce").mean())
    if "net_correct_trades_gained" in ts_metrics.columns:
        out["total_net_correct_trades_gained"] = float(pd.to_numeric(ts_metrics["net_correct_trades_gained"], errors="coerce").sum())
    return out


def _load_readiness(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"status": "missing", "items": []}
    return json.loads(path.read_text())


def evaluate(
    *,
    freeze_manifest_path: Path,
    readiness_audit_path: Path,
    score_dirs: list[Path],
    score_files: list[Path],
    label_dir: Path,
    min_later_hours: float,
    min_timestamp_rows: int,
    require_ready: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    freeze = pd.read_csv(freeze_manifest_path)
    readiness = _load_readiness(readiness_audit_path)
    if require_ready and readiness.get("status") != "ready":
        audit = {
            "status": "not_ready",
            "items": [
                {
                    "requirement": "fresh_oos_readiness_gate",
                    "status": "not_ready",
                    "metrics": {"readiness_status": readiness.get("status", "missing")},
                }
            ],
        }
        return pd.DataFrame(), pd.DataFrame(), audit

    timestamp_rows: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    audit_items: list[dict[str, Any]] = []
    for _, frozen in freeze.iterrows():
        head = str(frozen["head"])
        boundary = pd.Timestamp(str(frozen["effective_fresh_oos_after"]))
        if boundary.tzinfo is None:
            boundary = boundary.tz_localize("UTC")
        guarded_start = boundary + pd.Timedelta(hours=float(min_later_hours))
        files = _discover_score_files(score_dirs, score_files, head)
        panels: list[pd.DataFrame] = []
        for path in files:
            panel = _normalise_scores(_read_table(path), head=head, source_path=path)
            panel = _merge_labels_if_needed(panel, label_dir, head)
            panel = panel.loc[panel["timestamp"].gt(guarded_start)].copy()
            panels.append(panel)
        if not panels:
            audit_items.append(
                {
                    "requirement": f"{head}_score_rows_after_guard",
                    "status": "failed",
                    "metrics": {"score_files": [str(x) for x in files], "guarded_start": guarded_start.isoformat()},
                }
            )
            continue
        head_panel = pd.concat(panels, ignore_index=True)
        head_panel = head_panel.dropna(subset=["timestamp", "baseline_score", "candidate_score", "y_bin"])
        ts = timestamp_metrics(
            head_panel,
            rank_threshold=float(frozen.get("rank_threshold", 0.70)),
            min_timestamp_rows=int(min_timestamp_rows),
        )
        if not ts.empty:
            ts.insert(0, "head", head)
            timestamp_rows.append(ts)
            summary_rows.append(
                {
                    **aggregate_metrics(ts, head),
                    "effective_fresh_oos_after": boundary.isoformat(),
                    "guarded_fresh_oos_start": guarded_start.isoformat(),
                    "selected_contextual_feature_arm": frozen.get("selected_contextual_feature_arm", ""),
                    "selected_capacity_config": frozen.get("selected_capacity_config", ""),
                    "selected_distillation_variant": frozen.get("selected_distillation_variant", ""),
                    "score_files": ";".join(str(x) for x in files),
                }
            )
        audit_items.append(
            {
                "requirement": f"{head}_fixed_fresh_oos_metrics",
                "status": "passed" if not ts.empty else "failed",
                "metrics": {
                    "score_files": [str(x) for x in files],
                    "rows_after_guard": int(len(head_panel)),
                    "timestamp_metric_rows": int(len(ts)),
                    "guarded_start": guarded_start.isoformat(),
                },
            }
        )
    timestamp_df = pd.concat(timestamp_rows, ignore_index=True) if timestamp_rows else pd.DataFrame()
    summary_df = pd.DataFrame(summary_rows)
    audit = {
        "status": "passed" if audit_items and all(x["status"] == "passed" for x in audit_items) else "failed",
        "items": audit_items,
    }
    return summary_df, timestamp_df, audit


def _write_report(out_dir: Path, summary: pd.DataFrame, audit: dict[str, Any]) -> None:
    lines = [
        "# J4/J5 Fresh OOS Evaluation",
        "",
        "This report evaluates only fixed frozen decisions. It is not a selection/HPO artifact.",
        "",
        "## Audit",
        "",
        pd.DataFrame(audit.get("items", [])).to_markdown(index=False),
        "",
    ]
    if not summary.empty:
        cols = [
            "head",
            "timestamp_count",
            "timestamp_weighted_delta_hr_top30",
            "timestamp_weighted_delta_ndcg_top30",
            "timestamp_weighted_delta_hr_top10",
            "timestamp_weighted_delta_hr_top20",
            "total_net_correct_trades_gained",
            "guarded_fresh_oos_start",
        ]
        lines.extend(["## Summary", "", summary[[c for c in cols if c in summary.columns]].to_markdown(index=False, floatfmt=".6f"), ""])
    (out_dir / "j4_j5_contextual_meta_fresh_oos_eval_report.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-manifest", type=Path, default=DEFAULT_FREEZE_MANIFEST)
    parser.add_argument("--readiness-audit", type=Path, default=DEFAULT_READINESS_AUDIT)
    parser.add_argument("--score-dir", action="append", type=Path, default=None)
    parser.add_argument("--score-file", action="append", type=Path, default=None)
    parser.add_argument("--label-dir", type=Path, default=DEFAULT_LABEL_DIR)
    parser.add_argument("--min-later-hours", type=float, default=24.0)
    parser.add_argument("--min-timestamp-rows", type=int, default=3)
    parser.add_argument("--ignore-readiness", action="store_true")
    parser.add_argument("--fail-if-not-ready", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary, timestamps, audit = evaluate(
        freeze_manifest_path=args.freeze_manifest,
        readiness_audit_path=args.readiness_audit,
        score_dirs=list(args.score_dir or []),
        score_files=list(args.score_file or []),
        label_dir=args.label_dir,
        min_later_hours=float(args.min_later_hours),
        min_timestamp_rows=int(args.min_timestamp_rows),
        require_ready=not bool(args.ignore_readiness),
    )
    summary.to_csv(args.output_dir / "j4_j5_contextual_meta_fresh_oos_eval_summary.csv", index=False)
    timestamps.to_csv(args.output_dir / "j4_j5_contextual_meta_fresh_oos_eval_timestamp_metrics.csv", index=False)
    (args.output_dir / "j4_j5_contextual_meta_fresh_oos_eval_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True, default=_json_default)
    )
    (args.output_dir / "j4_j5_contextual_meta_fresh_oos_eval_inputs.json").write_text(
        json.dumps(
            {
                "freeze_manifest": str(args.freeze_manifest),
                "readiness_audit": str(args.readiness_audit),
                "score_dirs": [str(x) for x in list(args.score_dir or [])],
                "score_files": [str(x) for x in list(args.score_file or [])],
                "label_dir": str(args.label_dir),
                "min_later_hours": float(args.min_later_hours),
                "min_timestamp_rows": int(args.min_timestamp_rows),
                "ignore_readiness": bool(args.ignore_readiness),
            },
            indent=2,
            sort_keys=True,
        )
    )
    _write_report(args.output_dir, summary, audit)
    print(f"[j4_j5_fresh_oos_eval] status={audit['status']} wrote {args.output_dir}", flush=True)
    if args.fail_if_not_ready and audit["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
