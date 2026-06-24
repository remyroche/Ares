"""Prospective baseline-versus-contextual score collector for frozen J4/J5 heads.

This script is intentionally label-free.  It collects already-recorded live or
shadow prediction-ledger rows for a requested signal-bar window, optionally
merges frozen contextual candidate scores, and writes a coverage audit.  It is a
scoring/readiness artifact, not an OOS performance evaluation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_FREEZE_MANIFEST = Path(
    "data_perp/reports/j4_j5_contextual_meta_all_head_freeze_20260623/"
    "j4_j5_contextual_meta_all_head_freeze_manifest.csv"
)
DEFAULT_LEDGER_ROOT = Path("data_perp/exchanges/krakenfutures/live_state")
DEFAULT_OUTPUT_DIR = Path("data_perp/reports/j4_j5_contextual_meta_prospective_dual_scoring_20260616_20260622")

HEAD_PATTERNS: dict[str, tuple[str, ...]] = {
    "long_bars": ("long_bars_", "bars_in_high_vol_state_log_norm"),
    "long_dist": ("long_dist_", "dist_ema20_atr"),
    "short_asset": ("short_asset_", "asset_minus_mkt_oi_1d_peer_resid"),
    "short_boll": ("short_boll", "short_bollinger", "bollinger_band_width"),
}
TIME_COLUMNS = ("timestamp", "__ts__", "ts", "signal_bar_ts", "decision_ts", "entry_time", "bar_time")
SYMBOL_COLUMNS = ("symbol", "__symbol__", "asset", "ticker")
CANDIDATE_SCORE_COLUMNS = ("candidate_score", "contextual_score", "frozen_score", "score", "pred")
BASELINE_SCORE_COLUMNS = ("calibrated_score", "meta_pred", "raw_prediction_score", "base_pred")


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
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    return str(obj)


def _as_utc(value: str, *, is_end: bool = False) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    if is_end and len(value) == 10:
        ts = ts + pd.Timedelta(days=1)
    return ts


def _first_present(columns: list[str] | pd.Index, candidates: tuple[str, ...]) -> str | None:
    present = {str(c) for c in columns}
    for col in candidates:
        if col in present:
            return col
    return None


def _infer_head(strategy_id: Any) -> str | None:
    text = str(strategy_id).lower()
    for head, patterns in HEAD_PATTERNS.items():
        if any(pattern.lower() in text for pattern in patterns):
            return head
    return None


def _finite_score_and_source(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    score = np.full(len(df), np.nan, dtype=np.float32)
    source = np.full(len(df), "", dtype=object)
    for col in BASELINE_SCORE_COLUMNS:
        if col not in df.columns:
            continue
        values = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float32, copy=False)
        take = ~np.isfinite(score) & np.isfinite(values)
        score[take] = values[take]
        source[take] = col
    return score, source


def _read_ledger(path: Path, *, start: pd.Timestamp, end_exclusive: pd.Timestamp) -> pd.DataFrame:
    try:
        df = pd.read_parquet(path)
    except Exception as exc:
        return pd.DataFrame(
            {
                "ledger_path": [str(path)],
                "read_error": [f"{type(exc).__name__}: {exc}"],
            }
        )
    required = {"strategy_id", "symbol"}
    if not required.issubset(df.columns):
        return pd.DataFrame()
    signal_col = "signal_bar_ts" if "signal_bar_ts" in df.columns else _first_present(df.columns, TIME_COLUMNS)
    if signal_col is None:
        return pd.DataFrame()
    decision_col = "decision_ts" if "decision_ts" in df.columns else signal_col
    signal_ts = pd.to_datetime(df[signal_col], utc=True, errors="coerce")
    decision_ts = pd.to_datetime(df[decision_col], utc=True, errors="coerce")
    mask = signal_ts.ge(start) & signal_ts.lt(end_exclusive)
    if not bool(mask.any()):
        return pd.DataFrame()
    local = df.loc[mask].copy()
    local_signal = signal_ts.loc[mask]
    local_decision = decision_ts.loc[mask]
    baseline_score, baseline_source = _finite_score_and_source(local)
    out = pd.DataFrame(
        {
            "head": local["strategy_id"].map(_infer_head),
            "timestamp": local_signal.to_numpy(),
            "decision_ts": local_decision.to_numpy(),
            "symbol": local["symbol"].astype(str).to_numpy(),
            "strategy_id": local["strategy_id"].astype(str).to_numpy(),
            "ledger_run_id": path.parent.name if path.parent.name != "live_state" else "root_live_state",
            "ledger_path": str(path),
            "baseline_score": baseline_score,
            "baseline_score_source": baseline_source,
        }
    )
    for col in ("base_pred", "meta_pred", "calibrated_score", "raw_prediction_score", "policy_rank_pct", "auction_rank_pct"):
        if col in local.columns:
            out[col] = pd.to_numeric(local[col], errors="coerce").to_numpy(dtype=np.float32, copy=False)
    return out[out["head"].notna()].reset_index(drop=True)


def _discover_ledgers(ledger_root: Path) -> list[Path]:
    paths = []
    direct = ledger_root / "prediction_ledger.parquet"
    if direct.exists():
        paths.append(direct)
    scoped = ledger_root / "prediction_ledgers"
    if scoped.exists():
        paths.extend(sorted(scoped.glob("*/prediction_ledger.parquet")))
    return sorted(set(paths))


def _collect_baseline_scores(
    ledger_root: Path,
    *,
    start: pd.Timestamp,
    end_exclusive: pd.Timestamp,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    frames: list[pd.DataFrame] = []
    ledger_audit: list[dict[str, Any]] = []
    for path in _discover_ledgers(ledger_root):
        frame = _read_ledger(path, start=start, end_exclusive=end_exclusive)
        if not frame.empty and "read_error" in frame.columns:
            ledger_audit.append({"path": str(path), "status": "read_error", "error": str(frame["read_error"].iloc[0])})
            continue
        if frame.empty:
            ledger_audit.append({"path": str(path), "status": "no_rows_in_window", "rows": 0})
            continue
        frames.append(frame)
        ledger_audit.append(
            {
                "path": str(path),
                "status": "loaded",
                "rows_in_window": int(len(frame)),
                "heads": sorted(frame["head"].dropna().astype(str).unique().tolist()),
                "min_signal_ts": pd.to_datetime(frame["timestamp"], utc=True).min().isoformat(),
                "max_signal_ts": pd.to_datetime(frame["timestamp"], utc=True).max().isoformat(),
            }
        )
    if not frames:
        return pd.DataFrame(), ledger_audit
    out = pd.concat(frames, ignore_index=True)
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out["decision_ts"] = pd.to_datetime(out["decision_ts"], utc=True, errors="coerce")
    out = out[np.isfinite(pd.to_numeric(out["baseline_score"], errors="coerce"))].copy()
    out = out.sort_values(["head", "timestamp", "symbol", "strategy_id", "decision_ts"])
    dedupe_keys = ["head", "timestamp", "symbol", "strategy_id"]
    out = out.drop_duplicates(dedupe_keys, keep="last").reset_index(drop=True)
    return out, ledger_audit


def _normalise_candidate_scores(path: Path, head: str) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        df = pd.read_parquet(path)
    elif suffix in {".csv", ".gz"} or path.name.endswith(".csv.gz"):
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported candidate score file type: {path}")
    time_col = _first_present(df.columns, TIME_COLUMNS)
    symbol_col = _first_present(df.columns, SYMBOL_COLUMNS)
    score_col = _first_present(df.columns, CANDIDATE_SCORE_COLUMNS)
    missing = [
        name
        for name, col in {
            "timestamp": time_col,
            "symbol": symbol_col,
            "candidate_score": score_col,
        }.items()
        if col is None
    ]
    if missing:
        raise ValueError(f"{path} missing columns {missing}")
    out = pd.DataFrame(
        {
            "head": head,
            "timestamp": pd.to_datetime(df[time_col], utc=True, errors="coerce"),
            "symbol": df[symbol_col].astype(str).to_numpy(),
            "candidate_score": pd.to_numeric(df[score_col], errors="coerce").to_numpy(dtype=np.float32, copy=False),
            "candidate_score_path": str(path),
        }
    )
    out = out[np.isfinite(out["candidate_score"].to_numpy(dtype=np.float32, copy=False))].copy()
    return out.drop_duplicates(["head", "timestamp", "symbol"], keep="last").reset_index(drop=True)


def _discover_candidate_files(score_dirs: list[Path], score_files: list[Path], head: str) -> list[Path]:
    paths = [p for p in score_files if p.exists() and head in p.name]
    for score_dir in score_dirs:
        if not score_dir.exists():
            continue
        paths.extend(sorted(score_dir.glob(f"*{head}*.parquet")))
        paths.extend(sorted(score_dir.glob(f"*{head}*.csv")))
        paths.extend(sorted(score_dir.glob(f"*{head}*.csv.gz")))
    return sorted(set(paths))


def _load_candidate_scores(
    heads: list[str],
    score_dirs: list[Path],
    score_files: list[Path],
    *,
    start: pd.Timestamp,
    end_exclusive: pd.Timestamp,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    frames: list[pd.DataFrame] = []
    audit: list[dict[str, Any]] = []
    for head in heads:
        paths = _discover_candidate_files(score_dirs, score_files, head)
        if not paths:
            audit.append({"head": head, "status": "missing_candidate_score_file", "paths": []})
            continue
        for path in paths:
            try:
                frame = _normalise_candidate_scores(path, head)
            except Exception as exc:
                audit.append(
                    {
                        "head": head,
                        "path": str(path),
                        "status": "read_error",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue
            frame = frame[frame["timestamp"].ge(start) & frame["timestamp"].lt(end_exclusive)].copy()
            frames.append(frame)
            audit.append({"head": head, "path": str(path), "status": "loaded", "rows_in_window": int(len(frame))})
    if not frames:
        return pd.DataFrame(columns=["head", "timestamp", "symbol", "candidate_score", "candidate_score_path"]), audit
    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["head", "timestamp", "symbol", "candidate_score_path"])
    out = out.drop_duplicates(["head", "timestamp", "symbol"], keep="last").reset_index(drop=True)
    return out, audit


def _summary_by_head(dual: pd.DataFrame, heads: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for head in heads:
        group = dual.loc[dual["head"].astype(str) == head].copy() if not dual.empty else pd.DataFrame()
        complete = group[np.isfinite(pd.to_numeric(group.get("candidate_score", np.nan), errors="coerce"))].copy()
        rows.append(
            {
                "head": head,
                "baseline_rows": int(len(group)),
                "baseline_timestamps": int(group["timestamp"].nunique()) if not group.empty else 0,
                "baseline_min_ts": "" if group.empty else pd.to_datetime(group["timestamp"], utc=True).min().isoformat(),
                "baseline_max_ts": "" if group.empty else pd.to_datetime(group["timestamp"], utc=True).max().isoformat(),
                "candidate_rows_matched": int(len(complete)),
                "candidate_timestamps_matched": int(complete["timestamp"].nunique()) if not complete.empty else 0,
                "dual_complete": bool(len(group) > 0 and len(complete) == len(group)),
                "dual_coverage": float(len(complete) / len(group)) if len(group) else 0.0,
            }
        )
    return pd.DataFrame(rows)


def _daily_summary(
    dual: pd.DataFrame,
    heads: list[str],
    *,
    start: pd.Timestamp,
    end_exclusive: pd.Timestamp,
) -> pd.DataFrame:
    days = pd.date_range(start.normalize(), end_exclusive.normalize() - pd.Timedelta(days=1), freq="D", tz="UTC")
    rows: list[dict[str, Any]] = []
    if dual.empty:
        grouped: dict[tuple[str, pd.Timestamp], pd.DataFrame] = {}
    else:
        local = dual.copy()
        local["date"] = pd.to_datetime(local["timestamp"], utc=True, errors="coerce").dt.normalize()
        grouped = {key: group for key, group in local.groupby(["head", "date"], sort=True)}
    for head in heads:
        for day in days:
            group = grouped.get((head, day), pd.DataFrame())
            candidate = (
                group[np.isfinite(pd.to_numeric(group.get("candidate_score", np.nan), errors="coerce"))]
                if not group.empty
                else pd.DataFrame()
            )
            rows.append(
                {
                    "head": head,
                    "date": day.date().isoformat(),
                    "baseline_rows": int(len(group)),
                    "baseline_timestamps": int(group["timestamp"].nunique()) if not group.empty else 0,
                    "candidate_rows_matched": int(len(candidate)),
                    "dual_coverage": float(len(candidate) / len(group)) if len(group) else 0.0,
                }
            )
    return pd.DataFrame(rows)


def _write_report(out_dir: Path, summary: pd.DataFrame, daily: pd.DataFrame, audit: dict[str, Any]) -> None:
    lines = [
        "# J4/J5 Prospective Dual Scoring",
        "",
        "This artifact is label-free prospective scoring/readiness output. It is not an OOS performance evaluation.",
        "",
        f"- Window start: `{audit['window_start']}`",
        f"- Window end exclusive: `{audit['window_end_exclusive']}`",
        f"- Status: `{audit['status']}`",
        "",
        "## Coverage",
        "",
        summary.to_markdown(index=False),
        "",
        "## Daily Coverage",
        "",
        daily.to_markdown(index=False),
        "",
        "## Notes",
        "",
    ]
    for note in audit.get("notes", []):
        lines.append(f"- {note}")
    (out_dir / "prospective_dual_scoring_report.md").write_text("\n".join(lines))


def run(
    *,
    freeze_manifest: Path,
    ledger_root: Path,
    score_dirs: list[Path],
    score_files: list[Path],
    output_dir: Path,
    start: str,
    end: str,
) -> dict[str, Any]:
    start_ts = _as_utc(start)
    end_exclusive = _as_utc(end, is_end=True)
    freeze = pd.read_csv(freeze_manifest)
    heads = sorted(freeze["head"].astype(str).unique().tolist())
    baseline, ledger_audit = _collect_baseline_scores(ledger_root, start=start_ts, end_exclusive=end_exclusive)
    candidates, candidate_audit = _load_candidate_scores(
        heads,
        score_dirs,
        score_files,
        start=start_ts,
        end_exclusive=end_exclusive,
    )
    if baseline.empty:
        dual = pd.DataFrame(
            columns=[
                "head",
                "timestamp",
                "decision_ts",
                "symbol",
                "strategy_id",
                "baseline_score",
                "candidate_score",
            ]
        )
    else:
        dual = baseline.merge(candidates, on=["head", "timestamp", "symbol"], how="left")
    summary = _summary_by_head(dual, heads)
    daily = _daily_summary(dual, heads, start=start_ts, end_exclusive=end_exclusive)
    all_heads_have_baseline = bool(not summary.empty and (summary["baseline_rows"] > 0).all())
    all_heads_have_dual = bool(not summary.empty and summary["dual_complete"].all())
    any_baseline = bool(summary["baseline_rows"].sum() > 0) if not summary.empty else False
    any_candidate = bool(summary["candidate_rows_matched"].sum() > 0) if not summary.empty else False
    if all_heads_have_dual:
        status = "dual_scores_ready"
    elif any_baseline and any_candidate:
        status = "partial_dual_scores"
    elif any_baseline:
        status = "baseline_only_missing_candidate_scores"
    else:
        status = "no_prospective_scores_found"
    notes = [
        "Labels were not read or merged.",
        "Baseline scores come from live/shadow prediction ledgers and use calibrated_score, then meta_pred, then raw_prediction_score.",
    ]
    if not all_heads_have_baseline:
        notes.append("Baseline ledger coverage is incomplete for at least one frozen head in the requested window.")
    if not all_heads_have_dual:
        notes.append("Frozen contextual candidate score files were not available for all matched baseline rows.")
    audit = {
        "status": status,
        "freeze_manifest": str(freeze_manifest),
        "ledger_root": str(ledger_root),
        "window_start": start_ts.isoformat(),
        "window_end_exclusive": end_exclusive.isoformat(),
        "heads": heads,
        "score_dirs": [str(p) for p in score_dirs],
        "score_files": [str(p) for p in score_files],
        "all_heads_have_baseline": all_heads_have_baseline,
        "all_heads_have_dual_scores": all_heads_have_dual,
        "notes": notes,
        "ledger_audit": ledger_audit,
        "candidate_score_audit": candidate_audit,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    dual.to_parquet(output_dir / "prospective_dual_scores.parquet", index=False)
    summary.to_csv(output_dir / "prospective_dual_scoring_summary.csv", index=False)
    daily.to_csv(output_dir / "prospective_dual_scoring_daily_summary.csv", index=False)
    (output_dir / "prospective_dual_scoring_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True, default=_json_default) + "\n"
    )
    _write_report(output_dir, summary, daily, audit)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-manifest", type=Path, default=DEFAULT_FREEZE_MANIFEST)
    parser.add_argument("--ledger-root", type=Path, default=DEFAULT_LEDGER_ROOT)
    parser.add_argument("--score-dir", type=Path, action="append", default=[])
    parser.add_argument("--score-file", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--start", default="2026-06-16")
    parser.add_argument("--end", default="2026-06-22")
    args = parser.parse_args()
    audit = run(
        freeze_manifest=args.freeze_manifest,
        ledger_root=args.ledger_root,
        score_dirs=list(args.score_dir),
        score_files=list(args.score_file),
        output_dir=args.output_dir,
        start=str(args.start),
        end=str(args.end),
    )
    print(json.dumps({"status": audit["status"], "output_dir": str(args.output_dir)}, default=_json_default))


if __name__ == "__main__":
    main()
