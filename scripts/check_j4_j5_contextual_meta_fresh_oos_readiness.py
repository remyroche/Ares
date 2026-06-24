"""Check whether frozen J4/J5 contextual meta decisions have fresh OOS rows.

This script is intentionally conservative.  It does not turn slightly later
training labels into an OOS result.  A head is ready only when it has:

1. a frozen decision in the all-head freeze manifest;
2. labelled rows after the effective development boundary plus a guard period;
3. candidate score rows for the frozen model over that same guarded interval.

The output is a readiness artifact, not a model-selection artifact.
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
DEFAULT_LABEL_DIR = Path("data_perp/artifacts/20260617_225212_current4_final_fit/labels")
DEFAULT_OUTPUT_DIR = Path("data_perp/reports/j4_j5_contextual_meta_fresh_oos_readiness_20260623")
HEAD_LABEL_PREFIX = {
    "long_bars": "train_bars_",
    "long_dist": "train_dist_",
    "short_asset": "train_asset_",
    "short_boll": "train_bollinger_",
}
TIME_COLUMNS = ("timestamp", "__ts__", "ts", "entry_time", "bar_time")


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


def _find_one(root: Path, pattern: str) -> Path | None:
    matches = sorted(root.glob(pattern)) if root.exists() else []
    return matches[0] if matches else None


def _parquet_time_column(path: Path) -> str | None:
    try:
        cols = list(pd.read_parquet(path, nrows=0).columns)
    except TypeError:
        # Older pandas/pyarrow combinations do not support nrows here.
        try:
            cols = list(pd.read_parquet(path).head(0).columns)
        except Exception:
            return None
    except Exception:
        return None
    for col in TIME_COLUMNS:
        if col in cols:
            return col
    return None


def _time_stats(path: Path | None, after: pd.Timestamp | None = None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {"path": "", "rows": 0, "time_column": "", "min_ts": "", "max_ts": "", "rows_after": 0}
    col = _parquet_time_column(path)
    if not col:
        return {"path": str(path), "rows": 0, "time_column": "", "min_ts": "", "max_ts": "", "rows_after": 0}
    try:
        df = pd.read_parquet(path, columns=[col])
    except Exception:
        return {"path": str(path), "rows": 0, "time_column": col, "min_ts": "", "max_ts": "", "rows_after": 0}
    ts = pd.to_datetime(df[col], utc=True, errors="coerce").dropna()
    if ts.empty:
        return {"path": str(path), "rows": int(len(df)), "time_column": col, "min_ts": "", "max_ts": "", "rows_after": 0}
    rows_after = int((ts > after).sum()) if after is not None else 0
    return {
        "path": str(path),
        "rows": int(len(df)),
        "time_column": col,
        "min_ts": ts.min().isoformat(),
        "max_ts": ts.max().isoformat(),
        "rows_after": rows_after,
    }


def _score_stats(score_dirs: list[Path], head: str, after: pd.Timestamp) -> dict[str, Any]:
    rows_after = 0
    max_ts: pd.Timestamp | None = None
    paths: list[str] = []
    for score_dir in score_dirs:
        if not score_dir.exists():
            continue
        for path in sorted(score_dir.glob(f"*{head}*.parquet")):
            stats = _time_stats(path, after)
            rows_after += int(stats["rows_after"])
            paths.append(str(path))
            if stats["max_ts"]:
                ts = pd.Timestamp(stats["max_ts"])
                max_ts = ts if max_ts is None else max(max_ts, ts)
    return {
        "candidate_score_paths": ";".join(paths),
        "candidate_score_rows_after_guard": rows_after,
        "candidate_score_max_ts": "" if max_ts is None else max_ts.isoformat(),
    }


def build_readiness(
    freeze_manifest_path: Path,
    label_dir: Path,
    score_dirs: list[Path],
    *,
    min_later_hours: float,
    min_rows_per_head: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    freeze = pd.read_csv(freeze_manifest_path)
    rows: list[dict[str, Any]] = []
    for _, frozen in freeze.iterrows():
        head = str(frozen["head"])
        boundary = pd.Timestamp(str(frozen["effective_fresh_oos_after"]))
        if boundary.tzinfo is None:
            boundary = boundary.tz_localize("UTC")
        guard_start = boundary + pd.Timedelta(hours=float(min_later_hours))
        label_path = _find_one(label_dir, f"{HEAD_LABEL_PREFIX.get(head, f'train_{head}_')}*.parquet")
        label_all = _time_stats(label_path, boundary)
        label_guarded = _time_stats(label_path, guard_start)
        score = _score_stats(score_dirs, head, guard_start)
        has_labels = int(label_guarded["rows_after"]) >= int(min_rows_per_head)
        has_scores = int(score["candidate_score_rows_after_guard"]) >= int(min_rows_per_head)
        rows.append(
            {
                "head": head,
                "selected_contextual_feature_arm": frozen.get("selected_contextual_feature_arm", ""),
                "selected_capacity_config": frozen.get("selected_capacity_config", ""),
                "selected_distillation_variant": frozen.get("selected_distillation_variant", ""),
                "effective_fresh_oos_after": boundary.isoformat(),
                "guarded_fresh_oos_start": guard_start.isoformat(),
                "min_later_hours": float(min_later_hours),
                "min_rows_per_head": int(min_rows_per_head),
                "label_path": label_all["path"],
                "label_min_ts": label_all["min_ts"],
                "label_max_ts": label_all["max_ts"],
                "label_rows_after_boundary": int(label_all["rows_after"]),
                "label_rows_after_guard": int(label_guarded["rows_after"]),
                "has_fresh_labels": bool(has_labels),
                **score,
                "has_candidate_scores": bool(has_scores),
                "ready_for_fresh_oos_confirmation": bool(has_labels and has_scores),
            }
        )
    out = pd.DataFrame(rows)
    items = [
        {
            "requirement": "freeze_manifest_present",
            "status": "passed" if freeze_manifest_path.exists() else "failed",
            "metrics": {"path": str(freeze_manifest_path), "rows": int(len(freeze))},
        },
        {
            "requirement": "fresh_labels_after_guard",
            "status": "passed" if not out.empty and out["has_fresh_labels"].all() else "not_ready",
            "metrics": {
                row["head"]: {
                    "label_rows_after_guard": int(row["label_rows_after_guard"]),
                    "label_max_ts": row["label_max_ts"],
                    "guarded_fresh_oos_start": row["guarded_fresh_oos_start"],
                }
                for row in out.to_dict(orient="records")
            },
        },
        {
            "requirement": "candidate_scores_after_guard",
            "status": "passed" if not out.empty and out["has_candidate_scores"].all() else "not_ready",
            "metrics": {
                row["head"]: int(row["candidate_score_rows_after_guard"]) for row in out.to_dict(orient="records")
            },
        },
        {
            "requirement": "all_heads_ready_for_single_fresh_oos_evaluation",
            "status": "passed" if not out.empty and out["ready_for_fresh_oos_confirmation"].all() else "not_ready",
            "metrics": {
                "ready_heads": sorted(
                    out.loc[out["ready_for_fresh_oos_confirmation"], "head"].astype(str).tolist()
                )
                if not out.empty
                else [],
                "not_ready_heads": sorted(
                    out.loc[~out["ready_for_fresh_oos_confirmation"], "head"].astype(str).tolist()
                )
                if not out.empty
                else [],
            },
        },
    ]
    audit = {
        "status": "ready" if all(item["status"] == "passed" for item in items) else "not_ready",
        "items": items,
    }
    return out, audit


def _write_report(out_dir: Path, readiness: pd.DataFrame, audit: dict[str, Any]) -> None:
    lines = [
        "# J4/J5 Fresh OOS Readiness",
        "",
        "This is a readiness check only. It does not consume fresh OOS for model selection.",
        "",
        "## Audit",
        "",
        pd.DataFrame(audit.get("items", [])).to_markdown(index=False),
        "",
    ]
    if not readiness.empty:
        cols = [
            "head",
            "effective_fresh_oos_after",
            "guarded_fresh_oos_start",
            "label_max_ts",
            "label_rows_after_guard",
            "candidate_score_rows_after_guard",
            "ready_for_fresh_oos_confirmation",
        ]
        lines.extend(["## By Head", "", readiness[cols].to_markdown(index=False), ""])
    (out_dir / "j4_j5_contextual_meta_fresh_oos_readiness_report.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-manifest", type=Path, default=DEFAULT_FREEZE_MANIFEST)
    parser.add_argument("--label-dir", type=Path, default=DEFAULT_LABEL_DIR)
    parser.add_argument("--score-dir", action="append", type=Path, default=None)
    parser.add_argument("--min-later-hours", type=float, default=24.0)
    parser.add_argument("--min-rows-per-head", type=int, default=1000)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--fail-if-not-ready", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    readiness, audit = build_readiness(
        args.freeze_manifest,
        args.label_dir,
        list(args.score_dir or []),
        min_later_hours=float(args.min_later_hours),
        min_rows_per_head=int(args.min_rows_per_head),
    )
    readiness.to_csv(args.output_dir / "j4_j5_contextual_meta_fresh_oos_readiness_by_head.csv", index=False)
    (args.output_dir / "j4_j5_contextual_meta_fresh_oos_readiness_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True, default=_json_default)
    )
    (args.output_dir / "j4_j5_contextual_meta_fresh_oos_readiness_inputs.json").write_text(
        json.dumps(
            {
                "freeze_manifest": str(args.freeze_manifest),
                "label_dir": str(args.label_dir),
                "score_dirs": [str(x) for x in list(args.score_dir or [])],
                "min_later_hours": float(args.min_later_hours),
                "min_rows_per_head": int(args.min_rows_per_head),
            },
            indent=2,
            sort_keys=True,
        )
    )
    _write_report(args.output_dir, readiness, audit)
    print(f"[j4_j5_fresh_oos] readiness={audit['status']} wrote {args.output_dir}", flush=True)
    if args.fail_if_not_ready and audit["status"] != "ready":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
