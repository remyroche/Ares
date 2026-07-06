#!/usr/bin/env python3
"""Scan frozen challenger candidate ledgers for replay-ready post-freeze rows.

The scanner is intentionally report-only.  It answers three questions before a
delayed/prospective portfolio replay is attempted:

1. Does the baseline ledger have labelled replay rows after the frozen cutoff?
2. Do challenger ledgers have the exact same candidate universe/order?
3. Does any challenger actually bind after the cutoff?

If all checks pass and at least one challenger binds, the report prints the
`replay_frozen_smooth_penalty_dual_scoring.py` command to run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REQUIRED_KEYS = ("timestamp", "strategy_id", "symbol")
REPLAY_COLUMNS = (
    "rank_pct",
    "strategy_rank_pct",
    "normalized_rank_score",
    "calibrated_score",
    "entry_price",
    "exit_timestamp",
    "exit_price",
    "net_return",
    "gross_return",
    "barrier_pct",
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(float(value)) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _label_from_path(path: Path) -> str:
    name = path.name
    for suffix in (
        "_smooth_penalty_combo_candidates.parquet",
        "_candidates.parquet",
        ".parquet",
    ):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def _load(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    missing = sorted(set(REQUIRED_KEYS) - set(frame.columns))
    if missing:
        raise ValueError(f"{path} missing required keys: {missing}")
    out = frame.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out = out[out["timestamp"].notna()].copy()
    out["strategy_id"] = out["strategy_id"].astype(str)
    out["symbol"] = out["symbol"].astype(str)
    if "portfolio_rank_adjustment" not in out.columns:
        out["portfolio_rank_adjustment"] = np.float32(0.0)
    else:
        out["portfolio_rank_adjustment"] = (
            pd.to_numeric(out["portfolio_rank_adjustment"], errors="coerce")
            .fillna(0.0)
            .astype("float32")
        )
    return out.sort_values(list(REQUIRED_KEYS)).reset_index(drop=True)


def _key(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[:, list(REQUIRED_KEYS)].reset_index(drop=True)


def _candidate_paths(values: list[str], candidate_glob: str | None) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for raw in values:
        if "=" in raw:
            label, path = raw.split("=", 1)
            paths[label.strip()] = Path(path)
        else:
            path = Path(raw)
            paths[_label_from_path(path)] = path
    if candidate_glob:
        for path in sorted(Path().glob(candidate_glob)):
            paths.setdefault(_label_from_path(path), path)
    return paths


def _audit_frame(label: str, path: Path, frame: pd.DataFrame, eval_start: pd.Timestamp, eval_end: pd.Timestamp | None) -> dict[str, Any]:
    mask = frame["timestamp"].ge(eval_start)
    if eval_end is not None:
        mask &= frame["timestamp"].le(eval_end)
    post = frame.loc[mask].copy()
    adj = pd.to_numeric(post["portfolio_rank_adjustment"], errors="coerce").fillna(0.0)
    replay_present = [col for col in REPLAY_COLUMNS if col in frame.columns]
    replay_missing = [col for col in REPLAY_COLUMNS if col not in frame.columns]
    return {
        "label": label,
        "path": str(path),
        "sha256": _sha256(path),
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "timestamp_min": frame["timestamp"].min(),
        "timestamp_max": frame["timestamp"].max(),
        "eval_rows": int(len(post)),
        "eval_timestamp_min": post["timestamp"].min() if len(post) else None,
        "eval_timestamp_max": post["timestamp"].max() if len(post) else None,
        "replay_column_count": int(len(replay_present)),
        "missing_replay_columns": ",".join(replay_missing),
        "adjusted_eval_rows": int(adj.ne(0.0).sum()),
        "adjusted_eval_share": float(adj.ne(0.0).mean()) if len(adj) else 0.0,
        "min_eval_adjustment": float(adj.min()) if len(adj) else 0.0,
        "mean_eval_adjustment_on_adjusted": float(adj[adj.ne(0.0)].mean()) if adj.ne(0.0).any() else 0.0,
    }


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    if frame.empty:
        return "_No rows._"
    view = frame.loc[:, [col for col in columns if col in frame.columns]].copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{x:,.6g}")
    return view.to_markdown(index=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", action="append", default=[], help="label=path or path. Repeatable.")
    parser.add_argument("--candidate-glob", default="")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--eval-start", default="2026-06-27T13:00:00+00:00")
    parser.add_argument("--eval-end", default="")
    parser.add_argument("--market-mode", default="perps", choices=["spot", "perps"])
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    eval_start = pd.Timestamp(args.eval_start, tz="UTC")
    eval_end = pd.Timestamp(args.eval_end, tz="UTC") if args.eval_end else None
    baseline = _load(args.baseline)
    candidates = _candidate_paths(list(args.candidate), args.candidate_glob or None)
    if not candidates:
        raise ValueError("Provide --candidate or --candidate-glob")

    base_audit = _audit_frame("baseline", args.baseline, baseline, eval_start, eval_end)
    base_key = _key(baseline)
    rows = [base_audit]
    commands: list[str] = []
    eligible: list[str] = []
    for label, path in candidates.items():
        frame = _load(path)
        row = _audit_frame(label, path, frame, eval_start, eval_end)
        row["same_row_count_as_baseline"] = bool(len(frame) == len(baseline))
        row["same_universe_order_as_baseline"] = bool(len(frame) == len(baseline) and _key(frame).equals(base_key))
        row["replay_ready"] = bool(
            base_audit["eval_rows"] > 0
            and row["eval_rows"] == base_audit["eval_rows"]
            and row["same_universe_order_as_baseline"]
            and row["replay_column_count"] == len(REPLAY_COLUMNS)
        )
        row["informative_after_cutoff"] = bool(row["replay_ready"] and row["adjusted_eval_rows"] > 0)
        if row["replay_ready"]:
            eligible.append(label)
            if row["informative_after_cutoff"]:
                commands.append(f"--candidate {label}={path}")
        rows.append(row)

    audit = pd.DataFrame(rows)
    for col in ("same_universe_order_as_baseline", "replay_ready", "informative_after_cutoff"):
        if col not in audit.columns:
            audit[col] = False
        audit[col] = audit[col].where(audit[col].notna(), False).astype(bool)
    audit.to_csv(args.output_dir / "frozen_dual_scoring_readiness_audit.csv", index=False)
    replay_command = ""
    if commands:
        replay_command = (
            "env PYTHONUNBUFFERED=1 PYTHONPATH=. python3 -u "
            "scripts/replay_frozen_smooth_penalty_dual_scoring.py "
            f"--baseline {args.baseline} "
            f"--output-dir {args.output_dir / 'dual_scoring_replay'} "
            f"--eval-start {eval_start.isoformat()} "
            f"--market-mode {args.market_mode} "
            + " ".join(commands)
        )
        if eval_end is not None:
            replay_command += f" --eval-end {eval_end.isoformat()}"

    manifest = {
        "generated_by": "scan_frozen_dual_scoring_readiness",
        "baseline": str(args.baseline),
        "candidate_count": int(len(candidates)),
        "eval_start": eval_start.isoformat(),
        "eval_end": eval_end.isoformat() if eval_end is not None else "",
        "baseline_eval_rows": int(base_audit["eval_rows"]),
        "replay_ready_candidates": eligible,
        "informative_candidates": sorted(
            audit.loc[audit["informative_after_cutoff"], "label"].astype(str).tolist()
        ),
        "replay_command": replay_command,
    }
    (args.output_dir / "frozen_dual_scoring_readiness_manifest.json").write_text(
        json.dumps(_json_safe(manifest), indent=2, sort_keys=True) + "\n"
    )

    lines = [
        "# Frozen Dual-Scoring Readiness",
        "",
        f"Evaluation start: `{eval_start.isoformat()}`",
        f"Evaluation end: `{manifest['eval_end'] or 'open'}`",
        f"Baseline: `{args.baseline}`",
        "",
        "## Audit",
        "",
        _markdown_table(
            audit,
            [
                "label",
                "rows",
                "timestamp_min",
                "timestamp_max",
                "eval_rows",
                "replay_column_count",
                "same_universe_order_as_baseline",
                "replay_ready",
                "adjusted_eval_rows",
                "informative_after_cutoff",
            ],
        ),
        "",
        "## Replay Command",
        "",
        f"```bash\n{replay_command or '# No informative post-cutoff challenger rows yet.'}\n```",
    ]
    (args.output_dir / "frozen_dual_scoring_readiness_report.md").write_text("\n".join(lines) + "\n")
    print(args.output_dir / "frozen_dual_scoring_readiness_report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
