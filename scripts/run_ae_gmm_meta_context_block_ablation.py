#!/usr/bin/env python3
"""Run train_meta smoke ablations over AE/GMM context feature blocks."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_ARMS = {
    "ctx_off": {"policy": "off", "blocks": "all"},
    "ctx_all_ae_gmm": {"policy": "ae_gmm_only", "blocks": "all"},
    "ctx_global": {"policy": "ae_gmm_only", "blocks": "global"},
    "ctx_side": {"policy": "ae_gmm_only", "blocks": "long,short"},
    "ctx_long": {"policy": "ae_gmm_only", "blocks": "long"},
    "ctx_short": {"policy": "ae_gmm_only", "blocks": "short"},
    "ctx_soft_prob": {"policy": "ae_gmm_only", "blocks": "soft_prob"},
    "ctx_distance": {"policy": "ae_gmm_only", "blocks": "distance"},
    "ctx_transition": {"policy": "ae_gmm_only", "blocks": "transition"},
    "drop_global": {"policy": "ae_gmm_only", "blocks": "-global"},
    "drop_long": {"policy": "ae_gmm_only", "blocks": "-long"},
    "drop_short": {"policy": "ae_gmm_only", "blocks": "-short"},
}


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        out = float(value)
        return out if np.isfinite(out) else None
    if pd.isna(value):
        return None
    return value


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def _best_row(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty or "mean_u" not in frame.columns:
        return {}
    work = frame.copy()
    for col in ("mean_u", "worst_month_mean_u", "bad_mae_1r_rate", "timeout_rate", "final_oracle_recall"):
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")
    sort_cols = [
        col
        for col in ("mean_u", "worst_month_mean_u", "final_oracle_recall", "bad_mae_1r_rate", "timeout_rate")
        if col in work.columns
    ]
    ascending = [
        False if col in {"mean_u", "worst_month_mean_u", "final_oracle_recall"} else True
        for col in sort_cols
    ]
    return work.sort_values(sort_cols, ascending=ascending, na_position="last").iloc[0].to_dict()


def _metric(row: dict[str, Any], key: str) -> float:
    try:
        return float(row.get(key, float("nan")))
    except Exception:
        return float("nan")


def _run_arm(
    *,
    arm_name: str,
    policy: str,
    blocks: str,
    candidate_ledger_path: Path,
    output_dir: Path,
    keep_fracs: str,
    max_side_share: float,
    min_train_rows: int,
    seed: int,
    execute: bool,
) -> dict[str, Any]:
    arm_dir = output_dir / arm_name
    cmd = [
        sys.executable,
        "scripts/run_gmm_train_meta_path_filter_smoke.py",
        "--candidate-ledger-path",
        str(candidate_ledger_path),
        "--output-dir",
        str(arm_dir),
        "--candidate-streams",
        "s8_lgbm_utility_ranker_stageA_rerank_side_cap_70",
        "--keep-fracs",
        str(keep_fracs),
        "--max-side-share",
        str(float(max_side_share)),
        "--min-train-rows",
        str(int(min_train_rows)),
        "--seeds",
        str(int(seed)),
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = "."
    env["PYTHONUNBUFFERED"] = "1"
    env["EPM_META_CONTEXT_FEATURE_POLICY"] = str(policy)
    env["EPM_META_CONTEXT_FEATURE_BLOCKS"] = str(blocks)
    record: dict[str, Any] = {
        "arm": arm_name,
        "policy": str(policy),
        "blocks": str(blocks),
        "output_dir": str(arm_dir),
        "cmd": cmd,
        "execute": bool(execute),
    }
    if execute:
        proc = subprocess.run(cmd, text=True, capture_output=True, check=False, env=env)
        record.update(
            {
                "returncode": int(proc.returncode),
                "stdout_tail": proc.stdout[-3000:],
                "stderr_tail": proc.stderr[-3000:],
            }
        )
        aggregate_path = arm_dir / "gmm_train_meta_path_filter_smoke_aggregate.csv"
        if proc.returncode != 0 and not aggregate_path.exists():
            raise RuntimeError(f"{arm_name} failed ({proc.returncode}): {proc.stderr[-2000:]}")
    manifest_path = arm_dir / "manifest.json"
    aggregate_path = arm_dir / "gmm_train_meta_path_filter_smoke_aggregate.csv"
    best = _best_row(_read_csv(aggregate_path)) if execute else {}
    record.update(
        {
            "manifest_path": str(manifest_path) if manifest_path.exists() else None,
            "aggregate_path": str(aggregate_path) if aggregate_path.exists() else None,
            "meta_context_feature_count": None,
            "mean_u": _metric(best, "mean_u"),
            "worst_month_mean_u": _metric(best, "worst_month_mean_u"),
            "bad_mae_1r_rate": _metric(best, "bad_mae_1r_rate"),
            "timeout_rate": _metric(best, "timeout_rate"),
            "final_oracle_recall": _metric(best, "final_oracle_recall"),
            "positive_months": _metric(best, "positive_months"),
        }
    )
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            record["meta_context_feature_count"] = int(manifest.get("meta_context_feature_count", 0) or 0)
        except Exception:
            pass
    return record


def run_ablation(
    *,
    candidate_ledger_path: Path,
    output_dir: Path,
    arms: list[str],
    keep_fracs: str,
    max_side_share: float,
    min_train_rows: int,
    seed: int,
    execute: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for arm_name in arms:
        if arm_name not in DEFAULT_ARMS:
            raise ValueError(f"Unknown arm {arm_name!r}; options={sorted(DEFAULT_ARMS)}")
        spec = DEFAULT_ARMS[arm_name]
        records.append(
            _run_arm(
                arm_name=arm_name,
                policy=str(spec["policy"]),
                blocks=str(spec["blocks"]),
                candidate_ledger_path=candidate_ledger_path,
                output_dir=output_dir,
                keep_fracs=keep_fracs,
                max_side_share=float(max_side_share),
                min_train_rows=int(min_train_rows),
                seed=int(seed),
                execute=bool(execute),
            )
        )
    rows = pd.DataFrame(records)
    if not rows.empty:
        baseline = rows.loc[rows["arm"].eq("ctx_off")]
        if not baseline.empty:
            base = baseline.iloc[0]
            for col in ("mean_u", "worst_month_mean_u", "bad_mae_1r_rate", "timeout_rate", "final_oracle_recall"):
                rows[f"{col}_delta_vs_ctx_off"] = pd.to_numeric(rows[col], errors="coerce") - float(base[col])
    csv_path = output_dir / "ae_gmm_meta_context_block_ablation.csv"
    json_path = output_dir / "ae_gmm_meta_context_block_ablation.json"
    md_path = output_dir / "ae_gmm_meta_context_block_ablation.md"
    rows.to_csv(csv_path, index=False)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_ledger_path": str(candidate_ledger_path),
        "execute": bool(execute),
        "arms": records,
        "outputs": {"csv": str(csv_path), "json": str(json_path), "markdown": str(md_path)},
    }
    json_path.write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    cols = [
        "arm",
        "policy",
        "blocks",
        "meta_context_feature_count",
        "mean_u",
        "worst_month_mean_u",
        "bad_mae_1r_rate",
        "timeout_rate",
        "final_oracle_recall",
        "mean_u_delta_vs_ctx_off",
        "bad_mae_1r_rate_delta_vs_ctx_off",
        "timeout_rate_delta_vs_ctx_off",
    ]
    present = [col for col in cols if col in rows.columns]
    lines = ["# AE/GMM Meta Context Block Ablation", "", f"- Candidate ledger: `{candidate_ledger_path}`", ""]
    if present:
        lines.append(rows[present].to_markdown(index=False))
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-ledger-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--arms", default=",".join(DEFAULT_ARMS.keys()))
    parser.add_argument("--keep-fracs", default="0.50,0.60,0.70,0.80")
    parser.add_argument("--max-side-share", type=float, default=0.70)
    parser.add_argument("--min-train-rows", type=int, default=500)
    parser.add_argument("--seed", type=int, default=913)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    arms = [part.strip() for part in str(args.arms).split(",") if part.strip()]
    manifest = run_ablation(
        candidate_ledger_path=args.candidate_ledger_path,
        output_dir=args.output_dir,
        arms=arms,
        keep_fracs=str(args.keep_fracs),
        max_side_share=float(args.max_side_share),
        min_train_rows=int(args.min_train_rows),
        seed=int(args.seed),
        execute=bool(args.execute),
    )
    print(json.dumps(_json_safe(manifest), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
