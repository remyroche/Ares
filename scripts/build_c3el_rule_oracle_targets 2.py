#!/usr/bin/env python3
"""Build exact-state oracle target queues from C3el shadow-monitor rule tags.

This is a target materializer only.  It consumes `tagged_score_rows.csv` from
`monitor_c3el_rule_candidates.py`, filters rows where a named monitoring rule
fired, excludes actions that already have exact-state labels, and writes a
small capped `target_actions.csv` suitable for
`run_exact_state_counterfactual_oracle.py --target-actions`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


KEYS = ["timestamp", "strategy_id", "action_family", "action_value"]
DEFAULT_RULE = "rule_p80_d320_cooldown_lte_38_5"


def _read_frame(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _normalise_action_keys(frame: pd.DataFrame, *, action_value: float) -> pd.DataFrame:
    out = frame.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out = out.loc[out["timestamp"].notna()].copy()
    out["strategy_id"] = out["strategy_id"].astype(str)
    if "action_family" not in out.columns:
        out["action_family"] = "size"
    out["action_family"] = out["action_family"].astype(str)
    if "action_value" not in out.columns:
        if "multiplier" in out.columns:
            out["action_value"] = out["multiplier"]
        else:
            out["action_value"] = action_value
    out["action_value"] = pd.to_numeric(out["action_value"], errors="coerce").fillna(action_value).round(6)
    return out


def _load_existing(paths: list[Path], *, action_value: float) -> pd.DataFrame:
    frames = []
    for path in paths:
        if not path.exists():
            continue
        frames.append(_normalise_action_keys(_read_frame(path), action_value=action_value)[KEYS])
    if not frames:
        return pd.DataFrame(columns=KEYS)
    return pd.concat(frames, ignore_index=True).drop_duplicates(KEYS).reset_index(drop=True)


def _priority(frame: pd.DataFrame) -> pd.Series:
    p = pd.to_numeric(frame.get("p_intervene"), errors="coerce").fillna(0.0)
    delta = pd.to_numeric(frame.get("pred_action_delta_J"), errors="coerce").fillna(0.0)
    condition_count = pd.to_numeric(frame.get("monitor_condition_count"), errors="coerce").fillna(0.0)
    return 1000.0 * p + delta + 50.0 * condition_count


def build_targets(
    *,
    tagged_path: Path,
    existing_label_paths: list[Path],
    out_dir: Path,
    rule: str = DEFAULT_RULE,
    action_value: float = 0.0,
    max_targets: int = 40,
    max_per_day: int = 6,
) -> dict[str, Any]:
    tagged = _normalise_action_keys(_read_frame(tagged_path), action_value=action_value)
    if rule not in tagged.columns:
        raise ValueError(f"Tagged score rows are missing requested rule column: {rule}")
    existing = _load_existing(existing_label_paths, action_value=action_value)

    candidates = tagged.loc[tagged[rule].fillna(False).astype(bool)].copy()
    candidates = candidates.drop_duplicates(KEYS, keep="last")
    if not existing.empty and not candidates.empty:
        candidates = candidates.merge(existing.assign(already_labeled=True), on=KEYS, how="left")
        candidates["already_labeled"] = candidates["already_labeled"].eq(True)
        candidates = candidates.loc[~candidates["already_labeled"]].copy()
    else:
        candidates["already_labeled"] = False

    if not candidates.empty:
        candidates["day"] = candidates["timestamp"].dt.floor("D")
        candidates["target_priority"] = _priority(candidates)
        capped_rows = []
        for _, group in candidates.sort_values("target_priority", ascending=False).groupby("day", sort=True):
            capped_rows.append(group.head(int(max_per_day)))
        capped = pd.concat(capped_rows, ignore_index=True) if capped_rows else candidates.head(0).copy()
        capped = capped.sort_values(["target_priority", "timestamp"], ascending=[False, True]).head(int(max_targets))
    else:
        candidates["day"] = pd.NaT
        candidates["target_priority"] = np.nan
        capped = candidates.copy()

    out_dir.mkdir(parents=True, exist_ok=True)
    output_cols = [
        "timestamp",
        "strategy_id",
        "action_family",
        "action_value",
        "p_intervene",
        "pred_action_delta_J",
        "monitor_condition_count",
        rule,
        "target_priority",
    ]
    for col in output_cols:
        if col not in capped.columns:
            capped[col] = np.nan
    target_actions = capped[output_cols].copy()
    target_actions.to_csv(out_dir / "target_actions.csv", index=False)
    candidates.to_csv(out_dir / "candidate_pool.csv", index=False)

    if capped.empty:
        by_day = pd.DataFrame(columns=["day", "target_count", "p_intervene_mean", "pred_delta_mean"])
    else:
        by_day = (
            capped.groupby("day", dropna=False)
            .agg(
                target_count=("timestamp", "size"),
                p_intervene_mean=("p_intervene", "mean"),
                pred_delta_mean=("pred_action_delta_J", "mean"),
                condition_count_mean=("monitor_condition_count", "mean"),
            )
            .reset_index()
        )
    by_day.to_csv(out_dir / "target_summary_by_day.csv", index=False)

    manifest = {
        "generated_by": "build_c3el_rule_oracle_targets",
        "tagged_path": str(tagged_path),
        "existing_label_paths": [str(p) for p in existing_label_paths],
        "rule": str(rule),
        "action_value": float(action_value),
        "max_targets": int(max_targets),
        "max_per_day": int(max_per_day),
        "tagged_rows": int(len(tagged)),
        "rule_candidate_rows_before_existing_exclusion": int(tagged[rule].fillna(False).astype(bool).sum()),
        "existing_labeled_actions": int(len(existing)),
        "candidate_pool_rows": int(len(candidates)),
        "target_rows": int(len(target_actions)),
        "target_min_timestamp": None if target_actions.empty else target_actions["timestamp"].min(),
        "target_max_timestamp": None if target_actions.empty else target_actions["timestamp"].max(),
    }
    (out_dir / "manifest.json").write_text(json.dumps(_json_safe(manifest), indent=2, sort_keys=True))

    lines = [
        "# C3el Rule Exact-State Target Queue",
        "",
        f"Rule: `{rule}`",
        f"Tagged rows: `{len(tagged)}`",
        f"Rule rows before existing-label exclusion: `{manifest['rule_candidate_rows_before_existing_exclusion']}`",
        f"Existing labeled actions: `{len(existing)}`",
        f"Rows in candidate pool after exclusion: `{len(candidates)}`",
        f"Rows selected for oracle: `{len(target_actions)}`",
        f"Per-day cap: `{max_per_day}`",
        f"Global cap: `{max_targets}`",
        "",
        "## By Day",
        "",
    ]
    lines.append("No selected targets." if by_day.empty else by_day.to_markdown(index=False, floatfmt=".4f"))
    lines.extend(["", "## Top Targets", ""])
    if target_actions.empty:
        lines.append("No selected targets.")
    else:
        lines.append(target_actions.head(25).to_markdown(index=False, floatfmt=".4f"))
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    return manifest


def _json_safe(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tagged", type=Path, required=True)
    parser.add_argument("--existing-labels", type=Path, action="append", default=[])
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--rule", default=DEFAULT_RULE)
    parser.add_argument("--action-value", type=float, default=0.0)
    parser.add_argument("--max-targets", type=int, default=40)
    parser.add_argument("--max-per-day", type=int, default=6)
    args = parser.parse_args()
    manifest = build_targets(
        tagged_path=args.tagged,
        existing_label_paths=list(args.existing_labels or []),
        out_dir=args.out_dir,
        rule=args.rule,
        action_value=args.action_value,
        max_targets=args.max_targets,
        max_per_day=args.max_per_day,
    )
    print((args.out_dir / "summary.md").read_text())
    print(json.dumps(_json_safe(manifest), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
