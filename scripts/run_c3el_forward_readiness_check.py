#!/usr/bin/env python3
"""Run the C3el forward monitor and exact-state target materializer.

This is the lightweight forward check to run whenever new short_asset score
rows are available.  It does not run exact-state replay.  It tags C3el rules,
excludes already labeled actions, writes a capped target queue, and records
whether a replay job is warranted.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.build_c3el_rule_oracle_targets import DEFAULT_RULE, KEYS, build_targets
from scripts.monitor_c3el_rule_candidates import load_scored_features, summarize_rules, tag_rules, write_report


DEFAULT_OUT_DIR = Path("data_perp/reports/c3el_forward_readiness_check_20260628")
ROBUST_RULE = "rule_p80_d320_cooldown_lte_38_5_open_or_cooldown_share_lte_0_3949"
DEFAULT_RULES = [DEFAULT_RULE, ROBUST_RULE]


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


def _read_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _rule_slug(rule: str) -> str:
    text = str(rule).strip().lower().replace("rule_", "")
    safe = "".join(ch if ch.isalnum() else "_" for ch in text)
    while "__" in safe:
        safe = safe.replace("__", "_")
    return safe.strip("_")[:96] or "rule"


def _normalise_rules(rule: str | list[str] | tuple[str, ...] | None) -> list[str]:
    if rule is None:
        values = DEFAULT_RULES
    elif isinstance(rule, str):
        values = [rule]
    else:
        values = list(rule)
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        item = str(value).strip()
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    if not out:
        raise ValueError("At least one C3el monitoring rule is required")
    return out


def _unique_action_count(paths: list[Path]) -> int:
    frames = []
    for path in paths:
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        if frame.empty or any(col not in frame.columns for col in KEYS):
            continue
        frame = frame[KEYS].copy()
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
        frame = frame.loc[frame["timestamp"].notna()].copy()
        frame["strategy_id"] = frame["strategy_id"].astype(str)
        frame["action_family"] = frame["action_family"].astype(str)
        frame["action_value"] = pd.to_numeric(frame["action_value"], errors="coerce").round(6)
        frames.append(frame)
    if not frames:
        return 0
    return int(pd.concat(frames, ignore_index=True).drop_duplicates(KEYS).shape[0])


def _unique_tagged_rule_count(tagged: pd.DataFrame, rules: list[str]) -> int:
    if tagged.empty:
        return 0
    missing = [rule for rule in rules if rule not in tagged.columns]
    if missing:
        raise ValueError(f"Tagged score rows are missing requested rule columns: {missing}")
    mask = tagged[rules].fillna(False).astype(bool).any(axis=1)
    selected = tagged.loc[mask].copy()
    if selected.empty:
        return 0
    return int(selected.drop_duplicates(KEYS).shape[0])


def run_check(
    *,
    scores: Path,
    action_features: Path,
    existing_labels: list[Path],
    out_dir: Path,
    head: str = "short_asset",
    action_value: float = 0.0,
    rule: str | list[str] | tuple[str, ...] | None = None,
    max_targets: int = 40,
    max_per_day: int = 6,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    monitor_dir = out_dir / "monitor"
    rules = _normalise_rules(rule)

    frame = load_scored_features(scores, action_features, action_value=action_value, head=head)
    tagged = tag_rules(frame)
    summary, by_day = summarize_rules(tagged)
    write_report(tagged, summary, by_day, monitor_dir)

    target_manifests: list[dict[str, Any]] = []
    target_dirs: list[Path] = []
    for idx, current_rule in enumerate(rules):
        targets_dir = out_dir / "targets" if idx == 0 else out_dir / f"targets_{_rule_slug(current_rule)}"
        target_dirs.append(targets_dir)
        target_manifests.append(
            build_targets(
                tagged_path=monitor_dir / "tagged_score_rows.csv",
                existing_label_paths=existing_labels,
                out_dir=targets_dir,
                rule=current_rule,
                action_value=action_value,
                max_targets=max_targets,
                max_per_day=max_per_day,
            )
        )

    monitor_manifest = _read_manifest(monitor_dir / "manifest.json")
    target_rows = _unique_action_count([target_dir / "target_actions.csv" for target_dir in target_dirs])
    rule_rows_sum_by_rule = int(
        sum(int(m.get("rule_candidate_rows_before_existing_exclusion", 0)) for m in target_manifests)
    )
    rule_rows = _unique_tagged_rule_count(tagged, rules)
    candidate_pool_rows = _unique_action_count([target_dir / "candidate_pool.csv" for target_dir in target_dirs])
    decision = "run_exact_state_replay" if target_rows > 0 else "no_replay_wait_for_new_firings"
    target_rows_by_rule = {str(m.get("rule")): int(m.get("target_rows", 0)) for m in target_manifests}
    candidate_pool_rows_by_rule = {str(m.get("rule")): int(m.get("candidate_pool_rows", 0)) for m in target_manifests}
    rule_rows_by_rule = {
        str(m.get("rule")): int(m.get("rule_candidate_rows_before_existing_exclusion", 0))
        for m in target_manifests
    }

    manifest = {
        "generated_by": "run_c3el_forward_readiness_check",
        "scores": str(scores),
        "action_features": str(action_features),
        "existing_labels": [str(path) for path in existing_labels],
        "head": str(head),
        "action_value": float(action_value),
        "rule": str(rules[0]),
        "rules": rules,
        "decision": decision,
        "score_rows": int(monitor_manifest.get("rows", len(tagged))),
        "feature_rows_matched": int(monitor_manifest.get("feature_rows_matched", 0)),
        "rule_rows_before_existing_exclusion": rule_rows,
        "rule_rows_before_existing_exclusion_sum_by_rule": rule_rows_sum_by_rule,
        "candidate_pool_rows": candidate_pool_rows,
        "target_rows": target_rows,
        "rule_rows_by_rule": rule_rows_by_rule,
        "candidate_pool_rows_by_rule": candidate_pool_rows_by_rule,
        "target_rows_by_rule": target_rows_by_rule,
        "outputs": {
            "monitor_summary": str(monitor_dir / "summary.md"),
            "target_summaries": {
                str(m.get("rule")): str(target_dirs[i] / "summary.md")
                for i, m in enumerate(target_manifests)
            },
            "target_actions": {
                str(m.get("rule")): str(target_dirs[i] / "target_actions.csv")
                for i, m in enumerate(target_manifests)
            },
            "summary": str(out_dir / "summary.md"),
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(_json_safe(manifest), indent=2, sort_keys=True))
    _write_summary(out_dir / "summary.md", manifest)
    return manifest


def _write_summary(path: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# C3el Forward Readiness Check",
        "",
        f"Decision: `{manifest['decision']}`",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Score rows | {manifest['score_rows']} |",
        f"| Feature rows matched | {manifest['feature_rows_matched']} |",
        f"| Unique rule rows before existing-label exclusion | {manifest['rule_rows_before_existing_exclusion']} |",
        f"| Sum of per-rule rows before existing-label exclusion | {manifest['rule_rows_before_existing_exclusion_sum_by_rule']} |",
        f"| Candidate pool rows after exclusion | {manifest['candidate_pool_rows']} |",
        f"| Selected target rows | {manifest['target_rows']} |",
        "",
        "## Rule",
        "",
        ", ".join(f"`{rule}`" for rule in manifest.get("rules", [manifest["rule"]])),
        "",
        "## Targets By Rule",
        "",
        "| Rule | Rule rows | Candidate pool rows | Selected target rows |",
        "|---|---:|---:|---:|",
    ]
    for rule in manifest.get("rules", [manifest["rule"]]):
        lines.append(
            f"| `{rule}` | {manifest['rule_rows_by_rule'].get(rule, 0)} | "
            f"{manifest['candidate_pool_rows_by_rule'].get(rule, 0)} | {manifest['target_rows_by_rule'].get(rule, 0)} |"
        )
    lines.extend(
        [
        "",
        "## Outputs",
        "",
        f"- Monitor summary: `{manifest['outputs']['monitor_summary']}`",
        ]
    )
    for rule, target_summary in manifest["outputs"]["target_summaries"].items():
        target_actions = manifest["outputs"]["target_actions"][rule]
        lines.append(f"- `{rule}` target summary: `{target_summary}`")
        lines.append(f"- `{rule}` target actions: `{target_actions}`")
    lines.append("")
    if manifest["decision"] == "run_exact_state_replay":
        lines.extend(
            [
                "## Next Action",
                "",
                "Run `scripts/run_exact_state_counterfactual_oracle.py` with the generated target-actions file.",
            ]
        )
    else:
        lines.extend(
            [
                "## Next Action",
                "",
                "No replay is warranted. Wait for new forward score rows that fire the preferred C3el rule.",
            ]
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--action-features", type=Path, required=True)
    parser.add_argument("--existing-labels", type=Path, action="append", default=[])
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--head", default="short_asset")
    parser.add_argument("--action-value", type=float, default=0.0)
    parser.add_argument(
        "--rule",
        action="append",
        default=None,
        help=(
            "Monitoring rule to materialize. May be passed multiple times. "
            "Defaults to the cooldown rule plus the robust cooldown/open-share rule."
        ),
    )
    parser.add_argument("--max-targets", type=int, default=40)
    parser.add_argument("--max-per-day", type=int, default=6)
    args = parser.parse_args()
    manifest = run_check(
        scores=args.scores,
        action_features=args.action_features,
        existing_labels=list(args.existing_labels or []),
        out_dir=args.out_dir,
        head=args.head,
        action_value=args.action_value,
        rule=args.rule,
        max_targets=args.max_targets,
        max_per_day=args.max_per_day,
    )
    print((args.out_dir / "summary.md").read_text())
    print(json.dumps(_json_safe(manifest), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
