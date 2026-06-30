#!/usr/bin/env python3
"""Consolidate current C3el exact-state and shadow-monitor readiness.

This report is intentionally narrow: it summarizes whether the current
short_asset C3el rule evidence supports deployment, shadow monitoring, or a new
exact-state replay.  It combines the latest exact-state score-boundary audit,
rule-candidate audit, shadow-monitor counts, and rule target queues.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_BUCKET_SUMMARY = Path("data_perp/reports/c3el_exact_score_boundary_audit_20260628/summary_by_bucket.csv")
DEFAULT_RULE_SUMMARY = Path("data_perp/reports/c3el_exact_rule_candidate_audit_20260628/rule_candidate_summary.csv")
DEFAULT_MONITOR_DIRS = {
    "may06_may29": Path("data_perp/reports/c3el_rule_candidate_shadow_monitor_20260628_may06_may29"),
    "last4w": Path(
        "data_perp/reports/c3el_rule_candidate_shadow_monitor_20260628_last4w_shortasset_robust"
    ),
    "postjun26": Path("data_perp/reports/c3el_rule_candidate_shadow_monitor_20260628_postjun26"),
}
DEFAULT_TARGET_MANIFESTS = {
    "last4w_cooldown_unlabeled": Path(
        "data_perp/reports/c3el_rule_oracle_targets_20260628_last4w_cooldown_unlabeled/manifest.json"
    ),
    "last4w_robust_cooldown_open039_unlabeled": Path(
        "data_perp/reports/c3el_rule_oracle_targets_20260628_last4w_robust_cooldown_open039_unlabeled/manifest.json"
    ),
    "postjun26_cooldown_unlabeled": Path(
        "data_perp/reports/c3el_forward_readiness_check_20260628_postjun26_two_rule_unique/targets/manifest.json"
    ),
    "postjun26_robust_cooldown_open039_unlabeled": Path(
        "data_perp/reports/c3el_forward_readiness_check_20260628_postjun26_two_rule_unique/"
        "targets_p80_d320_cooldown_lte_38_5_open_or_cooldown_share_lte_0_3949/manifest.json"
    ),
}
DEFAULT_OUT_DIR = Path("data_perp/reports/c3el_current_readiness_audit_20260628")

STRICT_BUCKET = "p80_d320"
BROAD_BAD_BUCKET = "p80_d250_320"
PREFERRED_RULE = "strict__cooldown_count_lte_38_5"
PREFERRED_MONITOR_RULE = "rule_p80_d320_cooldown_lte_38_5"
ROBUST_RULE = "strict__cooldown_count_lte_38_5__open_or_cooldown_share_lte_0_3949"
ROBUST_MONITOR_RULE = "rule_p80_d320_cooldown_lte_38_5_open_or_cooldown_share_lte_0_3949"
TERTIARY_RULE = "strict__at_least_4_conditions"
ACTION_KEYS = ["timestamp", "strategy_id", "action_family", "action_value"]


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def _row(frame: pd.DataFrame, column: str, value: str) -> pd.Series:
    rows = frame.loc[frame[column].astype(str).eq(str(value))]
    if rows.empty:
        raise ValueError(f"missing {column}={value}")
    return rows.iloc[0]


def _gate(name: str, status: str, evidence: str, action: str) -> dict[str, str]:
    return {"gate": name, "status": status, "evidence": evidence, "action": action}


def _as_bool(value: Any, *, default: bool = False) -> bool:
    if value is None or pd.isna(value):
        return default
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "t", "yes", "y", "1"}:
        return True
    if text in {"false", "f", "no", "n", "0", ""}:
        return False
    return default


def _normalise_action_keys(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame[ACTION_KEYS].copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out = out.loc[out["timestamp"].notna()].copy()
    out["strategy_id"] = out["strategy_id"].astype(str)
    out["action_family"] = out["action_family"].astype(str)
    out["action_value"] = pd.to_numeric(out["action_value"], errors="coerce").round(6)
    return out.dropna(subset=["action_value"])


def _unique_action_count(paths: list[Path]) -> int:
    frames = []
    for path in paths:
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        if frame.empty or any(col not in frame.columns for col in ACTION_KEYS):
            continue
        frames.append(_normalise_action_keys(frame))
    if not frames:
        return 0
    return int(pd.concat(frames, ignore_index=True).drop_duplicates(ACTION_KEYS).shape[0])


def _unique_backlog_count(frame: pd.DataFrame, *, path_column: str, fallback_column: str) -> int:
    if frame.empty:
        return 0
    if path_column in frame.columns:
        return _unique_action_count([Path(p) for p in frame[path_column]])
    return int(frame[fallback_column].sum()) if fallback_column in frame.columns else 0


def _monitor_counts(monitor_dirs: dict[str, Path]) -> pd.DataFrame:
    rows = []
    for panel, directory in monitor_dirs.items():
        path = directory / "rule_summary.csv"
        summary = _read_csv(path)
        for _, row in summary.iterrows():
            rows.append(
                {
                    "panel": panel,
                    "rule": str(row["rule"]),
                    "rows": int(row["rows"]),
                    "day_count": int(row["day_count"]),
                    "first_timestamp": row.get("first_timestamp"),
                    "last_timestamp": row.get("last_timestamp"),
                }
            )
    return pd.DataFrame(rows)


def _target_backlog(manifests: dict[str, Path]) -> pd.DataFrame:
    rows = []
    for name, path in manifests.items():
        payload = _read_json(path)
        target_actions_path = path.parent / "target_actions.csv"
        candidate_pool_path = path.parent / "candidate_pool.csv"
        rows.append(
            {
                "queue": name,
                "rule": payload.get("rule"),
                "tagged_rows": int(payload.get("tagged_rows", 0)),
                "rule_rows_before_existing_exclusion": int(
                    payload.get("rule_candidate_rows_before_existing_exclusion", 0)
                ),
                "existing_labeled_actions": int(payload.get("existing_labeled_actions", 0)),
                "candidate_pool_rows": int(payload.get("candidate_pool_rows", 0)),
                "target_rows": int(payload.get("target_rows", 0)),
                "candidate_pool_path": str(candidate_pool_path),
                "target_actions_path": str(target_actions_path),
            }
        )
    return pd.DataFrame(rows)


def build_readiness(
    *,
    bucket_summary: pd.DataFrame,
    rule_summary: pd.DataFrame,
    monitor_counts: pd.DataFrame,
    target_backlog: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    strict = _row(bucket_summary, "bucket", STRICT_BUCKET)
    broad_bad = _row(bucket_summary, "bucket", BROAD_BAD_BUCKET)
    preferred = _row(rule_summary, "rule", PREFERRED_RULE)
    robust = _row(rule_summary, "rule", ROBUST_RULE)
    tertiary = _row(rule_summary, "rule", TERTIARY_RULE)
    strict_rule = _row(rule_summary, "rule", "strict_p80_d320")

    strict_sum = float(strict["sum_delta_full_J"])
    strict_pos = float(strict["pos_share"])
    strict_rows = int(strict["rows"])
    strict_days = int(strict_rule["day_count"])
    broad_sum = float(broad_bad["sum_delta_full_J"])
    preferred_sum = float(preferred["sum_delta_full_J"])
    preferred_pos = float(preferred["positive_share"])
    preferred_day_pos = float(preferred["positive_day_share"])
    robust_sum = float(robust["sum_delta_full_J"])
    robust_pos = float(robust["positive_share"])
    robust_day_pos = float(robust["positive_day_share"])
    tertiary_sum = float(tertiary["sum_delta_full_J"])
    tertiary_pos = float(tertiary["positive_share"])
    tertiary_worst = float(tertiary["worst_delta_full_J"])
    tertiary_passes_min_rows = _as_bool(tertiary.get("passes_min_rows", False))

    postjun = monitor_counts.loc[
        monitor_counts["panel"].eq("postjun26") & monitor_counts["rule"].eq(PREFERRED_MONITOR_RULE)
    ]
    postjun_firings = int(postjun["rows"].sum()) if not postjun.empty else 0
    last4w = monitor_counts.loc[
        monitor_counts["panel"].eq("last4w") & monitor_counts["rule"].eq(PREFERRED_MONITOR_RULE)
    ]
    last4w_firings = int(last4w["rows"].sum()) if not last4w.empty else 0
    robust_last4w = monitor_counts.loc[monitor_counts["panel"].eq("last4w") & monitor_counts["rule"].eq(ROBUST_MONITOR_RULE)]
    if robust_last4w.empty:
        # Backward-compatible fallback for older reports that used a separate
        # last4w_robust panel name for the same time window.
        robust_last4w = monitor_counts.loc[
            monitor_counts["panel"].eq("last4w_robust") & monitor_counts["rule"].eq(ROBUST_MONITOR_RULE)
        ]
    robust_last4w_firings = int(robust_last4w["rows"].sum()) if not robust_last4w.empty else 0
    backlog_targets = _unique_backlog_count(
        target_backlog,
        path_column="target_actions_path",
        fallback_column="target_rows",
    )
    backlog_pool = _unique_backlog_count(
        target_backlog,
        path_column="candidate_pool_path",
        fallback_column="candidate_pool_rows",
    )
    if "rule" in target_backlog.columns:
        robust_backlog = target_backlog.loc[target_backlog["rule"].astype(str).eq(ROBUST_MONITOR_RULE)]
    else:
        robust_backlog = target_backlog.head(0).copy()
    robust_backlog_targets = _unique_backlog_count(
        robust_backlog,
        path_column="target_actions_path",
        fallback_column="target_rows",
    )

    gates = [
        _gate(
            "strict_exact_state_evidence",
            "pass" if strict_rows >= 20 and strict_sum > 0.0 and strict_pos >= 0.60 else "fail",
            f"{STRICT_BUCKET}: rows={strict_rows}, pos_share={strict_pos:.2%}, sum_delta_full_J={strict_sum:.2f}",
            "keep as high-conviction research state",
        ),
        _gate(
            "broadening_rejected",
            "pass" if broad_sum < 0.0 else "fail",
            f"{BROAD_BAD_BUCKET}: sum_delta_full_J={broad_sum:.2f}",
            "do not broaden to p80/d250",
        ),
        _gate(
            "preferred_rule_lift",
            "pass" if preferred_sum > strict_sum and preferred_pos > strict_pos and preferred_day_pos > 0.70 else "fail",
            (
                f"{PREFERRED_RULE}: rows={int(preferred['rows'])}, pos_share={preferred_pos:.2%}, "
                f"positive_day_share={preferred_day_pos:.2%}, sum_delta_full_J={preferred_sum:.2f}"
            ),
            "shadow-monitor cooldown-filtered strict rule",
        ),
        _gate(
            "robust_rule_precision",
            "pass" if robust_pos >= preferred_pos and robust_day_pos >= preferred_day_pos else "watch",
            (
                f"{ROBUST_RULE}: rows={int(robust['rows'])}, pos_share={robust_pos:.2%}, "
                f"positive_day_share={robust_day_pos:.2%}, sum_delta_full_J={robust_sum:.2f}; "
                f"last4w robust monitor firings={robust_last4w_firings}, unlabeled targets={robust_backlog_targets}"
            ),
            "shadow-monitor as the robust challenger; require forward recurrence before promotion",
        ),
        _gate(
            "ultra_conservative_subset",
            "watch" if tertiary_pos >= robust_pos and tertiary_worst > 0.0 and not tertiary_passes_min_rows else "pass",
            (
                f"{TERTIARY_RULE}: rows={int(tertiary['rows'])}, pos_share={tertiary_pos:.2%}, "
                f"worst_delta_full_J={tertiary_worst:.2f}, sum_delta_full_J={tertiary_sum:.2f}, "
                f"passes_min_rows={tertiary_passes_min_rows}"
            ),
            "track as a stricter subset already covered by the robust rule; do not queue separately yet",
        ),
        _gate(
            "forward_recurrence",
            "waiting" if postjun_firings == 0 else "ready",
            (
                f"post-Jun-26 {PREFERRED_MONITOR_RULE} firings={postjun_firings}; "
                f"last4w cooldown firings={last4w_firings}; last4w robust firings={robust_last4w_firings}"
            ),
            "wait for new forward firings before replay",
        ),
        _gate(
            "unlabeled_target_backlog",
            "waiting" if backlog_targets == 0 else "ready",
            f"unlabeled candidate_pool_rows={backlog_pool}, selected target_rows={backlog_targets}",
            "run exact-state replay only when target_rows > 0",
        ),
        _gate(
            "production_promotion",
            "fail",
            "same labels were used to discover rule slices; forward recurrence is missing",
            "do not deploy as hard gate",
        ),
    ]
    payload = {
        "decision": "monitor_only_wait_for_forward_recurrence",
        "strict_rows": strict_rows,
        "strict_days": strict_days,
        "strict_sum_delta_full_J": strict_sum,
        "preferred_sum_delta_full_J": preferred_sum,
        "robust_sum_delta_full_J": robust_sum,
        "tertiary_sum_delta_full_J": tertiary_sum,
        "postjun_preferred_firings": postjun_firings,
        "last4w_robust_firings": robust_last4w_firings,
        "unlabeled_target_rows": backlog_targets,
        "robust_unlabeled_target_rows": robust_backlog_targets,
    }
    return pd.DataFrame(gates), payload


def _json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def write_report(
    *,
    gates: pd.DataFrame,
    payload: dict[str, Any],
    bucket_summary: pd.DataFrame,
    rule_summary: pd.DataFrame,
    monitor_counts: pd.DataFrame,
    target_backlog: pd.DataFrame,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    gates.to_csv(out_dir / "readiness_gates.csv", index=False)
    bucket_summary.to_csv(out_dir / "score_bucket_summary.csv", index=False)
    rule_summary.to_csv(out_dir / "rule_candidate_summary.csv", index=False)
    monitor_counts.to_csv(out_dir / "monitor_rule_counts.csv", index=False)
    target_backlog.to_csv(out_dir / "target_backlog.csv", index=False)

    selected_rules = rule_summary.loc[
        rule_summary["rule"]
        .astype(str)
        .isin(["strict_p80_d320", PREFERRED_RULE, ROBUST_RULE, TERTIARY_RULE, "strict__at_least_3_conditions"])
    ].copy()
    monitor_pivot = monitor_counts.pivot_table(index="rule", columns="panel", values="rows", aggfunc="sum", fill_value=0)
    monitor_pivot = monitor_pivot.reset_index()
    lines = [
        "# C3el Current Readiness Audit",
        "",
        f"Decision: `{payload['decision']}`",
        "",
        "## Gates",
        "",
        gates.to_markdown(index=False),
        "",
        "## Exact-State Bucket Evidence",
        "",
        bucket_summary.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Key Rule Candidates",
        "",
        selected_rules[
            [
                "rule",
                "rows",
                "coverage_of_strict",
                "positive_share",
                "positive_day_share",
                "sum_delta_full_J",
                "median_delta_full_J",
                "worst_delta_full_J",
            ]
        ].to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Shadow-Monitor Firings",
        "",
        monitor_pivot.to_markdown(index=False),
        "",
        "## Target Backlog",
        "",
        target_backlog[
            [
                c
                for c in target_backlog.columns
                if c not in {"candidate_pool_path", "target_actions_path"}
            ]
        ].to_markdown(index=False),
        "",
        "## Recommendation",
        "",
        "Keep `p80_d320` as a high-conviction research state. Keep two short_asset "
        "monitoring rules: the higher-PnL cooldown rule "
        "`p80_d320 AND cooldown_count <= 38.5`, and the robust challenger "
        "`p80_d320 AND cooldown_count <= 38.5 AND open_or_cooldown_share <= 0.3949`. "
        "Also watch the all-four-condition subset as a precision diagnostic, but do not "
        "queue it separately because it is already captured by the robust rule and has "
        "only nine exact-state labels. "
        "Do not broaden to p80/d250 and do not deploy a hard gate. The next replay "
        "should only run after new forward rows fire one of the preferred rules and "
        "the materializer produces nonzero targets.",
        "",
        "## Next Ablation Criteria",
        "",
        "| Situation | Action | Minimum evidence |",
        "|:--|:--|:--|",
        "| `target_rows == 0` | Do not run exact-state replay | Continue shadow monitoring only |",
        "| `0 < target_rows < 10` | Materialize labels only | Treat as anecdotal recurrence; no policy change |",
        "| `target_rows >= 10` across at least 3 days | Run exact-state replay on the generated queue | Compare baseline, cooldown rule, robust rule, and all-four-condition subset diagnostics |",
        "| `target_rows >= 30` across at least 5 days | Consider a sparse forward ablation | Require positive median day, non-negative worst-day or explainable worst-day, and no degradation in costs/full-SL behavior |",
        "",
        "Promotion remains blocked until all of the following are true:",
        "",
        "- forward rows, not rediscovered in-sample rows, produce the labels;",
        "- unique target actions are nonzero after overlap de-duplication;",
        "- robust/cooldown rules beat the raw `p80_d320` bucket on full-path utility and downside behavior;",
        "- the all-four-condition subset remains a precision diagnostic unless it reaches minimum support;",
        "- broadening toward `p80_d250_320` remains rejected unless new forward evidence overturns its current negative full-path result.",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    (out_dir / "manifest.json").write_text(
        json.dumps(
            _json_safe(
                {
                    "generated_by": "report_c3el_current_readiness",
                    **payload,
                    "outputs": {
                        "summary": str(out_dir / "summary.md"),
                        "readiness_gates": str(out_dir / "readiness_gates.csv"),
                        "target_backlog": str(out_dir / "target_backlog.csv"),
                    },
                }
            ),
            indent=2,
            sort_keys=True,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket-summary", type=Path, default=DEFAULT_BUCKET_SUMMARY)
    parser.add_argument("--rule-summary", type=Path, default=DEFAULT_RULE_SUMMARY)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    bucket_summary = _read_csv(args.bucket_summary)
    rule_summary = _read_csv(args.rule_summary)
    monitor_counts = _monitor_counts(DEFAULT_MONITOR_DIRS)
    target_backlog = _target_backlog(DEFAULT_TARGET_MANIFESTS)
    gates, payload = build_readiness(
        bucket_summary=bucket_summary,
        rule_summary=rule_summary,
        monitor_counts=monitor_counts,
        target_backlog=target_backlog,
    )
    write_report(
        gates=gates,
        payload=payload,
        bucket_summary=bucket_summary,
        rule_summary=rule_summary,
        monitor_counts=monitor_counts,
        target_backlog=target_backlog,
        out_dir=args.out_dir,
    )
    print((args.out_dir / "summary.md").read_text())


if __name__ == "__main__":
    main()
