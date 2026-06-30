#!/usr/bin/env python3
"""Shadow-monitor C3el rule candidates on score/action-feature panels.

This script does not use outcomes and does not replay the portfolio.  It tags
candidate action-score rows with the predeclared C3el monitoring rules derived
from exact-state diagnostics, so future runs can collect labels under the same
contract and compare rule recurrence across periods.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


KEYS = ["timestamp", "strategy_id", "action_family", "action_value"]

RULE_COLUMNS = [
    "rule_strict_p80_d320",
    "rule_p80_d320_cooldown_lte_38_5",
    "rule_p80_d320_cooldown_lte_38_5_open_or_cooldown_share_lte_0_3949",
    "rule_p80_d320_timestamp_rank_q90_lte_0_8641",
    "rule_p80_d320_open_or_cooldown_share_lte_0_3949",
    "rule_p80_d320_strategy_rank_max_lte_0_9054",
    "rule_p80_d320_at_least_2_conditions",
    "rule_p80_d320_at_least_3_conditions",
    "rule_p80_d320_at_least_4_conditions",
    "rule_weak_p70_d100",
]


def _read_frame(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _normalise_score_keys(frame: pd.DataFrame, action_value: float) -> pd.DataFrame:
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
    out["p_intervene"] = pd.to_numeric(out["p_intervene"], errors="coerce")
    out["pred_action_delta_J"] = pd.to_numeric(out["pred_action_delta_J"], errors="coerce")
    return out


def _normalise_action_features(frame: pd.DataFrame, action_value: float) -> pd.DataFrame:
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


def _strategy_head(strategy_id: pd.Series) -> pd.Series:
    text = strategy_id.astype(str)
    out = pd.Series("unknown", index=text.index, dtype="object")
    for head in ("short_asset", "short_boll", "long_bars", "long_dist"):
        out.loc[text.str.startswith(head)] = head
    return out


def load_scored_features(scores_path: Path, action_features_path: Path, *, action_value: float, head: str) -> pd.DataFrame:
    scores = _normalise_score_keys(_read_frame(scores_path), action_value)
    features = _normalise_action_features(_read_frame(action_features_path), action_value)
    if "head" not in scores.columns:
        scores["head"] = _strategy_head(scores["strategy_id"])
    scores = scores.loc[
        scores["head"].astype(str).eq(str(head))
        & scores["action_family"].astype(str).eq("size")
        & scores["action_value"].eq(round(float(action_value), 6))
    ].copy()

    feature_cols = [
        c
        for c in features.columns
        if c not in set(KEYS + ["multiplier", "head"])
        and c not in scores.columns
        and pd.api.types.is_numeric_dtype(features[c])
    ]
    features = features.drop_duplicates(KEYS, keep="last")
    joined = scores.merge(features[KEYS + feature_cols], on=KEYS, how="left", validate="many_to_one")
    joined["feature_row_matched"] = joined[feature_cols].notna().any(axis=1) if feature_cols else False
    return joined


def tag_rules(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    p = pd.to_numeric(out["p_intervene"], errors="coerce")
    delta = pd.to_numeric(out["pred_action_delta_J"], errors="coerce")
    strict = p.ge(0.80) & delta.ge(320.0)
    weak = p.ge(0.70) & delta.ge(100.0)

    cooldown_ok = pd.to_numeric(out.get("cooldown_count"), errors="coerce").le(38.5)
    timestamp_rank_ok = pd.to_numeric(out.get("timestamp_rank_q90"), errors="coerce").le(0.8641)
    open_or_cooldown_ok = pd.to_numeric(
        out.get("strategy_candidate_open_or_cooldown_symbol_share"), errors="coerce"
    ).le(0.3949)
    strategy_rank_ok = pd.to_numeric(out.get("strategy_rank_max"), errors="coerce").le(0.9054)
    condition_count = (
        cooldown_ok.fillna(False).astype(int)
        + timestamp_rank_ok.fillna(False).astype(int)
        + open_or_cooldown_ok.fillna(False).astype(int)
        + strategy_rank_ok.fillna(False).astype(int)
    )

    out["rule_strict_p80_d320"] = strict.fillna(False)
    out["rule_p80_d320_cooldown_lte_38_5"] = strict.fillna(False) & cooldown_ok.fillna(False)
    out["rule_p80_d320_cooldown_lte_38_5_open_or_cooldown_share_lte_0_3949"] = (
        strict.fillna(False) & cooldown_ok.fillna(False) & open_or_cooldown_ok.fillna(False)
    )
    out["rule_p80_d320_timestamp_rank_q90_lte_0_8641"] = strict.fillna(False) & timestamp_rank_ok.fillna(False)
    out["rule_p80_d320_open_or_cooldown_share_lte_0_3949"] = strict.fillna(False) & open_or_cooldown_ok.fillna(False)
    out["rule_p80_d320_strategy_rank_max_lte_0_9054"] = strict.fillna(False) & strategy_rank_ok.fillna(False)
    out["rule_p80_d320_at_least_2_conditions"] = strict.fillna(False) & condition_count.ge(2)
    out["rule_p80_d320_at_least_3_conditions"] = strict.fillna(False) & condition_count.ge(3)
    out["rule_p80_d320_at_least_4_conditions"] = strict.fillna(False) & condition_count.ge(4)
    out["rule_weak_p70_d100"] = weak.fillna(False)
    out["monitor_condition_count"] = condition_count.astype(int)
    out["day"] = out["timestamp"].dt.floor("D")
    return out


def summarize_rules(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    day_rows = []
    total_rows = max(len(frame), 1)
    for rule in RULE_COLUMNS:
        mask = frame[rule].fillna(False)
        selected = frame.loc[mask].copy()
        rows.append(
            {
                "rule": rule,
                "rows": int(mask.sum()),
                "share_of_score_rows": float(mask.sum() / total_rows),
                "day_count": int(selected["day"].nunique()) if not selected.empty else 0,
                "first_timestamp": selected["timestamp"].min() if not selected.empty else pd.NaT,
                "last_timestamp": selected["timestamp"].max() if not selected.empty else pd.NaT,
                "p_mean": float(selected["p_intervene"].mean()) if not selected.empty else np.nan,
                "pred_delta_mean": float(selected["pred_action_delta_J"].mean()) if not selected.empty else np.nan,
                "feature_match_share": float(selected["feature_row_matched"].mean()) if not selected.empty else np.nan,
            }
        )
        if not selected.empty:
            by_day = selected.groupby("day", dropna=False).agg(
                rows=("timestamp", "size"),
                p_mean=("p_intervene", "mean"),
                pred_delta_mean=("pred_action_delta_J", "mean"),
                feature_match_share=("feature_row_matched", "mean"),
            )
            by_day = by_day.reset_index()
            by_day.insert(0, "rule", rule)
            day_rows.append(by_day)
    return pd.DataFrame(rows), pd.concat(day_rows, ignore_index=True, sort=False) if day_rows else pd.DataFrame()


def _json_safe(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return None if pd.isna(value) else value.isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    return value


def write_report(tagged: pd.DataFrame, summary: pd.DataFrame, by_day: pd.DataFrame, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    tagged.to_csv(out_dir / "tagged_score_rows.csv", index=False)
    summary.to_csv(out_dir / "rule_summary.csv", index=False)
    by_day.to_csv(out_dir / "rule_by_day.csv", index=False)

    display = summary[
        [
            "rule",
            "rows",
            "share_of_score_rows",
            "day_count",
            "first_timestamp",
            "last_timestamp",
            "p_mean",
            "pred_delta_mean",
            "feature_match_share",
        ]
    ]
    fired = display.loc[display["rows"].gt(0)].copy()
    lines = [
        "# C3el Rule Candidate Shadow Monitor",
        "",
        "This is outcome-free monitoring over C3el score rows and action-state features.",
        "",
        f"Score rows: `{len(tagged)}`",
        f"Feature rows matched: `{int(tagged['feature_row_matched'].sum())}`",
        f"Period: `{tagged['timestamp'].min()}` to `{tagged['timestamp'].max()}`",
        "",
        "## Rules With Firings",
        "",
        "No rule fired." if fired.empty else fired.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## All Rules",
        "",
        display.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Readout",
        "",
    ]
    strict_rows = int(summary.loc[summary["rule"].eq("rule_strict_p80_d320"), "rows"].sum())
    cooldown_rows = int(summary.loc[summary["rule"].eq("rule_p80_d320_cooldown_lte_38_5"), "rows"].sum())
    robust_rows = int(
        summary.loc[
            summary["rule"].eq("rule_p80_d320_cooldown_lte_38_5_open_or_cooldown_share_lte_0_3949"),
            "rows",
        ].sum()
    )
    if strict_rows == 0:
        lines.append("No strict p80/d320 C3el state fired in this panel.")
    else:
        lines.append(
            f"Strict p80/d320 fired `{strict_rows}` times; cooldown-filtered monitoring state fired `{cooldown_rows}` times; "
            f"robust cooldown+open/cooldown-share state fired `{robust_rows}` times."
        )
    lines.append("These counts are for label-collection/shadow monitoring only; they do not imply realized utility.")

    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    manifest = {
        "generated_by": "monitor_c3el_rule_candidates",
        "rows": int(len(tagged)),
        "feature_rows_matched": int(tagged["feature_row_matched"].sum()),
        "start": tagged["timestamp"].min(),
        "end": tagged["timestamp"].max(),
        "outputs": {
            "tagged": str(out_dir / "tagged_score_rows.csv"),
            "summary": str(out_dir / "rule_summary.csv"),
            "by_day": str(out_dir / "rule_by_day.csv"),
            "report": str(out_dir / "summary.md"),
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(_json_safe(manifest), indent=2, sort_keys=True))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--action-features", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--head", default="short_asset")
    parser.add_argument("--action-value", type=float, default=0.0)
    args = parser.parse_args()

    frame = load_scored_features(args.scores, args.action_features, action_value=args.action_value, head=args.head)
    tagged = tag_rules(frame)
    summary, by_day = summarize_rules(tagged)
    manifest = write_report(tagged, summary, by_day, args.out_dir)
    print((args.out_dir / "summary.md").read_text())
    print(json.dumps(_json_safe(manifest), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
