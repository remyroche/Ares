#!/usr/bin/env python3
"""Build a capped exact-state oracle target queue for C3el fallback states.

This is deliberately a target materializer, not a replay runner.  It selects
new high-priority short-asset fallback-like size-cut actions from deployable
C3el action scores, excludes actions that already have exact cloned-state
labels, and caps the queue by day so the next oracle run is memory-bounded.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


KEYS = ["timestamp", "strategy_id", "action_family", "action_value"]


def _read_frame(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _normalise_timestamp(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out = out.loc[out["timestamp"].notna()].copy()
    return out


def _normalise_action_key(frame: pd.DataFrame, *, default_action_value: float) -> pd.DataFrame:
    out = _normalise_timestamp(frame)
    out["strategy_id"] = out["strategy_id"].astype(str)
    if "action_family" not in out.columns:
        out["action_family"] = "size"
    out["action_family"] = out["action_family"].astype(str).fillna("size")
    if "action_value" in out.columns:
        values = pd.to_numeric(out["action_value"], errors="coerce")
    elif "multiplier" in out.columns:
        values = pd.to_numeric(out["multiplier"], errors="coerce")
    else:
        values = pd.Series(np.nan, index=out.index)
    out["action_value"] = values.fillna(float(default_action_value)).round(6)
    return out


def _strategy_head(strategy_id: pd.Series) -> pd.Series:
    text = strategy_id.astype(str)
    out = pd.Series("unknown", index=text.index, dtype="object")
    for head in ("short_asset", "short_boll", "long_bars", "long_dist"):
        out.loc[text.str.startswith(head)] = head
    return out


def _load_scores(path: Path, *, action_value: float) -> pd.DataFrame:
    required = {"timestamp", "strategy_id", "p_intervene", "pred_action_delta_J"}
    frame = _normalise_action_key(_read_frame(path), default_action_value=action_value)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"score file is missing required columns: {missing}")
    if "head" not in frame.columns:
        frame["head"] = _strategy_head(frame["strategy_id"])
    frame["p_intervene"] = pd.to_numeric(frame["p_intervene"], errors="coerce")
    frame["pred_action_delta_J"] = pd.to_numeric(frame["pred_action_delta_J"], errors="coerce")
    if "selected_multiplier" in frame.columns:
        frame["selected_multiplier"] = pd.to_numeric(frame["selected_multiplier"], errors="coerce")
    else:
        frame["selected_multiplier"] = np.nan
    return frame


def _load_action_features(path: Path, *, action_value: float) -> pd.DataFrame:
    frame = _normalise_action_key(_read_frame(path), default_action_value=action_value)
    if "head" not in frame.columns:
        frame["head_feature"] = _strategy_head(frame["strategy_id"])
    else:
        frame["head_feature"] = frame["head"].astype(str)
    return frame


def _load_existing(path: Path | None, *, action_value: float) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame(columns=KEYS)
    frame = _normalise_action_key(_read_frame(path), default_action_value=action_value)
    return frame[KEYS].drop_duplicates().reset_index(drop=True)


def _candidate_frame(
    *,
    scores: pd.DataFrame,
    action_features: pd.DataFrame,
    existing: pd.DataFrame,
    head: str,
    action_value: float,
    min_p_intervene: float,
    min_pred_delta_j: float,
    max_selected_multiplier: float | None,
    quality_rule: str | None,
) -> pd.DataFrame:
    work = scores.loc[
        scores["head"].astype(str).eq(str(head))
        & scores["action_family"].astype(str).eq("size")
        & scores["action_value"].eq(round(float(action_value), 6))
        & scores["p_intervene"].ge(float(min_p_intervene))
        & scores["pred_action_delta_J"].ge(float(min_pred_delta_j))
    ].copy()
    if max_selected_multiplier is not None:
        work = work.loc[work["selected_multiplier"].le(float(max_selected_multiplier))].copy()
    work = work.drop_duplicates(KEYS, keep="last")
    if existing.empty:
        work["already_labeled"] = False
    else:
        work = work.merge(existing.assign(already_labeled=True), on=KEYS, how="left")
        work["already_labeled"] = work["already_labeled"].eq(True)
        work = work.loc[~work["already_labeled"]].copy()
    feature_cols = [
        c
        for c in action_features.columns
        if c not in set(KEYS + ["head", "head_feature"])
        and c not in work.columns
        and pd.api.types.is_numeric_dtype(action_features[c])
    ]
    joined = work.merge(action_features[KEYS + ["head_feature"] + feature_cols], on=KEYS, how="left")
    joined["feature_row_matched"] = joined["head_feature"].notna()
    if quality_rule:
        joined = _apply_quality_rule(joined, quality_rule).copy()
    joined["day"] = pd.to_datetime(joined["timestamp"], utc=True).dt.floor("D")
    return joined


def _apply_quality_rule(frame: pd.DataFrame, quality_rule: str) -> pd.DataFrame:
    feature, op, raw_threshold = _parse_quality_rule(quality_rule)
    vals = pd.to_numeric(frame.get(feature), errors="coerce")
    threshold = float(raw_threshold)
    if op == ">=":
        mask = vals.ge(threshold)
    elif op == "<=":
        mask = vals.le(threshold)
    elif op == ">":
        mask = vals.gt(threshold)
    elif op == "<":
        mask = vals.lt(threshold)
    else:
        raise ValueError(f"Unsupported quality rule operator: {op}")
    out = frame.loc[mask.fillna(False)].copy()
    out["quality_rule"] = quality_rule
    return out


def _parse_quality_rule(rule: str) -> tuple[str, str, str]:
    for op in (">=", "<=", ">", "<"):
        if op in rule:
            left, right = rule.split(op, 1)
            return left.strip(), op, right.strip()
    raise ValueError(f"Quality rule must contain one of >=, <=, >, <: {rule}")


def _score_candidates(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    p = pd.to_numeric(out["p_intervene"], errors="coerce").fillna(0.0)
    delta = pd.to_numeric(out["pred_action_delta_J"], errors="coerce").fillna(0.0)
    rank = delta.rank(method="average", pct=True).fillna(0.0)
    out["target_priority"] = (1000.0 * p + delta + 25.0 * rank).astype(float)
    return out.sort_values(["target_priority", "p_intervene", "pred_action_delta_J"], ascending=[False, False, False])


def _cap_targets(frame: pd.DataFrame, *, max_targets: int, max_per_day: int) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    rows = []
    for _, group in frame.groupby("day", sort=True):
        rows.append(group.head(int(max_per_day)))
    capped = pd.concat(rows, ignore_index=True) if rows else frame.head(0).copy()
    return capped.sort_values(["target_priority", "timestamp"], ascending=[False, True]).head(int(max_targets)).reset_index(drop=True)


def _summary_by_day(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["day", "target_count", "p_intervene_mean", "pred_action_delta_J_mean"])
    return (
        frame.groupby("day", dropna=False)
        .agg(
            target_count=("timestamp", "size"),
            p_intervene_mean=("p_intervene", "mean"),
            pred_action_delta_J_mean=("pred_action_delta_J", "mean"),
            target_priority_mean=("target_priority", "mean"),
        )
        .reset_index()
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def build_targets(
    *,
    scores_path: Path,
    action_features_path: Path,
    existing_labels_path: Path | None,
    out_dir: Path,
    head: str,
    action_value: float,
    min_p_intervene: float,
    min_pred_delta_j: float,
    max_selected_multiplier: float | None,
    quality_rule: str | None,
    max_targets: int,
    max_per_day: int,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    scores = _load_scores(scores_path, action_value=action_value)
    action_features = _load_action_features(action_features_path, action_value=action_value)
    existing = _load_existing(existing_labels_path, action_value=action_value)
    candidates = _candidate_frame(
        scores=scores,
        action_features=action_features,
        existing=existing,
        head=head,
        action_value=action_value,
        min_p_intervene=min_p_intervene,
        min_pred_delta_j=min_pred_delta_j,
        max_selected_multiplier=max_selected_multiplier,
        quality_rule=quality_rule,
    )
    ranked = _score_candidates(candidates)
    capped = _cap_targets(ranked, max_targets=max_targets, max_per_day=max_per_day)
    target_cols = [
        "timestamp",
        "strategy_id",
        "head",
        "action_family",
        "action_value",
        "p_intervene",
        "pred_action_delta_J",
        "selected_multiplier",
        "target_priority",
        "feature_row_matched",
    ]
    if "quality_rule" in capped.columns:
        target_cols.append("quality_rule")
    target_actions = capped[target_cols].copy()
    target_actions.to_csv(out_dir / "target_actions.csv", index=False)
    ranked.to_csv(out_dir / "candidate_pool.csv", index=False)
    by_day = _summary_by_day(capped)
    by_day.to_csv(out_dir / "target_summary_by_day.csv", index=False)
    manifest = {
        "generated_by": "build_c3el_fallback_oracle_targets",
        "scores_path": str(scores_path),
        "action_features_path": str(action_features_path),
        "existing_labels_path": str(existing_labels_path) if existing_labels_path else None,
        "head": str(head),
        "action_value": float(action_value),
        "min_p_intervene": float(min_p_intervene),
        "min_pred_delta_j": float(min_pred_delta_j),
        "max_selected_multiplier": None if max_selected_multiplier is None else float(max_selected_multiplier),
        "quality_rule": quality_rule,
        "max_targets": int(max_targets),
        "max_per_day": int(max_per_day),
        "score_rows": int(len(scores)),
        "existing_labeled_actions": int(len(existing)),
        "candidate_pool_rows": int(len(ranked)),
        "target_rows": int(len(target_actions)),
        "feature_rows_matched": int(capped["feature_row_matched"].sum()) if not capped.empty else 0,
        "target_min_timestamp": None if capped.empty else capped["timestamp"].min(),
        "target_max_timestamp": None if capped.empty else capped["timestamp"].max(),
    }
    (out_dir / "manifest.json").write_text(json.dumps(_json_safe(manifest), indent=2, sort_keys=True))
    lines = [
        "# C3el fallback exact-state oracle target queue",
        "",
        "This file is only a capped queue for a future exact-state oracle run. It does not run portfolio replay.",
        "",
        f"Head: `{head}`",
        f"Rows in candidate pool after filters and duplicate-label exclusion: `{len(ranked)}`",
        f"Rows selected for oracle: `{len(target_actions)}`",
        f"Per-day cap: `{max_per_day}`",
        f"Global cap: `{max_targets}`",
        f"Feature rows matched: `{manifest['feature_rows_matched']}`",
        "",
        "## By Day",
        "",
    ]
    lines.append("No selected targets." if by_day.empty else by_day.to_markdown(index=False, floatfmt=".4f"))
    lines.extend(["", "## Top Targets", ""])
    if target_actions.empty:
        lines.append("No selected targets.")
    else:
        cols = ["timestamp", "strategy_id", "p_intervene", "pred_action_delta_J", "selected_multiplier", "target_priority"]
        lines.append(target_actions[cols].head(20).to_markdown(index=False, floatfmt=".4f"))
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--action-features", type=Path, required=True)
    parser.add_argument("--existing-labels", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--head", default="short_asset")
    parser.add_argument("--action-value", type=float, default=0.0)
    parser.add_argument("--min-p-intervene", type=float, default=0.80)
    parser.add_argument("--min-pred-delta-j", type=float, default=320.0)
    parser.add_argument("--max-selected-multiplier", type=float, default=1.0)
    parser.add_argument("--quality-rule", default=None)
    parser.add_argument("--max-targets", type=int, default=40)
    parser.add_argument("--max-per-day", type=int, default=6)
    args = parser.parse_args()
    manifest = build_targets(
        scores_path=args.scores,
        action_features_path=args.action_features,
        existing_labels_path=args.existing_labels,
        out_dir=args.out_dir,
        head=args.head,
        action_value=args.action_value,
        min_p_intervene=args.min_p_intervene,
        min_pred_delta_j=args.min_pred_delta_j,
        max_selected_multiplier=args.max_selected_multiplier,
        quality_rule=args.quality_rule,
        max_targets=args.max_targets,
        max_per_day=args.max_per_day,
    )
    print((args.out_dir / "summary.md").read_text())
    print(json.dumps(_json_safe(manifest), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
