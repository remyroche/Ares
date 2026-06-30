#!/usr/bin/env python3
"""Find observable slices where head-native C3el actions have positive value.

This is an artifact-only diagnostic.  It reads the exact-state action panel,
filters one strategy head, and searches simple one-feature quantile slices over
pre-action portfolio/opportunity features.  It does not train or replay.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


HEADS = ("long_bars", "long_dist", "short_asset", "short_boll")
KEY_COLS = {"timestamp", "strategy_id", "multiplier", "fold_id", "split"}
TARGET_COLS = {
    "action_binds",
    "group_can_bind",
    "delta_full_J",
    "delta_immediate_J",
    "delta_full_net_pnl",
    "delta_full_cost_pnl",
    "delta_full_turnover",
    "delta_full_J_per_notional",
    "delta_immediate_J_per_notional",
    "best_multiplier",
    "best_gain",
    "best_margin",
    "best_gain_per_notional",
    "best_margin_per_notional",
    "best_immediate_gain",
    "best_nonbaseline_gain",
    "worst_nonbaseline_gain",
    "best_nonbaseline_multiplier",
    "y_intervene",
}


def _head_from_strategy(strategy_id: str) -> str:
    text = str(strategy_id)
    for head in HEADS:
        if text.startswith(head):
            return head
    return "unknown"


def _read_header(path: Path) -> list[str]:
    if path.suffix.lower() in {".parquet", ".pq"}:
        try:
            import pyarrow.parquet as pq

            return list(pq.ParquetFile(path).schema.names)
        except Exception:
            return list(pd.read_parquet(path).columns)
    return list(pd.read_csv(path, nrows=0).columns)


def _read_panel(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path, columns=columns)
    return pd.read_csv(path, usecols=columns)


def _candidate_columns(columns: list[str], *, extra_exclude: set[str] | None = None) -> list[str]:
    exclude = set(KEY_COLS) | set(TARGET_COLS) | {"head", "day", "week_start"}
    if extra_exclude:
        exclude.update(extra_exclude)
    return [col for col in columns if col not in exclude]


def load_actions(path: Path, *, head: str) -> pd.DataFrame:
    columns = _read_header(path)
    required = [
        "timestamp",
        "strategy_id",
        "multiplier",
        "action_binds",
        "delta_full_J",
        "delta_immediate_J",
        "delta_full_net_pnl",
        "delta_full_cost_pnl",
        "delta_full_turnover",
        "affected_notional",
    ]
    required = [col for col in required if col in columns]
    features = _candidate_columns(columns)
    frame = _read_panel(path, columns=sorted(set(required + features)))
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame = frame.loc[frame["timestamp"].notna()].copy()
    frame["strategy_id"] = frame["strategy_id"].astype(str)
    frame["head"] = frame["strategy_id"].map(_head_from_strategy)
    frame = frame.loc[frame["head"].eq(head)].copy()
    frame["multiplier"] = pd.to_numeric(frame["multiplier"], errors="coerce")
    if "action_binds" in frame.columns:
        frame["action_binds"] = pd.to_numeric(frame["action_binds"], errors="coerce").fillna(0.0)
        frame = frame.loc[frame["action_binds"].gt(0.0)].copy()
    frame = frame.loc[frame["multiplier"].lt(1.0)].copy()
    for col in TARGET_COLS | {"affected_notional"}:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame["week_start"] = frame["timestamp"].dt.normalize() - pd.to_timedelta(frame["timestamp"].dt.weekday, unit="D")
    return frame.reset_index(drop=True)


def _feature_candidates(frame: pd.DataFrame) -> list[str]:
    candidates = []
    for col in frame.columns:
        if col in KEY_COLS or col in TARGET_COLS or col in {"head", "week_start"}:
            continue
        if col.endswith("_to_open_notional"):
            continue
        if not pd.api.types.is_numeric_dtype(frame[col]):
            continue
        vals = pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if vals.notna().sum() < 20 or vals.nunique(dropna=True) < 3:
            continue
        candidates.append(col)
    return candidates


def _summary(frame: pd.DataFrame, *, epsilon: float) -> dict[str, Any]:
    if frame.empty:
        return {
            "rows": 0,
            "groups": 0,
            "positive_share": np.nan,
            "positive_epsilon_share": np.nan,
            "sum_delta_full_J": 0.0,
            "mean_delta_full_J": np.nan,
            "median_delta_full_J": np.nan,
            "worst_delta_full_J": np.nan,
            "gain_sum_positive": 0.0,
            "loss_sum_negative_abs": 0.0,
            "gain_to_loss_abs_ratio": np.nan,
            "positive_week_share": np.nan,
            "worst_week_delta_full_J": np.nan,
        }
    delta = pd.to_numeric(frame["delta_full_J"], errors="coerce").fillna(0.0)
    week_delta = frame.assign(_delta=delta).groupby("week_start", dropna=False)["_delta"].sum()
    gain = float(delta.clip(lower=0.0).sum())
    loss = float((-delta.clip(upper=0.0)).sum())
    return {
        "rows": int(len(frame)),
        "groups": int(frame[["timestamp", "strategy_id"]].drop_duplicates().shape[0]),
        "positive_share": float(delta.gt(0.0).mean()),
        "positive_epsilon_share": float(delta.gt(float(epsilon)).mean()),
        "sum_delta_full_J": float(delta.sum()),
        "mean_delta_full_J": float(delta.mean()),
        "median_delta_full_J": float(delta.median()),
        "worst_delta_full_J": float(delta.min()),
        "gain_sum_positive": gain,
        "loss_sum_negative_abs": loss,
        "gain_to_loss_abs_ratio": float(gain / loss) if loss > 0.0 else np.inf,
        "positive_week_share": float(week_delta.gt(0.0).mean()) if len(week_delta) else np.nan,
        "worst_week_delta_full_J": float(week_delta.min()) if len(week_delta) else np.nan,
    }


def _slice_rows(frame: pd.DataFrame, *, feature: str, quantile: float, direction: str) -> tuple[pd.DataFrame, float]:
    vals = pd.to_numeric(frame[feature], errors="coerce").replace([np.inf, -np.inf], np.nan)
    valid = frame.loc[vals.notna()].copy()
    threshold = float(pd.to_numeric(valid[feature], errors="coerce").quantile(float(quantile)))
    if direction == "low":
        return valid.loc[pd.to_numeric(valid[feature], errors="coerce").le(threshold)].copy(), threshold
    return valid.loc[pd.to_numeric(valid[feature], errors="coerce").ge(threshold)].copy(), threshold


def build_slice_report(frame: pd.DataFrame, *, min_rows: int, epsilon: float, quantiles: list[float]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    base = _summary(frame, epsilon=epsilon)
    for feature in _feature_candidates(frame):
        vals = pd.to_numeric(frame[feature], errors="coerce").replace([np.inf, -np.inf], np.nan)
        valid = frame.loc[vals.notna()].copy()
        if len(valid) < max(int(min_rows) * 2, 20):
            continue
        for quantile in quantiles:
            for direction in ("low", "high"):
                selected, threshold = _slice_rows(valid, feature=feature, quantile=quantile, direction=direction)
                if len(selected) < int(min_rows):
                    continue
                rejected = valid.loc[~valid.index.isin(selected.index)].copy()
                if len(rejected) < int(min_rows):
                    continue
                selected_summary = _summary(selected, epsilon=epsilon)
                rejected_summary = _summary(rejected, epsilon=epsilon)
                rows.append(
                    {
                        "feature": feature,
                        "direction": direction,
                        "quantile": float(quantile),
                        "threshold": threshold,
                        "base_rows": base["rows"],
                        "selected_rows": selected_summary["rows"],
                        "rejected_rows": rejected_summary["rows"],
                        "selected_groups": selected_summary["groups"],
                        "selected_positive_share": selected_summary["positive_share"],
                        "rejected_positive_share": rejected_summary["positive_share"],
                        "selected_positive_epsilon_share": selected_summary["positive_epsilon_share"],
                        "selected_sum_delta_full_J": selected_summary["sum_delta_full_J"],
                        "rejected_sum_delta_full_J": rejected_summary["sum_delta_full_J"],
                        "selected_mean_delta_full_J": selected_summary["mean_delta_full_J"],
                        "rejected_mean_delta_full_J": rejected_summary["mean_delta_full_J"],
                        "selected_median_delta_full_J": selected_summary["median_delta_full_J"],
                        "selected_worst_delta_full_J": selected_summary["worst_delta_full_J"],
                        "selected_gain_sum_positive": selected_summary["gain_sum_positive"],
                        "selected_loss_sum_negative_abs": selected_summary["loss_sum_negative_abs"],
                        "selected_gain_to_loss_abs_ratio": selected_summary["gain_to_loss_abs_ratio"],
                        "selected_positive_week_share": selected_summary["positive_week_share"],
                        "selected_worst_week_delta_full_J": selected_summary["worst_week_delta_full_J"],
                        "mean_lift_vs_rejected": selected_summary["mean_delta_full_J"] - rejected_summary["mean_delta_full_J"],
                    }
                )
    report = pd.DataFrame(rows)
    if report.empty:
        return report
    report["objective"] = (
        report["selected_sum_delta_full_J"]
        + 0.25 * report["selected_worst_week_delta_full_J"].fillna(0.0)
        + 500.0 * report["selected_positive_week_share"].fillna(0.0)
        + 100.0 * report["selected_positive_epsilon_share"].fillna(0.0)
    )
    return report.sort_values(
        ["selected_sum_delta_full_J", "selected_positive_week_share", "selected_mean_delta_full_J"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def build_week_report(frame: pd.DataFrame, *, epsilon: float) -> pd.DataFrame:
    rows = []
    for week, group in frame.groupby("week_start", dropna=False):
        row = _summary(group, epsilon=epsilon)
        row["week_start"] = str(week)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("week_start").reset_index(drop=True)


def _json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    return value


def write_report(
    *,
    frame: pd.DataFrame,
    slices: pd.DataFrame,
    weeks: pd.DataFrame,
    out_dir: Path,
    action_panel: Path,
    head: str,
    epsilon: float,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    slices_out = slices.copy()
    weeks_out = weeks.copy()
    if not slices_out.empty:
        slices_out.insert(0, "head", head)
    if not weeks_out.empty:
        weeks_out.insert(0, "head", head)
    slices_out.to_csv(out_dir / "head_action_conditioning_slices.csv", index=False)
    weeks_out.to_csv(out_dir / "head_action_conditioning_by_week.csv", index=False)
    base = _summary(frame, epsilon=epsilon)
    top = slices.head(20) if not slices.empty else pd.DataFrame()
    display_cols = [
        "feature",
        "direction",
        "quantile",
        "threshold",
        "selected_rows",
        "selected_positive_share",
        "selected_positive_epsilon_share",
        "selected_sum_delta_full_J",
        "selected_mean_delta_full_J",
        "selected_positive_week_share",
        "selected_worst_week_delta_full_J",
        "selected_gain_to_loss_abs_ratio",
    ]
    lines = [
        f"# C3el {head} action-conditioning diagnostic",
        "",
        "This artifact searches observable pre-action slices where size-cut actions have positive exact-state value.",
        "",
        "## Baseline Action Rows",
        "",
        f"- rows: `{base['rows']}`",
        f"- groups: `{base['groups']}`",
        f"- positive share: `{base['positive_share']:.2%}`",
        f"- positive > epsilon share: `{base['positive_epsilon_share']:.2%}`",
        f"- sum delta full-path utility: `{base['sum_delta_full_J']:.2f}`",
        f"- gain/loss abs ratio: `{base['gain_to_loss_abs_ratio']:.3f}`",
        f"- positive week share: `{base['positive_week_share']:.2%}`",
        "",
        "## Weekly Action Value",
        "",
        weeks.to_markdown(index=False, floatfmt=".4f") if not weeks.empty else "No weekly rows.",
        "",
        "## Top Observable Slices",
        "",
        top[display_cols].to_markdown(index=False, floatfmt=".4f") if not top.empty else "No slices met the minimum-row threshold.",
        "",
        "## Readout",
        "",
    ]
    if slices.empty:
        lines.append("No observable slice met support requirements. Keep this head diagnostic-only.")
    else:
        best = slices.iloc[0]
        lines.extend(
            [
                f"Best slice: `{best['feature']} {best['direction']} q{best['quantile']:.2f}`.",
                f"It keeps `{int(best['selected_rows'])}` action rows with sum delta `{best['selected_sum_delta_full_J']:.2f}` and positive-week share `{best['selected_positive_week_share']:.2%}`.",
                "",
                "This remains a hypothesis-mining diagnostic. A promotable ablation still needs chronological holdout validation using the slice fixed in advance.",
            ]
        )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    manifest = {
        "generated_by": "diagnose_c3el_head_action_conditioning",
        "action_panel": str(action_panel),
        "head": head,
        "epsilon": float(epsilon),
        "rows": int(base["rows"]),
        "groups": int(base["groups"]),
        "outputs": {
            "summary": str(out_dir / "summary.md"),
            "slices": str(out_dir / "head_action_conditioning_slices.csv"),
            "weeks": str(out_dir / "head_action_conditioning_by_week.csv"),
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(_json_safe(manifest), indent=2, sort_keys=True))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action-panel", type=Path, required=True)
    parser.add_argument("--head", choices=HEADS, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--min-rows", type=int, default=25)
    parser.add_argument("--epsilon", type=float, default=50.0)
    parser.add_argument("--quantiles", default="0.2,0.3,0.5,0.7,0.8")
    args = parser.parse_args()
    quantiles = [float(x.strip()) for x in str(args.quantiles).split(",") if x.strip()]
    frame = load_actions(args.action_panel, head=args.head)
    slices = build_slice_report(frame, min_rows=args.min_rows, epsilon=args.epsilon, quantiles=quantiles)
    weeks = build_week_report(frame, epsilon=args.epsilon)
    write_report(
        frame=frame,
        slices=slices,
        weeks=weeks,
        out_dir=args.out_dir,
        action_panel=args.action_panel,
        head=args.head,
        epsilon=args.epsilon,
    )
    print((args.out_dir / "summary.md").read_text())


if __name__ == "__main__":
    main()
