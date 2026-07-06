#!/usr/bin/env python3
"""Sparse week-forward override for contextual TP/SL overlays.

The default policy remains a chosen overlay label.  For each evaluation week,
this script uses only prior weekly deltas to decide whether to override the
default with a defensive alternative such as no-op or a softer multiplier.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class OverrideConfig:
    signal: str
    threshold_quantile: float
    action_label: str


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _read_weeks(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"week", "label", "delta_net_pnl"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    out = frame.copy()
    out["week"] = out["week"].astype(str)
    out["label"] = out["label"].astype(str)
    out["delta_net_pnl"] = pd.to_numeric(out["delta_net_pnl"], errors="coerce").astype("float64")
    out = out.dropna(subset=["delta_net_pnl"])
    if out.empty:
        raise ValueError(f"{path} has no finite weekly deltas")
    return out


def _wide_delta(frame: pd.DataFrame, default_label: str) -> pd.DataFrame:
    wide = frame.pivot_table(index="week", columns="label", values="delta_net_pnl", aggfunc="first").sort_index()
    if default_label not in wide.columns:
        raise ValueError(f"default label {default_label!r} not found; labels={sorted(wide.columns)}")
    wide["__noop__"] = 0.0
    return wide


def _series_for_signal(default: pd.Series, signal: str) -> pd.Series:
    if signal == "prev1":
        return default.shift(1)
    if signal == "roll2_mean":
        return default.shift(1).rolling(2, min_periods=1).mean()
    if signal == "roll4_mean":
        return default.shift(1).rolling(4, min_periods=2).mean()
    if signal == "roll4_min":
        return default.shift(1).rolling(4, min_periods=2).min()
    if signal == "roll8_mean":
        return default.shift(1).rolling(8, min_periods=4).mean()
    if signal == "roll8_min":
        return default.shift(1).rolling(8, min_periods=4).min()
    raise ValueError(f"unknown signal: {signal}")


def _objective(values: pd.Series, q35_weight: float, q20_weight: float) -> float:
    vals = pd.to_numeric(values, errors="coerce").dropna()
    if vals.empty:
        return float("-inf")
    return float(vals.mean() + q35_weight * vals.quantile(0.35) + q20_weight * vals.quantile(0.20))


def _metrics(values: pd.Series) -> Dict[str, float]:
    vals = pd.to_numeric(values, errors="coerce").dropna()
    if vals.empty:
        return {
            "weeks": 0,
            "sum_delta_net_pnl": np.nan,
            "mean_delta_net_pnl": np.nan,
            "median_delta_net_pnl": np.nan,
            "q15_delta_net_pnl": np.nan,
            "q20_delta_net_pnl": np.nan,
            "q25_delta_net_pnl": np.nan,
            "q35_delta_net_pnl": np.nan,
            "positive_week_count": 0,
            "worst_delta_net_pnl": np.nan,
        }
    return {
        "weeks": int(len(vals)),
        "sum_delta_net_pnl": float(vals.sum()),
        "mean_delta_net_pnl": float(vals.mean()),
        "median_delta_net_pnl": float(vals.median()),
        "q15_delta_net_pnl": float(vals.quantile(0.15)),
        "q20_delta_net_pnl": float(vals.quantile(0.20)),
        "q25_delta_net_pnl": float(vals.quantile(0.25)),
        "q35_delta_net_pnl": float(vals.quantile(0.35)),
        "positive_week_count": int((vals > 0.0).sum()),
        "worst_delta_net_pnl": float(vals.min()),
    }


def _candidate_configs(
    *,
    labels: Iterable[str],
    signals: Iterable[str],
    threshold_quantiles: Iterable[float],
    action_labels: Iterable[str],
) -> List[OverrideConfig]:
    available = set(labels) | {"__noop__"}
    out: List[OverrideConfig] = []
    for action in action_labels:
        if action not in available:
            continue
        for signal in signals:
            for q in threshold_quantiles:
                out.append(OverrideConfig(signal=signal, threshold_quantile=float(q), action_label=action))
    return out


def _apply_config(
    *,
    wide: pd.DataFrame,
    default_label: str,
    config: OverrideConfig,
    threshold: float,
    eval_weeks: List[str],
) -> pd.Series:
    default = wide[default_label].astype("float64")
    signal = _series_for_signal(default, config.signal)
    action = wide[config.action_label].astype("float64")
    values = default.copy()
    trigger = signal.lt(float(threshold))
    values.loc[trigger] = action.loc[trigger]
    return values.loc[eval_weeks]


def _select_config_for_week(
    *,
    wide: pd.DataFrame,
    default_label: str,
    configs: List[OverrideConfig],
    train_weeks: List[str],
    q35_weight: float,
    q20_weight: float,
    min_train_objective_improvement: float,
    max_train_trigger_share: float,
    min_train_q35_delta_vs_default: float,
) -> tuple[OverrideConfig | None, float, float, float]:
    best: OverrideConfig | None = None
    best_score = float("-inf")
    best_threshold = np.nan
    default = wide[default_label].astype("float64")
    default_values = default.loc[train_weeks]
    default_score = _objective(default_values, q35_weight, q20_weight)
    finite_default_values = pd.to_numeric(default_values, errors="coerce").dropna()
    default_q35 = float(finite_default_values.quantile(0.35)) if finite_default_values.size else np.nan
    for config in configs:
        signal = _series_for_signal(default, config.signal)
        train_signal = signal.loc[train_weeks].dropna()
        if train_signal.empty:
            continue
        threshold = float(train_signal.quantile(config.threshold_quantile))
        trigger_count = int(signal.loc[train_weeks].lt(threshold).sum())
        trigger_share = trigger_count / float(len(train_weeks)) if train_weeks else 0.0
        if trigger_share > float(max_train_trigger_share):
            continue
        values = _apply_config(
            wide=wide,
            default_label=default_label,
            config=config,
            threshold=threshold,
            eval_weeks=train_weeks,
        )
        score = _objective(values, q35_weight, q20_weight)
        if score - default_score < float(min_train_objective_improvement):
            continue
        finite_values = pd.to_numeric(values, errors="coerce").dropna()
        q35_delta = float(finite_values.quantile(0.35) - default_q35) if finite_values.size and np.isfinite(default_q35) else np.nan
        if np.isfinite(q35_delta) and q35_delta < float(min_train_q35_delta_vs_default):
            continue
        # Prefer sparse overrides on ties, then higher total delta.
        tie_break = score - 1e-6 * trigger_count + 1e-9 * float(values.sum())
        if tie_break > best_score:
            best = config
            best_score = tie_break
            best_threshold = threshold
    return best, best_threshold, best_score, default_score


def _format_table(frame: pd.DataFrame, columns: List[str], limit: int = 30) -> str:
    cur = frame[[col for col in columns if col in frame.columns]].head(limit).copy()
    if cur.empty:
        return "_No rows._"
    for col in cur.columns:
        if pd.api.types.is_float_dtype(cur[col]):
            cur[col] = cur[col].map(lambda value: "" if pd.isna(value) else f"{value:.6g}")
    return cur.to_markdown(index=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weekly-variants", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--default-label", default="nodrift_lb250_ld250_sa250")
    parser.add_argument("--min-train-weeks", type=int, default=8)
    parser.add_argument("--rolling-train-weeks", type=int, default=0)
    parser.add_argument("--q35-weight", type=float, default=0.70)
    parser.add_argument("--q20-weight", type=float, default=0.30)
    parser.add_argument(
        "--signals",
        default="prev1,roll2_mean,roll4_mean,roll4_min,roll8_mean,roll8_min",
    )
    parser.add_argument("--threshold-quantiles", default="0.05,0.10,0.15,0.20,0.25,0.35")
    parser.add_argument(
        "--min-train-objective-improvement",
        type=float,
        default=0.0,
        help="Required prior-week objective lift versus staying with the default overlay.",
    )
    parser.add_argument(
        "--max-train-trigger-share",
        type=float,
        default=1.0,
        help="Maximum allowed share of prior weeks where a candidate override would trigger.",
    )
    parser.add_argument(
        "--min-train-q35-delta-vs-default",
        type=float,
        default=-1e18,
        help="Minimum prior-week q35 delta versus the default overlay for a candidate override.",
    )
    parser.add_argument(
        "--edge-trigger",
        action="store_true",
        help="Trigger only when the selected signal newly crosses below the threshold.",
    )
    parser.add_argument(
        "--trigger-cooldown-weeks",
        type=int,
        default=0,
        help="Suppress new triggers for this many eval weeks after a trigger fires.",
    )
    parser.add_argument(
        "--action-labels",
        default="__noop__,nodrift_lb250_ld500_sa250,nodrift_lb250_ld750_sa250,nodrift_lb250_ld250_sa500,nodrift_lb250_ld250_sa750,nodrift_lb500_ld250_sa250,nodrift_lb750_ld250_sa250",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    weeks = _read_weeks(Path(args.weekly_variants))
    wide = _wide_delta(weeks, args.default_label)
    week_order = list(wide.index.astype(str))
    signals = [x.strip() for x in str(args.signals).split(",") if x.strip()]
    quantiles = [float(x.strip()) for x in str(args.threshold_quantiles).split(",") if x.strip()]
    action_labels = [x.strip() for x in str(args.action_labels).split(",") if x.strip()]
    configs = _candidate_configs(
        labels=wide.columns.astype(str),
        signals=signals,
        threshold_quantiles=quantiles,
        action_labels=action_labels,
    )
    if not configs:
        raise ValueError("No candidate override configs after filtering action labels")

    selections: List[Dict[str, Any]] = []
    cooldown_remaining = 0
    for pos, eval_week in enumerate(week_order):
        prior = week_order[:pos]
        if len(prior) < int(args.min_train_weeks):
            continue
        if int(args.rolling_train_weeks) > 0:
            prior = prior[-int(args.rolling_train_weeks) :]
        best, threshold, train_score, default_train_score = _select_config_for_week(
            wide=wide,
            default_label=str(args.default_label),
            configs=configs,
            train_weeks=prior,
            q35_weight=float(args.q35_weight),
            q20_weight=float(args.q20_weight),
            min_train_objective_improvement=float(args.min_train_objective_improvement),
            max_train_trigger_share=float(args.max_train_trigger_share),
            min_train_q35_delta_vs_default=float(args.min_train_q35_delta_vs_default),
        )
        default = wide[str(args.default_label)].astype("float64")
        if best is None:
            signal_val = np.nan
            triggered = False
            chosen_label = str(args.default_label)
            action_label = "__default__"
            signal_name = "__default__"
            threshold_quantile = np.nan
            threshold_value = np.nan
            train_score = default_train_score
        else:
            signal_series = _series_for_signal(default, best.signal)
            signal_val = float(signal_series.loc[eval_week])
            triggered = bool(np.isfinite(signal_val) and signal_val < float(threshold))
            edge_ok = True
            if bool(args.edge_trigger):
                prev_pos = week_order.index(eval_week) - 1
                prev_week = week_order[prev_pos] if prev_pos >= 0 else None
                prev_signal = float(signal_series.loc[prev_week]) if prev_week is not None else np.nan
                edge_ok = bool(np.isfinite(prev_signal) and prev_signal >= float(threshold))
                triggered = bool(triggered and edge_ok)
            if cooldown_remaining > 0:
                triggered = False
            chosen_label = best.action_label if triggered else str(args.default_label)
            action_label = best.action_label
            signal_name = best.signal
            threshold_quantile = best.threshold_quantile
            threshold_value = float(threshold)
        eval_delta = float(wide.loc[eval_week, chosen_label])
        default_delta = float(wide.loc[eval_week, str(args.default_label)])
        selections.append(
            {
                "eval_week": eval_week,
                "chosen_label": chosen_label,
                "default_label": str(args.default_label),
                "action_label": action_label,
                "signal": signal_name,
                "threshold_quantile": threshold_quantile,
                "threshold": threshold_value,
                "signal_value": signal_val,
                "triggered": triggered,
                "cooldown_remaining_before": int(cooldown_remaining),
                "train_weeks": len(prior),
                "train_selector_score": float(train_score),
                "default_train_selector_score": float(default_train_score),
                "train_selector_score_lift": float(train_score - default_train_score),
                "eval_delta_net_pnl": eval_delta,
                "default_eval_delta_net_pnl": default_delta,
                "incremental_delta_vs_default": eval_delta - default_delta,
            }
        )
        if cooldown_remaining > 0:
            cooldown_remaining -= 1
        if triggered and int(args.trigger_cooldown_weeks) > 0:
            cooldown_remaining = int(args.trigger_cooldown_weeks)
    selections_df = pd.DataFrame(selections)
    selections_df.to_csv(out_dir / "sparse_override_selections.csv", index=False)

    eval_weeks = selections_df["eval_week"].tolist() if not selections_df.empty else []
    default_eval = wide.loc[eval_weeks, str(args.default_label)] if eval_weeks else pd.Series(dtype=float)
    override_eval = selections_df["eval_delta_net_pnl"] if not selections_df.empty else pd.Series(dtype=float)
    summary = {
        "mode": "rolling" if int(args.rolling_train_weeks) > 0 else "expanding",
        "default_label": str(args.default_label),
        "min_train_objective_improvement": float(args.min_train_objective_improvement),
        "max_train_trigger_share": float(args.max_train_trigger_share),
        "min_train_q35_delta_vs_default": float(args.min_train_q35_delta_vs_default),
        "edge_trigger": bool(args.edge_trigger),
        "trigger_cooldown_weeks": int(args.trigger_cooldown_weeks),
        "eval_weeks": int(len(eval_weeks)),
        "triggered_weeks": int(selections_df["triggered"].sum()) if not selections_df.empty else 0,
        "chosen_labels": selections_df["chosen_label"].value_counts().to_dict() if not selections_df.empty else {},
        "actions_selected": selections_df["action_label"].value_counts().to_dict() if not selections_df.empty else {},
        "override": _metrics(override_eval),
        "default_on_same_weeks": _metrics(default_eval),
    }
    for key, val in summary["default_on_same_weeks"].items():
        if key == "weeks":
            continue
        summary[f"delta_{key}_vs_default"] = summary["override"].get(key, np.nan) - val
    summary["override_objective"] = _objective(override_eval, float(args.q35_weight), float(args.q20_weight))
    summary["default_objective"] = _objective(default_eval, float(args.q35_weight), float(args.q20_weight))
    summary["delta_objective_vs_default"] = summary["override_objective"] - summary["default_objective"]
    (out_dir / "sparse_override_summary.json").write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Sparse Weak-Week Override Report",
        "",
        f"Default label: `{args.default_label}`",
        f"Training mode: `{summary['mode']}`; min train weeks: `{args.min_train_weeks}`; rolling train weeks: `{args.rolling_train_weeks}`",
        f"Guardrails: min train objective lift `{args.min_train_objective_improvement}`, max train trigger share `{args.max_train_trigger_share}`, min train q35 delta `{args.min_train_q35_delta_vs_default}`",
        f"Trigger controls: edge trigger `{bool(args.edge_trigger)}`, cooldown weeks `{int(args.trigger_cooldown_weeks)}`",
        "",
        "## Summary",
        "",
        _format_table(
            pd.DataFrame(
                [
                    {
                        "variant": "sparse_override",
                        **summary["override"],
                        "objective": summary["override_objective"],
                    },
                    {
                        "variant": "default_same_weeks",
                        **summary["default_on_same_weeks"],
                        "objective": summary["default_objective"],
                    },
                ]
            ),
            [
                "variant",
                "weeks",
                "sum_delta_net_pnl",
                "mean_delta_net_pnl",
                "median_delta_net_pnl",
                "q15_delta_net_pnl",
                "q20_delta_net_pnl",
                "q25_delta_net_pnl",
                "q35_delta_net_pnl",
                "positive_week_count",
                "worst_delta_net_pnl",
                "objective",
            ],
        ),
        "",
        "## Chosen Labels",
        "",
        json.dumps(_json_safe(summary["chosen_labels"]), indent=2, sort_keys=True),
        "",
        "## Recent Selections",
        "",
        _format_table(
            selections_df.tail(20),
            [
                "eval_week",
                "chosen_label",
                "action_label",
                "signal",
                "threshold_quantile",
                "threshold",
                "signal_value",
                "triggered",
                "cooldown_remaining_before",
                "train_selector_score_lift",
                "eval_delta_net_pnl",
                "default_eval_delta_net_pnl",
                "incremental_delta_vs_default",
            ],
            limit=20,
        ),
    ]
    (out_dir / "sparse_override_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
