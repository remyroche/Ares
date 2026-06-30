#!/usr/bin/env python3
"""Run a lightweight loss-gate schedule grid on scored controller bundles.

This script reuses already-scored market-state controller artifacts.  It does
not retrain state or response models and does not replay the portfolio.  The
goal is to test whether stricter action gates can turn the threshold-only
controller into a recurrently defensive shadow schedule before paying the cost
of full materialization.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_market_state_threshold_controller as mstc  # noqa: E402


DEFAULT_OUTPUT_DIR = Path("data_perp/reports/market_state_shadow_controller_loss_gate_schedule_grid")
DEFAULT_WINDOWS = [
    Path("data_perp/reports/market_state_controller_bundle_t1_lgbm_maturity_shadow_s2_loss_action5_sl030_to050_20260627_v1"),
    Path(
        "data_perp/reports/"
        "market_state_controller_bundle_score_t1_lgbm_maturity_shadow_s2_loss_action5_sl030_to050_20260627_v1_jun23_00_08"
    ),
    Path(
        "data_perp/reports/"
        "market_state_controller_bundle_score_t1_lgbm_maturity_shadow_s2_loss_action5_sl030_to050_20260627_v1_jun23_09_jun24_08"
    ),
]


@dataclass(frozen=True)
class GridConfig:
    name: str
    min_frontier_candidates: int
    min_removed_full_sl: float
    max_removed_timeout: float


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
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    view = frame.copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.6f}")
    view = view.fillna("").astype(str)
    lines = [
        "| " + " | ".join(view.columns) + " |",
        "| " + " | ".join(["---"] * len(view.columns)) + " |",
    ]
    for _, row in view.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in view.columns) + " |")
    return "\n".join(lines)


def _load_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _load_curves(bundle_path: Path) -> mstc.RankOutcomeCurves:
    bundle = joblib.load(bundle_path)
    if not isinstance(bundle, dict):
        raise TypeError(f"Expected dict bundle in {bundle_path}")
    models = bundle.get("models")
    if not isinstance(models, dict) or "curves" not in models:
        raise KeyError(f"Bundle {bundle_path} does not contain models.curves")
    return models["curves"]


def _default_grid() -> list[GridConfig]:
    rows: list[GridConfig] = []
    for min_frontier in [3, 5, 7]:
        for min_sl in [0.20, 0.25, 0.30, 0.35, 0.40]:
            for max_timeout in [0.40, 0.50, 0.60, 0.70]:
                rows.append(
                    GridConfig(
                        name=f"frontier{min_frontier}_sl{int(min_sl * 100):02d}_to{int(max_timeout * 100):02d}",
                        min_frontier_candidates=min_frontier,
                        min_removed_full_sl=min_sl,
                        max_removed_timeout=max_timeout,
                    )
                )
    return rows


def _window_label(path: Path) -> str:
    name = path.name
    if name.endswith("_jun23_00_08"):
        return "jun23_00_08"
    if name.endswith("_jun23_09_jun24_08"):
        return "jun23_09_jun24_08"
    return "jun15_22"


def _schedule_for_config(
    *,
    predictions: pd.DataFrame,
    curves: mstc.RankOutcomeCurves,
    config: GridConfig,
    arm_name: str,
    enabled_heads: set[str],
    delta_max: float,
    max_down_step: float,
    relax_alpha: float,
    min_lcb_utility: float,
    min_action_edge: float,
    winner_sacrifice_multiplier: float,
    min_prediction_coverage: float,
    min_usable_candidates: int,
) -> pd.DataFrame:
    schedule = mstc.threshold_schedule(
        predictions,
        predictions,
        curves,
        delta_max=float(delta_max),
        max_down_step=float(max_down_step),
        relax_alpha=float(relax_alpha),
        controller_mode="frontier_action_rank_grid",
        min_lcb_utility=float(min_lcb_utility),
        use_timeout_cap=False,
        min_action_edge=float(min_action_edge),
        winner_sacrifice_multiplier=float(winner_sacrifice_multiplier),
        min_removed_full_sl=float(config.min_removed_full_sl),
        max_removed_timeout=float(config.max_removed_timeout),
        enabled_heads=enabled_heads,
        min_prediction_coverage=float(min_prediction_coverage),
        min_usable_candidates=int(min_usable_candidates),
        min_frontier_candidates=int(config.min_frontier_candidates),
    )
    schedule.insert(0, "grid_config", config.name)
    schedule["arm"] = arm_name
    return schedule


def _first_all(suppression: pd.DataFrame) -> pd.Series:
    if suppression.empty:
        return pd.Series(dtype=object)
    mask = suppression["scope"].astype(str).eq("all") & suppression["scope_value"].astype(str).eq("all")
    if bool(mask.any()):
        return suppression.loc[mask].iloc[0]
    return suppression.iloc[0]


def _num(row: pd.Series, key: str, default: float = 0.0) -> float:
    if row.empty or key not in row:
        return float(default)
    value = pd.to_numeric(pd.Series([row.get(key)]), errors="coerce").iloc[0]
    return float(value) if np.isfinite(float(value)) else float(default)


def _window_summary(
    *,
    window: str,
    window_dir: Path,
    config: GridConfig,
    schedule: pd.DataFrame,
    suppression: pd.DataFrame,
) -> dict[str, Any]:
    delta = pd.to_numeric(schedule["state_threshold"], errors="coerce") - pd.to_numeric(
        schedule["base_threshold"],
        errors="coerce",
    )
    row = _first_all(suppression)
    suppressed = _num(row, "suppressed_candidates")
    loss_avoided = _num(row, "suppressed_loss_avoided")
    winner_sacrificed = _num(row, "suppressed_winner_pnl_sacrificed")
    defensive = _num(row, "realized_defensive_success")
    raised = int((delta > 1e-9).sum()) if len(delta) else 0
    return {
        "grid_config": config.name,
        "window": window,
        "window_dir": str(window_dir),
        "min_frontier_candidates": int(config.min_frontier_candidates),
        "min_removed_full_sl": float(config.min_removed_full_sl),
        "max_removed_timeout": float(config.max_removed_timeout),
        "schedule_rows": int(len(schedule)),
        "threshold_raises": raised,
        "suppressed_candidates": int(round(suppressed)),
        "loss_avoided": float(loss_avoided),
        "winner_pnl_sacrificed": float(winner_sacrificed),
        "defensive_success": float(defensive),
        "defensive_success_per_candidate": float(defensive / suppressed) if suppressed > 0 else 0.0,
        "suppressed_win_rate": _num(row, "suppressed_win_rate", np.nan),
        "suppressed_full_sl_rate": _num(row, "suppressed_full_sl_rate", np.nan),
        "suppressed_timeout_rate": _num(row, "suppressed_timeout_rate", np.nan),
        "mean_predicted_action_edge": _num(row, "mean_predicted_action_edge", np.nan),
        "positive_active_window": bool(suppressed > 0 and defensive > 0.0 and loss_avoided > winner_sacrificed),
        "negative_active_window": bool(suppressed > 0 and defensive < 0.0),
        "no_action_window": bool(suppressed <= 0),
    }


def _rollup(window_rows: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for config, g in window_rows.groupby("grid_config", sort=False):
        active = g.loc[pd.to_numeric(g["suppressed_candidates"], errors="coerce").fillna(0) > 0]
        total_suppressed = float(pd.to_numeric(g["suppressed_candidates"], errors="coerce").fillna(0).sum())
        total_loss = float(pd.to_numeric(g["loss_avoided"], errors="coerce").fillna(0).sum())
        total_winner = float(pd.to_numeric(g["winner_pnl_sacrificed"], errors="coerce").fillna(0).sum())
        total_def = float(pd.to_numeric(g["defensive_success"], errors="coerce").fillna(0).sum())
        rows.append(
            {
                "grid_config": str(config),
                "min_frontier_candidates": int(g["min_frontier_candidates"].iloc[0]),
                "min_removed_full_sl": float(g["min_removed_full_sl"].iloc[0]),
                "max_removed_timeout": float(g["max_removed_timeout"].iloc[0]),
                "window_count": int(len(g)),
                "active_window_count": int(len(active)),
                "no_action_window_count": int(g["no_action_window"].sum()),
                "negative_active_window_count": int(g["negative_active_window"].sum()),
                "positive_active_window_share": float(g["positive_active_window"].sum() / max(len(active), 1)),
                "positive_all_window_share": float(g["positive_active_window"].sum() / max(len(g), 1)),
                "total_suppressed_candidates": int(round(total_suppressed)),
                "total_loss_avoided": total_loss,
                "total_winner_pnl_sacrificed": total_winner,
                "total_defensive_success": total_def,
                "defensive_success_per_candidate": float(total_def / total_suppressed) if total_suppressed > 0 else 0.0,
                "loss_avoided_gt_winner_sacrificed": bool(total_loss > total_winner),
                "has_no_negative_active_windows": bool(int(g["negative_active_window"].sum()) == 0),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["screen_pass"] = (
        (out["total_suppressed_candidates"] > 0)
        & (out["total_defensive_success"] > 0.0)
        & (out["loss_avoided_gt_winner_sacrificed"])
        & (out["has_no_negative_active_windows"])
        & (out["positive_active_window_share"] >= 1.0)
    )
    return out.sort_values(
        [
            "screen_pass",
            "negative_active_window_count",
            "total_defensive_success",
            "defensive_success_per_candidate",
        ],
        ascending=[False, True, False, False],
    ).reset_index(drop=True)


def _write_report(output_dir: Path, rollup: pd.DataFrame, windows: pd.DataFrame, manifest: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rollup.to_csv(output_dir / "loss_gate_schedule_grid_rollup.csv", index=False)
    windows.to_csv(output_dir / "loss_gate_schedule_grid_windows.csv", index=False)
    (output_dir / "manifest.json").write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")

    top = rollup.head(12).copy()
    window_top = windows.loc[windows["grid_config"].isin(top["grid_config"].astype(str))].copy()
    lines = [
        "# Market-State Loss-Gate Schedule Grid",
        "",
        "Lightweight schedule-only comparison using already-scored controller bundles. "
        "No state/response models were retrained and no portfolio replay was rerun.",
        "",
        "## Top Rollup",
        "",
        _markdown_table(top),
        "",
        "## Top-Config Window Detail",
        "",
        _markdown_table(
            window_top[
                [
                    "grid_config",
                    "window",
                    "threshold_raises",
                    "suppressed_candidates",
                    "loss_avoided",
                    "winner_pnl_sacrificed",
                    "defensive_success",
                    "suppressed_full_sl_rate",
                    "suppressed_timeout_rate",
                    "positive_active_window",
                    "negative_active_window",
                    "no_action_window",
                ]
            ]
        ),
        "",
        "## Interpretation",
        "",
        "- `screen_pass` is a research screen, not a promotion gate. It requires positive total defensive success, "
        "loss avoided above winner PnL sacrificed, no negative active windows, and all active windows positive.",
        "- No-action windows are reported separately because fail-closed behavior can be acceptable in shadow, "
        "but it is not enough for production promotion without later recurrent evidence.",
    ]
    (output_dir / "loss_gate_schedule_grid_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_WINDOWS[0] / "market_state_controller_bundle.joblib")
    parser.add_argument("--window-dir", type=Path, action="append", default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--enabled-head", action="append", default=["short_asset", "short_boll"])
    parser.add_argument("--delta-max", type=float, default=0.10)
    parser.add_argument("--max-down-step", type=float, default=0.03)
    parser.add_argument("--relax-alpha", type=float, default=0.25)
    parser.add_argument("--min-lcb-utility", type=float, default=0.0)
    parser.add_argument("--min-action-edge", type=float, default=0.0)
    parser.add_argument("--winner-sacrifice-multiplier", type=float, default=1.0)
    parser.add_argument("--min-prediction-coverage", type=float, default=0.80)
    parser.add_argument("--min-usable-candidates", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0, help="Limit grid configs for smoke testing.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    windows = args.window_dir if args.window_dir else DEFAULT_WINDOWS
    curves = _load_curves(args.bundle)
    enabled_heads = {str(head) for head in args.enabled_head}
    configs = _default_grid()
    if int(args.limit) > 0:
        configs = configs[: int(args.limit)]

    window_rows: list[dict[str, Any]] = []
    schedules: list[pd.DataFrame] = []
    suppressions: list[pd.DataFrame] = []
    for window_dir in windows:
        candidates = _load_frame(window_dir / "controller_scored_candidates.parquet")
        predictions = _load_frame(window_dir / "controller_predictions.parquet")
        label = _window_label(window_dir)
        for config in configs:
            arm_name = f"S2_loss_gate_grid__{config.name}"
            schedule = _schedule_for_config(
                predictions=predictions,
                curves=curves,
                config=config,
                arm_name=arm_name,
                enabled_heads=enabled_heads,
                delta_max=float(args.delta_max),
                max_down_step=float(args.max_down_step),
                relax_alpha=float(args.relax_alpha),
                min_lcb_utility=float(args.min_lcb_utility),
                min_action_edge=float(args.min_action_edge),
                winner_sacrifice_multiplier=float(args.winner_sacrifice_multiplier),
                min_prediction_coverage=float(args.min_prediction_coverage),
                min_usable_candidates=int(args.min_usable_candidates),
            )
            suppression = mstc._threshold_candidate_suppression_utility(candidates, schedule)
            schedules.append(schedule.assign(window=label))
            suppressions.append(suppression.assign(window=label, grid_config=config.name))
            window_rows.append(
                _window_summary(
                    window=label,
                    window_dir=window_dir,
                    config=config,
                    schedule=schedule,
                    suppression=suppression,
                )
            )

    window_frame = pd.DataFrame(window_rows)
    rollup = _rollup(window_frame)
    output_dir.mkdir(parents=True, exist_ok=True)
    if schedules:
        pd.concat(schedules, ignore_index=True).to_parquet(output_dir / "loss_gate_schedule_grid_schedules.parquet", index=False)
    if suppressions:
        pd.concat(suppressions, ignore_index=True).to_csv(
            output_dir / "loss_gate_schedule_grid_suppression_utility.csv",
            index=False,
        )
    manifest = {
        "generated_by": Path(__file__).name,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "bundle": str(args.bundle),
        "window_dirs": [str(path) for path in windows],
        "enabled_heads": sorted(enabled_heads),
        "grid_config_count": len(configs),
        "controller_mode": "frontier_action_rank_grid",
        "delta_max": float(args.delta_max),
        "max_down_step": float(args.max_down_step),
        "relax_alpha": float(args.relax_alpha),
        "min_lcb_utility": float(args.min_lcb_utility),
        "min_action_edge": float(args.min_action_edge),
        "winner_sacrifice_multiplier": float(args.winner_sacrifice_multiplier),
        "min_prediction_coverage": float(args.min_prediction_coverage),
        "min_usable_candidates": int(args.min_usable_candidates),
    }
    _write_report(output_dir, rollup, window_frame, manifest)
    print(f"Wrote {output_dir}")
    if not rollup.empty:
        print(rollup.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
