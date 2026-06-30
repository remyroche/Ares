#!/usr/bin/env python3
"""Compare performance-regime market-state modulator runs and replay actions."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.performance_regimes.archetypes import MarketStateArchetype  # noqa: E402
from extreme_price_movements.performance_regimes.portfolio_calibration import (  # noqa: E402
    score_frozen_portfolio_calibrator,
)


def _load_runner_module():
    script = ROOT / "scripts" / "run_performance_market_state_modulator.py"
    spec = importlib.util.spec_from_file_location("run_performance_market_state_modulator", script)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f"Could not load {script}")
    spec.loader.exec_module(module)
    return module


def _write_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    frame.to_csv(path.with_suffix(".csv"), index=False)


def _read_json(path: Path) -> Any:
    with path.open() as fh:
        return json.load(fh)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        path = Path(value)
        return path.name, path
    name, path = value.split("=", 1)
    return name.strip(), Path(path.strip())


def _cap_from_name(name: str) -> int | None:
    digits = ""
    marker = "cap"
    if marker in name:
        tail = name.split(marker, 1)[1]
        for char in tail:
            if char.isdigit():
                digits += char
            else:
                break
    return int(digits) if digits else None


def _load_run_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def _iter_scope_roots(root: Path) -> list[tuple[str, Path]]:
    head_roots = sorted(path for path in root.glob("head_*") if path.is_dir())
    if head_roots:
        return [(path.name.replace("head_", "", 1), path) for path in head_roots]
    return [("global", root)]


def _iter_fold_roots(root: Path) -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    for scope, scope_root in _iter_scope_roots(root):
        out.extend((scope, path) for path in sorted(scope_root.glob("fold_*")) if path.is_dir())
    return out


def _load_archetypes(path: Path) -> tuple[MarketStateArchetype, ...]:
    raw = _read_json(path)
    out: list[MarketStateArchetype] = []
    for item in raw:
        out.append(
            MarketStateArchetype(
                archetype_id=str(item.get("archetype_id")),
                strategy=str(item.get("strategy")),
                direction=str(item.get("direction")),
                leaf_ids=tuple(str(v) for v in item.get("leaf_ids", ())),
                dominant_features=tuple(str(v) for v in item.get("dominant_features", ())),
                dominant_feature_families=tuple(str(v) for v in item.get("dominant_feature_families", ())),
                total_weighted_coverage=float(item.get("total_weighted_coverage", 0.0)),
                mean_edge_mass=float(item.get("mean_edge_mass", 0.0)),
                mean_contribution_share=float(item.get("mean_contribution_share", 0.0)),
                mean_stability=float(item.get("mean_stability", 0.0)),
                activation_timestamps=np.asarray(item.get("activation_timestamps", ()), dtype=bool),
                diagnostics=dict(item.get("diagnostics", {})),
            )
        )
    return tuple(out)


def _ensure_compression_diagnostics(run_root: Path, fold: str, runner) -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_root = run_root / fold
    selection_path = fold_root / "archetypes" / "archetype_selection.parquet"
    diagnostics_path = fold_root / "archetypes" / "archetype_compression_diagnostics.parquet"
    selection = _load_run_table(selection_path)
    diagnostics = _load_run_table(diagnostics_path)
    if not selection.empty and not diagnostics.empty and "compression_silhouette" in selection.columns:
        return selection, diagnostics

    all_path = fold_root / "archetypes" / "archetype_definitions_all.json"
    if not all_path.exists() or selection.empty:
        return selection, diagnostics
    archetypes = _load_archetypes(all_path)
    selection, diagnostics, _metrics = runner._build_archetype_compression_diagnostics(archetypes, selection)
    return selection, diagnostics


def build_comparison(runs: dict[str, Path], output_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    runner = _load_runner_module()
    for run_name, root in runs.items():
        gate_frames = []
        stage_frames = []
        for scope, scope_root in _iter_scope_roots(root):
            gate_frame = _load_run_table(scope_root / "performance_market_state_gate_report.parquet")
            if not gate_frame.empty:
                gate_frame = gate_frame.copy()
                gate_frame["scope"] = scope
                gate_frames.append(gate_frame)
            stage_frame = _load_run_table(scope_root / "performance_market_state_stage_report.parquet")
            if not stage_frame.empty:
                stage_frame = stage_frame.copy()
                stage_frame["scope"] = scope
                stage_frames.append(stage_frame)
        gate = pd.concat(gate_frames, ignore_index=True) if gate_frames else pd.DataFrame()
        stage = pd.concat(stage_frames, ignore_index=True) if stage_frames else pd.DataFrame()
        rows.append(
            {
                "run": run_name,
                "fold": np.nan,
                "scope": "all",
                "category": "run",
                "metric": "gate_failures",
                "value": int((~gate["gate_passed"].fillna(False)).sum()) if not gate.empty else np.nan,
            }
        )
        if not stage.empty and "status" in stage.columns:
                rows.append(
                    {
                        "run": run_name,
                        "fold": np.nan,
                        "scope": "all",
                        "category": "run",
                        "metric": "failed_stage_rows",
                        "value": int(stage["status"].eq("fail").sum()),
                }
            )
        for _idx, row in gate.iterrows():
            fold = row.get("fold")
            scope = str(row.get("scope", "global"))
            stage_name = str(row.get("stage"))
            for metric in [
                "mean_oof_weighted_brier",
                "median_prediction_std",
                "extracted_leaf_count",
                "pruned_leaf_count",
                "pair_count",
                "triple_count",
                "base_oof_weighted_brier",
                "second_pass_oof_weighted_brier",
                "raw_archetype_count",
                "archetype_count",
                "compression_silhouette_mean",
                "compression_silhouette_q10",
                "compression_member_count_cov_max",
                "compression_distance_to_seed_p95",
                "compression_source_coverage_min",
                "expert_count",
                "predictive_expert_count",
                "predictive_expert_fold_share",
                "mean_prediction_std",
                "input_score_count",
                "mean_action_prediction_std",
                "activation_target_deactivation_share",
            ]:
                if metric in gate.columns and pd.notna(row.get(metric)):
                    rows.append(
                        {
                            "run": run_name,
                            "fold": fold,
                            "scope": scope,
                            "category": stage_name,
                            "metric": metric,
                            "value": row.get(metric),
                        }
                    )
            if stage_name == "feedback_operator_generation_and_second_pass":
                accepted = row.get("second_pass_accepted")
                if pd.notna(accepted):
                    rows.append(
                        {
                            "run": run_name,
                            "fold": fold,
                            "scope": scope,
                            "category": stage_name,
                            "metric": "second_pass_accepted",
                            "value": float(bool(accepted)),
                        }
                    )
        for scope, fold_root in _iter_fold_roots(root):
            fold = int(fold_root.name.split("_")[-1])
            scope_gate = gate.loc[gate.get("scope", pd.Series(dtype=object)).astype(str).eq(scope)] if not gate.empty else gate
            already_has_compression = (
                not scope_gate.empty
                and "compression_silhouette_mean" in gate.columns
                and scope_gate.loc[
                    (pd.to_numeric(scope_gate.get("fold"), errors="coerce") == fold)
                    & scope_gate["stage"].eq("cluster_archetypes"),
                    "compression_silhouette_mean",
                ].notna().any()
            )
            if already_has_compression:
                continue
            selection, diagnostics = _ensure_compression_diagnostics(fold_root.parent, fold_root.name, runner)
            if selection.empty or diagnostics.empty:
                continue
            distances = pd.to_numeric(selection.get("compression_distance_to_seed"), errors="coerce")
            silhouettes = pd.to_numeric(selection.get("compression_silhouette"), errors="coerce")
            metrics = {
                "compression_silhouette_mean": silhouettes.mean(),
                "compression_silhouette_q10": silhouettes.quantile(0.10),
                "compression_member_count_cov_max": pd.to_numeric(
                    diagnostics.get("member_count_cov"),
                    errors="coerce",
                ).max(),
                "compression_distance_to_seed_p95": distances.quantile(0.95),
                "compression_source_coverage_min": pd.to_numeric(
                    diagnostics.get("source_coverage"),
                    errors="coerce",
                ).min(),
            }
            for metric, value in metrics.items():
                if pd.notna(value):
                    rows.append(
                        {
                            "run": run_name,
                            "fold": fold,
                            "scope": scope,
                            "category": "cluster_archetypes_offline_diagnostics",
                            "metric": metric,
                            "value": float(value),
                        }
                    )
    out = pd.DataFrame(rows)
    _write_frame(output_dir / "run_comparison_metrics.parquet", out)
    wide = out.pivot_table(index=["scope", "fold", "category", "metric"], columns="run", values="value", aggfunc="first")
    wide = wide.reset_index()
    _write_frame(output_dir / "run_comparison_wide.parquet", wide)
    return out


def _timestamp_indexed(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    first = str(out.columns[0])
    if "timestamp" in out.columns:
        ts_col = "timestamp"
    else:
        ts_col = first
    out[ts_col] = pd.to_datetime(out[ts_col], utc=True, errors="coerce")
    out = out.dropna(subset=[ts_col]).set_index(ts_col).sort_index()
    return out


def _performance_pivot(input_path: Path, strategy_col: str, performance_col: str) -> pd.DataFrame:
    frame = pd.read_parquet(input_path) if input_path.suffix.lower() in {".parquet", ".pq"} else pd.read_csv(input_path)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    return (
        frame.dropna(subset=["timestamp"])
        .pivot_table(index="timestamp", columns=strategy_col, values=performance_col, aggfunc="mean")
        .sort_index()
    )


def _calibrator_direct_impact(calibrator) -> dict[str, float]:
    feature_columns = list(calibrator.feature_columns)
    impacts = {col: 0.0 for col in feature_columns}
    for action_map in calibrator.coefficients.values():
        for coef in action_map.values():
            coef_arr = np.asarray(coef, dtype=float)
            for i, value in enumerate(coef_arr[: len(feature_columns)]):
                impacts[feature_columns[i]] += abs(float(value))
    return impacts


def build_archetype_inspection(
    runs: dict[str, Path],
    *,
    input_path: Path,
    strategy_col: str,
    performance_col: str,
    output_dir: Path,
    top_n: int,
) -> pd.DataFrame:
    returns = _performance_pivot(input_path, strategy_col, performance_col)
    runner = _load_runner_module()
    rows: list[dict[str, Any]] = []
    for run_name, root in runs.items():
        for scope, fold_root in _iter_fold_roots(root):
            fold = int(fold_root.name.split("_")[-1])
            definitions_path = fold_root / "archetypes" / "archetype_definitions.json"
            if not definitions_path.exists():
                continue
            archetypes = _load_archetypes(definitions_path)
            selection, _diagnostics = _ensure_compression_diagnostics(fold_root.parent, fold_root.name, runner)
            source_counts = (
                selection.groupby("compressed_archetype_id").size().to_dict()
                if not selection.empty and "compressed_archetype_id" in selection.columns
                else {}
            )
            expert = _load_run_table(fold_root / "evaluation" / "archetype_expert_oof_metrics.parquet")
            expert_agg = (
                expert.groupby("archetype_id")
                .agg(
                    expert_oof_brier=("oof_weighted_brier", "mean"),
                    expert_prediction_std=("prediction_std", "mean"),
                )
                .to_dict("index")
                if not expert.empty
                else {}
            )
            intensities = _timestamp_indexed(_load_run_table(fold_root / "archetypes" / "archetype_intensities.parquet"))
            calibrator_path = fold_root / "models" / "portfolio_calibrator" / "portfolio_calibrator.joblib"
            direct_impact = {}
            if calibrator_path.exists():
                direct_impact = _calibrator_direct_impact(joblib.load(calibrator_path))
            for archetype in archetypes:
                intensity = (
                    pd.to_numeric(intensities.get(archetype.archetype_id), errors="coerce")
                    .reindex(intensities.index)
                    .fillna(0.0)
                    if archetype.archetype_id in intensities.columns
                    else pd.Series(0.0, index=intensities.index)
                )
                strategy_returns = pd.to_numeric(
                    returns.get(archetype.strategy, pd.Series(np.nan, index=intensities.index)),
                    errors="coerce",
                ).reindex(intensity.index)
                active = intensity >= 0.5
                active_perf = strategy_returns.loc[active].mean() if active.any() else np.nan
                inactive_perf = strategy_returns.loc[~active].mean() if (~active).any() else np.nan
                exp = expert_agg.get(archetype.archetype_id, {})
                rows.append(
                    {
                        "run": run_name,
                        "scope": scope,
                        "fold": fold,
                        "archetype_id": archetype.archetype_id,
                        "strategy": archetype.strategy,
                        "direction": archetype.direction,
                        "source_archetype_count": int(source_counts.get(archetype.archetype_id, 1)),
                        "leaf_count": int(len(archetype.leaf_ids)),
                        "dominant_features": ", ".join(archetype.dominant_features[:8]),
                        "dominant_feature_families": ", ".join(archetype.dominant_feature_families[:6]),
                        "total_weighted_coverage": float(archetype.total_weighted_coverage),
                        "mean_edge_mass": float(archetype.mean_edge_mass),
                        "mean_contribution_share": float(archetype.mean_contribution_share),
                        "mean_stability": float(archetype.mean_stability),
                        "mean_intensity": float(intensity.mean()),
                        "max_intensity": float(intensity.max()),
                        "active_share_ge_0_5": float(active.mean()),
                        "strategy_perf_when_active": float(active_perf) if pd.notna(active_perf) else np.nan,
                        "strategy_perf_when_inactive": float(inactive_perf) if pd.notna(inactive_perf) else np.nan,
                        "active_minus_inactive_perf": float(active_perf - inactive_perf)
                        if pd.notna(active_perf) and pd.notna(inactive_perf)
                        else np.nan,
                        "expert_oof_brier": float(exp.get("expert_oof_brier", np.nan)),
                        "expert_prediction_std": float(exp.get("expert_prediction_std", np.nan)),
                        "portfolio_direct_coefficient_abs_sum": float(direct_impact.get(archetype.archetype_id, 0.0)),
                    }
                )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["inspection_score"] = (
            pd.to_numeric(out["portfolio_direct_coefficient_abs_sum"], errors="coerce").fillna(0.0)
            * (1.0 + pd.to_numeric(out["expert_prediction_std"], errors="coerce").fillna(0.0))
            * (1.0 + pd.to_numeric(out["source_archetype_count"], errors="coerce").fillna(1.0))
        )
        out = out.sort_values(["run", "fold", "inspection_score"], ascending=[True, True, False])
    _write_frame(output_dir / "archetype_inspection.parquet", out)
    md_path = output_dir / "archetype_inspection_top.md"
    lines = ["# Top Archetype Inspection", ""]
    for (run_name, scope, fold), group in out.groupby(["run", "scope", "fold"], sort=True):
        lines.append(f"## {run_name} {scope} fold {int(fold)}")
        lines.append("")
        for row in group.head(int(top_n)).itertuples(index=False):
            lines.append(
                f"- `{row.archetype_id}` {row.strategy}/{row.direction}: "
                f"sources={row.source_archetype_count}, active_share={row.active_share_ge_0_5:.3f}, "
                f"active_perf={row.strategy_perf_when_active:.6g}, inactive_perf={row.strategy_perf_when_inactive:.6g}, "
                f"expert_brier={row.expert_oof_brier:.6g}, coeff_impact={row.portfolio_direct_coefficient_abs_sum:.6g}"
            )
            if row.dominant_features:
                lines.append(f"  - features: {row.dominant_features}")
        lines.append("")
    md_path.write_text("\n".join(lines))
    return out


def _normalize_available_weights(weights: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
    aligned = weights.reindex(index=returns.index, columns=returns.columns).fillna(0.0).astype(float)
    aligned = aligned.where(returns.notna(), 0.0)
    row_sum = aligned.sum(axis=1)
    mask = row_sum.abs() > 1e-12
    aligned.loc[mask] = aligned.loc[mask].div(row_sum.loc[mask], axis=0)
    return aligned


def _portfolio_metrics(weights: pd.DataFrame, returns: pd.DataFrame, *, policy: str) -> dict[str, float | str]:
    weights = _normalize_available_weights(weights, returns)
    pnl = (weights * returns.fillna(0.0)).sum(axis=1)
    active_count = (weights.abs() > 1e-12).sum(axis=1)
    cumulative = pnl.cumsum()
    drawdown = cumulative - cumulative.cummax()
    turnover = weights.diff().abs().sum(axis=1).fillna(0.0)
    std = float(pnl.std(ddof=0))
    out: dict[str, float | str] = {
        "policy": policy,
        "timestamp_count": int(len(pnl)),
        "total_return_sum": float(pnl.sum()),
        "mean_return": float(pnl.mean()),
        "std_return": std,
        "sharpe_like": float(pnl.mean() / std) if std > 1e-12 else 0.0,
        "hit_rate": float((pnl > 0.0).mean()) if len(pnl) else np.nan,
        "loss_rate": float((pnl < 0.0).mean()) if len(pnl) else np.nan,
        "max_drawdown_abs": float(abs(drawdown.min())) if len(drawdown) else 0.0,
        "cash_share": float((active_count == 0).mean()) if len(active_count) else np.nan,
        "avg_active_strategies": float(active_count.mean()) if len(active_count) else np.nan,
        "turnover_mean": float(turnover.mean()) if len(turnover) else np.nan,
        "turnover_sum": float(turnover.sum()) if len(turnover) else np.nan,
    }
    for strategy in returns.columns:
        out[f"{strategy}__active_share"] = float((weights[strategy].abs() > 1e-12).mean())
        out[f"{strategy}__avg_weight"] = float(weights[strategy].mean())
    return out


def _median_step_hours(index: pd.Index) -> float:
    if isinstance(index, pd.DatetimeIndex) and len(index) > 1:
        diffs = index.sort_values().to_series().diff().dropna().dt.total_seconds().to_numpy(dtype=float)
        diffs = diffs[np.isfinite(diffs) & (diffs > 0.0)]
        if len(diffs):
            return max(float(np.nanmedian(diffs) / 3600.0), 1e-9)
    return 1.0


def _condition_streak_hours(condition: pd.Series, *, step_hours: float | None = None) -> pd.Series:
    mask = condition.fillna(False).to_numpy(dtype=bool, copy=False)
    bar_hours = float(step_hours if step_hours is not None else _median_step_hours(condition.index))
    out = np.zeros(len(mask), dtype=np.float32)
    current = 0.0
    for i, active in enumerate(mask):
        if bool(active):
            current += bar_hours
        else:
            current = 0.0
        out[i] = current
    return pd.Series(out, index=condition.index)


def _safe_mask_mean(mask: pd.Series, values: pd.Series) -> float:
    selected = values.loc[mask.fillna(False)]
    return float(selected.mean()) if len(selected) else np.nan


def _bad_regime_exposure_rows(
    *,
    run_name: str,
    scope: str,
    fold: int,
    policy: str,
    weights: pd.DataFrame,
    baseline_weights: pd.DataFrame,
    returns: pd.DataFrame,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    aligned_weights = weights.reindex(index=returns.index, columns=returns.columns).fillna(0.0)
    aligned_baseline = baseline_weights.reindex(index=returns.index, columns=returns.columns).fillna(0.0)
    step_hours = _median_step_hours(returns.index)
    for strategy in returns.columns:
        ret = pd.to_numeric(returns[strategy], errors="coerce")
        available = ret.notna()
        if not bool(available.any()):
            continue
        weight = pd.to_numeric(aligned_weights[strategy], errors="coerce").fillna(0.0)
        baseline = pd.to_numeric(aligned_baseline[strategy], errors="coerce").fillna(0.0)
        active = weight.abs() > 1e-12
        baseline_active = baseline.abs() > 1e-12
        clean_ret = ret.fillna(0.0)
        q05 = float(ret.loc[available].quantile(0.05))
        q10 = float(ret.loc[available].quantile(0.10))
        worst_05 = available & ret.le(q05)
        worst_10 = available & ret.le(q10)
        negative = available & ret.lt(0.0)
        raw_loss_streak = _condition_streak_hours(negative, step_hours=step_hours)
        prior_loss_streak = raw_loss_streak.shift(1).fillna(0.0)
        active_loss_streak = _condition_streak_hours(active & negative, step_hours=step_hours)

        base_return = baseline * clean_ret
        policy_return = weight * clean_ret
        row: dict[str, Any] = {
            "run": run_name,
            "scope": scope,
            "fold": int(fold),
            "policy": policy,
            "strategy": strategy,
            "timestamp_count": int(len(ret)),
            "available_count": int(available.sum()),
            "step_hours": float(step_hours),
            "return_q05": q05,
            "return_q10": q10,
            "active_share": _safe_mask_mean(available, active.astype(float)),
            "baseline_active_share": _safe_mask_mean(available, baseline_active.astype(float)),
            "negative_return_count": int(negative.sum()),
            "negative_return_active_share": _safe_mask_mean(negative, active.astype(float)),
            "negative_return_baseline_active_share": _safe_mask_mean(negative, baseline_active.astype(float)),
            "negative_return_policy_return_sum": float(policy_return.loc[negative].sum()),
            "negative_return_baseline_return_sum": float(base_return.loc[negative].sum()),
            "negative_return_delta_vs_baseline": float(
                policy_return.loc[negative].sum() - base_return.loc[negative].sum()
            ),
            "worst_5pct_count": int(worst_05.sum()),
            "worst_5pct_active_share": _safe_mask_mean(worst_05, active.astype(float)),
            "worst_5pct_baseline_active_share": _safe_mask_mean(worst_05, baseline_active.astype(float)),
            "worst_5pct_policy_return_sum": float(policy_return.loc[worst_05].sum()),
            "worst_5pct_baseline_return_sum": float(base_return.loc[worst_05].sum()),
            "worst_5pct_delta_vs_baseline": float(
                policy_return.loc[worst_05].sum() - base_return.loc[worst_05].sum()
            ),
            "worst_10pct_count": int(worst_10.sum()),
            "worst_10pct_active_share": _safe_mask_mean(worst_10, active.astype(float)),
            "worst_10pct_baseline_active_share": _safe_mask_mean(worst_10, baseline_active.astype(float)),
            "worst_10pct_policy_return_sum": float(policy_return.loc[worst_10].sum()),
            "worst_10pct_baseline_return_sum": float(base_return.loc[worst_10].sum()),
            "worst_10pct_delta_vs_baseline": float(
                policy_return.loc[worst_10].sum() - base_return.loc[worst_10].sum()
            ),
            "raw_max_loss_streak_hours": float(raw_loss_streak.max()),
            "policy_max_active_loss_streak_hours": float(active_loss_streak.max()),
            "max_prior_loss_streak_hours": float(prior_loss_streak.max()),
        }
        for hours in (24.0, 72.0, 168.0):
            streak_mask = available & prior_loss_streak.ge(hours)
            prefix = f"prior_loss_streak_ge_{int(hours)}h"
            row[f"{prefix}_count"] = int(streak_mask.sum())
            row[f"{prefix}_active_share"] = _safe_mask_mean(streak_mask, active.astype(float))
            row[f"{prefix}_baseline_active_share"] = _safe_mask_mean(streak_mask, baseline_active.astype(float))
            row[f"{prefix}_continuation_loss_rate"] = _safe_mask_mean(streak_mask, clean_ret.lt(0.0).astype(float))
            row[f"{prefix}_policy_return_sum"] = float(policy_return.loc[streak_mask].sum())
            row[f"{prefix}_baseline_return_sum"] = float(base_return.loc[streak_mask].sum())
            row[f"{prefix}_delta_vs_baseline"] = float(
                policy_return.loc[streak_mask].sum() - base_return.loc[streak_mask].sum()
            )
        rows.append(row)
    return rows


def _weights_from_actions(
    actions: pd.DataFrame,
    *,
    strategies: list[str],
    index: pd.Index,
    mode: str,
) -> pd.DataFrame:
    weights = pd.DataFrame(0.0, index=index, columns=strategies, dtype=float)
    for strategy in strategies:
        weight_log_delta = pd.to_numeric(
            actions.get(f"{strategy}__weight_log_delta", pd.Series(0.0, index=index)),
            errors="coerce",
        ).reindex(index).fillna(0.0)
        gate = pd.to_numeric(
            actions.get(f"{strategy}__activation_gate", pd.Series(1.0, index=index)),
            errors="coerce",
        ).reindex(index).fillna(1.0)
        raw = np.exp(np.clip(weight_log_delta, -8.0, 4.0))
        if mode == "weight_only":
            multiplier = pd.Series(1.0, index=index)
        elif mode == "soft_gate_sigmoid_scale_1":
            multiplier = 1.0 / (1.0 + np.exp(-np.clip(gate, -50.0, 50.0)))
        elif mode == "soft_gate_sigmoid_scale_2":
            multiplier = 1.0 / (1.0 + np.exp(-np.clip(gate / 2.0, -50.0, 50.0)))
        elif mode == "hard_gate_cutoff_-2":
            multiplier = (gate >= -2.0).astype(float)
        elif mode == "hard_gate_cutoff_-1":
            multiplier = (gate >= -1.0).astype(float)
        else:
            multiplier = (gate >= 0.0).astype(float)
        weights[strategy] = np.asarray(raw, dtype=float) * np.asarray(multiplier, dtype=float)
    return weights


def _reconstruct_modulation_scores(
    fold_root: Path,
    experts,
    calibrator,
) -> pd.DataFrame:
    p_active = (
        experts.scores.replace([np.inf, -np.inf], np.nan)
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .clip(0.0, 1.0)
        .astype(np.float32, copy=False)
    )
    if p_active.empty:
        return p_active
    diagnostics = _load_run_table(
        fold_root / "models" / "portfolio_calibrator" / "archetype_modulation_threshold_diagnostics.parquet"
    )
    threshold_map: dict[str, float] = {}
    if not diagnostics.empty and {"archetype_id", "effective_min_p_active"}.issubset(diagnostics.columns):
        threshold_map = (
            diagnostics.assign(
                archetype_id=lambda frame: frame["archetype_id"].astype(str),
                effective_min_p_active=lambda frame: pd.to_numeric(
                    frame["effective_min_p_active"],
                    errors="coerce",
                ),
            )
            .dropna(subset=["effective_min_p_active"])
            .set_index("archetype_id")["effective_min_p_active"]
            .astype(float)
            .to_dict()
        )
    default_threshold = float(np.clip(calibrator.config.archetype_base_p_active_floor, 0.0, 1.0))
    columns: dict[str, pd.Series] = {}
    for column in p_active.columns:
        threshold = float(np.clip(threshold_map.get(str(column), default_threshold), 0.0, 1.0))
        columns[str(column)] = ((p_active[column] - threshold) / max(1.0 - threshold, 1e-6)).clip(0.0, 1.0)
    return pd.DataFrame(columns, index=p_active.index).astype(np.float32, copy=False)


def build_replay_report(
    runs: dict[str, Path],
    *,
    input_path: Path,
    strategy_col: str,
    performance_col: str,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    returns_all = _performance_pivot(input_path, strategy_col, performance_col)
    rows: list[dict[str, Any]] = []
    action_rows: list[pd.DataFrame] = []
    exposure_rows: list[dict[str, Any]] = []
    for run_name, root in runs.items():
        for scope, fold_root in _iter_fold_roots(root):
            fold = int(fold_root.name.split("_")[-1])
            calibrator_path = fold_root / "models" / "portfolio_calibrator" / "portfolio_calibrator.joblib"
            expert_path = fold_root / "models" / "archetype_experts" / "archetype_expert_bundle.joblib"
            cross_path = fold_root / "features" / "cross_strategy_archetype_features.parquet"
            head_streak_path = fold_root / "features" / "head_streak_risk_features.parquet"
            if not calibrator_path.exists() or not expert_path.exists() or not cross_path.exists():
                continue
            calibrator = joblib.load(calibrator_path)
            experts = joblib.load(expert_path)
            cross = _timestamp_indexed(pd.read_parquet(cross_path))
            modulation_scores = _reconstruct_modulation_scores(fold_root, experts, calibrator)
            feature_parts = [modulation_scores.fillna(0.0), cross.fillna(0.0)]
            if head_streak_path.exists():
                feature_parts.append(_timestamp_indexed(pd.read_parquet(head_streak_path)).fillna(0.0))
            X = pd.concat(feature_parts, axis=1)
            X = X.reindex(columns=calibrator.feature_columns).fillna(0.0)
            actions = score_frozen_portfolio_calibrator(X, calibrator)
            strategies = list(calibrator.strategies)
            returns = returns_all.reindex(index=X.index, columns=strategies)
            available = returns.notna()
            baseline = available.astype(float)
            baseline = _normalize_available_weights(baseline, returns)
            variant_weights = {
                "baseline_equal_available": baseline,
                "hard_gate_cutoff_0": _weights_from_actions(
                    actions,
                    strategies=strategies,
                    index=X.index,
                    mode="hard_gate_cutoff_0",
                ),
                "hard_gate_cutoff_-1": _weights_from_actions(
                    actions,
                    strategies=strategies,
                    index=X.index,
                    mode="hard_gate_cutoff_-1",
                ),
                "hard_gate_cutoff_-2": _weights_from_actions(
                    actions,
                    strategies=strategies,
                    index=X.index,
                    mode="hard_gate_cutoff_-2",
                ),
                "soft_gate_sigmoid_scale_1": _weights_from_actions(
                    actions,
                    strategies=strategies,
                    index=X.index,
                    mode="soft_gate_sigmoid_scale_1",
                ),
                "soft_gate_sigmoid_scale_2": _weights_from_actions(
                    actions,
                    strategies=strategies,
                    index=X.index,
                    mode="soft_gate_sigmoid_scale_2",
                ),
                "weight_only": _weights_from_actions(
                    actions,
                    strategies=strategies,
                    index=X.index,
                    mode="weight_only",
                ),
            }
            variant_weights = {
                policy: _normalize_available_weights(weights, returns)
                for policy, weights in variant_weights.items()
            }
            for policy, weights in variant_weights.items():
                metrics = _portfolio_metrics(weights, returns, policy=policy)
                metrics.update({"run": run_name, "scope": scope, "fold": fold})
                rows.append(metrics)
                exposure_rows.extend(
                    _bad_regime_exposure_rows(
                        run_name=run_name,
                        scope=scope,
                        fold=fold,
                        policy=policy,
                        weights=weights,
                        baseline_weights=baseline,
                        returns=returns,
                    )
                )
            base_pnl = (baseline * returns.fillna(0.0)).sum(axis=1)
            for policy, weights in variant_weights.items():
                if policy == "baseline_equal_available":
                    continue
                mod_pnl = (weights * returns.fillna(0.0)).sum(axis=1)
                rows.append(
                    {
                        "run": run_name,
                        "scope": scope,
                        "fold": fold,
                        "policy": f"{policy}_minus_baseline",
                        "timestamp_count": int(len(X)),
                        "total_return_sum": float(mod_pnl.sum() - base_pnl.sum()),
                        "mean_return": float(mod_pnl.mean() - base_pnl.mean()),
                        "std_return": float((mod_pnl - base_pnl).std(ddof=0)),
                        "sharpe_like": np.nan,
                        "hit_rate": float((mod_pnl > base_pnl).mean()),
                        "loss_rate": float((mod_pnl < base_pnl).mean()),
                        "max_drawdown_abs": np.nan,
                        "cash_share": float((weights.abs().sum(axis=1) <= 1e-12).mean()),
                        "avg_active_strategies": float((weights.abs() > 1e-12).sum(axis=1).mean()),
                        "turnover_mean": float(weights.diff().abs().sum(axis=1).fillna(0.0).mean()),
                        "turnover_sum": float(weights.diff().abs().sum(axis=1).fillna(0.0).sum()),
                    }
                )
            summary = actions.agg(["mean", "std", "min", "max"]).T.reset_index().rename(columns={"index": "action"})
            summary["run"] = run_name
            summary["scope"] = scope
            summary["fold"] = fold
            action_rows.append(summary)
    replay = pd.DataFrame(rows)
    actions_summary = pd.concat(action_rows, ignore_index=True) if action_rows else pd.DataFrame()
    exposure = pd.DataFrame(exposure_rows)
    _write_frame(output_dir / "portfolio_replay_metrics.parquet", replay)
    _write_frame(output_dir / "portfolio_action_prediction_summary.parquet", actions_summary)
    _write_frame(output_dir / "bad_regime_exposure_report.parquet", exposure)
    return replay, actions_summary


def build_promotion_report(
    comparison: pd.DataFrame,
    replay: pd.DataFrame,
    *,
    output_dir: Path,
) -> dict[str, Any]:
    gate_failures = (
        comparison.loc[comparison["metric"].eq("gate_failures")]
        .set_index("run")["value"]
        .to_dict()
        if not comparison.empty
        else {}
    )
    mod_delta = replay.loc[replay["policy"].astype(str).str.endswith("_minus_baseline")].copy()
    if not mod_delta.empty:
        mod_delta["policy_variant"] = mod_delta["policy"].astype(str).str.replace(
            "_minus_baseline",
            "",
            regex=False,
        )
    replay_summary = (
        mod_delta.groupby(["run", "policy_variant"])
        .agg(
            mean_total_return_delta=("total_return_sum", "mean"),
            min_total_return_delta=("total_return_sum", "min"),
            mean_hit_rate_vs_baseline=("hit_rate", "mean"),
            mean_cash_share=("cash_share", "mean"),
            mean_active_strategies=("avg_active_strategies", "mean"),
        )
        .reset_index()
        if not mod_delta.empty
        else pd.DataFrame()
    )
    expert_share = (
        comparison.loc[comparison["metric"].eq("predictive_expert_fold_share")]
        .groupby("run")["value"]
        .min()
        .to_dict()
    )
    archetype_count = (
        comparison.loc[comparison["metric"].eq("archetype_count")]
        .groupby("run")["value"]
        .max()
        .to_dict()
    )
    candidates = []
    for row in replay_summary.itertuples(index=False):
        run_name = str(row.run)
        policy_variant = str(row.policy_variant)
        candidates.append(
            {
                "run": run_name,
                "policy_variant": policy_variant,
                "max_archetypes": int(archetype_count.get(run_name, _cap_from_name(run_name) or -1)),
                "gate_failures": int(gate_failures.get(run_name, 999)),
                "min_predictive_expert_fold_share": float(expert_share.get(run_name, np.nan)),
                "mean_total_return_delta": float(row.mean_total_return_delta),
                "min_total_return_delta": float(row.min_total_return_delta),
                "mean_hit_rate_vs_baseline": float(row.mean_hit_rate_vs_baseline),
                "mean_cash_share": float(row.mean_cash_share),
                "mean_active_strategies": float(row.mean_active_strategies),
            }
        )
    candidate_frame = pd.DataFrame(candidates)
    if candidate_frame.empty:
        decision = {
            "promoted_run": None,
            "reason": "No replay candidates available.",
            "candidates": [],
        }
    else:
        eligible = candidate_frame.loc[
            (candidate_frame["gate_failures"] == 0)
            & (candidate_frame["min_predictive_expert_fold_share"] >= 0.25)
            & (candidate_frame["mean_total_return_delta"] >= 0.0)
            & (candidate_frame["mean_hit_rate_vs_baseline"] >= 0.50)
        ].copy()
        if eligible.empty:
            best = candidate_frame.sort_values(
                ["gate_failures", "mean_total_return_delta", "min_predictive_expert_fold_share"],
                ascending=[True, False, False],
            ).iloc[0]
            decision = {
                "promoted_run": None,
                "reason": (
                    "No run satisfied all promotion checks. Best diagnostic candidate was "
                    f"{best['run']} / {best['policy_variant']} but promotion requires zero gates, "
                    "predictive share >= 0.25, non-negative mean replay delta, and hit-rate versus baseline >= 0.50."
                ),
                "best_diagnostic_run": str(best["run"]),
                "best_diagnostic_policy_variant": str(best["policy_variant"]),
                "candidates": candidate_frame.to_dict("records"),
            }
        else:
            eligible = eligible.sort_values(
                ["max_archetypes", "mean_total_return_delta", "min_predictive_expert_fold_share"],
                ascending=[True, False, False],
            )
            selected = eligible.iloc[0]
            decision = {
                "promoted_run": str(selected["run"]),
                "promoted_policy_variant": str(selected["policy_variant"]),
                "reason": "Smallest archetype cap satisfying gate, expert, and replay checks.",
                "recommended_max_archetypes_for_experts": int(selected["max_archetypes"]),
                "candidates": candidate_frame.to_dict("records"),
            }
    _write_frame(output_dir / "candidate_promotion_candidates.parquet", candidate_frame)
    (output_dir / "candidate_promotion.json").write_text(json.dumps(_json_safe(decision), indent=2))
    return decision


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--strategy-col", default="head")
    parser.add_argument("--performance-col", default="net_return")
    parser.add_argument("--run", action="append", required=True, help="NAME=PATH")
    parser.add_argument("--top-archetypes", type=int, default=8)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    runs = dict(_parse_run(value) for value in args.run)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    comparison = build_comparison(runs, args.output_dir)
    build_archetype_inspection(
        runs,
        input_path=args.input,
        strategy_col=args.strategy_col,
        performance_col=args.performance_col,
        output_dir=args.output_dir,
        top_n=int(args.top_archetypes),
    )
    replay, _actions = build_replay_report(
        runs,
        input_path=args.input,
        strategy_col=args.strategy_col,
        performance_col=args.performance_col,
        output_dir=args.output_dir,
    )
    decision = build_promotion_report(comparison, replay, output_dir=args.output_dir)
    print(json.dumps(_json_safe(decision), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
