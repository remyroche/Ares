#!/usr/bin/env python3
"""Tune bounded geometry rank nudges on historical OOS, then test on July."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from numba import njit
except Exception:  # pragma: no cover
    njit = None


ARMS = (
    "cross_sectional_geometry",
    "joint_state_geometry",
    "joint_state_geometry_breakout_weighted",
    "joint_state_geometry_breakout_day_balanced",
)
ALPHAS = (0.10, 0.20, 0.30)
CAPS = (0.10, 0.20)
PROMOTION_RATIOS = (0.50,)
SCOPES = ("all", "short_breakout")
KEY_COLUMNS = ["__ts__", "__symbol__", "side_name", "archetype_policy_key"]


def _top10_mask_python(scores: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    output = np.zeros(len(scores), dtype=np.bool_)
    for group in range(len(offsets) - 1):
        start, end = int(offsets[group]), int(offsets[group + 1])
        size = end - start
        if size <= 0:
            continue
        first_rank = int(np.ceil(0.90 * size))
        count = max(1, size - first_rank + 1)
        local = np.argpartition(scores[start:end], size - count)[size - count :]
        output[start + local] = True
    return output


if njit is not None:
    _top10_mask = njit(cache=True)(_top10_mask_python)
else:  # pragma: no cover
    _top10_mask = _top10_mask_python


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _prepare(frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    out = frame.sort_values(
        ["__ts__", "__symbol__", "side_name"], kind="stable"
    ).reset_index(drop=True)
    out["__ts__"] = pd.to_datetime(out["__ts__"], utc=True, errors="coerce")
    out["day"] = out["__ts__"].dt.floor("D")
    out["week"] = out["day"] - pd.to_timedelta(out["day"].dt.weekday, unit="D")
    out["month"] = (
        out["calendar_month"].astype(str)
        if "calendar_month" in out.columns
        else out["__ts__"].dt.to_period("M").astype(str)
    )
    starts = np.flatnonzero(
        np.r_[True, out["__ts__"].to_numpy()[1:] != out["__ts__"].to_numpy()[:-1]]
    ).astype(np.int64)
    offsets = np.r_[starts, len(out)].astype(np.int64)
    return out, offsets


def _scope_mask(frame: pd.DataFrame, scope: str) -> np.ndarray:
    if scope == "all":
        return np.ones(len(frame), dtype=bool)
    return (
        frame["side_name"].astype(str).str.lower().eq("short")
        & frame["archetype_policy_key"]
        .astype(str)
        .str.contains("breakout", case=False, na=False)
    ).to_numpy(dtype=bool)


def _adjusted_scores(
    base: np.ndarray,
    overlay: np.ndarray,
    apply: np.ndarray,
    *,
    mode: str,
    alpha: float,
    cap: float,
    promotion_ratio: float,
    scope: str,
) -> np.ndarray:
    delta = np.clip(overlay - 0.5, -float(cap), float(cap)).astype(np.float32)
    if mode == "risk_only":
        delta = np.minimum(delta, 0.0)
    elif mode == "asymmetric":
        delta = np.where(delta > 0.0, delta * float(promotion_ratio), delta)
    elif mode != "symmetric":
        raise ValueError(f"Unknown mode: {mode}")
    delta = np.where(apply, delta, 0.0)
    return np.clip(base + float(alpha) * delta, 0.0, 1.0).astype(np.float32)


def _metric_context(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        "ev": pd.to_numeric(frame["ev_after_1pct"], errors="coerce").to_numpy(
            dtype=np.float32
        ),
        "clean": pd.to_numeric(frame["clean_exec"], errors="coerce").to_numpy(
            dtype=np.float32
        ),
        "day": pd.factorize(frame["day"], sort=True)[0].astype(np.int32),
        "week": pd.factorize(frame["week"], sort=True)[0].astype(np.int32),
        "month": pd.factorize(frame["month"], sort=True)[0].astype(np.int32),
        "breakout": (
            frame["side_name"].astype(str).str.lower().eq("short")
            & frame["archetype_policy_key"]
            .astype(str)
            .str.contains("breakout", case=False, na=False)
        ).to_numpy(dtype=bool),
    }


def _group_means(codes: np.ndarray, values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    local_codes = codes[mask]
    if len(local_codes) == 0:
        return np.asarray([], dtype=np.float32)
    size = int(local_codes.max()) + 1
    counts = np.bincount(local_codes, minlength=size)
    sums = np.bincount(local_codes, weights=values[mask], minlength=size)
    present = counts > 0
    return (sums[present] / counts[present]).astype(np.float32)


def _metrics(context: dict[str, np.ndarray], selected: np.ndarray) -> dict[str, Any]:
    ev = context["ev"]
    valid = selected & np.isfinite(ev)
    selected_ev = ev[valid]
    daily = _group_means(context["day"], ev, valid)
    weekly = _group_means(context["week"], ev, valid)
    monthly = _group_means(context["month"], ev, valid)
    breakout_ev = ev[valid & context["breakout"]]
    mean_ev = float(np.mean(selected_ev))
    return {
        "selected_rows": int(valid.sum()),
        "mean_ev": mean_ev,
        "clean": float(np.nanmean(context["clean"][valid])),
        "worst_day_ev": float(np.min(daily)),
        "daily_std": float(np.std(daily)),
        "positive_days": int(np.sum(daily > 0.0)),
        "days": int(len(daily)),
        "worst_week_ev": float(np.min(weekly)),
        "positive_weeks": int(np.sum(weekly > 0.0)),
        "weeks": int(len(weekly)),
        "mean_month_ev": float(np.mean(monthly)),
        "month_std": float(np.std(monthly)),
        "worst_month_ev": float(np.min(monthly)),
        "breakout_ev": float(np.mean(breakout_ev)) if len(breakout_ev) else np.nan,
        "breakout_rows": int(len(breakout_ev)),
        "objective": float(
            np.mean(monthly)
            - 0.50 * np.std(monthly)
            + 0.25 * np.min(monthly)
            + 0.10 * np.min(daily)
            + 0.05 * np.mean(breakout_ev)
        ),
    }


def _daily(frame: pd.DataFrame, selected: np.ndarray, label: str) -> pd.DataFrame:
    part = frame.loc[
        selected & frame["ev_after_1pct"].notna().to_numpy(dtype=bool)
    ].copy()
    out = (
        part.groupby("day", observed=True)
        .agg(
            selected_rows=("ev_after_1pct", "size"),
            mean_ev=("ev_after_1pct", "mean"),
            clean=("clean_exec", "mean"),
            bad_mae=("full_path_bad_mae_1r", "mean"),
            timeout=("timeout", "mean"),
        )
        .reset_index()
    )
    out["selector"] = label
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ablation-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    search_rows: list[dict[str, Any]] = []
    july_rows: list[dict[str, Any]] = []
    daily_parts: list[pd.DataFrame] = []
    selected_configs: list[dict[str, Any]] = []
    for arm in ARMS:
        source = pd.read_parquet(
            args.ablation_dir / f"{arm}_predictions.parquet",
            columns=[
                "__ts__",
                "__symbol__",
                "side_name",
                "archetype_policy_key",
                "calendar_month",
                "evaluation_scope",
                "base_batch_rank",
                "overlay_rank",
                "ev_after_1pct",
                "clean_exec",
                "full_path_bad_mae_1r",
                "timeout",
            ],
        )
        historical, hist_offsets = _prepare(
            source[source["evaluation_scope"].eq("historical_walkforward_oos")]
        )
        july, july_offsets = _prepare(source[source["evaluation_scope"].eq("july_oos")])
        historical_context = _metric_context(historical)
        july_context = _metric_context(july)
        historical_base = historical["base_batch_rank"].to_numpy(dtype=np.float32)
        historical_overlay = historical["overlay_rank"].to_numpy(dtype=np.float32)
        july_base = july["base_batch_rank"].to_numpy(dtype=np.float32)
        july_overlay = july["overlay_rank"].to_numpy(dtype=np.float32)
        historical_scopes = {scope: _scope_mask(historical, scope) for scope in SCOPES}
        july_scopes = {scope: _scope_mask(july, scope) for scope in SCOPES}
        baseline_hist = _top10_mask(
            historical["base_batch_rank"].to_numpy(dtype=np.float32), hist_offsets
        )
        baseline_july = _top10_mask(
            july["base_batch_rank"].to_numpy(dtype=np.float32), july_offsets
        )
        baseline_hist_metrics = _metrics(historical_context, baseline_hist)
        baseline_july_metrics = _metrics(july_context, baseline_july)
        daily_parts.extend(
            [
                _daily(historical, baseline_hist, f"{arm}__baseline_historical"),
                _daily(july, baseline_july, f"{arm}__baseline_july"),
            ]
        )
        configs: list[dict[str, Any]] = []
        scopes = SCOPES if "breakout_" in arm else ("all",)
        for mode in ("symmetric", "risk_only", "asymmetric"):
            ratios = PROMOTION_RATIOS if mode == "asymmetric" else (1.0,)
            for alpha in ALPHAS:
                for cap in CAPS:
                    for ratio in ratios:
                        for scope in scopes:
                            scores = _adjusted_scores(
                                historical_base,
                                historical_overlay,
                                historical_scopes[scope],
                                mode=mode,
                                alpha=alpha,
                                cap=cap,
                                promotion_ratio=ratio,
                                scope=scope,
                            )
                            selected = _top10_mask(scores, hist_offsets)
                            metrics = _metrics(historical_context, selected)
                            row = {
                                "arm": arm,
                                "mode": mode,
                                "alpha": alpha,
                                "cap": cap,
                                "promotion_ratio": ratio,
                                "scope": scope,
                                **metrics,
                                "mean_ev_delta": metrics["mean_ev"]
                                - baseline_hist_metrics["mean_ev"],
                                "worst_day_delta": metrics["worst_day_ev"]
                                - baseline_hist_metrics["worst_day_ev"],
                                "worst_week_delta": metrics["worst_week_ev"]
                                - baseline_hist_metrics["worst_week_ev"],
                                "breakout_ev_delta": metrics["breakout_ev"]
                                - baseline_hist_metrics["breakout_ev"],
                            }
                            configs.append(row)
                            search_rows.append(row)
        search = pd.DataFrame(configs)
        safe = search.loc[
            search["mean_ev"].ge(baseline_hist_metrics["mean_ev"])
            & search["worst_day_ev"].ge(baseline_hist_metrics["worst_day_ev"] - 0.0010)
            & search["worst_week_ev"].ge(
                baseline_hist_metrics["worst_week_ev"] - 0.0005
            )
        ]
        if safe.empty:
            safe = search
        best = (
            safe.sort_values(
                ["objective", "mean_ev", "worst_day_ev"], ascending=False, kind="stable"
            )
            .iloc[0]
            .to_dict()
        )
        selected_configs.append(best)
        july_scores = _adjusted_scores(
            july_base,
            july_overlay,
            july_scopes[str(best["scope"])],
            mode=str(best["mode"]),
            alpha=float(best["alpha"]),
            cap=float(best["cap"]),
            promotion_ratio=float(best["promotion_ratio"]),
            scope=str(best["scope"]),
        )
        july_selected = _top10_mask(july_scores, july_offsets)
        july_metrics = _metrics(july_context, july_selected)
        july_rows.append(
            {
                "arm": arm,
                "selector": "baseline",
                **baseline_july_metrics,
            }
        )
        july_rows.append(
            {
                "arm": arm,
                "selector": "selected_nudge",
                "mode": best["mode"],
                "alpha": best["alpha"],
                "cap": best["cap"],
                "promotion_ratio": best["promotion_ratio"],
                "scope": best["scope"],
                **july_metrics,
                "mean_ev_delta": july_metrics["mean_ev"]
                - baseline_july_metrics["mean_ev"],
                "worst_day_delta": july_metrics["worst_day_ev"]
                - baseline_july_metrics["worst_day_ev"],
                "worst_week_delta": july_metrics["worst_week_ev"]
                - baseline_july_metrics["worst_week_ev"],
                "breakout_ev_delta": july_metrics["breakout_ev"]
                - baseline_july_metrics["breakout_ev"],
            }
        )
        daily_parts.append(_daily(july, july_selected, f"{arm}__selected_nudge_july"))

    # Compose the general geometry nudge with a dedicated short-breakout local
    # nudge. Every parameter below is selected on historical OOS predictions;
    # July is evaluated only after the historical choice is frozen.
    columns = [
        "__ts__",
        "__symbol__",
        "side_name",
        "archetype_policy_key",
        "calendar_month",
        "evaluation_scope",
        "base_batch_rank",
        "overlay_rank",
        "ev_after_1pct",
        "clean_exec",
        "full_path_bad_mae_1r",
        "timeout",
    ]
    cross_source = pd.read_parquet(
        args.ablation_dir / "cross_sectional_geometry_predictions.parquet",
        columns=columns,
    )
    breakout_source = pd.read_parquet(
        args.ablation_dir
        / "joint_state_geometry_breakout_weighted_predictions.parquet",
        columns=columns,
    )
    composite_rows: list[dict[str, Any]] = []
    composite_predictions: list[pd.DataFrame] = []
    chosen_composite: dict[str, Any] | None = None
    for evaluation_scope in ("historical_walkforward_oos", "july_oos"):
        cross, offsets = _prepare(
            cross_source[cross_source["evaluation_scope"].eq(evaluation_scope)]
        )
        breakout, _ = _prepare(
            breakout_source[breakout_source["evaluation_scope"].eq(evaluation_scope)]
        )
        if not cross[KEY_COLUMNS].equals(breakout[KEY_COLUMNS]):
            raise RuntimeError("Composite overlay rows are not aligned")
        context = _metric_context(cross)
        base = cross["base_batch_rank"].to_numpy(dtype=np.float32)
        is_breakout = _scope_mask(cross, "short_breakout")
        cross_delta = np.clip(
            cross["overlay_rank"].to_numpy(dtype=np.float32) - 0.5, -0.20, 0.20
        )
        breakout_delta = np.clip(
            breakout["overlay_rank"].to_numpy(dtype=np.float32) - 0.5, -0.20, 0.20
        )
        breakout_delta = np.where(
            breakout_delta > 0.0, 0.50 * breakout_delta, breakout_delta
        )
        baseline_selected = _top10_mask(base, offsets)
        baseline_metrics = _metrics(context, baseline_selected)
        if evaluation_scope == "historical_walkforward_oos":
            candidates: list[dict[str, Any]] = []
            for breakout_alpha in (0.10, 0.20, 0.30, 0.40, 0.50):
                score = np.clip(
                    base
                    + 0.20 * np.where(~is_breakout, cross_delta, 0.0)
                    + float(breakout_alpha)
                    * np.where(is_breakout, breakout_delta, 0.0),
                    0.0,
                    1.0,
                ).astype(np.float32)
                selected = _top10_mask(score, offsets)
                metrics = _metrics(context, selected)
                candidates.append(
                    {
                        "breakout_alpha": breakout_alpha,
                        **metrics,
                        "mean_ev_delta": metrics["mean_ev"]
                        - baseline_metrics["mean_ev"],
                        "worst_day_delta": metrics["worst_day_ev"]
                        - baseline_metrics["worst_day_ev"],
                        "worst_week_delta": metrics["worst_week_ev"]
                        - baseline_metrics["worst_week_ev"],
                        "breakout_ev_delta": metrics["breakout_ev"]
                        - baseline_metrics["breakout_ev"],
                    }
                )
            candidate_frame = pd.DataFrame(candidates)
            safe = candidate_frame.loc[
                candidate_frame["mean_ev_delta"].ge(0.0)
                & candidate_frame["worst_day_delta"].ge(0.0)
                & candidate_frame["worst_week_delta"].ge(0.0)
                & candidate_frame["breakout_ev_delta"].ge(0.0)
            ]
            if safe.empty:
                safe = candidate_frame
            chosen_composite = (
                safe.sort_values(
                    ["objective", "mean_ev", "worst_day_ev"],
                    ascending=False,
                    kind="stable",
                )
                .iloc[0]
                .to_dict()
            )
            candidate_frame.to_csv(
                args.output_dir / "balanced_composite_historical_search.csv",
                index=False,
            )
        if chosen_composite is None:
            raise RuntimeError(
                "Historical composite selection did not produce a configuration"
            )
        breakout_alpha = float(chosen_composite["breakout_alpha"])
        score = np.clip(
            base
            + 0.20 * np.where(~is_breakout, cross_delta, 0.0)
            + breakout_alpha * np.where(is_breakout, breakout_delta, 0.0),
            0.0,
            1.0,
        ).astype(np.float32)
        selected = _top10_mask(score, offsets)
        metrics = _metrics(context, selected)
        composite_rows.extend(
            [
                {
                    "evaluation_scope": evaluation_scope,
                    "selector": "baseline",
                    **baseline_metrics,
                },
                {
                    "evaluation_scope": evaluation_scope,
                    "selector": "balanced_composite_v1",
                    "global_alpha": 0.20,
                    "global_cap": 0.20,
                    "breakout_alpha": breakout_alpha,
                    "breakout_cap": 0.20,
                    "breakout_promotion_ratio": 0.50,
                    **metrics,
                    "mean_ev_delta": metrics["mean_ev"] - baseline_metrics["mean_ev"],
                    "worst_day_delta": metrics["worst_day_ev"]
                    - baseline_metrics["worst_day_ev"],
                    "worst_week_delta": metrics["worst_week_ev"]
                    - baseline_metrics["worst_week_ev"],
                    "breakout_ev_delta": metrics["breakout_ev"]
                    - baseline_metrics["breakout_ev"],
                },
            ]
        )
        daily_parts.append(
            _daily(cross, selected, f"balanced_composite_v1__{evaluation_scope}")
        )
        prediction = cross[KEY_COLUMNS + ["calendar_month", "evaluation_scope"]].copy()
        prediction["score_balanced_composite"] = score
        prediction["selected_top10_balanced_composite"] = selected
        composite_predictions.append(prediction)

    search_frame = pd.DataFrame(search_rows)
    july_frame = pd.DataFrame(july_rows)
    selected_frame = pd.DataFrame(selected_configs)
    search_frame.to_csv(
        args.output_dir / "historical_oos_nudge_search.csv", index=False
    )
    selected_frame.to_csv(args.output_dir / "selected_configs.csv", index=False)
    july_frame.to_csv(args.output_dir / "july_untouched_scorecard.csv", index=False)
    pd.concat(daily_parts, ignore_index=True).to_csv(
        args.output_dir / "daily_metrics.csv", index=False
    )
    pd.DataFrame(composite_rows).to_csv(
        args.output_dir / "balanced_composite_scorecard.csv", index=False
    )
    pd.concat(composite_predictions, ignore_index=True).to_parquet(
        args.output_dir / "balanced_composite_predictions.parquet",
        index=False,
        compression="zstd",
    )
    manifest = {
        "schema": "meta_geometry_rank_nudge_tuning_v1",
        "historical_role": "OOS model predictions used only to select bounded nudge parameters",
        "july_role": "untouched final evaluation; never used in nudge selection",
        "activity_contract": "same global within-timestamp top-10 count for every configuration",
        "cost_contract": "ev_after_1pct includes 1% round-trip cost",
        "safe_selection_constraints": {
            "mean_ev": "not below baseline",
            "worst_day_tolerance": -0.001,
            "worst_week_tolerance": -0.0005,
        },
        "numba_topk": bool(njit is not None),
        "selected_configs": selected_configs,
        "balanced_composite": chosen_composite,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(_json_safe(manifest), indent=2, sort_keys=True), encoding="utf-8"
    )
    print("Selected configurations:")
    print(selected_frame.to_string(index=False))
    print("\nUntouched July:")
    print(july_frame.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
