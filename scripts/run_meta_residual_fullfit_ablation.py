#!/usr/bin/env python3
"""Full-fit residual-expert ablation with causal recency and local AE/GMM states.

This runner deliberately does *not* rerun feature selection or LightGBM HPO.
It reuses the validated residual-expert selection/HPO contract, then compares
full rolling fits using only:

* training-sample half life;
* a conservative scale-up of the HPO leaf floor; and
* frozen side-local base AE/GMM state outputs as an explicit meta context arm.

All OOS scores remain based on train-only EV maps and models.  The side-local
states are pre-entry transforms only; their output columns are local to their
side and cannot be compared as raw cluster IDs across sides.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.side_local_ae_gmm_search import SideLocalState  # noqa: E402
from scripts import run_meta_v9_ev_mapped_side_residual_ablation as residual  # noqa: E402


@dataclass(frozen=True)
class Trial:
    stage: str
    half_life_months: float
    leaf_alpha: float
    state_mode: str

    @property
    def name(self) -> str:
        return (
            f"{self.stage}__hl{self.half_life_months:g}m__"
            f"alpha{self.leaf_alpha:.1f}__{self.state_mode}"
        )


def _parse_csv_floats(raw: str) -> list[float]:
    values = [float(value.strip()) for value in raw.split(",") if value.strip()]
    if not values or any(value <= 0.0 for value in values):
        raise ValueError("expected non-empty positive comma-separated floats")
    return values


def _load_state(path: Path) -> SideLocalState:
    state = joblib.load(path)
    if not isinstance(state, SideLocalState):
        raise TypeError(f"Expected SideLocalState in {path}, found {type(state)!r}")
    if state.config.layer != "base":
        raise ValueError(f"Expected a base-layer state in {path}")
    return state


def _state_context_features(states: dict[str, SideLocalState]) -> list[str]:
    return list(
        dict.fromkeys(
            feature
            for state in states.values()
            for feature in state.feature_names
        )
    )


def _append_state_context(
    frame: pd.DataFrame,
    states: dict[str, SideLocalState],
    mode: str,
) -> pd.DataFrame:
    active: list[SideLocalState] = []
    if mode in {"side_local", "long_local"}:
        active.append(states["long"])
    if mode in {"side_local", "short_local"}:
        active.append(states["short"])
    if not active:
        return frame
    parts = [frame]
    for state in active:
        block = state.transform(frame)
        # Cluster ID is local and diagnostic. Continuous posterior/distance,
        # entropy, reconstruction and latent coordinates are the usable model
        # context. The validity flag is retained for missingness awareness.
        drop = f"{state.prefix}_component_id_local"
        if drop in block:
            block = block.drop(columns=[drop])
        parts.append(block.astype(np.float32, copy=False))
    return pd.concat(parts, axis=1, copy=False)


def _state_output_features(states: dict[str, SideLocalState], mode: str) -> dict[str, list[str]]:
    output: dict[str, list[str]] = {"long": [], "short": []}
    if mode in {"side_local", "long_local"}:
        state = states["long"]
        output["long"] = [
            name for name in state.manifest()["output_columns"]
            if not name.endswith("_component_id_local")
        ]
    if mode in {"side_local", "short_local"}:
        state = states["short"]
        output["short"] = [
            name for name in state.manifest()["output_columns"]
            if not name.endswith("_component_id_local")
        ]
    return output


def _rolling_train(frame: pd.DataFrame, cutoff: pd.Timestamp, max_days: int) -> pd.DataFrame:
    train = residual._resolved_train_before(frame, cutoff)
    start = cutoff - pd.Timedelta(days=int(max_days))
    return train.loc[train["__ts__"].ge(start)].copy()


def _recency_tail_weights(
    frame: pd.DataFrame,
    raw: np.ndarray,
    half_life_months: float,
) -> np.ndarray:
    finite = np.isfinite(raw)
    q80, q90 = np.quantile(raw[finite], [0.80, 0.90])
    tail = np.where(raw >= q90, 3.0, np.where(raw >= q80, 1.5, 0.5))
    ts = pd.to_datetime(frame["__ts__"], utc=True, errors="coerce")
    latest = ts.max()
    age_days = (latest - ts).dt.total_seconds().to_numpy(np.float64) / 86_400.0
    decay_days = max(float(half_life_months) * 30.4375, 1.0)
    recency = np.exp2(-np.maximum(age_days, 0.0) / decay_days)
    weight = tail * recency
    weight /= max(float(np.mean(weight)), 1e-12)
    return weight.astype(np.float32)


def _scaled_params(
    params_by_side: dict[str, dict[str, Any]],
    side: str,
    n_side_rows: int,
    hpo_rows: int,
    alpha: float,
    seed: int,
) -> tuple[dict[str, Any], int, int]:
    params, rounds = residual._model_params(seed, params_by_side[side])
    base_leaf = int(params.get("min_data_in_leaf", 500))
    multiplier = max(1.0, (float(n_side_rows) / max(float(hpo_rows), 1.0)) ** float(alpha))
    scaled_leaf = int(math.ceil(base_leaf * multiplier))
    params["min_data_in_leaf"] = scaled_leaf
    return params, rounds, scaled_leaf


def _fit_models(
    train: pd.DataFrame,
    features_by_side: dict[str, list[str]],
    params_by_side: dict[str, dict[str, Any]],
    *,
    half_life_months: float,
    leaf_alpha: float,
    hpo_rows: int,
    seed: int,
) -> tuple[Any, dict[str, lgb.Booster], dict[str, int]]:
    ev_map = residual._fit_ev_map(train, "score_base")
    raw = pd.to_numeric(train["score_base"], errors="coerce").to_numpy(np.float32)
    expected = residual.predict_hierarchical_ev(ev_map, train, raw)
    realized = pd.to_numeric(train["ev_after_1pct"], errors="coerce").to_numpy(np.float32)
    target = realized - expected
    sides = train["side_name"].astype(str).to_numpy()
    models: dict[str, lgb.Booster] = {}
    leaf_sizes: dict[str, int] = {}
    for offset, side in enumerate(("long", "short"), start=1):
        mask = (sides == side) & np.isfinite(target) & np.isfinite(raw)
        if int(mask.sum()) < 5_000:
            raise RuntimeError(f"Insufficient {side} rows for full fit: {int(mask.sum())}")
        params, rounds, leaf = _scaled_params(
            params_by_side, side, int(mask.sum()), hpo_rows, leaf_alpha, seed + offset
        )
        weights = _recency_tail_weights(train.loc[mask], raw[mask], half_life_months)
        dataset = lgb.Dataset(
            residual._matrix(train.loc[mask], features_by_side[side]),
            label=target[mask],
            weight=weights,
            feature_name=features_by_side[side],
            free_raw_data=True,
        )
        models[side] = lgb.train(params, dataset, num_boost_round=rounds)
        leaf_sizes[side] = leaf
    return ev_map, models, leaf_sizes


def _alpha_calibration(
    calibration: pd.DataFrame,
    features_by_side: dict[str, list[str]],
    params_by_side: dict[str, dict[str, Any]],
    *,
    half_life_months: float,
    leaf_alpha: float,
    hpo_rows: int,
    seed: int,
) -> tuple[dict[str, float], pd.DataFrame]:
    train = _rolling_train(calibration, calibration["__ts__"].min(), 365)
    ev_map, models, _ = _fit_models(
        train, features_by_side, params_by_side,
        half_life_months=half_life_months, leaf_alpha=leaf_alpha,
        hpo_rows=hpo_rows, seed=seed,
    )
    raw = pd.to_numeric(calibration["score_base"], errors="coerce").to_numpy(np.float32)
    expected = residual.predict_hierarchical_ev(ev_map, calibration, raw)
    predicted = residual._predict_side_residuals(calibration, models, features_by_side)
    return residual._tune_alpha(calibration, expected, predicted)


def _run_trial(
    frame: pd.DataFrame,
    base_features: dict[str, list[str]],
    params_by_side: dict[str, dict[str, Any]],
    state_outputs: dict[str, list[str]],
    trial: Trial,
    *,
    calibration_start: pd.Timestamp,
    calibration_end: pd.Timestamp,
    eval_start: pd.Timestamp,
    eval_end: pd.Timestamp,
    max_train_days: int,
    hpo_rows: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    features = {
        side: list(dict.fromkeys([*base_features[side], *state_outputs[side]]))
        for side in ("long", "short")
    }
    calibration = frame.loc[
        frame["__ts__"].ge(calibration_start) & frame["__ts__"].lt(calibration_end)
    ].copy()
    calibration_train = _rolling_train(frame, calibration_start, max_train_days)
    ev_map, models, calibration_leaf = _fit_models(
        calibration_train, features, params_by_side,
        half_life_months=trial.half_life_months, leaf_alpha=trial.leaf_alpha,
        hpo_rows=hpo_rows, seed=seed + 101,
    )
    cal_raw = pd.to_numeric(calibration["score_base"], errors="coerce").to_numpy(np.float32)
    cal_expected = residual.predict_hierarchical_ev(ev_map, calibration, cal_raw)
    cal_residual = residual._predict_side_residuals(calibration, models, features)
    alpha_by_side, alpha_search = residual._tune_alpha(calibration, cal_expected, cal_residual)

    folds: list[pd.DataFrame] = []
    fold_rows: list[dict[str, Any]] = []
    current = eval_start
    fold_index = 0
    while current < eval_end:
        end = min(current + pd.offsets.MonthBegin(1), eval_end)
        train = _rolling_train(frame, current, max_train_days)
        test = frame.loc[frame["__ts__"].ge(current) & frame["__ts__"].lt(end)].copy()
        if test.empty:
            current = end
            continue
        ev_map, models, leaf_sizes = _fit_models(
            train, features, params_by_side,
            half_life_months=trial.half_life_months, leaf_alpha=trial.leaf_alpha,
            hpo_rows=hpo_rows, seed=seed + 1_000 + fold_index * 11,
        )
        corrected_map = residual._fit_corrected_ev_map(
            train, ev_map, models, features, alpha_by_side, backbone_score_col="score_base"
        )
        raw = pd.to_numeric(test["score_base"], errors="coerce").to_numpy(np.float32)
        expected = residual.predict_hierarchical_ev(ev_map, test, raw)
        residual_ev = residual._predict_side_residuals(test, models, features)
        alpha = test["side_name"].astype(str).map(alpha_by_side).fillna(0.0).to_numpy(np.float32)
        corrected = expected + alpha * residual_ev
        test["score_base_rank"] = raw
        test["score_base_ev_residual_fullfit"] = corrected.astype(np.float32)
        test["score_base_ev_residual_fullfit_hier_mapped"] = residual.predict_hierarchical_ev(
            corrected_map, test, corrected
        ).astype(np.float32)
        test["calendar_month"] = test["__ts__"].dt.strftime("%Y-%m")
        day = test["__ts__"].dt.normalize()
        test["week_start"] = day - pd.to_timedelta(day.dt.weekday, unit="D")
        folds.append(test)
        fold_rows.append({
            "trial": trial.name, "test_start": current.isoformat(), "test_end": end.isoformat(),
            "train_rows": int(len(train)), "test_rows": int(len(test)),
            "long_leaf": leaf_sizes["long"], "short_leaf": leaf_sizes["short"],
        })
        current = end
        fold_index += 1
    scored = pd.concat(folds, ignore_index=True)
    metrics = residual._breakdown(
        scored, ["score_base_rank", "score_base_ev_residual_fullfit_hier_mapped"]
    )
    metrics["trial"] = trial.name
    metric = residual._score_metrics(
        scored, scored["score_base_ev_residual_fullfit_hier_mapped"].to_numpy(np.float32)
    )
    baseline = residual._score_metrics(
        scored, scored["score_base_rank"].to_numpy(np.float32)
    )
    gain = float(metric["mean_ev_after_1pct"] - baseline["mean_ev_after_1pct"])
    allowance = max(gain / 5.0, 0.0)
    admissible = (
        metric["worst_week_ev_after_1pct"] >= baseline["worst_week_ev_after_1pct"] - allowance
        and metric["worst_month_ev_after_1pct"] >= baseline["worst_month_ev_after_1pct"] - allowance
    )
    summary = {
        "trial": trial.name, "stage": trial.stage, "half_life_months": trial.half_life_months,
        "leaf_alpha": trial.leaf_alpha, "state_mode": trial.state_mode,
        "alpha_long": alpha_by_side["long"], "alpha_short": alpha_by_side["short"],
        "calibration_long_leaf": calibration_leaf["long"], "calibration_short_leaf": calibration_leaf["short"],
        "admissible": bool(admissible), "top10_ev_after_1pct": metric["mean_ev_after_1pct"],
        "worst_week_ev_after_1pct": metric["worst_week_ev_after_1pct"],
        "worst_month_ev_after_1pct": metric["worst_month_ev_after_1pct"],
        "base_top10_ev_after_1pct": baseline["mean_ev_after_1pct"],
        "delta_top10_ev": gain,
    }
    return metrics, pd.DataFrame(fold_rows), summary


def _winner(summary: pd.DataFrame) -> pd.Series:
    eligible = summary.loc[summary["admissible"]]
    pool = eligible if not eligible.empty else summary
    return pool.sort_values(
        ["top10_ev_after_1pct", "worst_week_ev_after_1pct", "worst_month_ev_after_1pct"],
        ascending=False, kind="stable",
    ).iloc[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--handoff", type=Path, required=True)
    parser.add_argument("--scored-ledger", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--long-state", type=Path, required=True)
    parser.add_argument("--short-state", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--calibration-month", default="2026-03")
    parser.add_argument("--eval-start", default="2026-04-01")
    parser.add_argument("--eval-end", default="2026-07-11")
    parser.add_argument("--max-train-days", type=int, default=365)
    parser.add_argument("--hpo-reference-rows", type=int, default=45_000)
    parser.add_argument("--half-lives", default="2,3,4,5,6")
    parser.add_argument("--alpha-grid", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    calibration_start = pd.Timestamp(f"{args.calibration_month}-01", tz="UTC")
    calibration_end = calibration_start + pd.offsets.MonthBegin(1)
    eval_start = pd.Timestamp(args.eval_start, tz="UTC")
    eval_end = pd.Timestamp(args.eval_end, tz="UTC")
    half_lives = _parse_csv_floats(args.half_lives)
    alpha_grid = _parse_csv_floats(args.alpha_grid)
    if any(value < 0.1 or value > 0.9 for value in alpha_grid):
        raise ValueError("leaf alpha grid must lie in [0.1, 0.9]")

    selection = json.loads(args.selection_manifest.read_text(encoding="utf-8"))
    base_features = {
        side: [str(value) for value in selection["selected_features"][side]]
        for side in ("long", "short")
    }
    params_by_side = {
        side: dict(selection["hpo_params"][side]) for side in ("long", "short")
    }
    states = {"long": _load_state(args.long_state), "short": _load_state(args.short_state)}
    requested = list(dict.fromkeys(
        [
            *[name for values in base_features.values() for name in values if name not in residual.ANCHORS],
            *_state_context_features(states),
        ]
    ))
    frame = residual._load_current_handoff_with_feature_store(
        args.handoff, args.scored_ledger, args.feature_dir, requested,
        rank_fit_end_exclusive=calibration_start,
    )
    frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True, errors="coerce")
    if frame["__ts__"].isna().any():
        raise ValueError("handoff has invalid timestamps")
    # State outputs are materialised once; trials only choose whether each side
    # may consume its own frozen state block.
    frame = _append_state_context(frame, states, "side_local")
    for col in frame.columns:
        if col.startswith("base_long_state_") or col.startswith("base_short_state_"):
            frame[col] = pd.to_numeric(frame[col], errors="coerce").astype(np.float32, copy=False)
    state_outputs_all = _state_output_features(states, "side_local")
    print(f"[load] rows={len(frame):,} base_features=({len(base_features['long'])},{len(base_features['short'])})", flush=True)
    coverage_rows: list[dict[str, Any]] = []
    for side, state in states.items():
        local = frame.loc[frame["side_name"].astype(str).eq(side)]
        required = list(state.feature_names)
        missing = [name for name in required if name not in frame]
        finite_joint = (
            np.isfinite(local.loc[:, required].to_numpy(np.float32, copy=False)).all(axis=1).mean()
            if not missing else 0.0
        )
        active_col = f"{state.prefix}_active"
        coverage_rows.append({
            "side": side,
            "state": state.selected_candidate_id,
            "required_inputs": len(required),
            "missing_inputs": len(missing),
            "joint_finite_rate": float(finite_joint),
            "state_active_rate": float(pd.to_numeric(local[active_col], errors="coerce").mean()),
        })
    coverage = pd.DataFrame(coverage_rows)
    coverage.to_csv(args.out_dir / "side_local_state_coverage.csv", index=False)
    if (coverage["missing_inputs"] > 0).any() or (coverage["state_active_rate"] < 0.90).any():
        raise RuntimeError("Side-local AE/GMM state coverage is insufficient; refusing a null representation ablation")
    if args.dry_run:
        print(coverage.to_string(index=False), flush=True)
        return

    summaries: list[dict[str, Any]] = []
    all_metrics: list[pd.DataFrame] = []
    all_folds: list[pd.DataFrame] = []

    def execute(trial: Trial) -> None:
        outputs = _state_output_features(states, trial.state_mode)
        metrics, folds, summary = _run_trial(
            frame, base_features, params_by_side, outputs, trial,
            calibration_start=calibration_start, calibration_end=calibration_end,
            eval_start=eval_start, eval_end=eval_end,
            max_train_days=int(args.max_train_days),
            hpo_rows=int(args.hpo_reference_rows), seed=int(args.seed) + len(summaries) * 101,
        )
        summaries.append(summary); all_metrics.append(metrics); all_folds.append(folds)
        pd.DataFrame(summaries).to_csv(args.out_dir / "trial_summary_checkpoint.csv", index=False)
        print(
            f"[trial] {trial.name} top10={100*summary['top10_ev_after_1pct']:+.4f}% "
            f"worst_week={100*summary['worst_week_ev_after_1pct']:+.4f}% "
            f"admissible={summary['admissible']}", flush=True,
        )

    # Stage A: comparable half-life arms at the neutral leaf-scale anchor.
    for half_life in half_lives:
        execute(Trial("half_life", half_life, 0.5, "global"))
    summary = pd.DataFrame(summaries)
    half_life_winner = _winner(summary.loc[summary["stage"].eq("half_life")])

    # Stage B: exact 0.1-grid leaf-scale scan. This is intentionally limited to
    # the selected half life, avoiding a 45-arm Cartesian HPO-style search.
    for alpha in alpha_grid:
        execute(Trial("leaf_scale", float(half_life_winner.half_life_months), alpha, "global"))
    summary = pd.DataFrame(summaries)
    scale_winner = _winner(summary.loc[summary["stage"].eq("leaf_scale")])

    # Stage C: direct representation ablations using the selected side-local
    # state packages. Long/short-only arms identify which state block adds value.
    for mode in ("global", "long_local", "short_local", "side_local"):
        execute(Trial("side_local_aegmm", float(scale_winner.half_life_months), float(scale_winner.leaf_alpha), mode))
    summary = pd.DataFrame(summaries)
    final_winner = _winner(summary)

    summary.to_csv(args.out_dir / "trial_summary.csv", index=False)
    pd.concat(all_metrics, ignore_index=True).to_csv(args.out_dir / "metrics_by_trial.csv", index=False)
    pd.concat(all_folds, ignore_index=True).to_csv(args.out_dir / "folds_by_trial.csv", index=False)
    canonical = {
        "schema": "meta_residual_fullfit_defaults_v1",
        "architecture": "base_correctness_residual_expert_single_head_per_side",
        "target": "residual_net_ev_after_1pct",
        "selection_manifest": str(args.selection_manifest),
        "base_ae_gmm_states": {side: str(path) for side, path in (("long", args.long_state), ("short", args.short_state))},
        "max_train_days": int(args.max_train_days),
        "hpo_reference_rows": int(args.hpo_reference_rows),
        "winner": {key: value.item() if isinstance(value, np.generic) else value for key, value in final_winner.to_dict().items()},
        "note": "Requires OOS confirmation before production promotion.",
    }
    (args.out_dir / "canonical_recommendation.json").write_text(
        json.dumps(canonical, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
