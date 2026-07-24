#!/usr/bin/env python3
"""Nested activation-curve, causal sizing-normalization, and mixed 1m ablations."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import optuna
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.simple_policy_1m_ablation import evaluate_results  # noqa: E402
from extreme_price_movements.simple_policy_1m_constrained import (  # noqa: E402
    ACTIVATION_CURVE_BLENDED, ACTIVATION_CURVE_POST_ACTIVATION, ACTIVATION_CURVE_TOTAL_MFE,
    FAMILY_TRAILING_ONLY, ConstrainedReplaySpec, constrained_params_to_vector, simulate_constrained_1m_paths,
)
from extreme_price_movements.simple_policy_1m_contextual import stable_fold_objective  # noqa: E402
from extreme_price_movements.simple_policy_1m_sizing_normalizers import (  # noqa: E402
    archetype_ewma_normalize, bounded_dynamic_exposure_normalize,
    open_portfolio_budget_normalize, rolling_window_normalize,
)
from scripts.run_simple_policy_1m_capital_ablation import FOLDS, _load_deployed_side_params, _load_or_build_path_cache, _write_json  # noqa: E402
from scripts.run_simple_policy_1m_constrained_search import INNER_FOLDS, ExperimentData, _evaluate, _indices_between, _objective  # noqa: E402
from scripts.run_simple_policy_1m_contextual_ablation import (  # noqa: E402
    OUTPUT_DTYPES, OUTPUT_FILLS, OUTPUT_KEYS, _bar_neutral_sizes, _bayesian_sizes,
    _load_atr, _load_context, _weighted_evaluate,
)


ACTIVATION_MODES = {
    "total_mfe_curve": ACTIVATION_CURVE_TOTAL_MFE,
    "post_activation_curve": ACTIVATION_CURVE_POST_ACTIVATION,
    "blended_curve": ACTIVATION_CURVE_BLENDED,
}
NORMALIZERS = (
    "raw", "bar_neutral", "rolling_6h", "rolling_12h", "rolling_18h",
    "rolling_24h", "rolling_36h", "rolling_48h", "rolling_72h",
    "archetype_ewma_24h", "archetype_ewma_72h", "open_portfolio_budget",
    "bounded_dynamic_2p5", "bounded_dynamic_5", "bounded_dynamic_7p5", "bounded_dynamic_10",
)


def _empty_outputs(n: int) -> dict[str, np.ndarray]:
    return {k: np.full(n, fill, dtype=dtype) for k, dtype, fill in zip(OUTPUT_KEYS, OUTPUT_DTYPES, OUTPUT_FILLS)}


def _simulate_activation(
    data: ExperimentData,
    indices: np.ndarray,
    params_by_side: Mapping[str, Mapping[str, Any]],
    *,
    mode: int,
    blend: float,
) -> dict[str, np.ndarray]:
    ordered = np.asarray(indices, dtype=np.int64)
    outputs = _empty_outputs(len(ordered))
    for side_name, sign in (("long", 1.0), ("short", -1.0)):
        local = np.flatnonzero(data.side[ordered] * sign > 0.0)
        if not len(local): continue
        result = simulate_constrained_1m_paths(
            ordered[local], data.open0, data.high, data.low, data.close, data.side,
            data.atr_frac, data.entry_spread, data.exit_spread,
            constrained_params_to_vector(params_by_side[side_name]), FAMILY_TRAILING_ONLY,
            data.spec.fee_per_side, data.spec.stop_base_gap_bps, data.spec.stop_through_fraction,
            data.spec.stop_max_gap_bps, data.spec.capital_trail_epsilon_atr,
            int(mode), float(blend),
        )
        for key, values in zip(OUTPUT_KEYS, result): outputs[key][local] = values
    return outputs


def _activation_trial_params(
    trial: optuna.Trial,
    base: Mapping[str, Mapping[str, Any]],
    *,
    mode: int,
) -> tuple[dict[str, dict[str, Any]], float, dict[str, Any]]:
    global_act = trial.suggest_float("activation_log_scale", -0.70, 0.70)
    side_act = trial.suggest_float("activation_side_delta", -0.30, 0.30)
    global_beta = trial.suggest_float("beta_log_scale", -0.40, 0.40)
    side_beta = trial.suggest_float("beta_side_delta", -0.25, 0.25)
    power_scale = trial.suggest_float("power_log_scale", -0.50, 0.50)
    divisor_scale = trial.suggest_float("divisor_log_scale", -0.60, 0.60)
    half_scale = trial.suggest_float("decay_half_log_scale", math.log(0.5), math.log(5.0))
    long_start = trial.suggest_float("long_decay_start_minutes", 0.0, 180.0)
    short_start = trial.suggest_float("short_decay_start_minutes", 0.0, 120.0)
    long_min = trial.suggest_float("long_decay_min_mult", 0.40, 1.0)
    short_min = trial.suggest_float("short_decay_min_mult", 0.15, 0.80)
    blend = trial.suggest_float("activation_curve_blend", 0.10, 0.90) if mode == ACTIVATION_CURVE_BLENDED else (1.0 if mode == ACTIVATION_CURVE_POST_ACTIVATION else 0.0)
    params: dict[str, dict[str, Any]] = {}
    for side_name, sign in (("long", 1.0), ("short", -1.0)):
        out = dict(base[side_name])
        out["trailing_activation_mult"] = float(np.clip(float(out["trailing_activation_mult"]) * math.exp(global_act + sign * side_act), 0.35, 5.0))
        out["giveback_beta"] = float(np.clip(float(out["giveback_beta"]) * math.exp(global_beta + sign * side_beta), 0.10, 1.10))
        out["trailing_power"] = float(np.clip(float(out["trailing_power"]) * math.exp(power_scale), 0.60, 3.50))
        out["trailing_squash_divisor"] = float(np.clip(float(out["trailing_squash_divisor"]) * math.exp(divisor_scale), 1.25, 8.0))
        out["trailing_activation_decay_half_life_minutes"] = float(np.clip(float(out.get("trailing_activation_decay_half_life_minutes", 60.0)) * math.exp(half_scale), 30.0, 300.0))
        out["trailing_activation_decay_start_minutes"] = long_start if side_name == "long" else short_start
        out["trailing_activation_min_mult"] = long_min if side_name == "long" else short_min
        params[side_name] = out
    deltas = np.asarray([global_act, side_act, global_beta, side_beta, power_scale, divisor_scale, half_scale])
    penalty = float(0.001 * np.sum(np.square(deltas)))
    return params, blend, {"penalty": penalty, "raw_deltas": deltas.tolist()}


def _optimise_activation(
    data: ExperimentData,
    indices: np.ndarray,
    base: Mapping[str, Mapping[str, Any]],
    *,
    mode: int,
    trials: int,
    seeds: list[int],
) -> tuple[dict[str, dict[str, Any]], float, dict[str, Any]]:
    best = (-1e100, None, 0.0, None)
    summaries = []
    for seed in seeds:
        sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True, group=True, n_startup_trials=min(24, max(8, trials // 3)))
        study = optuna.create_study(direction="maximize", sampler=sampler)
        # Include the exact locked parent so a limited multidimensional search
        # cannot make the unchanged formulation lose by simple non-recovery.
        anchor = {
            "activation_log_scale": 0.0,
            "activation_side_delta": 0.0,
            "beta_log_scale": 0.0,
            "beta_side_delta": 0.0,
            "power_log_scale": 0.0,
            "divisor_log_scale": 0.0,
            "decay_half_log_scale": 0.0,
            "long_decay_start_minutes": float(base["long"].get("trailing_activation_decay_start_minutes", 0.0)),
            "short_decay_start_minutes": float(base["short"].get("trailing_activation_decay_start_minutes", 0.0)),
            "long_decay_min_mult": float(base["long"].get("trailing_activation_min_mult", 1.0)),
            "short_decay_min_mult": float(base["short"].get("trailing_activation_min_mult", 1.0)),
        }
        if mode == ACTIVATION_CURVE_BLENDED:
            anchor["activation_curve_blend"] = 0.5
        study.enqueue_trial(anchor)
        def objective(trial: optuna.Trial) -> float:
            params, blend, meta = _activation_trial_params(trial, base, mode=mode)
            outputs = _simulate_activation(data, indices, params, mode=mode, blend=blend)
            score, diag = _objective(data, indices, outputs)
            trial.set_user_attr("params_by_side", params); trial.set_user_attr("blend", blend); trial.set_user_attr("diag", diag); trial.set_user_attr("meta", meta)
            return score - float(meta["penalty"])
        study.optimize(objective, n_trials=int(trials), show_progress_bar=False, gc_after_trial=False)
        bt = study.best_trial
        summaries.append({"seed": seed, "best_value": float(study.best_value), "trial": bt.number, "diag": bt.user_attrs["diag"]})
        if study.best_value > best[0]: best = (float(study.best_value), bt.user_attrs["params_by_side"], float(bt.user_attrs["blend"]), dict(bt.params))
    if best[1] is None: raise RuntimeError("activation optimizer returned no result")
    return best[1], best[2], {"best_value_penalized": best[0], "trial_params": best[3], "seeds": summaries, "trials": int(trials) * len(seeds)}


def _normalizer(
    name: str,
    data: ExperimentData,
    fit_idx: np.ndarray,
    fit_outputs: Mapping[str, np.ndarray],
    raw_size: np.ndarray,
    apply_idx: np.ndarray,
    apply_outputs: Mapping[str, np.ndarray],
) -> np.ndarray:
    result = np.asarray(raw_size, dtype=np.float64).copy()
    if name == "raw": return result
    if name == "bar_neutral": return _bar_neutral_sizes(data, apply_idx, apply_outputs, result)
    fit_rows = data.rows.iloc[fit_idx].reset_index(drop=True)
    apply_rows = data.rows.iloc[apply_idx].reset_index(drop=True)
    fit_local = result[fit_idx]; apply_local = result[apply_idx]
    if name.startswith("rolling_"):
        hours = float(name.split("_")[1].removesuffix("h"))
        adjusted = rolling_window_normalize(fit_rows, fit_outputs, fit_local, apply_rows, apply_outputs, apply_local, window_hours=hours)
    elif name.startswith("archetype_ewma_"):
        hours = float(name.rsplit("_", 1)[1].removesuffix("h"))
        adjusted = archetype_ewma_normalize(fit_rows, fit_outputs, fit_local, apply_rows, apply_outputs, apply_local, half_life_hours=hours)
    elif name == "open_portfolio_budget":
        adjusted = open_portfolio_budget_normalize(apply_rows, apply_outputs, apply_local)
    elif name.startswith("bounded_dynamic_"):
        bands = {"2p5": 0.025, "5": 0.05, "7p5": 0.075, "10": 0.10}
        adjusted = bounded_dynamic_exposure_normalize(apply_rows, apply_outputs, apply_local, exposure_band=bands[name.rsplit("_", 1)[1]])
    else:
        raise KeyError(name)
    result[apply_idx] = adjusted
    return result


def _metrics(data: ExperimentData, idx: np.ndarray, outputs: Mapping[str, np.ndarray], size: np.ndarray | None = None) -> dict[str, Any]:
    metrics, _ = _evaluate(data, idx, outputs, family=FAMILY_TRAILING_ONLY)
    if size is not None: metrics.update(_weighted_evaluate(data, idx, outputs, size))
    else: metrics.update({"oos_exposure_ratio": 1.0, "exposure_normalized_objective": metrics["objective"], "exposure_normalized_pnl": metrics["net_pnl_bankroll"]})
    return metrics


def _summarize(frame: pd.DataFrame, group_name: str, deployed: pd.DataFrame) -> pd.DataFrame:
    deployed_idx = deployed.set_index("fold")
    rows = []
    for ablation, group in frame.groupby("ablation", sort=False):
        obj = group.objective.to_numpy(dtype=float); neutral = group.exposure_normalized_objective.to_numpy(dtype=float)
        delta = np.asarray([row.objective - deployed_idx.loc[row.fold, "objective"] for row in group.itertuples()])
        neutral_delta = np.asarray([row.exposure_normalized_objective - deployed_idx.loc[row.fold, "objective"] for row in group.itertuples()])
        rows.append({
            "group": group_name, "ablation": ablation, "folds": len(group),
            "stable_objective": stable_fold_objective(obj), "stable_neutral_objective": stable_fold_objective(neutral),
            "mean_objective": float(obj.mean()), "mean_delta_vs_deployed": float(delta.mean()),
            "mean_neutral_delta_vs_deployed": float(neutral_delta.mean()),
            "positive_folds": int(np.sum(delta > 0)), "positive_neutral_folds": int(np.sum(neutral_delta > 0)),
            "mean_pnl": float(group.net_pnl_bankroll.mean()), "mean_neutral_pnl": float(group.exposure_normalized_pnl.mean()),
            "worst_fold_pnl": float(group.net_pnl_bankroll.min()), "worst_week": float(group.worst_week.min()),
            "worst_drawdown": float(group.max_drawdown.min()), "total_trades": int(group.n_trades.sum()),
            "mean_return_per_trade": float(group.mean_net_return.mean()), "hit_rate": float(group.hit_rate.mean()),
            "mean_exposure_ratio": float(group.oos_exposure_ratio.mean()),
        })
    return pd.DataFrame(rows).sort_values("stable_neutral_objective", ascending=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True); parser.add_argument("--rich-ledger", required=True)
    parser.add_argument("--posterior-state", required=True); parser.add_argument("--deployed-parent-summary", required=True)
    parser.add_argument("--path-cache-dir", required=True); parser.add_argument("--atr-audit", required=True)
    parser.add_argument("--joint-parent-params", required=True); parser.add_argument("--output-dir", required=True)
    parser.add_argument("--search-trials", type=int, default=32); parser.add_argument("--full-trials", type=int, default=64)
    parser.add_argument("--seeds", type=int, default=2); parser.add_argument("--seed", type=int, default=20260717)
    args = parser.parse_args(); optuna.logging.set_verbosity(optuna.logging.WARNING)
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    rows = pd.read_parquet(args.candidates); rows["timestamp"] = pd.to_datetime(rows["timestamp"], utc=True)
    rows = rows.sort_values(["timestamp", "rank_pct"], ascending=[True, False], kind="mergesort").reset_index(drop=True)
    context, _, provenance = _load_context(rows, Path(args.rich_ledger), Path(args.posterior_state))
    atr = _load_atr(rows, Path(args.atr_audit)); deployed_by_side, _ = _load_deployed_side_params(Path(args.deployed_parent_summary))
    spec = ConstrainedReplaySpec(); open0, high, low, close, valid, path_manifest = _load_or_build_path_cache(
        rows, store_root=Path("data_perp/exchanges/krakenfutures/execution_1m"), cache_dir=Path(args.path_cache_dir), spec=spec, rebuild=False,
    )
    data = ExperimentData(rows, open0, high, low, close, valid, atr, spec, deployed_by_side)
    parents = json.loads(Path(args.joint_parent_params).read_text(encoding="utf-8"))
    all_rows: list[dict[str, Any]] = []; params_out: dict[str, Any] = {}; choices: dict[str, Any] = {}
    for fold_no, fold in enumerate(FOLDS, 1):
        inner = INNER_FOLDS[fold["fold"]]; search_idx = _indices_between(data, fold["train_start"], inner["search_end"])
        inner_idx = _indices_between(data, inner["inner_start"], inner["inner_end"]); train_idx = _indices_between(data, fold["train_start"], fold["train_end"])
        outer_idx = _indices_between(data, fold["validation_start"], fold["validation_end"])
        base_search = parents[fold["fold"]]["search_parent"]; base_full = parents[fold["fold"]]["full_train_parent"]
        sizing_cfg = parents[fold["fold"]]["sizing"]; strength = float(sizing_cfg["strength"]); ood_weight = float(sizing_cfg["ood_weight"])
        seeds = [args.seed + fold_no * 10000 + i * 1000 for i in range(args.seeds)]
        deployed_outer = data.simulate_deployed(outer_idx)
        all_rows.append({"group": "control", "fold": fold["fold"], "ablation": "current_deployed", **_metrics(data, outer_idx, deployed_outer)})
        base_search_out = data.simulate(search_idx, base_search, FAMILY_TRAILING_ONLY); base_inner_out = data.simulate(inner_idx, base_search, FAMILY_TRAILING_ONLY)
        base_train_out = data.simulate(train_idx, base_full, FAMILY_TRAILING_ONLY); base_outer_out = data.simulate(outer_idx, base_full, FAMILY_TRAILING_ONLY)
        all_rows.append({"group": "activation", "fold": fold["fold"], "ablation": "joint_trailing_only", **_metrics(data, outer_idx, base_outer_out)})

        activation_payload: dict[str, Any] = {}; activation_inner_rank = []
        for offset, (name, mode) in enumerate(ACTIVATION_MODES.items()):
            fixed_blend = 0.5 if mode == ACTIVATION_CURVE_BLENDED else (1.0 if mode == ACTIVATION_CURVE_POST_ACTIVATION else 0.0)
            fixed_outer = _simulate_activation(data, outer_idx, base_full, mode=mode, blend=fixed_blend)
            all_rows.append({
                "group": "activation", "fold": fold["fold"],
                "ablation": f"{name}__fixed_parent", "blend": fixed_blend,
                **_metrics(data, outer_idx, fixed_outer),
            })
            search_params, search_blend, search_diag = _optimise_activation(data, search_idx, base_search, mode=mode, trials=args.search_trials, seeds=[s + 100 + offset for s in seeds])
            search_outer = _simulate_activation(data, inner_idx, search_params, mode=mode, blend=search_blend)
            inner_score = _metrics(data, inner_idx, search_outer)["objective"]
            full_params, full_blend, full_diag = _optimise_activation(data, train_idx, base_full, mode=mode, trials=args.full_trials, seeds=[s + 500 + offset for s in seeds])
            outer_out = _simulate_activation(data, outer_idx, full_params, mode=mode, blend=full_blend)
            train_out = _simulate_activation(data, train_idx, full_params, mode=mode, blend=full_blend)
            metrics = _metrics(data, outer_idx, outer_out)
            all_rows.append({"group": "activation", "fold": fold["fold"], "ablation": name, "blend": full_blend, **metrics})
            activation_inner_rank.append((float(inner_score), name))
            activation_payload[name] = {
                "mode": mode, "search_params": search_params, "search_blend": search_blend, "search_diag": search_diag,
                "full_params": full_params, "full_blend": full_blend, "full_diag": full_diag,
                "search_outputs": search_outer, "train_outputs": train_out, "outer_outputs": outer_out,
            }

        raw_size_base, _ = _bayesian_sizes(data, train_idx, outer_idx, base_train_out, context, strength=strength, ood_weight=ood_weight)
        raw_size_search, _ = _bayesian_sizes(data, search_idx, inner_idx, base_search_out, context, strength=strength, ood_weight=ood_weight)
        normalization_inner = []
        for name in NORMALIZERS:
            inner_size = _normalizer(name, data, search_idx, base_search_out, raw_size_search, inner_idx, base_inner_out)
            inner_metric = _weighted_evaluate(data, inner_idx, base_inner_out, inner_size)
            outer_size = _normalizer(name, data, train_idx, base_train_out, raw_size_base, outer_idx, base_outer_out)
            metrics = _metrics(data, outer_idx, base_outer_out, outer_size)
            all_rows.append({"group": "normalization", "fold": fold["fold"], "ablation": name, **metrics})
            normalization_inner.append((float(inner_metric["exposure_normalized_objective"]), name))

        activation_inner_rank.sort(reverse=True); top_activations = [name for _, name in activation_inner_rank[:2]]
        mixed_specs = []
        for act_rank, act_name in enumerate(top_activations):
            payload = activation_payload[act_name]
            mode = int(payload["mode"])
            act_search_fit = _simulate_activation(data, search_idx, payload["search_params"], mode=mode, blend=payload["search_blend"])
            act_inner = payload["search_outputs"]
            raw_mixed_search, _ = _bayesian_sizes(data, search_idx, inner_idx, act_search_fit, context, strength=strength, ood_weight=ood_weight)
            norm_rank = []
            for norm in NORMALIZERS:
                size = _normalizer(norm, data, search_idx, act_search_fit, raw_mixed_search, inner_idx, act_inner)
                metric = _weighted_evaluate(data, inner_idx, act_inner, size)
                norm_rank.append((float(metric["exposure_normalized_objective"]), norm))
            norm_rank.sort(reverse=True)
            if act_rank == 0:
                mixed_specs.extend([(act_name, norm_rank[0][1], "top1_activation_top1_normalizer"), (act_name, norm_rank[1][1], "top1_activation_top2_normalizer")])
            else:
                mixed_specs.append((act_name, norm_rank[0][1], "top2_activation_top1_normalizer"))
        for act_name, norm, label in mixed_specs:
            payload = activation_payload[act_name]
            raw_mixed, _ = _bayesian_sizes(data, train_idx, outer_idx, payload["train_outputs"], context, strength=strength, ood_weight=ood_weight)
            mixed_size = _normalizer(norm, data, train_idx, payload["train_outputs"], raw_mixed, outer_idx, payload["outer_outputs"])
            metrics = _metrics(data, outer_idx, payload["outer_outputs"], mixed_size)
            all_rows.append({"group": "mixed", "fold": fold["fold"], "ablation": label, "activation_choice": act_name, "normalizer_choice": norm, **metrics})

        choices[fold["fold"]] = {"activation_inner_rank": activation_inner_rank, "normalization_inner_rank": sorted(normalization_inner, reverse=True), "mixed_specs": mixed_specs, "strength": strength, "ood_weight": ood_weight}
        params_out[fold["fold"]] = {name: {k: v for k, v in payload.items() if not k.endswith("outputs")} for name, payload in activation_payload.items()}
        pd.DataFrame(all_rows).to_csv(output / "fold_metrics.partial.csv", index=False); _write_json(output / "choices.partial.json", choices); _write_json(output / "activation_params.partial.json", params_out)

    frame = pd.DataFrame(all_rows); deployed = frame[frame.ablation == "current_deployed"]
    activation = frame[frame.group == "activation"]; normalization = frame[frame.group == "normalization"]; mixed = frame[frame.group == "mixed"]
    s_activation = _summarize(activation, "activation", deployed); s_normalization = _summarize(normalization, "normalization", deployed); s_mixed = _summarize(mixed, "mixed", deployed)
    frame.to_csv(output / "fold_metrics.csv", index=False); s_activation.to_csv(output / "summary_activation.csv", index=False); s_normalization.to_csv(output / "summary_normalization.csv", index=False); s_mixed.to_csv(output / "summary_mixed.csv", index=False)
    _write_json(output / "choices.json", choices); _write_json(output / "activation_params.json", params_out)
    _write_json(output / "manifest.json", {
        "evidence_status": "nested policy-validation OOS; July excluded", "activation_modes": ACTIVATION_MODES,
        "normalizers": NORMALIZERS,
        "mixed_selection": "top two activation choices and exposure-neutral normalizer choices selected on inner validation only",
        "activation_search": {
            "search_trials_per_seed": args.search_trials, "full_trials_per_seed": args.full_trials,
            "seeds": args.seeds, "base_seed": args.seed,
            "locked_parent_enqueued_in_every_study": True,
        },
        "inputs": {
            "candidates": str(Path(args.candidates)), "rich_ledger": str(Path(args.rich_ledger)),
            "posterior_state": str(Path(args.posterior_state)),
            "deployed_parent_summary": str(Path(args.deployed_parent_summary)),
            "atr_audit": str(Path(args.atr_audit)), "joint_parent_params": str(Path(args.joint_parent_params)),
            "path_cache_dir": str(Path(args.path_cache_dir)),
        },
        "outer_folds": FOLDS, "inner_folds": INNER_FOLDS,
        "replay": "exact 1m, causal ATR, 1% round trip, spread, 8-open/2-new", "provenance": provenance, "path": path_manifest,
    })
    print("\nACTIVATION\n", s_activation.to_string(index=False)); print("\nNORMALIZATION\n", s_normalization.to_string(index=False)); print("\nMIXED\n", s_mixed.to_string(index=False))
    return 0


if __name__ == "__main__": raise SystemExit(main())
