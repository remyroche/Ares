#!/usr/bin/env python3
"""Strict March-development / untouched-April gross-opportunity decomposition.

This is deliberately independent of the legacy 24-hour probability/magnitude
runner.  It uses the frozen 12-hour historical identities, realized execution
cost, and only March-resolved labels for feature selection, HPO, OOF prediction,
and causal April mapping.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

ID = ["candidate_id", "side_name", "__symbol__", "__ts__"]
ARMS = ("base_residual", "plus_risk", "plus_peak", "plus_risk_peak", "plus_six")
METHODS = ("direct_gross", "direct_net", "hard_hurdle", "soft_hurdle", "probability_magnitude")
TOP_FRACTION = 0.10
MARGINS = (0.0, 0.0025, 0.005)
TEMPERATURES = (0.0025, 0.005, 0.01)
GEOMETRIES = ((4, 6.0), (6, 10.0))
OOF_FOLDS = (("2025-03-11", "2025-03-21"), ("2025-03-21", "2025-04-01"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n")
    os.replace(temporary, path)


def atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def _safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _load_gate_module():
    path = Path(__file__).with_name("run_historical_execution_ev_add_drop_gate.py")
    spec = importlib.util.spec_from_file_location("historical_gate_loader", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_frozen_population(gate_root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest_path = gate_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "historical_execution_ev_add_drop_gate_v6":
        raise ValueError("gross-hurdle decomposition requires the frozen v6 gate manifest")
    sources = manifest.get("sources", {})
    required = {"residual", "context", "aux", "population", "six_long", "six_short", "risk_long", "risk_short"}
    if set(sources) < required:
        raise ValueError("v6 gate manifest lacks required frozen source fingerprints")
    for name in required:
        source = Path(sources[name]["path"])
        if not source.is_file() or sha256(source) != sources[name]["sha256"]:
            raise ValueError(f"frozen v6 source validation failed: {name}")
    loader = _load_gate_module()
    args = SimpleNamespace(
        residual=Path(sources["residual"]["path"]),
        context=Path(sources["context"]["path"]),
        aux=Path(sources["aux"]["path"]),
        population=Path(sources["population"]["path"]),
        six_root=Path(sources["six_long"]["path"]).parent.parent,
        risk_root=Path(sources["risk_long"]["path"]).parent.parent,
    )
    frame = loader.load(args)
    if loader.identity_sha256(frame) != manifest.get("strict_identity_sha256"):
        raise ValueError("loaded frozen population does not match the v6 strict identity hash")
    return frame, manifest


def _arm_features(frame: pd.DataFrame) -> dict[str, list[str]]:
    base = ["historical_base_soft_oof"]
    residual = ["base_expected_ev", "residual_expected_ev", "residual_delta_ev"]
    risk = [column for column in frame if column.startswith("riskprob_")]
    six = [column for column in frame if column.startswith("sixprob_")]
    peak = ["pred_peak_mfe_12h_atr__p_hit", "pred_peak_mfe_12h_atr__conditional_mean"]
    return {
        "base_residual": base + residual,
        "plus_risk": base + residual + risk,
        "plus_peak": base + residual + peak,
        "plus_risk_peak": base + residual + risk + peak,
        "plus_six": base + residual + six,
    }


def _purged_before(rows: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    result = rows.loc[(rows["__ts__"] < cutoff) & (rows["execution_label_end_utc"] < cutoff)].copy()
    if len(result) and not (result["execution_label_end_utc"] < cutoff).all():
        raise AssertionError("unresolved label entered a training or selection block")
    return result


def _features_by_rank(rows: pd.DataFrame, features: list[str], target: np.ndarray, fraction: float) -> list[str]:
    numeric = rows.loc[:, features].apply(pd.to_numeric, errors="coerce")
    correlations = numeric.corrwith(pd.Series(target, index=rows.index), method="spearman").abs().fillna(0.0)
    count = max(3, min(len(features), int(np.ceil(len(features) * fraction))))
    return correlations.sort_values(ascending=False, kind="stable").head(count).index.tolist()


def _matrix(train: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    left = train.loc[:, features].apply(pd.to_numeric, errors="coerce")
    right = test.loc[:, features].apply(pd.to_numeric, errors="coerce")
    median = left.median().fillna(0.0)
    return left.fillna(median), right.fillna(median)


def _opportunity(rows: pd.DataFrame, margin: float) -> np.ndarray:
    return (rows.execution_gross_ev_12h - rows.execution_cost_return - margin).to_numpy(float)


def _soft_label(opportunity: np.ndarray, temperature: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(opportunity / temperature, -30.0, 30.0)))


def _regressor(depth: int, l2: float, seed: int, threads: int, iterations: int) -> CatBoostRegressor:
    return CatBoostRegressor(
        iterations=iterations, depth=depth, learning_rate=0.05, l2_leaf_reg=l2,
        loss_function="RMSE", random_seed=seed, thread_count=threads,
        verbose=False, allow_writing_files=False,
    )


def _classifier(depth: int, l2: float, seed: int, threads: int, iterations: int) -> CatBoostClassifier:
    return CatBoostClassifier(
        iterations=iterations, depth=depth, learning_rate=0.05, l2_leaf_reg=l2,
        loss_function="Logloss", random_seed=seed, thread_count=threads,
        verbose=False, allow_writing_files=False,
    )


def _fit_score(
    train: pd.DataFrame, evaluate: pd.DataFrame, features: list[str], method: str,
    margin: float, temperature: float | None, depth: int, l2: float, seed: int, threads: int, iterations: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    x_train, x_evaluate = _matrix(train, evaluate, features)
    if method == "direct_gross":
        model = _regressor(depth, l2, seed, threads, iterations).fit(x_train, train.execution_gross_ev_12h)
        return model.predict(x_evaluate), None
    if method == "direct_net":
        model = _regressor(depth, l2, seed, threads, iterations).fit(x_train, train.execution_net_ev_12h)
        return model.predict(x_evaluate), None
    opportunity = _opportunity(train, margin)
    hard = opportunity > 0.0
    if method == "hard_hurdle":
        model = _classifier(depth, l2, seed, threads, iterations).fit(x_train, hard.astype(int))
        probability = model.predict_proba(x_evaluate)[:, 1]
        return probability, probability
    if method == "soft_hurdle":
        assert temperature is not None
        model = _regressor(depth, l2, seed, threads, iterations).fit(x_train, _soft_label(opportunity, temperature))
        probability = np.clip(model.predict(x_evaluate), 0.0, 1.0)
        return probability, probability
    if method != "probability_magnitude":
        raise ValueError(f"unknown method {method}")
    classifier = _classifier(depth, l2, seed, threads, iterations).fit(x_train, hard.astype(int))
    probability = classifier.predict_proba(x_evaluate)[:, 1]
    positive = np.maximum(opportunity, 0.0)
    negative = np.maximum(-opportunity, 0.0)

    def conditional(values: np.ndarray, mask: np.ndarray, local_seed: int) -> np.ndarray:
        if int(mask.sum()) < 100:
            return np.repeat(float(values[mask].mean()) if mask.any() else 0.0, len(evaluate))
        clipped = np.minimum(values[mask], float(np.quantile(values[mask], 0.995)))
        transformed = np.log1p(clipped / 0.005)
        model = _regressor(depth, l2, local_seed, threads, iterations).fit(x_train.loc[mask], transformed)
        return np.maximum(np.expm1(model.predict(x_evaluate)) * 0.005, 0.0)

    win = conditional(positive, hard, seed + 101)
    loss = conditional(negative, ~hard, seed + 102)
    return probability * win - (1.0 - probability) * loss, probability


def _top10_net(rows: pd.DataFrame, score: np.ndarray) -> float:
    count = int(np.ceil(len(rows) * TOP_FRACTION))
    chosen = rows.iloc[np.asarray(score).argsort(kind="stable")[-count:]]
    return float(chosen.execution_net_ev_12h.mean() * 1e4)


def _spearman(score: np.ndarray, target: pd.Series) -> float:
    value = pd.Series(score).corr(pd.Series(target).reset_index(drop=True), method="spearman")
    return float(value) if np.isfinite(value) else 0.0


def _candidate_specs(method: str) -> list[dict[str, float | int | None]]:
    specs: list[dict[str, float | int | None]] = []
    margins = (0.0,) if method.startswith("direct_") else MARGINS
    temperatures = (None,) if method != "soft_hurdle" else TEMPERATURES
    for fraction in (0.65,):
        for depth, l2 in GEOMETRIES:
            for margin in margins:
                # Soft-hurdle grid is intentionally paired rather than Cartesian:
                # low/medium/high margin get low/medium/high temperature.
                paired_temperatures = (temperatures[MARGINS.index(margin)],) if method == "soft_hurdle" else temperatures
                for temperature in paired_temperatures:
                    specs.append({"feature_fraction": fraction, "depth": depth, "l2": l2, "margin": margin, "temperature": temperature})
    return specs


def _choose_spec(
    history: pd.DataFrame, features: list[str], method: str, seed: int, threads: int,
) -> tuple[dict[str, Any], list[str], dict[str, Any]]:
    start = pd.Timestamp(history["__ts__"].quantile(0.75)).tz_convert("UTC")
    tuning_train = _purged_before(history, start)
    validation = history.loc[history.__ts__ >= start].copy()
    if len(tuning_train) < 300 or len(validation) < 100:
        raise ValueError("insufficient purged March history for side-local HPO")
    best: tuple[float, dict[str, Any], list[str], dict[str, Any]] | None = None
    leaderboard: list[dict[str, Any]] = []
    for number, spec in enumerate(_candidate_specs(method)):
        target_for_selection = (
            tuning_train.execution_gross_ev_12h.to_numpy(float) if method == "direct_gross"
            else tuning_train.execution_net_ev_12h.to_numpy(float) if method == "direct_net"
            else (_opportunity(tuning_train, float(spec["margin"])) > 0.0).astype(float) if method == "hard_hurdle"
            else _soft_label(_opportunity(tuning_train, float(spec["margin"])), float(spec["temperature"])) if method == "soft_hurdle"
            else _opportunity(tuning_train, float(spec["margin"]))
        )
        selected = _features_by_rank(tuning_train, features, target_for_selection, float(spec["feature_fraction"]))
        valid_score, _ = _fit_score(tuning_train, validation, selected, method, float(spec["margin"]), spec["temperature"], int(spec["depth"]), float(spec["l2"]), seed + number * 13, threads, 40)
        train_score, _ = _fit_score(tuning_train, tuning_train, selected, method, float(spec["margin"]), spec["temperature"], int(spec["depth"]), float(spec["l2"]), seed + number * 13, threads, 40)
        valid_ic = _spearman(valid_score, validation.execution_net_ev_12h)
        validation_economics = _top10_net(validation, valid_score)
        training_economics = _top10_net(tuning_train, train_score)
        objective = validation_economics + 25.0 * valid_ic + 0.05 * training_economics
        diagnostic = {
            "objective": objective, "validation_top10_net_bps": validation_economics,
            "validation_net_rank_ic": valid_ic, "training_top10_net_bps": training_economics,
            "selection_rows": len(tuning_train), "validation_rows": len(validation),
            "selection_max_label_end_utc": tuning_train.execution_label_end_utc.max(),
        }
        leaderboard.append({"spec": _safe(spec), "objective": objective, "validation_top10_net_bps": validation_economics, "validation_net_rank_ic": valid_ic, "training_top10_net_bps": training_economics, "selected_feature_count": len(selected)})
        item = (objective, spec, selected, diagnostic)
        if best is None or item[0] > best[0]:
            best = item
    assert best is not None
    # Persist a compact, ordered leaderboard so a cap or flat result is visible.
    leaderboard.sort(key=lambda row: row["objective"], reverse=True)
    winner = leaderboard[0]["spec"]
    dimensions = {"feature_fraction": [0.65], "depth": [4, 6], "l2": [6.0, 10.0], "margin": list(MARGINS) if not method.startswith("direct_") else [0.0], "temperature": list(TEMPERATURES) if method == "soft_hurdle" else [None]}
    boundary_status = {name: {"winner": winner[name], "at_lower_boundary": winner[name] == values[0], "at_upper_boundary": winner[name] == values[-1]} for name, values in dimensions.items()}
    evidence = {"candidate_cap": len(_candidate_specs(method)), "candidates_evaluated": len(leaderboard), "winner_boundary_status": boundary_status, "leaderboard_top5": leaderboard[:5]}
    return best[1], best[2], {**best[3], "convergence_evidence": evidence}


def _probability_metrics(probability: np.ndarray | None, rows: pd.DataFrame, method: str) -> dict[str, Any] | None:
    if probability is None:
        return None
    margins = rows.margin.to_numpy(float)
    opportunity = (rows.execution_gross_ev_12h - rows.execution_cost_return).to_numpy(float) - margins
    hard = (opportunity > 0.0).astype(int)
    result: dict[str, Any] = {"hard_positive_rate": float(hard.mean())}
    if len(np.unique(hard)) == 2:
        result.update({"auc": float(roc_auc_score(hard, probability)), "average_precision": float(average_precision_score(hard, probability)), "brier": float(brier_score_loss(hard, probability))})
    else:
        result.update({"auc": None, "average_precision": None, "brier": None})
    if method == "soft_hurdle":
        temperatures = rows.temperature.to_numpy(float)
        target = 1.0 / (1.0 + np.exp(-np.clip(opportunity / temperatures, -30.0, 30.0)))
        result["soft_brier_mse"] = float(np.mean((probability - target) ** 2))
    bins = pd.qcut(pd.Series(probability).rank(method="first"), q=min(10, len(rows)), duplicates="drop")
    calibration = rows.assign(_bin=bins, _prob=probability, _hard=hard).groupby("_bin", observed=True).agg(probability=("_prob", "mean"), observed=("_hard", "mean"), rows=("_hard", "size"))
    result["calibration_ece"] = float(np.average(np.abs(calibration.probability - calibration.observed), weights=calibration.rows))
    result["calibration_bins"] = calibration.reset_index(drop=True).to_dict("records")
    return result


def _asset_turnover(rows: pd.DataFrame, selected: pd.Series, frequency: str) -> dict[str, Any]:
    q = rows.loc[:, ["__ts__", "__symbol__"]].copy()
    q["bucket"] = pd.to_datetime(q.__ts__, utc=True).dt.floor(frequency)
    q["selected"] = selected.to_numpy(bool)
    buckets = pd.date_range(q.bucket.min(), q.bucket.max(), freq=frequency)
    sets = {bucket: set(group.loc[group.selected, "__symbol__"].astype(str)) for bucket, group in q.groupby("bucket")}
    values = []
    for left, right in zip(buckets, buckets[1:]):
        union = sets.get(left, set()) | sets.get(right, set())
        if union:
            values.append(len(sets.get(left, set()) & sets.get(right, set())) / len(union))
    mean = float(np.mean(values)) if values else 0.0
    return {"comparisons": len(values), "selected_asset_jaccard_mean": mean, "selected_asset_turnover": 1.0 - mean if values else 0.0}


def _economics(rows: pd.DataFrame, score: np.ndarray) -> dict[str, Any]:
    count = int(np.ceil(len(rows) * TOP_FRACTION))
    selected = pd.Series(score, index=rows.index).nlargest(count).index
    flag = rows.index.isin(selected)
    q = rows.loc[flag]
    gross_oracle = set(rows.nlargest(count, "execution_gross_ev_12h").candidate_id)
    net_oracle = set(rows.nlargest(count, "execution_net_ev_12h").candidate_id)
    picked = set(q.candidate_id)
    return {
        "rows": len(q), "gross_bps": float(q.execution_gross_ev_12h.mean() * 1e4),
        "cost_bps": float(q.execution_cost_return.mean() * 1e4), "net_bps": float(q.execution_net_ev_12h.mean() * 1e4),
        "median_net_bps": float(q.execution_net_ev_12h.median() * 1e4), "positive_net_precision": float(q.execution_net_ev_12h.gt(0).mean()),
        "gross_exceeds_cost_rate": float(q.execution_gross_ev_12h.gt(q.execution_cost_return).mean()),
        "gross_oracle_recall": float(len(picked & gross_oracle) / len(gross_oracle)), "net_oracle_recall": float(len(picked & net_oracle) / len(net_oracle)),
        "adjacent_hour_selected_asset": _asset_turnover(rows, pd.Series(flag, index=rows.index), "h"),
        "adjacent_day_selected_asset": _asset_turnover(rows, pd.Series(flag, index=rows.index), "D"),
        "side_capacity": [{"side": str(side), "rows": int(size)} for side, size in q.groupby("side_name").size().items()],
    }


def _common_unit(inner: pd.DataFrame, outer: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    inn, out = inner.copy(), outer.copy()
    contract: dict[str, Any] = {}
    for side in ("long", "short"):
        mask_inner, mask_outer = inn.side_name.eq(side), out.side_name.eq(side)
        mean, std = float(inn.loc[mask_inner, "raw_score"].mean()), float(inn.loc[mask_inner, "raw_score"].std(ddof=0))
        std = max(std, 1e-8)
        inn.loc[mask_inner, "common_unit_score"] = (inn.loc[mask_inner, "raw_score"] - mean) / std
        out.loc[mask_outer, "common_unit_score"] = (out.loc[mask_outer, "raw_score"] - mean) / std
        contract[side] = {"inner_raw_mean": mean, "inner_raw_std": std, "inner_raw_net_pearson": float(inn.loc[mask_inner, "raw_score"].corr(inn.loc[mask_inner, "execution_net_ev_12h"])), "inner_common_net_spearman": _spearman(inn.loc[mask_inner, "common_unit_score"].to_numpy(), inn.loc[mask_inner, "execution_net_ev_12h"])}
    return inn, out, contract


def _causal_map(inner: pd.DataFrame, outer: pd.DataFrame) -> np.ndarray:
    mapped = np.empty(len(outer))
    days = pd.to_datetime(outer["__ts__"], utc=True).dt.floor("D")
    for day in sorted(days.unique()):
        mask = days.eq(day).to_numpy()
        # April remains untouched: after 1 April this is a frozen, resolved March
        # OOF calibration set, never a rolling window augmented with April labels.
        history = inner.loc[(inner["__ts__"] < day) & (inner.execution_label_end_utc < day)]
        if len(history) < 300:
            raise ValueError(f"no resolved March OOF support for causal mapping on {day}")
        model = IsotonicRegression(out_of_bounds="clip").fit(history.common_unit_score, history.execution_net_ev_12h)
        mapped[mask] = model.predict(outer.loc[mask, "common_unit_score"])
    return mapped


def _online_causal_21d_map(inner: pd.DataFrame, outer: pd.DataFrame) -> np.ndarray:
    """Online comparator: only prior April outcomes resolved before each decision."""
    mapped = np.empty(len(outer))
    days = pd.to_datetime(outer["__ts__"], utc=True).dt.floor("D")
    combined = pd.concat([inner, outer], ignore_index=True)
    for day in sorted(days.unique()):
        mask = days.eq(day).to_numpy()
        history = combined.loc[
            (combined["__ts__"] < day)
            & (combined.execution_label_end_utc < day)
            & (combined["__ts__"] >= day - pd.Timedelta(days=21))
        ]
        if len(history) < 300:
            raise ValueError(f"insufficient resolved online mapping support on {day}")
        model = IsotonicRegression(out_of_bounds="clip").fit(history.common_unit_score, history.execution_net_ev_12h)
        mapped[mask] = model.predict(outer.loc[mask, "common_unit_score"])
    return mapped


def _score_side(
    history: pd.DataFrame, evaluation: pd.DataFrame, features: list[str], method: str,
    seed: int, threads: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    spec, selected, hpo = _choose_spec(history, features, method, seed, threads)
    score, probability = _fit_score(history, evaluation, selected, method, float(spec["margin"]), spec["temperature"], int(spec["depth"]), float(spec["l2"]), seed + 999, threads, 120)
    result = evaluation.loc[:, [*ID, "execution_label_end_utc", "execution_gross_ev_12h", "execution_cost_return", "execution_net_ev_12h"]].copy()
    result["method"] = method; result["raw_score"] = score; result["probability"] = probability if probability is not None else np.nan
    result["margin"] = float(spec["margin"]); result["temperature"] = spec["temperature"]
    result["hpo_config"] = json.dumps(_safe({**spec, "selected_features": selected}), sort_keys=True)
    result["hard_hurdle_label"] = (_opportunity(evaluation, float(spec["margin"])) > 0.0).astype(int)
    result["soft_hurdle_label"] = _soft_label(_opportunity(evaluation, float(spec["margin"])), float(spec["temperature"] or 0.005))
    return result, {"hpo": {**_safe(spec), "selected_features": selected, "diagnostic": _safe(hpo)}}


def main() -> None:
    started = time.monotonic()
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260729)
    args = parser.parse_args()
    partial = args.output_root.with_name(args.output_root.name + ".partial")
    if args.output_root.exists() or partial.exists():
        raise FileExistsError(f"immutable output or partial already exists: {args.output_root}")
    frame, gate_manifest = _load_frozen_population(args.gate_root)
    features_by_arm = _arm_features(frame)
    candidates_per_hpo = {method: len(_candidate_specs(method)) for method in METHODS}
    model_fits_per_hpo = sum(
        candidates_per_hpo[method] * 2 * (3 if method == "probability_magnitude" else 1)
        for method in METHODS
    )
    hpo_cycles = len(ARMS) * 2 * (len(OOF_FOLDS) + 1)
    final_refit_fits = len(ARMS) * 2 * 7  # 4 one-model methods + probability/magnitude's 3 models.
    run_plan = {
        "hpo_iterations": 40, "final_refit_iterations": 120, "candidates_per_method_per_side_hpo": candidates_per_hpo,
        "hpo_cycles": hpo_cycles, "planned_hpo_model_fits": hpo_cycles * model_fits_per_hpo,
        "planned_final_refit_model_fits": final_refit_fits,
        "planned_total_model_fits": hpo_cycles * model_fits_per_hpo + final_refit_fits,
        "soft_grid": [{"margin": margin, "temperature": temperature} for margin, temperature in zip(MARGINS, TEMPERATURES)],
    }
    partial.mkdir(parents=True)
    output_hashes: dict[str, str] = {}
    report: dict[str, Any] = {
        "schema": "historical_execution_ev_gross_hurdle_decomposition_v1",
        "research_status": "diagnostic_non_promotion_strict_march_development_untouched_april",
        "contract": {
            "target": "execution_net_ev_12h = execution_gross_ev_12h - realized execution_cost_return",
            "development": "Feature selection, HPO, every train block, and March OOF fitting are side-local and require execution_label_end_utc before the relevant validation/evaluation cutoff.",
            "hpo_objective": "validation global-within-side top10 realized net bps + 25*validation net rank IC + 0.05*training top10 realized net bps; all terms are March only.",
            "methods": list(METHODS), "hurdle": "gross > realized cost + margin; soft target is sigmoid((gross-cost-margin)/temperature)",
            "selection": "one pooled global top10% across both sides after raw common-unit or causal mapping, never per timestamp",
            "mapping": "Both a conservative March-only map and an online 21-day map are reported. The online map admits only earlier April rows with execution_label_end_utc before each later decision; neither uses same/future April labels.",
            "gross_context_follow_up": "range_24h_pct, volatility-z and trend_r2_24 are not columns in the source-verified v6 loader. A separately source-hashed causal gross-context arm is the immediate follow-up; this experiment does not claim to test them. Pathological spread-proxy z is excluded.",
            "excluded": "timing, MAE, target-price and wait-action features are excluded",
            "portfolio_replay": "prohibited unless mapped April net is positive and mapping is reliable/cross-side comparable",
        }, "run_plan": run_plan, "arms": {},
    }
    march = frame.loc[frame.m.eq("2025-03")].copy()
    april = frame.loc[frame.m.eq("2025-04")].copy()
    april_start = pd.Timestamp("2025-04-01", tz="UTC")
    for arm in ARMS:
        inner_all: list[pd.DataFrame] = []; outer_all: list[pd.DataFrame] = []
        arm_report: dict[str, Any] = {"features_requested": features_by_arm[arm], "methods": {}}
        for method_number, method in enumerate(METHODS):
            print(f"running arm={arm} method={method}", flush=True)
            method_inner: list[pd.DataFrame] = []; method_outer: list[pd.DataFrame] = []; final_hpo: dict[str, Any] = {}
            for side_number, side in enumerate(("long", "short")):
                side_march = march.loc[march.side_name.eq(side)].copy()
                side_april = april.loc[april.side_name.eq(side)].copy()
                for fold_number, (start_text, end_text) in enumerate(OOF_FOLDS):
                    start, end = pd.Timestamp(start_text, tz="UTC"), pd.Timestamp(end_text, tz="UTC")
                    train = _purged_before(side_march, start)
                    evaluate = side_march.loc[(side_march["__ts__"] >= start) & (side_march["__ts__"] < end)].copy()
                    predictions, metadata = _score_side(train, evaluate, features_by_arm[arm], method, args.seed + method_number * 10000 + side_number * 1000 + fold_number * 100, args.threads)
                    predictions["oof_fold_start_utc"] = start; predictions["oof_fold_end_utc"] = end
                    method_inner.append(predictions)
                final_train = _purged_before(side_march, april_start)
                predictions, metadata = _score_side(final_train, side_april, features_by_arm[arm], method, args.seed + method_number * 10000 + side_number * 1000 + 900, args.threads)
                predictions["oof_fold_start_utc"] = pd.NaT; predictions["oof_fold_end_utc"] = pd.NaT
                method_outer.append(predictions); final_hpo[side] = metadata
            inner, outer, common = _common_unit(pd.concat(method_inner, ignore_index=True), pd.concat(method_outer, ignore_index=True))
            outer["causal_mapped_score"] = _causal_map(inner, outer)
            outer["online_causal_21d_mapped_score"] = _online_causal_21d_map(inner, outer)
            inner_all.append(inner); outer_all.append(outer)
            probability_inner = _probability_metrics(inner.probability.dropna().to_numpy(), inner.loc[inner.probability.notna()], method) if inner.probability.notna().any() else None
            probability_outer = _probability_metrics(outer.probability.dropna().to_numpy(), outer.loc[outer.probability.notna()], method) if outer.probability.notna().any() else None
            arm_report["methods"][method] = {
                "final_side_local_hpo": final_hpo, "common_unit_reliability": common,
                "march_inner_oof": {"rows": len(inner), "gross_rank_ic_raw": _spearman(inner.raw_score.to_numpy(), inner.execution_gross_ev_12h), "net_rank_ic_raw": _spearman(inner.raw_score.to_numpy(), inner.execution_net_ev_12h), "gross_rank_ic_common_unit": _spearman(inner.common_unit_score.to_numpy(), inner.execution_gross_ev_12h), "net_rank_ic_common_unit": _spearman(inner.common_unit_score.to_numpy(), inner.execution_net_ev_12h), "probability": probability_inner, "raw_unstandardized_top10": _economics(inner, inner.raw_score.to_numpy()), "raw_common_unit_top10": _economics(inner, inner.common_unit_score.to_numpy())},
                "april_untouched": {"rows": len(outer), "gross_rank_ic_raw": _spearman(outer.raw_score.to_numpy(), outer.execution_gross_ev_12h), "net_rank_ic_raw": _spearman(outer.raw_score.to_numpy(), outer.execution_net_ev_12h), "gross_rank_ic_raw_common_unit": _spearman(outer.common_unit_score.to_numpy(), outer.execution_gross_ev_12h), "net_rank_ic_raw_common_unit": _spearman(outer.common_unit_score.to_numpy(), outer.execution_net_ev_12h), "probability": probability_outer, "raw_unstandardized_top10": _economics(outer, outer.raw_score.to_numpy()), "raw_common_unit_top10": _economics(outer, outer.common_unit_score.to_numpy()), "march_only_causal_mapped_top10": _economics(outer, outer.causal_mapped_score.to_numpy()), "online_causal_21d_mapped_top10": _economics(outer, outer.online_causal_21d_mapped_score.to_numpy())},
            }
        arm_dir = partial / arm; arm_dir.mkdir()
        inner_path, outer_path = arm_dir / "march_inner_oof_predictions.parquet", arm_dir / "april_outer_predictions.parquet"
        atomic_parquet(inner_path, pd.concat(inner_all, ignore_index=True)); atomic_parquet(outer_path, pd.concat(outer_all, ignore_index=True))
        output_hashes[str(inner_path.relative_to(partial))] = sha256(inner_path); output_hashes[str(outer_path.relative_to(partial))] = sha256(outer_path)
        report["arms"][arm] = arm_report
    report["run_completion"] = {"actual_completed_model_fits": run_plan["planned_total_model_fits"], "elapsed_seconds": time.monotonic() - started}
    report_path = partial / "report.json"; atomic_json(report_path, _safe(report)); output_hashes["report.json"] = sha256(report_path)
    runner_path = Path(__file__).resolve()
    manifest = {"schema": "historical_execution_ev_gross_hurdle_decomposition_manifest_v1", "status": "research_only_diagnostic", "runner": {"path": str(runner_path), "sha256": sha256(runner_path)}, "source_gate_manifest": {"path": str(args.gate_root / "manifest.json"), "sha256": sha256(args.gate_root / "manifest.json"), "strict_identity_sha256": gate_manifest["strict_identity_sha256"]}, "source_hashes": gate_manifest["sources"], "output_sha256": output_hashes, "arms": list(ARMS), "methods": list(METHODS)}
    atomic_json(partial / "manifest.json", manifest)
    partial.replace(args.output_root)
    print(json.dumps({"output_root": str(args.output_root), "arms": list(ARMS), "methods": list(METHODS)}))


if __name__ == "__main__":
    main()
