#!/usr/bin/env python3
"""Sequential, causal execution-plan ablation for market-regime features.

This is deliberately not a factorial HPO.  It first records intrinsic
feasibility for G0/G1/G2/G3 primary-state generators, then compares the same
frozen residual score in four uses of the representation:

* U0 direct contextual residual mapping;
* U1 a side x soft-state, hierarchically shrunk residual prior;
* U2 a monotonic uncertainty trust shrinkage;
* U3 the prior followed by that trust shrinkage.

All rankings are one pooled global top-k after a causal common-net-bps map.
No target, action, state ID, or outcome is part of a regime generator input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.regime_oof_stack import IDENTITY_COLUMNS, validate_candidate_identity  # noqa: E402


SCORES = ROOT / "data_perp/artifacts/reconstructed_base_residual_stack_2022_2024_20260730_v3/oof_scores.parquet"
GEOMETRY_STATES = ROOT / "data_perp/artifacts/oof_causal_market_regime_systems_2023q3_2024_20260803_v1/candidate_oof_market_regimes.parquet"
SOURCE_PANEL = ROOT / "data_perp/artifacts/regime_multiview_panel_2022_2026_20260730_v2/multiview_regime_features.parquet"
OUTPUT = ROOT / "data_perp/artifacts/regime_execution_plan_20260803_v1"
SCHEMA = "regime_execution_plan_v1"
TOPS = (0.01, 0.05, 0.10)
LOOKBACK_DAYS = 180
LABEL_DELAY_HOURS = 12
RIDGE_ALPHA = 30.0
PRIOR_SHRINK_ROWS = 2_000.0

PRIMARY_PERSISTENT = (
    "market_regime__entropy", "market_regime__top2_margin",
    "market_regime__state_age_hours", "market_regime__state_switch_probability",
    "market_regime__ood_distance_percentile",
)
PRIMARY_POSITION = (
    "market_regime__assigned_centroid_distance",
    "market_regime__within_state_radius_percentile",
    "market_regime__state_boundary_margin",
    "market_regime__centroid_distance_velocity",
)
PHASE = tuple(f"market_regime__phase_p_{name}" for name in ("stable", "onset", "active", "settling"))
LEVERAGE = tuple(
    f"geometry_regime__leverage_flow__{name}"
    for name in ("entropy", "top2_margin", "state_age_hours", "state_switch_probability", "ood_distance_percentile")
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _rank_ic(left: pd.Series, right: pd.Series) -> float:
    mask = left.notna() & right.notna()
    if int(mask.sum()) < 3:
        return float("nan")
    return float(left.loc[mask].rank(method="average").corr(right.loc[mask].rank(method="average")))


def _top_mask(frame: pd.DataFrame, score: pd.Series, fraction: float) -> np.ndarray:
    count = max(1, int(np.ceil(len(frame) * float(fraction))))
    order = pd.DataFrame({"score": score.to_numpy(float), "candidate_id": frame["candidate_id"].astype(str)}, index=frame.index)
    return frame.index.isin(order.sort_values(["score", "candidate_id"], ascending=[False, True], kind="stable").head(count).index)


def _metrics(frame: pd.DataFrame, mask: np.ndarray) -> dict[str, float]:
    local = frame.loc[mask]
    return {
        "trades": int(len(local)),
        "gross_bps": float(local["execution_gross_ev_12h"].mean() * 10_000),
        "net_bps": float(local["execution_net_ev_12h"].mean() * 10_000),
        "cost_bps": float(local["execution_cost_return"].mean() * 10_000),
        "positive_net_rate": float(local["execution_net_ev_12h"].gt(0.0).mean()),
    }


def _read_generator_bindings(values: Sequence[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for item in values:
        name, separator, raw = item.partition("=")
        if not separator or not name or not raw:
            raise ValueError("generator bindings must be NAME=/path/to/candidate_oof_market_regimes.parquet")
        if name in result:
            raise ValueError(f"duplicate generator: {name}")
        result[name] = Path(raw)
    if set(result) != {"G0_k5", "G1_k4", "G2_k3", "G3_k5_merge"}:
        raise ValueError("generator bindings must contain exactly G0_k5, G1_k4, G2_k3, G3_k5_merge")
    return result


def _candidate_states(path: Path) -> pd.DataFrame:
    states = validate_candidate_identity(pd.read_parquet(path))
    states["__ts__"] = pd.to_datetime(states["__ts__"], utc=True, errors="raise")
    for column in ("regime_available_utc", "transition_available_utc"):
        if (pd.to_datetime(states[column], utc=True) > states["__ts__"]).any():
            raise ValueError(f"{path}: {column} is after decision")
    for column in ("regime_train_end_utc", "transition_train_end_utc"):
        if (pd.to_datetime(states[column], utc=True) >= states["__ts__"]).any():
            raise ValueError(f"{path}: {column} is not strictly prior")
    return states


def _source_contract(generator_path: Path) -> list[str]:
    root = generator_path.parent
    diagnostics = json.loads((root / "parameter_diagnostics.json").read_text())
    fields = {
        field
        for item in diagnostics
        if item.get("system") == "primary"
        for field in item.get("feature_columns", [])
    }
    if not fields:
        raise ValueError(f"{root}: no primary feature contract")
    return sorted(fields)


def _join_source_dimensions(frame: pd.DataFrame, *, panel_path: Path, fields: Sequence[str]) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    hourly = pd.read_parquet(panel_path, columns=["source_utc", *fields])
    hourly["source_utc"] = pd.to_datetime(hourly["source_utc"], utc=True, errors="raise")
    hourly = hourly.sort_values("source_utc", kind="stable").drop_duplicates("source_utc", keep="last")
    renamed = {name: f"source_primary__{position:02d}" for position, name in enumerate(fields)}
    hourly = hourly.rename(columns={"source_utc": "source_feature_utc", **renamed})
    left = frame.sort_values("__ts__", kind="stable")
    joined = pd.merge_asof(left, hourly.sort_values("source_feature_utc", kind="stable"), left_on="__ts__", right_on="source_feature_utc", direction="backward", tolerance=pd.Timedelta(hours=2))
    if (joined["source_feature_utc"] > joined["__ts__"]).fillna(False).any():
        raise ValueError("source-dimension join looked ahead")
    source = list(renamed.values())
    coverage = pd.DataFrame({
        "source_field": list(fields), "feature": source,
        "coverage": [float(joined[column].notna().mean()) for column in source],
        "nonconstant": [bool(joined[column].nunique(dropna=True) > 1) for column in source],
    })
    coverage["admitted"] = coverage["coverage"].ge(0.90) & coverage["nonconstant"]
    admitted = coverage.loc[coverage["admitted"], "feature"].tolist()
    if not admitted:
        raise ValueError("no supported continuous source dimensions")
    return joined.drop(columns=["source_feature_utc"]), admitted, coverage


def load_generator_panel(*, scores_path: Path, generator_path: Path, geometry_path: Path, panel_path: Path) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    wanted = [
        *IDENTITY_COLUMNS, "__reconstructed_soft_alpha_12h__", "execution_net_ev_12h",
        "execution_gross_ev_12h", "execution_cost_return", "score_residual_expected_ev",
        "score_base_expected_ev",
        "residual_is_oof",
    ]
    scores = validate_candidate_identity(pd.read_parquet(scores_path, columns=wanted))
    scores["__ts__"] = pd.to_datetime(scores["__ts__"], utc=True, errors="raise")
    primary = _candidate_states(generator_path)
    geometry = _candidate_states(geometry_path)
    geometry_fields = [name for name in geometry if name.startswith("geometry_regime__")]
    panel = scores.merge(primary, on=list(IDENTITY_COLUMNS), how="inner", validate="one_to_one")
    panel = panel.merge(geometry.loc[:, [*IDENTITY_COLUMNS, *geometry_fields]], on=list(IDENTITY_COLUMNS), how="left", validate="one_to_one")
    if len(panel) != len(primary) or not panel["residual_is_oof"].astype(bool).all():
        raise ValueError("generator is not an exact OOF residual-score population")
    panel["side_is_long"] = panel["side_name"].astype(str).eq("long").astype(float)
    panel, source, coverage = _join_source_dimensions(panel, panel_path=panel_path, fields=_source_contract(generator_path))
    return panel.sort_values(["__ts__", "candidate_id"], kind="stable").reset_index(drop=True), source, coverage


def _finite(frame: pd.DataFrame, features: Iterable[str]) -> None:
    values = frame.loc[:, list(features)].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if np.isinf(values).any():
        raise ValueError("infinite direct regime feature")


def direct_arms(frame: pd.DataFrame, source: Sequence[str]) -> dict[str, list[str]]:
    base = ["score_residual_expected_ev", "side_is_long"]
    state = [f"market_regime__state_p_{index}" for index in range(5)]
    available = set(frame.columns)
    groups = {
        "primary": [name for name in PRIMARY_PERSISTENT if name in available],
        "position": [name for name in PRIMARY_POSITION if name in available],
        "transition": [name for name in PHASE if name in available],
        "leverage": [name for name in LEVERAGE if name in available],
        "membership": [name for name in state if name in available],
        "source": list(source),
    }
    arms = {
        "U0_baseline": base,
        "U0_primary": [*base, *groups["primary"]],
        "U0_transition": [*base, *groups["transition"]],
        "U0_primary_transition": [*base, *groups["primary"], *groups["transition"]],
        "U0_primary_position": [*base, *groups["primary"], *groups["position"]],
        "U0_leverage": [*base, *groups["leverage"]],
        "U0_leverage_transition": [*base, *groups["leverage"], *groups["transition"]],
        "U0_primary_leverage": [*base, *groups["primary"], *groups["leverage"]],
        "U0_primary_leverage_transition": [*base, *groups["primary"], *groups["leverage"], *groups["transition"]],
        "C0_source_dimensions": [*base, *groups["source"]],
        "C1_soft_memberships": [*base, *groups["membership"]],
        "C2_source_plus_memberships": [*base, *groups["source"], *groups["membership"]],
    }
    for fields in arms.values():
        _finite(frame, fields)
    return {name: list(dict.fromkeys(fields)) for name, fields in arms.items()}


def _fit_ridge(train: pd.DataFrame, evaluate: pd.DataFrame, features: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    model = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("ridge", Ridge(alpha=RIDGE_ALPHA))])
    model.fit(train.loc[:, features], train["execution_net_ev_12h"])
    return model.predict(train.loc[:, features]), model.predict(evaluate.loc[:, features])


def _side_reference(train: pd.DataFrame, evaluate: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    global_mean = float(train["execution_net_ev_12h"].mean())
    by_side = train.groupby("side_name", observed=True)["execution_net_ev_12h"].mean().to_dict()
    train_ref = train["side_name"].map(by_side).fillna(global_mean).to_numpy(float)
    eval_ref = evaluate["side_name"].map(by_side).fillna(global_mean).to_numpy(float)
    return train_ref, eval_ref


def _shrunk_soft_prior(train: pd.DataFrame, evaluate: pd.DataFrame, *, residual: np.ndarray, columns: Sequence[str]) -> np.ndarray:
    if not columns:
        return np.zeros(len(evaluate), dtype=float)
    global_mean = float(np.mean(residual))
    output = np.zeros(len(evaluate), dtype=float)
    for side in ("long", "short"):
        train_side = train["side_name"].astype(str).eq(side).to_numpy()
        eval_side = evaluate["side_name"].astype(str).eq(side).to_numpy()
        if not eval_side.any():
            continue
        values = np.zeros(len(columns), dtype=float)
        for position, column in enumerate(columns):
            weights = train.loc[train_side, column].to_numpy(float)
            support = float(weights.sum())
            weighted = float(np.dot(weights, residual[train_side])) if support > 0.0 else 0.0
            values[position] = (weighted + PRIOR_SHRINK_ROWS * global_mean) / (support + PRIOR_SHRINK_ROWS)
        output[eval_side] = evaluate.loc[eval_side, list(columns)].to_numpy(float).dot(values)
    return output


def _monotonic_trust(frame: pd.DataFrame) -> np.ndarray:
    entropy = frame["market_regime__entropy"].to_numpy(float)
    margin = frame["market_regime__top2_margin"].to_numpy(float)
    ood = frame["market_regime__ood_distance_percentile"].to_numpy(float)
    switch = frame["market_regime__state_switch_probability"].to_numpy(float)
    active = frame.get("market_regime__phase_p_active", pd.Series(0.0, index=frame.index)).to_numpy(float)
    onset = frame.get("market_regime__phase_p_onset", pd.Series(0.0, index=frame.index)).to_numpy(float)
    trust = np.clip((1.0 - entropy) * margin * (1.0 - ood) * (1.0 - switch) * (1.0 - 0.50 * active - 0.25 * onset), 0.05, 1.0)
    return trust.astype(float)


def _utility_scores(train: pd.DataFrame, evaluate: pd.DataFrame) -> dict[str, np.ndarray]:
    base_train, base_eval = _fit_ridge(train, evaluate, ["score_base_expected_ev", "side_is_long"])
    _, final_eval = _fit_ridge(train, evaluate, ["score_residual_expected_ev", "side_is_long"])
    residual = train["execution_net_ev_12h"].to_numpy(float) - base_train
    state = [f"market_regime__state_p_{index}" for index in range(5)]
    phase = [name for name in PHASE if name in train]
    state_prior = _shrunk_soft_prior(train, evaluate, residual=residual, columns=state)
    phase_prior = _shrunk_soft_prior(train, evaluate, residual=residual, columns=phase)
    prior = 0.5 * (state_prior + phase_prior)
    trust = _monotonic_trust(evaluate)
    return {
        # The existing residual score is a correction over the base expected
        # value.  Regime prior models the broad conversion offset; trust only
        # attenuates that correction back toward the causal base, never toward
        # a side mean or an arbitrary global score level.
        "U1_regime_prior": final_eval + prior,
        "U2_trust_shrinkage": base_eval + trust * (final_eval - base_eval),
        "U3_prior_plus_trust": base_eval + trust * ((final_eval + prior) - base_eval),
    }


def causal_monthly_scores(panel: pd.DataFrame, arms: dict[str, list[str]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = panel.copy()
    folds: list[dict[str, Any]] = []
    months = pd.date_range(out["__ts__"].min().floor("D").replace(day=1), out["__ts__"].max().floor("D").replace(day=1) + pd.offsets.MonthBegin(1), freq="MS", tz="UTC")
    for start, end in zip(months[:-1], months[1:]):
        evaluate = out["__ts__"].ge(start) & out["__ts__"].lt(end)
        train = out["__ts__"].lt(start - pd.Timedelta(hours=LABEL_DELAY_HOURS)) & out["__ts__"].ge(start - pd.Timedelta(days=LOOKBACK_DAYS))
        if int(train.sum()) < 500:
            for arm in arms:
                out.loc[evaluate, f"mapped__{arm}"] = out.loc[evaluate, "score_residual_expected_ev"]
            for arm in ("U1_regime_prior", "U2_trust_shrinkage", "U3_prior_plus_trust"):
                out.loc[evaluate, f"mapped__{arm}"] = out.loc[evaluate, "score_residual_expected_ev"]
            mode = "cold_start_raw_residual"
        else:
            for arm, features in arms.items():
                _, prediction = _fit_ridge(out.loc[train], out.loc[evaluate], features)
                out.loc[evaluate, f"mapped__{arm}"] = prediction
            for arm, prediction in _utility_scores(out.loc[train], out.loc[evaluate]).items():
                out.loc[evaluate, f"mapped__{arm}"] = prediction
            mode = "trailing_180d_causal_fit"
        folds.append({"month": str(start.date())[:7], "mode": mode, "train_rows": int(train.sum()), "evaluation_rows": int(evaluate.sum()), "train_end_utc": out.loc[train, "__ts__"].max() if train.any() else None})
    mapped = [name for name in out if name.startswith("mapped__")]
    if not np.isfinite(out.loc[:, mapped].to_numpy(float)).all():
        raise ValueError("a causal map left non-finite scores")
    return out, pd.DataFrame(folds)


def _split_scores(train: pd.DataFrame, evaluate: pd.DataFrame, arms: dict[str, list[str]]) -> dict[str, np.ndarray]:
    output = {arm: _fit_ridge(train, evaluate, fields)[1] for arm, fields in arms.items()}
    output.update(_utility_scores(train, evaluate))
    return output


def _transport(panel: pd.DataFrame, arms: dict[str, list[str]]) -> pd.DataFrame:
    windows = (("2023q4_to_2024", "2023-09-01", "2024-01-01", "2025-01-01"), ("2024h1_to_2024h2", "2024-01-01", "2024-07-01", "2025-01-01"))
    rows: list[dict[str, Any]] = []
    for name, train_start, split, test_end in windows:
        train = panel["__ts__"].ge(pd.Timestamp(train_start, tz="UTC")) & panel["__ts__"].lt(pd.Timestamp(split, tz="UTC") - pd.Timedelta(hours=LABEL_DELAY_HOURS))
        test = panel["__ts__"].ge(pd.Timestamp(split, tz="UTC")) & panel["__ts__"].lt(pd.Timestamp(test_end, tz="UTC"))
        for arm, score in _split_scores(panel.loc[train], panel.loc[test], arms).items():
            local = panel.loc[test].copy()
            local["score"] = score
            selected = _top_mask(local, local["score"], 0.10)
            rows.append({"transport": name, "arm": arm, "train_rows": int(train.sum()), "test_rows": int(test.sum()), "top_fraction": 0.10, "net_rank_ic": _rank_ic(local["score"], local["execution_net_ev_12h"]), **_metrics(local, selected)})
    return pd.DataFrame(rows)


def _state_diagnostics(mapped: pd.DataFrame, arms: Iterable[str]) -> pd.DataFrame:
    state = mapped["regime_fold_id"].astype(str) + ":state_" + pd.to_numeric(mapped["regime_state_id"], errors="raise").astype(int).astype(str)
    records: list[dict[str, Any]] = []
    for arm in arms:
        score = mapped[f"mapped__{arm}"]
        selected = _top_mask(mapped, score, 0.10)
        for name, positions in state.groupby(state, observed=True).groups.items():
            local = mapped.loc[positions]
            local_score = score.loc[positions]
            top = _top_mask(local, local_score, 0.10)
            bottom = _top_mask(local, -local_score, 0.10)
            selected_local = selected[mapped.index.get_indexer(local.index)]
            records.append({
                "arm": arm, "fold_local_state": name, "rows": int(len(local)),
                "within_state_net_ic": _rank_ic(local_score, local["execution_net_ev_12h"]),
                "within_state_top10_spread_bps": float((local.loc[top, "execution_net_ev_12h"].mean() - local.loc[bottom, "execution_net_ev_12h"].mean()) * 10_000),
                "mean_predicted_correction_bps": float((local_score - local["score_residual_expected_ev"]).mean() * 10_000),
                "mean_realised_residual_bps": float((local["execution_net_ev_12h"] - local["score_residual_expected_ev"]).mean() * 10_000),
                "global_top10_selection_share": float(selected_local.mean()),
                "selected_long_share": float(local.loc[selected_local, "side_name"].astype(str).eq("long").mean()) if selected_local.any() else float("nan"),
            })
    return pd.DataFrame(records)


def _environment_classifier(frame: pd.DataFrame, *, source: Sequence[str]) -> pd.DataFrame:
    sample = frame.iloc[np.linspace(0, len(frame) - 1, num=min(len(frame), 60_000), dtype=np.int64)].copy()
    target = sample["__ts__"].dt.year.ge(2024).astype(int)
    memberships = [f"market_regime__state_p_{index}" for index in range(5)]
    contracts = {"source_dimensions": list(source), "soft_memberships": memberships, "combined": [*source, *memberships]}
    rows: list[dict[str, Any]] = []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=20260803)
    for name, features in contracts.items():
        model = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("logit", LogisticRegression(max_iter=200, C=0.5))])
        probability = cross_val_predict(model, sample.loc[:, features], target, cv=cv, method="predict_proba", n_jobs=1)[:, 1]
        auc = _rank_ic(pd.Series(probability), target.astype(float))
        rows.append({"representation": name, "rows": int(len(sample)), "descriptive_random_cv_rank_correlation_with_calendar_era": auc})
    return pd.DataFrame(rows)


def _intrinsic_diagnostics(generator_paths: dict[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for generator, path in generator_paths.items():
        diagnostics = json.loads((path.parent / "parameter_diagnostics.json").read_text())
        for item in diagnostics:
            if item.get("system") != "primary":
                continue
            selected = next(value for value in item["stickiness_selection"] if value["stickiness"] == item["selected_stickiness"])
            alignment = item.get("primary_fold_alignment", {})
            merge = item.get("postfit_low_support_merge", {})
            rows.append({
                "generator": generator, "fold_id": item["fold_id"],
                "effective_k": int(item["state_count"]), "gmm_k": int(item["gmm_state_count_before_merge"]),
                "stickiness": float(item["selected_stickiness"]),
                "support_gate": bool(selected["persistent_state_gate_passed"]),
                "minimum_occupancy": float(selected["minimum_state_occupancy"]),
                "median_dwell_hours": float(selected["median_dwell_hours"]),
                "temporal_switch_rate": float(selected["temporal_switch_rate"]),
                "posterior_confidence": float(selected["mean_max_probability"]),
                "stability_objective": float(selected["objective"]),
                "alignment_gate": bool(alignment.get("passed", False)),
                "alignment_status": alignment.get("status"),
                "alignment_distance": alignment.get("mean_matched_centroid_distance"),
                "merge_source_state": merge.get("source_state"), "merge_target_state": merge.get("target_state"),
            })
    detail = pd.DataFrame(rows)
    summary = detail.groupby("generator", observed=True).agg(
        effective_k_min=("effective_k", "min"), effective_k_max=("effective_k", "max"),
        all_support_gates=("support_gate", "all"), all_alignment_gates=("alignment_gate", "all"),
        mean_stability_objective=("stability_objective", "mean"),
        stability_objective_se=("stability_objective", lambda value: float(value.std(ddof=1) / np.sqrt(len(value))) if len(value) > 1 else 0.0),
        min_occupancy=("minimum_occupancy", "min"), max_switch_rate=("temporal_switch_rate", "max"),
    ).reset_index()
    summary["intrinsic_feasible"] = summary["all_support_gates"] & summary["all_alignment_gates"]
    feasible = summary.loc[summary["intrinsic_feasible"]].copy()
    summary["within_one_se_of_best_feasible"] = False
    summary["intrinsic_selection"] = False
    if not feasible.empty:
        best = feasible.loc[feasible["mean_stability_objective"].idxmax()]
        eligible = feasible.loc[feasible["mean_stability_objective"] >= float(best["mean_stability_objective"] - best["stability_objective_se"])]
        winner = eligible.sort_values(["effective_k_max", "generator"], kind="stable").iloc[0]
        summary.loc[summary["generator"].isin(eligible["generator"]), "within_one_se_of_best_feasible"] = True
        summary.loc[summary["generator"].eq(winner["generator"]), "intrinsic_selection"] = True
    return detail, summary


def evaluate(mapped: pd.DataFrame, arms: Iterable[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    aggregate: list[dict[str, Any]] = []
    monthly: list[dict[str, Any]] = []
    side: list[dict[str, Any]] = []
    phase: list[dict[str, Any]] = []
    for arm in arms:
        score = mapped[f"mapped__{arm}"]
        for fraction in TOPS:
            selected = _top_mask(mapped, score, fraction)
            aggregate.append({"arm": arm, "top_fraction": fraction, "rows": int(len(mapped)), "selection": "pooled_global_post_causal_mapping", "net_rank_ic": _rank_ic(score, mapped["execution_net_ev_12h"]), "alpha_rank_ic": _rank_ic(score, mapped["__reconstructed_soft_alpha_12h__"]), **_metrics(mapped, selected)})
            chosen = mapped.loc[selected].copy()
            chosen["month"] = chosen["__ts__"].dt.strftime("%Y-%m")
            for name, local in chosen.groupby("month", observed=True, sort=True):
                monthly.append({"arm": arm, "top_fraction": fraction, "month": name, **_metrics(local, np.ones(len(local), dtype=bool))})
            for name, local in chosen.groupby("side_name", observed=True, sort=True):
                side.append({"arm": arm, "top_fraction": fraction, "side_name": name, **_metrics(local, np.ones(len(local), dtype=bool))})
            phase_name = chosen.loc[:, list(PHASE)].idxmax(axis=1).str.removeprefix("market_regime__phase_p_")
            for name, local in chosen.assign(phase=phase_name).groupby("phase", observed=True, sort=True):
                phase.append({"arm": arm, "top_fraction": fraction, "phase": name, **_metrics(local, np.ones(len(local), dtype=bool))})
    return pd.DataFrame(aggregate), pd.DataFrame(monthly), pd.DataFrame(side), pd.DataFrame(phase)


def run(*, scores_path: Path, geometry_path: Path, source_panel: Path, generator_paths: dict[str, Path], output_dir: Path) -> Path:
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(output)
    intrinsic_detail, intrinsic_summary = _intrinsic_diagnostics(generator_paths)
    all_aggregate: list[pd.DataFrame] = []
    all_monthly: list[pd.DataFrame] = []
    all_side: list[pd.DataFrame] = []
    all_phase: list[pd.DataFrame] = []
    all_transport: list[pd.DataFrame] = []
    all_state: list[pd.DataFrame] = []
    all_coverage: list[pd.DataFrame] = []
    all_classifier: list[pd.DataFrame] = []
    all_folds: list[pd.DataFrame] = []
    contracts: dict[str, Any] = {}
    for generator, state_path in generator_paths.items():
        panel, source, coverage = load_generator_panel(scores_path=scores_path, generator_path=state_path, geometry_path=geometry_path, panel_path=source_panel)
        arms = direct_arms(panel, source)
        mapped, folds = causal_monthly_scores(panel, arms)
        names = [*arms, "U1_regime_prior", "U2_trust_shrinkage", "U3_prior_plus_trust"]
        aggregate, monthly, side, phase = evaluate(mapped, names)
        transport = _transport(panel, arms)
        state = _state_diagnostics(mapped, names)
        classifier = _environment_classifier(panel, source=source)
        for frame in (aggregate, monthly, side, phase, folds, coverage, transport, state, classifier):
            frame.insert(0, "generator", generator)
        all_aggregate.append(aggregate); all_monthly.append(monthly); all_side.append(side); all_phase.append(phase)
        all_folds.append(folds); all_coverage.append(coverage); all_transport.append(transport); all_state.append(state); all_classifier.append(classifier)
        contracts[generator] = {"direct_arms": arms, "source_dimension_fields": source}
    aggregate = pd.concat(all_aggregate, ignore_index=True)
    monthly = pd.concat(all_monthly, ignore_index=True)
    transport = pd.concat(all_transport, ignore_index=True)
    baseline = aggregate.loc[(aggregate.generator == "G0_k5") & (aggregate.arm == "U0_baseline") & aggregate.top_fraction.eq(0.10)].iloc[0]
    gate = aggregate.loc[aggregate.top_fraction.eq(0.10), ["generator", "arm", "net_bps", "net_rank_ic"]].copy()
    gate["net_uplift_vs_g0_baseline"] = gate["net_bps"] - float(baseline.net_bps)
    gate["ic_uplift_vs_g0_baseline"] = gate["net_rank_ic"] - float(baseline.net_rank_ic)
    baseline_transport = transport.loc[transport.arm.eq("U0_baseline") & transport.top_fraction.eq(0.10), ["generator", "transport", "net_bps"]].rename(columns={"net_bps": "baseline_net_bps"})
    current_transport = transport.loc[transport.top_fraction.eq(0.10), ["generator", "arm", "transport", "net_bps"]]
    transport_gate = current_transport.merge(baseline_transport, on=["generator", "transport"], how="left", validate="many_to_one")
    transport_gate["net_uplift_vs_generator_baseline"] = transport_gate["net_bps"] - transport_gate["baseline_net_bps"]
    transport_ok = transport_gate.groupby(["generator", "arm"], observed=True)["net_uplift_vs_generator_baseline"].min().rename("worst_transport_uplift_bps")
    baseline_monthly = monthly.loc[monthly.arm.eq("U0_baseline") & monthly.top_fraction.eq(0.10), ["generator", "month", "net_bps"]].rename(columns={"net_bps": "baseline_net_bps"})
    current_monthly = monthly.loc[monthly.top_fraction.eq(0.10), ["generator", "arm", "month", "net_bps"]]
    month_gate = current_monthly.merge(baseline_monthly, on=["generator", "month"], how="left", validate="many_to_one")
    month_gate["net_uplift_vs_generator_baseline"] = month_gate["net_bps"] - month_gate["baseline_net_bps"]
    worst_month = month_gate.groupby(["generator", "arm"], observed=True)["net_uplift_vs_generator_baseline"].min().rename("worst_month_uplift_bps")
    gate = gate.merge(transport_ok, left_on=["generator", "arm"], right_index=True, how="left")
    gate = gate.merge(worst_month, left_on=["generator", "arm"], right_index=True, how="left")
    feasible = intrinsic_summary.set_index("generator")["intrinsic_feasible"]
    gate["intrinsic_feasible"] = gate["generator"].map(feasible).fillna(False).astype(bool)
    gate["advances"] = (
        gate["intrinsic_feasible"]
        & gate["net_uplift_vs_g0_baseline"].gt(0.0)
        & gate["ic_uplift_vs_g0_baseline"].ge(0.0)
        & gate["worst_transport_uplift_bps"].ge(-2.0)
        & gate["worst_month_uplift_bps"].ge(-2.0)
    )
    conditional = []
    for generator, local in aggregate.loc[aggregate.top_fraction.eq(0.10)].groupby("generator", observed=True):
        values = local.set_index("arm")
        full = values.loc["U0_primary_leverage_transition"]
        for bundle, without in (("primary", "U0_leverage_transition"), ("leverage", "U0_primary_transition"), ("transition", "U0_primary_leverage")):
            conditional.append({"generator": generator, "full_arm": "U0_primary_leverage_transition", "bundle_removed": bundle, "without_arm": without, "full_net_bps": float(full.net_bps), "without_net_bps": float(values.loc[without, "net_bps"]), "conditional_net_delta_bps": float(full.net_bps - values.loc[without, "net_bps"]), "full_net_ic": float(full.net_rank_ic), "without_net_ic": float(values.loc[without, "net_rank_ic"])})
    output.mkdir(parents=True)
    outputs = {
        "generator_intrinsic_folds.csv": intrinsic_detail, "generator_intrinsic_selection.csv": intrinsic_summary,
        "aggregate_metrics.csv": aggregate, "monthly_global_topk.csv": monthly,
        "side_global_topk.csv": pd.concat(all_side, ignore_index=True), "phase_global_topk.csv": pd.concat(all_phase, ignore_index=True),
        "transport_metrics.csv": pd.concat(all_transport, ignore_index=True), "within_state_diagnostics.csv": pd.concat(all_state, ignore_index=True),
        "source_dimension_coverage.csv": pd.concat(all_coverage, ignore_index=True), "environment_predictability.csv": pd.concat(all_classifier, ignore_index=True),
        "causal_mapping_folds.csv": pd.concat(all_folds, ignore_index=True), "advancement_gate.csv": gate,
        "conditional_bundle_ablation.csv": pd.DataFrame(conditional),
    }
    for name, frame in outputs.items():
        frame.to_csv(output / name, index=False)
    manifest = {
        "schema": SCHEMA, "status": "COMPLETED_SEQUENTIAL_REGIME_EXECUTION_PLAN",
        "inputs": {"scores": {"path": str(scores_path.resolve()), "sha256": _sha(scores_path)}, "geometry": {"path": str(geometry_path.resolve()), "sha256": _sha(geometry_path)}, "source_panel": {"path": str(source_panel.resolve()), "sha256": _sha(source_panel)}, "generators": {name: {"path": str(path.resolve()), "sha256": _sha(path)} for name, path in generator_paths.items()}},
        "contract": {"generator_selection": "label-free hierarchical feasibility; smallest K within one SE of the best feasible stability objective", "utility_modes": "direct / side-soft-state residual prior / monotonic uncertainty shrinkage / combined", "ranking": "one pooled global top-k after causal common-net-bps mapping", "source_control": "continuous primary inputs vs aligned soft memberships vs combined", "state_ids": "diagnostic only; priors use aligned soft memberships", "no_action_features": True, "prior_shrink_rows": PRIOR_SHRINK_ROWS, "direct_arms": contracts},
        "outputs": {name: _sha(output / name) for name in outputs},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def _args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, default=SCORES)
    parser.add_argument("--geometry-states", type=Path, default=GEOMETRY_STATES)
    parser.add_argument("--source-panel", type=Path, default=SOURCE_PANEL)
    parser.add_argument("--generator", action="append", required=True, help="NAME=/path/to/candidate_oof_market_regimes.parquet; provide G0_k5,G1_k4,G2_k3,G3_k5_merge")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    return parser.parse_args(argv)


if __name__ == "__main__":
    values = _args()
    print(run(scores_path=values.scores, geometry_path=values.geometry_states, source_panel=values.source_panel, generator_paths=_read_generator_bindings(values.generator), output_dir=values.output_dir))
