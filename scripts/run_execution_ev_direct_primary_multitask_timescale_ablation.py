#!/usr/bin/env python3
"""Direct-primary multi-task and transition-timescale ablations.

This is deliberately a *second-stage* experiment over frozen, strict outer-OOF
execution-EV component heads.  It does not refit the direct execution model or
change its label.  Every learned multi-task score is fitted only on earlier
outer-OOF predictions whose 12-hour execution outcome has resolved before the
next fold.  Its direct-EV output is the only ranking score.

Transition probabilities are kept out of that ranker.  The 1h/3h probabilities
produce a separate ``wait_or_reprice`` action; 6h/12h probabilities, persistence,
horizon disagreement, an expected causal state age, and raw-state velocity feed
an uncertainty/abstention action.  One pooled global top-k is formed from the
unchanged score before action-layer diagnostics are calculated.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_execution_ev_decomposition_calibration_ablation import (  # noqa: E402
    DIRECT,
    MULTITASK_FEATURES,
)
from scripts.run_execution_ev_recent_residual_shrinkage_ablation import (  # noqa: E402
    policy_global_topk_mask,
)


SCHEMA = "execution_ev_direct_primary_multitask_timescale_ablation_v1"
DECISION = "execution_decision_utc"
RESOLUTION = "execution_label_end_utc"
TARGET = "execution_net_ev_12h"
SIDE = "side_name"
HORIZONS = (1, 3, 6, 12)
DEFAULT_INPUT = ROOT / "data_perp/artifacts/execution_ev_hierarchical_shared_multitask_compact_july19_20260726_v3/oof_predictions.parquet"
DEFAULT_TRANSITIONS = ROOT / "data_perp/artifacts/execution_ev_raw_market_state_transition_heads_20260726_v2/strict_weekly_oof_transition_predictions.parquet"
DEFAULT_RAW_STATE = ROOT / "data_perp/artifacts/execution_ev_raw_market_state_transition_heads_20260726_v1/raw_market_state_transition_rows.parquet"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/execution_ev_direct_primary_multitask_timescale_20260726_v1"

# Five economically distinct auxiliary target groups.  Add-one-out therefore
# means removing one group while keeping the direct target and all other groups.
AUXILIARY_GROUPS = ("positive", "clean", "severe", "positive_magnitude", "loss_magnitude")


def _safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(_safe(payload), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, utc=True, errors="raise")


def _soft_step(values: np.ndarray, scale: float = 0.005) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values / float(scale), -40.0, 40.0)))


def _head_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in MULTITASK_FEATURES if column not in frame]
    if missing:
        raise ValueError(f"input lacks frozen multi-task heads: {missing}")
    x = frame.loc[:, list(MULTITASK_FEATURES)].apply(pd.to_numeric, errors="coerce")
    if x.isna().any().any():
        raise ValueError("shared OOF matrix contains non-finite component-head values")
    return x


def _auxiliary_target(group: str, net_ev: np.ndarray, clean: np.ndarray, severe: np.ndarray) -> np.ndarray:
    if group == "positive":
        return _soft_step(net_ev)
    if group == "clean":
        return clean.astype(np.float64)
    if group == "severe":
        return _soft_step(-net_ev - severe)
    if group == "positive_magnitude":
        return np.clip(np.maximum(net_ev, 0.0) / 0.05, 0.0, 4.0)
    if group == "loss_magnitude":
        return np.clip(np.maximum(-net_ev, 0.0) / 0.05, 0.0, 4.0)
    raise ValueError(f"unknown auxiliary group {group!r}")


def _deterministic_train_subset(indices: np.ndarray, max_rows: int) -> np.ndarray:
    """A timestamp-preserving spread sample, never outcome-selected."""
    if len(indices) <= max_rows:
        return indices
    position = np.linspace(0, len(indices) - 1, int(max_rows), dtype=np.int64)
    return indices[position]


def fit_direct_primary_multitask_oof(
    frame: pd.DataFrame,
    *,
    active_auxiliary: Iterable[str],
    direct_weight: int,
    clip_direct_z: float | None,
    residual_to_frozen_direct: bool,
    min_prior_rows: int,
    max_train_rows: int,
    max_iter: int,
    random_state: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Side-head/shared-trunk OOF stacker with exact direct-score fallback.

    ``direct_weight`` is implemented as repeated direct outputs in one MLP
    loss.  The score is their mean in original net-return units.  This is the
    only intentional output weighting; no target-derived sample weights enter
    features or rank normalization.
    """
    active_auxiliary = tuple(active_auxiliary)
    if any(name not in AUXILIARY_GROUPS for name in active_auxiliary):
        raise ValueError("active auxiliary groups must be a subset of the declared groups")
    if int(direct_weight) < 1:
        raise ValueError("direct_weight must be positive")
    x = _head_matrix(frame)
    net_ev = pd.to_numeric(frame[TARGET], errors="raise").to_numpy(float)
    frozen_direct = pd.to_numeric(frame[DIRECT], errors="raise").to_numpy(float)
    clean = frame["clean_favorable_first_exact_policy"].astype(bool).to_numpy()
    severe = pd.to_numeric(frame["severe_loss_floor"], errors="raise").to_numpy(float)
    fold = pd.to_numeric(frame["oof_fold"], errors="raise").to_numpy(int)
    decision = _utc(frame[DECISION])
    resolution = _utc(frame[RESOLUTION])
    side = frame[SIDE].astype(str).str.lower().to_numpy()
    result = frozen_direct.copy()
    audit: list[dict[str, Any]] = []
    for fold_id in sorted(np.unique(fold)):
        current_fold = fold == int(fold_id)
        cutoff = decision[current_fold].min()
        for side_offset, side_name in enumerate(("long", "short")):
            current = current_fold & (side == side_name)
            reference = (fold < int(fold_id)) & (side == side_name) & resolution.lt(cutoff).to_numpy()
            status = "frozen_direct_fallback_insufficient_prior_outer_oof"
            reference_indices = np.flatnonzero(reference)
            if current.any() and len(reference_indices) >= int(min_prior_rows):
                fitted_indices = _deterministic_train_subset(reference_indices, int(max_train_rows))
                base_target = net_ev[fitted_indices] - frozen_direct[fitted_indices] if residual_to_frozen_direct else net_ev[fitted_indices]
                location = float(np.mean(base_target))
                scale = max(float(np.std(base_target)), 1e-4)
                direct_z = (base_target - location) / scale
                if clip_direct_z is not None:
                    direct_z = np.clip(direct_z, -float(clip_direct_z), float(clip_direct_z))
                target_columns = [direct_z] * int(direct_weight)
                target_columns.extend(
                    _auxiliary_target(group, net_ev[fitted_indices], clean[fitted_indices], severe[fitted_indices])
                    for group in active_auxiliary
                )
                targets = np.column_stack(target_columns)
                model = make_pipeline(
                    StandardScaler(),
                    MLPRegressor(
                        hidden_layer_sizes=(18, 10), activation="tanh", solver="adam",
                        alpha=0.02, batch_size=512, learning_rate_init=0.002,
                        max_iter=int(max_iter), early_stopping=True,
                        validation_fraction=0.10, n_iter_no_change=5,
                        random_state=int(random_state + 1000 * fold_id + side_offset),
                    ),
                )
                model.fit(x.iloc[fitted_indices], targets)
                predicted = np.asarray(model.predict(x.loc[current]), dtype=float)
                if predicted.ndim == 1:
                    predicted = predicted[:, None]
                direct_prediction = predicted[:, : int(direct_weight)].mean(axis=1)
                correction = location + scale * direct_prediction
                result[np.flatnonzero(current)] = (
                    frozen_direct[current] + correction if residual_to_frozen_direct else correction
                )
                status = "side_head_shared_trunk_prior_outer_oof_direct_primary"
            audit.append(
                {
                    "fold": int(fold_id), "side": side_name,
                    "validation_rows": int(current.sum()), "reference_oof_rows": int(reference.sum()),
                    "fitted_rows": int(min(len(reference_indices), int(max_train_rows))),
                    "reference_max_resolution_utc": resolution[reference].max().isoformat() if reference.any() else None,
                    "validation_start_utc": cutoff.isoformat(), "status": status,
                    "direct_loss_repetitions": int(direct_weight),
                    "direct_target": "residual_to_frozen_direct" if residual_to_frozen_direct else "net_ev",
                    "direct_target_clip_z": clip_direct_z,
                    "auxiliary_groups": list(active_auxiliary),
                }
            )
    return result, audit


def _wide_transition_predictions(transitions: pd.DataFrame) -> pd.DataFrame:
    required = {"candidate_id", "feature_set", "horizon_hours", "oof_transition_probability", "oof_persistence_probability"}
    missing = required.difference(transitions.columns)
    if missing:
        raise ValueError(f"transition artifact missing {sorted(missing)}")
    work = transitions.loc[
        transitions["feature_set"].eq("combined") & transitions["horizon_hours"].isin(HORIZONS),
        ["candidate_id", "horizon_hours", "oof_transition_probability", "oof_persistence_probability"],
    ].copy()
    if work.duplicated(["candidate_id", "horizon_hours"]).any():
        raise ValueError("transition rows must be unique per candidate and horizon")
    probability = work.pivot(index="candidate_id", columns="horizon_hours", values="oof_transition_probability")
    persistence = work.pivot(index="candidate_id", columns="horizon_hours", values="oof_persistence_probability")
    if set(HORIZONS).difference(probability.columns) or set(HORIZONS).difference(persistence.columns):
        raise ValueError("transition artifact lacks one or more required horizons")
    probability.columns = [f"transition_p_h{int(item)}" for item in probability.columns]
    persistence.columns = [f"persistence_p_h{int(item)}" for item in persistence.columns]
    return probability.join(persistence).reset_index()


def build_timescale_features(frame: pd.DataFrame, raw_state: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build causal opportunity/environment timescale fields.

    The expected age recursively uses only a current OOF persistence forecast
    and an earlier row of the same symbol/side.  It is explicitly not an
    outcome-known time since actual transition.
    """
    out = pd.DataFrame(index=frame.index)
    p = pd.DataFrame({h: pd.to_numeric(frame[f"transition_p_h{h}"], errors="coerce").clip(0.0, 1.0) for h in HORIZONS})
    q = pd.DataFrame({h: pd.to_numeric(frame[f"persistence_p_h{h}"], errors="coerce").clip(0.0, 1.0) for h in HORIZONS})
    near = p[[1, 3]].mean(axis=1)
    environment = p[[6, 12]].mean(axis=1)
    out["timing_transition_risk_1h3h"] = near
    out["timing_persistence_1h3h"] = q[[1, 3]].mean(axis=1)
    out["environment_transition_risk_6h12h"] = environment
    out["environment_persistence_6h12h"] = q[[6, 12]].mean(axis=1)
    out["environment_horizon_disagreement_6h12h"] = (p[6] - p[12]).abs()
    out["near_environment_disagreement"] = (near - environment).abs()
    out["transition_horizon_velocity_1h_to_12h"] = p[12] - p[1]
    out["transition_uncertainty_6h12h"] = (p[[6, 12]] * (1.0 - p[[6, 12]])).mean(axis=1)

    ordered = frame.loc[:, ["candidate_id", "__symbol__", SIDE, DECISION]].copy()
    ordered[DECISION] = _utc(ordered[DECISION])
    ordered["__row__"] = np.arange(len(ordered))
    ordered = ordered.sort_values(["__symbol__", SIDE, DECISION, "candidate_id"], kind="stable")
    state_age = np.zeros(len(frame), dtype=np.float64)
    for _, group in ordered.groupby(["__symbol__", SIDE], sort=False):
        previous_age = 0.0
        previous_time: pd.Timestamp | None = None
        for _, row in group.iterrows():
            now = row[DECISION]
            elapsed = 1.0 if previous_time is None else max(1.0, min(24.0, (now - previous_time).total_seconds() / 3600.0))
            row_position = int(row["__row__"])
            persistence = float(out.iloc[row_position]["timing_persistence_1h3h"])
            previous_age = persistence * min(24.0, previous_age + elapsed)
            state_age[row_position] = previous_age
            previous_time = now
    out["expected_causal_state_age_hours"] = state_age
    out["raw_state_velocity_l1"] = 0.0

    if raw_state is not None:
        required = {"__symbol__", DECISION, "raw_state_source_utc_h0"}
        if not required.issubset(raw_state.columns):
            raise ValueError("raw-state cache lacks causal h0 identity/source fields")
        state = raw_state.copy()
        state[DECISION] = _utc(state[DECISION])
        state["raw_state_source_utc_h0"] = _utc(state["raw_state_source_utc_h0"])
        if (state["raw_state_source_utc_h0"] > state[DECISION]).any():
            raise ValueError("raw h0 state is not point-in-time available")
        columns = [column for column in state if column.startswith("mkt_state__") and column.endswith("__h0")]
        state = state.drop_duplicates(["__symbol__", DECISION], keep="first").sort_values(["__symbol__", DECISION], kind="stable")
        values = state.loc[:, columns].apply(pd.to_numeric, errors="coerce")
        previous = values.groupby(state["__symbol__"], sort=False).shift(1)
        velocity = ((values - previous).abs() / (previous.abs() + 1e-6)).clip(upper=10.0).median(axis=1).fillna(0.0)
        velocity_frame = state.loc[:, ["__symbol__", DECISION]].copy()
        velocity_frame["raw_state_velocity_l1"] = velocity.to_numpy(float)
        key = frame.loc[:, ["__symbol__", DECISION]].copy()
        key[DECISION] = _utc(key[DECISION])
        merged = key.merge(velocity_frame, on=["__symbol__", DECISION], how="left", validate="many_to_one")
        out["raw_state_velocity_l1"] = merged["raw_state_velocity_l1"].fillna(0.0).to_numpy(float)
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _action_metrics(frame: pd.DataFrame, mask: np.ndarray) -> dict[str, Any]:
    net = pd.to_numeric(frame[TARGET], errors="coerce").to_numpy(float)
    selected = np.asarray(mask, dtype=bool) & np.isfinite(net)
    return {
        "rows": int(selected.sum()),
        "mean_net_ev": float(net[selected].mean()) if selected.any() else float("nan"),
        "mean_net_ev_bps": float(10_000.0 * net[selected].mean()) if selected.any() else float("nan"),
        "sum_net_ev": float(net[selected].sum()) if selected.any() else 0.0,
        "positive_rate": float((net[selected] > 0.0).mean()) if selected.any() else float("nan"),
    }


def fit_timescale_action_layer(frame: pd.DataFrame, features: pd.DataFrame, *, min_prior_rows: int, random_state: int) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Strict weekly OOF uncertainty/action layer; it never changes rank score."""
    work = frame.copy()
    work[DECISION] = _utc(work[DECISION])
    work[RESOLUTION] = _utc(work[RESOLUTION])
    work = pd.concat([work, features], axis=1)
    output = pd.DataFrame(index=work.index)
    output["ev_downside_uncertainty_6h12h"] = 0.0
    output["action_wait_or_reprice_1h3h"] = False
    audit: list[dict[str, Any]] = []
    start = work[DECISION].min().floor("7D")
    weeks = pd.date_range(start, work[DECISION].max().ceil("D"), freq="7D", tz="UTC")
    environment_columns = [
        "environment_transition_risk_6h12h", "environment_persistence_6h12h",
        "environment_horizon_disagreement_6h12h", "near_environment_disagreement",
        "transition_horizon_velocity_1h_to_12h", "transition_uncertainty_6h12h",
        "expected_causal_state_age_hours", "raw_state_velocity_l1",
    ]
    for week_start in weeks:
        week_end = week_start + pd.Timedelta(days=7)
        evaluation = work[work[DECISION].ge(week_start) & work[DECISION].lt(week_end)]
        if evaluation.empty:
            continue
        train = work[work[RESOLUTION].lt(week_start)]
        for side_offset, side_name in enumerate(("long", "short")):
            fit = train[train[SIDE].eq(side_name)]
            current = evaluation[evaluation[SIDE].eq(side_name)]
            if current.empty:
                continue
            local = output.index.get_indexer(current.index)
            # A direct forecast is over-optimistic by a material 50 bps. This
            # is directional EV uncertainty, not generic outcome uncertainty.
            target = (
                pd.to_numeric(fit[TARGET], errors="coerce").to_numpy(float)
                < pd.to_numeric(fit[DIRECT], errors="coerce").to_numpy(float) - 0.005
            ).astype(np.int8)
            status = "zero_fallback_insufficient_prior_resolved_rows"
            risk = np.zeros(len(current), dtype=float)
            evaluation_target = (
                pd.to_numeric(current[TARGET], errors="coerce").to_numpy(float)
                < pd.to_numeric(current[DIRECT], errors="coerce").to_numpy(float) - 0.005
            ).astype(np.int8)
            if len(fit) >= int(min_prior_rows) and target.min() != target.max():
                x_train = fit.loc[:, environment_columns].apply(pd.to_numeric, errors="coerce")
                x_eval = current.loc[:, environment_columns].apply(pd.to_numeric, errors="coerce")
                median = x_train.median(axis=0).fillna(0.0)
                model = HistGradientBoostingClassifier(
                    learning_rate=0.05, max_iter=48, max_leaf_nodes=8,
                    min_samples_leaf=100, l2_regularization=5.0,
                    random_state=int(random_state + side_offset + int(week_start.value // 10**12)),
                ).fit(x_train.fillna(median), target)
                risk = model.predict_proba(x_eval.fillna(median))[:, 1]
                status = "side_local_prior_resolved_oof_environment_uncertainty"
            output.iloc[local, output.columns.get_loc("ev_downside_uncertainty_6h12h")] = risk
            # A timing recommendation only: no current/future EV or timing
            # outcome participates in this rule. A high near-term transition
            # risk plus low predicted persistence says wait/reprice, not rank
            # the candidate lower.
            near = current["timing_transition_risk_1h3h"].to_numpy(float)
            persistence = current["timing_persistence_1h3h"].to_numpy(float)
            # Fixed action semantics, not tuned on this evaluation: a 35%+
            # near-term state-change forecast with <70% persistence is enough
            # to flag a potentially better delayed/re-priced entry.  This is
            # deliberately a recommendation only until delayed-entry labels
            # and executable re-entry prices are available.
            output.iloc[local, output.columns.get_loc("action_wait_or_reprice_1h3h")] = (near >= 0.35) & (persistence <= 0.70)
            probability_metrics: dict[str, Any] = {
                "evaluation_downside_overestimate_rate": float(evaluation_target.mean()),
                "evaluation_auc": None, "evaluation_average_precision": None, "evaluation_brier": None,
            }
            if np.unique(evaluation_target).size == 2:
                probability_metrics.update({
                    "evaluation_auc": float(roc_auc_score(evaluation_target, risk)),
                    "evaluation_average_precision": float(average_precision_score(evaluation_target, risk)),
                    "evaluation_brier": float(brier_score_loss(evaluation_target, risk)),
                })
            audit.append({
                "week_start_utc": week_start.isoformat(), "side": side_name,
                "evaluation_rows": int(len(current)), "prior_resolved_rows": int(len(fit)),
                "prior_max_resolution_utc": fit[RESOLUTION].max().isoformat() if len(fit) else None,
                "status": status, "target": "realized_net_ev < frozen_direct_ev - 0.005",
                "features": environment_columns,
                "evaluation_probability_metrics": probability_metrics,
            })
    return output, audit


def _metric_slices(frame: pd.DataFrame, scores: Mapping[str, np.ndarray], *, top_fraction: float) -> dict[str, dict[str, Any]]:
    decision = _utc(frame[DECISION])
    net = pd.to_numeric(frame[TARGET], errors="raise").to_numpy(float)
    side = frame[SIDE].astype(str).to_numpy()
    valid = np.isfinite(net)
    output: dict[str, dict[str, Any]] = {}
    masks: dict[str, np.ndarray] = {"all_oof": valid}
    for month in sorted(decision[valid].dt.strftime("%Y-%m").unique()):
        masks[f"month_{month}"] = valid & decision.dt.strftime("%Y-%m").eq(month).to_numpy()
    latest_start = decision[valid].max() - pd.Timedelta(days=7)
    masks["latest_week"] = valid & decision.ge(latest_start).to_numpy()
    def pooled_topk(score: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
        eligible = np.flatnonzero(mask & np.isfinite(score))
        top_rows = max(1, int(np.ceil(len(eligible) * float(top_fraction)))) if len(eligible) else 0
        selected = eligible[np.argsort(-score[eligible], kind="stable")[:top_rows]] if top_rows else np.array([], dtype=int)
        return {
            "rows": int(len(eligible)), "top_k_rows": int(len(selected)),
            "top_k_mean_net_ev": float(net[selected].mean()) if len(selected) else float("nan"),
            "top_k_sum_net_ev": float(net[selected].sum()) if len(selected) else 0.0,
            "top_k_positive_ev_rate": float((net[selected] > 0.0).mean()) if len(selected) else float("nan"),
            "top_k_predicted_net_ev": float(score[selected].mean()) if len(selected) else float("nan"),
            "top_k_long_rows": int((side[selected] == "long").sum()),
            "top_k_short_rows": int((side[selected] == "short").sum()),
            "ranking_scope": "one_pooled_global_top_k",
        }
    for scope, mask in masks.items():
        output[scope] = {
            name: pooled_topk(np.asarray(score, dtype=float), mask)
            for name, score in scores.items()
        }
    return output


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists() and not args.action_only:
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    if args.action_only and not (args.output_dir / "strict_outer_oof_multitask_predictions.parquet").exists():
        raise FileNotFoundError("--action-only requires the completed multitask prediction artifact")
    frame = pd.read_parquet(args.input)
    required = {"candidate_id", "__symbol__", SIDE, DECISION, RESOLUTION, TARGET, "oof_fold", DIRECT, "clean_favorable_first_exact_policy", "severe_loss_floor"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"input missing {sorted(missing)}")
    for column in (DECISION, RESOLUTION):
        frame[column] = _utc(frame[column])
    # Restrict to the identical strict outer-OOF component-score population.
    frame = frame.loc[frame["oof_fold"].notna() & frame[list(MULTITASK_FEATURES)].notna().all(axis=1)].copy()
    frame["oof_fold"] = pd.to_numeric(frame["oof_fold"], errors="raise").astype(int)
    frame = frame.sort_values([DECISION, "candidate_id"], kind="stable").reset_index(drop=True)

    variants: dict[str, dict[str, Any]] = {
        "direct_only": {"aux": (), "weight": 1, "clip": None, "residual": False},
        "full_aux_w2": {"aux": AUXILIARY_GROUPS, "weight": 2, "clip": None, "residual": False},
        "full_aux_w4": {"aux": AUXILIARY_GROUPS, "weight": 4, "clip": None, "residual": False},
        "full_aux_w8": {"aux": AUXILIARY_GROUPS, "weight": 8, "clip": None, "residual": False},
        "full_aux_w4_clipped_z3": {"aux": AUXILIARY_GROUPS, "weight": 4, "clip": 3.0, "residual": False},
        "residual_to_frozen_direct_w4": {"aux": AUXILIARY_GROUPS, "weight": 4, "clip": 3.0, "residual": True},
    }
    for auxiliary in AUXILIARY_GROUPS:
        variants[f"full_aux_w4_drop_{auxiliary}"] = {
            "aux": tuple(item for item in AUXILIARY_GROUPS if item != auxiliary),
            "weight": 4, "clip": None, "residual": False,
        }
    if args.variants:
        unknown = sorted(set(args.variants).difference(variants))
        if unknown:
            raise ValueError(f"unknown multi-task variants: {unknown}")
        requested = set(args.variants)
        variants = {name: config for name, config in variants.items() if name in requested}
    scores: dict[str, np.ndarray] = {"frozen_direct": pd.to_numeric(frame[DIRECT], errors="raise").to_numpy(float)}
    multitask_audit: dict[str, list[dict[str, Any]]] = {}
    run_parameters: dict[str, Any] = {
        "multitask_max_train_rows": int(args.max_train_rows),
        "multitask_max_iter": int(args.max_iter),
        "action_min_prior_rows": int(args.action_min_prior_rows),
    }
    if args.action_only:
        stored = pd.read_parquet(args.output_dir / "strict_outer_oof_multitask_predictions.parquet")
        score_columns = [column for column in stored if column.startswith("score__")]
        lookup = stored.set_index("candidate_id")[score_columns]
        for column in score_columns:
            scores[column.removeprefix("score__")] = frame["candidate_id"].map(lookup[column]).to_numpy(float)
        prior_summary = json.loads((args.output_dir / "summary.json").read_text())
        run_parameters = dict(prior_summary.get("run_parameters", run_parameters))
        metrics = prior_summary.get("metrics") or _metric_slices(frame, scores, top_fraction=float(args.top_fraction))
        multitask_audit = prior_summary.get("multitask_fit_audit", {})
    else:
        for index, (name, config) in enumerate(variants.items()):
            score, audit = fit_direct_primary_multitask_oof(
                frame, active_auxiliary=config["aux"], direct_weight=int(config["weight"]),
                clip_direct_z=config["clip"], residual_to_frozen_direct=bool(config["residual"]),
                min_prior_rows=int(args.min_prior_rows), max_train_rows=int(args.max_train_rows),
                max_iter=int(args.max_iter), random_state=int(args.random_state + index * 100),
            )
            scores[name] = score
            multitask_audit[name] = audit
            # The grid deliberately fits many MLPs.  Release their fitted arrays
            # before materialising the transition/raw-state action data.
            del score, audit
            gc.collect()
            print(f"[direct-primary-multitask] completed {name}", flush=True)
        metrics = {} if args.no_metrics else _metric_slices(frame, scores, top_fraction=float(args.top_fraction))
        args.output_dir.mkdir(parents=True)
        output = frame.loc[:, ["candidate_id", "__symbol__", SIDE, DECISION, RESOLUTION, TARGET, "oof_fold"]].copy()
        for name, score in scores.items():
            output[f"score__{name}"] = score
        output.to_parquet(args.output_dir / "strict_outer_oof_multitask_predictions.parquet", index=False, compression="zstd")
        if metrics:
            pd.DataFrame([
                {"scope": scope, "arm": arm, **values}
                for scope, arms in metrics.items() for arm, values in arms.items()
            ]).to_csv(args.output_dir / "pooled_global_top10_metrics.csv", index=False)
        if args.multitask_only:
            summary = {
                "schema": SCHEMA, "status": "strict_outer_oof_multitask_stage_complete_pending_timescale_action",
                "contracts": {"direct_primary": "side-specific heads with a shared MLP trunk over frozen component-head inputs; direct net EV repeated in the loss and is the sole score output", "add_one_out": "direct-only, full auxiliary set, and each of five auxiliary target groups removed one at a time", "robustness": "direct target clipping at +/-3 training-standard-deviation units; separate residual-to-frozen-direct variant", "folding": "fit uses only strictly earlier outer-OOF component predictions whose 12h outcome resolves before current fold start; frozen direct score is exact fallback", "ranking": "one pooled global top-k after each score; no timestamp-local or side quota"},
                "rows": {"multitask_outer_oof": int(len(frame))}, "run_parameters": run_parameters, "variant_config": variants,
                "multitask_fit_audit": multitask_audit, "metrics": metrics,
                "sources": {"input": {"path": str(args.input), "sha256": _sha256(args.input)}},
            }
            _write_json(args.output_dir / "summary.json", summary)
            return summary

    gc.collect()
    transition = _wide_transition_predictions(pd.read_parquet(args.transitions))
    action_frame = frame.merge(transition, on="candidate_id", how="inner", validate="one_to_one")
    if action_frame.empty:
        raise ValueError("no exact OOF candidate overlap with transition probabilities")
    raw_state = pd.read_parquet(args.raw_state) if args.raw_state is not None and args.raw_state.exists() else None
    features = build_timescale_features(action_frame, raw_state)
    action, action_audit = fit_timescale_action_layer(
        action_frame, features, min_prior_rows=int(args.action_min_prior_rows), random_state=int(args.random_state + 90_000),
    )
    action_frame = pd.concat([action_frame.reset_index(drop=True), features.reset_index(drop=True), action.reset_index(drop=True)], axis=1)
    action_metrics: list[dict[str, Any]] = []
    for ranking_name in ("frozen_direct", "full_aux_w4", "residual_to_frozen_direct_w4"):
        # Scores are joined by stable candidate identity, never re-ranked per timestamp.
        lookup = pd.Series(scores[ranking_name], index=frame["candidate_id"])
        action_frame[f"ranking_score__{ranking_name}"] = action_frame["candidate_id"].map(lookup).astype(float)
        admitted = np.asarray(policy_global_topk_mask(action_frame, f"ranking_score__{ranking_name}", float(args.top_fraction)), dtype=bool)
        action_frame[f"global_topk__{ranking_name}"] = admitted
        for threshold in (0.50, 0.60, 0.70):
            executed = admitted & (action_frame["ev_downside_uncertainty_6h12h"].to_numpy(float) < threshold)
            action_metrics.append({"ranking": ranking_name, "action_arm": f"environment_abstain_risk_lt_{threshold:.2f}", "threshold": threshold, **_action_metrics(action_frame, executed)})
        action_metrics.append({"ranking": ranking_name, "action_arm": "rank_only_no_action_filter", "threshold": None, **_action_metrics(action_frame, admitted)})
        action_metrics.append({"ranking": ranking_name, "action_arm": "timing_wait_or_reprice_diagnostic", "threshold": None, **_action_metrics(action_frame, admitted & action_frame["action_wait_or_reprice_1h3h"].to_numpy(bool))})
        action_metrics.append({"ranking": ranking_name, "action_arm": "timing_defer_not_executed_counterfactual", "threshold": None, **_action_metrics(action_frame, admitted & ~action_frame["action_wait_or_reprice_1h3h"].to_numpy(bool))})
    action_metric_frame = pd.DataFrame(action_metrics)
    action_frame.to_parquet(args.output_dir / "strict_weekly_oof_timescale_action_predictions.parquet", index=False, compression="zstd")
    action_metric_frame.to_csv(args.output_dir / "timescale_action_metrics.csv", index=False)
    summary = {
        "schema": SCHEMA,
        "status": "strict_outer_oof_diagnostic_not_promoted",
        "contracts": {
            "direct_primary": "side-specific heads with a shared MLP trunk over frozen component-head inputs; direct net EV repeated in the loss and is the sole score output",
            "add_one_out": "direct-only, full auxiliary set, and each of five auxiliary target groups removed one at a time",
            "robustness": "direct target clipping at +/-3 training-standard-deviation units; separate residual-to-frozen-direct variant",
            "folding": "fit uses only strictly earlier outer-OOF component predictions whose 12h outcome resolves before current fold start; frozen direct score is exact fallback",
            "ranking": "one pooled global top-k after each score; no timestamp-local or side quota",
            "timing": "1h/3h transition and persistence produce only wait/reprice diagnostics",
            "environment": "6h/12h transition/persistence, horizon disagreement, expected causal state age, and raw-state velocity feed an OOF downside-overestimation uncertainty action; they never alter rank scores",
            "abstention": "evaluated only after global admission at fixed reported risk thresholds; no threshold is promoted or HPO-selected",
        },
        "rows": {"multitask_outer_oof": int(len(frame)), "timescale_overlap": int(len(action_frame))},
        "run_parameters": run_parameters,
        "variant_config": variants,
        "multitask_fit_audit": multitask_audit,
        "timescale_action_fit_audit": action_audit,
        "metrics": metrics,
        "timescale_action_metrics": action_metrics,
        "sources": {
            "input": {"path": str(args.input), "sha256": _sha256(args.input)},
            "transition_oof": {"path": str(args.transitions), "sha256": _sha256(args.transitions)},
            "raw_state_cache": {"path": str(args.raw_state) if args.raw_state is not None else None, "sha256": _sha256(args.raw_state) if args.raw_state is not None and args.raw_state.exists() else None},
        },
        "outputs": {
            "multitask_predictions": str(args.output_dir / "strict_outer_oof_multitask_predictions.parquet"),
            "timescale_actions": str(args.output_dir / "strict_weekly_oof_timescale_action_predictions.parquet"),
            "metrics": str(args.output_dir / "pooled_global_top10_metrics.csv"),
            "action_metrics": str(args.output_dir / "timescale_action_metrics.csv"),
        },
    }
    _write_json(args.output_dir / "summary.json", summary)
    return summary


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    result.add_argument("--transitions", type=Path, default=DEFAULT_TRANSITIONS)
    result.add_argument("--raw-state", type=Path, default=DEFAULT_RAW_STATE)
    result.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    result.add_argument("--top-fraction", type=float, default=0.10)
    result.add_argument("--min-prior-rows", type=int, default=1500)
    result.add_argument("--action-min-prior-rows", type=int, default=1500)
    result.add_argument("--max-train-rows", type=int, default=30000)
    result.add_argument("--max-iter", type=int, default=28)
    result.add_argument(
        "--variants",
        nargs="+",
        help="Optional named subset for a full-capacity confirmation run.",
    )
    result.add_argument("--random-state", type=int, default=20260726)
    stages = result.add_mutually_exclusive_group()
    stages.add_argument("--multitask-only", action="store_true", help="Write the strict-OOF multi-task grid, then stop before the action layer.")
    stages.add_argument("--action-only", action="store_true", help="Read an already-written grid and run only transition/action diagnostics.")
    result.add_argument("--no-metrics", action="store_true", help="For resource-constrained staged runs, write OOF scores before computing reports.")
    return result


if __name__ == "__main__":
    print(json.dumps(_safe(run(parser().parse_args())), indent=2, sort_keys=True))
