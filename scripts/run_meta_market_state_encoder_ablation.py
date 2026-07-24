#!/usr/bin/env python3
"""Compare AE/GMM and supervised-MLP market-state score overlays.

Encoder models are local to side x policy archetype. Expanding-window mode uses
eligible 2026 parent-meta OOS rows before April for tuning, then emits monthly
April-June OOS predictions. The fixed parent is the promoted v9 95%-tail overlay.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import optuna
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.supervised_market_state_calibration import (  # noqa: E402
    EncoderArm,
    FrozenCompositeLocalOverlay,
    FrozenLocalExtremeTailOverlay,
    FrozenLocalMLPOverlay,
    LocalStateEncoder,
    expected_ev_rank,
    fit_hierarchical_ev_calibrator,
    fit_local_state_encoder,
    hierarchical_ev_calibrator_payload,
    predict_hierarchical_ev,
    predict_local_state_encoder,
)
from extreme_price_movements.lgbm_pipeline import (  # noqa: E402
    _recent_feature_coverage_survivors,
)
from scripts.run_meta_market_state_threshold_calibration import (  # noqa: E402
    _feature_block,
    _merge_observable_context,
    _num,
)
from scripts.run_meta_residual_extreme_local_champion_overlay import (  # noqa: E402
    KEYS,
    PARENT,
    _feature_catalog,
    _fit_references,
    _load_joined,
    _rank_for_params,
)


ARMS: tuple[EncoderArm, ...] = ("ae_gmm", "mlp_gmm", "mlp_direct", "ae_mlp_gmm")
PREDECESSOR_ID = (
    "meta_residual_extreme_local_champion_overlay_ooftrain_tieaware_downonly_"
    "20260712_v9::forced_local_tail_0.950"
)
PREDECESSOR_ARTIFACT = Path(
    "data_perp/reports/"
    "meta_residual_extreme_local_champion_overlay_ooftrain_tieaware_downonly_"
    "20260712_v9"
)
COMPOSITE_POLICY_ID = "meta_residual_v9_tail95_market_state_mlp_hier_ev_v1"
BLACKLISTED_SIDE_ARCHETYPES = ("long||long_dirtyavoid_sparse_questionable",)

# Stable production fallback. An explicitly promoted HPO JSON takes precedence.
DEFAULT_MLP_PARAMS: dict[str, Any] = {
    "hidden_layer_sizes": (40, 20, 10),
    "alpha": 0.19352885579109644,
    "noise_std": 0.037922992076789966,
    "learning_rate_init": 0.0003285878756092489,
    "batch_size": 1024,
    "max_iter": 120,
    "tol": 0.00013979372134149541,
    "hit_target_weight": 0.7099587519866526,
}

MARKET_STATE_TOKENS: tuple[str, ...] = (
    "market_", "mkt_", "xasset_", "xs_mean__", "xs_std__", "xs_dispersion__",
    "asset_minus_mkt_", "funding", "oi_", "breadth", "shock", "entropy",
    "volatility", "volume_", "recovery", "liquidation", "deleveraging",
    "short_cover", "flush_", "pc1_", "pairwise_corr", "downside_corr",
)
FORBIDDEN_FEATURE_TOKENS: tuple[str, ...] = (
    "target", "label", "future", "oracle", "realized_", "bad_mae", "timeout",
    "full_stop", "exec_margin", "ev_after_1pct", "clean_exec", "dirty_positive",
)


def _is_allowed_observable_feature(name: str) -> bool:
    key = str(name).lower()
    if key.startswith("regime_lgbm_leaf_"):
        return True
    return not any(token in key for token in FORBIDDEN_FEATURE_TOKENS)


def _same_identity_rows(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    """Compare fixed parent identity without treating dtype coercion as mutation."""
    if len(left) != len(right):
        return False
    for name in KEYS:
        if name == "__ts__":
            lhs = pd.to_datetime(left[name], utc=True, errors="coerce")
            rhs = pd.to_datetime(right[name], utc=True, errors="coerce")
        else:
            lhs = left[name].astype("string").fillna("__missing__")
            rhs = right[name].astype("string").fillna("__missing__")
        if not lhs.reset_index(drop=True).equals(rhs.reset_index(drop=True)):
            return False
    return True

RELIABILITY_CONTEXT_TOKENS: tuple[str, ...] = (
    "uncertainty", "prob_std", "prob_range", "vote_entropy", "vote_margin",
    "leaf_", "rare_leaf", "support_", "support_count", "support_log",
    "ood", "out_of_distribution", "reconstruction", "mahalanobis",
    "drift", "feature_drift", "prediction_drift", "inference_drift", "contribution_drift",
    "margin_to_cutoff", "rank_margin", "score_margin",
)

# Explicit contract for uncertainty exported by the frozen base model.  These
# are all decision-time diagnostics: no realized returns, realized error, or
# same-row calibration residual is admissible here.
BASE_PREDICTIVE_UNCERTAINTY_FEATURES: tuple[str, ...] = (
    "base_lgbm_prob_std",
    "base_lgbm_prob_range",
    "base_lgbm_raw_score_std",
    "base_lgbm_raw_score_range",
    "base_lgbm_prob_uncertainty",
    "base_lgbm_entropy",
    "base_lgbm_variance_proxy",
    "base_lgbm_tree_vote_entropy",
    "base_lgbm_tree_vote_margin",
    "base_lgbm_tree_vote_top_gap",
    "base_lgbm_rare_leaf_fraction",
    "base_lgbm_leaf_train_freq_p10",
    "base_lgbm_leaf_surprisal_p90",
    "base_lgbm_leaf_low_freq_fraction",
    "base_lgbm_leaf_model_space_distance_mean",
    "base_lgbm_uncertainty_score",
    "base_lgbm_inference_drift_score",
    "base_lgbm_mahalanobis_mean_shift",
)

# These are pre-MLP quantities produced by the parent meta score.  They are
# safe inputs to the local MLP; internal diagnostics of the MLP being fitted
# are not, because they would make train/inference schemas diverge.
META_PARENT_RELIABILITY_FEATURES: tuple[str, ...] = (
    "hit_probability",
    "policy_parent_rank",
    "meta_hit_probability_uncertainty_p1mp",
    "meta_parent_rank_uncertainty_p1mp",
    "meta_parent_rank_margin_top10",
    "base_margin_to_cutoff",
    "base_margin_to_cutoff_z",
    "meta_parent_rank_local_top10_margin",
    "meta_hit_probability_local_top10_margin",
    "meta_parent_reliability_local_support_log1p",
)


def _feature_family_name(feature: str) -> str:
    name = str(feature).lower()
    if any(token in name for token in ("leaf", "support", "drift", "uncertainty", "ood", "mahal", "reconstruction")):
        return "model_reliability"
    if any(token in name for token in ("gmm", "posterior", "entropy", "aegmm", "cluster")):
        return "latent_state"
    if any(token in name for token in ("market_", "mkt_", "breadth", "xasset", "pc1", "pairwise")):
        return "market_cross_asset"
    if any(token in name for token in ("funding", "oi_", "open_interest", "orderbook", "spread", "liquidity")):
        return "derivatives_liquidity"
    if any(token in name for token in ("volume", "volatility", "ret_", "trend", "momentum", "recovery", "shock")):
        return "price_volume_state"
    return "other"


def _time_codes(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    if "__week_code__" in frame and "__month_code__" in frame:
        return (
            frame["__week_code__"].to_numpy(dtype=np.int32, copy=False),
            frame["__month_code__"].to_numpy(dtype=np.int16, copy=False),
        )
    ts = pd.to_datetime(frame["__ts__"], utc=True, errors="coerce")
    day = ts.dt.floor("D")
    week = day - pd.to_timedelta(day.dt.weekday.to_numpy(), unit="D")
    return (
        pd.factorize(week, sort=True)[0].astype(np.int32),
        pd.factorize(ts.dt.strftime("%Y-%m"), sort=True)[0].astype(np.int16),
    )


def _group_mean(values: np.ndarray, codes: np.ndarray, mask: np.ndarray) -> np.ndarray:
    c = codes[mask]
    v = values[mask]
    valid = c >= 0
    if not valid.any():
        return np.array([np.nan])
    c, v = c[valid], v[valid]
    sums = np.bincount(c, weights=v)
    counts = np.bincount(c)
    return sums[counts > 0] / counts[counts > 0]


def _top_budget(score: np.ndarray, budget: int) -> np.ndarray:
    mask = np.zeros(len(score), dtype=bool)
    if budget <= 0:
        return mask
    values = np.nan_to_num(score, nan=-np.inf)
    finite = np.isfinite(values)
    available = int(finite.sum())
    budget = min(int(budget), available)
    if budget <= 0:
        return mask
    if budget >= available:
        mask[finite] = True
        return mask
    finite_values = values[finite]
    cutoff = float(np.partition(finite_values, available - budget)[available - budget])
    above = np.flatnonzero(finite & (values > cutoff))
    mask[above] = True
    remaining = budget - len(above)
    if remaining > 0:
        # Preserve the old stable-sort tie contract without sorting the entire
        # candidate universe.
        tied = np.flatnonzero(finite & (values == cutoff))
        mask[tied[:remaining]] = True
    return mask


def _economic_metrics(
    frame: pd.DataFrame,
    score: np.ndarray,
    budget: int,
    *,
    target_activity: int,
) -> dict[str, float]:
    selected = _top_budget(score, budget)
    ev = _num(frame, "ev_after_1pct")
    week, month = _time_codes(frame)
    weekly = _group_mean(ev, week, selected)
    monthly = _group_mean(ev, month, selected)
    finite_weekly = weekly[np.isfinite(weekly)]
    if not len(finite_weekly):
        finite_weekly = np.array([-1.0])
    return {
        "selected_rows": float(selected.sum()),
        "mean_net_ev_top10": float(np.mean(ev[selected])),
        "worst_week_net_ev_top10": float(np.min(finite_weekly)),
        "q10_week_net_ev_top10": float(np.quantile(finite_weekly, 0.10)),
        "q20_week_net_ev_top10": float(np.quantile(finite_weekly, 0.20)),
        "q30_week_net_ev_top10": float(np.quantile(finite_weekly, 0.30)),
        "worst_month_net_ev_top10": float(np.nanmin(monthly)),
        "activity_deviation": float(abs(selected.sum() - target_activity) / max(target_activity, 1)),
    }


def _monthly_rank_contract_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    """Compare all rank contracts at the parent's causal monthly activity."""
    timestamps = pd.to_datetime(frame["__ts__"], utc=True)
    month_ids = timestamps.dt.strftime("%Y-%m")
    arms = {
        "parent_95": "policy_parent_rank",
        "mlp_direct": "rank_mlp_direct",
        "mlp_direct_ev_calibrated": "expected_ev_rank_score",
    }
    rows: list[dict[str, Any]] = []
    for month_id in sorted(month_ids.unique()):
        month_mask = month_ids.eq(month_id).to_numpy()
        month = frame.loc[month_mask]
        month_timestamps = timestamps.loc[month_mask]
        parent_rank = _num(month, "policy_parent_rank")
        budget = int(np.sum(parent_rank >= 0.90))
        ev = _num(month, "ev_after_1pct")
        for arm, score_column in arms.items():
            score = _num(month, score_column)
            selected = _top_budget(score, budget)
            selected_ev = ev[selected]
            rows.append(
                {
                    "month": month_id,
                    "arm": arm,
                    "source_rows": int(len(month)),
                    "source_days": int(month_timestamps.dt.floor("D").nunique()),
                    "folds": ",".join(
                        str(int(value))
                        for value in sorted(
                            pd.to_numeric(month["__fold__"], errors="coerce")
                            .dropna()
                            .unique()
                        )
                    ),
                    "target_activity": budget,
                    "selected_rows": int(selected.sum()),
                    "mean_net_ev_after_1pct": (
                        float(np.mean(selected_ev)) if len(selected_ev) else np.nan
                    ),
                    "sum_net_ev_after_1pct": (
                        float(np.sum(selected_ev)) if len(selected_ev) else np.nan
                    ),
                    "positive_ev_rate": (
                        float(np.mean(selected_ev > 0.0))
                        if len(selected_ev)
                        else np.nan
                    ),
                    "clean_exec_rate": (
                        float(np.mean(_num(month, "clean_exec")[selected]))
                        if selected.any()
                        else np.nan
                    ),
                    "dirty_positive_rate": (
                        float(np.mean(_num(month, "dirty_positive")[selected]))
                        if selected.any()
                        else np.nan
                    ),
                    "full_path_bad_mae_rate": (
                        float(
                            np.mean(_num(month, "full_path_bad_mae_1r")[selected])
                        )
                        if selected.any()
                        else np.nan
                    ),
                    "timeout_rate": (
                        float(np.mean(_num(month, "timeout")[selected]))
                        if selected.any()
                        else np.nan
                    ),
                    "missing_score_rows": int(np.sum(~np.isfinite(score))),
                }
            )
    return pd.DataFrame(rows)


def _objective(metric: dict[str, float], baseline: dict[str, float]) -> float:
    mean_gain = metric["mean_net_ev_top10"] - baseline["mean_net_ev_top10"]
    allowed_degradation = max(0.0, mean_gain / 5.0)
    # A degradation is acceptable only when the average-EV gain is at least 5x
    # larger. Negative/flat average EV receives no downside allowance.
    if metric["worst_week_net_ev_top10"] < baseline["worst_week_net_ev_top10"] - allowed_degradation:
        return -1e9
    if metric["worst_month_net_ev_top10"] < baseline["worst_month_net_ev_top10"] - allowed_degradation:
        return -1e9
    return float(
        metric["mean_net_ev_top10"]
        + 0.25
        * (
            metric["worst_week_net_ev_top10"]
            + metric["q10_week_net_ev_top10"]
            + metric["q20_week_net_ev_top10"]
            + metric["q30_week_net_ev_top10"]
        )
        - 0.0025 * metric["activity_deviation"]
    )


def _tune_ev_mapping(
    folds: pd.DataFrame, score: np.ndarray
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Tune hierarchical EV shrinkage on fold 0 -> fold 1 only."""
    train_mask = folds["__fold__"].to_numpy() == 0
    valid_mask = folds["__fold__"].to_numpy() == 1
    train = folds.loc[train_mask]
    valid = folds.loc[valid_mask]
    valid_score = score[valid_mask]
    budget = int(np.sum(_num(valid, "policy_parent_rank") >= 0.90))
    baseline = _economic_metrics(
        valid, _num(valid, "policy_parent_rank"), budget, target_activity=budget
    )
    rows: list[dict[str, Any]] = []
    for shrink_rows in (500.0, 1_000.0, 2_000.0, 4_000.0, 8_000.0):
        for local_cap in (0.35, 0.50, 0.65, 0.80):
            for min_rows in (300, 600, 1_200):
                for top10_weight in (2.0, 4.0, 6.0):
                    for rank_blend in (0.0, 0.25, 0.50, 0.75, 1.0):
                        calibrator = fit_hierarchical_ev_calibrator(
                            train,
                            score[train_mask],
                            _num(train, "ev_after_1pct"),
                            shrink_rows=shrink_rows,
                            min_local_rows=min_rows,
                            local_weight_cap=local_cap,
                            tail_weight_top10=top10_weight,
                            rank_blend=rank_blend,
                        )
                        expected = predict_hierarchical_ev(calibrator, valid, valid_score)
                        rank = expected_ev_rank(calibrator, expected, valid_score)
                        metric = _economic_metrics(
                            valid, rank, budget, target_activity=budget
                        )
                        rows.append(
                            {
                                "shrink_rows": shrink_rows,
                                "local_weight_cap": local_cap,
                                "min_local_rows": min_rows,
                                "tail_weight_top10": top10_weight,
                                "rank_blend": rank_blend,
                                **metric,
                                "objective": _objective(metric, baseline),
                            }
                        )
    search = pd.DataFrame(rows).sort_values("objective", ascending=False, kind="stable")
    best = search.iloc[0].to_dict()
    return best, search


def _tune_mlp_hyperparameters(
    history: pd.DataFrame,
    candidates: list[str],
    ae_cols: list[str],
    frozen_features: dict[tuple[str, str], list[str]],
    *,
    min_rows: int,
    seed: int,
    n_trials: int,
    valid_start: pd.Timestamp = pd.Timestamp("2026-02-01", tz="UTC"),
    valid_end: pd.Timestamp = pd.Timestamp("2026-03-01", tz="UTC"),
    scoring_overlay_params: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Tune MLP regularization/architecture on a chronological OOF fold."""
    train = history.loc[history["__ts__"].lt(valid_start)]
    valid = history.loc[
        history["__ts__"].ge(valid_start) & history["__ts__"].lt(valid_end)
    ].copy()
    budget = int(np.sum(_num(valid, "policy_parent_rank") >= 0.90))
    baseline = _economic_metrics(
        valid, _num(valid, "policy_parent_rank"), budget, target_activity=budget
    )
    trial_rows: list[dict[str, Any]] = []
    architectures = {
        "32_16_8": (32, 16, 8),
        "40_20_10": (40, 20, 10),
        "48_24_12": (48, 24, 12),
        "48_24_12_8": (48, 24, 12, 8),
    }
    fixed_overlay = dict(
        scoring_overlay_params
        or {
            "alpha": 0.005,
            "cap": 0.01,
            "posterior_power": 2.0,
            "local_alpha": {},
        }
    )

    def trial_objective(trial: optuna.Trial) -> float:
        architecture = trial.suggest_categorical("architecture", list(architectures))
        params = {
            "hidden_layer_sizes": architectures[architecture],
            "alpha": trial.suggest_float("alpha", 0.08, 0.8, log=True),
            "noise_std": trial.suggest_float("noise_std", 0.02, 0.10),
            "learning_rate_init": trial.suggest_float(
                "learning_rate_init", 1.5e-4, 8e-4, log=True
            ),
            "batch_size": trial.suggest_categorical("batch_size", [256, 512, 1024]),
            "max_iter": trial.suggest_int("max_iter", 80, 180, step=20),
            "tol": trial.suggest_float("tol", 5e-5, 5e-4, log=True),
            "hit_target_weight": trial.suggest_float(
                "hit_target_weight", 0.25, 1.0
            ),
        }
        models = _fit_models(
            train,
            "mlp_direct",
            candidates,
            ae_cols,
            min_rows,
            seed,
            frozen_features,
            mlp_params=params,
            max_local_fit_rows=45_000,
        )
        correction, confidence, _ = _predict_models(valid, models, ae_cols)
        score = _overlay_score(valid, correction, confidence, fixed_overlay)
        metric = _economic_metrics(valid, score, budget, target_activity=budget)
        value = _objective(metric, baseline)
        # Prefer the smaller/stronger-regularized solution when economics tie.
        complexity = sum(architectures[architecture]) / 10_000_000.0
        value -= complexity
        trial_rows.append(
            {
                "trial": trial.number,
                **params,
                "architecture": architecture,
                "models": len(models),
                **metric,
                "objective": value,
            }
        )
        return value

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=min(8, n_trials)),
    )
    default_architecture = next(
        name
        for name, hidden in architectures.items()
        if tuple(hidden) == tuple(DEFAULT_MLP_PARAMS["hidden_layer_sizes"])
    )
    study.enqueue_trial(
        {
            "architecture": default_architecture,
            "alpha": float(DEFAULT_MLP_PARAMS["alpha"]),
            "noise_std": float(DEFAULT_MLP_PARAMS["noise_std"]),
            "learning_rate_init": float(DEFAULT_MLP_PARAMS["learning_rate_init"]),
            "batch_size": int(DEFAULT_MLP_PARAMS["batch_size"]),
            "max_iter": int(DEFAULT_MLP_PARAMS["max_iter"]),
            "tol": float(DEFAULT_MLP_PARAMS["tol"]),
            "hit_target_weight": float(DEFAULT_MLP_PARAMS["hit_target_weight"]),
        }
    )
    study.optimize(
        trial_objective, n_trials=max(1, int(n_trials)), show_progress_bar=False
    )
    best = dict(study.best_trial.params)
    best["hidden_layer_sizes"] = architectures[str(best.pop("architecture"))]
    return best, pd.DataFrame(trial_rows).sort_values(
        "objective", ascending=False, kind="stable"
    )


def _load_mlp_params(path: Path | None) -> tuple[dict[str, Any], str]:
    if path is None:
        return dict(DEFAULT_MLP_PARAMS), "checked_in_default"
    payload = json.loads(path.read_text())
    params = payload.get("mlp_params", payload)
    if not isinstance(params, dict):
        raise ValueError(f"invalid MLP parameter payload: {path}")
    result = dict(DEFAULT_MLP_PARAMS)
    result.update(params)
    result["hidden_layer_sizes"] = tuple(result["hidden_layer_sizes"])
    return result, str(path)


def _write_mlp_params(path: Path, params: dict[str, Any], *, source: str) -> None:
    payload = {
        "schema": "market_state_mlp_params_v1",
        "source": source,
        "mlp_params": {
            key: list(value) if isinstance(value, tuple) else value
            for key, value in params.items()
        },
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _export_composite_policy(
    output_dir: Path,
    feature_frame: pd.DataFrame,
    models: list[LocalStateEncoder],
    params: dict[str, Any],
    ev_calibrator: Any,
    predecessor_references: dict[tuple[Any, ...], list[tuple[str, float, np.ndarray]]],
) -> Path:
    model_dir = output_dir / "policy_models"
    model_dir.mkdir(parents=True, exist_ok=True)
    effects: list[dict[str, Any]] = []
    handoff_features: list[str] = []
    for index, model in enumerate(models):
        alpha = float(
            params.get("local_alpha", {}).get(
                f"{model.side}||{model.archetype}", params["alpha"]
            )
        )
        adverse = predecessor_references.get(
            (model.side, model.archetype, "adverse"), []
        )
        predecessor = None
        if adverse:
            predecessor = FrozenLocalExtremeTailOverlay(
                source_score_col="calibrated_score",
                features=[item[0] for item in adverse],
                directions=np.asarray([item[1] for item in adverse], dtype=np.float32),
                references=[item[2] for item in adverse],
                threshold=0.95,
                alpha_down=0.02,
            )
        wrapper = FrozenCompositeLocalOverlay(
            predecessor=predecessor,
            mlp_overlay=FrozenLocalMLPOverlay(model, alpha, float(params["cap"])),
        )
        model_path = model_dir / f"local_overlay_{index:02d}.joblib"
        joblib.dump(wrapper, model_path)
        feature_cols = [c for c in model.features if c != "policy_parent_rank"]
        if predecessor is not None:
            feature_cols.extend(predecessor.features)
        feature_cols.extend(["calibrated_score", "hit_probability"])
        handoff_features.extend(feature_cols)
        effects.append(
            {
                "shape": "sklearn_pickle",
                "side_name": model.side,
                "archetype_policy_key": model.archetype,
                "model_path": str(model_path.relative_to(output_dir)),
                "feature_cols": list(dict.fromkeys(feature_cols)),
                "fill_values": {},
            }
        )
    handoff_cols = [
        c for c in [*KEYS, *dict.fromkeys(handoff_features)]
        if c in feature_frame.columns and c not in {"calibrated_score"}
    ]
    handoff_path = output_dir / "composite_policy_feature_handoff.parquet"
    feature_frame[handoff_cols].to_parquet(
        handoff_path, index=False, compression="zstd"
    )
    payload = {
        "schema": "regime_ev_calibration_v2",
        "artifact_id": COMPOSITE_POLICY_ID,
        "policy_id": COMPOSITE_POLICY_ID,
        "policy_name": COMPOSITE_POLICY_ID,
        "source_score_col": "calibrated_score",
        "adjusted_score_col": "score_regime_calibrated",
        "risk_score_col": "regime_ev_risk_score",
        "effect_count_col": "regime_ev_effect_count",
        "score_application": {
            "mode": "additive", "scale": 1.0,
            "max_upscore": float(params["cap"]),
            "max_downscore": 0.02 + float(params["cap"]),
        },
        "risk_cap_positive": 0.02 + float(params["cap"]),
        "risk_cap_negative": float(params["cap"]),
        "effects": effects,
        "strict_required_features": True,
        "feature_handoff_path": handoff_path.name,
        "expected_ev_mapping": hierarchical_ev_calibrator_payload(ev_calibrator),
        "expected_ev_col": "expected_net_ev_after_1pct",
        "expected_ev_rank_col": "expected_ev_rank_score",
        "blacklisted_side_archetypes": list(BLACKLISTED_SIDE_ARCHETYPES),
        "predecessor": PREDECESSOR_ID,
        "chain": ["v9_tail95_downonly", "market_state_mlp", "hierarchical_expected_ev"],
    }
    path = output_dir / "composite_policy_regime_ev_calibration.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def _monthly_specs(start: pd.Timestamp, end: pd.Timestamp) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    boundaries = list(pd.date_range(start, end, freq="MS", tz="UTC"))
    if not boundaries or boundaries[0] != start:
        boundaries.insert(0, start)
    if boundaries[-1] != end:
        boundaries.append(end)
    return list(zip(boundaries[:-1], boundaries[1:]))


def _run_expanding_walkforward(
    *,
    all_rows: pd.DataFrame,
    candidates: list[str],
    ae_cols: list[str],
    frozen_features: dict[tuple[str, str], list[str]],
    mlp_params: dict[str, Any],
    min_rows: int,
    seed: int,
    tuning_start: pd.Timestamp,
    policy_start: pd.Timestamp,
    end: pd.Timestamp,
    output_dir: Path,
    catalog: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Generate monthly expanding-window OOS predictions and final bundle."""
    all_rows = all_rows.sort_values("__ts__", kind="stable").reset_index(drop=True)
    specs = _monthly_specs(tuning_start, end)
    oof_parts: list[pd.DataFrame] = []
    oof_corr: list[np.ndarray] = []
    oof_conf: list[np.ndarray] = []
    fold_manifest: list[dict[str, Any]] = []
    for fold_no, (valid_start, valid_end) in enumerate(specs):
        train = all_rows.loc[all_rows["__ts__"].lt(valid_start)]
        valid = all_rows.loc[
            all_rows["__ts__"].ge(valid_start)
            & all_rows["__ts__"].lt(valid_end)
        ].copy()
        if valid.empty:
            continue
        models = _fit_models(
            train,
            "mlp_direct",
            candidates,
            ae_cols,
            min_rows,
            seed + fold_no * 1000,
            frozen_features,
            mlp_params=mlp_params,
        )
        correction, confidence, entropy = _predict_models(valid, models, ae_cols)
        valid["__fold__"] = fold_no
        valid["__valid_start__"] = valid_start
        valid["state_posterior_entropy_mlp_direct"] = entropy
        oof_parts.append(valid)
        oof_corr.append(correction)
        oof_conf.append(confidence)
        fold_manifest.append(
            {
                "fold": fold_no,
                "train_start": str(train["__ts__"].min()),
                "train_end": str(valid_start),
                "train_rows": int(len(train)),
                "valid_start": str(valid_start),
                "valid_end": str(valid_end),
                "valid_rows": int(len(valid)),
                "local_models": int(len(models)),
            }
        )
        print(
            f"Walk-forward fold={fold_no} train={len(train):,} "
            f"valid={valid_start:%Y-%m} rows={len(valid):,} models={len(models)}",
            flush=True,
        )
    folds = pd.concat(oof_parts, ignore_index=True)
    correction = np.concatenate(oof_corr)
    confidence = np.concatenate(oof_conf)
    # The MLP is refit on every growing window, so its correction scale can
    # change between folds. Calibrate the overlay on prior OOF folds for each
    # validation fold instead of reusing one February/March strength forever.
    # Fold zero has no earlier OOF evidence and therefore remains a no-op.
    all_rank = _num(folds, "policy_parent_rank").astype(np.float32, copy=True)
    overlay_search_parts: list[pd.DataFrame] = []
    overlay_params_by_fold: dict[int, dict[str, Any]] = {}
    fold_ids = sorted(int(value) for value in folds["__fold__"].unique())
    for fold_id in fold_ids:
        current_mask = folds["__fold__"].to_numpy() == fold_id
        prior_mask = folds["__fold__"].to_numpy() < fold_id
        if not prior_mask.any():
            overlay_params_by_fold[fold_id] = {
                "alpha": 0.0,
                "cap": 0.01,
                "posterior_power": 2.0,
                "local_alpha": {},
                "source": "no_prior_oof_fold",
            }
            continue
        prior = folds.loc[prior_mask].copy()
        params, search = _tune_overlay(
            prior, correction[prior_mask], confidence[prior_mask]
        )
        params["source"] = "expanding_prior_oof_folds"
        overlay_params_by_fold[fold_id] = params
        all_rank[current_mask] = _overlay_score(
            folds.loc[current_mask],
            correction[current_mask],
            confidence[current_mask],
            params,
        )
        search.insert(0, "apply_fold", fold_id)
        search.insert(1, "prior_fold_count", int(fold_id))
        overlay_search_parts.append(search)
    if overlay_search_parts:
        pd.concat(overlay_search_parts, ignore_index=True).to_csv(
            output_dir / "mlp_direct_overlay_search.csv", index=False
        )
    folds["rank_mlp_direct"] = all_rank
    folds["state_ev_correction_mlp_direct"] = correction
    folds["state_posterior_confidence_mlp_direct"] = confidence
    rank_history_columns = [
        column
        for column in (
            "__ts__",
            "__symbol__",
            "side_name",
            "archetype_policy_key",
            "policy_parent_rank",
            "rank_mlp_direct",
            "ev_after_1pct",
            "clean_exec",
            "dirty_positive",
            "full_path_bad_mae_1r",
            "timeout",
            "__fold__",
            "state_ev_correction_mlp_direct",
            "state_posterior_confidence_mlp_direct",
        )
        if column in folds
    ]
    folds.loc[:, rank_history_columns].to_parquet(
        output_dir / "walkforward_rank_history.parquet",
        index=False,
        compression="zstd",
    )

    scored_parts: list[pd.DataFrame] = []
    policy_mask = folds["__ts__"].ge(policy_start).to_numpy()
    for valid_start, valid_end in _monthly_specs(policy_start, end):
        current = (
            folds["__ts__"].ge(valid_start) & folds["__ts__"].lt(valid_end)
        ).to_numpy()
        prior = folds["__ts__"].lt(valid_start).to_numpy()
        if not current.any() or prior.sum() < 1_000:
            continue
        calibrator = fit_hierarchical_ev_calibrator(
            folds.loc[prior],
            all_rank[prior],
            _num(folds.loc[prior], "ev_after_1pct"),
            shrink_rows=2_000.0,
            min_local_rows=600,
            local_weight_cap=0.50,
            tail_weight_top10=4.0,
            # Keep the hierarchical map as a common EV unit for sizing and
            # filtering. The OOS MLP ordering is stronger than pure EV-map
            # percentile ordering, so ranking remains on the raw MLP score.
            rank_blend=0.0,
        )
        current_frame = folds.loc[current].copy()
        expected = predict_hierarchical_ev(
            calibrator, current_frame, all_rank[current]
        )
        current_frame["expected_net_ev_after_1pct_mlp_direct"] = expected
        current_frame["expected_ev_rank_score"] = expected_ev_rank(
            calibrator, expected, all_rank[current]
        )
        scored_parts.append(current_frame)
    scored = pd.concat(scored_parts, ignore_index=True)
    final_calibrator = fit_hierarchical_ev_calibrator(
        folds,
        all_rank,
        _num(folds, "ev_after_1pct"),
        shrink_rows=2_000.0,
        min_local_rows=600,
        local_weight_cap=0.50,
        tail_weight_top10=4.0,
        rank_blend=0.0,
    )
    final_models = _fit_models(
        all_rows,
        "mlp_direct",
        candidates,
        ae_cols,
        min_rows,
        seed + 90_000,
        frozen_features,
        mlp_params=mlp_params,
    )
    predecessor_references = _fit_references(all_rows, catalog, 1)
    # The forward bundle is calibrated on every completed OOF fold. This is
    # separate from the fold-local parameters that generated reported OOS
    # ranks and is never used retroactively.
    overlay_params, final_overlay_search = _tune_overlay(
        folds, correction, confidence
    )
    final_overlay_search.to_csv(
        output_dir / "mlp_direct_final_overlay_search.csv", index=False
    )
    policy_path = _export_composite_policy(
        output_dir,
        scored,
        final_models,
        overlay_params,
        final_calibrator,
        predecessor_references,
    )
    budget = int(np.sum(_num(scored, "policy_parent_rank") >= 0.90))
    baseline = _economic_metrics(
        scored, _num(scored, "policy_parent_rank"), budget, target_activity=budget
    )
    raw_metric = _economic_metrics(
        scored, _num(scored, "rank_mlp_direct"), budget, target_activity=budget
    )
    ev_metric = _economic_metrics(
        scored, _num(scored, "expected_ev_rank_score"), budget, target_activity=budget
    )
    metrics = pd.DataFrame(
        [
            {"arm": "parent_95", **baseline},
            {"arm": "mlp_direct", **raw_metric},
            {"arm": "mlp_direct_ev_calibrated", **ev_metric},
        ]
    )
    artifacts = {
        "params": overlay_params,
        "models": final_models,
        "ev_calibrator": final_calibrator,
        "mlp_hpo_params": mlp_params,
        "policy_path": str(policy_path),
        "fold_manifest": fold_manifest,
        "overlay_params_by_fold": overlay_params_by_fold,
    }
    return scored, metrics, artifacts


def _ae_features(frame: pd.DataFrame) -> list[str]:
    tokens = ("aegmm_ae_", "aegmm_gmm_", "reconstruction", "mahal", "posterior", "entropy")
    return [
        c for c in frame
        if c.startswith("resid_event_") and any(t in c.lower() for t in tokens)
        and pd.api.types.is_numeric_dtype(frame[c])
    ]


def _merge_market_state_features(frame: pd.DataFrame, source: Path) -> pd.DataFrame:
    """Add only causal market/context candidates; never outcome descriptors."""
    if not source.exists():
        return frame
    names = pq.read_schema(source).names
    wanted = [
        c for c in names
        if c not in frame.columns
        and any(
            token in c.lower()
            for token in (*MARKET_STATE_TOKENS, *RELIABILITY_CONTEXT_TOKENS)
        )
        and _is_allowed_observable_feature(c)
        and not c.lower().endswith(("_x", "_y"))
    ]
    # The broad source predates the normalized archetype_policy_key column.  Its
    # observable market features are row-identical across archetype naming, so
    # join on timestamp/symbol/side and retain the fixed parent's archetype.
    keys = [c for c in ("__ts__", "__symbol__", "side_name") if c in names]
    if len(keys) != 3 or not wanted:
        return frame
    extra = pd.read_parquet(source, columns=[*keys, *wanted])
    extra["__ts__"] = pd.to_datetime(extra["__ts__"], utc=True, errors="coerce")
    extra = extra.drop_duplicates(keys, keep="last")
    return frame.merge(extra, on=keys, how="left", validate="many_to_one")


def _merge_meta_oof_observable_features(
    frame: pd.DataFrame, shard_dir: Path
) -> pd.DataFrame:
    """Project inference-available reliability/context columns from meta OOF shards."""
    shard_paths = sorted(shard_dir.glob("*.parquet"))
    if not shard_paths or frame.empty:
        return frame
    schema_by_path = {path: set(pq.read_schema(path).names) for path in shard_paths}
    schema_names = set().union(*schema_by_path.values())
    wanted = sorted(
        col
        for col in schema_names
        if col not in frame.columns
        and (
            col == "score_meta_base_soft_label"
            or any(
                token in col.lower()
                for token in (*MARKET_STATE_TOKENS, *RELIABILITY_CONTEXT_TOKENS)
            )
        )
        and (
            col == "score_meta_base_soft_label"
            or _is_allowed_observable_feature(col)
        )
    )
    keys = [col for col in KEYS if col in schema_names]
    if not wanted or len(keys) != len(KEYS):
        return frame
    start, end = frame["__ts__"].min(), frame["__ts__"].max()
    parts: list[pd.DataFrame] = []
    for path in shard_paths:
        available = schema_by_path[path]
        local_keys = [col for col in keys if col in available]
        local_wanted = [col for col in wanted if col in available]
        if len(local_keys) != len(keys) or not local_wanted:
            continue
        part = pd.read_parquet(path, columns=[*local_keys, *local_wanted])
        part["__ts__"] = pd.to_datetime(part["__ts__"], utc=True, errors="coerce")
        part = part.loc[part["__ts__"].between(start, end, inclusive="both")]
        if not part.empty:
            parts.append(part)
    if not parts:
        return frame
    context = pd.concat(parts, ignore_index=True, copy=False).drop_duplicates(
        keys, keep="last"
    )
    for col in wanted:
        if pd.api.types.is_float_dtype(context[col]):
            context[col] = pd.to_numeric(context[col], downcast="float")
    return frame.merge(context, on=keys, how="left", validate="one_to_one")


def _merge_base_predictive_uncertainty(
    frame: pd.DataFrame,
    source: Path | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Join the base-model uncertainty contract without leaking outcomes.

    The base handoff may be archetype-agnostic, so use timestamp/symbol/side
    keys.  Values must have been emitted at base prediction time; this function
    does not synthesize uncertainty from realized outcomes.
    """
    audit: dict[str, Any] = {
        "source": str(source) if source else "",
        "requested": list(BASE_PREDICTIVE_UNCERTAINTY_FEATURES),
        "available": [],
        "coverage": {},
    }
    if source is None or not source.exists() or frame.empty:
        return frame, audit
    names = set(pq.read_schema(source).names)
    keys = [key for key in ("__ts__", "__symbol__", "side_name") if key in names]
    available = [key for key in BASE_PREDICTIVE_UNCERTAINTY_FEATURES if key in names]
    audit["available"] = available
    if len(keys) != 3 or not available:
        return frame, audit
    extra = pd.read_parquet(source, columns=[*keys, *available])
    extra["__ts__"] = pd.to_datetime(extra["__ts__"], utc=True, errors="coerce")
    extra = extra.drop_duplicates(keys, keep="last")
    overlap = [key for key in available if key in frame.columns]
    base = frame.drop(columns=overlap) if overlap else frame
    merged = base.merge(extra, on=keys, how="left", validate="many_to_one", copy=False)
    for key in available:
        values = pd.to_numeric(merged[key], errors="coerce")
        audit["coverage"][key] = float(values.notna().mean())
        merged[key] = values.astype(np.float32)
    return merged, audit


def _merge_projected_context(
    frame: pd.DataFrame,
    sources: list[tuple[Path, bool]],
    allowed: set[str],
) -> pd.DataFrame:
    """Low-memory context merge used when feature selection is already frozen."""
    out = frame
    for source, archetype_aligned in sources:
        if not source.exists():
            continue
        names = set(pq.read_schema(source).names)
        wanted = sorted((allowed & names) - set(out.columns))
        override = sorted(allowed & names & set(out.columns))
        if override:
            out = out.drop(columns=override)
            wanted = sorted(set(wanted) | set(override))
        key_contract = KEYS if archetype_aligned else KEYS[:3]
        keys = [c for c in key_contract if c in names]
        if len(keys) != len(key_contract) or not wanted:
            continue
        extra = pd.read_parquet(source, columns=[*keys, *wanted])
        extra["__ts__"] = pd.to_datetime(extra["__ts__"], utc=True, errors="coerce")
        extra = extra.drop_duplicates(keys, keep="last")
        out = out.merge(
            extra,
            on=keys,
            how="left",
            validate="one_to_one" if archetype_aligned else "many_to_one",
            copy=False,
        )
    return out


def _add_observable_reliability_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add inference-available uncertainty proxies before feature screening."""
    out = frame.copy(deep=False)
    hit = np.clip(_num(out, "hit_probability", 0.5), 0.0, 1.0)
    rank = np.clip(_num(out, "policy_parent_rank", 0.5), 0.0, 1.0)
    out["meta_hit_probability_uncertainty_p1mp"] = (hit * (1.0 - hit)).astype(np.float32)
    out["meta_parent_rank_uncertainty_p1mp"] = (rank * (1.0 - rank)).astype(np.float32)
    out["meta_parent_rank_margin_top10"] = (rank - 0.90).astype(np.float32)
    return out


def _add_side_archetype_parent_reliability(
    train: pd.DataFrame,
    target: pd.DataFrame,
) -> pd.DataFrame:
    """Apply frozen side x archetype parent-score references to ``target``."""
    out = target.copy(deep=False)
    group_cols = ["side_name", "archetype_policy_key"]
    ref = (
        train.groupby(group_cols, observed=True)
        .agg(
            parent_rank_q90=("policy_parent_rank", lambda x: x.quantile(0.90)),
            hit_probability_q90=("hit_probability", lambda x: x.quantile(0.90)),
            local_support=("policy_parent_rank", "count"),
        )
        .reset_index()
    )
    global_rank_q90 = float(pd.to_numeric(train["policy_parent_rank"], errors="coerce").quantile(0.90))
    global_hit_q90 = float(pd.to_numeric(train["hit_probability"], errors="coerce").quantile(0.90))
    merged = out.merge(ref, on=group_cols, how="left", validate="many_to_one", copy=False)
    rank_q90 = pd.to_numeric(merged["parent_rank_q90"], errors="coerce").fillna(global_rank_q90)
    hit_q90 = pd.to_numeric(merged["hit_probability_q90"], errors="coerce").fillna(global_hit_q90)
    support = pd.to_numeric(merged["local_support"], errors="coerce").fillna(0.0)
    merged["meta_parent_rank_local_top10_margin"] = (
        pd.to_numeric(merged["policy_parent_rank"], errors="coerce") - rank_q90
    ).astype(np.float32)
    merged["meta_hit_probability_local_top10_margin"] = (
        pd.to_numeric(merged["hit_probability"], errors="coerce") - hit_q90
    ).astype(np.float32)
    merged["meta_parent_reliability_local_support_log1p"] = np.log1p(
        support
    ).astype(np.float32)
    return merged.drop(columns=["parent_rank_q90", "hit_probability_q90", "local_support"])


def _ev_residual(group: pd.DataFrame) -> np.ndarray:
    ev = _num(group, "ev_after_1pct")
    rank = _num(group, "policy_parent_rank", 0.5)
    order = np.argsort(rank, kind="stable")
    expected = np.zeros(len(group), dtype=np.float32)
    for idx in np.array_split(order, min(30, max(6, len(group) // 600))):
        if len(idx):
            expected[idx] = np.float32(np.mean(ev[idx]))
    return ev - expected


def _causal_ev_residual(group: pd.DataFrame) -> np.ndarray:
    """EV residual whose expectation for each block is fitted on prior rows only."""
    ev = _num(group, "ev_after_1pct")
    rank = _num(group, "policy_parent_rank", 0.5)
    hit = _num(group, "hit_probability", 0.5)
    blocks = np.array_split(np.arange(len(group), dtype=np.int32), 4)
    residual = np.full(len(group), np.nan, dtype=np.float32)
    for fold in range(1, len(blocks)):
        train_idx = np.concatenate(blocks[:fold])
        valid_idx = blocks[fold]
        if len(train_idx) < 200 or not len(valid_idx):
            continue
        rank_edges = np.unique(np.quantile(rank[train_idx], np.linspace(0, 1, 9)))
        hit_edges = np.unique(np.quantile(hit[train_idx], np.linspace(0, 1, 5)))
        if len(rank_edges) < 3 or len(hit_edges) < 3:
            residual[valid_idx] = ev[valid_idx] - np.float32(np.mean(ev[train_idx]))
            continue
        nr, nh = len(rank_edges) - 1, len(hit_edges) - 1
        rb = np.clip(np.searchsorted(rank_edges, rank[train_idx], side="right") - 1, 0, nr - 1)
        hb = np.clip(np.searchsorted(hit_edges, hit[train_idx], side="right") - 1, 0, nh - 1)
        cell = rb * nh + hb
        sums = np.bincount(cell, weights=ev[train_idx], minlength=nr * nh)
        counts = np.bincount(cell, minlength=nr * nh)
        global_mean = float(np.mean(ev[train_idx]))
        means = np.divide(
            sums + 40.0 * global_mean,
            counts + 40.0,
            out=np.full(nr * nh, global_mean),
            where=(counts + 40.0) > 0,
        )
        rv = np.clip(np.searchsorted(rank_edges, rank[valid_idx], side="right") - 1, 0, nr - 1)
        hv = np.clip(np.searchsorted(hit_edges, hit[valid_idx], side="right") - 1, 0, nh - 1)
        residual[valid_idx] = ev[valid_idx] - means[rv * nh + hv].astype(np.float32)
    return residual


def _weighted_binned_mi(values: np.ndarray, target: np.ndarray, weight: np.ndarray) -> float:
    mask = np.isfinite(values) & np.isfinite(target) & np.isfinite(weight) & (weight > 0)
    if int(mask.sum()) < 100:
        return 0.0
    x, y, w = values[mask], target[mask], weight[mask]
    xe = np.unique(np.quantile(x, np.linspace(0, 1, 9)))
    ye = np.unique(np.quantile(y, np.linspace(0, 1, 9)))
    if len(xe) < 4 or len(ye) < 4:
        return 0.0
    xb = np.clip(np.searchsorted(xe, x, side="right") - 1, 0, len(xe) - 2)
    yb = np.clip(np.searchsorted(ye, y, side="right") - 1, 0, len(ye) - 2)
    joint = np.zeros((len(xe) - 1, len(ye) - 1), dtype=np.float64)
    np.add.at(joint, (xb, yb), w)
    joint /= max(float(joint.sum()), 1e-12)
    px, py = joint.sum(axis=1), joint.sum(axis=0)
    expected = px[:, None] * py[None, :]
    nz = (joint > 0) & (expected > 0)
    return float(np.sum(joint[nz] * np.log(joint[nz] / expected[nz])))


def _binned_oof_gain(
    values: np.ndarray,
    target: np.ndarray,
    weight: np.ndarray,
    temporal: list[np.ndarray],
) -> tuple[float, float]:
    gains: list[float] = []
    for fold in range(1, len(temporal)):
        tr = np.concatenate(temporal[:fold])
        va = temporal[fold]
        tr = tr[np.isfinite(target[tr]) & np.isfinite(values[tr])]
        va = va[np.isfinite(target[va]) & np.isfinite(values[va])]
        if len(tr) < 250 or len(va) < 100:
            continue
        edges = np.unique(np.quantile(values[tr], np.linspace(0, 1, 9)))
        if len(edges) < 4:
            continue
        tb = np.clip(np.searchsorted(edges, values[tr], side="right") - 1, 0, len(edges) - 2)
        vb = np.clip(np.searchsorted(edges, values[va], side="right") - 1, 0, len(edges) - 2)
        sums = np.bincount(tb, weights=target[tr] * weight[tr], minlength=len(edges) - 1)
        counts = np.bincount(tb, weights=weight[tr], minlength=len(edges) - 1)
        means = np.divide(sums, counts + 30.0, out=np.zeros_like(sums), where=(counts + 30.0) > 0)
        pred = means[vb]
        base_loss = np.average(target[va] ** 2, weights=weight[va])
        model_loss = np.average((target[va] - pred) ** 2, weights=weight[va])
        gains.append(float((base_loss - model_loss) / max(base_loss, 1e-8)))
    if not gains:
        return 0.0, 0.0
    return float(np.mean(gains)), float(min(gains))


def _select_ev_features(
    group: pd.DataFrame,
    candidates: list[str],
    *,
    max_features: int | None,
    seed: int,
    auto_feature_ceiling: int = 48,
) -> tuple[list[str], pd.DataFrame]:
    """Top-tail EV screening with forward-stable nonlinear bin effects."""
    usable = [
        c for c in candidates if c in group and group[c].notna().mean() >= 0.40
        and group[c].nunique(dropna=True) >= 4
    ]
    if not usable:
        return [], pd.DataFrame()
    rank_all = _num(group, "policy_parent_rank", 0.5)
    blocks = np.array_split(np.arange(len(group)), 3)
    sampled: list[np.ndarray] = []
    rng = np.random.default_rng(seed)
    for block in blocks:
        if not len(block):
            continue
        # Preserve chronology while over-sampling the region that can compete
        # for the global top-10 book.
        high = block[rank_all[block] >= 0.80]
        rest = block[rank_all[block] < 0.80]
        take_high = min(len(high), 2_800)
        take_rest = min(len(rest), 1_200)
        h = np.sort(rng.choice(high, take_high, replace=False)) if take_high else high
        r = np.sort(rng.choice(rest, take_rest, replace=False)) if take_rest else rest
        sampled.append(np.sort(np.concatenate([h, r])))
    idx = np.sort(np.concatenate(sampled)) if sampled else np.arange(len(group))
    sample = group.iloc[idx]
    x = sample[usable].apply(pd.to_numeric, errors="coerce")
    x = x.fillna(x.median(axis=0)).fillna(0.0)
    arr = x.to_numpy(dtype=np.float32, copy=False)
    target = _causal_ev_residual(sample)
    rank = _num(sample, "policy_parent_rank", 0.5)
    weights = np.where(rank >= 0.90, 4.0, np.where(rank >= 0.80, 2.0, 0.25)).astype(np.float32)
    temporal = np.array_split(np.arange(len(sample)), 3)
    nonlinear = np.zeros(len(usable), dtype=np.float32)
    stability = np.zeros(len(usable), dtype=np.float32)
    linear = np.zeros(len(usable), dtype=np.float32)
    weighted_mi = np.zeros(len(usable), dtype=np.float32)
    for j in range(len(usable)):
        values = arr[:, j]
        valid = np.isfinite(target) & np.isfinite(values)
        weighted_y = target[valid] * weights[valid]
        if int(valid.sum()) > 100 and np.std(values[valid]) > 1e-7:
            corr = np.corrcoef(values[valid], weighted_y)[0, 1]
            linear[j] = np.float32(abs(corr)) if np.isfinite(corr) else 0.0
        gain, worst_gain = _binned_oof_gain(values, target, weights, temporal)
        nonlinear[j] = np.float32(max(0.0, gain))
        stability[j] = np.float32(max(0.0, worst_gain))
        weighted_mi[j] = np.float32(_weighted_binned_mi(values, target, weights))
    mi_scale = max(float(np.nanmax(weighted_mi)), 1e-8)
    weighted_mi /= np.float32(mi_scale)
    score = nonlinear + 0.60 * stability + 0.15 * linear + 0.20 * weighted_mi

    # Cheap nonlinear interaction screen: reward constituents whose joint bins
    # add causal OOF residual information beyond either feature alone.
    top_single = np.argsort(-score, kind="stable")[: min(16, len(usable))]
    interaction_gain = np.zeros(len(usable), dtype=np.float32)
    pair_records: list[dict[str, Any]] = []
    for left_pos, left in enumerate(top_single):
        for right in top_single[left_pos + 1 :]:
            product = np.sign(arr[:, left] * arr[:, right]) * np.sqrt(
                np.abs(arr[:, left] * arr[:, right])
            )
            gain, worst_gain = _binned_oof_gain(product, target, weights, temporal)
            incremental = max(0.0, gain - max(float(nonlinear[left]), float(nonlinear[right])))
            if incremental <= 0.0 or worst_gain <= 0.0:
                continue
            boost = np.float32(0.25 * incremental)
            interaction_gain[left] += boost
            interaction_gain[right] += boost
            pair_records.append(
                {
                    "interaction_left": usable[int(left)],
                    "interaction_right": usable[int(right)],
                    "interaction_oof_gain": float(gain),
                    "interaction_incremental_gain": float(incremental),
                }
            )
    score += interaction_gain
    order = np.argsort(-score, kind="stable")
    positive = score[np.isfinite(score) & (score > 0)]
    if max_features is None:
        threshold = (
            float(np.median(positive) + 0.25 * np.std(positive))
            if len(positive)
            else np.inf
        )
        # The old local-MLP path used a fixed 48-feature ceiling.  Keep that
        # compatibility default, but let new pooled experiments use the same
        # evidence-based stopping rule with a larger explicit ceiling.
        auto_cap = int(np.clip(np.sum(score >= threshold), 8, auto_feature_ceiling))
    else:
        auto_cap = max(4, int(max_features))
    selected: list[str] = []
    for pos in order:
        feature = usable[int(pos)]
        if selected and x[selected].corrwith(x[feature]).abs().gt(0.95).any():
            continue
        selected.append(feature)
        if len(selected) >= auto_cap:
            break
    selected_set = set(selected)
    rank_position = np.empty(len(usable), dtype=np.int32)
    rank_position[order] = np.arange(1, len(order) + 1, dtype=np.int32)
    best_partner: dict[str, tuple[str, float]] = {}
    for rec in pair_records:
        left = str(rec["interaction_left"])
        right = str(rec["interaction_right"])
        gain = float(rec["interaction_incremental_gain"])
        if gain > best_partner.get(left, ("", -np.inf))[1]:
            best_partner[left] = (right, gain)
        if gain > best_partner.get(right, ("", -np.inf))[1]:
            best_partner[right] = (left, gain)
    report = pd.DataFrame(
        {
            "feature": usable,
            "feature_family": [_feature_family_name(c) for c in usable],
            "coverage": [float(group[c].notna().mean()) for c in usable],
            "linear_score": linear,
            "conditional_oof_gain": nonlinear,
            "worst_fold_oof_gain": stability,
            "weighted_binned_mi": weighted_mi,
            "interaction_constituent_gain": interaction_gain,
            "final_score": score,
            "score_rank": rank_position,
            "selected": [feature in selected_set for feature in usable],
            "selection_reason": [
                "automatic_oof_score_with_redundancy_pass"
                if feature in selected_set
                else ""
                for feature in usable
            ],
            "rejection_reason": [
                ""
                if feature in selected_set
                else (
                    "below_automatic_feature_count"
                    if int(rank_position[i]) > auto_cap
                    else "correlation_redundant"
                )
                for i, feature in enumerate(usable)
            ],
            "best_interaction_partner": [
                best_partner.get(feature, ("", 0.0))[0] for feature in usable
            ],
            "best_interaction_incremental_gain": [
                best_partner.get(feature, ("", 0.0))[1] for feature in usable
            ],
            "automatic_feature_cap": auto_cap,
            "automatic_feature_ceiling": int(auto_feature_ceiling),
        }
    )
    if pair_records:
        report.attrs["interaction_screen"] = pair_records
    return selected, report


def _fit_models(
    train: pd.DataFrame,
    arm: EncoderArm,
    candidates: list[str],
    ae_cols: list[str],
    min_rows: int,
    seed: int,
    frozen_features: dict[tuple[str, str], list[str]] | None = None,
    mlp_params: dict[str, Any] | None = None,
    max_local_fit_rows: int | None = None,
) -> list[LocalStateEncoder]:
    models: list[LocalStateEncoder] = []
    for (side, arch), group in train.groupby(
        ["side_name", "archetype_policy_key"], observed=True, sort=True
    ):
        if len(group) < min_rows:
            continue
        key = (str(side), str(arch))
        features = (frozen_features or {}).get(key)
        if not features:
            features, _selection_report = _select_ev_features(
                group,
                [*candidates, "policy_parent_rank", "hit_probability"],
                max_features=None,
                seed=seed + len(models) * 41,
            )
        features = [c for c in features if c in group]
        if len(features) < 4:
            continue
        fit_group = group
        if max_local_fit_rows and len(group) > max_local_fit_rows:
            blocks = np.array_split(np.arange(len(group), dtype=np.int64), 3)
            per_block = max(1, int(max_local_fit_rows) // 3)
            fit_idx = np.concatenate(
                [
                    np.linspace(block[0], block[-1], min(per_block, len(block)), dtype=np.int64)
                    for block in blocks if len(block)
                ]
            )
            fit_group = group.iloc[np.unique(fit_idx)]
        ev_residual = _ev_residual(fit_group)
        hit_residual = _num(fit_group, "clean_exec") - _num(fit_group, "hit_probability", 0.5)
        rank = _num(fit_group, "policy_parent_rank", 0.5)
        weight = (0.5 + rank + 1.5 * (rank >= 0.90)).astype(np.float32)
        models.append(
            fit_local_state_encoder(
                fit_group,
                side=key[0], archetype=key[1], arm=arm,
                features=features, ae_features=ae_cols,
                ev_residual=ev_residual, hit_residual=hit_residual,
                sample_weight=weight, n_components=(3, 4, 5, 6), shrink_rows=500.0,
                seed=seed + len(models) * 101,
                mlp_params=mlp_params,
            )
        )
    return models


def _build_feature_map(
    train: pd.DataFrame,
    candidates: list[str],
    min_rows: int,
    seed: int,
) -> tuple[dict[tuple[str, str], list[str]], pd.DataFrame]:
    result: dict[tuple[str, str], list[str]] = {}
    reports: list[pd.DataFrame] = []
    for index, ((side, arch), group) in enumerate(
        train.groupby(["side_name", "archetype_policy_key"], observed=True, sort=True)
    ):
        if len(group) < min_rows:
            continue
        selected, report = _select_ev_features(
            group,
            [*candidates, "policy_parent_rank", "hit_probability"],
            max_features=None,
            seed=seed + index * 41,
        )
        if not report.empty:
            report = report.copy()
            report.insert(0, "archetype_policy_key", str(arch))
            report.insert(0, "side_name", str(side))
            reports.append(report)
        if len(selected) >= 4:
            result[(str(side), str(arch))] = selected
    return result, (pd.concat(reports, ignore_index=True) if reports else pd.DataFrame())


def _predict_models(
    frame: pd.DataFrame, models: list[LocalStateEncoder], ae_cols: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    correction = np.zeros(len(frame), dtype=np.float32)
    confidence = np.zeros(len(frame), dtype=np.float32)
    entropy = np.ones(len(frame), dtype=np.float32)
    sides = frame["side_name"].astype(str).to_numpy()
    arches = frame["archetype_policy_key"].astype(str).to_numpy()
    for model in models:
        pos = np.flatnonzero((sides == model.side) & (arches == model.archetype))
        if not len(pos):
            continue
        pred = predict_local_state_encoder(model, frame.iloc[pos], ae_features=ae_cols)
        correction[pos] = pred["ev_correction"] / max(model.ev_scale, 1e-4)
        confidence[pos] = pred["posterior_confidence"]
        entropy[pos] = pred["posterior_entropy"]
    return np.clip(correction, -4, 4), confidence, entropy


def _overlay_score(
    frame: pd.DataFrame,
    correction: np.ndarray,
    confidence: np.ndarray,
    params: dict[str, Any],
) -> np.ndarray:
    base = _num(frame, "policy_parent_rank", 0.5)
    local_alpha: dict[str, float] = params.get("local_alpha", {})
    default_alpha = float(params["alpha"])
    if local_alpha:
        side = frame["side_name"].astype(str).to_numpy()
        arch = frame["archetype_policy_key"].astype(str).to_numpy()
        alpha: float | np.ndarray = np.full(
            len(frame), default_alpha, dtype=np.float32
        )
        for key, value in local_alpha.items():
            local_side, separator, local_arch = str(key).partition("||")
            if not separator:
                continue
            alpha[(side == local_side) & (arch == local_arch)] = float(value)
    else:
        alpha = default_alpha
    posterior = np.power(np.clip(confidence, 0.0, 1.0), float(params["posterior_power"]))
    delta = np.clip(alpha * posterior * correction, -float(params["cap"]), float(params["cap"]))
    return np.clip(base + delta, 0.0, 1.0).astype(np.float32)


def _tune_overlay(
    folds: pd.DataFrame,
    correction: np.ndarray,
    confidence: np.ndarray,
) -> tuple[dict[str, Any], pd.DataFrame]:
    # This function evaluates hundreds of score candidates. Deriving calendar
    # groups inside every metric call dominated runtime on the full OOF stream.
    # Cache them once while retaining the original frame ordering.
    if "__week_code__" not in folds or "__month_code__" not in folds:
        folds = folds.copy(deep=False)
        timestamps = pd.to_datetime(folds["__ts__"], utc=True, errors="coerce")
        days = timestamps.dt.floor("D")
        week_start = days - pd.to_timedelta(
            days.dt.weekday.to_numpy(), unit="D"
        )
        folds["__week_code__"] = pd.factorize(
            week_start, sort=True
        )[0].astype(np.int32)
        folds["__month_code__"] = pd.factorize(
            timestamps.dt.strftime("%Y-%m"), sort=True
        )[0].astype(np.int16)
    base = _num(folds, "policy_parent_rank", 0.5)
    budget = int(np.sum(base >= 0.90))
    baseline = _economic_metrics(folds, base, budget, target_activity=budget)

    def rolling_objective(score: np.ndarray) -> float:
        aggregate = _economic_metrics(folds, score, budget, target_activity=budget)
        aggregate_obj = _objective(aggregate, baseline)
        if aggregate_obj <= -1e8:
            return aggregate_obj
        fold_objectives: list[float] = []
        for fold_id in sorted(folds["__fold__"].unique()):
            pos = np.flatnonzero(folds["__fold__"].to_numpy() == fold_id)
            local = folds.iloc[pos]
            local_base_score = base[pos]
            local_budget = int(np.sum(local_base_score >= 0.90))
            local_base = _economic_metrics(
                local, local_base_score, local_budget, target_activity=local_budget
            )
            local_metric = _economic_metrics(
                local, score[pos], local_budget, target_activity=local_budget
            )
            value = _objective(local_metric, local_base)
            if value <= -1e8:
                return value
            fold_objectives.append(value)
        fold_arr = np.asarray(fold_objectives, dtype=np.float64)
        return float(
            0.50 * aggregate_obj
            + 0.50 * (fold_arr.mean() - 0.25 * fold_arr.std() + 0.10 * fold_arr.min())
        )
    rows: list[dict[str, Any]] = []
    for alpha in (0.0, 0.001, 0.0025, 0.005, 0.01, 0.02):
        for cap in (0.005, 0.01, 0.025, 0.05):
            for power in (0.5, 1.0, 2.0):
                params = {"alpha": alpha, "cap": cap, "posterior_power": power, "local_alpha": {}}
                score = _overlay_score(
                    folds, correction, confidence, params
                )
                metric = _economic_metrics(
                    folds, score,
                    budget, target_activity=budget,
                )
                rows.append(
                    {
                        **params,
                        **metric,
                        "objective": rolling_objective(score),
                    }
                )
    search = pd.DataFrame(rows).sort_values("objective", ascending=False, kind="stable")
    baseline_obj = rolling_objective(base)
    max_obj = float(search.iloc[0]["objective"])
    conservative_floor = baseline_obj + 0.25 * max(0.0, max_obj - baseline_obj)
    eligible = search.loc[search["objective"].ge(conservative_floor)].copy()
    best = eligible.sort_values(
        ["alpha", "cap", "posterior_power"], ascending=[True, True, False], kind="stable"
    ).iloc[0].to_dict()
    best["local_alpha"] = {}
    # Coordinate refinement gives each supported side x archetype its own overlay
    # strength while scoring the globally competing top-10 book after each change.
    keys = list(
        folds.groupby(["side_name", "archetype_policy_key"], observed=True).size()
        .sort_values(ascending=False).index
    )
    current_score = _overlay_score(folds, correction, confidence, best)
    current_obj = rolling_objective(current_score)
    for side, arch in keys:
        key = f"{side}||{arch}"
        chosen = float(best["alpha"])
        base_strength = max(float(best["alpha"]), 0.001)
        local_candidates = tuple(
            sorted(set((0.0, 0.5 * base_strength, base_strength, 1.5 * base_strength, 2.0 * base_strength)))
        )
        for local_strength in local_candidates:
            trial = {**best, "local_alpha": {**best["local_alpha"], key: local_strength}}
            score = _overlay_score(folds, correction, confidence, trial)
            obj = rolling_objective(score)
            if obj > current_obj + 1e-12:
                current_obj, chosen = obj, local_strength
        best["local_alpha"][key] = chosen
        current_score = _overlay_score(folds, correction, confidence, best)
        current_obj = rolling_objective(current_score)
    # Prefer a sparse, stable set of local corrections. Every retained effect
    # must improve the globally competing book in each available prior OOF
    # fold. Among subsets within 25% of the best prior-fold uplift, select the
    # smallest one. This deliberately underfits rather than carrying weak local
    # effects into later regimes.
    stable_locals: dict[str, float] = {}
    fold_ids = sorted(int(value) for value in folds["__fold__"].unique())
    for key, strength in best["local_alpha"].items():
        if abs(float(strength)) < 1e-12 or key in BLACKLISTED_SIDE_ARCHETYPES:
            continue
        single = {
            **best,
            "alpha": 0.0,
            "local_alpha": {key: float(strength)},
        }
        single_score = _overlay_score(folds, correction, confidence, single)
        stable = True
        for fold_id in fold_ids:
            pos = np.flatnonzero(folds["__fold__"].to_numpy() == fold_id)
            local = folds.iloc[pos]
            local_budget = int(
                np.sum(_num(local, "policy_parent_rank") >= 0.90)
            )
            if local_budget <= 0:
                continue
            base_metric = _economic_metrics(
                local,
                base[pos],
                local_budget,
                target_activity=local_budget,
            )
            single_metric = _economic_metrics(
                local,
                single_score[pos],
                local_budget,
                target_activity=local_budget,
            )
            if (
                single_metric["mean_net_ev_top10"]
                < base_metric["mean_net_ev_top10"] - 1e-12
            ):
                stable = False
                break
        if stable:
            stable_locals[key] = float(strength)
    subset_rows: list[tuple[float, int, float, dict[str, float]]] = []
    stable_items = sorted(stable_locals.items())
    if len(stable_items) <= 12:
        for bits in itertools.product((False, True), repeat=len(stable_items)):
            local = {
                key: strength
                for (key, strength), enabled in zip(stable_items, bits)
                if enabled
            }
            params = {**best, "alpha": 0.0, "local_alpha": local}
            objective = rolling_objective(
                _overlay_score(folds, correction, confidence, params)
            )
            subset_rows.append(
                (
                    float(objective),
                    len(local),
                    float(sum(abs(value) for value in local.values())),
                    local,
                )
            )
    if subset_rows:
        baseline_objective = rolling_objective(base)
        max_subset_objective = max(row[0] for row in subset_rows)
        sparse_floor = baseline_objective + 0.25 * max(
            0.0, max_subset_objective - baseline_objective
        )
        eligible_subsets = [
            row for row in subset_rows if row[0] >= sparse_floor - 1e-12
        ]
        chosen_subset = sorted(
            eligible_subsets,
            key=lambda row: (row[1], row[2], -row[0]),
        )[0]
        best["alpha"] = 0.0
        best["local_alpha"] = chosen_subset[3]
        best["sparse_local_objective"] = chosen_subset[0]
        best["sparse_local_count"] = chosen_subset[1]
        best["sparse_local_candidates"] = len(stable_items)
        best["sparse_local_floor"] = sparse_floor
    return best, search


def main() -> None:
    global PREDECESSOR_ARTIFACT
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, default=Path("data_perp/reports/meta_market_state_encoder_ablation_20260712_v1"))
    p.add_argument("--train-start", default="2025-01-01")
    p.add_argument("--eval-end", default="2026-07-01")
    p.add_argument("--min-local-rows", type=int, default=1200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--run-mlp-hpo",
        action="store_true",
        help="Run chronological MLP-model HPO. Disabled by default.",
    )
    p.add_argument("--mlp-hpo-trials", type=int, default=20)
    p.add_argument(
        "--mlp-params-json",
        type=Path,
        default=None,
        help="Promoted best_mlp_params.json to use when HPO is disabled.",
    )
    p.add_argument("--expanding-walkforward", action="store_true")
    p.add_argument(
        "--parent-meta-oos-start",
        default="2026-01-01",
        help="Earliest parent-meta OOS row eligible for MLP fitting or scoring.",
    )
    p.add_argument(
        "--walkforward-tuning-start",
        default="2026-02-01",
        help="First month with eligible parent-meta OOS rows for MLP tuning/refits.",
    )
    p.add_argument(
        "--walkforward-policy-start",
        default="2026-04-01",
        help="First untouched monthly OOS month emitted for policy optimisation.",
    )
    p.add_argument(
        "--history-train-end",
        default="2026-04-01",
        help=(
            "Boundary used to assemble chronological OOF history versus the "
            "parent evaluation stream. This controls data loading only; every "
            "expanding fold still fits strictly before its own prediction time."
        ),
    )
    p.add_argument("--arms", default=",".join(ARMS), help="Comma-separated encoder arms")
    p.add_argument(
        "--frozen-encoder-artifact",
        type=Path,
        default=None,
        help=(
            "Optional prior encoder_calibrators.joblib whose side x archetype "
            "feature sets are reused; models and calibration parameters are refit."
        ),
    )
    p.add_argument("--expanded-source", type=Path, default=Path("data_perp/reports/s59_h5_2025start_monthly_v4_base_configfull_mdafs120_hpo150_largestfold_oos15_ae3000_nocrossfit_k34567_payload300k_20260706/s52_trailing_regime_meta_handoff_top30_allsafe_aegmm_fixedtargets_oos15_20260706/s52_trailing_regime_scored_ledger.parquet"))
    p.add_argument("--champion-ledger", type=Path, default=Path("data_perp/reports/train_meta_residual_archetype_enhancement_20260711_v1/champion_frozen_single_source_202501_20260710/frozen_champion_single_source_ledger.parquet"))
    p.add_argument("--train-oof-predictions-dir", type=Path, default=Path("data_perp/reports/s59_h5_2025start_monthly_v4_base_configfull_mdafs120_hpo150_largestfold_oos15_ae3000_nocrossfit_k34567_payload300k_20260706/train_meta_regime_handoff_singlehead_base_soft_lgbmpipeline_auto_hpo150_oos15_top30_hpo45k_20260706_v5/best_full_oos_fixedfs_streamed_v1/prediction_shards"))
    p.add_argument(
        "--base-uncertainty-source",
        type=Path,
        default=None,
        help=(
            "Base OOF/prediction ledger with the base_lgbm_* predictive-"
            "uncertainty contract. Joined on timestamp/symbol/side."
        ),
    )
    p.add_argument(
        "--require-base-uncertainty",
        action="store_true",
        help=(
            "Fail before HPO unless at least four base predictive-uncertainty "
            "features have >=95% coverage in both train and OOS frames."
        ),
    )
    p.add_argument("--train-oof-rank-cache", type=Path, default=Path("data_perp/reports/residual_event_archetype_true_base_oof_compactlocal_market_20260712_v3/meta_oof_global_rank_202504_202603.parquet"))
    p.add_argument("--state-artifact", type=Path, default=Path("data_perp/reports/residual_event_archetype_true_base_oof_compactlocal_market_20260712_v3/oos_residual_event_states.parquet"))
    p.add_argument("--context-state-artifact", type=Path, default=Path("data_perp/reports/residual_event_archetype_enhanced_gmm_oos_history_20260712_v2/oos_residual_event_states.parquet"))
    p.add_argument("--parent-eval-predictions", type=Path, default=Path("data_perp/reports/train_meta_residual_archetype_enhancement_20260711_v1/lifecycle_residual_aware_ae_gmm_overlay_pca8_clip8_baseline_globaloverlay_sparse_shock_composite/oos_predictions_historical_rank.parquet"))
    p.add_argument(
        "--predecessor-artifact",
        type=Path,
        default=PREDECESSOR_ARTIFACT,
        help=(
            "Run-scoped v9 tail95 predecessor directory. The directory must "
            "contain forced_nonzero_tail_summary.csv."
        ),
    )
    args = p.parse_args()
    PREDECESSOR_ARTIFACT = args.predecessor_artifact
    predecessor_summary = PREDECESSOR_ARTIFACT / "forced_nonzero_tail_summary.csv"
    if not predecessor_summary.exists():
        raise FileNotFoundError(
            f"required direct predecessor artifact is missing: {predecessor_summary}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    start = pd.Timestamp(args.train_start, tz="UTC")
    tune_end = pd.Timestamp(args.history_train_end, tz="UTC")
    eval_end = pd.Timestamp(args.eval_end, tz="UTC")
    history, test, coverage = _load_joined(
        champion_path=args.champion_ledger, parent_eval_path=args.parent_eval_predictions,
        state_path=args.state_artifact, train_oof_predictions_dir=args.train_oof_predictions_dir,
        train_oof_rank_cache=args.train_oof_rank_cache, train_start=start,
        train_end=tune_end, eval_end=eval_end,
        strict_eval_state_coverage=True,
    )
    history = _merge_meta_oof_observable_features(
        history, args.train_oof_predictions_dir
    )
    test = _merge_meta_oof_observable_features(test, args.train_oof_predictions_dir)
    if "score_meta_base_soft_label" in history:
        history["hit_probability"] = pd.to_numeric(
            history["hit_probability"], errors="coerce"
        ).fillna(
            pd.to_numeric(
                history["score_meta_base_soft_label"], errors="coerce"
            )
        )
    history, base_uncertainty_train_audit = _merge_base_predictive_uncertainty(
        history, args.base_uncertainty_source
    )
    test, base_uncertainty_test_audit = _merge_base_predictive_uncertainty(
        test, args.base_uncertainty_source
    )
    if args.require_base_uncertainty:
        covered_train = {
            key
            for key, value in base_uncertainty_train_audit["coverage"].items()
            if value >= 0.95
        }
        covered_test = {
            key
            for key, value in base_uncertainty_test_audit["coverage"].items()
            if value >= 0.95
        }
        supported = covered_train & covered_test
        if len(supported) < 4:
            raise RuntimeError(
                "base uncertainty contract is incomplete; need >=4 fields with "
                f"95% train/OOS coverage, got {sorted(supported)}"
            )
    parent_params = {"top_feature_count": 1, "threshold": 0.95, "alpha_down": 0.02, "alpha_up": 0.0}
    catalog = _feature_catalog(history)
    # The predecessor catalog and empirical references are one frozen
    # transform contract. Later context projection is allowed to replace
    # columns for MLP screening, so capture V9 references on the exact frame
    # used to construct its catalog.
    predecessor_references = _fit_references(history, catalog, 1)
    history["policy_parent_rank"], _, _ = _rank_for_params(history, history, catalog, parent_params)
    test["policy_parent_rank"], _, _ = _rank_for_params(history, test, catalog, parent_params)
    # Freeze local score references on the fitting history.  The OOS frame is
    # transformed only by those historical side x archetype quantiles/supports.
    history = _add_side_archetype_parent_reliability(history, history)
    test = _add_side_archetype_parent_reliability(history, test)
    history_keys, test_keys = history[KEYS].copy(), test[KEYS].copy()
    frozen_features: dict[tuple[str, str], list[str]] = {}
    frozen_overlay_params: dict[str, Any] | None = None
    frozen_feature_source = "fresh_ev_screen"
    if args.frozen_encoder_artifact is not None:
        prior = joblib.load(args.frozen_encoder_artifact)
        prior_models = (prior.get("mlp_direct") or {}).get("models") or []
        frozen_overlay_params = dict(
            (prior.get("mlp_direct") or {}).get("params") or {}
        ) or None
        frozen_features = {
            (str(model.side), str(model.archetype)): list(model.features)
            for model in prior_models
            if getattr(model, "features", None)
        }
        frozen_feature_source = str(args.frozen_encoder_artifact)
    if args.run_mlp_hpo and frozen_features:
        print(
            "MLP HPO requested: prior feature maps are diagnostic only; "
            "re-screening the complete observable basket.",
            flush=True,
        )
        frozen_features = {}
        frozen_feature_source = "fresh_complete_basket_for_mlp_hpo"
    if frozen_features:
        selected_union = {
            feature for features in frozen_features.values() for feature in features
        }
        selected_union.update(("hit_probability", "policy_parent_rank"))
        sources = [
            (args.context_state_artifact, True),
            (args.expanded_source, False),
        ]
        history = _merge_projected_context(history, sources, selected_union)
        test = _merge_projected_context(test, sources, selected_union)
    else:
        history = _merge_observable_context(history, state_artifact=args.context_state_artifact, expanded_source=args.expanded_source, override_existing=True)
        test = _merge_observable_context(test, state_artifact=args.context_state_artifact, expanded_source=args.expanded_source, override_existing=True)
        history = _merge_market_state_features(history, args.expanded_source)
        test = _merge_market_state_features(test, args.expanded_source)
    history = _add_observable_reliability_features(history)
    test = _add_observable_reliability_features(test)
    if not _same_identity_rows(history[KEYS], history_keys) or not _same_identity_rows(
        test[KEYS], test_keys
    ):
        raise AssertionError("context merge changed fixed parent rows")
    candidates = list(dict.fromkeys([
        *_feature_block(history, "joint_expanded_context"),
        *[key for key in BASE_PREDICTIVE_UNCERTAINTY_FEATURES if key in history],
        *[key for key in META_PARENT_RELIABILITY_FEATURES if key in history],
        *[
            c for c in history
            if any(token in c.lower() for token in (*MARKET_STATE_TOKENS, *RELIABILITY_CONTEXT_TOKENS))
            and _is_allowed_observable_feature(c)
        ],
    ]))
    reliability_audit = {
        family: [c for c in candidates if token in c.lower()]
        for family, token in {
            "uncertainty": "uncertainty",
            "leaf": "leaf",
            "support": "support",
            "drift": "drift",
            "ood": "ood",
        }.items()
    }
    reliability_audit["base_predictive_uncertainty"] = {
        "train": base_uncertainty_train_audit,
        "oos": base_uncertainty_test_audit,
        "candidate_features": [
            key for key in BASE_PREDICTIVE_UNCERTAINTY_FEATURES if key in candidates
        ],
    }
    reliability_audit["meta_parent_reliability"] = [
        key for key in META_PARENT_RELIABILITY_FEATURES if key in candidates
    ]
    (args.output_dir / "initial_feature_basket_audit.json").write_text(
        json.dumps(reliability_audit, indent=2) + "\n"
    )
    print(
        "Reliability basket: "
        + ", ".join(f"{key}={len(value)}" for key, value in reliability_audit.items()),
        flush=True,
    )
    if args.run_mlp_hpo:
        missing_families = [
            family
            for family in ("uncertainty", "leaf", "support", "drift", "ood")
            if not reliability_audit.get(family)
        ]
        if missing_families:
            raise RuntimeError(
                "MLP HPO requires a complete observable reliability basket; "
                f"missing families={missing_families}"
            )
    ae_cols = _ae_features(history)
    fold_specs = [
        (pd.Timestamp("2026-02-01", tz="UTC"), pd.Timestamp("2026-03-01", tz="UTC")),
        (pd.Timestamp("2026-03-01", tz="UTC"), pd.Timestamp("2026-04-01", tz="UTC")),
    ]
    feature_selection_train = history.loc[
        history["__ts__"].lt(pd.Timestamp("2026-02-01", tz="UTC"))
    ]
    if args.expanding_walkforward:
        feature_selection_train = feature_selection_train.loc[
            feature_selection_train["__ts__"].ge(
                pd.Timestamp(args.parent_meta_oos_start, tz="UTC")
            )
        ]
    if not frozen_features:
        coverage_features = list(
            dict.fromkeys([*candidates, "policy_parent_rank", "hit_probability"])
        )
        coverage_features = [
            feature
            for feature in coverage_features
            if feature in feature_selection_train.columns
        ]
        coverage_survivors, coverage_report = _recent_feature_coverage_survivors(
            feature_selection_train.loc[:, coverage_features],
            feature_selection_train["__ts__"].to_numpy(),
            require_joint_complete_case=True,
            min_feature_coverage=0.90,
            coverage_scope="all_post_warmup",
            warmup_days=30,
            warmup_reference_start=history["__ts__"].min(),
        )
        survivor_set = set(coverage_survivors)
        missing_required = {
            "policy_parent_rank",
            "hit_probability",
        } - survivor_set
        if missing_required:
            raise RuntimeError(
                "Required MLP anchors fail the 90% joint feature-availability "
                f"contract: {sorted(missing_required)}"
            )
        candidates = [feature for feature in candidates if feature in survivor_set]
        (args.output_dir / "pre_mlp_joint_feature_coverage.json").write_text(
            json.dumps(coverage_report, indent=2) + "\n"
        )
        print(
            f"Selecting EV features once: rows={len(feature_selection_train):,} "
            f"joint_coverage={float(coverage_report.get('feature_recent_joint_coverage', float('nan'))):.1%} "
            f"candidates={len(coverage_features)}->{len(candidates)}",
            flush=True,
        )
        frozen_features, selection_report = _build_feature_map(
            feature_selection_train, candidates, args.min_local_rows, args.seed
        )
        selection_report.to_csv(
            args.output_dir / "pre_mlp_feature_selection_report.csv", index=False
        )
    else:
        print(
            f"Reusing frozen local feature sets={len(frozen_features)} from "
            f"{frozen_feature_source}; refitting all MLP/calibration models",
            flush=True,
        )
    print(f"Frozen local feature sets={len(frozen_features)}", flush=True)
    all_metrics: list[dict[str, Any]] = []
    scored = test.copy(deep=False)
    artifacts: dict[str, Any] = {}
    requested_arms = tuple(a.strip() for a in args.arms.split(",") if a.strip())
    unknown = sorted(set(requested_arms) - set(ARMS))
    if unknown:
        raise ValueError(f"unknown arms: {unknown}")
    mlp_hpo_params, mlp_params_source = _load_mlp_params(args.mlp_params_json)
    if "mlp_direct" in requested_arms and args.run_mlp_hpo:
        hpo_history = history
        if args.expanding_walkforward:
            parent_meta_start = pd.Timestamp(args.parent_meta_oos_start, tz="UTC")
            hpo_history = history.loc[history["__ts__"].ge(parent_meta_start)]
        print(
            f"Starting chronological MLP HPO trials={args.mlp_hpo_trials}",
            flush=True,
        )
        mlp_hpo_params, mlp_hpo_results = _tune_mlp_hyperparameters(
            hpo_history,
            candidates,
            ae_cols,
            frozen_features,
            min_rows=args.min_local_rows,
            seed=args.seed + 17_000,
            n_trials=args.mlp_hpo_trials,
            valid_start=(
                pd.Timestamp(args.walkforward_tuning_start, tz="UTC")
                if args.expanding_walkforward
                else pd.Timestamp("2026-02-01", tz="UTC")
            ),
            valid_end=(
                pd.Timestamp(args.walkforward_tuning_start, tz="UTC")
                + pd.offsets.MonthBegin(1)
                if args.expanding_walkforward
                else pd.Timestamp("2026-03-01", tz="UTC")
            ),
            scoring_overlay_params=frozen_overlay_params,
        )
        mlp_hpo_results.to_csv(
            args.output_dir / "mlp_direct_model_hpo.csv", index=False
        )
        mlp_params_source = "chronological_mlp_hpo"
        _write_mlp_params(
            args.output_dir / "best_mlp_params.json",
            mlp_hpo_params,
            source=mlp_params_source,
        )
        print(f"MLP HPO winner={mlp_hpo_params}", flush=True)
    elif "mlp_direct" in requested_arms:
        print(
            f"MLP HPO disabled; params_source={mlp_params_source} "
            f"params={mlp_hpo_params}",
            flush=True,
        )
        _write_mlp_params(
            args.output_dir / "best_mlp_params.json",
            mlp_hpo_params,
            source=mlp_params_source,
        )
    if args.expanding_walkforward:
        if requested_arms != ("mlp_direct",):
            raise ValueError("expanding walk-forward currently supports --arms mlp_direct")
        all_rows = (
            pd.concat([history, test], ignore_index=True)
            .drop_duplicates(KEYS, keep="last")
            .sort_values("__ts__", kind="stable")
            .reset_index(drop=True)
        )
        parent_meta_start = pd.Timestamp(args.parent_meta_oos_start, tz="UTC")
        all_rows = all_rows.loc[all_rows["__ts__"].ge(parent_meta_start)].reset_index(
            drop=True
        )
        tuning_start = pd.Timestamp(args.walkforward_tuning_start, tz="UTC")
        if tuning_start <= parent_meta_start:
            raise ValueError(
                "walkforward tuning must start after parent-meta OOS eligibility "
                "so the first fold has prior training rows"
            )
        scored, metrics, artifacts = _run_expanding_walkforward(
            all_rows=all_rows,
            candidates=candidates,
            ae_cols=ae_cols,
            frozen_features=frozen_features,
            mlp_params=mlp_hpo_params,
            min_rows=args.min_local_rows,
            seed=args.seed,
            tuning_start=tuning_start,
            policy_start=pd.Timestamp(args.walkforward_policy_start, tz="UTC"),
            end=eval_end,
            output_dir=args.output_dir,
            catalog=catalog,
        )
        scored.to_parquet(
            args.output_dir / "oos_predictions.parquet", index=False, compression="zstd"
        )
        metrics.to_csv(args.output_dir / "summary.csv", index=False)
        monthly_metrics = _monthly_rank_contract_metrics(scored)
        monthly_metrics.to_csv(
            args.output_dir / "monthly_rank_contract_metrics.csv", index=False
        )
        joblib.dump(
            {"mlp_direct": artifacts}, args.output_dir / "encoder_calibrators.joblib"
        )
        manifest = {
            "schema": "meta_market_state_encoder_expanding_walkforward_v1",
            "direct_predecessor": PREDECESSOR_ID,
            "composite_policy_id": COMPOSITE_POLICY_ID,
            "walkforward_tuning_start": args.walkforward_tuning_start,
            "walkforward_policy_start": args.walkforward_policy_start,
            "history_train_end": args.history_train_end,
            "parent_meta_oos_start": args.parent_meta_oos_start,
            "evaluation_end": str(eval_end),
            "folds": artifacts["fold_manifest"],
            "mlp_hpo_params": mlp_hpo_params,
            "mlp_hpo_enabled": bool(args.run_mlp_hpo),
            "mlp_params_source": mlp_params_source,
            "frozen_feature_source": frozen_feature_source,
            "blacklisted_side_archetypes": list(BLACKLISTED_SIDE_ARCHETYPES),
            "source_coverage": coverage,
            "rank_contract": {
                "parent_admission_threshold": 0.90,
                "comparison_activity": "same monthly row count as parent admission",
                "monthly_metrics": "monthly_rank_contract_metrics.csv",
            },
            "walkforward_rank_history": "walkforward_rank_history.parquet",
            "oos_prediction_contract": (
                "Each row in oos_predictions.parquet is produced by the monthly "
                "model and hierarchical EV map fitted only on earlier rows."
            ),
            "forward_bundle_contract": (
                "The exported policy models and EV map are refitted through the "
                "evaluation end for future inference only; they do not generate "
                "the reported OOS predictions."
            ),
            "leakage_contract": (
                "MLP HPO uses the first chronological tuning month only. Parameters "
                "are frozen. Each monthly OOS prediction is generated by models fitted "
                "only on prior rows. Overlay tuning uses pre-policy OOF months. Each "
                "monthly EV map uses only accumulated prior OOF predictions/outcomes."
            ),
        }
        (args.output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, default=str) + "\n"
        )
        print(metrics.to_string(index=False), flush=True)
        return
    for arm in requested_arms:
        print(f"Starting arm={arm}", flush=True)
        fold_frames, fold_corr, fold_conf = [], [], []
        for fold_no, (valid_start, valid_end) in enumerate(fold_specs):
            train = history.loc[history["__ts__"].lt(valid_start)]
            valid = history.loc[history["__ts__"].ge(valid_start) & history["__ts__"].lt(valid_end)].copy()
            models = _fit_models(
                train, arm, candidates, ae_cols, args.min_local_rows,
                args.seed + fold_no * 1000, frozen_features,
                mlp_params=mlp_hpo_params if arm == "mlp_direct" else None,
            )
            corr, conf, _ = _predict_models(valid, models, ae_cols)
            valid["__fold__"] = fold_no
            fold_frames.append(valid); fold_corr.append(corr); fold_conf.append(conf)
        folds = pd.concat(fold_frames, ignore_index=True)
        fold_ts = pd.to_datetime(folds["__ts__"], utc=True, errors="coerce")
        fold_day = fold_ts.dt.floor("D")
        folds["__week_code__"] = pd.factorize(
            fold_day - pd.to_timedelta(fold_day.dt.weekday.to_numpy(), unit="D"),
            sort=True,
        )[0].astype(np.int32)
        folds["__month_code__"] = pd.factorize(
            fold_ts.dt.strftime("%Y-%m"), sort=True
        )[0].astype(np.int16)
        corr_oof = np.concatenate(fold_corr); conf_oof = np.concatenate(fold_conf)
        best, search = _tune_overlay(folds, corr_oof, conf_oof)
        oof_rank = _overlay_score(folds, corr_oof, conf_oof, best)
        ev_calibrator = fit_hierarchical_ev_calibrator(
            folds,
            oof_rank,
            _num(folds, "ev_after_1pct"),
            shrink_rows=2_000.0,
            min_local_rows=600,
            local_weight_cap=0.50,
            tail_weight_top10=4.0,
            rank_blend=1.0,
        )
        print(
            f"Tuned arm={arm} objective={float(best['objective']):.6f} "
            f"alpha={float(best['alpha']):.4f}",
            flush=True,
        )
        search.to_csv(args.output_dir / f"{arm}_overlay_search.csv", index=False)
        all_local_zero = all(
            abs(float(value)) < 1e-12 for value in best.get("local_alpha", {}).values()
        )
        if abs(float(best["alpha"])) < 1e-12 and all_local_zero:
            final_models = []
            corr = np.zeros(len(test), dtype=np.float32)
            conf = np.zeros(len(test), dtype=np.float32)
            entropy = np.ones(len(test), dtype=np.float32)
        else:
            final_models = _fit_models(
                history, arm, candidates, ae_cols, args.min_local_rows,
                args.seed + 9000, frozen_features,
                mlp_params=mlp_hpo_params if arm == "mlp_direct" else None,
            )
            corr, conf, entropy = _predict_models(test, final_models, ae_cols)
        rank = _overlay_score(test, corr, conf, best)
        expected_ev = predict_hierarchical_ev(ev_calibrator, test, rank)
        # Expected EV is the common cross-archetype unit. A tiny rank tie-break
        # preserves deterministic ordering across isotonic plateaus.
        ev_order_score = expected_ev_rank(ev_calibrator, expected_ev, rank)
        scored[f"rank_{arm}"] = rank
        scored[f"expected_net_ev_after_1pct_{arm}"] = expected_ev
        scored[f"state_ev_correction_{arm}"] = corr
        scored[f"state_posterior_confidence_{arm}"] = conf
        scored[f"state_posterior_entropy_{arm}"] = entropy
        budget = int(np.sum(_num(test, "policy_parent_rank") >= 0.90))
        baseline = _economic_metrics(test, _num(test, "policy_parent_rank"), budget, target_activity=budget)
        metric = _economic_metrics(test, rank, budget, target_activity=budget)
        ev_metric = _economic_metrics(test, ev_order_score, budget, target_activity=budget)
        all_metrics.extend([
            {"arm": "parent_95", "comparison_arm": arm, **baseline, "objective": _objective(baseline, baseline)},
            {"arm": arm, "comparison_arm": arm, **metric, "objective": _objective(metric, baseline)},
            {"arm": f"{arm}_ev_calibrated", "comparison_arm": arm, **ev_metric, "objective": _objective(ev_metric, baseline)},
        ])
        artifacts[arm] = {
            "params": best,
            "models": final_models,
            "ev_calibrator": ev_calibrator,
            "mlp_hpo_params": mlp_hpo_params if arm == "mlp_direct" else None,
        }
        if arm == "mlp_direct" and final_models:
            policy_path = _export_composite_policy(
                args.output_dir,
                test,
                final_models,
                best,
                ev_calibrator,
                predecessor_references,
            )
            print(f"Exported composite policy={policy_path}", flush=True)
    scored.to_parquet(args.output_dir / "oos_predictions.parquet", index=False, compression="zstd")
    metrics = pd.DataFrame(all_metrics)
    metrics.to_csv(args.output_dir / "summary.csv", index=False)
    joblib.dump(artifacts, args.output_dir / "encoder_calibrators.joblib")
    manifest = {
        "schema": "meta_market_state_encoder_ablation_v1",
        "direct_predecessor": PREDECESSOR_ID,
        "direct_predecessor_artifact": str(PREDECESSOR_ARTIFACT),
        "underlying_model_basis": PARENT,
        "parent_params": parent_params,
        "arms": list(requested_arms),
        "mlp_hpo_enabled": bool(args.run_mlp_hpo),
        "mlp_hpo_params": mlp_hpo_params,
        "mlp_params_source": mlp_params_source,
        "objective": "mean_top10_ev + .25*(worst_week+q10_week+q20_week+q30_week) - small_activity_penalty; worst-week/month degradation allowed only up to one fifth of positive average-EV gain",
        "rolling_tune_folds": [[str(a), str(b)] for a, b in fold_specs],
        "evaluation": ["2026-04-01", str(eval_end)],
        "candidate_features": candidates,
        "candidate_reliability_features": [
            c for c in candidates
            if any(token in c.lower() for token in RELIABILITY_CONTEXT_TOKENS)
        ],
        "frozen_feature_source": frozen_feature_source,
        "expected_ev_contract": "Hierarchical isotonic expected net EV after 1% cost; global curve plus support-shrunk side x archetype curves fit on February/March rolling-OOF rows only.",
        "composite_policy_id": COMPOSITE_POLICY_ID,
        "blacklisted_side_archetypes": list(BLACKLISTED_SIDE_ARCHETYPES),
        "ae_features": ae_cols,
        "coverage": coverage,
        "leakage_contract": "All MLP/GMM/scaler fits and cluster EV priors use fold-train rows only. OOS transforms use observable pre-entry features. February/March select overlay parameters; April-June is untouched evaluation. The direct predecessor is the frozen v9 forced-local-tail-0.950 down-only overlay; its lifecycle model is only the underlying score basis. Candidate rows and predecessor rank are fixed across arms.",
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    print(metrics.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
