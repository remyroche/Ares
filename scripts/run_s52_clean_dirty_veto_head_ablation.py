#!/usr/bin/env python3
"""S52 clean-vs-dirty veto head ablation.

This diagnostic keeps the broad S52 opportunity ranker fixed, then trains a
month-forward clean-vs-dirty head on prior months only. The veto head is applied
to the current month as a deterministic agreement layer. The purpose is to test
whether the learnable clean-vs-dirty signal can reduce dirty path selection
without reworking the base ranker again.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_clean_dirty_learnability_oos import (  # noqa: E402
    _fit_predict_clean_head_by_side,
    _positive_clean_variant,
)
from scripts.run_gate3_side_soft_label_hpo import (  # noqa: E402
    DEFAULT_FEATURE_DIR,
    DEFAULT_FEATURE_LIST_CSV,
    DEFAULT_ROUND_TRIP_COST,
    _json_safe,
    _objective,
    _prepare_folds,
    _score_fold,
    _summarize_trial,
)
from scripts.run_s52_ranker_smoke import (  # noqa: E402
    DEFAULT_BEST_CONFIG,
    DEFAULT_LABELS_PATH,
    DEFAULT_MONTHS,
    DEFAULT_RANKER_PARAMS,
    LabelConfig,
    _cap_indices,
    _fit_ranker,
    _load_config,
    _materialized_soft_label,
    _ranker_sample_weight,
    _scored_ledger,
)


DEFAULT_OUTPUT_DIR = Path("data_perp/reports/s52_clean_dirty_veto_head_ablation_20260705_v1")
DEFAULT_LABEL_VARIANT = "positive_econ_sideaware_short_decisive"
DEFAULT_BASE_VARIANT = "ranker_timestamp_soft_ordered_ev"
DEFAULT_ALPHA_GRID = (0.0, 0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 1.0)
S52_MATERIALIZED_LABEL_VARIANTS = {
    "s52_materialized_path_clean",
    "materialized_s52_path_clean",
    "s52_path_clean",
}


def _side_values(frame: pd.DataFrame) -> pd.Series:
    if "side" in frame.columns:
        return pd.to_numeric(frame["side"], errors="coerce").fillna(1.0).reset_index(drop=True)
    if "__side__" in frame.columns:
        return pd.to_numeric(frame["__side__"], errors="coerce").fillna(1.0).reset_index(drop=True)
    if "side_name" in frame.columns:
        return pd.Series(
            np.where(frame["side_name"].astype(str).str.lower().eq("short"), -1.0, 1.0),
            index=frame.index,
        ).reset_index(drop=True)
    return pd.Series(1.0, index=frame.reset_index(drop=True).index)


def _rank_pct(score: pd.Series) -> pd.Series:
    values = pd.to_numeric(score, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if int(values.notna().sum()) == 0:
        return pd.Series(0.0, index=score.index, dtype=np.float64)
    return values.rank(method="average", pct=True).fillna(0.0).astype(np.float64)


def _logit_prob(prob: pd.Series) -> pd.Series:
    p = pd.to_numeric(prob, errors="coerce").fillna(0.0).clip(1e-4, 1.0 - 1e-4)
    return pd.Series(np.log(p / (1.0 - p)), index=prob.index, dtype=np.float64)


def _parse_alpha_grid(value: str | None) -> list[float]:
    if value is None or not str(value).strip():
        return list(DEFAULT_ALPHA_GRID)
    out: list[float] = []
    for part in str(value).split(","):
        if not part.strip():
            continue
        alpha = float(part)
        if not math.isfinite(alpha):
            continue
        out.append(float(np.clip(alpha, 0.0, 1.0)))
    return sorted(set(out)) or list(DEFAULT_ALPHA_GRID)


def _agreement_score(base_score: pd.Series, clean_prob: pd.Series, *, alpha: float) -> pd.Series:
    clean_weight = float(np.clip(alpha, 0.0, 1.0))
    base_rank = _rank_pct(base_score)
    clean = pd.to_numeric(clean_prob, errors="coerce").fillna(0.0).clip(0.0, 1.0)
    return ((1.0 - clean_weight) * base_rank + clean_weight * clean).astype(np.float64)


def _combine_scores(
    base_score: pd.Series,
    clean_prob: pd.Series,
    *,
    policy: str,
    alpha: float | None = None,
) -> pd.Series:
    """Combine opportunity score and OOS clean-path probability.

    Policies are deterministic; no threshold is selected on the validation
    month. Ranking is performed on the returned score.
    """
    policy_norm = str(policy).strip().lower()
    base_rank = _rank_pct(base_score)
    clean = pd.to_numeric(clean_prob, errors="coerce").fillna(0.0).clip(0.0, 1.0)
    if policy_norm in {"base", "baseline"}:
        return pd.to_numeric(base_score, errors="coerce").fillna(0.0).astype(np.float64)
    if policy_norm in {"calibrated_agreement", "calibrated_agreement_path_ordered"}:
        if alpha is None:
            raise ValueError(f"{policy} requires a calibrated alpha")
        return _agreement_score(base_score, clean, alpha=float(alpha))
    if policy_norm.startswith("agreement_alpha_"):
        return _agreement_score(base_score, clean, alpha=float(policy_norm.rsplit("_", 1)[-1]))
    if policy_norm == "agreement_30":
        return (0.70 * base_rank + 0.30 * clean).astype(np.float64)
    if policy_norm == "agreement_10":
        return (0.90 * base_rank + 0.10 * clean).astype(np.float64)
    if policy_norm == "agreement_20":
        return (0.80 * base_rank + 0.20 * clean).astype(np.float64)
    if policy_norm == "agreement_50":
        return (0.50 * base_rank + 0.50 * clean).astype(np.float64)
    if policy_norm == "multiplicative":
        return (base_rank * clean).astype(np.float64)
    if policy_norm == "multiplicative_sqrt":
        return (base_rank * np.sqrt(clean)).astype(np.float64)
    if policy_norm == "logit_agreement":
        return (base_rank + 0.15 * _logit_prob(clean)).astype(np.float64)
    if policy_norm == "veto_50":
        return (base_rank * np.where(clean.ge(0.50), 1.0, 0.10)).astype(np.float64)
    if policy_norm == "veto_60":
        return (base_rank * np.where(clean.ge(0.60), 1.0, 0.05)).astype(np.float64)
    if policy_norm == "veto_70":
        return (base_rank * np.where(clean.ge(0.70), 1.0, 0.02)).astype(np.float64)
    raise ValueError(f"unknown veto policy: {policy}")


def _policy_alpha(policy: str, calibrated_alpha: float | None = None) -> float:
    policy_norm = str(policy).strip().lower()
    if policy_norm in {"calibrated_agreement", "calibrated_agreement_path_ordered"}:
        return float("nan") if calibrated_alpha is None else float(calibrated_alpha)
    if policy_norm.startswith("agreement_alpha_"):
        return float(policy_norm.rsplit("_", 1)[-1])
    fixed = {
        "agreement_10": 0.10,
        "agreement_20": 0.20,
        "agreement_30": 0.30,
        "agreement_50": 0.50,
    }
    return float(fixed.get(policy_norm, float("nan")))


def _month_values(frame: pd.DataFrame) -> pd.Series:
    if "__ts__" not in frame.columns:
        return pd.Series("", index=pd.RangeIndex(len(frame)), dtype="object")
    ts = pd.to_datetime(frame["__ts__"], errors="coerce")
    return ts.dt.to_period("M").astype(str).reset_index(drop=True)


def _select_calibrated_alpha(
    alpha_rows: dict[float, list[dict[str, Any]]],
    *,
    objective_mode: str,
    default_alpha: float,
) -> dict[str, Any]:
    best_alpha = float(default_alpha)
    best_objective = float("-inf")
    objectives: dict[str, float] = {}
    for alpha, rows in sorted(alpha_rows.items()):
        if not rows:
            continue
        objective = _objective(rows, objective_mode=str(objective_mode))
        objectives[f"{float(alpha):.6g}"] = float(objective)
        if math.isfinite(objective) and objective > best_objective:
            best_alpha = float(alpha)
            best_objective = float(objective)
    if not math.isfinite(best_objective):
        best_objective = float("nan")
    return {
        "alpha": best_alpha,
        "objective": best_objective,
        "objectives": objectives,
        "calibration_folds": int(max((len(rows) for rows in alpha_rows.values()), default=0)),
    }


def _base_score_for_fold(
    fold: dict[str, Any],
    *,
    fold_i: int,
    max_train_rows: int,
    round_trip_cost: float,
    seed: int,
    ranker_params: dict[str, Any],
) -> tuple[np.ndarray, pd.DataFrame]:
    train_label_full = _materialized_soft_label(fold["train_frame"], fold["train_metrics"])
    idx = _cap_indices(int(fold["train_rows"]), int(max_train_rows), seed=int(seed) + fold_i * 17)
    x_train = fold["x_train"].iloc[idx].reset_index(drop=True)
    train_frame = fold["train_frame"].iloc[idx].reset_index(drop=True)
    train_metrics = fold["train_metrics"].iloc[idx].reset_index(drop=True)
    train_label = train_label_full.iloc[idx].reset_index(drop=True)
    weights = _ranker_sample_weight(
        train_metrics,
        train_label,
        round_trip_cost=float(round_trip_cost),
        mode="base",
    )
    score = _fit_ranker(
        x_train,
        train_frame,
        train_metrics,
        train_label,
        weights,
        fold["x_valid"],
        group_mode="timestamp",
        relevance_mode="soft_ordered_ev",
        round_trip_cost=float(round_trip_cost),
        seed=int(seed) + fold_i,
        ranker_params=ranker_params,
    )
    valid_label = _materialized_soft_label(fold["valid_frame"], fold["valid_metrics"])
    return score, valid_label.reset_index(drop=True)


def _veto_population_and_label(
    frame: pd.DataFrame,
    metrics: pd.DataFrame,
    *,
    label_variant: str,
) -> tuple[np.ndarray, pd.Series, dict[str, Any]]:
    variant = str(label_variant).strip().lower()
    if variant in S52_MATERIALIZED_LABEL_VARIANTS:
        label = _materialized_soft_label(frame, metrics)
        target = pd.to_numeric(label["target_hard"], errors="coerce").fillna(0).astype(int)
        population = (
            pd.to_numeric(label.get("positive_u"), errors="coerce").fillna(0).gt(0.5)
            | pd.to_numeric(label.get("dirty_positive"), errors="coerce").fillna(0).gt(0.5)
            | pd.to_numeric(label.get("first_pass_good"), errors="coerce").fillna(0).gt(0.5)
            | pd.to_numeric(label.get("first_pass_bad"), errors="coerce").fillna(0).gt(0.5)
            | target.gt(0)
        )
        return population.to_numpy(dtype=bool), target.reset_index(drop=True), {
            "label_variant": str(label_variant),
            "label_description": "materialized S52 first-touch path clean label",
            "positive_rows": int(population.sum()),
            "clean_rows": int(target.sum()),
            "dirty_rows": int((population & target.eq(0)).sum()),
            "clean_rate": float(target.sum() / max(int(population.sum()), 1)),
        }
    return _positive_clean_variant(metrics, variant=str(label_variant), frame=frame)


def _clean_head_score_for_fold(
    fold: dict[str, Any],
    *,
    label_variant: str,
    fold_i: int,
    model_kind: str,
    lgbm_hpo: bool,
    side_feature_select_top_k: int,
) -> tuple[pd.Series, dict[str, Any]]:
    train_pop, train_label, train_diag = _veto_population_and_label(
        fold["train_frame"],
        fold["train_metrics"],
        label_variant=str(label_variant),
    )
    valid_pop, valid_label, valid_diag = _veto_population_and_label(
        fold["valid_frame"],
        fold["valid_metrics"],
        label_variant=str(label_variant),
    )
    train_pop_s = pd.Series(train_pop, index=pd.RangeIndex(len(train_pop))).astype(bool)
    y_train = train_label.iloc[train_pop_s.to_numpy(dtype=bool)].reset_index(drop=True)
    x_train = (
        fold["x_train"]
        .iloc[train_pop_s.to_numpy(dtype=bool)]
        .reset_index(drop=True)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .astype(np.float32)
    )
    x_valid = (
        fold["x_valid"]
        .reset_index(drop=True)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .astype(np.float32)
    )
    train_side = _side_values(fold["train_frame"].iloc[train_pop_s.to_numpy(dtype=bool)].reset_index(drop=True))
    valid_side = _side_values(fold["valid_frame"].reset_index(drop=True))
    if len(y_train) < 500 or int(y_train.sum()) < 100 or int((1 - y_train).sum()) < 100:
        return pd.Series(0.0, index=x_valid.index, dtype=np.float32), {
            "clean_head_status": "insufficient_rows",
            "clean_head_train_rows": int(len(y_train)),
            "clean_head_train_clean_rows": int(y_train.sum()),
            "clean_head_valid_population_rows": int(np.asarray(valid_pop, dtype=bool).sum()),
            **{f"train_{k}": v for k, v in train_diag.items() if k != "label_variant"},
            **{f"valid_{k}": v for k, v in valid_diag.items() if k != "label_variant"},
        }
    score, model_diag = _fit_predict_clean_head_by_side(
        x_train=x_train,
        y_train=y_train,
        train_side=train_side,
        x_valid=x_valid,
        valid_side=valid_side,
        seed=52000 + int(fold_i),
        model_kind=str(model_kind),
        lgbm_hpo=bool(lgbm_hpo),
        side_feature_select_top_k=int(side_feature_select_top_k),
    )
    return score.reset_index(drop=True).astype(np.float32), {
        "clean_head_status": str(model_diag.get("model_status", "ok")),
        "clean_head_model_kind": str(model_kind),
        "clean_head_lgbm_hpo": bool(lgbm_hpo),
        "clean_head_train_rows": int(len(y_train)),
        "clean_head_train_clean_rows": int(y_train.sum()),
        "clean_head_train_dirty_rows": int((1 - y_train).sum()),
        "clean_head_valid_population_rows": int(np.asarray(valid_pop, dtype=bool).sum()),
        "clean_head_valid_clean_rows": int(valid_label.iloc[np.asarray(valid_pop, dtype=bool)].sum()),
        "clean_head_score_mean": float(pd.to_numeric(score, errors="coerce").mean()),
        **{f"clean_head_{k}": v for k, v in model_diag.items()},
        **{f"train_{k}": v for k, v in train_diag.items() if k != "label_variant"},
        **{f"valid_{k}": v for k, v in valid_diag.items() if k != "label_variant"},
    }


def _calibrate_agreement_alpha_for_fold(
    fold: dict[str, Any],
    *,
    outer_fold_i: int,
    alpha_grid: list[float],
    calibration_objective: str,
    max_calibration_months: int,
    default_alpha: float,
    max_train_rows: int,
    round_trip_cost: float,
    seed: int,
    ranker_params: dict[str, Any],
    label_variant: str,
    model_kind: str,
    lgbm_hpo: bool,
    side_feature_select_top_k: int,
) -> dict[str, Any]:
    month_s = _month_values(fold["train_frame"])
    months = sorted([m for m in month_s.dropna().unique().tolist() if str(m) and str(m).lower() != "nat"])
    candidate_months = months[1:]
    if int(max_calibration_months) > 0:
        candidate_months = candidate_months[-int(max_calibration_months) :]
    alpha_rows: dict[float, list[dict[str, Any]]] = {float(alpha): [] for alpha in alpha_grid}
    used_months: list[str] = []
    skipped_months: list[str] = []
    for inner_i, month in enumerate(candidate_months):
        train_mask = month_s < month
        valid_mask = month_s == month
        if int(train_mask.sum()) < 1000 or int(valid_mask.sum()) < 200:
            skipped_months.append(str(month))
            continue
        subfold = {
            "month": str(month),
            "x_train": fold["x_train"].loc[train_mask.to_numpy(dtype=bool)].reset_index(drop=True),
            "x_valid": fold["x_train"].loc[valid_mask.to_numpy(dtype=bool)].reset_index(drop=True),
            "train_frame": fold["train_frame"].loc[train_mask.to_numpy(dtype=bool)].reset_index(drop=True),
            "valid_frame": fold["train_frame"].loc[valid_mask.to_numpy(dtype=bool)].reset_index(drop=True),
            "train_metrics": fold["train_metrics"].loc[train_mask.to_numpy(dtype=bool)].reset_index(drop=True),
            "valid_metrics": fold["train_metrics"].loc[valid_mask.to_numpy(dtype=bool)].reset_index(drop=True),
            "train_rows": int(train_mask.sum()),
            "valid_rows": int(valid_mask.sum()),
        }
        base_score, valid_label = _base_score_for_fold(
            subfold,
            fold_i=outer_fold_i * 100 + inner_i,
            max_train_rows=int(max_train_rows),
            round_trip_cost=float(round_trip_cost),
            seed=int(seed),
            ranker_params=ranker_params,
        )
        clean_prob, _clean_diag = _clean_head_score_for_fold(
            subfold,
            label_variant=str(label_variant),
            fold_i=outer_fold_i * 100 + inner_i,
            model_kind=str(model_kind),
            lgbm_hpo=bool(lgbm_hpo),
            side_feature_select_top_k=int(side_feature_select_top_k),
        )
        for alpha in alpha_grid:
            score = _agreement_score(pd.Series(base_score), clean_prob, alpha=float(alpha))
            row = _score_fold(
                score.reset_index(drop=True),
                valid_label.reset_index(drop=True),
                subfold["valid_metrics"].reset_index(drop=True),
                str(month),
                round_trip_cost=float(round_trip_cost),
            )
            alpha_rows[float(alpha)].append(row)
        used_months.append(str(month))
    selected = _select_calibrated_alpha(
        alpha_rows,
        objective_mode=str(calibration_objective),
        default_alpha=float(default_alpha),
    )
    selected.update(
        {
            "calibration_objective": str(calibration_objective),
            "calibration_months": used_months,
            "skipped_calibration_months": skipped_months,
            "default_alpha": float(default_alpha),
        }
    )
    return selected


def _annotate_ledger(
    ledger: pd.DataFrame,
    *,
    clean_prob: pd.Series,
    policy: str,
    alpha: float | None = None,
) -> pd.DataFrame:
    out = ledger.copy()
    out["clean_veto_prob"] = pd.to_numeric(clean_prob, errors="coerce").reset_index(drop=True).to_numpy()
    out["veto_policy"] = str(policy)
    out["clean_veto_alpha"] = float("nan") if alpha is None else float(alpha)
    for frac in (0.10, 0.20, 0.30):
        col = f"selected_top{int(round(frac * 100)):02d}"
        if col in out.columns:
            selected = out[col].astype(bool)
            out[f"{col}_clean_veto_prob_mean"] = float(
                pd.to_numeric(out.loc[selected, "clean_veto_prob"], errors="coerce").mean()
            ) if bool(selected.any()) else float("nan")
    return out


def run_ablation(
    *,
    labels_path: Path,
    feature_dir: Path,
    feature_list_csv: Path,
    best_config_path: Path,
    output_dir: Path,
    months: list[str],
    max_train_rows: int,
    round_trip_cost: float,
    label_variant: str,
    policies: list[str],
    alpha_grid: list[float],
    calibration_objective: str,
    max_calibration_months: int,
    default_calibrated_alpha: float,
    model_kind: str,
    lgbm_hpo: bool,
    side_feature_select_top_k: int,
    seed: int,
    include_ae_gmm_state_features: bool,
    ae_gmm_state_feature_max_train_rows: int,
    ae_gmm_state_feature_max_iter: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    config = _load_config(best_config_path)
    folds, manifest = _prepare_folds(
        labels_path=labels_path,
        feature_dir=feature_dir,
        feature_list_csv=feature_list_csv,
        months=months,
        spread_baseline_path=None,
        spread_rank_column="p75_spread_bps",
        target_symbol_count=None,
        max_feature_store_features=None,
        include_ae_gmm_state_features=bool(include_ae_gmm_state_features),
        ae_gmm_state_feature_max_train_rows=int(ae_gmm_state_feature_max_train_rows),
        ae_gmm_state_feature_max_iter=int(ae_gmm_state_feature_max_iter),
        seed=int(seed),
    )
    ranker_params = dict(DEFAULT_RANKER_PARAMS)
    fold_rows: list[dict[str, Any]] = []
    ledgers: list[pd.DataFrame] = []
    policies_norm = list(dict.fromkeys(["base", *[str(p).strip() for p in policies if str(p).strip()]]))
    for fold_i, fold in enumerate(folds):
        base_score, valid_label = _base_score_for_fold(
            fold,
            fold_i=fold_i,
            max_train_rows=int(max_train_rows),
            round_trip_cost=float(round_trip_cost),
            seed=int(seed),
            ranker_params=ranker_params,
        )
        clean_prob, clean_diag = _clean_head_score_for_fold(
            fold,
            label_variant=str(label_variant),
            fold_i=fold_i,
            model_kind=str(model_kind),
            lgbm_hpo=bool(lgbm_hpo),
            side_feature_select_top_k=int(side_feature_select_top_k),
        )
        calibration_cache: dict[str, dict[str, Any]] = {}
        if any(str(policy).strip().lower() in {"calibrated_agreement", "calibrated_agreement_path_ordered"} for policy in policies_norm):
            calibration_cache["calibrated_agreement"] = _calibrate_agreement_alpha_for_fold(
                fold,
                outer_fold_i=fold_i,
                alpha_grid=list(alpha_grid),
                calibration_objective=str(calibration_objective),
                max_calibration_months=int(max_calibration_months),
                default_alpha=float(default_calibrated_alpha),
                max_train_rows=int(max_train_rows),
                round_trip_cost=float(round_trip_cost),
                seed=int(seed),
                ranker_params=ranker_params,
                label_variant=str(label_variant),
                model_kind=str(model_kind),
                lgbm_hpo=bool(lgbm_hpo),
                side_feature_select_top_k=int(side_feature_select_top_k),
            )
        for policy in policies_norm:
            policy_norm = str(policy).strip().lower()
            calibration_diag: dict[str, Any] = {}
            calibrated_alpha: float | None = None
            if policy_norm in {"calibrated_agreement", "calibrated_agreement_path_ordered"}:
                calibration_diag = calibration_cache.get("calibrated_agreement", {})
                calibrated_alpha = float(calibration_diag.get("alpha", default_calibrated_alpha))
            combined = _combine_scores(pd.Series(base_score), clean_prob, policy=policy, alpha=calibrated_alpha)
            variant = f"{DEFAULT_BASE_VARIANT}__clean_veto_{policy}"
            row = _score_fold(
                combined.reset_index(drop=True),
                valid_label.reset_index(drop=True),
                fold["valid_metrics"].reset_index(drop=True),
                str(fold["month"]),
                round_trip_cost=float(round_trip_cost),
            )
            selected10 = _scored_ledger(
                variant=variant,
                fold=fold,
                score=combined.to_numpy(dtype=np.float32),
                valid_label=valid_label.reset_index(drop=True),
            )
            row.update(
                {
                    "variant": variant,
                    "stage": variant,
                    "trial_number": 0,
                    "label_name": f"{config.name}_{variant}",
                    "family": config.family,
                    "base_variant": DEFAULT_BASE_VARIANT,
                    "veto_policy": str(policy),
                    "clean_veto_alpha": _policy_alpha(policy, calibrated_alpha),
                    "clean_veto_alpha_objective": calibration_diag.get("objective"),
                    "clean_veto_alpha_objective_mode": calibration_diag.get("calibration_objective"),
                    "clean_veto_alpha_calibration_folds": calibration_diag.get("calibration_folds"),
                    "clean_veto_alpha_calibration_months": ",".join(calibration_diag.get("calibration_months", [])),
                    "clean_veto_alpha_skipped_calibration_months": ",".join(
                        calibration_diag.get("skipped_calibration_months", [])
                    ),
                    "clean_veto_alpha_objectives": json.dumps(
                        _json_safe(calibration_diag.get("objectives", {})),
                        sort_keys=True,
                    ),
                    "clean_label_variant": str(label_variant),
                    "train_rows": int(min(max_train_rows, fold["train_rows"])),
                    "train_rows_uncapped": int(fold["train_rows"]),
                    "valid_rows": int(fold["valid_rows"]),
                    "ranker_params": json.dumps(_json_safe(ranker_params), sort_keys=True),
                    "round_trip_cost": float(round_trip_cost),
                    "target_source": "materialized",
                    **clean_diag,
                }
            )
            fold_rows.append(row)
            ledgers.append(_annotate_ledger(selected10, clean_prob=clean_prob, policy=policy, alpha=_policy_alpha(policy, calibrated_alpha)))

    summaries: list[dict[str, Any]] = []
    for variant, group in pd.DataFrame(fold_rows).groupby("variant", observed=True, dropna=False):
        rows = group.to_dict(orient="records")
        summary = _summarize_trial(
            str(variant),
            0,
            LabelConfig(name=f"{config.name}_{variant}", family=config.family, long=config.long, short=config.short),
            rows,
            objective_mode="precision_topk",
        )
        summary["variant"] = str(variant)
        summary["clean_label_variant"] = str(label_variant)
        summary["veto_policy"] = str(group["veto_policy"].iloc[0])
        summary["objective_path_ordered"] = _objective(rows, objective_mode="path_ordered")
        if "clean_veto_alpha" in group.columns:
            alpha = pd.to_numeric(group["clean_veto_alpha"], errors="coerce")
            summary["mean_clean_veto_alpha"] = float(alpha.mean()) if alpha.notna().any() else float("nan")
            summary["min_clean_veto_alpha"] = float(alpha.min()) if alpha.notna().any() else float("nan")
            summary["max_clean_veto_alpha"] = float(alpha.max()) if alpha.notna().any() else float("nan")
        if "clean_veto_alpha_objective" in group.columns:
            alpha_obj = pd.to_numeric(group["clean_veto_alpha_objective"], errors="coerce")
            summary["mean_clean_veto_alpha_objective"] = (
                float(alpha_obj.mean()) if alpha_obj.notna().any() else float("nan")
            )
        summaries.append(summary)
    summary_df = pd.DataFrame(summaries).sort_values("objective", ascending=False).reset_index(drop=True)
    fold_df = pd.DataFrame(fold_rows)
    ledger_df = pd.concat(ledgers, ignore_index=True) if ledgers else pd.DataFrame()
    paths = {
        "summary": output_dir / "s52_clean_dirty_veto_summary.csv",
        "folds": output_dir / "s52_clean_dirty_veto_folds.csv",
        "scored_ledger": output_dir / "s52_clean_dirty_veto_scored_ledger.parquet",
        "manifest": output_dir / "manifest.json",
        "report": output_dir / "s52_clean_dirty_veto.md",
    }
    summary_df.to_csv(paths["summary"], index=False)
    fold_df.to_csv(paths["folds"], index=False)
    ledger_df.to_parquet(paths["scored_ledger"], index=False)
    out_manifest = {
        "scope": "s52_clean_dirty_veto_head_ablation",
        "labels_path": str(labels_path),
        "feature_dir": str(feature_dir),
        "feature_list_csv": str(feature_list_csv),
        "months": list(months),
        "base_variant": DEFAULT_BASE_VARIANT,
        "clean_label_variant": str(label_variant),
        "policies": policies_norm,
        "alpha_grid": [float(alpha) for alpha in alpha_grid],
        "calibration_objective": str(calibration_objective),
        "max_calibration_months": int(max_calibration_months),
        "default_calibrated_alpha": float(default_calibrated_alpha),
        "model_kind": str(model_kind),
        "lgbm_hpo": bool(lgbm_hpo),
        "max_train_rows": int(max_train_rows),
        "round_trip_cost": float(round_trip_cost),
        "fold_manifest": manifest,
        "outputs": {k: str(v) for k, v in paths.items()},
    }
    paths["manifest"].write_text(json.dumps(_json_safe(out_manifest), indent=2, sort_keys=True), encoding="utf-8")
    _write_report(paths["report"], summary_df, fold_df, out_manifest)
    return out_manifest


def _format_table(df: pd.DataFrame, cols: list[str]) -> str:
    if df.empty:
        return "No rows."
    view = df[[c for c in cols if c in df.columns]].copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda v: f"{float(v):.4f}" if pd.notna(v) else "")
    return view.to_markdown(index=False)


def _write_report(path: Path, summary: pd.DataFrame, folds: pd.DataFrame, manifest: dict[str, Any]) -> None:
    summary_cols = [
        "variant",
        "objective",
        "objective_path_ordered",
        "mean_clean_veto_alpha",
        "min_clean_veto_alpha",
        "max_clean_veto_alpha",
        "mean_top10_ev_weighted_first_touch_precision",
        "mean_top20_ev_weighted_first_touch_precision",
        "mean_top30_ev_weighted_first_touch_precision",
        "mean_top10_clean_precision",
        "mean_top10_first_touch_bad_mae_1r_rate",
        "mean_top10_first_touch_full_path_bad_mae_1r_rate",
        "mean_top10_mae_1r_before_mfe_1r_rate",
        "mean_top10_mean_underwater_bars_before_mfe",
        "mean_top10_mean_max_adverse_before_mfe_1r",
        "mean_top10_mean_ev",
    ]
    fold_cols = [
        "variant",
        "month",
        "top10_ev_weighted_first_touch_precision",
        "top10_clean_precision",
        "top10_first_touch_bad_mae_1r_rate",
        "top10_first_touch_full_path_bad_mae_1r_rate",
        "top10_mae_1r_before_mfe_1r_rate",
        "top10_mean_underwater_bars_before_mfe",
        "top10_mean_max_adverse_before_mfe_1r",
        "top10_mean_ev",
        "clean_veto_alpha",
        "clean_veto_alpha_objective",
        "clean_veto_alpha_calibration_months",
        "clean_head_score_mean",
    ]
    text = [
        "# S52 Clean-vs-Dirty Veto Head Ablation",
        "",
        f"Base variant: `{manifest['base_variant']}`",
        f"Clean label variant: `{manifest['clean_label_variant']}`",
        f"Months: `{', '.join(manifest['months'])}`",
        f"Alpha grid: `{manifest.get('alpha_grid', [])}`",
        f"Calibration objective: `{manifest.get('calibration_objective')}`",
        "",
        "## Summary",
        "",
        _format_table(summary, summary_cols),
        "",
        "## Folds",
        "",
        _format_table(folds, fold_cols),
        "",
        f"- Summary: `{manifest['outputs']['summary']}`",
        f"- Folds: `{manifest['outputs']['folds']}`",
        f"- Scored ledger: `{manifest['outputs']['scored_ledger']}`",
    ]
    path.write_text("\n".join(text) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-path", type=Path, default=DEFAULT_LABELS_PATH)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--feature-list-csv", type=Path, default=DEFAULT_FEATURE_LIST_CSV)
    parser.add_argument("--best-config-path", type=Path, default=DEFAULT_BEST_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--months", default=",".join(DEFAULT_MONTHS))
    parser.add_argument("--max-train-rows", type=int, default=150000)
    parser.add_argument("--round-trip-cost", type=float, default=DEFAULT_ROUND_TRIP_COST)
    parser.add_argument("--label-variant", default=DEFAULT_LABEL_VARIANT)
    parser.add_argument(
        "--policies",
        default=(
            "agreement_10,agreement_20,agreement_30,agreement_50,"
            "multiplicative_sqrt,multiplicative,logit_agreement,veto_50,veto_60,veto_70"
        ),
    )
    parser.add_argument("--alpha-grid", default=",".join(str(v) for v in DEFAULT_ALPHA_GRID))
    parser.add_argument("--calibration-objective", choices=["path_ordered", "precision_topk", "pnl_only"], default="path_ordered")
    parser.add_argument("--max-calibration-months", type=int, default=2)
    parser.add_argument("--default-calibrated-alpha", type=float, default=0.20)
    parser.add_argument("--model-kind", choices=["extratrees", "lgbm"], default="lgbm")
    parser.add_argument("--lgbm-hpo", action="store_true")
    parser.add_argument("--side-feature-select-top-k", type=int, default=0)
    parser.add_argument("--seed", type=int, default=52013)
    parser.add_argument("--no-ae-gmm-state-features", action="store_true")
    parser.add_argument("--ae-gmm-state-feature-max-train-rows", type=int, default=30000)
    parser.add_argument("--ae-gmm-state-feature-max-iter", type=int, default=32)
    args = parser.parse_args(argv)
    manifest = run_ablation(
        labels_path=args.labels_path,
        feature_dir=args.feature_dir,
        feature_list_csv=args.feature_list_csv,
        best_config_path=args.best_config_path,
        output_dir=args.output_dir,
        months=[part.strip() for part in str(args.months).split(",") if part.strip()],
        max_train_rows=int(args.max_train_rows),
        round_trip_cost=float(args.round_trip_cost),
        label_variant=str(args.label_variant),
        policies=[part.strip() for part in str(args.policies).split(",") if part.strip()],
        alpha_grid=_parse_alpha_grid(args.alpha_grid),
        calibration_objective=str(args.calibration_objective),
        max_calibration_months=int(args.max_calibration_months),
        default_calibrated_alpha=float(args.default_calibrated_alpha),
        model_kind=str(args.model_kind),
        lgbm_hpo=bool(args.lgbm_hpo),
        side_feature_select_top_k=int(args.side_feature_select_top_k),
        seed=int(args.seed),
        include_ae_gmm_state_features=not bool(args.no_ae_gmm_state_features),
        ae_gmm_state_feature_max_train_rows=int(args.ae_gmm_state_feature_max_train_rows),
        ae_gmm_state_feature_max_iter=int(args.ae_gmm_state_feature_max_iter),
    )
    print(json.dumps(_json_safe(manifest), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
