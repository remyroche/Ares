"""Opt-in label/sample-weight recipe and Optuna study scaffolding.

Default training applies an explicit ``EPM_LABEL_WEIGHT_RECIPE``/cfg recipe, or
the persisted best recipe when one exists and best-default loading is enabled.
Set ``EPM_LABEL_WEIGHT_DISABLE=1``/``label_weight_disable=true`` to recover the
pre-HPO baseline path with no recipe transforms.
The recipe lets experiments override label geometry, soft-label construction,
sample weights, and self-distillation multipliers without crowding the main
training modules.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from .lgbm_recency_hpo import recency_hpo_decay_from_config
except Exception:  # pragma: no cover - standalone script fallback
    def recency_hpo_decay_from_config(*_args, **_kwargs):
        return None, None


DEFAULT_EXECUTION_COST_KEYS = {
    "execution_cost_bps",
    "fee_bps",
    "spread_bps",
    "spread_margin_bps",
    "delay_slippage_baseline_bps",
    "execution_margin_bps",
    "stop_fill_gap_bps",
}


@dataclass
class LabelGeometryParams:
    enabled: bool = False
    tp_vol_mult: float = 1.0
    sl_as_tp_pct: float = 0.75
    label_horizon_bars: float = 5.0
    timeout_value: float = 0.35
    trailing_activation_vol_mult: float = 1.25
    trailing_giveback_pct: float = 0.50
    min_executable_net_bps: float = 20.0
    mae_failure_vol_mult: float = 1.0
    geometry_anchor_mix: float = 0.50


@dataclass
class LabelParams:
    label_modifier_strength: float = 0.0
    mfe_scale_bps: float = 100.0
    mae_penalty_scale: float = 1.0
    net_return_center_bps: float = 20.0
    net_return_temperature_bps: float = 75.0
    stop_penalty: float = 0.35
    path_quality_mix: float = 0.50
    max_stop_soft_label: float = 0.30
    max_bad_path_soft_label: float = 0.45


@dataclass
class WeightParams:
    base_weight_power: float = 1.0
    weight_modifier_strength: float = 0.0
    positive_mass_target: float = 0.45
    class_rebalance_strength: float = 0.50
    mfe_weight_power: float = 0.75
    mae_weight_power: float = 0.75
    net_ev_weight_power: float = 1.0
    hard_negative_weight: float = 1.25
    ambiguous_weight: float = 0.60
    recency_half_life_days: float = 300.0
    concurrency_penalty: float = 0.25
    concurrency_window_hours: float = 1.0
    robustness_strength: float = 0.0
    path_quality_strength: float = 0.0
    utility_tail_rank_strength: float = 0.0
    utility_tail_rank_power: float = 4.0
    utility_tail_rank_base: float = 0.50
    utility_tail_rank_scale: float = 4.0
    timestamp_balance_strength: float = 0.0


@dataclass
class GeneratorParams:
    lgbm_soft_label_costs: float | None = None
    lgbm_soft_label_min_opportunity_mult: float | None = None
    lgbm_soft_label_temperature: float | None = None
    net_executable_mae_lambda: float | None = None
    net_executable_center_vol: float | None = None
    net_executable_temperature_vol: float | None = None
    policy_net_label_center: float | None = None
    policy_net_label_temperature: float | None = None
    policy_net_label_min_std: float | None = None
    policy_net_label_min_finite_frac: float | None = None
    timeout_weight: float | None = None
    outcome_weight_clip_min: float | None = None
    outcome_weight_clip_max: float | None = None
    mfe_mae_w_min: float | None = None
    mfe_mae_tau: float | None = None
    mfe_mae_cost_floor: float | None = None
    meta_weight_sigmoid_alpha: float | None = None
    meta_mfe_mae_tau: float | None = None
    policy_label_sl_atr_mult: float | None = None
    policy_label_tp_sl_ratio: float | None = None
    policy_label_trailing_pct: float | None = None
    policy_label_max_hold_hours: float | None = None


GENERATOR_DEFAULTS: dict[str, float] = {
    "lgbm_soft_label_costs": 0.0,
    "lgbm_soft_label_min_opportunity_mult": float(os.getenv("EPM_LGBM_SOFT_LABEL_MIN_OPPORTUNITY_MULT", "0.25")),
    "lgbm_soft_label_temperature": float(os.getenv("EPM_LGBM_SOFT_LABEL_TEMPERATURE", "0.20")),
    "net_executable_mae_lambda": 0.35,
    "net_executable_center_vol": 0.0,
    "net_executable_temperature_vol": 0.35,
    "policy_net_label_center": 0.0,
    "policy_net_label_temperature": 0.004,
    "policy_net_label_min_std": 1e-8,
    "policy_net_label_min_finite_frac": 0.98,
    "timeout_weight": 0.4,
    "outcome_weight_clip_min": 0.5,
    "outcome_weight_clip_max": 2.0,
    "mfe_mae_w_min": 0.5,
    "mfe_mae_tau": 1.0,
    "mfe_mae_cost_floor": 0.001,
    "meta_weight_sigmoid_alpha": 0.0,
    "meta_mfe_mae_tau": 1.0,
    "policy_label_sl_atr_mult": 1.2,
    "policy_label_tp_sl_ratio": 2.0,
    "policy_label_trailing_pct": 0.35,
    "policy_label_max_hold_hours": 24.0,
}


@dataclass
class DistillationParams:
    distillation_strength: float = 1.0
    distill_error_power: float = 1.0
    false_positive_focus: float = 1.0
    false_negative_focus: float = 0.50
    distill_age_impact: float = 0.50
    economic_error_mix: float = 0.0
    distill_net_loss_power: float = 1.0
    distill_stop_hit_focus: float = 0.0
    distill_missed_net_power: float = 1.0
    distill_rank_focus_threshold: float = 0.80
    distill_rank_focus_temperature: float = 0.08


def neutral_distillation_params() -> DistillationParams:
    return DistillationParams(
        distillation_strength=0.0,
        distill_error_power=0.0,
        false_positive_focus=0.0,
        false_negative_focus=0.0,
        distill_age_impact=0.0,
        economic_error_mix=0.0,
        distill_net_loss_power=0.0,
        distill_stop_hit_focus=0.0,
        distill_missed_net_power=0.0,
    )


@dataclass
class ObjectiveParams:
    base_k_multiplier: float = 2.0
    hit_fee_floor_bps: float = 0.0
    impact_scale_bps: float = 75.0
    portfolio_alignment_strength: float = 0.0
    edge_lcb_se_divisor: float = 1.64
    min_score_std_ratio: float = 0.75
    min_score_std_abs_base: float = 0.060
    min_score_std_abs_meta: float = 0.120
    min_rank_monotonicity: float = 0.62
    min_economic_rank_ic: float = 0.0
    economic_ic_floor: float = -0.02
    economic_ic_good: float = 0.12
    lgbm_j_floor: float = 0.55
    lgbm_j_good: float = 0.80
    max_top20_mean_net_bps_baseline_drawdown: float = 3.0
    max_top20_bps_weighted_hit_baseline_drawdown: float = 0.02
    max_top10_mean_net_bps_baseline_drawdown: float = 5.0
    max_top10_bps_weighted_hit_baseline_drawdown: float = 0.03
    max_score_std_baseline_drawdown_ratio: float = 0.25
    max_score_gap_baseline_drawdown_ratio: float = 0.45
    max_economic_weighted_ic_baseline_drawdown: float = 0.035
    max_top_weighted_ic_baseline_drawdown: float = 0.040
    min_effective_sample_frac_weight: float = 0.65
    min_weight_rank_corr_to_baseline: float = 0.40
    max_weight_final_delta_abs_mean: float = 0.65
    max_stop_hit_rate_at_20: float = 0.35
    max_avg_stop_loss_bps_at_20: float = 125.0
    min_mean_net_bps_at_20: float = 0.0
    min_window_mean_net_bps_at_20: float = 0.0
    max_window_stop_hit_rate_at_20: float = 0.50
    max_symbol_hhi_at_20: float = 0.25
    max_week_hhi_at_20: float = 0.35
    min_unique_symbols_at_20: float = 6.0
    min_label_final_changed_frac: float = 1e-4
    min_label_final_delta_abs_mean: float = 1e-7
    min_weight_final_changed_frac: float = 1e-4
    min_weight_final_delta_abs_mean: float = 1e-7


@dataclass
class LabelWeightRecipe:
    version: int = 1
    name: str = "default"
    stage: str = "all"
    geometry: LabelGeometryParams = field(default_factory=LabelGeometryParams)
    label: LabelParams = field(default_factory=LabelParams)
    weight: WeightParams = field(default_factory=WeightParams)
    generator: GeneratorParams = field(default_factory=GeneratorParams)
    distillation: DistillationParams = field(default_factory=DistillationParams)
    objective: ObjectiveParams = field(default_factory=ObjectiveParams)
    execution_costs: dict[str, float] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "LabelWeightRecipe":
        def _filtered(model: type[Any], key: str) -> dict[str, Any]:
            allowed = {f.name for f in fields(model)}
            return {k: v for k, v in dict(raw.get(key, {})).items() if k in allowed}

        return cls(
            version=int(raw.get("version", 1)),
            name=str(raw.get("name", "recipe")),
            stage=str(raw.get("stage", "all")),
            geometry=LabelGeometryParams(**_filtered(LabelGeometryParams, "geometry")),
            label=LabelParams(**_filtered(LabelParams, "label")),
            weight=WeightParams(**_filtered(WeightParams, "weight")),
            generator=GeneratorParams(**_filtered(GeneratorParams, "generator")),
            distillation=DistillationParams(**_filtered(DistillationParams, "distillation")),
            objective=ObjectiveParams(**_filtered(ObjectiveParams, "objective")),
            execution_costs={
                str(k): float(v)
                for k, v in dict(raw.get("execution_costs", {})).items()
                if str(k) in DEFAULT_EXECUTION_COST_KEYS
            },
            provenance=dict(raw.get("provenance", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_RECIPE_CACHE: dict[str, LabelWeightRecipe | None] = {}
DISABLED_RECIPE_KEY = "__label_weight_disabled__"
DISABLED_RECIPE_STAGE = "__label_weight_noop__"
HARDCODED_DEFAULT_RECIPE_KEY = "__label_weight_hardcoded_default__"
DEFAULT_BEST_RECIPE_PATH = Path("reports_perp/label_weight_optuna/best_recipe.json")
DEFAULT_STUDY_PATIENCE_TRIALS = 40
DEFAULT_LGBM_N_ESTIMATORS_CAP = 200
DEFAULT_LGBM_HPO_TRIALS = 80
DEFAULT_LGBM_EARLY_STOPPING_ROUNDS = 40
DEFAULT_PROMOTION_MARGIN = 0.01
LAYER_ALL = "all"
LAYER_AUTO = "auto"
GEOMETRY_LAYERS = ("barrier_feasibility", "path_quality")
LABEL_LAYERS = ("net_edge_mapping", "risk_path_caps")
WEIGHT_LAYERS = (
    "class_balance",
    "economic_emphasis",
    "robustness_diversification",
    "portfolio_alignment",
)
DISTILLATION_LAYERS = ("error_refocus", "rank_tail_refocus")
OPTUNA_LAYER_CHOICES = (
    LAYER_ALL,
    LAYER_AUTO,
    *GEOMETRY_LAYERS,
    *LABEL_LAYERS,
    *WEIGHT_LAYERS,
    *DISTILLATION_LAYERS,
)


def _scope_from_stage(stage: str | None) -> str | None:
    stage_l = str(stage or "").strip().lower()
    if stage_l in {"train_base", "base"}:
        return "base"
    if stage_l in {"train_meta", "meta"}:
        return "meta"
    return None


def _scope_key_prefix(scope: str | None) -> str:
    scope_l = str(scope or "").strip().lower()
    return scope_l if scope_l in {"base", "meta"} else ""


def recipe_path_from_env_or_cfg(cfg: dict[str, Any] | None = None, *, scope: str | None = None) -> str:
    cfg_local = cfg if isinstance(cfg, dict) else {}
    scope_prefix = _scope_key_prefix(scope)
    if _label_weight_disabled(cfg_local, scope=scope_prefix or None):
        return DISABLED_RECIPE_KEY
    explicit = ""
    if scope_prefix:
        explicit = os.getenv(
            f"EPM_LABEL_WEIGHT_{scope_prefix.upper()}_RECIPE",
            cfg_local.get(
                f"label_weight_{scope_prefix}_recipe",
                cfg_local.get(f"label_weight_{scope_prefix}_recipe_path", ""),
            ),
        )
        if explicit:
            return str(explicit).strip()
    explicit = os.getenv(
        "EPM_LABEL_WEIGHT_RECIPE",
        cfg_local.get("label_weight_recipe", cfg_local.get("label_weight_recipe_path", "")),
    )
    if explicit:
        return str(explicit).strip()
    use_best_default = _truthy_env_or_cfg("EPM_LABEL_WEIGHT_USE_BEST_DEFAULT", cfg_local, default=True)
    bypass_best_default = _truthy_env_or_cfg("EPM_LABEL_WEIGHT_BYPASS_BEST_DEFAULT", cfg_local, default=False)
    if use_best_default and not bypass_best_default:
        scoped_best = ""
        if scope_prefix:
            scoped_best = os.getenv(
                f"EPM_LABEL_WEIGHT_{scope_prefix.upper()}_BEST_RECIPE",
                cfg_local.get(f"label_weight_{scope_prefix}_best_recipe_path", ""),
            )
        if scoped_best:
            best_path = Path(str(scoped_best)).expanduser()
            if best_path.exists():
                return str(best_path)
        best_path = Path(
            str(
                os.getenv(
                    "EPM_LABEL_WEIGHT_BEST_RECIPE",
                    cfg_local.get("label_weight_best_recipe_path", DEFAULT_BEST_RECIPE_PATH),
                )
            )
        ).expanduser()
        if best_path.exists():
            return str(best_path)
    if bypass_best_default or not use_best_default:
        return HARDCODED_DEFAULT_RECIPE_KEY
    return ""


def _truthy_env_or_cfg(name: str, cfg: dict[str, Any], *, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        key = name.lower()
        prefix = "epm_"
        cfg_key = key[len(prefix) :] if key.startswith(prefix) else key
        raw = cfg.get(cfg_key, default)
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


def _label_weight_disabled(cfg: dict[str, Any], *, scope: str | None = None) -> bool:
    scope_prefix = _scope_key_prefix(scope)
    raw = None
    if scope_prefix:
        raw = os.getenv(f"EPM_LABEL_WEIGHT_{scope_prefix.upper()}_DISABLE")
        if raw is None and f"label_weight_{scope_prefix}_disable" in cfg:
            raw = cfg.get(f"label_weight_{scope_prefix}_disable")
        if raw is None and f"label_weight_{scope_prefix}_enabled" in cfg:
            return str(cfg.get(f"label_weight_{scope_prefix}_enabled")).strip().lower() in {"0", "false", "no", "n", "off"}
    if raw is None:
        raw = os.getenv("EPM_LABEL_WEIGHT_DISABLE")
    if raw is not None:
        return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}
    if "label_weight_disable" in cfg:
        return str(cfg.get("label_weight_disable")).strip().lower() in {"1", "true", "yes", "y", "on"}
    if "label_weight_enabled" in cfg:
        return str(cfg.get("label_weight_enabled")).strip().lower() in {"0", "false", "no", "n", "off"}
    return False


def _require_equal_lengths(context: str, **arrays: Any) -> int:
    lengths = {name: len(value) for name, value in arrays.items() if value is not None}
    unique = sorted(set(lengths.values()))
    if len(unique) > 1:
        detail = ", ".join(f"{name}={length}" for name, length in sorted(lengths.items()))
        raise ValueError(f"{context} length mismatch: {detail}")
    return unique[0] if unique else 0


def _fit_mask_from_indices(n: int, fit_indices: Any = None, fit_mask: Any = None) -> np.ndarray:
    if fit_mask is not None:
        mask = np.asarray(fit_mask, dtype=bool).reshape(-1)
        if len(mask) != n:
            raise ValueError(f"fit_mask length {len(mask)} != row length {n}")
        if not bool(mask.any()):
            raise ValueError("fit_mask must select at least one row")
        return mask
    if fit_indices is not None:
        idx = np.asarray(fit_indices, dtype=np.int64).reshape(-1)
        if idx.size == 0:
            raise ValueError("fit_indices must select at least one row")
        if np.any((idx < 0) | (idx >= n)):
            raise ValueError(f"fit_indices out of bounds for row length {n}")
        mask = np.zeros(n, dtype=bool)
        mask[idx] = True
        return mask
    return np.ones(n, dtype=bool)


def _normalize_weights(w: np.ndarray, *, fit_mask: np.ndarray | None = None) -> np.ndarray:
    arr = np.nan_to_num(np.asarray(w, dtype=np.float64), nan=1.0, posinf=1.0, neginf=1.0)
    arr = np.clip(arr, 1e-6, None)
    if fit_mask is not None:
        mask = np.asarray(fit_mask, dtype=bool)
        if len(mask) != len(arr):
            raise ValueError(f"fit_mask length {len(mask)} != weight length {len(arr)}")
        ref = arr[mask]
    else:
        ref = arr
    mean = float(np.mean(ref)) if len(ref) else 1.0
    return (arr / max(mean, 1e-12)).astype(np.float32)


def _normalize_weights_to_reference(
    w: np.ndarray,
    *,
    reference: np.ndarray,
    fit_mask: np.ndarray,
) -> np.ndarray:
    arr = np.nan_to_num(np.asarray(w, dtype=np.float64), nan=1.0, posinf=1.0, neginf=1.0)
    arr = np.clip(arr, 1e-6, None)
    ref = np.nan_to_num(np.asarray(reference, dtype=np.float64), nan=1.0, posinf=1.0, neginf=1.0)
    ref = np.clip(ref, 1e-6, None)
    mask = np.asarray(fit_mask, dtype=bool)
    arr_mean = float(np.mean(arr[mask])) if np.any(mask) else float(np.mean(arr))
    ref_mean = float(np.mean(ref[mask])) if np.any(mask) else float(np.mean(ref))
    return (arr / max(arr_mean, 1e-12) * max(ref_mean, 1e-12)).astype(np.float32)


def _sigmoid(x: np.ndarray | float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


POLICY_NET_SOFT_LABEL_MODES = {
    "s10",
    "policy_net",
    "policy_net_soft_label",
    "replayed_policy_net",
    "u_policy_net",
    "vanilla_independent_net",
    "vanilla_independent_net_soft_label",
}

POLICY_NET_PATH_BLEND_LABEL_MODES = {
    "s14",
    "policy_net_path_blend",
    "s14_policy_net_path_blend",
    "policy_net_path_blend_soft_label",
}

POLICY_NET_EXEC_GUARD_LABEL_MODES = {
    "s34",
    "s34_exec_guard_broad_policy",
    "exec_guard_broad_policy",
}

POLICY_NET_TIMEOUT_CAPPED_LABEL_MODES = {
    "s53": ("barrier", "path_blend", "timeout_barrier_cap_path_blend"),
    "s53_timeout_barrier_cap_path_blend": ("barrier", "path_blend", "timeout_barrier_cap_path_blend"),
    "timeout_barrier_cap_path_blend": ("barrier", "path_blend", "timeout_barrier_cap_path_blend"),
    "s55": ("barrier", "exec_guard", "timeout_barrier_cap_exec_guard"),
    "s55_timeout_barrier_cap_exec_guard": ("barrier", "exec_guard", "timeout_barrier_cap_exec_guard"),
    "timeout_barrier_cap_exec_guard": ("barrier", "exec_guard", "timeout_barrier_cap_exec_guard"),
    "s57": ("tpnet", "path_blend", "timeout_tpnet_cap_path_blend"),
    "s57_timeout_tpnet_cap_path_blend": ("tpnet", "path_blend", "timeout_tpnet_cap_path_blend"),
    "timeout_tpnet_cap_path_blend": ("tpnet", "path_blend", "timeout_tpnet_cap_path_blend"),
}


def policy_net_soft_label_mode_requested(cfg: dict[str, Any] | None = None) -> bool:
    cfg_local = cfg if isinstance(cfg, dict) else {}
    mode = str(
        os.getenv("EPM_LABEL_ABLATION_MODE", cfg_local.get("label_ablation_mode", ""))
        or ""
    ).strip().lower()
    if mode in POLICY_NET_SOFT_LABEL_MODES:
        return True
    flag = os.getenv(
        "EPM_POLICY_NET_SOFT_LABEL",
        str(cfg_local.get("policy_net_soft_label_enabled", "") or ""),
    ).strip().lower()
    return flag in {"1", "true", "yes", "y", "on"}


def policy_net_path_blend_label_mode_requested(cfg: dict[str, Any] | None = None) -> bool:
    cfg_local = cfg if isinstance(cfg, dict) else {}
    mode = str(
        os.getenv("EPM_LABEL_ABLATION_MODE", cfg_local.get("label_ablation_mode", ""))
        or ""
    ).strip().lower()
    return mode in POLICY_NET_PATH_BLEND_LABEL_MODES


def policy_net_exec_guard_label_mode_requested(cfg: dict[str, Any] | None = None) -> bool:
    cfg_local = cfg if isinstance(cfg, dict) else {}
    mode = str(
        os.getenv("EPM_LABEL_ABLATION_MODE", cfg_local.get("label_ablation_mode", ""))
        or ""
    ).strip().lower()
    return mode in POLICY_NET_EXEC_GUARD_LABEL_MODES


def policy_net_timeout_capped_label_mode(
    cfg: dict[str, Any] | None = None,
) -> tuple[str, str, str] | None:
    cfg_local = cfg if isinstance(cfg, dict) else {}
    mode = str(
        os.getenv("EPM_LABEL_ABLATION_MODE", cfg_local.get("label_ablation_mode", ""))
        or ""
    ).strip().lower()
    return POLICY_NET_TIMEOUT_CAPPED_LABEL_MODES.get(mode)


def build_policy_net_soft_label_from_frame(
    df: pd.DataFrame,
    y_hard: np.ndarray,
    *,
    cfg: dict[str, Any] | None = None,
    label: str = "native",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Map replayed policy net utility to a soft label.

    This is the execution-aligned S10-style target: the continuous label is
    derived from the net utility that the current policy replay assigns to the
    row, not from a generic MFE/MAE proxy.
    """

    cfg_local = cfg if isinstance(cfg, dict) else {}
    y_ref = np.asarray(y_hard, dtype=np.float32).reshape(-1)
    fallback = np.clip(np.nan_to_num(y_ref, nan=0.0), 0.0, 1.0).astype(np.float32)
    n = min(len(df), len(y_ref))
    if n <= 0:
        return fallback[:0], {
            "enabled": False,
            "reason": "empty",
            "target_mode": "policy_net_replay",
        }

    source_col = next((c for c in ("__u_policy_net__", "u_policy_net") if c in df.columns), None)
    if source_col is None:
        raise RuntimeError(
            f"{label}: policy-net soft-label mode requires '__u_policy_net__' or "
            "'u_policy_net'. Refusing to fall back to the old label."
        )

    raw = pd.to_numeric(df[source_col].iloc[:n], errors="coerce").to_numpy(dtype=np.float64)
    finite = np.isfinite(raw)
    finite_frac = float(np.mean(finite)) if len(finite) else 0.0
    min_finite_frac = float(
        os.getenv(
            "EPM_POLICY_NET_LABEL_MIN_FINITE_FRAC",
            cfg_local.get("policy_net_label_min_finite_frac", 0.98),
        )
    )
    if finite_frac < min_finite_frac:
        raise RuntimeError(
            f"{label}: policy-net soft-label source '{source_col}' has only "
            f"{finite_frac:.3f} finite values; required >= {min_finite_frac:.3f}."
        )

    finite_raw = raw[finite]
    raw_std = float(np.std(finite_raw)) if len(finite_raw) else 0.0
    min_std = float(
        os.getenv(
            "EPM_POLICY_NET_LABEL_MIN_STD",
            cfg_local.get("policy_net_label_min_std", 1e-8),
        )
    )
    if raw_std <= min_std:
        raise RuntimeError(
            f"{label}: policy-net soft-label source '{source_col}' is effectively "
            f"constant (std={raw_std:.6g}). Refusing to train on an uninformative "
            "execution target."
        )

    center = float(
        os.getenv(
            "EPM_POLICY_NET_LABEL_CENTER",
            cfg_local.get("policy_net_label_center", 0.0),
        )
    )
    temperature = max(
        1e-12,
        float(
            os.getenv(
                "EPM_POLICY_NET_LABEL_TEMPERATURE",
                cfg_local.get("policy_net_label_temperature", 0.004),
            )
        ),
    )
    cleaned = np.where(finite, raw, center)
    soft = _sigmoid((cleaned - center) / temperature)
    soft = np.clip(np.nan_to_num(soft, nan=0.5), 0.0, 1.0).astype(np.float32)
    if n < len(fallback):
        merged = fallback.copy()
        merged[:n] = soft
        soft = merged

    stats = {
        "enabled": True,
        "label": str(label),
        "target_mode": "policy_net_replay",
        "source_column": str(source_col),
        "n": int(len(soft)),
        "finite_frac": float(finite_frac),
        "raw_mean": float(np.mean(finite_raw)) if len(finite_raw) else float("nan"),
        "raw_std": float(raw_std),
        "raw_p10": float(np.percentile(finite_raw, 10)) if len(finite_raw) else float("nan"),
        "raw_p50": float(np.percentile(finite_raw, 50)) if len(finite_raw) else float("nan"),
        "raw_p90": float(np.percentile(finite_raw, 90)) if len(finite_raw) else float("nan"),
        "soft_mean": float(np.mean(soft)) if len(soft) else float("nan"),
        "soft_std": float(np.std(soft)) if len(soft) else float("nan"),
        "hard_mean": float(np.mean(fallback)) if len(fallback) else float("nan"),
        "center": float(center),
        "temperature": float(temperature),
        "min_std": float(min_std),
        "min_finite_frac": float(min_finite_frac),
    }
    return soft.astype(np.float32, copy=False), stats


def build_policy_net_path_blend_label_from_frame(
    df: pd.DataFrame,
    y_hard: np.ndarray,
    *,
    cfg: dict[str, Any] | None = None,
    label: str = "native",
) -> tuple[np.ndarray, dict[str, Any]]:
    """S14-style replay-utility/path-quality blend used by proxy screens."""

    cfg_local = cfg if isinstance(cfg, dict) else {}
    y_ref = np.asarray(y_hard, dtype=np.float32).reshape(-1)
    fallback = np.clip(np.nan_to_num(y_ref, nan=0.0), 0.0, 1.0).astype(np.float32)
    n = min(len(df), len(y_ref))
    if n <= 0:
        return fallback[:0], {
            "enabled": False,
            "reason": "empty",
            "target_mode": "policy_net_path_blend",
        }

    required = {"__mfe_ret__", "__mae_ret__", "__barrier_pct__"}
    missing = sorted(c for c in required if c not in df.columns)
    source_col = next((c for c in ("__u_policy_net__", "u_policy_net") if c in df.columns), None)
    if source_col is None:
        missing.append("__u_policy_net__")
    if missing:
        raise RuntimeError(
            f"{label}: S14 policy-net path-blend mode requires columns {sorted(set(missing))}."
        )

    df_local = df.reset_index(drop=True)
    raw_u = pd.to_numeric(df_local[source_col].iloc[:n], errors="coerce").to_numpy(dtype=np.float64)
    finite = np.isfinite(raw_u)
    finite_frac = float(np.mean(finite)) if len(finite) else 0.0
    min_finite_frac = float(
        os.getenv(
            "EPM_POLICY_NET_LABEL_MIN_FINITE_FRAC",
            cfg_local.get("policy_net_label_min_finite_frac", 0.98),
        )
    )
    if finite_frac < min_finite_frac:
        raise RuntimeError(
            f"{label}: S14 source '{source_col}' has only {finite_frac:.3f} finite values; "
            f"required >= {min_finite_frac:.3f}."
        )
    finite_raw = raw_u[finite]
    raw_std = float(np.std(finite_raw)) if len(finite_raw) else 0.0
    min_std = float(
        os.getenv(
            "EPM_POLICY_NET_LABEL_MIN_STD",
            cfg_local.get("policy_net_label_min_std", 1e-8),
        )
    )
    if raw_std <= min_std:
        raise RuntimeError(
            f"{label}: S14 source '{source_col}' is effectively constant (std={raw_std:.6g})."
        )

    center = float(
        os.getenv(
            "EPM_POLICY_NET_LABEL_CENTER",
            cfg_local.get("policy_net_label_center", 0.0),
        )
    )
    policy_temperature = max(
        1e-12,
        float(
            os.getenv(
                "EPM_POLICY_NET_LABEL_TEMPERATURE",
                cfg_local.get("policy_net_label_temperature", 0.012),
            )
        ),
    )
    policy_soft = _sigmoid((np.where(finite, raw_u, center) - center) / policy_temperature)

    mfe, mae_abs, barrier, _timeout = _path_arrays(df_local, n)
    mfe_norm = np.clip(mfe / np.maximum(barrier, 1e-8), 0.0, 50.0)
    mae_norm = np.clip(mae_abs / np.maximum(barrier, 1e-8), 0.0, 50.0)
    if "__bars_to_mfe__" in df_local.columns:
        bars_to_mfe = pd.to_numeric(
            df_local["__bars_to_mfe__"].iloc[:n],
            errors="coerce",
        ).to_numpy(dtype=np.float64)
    elif "__bars_policy__" in df_local.columns:
        bars_to_mfe = pd.to_numeric(
            df_local["__bars_policy__"].iloc[:n],
            errors="coerce",
        ).to_numpy(dtype=np.float64)
    else:
        bars_to_mfe = np.full(n, 24.0, dtype=np.float64)
    bars_to_mfe = np.clip(np.nan_to_num(bars_to_mfe, nan=24.0, posinf=24.0, neginf=24.0), 0.0, None)

    if "__y_ret__" in df_local.columns:
        y_ret = pd.to_numeric(df_local["__y_ret__"].iloc[:n], errors="coerce").to_numpy(dtype=np.float64)
    elif "__r_policy_net__" in df_local.columns:
        y_ret = pd.to_numeric(df_local["__r_policy_net__"].iloc[:n], errors="coerce").to_numpy(dtype=np.float64)
    else:
        y_ret = raw_u
    round_trip_cost = max(
        0.0,
        float(
            os.getenv(
                "EPM_POLICY_NET_PATH_BLEND_COST",
                cfg_local.get("policy_net_path_blend_cost", 0.0030),
            )
        ),
    )
    ret_net = np.nan_to_num(y_ret, nan=0.0) - round_trip_cost
    downside_raw = (
        0.90 * mfe_norm
        - 1.85 * mae_norm
        + ret_net / np.maximum(barrier, 1e-8)
        - 0.15 * np.log1p(bars_to_mfe)
    )
    path_temperature = max(
        1e-12,
        float(
            os.getenv(
                "EPM_POLICY_NET_PATH_BLEND_PATH_TEMPERATURE",
                cfg_local.get("policy_net_path_blend_path_temperature", 1.25),
            )
        ),
    )
    path_center = float(
        os.getenv(
            "EPM_POLICY_NET_PATH_BLEND_PATH_CENTER",
            cfg_local.get("policy_net_path_blend_path_center", 0.10),
        )
    )
    asymmetric = _sigmoid((downside_raw - path_center) / path_temperature)
    if "__y_outcome__" in df_local.columns:
        outcome = pd.to_numeric(df_local["__y_outcome__"].iloc[:n], errors="coerce").to_numpy(dtype=np.float64)
        bad_path = (mae_norm >= 1.0) | (np.isfinite(outcome) & (outcome == 0.0))
    else:
        bad_path = mae_norm >= 1.0
    bad_cap = float(
        os.getenv(
            "EPM_POLICY_NET_PATH_BLEND_BAD_CAP",
            cfg_local.get("policy_net_path_blend_bad_cap", 0.25),
        )
    )
    asymmetric = np.where(bad_path, np.minimum(asymmetric, bad_cap), asymmetric)
    policy_weight = float(
        os.getenv(
            "EPM_POLICY_NET_PATH_BLEND_POLICY_WEIGHT",
            cfg_local.get("policy_net_path_blend_policy_weight", 0.50),
        )
    )
    policy_weight = float(np.clip(policy_weight, 0.0, 1.0))
    soft = (policy_weight * policy_soft) + ((1.0 - policy_weight) * asymmetric)
    soft = np.clip(np.nan_to_num(soft, nan=0.5), 0.0, 1.0).astype(np.float32)
    if n < len(fallback):
        merged = fallback.copy()
        merged[:n] = soft
        soft = merged

    stats = {
        "enabled": True,
        "label": str(label),
        "target_mode": "policy_net_path_blend",
        "source_column": str(source_col),
        "n": int(len(soft)),
        "finite_frac": float(finite_frac),
        "raw_mean": float(np.mean(finite_raw)) if len(finite_raw) else float("nan"),
        "raw_std": float(raw_std),
        "policy_center": float(center),
        "policy_temperature": float(policy_temperature),
        "policy_weight": float(policy_weight),
        "path_center": float(path_center),
        "path_temperature": float(path_temperature),
        "bad_cap": float(bad_cap),
        "bad_path_rate": float(np.mean(bad_path)) if len(bad_path) else float("nan"),
        "soft_mean": float(np.mean(soft)) if len(soft) else float("nan"),
        "soft_std": float(np.std(soft)) if len(soft) else float("nan"),
        "hard_mean": float(np.mean(fallback)) if len(fallback) else float("nan"),
        "min_std": float(min_std),
        "min_finite_frac": float(min_finite_frac),
    }
    return soft.astype(np.float32, copy=False), stats


def build_policy_net_capped_execution_label_from_frame(
    df: pd.DataFrame,
    y_hard: np.ndarray,
    *,
    cfg: dict[str, Any] | None = None,
    label: str = "native",
    cap_kind: str = "none",
    family: str = "exec_guard",
    target_mode: str = "exec_guard_broad_policy",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build S34/S53/S55/S57-style execution-aligned soft labels.

    These modes mirror the proxy candidates promoted by
    ``run_label_economic_proxy_ablation.py``. The timeout-capped variants bound
    positive timeout utility to executable TP/barrier geometry before deriving
    the soft label, so the learner is not rewarded for unbounded late marks.
    """

    cfg_local = cfg if isinstance(cfg, dict) else {}
    y_ref = np.asarray(y_hard, dtype=np.float32).reshape(-1)
    fallback = np.clip(np.nan_to_num(y_ref, nan=0.0), 0.0, 1.0).astype(np.float32)
    n = min(len(df), len(y_ref))
    if n <= 0:
        return fallback[:0], {
            "enabled": False,
            "reason": "empty",
            "target_mode": target_mode,
        }

    required = {"__mfe_ret__", "__mae_ret__", "__barrier_pct__"}
    missing = sorted(c for c in required if c not in df.columns)
    source_col = next((c for c in ("__u_policy_net__", "u_policy_net") if c in df.columns), None)
    if source_col is None:
        missing.append("__u_policy_net__")
    if missing:
        raise RuntimeError(
            f"{label}: {target_mode} mode requires columns {sorted(set(missing))}."
        )

    df_local = df.reset_index(drop=True)

    def numeric_col(name: str, default: np.ndarray | float) -> np.ndarray:
        if name in df_local.columns:
            values = pd.to_numeric(df_local[name].iloc[:n], errors="coerce").to_numpy(dtype=np.float64)
        else:
            values = np.full(n, np.nan, dtype=np.float64)
        if np.isscalar(default):
            default_arr = np.full(n, float(default), dtype=np.float64)
        else:
            default_arr = np.asarray(default, dtype=np.float64).reshape(-1)[:n]
            if len(default_arr) < n:
                default_arr = np.pad(default_arr, (0, n - len(default_arr)), constant_values=np.nan)
        return np.where(np.isfinite(values), values, default_arr)

    raw_u = pd.to_numeric(df_local[source_col].iloc[:n], errors="coerce").to_numpy(dtype=np.float64)
    finite = np.isfinite(raw_u)
    finite_frac = float(np.mean(finite)) if len(finite) else 0.0
    min_finite_frac = float(
        os.getenv(
            "EPM_POLICY_NET_LABEL_MIN_FINITE_FRAC",
            cfg_local.get("policy_net_label_min_finite_frac", 0.98),
        )
    )
    if finite_frac < min_finite_frac:
        raise RuntimeError(
            f"{label}: {target_mode} source '{source_col}' has only "
            f"{finite_frac:.3f} finite values; required >= {min_finite_frac:.3f}."
        )
    finite_raw = raw_u[finite]
    raw_std = float(np.std(finite_raw)) if len(finite_raw) else 0.0
    min_std = float(
        os.getenv(
            "EPM_POLICY_NET_LABEL_MIN_STD",
            cfg_local.get("policy_net_label_min_std", 1e-8),
        )
    )
    if raw_std <= min_std:
        raise RuntimeError(
            f"{label}: {target_mode} source '{source_col}' is effectively constant "
            f"(std={raw_std:.6g})."
        )

    center = float(
        os.getenv(
            "EPM_POLICY_NET_LABEL_CENTER",
            cfg_local.get("policy_net_label_center", 0.0),
        )
    )
    policy_temperature = max(
        1e-12,
        float(
            os.getenv(
                "EPM_POLICY_NET_LABEL_TEMPERATURE",
                cfg_local.get("policy_net_label_temperature", 0.012),
            )
        ),
    )
    u = np.where(finite, raw_u, center)
    policy_soft = np.clip(_sigmoid((u - center) / policy_temperature), 0.0, 1.0)

    mfe, mae_abs, barrier, timeout_bool = _path_arrays(df_local, n)
    mfe_norm = np.clip(mfe / np.maximum(barrier, 1e-8), 0.0, 50.0)
    mae_norm = np.clip(mae_abs / np.maximum(barrier, 1e-8), 0.0, 50.0)
    bars_to_mfe = numeric_col(
        "__bars_to_mfe__",
        numeric_col("__bars_policy__", 24.0),
    )
    bars_to_mfe = np.clip(np.nan_to_num(bars_to_mfe, nan=24.0, posinf=24.0, neginf=24.0), 0.0, None)
    timeout_float = timeout_bool.astype(np.float64)

    if "__y_ret__" in df_local.columns:
        y_ret = pd.to_numeric(df_local["__y_ret__"].iloc[:n], errors="coerce").to_numpy(dtype=np.float64)
    elif "__r_policy_net__" in df_local.columns:
        y_ret = pd.to_numeric(df_local["__r_policy_net__"].iloc[:n], errors="coerce").to_numpy(dtype=np.float64)
    else:
        y_ret = u
    round_trip_cost = max(
        0.0,
        float(
            os.getenv(
                "EPM_POLICY_NET_CAPPED_LABEL_COST",
                cfg_local.get("policy_net_capped_label_cost", 0.0030),
            )
        ),
    )
    ret_net = np.nan_to_num(y_ret, nan=0.0) - round_trip_cost

    downside_raw = (
        0.90 * mfe_norm
        - 1.85 * mae_norm
        + ret_net / np.maximum(barrier, 1e-8)
        - 0.15 * np.log1p(bars_to_mfe)
    )
    path_temperature = max(
        1e-12,
        float(
            os.getenv(
                "EPM_POLICY_NET_PATH_BLEND_PATH_TEMPERATURE",
                cfg_local.get("policy_net_path_blend_path_temperature", 1.25),
            )
        ),
    )
    path_center = float(
        os.getenv(
            "EPM_POLICY_NET_PATH_BLEND_PATH_CENTER",
            cfg_local.get("policy_net_path_blend_path_center", 0.10),
        )
    )
    path_component = _sigmoid((downside_raw - path_center) / path_temperature)
    if "__y_outcome__" in df_local.columns:
        outcome = pd.to_numeric(df_local["__y_outcome__"].iloc[:n], errors="coerce").to_numpy(dtype=np.float64)
        bad_path = (mae_norm >= 1.0) | (np.isfinite(outcome) & (outcome == 0.0))
    else:
        bad_path = mae_norm >= 1.0
    bad_cap = float(
        os.getenv(
            "EPM_POLICY_NET_PATH_BLEND_BAD_CAP",
            cfg_local.get("policy_net_path_blend_bad_cap", 0.25),
        )
    )
    path_component = np.where(bad_path, np.minimum(path_component, bad_cap), path_component)
    base_path_blend = np.clip(0.50 * policy_soft + 0.50 * path_component, 0.0, 1.0)

    tail_soft = np.clip(_sigmoid((u - 0.005) / 0.018), 0.0, 1.0)
    risk_adjusted_u = (
        u
        - 0.0012 * np.maximum(mae_norm - 1.50, 0.0)
        - 0.00030 * np.log1p(np.maximum(bars_to_mfe, 0.0))
        - 0.08 * np.maximum(barrier - 0.025, 0.0)
        - 0.0010 * timeout_float
    )
    tail_risk_soft = np.clip(_sigmoid((risk_adjusted_u - 0.003) / 0.018), 0.0, 1.0)

    fast_score = _sigmoid((14.0 - bars_to_mfe) / 6.0)
    risk_clean_score = _sigmoid((3.00 - mae_norm) / 1.00) * _sigmoid((0.060 - barrier) / 0.020)
    mfe_score = _sigmoid((mfe_norm - 1.20) / 0.75)
    path_fast = np.clip(0.35 * fast_score + 0.35 * risk_clean_score + 0.30 * mfe_score, 0.0, 1.0)

    exec_clean_score = (
        _sigmoid((0.85 - mae_norm) / 0.25)
        * _sigmoid((0.022 - barrier) / 0.004)
        * (0.40 + 0.60 * _sigmoid((8.0 - bars_to_mfe) / 3.0))
        * (0.35 + 0.65 * _sigmoid((mfe_norm - 1.15) / 0.45))
    )
    exec_clean_score = np.clip(exec_clean_score, 0.0, 1.0)

    cap_kind_l = str(cap_kind or "none").strip().lower()
    capped_u = u.copy()
    if cap_kind_l in {"barrier", "tpnet"}:
        effective_tp = np.abs(numeric_col("__first_touch_effective_tp_abs__", numeric_col("__tp__", barrier * 0.75)))
        effective_sl = np.abs(numeric_col("__first_touch_effective_sl_abs__", numeric_col("__sl__", barrier * 1.50)))
        barrier_cap = np.maximum(effective_tp, barrier)
        if cap_kind_l == "tpnet":
            positive_cap = np.clip(effective_tp - round_trip_cost, 0.00075, None)
        else:
            positive_cap = np.clip(barrier_cap, 0.00075, None)
        loss_floor = -np.clip(effective_sl + round_trip_cost, 0.00075, None)
        timeout_positive = timeout_bool & (u > 0.0)
        capped_u = np.where(timeout_positive, np.minimum(capped_u, positive_cap), capped_u)
        capped_u = np.where(timeout_bool, np.maximum(capped_u, loss_floor), capped_u)

    capped_policy_soft = np.clip(_sigmoid(capped_u / 0.012), 0.0, 1.0)
    capped_tail_soft = np.clip(_sigmoid((capped_u - 0.005) / 0.018), 0.0, 1.0)
    capped_risk_u = (
        capped_u
        - 0.0012 * np.maximum(mae_norm - 1.50, 0.0)
        - 0.00030 * np.log1p(np.maximum(bars_to_mfe, 0.0))
        - 0.08 * np.maximum(barrier - 0.025, 0.0)
        - 0.0010 * timeout_float
    )
    capped_tail_risk_soft = np.clip(_sigmoid((capped_risk_u - 0.003) / 0.018), 0.0, 1.0)
    capped_path_blend = np.clip(0.50 * capped_policy_soft + 0.50 * path_component, 0.0, 1.0)
    capped_broad_soft = np.clip(
        0.35 * capped_path_blend
        + 0.30 * capped_tail_risk_soft
        + 0.20 * capped_tail_soft
        + 0.15 * path_fast,
        0.0,
        1.0,
    )

    family_l = str(family or "exec_guard").strip().lower()
    if cap_kind_l == "none":
        broad_soft = np.clip(
            0.35 * base_path_blend + 0.30 * tail_risk_soft + 0.20 * tail_soft + 0.15 * path_fast,
            0.0,
            1.0,
        )
        soft = broad_soft * (0.05 + 0.95 * exec_clean_score)
        hard = (
            (u > 0.001)
            & (barrier <= 0.022)
            & (mae_norm <= 1.05)
            & (mfe_norm >= 1.00)
            & (bars_to_mfe <= 12.0)
            & (path_fast >= 0.40)
        )
    elif family_l == "path_blend":
        soft = capped_path_blend
        hard = (capped_u > 0.0) & (path_component >= 0.45)
    elif family_l == "exec_guard":
        soft = capped_broad_soft * (0.05 + 0.95 * exec_clean_score)
        hard = (
            (capped_u > (0.001 if cap_kind_l == "barrier" else 0.0))
            & (barrier <= 0.022)
            & (mae_norm <= 1.05)
            & (mfe_norm >= 1.00)
            & (bars_to_mfe <= 12.0)
            & (path_fast >= 0.40)
        )
    else:
        raise RuntimeError(f"{label}: unknown capped execution label family {family!r}.")

    soft = np.clip(np.nan_to_num(soft, nan=0.5), 0.0, 1.0).astype(np.float32)
    if n < len(fallback):
        merged = fallback.copy()
        merged[:n] = soft
        soft = merged

    stats = {
        "enabled": True,
        "label": str(label),
        "target_mode": str(target_mode),
        "source_column": str(source_col),
        "n": int(len(soft)),
        "finite_frac": float(finite_frac),
        "raw_mean": float(np.mean(finite_raw)) if len(finite_raw) else float("nan"),
        "raw_std": float(raw_std),
        "policy_center": float(center),
        "policy_temperature": float(policy_temperature),
        "path_center": float(path_center),
        "path_temperature": float(path_temperature),
        "bad_cap": float(bad_cap),
        "cap_kind": str(cap_kind_l),
        "family": str(family_l),
        "round_trip_cost": float(round_trip_cost),
        "timeout_rate": float(np.mean(timeout_bool)) if len(timeout_bool) else float("nan"),
        "timeout_positive_rate": float(np.mean(timeout_bool & (u > 0.0))) if len(timeout_bool) else float("nan"),
        "soft_mean": float(np.mean(soft)) if len(soft) else float("nan"),
        "soft_std": float(np.std(soft)) if len(soft) else float("nan"),
        "hard_positive_rate_proxy": float(np.mean(hard)) if len(hard) else float("nan"),
        "path_component_mean": float(np.mean(path_component)) if len(path_component) else float("nan"),
        "exec_clean_mean": float(np.mean(exec_clean_score)) if len(exec_clean_score) else float("nan"),
        "min_std": float(min_std),
        "min_finite_frac": float(min_finite_frac),
    }
    if cap_kind_l in {"barrier", "tpnet"}:
        stats.update(
            {
                "timeout_cap_delta_mean": float(np.mean(capped_u - u)) if len(capped_u) else float("nan"),
                "timeout_cap_changed_rate": float(np.mean(np.abs(capped_u - u) > 1e-12)) if len(capped_u) else float("nan"),
            }
        )
    return soft.astype(np.float32, copy=False), stats


def _safe_logit(p: np.ndarray | float) -> np.ndarray:
    arr = np.clip(np.asarray(p, dtype=np.float64), 1e-5, 1.0 - 1e-5)
    return np.log(arr / (1.0 - arr))


def _robustness_uncertainty(
    *,
    mfe_bps: np.ndarray,
    mae_bps: np.ndarray,
    vol_bps: np.ndarray,
    net_edge_bps: np.ndarray,
) -> np.ndarray:
    vol_safe = np.maximum(np.asarray(vol_bps, dtype=np.float64), 1e-6)
    tp_margin = np.abs(np.asarray(mfe_bps, dtype=np.float64) - vol_safe) / vol_safe
    sl_margin = np.abs(np.asarray(mae_bps, dtype=np.float64) - vol_safe) / vol_safe
    near_barrier = np.exp(-np.minimum(tp_margin, sl_margin) / 0.15)

    perturb_grid = np.asarray([-60.0, -30.0, -15.0, 0.0, 15.0, 30.0, 60.0], dtype=np.float64)
    signs = (np.asarray(net_edge_bps, dtype=np.float64)[:, None] - perturb_grid[None, :]) > 0.0
    p_pos = np.mean(signs.astype(np.float64), axis=1)
    flip_risk = 1.0 - np.abs(2.0 * p_pos - 1.0)

    sigma = np.maximum(25.0, 0.50 * vol_safe)
    ci_risk = _sigmoid(-np.abs(np.asarray(net_edge_bps, dtype=np.float64)) / sigma)
    return np.clip(np.maximum.reduce([near_barrier, flip_risk, ci_risk]), 0.0, 1.0)


def _path_timing_penalty(df: pd.DataFrame, n: int, is_timeout: np.ndarray) -> np.ndarray:
    penalty = np.zeros(n, dtype=np.float64)
    if "__bars_to_mfe__" in df.columns and "__bars_to_mae__" in df.columns:
        bars_to_mfe = np.asarray(df["__bars_to_mfe__"].values[:n], dtype=np.float64)
        bars_to_mae = np.asarray(df["__bars_to_mae__"].values[:n], dtype=np.float64)
        finite_mfe = bars_to_mfe[np.isfinite(bars_to_mfe)]
        horizon = float(np.nanpercentile(finite_mfe, 90)) if finite_mfe.size else 1.0
        horizon = max(horizon, 1.0)
        adverse_first = np.isfinite(bars_to_mae) & np.isfinite(bars_to_mfe) & (bars_to_mae < bars_to_mfe)
        late_mfe = np.clip(np.nan_to_num(bars_to_mfe / horizon, nan=1.0, posinf=1.0), 0.0, 1.0)
        penalty += 0.55 * adverse_first.astype(np.float64)
        penalty += 0.30 * late_mfe
    penalty += 0.20 * np.asarray(is_timeout[:n], dtype=np.float64)
    return np.clip(penalty, 0.0, 1.0)


def _stop_path_masks(
    df: pd.DataFrame,
    n: int,
    *,
    y_geom: np.ndarray | None = None,
    is_timeout: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return actual-stop and broader adverse-path masks from label path columns."""

    df_local = df.reset_index(drop=True)
    actual_stop = np.zeros(n, dtype=bool)
    for col in ("__y_outcome__", "exit_code", "__exit_code__"):
        if col in df_local.columns:
            raw = pd.to_numeric(df_local[col].iloc[:n], errors="coerce").to_numpy(dtype=np.float64)
            actual_stop |= np.isfinite(raw) & (raw == 0.0)
            break
    if y_geom is not None:
        timeout = np.asarray(is_timeout[:n], dtype=bool) if is_timeout is not None else np.zeros(n, dtype=bool)
        actual_stop |= (np.asarray(y_geom[:n], dtype=np.float64) < 0.5) & (~timeout)

    adverse_first = np.zeros(n, dtype=bool)
    if "__bars_to_mfe__" in df_local.columns and "__bars_to_mae__" in df_local.columns:
        bars_to_mfe = pd.to_numeric(df_local["__bars_to_mfe__"].iloc[:n], errors="coerce").to_numpy(dtype=np.float64)
        bars_to_mae = pd.to_numeric(df_local["__bars_to_mae__"].iloc[:n], errors="coerce").to_numpy(dtype=np.float64)
        adverse_first = np.isfinite(bars_to_mae) & (
            ~np.isfinite(bars_to_mfe) | (bars_to_mae <= bars_to_mfe)
        )
    return actual_stop, actual_stop | adverse_first


def _quick_mfe_profit_score(
    df: pd.DataFrame,
    *,
    n: int,
    mfe: np.ndarray,
    vol: np.ndarray,
    fit_mask: np.ndarray | None = None,
) -> np.ndarray:
    if "__bars_to_mfe__" not in df.columns or n <= 0:
        return np.zeros(n, dtype=np.float64)
    bars = np.asarray(df["__bars_to_mfe__"].values[:n], dtype=np.float64)
    valid = np.isfinite(bars) & (bars >= 0.0) & np.isfinite(mfe[:n]) & (mfe[:n] > 0.0)
    if fit_mask is not None:
        ref_valid = valid & np.asarray(fit_mask, dtype=bool)
    else:
        ref_valid = valid
    ref_bars = bars[ref_valid]
    if ref_bars.size == 0:
        ref_bars = bars[valid]
    half_life = float(np.nanpercentile(ref_bars, 50)) if ref_bars.size else 1.0
    half_life = max(half_life, 1.0)
    quickness = np.exp(-np.maximum(np.nan_to_num(bars, nan=half_life, posinf=half_life), 0.0) / half_life)
    magnitude = np.tanh(np.maximum(np.nan_to_num(mfe[:n], nan=0.0), 0.0) / np.maximum(vol[:n], 1e-8))
    score = np.where(valid, quickness * magnitude, 0.0)
    return np.clip(np.nan_to_num(score, nan=0.0), 0.0, 1.0)


def _path_economics_state(
    df: pd.DataFrame,
    *,
    n: int,
    y_hard: np.ndarray,
    y_soft: np.ndarray,
    recipe: LabelWeightRecipe,
    cfg: dict[str, Any] | None,
    fit_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """Shared label/weight construction state from path geometry and economics."""

    df_local = df.reset_index(drop=True)
    p_label = recipe.label
    mfe, mae_abs, vol, is_timeout = _path_arrays(df_local, n)
    costs_bps = fixed_execution_cost_bps(cfg, recipe)
    mfe_bps = 10_000.0 * mfe[:n]
    mae_bps = 10_000.0 * mae_abs[:n]
    vol_bps = np.clip(10_000.0 * vol[:n], 1e-6, None)
    y_ref = np.asarray(y_hard, dtype=np.float64).reshape(-1)[:n]
    ys = np.clip(np.asarray(y_soft, dtype=np.float64).reshape(-1)[:n], 0.0, 1.0)

    y_geom, geom_anchor, geom_stats = _geometry_reference(
        df_local,
        n=n,
        y_hard=y_ref,
        mfe=mfe,
        mae_abs=mae_abs,
        vol=vol,
        recipe=recipe,
        costs_bps=costs_bps,
    )
    actual_stop, bad_path = _stop_path_masks(
        df_local,
        n,
        y_geom=y_geom,
        is_timeout=is_timeout,
    )

    geom = recipe.geometry
    if bool(geom.enabled):
        tp_ref_bps = np.maximum(
            float(geom.tp_vol_mult) * vol_bps,
            float(geom.min_executable_net_bps) + float(costs_bps),
        )
        sl_ref_bps = np.maximum(
            float(geom.sl_as_tp_pct) * tp_ref_bps,
            float(geom.mae_failure_vol_mult) * vol_bps,
        )
    else:
        tp_ref_bps = vol_bps
        sl_ref_bps = vol_bps
    tp_margin = np.abs(mfe_bps - tp_ref_bps) / np.maximum(tp_ref_bps, 1e-6)
    sl_margin = np.abs(mae_bps - sl_ref_bps) / np.maximum(sl_ref_bps, 1e-6)
    near_barrier = np.exp(-np.minimum(tp_margin, sl_margin) / 0.15)
    soft_ambiguity = 1.0 - np.minimum(1.0, np.abs(ys - 0.5) * 2.0)
    ambiguity = np.clip(np.maximum(near_barrier, soft_ambiguity), 0.0, 1.0)

    net_edge_bps = mfe_bps - costs_bps - float(p_label.mae_penalty_scale) * mae_bps
    edge_se_bps = np.maximum(25.0, 0.50 * vol_bps + 0.25 * mae_bps)
    edge_lcb_bps = net_edge_bps - max(float(recipe.objective.edge_lcb_se_divisor), 0.0) * edge_se_bps
    temperature = max(float(p_label.net_return_temperature_bps), 1e-6)
    edge_score = np.tanh((edge_lcb_bps - float(p_label.net_return_center_bps)) / temperature)
    mfe_score = np.tanh(mfe_bps / max(float(p_label.mfe_scale_bps), 1e-6))
    mae_score = np.tanh(mae_bps / np.maximum(vol_bps, 1e-6))
    path_quality = np.clip(mfe_score - float(p_label.mae_penalty_scale) * mae_score, -1.0, 1.0)
    uncertainty = _robustness_uncertainty(
        mfe_bps=mfe_bps,
        mae_bps=mae_bps,
        vol_bps=vol_bps,
        net_edge_bps=net_edge_bps,
    )
    timing_penalty = _path_timing_penalty(df_local, n, is_timeout)
    quick_profit = _quick_mfe_profit_score(
        df_local,
        n=n,
        mfe=mfe[:n],
        vol=vol[:n],
        fit_mask=fit_mask,
    )
    geom_signal = np.clip(2.0 * np.asarray(geom_anchor[:n], dtype=np.float64) - 1.0, -1.0, 1.0)
    return {
        "mfe": mfe[:n],
        "mae_abs": mae_abs[:n],
        "vol": vol[:n],
        "is_timeout": is_timeout[:n],
        "mfe_bps": mfe_bps,
        "mae_bps": mae_bps,
        "vol_bps": vol_bps,
        "costs_bps": float(costs_bps),
        "y_geom": y_geom[:n],
        "geom_anchor": geom_anchor[:n],
        "geom_signal": geom_signal,
        "geom_stats": geom_stats,
        "actual_stop": actual_stop,
        "bad_path": bad_path,
        "near_barrier": near_barrier,
        "ambiguity": ambiguity,
        "net_edge_bps": net_edge_bps,
        "edge_lcb_bps": edge_lcb_bps,
        "edge_score": edge_score,
        "path_quality": path_quality,
        "uncertainty": uncertainty,
        "label_stability": 1.0 - uncertainty,
        "timing_penalty": timing_penalty,
        "quick_profit": quick_profit,
    }


def _group_balance_multiplier(
    df: pd.DataFrame,
    *,
    n: int,
    fit_mask: np.ndarray,
    strength: float,
) -> np.ndarray:
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength <= 0.0 or n <= 0:
        return np.ones(n, dtype=np.float64)
    cols: list[str] = []
    for aliases in (
        ("__symbol__", "symbol"),
        ("__regime__", "regime"),
        ("__vol_regime__",),
        ("__spread_regime__",),
        ("__market_regime__",),
    ):
        for col in aliases:
            if col in df.columns:
                cols.append(col)
                break
    if not cols:
        return np.ones(n, dtype=np.float64)
    out = np.ones(n, dtype=np.float64)
    mask = np.asarray(fit_mask, dtype=bool)
    for col in cols:
        values = pd.Series(np.asarray(df[col].values[:n]).astype(str))
        ref_values = values[mask]
        if len(ref_values) == 0:
            continue
        counts = ref_values.value_counts()
        median_count = max(float(np.nanmedian(counts.to_numpy(dtype=np.float64))), 1.0)
        row_counts = values.map(counts).fillna(median_count).to_numpy(dtype=np.float64)
        out *= np.power(np.clip(row_counts / median_count, 1e-6, None), -strength)
    return np.clip(out, 0.25, 3.0)


def _params_from_recipe(recipe: LabelWeightRecipe, *, phase: str = "all") -> dict[str, Any]:
    payload: dict[str, Any] = {"recipe_enabled": str(recipe.stage).strip().lower() != DISABLED_RECIPE_STAGE}
    phase_norm = str(phase or "all").strip().lower()
    if phase_norm in {"label_geometry", "geometry", "hard_labels", "all"}:
        payload.update(asdict(recipe.geometry))
    if phase_norm in {"labels", "all"}:
        payload.update(asdict(recipe.label))
    if phase_norm in {"weights", "all"}:
        payload.update(asdict(recipe.weight))
    if phase_norm in {"label_geometry", "geometry", "hard_labels", "labels", "weights", "all"}:
        generator_payload = asdict(recipe.generator)
        for key, default in GENERATOR_DEFAULTS.items():
            if key in generator_payload and generator_payload[key] is None:
                generator_payload[key] = default
        payload.update({k: v for k, v in generator_payload.items() if v is not None})
    if phase_norm in {"distillation", "all"}:
        payload.update(asdict(recipe.distillation))
    if phase_norm in {"weights", "all"}:
        payload.update({"portfolio_alignment_strength": recipe.objective.portfolio_alignment_strength})
    return payload


def _recipe_from_trial_params(params: dict[str, Any], *, phase: str, name: str = "best") -> LabelWeightRecipe:
    if _param_is_false(params.get("recipe_enabled", True)):
        recipe = LabelWeightRecipe(name=name, stage=DISABLED_RECIPE_STAGE)
        recipe.provenance = {"phase": str(phase or "all").strip().lower(), "source": "optuna_noop_trial"}
        return recipe
    recipe = LabelWeightRecipe(name=name, stage="all")
    phase_norm = str(phase or "all").strip().lower()
    if phase_norm in {"label_geometry", "geometry", "hard_labels", "all"}:
        recipe.geometry = LabelGeometryParams(
            **{k: params[k] for k in asdict(recipe.geometry) if k in params}
        )
    if phase_norm in {"labels", "all"}:
        recipe.label = LabelParams(**{k: params[k] for k in asdict(recipe.label) if k in params})
    if phase_norm in {"weights", "all"}:
        recipe.weight = WeightParams(**{k: params[k] for k in asdict(recipe.weight) if k in params})
    if phase_norm in {"label_geometry", "geometry", "hard_labels", "labels", "weights", "all"}:
        recipe.generator = GeneratorParams(
            **{k: params[k] for k in asdict(recipe.generator) if k in params}
        )
    if phase_norm in {"distillation", "all"}:
        recipe.distillation = DistillationParams(
            **{k: params[k] for k in asdict(recipe.distillation) if k in params}
        )
    if phase_norm in {"weights", "all"} and "portfolio_alignment_strength" in params:
        recipe.objective.portfolio_alignment_strength = float(params["portfolio_alignment_strength"])
    recipe.provenance = {"phase": phase_norm, "source": "optuna_best_trial_params"}
    return recipe


def _param_is_false(value: Any) -> bool:
    if isinstance(value, bool):
        return not value
    return str(value).strip().lower() in {"0", "false", "no", "n", "off"}


def _phase_norm(phase: str) -> str:
    raw = str(phase or "all").strip().lower()
    if raw in {"geometry", "hard_labels"}:
        return "label_geometry"
    return raw


def _layers_for_phase(phase: str) -> tuple[str, ...]:
    phase_norm = _phase_norm(phase)
    if phase_norm == "label_geometry":
        return GEOMETRY_LAYERS
    if phase_norm == "labels":
        return LABEL_LAYERS
    if phase_norm == "weights":
        return WEIGHT_LAYERS
    if phase_norm == "distillation":
        return DISTILLATION_LAYERS
    if phase_norm == "all":
        return GEOMETRY_LAYERS + LABEL_LAYERS + WEIGHT_LAYERS + DISTILLATION_LAYERS
    return ()


def _resolve_optuna_layer(trial: Any, *, phase: str, layer: str) -> str:
    layer_norm = str(layer or LAYER_ALL).strip().lower()
    if layer_norm in {"", "none"}:
        layer_norm = LAYER_ALL
    if layer_norm == LAYER_ALL:
        return LAYER_ALL
    allowed = _layers_for_phase(phase)
    if layer_norm == LAYER_AUTO:
        if not allowed:
            return LAYER_ALL
        return str(trial.suggest_categorical("recipe_layer", list(allowed)))
    if layer_norm not in allowed:
        allowed_text = ", ".join((LAYER_ALL, LAYER_AUTO, *allowed))
        raise ValueError(
            f"Layer {layer_norm!r} is not valid for phase {_phase_norm(phase)!r}. "
            f"Allowed: {allowed_text}"
        )
    return layer_norm


def _layer_active(active_layer: str, *layers: str) -> bool:
    layer_norm = str(active_layer or LAYER_ALL).strip().lower()
    return layer_norm == LAYER_ALL or layer_norm in set(layers)


def _noop_trial_params() -> dict[str, Any]:
    return {"recipe_enabled": False}


def _noop_recipe_for_phase(
    *,
    phase: str,
    base_recipe: LabelWeightRecipe | None,
    name: str,
) -> LabelWeightRecipe:
    phase_norm = str(phase or "all").strip().lower()
    if base_recipe is not None:
        recipe = LabelWeightRecipe.from_dict(base_recipe.to_dict())
        recipe.name = name
        if phase_norm == "distillation":
            recipe.distillation = neutral_distillation_params()
        recipe.provenance = dict(recipe.provenance)
        recipe.provenance.update(
            {
                "phase": phase_norm,
                "source": "optuna_noop_trial",
                "noop_meaning": "fixed_base_recipe_unchanged",
            }
        )
        return recipe
    recipe = LabelWeightRecipe(name=name, stage=DISABLED_RECIPE_STAGE)
    recipe.provenance = {
        "phase": phase_norm,
        "source": "optuna_noop_trial",
        "noop_meaning": "pre_hpo_neutral_baseline_no_recipe_transforms",
    }
    return recipe


def _trial_is_noop(trial: Any) -> bool:
    return _param_is_false(dict(getattr(trial, "params", {})).get("recipe_enabled", True))


def _read_json_if_exists(path: Path) -> dict[str, Any] | None:
    p = path.expanduser()
    if not p.exists():
        return None
    with p.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    return dict(payload) if isinstance(payload, dict) else None


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=str(path.parent), delete=False) as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")
        tmp = Path(fh.name)
    tmp.replace(path)


def _enqueue_previous_best(study: Any, path: Path, *, phase: str) -> bool:
    payload = _read_json_if_exists(path)
    if not payload:
        return False
    params = None
    recipe_payload = payload.get("recipe")
    if isinstance(recipe_payload, dict):
        params = _params_from_recipe(LabelWeightRecipe.from_dict(recipe_payload), phase=phase)
    elif {"label", "weight", "distillation"}.intersection(payload):
        params = _params_from_recipe(LabelWeightRecipe.from_dict(payload), phase=phase)
    if not isinstance(params, dict):
        params = payload.get("params")
    if not isinstance(params, dict) or not params:
        return False
    try:
        study.enqueue_trial(dict(params), skip_if_exists=True)
    except TypeError:
        study.enqueue_trial(dict(params))
    return True


def _enqueue_noop_trial(study: Any) -> bool:
    params = _noop_trial_params()
    try:
        study.enqueue_trial(params, skip_if_exists=True)
    except TypeError:
        study.enqueue_trial(params)
    return True


def _write_best_artifacts(
    *,
    out_dir: Path,
    best_path: Path,
    study_name: str,
    phase: str,
    trial: Any,
    recipe_path: str | None,
) -> None:
    recipe_payload: dict[str, Any] | None = None
    if recipe_path:
        raw = _read_json_if_exists(Path(recipe_path))
        if raw is not None:
            recipe_payload = raw
    if recipe_payload is None:
        recipe_payload = _recipe_from_trial_params(dict(trial.params), phase=phase).to_dict()
    recipe_payload["name"] = f"{study_name}_best_trial_{trial.number}"
    recipe_payload.setdefault("provenance", {})
    if isinstance(recipe_payload["provenance"], dict):
        recipe_payload["provenance"].update(
            {
                "study_name": str(study_name),
                "phase": str(phase),
                "best_trial_number": int(trial.number),
                "best_trial_value": float(trial.value),
            }
        )
    trial_params = dict(trial.params)
    effective_params = dict(trial_params)
    if _trial_is_noop(trial):
        try:
            effective_params = _params_from_recipe(LabelWeightRecipe.from_dict(recipe_payload), phase=phase)
        except Exception:
            effective_params = dict(trial_params)
    best_trial_payload = {
        "number": int(trial.number),
        "value": float(trial.value),
        "params": effective_params,
        "trial_params": trial_params,
        "user_attrs": dict(trial.user_attrs),
        "recipe_path": str(recipe_path or ""),
        "recipe": recipe_payload,
    }
    _atomic_write_json(out_dir / "best_trial.json", best_trial_payload)
    _atomic_write_json(out_dir / "best_recipe.json", recipe_payload)
    _atomic_write_json(best_path, recipe_payload)


def _write_rejected_promotion_artifact(
    *,
    out_dir: Path,
    study_name: str,
    phase: str,
    best_trial: Any,
    incumbent_trial: Any | None,
    promotion_margin: float,
    promotion_comparison: dict[str, Any] | None = None,
) -> None:
    payload = {
        "study_name": str(study_name),
        "phase": str(phase),
        "promotion_rejected": True,
        "reason": "best trial did not beat incumbent noop/base recipe by the required margin",
        "promotion_margin": float(promotion_margin),
        "best_trial_number": int(best_trial.number),
        "best_trial_value": float(best_trial.value),
        "best_trial_params": dict(best_trial.params),
        "best_trial_recipe_path": str(best_trial.user_attrs.get("recipe_path", "")),
        "incumbent_trial_number": None if incumbent_trial is None else int(incumbent_trial.number),
        "incumbent_trial_value": None if incumbent_trial is None else float(incumbent_trial.value),
        "incumbent_trial_params": {} if incumbent_trial is None else dict(incumbent_trial.params),
        "incumbent_recipe_path": "" if incumbent_trial is None else str(incumbent_trial.user_attrs.get("recipe_path", "")),
    }
    if promotion_comparison is not None:
        payload["promotion_comparison"] = promotion_comparison
    _atomic_write_json(out_dir / "promotion_rejected.json", payload)


def _make_optuna_pruner(optuna_mod: Any, pruner_name: str) -> Any:
    name = str(pruner_name or "successive_halving").strip().lower()
    if name in {"successive_halving", "sha", "halving"}:
        return optuna_mod.pruners.SuccessiveHalvingPruner(
            min_resource=1,
            reduction_factor=2,
            min_early_stopping_rate=0,
        )
    if name in {"median", "median_pruner"}:
        return optuna_mod.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=0,
            interval_steps=1,
            n_min_trials=2,
        )
    if name in {"hyperband", "hb"}:
        return optuna_mod.pruners.HyperbandPruner(
            min_resource=1,
            max_resource=3,
            reduction_factor=2,
        )
    if name in {"none", "nop", "no_pruner"}:
        return optuna_mod.pruners.NopPruner()
    raise ValueError(f"Unknown Optuna pruner: {pruner_name}")


def load_recipe(path: str) -> LabelWeightRecipe | None:
    if not path:
        return None
    if path == DISABLED_RECIPE_KEY:
        return None
    if path == HARDCODED_DEFAULT_RECIPE_KEY:
        return LabelWeightRecipe(
            name="hardcoded_default",
            provenance={"source": "hardcoded_defaults", "best_recipe_bypassed": True},
        )
    if path in _RECIPE_CACHE:
        return _RECIPE_CACHE[path]
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"Label/weight recipe does not exist: {p}")
    with p.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)
    recipe = LabelWeightRecipe.from_dict(raw)
    _RECIPE_CACHE[path] = recipe
    return recipe


def load_recipe_from_env_or_cfg(
    cfg: dict[str, Any] | None = None,
    *,
    scope: str | None = None,
) -> LabelWeightRecipe | None:
    return load_recipe(recipe_path_from_env_or_cfg(cfg, scope=scope))


def recipe_applies(recipe: LabelWeightRecipe | None, stage: str) -> bool:
    if recipe is None:
        return False
    target = str(recipe.stage or "all").strip().lower()
    current = str(stage or "").strip().lower()
    return target in {"", "all", current} or (target == "base" and current == "train_base") or (
        target == "meta" and current == "train_meta"
    )


def _non_null_generator_overrides(generator: GeneratorParams) -> dict[str, float]:
    raw = asdict(generator)
    out: dict[str, float] = {}
    for key, value in raw.items():
        if value is None:
            continue
        try:
            val = float(value)
        except Exception:
            continue
        if math.isfinite(val):
            out[str(key)] = val
    if "outcome_weight_clip_min" in out and "outcome_weight_clip_max" in out:
        lo = float(out["outcome_weight_clip_min"])
        hi = float(out["outcome_weight_clip_max"])
        if lo > hi:
            out["outcome_weight_clip_min"] = hi
            out["outcome_weight_clip_max"] = lo
    if "policy_label_max_hold_hours" in out:
        out["policy_label_max_hold_hours"] = float(max(1, int(round(out["policy_label_max_hold_hours"]))))
    return out


def apply_generator_recipe_to_cfg(
    cfg: dict[str, Any] | None,
    *,
    stage: str,
) -> dict[str, Any]:
    """Return cfg with recipe-controlled native label/weight generator knobs applied."""

    cfg_local = dict(cfg or {})
    recipe = load_recipe_from_env_or_cfg(cfg_local, scope=_scope_from_stage(stage))
    if not recipe_applies(recipe, stage):
        return cfg_local
    assert recipe is not None
    overrides = _non_null_generator_overrides(recipe.generator)
    if not overrides:
        return cfg_local
    cfg_local.update(overrides)
    cfg_local["label_weight_generator_recipe"] = str(recipe.name)
    cfg_local["label_weight_generator_stage"] = str(stage)
    return cfg_local


def build_native_mfe_mae_soft_label_from_frame(
    df: pd.DataFrame,
    y_hard: np.ndarray,
    *,
    cfg: dict[str, Any] | None = None,
    stage: str = "train_base",
    label: str = "native",
) -> tuple[np.ndarray, dict[str, Any]]:
    cfg_local = apply_generator_recipe_to_cfg(cfg, stage=stage)
    y_ref = np.asarray(y_hard, dtype=np.float32).reshape(-1)
    fallback = np.clip(np.nan_to_num(y_ref, nan=0.0), 0.0, 1.0).astype(np.float32)
    capped_mode = policy_net_timeout_capped_label_mode(cfg_local)
    if capped_mode is not None:
        cap_kind, family, target_mode = capped_mode
        return build_policy_net_capped_execution_label_from_frame(
            df,
            y_ref,
            cfg=cfg_local,
            label=label,
            cap_kind=cap_kind,
            family=family,
            target_mode=target_mode,
        )
    if policy_net_exec_guard_label_mode_requested(cfg_local):
        return build_policy_net_capped_execution_label_from_frame(
            df,
            y_ref,
            cfg=cfg_local,
            label=label,
            cap_kind="none",
            family="exec_guard",
            target_mode="s34_exec_guard_broad_policy",
        )
    if policy_net_path_blend_label_mode_requested(cfg_local):
        return build_policy_net_path_blend_label_from_frame(
            df,
            y_ref,
            cfg=cfg_local,
            label=label,
        )
    if policy_net_soft_label_mode_requested(cfg_local):
        return build_policy_net_soft_label_from_frame(
            df,
            y_ref,
            cfg=cfg_local,
            label=label,
        )
    required = {"__mfe_ret__", "__mae_ret__", "__barrier_pct__"}
    missing = sorted(c for c in required if c not in df.columns)
    if missing:
        return fallback, {"enabled": False, "reason": "missing_columns", "missing_columns": missing}
    n = min(len(df), len(y_ref))
    df_local = df.reset_index(drop=True)
    mfe, mae_abs, vol, is_timeout = _path_arrays(df_local, n)
    y_ref = y_ref[:n]
    ablation_mode = str(
        os.getenv("EPM_LABEL_ABLATION_MODE", cfg_local.get("label_ablation_mode", ""))
        or ""
    ).strip().lower()
    if ablation_mode in {"3", "net_executable", "net_executable_soft_label"}:
        cost = max(
            0.0,
            float(
                os.getenv(
                    "EPM_EXECUTION_AWARE_COST_BPS",
                    cfg_local.get("execution_aware_cost_bps", 68.83),
                )
            ),
        ) / 10_000.0
        adverse_lambda = float(
            os.getenv(
                "EPM_NET_EXECUTABLE_MAE_LAMBDA",
                cfg_local.get("net_executable_mae_lambda", 0.35),
            )
        )
        center = float(
            os.getenv(
                "EPM_NET_EXECUTABLE_CENTER_VOL",
                cfg_local.get("net_executable_center_vol", 0.0),
            )
        )
        temperature = max(
            1e-6,
            float(
                os.getenv(
                    "EPM_NET_EXECUTABLE_TEMPERATURE_VOL",
                    cfg_local.get("net_executable_temperature_vol", 0.35),
                )
            ),
        )
        net_path_edge = ((mfe[:n] - cost) / np.maximum(vol[:n], 1e-12)) - adverse_lambda * (
            mae_abs[:n] / np.maximum(vol[:n], 1e-12)
        )
        soft = _sigmoid((net_path_edge - center) / temperature)
        soft = np.where(is_timeout[:n], 0.5 * soft + 0.25, soft)
        stats = {
            "enabled": True,
            "label": str(label),
            "target_mode": "net_executable_vol_normalized",
            "soft_mean": float(np.mean(soft)) if len(soft) else float("nan"),
            "soft_std": float(np.std(soft)) if len(soft) else float("nan"),
            "execution_cost_bps": float(cost * 10_000.0),
            "mae_lambda": float(adverse_lambda),
            "center_vol": float(center),
            "temperature_vol": float(temperature),
        }
    else:
        costs = float(
            cfg_local.get(
                "lgbm_soft_label_costs",
                cfg_local.get("round_trip_fee_pct", cfg_local.get("fee_pct", 0.0)),
            )
            or 0.0
        )
        min_opportunity_mult = float(
            cfg_local.get(
                "lgbm_soft_label_min_opportunity_mult",
                os.getenv("EPM_LGBM_SOFT_LABEL_MIN_OPPORTUNITY_MULT", "0.25"),
            )
        )
        temperature = max(
            1e-12,
            float(
                cfg_local.get(
                    "lgbm_soft_label_temperature",
                    os.getenv("EPM_LGBM_SOFT_LABEL_TEMPERATURE", "0.20"),
                )
            ),
        )
        mfe_net = np.maximum(mfe[:n] - costs, 0.0)
        ratio_score = mfe_net / (mfe_net + mae_abs[:n] + 1e-12)
        opportunity_score = np.clip(
            mfe_net / (max(min_opportunity_mult, 1e-12) * vol[:n] + 1e-12),
            0.0,
            1.0,
        )
        raw_score = ratio_score * opportunity_score
        quality = _sigmoid((raw_score - 0.5) / temperature)
        is_tp = (y_ref >= 0.5) & (~is_timeout[:n])
        is_sl = (y_ref < 0.5) & (~is_timeout[:n])
        soft = np.empty(n, dtype=np.float64)
        soft[is_tp] = 0.65 + 0.35 * quality[is_tp]
        soft[is_sl] = 0.35 * quality[is_sl]
        soft[is_timeout[:n]] = np.clip(quality[is_timeout[:n]], 0.3, 0.7)
        stats = {
            "enabled": True,
            "label": str(label),
            "target_mode": "mfe_mae_quality",
            "quality_mean": float(np.mean(quality)) if len(quality) else float("nan"),
            "quality_std": float(np.std(quality)) if len(quality) else float("nan"),
            "soft_mean": float(np.mean(soft)) if len(soft) else float("nan"),
            "soft_std": float(np.std(soft)) if len(soft) else float("nan"),
            "costs": float(costs),
            "min_opportunity_mult": float(min_opportunity_mult),
            "temperature": float(temperature),
        }
    soft = np.clip(np.nan_to_num(soft, nan=0.5), 0.0, 1.0).astype(np.float32)
    if n < len(fallback):
        merged = fallback.copy()
        merged[:n] = soft
        soft = merged
    return soft.astype(np.float32, copy=False), stats


def build_native_base_sample_weight_from_frame(
    df: pd.DataFrame,
    y_hard: np.ndarray,
    y_ret: np.ndarray,
    *,
    cfg: dict[str, Any] | None = None,
    stage: str = "train_base",
) -> tuple[np.ndarray, dict[str, Any]]:
    from .sample_weights import compute_mfe_mae_weights

    cfg_local = apply_generator_recipe_to_cfg(cfg, stage=stage)
    n = min(len(df), len(y_hard), len(y_ret))
    if n <= 0:
        return np.asarray([], dtype=np.float32), {"enabled": False, "reason": "empty"}
    df_local = df.reset_index(drop=True)
    y = np.asarray(y_hard, dtype=np.float32).reshape(-1)[:n]
    ret = np.nan_to_num(np.asarray(y_ret, dtype=np.float32).reshape(-1)[:n], nan=0.0)
    if "__y_outcome__" in df_local.columns:
        outcome = pd.to_numeric(df_local["__y_outcome__"].iloc[:n], errors="coerce").to_numpy(dtype=np.float64)
    else:
        timeout = (
            pd.to_numeric(df_local["__is_timeout__"].iloc[:n], errors="coerce").to_numpy(dtype=np.float64) > 0.5
            if "__is_timeout__" in df_local.columns
            else np.zeros(n, dtype=bool)
        )
        outcome = np.where(timeout, 1.0, np.where(y >= 0.5, 2.0, 0.0))
    is_tp = outcome == 2.0
    is_sl = outcome == 0.0
    is_to = outcome == 1.0
    w_outcome = np.ones(n, dtype=np.float64)
    timeout_weight = float(cfg_local.get("timeout_weight", 0.4))
    if np.any(is_to):
        tp_for_to = (
            pd.to_numeric(df_local["__tp__"].iloc[:n], errors="coerce").to_numpy(dtype=np.float64)
            if "__tp__" in df_local.columns
            else pd.to_numeric(df_local.get("__barrier_pct__", pd.Series(0.02, index=df_local.index)).iloc[:n], errors="coerce").to_numpy(dtype=np.float64)
        )
        sl_for_to = (
            pd.to_numeric(df_local["__sl__"].iloc[:n], errors="coerce").to_numpy(dtype=np.float64)
            if "__sl__" in df_local.columns
            else 0.5 * tp_for_to
        )
        tp_for_to = np.clip(np.abs(np.nan_to_num(tp_for_to, nan=0.02)), 1e-4, None)
        sl_for_to = np.clip(np.abs(np.nan_to_num(sl_for_to, nan=0.01)), 1e-4, None)
        s_score = np.clip((ret[is_to] + sl_for_to[is_to]) / (tp_for_to[is_to] + sl_for_to[is_to] + 1e-9), 0.0, 1.0)
        w_to = (1.0 + 1.5 * ((1.0 - s_score) ** 2)) * timeout_weight
        if np.any(is_sl):
            avg_to = float(np.mean(w_to)) if len(w_to) else 0.0
            target_avg_to = 0.05 * float(np.mean(w_outcome[is_sl]))
            if avg_to < target_avg_to and avg_to > 1e-12:
                w_to *= target_avg_to / avg_to
        w_outcome[is_to] = w_to
    clip_min = float(cfg_local.get("outcome_weight_clip_min", 0.5))
    clip_max = float(cfg_local.get("outcome_weight_clip_max", 2.0))
    if clip_min > clip_max:
        clip_min, clip_max = clip_max, clip_min
    w_outcome = np.clip(w_outcome, clip_min, clip_max)

    abs_ret = np.abs(ret).astype(np.float64)
    mag_q = float(np.nanquantile(abs_ret, 0.95)) if abs_ret.size else 1.0
    mag_q = max(mag_q if math.isfinite(mag_q) else 1.0, 1e-9)
    w_magnitude = 0.5 + np.clip(abs_ret, 0.0, mag_q) / mag_q

    mfe, mae_abs, vol, timeout = _path_arrays(df_local, n)
    if "__tp__" in df_local.columns:
        tp = pd.to_numeric(df_local["__tp__"].iloc[:n], errors="coerce").to_numpy(dtype=np.float64)
    else:
        tp = vol[:n]
    if "__sl__" in df_local.columns:
        sl = pd.to_numeric(df_local["__sl__"].iloc[:n], errors="coerce").to_numpy(dtype=np.float64)
    else:
        sl = 0.5 * np.asarray(tp, dtype=np.float64)
    tp = np.clip(np.abs(np.nan_to_num(tp, nan=0.02)), 1e-4, None)
    sl = np.clip(np.abs(np.nan_to_num(sl, nan=0.01)), 1e-4, None)
    w_mfe_mae = compute_mfe_mae_weights(
        mfe=mfe[:n],
        mae=mae_abs[:n],
        tp=tp,
        sl=sl,
        is_timeout=timeout[:n] | is_to,
        touch_margin=None,
        w_min=float(cfg_local.get("mfe_mae_w_min", 0.5)),
        tau=float(cfg_local.get("mfe_mae_tau", 1.0)),
        cost_floor=float(cfg_local.get("mfe_mae_cost_floor", 0.001)),
    )
    w_mfe_mae = np.clip((2.0 * np.asarray(w_mfe_mae, dtype=np.float32)) - 0.5, 0.5, 1.5)
    weight = np.nan_to_num(w_magnitude * w_outcome * w_mfe_mae, nan=1.0, posinf=1.5, neginf=0.5)
    stats = {
        "enabled": True,
        "timeout_weight": float(timeout_weight),
        "outcome_weight_clip_min": float(clip_min),
        "outcome_weight_clip_max": float(clip_max),
        "mfe_mae_w_min": float(cfg_local.get("mfe_mae_w_min", 0.5)),
        "mfe_mae_tau": float(cfg_local.get("mfe_mae_tau", 1.0)),
        "mean": float(np.mean(weight)) if len(weight) else float("nan"),
        "p95": float(np.percentile(weight, 95)) if len(weight) else float("nan"),
    }
    return weight.astype(np.float32, copy=False), stats


def _path_arrays(df: pd.DataFrame, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mfe = np.asarray(df.get("__mfe_ret__", pd.Series(np.zeros(n))).values[:n], dtype=np.float64)
    mae_raw = np.asarray(df.get("__mae_ret__", pd.Series(np.zeros(n))).values[:n], dtype=np.float64)
    finite_mae = mae_raw[np.isfinite(mae_raw)]
    mae_abs = (
        np.maximum(mae_raw, 0.0)
        if finite_mae.size and float(np.nanmedian(finite_mae)) >= 0.0
        else np.maximum(-mae_raw, 0.0)
    )
    vol = np.asarray(df.get("__barrier_pct__", pd.Series(np.ones(n))).values[:n], dtype=np.float64)
    timeout = (
        np.asarray(df["__is_timeout__"].values[:n], dtype=float) > 0.5
        if "__is_timeout__" in df.columns
        else np.zeros(n, dtype=bool)
    )
    return (
        np.nan_to_num(np.maximum(mfe, 0.0), nan=0.0),
        np.nan_to_num(np.maximum(mae_abs, 0.0), nan=0.0),
        np.clip(np.nan_to_num(np.abs(vol), nan=1.0), 1e-8, None),
        timeout,
    )


def fixed_execution_cost_bps(cfg: dict[str, Any] | None, recipe: LabelWeightRecipe) -> float:
    cfg_local = cfg if isinstance(cfg, dict) else {}
    if recipe.execution_costs:
        return float(
            sum(float(v) for k, v in recipe.execution_costs.items() if k in DEFAULT_EXECUTION_COST_KEYS)
        )
    return float(
        os.getenv(
            "EPM_EXECUTION_AWARE_COST_BPS",
            cfg_local.get("execution_aware_cost_bps", 68.83),
        )
    )


def _geometry_reference(
    df: pd.DataFrame,
    *,
    n: int,
    y_hard: np.ndarray,
    mfe: np.ndarray,
    mae_abs: np.ndarray,
    vol: np.ndarray,
    recipe: LabelWeightRecipe,
    costs_bps: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    geom = recipe.geometry
    y_ref = (np.asarray(y_hard, dtype=np.float64).reshape(-1)[:n] >= 0.5).astype(np.float64)
    if not bool(geom.enabled):
        return y_ref, y_ref, {"enabled": False}

    mfe_bps = 10_000.0 * np.asarray(mfe[:n], dtype=np.float64)
    mae_bps = 10_000.0 * np.asarray(mae_abs[:n], dtype=np.float64)
    vol_bps = np.clip(10_000.0 * np.asarray(vol[:n], dtype=np.float64), 1e-6, None)
    tp_req_bps = np.maximum(float(geom.tp_vol_mult) * vol_bps, float(geom.min_executable_net_bps) + float(costs_bps))
    sl_req_bps = np.maximum(
        float(geom.sl_as_tp_pct) * tp_req_bps,
        float(geom.mae_failure_vol_mult) * vol_bps,
    )
    mfe_ok = mfe_bps >= tp_req_bps
    mae_bad = mae_bps >= sl_req_bps

    bars_to_mfe = np.full(n, np.inf, dtype=np.float64)
    bars_to_mae = np.full(n, np.inf, dtype=np.float64)
    df_local = df.reset_index(drop=True)
    if "__bars_to_mfe__" in df_local.columns:
        bars_to_mfe = np.asarray(df_local["__bars_to_mfe__"].values[:n], dtype=np.float64)
    if "__bars_to_mae__" in df_local.columns:
        bars_to_mae = np.asarray(df_local["__bars_to_mae__"].values[:n], dtype=np.float64)
    elif "__y_outcome__" in df_local.columns:
        outcome = pd.to_numeric(df_local["__y_outcome__"].iloc[:n], errors="coerce").to_numpy(dtype=np.float64)
        bars_to_mae = np.where(outcome == 0.0, np.minimum(np.nan_to_num(bars_to_mfe, nan=np.inf), 1.0), np.inf)
    horizon = max(float(geom.label_horizon_bars), 1.0)
    mfe_in_time = mfe_ok & np.isfinite(bars_to_mfe) & (bars_to_mfe <= horizon)
    sl_in_time = mae_bad & np.isfinite(bars_to_mae) & (bars_to_mae <= horizon)
    stop_before_tp = sl_in_time & (~mfe_in_time | (bars_to_mae <= bars_to_mfe))

    net_proxy_bps = mfe_bps - float(costs_bps) - mae_bps
    executable = net_proxy_bps >= float(geom.min_executable_net_bps)
    hard = mfe_in_time & executable & ~stop_before_tp

    trailing_penalty = np.zeros(n, dtype=np.float64)
    activation_bps = float(geom.trailing_activation_vol_mult) * vol_bps
    trail_active = mfe_bps >= activation_bps
    giveback_bad = trail_active & (mae_bps >= float(geom.trailing_giveback_pct) * np.maximum(mfe_bps, 1e-6))
    trailing_penalty[giveback_bad] = 0.35

    soft_anchor = np.where(hard, 1.0, 0.0).astype(np.float64)
    timed_out = ~(mfe_in_time | sl_in_time)
    soft_anchor = np.where(timed_out, float(geom.timeout_value), soft_anchor)
    soft_anchor = np.clip(soft_anchor - trailing_penalty, 0.0, 1.0)
    stats = {
        "enabled": True,
        "hard_positive_rate": float(np.mean(hard)) if n else float("nan"),
        "soft_anchor_mean": float(np.mean(soft_anchor)) if n else float("nan"),
        "tp_req_bps_mean": float(np.mean(tp_req_bps)) if n else float("nan"),
        "sl_req_bps_mean": float(np.mean(sl_req_bps)) if n else float("nan"),
        "timeout_rate": float(np.mean(timed_out)) if n else float("nan"),
        "stop_before_tp_rate": float(np.mean(stop_before_tp)) if n else float("nan"),
        "trailing_giveback_rate": float(np.mean(giveback_bad)) if n else float("nan"),
    }
    return hard.astype(np.float64), soft_anchor.astype(np.float64), stats


def apply_geometry_recipe_to_labels(
    df: pd.DataFrame,
    y_hard: np.ndarray,
    *,
    cfg: dict[str, Any] | None = None,
    stage: str,
    label: str,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
    """Apply recipe label geometry as an upstream hard-label regeneration step."""

    recipe = load_recipe_from_env_or_cfg(cfg, scope=_scope_from_stage(stage))
    if not recipe_applies(recipe, stage):
        return df, np.asarray(y_hard, dtype=np.float32), {"enabled": False, "reason": "no_recipe"}
    assert recipe is not None
    geom = recipe.geometry
    if not bool(geom.enabled):
        return df, np.asarray(y_hard, dtype=np.float32), {"enabled": False, "reason": "geometry_disabled"}
    y_ref = np.asarray(y_hard, dtype=np.float64).reshape(-1)
    n = _require_equal_lengths("apply_geometry_recipe_to_labels", df=df, y_hard=y_ref)
    required = ("__mfe_ret__", "__mae_ret__", "__barrier_pct__", "__bars_to_mfe__")
    missing = [col for col in required if col not in df.columns]
    unusable: list[str] = []
    for col in required:
        if col not in df.columns:
            continue
        vals = pd.to_numeric(df[col].iloc[:n], errors="coerce").to_numpy(dtype=np.float64)
        if not np.isfinite(vals).any():
            unusable.append(col)
    if missing or unusable:
        raise RuntimeError(
            "Label geometry recipe requires path timing columns; "
            f"missing={missing}, unusable={unusable}, label={label!r}. "
            "Regenerate or rehydrate labels before tuning geometry."
        )

    df_local = df.reset_index(drop=True).copy()
    mfe, mae_abs, vol, _ = _path_arrays(df_local, n)
    costs_bps = fixed_execution_cost_bps(cfg, recipe)
    y_geom, soft_anchor, stats = _geometry_reference(
        df_local,
        n=n,
        y_hard=y_ref,
        mfe=mfe,
        mae_abs=mae_abs,
        vol=vol,
        recipe=recipe,
        costs_bps=costs_bps,
    )
    mfe_bps = 10_000.0 * np.asarray(mfe[:n], dtype=np.float64)
    mae_bps = 10_000.0 * np.asarray(mae_abs[:n], dtype=np.float64)
    vol_bps = np.clip(10_000.0 * np.asarray(vol[:n], dtype=np.float64), 1e-6, None)
    tp_req_bps = np.maximum(float(geom.tp_vol_mult) * vol_bps, float(geom.min_executable_net_bps) + float(costs_bps))
    sl_req_bps = np.maximum(
        float(geom.sl_as_tp_pct) * tp_req_bps,
        float(geom.mae_failure_vol_mult) * vol_bps,
    )
    bars_to_mfe = pd.to_numeric(df_local["__bars_to_mfe__"].iloc[:n], errors="coerce").to_numpy(dtype=np.float64)
    bars_to_mae_was_generated = "__bars_to_mae__" not in df_local.columns
    if bars_to_mae_was_generated:
        if "__y_outcome__" in df_local.columns:
            outcome_ref = pd.to_numeric(df_local["__y_outcome__"].iloc[:n], errors="coerce").to_numpy(dtype=np.float64)
            bars_to_mae = np.where(outcome_ref == 0.0, np.minimum(np.nan_to_num(bars_to_mfe, nan=np.inf), 1.0), np.inf)
        else:
            bars_to_mae = np.full(n, np.inf, dtype=np.float64)
        df_local.loc[: n - 1, "__bars_to_mae__"] = bars_to_mae.astype(np.float32)
    else:
        bars_to_mae = pd.to_numeric(df_local["__bars_to_mae__"].iloc[:n], errors="coerce").to_numpy(dtype=np.float64)
    horizon = max(float(geom.label_horizon_bars), 1.0)
    mfe_in_time = (mfe_bps >= tp_req_bps) & np.isfinite(bars_to_mfe) & (bars_to_mfe <= horizon)
    sl_in_time = (mae_bps >= sl_req_bps) & np.isfinite(bars_to_mae) & (bars_to_mae <= horizon)
    stop_before_tp = sl_in_time & (~mfe_in_time | (bars_to_mae <= bars_to_mfe))
    timed_out = ~(mfe_in_time | sl_in_time)
    outcome = np.where(y_geom >= 0.5, 2.0, np.where(timed_out, 1.0, 0.0)).astype(np.float32)

    df_local.loc[: n - 1, "__y_bin__"] = y_geom.astype(np.float32)
    df_local.loc[: n - 1, "__y_outcome__"] = outcome
    df_local.loc[: n - 1, "__is_timeout__"] = (outcome == 1.0).astype(np.float32)
    df_local.loc[: n - 1, "__tp__"] = (tp_req_bps / 10_000.0).astype(np.float32)
    df_local.loc[: n - 1, "__sl__"] = (sl_req_bps / 10_000.0).astype(np.float32)
    out_y = np.asarray(y_ref, dtype=np.float32).copy()
    out_y[:n] = y_geom.astype(np.float32)
    stats = {
        **stats,
        "enabled": True,
        "recipe": recipe.name,
        "label": str(label),
        "stage": str(stage),
        "modifier_mode": "upstream_geometry",
        "hard_positive_rate_before": float(np.mean(y_ref[:n] >= 0.5)) if n else float("nan"),
        "hard_positive_rate_after": float(np.mean(out_y[:n] >= 0.5)) if n else float("nan"),
        "hard_changed_frac": float(np.mean((out_y[:n] >= 0.5) != (y_ref[:n] >= 0.5))) if n else float("nan"),
        "soft_anchor_std": float(np.std(soft_anchor)) if len(soft_anchor) else float("nan"),
        "geometry_stop_before_tp_rate": float(np.mean(stop_before_tp)) if n else float("nan"),
        "bars_to_mae_generated": bool(bars_to_mae_was_generated),
    }
    return df_local, out_y, stats


def apply_label_recipe(
    df: pd.DataFrame,
    y_hard: np.ndarray,
    current_soft: np.ndarray,
    *,
    cfg: dict[str, Any] | None = None,
    stage: str,
    label: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    recipe = load_recipe_from_env_or_cfg(cfg, scope=_scope_from_stage(stage))
    if not recipe_applies(recipe, stage):
        return np.asarray(current_soft, dtype=np.float32), {"enabled": False, "reason": "no_recipe"}
    assert recipe is not None
    y_ref = np.asarray(y_hard, dtype=np.float64).reshape(-1)
    base = np.asarray(current_soft, dtype=np.float64).reshape(-1)
    n = _require_equal_lengths("apply_label_recipe", df=df, y_hard=y_ref, current_soft=base)
    label_l = str(label or "").lower()
    stage_l = str(stage or "").lower()
    p = recipe.label
    label_strength = float(np.clip(p.label_modifier_strength, 0.0, 1.0))
    if label_strength <= 0.0:
        return np.asarray(current_soft, dtype=np.float32), {
            "enabled": True,
            "recipe": recipe.name,
            "label": str(label),
            "stage": str(stage),
            "modifier_mode": "residual",
            "label_modifier_strength": 0.0,
            "reason": "zero_label_modifier_strength",
        }
    if stage_l == "train_meta" and "tbm" in label_l:
        required = ("__mfe_ret__", "__mae_ret__", "__barrier_pct__", "__bars_to_mfe__")
        missing = [col for col in required if col not in df.columns]
        unusable = []
        for col in required:
            if col not in df.columns:
                continue
            try:
                vals = np.asarray(df[col].values[:n], dtype=np.float64)
                if not np.isfinite(vals).any():
                    unusable.append(col)
            except Exception:
                unusable.append(col)
        if missing or unusable:
            raise RuntimeError(
                "Meta TBM label/weight recipe requires path-quality columns; "
                f"missing={missing}, unusable={unusable}, label={label!r}. "
                "Regenerate or rehydrate the meta source frame instead of using "
                "silent defaults."
            )
    state = _path_economics_state(
        df,
        n=n,
        y_hard=y_ref,
        y_soft=base,
        recipe=recipe,
        cfg=cfg,
    )
    geom_stats = dict(state["geom_stats"])
    actual_stop = np.asarray(state["actual_stop"], dtype=bool)
    bad_path = np.asarray(state["bad_path"], dtype=bool)
    is_timeout = np.asarray(state["is_timeout"], dtype=bool)
    uncertainty = np.asarray(state["uncertainty"], dtype=np.float64)
    timing_penalty = np.asarray(state["timing_penalty"], dtype=np.float64)
    quick_profit = np.asarray(state["quick_profit"], dtype=np.float64)
    ambiguity = np.asarray(state["ambiguity"], dtype=np.float64)

    delta = np.asarray(state["edge_score"], dtype=np.float64)
    delta += float(p.path_quality_mix) * np.asarray(state["path_quality"], dtype=np.float64)
    if bool(recipe.geometry.enabled):
        delta += float(np.clip(recipe.geometry.geometry_anchor_mix, 0.0, 1.0)) * np.asarray(
            state["geom_signal"], dtype=np.float64
        )
    robust_strength = float(np.clip(recipe.weight.robustness_strength, 0.0, 1.0))
    timing_strength = float(np.clip(recipe.weight.path_quality_strength, 0.0, 1.0))
    delta += timing_strength * (0.75 * quick_profit - timing_penalty)
    delta -= robust_strength * uncertainty
    delta -= (1.0 - float(np.clip(recipe.weight.ambiguous_weight, 0.0, 1.0))) * ambiguity
    delta -= float(p.stop_penalty) * actual_stop.astype(np.float64)
    delta -= 0.50 * float(p.stop_penalty) * (bad_path & (~actual_stop)).astype(np.float64)

    candidate_logit = _safe_logit(base[:n]) + label_strength * np.clip(delta, -8.0, 8.0)
    soft = _sigmoid(candidate_logit)
    soft = np.where(
        is_timeout,
        (1.0 - label_strength) * soft + label_strength * float(recipe.geometry.timeout_value),
        soft,
    )
    stop_cap = (1.0 - label_strength) * 0.98 + label_strength * float(
        np.clip(p.max_stop_soft_label, 0.02, 0.98)
    )
    bad_path_cap = (1.0 - label_strength) * 0.98 + label_strength * float(
        np.clip(p.max_bad_path_soft_label, 0.02, 0.98)
    )
    soft = np.where(
        actual_stop,
        np.minimum(soft, stop_cap),
        soft,
    )
    soft = np.where(
        bad_path & (~actual_stop),
        np.minimum(soft, bad_path_cap),
        soft,
    )
    soft = np.clip(np.nan_to_num(soft, nan=0.5), 0.02, 0.98).astype(np.float32)
    if stage_l == "train_meta" and "tbm" in label_l:
        soft_std = float(np.std(soft)) if len(soft) else 0.0
        y_std = float(np.std(y_ref[:n])) if n else 0.0
        if y_std > 1e-6 and soft_std < 1e-6:
            raise RuntimeError(
                "Meta TBM label/weight recipe produced a constant soft target "
                f"(soft_std={soft_std:.6g}, hard_std={y_std:.6g}, "
                f"actual_stop_rate={float(np.mean(actual_stop)) if n else 0.0:.4f}, "
                f"geometry_hard_positive_rate={float(geom_stats.get('hard_positive_rate', np.nan)):.4f}, "
                f"label={label!r})."
            )
    out = np.asarray(current_soft, dtype=np.float32).copy()
    out[:n] = soft
    stats = {
        "enabled": True,
        "recipe": recipe.name,
        "label": str(label),
        "stage": str(stage),
        "modifier_mode": "residual",
        "refinement_mode": "path_economics_logit_delta",
        "label_modifier_strength": label_strength,
        "delta_mean": float(np.mean(delta)) if len(delta) else float("nan"),
        "delta_std": float(np.std(delta)) if len(delta) else float("nan"),
        "soft_mean": float(np.mean(soft)) if len(soft) else float("nan"),
        "soft_std": float(np.std(soft)) if len(soft) else float("nan"),
        "net_bps_mean": float(np.mean(state["net_edge_bps"])) if len(state["net_edge_bps"]) else float("nan"),
        "edge_lcb_bps_mean": float(np.mean(state["edge_lcb_bps"])) if len(state["edge_lcb_bps"]) else float("nan"),
        "execution_cost_bps": float(state["costs_bps"]),
        "ambiguity_mean": float(np.mean(ambiguity)) if len(ambiguity) else float("nan"),
        "near_barrier_mean": float(np.mean(state["near_barrier"])) if len(state["near_barrier"]) else float("nan"),
        "robustness_strength": robust_strength,
        "uncertainty_mean": float(np.mean(uncertainty)) if len(uncertainty) else float("nan"),
        "path_quality_strength": timing_strength,
        "path_timing_penalty_mean": float(np.mean(timing_penalty)) if len(timing_penalty) else float("nan"),
        "quick_mfe_profit_mean": float(np.mean(quick_profit)) if len(quick_profit) else float("nan"),
        "geometry_enabled": bool(geom_stats.get("enabled", False)),
        "geometry_hard_positive_rate": float(geom_stats.get("hard_positive_rate", np.nan)),
        "geometry_soft_anchor_mean": float(geom_stats.get("soft_anchor_mean", np.nan)),
        "geometry_timeout_rate": float(geom_stats.get("timeout_rate", np.nan)),
        "geometry_stop_before_tp_rate": float(geom_stats.get("stop_before_tp_rate", np.nan)),
        "actual_stop_rate": float(np.mean(actual_stop)) if n else 0.0,
        "bad_path_rate": float(np.mean(bad_path)) if n else 0.0,
        "max_stop_soft_label": float(p.max_stop_soft_label),
        "max_bad_path_soft_label": float(p.max_bad_path_soft_label),
    }
    return out.astype(np.float32, copy=False), stats

def apply_weight_recipe(
    df: pd.DataFrame,
    y_hard: np.ndarray,
    y_soft: np.ndarray,
    current_weight: np.ndarray,
    *,
    cfg: dict[str, Any] | None = None,
    stage: str,
    label: str,
    fit_indices: Any = None,
    fit_mask: Any = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    recipe = load_recipe_from_env_or_cfg(cfg, scope=_scope_from_stage(stage))
    if not recipe_applies(recipe, stage):
        return np.asarray(current_weight, dtype=np.float32), {"enabled": False, "reason": "no_recipe"}
    assert recipe is not None
    w0 = np.asarray(current_weight, dtype=np.float64).reshape(-1)
    ys = np.clip(np.asarray(y_soft, dtype=np.float64).reshape(-1), 0.0, 1.0)
    yh = np.asarray(y_hard, dtype=np.float64).reshape(-1) >= 0.5
    n = _require_equal_lengths("apply_weight_recipe", df=df, current_weight=w0, y_soft=ys, y_hard=yh)
    mask = _fit_mask_from_indices(n, fit_indices=fit_indices, fit_mask=fit_mask)
    p = recipe.weight
    weight_strength = float(np.clip(p.weight_modifier_strength, 0.0, 1.0))
    utility_tail_strength = float(
        np.clip(getattr(p, "utility_tail_rank_strength", 0.0), 0.0, 1.0)
    )
    timestamp_balance_strength = float(
        np.clip(getattr(p, "timestamp_balance_strength", 0.0), 0.0, 1.0)
    )
    base_weight_power = float(np.clip(getattr(p, "base_weight_power", 1.0), 0.0, 1.0))
    if (
        weight_strength <= 0.0
        and utility_tail_strength <= 0.0
        and timestamp_balance_strength <= 0.0
        and abs(base_weight_power - 1.0) <= 1e-12
    ):
        return np.asarray(current_weight, dtype=np.float32), {
            "enabled": True,
            "recipe": recipe.name,
            "label": str(label),
            "stage": str(stage),
            "modifier_mode": "residual",
            "base_weight_power": 1.0,
            "weight_modifier_strength": 0.0,
            "utility_tail_rank_strength": 0.0,
            "timestamp_balance_strength": 0.0,
            "reason": "zero_weight_modifier_strength",
            "fit_rows": int(np.sum(mask)),
        }
    df_local = df.reset_index(drop=True)
    base_reference = np.power(
        np.clip(np.nan_to_num(w0, nan=1.0, posinf=1.0, neginf=1.0), 1e-6, None),
        base_weight_power,
    )
    out = base_reference.copy()
    geom_stats: dict[str, Any] = {}
    robust_strength = float(np.clip(p.robustness_strength, 0.0, 1.0))
    timing_strength = float(np.clip(p.path_quality_strength, 0.0, 1.0))
    ambiguity = np.zeros(n, dtype=np.float64)
    uncertainty = np.zeros(n, dtype=np.float64)
    timing_penalty = np.zeros(n, dtype=np.float64)
    quick_profit = np.zeros(n, dtype=np.float64)
    near_barrier = np.full(n, np.nan, dtype=np.float64)
    edge_lcb_bps = np.full(n, np.nan, dtype=np.float64)
    log_delta = np.zeros(n, dtype=np.float64)
    if weight_strength > 0.0:
        state = _path_economics_state(
            df_local,
            n=n,
            y_hard=yh.astype(np.float64),
            y_soft=ys,
            recipe=recipe,
            cfg=cfg,
            fit_mask=mask,
        )
        geom_stats = dict(state["geom_stats"])
        if bool(recipe.geometry.enabled):
            yh = np.asarray(state["y_geom"], dtype=np.float64) >= 0.5
        ambiguity = np.asarray(state["ambiguity"], dtype=np.float64)
        uncertainty = np.asarray(state["uncertainty"], dtype=np.float64)
        label_stability = np.asarray(state["label_stability"], dtype=np.float64)
        timing_penalty = np.asarray(state["timing_penalty"], dtype=np.float64)
        quick_profit = np.asarray(state["quick_profit"], dtype=np.float64)
        near_barrier = np.asarray(state["near_barrier"], dtype=np.float64)
        edge_lcb_bps = np.asarray(state["edge_lcb_bps"], dtype=np.float64)
        mfe_ratio = np.clip(
            np.asarray(state["mfe"], dtype=np.float64)
            / np.maximum(state["vol"], 1e-8),
            0.0,
            10.0,
        )
        mae_ratio = np.clip(
            np.asarray(state["mae_abs"], dtype=np.float64)
            / np.maximum(state["vol"], 1e-8),
            0.0,
            10.0,
        )
        edge_budget = np.clip(np.maximum(edge_lcb_bps, 0.0) / 100.0, 0.0, 10.0)
        hard_negative_signal = (~np.asarray(yh[:n], dtype=bool)).astype(np.float64) * label_stability
        log_delta += float(p.net_ev_weight_power) * np.log1p(edge_budget)
        log_delta += float(p.mfe_weight_power) * np.log1p(mfe_ratio)
        log_delta -= float(p.mae_weight_power) * np.log1p(mae_ratio)
        log_delta -= (1.0 - float(np.clip(p.ambiguous_weight, 0.0, 1.0))) * ambiguity
        log_delta -= robust_strength * uncertainty
        log_delta += timing_strength * (0.75 * quick_profit - timing_penalty)
        log_delta += np.log(max(float(p.hard_negative_weight), 1e-6)) * hard_negative_signal
        out[:n] = out[:n] * np.exp(weight_strength * np.clip(log_delta, -4.0, 4.0))

    utility_tail_multiplier = np.ones(n, dtype=np.float64)
    proxy_multiplier = np.ones(n, dtype=np.float64)
    utility_tail_source = "none"
    if utility_tail_strength > 0.0:
        if "__u_policy_net__" in df_local.columns:
            utility_raw = pd.to_numeric(
                df_local["__u_policy_net__"].iloc[:n],
                errors="coerce",
            ).to_numpy(dtype=np.float64)
            utility_tail_source = "__u_policy_net__"
        elif "u_policy_net" in df_local.columns:
            utility_raw = pd.to_numeric(
                df_local["u_policy_net"].iloc[:n],
                errors="coerce",
            ).to_numpy(dtype=np.float64)
            utility_tail_source = "u_policy_net"
        else:
            utility_raw = ys[:n].astype(np.float64, copy=False)
            utility_tail_source = "y_soft"
        valid_ref = mask[:n] & np.isfinite(utility_raw)
        if bool(np.any(valid_ref)):
            sorted_ref = np.sort(utility_raw[valid_ref])
            rank_pct = np.searchsorted(sorted_ref, utility_raw, side="right").astype(np.float64)
            rank_pct /= max(float(len(sorted_ref)), 1.0)
            rank_pct = np.clip(np.nan_to_num(rank_pct, nan=0.0), 0.0, 1.0)
            tail_power = max(float(getattr(p, "utility_tail_rank_power", 4.0)), 0.0)
            tail_base = max(float(getattr(p, "utility_tail_rank_base", 0.50)), 1e-6)
            tail_scale = max(float(getattr(p, "utility_tail_rank_scale", 4.0)), 0.0)
            raw_tail = tail_base + tail_scale * np.power(rank_pct, tail_power)
            utility_tail_multiplier = _normalize_weights(raw_tail, fit_mask=mask[:n])
            utility_tail_multiplier = np.clip(utility_tail_multiplier, 0.10, 5.0)
            proxy_multiplier *= np.power(utility_tail_multiplier, utility_tail_strength)

    timestamp_multiplier = np.ones(n, dtype=np.float64)
    if timestamp_balance_strength > 0.0 and "__ts__" in df_local.columns:
        ts = pd.to_datetime(df_local["__ts__"].iloc[:n], utc=True, errors="coerce")
        ref_ts = ts[mask[:n]]
        ref_counts = ref_ts.value_counts(dropna=False)
        if len(ref_counts):
            median_count = max(float(np.nanmedian(ref_counts.to_numpy(dtype=np.float64))), 1.0)
            counts = ts.map(ref_counts).fillna(median_count).to_numpy(dtype=np.float64)
            counts = np.clip(np.nan_to_num(counts, nan=median_count), 1.0, None)
            raw_ts = 1.0 / counts
            timestamp_multiplier = _normalize_weights(raw_ts, fit_mask=mask[:n])
            timestamp_multiplier = np.clip(timestamp_multiplier, 0.10, 5.0)
            proxy_multiplier *= np.power(timestamp_multiplier, timestamp_balance_strength)

    if utility_tail_strength > 0.0 or timestamp_balance_strength > 0.0:
        proxy_multiplier = _normalize_weights(proxy_multiplier, fit_mask=mask[:n])
        proxy_multiplier = np.clip(proxy_multiplier, 0.10, 5.0)
        out[:n] *= proxy_multiplier

    portfolio_strength = float(np.clip(recipe.objective.portfolio_alignment_strength, 0.0, 1.0))
    if portfolio_strength > 0.0:
        out[:n] *= _group_balance_multiplier(
            df_local,
            n=n,
            fit_mask=mask,
            strength=portfolio_strength,
        )

    pos_mass = float(np.sum(out[:n][mask] * ys[:n][mask]))
    neg_mass = float(np.sum(out[:n][mask] * (1.0 - ys[:n][mask])))
    frac = pos_mass / max(pos_mass + neg_mass, 1e-12)
    target = float(np.clip(p.positive_mass_target, 0.05, 0.95))
    strength = float(np.clip(p.class_rebalance_strength, 0.0, 1.0))
    if strength > 0.0 and pos_mass > 1e-12 and neg_mass > 1e-12:
        pos_factor = (target * (pos_mass + neg_mass)) / pos_mass
        neg_factor = ((1.0 - target) * (pos_mass + neg_mass)) / neg_mass
        row_factor = ys[:n] * pos_factor + (1.0 - ys[:n]) * neg_factor
        out[:n] *= 1.0 + strength * (row_factor - 1.0)

    if p.concurrency_penalty > 0.0 and "__ts__" in df.columns:
        window_hours = max(float(p.concurrency_window_hours), 1e-6)
        freq_seconds = max(1, int(round(window_hours * 3600.0)))
        ts = pd.to_datetime(df["__ts__"].iloc[:n], utc=True, errors="coerce").dt.floor(f"{freq_seconds}s")
        counts = ts.map(ts.value_counts()).to_numpy(dtype=np.float64)
        counts = np.clip(counts, 1.0, None)
        ref_counts = counts[mask]
        ref_median = float(np.nanmedian(ref_counts)) if len(ref_counts) else 1.0
        out[:n] *= np.power(counts / max(ref_median, 1.0), -float(p.concurrency_penalty))

    recency_hpo_weight_active = False
    if "__ts__" in df.columns:
        recency_hpo_decay, _recency_hpo_active = recency_hpo_decay_from_config(
            df["__ts__"].iloc[:n],
            n,
            cfg=cfg,
            objective_mode=stage,
        )
        recency_hpo_weight_active = recency_hpo_decay is not None
    if (
        "__ts__" in df.columns
        and float(p.recency_half_life_days) > 0.0
        and not recency_hpo_weight_active
    ):
        ts = pd.to_datetime(df["__ts__"].iloc[:n], utc=True, errors="coerce")
        ref_ts = ts[mask]
        if ref_ts.notna().any():
            latest = ref_ts.max()
            age_days = (latest - ts).dt.total_seconds().to_numpy(dtype=np.float64) / 86400.0
            recency = np.power(0.5, np.maximum(age_days, 0.0) / max(float(p.recency_half_life_days), 1e-6))
            out[:n] *= np.clip(np.nan_to_num(recency, nan=1.0), 1e-6, 1.0)

    out = _normalize_weights_to_reference(out, reference=base_reference, fit_mask=mask)
    edge_lcb_fit = edge_lcb_bps[mask[:n]]
    near_barrier_fit = near_barrier[mask[:n]]
    edge_lcb_mean = (
        float(np.nanmean(edge_lcb_fit))
        if np.isfinite(edge_lcb_fit).any()
        else float("nan")
    )
    near_barrier_mean = (
        float(np.nanmean(near_barrier_fit))
        if np.isfinite(near_barrier_fit).any()
        else float("nan")
    )
    stats = {
        "enabled": True,
        "recipe": recipe.name,
        "label": str(label),
        "stage": str(stage),
        "modifier_mode": "residual",
        "refinement_mode": "path_economics_log_budget",
        "base_weight_power": base_weight_power,
        "weight_modifier_strength": weight_strength,
        "utility_tail_rank_strength": utility_tail_strength,
        "utility_tail_rank_source": utility_tail_source,
        "utility_tail_rank_power": float(getattr(p, "utility_tail_rank_power", 4.0)),
        "utility_tail_multiplier_p95": float(np.percentile(utility_tail_multiplier[mask[:n]], 95))
        if np.any(mask[:n])
        else float("nan"),
        "timestamp_balance_strength": timestamp_balance_strength,
        "timestamp_balance_multiplier_p95": float(np.percentile(timestamp_multiplier[mask[:n]], 95))
        if np.any(mask[:n])
        else float("nan"),
        "proxy_multiplier_p95": float(np.percentile(proxy_multiplier[mask[:n]], 95))
        if np.any(mask[:n])
        else float("nan"),
        "proxy_multiplier_p99": float(np.percentile(proxy_multiplier[mask[:n]], 99))
        if np.any(mask[:n])
        else float("nan"),
        "positive_mass_before": frac,
        "positive_mass_target": target,
        "fit_rows": int(np.sum(mask)),
        "log_delta_mean": float(np.mean(log_delta[mask])) if np.any(mask) else float("nan"),
        "log_delta_std": float(np.std(log_delta[mask])) if np.any(mask) else float("nan"),
        "mean": float(np.mean(out[mask])) if np.any(mask) else float("nan"),
        "p95": float(np.percentile(out[mask], 95)) if np.any(mask) else float("nan"),
        "edge_lcb_bps_mean": edge_lcb_mean,
        "ambiguity_mean": float(np.mean(ambiguity[mask])) if np.any(mask) else float("nan"),
        "near_barrier_mean": near_barrier_mean,
        "robustness_strength": robust_strength,
        "uncertainty_mean": float(np.mean(uncertainty[mask])) if np.any(mask) else float("nan"),
        "path_quality_strength": timing_strength,
        "path_timing_penalty_mean": float(np.mean(timing_penalty[mask])) if np.any(mask) else float("nan"),
        "quick_mfe_profit_mean": float(np.mean(quick_profit[mask])) if np.any(mask) else float("nan"),
        "portfolio_alignment_strength": portfolio_strength,
        "geometry_enabled": bool(geom_stats.get("enabled", False)),
        "geometry_hard_positive_rate": float(geom_stats.get("hard_positive_rate", np.nan)),
        "geometry_timeout_rate": float(geom_stats.get("timeout_rate", np.nan)),
        "geometry_stop_before_tp_rate": float(geom_stats.get("stop_before_tp_rate", np.nan)),
        "legacy_recency_disabled_by_recency_hpo": bool(recency_hpo_weight_active),
    }
    return out.astype(np.float32, copy=False), stats


def apply_distillation_recipe(
    distill: np.ndarray,
    fp_weight: np.ndarray,
    *,
    y_metric: np.ndarray,
    pred: np.ndarray,
    returns: Any = None,
    timestamps: Any = None,
    objective_mode: str | None = None,
    cfg: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    recipe = load_recipe_from_env_or_cfg(cfg, scope=_scope_from_stage(objective_mode))
    if not recipe_applies(recipe, str(objective_mode or "all")):
        return np.asarray(distill, dtype=np.float32), np.asarray(fp_weight, dtype=np.float32)
    assert recipe is not None
    d = recipe.distillation
    y = np.asarray(y_metric, dtype=np.float64)
    p = np.asarray(pred, dtype=np.float64)
    pred_clip = np.clip(np.nan_to_num(p, nan=0.5), 0.0, 1.0)
    y_clip = np.clip(np.nan_to_num(y, nan=0.5), 0.0, 1.0)
    prob_error = np.abs(y_clip - pred_clip)
    ret_bps = np.zeros_like(pred_clip, dtype=np.float64)
    if returns is not None:
        ret_arr = np.asarray(returns, dtype=np.float64).reshape(-1)
        if len(ret_arr) != len(pred_clip):
            raise ValueError(f"returns length {len(ret_arr)} != pred length {len(pred_clip)}")
        ret_bps = 10_000.0 * np.nan_to_num(ret_arr, nan=0.0, posinf=0.0, neginf=0.0)
    costs_bps = fixed_execution_cost_bps(cfg, recipe)
    executable_net_bps = ret_bps - costs_bps
    utility_center = float(recipe.geometry.min_executable_net_bps)
    utility_scale = max(float(recipe.objective.impact_scale_bps), 1e-6)
    economic_target = _sigmoid((executable_net_bps - utility_center) / utility_scale)
    economic_error = np.abs(economic_target - pred_clip)
    econ_mix = float(np.clip(d.economic_error_mix, 0.0, 1.0))
    err = (1.0 - econ_mix) * prob_error + econ_mix * economic_error
    err_rank = pd.Series(np.nan_to_num(err, nan=0.0)).rank(pct=True).to_numpy(dtype=np.float64)
    pred_rank = pd.Series(np.nan_to_num(pred_clip, nan=0.5)).rank(pct=True).to_numpy(dtype=np.float64)
    target_rank = pd.Series(np.nan_to_num(economic_target, nan=0.5)).rank(pct=True).to_numpy(dtype=np.float64)
    rank_temp = max(float(d.distill_rank_focus_temperature), 1e-6)
    rank_threshold = float(np.clip(d.distill_rank_focus_threshold, 0.0, 1.0))
    high_pred_focus = _sigmoid((pred_rank - rank_threshold) / rank_temp)
    missed_target_focus = _sigmoid((target_rank - rank_threshold) / rank_temp)
    dist = np.asarray(distill, dtype=np.float64) * np.power(0.5 + err_rank, float(d.distill_error_power))
    fp = np.asarray(fp_weight, dtype=np.float64)
    loss_severity = np.power(
        1.0 + np.clip(np.maximum(-executable_net_bps, 0.0) / utility_scale, 0.0, 10.0),
        float(d.distill_net_loss_power),
    )
    missed_upside = np.power(
        1.0 + np.clip(np.maximum(executable_net_bps - utility_center, 0.0) / utility_scale, 0.0, 10.0),
        float(d.distill_missed_net_power),
    )
    severe_loss = _sigmoid((-executable_net_bps - utility_scale) / max(0.50 * utility_scale, 1e-6))
    fp_signal = high_pred_focus * np.clip(loss_severity - 1.0, 0.0, None)
    fn_signal = (1.0 - pred_rank) * missed_target_focus * np.clip(missed_upside - 1.0, 0.0, None)
    if returns is None:
        fp_mask = (pred_rank >= 0.80) & (y_clip < 0.5)
        fn_mask = (pred_rank <= 0.20) & (y_clip >= 0.5)
        fp_signal = fp_signal + fp_mask.astype(np.float64)
        fn_signal = fn_signal + fn_mask.astype(np.float64)
    fp *= 1.0 + float(d.false_positive_focus) * np.clip(fp_signal, 0.0, 10.0)
    fp *= 1.0 + float(d.false_negative_focus) * np.clip(fn_signal, 0.0, 10.0)
    fp *= 1.0 + float(d.distill_stop_hit_focus) * high_pred_focus * severe_loss
    recency_hpo_decay = None
    if timestamps is not None:
        recency_hpo_decay, _recency_hpo_active = recency_hpo_decay_from_config(
            timestamps,
            len(pred_clip),
            cfg=cfg,
            objective_mode=objective_mode,
        )
    if recency_hpo_decay is not None:
        strength = np.clip(
            np.asarray(recency_hpo_decay, dtype=np.float64).reshape(-1),
            0.0,
            1.0,
        )
        if len(strength) == len(dist):
            dist = 1.0 + (dist - 1.0) * strength
            fp = 1.0 + (fp - 1.0) * strength
    elif timestamps is not None and float(d.distill_age_impact) > 0.0:
        ts = pd.to_datetime(pd.Series(timestamps), utc=True, errors="coerce")
        if ts.notna().any():
            latest = ts.max()
            age_days = (latest - ts).dt.total_seconds().to_numpy(dtype=np.float64) / 86400.0
            half_life = (
                recipe.weight.recency_half_life_days * 0.5
                if str(objective_mode or "").strip().lower() == "train_meta"
                else recipe.weight.recency_half_life_days
            )
            age_decay = np.power(0.5, np.maximum(age_days, 0.0) / max(float(half_life), 1e-6))
            strength = np.clip(age_decay, 0.0, 1.0) ** float(d.distill_age_impact)
            dist = 1.0 + (dist - 1.0) * strength
            fp = 1.0 + (fp - 1.0) * strength
    distill_strength = float(np.clip(d.distillation_strength, 0.0, 1.0))
    if distill_strength <= 0.0:
        return np.asarray(distill, dtype=np.float32), np.asarray(fp_weight, dtype=np.float32)
    if distill_strength < 1.0:
        dist = 1.0 + (dist - 1.0) * distill_strength
        fp = 1.0 + (fp - 1.0) * distill_strength
    return _normalize_weights(dist), _normalize_weights(fp)


def suggest_optuna_params(
    trial: Any,
    *,
    phase: str,
    layer: str = LAYER_ALL,
    base_recipe: LabelWeightRecipe | None = None,
) -> LabelWeightRecipe:
    """Build a recipe from an Optuna trial.

    Phases are intended to be run successively:
    label_geometry -> labels -> weights -> distillation.
    Later phases can enqueue/fix the best params from prior phases.
    """
    phase = str(phase).strip().lower()
    if int(getattr(trial, "number", 0)) == 0:
        recipe_enabled = trial.suggest_categorical("recipe_enabled", [True, False])
    else:
        recipe_enabled = True
    if not bool(recipe_enabled):
        return _noop_recipe_for_phase(
            phase=phase,
            base_recipe=base_recipe,
            name=f"trial_{getattr(trial, 'number', 'unknown')}",
        )
    active_layer = _resolve_optuna_layer(trial, phase=phase, layer=layer)
    if base_recipe is not None:
        recipe = LabelWeightRecipe.from_dict(base_recipe.to_dict())
        recipe.name = f"trial_{getattr(trial, 'number', 'unknown')}"
        recipe.stage = "all"
    else:
        recipe = LabelWeightRecipe(name=f"trial_{getattr(trial, 'number', 'unknown')}", stage="all")
    if phase in {"label_geometry", "geometry", "hard_labels", "all"}:
        current = recipe.geometry
        geometry_layer_active = _layer_active(active_layer, *GEOMETRY_LAYERS)
        recipe.geometry = LabelGeometryParams(
            enabled=True if geometry_layer_active else current.enabled,
            tp_vol_mult=trial.suggest_float("tp_vol_mult", 0.35, 2.50, log=True)
            if _layer_active(active_layer, "barrier_feasibility")
            else current.tp_vol_mult,
            sl_as_tp_pct=trial.suggest_float("sl_as_tp_pct", 0.35, 1.50)
            if _layer_active(active_layer, "barrier_feasibility")
            else current.sl_as_tp_pct,
            label_horizon_bars=trial.suggest_categorical(
                "label_horizon_bars",
                [2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0],
            )
            if _layer_active(active_layer, "barrier_feasibility")
            else current.label_horizon_bars,
            timeout_value=trial.suggest_float("timeout_value", 0.10, 0.60)
            if _layer_active(active_layer, "barrier_feasibility")
            else current.timeout_value,
            trailing_activation_vol_mult=trial.suggest_float(
                "trailing_activation_vol_mult",
                0.50,
                3.00,
                log=True,
            )
            if _layer_active(active_layer, "barrier_feasibility")
            else current.trailing_activation_vol_mult,
            trailing_giveback_pct=trial.suggest_float("trailing_giveback_pct", 0.20, 0.90)
            if _layer_active(active_layer, "barrier_feasibility")
            else current.trailing_giveback_pct,
            min_executable_net_bps=trial.suggest_float("min_executable_net_bps", 0.0, 120.0)
            if _layer_active(active_layer, "path_quality")
            else current.min_executable_net_bps,
            mae_failure_vol_mult=trial.suggest_float("mae_failure_vol_mult", 0.35, 2.50, log=True)
            if _layer_active(active_layer, "path_quality")
            else current.mae_failure_vol_mult,
            geometry_anchor_mix=trial.suggest_float("geometry_anchor_mix", 0.20, 0.85)
            if _layer_active(active_layer, "path_quality")
            else current.geometry_anchor_mix,
        )
        if geometry_layer_active:
            current_gen = recipe.generator
            if _truthy_env_or_cfg("EPM_LABEL_WEIGHT_HPO_POLICY_GENERATOR", {}, default=False):
                recipe.generator = GeneratorParams(
                    **{
                        **asdict(current_gen),
                        "policy_label_sl_atr_mult": trial.suggest_float(
                            "policy_label_sl_atr_mult", 0.65, 2.25
                        ),
                        "policy_label_tp_sl_ratio": trial.suggest_float(
                            "policy_label_tp_sl_ratio", 1.0, 3.5
                        ),
                        "policy_label_trailing_pct": trial.suggest_float(
                            "policy_label_trailing_pct", 0.10, 0.75
                        ),
                        "policy_label_max_hold_hours": trial.suggest_categorical(
                            "policy_label_max_hold_hours",
                            [8.0, 12.0, 16.0, 24.0, 36.0],
                        ),
                    }
                )
    if phase in {"labels", "all"}:
        current = recipe.label
        label_layer_active = _layer_active(active_layer, *LABEL_LAYERS)
        label_residual_active = label_layer_active or float(current.label_modifier_strength) > 0.0
        if label_residual_active:
            recipe.label = LabelParams(
                label_modifier_strength=trial.suggest_float("label_modifier_strength", 0.0, 0.80)
                if label_layer_active
                else current.label_modifier_strength,
                mfe_scale_bps=trial.suggest_float("mfe_scale_bps", 20.0, 300.0, log=True)
                if _layer_active(active_layer, "net_edge_mapping")
                else current.mfe_scale_bps,
                mae_penalty_scale=trial.suggest_float("mae_penalty_scale", 1.0, 8.0, log=True)
                if _layer_active(active_layer, "risk_path_caps")
                else current.mae_penalty_scale,
                net_return_center_bps=trial.suggest_float("net_return_center_bps", 0.0, 180.0)
                if _layer_active(active_layer, "net_edge_mapping")
                else current.net_return_center_bps,
                net_return_temperature_bps=trial.suggest_float("net_return_temperature_bps", 20.0, 220.0, log=True)
                if _layer_active(active_layer, "net_edge_mapping")
                else current.net_return_temperature_bps,
                stop_penalty=trial.suggest_float("stop_penalty", 0.0, 3.00)
                if _layer_active(active_layer, "risk_path_caps")
                else current.stop_penalty,
                path_quality_mix=trial.suggest_float("path_quality_mix", 0.0, 1.0)
                if _layer_active(active_layer, "risk_path_caps")
                else current.path_quality_mix,
                max_stop_soft_label=trial.suggest_float("max_stop_soft_label", 0.05, 0.35)
                if _layer_active(active_layer, "risk_path_caps")
                else current.max_stop_soft_label,
                max_bad_path_soft_label=trial.suggest_float("max_bad_path_soft_label", 0.20, 0.50)
                if _layer_active(active_layer, "risk_path_caps")
                else current.max_bad_path_soft_label,
            )
        if label_layer_active:
            current_gen = recipe.generator
            recipe.generator = GeneratorParams(
                **{
                    **asdict(current_gen),
                    "lgbm_soft_label_costs": trial.suggest_float(
                        "lgbm_soft_label_costs", 0.0, 0.010
                    )
                    if _layer_active(active_layer, "net_edge_mapping")
                    else current_gen.lgbm_soft_label_costs,
                    "lgbm_soft_label_min_opportunity_mult": trial.suggest_float(
                        "lgbm_soft_label_min_opportunity_mult", 0.05, 0.80, log=True
                    )
                    if _layer_active(active_layer, "net_edge_mapping")
                    else current_gen.lgbm_soft_label_min_opportunity_mult,
                    "lgbm_soft_label_temperature": trial.suggest_float(
                        "lgbm_soft_label_temperature", 0.05, 0.60, log=True
                    )
                    if _layer_active(active_layer, "net_edge_mapping")
                    else current_gen.lgbm_soft_label_temperature,
                    "net_executable_mae_lambda": trial.suggest_float(
                        "net_executable_mae_lambda", 0.05, 1.25
                    )
                    if _layer_active(active_layer, "risk_path_caps")
                    else current_gen.net_executable_mae_lambda,
                    "net_executable_center_vol": trial.suggest_float(
                        "net_executable_center_vol", -0.40, 0.80
                    )
                    if _layer_active(active_layer, "risk_path_caps")
                    else current_gen.net_executable_center_vol,
                    "net_executable_temperature_vol": trial.suggest_float(
                        "net_executable_temperature_vol", 0.08, 1.20, log=True
                    )
                    if _layer_active(active_layer, "risk_path_caps")
                    else current_gen.net_executable_temperature_vol,
                    "policy_net_label_center": trial.suggest_float(
                        "policy_net_label_center", -0.010, 0.010
                    )
                    if _layer_active(active_layer, "net_edge_mapping")
                    else current_gen.policy_net_label_center,
                    "policy_net_label_temperature": trial.suggest_float(
                        "policy_net_label_temperature", 0.001, 0.030, log=True
                    )
                    if _layer_active(active_layer, "net_edge_mapping")
                    else current_gen.policy_net_label_temperature,
                }
            )
    if phase in {"weights", "all"}:
        current = recipe.weight
        weight_layer_active = _layer_active(active_layer, *WEIGHT_LAYERS)
        weight_residual_active = weight_layer_active or float(current.weight_modifier_strength) > 0.0
        if weight_residual_active:
            half_life = (
                trial.suggest_categorical("recency_half_life_days", [0.0, 90.0, 150.0, 300.0, 450.0])
                if _layer_active(active_layer, "robustness_diversification")
                else current.recency_half_life_days
            )
            recipe.weight = WeightParams(
                weight_modifier_strength=trial.suggest_float("weight_modifier_strength", 0.0, 0.65)
                if weight_layer_active
                else current.weight_modifier_strength,
                positive_mass_target=trial.suggest_float("positive_mass_target", 0.20, 0.50)
                if _layer_active(active_layer, "class_balance")
                else current.positive_mass_target,
                class_rebalance_strength=trial.suggest_float("class_rebalance_strength", 0.0, 1.0)
                if _layer_active(active_layer, "class_balance")
                else current.class_rebalance_strength,
                mfe_weight_power=trial.suggest_float("mfe_weight_power", 0.0, 3.0)
                if _layer_active(active_layer, "economic_emphasis")
                else current.mfe_weight_power,
                mae_weight_power=trial.suggest_float("mae_weight_power", 0.0, 3.0)
                if _layer_active(active_layer, "economic_emphasis")
                else current.mae_weight_power,
                net_ev_weight_power=trial.suggest_float("net_ev_weight_power", 0.0, 3.0)
                if _layer_active(active_layer, "economic_emphasis")
                else current.net_ev_weight_power,
                hard_negative_weight=trial.suggest_float("hard_negative_weight", 1.0, 5.0, log=True)
                if _layer_active(active_layer, "economic_emphasis")
                else current.hard_negative_weight,
                ambiguous_weight=trial.suggest_float("ambiguous_weight", 0.10, 1.0)
                if _layer_active(active_layer, "robustness_diversification")
                else current.ambiguous_weight,
                recency_half_life_days=half_life,
                concurrency_penalty=trial.suggest_float("concurrency_penalty", 0.0, 1.0)
                if _layer_active(active_layer, "robustness_diversification")
                else current.concurrency_penalty,
                concurrency_window_hours=trial.suggest_categorical(
                    "concurrency_window_hours",
                    [0.5, 1.0, 2.0, 4.0],
                )
                if _layer_active(active_layer, "robustness_diversification")
                else current.concurrency_window_hours,
                robustness_strength=trial.suggest_float("robustness_strength", 0.0, 1.0)
                if _layer_active(active_layer, "robustness_diversification")
                else current.robustness_strength,
                path_quality_strength=trial.suggest_float("path_quality_strength", 0.0, 1.0)
                if _layer_active(active_layer, "robustness_diversification")
                else current.path_quality_strength,
            )
        if weight_layer_active:
            current_gen = recipe.generator
            clip_min = (
                trial.suggest_float("outcome_weight_clip_min", 0.10, 0.90)
                if _layer_active(active_layer, "class_balance")
                else current_gen.outcome_weight_clip_min
            )
            clip_max_low = max(float(clip_min or 0.10), 1.0)
            recipe.generator = GeneratorParams(
                **{
                    **asdict(current_gen),
                    "timeout_weight": trial.suggest_float("timeout_weight", 0.05, 0.90)
                    if _layer_active(active_layer, "class_balance")
                    else current_gen.timeout_weight,
                    "outcome_weight_clip_min": clip_min,
                    "outcome_weight_clip_max": trial.suggest_float(
                        "outcome_weight_clip_max", clip_max_low, 3.00
                    )
                    if _layer_active(active_layer, "class_balance")
                    else current_gen.outcome_weight_clip_max,
                    "mfe_mae_w_min": trial.suggest_float("mfe_mae_w_min", 0.10, 0.85)
                    if _layer_active(active_layer, "economic_emphasis")
                    else current_gen.mfe_mae_w_min,
                    "mfe_mae_tau": trial.suggest_float("mfe_mae_tau", 0.25, 3.00, log=True)
                    if _layer_active(active_layer, "economic_emphasis")
                    else current_gen.mfe_mae_tau,
                    "mfe_mae_cost_floor": trial.suggest_float(
                        "mfe_mae_cost_floor", 0.0001, 0.005, log=True
                    )
                    if _layer_active(active_layer, "economic_emphasis")
                    else current_gen.mfe_mae_cost_floor,
                    "meta_weight_sigmoid_alpha": trial.suggest_float(
                        "meta_weight_sigmoid_alpha", 0.0, 0.80
                    )
                    if _layer_active(active_layer, "economic_emphasis")
                    else current_gen.meta_weight_sigmoid_alpha,
                    "meta_mfe_mae_tau": trial.suggest_float(
                        "meta_mfe_mae_tau", 0.25, 3.00, log=True
                    )
                    if _layer_active(active_layer, "economic_emphasis")
                    else current_gen.meta_mfe_mae_tau,
                }
            )
        recipe.objective.portfolio_alignment_strength = (
            trial.suggest_float(
                "portfolio_alignment_strength",
                0.0,
                0.5,
            )
            if _layer_active(active_layer, "portfolio_alignment")
            else recipe.objective.portfolio_alignment_strength
        )
    if phase in {"distillation", "all"}:
        current = recipe.distillation
        recipe.distillation = DistillationParams(
            distillation_strength=trial.suggest_float("distillation_strength", 0.0, 0.60)
            if _layer_active(active_layer, "error_refocus", "rank_tail_refocus")
            else current.distillation_strength,
            distill_error_power=trial.suggest_float("distill_error_power", 0.0, 3.0)
            if _layer_active(active_layer, "error_refocus")
            else current.distill_error_power,
            false_positive_focus=trial.suggest_float("false_positive_focus", 0.0, 3.0)
            if _layer_active(active_layer, "error_refocus")
            else current.false_positive_focus,
            false_negative_focus=trial.suggest_float("false_negative_focus", 0.0, 1.5)
            if _layer_active(active_layer, "error_refocus")
            else current.false_negative_focus,
            distill_age_impact=trial.suggest_float("distill_age_impact", 0.0, 2.0)
            if _layer_active(active_layer, "error_refocus")
            else current.distill_age_impact,
            economic_error_mix=trial.suggest_float("economic_error_mix", 0.0, 1.0)
            if _layer_active(active_layer, "error_refocus")
            else current.economic_error_mix,
            distill_net_loss_power=trial.suggest_float("distill_net_loss_power", 0.0, 4.0)
            if _layer_active(active_layer, "rank_tail_refocus")
            else current.distill_net_loss_power,
            distill_stop_hit_focus=trial.suggest_float("distill_stop_hit_focus", 0.0, 4.0)
            if _layer_active(active_layer, "rank_tail_refocus")
            else current.distill_stop_hit_focus,
            distill_missed_net_power=trial.suggest_float("distill_missed_net_power", 0.0, 2.0)
            if _layer_active(active_layer, "rank_tail_refocus")
            else current.distill_missed_net_power,
            distill_rank_focus_threshold=trial.suggest_float("distill_rank_focus_threshold", 0.60, 0.92)
            if _layer_active(active_layer, "rank_tail_refocus")
            else current.distill_rank_focus_threshold,
            distill_rank_focus_temperature=trial.suggest_float(
                "distill_rank_focus_temperature",
                0.02,
                0.20,
                log=True,
            )
            if _layer_active(active_layer, "rank_tail_refocus")
            else current.distill_rank_focus_temperature,
        )
    recipe.provenance = {
        "phase": phase,
        "layer": active_layer,
        "cv_mode": "interleaved_spread",
        "max_trees": 200,
        "max_trees_env": "EPM_LGBM_N_ESTIMATORS_CAP=200",
        "subsample": "spread_through_full_period",
        "features": "frozen_native_preset",
        "hyperparameters": "frozen_native_preset_except_n_estimators_cap",
    }
    return recipe


def _metric_first(metrics: dict[str, Any], *keys: str, default: float = 0.0) -> float:
    for key in keys:
        if key in metrics:
            try:
                value = float(metrics[key])
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                return value
    return float(default)


def _per_window_topk_values(metrics: dict[str, Any], *, k: int, key: str) -> list[float]:
    per_window = metrics.get("per_window")
    if not isinstance(per_window, dict):
        return []
    out: list[float] = []
    for window_metrics in per_window.values():
        if not isinstance(window_metrics, dict):
            continue
        topk = window_metrics.get(str(k), window_metrics.get(k))
        if not isinstance(topk, dict) or key not in topk:
            continue
        try:
            value = float(topk[key])
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            out.append(value)
    return out


def _edge_lcb(
    metrics: dict[str, Any],
    *,
    k: int,
    aggregate: float,
    se_divisor: float,
) -> float:
    values = _per_window_topk_values(metrics, k=k, key="mean_net_bps")
    if len(values) < 2:
        return float(aggregate)
    arr = np.asarray(values, dtype=np.float64)
    se = float(np.std(arr, ddof=1) / math.sqrt(len(arr)))
    return float(aggregate - se / max(float(se_divisor), 1e-6))


def _window_metric_lcb(
    metrics: dict[str, Any],
    *,
    k: int,
    key: str,
    aggregate: float,
    se_divisor: float,
) -> float:
    values = _per_window_topk_values(metrics, k=k, key=key)
    if len(values) < 2:
        return float(aggregate)
    arr = np.asarray(values, dtype=np.float64)
    se = float(np.std(arr, ddof=1) / math.sqrt(len(arr)))
    return float(aggregate - se / max(float(se_divisor), 1e-6))


def _stage_norm(model_stage: str) -> str:
    stage = str(model_stage or "base").strip().lower()
    return "meta" if "meta" in stage else "base"


def _baseline_metric(metrics: dict[str, Any], name: str) -> float | None:
    candidates = (
        f"baseline_{name}",
        f"neutral_{name}",
        f"incumbent_{name}",
        f"base_recipe_{name}",
    )
    for key in candidates:
        if key not in metrics:
            continue
        try:
            value = float(metrics[key])
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            return value
    baseline = metrics.get("baseline")
    if isinstance(baseline, dict) and name in baseline:
        try:
            value = float(baseline[name])
        except (TypeError, ValueError):
            return None
        return value if math.isfinite(value) else None
    neutral = metrics.get("neutral")
    if isinstance(neutral, dict) and name in neutral:
        try:
            value = float(neutral[name])
        except (TypeError, ValueError):
            return None
        return value if math.isfinite(value) else None
    return None


def _ranking_metric(metrics: dict[str, Any], *names: str, default: float = 0.0) -> float:
    return _metric_first(metrics, *names, default=default)


def _unit_interval(value: float, *, floor: float, good: float) -> float:
    try:
        val = float(value)
        lo = float(floor)
        hi = float(good)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(val) or abs(hi - lo) <= 1e-12:
        return 0.0
    return float(np.clip((val - lo) / (hi - lo), 0.0, 1.0))


def _weighted_economic_ic(metrics: dict[str, Any]) -> float:
    full = _ranking_metric(
        metrics,
        "economic_weighted_ic_full",
        "economic_rank_ic_full",
        "economic_rank_ic",
        "return_score_ic",
        "rank_ic",
        default=0.0,
    )
    top30 = _ranking_metric(
        metrics,
        "economic_weighted_ic_top30",
        "economic_rank_ic_top30",
        "economic_rank_ic",
        default=full,
    )
    top20 = _ranking_metric(
        metrics,
        "economic_weighted_ic_top20",
        "economic_rank_ic_top20",
        "economic_rank_ic",
        default=top30,
    )
    top10 = _ranking_metric(
        metrics,
        "economic_weighted_ic_top10",
        "economic_rank_ic_top10",
        "economic_rank_ic",
        default=top20,
    )
    weighted = (
        0.25 * full
        + 0.20 * top30
        + 0.15 * top20
        + 0.10 * top10
    ) / 0.70
    direct = _ranking_metric(metrics, "economic_weighted_ic", default=float("nan"))
    return float(direct) if math.isfinite(direct) else float(weighted)


def _lgbm_style_j(metrics: dict[str, Any], *, stage: str) -> float:
    stage_key = "J_meta" if _stage_norm(stage) == "meta" else "J_base"
    other_stage_key = "J_base" if stage_key == "J_meta" else "J_meta"
    direct = _metric_first(
        metrics,
        stage_key,
        f"{stage_key}_proxy",
        "J_final",
        "J_Score",
        "J",
        "lgbm_pipeline_j",
        "lgbm_pipeline_j_proxy",
        "J_proxy",
        other_stage_key,
        f"{other_stage_key}_proxy",
        default=float("nan"),
    )
    if math.isfinite(direct):
        return float(direct)
    rank_mono = _ranking_metric(
        metrics,
        "economic_rank_monotonicity",
        "rank_monotonicity",
        "return_bucket_monotonicity",
        default=0.5,
    )
    weighted_ic = _weighted_economic_ic(metrics)
    score_std = _ranking_metric(
        metrics,
        "prediction_score_std",
        "score_std",
        "oof_score_std",
        "oof_pred_std",
        default=0.0,
    )
    bucket_spread_bps = _ranking_metric(
        metrics,
        "economic_bucket_spread_bps",
        "top_decile_minus_bottom_decile_net_bps",
        "rank_bucket_spread_bps",
        default=0.0,
    )
    ic_component = _unit_interval(weighted_ic, floor=-0.02, good=0.12)
    mono_component = float(np.clip(rank_mono, 0.0, 1.0))
    spread_component = _unit_interval(bucket_spread_bps, floor=-25.0, good=100.0)
    std_component = _unit_interval(score_std, floor=0.01, good=0.10)
    return float(
        0.25
        + 0.30 * ic_component
        + 0.25 * mono_component
        + 0.12 * spread_component
        + 0.08 * std_component
    )


def objective_score(
    metrics: dict[str, Any],
    *,
    model_stage: str = "base",
    objective: ObjectiveParams | None = None,
    phase: str = "all",
) -> float:
    obj = objective if objective is not None else ObjectiveParams()
    stage = _stage_norm(model_stage)
    phase_norm = _phase_norm(phase)
    mean10 = float(metrics.get("mean_net_bps_at_10", metrics.get("mean_net_bps@10", 0.0)))
    mean20 = float(metrics.get("mean_net_bps_at_20", metrics.get("mean_net_bps@20", 0.0)))
    mean30 = float(metrics.get("mean_net_bps_at_30", metrics.get("mean_net_bps@30", mean20)))
    mean50 = float(metrics.get("mean_net_bps_at_50", metrics.get("mean_net_bps@50", mean30)))
    bps_hit10 = float(metrics.get("bps_weighted_hit_at_10", metrics.get("bps_weighted_hit@10", 0.0)))
    bps_hit20 = float(metrics.get("bps_weighted_hit_at_20", metrics.get("bps_weighted_hit@20", bps_hit10)))
    bps_hit30 = float(metrics.get("bps_weighted_hit_at_30", metrics.get("bps_weighted_hit@30", bps_hit20)))
    stop20 = float(metrics.get("stop_hit_rate_at_20", metrics.get("stop_hit_rate@20", 0.0)))
    stop_loss20 = float(metrics.get("avg_stop_loss_bps_at_20", metrics.get("stop_loss_bps_at_20", 0.0)))
    instability = float(metrics.get("prediction_instability", 0.0))
    impact_scale = max(float(obj.impact_scale_bps), 1e-6)
    se_divisor = max(float(obj.edge_lcb_se_divisor), 1e-6)
    edge10 = _edge_lcb(metrics, k=10, aggregate=mean10, se_divisor=se_divisor)
    edge20 = _edge_lcb(metrics, k=20, aggregate=mean20, se_divisor=se_divisor)
    edge30 = _edge_lcb(metrics, k=30, aggregate=mean30, se_divisor=se_divisor)
    edge50 = _edge_lcb(metrics, k=50, aggregate=mean50, se_divisor=se_divisor)
    bps_hit10_lcb = _window_metric_lcb(
        metrics,
        k=10,
        key="bps_weighted_hit",
        aggregate=bps_hit10,
        se_divisor=se_divisor,
    )
    bps_hit20_lcb = _window_metric_lcb(
        metrics,
        k=20,
        key="bps_weighted_hit",
        aggregate=bps_hit20,
        se_divisor=se_divisor,
    )
    bps_hit30_lcb = _window_metric_lcb(
        metrics,
        k=30,
        key="bps_weighted_hit",
        aggregate=bps_hit30,
        se_divisor=se_divisor,
    )
    window_mean20 = _per_window_topk_values(metrics, k=20, key="mean_net_bps")
    window_stop20 = _per_window_topk_values(metrics, k=20, key="stop_hit_rate")
    window_stop_loss20 = _per_window_topk_values(metrics, k=20, key="avg_stop_loss_bps")
    window_symbol_hhi20 = _per_window_topk_values(metrics, k=20, key="symbol_hhi")
    window_week_hhi20 = _per_window_topk_values(metrics, k=20, key="week_hhi")
    window_unique_symbols20 = _per_window_topk_values(metrics, k=20, key="unique_symbols")
    worst_window_mean20 = min(window_mean20) if window_mean20 else mean20
    worst_window_stop20 = max(window_stop20) if window_stop20 else stop20
    worst_window_stop_loss20 = max(window_stop_loss20) if window_stop_loss20 else stop_loss20
    symbol_hhi20 = _metric_first(metrics, "symbol_concentration_hhi_at_20", "symbol_hhi_at_20", default=0.0)
    week_hhi20 = _metric_first(metrics, "week_concentration_hhi_at_20", "week_hhi_at_20", default=0.0)
    unique_symbols20 = _metric_first(metrics, "unique_symbols_at_20", default=float("inf"))
    worst_symbol_hhi20 = max(window_symbol_hhi20) if window_symbol_hhi20 else symbol_hhi20
    worst_week_hhi20 = max(window_week_hhi20) if window_week_hhi20 else week_hhi20
    worst_unique_symbols20 = min(window_unique_symbols20) if window_unique_symbols20 else unique_symbols20
    stop_excess = max(0.0, stop20 - float(obj.max_stop_hit_rate_at_20))
    stop_loss_excess = max(0.0, stop_loss20 - float(obj.max_avg_stop_loss_bps_at_20))
    mean_shortfall = max(0.0, float(obj.min_mean_net_bps_at_20) - edge20)
    window_mean_shortfall = max(0.0, float(obj.min_window_mean_net_bps_at_20) - worst_window_mean20)
    window_stop_excess = max(0.0, worst_window_stop20 - float(obj.max_window_stop_hit_rate_at_20))
    window_stop_loss_excess = max(0.0, worst_window_stop_loss20 - float(obj.max_avg_stop_loss_bps_at_20))
    symbol_concentration_excess = max(0.0, max(symbol_hhi20, worst_symbol_hhi20) - float(obj.max_symbol_hhi_at_20))
    week_concentration_excess = max(0.0, max(week_hhi20, worst_week_hhi20) - float(obj.max_week_hhi_at_20))
    unique_symbol_shortfall = max(0.0, float(obj.min_unique_symbols_at_20) - worst_unique_symbols20)
    score_std = _ranking_metric(
        metrics,
        "prediction_score_std",
        "score_std",
        "oof_score_std",
        "oof_pred_std",
        default=float("nan"),
    )
    score_iqr = _ranking_metric(
        metrics,
        "prediction_score_iqr",
        "score_iqr",
        "oof_score_iqr",
        default=float("nan"),
    )
    score_gap = _ranking_metric(
        metrics,
        "score_gap_top10_to_30_40",
        "score_gap_top10_top30",
        "score_separation_top10_30",
        default=float("nan"),
    )
    rank_mono = _ranking_metric(
        metrics,
        "economic_rank_monotonicity",
        "rank_monotonicity",
        "return_bucket_monotonicity",
        default=0.5,
    )
    econ_ic = _ranking_metric(
        metrics,
        "economic_rank_ic",
        "return_score_ic",
        "rank_ic",
        default=0.0,
    )
    weighted_econ_ic = _weighted_economic_ic(metrics)
    top_weighted_econ_ic = float(
        0.45
        * _ranking_metric(
            metrics,
            "economic_weighted_ic_top20",
            "economic_rank_ic_top20",
            default=weighted_econ_ic,
        )
        + 0.35
        * _ranking_metric(
            metrics,
            "economic_weighted_ic_top10",
            "economic_rank_ic_top10",
            default=weighted_econ_ic,
        )
        + 0.20
        * _ranking_metric(
            metrics,
            "economic_weighted_ic_top30",
            "economic_rank_ic_top30",
            default=weighted_econ_ic,
        )
    )
    bucket_spread_bps = _ranking_metric(
        metrics,
        "economic_bucket_spread_bps",
        "top_decile_minus_bottom_decile_net_bps",
        "rank_bucket_spread_bps",
        default=0.0,
    )
    base_score_std = _baseline_metric(metrics, "prediction_score_std")
    if base_score_std is None:
        base_score_std = _baseline_metric(metrics, "score_std")
    min_abs_std = (
        float(obj.min_score_std_abs_meta)
        if stage == "meta"
        else float(obj.min_score_std_abs_base)
    )
    min_ratio_std = (
        max(1e-6, float(base_score_std)) * float(obj.min_score_std_ratio)
        if base_score_std is not None
        else 0.0
    )
    min_required_std = max(min_abs_std, min_ratio_std)
    score_std_shortfall = (
        max(0.0, min_required_std - score_std)
        if math.isfinite(score_std)
        else 0.0
    )
    monotonicity_shortfall = max(0.0, float(obj.min_rank_monotonicity) - rank_mono)
    ic_shortfall = max(0.0, float(obj.min_economic_rank_ic) - weighted_econ_ic)
    baseline_mean20 = _baseline_metric(metrics, "mean_net_bps_at_20")
    baseline_mean10 = _baseline_metric(metrics, "mean_net_bps_at_10")
    baseline_bps_hit20 = _baseline_metric(metrics, "bps_weighted_hit_at_20")
    baseline_bps_hit10 = _baseline_metric(metrics, "bps_weighted_hit_at_10")
    baseline_score_gap = _baseline_metric(metrics, "score_gap_top10_to_30_40")
    baseline_weighted_ic = _baseline_metric(metrics, "economic_weighted_ic")
    baseline_top20_ic = _baseline_metric(metrics, "economic_weighted_ic_top20")
    baseline_top10_ic = _baseline_metric(metrics, "economic_weighted_ic_top10")
    baseline_top30_ic = _baseline_metric(metrics, "economic_weighted_ic_top30")
    baseline_drawdown = (
        max(0.0, float(baseline_mean20) - mean20 - float(obj.max_top20_mean_net_bps_baseline_drawdown))
        if baseline_mean20 is not None
        else 0.0
    )
    baseline_top10_drawdown = (
        max(0.0, float(baseline_mean10) - mean10 - float(obj.max_top10_mean_net_bps_baseline_drawdown))
        if baseline_mean10 is not None
        else 0.0
    )
    baseline_bps_hit_drawdown = (
        max(0.0, float(baseline_bps_hit20) - bps_hit20 - float(obj.max_top20_bps_weighted_hit_baseline_drawdown))
        if baseline_bps_hit20 is not None
        else 0.0
    )
    baseline_top10_bps_hit_drawdown = (
        max(0.0, float(baseline_bps_hit10) - bps_hit10 - float(obj.max_top10_bps_weighted_hit_baseline_drawdown))
        if baseline_bps_hit10 is not None
        else 0.0
    )
    baseline_std_drawdown = 0.0
    if base_score_std is not None and math.isfinite(score_std):
        min_vs_baseline = float(base_score_std) * (1.0 - float(obj.max_score_std_baseline_drawdown_ratio))
        baseline_std_drawdown = max(0.0, min_vs_baseline - score_std)
    baseline_gap_drawdown = 0.0
    if baseline_score_gap is not None and math.isfinite(score_gap):
        min_gap_vs_baseline = float(baseline_score_gap) * (
            1.0 - float(obj.max_score_gap_baseline_drawdown_ratio)
        )
        baseline_gap_drawdown = max(0.0, min_gap_vs_baseline - score_gap)
    baseline_ic_drawdown = (
        max(
            0.0,
            float(baseline_weighted_ic)
            - weighted_econ_ic
            - float(obj.max_economic_weighted_ic_baseline_drawdown),
        )
        if baseline_weighted_ic is not None
        else 0.0
    )
    baseline_top_ic_drawdown = 0.0
    top_ic_pairs = (
        (baseline_top20_ic, _ranking_metric(metrics, "economic_weighted_ic_top20", default=top_weighted_econ_ic)),
        (baseline_top10_ic, _ranking_metric(metrics, "economic_weighted_ic_top10", default=top_weighted_econ_ic)),
        (baseline_top30_ic, _ranking_metric(metrics, "economic_weighted_ic_top30", default=top_weighted_econ_ic)),
    )
    for baseline_ic, candidate_ic in top_ic_pairs:
        if baseline_ic is None:
            continue
        baseline_top_ic_drawdown = max(
            baseline_top_ic_drawdown,
            max(
                0.0,
                float(baseline_ic) - float(candidate_ic) - float(obj.max_top_weighted_ic_baseline_drawdown),
            ),
        )
    effective_sample_frac = _metric_first(metrics, "effective_sample_frac", default=1.0)
    weight_rank_corr = _metric_first(metrics, "weight_rank_corr_to_baseline", default=1.0)
    weight_delta_abs = _metric_first(metrics, "weight_final_delta_abs_mean", default=0.0)
    weight_concentration_shortfall = 0.0
    weight_rank_shortfall = 0.0
    weight_delta_excess = 0.0
    if phase_norm == "weights":
        weight_concentration_shortfall = max(
            0.0,
            float(obj.min_effective_sample_frac_weight) - effective_sample_frac,
        )
        weight_rank_shortfall = max(
            0.0,
            float(obj.min_weight_rank_corr_to_baseline) - weight_rank_corr,
        )
        weight_delta_excess = max(
            0.0,
            weight_delta_abs - float(obj.max_weight_final_delta_abs_mean),
        )
    hard_stop_excess = max(0.0, stop20 - min(0.95, float(obj.max_stop_hit_rate_at_20) + 0.25))
    hard_window_stop_excess = max(
        0.0,
        worst_window_stop20 - min(0.95, float(obj.max_window_stop_hit_rate_at_20) + 0.25),
    )
    hard_stop_loss_excess = max(
        0.0,
        max(stop_loss20, worst_window_stop_loss20) - (float(obj.max_avg_stop_loss_bps_at_20) + impact_scale),
    )
    hard_mean_shortfall = max(
        0.0,
        (float(obj.min_mean_net_bps_at_20) - impact_scale) - min(edge20, worst_window_mean20),
    )
    if (
        hard_stop_excess > 0.0
        or hard_window_stop_excess > 0.0
        or hard_stop_loss_excess > 0.0
        or hard_mean_shortfall > 0.0
        or baseline_drawdown > impact_scale
        or baseline_top10_drawdown > impact_scale
        or baseline_bps_hit_drawdown > 0.12
        or baseline_top10_bps_hit_drawdown > 0.15
        or weight_concentration_shortfall > 0.35
        or weight_rank_shortfall > 0.50
    ):
        return float(
            -10.0
            - 8.0 * hard_stop_excess
            - 8.0 * hard_window_stop_excess
            - 2.0 * math.tanh(hard_stop_loss_excess / impact_scale)
            - 2.0 * math.tanh(hard_mean_shortfall / impact_scale)
            - 2.0 * math.tanh(baseline_drawdown / impact_scale)
            - 1.5 * math.tanh(baseline_top10_drawdown / impact_scale)
            - 2.0 * math.tanh(baseline_bps_hit_drawdown / 0.08)
            - 1.5 * math.tanh(baseline_top10_bps_hit_drawdown / 0.10)
            - 1.5 * math.tanh(weight_concentration_shortfall / 0.20)
            - 1.0 * math.tanh(weight_rank_shortfall / 0.25)
        )
    smooth_penalty = (
        1.25 * math.tanh(stop_excess / 0.15)
        + 0.75 * math.tanh(window_stop_excess / 0.15)
        + 0.75 * math.tanh(stop_loss_excess / impact_scale)
        + 0.50 * math.tanh(window_stop_loss_excess / impact_scale)
        + 0.75 * math.tanh(mean_shortfall / impact_scale)
        + 0.75 * math.tanh(window_mean_shortfall / impact_scale)
        + 0.50 * math.tanh(symbol_concentration_excess / 0.20)
        + 0.35 * math.tanh(week_concentration_excess / 0.20)
        + 0.25 * math.tanh(unique_symbol_shortfall / 4.0)
        + 1.10 * math.tanh(score_std_shortfall / 0.04)
        + 0.65 * math.tanh(monotonicity_shortfall / 0.20)
        + 0.50 * math.tanh(ic_shortfall / 0.05)
        + 1.20 * math.tanh(baseline_drawdown / impact_scale)
        + 0.85 * math.tanh(baseline_top10_drawdown / impact_scale)
        + 0.85 * math.tanh(baseline_bps_hit_drawdown / 0.08)
        + 0.65 * math.tanh(baseline_top10_bps_hit_drawdown / 0.10)
        + 0.80 * math.tanh(baseline_std_drawdown / 0.04)
        + 0.85 * math.tanh(baseline_gap_drawdown / 0.06)
        + 0.65 * math.tanh(baseline_ic_drawdown / 0.04)
        + 0.85 * math.tanh(baseline_top_ic_drawdown / 0.04)
        + 1.20 * math.tanh(weight_concentration_shortfall / 0.25)
        + 0.85 * math.tanh(weight_rank_shortfall / 0.30)
        + 0.75 * math.tanh(weight_delta_excess / 0.50)
    )
    ic_norm = _unit_interval(
        weighted_econ_ic,
        floor=float(obj.economic_ic_floor),
        good=float(obj.economic_ic_good),
    )
    lgbm_j = _lgbm_style_j(metrics, stage=stage)
    j_norm = _unit_interval(
        lgbm_j,
        floor=float(obj.lgbm_j_floor),
        good=float(obj.lgbm_j_good),
    )
    rank_model_score = math.sqrt(max(0.0, ic_norm) * max(0.0, j_norm))
    edge_score = float(
        0.45 * _unit_interval(edge20, floor=-impact_scale, good=impact_scale / 3.0)
        + 0.30 * _unit_interval(edge10, floor=-impact_scale, good=impact_scale / 2.0)
        + 0.20 * _unit_interval(edge30, floor=-impact_scale, good=impact_scale / 4.0)
        + 0.05 * _unit_interval(edge50, floor=-impact_scale, good=0.0)
    )
    bps_hit_score = float(
        0.50 * _unit_interval(bps_hit20_lcb, floor=0.25, good=0.55)
        + 0.30 * _unit_interval(bps_hit10_lcb, floor=0.25, good=0.60)
        + 0.20 * _unit_interval(bps_hit30_lcb, floor=0.25, good=0.52)
    )
    score = float(
        0.35 * rank_model_score
        + 0.40 * edge_score
        + 0.25 * bps_hit_score
        - 0.10 * instability
        - smooth_penalty
    )
    portfolio_strength = float(np.clip(obj.portfolio_alignment_strength, 0.0, 1.0))
    if portfolio_strength > 0.0:
        sharpe = _metric_first(metrics, "portfolio_sharpe", "sharpe", "Sharpe", "Sharpe_t30")
        sortino = _metric_first(metrics, "portfolio_sortino", "sortino", "Sortino", "Sortino_t30")
        drawdown = abs(_metric_first(metrics, "max_drawdown_bps", "drawdown_bps", "max_drawdown", default=0.0))
        concentration = _metric_first(
            metrics,
            "symbol_concentration_hhi",
            "regime_concentration_hhi",
            "concentration_hhi",
            default=0.0,
        )
        portfolio_term = (
            0.35 * math.tanh(sharpe / 2.0)
            + 0.25 * math.tanh(sortino / 2.0)
            - 0.25 * math.tanh(drawdown / 500.0)
            - 0.15 * float(np.clip(concentration, 0.0, 1.0))
        )
        score += portfolio_strength * portfolio_term
    return float(score)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")


SELECTED_PROMOTION_METRICS = (
    "J_final",
    "economic_weighted_ic",
    "economic_weighted_ic_full",
    "economic_weighted_ic_top30",
    "economic_weighted_ic_top20",
    "economic_weighted_ic_top10",
    "economic_rank_ic",
    "economic_rank_monotonicity",
    "mean_net_bps_at_10",
    "mean_net_bps_at_20",
    "mean_net_bps_at_30",
    "mean_net_bps_at_50",
    "bps_weighted_hit_at_10",
    "bps_weighted_hit_at_20",
    "bps_weighted_hit_at_30",
    "bps_weighted_hit_at_50",
    "stop_hit_rate_at_20",
    "avg_stop_loss_bps_at_20",
    "prediction_score_std",
    "prediction_score_iqr",
    "score_gap_top10_to_30_40",
    "unique_symbols_at_20",
    "unique_weeks_at_20",
    "symbol_concentration_hhi_at_20",
    "week_concentration_hhi_at_20",
    "label_final_changed_frac",
    "label_final_delta_abs_mean",
    "weight_final_changed_frac",
    "weight_final_delta_abs_mean",
    "weight_rank_corr_to_baseline",
    "effective_sample_frac",
)


def _selected_metric_snapshot(metrics: dict[str, Any] | None) -> dict[str, float]:
    if not isinstance(metrics, dict):
        return {}
    out: dict[str, float] = {}
    for key in SELECTED_PROMOTION_METRICS:
        if key not in metrics:
            continue
        try:
            value = float(metrics[key])
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            out[key] = value
    return out


def _selected_metric_deltas(candidate: dict[str, float], reference: dict[str, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in candidate.items():
        if key not in reference:
            continue
        out[key] = float(value - reference[key])
    return out


def _promotion_guard_decision(
    *,
    candidate_metrics: dict[str, Any],
    incumbent_metrics: dict[str, Any] | None,
    neutral_metrics: dict[str, Any] | None,
    objective: ObjectiveParams,
    phase: str = "all",
) -> tuple[bool, dict[str, Any]]:
    candidate = _selected_metric_snapshot(candidate_metrics)
    incumbent = _selected_metric_snapshot(incumbent_metrics)
    neutral = _selected_metric_snapshot(neutral_metrics)
    reference = neutral or incumbent
    failures: list[str] = []

    def _candidate_at_least(key: str, floor: float, *, label: str | None = None) -> None:
        if key not in candidate:
            failures.append(f"candidate missing {key}")
            return
        if candidate[key] < floor:
            failures.append(f"{label or key}: {candidate[key]:.6g} < {floor:.6g}")

    def _candidate_at_most(key: str, ceiling: float, *, label: str | None = None) -> None:
        if key not in candidate:
            failures.append(f"candidate missing {key}")
            return
        if candidate[key] > ceiling:
            failures.append(f"{label or key}: {candidate[key]:.6g} > {ceiling:.6g}")

    phase_norm = _phase_norm(phase)
    if phase_norm == "labels":
        _candidate_at_least(
            "label_final_changed_frac",
            float(objective.min_label_final_changed_frac),
            label="label residual changed-frac guard",
        )
        _candidate_at_least(
            "label_final_delta_abs_mean",
            float(objective.min_label_final_delta_abs_mean),
            label="label residual delta guard",
        )
    if phase_norm == "weights":
        _candidate_at_least(
            "weight_final_changed_frac",
            float(objective.min_weight_final_changed_frac),
            label="weight residual changed-frac guard",
        )
        _candidate_at_least(
            "weight_final_delta_abs_mean",
            float(objective.min_weight_final_delta_abs_mean),
            label="weight residual delta guard",
        )
        _candidate_at_least(
            "effective_sample_frac",
            float(objective.min_effective_sample_frac_weight),
            label="weight effective-sample guard",
        )
        _candidate_at_least(
            "weight_rank_corr_to_baseline",
            float(objective.min_weight_rank_corr_to_baseline),
            label="weight rank-correlation guard",
        )
        _candidate_at_most(
            "weight_final_delta_abs_mean",
            float(objective.max_weight_final_delta_abs_mean),
            label="weight delta guard",
        )

    if reference:
        if "mean_net_bps_at_20" in reference:
            _candidate_at_least(
                "mean_net_bps_at_20",
                reference["mean_net_bps_at_20"] - float(objective.max_top20_mean_net_bps_baseline_drawdown),
                label="top20 mean net baseline guard",
            )
        if "mean_net_bps_at_10" in reference:
            _candidate_at_least(
                "mean_net_bps_at_10",
                reference["mean_net_bps_at_10"] - float(objective.max_top10_mean_net_bps_baseline_drawdown),
                label="top10 mean net baseline guard",
            )
        if "bps_weighted_hit_at_20" in reference:
            _candidate_at_least(
                "bps_weighted_hit_at_20",
                reference["bps_weighted_hit_at_20"]
                - float(objective.max_top20_bps_weighted_hit_baseline_drawdown),
                label="top20 bps-weighted hit baseline guard",
            )
        if "bps_weighted_hit_at_10" in reference:
            _candidate_at_least(
                "bps_weighted_hit_at_10",
                reference["bps_weighted_hit_at_10"]
                - float(objective.max_top10_bps_weighted_hit_baseline_drawdown),
                label="top10 bps-weighted hit baseline guard",
            )
        if "prediction_score_std" in reference:
            _candidate_at_least(
                "prediction_score_std",
                reference["prediction_score_std"]
                * (1.0 - float(objective.max_score_std_baseline_drawdown_ratio)),
                label="score std baseline guard",
            )
        if "score_gap_top10_to_30_40" in reference:
            _candidate_at_least(
                "score_gap_top10_to_30_40",
                reference["score_gap_top10_to_30_40"]
                * (1.0 - float(objective.max_score_gap_baseline_drawdown_ratio)),
                label="top score-gap baseline guard",
            )
        if "economic_weighted_ic" in reference:
            _candidate_at_least(
                "economic_weighted_ic",
                reference["economic_weighted_ic"]
                - float(objective.max_economic_weighted_ic_baseline_drawdown),
                label="weighted IC baseline guard",
            )
        for key in (
            "economic_weighted_ic_top30",
            "economic_weighted_ic_top20",
            "economic_weighted_ic_top10",
        ):
            if key in reference:
                _candidate_at_least(
                    key,
                    reference[key] - float(objective.max_top_weighted_ic_baseline_drawdown),
                    label=f"{key} baseline guard",
                )

    payload = {
        "passes": not failures,
        "failures": failures,
        "candidate": candidate,
        "incumbent": incumbent,
        "neutral": neutral,
        "candidate_minus_incumbent": _selected_metric_deltas(candidate, incumbent),
        "candidate_minus_neutral": _selected_metric_deltas(candidate, neutral),
    }
    return not failures, payload


def _write_promotion_comparison(out_dir: Path, payload: dict[str, Any]) -> None:
    _atomic_write_json(out_dir / "promotion_comparison.json", payload)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-template", type=Path, help="Write a default recipe JSON and exit.")
    parser.add_argument(
        "--phase",
        choices=["label_geometry", "geometry", "hard_labels", "labels", "weights", "distillation", "all"],
        default="label_geometry",
    )
    parser.add_argument(
        "--layer",
        choices=list(OPTUNA_LAYER_CHOICES),
        default=LAYER_ALL,
        help=(
            "Restrict a phase to one semantic sub-layer. Use all for the historical broad "
            "phase search, or auto to let Optuna choose one valid sub-layer before sampling "
            "that sub-layer's knobs."
        ),
    )
    parser.add_argument("--study-name", default="label_weight_optuna")
    parser.add_argument("--storage", default="")
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--study-patience", type=int, default=DEFAULT_STUDY_PATIENCE_TRIALS)
    parser.add_argument(
        "--pruner",
        choices=["successive_halving", "median", "hyperband", "none"],
        default="successive_halving",
        help="Aggressive early-pruning strategy for the label/weight Optuna study.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("reports_perp/label_weight_optuna"))
    parser.add_argument(
        "--best-recipe-path",
        type=Path,
        default=DEFAULT_BEST_RECIPE_PATH,
        help=(
            "Reusable best recipe path loaded by default in future training runs. "
            "Set EPM_LABEL_WEIGHT_BYPASS_BEST_DEFAULT=1 or "
            "EPM_LABEL_WEIGHT_USE_BEST_DEFAULT=0 to use hardcoded recipe defaults instead. "
            "Set EPM_LABEL_WEIGHT_DISABLE=1 for the pre-HPO neutral baseline with no recipe transforms."
        ),
    )
    parser.add_argument(
        "--previous-best",
        type=Path,
        default=None,
        help="Previous best_trial.json or best_recipe.json to enqueue as the first trial.",
    )
    parser.add_argument(
        "--base-recipe",
        type=Path,
        default=None,
        help=(
            "Fixed recipe to carry into this phase. For sequential studies, pass "
            "the labels best recipe into weights, and the weights best recipe into "
            "distillation so earlier optimised fields are preserved."
        ),
    )
    parser.add_argument(
        "--no-enqueue-previous-best",
        action="store_true",
        help="Do not enqueue the previous best recipe/params as the first trial.",
    )
    parser.add_argument(
        "--no-enqueue-noop",
        action="store_true",
        help="Do not enqueue the neutral do-nothing/incumbent recipe trial before other trials.",
    )
    parser.add_argument(
        "--promotion-margin",
        type=float,
        default=DEFAULT_PROMOTION_MARGIN,
        help=(
            "Minimum objective improvement required over the neutral/incumbent trial before "
            "best_recipe.json is promoted."
        ),
    )
    parser.add_argument(
        "--dry-run-recipes",
        action="store_true",
        help="Emit trial recipe JSON files without requiring an evaluator or optimizing metrics.",
    )
    parser.add_argument(
        "--lgbm-n-estimators-cap",
        type=int,
        default=DEFAULT_LGBM_N_ESTIMATORS_CAP,
        help="Evaluator-side hard cap for LightGBM boosting iterations.",
    )
    parser.add_argument(
        "--lgbm-hpo-trials",
        type=int,
        default=DEFAULT_LGBM_HPO_TRIALS,
        help="Evaluator-side cap for internal LGBM HPO trials.",
    )
    parser.add_argument(
        "--lgbm-early-stopping-rounds",
        type=int,
        default=DEFAULT_LGBM_EARLY_STOPPING_ROUNDS,
        help="Evaluator-side LightGBM early stopping rounds.",
    )
    parser.add_argument(
        "--eval-command",
        default="",
        help=(
            "Command to run per trial. The command receives "
            "EPM_LABEL_WEIGHT_RECIPE, EPM_LGBM_CV_MODE=interleaved_spread, "
            "EPM_LGBM_N_ESTIMATORS_CAP, EPM_LGBM_HPO_TRIALS, "
            "and EPM_LGBM_EARLY_STOPPING_ROUNDS in its environment."
        ),
    )
    parser.add_argument(
        "--metrics-json",
        default="",
        help="Optional metrics JSON path, may include {trial}; read after --eval-command.",
    )
    parser.add_argument(
        "--neutral-baseline-metrics",
        type=Path,
        default=None,
        help=(
            "Metrics JSON for the original do-nothing recipe. When provided, every trial "
            "is scored and promoted against this neutral baseline in addition to the "
            "phase incumbent."
        ),
    )
    args = parser.parse_args(argv)
    if args.write_template:
        recipe = LabelWeightRecipe(
            name="template",
            provenance={
                "note": "Set EPM_LABEL_WEIGHT_RECIPE to this file to override labels/weights/distillation.",
                "execution_costs": "Leave empty to inherit downstream fixed execution-aware cost.",
            },
        )
        _write_json(args.write_template, recipe.to_dict())
        print(str(args.write_template))
        return 0
    try:
        import optuna  # type: ignore
    except Exception as exc:
        raise SystemExit(f"Optuna is required for studies: {exc}") from exc
    fast_eval_sentinel = "__fast_long_dist__"
    if not args.eval_command and not bool(args.dry_run_recipes):
        raise SystemExit("--eval-command is required unless --dry-run-recipes is set.")
    if args.eval_command and not args.metrics_json:
        raise SystemExit("--metrics-json is required when --eval-command is set.")

    sampler = optuna.samplers.TPESampler(
        multivariate=True,
        group=True,
        seed=20260606,
        n_startup_trials=5,
    )
    pruner = _make_optuna_pruner(optuna, str(args.pruner))
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage or None,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    best_recipe_path = args.best_recipe_path.expanduser()
    best_seen = {"value": -np.inf, "trial": -1}
    if not bool(args.no_enqueue_noop):
        _enqueue_noop_trial(study)
        print("Enqueued neutral do-nothing/incumbent recipe as the first trial.", flush=True)
    if not bool(args.no_enqueue_previous_best):
        previous_candidates = [
            args.previous_best.expanduser() if args.previous_best is not None else None,
            args.out_dir / "best_trial.json",
            args.out_dir / "best_recipe.json",
            best_recipe_path,
        ]
        enqueued = False
        seen: set[str] = set()
        for candidate in previous_candidates:
            if candidate is None:
                continue
            key = str(candidate)
            if key in seen:
                continue
            seen.add(key)
            if _enqueue_previous_best(study, candidate, phase=args.phase):
                enqueued = True
                break
        if enqueued:
            print("Enqueued previous best label/weight recipe as the first trial.", flush=True)

    fixed_base_recipe = (
        load_recipe(str(args.base_recipe.expanduser()))
        if args.base_recipe is not None
        else None
    )
    fast_evaluator = None
    if str(args.eval_command).strip() == fast_eval_sentinel:
        from scripts.fast_label_weight_optuna_eval_20260606 import FastLongDistEvaluator

        fast_evaluator = FastLongDistEvaluator()
    neutral_baseline_metrics: dict[str, Any] | None = None
    if args.neutral_baseline_metrics is not None:
        neutral_raw = _read_json_if_exists(args.neutral_baseline_metrics.expanduser())
        if neutral_raw is None:
            raise FileNotFoundError(f"Neutral baseline metrics not found: {args.neutral_baseline_metrics}")
        neutral_baseline_metrics = _selected_metric_snapshot(neutral_raw)

    def objective(trial: Any) -> float:
        recipe = suggest_optuna_params(
            trial,
            phase=args.phase,
            layer=args.layer,
            base_recipe=fixed_base_recipe,
        )
        recipe.provenance.update(
            {
                "study_pruner": str(args.pruner),
                "requested_layer": str(args.layer),
                "study_patience_trials": int(args.study_patience),
                "lgbm_n_estimators_cap": int(args.lgbm_n_estimators_cap),
                "lgbm_hpo_trials": int(args.lgbm_hpo_trials),
                "lgbm_early_stopping_rounds": int(args.lgbm_early_stopping_rounds),
            }
        )
        recipe_path = args.out_dir / f"trial_{trial.number:04d}_recipe.json"
        _write_json(recipe_path, recipe.to_dict())
        metrics = dict(trial.user_attrs.get("metrics", {}))
        if args.eval_command:
            metrics_path = Path(str(args.metrics_json).format(trial=trial.number))
            if fast_evaluator is not None:
                os.environ.update(
                    {
                        "EPM_LABEL_WEIGHT_RECIPE": str(recipe_path),
                        "EPM_LABEL_WEIGHT_TRIAL_NUMBER": str(trial.number),
                        "EPM_LABEL_WEIGHT_METRICS_JSON": str(metrics_path),
                        "EPM_LABEL_WEIGHT_PHASE": str(args.phase),
                        "EPM_LGBM_CV_MODE": "interleaved_spread",
                        "EPM_LGBM_TRUE_SOFT_LABELS": "1",
                        "EPM_LGBM_N_ESTIMATORS_CAP": str(max(1, int(args.lgbm_n_estimators_cap))),
                        "EPM_LGBM_HPO_TRIALS": str(max(0, int(args.lgbm_hpo_trials))),
                        "EPM_LGBM_HPO_EARLY_STOP_PATIENCE": str(max(1, int(args.study_patience))),
                        "EPM_LGBM_EARLY_STOPPING_ROUNDS": str(max(1, int(args.lgbm_early_stopping_rounds))),
                        "EPM_LABEL_WEIGHT_DISABLE": "0",
                        "EPM_LABEL_WEIGHT_USE_BEST_DEFAULT": "0",
                    }
                )
                metrics = dict(
                    fast_evaluator.evaluate(
                        recipe_path=str(recipe_path),
                        trial_number=int(trial.number),
                        phase=str(args.phase),
                        metrics_json=metrics_path,
                    )
                )
            else:
                env = dict(os.environ)
                env.update(
                    {
                        "EPM_LABEL_WEIGHT_RECIPE": str(recipe_path),
                        "EPM_LABEL_WEIGHT_TRIAL_NUMBER": str(trial.number),
                        "EPM_LABEL_WEIGHT_METRICS_JSON": str(metrics_path),
                        "EPM_LABEL_WEIGHT_PHASE": str(args.phase),
                        "EPM_LGBM_CV_MODE": "interleaved_spread",
                        "EPM_LGBM_TRUE_SOFT_LABELS": "1",
                        "EPM_LGBM_N_ESTIMATORS_CAP": str(max(1, int(args.lgbm_n_estimators_cap))),
                        "EPM_LGBM_HPO_TRIALS": str(max(0, int(args.lgbm_hpo_trials))),
                        "EPM_LGBM_HPO_EARLY_STOP_PATIENCE": str(max(1, int(args.study_patience))),
                        "EPM_LGBM_EARLY_STOPPING_ROUNDS": str(max(1, int(args.lgbm_early_stopping_rounds))),
                        "EPM_LABEL_WEIGHT_DISABLE": "0",
                        "EPM_LABEL_WEIGHT_USE_BEST_DEFAULT": "0",
                    }
                )
                subprocess.run(shlex.split(args.eval_command), env=env, check=True)
            if not metrics_path.exists():
                raise FileNotFoundError(f"Trial metrics JSON was not produced: {metrics_path}")
            with metrics_path.open("r", encoding="utf-8") as fh:
                metrics = dict(json.load(fh))
        if neutral_baseline_metrics:
            metrics["neutral"] = dict(neutral_baseline_metrics)
        if not metrics:
            trial.set_user_attr("recipe_path", str(recipe_path))
            trial.set_user_attr("dry_run_recipe_only", True)
            if bool(args.dry_run_recipes):
                return 0.0
            raise RuntimeError(f"No metrics available for trial {trial.number}")
        trial.set_user_attr("recipe_path", str(recipe_path))
        trial.set_user_attr("metrics", metrics)
        phase_norm = _phase_norm(str(args.phase))
        if not _trial_is_noop(trial):
            if phase_norm == "labels":
                changed = _metric_first(metrics, "label_final_changed_frac", default=0.0)
                delta = _metric_first(metrics, "label_final_delta_abs_mean", default=0.0)
                if changed < float(recipe.objective.min_label_final_changed_frac) or delta < float(
                    recipe.objective.min_label_final_delta_abs_mean
                ):
                    score = -25.0
                    trial.set_user_attr(
                        "noop_guard_failure",
                        {
                            "phase": phase_norm,
                            "label_final_changed_frac": changed,
                            "label_final_delta_abs_mean": delta,
                        },
                    )
                    trial.report(float(score), 0)
                    return float(score)
            if phase_norm == "weights":
                changed = _metric_first(metrics, "weight_final_changed_frac", default=0.0)
                delta = _metric_first(metrics, "weight_final_delta_abs_mean", default=0.0)
                if changed < float(recipe.objective.min_weight_final_changed_frac) or delta < float(
                    recipe.objective.min_weight_final_delta_abs_mean
                ):
                    score = -25.0
                    trial.set_user_attr(
                        "noop_guard_failure",
                        {
                            "phase": phase_norm,
                            "weight_final_changed_frac": changed,
                            "weight_final_delta_abs_mean": delta,
                        },
                    )
                    trial.report(float(score), 0)
                    return float(score)
        score = objective_score(
            metrics,
            model_stage=str(metrics.get("model_stage", "base")),
            objective=recipe.objective,
            phase=str(args.phase),
        )
        trial.report(float(score), 0)
        if trial.should_prune():
            raise optuna.TrialPruned()
        return float(score)

    def _trial_is_complete(trial_obj: Any) -> bool:
        state = getattr(trial_obj, "state", None)
        state_name = getattr(state, "name", None)
        if state_name is not None:
            return str(state_name).upper() == "COMPLETE"
        return str(state).split(".")[-1].upper() == "COMPLETE"

    def early_stop_callback(study_obj: Any, trial_obj: Any) -> None:
        if trial_obj.value is None or not _trial_is_complete(trial_obj):
            return
        if float(trial_obj.value) > float(best_seen["value"]):
            best_seen["value"] = float(trial_obj.value)
            best_seen["trial"] = int(trial_obj.number)
            return
        if int(trial_obj.number) - int(best_seen["trial"]) >= max(1, int(args.study_patience)):
            study_obj.stop()

    study.optimize(
        objective,
        n_trials=int(args.trials),
        callbacks=[early_stop_callback],
        n_jobs=1,
        show_progress_bar=False,
    )
    complete_trials = [
        trial
        for trial in study.trials
        if trial.value is not None and _trial_is_complete(trial)
    ]
    if complete_trials and not bool(study.best_trial.user_attrs.get("dry_run_recipe_only", False)):
        incumbent_trials = [trial for trial in complete_trials if _trial_is_noop(trial)]
        incumbent_trial = max(incumbent_trials, key=lambda trial: float(trial.value)) if incumbent_trials else None
        promotion_margin = max(0.0, float(args.promotion_margin))
        best_metrics = dict(study.best_trial.user_attrs.get("metrics", {}))
        incumbent_metrics = (
            dict(incumbent_trial.user_attrs.get("metrics", {}))
            if incumbent_trial is not None
            else None
        )
        best_recipe_raw = _read_json_if_exists(Path(str(study.best_trial.user_attrs.get("recipe_path", ""))))
        best_objective = ObjectiveParams(**dict(best_recipe_raw.get("objective", {}))) if isinstance(best_recipe_raw, dict) else ObjectiveParams()
        guard_pass, promotion_comparison = _promotion_guard_decision(
            candidate_metrics=best_metrics,
            incumbent_metrics=incumbent_metrics,
            neutral_metrics=neutral_baseline_metrics,
            objective=best_objective,
            phase=str(args.phase),
        )
        _write_promotion_comparison(
            args.out_dir,
            {
                "study_name": str(args.study_name),
                "phase": str(args.phase),
                "best_trial_number": int(study.best_trial.number),
                "best_trial_value": float(study.best_trial.value),
                "incumbent_trial_number": None if incumbent_trial is None else int(incumbent_trial.number),
                "incumbent_trial_value": None if incumbent_trial is None else float(incumbent_trial.value),
                "promotion_margin": promotion_margin,
                **promotion_comparison,
            },
        )
        if incumbent_trial is None:
            promote = guard_pass
        else:
            promote = (
                float(study.best_trial.value) >= float(incumbent_trial.value) + promotion_margin
                and guard_pass
            )
        if promote:
            _write_best_artifacts(
                out_dir=args.out_dir,
                best_path=best_recipe_path,
                study_name=str(args.study_name),
                phase=str(args.phase),
                trial=study.best_trial,
                recipe_path=str(study.best_trial.user_attrs.get("recipe_path", "")),
            )
        else:
            _write_rejected_promotion_artifact(
                out_dir=args.out_dir,
                study_name=str(args.study_name),
                phase=str(args.phase),
                best_trial=study.best_trial,
                incumbent_trial=incumbent_trial,
                promotion_margin=promotion_margin,
                promotion_comparison=promotion_comparison,
            )
            if incumbent_trial is not None:
                _write_best_artifacts(
                    out_dir=args.out_dir,
                    best_path=best_recipe_path,
                    study_name=str(args.study_name),
                    phase=str(args.phase),
                    trial=incumbent_trial,
                    recipe_path=str(incumbent_trial.user_attrs.get("recipe_path", "")),
                )
            print(
                "Promotion rejected: "
                f"best={float(study.best_trial.value):.6f}, "
                f"incumbent={'none' if incumbent_trial is None else f'{float(incumbent_trial.value):.6f}'}, "
                f"margin={promotion_margin:.6f}, "
                f"guard_pass={guard_pass}.",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
