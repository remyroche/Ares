#!/usr/bin/env python3
"""Raw-feature shared-trunk direct execution-utility experiment.

The experiment is intentionally chronological and bounded:

* February 2025 is the development/training period.
* March 2025 selects one predeclared feature/task arm.
* The selected arm is refit on February+March labels resolved before April.
* April is scored once and is a reused diagnostic, not promotion evidence.

The direct exact-policy 12-hour net-return head is the sole ranking output.
Auxiliary heads regularise one side-local shared representation; their
predictions are never combined algebraically into the admission score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

# PyTorch, NumPy/OpenBLAS and Arrow can otherwise load competing OpenMP
# runtimes on macOS.  The resulting native barrier crash is nondeterministic.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCHEMA = "canonical_raw_feature_direct_utility_multitask_v1"
SEED = 20260729
SIDES = ("long", "short")
IDENTITY = ("candidate_id", "side_name", "__symbol__", "__ts__")
FRACTIONS = (0.01, 0.05, 0.10, 0.20)
MARCH_START = pd.Timestamp("2025-03-01T00:00:00Z")
FEBRUARY_MID = pd.Timestamp("2025-02-15T00:00:00Z")
APRIL_START = pd.Timestamp("2025-04-01T00:00:00Z")
MAY_START = pd.Timestamp("2025-05-01T00:00:00Z")

DEFAULT_PANEL_ROOT = ROOT / "data_perp/artifacts/canonical_opportunity_payoff_trust_panel_20260729_v2"
DEFAULT_CONTEXT_ROOT = ROOT / "data_perp/artifacts/febapr2025_historical_path_head_context_20260727_v1"
DEFAULT_PATH_LABEL_ROOT = ROOT / "data_perp/artifacts/20260723_s59_h5_path_aux_targets_v11_resolved_supportive_15atr/labels"
DEFAULT_HEALTH = ROOT / "data_perp/artifacts/historical_exact_model_health_failure_20260729_v3/hourly_exact_model_health_and_failure_labels.parquet"
DEFAULT_ACTIVE = ROOT / "data_perp/artifacts/regime_transition_active_head_chronological_oos_20260729_v2/chronological_oos.parquet"
DEFAULT_DESTINATION = ROOT / "data_perp/artifacts/regime_transition_destination_chronological_oos_20260729_v1/destination_chronological_oos.parquet"
DEFAULT_HAZARD = ROOT / "data_perp/artifacts/regime_transition_hazard_challenger_20260727_v1/grouped_oof_cumulative_probabilities.parquet"
DEFAULT_BOCPD = ROOT / "data_perp/artifacts/regime_transition_changepoint_ablation_20260727_v2/grouped_oof_predictions_and_changepoint_context.parquet"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/canonical_raw_feature_direct_utility_multitask_20260729_v1"

BASE_LONG = (
    "base_input__climax_decay", "base_input__cross_asset_corr_1h",
    "base_input__delta_stall_6", "base_input__dow_cos", "base_input__dow_sin",
    "base_input__eig_effective_rank__breakout_all",
    "base_input__eig_participation_ratio__breakout_all", "base_input__eth_btc_ret_1h",
    "base_input__fragmented_flush_recovery", "base_input__giveback_vol_units",
    "base_input__hour_cos", "base_input__hour_sin", "base_input__liquidation_onset_score",
    "base_input__mark_perp_dislocation", "base_input__mark_vs_perp_bps",
    "base_input__market_breadth_1h", "base_input__median_volume_z",
    "base_input__mkt_atr_expansion_1h", "base_input__pct_assets_above_ema_fast",
    "base_input__pct_assets_above_vwap", "base_input__prog_eff_12",
    "base_input__prog_eff_24", "base_input__q_iqr__amihud_z_peer_resid",
    "base_input__qv", "base_input__range_12h_pct",
    "base_input__regime_transition_entropy_48h", "base_input__rejection_proxy",
    "base_input__rvol_z_peer_resid", "base_input__z_r_24",
)
GEOMETRY_LONG = ("base_input__dae_b16_02", "base_input__gmm_ood_score")
BASE_SHORT = (
    "base_input__mark_perp_dislocation", "base_input__mark_vs_perp_bps",
    "base_input__climax_decay", "base_input__impact_12",
    "base_input__post_flush_leverage_rebuild", "base_input__shock_12h",
    "base_input__bb_pos_12", "base_input__liquidation_onset_score",
)
SCORE_CONTEXT = (
    "base_rank_pct_timestamp_side", "base_score_z_timestamp_side",
    "base_group_rows_timestamp_side", "base_margin_to_top40_cutoff_z",
    "base_rank_pct_timestamp_global", "base_score_z_timestamp_global",
    "base_group_rows_timestamp_global",
)
CORE_MARKET = (
    "range_24h_pct", "__meta_raw__volatility_zscore", "trend_r2_24",
    "jump_intensity", "__meta_raw__chop_score",
)
TRANSITION_PREFIXES = ("preentry_transition__", "__regime_source_")
EXTERNAL_TRANSITION = (
    "transition__active_probability",
    "transition__active_available",
    "transition__bocpd_mean", "transition__bocpd_max",
    "transition__bocpd_breaks_010", "transition__bocpd_breaks_025",
    "transition__destination_confidence", "transition__destination_entropy",
    "transition__destination_available",
    "transition__p_destination_state_0", "transition__p_destination_state_1",
    "transition__p_destination_state_2", "transition__p_destination_state_3",
    "transition__p_destination_state_4",
)
HEALTH_FEATURES = (
    "health__available",
    "health__candidate_rows", "health__distinct_assets", "health__long_share",
    "health__raw_score_mean", "health__raw_score_std", "health__raw_score_p90",
    "health__mapped_net_mean", "health__mapped_net_std", "health__mapped_net_p90",
    "health__causal_percentile_mean", "health__causal_percentile_std",
    "health__causal_percentile_entropy", "health__map_reference_log1p_mean",
    "health__low_map_support_share", "health__raw_score_long_minus_short",
    "health__mapped_net_long_minus_short", "health__raw_mapped_rank_spearman",
    "health__raw_mapped_rank_abs_gap", "health__selected_rows",
    "health__selected_symbol_hhi", "health__selected_long_share",
    "health__candidate_rows_delta_24h", "health__recent_resolved_net_ev_hl3d",
    "health__recent_resolved_hit_rate_hl3d",
    "health__recent_resolved_mapping_error_hl3d",
    "health__recent_resolved_cost_bps_hl3d",
    "health__recent_resolved_full_stop_rate_hl3d",
    "health__recent_resolved_effective_rows_hl3d",
)

ECONOMIC_TASKS = (
    "opportunity", "favorable_magnitude", "adverse_magnitude",
    "exit_conversion_loss", "timeout",
)
PATH_TASKS = (
    "path_meaningful_hit", "path_peak_mfe", "path_fast_hit_2h",
    "path_mae_if_hit", "path_mae_if_no_hit",
    "path_confirmed_adverse_trough", "path_future_slope",
)
TASK_ARMS: Mapping[str, tuple[str, ...]] = {
    "direct_only": (),
    "economic_multitask": ECONOMIC_TASKS,
    "economic_plus_path_low_weight": ECONOMIC_TASKS + PATH_TASKS,
    **{
        f"without_{removed}": tuple(name for name in ECONOMIC_TASKS if name != removed)
        for removed in ECONOMIC_TASKS
    },
}
FEATURE_ARMS = (
    "base",
    "base_transition",
    "base_health",
    "base_transition_health",
    "base_transition_health_interactions",
)
DIRECT_WEIGHT = 4.0
ECONOMIC_WEIGHTS = {
    "opportunity": 0.25,
    "favorable_magnitude": 0.20,
    "adverse_magnitude": 0.20,
    "exit_conversion_loss": 0.20,
    "timeout": 0.15,
}
PATH_WEIGHT = 0.05

FORBIDDEN_PREFIXES = (
    "execution_", "exit_", "opportunity_", "mapped_", "causal_score_",
    "__label_", "__path_", "__soft_tb_", "__first_touch_",
)
FORBIDDEN_TOKENS = (
    "target_price", "wait_action", "timing", "mfe", "mae", "future_slope",
    "timeout", "realized", "realised", "target_weight",
)


@dataclass(frozen=True)
class TaskTarget:
    values: np.ndarray
    mask: np.ndarray
    kind: str
    weight: float


@dataclass(frozen=True)
class TaskSpec:
    name: str
    kind: str
    weight: float


@dataclass(frozen=True)
class TrainConfig:
    hidden: tuple[int, ...] = (64, 32)
    dropout: float = 0.10
    learning_rate: float = 0.0015
    weight_decay: float = 0.01
    batch_size: int = 1024
    max_epochs: int = 28
    patience: int = 4
    inner_validation_fraction: float = 0.20


TRAIN_CONFIG = TrainConfig()
CORE_AUXILIARY_HEADS = ECONOMIC_TASKS
_TORCH_RUNTIME_CONFIGURED = False


def configure_torch_runtime(torch_module: Any) -> None:
    global _TORCH_RUNTIME_CONFIGURED
    if _TORCH_RUNTIME_CONFIGURED:
        return
    torch_module.set_num_threads(1)
    torch_module.set_num_interop_threads(1)
    _TORCH_RUNTIME_CONFIGURED = True


def population_contract() -> dict[str, Any]:
    return {
        "identity": list(IDENTITY),
        "expected_rows": 205_194,
        "expected_rows_by_month": {
            "2025-02": 64_512, "2025-03": 71_424, "2025-04": 69_258,
        },
        "expected_rows_by_side": {"long": 102_597, "short": 102_597},
        "primary_target": "execution_net_ev_12h",
        "label_resolution_column": "execution_label_end_utc",
        "label_horizon_hours": 12,
        "architecture_loss_development": ["2025-02"],
        "model_selection": ["2025-03"],
        "final_refit": ["2025-02", "2025-03"],
        "diagnostic_only": ["2025-04"],
    }


def intersect_exact_identities(
    sources: Mapping[str, pd.DataFrame], *, expected_rows: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not sources:
        raise ValueError("at least one identity source is required")
    prepared: dict[str, pd.DataFrame] = {}
    for name, source in sources.items():
        missing = set(IDENTITY).difference(source.columns)
        if missing:
            raise ValueError(f"{name} lacks identity fields: {sorted(missing)}")
        if source.duplicated(list(IDENTITY)).any() or source["candidate_id"].duplicated().any():
            raise ValueError(f"{name} has duplicate or non-one-to-one identity")
        prepared[name] = source.copy()
    iterator = iter(prepared.items())
    _, joined = next(iterator)
    for name, source in iterator:
        overlap = [column for column in source.columns if column in joined.columns and column not in IDENTITY]
        if overlap:
            source = source.drop(columns=overlap)
        joined = joined.merge(source, on=list(IDENTITY), how="inner", validate="one_to_one")
    if expected_rows is not None and len(joined) != int(expected_rows):
        raise ValueError(f"identity intersection expected {expected_rows} rows, got {len(joined)}")
    return joined, {
        "mode": "explicit_common_identity_intersection_one_to_one",
        "keys": list(IDENTITY), "common_rows": len(joined),
        "source_rows": {name: len(value) for name, value in prepared.items()},
    }


def split_february_march_april(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    work = frame.copy()
    work["__ts__"] = pd.to_datetime(work["__ts__"], utc=True, errors="raise")
    month = work["__ts__"].dt.strftime("%Y-%m")
    return (
        work.loc[month.eq("2025-02")].copy(),
        work.loc[month.eq("2025-03")].copy(),
        work.loc[month.eq("2025-04")].copy(),
    )


def validate_exact_execution_targets(frame: pd.DataFrame) -> None:
    validate_exact_targets(frame)


def resolved_training_mask(frame: pd.DataFrame, cutoff: pd.Timestamp) -> np.ndarray:
    return strict_train_mask(frame, cutoff)


def select_raw_feature_columns(
    frame: pd.DataFrame, requested: Sequence[str],
) -> tuple[str, ...]:
    missing = [name for name in requested if name not in frame]
    if missing:
        raise ValueError(f"requested feature columns are missing: {missing}")
    validate_feature_names(requested)
    return tuple(requested)


def validate_context_availability(
    frame: pd.DataFrame,
    *,
    source_columns: Mapping[str, str],
    health_lineage: str,
) -> None:
    if health_lineage != "historical_raw_alpha_v3":
        raise ValueError("health lineage must be historical_raw_alpha_v3")
    decision = pd.to_datetime(frame["__decision_ts__"], utc=True, errors="raise")
    for feature, source in source_columns.items():
        if feature not in frame or source not in frame:
            raise ValueError(f"context feature/source is missing: {feature}/{source}")
        available = pd.to_datetime(frame[source], utc=True, errors="raise")
        if available.gt(decision).any():
            raise ValueError(f"future point-in-time context source for {feature}")


def task_specs(arm: str) -> dict[str, TaskSpec]:
    all_auxiliary = {
        "direct_net": TaskSpec("direct_net", "regression", 1.0),
        **{
            name: TaskSpec(
                name, "binary" if name in {"opportunity", "timeout"} else "regression",
                ECONOMIC_WEIGHTS[name] / DIRECT_WEIGHT,
            )
            for name in ECONOMIC_TASKS
        },
    }
    if arm == "direct_only":
        return {"direct_net": all_auxiliary["direct_net"]}
    if arm in {"all_aux_low_weight", "economic_multitask"}:
        return all_auxiliary
    if arm.startswith("without_"):
        removed = arm.removeprefix("without_")
        if removed not in ECONOMIC_TASKS:
            raise ValueError(f"unknown add-one-out auxiliary: {removed}")
        return {name: spec for name, spec in all_auxiliary.items() if name != removed}
    if arm == "economic_plus_path_low_weight":
        result = dict(all_auxiliary)
        result.update({
            name: TaskSpec(name, "binary" if name in {"path_meaningful_hit", "path_fast_hit_2h"} else "regression", PATH_WEIGHT / DIRECT_WEIGHT)
            for name in PATH_TASKS
        })
        return result
    raise ValueError(f"unknown task arm: {arm}")


def ranking_score_column() -> str:
    return "direct_net_score"


def validate_ranking_score_column(column: str) -> None:
    if column != ranking_score_column():
        raise ValueError("direct_net_score is the only eligible ranking score")


def ranking_scope() -> str:
    return "one_pooled_global_cross_timestamp_cross_side"


def causal_mapping_reference_mask(
    frame: pd.DataFrame, snapshot: pd.Timestamp,
) -> np.ndarray:
    return pd.to_datetime(
        frame["execution_label_end_utc"], utc=True, errors="raise",
    ).lt(pd.Timestamp(snapshot)).to_numpy()


def experiment_manifest_contract() -> dict[str, Any]:
    return {
        "selection": {
            "ranking_score": ranking_score_column(),
            "scope": "pooled_global",
            "no_per_timestamp_quota": True,
            "auxiliary_outputs_are_ranking_inputs": False,
        },
        "validation": {
            "april_untouched_by_selection": True,
            "april_status": "diagnostic_only_not_promotion_evidence",
        },
        "outputs": {"immutable": True, "sha256_manifest_sidecar": True},
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def identity_sha256(frame: pd.DataFrame) -> str:
    work = frame.loc[:, list(IDENTITY)].copy()
    work["__ts__"] = pd.to_datetime(work["__ts__"], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    work = work.sort_values(list(IDENTITY), kind="stable")
    digest = hashlib.sha256()
    for row in work.itertuples(index=False, name=None):
        digest.update(("\x1f".join(map(str, row)) + "\n").encode())
    return digest.hexdigest()


def validate_feature_names(features: Sequence[str]) -> None:
    for name in features:
        lower = name.lower()
        if name.startswith(FORBIDDEN_PREFIXES) or any(token in lower for token in FORBIDDEN_TOKENS):
            raise ValueError(f"forbidden target/action feature: {name}")


def stable_global_top_mask(
    frame: pd.DataFrame, score: Sequence[float], fraction: float,
) -> np.ndarray:
    if not 0.0 < float(fraction) <= 1.0:
        raise ValueError("global selection fraction must be in (0, 1]")
    values = np.asarray(score, dtype=float)
    if len(values) != len(frame) or not np.isfinite(values).all():
        raise ValueError("ranking score must be finite and aligned")
    order = pd.DataFrame({
        "position": np.arange(len(frame)),
        "candidate_id": frame["candidate_id"].astype(str),
        "score": values,
    }).sort_values(["score", "candidate_id"], ascending=[False, True], kind="stable")
    mask = np.zeros(len(frame), dtype=bool)
    mask[order["position"].to_numpy()[:max(1, int(math.ceil(len(frame) * fraction)))]] = True
    return mask


def select_rank_score(predictions: Mapping[str, np.ndarray]) -> np.ndarray:
    """Return the only deployable score, intentionally ignoring auxiliaries."""
    if "direct_net" not in predictions:
        raise ValueError("direct_net prediction is required")
    return np.asarray(predictions["direct_net"], dtype=float)


def validate_exact_targets(frame: pd.DataFrame) -> None:
    if frame["candidate_id"].duplicated().any():
        raise ValueError("candidate identity is not unique")
    decision = pd.to_datetime(frame["__decision_ts__"], utc=True)
    signal = pd.to_datetime(frame["__ts__"], utc=True)
    resolution = pd.to_datetime(frame["execution_label_end_utc"], utc=True)
    if not decision.eq(signal + pd.Timedelta(hours=1)).all():
        raise ValueError("decision must equal signal + 1 hour")
    if not resolution.eq(decision + pd.Timedelta(hours=12)).all():
        raise ValueError("execution label must resolve exactly 12 hours after decision")
    net = pd.to_numeric(frame["execution_net_ev_12h"], errors="raise").to_numpy(float)
    gross = pd.to_numeric(frame["execution_gross_ev_12h"], errors="raise").to_numpy(float)
    cost = pd.to_numeric(frame["execution_cost_return"], errors="raise").to_numpy(float)
    if not np.allclose(gross - cost, net, atol=1e-12, rtol=0.0):
        raise ValueError("exact target violates gross - cost = net")
    exit_flags = [
        "exit_is_trailing", "exit_is_timeout", "exit_is_full_stop",
        "exit_is_adverse_exit",
    ]
    if all(name in frame for name in exit_flags):
        flags = frame[exit_flags].astype(int)
        if not flags.sum(axis=1).eq(1).all():
            raise ValueError("exit flags must be mutually exclusive and exhaustive")


def strict_train_mask(frame: pd.DataFrame, cutoff: pd.Timestamp) -> np.ndarray:
    resolution = pd.to_datetime(frame["execution_label_end_utc"], utc=True)
    mask = resolution.lt(cutoff).to_numpy()
    if mask.any() and not resolution.loc[mask].lt(cutoff).all():
        raise AssertionError("training label is not strictly resolved before cutoff")
    return mask


def split_calendar(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    signal = pd.to_datetime(frame["__ts__"], utc=True)
    return {
        "february": signal.lt(MARCH_START).to_numpy(),
        "march": signal.ge(MARCH_START).to_numpy() & signal.lt(APRIL_START).to_numpy(),
        "april": signal.ge(APRIL_START).to_numpy() & signal.lt(MAY_START).to_numpy(),
    }


def _load_verified_panel(panel_root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest_path = panel_root / "manifest.json"
    panel_path = panel_root / "panel.parquet"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "canonical_opportunity_payoff_trust_panel_v2":
        raise ValueError("canonical panel v2 is required")
    if manifest["outputs_sha256"]["panel.parquet"] != sha256_file(panel_path):
        raise ValueError("canonical panel hash mismatch")
    frame = pd.read_parquet(panel_path)
    for name in ("__ts__", "__decision_ts__", "execution_label_end_utc"):
        frame[name] = pd.to_datetime(frame[name], utc=True, errors="raise")
    validate_exact_targets(frame)
    return frame, manifest


def _load_context_population(
    context_root: Path,
) -> tuple[pd.DataFrame, tuple[str, ...], dict[str, Any]]:
    from extreme_price_movements.path_auxiliary_lgbm import (
        configured_auxiliary_feature_universe,
    )

    manifest = json.loads((context_root / "manifest.json").read_text())
    index = pd.read_parquet(context_root / "context_index.parquet")
    index["__ts__"] = pd.to_datetime(index["__ts__"], utc=True, errors="raise")
    if index.duplicated(list(IDENTITY)).any():
        raise ValueError("context index is not one-to-one")
    if len(index) == 205_194:
        counts = index["__ts__"].dt.strftime("%Y-%m").value_counts().to_dict()
        if counts != {"2025-03": 71_424, "2025-04": 69_258, "2025-02": 64_512}:
            raise ValueError(f"unexpected context month counts: {counts}")
    first_manifest = json.loads(Path(str(index["shard_manifest"].iloc[0])).read_text())
    first_path = Path(str(first_manifest["output_path"]))
    available = pd.read_parquet(first_path).columns.tolist()
    feature_names, universe = configured_auxiliary_feature_universe(available)
    requested = list(dict.fromkeys([*IDENTITY, "__decision_ts__", "base_oof_score", *feature_names]))
    pieces = []
    for shard_manifest in index["shard_manifest"].drop_duplicates():
        shard = json.loads(Path(str(shard_manifest)).read_text())
        pieces.append(pd.read_parquet(Path(str(shard["output_path"])), columns=requested))
    context = pd.concat(pieces, ignore_index=True)
    context["__ts__"] = pd.to_datetime(context["__ts__"], utc=True, errors="raise")
    context["__decision_ts__"] = pd.to_datetime(context["__decision_ts__"], utc=True, errors="raise")
    if context.duplicated(list(IDENTITY)).any() or len(context) != len(index):
        raise ValueError("loaded PIT context is not the exact one-to-one population")
    context["score"] = pd.to_numeric(context["base_oof_score"], errors="coerce")
    features = tuple(dict.fromkeys([*feature_names, "score"]))
    validate_feature_names(features)
    manifest = dict(manifest)
    manifest["configured_feature_universe"] = universe
    return context, features, manifest


def _load_path_labels(label_root: Path, identities: pd.DataFrame) -> pd.DataFrame:
    columns = [
        *IDENTITY, "__label_end_ts__", "__meaningful_mfe_reached_12h__",
        "__peak_mfe_atr_12h__", "__time_to_first_meaningful_mfe_hours_12h__",
        "__mae_before_1_5atr_mfe__", "__mae_until_horizon_if_no_1_5atr__",
        "__bars_to_confirmed_adverse_trough__", "__future_slope_atr_per_hour_12h__",
        "__path_auxiliary_target_valid__", "__time_to_first_meaningful_mfe_target_valid__",
    ]
    pieces = []
    wanted = set(identities["candidate_id"].astype(str))
    for side in SIDES:
        path = label_root / f"train_global_{side}_3.parquet"
        piece = pd.read_parquet(path, columns=columns)
        piece = piece.loc[piece["candidate_id"].astype(str).isin(wanted)].copy()
        pieces.append(piece)
    labels = pd.concat(pieces, ignore_index=True)
    labels["__ts__"] = pd.to_datetime(labels["__ts__"], utc=True, errors="raise")
    labels["__label_end_ts__"] = pd.to_datetime(labels["__label_end_ts__"], utc=True, errors="raise")
    # The target store uses exchange symbols such as ``BTC/USD:USD`` while
    # canonical candidate matrices use their feature-store form
    # ``BTC_USD:USD``.  Candidate ID, side and timestamp remain unchanged.
    labels["__symbol__"] = labels["__symbol__"].astype(str).str.replace("/", "_", regex=False)
    if labels.duplicated(list(IDENTITY)).any():
        raise ValueError("path labels are not one-to-one")
    return labels


def _external_context(
    identities: pd.DataFrame,
    *,
    active_path: Path,
    destination_path: Path,
    hazard_path: Path,
    bocpd_path: Path,
    health_path: Path,
) -> pd.DataFrame:
    hours = identities.loc[:, list(IDENTITY)].copy()
    hours["source_utc"] = pd.to_datetime(hours["__ts__"], utc=True)

    active = pd.read_parquet(active_path, columns=["source_utc", "prediction"])
    active["source_utc"] = pd.to_datetime(active["source_utc"], utc=True)
    active = active.rename(columns={"prediction": "transition__active_probability"})
    active["transition__active_available"] = 1.0

    hazard_columns = ["source_utc", *[f"p_onset_within_{h}h" for h in (1, 3, 6, 12)]]
    hazard = pd.read_parquet(hazard_path, columns=hazard_columns)
    hazard["source_utc"] = pd.to_datetime(hazard["source_utc"], utc=True)
    hazard = hazard.rename(columns={f"p_onset_within_{h}h": f"transition__hazard_{h}h" for h in (1, 3, 6, 12)})

    bocpd_columns = [
        "source_utc", "bocpd_context__mean_probability",
        "bocpd_context__max_probability", "bocpd_context__break_count_ge_0_10",
        "bocpd_context__break_count_ge_0_25",
    ]
    bocpd = pd.read_parquet(bocpd_path, columns=bocpd_columns)
    bocpd["source_utc"] = pd.to_datetime(bocpd["source_utc"], utc=True)
    bocpd = bocpd.rename(columns={
        "bocpd_context__mean_probability": "transition__bocpd_mean",
        "bocpd_context__max_probability": "transition__bocpd_max",
        "bocpd_context__break_count_ge_0_10": "transition__bocpd_breaks_010",
        "bocpd_context__break_count_ge_0_25": "transition__bocpd_breaks_025",
    })

    destination_columns = [
        "source_utc", *[f"p_destination__state_{i}" for i in range(5)],
        "destination_confidence", "destination_entropy",
    ]
    destination = pd.read_parquet(destination_path, columns=destination_columns)
    destination["source_utc"] = pd.to_datetime(destination["source_utc"], utc=True)
    destination = destination.rename(columns={
        **{f"p_destination__state_{i}": f"transition__p_destination_state_{i}" for i in range(5)},
        "destination_confidence": "transition__destination_confidence",
        "destination_entropy": "transition__destination_entropy",
    })
    destination["transition__destination_available"] = 1.0

    health_columns = [name for name in HEALTH_FEATURES if name != "health__available"]
    health = pd.read_parquet(health_path, columns=["source_utc", *health_columns])
    health["source_utc"] = pd.to_datetime(health["source_utc"], utc=True)
    health["health__available"] = 1.0

    for name, table in (
        ("active", active), ("hazard", hazard), ("bocpd", bocpd),
        ("destination", destination), ("health", health),
    ):
        if table["source_utc"].duplicated().any():
            raise ValueError(f"{name} context is not unique by source hour")
        hours = hours.merge(table, on="source_utc", how="left", validate="many_to_one")
    availability = (
        "transition__active_available", "transition__destination_available",
        "health__available",
    )
    for name in availability:
        hours[name] = hours[name].fillna(0.0)
    for name in (*EXTERNAL_TRANSITION, *HEALTH_FEATURES):
        if name not in availability:
            hours[name] = pd.to_numeric(hours[name], errors="coerce").fillna(0.0)
    return hours.drop(columns=["source_utc"])


def materialize_population(
    *,
    panel_root: Path,
    context_root: Path,
    label_root: Path,
    active_path: Path,
    destination_path: Path,
    hazard_path: Path,
    bocpd_path: Path,
    health_path: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    panel, panel_manifest = _load_verified_panel(panel_root)
    context, pit_features, context_manifest = _load_context_population(context_root)
    identities = context.loc[:, list(IDENTITY)]
    # The PIT shards are the authoritative raw-feature surface.  Bring only
    # exact economics and canonical candidate context from the full panel so
    # overlapping raw fields cannot acquire merge suffixes.
    panel_columns = list(IDENTITY)
    for name in panel.columns:
        if (
            name in {
                "__decision_ts__", "execution_label_end_utc",
                "execution_net_ev_12h", "execution_gross_ev_12h",
                "execution_cost_return", "execution_mfe_return_12h",
                "opportunity_gross_above_cost_0bps",
                "opportunity_gross_above_cost_25bps", "execution_exit_class",
                "exit_is_trailing", "exit_is_timeout", "exit_is_full_stop",
                "exit_is_adverse_exit",
            }
            or name in SCORE_CONTEXT
            or name in CORE_MARKET
            or name.startswith(TRANSITION_PREFIXES)
        ):
            if name not in panel_columns and name not in context.columns:
                panel_columns.append(name)
    joined = context.merge(
        panel.loc[:, panel_columns], on=list(IDENTITY), how="inner",
        validate="one_to_one",
    )
    if len(joined) != len(context):
        raise ValueError("canonical panel does not cover the exact context population")
    labels = _load_path_labels(label_root, identities)
    joined = joined.merge(labels, on=list(IDENTITY), how="inner", validate="one_to_one")
    if len(joined) != len(identities):
        raise ValueError("path labels do not cover the exact context population")
    external = _external_context(
        identities, active_path=active_path, destination_path=destination_path,
        hazard_path=hazard_path, bocpd_path=bocpd_path, health_path=health_path,
    )
    joined = joined.merge(external, on=list(IDENTITY), how="left", validate="one_to_one")
    joined = joined.sort_values(list(IDENTITY), kind="stable").reset_index(drop=True)
    joined.attrs["pit_feature_columns"] = pit_features
    validate_exact_targets(joined)
    if len(joined) == 205_194:
        side_counts = joined["side_name"].value_counts().to_dict()
        if side_counts != {"long": 102_597, "short": 102_597}:
            raise ValueError(f"unexpected side counts: {side_counts}")
    audit = {
        "rows": len(joined),
        "identity_sha256": identity_sha256(joined),
        "panel_manifest_sha256": sha256_file(panel_root / "manifest.json"),
        "panel_sha256": panel_manifest["outputs_sha256"]["panel.parquet"],
        "context_manifest_sha256": sha256_file(context_root / "manifest.json"),
        "context_identity_sha256": context_manifest["context_index"]["identity_sha256"],
        "pit_raw_feature_count": len(pit_features),
        "pit_raw_features": list(pit_features),
        "path_labels_manifest_sha256": sha256_file(label_root / "labels_manifest.json"),
        "external_context": {
            "active_sha256": sha256_file(active_path),
            "destination_sha256": sha256_file(destination_path),
            "hazard_sha256": sha256_file(hazard_path),
            "bocpd_sha256": sha256_file(bocpd_path),
            "health_sha256": sha256_file(health_path),
        },
    }
    return joined, audit


def transition_features(frame: pd.DataFrame) -> tuple[str, ...]:
    raw = tuple(
        name for name in frame.columns
        if name.startswith(TRANSITION_PREFIXES)
    )
    result = (*CORE_MARKET, *raw, *EXTERNAL_TRANSITION)
    validate_feature_names(result)
    return tuple(dict.fromkeys(result))


def interaction_features(frame: pd.DataFrame) -> tuple[str, ...]:
    pairs = (
        ("transition__active_probability", "health__recent_resolved_mapping_error_hl3d"),
        ("transition__active_probability", "health__recent_resolved_net_ev_hl3d"),
        ("transition__bocpd_max", "health__raw_mapped_rank_abs_gap"),
        ("transition__destination_entropy", "health__low_map_support_share"),
    )
    names = []
    for left, right in pairs:
        name = f"interaction__{left}__x__{right}"
        frame[name] = (
            pd.to_numeric(frame[left], errors="coerce").fillna(0.0)
            * pd.to_numeric(frame[right], errors="coerce").fillna(0.0)
        )
        names.append(name)
    return tuple(names)


def feature_columns(frame: pd.DataFrame, arm: str, side: str) -> tuple[str, ...]:
    if arm not in FEATURE_ARMS or side not in SIDES:
        raise ValueError("unknown feature arm or side")
    pit = tuple(frame.attrs.get("pit_feature_columns", ()))
    if pit:
        features: tuple[str, ...] = (*pit, *SCORE_CONTEXT)
    else:
        features = (
            "base_oof_score", *SCORE_CONTEXT,
            *(BASE_LONG if side == "long" else BASE_SHORT),
        )
    if "transition" in arm:
        features += transition_features(frame)
    if "health" in arm:
        features += HEALTH_FEATURES
    if arm.endswith("interactions"):
        features += interaction_features(frame)
    features = tuple(dict.fromkeys(features))
    validate_feature_names(features)
    missing = [name for name in features if name not in frame]
    if missing:
        raise ValueError(f"feature arm lacks columns: {missing}")
    return features


def build_task_targets(frame: pd.DataFrame, active_tasks: Iterable[str]) -> dict[str, TaskTarget]:
    active = tuple(active_tasks)
    unknown = set(active).difference((*ECONOMIC_TASKS, *PATH_TASKS))
    if unknown:
        raise ValueError(f"unknown task targets: {sorted(unknown)}")
    net = pd.to_numeric(frame["execution_net_ev_12h"], errors="raise").to_numpy(float)
    gross = pd.to_numeric(frame["execution_gross_ev_12h"], errors="raise").to_numpy(float)
    mfe = pd.to_numeric(frame["execution_mfe_return_12h"], errors="raise").to_numpy(float)
    positive = net > 0.0
    negative = net < 0.0
    ones = np.ones(len(frame), dtype=bool)
    targets: dict[str, TaskTarget] = {
        "direct_net": TaskTarget(net, ones, "regression", DIRECT_WEIGHT),
        "opportunity": TaskTarget(positive.astype(float), ones, "binary", ECONOMIC_WEIGHTS["opportunity"]),
        "favorable_magnitude": TaskTarget(np.maximum(net, 0.0), positive, "regression", ECONOMIC_WEIGHTS["favorable_magnitude"]),
        "adverse_magnitude": TaskTarget(np.maximum(-net, 0.0), negative, "regression", ECONOMIC_WEIGHTS["adverse_magnitude"]),
        "exit_conversion_loss": TaskTarget(np.maximum(mfe - gross, 0.0), ones, "regression", ECONOMIC_WEIGHTS["exit_conversion_loss"]),
        "timeout": TaskTarget(frame["exit_is_timeout"].astype(float).to_numpy(), ones, "binary", ECONOMIC_WEIGHTS["timeout"]),
    }
    if set(active).intersection(PATH_TASKS):
        path_valid = frame["__path_auxiliary_target_valid__"].fillna(False).astype(bool).to_numpy()
        timing_valid = frame["__time_to_first_meaningful_mfe_target_valid__"].fillna(False).astype(bool).to_numpy()
        hit = frame["__meaningful_mfe_reached_12h__"].fillna(False).astype(bool).to_numpy()
        time = pd.to_numeric(frame["__time_to_first_meaningful_mfe_hours_12h__"], errors="coerce").to_numpy(float)
        targets.update({
            "path_meaningful_hit": TaskTarget(hit.astype(float), path_valid, "binary", PATH_WEIGHT),
            "path_peak_mfe": TaskTarget(pd.to_numeric(frame["__peak_mfe_atr_12h__"], errors="coerce").to_numpy(float), path_valid & hit, "regression", PATH_WEIGHT),
            "path_fast_hit_2h": TaskTarget((hit & (time <= 2.0)).astype(float), timing_valid, "binary", PATH_WEIGHT),
            "path_mae_if_hit": TaskTarget(pd.to_numeric(frame["__mae_before_1_5atr_mfe__"], errors="coerce").to_numpy(float), path_valid & hit, "regression", PATH_WEIGHT),
            "path_mae_if_no_hit": TaskTarget(pd.to_numeric(frame["__mae_until_horizon_if_no_1_5atr__"], errors="coerce").to_numpy(float), path_valid & ~hit, "regression", PATH_WEIGHT),
            "path_confirmed_adverse_trough": TaskTarget(pd.to_numeric(frame["__bars_to_confirmed_adverse_trough__"], errors="coerce").to_numpy(float), path_valid, "regression", PATH_WEIGHT),
            "path_future_slope": TaskTarget(pd.to_numeric(frame["__future_slope_atr_per_hour_12h__"], errors="coerce").to_numpy(float), path_valid, "regression", PATH_WEIGHT),
        })
    selected = {"direct_net": targets["direct_net"]}
    selected.update({name: targets[name] for name in active})
    return selected


def _masked_torch_loss(
    outputs: Mapping[str, Any],
    targets: Mapping[str, Any],
    masks: Mapping[str, Any],
    kinds: Mapping[str, str],
    weights: Mapping[str, float],
):
    import torch
    import torch.nn.functional as functional
    loss = torch.zeros((), dtype=next(iter(outputs.values())).dtype, device=next(iter(outputs.values())).device)
    parts = {}
    for name, prediction in outputs.items():
        mask = masks[name].bool()
        if not bool(mask.any()):
            parts[name] = torch.zeros_like(loss)
            continue
        if kinds[name] == "binary":
            value = functional.binary_cross_entropy_with_logits(prediction[mask], targets[name][mask])
        else:
            value = functional.smooth_l1_loss(prediction[mask], targets[name][mask], beta=1.0)
        parts[name] = value
        loss = loss + float(weights[name]) * value
    return loss, parts


def masked_multitask_loss(
    predictions: Mapping[str, Any],
    targets: Mapping[str, Any],
    masks: Mapping[str, Any],
    specs_or_kinds: Mapping[str, Any],
    weights: Mapping[str, float] | None = None,
):
    """Masked loss for both unit-scale numpy checks and torch training."""
    first = next(iter(predictions.values()))
    if isinstance(first, np.ndarray):
        total = 0.0
        specs = specs_or_kinds
        for name, prediction in predictions.items():
            mask = np.asarray(masks[name], dtype=bool)
            if not mask.any():
                continue
            actual = np.asarray(targets[name], dtype=float)[mask]
            estimate = np.asarray(prediction, dtype=float)[mask]
            spec = specs[name]
            if spec.kind == "binary":
                probability = np.clip(estimate, 1e-6, 1.0 - 1e-6)
                value = -np.mean(actual * np.log(probability) + (1.0 - actual) * np.log(1.0 - probability))
            else:
                error = np.abs(estimate - actual)
                value = np.mean(np.where(error < 1.0, 0.5 * error ** 2, error - 0.5))
            total += float(spec.weight) * float(value)
        return float(total)
    if weights is None:
        raise ValueError("torch masked loss requires explicit weights")
    return _masked_torch_loss(predictions, targets, masks, specs_or_kinds, weights)


def _preprocess(
    train: pd.DataFrame, evaluate: pd.DataFrame, features: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, dict[str, list[float]]]:
    train_x = train.loc[:, list(features)].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    eval_x = evaluate.loc[:, list(features)].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    train_x[~np.isfinite(train_x)] = np.nan
    eval_x[~np.isfinite(eval_x)] = np.nan
    median = np.nanmedian(train_x, axis=0)
    median[~np.isfinite(median)] = 0.0
    train_x = np.where(np.isnan(train_x), median, train_x)
    eval_x = np.where(np.isnan(eval_x), median, eval_x)
    mean = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale[scale < 1e-8] = 1.0
    train_x = ((train_x - mean) / scale).astype(np.float32)
    eval_x = ((eval_x - mean) / scale).astype(np.float32)
    return train_x, eval_x, {
        "median": median.tolist(), "mean": mean.tolist(), "scale": scale.tolist(),
    }


def _target_tensors(
    targets: Mapping[str, TaskTarget], indices: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, str], dict[str, float], dict[str, tuple[float, float]]]:
    values: dict[str, np.ndarray] = {}
    masks: dict[str, np.ndarray] = {}
    kinds: dict[str, str] = {}
    weights: dict[str, float] = {}
    scaling: dict[str, tuple[float, float]] = {}
    for name, spec in targets.items():
        mask = spec.mask[indices] & np.isfinite(spec.values[indices])
        raw = spec.values[indices].astype(float)
        if spec.kind == "regression":
            location = float(np.mean(raw[mask])) if mask.any() else 0.0
            scale = max(float(np.std(raw[mask])) if mask.any() else 1.0, 1e-6)
            value = (raw - location) / scale
            scaling[name] = (location, scale)
        else:
            value = raw
            scaling[name] = (0.0, 1.0)
        value[~np.isfinite(value)] = 0.0
        values[name] = value.astype(np.float32)
        masks[name] = mask
        kinds[name] = spec.kind
        weights[name] = spec.weight
    return values, masks, kinds, weights, scaling


def _apply_target_scaling(
    targets: Mapping[str, TaskTarget], indices: np.ndarray,
    scaling: Mapping[str, tuple[float, float]],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    values, masks = {}, {}
    for name, spec in targets.items():
        raw = spec.values[indices].astype(float)
        location, scale = scaling[name]
        value = (raw - location) / scale if spec.kind == "regression" else raw
        mask = spec.mask[indices] & np.isfinite(raw)
        value[~np.isfinite(value)] = 0.0
        values[name], masks[name] = value.astype(np.float32), mask
    return values, masks


def fit_shared_trunk(
    train: pd.DataFrame,
    evaluate: pd.DataFrame,
    *,
    features: Sequence[str],
    active_tasks: Sequence[str],
    seed: int,
    config: TrainConfig = TRAIN_CONFIG,
    fixed_epochs: int | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    class SharedUtilityNet(nn.Module):
        def __init__(self, inputs: int, task_names: Sequence[str]):
            super().__init__()
            layers = []
            previous = inputs
            for width in config.hidden:
                layers.extend((nn.Linear(previous, width), nn.ReLU(), nn.Dropout(config.dropout)))
                previous = width
            self.trunk = nn.Sequential(*layers)
            self.heads = nn.ModuleDict({name: nn.Linear(previous, 1) for name in task_names})

        def forward(self, values):
            shared = self.trunk(values)
            return {name: head(shared).squeeze(-1) for name, head in self.heads.items()}

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    configure_torch_runtime(torch)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tasks = build_task_targets(train, active_tasks)
    train_x, eval_x, preprocessing = _preprocess(train, evaluate, features)
    order = np.argsort(pd.to_datetime(train["__ts__"], utc=True).to_numpy(), kind="stable")
    if fixed_epochs is None:
        split = max(1, int(len(order) * (1.0 - config.inner_validation_fraction)))
        validation_indices = order[split:]
        cutoff = pd.to_datetime(train.iloc[validation_indices]["__ts__"], utc=True).min()
        fit_allowed = strict_train_mask(train, cutoff)
        fit_indices = order[:split]
        fit_indices = fit_indices[fit_allowed[fit_indices]]
        if len(fit_indices) < 100 or len(validation_indices) < 100:
            raise ValueError("insufficient inner temporal support")
    else:
        fit_indices = order
        validation_indices = np.array([], dtype=int)
    fit_values, fit_masks, kinds, weights, scaling = _target_tensors(tasks, fit_indices)
    validation_values, validation_masks = _apply_target_scaling(tasks, validation_indices, scaling)

    fit_tensor = torch.from_numpy(train_x[fit_indices])
    dataset_parts = [fit_tensor]
    task_names = list(tasks)
    for name in task_names:
        dataset_parts.extend((
            torch.from_numpy(fit_values[name]),
            torch.from_numpy(fit_masks[name]),
        ))
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(*dataset_parts), batch_size=config.batch_size,
        shuffle=True, generator=generator,
    )
    model = SharedUtilityNet(train_x.shape[1], task_names).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay,
    )
    best_state, best_loss, stale, best_epoch = None, float("inf"), 0, 0
    epochs = int(fixed_epochs or config.max_epochs)
    for epoch in range(epochs):
        model.train()
        for batch in loader:
            x_batch = batch[0].to(device)
            target_batch = {
                name: batch[1 + 2 * position].to(device)
                for position, name in enumerate(task_names)
            }
            mask_batch = {
                name: batch[2 + 2 * position].to(device)
                for position, name in enumerate(task_names)
            }
            optimizer.zero_grad(set_to_none=True)
            output = model(x_batch)
            loss, _ = masked_multitask_loss(output, target_batch, mask_batch, kinds, weights)
            loss.backward()
            optimizer.step()
        if fixed_epochs is not None:
            best_epoch = epoch + 1
            continue
        model.eval()
        with torch.no_grad():
            output = model(torch.from_numpy(train_x[validation_indices]).to(device))
            target_t = {name: torch.from_numpy(validation_values[name]).to(device) for name in task_names}
            mask_t = {name: torch.from_numpy(validation_masks[name]).to(device) for name in task_names}
            validation_loss, _ = masked_multitask_loss(output, target_t, mask_t, kinds, weights)
            current = float(validation_loss)
        if current < best_loss - 1e-5:
            best_loss, stale, best_epoch = current, 0, epoch + 1
            best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= config.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        raw = model(torch.from_numpy(eval_x).to(device))
    predictions = {}
    for name, tensor in raw.items():
        array = tensor.cpu().numpy().astype(float)
        if kinds[name] == "binary":
            array = 1.0 / (1.0 + np.exp(-np.clip(array, -40.0, 40.0)))
        else:
            location, scale = scaling[name]
            array = location + scale * array
        predictions[name] = array
    audit = {
        "features": list(features), "active_tasks": list(active_tasks),
        "task_weights": weights, "target_scaling": scaling,
        "preprocessing": preprocessing, "selected_epochs": int(max(best_epoch, 1)),
        "fit_rows": int(len(fit_indices)),
        "inner_validation_rows": int(len(validation_indices)),
        "inner_best_loss": best_loss if np.isfinite(best_loss) else None,
        "shared_trunk": list(config.hidden), "task_heads": task_names,
        "device": str(device),
    }
    return predictions, audit


def rank_ic(actual: Sequence[float], predicted: Sequence[float]) -> float:
    left = pd.Series(np.asarray(actual, dtype=float)).rank(method="average")
    right = pd.Series(np.asarray(predicted, dtype=float)).rank(method="average")
    if left.nunique() < 2 or right.nunique() < 2:
        return float("nan")
    return float(left.corr(right))


def tail_metrics(
    frame: pd.DataFrame, score: Sequence[float], *,
    split: str, feature_arm: str, task_arm: str, score_name: str = "raw_direct",
) -> list[dict[str, Any]]:
    rows = []
    net = frame["execution_net_ev_12h"].to_numpy(float)
    opportunity = net > 0.0
    for fraction in FRACTIONS:
        mask = stable_global_top_mask(frame, score, fraction)
        selected = frame.loc[mask]
        rows.append({
            "split": split, "feature_arm": feature_arm, "task_arm": task_arm,
            "score_name": score_name,
            "fraction": fraction, "selected_rows": int(mask.sum()),
            "mean_net_bps": float(net[mask].mean() * 10_000.0),
            "sum_net": float(net[mask].sum()),
            "positive_precision": float(opportunity[mask].mean()),
            "long_share": float(selected["side_name"].eq("long").mean()),
            "distinct_assets": int(selected["__symbol__"].nunique()),
        })
    return rows


def _candidate_selection_score(rows: Sequence[Mapping[str, Any]]) -> float:
    table = pd.DataFrame(rows)
    core = table.loc[table["fraction"].isin((0.05, 0.10, 0.20))]
    if len(core) != 3:
        raise ValueError("selection score lacks the predeclared tail depths")
    side_ok = table.loc[table["fraction"].eq(0.10), "long_share"].iloc[0]
    if not 0.05 <= side_ok <= 0.95:
        return -1.0e9
    return float(core["mean_net_bps"].mean())


def apply_causal_recent_mapping(
    history_and_current: pd.DataFrame,
    *,
    score_column: str = "direct_net_score",
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    from scripts.run_execution_ev_recent_mapping_ablation import causal_mappings

    required = {
        *IDENTITY, "__decision_ts__", "execution_label_end_utc",
        "execution_net_ev_12h", score_column,
    }
    missing = required.difference(history_and_current.columns)
    if missing:
        raise ValueError(f"causal mapping stream lacks {sorted(missing)}")
    stream = history_and_current.copy()
    stream["execution_decision_utc"] = pd.to_datetime(stream["__decision_ts__"], utc=True)
    mapped, audit = causal_mappings(
        stream, score_col=score_column, window_days=21,
        min_reference_rows=500, side_support_target=500.0,
    )
    return mapped, audit


def run(args: argparse.Namespace) -> Path:
    if args.output.exists():
        raise FileExistsError(f"immutable output already exists: {args.output}")
    frame, population_audit = materialize_population(
        panel_root=args.panel_root, context_root=args.context_root,
        label_root=args.path_label_root, active_path=args.active,
        destination_path=args.destination, hazard_path=args.hazard,
        bocpd_path=args.bocpd, health_path=args.health,
    )
    calendar = split_calendar(frame)
    counts = {name: int(mask.sum()) for name, mask in calendar.items()}
    if len(frame) == 205_194 and counts != {
        "february": 64_512, "march": 71_424, "april": 69_258,
    }:
        raise ValueError(f"unexpected calendar counts: {counts}")
    signal = pd.to_datetime(frame["__ts__"], utc=True)
    feb_oof_train = signal.lt(FEBRUARY_MID).to_numpy() & strict_train_mask(frame, FEBRUARY_MID)
    feb_oof_eval = signal.ge(FEBRUARY_MID).to_numpy() & signal.lt(MARCH_START).to_numpy()
    feb_train = calendar["february"] & strict_train_mask(frame, MARCH_START)
    march_predict = calendar["march"]
    march_select = calendar["march"] & strict_train_mask(frame, APRIL_START)
    final_train = (calendar["february"] | calendar["march"]) & strict_train_mask(frame, APRIL_START)
    april_eval = calendar["april"]

    # Bounded matrix: task-role ablations use the frozen base feature block;
    # context add/drop arms use the complete economic multi-task contract.
    candidates = [
        ("base", task_arm) for task_arm in TASK_ARMS
    ] + [
        (feature_arm, "economic_multitask")
        for feature_arm in FEATURE_ARMS if feature_arm != "base"
    ]
    if args.candidate_limit is not None:
        candidates = candidates[: int(args.candidate_limit)]
    if args.plan_only:
        print(json.dumps(json_safe({
            "population": population_audit, "calendar_rows": counts,
            "resolved_february_oof_train_rows": int(feb_oof_train.sum()),
            "february_oof_prediction_rows": int(feb_oof_eval.sum()),
            "resolved_february_train_rows": int(feb_train.sum()),
            "march_prediction_rows": int(march_predict.sum()),
            "resolved_march_selection_rows": int(march_select.sum()),
            "resolved_final_train_rows": int(final_train.sum()),
            "april_diagnostic_rows": int(april_eval.sum()),
            "candidate_arms": candidates,
            "maximum_selection_fits": len(candidates) * len(SIDES) * 2,
            "maximum_final_fits": len(SIDES),
        }), indent=2))
        return args.output

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(dir=args.output.parent, prefix=f".{args.output.name}."))
    try:
        print(f"loaded population rows={len(frame)} candidates={len(candidates)}", flush=True)
        march_predictions, march_tails = [], []
        candidate_mapping_streams: dict[str, pd.DataFrame] = {}
        mapping_audit: dict[str, Any] = {}
        fit_audit: dict[str, Any] = {}
        feb_oof_rows = frame.loc[feb_oof_eval].copy().reset_index(drop=True)
        march_rows = frame.loc[march_predict].copy().reset_index(drop=True)
        for candidate_index, (feature_arm, task_arm) in enumerate(candidates):
            print(
                f"selection candidate {candidate_index + 1}/{len(candidates)} "
                f"{feature_arm}::{task_arm}",
                flush=True,
            )
            feb_direct = np.full(len(feb_oof_rows), np.nan)
            direct = np.full(len(march_rows), np.nan)
            auxiliary: dict[str, np.ndarray] = {}
            candidate_audit = {}
            for side_index, side in enumerate(SIDES):
                features = feature_columns(frame, feature_arm, side)
                early_train_side = feb_oof_train & frame["side_name"].eq(side).to_numpy()
                early_eval_side = feb_oof_eval & frame["side_name"].eq(side).to_numpy()
                early_eval_local = feb_oof_rows["side_name"].eq(side).to_numpy()
                early_predictions, early_audit = fit_shared_trunk(
                    frame.loc[early_train_side].reset_index(drop=True),
                    frame.loc[early_eval_side].reset_index(drop=True),
                    features=features, active_tasks=TASK_ARMS[task_arm],
                    seed=args.seed + candidate_index * 1000 + side_index,
                )
                feb_direct[early_eval_local] = select_rank_score(early_predictions)

                train_side = feb_train & frame["side_name"].eq(side).to_numpy()
                eval_side_global = march_predict & frame["side_name"].eq(side).to_numpy()
                eval_side_local = march_rows["side_name"].eq(side).to_numpy()
                predictions, audit = fit_shared_trunk(
                    frame.loc[train_side].reset_index(drop=True),
                    frame.loc[eval_side_global].reset_index(drop=True),
                    features=features, active_tasks=TASK_ARMS[task_arm],
                    seed=args.seed + candidate_index * 1000 + 100 + side_index,
                )
                direct[eval_side_local] = select_rank_score(predictions)
                for name, values in predictions.items():
                    auxiliary.setdefault(name, np.full(len(march_rows), np.nan))
                    auxiliary[name][eval_side_local] = values
                candidate_audit[side] = {
                    "february_oof": early_audit, "march": audit,
                }
            if not np.isfinite(feb_direct).all() or not np.isfinite(direct).all():
                raise ValueError("chronological direct score contains gaps")
            history = feb_oof_rows.loc[:, [
                *IDENTITY, "__decision_ts__", "execution_label_end_utc",
                "execution_net_ev_12h",
            ]].copy()
            history["direct_net_score"] = feb_direct
            history["evaluation_block"] = "february_oof"
            current = march_rows.loc[:, [
                *IDENTITY, "__decision_ts__", "execution_label_end_utc",
                "execution_net_ev_12h",
            ]].copy()
            current["direct_net_score"] = direct
            current["evaluation_block"] = "march_oos"
            mapped_stream, map_audit = apply_causal_recent_mapping(
                pd.concat([history, current], ignore_index=True),
            )
            candidate_key = f"{feature_arm}::{task_arm}"
            candidate_mapping_streams[candidate_key] = mapped_stream
            mapping_audit[candidate_key] = map_audit
            mapped_march = mapped_stream.loc[
                mapped_stream["evaluation_block"].eq("march_oos")
            ].reset_index(drop=True)
            resolved_selection = pd.to_datetime(
                mapped_march["execution_label_end_utc"], utc=True,
            ).lt(APRIL_START).to_numpy()
            if not np.isfinite(
                mapped_march.loc[resolved_selection, "causal_recent_side_isotonic_ev"]
            ).all():
                raise ValueError("March causal mapping lacks eligible selection rows")
            for score_name, score in (
                ("raw_direct", mapped_march.loc[resolved_selection, "direct_net_score"]),
                ("causal_recent_side_isotonic_ev", mapped_march.loc[resolved_selection, "causal_recent_side_isotonic_ev"]),
            ):
                march_tails.extend(tail_metrics(
                    mapped_march.loc[resolved_selection].reset_index(drop=True),
                    score.to_numpy(float), split="march_selection",
                    feature_arm=feature_arm, task_arm=task_arm,
                    score_name=score_name,
                ))
            output = march_rows.loc[:, [*IDENTITY, "__decision_ts__", "execution_label_end_utc", "execution_net_ev_12h"]].copy()
            output["feature_arm"] = feature_arm
            output["task_arm"] = task_arm
            output["direct_net_score"] = direct
            output["causal_recent_side_isotonic_ev"] = mapped_march[
                "causal_recent_side_isotonic_ev"
            ].to_numpy(float)
            for name, values in auxiliary.items():
                if name != "direct_net":
                    output[f"diagnostic__{name}"] = values
            march_predictions.append(output)
            fit_audit[f"{feature_arm}::{task_arm}"] = candidate_audit

        tail_table = pd.DataFrame(march_tails)
        mapped_tail_table = tail_table.loc[
            tail_table["score_name"].eq("causal_recent_side_isotonic_ev")
        ]
        score_table = (
            mapped_tail_table.groupby(["feature_arm", "task_arm"], sort=False)
            .apply(lambda group: _candidate_selection_score(group.to_dict("records")), include_groups=False)
            .rename("selection_score")
            .reset_index()
            .sort_values(["selection_score", "feature_arm", "task_arm"], ascending=[False, True, True], kind="stable")
        )
        winner = score_table.iloc[0]
        winning_feature, winning_task = str(winner["feature_arm"]), str(winner["task_arm"])

        april_rows = frame.loc[april_eval].copy().reset_index(drop=True)
        april_direct = np.full(len(april_rows), np.nan)
        april_auxiliary: dict[str, np.ndarray] = {}
        final_audit = {}
        for side_index, side in enumerate(SIDES):
            print(f"final winner fit side={side}", flush=True)
            train_side = final_train & frame["side_name"].eq(side).to_numpy()
            eval_side_global = april_eval & frame["side_name"].eq(side).to_numpy()
            eval_side_local = april_rows["side_name"].eq(side).to_numpy()
            features = feature_columns(frame, winning_feature, side)
            # Epoch count is the median of the frozen February inner selections
            # for this winning candidate; April never chooses it.
            selected_epochs = int(np.median([
                fit_audit[f"{winning_feature}::{winning_task}"][name]["march"]["selected_epochs"]
                for name in SIDES
            ]))
            predictions, audit = fit_shared_trunk(
                frame.loc[train_side].reset_index(drop=True),
                frame.loc[eval_side_global].reset_index(drop=True),
                features=features, active_tasks=TASK_ARMS[winning_task],
                seed=args.seed + 100_000 + side_index,
                fixed_epochs=max(1, selected_epochs),
            )
            april_direct[eval_side_local] = select_rank_score(predictions)
            for name, values in predictions.items():
                april_auxiliary.setdefault(name, np.full(len(april_rows), np.nan))
                april_auxiliary[name][eval_side_local] = values
            final_audit[side] = audit
        if not np.isfinite(april_direct).all():
            raise ValueError("April direct score contains gaps")
        april_output = april_rows.loc[:, [
            *IDENTITY, "__decision_ts__", "execution_label_end_utc",
            "execution_net_ev_12h", "execution_gross_ev_12h",
            "execution_cost_return", "base_oof_score",
        ]].copy()
        april_output["feature_arm"] = winning_feature
        april_output["task_arm"] = winning_task
        april_output["direct_net_score"] = april_direct
        for name, values in april_auxiliary.items():
            if name != "direct_net":
                april_output[f"diagnostic__{name}"] = values
        winning_key = f"{winning_feature}::{winning_task}"
        prior_stream = candidate_mapping_streams[winning_key]
        april_map_input = april_output.loc[:, [
            *IDENTITY, "__decision_ts__", "execution_label_end_utc",
            "execution_net_ev_12h", "direct_net_score",
        ]].copy()
        april_map_input["evaluation_block"] = "april_diagnostic"
        april_mapped_stream, april_map_audit = apply_causal_recent_mapping(
            pd.concat([prior_stream, april_map_input], ignore_index=True),
        )
        mapped_april = april_mapped_stream.loc[
            april_mapped_stream["evaluation_block"].eq("april_diagnostic")
        ].reset_index(drop=True)
        april_output["causal_recent_side_isotonic_ev"] = mapped_april[
            "causal_recent_side_isotonic_ev"
        ].to_numpy(float)
        april_tails = []
        for score_name in ("direct_net_score", "causal_recent_side_isotonic_ev"):
            april_tails.extend(tail_metrics(
                april_output, april_output[score_name].to_numpy(float),
                split="april_reused_diagnostic",
                feature_arm=winning_feature, task_arm=winning_task,
                score_name="raw_direct" if score_name == "direct_net_score" else score_name,
            ))

        pd.concat(march_predictions, ignore_index=True).to_parquet(
            temporary / "march_selection_predictions.parquet", index=False, compression="zstd",
        )
        april_output.to_parquet(
            temporary / "april_reused_diagnostic_predictions.parquet", index=False, compression="zstd",
        )
        pd.concat([tail_table, pd.DataFrame(april_tails)], ignore_index=True).to_parquet(
            temporary / "tail_metrics.parquet", index=False,
        )
        score_table.to_parquet(temporary / "march_candidate_ranking.parquet", index=False)
        manifest = {
            "schema": SCHEMA,
            "status": "COMPLETED_REUSED_APRIL_DIAGNOSTIC_NOT_PROMOTION_EVIDENCE",
            "population": population_audit,
            "calendar": {
                "rows": counts,
                "february_role": "training_and_inner_temporal_epoch_selection",
                "march_role": "predeclared_arm_selection",
                "april_role": "scored_once_after_winner_freeze_reused_diagnostic",
                "february_oof_train_rows": int(feb_oof_train.sum()),
                "february_oof_prediction_rows": int(feb_oof_eval.sum()),
                "february_resolved_train_rows": int(feb_train.sum()),
                "march_prediction_rows": int(march_predict.sum()),
                "march_resolved_selection_rows": int(march_select.sum()),
                "final_resolved_train_rows": int(final_train.sum()),
            },
            "target": {
                "primary": "exact_1m_deployed_policy_execution_net_ev_12h",
                "resolution": "decision_plus_12h_strictly_before_cutoff",
                "cost": "gross_minus_cost_equals_net_cost_subtracted_once",
            },
            "architecture": {
                "side_local": True, "shared_raw_feature_trunk": list(TRAIN_CONFIG.hidden),
                "task_specific_heads": True, "sole_rank_output": "direct_net",
                "no_algebraic_auxiliary_composition": True,
                "task_arms": {name: list(tasks) for name, tasks in TASK_ARMS.items()},
                "task_weights": {
                    "direct": DIRECT_WEIGHT, "economic_auxiliary": ECONOMIC_WEIGHTS,
                    "hourly_path_regularizer": PATH_WEIGHT,
                },
                "hourly_path_caveat": "causal 12h supporting labels at hourly geometry; low-weight regularizers only, not exact-policy components",
            },
            "features": {
                "arms": list(FEATURE_ARMS),
                "dae_gmm": "excluded from default base; separate future add/drop only",
                "transition": "chronological research context; pooled upstream geometry prevents production promotion",
                "hazard": "excluded: available hazard artifacts are grouped OOF with pooled upstream geometry, not chronological execution inputs",
                "health": "historical raw-alpha lineage only; current 2026 health is not joined",
                "forbidden_feature_scan": "passed",
            },
            "selection": {
                "winner": {"feature_arm": winning_feature, "task_arm": winning_task},
                "criterion": "mean March causal-recent-side-isotonic pooled-global top-5/10/20 net bps with 5%-95% side-share gate",
                "ranking_scope": "one pooled global book across sides and timestamps",
                "fractions": list(FRACTIONS), "tie_break": "candidate_id ascending",
                "no_timestamp_or_side_quota": True,
                "causal_mapping": {
                    "window_days": 21, "minimum_reference_rows": 500,
                    "side_support_target": 500,
                    "reference_contract": "execution_label_end_utc strictly before UTC-day snapshot",
                    "march_audit": mapping_audit,
                    "april_winner_audit": april_map_audit,
                },
            },
            "fit_audit": fit_audit, "final_fit_audit": final_audit,
            "training_geometry": asdict(TRAIN_CONFIG),
            "seed": args.seed,
            "runner_sha256": sha256_file(Path(__file__).resolve()),
        }
        outputs = {
            str(path.relative_to(temporary)): sha256_file(path)
            for path in sorted(temporary.iterdir()) if path.is_file()
        }
        manifest["outputs_sha256"] = outputs
        manifest_path = temporary / "manifest.json"
        manifest_path.write_text(json.dumps(json_safe(manifest), indent=2, sort_keys=True, allow_nan=False) + "\n")
        (temporary / "manifest.sha256").write_text(f"{sha256_file(manifest_path)}  manifest.json\n")
        os.replace(temporary, args.output)
        return args.output
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-root", type=Path, default=DEFAULT_PANEL_ROOT)
    parser.add_argument("--context-root", type=Path, default=DEFAULT_CONTEXT_ROOT)
    parser.add_argument("--path-label-root", type=Path, default=DEFAULT_PATH_LABEL_ROOT)
    parser.add_argument("--health", type=Path, default=DEFAULT_HEALTH)
    parser.add_argument("--active", type=Path, default=DEFAULT_ACTIVE)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument("--hazard", type=Path, default=DEFAULT_HAZARD)
    parser.add_argument("--bocpd", type=Path, default=DEFAULT_BOCPD)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--candidate-limit", type=int)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args(argv)
    if args.candidate_limit is not None and args.candidate_limit < 1:
        parser.error("--candidate-limit must be positive")
    return args


if __name__ == "__main__":
    run(parse_args())
