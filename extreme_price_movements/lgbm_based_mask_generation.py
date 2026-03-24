from __future__ import annotations

import collections
import glob
import hashlib
import itertools
import json
import logging
import multiprocessing as mp
import os
import pickle
import re
import time
import traceback
from dataclasses import dataclass, field, replace
from math import sqrt
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import scipy.stats
from lightgbm import LGBMRegressor
from numba import njit, prange
from sklearn.metrics import roc_auc_score

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "3")

from extreme_price_movements.config import (
    CFG,
    CONTINUOUS_LOCATION_COLS,
    LOC_CONTINUOUS_FAMILY_MAP,
    CONTINUOUS_TRIGGER_COLS,
    RIDGE_FEATURE_COLS,
    RIDGE_FEATURE_META,
    TEST_FEATURE_KEYS,
)
from extreme_price_movements.data_store import (
    PartitionedOHLCVStore,
    load_features_selected,
    to_panel,
)
from extreme_price_movements.intraday_crypto_library import (
    INTRADAY_TRIGGER_COLUMNS,
    LOCATION_FILTER_COLUMNS,
)
from extreme_price_movements.periods_symbols_management import (
    EventSchema,
    SlicePlanner,
    SlicePlannerConfig,
)
from extreme_price_movements.universe import get_training_universe
from extreme_price_movements.utils import tprint
from extreme_price_movements.hpo_lgbm_regime_miner import run_short_hpo_for_target_horizon

LOGGER = logging.getLogger(__name__)

# =============================================================================
# TRIAD TARGET CONFIGURATION
# =============================================================================

# Default horizons for triad target training (in bars)
TRIAD_DEFAULT_HORIZONS: List[int] = [3, 10]

# Default triad target names
TRIAD_DEFAULT_TARGET_NAMES: List[str] = ["target_eff", "target_ela", "target_vame"]

# Per-target configuration for triad targets
TRIAD_TARGET_CONFIGS: Dict[str, Dict[str, Any]] = {
    "target_eff": {
        "huber_alpha": 1.0,
        "learning_rate": 0.03,
        "min_support_pct": 0.05,
        "ic_hurdle": 0.02,
        "description": "Efficiency: direct vs actual path ratio",
    },
    "target_ela": {
        "huber_alpha": 2.0,
        "learning_rate": 0.02,
        "min_support_pct": 0.04,
        "ic_hurdle": 0.015,
        "description": "Elasticity: reversion tendency at extremes",
    },
    "target_vame": {
        "huber_alpha": 0.5,
        "learning_rate": 0.04,
        "min_support_pct": 0.06,
        "ic_hurdle": 0.025,
        "description": "Volume-adjusted momentum efficiency",
    },
}

# Horizon-specific configuration multipliers
HORIZON_CONFIGS: Dict[int, Dict[str, Any]] = {
    3: {"min_data_in_leaf_multiplier": 0.8, "description": "Very short-term"},
    10: {"min_data_in_leaf_multiplier": 1.0, "description": "Short-term"},
}

SCORER_REGISTRY_COLUMNS: List[str] = [
    "canonical_key",
    "mean_net_ret",
    "directional_mean_ret",
    "std_net_ret",
    "mean_within_fold_std",
    "mean_support_pct",
    "std_support_pct",
    "presence_freq",
    "presence_freq_units",
    "sign_consistency",
    "min_support_actual",
    "mean_uplift",
    "mean_baseline_ret",
    "mean_oos_ic",
    "p25_oos_ic",
    "p50_oos_ic",
    "p75_oos_ic",
    "mean_delta_ic",
    "positive_ic_fraction",
    "composite_score",
    "required_hurdle",
    "hurdle_excess",
    "n_folds",
    "discovery_count",
    "n_instances",
    "display_arity",
    "structural_depth",
    "pipeline_stage",
    "parent_context_key",
    "side",
    "rule_type",
    "accepted",
    "rejection_reason",
    "dominant_parent_key",
    "dominated_by_parent",
]


def _with_expected_columns(
    df: Optional[pd.DataFrame], expected_columns: Sequence[str]
) -> pd.DataFrame:
    if df is None:
        return pd.DataFrame(columns=list(expected_columns))
    work = df.copy()
    for column in expected_columns:
        if column not in work.columns:
            work[column] = pd.Series([np.nan] * len(work), index=work.index)
    ordered = list(expected_columns) + [
        column for column in work.columns if column not in expected_columns
    ]
    return work.loc[:, ordered]


def atomic_to_csv(
    df: Optional[pd.DataFrame],
    output_path: Path,
    expected_columns: Optional[Sequence[str]] = None,
    index: bool = False,
) -> pd.DataFrame:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prepared = (
        _with_expected_columns(df, expected_columns)
        if expected_columns is not None
        else (pd.DataFrame() if df is None else df.copy())
    )
    with NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        delete=False,
    ) as tmp_file:
        tmp_path = Path(tmp_file.name)
        prepared.to_csv(tmp_file, index=index)
    os.replace(tmp_path, output_path)
    return prepared


def build_run_output_dir(
    cfg: Dict[str, Any],
    target_name: Optional[str] = None,
    horizon: Optional[int] = None,
    side: Optional[str] = None,
    stage: Optional[str] = None,
) -> Path:
    """
    Build output directory path for run artifacts.

    Parameters
    ----------
    cfg : Dict[str, Any]
        Configuration dictionary containing 'output_dir' and 'timestamped_run_outputs'
    target_name : Optional[str]
        Target name for triad mode (e.g., 'target_eff', 'target_ela', 'target_vame')
    horizon : Optional[int]
        Horizon in bars for triad mode
    side : Optional[str]
        Side ('long' or 'short') for stage-specific paths
    stage : Optional[str]
        Stage name (e.g., 'stage_a_context', 'stage_b_trigger_refinement')

    Returns
    -------
    Path
        Output directory path

    Notes
    -----
    When target_name and horizon are provided, creates a triad-style path:
        base_dir/h{horizon}/{target_name}/{side}/{stage}
    Otherwise uses the legacy path structure.
    """
    base_output_dir = Path(cfg.get("output_dir", "./lgbm_outputs"))
    base_output_dir.mkdir(parents=True, exist_ok=True)

    # Determine if we should use timestamped run directory
    if not bool(cfg.get("timestamped_run_outputs", True)):
        run_base = base_output_dir
    else:
        timestamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%d_%H%M%S")
        run_base = base_output_dir / f"run_{timestamp}"
        suffix = 1
        while run_base.exists():
            suffix += 1
            run_base = base_output_dir / f"run_{timestamp}_{suffix:02d}"

    # Build path based on triad vs legacy mode
    if target_name is not None and horizon is not None:
        # Triad mode: base_dir/h{horizon}/{target_name}/...
        run_output_dir = run_base / f"h{horizon}" / target_name
        if side is not None:
            run_output_dir = run_output_dir / side
        if stage is not None:
            run_output_dir = run_output_dir / stage
    else:
        # Legacy mode
        run_output_dir = run_base

    # Only create directory if this is the root call (no side/stage specified)
    if side is None and stage is None:
        run_output_dir.mkdir(parents=True, exist_ok=True)

    return run_output_dir


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3:
        return np.nan
    # Check for constant arrays
    if np.all(x == x[0]) or np.all(y == y[0]):
        return np.nan
    return float(scipy.stats.spearmanr(x, y).correlation)


def _clip_returns(x: np.ndarray) -> np.ndarray:
    if len(x) == 0:
        return x
    lo = float(np.nanpercentile(x, 2.0))
    hi = float(np.nanpercentile(x, 98.0))
    return np.clip(x, lo, hi)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.nan_to_num(a, 0.0)
    b = np.nan_to_num(b, 0.0)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a_valid = np.isfinite(a)
    b_valid = np.isfinite(b)
    valid = a_valid & b_valid
    if np.sum(valid) < 3:
        return np.nan
    # Check for constant arrays
    if np.all(a[valid] == a[valid][0]) or np.all(b[valid] == b[valid][0]):
        return np.nan
    return float(np.corrcoef(a[valid], b[valid])[0, 1])


# =============================================================================
# DATA STRUCTURES & METADATA
# =============================================================================


@dataclass(frozen=True)
class FeatureMetadata:
    feature_name: str
    feature_index: int
    group: str  # 'trigger', 'location', 'regime'
    source_name: str
    source_family: str
    source_type: str  # 'boolean', 'continuous'
    booleanization_method: Optional[str] = None
    threshold_type: Optional[str] = None
    threshold_value: Optional[float] = None
    description: str = ""
    regime_family: Optional[str] = None

    @property
    def interaction_group(self) -> str:
        if self.group == "location":
            return "location"
        elif self.group == "regime":
            return (
                f"regime:{self.regime_family}"
                if self.regime_family
                else "regime:unknown"
            )
        return self.group


@dataclass(frozen=True)
class MiningStageSpec:
    stage_name: str
    active_groups: Tuple[str, ...]  # e.g. ("regime", "location")
    allow_groups_in_rule: Tuple[str, ...]  # same as above
    output_dir_name: str  # e.g. "stage_a_context"
    allowed_group_pairs: Tuple[Tuple[str, str], ...]
    slot_order: Tuple[str, ...] = ("trigger", "location", "regime")
    context_rule_keys: Optional[List[str]] = None  # used only in Stage B
    use_context_features: bool = False
    context_feature_group_name: str = "context"
    require_uplift: bool = False


@dataclass(frozen=True)
class RuleCondition:
    feature_name: str
    feature_index: int
    group: str
    normalized_value: int  # 1 (feature==1) or 0 (feature==0)
    raw_operator: str  # '<=', '>', '==', etc.
    raw_threshold: float
    raw_decision_type: Optional[str] = None
    default_left: Optional[bool] = None
    missing_type: Optional[str] = None

    def __repr__(self):
        val_str = "==1" if self.normalized_value == 1 else "==0"
        return f"{self.group}:{self.feature_name}{val_str}"


@dataclass
class ExtractedRule:
    rule_id: str  # Instance-specific ID
    canonical_key: str  # Slot-based identity
    conditions: List[RuleCondition]
    model_id: str
    fold_id: int
    seed: int
    tree_index: int
    leaf_index: int
    leaf_value: float
    support_train: int
    support_val: int = 0
    source_target: str = "fwd_ret_norm"  # Target name provenance
    source_horizon: int = 0  # Horizon in bars provenance


# =============================================================================
# FEATURE PREPARATION
# =============================================================================


class FeatureProcessor:
    def __init__(self):
        self.metadata: Dict[str, FeatureMetadata] = {}
        self.feature_names: List[str] = []
        self.rank_audit_rows = []
        self.bool_support_audit_rows = []

    def prepare_features(
        self,
        feature_dict: Dict[str, np.ndarray],
        timestamps: np.ndarray,
        symbol_codes: np.ndarray,
        cfg: Dict[str, Any],
        active_groups: Optional[Sequence[str]] = None,
        extra_binary_features: Optional[Dict[str, np.ndarray]] = None,
        extra_feature_group: str = "context",
    ) -> Tuple[np.ndarray, List[FeatureMetadata], Dict[str, pd.DataFrame]]:
        """
        Groups and booleanizes features with quality hardening.
        """
        raw_cols = []
        raw_names = []
        self.metadata = {}
        self.feature_names = []

        # If active_groups is None, default to all primary groups
        if active_groups is None:
            active_groups = ("trigger", "location", "regime")

        raw_source_features_by_group = collections.defaultdict(set)

        def _add_continuous_features_as_booleans(sources, group_name):
            min_support = int(cfg.get("min_feature_support", 10))
            n_samples = len(timestamps)

            for src in sources:
                if src in feature_dict:
                    raw_source_features_by_group[group_name].add(src)
                    raw_arr = feature_dict[src]
                    nan_rate_before = float(np.isnan(raw_arr).mean())

                    ts_ranks = self._compute_ts_ranks(raw_arr, symbol_codes)

                    nan_rate_ts = float(np.isnan(ts_ranks).mean())

                    self.rank_audit_rows.append(
                        {
                            "source_feature": src,
                            "group": group_name,
                            "nan_rate_before": nan_rate_before,
                            "nan_rate_ts": nan_rate_ts,
                        }
                    )

                    if group_name == "location":
                        family = LOC_CONTINUOUS_FAMILY_MAP.get(src, "context")
                    else:
                        family = src.split("_")[0] if "_" in src else group_name

                    # NaN-preserving booleanization.
                    # ts_ranks is NaN wherever the raw feature was NaN for that symbol.
                    # We must propagate NaN through to the boolean column so that
                    # filter_complete_feature_rows correctly drops rows missing feature data.
                    nan_mask = np.isnan(ts_ranks)

                    def _bool_float(condition: np.ndarray) -> np.ndarray:
                        out = condition.astype(np.float32)
                        out[nan_mask] = np.nan
                        return out

                    n_valid = int((~nan_mask).sum())

                    for q in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
                        # Top quantiles (>= q)
                        bool_name_top = f"{group_name[:3]}_{src}_ts_top{int(q*100)}"
                        bool_arr_top = _bool_float(ts_ranks >= q)

                        support_top = int(np.nansum(bool_arr_top))
                        support_top_pct = support_top / n_valid if n_valid > 0 else 0

                        self.bool_support_audit_rows.append(
                            {
                                "generated_boolean": bool_name_top,
                                "group": group_name,
                                "source_feature": src,
                                "support": support_top,
                                "support_pct": support_top_pct,
                            }
                        )

                        if support_top < min_support or support_top_pct > 0.95:
                            tprint(
                                f"WARNING: generated boolean {bool_name_top} has extreme support ({support_top}, {support_top_pct:.2%})"
                            )

                        self._add_metadata(
                            bool_name_top,
                            group_name,
                            "boolean",
                            source_name=src,
                            source_family=family,
                            booleanization_method="ts_rank",
                            threshold_type="top_quantile",
                            threshold_value=q,
                            description=f"TS Rank >= {q}",
                        )
                        raw_cols.append(bool_arr_top)
                        raw_names.append(bool_name_top)

                        # Bottom quantiles (<= q)
                        bool_name_bot = f"{group_name[:3]}_{src}_ts_bot{int(q*100)}"
                        bool_arr_bot = _bool_float(ts_ranks <= q)

                        support_bot = int(np.nansum(bool_arr_bot))
                        support_bot_pct = support_bot / n_valid if n_valid > 0 else 0

                        self.bool_support_audit_rows.append(
                            {
                                "generated_boolean": bool_name_bot,
                                "group": group_name,
                                "source_feature": src,
                                "support": support_bot,
                                "support_pct": support_bot_pct,
                            }
                        )

                        if support_bot < min_support or support_bot_pct > 0.95:
                            tprint(
                                f"WARNING: generated boolean {bool_name_bot} has extreme support ({support_bot}, {support_bot_pct:.2%})"
                            )

                        self._add_metadata(
                            bool_name_bot,
                            group_name,
                            "boolean",
                            source_name=src,
                            source_family=family,
                            booleanization_method="ts_rank",
                            threshold_type="bot_quantile",
                            threshold_value=q,
                            description=f"TS Rank <= {q}",
                        )
                        raw_cols.append(bool_arr_bot)
                        raw_names.append(bool_name_bot)

                    # Expanded median bands
                    for q_band in [0.20, 0.25, 0.30, 0.40]:
                        q_band_upper = 1.0 - q_band
                        band_name = f"{group_name[:3]}_{src}_ts_band{int(q_band*100)}_{int(q_band_upper*100)}"
                        band_arr = _bool_float(
                            (ts_ranks >= q_band) & (ts_ranks <= q_band_upper)
                        )

                        support_band = int(np.nansum(band_arr))
                        support_band_pct = support_band / n_valid if n_valid > 0 else 0

                        self.bool_support_audit_rows.append(
                            {
                                "generated_boolean": band_name,
                                "group": group_name,
                                "source_feature": src,
                                "support": support_band,
                                "support_pct": support_band_pct,
                            }
                        )
                        if support_band < min_support or support_band_pct > 0.95:
                            tprint(
                                f"WARNING: generated boolean {band_name} has extreme support ({support_band}, {support_band_pct:.2%})"
                            )

                        self._add_metadata(
                            band_name,
                            group_name,
                            "boolean",
                            source_name=src,
                            source_family=family,
                            booleanization_method="ts_rank",
                            threshold_type="band_quantile",
                            threshold_value=0.50,
                            description=f"TS Rank inside {int(q_band*100)}-{int(q_band_upper*100)} band",
                        )
                        raw_cols.append(band_arr)
                        raw_names.append(band_name)

        # 1. Trigger Features — DISABLED (all trigger features removed from pipeline)
        if "trigger" in active_groups:
            tprint("TRIGGER features disabled — skipping trigger group.")

        # 2. Location Features
        if "location" in active_groups:
            # Continuous location features are the sole location source family.
            _add_continuous_features_as_booleans(CONTINUOUS_LOCATION_COLS, "location")

        # 3. Regime Features (continuous -> hybrid booleanize)
        if "regime" in active_groups:
            regime_sources = sorted(
                list(set(RIDGE_FEATURE_COLS) | set(TEST_FEATURE_KEYS))
            )
            _add_continuous_features_as_booleans(regime_sources, "regime")

        # 4. Extra Binary Features (e.g. Stage A Contexts)
        if extra_binary_features:
            for name, arr in extra_binary_features.items():
                raw_source_features_by_group[extra_feature_group].add(name)
                arr_f32 = arr.astype(np.float32)
                self._add_metadata(
                    name,
                    extra_feature_group,
                    "boolean",
                    source_name=name,
                    source_family=extra_feature_group,
                    description=f"Extra feature from {extra_feature_group}",
                )
                raw_cols.append(arr_f32)
                raw_names.append(name)

        if not raw_cols:
            return (
                np.empty((len(timestamps), 0)),
                [],
                pd.DataFrame(
                    columns=[
                        "feature_name",
                        "status",
                        "reason",
                        "support",
                        "group",
                        "regime_family",
                    ]
                ),
            )

        X_raw = np.column_stack(raw_cols)

        # Quality Hardening: Drop degenerate/duplicate columns
        X_clean, retained_names, audit_df = self._run_feature_quality_checks(
            X_raw, raw_names, cfg
        )

        if cfg.get("boolean_only", False):
            is_bool = np.array(
                [self.metadata[n].source_type == "boolean" for n in retained_names]
            )
            if np.any(is_bool):
                X_clean = X_clean[:, is_bool]
                retained_names = [n for i, n in enumerate(retained_names) if is_bool[i]]
            else:
                tprint("WARNING: boolean_only=True but no boolean features found!")

        audit_df["group"] = [self.metadata[n].group for n in audit_df["feature_name"]]
        audit_df["regime_family"] = [
            self.metadata[n].regime_family for n in audit_df["feature_name"]
        ]

        raw_source_counts = {k: len(v) for k, v in raw_source_features_by_group.items()}

        # Summary by group
        group_summary = []
        all_groups = list(active_groups) if active_groups else []
        if extra_feature_group and extra_feature_group not in all_groups:
            all_groups.append(extra_feature_group)

        for g in all_groups:
            g_df = audit_df[audit_df["group"] == g]
            if g_df.empty:
                continue
            retained = g_df[g_df["status"] == "retained"]
            dropped = g_df[g_df["status"] == "dropped"]
            drop_reasons = dropped["reason"].value_counts().to_dict()

            support_stats = (
                retained["support"].describe()
                if not retained.empty
                else pd.Series(dtype=float)
            )

            group_summary.append(
                {
                    "group": g,
                    "raw_source_features": raw_source_counts.get(g, 0),
                    "generated_booleans": len(g_df),
                    "retained": len(retained),
                    "dropped": len(dropped),
                    "drop_reason_all_zeros": drop_reasons.get("all_zeros", 0),
                    "drop_reason_all_ones": drop_reasons.get("all_ones", 0),
                    "drop_reason_low_support": sum(
                        v
                        for k, v in drop_reasons.items()
                        if k.startswith("low_support")
                    ),
                    "drop_reason_duplicate": sum(
                        v
                        for k, v in drop_reasons.items()
                        if k.startswith("duplicate_of")
                    ),
                    "support_min": support_stats.get("min", np.nan),
                    "support_p25": support_stats.get("25%", np.nan),
                    "support_median": support_stats.get("50%", np.nan),
                    "support_p75": support_stats.get("75%", np.nan),
                    "support_max": support_stats.get("max", np.nan),
                }
            )

        feature_quality_summary_by_group = pd.DataFrame(group_summary)

        # Summary by regime family
        regime_df = audit_df[audit_df["group"] == "regime"]
        regime_summary = []
        if not regime_df.empty:
            for fam, f_df in regime_df.groupby("regime_family"):
                retained = f_df[f_df["status"] == "retained"]
                dropped = f_df[f_df["status"] == "dropped"]
                regime_summary.append(
                    {
                        "regime_family": fam,
                        "generated_booleans": len(f_df),
                        "retained": len(retained),
                        "dropped": len(dropped),
                    }
                )
        feature_quality_summary_by_regime_family = pd.DataFrame(regime_summary)

        if not feature_quality_summary_by_regime_family.empty:
            retained_counts = feature_quality_summary_by_regime_family.set_index(
                "regime_family"
            )["retained"]
            total_retained_regime = retained_counts.sum()
            if total_retained_regime > 0:
                max_fam = retained_counts.idxmax()
                max_val = retained_counts.max()
                if max_val / total_retained_regime > 0.5:
                    tprint(
                        f"WARNING: Regime family '{max_fam}' dominates retained regime features ({max_val}/{total_retained_regime})."
                    )

        # tprints
        total_raw = sum(raw_source_counts.values())
        total_gen = len(audit_df)
        total_retained = len(retained_names)
        tprint(
            f"FeaturePrep: total raw={total_raw}, generated={total_gen}, retained={total_retained}"
        )

        for g_sum in group_summary:
            tprint(
                f"  - {g_sum['group']}: retained {g_sum['retained']} / {g_sum['generated_booleans']} generated (from {g_sum['raw_source_features']} raw)"
            )

        dropped_df = audit_df[audit_df["status"] == "dropped"]
        if not dropped_df.empty:
            top_dropped = dropped_df["reason"].value_counts().head(10)
            tprint("Top 10 dropped features by reason:")
            for reason, count in top_dropped.items():
                tprint(f"  - {reason}: {count}")

        # Rank Audit tprints
        if self.rank_audit_rows:
            rank_audit_df = pd.DataFrame(self.rank_audit_rows)
            rank_audit_df["worst_nan"] = rank_audit_df[
                ["nan_rate_before", "nan_rate_ts"]
            ].max(axis=1)
            top_nan = rank_audit_df.sort_values("worst_nan", ascending=False).head(10)
            tprint("Top 10 features with worst NaN rates:")
            for _, row in top_nan.iterrows():
                tprint(
                    f"  - {row['group']}:{row['source_feature']} -> before={row['nan_rate_before']:.2%}, ts={row['nan_rate_ts']:.2%}"
                )
        else:
            rank_audit_df = pd.DataFrame()

        if self.bool_support_audit_rows:
            bool_support_audit_df = pd.DataFrame(self.bool_support_audit_rows)
            n_samples = len(timestamps)
            bool_support_audit_df["usable_support"] = np.minimum(
                bool_support_audit_df["support"],
                n_samples - bool_support_audit_df["support"],
            )
            top_imbal = bool_support_audit_df.sort_values("usable_support").head(10)
            tprint("Top 10 generated booleans with lowest usable support:")
            for _, row in top_imbal.iterrows():
                tprint(
                    f"  - {row['generated_boolean']}: support={row['support']} ({row['support_pct']:.2%})"
                )
        else:
            bool_support_audit_df = pd.DataFrame()

        if not rank_audit_df.empty and not bool_support_audit_df.empty:
            booleanization_support_audit = pd.merge(
                bool_support_audit_df,
                rank_audit_df.drop(columns=["worst_nan"]),
                on=["source_feature", "group"],
                how="left",
            )
        else:
            booleanization_support_audit = bool_support_audit_df

        audits = {
            "feature_quality_audit": audit_df,
            "feature_quality_summary_by_group": feature_quality_summary_by_group,
            "feature_quality_summary_by_regime_family": feature_quality_summary_by_regime_family,
            "booleanization_support_audit": booleanization_support_audit,
        }

        # Re-index metadata based on final retained columns
        retained_metadata = []
        old_metadata = {m.feature_name: m for m in self.metadata.values()}
        for i, name in enumerate(retained_names):
            m = old_metadata[name]
            # Create fresh copy with updated index
            new_m = FeatureMetadata(
                feature_name=m.feature_name,
                feature_index=i,
                group=m.group,
                source_name=m.source_name,
                source_family=m.source_family,
                source_type=m.source_type,
                booleanization_method=m.booleanization_method,
                threshold_type=m.threshold_type,
                threshold_value=m.threshold_value,
                description=m.description,
                regime_family=m.regime_family,
            )
            retained_metadata.append(new_m)

        return X_clean, retained_metadata, audits

    def _add_metadata(self, name, group, src_type, **kwargs):
        idx = len(self.feature_names)
        self.feature_names.append(name)

        source_name = kwargs.get("source_name", name)
        regime_family = None
        if group == "regime":
            if source_name in RIDGE_FEATURE_META:
                regime_family = RIDGE_FEATURE_META[source_name].get("family")
            else:
                regime_family = kwargs.get("source_family", "unknown")

        self.metadata[name] = FeatureMetadata(
            feature_name=name,
            feature_index=idx,
            group=group,
            source_name=source_name,
            source_family=kwargs.get("source_family", "unknown"),
            source_type=src_type,
            booleanization_method=kwargs.get("booleanization_method"),
            threshold_type=kwargs.get("threshold_type"),
            threshold_value=kwargs.get("threshold_value"),
            description=kwargs.get("description", ""),
            regime_family=regime_family,
        )

    def _compute_ts_ranks(
        self, arr: np.ndarray, symbol_codes: np.ndarray
    ) -> np.ndarray:
        """
        Time-series ranking using pandas for vectorization speed.
        """
        s = pd.Series(arr, index=symbol_codes)
        # Fix root cause of 0.5 artifacts: if cross-section has no variance, return NaN
        def _rank_safe(g):
            if g.nunique() <= 1:
                return np.full(len(g), np.nan)
            return g.rank(pct=True)
        # Using transform for speed while preserving index
        ranks = s.groupby(level=0, sort=False).transform(_rank_safe)
        return ranks.values

    def _run_feature_quality_checks(
        self, X: np.ndarray, names: List[str], cfg: Dict[str, Any]
    ) -> Tuple[np.ndarray, List[str], pd.DataFrame]:
        """
        Drops degenerate or duplicate boolean columns.
        """
        n_samples = X.shape[0]
        min_support = int(cfg.get("min_feature_support", 10))

        audit_rows = []
        retained_indices = []
        retained_names = []
        hash_registry = {}  # To detect exact duplicates

        for i, name in enumerate(names):
            col = X[:, i]
            n_ones = np.sum(col == 1)
            n_zeros = np.sum(col == 0)

            dropped = False
            reason = "retained"

            if n_ones == 0:
                dropped = True
                reason = "all_zeros"
            elif n_zeros == 0:
                dropped = True
                reason = "all_ones"
            elif n_ones < min_support:
                dropped = True
                reason = f"low_support_{int(n_ones)}<{min_support}"
            else:
                # Duplicate check via hash
                col_hash = hashlib.sha1(col.tobytes()).hexdigest()
                if col_hash in hash_registry:
                    dropped = True
                    reason = f"duplicate_of_{hash_registry[col_hash]}"
                else:
                    hash_registry[col_hash] = name

            audit_rows.append(
                {
                    "feature_name": name,
                    "status": "dropped" if dropped else "retained",
                    "reason": reason,
                    "support": int(n_ones),
                }
            )

            if not dropped:
                retained_indices.append(i)
                retained_names.append(name)

        if not retained_indices:
            return (
                np.empty((X.shape[0], 0)),
                [],
                pd.DataFrame(columns=["feature_name", "status", "reason", "support"]),
            )

        X_clean = X[:, retained_indices]
        return X_clean, retained_names, pd.DataFrame(audit_rows)


# =============================================================================
# MODEL TRAINING & CONSTRAINTS
# =============================================================================

def make_regime_weights(fwd_ret: np.ndarray) -> np.ndarray:
    """
    Bounded multiplicative sample weight built from move size, tradeability,
    and tail emphasis.
    """
    fee = 0.003
    net = np.maximum(np.abs(fwd_ret) - fee, 0.0)

    # Scale by 90th percentile of net move
    scale = np.quantile(net, 0.90) if len(net) > 0 else 1.0
    scale = max(scale, 1e-6)

    move_trade = np.clip(net / scale, 0.0, 1.0)

    # Slight tail emphasis (fwd_ret is assumed to be direction-aligned)
    q80 = np.quantile(fwd_ret, 0.80) if len(fwd_ret) > 0 else 0.0
    q = (fwd_ret > q80).astype(float)

    # w = (1.0 + 1.5 * move_trade) * (1.0 + 0.15 * q)
    w = (1.0 + 1.5 * move_trade) * (1.0 + 0.15 * q)
    w = np.clip(w, 0.75, 2.0)
    w = w / max(w.mean(), 1e-6)
    return w.astype(np.float32)


class InteractionModel:
    """
    LightGBM is trained without strict interaction constraints.
    Structural validity of rule paths is enforced in:
        RuleExtractor._is_path_valid()
    using interaction_group metadata.
    """

    def __init__(
        self,
        metadata: List[FeatureMetadata],
        cfg: Dict[str, Any],
        side: str = "long",
        allowed_group_pairs: Optional[Sequence[Tuple[str, str]]] = None,
    ):
        self.metadata = metadata
        self.cfg = cfg
        self.side = side
        self.allowed_group_pairs = allowed_group_pairs
        self.constraints = self._build_interaction_constraints()

    def _build_interaction_constraints(self) -> List[List[int]]:
        """
        Build interaction constraints for LightGBM.
        """
        # Interaction constraints removed per user request to allow more combinations.
        return []

    def _verify_constraints(
        self, constraints, trigger_idxs, location_idxs, regime_idxs
    ):
        """
        Hardening: Verify no same-group pairs.
        """
        for c in constraints:
            if len(c) == 2:
                idx1, idx2 = c
                m1 = self.metadata[idx1]
                m2 = self.metadata[idx2]
                if m1.group == m2.group:
                    raise ValueError(
                        f"Constraint violation: {m1.group}-{m2.group} pair ({idx1}, {idx2})"
                    )

        # Summary
        t_l = sum(
            1
            for c in constraints
            if len(c) == 2
            and {self.metadata[c[0]].group, self.metadata[c[1]].group}
            == {"trigger", "location"}
        )
        t_r = sum(
            1
            for c in constraints
            if len(c) == 2
            and {self.metadata[c[0]].group, self.metadata[c[1]].group}
            == {"trigger", "regime"}
        )
        l_r = sum(
            1
            for c in constraints
            if len(c) == 2
            and {self.metadata[c[0]].group, self.metadata[c[1]].group}
            == {"location", "regime"}
        )
        tprint(
            f"Constraints built: T-L={t_l}, T-R={t_r}, L-R={l_r}, Singletons={len(self.metadata)}"
        )

    def get_constraint_summary(self) -> Dict[str, Any]:
        import collections

        result = {
            "total_singletons": len(self.metadata),
            "total_constraints": len(self.constraints)
            if self.constraints is not None
            else 0,
            "mode": "training permissive / validation strict",
        }

        groups = set(m.group for m in self.metadata)
        for g in groups:
            result[f"num_{g}"] = sum(1 for m in self.metadata if m.group == g)

        regime_families = set(
            m.regime_family for m in self.metadata if m.group == "regime"
        )
        for rf in regime_families:
            result[f"num_regime_{rf}"] = sum(
                1
                for m in self.metadata
                if m.group == "regime" and m.regime_family == rf
            )

        if not self.constraints:
            return result

        summary = collections.defaultdict(int)
        for c in self.constraints:
            if len(c) == 1:
                summary["singleton"] += 1
                m = self.metadata[c[0]]
                summary[f"singleton_{m.group}"] += 1
            else:
                groups = set(self.metadata[i].group for i in c)
                if groups == {"regime"}:
                    summary["regime_cluster"] += 1
                elif groups == {"location"}:
                    summary["location_cluster"] += 1
                else:
                    summary["mixed_cluster"] += 1
        result.update(summary)
        return result

    def train_fold(
        self,
        X_tr,
        y_tr,
        X_va,
        y_va,
        fold_id: int,
        seed: int,
        target_type: str = "quantile",
    ):
        """
        Train a LightGBM model on a single fold.

        Parameters
        ----------
        X_tr : np.ndarray
            Training features
        y_tr : np.ndarray
            Training targets
        X_va : np.ndarray
            Validation features
        y_va : np.ndarray
            Validation targets
        fold_id : int
            Fold identifier
        seed : int
            Random seed
        target_type : str
            Target type (deprecated, always uses quantile regression)

        Returns
        -------
        Tuple[LGBMRegressor, Dict[str, Any]]
            Trained model and fit metadata
        """
        from lightgbm import early_stopping, log_evaluation

        # ENFORCE FINITE TARGETS
        tr_mask = np.isfinite(y_tr)
        va_mask = np.isfinite(y_va)
        X_tr, y_tr = X_tr[tr_mask], y_tr[tr_mask]
        X_va, y_va = X_va[va_mask], y_va[va_mask]

        if len(y_tr) < 100:
            tprint(
                f"WARNING: Fold {fold_id} has very few training samples ({len(y_tr)})"
            )
        if len(y_va) == 0:
            raise ValueError(f"Fold {fold_id} has no finite validation samples")

        # Sample weights for regime mining
        sample_weight = make_regime_weights(y_tr)

        y_lo = float(np.nanquantile(y_tr, 0.01))
        y_hi = float(np.nanquantile(y_tr, 0.99))
        y_tr_reg = np.clip(y_tr, y_lo, y_hi).astype(np.float32, copy=False)
        y_va_reg = np.clip(y_va, y_lo, y_hi).astype(np.float32, copy=False)

        max_depth = int(self.cfg.get("lgbm_max_depth", 5)) + 1
        num_leaves = int(self.cfg.get("lgbm_num_leaves", 64))

        lambda_l1 = float(self.cfg.get("lambda_l1", 0.0)) * 1.33
        lambda_l2 = float(self.cfg.get("lambda_l2", 0.0)) * 1.33

        min_gain_to_split = float(self.cfg.get("min_gain_to_split", 0.0))
        if "hpo_min_gain_to_split" in self.cfg:
            min_gain_to_split = float(self.cfg["hpo_min_gain_to_split"])

        min_data_in_leaf = max(20, int(0.001 * X_tr.shape[0]))
        if "hpo_min_data_in_leaf" in self.cfg:
            min_data_in_leaf = int(self.cfg["hpo_min_data_in_leaf"])

        # Use quantile loss for all targets (triad targets work with quantile regression)
        alpha_hpo = float(self.cfg.get("alpha_hpo", 0.95))
        alpha = alpha_hpo if self.side == "long" else (1.0 - alpha_hpo)

        # Override with dynamic HPO results if available
        if "hpo_best_alpha" in self.cfg:
            alpha = float(self.cfg["hpo_best_alpha"])

        params = {
            "objective": "quantile",
            "alpha": alpha,
            "metric": "quantile",
            "boosting_type": "gbdt",
            "max_depth": max_depth,
            "num_leaves": num_leaves,
            "min_data_in_leaf": min_data_in_leaf,
            "lambda_l1": lambda_l1,
            "lambda_l2": lambda_l2,
            "min_gain_to_split": min_gain_to_split,
            "learning_rate": float(self.cfg.get("learning_rate", 0.01)),
            "n_estimators": 1000,
            "verbosity": -1,
            "random_state": seed,
            "extra_trees": self.cfg.get("extra_trees", True),
            "n_jobs": max(1, min(3, int(self.cfg.get("lgbm_n_jobs", 3)))),
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "feature_fraction": 0.8,
        }
        target_mode = "quantile_regression"

        if self.constraints:
            params["interaction_constraints"] = self.constraints

        model = LGBMRegressor(**params)
        evals_result = {}
        model.fit(
            X_tr,
            y_tr_reg,
            sample_weight=sample_weight,
            eval_set=[(X_va, y_va_reg)],
            callbacks=[
                early_stopping(stopping_rounds=50),
                log_evaluation(period=0),
                # Record evaluation results
                lambda env: evals_result.setdefault(
                    env.iteration, env.evaluation_result_list
                ),
            ],
        )

        # Get best metric
        best_iter = model.best_iteration_
        best_val_metric = np.nan
        if best_iter in evals_result:
            for dataset_name, metric_name, val, is_higher_better in evals_result[
                best_iter
            ]:
                if metric_name == "l2":
                    best_val_metric = val
                    break

        feature_importances_gain = model.booster_.feature_importance(
            importance_type="gain"
        )
        feature_importances_split = model.booster_.feature_importance(
            importance_type="split"
        )

        # Metadata persistence
        fit_meta = {
            "model_id": "lgbm_discovery",
            "fold_id": fold_id,
            "seed": seed,
            "best_iteration": best_iter,
            "best_val_metric": best_val_metric,
            "train_samples": X_tr.shape[0],
            "val_samples": X_va.shape[0],
            "params_hash": hashlib.sha1(str(params).encode()).hexdigest()[:8],
            "classification": False,
            "threshold_tr": np.nan,
            "threshold_va": np.nan,
            "target_mode": target_mode,
            "target_type": target_type,
            "feature_importances_gain": feature_importances_gain,
            "feature_importances_split": feature_importances_split,
            "params": params,
        }

        return model, fit_meta


# =============================================================================
# LEAF EXTRACTION & RULE SCORING
# =============================================================================


class RuleExtractor:
    def __init__(
        self,
        metadata: List[FeatureMetadata],
        cfg: Dict[str, Any],
        slot_order: Sequence[str] = ("trigger", "location", "regime"),
        positive_only_groups: Optional[Sequence[str]] = None,
        required_positive_groups: Optional[Sequence[str]] = None,
        collapse_duplicate_groups: Optional[Sequence[str]] = None,
    ):
        self.metadata_lookup = {m.feature_index: m for m in metadata}
        self.cfg = cfg
        self.slot_order = slot_order
        self.positive_only_groups = set(positive_only_groups or [])
        self.required_positive_groups = set(required_positive_groups or [])
        self.collapse_duplicate_groups = set(collapse_duplicate_groups or [])
        self.rejection_audit = []
        self.total_leaf_paths = 0
        self.total_non_empty_paths = 0

    def extract_rules(
        self,
        model: LGBMRegressor,
        model_id: str,
        fold_id: int,
        seed: int,
        target_name: str = "fwd_ret_norm",
        horizon: int = 0,
    ) -> List[ExtractedRule]:
        """
        Extract rules from a trained LightGBM model.
        
        Parameters
        ----------
        model : LGBMRegressor
            Trained LightGBM model
        model_id : str
            Identifier for the model
        fold_id : int
            Fold identifier
        seed : int
            Random seed used for training
        target_name : str
            Name of the target for provenance tracking (default: 'fwd_ret_norm')
        horizon : int
            Horizon in bars for provenance tracking (default: 0)
            
        Returns
        -------
        List[ExtractedRule]
            List of extracted rules with target/horizon provenance
        """
        # ALWAYS use native booster dump for correct semantics according to fix spec
        dump = model.booster_.dump_model()
        rules = []
        self.rejection_audit = []  # For diagnostics
        self.total_leaf_paths = 0
        self.total_non_empty_paths = 0
        
        # Store target/horizon for provenance
        self._current_target_name = target_name
        self._current_horizon = horizon

        for tree_idx, tree in enumerate(dump["tree_info"]):
            self._traverse_tree(
                tree["tree_structure"], [], tree_idx, model_id, fold_id, seed, rules
            )

        reject_counts = collections.Counter(r["reason"] for r in self.rejection_audit)

        tprint(
            f"Extracted {len(rules)} valid paths from {self.total_leaf_paths} total paths ({self.total_non_empty_paths} non-empty)."
        )
        if reject_counts:
            tprint("Top rejection reasons:")
            for reason, count in reject_counts.most_common(5):
                tprint(f"  - {reason}: {count}")

        # Check if almost all paths rejected for same-family regime violations
        same_family_violations = sum(
            count
            for reason, count in reject_counts.items()
            if reason.startswith("group_violation_regime")
        )
        if (
            self.total_non_empty_paths > 0
            and same_family_violations / self.total_non_empty_paths > 0.8
        ):
            tprint(
                "WARNING: Almost all paths (>80%) rejected for same-family regime violations. Consider relaxing constraints."
            )

        return rules

    def _normalize_predicate(
        self, node: Dict[str, Any], direction: int
    ) -> Optional[Tuple[int, str, float]]:
        """
        Simplified and hardened normalization for [0, 1] boolean features.
        LightGBM JSON format:
        Left child (direction 1) is 'value <= threshold'
        Right child (direction 0) is 'value > threshold'
        """
        threshold = node.get("threshold")
        if threshold is None:
            return None

        # Standard LGBM boolean split is at 0.5
        if abs(threshold - 0.5) > 1e-4:
            tprint(
                f"WARNING: Unexpected split threshold {threshold} in boolean feature."
            )

        # Direction 1: Left (<= 0.5) -> Feature is 0
        if direction == 1:
            return (0, "<=", threshold)

        # Direction 0: Right (> 0.5) -> Feature is 1
        else:
            return (1, ">", threshold)

    def _traverse_tree(
        self, node, current_conditions, tree_idx, model_id, fold_id, seed, rules
    ):
        if "leaf_value" in node:
            self.total_leaf_paths += 1
            if not current_conditions:
                self.rejection_audit.append(
                    {
                        "model_id": model_id,
                        "fold_id": fold_id,
                        "seed": seed,
                        "tree_idx": tree_idx,
                        "leaf_idx": node.get("leaf_index", -1),
                        "reason": "empty_path",
                    }
                )
                return

            self.total_non_empty_paths += 1

            reduced_conditions, reduce_reason = self._reduce_conditions(
                current_conditions
            )
            if reduce_reason is not None:
                self.rejection_audit.append(
                    {
                        "model_id": model_id,
                        "fold_id": fold_id,
                        "seed": seed,
                        "tree_idx": tree_idx,
                        "leaf_idx": node.get("leaf_index", -1),
                        "reason": reduce_reason,
                    }
                )
                return

            # 1. Path Validation Gates (Hardened)
            is_valid, reason = self._is_path_valid(reduced_conditions)
            if not is_valid:
                self.rejection_audit.append(
                    {
                        "model_id": model_id,
                        "fold_id": fold_id,
                        "seed": seed,
                        "tree_idx": tree_idx,
                        "leaf_idx": node.get("leaf_index", -1),
                        "reason": reason,
                    }
                )
                return

            # 2. Canonical Identity (Slot-based)
            canonical_key = self._build_canonical_key(reduced_conditions)
            if not canonical_key:
                return

            # 3. Instance-specific ID
            prov_str = f"{canonical_key}_{model_id}_{fold_id}_{seed}_{tree_idx}_{node.get('leaf_index', -1)}"
            rule_id = hashlib.sha1(prov_str.encode()).hexdigest()[:12]

            rules.append(
                ExtractedRule(
                    rule_id=rule_id,
                    canonical_key=canonical_key,
                    conditions=list(reduced_conditions),
                    model_id=model_id,
                    fold_id=fold_id,
                    seed=seed,
                    tree_index=tree_idx,
                    leaf_index=node.get("leaf_index", -1),
                    leaf_value=node["leaf_value"],
                    support_train=node.get("leaf_count", 0),
                    source_target=getattr(self, "_current_target_name", "fwd_ret_norm"),
                    source_horizon=getattr(self, "_current_horizon", 0),
                )
            )
            return

        split_feat_idx = node["split_feature"]
        m = self.metadata_lookup.get(split_feat_idx)
        if not m:
            return

        # Normalized branching
        for direction in [1, 0]:  # 1=Left, 0=Right
            norm = self._normalize_predicate(node, direction)
            if norm is None:
                continue

            norm_val, raw_op, raw_thr = norm
            cond = RuleCondition(
                feature_name=m.feature_name,
                feature_index=split_feat_idx,
                group=m.group,
                normalized_value=norm_val,
                raw_operator=raw_op,
                raw_threshold=raw_thr,
                raw_decision_type=node.get("decision_type"),
                default_left=node.get("default_left"),
                missing_type=node.get("missing_type"),
            )

            child_node = node["left_child"] if direction == 1 else node["right_child"]
            self._traverse_tree(
                child_node,
                current_conditions + [cond],
                tree_idx,
                model_id,
                fold_id,
                seed,
                rules,
            )

    def _reduce_conditions(
        self, conditions: List[RuleCondition]
    ) -> Tuple[Optional[List[RuleCondition]], Optional[str]]:
        """
        Collapse duplicate groups when they contain a single positive condition plus
        same-group negatives. This lets Stage A preserve one location and one regime
        slot instead of rejecting paths that refine the same group multiple times.
        """
        by_group: Dict[str, List[RuleCondition]] = collections.defaultdict(list)
        group_order: List[str] = []
        for c in conditions:
            if c.group not in by_group:
                group_order.append(c.group)
            by_group[c.group].append(c)

        reduced: List[RuleCondition] = []
        for group in group_order:
            group_conditions = by_group[group]
            if group not in self.collapse_duplicate_groups:
                feat_map: Dict[int, int] = {}
                for c in group_conditions:
                    prev = feat_map.get(c.feature_index)
                    if prev is not None:
                        if prev != c.normalized_value:
                            return None, f"contradiction_{c.feature_name}"
                        continue
                    feat_map[c.feature_index] = c.normalized_value
                    reduced.append(c)
                continue

            positive_by_feature: Dict[int, RuleCondition] = {}
            negative_by_feature: Dict[int, RuleCondition] = {}
            for c in group_conditions:
                if c.normalized_value == 1:
                    positive_by_feature[c.feature_index] = c
                else:
                    negative_by_feature[c.feature_index] = c

            positive_conditions = list(positive_by_feature.values())
            if len(positive_conditions) > 1:
                return None, f"group_violation_{group}_{len(group_conditions)}"
            if len(positive_conditions) == 1:
                reduced.append(positive_conditions[0])
                continue
            reduced.extend(negative_by_feature.values())

        return reduced, None

    def _is_path_valid(self, conditions: List[RuleCondition]) -> Tuple[bool, str]:
        """
        Hardened validation: Group limits, contradictions, and polarity.
        """
        if not conditions:
            return False, "empty_path"

        seen_groups = {}
        seen_features = {}

        for c in conditions:
            m = self.metadata_lookup.get(c.feature_index)
            if m is None:
                continue

            # Interaction group constraints removed per user request.
            # ig = m.interaction_group
            # prev_feat = seen_groups.get(ig)
            # if prev_feat is not None and prev_feat != c.feature_index:
            #     return False, f"interaction_group_violation_{ig}"
            # seen_groups[ig] = c.feature_index

            prev_val = seen_features.get(c.feature_index)
            if prev_val is not None and prev_val != c.normalized_value:
                return False, f"contradiction_{c.feature_name}"

            seen_features[c.feature_index] = c.normalized_value

        # Polarity Check: reject only all-negative paths
        # A path is all-negative if NO condition has normalized_value == 1
        if not any(c.normalized_value == 1 for c in conditions):
            return False, "all_negative_path"

        for c in conditions:
            if c.group in self.positive_only_groups and c.normalized_value != 1:
                return False, f"negative_not_allowed_{c.group}"

        positive_groups = {c.group for c in conditions if c.normalized_value == 1}
        missing_required = sorted(self.required_positive_groups - positive_groups)
        if missing_required:
            return False, f"missing_required_group_{missing_required[0]}"

        return True, "valid"

    def _build_canonical_key(self, conditions: List[RuleCondition]) -> Optional[str]:
        """
        Deterministic slot-based key using slot_order.
        """
        slots = collections.defaultdict(list)
        for c in conditions:
            if c.group in self.slot_order:
                slots[c.group].append(c)

        out_slots = []
        for s in self.slot_order:
            group_conds = slots.get(s, [])
            if not group_conds:
                out_slots.append("(*)")
            else:
                # Sort by feature name for canonical ordering
                group_conds.sort(key=lambda x: x.feature_name)
                # Deduplicate same feature identical conditions
                seen = set()
                joined = []
                for c in group_conds:
                    rep = f"{c.feature_name}=={int(c.normalized_value)}"
                    if rep not in seen:
                        joined.append(rep)
                        seen.add(rep)
                out_slots.append(f"({'&'.join(joined)})")

        return "|".join(out_slots)


COMPOSITE_RULE_PATTERN = re.compile(r"^Composite\((.+)\)_OR_\((.+)\)$")


def split_composite_key(canonical_key: str) -> Optional[Tuple[str, str]]:
    match = COMPOSITE_RULE_PATTERN.match(canonical_key)
    if not match:
        return None
    return match.group(1), match.group(2)


def parse_slot_map(
    canonical_key: str,
    slot_order: Sequence[str] = ("trigger", "location", "regime"),
) -> Dict[str, str]:
    parts = split_composite_key(canonical_key)
    if parts is not None:
        raise ValueError(f"Composite key {canonical_key} has no direct slot map")
    slots = canonical_key.split("|")
    if len(slots) != len(slot_order):
        raise ValueError(
            f"Key {canonical_key} has {len(slots)} slots but expected {len(slot_order)}"
        )
    return {group: slot.strip("()") for group, slot in zip(slot_order, slots)}


def build_stage_a_parent_key_from_slot_map(slot_map: Dict[str, str]) -> Optional[str]:
    loc = slot_map.get("location", "*")
    reg = slot_map.get("regime", "*")
    if loc == "*" and reg == "*":
        return None
    return f"(*)|({loc})|({reg})"


def iter_primitive_keys(canonical_key: str) -> List[str]:
    composite_parts = split_composite_key(canonical_key)
    if composite_parts is None:
        return [canonical_key]
    out: List[str] = []
    for part in composite_parts:
        out.extend(iter_primitive_keys(part))
    return out


def extract_feature_names_from_key(canonical_key: str) -> List[str]:
    names: List[str] = []
    for part in iter_primitive_keys(canonical_key):
        for slot in part.split("|"):
            slot_value = slot.strip("()")
            if slot_value == "*":
                continue
            for cond_str in slot_value.split("&"):
                if "==" not in cond_str:
                    continue
                names.append(cond_str.split("==")[0])
    return sorted(set(names))


def infer_rule_side(
    canonical_key: str,
    mean_net_ret: Optional[float] = None,
    explicit_side: Optional[str] = None,
) -> str:
    if explicit_side:
        return explicit_side
    names = [name.lower() for name in extract_feature_names_from_key(canonical_key)]
    has_long = any(token in name for name in names for token in ("long", "bull", "up"))
    has_short = any(
        token in name for name in names for token in ("short", "bear", "down")
    )
    if has_long and has_short:
        return "mixed"
    if has_long:
        return "long"
    if has_short:
        return "short"
    if mean_net_ret is not None and np.isfinite(mean_net_ret):
        if mean_net_ret > 0:
            return "long"
        if mean_net_ret < 0:
            return "short"
    return "unknown"


def display_arity_for_key(canonical_key: str) -> int:
    composite_parts = split_composite_key(canonical_key)
    if composite_parts is not None:
        return max(display_arity_for_key(part) for part in composite_parts)

    total = 0
    for slot in canonical_key.split("|"):
        slot_value = slot.strip("()")
        if slot_value == "*":
            continue
        total += sum(1 for cond_str in slot_value.split("&") if "==" in cond_str)
    return total


def structural_depth_for_key(canonical_key: str) -> int:
    composite_parts = split_composite_key(canonical_key)
    if composite_parts is not None:
        return sum(structural_depth_for_key(part) for part in composite_parts)
    return display_arity_for_key(canonical_key)


def build_walk_forward_folds(
    n_samples: int,
    n_folds: int,
    min_train_frac: float = 0.5,
    embargo: int = 0,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    if n_samples <= 1:
        return []
    min_train = max(1, int(np.floor(n_samples * min_train_frac)))
    min_train = min(min_train, n_samples - 1)
    remaining = n_samples - min_train
    if remaining <= 0:
        return []
    n_val_folds = min(max(1, n_folds), remaining)
    base_size = remaining // n_val_folds
    remainder = remaining % n_val_folds
    folds: List[Tuple[np.ndarray, np.ndarray]] = []
    va_start = min_train
    for fold_id in range(n_val_folds):
        fold_size = base_size + (1 if fold_id < remainder else 0)
        va_end = min(n_samples, va_start + fold_size)
        tr_end = max(0, va_start - embargo)
        tr_idx = np.arange(0, tr_end, dtype=np.int32)
        va_idx = np.arange(va_start, va_end, dtype=np.int32)
        if tr_idx.size == 0 or va_idx.size == 0:
            va_start = va_end
            continue
        if tr_idx.max() >= va_idx.min():
            raise ValueError(
                f"Invalid walk-forward fold {fold_id}: train leaks into validation"
            )
        folds.append((tr_idx, va_idx))
        va_start = va_end
    return folds


class DictionaryMaskResolver:
    def __init__(
        self,
        mask_map: Dict[str, np.ndarray],
        parent_context_map: Optional[Dict[str, str]] = None,
        side_map: Optional[Dict[str, str]] = None,
    ):
        self.mask_map = {
            key: np.asarray(mask, dtype=bool) for key, mask in mask_map.items()
        }
        self.parent_context_map = parent_context_map or {}
        self.side_map = side_map or {}

    def register_mask(
        self,
        canonical_key: str,
        mask: np.ndarray,
        parent_context_key: Optional[str] = None,
        side: Optional[str] = None,
    ) -> None:
        self.mask_map[canonical_key] = np.asarray(mask, dtype=bool)
        if parent_context_key:
            self.parent_context_map[canonical_key] = parent_context_key
        if side:
            self.side_map[canonical_key] = side

    def get_mask(
        self, canonical_key: str, indices: Optional[np.ndarray] = None
    ) -> np.ndarray:
        composite_parts = split_composite_key(canonical_key)
        if composite_parts is not None:
            left = self.get_mask(composite_parts[0], indices)
            right = self.get_mask(composite_parts[1], indices)
            return left | right
        if canonical_key not in self.mask_map:
            raise KeyError(f"Cannot resolve mask for {canonical_key}")
        mask = self.mask_map[canonical_key]
        if indices is None:
            return mask.copy()
        return mask[indices]

    def get_parent_context_key(self, canonical_key: str) -> Optional[str]:
        composite_parts = split_composite_key(canonical_key)
        if composite_parts is not None:
            left = self.get_parent_context_key(composite_parts[0])
            right = self.get_parent_context_key(composite_parts[1])
            return left if left == right else None
        return self.parent_context_map.get(canonical_key)

    def get_rule_side(self, canonical_key: str) -> Optional[str]:
        composite_parts = split_composite_key(canonical_key)
        if composite_parts is not None:
            left = self.get_rule_side(composite_parts[0])
            right = self.get_rule_side(composite_parts[1])
            return left if left == right else "mixed"
        return self.side_map.get(canonical_key)


malformed_key_count = 0
unresolved_feature_count = 0
unresolved_feature_names = set()
stage_b_reconstruction_success = 0
stage_b_reconstruction_failed = 0


class CanonicalRuleMaskResolver:
    def __init__(
        self,
        X: np.ndarray,
        metadata: List[FeatureMetadata],
        context_lookup: Optional[Dict[str, np.ndarray]] = None,
        context_key_map: Optional[Dict[str, str]] = None,
        slot_order: Sequence[str] = ("trigger", "location", "regime"),
    ):
        self.X = X
        self.metadata = metadata
        self.context_lookup = {
            key: np.asarray(val, dtype=bool)
            for key, val in (context_lookup or {}).items()
        }
        self.context_key_map = context_key_map or {}
        self.slot_order = tuple(slot_order)
        self.name_to_idx = {m.feature_name: m.feature_index for m in metadata}
        self.parent_key_to_context_name = {
            parent_key: ctx_name
            for ctx_name, parent_key in self.context_key_map.items()
        }

    def _slice_mask(
        self, mask: np.ndarray, indices: Optional[np.ndarray]
    ) -> np.ndarray:
        if indices is None:
            return mask.copy()
        return mask[indices]

    def _resolve_feature_mask(
        self, feature_name: str, target_val: int, indices: Optional[np.ndarray]
    ) -> np.ndarray:
        if feature_name in self.name_to_idx:
            values = (
                self.X[:, self.name_to_idx[feature_name]]
                if indices is None
                else self.X[indices, self.name_to_idx[feature_name]]
            )
            return values == target_val
        if feature_name in self.context_lookup:
            base_mask = self._slice_mask(self.context_lookup[feature_name], indices)
            return base_mask if target_val == 1 else ~base_mask
        raise KeyError(f"Unknown feature {feature_name} in canonical key")

    def _resolve_context_parent_mask(
        self, canonical_key: str, indices: Optional[np.ndarray]
    ) -> Optional[np.ndarray]:
        try:
            slot_map = parse_slot_map(canonical_key, self.slot_order)
        except ValueError:
            slot_map = parse_slot_map(canonical_key, ("trigger", "location", "regime"))
        parent_key = build_stage_a_parent_key_from_slot_map(slot_map)
        if parent_key is None:
            return None
        ctx_name = self.parent_key_to_context_name.get(parent_key)
        if ctx_name is None:
            return None
        if not ctx_name.startswith("ctx__"):
            raise ValueError(
                f"Unexpected unresolved feature in canonical key: {ctx_name}"
            )
        return self._slice_mask(self.context_lookup[ctx_name], indices)

    def get_mask(
        self, canonical_key: str, indices: Optional[np.ndarray] = None
    ) -> np.ndarray:
        composite_parts = split_composite_key(canonical_key)
        if composite_parts is not None:
            return self.get_mask(composite_parts[0], indices) | self.get_mask(
                composite_parts[1], indices
            )

        try:
            slot_map = parse_slot_map(canonical_key, self.slot_order)
        except ValueError:
            slot_map = parse_slot_map(canonical_key, ("trigger", "location", "regime"))

        n_samples = self.X.shape[0] if indices is None else len(indices)
        mask = np.ones(n_samples, dtype=bool)
        unresolved: List[Tuple[str, str]] = []

        global malformed_key_count, unresolved_feature_count, unresolved_feature_names
        for group, slot_value in slot_map.items():
            if slot_value == "*":
                continue

            for cond_str in slot_value.split("&"):
                if "==" not in cond_str:
                    malformed_key_count += 1
                    raise ValueError(f"Malformed slot {cond_str} in {canonical_key}")
                feature_name, target_val_raw = cond_str.split("==")
                target_val = int(target_val_raw)
                if (
                    feature_name in self.name_to_idx
                    or feature_name in self.context_lookup
                ):
                    mask &= self._resolve_feature_mask(
                        feature_name, target_val, indices
                    )
                else:
                    unresolved.append((group, feature_name))
                    unresolved_feature_count += 1
                    unresolved_feature_names.add(feature_name)

        if unresolved:
            unresolved_groups = {g for g, _ in unresolved}
            unresolved_features = [f for _, f in unresolved]

            if not unresolved_groups.issubset({"location", "regime"}):
                raise KeyError(
                    f"Cannot resolve groups {unresolved_groups} for key {canonical_key}"
                )

            # Stricter fallback safety: Allow context fallback if features explicitly
            # start with 'ctx__', OR if we successfully locate a parent context mask
            # mapped to this rule structure.
            context_mask = self._resolve_context_parent_mask(canonical_key, indices)
            allow_context_fallback = all(
                f.startswith("ctx__") for f in unresolved_features
            )

            if context_mask is None and not allow_context_fallback:
                raise KeyError(
                    f"Unresolved features {unresolved_features} in key {canonical_key}"
                )
            elif allow_context_fallback:
                tprint(
                    f"WARNING: Unresolved feature fallback used for {unresolved_features} in {canonical_key}"
                )

            if context_mask is None:
                raise KeyError(f"Cannot map {canonical_key} to a saved Stage A context")

            mask &= context_mask

        return mask

    def get_parent_context_key(self, canonical_key: str) -> Optional[str]:
        composite_parts = split_composite_key(canonical_key)
        if composite_parts is not None:
            left = self.get_parent_context_key(composite_parts[0])
            right = self.get_parent_context_key(composite_parts[1])
            return left if left == right else None

        try:
            slot_map = parse_slot_map(canonical_key, self.slot_order)
        except ValueError:
            slot_map = parse_slot_map(canonical_key, ("trigger", "location", "regime"))

        if "context" in slot_map and slot_map["context"] != "*":
            ctx_name = slot_map["context"].split("==")[0]
            return self.context_key_map.get(ctx_name)

        parent_key = build_stage_a_parent_key_from_slot_map(slot_map)
        if parent_key in self.parent_key_to_context_name:
            return parent_key
        return None

    def get_rule_side(self, canonical_key: str) -> Optional[str]:
        return infer_rule_side(canonical_key)


class RuleScorer:
    def __init__(
        self,
        metadata: List[FeatureMetadata],
        cfg: Dict[str, Any],
        mask_resolver: Optional[
            Union[CanonicalRuleMaskResolver, DictionaryMaskResolver]
        ] = None,
    ):
        self.metadata = metadata
        self.cfg = cfg
        self.mask_resolver = mask_resolver

    def _compute_required_hurdle(self, support_pct: float, display_arity: int) -> float:
        base_hurdle = float(self.cfg.get("prune_base_hurdle", 0.0002))
        penalty_exp = float(self.cfg.get("prune_support_exp", 0.5))
        complexity_bonus = float(
            self.cfg.get("prune_complexity_bonus_map", {}).get(str(display_arity), 0.0)
        )
        safe_support = max(float(support_pct), 0.0005)
        return (base_hurdle * (1.0 - complexity_bonus)) / (safe_support**penalty_exp)

    def _compute_within_mask_ic(
        self,
        predictions: np.ndarray,
        target: np.ndarray,
        mask: np.ndarray,
        method: str = "spearman",
    ) -> Tuple[float, float]:
        """
        Compute IC within mask and delta from global IC.

        Args:
            predictions: Model predictions array
            target: Target values array
            mask: Boolean mask indicating regime activation
            method: Correlation method ("spearman" or "pearson")

        Returns:
            Tuple of (within_mask_ic, delta_ic) where delta_ic = within_mask_ic - global_ic
        """
        # Global IC
        valid = ~(np.isnan(predictions) | np.isnan(target))
        if valid.sum() < 10:
            return np.nan, np.nan

        if method == "spearman":
            global_ic = _safe_spearman(predictions[valid], target[valid])
        else:
            # Pearson correlation
            valid_preds = predictions[valid]
            valid_targets = target[valid]
            if np.all(valid_preds == valid_preds[0]) or np.all(
                valid_targets == valid_targets[0]
            ):
                global_ic = np.nan
            else:
                global_ic = float(np.corrcoef(valid_preds, valid_targets)[0, 1])

        # Within-mask IC
        mask_active = mask.astype(bool) & valid
        if mask_active.sum() < 10:
            return np.nan, np.nan

        if method == "spearman":
            within_ic = _safe_spearman(predictions[mask_active], target[mask_active])
        else:
            masked_preds = predictions[mask_active]
            masked_targets = target[mask_active]
            if np.all(masked_preds == masked_preds[0]) or np.all(
                masked_targets == masked_targets[0]
            ):
                within_ic = np.nan
            else:
                within_ic = float(np.corrcoef(masked_preds, masked_targets)[0, 1])

        delta_ic = within_ic - global_ic if not np.isnan(within_ic) else np.nan

        return within_ic, delta_ic

    def _compute_entropy_reduction(
        self,
        target: np.ndarray,
        mask: np.ndarray,
        n_bins: int = 20,
    ) -> float:
        """
        Compute reduction in target uncertainty conditional on mask.

        Uses histogram-based entropy estimation.
        Higher is better - means the regime reduces target uncertainty.

        Args:
            target: Target values array
            mask: Boolean mask indicating regime activation
            n_bins: Number of bins for histogram estimation

        Returns:
            Entropy reduction (global entropy - conditional entropy).
            Positive means the mask concentrates the target distribution.
        """
        valid = ~np.isnan(target)
        target_valid = target[valid]

        if len(target_valid) < 100:
            return np.nan

        # Global entropy
        hist_global, _ = np.histogram(target_valid, bins=n_bins, density=True)
        hist_global = hist_global / hist_global.sum()
        hist_global = hist_global[hist_global > 0]
        entropy_global = -np.sum(hist_global * np.log2(hist_global + 1e-9))

        # Conditional entropy (within mask)
        mask_active = mask.astype(bool) & valid
        if mask_active.sum() < 50:
            return np.nan

        target_masked = target[mask_active]
        hist_masked, _ = np.histogram(target_masked, bins=n_bins, density=True)
        hist_masked = hist_masked / hist_masked.sum()
        hist_masked = hist_masked[hist_masked > 0]
        entropy_masked = -np.sum(hist_masked * np.log2(hist_masked + 1e-9))

        # Reduction = global - conditional (positive is good)
        return entropy_global - entropy_masked

    def score_key_oos(
        self,
        canonical_key: str,
        fwd_ret: np.ndarray,
        folds: List[Tuple[np.ndarray, np.ndarray]],
        resolver: Optional[
            Union[CanonicalRuleMaskResolver, DictionaryMaskResolver]
        ] = None,
        require_uplift: bool = False,
        parent_context_key: Optional[str] = None,
        discovery_count: int = 0,
        n_instances: Optional[int] = None,
        pipeline_stage: Optional[str] = None,
        explicit_side: Optional[str] = None,
        bounded_target: Optional[np.ndarray] = None,
        target_name: str = "fwd_ret_norm",
        horizon: int = 0,
        predictions: Optional[np.ndarray] = None,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Score rules with optional bounded target support.

        Args:
            canonical_key: Rule identifier
            fwd_ret: Forward returns array
            folds: List of (train_idx, val_idx) tuples for cross-validation
            resolver: Mask resolver for computing rule masks
            require_uplift: Whether to require uplift over parent context
            parent_context_key: Parent context key for uplift computation
            discovery_count: Number of times this rule was discovered
            n_instances: Total number of instances
            pipeline_stage: Pipeline stage identifier
            explicit_side: Explicit side (long/short) override
            bounded_target: Optional bounded target for within-mask IC computation
            target_name: Name of the target for logging
            horizon: Horizon identifier for logging
            predictions: Optional model predictions for within-mask IC computation

        Returns:
            Tuple of (summary_dict, fold_records_list)
        """
        resolver = resolver or self.mask_resolver
        if resolver is None:
            raise ValueError("RuleScorer requires a mask resolver")

        fold_records: List[Dict[str, Any]] = []
        epsilon = float(self.cfg.get("sign_dead_zone", 1e-6))

        if require_uplift and not parent_context_key:
            parent_context_key = resolver.get_parent_context_key(canonical_key)

        # Track within-mask IC metrics across folds
        within_mask_ic_values: List[float] = []
        delta_within_mask_ic_values: List[float] = []
        entropy_reduction_values: List[float] = []

        for fold_id, (_, va_idx) in enumerate(folds):
            y_va = fwd_ret[va_idx]
            mask = resolver.get_mask(canonical_key, va_idx)
            support = int(mask.sum())
            baseline_support = 0
            baseline_ret = np.nan
            uplift = np.nan
            fold_ic = np.nan
            baseline_ic = np.nan
            delta_ic = np.nan
            within_mask_ic = np.nan
            delta_within_mask_ic = np.nan
            entropy_reduction = np.nan

            if parent_context_key:
                parent_mask = resolver.get_mask(parent_context_key, va_idx)
                baseline_support = int(parent_mask.sum())
                if baseline_support > 0:
                    baseline_ret = float(np.nanmean(y_va[parent_mask]))

                    # Compute baseline IC
                    valid_idx = np.isfinite(y_va) & np.isfinite(parent_mask)
                    if np.sum(valid_idx) >= 3:
                        baseline_ic = _safe_spearman(
                            parent_mask[valid_idx].astype(np.float32), y_va[valid_idx]
                        )

            if support > 0:
                masked_ret = y_va[mask]
                mean_ret = float(np.nanmean(masked_ret))
                std_ret = float(np.nanstd(masked_ret))
                if np.isfinite(baseline_ret):
                    uplift = mean_ret - baseline_ret
                sign = 1 if mean_ret > epsilon else (-1 if mean_ret < -epsilon else 0)

                # Compute mask IC
                valid_idx = np.isfinite(y_va) & np.isfinite(mask)
                if np.sum(valid_idx) >= 3:
                    fold_ic = _safe_spearman(
                        mask[valid_idx].astype(np.float32), y_va[valid_idx]
                    )

                if np.isfinite(fold_ic) and np.isfinite(baseline_ic):
                    delta_ic = fold_ic - baseline_ic

                # Compute within-mask IC if bounded_target and predictions available
                if bounded_target is not None and predictions is not None:
                    target_va = bounded_target[va_idx]
                    preds_va = predictions[va_idx]
                    within_mask_ic, delta_within_mask_ic = self._compute_within_mask_ic(
                        preds_va, target_va, mask
                    )
                    if np.isfinite(within_mask_ic):
                        within_mask_ic_values.append(within_mask_ic)
                    if np.isfinite(delta_within_mask_ic):
                        delta_within_mask_ic_values.append(delta_within_mask_ic)

                # Compute entropy reduction if bounded_target available
                if bounded_target is not None:
                    target_va = bounded_target[va_idx]
                    entropy_reduction = self._compute_entropy_reduction(target_va, mask)
                    if np.isfinite(entropy_reduction):
                        entropy_reduction_values.append(entropy_reduction)
            else:
                mean_ret = np.nan
                std_ret = np.nan
                sign = 0

            fold_records.append(
                {
                    "canonical_key": canonical_key,
                    "fold_id": fold_id,
                    "support": support,
                    "support_pct": support / max(len(va_idx), 1),
                    "mean_ret": mean_ret,
                    "std_ret": std_ret,
                    "sign": sign,
                    "baseline_support": baseline_support,
                    "baseline_ret": baseline_ret,
                    "uplift": uplift,
                    "parent_context_key": parent_context_key,
                    "fold_ic": fold_ic,
                    "baseline_ic": baseline_ic,
                    "delta_ic": delta_ic,
                    "within_mask_ic": within_mask_ic,
                    "delta_within_mask_ic": delta_within_mask_ic,
                    "entropy_reduction": entropy_reduction,
                }
            )

        df_folds = pd.DataFrame(fold_records)
        present = df_folds[df_folds["support"] > 0].copy()
        if present.empty:
            summary = {
                "canonical_key": canonical_key,
                "mean_net_ret": np.nan,
                "directional_mean_ret": np.nan,
                "std_net_ret": np.nan,
                "mean_within_fold_std": np.nan,
                "mean_support_pct": 0.0,
                "std_support_pct": 0.0,
                "presence_freq": 0.0,
                "presence_freq_units": 0.0,
                "sign_consistency": 0.0,
                "min_support_actual": 0,
                "mean_uplift": np.nan,
                "mean_baseline_ret": np.nan,
                "mean_oos_ic": np.nan,
                "p25_oos_ic": np.nan,
                "p50_oos_ic": np.nan,
                "p75_oos_ic": np.nan,
                "mean_delta_ic": np.nan,
                "positive_ic_fraction": 0.0,
                "within_mask_ic": np.nan,
                "delta_within_mask_ic": np.nan,
                "entropy_reduction": np.nan,
                "source_target": target_name,
                "source_horizon": horizon,
                "composite_score": -np.inf,
                "required_hurdle": np.nan,
                "hurdle_excess": np.nan,
                "n_folds": 0,
                "discovery_count": discovery_count,
                "n_instances": 0 if n_instances is None else n_instances,
                "display_arity": display_arity_for_key(canonical_key),
                "structural_depth": structural_depth_for_key(canonical_key),
                "pipeline_stage": pipeline_stage or "unknown",
                "parent_context_key": parent_context_key,
                "side": infer_rule_side(canonical_key, explicit_side=explicit_side),
                "rule_type": "composite"
                if split_composite_key(canonical_key) is not None
                else f"{display_arity_for_key(canonical_key)}-way",
                "accepted": False,
                "rejection_reason": "no_validation_support",
            }
            return summary, fold_records

        mean_net_ret = float(present["mean_ret"].mean())
        std_net_ret = float(present["mean_ret"].std(ddof=0))
        mean_within_fold_std = (
            float(present["std_ret"].mean())
            if present["std_ret"].notna().any()
            else np.nan
        )
        mean_support_pct = float(present["support_pct"].mean())
        std_support_pct = float(present["support_pct"].std(ddof=0))
        presence_freq = float(len(present) / max(len(folds), 1))
        nonzero_signs = present[present["sign"] != 0]["sign"]
        if len(nonzero_signs) == 0:
            if not present.empty:
                tprint(f"DEBUG: Rule {canonical_key[:40]}... has {len(present)} folds but 0 nonzero signs. Root cause: all trades have exactly 0 return.")
            sign_consistency = 0.0
        else:
            major_sign = 1 if mean_net_ret > 0 else -1
            sign_consistency = float((nonzero_signs == major_sign).mean())
        display_arity = display_arity_for_key(canonical_key)
        required_hurdle = self._compute_required_hurdle(mean_support_pct, display_arity)
        use_directional = bool(self.cfg.get("stage_a_directional", True)) and (
            (pipeline_stage or "") == "stage_a_context"
        )
        directional_mean_ret = (
            abs(mean_net_ret)
            if (use_directional and np.isfinite(mean_net_ret))
            else mean_net_ret
        )
        hurdle_excess = directional_mean_ret - required_hurdle
        mean_uplift = (
            float(present["uplift"].mean())
            if present["uplift"].notna().any()
            else np.nan
        )
        mean_baseline_ret = (
            float(present["baseline_ret"].mean())
            if present["baseline_ret"].notna().any()
            else np.nan
        )

        # OOS IC metrics
        mean_oos_ic = (
            float(present["fold_ic"].mean())
            if present["fold_ic"].notna().any()
            else np.nan
        )
        p25_oos_ic = (
            float(present["fold_ic"].quantile(0.25))
            if present["fold_ic"].notna().any()
            else np.nan
        )
        p50_oos_ic = (
            float(present["fold_ic"].quantile(0.50))
            if present["fold_ic"].notna().any()
            else np.nan
        )
        p75_oos_ic = (
            float(present["fold_ic"].quantile(0.75))
            if present["fold_ic"].notna().any()
            else np.nan
        )
        mean_delta_ic = (
            float(present["delta_ic"].mean())
            if present["delta_ic"].notna().any()
            else np.nan
        )

        positive_ic_fraction = (
            float((present["fold_ic"] > 0).mean())
            if present["fold_ic"].notna().any()
            else 0.0
        )

        # Within-mask IC metrics (computed from fold-level values)
        mean_within_mask_ic = (
            float(np.mean(within_mask_ic_values))
            if within_mask_ic_values
            else np.nan
        )
        mean_delta_within_mask_ic = (
            float(np.mean(delta_within_mask_ic_values))
            if delta_within_mask_ic_values
            else np.nan
        )
        mean_entropy_reduction = (
            float(np.mean(entropy_reduction_values))
            if entropy_reduction_values
            else np.nan
        )

        composite_score = (
            mean_net_ret
            * sqrt(max(mean_support_pct, 1e-12))
            * presence_freq
            * sign_consistency
            / (1.0 + max(std_net_ret, 0.0))
        )

        summary = {
            "canonical_key": canonical_key,
            "mean_net_ret": mean_net_ret,
            "directional_mean_ret": directional_mean_ret,
            "std_net_ret": std_net_ret,
            "mean_within_fold_std": mean_within_fold_std,
            "mean_support_pct": mean_support_pct,
            "std_support_pct": std_support_pct,
            "presence_freq": presence_freq,
            "presence_freq_units": presence_freq,
            "sign_consistency": sign_consistency,
            "min_support_actual": int(present["support"].min()),
            "mean_uplift": mean_uplift,
            "mean_baseline_ret": mean_baseline_ret,
            "mean_oos_ic": mean_oos_ic,
            "p25_oos_ic": p25_oos_ic,
            "p50_oos_ic": p50_oos_ic,
            "p75_oos_ic": p75_oos_ic,
            "mean_delta_ic": mean_delta_ic,
            "positive_ic_fraction": positive_ic_fraction,
            "within_mask_ic": mean_within_mask_ic,
            "delta_within_mask_ic": mean_delta_within_mask_ic,
            "entropy_reduction": mean_entropy_reduction,
            "source_target": target_name,
            "source_horizon": horizon,
            "composite_score": composite_score,
            "required_hurdle": required_hurdle,
            "hurdle_excess": hurdle_excess,
            "n_folds": int(len(present)),
            "discovery_count": int(discovery_count),
            "n_instances": int(len(present) if n_instances is None else n_instances),
            "display_arity": display_arity,
            "structural_depth": structural_depth_for_key(canonical_key),
            "pipeline_stage": pipeline_stage or "unknown",
            "parent_context_key": parent_context_key,
            "side": infer_rule_side(
                canonical_key, mean_net_ret=mean_net_ret, explicit_side=explicit_side
            ),
            "rule_type": "composite"
            if split_composite_key(canonical_key) is not None
            else f"{display_arity}-way",
        }

        rejected: List[str] = []
        if summary["min_support_actual"] < int(
            self.cfg.get("min_support_count_validation", 10)
        ):
            rejected.append("low_support")
        if summary["presence_freq"] < float(self.cfg.get("min_presence_freq", 0.4)):
            rejected.append("low_presence")
        if summary["sign_consistency"] < float(
            self.cfg.get("min_sign_consistency", 0.75)
        ):
            rejected.append("low_sign_consistency")
        if (
            not np.isfinite(summary["directional_mean_ret"])
            or summary["directional_mean_ret"] <= 0
        ):
            rejected.append("non_positive_directional_ret")
        if summary["hurdle_excess"] <= 0:
            rejected.append("below_hurdle")
        if require_uplift:
            if not np.isfinite(summary["mean_uplift"]):
                rejected.append("missing_uplift")
            elif summary["mean_uplift"] <= 0:
                rejected.append("non_positive_uplift")

        summary["accepted"] = len(rejected) == 0
        summary["rejection_reason"] = "|".join(rejected)
        
        # NEW: Classify rule type (ranking vs gate)
        rule_type_class = classify_rule_type(
            mean_oos_ic=summary.get("mean_oos_ic", 0),
            within_mask_ic=summary.get("within_mask_ic", 0),
            delta_within_mask_ic=summary.get("delta_within_mask_ic", 0),
        )
        summary["rule_type_class"] = rule_type_class
        
        # NEW: Production classification
        classification, diagnostics = classify_rule_production_quality(
            rule=summary
        )
        summary["production_classification"] = classification
        summary["classification_diagnostics"] = json.dumps(diagnostics)
        
        return summary, fold_records

    def score_registry_oos(
        self,
        keys: Sequence[str],
        fwd_ret: np.ndarray,
        folds: List[Tuple[np.ndarray, np.ndarray]],
        resolver: Optional[
            Union[CanonicalRuleMaskResolver, DictionaryMaskResolver]
        ] = None,
        parent_context_map: Optional[Dict[str, str]] = None,
        require_uplift_keys: Optional[Sequence[str]] = None,
        discovery_count_map: Optional[Dict[str, int]] = None,
        n_instances_map: Optional[Dict[str, int]] = None,
        pipeline_stage_map: Optional[Dict[str, str]] = None,
        side_map: Optional[Dict[str, str]] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        resolver = resolver or self.mask_resolver
        if resolver is None:
            raise ValueError("RuleScorer requires a mask resolver")

        require_uplift_set = set(require_uplift_keys or [])
        summaries: List[Dict[str, Any]] = []
        audits: List[Dict[str, Any]] = []
        seen: set[str] = set()

        # Fast path scoring using NumbaRuleInferenceEngine if we have simple non-composite keys
        # and our resolver supports giving us the underlying X array.
        fast_path = False
        try:
            if isinstance(resolver, CanonicalRuleMaskResolver):
                fast_registry = pd.DataFrame({"canonical_key": keys})
                engine = NumbaRuleInferenceEngine(fast_registry, resolver.metadata)
                mask_matrix = engine.apply(resolver.X)
                fast_path = True
        except KeyError:
            fast_path = False

        for idx, key in enumerate(keys):
            if key in seen:
                continue
            seen.add(key)

            if fast_path and "Composite" not in key:
                # Override the mask in the resolver dynamically
                resolver.context_lookup[key] = mask_matrix[:, idx]

            summary, fold_records = self.score_key_oos(
                canonical_key=key,
                fwd_ret=fwd_ret,
                folds=folds,
                resolver=resolver,
                require_uplift=key in require_uplift_set,
                parent_context_key=(parent_context_map or {}).get(key),
                discovery_count=(discovery_count_map or {}).get(key, 0),
                n_instances=(n_instances_map or {}).get(key),
                pipeline_stage=(pipeline_stage_map or {}).get(key),
                explicit_side=(side_map or {}).get(key),
            )
            summaries.append(summary)
            audits.extend(fold_records)

        if not summaries:
            tprint("WARNING: No rules scored successfully. Returning empty registry.")
            return pd.DataFrame(), pd.DataFrame(audits)

        summary_df = pd.DataFrame(summaries).sort_values(
            ["accepted", "composite_score"], ascending=[False, False]
        )
        summary_df = self._identify_dominated_rules(summary_df)

        # Scorer Reporting Diagnostics
        accepted_count = summary_df["accepted"].sum()
        rejected_count = len(summary_df) - accepted_count
        tprint(
            f"Scorer Input: {len(summary_df)} rules | Accepted: {accepted_count} | Rejected: {rejected_count}"
        )

        rejection_reasons = collections.Counter(
            reason.strip()
            for reasons in summary_df[~summary_df["accepted"]][
                "rejection_reason"
            ].dropna()
            for reason in reasons.split("|")
            if reason.strip()
        )
        if rejection_reasons:
            tprint("Top scorer rejection reasons:")
            for reason, count in rejection_reasons.most_common(5):
                tprint(f"  - {reason}: {count}")

        dom_count = summary_df.get("dominated_by_parent", pd.Series(False)).sum()
        if dom_count > 0:
            tprint(f"Dominated rules flagged: {dom_count}")
            top_dom = summary_df[summary_df["dominated_by_parent"]].head(5)
            for _, row in top_dom.iterrows():
                tprint(
                    f"  - {row['canonical_key']} dominated by {row['dominant_parent_key']}"
                )

        return summary_df, pd.DataFrame(audits)

    def _identify_dominated_rules(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        df = df.copy()
        df["dominated_by_parent"] = False
        df["dominant_parent_key"] = None
        lookup = df.set_index("canonical_key")
        for idx, row in df.iterrows():
            if split_composite_key(row["canonical_key"]) is not None:
                continue
            if row["display_arity"] <= 1:
                continue
            slots = row["canonical_key"].split("|")
            active_positions = [i for i, slot in enumerate(slots) if slot != "(*)"]
            for parent_size in range(len(active_positions) - 1, 0, -1):
                found_parent = False
                for combo in itertools.combinations(active_positions, parent_size):
                    parent_slots = ["(*)"] * len(slots)
                    for slot_idx in combo:
                        parent_slots[slot_idx] = slots[slot_idx]
                    parent_key = "|".join(parent_slots)
                    if parent_key not in lookup.index:
                        continue
                    parent = lookup.loc[parent_key]
                    if parent["accepted"] and (
                        parent["hurdle_excess"] >= row["hurdle_excess"]
                        or parent["composite_score"] >= 0.95 * row["composite_score"]
                    ):
                        df.at[idx, "dominated_by_parent"] = True
                        df.at[idx, "dominant_parent_key"] = parent_key
                        found_parent = True
                        break
                if found_parent:
                    break
        return df


@njit(cache=True, fastmath=True)
def tbm_outcomes_atr_nb(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    atr: np.ndarray,
    horizon: int,
    tp_atr: float,
    sl_atr: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = close.shape[0]
    tp_first = np.zeros(n, dtype=np.int8)
    sl_first = np.zeros(n, dtype=np.int8)
    timeout = np.zeros(n, dtype=np.int8)

    for i in range(n - horizon):
        entry = close[i]
        atr_i = max(atr[i], 1e-9)

        tp_price = entry + tp_atr * atr_i
        sl_price = entry - sl_atr * atr_i

        for j in range(i + 1, i + horizon + 1):
            hi = high[j]
            lo = low[j]

            hit_tp = hi >= tp_price
            hit_sl = lo <= sl_price

            if hit_tp and not hit_sl:
                tp_first[i] = 1
                break
            if hit_sl and not hit_tp:
                sl_first[i] = 1
                break
            if hit_tp and hit_sl:
                median = 0.5 * (hi + lo)
                d_tp = abs(median - tp_price)
                d_sl = abs(median - sl_price)
                if d_tp < d_sl:
                    tp_first[i] = 1
                elif d_sl < d_tp:
                    sl_first[i] = 1
                else:
                    timeout[i] = 1
                break
        else:
            timeout[i] = 1
    return tp_first, sl_first, timeout


class RulePruner:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg

    def prune_for_assessment(
        self, scored_df: pd.DataFrame, all_rules: List[ExtractedRule], top_n: int = 50
    ) -> pd.DataFrame:
        """
        Prunes rules using a hybrid of Scorer metrics (OOS) and LGBM Native metrics (IS).
        """
        if scored_df.empty:
            return scored_df

        # 1. Aggregate Model-Native Metrics from ExtractedRule objects
        # We want to know how the model 'felt' about this canonical rule during training
        native_stats = []
        unique_keys = scored_df["canonical_key"].unique()

        for key in unique_keys:
            # Get all instances (across trees/folds/seeds) of this canonical rule
            instances = [r for r in all_rules if r.canonical_key == key]

            # Calculate Model conviction
            avg_leaf_val = np.mean([r.leaf_value for r in instances])
            total_is_support = np.sum([r.support_train for r in instances])
            occurrence_count = len(instances)  # How many trees used this rule?

            native_stats.append(
                {
                    "canonical_key": key,
                    "avg_model_conviction": abs(avg_leaf_val),
                    "total_is_support": total_is_support,
                    "discovery_count": occurrence_count,
                }
            )

        native_df = pd.DataFrame(native_stats)

        # 2. Merge Native metrics into the Scored Registry
        df = scored_df.merge(native_df, on="canonical_key", how="left")

        # 3. Hard Gates based on Model conviction
        # Reject rules that the model only used once or with very low importance (leaf value)
        min_conviction = float(self.cfg.get("min_avg_leaf_value", 0.001))
        min_discoveries = int(self.cfg.get("min_tree_discoveries", 2))

        mask = (
            (df["avg_model_conviction"] >= min_conviction)
            & (df["discovery_count"] >= min_discoveries)
            & (df["dominated_by_parent"] == False)
            & (df["mean_net_ret"] > 0)  # Basic OOS sanity check
        )

        pruned_df = df[mask].copy()

        # 4. Final Ranking for Assessment
        # We rank by a hybrid of OOS performance and Model Discovery Count
        # Discovery Count is a great proxy for 'Structural Stability'
        pruned_df["prune_rank_score"] = pruned_df["composite_score"] * np.log1p(
            pruned_df["discovery_count"]
        )

        return pruned_df.sort_values("prune_rank_score", ascending=False).head(top_n)


class LineageTracker:
    def __init__(self):
        self.history = []  # List of dicts: {child_id, parent_ids, merge_type, round}

    def record_merge(
        self, child_id, parent_ids, merge_type, iteration_round, details=None
    ):
        self.history.append(
            {
                "child_id": child_id,
                "parent_ids": list(parent_ids),
                "merge_type": merge_type,
                "round": iteration_round,
                "details": json.dumps(details or {}, sort_keys=True),
                "timestamp": pd.Timestamp.now(),
            }
        )

    def get_ancestors(self, rule_id):
        """Recursively find original LGBM tree-path IDs for a composite rule."""
        ancestors = set()
        to_visit = [rule_id]

        while to_visit:
            current_id = to_visit.pop()
            for record in self.history:
                if record["child_id"] == current_id:
                    for parent_id in record["parent_ids"]:
                        if parent_id not in ancestors:
                            ancestors.add(parent_id)
                            to_visit.append(parent_id)
        return list(ancestors)

    def get_audit_df(self) -> pd.DataFrame:
        """Return lineage history as a DataFrame."""
        return pd.DataFrame(self.history)


class IndependentRulePruner:
    """
    Independent Rule Pruner (Hurdle Edition v2.0)
    Updated with:
    1. Hard Max Support Gate (to kill global 'nothingness' rules)
    2. Complexity Bonus (to reward 2-way and 3-way interactions)
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.base_hurdle = float(cfg.get("prune_base_hurdle", 0.0002))
        self.penalty_exponent = float(cfg.get("prune_support_exp", 0.5))
        self.min_discoveries = int(cfg.get("min_tree_discoveries", 2))
        self.min_sign_consistency = float(cfg.get("min_sign_consistency", 0.80))

        # New Gates
        self.max_support_pct = float(
            cfg.get("max_support_pct", 0.25)
        )  # Hard ceiling at 25%
        self.arity_bonus = cfg.get(
            "prune_complexity_bonus_map",
            {"1": 0.0, "2": 0.15, "3": 0.30, "4": 0.10, "5": 0.10, "6": 0.10},
        )

    def prune(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df

        df = df.copy()
        if "dominated_by_parent" not in df.columns:
            df["dominated_by_parent"] = False

        # 1. Hard Gate: Max Support (The 'Lazy Rule' Killer)
        df["is_too_broad"] = df["mean_support_pct"] > self.max_support_pct

        # 2. Determine Rule Complexity from normalized metadata
        df["comp_bonus"] = df["display_arity"].apply(
            lambda val: float(self.arity_bonus.get(str(int(val)), 0.0))
        )

        # 3. Calculate the Complexity-Adjusted Hurdle
        # Formula: (Base * (1-Bonus)) / (Support^Exp)
        safe_support = df["mean_support_pct"].clip(lower=0.0005)
        df["required_hurdle"] = (self.base_hurdle * (1.0 - df["comp_bonus"])) / (
            safe_support**self.penalty_exponent
        )

        # 4. Gate A: Alpha Performance vs Hurdle
        df["hurdle_excess"] = df["mean_net_ret"] - df["required_hurdle"]
        df["beats_hurdle"] = df["hurdle_excess"] > 0

        # 5. Final Selection
        gate_summary = {
            "is_too_broad_rejected": int(df["is_too_broad"].sum()),
            "beats_hurdle_rejected": int((~df["beats_hurdle"]).sum()),
            "sign_consistency_rejected": int(
                (df["sign_consistency"] < self.min_sign_consistency).sum()
            ),
            "discovery_count_rejected": int(
                (df["discovery_count"] < self.min_discoveries).sum()
            ),
            "dominated_by_parent_rejected": int(df["dominated_by_parent"].sum()),
        }

        mask = (
            (~df["is_too_broad"])
            & (df["beats_hurdle"])
            & (df["sign_consistency"] >= self.min_sign_consistency)
            & (df["discovery_count"] >= self.min_discoveries)
            & (~df["dominated_by_parent"].fillna(False).astype(bool))
        )

        final_registry = df[mask].copy()

        tprint(
            f"Pruning Gate-by-Gate Funnel: Total={len(df)} | "
            f"Broad Rejected={gate_summary['is_too_broad_rejected']} | "
            f"Hurdle Failed={gate_summary['beats_hurdle_rejected']} | "
            f"Sign Inconsistent={gate_summary['sign_consistency_rejected']} | "
            f"Low Discoveries={gate_summary['discovery_count_rejected']} | "
            f"Dominated={gate_summary['dominated_by_parent_rejected']} | "
            f"Final Accepted={len(final_registry)}"
        )

        # Save gate summary as attribute to extract later
        self.gate_summary = gate_summary

        return final_registry.sort_values("hurdle_excess", ascending=False)


class EconomicRuleConsolidator:
    """
    Economics-first replacement for RuleConsolidator.
    """

    def __init__(
        self,
        metadata: List[FeatureMetadata],
        cfg: Dict[str, Any],
        mask_resolver: Optional[
            Union[CanonicalRuleMaskResolver, DictionaryMaskResolver]
        ] = None,
        scorer: Optional[RuleScorer] = None,
    ):
        self.metadata = metadata
        self.cfg = cfg
        self.mask_resolver = mask_resolver
        self.scorer = scorer or RuleScorer(metadata, cfg, mask_resolver=mask_resolver)
        self.lineage = LineageTracker()
        self._symbol_groups_cache: Optional[Dict[str, np.ndarray]] = None

    def _make_composite_key(self, key_a: str, key_b: str) -> str:
        ordered = sorted([key_a, key_b])
        return f"Composite({ordered[0]})_OR_({ordered[1]})"

    def _build_rule_profile(
        self,
        row: pd.Series,
        mask: np.ndarray,
        fwd_ret: np.ndarray,
        folds: List[Tuple[np.ndarray, np.ndarray]],
    ) -> Dict[str, Any]:
        fold_returns = []
        fold_ics = []
        fold_stds = []

        mask_f = mask.astype(np.float32)
        target_ret = -fwd_ret if row.get("side") == "short" else fwd_ret

        for tr_idx, va_idx in folds:
            mask_va = mask[va_idx]
            y_va = target_ret[va_idx]

            # fold mean return
            selected_returns = y_va[mask_va]
            valid_returns = selected_returns[np.isfinite(selected_returns)]
            if len(valid_returns) > 0:
                fold_returns.append(float(np.mean(valid_returns)))
                clipped_returns = _clip_returns(valid_returns)
                fold_stds.append(float(np.std(clipped_returns)))
            else:
                fold_returns.append(np.nan)
                fold_stds.append(np.nan)

            # fold IC
            valid_idx = np.isfinite(y_va) & np.isfinite(mask_f[va_idx])
            ic = _safe_spearman(mask_f[va_idx][valid_idx], y_va[valid_idx])
            fold_ics.append(ic)

        returns_arr = np.array(fold_returns, dtype=float)
        stds_arr = np.array(fold_stds, dtype=float)
        ics_arr = np.array(fold_ics, dtype=float)

        mean_net_ret = (
            float(np.nanmean(returns_arr))
            if not np.all(np.isnan(returns_arr))
            else np.nan
        )
        std_net_ret = (
            float(np.nanmean(stds_arr)) if not np.all(np.isnan(stds_arr)) else np.nan
        )
        mean_ic = (
            float(np.nanmean(ics_arr)) if not np.all(np.isnan(ics_arr)) else np.nan
        )

        positive_fold_fraction = (
            float(np.mean(returns_arr[np.isfinite(returns_arr)] > 0))
            if np.any(np.isfinite(returns_arr))
            else 0.0
        )
        ic_positive_fold_fraction = (
            float(np.mean(ics_arr[np.isfinite(ics_arr)] > 0))
            if np.any(np.isfinite(ics_arr))
            else 0.0
        )

        sharpe = (
            mean_net_ret / max(std_net_ret, 1e-12)
            if np.isfinite(mean_net_ret) and np.isfinite(std_net_ret)
            else np.nan
        )

        selection_score = (
            sharpe * max(mean_ic, 0.0)
            if np.isfinite(sharpe) and np.isfinite(mean_ic)
            else -np.inf
        )

        all_valid_ret = target_ret[mask]
        all_valid_ret = all_valid_ret[np.isfinite(all_valid_ret)]
        win_rate = float(np.mean(all_valid_ret > 0)) if len(all_valid_ret) > 0 else 0.0

        return {
            "rule_id": row.get("rule_id", getattr(row, "name", "unknown")),
            "rule_name": row["canonical_key"],
            "support_count": int(mask.sum()),
            "support_pct": float(mask.sum() / max(len(mask), 1)),
            "mean_net_ret": mean_net_ret,
            "std_net_ret": std_net_ret,
            "mean_within_fold_std": float(np.nanmean(stds_arr))
            if not np.all(np.isnan(stds_arr))
            else np.nan,
            "sharpe": sharpe,
            "mean_ic": mean_ic,
            "ic_positive_fold_fraction": ic_positive_fold_fraction,
            "positive_fold_fraction": positive_fold_fraction,
            "fold_mean_return_vector": returns_arr,
            "win_rate": win_rate,
            "selection_score": selection_score,
            "mask": mask,
        }

    def _score_candidate_pair(
        self, profile_a: Dict[str, Any], profile_b: Dict[str, Any]
    ) -> Dict[str, Any]:
        mask_a = profile_a["mask"]
        mask_b = profile_b["mask"]
        eps = float(self.cfg.get("econ_eps", 1e-12))

        intersection = (mask_a & mask_b).sum()
        union = (mask_a | mask_b).sum()
        support_a = mask_a.sum()
        support_b = mask_b.sum()

        jaccard = intersection / max(union, eps)
        overlap_coeff = intersection / max(min(support_a, support_b), eps)
        contain_ab = intersection / max(support_a, eps)
        contain_ba = intersection / max(support_b, eps)
        containment_score = max(contain_ab, contain_ba)

        def zscore_safe(v):
            if np.std(v) == 0:
                return np.zeros_like(v)
            return (v - np.mean(v)) / np.std(v)

        # behavior vector
        behavior_vector_a = np.array(
            [
                profile_a["sharpe"],
                profile_a["mean_ic"],
                profile_a["positive_fold_fraction"],
                profile_a["ic_positive_fold_fraction"],
                profile_a["win_rate"],
                profile_a["std_net_ret"],
            ],
            dtype=float,
        )
        behavior_vector_b = np.array(
            [
                profile_b["sharpe"],
                profile_b["mean_ic"],
                profile_b["positive_fold_fraction"],
                profile_b["ic_positive_fold_fraction"],
                profile_b["win_rate"],
                profile_b["std_net_ret"],
            ],
            dtype=float,
        )

        behavior_similarity = _cosine_similarity(
            zscore_safe(behavior_vector_a), zscore_safe(behavior_vector_b)
        )

        fold_similarity = _safe_corr(
            profile_a["fold_mean_return_vector"], profile_b["fold_mean_return_vector"]
        )
        if np.isnan(fold_similarity):
            fold_similarity = 0.0

        w_containment = float(self.cfg.get("econ_weight_containment", 0.35))
        w_overlap_coeff = float(self.cfg.get("econ_weight_overlap_coeff", 0.10))
        w_behavior_similarity = float(
            self.cfg.get("econ_weight_behavior_similarity", 0.30)
        )
        w_fold_similarity = float(self.cfg.get("econ_weight_fold_similarity", 0.25))

        pair_retrieval_score = (
            w_containment * containment_score
            + w_overlap_coeff * overlap_coeff
            + w_behavior_similarity * max(behavior_similarity, 0.0)
            + w_fold_similarity * max(fold_similarity, 0.0)
        )

        return {
            "pair_retrieval_score": pair_retrieval_score,
            "containment_score": containment_score,
            "overlap_coeff": overlap_coeff,
            "jaccard": jaccard,
            "behavior_similarity": behavior_similarity,
            "fold_similarity": fold_similarity,
        }

    def _generate_candidate_pairs(
        self, profiles: Dict[str, Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        pairs = []
        keys = sorted(profiles.keys())
        min_pair_score = float(self.cfg.get("econ_min_pair_score", 0.15))

        for i, key_a in enumerate(keys):
            for key_b in keys[i + 1 :]:
                prof_a = profiles[key_a]
                prof_b = profiles[key_b]
                pair_score_dict = self._score_candidate_pair(prof_a, prof_b)
                if pair_score_dict["pair_retrieval_score"] >= min_pair_score:
                    pairs.append({"key_a": key_a, "key_b": key_b, **pair_score_dict})

        # Sort desc by retrieval score
        pairs.sort(key=lambda x: -x["pair_retrieval_score"])
        return pairs

    def _evaluate_pair_economically(
        self,
        key_a: str,
        key_b: str,
        resolver,
        fwd_ret: np.ndarray,
        folds: List[Tuple[np.ndarray, np.ndarray]],
    ) -> Dict[str, Any]:
        from sklearn.linear_model import Ridge

        min_train = int(self.cfg.get("merge_ridge_min_train", 20))
        min_valid = int(self.cfg.get("merge_ridge_min_valid", 10))
        mask_a_full = resolver.get_mask(key_a).astype(np.float32)
        mask_b_full = resolver.get_mask(key_b).astype(np.float32)
        union_full = (mask_a_full > 0) | (mask_b_full > 0)

        side = resolver.get_rule_side(key_a)
        target_ret = -fwd_ret if side == "short" else fwd_ret

        eps = 1e-12
        candidates = ["composite_or", "parent_a_only", "parent_b_only"]
        fold_metrics = {c: {"returns": [], "stds": [], "ics": []} for c in candidates}

        for tr_idx, va_idx in folds:
            tr_union = union_full[tr_idx]
            va_union = union_full[va_idx]

            mask_a_va = mask_a_full[va_idx]
            mask_b_va = mask_b_full[va_idx]
            y_va_full = target_ret[va_idx]

            preds = np.zeros_like(y_va_full)
            if np.sum(tr_union) >= min_train and np.sum(va_union) >= min_valid:
                X_tr = np.column_stack(
                    [mask_a_full[tr_idx][tr_union], mask_b_full[tr_idx][tr_union]]
                ).astype(np.float32)
                y_tr = target_ret[tr_idx][tr_union].astype(np.float32, copy=True)
                X_va = np.column_stack(
                    [mask_a_full[va_idx][va_union], mask_b_full[va_idx][va_union]]
                ).astype(np.float32)
                y_va = target_ret[va_idx][va_union].astype(np.float32, copy=False)

                if X_tr.shape[0] >= min_train and X_va.shape[0] >= min_valid:
                    hi = np.nanquantile(y_tr, 0.98)
                    lo = np.nanquantile(y_tr, 0.02)
                    y_tr_clipped = np.clip(y_tr, lo, hi)
                    model = Ridge(alpha=float(self.cfg.get("merge_ridge_alpha", 1.0)))
                    model.fit(X_tr, y_tr_clipped)
                    preds[va_union] = model.predict(X_va)

            scores = {
                "composite_or": preds,
                "parent_a_only": mask_a_va,
                "parent_b_only": mask_b_va,
            }
            trade_masks = {
                "composite_or": preds > 0.0,
                "parent_a_only": mask_a_va > 0,
                "parent_b_only": mask_b_va > 0,
            }

            for cand in candidates:
                cand_score = scores[cand]
                cand_mask = trade_masks[cand]

                selected_returns = y_va_full[cand_mask]
                valid_returns = selected_returns[np.isfinite(selected_returns)]
                if len(valid_returns) > 0:
                    fold_metrics[cand]["returns"].append(float(np.mean(valid_returns)))
                    clipped_returns = _clip_returns(valid_returns)
                    fold_metrics[cand]["stds"].append(float(np.std(clipped_returns)))
                else:
                    fold_metrics[cand]["returns"].append(np.nan)
                    fold_metrics[cand]["stds"].append(np.nan)

                valid_idx = np.isfinite(y_va_full) & np.isfinite(cand_score)
                ic = _safe_spearman(cand_score[valid_idx], y_va_full[valid_idx])
                fold_metrics[cand]["ics"].append(ic)

        agg_metrics = {}
        for cand in candidates:
            returns_arr = np.array(fold_metrics[cand]["returns"], dtype=float)
            stds_arr = np.array(fold_metrics[cand]["stds"], dtype=float)
            ics_arr = np.array(fold_metrics[cand]["ics"], dtype=float)

            mean_net_ret = (
                float(np.nanmean(returns_arr))
                if not np.all(np.isnan(returns_arr))
                else np.nan
            )
            std_net_ret = (
                float(np.nanmean(stds_arr))
                if not np.all(np.isnan(stds_arr))
                else np.nan
            )
            mean_ic = (
                float(np.nanmean(ics_arr)) if not np.all(np.isnan(ics_arr)) else np.nan
            )
            positive_fold_fraction = (
                float(np.mean(returns_arr[np.isfinite(returns_arr)] > 0))
                if np.any(np.isfinite(returns_arr))
                else 0.0
            )

            if np.isfinite(mean_net_ret) and np.isfinite(std_net_ret):
                sharpe = mean_net_ret / max(std_net_ret, eps)
            else:
                sharpe = np.nan

            if np.isfinite(sharpe) and np.isfinite(mean_ic):
                selection_score = sharpe * max(mean_ic, 0.0)
            else:
                selection_score = -np.inf

            agg_metrics[cand] = {
                "mean_net_ret": mean_net_ret,
                "std_net_ret": std_net_ret,
                "mean_within_fold_std": float(np.nanmean(stds_arr))
                if not np.all(np.isnan(stds_arr))
                else np.nan,
                "sharpe": sharpe,
                "mean_ic": mean_ic,
                "positive_fold_fraction": positive_fold_fraction,
                "selection_score": selection_score,
            }

        score_a = agg_metrics["parent_a_only"]["selection_score"]
        score_b = agg_metrics["parent_b_only"]["selection_score"]

        if score_a >= score_b:
            better_parent_name = "parent_a_only"
            worse_parent_name = "parent_b_only"
        else:
            better_parent_name = "parent_b_only"
            worse_parent_name = "parent_a_only"

        winner = "composite_or"
        best_score = agg_metrics["composite_or"]["selection_score"]

        if score_a >= best_score:
            winner = "parent_a_only"
            best_score = score_a
        if score_b >= best_score:
            winner = "parent_b_only"
            best_score = score_b

        accept_merge = False
        reason = "accepted"
        decision = ""

        child = agg_metrics[winner]
        parent = agg_metrics[better_parent_name]
        worse_parent = agg_metrics[worse_parent_name]

        if winner != "composite_or":
            accept_merge = False
            reason = "composite_lost_to_parent"
            decision = f"keep_{winner.replace('_only', '')}"
        else:
            min_abs_sharpe_delta = float(
                self.cfg.get("econ_min_abs_sharpe_delta", 0.02)
            )
            min_abs_ic_delta = float(self.cfg.get("econ_min_abs_ic_delta", 0.002))
            mult_sharpe = float(
                self.cfg.get("econ_child_sharpe_improvement_mult", 1.05)
            )
            mult_ic = float(self.cfg.get("econ_child_ic_improvement_mult", 1.05))

            if np.isfinite(parent["sharpe"]) and parent["sharpe"] > 0:
                sharpe_ok = child["sharpe"] > parent["sharpe"] * mult_sharpe
            else:
                sharpe_ok = (
                    child["sharpe"]
                    > (parent["sharpe"] if np.isfinite(parent["sharpe"]) else 0)
                    + min_abs_sharpe_delta
                )

            if np.isfinite(parent["mean_ic"]) and parent["mean_ic"] > 0:
                ic_ok = child["mean_ic"] > parent["mean_ic"] * mult_ic
            else:
                ic_ok = (
                    child["mean_ic"]
                    > (parent["mean_ic"] if np.isfinite(parent["mean_ic"]) else 0)
                    + min_abs_ic_delta
                )

            worst_parent_std = (
                worse_parent["std_net_ret"]
                if np.isfinite(worse_parent["std_net_ret"])
                else np.inf
            )
            risk_ok = child["std_net_ret"] <= worst_parent_std

            if not sharpe_ok:
                reason = "failed_sharpe_improvement"
            elif not ic_ok:
                reason = "failed_ic_improvement"
            elif not risk_ok:
                reason = "failed_std_constraint"
            else:
                accept_merge = True

            if not accept_merge:
                decision = f"keep_{better_parent_name.replace('_only', '')}"
            else:
                decision = "accepted_composite"

        return {
            "child_candidate_name": winner,
            "child_selection_score": child["selection_score"],
            "child_mean_net_ret": child["mean_net_ret"],
            "child_std_net_ret": child["std_net_ret"],
            "child_sharpe": child["sharpe"],
            "child_mean_ic": child["mean_ic"],
            "child_positive_fold_fraction": child["positive_fold_fraction"],
            "better_parent_name": better_parent_name,
            "parent_selection_score": parent["selection_score"],
            "parent_mean_net_ret": parent["mean_net_ret"],
            "parent_std_net_ret": parent["std_net_ret"],
            "parent_sharpe": parent["sharpe"],
            "parent_mean_ic": parent["mean_ic"],
            "worse_parent_name": worse_parent_name,
            "worse_parent_std_net_ret": worse_parent["std_net_ret"],
            "composite_selection_score": agg_metrics["composite_or"]["selection_score"],
            "parent_a_selection_score": score_a,
            "parent_b_selection_score": score_b,
            "accept_merge": accept_merge,
            "decision": decision,
            "decision_reason": reason,
            "ridge_mean_net_ret": agg_metrics["composite_or"]["mean_net_ret"],
            "ridge_positive_fold_fraction": agg_metrics["composite_or"][
                "positive_fold_fraction"
            ],
        }

    def _is_near_duplicate(self, pair_score_dict: Dict[str, Any]) -> bool:
        cont_thres = float(self.cfg.get("econ_duplicate_containment_threshold", 0.97))
        beh_thres = float(
            self.cfg.get("econ_duplicate_behavior_similarity_threshold", 0.90)
        )
        return (
            pair_score_dict["containment_score"] >= cont_thres
            and pair_score_dict["behavior_similarity"] >= beh_thres
        )

    def _compute_ridge_signature(
        self,
        mask: np.ndarray,
        X_std: np.ndarray,
        fwd_ret: np.ndarray,
        folds: List[Tuple[np.ndarray, np.ndarray]],
    ) -> Dict[str, Any]:
        import scipy.stats
        from sklearn.linear_model import Ridge

        min_train = int(self.cfg.get("merge_ridge_min_train", 20))

        fold_ics = []
        fold_returns = []

        train_support_total = 0

        valid_idx = np.where(mask & np.isfinite(fwd_ret))[0]

        if len(valid_idx) < min_train:
            return None

        X_valid = X_std[valid_idx]
        y_valid = fwd_ret[valid_idx]

        hi = np.nanquantile(y_valid, 0.98) if len(y_valid) > 0 else 1.0
        lo = np.nanquantile(y_valid, 0.02) if len(y_valid) > 0 else -1.0
        y_valid_clipped = np.clip(y_valid, lo, hi)

        full_model = Ridge(solver="auto")
        full_model.fit(X_valid, y_valid_clipped)

        coefs = full_model.coef_
        intercept = full_model.intercept_

        abs_coefs = np.abs(coefs)

        k = min(10, len(coefs))
        top_k_indices = set(np.argsort(abs_coefs)[-k:]) if k > 0 else set()

        for tr_idx, va_idx in folds:
            tr_mask = mask[tr_idx]
            va_mask = mask[va_idx]

            y_tr = fwd_ret[tr_idx]
            y_va = fwd_ret[va_idx]

            valid_tr = tr_mask & np.isfinite(y_tr)
            valid_va = va_mask & np.isfinite(y_va)

            if np.sum(valid_tr) >= min_train and np.sum(valid_va) > 0:
                X_tr = X_std[tr_idx][valid_tr]

                # Compute fold-specific clipping limits
                y_tr_valid = y_tr[valid_tr]
                hi_fold = (
                    np.nanquantile(y_tr_valid, 0.98) if len(y_tr_valid) > 0 else 1.0
                )
                lo_fold = (
                    np.nanquantile(y_tr_valid, 0.02) if len(y_tr_valid) > 0 else -1.0
                )
                y_tr_fold = np.clip(y_tr_valid, lo_fold, hi_fold)

                fold_model = Ridge(solver="auto")
                fold_model.fit(X_tr, y_tr_fold)

                X_va_fold = X_std[va_idx][valid_va]
                preds = fold_model.predict(X_va_fold)

                ic = _safe_spearman(preds, y_va[valid_va])
                fold_ics.append(ic)

                mask_trades = preds > 0
                ret = y_va[valid_va][mask_trades]
                if len(ret) > 0:
                    fold_returns.append(np.mean(ret))

            train_support_total += np.sum(valid_tr)

        oos_ic = np.nanmean(fold_ics) if fold_ics else np.nan
        oos_mean_return = np.nanmean(fold_returns) if fold_returns else np.nan

        return {
            "coefficients": coefs,
            "top_k_indices": top_k_indices,
            "intercept": intercept,
            "oos_ic": oos_ic,
            "oos_mean_return": oos_mean_return,
            "mask_support": len(valid_idx),
            "fold_train_support_total": train_support_total,
        }

    def _score_ridge_similarity(
        self, sig_a: Dict[str, Any], sig_b: Dict[str, Any]
    ) -> Dict[str, float]:
        import scipy.stats

        if sig_a is None or sig_b is None:
            return {
                "ridge_context_similarity": 0.0,
                "coef_sim": 0.0,
                "topk_jaccard": 0.0,
                "topk_sign_agreement": 0.0,
                "topk_sign_conflict_rate": 0.0,
                "rank_corr": 0.0,
            }

        beta_a = sig_a["coefficients"]
        beta_b = sig_b["coefficients"]

        coef_sim = _cosine_similarity(beta_a, beta_b)

        top_a = sig_a["top_k_indices"]
        top_b = sig_b["top_k_indices"]

        overlap = top_a.intersection(top_b)
        union_set = top_a.union(top_b)

        topk_jaccard = len(overlap) / max(len(union_set), 1)

        if overlap:
            overlap_idx = np.array(sorted(overlap), dtype=int)
            topk_sign_agreement = float(
                np.mean(np.sign(beta_a[overlap_idx]) == np.sign(beta_b[overlap_idx]))
            )
            topk_sign_conflict_rate = 1.0 - topk_sign_agreement
        else:
            topk_sign_agreement = 0.0
            topk_sign_conflict_rate = 0.0

        rank_corr = _safe_spearman(
            scipy.stats.rankdata(np.abs(beta_a)), scipy.stats.rankdata(np.abs(beta_b))
        )
        if np.isnan(rank_corr):
            rank_corr = 0.0

        ridge_context_similarity = (
            0.45 * max(coef_sim, 0.0)
            + 0.25 * topk_jaccard
            + 0.20 * topk_sign_agreement
            + 0.10 * max(rank_corr, 0.0)
            - 0.15 * topk_sign_conflict_rate
        )

        return {
            "ridge_context_similarity": ridge_context_similarity,
            "coef_sim": coef_sim,
            "topk_jaccard": topk_jaccard,
            "topk_sign_agreement": topk_sign_agreement,
            "topk_sign_conflict_rate": topk_sign_conflict_rate,
            "rank_corr": rank_corr,
        }

    def _cross_context_transport_test(
        self,
        mask_a: np.ndarray,
        mask_b: np.ndarray,
        X_std: np.ndarray,
        fwd_ret: np.ndarray,
        folds: List[Tuple[np.ndarray, np.ndarray]],
    ) -> Dict[str, float]:
        from sklearn.linear_model import Ridge

        min_train = int(self.cfg.get("merge_ridge_min_train", 20))

        transport_ab_ics = []
        transport_ba_ics = []

        for tr_idx, va_idx in folds:
            tr_mask_a = mask_a[tr_idx]
            va_mask_a = mask_a[va_idx]
            tr_mask_b = mask_b[tr_idx]
            va_mask_b = mask_b[va_idx]

            y_tr = fwd_ret[tr_idx]
            y_va = fwd_ret[va_idx]

            valid_tr_a = tr_mask_a & np.isfinite(y_tr)
            valid_tr_b = tr_mask_b & np.isfinite(y_tr)
            valid_va_a = va_mask_a & np.isfinite(y_va)
            valid_va_b = va_mask_b & np.isfinite(y_va)

            if np.sum(valid_tr_a) >= min_train and np.sum(valid_va_b) > 0:
                X_tr_a = X_std[tr_idx][valid_tr_a]

                # Compute source-specific clipping bounds for A
                y_tr_a_valid = y_tr[valid_tr_a]
                hi_a = (
                    np.nanquantile(y_tr_a_valid, 0.98) if len(y_tr_a_valid) > 0 else 1.0
                )
                lo_a = (
                    np.nanquantile(y_tr_a_valid, 0.02)
                    if len(y_tr_a_valid) > 0
                    else -1.0
                )
                y_tr_a = np.clip(y_tr_a_valid, lo_a, hi_a)

                model_a = Ridge(solver="auto")
                model_a.fit(X_tr_a, y_tr_a)

                X_va_b = X_std[va_idx][valid_va_b]
                preds_ab = model_a.predict(X_va_b)

                ic_ab = _safe_spearman(preds_ab, y_va[valid_va_b])
                transport_ab_ics.append(ic_ab)

            if np.sum(valid_tr_b) >= min_train and np.sum(valid_va_a) > 0:
                X_tr_b = X_std[tr_idx][valid_tr_b]

                # Compute source-specific clipping bounds for B
                y_tr_b_valid = y_tr[valid_tr_b]
                hi_b = (
                    np.nanquantile(y_tr_b_valid, 0.98) if len(y_tr_b_valid) > 0 else 1.0
                )
                lo_b = (
                    np.nanquantile(y_tr_b_valid, 0.02)
                    if len(y_tr_b_valid) > 0
                    else -1.0
                )
                y_tr_b = np.clip(y_tr_b_valid, lo_b, hi_b)

                model_b = Ridge(solver="auto")
                model_b.fit(X_tr_b, y_tr_b)

                X_va_a = X_std[va_idx][valid_va_a]
                preds_ba = model_b.predict(X_va_a)

                ic_ba = _safe_spearman(preds_ba, y_va[valid_va_a])
                transport_ba_ics.append(ic_ba)

        transport_ab_ic = np.nanmean(transport_ab_ics) if transport_ab_ics else np.nan
        transport_ba_ic = np.nanmean(transport_ba_ics) if transport_ba_ics else np.nan

        if np.isnan(transport_ab_ic) or np.isnan(transport_ba_ic):
            transport_mean_ic = np.nan
            transport_min_ic = np.nan
        else:
            transport_mean_ic = 0.5 * (transport_ab_ic + transport_ba_ic)
            transport_min_ic = min(transport_ab_ic, transport_ba_ic)

        return {
            "transport_ab_ic": transport_ab_ic,
            "transport_ba_ic": transport_ba_ic,
            "transport_mean_ic": transport_mean_ic,
            "transport_min_ic": transport_min_ic,
        }

    def _ridge_based_consolidation(
        self,
        active: pd.DataFrame,
        fwd_ret: np.ndarray,
        folds: List[Tuple[np.ndarray, np.ndarray]],
        resolver,
        data: Optional[pd.DataFrame] = None,
        max_rounds: int = 2,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        if getattr(resolver, "X", None) is None:
            return active, {}

        # NOTE: A single standardized matrix X_std is built once from resolver.X
        # for the cheap signature stage, using global column-wise mean/std.
        # This is acceptable as a pragmatic retrieval heuristic for signature similarity.
        X_raw = resolver.X
        X_std = np.zeros_like(X_raw)
        for i in range(X_raw.shape[1]):
            col = X_raw[:, i]
            valid = np.isfinite(col)
            if np.sum(valid) > 0:
                mean = np.mean(col[valid])
                std = np.std(col[valid])
                if std == 0:
                    std = 1.0
                X_std[valid, i] = (col[valid] - mean) / std

        diag = {}
        total_evals = 0
        total_merges = 0

        signature_cache = {}
        mask_cache = {}

        def get_signature(key):
            if key not in signature_cache:
                if key not in mask_cache:
                    mask_cache[key] = resolver.get_mask(key)
                side = resolver.get_rule_side(key)
                target_ret = -fwd_ret if side == "short" else fwd_ret
                signature_cache[key] = self._compute_ridge_signature(
                    mask_cache[key], X_std, target_ret, folds
                )
            return signature_cache[key]

        candidate_seed_keys = set()
        ridge_evals_log = []

        for round_idx in range(max_rounds):
            keys = active["canonical_key"].tolist()

            signatures = {}
            for k in keys:
                sig = get_signature(k)
                if sig is not None:
                    signatures[k] = sig

            if len(signatures) < 2:
                break

            valid_keys = list(signatures.keys())
            pair_sims = []

            for i in range(len(valid_keys)):
                for j in range(i + 1, len(valid_keys)):
                    k_a = valid_keys[i]
                    k_b = valid_keys[j]

                    if round_idx > 0 and (
                        k_a not in candidate_seed_keys
                        and k_b not in candidate_seed_keys
                    ):
                        continue

                    sim_dict = self._score_ridge_similarity(
                        signatures[k_a], signatures[k_b]
                    )
                    pair_sims.append({"key_a": k_a, "key_b": k_b, **sim_dict})

            if not pair_sims:
                break

            pair_sims.sort(key=lambda x: x["ridge_context_similarity"], reverse=True)

            n_top_25 = max(1, len(pair_sims) // 4)
            top_pairs = pair_sims[: min(n_top_25, 1000)]

            transport_results = []
            for p in top_pairs:
                k_a = p["key_a"]
                k_b = p["key_b"]
                if k_a not in mask_cache:
                    mask_cache[k_a] = resolver.get_mask(k_a)
                if k_b not in mask_cache:
                    mask_cache[k_b] = resolver.get_mask(k_b)

                # Here we assume both rules in a pair have the same side
                side = resolver.get_rule_side(k_a)
                target_ret = -fwd_ret if side == "short" else fwd_ret

                trans_metrics = self._cross_context_transport_test(
                    mask_cache[k_a], mask_cache[k_b], X_std, target_ret, folds
                )

                # Only keep pairs with valid symmetric transport
                if not np.isnan(trans_metrics["transport_min_ic"]):
                    transport_results.append({**p, **trans_metrics})

            transport_results.sort(key=lambda x: x["transport_min_ic"], reverse=True)

            if round_idx == 0:
                n_top = max(1, len(transport_results) // 2)
                final_pairs = transport_results[: min(n_top, 500)]
            else:
                n_top = max(1, len(transport_results) // 10)
                final_pairs = transport_results[: min(n_top, 500)]

            accepted_in_round = 0
            created_this_round = set()
            active_keys = set(active["canonical_key"].tolist())

            for p in final_pairs:
                key_a = p["key_a"]
                key_b = p["key_b"]

                if key_a not in active_keys or key_b not in active_keys:
                    continue

                total_evals += 1

                diag_eval = self._evaluate_pair_economically(
                    key_a, key_b, resolver, fwd_ret, folds
                )

                ridge_evals_log.append(
                    {
                        "round": round_idx,
                        **p,
                        "final_merge_decision": diag_eval["accept_merge"],
                        "decision_reason": diag_eval["decision_reason"],
                    }
                )

                if diag_eval["accept_merge"]:
                    accepted_in_round += 1
                    total_merges += 1
                    child_key = self._make_composite_key(key_a, key_b)

                    row_a = active[active["canonical_key"] == key_a].iloc[0]
                    row_b = active[active["canonical_key"] == key_b].iloc[0]

                    parent_context_key = (
                        row_a["parent_context_key"]
                        if row_a["parent_context_key"] == row_b["parent_context_key"]
                        else None
                    )
                    side_a = row_a.get("side", "unknown")
                    side_b = row_b.get("side", "unknown")

                    child_summary, _ = self.scorer.score_key_oos(
                        canonical_key=child_key,
                        fwd_ret=fwd_ret,
                        folds=folds,
                        resolver=resolver,
                        require_uplift=bool(parent_context_key),
                        parent_context_key=parent_context_key,
                        discovery_count=int(
                            row_a["discovery_count"] + row_b["discovery_count"]
                        ),
                        n_instances=int(
                            row_a.get("n_instances", 0) + row_b.get("n_instances", 0)
                        ),
                        pipeline_stage="ridge_composite",
                        explicit_side=side_a if side_a == side_b else "mixed",
                    )

                    active = active[
                        ~active["canonical_key"].isin([key_a, key_b])
                    ].copy()
                    active = pd.concat(
                        [active, pd.DataFrame([child_summary])], ignore_index=True
                    )

                    active_keys.remove(key_a)
                    active_keys.remove(key_b)
                    active_keys.add(child_key)

                    created_this_round.add(child_key)

                    self.lineage.record_merge(
                        child_key,
                        [key_a, key_b],
                        "accepted_ridge_signature_composite",
                        round_idx + 10,
                        {"decision_reason": diag_eval["decision_reason"]},
                    )

            candidate_seed_keys = created_this_round.copy()

            if accepted_in_round == 0:
                break

        diag["ridge_signature_evals"] = total_evals
        diag["ridge_signature_merges"] = total_merges
        diag["ridge_signature_evals_log"] = ridge_evals_log

        return active, diag

    def consolidate(
        self,
        registry: pd.DataFrame,
        fwd_ret: np.ndarray,
        folds: List[Tuple[np.ndarray, np.ndarray]],
        resolver: Optional[
            Union[CanonicalRuleMaskResolver, DictionaryMaskResolver]
        ] = None,
        data: Optional[pd.DataFrame] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        resolver = resolver or self.mask_resolver
        if resolver is None or registry.empty:
            return registry, pd.DataFrame(), {}

        active = registry.copy()
        profiles = {}
        for idx, row in active.iterrows():
            key = str(row.get("canonical_key", row.name))
            mask = resolver.get_mask(key)
            profiles[key] = self._build_rule_profile(row, mask, fwd_ret, folds)

        candidate_pairs = self._generate_candidate_pairs(profiles)

        # Filter out cross-side and cross-stage pairs
        # We explicitly never merge Stage B1 and B2 rules together
        side_lookup = active.set_index("canonical_key")["side"].to_dict()
        stage_lookup = (
            active.set_index("canonical_key")
            .get("pipeline_stage", pd.Series(index=active.index))
            .to_dict()
        )

        def can_merge(ka, kb):
            if side_lookup.get(ka) != side_lookup.get(kb):
                return False

            stage_a = stage_lookup.get(ka)
            stage_b = stage_lookup.get(kb)

            if (
                stage_a == "stage_b_trigger_refinement"
                and stage_b == "stage_b_location_refinement"
            ):
                return False
            if (
                stage_b == "stage_b_trigger_refinement"
                and stage_a == "stage_b_location_refinement"
            ):
                return False

            return True

        candidate_pairs = [
            p for p in candidate_pairs if can_merge(p["key_a"], p["key_b"])
        ]

        prune_dups = bool(self.cfg.get("econ_prune_weaker_duplicates", True))

        max_total_pair_evals = int(self.cfg.get("max_total_pair_evals", 1000))
        evals = 0
        accepted_merges = 0
        pruned_dups = 0

        pair_evals_log = []

        tprint(
            f"Economic Consolidation Start: {len(active)} active rules, {len(candidate_pairs)} candidate pairs."
        )

        for pair in candidate_pairs:
            if evals >= max_total_pair_evals:
                break

            key_a = pair["key_a"]
            key_b = pair["key_b"]

            # Skip if either rule is no longer active
            if (
                key_a not in active["canonical_key"].values
                or key_b not in active["canonical_key"].values
            ):
                continue

            evals += 1

            diag = self._evaluate_pair_economically(
                key_a, key_b, resolver, fwd_ret, folds
            )

            # We construct pair_diag starting with pair_score_dict
            pair_diag = {**pair, **diag}

            row_a = active[active["canonical_key"] == key_a].iloc[0]
            row_b = active[active["canonical_key"] == key_b].iloc[0]

            pair_evals_log.append({**pair, **diag})
            if diag["accept_merge"]:
                accepted_merges += 1
                # Create composite
                child_key = self._make_composite_key(key_a, key_b)

                parent_context_key = (
                    row_a["parent_context_key"]
                    if row_a["parent_context_key"] == row_b["parent_context_key"]
                    else None
                )
                side_a = row_a.get("side", "unknown")
                side_b = row_b.get("side", "unknown")

                child_summary, _ = self.scorer.score_key_oos(
                    canonical_key=child_key,
                    fwd_ret=fwd_ret,
                    folds=folds,
                    resolver=resolver,
                    require_uplift=bool(parent_context_key),
                    parent_context_key=parent_context_key,
                    discovery_count=int(
                        row_a["discovery_count"] + row_b["discovery_count"]
                    ),
                    n_instances=int(
                        row_a.get("n_instances", 0) + row_b.get("n_instances", 0)
                    ),
                    pipeline_stage="global_composite"
                    if "global" in str(row_a.get("pipeline_stage", ""))
                    or "global" in str(row_b.get("pipeline_stage", ""))
                    else "composite",
                    explicit_side=side_a if side_a == side_b else "mixed",
                )

                # We enforce that the created child summary takes the place of A and B
                active = active[~active["canonical_key"].isin([key_a, key_b])].copy()
                active = pd.concat(
                    [active, pd.DataFrame([child_summary])], ignore_index=True
                )

                self.lineage.record_merge(
                    child_key,
                    [key_a, key_b],
                    "accepted_composite",
                    1,
                    {"decision_reason": diag["decision_reason"]},
                )

                # We must build a new profile for the newly created child to allow future merges
                mask = resolver.get_mask(child_key)
                profiles[child_key] = self._build_rule_profile(
                    child_summary, mask, fwd_ret, folds
                )

            else:
                # Rejected merge -> check for duplicate pruning
                is_dup = self._is_near_duplicate(pair)

                better_parent_key = (
                    key_a if diag["better_parent_name"] == "parent_a_only" else key_b
                )
                weaker_parent_key = key_b if better_parent_key == key_a else key_a

                if prune_dups and is_dup:
                    pruned_dups += 1
                    active = active[active["canonical_key"] != weaker_parent_key].copy()
                    self.lineage.record_merge(
                        better_parent_key,
                        [key_a, key_b],
                        "prune_duplicate",
                        1,
                        {"decision_reason": "duplicate_pruned"},
                    )
                else:
                    self.lineage.record_merge(
                        "none",
                        [key_a, key_b],
                        "rejected_pair",
                        1,
                        {"decision_reason": diag["decision_reason"]},
                    )

        active = active.drop_duplicates(subset=["canonical_key"], keep="first")
        active = active.sort_values(
            ["composite_score", "hurdle_excess"], ascending=False
        )

        tprint(
            f"Economic Consolidation End: {len(active)} rules remaining. {evals} evaluated pairs, {accepted_merges} merges, {pruned_dups} duplicate prunes."
        )

        # Diagnostic prints
        evals_df = pd.DataFrame(pair_evals_log)
        if not evals_df.empty:
            rejected_pairs = (
                evals_df[~evals_df["accept_merge"]]
                .sort_values("pair_retrieval_score", ascending=False)
                .head(10)
            )
            tprint("Top 10 rejected pairs by retrieval score:")
            for _, row in rejected_pairs.iterrows():
                tprint(
                    f"  - {row['key_a']} + {row['key_b']}: reason={row['decision_reason']}, retrieval={row['pair_retrieval_score']:.2f}"
                )

            accepted_pairs = (
                evals_df[evals_df["accept_merge"]]
                .sort_values("child_selection_score", ascending=False)
                .head(10)
            )
            tprint("Top 10 accepted composites by child selection score:")
            for _, row in accepted_pairs.iterrows():
                tprint(
                    f"  - {row['key_a']} + {row['key_b']}: score={row['child_selection_score']:.2f}, gain={row['child_selection_score'] - row['parent_selection_score']:.2f}"
                )

        # Now run the Ridge-based signature consolidation on the resulting active registry
        active, ridge_diag = self._ridge_based_consolidation(
            active, fwd_ret, folds, resolver, data, max_rounds=2
        )

        diag_dict = {
            "economic_pair_candidates": pd.DataFrame(candidate_pairs),
            "economic_pair_evaluations": evals_df,
            "economic_consolidation_summary": {
                "active_start": len(registry),
                "active_end": len(active),
                "pairs_generated": len(candidate_pairs),
                "evals_count": evals,
                "accepted_merges": accepted_merges,
                "duplicate_prunes": pruned_dups,
            },
            "ridge_signature_evals": ridge_diag.get("ridge_signature_evals", 0),
            "ridge_signature_merges": ridge_diag.get("ridge_signature_merges", 0),
        }

        if (
            "ridge_signature_evals_log" in ridge_diag
            and ridge_diag["ridge_signature_evals_log"]
        ):
            diag_dict["ridge_signature_evals_log"] = pd.DataFrame(
                ridge_diag["ridge_signature_evals_log"]
            )

        tprint(
            f"Ridge Signature Consolidation End: {len(active)} rules remaining. {ridge_diag.get('ridge_signature_evals', 0)} evaluated pairs, {ridge_diag.get('ridge_signature_merges', 0)} merges."
        )

        return active, self.lineage.get_audit_df(), diag_dict


class RuleConsolidator:
    def __init__(
        self,
        metadata: List[FeatureMetadata],
        cfg: Dict[str, Any],
        mask_resolver: Optional[
            Union[CanonicalRuleMaskResolver, DictionaryMaskResolver]
        ] = None,
        scorer: Optional[RuleScorer] = None,
    ):
        self.metadata = metadata
        self.cfg = cfg
        self.mask_resolver = mask_resolver
        self.scorer = scorer or RuleScorer(metadata, cfg, mask_resolver=mask_resolver)
        self.lineage = LineageTracker()
        self._symbol_groups_cache: Optional[Dict[str, np.ndarray]] = None

    def _build_symbol_groups(
        self, data: Optional[pd.DataFrame]
    ) -> Dict[str, np.ndarray]:
        if self._symbol_groups_cache is not None:
            return self._symbol_groups_cache
        if data is None or "symbol" not in data.columns:
            self._symbol_groups_cache = {}
            return self._symbol_groups_cache
        groups: Dict[str, np.ndarray] = {}
        for symbol, idx in data.groupby("symbol", sort=False).groups.items():
            groups[str(symbol)] = np.asarray(sorted(idx), dtype=np.int32)
        self._symbol_groups_cache = groups
        return groups

    def _dilate_mask_by_symbol(
        self,
        mask: np.ndarray,
        data: Optional[pd.DataFrame],
        bars: int,
    ) -> np.ndarray:
        if bars <= 0:
            return np.asarray(mask, dtype=bool)
        mask_bool = np.asarray(mask, dtype=bool)
        out = mask_bool.copy()
        symbol_groups = self._build_symbol_groups(data)
        if not symbol_groups:
            active = np.flatnonzero(mask_bool)
            for idx in active:
                lo = max(0, idx - bars)
                hi = min(mask_bool.shape[0], idx + bars + 1)
                out[lo:hi] = True
            return out
        for idx_group in symbol_groups.values():
            local_mask = mask_bool[idx_group]
            active_local = np.flatnonzero(local_mask)
            for pos in active_local:
                lo = max(0, pos - bars)
                hi = min(idx_group.shape[0], pos + bars + 1)
                out[idx_group[lo:hi]] = True
        return out

    def _extract_context_signature(self, row: pd.Series) -> Tuple[str, str]:
        parent_context_key = row.get("parent_context_key")
        if isinstance(parent_context_key, str) and parent_context_key:
            try:
                slots = parse_slot_map(
                    parent_context_key, ("trigger", "location", "regime")
                )
                return slots.get("location", "*"), slots.get("regime", "*")
            except Exception:
                pass
        key = row.get("canonical_key", row.name)
        if not isinstance(key, str) or not key:
            return "*", "*"
        if split_composite_key(key) is not None:
            return "*", "*"
        try:
            slots = parse_slot_map(key, ("trigger", "location", "regime"))
            return slots.get("location", "*"), slots.get("regime", "*")
        except Exception:
            return "*", "*"

    def _semantic_relation(self, row_a: pd.Series, row_b: pd.Series) -> Optional[str]:
        loc_a, reg_a = self._extract_context_signature(row_a)
        loc_b, reg_b = self._extract_context_signature(row_b)
        same_loc = loc_a != "*" and loc_a == loc_b
        same_reg = reg_a != "*" and reg_a == reg_b
        if same_loc and same_reg:
            return "same_regime_location"
        if same_reg:
            return "same_regime_only"
        if same_loc:
            return "same_location_only"
        return None

    def _pair_score(
        self,
        key_a: str,
        key_b: str,
        row_a: pd.Series,
        row_b: pd.Series,
        resolver: Union[CanonicalRuleMaskResolver, DictionaryMaskResolver],
        data: Optional[pd.DataFrame],
        dilation_bars: int,
    ) -> Tuple[int, float, Optional[str], float]:
        relation = self._semantic_relation(row_a, row_b)
        dilated_jaccard = self._pair_jaccard(
            key_a,
            key_b,
            resolver,
            data=data,
            dilation_bars=dilation_bars,
        )
        if relation == "same_regime_location":
            return 0, dilated_jaccard, relation, dilated_jaccard
        if dilated_jaccard >= float(self.cfg.get("min_jaccard_stop_threshold", 0.60)):
            return 1, dilated_jaccard, "dilated_jaccard", dilated_jaccard
        if relation == "same_regime_only":
            return 2, dilated_jaccard, relation, dilated_jaccard
        if relation == "same_location_only":
            return 3, dilated_jaccard, relation, dilated_jaccard
        return 99, dilated_jaccard, relation, dilated_jaccard

    def _pair_jaccard(
        self,
        key_a: str,
        key_b: str,
        resolver: Union[CanonicalRuleMaskResolver, DictionaryMaskResolver],
        data: Optional[pd.DataFrame] = None,
        dilation_bars: int = 0,
    ) -> float:
        mask_a = resolver.get_mask(key_a)
        mask_b = resolver.get_mask(key_b)
        if dilation_bars > 0:
            mask_a = self._dilate_mask_by_symbol(mask_a, data, dilation_bars)
            mask_b = self._dilate_mask_by_symbol(mask_b, data, dilation_bars)
        union = np.sum(mask_a | mask_b)
        if union == 0:
            return 0.0
        return float(np.sum(mask_a & mask_b) / union)

    def _make_composite_key(self, key_a: str, key_b: str) -> str:
        ordered = sorted([key_a, key_b])
        return f"Composite({ordered[0]})_OR_({ordered[1]})"

    def _best_parent(self, row_a: pd.Series, row_b: pd.Series) -> pd.Series:
        metric_a = (row_a["hurdle_excess"], row_a["composite_score"])
        metric_b = (row_b["hurdle_excess"], row_b["composite_score"])
        return row_a if metric_a >= metric_b else row_b

    @staticmethod
    def _row_key(row: pd.Series) -> str:
        key = row.get("canonical_key", row.name)
        if key is None:
            raise KeyError("Row has no canonical_key column or index name")
        return str(key)

    def consolidate(
        self,
        registry: pd.DataFrame,
        fwd_ret: np.ndarray,
        folds: List[Tuple[np.ndarray, np.ndarray]],
        resolver: Optional[
            Union[CanonicalRuleMaskResolver, DictionaryMaskResolver]
        ] = None,
        data: Optional[pd.DataFrame] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        resolver = resolver or self.mask_resolver
        if resolver is None or registry.empty:
            return registry, pd.DataFrame(), {}

        round_summaries = []
        ridge_evals = []

        top_n = int(self.cfg.get("consolidation_top_n", 100))
        support_gain_factor = float(self.cfg.get("support_gain_factor", 1.05))
        max_failed_pair_checks = int(self.cfg.get("max_failed_pair_checks", 250))
        max_total_pair_evals = int(self.cfg.get("max_total_pair_evals", 1000))
        max_rounds = int(self.cfg.get("max_consolidation_rounds", 3))
        min_jaccard_floor = float(self.cfg.get("min_jaccard_stop_threshold", 0.60))
        dilation_bars = int(self.cfg.get("consolidation_dilation_bars", 2))
        round_thresholds = [
            float(self.cfg.get("jaccard_round1", 0.95)),
            float(self.cfg.get("jaccard_round2", 0.90)),
            float(self.cfg.get("jaccard_round3", 0.75)),
        ]

        active = (
            registry.sort_values(["composite_score", "hurdle_excess"], ascending=False)
            .head(top_n)
            .copy()
        )
        tried_pairs: set[Tuple[str, str]] = set()
        total_pair_evals = 0

        for round_num in range(1, max_rounds + 1):
            start_count = len(active)
            tprint(
                f"Legacy Consolidation Round {round_num} Start: {start_count} active rules"
            )
            threshold = round_thresholds[min(round_num - 1, len(round_thresholds) - 1)]
            candidates: List[Tuple[int, float, str, str, str, float]] = []
            keys = sorted(active["canonical_key"].tolist())
            rows_by_key = active.set_index("canonical_key")
            for key_a, key_b in itertools.combinations(keys, 2):
                pair = (key_a, key_b)
                if pair in tried_pairs:
                    continue
                if key_a not in rows_by_key.index or key_b not in rows_by_key.index:
                    continue
                row_a = rows_by_key.loc[key_a]
                row_b = rows_by_key.loc[key_b]

                if row_a.get("side") != row_b.get("side"):
                    continue

                priority, rank_score, source, dilated_jaccard = self._pair_score(
                    key_a,
                    key_b,
                    row_a,
                    row_b,
                    resolver,
                    data,
                    dilation_bars,
                )
                if priority == 99:
                    continue
                candidates.append(
                    (
                        priority,
                        rank_score,
                        key_a,
                        key_b,
                        source or "unknown",
                        dilated_jaccard,
                    )
                )

            if not candidates:
                break
            candidates.sort(key=lambda item: (item[0], -item[1], item[2], item[3]))
            if candidates[0][0] == 1 and candidates[0][1] < max(
                threshold, min_jaccard_floor
            ):
                break

            accepted_merges = 0
            failed_checks = 0

            for (
                priority,
                jaccard,
                key_a,
                key_b,
                merge_source,
                dilated_jaccard,
            ) in candidates:
                if priority == 1 and jaccard < threshold:
                    break
                if total_pair_evals >= max_total_pair_evals:
                    break
                pair = (key_a, key_b)
                tried_pairs.add(pair)
                total_pair_evals += 1

                if key_a not in rows_by_key.index or key_b not in rows_by_key.index:
                    continue

                row_a = rows_by_key.loc[key_a]
                row_b = rows_by_key.loc[key_b]
                side_a = row_a["side"]
                side_b = row_b["side"]
                if side_a not in ("unknown", side_b) and side_b not in (
                    "unknown",
                    side_a,
                ):
                    self.lineage.record_merge(
                        key_a,
                        [key_a, key_b],
                        "reject_sign_conflict",
                        round_num,
                        {"jaccard": jaccard, "merge_source": merge_source},
                    )
                    failed_checks += 1
                    if failed_checks >= max_failed_pair_checks:
                        break
                    continue

                best_parent = self._best_parent(row_a, row_b)
                best_parent_key = self._row_key(best_parent)
                weaker_parent = row_b if best_parent_key == key_a else row_a
                weaker_parent_key = self._row_key(weaker_parent)
                ridge_diag = self._evaluate_ridge_pair(
                    key_a,
                    key_b,
                    resolver,
                    fwd_ret,
                    folds,
                )

                ridge_evals.append(
                    {
                        "round": round_num,
                        "key_a": key_a,
                        "key_b": key_b,
                        "merge_source": merge_source,
                        "priority": priority,
                        "jaccard": jaccard,
                        "dilated_jaccard": dilated_jaccard,
                        **ridge_diag,
                    }
                )

                if (
                    merge_source == "dilated_jaccard"
                    and dilated_jaccard >= float(self.cfg.get("jaccard_round1", 0.95))
                    and ridge_diag["ridge_positive_fold_fraction"]
                    < float(self.cfg.get("merge_ridge_positive_fold_fraction", 0.50))
                ):
                    active = active[active["canonical_key"] != weaker_parent_key]
                    self.lineage.record_merge(
                        best_parent_key,
                        [key_a, key_b],
                        "keep_stronger_parent",
                        round_num,
                        {"jaccard": jaccard, "merge_source": merge_source},
                    )
                    accepted_merges += 1
                    rows_by_key = active.set_index("canonical_key")
                    continue

                child_key = self._make_composite_key(key_a, key_b)
                parent_context_key = (
                    row_a["parent_context_key"]
                    if row_a["parent_context_key"] == row_b["parent_context_key"]
                    else None
                )
                child_summary, _ = self.scorer.score_key_oos(
                    canonical_key=child_key,
                    fwd_ret=fwd_ret,
                    folds=folds,
                    resolver=resolver,
                    require_uplift=bool(parent_context_key),
                    parent_context_key=parent_context_key,
                    discovery_count=int(
                        row_a["discovery_count"] + row_b["discovery_count"]
                    ),
                    n_instances=int(row_a["n_instances"] + row_b["n_instances"]),
                    pipeline_stage="global_composite"
                    if "global" in str(row_a["pipeline_stage"])
                    or "global" in str(row_b["pipeline_stage"])
                    else "composite",
                    explicit_side=side_a if side_a == side_b else "mixed",
                )
                child_summary["merge_source"] = merge_source
                child_summary["ridge_mean_net_ret"] = ridge_diag["ridge_mean_net_ret"]
                child_summary["ridge_positive_fold_fraction"] = ridge_diag[
                    "ridge_positive_fold_fraction"
                ]
                child_summary["ridge_mean_support_pct"] = ridge_diag.get(
                    "ridge_mean_support_pct", 0.0
                )
                child_summary["ridge_vs_parent_gain"] = (
                    ridge_diag["child_mean_net_ret"] - ridge_diag["parent_mean_net_ret"]
                    if np.isfinite(ridge_diag["child_mean_net_ret"])
                    and np.isfinite(ridge_diag["parent_mean_net_ret"])
                    else np.nan
                )

                child_improves_uplift = (
                    np.isfinite(child_summary["mean_uplift"])
                    and np.isfinite(best_parent["mean_uplift"])
                    and child_summary["mean_uplift"] > best_parent["mean_uplift"]
                )
                child_improves_support = child_summary["mean_support_pct"] >= (
                    support_gain_factor * best_parent["mean_support_pct"]
                )
                ridge_supports_merge = ridge_diag["accept_merge"]

                sign_ok = child_summary["sign_consistency"] >= (
                    best_parent["sign_consistency"] - 0.05
                )
                presence_ok = child_summary["presence_freq"] >= (
                    best_parent["presence_freq"] - 0.05
                )

                if (
                    child_summary["accepted"]
                    and child_summary["hurdle_excess"] > 0
                    and sign_ok
                    and presence_ok
                    and (
                        child_improves_uplift
                        or child_improves_support
                        or ridge_supports_merge
                    )
                ):
                    active = active[
                        ~active["canonical_key"].isin([key_a, key_b])
                    ].copy()
                    active = pd.concat(
                        [active, pd.DataFrame([child_summary])], ignore_index=True
                    )
                    self.lineage.record_merge(
                        child_key,
                        [key_a, key_b],
                        "accepted_child_union",
                        round_num,
                        {
                            "jaccard": jaccard,
                            "merge_source": merge_source,
                            "ridge_mean_net_ret": ridge_diag["ridge_mean_net_ret"],
                            "accept_merge_reason": ridge_diag["accept_merge_reason"],
                        },
                    )
                    accepted_merges += 1
                    rows_by_key = active.set_index("canonical_key")
                else:
                    self.lineage.record_merge(
                        child_key,
                        [key_a, key_b],
                        "rejected_child_union",
                        round_num,
                        {
                            "jaccard": jaccard,
                            "accepted": bool(child_summary["accepted"]),
                            "support_gain": child_improves_support,
                            "uplift_gain": child_improves_uplift,
                            "merge_source": merge_source,
                            "ridge_mean_net_ret": ridge_diag["ridge_mean_net_ret"],
                            "ridge_positive_fold_fraction": ridge_diag[
                                "ridge_positive_fold_fraction"
                            ],
                            "accept_merge_reason": ridge_diag["accept_merge_reason"],
                        },
                    )
                    failed_checks += 1
                    if failed_checks >= max_failed_pair_checks:
                        break

            tprint(
                f"Legacy Consolidation Round {round_num} End: {len(active)} active rules. Accepted Merges: {accepted_merges}"
            )
            round_summaries.append(
                {
                    "round": round_num,
                    "start_count": start_count,
                    "end_count": len(active),
                    "accepted_merges": accepted_merges,
                    "failed_checks": failed_checks,
                }
            )

            if accepted_merges == 0 or total_pair_evals >= max_total_pair_evals:
                break

        active = active.drop_duplicates(subset=["canonical_key"], keep="first")
        active = active.sort_values(
            ["composite_score", "hurdle_excess"], ascending=False
        )

        diag_dict = {
            "legacy_consolidation_round_summary": pd.DataFrame(round_summaries),
            "legacy_ridge_pair_eval_audit": pd.DataFrame(ridge_evals),
        }

        if ridge_evals:
            evals_df = pd.DataFrame(ridge_evals)
            reasons = (
                evals_df[~evals_df["accept_merge"]]["accept_merge_reason"]
                .value_counts()
                .head(5)
            )
            tprint("Legacy top rejection reasons:")
            for reason, count in reasons.items():
                tprint(f"  - {reason}: {count}")

        return active, self.lineage.get_audit_df(), diag_dict

    def _evaluate_ridge_pair(
        self,
        key_a: str,
        key_b: str,
        resolver: Union[CanonicalRuleMaskResolver, DictionaryMaskResolver],
        fwd_ret: np.ndarray,
        folds: List[Tuple[np.ndarray, np.ndarray]],
    ) -> Dict[str, Any]:
        from sklearn.linear_model import Ridge

        min_train = int(self.cfg.get("merge_ridge_min_train", 20))
        min_valid = int(self.cfg.get("merge_ridge_min_valid", 10))
        mask_a_full = resolver.get_mask(key_a).astype(np.float32)
        mask_b_full = resolver.get_mask(key_b).astype(np.float32)
        union_full = (mask_a_full > 0) | (mask_b_full > 0)

        side = resolver.get_rule_side(key_a)
        target_ret = -fwd_ret if side == "short" else fwd_ret

        eps = 1e-12
        candidates = ["composite_or", "parent_a_only", "parent_b_only"]
        fold_metrics = {c: {"returns": [], "stds": [], "ics": []} for c in candidates}

        for tr_idx, va_idx in folds:
            tr_union = union_full[tr_idx]
            va_union = union_full[va_idx]

            # Use out-of-sample data
            mask_a_va = mask_a_full[va_idx]
            mask_b_va = mask_b_full[va_idx]
            y_va_full = target_ret[va_idx]

            # 1. Train Ridge for composite
            preds = np.zeros_like(y_va_full)
            if np.sum(tr_union) >= min_train and np.sum(va_union) >= min_valid:
                X_tr = np.column_stack(
                    [mask_a_full[tr_idx][tr_union], mask_b_full[tr_idx][tr_union]]
                ).astype(np.float32)
                y_tr = target_ret[tr_idx][tr_union].astype(np.float32, copy=True)
                X_va = np.column_stack(
                    [mask_a_full[va_idx][va_union], mask_b_full[va_idx][va_union]]
                ).astype(np.float32)
                y_va = target_ret[va_idx][va_union].astype(np.float32, copy=False)

                if X_tr.shape[0] >= min_train and X_va.shape[0] >= min_valid:
                    hi = np.nanquantile(y_tr, 0.98)
                    lo = np.nanquantile(y_tr, 0.02)
                    y_tr_clipped = np.clip(y_tr, lo, hi)
                    model = Ridge(alpha=float(self.cfg.get("merge_ridge_alpha", 1.0)))
                    model.fit(X_tr, y_tr_clipped)
                    preds[va_union] = model.predict(X_va)

            # Gather validation metrics per candidate
            scores = {
                "composite_or": preds,
                "parent_a_only": mask_a_va,
                "parent_b_only": mask_b_va,
            }
            trade_masks = {
                "composite_or": preds > 0.0,
                "parent_a_only": mask_a_va > 0,
                "parent_b_only": mask_b_va > 0,
            }

            for cand in candidates:
                cand_score = scores[cand]
                cand_mask = trade_masks[cand]

                # Mean and Std (on clipped returns)
                selected_returns = y_va_full[cand_mask]
                valid_returns = selected_returns[np.isfinite(selected_returns)]
                if len(valid_returns) > 0:
                    fold_metrics[cand]["returns"].append(float(np.mean(valid_returns)))
                    clipped_returns = _clip_returns(valid_returns)
                    fold_metrics[cand]["stds"].append(float(np.std(clipped_returns)))
                else:
                    fold_metrics[cand]["returns"].append(np.nan)
                    fold_metrics[cand]["stds"].append(np.nan)

                # IC (on all valid rows in the fold)
                valid_idx = np.isfinite(y_va_full) & np.isfinite(cand_score)
                ic = _safe_spearman(cand_score[valid_idx], y_va_full[valid_idx])
                fold_metrics[cand]["ics"].append(ic)

        # Aggregate fold metrics
        agg_metrics = {}
        for cand in candidates:
            returns_arr = np.array(fold_metrics[cand]["returns"], dtype=float)
            stds_arr = np.array(fold_metrics[cand]["stds"], dtype=float)
            ics_arr = np.array(fold_metrics[cand]["ics"], dtype=float)

            mean_net_ret = (
                float(np.nanmean(returns_arr))
                if not np.all(np.isnan(returns_arr))
                else np.nan
            )
            std_net_ret = (
                float(np.nanmean(stds_arr))
                if not np.all(np.isnan(stds_arr))
                else np.nan
            )
            mean_ic = (
                float(np.nanmean(ics_arr)) if not np.all(np.isnan(ics_arr)) else np.nan
            )
            positive_fold_fraction = (
                float(np.mean(returns_arr[np.isfinite(returns_arr)] > 0))
                if np.any(np.isfinite(returns_arr))
                else 0.0
            )

            if np.isfinite(mean_net_ret) and np.isfinite(std_net_ret):
                sharpe = mean_net_ret / max(std_net_ret, eps)
            else:
                sharpe = np.nan

            if np.isfinite(sharpe) and np.isfinite(mean_ic):
                selection_score = sharpe * max(mean_ic, 0.0)
            else:
                selection_score = -np.inf

            agg_metrics[cand] = {
                "mean_net_ret": mean_net_ret,
                "std_net_ret": std_net_ret,
                "mean_within_fold_std": float(np.nanmean(stds_arr))
                if not np.all(np.isnan(stds_arr))
                else np.nan,
                "sharpe": sharpe,
                "mean_ic": mean_ic,
                "positive_fold_fraction": positive_fold_fraction,
                "selection_score": selection_score,
            }

        # Compare parents
        score_a = agg_metrics["parent_a_only"]["selection_score"]
        score_b = agg_metrics["parent_b_only"]["selection_score"]

        if score_a >= score_b:
            better_parent_name = "parent_a_only"
            worse_parent_name = "parent_b_only"
        else:
            better_parent_name = "parent_b_only"
            worse_parent_name = "parent_a_only"

        # Determine winner
        winner = "composite_or"
        best_score = agg_metrics["composite_or"]["selection_score"]

        if score_a >= best_score:
            winner = "parent_a_only"
            best_score = score_a
        if score_b >= best_score:
            winner = "parent_b_only"
            best_score = score_b

        # Acceptance logic
        accept_merge = False
        reason = "accepted"

        child = agg_metrics[winner]
        parent = agg_metrics[better_parent_name]
        worse_parent = agg_metrics[worse_parent_name]

        if winner != "composite_or":
            accept_merge = False
            reason = "composite_lost_to_parent"
        else:
            min_abs_sharpe_delta = 0.02
            min_abs_ic_delta = 0.002

            # Sharpe OK
            if np.isfinite(parent["sharpe"]) and parent["sharpe"] > 0:
                sharpe_ok = child["sharpe"] > parent["sharpe"] * 1.05
            else:
                sharpe_ok = (
                    child["sharpe"]
                    > (parent["sharpe"] if np.isfinite(parent["sharpe"]) else 0)
                    + min_abs_sharpe_delta
                )

            # IC OK
            if np.isfinite(parent["mean_ic"]) and parent["mean_ic"] > 0:
                ic_ok = child["mean_ic"] > parent["mean_ic"] * 1.05
            else:
                ic_ok = (
                    child["mean_ic"]
                    > (parent["mean_ic"] if np.isfinite(parent["mean_ic"]) else 0)
                    + min_abs_ic_delta
                )

            # Risk OK
            worst_parent_std = (
                worse_parent["std_net_ret"]
                if np.isfinite(worse_parent["std_net_ret"])
                else np.inf
            )
            risk_ok = child["std_net_ret"] <= worst_parent_std

            if not sharpe_ok:
                reason = "failed_sharpe_improvement"
            elif not ic_ok:
                reason = "failed_ic_improvement"
            elif not risk_ok:
                reason = "failed_std_constraint"
            else:
                accept_merge = True

        return {
            "child_candidate_name": winner,
            "child_selection_score": child["selection_score"],
            "child_mean_net_ret": child["mean_net_ret"],
            "child_std_net_ret": child["std_net_ret"],
            "child_sharpe": child["sharpe"],
            "child_mean_ic": child["mean_ic"],
            "child_positive_fold_fraction": child["positive_fold_fraction"],
            "better_parent_name": better_parent_name,
            "parent_selection_score": parent["selection_score"],
            "parent_mean_net_ret": parent["mean_net_ret"],
            "parent_std_net_ret": parent["std_net_ret"],
            "parent_sharpe": parent["sharpe"],
            "parent_mean_ic": parent["mean_ic"],
            "worse_parent_name": worse_parent_name,
            "worse_parent_std_net_ret": worse_parent["std_net_ret"],
            "composite_selection_score": agg_metrics["composite_or"]["selection_score"],
            "parent_a_selection_score": score_a,
            "parent_b_selection_score": score_b,
            "accept_merge": accept_merge,
            "accept_merge_reason": reason,
            # Maintain legacy keys for upstream
            "ridge_mean_net_ret": agg_metrics["composite_or"]["mean_net_ret"],
            "ridge_positive_fold_fraction": agg_metrics["composite_or"][
                "positive_fold_fraction"
            ],
        }


def compute_tbm_outcomes_per_symbol(
    data: pd.DataFrame,
    horizon: int,
    tp_atr: float,
    sl_atr: float,
    side: str = "long",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute TBM outcomes independently within each symbol's time series.

    Assumes `data` has columns:
      - symbol
      - timestamp
      - close
      - high
      - low
      - atr

    Returns arrays aligned to `data.index`.
    """
    if data.empty:
        z = np.zeros(0, dtype=np.int8)
        return z, z, z

    # Preserve original row order for final alignment
    out_tp = np.zeros(len(data), dtype=np.int8)
    out_sl = np.zeros(len(data), dtype=np.int8)
    out_to = np.zeros(len(data), dtype=np.int8)

    # Sort once for temporal correctness inside each symbol
    work = data.reset_index(drop=False).rename(columns={"index": "_orig_idx"})
    work = work.sort_values(["symbol", "timestamp"], kind="mergesort")

    for sym, g in work.groupby("symbol", sort=False):
        idx = g["_orig_idx"].to_numpy()

        close = g["close"].to_numpy(dtype=np.float64, copy=False)
        high = g["high"].to_numpy(dtype=np.float64, copy=False)
        low = g["low"].to_numpy(dtype=np.float64, copy=False)
        atr = g["atr"].to_numpy(dtype=np.float64, copy=False)

        if side == "short":
            c, h, l = -close, -low, -high
        else:
            c, h, l = close, high, low

        tp_f, sl_f, to_f = tbm_outcomes_atr_nb(
            close=c,
            high=h,
            low=l,
            atr=atr,
            horizon=horizon,
            tp_atr=tp_atr,
            sl_atr=sl_atr,
        )

        out_tp[idx] = tp_f
        out_sl[idx] = sl_f
        out_to[idx] = to_f

    return out_tp, out_sl, out_to


def build_context_feature_dict_from_registry(
    registry: pd.DataFrame,
    data: pd.DataFrame,
    X_stage_a: np.ndarray,
    metadata_stage_a: List[FeatureMetadata],
) -> Tuple[Dict[str, np.ndarray], Dict[str, str]]:
    if registry.empty:
        return {}, {}

    context_feature_dict: Dict[str, np.ndarray] = {}
    context_feature_to_stage_a_key: Dict[str, str] = {}
    resolver = CanonicalRuleMaskResolver(X_stage_a, metadata_stage_a)

    for _, row in registry.iterrows():
        key = row["canonical_key"]
        ctx_hash = hashlib.sha1(key.encode()).hexdigest()[:8]
        ctx_name = f"ctx__{ctx_hash}"
        mask = resolver.get_mask(key)
        context_feature_dict[ctx_name] = mask.astype(np.float32)
        context_feature_to_stage_a_key[ctx_name] = key

    return context_feature_dict, context_feature_to_stage_a_key


def select_stage_a_contexts(
    stage_a_result: Dict[str, Any], cfg: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    registry = stage_a_result.get("accepted_registry")
    if registry is None or registry.empty:
        return pd.DataFrame(), pd.DataFrame(columns=["reason", "count"])

    registry = registry.copy()
    registry["reject_support"] = registry["mean_support_pct"] < float(
        cfg.get("min_context_support_pct", 0.01)
    )
    registry["reject_ret"] = registry["directional_mean_ret"] <= float(
        cfg.get("min_context_mean_ret", 0.0)
    )
    registry["reject_presence"] = registry["presence_freq"] < float(
        cfg.get("min_context_presence_freq", cfg.get("min_presence_freq", 0.4))
    )
    registry["reject_sign"] = registry["sign_consistency"] < float(
        cfg.get("min_context_sign_consistency", cfg.get("min_sign_consistency", 0.75))
    )
    registry["reject_arity"] = registry["display_arity"] < int(
        cfg.get("min_context_display_arity", 2)
    )
    registry["reject_dominated"] = registry.get(
        "dominated_by_parent", pd.Series(False, index=registry.index)
    )
    registry["reject_structural"] = ~registry.get(
        "is_structurally_sound", pd.Series(True, index=registry.index)
    ).fillna(False)

    mask = ~(
        registry["reject_support"]
        | registry["reject_ret"]
        | registry["reject_presence"]
        | registry["reject_sign"]
        | registry["reject_arity"]
        | registry["reject_dominated"]
        | registry["reject_structural"]
    )

    selected = registry[mask].copy()

    rejection_reasons = []
    for col in [
        "reject_support",
        "reject_ret",
        "reject_presence",
        "reject_sign",
        "reject_arity",
        "reject_dominated",
        "reject_structural",
    ]:
        rejection_reasons.append({"reason": col, "count": int(registry[col].sum())})
    rejection_summary = pd.DataFrame(rejection_reasons, columns=["reason", "count"])

    return selected, rejection_summary


def build_stage_a_rejection_map(
    stage_a_result: Dict[str, Any],
    winning_contexts: pd.DataFrame,
    cfg: Dict[str, Any],
) -> pd.DataFrame:
    """Summarize the Stage A rejection funnel across scorer, pruner, assessor,
    and final context selection gates."""
    rows: List[Dict[str, Any]] = []

    def _append_stage_rows(
        stage_name: str,
        stage_order: int,
        input_count: int,
        gate_items: List[Tuple[str, str, int, str]],
        passed_count: int,
    ) -> None:
        for gate_name, metric_name, rejected_count, threshold in gate_items:
            rows.append(
                {
                    "stage_order": stage_order,
                    "stage_name": stage_name,
                    "gate_name": gate_name,
                    "metric_name": metric_name,
                    "threshold": threshold,
                    "input_count": int(input_count),
                    "rejected_count": int(rejected_count),
                    "passed_count": int(passed_count),
                }
            )

    scored_registry = stage_a_result.get("scored_registry", pd.DataFrame())
    scorer_accepted = stage_a_result.get("scorer_accepted", pd.DataFrame())
    if not scored_registry.empty:
        scorer_rejections = collections.Counter(
            reason.strip()
            for reasons in scored_registry.loc[
                ~scored_registry["accepted"].fillna(False), "rejection_reason"
            ].dropna()
            for reason in str(reasons).split("|")
            if reason.strip()
        )
        _append_stage_rows(
            stage_name="scorer",
            stage_order=1,
            input_count=len(scored_registry),
            gate_items=[
                (
                    "low_support",
                    "min_support_actual",
                    scorer_rejections.get("low_support", 0),
                    f">= {int(cfg.get('min_support_count_validation', 10))}",
                ),
                (
                    "low_presence",
                    "presence_freq",
                    scorer_rejections.get("low_presence", 0),
                    f">= {float(cfg.get('min_presence_freq', 0.4)):.4f}",
                ),
                (
                    "low_sign_consistency",
                    "sign_consistency",
                    scorer_rejections.get("low_sign_consistency", 0),
                    f">= {float(cfg.get('min_sign_consistency', 0.75)):.4f}",
                ),
                (
                    "non_positive_directional_ret",
                    "directional_mean_ret",
                    scorer_rejections.get("non_positive_directional_ret", 0),
                    "> 0",
                ),
                (
                    "below_hurdle",
                    "hurdle_excess",
                    scorer_rejections.get("below_hurdle", 0),
                    "> 0",
                ),
            ],
            passed_count=len(scorer_accepted),
        )

    consolidated_registry = stage_a_result.get("consolidated_registry", pd.DataFrame())
    candidate_registry = stage_a_result.get("candidate_registry", pd.DataFrame())
    if not consolidated_registry.empty:
        _append_stage_rows(
            stage_name="pruner",
            stage_order=2,
            input_count=len(consolidated_registry),
            gate_items=[
                (
                    "is_too_broad",
                    "mean_support_pct",
                    int(
                        (
                            consolidated_registry["mean_support_pct"]
                            > float(cfg.get("max_support_pct", 0.25))
                        ).sum()
                    ),
                    f"<= {float(cfg.get('max_support_pct', 0.25)):.4f}",
                ),
                (
                    "beats_hurdle",
                    "hurdle_excess",
                    int(
                        (
                            consolidated_registry.get(
                                "hurdle_excess",
                                pd.Series(
                                    index=consolidated_registry.index, dtype=float
                                ),
                            )
                            <= 0
                        ).sum()
                    ),
                    "> 0",
                ),
                (
                    "low_sign_consistency",
                    "sign_consistency",
                    int(
                        (
                            consolidated_registry["sign_consistency"]
                            < float(cfg.get("min_sign_consistency", 0.80))
                        ).sum()
                    ),
                    f">= {float(cfg.get('min_sign_consistency', 0.80)):.4f}",
                ),
                (
                    "low_discovery_count",
                    "discovery_count",
                    int(
                        (
                            consolidated_registry["discovery_count"]
                            < int(cfg.get("min_tree_discoveries", 2))
                        ).sum()
                    ),
                    f">= {int(cfg.get('min_tree_discoveries', 2))}",
                ),
                (
                    "dominated_by_parent",
                    "dominated_by_parent",
                    int(
                        consolidated_registry.get(
                            "dominated_by_parent",
                            pd.Series(False, index=consolidated_registry.index),
                        )
                        .fillna(False)
                        .sum()
                    ),
                    "must be False",
                ),
            ],
            passed_count=len(candidate_registry),
        )

    assessment_df = stage_a_result.get("assessment_df", pd.DataFrame())
    accepted_registry = stage_a_result.get("accepted_registry", pd.DataFrame())
    if not candidate_registry.empty:
        assessed_keys = set()
        if not assessment_df.empty and "canonical_key" in assessment_df.columns:
            assessed_keys = set(assessment_df["canonical_key"].astype(str))
        candidate_keys = set(candidate_registry["canonical_key"].astype(str))
        unassessed_count = len(candidate_keys - assessed_keys)
        assessment_rejections = collections.Counter()
        if not assessment_df.empty:
            assessment_rejections.update(
                reason
                for reason in assessment_df.loc[
                    ~assessment_df["is_structurally_sound"].fillna(False),
                    "rejection_reason",
                ].dropna()
                if str(reason).strip()
            )
        _append_stage_rows(
            stage_name="mask_assessor",
            stage_order=3,
            input_count=len(candidate_registry),
            gate_items=[
                (
                    "not_assessed_min_mask_support",
                    "mask_support_count",
                    unassessed_count,
                    ">= 20 rows",
                ),
                (
                    "low_trades_per_day",
                    "avg_trades_per_day",
                    assessment_rejections.get("low_trades_per_day", 0),
                    f">= {float(cfg.get('min_avg_trades_per_day_10_symbols', 0.1)):.4f}",
                ),
                (
                    "insufficient_baseline_oof_coverage",
                    "baseline_oof_coverage",
                    assessment_rejections.get("insufficient_baseline_oof_coverage", 0),
                    f">= {float(cfg.get('learnability_min_oof_coverage', 0.25)):.4f}",
                ),
                (
                    "insufficient_subset_oof_coverage",
                    "subset_oof_coverage",
                    assessment_rejections.get("insufficient_subset_oof_coverage", 0),
                    f">= {float(cfg.get('learnability_min_oof_coverage', 0.25)):.4f}",
                ),
                (
                    "missing_learnability",
                    "learn_eff_ratio",
                    assessment_rejections.get("missing_learnability", 0),
                    "must be finite",
                ),
                (
                    "low_sign_consistency",
                    "sign_consistency",
                    assessment_rejections.get("low_sign_consistency", 0),
                    ">= 0.7500",
                ),
                (
                    "low_lift",
                    "learn_eff_ratio",
                    assessment_rejections.get("low_lift", 0),
                    ">= 1.1000",
                ),
                # NOTE: low_entropy_reduction gate removed per user request
            ],
            passed_count=len(accepted_registry),
        )

    if not accepted_registry.empty or not winning_contexts.empty:
        selected_registry, selection_summary = select_stage_a_contexts(
            stage_a_result, cfg
        )
        if winning_contexts is not None and not winning_contexts.empty:
            selected_registry = winning_contexts
        selection_counts = {
            str(row["reason"]): int(row["count"])
            for _, row in selection_summary.iterrows()
        }
        _append_stage_rows(
            stage_name="context_selector",
            stage_order=4,
            input_count=len(accepted_registry),
            gate_items=[
                (
                    "reject_support",
                    "mean_support_pct",
                    selection_counts.get("reject_support", 0),
                    f">= {float(cfg.get('min_context_support_pct', 0.01)):.4f}",
                ),
                (
                    "reject_ret",
                    "directional_mean_ret",
                    selection_counts.get("reject_ret", 0),
                    f"> {float(cfg.get('min_context_mean_ret', 0.0)):.4f}",
                ),
                (
                    "reject_presence",
                    "presence_freq",
                    selection_counts.get("reject_presence", 0),
                    f">= {float(cfg.get('min_context_presence_freq', cfg.get('min_presence_freq', 0.4))):.4f}",
                ),
                (
                    "reject_sign",
                    "sign_consistency",
                    selection_counts.get("reject_sign", 0),
                    f">= {float(cfg.get('min_context_sign_consistency', cfg.get('min_sign_consistency', 0.75))):.4f}",
                ),
                (
                    "reject_arity",
                    "display_arity",
                    selection_counts.get("reject_arity", 0),
                    f">= {int(cfg.get('min_context_display_arity', 2))}",
                ),
                (
                    "reject_dominated",
                    "dominated_by_parent",
                    selection_counts.get("reject_dominated", 0),
                    "must be False",
                ),
                (
                    "reject_structural",
                    "is_structurally_sound",
                    selection_counts.get("reject_structural", 0),
                    "must be True",
                ),
            ],
            passed_count=len(selected_registry),
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "stage_order",
                "stage_name",
                "gate_name",
                "metric_name",
                "threshold",
                "input_count",
                "rejected_count",
                "passed_count",
            ]
        )

    return (
        pd.DataFrame(rows)
        .sort_values(["stage_order", "gate_name"])
        .reset_index(drop=True)
    )


def create_pre_global_registry(side_results: Dict[str, Any]) -> pd.DataFrame:
    stage_a_selected = side_results.get("stage_a")
    if stage_a_selected is None or stage_a_selected.empty:
        stage_a_selected = pd.DataFrame()
    else:
        stage_a_selected = stage_a_selected.copy()
        stage_a_selected["origin_stage"] = "stage_a"

    stage_b_accepted = side_results.get("stage_b")
    if stage_b_accepted is None or stage_b_accepted.empty:
        stage_b_accepted = pd.DataFrame()
    else:
        stage_b_accepted = stage_b_accepted.copy()
        stage_b_accepted["origin_stage"] = "stage_b"

    if stage_a_selected.empty and stage_b_accepted.empty:
        return pd.DataFrame()
    if stage_a_selected.empty:
        return stage_b_accepted
    if stage_b_accepted.empty:
        return stage_a_selected
    return pd.concat([stage_a_selected, stage_b_accepted], ignore_index=True)


def log_stage_gate_diagnostics(
    stage_name: str, stage_result: Dict[str, Any], cfg: Dict[str, Any]
) -> None:
    scored = stage_result.get("scored_registry")
    scorer_accepted = stage_result.get("scorer_accepted")
    consolidated = stage_result.get("consolidated_registry")
    candidate = stage_result.get("candidate_registry")
    assessed = stage_result.get("assessment_df")
    accepted = stage_result.get("accepted_registry")
    tprint(
        f"{stage_name} gate counts: extracted={len(stage_result.get('all_extracted_rules', []))} "
        f"scored={0 if scored is None else len(scored)} "
        f"scorer_accepted={0 if scorer_accepted is None else len(scorer_accepted)} "
        f"consolidated={0 if consolidated is None else len(consolidated)} "
        f"candidate={0 if candidate is None else len(candidate)} "
        f"assessed={0 if assessed is None else len(assessed)} "
        f"accepted={0 if accepted is None else len(accepted)}"
    )
    if (
        scored is not None
        and not scored.empty
        and (scorer_accepted is None or scorer_accepted.empty)
    ):
        rejected = scored[~scored["accepted"]].copy()
        if "rejection_reason" in rejected.columns:
            reason_counts = (
                rejected["rejection_reason"]
                .fillna("")
                .astype(str)
                .str.split("|", regex=False)
                .explode()
                .str.strip()
            )
            reason_counts = reason_counts[reason_counts != ""].value_counts().head(8)
            if not reason_counts.empty:
                tprint(
                    f"{stage_name} scorer rejection reasons: "
                    + ", ".join(
                        f"{reason}={count}" for reason, count in reason_counts.items()
                    )
                )
        if not rejected.empty:
            top_rejected = rejected.sort_values(
                ["hurdle_excess", "composite_score"], ascending=[False, False]
            ).head(10)
            cols = [
                c
                for c in [
                    "canonical_key",
                    "side",
                    "mean_net_ret",
                    "directional_mean_ret",
                    "mean_support_pct",
                    "presence_freq",
                    "sign_consistency",
                    "required_hurdle",
                    "hurdle_excess",
                    "rejection_reason",
                ]
                if c in top_rejected.columns
            ]
            tprint(
                f"{stage_name} top near-miss rules:\n{top_rejected[cols].to_string(index=False)}"
            )
        tprint(
            f"{stage_name} score summary: "
            f"mean_ret_med={rejected['mean_net_ret'].median():.6f} "
            f"support_med={rejected['mean_support_pct'].median():.4f} "
            f"presence_med={rejected['presence_freq'].median():.3f} "
            f"sign_consistency_med={rejected['sign_consistency'].median():.3f}"
        )


def run_mining_stage(
    data: pd.DataFrame,
    fwd_ret: np.ndarray,
    fwd_ret_norm: np.ndarray,
    X: np.ndarray,
    metadata: List[FeatureMetadata],
    cfg: Dict[str, Any],
    output_dir: Path,
    stage_name: str,
    allowed_group_pairs: Sequence[Tuple[str, str]],
    slot_order: Sequence[str] = ("trigger", "location", "regime"),
    folds: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
    mask_resolver: Optional[
        Union[CanonicalRuleMaskResolver, DictionaryMaskResolver]
    ] = None,
    require_uplift: bool = False,
    rule_key_rewriter: Optional[
        Callable[[str], Tuple[Optional[str], Optional[str]]]
    ] = None,
    pipeline_stage_name: Optional[str] = None,
    explicit_side: Optional[str] = None,
    target_name: str = "fwd_ret_norm",
    horizon: int = 0,
    primary_target_override: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Run a single mining stage.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with event metadata
    fwd_ret : np.ndarray
        Forward returns (raw)
    fwd_ret_norm : np.ndarray
        Forward returns (normalized by ATR)
    X : np.ndarray
        Feature matrix
    metadata : List[FeatureMetadata]
        Feature metadata
    cfg : Dict[str, Any]
        Configuration dictionary
    output_dir : Path
        Output directory for this stage
    stage_name : str
        Name of the stage
    allowed_group_pairs : Sequence[Tuple[str, str]]
        Allowed interaction group pairs
    slot_order : Sequence[str]
        Order of slots in rules
    folds : Optional[List[Tuple[np.ndarray, np.ndarray]]]
        Pre-computed folds
    mask_resolver : Optional[Union[CanonicalRuleMaskResolver, DictionaryMaskResolver]]
        Mask resolver for rule evaluation
    require_uplift : bool
        Whether to require uplift over parent context
    rule_key_rewriter : Optional[Callable[[str], Tuple[Optional[str], Optional[str]]]]
        Function to rewrite rule keys
    pipeline_stage_name : Optional[str]
        Pipeline stage name for context
    explicit_side : Optional[str]
        Explicit side ('long' or 'short')
    target_name : str
        Name of the target for provenance tracking (default: 'fwd_ret_norm')
    horizon : int
        Horizon in bars for provenance tracking (default: 0)

    Returns
    -------
    Dict[str, Any]
        Stage results including extracted rules and registries
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    tprint(f"--- RUNNING MINING STAGE: {stage_name} ---")
    
    # Log target provenance
    tprint(f"Target: {target_name} | Horizon: {horizon}")

    # Use primary_target_override (bounded triad target) if provided; fall back to fwd_ret_norm.
    primary_target = primary_target_override if primary_target_override is not None else fwd_ret_norm

    # Relaxed completeness check per user request.
    # We only require the target to be finite and at least ONE feature to be present.
    _feature_any_finite = np.any(np.isfinite(X), axis=1)
    _complete = np.isfinite(primary_target) & _feature_any_finite

    n_before = len(data)
    n_remain = int(_complete.sum())
    
    if n_remain < n_before:
        tprint(
            f"{pipeline_stage_name}: upstream filter invalidating {n_before - n_remain} rows "
            f"({100.0 * (n_before - n_remain) / max(n_before, 1):.1f}%) with missing features/targets. "
            f"{n_remain} rows will be used for training/metrics."
        )
        # We set target to NaN for incomplete rows so train_fold and Scorer skip them.
        # Use copies to avoid side effects on other stages.
        primary_target = primary_target.copy()
        primary_target[~_complete] = np.nan
        fwd_ret = fwd_ret.copy()
        fwd_ret[~_complete] = np.nan
        fwd_ret_norm = fwd_ret_norm.copy()
        fwd_ret_norm[~_complete] = np.nan

    if n_remain == 0:
        tprint(f"WARNING: {pipeline_stage_name} has 0 valid rows after filter. Skipping stage.")
        return {
            "accepted_registry": pd.DataFrame(),
            "candidate_registry": pd.DataFrame(),
            "stage_a": pd.DataFrame(),
            "stage_b": pd.DataFrame(),
            "all_extracted_rules": [],
            "all_rejection_audit": [],
            "all_split_usage": [],
            "fold_quality_reports": [],
            "model_fit_reports": [],
            "feature_importance_records": [],
        }

    if folds is None:
        folds = build_walk_forward_folds(
            n_samples=len(data),
            n_folds=int(cfg.get("n_folds", 5)),
            min_train_frac=float(cfg.get("cv_min_train_frac", 0.5)),
            embargo=int(cfg.get("cv_embargo", 0)),
        )
    for fold_id, (tr_idx, va_idx) in enumerate(folds):
        if tr_idx.size == 0 or va_idx.size == 0 or tr_idx.max() >= va_idx.min():
            raise ValueError(f"Invalid fold {fold_id} in {stage_name}")

    model_engine = InteractionModel(
        metadata, cfg, side=explicit_side, allowed_group_pairs=allowed_group_pairs
    )
    constraint_summary = model_engine.get_constraint_summary()
    with open(output_dir / "interaction_constraint_summary.json", "w") as f:
        json.dump(constraint_summary, f, indent=2)

    stage_input_feature_inventory = pd.DataFrame(
        [
            {
                "feature_name": m.feature_name,
                "group": m.group,
                "regime_family": m.regime_family,
                "interaction_group": m.interaction_group,
            }
            for m in metadata
        ]
    )
    stage_input_feature_inventory.to_csv(
        output_dir / "stage_input_feature_inventory.csv", index=False
    )

    tprint(f"Constraints Mode: {constraint_summary.get('mode', 'unknown')}")
    tprint(
        f"Group Counts: "
        + ", ".join(
            [
                f"{k}={v}"
                for k, v in constraint_summary.items()
                if k.startswith("num_") and not k.startswith("num_regime_")
            ]
        )
    )
    tprint(
        f"Regime Family Counts: "
        + ", ".join(
            [
                f"{k.replace('num_regime_', '')}={v}"
                for k, v in constraint_summary.items()
                if k.startswith("num_regime_")
            ]
        )
    )

    positive_only_groups: Tuple[str, ...] = ()
    required_positive_groups: Tuple[str, ...] = ()
    collapse_duplicate_groups: Tuple[str, ...] = ()
    if pipeline_stage_name == "stage_a_context":
        collapse_duplicate_groups = ("location",)
        if not bool(cfg.get("stage_a_relax_positive_groups", True)):
            positive_only_groups = ("location", "regime")
            required_positive_groups = ("location", "regime")
    extractor = RuleExtractor(
        metadata,
        cfg,
        slot_order=slot_order,
        positive_only_groups=positive_only_groups,
        required_positive_groups=required_positive_groups,
        collapse_duplicate_groups=collapse_duplicate_groups,
    )
    all_extracted_rules = []
    all_rejection_audit = []
    all_split_usage = []
    seeds = cfg.get("seeds", [42])

    fold_quality_reports = []
    model_fit_reports = []
    feature_importance_records = []

    for fold_id, (tr_idx, va_idx) in enumerate(folds):
        # All rows are guaranteed feature-complete (filtered upstream).
        if len(tr_idx) == 0 or len(va_idx) == 0:
            tprint(f"Skipping fold {fold_id}: empty split (train={len(tr_idx)}, val={len(va_idx)}).")
            continue

        # Determine available features per group for logging
        group_to_features = collections.defaultdict(list)
        for m in metadata:
            group_to_features[m.group].append(m.feature_name)

        tprint(
            f"Fold {fold_id}: train_rows={len(tr_idx)} val_rows={len(va_idx)}"
        )

        # Target distribution summary (using primary_target, which is the real training target)
        _tr_tgt = primary_target[tr_idx]
        tr_target_valid = _tr_tgt[~np.isnan(_tr_tgt)]
        if len(tr_target_valid) > 0:
            tr_mean = tr_target_valid.mean()
            tr_std = tr_target_valid.std()
            tr_p1 = np.percentile(tr_target_valid, 1)
            tr_p50 = np.percentile(tr_target_valid, 50)
            tr_p99 = np.percentile(tr_target_valid, 99)

            # Check severe clipping
            clipped = np.clip(tr_target_valid, tr_p1, tr_p99)
            clip_diff = (
                np.abs(tr_target_valid - clipped).sum() / np.abs(tr_target_valid).sum()
            )
            if clip_diff > 0.05:
                tprint(
                    f"WARNING: Severe target clipping in Fold {fold_id} ({clip_diff:.1%} diff)"
                )

            pos_ratio = (tr_target_valid > 0).mean()
            if pos_ratio < 0.2 or pos_ratio > 0.8:
                tprint(
                    f"WARNING: Fold {fold_id} has extremely imbalanced targets ({pos_ratio:.1%} positive)"
                )
        else:
            tr_mean, tr_std, tr_p1, tr_p50, tr_p99 = (
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            )

        fold_quality_reports.append(
            {
                "fold_id": fold_id,
                "tr_rows": len(tr_idx),
                "va_rows": len(va_idx),
                "target_mean": tr_mean,
                "target_std": tr_std,
                "target_p1": tr_p1,
                "target_p50": tr_p50,
                "target_p99": tr_p99,
            }
        )

        tprint(
            f"Fold {fold_id}: Target fwd_ret_norm -> mean={tr_mean:.4f}, std={tr_std:.4f}, p1={tr_p1:.4f}, p50={tr_p50:.4f}, p99={tr_p99:.4f}"
        )

        # All rows in tr_idx and va_idx are already complete (pre-filtered upstream).
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr_raw = primary_target[tr_idx]
        y_va_raw = primary_target[va_idx]

        y_tr_clip = np.clip(
            y_tr_raw,
            np.nanquantile(y_tr_raw, 0.01),
            np.nanquantile(y_tr_raw, 0.99),
        )

        for seed in seeds:
            # Use quantile regression for all targets (triad targets work with quantile regression)
            target_type = "quantile"
            model, fit_meta = model_engine.train_fold(
                X_tr, y_tr_clip, X_va, y_va_raw, fold_id, seed, target_type=target_type
            )
            tprint(
                f"{stage_name} fold {fold_id} seed {seed}: "
                f"train_samples={fit_meta['train_samples']} val_samples={fit_meta['val_samples']} "
                f"best_iteration={fit_meta['best_iteration']} best_val_metric={fit_meta['best_val_metric']:.5f} "
                f"target_type={target_type}"
            )

            model_fit_reports.append(
                {
                    "fold_id": fold_id,
                    "seed": seed,
                    "best_iteration": fit_meta["best_iteration"],
                    "best_val_metric": fit_meta["best_val_metric"],
                    "max_depth": fit_meta["params"]["max_depth"],
                    "num_leaves": fit_meta["params"]["num_leaves"],
                    "min_data_in_leaf": fit_meta["params"]["min_data_in_leaf"],
                    "objective": fit_meta["params"]["objective"],
                    "metric": fit_meta["params"]["metric"],
                    "target_type": target_type,
                    "target_name": target_name,
                    "horizon": horizon,
                }
            )

            # Print hyperparams only on first fold/seed
            if fold_id == 0 and seed == seeds[0]:
                tprint(
                    f"Model Hyperparams: max_depth={fit_meta['params']['max_depth']}, num_leaves={fit_meta['params']['num_leaves']}, min_data_in_leaf={fit_meta['params']['min_data_in_leaf']}, objective={fit_meta['params']['objective']}, metric={fit_meta['params']['metric']}, n_estimators={fit_meta['params']['n_estimators']}, seeds={len(seeds)}, folds={len(folds)}"
                )

            # Extract Feature Importance
            gain_imp = fit_meta["feature_importances_gain"]
            split_imp = fit_meta["feature_importances_split"]

            fi_records = []
            for m in metadata:
                idx = m.feature_index
                gain = gain_imp[idx] if idx < len(gain_imp) else 0.0
                split = split_imp[idx] if idx < len(split_imp) else 0.0
                if gain > 0 or split > 0:
                    fi_records.append(
                        {
                            "fold_id": fold_id,
                            "seed": seed,
                            "feature_name": m.feature_name,
                            "group": m.group,
                            "regime_family": m.regime_family,
                            "gain": gain,
                            "split": split,
                        }
                    )
                    feature_importance_records.append(fi_records[-1])

            if fi_records:
                fi_df = pd.DataFrame(fi_records)
                top_gain = fi_df.sort_values("gain", ascending=False).head(5)
                tprint("Top 5 features by gain:")
                for _, row in top_gain.iterrows():
                    tprint(f"  - {row['feature_name']}: {row['gain']:.2f}")

                top_fam = (
                    fi_df.groupby("regime_family")["split"]
                    .sum()
                    .sort_values(ascending=False)
                    .head(5)
                )
                tprint("Top 5 regime families by split count:")
                for fam, count in top_fam.items():
                    if pd.notna(fam):
                        tprint(f"  - {fam}: {count}")
            split_usage_df = collect_split_usage_from_model(
                model, metadata, fold_id, seed
            )
            if not split_usage_df.empty:
                all_split_usage.append(split_usage_df)
                group_summary = summarize_fold_feature_usage(split_usage_df)
                if not group_summary.empty:
                    summary_text = ", ".join(
                        f"{row.group}={int(row.used_feature_count)}f/{int(row.split_count)}s"
                        for row in group_summary.itertuples(index=False)
                    )
                    tprint(
                        f"{stage_name} fold {fold_id} seed {seed} feature usage by group: {summary_text}"
                    )
            fold_rules = extractor.extract_rules(
                model, f"{stage_name}_model", fold_id, seed,
                target_name=target_name,
                horizon=horizon,
            )
            all_extracted_rules.extend(fold_rules)
            if extractor.rejection_audit:
                all_rejection_audit.extend(extractor.rejection_audit)

    parent_context_map: Dict[str, str] = {}

    if rule_key_rewriter is not None:
        rewritten_rules: List[ExtractedRule] = []
        for rule in all_extracted_rules:
            rewritten_key, parent_context_key = rule_key_rewriter(rule.canonical_key)
            if rewritten_key is None:
                continue
            rule.canonical_key = rewritten_key
            rewritten_rules.append(rule)
            if parent_context_key:
                parent_context_map[rewritten_key] = parent_context_key
        all_extracted_rules = rewritten_rules

    require_groups = cfg.get("require_groups", [])
    if require_groups:
        filtered_rules = []
        for rule in all_extracted_rules:
            # Check if all required groups are present in the rule conditions
            rule_groups = set(c.group for c in rule.conditions)
            # Some conditions might be context and we don't have conditions for them inside rule.conditions directly?
            # Actually, `rule.conditions` should have `group="context"` if we used context features.
            if all(rg in rule_groups for rg in require_groups):
                filtered_rules.append(rule)
        all_extracted_rules = filtered_rules

    if fold_quality_reports:
        fq_df = pd.DataFrame(fold_quality_reports)
        # Simplified columns after removing per-fold filtering
        data_cols = ["fold_id", "tr_rows", "va_rows"]
        tgt_cols = [
            "fold_id",
            "target_mean",
            "target_std",
            "target_p1",
            "target_p50",
            "target_p99",
        ]

        fq_df[data_cols].to_csv(
            output_dir / "fold_data_quality_report.csv", index=False
        )
        # Check if tgt_cols exist (they might be missing if all targets were NaN in a fold)
        actual_tgt_cols = [c for c in tgt_cols if c in fq_df.columns]
        fq_df[actual_tgt_cols].to_csv(
            output_dir / "fold_target_distribution_report.csv", index=False
        )

    if model_fit_reports:
        pd.DataFrame(model_fit_reports).to_csv(
            output_dir / "fold_model_fit_summary.csv", index=False
        )

    if feature_importance_records:
        fi_df = pd.DataFrame(feature_importance_records)
        fi_df.to_csv(output_dir / "fold_feature_importance_by_feature.csv", index=False)

        fi_by_group = (
            fi_df.groupby(["fold_id", "seed", "group"])[["gain", "split"]]
            .sum()
            .reset_index()
        )
        fi_by_group.to_csv(
            output_dir / "fold_feature_importance_by_group.csv", index=False
        )

        fi_by_family = (
            fi_df.groupby(["fold_id", "seed", "regime_family"])[["gain", "split"]]
            .sum()
            .reset_index()
        )
        fi_by_family.to_csv(
            output_dir / "fold_feature_importance_by_regime_family.csv", index=False
        )

    if all_rejection_audit:
        audit_df = pd.DataFrame(all_rejection_audit)
        audit_df.to_csv(output_dir / "invalid_path_audit.csv", index=False)
        summarize_feature_usage(
            audit_df, output_dir / "invalid_path_reason_summary.csv", ["reason"]
        )

    if all_extracted_rules:
        shape_records = []
        family_combos = collections.defaultdict(int)
        for r in all_extracted_rules:
            # arity, structural depth, groups used, regime families used
            arity = display_arity_for_key(r.canonical_key)
            depth = structural_depth_for_key(r.canonical_key)
            groups_used = tuple(sorted(set(c.group for c in r.conditions)))

            regime_families = []
            for c in r.conditions:
                if c.group == "regime":
                    fam = (
                        m.regime_family
                        if (m := metadata[c.feature_index])
                        else "unknown"
                    )
                    if fam:
                        regime_families.append(fam)

            regime_families_tuple = tuple(sorted(set(regime_families)))
            family_combos[regime_families_tuple] += 1

            shape_records.append(
                {
                    "canonical_key": r.canonical_key,
                    "display_arity": arity,
                    "structural_depth": depth,
                    "groups_used": "|".join(groups_used),
                    "regime_families_used": "|".join(regime_families_tuple),
                }
            )

        pd.DataFrame(shape_records).to_csv(
            output_dir / "extracted_rule_shape_summary.csv", index=False
        )

        family_df = pd.DataFrame(
            [
                {"regime_families_combo": "|".join(k), "count": v}
                for k, v in family_combos.items()
            ]
        )
        family_df.sort_values("count", ascending=False).to_csv(
            output_dir / "extracted_rule_family_combo_summary.csv", index=False
        )

        tprint("Top valid family combinations:")
        for _, row in (
            family_df.sort_values("count", ascending=False).head(5).iterrows()
        ):
            tprint(f"  - {row['regime_families_combo']}: {row['count']}")

    if all_split_usage:
        split_usage_all = pd.concat(all_split_usage, ignore_index=True)
        split_usage_all.to_csv(
            output_dir / "model_split_usage_detailed.csv", index=False
        )
        summarize_feature_usage(
            split_usage_all,
            output_dir / "model_split_usage_by_feature.csv",
            ["feature_name", "group"],
        )
        summarize_feature_usage(
            split_usage_all, output_dir / "model_split_usage_by_group.csv", ["group"]
        )
        summarize_fold_feature_usage(split_usage_all).to_csv(
            output_dir / "model_split_usage_by_fold_group.csv", index=False
        )
    else:
        split_usage_all = pd.DataFrame()

    rule_usage_df = collect_extracted_rule_feature_usage(all_extracted_rules, metadata)
    rule_usage_df.to_csv(
        output_dir / "extracted_rule_feature_usage_detailed.csv", index=False
    )
    summarize_feature_usage(
        rule_usage_df,
        output_dir / "extracted_rule_feature_usage_by_feature.csv",
        ["feature_name", "group"],
    )
    summarize_feature_usage(
        rule_usage_df,
        output_dir / "extracted_rule_feature_usage_by_group.csv",
        ["group"],
    )

    scorer = RuleScorer(metadata, cfg, mask_resolver=mask_resolver)
    unique_keys = sorted({rule.canonical_key for rule in all_extracted_rules})
    discovery_count_map = collections.Counter(
        rule.canonical_key for rule in all_extracted_rules
    )
    n_instances_map = discovery_count_map.copy()
    pipeline_stage_map = {
        key: (pipeline_stage_name or stage_name) for key in unique_keys
    }
    require_uplift_keys = unique_keys if require_uplift else []
    side_map = {key: explicit_side for key in unique_keys} if explicit_side else None
    scored_registry, full_scorer_audit = scorer.score_registry_oos(
        keys=unique_keys,
        fwd_ret=fwd_ret,
        folds=folds,
        resolver=mask_resolver,
        parent_context_map=parent_context_map,
        require_uplift_keys=require_uplift_keys,
        discovery_count_map=discovery_count_map,
        n_instances_map=n_instances_map,
        pipeline_stage_map=pipeline_stage_map,
        side_map=side_map,
    )
    scored_registry["preset"] = cfg.get("preset", "exploration")
    atomic_to_csv(
        full_scorer_audit, output_dir / "fold_level_rule_aggregation_audit.csv"
    )
    scored_registry = atomic_to_csv(
        scored_registry,
        output_dir / "scored_rule_registry_full.csv",
        expected_columns=SCORER_REGISTRY_COLUMNS + ["preset"],
    )

    # Handle empty registry
    if scored_registry.empty:
        tprint(
            "WARNING: No rules scored. Skipping consolidation and returning empty results."
        )
        return {
            "scored_registry": scored_registry,
            "scorer_accepted": pd.DataFrame(),
            "accepted_registry": pd.DataFrame(),
            "consolidated_registry": pd.DataFrame(),
            "final_registry": pd.DataFrame(),
            "candidate_registry": pd.DataFrame(),
            "assessment_df": pd.DataFrame(),
        }

    # Save scorer diagnostics
    rejection_reasons = collections.Counter(
        reason.strip()
        for reasons in scored_registry[~scored_registry["accepted"]][
            "rejection_reason"
        ].dropna()
        for reason in reasons.split("|")
        if reason.strip()
    )
    atomic_to_csv(
        pd.DataFrame(
            list(rejection_reasons.items()), columns=["rejection_reason", "count"]
        ),
        output_dir / "scorer_rejection_reason_summary.csv",
        expected_columns=["rejection_reason", "count"],
    )

    dom_df = scored_registry[scored_registry.get("dominated_by_parent", False)][
        ["canonical_key", "dominant_parent_key", "composite_score", "hurdle_excess"]
    ]
    if not dom_df.empty:
        atomic_to_csv(dom_df, output_dir / "dominated_rule_summary.csv")

    scorer_accepted = scored_registry[scored_registry["accepted"]].copy()

    if cfg.get("skip_stage_a_consolidation", True) and pipeline_stage_name == "stage_a_context":
        tprint("SKIPPING Stage A consolidation per configuration.")
        consolidated_registry = scorer_accepted.copy()
        lineage_audit = pd.DataFrame()
        consol_diag = {}
    else:
        use_economic_consolidator = cfg.get("use_economic_consolidator", True)
        if use_economic_consolidator:
            consolidator = EconomicRuleConsolidator(
                metadata, cfg, mask_resolver=mask_resolver, scorer=scorer
            )
        else:
            consolidator = RuleConsolidator(
                metadata, cfg, mask_resolver=mask_resolver, scorer=scorer
            )

        consolidated_registry, lineage_audit, consol_diag = consolidator.consolidate(
            scorer_accepted, fwd_ret, folds, resolver=mask_resolver, data=data
        )

    atomic_to_csv(consolidated_registry, output_dir / "consolidated_rule_registry.csv")
    atomic_to_csv(lineage_audit, output_dir / "consolidation_lineage_audit.csv")

    # Save Consolidator Specific Diagnostics
    for name, obj in consol_diag.items():
        if isinstance(obj, pd.DataFrame):
            obj.to_csv(output_dir / f"{name}.csv", index=False)
        else:
            with open(output_dir / f"{name}.json", "w") as f:
                json.dump(obj, f, indent=2)

    pruner = IndependentRulePruner(cfg)
    candidate_registry = pruner.prune(consolidated_registry)
    candidate_registry["preset"] = cfg.get("preset", "exploration")
    candidate_registry = atomic_to_csv(
        candidate_registry,
        output_dir / "candidate_rule_registry.csv",
        expected_columns=list(consolidated_registry.columns) + ["preset"],
    )
    atomic_to_csv(
        candidate_registry,
        output_dir / "pruned_rule_registry.csv",
        expected_columns=list(candidate_registry.columns),
    )

    if hasattr(pruner, "gate_summary"):
        atomic_to_csv(
            pd.DataFrame([pruner.gate_summary]),
            output_dir / "pruner_gate_summary.csv",
        )

    if not candidate_registry.empty:
        arity_counts = candidate_registry["display_arity"].value_counts().reset_index()
        arity_counts.columns = ["display_arity", "count"]
        atomic_to_csv(arity_counts, output_dir / "pruner_arity_summary.csv")
        tprint("Accepted by Arity (Pruner):")
        for _, row in arity_counts.iterrows():
            tprint(f"  - {int(row['display_arity'])}: {int(row['count'])}")

    assessor = MaskAssessor(metadata, cfg, mask_resolver=mask_resolver)
    assessment_df = assessor.assess_rules(candidate_registry, X, data, fwd_ret, folds)
    if not assessment_df.empty:
        atomic_to_csv(assessment_df, output_dir / "final_mask_assessment_audit.csv")
        accepted_registry = candidate_registry.merge(
            assessment_df, on="canonical_key", how="left"
        )

        if hasattr(assessor, "rejection_summary") and assessor.rejection_summary:
            atomic_to_csv(
                pd.DataFrame(
                    list(assessor.rejection_summary.items()),
                    columns=["reason", "count"],
                ),
                output_dir / "mask_assessment_rejection_summary.csv",
                expected_columns=["reason", "count"],
            )
    else:
        accepted_registry = candidate_registry.copy()

    if "is_structurally_sound" in accepted_registry.columns:
        accepted_registry = accepted_registry[
            accepted_registry["is_structurally_sound"].fillna(False)
        ].copy()
    accepted_registry = accepted_registry[
        ~accepted_registry.get(
            "dominated_by_parent", pd.Series(False, index=accepted_registry.index)
        ).fillna(False)
    ].copy()
    accepted_registry["preset"] = cfg.get("preset", "exploration")
    accepted_registry = atomic_to_csv(
        accepted_registry,
        output_dir / "accepted_rule_registry.csv",
        expected_columns=list(candidate_registry.columns),
    )
    atomic_to_csv(
        accepted_registry,
        output_dir / "final_rule_registry.csv",
        expected_columns=list(accepted_registry.columns),
    )

    final_usage_df = collect_registry_feature_usage(accepted_registry, metadata)
    export_coverage_sanity_report(
        metadata, split_usage_all, rule_usage_df, final_usage_df, output_dir
    )

    return {
        "X": X,
        "metadata": metadata,
        "folds": folds,
        "all_extracted_rules": all_extracted_rules,
        "scored_registry": scored_registry,
        "parent_context_map": parent_context_map,
        "scorer_accepted": scorer_accepted,
        "consolidated_registry": consolidated_registry,
        "candidate_registry": candidate_registry,
        "assessment_df": assessment_df,
        "accepted_registry": accepted_registry,
        "output_dir": output_dir,
    }


def run_side_pipeline(
    side: str,
    data: pd.DataFrame,
    feature_dict: Dict[str, np.ndarray],
    fwd_ret: np.ndarray,
    fwd_ret_norm: np.ndarray,
    cfg: Dict[str, Any],
    folds: List[Tuple[np.ndarray, np.ndarray]],
    root_output_dir: Path,
    target_name: str = "fwd_ret_norm",
    horizon: int = 0,
    bounded_target: Optional[np.ndarray] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Run the full side pipeline (long or short).

    Parameters
    ----------
    side : str
        'long' or 'short'
    data : pd.DataFrame
        DataFrame with event metadata
    feature_dict : Dict[str, np.ndarray]
        Dictionary of feature arrays
    fwd_ret : np.ndarray
        Forward returns (raw)
    fwd_ret_norm : np.ndarray
        Forward returns (normalized by ATR)
    cfg : Dict[str, Any]
        Configuration dictionary
    folds : List[Tuple[np.ndarray, np.ndarray]]
        Pre-computed folds
    root_output_dir : Path
        Root output directory
    target_name : str
        Name of the target for provenance tracking (default: 'fwd_ret_norm')
    horizon : int
        Horizon in bars for provenance tracking (default: 0)

    Returns
    -------
    Dict[str, pd.DataFrame]
        Pipeline results including stage registries
    """
    tprint(f"--- RUNNING PIPELINE FOR SIDE: {side.upper()} ---")
    
    # Log target provenance
    tprint(f"Target: {target_name} | Horizon: {horizon}")

    side_fwd_ret = fwd_ret if side == "long" else -fwd_ret
    side_fwd_ret_norm = fwd_ret_norm if side == "long" else -fwd_ret_norm

    side_output_dir = root_output_dir / side
    side_output_dir.mkdir(parents=True, exist_ok=True)

    # --- STAGE A: CONTEXT MINING ---
    tprint(f"STAGE A: Context Mining (Regime x Location) [{side}]")
    fp_a = FeatureProcessor()
    X_a, metadata_a, audits_a = fp_a.prepare_features(
        feature_dict,
        data["timestamp"].to_numpy(),
        data["symbol"].to_numpy(),
        cfg,
        active_groups=("regime", "location"),
    )
    stage_a_output_dir = side_output_dir / "stage_a_context"
    stage_a_output_dir.mkdir(parents=True, exist_ok=True)

    for k, v in audits_a.items():
        if not v.empty:
            v.to_csv(stage_a_output_dir / f"{k}.csv", index=False)

    stage_a_spec = MiningStageSpec(
        stage_name="stage_a_context",
        active_groups=("regime", "location"),
        allow_groups_in_rule=("regime", "location"),
        output_dir_name="stage_a_context",
        allowed_group_pairs=(("regime", "location"),),
        slot_order=("trigger", "location", "regime"),
    )

    if cfg.get("use_dynamic_hpo", False):
        tprint(f"--- DYNAMIC HPO: Tuning Stage A for {side.upper()} ---")
        try:
            # We use X_a and side_fwd_ret for HPO
            # The HPO script expects MAIN_MINER_PARAMS but we can build them from cfg
            hpo_main_params = {
                "objective": "quantile",
                "metric": "quantile",
                "learning_rate": float(cfg.get("learning_rate", 0.03)),
                "num_leaves": int(cfg.get("lgbm_num_leaves", 64)),
                "max_depth": int(cfg.get("lgbm_max_depth", 5)) + 1,
                "lambda_l1": float(cfg.get("lambda_l1", 0.0)),
                "lambda_l2": float(cfg.get("lambda_l2", 0.0)),
                "verbosity": -1,
                "bagging_fraction": 0.8,
                "bagging_freq": 1,
                "feature_fraction": 0.8,
            }
            
            hpo_results = run_short_hpo_for_target_horizon(
                X=X_a,
                y=side_fwd_ret,
                main_params=hpo_main_params,
                seed=cfg.get("random_state", 42)
            )
            
            best_cfg = hpo_results.get("best_final_result")
            if best_cfg and best_cfg.valid:
                tprint(f"Dynamic HPO Results: alpha={best_cfg.cfg.alpha:.3f}, "
                       f"min_gain={best_cfg.cfg.min_gain_to_split:.5f}, "
                       f"min_leaf_frac={best_cfg.cfg.min_leaf_frac:.5f}")
                
                # Update cfg for this side pipeline
                cfg = cfg.copy() # Avoid polluting other sides
                cfg["hpo_best_alpha"] = best_cfg.cfg.alpha
                cfg["hpo_min_gain_to_split"] = best_cfg.cfg.min_gain_to_split
                
                # Use train_n if available, otherwise fallback to full X_a size
                train_n = hpo_results.get("train_n", len(X_a))
                cfg["hpo_min_data_in_leaf"] = max(25, int(round(best_cfg.cfg.min_leaf_frac * train_n)))
            else:
                tprint(f"Dynamic HPO failed or invalid: {best_cfg.reason if best_cfg else 'Unknown'}. Using defaults.")

        except Exception as e:
            tprint(f"Dynamic HPO encountered error: {e}. Using defaults.")
            traceback.print_exc()

    # Use bounded_target as the real training target if provided.
    # For short side, flip sign (higher target_eff = more favourable for that side).
    if bounded_target is not None:
        side_target = -bounded_target if side == "short" else bounded_target
    else:
        side_target = side_fwd_ret_norm  # legacy fallback

    stage_a_result = run_mining_stage(
        data,
        side_fwd_ret,
        side_fwd_ret_norm,
        X_a,
        metadata_a,
        cfg,
        stage_a_output_dir,
        stage_a_spec.stage_name,
        stage_a_spec.allowed_group_pairs,
        slot_order=stage_a_spec.slot_order,
        folds=folds,
        mask_resolver=CanonicalRuleMaskResolver(X_a, metadata_a),
        pipeline_stage_name="stage_a_context",
        explicit_side=side,
        target_name=target_name,
        horizon=horizon,
        primary_target_override=side_target,
    )
    log_stage_gate_diagnostics("Stage A", stage_a_result, cfg)

    winning_contexts, stage_a_rejection_summary = select_stage_a_contexts(
        stage_a_result, cfg
    )
    atomic_to_csv(
        stage_a_rejection_summary,
        stage_a_output_dir / "stage_a_context_selection_summary.csv",
        expected_columns=["reason", "count"],
    )
    stage_a_rejection_map = build_stage_a_rejection_map(
        stage_a_result, winning_contexts, cfg
    )
    atomic_to_csv(
        stage_a_rejection_map,
        stage_a_output_dir / "stage_a_rejection_map.csv",
        expected_columns=[
            "stage_order",
            "stage_name",
            "gate_name",
            "metric_name",
            "threshold",
            "input_count",
            "rejected_count",
            "passed_count",
        ],
    )

    stage_a_accepted_count = len(stage_a_result.get("accepted_registry", []))
    tprint(
        f"Stage A accepted -> winning contexts funnel: {stage_a_accepted_count} -> {len(winning_contexts)}"
    )

    if len(winning_contexts) < 5 and stage_a_accepted_count > 10:
        tprint(
            f"WARNING: Very few contexts survived selection ({len(winning_contexts)} out of {stage_a_accepted_count})."
        )

    if not winning_contexts.empty:
        tprint("Top selected contexts by hurdle excess:")
        top_ctx = winning_contexts.sort_values("hurdle_excess", ascending=False).head(5)
        for _, row in top_ctx.iterrows():
            tprint(
                f"  - {row['canonical_key']}: hurdle_excess={row['hurdle_excess']:.5f}"
            )

    if winning_contexts.empty:
        tprint("No contexts found in Stage A. Aborting Stage B.")
        stage_b_output_dir = side_output_dir / "stage_b_trigger_refinement"
        stage_b_output_dir.mkdir(parents=True, exist_ok=True)
        atomic_to_csv(
            pd.DataFrame(columns=["context_feature", "canonical_key"]),
            stage_b_output_dir / "stage_b_context_mapping.csv",
            expected_columns=["context_feature", "canonical_key"],
        )
        atomic_to_csv(
            pd.DataFrame(columns=SCORER_REGISTRY_COLUMNS + ["preset"]),
            stage_b_output_dir / "candidate_rule_registry.csv",
            expected_columns=SCORER_REGISTRY_COLUMNS + ["preset"],
        )
        atomic_to_csv(
            pd.DataFrame(columns=SCORER_REGISTRY_COLUMNS + ["preset"]),
            stage_b_output_dir / "accepted_rule_registry.csv",
            expected_columns=SCORER_REGISTRY_COLUMNS + ["preset"],
        )
        atomic_to_csv(
            pd.DataFrame(
                columns=[
                    "canonical_key",
                    "regime_score",
                    "is_structurally_sound",
                    "rejection_reason",
                    "support_pct",
                    "support_ok",
                    "support_score",
                    "avg_trades_per_day",
                    "density_dispersion",
                    "tail_ratio",
                    "mae",
                    "mfe",
                    "mean_ret_global",
                    "mean_ret_mask",
                    "ret_uplift",
                    "lift",
                    "learn_eff_ratio",
                    "subset_oof_coverage",
                    "baseline_oof_coverage",
                    "entropy_reduction",
                    "tp_rate",
                    "sl_rate",
                    "timeout_rate",
                    "ev_per_trade",
                    "ev_per_event",
                    "win_rate_conditional",
                    "win_rate_unconditional",
                    "production_classification",
                    "classification_diagnostics",
                    "rule_type_class",
                ]
            ),
            stage_b_output_dir / "final_mask_assessment_audit.csv",
        )
        return {
            "stage_a": winning_contexts,
            "stage_a_result": stage_a_result,
            "stage_b": pd.DataFrame(),
            "stage_b_trig_result": None,
            "stage_b_loc_result": None,
            "metadata_a": metadata_a,
            "metadata_b_trig": [],
            "metadata_b_loc": [],
            "X_a": X_a,
            "X_b_trig": None,
            "X_b_loc": None,
            "context_feature_dict": {},
            "context_to_key": {},
        }

    if cfg.get("skip_stage_b", True):
        tprint(f"SKIPPING Stage B for side {side.upper()} per configuration.")
        return {
            "stage_a": winning_contexts,
            "stage_a_result": stage_a_result,
            "stage_b": pd.DataFrame(),
            "stage_b_trig_result": None,
            "stage_b_loc_result": None,
            "metadata_a": metadata_a,
            "metadata_b_trig": [],
            "metadata_b_loc": [],
            "X_a": X_a,
            "X_b_trig": None,
            "X_b_loc": None,
            "context_feature_dict": {},
            "context_to_key": {},
        }

    context_feature_dict, context_to_key = build_context_feature_dict_from_registry(
        winning_contexts, data, X_a, metadata_a
    )
    context_mapping_df = pd.DataFrame(
        list(context_to_key.items()), columns=["context_feature", "canonical_key"]
    )

    # --- STAGE B1: TRIGGER REFINEMENT --- DISABLED (trigger features removed)
    tprint(f"STAGE B1: Trigger Refinement SKIPPED (trigger features disabled) [{side}]")
    stage_b_trig_result = {"accepted_registry": pd.DataFrame()}
    X_b_trig = None
    metadata_b_trig = []
    stage_b_trig_output_dir = side_output_dir / "stage_b_trigger_refinement"
    stage_b_trig_output_dir.mkdir(parents=True, exist_ok=True)

    # --- STAGE B2: LOCATION REFINEMENT ---
    tprint(f"STAGE B2: Location Refinement (Winning Contexts x Location) [{side}]")

    fp_b_loc = FeatureProcessor()
    X_b_loc, metadata_b_loc, audits_b_loc = fp_b_loc.prepare_features(
        feature_dict,
        data["timestamp"].to_numpy(),
        data["symbol"].to_numpy(),
        cfg,
        active_groups=("location",),
        extra_binary_features=context_feature_dict,
        extra_feature_group="context",
    )
    stage_b_loc_output_dir = side_output_dir / "stage_b_location_refinement"
    stage_b_loc_output_dir.mkdir(parents=True, exist_ok=True)

    for k, v in audits_b_loc.items():
        if not v.empty:
            v.to_csv(stage_b_loc_output_dir / f"{k}.csv", index=False)
    context_mapping_df.to_csv(
        stage_b_loc_output_dir / "stage_b_context_mapping.csv", index=False
    )

    stage_b_loc_spec = MiningStageSpec(
        stage_name="stage_b_location_refinement",
        active_groups=("location", "context"),
        allow_groups_in_rule=("location", "context"),
        output_dir_name="stage_b_location_refinement",
        allowed_group_pairs=(("location", "context"),),
        slot_order=("location", "context"),
        require_uplift=True,
    )

    def reconstruct_stage_b_loc_key(
        raw_key: str,
    ) -> Tuple[Optional[str], Optional[str]]:
        global stage_b_reconstruction_success, stage_b_reconstruction_failed
        slots = raw_key.split("|")
        location_conditions = []
        parent_context_key = None
        for slot in slots:
            slot_value = slot.strip("()")
            if slot_value == "*":
                continue
            for cond_str in slot_value.split("&"):
                if "==" not in cond_str:
                    continue
                feature_name = cond_str.split("==")[0]
                if feature_name in LOCATION_FILTER_COLUMNS or any(
                    feature_name.startswith(c[:3]) for c in CONTINUOUS_LOCATION_COLS
                ):
                    location_conditions.append(cond_str)
                elif feature_name.startswith("ctx__"):
                    parent_context_key = context_to_key.get(feature_name)

        if parent_context_key is None or not location_conditions:
            stage_b_reconstruction_failed += 1
            return None, None

        location_slot = f"({'&'.join(sorted(location_conditions))})"
        parent_slots = parent_context_key.split("|")
        stage_b_reconstruction_success += 1
        # Merge with the existing location slot from the parent context
        existing_loc = parent_slots[1].strip("()")
        if existing_loc != "*":
            new_loc_slot = f"({existing_loc}&{location_slot.strip('()')})"
        else:
            new_loc_slot = location_slot
        return f"{parent_slots[0]}|{new_loc_slot}|{parent_slots[2]}", parent_context_key

    cfg_loc = cfg.copy()
    cfg_loc["lgbm_max_depth"] = 2
    context_features_loc = [
        m.feature_name for m in metadata_b_loc if m.group == "context"
    ]
    location_features = [
        m.feature_name for m in metadata_b_loc if m.group == "location"
    ]
    cfg_loc["interaction_constraints"] = [context_features_loc + location_features]
    cfg_loc["require_groups"] = ["context", "location"]

    stage_b_loc_result = run_mining_stage(
        data,
        side_fwd_ret,
        side_fwd_ret_norm,
        X_b_loc,
        metadata_b_loc,
        cfg_loc,
        stage_b_loc_output_dir,
        stage_b_loc_spec.stage_name,
        stage_b_loc_spec.allowed_group_pairs,
        slot_order=stage_b_loc_spec.slot_order,
        folds=folds,
        mask_resolver=CanonicalRuleMaskResolver(
            X_b_loc,
            metadata_b_loc,
            context_lookup=context_feature_dict,
            context_key_map=context_to_key,
            slot_order=("trigger", "location", "regime"),
        ),
        require_uplift=stage_b_loc_spec.require_uplift,
        rule_key_rewriter=reconstruct_stage_b_loc_key,
        pipeline_stage_name="stage_b_location_refinement",
        explicit_side=side,
        target_name=target_name,
        horizon=horizon,
    )

    # Only location refinement active (trigger disabled)
    stage_b_accepted = stage_b_loc_result["accepted_registry"].copy()

    return {
        "stage_a": winning_contexts,
        "stage_a_result": stage_a_result,
        "stage_b": stage_b_accepted,
        "stage_b_trig_result": stage_b_trig_result,
        "stage_b_loc_result": stage_b_loc_result,
        "metadata_a": metadata_a,
        "metadata_b_trig": metadata_b_trig,
        "metadata_b_loc": metadata_b_loc,
        "X_a": X_a,
        "X_b_trig": X_b_trig,
        "X_b_loc": X_b_loc,
        "context_feature_dict": context_feature_dict,
        "context_to_key": context_to_key,
    }


def run_two_stage_lgbm_mask_generation(
    data: pd.DataFrame,
    feature_dict: Dict[str, np.ndarray],
    fwd_ret: np.ndarray,
    fwd_ret_norm: np.ndarray,
    cfg: Dict[str, Any],
) -> Dict[str, pd.DataFrame]:
    tprint("=" * 80)
    tprint("TWO-STAGE LGBM MASK GENERATION: START")
    tprint("=" * 80)

    root_output_dir = build_run_output_dir(cfg)
    folds = build_walk_forward_folds(
        n_samples=len(data),
        n_folds=int(cfg.get("n_folds", 5)),
        min_train_frac=float(cfg.get("cv_min_train_frac", 0.5)),
        embargo=int(cfg.get("cv_embargo", 0)),
    )

    long_results = run_side_pipeline(
        "long", data, feature_dict, fwd_ret, fwd_ret_norm, cfg, folds, root_output_dir
    )
    short_results = run_side_pipeline(
        "short", data, feature_dict, fwd_ret, fwd_ret_norm, cfg, folds, root_output_dir
    )

    # --- COMBINE METRICS FOR B1 vs B2 ---
    def process_side_metrics(
        side: str, side_results: Dict[str, Any], side_output_dir: Path
    ):
        stage_b_trig_result = side_results["stage_b_trig_result"]
        stage_b_loc_result = side_results["stage_b_loc_result"]
        stage_a_result = side_results.get("stage_a_result", {})

        if stage_b_trig_result is None or stage_b_loc_result is None:
            return

        # --- CALIBRATION BY SCORE BUCKET ---
        def compute_calibration(registry_df, output_prefix, output_dir):
            if registry_df is None or registry_df.empty:
                return

            # We bucket by composite_score into deciles
            try:
                registry_df["score_bucket"] = pd.qcut(
                    registry_df["composite_score"],
                    q=10,
                    labels=False,
                    duplicates="drop",
                )
            except ValueError:
                # Fallback if too few unique values
                registry_df["score_bucket"] = pd.cut(
                    registry_df["composite_score"], bins=10, labels=False
                )

            calib_stats = (
                registry_df.groupby("score_bucket")
                .agg(
                    mean_score=("composite_score", "mean"),
                    min_score=("composite_score", "min"),
                    max_score=("composite_score", "max"),
                    count=("canonical_key", "count"),
                    mean_oos_ret=("mean_net_ret", "mean"),
                    mean_oos_ic=("mean_oos_ic", "mean"),
                    median_support=("mean_support_pct", "median"),
                )
                .reset_index()
            )

            calib_stats.to_csv(
                output_dir / f"{output_prefix}_calibration_by_score_bucket.csv",
                index=False,
            )

        compute_calibration(
            stage_b_trig_result["accepted_registry"].copy(),
            "stage_b1_trigger",
            side_output_dir,
        )
        compute_calibration(
            stage_b_loc_result["accepted_registry"].copy(),
            "stage_b2_location",
            side_output_dir,
        )
        if stage_a_result.get("candidate_registry") is not None:
            compute_calibration(
                stage_a_result["candidate_registry"].copy(),
                "stage_a",
                side_output_dir,
            )

        # --- METRICS COMPARISON (STAGE B1 vs STAGE B2) ---
        def compute_set_metrics(registry_df, lineage_df=None):
            if registry_df is None or registry_df.empty:
                return {}

            # Sort by mean_oos_ic to get top 10
            sorted_by_ic = registry_df.sort_values(by="mean_oos_ic", ascending=False)
            top10 = sorted_by_ic.head(10)

            metrics = {
                "median_support_pct": float(registry_df["mean_support_pct"].median()),
                "p10_support_pct": float(top10["mean_support_pct"].mean()),
                "p10_mean_oos_return": float(top10["mean_net_ret"].mean()),
                "p10_std_oos_return_between_folds": float(top10["std_net_ret"].mean())
                if "std_net_ret" in top10
                else 0.0,
                "p10_std_oos_return_within_folds": float(
                    top10["mean_within_fold_std"].mean()
                )
                if "mean_within_fold_std" in top10
                else 0.0,
                "hurdle_excess_mean": float(registry_df["hurdle_excess"].mean()),
                "hurdle_excess_p10": float(top10["hurdle_excess"].mean()),
            }

            # Calculate merge average for p10
            if lineage_df is not None and not lineage_df.empty:
                merge_counts = []
                for key in top10["canonical_key"]:
                    ancestors = set()
                    to_visit = [key]
                    while to_visit:
                        curr = to_visit.pop()
                        parents = lineage_df[lineage_df["child_id"] == curr][
                            "parent_ids"
                        ].tolist()
                        if parents:
                            for p_list in parents:
                                if isinstance(p_list, str):
                                    import ast

                                    try:
                                        p_list = ast.literal_eval(p_list)
                                    except Exception:
                                        continue
                                for p in p_list:
                                    if p not in ancestors:
                                        ancestors.add(p)
                                        to_visit.append(p)
                    merge_counts.append(len(ancestors))
                metrics["p10_merge_average"] = (
                    float(np.mean(merge_counts)) if merge_counts else 0.0
                )
            else:
                metrics["p10_merge_average"] = 0.0

            return metrics

        stage_b_trig_output_dir = side_output_dir / "stage_b_trigger_refinement"
        stage_b_loc_output_dir = side_output_dir / "stage_b_location_refinement"

        try:
            b1_lineage = pd.read_csv(
                stage_b_trig_output_dir / "consolidation_lineage_audit.csv"
            )
        except Exception:
            b1_lineage = pd.DataFrame()

        try:
            b2_lineage = pd.read_csv(
                stage_b_loc_output_dir / "consolidation_lineage_audit.csv"
            )
        except Exception:
            b2_lineage = pd.DataFrame()

        b1_metrics = compute_set_metrics(
            stage_b_trig_result["accepted_registry"].copy(), b1_lineage
        )
        b2_metrics = compute_set_metrics(
            stage_b_loc_result["accepted_registry"].copy(), b2_lineage
        )

        stage_b_comparison = pd.DataFrame(
            [
                {"stage": "b1_trigger", **b1_metrics},
                {"stage": "b2_location", **b2_metrics},
            ]
        )
        stage_b_comparison.to_csv(
            side_output_dir / "stage_b_comparison_metrics.csv", index=False
        )
        tprint(f"Stage B Comparison Metrics saved to stage_b_comparison_metrics.csv")

    process_side_metrics("long", long_results, root_output_dir / "long")
    process_side_metrics("short", short_results, root_output_dir / "short")

    long_pre_global_raw = create_pre_global_registry(long_results)
    short_pre_global_raw = create_pre_global_registry(short_results)

    combined_pre_global_raw = pd.concat(
        [long_pre_global_raw, short_pre_global_raw], ignore_index=True
    )
    if combined_pre_global_raw.empty:
        tprint("No rules generated from either side. Aborting Global Consolidation.")
        return {
            "stage_a": pd.DataFrame(),
            "stage_b": pd.DataFrame(),
            "combined": pd.DataFrame(),
        }

    origin_counts = {"long_only": 0, "short_only": 0, "overlapping_keys": 0}
    l_keys = (
        set(long_pre_global_raw["canonical_key"])
        if not long_pre_global_raw.empty
        else set()
    )
    s_keys = (
        set(short_pre_global_raw["canonical_key"])
        if not short_pre_global_raw.empty
        else set()
    )

    origin_counts["overlapping_keys"] = len(l_keys.intersection(s_keys))
    origin_counts["long_only"] = len(l_keys - s_keys)
    origin_counts["short_only"] = len(s_keys - l_keys)

    pd.DataFrame([origin_counts]).to_csv(
        root_output_dir / "combined_registry_side_summary.csv", index=False
    )

    combined_pre_global = combined_pre_global_raw.sort_values(
        ["composite_score", "hurdle_excess"], ascending=False
    )
    combined_pre_global = combined_pre_global.drop_duplicates(
        subset=["canonical_key", "side"], keep="first"
    )
    combined_pre_global["preset"] = cfg.get("preset", "exploration")
    combined_pre_global.to_csv(
        root_output_dir / "combined_accepted_registry_pre_global.csv", index=False
    )

    combined_mask_map: Dict[str, np.ndarray] = {}
    combined_parent_context_map: Dict[str, str] = {}
    combined_side_map: Dict[str, str] = {}

    def extract_masks_from_results(results: Dict[str, Any], side: str):
        if results["X_a"] is None:
            return
        stage_a_accepted = results["stage_a"]
        stage_b_accepted = results["stage_b"]

        stage_a_resolver = CanonicalRuleMaskResolver(
            results["X_a"], results["metadata_a"]
        )
        
        # Only build Stage B resolver if it was run
        stage_b_resolver = None
        if results["X_b_trig"] is not None:
            stage_b_resolver = CanonicalRuleMaskResolver(
                results["X_b_trig"],
                results["metadata_b_trig"],
                context_lookup=results["context_feature_dict"],
                context_key_map=results["context_to_key"],
                slot_order=("trigger", "location", "regime"),
            )
        elif results["X_b_loc"] is not None:
            stage_b_resolver = CanonicalRuleMaskResolver(
                results["X_b_loc"],
                results["metadata_b_loc"],
                context_lookup=results["context_feature_dict"],
                context_key_map=results["context_to_key"],
                slot_order=("trigger", "location", "regime"),
            )
        stage_a_accepted = results["stage_a"]
        if stage_a_accepted is not None and not stage_a_accepted.empty:
            for _, row in stage_a_accepted.iterrows():
                combined_mask_map[row["canonical_key"]] = stage_a_resolver.get_mask(
                    row["canonical_key"]
                )
                combined_side_map[row["canonical_key"]] = row["side"]

        stage_b_accepted = results["stage_b"]
        if stage_b_accepted is not None and not stage_b_accepted.empty:
            for _, row in stage_b_accepted.iterrows():
                try:
                    combined_mask_map[row["canonical_key"]] = stage_b_resolver.get_mask(
                        row["canonical_key"]
                    )
                except Exception:
                    # fallback to loc resolver
                    stage_b_loc_resolver = CanonicalRuleMaskResolver(
                        results["X_b_loc"],
                        results["metadata_b_loc"],
                        context_lookup=results["context_feature_dict"],
                        context_key_map=results["context_to_key"],
                        slot_order=("trigger", "location", "regime"),
                    )
                    combined_mask_map[
                        row["canonical_key"]
                    ] = stage_b_loc_resolver.get_mask(row["canonical_key"])

                combined_side_map[row["canonical_key"]] = row["side"]
                if pd.notna(row.get("parent_context_key")):
                    combined_parent_context_map[row["canonical_key"]] = row[
                        "parent_context_key"
                    ]

    extract_masks_from_results(long_results, "long")
    extract_masks_from_results(short_results, "short")

    combined_resolver = DictionaryMaskResolver(
        combined_mask_map,
        parent_context_map=combined_parent_context_map,
        side_map=combined_side_map,
    )

    # Merge metadata from long/short, removing duplicates
    all_metadata = []
    seen_meta = set()
    for res in [long_results, short_results]:
        if res["metadata_a"]:
            for m in res["metadata_a"] + res["metadata_b_trig"] + res["metadata_b_loc"]:
                if m.feature_name not in seen_meta:
                    all_metadata.append(m)
                    seen_meta.add(m.feature_name)

    global_scorer = RuleScorer(all_metadata, cfg, mask_resolver=combined_resolver)
    use_economic_consolidator = cfg.get("use_economic_consolidator", True)

    tprint(
        f"Cross-Stage Funnel: Long ({len(long_pre_global_raw)}) + Short ({len(short_pre_global_raw)}) -> Combined Pre-Global ({len(combined_pre_global)})"
    )
    consolidator_type = (
        "EconomicRuleConsolidator" if use_economic_consolidator else "RuleConsolidator"
    )
    tprint(f"Chosen Global Consolidator: {consolidator_type}")

    if use_economic_consolidator:
        global_consolidator = EconomicRuleConsolidator(
            all_metadata,
            cfg,
            mask_resolver=combined_resolver,
            scorer=global_scorer,
        )
    else:
        global_consolidator = RuleConsolidator(
            all_metadata,
            cfg,
            mask_resolver=combined_resolver,
            scorer=global_scorer,
        )

    (
        combined_global_registry,
        global_lineage,
        global_consol_diag,
    ) = global_consolidator.consolidate(
        combined_pre_global,
        fwd_ret,  # Consolidator handles side internally via rule_profile which reads side_map and negates returns if needed, wait.
        folds,
        resolver=combined_resolver,
        data=data,
    )

    for name, obj in global_consol_diag.items():
        if isinstance(obj, pd.DataFrame):
            obj.to_csv(root_output_dir / f"{name}.csv", index=False)
        else:
            with open(root_output_dir / f"{name}.json", "w") as f:
                json.dump(obj, f, indent=2)
    combined_global_registry = combined_global_registry.drop_duplicates(
        subset=["canonical_key", "side"], keep="first"
    )
    tprint(f"Global Consolidation resulted in {len(combined_global_registry)} rules.")
    combined_global_registry["preset"] = cfg.get("preset", "exploration")
    combined_global_registry.to_csv(
        root_output_dir / "combined_accepted_rule_registry.csv", index=False
    )
    global_lineage.to_csv(
        root_output_dir / "global_consolidation_lineage.csv", index=False
    )

    # X_a represents regime + location features which are primary driver of correlation anyway
    X_for_ridge = None
    if long_results["X_a"] is not None:
        X_for_ridge = long_results["X_a"]
    elif short_results["X_a"] is not None:
        X_for_ridge = short_results["X_a"]

    portfolio_diversity_report = build_portfolio_diversity_report(
        combined_global_registry,
        combined_resolver,
        data,
        fwd_ret,
        X_for_ridge=X_for_ridge,
    )
    portfolio_diversity_report.to_csv(
        root_output_dir / "portfolio_diversity_report.csv", index=False
    )

    global malformed_key_count, unresolved_feature_count, unresolved_feature_names, stage_b_reconstruction_success, stage_b_reconstruction_failed
    tprint(
        f"Canonical Key Diagnostics: malformed={malformed_key_count}, unresolved_features={unresolved_feature_count}"
    )
    tprint(
        f"Stage B Reconstruction: success={stage_b_reconstruction_success}, failed={stage_b_reconstruction_failed}"
    )
    if unresolved_feature_names:
        tprint(f"Unresolved features: {', '.join(list(unresolved_feature_names)[:10])}")

    audit_data = {
        "malformed_key_count": malformed_key_count,
        "unresolved_feature_count": unresolved_feature_count,
        "stage_b_reconstruction_success": stage_b_reconstruction_success,
        "stage_b_reconstruction_failed": stage_b_reconstruction_failed,
    }
    pd.DataFrame([audit_data]).to_csv(
        root_output_dir / "canonical_key_parse_audit.csv", index=False
    )

    # Final Registry Breakdowns
    if not combined_global_registry.empty:
        breakdown = (
            combined_global_registry.groupby(["side", "display_arity"])
            .size()
            .reset_index(name="count")
        )
        breakdown.to_csv(
            root_output_dir / "final_registry_breakdown_by_side_arity.csv", index=False
        )

        final_summary = {
            "total_accepted": len(combined_global_registry),
            "mean_support_pct": float(
                combined_global_registry["mean_support_pct"].mean()
            ),
            "median_support_pct": float(
                combined_global_registry["mean_support_pct"].median()
            ),
            "mean_hurdle_excess": float(
                combined_global_registry["hurdle_excess"].mean()
            ),
            "median_hurdle_excess": float(
                combined_global_registry["hurdle_excess"].median()
            ),
        }

        # Count by origin
        if "origin_stage" in combined_global_registry.columns:
            for origin, count in (
                combined_global_registry["origin_stage"].value_counts().items()
            ):
                final_summary[f"origin_{origin}"] = count

        # Count by rule type
        if "rule_type" in combined_global_registry.columns:
            for rtype, count in (
                combined_global_registry["rule_type"].value_counts().items()
            ):
                final_summary[f"type_{rtype}"] = count

        # Portfolio Diversity Highlights
        eff_rules = portfolio_diversity_report[
            portfolio_diversity_report["metric"] == "effective_independent_rules"
        ]["value"].values
        if len(eff_rules) > 0:
            final_summary["effective_independent_rules"] = float(eff_rules[0])

        top_rule_share = portfolio_diversity_report[
            portfolio_diversity_report["metric"] == "top_rule_share"
        ]["value"].values
        if len(top_rule_share) > 0:
            final_summary["top_rule_share"] = float(top_rule_share[0])

        with open(root_output_dir / "final_registry_summary.json", "w") as f:
            json.dump(final_summary, f, indent=2)

        tprint(f"Final Output Summary: {len(combined_global_registry)} rules.")
        tprint(f"  - Mean Support: {final_summary['mean_support_pct']:.2%}")
        tprint(f"  - Median Hurdle Excess: {final_summary['median_hurdle_excess']:.5f}")
        if "effective_independent_rules" in final_summary:
            tprint(
                f"  - Effective Independent Rules: {final_summary['effective_independent_rules']:.2f}"
            )

        tprint("Side Mix:")
        side_counts = combined_global_registry["side"].value_counts()
        for side, count in side_counts.items():
            tprint(f"  - {side}: {count}")

        tprint("Top 10 Final Rules:")
        top_final = combined_global_registry.sort_values(
            "composite_score", ascending=False
        ).head(10)
        for _, row in top_final.iterrows():
            tprint(
                f"  - {row['canonical_key']}: score={row['composite_score']:.3f}, arity={row['display_arity']}, side={row['side']}"
            )
    else:
        tprint("Final Output Summary: 0 rules accepted.")

    tprint(
        f"Two-stage mining complete. Total accepted rules: {len(combined_global_registry)}"
    )
    return {
        "stage_a_long": long_results["stage_a"],
        "stage_b_long": long_results["stage_b"],
        "stage_a_short": short_results["stage_a"],
        "stage_b_short": short_results["stage_b"],
        "combined": combined_global_registry,
    }


def run_two_stage_lgbm_mask_generation_triad(
    data: pd.DataFrame,
    feature_dict: Dict[str, np.ndarray],
    triad_targets: Dict[str, Dict[int, np.ndarray]],
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Triad-target version of the mask generation pipeline.

    Runs the Cartesian product:
        for horizon in horizons:
            for target_name in [eff, ela, vame]:
                for side in [long, short]:
                    run_side_pipeline(...)

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with event metadata
    feature_dict : Dict[str, np.ndarray]
        Dictionary of feature arrays
    triad_targets : Dict[str, Dict[int, np.ndarray]]
        Nested dict of bounded targets: {target_name: {horizon: target_array}}
        e.g., {"target_eff": {5: array, 10: array, ...}, ...}
    cfg : Dict[str, Any]
        Configuration dictionary

    Returns
    -------
    Dict[str, Any]
        Merged discovery outputs with provenance. Structure:
        {
            "results_by_target_horizon": {
                ("target_eff", 5): {"long": ..., "short": ...},
                ...
            },
            "combined_registry": pd.DataFrame,
        }
    """
    tprint("=" * 80)
    tprint("TRIAD-TARGET TWO-STAGE LGBM MASK GENERATION: START")
    tprint("=" * 80)

    # Get horizons and target names from config or defaults
    horizons = cfg.get("triad_horizons", TRIAD_DEFAULT_HORIZONS)
    target_names = cfg.get("triad_target_names", TRIAD_DEFAULT_TARGET_NAMES)

    tprint(f"Horizons: {horizons}")
    tprint(f"Targets: {target_names}")
    tprint(f"Total combinations: {len(horizons) * len(target_names) * 2} (horizons × targets × sides)")

    root_output_dir = build_run_output_dir(cfg)
    
    # Build folds once (shared across all combinations)
    folds = build_walk_forward_folds(
        n_samples=len(data),
        n_folds=int(cfg.get("n_folds", 5)),
        min_train_frac=float(cfg.get("cv_min_train_frac", 0.5)),
        embargo=int(cfg.get("cv_embargo", 0)),
    )

    # Store results by (target_name, horizon, side)
    results_by_target_horizon: Dict[Tuple[str, int, str], Dict[str, Any]] = {}
    
    # Track all registries for final combination
    all_registries: List[pd.DataFrame] = []

    total_combinations = len(horizons) * len(target_names)
    current_combination = 0

    for horizon in horizons:
        for target_name in target_names:
            current_combination += 1
            tprint(f"\n{'='*60}")
            tprint(f"COMBINATION {current_combination}/{total_combinations}: {target_name} @ H{horizon}")
            tprint(f"{'='*60}")

            # Get bounded target for this target_name/horizon
            bounded_target = None
            if target_name in triad_targets:
                if horizon in triad_targets[target_name]:
                    bounded_target = triad_targets[target_name][horizon]
                    tprint(f"Using bounded target array: shape={bounded_target.shape}, "
                           f"range=[{np.nanmin(bounded_target):.3f}, {np.nanmax(bounded_target):.3f}]")
                else:
                    tprint(f"WARNING: No bounded target for {target_name} @ H{horizon}, skipping")
                    continue
            else:
                tprint(f"WARNING: Target {target_name} not found in triad_targets, skipping")
                continue

            # Build horizon-specific output directory
            horizon_target_dir = root_output_dir / f"h{horizon}" / target_name
            horizon_target_dir.mkdir(parents=True, exist_ok=True)

            # Create horizon-specific config with target settings
            horizon_cfg = cfg.copy()
            horizon_cfg["target_name"] = target_name
            horizon_cfg["horizon"] = horizon
            
            # Apply per-target config from TRIAD_TARGET_CONFIGS
            if target_name in TRIAD_TARGET_CONFIGS:
                target_config = TRIAD_TARGET_CONFIGS[target_name]
                horizon_cfg["huber_alpha"] = target_config.get("huber_alpha", 0.9)
                horizon_cfg["learning_rate"] = target_config.get("learning_rate", 0.03)
                horizon_cfg["min_support_pct"] = target_config.get("min_support_pct", 0.05)
                horizon_cfg["ic_hurdle"] = target_config.get("ic_hurdle", 0.02)
                tprint(f"Applied target config: huber_alpha={target_config.get('huber_alpha')}, "
                       f"lr={target_config.get('learning_rate')}")

            # Apply horizon-specific config from HORIZON_CONFIGS
            if horizon in HORIZON_CONFIGS:
                horizon_config = HORIZON_CONFIGS[horizon]
                min_leaf_mult = horizon_config.get("min_data_in_leaf_multiplier", 1.0)
                base_min_leaf = int(horizon_cfg.get("min_data_in_leaf", 64))
                horizon_cfg["min_data_in_leaf"] = int(base_min_leaf * min_leaf_mult)
                tprint(f"Applied horizon config: min_data_in_leaf_multiplier={min_leaf_mult}")

            # For triad targets, we use the bounded_target itself as the "return" for scoring
            # so that RuleScorer and MaskAssessor see the actual edge we are mining.
            dummy_fwd_ret = bounded_target.astype(np.float32, copy=True)
            dummy_fwd_ret_norm = bounded_target.astype(np.float32, copy=True)

            # Run both sides
            for side in ["long", "short"]:
                tprint(f"\n--- Running {side.upper()} side for {target_name} @ H{horizon} ---")
                
                side_results = run_side_pipeline(
                    side=side,
                    data=data,
                    feature_dict=feature_dict,
                    fwd_ret=dummy_fwd_ret,
                    fwd_ret_norm=dummy_fwd_ret_norm,
                    cfg=horizon_cfg,
                    folds=folds,
                    root_output_dir=horizon_target_dir,
                    target_name=target_name,
                    horizon=horizon,
                    bounded_target=bounded_target,
                )

                # Store results
                results_by_target_horizon[(target_name, horizon, side)] = side_results

                # Add provenance to registry
                if not side_results["stage_a"].empty:
                    stage_a_with_prov = side_results["stage_a"].copy()
                    stage_a_with_prov["source_target"] = target_name
                    stage_a_with_prov["source_horizon"] = horizon
                    all_registries.append(stage_a_with_prov)

                if not side_results["stage_b"].empty:
                    stage_b_with_prov = side_results["stage_b"].copy()
                    stage_b_with_prov["source_target"] = target_name
                    stage_b_with_prov["source_horizon"] = horizon
                    all_registries.append(stage_b_with_prov)

    # Combine all registries
    tprint(f"\n{'='*60}")
    tprint("COMBINING ALL TRIAD RESULTS")
    tprint(f"{'='*60}")

    if all_registries:
        combined_registry = pd.concat(all_registries, ignore_index=True)
        combined_registry = combined_registry.drop_duplicates(
            subset=["canonical_key", "side", "source_target", "source_horizon"],
            keep="first"
        )
    else:
        combined_registry = pd.DataFrame()

    # Save combined registry
    combined_registry["preset"] = cfg.get("preset", "triad_exploration")
    combined_registry.to_csv(
        root_output_dir / "triad_combined_registry.csv", index=False
    )

    # Summary statistics
    tprint(f"\nTRIAD TRAINING COMPLETE")
    tprint(f"Total combinations processed: {current_combination}/{total_combinations}")
    tprint(f"Total rules discovered: {len(combined_registry)}")

    if not combined_registry.empty:
        # Breakdown by target/horizon
        breakdown = (
            combined_registry
            .groupby(["source_target", "source_horizon", "side"])
            .size()
            .reset_index(name="count")
        )
        breakdown.to_csv(
            root_output_dir / "triad_breakdown_by_target_horizon.csv", index=False
        )
        tprint("\nBreakdown by target/horizon/side:")
        for _, row in breakdown.iterrows():
            tprint(f"  {row['source_target']} H{row['source_horizon']} {row['side']}: {row['count']}")

    # Save configuration used
    triad_config = {
        "horizons": horizons,
        "target_names": target_names,
        "total_combinations": total_combinations,
        "completed_combinations": current_combination,
        "total_rules": len(combined_registry),
    }
    with open(root_output_dir / "triad_config.json", "w") as f:
        json.dump(triad_config, f, indent=2)

    # Run merged discovery analysis
    tprint(f"\n{'='*60}")
    tprint("MERGED DISCOVERY ANALYSIS")
    tprint(f"{'='*60}")
    
    # Build all_results list for merge functions
    all_results = []
    for (t_name, h, side), result in results_by_target_horizon.items():
        all_results.append({
            "target_name": t_name,
            "horizon": h,
            "side": side,
            "accepted_rules": result.get("accepted_registry", pd.DataFrame()).to_dict("records") if not result.get("accepted_registry", pd.DataFrame()).empty else [],
            "candidate_rules": result.get("candidate_registry", pd.DataFrame()).to_dict("records") if not result.get("candidate_registry", pd.DataFrame()).empty else [],
        })
    
    # Merge discovery outputs
    merged_output = merge_discovery_outputs_across_targets(
        all_results=all_results,
        output_dir=str(root_output_dir),
    )
    
    # Create cross-target analysis
    cross_target_analysis = analyze_cross_target_rules(merged_output["merged_rules"])
    
    # Create target quality summary
    quality_summary = create_target_quality_summary(
        all_results=all_results,
        output_path=str(root_output_dir / "target_quality_summary.csv"),
    )
    
    # Save merged outputs
    if not merged_output["merged_rules"].empty:
        merged_output["merged_rules"].to_csv(
            root_output_dir / "merged_discovery_all.csv", index=False
        )
        tprint(f"Saved merged_discovery_all.csv: {len(merged_output['merged_rules'])} rules")
    
    if not merged_output["dedup_rules"].empty:
        merged_output["dedup_rules"].to_csv(
            root_output_dir / "merged_discovery_dedup.csv", index=False
        )
        tprint(f"Saved merged_discovery_dedup.csv: {len(merged_output['dedup_rules'])} deduplicated rules")
    
    # Save cross-target analysis
    with open(root_output_dir / "cross_target_analysis.json", "w") as f:
        json.dump(cross_target_analysis, f, indent=2, default=str)
    
    # Save triad run summary
    triad_run_summary = {
        "horizons": horizons,
        "target_names": target_names,
        "total_combinations": total_combinations,
        "completed_combinations": current_combination,
        "total_rules": len(combined_registry),
        "deduplicated_rules": len(merged_output["dedup_rules"]),
        "cross_target_rules_count": len(cross_target_analysis.get("cross_target_rules", [])),
        "universal_rules_count": len(cross_target_analysis.get("universal_rules", [])),
        "merge_stats": merged_output.get("summary_stats", {}),
    }
    with open(root_output_dir / "triad_run_summary.json", "w") as f:
        json.dump(triad_run_summary, f, indent=2, default=str)
    
    tprint(f"\nTRIAD TRAINING COMPLETE")
    tprint(f"Total rules: {len(combined_registry)}")
    tprint(f"Deduplicated rules: {len(merged_output['dedup_rules'])}")
    tprint(f"Cross-target rules: {len(cross_target_analysis.get('cross_target_rules', []))}")
    tprint(f"Universal rules: {len(cross_target_analysis.get('universal_rules', []))}")

    return {
        "results_by_target_horizon": {
            (t, h): {
                "long": results_by_target_horizon.get((t, h, "long"), {}),
                "short": results_by_target_horizon.get((t, h, "short"), {}),
            }
            for t in target_names
            for h in horizons
        },
        "combined_registry": combined_registry,
        "merged_output": merged_output,
        "cross_target_analysis": cross_target_analysis,
        "quality_summary": quality_summary,
    }


# =============================================================================
# MERGED DISCOVERY OUTPUT FUNCTIONS
# =============================================================================


def merge_discovery_outputs_across_targets(
    all_results: List[Dict[str, Any]],
    output_dir: str,
    dedup_strategy: str = "canonical_key",
) -> Dict[str, Any]:
    """
    Merge discovered rules/contexts from all (target, horizon) runs.
    
    Parameters
    ----------
    all_results : List[Dict[str, Any]]
        List of result dicts from run_side_pipeline().
        Each contains:
        - target_name: str
        - horizon: int
        - side: str
        - accepted_rules: list of rule dicts
        - candidate_rules: list of rule dicts
    output_dir : str
        Base output directory
    dedup_strategy : str
        How to deduplicate ("canonical_key" or "structural")
    
    Returns
    -------
    Dict[str, Any]
        Dict with:
        - merged_rules: pd.DataFrame with all rules and provenance
        - dedup_rules: pd.DataFrame deduplicated by canonical key
        - cross_target_rules: rules appearing in multiple targets
        - summary_stats: dict of merge statistics
    """
    tprint(f"Merging discovery outputs from {len(all_results)} result sets...")
    
    all_rules = []
    for result in all_results:
        target_name = result.get("target_name", "unknown")
        horizon = result.get("horizon", 0)
        side = result.get("side", "unknown")
        
        # Process accepted rules
        for rule in result.get("accepted_rules", []):
            rule_with_prov = dict(rule)
            rule_with_prov["source_target"] = target_name
            rule_with_prov["source_horizon"] = horizon
            rule_with_prov["side"] = side
            rule_with_prov["rule_status"] = "accepted"
            all_rules.append(rule_with_prov)
        
        # Process candidate rules
        for rule in result.get("candidate_rules", []):
            rule_with_prov = dict(rule)
            rule_with_prov["source_target"] = target_name
            rule_with_prov["source_horizon"] = horizon
            rule_with_prov["side"] = side
            rule_with_prov["rule_status"] = "candidate"
            all_rules.append(rule_with_prov)
    
    if not all_rules:
        tprint("No rules to merge.")
        return {
            "merged_rules": pd.DataFrame(),
            "dedup_rules": pd.DataFrame(),
            "cross_target_rules": pd.DataFrame(),
            "summary_stats": {"total_rules": 0, "unique_canonical_keys": 0},
        }
    
    merged_df = pd.DataFrame(all_rules)
    tprint(f"Total rules collected: {len(merged_df)}")
    
    # Deduplicate by canonical key
    dedup_df = deduplicate_rules_by_canonical_key(all_rules, aggregation="mean")
    tprint(f"Unique canonical keys: {len(dedup_df)}")
    
    # Summary stats
    summary_stats = {
        "total_rules": len(merged_df),
        "unique_canonical_keys": len(dedup_df),
        "targets_represented": merged_df["source_target"].nunique() if "source_target" in merged_df.columns else 0,
        "horizons_represented": merged_df["source_horizon"].nunique() if "source_horizon" in merged_df.columns else 0,
        "sides_represented": merged_df["side"].nunique() if "side" in merged_df.columns else 0,
    }
    
    return {
        "merged_rules": merged_df,
        "dedup_rules": dedup_df,
        "cross_target_rules": dedup_df[dedup_df["supporting_targets_count"] > 1] if "supporting_targets_count" in dedup_df.columns else pd.DataFrame(),
        "summary_stats": summary_stats,
    }


def deduplicate_rules_by_canonical_key(
    rules: List[Dict[str, Any]],
    aggregation: str = "mean",
) -> pd.DataFrame:
    """
    Deduplicate rules by canonical key, aggregating metrics.
    
    For each unique canonical_key:
    - Keep provenance from all sources
    - Aggregate numeric metrics (mean, min, max)
    - Track all source targets/horizons
    
    Parameters
    ----------
    rules : List[Dict[str, Any]]
        List of rule dictionaries
    aggregation : str
        Aggregation method ("mean", "min", "max")
    
    Returns
    -------
    pd.DataFrame
        Deduplicated rules with aggregated metrics
    """
    if not rules:
        return pd.DataFrame()
    
    df = pd.DataFrame(rules)
    
    if "canonical_key" not in df.columns:
        tprint("WARNING: No canonical_key column found, returning as-is")
        return df
    
    # Columns to aggregate (numeric metrics)
    ic_metric_cols = [
        "mean_oos_ic", "p25_oos_ic", "p50_oos_ic", "p75_oos_ic",
        "positive_ic_fraction", "within_mask_ic", "delta_within_mask_ic",
    ]
    rule_metric_cols = [
        "mean_net_ret", "std_net_ret", "mean_support_pct", "presence_freq",
        "sign_consistency", "entropy_reduction",
    ]
    all_metric_cols = ic_metric_cols + rule_metric_cols
    
    # Group by canonical_key
    grouped = df.groupby("canonical_key")
    
    # Build aggregated records
    aggregated_records = []
    for canonical_key, group in grouped:
        record = {"canonical_key": canonical_key}
        
        # Source provenance (take first/original)
        if "source_target" in group.columns:
            record["source_target"] = group["source_target"].iloc[0]
        if "source_horizon" in group.columns:
            record["source_horizon"] = group["source_horizon"].iloc[0]
        if "side" in group.columns:
            record["side"] = group["side"].iloc[0]
        
        # Supporting counts
        if "source_target" in group.columns:
            record["supporting_targets_count"] = group["source_target"].nunique()
            record["targets_supporting_rule"] = json.dumps(sorted(group["source_target"].unique().tolist()))
        else:
            record["supporting_targets_count"] = 1
            record["targets_supporting_rule"] = json.dumps([])
        
        if "source_horizon" in group.columns:
            record["supporting_horizons_count"] = group["source_horizon"].nunique()
            record["horizons_supporting_rule"] = json.dumps(sorted(group["source_horizon"].unique().tolist()))
        else:
            record["supporting_horizons_count"] = 1
            record["horizons_supporting_rule"] = json.dumps([])
        
        # Aggregate numeric metrics
        for col in all_metric_cols:
            if col in group.columns:
                col_values = group[col].dropna()
                if len(col_values) > 0:
                    if aggregation == "mean":
                        record[col] = float(col_values.mean())
                    elif aggregation == "min":
                        record[col] = float(col_values.min())
                    elif aggregation == "max":
                        record[col] = float(col_values.max())
                    else:
                        record[col] = float(col_values.mean())
                    
                    # Also store min/max for IC metrics
                    if col in ic_metric_cols:
                        record[f"{col}_min"] = float(col_values.min())
                        record[f"{col}_max"] = float(col_values.max())
        
        # Conservative metrics (min across sources)
        if "presence_freq" in group.columns:
            record["presence_freq_conservative"] = float(group["presence_freq"].dropna().min())
        if "sign_consistency" in group.columns:
            record["sign_consistency_conservative"] = float(group["sign_consistency"].dropna().min())
        
        # Structural soundness (all must be True)
        if "is_structurally_sound" in group.columns:
            record["is_structurally_sound"] = bool(group["is_structurally_sound"].fillna(False).all())
        
        # Production status (any source is production)
        # Check both production_status and production_classification columns
        if "production_classification" in group.columns:
            record["merged_production_status"] = "production" if (group["production_classification"] == "production").any() else "research"
        elif "production_status" in group.columns:
            record["merged_production_status"] = "production" if (group["production_status"] == "production").any() else "exploration"
        else:
            record["merged_production_status"] = "research"
        
        # Rule type class (ranking or gate) - use most common across sources
        if "rule_type_class" in group.columns:
            rule_type_classes = group["rule_type_class"].dropna().tolist()
            if rule_type_classes:
                record["merged_rule_type_class"] = max(set(rule_type_classes), key=rule_type_classes.count)
            else:
                record["merged_rule_type_class"] = "rejected"
        else:
            record["merged_rule_type_class"] = "rejected"
        
        # Rule type (original column - ranking or gate)
        if "rule_type" in group.columns:
            # Take most common rule type
            record["rule_type"] = group["rule_type"].mode().iloc[0] if len(group["rule_type"].mode()) > 0 else "unknown"
        
        # Copy other important columns from first row
        for col in ["display_arity", "composite_score", "hurdle_excess", "conditions"]:
            if col in group.columns:
                record[col] = group[col].iloc[0]
        
        aggregated_records.append(record)
    
    result_df = pd.DataFrame(aggregated_records)
    
    # Sort by supporting targets count (descending), then by composite score
    sort_cols = []
    if "supporting_targets_count" in result_df.columns:
        sort_cols.append("supporting_targets_count")
    if "composite_score" in result_df.columns:
        sort_cols.append("composite_score")
    if sort_cols:
        result_df = result_df.sort_values(sort_cols, ascending=[False, False])
    
    return result_df.reset_index(drop=True)


def analyze_cross_target_rules(
    merged_df: pd.DataFrame
) -> Dict[str, Any]:
    """
    Analyze rules that appear across multiple targets.
    
    Parameters
    ----------
    merged_df : pd.DataFrame
        Merged rules DataFrame with provenance
    
    Returns
    -------
    Dict[str, Any]
        - cross_target_rules: Rules appearing in 2+ targets
        - universal_rules: Rules appearing in all 3 targets
        - target_specific_rules: Rules unique to one target
        - cross_horizon_rules: Rules appearing at multiple horizons
        - single_horizon_rules: Rules at only one horizon
    """
    if merged_df.empty:
        return {
            "cross_target_rules": [],
            "universal_rules": [],
            "target_specific_rules": [],
            "cross_horizon_rules": [],
            "single_horizon_rules": [],
            "summary": {},
        }
    
    result = {
        "cross_target_rules": [],
        "universal_rules": [],
        "target_specific_rules": [],
        "cross_horizon_rules": [],
        "single_horizon_rules": [],
        "summary": {},
    }
    
    # Check if we have the necessary columns
    if "supporting_targets_count" not in merged_df.columns:
        return result
    
    # Cross-target rules (2+ targets)
    cross_target_mask = merged_df["supporting_targets_count"] >= 2
    cross_target_df = merged_df[cross_target_mask]
    result["cross_target_rules"] = cross_target_df.to_dict("records")
    
    # Universal rules (all 3 targets)
    universal_mask = merged_df["supporting_targets_count"] >= 3
    universal_df = merged_df[universal_mask]
    result["universal_rules"] = universal_df.to_dict("records")
    
    # Target-specific rules (1 target only)
    specific_mask = merged_df["supporting_targets_count"] == 1
    specific_df = merged_df[specific_mask]
    result["target_specific_rules"] = specific_df.to_dict("records")
    
    # Cross-horizon rules
    if "supporting_horizons_count" in merged_df.columns:
        cross_horizon_mask = merged_df["supporting_horizons_count"] >= 2
        cross_horizon_df = merged_df[cross_horizon_mask]
        result["cross_horizon_rules"] = cross_horizon_df.to_dict("records")
        
        single_horizon_mask = merged_df["supporting_horizons_count"] == 1
        single_horizon_df = merged_df[single_horizon_mask]
        result["single_horizon_rules"] = single_horizon_df.to_dict("records")
    
    # Summary statistics
    result["summary"] = {
        "total_rules": len(merged_df),
        "cross_target_count": len(result["cross_target_rules"]),
        "universal_count": len(result["universal_rules"]),
        "target_specific_count": len(result["target_specific_rules"]),
        "cross_horizon_count": len(result["cross_horizon_rules"]),
        "single_horizon_count": len(result["single_horizon_rules"]),
    }
    
    tprint(f"Cross-target analysis: {len(result['cross_target_rules'])} cross-target, "
           f"{len(result['universal_rules'])} universal, "
           f"{len(result['target_specific_rules'])} target-specific")
    
    return result


def classify_rule_type(
    mean_oos_ic: float,
    within_mask_ic: float,
    delta_within_mask_ic: float,
    ic_threshold: float = 0.02,
    delta_threshold: float = 0.03,
) -> str:
    """
    Classify rule as ranking vs gate regime.
    
    Ranking regime:
    - mean_oos_ic > ic_threshold
    - Can be used for continuous ranking/sizing
    
    Gate/filter regime:
    - mean_oos_ic <= ic_threshold (weak or negative global IC)
    - BUT within_mask_ic > mean_oos_ic + delta_threshold
    - OR delta_within_mask_ic > delta_threshold
    - Used for selection/gating, not continuous ranking
    
    Rejected:
    - Neither ranking nor gate characteristics
    
    Parameters
    ----------
    mean_oos_ic : float
        Mean out-of-sample IC across folds
    within_mask_ic : float
        IC computed within the mask (regime-specific IC)
    delta_within_mask_ic : float
        Difference between within_mask_ic and global IC
    ic_threshold : float
        Minimum IC for ranking regime (default 0.02)
    delta_threshold : float
        Minimum delta IC for gate regime (default 0.03)
    
    Returns
    -------
    str
        "ranking", "gate", or "rejected"
    """
    # Handle NaN values
    if not np.isfinite(mean_oos_ic):
        mean_oos_ic = -np.inf
    if not np.isfinite(within_mask_ic):
        within_mask_ic = -np.inf
    if not np.isfinite(delta_within_mask_ic):
        delta_within_mask_ic = 0.0
    
    # Ranking regime: strong global IC
    if mean_oos_ic > ic_threshold:
        return "ranking"
    
    # Gate regime: weak global IC but strong within-mask improvement
    if within_mask_ic > mean_oos_ic + delta_threshold:
        return "gate"
    
    if delta_within_mask_ic > delta_threshold:
        return "gate"
    
    # Neither ranking nor gate
    return "rejected"


def classify_rule_production_quality(
    rule: Dict[str, Any],
    min_folds: int = 3,
    min_presence_freq: float = 0.75,
    min_positive_ic_fraction: float = 0.6,
    min_mean_oos_ic: float = 0.0,
    min_support_actual: int = 50,
    min_within_mask_ic_lift: float = 0.02,
) -> Tuple[str, Dict[str, Any]]:
    """
    Classify rule as production-quality, research, or rejected.
    
    Production-quality rules must meet ALL of:
    - n_folds >= min_folds
    - presence_freq >= min_presence_freq
    - positive_ic_fraction >= min_positive_ic_fraction
    - mean_oos_ic > 0, OR within_mask_ic materially > global_ic (for gate rules)
    - min_support_actual >= min_support_actual
    - not dominated by simpler parent unless uplift is material
    - structurally sound
    
    Classification:
    - "production": Meets all criteria AND (strong IC OR strong gate characteristics)
    - "research": Meets some criteria, needs further study
    - "rejected": Fails critical criteria
    
    Parameters
    ----------
    rule : Dict[str, Any]
        Rule dictionary with metrics
    min_folds : int
        Minimum number of folds rule must appear in (default 3)
    min_presence_freq : float
        Minimum presence frequency across folds (default 0.75)
    min_positive_ic_fraction : float
        Minimum fraction of folds with positive IC (default 0.6)
    min_mean_oos_ic : float
        Minimum mean OOS IC (default 0.0)
    min_support_actual : int
        Minimum actual support count (default 50)
    min_within_mask_ic_lift : float
        Minimum within-mask IC lift for gate rules (default 0.02)
    
    Returns
    -------
    Tuple[str, Dict[str, Any]]
        (classification, diagnostics_dict)
        classification is "production", "research", or "rejected"
    """
    diagnostics: Dict[str, Any] = {
        "checks": {},
        "failures": [],
        "warnings": [],
    }
    
    # Extract metrics with safe defaults
    n_folds = rule.get("n_folds", 0)
    presence_freq = rule.get("presence_freq", 0.0)
    positive_ic_fraction = rule.get("positive_ic_fraction", 0.0)
    mean_oos_ic = rule.get("mean_oos_ic", np.nan)
    min_support_actual = rule.get("min_support_actual", 0)
    within_mask_ic = rule.get("within_mask_ic", np.nan)
    delta_within_mask_ic = rule.get("delta_within_mask_ic", np.nan)
    is_structurally_sound = rule.get("is_structurally_sound", False)
    dominated_by_parent = rule.get("dominated_by_parent", False)
    sign_consistency = rule.get("sign_consistency", 0.0)
    
    # Handle NaN values
    if not np.isfinite(mean_oos_ic):
        mean_oos_ic = -np.inf
    if not np.isfinite(within_mask_ic):
        within_mask_ic = -np.inf
    if not np.isfinite(delta_within_mask_ic):
        delta_within_mask_ic = 0.0
    
    # Check 1: Fold count
    fold_check = n_folds >= min_folds
    diagnostics["checks"]["n_folds"] = {
        "value": n_folds,
        "threshold": min_folds,
        "passed": fold_check,
    }
    if not fold_check:
        diagnostics["failures"].append(f"n_folds={n_folds} < {min_folds}")
    
    # Check 2: Presence frequency
    presence_check = presence_freq >= min_presence_freq
    diagnostics["checks"]["presence_freq"] = {
        "value": presence_freq,
        "threshold": min_presence_freq,
        "passed": presence_check,
    }
    if not presence_check:
        diagnostics["failures"].append(f"presence_freq={presence_freq:.3f} < {min_presence_freq}")
    
    # Check 3: Positive IC fraction
    ic_frac_check = positive_ic_fraction >= min_positive_ic_fraction
    diagnostics["checks"]["positive_ic_fraction"] = {
        "value": positive_ic_fraction,
        "threshold": min_positive_ic_fraction,
        "passed": ic_frac_check,
    }
    if not ic_frac_check:
        diagnostics["failures"].append(f"positive_ic_fraction={positive_ic_fraction:.3f} < {min_positive_ic_fraction}")
    
    # Check 4: Mean OOS IC (or gate exception)
    # For gate rules, we allow negative global IC if within-mask IC is strong
    is_gate_rule = (
        mean_oos_ic <= 0.02 and
        (within_mask_ic > mean_oos_ic + 0.03 or delta_within_mask_ic > 0.03)
    )
    ic_check = mean_oos_ic > min_mean_oos_ic or (is_gate_rule and delta_within_mask_ic > min_within_mask_ic_lift)
    diagnostics["checks"]["mean_oos_ic"] = {
        "value": mean_oos_ic,
        "threshold": min_mean_oos_ic,
        "is_gate_rule": is_gate_rule,
        "within_mask_ic_lift": delta_within_mask_ic,
        "passed": ic_check,
    }
    if not ic_check:
        diagnostics["failures"].append(f"mean_oos_ic={mean_oos_ic:.4f} <= {min_mean_oos_ic} and not a gate rule")
    
    # Check 5: Support count
    support_check = min_support_actual >= min_support_actual
    diagnostics["checks"]["min_support_actual"] = {
        "value": min_support_actual,
        "threshold": min_support_actual,
        "passed": support_check,
    }
    if not support_check:
        diagnostics["failures"].append(f"min_support_actual={min_support_actual} < {min_support_actual}")
    
    # Check 6: Structural soundness
    structural_check = is_structurally_sound
    diagnostics["checks"]["is_structurally_sound"] = {
        "value": is_structurally_sound,
        "passed": structural_check,
    }
    if not structural_check:
        diagnostics["failures"].append("not_structurally_sound")
    
    # Check 7: Not dominated by parent (warning only)
    if dominated_by_parent:
        diagnostics["warnings"].append("dominated_by_simpler_parent")
    
    # Check 8: Sign consistency (warning only)
    if sign_consistency < 0.75:
        diagnostics["warnings"].append(f"low_sign_consistency={sign_consistency:.3f}")
    
    # Determine classification
    critical_failures = [f for f in diagnostics["failures"] if "n_folds" in f or "structurally_sound" in f]
    
    if critical_failures:
        # Critical failures -> rejected
        classification = "rejected"
    elif len(diagnostics["failures"]) == 0:
        # All checks passed -> production
        classification = "production"
    elif len(diagnostics["failures"]) <= 2:
        # Minor failures -> research
        classification = "research"
    else:
        # Too many failures -> rejected
        classification = "rejected"
    
    diagnostics["classification"] = classification
    diagnostics["is_gate_rule"] = is_gate_rule
    
    return classification, diagnostics


def compute_overall_target_quality_score(
    mean_oos_ic: float,
    positive_ic_fraction: float,
    p25_oos_ic: float,
    entropy_reduction: float,
    delta_within_mask_ic: float,
    production_rule_count: int,
    total_rule_count: int,
) -> float:
    """
    Compute overall target quality score.
    
    Weighted combination:
    - 30%: mean_oos_ic (global learnability)
    - 20%: positive_ic_fraction (stability)
    - 15%: p25_oos_ic (worst-case IC)
    - 15%: entropy_reduction (regime separation)
    - 10%: delta_within_mask_ic (within-regime improvement)
    - 10%: production_rule_ratio (rule quality)
    
    Parameters
    ----------
    mean_oos_ic : float
        Mean out-of-sample IC
    positive_ic_fraction : float
        Fraction of folds with positive IC
    p25_oos_ic : float
        25th percentile of OOS IC
    entropy_reduction : float
        Entropy reduction from regime
    delta_within_mask_ic : float
        Within-mask IC improvement
    production_rule_count : int
        Number of production-quality rules
    total_rule_count : int
        Total number of rules
    
    Returns
    -------
    float
        Score in [0, 1] range
    """
    # Normalize each component to [0, 1]
    # IC score: assume IC in [-0.2, 0.2]
    ic_score = (mean_oos_ic + 0.2) / 0.4
    ic_score = np.clip(ic_score, 0, 1)
    
    # Stability score: already in [0, 1]
    stability_score = positive_ic_fraction
    
    # Worst-case score: same normalization as IC
    worst_case_score = (p25_oos_ic + 0.2) / 0.4
    worst_case_score = np.clip(worst_case_score, 0, 1)
    
    # Entropy reduction can be negative, normalize
    # Assume range [-0.5, 0.5]
    entropy_score = (entropy_reduction + 0.5) / 1.0
    entropy_score = np.clip(entropy_score, 0, 1)
    
    # Delta score: assume range [-0.1, 0.1]
    delta_score = (delta_within_mask_ic + 0.1) / 0.2
    delta_score = np.clip(delta_score, 0, 1)
    
    # Rule ratio: already in [0, 1]
    rule_ratio = production_rule_count / max(total_rule_count, 1)
    
    # Weighted combination
    score = (
        0.30 * ic_score +
        0.20 * stability_score +
        0.15 * worst_case_score +
        0.15 * entropy_score +
        0.10 * delta_score +
        0.10 * rule_ratio
    )
    
    return float(np.clip(score, 0, 1))


def create_triad_run_summary(
    all_results: List[Dict[str, Any]],
    merged_rules: Optional[pd.DataFrame] = None,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create summary of triad run including:
    - Per-target production rule counts
    - Per-horizon production rule counts
    - Cross-target rule counts
    - Overall quality assessment
    
    Parameters
    ----------
    all_results : List[Dict[str, Any]]
        List of result dicts from run_side_pipeline()
    merged_rules : Optional[pd.DataFrame]
        Merged/deduplicated rules DataFrame
    output_path : Optional[str]
        Path to save the summary JSON
    
    Returns
    -------
    Dict[str, Any]
        Summary dictionary with production rule counts and quality metrics
    """
    summary: Dict[str, Any] = {
        "per_target_summary": {},
        "per_horizon_summary": {},
        "cross_target_summary": {},
        "overall_quality": {},
    }
    
    # Per-target summary
    target_names = set()
    for result in all_results:
        target_name = result.get("target_name", "unknown")
        target_names.add(target_name)
        
        if target_name not in summary["per_target_summary"]:
            summary["per_target_summary"][target_name] = {
                "total_rules": 0,
                "production_rules": 0,
                "research_rules": 0,
                "rejected_rules": 0,
                "ranking_rules": 0,
                "gate_rules": 0,
            }
        
        accepted_rules = result.get("accepted_rules", [])
        summary["per_target_summary"][target_name]["total_rules"] += len(accepted_rules)
        
        for rule in accepted_rules:
            classification = rule.get("production_classification", "research")
            rule_type = rule.get("rule_type_class", "rejected")
            
            if classification == "production":
                summary["per_target_summary"][target_name]["production_rules"] += 1
            elif classification == "research":
                summary["per_target_summary"][target_name]["research_rules"] += 1
            else:
                summary["per_target_summary"][target_name]["rejected_rules"] += 1
            
            if rule_type == "ranking":
                summary["per_target_summary"][target_name]["ranking_rules"] += 1
            elif rule_type == "gate":
                summary["per_target_summary"][target_name]["gate_rules"] += 1
    
    # Per-horizon summary
    horizons = set()
    for result in all_results:
        horizon = result.get("horizon", 0)
        horizons.add(horizon)
        
        horizon_key = f"h{horizon}"
        if horizon_key not in summary["per_horizon_summary"]:
            summary["per_horizon_summary"][horizon_key] = {
                "total_rules": 0,
                "production_rules": 0,
                "research_rules": 0,
                "rejected_rules": 0,
                "ranking_rules": 0,
                "gate_rules": 0,
            }
        
        accepted_rules = result.get("accepted_rules", [])
        summary["per_horizon_summary"][horizon_key]["total_rules"] += len(accepted_rules)
        
        for rule in accepted_rules:
            classification = rule.get("production_classification", "research")
            rule_type = rule.get("rule_type_class", "rejected")
            
            if classification == "production":
                summary["per_horizon_summary"][horizon_key]["production_rules"] += 1
            elif classification == "research":
                summary["per_horizon_summary"][horizon_key]["research_rules"] += 1
            else:
                summary["per_horizon_summary"][horizon_key]["rejected_rules"] += 1
            
            if rule_type == "ranking":
                summary["per_horizon_summary"][horizon_key]["ranking_rules"] += 1
            elif rule_type == "gate":
                summary["per_horizon_summary"][horizon_key]["gate_rules"] += 1
    
    # Cross-target summary (from merged rules)
    if merged_rules is not None and not merged_rules.empty:
        cross_target_count = 0
        universal_count = 0
        production_cross_target_count = 0
        
        if "supporting_targets_count" in merged_rules.columns:
            cross_target_mask = merged_rules["supporting_targets_count"] >= 2
            cross_target_count = int(cross_target_mask.sum())
            
            universal_mask = merged_rules["supporting_targets_count"] >= 3
            universal_count = int(universal_mask.sum())
            
            # Production rules that are cross-target
            if "merged_production_status" in merged_rules.columns:
                production_cross_target = merged_rules[cross_target_mask & (merged_rules["merged_production_status"] == "production")]
                production_cross_target_count = len(production_cross_target)
        
        summary["cross_target_summary"] = {
            "cross_target_rules": cross_target_count,
            "universal_rules": universal_count,
            "production_cross_target_rules": production_cross_target_count,
            "total_merged_rules": len(merged_rules),
        }
    
    # Overall quality assessment
    total_rules = sum(t["total_rules"] for t in summary["per_target_summary"].values())
    total_production = sum(t["production_rules"] for t in summary["per_target_summary"].values())
    total_research = sum(t["research_rules"] for t in summary["per_target_summary"].values())
    total_ranking = sum(t["ranking_rules"] for t in summary["per_target_summary"].values())
    total_gate = sum(t["gate_rules"] for t in summary["per_target_summary"].values())
    
    summary["overall_quality"] = {
        "total_rules": total_rules,
        "production_rules": total_production,
        "research_rules": total_research,
        "production_ratio": total_production / max(total_rules, 1),
        "ranking_rules": total_ranking,
        "gate_rules": total_gate,
        "ranking_to_gate_ratio": total_ranking / max(total_gate, 1),
    }
    
    # Compute overall quality score
    if all_results:
        # Aggregate metrics across all results
        all_mean_oos_ic = []
        all_positive_ic_frac = []
        all_p25_oos_ic = []
        all_entropy_reduction = []
        all_delta_within_mask_ic = []
        
        for result in all_results:
            for rule in result.get("accepted_rules", []):
                if np.isfinite(rule.get("mean_oos_ic", np.nan)):
                    all_mean_oos_ic.append(rule["mean_oos_ic"])
                if np.isfinite(rule.get("positive_ic_fraction", np.nan)):
                    all_positive_ic_frac.append(rule["positive_ic_fraction"])
                if np.isfinite(rule.get("p25_oos_ic", np.nan)):
                    all_p25_oos_ic.append(rule["p25_oos_ic"])
                if np.isfinite(rule.get("entropy_reduction", np.nan)):
                    all_entropy_reduction.append(rule["entropy_reduction"])
                if np.isfinite(rule.get("delta_within_mask_ic", np.nan)):
                    all_delta_within_mask_ic.append(rule["delta_within_mask_ic"])
        
        if all_mean_oos_ic:
            overall_score = compute_overall_target_quality_score(
                mean_oos_ic=float(np.mean(all_mean_oos_ic)),
                positive_ic_fraction=float(np.mean(all_positive_ic_frac)) if all_positive_ic_frac else 0.0,
                p25_oos_ic=float(np.mean(all_p25_oos_ic)) if all_p25_oos_ic else 0.0,
                entropy_reduction=float(np.mean(all_entropy_reduction)) if all_entropy_reduction else 0.0,
                delta_within_mask_ic=float(np.mean(all_delta_within_mask_ic)) if all_delta_within_mask_ic else 0.0,
                production_rule_count=total_production,
                total_rule_count=total_rules,
            )
            summary["overall_quality"]["quality_score"] = overall_score
    
    # Print summary
    tprint("=" * 60)
    tprint("TRIAD RUN SUMMARY")
    tprint("=" * 60)
    tprint(f"Total rules discovered: {total_rules}")
    tprint(f"Production-quality rules: {total_production} ({100*total_production/max(total_rules,1):.1f}%)")
    tprint(f"Research-grade rules: {total_research}")
    tprint(f"Ranking regimes: {total_ranking}")
    tprint(f"Gate regimes: {total_gate}")
    tprint("")
    tprint("Per-target breakdown:")
    for target_name, target_summary in summary["per_target_summary"].items():
        tprint(f"  {target_name}: {target_summary['production_rules']} production / {target_summary['total_rules']} total")
    tprint("")
    tprint("Per-horizon breakdown:")
    for horizon_key, horizon_summary in summary["per_horizon_summary"].items():
        tprint(f"  {horizon_key}: {horizon_summary['production_rules']} production / {horizon_summary['total_rules']} total")
    if "quality_score" in summary["overall_quality"]:
        tprint(f"\nOverall quality score: {summary['overall_quality']['quality_score']:.3f}")
    tprint("=" * 60)
    
    # Save to file if path provided
    if output_path:
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        tprint(f"Saved triad run summary to {output_path}")
    
    return summary


def create_target_quality_summary(
    all_results: List[Dict[str, Any]],
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Create a summary of quality metrics per target/horizon combination.
    
    Parameters
    ----------
    all_results : List[Dict[str, Any]]
        List of result dicts from run_side_pipeline()
    output_path : Optional[str]
        Path to save the summary CSV
    
    Returns
    -------
    pd.DataFrame
        Summary DataFrame with quality metrics per target/horizon
    """
    summary_records = []
    
    for result in all_results:
        target_name = result.get("target_name", "unknown")
        horizon = result.get("horizon", 0)
        side = result.get("side", "unknown")
        
        accepted_rules = result.get("accepted_rules", [])
        candidate_rules = result.get("candidate_rules", [])
        
        record = {
            "target_name": target_name,
            "horizon": horizon,
            "side": side,
            "accepted_rules_count": len(accepted_rules),
            "candidate_rules_count": len(candidate_rules),
        }
        
        # Aggregate metrics from accepted rules
        if accepted_rules:
            accepted_df = pd.DataFrame(accepted_rules)
            
            # IC metrics
            for col in ["mean_oos_ic", "positive_ic_fraction", "within_mask_ic"]:
                if col in accepted_df.columns:
                    record[f"mean_{col}"] = accepted_df[col].mean()
                    record[f"max_{col}"] = accepted_df[col].max()
            
            # Rule metrics
            for col in ["composite_score", "hurdle_excess", "presence_freq", "sign_consistency"]:
                if col in accepted_df.columns:
                    record[f"mean_{col}"] = accepted_df[col].mean()
                    record[f"max_{col}"] = accepted_df[col].max()
        
        summary_records.append(record)
    
    summary_df = pd.DataFrame(summary_records)
    
    if output_path and not summary_df.empty:
        summary_df.to_csv(output_path, index=False)
        tprint(f"Saved target quality summary to {output_path}")
    
    return summary_df


class MaskAssessor:
    def __init__(
        self,
        metadata: List[FeatureMetadata],
        cfg: Dict[str, Any],
        mask_resolver: Optional[CanonicalRuleMaskResolver] = None,
    ):
        self.metadata = metadata
        self.cfg = cfg
        self.mask_resolver = mask_resolver

    @staticmethod
    def _compute_avg_trades_per_day(mask: np.ndarray, data: pd.DataFrame) -> float:
        selected_count = int(np.sum(mask))
        if selected_count == 0:
            return 0.0
        if "timestamp" not in data.columns or "symbol" not in data.columns:
            return float(selected_count)

        timestamps = pd.to_datetime(data["timestamp"], errors="coerce")
        valid_rows = timestamps.notna().to_numpy()
        if not np.any(valid_rows):
            return float(selected_count)

        working = pd.DataFrame(
            {
                "symbol": data.loc[valid_rows, "symbol"].astype(str).to_numpy(),
                "day": timestamps.loc[valid_rows].dt.floor("D").to_numpy(),
            }
        )
        rows_per_symbol_day = working.groupby(["symbol", "day"]).size()
        typical_rows_per_symbol_day = float(rows_per_symbol_day.median())
        if (
            not np.isfinite(typical_rows_per_symbol_day)
            or typical_rows_per_symbol_day <= 0
        ):
            return float(selected_count)

        total_symbol_days = float(valid_rows.sum()) / typical_rows_per_symbol_day
        if total_symbol_days <= 0:
            return float(selected_count)

        trades_per_day_per_symbol = selected_count / total_symbol_days
        return float(trades_per_day_per_symbol * 10.0)

    @staticmethod
    def _compute_oof_auc(
        oof_preds: np.ndarray,
        y: np.ndarray,
        coverage_denominator: np.ndarray,
        min_predicted_points: int = 100,
    ) -> Tuple[float, float]:
        predicted_mask = np.isfinite(oof_preds) & np.isfinite(y)
        coverage_base_mask = np.isfinite(y) & coverage_denominator.astype(bool)
        coverage_base = int(np.sum(coverage_base_mask))
        predicted_count = int(np.sum(predicted_mask))
        coverage = float(predicted_count / coverage_base) if coverage_base > 0 else 0.0

        if predicted_count < min_predicted_points:
            return np.nan, coverage
        if np.unique(y[predicted_mask]).size < 2:
            return np.nan, coverage

        try:
            auc = roc_auc_score(y[predicted_mask], oof_preds[predicted_mask])
            return max(auc, 1.0 - auc), coverage
        except ValueError:
            return np.nan, coverage

    def assess_rules(
        self,
        registry: pd.DataFrame,
        X: np.ndarray,
        data: pd.DataFrame,
        fwd_ret: np.ndarray,
        folds: List[Tuple[np.ndarray, np.ndarray]],
    ) -> pd.DataFrame:
        if registry.empty:
            return registry

        tprint(
            f"Assessing {len(registry)} rules for Structural Alpha & Learnability..."
        )
        assessment_results = []

        # For mixed side assessing, we need to be careful with global_auc/entropy.
        # It's better to compute them per-side inside the loop, but for performance, we can compute both:
        global_auc_long, global_cov_long = self._compute_baseline_auc(X, fwd_ret, folds)
        global_entropy_long = self._compute_entropy(fwd_ret)

        global_auc_short, global_cov_short = self._compute_baseline_auc(
            X, -fwd_ret, folds
        )
        global_entropy_short = self._compute_entropy(-fwd_ret)

        if np.nanstd(fwd_ret) < 1e-9:
            tprint("WARNING: Root cause for degenerate metrics: fwd_ret has zero variance!")

        # Prepare TBM data
        close = (
            data["close"].to_numpy() if "close" in data.columns else np.zeros(len(data))
        )
        high = (
            data["high"].to_numpy() if "high" in data.columns else np.zeros(len(data))
        )
        low = data["low"].to_numpy() if "low" in data.columns else np.zeros(len(data))
        atr = (
            data["atr"].to_numpy()
            if "atr" in data.columns
            else np.full(len(data), 0.001)
        )

        # TBM horizon should match assessment horizon + 2 bars
        assessment_horizon = int(self.cfg.get("horizon", 100))
        horizon = assessment_horizon + 2

        # Use the maximum horizon in the current run as reference for TP/SL scaling
        horizons = self.cfg.get("triad_horizons", [100])
        h_ref = max(horizons) if horizons else 100

        # Scale TP and SL by sqrt(H / H_ref)
        scale_factor = np.sqrt(horizon / h_ref)

        # Base TP/SL multipliers (at reference horizon)
        base_tp_atr = float(self.cfg.get("tbm_tp_atr", 1.25))
        base_sl_atr = float(self.cfg.get("tbm_sl_atr", 0.50))

        # Scaled TP/SL for current horizon
        tp_atr = base_tp_atr * scale_factor
        sl_atr = base_sl_atr * scale_factor

        tprint(f"TBM Configuration: H={horizon} bars (assessment H={assessment_horizon}), "
               f"TP={tp_atr:.3f} ATR, SL={sl_atr:.3f} ATR, scale_factor={scale_factor:.3f} (h_ref={h_ref})")

        tp_f_long, sl_f_long, to_f_long = compute_tbm_outcomes_per_symbol(
            data=data,
            horizon=horizon,
            tp_atr=tp_atr,
            sl_atr=sl_atr,
            side="long",
        )

        tp_f_short, sl_f_short, to_f_short = compute_tbm_outcomes_per_symbol(
            data=data,
            horizon=horizon,
            tp_atr=tp_atr,
            sl_atr=sl_atr,
            side="short",
        )

        for _, row in registry.iterrows():
            if self.mask_resolver:
                mask = self.mask_resolver.get_mask(row["canonical_key"])
            else:
                mask = self._get_mask_for_rule(row["canonical_key"], X)
            if np.sum(mask) < 20:
                continue

            side = row.get("side", "long")
            target_ret = -fwd_ret if side == "short" else fwd_ret

            # 1. Triple Barrier
            rule_tp_f = tp_f_short if side == "short" else tp_f_long
            rule_sl_f = sl_f_short if side == "short" else sl_f_long
            rule_to_f = to_f_short if side == "short" else to_f_long

            tbm_metrics = self._compute_tbm_metrics(
                mask, rule_tp_f, rule_sl_f, rule_to_f, target_ret
            )

            # 2. Economic Viability
            avg_trades = self._compute_avg_trades_per_day(mask, data)

            if "timestamp" in data.columns:
                days = pd.to_datetime(data["timestamp"]).dt.date
                trades_per_day = pd.Series(mask).groupby(days).sum()
                density_dispersion = trades_per_day.std() / (
                    trades_per_day.mean() + 1e-9
                )
            else:
                tprint("DEBUG: Root cause for 0.0 density_dispersion: 'timestamp' missing from data columns.")
                density_dispersion = 0.0

            # 3. Risk & Stability
            tail_ratio = self._compute_tail_ratio(target_ret[mask])
            sign_consistency = row["sign_consistency"]

            # 4. Support Constraint
            support_pct = float(np.mean(mask))
            support_ok = 0.05 <= support_pct <= 0.10
            target_support = 0.075
            support_score = -abs(support_pct - target_support)

            # 5. Return Uplift
            mean_ret_global = float(np.nanmean(target_ret))
            mean_ret_mask = float(np.nanmean(target_ret[mask]))
            ret_uplift = mean_ret_mask - mean_ret_global

            # 6. MAE/MFE Metrics
            mae, mfe = self._compute_mae_mfe(target_ret[mask])

            # 7. Learnability (Efficiency Frontier)
            global_auc = global_auc_short if side == "short" else global_auc_long
            global_entropy = (
                global_entropy_short if side == "short" else global_entropy_long
            )

            mask_auc, subset_oof_coverage = self._compute_subset_auc(
                X, target_ret, mask, folds
            )
            baseline_oof_coverage = (
                global_cov_short if side == "short" else global_cov_long
            )
            if np.isfinite(mask_auc) and np.isfinite(global_auc):
                lift = mask_auc - global_auc
            else:
                lift = np.nan

            mask_entropy = self._compute_entropy(target_ret[mask])
            entropy_red = 1.0 - (mask_entropy / (global_entropy + 1e-9))

            # 8. Event-based Expected Value
            tp_payoff = tp_atr  # TP payoff in ATR units
            sl_payoff = sl_atr  # SL payoff in ATR units
            timeout_payoff = mean_ret_mask  # Average return for timeouts

            ev_per_event = (
                tbm_metrics["tp_rate"] * tp_payoff
                - tbm_metrics["sl_rate"] * sl_payoff
                + tbm_metrics["timeout_rate"] * timeout_payoff
            )

            # 9. Final Regime Score
            regime_score = (
                0.20 * support_score +
                0.20 * lift +
                0.20 * ret_uplift +
                0.20 * ev_per_event +
                0.10 * sign_consistency
            )

            # Rejection Gates
            rejected = False
            rejection_reason = ""
            min_oof_coverage = float(
                self.cfg.get("learnability_min_oof_coverage", 0.05)
            )
            if avg_trades < float(
                self.cfg.get("min_avg_trades_per_day_10_symbols", 0.1)
            ):
                rejected, rejection_reason = True, "low_trades_per_day"
            elif not support_ok:
                rejected, rejection_reason = True, "support_out_of_range"
            elif baseline_oof_coverage < min_oof_coverage:
                rejected, rejection_reason = True, "insufficient_baseline_oof_coverage"
            elif subset_oof_coverage < min_oof_coverage:
                rejected, rejection_reason = True, "insufficient_subset_oof_coverage"
            elif not np.isfinite(lift):
                rejected, rejection_reason = True, "missing_learnability"
            elif sign_consistency < 0.75:
                rejected, rejection_reason = True, "low_sign_consistency"
            elif lift < 0.01:  # Lower threshold for lift (was 1.10)
                rejected, rejection_reason = True, "low_lift"
            # NOTE: low_entropy_reduction gate removed per user request

            # Production classification
            rule_for_classification = {
                "n_folds": row.get("n_folds", 0),
                "presence_freq": row.get("presence_freq", 0.0),
                "positive_ic_fraction": row.get("positive_ic_fraction", 0.0),
                "mean_oos_ic": row.get("mean_oos_ic", np.nan),
                "min_support_actual": row.get("min_support_actual", 0),
                "within_mask_ic": row.get("within_mask_ic", np.nan),
                "delta_within_mask_ic": row.get("delta_within_mask_ic", np.nan),
                "is_structurally_sound": not rejected,
                "dominated_by_parent": row.get("dominated_by_parent", False),
                "sign_consistency": sign_consistency,
            }
            production_classification, classification_diagnostics = classify_rule_production_quality(
                rule=rule_for_classification
            )
            
            # Rule type classification
            rule_type_class = classify_rule_type(
                mean_oos_ic=row.get("mean_oos_ic", np.nan),
                within_mask_ic=row.get("within_mask_ic", np.nan),
                delta_within_mask_ic=row.get("delta_within_mask_ic", np.nan),
            )

            assessment_results.append(
                {
                    "canonical_key": row["canonical_key"],
                    "regime_score": regime_score,
                    "is_structurally_sound": not rejected,
                    "rejection_reason": rejection_reason,
                    "support_pct": support_pct,
                    "support_ok": support_ok,
                    "support_score": support_score,
                    "avg_trades_per_day": avg_trades,
                    "density_dispersion": density_dispersion,
                    "tail_ratio": tail_ratio,
                    "mae": mae,
                    "mfe": mfe,
                    "mean_ret_global": mean_ret_global,
                    "mean_ret_mask": mean_ret_mask,
                    "ret_uplift": ret_uplift,
                    "lift": lift,
                    "learn_eff_ratio": np.nan,  # Deprecated - same as lift
                    "subset_oof_coverage": subset_oof_coverage,
                    "baseline_oof_coverage": baseline_oof_coverage,
                    "entropy_reduction": entropy_red,
                    "tp_rate": tbm_metrics["tp_rate"],
                    "sl_rate": tbm_metrics["sl_rate"],
                    "timeout_rate": tbm_metrics["timeout_rate"],
                    "ev_per_trade": tbm_metrics["ev_per_trade"],
                    "ev_per_event": ev_per_event,
                    "win_rate_conditional": tbm_metrics["win_rate_conditional"],
                    "win_rate_unconditional": tbm_metrics["win_rate_unconditional"],
                    "production_classification": production_classification,
                    "classification_diagnostics": json.dumps(classification_diagnostics),
                    "rule_type_class": rule_type_class,
                }
            )

        assessment_df = pd.DataFrame(assessment_results)
        if assessment_df.empty:
            return assessment_df

        assessed_count = len(assessment_df)
        sound_count = assessment_df["is_structurally_sound"].sum()
        rejected_count = assessed_count - sound_count

        tprint(
            f"Mask Assessor: Assessed {assessed_count} | Structurally Sound {sound_count} | Rejected {rejected_count}"
        )

        rejection_counts = assessment_df[~assessment_df["is_structurally_sound"]][
            "rejection_reason"
        ].value_counts()
        if not rejection_counts.empty:
            tprint("Top Assessor Rejection Reasons:")
            for reason, count in rejection_counts.items():
                tprint(f"  - {reason}: {count}")

        # Save rejection summary as attribute
        self.rejection_summary = rejection_counts.to_dict()

        top_sound = (
            assessment_df[assessment_df["is_structurally_sound"]]
            .sort_values("regime_score", ascending=False)
            .head(5)
        )
        if not top_sound.empty:
            tprint("Top 5 Structurally Sound Rules by Regime Score:")
            for _, row in top_sound.iterrows():
                tprint(f"  - {row['canonical_key']}: {row['regime_score']:.3f}")

        return assessment_df

    def _compute_tbm_metrics(self, mask, tp_f, sl_f, to_f, fwd_ret) -> Dict[str, float]:
        """Compute triple barrier metrics."""
        m = mask.astype(bool)
        if not np.any(m):
            return {
                "tp_rate": 0.0,
                "sl_rate": 0.0,
                "timeout_rate": 0.0,
                "ev_per_trade": 0.0,
                "win_rate_conditional": 0.0,
                "win_rate_unconditional": 0.0,
            }

        tp = np.sum(tp_f[m])
        sl = np.sum(sl_f[m])
        to = np.sum(to_f[m])
        total = np.sum(m)

        ev = np.nanmean(fwd_ret[m])

        # Conditional on a barrier hit
        win_rate_conditional = tp / (tp + sl + 1e-9)

        # Unconditional share of selected events
        win_rate_unconditional = tp / (total + 1e-9)

        return {
            "tp_rate": float(tp / total),
            "sl_rate": float(sl / total),
            "timeout_rate": float(to / total),
            "ev_per_trade": float(ev),
            "win_rate_conditional": float(win_rate_conditional),
            "win_rate_unconditional": float(win_rate_unconditional),
        }

    def _compute_cvar(self, returns, alpha=0.05) -> float:
        """Compute Conditional Value at Risk."""
        if len(returns) == 0:
            return 0.0
        n = len(returns)
        cutoff_idx = max(int(n * alpha), 1)
        sorted_rets = np.sort(returns)
        return float(np.mean(sorted_rets[:cutoff_idx]))

    def _compute_tail_ratio(self, returns) -> float:
        """Compute tail ratio (95th percentile / 5th percentile)."""
        if len(returns) < 20:
            # Root cause: insufficient samples for reliable tail ratio
            return 1.0
        p95 = abs(np.percentile(returns, 95))
        p5 = abs(np.percentile(returns, 5))
        return float(p95 / (p5 + 1e-9))

    def _compute_mae_mfe(self, returns) -> Tuple[float, float]:
        """
        Compute Maximum Adverse Excursion (MAE) and Maximum Favorable Excursion (MFE).

        MAE: Average of negative returns (average loss)
        MFE: Average of positive returns (average gain)
        """
        if len(returns) == 0:
            return 0.0, 0.0

        returns = returns[np.isfinite(returns)]
        if len(returns) == 0:
            return 0.0, 0.0

        negative_returns = returns[returns < 0]
        positive_returns = returns[returns > 0]

        mae = float(np.mean(negative_returns)) if len(negative_returns) > 0 else 0.0
        mfe = float(np.mean(positive_returns)) if len(positive_returns) > 0 else 0.0

        return mae, mfe

    def _compute_subset_auc(self, X, fwd_ret, mask, folds) -> Tuple[float, float]:
        """Compute AUC for a subset of data defined by mask."""
        if not np.any(mask):
            return np.nan, 0.0

        # Select regime features
        regime_feats = []
        for i, m in enumerate(self.metadata):
            if m.group == "regime":
                regime_feats.append(i)

        if not regime_feats:
            return np.nan, 0.0

        X_regime = X[:, regime_feats]
        y = (fwd_ret > 0).astype(float)
        y[np.isnan(fwd_ret)] = np.nan

        # Compute OOF predictions using Ridge
        from sklearn.linear_model import Ridge

        oof_preds = np.full(len(X), np.nan, dtype=np.float32)
        rng = np.random.RandomState(42)

        for fold_id, (tr_idx, va_idx) in enumerate(folds):
            # Apply mask to fold indices
            tr_masked = tr_idx[mask[tr_idx]]
            va_masked = va_idx[mask[va_idx]]

            if len(tr_masked) < 20 or len(va_masked) < 20:
                continue

            X_tr, X_va = X_regime[tr_masked], X_regime[va_masked]
            y_tr, y_va = y[tr_masked], y[va_masked]

            # Filter valid samples (y must be finite, and ALL regime features must be finite)
            valid_tr = np.isfinite(y_tr) & np.all(np.isfinite(X_tr), axis=1)
            valid_va = np.isfinite(y_va) & np.all(np.isfinite(X_va), axis=1)

            if np.sum(valid_tr) < 20 or np.sum(valid_va) < 20:
                continue

            X_tr_clean = X_tr[valid_tr]
            y_tr_clean = y_tr[valid_tr]
            X_va_clean = X_va[valid_va]
            
            # Defensive check for any remaining NaNs (should not happen with valid_tr/valid_va)
            if not np.all(np.isfinite(X_tr_clean)) or not np.all(np.isfinite(y_tr_clean)):
                continue

            # Subsample to 50% of training data
            n_samples = len(X_tr_clean)
            n_subsample = max(20, int(n_samples * 0.5))
            subsample_idx = rng.choice(n_samples, size=n_subsample, replace=False)
            X_tr_subsample = X_tr_clean[subsample_idx]
            y_tr_subsample = y_tr_clean[subsample_idx]

            # Fit Ridge on subsampled data
            model = Ridge(alpha=1.0)
            model.fit(X_tr_subsample, y_tr_subsample)
            preds = model.predict(X_va_clean)

            # Store predictions
            oof_preds[va_masked[valid_va]] = preds

        # Use lower min_predicted_points threshold for subset AUC since we're working
        # with a masked subset that may have fewer total samples
        return self._compute_oof_auc(oof_preds, y, mask, min_predicted_points=30)

    def _compute_entropy(self, y) -> float:
        """Compute entropy of the target distribution."""
        if len(y) == 0:
            return 0.0
        if np.all(np.isin(y, [0, 1])):
            p1 = np.mean(y)
            if p1 <= 0 or p1 >= 1:
                return 0.0
            return float(-(p1 * np.log2(p1) + (1 - p1) * np.log2(1 - p1)))
        else:
            return float(np.log2(np.std(y) + 1e-9))

    def _compute_baseline_auc(
        self,
        X: np.ndarray,
        fwd_ret: np.ndarray,
        folds: List[Tuple[np.ndarray, np.ndarray]],
    ) -> Tuple[float, float]:
        """
        Compute baseline AUC using regime features across all folds.
        Uses only 50% of the data for Ridge model training.
        """
        # Select regime features
        regime_feats = [i for i, m in enumerate(self.metadata) if m.group == "regime"]
        if not regime_feats:
            return np.nan, 0.0

        X_regime = X[:, regime_feats]
        y = (fwd_ret > 0).astype(float)
        y[np.isnan(fwd_ret)] = np.nan

        # Compute OOF predictions using Ridge (use 50% of data)
        from sklearn.linear_model import Ridge

        oof_preds = np.full(len(X), np.nan, dtype=np.float32)
        rng = np.random.RandomState(42)

        for fold_id, (tr_idx, va_idx) in enumerate(folds):
            X_tr, X_va = X_regime[tr_idx], X_regime[va_idx]
            y_tr, y_va = y[tr_idx], y[va_idx]

            # Filter valid samples
            valid_tr = np.isfinite(y_tr) & np.all(np.isfinite(X_tr), axis=1)
            valid_va = np.isfinite(y_va) & np.all(np.isfinite(X_va), axis=1)

            if np.sum(valid_tr) < 20 or np.sum(valid_va) < 20:
                continue

            X_tr_clean = X_tr[valid_tr]
            y_tr_clean = y_tr[valid_tr]
            X_va_clean = X_va[valid_va]

            # Subsample to 50% of training data
            n_samples = len(X_tr_clean)
            n_subsample = max(20, int(n_samples * 0.5))
            subsample_idx = rng.choice(n_samples, size=n_subsample, replace=False)
            X_tr_subsample = X_tr_clean[subsample_idx]
            y_tr_subsample = y_tr_clean[subsample_idx]

            # Fit Ridge on subsampled data
            model = Ridge(alpha=1.0)
            model.fit(X_tr_subsample, y_tr_subsample)
            preds = model.predict(X_va_clean)

            # Store predictions
            oof_preds[va_idx[valid_va]] = preds

        return self._compute_oof_auc(
            oof_preds,
            y,
            np.isfinite(y),
        )

    def _get_mask_for_rule(self, key: str, X: np.ndarray) -> np.ndarray:
        """
        Parses '(F1==1)|(LOC1==0)|(*)' into a boolean mask.
        """
        parts = key.split("|")
        mask = np.ones(X.shape[0], dtype=bool)
        for p in parts:
            p = p.strip("()")
            if p == "*":
                continue
            for cond_str in p.split("&"):
                if "==" not in cond_str:
                    continue
                fname, val_part = cond_str.split("==")
                val = int(val_part)
                # Find matching metadata for feature index
                f_idx = next(
                    m.feature_index for m in self.metadata if m.feature_name == fname
                )
                mask &= X[:, f_idx] == val
        return mask


# =============================================================================
# NUMBA-OPTIMIZED INFERENCE ENGINE
# =============================================================================


@njit(parallel=True, cache=True, fastmath=True)
def _generate_masks_numba_kernel(
    X: np.ndarray,
    cond_feat_idxs: np.ndarray,
    cond_vals: np.ndarray,
    rule_ptr: np.ndarray,
) -> np.ndarray:
    """
    Highly optimized kernel to apply N-rules to M-samples.
    rule_ptr: array of indices marking the start/end of each rule's conditions.
    """
    n_samples = X.shape[0]
    n_rules = len(rule_ptr) - 1
    out = np.ones((n_samples, n_rules), dtype=np.bool_)

    # Parallelize across samples for high-throughput
    for i in prange(n_samples):
        for r in range(n_rules):
            start = rule_ptr[r]
            end = rule_ptr[r + 1]

            # Intersection of conditions (AND logic within a path)
            for c in range(start, end):
                f_idx = cond_feat_idxs[c]
                target_val = cond_vals[c]

                # Check if boolean feature matches target normalized value
                if X[i, f_idx] != target_val:
                    out[i, r] = False
                    break
    return out


class NumbaRuleInferenceEngine:
    def __init__(self, registry: pd.DataFrame, metadata: List[FeatureMetadata]):
        self.registry = registry
        self.metadata = metadata
        self.name_to_idx = {m.feature_name: m.feature_index for m in metadata}

        # Flattened structures for Numba
        self.feat_idxs = []
        self.target_vals = []
        self.rule_ptrs = [0]

        self._compile_registry()

    def _compile_registry(self):
        """Pre-processes strings into flat integer arrays for Numba."""
        for _, row in self.registry.iterrows():
            key = row["canonical_key"]
            # Note: For Composite rules, we currently treat them as their
            # atomic components in the registry or manage them as separate vectors.
            # This engine handles standard (T|L|R) path logic.
            conditions = self._parse_key(key)

            for f_idx, val in conditions:
                self.feat_idxs.append(f_idx)
                self.target_vals.append(val)

            self.rule_ptrs.append(len(self.feat_idxs))

        self.feat_idxs_np = np.array(self.feat_idxs, dtype=np.int32)
        self.target_vals_np = np.array(self.target_vals, dtype=np.int32)
        self.rule_ptrs_np = np.array(self.rule_ptrs, dtype=np.int32)

    def _parse_key(self, key: str) -> List[Tuple[int, int]]:
        parts = key.split("|")
        parsed = []
        for p in parts:
            p = p.strip("()")
            if p == "*":
                continue
            for cond_str in p.split("&"):
                if "==" not in cond_str:
                    continue
                name, val = cond_str.split("==")
                if name in self.name_to_idx:
                    parsed.append((self.name_to_idx[name], int(val)))
                else:
                    raise KeyError(f"Feature {name} not found in metadata.")
        return parsed

    def apply(self, X: np.ndarray) -> np.ndarray:
        """Entry point for inference."""
        # X must be float32 or int for Numba kernel
        return _generate_masks_numba_kernel(
            X.astype(np.float32),
            self.feat_idxs_np,
            self.target_vals_np,
            self.rule_ptrs_np,
        )


# =============================================================================
# FEATURE USAGE AUDIT HELPERS
# =============================================================================


def export_feature_group_summary(
    metadata: List[FeatureMetadata], output_dir: Path
) -> pd.DataFrame:
    """Export feature metadata and group summary."""
    rows = []
    for m in metadata:
        rows.append(
            {
                "feature_name": m.feature_name,
                "feature_index": m.feature_index,
                "group": m.group,
                "source_name": m.source_name,
                "source_family": m.source_family,
                "source_type": m.source_type,
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "retained_feature_metadata.csv", index=False)

    summary = df.groupby("group").size().reset_index(name="retained_feature_count")
    summary.to_csv(output_dir / "retained_feature_group_summary.csv", index=False)
    return df


def collect_split_usage_from_model(
    model, metadata: List[FeatureMetadata], fold_id: int, seed: int
) -> pd.DataFrame:
    """
    Count split-feature usage directly from LightGBM tree dump.
    """
    idx_to_meta = {m.feature_index: m for m in metadata}
    dump = model.booster_.dump_model()

    counts = collections.Counter()

    def walk(node):
        if "split_feature" in node:
            counts[node["split_feature"]] += 1
            walk(node["left_child"])
            walk(node["right_child"])

    for tree in dump["tree_info"]:
        walk(tree["tree_structure"])

    rows = []
    for feat_idx, split_count in counts.items():
        m = idx_to_meta.get(feat_idx)
        if m is None:
            continue
        rows.append(
            {
                "fold_id": fold_id,
                "seed": seed,
                "feature_index": feat_idx,
                "feature_name": m.feature_name,
                "group": m.group,
                "source_name": m.source_name,
                "source_family": m.source_family,
                "split_count": split_count,
            }
        )

    return pd.DataFrame(rows)


def summarize_fold_feature_usage(split_usage_df: pd.DataFrame) -> pd.DataFrame:
    if split_usage_df.empty:
        return pd.DataFrame(
            columns=["fold_id", "seed", "group", "used_feature_count", "split_count"]
        )
    grouped = (
        split_usage_df.groupby(["fold_id", "seed", "group"], as_index=False)
        .agg(
            used_feature_count=("feature_name", "nunique"),
            split_count=("split_count", "sum"),
        )
        .sort_values(["fold_id", "seed", "group"])
    )
    return grouped


def collect_extracted_rule_feature_usage(
    rules: List[ExtractedRule], metadata: List[FeatureMetadata]
) -> pd.DataFrame:
    """Collect feature usage from extracted rules."""
    idx_to_meta = {m.feature_index: m for m in metadata}
    rows = []

    for r in rules:
        used = set()
        for c in r.conditions:
            if c.feature_index in used:
                continue
            used.add(c.feature_index)
            m = idx_to_meta[c.feature_index]
            rows.append(
                {
                    "canonical_key": r.canonical_key,
                    "rule_id": r.rule_id,
                    "fold_id": r.fold_id,
                    "seed": r.seed,
                    "feature_index": c.feature_index,
                    "feature_name": m.feature_name,
                    "group": m.group,
                    "source_name": m.source_name,
                    "source_family": m.source_family,
                    "normalized_value": c.normalized_value,
                }
            )

    return pd.DataFrame(rows)


def collect_registry_feature_usage(
    registry: pd.DataFrame, metadata: List[FeatureMetadata]
) -> pd.DataFrame:
    """Collect feature usage from final registry."""
    name_to_meta = {m.feature_name: m for m in metadata}
    rows = []

    for _, row in registry.iterrows():
        canonical_key = row["canonical_key"]
        for feature_name in extract_feature_names_from_key(canonical_key):
            m = name_to_meta.get(feature_name)
            if m is None:
                continue
            rows.append(
                {
                    "canonical_key": canonical_key,
                    "feature_index": m.feature_index,
                    "feature_name": m.feature_name,
                    "group": m.group,
                    "source_name": m.source_name,
                    "source_family": m.source_family,
                }
            )

    return pd.DataFrame(rows)


def build_portfolio_diversity_report(
    registry: pd.DataFrame,
    resolver: Union[CanonicalRuleMaskResolver, DictionaryMaskResolver],
    data: pd.DataFrame,
    fwd_ret: np.ndarray,
    X_for_ridge: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if registry.empty:
        return pd.DataFrame(rows)

    mask_map = {
        key: resolver.get_mask(key) for key in registry["canonical_key"].tolist()
    }

    # Pre-compute Ridge Signatures for orthogonality metrics
    signatures = {}
    X_raw = getattr(resolver, "X", None) if X_for_ridge is None else X_for_ridge
    if X_raw is not None:
        X_std = np.zeros_like(X_raw)
        for i in range(X_raw.shape[1]):
            col = X_raw[:, i]
            valid = np.isfinite(col)
            if np.sum(valid) > 0:
                mean = np.mean(col[valid])
                std = np.std(col[valid])
                if std == 0:
                    std = 1.0
                X_std[valid, i] = (col[valid] - mean) / std

        # Simple fold definition for signature computation (not strictly needed for just coefficients, but the method requires it)
        # We can just pass an empty list if we only care about the full-data coefficients, but let's just make one mock fold or use fwd_ret directly.
        # Actually, the method is _compute_ridge_signature which requires an EconomicRuleConsolidator instance.
        # We can do a simpler Ridge fit here.
        import scipy.stats
        from sklearn.linear_model import Ridge

        for key, mask in mask_map.items():
            valid_idx = np.where(mask & np.isfinite(fwd_ret))[0]
            if len(valid_idx) >= 20:
                side = resolver.get_rule_side(key)
                target_ret = -fwd_ret if side == "short" else fwd_ret
                X_valid = X_std[valid_idx]
                y_valid = target_ret[valid_idx]
                hi = np.nanquantile(y_valid, 0.98) if len(y_valid) > 0 else 1.0
                lo = np.nanquantile(y_valid, 0.02) if len(y_valid) > 0 else -1.0
                y_valid_clipped = np.clip(y_valid, lo, hi)
                model = Ridge(solver="auto")
                model.fit(X_valid, y_valid_clipped)
                signatures[key] = model.coef_
    activation_counts = {key: int(mask.sum()) for key, mask in mask_map.items()}
    total_activations = sum(activation_counts.values())
    if total_activations > 0:
        shares = (
            np.array(list(activation_counts.values()), dtype=np.float64)
            / total_activations
        )
        effective_rules = 1.0 / np.sum(shares**2)
        top_rule_share = float(np.max(shares))
    else:
        effective_rules = 0.0
        top_rule_share = 0.0

    rows.append(
        {"category": "summary", "metric": "top_rule_share", "value": top_rule_share}
    )
    rows.append(
        {
            "category": "summary",
            "metric": "effective_independent_rules",
            "value": effective_rules,
        }
    )

    keys = registry["canonical_key"].tolist()
    for key_a, key_b in itertools.combinations(keys, 2):
        mask_a = mask_map[key_a]
        mask_b = mask_map[key_b]
        union = np.sum(mask_a | mask_b)
        jaccard = float(np.sum(mask_a & mask_b) / union) if union > 0 else 0.0

        side_a = resolver.get_rule_side(key_a)
        side_b = resolver.get_rule_side(key_b)
        target_ret_a = -fwd_ret if side_a == "short" else fwd_ret
        target_ret_b = -fwd_ret if side_b == "short" else fwd_ret

        ret_a = np.where(mask_a, target_ret_a, np.nan)
        ret_b = np.where(mask_b, target_ret_b, np.nan)
        valid = np.isfinite(ret_a) & np.isfinite(ret_b)
        ret_corr = np.nan
        if np.sum(valid) >= 3:
            ret_corr = float(np.corrcoef(ret_a[valid], ret_b[valid])[0, 1])
        rows.append(
            {
                "category": "pairwise",
                "metric": "jaccard_overlap",
                "item_a": key_a,
                "item_b": key_b,
                "value": jaccard,
            }
        )
        rows.append(
            {
                "category": "pairwise",
                "metric": "return_correlation",
                "item_a": key_a,
                "item_b": key_b,
                "value": ret_corr,
            }
        )
        if key_a in signatures and key_b in signatures:
            beta_a = signatures[key_a]
            beta_b = signatures[key_b]
            coef_sim = _cosine_similarity(beta_a, beta_b)

            rank_corr = _safe_spearman(
                scipy.stats.rankdata(np.abs(beta_a)),
                scipy.stats.rankdata(np.abs(beta_b)),
            )
            if np.isnan(rank_corr):
                rank_corr = 0.0

            rows.append(
                {
                    "category": "pairwise",
                    "metric": "ridge_coefficient_cosine_similarity",
                    "item_a": key_a,
                    "item_b": key_b,
                    "value": coef_sim,
                }
            )
            rows.append(
                {
                    "category": "pairwise",
                    "metric": "ridge_coefficient_spearman_rank_corr",
                    "item_a": key_a,
                    "item_b": key_b,
                    "value": rank_corr,
                }
            )

    if "symbol" in data.columns:
        symbol_series = data["symbol"].astype(str)
        for key, mask in mask_map.items():
            counts = symbol_series[mask].value_counts(normalize=True)
            for symbol, share in counts.items():
                rows.append(
                    {
                        "category": "coverage_symbol",
                        "metric": key,
                        "item_a": symbol,
                        "value": float(share),
                    }
                )

    for side, count in registry["side"].fillna("unknown").value_counts().items():
        rows.append(
            {"category": "coverage_side", "metric": side, "value": float(count)}
        )

    if "timestamp" in data.columns:
        hours = pd.to_datetime(data["timestamp"]).dt.hour.fillna(-1)
        for key, mask in mask_map.items():
            counts = hours[mask].value_counts(normalize=True)
            for hour, share in counts.items():
                rows.append(
                    {
                        "category": "coverage_hour",
                        "metric": key,
                        "item_a": int(hour),
                        "value": float(share),
                    }
                )

    regime_family_counts = collections.Counter()
    for key in keys:
        for feature_name in extract_feature_names_from_key(key):
            if feature_name.startswith("reg_"):
                regime_family = (
                    feature_name.split("_")[1] if "_" in feature_name else feature_name
                )
                regime_family_counts[regime_family] += 1
    for family, count in regime_family_counts.items():
        rows.append(
            {
                "category": "coverage_regime_family",
                "metric": family,
                "value": float(count),
            }
        )

    return pd.DataFrame(rows)


def apply_label_step_sliceplanner_filter(
    data: pd.DataFrame,
    feature_dict: Dict[str, np.ndarray],
    fwd_ret: np.ndarray,
    cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], np.ndarray, Dict[str, Any]]:
    events = pd.DataFrame(
        {
            "event_id": np.arange(len(data), dtype=np.int64),
            "symbol": data["symbol"].astype(object).to_numpy(copy=False),
            "t0": pd.to_datetime(data["timestamp"], utc=True, errors="coerce"),
            "t1": pd.to_datetime(data["timestamp"], utc=True, errors="coerce"),
        }
    )

    planner_cfg = build_mining_sliceplanner_config(cfg)
    bundle = SlicePlanner(planner_cfg).build(events)

    train_indices: set[int] = set()
    for plan in bundle["consumer_plans"].get("regime_search", []):
        if plan.tag in {"fit_inner", "fit_outer", "predict_inner"}:
            train_indices.update(np.asarray(plan.fit_idx, dtype=np.int64).tolist())

    if not train_indices:
        return (
            data,
            feature_dict,
            fwd_ret,
            {
                "sliceplanner_applied": False,
                "reason": "no_training_indices",
                "rows_before": int(len(data)),
                "rows_after": int(len(data)),
                "symbols_before": int(data["symbol"].nunique()),
                "symbols_after": int(data["symbol"].nunique()),
            },
        )

    keep_idx = np.array(sorted(train_indices), dtype=np.int64)
    filtered_data = data.iloc[keep_idx].reset_index(drop=True)
    filtered_features = {
        name: np.asarray(values)[keep_idx] for name, values in feature_dict.items()
    }
    filtered_fwd_ret = np.asarray(fwd_ret)[keep_idx]

    metadata = {
        "sliceplanner_applied": True,
        "preset": planner_cfg.preset.preset_name,
        "rows_before": int(len(data)),
        "rows_after": int(len(filtered_data)),
        "symbols_before": int(data["symbol"].nunique()),
        "symbols_after": int(filtered_data["symbol"].nunique()),
        "row_fraction_kept": float(len(filtered_data) / max(len(data), 1)),
    }
    return filtered_data, filtered_features, filtered_fwd_ret, metadata


def build_mining_sliceplanner_config(
    cfg: Optional[Dict[str, Any]] = None
) -> SlicePlannerConfig:
    planner_cfg = SlicePlannerConfig.fast_defaults(schema=EventSchema())
    cfg = cfg or {}
    outer_n_folds = cfg.get("sliceplanner_outer_n_folds")
    if outer_n_folds is not None:
        planner_cfg = replace(
            planner_cfg,
            preset=replace(
                planner_cfg.preset,
                outer=replace(planner_cfg.preset.outer, n_folds=int(outer_n_folds)),
            ),
        )
    return planner_cfg


def estimate_pretrim_start_ts(
    end_ts: pd.Timestamp,
    cfg: Dict[str, Any],
) -> pd.Timestamp:
    planner_cfg = build_mining_sliceplanner_config(cfg)
    outer = planner_cfg.preset.outer
    outer_folds = int(outer.n_folds or cfg.get("sliceplanner_outer_n_folds", 8) or 8)
    warmup_days = int(cfg.get("sliceplanner_warmup_days", 90))
    total_span = (
        outer.train_span
        + (outer.valid_span or pd.Timedelta(0))
        + outer.test_span
        + outer.step_span * max(outer_folds - 1, 0)
        + pd.Timedelta(days=warmup_days)
    )
    return end_ts - total_span


def build_label_step_sliceplanner_keep_idx(
    timestamps: pd.Index,
    symbols: pd.Index,
    cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    ts_arr = np.repeat(timestamps.to_numpy(), len(symbols))
    events = pd.DataFrame(
        {
            "event_id": np.arange(len(ts_arr), dtype=np.int64),
            "symbol": np.tile(symbols.to_numpy(dtype=object), len(timestamps)),
            "t0": pd.to_datetime(ts_arr, utc=True, errors="coerce"),
            "t1": pd.to_datetime(ts_arr, utc=True, errors="coerce"),
        }
    )

    planner_cfg = build_mining_sliceplanner_config(cfg)
    bundle = SlicePlanner(planner_cfg).build(events)
    clean_events = bundle["events"]

    train_event_ids: set[int] = set()
    for plan in bundle["consumer_plans"].get("regime_search", []):
        if plan.tag in {"fit_inner", "fit_outer", "predict_inner"}:
            fit_idx = np.asarray(plan.fit_idx, dtype=np.int64)
            if fit_idx.size == 0:
                continue
            train_event_ids.update(
                clean_events.iloc[fit_idx]["event_id"].to_numpy(dtype=np.int64).tolist()
            )

    total_rows = int(len(events))
    symbols_before = int(len(symbols))
    if not train_event_ids:
        keep_idx = np.arange(total_rows, dtype=np.int64)
        metadata = {
            "sliceplanner_applied": False,
            "reason": "no_training_indices",
            "preset": planner_cfg.preset.preset_name,
            "rows_before": total_rows,
            "rows_after": total_rows,
            "symbols_before": symbols_before,
            "symbols_after": symbols_before,
            "row_fraction_kept": 1.0,
        }
        return keep_idx, metadata

    keep_idx = np.array(sorted(train_event_ids), dtype=np.int64)
    kept_symbol_count = int(np.unique(keep_idx % max(len(symbols), 1)).size)
    metadata = {
        "sliceplanner_applied": True,
        "preset": planner_cfg.preset.preset_name,
        "rows_before": total_rows,
        "rows_after": int(len(keep_idx)),
        "symbols_before": symbols_before,
        "symbols_after": kept_symbol_count,
        "row_fraction_kept": float(len(keep_idx) / max(total_rows, 1)),
    }
    return keep_idx, metadata


def _extract_selected_wide_values(
    df: pd.DataFrame,
    common_idx: pd.Index,
    common_syms: pd.Index,
    time_idx: np.ndarray,
    sym_idx: np.ndarray,
    dtype: Optional[np.dtype] = np.float32,
) -> np.ndarray:
    aligned = df.reindex(index=common_idx, columns=common_syms)
    values = aligned.to_numpy()
    extracted = values[time_idx, sym_idx]
    if dtype is None:
        return extracted
    return extracted.astype(dtype, copy=False)


def filter_complete_feature_rows(
    data: pd.DataFrame,
    feature_dict: Dict[str, np.ndarray],
    fwd_ret: np.ndarray,
    fwd_ret_norm: np.ndarray,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Retain only symbol-timestamp rows where every extracted feature value is finite.

    Each entry in ``feature_dict`` is expected to already be reduced to the selected
    event rows via ``feat_values[time_idx, compact_sym_idx]``. Missing values for
    other symbols elsewhere in the universe must not affect a retained row for the
    current symbol/timestamp.
    """
    n_rows = len(data)
    if n_rows == 0 or not feature_dict:
        return (
            data,
            feature_dict,
            fwd_ret,
            fwd_ret_norm,
            {
                "rows_before": int(n_rows),
                "rows_after": int(n_rows),
                "dropped_rows": 0,
                "drop_fraction": 0.0,
                "worst_features": [],
                "worst_symbols": [],
            },
        )

    keep_mask = np.ones(n_rows, dtype=bool)
    missing_counts: List[Tuple[str, int]] = []
    for name, values in feature_dict.items():
        arr = np.asarray(values)
        finite_mask = np.isfinite(arr)
        if arr.ndim > 1:
            finite_mask = np.all(finite_mask, axis=1)
        keep_mask &= finite_mask
        missing_counts.append((str(name), int((~finite_mask).sum())))

    filtered_data = data.loc[keep_mask].reset_index(drop=True)
    filtered_features = {
        name: np.asarray(values)[keep_mask] for name, values in feature_dict.items()
    }
    filtered_fwd_ret = np.asarray(fwd_ret)[keep_mask]
    filtered_fwd_ret_norm = np.asarray(fwd_ret_norm)[keep_mask]
    missing_counts.sort(key=lambda item: item[1], reverse=True)
    dropped_rows = int((~keep_mask).sum())
    symbol_drop_counts: List[Tuple[str, int]] = []
    if "symbol" in data.columns:
        dropped_by_symbol = (
            data.loc[~keep_mask, "symbol"]
            .astype(str)
            .value_counts()
            .sort_values(ascending=False)
        )
        symbol_drop_counts = [
            (str(symbol), int(count))
            for symbol, count in dropped_by_symbol.head(10).items()
        ]
    meta = {
        "rows_before": int(n_rows),
        "rows_after": int(len(filtered_data)),
        "dropped_rows": dropped_rows,
        "drop_fraction": float(dropped_rows / max(n_rows, 1)),
        "worst_features": missing_counts[:10],
        "worst_symbols": symbol_drop_counts,
    }
    return (
        filtered_data,
        filtered_features,
        filtered_fwd_ret,
        filtered_fwd_ret_norm,
        meta,
    )


def compute_atr_wide(
    high_wide: np.ndarray,
    low_wide: np.ndarray,
    close_wide: np.ndarray,
    atr_period: int = 14,
) -> np.ndarray:
    n_ts, n_syms = high_wide.shape
    atr_wide = np.zeros((n_ts, n_syms), dtype=np.float32)

    for sym_idx in range(n_syms):
        high_sym = high_wide[:, sym_idx]
        low_sym = low_wide[:, sym_idx]
        close_sym = close_wide[:, sym_idx]

        tr = np.zeros(n_ts, dtype=np.float32)
        if n_ts > 1:
            tr[1:] = np.maximum(
                high_sym[1:] - low_sym[1:],
                np.maximum(
                    np.abs(high_sym[1:] - close_sym[:-1]),
                    np.abs(low_sym[1:] - close_sym[:-1]),
                ),
            )

        if n_ts > atr_period:
            atr_sym = np.zeros(n_ts, dtype=np.float32)
            atr_sym[:atr_period] = float(np.mean(tr[:atr_period]))
            for i in range(atr_period, n_ts):
                atr_sym[i] = (atr_sym[i - 1] * (atr_period - 1) + tr[i]) / atr_period
        else:
            fallback = float(np.mean(tr[1:])) if n_ts > 1 else 0.001
            atr_sym = np.full(n_ts, fallback, dtype=np.float32)

        atr_wide[:, sym_idx] = atr_sym

    return atr_wide


def summarize_feature_usage(
    df: pd.DataFrame, output_path: Path, groupby_cols: List[str]
) -> None:
    """Summarize feature usage by specified columns."""
    if df.empty:
        pd.DataFrame(columns=groupby_cols + ["usage_count"]).to_csv(
            output_path, index=False
        )
        return

    summary = df.groupby(groupby_cols).size().reset_index(name="usage_count")
    summary = summary.sort_values("usage_count", ascending=False)
    summary.to_csv(output_path, index=False)


def export_coverage_sanity_report(
    metadata: List[FeatureMetadata],
    split_usage_all: pd.DataFrame,
    rule_usage_df: pd.DataFrame,
    final_usage_df: pd.DataFrame,
    output_dir: Path,
) -> pd.DataFrame:
    """Export a comprehensive coverage sanity report."""
    all_features = pd.DataFrame(
        [
            {
                "feature_name": m.feature_name,
                "group": m.group,
                "source_name": m.source_name,
                "source_family": m.source_family,
            }
            for m in metadata
        ]
    )

    split_counts = (
        split_usage_all.groupby("feature_name")["split_count"]
        .sum()
        .reset_index()
        .rename(columns={"split_count": "model_split_count"})
        if not split_usage_all.empty
        else pd.DataFrame(columns=["feature_name", "model_split_count"])
    )

    extracted_counts = (
        rule_usage_df.groupby("feature_name")
        .size()
        .reset_index(name="extracted_rule_count")
        if not rule_usage_df.empty
        else pd.DataFrame(columns=["feature_name", "extracted_rule_count"])
    )

    final_counts = (
        final_usage_df.groupby("feature_name")
        .size()
        .reset_index(name="final_registry_count")
        if not final_usage_df.empty
        else pd.DataFrame(columns=["feature_name", "final_registry_count"])
    )

    report = (
        all_features.merge(split_counts, on="feature_name", how="left")
        .merge(extracted_counts, on="feature_name", how="left")
        .merge(final_counts, on="feature_name", how="left")
        .fillna(0)
    )

    for c in ["model_split_count", "extracted_rule_count", "final_registry_count"]:
        report[c] = report[c].astype(int)

    report["used_in_model"] = report["model_split_count"] > 0
    report["used_in_extracted_rules"] = report["extracted_rule_count"] > 0
    report["used_in_final_registry"] = report["final_registry_count"] > 0

    report.to_csv(output_dir / "feature_coverage_sanity_report.csv", index=False)

    summary = (
        report.groupby("group")[
            ["used_in_model", "used_in_extracted_rules", "used_in_final_registry"]
        ]
        .sum()
        .reset_index()
    )
    summary.to_csv(
        output_dir / "final_registry_feature_usage_by_group.csv", index=False
    )

    return report


# =============================================================================
# MAIN ORCHESTRATION
# =============================================================================


def apply_cfg_preset(cfg: Dict[str, Any]) -> Dict[str, Any]:
    preset = str(cfg.get("preset", "exploration")).lower()
    out = dict(cfg)
    defaults = {
        "exploration": {
            "min_feature_support": 2,
            "min_support_count_validation": 5,
            "min_tree_discoveries": 1,
            "min_presence_freq": 0.33,
            "min_sign_consistency": 0.65,
            "prune_base_hurdle": 0.00005,
            "prune_support_exp": 0.5,
            "min_context_support_pct": 0.005,
            "min_context_presence_freq": 0.33,
            "min_context_sign_consistency": 0.65,
            "min_context_mean_ret": 0.0,
            "cv_min_train_frac": 0.5,
            "cv_embargo": 0,
            "max_support_pct": 0.25,
            "stage_a_directional": True,
            "stage_a_relax_positive_groups": True,
        },
        "production": {
            "min_feature_support": 5,
            "min_support_count_validation": 10,
            "min_tree_discoveries": 2,
            "min_presence_freq": 0.4,
            "min_sign_consistency": 0.75,
            "prune_base_hurdle": 0.00010,
            "prune_support_exp": 0.5,
            "min_context_support_pct": 0.01,
            "min_context_presence_freq": 0.5,
            "min_context_sign_consistency": 0.80,
            "min_context_mean_ret": 0.0,
            "cv_min_train_frac": 0.5,
            "cv_embargo": 0,
            "max_support_pct": 0.20,
            "stage_a_directional": True,
            "stage_a_relax_positive_groups": True,
        },
    }
    if preset not in defaults:
        raise ValueError(f"Unknown preset {preset}")
    for key, value in defaults[preset].items():
        out.setdefault(key, value)
    out["preset"] = preset
    out.setdefault(
        "prune_complexity_bonus_map",
        {"1": 0.0, "2": 0.15, "3": 0.30, "4": 0.10, "5": 0.10, "6": 0.10},
    )
    out.setdefault("n_folds", 5)
    out.setdefault("consolidation_top_n", 100)
    out.setdefault("jaccard_round1", 0.95)
    out.setdefault("jaccard_round2", 0.90)
    out.setdefault("jaccard_round3", 0.75)
    out.setdefault("support_gain_factor", 1.05)
    out.setdefault("max_failed_pair_checks", 250)
    out.setdefault("max_total_pair_evals", 1000)
    out.setdefault("max_consolidation_rounds", 3)
    out.setdefault("min_jaccard_stop_threshold", 0.60)
    return out


def run_lgbm_mask_generation(
    data: pd.DataFrame,
    feature_dict: Dict[str, np.ndarray],
    fwd_ret: np.ndarray,
    fwd_ret_norm: np.ndarray,
    cfg: Dict[str, Any],
):
    cfg = apply_cfg_preset(cfg)
    output_dir = Path(cfg.get("output_dir", "./lgbm_outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)
    fp = FeatureProcessor()
    X, metadata, audits = fp.prepare_features(
        feature_dict, data["timestamp"].to_numpy(), data["symbol"].to_numpy(), cfg
    )
    for k, v in audits.items():
        if not v.empty:
            v.to_csv(output_dir / f"{k}.csv", index=False)
    export_feature_group_summary(metadata, output_dir)
    result = run_mining_stage(
        data=data,
        fwd_ret=fwd_ret,
        fwd_ret_norm=fwd_ret_norm,
        X=X,
        metadata=metadata,
        cfg=cfg,
        output_dir=output_dir,
        stage_name="single_stage",
        allowed_group_pairs=(
            ("trigger", "location"),
            ("trigger", "regime"),
            ("location", "regime"),
        ),
        slot_order=("trigger", "location", "regime"),
        folds=build_walk_forward_folds(
            n_samples=len(data),
            n_folds=int(cfg.get("n_folds", 5)),
            min_train_frac=float(cfg.get("cv_min_train_frac", 0.5)),
            embargo=int(cfg.get("cv_embargo", 0)),
        ),
        mask_resolver=CanonicalRuleMaskResolver(X, metadata),
        pipeline_stage_name="single_stage",
    )
    return result["accepted_registry"]


def _flatten_wide_frame(
    df: pd.DataFrame, common_idx: pd.Index, common_syms: pd.Index
) -> np.ndarray:
    return df.reindex(index=common_idx, columns=common_syms).to_numpy().flatten()


def list_preload_training_symbols(
    store: PartitionedOHLCVStore,
    cfg: Dict[str, Any],
    max_symbols: int = 0,
) -> List[str]:
    """Return the same training universe used by the label step, before heavy data loading."""
    train_symbols = get_training_universe(None, cfg, store, ts_sig=None)
    if max_symbols > 0:
        return list(train_symbols[:max_symbols])
    return list(train_symbols)


# =============================================================================
# TARGET QUALITY METRICS
# =============================================================================


def compute_target_quality_metrics(
    target: np.ndarray,
    target_name: str,
    horizon: int,
    predictions: Optional[np.ndarray] = None,
    fold_ics: Optional[List[float]] = None,
    mask: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Compute comprehensive target quality metrics.

    Args:
        target: Target values array
        target_name: Name of the target for logging
        horizon: Horizon identifier for logging
        predictions: Optional model predictions for IC computation
        fold_ics: Optional list of fold-level ICs
        mask: Optional mask for within-mask metrics

    Returns:
        Dict with target quality metrics including:
        - target_name, horizon
        - target_mean, target_std, target_p05, p25, p50, p75, p95
        - target_nonzero_fraction
        - target_top_decile_support
        - target_autocorr_1
        - mean_oos_ic, p25_oos_ic, p50_oos_ic, p75_oos_ic, positive_ic_fraction
        - entropy_reduction (if predictions and mask available)
        - within_mask_ic, delta_within_mask_ic (if predictions and mask available)
    """
    valid = np.isfinite(target)
    target_valid = target[valid]

    result: Dict[str, Any] = {
        "target_name": target_name,
        "horizon": horizon,
    }

    if len(target_valid) < 10:
        result.update(
            {
                "target_mean": np.nan,
                "target_std": np.nan,
                "target_p05": np.nan,
                "target_p25": np.nan,
                "target_p50": np.nan,
                "target_p75": np.nan,
                "target_p95": np.nan,
                "target_nonzero_fraction": np.nan,
                "target_top_decile_support": np.nan,
                "target_autocorr_1": np.nan,
                "mean_oos_ic": np.nan,
                "p25_oos_ic": np.nan,
                "p50_oos_ic": np.nan,
                "p75_oos_ic": np.nan,
                "positive_ic_fraction": np.nan,
                "entropy_reduction": np.nan,
                "within_mask_ic": np.nan,
                "delta_within_mask_ic": np.nan,
            }
        )
        return result

    # Basic statistics
    result["target_mean"] = float(np.mean(target_valid))
    result["target_std"] = float(np.std(target_valid))
    result["target_p05"] = float(np.percentile(target_valid, 5))
    result["target_p25"] = float(np.percentile(target_valid, 25))
    result["target_p50"] = float(np.percentile(target_valid, 50))
    result["target_p75"] = float(np.percentile(target_valid, 75))
    result["target_p95"] = float(np.percentile(target_valid, 95))

    # Non-zero fraction
    nonzero_count = np.sum(target_valid != 0)
    result["target_nonzero_fraction"] = float(nonzero_count / len(target_valid))

    # Top decile support (fraction of values in top 10%)
    top_decile_threshold = np.percentile(np.abs(target_valid), 90)
    top_decile_count = np.sum(np.abs(target_valid) >= top_decile_threshold)
    result["target_top_decile_support"] = float(top_decile_count / len(target_valid))

    # Autocorrelation at lag 1
    if len(target_valid) > 10:
        target_centered = target_valid - np.mean(target_valid)
        autocorr_1 = np.corrcoef(target_centered[:-1], target_centered[1:])[0, 1]
        result["target_autocorr_1"] = float(autocorr_1) if np.isfinite(autocorr_1) else np.nan
    else:
        result["target_autocorr_1"] = np.nan

    # OOS IC metrics from fold ICs
    if fold_ics is not None and len(fold_ics) > 0:
        valid_ics = [ic for ic in fold_ics if np.isfinite(ic)]
        if valid_ics:
            result["mean_oos_ic"] = float(np.mean(valid_ics))
            result["p25_oos_ic"] = float(np.percentile(valid_ics, 25))
            result["p50_oos_ic"] = float(np.percentile(valid_ics, 50))
            result["p75_oos_ic"] = float(np.percentile(valid_ics, 75))
            result["positive_ic_fraction"] = float(sum(1 for ic in valid_ics if ic > 0) / len(valid_ics))
        else:
            result["mean_oos_ic"] = np.nan
            result["p25_oos_ic"] = np.nan
            result["p50_oos_ic"] = np.nan
            result["p75_oos_ic"] = np.nan
            result["positive_ic_fraction"] = np.nan
    else:
        result["mean_oos_ic"] = np.nan
        result["p25_oos_ic"] = np.nan
        result["p50_oos_ic"] = np.nan
        result["p75_oos_ic"] = np.nan
        result["positive_ic_fraction"] = np.nan

    # Entropy reduction and within-mask IC (if predictions and mask available)
    if predictions is not None and mask is not None:
        # Compute entropy reduction
        n_bins = 20
        if len(target_valid) >= 100:
            hist_global, _ = np.histogram(target_valid, bins=n_bins, density=True)
            hist_global = hist_global / hist_global.sum()
            hist_global = hist_global[hist_global > 0]
            entropy_global = -np.sum(hist_global * np.log2(hist_global + 1e-9))

            mask_active = mask.astype(bool) & valid
            if mask_active.sum() >= 50:
                target_masked = target[mask_active]
                hist_masked, _ = np.histogram(target_masked, bins=n_bins, density=True)
                hist_masked = hist_masked / hist_masked.sum()
                hist_masked = hist_masked[hist_masked > 0]
                entropy_masked = -np.sum(hist_masked * np.log2(hist_masked + 1e-9))
                result["entropy_reduction"] = entropy_global - entropy_masked
            else:
                result["entropy_reduction"] = np.nan
        else:
            result["entropy_reduction"] = np.nan

        # Compute within-mask IC
        pred_valid = ~(np.isnan(predictions) | np.isnan(target))
        if pred_valid.sum() >= 10:
            global_ic = _safe_spearman(predictions[pred_valid], target[pred_valid])

            mask_active = mask.astype(bool) & pred_valid
            if mask_active.sum() >= 10:
                within_ic = _safe_spearman(predictions[mask_active], target[mask_active])
                result["within_mask_ic"] = within_ic
                result["delta_within_mask_ic"] = (
                    within_ic - global_ic if np.isfinite(within_ic) else np.nan
                )
            else:
                result["within_mask_ic"] = np.nan
                result["delta_within_mask_ic"] = np.nan
        else:
            result["within_mask_ic"] = np.nan
            result["delta_within_mask_ic"] = np.nan
    else:
        result["entropy_reduction"] = np.nan
        result["within_mask_ic"] = np.nan
        result["delta_within_mask_ic"] = np.nan

    return result


def compute_cross_target_correlation(
    targets: Dict[str, np.ndarray],
    horizon: int,
    correlation_threshold: float = 0.7,
) -> Dict[str, Any]:
    """
    Compute correlation matrix between targets at same horizon.

    Args:
        targets: Dict mapping target_name -> target_array
        horizon: Horizon identifier for logging
        correlation_threshold: Threshold for flagging high correlations

    Returns:
        Dict with:
        - correlation_matrix: Dict[(target_a, target_b)] -> correlation
        - high_correlation_pairs: List of pairs with corr > threshold
        - quality_flags: List of any issues
    """
    result: Dict[str, Any] = {
        "horizon": horizon,
        "correlation_matrix": {},
        "high_correlation_pairs": [],
        "quality_flags": [],
    }

    target_names = list(targets.keys())
    n_targets = len(target_names)

    if n_targets < 2:
        result["quality_flags"].append("insufficient_targets_for_correlation")
        return result

    # Compute pairwise correlations
    for i in range(n_targets):
        for j in range(i + 1, n_targets):
            name_a = target_names[i]
            name_b = target_names[j]
            target_a = targets[name_a]
            target_b = targets[name_b]

            # Ensure same length
            min_len = min(len(target_a), len(target_b))
            if min_len < 10:
                continue

            valid = np.isfinite(target_a[:min_len]) & np.isfinite(target_b[:min_len])
            if valid.sum() < 10:
                continue

            corr = _safe_spearman(target_a[:min_len][valid], target_b[:min_len][valid])
            result["correlation_matrix"][(name_a, name_b)] = corr

            if np.isfinite(corr) and abs(corr) > correlation_threshold:
                result["high_correlation_pairs"].append(
                    {"pair": (name_a, name_b), "correlation": corr}
                )

    # Add quality flags for high correlations
    if result["high_correlation_pairs"]:
        result["quality_flags"].append(
            f"high_correlation_count_{len(result['high_correlation_pairs'])}"
        )

    return result


def create_target_quality_summary(
    all_results: List[Dict[str, Any]],
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Create summary table of target quality across all targets and horizons.

    Args:
        all_results: Results from all (target, horizon) runs, where each dict
                     contains rule summaries with target quality metrics
        output_path: Optional path to save the summary CSV

    Returns:
        DataFrame with columns:
        - target_name
        - horizon
        - target_mean, target_std, target_p05, p25, p50, p75, p95
        - mean_oos_ic, p25_oos_ic, p50_oos_ic, p75_oos_ic
        - positive_ic_fraction
        - entropy_reduction
        - mean_delta_ic
        - rule_count
        - structurally_sound_rule_count
        - production_rule_count
        - median_support
        - cross_target_corrs (as dict or JSON string)
        - overall_target_quality_score
    """
    if not all_results:
        return pd.DataFrame()

    summary_records: List[Dict[str, Any]] = []

    # Group results by (target_name, horizon)
    grouped: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
    for result in all_results:
        target_name = result.get("source_target", "unknown")
        horizon = result.get("source_horizon", 0)
        key = (target_name, horizon)
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(result)

    for (target_name, horizon), rules in grouped.items():
        record: Dict[str, Any] = {
            "target_name": target_name,
            "horizon": horizon,
        }

        # Aggregate target statistics from rules
        target_means = [r.get("target_mean") for r in rules if np.isfinite(r.get("target_mean", np.nan))]
        target_stds = [r.get("target_std") for r in rules if np.isfinite(r.get("target_std", np.nan))]

        record["target_mean"] = float(np.mean(target_means)) if target_means else np.nan
        record["target_std"] = float(np.mean(target_stds)) if target_stds else np.nan

        # Aggregate IC metrics
        mean_oos_ics = [r.get("mean_oos_ic") for r in rules if np.isfinite(r.get("mean_oos_ic", np.nan))]
        record["mean_oos_ic"] = float(np.mean(mean_oos_ics)) if mean_oos_ics else np.nan

        p25_oos_ics = [r.get("p25_oos_ic") for r in rules if np.isfinite(r.get("p25_oos_ic", np.nan))]
        record["p25_oos_ic"] = float(np.mean(p25_oos_ics)) if p25_oos_ics else np.nan

        p50_oos_ics = [r.get("p50_oos_ic") for r in rules if np.isfinite(r.get("p50_oos_ic", np.nan))]
        record["p50_oos_ic"] = float(np.mean(p50_oos_ics)) if p50_oos_ics else np.nan

        p75_oos_ics = [r.get("p75_oos_ic") for r in rules if np.isfinite(r.get("p75_oos_ic", np.nan))]
        record["p75_oos_ic"] = float(np.mean(p75_oos_ics)) if p75_oos_ics else np.nan

        positive_ic_fractions = [
            r.get("positive_ic_fraction")
            for r in rules
            if np.isfinite(r.get("positive_ic_fraction", np.nan))
        ]
        record["positive_ic_fraction"] = (
            float(np.mean(positive_ic_fractions)) if positive_ic_fractions else np.nan
        )

        # Entropy reduction
        entropy_reductions = [
            r.get("entropy_reduction")
            for r in rules
            if np.isfinite(r.get("entropy_reduction", np.nan))
        ]
        record["entropy_reduction"] = (
            float(np.mean(entropy_reductions)) if entropy_reductions else np.nan
        )

        # Delta IC
        delta_ics = [r.get("mean_delta_ic") for r in rules if np.isfinite(r.get("mean_delta_ic", np.nan))]
        record["mean_delta_ic"] = float(np.mean(delta_ics)) if delta_ics else np.nan

        # Rule counts
        record["rule_count"] = len(rules)
        record["structurally_sound_rule_count"] = sum(
            1 for r in rules if r.get("accepted", False)
        )
        record["production_rule_count"] = sum(
            1 for r in rules
            if r.get("accepted", False) and r.get("composite_score", -np.inf) > 0
        )

        # Median support
        supports = [r.get("min_support_actual", 0) for r in rules]
        record["median_support"] = float(np.median(supports)) if supports else 0

        # Compute overall target quality score
        # Formula: weighted combination of IC metrics, entropy reduction, and rule quality
        quality_score = 0.0
        quality_components = 0

        if np.isfinite(record.get("mean_oos_ic", np.nan)):
            quality_score += record["mean_oos_ic"] * 10  # Scale up IC
            quality_components += 1

        if np.isfinite(record.get("positive_ic_fraction", np.nan)):
            quality_score += record["positive_ic_fraction"] * 0.5
            quality_components += 1

        if np.isfinite(record.get("entropy_reduction", np.nan)):
            quality_score += max(0, record["entropy_reduction"]) * 2
            quality_components += 1

        if np.isfinite(record.get("mean_delta_ic", np.nan)):
            quality_score += max(0, record["mean_delta_ic"]) * 5
            quality_components += 1

        record["overall_target_quality_score"] = (
            quality_score / max(quality_components, 1) if quality_components > 0 else np.nan
        )

        summary_records.append(record)

    summary_df = pd.DataFrame(summary_records)

    # Sort by target name and horizon
    if not summary_df.empty:
        summary_df = summary_df.sort_values(["target_name", "horizon"])

    # Save to file if path provided
    if output_path and not summary_df.empty:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(output_path, index=False)
        tprint(f"Target quality summary saved to {output_path}")

    return summary_df


if __name__ == "__main__":
    import argparse
    import glob

    from extreme_price_movements.config import CFG
    from extreme_price_movements.data_store import (
        PartitionedOHLCVStore,
        load_features_selected,
        save_features,
        to_panel,
    )

    parser = argparse.ArgumentParser(description="Full LGBM Mask Generation Run")
    parser.add_argument(
        "--data-root",
        default="/Users/remyroche/Documents/Ares/data",
        help="Data root path",
    )
    parser.add_argument("--feature-path", help="Optional feature path override")
    parser.add_argument(
        "--lookback-years",
        type=float,
        default=0.0,
        help="Years of data to load before SlicePlanner filtering; 0 means no manual limit",
    )
    parser.add_argument(
        "--max-symbols",
        type=int,
        default=0,
        help="Max symbols to load before SlicePlanner filtering; 0 means no manual limit",
    )
    parser.add_argument(
        "--output-dir", default="./production_lgbm_outputs", help="Output directory"
    )
    parser.add_argument(
        "--preset",
        choices=["exploration", "production"],
        default="production",
        help="Threshold preset",
    )
    parser.add_argument(
        "--boolean-only",
        action="store_true",
        help="Only use boolean features (triggers)",
    )
    parser.add_argument(
        "--use-dynamic-hpo",
        action="store_true",
        help="Run dynamic HPO for Stage A",
    )
    parser.add_argument(
        "--run-stage-b",
        action="store_true",
        help="Run Stage B mining and assessment (default: False)",
    )
    parser.add_argument(
        "--run-stage-a-consolidation",
        action="store_true",
        help="Run Stage A rule consolidation (default: False)",
    )
    parser.add_argument(
        "--triad-horizons",
        type=str,
        default="3,10",
        help="Comma-separated list of horizons for triad targets (default: 3,10)",
    )
    args = parser.parse_args()

    cfg = dict(CFG)
    cfg["data_root"] = args.data_root
    cfg["output_dir"] = args.output_dir
    cfg["preset"] = args.preset
    cfg["skip_stage_b"] = not args.run_stage_b
    cfg["skip_stage_a_consolidation"] = not args.run_stage_a_consolidation
    cfg.setdefault("sliceplanner_outer_n_folds", 8)
    cfg.setdefault("sliceplanner_warmup_days", 90)
    
    # Triad target configuration (always use triad targets)
    cfg["use_triad_targets"] = True
    if args.triad_horizons:
        cfg["triad_horizons"] = [int(h.strip()) for h in args.triad_horizons.split(",")]
    else:
        cfg["triad_horizons"] = TRIAD_DEFAULT_HORIZONS
    cfg["triad_target_names"] = TRIAD_DEFAULT_TARGET_NAMES
    
    cfg = apply_cfg_preset(cfg)

    tprint(
        f"LGBM Full Run: root={args.data_root} | lookback={args.lookback_years}y | symbols={args.max_symbols}"
    )

    # 1. Data Store & Symbols
    store = PartitionedOHLCVStore(
        root_dir=cfg["data_root"], timeframe=cfg.get("timeframe", "1h")
    )
    symbols = list_preload_training_symbols(store, cfg, max_symbols=args.max_symbols)
    tprint(f"Selected {len(symbols)} pre-load training-universe symbols")
    feature_dir = os.path.join(cfg["data_root"], "features")
    feature_files = sorted(glob.glob(os.path.join(feature_dir, "202[0-9]*")))
    feature_path = args.feature_path or (feature_files[-1] if feature_files else None)

    if not feature_path:
        feature_files = sorted(glob.glob("202[0-9]*"))
        feature_path = feature_files[-1] if feature_files else None

    if not feature_path:
        tprint("ERROR: No feature path found.")
        exit(1)

    ts_str = os.path.basename(feature_path)
    try:
        feature_snapshot_ts = pd.Timestamp(ts_str.replace("_", " "))
    except Exception:
        feature_snapshot_ts = pd.Timestamp.now(tz="UTC")
    if feature_snapshot_ts.tzinfo is None:
        feature_snapshot_ts = feature_snapshot_ts.tz_localize("UTC")

    start_ts = estimate_pretrim_start_ts(feature_snapshot_ts, cfg)
    if args.lookback_years and args.lookback_years > 0:
        manual_start = pd.Timestamp.now(tz="UTC") - pd.Timedelta(
            days=int(365.25 * args.lookback_years)
        )
        start_ts = max(start_ts, manual_start)
    tprint(f"Pre-trim start_ts={start_ts} derived from planner horizon")

    # 2. Load OHLCV
    dfs_by_symbol: Dict[str, pd.DataFrame] = {}
    for s in symbols:
        try:
            df = store.load(s, start_ts=start_ts)
            if not df.empty:
                dfs_by_symbol[s] = df
        except Exception:
            continue

    if not dfs_by_symbol:
        tprint("ERROR: No data loaded.")
        exit(1)

    panel = to_panel(dfs_by_symbol)
    common_idx = panel["close"].index
    common_syms = panel["close"].columns

    # 3. Prepare planner-bounded indices before loading features
    fwd_hours = int(cfg.get("mask_opt_forward_hours", 5))
    fwd_ret_wide = (
        panel["close"].pct_change(fwd_hours, fill_method=None).shift(-fwd_hours)
    )

    common_idx = panel["close"].index
    common_syms = panel["close"].columns
    n_ts, n_syms = len(common_idx), len(common_syms)

    # TZ Normalization for alignment
    common_idx_naive = (
        common_idx.tz_localize(None) if common_idx.tz is not None else common_idx
    )

    tprint(
        f"Panel: {n_ts} timestamps x {n_syms} symbols. TZ: {common_idx.tz} -> Naive. Top syms: {list(common_syms[:3])}"
    )

    planner_filter_start = time.perf_counter()
    keep_idx, planner_filter_meta = build_label_step_sliceplanner_keep_idx(
        common_idx, common_syms, cfg=cfg
    )
    time_idx = (keep_idx // n_syms).astype(np.int32, copy=False)
    sym_idx = (keep_idx % n_syms).astype(np.int32, copy=False)
    kept_sym_positions = np.unique(sym_idx)
    kept_syms = common_syms.take(kept_sym_positions)
    compact_sym_idx = np.searchsorted(kept_sym_positions, sym_idx).astype(
        np.int32, copy=False
    )
    tprint(
        f"SlicePlanner keep-index build complete: kept_rows={len(keep_idx)} "
        f"in {time.perf_counter() - planner_filter_start:.1f}s"
    )

    # 4. Load features only for planner-surviving symbols
    ts = feature_snapshot_ts

    tprint(
        f"Loading features from {feature_path} for {len(kept_syms)} planner-surviving symbols..."
    )
    requested_feature_keys = sorted(
        set(
            list(CFG.get("FEATURE_SELECTION_KEYS", []))
            + RIDGE_FEATURE_COLS
            + list(CONTINUOUS_LOCATION_COLS)
        )
    )
    tprint(f"Requested feature keys: {len(requested_feature_keys)} (TRIGGER features disabled)")

    if args.boolean_only:
        cfg["boolean_only"] = True
        tprint("MODE: boolean_only. Continuous features will be converted to booleans via thresholding.")
    if args.use_dynamic_hpo:
        cfg["use_dynamic_hpo"] = True
    feat_dict_raw = load_features_selected(
        ts=ts,
        root_dir=os.path.dirname(os.path.dirname(feature_path)),
        feature_keys=requested_feature_keys,
        symbols=list(map(str, kept_syms)),
        start_ts=start_ts,
    )
    if feat_dict_raw is None:
        feat_dict_raw = {}

    # Check for missing features - RIDGE_FEATURE_COLS and FEATURE_SELECTION_KEYS are required,
    # but CONTINUOUS_LOCATION_COLS are optional (they'll be skipped if missing)
    required_keys = set(list(CFG.get("FEATURE_SELECTION_KEYS", [])) + RIDGE_FEATURE_COLS)
    missing_required_keys = sorted(required_keys - set(feat_dict_raw))
    if missing_required_keys:
        tprint(
            f"ERROR: Feature snapshot incomplete. "
            f"Missing {len(missing_required_keys)} required keys: {missing_required_keys[:20]}"
        )
        tprint(
            "Run run_feature_generation_step(force_full_recompute=False) first to "
            "backfill missing features."
        )
        raise RuntimeError(
            f"Cannot proceed: {len(missing_required_keys)} required features missing "
            f"from snapshot. Backfill with run_feature_generation_step first."
        )
    
    # Check for optional location features
    optional_location_keys = set(CONTINUOUS_LOCATION_COLS)
    missing_location_keys = sorted(optional_location_keys - set(feat_dict_raw))
    if missing_location_keys:
        tprint(
            f"WARNING: {len(missing_location_keys)} optional location features missing. "
            f"These will be skipped: {missing_location_keys[:10]}{'...' if len(missing_location_keys) > 10 else ''}"
        )

    # 5. Load features
    ts_arr = common_idx.to_numpy()[time_idx]
    ts_pd = pd.to_datetime(ts_arr, utc=True)
    symbol_arr = common_syms.to_numpy(dtype=object)[sym_idx]
    close_selected = _extract_selected_wide_values(
        panel["close"], common_idx, common_syms, time_idx, sym_idx
    )
    high_selected = _extract_selected_wide_values(
        panel["high"], common_idx, common_syms, time_idx, sym_idx
    )
    low_selected = _extract_selected_wide_values(
        panel["low"], common_idx, common_syms, time_idx, sym_idx
    )
    if "open" in panel:
        open_selected = _extract_selected_wide_values(
            panel["open"], common_idx, common_syms, time_idx, sym_idx
        )
    else:
        open_selected = close_selected
    stack_start = time.perf_counter()
    data_final = pd.DataFrame(
        {
            "event_id": np.arange(len(keep_idx), dtype=np.int64),
            "timestamp": ts_arr,
            "symbol": symbol_arr,
            "close": close_selected,
            "high": high_selected,
            "low": low_selected,
            "t0": ts_pd.to_numpy(),
            "t1": (ts_pd + pd.Timedelta(seconds=1)).to_numpy(),
            "open": open_selected,
        }
    )
    tprint(
        f"Filtered event frame built: rows={len(data_final)} cols={len(data_final.columns)} "
        f"in {time.perf_counter() - stack_start:.1f}s"
    )

    atr_start = time.perf_counter()
    high_wide = (
        panel["high"]
        .reindex(index=common_idx, columns=common_syms)
        .to_numpy(dtype=np.float32)
    )
    low_wide = (
        panel["low"]
        .reindex(index=common_idx, columns=common_syms)
        .to_numpy(dtype=np.float32)
    )
    close_wide = (
        panel["close"]
        .reindex(index=common_idx, columns=common_syms)
        .to_numpy(dtype=np.float32)
    )
    atr_wide = compute_atr_wide(high_wide, low_wide, close_wide, atr_period=14)
    # Compute ATR as percentage of close price
    atr_pct_matrix = np.where(close_wide > 1e-9, atr_wide / close_wide, 0.0).astype(
        np.float32
    )
    data_final["atr"] = atr_wide[time_idx, sym_idx]
    tprint(
        f"ATR computed in wide form and extracted in {time.perf_counter() - atr_start:.1f}s"
    )

    fwd_ret_start = time.perf_counter()
    fwd_ret_matrix = fwd_ret_wide.reindex(
        index=common_idx, columns=common_syms
    ).to_numpy(dtype=np.float32)
    target_signal = fwd_ret_matrix / np.maximum(np.sqrt(atr_pct_matrix), 1e-9)

    fwd_ret_norm_matrix = fwd_ret_matrix / np.maximum(atr_pct_matrix, 1e-9)
    fwd_ret_final = fwd_ret_matrix[time_idx, sym_idx]
    fwd_ret_norm_final = fwd_ret_norm_matrix[time_idx, sym_idx]
    tprint(
        f"Forward returns extracted for kept rows in {time.perf_counter() - fwd_ret_start:.1f}s"
    )

    feature_align_start = time.perf_counter()
    feat_final: Dict[str, np.ndarray] = {}
    feature_items = list(feat_dict_raw.items())
    feature_log_every = max(1, len(feature_items) // 10)
    alignment_issue_samples: List[str] = []
    alignment_issue_count = 0
    for feat_idx, (k, df_feat) in enumerate(feature_items, start=1):
        if isinstance(df_feat, pd.DataFrame):
            feat_df = df_feat
            if (
                isinstance(feat_df.index, pd.DatetimeIndex)
                and feat_df.index.tz is not None
            ):
                feat_df = feat_df.tz_localize(None)

            if len(feat_final) == 0:
                overlap = common_idx_naive.intersection(feat_df.index)
                tprint(
                    f"Alignment Check: overlap={len(overlap)}/{len(common_idx_naive)}"
                )
                if len(overlap) == 0:
                    tprint(f"Panel Index Sample: {common_idx_naive[:2].tolist()}")
                    tprint(f"Feat Index Sample: {feat_df.index[:2].tolist()}")

            missing_ts = common_idx_naive.difference(feat_df.index)
            missing_syms = kept_syms.difference(feat_df.columns)
            if len(missing_ts) > 0 or len(missing_syms) > 0:
                alignment_issue_count += 1
                if len(alignment_issue_samples) < 10:
                    ts_sample = [str(ts) for ts in missing_ts[:3].tolist()]
                    sym_sample = [str(sym) for sym in missing_syms[:3].tolist()]
                    alignment_issue_samples.append(
                        f"{k}: missing_ts={len(missing_ts)} sample_ts={ts_sample} "
                        f"missing_syms={len(missing_syms)} sample_syms={sym_sample}"
                    )

            feat_df_aligned = feat_df.reindex(index=common_idx_naive, columns=kept_syms)
            feat_values = feat_df_aligned.to_numpy(dtype=np.float32)
            feat_final[k] = feat_values[time_idx, compact_sym_idx]
            if feat_idx % feature_log_every == 0 or feat_idx == len(feature_items):
                tprint(
                    f"Feature extraction progress: {feat_idx}/{len(feature_items)} "
                    f"({100.0 * feat_idx / len(feature_items):.1f}%) in "
                    f"{time.perf_counter() - feature_align_start:.1f}s"
                )
    tprint(
        f"Feature extraction complete: {len(feat_final)} feature arrays "
        f"in {time.perf_counter() - feature_align_start:.1f}s"
    )
    if alignment_issue_count > 0:
        tprint(
            f"Feature alignment issues detected for {alignment_issue_count} features during "
            "timestamp/symbol reindexing."
        )
        for sample in alignment_issue_samples:
            tprint(f"Alignment issue sample: {sample}")

    (
        data_final,
        feat_final,
        fwd_ret_final,
        fwd_ret_norm_final,
        completeness_meta,
    ) = filter_complete_feature_rows(
        data_final,
        feat_final,
        fwd_ret_final,
        fwd_ret_norm_final,
    )
    tprint(
        "Feature completeness row filter: "
        f"rows_before={completeness_meta['rows_before']} "
        f"rows_after={completeness_meta['rows_after']} "
        f"dropped={completeness_meta['dropped_rows']} "
        f"drop_fraction={completeness_meta['drop_fraction']:.3f}"
    )
    if completeness_meta["worst_features"]:
        tprint(
            "Top incomplete features: "
            + ", ".join(
                f"{name}={count}"
                for name, count in completeness_meta["worst_features"][:5]
            )
        )
    if completeness_meta["worst_symbols"]:
        tprint(
            "Top symbols with dropped rows: "
            + ", ".join(
                f"{symbol}={count}"
                for symbol, count in completeness_meta["worst_symbols"][:5]
            )
        )

    tprint(
        "SlicePlanner label-step filter: "
        f"rows {planner_filter_meta['rows_before']} -> {planner_filter_meta['rows_after']} | "
        f"symbols {planner_filter_meta['symbols_before']} -> {planner_filter_meta['symbols_after']} | "
        f"applied={planner_filter_meta['sliceplanner_applied']} | "
        f"elapsed={time.perf_counter() - planner_filter_start:.1f}s"
    )

    # Check non-zero features
    non_zero_feats = 0
    for k, v in feat_final.items():
        if v.size > 0 and np.nanmax(np.abs(v)) > 0:
            non_zero_feats += 1
    tprint(
        f"Final Input: {len(data_final)} rows. {non_zero_feats}/{len(feat_final)} features have non-zero values."
    )

    Path(cfg["output_dir"]).mkdir(parents=True, exist_ok=True)
    with open(Path(cfg["output_dir"]) / "sliceplanner_filter_summary.json", "w") as f:
        json.dump(planner_filter_meta, f, indent=2, default=str)
    with open(Path(cfg["output_dir"]) / "run_config_snapshot.json", "w") as f:
        json.dump(cfg, f, indent=2, default=str)
    
    # Triad target mode is now the default and only mode
    tprint("=" * 60)
    tprint("TRIAD TARGET MODE")
    tprint("=" * 60)
    
    # Import triad target computation
    from extreme_price_movements.triad_targets import compute_triad_targets_for_horizons
    
    # Compute triad targets for all configured horizons
    horizons = cfg.get("triad_horizons", TRIAD_DEFAULT_HORIZONS)
    target_names = cfg.get("triad_target_names", TRIAD_DEFAULT_TARGET_NAMES)
    
    tprint(f"Computing triad targets for horizons: {horizons}")
    tprint(f"Target names: {target_names}")
    
    # Compute triad targets using data_final (long format)
    # compute_triad_targets_for_horizons expects a DataFrame with close, high, low, atr columns
    triad_results_by_horizon = compute_triad_targets_for_horizons(
        df=data_final,
        horizons=horizons,
        atr_col="atr",
    )
    
    # Convert to the format expected by run_two_stage_lgbm_mask_generation_triad:
    # {target_name: {horizon: target_array}}
    triad_targets: Dict[str, Dict[int, np.ndarray]] = {
        "target_eff": {},
        "target_ela": {},
        "target_vame": {},
    }
    
    for horizon, df_targets in triad_results_by_horizon.items():
        for target_base in ["target_eff", "target_ela", "target_vame"]:
            col_name = f"{target_base}_{horizon}"
            if col_name in df_targets.columns:
                triad_targets[target_base][horizon] = df_targets[col_name].to_numpy(dtype=np.float32)
                valid_count = int(np.isfinite(triad_targets[target_base][horizon]).sum())
                tprint(f"  {col_name}: {valid_count}/{len(triad_targets[target_base][horizon])} valid values")
    
    # Run triad-target mask generation
    run_two_stage_lgbm_mask_generation_triad(
        data_final, feat_final, triad_targets, cfg
    )
