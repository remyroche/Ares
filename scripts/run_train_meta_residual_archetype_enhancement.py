#!/usr/bin/env python3
"""Build and evaluate a residual-archetype alternative to the current meta model.

The current base model and current meta OOS predictions are frozen references.
Feature selection is fitted on data through February 2026 with March as its
held-out validation month. Alternative meta models reuse the current meta
hyperparameters and are evaluated on configurable expanding-window OOS folds.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import duckdb
import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.config import CFG  # noqa: E402
from extreme_price_movements.data_store import read_symbol_features  # noqa: E402
from extreme_price_movements.features_gmm_ae import (  # noqa: E402
    save_ae_gmm_state_artifact,
)
from extreme_price_movements.local_economic_aegmm import (  # noqa: E402
    META_CROSS_SECTIONAL_GEOMETRY_FEATURES,
    META_MARKET_STATE_FEATURES,
    EconomicAEGMMBlock,
    LocalEconomicAEGMM,
    LocalEconomicAEGMMConfig,
    LocalEconomicAEGMMModelBundle,
    default_meta_economic_aegmm_blocks,
    local_economic_aegmm_feature_names,
)
from extreme_price_movements.meta_cross_sectional_geometry import (  # noqa: E402
    DEFAULT_RELATIVE_FEATURES,
    materialize_cross_sectional_geometry,
)
from extreme_price_movements.meta_historical_rank import (  # noqa: E402
    HistoricalScoreRankReference,
)
from extreme_price_movements.meta_residual_archetypes import (  # noqa: E402
    OUTCOME_COLUMNS,
    ResidualArchetypeConfig,
    ResidualArchetypeRecognizer,
    add_reference_surprise_targets,
    residual_ae_gmm_feature_names,
    strip_outcomes_for_oos,
)
from extreme_price_movements.regime_ev_calibration import (  # noqa: E402
    apply_regime_ev_calibration,
    default_regime_ev_calibration_artifact,
    load_regime_ev_calibration,
)
from extreme_price_movements.regime_ev_calibration import (
    required_feature_columns as regime_required_feature_columns,
)
from scripts.run_s52_train_meta_regime_handoff_smoke import (  # noqa: E402
    HIT_SURPRISE_NUMERIC_FEATURES,
    META_POST_SELECTION_OOD_FEATURE_NAMES,
    _add_fold_base_prior_features,
    _add_fold_hit_surprise_features,
    _add_fold_reliability_features,
    _archetype_key,
    _base_soft_label_target,
    _fit_base_soft_label_model,
    _predict,
    _select_features_by_lgbm_pipeline,
)

DEFAULT_RUN_ROOT = Path(
    "data_perp/reports/"
    "s59_h5_2025start_monthly_v4_base_configfull_mdafs120_hpo150_"
    "largestfold_oos15_ae3000_nocrossfit_k34567_payload300k_20260706"
)
DEFAULT_HANDOFF = (
    DEFAULT_RUN_ROOT
    / "s52_trailing_regime_meta_handoff_top30_allsafe_aegmm_fixedtargets_oos15_20260706"
    / "s52_trailing_regime_scored_ledger.parquet"
)
DEFAULT_REFERENCE_DIR = (
    DEFAULT_RUN_ROOT
    / "train_meta_regime_handoff_singlehead_base_soft_lgbmpipeline_auto_hpo150_"
    "oos15_top30_hpo45k_20260706_v5" / "best_full_oos_fixedfs_streamed_v1"
)
DEFAULT_FEATURE_ROOT = Path("data_perp/features/20260711_070000")
DEFAULT_OUT_DIR = Path(
    "data_perp/reports/train_meta_residual_archetype_enhancement_20260711_v1"
)
DEFAULT_EXTERNAL_RESIDUAL_FEATURES = Path(
    "data_perp/reports/meta_residual_archetype_discovery_early2025_july_20260712_v1/"
    "oos_residual_state_predictions.parquet"
)
DEFAULT_THRESHOLD_BASIS_POLICY = Path(
    "data_perp/artifacts/s59_s52_frozen_native_shadow_20260709/"
    "policy_params/threshold_basis_policy.json"
)

KEY_COLUMNS = ("__ts__", "__symbol__", "side_name", "archetype_policy_key")
EVAL_MONTHS = ("2026-04", "2026-05", "2026-06")
MODEL_ARMS = (
    "baseline_retrained",
    "negative_residual_full_context",
    "lifecycle_only",
    "residual_archetypes",
    "residual_archetypes_ae_gmm",
    "local_aegmm_market",
    "local_aegmm_geometry",
    "local_aegmm_joint",
    "local_aegmm_all_three",
    "residual_states_v2_probabilities",
    "residual_states_v2_priors",
    "residual_states_v2_priors_market",
    "residual_states_v2_probabilities_priors",
    "residual_states_v2_probabilities_priors_market",
    "residual_states_v2_archetype_market_failure_posteriors_only",
    "residual_states_v2_archetype_market_failure_posteriors",
    "residual_states_v2_full_aegmm",
    "residual_states_v2_full_context",
    "residual_states_v2_distilled",
    "residual_states_v2_distilled_local_interactions",
    "residual_states_v2_distilled_priors",
    "residual_states_v2_priors_hitrate_context",
    "residual_states_v2_priors_outcome_context",
    "residual_states_v2_priors_outcome_context_weighted",
    "residual_states_v2_priors_outcome_context_identity",
    "residual_states_v2_priors_outcome_context_identity_weighted",
    "residual_states_v2_priors_outcome_context_interactions",
    "residual_states_v2_priors_failure_pressure_context",
    "residual_states_v2_priors_failure_pressure_identity_context",
    "residual_states_v2_distilled_priors_hitrate_context",
    "residual_states_v2_distilled_weighted_mild",
    "residual_states_v2_distilled_weighted_contrast",
    "residual_states_v2_distilled_weighted_short_adverse",
    "residual_states_v2_priors_local_models",
    "residual_states_v2_priors_local_models_shrunk",
)
RESIDUAL_STATES_V2_ARMS = frozenset(
    {
        "negative_residual_full_context",
        "residual_states_v2_probabilities",
        "residual_states_v2_priors",
        "residual_states_v2_priors_market",
        "residual_states_v2_probabilities_priors",
        "residual_states_v2_probabilities_priors_market",
        "residual_states_v2_archetype_market_failure_posteriors_only",
        "residual_states_v2_archetype_market_failure_posteriors",
        "residual_states_v2_full_aegmm",
        "residual_states_v2_full_context",
        "residual_states_v2_distilled",
        "residual_states_v2_distilled_local_interactions",
        "residual_states_v2_distilled_priors",
        "residual_states_v2_priors_hitrate_context",
        "residual_states_v2_priors_outcome_context",
        "residual_states_v2_priors_outcome_context_weighted",
        "residual_states_v2_priors_outcome_context_identity",
        "residual_states_v2_priors_outcome_context_identity_weighted",
        "residual_states_v2_priors_outcome_context_interactions",
        "residual_states_v2_priors_failure_pressure_context",
        "residual_states_v2_priors_failure_pressure_identity_context",
        "residual_states_v2_distilled_priors_hitrate_context",
        "residual_states_v2_distilled_weighted_mild",
        "residual_states_v2_distilled_weighted_contrast",
        "residual_states_v2_distilled_weighted_short_adverse",
        "residual_states_v2_priors_local_models",
        "residual_states_v2_priors_local_models_shrunk",
    }
)
LOCAL_META_MODEL_ARMS = frozenset(
    {
        "residual_states_v2_priors_local_models",
        "residual_states_v2_priors_local_models_shrunk",
    }
)
RESIDUAL_STATES_V2_DISTILLED_ARMS = frozenset(
    {
        "residual_states_v2_distilled",
        "residual_states_v2_distilled_local_interactions",
        "residual_states_v2_distilled_priors",
        "residual_states_v2_distilled_priors_hitrate_context",
        "residual_states_v2_distilled_weighted_mild",
        "residual_states_v2_distilled_weighted_contrast",
        "residual_states_v2_distilled_weighted_short_adverse",
    }
)
RESIDUAL_STATES_V2_WEIGHTED_ARMS = frozenset(
    {
        "residual_states_v2_distilled_weighted_mild",
        "residual_states_v2_distilled_weighted_contrast",
        "residual_states_v2_distilled_weighted_short_adverse",
        "residual_states_v2_priors_outcome_context_weighted",
        "residual_states_v2_priors_outcome_context_identity_weighted",
    }
)
LOCAL_AEGMM_ARMS = frozenset(
    {
        "local_aegmm_market",
        "local_aegmm_geometry",
        "local_aegmm_joint",
        "local_aegmm_all_three",
    }
)
AE_GMM_HINTS = (
    "aegmm",
    "gmm_",
    "mahalanobis",
    "reconstruction",
    "cluster_speed",
    "cluster_acceleration",
    "dae_b16",
)
OUTCOME_CONTEXT_HALFLIFE_DAYS = (3.0, 7.0, 14.0)
OUTCOME_CONTEXT_FEATURES = tuple(
    [
        name
        for hl in (3, 7, 14)
        for name in (
            f"base_arch_ev_recent_mean_hl{hl}d",
            f"base_arch_ev_expected_mean_hl{hl}d",
            f"base_arch_ev_surprise_hl{hl}d",
            f"base_arch_bad_mae_recent_rate_hl{hl}d",
            f"base_arch_bad_mae_expected_rate_hl{hl}d",
            f"base_arch_bad_mae_surprise_hl{hl}d",
            f"base_arch_timeout_recent_rate_hl{hl}d",
            f"base_arch_timeout_expected_rate_hl{hl}d",
            f"base_arch_timeout_surprise_hl{hl}d",
            f"base_arch_dirty_recent_rate_hl{hl}d",
            f"base_arch_dirty_expected_rate_hl{hl}d",
            f"base_arch_dirty_surprise_hl{hl}d",
            f"base_arch_outcome_support_log1p_hl{hl}d",
            f"base_arch_outcome_effective_n_hl{hl}d",
        )
    ]
)
FAILURE_PRESSURE_CONTEXT_FEATURES = tuple(
    [
        name
        for hl in (3, 7, 14)
        for name in (
            f"base_arch_failure_pressure_hl{hl}d",
            f"base_arch_bad_mae_pressure_hl{hl}d",
            f"base_arch_timeout_pressure_hl{hl}d",
            f"base_arch_dirty_pressure_hl{hl}d",
            f"base_arch_opportunity_pressure_hl{hl}d",
            f"base_arch_quality_balance_hl{hl}d",
            f"base_arch_failure_mode_bad_mae_share_hl{hl}d",
            f"base_arch_failure_mode_timeout_share_hl{hl}d",
            f"base_arch_failure_mode_dirty_share_hl{hl}d",
            f"base_arch_pressure_support_hl{hl}d",
        )
    ]
)
OUTCOME_IDENTITY_INTERACTION_BASES = tuple(
    name
    for hl in (3, 7, 14)
    for name in (
        f"base_arch_bad_mae_recent_rate_hl{hl}d",
        f"base_arch_bad_mae_surprise_hl{hl}d",
        f"base_arch_timeout_recent_rate_hl{hl}d",
        f"base_arch_ev_expected_mean_hl{hl}d",
        f"base_arch_ev_surprise_hl{hl}d",
        f"base_arch_hit_surprise_hl{hl}d",
    )
)
REQUIRED_COLUMNS = (
    "__ts__",
    "__symbol__",
    "side_name",
    "month",
    "selected_top30",
    "score",
    "source_tag",
    "archetype_policy_key",
    "archetype_label_family",
    "policy_archetype",
    "local_side_archetype",
    "__first_touch_target_soft__",
    "target_soft",
    "exec_margin",
    "ev_after_1pct",
    "first_touch_gross",
    "first_touch_bad_mae_1r",
    "full_path_bad_mae_1r",
    "timeout",
    "clean_exec",
    "dirty_positive",
    "mfe_before_mae_1r",
    "mae_before_mfe_1r",
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _parse_months(value: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(value, str):
        tokens = [part.strip() for part in value.split(",")]
    else:
        tokens = [str(part).strip() for part in value]
    months: list[str] = []
    for token in tokens:
        if not token:
            continue
        try:
            months.append(str(pd.Period(token, freq="M")))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid YYYY-MM month: {token!r}") from exc
    if not months:
        raise ValueError("At least one evaluation month is required")
    return tuple(dict.fromkeys(months))


def _feature_slug(value: Any, *, max_len: int = 80) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^0-9a-zA-Z]+", "_", text).strip("_")
    return (text or "missing")[:max_len]


def _append_meta_identity_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add observable side/archetype identity columns for local context learning."""

    out = data.copy()
    side = out.get("side_name", pd.Series("missing", index=out.index)).astype(str).str.lower()
    arch = out.get(
        "archetype_policy_key", pd.Series("missing", index=out.index)
    ).astype(str)
    side_arch = side + "__" + arch
    for value in sorted(side.dropna().unique().tolist()):
        out[f"meta_identity_side__{_feature_slug(value)}"] = side.eq(value).astype(
            np.float32
        )
    for value in sorted(arch.dropna().unique().tolist()):
        out[f"meta_identity_arch__{_feature_slug(value)}"] = arch.eq(value).astype(
            np.float32
        )
    for value in sorted(side_arch.dropna().unique().tolist()):
        out[f"meta_identity_side_arch__{_feature_slug(value)}"] = side_arch.eq(
            value
        ).astype(np.float32)
    return out


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True), encoding="utf-8"
    )


def _quote_ident(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def _reference_contract(
    reference_dir: Path,
) -> tuple[list[str], dict[str, Any], dict[str, Any]]:
    manifest_path = reference_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    selected = (
        manifest.get("selected_feature_union")
        or manifest.get("selected_features")
        or []
    )
    selected = [str(name) for name in selected]
    params = dict(
        manifest.get("regressor_params") or manifest.get("classifier_params") or {}
    )
    if not selected or not params:
        raise ValueError(f"Reference contract is incomplete: {manifest_path}")
    return selected, params, manifest


def _parquet_columns(path: Path) -> list[str]:
    return [str(name) for name in pq.ParquetFile(path).schema.names]


def _compact_projection_columns(handoff: Path, selected: Sequence[str]) -> list[str]:
    available = _parquet_columns(handoff)
    selected_raw = [
        name for name in selected if name not in META_POST_SELECTION_OOD_FEATURE_NAMES
    ]
    state_cols = [
        name
        for name in available
        if any(hint.lower() in name.lower() for hint in AE_GMM_HINTS)
        and name not in OUTCOME_COLUMNS
    ]
    requested = list(REQUIRED_COLUMNS) + selected_raw + state_cols
    return [name for name in dict.fromkeys(requested) if name in available]


def _materialize_compact_reference(
    *,
    handoff: Path,
    prediction_glob: str,
    selected: Sequence[str],
    output: Path,
) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    available = _parquet_columns(handoff)
    projection = _compact_projection_columns(handoff, selected)
    arch_sources = [
        f"cast({_quote_ident(name)} as varchar)"
        for name in (
            "archetype_policy_key",
            "__archetype_policy_key__",
            "policy_archetype",
            "source_tag",
        )
        if name in available
    ]
    arch_expr = (
        f"coalesce({', '.join(arch_sources)}, 'missing')"
        if arch_sources
        else "'missing'"
    )
    canonical_arch_projection = (
        ""
        if "archetype_policy_key" in projection
        else f", {arch_expr} AS archetype_policy_key"
    )
    select_cols = ",\n".join(f"h.{_quote_ident(name)}" for name in projection)
    selected_filter = (
        "WHERE coalesce(cast(selected_top30 as boolean), true)"
        if "selected_top30" in projection
        else ""
    )
    sql = f"""
    COPY (
      WITH h0 AS (
        SELECT {select_cols}{canonical_arch_projection}, {arch_expr} AS __join_arch,
               row_number() OVER (
                 PARTITION BY CAST(__ts__ AS TIMESTAMPTZ), __symbol__, lower(side_name), {arch_expr}
                 ORDER BY CAST(__ts__ AS TIMESTAMPTZ)
               ) AS __rn
        FROM read_parquet('{handoff.as_posix()}') h
        {selected_filter}
      ),
      h AS (
        SELECT * EXCLUDE (__rn) FROM h0 WHERE __rn = 1
      ),
      p AS (
        SELECT CAST(__ts__ AS TIMESTAMPTZ) AS __pred_ts,
               __symbol__ AS __pred_symbol,
               lower(side_name) AS __pred_side,
               coalesce(cast(archetype_policy_key as varchar), 'missing') AS __pred_arch,
               avg(score_meta_base_soft_label) AS score_meta_base_soft_label,
               min(oos_fold) AS oos_fold,
               min(valid_start) AS valid_start,
               max(valid_end) AS valid_end
        FROM read_parquet('{prediction_glob}', union_by_name=true)
        GROUP BY ALL
      )
      SELECT h.* EXCLUDE (__join_arch),
             p.score_meta_base_soft_label,
             p.oos_fold,
             p.valid_start,
             p.valid_end
      FROM h
      LEFT JOIN p
        ON CAST(h.__ts__ AS TIMESTAMPTZ) = p.__pred_ts
       AND h.__symbol__ = p.__pred_symbol
       AND lower(h.side_name) = p.__pred_side
       AND h.__join_arch = p.__pred_arch
    ) TO '{output.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """
    con = duckdb.connect()
    con.execute(sql)
    summary = con.execute(
        f"""
        SELECT count(*) AS row_count,
               min(CAST(__ts__ AS TIMESTAMPTZ)) min_ts,
               max(CAST(__ts__ AS TIMESTAMPTZ)) max_ts,
               count(score_meta_base_soft_label) reference_score_rows,
               count(distinct __symbol__) symbols
        FROM read_parquet('{output.as_posix()}')
        """
    ).fetchone()
    con.close()
    return {
        "path": str(output),
        "projected_columns": projection,
        "projected_column_count": len(projection),
        "rows": int(summary[0]),
        "min_ts": str(summary[1]),
        "max_ts": str(summary[2]),
        "reference_score_rows": int(summary[3]),
        "symbols": int(summary[4]),
    }


def _symbol_feature_path(root: Path, symbol: str) -> Path:
    token = str(symbol).replace("/", "_")
    return root / f"symbol={token}.parquet"


def _downcast(frame: pd.DataFrame) -> pd.DataFrame:
    for name in frame.select_dtypes(include=["float64"]).columns:
        frame[name] = pd.to_numeric(frame[name], errors="coerce", downcast="float")
    for name in frame.select_dtypes(include=["int64"]).columns:
        frame[name] = pd.to_numeric(frame[name], errors="coerce", downcast="integer")
    return frame


def _ensure_base_score(frame: pd.DataFrame) -> pd.DataFrame:
    """Expose one inference-available base-score column without mutating callers."""

    if "score_base" in frame.columns:
        return frame
    if "score" not in frame.columns:
        raise ValueError(
            "Local AE/GMM state fitting requires the frozen base score column "
            "'score_base' or source column 'score'."
        )
    out = frame.copy(deep=False)
    out["score_base"] = pd.to_numeric(out["score"], errors="coerce").astype(np.float32)
    return out


def _append_lifecycle_features(
    *,
    compact_path: Path,
    feature_root: Path,
    output: Path,
    lifecycle_keys: Sequence[str],
) -> dict[str, Any]:
    frame = pd.read_parquet(compact_path)
    frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True, errors="coerce")
    frame = _downcast(frame)
    values = np.full((len(frame), len(lifecycle_keys)), np.nan, dtype=np.float32)
    matched_rows = 0
    missing_symbol_files: list[str] = []
    grouped_positions = frame.groupby("__symbol__", sort=True).indices
    for symbol, positions_raw in grouped_positions.items():
        positions = np.asarray(positions_raw, dtype=np.int64)
        path = _symbol_feature_path(feature_root, str(symbol))
        if not path.exists():
            missing_symbol_files.append(str(symbol))
            continue
        ts = frame.iloc[positions]["__ts__"]
        features = read_symbol_features(
            str(path),
            columns=list(lifecycle_keys),
            start_ts=ts.min(),
            end_ts=ts.max(),
        )
        if features.empty:
            continue
        feature_index = pd.to_datetime(features.index, utc=True, errors="coerce")
        features = features.copy(deep=False)
        features.index = feature_index
        features = features[~features.index.duplicated(keep="last")]
        aligned = features.reindex(ts.to_numpy())
        available = [name for name in lifecycle_keys if name in aligned.columns]
        if available:
            indices = [lifecycle_keys.index(name) for name in available]
            values[np.ix_(positions, indices)] = aligned[available].to_numpy(
                dtype=np.float32, copy=False
            )
            matched_rows += int(
                np.isfinite(aligned[available].to_numpy(dtype=np.float32, copy=False))
                .any(axis=1)
                .sum()
            )
    for idx, name in enumerate(lifecycle_keys):
        frame[name] = values[:, idx]
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output, index=False, compression="zstd")
    latest = (
        frame.sort_values("__ts__", kind="stable")
        .groupby("__symbol__", sort=False)
        .tail(1)
    )
    finite = np.isfinite(
        latest[list(lifecycle_keys)].to_numpy(dtype=np.float32, copy=False)
    )
    result = {
        "path": str(output),
        "rows": int(len(frame)),
        "columns": int(frame.shape[1]),
        "lifecycle_feature_count": int(len(lifecycle_keys)),
        "rows_with_any_lifecycle_feature": int(matched_rows),
        "latest_lifecycle_finite_share": float(finite.mean()) if finite.size else 0.0,
        "missing_symbol_files": missing_symbol_files,
    }
    del frame, values
    gc.collect()
    return result


def _prepare_local_aegmm_training_archive(
    *,
    archive_path: Path,
    feature_root: Path,
    output_dir: Path,
    state_feature_keys: Sequence[str],
    force: bool,
) -> tuple[Path, dict[str, Any]]:
    """Build a compact, feature-hydrated historical state-fit archive.

    The source handoff contains many target and legacy feature columns.  The
    AE/GMM fit needs only identity, frozen base score, train outcomes for
    semantic selection, and the observable state features reloaded from the
    canonical feature store.  This keeps the historical fit memory-bounded and
    makes the state-input contract identical to OOS/inference rows.
    """

    cache_dir = output_dir / "cache"
    compact_path = cache_dir / "local_aegmm_training_archive_compact.parquet"
    hydrated_path = cache_dir / "local_aegmm_training_archive_state_features.parquet"
    manifest_path = cache_dir / "local_aegmm_training_archive.manifest.json"
    required = [
        "__ts__",
        "__symbol__",
        "side_name",
        "score",
        "ev_after_1pct",
        "exec_margin",
        "clean_exec",
        "dirty_positive",
        "first_touch_bad_mae_1r",
        "full_path_bad_mae_1r",
        "timeout",
        "__archetype_policy_key__",
        "archetype_policy_key",
    ]
    available = set(_parquet_columns(archive_path))
    selected = [name for name in required if name in available]
    essential = {"__ts__", "__symbol__", "side_name", "score"}
    missing_essential = sorted(essential - set(selected))
    if missing_essential:
        raise ValueError(
            "Local AE/GMM training archive misses required inference/train "
            f"columns: {missing_essential}; archive={archive_path}"
        )
    archive_columns = (
        set(_parquet_columns(hydrated_path)) if hydrated_path.exists() else set()
    )
    missing_state_features = [
        name for name in state_feature_keys if name not in archive_columns
    ]
    if force or not compact_path.exists():
        print(
            json.dumps(
                {
                    "event": "local_aegmm_archive_compact_start",
                    "source": str(archive_path),
                    "columns": len(selected),
                }
            ),
            flush=True,
        )
        compact = pd.read_parquet(archive_path, columns=selected)
        compact["__ts__"] = pd.to_datetime(compact["__ts__"], utc=True, errors="coerce")
        if "archetype_policy_key" not in compact.columns:
            compact["archetype_policy_key"] = compact.get(
                "__archetype_policy_key__", "missing"
            )
        compact["archetype_policy_key"] = (
            compact["archetype_policy_key"].astype(str).fillna("missing")
        )
        compact = _ensure_base_score(_downcast(compact))
        compact_path.parent.mkdir(parents=True, exist_ok=True)
        compact.to_parquet(compact_path, index=False, compression="zstd")
        print(
            json.dumps(
                {
                    "event": "local_aegmm_archive_compact_complete",
                    "rows": int(len(compact)),
                    "path": str(compact_path),
                }
            ),
            flush=True,
        )
        del compact
        gc.collect()
    if force or not hydrated_path.exists() or missing_state_features:
        print(
            json.dumps(
                {
                    "event": "local_aegmm_archive_hydration_start",
                    "rows_source": str(compact_path),
                    "state_feature_count": int(len(state_feature_keys)),
                    "missing_state_feature_count": int(len(missing_state_features)),
                }
            ),
            flush=True,
        )
        hydration = _append_lifecycle_features(
            compact_path=compact_path,
            feature_root=feature_root,
            output=hydrated_path,
            lifecycle_keys=state_feature_keys,
        )
        print(
            json.dumps(
                {"event": "local_aegmm_archive_hydration_complete", **hydration}
            ),
            flush=True,
        )
    else:
        hydration = {
            "path": str(hydrated_path),
            "status": "cache_hit",
            "required_feature_count": int(len(state_feature_keys)),
            "missing_required_features": [],
        }
    manifest = {
        "source_archive": str(archive_path),
        "compact_archive": str(compact_path),
        "hydrated_archive": str(hydrated_path),
        "state_feature_keys": list(state_feature_keys),
        "inference_score_contract": "score_base copied from the frozen source/base score",
        "outcome_contract": "outcomes are retained only for train-time state semantic selection and priors",
        "hydration": hydration,
    }
    _write_json(manifest_path, manifest)
    return hydrated_path, manifest


def prepare_dataset(
    *,
    handoff: Path,
    reference_dir: Path,
    feature_root: Path,
    output_dir: Path,
    force: bool,
) -> tuple[Path, dict[str, Any], list[str], dict[str, Any]]:
    selected, params, reference_manifest = _reference_contract(reference_dir)
    compact = output_dir / "cache" / "compact_reference.parquet"
    augmented = output_dir / "cache" / "compact_reference_with_lifecycle.parquet"
    output_dir.mkdir(parents=True, exist_ok=True)
    if force or not compact.exists():
        compact_manifest = _materialize_compact_reference(
            handoff=handoff,
            prediction_glob=str(reference_dir / "prediction_shards" / "*.parquet"),
            selected=selected,
            output=compact,
        )
    else:
        compact_manifest = {"path": str(compact), "status": "cache_hit"}
    lifecycle_keys = list(
        dict.fromkeys(
            [
                *CFG.get("CRASH_LIFECYCLE_NEW_FEATURE_KEYS", []),
                *META_MARKET_STATE_FEATURES,
                *DEFAULT_RELATIVE_FEATURES,
            ]
        )
    )
    augmented_columns = (
        set(_parquet_columns(augmented)) if augmented.exists() else set()
    )
    missing_augmented_features = [
        name for name in lifecycle_keys if name not in augmented_columns
    ]
    if force or not augmented.exists() or missing_augmented_features:
        lifecycle_manifest = _append_lifecycle_features(
            compact_path=compact,
            feature_root=feature_root,
            output=augmented,
            lifecycle_keys=lifecycle_keys,
        )
    else:
        lifecycle_manifest = {
            "path": str(augmented),
            "status": "cache_hit",
            "required_feature_count": int(len(lifecycle_keys)),
            "missing_required_features": [],
        }
    manifest = {
        "compact": compact_manifest,
        "lifecycle": lifecycle_manifest,
        "reference_manifest": str(reference_dir / "manifest.json"),
        "reference_selected_features": selected,
        "reference_model_params": params,
        "reference_model_preserved": True,
        "dataset_path": str(augmented),
    }
    _write_json(output_dir / "dataset_manifest.json", manifest)
    return augmented, manifest, selected, params


def _residual_cache_key(use_ae_gmm: bool) -> str:
    return "residual_walkforward_ae_gmm" if use_ae_gmm else "residual_walkforward_raw"


def build_walkforward_residual_features(
    *,
    data: pd.DataFrame,
    candidate_features: Sequence[str],
    output_dir: Path,
    use_ae_gmm: bool,
    force: bool,
    min_train_months: int = 1,
) -> tuple[pd.DataFrame, dict[str, Any], ResidualArchetypeRecognizer | None]:
    cache_key = _residual_cache_key(use_ae_gmm)
    feature_path = output_dir / "cache" / f"{cache_key}.parquet"
    manifest_path = output_dir / "cache" / f"{cache_key}.manifest.json"
    if feature_path.exists() and manifest_path.exists() and not force:
        return (
            pd.read_parquet(feature_path),
            json.loads(manifest_path.read_text()),
            None,
        )
    score_col = next(
        (
            name
            for name in (
                "score_regime_calibrated",
                "score_meta_base_soft_label",
                "score",
            )
            if name in data.columns
            and pd.to_numeric(data[name], errors="coerce").notna().any()
        ),
        None,
    )
    if score_col is None:
        raise ValueError("Residual walk-forward input has no usable frozen score")
    work = data[pd.to_numeric(data[score_col], errors="coerce").notna()].copy()
    work["__ts__"] = pd.to_datetime(work["__ts__"], utc=True, errors="coerce")
    work["calendar_month"] = work["__ts__"].dt.to_period("M").astype(str)
    work = add_reference_surprise_targets(
        work,
        ResidualArchetypeConfig(score_col=score_col, rank_scope="global"),
    )
    months = sorted(work["calendar_month"].dropna().unique().tolist())
    frames: list[pd.DataFrame] = []
    catalogs: list[pd.DataFrame] = []
    fold_manifests: list[dict[str, Any]] = []
    final_recognizer: ResidualArchetypeRecognizer | None = None
    frozen_local_ae_gmm: dict[
        tuple[str, str], tuple[dict[str, Any], list[str], list[str]]
    ] = {}
    for month_idx, month in enumerate(months):
        if month_idx < int(min_train_months):
            continue
        start = pd.Timestamp(pd.Period(month).start_time, tz="UTC")
        end = pd.Timestamp((pd.Period(month) + 1).start_time, tz="UTC")
        train = work[work["__ts__"].lt(start)]
        valid = work[work["__ts__"].ge(start) & work["__ts__"].lt(end)]
        if len(train) < 5_000 or len(valid) < 100:
            continue
        cfg = ResidualArchetypeConfig(
            score_col=score_col,
            rank_scope="global",
            use_residual_ae_gmm=bool(use_ae_gmm),
            allow_side_fallback=False,
            label_mode="economic_semantic",
            random_state=20260711 + month_idx * 101,
        )
        recognizer = ResidualArchetypeRecognizer(
            config=cfg,
            candidate_features=list(candidate_features),
        )
        recognizer.frozen_ae_gmm_by_local = dict(frozen_local_ae_gmm)
        recognizer.fit(train)
        for key, model in recognizer.local_models.items():
            if key not in frozen_local_ae_gmm and model.ae_gmm_state:
                frozen_local_ae_gmm[key] = (
                    model.ae_gmm_state,
                    list(model.ae_gmm_input_features),
                    list(model.ae_gmm_output_features),
                )
        safe_valid = strip_outcomes_for_oos(valid)
        generated = recognizer.transform_oos(safe_valid)
        keys = valid.loc[
            :, [name for name in KEY_COLUMNS if name in valid.columns]
        ].copy()
        keys["calendar_month"] = str(month)
        keys["residual_feature_provenance"] = "walkforward_oos"
        frames.append(
            pd.concat(
                [keys.reset_index(drop=True), generated.reset_index(drop=True)], axis=1
            )
        )
        if not recognizer.catalog_.empty:
            catalog = recognizer.catalog_.copy()
            catalog["fit_through"] = str(start - pd.Timedelta(nanoseconds=1))
            catalog["oos_month"] = str(month)
            catalogs.append(catalog)
        fold_manifests.append(
            {
                "month": str(month),
                "train_rows": int(len(train)),
                "valid_rows": int(len(valid)),
                "frozen_ae_gmm_key_count": int(len(frozen_local_ae_gmm)),
                **recognizer.manifest(),
            }
        )
        final_recognizer = recognizer
        print(
            json.dumps(
                {
                    "event": "residual_archetype_fold_complete",
                    "month": month,
                    "train_rows": len(train),
                    "valid_rows": len(valid),
                    "use_ae_gmm": bool(use_ae_gmm),
                }
            ),
            flush=True,
        )
        del train, valid, safe_valid, generated, recognizer
        gc.collect()
    output = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(feature_path, index=False, compression="zstd")
    catalog_all = pd.concat(catalogs, ignore_index=True) if catalogs else pd.DataFrame()
    catalog_all.to_csv(output_dir / f"{cache_key}_archetype_catalog.csv", index=False)
    manifest = {
        "schema": "walkforward_residual_archetype_features_v1",
        "use_residual_ae_gmm": bool(use_ae_gmm),
        "label_mode": "economic_semantic",
        "ae_gmm_contract": (
            "fit once at first supported side x archetype fold, then frozen"
        ),
        "rows": int(len(output)),
        "months": sorted(
            output.get("calendar_month", pd.Series(dtype=str))
            .astype(str)
            .unique()
            .tolist()
        ),
        "generated_features": [
            name for name in output.columns if name.startswith("meta_resid_")
        ],
        "folds": fold_manifests,
        "leakage_contract": "Each month uses only earlier frozen-current-meta OOS rows; OOS transforms reject outcomes.",
    }
    _write_json(manifest_path, manifest)
    if final_recognizer is not None:
        state_path = output_dir / "states" / f"{cache_key}_latest_recognizer.joblib"
        state_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(final_recognizer, state_path)
        _write_json(
            state_path.with_suffix(".manifest.json"), final_recognizer.manifest()
        )
        if final_recognizer.ae_gmm_state.get("enabled", False):
            save_ae_gmm_state_artifact(
                final_recognizer.ae_gmm_state,
                output_dir / "states" / f"{cache_key}_ae_gmm_state.pkl",
                manifest_path=output_dir
                / "states"
                / f"{cache_key}_ae_gmm_state_manifest.json",
                extra_manifest={
                    "role": "residual_archetype_recognizer",
                    "fit_scope": "train_only",
                },
            )
    return output, manifest, final_recognizer


def _matrix_fit_transform(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    columns: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    cols = [str(name) for name in columns if str(name) in train.columns]
    train_x = (
        train.reindex(columns=cols)
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
    )
    valid_x = (
        valid.reindex(columns=cols)
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
    )
    medians = (
        train_x.median(axis=0, numeric_only=True)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )
    train_x = train_x.fillna(medians).fillna(0.0).astype(np.float32)
    valid_x = valid_x.fillna(medians).fillna(0.0).astype(np.float32)
    return train_x, valid_x, {str(k): float(v) for k, v in medians.items()}


def _add_reference_fold_features(
    train: pd.DataFrame,
    valid: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reproduce causal base reliability overlays for meta training.

    Hit-surprise features are generated for every fold but only become model
    inputs for arms that explicitly include ``HIT_SURPRISE_NUMERIC_FEATURES``.
    Validation rows receive train-only priors; training rows use strictly
    earlier rows in the same fold, matching the S52 handoff contract.
    """

    train = train.copy()
    valid = valid.copy()
    for frame in (train, valid):
        if "clean_exec_label" not in frame.columns:
            clean_exec = frame.get(
                "clean_exec", pd.Series(np.nan, index=frame.index, dtype=np.float32)
            )
            frame["clean_exec_label"] = (
                pd.to_numeric(clean_exec, errors="coerce")
                .fillna(0.0)
                .gt(0.5)
                .astype(np.float32)
            )
    train, valid = _add_fold_base_prior_features(
        train, valid, selected_col="selected_top30"
    )
    train, valid = _add_fold_reliability_features(train, valid)
    train, valid = _add_fold_hit_surprise_features(train, valid)
    train, valid = _add_fold_outcome_context_features(train, valid)
    train = _add_failure_pressure_context_features(train)
    valid = _add_failure_pressure_context_features(valid)
    return _add_fold_outcome_identity_interactions(train, valid)


def _outcome_identity_interaction_columns(frame: pd.DataFrame) -> list[str]:
    identities = [
        name
        for name in frame.columns
        if name.startswith("meta_identity_arch__")
        or name.startswith("meta_identity_side_arch__")
    ]
    return [
        f"{base}__x__{identity}"
        for base in OUTCOME_IDENTITY_INTERACTION_BASES
        for identity in identities
    ]


def _add_fold_outcome_identity_interactions(
    train: pd.DataFrame, valid: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add compact local interactions for recent outcome context."""

    train = train.copy()
    valid = valid.copy()
    identity_columns = [
        name
        for name in train.columns
        if name.startswith("meta_identity_arch__")
        or name.startswith("meta_identity_side_arch__")
    ]
    for base in OUTCOME_IDENTITY_INTERACTION_BASES:
        if base not in train.columns:
            continue
        train_base = pd.to_numeric(train[base], errors="coerce").fillna(0.0).astype(
            np.float32
        )
        valid_base = pd.to_numeric(
            valid.get(base, pd.Series(0.0, index=valid.index)), errors="coerce"
        ).fillna(0.0).astype(np.float32)
        for identity in identity_columns:
            name = f"{base}__x__{identity}"
            train_identity = pd.to_numeric(
                train[identity], errors="coerce"
            ).fillna(0.0).astype(np.float32)
            valid_identity = pd.to_numeric(
                valid.get(identity, pd.Series(0.0, index=valid.index)),
                errors="coerce",
            ).fillna(0.0).astype(np.float32)
            train[name] = (train_base * train_identity).astype(np.float32)
            valid[name] = (valid_base * valid_identity).astype(np.float32)
    return train, valid


def _add_failure_pressure_context_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Compress recent side x archetype outcome context into robust pressures."""

    out = frame.copy()
    for hl in (3, 7, 14):
        suffix = f"hl{hl}d"
        bad = _numeric_feature(
            out, f"base_arch_bad_mae_surprise_{suffix}", default=0.0
        ).clip(-1.0, 1.0)
        timeout = _numeric_feature(
            out, f"base_arch_timeout_surprise_{suffix}", default=0.0
        ).clip(-1.0, 1.0)
        dirty = _numeric_feature(
            out, f"base_arch_dirty_surprise_{suffix}", default=0.0
        ).clip(-1.0, 1.0)
        ev = _numeric_feature(out, f"base_arch_ev_surprise_{suffix}", default=0.0)
        hit_z = _numeric_feature(
            out, f"base_arch_hit_surprise_z_{suffix}", default=0.0
        ).clip(-6.0, 6.0)
        support = _numeric_feature(
            out, f"base_arch_outcome_effective_n_{suffix}", default=0.0
        )
        support_factor = (
            np.log1p(np.maximum(support.to_numpy(dtype=np.float32), 0.0))
            / np.log(31.0)
        )
        support_factor = np.clip(support_factor, 0.0, 1.0).astype(np.float32)

        bad_pos = np.maximum(bad.to_numpy(dtype=np.float32), 0.0)
        timeout_pos = np.maximum(timeout.to_numpy(dtype=np.float32), 0.0)
        dirty_pos = np.maximum(dirty.to_numpy(dtype=np.float32), 0.0)
        ev_arr = ev.to_numpy(dtype=np.float32)
        hit_arr = hit_z.to_numpy(dtype=np.float32)
        ev_negative = np.maximum(-ev_arr / 0.01, 0.0)
        ev_positive = np.maximum(ev_arr / 0.01, 0.0)
        hit_negative = np.maximum(-hit_arr / 3.0, 0.0)
        hit_positive = np.maximum(hit_arr / 3.0, 0.0)

        failure_pressure = support_factor * (
            0.42 * bad_pos
            + 0.26 * timeout_pos
            + 0.20 * dirty_pos
            + 0.08 * np.clip(ev_negative, 0.0, 2.0)
            + 0.04 * np.clip(hit_negative, 0.0, 2.0)
        )
        opportunity_pressure = support_factor * (
            0.50 * np.clip(ev_positive, 0.0, 2.0)
            + 0.30 * np.clip(hit_positive, 0.0, 2.0)
            + 0.20
            * np.maximum(
                -bad.to_numpy(dtype=np.float32)
                - timeout.to_numpy(dtype=np.float32)
                - dirty.to_numpy(dtype=np.float32),
                0.0,
            )
        )
        total_failure = np.maximum(bad_pos + timeout_pos + dirty_pos, 1e-6)
        out[f"base_arch_failure_pressure_{suffix}"] = failure_pressure.astype(
            np.float32
        )
        out[f"base_arch_bad_mae_pressure_{suffix}"] = (
            support_factor * bad_pos
        ).astype(np.float32)
        out[f"base_arch_timeout_pressure_{suffix}"] = (
            support_factor * timeout_pos
        ).astype(np.float32)
        out[f"base_arch_dirty_pressure_{suffix}"] = (
            support_factor * dirty_pos
        ).astype(np.float32)
        out[f"base_arch_opportunity_pressure_{suffix}"] = (
            opportunity_pressure.astype(np.float32)
        )
        out[f"base_arch_quality_balance_{suffix}"] = (
            opportunity_pressure - failure_pressure
        ).astype(np.float32)
        out[f"base_arch_failure_mode_bad_mae_share_{suffix}"] = (
            bad_pos / total_failure
        ).astype(np.float32)
        out[f"base_arch_failure_mode_timeout_share_{suffix}"] = (
            timeout_pos / total_failure
        ).astype(np.float32)
        out[f"base_arch_failure_mode_dirty_share_{suffix}"] = (
            dirty_pos / total_failure
        ).astype(np.float32)
        out[f"base_arch_pressure_support_{suffix}"] = support_factor.astype(
            np.float32
        )
    return out


def _numeric_feature(
    frame: pd.DataFrame,
    column: str,
    *,
    default: float = 0.0,
    clip: tuple[float, float] | None = None,
) -> pd.Series:
    source = frame[column] if column in frame.columns else pd.Series(default, index=frame.index)
    out = pd.to_numeric(source, errors="coerce").fillna(float(default)).astype(float)
    if clip is not None:
        out = out.clip(float(clip[0]), float(clip[1]))
    return out


def _add_fold_outcome_context_features(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    *,
    shrinkage_k: float = 40.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add causal recent EV/path-quality context by side x archetype.

    These are model-facing descriptors of when an archetype is currently
    working or failing.  Validation rows use train-only history.  Training rows
    use strictly earlier rows from the same fold, excluding same-timestamp rows.
    """

    metric_specs = (
        ("ev", "ev_after_1pct", "mean", None),
        ("bad_mae", "full_path_bad_mae_1r", "rate", (0.0, 1.0)),
        ("timeout", "timeout", "rate", (0.0, 1.0)),
        ("dirty", "dirty_positive", "rate", (0.0, 1.0)),
    )
    train = train.copy()
    valid = valid.copy()
    train_days = (
        pd.to_datetime(train["__ts__"], utc=True, errors="coerce")
        .astype("int64")
        .astype(np.float64)
        / (1e9 * 86400.0)
    )
    valid_days = (
        pd.to_datetime(valid["__ts__"], utc=True, errors="coerce")
        .astype("int64")
        .astype(np.float64)
        / (1e9 * 86400.0)
    )
    train_days = pd.Series(train_days, index=train.index).where(
        pd.to_datetime(train["__ts__"], utc=True, errors="coerce").notna(), np.nan
    )
    valid_days = pd.Series(valid_days, index=valid.index).where(
        pd.to_datetime(valid["__ts__"], utc=True, errors="coerce").notna(), np.nan
    )
    train_key = (
        train["side_name"].astype(str).str.lower()
        + "__"
        + _archetype_key(train).astype(str)
    )
    valid_key = (
        valid["side_name"].astype(str).str.lower()
        + "__"
        + _archetype_key(valid).astype(str)
    )
    metric_values = {
        metric: _numeric_feature(train, column, default=0.0, clip=clip)
        for metric, column, _kind, clip in metric_specs
    }
    global_mean = {
        metric: float(values.mean()) if len(values) else 0.0
        for metric, values in metric_values.items()
    }
    source = pd.DataFrame(
        {
            "_key": train_key.astype(str),
            "_day": train_days,
            **{metric: values.astype(np.float64) for metric, values in metric_values.items()},
        },
        index=train.index,
    ).replace([np.inf, -np.inf], np.nan)
    source = source[source["_day"].notna()].sort_values(
        ["_key", "_day"], kind="mergesort"
    )
    source_groups: dict[str, dict[str, Any]] = {}
    for key, group in source.groupby("_key", sort=False):
        days = group["_day"].to_numpy(dtype=np.float64)
        payload: dict[str, Any] = {
            "days": days,
            "offset": float(days[0]) if len(days) else 0.0,
        }
        for metric, _column, _kind, _clip in metric_specs:
            values = group[metric].fillna(global_mean[metric]).to_numpy(dtype=np.float64)
            payload[f"{metric}_values"] = values
            payload[f"{metric}_cumsum"] = np.cumsum(values)
        source_groups[str(key)] = payload

    def assign(
        target: pd.DataFrame, target_days: pd.Series, target_key: pd.Series
    ) -> pd.DataFrame:
        target = target.copy()
        for hl in OUTCOME_CONTEXT_HALFLIFE_DAYS:
            suffix = int(hl)
            recent = {
                metric: np.full(len(target), global_mean[metric], dtype=np.float32)
                for metric, _column, _kind, _clip in metric_specs
            }
            expected = {
                metric: np.full(len(target), global_mean[metric], dtype=np.float32)
                for metric, _column, _kind, _clip in metric_specs
            }
            support_log = np.zeros(len(target), dtype=np.float32)
            effective_n = np.zeros(len(target), dtype=np.float32)
            target_tmp = pd.DataFrame(
                {
                    "_pos": np.arange(len(target), dtype=np.int64),
                    "_key": target_key.astype(str).to_numpy(),
                    "_day": target_days.to_numpy(dtype=np.float64),
                },
                index=target.index,
            )
            alpha = np.log(2.0) / float(hl)
            for key, group in target_tmp.groupby("_key", sort=False):
                payload = source_groups.get(str(key))
                if payload is None:
                    continue
                src_days = payload["days"]
                pos = group["_pos"].to_numpy(dtype=np.int64)
                day = group["_day"].to_numpy(dtype=np.float64)
                valid_day = np.isfinite(day)
                if not bool(np.any(valid_day)):
                    continue
                pos = pos[valid_day]
                day = day[valid_day]
                right = np.searchsorted(src_days, day, side="left").astype(np.int64)
                has_prior = right > 0
                if not bool(np.any(has_prior)):
                    continue
                pos = pos[has_prior]
                day = day[has_prior]
                right = right[has_prior]
                left = np.searchsorted(
                    src_days, day - 4.0 * float(hl), side="left"
                ).astype(np.int64)
                shifted_src_days = src_days - float(payload["offset"])
                shifted_day = day - float(payload["offset"])
                exp1 = np.exp(alpha * shifted_src_days)
                exp2 = np.exp(2.0 * alpha * shifted_src_days)
                c_exp1 = np.concatenate([[0.0], np.cumsum(exp1)])
                c_exp2 = np.concatenate([[0.0], np.cumsum(exp2)])
                win_exp1 = c_exp1[right] - c_exp1[left]
                win_exp2 = c_exp2[right] - c_exp2[left]
                scale1 = np.exp(-alpha * shifted_day)
                scale2 = np.exp(-2.0 * alpha * shifted_day)
                weight_sum = scale1 * win_exp1
                weight_sq_sum = scale2 * win_exp2
                eff_n = np.divide(
                    weight_sum * weight_sum,
                    np.maximum(weight_sq_sum, 1e-12),
                    out=np.zeros_like(weight_sum),
                    where=weight_sum > 1e-12,
                )
                support_log[pos] = np.log1p(weight_sum).astype(np.float32)
                effective_n[pos] = eff_n.astype(np.float32)
                prior_count = right.astype(np.float64)
                prior_weight = prior_count / (prior_count + float(shrinkage_k))
                for metric, _column, _kind, _clip in metric_specs:
                    values = payload[f"{metric}_values"]
                    cumsum = payload[f"{metric}_cumsum"]
                    prior_raw = cumsum[right - 1] / np.maximum(prior_count, 1.0)
                    prior = (
                        prior_weight * prior_raw
                        + (1.0 - prior_weight) * global_mean[metric]
                    )
                    weighted_values = values * exp1
                    c_weighted = np.concatenate([[0.0], np.cumsum(weighted_values)])
                    win_weighted = c_weighted[right] - c_weighted[left]
                    recent_value = np.divide(
                        scale1 * win_weighted,
                        np.maximum(weight_sum, 1e-12),
                        out=prior.copy(),
                        where=weight_sum > 1e-12,
                    )
                    recent[metric][pos] = recent_value.astype(np.float32)
                    expected[metric][pos] = prior.astype(np.float32)
            for metric, _column, kind, _clip in metric_specs:
                target[f"base_arch_{metric}_recent_{kind}_hl{suffix}d"] = recent[
                    metric
                ]
                target[f"base_arch_{metric}_expected_{kind}_hl{suffix}d"] = expected[
                    metric
                ]
                target[f"base_arch_{metric}_surprise_hl{suffix}d"] = (
                    recent[metric] - expected[metric]
                ).astype(np.float32)
            target[f"base_arch_outcome_support_log1p_hl{suffix}d"] = support_log
            target[f"base_arch_outcome_effective_n_hl{suffix}d"] = effective_n
        return target

    return assign(train, train_days, train_key), assign(valid, valid_days, valid_key)


def _fit_ood_state(frame: pd.DataFrame, columns: Sequence[str]) -> dict[str, Any]:
    cols = [str(name) for name in columns if str(name) in frame.columns]
    values = frame.reindex(columns=cols).to_numpy(dtype=np.float32, copy=True)
    values[~np.isfinite(values)] = np.nan
    mean = np.nan_to_num(np.nanmean(values, axis=0), nan=0.0).astype(np.float32)
    std = np.nanstd(values, axis=0).astype(np.float32)
    std = np.where(np.isfinite(std) & (std > 1e-6), std, 1.0).astype(np.float32)
    q25 = np.nanquantile(values, 0.25, axis=0).astype(np.float32)
    q75 = np.nanquantile(values, 0.75, axis=0).astype(np.float32)
    q25 = np.where(np.isfinite(q25), q25, mean).astype(np.float32)
    q75 = np.where(np.isfinite(q75), q75, mean).astype(np.float32)
    return {
        "columns": cols,
        "mean": mean,
        "std": std,
        "q25": q25,
        "q75": q75,
        "schema": "meta_post_selection_ood_v1",
    }


def _apply_ood_state(frame: pd.DataFrame, state: dict[str, Any]) -> pd.DataFrame:
    cols = list(state.get("columns", []))
    values = frame.reindex(columns=cols).to_numpy(dtype=np.float32, copy=True)
    finite = np.isfinite(values)
    mean = np.asarray(state["mean"], dtype=np.float32)
    std = np.asarray(state["std"], dtype=np.float32)
    q25 = np.asarray(state["q25"], dtype=np.float32)
    q75 = np.asarray(state["q75"], dtype=np.float32)
    iqr = np.maximum(q75 - q25, 1e-6)
    lower = q25 - 1.5 * iqr
    upper = q75 + 1.5 * iqr
    filled = np.where(finite, values, mean)
    z = (filled - mean) / std
    abs_z = np.abs(z)
    exceed = ((filled < lower) | (filled > upper)) & finite
    out = frame.copy()
    out["meta_sel_ood_abs_z_mean"] = np.mean(abs_z, axis=1).astype(np.float32)
    out["meta_sel_ood_abs_z_max"] = np.max(abs_z, axis=1).astype(np.float32)
    out["meta_sel_ood_abs_z_p95"] = np.quantile(abs_z, 0.95, axis=1).astype(np.float32)
    out["meta_sel_ood_iqr_exceed_frac"] = np.mean(exceed, axis=1).astype(np.float32)
    out["meta_sel_ood_missing_frac"] = np.mean(~finite, axis=1).astype(np.float32)
    out["meta_sel_ood_centroid_l2"] = np.sqrt(np.mean(z * z, axis=1)).astype(np.float32)
    return out


def _merge_residual_features(
    data: pd.DataFrame,
    residual: pd.DataFrame,
    *,
    fill_unknown: bool = False,
) -> pd.DataFrame:
    if residual.empty:
        return data
    keys = [
        name
        for name in KEY_COLUMNS
        if name in data.columns and name in residual.columns
    ]
    negative_residual_keys = set(CFG.get("NEGATIVE_RESIDUAL_META_FEATURE_KEYS", []))
    generated = [
        name
        for name in residual.columns
        if name.startswith(("meta_resid_", "residual_state_family_"))
        or name in negative_residual_keys
    ]
    right = residual.loc[:, keys + generated].drop_duplicates(keys, keep="last")
    out = data.merge(right, on=keys, how="left", validate="one_to_one")
    for name in generated:
        out[name] = pd.to_numeric(out[name], errors="coerce").astype(np.float32)
    if fill_unknown:
        for name in generated:
            if name.endswith("prob__base_low_edge_noise") or name.endswith("entropy"):
                default = 1.0
            else:
                default = 0.0
            out[name] = out[name].fillna(np.float32(default)).astype(np.float32)
    return out


def _append_residual_state_v2_composites(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy(deep=False)

    def values(name: str) -> pd.Series:
        if name not in out.columns:
            return pd.Series(0.0, index=out.index, dtype=np.float32)
        return pd.to_numeric(out[name], errors="coerce").fillna(0.0)

    adverse = (
        values("meta_resid_arch_prob__base_dirty_high_confidence")
        + values("meta_resid_arch_prob__base_slow_timeout_positive")
        + values("meta_resid_arch_prob__base_bad_mae_false_positive")
    ).clip(0.0, 1.0)
    favorable = (
        values("meta_resid_arch_prob__base_clean_high_confidence")
        + values("meta_resid_arch_prob__base_missed_clean_opportunity")
    ).clip(0.0, 1.0)
    uncertainty = (
        values("meta_resid_arch_prob__base_high_variance_uncertain")
        + values("meta_resid_arch_prob__base_low_edge_noise")
    ).clip(0.0, 1.0)
    path_risk = (
        values("meta_resid_arch_expected_bad_mae")
        + values("meta_resid_arch_expected_timeout")
        + values("meta_resid_arch_expected_dirty_positive")
    ) / np.float32(3.0)
    confidence = values("meta_resid_arch_confidence").clip(0.0, 1.0)
    out["meta_resid_v2_adverse_probability"] = adverse.astype(np.float32)
    out["meta_resid_v2_favorable_probability"] = favorable.astype(np.float32)
    out["meta_resid_v2_uncertainty_probability"] = uncertainty.astype(np.float32)
    out["meta_resid_v2_clean_minus_adverse"] = (favorable - adverse).astype(np.float32)
    out["meta_resid_v2_expected_path_risk"] = path_risk.astype(np.float32)
    out["meta_resid_v2_expected_ev"] = values(
        "meta_resid_arch_expected_ev"
    ).astype(np.float32)
    out["meta_resid_v2_expected_hit_surprise"] = values(
        "meta_resid_arch_expected_hit_surprise"
    ).astype(np.float32)
    out["meta_resid_v2_adverse_confident"] = (adverse * confidence).astype(
        np.float32
    )
    out["meta_resid_v2_favorable_confident"] = (favorable * confidence).astype(
        np.float32
    )
    out["meta_resid_v2_state_available"] = values(
        "meta_resid_arch_local_model"
    ).clip(0.0, 1.0).astype(np.float32)

    probability_columns = sorted(
        name for name in out.columns if name.startswith("meta_resid_arch_prob__")
    )
    if probability_columns:
        probability = out[probability_columns].to_numpy(dtype=np.float32, copy=False)
        probability = np.nan_to_num(probability, nan=0.0, posinf=0.0, neginf=0.0)
        order = np.argsort(probability, axis=1)
        best = order[:, -1]
        best_probability = probability[np.arange(len(out)), best]
        second_probability = (
            probability[np.arange(len(out)), order[:, -2]]
            if probability.shape[1] > 1
            else np.zeros(len(out), dtype=np.float32)
        )
        out["meta_resid_v2_hard_state_id"] = best.astype(np.float32)
        out["meta_resid_v2_hard_state_posterior_max"] = best_probability.astype(
            np.float32
        )
        out["meta_resid_v2_hard_state_posterior_margin"] = (
            best_probability - second_probability
        ).astype(np.float32)
        for state_index, source in enumerate(probability_columns):
            semantic = source.removeprefix("meta_resid_arch_prob__")
            out[f"meta_resid_v2_hard_state__{semantic}"] = (
                best == state_index
            ).astype(np.float32)

    base_archetype = out.get(
        "archetype_policy_key", pd.Series("missing", index=out.index)
    ).astype(str)
    for archetype in sorted(base_archetype.unique().tolist()):
        slug = "".join(
            char if char.isalnum() else "_" for char in str(archetype).lower()
        ).strip("_")
        out[f"meta_base_archetype__{slug}"] = base_archetype.eq(archetype).to_numpy(
            dtype=np.float32
        )
    side = out.get("side_name", pd.Series("missing", index=out.index)).astype(str)
    out["meta_side_short"] = side.str.lower().eq("short").to_numpy(dtype=np.float32)

    ae_probability_columns = sorted(
        name for name in out.columns if name.startswith("meta_resid_ae_gmm_prob_")
    )
    if ae_probability_columns:
        ae_probability = out[ae_probability_columns].to_numpy(
            dtype=np.float32, copy=False
        )
        ae_probability = np.nan_to_num(
            ae_probability, nan=0.0, posinf=0.0, neginf=0.0
        )
        ae_best = np.argmax(ae_probability, axis=1)
        out["meta_resid_v2_ae_hard_cluster_id"] = ae_best.astype(np.float32)
        for cluster_index in range(ae_probability.shape[1]):
            out[f"meta_resid_v2_ae_hard_cluster_{cluster_index}"] = (
                ae_best == cluster_index
            ).astype(np.float32)

    market_probability_columns = sorted(
        name for name in out.columns if name.startswith("meta_resid_market_prob__")
    )
    if market_probability_columns:
        market_probability = out[market_probability_columns].to_numpy(
            dtype=np.float32, copy=False
        )
        market_probability = np.nan_to_num(
            market_probability, nan=0.0, posinf=0.0, neginf=0.0
        )
        market_order = np.argsort(market_probability, axis=1)
        market_best = market_order[:, -1]
        market_max = market_probability[np.arange(len(out)), market_best]
        market_second = market_probability[np.arange(len(out)), market_order[:, -2]]
        out["meta_resid_v2_market_hard_state_id"] = market_best.astype(np.float32)
        out["meta_resid_v2_market_posterior_max"] = market_max.astype(np.float32)
        out["meta_resid_v2_market_posterior_margin"] = (
            market_max - market_second
        ).astype(np.float32)
        for state_index, source in enumerate(market_probability_columns):
            semantic = source.removeprefix("meta_resid_market_prob__")
            out[f"meta_resid_v2_market_hard_state__{semantic}"] = (
                market_best == state_index
            ).astype(np.float32)

    side = out.get("side_name", pd.Series("missing", index=out.index)).astype(str)
    arch = out.get(
        "archetype_policy_key", pd.Series("missing", index=out.index)
    ).astype(str)
    group = side.str.lower() + "__" + arch
    for key in sorted(group.unique().tolist()):
        slug = "".join(char if char.isalnum() else "_" for char in str(key)).strip(
            "_"
        )
        mask = group.eq(key).to_numpy(dtype=np.float32)
        for suffix, source in (
            ("adverse", adverse),
            ("favorable", favorable),
            ("path_risk", path_risk),
            ("expected_ev", out["meta_resid_v2_expected_ev"]),
        ):
            out[f"meta_resid_v2_local__{slug}__{suffix}"] = (
                source.to_numpy(dtype=np.float32) * mask
            ).astype(np.float32)
    return out


def _residual_v2_sample_weight_multiplier(
    frame: pd.DataFrame, arm: str
) -> pd.Series | None:
    """Return a causal train-only emphasis for difficult residual states.

    The multiplier uses only frozen OOS residual-state predictions available as
    model inputs. It never inspects the current fold's realized outcomes.
    """

    if arm not in RESIDUAL_STATES_V2_WEIGHTED_ARMS:
        return None

    def values(name: str) -> np.ndarray:
        if name not in frame.columns:
            return np.zeros(len(frame), dtype=np.float32)
        return (
            pd.to_numeric(frame[name], errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=np.float32, copy=False)
        )

    adverse = np.clip(values("meta_resid_v2_adverse_confident"), 0.0, 1.0)
    favorable = np.clip(values("meta_resid_v2_favorable_confident"), 0.0, 1.0)
    path_risk = np.clip(values("meta_resid_v2_expected_path_risk"), 0.0, 1.0)
    uncertainty = np.clip(
        values("meta_resid_v2_uncertainty_probability"), 0.0, 1.0
    )
    expected_surprise = np.clip(
        values("meta_resid_v2_expected_hit_surprise"), -1.0, 1.0
    )
    negative_surprise = np.maximum(-expected_surprise, 0.0)
    positive_surprise = np.maximum(expected_surprise, 0.0)

    if arm in {
        "residual_states_v2_priors_outcome_context_weighted",
        "residual_states_v2_priors_outcome_context_identity_weighted",
    }:
        recent_bad = np.maximum(
            values("base_arch_bad_mae_surprise_hl3d"),
            values("base_arch_bad_mae_surprise_hl7d"),
        )
        recent_timeout = np.maximum(
            values("base_arch_timeout_surprise_hl3d"),
            values("base_arch_timeout_surprise_hl7d"),
        )
        recent_dirty = np.maximum(
            values("base_arch_dirty_surprise_hl3d"),
            values("base_arch_dirty_surprise_hl7d"),
        )
        ev_shift = np.maximum(
            np.abs(values("base_arch_ev_surprise_hl3d")),
            np.abs(values("base_arch_ev_surprise_hl7d")),
        )
        hit_shift = np.maximum(
            np.abs(values("base_arch_hit_surprise_z_hl3d")),
            np.abs(values("base_arch_hit_surprise_z_hl7d")),
        )
        support = np.maximum(
            values("base_arch_outcome_effective_n_hl3d"),
            values("base_arch_outcome_effective_n_hl7d"),
        )
        support_factor = np.clip(
            np.log1p(np.maximum(support, 0.0)) / np.log(21.0), 0.0, 1.0
        )
        multiplier = (
            1.0
            + support_factor
            * (
                0.65 * np.clip(recent_bad, 0.0, 0.50)
                + 0.45 * np.clip(recent_timeout, 0.0, 0.50)
                + 0.25 * np.clip(recent_dirty, 0.0, 0.50)
                + 0.20 * np.clip(ev_shift / 0.01, 0.0, 2.0)
                + 0.10 * np.clip(hit_shift / 3.0, 0.0, 2.0)
            )
        )
    elif arm == "residual_states_v2_distilled_weighted_mild":
        multiplier = (
            1.0
            + 0.65 * adverse
            + 0.30 * path_risk
            + 0.20 * favorable
            + 0.25 * negative_surprise
            + 0.10 * positive_surprise
        )
    elif arm == "residual_states_v2_distilled_weighted_contrast":
        multiplier = (
            1.0
            + 1.10 * adverse
            + 0.55 * path_risk
            + 0.55 * favorable
            + 0.50 * negative_surprise
            + 0.25 * positive_surprise
            + 0.15 * uncertainty
        )
    else:
        side = frame.get(
            "side_name", pd.Series("missing", index=frame.index)
        ).astype(str).str.lower()
        archetype = frame.get(
            "archetype_policy_key", pd.Series("missing", index=frame.index)
        ).astype(str).str.lower()
        short_failure = (
            side.eq("short")
            & (archetype.str.contains("breakout") | archetype.str.contains("mixed"))
        ).to_numpy(dtype=np.float32)
        multiplier = (
            1.0
            + 0.55 * adverse
            + 0.25 * path_risk
            + 0.20 * favorable
            + short_failure * (0.85 * adverse + 0.60 * negative_surprise)
        )

    multiplier = np.clip(multiplier, 0.25, 4.0).astype(np.float32)
    mean = float(np.mean(multiplier))
    if np.isfinite(mean) and mean > 1e-8:
        multiplier /= np.float32(mean)
    return pd.Series(multiplier, index=frame.index, dtype=np.float32)


def _local_aegmm_blocks_for_arm(arm: str) -> tuple[EconomicAEGMMBlock, ...]:
    blocks = default_meta_economic_aegmm_blocks()
    if arm == "local_aegmm_market":
        return (blocks[0],)
    if arm == "local_aegmm_geometry":
        return (blocks[1],)
    if arm == "local_aegmm_joint":
        return (blocks[2],)
    if arm == "local_aegmm_all_three":
        return blocks
    return ()


def _append_cross_sectional_geometry(data: pd.DataFrame) -> pd.DataFrame:
    needed = set(META_CROSS_SECTIONAL_GEOMETRY_FEATURES)
    missing_geometry = [
        name
        for name in needed
        if name.startswith("meta_xsgeom_") and name not in data.columns
    ]
    if not missing_geometry:
        return data
    # This runs before the alternative meta model is fitted.  Geometry must be
    # based on the frozen base score available at inference, not the reference
    # meta prediction used only as a comparator elsewhere in this experiment.
    score_col = "score_base" if "score_base" in data.columns else "score"
    generated = materialize_cross_sectional_geometry(data, score_col=score_col)
    additions = generated.reindex(
        columns=[name for name in generated.columns if name not in data.columns]
    )
    return pd.concat([data, additions], axis=1, copy=False)


def _append_frozen_local_aegmm(
    data: pd.DataFrame,
    *,
    state_training_data: pd.DataFrame,
    arm: str,
    output_dir: Path,
    seed: int,
    force: bool,
    fit_start: str,
    fit_end: str,
    full_train_fit: bool,
    eval_months: Sequence[str] = EVAL_MONTHS,
) -> tuple[pd.DataFrame, LocalEconomicAEGMM, dict[str, Any]]:
    blocks = _local_aegmm_blocks_for_arm(arm)
    if not blocks:
        raise ValueError(f"Arm is not a local AE/GMM arm: {arm}")
    fit_start_ts = pd.Timestamp(fit_start, tz="UTC")
    fit_end_ts = pd.Timestamp(fit_end, tz="UTC")
    if fit_end_ts <= fit_start_ts:
        raise ValueError("Local AE/GMM fit end must be after fit start")
    fit_mode = "full" if full_train_fit else "sampled"
    semantic_tag = "".join(
        char if char.isalnum() else "_"
        for char in LocalEconomicAEGMMConfig().semantic_version
    ).strip("_")
    state_tag = f"{fit_start_ts:%Y%m%d}_{fit_end_ts:%Y%m%d}_{fit_mode}_{semantic_tag}"
    state_dir = output_dir / arm / "state" / state_tag
    state_path = state_dir / "local_economic_aegmm.joblib"
    manifest_path = state_dir / "local_economic_aegmm.manifest.json"
    if state_path.exists() and not force:
        state = joblib.load(state_path)
    else:
        state_ts = pd.to_datetime(
            state_training_data["__ts__"], utc=True, errors="coerce"
        )
        fit_train = state_training_data.loc[
            state_ts.ge(fit_start_ts) & state_ts.lt(fit_end_ts)
        ]
        if len(fit_train) < 5_000:
            raise ValueError(
                "Local AE/GMM state archive is too small: "
                f"rows={len(fit_train)} start={fit_start_ts} end={fit_end_ts}"
            )
        print(
            json.dumps(
                {
                    "event": "local_aegmm_fit_start",
                    "arm": arm,
                    "fit_rows": int(len(fit_train)),
                    "fit_start": str(fit_start_ts),
                    "fit_end_exclusive": str(fit_end_ts),
                    "full_train_fit": bool(full_train_fit),
                    "blocks": [block.name for block in blocks],
                }
            ),
            flush=True,
        )
        state = LocalEconomicAEGMM(
            config=LocalEconomicAEGMMConfig(
                random_state=int(seed),
                full_train_fit=bool(full_train_fit),
            ),
            blocks=blocks,
        ).fit(fit_train)
        state_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(state, state_path, compress=3)
        _write_json(manifest_path, state.manifest())
        state.catalog_.to_csv(
            state_dir / "local_economic_aegmm_catalog.csv", index=False
        )
        print(
            json.dumps(
                {
                    "event": "local_aegmm_fit_complete",
                    "arm": arm,
                    "side_models": int(len(state.side_models)),
                    "local_models": int(len(state.local_models)),
                    "state_path": str(state_path),
                }
            ),
            flush=True,
        )
    generated = state.transform_train(data)
    out = pd.concat(
        [
            data.drop(
                columns=[name for name in generated.columns if name in data.columns]
            ),
            generated,
        ],
        axis=1,
        copy=False,
    )
    manifest = state.manifest()
    manifest.update(
        {
            "state_path": str(state_path),
            "fit_start_inclusive": str(fit_start_ts),
            "fit_cutoff_exclusive": str(fit_end_ts),
            "fit_archive_rows": int(len(fit_train))
            if "fit_train" in locals()
            else None,
            "full_train_fit": bool(full_train_fit),
            "growing_oos_months": list(eval_months),
            "state_frozen_across_growing_oos_windows": True,
        }
    )
    return out, state, manifest


def _arm_candidate_features(
    arm: str,
    data: pd.DataFrame,
    reference_selected: Sequence[str],
    lifecycle_keys: Sequence[str],
) -> list[str]:
    base = [
        name
        for name in reference_selected
        if name not in META_POST_SELECTION_OOD_FEATURE_NAMES
        and (
            name in data.columns
            or name.startswith(("rel_rankband_", "rel_marginband_"))
        )
    ]
    if arm == "baseline_retrained":
        return base
    if arm == "negative_residual_full_context":
        additions = [
            name
            for name in data.columns
            if name
            in set(CFG.get("NEGATIVE_RESIDUAL_META_FEATURE_KEYS", []))
        ]
        return list(dict.fromkeys(base + additions))
    if arm in RESIDUAL_STATES_V2_ARMS:
        if arm in RESIDUAL_STATES_V2_DISTILLED_ARMS:
            distilled = [
                name
                for name in data.columns
                if name.startswith("meta_resid_v2_")
                and (
                    arm == "residual_states_v2_distilled_local_interactions"
                    or not name.startswith("meta_resid_v2_local__")
                )
            ]
            additions = list(distilled)
            if arm in {
                "residual_states_v2_distilled_priors",
                "residual_states_v2_distilled_priors_hitrate_context",
            }:
                additions.extend(
                    name
                    for name in data.columns
                    if name.startswith("meta_resid_arch_expected_")
                )
                additions.extend(
                    name
                    for name in (
                        "meta_resid_arch_entropy",
                        "meta_resid_arch_confidence",
                        "meta_resid_arch_support_log1p",
                        "meta_resid_arch_local_model",
                    )
                    if name in data.columns
                )
            if arm == "residual_states_v2_distilled_priors_hitrate_context":
                additions.extend(HIT_SURPRISE_NUMERIC_FEATURES)
            return list(dict.fromkeys(base + additions))
        probabilities = [
            name
            for name in data.columns
            if name.startswith("meta_resid_arch_prob__")
        ]
        priors = [
            name
            for name in data.columns
            if name.startswith("meta_resid_arch_expected_")
        ]
        reliability = [
            name
            for name in (
                "meta_resid_arch_entropy",
                "meta_resid_arch_confidence",
                "meta_resid_arch_support_log1p",
                "meta_resid_arch_local_model",
            )
            if name in data.columns
        ]
        additions: list[str] = []
        if arm == "residual_states_v2_archetype_market_failure_posteriors_only":
            additions.extend(
                name
                for name in data.columns
                if name.startswith("meta_resid_market_arch_")
                and "cluster_id" not in name
            )
            return list(dict.fromkeys(base + additions))
        if arm in {
            "residual_states_v2_probabilities",
            "residual_states_v2_probabilities_priors",
            "residual_states_v2_probabilities_priors_market",
            "residual_states_v2_full_aegmm",
            "residual_states_v2_full_context",
            "residual_states_v2_archetype_market_failure_posteriors",
        }:
            additions.extend(probabilities)
        if arm in {
            "residual_states_v2_priors",
            "residual_states_v2_priors_market",
            "residual_states_v2_priors_hitrate_context",
            "residual_states_v2_priors_outcome_context",
            "residual_states_v2_priors_outcome_context_weighted",
            "residual_states_v2_priors_outcome_context_identity",
            "residual_states_v2_priors_outcome_context_identity_weighted",
            "residual_states_v2_priors_outcome_context_interactions",
            "residual_states_v2_priors_failure_pressure_context",
            "residual_states_v2_priors_failure_pressure_identity_context",
            "residual_states_v2_probabilities_priors",
            "residual_states_v2_probabilities_priors_market",
            "residual_states_v2_full_aegmm",
            "residual_states_v2_full_context",
            "residual_states_v2_archetype_market_failure_posteriors",
            "residual_states_v2_priors_local_models",
            "residual_states_v2_priors_local_models_shrunk",
        }:
            additions.extend(priors)
        if arm in {
            "residual_states_v2_priors_hitrate_context",
            "residual_states_v2_priors_outcome_context",
            "residual_states_v2_priors_outcome_context_weighted",
            "residual_states_v2_priors_outcome_context_identity",
            "residual_states_v2_priors_outcome_context_identity_weighted",
            "residual_states_v2_priors_outcome_context_interactions",
            "residual_states_v2_priors_failure_pressure_context",
            "residual_states_v2_priors_failure_pressure_identity_context",
        }:
            additions.extend(HIT_SURPRISE_NUMERIC_FEATURES)
        if arm in {
            "residual_states_v2_priors_outcome_context",
            "residual_states_v2_priors_outcome_context_weighted",
            "residual_states_v2_priors_outcome_context_identity",
            "residual_states_v2_priors_outcome_context_identity_weighted",
            "residual_states_v2_priors_outcome_context_interactions",
            "residual_states_v2_priors_failure_pressure_context",
            "residual_states_v2_priors_failure_pressure_identity_context",
        }:
            additions.extend(OUTCOME_CONTEXT_FEATURES)
        if arm in {
            "residual_states_v2_priors_failure_pressure_context",
            "residual_states_v2_priors_failure_pressure_identity_context",
        }:
            additions.extend(FAILURE_PRESSURE_CONTEXT_FEATURES)
        if arm in {
            "residual_states_v2_priors_outcome_context_identity",
            "residual_states_v2_priors_outcome_context_identity_weighted",
            "residual_states_v2_priors_outcome_context_interactions",
            "residual_states_v2_priors_failure_pressure_identity_context",
        }:
            additions.extend(
                name for name in data.columns if name.startswith("meta_identity_")
            )
        if arm == "residual_states_v2_priors_outcome_context_interactions":
            additions.extend(_outcome_identity_interaction_columns(data))
        additions.extend(reliability)
        if arm in {
            "residual_states_v2_priors_market",
            "residual_states_v2_probabilities_priors_market",
            "residual_states_v2_full_context",
            "residual_states_v2_archetype_market_failure_posteriors",
        }:
            market_features = [
                name
                for name in data.columns
                if name.startswith("meta_resid_market_")
            ]
            if arm == "residual_states_v2_archetype_market_failure_posteriors":
                market_features = [
                    name
                    for name in market_features
                    if name.startswith("meta_resid_market_arch_")
                    and "cluster_id" not in name
                ]
            additions.extend(market_features)
        if arm in {"residual_states_v2_full_aegmm", "residual_states_v2_full_context"}:
            additions.extend(
                name for name in data.columns if name.startswith("meta_resid_ae_")
            )
        if arm == "residual_states_v2_full_context":
            additions.extend(
                name
                for name in data.columns
                if name.startswith(
                    (
                        "meta_resid_v2_hard_state_",
                        "meta_resid_v2_ae_hard_cluster_",
                        "meta_base_archetype__",
                        "meta_side_",
                        "meta_resid_market_",
                        "meta_resid_v2_market_",
                    )
                )
            )
        return list(dict.fromkeys(base + additions))
    features = base + [name for name in lifecycle_keys if name in data.columns]
    if arm in {"residual_archetypes", "residual_archetypes_ae_gmm"}:
        features += [
            name for name in data.columns if name.startswith("meta_resid_arch_")
        ]
    if arm == "residual_archetypes_ae_gmm":
        features += [
            name for name in residual_ae_gmm_feature_names() if name in data.columns
        ]
    if arm in LOCAL_AEGMM_ARMS:
        features += local_economic_aegmm_feature_names(
            [block.name for block in _local_aegmm_blocks_for_arm(arm)]
        )
    return list(dict.fromkeys(features))


def _local_aegmm_selected_feature_attribution(
    selected: Sequence[str], blocks: Sequence[EconomicAEGMMBlock]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for rank, feature in enumerate(selected, start=1):
        name = str(feature)
        for block in blocks:
            prefix = f"local_econ_aegmm_{block.name}_"
            if not name.startswith(prefix):
                continue
            suffix = name.removeprefix(prefix)
            if suffix.startswith("dae_b16_"):
                kind = "ae_latent"
            elif "posterior" in suffix or suffix == "gmm_cluster_id":
                kind = "gmm_assignment"
            elif suffix.startswith("prob__"):
                kind = "economic_semantic_probability"
            elif suffix.startswith("expected_"):
                kind = "posterior_weighted_train_prior"
            elif "mahal" in suffix or "reconstruction" in suffix or "entropy" in suffix:
                kind = "uncertainty_distance"
            else:
                kind = "state_support"
            rows.append(
                {
                    "selected_rank": int(rank),
                    "feature": name,
                    "block": block.name,
                    "feature_kind": kind,
                }
            )
            break
    return pd.DataFrame(rows)


def _select_arm_features(
    *,
    arm: str,
    data: pd.DataFrame,
    candidates: Sequence[str],
    output_dir: Path,
    seed: int,
    force: bool = False,
) -> tuple[list[str], pd.DataFrame]:
    if arm == "baseline_retrained" or arm in RESIDUAL_STATES_V2_ARMS:
        method = (
            "frozen_reference_plus_negative_residual_context"
            if arm == "negative_residual_full_context"
            else "frozen_reference_plus_residual_v2"
            if arm in RESIDUAL_STATES_V2_ARMS
            else "frozen_reference_contract"
        )
        rows = pd.DataFrame(
            {
                "feature": list(candidates)
                + list(META_POST_SELECTION_OOD_FEATURE_NAMES),
                "selected": True,
                "rank": np.arange(
                    1, len(candidates) + len(META_POST_SELECTION_OOD_FEATURE_NAMES) + 1
                ),
                "feature_selection_method": method,
            }
        )
        return list(candidates) + list(META_POST_SELECTION_OOD_FEATURE_NAMES), rows
    ts = pd.to_datetime(data["__ts__"], utc=True, errors="coerce")
    train_mask = ts.lt(pd.Timestamp("2026-03-01", tz="UTC"))
    valid_mask = ts.ge(pd.Timestamp("2026-03-01", tz="UTC")) & ts.lt(
        pd.Timestamp("2026-04-01", tz="UTC")
    )
    fs_train = data.loc[train_mask].reset_index(drop=True)
    fs_valid = data.loc[valid_mask].reset_index(drop=True)
    fs_train, fs_valid = _add_reference_fold_features(fs_train, fs_valid)
    x_train, x_valid, _ = _matrix_fit_transform(fs_train, fs_valid, candidates)
    selected_path = output_dir / arm / "selected_features.csv"
    cache_path = output_dir / arm / "selected_features.cache.json"
    candidate_contract = {
        "candidates": list(map(str, candidates)),
        "feature_selection_fit_end": "2026-02-28",
        "feature_selection_validation_month": "2026-03",
        "selector": "lgbm_pipeline_staged_side_aware_mda_v1",
    }
    candidate_hash = hashlib.sha256(
        json.dumps(candidate_contract, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    cache_valid = False
    if selected_path.exists() and cache_path.exists() and not force:
        try:
            cached = json.loads(cache_path.read_text())
            cache_valid = (
                str(cached.get("candidate_contract_hash", "")) == candidate_hash
            )
        except (OSError, json.JSONDecodeError):
            cache_valid = False
    if selected_path.exists() and cache_valid and not force:
        rows = pd.read_csv(selected_path)
        selected_mask = (
            rows["selected"].fillna(False).astype(bool)
            if "selected" in rows.columns
            else pd.Series(True, index=rows.index)
        )
        selected = rows.loc[selected_mask, "feature"].astype(str).tolist()
        if not selected:
            raise ValueError(
                f"Cached feature selection contains no selected features: {selected_path}"
            )
        return selected, rows
    if selected_path.exists() and not force:
        print(
            json.dumps(
                {
                    "event": "meta_state_feature_selection_cache_invalidated",
                    "arm": arm,
                    "reason": "candidate_contract_changed_or_legacy_cache",
                }
            ),
            flush=True,
        )
    x_train_sel, _x_valid_sel, selected, _selected_by_side, rows = _select_features_by_lgbm_pipeline(
        x_train,
        x_valid,
        fs_train,
        target_name="ev_frontier",
        top_n=0,
        fold="train_through_2026_02_validate_2026_03",
        seed=int(seed),
    )
    del x_train_sel, _x_valid_sel, x_train, x_valid, fs_train, fs_valid
    selected_path.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(selected_path, index=False)
    cache_path.write_text(
        json.dumps(
            {
                "candidate_contract_hash": candidate_hash,
                "candidate_feature_count": int(len(candidates)),
                "selected_feature_count": int(len(selected)),
                **candidate_contract,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return selected, rows


def _fit_platt(scores: pd.Series, hit: pd.Series) -> LogisticRegression | None:
    x = pd.to_numeric(scores, errors="coerce").to_numpy(dtype=np.float32)
    y = pd.to_numeric(hit, errors="coerce").to_numpy(dtype=np.float32)
    valid = np.isfinite(x) & np.isfinite(y)
    if int(valid.sum()) < 200 or len(np.unique((y[valid] >= 0.5).astype(np.int8))) < 2:
        return None
    model = LogisticRegression(C=0.20, solver="lbfgs", max_iter=300)
    model.fit(x[valid].reshape(-1, 1), (y[valid] >= 0.5).astype(np.int8))
    return model


def _calibrate(model: LogisticRegression | None, scores: pd.Series) -> np.ndarray:
    values = (
        pd.to_numeric(scores, errors="coerce").fillna(0.5).to_numpy(dtype=np.float32)
    )
    if model is None:
        return np.clip(values, 0.0, 1.0)
    return model.predict_proba(values.reshape(-1, 1))[:, 1].astype(np.float32)


def _fit_predict_meta_scores(
    *,
    arm: str,
    x_train: pd.DataFrame,
    target_train: pd.Series,
    train: pd.DataFrame,
    x_valid: pd.DataFrame,
    valid: pd.DataFrame,
    params: dict[str, Any],
    seed: int,
) -> tuple[Any, np.ndarray, dict[str, Any]]:
    """Fit the frozen global model contract, optionally specialized by archetype."""

    global_model = _fit_base_soft_label_model(
        x_train,
        target_train,
        train,
        seed,
        lgbm_params=params,
        sample_weight_multiplier=_residual_v2_sample_weight_multiplier(train, arm),
    )
    if global_model is None:
        return None, np.full(len(valid), np.nan, dtype=np.float32), {}
    global_scores = np.asarray(
        _predict(global_model, x_valid, classifier=False), dtype=np.float32
    )
    if arm not in LOCAL_META_MODEL_ARMS:
        return global_model, global_scores, {"global_rows": int(len(train))}

    train_arch = train.get(
        "archetype_policy_key", pd.Series("missing", index=train.index)
    ).astype(str)
    valid_arch = valid.get(
        "archetype_policy_key", pd.Series("missing", index=valid.index)
    ).astype(str)
    scores = global_scores.copy()
    local_models: dict[str, Any] = {}
    support: dict[str, int] = {}
    blend: dict[str, float] = {}
    for group_idx, archetype in enumerate(sorted(valid_arch.unique().tolist())):
        train_mask = train_arch.eq(archetype)
        valid_mask = valid_arch.eq(archetype)
        n_train = int(train_mask.sum())
        if n_train < 10_000 or int(valid_mask.sum()) == 0:
            continue
        local_model = _fit_base_soft_label_model(
            x_train.loc[train_mask],
            target_train.loc[train_mask],
            train.loc[train_mask],
            seed + 10_007 * (group_idx + 1),
            lgbm_params=params,
        )
        if local_model is None:
            continue
        local_scores = np.asarray(
            _predict(local_model, x_valid.loc[valid_mask], classifier=False),
            dtype=np.float32,
        )
        if arm == "residual_states_v2_priors_local_models_shrunk":
            local_fraction = float(np.clip(n_train / (n_train + 50_000.0), 0.0, 0.90))
        else:
            local_fraction = 1.0
        positions = np.flatnonzero(valid_mask.to_numpy())
        scores[positions] = (
            local_fraction * local_scores
            + (1.0 - local_fraction) * global_scores[positions]
        ).astype(np.float32)
        local_models[str(archetype)] = local_model
        support[str(archetype)] = n_train
        blend[str(archetype)] = local_fraction
    bundle = {"global": global_model, "local_by_archetype": local_models}
    return bundle, scores, {
        "global_rows": int(len(train)),
        "local_train_rows": support,
        "local_blend_fraction": blend,
    }


def _fit_walkforward_alternative_platt(
    *,
    data: pd.DataFrame,
    arm: str,
    month: str,
    raw_selected: Sequence[str],
    selected_features: Sequence[str],
    params: dict[str, Any],
    seed: int,
) -> tuple[LogisticRegression | None, int]:
    """Fit score calibration on the month immediately preceding an OOS fold.

    The temporary model is trained strictly before the calibration month. Its
    calibration-month predictions are therefore OOS, while the final meta
    model remains free to use all rows before the evaluation month.
    """

    calibration_end = pd.Timestamp(pd.Period(month).start_time, tz="UTC")
    calibration_start = calibration_end - pd.offsets.MonthBegin(1)
    timestamps = pd.to_datetime(data["__ts__"], utc=True, errors="coerce")
    fit = data.loc[timestamps.lt(calibration_start)].copy()
    calibration = data.loc[
        timestamps.ge(calibration_start) & timestamps.lt(calibration_end)
    ].copy()
    if len(fit) < 5_000 or len(calibration) < 200:
        return None, 0
    fit, calibration = _add_reference_fold_features(fit, calibration)
    fit_target, _ = _base_soft_label_target(fit)
    fit = fit.loc[fit_target.notna()].copy()
    fit_target = fit_target.loc[fit.index]
    x_fit, x_calibration, _ = _matrix_fit_transform(
        fit, calibration, raw_selected
    )
    ood_state = _fit_ood_state(x_fit, raw_selected)
    x_fit = _apply_ood_state(x_fit, ood_state).reindex(
        columns=selected_features, fill_value=0.0
    )
    x_calibration = _apply_ood_state(x_calibration, ood_state).reindex(
        columns=selected_features, fill_value=0.0
    )
    model, calibration_scores, _ = _fit_predict_meta_scores(
        arm=arm,
        x_train=x_fit,
        target_train=fit_target,
        train=fit,
        x_valid=x_calibration,
        valid=calibration,
        params=params,
        seed=seed,
    )
    if model is None:
        return None, 0
    platt = _fit_platt(
        pd.Series(calibration_scores, index=calibration.index),
        pd.to_numeric(calibration.get("clean_exec"), errors="coerce"),
    )
    return platt, int(len(calibration))


def train_arm_oos(
    *,
    arm: str,
    data: pd.DataFrame,
    selected_features: Sequence[str],
    params: dict[str, Any],
    output_dir: Path,
    seed: int,
    eval_months: Sequence[str] = EVAL_MONTHS,
    artifact_tag: str = "",
    local_aegmm_state: LocalEconomicAEGMM | None = None,
    alternative_hit_calibration: str = "walkforward",
    regime_calibration_mode: str = "strict",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw_selected = [
        name
        for name in selected_features
        if name not in META_POST_SELECTION_OOD_FEATURE_NAMES
    ]
    ts = pd.to_datetime(data["__ts__"], utc=True, errors="coerce")
    predictions: list[pd.DataFrame] = []
    fold_rows: list[dict[str, Any]] = []
    last_model = None
    last_medians: dict[str, float] = {}
    last_ood_state: dict[str, Any] = {}
    last_platt_alt = None
    last_platt_ref = None
    for fold_idx, month in enumerate(eval_months):
        start = pd.Timestamp(pd.Period(month).start_time, tz="UTC")
        end = pd.Timestamp((pd.Period(month) + 1).start_time, tz="UTC")
        train = data.loc[ts.lt(start)].copy()
        valid = data.loc[ts.ge(start) & ts.lt(end)].copy()
        required_generated = [
            name for name in raw_selected if name.startswith("meta_resid_")
        ]
        if required_generated and arm not in RESIDUAL_STATES_V2_ARMS:
            train = train[train[required_generated].notna().all(axis=1)]
            valid = valid[valid[required_generated].notna().all(axis=1)]
        train, valid = _add_reference_fold_features(train, valid)
        target_train, target_col = _base_soft_label_target(train)
        train = train[target_train.notna()].copy()
        target_train = target_train.loc[train.index]
        if len(train) < 5_000 or len(valid) < 100:
            continue
        platt_alt = None
        calibration_rows = 0
        calibration_contract = "raw_score_no_alternative_calibration"
        if alternative_hit_calibration == "walkforward":
            platt_alt, calibration_rows = _fit_walkforward_alternative_platt(
                data=data,
                arm=arm,
                month=month,
                raw_selected=raw_selected,
                selected_features=selected_features,
                params=params,
                seed=seed + fold_idx * 101 - 17,
            )
            calibration_contract = "prior_month_oos_predictions"
        x_train, x_valid, medians = _matrix_fit_transform(train, valid, raw_selected)
        ood_state = _fit_ood_state(x_train, raw_selected)
        x_train = _apply_ood_state(x_train, ood_state).reindex(
            columns=selected_features, fill_value=0.0
        )
        x_valid = _apply_ood_state(x_valid, ood_state).reindex(
            columns=selected_features, fill_value=0.0
        )
        model, score_valid, model_fit_manifest = _fit_predict_meta_scores(
            arm=arm,
            x_train=x_train,
            target_train=target_train,
            train=train,
            x_valid=x_valid,
            valid=valid,
            params=params,
            seed=seed + fold_idx * 101,
        )
        if model is None:
            raise RuntimeError(f"Meta model fit failed for arm={arm} month={month}")
        if (
            len(score_valid) != len(valid)
            or not np.isfinite(np.asarray(score_valid, dtype=np.float32)).any()
        ):
            raise RuntimeError(f"Meta prediction failed for arm={arm} month={month}")
        if alternative_hit_calibration == "train_in_sample":
            calibration_model = model.get("global") if isinstance(model, dict) else model
            calibration_scores = np.asarray(
                _predict(calibration_model, x_train, classifier=False),
                dtype=np.float32,
            )
            platt_alt = _fit_platt(
                pd.Series(calibration_scores, index=train.index),
                pd.to_numeric(train.get("clean_exec"), errors="coerce"),
            )
            calibration_rows = int(len(train))
            calibration_contract = "train_fold_in_sample_scores"
        elif alternative_hit_calibration not in {"walkforward", "none"}:
            raise ValueError(
                "alternative_hit_calibration must be one of "
                "'walkforward', 'train_in_sample', or 'none'"
            )
        hit_train = pd.to_numeric(train.get("clean_exec"), errors="coerce")
        platt_ref = _fit_platt(train.get("score_meta_base_soft_label"), hit_train)
        keep = [
            name
            for name in (
                "__ts__",
                "__symbol__",
                "side_name",
                "archetype_policy_key",
                "archetype_label_family",
                "ev_after_1pct",
                "exec_margin",
                "clean_exec",
                "dirty_positive",
                "first_touch_bad_mae_1r",
                "full_path_bad_mae_1r",
                "timeout",
                "score_meta_base_soft_label",
            )
            if name in valid.columns
        ]
        local_state_report_suffixes = (
            "gmm_cluster_id",
            "gmm_posterior_max",
            "gmm_entropy",
            "expected_ev",
            "expected_hit_surprise",
            "expected_bad_mae",
            "expected_timeout",
            "local_model",
        )
        keep.extend(
            name
            for name in valid.columns
            if name.startswith("local_econ_aegmm_")
            and (name.endswith(local_state_report_suffixes) or "_prob__" in name)
            and name not in keep
        )
        if arm in RESIDUAL_STATES_V2_ARMS:
            keep.extend(
                name
                for name in valid.columns
                if name.startswith("meta_resid_") and name not in keep
            )
        regime_artifact_path = default_regime_ev_calibration_artifact()
        if regime_artifact_path is None:
            raise FileNotFoundError("The promoted regime EV calibration artifact is missing")
        regime_required = regime_required_feature_columns(
            load_regime_ev_calibration(regime_artifact_path)
        )
        keep.extend(name for name in regime_required if name in valid.columns)
        scored = valid.loc[:, keep].copy()
        scored["calendar_month"] = str(month)
        # W-SUN periods start on Monday; W-MON would label Tuesday as the start.
        scored["week_start"] = (
            pd.to_datetime(scored["__ts__"], utc=True)
            .dt.to_period("W-SUN")
            .dt.start_time.dt.tz_localize("UTC")
        )
        scored["score_current_reference_uncalibrated"] = pd.to_numeric(
            scored["score_meta_base_soft_label"], errors="coerce"
        ).astype(np.float32)
        scored["score_alternative_uncalibrated"] = np.asarray(
            score_valid, dtype=np.float32
        )
        calibration_status: dict[str, Any] = {}
        scored["score_current_reference"], current_status = (
            _maybe_apply_frozen_regime_calibration_to_score(
                scored,
                source_col="score_current_reference_uncalibrated",
                adjusted_col="score_current_reference",
                mode=regime_calibration_mode,
            )
        )
        scored["score_alternative"], alternative_status = (
            _maybe_apply_frozen_regime_calibration_to_score(
                scored,
                source_col="score_alternative_uncalibrated",
                adjusted_col="score_alternative",
                mode=regime_calibration_mode,
            )
        )
        calibration_status["current_reference"] = current_status
        calibration_status["alternative"] = alternative_status
        scored["hit_prob_current_reference"] = _calibrate(
            platt_ref, scored["score_current_reference_uncalibrated"]
        )
        scored["hit_prob_alternative"] = _calibrate(
            platt_alt, scored["score_alternative_uncalibrated"]
        )
        scored["oos_fold"] = str(month)
        predictions.append(scored)
        fold_rows.append(
            {
                "arm": arm,
                "month": month,
                "train_rows": int(len(train)),
                "valid_rows": int(len(valid)),
                "selected_features": int(len(selected_features)),
                "target_column": target_col,
                "calibration_rows": int(calibration_rows),
                "calibration_contract": calibration_contract,
                "regime_calibration_mode": regime_calibration_mode,
                "regime_calibration_status": json.dumps(
                    _json_safe(calibration_status), sort_keys=True
                ),
                "model_fit_manifest": json.dumps(
                    _json_safe(model_fit_manifest), sort_keys=True
                ),
            }
        )
        last_model = model
        last_medians = medians
        last_ood_state = ood_state
        last_platt_alt = platt_alt
        last_platt_ref = platt_ref
        fold_model_dir = output_dir / arm / "fold_models"
        fold_model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": model,
                "selected_features": list(selected_features),
                "raw_selected_features": raw_selected,
                "feature_medians": medians,
                "ood_state": ood_state,
                "params": params,
                "hit_calibrator": platt_alt,
                "reference_hit_calibrator": platt_ref,
                "local_economic_aegmm_state": local_aegmm_state,
                "fit_through": str(
                    pd.Timestamp(pd.Period(month).start_time, tz="UTC")
                    - pd.Timedelta(nanoseconds=1)
                ),
                "role": "alternative_meta_oos_fold_model",
            },
            fold_model_dir / f"score_{month}.joblib",
        )
        print(
            json.dumps(
                {
                    "event": "alternative_meta_fold_complete",
                    "arm": arm,
                    "month": month,
                    "train_rows": len(train),
                    "valid_rows": len(valid),
                    "features": len(selected_features),
                }
            ),
            flush=True,
        )
        del train, valid, x_train, x_valid, model, scored
        gc.collect()
    out = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    arm_dir = output_dir / arm
    arm_dir.mkdir(parents=True, exist_ok=True)
    tag = f"_{artifact_tag.strip('_')}" if artifact_tag else ""
    out.to_parquet(
        arm_dir / f"oos_predictions{tag}.parquet", index=False, compression="zstd"
    )
    pd.DataFrame(fold_rows).to_csv(arm_dir / f"folds{tag}.csv", index=False)
    artifact = {
        "model": last_model,
        "selected_features": list(selected_features),
        "raw_selected_features": raw_selected,
        "feature_medians": last_medians,
        "ood_state": last_ood_state,
        "params": params,
        "hit_calibrator": last_platt_alt,
        "reference_hit_calibrator": last_platt_ref,
        "local_economic_aegmm_state": local_aegmm_state,
        "fit_through": str(
            pd.Timestamp(pd.Period(eval_months[-1]).start_time, tz="UTC")
            - pd.Timedelta(nanoseconds=1)
        )
        if eval_months
        else None,
        "role": "alternative_meta_oos_fold_model",
    }
    joblib.dump(artifact, arm_dir / f"latest_oos_fold_model{tag}.joblib")
    if local_aegmm_state is not None and last_model is not None:
        bundle = LocalEconomicAEGMMModelBundle(
            model=last_model,
            local_aegmm=local_aegmm_state,
            selected_features=list(selected_features),
            raw_selected_features=raw_selected,
            feature_medians=last_medians,
            ood_state=last_ood_state,
            fit_through=artifact["fit_through"],
            metadata={"arm": arm, "role": "direct_primary_meta_model"},
        )
        joblib.dump(
            bundle,
            arm_dir / f"local_economic_aegmm_model_bundle{tag}.joblib",
            compress=3,
        )
        _write_json(
            arm_dir / f"local_economic_aegmm_model_bundle{tag}.manifest.json",
            bundle.manifest(),
        )
    return out, {"arm": arm, "folds": fold_rows, "prediction_rows": int(len(out))}


def _selection_mask(
    frame: pd.DataFrame, score_col: str, fraction: float, group_cols: Sequence[str]
) -> pd.Series:
    score = pd.to_numeric(frame[score_col], errors="coerce")
    keys = [frame[name].astype(str) for name in group_cols if name in frame.columns]
    if keys:
        group = pd.concat(keys, axis=1).agg("|".join, axis=1)
        rank = score.groupby(group, sort=False).rank(method="first", pct=True)
    else:
        rank = score.rank(method="first", pct=True)
    return rank.ge(1.0 - float(fraction)).fillna(False)


def _apply_frozen_regime_calibration_to_score(
    frame: pd.DataFrame,
    *,
    source_col: str,
    adjusted_col: str,
) -> pd.Series:
    artifact_path = default_regime_ev_calibration_artifact()
    if artifact_path is None:
        raise FileNotFoundError("The promoted regime EV calibration artifact is missing")
    artifact = load_regime_ev_calibration(artifact_path)
    required = regime_required_feature_columns(artifact)
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise ValueError(
            "Exact regime-calibrated policy evaluation is missing features: "
            f"{missing}"
        )
    calibrated = apply_regime_ev_calibration(
        frame,
        artifact,
        source_score_col=source_col,
        adjusted_score_col=adjusted_col,
        side_col="side_name",
        archetype_col="archetype_policy_key",
        copy=True,
    )
    return pd.to_numeric(calibrated[adjusted_col], errors="coerce").astype(np.float32)


def _maybe_apply_frozen_regime_calibration_to_score(
    frame: pd.DataFrame,
    *,
    source_col: str,
    adjusted_col: str,
    mode: str,
) -> tuple[pd.Series, dict[str, Any]]:
    """Apply promoted regime calibration, or record an explicit research fallback.

    The residual archetype enhancement runner is often used with compact
    comparison frames that intentionally omit the full replay feature universe.
    In strict mode we keep the production-like contract and raise.  In fallback
    or skip mode the ablation remains runnable, but the manifest/fold rows make
    clear that exact regime-calibrated policy scoring was not available.
    """

    if mode not in {"strict", "fallback", "skip"}:
        raise ValueError("regime_calibration_mode must be strict, fallback, or skip")
    raw = pd.to_numeric(frame[source_col], errors="coerce").astype(np.float32)
    if mode == "skip":
        return raw, {
            "status": "skipped",
            "mode": mode,
            "source_col": source_col,
            "adjusted_col": adjusted_col,
        }
    try:
        calibrated = _apply_frozen_regime_calibration_to_score(
            frame,
            source_col=source_col,
            adjusted_col=adjusted_col,
        )
        return calibrated, {
            "status": "applied",
            "mode": mode,
            "source_col": source_col,
            "adjusted_col": adjusted_col,
        }
    except (FileNotFoundError, ValueError) as exc:
        if mode == "strict":
            raise
        return raw, {
            "status": "fallback_uncalibrated",
            "mode": mode,
            "source_col": source_col,
            "adjusted_col": adjusted_col,
            "reason": str(exc),
        }


def append_reachable_ev_policy_decisions(
    predictions: pd.DataFrame,
    *,
    policy_path: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply the promoted causal 8d reachable-EV admission to both score streams."""

    if not policy_path.exists():
        raise FileNotFoundError(f"Threshold-basis policy is missing: {policy_path}")
    from scripts.score_frozen_champion_full_history import (
        _apply_causal_reachable_ev_policy,
    )

    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    expected_id = (
        "ev_target_archetype_reachable_match_current_activity_8d_hr_off_regimecal_v1"
    )
    if str(policy.get("policy_id")) != expected_id:
        raise ValueError(
            f"Unexpected threshold-basis policy: {policy.get('policy_id')}"
        )
    out = predictions.copy()
    for suffix, score_col in (
        ("current_reference", "score_current_reference"),
        ("alternative", "score_alternative"),
    ):
        evaluated = _apply_causal_reachable_ev_policy(
            out,
            policy=policy,
            score_col=score_col,
            preserve_materialized_policy=False,
        )
        for target, source in (
            (f"policy_selected_{suffix}", "threshold_basis_selected"),
            (f"policy_rank_{suffix}", "threshold_basis_rank_score"),
            (f"policy_dynamic_ev_target_{suffix}", "threshold_basis_dynamic_ev_target"),
            (
                f"policy_dynamic_score_threshold_{suffix}",
                "threshold_basis_dynamic_score_threshold",
            ),
            (
                f"policy_recent_reference_rows_{suffix}",
                "threshold_basis_recent_reference_rows",
            ),
        ):
            out[target] = evaluated[source].to_numpy(copy=False)
    return out, {
        "policy_path": str(policy_path),
        "policy_id": expected_id,
        "window_days": int(policy.get("window_days", 8)),
        "hr_rank50": bool(policy.get("hr_rank50", False)),
        "regime_calibration_policy_id": str(
            policy.get("regime_calibration_policy_id", "")
        ),
        "current_selected_rows": int(out["policy_selected_current_reference"].sum()),
        "alternative_selected_rows": int(out["policy_selected_alternative"].sum()),
    }


def _metric_record(
    frame: pd.DataFrame, mask: pd.Series, selector: str, arm: str, fraction: float
) -> dict[str, Any]:
    selected = frame.loc[mask]
    ev = pd.to_numeric(selected.get("ev_after_1pct"), errors="coerce")
    hit = pd.to_numeric(selected.get("clean_exec"), errors="coerce")
    calibrated_col = (
        "hit_prob_current_reference"
        if selector == "current_reference"
        else "hit_prob_alternative"
    )
    predicted = pd.to_numeric(selected.get(calibrated_col), errors="coerce")
    return {
        "arm": arm,
        "selector": selector,
        "fraction": float(fraction),
        "candidate_rows": int(len(frame)),
        "selected_rows": int(len(selected)),
        "trades_per_day": float(
            len(selected)
            / max(pd.to_datetime(frame["__ts__"], utc=True).dt.date.nunique(), 1)
        ),
        "mean_ev_after_1pct": float(ev.mean()) if ev.notna().any() else np.nan,
        "sum_ev_after_1pct": float(ev.sum()) if ev.notna().any() else np.nan,
        "positive_ev_rate": float(ev.gt(0.0).mean()) if ev.notna().any() else np.nan,
        "clean_exec_precision": float(hit.mean()) if hit.notna().any() else np.nan,
        "dirty_positive_rate": float(
            pd.to_numeric(selected.get("dirty_positive"), errors="coerce").mean()
        ),
        "first_touch_bad_mae_rate": float(
            pd.to_numeric(
                selected.get("first_touch_bad_mae_1r"), errors="coerce"
            ).mean()
        ),
        "full_path_bad_mae_rate": float(
            pd.to_numeric(selected.get("full_path_bad_mae_1r"), errors="coerce").mean()
        ),
        "timeout_rate": float(
            pd.to_numeric(selected.get("timeout"), errors="coerce").mean()
        ),
        "mean_hit_surprise": float((hit - predicted).mean())
        if hit.notna().any()
        else np.nan,
        "mean_negative_hit_surprise": float((predicted - hit).clip(lower=0.0).mean())
        if hit.notna().any()
        else np.nan,
        "mean_positive_hit_surprise": float((hit - predicted).clip(lower=0.0).mean())
        if hit.notna().any()
        else np.nan,
    }


def metrics_by_scope(predictions: pd.DataFrame, arm: str) -> pd.DataFrame:
    scopes = {
        "overall": [],
        "month": ["calendar_month"],
        "week": ["week_start"],
        "side": ["side_name"],
        "archetype": ["archetype_policy_key"],
        "side_archetype": ["side_name", "archetype_policy_key"],
        "month_side_archetype": ["calendar_month", "side_name", "archetype_policy_key"],
        "week_side_archetype": ["week_start", "side_name", "archetype_policy_key"],
    }
    rows: list[dict[str, Any]] = []
    policy_columns_available = {
        "policy_selected_current_reference",
        "policy_selected_alternative",
    }.issubset(predictions.columns)
    selection_masks: dict[tuple[str, float], pd.Series] = {}
    for selector, score_col in (
        ("current_reference", "score_current_reference"),
        (arm, "score_alternative"),
    ):
        for fraction in (0.10, 0.20, 0.30):
            policy_column = (
                "policy_selected_current_reference"
                if selector == "current_reference"
                else "policy_selected_alternative"
            )
            if fraction == 0.10 and policy_column in predictions.columns:
                selection_masks[(selector, fraction)] = predictions[
                    policy_column
                ].fillna(False).astype(bool)
            else:
                selection_masks[(selector, fraction)] = _selection_mask(
                    predictions, score_col, fraction, ["__ts__"]
                )
    for scope, group_cols in scopes.items():
        grouped: Iterable[tuple[Any, pd.DataFrame]]
        grouped = (
            [((), predictions)]
            if not group_cols
            else predictions.groupby(group_cols, dropna=False, sort=True)
        )
        for keys, group in grouped:
            if not isinstance(keys, tuple):
                keys = (keys,)
            for selector, score_col in (
                ("current_reference", "score_current_reference"),
                (arm, "score_alternative"),
            ):
                for fraction in (0.10, 0.20, 0.30):
                    mask = selection_masks[(selector, fraction)].reindex(
                        group.index, fill_value=False
                    )
                    record = _metric_record(group, mask, selector, arm, fraction)
                    record["selection_basis"] = (
                        "ev_target_archetype_reachable_match_current_activity_8d_hr_off_regimecal_v1"
                        if fraction == 0.10 and policy_columns_available
                        else (
                            "global_within_timestamp"
                            if fraction == 0.10
                            else "within_timestamp_rank_diagnostic_only"
                        )
                    )
                    record["scope"] = scope
                    for name, value in zip(group_cols, keys, strict=False):
                        record[name] = value
                    rows.append(record)
    return pd.DataFrame(rows)


def append_walkforward_historical_ranks(
    burnin: pd.DataFrame,
    oos: pd.DataFrame,
    *,
    eval_months: Sequence[str] = EVAL_MONTHS,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Map fold scores to causal side-aware historical percentiles."""

    prior = burnin.copy()
    frames: list[pd.DataFrame] = []
    folds: list[dict[str, Any]] = []
    for month in eval_months:
        valid = oos[oos["calendar_month"].astype(str).eq(str(month))].copy()
        if valid.empty:
            continue
        current_state = HistoricalScoreRankReference(
            score_col="score_current_reference"
        ).fit(prior)
        alternative_state = HistoricalScoreRankReference(
            score_col="score_alternative"
        ).fit(prior)
        valid["historical_rank_current_reference"] = current_state.transform(
            valid, "score_current_reference"
        )
        valid["historical_rank_alternative"] = alternative_state.transform(
            valid, "score_alternative"
        )
        folds.append(
            {
                "month": str(month),
                "prior_rows": int(len(prior)),
                "valid_rows": int(len(valid)),
                "prior_end": str(pd.to_datetime(prior["__ts__"], utc=True).max()),
                "current_reference": current_state.manifest(),
                "alternative_reference": alternative_state.manifest(),
            }
        )
        frames.append(valid)
        # Scores may update each fold, matching the model available at that time.
        prior = pd.concat([prior, valid], ignore_index=True, copy=False)
    return (
        pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(),
        folds,
    )


def historical_rank_metrics_by_scope(
    predictions: pd.DataFrame, arm: str
) -> pd.DataFrame:
    scopes = {
        "overall": [],
        "month": ["calendar_month"],
        "week": ["week_start"],
        "side": ["side_name"],
        "archetype": ["archetype_policy_key"],
        "month_side_archetype": [
            "calendar_month",
            "side_name",
            "archetype_policy_key",
        ],
        "week_side_archetype": [
            "week_start",
            "side_name",
            "archetype_policy_key",
        ],
    }
    rows: list[dict[str, Any]] = []
    for scope, group_cols in scopes.items():
        grouped: Iterable[tuple[Any, pd.DataFrame]] = (
            [((), predictions)]
            if not group_cols
            else predictions.groupby(group_cols, dropna=False, sort=True)
        )
        for keys, group in grouped:
            if not isinstance(keys, tuple):
                keys = (keys,)
            for selector, rank_col in (
                ("current_reference", "historical_rank_current_reference"),
                (arm, "historical_rank_alternative"),
            ):
                for fraction in (0.10, 0.20, 0.30):
                    mask = pd.to_numeric(group[rank_col], errors="coerce").ge(
                        1.0 - fraction
                    )
                    record = _metric_record(group, mask, selector, arm, fraction)
                    record["selection_basis"] = (
                        "causal_side_historical_score_percentile"
                    )
                    record["scope"] = scope
                    for name, value in zip(group_cols, keys, strict=False):
                        record[name] = value
                    rows.append(record)
    return pd.DataFrame(rows)


def local_aegmm_metrics_by_state(predictions: pd.DataFrame, arm: str) -> pd.DataFrame:
    cluster_columns = [
        name
        for name in predictions.columns
        if name.startswith("local_econ_aegmm_") and name.endswith("gmm_cluster_id")
    ]
    if not cluster_columns:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for selector, score_col in (
        ("current_reference", "score_current_reference"),
        (arm, "score_alternative"),
    ):
        for fraction in (0.10, 0.20, 0.30):
            selected_mask = _selection_mask(
                predictions, score_col, fraction, ["__ts__"]
            )
            selected = predictions.loc[selected_mask]
            for cluster_col in cluster_columns:
                block = cluster_col.removeprefix("local_econ_aegmm_").removesuffix(
                    "_gmm_cluster_id"
                )
                scope_groups = {
                    "state": [cluster_col],
                    "month_state": ["calendar_month", cluster_col],
                    "side_archetype_state": [
                        "side_name",
                        "archetype_policy_key",
                        cluster_col,
                    ],
                    "month_side_archetype_state": [
                        "calendar_month",
                        "side_name",
                        "archetype_policy_key",
                        cluster_col,
                    ],
                }
                for scope, group_cols in scope_groups.items():
                    for keys, group in selected.groupby(
                        group_cols, observed=True, dropna=False, sort=True
                    ):
                        if not isinstance(keys, tuple):
                            keys = (keys,)
                        record = _metric_record(
                            group,
                            pd.Series(True, index=group.index),
                            selector,
                            arm,
                            fraction,
                        )
                        record.update(
                            {
                                "scope": scope,
                                "selection_basis": "global_within_timestamp",
                                "state_block": block,
                                "state_cluster": int(float(keys[-1])),
                            }
                        )
                        for name, value in zip(
                            group_cols[:-1], keys[:-1], strict=False
                        ):
                            record[name] = value
                        rows.append(record)
    return pd.DataFrame(rows)


def local_aegmm_state_transfer_metrics(
    predictions: pd.DataFrame,
    arm: str,
    state_catalog: pd.DataFrame,
) -> pd.DataFrame:
    """Compare frozen state priors with realized OOS state behavior.

    This is reporting only.  ``state_catalog`` is fitted from pre-cutoff rows,
    while every realized metric below is calculated solely from an OOS scored
    ledger.  The report deliberately keeps the all-candidate population apart
    from the actual global top-k tails, so a state is not called transferable
    merely because it works outside the trades the policy can admit.
    """

    cluster_columns = [
        name
        for name in predictions.columns
        if name.startswith("local_econ_aegmm_") and name.endswith("gmm_cluster_id")
    ]
    required = {
        "__ts__",
        "side_name",
        "archetype_policy_key",
        "ev_after_1pct",
        "clean_exec",
        "dirty_positive",
        "first_touch_bad_mae_1r",
        "timeout",
    }
    if not cluster_columns or not required.issubset(predictions.columns):
        return pd.DataFrame()

    catalog = state_catalog.copy(deep=False)
    required_catalog = {
        "model_key",
        "cluster",
        "semantic",
        "posterior_support",
        "ev",
        "clean_positive",
        "dirty_positive",
        "bad_mae",
        "timeout",
    }
    if not required_catalog.issubset(catalog.columns):
        return pd.DataFrame()
    catalog = catalog.loc[:, list(required_catalog)].drop_duplicates(
        ["model_key", "cluster"], keep="last"
    )
    catalog = catalog.rename(
        columns={
            "semantic": "train_semantic",
            "posterior_support": "train_state_support",
            "ev": "train_prior_ev",
            "clean_positive": "train_prior_clean_rate",
            "dirty_positive": "train_prior_dirty_rate",
            "bad_mae": "train_prior_bad_mae_rate",
            "timeout": "train_prior_timeout_rate",
        }
    )
    catalog["cluster"] = pd.to_numeric(catalog["cluster"], errors="coerce").astype(
        "Int16"
    )

    scopes: list[tuple[str, str | None, float | None]] = [
        ("all_candidates", None, None),
        ("current_reference_top10", "score_current_reference", 0.10),
        (f"{arm}_top10", "score_alternative", 0.10),
        (f"{arm}_top20", "score_alternative", 0.20),
        (f"{arm}_top30", "score_alternative", 0.30),
    ]
    rows: list[pd.DataFrame] = []
    for scope, score_col, fraction in scopes:
        if score_col is None:
            selected = predictions
        elif score_col not in predictions.columns:
            continue
        else:
            mask = _selection_mask(predictions, score_col, float(fraction), ["__ts__"])
            selected = predictions.loc[mask]
        if selected.empty:
            continue

        for cluster_col in cluster_columns:
            block = cluster_col.removeprefix("local_econ_aegmm_").removesuffix(
                "_gmm_cluster_id"
            )
            prefix = f"local_econ_aegmm_{block}_"
            enabled_col = f"{prefix}enabled"
            if enabled_col in selected.columns:
                enabled = (
                    pd.to_numeric(selected[enabled_col], errors="coerce")
                    .fillna(0.0)
                    .gt(0.5)
                )
                current = selected.loc[enabled]
            else:
                current = selected
            if current.empty:
                continue

            columns = [
                "side_name",
                "archetype_policy_key",
                cluster_col,
                "ev_after_1pct",
                "clean_exec",
                "dirty_positive",
                "first_touch_bad_mae_1r",
                "timeout",
            ]
            local_col = f"{prefix}local_model"
            expected_cols = [
                f"{prefix}expected_ev",
                f"{prefix}expected_clean_positive",
                f"{prefix}expected_dirty_positive",
                f"{prefix}expected_bad_mae",
                f"{prefix}expected_timeout",
            ]
            columns.extend(
                name for name in [local_col, *expected_cols] if name in current.columns
            )
            work = current.loc[:, columns].copy()
            work["state_cluster"] = (
                pd.to_numeric(work[cluster_col], errors="coerce")
                .round()
                .astype("Int16")
            )
            work = work.loc[work["state_cluster"].notna()].copy()
            if work.empty:
                continue
            work["_hit_surprise"] = pd.to_numeric(
                work["clean_exec"], errors="coerce"
            ) - pd.to_numeric(
                current.loc[work.index, score_col]
                if score_col is not None
                else current.loc[work.index, "score_alternative"],
                errors="coerce",
            )
            if local_col in work.columns:
                local_model = (
                    pd.to_numeric(work[local_col], errors="coerce").fillna(0.0).ge(0.5)
                )
            else:
                local_model = pd.Series(False, index=work.index)
            side_token = work["side_name"].astype(str).str.lower()
            arch_token = work["archetype_policy_key"].astype(str)
            work["model_key"] = np.where(
                local_model.to_numpy(),
                "local::" + side_token + "::" + arch_token + f"::{block}",
                "side::" + side_token + f"::{block}",
            )
            aggregation: dict[str, tuple[str, str]] = {
                "selected_rows": ("ev_after_1pct", "size"),
                "mean_ev_after_1pct": ("ev_after_1pct", "mean"),
                "clean_exec_precision": ("clean_exec", "mean"),
                "dirty_positive_rate": ("dirty_positive", "mean"),
                "first_touch_bad_mae_rate": ("first_touch_bad_mae_1r", "mean"),
                "timeout_rate": ("timeout", "mean"),
                "mean_hit_surprise": ("_hit_surprise", "mean"),
            }
            for expected in expected_cols:
                if expected in work.columns:
                    aggregation[f"oos_{expected.removeprefix(prefix)}"] = (
                        expected,
                        "mean",
                    )
            grouped = (
                work.groupby(
                    ["side_name", "archetype_policy_key", "model_key", "state_cluster"],
                    observed=True,
                    sort=True,
                )
                .agg(**aggregation)
                .reset_index()
            )
            parent = (
                work.groupby(["side_name", "archetype_policy_key"], observed=True)
                .agg(
                    parent_ev=("ev_after_1pct", "mean"),
                    parent_clean_rate=("clean_exec", "mean"),
                    parent_dirty_rate=("dirty_positive", "mean"),
                    parent_bad_mae_rate=("first_touch_bad_mae_1r", "mean"),
                    parent_timeout_rate=("timeout", "mean"),
                )
                .reset_index()
            )
            grouped = grouped.merge(
                parent,
                on=["side_name", "archetype_policy_key"],
                how="left",
                validate="many_to_one",
            ).merge(
                catalog,
                left_on=["model_key", "state_cluster"],
                right_on=["model_key", "cluster"],
                how="left",
                validate="many_to_one",
            )
            grouped["state_block"] = block
            grouped["scope"] = scope
            grouped["selector"] = (
                "all_candidates"
                if score_col is None
                else score_col.removeprefix("score_")
            )
            grouped["fraction"] = np.nan if fraction is None else float(fraction)
            grouped["oos_ev_lift_vs_side_archetype"] = (
                grouped["mean_ev_after_1pct"] - grouped["parent_ev"]
            )
            grouped["oos_clean_lift_vs_side_archetype"] = (
                grouped["clean_exec_precision"] - grouped["parent_clean_rate"]
            )
            grouped["oos_dirty_lift_vs_side_archetype"] = (
                grouped["dirty_positive_rate"] - grouped["parent_dirty_rate"]
            )
            grouped["oos_bad_mae_lift_vs_side_archetype"] = (
                grouped["first_touch_bad_mae_rate"] - grouped["parent_bad_mae_rate"]
            )
            grouped["prior_ev_error"] = (
                grouped["mean_ev_after_1pct"] - grouped["train_prior_ev"]
            )
            expected_ev = "oos_expected_ev"
            if expected_ev in grouped.columns:
                grouped["posterior_prior_ev_error"] = (
                    grouped["mean_ev_after_1pct"] - grouped[expected_ev]
                )
                expected_parent = grouped.groupby(
                    ["side_name", "archetype_policy_key"], observed=True
                )[expected_ev].transform("mean")
                expected_lift = grouped[expected_ev] - expected_parent
                realized_lift = grouped["oos_ev_lift_vs_side_archetype"]
                grouped["posterior_prior_ev_lift_sign_agrees"] = (
                    np.sign(expected_lift) == np.sign(realized_lift)
                ).astype(np.float32)
            rows.append(grouped)
    if not rows:
        return pd.DataFrame()
    output = pd.concat(rows, ignore_index=True, sort=False)
    numeric = output.select_dtypes(include=["float64"]).columns
    output.loc[:, numeric] = output.loc[:, numeric].apply(
        pd.to_numeric, downcast="float"
    )
    return output


def surprise_calendar(
    predictions: pd.DataFrame, arm: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    work = predictions.copy()
    work["date"] = pd.to_datetime(work["__ts__"], utc=True).dt.floor("D")
    calendar_parts: list[pd.DataFrame] = []
    for selector, score_col, prob_col in (
        ("current_reference", "score_current_reference", "hit_prob_current_reference"),
        (arm, "score_alternative", "hit_prob_alternative"),
    ):
        policy_column = (
            "policy_selected_current_reference"
            if selector == "current_reference"
            else "policy_selected_alternative"
        )
        selections = {
            "raw_global_within_timestamp_top10": _selection_mask(
                work, score_col, 0.10, ["__ts__"]
            )
        }
        if policy_column in work.columns:
            selections[
                "ev_target_archetype_reachable_match_current_activity_8d_hr_off_regimecal_v1"
            ] = work[policy_column].fillna(False).astype(bool)
        for selection_basis, mask in selections.items():
            selected = work.loc[mask].copy()
            selected["hit_surprise"] = pd.to_numeric(
                selected["clean_exec"], errors="coerce"
            ) - pd.to_numeric(selected[prob_col], errors="coerce")
            selected["negative_hit_surprise"] = selected["hit_surprise"].clip(
                upper=0.0
            )
            selected["positive_hit_surprise"] = selected["hit_surprise"].clip(
                lower=0.0
            )
            daily = (
                selected.groupby(
                    ["date", "side_name", "archetype_policy_key"], dropna=False
                )
                .agg(
                    rows=("clean_exec", "size"),
                    hit_rate=("clean_exec", "mean"),
                    mean_hit_surprise=("hit_surprise", "mean"),
                    mean_negative_hit_surprise=("negative_hit_surprise", "mean"),
                    mean_positive_hit_surprise=("positive_hit_surprise", "mean"),
                    mean_ev_after_1pct=("ev_after_1pct", "mean"),
                )
                .reset_index()
            )
            daily["selector"] = selector
            daily["arm"] = arm
            daily["selection_basis"] = selection_basis
            calendar_parts.append(daily)
    calendar = pd.concat(calendar_parts, ignore_index=True)
    autocorr_rows: list[dict[str, Any]] = []
    for (selection_basis, selector, side, arch), group in calendar.groupby(
        ["selection_basis", "selector", "side_name", "archetype_policy_key"],
        dropna=False,
    ):
        ordered = group.sort_values("date", kind="stable")
        series = ordered["mean_hit_surprise"].astype(float)
        negative = ordered["mean_negative_hit_surprise"].astype(float)
        positive = ordered["mean_positive_hit_surprise"].astype(float)
        autocorr_rows.append(
            {
                "arm": arm,
                "selection_basis": selection_basis,
                "selector": selector,
                "side_name": side,
                "archetype_policy_key": arch,
                "days": int(len(series)),
                "surprise_autocorr_lag1": float(series.autocorr(1))
                if len(series) >= 3
                else np.nan,
                "surprise_autocorr_lag3": float(series.autocorr(3))
                if len(series) >= 5
                else np.nan,
                "negative_surprise_autocorr_lag1": float(negative.autocorr(1))
                if len(negative) >= 3
                else np.nan,
                "positive_surprise_autocorr_lag1": float(positive.autocorr(1))
                if len(positive) >= 3
                else np.nan,
            }
        )
    autocorr = pd.DataFrame(autocorr_rows)
    base = calendar[calendar["selector"].eq("current_reference")].copy()
    alt = calendar[calendar["selector"].eq(arm)].copy()
    keys = ["selection_basis", "date", "side_name", "archetype_policy_key"]
    comparison = base.merge(alt, on=keys, suffixes=("_base", "_alt"), how="inner")
    comparison["surprise_abs_improvement"] = (
        comparison["mean_hit_surprise_base"].abs()
        - comparison["mean_hit_surprise_alt"].abs()
    )
    comparison["ev_delta"] = (
        comparison["mean_ev_after_1pct_alt"] - comparison["mean_ev_after_1pct_base"]
    )
    if not comparison.empty:
        comparison["baseline_tail_threshold"] = comparison.groupby(
            ["selection_basis", "side_name", "archetype_policy_key"]
        )["mean_hit_surprise_base"].transform(lambda s: s.abs().quantile(0.90))
        comparison["baseline_high_surprise"] = (
            comparison["mean_hit_surprise_base"]
            .abs()
            .ge(comparison["baseline_tail_threshold"])
        )
        comparison["high_surprise_significantly_improved"] = comparison[
            "surprise_abs_improvement"
        ].ge(0.20 * comparison["mean_hit_surprise_base"].abs())
    return calendar, autocorr, comparison


def _experiment_score(
    metrics: pd.DataFrame, autocorr: pd.DataFrame, calendar_cmp: pd.DataFrame, arm: str
) -> dict[str, Any]:
    raw_basis = "raw_global_within_timestamp_top10"
    policy_basis = (
        "ev_target_archetype_reachable_match_current_activity_8d_hr_off_regimecal_v1"
    )
    raw_autocorr = (
        autocorr[autocorr["selection_basis"].eq(raw_basis)]
        if "selection_basis" in autocorr.columns
        else autocorr
    )
    policy_autocorr = (
        autocorr[autocorr["selection_basis"].eq(policy_basis)]
        if "selection_basis" in autocorr.columns
        else pd.DataFrame(columns=autocorr.columns)
    )
    overall = metrics[(metrics["scope"].eq("overall")) & (metrics["fraction"].eq(0.10))]
    base = overall[overall["selector"].eq("current_reference")].iloc[0]
    alt = overall[overall["selector"].eq(arm)].iloc[0]
    week = metrics[(metrics["scope"].eq("week")) & (metrics["fraction"].eq(0.10))]
    month = metrics[(metrics["scope"].eq("month")) & (metrics["fraction"].eq(0.10))]
    base_worst = float(
        week[week["selector"].eq("current_reference")]["mean_ev_after_1pct"].min()
    )
    alt_worst = float(week[week["selector"].eq(arm)]["mean_ev_after_1pct"].min())
    base_worst_month = float(
        month[month["selector"].eq("current_reference")]["mean_ev_after_1pct"].min()
    )
    alt_worst_month = float(
        month[month["selector"].eq(arm)]["mean_ev_after_1pct"].min()
    )
    base_ac = (
        pd.to_numeric(
            raw_autocorr.loc[
                raw_autocorr["selector"].eq("current_reference"),
                "surprise_autocorr_lag1",
            ],
            errors="coerce",
        )
        .abs()
        .mean()
    )
    alt_ac = (
        pd.to_numeric(
            raw_autocorr.loc[
                raw_autocorr["selector"].eq(arm), "surprise_autocorr_lag1"
            ],
            errors="coerce",
        )
        .abs()
        .mean()
    )
    component_autocorr: dict[str, tuple[float, float]] = {}
    for column in (
        "negative_surprise_autocorr_lag1",
        "positive_surprise_autocorr_lag1",
    ):
        base_component = (
            pd.to_numeric(
                raw_autocorr.loc[
                    raw_autocorr["selector"].eq("current_reference"), column
                ],
                errors="coerce",
            )
            .abs()
            .mean()
        )
        alt_component = (
            pd.to_numeric(
                raw_autocorr.loc[raw_autocorr["selector"].eq(arm), column],
                errors="coerce",
            )
            .abs()
            .mean()
        )
        component_autocorr[column] = (float(base_component), float(alt_component))
    base_negative_ac, alt_negative_ac = component_autocorr[
        "negative_surprise_autocorr_lag1"
    ]
    base_positive_ac, alt_positive_ac = component_autocorr[
        "positive_surprise_autocorr_lag1"
    ]
    tail_source = (
        calendar_cmp[calendar_cmp["selection_basis"].eq(raw_basis)]
        if "selection_basis" in calendar_cmp.columns
        else calendar_cmp
    )
    tail = tail_source[tail_source.get("baseline_high_surprise", False)].copy()
    high_improved = (
        float(
            tail.get(
                "high_surprise_significantly_improved", pd.Series(dtype=float)
            ).mean()
        )
        if len(tail)
        else np.nan
    )
    score = (
        100.0 * (float(alt["mean_ev_after_1pct"]) - float(base["mean_ev_after_1pct"]))
        + 30.0 * (alt_worst - base_worst)
        + 20.0 * (alt_worst_month - base_worst_month)
        + 0.15 * (float(base_ac) - float(alt_ac))
        + 0.20 * (base_negative_ac - alt_negative_ac)
        + 0.10 * (base_positive_ac - alt_positive_ac)
        + 0.10 * (0.0 if not np.isfinite(high_improved) else high_improved)
    )
    return {
        "arm": arm,
        "objective_score": float(score),
        "top10_ev_base": float(base["mean_ev_after_1pct"]),
        "top10_ev_alt": float(alt["mean_ev_after_1pct"]),
        "top10_ev_delta": float(alt["mean_ev_after_1pct"] - base["mean_ev_after_1pct"]),
        "worst_week_ev_base": base_worst,
        "worst_week_ev_alt": alt_worst,
        "worst_week_ev_delta": float(alt_worst - base_worst),
        "worst_month_ev_base": base_worst_month,
        "worst_month_ev_alt": alt_worst_month,
        "worst_month_ev_delta": float(alt_worst_month - base_worst_month),
        "mean_abs_surprise_autocorr_base": float(base_ac),
        "mean_abs_surprise_autocorr_alt": float(alt_ac),
        "mean_abs_surprise_autocorr_delta": float(alt_ac - base_ac),
        "mean_abs_negative_surprise_autocorr_base": base_negative_ac,
        "mean_abs_negative_surprise_autocorr_alt": alt_negative_ac,
        "mean_abs_negative_surprise_autocorr_delta": float(
            alt_negative_ac - base_negative_ac
        ),
        "mean_abs_positive_surprise_autocorr_base": base_positive_ac,
        "mean_abs_positive_surprise_autocorr_alt": alt_positive_ac,
        "mean_abs_positive_surprise_autocorr_delta": float(
            alt_positive_ac - base_positive_ac
        ),
        "autocorr_objective_selection_basis": raw_basis,
        "policy_mean_abs_surprise_autocorr_base": float(
            pd.to_numeric(
                policy_autocorr.loc[
                    policy_autocorr["selector"].eq("current_reference"),
                    "surprise_autocorr_lag1",
                ],
                errors="coerce",
            )
            .abs()
            .mean()
        ),
        "policy_mean_abs_surprise_autocorr_alt": float(
            pd.to_numeric(
                policy_autocorr.loc[
                    policy_autocorr["selector"].eq(arm),
                    "surprise_autocorr_lag1",
                ],
                errors="coerce",
            )
            .abs()
            .mean()
        ),
        "high_surprise_period_improvement_rate": high_improved,
        "unimproved_high_surprise_periods": int(
            (
                ~tail.get("high_surprise_significantly_improved", pd.Series(dtype=bool))
            ).sum()
        )
        if len(tail)
        else 0,
    }


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    eval_months = _parse_months(args.eval_months)
    if args.prepared_dataset is not None:
        if not args.prepared_dataset.exists():
            raise FileNotFoundError(args.prepared_dataset)
        reference_selected, params, reference_manifest = _reference_contract(
            args.reference_dir
        )
        dataset_path = args.prepared_dataset
        dataset_manifest = {
            "path": str(dataset_path),
            "status": "external_prepared_dataset",
            "reference_manifest": str(args.reference_dir / "manifest.json"),
            "reference_selected_features": reference_selected,
            "reference_model_params": params,
            "reference_model_preserved": True,
        }
    else:
        dataset_path, dataset_manifest, reference_selected, params = prepare_dataset(
            handoff=args.handoff,
            reference_dir=args.reference_dir,
            feature_root=args.feature_root,
            output_dir=args.output_dir,
            force=args.force_prepare,
        )
    if args.prepare_only:
        return {"status": "prepared", "dataset": dataset_manifest}
    data = pd.read_parquet(dataset_path)
    data["__ts__"] = pd.to_datetime(data["__ts__"], utc=True, errors="coerce")
    data["archetype_policy_key"] = (
        data.get("archetype_policy_key", "missing").astype(str).fillna("missing")
    )
    data = (
        _ensure_base_score(_downcast(data))
        .sort_values(["__ts__", "__symbol__", "side_name"], kind="stable")
        .reset_index(drop=True)
    )
    data = _append_cross_sectional_geometry(data)
    data = _append_meta_identity_features(data)
    requested_arms = [name.strip() for name in args.arms.split(",") if name.strip()]
    unknown = sorted(set(requested_arms) - set(MODEL_ARMS))
    if unknown:
        raise ValueError(f"Unknown arms: {unknown}; available={MODEL_ARMS}")
    lifecycle_keys = [
        name
        for name in dict.fromkeys(
            [
                *CFG.get("CRASH_LIFECYCLE_NEW_FEATURE_KEYS", []),
                *META_MARKET_STATE_FEATURES,
                *DEFAULT_RELATIVE_FEATURES,
            ]
        )
        if name in data.columns
    ]
    state_training_data = data
    aegmm_archive_manifest: dict[str, Any] | None = None
    if any(arm in LOCAL_AEGMM_ARMS for arm in requested_arms):
        # The compact comparison frame is OOS-only.  Always construct a
        # historical state-fit archive instead of silently fitting on the
        # evaluation months or falling back to an under-supported state model.
        archive_source = args.aegmm_training_archive or args.handoff
        archive_path, aegmm_archive_manifest = _prepare_local_aegmm_training_archive(
            archive_path=archive_source,
            feature_root=args.feature_root,
            output_dir=args.output_dir,
            state_feature_keys=lifecycle_keys,
            force=bool(args.force_prepare),
        )
        archive = pd.read_parquet(archive_path)
        archive["__ts__"] = pd.to_datetime(archive["__ts__"], utc=True, errors="coerce")
        archive["archetype_policy_key"] = (
            archive.get("archetype_policy_key", "missing").astype(str).fillna("missing")
        )
        archive = _append_cross_sectional_geometry(
            _ensure_base_score(_downcast(archive))
        )
        state_training_data = pd.concat(
            [archive, data], ignore_index=True, sort=False, copy=False
        )
        dedupe_keys = [
            name
            for name in ("__ts__", "__symbol__", "side_name", "archetype_policy_key")
            if name in state_training_data.columns
        ]
        state_training_data = (
            state_training_data.sort_values(dedupe_keys, kind="stable")
            .drop_duplicates(dedupe_keys, keep="last")
            .reset_index(drop=True)
        )
    current_raw = [
        name
        for name in reference_selected
        if name not in META_POST_SELECTION_OOD_FEATURE_NAMES and name in data.columns
    ]
    existing_aegmm = [
        name
        for name in data.columns
        if any(h.lower() in name.lower() for h in AE_GMM_HINTS)
        and name not in OUTCOME_COLUMNS
    ]
    recognizer_inputs = list(
        dict.fromkeys(current_raw + lifecycle_keys + existing_aegmm)
    )
    raw_residual = pd.DataFrame()
    ae_residual = pd.DataFrame()
    external_residual = pd.DataFrame()
    v2_data: pd.DataFrame | None = None
    if any(name in RESIDUAL_STATES_V2_ARMS for name in requested_arms):
        if args.external_residual_features is None or not args.external_residual_features.exists():
            raise FileNotFoundError(
                "Residual-state v2 arms require --external-residual-features"
            )
        available = set(_parquet_columns(args.external_residual_features))
        include_full_aegmm = any(
            name in requested_arms
            for name in ("residual_states_v2_full_aegmm", "residual_states_v2_full_context")
        )
        external_columns = [
            name
            for name in [*KEY_COLUMNS, *sorted(available)]
            if name in available
            and (
                name in KEY_COLUMNS
                or name.startswith("meta_resid_arch_")
                or name.startswith("meta_resid_market_")
                or (include_full_aegmm and name.startswith("meta_resid_ae_"))
                or (
                    "negative_residual_full_context" in requested_arms
                    and name
                    in set(
                            CFG.get("NEGATIVE_RESIDUAL_META_FEATURE_KEYS", [])
                    )
                )
            )
        ]
        external_residual = pd.read_parquet(
            args.external_residual_features,
            columns=list(dict.fromkeys(external_columns)),
        )
        external_residual["__ts__"] = pd.to_datetime(
            external_residual["__ts__"], utc=True, errors="coerce"
        )
        v2_data = _merge_residual_features(
            data,
            external_residual,
            fill_unknown=True,
        )
        v2_data = _append_residual_state_v2_composites(v2_data)
    if "residual_archetypes" in requested_arms:
        raw_residual, _, _ = build_walkforward_residual_features(
            data=data,
            candidate_features=recognizer_inputs,
            output_dir=args.output_dir,
            use_ae_gmm=False,
            force=args.force_residual,
        )
    if "residual_archetypes_ae_gmm" in requested_arms:
        ae_residual, _, _ = build_walkforward_residual_features(
            data=data,
            candidate_features=recognizer_inputs,
            output_dir=args.output_dir,
            use_ae_gmm=True,
            force=args.force_residual,
        )
    scorecards: list[dict[str, Any]] = []
    all_metrics: list[pd.DataFrame] = []
    all_calendar: list[pd.DataFrame] = []
    all_autocorr: list[pd.DataFrame] = []
    all_period_cmp: list[pd.DataFrame] = []
    arm_manifests: list[dict[str, Any]] = []
    for arm_idx, arm in enumerate(requested_arms):
        print(json.dumps({"event": "meta_state_arm_start", "arm": arm}), flush=True)
        arm_data = data
        local_aegmm_manifest: dict[str, Any] | None = None
        local_aegmm_state: LocalEconomicAEGMM | None = None
        if arm == "residual_archetypes":
            arm_data = _merge_residual_features(data, raw_residual)
        elif arm == "residual_archetypes_ae_gmm":
            arm_data = _merge_residual_features(data, ae_residual)
        elif arm in RESIDUAL_STATES_V2_ARMS:
            if v2_data is None:
                raise RuntimeError("Residual-state v2 data was not materialized")
            arm_data = v2_data
        elif arm in LOCAL_AEGMM_ARMS:
            arm_data, local_aegmm_state, local_aegmm_manifest = (
                _append_frozen_local_aegmm(
                    data,
                    state_training_data=state_training_data,
                    arm=arm,
                    output_dir=args.output_dir,
                    seed=args.seed + arm_idx * 100_003,
                    force=args.force_residual,
                    fit_start=args.aegmm_fit_start,
                    fit_end=args.aegmm_fit_end,
                    full_train_fit=bool(args.aegmm_full_fit),
                    eval_months=eval_months,
                )
            )
        candidates = _arm_candidate_features(
            arm, arm_data, reference_selected, lifecycle_keys
        )
        selected, selection_rows = _select_arm_features(
            arm=arm,
            data=arm_data,
            candidates=candidates,
            output_dir=args.output_dir,
            seed=args.seed + arm_idx * 101,
            force=args.force_feature_selection,
        )
        print(
            json.dumps(
                {
                    "event": "meta_state_feature_selection_complete",
                    "arm": arm,
                    "candidate_features": int(len(candidates)),
                    "selected_features": int(len(selected)),
                }
            ),
            flush=True,
        )
        arm_dir = args.output_dir / arm
        arm_dir.mkdir(parents=True, exist_ok=True)
        selection_rows.to_csv(arm_dir / "selected_features.csv", index=False)
        if arm in LOCAL_AEGMM_ARMS:
            _local_aegmm_selected_feature_attribution(
                selected, _local_aegmm_blocks_for_arm(arm)
            ).to_csv(arm_dir / "selected_local_aegmm_features.csv", index=False)
        predictions, arm_manifest = train_arm_oos(
            arm=arm,
            data=arm_data,
            selected_features=selected,
            params=params,
            output_dir=args.output_dir,
            seed=args.seed + arm_idx * 1009,
            eval_months=eval_months,
            local_aegmm_state=local_aegmm_state,
            alternative_hit_calibration=str(args.alternative_hit_calibration),
            regime_calibration_mode=str(args.regime_calibration_mode),
        )
        print(
            json.dumps(
                {
                    "event": "meta_state_arm_oos_complete",
                    "arm": arm,
                    "prediction_rows": int(len(predictions)),
                }
            ),
            flush=True,
        )
        predictions, threshold_policy_manifest = append_reachable_ev_policy_decisions(
            predictions,
            policy_path=args.threshold_basis_policy,
        )
        predictions.to_parquet(
            arm_dir / "oos_predictions_policy_scored.parquet",
            index=False,
            compression="zstd",
        )
        metrics = metrics_by_scope(predictions, arm)
        state_metrics = local_aegmm_metrics_by_state(predictions, arm)
        state_transfer = (
            local_aegmm_state_transfer_metrics(
                predictions,
                arm,
                local_aegmm_state.catalog_,
            )
            if local_aegmm_state is not None
            else pd.DataFrame()
        )
        calendar, autocorr, period_cmp = surprise_calendar(predictions, arm)
        metrics.to_csv(arm_dir / "metrics_by_scope.csv", index=False)
        if not state_metrics.empty:
            state_metrics.to_csv(
                arm_dir / "metrics_by_local_aegmm_state.csv", index=False
            )
        if not state_transfer.empty:
            state_transfer.to_csv(
                arm_dir / "local_aegmm_state_transfer_metrics.csv", index=False
            )
        calendar.to_csv(arm_dir / "hit_surprise_calendar.csv", index=False)
        autocorr.to_csv(arm_dir / "hit_surprise_autocorrelation.csv", index=False)
        period_cmp.to_csv(arm_dir / "high_surprise_period_comparison.csv", index=False)
        scorecard = _experiment_score(metrics, autocorr, period_cmp, arm)
        scorecards.append(scorecard)
        all_metrics.append(metrics)
        all_calendar.append(calendar)
        all_autocorr.append(autocorr)
        all_period_cmp.append(period_cmp.assign(arm=arm))
        arm_manifest.update(
            {
                "candidate_features": candidates,
                "selected_features": selected,
                "selected_feature_count": len(selected),
                "feature_selection_fit_end": "2026-02-28",
                "feature_selection_validation_month": "2026-03",
                "eval_months": list(eval_months),
                "threshold_basis_policy": threshold_policy_manifest,
                "model_params_source": str(args.reference_dir / "manifest.json"),
                "current_meta_model_overwritten": False,
                "local_economic_aegmm": local_aegmm_manifest,
                "external_residual_features": (
                    str(args.external_residual_features)
                    if arm in RESIDUAL_STATES_V2_ARMS
                    else None
                ),
                "alternative_hit_calibration": str(args.alternative_hit_calibration),
                "regime_calibration_mode": str(args.regime_calibration_mode),
            }
        )
        _write_json(arm_dir / "manifest.json", arm_manifest)
        arm_manifests.append(arm_manifest)
        del (
            arm_data,
            predictions,
            metrics,
            state_metrics,
            state_transfer,
            calendar,
            autocorr,
            period_cmp,
        )
        gc.collect()
    scorecard_df = pd.DataFrame(scorecards).sort_values(
        "objective_score", ascending=False, kind="stable"
    )
    scorecard_df.to_csv(args.output_dir / "experiment_scorecard.csv", index=False)
    pd.concat(all_metrics, ignore_index=True).to_csv(
        args.output_dir / "all_metrics_by_scope.csv", index=False
    )
    pd.concat(all_calendar, ignore_index=True).to_csv(
        args.output_dir / "all_hit_surprise_calendar.csv", index=False
    )
    pd.concat(all_autocorr, ignore_index=True).to_csv(
        args.output_dir / "all_hit_surprise_autocorrelation.csv", index=False
    )
    pd.concat(all_period_cmp, ignore_index=True).to_csv(
        args.output_dir / "all_high_surprise_period_comparison.csv", index=False
    )
    winner = str(scorecard_df.iloc[0]["arm"]) if not scorecard_df.empty else None
    manifest = {
        "schema": "train_meta_residual_archetype_enhancement_v1",
        "winner": winner,
        "scorecards": scorecards,
        "dataset": dataset_manifest,
        "arms": arm_manifests,
        "eval_months": list(eval_months),
        "feature_selection": (
            "once with train rows before 2026-03 and 2026-03 validation; "
            "canonical lgbm_pipeline staged selector; no hard feature-count cap"
        ),
        "model_params": "frozen from current reference meta model",
        "alternative_hit_calibration": str(args.alternative_hit_calibration),
        "base_model_retrained": False,
        "current_meta_model_overwritten": False,
        "local_aegmm_training_archive": (
            str((aegmm_archive_manifest or {}).get("hydrated_archive", dataset_path))
        ),
        "local_aegmm_training_archive_manifest": aegmm_archive_manifest,
        "local_aegmm_fit_start": str(args.aegmm_fit_start),
        "local_aegmm_fit_end_exclusive": str(args.aegmm_fit_end),
        "local_aegmm_full_train_fit": bool(args.aegmm_full_fit),
        "transaction_cost": "ev_after_1pct includes 1% round-trip cost",
        "leakage_contract": {
            "residual_labels": "frozen current-meta OOS predictions only",
            "residual_recognizers": "monthly expanding train; OOS outcomes rejected",
            "feature_selection": "data through February 2026, March validation, frozen for configured OOS months",
            "evaluation": "configured expanding-window OOS months only",
        },
    }
    _write_json(args.output_dir / "manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoff", type=Path, default=DEFAULT_HANDOFF)
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE_DIR)
    parser.add_argument("--feature-root", type=Path, default=DEFAULT_FEATURE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--prepared-dataset",
        type=Path,
        default=None,
        help="Reuse an existing point-in-time prepared meta dataset without rebuilding it.",
    )
    parser.add_argument(
        "--external-residual-features",
        type=Path,
        default=DEFAULT_EXTERNAL_RESIDUAL_FEATURES,
    )
    parser.add_argument(
        "--threshold-basis-policy",
        type=Path,
        default=DEFAULT_THRESHOLD_BASIS_POLICY,
        help=(
            "Frozen reachable-EV 8d HR-off regime-calibrated policy used for "
            "top10 ranking assessment."
        ),
    )
    parser.add_argument(
        "--aegmm-training-archive",
        type=Path,
        default=None,
        help=(
            "Optional 2024+ train-only archetype archive with pre-entry state features, "
            "side/base archetype, frozen base score, and realized training outcomes."
        ),
    )
    parser.add_argument("--aegmm-fit-start", default="2025-02-01")
    parser.add_argument("--aegmm-fit-end", default="2026-03-01")
    parser.add_argument(
        "--aegmm-full-fit",
        action="store_true",
        help=(
            "Fit AE and GMM on every resolved row before --aegmm-fit-end. "
            "Use after the state design is selected; OOS rows remain excluded."
        ),
    )
    parser.add_argument("--arms", default=",".join(MODEL_ARMS))
    parser.add_argument(
        "--eval-months",
        default=",".join(EVAL_MONTHS),
        help=(
            "Comma-separated YYYY-MM OOS months. Include an earlier burn-in "
            "month when downstream sequential calibration needs prior OOS "
            "alternative scores."
        ),
    )
    parser.add_argument(
        "--alternative-hit-calibration",
        choices=("walkforward", "train_in_sample", "none"),
        default="walkforward",
        help=(
            "How to map alternative meta scores to hit probabilities for "
            "hit-surprise reporting. 'walkforward' is strict but slow because "
            "it trains an auxiliary prior-month model; 'train_in_sample' uses "
            "the fold train scores only; 'none' reports clipped raw scores."
        ),
    )
    parser.add_argument(
        "--regime-calibration-mode",
        choices=("strict", "fallback", "skip"),
        default="strict",
        help=(
            "How to apply the promoted regime EV calibration to residual-ablation "
            "score streams. 'strict' raises when exact replay features are absent; "
            "'fallback' records the missing-feature reason and uses uncalibrated "
            "scores; 'skip' always uses uncalibrated scores."
        ),
    )
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--force-prepare", action="store_true")
    parser.add_argument("--force-residual", action="store_true")
    parser.add_argument("--force-feature-selection", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_experiment(args)
    print(
        json.dumps(
            _json_safe(
                {
                    "status": result.get("status", "complete"),
                    "output_dir": str(args.output_dir),
                    "winner": result.get("winner"),
                }
            ),
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
