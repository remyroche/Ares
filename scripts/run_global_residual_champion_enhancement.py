#!/usr/bin/env python3
"""Greedily enhance the current meta champion with local residual states.

This is a native champion revision experiment, not an external correction
model.  The base stream, meta soft label, current selected features, model
parameters, costs, and candidate rows stay fixed while representation blocks
are added.  Every revision is fit once through the configured cutoff.  April to
June is the selection period and July is reported only for the final revision.
Each side x inference-archetype partition owns an independent frozen scaler,
AE/MLP, GMM, enrichment map, and optional temporal state sequence.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import zlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements import lgbm_pipeline as lp  # noqa: E402
from extreme_price_movements.global_residual_latent_state import (  # noqa: E402
    ENCODER_PRESETS,
    GLOBAL_RESIDUAL_SIGNATURE_BASES,
    GlobalGMMStateModel,
    GlobalResidualSignatureEncoder,
    GMMGridConfig,
    ResidualEncoderConfig,
    SideArchetypeStatePriors,
    add_causal_phase_state_features,
    add_temporal_state_features,
    archetype_state_token,
    build_global_residual_signature,
    prepare_archetype_state_partition,
    select_partition_state_features,
)
from scripts.run_s52_train_meta_regime_handoff_smoke import (  # noqa: E402
    META_POST_SELECTION_OOD_FEATURE_NAMES,
    _base_soft_label_target,
    _base_style_weights_for_soft_label,
    _feature_selection_label_context,
    _fit_base_soft_label_model,
    _predict,
)
from scripts.run_train_meta_residual_archetype_enhancement import (  # noqa: E402
    _add_reference_fold_features,
    _apply_ood_state,
    _calibrate,
    _fit_ood_state,
    _fit_platt,
    _matrix_fit_transform,
    _reference_contract,
    metrics_by_scope,
)
from scripts.score_compare_meta_residual_july_oos import (  # noqa: E402
    _append_store_features,
)

REFERENCE_ROOT = Path(
    "data_perp/reports/"
    "s59_h5_2025start_monthly_v4_base_configfull_mdafs120_hpo150_"
    "largestfold_oos15_ae3000_nocrossfit_k34567_payload300k_20260706"
)
DEFAULT_REFERENCE_DIR = (
    REFERENCE_ROOT
    / "train_meta_regime_handoff_singlehead_base_soft_lgbmpipeline_auto_hpo150_"
    "oos15_top30_hpo45k_20260706_v5" / "best_full_oos_fixedfs_streamed_v1"
)
DISCOVERY_ROOT = Path("data_perp/reports/global_residual_state_discovery_20260711_v1")
CHAMPION_HISTORY_ROOT = Path(
    "data_perp/reports/train_meta_residual_archetype_enhancement_20260711_v1/"
    "champion_frozen_single_source_202501_20260710"
)
DEFAULT_COMPACT = Path(
    "data_perp/reports/train_meta_residual_archetype_enhancement_20260711_v1/"
    "cache/compact_reference_with_lifecycle.parquet"
)
DEFAULT_LEDGER = CHAMPION_HISTORY_ROOT / "frozen_champion_single_source_ledger.parquet"
DEFAULT_JULY_SOURCE = (
    CHAMPION_HISTORY_ROOT / "prediction_shards/predictions_2026-07.parquet"
)
DEFAULT_STATES = (
    DISCOVERY_ROOT / "global_side_latent_states/side_timestamp_market_states.parquet"
)
DEFAULT_FEATURE_ROOT = Path("data_perp/features/20260710_170000")
DEFAULT_OUTPUT = DISCOVERY_ROOT / "champion_greedy_enhancement"

OUTCOME_COLUMNS = (
    "ev_after_1pct",
    "exec_margin",
    "clean_exec",
    "dirty_positive",
    "first_touch_bad_mae_1r",
    "full_path_bad_mae_1r",
    "timeout",
)
KEY_COLUMNS = ("__ts__", "__symbol__", "side_name", "archetype_policy_key")
FIXED_REFERENCE_SCORE = "__fixed_reference_score__"
FIXED_REFERENCE_HIT_PROB = "__fixed_reference_hit_probability__"


def _safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_safe(item) for item in value]
    if isinstance(value, (Path, pd.Timestamp, pd.Timedelta)):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (pd.Series, pd.Index)):
        return value.tolist()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_safe(dict(payload)), indent=2, sort_keys=True), encoding="utf-8"
    )


def _emit(event: str, **payload: Any) -> None:
    print(json.dumps(_safe({"event": event, **payload}), sort_keys=True), flush=True)


def _parquet_columns(path: Path) -> list[str]:
    return [str(name) for name in pq.ParquetFile(path).schema_arrow.names]


def _downcast(frame: pd.DataFrame) -> pd.DataFrame:
    for name in frame.select_dtypes(include=["float64"]).columns:
        frame[name] = pd.to_numeric(frame[name], downcast="float")
    for name in frame.select_dtypes(include=["int64"]).columns:
        frame[name] = pd.to_numeric(frame[name], downcast="integer")
    return frame


def _time_spread_sample(frame: pd.DataFrame, max_rows: int, seed: int) -> pd.DataFrame:
    if max_rows <= 0 or len(frame) <= max_rows:
        return frame
    ordered = frame.sort_values("__ts__", kind="stable")
    positions = np.array_split(np.arange(len(ordered), dtype=np.int64), 3)
    rng = np.random.default_rng(seed)
    per_part = max(1, max_rows // 3)
    selected: list[np.ndarray] = []
    for part in positions:
        take = min(len(part), per_part)
        selected.append(np.sort(rng.choice(part, size=take, replace=False)))
    remaining = max_rows - sum(len(values) for values in selected)
    if remaining > 0:
        used = np.concatenate(selected)
        pool = np.setdiff1d(
            np.arange(len(ordered), dtype=np.int64), used, assume_unique=False
        )
        if len(pool):
            selected.append(
                np.sort(rng.choice(pool, size=min(remaining, len(pool)), replace=False))
            )
    keep = np.sort(np.concatenate(selected))
    return ordered.iloc[keep].reset_index(drop=True)


def _purged_fit_boundaries(
    cutoff: pd.Timestamp, purge_hours: float
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return row-label and whole-day residual-signature fit boundaries."""
    train_fit_end = cutoff - pd.Timedelta(hours=max(0.0, float(purge_hours)))
    return train_fit_end, train_fit_end.floor("D")


def _load_comparison_data(
    compact_path: Path,
    ledger_path: Path,
    july_path: Path,
    feature_root: Path,
    reference_features: Sequence[str],
    *,
    data_start: pd.Timestamp | None,
    evaluation_end: pd.Timestamp,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    available = set(_parquet_columns(compact_path))
    raw_reference = [
        name
        for name in reference_features
        if name not in META_POST_SELECTION_OOD_FEATURE_NAMES and name in available
    ]
    required = [
        name
        for name in (
            *KEY_COLUMNS,
            "archetype_label_family",
            "source_tag",
            "selected_top30",
            "score",
            "score_meta_base_soft_label",
            "hit_probability",
            "__first_touch_target_soft__",
            *OUTCOME_COLUMNS,
        )
        if name in available
    ]
    filters = (
        [("__ts__", ">=", data_start.to_pydatetime())]
        if data_start is not None
        else None
    )
    historical = pd.read_parquet(
        compact_path,
        columns=list(dict.fromkeys(required + raw_reference)),
        filters=filters,
    )
    historical["__ts__"] = pd.to_datetime(
        historical["__ts__"], utc=True, errors="coerce"
    )
    historical_mask = historical["__ts__"].lt(pd.Timestamp("2026-07-01", tz="UTC"))
    if data_start is not None:
        historical_mask &= historical["__ts__"].ge(data_start)
    historical = historical.loc[historical_mask]

    july_available = set(_parquet_columns(july_path))
    july_columns = [
        name
        for name in (
            "row_id",
            *KEY_COLUMNS,
            "archetype_label_family",
            "source_tag",
            "score",
            "base_score",
            "score_meta_base_soft_label",
            "hit_probability",
            "selected_top30",
            *OUTCOME_COLUMNS,
        )
        if name in july_available
    ]
    july = pd.read_parquet(july_path, columns=july_columns)
    july["__ts__"] = pd.to_datetime(july["__ts__"], utc=True, errors="coerce")
    july = july.loc[
        july["__ts__"].ge(pd.Timestamp("2026-07-01", tz="UTC"))
        & july["__ts__"].lt(evaluation_end)
    ].copy()
    if "score" not in july and "base_score" in july:
        july["score"] = pd.to_numeric(july["base_score"], errors="coerce").astype(
            np.float32
        )
    if "selected_top30" not in july:
        july["selected_top30"] = True
    july, july_coverage = _append_store_features(july, feature_root, raw_reference)
    for name in historical.columns:
        if name not in july.columns:
            july[name] = np.nan
    data = pd.concat(
        [historical, july.reindex(columns=historical.columns)],
        ignore_index=True,
        sort=False,
        copy=False,
    )
    data["side_name"] = data["side_name"].astype(str).str.lower()
    data["archetype_policy_key"] = data["archetype_policy_key"].astype(str)
    data = (
        _downcast(data)
        .sort_values(["__ts__", "__symbol__", "side_name"], kind="stable")
        .reset_index(drop=True)
    )
    return data, {
        "historical_rows": int(len(historical)),
        "comparison_data_start_inclusive": (
            str(data_start) if data_start is not None else None
        ),
        "july_rows": int(len(july)),
        "july_outcome_rows": int(
            pd.to_numeric(july["ev_after_1pct"], errors="coerce").notna().sum()
            if "ev_after_1pct" in july
            else 0
        ),
        "july_hours": int(july["__ts__"].nunique()),
        "july_source": str(july_path),
        "raw_reference_features": raw_reference,
        "july_feature_coverage": july_coverage,
    }


def _load_states_with_signature(
    states_path: Path, ledger_path: Path
) -> tuple[pd.DataFrame, dict[str, Any]]:
    states = pd.read_parquet(states_path)
    states["__ts__"] = pd.to_datetime(states["__ts__"], utc=True, errors="coerce")
    ledger_columns = [
        name
        for name in (
            *KEY_COLUMNS,
            "selected_for_monitor",
            "threshold_basis_selected",
            "score_meta_base_soft_label",
            "hit_probability",
            *OUTCOME_COLUMNS,
        )
        if name in set(_parquet_columns(ledger_path))
    ]
    ledger = pd.read_parquet(ledger_path, columns=ledger_columns)
    ledger["__ts__"] = pd.to_datetime(ledger["__ts__"], utc=True, errors="coerce")
    if "selected_for_monitor" not in ledger and "threshold_basis_selected" in ledger:
        ledger["selected_for_monitor"] = ledger["threshold_basis_selected"]
    signature, signature_manifest = build_global_residual_signature(ledger)
    old_signature = [name for name in states if name.startswith("target_signature_")]
    states = states.drop(columns=old_signature, errors="ignore").merge(
        signature, on="__ts__", how="left", validate="many_to_one"
    )
    states, phase_manifest = add_causal_phase_state_features(states)
    signature_manifest = {
        **signature_manifest,
        "causal_phase_state_features": phase_manifest,
    }
    return states.sort_values(["side_name", "__ts__"], kind="stable").reset_index(
        drop=True
    ), signature_manifest


def _state_input_features(states: pd.DataFrame) -> list[str]:
    return [
        name
        for name in states.select_dtypes(include=[np.number, "bool"]).columns
        if not name.startswith(("target_", "placebo_target_"))
        and name not in {"global_state_id"}
    ]


def _latent_state_partitions(
    data: pd.DataFrame,
    states: pd.DataFrame,
    *,
    fit_end: pd.Timestamp,
) -> list[dict[str, Any]]:
    """Discover only side/archetype partitions observable before fitting."""
    train_pairs = (
        data.loc[data["__ts__"].lt(fit_end), ["side_name", "archetype_policy_key"]]
        .dropna()
        .astype(str)
        .drop_duplicates()
    )
    partitions: list[dict[str, Any]] = []
    for side_raw, archetype in train_pairs.itertuples(index=False, name=None):
        side = str(side_raw).lower()
        token = archetype_state_token(side, str(archetype))
        prefix = f"target_signature_arch__{token}_"
        targets = [name for name in states.columns if name.startswith(prefix)]
        if not targets:
            raise ValueError(
                f"Missing train residual targets for latent partition {token}"
            )
        local_train = states.loc[
            states["side_name"].astype(str).str.lower().eq(side)
            & states["__ts__"].lt(fit_end),
            targets,
        ]
        target_support = int(
            max(
                (
                    pd.to_numeric(local_train[name], errors="coerce").notna().sum()
                    for name in targets
                ),
                default=0,
            )
        )
        if target_support < 30:
            raise ValueError(
                f"Insufficient train residual support for {token}: {target_support}"
            )
        partitions.append(
            {
                "token": token,
                "side_name": side,
                "archetype_policy_key": str(archetype),
                "target_support": target_support,
            }
        )
    return sorted(partitions, key=lambda item: str(item["token"]))


def _load_partition_identity_history(
    compact_path: Path,
    *,
    fit_end: pd.Timestamp,
) -> pd.DataFrame:
    """Load only pre-freeze routing identities needed for state discovery."""
    columns = ["__ts__", "side_name", "archetype_policy_key"]
    history = pd.read_parquet(
        compact_path,
        columns=columns,
        filters=[("__ts__", "<", fit_end.to_pydatetime())],
    )
    history["__ts__"] = pd.to_datetime(history["__ts__"], utc=True, errors="coerce")
    return history.loc[history["__ts__"].lt(fit_end), columns]


def _load_frozen_state_cache_contract(
    state_cache_dir: Path,
    *,
    encoder_kind: str,
) -> tuple[list[dict[str, Any]], dict[str, list[str]], list[str], dict[str, Any]]:
    """Recover exact local partition inputs from serialized frozen bundles."""
    paths = sorted((state_cache_dir / "states").glob(f"{encoder_kind}__*.joblib"))
    if not paths:
        raise FileNotFoundError(
            f"No frozen {encoder_kind} state bundles under {state_cache_dir / 'states'}"
        )
    support_by_token: dict[str, int] = {}
    cache_manifest_path = state_cache_dir / "manifest.json"
    if cache_manifest_path.exists():
        cache_manifest = json.loads(cache_manifest_path.read_text(encoding="utf-8"))
        support_by_token = {
            str(row["token"]): int(row.get("target_support") or 0)
            for row in (
                (cache_manifest.get("latent_partition_contract") or {}).get(
                    "partitions", []
                )
            )
        }
    partitions: list[dict[str, Any]] = []
    selected_by_partition: dict[str, list[str]] = {}
    required_inputs: list[str] = []
    cache_rows: list[dict[str, Any]] = []
    for path in paths:
        bundle = joblib.load(path)
        token = str(bundle.get("partition_token"))
        side = str(bundle.get("side")).lower()
        archetype = str(bundle.get("archetype"))
        selected = list(map(str, bundle.get("state_feature_candidates") or ()))
        encoder = bundle.get("encoder")
        if not token or encoder is None or bundle.get("gmm") is None or not selected:
            raise ValueError(f"Incomplete frozen state bundle: {path}")
        partitions.append(
            {
                "token": token,
                "side_name": side,
                "archetype_policy_key": archetype,
                "target_support": int(support_by_token.get(token, 0)),
            }
        )
        selected_by_partition[token] = selected
        required_inputs.extend(map(str, encoder.feature_names))
        cache_rows.append(
            {
                "token": token,
                "side_name": side,
                "archetype_policy_key": archetype,
                "state_path": str(path),
                "selected_feature_count": int(len(selected)),
                "encoder_input_count": int(len(encoder.feature_names)),
            }
        )
    tokens = [row["token"] for row in partitions]
    if len(tokens) != len(set(tokens)):
        raise ValueError("Frozen state cache contains duplicate partition tokens")
    return (
        sorted(partitions, key=lambda row: str(row["token"])),
        selected_by_partition,
        list(dict.fromkeys(required_inputs)),
        {
            "schema": "frozen_side_archetype_state_feature_contract_v1",
            "source": str(state_cache_dir),
            "encoder_kind": encoder_kind,
            "partitions": cache_rows,
            "feature_selection_reused": True,
        },
    )


def _partition_state_feature_sets(
    states: pd.DataFrame,
    state_features: Sequence[str],
    partitions: Sequence[Mapping[str, Any]],
    *,
    fit_end: pd.Timestamp,
    max_features: int,
    output_dir: Path,
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    """Select state coordinates once per train-only side x archetype partition."""
    selected_by_partition: dict[str, list[str]] = {}
    relevance_frames: list[pd.DataFrame] = []
    partition_manifest: dict[str, Any] = {}
    for partition in partitions:
        side = str(partition["side_name"]).lower()
        archetype = str(partition["archetype_policy_key"])
        token = str(partition["token"])
        local = prepare_archetype_state_partition(
            states,
            side=side,
            archetype=archetype,
        )
        train = local.loc[local["__ts__"].lt(fit_end)].reset_index(drop=True)
        selected, relevance = select_partition_state_features(
            train,
            state_features,
            max_features=int(max_features),
        )
        if not selected:
            raise ValueError(f"No train-selected state features for {token}")
        selected_by_partition[token] = list(selected)
        if not relevance.empty:
            local_relevance = relevance.copy()
            local_relevance["state_partition_token"] = token
            local_relevance["side_name"] = side
            local_relevance["archetype_policy_key"] = archetype
            relevance_frames.append(local_relevance)
        partition_manifest[token] = {
            "side_name": side,
            "archetype_policy_key": archetype,
            "train_rows": int(len(train)),
            "candidate_features": int(len(state_features)),
            "selected_feature_count": int(len(selected)),
            "selected_features": list(selected),
        }
    relevance_path = output_dir / "partition_state_feature_relevance.csv"
    relevance_table = (
        pd.concat(relevance_frames, ignore_index=True, sort=False)
        if relevance_frames
        else pd.DataFrame()
    )
    relevance_table.to_csv(relevance_path, index=False)
    return selected_by_partition, {
        "schema": "side_archetype_state_feature_selection_v1",
        "fit_end_exclusive": str(fit_end),
        "max_features": int(max_features),
        "relevance_path": str(relevance_path),
        "partitions": partition_manifest,
        "leakage_contract": (
            "Economic relevance uses only each side x archetype partition's train rows; "
            "the selected coordinates are frozen for every encoder family and OOS row."
        ),
    }


def _localize_partition_outputs(
    generated: pd.DataFrame,
    *,
    encoder_kind: str,
    token: str,
) -> pd.DataFrame:
    """Collapse partition-specific target names into row-local feature names."""
    prefix = f"encoder_{encoder_kind}__"
    rename: dict[str, str] = {}
    prediction_prefix = f"{prefix}global_state_pred_signature_arch__{token}_"
    expected_prefix = f"{prefix}global_state_expected_signature_arch__{token}_"
    for name in generated.columns:
        if name.startswith(prediction_prefix):
            rename[name] = (
                f"{prefix}local_arch_signature_pred_"
                f"{name.removeprefix(prediction_prefix)}"
            )
        elif name.startswith(expected_prefix):
            rename[name] = (
                f"{prefix}local_arch_signature_expected_"
                f"{name.removeprefix(expected_prefix)}"
            )
    output = generated.rename(columns=rename)
    unresolved = [
        name
        for name in output.columns
        if name.startswith(
            (
                f"{prefix}global_state_pred_signature_arch__",
                f"{prefix}global_state_expected_signature_arch__",
            )
        )
    ]
    return output.drop(columns=unresolved, errors="ignore")


def _prune_generated_state_blocks(
    generated: pd.DataFrame,
    *,
    encoder_kind: str,
    required_blocks: Sequence[str] | None,
) -> pd.DataFrame:
    """Keep only requested state outputs before the cross-partition union."""
    if required_blocks is None:
        return generated
    requested = set(map(str, required_blocks))
    if not requested:
        return generated
    keys = ["__ts__", "side_name", "archetype_policy_key"]
    prefix = f"encoder_{encoder_kind}__"
    temporal_tokens = ("_delta_", "acceleration", "speed", "dwell", "transition")

    def is_temporal(name: str) -> bool:
        return any(token in name for token in temporal_tokens)

    keep = list(keys)
    for name in generated.columns:
        if name in keys or not name.startswith(prefix):
            continue
        include = False
        if "B2_encoder_bottleneck_signature" in requested:
            include |= any(
                token in name
                for token in (
                    "global_state_latent_",
                    "global_state_pred_signature_",
                    "global_state_expected_",
                    "local_arch_signature_",
                )
            ) and not is_temporal(name)
        if "B0_local_signature_heads" in requested:
            include |= "local_arch_signature_" in name and not is_temporal(name)
        if "B3_static_state_posteriors" in requested:
            include |= (
                "global_state_posterior_" in name or name.endswith("global_state_id")
            ) and not is_temporal(name)
        if "B4_state_uncertainty" in requested:
            include |= any(
                token in name
                for token in ("entropy", "novelty", "distance", "reconstruction")
            ) and not is_temporal(name)
        if "B5_state_transitions" in requested:
            include |= is_temporal(name)
        if include:
            keep.append(name)
    return generated.loc[:, list(dict.fromkeys(keep))]


def _fit_encoder_state_features(
    states: pd.DataFrame,
    state_features: Sequence[str],
    partitions: Sequence[Mapping[str, Any]],
    partition_features: Mapping[str, Sequence[str]] | None = None,
    *,
    encoder_kind: str,
    cutoff: pd.Timestamp,
    fit_end: pd.Timestamp,
    evaluation_end: pd.Timestamp,
    output_dir: Path,
    state_cache_dir: Path,
    latent_dim: int,
    epochs: int,
    components: tuple[int, ...],
    covariance_types: tuple[str, ...],
    reg_covars: tuple[float, ...],
    gmm_n_init: int,
    seed: int,
    reuse_existing_state: bool,
    required_blocks: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    frames: list[pd.DataFrame] = []
    partition_manifests: dict[str, Any] = {}
    for partition in partitions:
        side = str(partition["side_name"]).lower()
        archetype = str(partition["archetype_policy_key"])
        token = str(partition["token"])
        selected_state_features = list((partition_features or {}).get(token, ()))
        local: pd.DataFrame | None = None
        if not selected_state_features:
            local = prepare_archetype_state_partition(
                states,
                side=side,
                archetype=archetype,
            )
            local = local.loc[local["__ts__"].lt(evaluation_end)].reset_index(drop=True)
            train_for_selection = local.loc[local["__ts__"].lt(fit_end)].reset_index(
                drop=True
            )
            selected_state_features, _ = select_partition_state_features(
                train_for_selection,
                state_features,
                max_features=int(ResidualEncoderConfig().max_input_features),
            )
        if not selected_state_features:
            raise ValueError(f"No train-selected state features for {token}")
        cache_path = state_cache_dir / "states" / f"{encoder_kind}__{token}.joblib"
        state_path = output_dir / "states" / f"{encoder_kind}__{token}.joblib"
        bundle = None
        candidate = None
        cache_mismatch_reasons: list[str] = []
        expected_partition = {
            "side_name": side,
            "archetype_policy_key": archetype,
            "token": token,
        }
        if cache_path.exists() and reuse_existing_state:
            candidate = joblib.load(cache_path)
            checks = {
                "side": str(candidate.get("side")) == side,
                "archetype": str(candidate.get("archetype")) == archetype,
                "partition_token": str(candidate.get("partition_token")) == token,
                "cutoff": pd.Timestamp(candidate.get("cutoff")) == cutoff,
                "fit_end": candidate.get("fit_end") is not None
                and pd.Timestamp(candidate.get("fit_end")) == fit_end,
                "encoder_present": candidate.get("encoder") is not None,
                "gmm_present": candidate.get("gmm") is not None,
                "state_features": list(candidate.get("state_feature_candidates") or [])
                == selected_state_features,
            }
            if checks["encoder_present"]:
                checks.update(
                    {
                        "encoder_partition": getattr(
                            candidate["encoder"], "partition", None
                        )
                        == expected_partition,
                        "encoder_kind": str(candidate["encoder"].config.encoder_kind)
                        == encoder_kind,
                        "latent_dim": int(candidate["encoder"].config.latent_dim)
                        == int(latent_dim),
                    }
                )
            if checks["gmm_present"]:
                checks.update(
                    {
                        "gmm_partition": getattr(candidate["gmm"], "partition", None)
                        == expected_partition,
                        "gmm_components": tuple(candidate["gmm"].config.components)
                        == tuple(components),
                        "gmm_covariance": tuple(
                            candidate["gmm"].config.covariance_types
                        )
                        == tuple(covariance_types),
                        "gmm_reg_covars": tuple(candidate["gmm"].config.reg_covars)
                        == tuple(reg_covars),
                        "gmm_n_init": int(candidate["gmm"].config.n_init)
                        == int(gmm_n_init),
                    }
                )
            cache_mismatch_reasons = [
                name for name, matches in checks.items() if not matches
            ]
            if not cache_mismatch_reasons:
                bundle = candidate
        elif reuse_existing_state:
            cache_mismatch_reasons = ["cache_missing"]
        cache_reused = bundle is not None
        _emit(
            "encoder_partition_cache_resolved",
            token=token,
            cache_reused=cache_reused,
            cache_mismatch_reasons=cache_mismatch_reasons,
        )
        if bundle is not None:
            encoder = bundle["encoder"]
            required = list(
                dict.fromkeys(["__ts__", "side_name", *map(str, encoder.feature_names)])
            )
            side_mask = states["side_name"].astype(str).str.lower().eq(side)
            local = states.loc[side_mask, required].copy()
            local["archetype_policy_key"] = archetype
            local["state_partition_token"] = token
            local = local.loc[local["__ts__"].lt(evaluation_end)].reset_index(drop=True)
        elif local is None:
            local = prepare_archetype_state_partition(
                states,
                side=side,
                archetype=archetype,
            )
            local = local.loc[local["__ts__"].lt(evaluation_end)].reset_index(drop=True)
        train_mask = local["__ts__"].lt(fit_end)
        train_rows = int(train_mask.sum())
        if train_rows < 500:
            raise ValueError(f"Insufficient state rows for {token}: {train_rows}")
        if bundle is None:
            train = local.loc[train_mask].reset_index(drop=True)
            partition_seed = int(seed + zlib.crc32(token.encode("utf-8")))
            config = ResidualEncoderConfig(
                encoder_kind=encoder_kind,
                latent_dim=int(latent_dim),
                epochs=int(epochs),
                random_state=partition_seed,
            )
            encoder = GlobalResidualSignatureEncoder(config).fit(
                train,
                selected_state_features,
            )
        latent_all = encoder.transform(local).reset_index(drop=True)
        _emit(
            "encoder_partition_latent_transformed",
            token=token,
            rows=len(latent_all),
            columns=len(latent_all.columns),
        )
        if bundle is None:
            train_latent = latent_all.loc[train_mask.to_numpy()].reset_index(drop=True)
            gmm = GlobalGMMStateModel(
                GMMGridConfig(
                    components=components,
                    covariance_types=covariance_types,
                    reg_covars=reg_covars,
                    n_init=max(1, int(gmm_n_init)),
                    random_state=int(seed + zlib.crc32(token.encode("utf-8"))),
                )
            ).fit(train_latent, train, train["__ts__"])
        else:
            gmm = bundle["gmm"]
        static = gmm.transform(latent_all).reset_index(drop=True)
        _emit(
            "encoder_partition_gmm_transformed",
            token=token,
            rows=len(static),
            columns=len(static.columns),
        )
        temporal = (
            add_temporal_state_features(static, local["__ts__"]).reset_index(drop=True)
            if required_blocks is None or "B5_state_transitions" in required_blocks
            else static
        )
        generated = pd.concat(
            [
                local[["__ts__", "side_name"]].reset_index(drop=True),
                local[["archetype_policy_key"]].reset_index(drop=True),
                latent_all,
                temporal,
            ],
            axis=1,
        )
        prefix = f"encoder_{encoder_kind}__"
        generated = generated.rename(
            columns={
                name: f"{prefix}{name}"
                for name in generated.columns
                if name not in {"__ts__", "side_name", "archetype_policy_key"}
            }
        )
        generated = _localize_partition_outputs(
            generated,
            encoder_kind=encoder_kind,
            token=token,
        )
        generated = _prune_generated_state_blocks(
            generated,
            encoder_kind=encoder_kind,
            required_blocks=required_blocks,
        )
        frames.append(generated)
        _emit(
            "encoder_partition_output_materialized",
            token=token,
            rows=len(generated),
            columns=len(generated.columns),
        )
        bundle = {
            "encoder": encoder,
            "gmm": gmm,
            "cutoff": cutoff,
            "fit_end": fit_end,
            "side": side,
            "archetype": archetype,
            "partition_token": token,
            "state_feature_candidates": selected_state_features,
        }
        if cache_reused and required_blocks is not None:
            state_path = cache_path
        else:
            state_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(bundle, state_path, compress=3)
        partition_manifests[token] = {
            "side_name": side,
            "archetype_policy_key": archetype,
            "encoder": encoder.manifest(),
            "gmm": gmm.manifest(),
            "train_rows": train_rows,
            "state_feature_candidates": selected_state_features,
            "encoder_selected_features": list(encoder.feature_names),
            "fit_end_exclusive": str(fit_end),
            "state_path": str(state_path),
            "cache_path": str(cache_path),
            "cache_reused": bool(cache_reused),
            "cache_mismatch_reasons": cache_mismatch_reasons,
        }
        del local, latent_all, static, temporal, bundle, encoder, gmm
        if not cache_reused:
            del train, train_latent
        if candidate is not None:
            del candidate
        gc.collect()
    combined = pd.concat(frames, ignore_index=True, sort=False)
    posterior_features = [
        name
        for name in combined.columns
        if f"encoder_{encoder_kind}__global_state_posterior_" in name
    ]
    if posterior_features:
        combined[posterior_features] = combined[posterior_features].fillna(0.0)
    return combined, {
        "encoder_kind": encoder_kind,
        "preset": ENCODER_PRESETS[encoder_kind],
        "fit_granularity": "side_x_archetype",
        "partition_count": int(len(partition_manifests)),
        "partitions": partition_manifests,
    }


def _transform_frozen_state_cache_blocks(
    states: pd.DataFrame,
    partitions: Sequence[Mapping[str, Any]],
    *,
    encoder_kind: str,
    cutoff: pd.Timestamp,
    fit_end: pd.Timestamp,
    evaluation_end: pd.Timestamp,
    state_cache_dir: Path,
    required_blocks: Sequence[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Transform exact frozen local bundles without entering any fit path."""
    frames: list[pd.DataFrame] = []
    manifests: dict[str, Any] = {}
    for partition in partitions:
        side = str(partition["side_name"]).lower()
        archetype = str(partition["archetype_policy_key"])
        token = str(partition["token"])
        path = state_cache_dir / "states" / f"{encoder_kind}__{token}.joblib"
        _emit("frozen_encoder_partition_load_started", token=token)
        bundle = joblib.load(path)
        if (
            pd.Timestamp(bundle.get("cutoff")) != cutoff
            or pd.Timestamp(bundle.get("fit_end")) != fit_end
            or str(bundle.get("side")).lower() != side
            or str(bundle.get("archetype")) != archetype
            or str(bundle.get("partition_token")) != token
        ):
            raise ValueError(f"Frozen state cache split/partition mismatch: {path}")
        encoder = bundle["encoder"]
        gmm = bundle["gmm"]
        required = list(
            dict.fromkeys(["__ts__", "side_name", *map(str, encoder.feature_names)])
        )
        side_mask = states["side_name"].astype(str).str.lower().eq(side)
        local = states.loc[side_mask, required].copy()
        local = local.loc[local["__ts__"].lt(evaluation_end)].reset_index(drop=True)
        local["archetype_policy_key"] = archetype
        local["state_partition_token"] = token
        _emit(
            "frozen_encoder_partition_latent_started",
            token=token,
            rows=len(local),
            columns=len(local.columns),
        )
        latent = encoder.transform(local).reset_index(drop=True)
        _emit(
            "frozen_encoder_partition_gmm_started",
            token=token,
            rows=len(latent),
            columns=len(latent.columns),
        )
        static = gmm.transform(latent).reset_index(drop=True)
        transformed = (
            add_temporal_state_features(static, local["__ts__"]).reset_index(drop=True)
            if "B5_state_transitions" in required_blocks
            else static
        )
        generated = pd.concat(
            [
                local[["__ts__", "side_name", "archetype_policy_key"]],
                latent,
                transformed,
            ],
            axis=1,
            copy=False,
        )
        prefix = f"encoder_{encoder_kind}__"
        generated = generated.rename(
            columns={
                name: f"{prefix}{name}"
                for name in generated.columns
                if name not in {"__ts__", "side_name", "archetype_policy_key"}
            }
        )
        generated = _localize_partition_outputs(
            generated,
            encoder_kind=encoder_kind,
            token=token,
        )
        generated = _prune_generated_state_blocks(
            generated,
            encoder_kind=encoder_kind,
            required_blocks=required_blocks,
        )
        frames.append(generated)
        manifests[token] = {
            "side_name": side,
            "archetype_policy_key": archetype,
            "state_path": str(path),
            "cache_reused": True,
            "train_rows": int(local["__ts__"].lt(fit_end).sum()),
            "encoder": encoder.manifest(),
            "gmm": gmm.manifest(),
            "materialized_columns": int(len(generated.columns)),
        }
        _emit(
            "frozen_encoder_partition_transformed",
            token=token,
            rows=len(generated),
            columns=len(generated.columns),
        )
        del bundle, encoder, gmm, local, latent, static, transformed
        gc.collect()
    combined = pd.concat(frames, ignore_index=True, sort=False, copy=False)
    posterior_features = [
        name
        for name in combined.columns
        if f"encoder_{encoder_kind}__global_state_posterior_" in name
    ]
    if posterior_features:
        combined[posterior_features] = combined[posterior_features].fillna(0.0)
    return combined, {
        "schema": "frozen_side_archetype_state_transform_v1",
        "encoder_kind": encoder_kind,
        "fit_granularity": "side_x_archetype",
        "partition_count": int(len(manifests)),
        "partitions": manifests,
        "required_blocks": list(map(str, required_blocks)),
    }


def _state_blocks(
    states: pd.DataFrame,
    generated: pd.DataFrame,
    encoder_kind: str,
) -> dict[str, list[str]]:
    prefix = f"encoder_{encoder_kind}__"
    columns = [name for name in generated.columns if name.startswith(prefix)]
    temporal_tokens = ("_delta_", "acceleration", "speed", "dwell", "transition")

    def is_temporal(name: str) -> bool:
        return any(token in name for token in temporal_tokens)

    latent_and_heads = [
        name
        for name in columns
        if (
            "global_state_latent_" in name
            or "global_state_pred_signature_" in name
            or "global_state_expected_" in name
        )
        and "global_state_pred_signature_arch__" not in name
        and not is_temporal(name)
    ]
    posterior = [
        name
        for name in columns
        if "global_state_posterior_" in name or name.endswith("global_state_id")
        if not is_temporal(name)
    ]
    uncertainty = [
        name
        for name in columns
        if any(
            token in name
            for token in ("entropy", "novelty", "distance", "reconstruction")
        )
        and not is_temporal(name)
    ]
    temporal = [name for name in columns if is_temporal(name)]
    lifecycle = [
        name
        for name in states.columns
        if name.startswith(
            (
                "universe__median__",
                "full_universe__median__",
                "selected_minus_full_universe__",
                "state_phase__",
            )
        )
        and any(
            token in name
            for token in (
                "oi_drawdown",
                "oi_recovery",
                "price_down_oi",
                "price_up_oi",
                "breadth_recovery",
                "systemic_deleveraging",
                "flush_exhaustion",
                "short_covering",
                "pc1_variance",
            )
        )
    ]
    return {
        "B1_lifecycle_market": list(dict.fromkeys(lifecycle)),
        "B2_encoder_bottleneck_signature": list(dict.fromkeys(latent_and_heads)),
        "B3_static_state_posteriors": list(dict.fromkeys(posterior)),
        "B4_state_uncertainty": list(dict.fromkeys(uncertainty)),
        "B5_state_transitions": list(dict.fromkeys(temporal)),
    }


def _local_context_blocks(
    features: Sequence[str],
    encoder_kind: str,
) -> dict[str, list[str]]:
    """Expose local economic heads and priors as separately testable blocks."""
    prefix = f"encoder_{encoder_kind}__"
    local_signature = [
        str(name)
        for name in features
        if str(name).startswith(f"{prefix}local_arch_signature_")
    ]
    local_priors = [
        str(name)
        for name in features
        if str(name).startswith(f"{prefix}local_state_prior_")
    ]
    unresolved = sorted(
        set(map(str, features)) - set(local_signature) - set(local_priors)
    )
    if unresolved:
        raise ValueError(f"Unclassified local state context features: {unresolved[:8]}")
    return {
        "B0_local_signature_heads": list(dict.fromkeys(local_signature)),
        "B0_local_state_priors": list(dict.fromkeys(local_priors)),
    }


def _final_feature_union(
    baseline_features: Sequence[str],
    mandatory_context: Sequence[str],
    accepted_blocks: Sequence[str],
    blocks: Mapping[str, Sequence[str]],
) -> list[str]:
    """Keep side x archetype context even when no optional block is accepted."""
    return list(
        dict.fromkeys(
            list(map(str, baseline_features))
            + list(map(str, mandatory_context))
            + [
                str(feature)
                for block in accepted_blocks
                for feature in blocks.get(str(block), ())
            ]
        )
    )


def _materialize_local_signature_predictions(
    frame: pd.DataFrame,
    encoder_kind: str,
) -> pd.DataFrame:
    """Select each row's own side x archetype signature-head outputs."""
    encoder_prefix = f"encoder_{encoder_kind}__"
    persistence_metrics = (
        "signed_alignment_prev7d",
        "positive_persistence_prev7d",
        "negative_persistence_prev7d",
        "signed_autocov_2d",
        "positive_persistence_2d",
        "negative_persistence_2d",
    )
    metrics = list(
        dict.fromkeys([*GLOBAL_RESIDUAL_SIGNATURE_BASES, *persistence_metrics])
    )
    output = {
        metric: np.full(len(frame), np.nan, dtype=np.float32) for metric in metrics
    }
    groups = frame.groupby(["side_name", "archetype_policy_key"], sort=False).indices
    for (side, archetype), raw_positions in groups.items():
        positions = np.asarray(raw_positions, dtype=np.int64)
        token = archetype_state_token(str(side), str(archetype))
        for metric in metrics:
            local_col = (
                f"{encoder_prefix}global_state_pred_signature_arch__{token}_{metric}"
            )
            side_col = f"{encoder_prefix}global_state_pred_signature_{str(side).lower()}_{metric}"
            global_col = f"{encoder_prefix}global_state_pred_signature_global_{metric}"
            source = next(
                (
                    name
                    for name in (local_col, side_col, global_col)
                    if name in frame.columns
                ),
                None,
            )
            if source is not None:
                output[metric][positions] = pd.to_numeric(
                    frame.iloc[positions][source], errors="coerce"
                ).to_numpy(dtype=np.float32)
    result = pd.DataFrame(index=frame.index)
    for metric, values in output.items():
        if np.isfinite(values).any():
            result[f"{encoder_prefix}local_arch_signature_{metric}"] = values
    return result


def _append_mandatory_archetype_context(
    frame: pd.DataFrame,
    *,
    encoder_kind: str,
    fit_end: pd.Timestamp,
    prior_model: SideArchetypeStatePriors | None = None,
) -> tuple[pd.DataFrame, list[str], SideArchetypeStatePriors]:
    posterior_columns = [
        name
        for name in frame.columns
        if name.startswith(f"encoder_{encoder_kind}__global_state_posterior_")
        and not any(token in name for token in ("_delta_", "acceleration"))
    ]
    if prior_model is None:
        prior_model = SideArchetypeStatePriors(
            shrinkage_rows=120.0,
            output_prefix=f"encoder_{encoder_kind}__local_state_prior_",
            partition_local=True,
            strict_unknown=True,
        ).fit(frame.loc[frame["__ts__"].lt(fit_end)], posterior_columns)
    posterior_priors = prior_model.transform(frame)
    existing_signature_columns = [
        name
        for name in frame.columns
        if name.startswith(f"encoder_{encoder_kind}__local_arch_signature_")
    ]
    signature_predictions = (
        pd.DataFrame(index=frame.index)
        if existing_signature_columns
        else _materialize_local_signature_predictions(frame, encoder_kind)
    )
    generated = pd.concat(
        [posterior_priors, signature_predictions], axis=1, copy=False
    ).astype(np.float32, copy=False)
    output = pd.concat([frame, generated], axis=1, copy=False)
    raw_local_heads = [
        name
        for name in output
        if name.startswith(
            f"encoder_{encoder_kind}__global_state_pred_signature_arch__"
        )
    ]
    if raw_local_heads:
        output = output.drop(columns=raw_local_heads)
    mandatory = list(
        dict.fromkeys(existing_signature_columns + list(generated.columns))
    )
    return output, mandatory, prior_model


def _merge_state_features(
    data: pd.DataFrame, state_features: pd.DataFrame
) -> pd.DataFrame:
    keys = ["__ts__", "side_name"]
    if "archetype_policy_key" in data and "archetype_policy_key" in state_features:
        keys.append("archetype_policy_key")
    generated = [name for name in state_features if name not in set(keys)]
    right = state_features[[*keys, *generated]].drop_duplicates(keys, keep="last")
    out = data.merge(right, on=keys, how="left", validate="many_to_one")
    for name in generated:
        out[name] = pd.to_numeric(out[name], errors="coerce").astype(np.float32)
    return out


def _fit_fixed_revision(
    train: pd.DataFrame,
    evaluation: pd.DataFrame,
    features: Sequence[str],
    params: Mapping[str, Any],
    *,
    arm: str,
    seed: int,
) -> tuple[pd.DataFrame, Any, dict[str, Any]]:
    train_local, eval_local = _add_reference_fold_features(train, evaluation)
    raw_features = [
        name
        for name in dict.fromkeys(map(str, features))
        if name not in META_POST_SELECTION_OOD_FEATURE_NAMES
        and name in train_local.columns
    ]
    target, target_column = _base_soft_label_target(train_local)
    target_mask = target.notna()
    train_local = train_local.loc[target_mask].reset_index(drop=True)
    target = target.loc[target_mask].reset_index(drop=True)
    x_train, x_eval, medians = _matrix_fit_transform(
        train_local, eval_local, raw_features
    )
    ood_state = _fit_ood_state(x_train, raw_features)
    x_train = _apply_ood_state(x_train, ood_state)
    x_eval = _apply_ood_state(x_eval, ood_state)
    model_features = list(
        dict.fromkeys(raw_features + list(META_POST_SELECTION_OOD_FEATURE_NAMES))
    )
    x_train = x_train.reindex(columns=model_features, fill_value=0.0)
    x_eval = x_eval.reindex(columns=model_features, fill_value=0.0)
    model = _fit_base_soft_label_model(
        x_train,
        target,
        train_local,
        int(seed),
        lgbm_params=dict(params),
    )
    if model is None:
        raise RuntimeError(f"Fixed champion meta fit failed for {arm}")
    score_train = _predict(model, x_train, classifier=False)
    score_eval = _predict(model, x_eval, classifier=False)
    platt_alt = _fit_platt(
        score_train, pd.to_numeric(train_local["clean_exec"], errors="coerce")
    )
    platt_ref = _fit_platt(
        train_local["score_meta_base_soft_label"],
        pd.to_numeric(train_local["clean_exec"], errors="coerce"),
    )
    keep = [
        name
        for name in (
            *KEY_COLUMNS,
            "archetype_label_family",
            *OUTCOME_COLUMNS,
            "score_meta_base_soft_label",
            FIXED_REFERENCE_SCORE,
            FIXED_REFERENCE_HIT_PROB,
        )
        if name in eval_local.columns
    ]
    scored = eval_local[keep].copy()
    scored["calendar_month"] = scored["__ts__"].dt.strftime("%Y-%m")
    scored["week_start"] = scored["__ts__"].dt.floor("D") - pd.to_timedelta(
        scored["__ts__"].dt.weekday, unit="D"
    )
    reference_score_name = (
        FIXED_REFERENCE_SCORE
        if FIXED_REFERENCE_SCORE in scored
        else "score_meta_base_soft_label"
    )
    scored["score_current_reference"] = pd.to_numeric(
        scored[reference_score_name], errors="coerce"
    ).astype(np.float32)
    scored["score_alternative"] = np.asarray(score_eval, dtype=np.float32)
    if FIXED_REFERENCE_HIT_PROB in scored:
        scored["hit_prob_current_reference"] = pd.to_numeric(
            scored[FIXED_REFERENCE_HIT_PROB], errors="coerce"
        ).astype(np.float32)
    else:
        scored["hit_prob_current_reference"] = _calibrate(
            platt_ref, scored["score_current_reference"]
        )
    scored["hit_prob_alternative"] = _calibrate(platt_alt, scored["score_alternative"])
    return (
        scored,
        model,
        {
            "arm": arm,
            "target_column": target_column,
            "train_rows": int(len(train_local)),
            "evaluation_rows": int(len(eval_local)),
            "raw_features": raw_features,
            "model_features": model_features,
            "medians": medians,
            "ood_state": ood_state,
        },
    )


def _fit_symmetric_residual_regressor(
    x: pd.DataFrame,
    target: pd.Series,
    params: Mapping[str, Any],
    *,
    seed: int,
) -> Any:
    """Fit a signed residual head without favoring either correction direction."""
    y = pd.to_numeric(target, errors="coerce")
    valid = y.notna()
    if int(valid.sum()) < 100 or float(y.loc[valid].std()) <= 1e-12:
        return None
    n_rows = int(valid.sum())
    absolute = y.loc[valid].abs().to_numpy(dtype=np.float32)
    scale = max(float(np.quantile(absolute, 0.80)), 1e-4)
    weights = 1.0 + np.clip(absolute / scale, 0.0, 2.0)
    weights = (weights / max(float(weights.mean()), 1e-12)).astype(np.float32)
    requested_min_child = int(float(params.get("min_child_samples", 80)))
    local_min_child = min(requested_min_child, max(40, n_rows // 25))
    model = lgb.LGBMRegressor(
        objective="huber",
        alpha=float(params.get("alpha", 0.90)),
        n_estimators=int(float(params.get("n_estimators", 500))),
        learning_rate=float(params.get("learning_rate", 0.03)),
        num_leaves=int(float(params.get("num_leaves", 31))),
        max_depth=int(float(params.get("max_depth", 5))),
        min_child_samples=int(local_min_child),
        min_child_weight=float(params.get("min_child_weight", 1e-3)),
        min_split_gain=float(params.get("min_split_gain", 0.0)),
        max_bin=int(float(params.get("max_bin", 63))),
        min_data_in_bin=int(float(params.get("min_data_in_bin", 3))),
        subsample=float(params.get("subsample", 1.0)),
        subsample_freq=int(float(params.get("subsample_freq", 0))),
        colsample_bytree=float(params.get("colsample_bytree", 1.0)),
        reg_alpha=float(params.get("reg_alpha", 0.0)),
        reg_lambda=float(params.get("reg_lambda", 0.0)),
        path_smooth=float(params.get("path_smooth", 0.0)),
        random_state=int(seed),
        n_jobs=2,
        verbosity=-1,
    )
    model.fit(
        x.loc[valid],
        y.loc[valid].astype(np.float32),
        sample_weight=weights,
    )
    return model


def _daily_top10_surprise_target(
    frame: pd.DataFrame,
    score: pd.Series,
    residual: Sequence[float],
    valid: pd.Series,
    *,
    support_shrinkage: float,
) -> pd.Series:
    """Broadcast shrinkage-adjusted local daily top-10 surprise to train rows."""
    reference_rank = score.groupby(frame["__ts__"], observed=True).rank(pct=True)
    source = pd.DataFrame(
        {
            "day": frame["__ts__"].dt.floor("D"),
            "side_name": frame["side_name"].astype(str).str.lower(),
            "archetype_policy_key": frame["archetype_policy_key"].astype(str),
            "residual": np.asarray(residual, dtype=np.float32),
            "selected": reference_rank.ge(0.90) & valid,
        },
        index=frame.index,
    )
    daily = (
        source.loc[source["selected"]]
        .groupby(
            ["day", "side_name", "archetype_policy_key"],
            observed=True,
            sort=True,
        )["residual"]
        .agg(["mean", "count"])
    )
    shrinkage = daily["count"] / (daily["count"] + max(float(support_shrinkage), 0.0))
    daily["target"] = daily["mean"] * shrinkage
    row_keys = pd.MultiIndex.from_arrays(
        [
            source["day"],
            source["side_name"],
            source["archetype_policy_key"],
        ],
        names=daily.index.names,
    )
    return pd.Series(
        daily["target"].reindex(row_keys).to_numpy(dtype=np.float32),
        index=frame.index,
        dtype=np.float32,
    )


def _split_signed_surprise_targets(
    signed_target: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """Return separate favorable and adverse magnitudes for local fitting."""
    signed = pd.to_numeric(signed_target, errors="coerce").astype(np.float32)
    favorable = signed.clip(lower=0.0).astype(np.float32)
    adverse = (-signed).clip(lower=0.0).astype(np.float32)
    return favorable, adverse


def _daily_top10_persistence_targets(
    frame: pd.DataFrame,
    score: pd.Series,
    residual: Sequence[float],
    valid: pd.Series,
    *,
    support_shrinkage: float,
) -> tuple[pd.Series, pd.Series]:
    """Broadcast prior-week aligned favorable/adverse persistence labels."""
    signed = _daily_top10_surprise_target(
        frame,
        score,
        residual,
        valid,
        support_shrinkage=support_shrinkage,
    )
    source = pd.DataFrame(
        {
            "day": frame["__ts__"].dt.floor("D"),
            "side_name": frame["side_name"].astype(str).str.lower(),
            "archetype_policy_key": frame["archetype_policy_key"].astype(str),
            "signed": signed,
        },
        index=frame.index,
    )
    keys = ["day", "side_name", "archetype_policy_key"]
    daily = (
        source.dropna(subset=["signed"])
        .drop_duplicates(keys, keep="last")
        .sort_values(keys, kind="stable")
    )
    generated: list[pd.DataFrame] = []
    for _, local in daily.groupby(
        ["side_name", "archetype_policy_key"], observed=True, sort=False
    ):
        local = local.sort_values("day", kind="stable").copy()
        observed_days = pd.DatetimeIndex(local["day"])
        calendar = pd.date_range(observed_days.min(), observed_days.max(), freq="D")
        values = pd.Series(
            pd.to_numeric(local["signed"], errors="coerce").to_numpy(dtype=np.float64),
            index=observed_days,
        ).reindex(calendar)
        positive = values.clip(lower=0.0)
        negative = (-values).clip(lower=0.0)
        prior_positive = positive.shift(1).rolling(7, min_periods=3).mean()
        prior_negative = negative.shift(1).rolling(7, min_periods=3).mean()
        local["favorable_persistence"] = (
            (positive * prior_positive)
            .reindex(observed_days)
            .to_numpy(dtype=np.float32)
        )
        local["adverse_persistence"] = (
            (negative * prior_negative)
            .reindex(observed_days)
            .to_numpy(dtype=np.float32)
        )
        generated.append(local)
    if generated:
        daily_targets = pd.concat(generated, ignore_index=True, sort=False).set_index(
            keys
        )
    else:
        daily_targets = pd.DataFrame(
            columns=["favorable_persistence", "adverse_persistence"],
            index=pd.MultiIndex.from_arrays([[], [], []], names=keys),
        )
    row_keys = pd.MultiIndex.from_frame(source[keys])
    favorable = pd.Series(
        daily_targets["favorable_persistence"]
        .reindex(row_keys)
        .to_numpy(dtype=np.float32),
        index=frame.index,
        dtype=np.float32,
    )
    adverse = pd.Series(
        daily_targets["adverse_persistence"]
        .reindex(row_keys)
        .to_numpy(dtype=np.float32),
        index=frame.index,
        dtype=np.float32,
    )
    return favorable, adverse


def _fit_local_residual_correction(
    train: pd.DataFrame,
    evaluation: pd.DataFrame,
    features: Sequence[str],
    params: Mapping[str, Any],
    *,
    seed: int,
    target_mode: str = "row_clean_residual",
    daily_support_shrinkage: float = 20.0,
) -> tuple[
    np.ndarray,
    np.ndarray,
    dict[str, np.ndarray],
    dict[str, Any],
    dict[str, Any],
]:
    """Predict clean-hit calibration residuals within each side x archetype."""
    train_local, eval_local = _add_reference_fold_features(train, evaluation)
    clean = pd.to_numeric(train_local["clean_exec"], errors="coerce")
    score = pd.to_numeric(train_local["score_meta_base_soft_label"], errors="coerce")
    valid = clean.notna() & score.notna()
    platt_ref = _fit_platt(score.loc[valid], clean.loc[valid])
    base_prob_train = np.asarray(_calibrate(platt_ref, score), dtype=np.float32)
    base_prob_eval = np.asarray(
        _calibrate(
            platt_ref,
            pd.to_numeric(eval_local["score_meta_base_soft_label"], errors="coerce"),
        ),
        dtype=np.float32,
    )
    residual = clean.to_numpy(dtype=np.float32) - base_prob_train
    target_mode = str(target_mode).strip().lower()
    correction_targets: dict[str, pd.Series]
    if target_mode == "row_clean_residual":
        correction_target = pd.Series(
            residual, index=train_local.index, dtype=np.float32
        )
        correction_targets = {"signed": correction_target}
    elif target_mode == "daily_top10_surprise":
        correction_target = _daily_top10_surprise_target(
            train_local,
            score,
            residual,
            valid,
            support_shrinkage=float(daily_support_shrinkage),
        )
        correction_targets = {"signed": correction_target}
    elif target_mode == "daily_top10_surprise_two_head":
        correction_target = _daily_top10_surprise_target(
            train_local,
            score,
            residual,
            valid,
            support_shrinkage=float(daily_support_shrinkage),
        )
        favorable, adverse = _split_signed_surprise_targets(correction_target)
        correction_targets = {
            "favorable": favorable,
            "adverse": adverse,
        }
    elif target_mode == "daily_top10_persistence_two_head":
        favorable, adverse = _daily_top10_persistence_targets(
            train_local,
            score,
            residual,
            valid,
            support_shrinkage=float(daily_support_shrinkage),
        )
        correction_target = favorable - adverse
        correction_targets = {
            "favorable": favorable,
            "adverse": adverse,
        }
    else:
        raise ValueError(f"Unknown residual correction target mode: {target_mode}")
    correction = np.zeros(len(eval_local), dtype=np.float32)
    correction_components: dict[str, np.ndarray] = (
        {
            "favorable": np.zeros(len(eval_local), dtype=np.float32),
            "adverse": np.zeros(len(eval_local), dtype=np.float32),
        }
        if target_mode
        in {
            "daily_top10_surprise_two_head",
            "daily_top10_persistence_two_head",
        }
        else {}
    )
    raw_features = [
        name
        for name in dict.fromkeys(
            [
                *(
                    ["score_meta_base_soft_label"]
                    if target_mode == "row_clean_residual"
                    else []
                ),
                *map(str, features),
            ]
        )
        if name not in META_POST_SELECTION_OOD_FEATURE_NAMES
        and name in train_local.columns
    ]
    models: dict[str, Any] = {}
    partitions: dict[str, Any] = {}
    eval_groups = eval_local.groupby(
        ["side_name", "archetype_policy_key"], sort=False
    ).indices
    for (side_raw, archetype_raw), raw_positions in eval_groups.items():
        side = str(side_raw).lower()
        archetype = str(archetype_raw)
        token = archetype_state_token(side, archetype)
        train_mask = (
            train_local["side_name"].astype(str).str.lower().eq(side)
            & train_local["archetype_policy_key"].astype(str).eq(archetype)
            & correction_target.notna()
        )
        train_positions = np.flatnonzero(train_mask.to_numpy())
        eval_positions = np.asarray(raw_positions, dtype=np.int64)
        is_daily_target = target_mode.startswith("daily_top10_")
        minimum_support = 100 if is_daily_target else 500
        if len(train_positions) < minimum_support:
            partitions[token] = {
                "status": "insufficient_train_support",
                "train_rows": int(len(train_positions)),
                "evaluation_rows": int(len(eval_positions)),
            }
            continue
        local_train = train_local.iloc[train_positions].copy()
        for head_name, head_target in correction_targets.items():
            local_train[f"__residual_correction_target__{head_name}"] = (
                head_target.iloc[train_positions].to_numpy(dtype=np.float32)
            )
        if is_daily_target:
            local_train = local_train.drop_duplicates("__ts__", keep="last")
        local_train = local_train.reset_index(drop=True)
        local_eval = eval_local.iloc[eval_positions].reset_index(drop=True)
        x_train, x_eval, medians = _matrix_fit_transform(
            local_train,
            local_eval,
            raw_features,
        )
        head_models: dict[str, Any] = {}
        head_predictions: dict[str, np.ndarray] = {}
        head_manifests: dict[str, Any] = {}
        for head_index, head_name in enumerate(correction_targets):
            target_column = f"__residual_correction_target__{head_name}"
            local_target = pd.Series(
                local_train[target_column].to_numpy(dtype=np.float32),
                index=x_train.index,
                dtype=np.float32,
            )
            model = _fit_symmetric_residual_regressor(
                x_train,
                local_target,
                params,
                seed=int(
                    seed + zlib.crc32(token.encode("utf-8")) + 104_729 * head_index
                ),
            )
            if model is None:
                continue
            train_prediction = np.asarray(
                _predict(model, x_train, classifier=False), dtype=np.float32
            )
            prediction_bias = float(
                np.nanmean(train_prediction) - np.nanmean(local_target)
            )
            predicted = np.asarray(
                _predict(model, x_eval, classifier=False), dtype=np.float32
            ) - np.float32(prediction_bias)
            if target_mode in {
                "daily_top10_surprise_two_head",
                "daily_top10_persistence_two_head",
            }:
                predicted = np.clip(predicted, 0.0, 0.75)
            else:
                predicted = np.clip(predicted, -0.75, 0.75)
            head_models[head_name] = model
            head_predictions[head_name] = predicted
            head_manifests[head_name] = {
                "target_mean": float(np.nanmean(local_target)),
                "target_std": float(np.nanstd(local_target)),
                "prediction_mean": float(np.nanmean(predicted)),
                "prediction_std": float(np.nanstd(predicted)),
                "train_prediction_bias_removed": prediction_bias,
            }
        required_heads = set(correction_targets)
        if set(head_models) != required_heads:
            partitions[token] = {
                "status": "fit_failed",
                "train_rows": int(len(train_positions)),
                "evaluation_rows": int(len(eval_positions)),
                "required_heads": sorted(required_heads),
                "fitted_heads": sorted(head_models),
            }
            continue
        if target_mode in {
            "daily_top10_surprise_two_head",
            "daily_top10_persistence_two_head",
        }:
            predicted = head_predictions["favorable"] - head_predictions["adverse"]
            correction_components["favorable"][eval_positions] = head_predictions[
                "favorable"
            ]
            correction_components["adverse"][eval_positions] = head_predictions[
                "adverse"
            ]
        else:
            predicted = head_predictions["signed"]
        correction[eval_positions] = np.clip(predicted, -0.75, 0.75)
        models[token] = head_models
        partitions[token] = {
            "status": "fitted",
            "side_name": side,
            "archetype_policy_key": archetype,
            "train_rows": int(len(local_train)),
            "evaluation_rows": int(len(eval_positions)),
            "features": raw_features,
            "medians": medians,
            "prediction_mean": float(np.nanmean(predicted)),
            "prediction_std": float(np.nanstd(predicted)),
            "heads": head_manifests,
        }
    return (
        correction,
        base_prob_eval,
        correction_components,
        models,
        {
            "schema": "side_archetype_clean_hit_residual_correction_v2",
            "fit_granularity": "side_x_archetype",
            "target": target_mode,
            "daily_support_shrinkage": float(daily_support_shrinkage),
            "raw_features": raw_features,
            "partitions": partitions,
            "leakage_contract": (
                "Residual targets and reference calibration are fitted on pre-cutoff "
                "training rows only. OOS corrections use frozen local models and "
                "pre-entry state features."
            ),
        },
    )


def _score_local_residual_correction(
    evaluation: pd.DataFrame,
    *,
    correction: Sequence[float],
    base_probability: Sequence[float],
    scale: float,
    arm: str,
    rank_mode: str = "rerank",
    favorable_correction: Sequence[float] | None = None,
    adverse_correction: Sequence[float] | None = None,
    favorable_scale: float | None = None,
    adverse_scale: float | None = None,
) -> pd.DataFrame:
    keep = [
        name
        for name in (
            *KEY_COLUMNS,
            "archetype_label_family",
            *OUTCOME_COLUMNS,
            "score_meta_base_soft_label",
            FIXED_REFERENCE_SCORE,
            FIXED_REFERENCE_HIT_PROB,
        )
        if name in evaluation.columns
    ]
    scored = evaluation[keep].copy()
    scored["calendar_month"] = scored["__ts__"].dt.strftime("%Y-%m")
    scored["week_start"] = scored["__ts__"].dt.floor("D") - pd.to_timedelta(
        scored["__ts__"].dt.weekday, unit="D"
    )
    scored["score_current_reference"] = pd.to_numeric(
        scored["score_meta_base_soft_label"], errors="coerce"
    ).astype(np.float32)
    base = np.asarray(base_probability, dtype=np.float32)
    residual = np.asarray(correction, dtype=np.float32)
    if len(base) != len(scored) or len(residual) != len(scored):
        raise ValueError(
            "Residual correction vectors do not align with evaluation rows"
        )
    asymmetric = favorable_correction is not None or adverse_correction is not None
    if asymmetric:
        if favorable_correction is None or adverse_correction is None:
            raise ValueError(
                "Both favorable and adverse correction vectors are required"
            )
        favorable = np.asarray(favorable_correction, dtype=np.float32)
        adverse = np.asarray(adverse_correction, dtype=np.float32)
        if len(favorable) != len(scored) or len(adverse) != len(scored):
            raise ValueError("Asymmetric correction vectors do not align with rows")
        favorable_weight = float(
            favorable_scale if favorable_scale is not None else scale
        )
        adverse_weight = float(adverse_scale if adverse_scale is not None else scale)
        adjusted = np.clip(
            base + favorable_weight * favorable - adverse_weight * adverse,
            1e-4,
            1.0 - 1e-4,
        )
    else:
        favorable_weight = np.nan
        adverse_weight = np.nan
        adjusted = np.clip(base + float(scale) * residual, 1e-4, 1.0 - 1e-4)
    rank_mode = str(rank_mode).strip().lower()
    if rank_mode == "rerank":
        scored["score_alternative"] = adjusted.astype(np.float32)
    elif rank_mode == "calibration_only":
        scored["score_alternative"] = scored["score_current_reference"]
    else:
        raise ValueError(f"Unknown residual correction rank mode: {rank_mode}")
    scored["hit_prob_current_reference"] = base
    scored["hit_prob_alternative"] = adjusted.astype(np.float32)
    scored["local_residual_correction"] = residual
    scored["residual_correction_scale"] = np.float32(scale)
    scored["residual_correction_favorable_scale"] = np.float32(favorable_weight)
    scored["residual_correction_adverse_scale"] = np.float32(adverse_weight)
    scored["residual_correction_rank_mode"] = rank_mode
    scored["alternative_arm"] = str(arm)
    return scored


def _score_external_prediction_vector(
    train: pd.DataFrame,
    evaluation: pd.DataFrame,
    *,
    alternative_score: Sequence[float],
    alternative_train_oof_score: Sequence[float],
    arm: str,
) -> pd.DataFrame:
    """Calibrate and align a canonical model's saved OOS prediction vector."""
    target, _ = _base_soft_label_target(train)
    target_mask = target.notna()
    train_local = train.loc[target_mask].reset_index(drop=True)
    train_score = np.asarray(alternative_train_oof_score, dtype=np.float32)
    eval_score = np.asarray(alternative_score, dtype=np.float32)
    if len(train_score) != len(train_local):
        raise ValueError(
            f"Canonical train OOF score length mismatch: {len(train_score)} != "
            f"{len(train_local)}"
        )
    if len(eval_score) != len(evaluation):
        raise ValueError(
            f"Canonical evaluation score length mismatch: {len(eval_score)} != "
            f"{len(evaluation)}"
        )
    platt_alt = _fit_platt(
        pd.Series(train_score),
        pd.to_numeric(train_local["clean_exec"], errors="coerce"),
    )
    platt_ref = _fit_platt(
        train_local["score_meta_base_soft_label"],
        pd.to_numeric(train_local["clean_exec"], errors="coerce"),
    )
    keep = [
        name
        for name in (
            *KEY_COLUMNS,
            "archetype_label_family",
            *OUTCOME_COLUMNS,
            "score_meta_base_soft_label",
        )
        if name in evaluation.columns
    ]
    scored = evaluation[keep].copy()
    scored["calendar_month"] = scored["__ts__"].dt.strftime("%Y-%m")
    scored["week_start"] = scored["__ts__"].dt.floor("D") - pd.to_timedelta(
        scored["__ts__"].dt.weekday, unit="D"
    )
    reference_score_name = (
        FIXED_REFERENCE_SCORE
        if FIXED_REFERENCE_SCORE in scored
        else "score_meta_base_soft_label"
    )
    scored["score_current_reference"] = pd.to_numeric(
        scored[reference_score_name], errors="coerce"
    ).astype(np.float32)
    scored["score_alternative"] = eval_score
    if FIXED_REFERENCE_HIT_PROB in scored:
        scored["hit_prob_current_reference"] = pd.to_numeric(
            scored[FIXED_REFERENCE_HIT_PROB], errors="coerce"
        ).astype(np.float32)
    else:
        scored["hit_prob_current_reference"] = _calibrate(
            platt_ref,
            scored["score_current_reference"],
        )
    scored["hit_prob_alternative"] = _calibrate(
        platt_alt,
        scored["score_alternative"],
    )
    scored["alternative_arm"] = str(arm)
    return scored


def _daily_signed_autocorrelation(
    scored: pd.DataFrame, score_column: str, probability_column: str
) -> dict[str, float]:
    score = pd.to_numeric(scored[score_column], errors="coerce")
    rank = score.groupby(scored["__ts__"], observed=True).rank(pct=True)
    selected = scored.loc[rank.ge(0.90)].copy()
    surprise = pd.to_numeric(selected["clean_exec"], errors="coerce") - pd.to_numeric(
        selected[probability_column], errors="coerce"
    )
    daily = (
        pd.DataFrame(
            {
                "day": selected["__ts__"].dt.floor("D"),
                "signed": surprise,
                "positive": surprise.clip(lower=0.0),
                "negative": (-surprise).clip(lower=0.0),
            }
        )
        .groupby("day", observed=True)
        .mean()
    )
    if not daily.empty:
        daily = daily.reindex(
            pd.date_range(daily.index.min(), daily.index.max(), freq="D")
        )
    return {
        "signed_ac1": float(daily["signed"].autocorr(1)) if len(daily) >= 5 else np.nan,
        "positive_ac1": float(daily["positive"].autocorr(1))
        if len(daily) >= 5
        else np.nan,
        "negative_ac1": float(daily["negative"].autocorr(1))
        if len(daily) >= 5
        else np.nan,
    }


def _side_archetype_signed_autocorrelation(
    scored: pd.DataFrame,
    score_column: str,
    probability_column: str,
) -> pd.DataFrame:
    score = pd.to_numeric(scored[score_column], errors="coerce")
    rank = score.groupby(scored["__ts__"], observed=True).rank(pct=True)
    selected = scored.loc[rank.ge(0.90)].copy()
    selected["day"] = selected["__ts__"].dt.floor("D")
    selected["surprise"] = pd.to_numeric(
        selected["clean_exec"], errors="coerce"
    ) - pd.to_numeric(selected[probability_column], errors="coerce")
    daily = (
        selected.groupby(
            ["day", "side_name", "archetype_policy_key"],
            observed=True,
            sort=True,
        )["surprise"]
        .mean()
        .reset_index()
    )
    rows: list[dict[str, Any]] = []
    for (side, archetype), local in daily.groupby(
        ["side_name", "archetype_policy_key"], observed=True, sort=True
    ):
        local = local.sort_values("day").set_index("day")
        values = pd.to_numeric(local["surprise"], errors="coerce")
        if not values.empty:
            values = values.reindex(
                pd.date_range(values.index.min(), values.index.max(), freq="D")
            )
        positive = values.clip(lower=0.0)
        negative = (-values).clip(lower=0.0)
        rows.append(
            {
                "side_name": side,
                "archetype_policy_key": archetype,
                "days": int(values.notna().sum()),
                "signed_ac1": float(values.autocorr(1))
                if values.notna().sum() >= 5
                else np.nan,
                "positive_ac1": float(positive.autocorr(1))
                if positive.notna().sum() >= 5
                else np.nan,
                "negative_ac1": float(negative.autocorr(1))
                if negative.notna().sum() >= 5
                else np.nan,
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "side_name",
            "archetype_policy_key",
            "days",
            "signed_ac1",
            "positive_ac1",
            "negative_ac1",
        ],
    )


def _revision_summary(
    scored: pd.DataFrame,
    arm: str,
    *,
    priority: str = "balanced",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    metrics = metrics_by_scope(scored, arm)
    overall = metrics[
        metrics["scope"].eq("overall")
        & metrics["selector"].eq(arm)
        & metrics["fraction"].eq(0.10)
    ].iloc[0]
    baseline = metrics[
        metrics["scope"].eq("overall")
        & metrics["selector"].eq("current_reference")
        & metrics["fraction"].eq(0.10)
    ].iloc[0]
    week = metrics[
        metrics["scope"].eq("week")
        & metrics["selector"].eq(arm)
        & metrics["fraction"].eq(0.10)
    ]
    base_week = metrics[
        metrics["scope"].eq("week")
        & metrics["selector"].eq("current_reference")
        & metrics["fraction"].eq(0.10)
    ]
    month = metrics[
        metrics["scope"].eq("month")
        & metrics["selector"].eq(arm)
        & metrics["fraction"].eq(0.10)
    ]
    base_month = metrics[
        metrics["scope"].eq("month")
        & metrics["selector"].eq("current_reference")
        & metrics["fraction"].eq(0.10)
    ]
    ac = _daily_signed_autocorrelation(
        scored, "score_alternative", "hit_prob_alternative"
    )
    base_ac = _daily_signed_autocorrelation(
        scored, "score_current_reference", "hit_prob_current_reference"
    )
    mean_ev = float(overall["mean_ev_after_1pct"])
    base_ev = float(baseline["mean_ev_after_1pct"])
    worst_week = float(pd.to_numeric(week["mean_ev_after_1pct"], errors="coerce").min())
    base_worst_week = float(
        pd.to_numeric(base_week["mean_ev_after_1pct"], errors="coerce").min()
    )
    worst_month = float(
        pd.to_numeric(month["mean_ev_after_1pct"], errors="coerce").min()
    )
    base_worst_month = float(
        pd.to_numeric(base_month["mean_ev_after_1pct"], errors="coerce").min()
    )
    ac_abs = float(np.nanmean(np.abs([ac["positive_ac1"], ac["negative_ac1"]])))
    base_ac_abs = float(
        np.nanmean(np.abs([base_ac["positive_ac1"], base_ac["negative_ac1"]]))
    )
    side_arch = metrics[
        metrics["scope"].eq("side_archetype") & metrics["fraction"].eq(0.10)
    ]
    alt_side_arch = side_arch[side_arch["selector"].eq(arm)]
    base_side_arch = side_arch[side_arch["selector"].eq("current_reference")]
    side_arch_cmp = alt_side_arch.merge(
        base_side_arch,
        on=["side_name", "archetype_policy_key"],
        suffixes=("_alt", "_base"),
        how="inner",
    )
    supported = side_arch_cmp[
        side_arch_cmp["selected_rows_alt"].ge(50)
        & side_arch_cmp["selected_rows_base"].ge(50)
    ].copy()
    supported["ev_delta"] = pd.to_numeric(
        supported["mean_ev_after_1pct_alt"], errors="coerce"
    ) - pd.to_numeric(supported["mean_ev_after_1pct_base"], errors="coerce")
    worst_side_arch_delta = (
        float(supported["ev_delta"].min()) if len(supported) else np.nan
    )
    positive_side_arch_fraction = (
        float(supported["ev_delta"].gt(0.0).mean()) if len(supported) else np.nan
    )
    ac_side_arch = _side_archetype_signed_autocorrelation(
        scored, "score_alternative", "hit_prob_alternative"
    )
    base_ac_side_arch = _side_archetype_signed_autocorrelation(
        scored, "score_current_reference", "hit_prob_current_reference"
    )
    local_ac_abs = float(
        np.nanmean(
            np.abs(ac_side_arch[["positive_ac1", "negative_ac1"]].to_numpy(dtype=float))
        )
    )
    base_local_ac_abs = float(
        np.nanmean(
            np.abs(
                base_ac_side_arch[["positive_ac1", "negative_ac1"]].to_numpy(
                    dtype=float
                )
            )
        )
    )
    global_ac_improvement = float(base_ac_abs - ac_abs)
    local_ac_improvement = float(base_local_ac_abs - local_ac_abs)
    autocorrelation_guard_pass = bool(
        np.isfinite(global_ac_improvement)
        and np.isfinite(local_ac_improvement)
        and global_ac_improvement > 0.0
        and local_ac_improvement > 0.0
    )
    priority = str(priority).strip().lower()
    if priority == "ev_first":
        # Rank revisions primarily by realized top-10 economics. Worst-period
        # economics remain meaningful tie-breakers, while persistence is a
        # secondary diagnostic subject to the separate catastrophe guard.
        objective = (
            500.0 * (mean_ev - base_ev)
            + 20.0 * (worst_week - base_worst_week)
            + 20.0 * (worst_month - base_worst_month)
            + 0.01 * global_ac_improvement
            + 0.005 * local_ac_improvement
        )
    elif priority == "balanced":
        objective = (
            100.0 * (mean_ev - base_ev)
            + 35.0 * (worst_week - base_worst_week)
            + 20.0 * (worst_month - base_worst_month)
            + 0.75 * global_ac_improvement
            + 0.50 * local_ac_improvement
        )
    else:
        raise ValueError(f"Unknown revision priority: {priority}")
    summary = {
        "arm": arm,
        "revision_priority": priority,
        "objective": float(objective),
        "top10_ev": mean_ev,
        "top10_ev_delta": float(mean_ev - base_ev),
        "worst_week_ev": worst_week,
        "worst_week_ev_delta": float(worst_week - base_worst_week),
        "worst_month_ev": worst_month,
        "worst_month_ev_delta": float(worst_month - base_worst_month),
        "positive_ev_rate": float(overall["positive_ev_rate"]),
        "clean_precision": float(overall["clean_exec_precision"]),
        "bad_mae_rate": float(overall["full_path_bad_mae_rate"]),
        "timeout_rate": float(overall["timeout_rate"]),
        "signed_ac1": ac["signed_ac1"],
        "positive_ac1": ac["positive_ac1"],
        "negative_ac1": ac["negative_ac1"],
        "mean_abs_signed_component_ac1": ac_abs,
        "baseline_mean_abs_signed_component_ac1": base_ac_abs,
        "signed_component_autocorrelation_improvement": global_ac_improvement,
        "supported_side_archetypes": int(len(supported)),
        "worst_supported_side_archetype_ev_delta": worst_side_arch_delta,
        "positive_side_archetype_ev_delta_fraction": positive_side_arch_fraction,
        "mean_abs_side_archetype_signed_component_ac1": local_ac_abs,
        "baseline_mean_abs_side_archetype_signed_component_ac1": base_local_ac_abs,
        "side_archetype_autocorrelation_improvement": local_ac_improvement,
        "autocorrelation_guard_pass": autocorrelation_guard_pass,
    }
    return metrics, summary


def _passes_incremental_guard(
    summary: Mapping[str, Any], minimum_objective: float
) -> bool:
    top10_ev_delta = float(summary.get("top10_ev_delta", -np.inf))
    calibration_only = str(summary.get("rank_mode", "")).lower() == "calibration_only"
    ev_guard_pass = (
        top10_ev_delta >= -1e-12 if calibration_only else top10_ev_delta > 0.0
    )
    priority = str(summary.get("revision_priority", "balanced")).lower()
    if priority == "ev_first":
        global_ac = float(summary.get("mean_abs_signed_component_ac1", np.inf))
        base_global_ac = float(
            summary.get("baseline_mean_abs_signed_component_ac1", np.nan)
        )
        local_ac = float(
            summary.get("mean_abs_side_archetype_signed_component_ac1", np.inf)
        )
        base_local_ac = float(
            summary.get("baseline_mean_abs_side_archetype_signed_component_ac1", np.nan)
        )
        return bool(
            float(summary.get("objective", -np.inf)) > float(minimum_objective)
            and top10_ev_delta > 0.0
            and float(summary.get("worst_week_ev_delta", -np.inf)) >= -0.0005
            and float(summary.get("worst_month_ev_delta", -np.inf)) >= -0.0005
            and np.isfinite(global_ac)
            and np.isfinite(base_global_ac)
            # EV-first revisions may trade some persistence for materially better
            # economics. Keep only a catastrophe cap; local structure and tail EV
            # remain separately guarded below.
            and global_ac <= base_global_ac + 0.30
            and np.isfinite(local_ac)
            and np.isfinite(base_local_ac)
            and local_ac <= base_local_ac + 0.10
            and (
                not np.isfinite(
                    float(
                        summary.get("worst_supported_side_archetype_ev_delta", np.nan)
                    )
                )
                or float(summary["worst_supported_side_archetype_ev_delta"]) >= -0.0020
            )
        )
    return bool(
        float(summary.get("objective", -np.inf)) > float(minimum_objective)
        and ev_guard_pass
        and bool(summary.get("autocorrelation_guard_pass", False))
        and float(summary.get("worst_week_ev_delta", -np.inf)) >= -0.0010
        and float(summary.get("worst_month_ev_delta", -np.inf)) >= -0.0005
        and (
            not np.isfinite(
                float(summary.get("worst_supported_side_archetype_ev_delta", np.nan))
            )
            or float(summary["worst_supported_side_archetype_ev_delta"]) >= -0.0020
        )
    )


def _canonical_final_model(
    train: pd.DataFrame,
    evaluation: pd.DataFrame,
    candidate_features: Sequence[str],
    *,
    output_dir: Path,
    trials: int,
    seed: int,
) -> tuple[Any, np.ndarray, dict[str, Any]]:
    train_local, eval_local = _add_reference_fold_features(train, evaluation)
    target, target_column = _base_soft_label_target(train_local)
    valid = target.notna()
    train_local = train_local.loc[valid].reset_index(drop=True)
    target = target.loc[valid].reset_index(drop=True)
    raw_features = [
        name
        for name in dict.fromkeys(map(str, candidate_features))
        if name not in META_POST_SELECTION_OOD_FEATURE_NAMES
        and name in train_local.columns
    ]
    x_train, x_eval, medians = _matrix_fit_transform(
        train_local, eval_local, raw_features
    )
    weights = _base_style_weights_for_soft_label(train_local, target).to_numpy(
        dtype=np.float32
    )
    returns = pd.to_numeric(train_local["ev_after_1pct"], errors="coerce").fillna(0.0)

    # The final union alone pays the full selector/HPO cost.  A 45k beginning /
    # middle / end time-spread race yields about 15k rows in its evaluation tail.
    lp.LGBM_RACE_MAX_ROWS = 45_000
    lp.LGBM_HPO_MAX_ROWS = 45_000
    fast_cfg = {
        "mda_config": {
            "enabled": True,
            "objective": "topk_opportunity_precision",
            "topk_fracs": [0.10, 0.15, 0.20],
            "topk_frac_weights": [0.60, 0.25, 0.15],
            "archetype_conditioned_enabled": True,
            "archetype_global_weight": 0.20,
            "archetype_macro_weight": 0.65,
            "archetype_worst_weight": 0.15,
            "permutation_mode": "path_gated_lgbm",
            "min_repeats": 2,
            "max_repeats": 6,
            "repeat_batch_size": 2,
            "shadow_max_features": 16,
            "shadow_n_repeats": 3,
            "group_first_screen_enabled": True,
            "group_first_screen_kind": "feature_family",
            "group_first_min_repeats": 2,
            "group_first_max_repeats": 4,
            "group_first_drop_null": True,
            "group_mda_enabled": True,
        }
    }
    prior_objective = str(lp.LGBM_OBJECTIVE)
    lp.LGBM_OBJECTIVE = "topk_ev"
    try:
        model = lp.train_lgbm_stability_pipeline(
            x_train,
            target.to_numpy(dtype=np.float32),
            sample_weight=weights,
            random_state=int(seed),
            mode="regressor",
            timestamps=train_local["__ts__"],
            assets=train_local["__symbol__"].astype(str).to_numpy(),
            returns=returns.to_numpy(dtype=np.float32),
            hard_labels=(target.to_numpy(dtype=np.float32) >= 0.5).astype(np.float32),
            hpo_trials_override=int(trials),
            hpo_objective_mode="train_meta",
            reference_artifact_dir=output_dir / "canonical_final_fit",
            cfg=fast_cfg,
            label_context=_feature_selection_label_context(train_local),
        )
    finally:
        lp.LGBM_OBJECTIVE = prior_objective
    if model is None:
        raise RuntimeError("Canonical final selector/HPO returned no model")
    prediction = model.predict(x_eval)
    return (
        model,
        prediction,
        {
            "target_column": target_column,
            "candidate_features": raw_features,
            "selected_features": list(model.selected_features),
            "selected_feature_count": int(len(model.selected_features)),
            "best_params": dict(model.best_params),
            "metrics": dict(model.metrics),
            "medians": medians,
            "mda_profile": fast_cfg["mda_config"],
            "hpo_objective": "topk_ev",
        },
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    cutoff = pd.Timestamp(args.train_cutoff, tz="UTC")
    purge_hours = max(0.0, float(args.purge_hours))
    train_fit_end, default_state_fit_end = _purged_fit_boundaries(cutoff, purge_hours)
    state_fit_end = (
        pd.Timestamp(args.state_fit_end, tz="UTC")
        if args.state_fit_end
        else default_state_fit_end
    )
    if state_fit_end > default_state_fit_end:
        raise ValueError(
            "State fit end must not exceed the purged downstream train boundary"
        )
    downstream_train_start = (
        pd.Timestamp(args.downstream_train_start, tz="UTC")
        if args.downstream_train_start
        else None
    )
    if downstream_train_start is not None and downstream_train_start >= train_fit_end:
        raise ValueError("Downstream train start must precede its fit end")
    selection_end = pd.Timestamp(args.selection_end, tz="UTC")
    evaluation_end = pd.Timestamp(args.evaluation_end, tz="UTC")
    reference_features, params, reference_manifest = _reference_contract(
        args.reference_dir
    )
    data, data_manifest = _load_comparison_data(
        args.compact,
        args.ledger,
        args.july_source,
        args.feature_root,
        reference_features,
        data_start=downstream_train_start,
        evaluation_end=evaluation_end,
    )
    encoder_kinds = [
        name.strip() for name in args.encoder_kinds.split(",") if name.strip()
    ]
    unknown = sorted(set(encoder_kinds) - set(ENCODER_PRESETS))
    if unknown:
        raise ValueError(f"Unknown encoder kinds: {unknown}")
    state_cache_dir = Path(args.state_cache_dir or output)
    frozen_state_features_path = (
        Path(args.frozen_state_features) if args.frozen_state_features else None
    )
    correction_only_blocks = (
        [
            name.strip()
            for name in str(args.residual_correction_blocks).split(",")
            if name.strip()
        ]
        if args.run_residual_correction
        and args.skip_representation_comparison
        and args.skip_soft_label_greedy
        else None
    )
    if correction_only_blocks and any(
        name.startswith("B0_") for name in correction_only_blocks
    ):
        correction_only_blocks = list(
            dict.fromkeys(
                [
                    *correction_only_blocks,
                    "B3_static_state_posteriors",
                ]
            )
        )
    frozen_cache_fast_path = bool(
        args.state_cache_dir
        and (correction_only_blocks or frozen_state_features_path is not None)
        and len(encoder_kinds) == 1
        and not args.force_state_refit
    )
    if frozen_cache_fast_path:
        (
            latent_partitions,
            partition_features,
            state_input_features,
            partition_feature_manifest,
        ) = _load_frozen_state_cache_contract(
            state_cache_dir,
            encoder_kind=encoder_kinds[0],
        )
        if frozen_state_features_path is not None:
            frozen_generated_states = pd.read_parquet(frozen_state_features_path)
            frozen_generated_states["__ts__"] = pd.to_datetime(
                frozen_generated_states["__ts__"], utc=True, errors="coerce"
            )
            states = (
                frozen_generated_states[["__ts__", "side_name"]]
                .drop_duplicates()
                .sort_values(["side_name", "__ts__"], kind="stable")
                .reset_index(drop=True)
            )
        else:
            state_columns = list(
                dict.fromkeys(["__ts__", "side_name", *state_input_features])
            )
            states = pd.read_parquet(args.states, columns=state_columns)
            states["__ts__"] = pd.to_datetime(
                states["__ts__"], utc=True, errors="coerce"
            )
            states = states.sort_values(
                ["side_name", "__ts__"], kind="stable"
            ).reset_index(drop=True)
        signature_manifest = {
            "schema": "frozen_state_cache_fast_path_v1",
            "source": str(state_cache_dir),
            "residual_signature_regenerated": False,
        }
        pd.DataFrame(
            [
                {
                    "state_partition_token": token,
                    "feature": feature,
                    "frozen_rank": rank,
                }
                for token, features in partition_features.items()
                for rank, feature in enumerate(features, start=1)
            ]
        ).to_csv(output / "partition_state_feature_relevance.csv", index=False)
    else:
        states, signature_manifest = _load_states_with_signature(
            args.states, args.ledger
        )
        state_input_features = _state_input_features(states)
        partition_identity_source = (
            _load_partition_identity_history(args.compact, fit_end=state_fit_end)
            if downstream_train_start is not None
            and downstream_train_start >= state_fit_end
            else data
        )
        latent_partitions = _latent_state_partitions(
            partition_identity_source,
            states,
            fit_end=state_fit_end,
        )
        if partition_identity_source is not data:
            del partition_identity_source
            gc.collect()
        partition_features, partition_feature_manifest = _partition_state_feature_sets(
            states,
            state_input_features,
            latent_partitions,
            fit_end=state_fit_end,
            max_features=int(ResidualEncoderConfig().max_input_features),
            output_dir=output,
        )
    covariance_types = tuple(
        name.strip().lower()
        for name in str(args.gmm_covariance).split(",")
        if name.strip()
    )
    invalid_covariance = sorted(
        set(covariance_types) - {"diag", "full", "tied", "spherical"}
    )
    if invalid_covariance:
        raise ValueError(f"Unsupported GMM covariance types: {invalid_covariance}")
    reg_covars = tuple(
        float(value) for value in str(args.gmm_reg_covars).split(",") if value.strip()
    )
    _emit(
        "partition_state_feature_selection_complete",
        partitions={
            token: len(features) for token, features in partition_features.items()
        },
    )

    encoder_outputs: dict[str, pd.DataFrame] = {}
    encoder_manifests: dict[str, Any] = {}
    for idx, kind in enumerate(encoder_kinds):
        _emit(
            "encoder_state_fit_started",
            encoder_kind=kind,
            index=idx + 1,
            total=len(encoder_kinds),
        )
        if frozen_cache_fast_path and frozen_state_features_path is not None:
            generated = frozen_generated_states.copy()
            manifest = {
                "schema": "materialized_frozen_side_archetype_state_v1",
                "encoder_kind": kind,
                "source": str(frozen_state_features_path),
                "fit_granularity": "side_x_archetype",
                "partition_count": int(len(latent_partitions)),
                "required_blocks": correction_only_blocks,
            }
        elif frozen_cache_fast_path:
            generated, manifest = _transform_frozen_state_cache_blocks(
                states,
                latent_partitions,
                encoder_kind=kind,
                cutoff=cutoff,
                fit_end=state_fit_end,
                evaluation_end=evaluation_end,
                state_cache_dir=state_cache_dir,
                required_blocks=correction_only_blocks or (),
            )
        else:
            generated, manifest = _fit_encoder_state_features(
                states,
                state_input_features,
                latent_partitions,
                partition_features,
                encoder_kind=kind,
                cutoff=cutoff,
                fit_end=state_fit_end,
                evaluation_end=evaluation_end,
                output_dir=output,
                state_cache_dir=state_cache_dir,
                latent_dim=int(args.latent_dim),
                epochs=int(args.encoder_epochs),
                components=tuple(
                    int(value) for value in args.gmm_components.split(",")
                ),
                covariance_types=covariance_types,
                reg_covars=reg_covars,
                gmm_n_init=int(args.gmm_n_init),
                seed=int(args.seed + idx * 100_003),
                reuse_existing_state=not bool(args.force_state_refit),
                required_blocks=correction_only_blocks,
            )
        encoder_outputs[kind] = generated
        encoder_manifests[kind] = manifest
        _emit(
            "encoder_state_fit_complete",
            encoder_kind=kind,
            partitions={
                token: {
                    "side_name": partition_manifest.get("side_name"),
                    "archetype_policy_key": partition_manifest.get(
                        "archetype_policy_key"
                    ),
                    "train_rows": partition_manifest.get("train_rows"),
                    "gmm_selected": (partition_manifest.get("gmm") or {}).get(
                        "selected"
                    ),
                }
                for token, partition_manifest in (
                    manifest.get("partitions") or {}
                ).items()
            },
        )

    train_mask = data["__ts__"].lt(train_fit_end)
    if downstream_train_start is not None:
        train_mask &= data["__ts__"].ge(downstream_train_start)
    train = data.loc[train_mask].copy()
    selection = data.loc[
        data["__ts__"].ge(cutoff) & data["__ts__"].lt(selection_end)
    ].copy()
    final_test = data.loc[
        data["__ts__"].ge(selection_end) & data["__ts__"].lt(evaluation_end)
    ].copy()
    if args.smoke:
        train = _time_spread_sample(train, int(args.smoke_train_rows), int(args.seed))
        selection = _time_spread_sample(
            selection, int(args.smoke_selection_rows), int(args.seed + 1)
        )
        final_test = _time_spread_sample(
            final_test, int(args.smoke_test_rows), int(args.seed + 2)
        )
    if len(train) < 5_000 or len(selection) < 500:
        raise ValueError(
            f"Insufficient fixed split support: train={len(train)} selection={len(selection)}"
        )
    baseline_raw = [
        name
        for name in reference_features
        if name not in META_POST_SELECTION_OOD_FEATURE_NAMES
    ]
    revision_fit_seed = int(args.seed + 10_003)
    fixed_reference_eval = pd.concat(
        [selection, final_test], ignore_index=True, copy=False
    )
    fixed_reference_scored, fixed_reference_model, fixed_reference_manifest = (
        _fit_fixed_revision(
            train,
            fixed_reference_eval,
            baseline_raw,
            params,
            arm="fixed_single_fit_reference",
            seed=revision_fit_seed,
        )
    )
    fixed_reference = fixed_reference_scored[
        [*KEY_COLUMNS, "score_alternative", "hit_prob_alternative"]
    ].rename(
        columns={
            "score_alternative": FIXED_REFERENCE_SCORE,
            "hit_prob_alternative": FIXED_REFERENCE_HIT_PROB,
        }
    )
    if fixed_reference.duplicated(list(KEY_COLUMNS)).any():
        raise ValueError("Fixed reference predictions are not unique by row key")
    data = data.merge(
        fixed_reference,
        on=list(KEY_COLUMNS),
        how="left",
        validate="many_to_one",
    )
    for frame in (selection, final_test):
        frame.drop(
            columns=[FIXED_REFERENCE_SCORE, FIXED_REFERENCE_HIT_PROB],
            errors="ignore",
            inplace=True,
        )
    selection = selection.merge(
        fixed_reference,
        on=list(KEY_COLUMNS),
        how="left",
        validate="many_to_one",
    )
    final_test = final_test.merge(
        fixed_reference,
        on=list(KEY_COLUMNS),
        how="left",
        validate="many_to_one",
    )
    del fixed_reference_scored, fixed_reference_model, fixed_reference_eval
    gc.collect()

    # Compare representation families first using the same full static package.
    representation_rows: list[dict[str, Any]] = []
    representation_artifacts: dict[str, Any] = {}
    prior_models: dict[str, SideArchetypeStatePriors] = {}
    for idx, kind in enumerate(encoder_kinds):
        if args.skip_representation_comparison:
            continue
        _emit(
            "encoder_champion_comparison_started",
            encoder_kind=kind,
            index=idx + 1,
            total=len(encoder_kinds),
        )
        enriched = _merge_state_features(data, encoder_outputs[kind])
        enriched, local_context_candidates, prior_model = (
            _append_mandatory_archetype_context(
                enriched,
                encoder_kind=kind,
                fit_end=train_fit_end,
            )
        )
        prior_models[kind] = prior_model
        blocks = _state_blocks(states, encoder_outputs[kind], kind)
        blocks = {
            **_local_context_blocks(local_context_candidates, kind),
            **blocks,
        }
        representation_features = list(
            dict.fromkeys(
                blocks["B0_local_signature_heads"]
                + blocks["B0_local_state_priors"]
                + blocks["B2_encoder_bottleneck_signature"]
                + blocks["B3_static_state_posteriors"]
                + blocks["B4_state_uncertainty"]
            )
        )
        local_train = enriched.loc[enriched["__ts__"].lt(train_fit_end)].copy()
        local_selection = enriched.loc[
            enriched["__ts__"].ge(cutoff) & enriched["__ts__"].lt(selection_end)
        ].copy()
        if args.smoke:
            local_train = _time_spread_sample(
                local_train, int(args.smoke_train_rows), int(args.seed)
            )
            local_selection = _time_spread_sample(
                local_selection,
                int(args.smoke_selection_rows),
                int(args.seed + 1),
            )
        scored, model, fit_manifest = _fit_fixed_revision(
            local_train,
            local_selection,
            baseline_raw + representation_features,
            params,
            arm=f"encoder_{kind}",
            seed=int(args.seed + idx * 101),
        )
        metrics, summary = _revision_summary(
            scored,
            f"encoder_{kind}",
            priority=args.revision_priority,
        )
        representation_rows.append(summary)
        pd.DataFrame(representation_rows).to_csv(
            output / "encoder_comparison.partial.csv", index=False
        )
        representation_artifacts[kind] = {
            "fit": fit_manifest,
            "summary": summary,
            "blocks": blocks,
            "local_archetype_context_candidates": local_context_candidates,
        }
        arm_dir = output / "encoder_comparison" / kind
        arm_dir.mkdir(parents=True, exist_ok=True)
        metrics.to_csv(arm_dir / "metrics.csv", index=False)
        joblib.dump(model, arm_dir / "model.joblib", compress=3)
        _emit(
            "encoder_champion_comparison_complete",
            encoder_kind=kind,
            summary=summary,
        )
        del enriched, local_train, local_selection, scored, model
        gc.collect()
    representation_table = pd.DataFrame(representation_rows)
    if not representation_table.empty:
        representation_table = representation_table.sort_values(
            "objective", ascending=False, kind="stable"
        )
    representation_table.to_csv(output / "encoder_comparison.csv", index=False)
    if representation_table.empty:
        if len(encoder_kinds) != 1:
            raise ValueError(
                "--skip-representation-comparison requires exactly one encoder kind"
            )
        eligible_representations = pd.DataFrame()
        chosen_row = None
        chosen_encoder = encoder_kinds[0]
    else:
        eligible_representations = representation_table.loc[
            representation_table["autocorrelation_guard_pass"]
            .fillna(False)
            .astype(bool)
        ]
        if len(eligible_representations):
            chosen_row = eligible_representations.iloc[0]
        else:
            # Continue the diagnostic search from the least persistence-damaging
            # representation, but keep it explicitly non-promotable.
            chosen_row = representation_table.sort_values(
                [
                    "mean_abs_side_archetype_signed_component_ac1",
                    "mean_abs_signed_component_ac1",
                    "objective",
                ],
                ascending=[True, True, False],
                kind="stable",
            ).iloc[0]
        chosen_encoder = str(chosen_row["arm"]).removeprefix("encoder_")

    chosen_states = encoder_outputs[chosen_encoder]
    enriched = _merge_state_features(data, chosen_states)
    blocks = _state_blocks(states, chosen_states, chosen_encoder)
    # Raw lifecycle coordinates come from the state table, not the encoder output.
    lifecycle_frame = states[["__ts__", "side_name", *blocks["B1_lifecycle_market"]]]
    enriched = _merge_state_features(enriched, lifecycle_frame)
    enriched, local_context_candidates, prior_model = (
        _append_mandatory_archetype_context(
            enriched,
            encoder_kind=chosen_encoder,
            fit_end=train_fit_end,
            prior_model=prior_models.get(chosen_encoder),
        )
    )
    blocks = {
        **_local_context_blocks(local_context_candidates, chosen_encoder),
        **blocks,
    }
    prior_path = output / "states" / f"{chosen_encoder}_side_archetype_priors.joblib"
    prior_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(prior_model, prior_path, compress=3)
    train_mask = enriched["__ts__"].lt(train_fit_end)
    if downstream_train_start is not None:
        train_mask &= enriched["__ts__"].ge(downstream_train_start)
    train = enriched.loc[train_mask].copy()
    selection = enriched.loc[
        enriched["__ts__"].ge(cutoff) & enriched["__ts__"].lt(selection_end)
    ].copy()
    final_test = enriched.loc[
        enriched["__ts__"].ge(selection_end) & enriched["__ts__"].lt(evaluation_end)
    ].copy()
    if args.smoke:
        train = _time_spread_sample(train, int(args.smoke_train_rows), int(args.seed))
        selection = _time_spread_sample(
            selection, int(args.smoke_selection_rows), int(args.seed + 1)
        )
        final_test = _time_spread_sample(
            final_test, int(args.smoke_test_rows), int(args.seed + 2)
        )

    accepted: list[str] = []
    requested_greedy_blocks = {
        name.strip()
        for name in str(args.greedy_blocks or "").split(",")
        if name.strip()
    }
    unknown_greedy_blocks = sorted(requested_greedy_blocks - set(blocks))
    if unknown_greedy_blocks:
        raise ValueError(f"Unknown greedy blocks: {unknown_greedy_blocks}")
    remaining = (
        []
        if args.skip_soft_label_greedy
        else [
            name
            for name, values in blocks.items()
            if values
            and (not requested_greedy_blocks or name in requested_greedy_blocks)
        ]
    )
    greedy_rows: list[dict[str, Any]] = []
    round_index = 0
    accepted_objective = 0.0
    while remaining:
        round_index += 1
        candidates: list[tuple[str, dict[str, Any]]] = []
        for block in remaining:
            features = (
                baseline_raw
                + [
                    feature
                    for accepted_block in accepted
                    for feature in blocks[accepted_block]
                ]
                + blocks[block]
            )
            arm = f"G{round_index}_{block}"
            _emit(
                "greedy_revision_started",
                round=round_index,
                block=block,
                accepted=accepted,
            )
            scored, model, fit_manifest = _fit_fixed_revision(
                train,
                selection,
                features,
                params,
                arm=arm,
                seed=revision_fit_seed,
            )
            metrics, summary = _revision_summary(
                scored,
                arm,
                priority=args.revision_priority,
            )
            summary.update(
                {
                    "round": round_index,
                    "block": block,
                    "accepted_before": "|".join(accepted),
                    "fit_seed": revision_fit_seed,
                    "incremental_objective": float(
                        summary["objective"] - accepted_objective
                    ),
                }
            )
            candidates.append((block, summary))
            _emit(
                "greedy_revision_complete",
                round=round_index,
                block=block,
                summary=summary,
            )
            arm_dir = output / "greedy" / arm
            arm_dir.mkdir(parents=True, exist_ok=True)
            metrics.to_csv(arm_dir / "metrics.csv", index=False)
            _write_json(arm_dir / "fit_manifest.json", fit_manifest)
            joblib.dump(model, arm_dir / "model.joblib", compress=3)
            del scored, model
        greedy_rows.extend(summary for _, summary in candidates)
        winner_block, winner = max(
            candidates, key=lambda item: float(item[1]["objective"])
        )
        _emit(
            "greedy_round_winner",
            round=round_index,
            block=winner_block,
            summary=winner,
        )
        if float(winner["incremental_objective"]) <= float(
            args.minimum_objective
        ) or not _passes_incremental_guard(winner, 0.0):
            break
        accepted.append(winner_block)
        accepted_objective = float(winner["objective"])
        remaining.remove(winner_block)
        if int(args.max_greedy_rounds) > 0 and round_index >= int(
            args.max_greedy_rounds
        ):
            break
    greedy_table = pd.DataFrame(greedy_rows)
    greedy_table.to_csv(output / "greedy_revision_scorecard.csv", index=False)

    final_features = _final_feature_union(
        baseline_raw,
        (),
        accepted,
        blocks,
    )
    final_scored, final_model, final_fit_manifest = (
        _fit_fixed_revision(
            train,
            final_test,
            final_features,
            params,
            arm="final_greedy_champion",
            seed=revision_fit_seed,
        )
        if len(final_test) and not args.skip_soft_label_greedy
        else (pd.DataFrame(), None, {})
    )
    if len(final_scored):
        final_metrics, final_summary = _revision_summary(
            final_scored,
            "final_greedy_champion",
            priority=args.revision_priority,
        )
        final_metrics.to_csv(output / "final_test_metrics.csv", index=False)
        final_scored.to_parquet(
            output / "final_test_predictions.parquet", index=False, compression="zstd"
        )
        joblib.dump(final_model, output / "final_greedy_model.joblib", compress=3)
    else:
        final_summary = {"status": "no_final_test_rows"}

    residual_correction_rows: list[dict[str, Any]] = []
    residual_correction_manifest: dict[str, Any] = {"enabled": False}
    if args.run_residual_correction:
        correction_blocks = [
            name.strip()
            for name in str(args.residual_correction_blocks).split(",")
            if name.strip()
        ]
        unknown_blocks = sorted(set(correction_blocks) - set(blocks))
        if unknown_blocks:
            raise ValueError(f"Unknown residual-correction blocks: {unknown_blocks}")
        correction_scales = [
            float(value)
            for value in str(args.residual_correction_scales).split(",")
            if value.strip()
        ]
        if not correction_scales:
            raise ValueError("At least one residual-correction scale is required")
        favorable_scales = [
            float(value)
            for value in str(args.residual_favorable_scales or "").split(",")
            if value.strip()
        ]
        adverse_scales = [
            float(value)
            for value in str(args.residual_adverse_scales or "").split(",")
            if value.strip()
        ]
        if bool(favorable_scales) != bool(adverse_scales):
            raise ValueError(
                "Favorable and adverse residual scale grids must be provided together"
            )
        asymmetric_scale_grid = bool(favorable_scales and adverse_scales)
        if asymmetric_scale_grid and str(args.residual_correction_target) not in {
            "daily_top10_surprise_two_head",
            "daily_top10_persistence_two_head",
        }:
            raise ValueError(
                "Asymmetric correction scales require the two-head daily target"
            )
        correction_scale_configs = (
            [
                {
                    "scale": 1.0,
                    "favorable_scale": favorable_scale,
                    "adverse_scale": adverse_scale,
                }
                for favorable_scale in favorable_scales
                for adverse_scale in adverse_scales
            ]
            if asymmetric_scale_grid
            else [
                {
                    "scale": scale,
                    "favorable_scale": None,
                    "adverse_scale": None,
                }
                for scale in correction_scales
            ]
        )
        correction_rank_modes = [
            name.strip().lower()
            for name in str(args.residual_correction_rank_modes).split(",")
            if name.strip()
        ]
        invalid_rank_modes = sorted(
            set(correction_rank_modes) - {"rerank", "calibration_only"}
        )
        if invalid_rank_modes:
            raise ValueError(
                f"Unknown residual-correction rank modes: {invalid_rank_modes}"
            )
        correction_eval = pd.concat(
            [selection, final_test], ignore_index=True, copy=False
        )
        correction_cache: dict[str, dict[str, Any]] = {}
        correction_root = output / "residual_correction"
        correction_root.mkdir(parents=True, exist_ok=True)
        for block in correction_blocks:
            block_features = list(blocks.get(block, ()))
            if not block_features:
                continue
            _emit("residual_correction_fit_started", block=block)
            (
                correction,
                base_probability,
                correction_components,
                local_models,
                fit_manifest,
            ) = _fit_local_residual_correction(
                train,
                correction_eval,
                block_features,
                params,
                seed=int(args.seed + 2_300_003 + zlib.crc32(block.encode("utf-8"))),
                target_mode=str(args.residual_correction_target),
                daily_support_shrinkage=float(args.residual_daily_support_shrinkage),
            )
            block_root = correction_root / block
            block_root.mkdir(parents=True, exist_ok=True)
            joblib.dump(local_models, block_root / "models.joblib", compress=3)
            _write_json(block_root / "fit_manifest.json", fit_manifest)
            correction_cache[block] = {
                "correction": correction,
                "base_probability": base_probability,
                "correction_components": correction_components,
                "fit_manifest": fit_manifest,
            }
            correction_vectors = correction_eval[
                [name for name in KEY_COLUMNS if name in correction_eval]
            ].copy()
            correction_vectors["evaluation_partition"] = np.where(
                correction_vectors["__ts__"].lt(selection_end),
                "selection",
                "final_test",
            )
            correction_vectors["reference_hit_probability"] = base_probability
            correction_vectors["local_residual_correction"] = correction
            for component_name, component_values in correction_components.items():
                correction_vectors[f"local_residual_correction_{component_name}"] = (
                    component_values
                )
            correction_vectors.to_parquet(
                block_root / "correction_vectors.parquet",
                index=False,
                compression="zstd",
            )
            for scale_config in correction_scale_configs:
                scale = float(scale_config["scale"])
                favorable_scale = scale_config["favorable_scale"]
                adverse_scale = scale_config["adverse_scale"]
                for rank_mode in correction_rank_modes:
                    arm = (
                        f"RC_{block}_fav{int(round(float(favorable_scale) * 100)):03d}_"
                        f"adv{int(round(float(adverse_scale) * 100)):03d}_{rank_mode}"
                        if asymmetric_scale_grid
                        else (f"RC_{block}_s{int(round(scale * 100)):03d}_{rank_mode}")
                    )
                    scored = _score_local_residual_correction(
                        selection,
                        correction=correction[: len(selection)],
                        base_probability=base_probability[: len(selection)],
                        scale=scale,
                        arm=arm,
                        rank_mode=rank_mode,
                        favorable_correction=(
                            correction_components["favorable"][: len(selection)]
                            if asymmetric_scale_grid
                            else None
                        ),
                        adverse_correction=(
                            correction_components["adverse"][: len(selection)]
                            if asymmetric_scale_grid
                            else None
                        ),
                        favorable_scale=(
                            float(favorable_scale)
                            if favorable_scale is not None
                            else None
                        ),
                        adverse_scale=(
                            float(adverse_scale) if adverse_scale is not None else None
                        ),
                    )
                    metrics, summary = _revision_summary(
                        scored,
                        arm,
                        priority=args.revision_priority,
                    )
                    summary.update(
                        {
                            "block": block,
                            "scale": float(scale),
                            "favorable_scale": favorable_scale,
                            "adverse_scale": adverse_scale,
                            "rank_mode": rank_mode,
                        }
                    )
                    summary["incremental_guard_pass"] = _passes_incremental_guard(
                        summary, 0.0
                    )
                    residual_correction_rows.append(summary)
                    metrics.to_csv(
                        block_root
                        / (
                            f"selection_metrics_fav{float(favorable_scale):.2f}_"
                            f"adv{float(adverse_scale):.2f}_{rank_mode}.csv"
                            if asymmetric_scale_grid
                            else f"selection_metrics_s{scale:.2f}_{rank_mode}.csv"
                        ),
                        index=False,
                    )
                    _emit(
                        "residual_correction_scale_complete",
                        block=block,
                        scale=float(scale),
                        rank_mode=rank_mode,
                        summary=summary,
                    )
            del local_models
            gc.collect()
        correction_table = pd.DataFrame(residual_correction_rows)
        correction_table.to_csv(
            correction_root / "selection_scorecard.csv", index=False
        )
        if correction_table.empty:
            raise RuntimeError("Residual-correction run produced no valid arms")
        eligible_corrections = correction_table.loc[
            correction_table["incremental_guard_pass"].fillna(False).astype(bool)
        ]
        chosen_correction = (
            eligible_corrections.sort_values(
                "objective", ascending=False, kind="stable"
            ).iloc[0]
            if len(eligible_corrections)
            else correction_table.sort_values(
                [
                    "mean_abs_side_archetype_signed_component_ac1",
                    "mean_abs_signed_component_ac1",
                    "objective",
                ],
                ascending=[True, True, False],
                kind="stable",
            ).iloc[0]
        )
        chosen_block = str(chosen_correction["block"])
        chosen_scale = float(chosen_correction["scale"])
        chosen_rank_mode = str(chosen_correction["rank_mode"])
        chosen_favorable_scale = (
            float(chosen_correction["favorable_scale"])
            if pd.notna(chosen_correction.get("favorable_scale"))
            else None
        )
        chosen_adverse_scale = (
            float(chosen_correction["adverse_scale"])
            if pd.notna(chosen_correction.get("adverse_scale"))
            else None
        )
        chosen_cache = correction_cache[chosen_block]
        final_correction = np.asarray(chosen_cache["correction"])[len(selection) :]
        final_base_probability = np.asarray(chosen_cache["base_probability"])[
            len(selection) :
        ]
        final_arm = (
            f"RC_{chosen_block}_fav{int(round(chosen_favorable_scale * 100)):03d}_"
            f"adv{int(round(chosen_adverse_scale * 100)):03d}_"
            f"{chosen_rank_mode}_final"
            if chosen_favorable_scale is not None and chosen_adverse_scale is not None
            else (
                f"RC_{chosen_block}_s{int(round(chosen_scale * 100)):03d}_"
                f"{chosen_rank_mode}_final"
            )
        )
        chosen_components = chosen_cache.get("correction_components") or {}
        correction_final_scored = _score_local_residual_correction(
            final_test,
            correction=final_correction,
            base_probability=final_base_probability,
            scale=chosen_scale,
            arm=final_arm,
            rank_mode=chosen_rank_mode,
            favorable_correction=(
                np.asarray(chosen_components["favorable"])[len(selection) :]
                if chosen_favorable_scale is not None
                else None
            ),
            adverse_correction=(
                np.asarray(chosen_components["adverse"])[len(selection) :]
                if chosen_adverse_scale is not None
                else None
            ),
            favorable_scale=chosen_favorable_scale,
            adverse_scale=chosen_adverse_scale,
        )
        correction_final_metrics, correction_final_summary = _revision_summary(
            correction_final_scored,
            final_arm,
            priority=args.revision_priority,
        )
        correction_final_summary.update(
            {
                "block": chosen_block,
                "scale": chosen_scale,
                "favorable_scale": chosen_favorable_scale,
                "adverse_scale": chosen_adverse_scale,
                "rank_mode": chosen_rank_mode,
            }
        )
        correction_final_summary["incremental_guard_pass"] = _passes_incremental_guard(
            correction_final_summary, 0.0
        )
        correction_final_scored.to_parquet(
            correction_root / "final_test_predictions.parquet",
            index=False,
            compression="zstd",
        )
        correction_final_metrics.to_csv(
            correction_root / "final_test_metrics.csv", index=False
        )
        residual_correction_manifest = {
            "enabled": True,
            "fit_granularity": "side_x_archetype",
            "blocks": correction_blocks,
            "scales": correction_scales,
            "favorable_scales": favorable_scales,
            "adverse_scales": adverse_scales,
            "rank_modes": correction_rank_modes,
            "selection_arms": residual_correction_rows,
            "chosen_selection": chosen_correction.to_dict(),
            "final_test_summary": correction_final_summary,
            "promotion_guard_pass": bool(
                chosen_correction["incremental_guard_pass"]
                and correction_final_summary["incremental_guard_pass"]
            ),
        }
        _write_json(correction_root / "manifest.json", residual_correction_manifest)

    canonical_manifest: dict[str, Any] | None = None
    canonical_evaluation_summary: dict[str, Any] | None = None
    if args.run_final_selection_hpo:
        requested_canonical_blocks = [
            name.strip()
            for name in str(args.canonical_blocks or "").split(",")
            if name.strip()
        ]
        unknown_canonical_blocks = sorted(set(requested_canonical_blocks) - set(blocks))
        if unknown_canonical_blocks:
            raise ValueError(
                f"Unknown canonical feature blocks: {unknown_canonical_blocks}"
            )
        canonical_blocks = requested_canonical_blocks or list(accepted)
        canonical_features = _final_feature_union(
            baseline_raw,
            (),
            canonical_blocks,
            blocks,
        )
        _emit(
            "canonical_feature_selection_hpo_started",
            trials=int(args.final_hpo_trials),
            candidate_features=len(canonical_features),
            candidate_blocks=canonical_blocks,
        )
        canonical_train = train
        canonical_eval = pd.concat(
            [selection, final_test], ignore_index=True, copy=False
        )
        canonical_model, canonical_prediction, canonical_manifest = (
            _canonical_final_model(
                canonical_train,
                canonical_eval,
                canonical_features,
                output_dir=output,
                trials=int(args.final_hpo_trials),
                seed=int(args.seed + 1_700_003),
            )
        )
        joblib.dump(
            canonical_model, output / "canonical_final_fit/model.joblib", compress=3
        )
        np.save(
            output / "canonical_final_fit/evaluation_predictions.npy",
            canonical_prediction,
        )
        canonical_scored = _score_external_prediction_vector(
            canonical_train,
            canonical_eval,
            alternative_score=canonical_prediction,
            alternative_train_oof_score=np.asarray(canonical_model.oof_probs),
            arm="canonical_final_hpo",
        )
        canonical_selection_rows = canonical_scored.loc[
            canonical_scored["__ts__"].lt(selection_end)
        ].reset_index(drop=True)
        canonical_selection_metrics, canonical_selection_summary = _revision_summary(
            canonical_selection_rows,
            "canonical_final_hpo",
            priority=args.revision_priority,
        )
        canonical_final_rows = canonical_scored.loc[
            canonical_scored["__ts__"].ge(selection_end)
        ].reset_index(drop=True)
        canonical_final_metrics, canonical_final_summary = _revision_summary(
            canonical_final_rows,
            "canonical_final_hpo",
            priority=args.revision_priority,
        )
        canonical_scored.to_parquet(
            output / "canonical_final_fit/evaluation_scored.parquet",
            index=False,
            compression="zstd",
        )
        canonical_selection_metrics.to_csv(
            output / "canonical_final_fit/selection_metrics.csv",
            index=False,
        )
        canonical_final_metrics.to_csv(
            output / "canonical_final_fit/final_test_metrics.csv",
            index=False,
        )
        canonical_manifest["selection_summary"] = canonical_selection_summary
        canonical_manifest["final_test_summary"] = canonical_final_summary
        canonical_manifest["candidate_blocks"] = canonical_blocks
        canonical_manifest["evaluation_rows"] = int(len(canonical_scored))
        canonical_manifest["selection_rows"] = int(len(canonical_selection_rows))
        canonical_manifest["final_test_rows"] = int(len(canonical_final_rows))
        canonical_manifest["promotion_guard_pass"] = bool(
            _passes_incremental_guard(canonical_selection_summary, 0.0)
            and _passes_incremental_guard(canonical_final_summary, 0.0)
        )
        _write_json(output / "canonical_final_fit/manifest.json", canonical_manifest)
        _emit(
            "canonical_feature_selection_hpo_complete",
            selected_features=canonical_manifest.get("selected_feature_count"),
            best_params=canonical_manifest.get("best_params"),
            selection_summary=canonical_selection_summary,
            final_test_summary=canonical_final_summary,
        )
        canonical_evaluation_summary = canonical_selection_summary

    manifest = {
        "schema": "global_residual_champion_greedy_enhancement_v1",
        "revision_priority": str(args.revision_priority),
        "train_cutoff_exclusive": str(cutoff),
        "downstream_train_start_inclusive": (
            str(downstream_train_start) if downstream_train_start is not None else None
        ),
        "train_label_fit_end_exclusive": str(train_fit_end),
        "state_signature_fit_end_exclusive": str(state_fit_end),
        "label_purge_hours": float(purge_hours),
        "greedy_selection_end_exclusive": str(selection_end),
        "final_evaluation_end_exclusive": str(evaluation_end),
        "chosen_encoder": chosen_encoder,
        "chosen_encoder_autocorrelation_guard_pass": (
            bool(chosen_row["autocorrelation_guard_pass"])
            if chosen_row is not None
            else None
        ),
        "encoder_autocorrelation_guard_any_pass": bool(len(eligible_representations)),
        "encoder_comparison": representation_rows,
        "accepted_blocks": accepted,
        "rejected_or_unselected_blocks": [
            name for name in blocks if name not in accepted
        ],
        "state_sequence_model": "none",
        "temporal_posterior_features": {
            "block": "B5_state_transitions",
            "mandatory": False,
            "accepted": "B5_state_transitions" in accepted,
            "features": blocks.get("B5_state_transitions", []),
        },
        "final_feature_union": final_features,
        "mandatory_archetype_context": [],
        "local_archetype_context_candidates": local_context_candidates,
        "final_test_summary": final_summary,
        "canonical_evaluation_summary": canonical_evaluation_summary,
        "canonical_final_selection_hpo": canonical_manifest,
        "residual_correction": residual_correction_manifest,
        "reference_manifest": str(args.reference_dir / "manifest.json"),
        "reference_params": params,
        "fixed_single_fit_reference": fixed_reference_manifest,
        "reference_selected_features": reference_features,
        "data": data_manifest,
        "global_residual_signature": signature_manifest,
        "partition_state_feature_selection": partition_feature_manifest,
        "latent_partition_contract": {
            "fit_granularity": "side_x_archetype",
            "partitions": latent_partitions,
            "fallback": "none",
            "state_cache_dir": str(state_cache_dir),
            "routing_keys": [
                "side_name",
                "archetype_policy_key",
            ],
        },
        "encoder_manifests": encoder_manifests,
        "side_archetype_state_priors": {
            **prior_model.manifest(),
            "state_path": str(prior_path),
        },
        "leakage_contract": {
            "residual_signature": "train-label target only; no recent realized failure inputs",
            "label_purge": (
                f"row labels stop at {train_fit_end}; daily signature state fit stops at "
                f"{state_fit_end}"
            ),
            "encoders_gmm_scalers": (
                f"fit before {state_fit_end} and frozen from {cutoff} onward"
            ),
            "meta_fit": "one fixed fit per comparison revision; no growing monthly refits",
            "downstream_train_window": (
                f"[{downstream_train_start}, {train_fit_end})"
                if downstream_train_start is not None
                else f"(-inf, {train_fit_end})"
            ),
            "greedy_selection": "April-June only",
            "final_test": "July is evaluated only for the selected greedy revision",
            "base_predictions": "fixed current champion top30 handoff",
        },
        "cost_contract": "ev_after_1pct includes the 1% round-trip cost exactly once",
    }
    _write_json(output / "manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compact", type=Path, default=DEFAULT_COMPACT)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument(
        "--july-source",
        type=Path,
        default=DEFAULT_JULY_SOURCE,
        help="Complete frozen July top30 prediction shard for final evaluation.",
    )
    parser.add_argument("--states", type=Path, default=DEFAULT_STATES)
    parser.add_argument("--feature-root", type=Path, default=DEFAULT_FEATURE_ROOT)
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--state-cache-dir",
        type=Path,
        default=None,
        help=(
            "Optional prior run root containing frozen side x archetype bundles under "
            "states/. Exact partition, feature, split, and model-config parity is required."
        ),
    )
    parser.add_argument(
        "--frozen-state-features",
        type=Path,
        default=None,
        help=(
            "Optional pre-materialized outputs from the exact frozen side x archetype "
            "state bundles. Used only by correction-only cache runs."
        ),
    )
    parser.add_argument("--train-cutoff", default="2026-04-01")
    parser.add_argument(
        "--state-fit-end",
        default=None,
        help="Optional earlier exclusive cutoff for feature selection and AE/MLP/GMM fit.",
    )
    parser.add_argument(
        "--downstream-train-start",
        default=None,
        help="Optional lower bound for rows used to fit downstream comparison models.",
    )
    parser.add_argument(
        "--purge-hours",
        type=float,
        default=12.0,
        help="Exclude this forward-label horizon before the meta train cutoff.",
    )
    parser.add_argument("--selection-end", default="2026-07-01")
    parser.add_argument("--evaluation-end", default="2026-07-11")
    parser.add_argument(
        "--encoder-kinds",
        default="unsupervised_ae,residual_aware_ae,supervised_mlp,hybrid_mlp",
    )
    parser.add_argument("--latent-dim", type=int, default=12)
    parser.add_argument("--encoder-epochs", type=int, default=160)
    parser.add_argument("--gmm-components", default="4,6,8,10,12")
    parser.add_argument("--gmm-covariance", default="diag,full")
    parser.add_argument("--gmm-reg-covars", default="0.0001,0.001")
    parser.add_argument("--gmm-n-init", type=int, default=3)
    parser.add_argument("--force-state-refit", action="store_true")
    parser.add_argument("--minimum-objective", type=float, default=0.0)
    parser.add_argument(
        "--revision-priority",
        choices=("balanced", "ev_first"),
        default="ev_first",
        help="Use net-EV-first promotion or the stricter persistence-balanced mode.",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-train-rows", type=int, default=12_000)
    parser.add_argument("--smoke-selection-rows", type=int, default=3_000)
    parser.add_argument("--smoke-test-rows", type=int, default=1_000)
    parser.add_argument("--max-greedy-rounds", type=int, default=0)
    parser.add_argument(
        "--greedy-blocks",
        default=None,
        help="Optional comma-separated state blocks allowed in greedy revision search.",
    )
    parser.add_argument("--skip-representation-comparison", action="store_true")
    parser.add_argument("--skip-soft-label-greedy", action="store_true")
    parser.add_argument("--run-residual-correction", action="store_true")
    parser.add_argument(
        "--residual-correction-blocks",
        default="B3_static_state_posteriors,B4_state_uncertainty,B5_state_transitions",
    )
    parser.add_argument(
        "--residual-correction-scales",
        default="0.25,0.5,0.75,1.0",
    )
    parser.add_argument(
        "--residual-favorable-scales",
        default=None,
        help="Optional comma-separated favorable-head scale grid.",
    )
    parser.add_argument(
        "--residual-adverse-scales",
        default=None,
        help="Optional comma-separated adverse-head scale grid.",
    )
    parser.add_argument(
        "--residual-correction-rank-modes",
        default="rerank",
        help="Comma-separated: rerank, calibration_only.",
    )
    parser.add_argument(
        "--residual-correction-target",
        choices=(
            "row_clean_residual",
            "daily_top10_surprise",
            "daily_top10_surprise_two_head",
            "daily_top10_persistence_two_head",
        ),
        default="row_clean_residual",
    )
    parser.add_argument(
        "--residual-daily-support-shrinkage",
        type=float,
        default=20.0,
    )
    parser.add_argument("--run-final-selection-hpo", action="store_true")
    parser.add_argument(
        "--canonical-blocks",
        default=None,
        help=(
            "Optional comma-separated state blocks offered to canonical feature "
            "selection/HPO even when fixed-parameter greedy selection rejects them."
        ),
    )
    parser.add_argument("--final-hpo-trials", type=int, default=150)
    parser.add_argument("--seed", type=int, default=20260711)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = run(args)
    print(
        json.dumps(
            _safe(
                {
                    "status": "complete",
                    "output": str(args.output),
                    "chosen_encoder": manifest["chosen_encoder"],
                    "accepted_blocks": manifest["accepted_blocks"],
                }
            ),
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
