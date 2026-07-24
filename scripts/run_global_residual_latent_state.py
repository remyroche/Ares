#!/usr/bin/env python3
"""Materialize and validate per-archetype residual-aware AE/GMM states."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.spatial.distance import jensenshannon

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.config import CFG  # noqa: E402
from extreme_price_movements.data_store import read_symbol_features  # noqa: E402
from extreme_price_movements.global_residual_latent_state import (  # noqa: E402
    GlobalGMMStateModel,
    GMMGridConfig,
    ResidualAEConfig,
    ResidualAwareAutoencoder,
    StateVectorConfig,
    _aggregate_features,
    add_causal_phase_state_features,
    add_temporal_state_features,
    archetype_state_token,
    build_global_residual_signature,
    build_side_timestamp_states,
    centroid_matched_ari,
    prepare_archetype_state_partition,
    select_partition_state_features,
    select_state_features,
    state_recognition_metrics,
)
from scripts.score_compare_meta_residual_july_oos import (  # noqa: E402
    _append_store_features,
)

DISCOVERY_ROOT = ROOT / "data_perp/reports/global_residual_state_discovery_20260711_v1"
SOURCE_ROOT = ROOT / (
    "data_perp/reports/train_meta_residual_archetype_enhancement_20260711_v1/"
    "champion_frozen_single_source_202501_20260710"
)
DEFAULT_CANDIDATES = SOURCE_ROOT / "candidate_shards"
DEFAULT_LEDGER = SOURCE_ROOT / "frozen_champion_single_source_ledger.parquet"
DEFAULT_JULY_SOURCE = SOURCE_ROOT / "prediction_shards/predictions_2026-07.parquet"
DEFAULT_FEATURE_ROOT = ROOT / "data_perp/features/20260710_170000"
DEFAULT_OUTPUT = DISCOVERY_ROOT / "global_side_latent_states"

KEY_COLUMNS = (
    "row_id",
    "__ts__",
    "__symbol__",
    "side_name",
    "archetype_policy_key",
)
LEDGER_OVERLAY_COLUMNS = (
    "row_id",
    "score_meta_base_soft_label",
    "selected_for_monitor",
    "hit_probability",
    "ev_after_1pct",
    "clean_exec",
    "full_path_bad_mae_1r",
    "timeout",
)


def _safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(_safe(value), indent=2, sort_keys=True), encoding="utf-8"
    )


def _parquet_columns(path: Path) -> list[str]:
    return [str(name) for name in pq.ParquetFile(path).schema_arrow.names]


def _downcast(frame: pd.DataFrame) -> pd.DataFrame:
    for name in frame.select_dtypes(include=["float64"]).columns:
        frame[name] = pd.to_numeric(frame[name], downcast="float")
    for name in frame.select_dtypes(include=["int64"]).columns:
        frame[name] = pd.to_numeric(frame[name], downcast="integer")
    return frame


def _refresh_store_features(
    frame: pd.DataFrame,
    feature_root: Path,
    requested: list[str],
) -> pd.DataFrame:
    refresh = []
    for name in requested:
        if name not in frame.columns:
            refresh.append(name)
            continue
        values = pd.to_numeric(frame[name], errors="coerce")
        if float(values.notna().mean()) < 0.65:
            refresh.append(name)
    if not refresh:
        return frame
    frame = frame.drop(columns=[name for name in refresh if name in frame.columns])
    enriched, _ = _append_store_features(frame, feature_root, refresh)
    return enriched


def _read_feature_sample(candidate_root: Path, feature_root: Path) -> pd.DataFrame:
    first = sorted(candidate_root.glob("candidates_*.parquet"))[0]
    sample = pd.read_parquet(first).head(80_000)
    return _refresh_store_features(
        sample,
        feature_root,
        list(CFG.get("CRASH_LIFECYCLE_NEW_FEATURE_KEYS", [])),
    )


BROAD_STATE_FEATURE_FAMILIES = (
    "MODEL_DIRECT_BASE_FEATURE_KEYS",
    "MODEL_REGIME_CONTEXT_META_FEATURE_KEYS",
    "MODEL_REGIME_XS_META_FEATURE_KEYS",
    "MODEL_REGIME_TAIL_META_FEATURE_KEYS",
    "MODEL_REGIME_EIGEN_META_FEATURE_KEYS",
    "MARKET_SPECTRAL_POSITION_META_FEATURE_KEYS",
    "ORDERBOOK_META_FEATURE_KEYS",
    "FUNDING_META_FEATURE_KEYS",
    "CROSS_ASSET_FEATURE_KEYS",
    "CROSS_ASSET_META_FEATURE_KEYS",
    "CHANGE_POINT_REGIME_FEATURE_KEYS",
    "RESIDUAL_META_FEATURE_KEYS",
    "CRASH_LIFECYCLE_NEW_FEATURE_KEYS",
)

CALENDAR_OBSERVABLE_FAMILY_PATTERNS: dict[str, tuple[str, ...]] = {
    "btc_eth_alt_rotation": (
        "btc_ret_",
        "eth_ret_",
        "eth_btc_ret_",
        "ret_resid_btc",
        "ret_resid_eth",
        "asset_ret_vs_btc",
    ),
    "cross_asset_synchronization": (
        "return_dispersion",
        "market_dispersion",
        "cross_asset_corr",
        "market_pc1",
        "first_pc_variance",
        "downside_pairwise_corr",
    ),
    "downside_volatility": (
        "downside_semivariance",
        "downside_semivol",
        "mkt_rv_",
        "realized_vol",
    ),
    "volume_liquidity": (
        "volume_z",
        "climax_volume",
        "amihud",
        "spread",
        "liquidity",
        "quote_volume",
    ),
    "oi_liquidation_lifecycle": (
        "oi_flush",
        "oi_drawdown",
        "oi_recovery",
        "liquidation",
    ),
    "funding_crowding": ("funding",),
}


def _broad_state_feature_candidates(
    candidate_root: Path,
    feature_root: Path,
) -> tuple[list[str], dict[str, list[str]]]:
    candidate_schema = set(
        _parquet_columns(sorted(candidate_root.glob("candidates_*.parquet"))[0])
    )
    feature_paths = sorted(feature_root.glob("symbol=*.parquet"))
    store_schema = set(_parquet_columns(feature_paths[0])) if feature_paths else set()
    available = candidate_schema | store_schema
    by_family: dict[str, list[str]] = {}
    ordered: list[str] = []
    for family in BROAD_STATE_FEATURE_FAMILIES:
        values = [str(name) for name in CFG.get(family, []) if str(name) in available]
        by_family[family] = list(dict.fromkeys(values))
        ordered.extend(values)
    return list(dict.fromkeys(ordered)), by_family


def _derive_row_economic_targets(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy(deep=False)
    hit = pd.to_numeric(output["clean_exec"], errors="coerce")
    probability = pd.to_numeric(output["hit_probability"], errors="coerce")
    ev = pd.to_numeric(output["ev_after_1pct"], errors="coerce")
    output["target_negative_surprise"] = np.maximum(probability - hit, 0.0).astype(
        np.float32
    )
    output["target_positive_surprise"] = np.maximum(hit - probability, 0.0).astype(
        np.float32
    )
    output["target_negative_ev"] = np.maximum(-ev, 0.0).astype(np.float32)
    output["target_mean_ev"] = ev.astype(np.float32)
    output["target_bad_mae_rate"] = pd.to_numeric(
        output["full_path_bad_mae_1r"], errors="coerce"
    ).astype(np.float32)
    output["target_timeout_rate"] = pd.to_numeric(
        output["timeout"], errors="coerce"
    ).astype(np.float32)
    output["target_payoff_asymmetry"] = (ev * np.where(hit.gt(0.5), 1.0, -1.0)).astype(
        np.float32
    )
    return output


def _broad_binned_mi_preselection(
    *,
    candidate_root: Path,
    feature_root: Path,
    ledger_path: Path,
    output: Path,
    train_cutoff: pd.Timestamp,
    sample_rows: int,
    top_per_archetype: int,
    max_union_features: int,
) -> tuple[list[str], list[str], dict[str, Any]]:
    broad, by_family = _broad_state_feature_candidates(candidate_root, feature_root)
    paths = [
        path
        for path in sorted(candidate_root.glob("candidates_*.parquet"))
        if pd.Timestamp(
            pd.Period(path.stem.removeprefix("candidates_")).start_time, tz="UTC"
        )
        < train_cutoff
    ]
    if not paths or not broad:
        raise ValueError("No train-period broad state features are available")
    anchor_indices = np.unique(np.linspace(0, len(paths) - 1, 3, dtype=np.int64))
    anchors = [paths[int(index)] for index in anchor_indices]
    rows_per_anchor = max(1, int(sample_rows) // len(anchors))
    sample_parts: list[pd.DataFrame] = []
    identity = ["row_id", "__ts__", "__symbol__", "side_name", "archetype_policy_key"]
    for path in anchors:
        schema = set(_parquet_columns(path))
        columns = [name for name in identity + broad if name in schema]
        part = pd.read_parquet(path, columns=columns)
        if len(part) > rows_per_anchor:
            positions = np.linspace(0, len(part) - 1, rows_per_anchor, dtype=np.int64)
            part = part.iloc[positions].copy()
        sample_parts.append(part)
    sample = pd.concat(sample_parts, ignore_index=True, sort=False, copy=False)
    sample = _refresh_store_features(sample, feature_root, broad)
    ledger = pd.read_parquet(ledger_path, columns=list(LEDGER_OVERLAY_COLUMNS))
    ledger = ledger.drop_duplicates("row_id", keep="last").set_index("row_id")
    overlay = ledger.reindex(sample["row_id"].to_numpy()).reset_index(drop=True)
    for name in LEDGER_OVERLAY_COLUMNS:
        if name != "row_id":
            sample[name] = overlay[name].to_numpy()
    sample = _derive_row_economic_targets(_downcast(sample))

    selected_by_partition: dict[str, list[str]] = {}
    relevance_parts: list[pd.DataFrame] = []
    for (side, archetype), local in sample.groupby(
        ["side_name", "archetype_policy_key"], observed=True
    ):
        selected, relevance = select_partition_state_features(
            local.reset_index(drop=True),
            broad,
            max_features=int(top_per_archetype),
            min_coverage=0.50,
            max_rows=max(2_000, len(local)),
        )
        side_name = str(side).strip().lower()
        key = archetype_state_token(side_name, str(archetype))
        selected_by_partition[key] = selected
        if not relevance.empty:
            report = relevance.copy(deep=False)
            report["side_name"] = side_name
            report["archetype_policy_key"] = str(archetype)
            report["state_partition_token"] = key
            relevance_parts.append(report)
    if not selected_by_partition:
        raise ValueError(
            "Binned-MI preselection found no supported side x archetype features"
        )

    selected_union: list[str] = []
    for rank in range(int(top_per_archetype)):
        for partition in sorted(selected_by_partition):
            values = selected_by_partition[partition]
            if rank < len(values) and values[rank] not in selected_union:
                selected_union.append(values[rank])
                if len(selected_union) >= int(max_union_features):
                    break
        if len(selected_union) >= int(max_union_features):
            break
    relevance_table = (
        pd.concat(relevance_parts, ignore_index=True, sort=False)
        if relevance_parts
        else pd.DataFrame()
    )
    family_anchors: dict[str, list[str]] = {}
    if not relevance_table.empty:
        feature_relevance = (
            relevance_table.groupby("feature", observed=True)["relevance"]
            .agg(["max", "mean"])
            .sort_values(["max", "mean"], ascending=False, kind="stable")
        )
        for family, patterns in CALENDAR_OBSERVABLE_FAMILY_PATTERNS.items():
            eligible = [
                str(name)
                for name in feature_relevance.index
                if any(pattern in str(name).lower() for pattern in patterns)
            ]
            selected_anchors = eligible[:2]
            family_anchors[family] = selected_anchors
            for name in selected_anchors:
                if name not in selected_union and len(selected_union) < int(
                    max_union_features
                ):
                    selected_union.append(name)

    market_family_names = {
        "MODEL_REGIME_CONTEXT_META_FEATURE_KEYS",
        "MODEL_REGIME_XS_META_FEATURE_KEYS",
        "MODEL_REGIME_TAIL_META_FEATURE_KEYS",
        "MODEL_REGIME_EIGEN_META_FEATURE_KEYS",
        "MARKET_SPECTRAL_POSITION_META_FEATURE_KEYS",
        "CROSS_ASSET_META_FEATURE_KEYS",
        "CROSS_ASSET_FEATURE_KEYS",
        "CHANGE_POINT_REGIME_FEATURE_KEYS",
    }
    market_candidates = {
        name
        for family, names in by_family.items()
        if family in market_family_names
        for name in names
    }
    market = [
        name
        for name in selected_union
        if name in market_candidates
        or str(name).lower().startswith(
            ("mkt_", "market_", "cross_asset_", "pct_assets_")
        )
    ]
    asset = [name for name in selected_union if name not in set(market)]
    relevance_table.to_csv(output / "raw_feature_binned_mi_relevance.csv", index=False)
    manifest = {
        "schema": "per_archetype_broad_binned_mi_preselection_v1",
        "fit_partition": "side_name_x_archetype_policy_key",
        "train_cutoff": train_cutoff,
        "anchor_months": [path.stem.removeprefix("candidates_") for path in anchors],
        "sample_rows": len(sample),
        "candidate_feature_count": len(broad),
        "selected_union_count": len(selected_union),
        "market_feature_count": len(market),
        "asset_feature_count": len(asset),
        "top_per_archetype": int(top_per_archetype),
        "max_union_features": int(max_union_features),
        "selected_by_partition": selected_by_partition,
        "calendar_observable_family_anchors": family_anchors,
        "feature_families": by_family,
        "leakage_contract": (
            "Feature relevance uses only rows before the purged search-validation boundary; "
            "April-July outcomes are unavailable to this preselection."
        ),
    }
    _write_json(output / "raw_feature_binned_mi_manifest.json", manifest)
    return market, asset, manifest


def _full_universe_month_aggregate(
    feature_root: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
    features: list[str],
    eligible_symbols: set[str] | None = None,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for path in sorted(feature_root.glob("symbol=*.parquet")):
        symbol = path.stem.removeprefix("symbol=").replace("_USD:USD", "/USD:USD")
        if eligible_symbols and symbol not in eligible_symbols:
            continue
        data = read_symbol_features(
            str(path),
            columns=features,
            start_ts=start,
            end_ts=end - pd.Timedelta(nanoseconds=1),
        )
        if data.empty:
            continue
        data = data.reset_index()
        timestamp_name = "ts" if "ts" in data.columns else data.columns[0]
        data = data.rename(columns={timestamp_name: "__ts__"})
        data["__symbol__"] = symbol
        parts.append(
            data[["__ts__", "__symbol__", *[name for name in features if name in data]]]
        )
    if not parts:
        return pd.DataFrame(columns=["__ts__"])
    universe = pd.concat(parts, ignore_index=True, sort=False, copy=False)
    universe["__ts__"] = pd.to_datetime(universe["__ts__"], utc=True, errors="coerce")
    aggregate = _aggregate_features(
        universe,
        ["__ts__"],
        [name for name in features if name in universe],
        "full_universe__",
        include_quantiles=True,
    )
    del universe, parts
    gc.collect()
    return aggregate


def _merge_full_universe(
    states: pd.DataFrame,
    full_universe: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:
    if full_universe.empty:
        return states
    output = states.merge(
        full_universe, on="__ts__", how="left", validate="many_to_one"
    )
    for name in features:
        selected = f"selected__median__{name}"
        full = f"full_universe__median__{name}"
        if selected in output and full in output:
            output[f"selected_minus_full_universe__{name}"] = (
                pd.to_numeric(output[selected], errors="coerce")
                - pd.to_numeric(output[full], errors="coerce")
            ).astype(np.float32)
    # Prefer full-universe market aggregates where available, then recompute
    # the fixed phase coordinates. This does not refit any transform and keeps
    # the source representation identical for train/OOS assignment.
    output, _ = add_causal_phase_state_features(output)
    return output


def _hydrate_missing_state_targets(
    states: pd.DataFrame,
    ledger_path: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fill only missing retrospective targets from the frozen selected ledger.

    This is necessary for an imported state table whose July pre-entry rows were
    materialized before realized outcomes were appended. It does not add a
    target-derived feature: the values remain ``target_*`` columns used only for
    train auxiliaries and OOS evaluation, and are excluded by every transform.
    """
    available = set(_parquet_columns(ledger_path))
    requested = [
        "__ts__",
        "__symbol__",
        "side_name",
        "archetype_policy_key",
        "selected_for_monitor",
        "hit_probability",
        "ev_after_1pct",
        "clean_exec",
        "full_path_bad_mae_1r",
        "timeout",
        "dirty_positive",
    ]
    ledger = pd.read_parquet(
        ledger_path, columns=[name for name in requested if name in available]
    )
    if ledger.empty:
        return states, {"hydrated_rows": 0, "reason": "empty_ledger"}
    ledger["__ts__"] = pd.to_datetime(ledger["__ts__"], utc=True, errors="coerce")
    ledger = ledger.loc[
        ledger["__ts__"].notna()
        & ledger["__ts__"].between(states["__ts__"].min(), states["__ts__"].max())
    ].copy()
    if "ev_after_1pct" in ledger:
        ledger = ledger.loc[
            pd.to_numeric(ledger["ev_after_1pct"], errors="coerce").notna()
        ].copy()
    if ledger.empty:
        return states, {"hydrated_rows": 0, "reason": "no_resolved_ledger_rows"}
    signature, signature_manifest = build_global_residual_signature(
        ledger, StateVectorConfig()
    )
    if signature.empty:
        return states, {"hydrated_rows": 0, "reason": "empty_signature"}
    signature["__ts__"] = pd.to_datetime(signature["__ts__"], utc=True, errors="coerce")
    target_columns = [
        name for name in signature.columns if name.startswith("target_signature_")
    ]
    output = states.merge(
        signature[["__ts__", *target_columns]],
        on="__ts__",
        how="left",
        validate="many_to_one",
        suffixes=("", "__ledger"),
    )
    fills = 0
    for name in target_columns:
        ledger_name = f"{name}__ledger"
        if ledger_name not in output:
            continue
        if name in states.columns:
            before = output[name].isna()
            output[name] = output[name].where(~before, output[ledger_name])
            fills += int((before & output[name].notna()).sum())
            output = output.drop(columns=ledger_name)
        else:
            output = output.rename(columns={ledger_name: name})
            fills += int(output[name].notna().sum())

    # The generic side targets used by reporting are the corresponding side
    # signatures. Partition preparation will still use the more specific
    # ``target_signature_arch__*`` fields.
    bases = (
        "signed_surprise",
        "positive_surprise",
        "negative_surprise",
        "mean_ev",
        "negative_ev",
        "payoff_asymmetry",
        "bad_mae_rate",
        "timeout_rate",
    )
    for side in ("long", "short"):
        side_mask = output["side_name"].astype(str).str.lower().eq(side)
        for base in bases:
            destination = f"target_{base}"
            source = f"target_signature_{side}_{base}"
            if destination not in output or source not in output:
                continue
            before = output[destination].isna() & side_mask
            output.loc[before, destination] = output.loc[before, source]
            fills += int(before.sum())
    return output, {
        "hydrated_rows": int(fills),
        "ledger": str(ledger_path.resolve()),
        "signature_target_columns": len(target_columns),
        "signature_contract": signature_manifest["definition"],
    }


def _materialize_states(
    args: argparse.Namespace, output: Path
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    state_path = output / "side_timestamp_market_states.parquet"
    manifest_path = output / "side_timestamp_market_states.manifest.json"
    if state_path.exists() and manifest_path.exists() and not args.force:
        manifest = json.loads(manifest_path.read_text())
        return pd.read_parquet(state_path), list(manifest["state_features"]), manifest

    # Research iterations that only add fixed observable state combinations do
    # not need to recompute the expensive cross-sectional aggregate table. The
    # source table already contains point-in-time full-universe coordinates;
    # this branch merely derives deterministic phase features from them.
    if args.existing_states is not None:
        source_path = Path(args.existing_states)
        if not source_path.exists():
            raise FileNotFoundError(f"Existing state source not found: {source_path}")
        source_manifest_path = source_path.with_name(
            "side_timestamp_market_states.manifest.json"
        )
        source_manifest = (
            json.loads(source_manifest_path.read_text(encoding="utf-8"))
            if source_manifest_path.exists()
            else {}
        )
        frame = pd.read_parquet(source_path)
        frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True, errors="coerce")
        frame, target_hydration = _hydrate_missing_state_targets(
            frame, Path(args.ledger)
        )
        if bool(args.phase_features):
            frame, phase_manifest = add_causal_phase_state_features(frame)
        else:
            phase_columns = [
                name for name in frame.columns if name.startswith("state_phase__")
            ]
            frame = frame.drop(columns=phase_columns, errors="ignore")
            phase_manifest = {
                "schema": "causal_phase_state_features_v1",
                "enabled": False,
                "features": [],
                "source_contract": "disabled_for_matched_ablation_control",
            }
        frame = _downcast(
            frame.sort_values(["__ts__", "side_name"], kind="stable").reset_index(
                drop=True
            )
        )
        feature_names = [
            name
            for name in frame.select_dtypes(include=[np.number, "bool"]).columns
            if not name.startswith(("target_", "placebo_target_"))
        ]
        frame.to_parquet(state_path, index=False, compression="zstd")
        manifest = {
            "schema": "global_side_timestamp_market_state_materialization_v1",
            "reused_state_source": str(source_path.resolve()),
            "rows": len(frame),
            "start": frame["__ts__"].min(),
            "end": frame["__ts__"].max(),
            "market_features": source_manifest.get("market_features", []),
            "asset_features": source_manifest.get("asset_features", []),
            "state_features": feature_names,
            "archetype_partitions": source_manifest.get("archetype_partitions", []),
            "causal_phase_features": phase_manifest,
            "target_hydration": target_hydration,
            "population_contract": source_manifest.get(
                "population_contract",
                "Reused point-in-time side timestamp state table.",
            ),
            "leakage_contract": (
                "The reused source contains point-in-time state coordinates. New phase fields are "
                "deterministic current/past combinations only; realized target_* columns remain excluded."
            ),
        }
        _write_json(manifest_path, manifest)
        return frame, feature_names, manifest

    candidate_root = Path(args.candidate_root)
    feature_root = Path(args.feature_root)
    state_cfg = StateVectorConfig(
        max_asset_features=int(args.max_asset_features),
        max_market_features=int(args.max_market_features),
    )
    raw_preselection: dict[str, Any] | None = None
    if bool(args.broad_binned_mi_preselection):
        search_boundary = pd.Timestamp(args.search_validation_start, tz="UTC")
        train_cutoff = search_boundary - pd.Timedelta(hours=float(args.purge_hours))
        market_features, asset_features, raw_preselection = (
            _broad_binned_mi_preselection(
                candidate_root=candidate_root,
                feature_root=feature_root,
                ledger_path=Path(args.ledger),
                output=output,
                train_cutoff=train_cutoff,
                sample_rows=int(args.raw_mi_sample_rows),
                top_per_archetype=int(args.raw_mi_top_per_archetype),
                max_union_features=int(args.raw_mi_max_union_features),
            )
        )
        sample = _read_feature_sample(candidate_root, feature_root)
    else:
        sample = _read_feature_sample(candidate_root, feature_root)
        market_features, asset_features = select_state_features(sample, state_cfg)
    selected_raw_features = list(dict.fromkeys(market_features + asset_features))
    ledger = pd.read_parquet(args.ledger, columns=list(LEDGER_OVERLAY_COLUMNS))
    ledger = ledger.drop_duplicates("row_id", keep="last").set_index("row_id")
    monthly: list[pd.DataFrame] = []
    manifests: list[dict[str, Any]] = []
    all_state_features: set[str] = set()
    archetype_partitions: dict[str, dict[str, str]] = {}
    for path in sorted(candidate_root.glob("candidates_*.parquet")):
        path_schema = set(_parquet_columns(path))
        candidate_columns = list(
            dict.fromkeys(
                [name for name in KEY_COLUMNS if name in path_schema]
                + [name for name in selected_raw_features if name in path_schema]
            )
        )
        candidates = pd.read_parquet(path, columns=candidate_columns)
        candidates = _refresh_store_features(
            candidates, feature_root, market_features + asset_features
        )
        overlay = ledger.reindex(candidates["row_id"].to_numpy()).reset_index(drop=True)
        for name in LEDGER_OVERLAY_COLUMNS:
            if name == "row_id":
                continue
            candidates[name] = overlay[name].to_numpy()
        candidates = _downcast(candidates)
        for side, archetype in (
            candidates[["side_name", "archetype_policy_key"]]
            .dropna()
            .astype(str)
            .drop_duplicates()
            .itertuples(index=False, name=None)
        ):
            token = archetype_state_token(side, archetype)
            archetype_partitions[token] = {
                "token": token,
                "side_name": str(side).lower(),
                "archetype_policy_key": str(archetype),
            }
        states, state_features, manifest = build_side_timestamp_states(
            candidates, market_features, asset_features, state_cfg
        )
        if bool(args.full_universe_aggregation):
            month = path.stem.removeprefix("candidates_")
            start = pd.Timestamp(pd.Period(month).start_time, tz="UTC")
            end = pd.Timestamp((pd.Period(month) + 1).start_time, tz="UTC")
            full_universe = _full_universe_month_aggregate(
                feature_root,
                start,
                end,
                market_features + asset_features,
                eligible_symbols=set(candidates["__symbol__"].astype(str)),
            )
            states = _merge_full_universe(
                states, full_universe, market_features + asset_features
            )
            state_features = [
                name
                for name in states.select_dtypes(include=[np.number, "bool"]).columns
                if not name.startswith(("target_", "placebo_target_"))
            ]
        states["source_month"] = path.stem.removeprefix("candidates_")
        monthly.append(states)
        manifests.append({"source": str(path), **manifest})
        all_state_features.update(state_features)
        print(
            json.dumps(
                {
                    "event": "state_month_complete",
                    "month": states["source_month"].iloc[0],
                    "candidate_rows": len(candidates),
                    "state_rows": len(states),
                }
            ),
            flush=True,
        )
        del candidates, overlay, states
        gc.collect()

    # July is the genuine frozen post-fit period and is already present in the
    # canonical ledger, but not in the retrospective candidate shard directory.
    july_columns = list(
        dict.fromkeys(
            [
                "row_id",
                "__ts__",
                "__symbol__",
                "side_name",
                "archetype_policy_key",
                "score_meta_base_soft_label",
                "selected_for_monitor",
                "hit_probability",
                "ev_after_1pct",
                "clean_exec",
                "full_path_bad_mae_1r",
                "timeout",
            ]
        )
    )
    july_source = Path(args.july_source)
    july_available = set(_parquet_columns(july_source))
    july = pd.read_parquet(
        july_source,
        columns=[name for name in july_columns if name in july_available],
    )
    july["__ts__"] = pd.to_datetime(july["__ts__"], utc=True, errors="coerce")
    july = july.loc[
        july["__ts__"].ge(pd.Timestamp("2026-07-01", tz="UTC"))
        & july["__ts__"].lt(pd.Timestamp("2026-07-11", tz="UTC"))
    ].copy()
    if not july.empty:
        for side, archetype in (
            july[["side_name", "archetype_policy_key"]]
            .dropna()
            .astype(str)
            .drop_duplicates()
            .itertuples(index=False, name=None)
        ):
            token = archetype_state_token(side, archetype)
            archetype_partitions[token] = {
                "token": token,
                "side_name": str(side).lower(),
                "archetype_policy_key": str(archetype),
            }
        july = _refresh_store_features(
            july, feature_root, market_features + asset_features
        )
        july_states, july_features, july_manifest = build_side_timestamp_states(
            _downcast(july), market_features, asset_features, state_cfg
        )
        if bool(args.full_universe_aggregation):
            july_universe = _full_universe_month_aggregate(
                feature_root,
                pd.Timestamp("2026-07-01", tz="UTC"),
                pd.Timestamp("2026-07-11", tz="UTC"),
                market_features + asset_features,
                eligible_symbols=set(july["__symbol__"].astype(str)),
            )
            july_states = _merge_full_universe(
                july_states, july_universe, market_features + asset_features
            )
            july_features = [
                name
                for name in july_states.select_dtypes(
                    include=[np.number, "bool"]
                ).columns
                if not name.startswith(("target_", "placebo_target_"))
            ]
        july_states["source_month"] = "2026-07"
        monthly.append(july_states)
        manifests.append({"source": str(july_source), **july_manifest})
        all_state_features.update(july_features)
        print(
            json.dumps(
                {
                    "event": "state_month_complete",
                    "month": "2026-07",
                    "candidate_rows": len(july),
                    "state_rows": len(july_states),
                }
            ),
            flush=True,
        )
    frame = pd.concat(monthly, ignore_index=True, sort=False)
    frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True, errors="coerce")
    event_membership_path = DISCOVERY_ROOT / "unreliability_event_membership.parquet"
    if event_membership_path.exists():
        membership = pd.read_parquet(
            event_membership_path, columns=["day", "side_name", "event_id"]
        )
        membership["day"] = pd.to_datetime(membership["day"], utc=True, errors="coerce")
        event_side_day = (
            membership.groupby(["day", "side_name"], observed=True)["event_id"]
            .agg(lambda values: "|".join(sorted(set(map(str, values)))))
            .rename("diagnostic_event_ids")
            .reset_index()
        )
        frame["day"] = frame["__ts__"].dt.floor("D")
        frame = frame.merge(event_side_day, on=["day", "side_name"], how="left")
        frame = frame.drop(columns="day")
    frame = _downcast(
        frame.sort_values(["__ts__", "side_name"], kind="stable").reset_index(drop=True)
    )
    feature_names = [
        name
        for name in sorted(all_state_features)
        if name in frame.columns and not name.startswith("target_")
    ]
    frame.to_parquet(state_path, index=False, compression="zstd")
    manifest = {
        "schema": "global_side_timestamp_market_state_materialization_v1",
        "candidate_root": str(candidate_root.resolve()),
        "feature_root": str(feature_root.resolve()),
        "ledger": str(Path(args.ledger).resolve()),
        "july_source": str(Path(args.july_source).resolve()),
        "rows": len(frame),
        "start": frame["__ts__"].min(),
        "end": frame["__ts__"].max(),
        "market_features": market_features,
        "asset_features": asset_features,
        "raw_feature_preselection": raw_preselection,
        "state_features": feature_names,
        "population_contract": (
            "full_universe__ aggregates use every point-in-time symbol appearing in the month's "
            "spread-filtered candidate universe; universe__ aggregates use the base top30 handoff; "
            "selected__ aggregates use the exact frozen policy selection."
        ),
        "monthly": manifests,
        "archetype_partitions": [
            archetype_partitions[token] for token in sorted(archetype_partitions)
        ],
        "leakage_contract": (
            "Only pre-entry aggregated coordinates are listed in state_features. Realized target_* "
            "columns are retained solely for train auxiliaries and evaluation."
        ),
    }
    _write_json(manifest_path, manifest)
    return frame, feature_names, manifest


def _fit_archetype_model(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    features: list[str],
    ae_config: ResidualAEConfig,
    gmm_config: GMMGridConfig,
    selected_features: list[str] | None = None,
    feature_relevance: pd.DataFrame | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    train = train.copy(deep=False)
    adverse = (
        pd.to_numeric(train["target_mean_ev"], errors="coerce").lt(0.0)
        | pd.to_numeric(train["target_signed_surprise"], errors="coerce").lt(0.0)
        | pd.to_numeric(train["target_bad_mae_rate"], errors="coerce").ge(0.75)
    )
    train_days = pd.to_datetime(train["__ts__"], utc=True).dt.floor("D")
    day_active = adverse.groupby(train_days, observed=True).any().sort_index()
    train_event_ids: dict[pd.Timestamp, str] = {}
    event_index = -1
    previous_day: pd.Timestamp | None = None
    for day, active in day_active.items():
        if not bool(active):
            previous_day = None
            continue
        if previous_day is None or (pd.Timestamp(day) - previous_day).days > 1:
            event_index += 1
        train_event_ids[pd.Timestamp(day)] = f"TRAIN-{event_index:04d}"
        previous_day = pd.Timestamp(day)
    train["diagnostic_event_ids"] = train_days.map(train_event_ids).fillna("")
    if selected_features is None:
        selected_features, feature_relevance = select_partition_state_features(
            train,
            features,
            max_features=int(ae_config.max_input_features),
        )
    if not selected_features:
        raise ValueError("No train-selected state features for archetype partition")
    ae = ResidualAwareAutoencoder(ae_config).fit(train, selected_features)
    train_latent = ae.transform(train)
    valid_latent = ae.transform(valid)
    gmm = GlobalGMMStateModel(gmm_config).fit(train_latent, train, train["__ts__"])
    train_state = gmm.transform(train_latent)
    valid_state = gmm.transform(valid_latent)
    valid_state = add_temporal_state_features(valid_state, valid["__ts__"])
    combined = pd.concat(
        [
            valid.reset_index(drop=True),
            valid_latent.reset_index(drop=True),
            valid_state.reset_index(drop=True),
        ],
        axis=1,
    )
    metrics = state_recognition_metrics(
        combined,
        "global_state_expected_negative_ev",
        "global_state_expected_positive_surprise",
    )
    train_post = train_state.filter(regex=r"^global_state_posterior_[0-9]+$").to_numpy(
        dtype=float
    )
    valid_post = valid_state.filter(regex=r"^global_state_posterior_[0-9]+$").to_numpy(
        dtype=float
    )
    train_occ = np.clip(train_post.mean(axis=0), 1e-8, 1.0)
    valid_occ = np.clip(valid_post.mean(axis=0), 1e-8, 1.0)
    metrics.update(
        {
            "reconstruction_error": float(
                valid_latent["global_state_reconstruction_error"].mean()
            ),
            "occupancy_js_distance": float(jensenshannon(train_occ, valid_occ)),
            "minimum_oos_component_occupancy": float(valid_occ.min()),
        }
    )
    try:
        chosen = gmm.model
        assert chosen is not None
        challenger_cfg = replace(
            gmm_config,
            components=(int(chosen.n_components),),
            covariance_types=(str(chosen.covariance_type),),
            reg_covars=(float(chosen.reg_covar),),
            random_state=int(gmm_config.random_state) + 17,
        )
        challenger = GlobalGMMStateModel(challenger_cfg).fit(
            train_latent, train, train["__ts__"]
        )
        latent_matrix = train_latent.filter(regex=r"^global_state_latent_").to_numpy(
            dtype=float
        )
        assert challenger.model is not None
        metrics["seed_centroid_matched_ari"] = centroid_matched_ari(
            chosen, challenger.model, latent_matrix
        )
    except (AssertionError, ValueError, np.linalg.LinAlgError):
        metrics["seed_centroid_matched_ari"] = np.nan
    risk_base = float(
        pd.to_numeric(valid["target_negative_ev"], errors="coerce").gt(0).mean()
    )
    opportunity_base = float(
        pd.to_numeric(valid["target_positive_surprise"], errors="coerce").gt(0).mean()
    )
    metrics["selection_objective"] = float(
        0.35 * (metrics["negative_ev_auprc"] - risk_base)
        + 0.25 * (metrics["positive_surprise_auprc"] - opportunity_base)
        + 10.0 * metrics["incremental_ev_top_opportunity_state"]
        - 0.05 * metrics["occupancy_js_distance"]
        - 0.02 * metrics["reconstruction_error"]
    )
    bundle = {
        "ae": ae,
        "gmm": gmm,
        "train_state": train_state,
        "valid_output": combined,
        "selected_state_features": list(selected_features),
        "feature_relevance": (
            feature_relevance.copy(deep=False)
            if feature_relevance is not None
            else pd.DataFrame()
        ),
    }
    manifest = {
        "ae": ae.manifest(),
        "gmm": gmm.manifest(),
        "metrics": metrics,
        "train_rows": len(train),
        "valid_rows": len(valid),
        "selected_state_feature_count": len(selected_features),
        "selected_phase_state_feature_count": int(
            sum(name.startswith("state_phase__") for name in selected_features)
        ),
        "state_feature_selection_contract": (
            "Partition-local relevance is fitted on train rows only using adverse/favorable "
            "signature association, nonlinear tail lift, and stable binned mutual information; "
            "target columns are not inputs."
        ),
    }
    return bundle, manifest


def _search_configs(
    states: pd.DataFrame,
    features: list[str],
    partitions: list[dict[str, str]],
    args: argparse.Namespace,
    output: Path,
) -> tuple[dict[str, ResidualAEConfig], dict[str, GMMGridConfig], pd.DataFrame]:
    search_rows: list[dict[str, Any]] = []
    relevance_parts: list[pd.DataFrame] = []
    best_by_partition: dict[str, ResidualAEConfig] = {}
    best_gmm_by_partition: dict[str, GMMGridConfig] = {}
    validation_start = pd.Timestamp(args.search_validation_start, tz="UTC")
    validation_end = pd.Timestamp(args.search_validation_end, tz="UTC")
    train_cutoff = validation_start - pd.Timedelta(hours=float(args.purge_hours))
    dims = tuple(int(value) for value in args.latent_dims.split(","))
    lambdas = tuple(float(value) for value in args.aux_lambdas.split(","))
    full_gmm_cfg = GMMGridConfig(
        components=tuple(int(value) for value in args.gmm_components.split(",")),
        covariance_types=tuple(args.gmm_covariance.split(",")),
        reg_covars=tuple(float(value) for value in args.gmm_reg_covars.split(",")),
        n_init=int(args.gmm_n_init),
    )
    probe_components = (
        tuple(value for value in (4, 8, 12) if value in full_gmm_cfg.components)
        or full_gmm_cfg.components[: min(3, len(full_gmm_cfg.components))]
    )
    probe_gmm_cfg = replace(
        full_gmm_cfg,
        components=probe_components,
        covariance_types=("diag",),
        reg_covars=(1e-3,),
        n_init=1,
    )
    for partition in partitions:
        token = partition["token"]
        side = partition["side_name"]
        archetype = partition["archetype_policy_key"]
        local = prepare_archetype_state_partition(
            states,
            side=side,
            archetype=archetype,
        )
        train = local[local["__ts__"].lt(train_cutoff)].reset_index(drop=True)
        valid = local[
            local["__ts__"].ge(validation_start) & local["__ts__"].lt(validation_end)
        ].reset_index(drop=True)
        if (
            pd.to_numeric(train["target_mean_ev"], errors="coerce").notna().sum() < 50
            or pd.to_numeric(valid["target_mean_ev"], errors="coerce").notna().sum()
            < 10
        ):
            print(
                json.dumps(
                    {
                        "event": "archetype_search_skipped_low_support",
                        "state_partition_token": token,
                        "side_name": side,
                        "archetype_policy_key": archetype,
                    }
                ),
                flush=True,
            )
            continue
        selected_features, feature_relevance = select_partition_state_features(
            train,
            features,
            max_features=int(ResidualAEConfig().max_input_features),
        )
        if not selected_features:
            print(
                json.dumps(
                    {
                        "event": "archetype_search_skipped_no_train_features",
                        "state_partition_token": token,
                        "side_name": side,
                        "archetype_policy_key": archetype,
                    }
                ),
                flush=True,
            )
            continue
        if not feature_relevance.empty:
            report = feature_relevance.copy()
            report["state_partition_token"] = token
            report["side_name"] = side
            report["archetype_policy_key"] = archetype
            report["fit_scope"] = "search_train"
            relevance_parts.append(report)
        for latent_dim in dims:
            for aux_lambda in lambdas:
                ae_cfg = ResidualAEConfig(
                    latent_dim=latent_dim,
                    hidden_dim=max(32, latent_dim * 4),
                    lambda_surprise=aux_lambda,
                    lambda_ev=aux_lambda,
                    lambda_asymmetry=aux_lambda * 0.5,
                    epochs=int(args.ae_epochs),
                    batch_size=int(args.ae_batch_size),
                    random_state=20260711 + latent_dim * 31 + int(aux_lambda * 1000),
                )
                bundle, manifest = _fit_archetype_model(
                    train,
                    valid,
                    features,
                    ae_cfg,
                    probe_gmm_cfg,
                    selected_features,
                    feature_relevance,
                )
                row = {
                    "search_stage": "probe",
                    "state_partition_token": token,
                    "side_name": side,
                    "archetype_policy_key": archetype,
                    "latent_dim": latent_dim,
                    "aux_lambda": aux_lambda,
                    **manifest["metrics"],
                    "gmm_components": manifest["gmm"]["selected"]["components"],
                    "gmm_covariance": manifest["gmm"]["selected"]["covariance_type"],
                    "gmm_reg_covar": manifest["gmm"]["selected"]["reg_covar"],
                }
                search_rows.append(row)
                print(
                    json.dumps({"event": "ae_search_complete", **_safe(row)}),
                    flush=True,
                )
                del bundle
                gc.collect()
        partition_rows = pd.DataFrame(
            [
                row
                for row in search_rows
                if row["state_partition_token"] == token
                and row["search_stage"] == "probe"
            ]
        )
        finalists = partition_rows.sort_values(
            "selection_objective", ascending=False, kind="stable"
        ).head(int(args.search_finalists))
        for finalist in finalists.itertuples(index=False):
            latent_dim = int(finalist.latent_dim)
            aux_lambda = float(finalist.aux_lambda)
            ae_cfg = ResidualAEConfig(
                latent_dim=latent_dim,
                hidden_dim=max(32, latent_dim * 4),
                lambda_surprise=aux_lambda,
                lambda_ev=aux_lambda,
                lambda_asymmetry=aux_lambda * 0.5,
                epochs=int(args.ae_epochs),
                batch_size=int(args.ae_batch_size),
                random_state=20260711 + latent_dim * 31 + int(aux_lambda * 1000),
            )
            bundle, manifest = _fit_archetype_model(
                train,
                valid,
                features,
                ae_cfg,
                full_gmm_cfg,
                selected_features,
                feature_relevance,
            )
            row = {
                "search_stage": "full_grid_finalist",
                "state_partition_token": token,
                "side_name": side,
                "archetype_policy_key": archetype,
                "latent_dim": latent_dim,
                "aux_lambda": aux_lambda,
                **manifest["metrics"],
                "gmm_components": manifest["gmm"]["selected"]["components"],
                "gmm_covariance": manifest["gmm"]["selected"]["covariance_type"],
                "gmm_reg_covar": manifest["gmm"]["selected"]["reg_covar"],
            }
            search_rows.append(row)
            print(
                json.dumps({"event": "ae_full_grid_finalist_complete", **_safe(row)}),
                flush=True,
            )
            del bundle
            gc.collect()
        full_rows = pd.DataFrame(
            [
                row
                for row in search_rows
                if row["state_partition_token"] == token
                and row["search_stage"] == "full_grid_finalist"
            ]
        )
        best = full_rows.sort_values(
            "selection_objective", ascending=False, kind="stable"
        ).iloc[0]
        best_by_partition[token] = ResidualAEConfig(
            latent_dim=int(best["latent_dim"]),
            hidden_dim=max(32, int(best["latent_dim"]) * 4),
            lambda_surprise=float(best["aux_lambda"]),
            lambda_ev=float(best["aux_lambda"]),
            lambda_asymmetry=float(best["aux_lambda"]) * 0.5,
            epochs=int(args.ae_epochs),
            batch_size=int(args.ae_batch_size),
            random_state=20260711
            + int(best["latent_dim"]) * 31
            + int(float(best["aux_lambda"]) * 1000),
        )
        best_gmm_by_partition[token] = replace(
            full_gmm_cfg,
            components=(int(best["gmm_components"]),),
            covariance_types=(str(best["gmm_covariance"]),),
            reg_covars=(float(best["gmm_reg_covar"]),),
        )
    search = pd.DataFrame(search_rows)
    search.to_csv(output / "ae_gmm_search.csv", index=False)
    if relevance_parts:
        pd.concat(relevance_parts, ignore_index=True).to_csv(
            output / "partition_train_feature_relevance.csv", index=False
        )
    _write_json(
        output / "selected_ae_configs.json",
        {token: asdict(config) for token, config in best_by_partition.items()},
    )
    _write_json(
        output / "selected_gmm_configs.json",
        {token: asdict(config) for token, config in best_gmm_by_partition.items()},
    )
    _write_json(
        output / "archetype_partition_manifest.json",
        {
            "schema": "residual_latent_archetype_partitions_v1",
            "partitions": partitions,
            "fit_partition": "archetype_policy_key",
            "fit_contract": (
                "Each archetype owns an independent scaler, AE, GMM state model, and "
                "train-only enrichment map. Side only selects the matching observable rows."
            ),
        },
    )
    return best_by_partition, best_gmm_by_partition, search


def _rolling_origin(
    states: pd.DataFrame,
    features: list[str],
    configs: dict[str, ResidualAEConfig],
    gmm_configs: dict[str, GMMGridConfig],
    partitions: list[dict[str, str]],
    args: argparse.Namespace,
    output: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions: list[pd.DataFrame] = []
    fold_rows: list[dict[str, Any]] = []
    months = [value.strip() for value in args.eval_months.split(",") if value.strip()]
    state_dir = output / "states"
    state_dir.mkdir(exist_ok=True)
    for month in months:
        start = pd.Timestamp(pd.Period(month).start_time, tz="UTC")
        end = pd.Timestamp((pd.Period(month) + 1).start_time, tz="UTC")
        train_cutoff = start - pd.Timedelta(hours=float(args.purge_hours))
        for partition in partitions:
            token = partition["token"]
            if token not in configs:
                continue
            if token not in gmm_configs:
                continue
            side = partition["side_name"]
            archetype = partition["archetype_policy_key"]
            local = prepare_archetype_state_partition(
                states,
                side=side,
                archetype=archetype,
            )
            train = local[local["__ts__"].lt(train_cutoff)].reset_index(drop=True)
            valid = local[
                local["__ts__"].ge(start) & local["__ts__"].lt(end)
            ].reset_index(drop=True)
            if len(train) < 500 or len(valid) < 24:
                continue
            if (
                pd.to_numeric(train["target_mean_ev"], errors="coerce").notna().sum()
                < 50
                or pd.to_numeric(valid["target_mean_ev"], errors="coerce").notna().sum()
                < 5
            ):
                continue
            selected_features, feature_relevance = select_partition_state_features(
                train,
                features,
                max_features=int(configs[token].max_input_features),
            )
            if not selected_features:
                continue
            config = replace(
                configs[token],
                random_state=configs[token].random_state + int(start.month) * 101,
            )
            bundle, manifest = _fit_archetype_model(
                train,
                valid,
                features,
                config,
                gmm_configs[token],
                selected_features,
                feature_relevance,
            )
            generated = bundle["valid_output"].copy()
            generated["oos_month"] = month
            generated["fit_through"] = train_cutoff - pd.Timedelta(nanoseconds=1)
            generated["archetype_policy_key"] = archetype
            generated["state_partition_token"] = token
            predictions.append(generated)
            grid_path = output / f"gmm_grid_{token}_{month}.csv"
            bundle["gmm"].grid.to_csv(grid_path, index=False)
            joblib.dump(
                {
                    "ae": bundle["ae"],
                    "gmm": bundle["gmm"],
                    "features": bundle["selected_state_features"],
                    "fit_through": train_cutoff,
                    "side_name": side,
                    "archetype_policy_key": archetype,
                    "state_partition_token": token,
                },
                state_dir / f"global_residual_state_{token}_{month}.joblib",
            )
            if not bundle["feature_relevance"].empty:
                relevance = bundle["feature_relevance"].copy()
                relevance["oos_month"] = month
                relevance["fit_through"] = start - pd.Timedelta(nanoseconds=1)
                relevance["state_partition_token"] = token
                relevance["side_name"] = side
                relevance["archetype_policy_key"] = archetype
                relevance.to_csv(
                    output / f"feature_relevance_{token}_{month}.csv", index=False
                )
            fold_rows.append(
                {
                    "oos_month": month,
                    "state_partition_token": token,
                    "side_name": side,
                    "archetype_policy_key": archetype,
                    "fit_through": train_cutoff - pd.Timedelta(nanoseconds=1),
                    **manifest["metrics"],
                    "train_rows": len(train),
                    "valid_rows": len(valid),
                    "gmm_components": manifest["gmm"]["selected"]["components"],
                    "gmm_covariance": manifest["gmm"]["selected"]["covariance_type"],
                }
            )
            print(
                json.dumps({"event": "rolling_fold_complete", **_safe(fold_rows[-1])}),
                flush=True,
            )
            del bundle, generated, train, valid
            gc.collect()
    combined = (
        pd.concat(predictions, ignore_index=True, sort=False)
        if predictions
        else pd.DataFrame()
    )
    folds = pd.DataFrame(fold_rows)
    combined.to_parquet(
        output / "rolling_origin_state_predictions.parquet",
        index=False,
        compression="zstd",
    )
    folds.to_csv(output / "rolling_origin_metrics.csv", index=False)
    return combined, folds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("build", "fit", "rolling", "all"), default="all"
    )
    parser.add_argument("--candidate-root", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument(
        "--july-source",
        type=Path,
        default=DEFAULT_JULY_SOURCE,
        help="Complete frozen July top30 prediction shard used for state inputs.",
    )
    parser.add_argument("--feature-root", type=Path, default=DEFAULT_FEATURE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--existing-states",
        type=Path,
        default=None,
        help=(
            "Reuse a point-in-time side_timestamp_market_states.parquet source and only "
            "materialize deterministic phase coordinates before fitting."
        ),
    )
    parser.add_argument(
        "--config-source",
        type=Path,
        default=None,
        help=(
            "Optional prior state-run directory supplying frozen selected AE/GMM configs for a "
            "clean feature/state ablation without re-running hyperparameter search."
        ),
    )
    parser.add_argument(
        "--phase-features",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include deterministic liquidation lifecycle phase coordinates in the state input.",
    )
    parser.add_argument(
        "--broad-binned-mi-preselection",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Screen the full causal base/meta/context universe independently per archetype "
            "with stable quantile-binned MI before state aggregation."
        ),
    )
    parser.add_argument("--raw-mi-sample-rows", type=int, default=45_000)
    parser.add_argument("--raw-mi-top-per-archetype", type=int, default=32)
    parser.add_argument("--raw-mi-max-union-features", type=int, default=192)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-asset-features", type=int, default=36)
    parser.add_argument("--max-market-features", type=int, default=48)
    parser.add_argument(
        "--full-universe-aggregation",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--latent-dims", default="6,8,12,16")
    parser.add_argument("--aux-lambdas", default="0,0.02,0.05,0.1,0.2")
    parser.add_argument("--gmm-components", default="4,6,8,10,12,16")
    parser.add_argument("--gmm-covariance", default="diag,full")
    parser.add_argument("--gmm-reg-covars", default="0.0001,0.001")
    parser.add_argument("--gmm-n-init", type=int, default=3)
    parser.add_argument("--ae-epochs", type=int, default=120)
    parser.add_argument("--ae-batch-size", type=int, default=512)
    parser.add_argument("--search-finalists", type=int, default=4)
    parser.add_argument("--search-validation-start", default="2026-03-01")
    parser.add_argument("--search-validation-end", default="2026-04-01")
    parser.add_argument("--eval-months", default="2026-04,2026-05,2026-06,2026-07")
    parser.add_argument(
        "--purge-hours",
        type=float,
        default=12.0,
        help="Exclude train rows this many hours before each validation/OOS boundary.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    states, features, state_manifest = _materialize_states(args, output)
    if args.stage == "build":
        return
    partitions = list(state_manifest.get("archetype_partitions", []))
    if not partitions:
        raise ValueError(
            "No archetype partitions were materialized; rerun with --force"
        )
    if args.stage == "rolling":
        config_path = output / "selected_ae_configs.json"
        config_source = (
            Path(args.config_source) if args.config_source is not None else output
        )
        source_config_path = config_source / "selected_ae_configs.json"
        if not source_config_path.exists():
            raise FileNotFoundError(
                "Rolling-only mode requires frozen selected configs: "
                f"{source_config_path}"
            )
        raw_configs = json.loads(source_config_path.read_text(encoding="utf-8"))
        configs = {
            str(token): ResidualAEConfig(**values)
            for token, values in raw_configs.items()
        }
        search_path = config_source / "ae_gmm_search.csv"
        search = pd.read_csv(search_path) if search_path.exists() else pd.DataFrame()
        source_gmm_config_path = config_source / "selected_gmm_configs.json"
        gmm_config_path = output / "selected_gmm_configs.json"
        if source_gmm_config_path.exists():
            raw_gmm_configs = json.loads(
                source_gmm_config_path.read_text(encoding="utf-8")
            )
            gmm_configs = {
                str(token): GMMGridConfig(
                    **{
                        **values,
                        "components": tuple(values["components"]),
                        "covariance_types": tuple(values["covariance_types"]),
                        "reg_covars": tuple(values["reg_covars"]),
                    }
                )
                for token, values in raw_gmm_configs.items()
            }
        else:
            if search.empty:
                raise FileNotFoundError(
                    "Rolling-only mode requires selected_gmm_configs.json or ae_gmm_search.csv"
                )
            finalists = search[search["search_stage"].eq("full_grid_finalist")].copy()
            finalists = finalists.sort_values(
                "selection_objective", ascending=False, kind="stable"
            ).drop_duplicates("state_partition_token")
            gmm_configs = {}
            default_reg = float(args.gmm_reg_covars.split(",")[-1])
            for row in finalists.itertuples(index=False):
                token = str(row.state_partition_token)
                reg_covar = float(getattr(row, "gmm_reg_covar", default_reg))
                gmm_configs[token] = GMMGridConfig(
                    components=(int(row.gmm_components),),
                    covariance_types=(str(row.gmm_covariance),),
                    reg_covars=(reg_covar,),
                    n_init=int(args.gmm_n_init),
                )
            _write_json(
                gmm_config_path,
                {token: asdict(config) for token, config in gmm_configs.items()},
            )
        # Materialize the source configs inside the new artifact so its rolling
        # replay is self-contained and provenance is explicit.
        _write_json(
            config_path,
            {token: asdict(config) for token, config in configs.items()},
        )
        _write_json(
            gmm_config_path,
            {token: asdict(config) for token, config in gmm_configs.items()},
        )
    else:
        configs, gmm_configs, search = _search_configs(
            states, features, partitions, args, output
        )
    predictions, folds = _rolling_origin(
        states, features, configs, gmm_configs, partitions, args, output
    )
    _write_json(
        output / "manifest.json",
        {
            "schema": "archetype_residual_latent_state_rolling_origin_v1",
            "state_manifest": state_manifest,
            "selected_configs": {
                token: asdict(config) for token, config in configs.items()
            },
            "selected_gmm_configs": {
                token: asdict(config) for token, config in gmm_configs.items()
            },
            "search_rows": len(search),
            "prediction_rows": len(predictions),
            "folds": folds.to_dict(orient="records"),
            "july_contract": "July rows never enter AE/scaler/GMM/enrichment fitting for the July fold.",
            "purge_hours": float(args.purge_hours),
            "evidence_contract": (
                "April-June are rolling-origin OOS for the new latent layer over fixed-model "
                "retrospective base/meta backcasts; July is frozen post-fit OOS end to end."
            ),
            "existing_archetype_contract": (
                "Each existing inference archetype receives a separate scaler, AE, GMM "
                "state model, and train-only enrichment targets. Side is observable context, "
                "not the fitting partition; there is no cross-archetype fallback."
            ),
            "config_source": (
                str(Path(args.config_source).resolve())
                if args.config_source is not None
                else str(output.resolve())
            ),
        },
    )


if __name__ == "__main__":
    main()
