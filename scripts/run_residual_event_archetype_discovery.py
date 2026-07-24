#!/usr/bin/env python3
"""Walk-forward discovery of local residual-event AE/GMM states.

This is intentionally upstream of policy calibration.  It fits a frozen
side x archetype residual-state bundle on prior rows, emits only pre-entry
AE/GMM outputs for the next OOS month, then uses realized OOS outcomes solely
to assess whether large local surprise episodes are separated.

The default input is the frozen champion candidate stream (January 2025 through
the latest materialized month).  Extra candidate shards can be supplied later
for late-July rare-event research.  The manifest records coverage explicitly;
it never describes a state as transferable when the relevant local event has no
prior training support.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.config import CFG  # noqa: E402
from extreme_price_movements.data_store import read_symbol_features  # noqa: E402
from extreme_price_movements.residual_event_archetypes import (  # noqa: E402
    OUTCOME_COLUMNS,
    RESIDUAL_EVENT_PREFIX,
    RESIDUAL_EVENT_TARGET_PREFIX,
    ResidualEventArchetypeConfig,
    ResidualEventArchetypeState,
    causal_eight_day_hit_rate_overlay,
)

DEFAULT_ROOT = ROOT / (
    "data_perp/reports/train_meta_residual_archetype_enhancement_20260711_v1/"
    "champion_frozen_single_source_202501_20260710/candidate_shards"
)
DEFAULT_OUTPUT = (
    ROOT / "data_perp/reports/residual_event_archetype_discovery_20260712_v1"
)
DEFAULT_FEATURE_ROOT = ROOT / "data_perp/features/20260710_170000"
# Evaluate every available month after an initial early-2025 fit period.  The
# difficult dates remain explicit in the report, but quiet months are required
# to establish false-positive and transfer behavior around those rare states.
DEFAULT_MONTHS = (
    "2025-04,2025-05,2025-06,2025-07,2025-08,2025-09,2025-10,2025-11,"
    "2025-12,2026-01,2026-02,2026-03,2026-04,2026-05,2026-06,2026-07"
)
DEFAULT_REVISION_CUTOFFS = (
    "2025-04-01",
    "2025-10-01",
    "2026-01-01",
    "2026-04-01",
    "2026-07-01",
)

KEY_COLUMNS = ("row_id", "__ts__", "__symbol__", "side_name", "archetype_policy_key")
CANDIDATE_REQUIRED_COLUMNS = (
    "row_id",
    "__ts__",
    "__symbol__",
    "side_name",
    "side",
    "archetype_policy_key",
    "__archetype_policy_key__",
    "policy_archetype",
    "local_side_archetype",
    "archetype_label_family",
    "__archetype_label_family__",
    "source_archetype",
    "source_tag",
    "score",
    "score_meta_base_soft_label",
    "hit_probability",
    "selected_top30",
    "clean_exec",
    "ev_after_1pct",
    "exec_margin",
    "dirty_positive",
    "first_touch_bad_mae_1r",
    "full_path_bad_mae_1r",
    "timeout",
    "full_stop_loss",
    "stop_or_adverse",
    "__first_touch_target_soft__",
    "__first_touch_policy_soft__",
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(dict(payload)), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _parse_months(value: str) -> list[str]:
    months = [str(item).strip() for item in str(value).split(",") if str(item).strip()]
    for month in months:
        pd.Period(month, freq="M")
    return list(dict.fromkeys(months))


def _parse_int_csv(value: str, *, name: str) -> tuple[int, ...]:
    values = tuple(
        int(item.strip()) for item in str(value).split(",") if item.strip()
    )
    if not values or any(item < 2 for item in values):
        raise ValueError(f"{name} must contain integer values >= 2")
    return tuple(dict.fromkeys(values))


def _parse_float_csv(value: str, *, name: str) -> tuple[float, ...]:
    values = tuple(
        float(item.strip()) for item in str(value).split(",") if item.strip()
    )
    if not values or any(not np.isfinite(item) or item <= 0.0 for item in values):
        raise ValueError(f"{name} must contain finite positive values")
    return tuple(dict.fromkeys(values))


def _parse_choice_csv(
    value: str,
    *,
    name: str,
    allowed: set[str],
) -> tuple[str, ...]:
    values = tuple(item.strip() for item in str(value).split(",") if item.strip())
    invalid = sorted(set(values).difference(allowed))
    if not values or invalid:
        raise ValueError(f"{name} has invalid values {invalid}; expected one of {sorted(allowed)}")
    return tuple(dict.fromkeys(values))


def _load_candidate_columns(path: Path | None) -> list[str] | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        values = payload
    elif isinstance(payload, dict):
        values = payload.get("feature_names") or payload.get("selected_features")
    else:
        values = None
    if not isinstance(values, list) or not values:
        raise ValueError(f"No feature list found in candidate column contract: {path}")
    return list(dict.fromkeys(map(str, values)))


MODEL_STATE_CONTEXT_TOKENS: tuple[str, ...] = (
    "ood_",
    "_ood",
    "leaf_",
    "_leaf",
    "support_drift",
    "leaf_drift",
    "feature_drift",
    "prediction_drift",
    "support_count",
    "support_log",
    "posterior",
    "mahal",
    "reconstruction",
    "entropy",
    "score_margin",
    "margin_to_cutoff",
    "base_score_z",
    "base_rank_pct",
    "signal_zscore",
)

# A three-year candidate backcast can contain more than two million rows and
# 500+ observable columns.  Loading every column before local screening is a
# memory anti-pattern: it prevents the discovery stage from reaching the OOS
# folds it is meant to validate.  These groups retain the diverse, causal
# feature families needed for residual-state discovery while making the input
# projection deterministic and reproducible.  Final feature choice remains
# local side x archetype MI/LGBM screening inside the fitted state.
PROJECTED_SOURCE_FEATURE_FAMILIES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("base_anchor", ("score", "rank", "margin", "signal", "probability")),
    ("price_direction", ("ret", "trend", "ema", "momentum", "breakout", "pullback", "support", "boll", "atr", "price_")),
    ("volatility_path", ("vol", "range", "rv_", "variance", "tail", "wick", "path", "chop", "entropy", "spectral")),
    ("liquidity_microstructure", ("liquid", "spread", "depth", "book", "impact", "amihud", "dislocation", "volume")),
    ("open_interest", ("oi_", "open_interest", "flush", "leverage")),
    ("funding", ("fund", "crowd")),
    ("market_cross_asset", ("mkt", "market", "breadth", "cross", "dispersion", "corr", "pc1", "asset_minus")),
    ("aegmm", ("gmm", "aegmm", "dae", "reconstruction", "mahal", "posterior", "cluster")),
    ("model_context", ("ood", "leaf", "drift", "support_count", "support_log", "uncertainty")),
)


def _project_source_columns(
    root: Path,
    *,
    end: pd.Timestamp | None,
    max_features: int,
) -> list[str] | None:
    """Return a deterministic, family-balanced projection for wide backcasts.

    ``None`` preserves the legacy full-width read.  Otherwise the result is a
    feature projection only; `_load_shards` always adds the structural and
    outcome columns required for leakage-safe train/OOS assessment.
    """

    cap = int(max_features)
    if cap <= 0:
        return None
    paths = sorted(root.glob("candidates_*.parquet"))
    if end is not None:
        kept: list[Path] = []
        for path in paths:
            token = path.stem.removeprefix("candidates_")
            try:
                month_start = pd.Timestamp(pd.Period(token, freq="M").start_time, tz="UTC")
            except ValueError:
                month_start = pd.Timestamp.min.tz_localize("UTC")
            if month_start < end:
                kept.append(path)
        paths = kept
    if not paths:
        return []
    names = set(pq.ParquetFile(paths[-1]).schema_arrow.names)
    # Candidate shards are schema-compatible by contract; intersecting with
    # the earliest shard also prevents a late-only column from producing an
    # all-missing historical feature.
    names.intersection_update(pq.ParquetFile(paths[0]).schema_arrow.names)
    forbidden = set(OUTCOME_COLUMNS).union(
        {
            "row_id", "__ts__", "__symbol__", "side", "side_name",
            "archetype_policy_key", "policy_archetype", "local_side_archetype",
            "archetype_label_family", "source_archetype", "source_tag",
            "selected_top30", "selected_for_monitor", "threshold_basis_selected",
            "evidence_scope", "prediction_evidence",
        }
    )
    candidates = sorted(
        name for name in names
        if name not in forbidden
        and not name.startswith("expost__")
        and not name.startswith("resid_event_target_")
        and not name.endswith("_label")
    )
    selected: list[str] = []
    # Use a round-robin pass over families so no large prefix (for example
    # OI) crowds out microstructure, market, or AE/GMM context.
    buckets: list[list[str]] = []
    for _, tokens in PROJECTED_SOURCE_FEATURE_FAMILIES:
        bucket = [
            name for name in candidates
            if any(token in name.lower() for token in tokens)
        ]
        buckets.append(bucket)
    offsets = [0] * len(buckets)
    while len(selected) < cap:
        made_progress = False
        for idx, bucket in enumerate(buckets):
            while offsets[idx] < len(bucket) and bucket[offsets[idx]] in selected:
                offsets[idx] += 1
            if offsets[idx] < len(bucket):
                selected.append(bucket[offsets[idx]])
                offsets[idx] += 1
                made_progress = True
                if len(selected) >= cap:
                    break
        if not made_progress:
            break
    if len(selected) < cap:
        selected.extend(name for name in candidates if name not in selected)
    return selected[:cap]


def _expanded_model_context_columns(paths: Sequence[Path]) -> list[str]:
    """Project observable model-trust/state columns without loading 1k+ fields."""

    selected: list[str] = list(CANDIDATE_REQUIRED_COLUMNS)
    for path in paths:
        if not path.exists() or path.suffix != ".parquet":
            continue
        names = pq.read_schema(path).names
        selected.extend(
            name
            for name in names
            if any(token in name.lower() for token in MODEL_STATE_CONTEXT_TOKENS)
        )
    return list(dict.fromkeys(selected))


def _revision_cutoff_for_month(
    month_start: pd.Timestamp,
    *,
    schedule: str,
) -> pd.Timestamp:
    if str(schedule) == "monthly":
        return month_start
    cutoffs = [pd.Timestamp(value, tz="UTC") for value in DEFAULT_REVISION_CUTOFFS]
    eligible = [value for value in cutoffs if value <= month_start]
    if not eligible:
        raise ValueError(f"No frozen revision cutoff precedes {month_start}")
    return max(eligible)


def _downcast_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep the broad research basket compact without copying object blocks."""

    for name in frame.columns:
        series = frame[name]
        if pd.api.types.is_float_dtype(series):
            frame[name] = pd.to_numeric(series, errors="coerce", downcast="float")
        elif pd.api.types.is_integer_dtype(series):
            frame[name] = pd.to_numeric(series, errors="coerce", downcast="integer")
    for name in ("__symbol__", "side_name", "archetype_policy_key", "source_tag"):
        if name in frame.columns and frame[name].dtype == object:
            frame[name] = frame[name].astype("category")
    return frame


def _load_shards(
    root: Path,
    extra: Iterable[Path] = (),
    *,
    end: pd.Timestamp | None = None,
    requested_columns: Sequence[str] | None = None,
    max_train_rows_per_shard: int = 0,
    full_months: Sequence[str] = (),
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    paths = sorted(root.glob("candidates_*.parquet"))
    if end is not None:
        kept: list[Path] = []
        for path in paths:
            token = path.stem.removeprefix("candidates_")
            try:
                month_start = pd.Timestamp(
                    pd.Period(token, freq="M").start_time, tz="UTC"
                )
            except ValueError:
                month_start = pd.Timestamp.min.tz_localize("UTC")
            if month_start < end:
                kept.append(path)
        paths = kept
    paths += [path for path in extra if path.exists()]
    if not paths:
        raise FileNotFoundError(f"No candidate shards found under {root}")
    frames: list[pd.DataFrame] = []
    coverage: list[dict[str, Any]] = []
    full_month_tokens = {str(item).replace("-", "")[:6] for item in full_months}
    for path in paths:
        schema = pq.ParquetFile(path)
        names = schema.schema_arrow.names
        if requested_columns is None:
            read_columns = None
        else:
            read_columns = [
                name
                for name in dict.fromkeys(
                    [*CANDIDATE_REQUIRED_COLUMNS, *requested_columns]
                )
                if name in names
            ]
        part = pd.read_parquet(path, columns=read_columns)
        if "__ts__" not in part.columns:
            continue
        token = path.stem.removeprefix("candidates_").replace("-", "")[:6]
        source_rows = int(len(part))
        sampled = False
        cap = int(max_train_rows_per_shard)
        if cap > 0 and token not in full_month_tokens and len(part) > cap:
            # Deterministic beginning/middle/end sampling keeps both quiet and
            # rare late-period states without retaining an entire wide shard.
            thirds = (
                (0, len(part) // 3),
                (len(part) // 3, (2 * len(part)) // 3),
                ((2 * len(part)) // 3, len(part)),
            )
            per = max(1, cap // 3)
            indices = np.unique(
                np.concatenate(
                    [
                        np.linspace(start, stop - 1, min(per, stop - start), dtype=np.int64)
                        for start, stop in thirds
                        if stop > start
                    ]
                )
            )[:cap]
            part = part.iloc[indices].copy()
            sampled = True
        if "score" not in part.columns and "score_meta_base_soft_label" in part.columns:
            part["score"] = pd.to_numeric(
                part["score_meta_base_soft_label"], errors="coerce"
            ).astype(np.float32)
        elif "score" in part.columns and "score_meta_base_soft_label" in part.columns:
            missing_score = pd.to_numeric(part["score"], errors="coerce").isna()
            if bool(missing_score.any()):
                part.loc[missing_score, "score"] = pd.to_numeric(
                    part.loc[missing_score, "score_meta_base_soft_label"],
                    errors="coerce",
                ).to_numpy(dtype=np.float32)
        part["__ts__"] = pd.to_datetime(part["__ts__"], utc=True, errors="coerce")
        frames.append(_downcast_frame(part))
        coverage.append(
            {
                "path": str(path),
                "rows": int(len(part)),
                "source_rows": source_rows,
                "sampled_train_shard": sampled,
                "columns": int(len(names)),
                "start": str(part["__ts__"].min()),
                "end": str(part["__ts__"].max()),
            }
        )
    if not frames:
        raise ValueError("Candidate shards contained no timestamped rows")
    frame = pd.concat(frames, ignore_index=True, copy=False)
    frame = frame.loc[frame["__ts__"].notna()].sort_values(
        ["__ts__", "__symbol__", "side_name"], kind="stable"
    )
    dedupe_keys = [
        name
        for name in ("row_id", "__ts__", "__symbol__", "side_name")
        if name in frame.columns
    ]
    if dedupe_keys:
        frame = frame.drop_duplicates(dedupe_keys, keep="last")
    return frame.reset_index(drop=True), coverage


def _normalise_candidate_contract(
    frame: pd.DataFrame,
    *,
    score_col: str,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Normalize frozen candidate/prediction shards without touching outcomes.

    Candidate shards expose ``score`` while complete frozen prediction shards
    expose ``score_meta_base_soft_label``. Both are point-in-time meta scores;
    this adapter makes the source choice visible in the manifest and lets a
    late, resolved month be assessed without a second data format.
    """

    out = frame.copy(deep=False)
    structural_aliases = {
        "archetype_policy_key": (
            "__archetype_policy_key__",
            "policy_archetype",
            "local_side_archetype",
        ),
        "archetype_label_family": (
            "__archetype_label_family__",
            "source_archetype",
        ),
    }
    provenance: dict[str, str] = {}
    for target, candidates in structural_aliases.items():
        sources = [name for name in candidates if name in out.columns]
        if not sources:
            continue
        if target not in out.columns:
            out[target] = out[sources[0]]
            provenance[target] = str(sources[0])
            continue
        combined = out[target].astype(object)
        filled_from: list[str] = []
        for source in sources:
            missing = combined.isna() | combined.astype(str).str.strip().isin(
                ("", "missing", "nan", "none")
            )
            if not bool(missing.any()):
                break
            source_values = out[source].astype(object)
            usable = missing & source_values.notna()
            if bool(usable.any()):
                combined.loc[usable] = source_values.loc[usable]
                filled_from.append(str(source))
        out[target] = combined
        if filled_from:
            provenance[target] = "coalesce:" + ",".join(filled_from)
    aliases = (
        "score_meta_base_soft_label",
        "score_regime_current",
        "score_current_reference",
        "hit_probability",
        "base_score",
    )
    if str(score_col) not in out.columns:
        source = next(
            (
                name
                for name in aliases
                if name in out.columns
                and pd.to_numeric(out[name], errors="coerce").notna().any()
            ),
            None,
        )
        if source is None:
            raise KeyError(
                f"No usable score column for residual discovery: requested={score_col!r}"
            )
        out[str(score_col)] = pd.to_numeric(out[source], errors="coerce").astype(
            np.float32
        )
        provenance[str(score_col)] = str(source)
    if "selected_top30" not in out.columns:
        # Informational only: state admission is defined by train-fitted global
        # EV targets, not by this source-period marker.
        out["selected_top30"] = np.int8(1)
        provenance["selected_top30"] = "implicit_frozen_candidate_stream"
    return _downcast_frame(out), provenance


def _configured_supplemental_features(frame: pd.DataFrame, cap: int) -> list[str]:
    """Return a broad but explicit meta/state feature basket from config.py."""

    if int(cap) <= 0:
        return []
    sources = (
        "MODEL_REGIME_COMPOSITE_META_FEATURE_KEYS",
        "MODEL_REGIME_CONTEXT_META_FEATURE_KEYS",
        "MODEL_REGIME_XS_META_FEATURE_KEYS",
        "meta_shared_feature_keys",
        "CROSS_ASSET_META_FEATURE_KEYS",
        "ORDERBOOK_META_FEATURE_KEYS",
        "LGBM_PERP_FEATURE_KEYS",
        "RESIDUAL_META_FEATURE_KEYS",
    )
    names: list[str] = []
    for source in sources:
        values = CFG.get(source, [])
        if isinstance(values, (list, tuple)):
            names.extend(str(name) for name in values)
    return [
        name
        for name in dict.fromkeys(names)
        if name not in frame.columns and name not in OUTCOME_COLUMNS
    ][: int(cap)]


def _is_broadcast_market_feature(name: str) -> bool:
    lower = str(name).lower()
    if lower.startswith(
        (
            "mkt_",
            "market_",
            "pct_assets_",
            "cross_asset_",
            "crossasset_",
            "xs_mean__",
            "xs_median__",
            "xs_std__",
            "state_spectral_",
        )
    ):
        return True
    return lower in {
        "liquidation_onset_score",
        "liquidation_climax_score",
        "post_liquidation_rebound_score",
        "positive_funding_x_price_down",
        "positive_funding_x_oi_drop",
        "negative_funding_x_price_up",
        "negative_funding_x_oi_drop",
        "funding_crowding_x_vol_expansion",
        "funding_flip_x_oi_flush",
        "funding_mean_reversion_after_oi_flush",
        "breadth_chg_15m",
        "breadth_chg_1h",
        "breadth_accel_1h",
        "breadth_min_6h",
        "breadth_recovery_from_6h_min",
        "return_dispersion_1h",
        "return_dispersion_4h",
    }


def _append_feature_store_basket(
    frame: pd.DataFrame,
    *,
    feature_root: Path,
    requested: Sequence[str],
    batch_size: int,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Append broad pre-entry features in bounded float32 batches.

    The old helper allocated every requested column at once.  State discovery
    can use a wider basket, so this version keeps the transient allocation to
    ``rows x batch_size`` and downcasts each completed batch immediately.
    """

    names = [
        str(name)
        for name in requested
        if str(name) not in frame.columns
        or float(pd.to_numeric(frame[str(name)], errors="coerce").notna().mean())
        < 0.999
    ]
    if not names:
        return frame, {}
    if not feature_root.exists():
        raise FileNotFoundError(f"feature root does not exist: {feature_root}")
    grouped = frame.groupby("__symbol__", sort=False, observed=True).indices
    coverage: dict[str, float] = {}
    market_reference: Path | None = feature_root / "symbol=BTC_USD:USD.parquet"
    if not market_reference.exists():
        market_reference = next(feature_root.glob("symbol=*.parquet"), None)
    for start in range(0, len(names), max(1, int(batch_size))):
        batch = names[start : start + max(1, int(batch_size))]
        values = np.full((len(frame), len(batch)), np.nan, dtype=np.float32)
        market_batch = [name for name in batch if _is_broadcast_market_feature(name)]
        missing_by_feature = {
            name: (
                np.ones(len(frame), dtype=bool)
                if name not in frame.columns
                else pd.to_numeric(frame[name], errors="coerce").isna().to_numpy()
            )
            for name in batch
        }
        if market_batch and market_reference is not None:
            market_missing = np.logical_or.reduce(
                [missing_by_feature[name] for name in market_batch]
            )
            market_positions = np.flatnonzero(market_missing)
            market_features = read_symbol_features(
                str(market_reference),
                columns=market_batch,
                start_ts=frame.iloc[market_positions]["__ts__"].min(),
                end_ts=frame.iloc[market_positions]["__ts__"].max(),
            )
            if not market_features.empty:
                market_features.index = pd.to_datetime(
                    market_features.index, utc=True, errors="coerce"
                )
                market_features = market_features.loc[
                    ~market_features.index.duplicated(keep="last")
                ]
                aligned = market_features.reindex(
                    frame.iloc[market_positions]["__ts__"].to_numpy()
                )
                available = [name for name in market_batch if name in aligned]
                if available:
                    target_cols = [batch.index(name) for name in available]
                    values[np.ix_(market_positions, target_cols)] = aligned[
                        available
                    ].to_numpy(dtype=np.float32, copy=False)
        asset_batch = [name for name in batch if name not in market_batch]
        asset_groups = grouped.items() if asset_batch else ()
        for symbol, raw_positions in asset_groups:
            raw = np.asarray(raw_positions, dtype=np.int64)
            local_missing = np.logical_or.reduce(
                [missing_by_feature[name][raw] for name in asset_batch]
            )
            positions = raw[local_missing]
            if not len(positions):
                continue
            path = feature_root / f"symbol={str(symbol).replace('/', '_')}.parquet"
            if not path.exists():
                continue
            timestamps = frame.iloc[positions]["__ts__"]
            features = read_symbol_features(
                str(path),
                columns=asset_batch,
                start_ts=timestamps.min(),
                end_ts=timestamps.max(),
            )
            if features.empty:
                continue
            features.index = pd.to_datetime(features.index, utc=True, errors="coerce")
            features = features.loc[~features.index.duplicated(keep="last")]
            aligned = features.reindex(timestamps.to_numpy())
            available = [name for name in asset_batch if name in aligned.columns]
            if available:
                target_cols = [batch.index(name) for name in available]
                values[np.ix_(positions, target_cols)] = aligned[available].to_numpy(
                    dtype=np.float32, copy=False
                )
        appended = pd.DataFrame(values, columns=batch, index=frame.index)
        new_columns = [name for name in batch if name not in frame.columns]
        if new_columns:
            frame = pd.concat([frame, appended[new_columns]], axis=1, copy=False)
        for name in batch:
            if name in new_columns:
                continue
            current = pd.to_numeric(frame[name], errors="coerce")
            missing = current.isna()
            if bool(missing.any()):
                frame.loc[missing, name] = appended.loc[missing, name].to_numpy(
                    dtype=np.float32, copy=False
                )
        coverage.update(
            {
                name: float(pd.to_numeric(frame[name], errors="coerce").notna().mean())
                for name in batch
            }
        )
        print(
            json.dumps(
                {
                    "event": "residual_event_feature_backfill_batch",
                    "completed": min(start + len(batch), len(names)),
                    "total": len(names),
                    "market_features": len(market_batch),
                    "asset_features": len(asset_batch),
                }
            ),
            flush=True,
        )
        del values, appended
    return _downcast_frame(frame), coverage


def _tail_sparse_preentry_features(
    frame: pd.DataFrame,
    *,
    tail_start: pd.Timestamp,
    cap: int,
) -> list[str]:
    """Find historical state inputs missing from an appended prediction shard."""

    if int(cap) <= 0:
        return []
    tail = pd.to_datetime(frame["__ts__"], utc=True, errors="coerce").ge(tail_start)
    if not bool(tail.any()):
        return []
    forbidden = set(OUTCOME_COLUMNS) | {
        "row_id",
        "selected_top30",
        "selected_for_monitor",
        "prediction_evidence",
        "feature_parity_eligible",
    }
    candidates: list[tuple[float, str]] = []
    for name in frame.columns:
        if name in forbidden or str(name).startswith(("__", "threshold_", "policy_")):
            continue
        if not (
            pd.api.types.is_numeric_dtype(frame[name])
            or pd.api.types.is_bool_dtype(frame[name])
        ):
            continue
        values = pd.to_numeric(frame[name], errors="coerce")
        historical_coverage = float(values.loc[~tail].notna().mean())
        tail_coverage = float(values.loc[tail].notna().mean())
        if historical_coverage >= 0.35 and tail_coverage < 0.95:
            candidates.append((historical_coverage - tail_coverage, str(name)))
    candidates.sort(key=lambda item: (-item[0], item[1]))
    return [name for _, name in candidates[: int(cap)]]


def _outcome_safe(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.drop(
        columns=[name for name in OUTCOME_COLUMNS if name in frame.columns],
        errors="ignore",
    )


def _event_support(labelled: pd.DataFrame) -> pd.DataFrame:
    if labelled.empty or "resid_event_class" not in labelled:
        return pd.DataFrame()
    work = labelled.copy(deep=False)
    work["day"] = pd.to_datetime(work["__ts__"], utc=True, errors="coerce").dt.floor(
        "D"
    )
    work["event_class"] = work["resid_event_class"].astype(str)
    keys = ["side_name", "archetype_policy_key"]
    base = (
        work.groupby(keys, observed=True, sort=True)
        .agg(train_rows=("event_class", "size"))
        .reset_index()
    )
    support = (
        work.loc[work["event_class"].ne("normal")]
        .groupby([*keys, "event_class"], observed=True, sort=True)
        .agg(
            train_event_rows=("event_class", "size"),
            train_event_days=("day", "nunique"),
        )
        .reset_index()
    )
    if support.empty:
        base["train_event_rows"] = 0
        base["train_event_days"] = 0
        return base
    total = (
        support.groupby(keys, observed=True, sort=True)
        .agg(
            train_event_rows=("train_event_rows", "sum"),
            train_event_days=("train_event_days", "sum"),
        )
        .reset_index()
    )
    classes = support.pivot_table(
        index=keys,
        columns="event_class",
        values="train_event_days",
        aggfunc="sum",
        fill_value=0,
        observed=True,
    ).reset_index()
    classes.columns = [
        name if isinstance(name, str) and name in keys else f"train_{str(name)}_days"
        for name in classes.columns
    ]
    result = base.merge(total, on=keys, how="left").merge(classes, on=keys, how="left")
    support_columns = [name for name in result.columns if name not in keys]
    result[support_columns] = result[support_columns].fillna(0)
    return result


def _same_class_transfer_eligible(frame: pd.DataFrame) -> pd.Series:
    cls = frame["resid_event_class"].astype(str)
    support_col = "train_" + cls + "_days"
    values = np.zeros(len(frame), dtype=bool)
    for name in sorted(set(support_col)):
        if name not in frame.columns:
            continue
        mask = support_col.eq(name).to_numpy()
        values[mask] = (
            pd.to_numeric(frame.loc[mask, name], errors="coerce")
            .fillna(0)
            .ge(2)
            .to_numpy()
        )
    # Normal rows do not claim rare-event transfer; they remain valid ordinary
    # state observations but cannot be counted as evidence about event recall.
    values[cls.eq("normal").to_numpy()] = False
    return pd.Series(values, index=frame.index, dtype=bool)


def _metrics(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    work = frame.copy(deep=False)
    work["week_start"] = pd.to_datetime(
        work["__ts__"], utc=True, errors="coerce"
    ).dt.floor("D") - pd.to_timedelta(
        pd.to_datetime(work["__ts__"], utc=True, errors="coerce").dt.weekday, unit="D"
    )
    work["month"] = pd.to_datetime(
        work["__ts__"], utc=True, errors="coerce"
    ).dt.strftime("%Y-%m")
    group = work.groupby(keys, observed=True, dropna=False)
    return group.agg(
        rows=("__ts__", "size"),
        timestamps=("__ts__", "nunique"),
        symbols=("__symbol__", "nunique"),
        mean_ev_after_1pct=("ev_after_1pct", "mean"),
        sum_ev_after_1pct=("ev_after_1pct", "sum"),
        clean_exec_precision=("clean_exec", "mean"),
        dirty_positive_rate=("dirty_positive", "mean"),
        bad_mae_rate=("full_path_bad_mae_1r", "mean"),
        timeout_rate=("timeout", "mean"),
        mean_global_surprise=("resid_event_global_surprise", "mean"),
        mean_timestamp_neutral_surprise=(
            "resid_event_timestamp_neutral_surprise",
            "mean",
        ),
        mean_ev_timestamp_neutral_surprise=(
            "resid_event_ev_timestamp_neutral_surprise",
            "mean",
        ),
        mean_daily_neutral_z=("resid_event_daily_neutral_z", "mean"),
        mean_daily_ev_neutral_z=("resid_event_daily_ev_neutral_z", "mean"),
        assessment_hr8_surprise=("assessment_hr8_surprise", "mean"),
        market_peer_surprise=("resid_event_market_peer_surprise", "mean"),
        persistence_strength=("resid_event_persistence_strength", "mean"),
        large_event_strength=("resid_event_large_event_strength", "mean"),
    ).reset_index()


def _state_separation(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    state_cols = [
        name
        for name in frame.columns
        if name.startswith(
            (
                "resid_event_aegmm_gmm_cluster_posterior_",
                "resid_event_market_aegmm_gmm_cluster_posterior_",
            )
        )
    ]
    if not state_cols:
        return pd.DataFrame()
    out: list[dict[str, Any]] = []
    group_cols = ["side_name", "archetype_policy_key"]
    if "oos_month" in frame.columns:
        group_cols.insert(0, "oos_month")
    if "state_revision_cutoff" in frame.columns:
        group_cols.insert(0, "state_revision_cutoff")
    for group_key, group in frame.groupby(group_cols, observed=True, sort=True):
        values = group_key if isinstance(group_key, tuple) else (group_key,)
        group_values = dict(zip(group_cols, values, strict=True))
        for state_col in state_cols:
            posterior = pd.to_numeric(group[state_col], errors="coerce")
            if posterior.notna().sum() < 30 or float(posterior.max()) <= 0.0:
                continue
            weight = posterior.fillna(0.0).to_numpy(dtype=np.float64)
            denom = max(float(weight.sum()), 1e-8)
            out.append(
                {
                    **group_values,
                    "state_scope": (
                        "market"
                        if state_col.startswith("resid_event_market_aegmm_")
                        else "side_x_archetype"
                    ),
                    "state_feature": state_col,
                    "effective_rows": float(denom),
                    "mean_ev_after_1pct": float(
                        np.dot(
                            weight,
                            pd.to_numeric(
                                group["ev_after_1pct"], errors="coerce"
                            ).fillna(0.0),
                        )
                        / denom
                    ),
                    "clean_exec_precision": float(
                        np.dot(
                            weight,
                            pd.to_numeric(group["clean_exec"], errors="coerce").fillna(
                                0.0
                            ),
                        )
                        / denom
                    ),
                    "negative_event_rate": float(
                        np.dot(
                            weight,
                            group["resid_event_class"]
                            .astype(str)
                            .isin(["negative_residual_event", "adverse_path_event"])
                            .astype(float),
                        )
                        / denom
                    ),
                    "positive_event_rate": float(
                        np.dot(
                            weight,
                            group["resid_event_class"]
                            .astype(str)
                            .isin(
                                ["positive_residual_event", "favorable_near_miss_event"]
                            )
                            .astype(float),
                        )
                        / denom
                    ),
                    "mean_timestamp_neutral_surprise": float(
                        np.dot(
                            weight,
                            pd.to_numeric(
                                group["resid_event_timestamp_neutral_surprise"],
                                errors="coerce",
                            ).fillna(0.0),
                        )
                        / denom
                    ),
                }
            )
    return pd.DataFrame(out)


def _state_target_probability_separation(frame: pd.DataFrame) -> pd.DataFrame:
    """Check whether each frozen state prior separates its own OOS outcome.

    A posterior prior is only useful if its high-probability tail materially
    enriches the target mechanism on unseen rows.  The comparison is local to
    side x archetype so a broad long/short or archetype mix cannot manufacture
    apparent separation.
    """

    if frame.empty:
        return pd.DataFrame()
    expected_prefix = f"{RESIDUAL_EVENT_PREFIX}expected_"
    expected = [
        name for name in frame.columns
        if name.startswith(expected_prefix)
        and f"{RESIDUAL_EVENT_TARGET_PREFIX}{name[len(expected_prefix):]}" in frame
    ]
    if not expected:
        return pd.DataFrame()
    group_cols = ["oos_month", "side_name", "archetype_policy_key"]
    rows: list[dict[str, Any]] = []
    for group_key, group in frame.groupby(group_cols, observed=True, sort=True):
        group_values = dict(zip(group_cols, group_key, strict=True))
        for expected_name in expected:
            suffix = expected_name[len(expected_prefix) :]
            target_name = f"{RESIDUAL_EVENT_TARGET_PREFIX}{suffix}"
            probability = pd.to_numeric(group[expected_name], errors="coerce")
            target = pd.to_numeric(group[target_name], errors="coerce")
            valid = probability.notna() & target.notna()
            if int(valid.sum()) < 80 or probability.loc[valid].nunique() < 4:
                continue
            # This report is intentionally a *rate* separation report.  The
            # residual-state contract also carries continuous path-damage and
            # quality-gap targets, but a ratio of their signed means is not a
            # probability lift and produces meaningless negative/huge values.
            values = np.unique(target.loc[valid].to_numpy(dtype=np.float32))
            if values.size == 0 or not bool(np.isin(values, (0.0, 1.0)).all()):
                continue
            threshold = float(probability.loc[valid].quantile(0.80))
            high = valid & probability.ge(threshold)
            if int(high.sum()) < 16:
                continue
            base_rate = float(target.loc[valid].mean())
            if base_rate <= 0.0:
                continue
            high_rate = float(target.loc[high].mean())
            rows.append(
                {
                    **group_values,
                    "state_target": suffix,
                    "rows": int(valid.sum()),
                    "high_probability_rows": int(high.sum()),
                    "high_probability_cutoff": threshold,
                    "base_target_rate": base_rate,
                    "high_probability_target_rate": high_rate,
                    "target_rate_lift": high_rate / max(base_rate, 1e-6),
                    "mean_ev_high_probability": float(
                        pd.to_numeric(group.loc[high, "ev_after_1pct"], errors="coerce").mean()
                    ),
                }
            )
    return pd.DataFrame(rows)


def _surprise_autocorrelation(
    frame: pd.DataFrame,
    keys: list[str],
    *,
    surprise_col: str,
    population: str,
) -> pd.DataFrame:
    """Signed consecutive-day surprise persistence without gap compression."""

    if frame.empty or surprise_col not in frame:
        return pd.DataFrame()
    work = frame.copy(deep=False)
    work["day"] = pd.to_datetime(work["__ts__"], utc=True, errors="coerce").dt.floor(
        "D"
    )
    daily_keys = [*keys, "day"]
    daily = (
        work.groupby(daily_keys, observed=True, dropna=False, sort=True)[surprise_col]
        .mean()
        .rename("surprise")
        .reset_index()
    )
    rows: list[dict[str, Any]] = []
    grouped: Iterable[tuple[Any, pd.DataFrame]]
    if keys:
        grouper: str | list[str] = keys[0] if len(keys) == 1 else keys
        grouped = daily.groupby(grouper, observed=True, dropna=False, sort=True)
    else:
        grouped = [("global", daily)]
    for group_key, group in grouped:
        ordered = group.sort_values("day", kind="stable")
        current = pd.to_numeric(ordered["surprise"], errors="coerce").to_numpy(
            dtype=np.float64
        )
        days = pd.to_datetime(ordered["day"], utc=True, errors="coerce")
        contiguous = days.diff().dt.days.eq(1).to_numpy()
        valid = contiguous & np.isfinite(current) & np.isfinite(np.roll(current, 1))
        valid[0] = False
        now = current[valid]
        previous = np.roll(current, 1)[valid]
        if len(now) >= 3 and np.std(now) > 1e-10 and np.std(previous) > 1e-10:
            autocorr = float(np.corrcoef(now, previous)[0, 1])
        else:
            autocorr = np.nan
        payload: dict[str, Any] = {}
        if keys:
            values = group_key if isinstance(group_key, tuple) else (group_key,)
            payload.update(dict(zip(keys, values, strict=True)))
        payload.update(
            {
                "population": population,
                "surprise_col": surprise_col,
                "days": int(len(ordered)),
                "consecutive_pairs": int(len(now)),
                "signed_surprise_mean": float(np.nanmean(current)),
                "signed_lag1_autocorrelation": autocorr,
                "signed_lag1_product_mean": float(np.mean(now * previous))
                if len(now)
                else np.nan,
                "adverse_lag1_product_mean": float(
                    np.mean(np.maximum(-now, 0.0) * np.maximum(-previous, 0.0))
                )
                if len(now)
                else np.nan,
                "favorable_lag1_product_mean": float(
                    np.mean(np.maximum(now, 0.0) * np.maximum(previous, 0.0))
                )
                if len(now)
                else np.nan,
            }
        )
        rows.append(payload)
    return pd.DataFrame(rows)


def _event_calendar(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    work = frame.copy(deep=False)
    work["day"] = pd.to_datetime(work["__ts__"], utc=True, errors="coerce").dt.floor(
        "D"
    )
    return (
        work.groupby(
            ["day", "side_name", "archetype_policy_key"],
            observed=True,
            dropna=False,
            sort=True,
        )
        .agg(
            rows=("__ts__", "size"),
            mean_ev_after_1pct=("ev_after_1pct", "mean"),
            clean_exec_precision=("clean_exec", "mean"),
            mean_timestamp_neutral_surprise=(
                "resid_event_timestamp_neutral_surprise",
                "mean",
            ),
            mean_ev_timestamp_neutral_surprise=(
                "resid_event_ev_timestamp_neutral_surprise",
                "mean",
            ),
            mean_market_peer_surprise=("resid_event_market_peer_surprise", "mean"),
            daily_neutral_z=("resid_event_daily_neutral_z", "mean"),
            daily_ev_neutral_z=("resid_event_daily_ev_neutral_z", "mean"),
            persistence_strength=("resid_event_persistence_strength", "mean"),
            large_event_strength=("resid_event_large_event_strength", "mean"),
            adverse_event_rows=(
                "resid_event_class",
                lambda values: int(
                    values.astype(str)
                    .isin(["negative_residual_event", "adverse_path_event"])
                    .sum()
                ),
            ),
            favorable_event_rows=(
                "resid_event_class",
                lambda values: int(
                    values.astype(str)
                    .isin(["positive_residual_event", "favorable_near_miss_event"])
                    .sum()
                ),
            ),
        )
        .reset_index()
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--candidate-columns-json",
        type=Path,
        default=None,
        help="Optional projected candidate feature contract (list or feature_names JSON).",
    )
    parser.add_argument(
        "--max-source-features",
        type=int,
        default=0,
        help=(
            "Project a deterministic, family-balanced subset of observable source "
            "features before loading wide historical shards; 0 retains every source column."
        ),
    )
    parser.add_argument(
        "--max-train-rows-per-source-shard",
        type=int,
        default=0,
        help=(
            "Deterministically sample beginning/middle/end rows from each historical "
            "training shard before concatenation; requested OOS months remain complete."
        ),
    )
    parser.add_argument("--extra-candidate", type=Path, action="append", default=[])
    parser.add_argument(
        "--extra-prediction",
        type=Path,
        action="append",
        default=[],
        help=(
            "Frozen prediction shard with resolved outcomes, normalized to the "
            "candidate contract for a later OOS month."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--oos-months", default=DEFAULT_MONTHS)
    parser.add_argument("--score-col", default="score")
    parser.add_argument(
        "--feature-scope",
        choices=("meta_full", "base_directional"),
        default="meta_full",
    )
    parser.add_argument("--feature-root", type=Path, default=None)
    parser.add_argument("--supplemental-feature-cap", type=int, default=0)
    parser.add_argument(
        "--local-feature-basket",
        choices=("all", "source"),
        default="all",
        help=(
            "Use all hydrated features for local models, or keep local side x "
            "archetype screening on the original candidate-source columns while "
            "allowing the secondary market block to use the hydrated basket."
        ),
    )
    parser.add_argument(
        "--broad-config-basket",
        action="store_true",
        help=(
            "Append the configured observable meta/state basket from the frozen "
            "feature store before local MI and shallow-LGBM screening."
        ),
    )
    parser.add_argument(
        "--expanded-model-context",
        action="store_true",
        help=(
            "Append existing inference-safe meta uncertainty, OOD, leaf, support, "
            "drift, entropy, posterior and distance columns to the projected source."
        ),
    )
    parser.add_argument("--tail-feature-backfill-cap", type=int, default=256)
    parser.add_argument("--feature-append-batch-size", type=int, default=32)
    parser.add_argument("--embargo-hours", type=float, default=12.0)
    parser.add_argument("--min-train-rows", type=int, default=50_000)
    parser.add_argument("--min-local-state-rows", type=int, default=1_500)
    parser.add_argument("--min-market-state-rows", type=int, default=1_000)
    parser.add_argument("--state-train-through", default="")
    parser.add_argument(
        "--refit-schedule",
        choices=("milestone", "monthly"),
        default="milestone",
        help="Milestone freezes AE/GMM semantics between rare-state revisions.",
    )
    parser.add_argument("--no-lgbm-screen", action="store_true")
    parser.add_argument("--max-features-after-mi", type=int, default=72)
    parser.add_argument("--max-features-after-lgbm", type=int, default=48)
    parser.add_argument("--market-max-features", type=int, default=48)
    parser.add_argument("--ae-max-train-rows", type=int, default=15_000)
    parser.add_argument("--gmm-max-train-rows", type=int, default=100_000)
    parser.add_argument("--ae-max-iter", type=int, default=160)
    parser.add_argument(
        "--ae-gmm-clusters",
        default="3,4,5,6",
        help="Comma-separated local AE/GMM cluster-count candidates.",
    )
    parser.add_argument(
        "--ae-gmm-reg-covars",
        default="0.0001,0.001,0.003",
        help="Comma-separated positive GMM covariance regularizers.",
    )
    parser.add_argument(
        "--ae-gmm-covariance-types",
        default="diag,tied,full",
        help="Comma-separated GMM covariance types.",
    )
    parser.add_argument(
        "--allow-side-fallback",
        action="store_true",
        help="Diagnostic only; production research defaults to strict side x archetype fits.",
    )
    parser.add_argument(
        "--market-secondary",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Fit the broader market residual-state block after local side x archetype states.",
    )
    parser.add_argument(
        "--executable-quality-targets",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Fit train-only correct-direction versus executable-quality priors: "
            "negative net EV, adverse-path damage and correct-direction-but-bad-trade."
        ),
    )
    parser.add_argument(
        "--transition-trajectory-features",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Emit causal side x archetype trajectory summaries for OI/funding, "
            "breadth, short-covering and liquidity state changes."
        ),
    )
    parser.add_argument("--small-smoke", action="store_true")
    args = parser.parse_args()

    months = _parse_months(args.oos_months)
    max_required_end = max(
        pd.Timestamp((pd.Period(month, freq="M") + 1).start_time, tz="UTC")
        for month in months
    )
    if args.state_train_through:
        max_required_end = min(
            max_required_end, pd.Timestamp(args.state_train_through, tz="UTC")
        )
    extra_sources = [*args.extra_candidate, *args.extra_prediction]
    candidate_columns = _load_candidate_columns(args.candidate_columns_json)
    if candidate_columns is None and int(args.max_source_features) > 0:
        candidate_columns = _project_source_columns(
            args.candidate_root,
            end=max_required_end,
            max_features=int(args.max_source_features),
        )
    if args.expanded_model_context:
        expanded = _expanded_model_context_columns(
            [*args.extra_candidate, *args.extra_prediction]
        )
        candidate_columns = list(
            dict.fromkeys([*(candidate_columns or CANDIDATE_REQUIRED_COLUMNS), *expanded])
        )
    data, source_coverage = _load_shards(
        args.candidate_root,
        extra_sources,
        end=max_required_end,
        requested_columns=candidate_columns,
        max_train_rows_per_shard=int(args.max_train_rows_per_source_shard),
        full_months=months,
    )
    data, candidate_contract_aliases = _normalise_candidate_contract(
        data, score_col=str(args.score_col)
    )
    source_candidate_features = list(data.columns)
    if args.state_train_through:
        cutoff = pd.Timestamp(args.state_train_through, tz="UTC")
        data = data.loc[data["__ts__"].lt(cutoff)].copy()
    supplemental_coverage: dict[str, float] = {}
    supplemental_features: list[str] = []
    tail_backfill_features: list[str] = []
    feature_root = args.feature_root
    supplemental_cap = int(args.supplemental_feature_cap)
    if args.broad_config_basket:
        feature_root = feature_root or DEFAULT_FEATURE_ROOT
        supplemental_cap = max(supplemental_cap, 320)
    if feature_root is not None:
        if supplemental_cap > 0:
            supplemental_features = _configured_supplemental_features(
                data, supplemental_cap
            )
        latest_month_start = max(
            pd.Timestamp(pd.Period(month, freq="M").start_time, tz="UTC")
            for month in months
        )
        tail_backfill_features = _tail_sparse_preentry_features(
            data,
            tail_start=latest_month_start,
            cap=int(args.tail_feature_backfill_cap),
        )
        requested_features = list(
            dict.fromkeys([*supplemental_features, *tail_backfill_features])
        )
        data, supplemental_coverage = _append_feature_store_basket(
            data,
            feature_root=feature_root,
            requested=requested_features,
            batch_size=int(args.feature_append_batch_size),
        )
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    config = ResidualEventArchetypeConfig(
        score_col=str(args.score_col),
        feature_scope=str(args.feature_scope),
        lgbm_enabled=not bool(args.no_lgbm_screen),
        max_features_after_mi=max(8, int(args.max_features_after_mi)),
        max_features_after_lgbm=max(8, int(args.max_features_after_lgbm)),
        market_max_features=max(8, int(args.market_max_features)),
        allow_side_fallback=bool(args.allow_side_fallback),
        enable_market_secondary=bool(args.market_secondary),
        enable_executable_quality_targets=bool(args.executable_quality_targets),
        enable_transition_trajectory_features=bool(args.transition_trajectory_features),
        ae_gmm_max_train_rows=(
            600 if args.small_smoke else max(1_000, int(args.ae_max_train_rows))
        ),
        gmm_max_train_rows=(
            1_500 if args.small_smoke else max(1_000, int(args.gmm_max_train_rows))
        ),
        ae_gmm_max_iter=12 if args.small_smoke else max(20, int(args.ae_max_iter)),
        ae_gmm_clusters=_parse_int_csv(
            args.ae_gmm_clusters, name="--ae-gmm-clusters"
        ),
        ae_gmm_reg_covars=_parse_float_csv(
            args.ae_gmm_reg_covars, name="--ae-gmm-reg-covars"
        ),
        ae_gmm_covariance_types=_parse_choice_csv(
            args.ae_gmm_covariance_types,
            name="--ae-gmm-covariance-types",
            allowed={"diag", "tied", "full", "spherical"},
        ),
        min_global_threshold_rows=500 if args.small_smoke else 2_000,
        min_local_threshold_rows=150 if args.small_smoke else 600,
        min_local_state_rows=(
            300 if args.small_smoke else max(200, int(args.min_local_state_rows))
        ),
        min_side_state_rows=600 if args.small_smoke else 3_000,
        market_min_rows=(
            300 if args.small_smoke else max(200, int(args.min_market_state_rows))
        ),
        min_event_class_rows=10 if args.small_smoke else 30,
        mi_sample_rows=5_000 if args.small_smoke else 45_000,
    )
    all_predictions: list[pd.DataFrame] = []
    all_screening: list[pd.DataFrame] = []
    all_catalogs: list[pd.DataFrame] = []
    fold_rows: list[dict[str, Any]] = []
    state_cache: dict[
        pd.Timestamp, tuple[ResidualEventArchetypeState, pd.DataFrame]
    ] = {}
    emitted_revisions: set[pd.Timestamp] = set()
    for fold_no, month in enumerate(months):
        start = pd.Timestamp(pd.Period(month, freq="M").start_time, tz="UTC")
        end = pd.Timestamp((pd.Period(month, freq="M") + 1).start_time, tz="UTC")
        revision_cutoff = _revision_cutoff_for_month(
            start, schedule=str(args.refit_schedule)
        )
        train_end = revision_cutoff - pd.Timedelta(hours=float(args.embargo_hours))
        train = data.loc[data["__ts__"].lt(train_end)].copy()
        valid = data.loc[data["__ts__"].ge(start) & data["__ts__"].lt(end)].copy()
        if len(train) < int(args.min_train_rows) or len(valid) < 100:
            fold_rows.append(
                {
                    "month": month,
                    "status": "skipped_insufficient_coverage",
                    "train_rows": int(len(train)),
                    "valid_rows": int(len(valid)),
                }
            )
            continue
        cached = state_cache.get(revision_cutoff)
        if cached is None:
            local_candidates = (
                source_candidate_features
                if str(args.local_feature_basket) == "source"
                else None
            )
            state = ResidualEventArchetypeState(config).fit(
                train,
                candidate_features=local_candidates,
                market_candidate_features=None,
            )
            train_labelled = state.annotate_outcomes_for_assessment(train)
            support = _event_support(train_labelled)
            del train_labelled
            state_cache[revision_cutoff] = (state, support)
            revision_dir = (
                output_dir / "revisions" / revision_cutoff.strftime("%Y-%m-%d")
            )
            revision_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(
                state,
                revision_dir / "residual_event_state.joblib",
                compress=3,
            )
            _write_json(
                revision_dir / "manifest.json",
                {
                    "revision_cutoff": revision_cutoff,
                    "train_end": train_end,
                    "train_rows": len(train),
                    **state.manifest(),
                },
            )
        else:
            state, support = cached
        valid_labelled = state.annotate_outcomes_for_assessment(valid)
        valid_features = state.transform_oos(_outcome_safe(valid))
        overlay = causal_eight_day_hit_rate_overlay(valid_labelled, config=config)
        # Extra OOS shards may already carry outputs from an older residual
        # state revision.  The current frozen fold is authoritative: replace
        # stale generated columns instead of emitting duplicate Parquet names.
        refreshed_columns = set(valid_features.columns).union(overlay.columns)
        valid_labelled = valid_labelled.drop(
            columns=[
                name for name in refreshed_columns if name in valid_labelled.columns
            ],
            errors="ignore",
        )
        valid_result = pd.concat(
            [
                valid_labelled.reset_index(drop=True),
                valid_features.reset_index(drop=True),
                overlay.reset_index(drop=True),
            ],
            axis=1,
        )
        if not support.empty:
            valid_result = valid_result.merge(
                support, on=["side_name", "archetype_policy_key"], how="left"
            )
            valid_result["prior_event_transfer_eligible"] = (
                valid_result["train_event_days"].fillna(0).ge(2)
            )
            valid_result["prior_same_class_transfer_eligible"] = (
                _same_class_transfer_eligible(valid_result)
            )
        else:
            valid_result["prior_event_transfer_eligible"] = False
            valid_result["prior_same_class_transfer_eligible"] = False
        valid_result["oos_month"] = month
        valid_result["fold_train_end"] = train_end
        valid_result["state_revision_cutoff"] = revision_cutoff
        fold_dir = output_dir / "folds" / month
        fold_dir.mkdir(parents=True, exist_ok=True)
        valid_result.to_parquet(
            fold_dir / "oos_residual_event_states.parquet",
            index=False,
            compression="zstd",
        )
        _write_json(
            fold_dir / "manifest.json",
            {
                "month": month,
                "state_revision_cutoff": revision_cutoff,
                "train_end": train_end,
                "train_rows": len(train),
                "valid_rows": len(valid),
                **state.manifest(),
            },
        )
        all_predictions.append(valid_result)
        if (
            revision_cutoff not in emitted_revisions
            and not state.feature_metrics_.empty
        ):
            screen = state.feature_metrics_.copy()
            screen["state_revision_cutoff"] = revision_cutoff
            all_screening.append(screen)
        if revision_cutoff not in emitted_revisions and not state.event_catalog_.empty:
            catalog = state.event_catalog_.copy()
            catalog["state_revision_cutoff"] = revision_cutoff
            all_catalogs.append(catalog)
        emitted_revisions.add(revision_cutoff)
        fold_rows.append(
            {
                "month": month,
                "status": "complete",
                "state_revision_cutoff": str(revision_cutoff),
                "train_rows": int(len(train)),
                "valid_rows": int(len(valid)),
                "train_start": str(train["__ts__"].min()),
                "train_end": str(train_end),
                "valid_start": str(start),
                "valid_end": str(end),
                "local_models": int(len(state.local_models)),
                "side_fallback_models": int(len(state.side_models)),
            }
        )
        print(
            json.dumps(
                {
                    "event": "residual_event_state_fold_complete",
                    "month": month,
                    "state_revision_cutoff": str(revision_cutoff),
                    "train_rows": len(train),
                    "valid_rows": len(valid),
                    "local_models": len(state.local_models),
                }
            ),
            flush=True,
        )

    combined = (
        pd.concat(all_predictions, ignore_index=True)
        if all_predictions
        else pd.DataFrame()
    )
    screening = (
        pd.concat(all_screening, ignore_index=True) if all_screening else pd.DataFrame()
    )
    catalog = (
        pd.concat(all_catalogs, ignore_index=True) if all_catalogs else pd.DataFrame()
    )
    combined.to_parquet(
        output_dir / "oos_residual_event_states.parquet",
        index=False,
        compression="zstd",
    )
    screening.to_csv(output_dir / "local_feature_screening.csv", index=False)
    catalog.to_csv(output_dir / "local_state_catalog.csv", index=False)
    if not combined.empty:
        autocorrelation_rows: list[pd.DataFrame] = []
        for population, column in (
            ("top10", "resid_event_top10_population"),
            ("top20", "resid_event_top20_population"),
        ):
            selected = combined.loc[
                pd.to_numeric(combined[column], errors="coerce").gt(0.5)
            ].copy()
            _metrics(selected, ["oos_month"]).to_csv(
                output_dir / f"{population}_month_metrics.csv", index=False
            )
            _metrics(selected, ["week_start"]).to_csv(
                output_dir / f"{population}_week_metrics.csv", index=False
            )
            _metrics(
                selected, ["oos_month", "side_name", "archetype_policy_key"]
            ).to_csv(
                output_dir / f"{population}_month_side_archetype_metrics.csv",
                index=False,
            )
            _metrics(
                selected, ["week_start", "side_name", "archetype_policy_key"]
            ).to_csv(
                output_dir / f"{population}_week_side_archetype_metrics.csv",
                index=False,
            )
            _metrics(
                selected,
                [
                    "oos_month",
                    "side_name",
                    "archetype_policy_key",
                    "resid_event_class",
                ],
            ).to_csv(output_dir / f"{population}_event_class_metrics.csv", index=False)
            _state_separation(selected).to_csv(
                output_dir / f"{population}_state_separation.csv", index=False
            )
            _state_target_probability_separation(selected).to_csv(
                output_dir / f"{population}_state_target_probability_separation.csv",
                index=False,
            )
            for keys in (
                [],
                ["side_name"],
                ["side_name", "archetype_policy_key"],
                ["oos_month", "side_name", "archetype_policy_key"],
            ):
                for surprise_col in (
                    "resid_event_global_surprise",
                    "resid_event_timestamp_neutral_surprise",
                    "resid_event_ev_global_surprise",
                    "resid_event_ev_timestamp_neutral_surprise",
                    "assessment_hr8_surprise",
                ):
                    autocorrelation_rows.append(
                        _surprise_autocorrelation(
                            selected,
                            keys,
                            surprise_col=surprise_col,
                            population=population,
                        )
                    )
        pd.concat(autocorrelation_rows, ignore_index=True).to_csv(
            output_dir / "signed_surprise_autocorrelation.csv", index=False
        )
        _event_calendar(combined).to_csv(
            output_dir / "residual_event_calendar.csv", index=False
        )
    fold = pd.DataFrame(fold_rows)
    fold.to_csv(output_dir / "fold_coverage.csv", index=False)
    _write_json(
        output_dir / "manifest.json",
        {
            "schema": "residual_event_archetype_discovery_runner_v1",
            "config": asdict(config),
            "candidate_root": str(args.candidate_root),
            "extra_candidates": [str(path) for path in args.extra_candidate],
            "extra_predictions": [str(path) for path in args.extra_prediction],
            "candidate_contract_aliases": candidate_contract_aliases,
            "candidate_columns_contract": {
                "path": str(args.candidate_columns_json or ""),
                "requested_feature_count": int(len(candidate_columns or [])),
                "projected": candidate_columns is not None,
            },
            "source_coverage": source_coverage,
            "supplemental_feature_store": {
                "root": str(feature_root) if feature_root is not None else "",
                "requested_features": supplemental_features,
                "tail_backfill_features": tail_backfill_features,
                "coverage": supplemental_coverage,
                "append_batch_size": int(args.feature_append_batch_size),
            },
            "local_feature_basket": {
                "mode": str(args.local_feature_basket),
                "source_candidate_count": int(len(source_candidate_features)),
                "secondary_market_uses_hydrated_basket": True,
            },
            "folds": fold_rows,
            "rows": int(len(combined)),
            "oos_months_requested": months,
            "refit_schedule": str(args.refit_schedule),
            "revision_cutoffs": list(DEFAULT_REVISION_CUTOFFS),
            "oos_months_completed": sorted(
                combined.get("oos_month", pd.Series(dtype=str))
                .astype(str)
                .unique()
                .tolist()
            ),
            "assessment_overlay": {
                "name": "causal_8d_hit_rate_smoother_without_regime_calibration",
                "role": "assessment only; never input to feature screening or AE/GMM",
            },
            "discovery_order": [
                "side_x_archetype_timestamp_neutral_residual_events",
                "same_timestamp_broader_market_peer_surprise",
            ],
            "leakage_contract": {
                "fit": "all thresholds, score calibration, feature screening, AE/GMM and outcome priors use only rows before each OOS month minus embargo",
                "oos": "OOS AE/GMM transform receives outcome-stripped rows only",
                "event_labels": "realized OOS outcomes are used after scoring only for assessment",
                "rare_regime_transfer": "prior_event_transfer_eligible requires at least two earlier event days in the same side x archetype",
            },
        },
    )


if __name__ == "__main__":
    main()
