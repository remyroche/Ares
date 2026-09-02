#!/usr/bin/env python3
"""Rebuild frozen base/meta predictions and outcomes for an arbitrary period.

The original entry point was July-specific.  It now also supports an explicit
label-free frozen-backcast mode for historical diagnostic work.  A backcast is
not OOF/OOS evidence: its evidence scope is persisted in every output row and
in the manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from extreme_price_movements.data_store import read_symbol_features
from extreme_price_movements.fast_funcs import numba_rolling_zscore_fused
from extreme_price_movements.features_gmm_ae import transform_ae_gmm_features
from extreme_price_movements.inference.canonical_meta_postprocessor import (
    CanonicalMetaPostprocessor,
)
from extreme_price_movements.inference.feature_generator import (
    raw_required_feature_keys,
)
from extreme_price_movements.inference.live_meta_feature_overlays import (
    apply_live_meta_reliability_priors,
    materialize_live_source_regime_features,
)
from extreme_price_movements.inference.live_policy_archetype import (
    load_live_policy_archetype_classifier,
    predict_live_policy_archetype,
    predict_observable_policy_archetype,
)
from extreme_price_movements.inference.s52_meta_ood import (
    append_s52_meta_ood_features,
)
from extreme_price_movements.inference.s52_meta_score_alignment import (
    apply_s52_meta_score_alignment,
)
from extreme_price_movements.inference.side_residual_expert import (
    SideResidualExpertBundle,
)
from extreme_price_movements.inference.threshold_basis_policy import (
    apply_threshold_basis_policy_to_decisions,
    load_threshold_basis_policy,
)
from extreme_price_movements.lgbm_pipeline import (
    _append_meta_post_selection_ood_features,
    _fit_meta_post_selection_ood_reference,
)
from extreme_price_movements.static_feature_store import read_static_features
from extreme_price_movements.model_loader import load_meta_models_from_pickle
from extreme_price_movements.regime_ev_calibration import (
    required_feature_columns,
)
from extreme_price_movements.unsupervised_regime_learning.feature_registry import (
    UNSUPERVISED_REGIME_FEATURE_MECHANISMS,
)
from scripts.materialize_archetype_conditioned_trailing_labels import (
    _arm_from_policy,
)
from scripts.report_meta_residual_daily_old_new import _outcomes_from_labels
from scripts.run_label_first_touch_capture_proxy import (
    _fetch_policy_paths,
    _first_touch_capture_outcome,
)
from scripts.run_s52_train_meta_regime_handoff_smoke import (
    _add_fold_base_prior_features,
    _add_fold_reliability_features,
    _load_joined_frame,
)
from scripts.run_train_meta_residual_archetype_enhancement import (
    _add_reference_fold_features,
)
from scripts.score_compare_meta_residual_july_oos import _append_store_features

KEYS = ["__ts__", "__symbol__", "side_name"]
OOD_PREFIX = "meta_sel_ood_"
DERIVED_META_FEATURES = {"carry_adj_ret_self_z_10h"}


def _attribution_feature_family(name: str) -> str:
    """Map a model input to a compact economic attribution family."""
    feature = str(name)
    registered = UNSUPERVISED_REGIME_FEATURE_MECHANISMS.get(feature)
    if registered:
        return str(registered)
    low = feature.lower()
    if any(
        token in low for token in ("gmm", "dae", "reconstruction", "mahal", "latent")
    ):
        return "latent_state"
    if any(token in low for token in ("fund", "open_interest", "oi_", "_oi")):
        return "derivatives_positioning"
    if any(
        token in low
        for token in ("breadth", "market_", "mkt_", "xasset", "cross_asset")
    ):
        return "market_cross_asset"
    if any(
        token in low
        for token in ("spread", "liquidity", "volume", "amihud", "orderbook", "ob_")
    ):
        return "liquidity_microstructure"
    if any(token in low for token in ("vol", "atr", "range", "shock")):
        return "volatility_path"
    if any(
        token in low
        for token in ("trend", "momentum", "mom_", "ema", "breakout", "pullback")
    ):
        return "direction_trend"
    if any(token in low for token in ("hour_", "dow_", "session", "season")):
        return "calendar_session"
    if "resid" in low or "relative" in low or "minus_mkt" in low:
        return "relative_residual"
    if low in {"side", "__side__"} or "archetype" in low:
        return "side_archetype"
    return "other"


def _predict_with_family_attributions(
    model: Any,
    matrix: pd.DataFrame,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Predict and aggregate LightGBM contributions without retaining wide SHAP rows."""
    scores = np.asarray(model.predict(matrix), dtype=np.float32).reshape(-1)
    predictor = getattr(model, "booster_", model)
    try:
        raw = np.asarray(predictor.predict(matrix, pred_contrib=True), dtype=np.float32)
    except (AttributeError, TypeError, ValueError):
        return scores, pd.DataFrame(index=matrix.index)
    if raw.ndim != 2 or raw.shape != (len(matrix), matrix.shape[1] + 1):
        return scores, pd.DataFrame(index=matrix.index)

    feature_values = raw[:, :-1]
    families = np.asarray(
        [_attribution_feature_family(name) for name in matrix.columns], dtype=object
    )
    data: dict[str, np.ndarray] = {
        "base_attr_bias": raw[:, -1].astype(np.float32, copy=False),
    }
    total_abs = np.maximum(np.abs(feature_values).sum(axis=1), 1e-12)
    family_abs_shares: list[np.ndarray] = []
    for family in sorted(set(families.tolist())):
        family_mask = families == family
        signed = feature_values[:, family_mask].sum(axis=1).astype(np.float32)
        abs_share = (
            np.abs(feature_values[:, family_mask]).sum(axis=1) / total_abs
        ).astype(np.float32)
        data[f"base_attr_signed__{family}"] = signed
        data[f"base_attr_abs_share__{family}"] = abs_share
        family_abs_shares.append(abs_share)
    data["base_attr_abs_concentration"] = (
        np.max(np.column_stack(family_abs_shares), axis=1).astype(np.float32)
        if family_abs_shares
        else np.zeros(len(matrix), dtype=np.float32)
    )
    return scores, pd.DataFrame(data, index=matrix.index)


def _store_input_columns(
    ae_gmm_columns: list[str],
    base_columns: list[str],
    extra_columns: list[str] | None = None,
) -> list[str]:
    """Expand generated model aliases to the raw columns needed to build them."""
    model_columns = list(
        dict.fromkeys(ae_gmm_columns + base_columns + list(extra_columns or []))
    )
    raw_columns = sorted(raw_required_feature_keys(model_columns))
    return list(dict.fromkeys(model_columns + raw_columns))


def _utc(values: Any) -> pd.Series:
    return pd.to_datetime(values, utc=True, errors="coerce")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_provenance(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(path),
        "resolved_path": str(resolved),
        "sha256": _sha256(resolved),
        "size_bytes": int(resolved.stat().st_size),
    }


def _feature_symbols(feature_root: Path) -> list[str]:
    symbols: list[str] = []
    for path in feature_root.glob("symbol=*.parquet"):
        encoded = path.name[len("symbol=") : -len(".parquet")]
        if "_USD:USD" not in encoded:
            continue
        symbols.append(encoded.replace("_USD:USD", "/USD:USD"))
    return sorted(set(symbols))


def _load_optional_labels(
    labels_dir: Path | None,
    *,
    year: int,
    month: int,
) -> pd.DataFrame:
    """Load side labels when available, otherwise return an empty contract."""

    columns = [
        "__ts__",
        "__symbol__",
        "side_name",
        "__barrier_pct__",
    ]
    if labels_dir is None:
        return pd.DataFrame(columns=columns)
    parts: list[pd.DataFrame] = []
    for side in ("long", "short"):
        path = labels_dir / f"train_global_{side}_5_{year:04d}_{month:02d}.parquet"
        if not path.exists():
            continue
        frame = pd.read_parquet(path)
        frame["__ts__"] = _utc(frame["__ts__"])
        if "side_name" not in frame:
            frame["side_name"] = side
        if "__barrier_pct__" not in frame:
            frame["__barrier_pct__"] = np.nan
        parts.append(frame)
    if not parts:
        return pd.DataFrame(columns=columns)
    return pd.concat(parts, ignore_index=True, copy=False)


def _load_optional_label_period(
    labels_dir: Path | None,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    if labels_dir is None:
        return _load_optional_labels(None, year=start.year, month=start.month)
    months = pd.period_range(
        start=start.tz_localize(None).to_period("M"),
        end=(end - pd.Timedelta(nanoseconds=1)).tz_localize(None).to_period("M"),
        freq="M",
    )
    parts = [
        _load_optional_labels(labels_dir, year=period.year, month=period.month)
        for period in months
    ]
    parts = [part for part in parts if not part.empty]
    if not parts:
        return _load_optional_labels(None, year=start.year, month=start.month)
    return pd.concat(parts, ignore_index=True, copy=False)


def _read_tail_feature_rows(
    feature_root: Path,
    *,
    symbols: list[str],
    timestamps: pd.DatetimeIndex,
    columns: list[str],
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    start, end = timestamps.min(), timestamps.max()
    for symbol in symbols:
        path = feature_root / f"symbol={symbol.replace('/', '_')}.parquet"
        if not path.exists():
            continue
        try:
            values = read_symbol_features(
                str(path), columns=columns, start_ts=start, end_ts=end
            )
        except Exception:
            continue
        if values.empty:
            continue
        values.index = _utc(pd.Series(values.index)).to_numpy()
        values = values.loc[~values.index.duplicated(keep="last")].reindex(timestamps)
        values["__ts__"] = timestamps
        values["__symbol__"] = symbol
        parts.append(values.reset_index(drop=True))
    if not parts:
        raise RuntimeError("No tail feature rows could be read")
    return pd.concat(parts, ignore_index=True, copy=False)


def _source_tags(scores: pd.Series, sides: pd.Series, edges: list[float]) -> pd.Series:
    internal = np.asarray([float(v) for v in edges[1:-1]], dtype=np.float64)
    values = pd.to_numeric(scores, errors="coerce").to_numpy(dtype=np.float64)
    bins = np.searchsorted(internal, values, side="right")
    intensity = np.full(len(scores), "model_candidate_background", dtype=object)
    intensity[bins == 7] = "model_frontier_top30"
    intensity[bins == 8] = "model_frontier_top20"
    intensity[bins >= 9] = "model_frontier_top10"
    return (
        sides.astype(str).str.lower().reset_index(drop=True)
        + "__"
        + pd.Series(intensity)
    )


def _fill_store_features(
    frame: pd.DataFrame,
    feature_root: Path,
    requested: list[str],
    *,
    prefer_existing_finite: bool = False,
) -> tuple[pd.DataFrame, dict[str, float]]:
    identity_columns = set(KEYS) | {
        "archetype_policy_key",
        "policy_archetype",
        "source_tag",
        "source_family",
    }
    names = list(
        dict.fromkeys(
            str(name)
            for name in requested
            if str(name) and str(name) not in identity_columns
        )
    )
    fallback = {
        name: pd.to_numeric(frame[name], errors="coerce").to_numpy(
            dtype=np.float32, copy=True
        )
        for name in names
        if name in frame.columns
    }
    stripped = frame.drop(columns=[name for name in names if name in frame.columns])
    loaded, coverage = _append_store_features(stripped, feature_root, names)
    for name, values in fallback.items():
        if name not in loaded.columns:
            loaded[name] = values
            continue
        current = pd.to_numeric(loaded[name], errors="coerce").to_numpy(
            dtype=np.float32, copy=True
        )
        existing_finite = np.isfinite(values)
        if prefer_existing_finite:
            current[existing_finite] = values[existing_finite]
        else:
            missing = ~np.isfinite(current)
            current[missing] = values[missing]
        loaded[name] = current
    return loaded, coverage


def _feature_store_timestamp(feature_root: Path) -> pd.Timestamp:
    """Parse the canonical static-store identity without host-local time."""
    return pd.to_datetime(feature_root.name, format="%Y%m%d_%H%M%S", utc=True)


def _hydrate_derived_meta_features(
    frame: pd.DataFrame,
    *,
    feature_root: Path,
    requested: Iterable[str],
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Fill selected, causal derived fields absent from older static partitions.

    This deliberately reads the same logical static store used by training and
    inference.  The only current derived feature is the frozen meta contract's
    carry-adjusted 10-hour return z-score.  It is reconstructed from persisted
    pre-entry ``ret10h`` and ``fund_rate`` using the registered 336-hour
    self-z-score definition, never from labels or outcomes.
    """
    needed = set(str(name) for name in requested)
    if "carry_adj_ret_self_z_10h" not in needed or frame.empty:
        return frame, {}
    output = frame.copy(deep=False)
    current = pd.to_numeric(
        output.get(
            "carry_adj_ret_self_z_10h",
            pd.Series(np.nan, index=output.index, dtype=np.float32),
        ),
        errors="coerce",
    ).to_numpy(dtype=np.float32, copy=True)
    missing = ~np.isfinite(current)
    if not missing.any():
        return output, {"carry_adj_ret_self_z_10h": 1.0}

    timestamps = _utc(output["__ts__"])
    symbols = sorted(output["__symbol__"].dropna().astype(str).unique())
    start = timestamps.min() - pd.Timedelta(days=15)
    end = timestamps.max()
    loaded = read_static_features(
        feature_store_ts=_feature_store_timestamp(feature_root),
        data_root=feature_root.parents[1],
        feature_keys=["ret10h", "fund_rate"],
        symbols=symbols,
        start_ts=start,
        end_ts=end,
        output_layout="panels",
    )
    if not loaded or "ret10h" not in loaded or "fund_rate" not in loaded:
        return output, {
            "carry_adj_ret_self_z_10h": float(np.isfinite(current).mean())
        }
    ret = loaded["ret10h"].reindex(columns=symbols).sort_index()
    funding = loaded["fund_rate"].reindex(index=ret.index, columns=ret.columns)
    raw = (
        ret.astype(np.float32, copy=False)
        - funding.astype(np.float32, copy=False) * np.float32(10.0 / 8.0)
    )
    # Match the registered feature implementation: the fused Numba transform
    # receives the complete timestamp x symbol panel and applies the causal
    # 336-hour z-score independently to every symbol column.
    panel = numba_rolling_zscore_fused(raw, 14 * 24).clip(-6.0, 6.0)
    row_index = pd.MultiIndex.from_arrays(
        [timestamps.to_numpy(), output["__symbol__"].astype(str).to_numpy()],
        names=["__ts__", "__symbol__"],
    )
    stacked = panel.rename_axis(index="__ts__", columns="__symbol__").stack(
        dropna=False
    )
    reconstructed = pd.to_numeric(
        stacked.reindex(row_index), errors="coerce"
    ).to_numpy(dtype=np.float32, copy=False)
    use = missing & np.isfinite(reconstructed)
    current[use] = reconstructed[use]
    output["carry_adj_ret_self_z_10h"] = current
    return output, {
        "carry_adj_ret_self_z_10h": float(np.isfinite(current).mean()),
        "carry_adj_ret_self_z_10h_reconstructed_rows": int(use.sum()),
    }


def _finite_complete_case_mask(
    frame: pd.DataFrame, columns: list[str]
) -> tuple[pd.Series, dict[str, int]]:
    """Return rows finite across the exact numeric model contract."""
    mask = pd.Series(True, index=frame.index, dtype=bool)
    missing_counts: dict[str, int] = {}
    for name in dict.fromkeys(str(column) for column in columns if str(column)):
        if name not in frame.columns:
            finite = np.zeros(len(frame), dtype=bool)
        else:
            finite = np.isfinite(
                pd.to_numeric(frame[name], errors="coerce").to_numpy(
                    dtype=np.float64, copy=False
                )
            )
        count = int((~finite).sum())
        if count:
            missing_counts[name] = count
            mask &= finite
    return mask, missing_counts


def _attrition_summary(
    frame: pd.DataFrame,
    mask: pd.Series,
    missing_counts: dict[str, int],
) -> dict[str, Any]:
    rejected = frame.loc[~mask]
    return {
        "input_rows": int(len(frame)),
        "accepted_rows": int(mask.sum()),
        "rejected_rows": int((~mask).sum()),
        "missing_feature_counts": dict(
            sorted(missing_counts.items(), key=lambda item: (-item[1], item[0]))
        ),
        "rejected_symbols": sorted(
            rejected.get("__symbol__", pd.Series(dtype=object))
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        ),
    }


def _hydrate_live_gated_inputs(
    frame: pd.DataFrame,
    *,
    data_root: Path,
    symbols: list[str],
    timestamps: pd.DatetimeIndex,
    required_columns: list[str],
) -> pd.DataFrame:
    """Recreate the gate-expanded inputs from the same raw panel as live.

    Gate-expanded columns are deliberately absent from the selected-feature
    sidecar.  Live inference derives them from the causal Kraken perpetual
    close panel, so a historical parity scorer must not substitute a similarly
    named cached return column.
    """

    gated_columns = [
        str(column)
        for column in required_columns
        if "_G_VOL_" in str(column) or "_G_TREND_" in str(column)
    ]
    if not gated_columns or frame.empty or timestamps.empty:
        return frame

    from extreme_price_movements.data_store import PartitionedOHLCVStore
    from extreme_price_movements.inference.feature_generator import (
        _synthesize_gated_feature_keys,
    )
    from scripts.replay_live_signal_predictions import _market_data_root

    market_root = _market_data_root(
        data_root,
        market_mode="perps",
        exchange_id="krakenfutures",
    )
    store = PartitionedOHLCVStore(str(market_root), timeframe="1h")
    start = pd.Timestamp(timestamps.min()) - pd.Timedelta(days=9)
    end = pd.Timestamp(timestamps.max())
    close_parts: list[pd.Series] = []
    for symbol in symbols:
        history = store.load(
            symbol,
            columns=["close"],
            start_ts=start,
            end_ts=end,
        )
        if history.empty or "close" not in history.columns:
            continue
        close_parts.append(
            pd.to_numeric(history["close"], errors="coerce").rename(symbol)
        )
    if not close_parts:
        return frame

    close = pd.concat(close_parts, axis=1, copy=False).sort_index()
    generated = _synthesize_gated_feature_keys(
        {},
        {"close": close},
        symbols,
        set(gated_columns),
    )
    row_index = pd.MultiIndex.from_arrays(
        [
            _utc(frame["__ts__"]),
            frame["__symbol__"].astype(str),
        ],
        names=["__ts__", "__symbol__"],
    )
    for column in gated_columns:
        values = generated.get(column)
        if not isinstance(values, pd.DataFrame) or values.empty:
            continue
        stacked = values.rename_axis(index="__ts__", columns="__symbol__").stack(
            dropna=False
        )
        frame[column] = pd.to_numeric(
            stacked.reindex(row_index), errors="coerce"
        ).to_numpy(dtype=np.float32, copy=False)
    return frame


def _policy_lookup(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for policy in (manifest.get("default") or {}).values():
        if isinstance(policy, dict) and policy.get("policy_key"):
            out[str(policy["policy_key"])] = dict(policy)
    for policy in manifest.get("overrides") or []:
        if isinstance(policy, dict) and policy.get("policy_key"):
            out[str(policy["policy_key"])] = dict(policy)
    return out


def _capture_for_policy_keys(
    rows: pd.DataFrame,
    *,
    side: str,
    policy_keys: pd.Series,
    policy_manifest: dict[str, Any],
    data_root: Path,
    path_len: int,
    min_path_coverage: float = 0.75,
    allow_partial_paths: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if rows.empty:
        return pd.DataFrame(index=rows.index), {
            "rows": 0,
            "finite_path_rows": 0,
            "finite_path_coverage": 1.0,
            "missing_paths": [],
        }
    rows = rows.reset_index(drop=True)
    policy_keys = policy_keys.reset_index(drop=True).astype(str)
    _, paths, stats = _fetch_policy_paths(
        rows,
        labels_path=Path("synthetic_july_tail.parquet"),
        side=side,
        data_root=data_root,
        market_mode="perps",
        exchange="krakenfutures",
        path_len=int(path_len),
        apply_delayed_entry=False,
    )
    finite_matrix = np.isfinite(paths[0]) & (paths[0] > 0.0)
    for path in paths[1:]:
        finite_matrix &= np.isfinite(path) & (path > 0.0)
    observed_bars = np.zeros(len(rows), dtype=np.int32)
    for idx, valid_bars in enumerate(finite_matrix):
        missing = np.flatnonzero(~valid_bars)
        observed_bars[idx] = int(missing[0]) if len(missing) else int(len(valid_bars))
    finite_path = observed_bars >= int(path_len)
    entry_path = observed_bars > 0
    capture_paths = paths
    if allow_partial_paths:
        filled = [np.asarray(path, dtype=np.float32).copy() for path in paths]
        for idx, count in enumerate(observed_bars):
            if count <= 0 or count >= int(path_len):
                continue
            last_close = float(filled[3][idx, count - 1])
            for path in filled:
                path[idx, count:] = last_close
        capture_paths = tuple(filled)
    missing_columns = [name for name in KEYS if name in rows.columns]
    stats["missing_paths"] = (
        rows.loc[~finite_path, missing_columns].astype(str).to_dict(orient="records")
    )
    stats["entry_path_rows"] = int(entry_path.sum())
    stats["entry_path_coverage"] = float(entry_path.mean()) if len(entry_path) else 0.0
    # Prediction coverage must be complete. Outcome repair also includes rows
    # already known to lack an executable Kraken chart, so require a diagnostic
    # floor while retaining every missing path explicitly in the manifest.
    coverage_key = (
        "entry_path_coverage" if allow_partial_paths else "finite_path_coverage"
    )
    if float(stats.get(coverage_key, 0.0)) < float(min_path_coverage):
        raise RuntimeError(f"Synthetic tail path coverage is too low: {stats}")
    policies = _policy_lookup(policy_manifest)
    captures: dict[str, pd.DataFrame] = {}
    for key in sorted(policy_keys.unique()):
        policy = policies.get(key)
        if policy is None:
            raise KeyError(f"No S59 geometry found for policy key {key!r}")
        captures[key] = _first_touch_capture_outcome(
            rows,
            capture_paths,
            _arm_from_policy(policy),
            side_name=side,
            outcome_mode="trailing_profit",
            round_trip_cost=0.01,
        )
    output = pd.DataFrame(
        index=rows.index, columns=next(iter(captures.values())).columns
    )
    for key, capture in captures.items():
        mask = policy_keys.eq(key)
        output.loc[mask, :] = capture.loc[mask, :]
    first_bar = pd.to_numeric(output["first_touch_bar"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    event_exit = (
        pd.to_numeric(output["capture_hit"], errors="coerce").fillna(0.0).to_numpy()
        > 0.5
    ) | (
        pd.to_numeric(output["capture_stop"], errors="coerce").fillna(0.0).to_numpy()
        > 0.5
    )
    outcome_resolved = (
        entry_path & event_exit & np.isfinite(first_bar) & (first_bar <= observed_bars)
    )
    outcome_resolved |= finite_path & (
        pd.to_numeric(output["capture_timeout"], errors="coerce").fillna(0.0).to_numpy()
        > 0.5
    )
    output["capture_observed_bars"] = observed_bars
    output["capture_horizon_complete"] = finite_path.astype(np.float32)
    output["capture_outcome_resolved"] = outcome_resolved.astype(np.float32)
    output["capture_valid_path"] = entry_path.astype(np.float32)
    stats["resolved_outcome_rows"] = int(outcome_resolved.sum())
    stats["resolved_outcome_coverage"] = (
        float(outcome_resolved.mean()) if len(outcome_resolved) else 0.0
    )
    return output.infer_objects(copy=False), stats


def _capture_outcomes(capture: pd.DataFrame) -> pd.DataFrame:
    net = pd.to_numeric(capture["capture_net"], errors="coerce")
    validity_column = (
        "capture_outcome_resolved"
        if "capture_outcome_resolved" in capture.columns
        else "capture_valid_path"
    )
    valid = pd.to_numeric(capture[validity_column], errors="coerce").gt(0.5)
    first_mae = pd.to_numeric(capture["first_touch_mae_norm"], errors="coerce")
    full_mae = pd.to_numeric(capture["full_path_mae_norm"], errors="coerce")
    timeout = pd.to_numeric(capture["capture_timeout"], errors="coerce").fillna(0.0)
    mfe_first = pd.to_numeric(capture["mfe_1r_before_mae_1r"], errors="coerce").fillna(
        0.0
    )
    mae_first = pd.to_numeric(capture["mae_1r_before_mfe_1r"], errors="coerce").fillna(
        0.0
    )
    out = pd.DataFrame(index=capture.index)
    out["exec_margin"] = net.where(valid)
    # capture_net is already net of the round-trip cost supplied to the
    # first-touch simulator. Do not subtract the same fee a second time.
    out["ev_after_1pct"] = net.where(valid)
    out["first_touch_bad_mae_1r"] = first_mae.ge(1.0).astype(np.float32).where(valid)
    out["full_path_bad_mae_1r"] = full_mae.ge(1.0).astype(np.float32).where(valid)
    out["timeout"] = timeout.gt(0.5).astype(np.float32).where(valid)
    out["clean_exec"] = (
        (net.gt(0.0) & first_mae.lt(1.0) & timeout.lt(0.5) & mfe_first.gt(0.5))
        .astype(np.float32)
        .where(valid)
    )
    out["dirty_positive"] = (
        (
            net.gt(0.0)
            & (
                first_mae.ge(1.0)
                | full_mae.ge(1.0)
                | timeout.gt(0.5)
                | mae_first.gt(0.5)
            )
        )
        .astype(np.float32)
        .where(valid)
    )
    return out


def _assign_frozen_policy_archetypes(
    frame: pd.DataFrame,
    *,
    model_input: pd.DataFrame | None,
    classifier: dict[str, Any],
) -> tuple[pd.DataFrame, pd.Series]:
    """Assign live-predictable policy archetypes with frozen side fallbacks."""

    out = frame.copy()
    if "__archetype_policy_key__" not in out:
        out["__archetype_policy_key__"] = ""
    else:
        out["__archetype_policy_key__"] = (
            out["__archetype_policy_key__"].fillna("").astype(str)
        )
    side_defaults = dict(classifier.get("side_defaults") or {})
    assigned_by = pd.Series("existing", index=out.index, dtype=object)
    needs_assignment = out["__archetype_policy_key__"].eq("")
    for idx in out.index[needs_assignment]:
        side = str(out.at[idx, "side_name"]).lower()
        candidate_row = out.loc[[idx]]
        input_row = (
            model_input.loc[[idx]]
            if isinstance(model_input, pd.DataFrame) and idx in model_input.index
            else None
        )
        predicted = predict_observable_policy_archetype(
            side=side,
            candidate_feature_row=candidate_row,
            meta_model_input_row=input_row,
        )
        source = "observable_regime"
        if not predicted:
            predicted = predict_live_policy_archetype(
                side=side,
                payload=classifier,
                candidate_feature_row=candidate_row,
                meta_model_input_row=input_row,
            )
            source = "frozen_classifier"
        if not predicted:
            predicted = str(side_defaults.get(side, ""))
            source = "frozen_side_default"
        prefix = f"{side}__"
        out.at[idx, "__archetype_policy_key__"] = (
            predicted[len(prefix) :] if predicted.startswith(prefix) else predicted
        )
        assigned_by.at[idx] = source
    complete = out["__archetype_policy_key__"].astype(str).str.len().gt(0)
    out["archetype_policy_key"] = out["__archetype_policy_key__"].astype(str)
    out["policy_archetype_assignment_source"] = assigned_by.astype(str)
    return out, complete


def _hourly_close_proxy_outcomes(
    rows: pd.DataFrame,
    *,
    feature_root: Path,
    policy_manifest: dict[str, Any],
    horizon_hours: int,
    policy_bar_minutes: int = 15,
    round_trip_cost: float = 0.01,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Replay side/archetype geometry on canonical hourly close paths.

    This is deliberately a diagnostic proxy. It uses future hourly OHLCV closes,
    never inference features, and cannot prove intrabar execution parity.  Do not
    use the model feature named ``ret1h`` here: that key is fractionally
    differenced in the current feature contract and is not a tradable return.
    """

    output = pd.DataFrame(index=rows.index)
    if rows.empty:
        return output, {"rows": 0, "resolved_rows": 0, "coverage": 1.0}
    horizon = max(1, int(horizon_hours))
    start = pd.Timestamp(rows["__ts__"].min())
    end = pd.Timestamp(rows["__ts__"].max()) + pd.Timedelta(hours=horizon)
    symbols = sorted(rows["__symbol__"].astype(str).unique())
    from extreme_price_movements.data_store import PartitionedOHLCVStore
    from scripts.replay_live_signal_predictions import _market_data_root

    market_root = _market_data_root(
        feature_root.parents[1],
        market_mode="perps",
        exchange_id="krakenfutures",
    )
    store = PartitionedOHLCVStore(str(market_root), timeframe="1h")
    closes: dict[str, pd.Series] = {}
    for symbol in symbols:
        history = store.load(
            symbol,
            columns=["close"],
            start_ts=start,
            end_ts=end,
        )
        if history.empty or "close" not in history.columns:
            continue
        series = pd.to_numeric(history["close"], errors="coerce")
        series.index = pd.DatetimeIndex(pd.to_datetime(series.index, utc=True))
        closes[symbol] = series.loc[~series.index.duplicated(keep="last")]
    policies = _policy_lookup(policy_manifest)

    net = np.full(len(rows), np.nan, dtype=np.float32)
    first_bad = np.full(len(rows), np.nan, dtype=np.float32)
    full_bad = np.full(len(rows), np.nan, dtype=np.float32)
    timeout = np.full(len(rows), np.nan, dtype=np.float32)
    clean = np.full(len(rows), np.nan, dtype=np.float32)
    dirty = np.full(len(rows), np.nan, dtype=np.float32)
    historically_tradable = np.zeros(len(rows), dtype=bool)
    resolved = np.zeros(len(rows), dtype=bool)
    entry_timestamps = _utc(rows["__ts__"]).to_numpy(copy=False)
    row_symbols = rows["__symbol__"].astype(str).to_numpy(copy=False)
    row_sides = rows["side_name"].astype(str).str.lower().to_numpy(copy=False)
    row_keys = rows["archetype_policy_key"].astype(str).to_numpy(copy=False)
    barriers = pd.to_numeric(rows["__barrier_pct__"], errors="coerce").to_numpy(
        dtype=np.float64, copy=False
    )
    for pos in range(len(rows)):
        entry_ts = pd.Timestamp(entry_timestamps[pos])
        symbol = str(row_symbols[pos])
        side = str(row_sides[pos])
        key = str(row_keys[pos])
        policy = policies.get(key)
        if policy is None or symbol not in closes:
            continue
        history = closes[symbol]
        path_end = entry_ts + pd.Timedelta(hours=horizon)
        if (
            history.empty
            or entry_ts < history.index.min()
            or path_end > history.index.max()
        ):
            # The frozen model can score symbols before their Kraken futures
            # listing. Those rows are not executable-path gaps and must not
            # dilute path coverage for the historically tradable universe.
            continue
        historically_tradable[pos] = True
        entry_close = float(closes[symbol].get(entry_ts, np.nan))
        path_index = pd.date_range(
            entry_ts + pd.Timedelta(hours=1), periods=horizon, freq="h"
        )
        future_close = pd.to_numeric(
            closes[symbol].reindex(path_index), errors="coerce"
        ).to_numpy(dtype=np.float64, copy=False)
        if (
            not np.isfinite(entry_close)
            or entry_close <= 0.0
            or len(future_close) != horizon
            or not np.isfinite(future_close).all()
        ):
            continue
        price_return = future_close / entry_close - 1.0
        directional = price_return if side == "long" else -price_return
        barrier = float(barriers[pos])
        activation = max(0.0, float(policy.get("tp_r", 0.0)) * barrier)
        stop_level = max(0.0, float(policy.get("sl_r", 1.0)) * barrier)
        trail_gap = max(0.0, float(policy.get("trail_r", 0.0)) * barrier)
        max_activation_hours = max(
            1,
            int(
                np.ceil(
                    float(policy.get("max_bars_to_mfe", horizon))
                    * max(int(policy_bar_minutes), 1)
                    / 60.0
                )
            ),
        )
        peak = -np.inf
        armed = False
        exit_value = float(directional[-1])
        timed_out = True
        for step, value in enumerate(directional, start=1):
            value = float(value)
            peak = max(peak, value)
            if value <= -stop_level:
                exit_value = -stop_level
                timed_out = False
                break
            armed = armed or (
                step <= max_activation_hours and value >= activation
            )
            if armed and trail_gap > 0.0 and (peak - value) >= trail_gap:
                exit_value = value
                timed_out = False
                break
        max_adverse = max(0.0, -float(np.min(directional)))
        max_favorable = max(0.0, float(np.max(directional)))
        value_net = exit_value - float(round_trip_cost)
        net[pos] = np.float32(value_net)
        first_bad[pos] = np.float32(max_adverse >= barrier)
        full_bad[pos] = np.float32(max_adverse >= barrier)
        timeout[pos] = np.float32(timed_out)
        clean[pos] = np.float32(
            value_net > 0.0
            and max_adverse < barrier
            and not timed_out
            and max_favorable >= 0.5 * barrier
        )
        dirty[pos] = np.float32(
            value_net > 0.0 and (max_adverse >= barrier or timed_out)
        )
        resolved[pos] = True
    output["exec_margin"] = net
    output["ev_after_1pct"] = net
    output["first_touch_bad_mae_1r"] = first_bad
    output["full_path_bad_mae_1r"] = full_bad
    output["timeout"] = timeout
    output["clean_exec"] = clean
    output["dirty_positive"] = dirty
    tradable_rows = int(historically_tradable.sum())
    resolved_rows = int(resolved.sum())
    return output, {
        "source": "canonical_krakenfutures_hourly_close_path",
        "timeframe": "1h",
        "intrabar_high_low_available": False,
        "execution_parity_claim": False,
        "horizon_hours": horizon,
        "policy_bar_minutes": int(policy_bar_minutes),
        "activation_deadline_contract": "ceil(max_bars_to_mfe * policy_bar_minutes / 60)",
        "outcome_contract_version": "hourly_close_policy_proxy_v2_activation_deadline",
        "round_trip_cost": float(round_trip_cost),
        "rows": int(len(rows)),
        "raw_candidate_rows": int(len(rows)),
        "historically_tradable_rows": tradable_rows,
        "unavailable_contract_rows": int(len(rows) - tradable_rows),
        "resolved_rows": resolved_rows,
        "internal_gap_rows": int(tradable_rows - resolved_rows),
        "coverage_denominator": "historically_tradable_rows",
        "coverage": (
            float(resolved_rows / tradable_rows) if tradable_rows else 1.0
        ),
        "raw_universe_coverage": (
            float(tradable_rows / len(rows)) if len(rows) else 1.0
        ),
    }


def _run_base_only_backcast(
    valid: pd.DataFrame,
    *,
    args: argparse.Namespace,
    classifier: dict[str, Any],
    complete_case_attrition: dict[str, Any],
    symbols: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> int:
    """Persist a diagnostic top-k base backcast when meta inputs do not exist."""

    valid, archetype_complete = _assign_frozen_policy_archetypes(
        valid,
        model_input=None,
        classifier=classifier,
    )
    complete_case_attrition["policy_archetype"] = _attrition_summary(
        valid,
        archetype_complete,
        {"__archetype_policy_key__": int((~archetype_complete).sum())},
    )
    valid = valid.loc[archetype_complete].copy()
    if valid.empty:
        raise RuntimeError("No base rows received a frozen policy archetype")

    keep_ratio = min(
        1.0,
        float(args.backcast_admission_frac) / max(float(args.base_top_frac), 1e-9),
    )
    selected: list[int] = []
    for _, batch in valid.groupby(["__ts__", "side_name"], sort=True):
        scores = pd.to_numeric(batch["score"], errors="coerce").dropna()
        top_n = max(1, int(np.ceil(len(scores) * keep_ratio)))
        selected.extend(scores.nlargest(top_n).index.tolist())
    selected_index = pd.Index(selected)
    valid["selected_for_monitor"] = valid.index.isin(selected_index)
    valid["historical_rank"] = valid.groupby(["__ts__", "side_name"], observed=True)[
        "score"
    ].rank(method="average", pct=True)
    valid["hit_probability"] = valid["historical_rank"].astype(np.float32)
    valid["base_score"] = pd.to_numeric(valid["score"], errors="coerce").astype(
        np.float32
    )
    valid["score_meta_base_soft_label"] = np.nan

    policy_manifest = _load_json(args.policy_manifest)
    path_stats: dict[str, Any] = {}
    outcome_parts: list[pd.DataFrame] = []
    for side in ("long", "short"):
        side_rows = valid.loc[valid["side_name"].eq(side)].copy()
        if side_rows.empty:
            continue
        outcome_source = str(args.backcast_outcome_source)
        outcomes: pd.DataFrame
        if outcome_source in {"auto", "execution_1m"}:
            try:
                capture, stats = _capture_for_policy_keys(
                    side_rows.reset_index(drop=True),
                    side=side,
                    policy_keys=side_rows["archetype_policy_key"],
                    policy_manifest=policy_manifest,
                    data_root=Path("data_perp"),
                    path_len=int(args.path_len),
                    allow_partial_paths=False,
                )
                outcomes = _capture_outcomes(capture)
                path_stats[side] = stats
            except RuntimeError:
                if outcome_source != "auto":
                    raise
                outcomes, stats = _hourly_close_proxy_outcomes(
                    side_rows.reset_index(drop=True),
                    feature_root=args.feature_root,
                    policy_manifest=policy_manifest,
                    horizon_hours=int(args.backcast_proxy_horizon_hours),
                    policy_bar_minutes=int(args.backcast_policy_bar_minutes),
                    round_trip_cost=0.01,
                )
                stats["fallback_reason"] = "execution_1m_path_unavailable"
                path_stats[side] = stats
        else:
            outcomes, stats = _hourly_close_proxy_outcomes(
                side_rows.reset_index(drop=True),
                feature_root=args.feature_root,
                policy_manifest=policy_manifest,
                horizon_hours=int(args.backcast_proxy_horizon_hours),
                policy_bar_minutes=int(args.backcast_policy_bar_minutes),
                round_trip_cost=0.01,
            )
            path_stats[side] = stats
        outcomes.index = side_rows.index
        outcome_parts.append(outcomes)
    if not outcome_parts:
        raise RuntimeError("No base backcast outcomes were replayed")
    outcomes = pd.concat(outcome_parts).sort_index()
    for name in outcomes:
        valid.loc[outcomes.index, name] = outcomes[name]
    valid = valid.loc[pd.to_numeric(valid["ev_after_1pct"], errors="coerce").notna()]
    valid["evidence_scope"] = "frozen_backcast_diagnostic"
    valid["prediction_evidence"] = "frozen_base_historical_backcast_not_oos"
    valid["selected_top30"] = True
    valid["threshold_basis_selected"] = valid["selected_for_monitor"].astype(bool)

    columns = [
        *KEYS,
        "__barrier_pct__",
        "archetype_policy_key",
        "policy_archetype_assignment_source",
        "base_score",
        "score_meta_base_soft_label",
        "historical_rank",
        "hit_probability",
        "ev_after_1pct",
        "exec_margin",
        "clean_exec",
        "dirty_positive",
        "first_touch_bad_mae_1r",
        "full_path_bad_mae_1r",
        "timeout",
        "selected_top30",
        "selected_for_monitor",
        "threshold_basis_selected",
        "evidence_scope",
        "prediction_evidence",
    ]
    persisted_observable: list[str] = []
    if bool(args.backcast_include_observable_features):
        from extreme_price_movements.unsupervised_regime_learning.feature_registry import (
            UNSUPERVISED_REGIME_PRIMITIVE_FEATURES,
        )

        observable = [
            str(name)
            for name in UNSUPERVISED_REGIME_PRIMITIVE_FEATURES
            if str(name) in valid.columns
        ]
        observable.extend(
            name
            for name in valid.columns
            if name.startswith(
                (
                    "gmm_",
                    "dae_",
                    "AE_",
                    "mahalanobis_",
                    "cluster_",
                    "base_attr_",
                    "__regime_source_",
                )
            )
        )
        persisted_observable = list(dict.fromkeys(observable))
        columns.extend(persisted_observable)
    complete = valid.reindex(columns=columns).sort_values(KEYS, kind="stable")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / str(args.output_filename)
    complete.to_parquet(output_path, index=False, compression="zstd")
    manifest = {
        "schema": "frozen_base_failure_backcast_v1",
        "start": start.isoformat(),
        "end_exclusive": end.isoformat(),
        "evidence_scope": "frozen_backcast_diagnostic",
        "model_stage": "base_only_due_to_historical_meta_feature_unavailability",
        "base_top_frac": float(args.base_top_frac),
        "admission_frac": float(args.backcast_admission_frac),
        "symbol_count": int(len(symbols)),
        "rows": int(len(complete)),
        "selected_for_monitor_rows": int(
            complete["selected_for_monitor"].fillna(False).astype(bool).sum()
        ),
        "days": int(complete["__ts__"].dt.floor("D").nunique()),
        "negative_row_ev_days": int(
            complete.assign(day=complete["__ts__"].dt.floor("D"))
            .groupby("day", observed=True)["ev_after_1pct"]
            .sum()
            .lt(0.0)
            .sum()
        ),
        "round_trip_cost": 0.01,
        "return_unit": "decimal_notional_return",
        "cost_counted_once": True,
        "selected_top30_rows": int(
            complete["selected_top30"].fillna(False).astype(bool).sum()
        ),
        "outcome_source_requested": str(args.backcast_outcome_source),
        "outcome_contract_version": "hourly_close_policy_proxy_v2_activation_deadline",
        "policy_bar_minutes": int(args.backcast_policy_bar_minutes),
        "proxy_horizon_hours": int(args.backcast_proxy_horizon_hours),
        "execution_parity_claim": bool(
            path_stats
            and all(
                bool(stats.get("execution_parity_claim", True))
                for stats in path_stats.values()
            )
        ),
        "observable_feature_columns": int(len(persisted_observable)),
        "observable_feature_names": persisted_observable,
        "base_attribution_feature_columns": int(
            sum(name.startswith("base_attr_") for name in persisted_observable)
        ),
        "base_attribution_feature_names": [
            name for name in persisted_observable if name.startswith("base_attr_")
        ],
        "path_stats": path_stats,
        "complete_case_attrition": complete_case_attrition,
        "input_contract": {
            "feature_root": str(args.feature_root),
            "base_reference": str(args.base_reference),
            "model_artifact_root": str(args.model_artifact_root)
            if args.model_artifact_root is not None
            else None,
            "ae_gmm_state": str(args.ae_gmm_state),
            "meta_handoff_dir": str(args.meta_handoff_dir),
            "residual_bundle": str(args.residual_bundle),
            "residual_train_reference": str(args.residual_train_reference),
            "policy_manifest": str(args.policy_manifest),
            "source_manifest": str(args.source_manifest),
            "native_run_id": str(args.native_run_id),
        },
        "output": str(output_path),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, default=str))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=None,
        help=(
            "Optional monthly label directory. When omitted, score every "
            "requested feature-store timestamp and replay outcomes from paths."
        ),
    )
    parser.add_argument("--old-labels-dir", type=Path, default=None)
    parser.add_argument("--base-reference", type=Path, required=True)
    parser.add_argument(
        "--base-top-frac",
        type=float,
        default=0.30,
        help=(
            "Per-timestamp, per-side base handoff fraction used by packaged "
            "production artifacts. Legacy exported-model mode retains its "
            "historical fixed-cutoff behavior."
        ),
    )
    parser.add_argument("--base-model-dir", type=Path, default=None)
    parser.add_argument("--meta-model-dir", type=Path, default=None)
    parser.add_argument(
        "--model-artifact-root",
        type=Path,
        default=None,
        help=(
            "Load the packaged production base/meta models and feature contracts "
            "directly from one inference artifact. This takes precedence over "
            "--base-model-dir/--meta-model-dir and preserves side-specific models."
        ),
    )
    parser.add_argument(
        "--meta-score-alignment",
        type=Path,
        default=None,
        help=(
            "Optional frozen side-specific bridge from a final-refit score "
            "domain into the OOF champion domain used by V9/MLP."
        ),
    )
    parser.add_argument(
        "--disable-meta-score-alignment",
        action="store_true",
        help=(
            "Keep packaged meta scores in their native domain. This is required "
            "for OOS checkpoints whose package manifest declares an identity "
            "score-alignment contract."
        ),
    )
    parser.add_argument("--ae-gmm-state", type=Path, required=True)
    parser.add_argument("--meta-handoff-dir", type=Path, required=True)
    parser.add_argument("--residual-bundle", type=Path, required=True)
    parser.add_argument(
        "--residual-train-reference",
        type=Path,
        required=True,
        help=(
            "Frozen compact training reference used to reproduce the residual "
            "bundle's causal reliability, surprise, and outcome-context inputs."
        ),
    )
    parser.add_argument("--native-run-id", required=True)
    parser.add_argument("--policy-manifest", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument(
        "--threshold-policy",
        type=Path,
        default=None,
        help="Optional frozen causal admission policy applied to scored rows.",
    )
    parser.add_argument(
        "--residual-event-state",
        type=Path,
        default=None,
        help="Frozen residual-event state used to materialize V9/MLP inputs.",
    )
    parser.add_argument(
        "--regime-ev-calibration",
        type=Path,
        default=None,
        help="Optional frozen V9+MLP hierarchical-EV calibration artifact.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--allowed-symbols-matrix",
        type=Path,
        default=None,
        help=(
            "Optional repaired live feature matrix whose symbol index defines "
            "the spread-eligible scoring universe."
        ),
    )
    parser.add_argument(
        "--allowed-symbols-report",
        type=Path,
        default=None,
        help=(
            "Optional live source-parity JSON. Its accepted_symbols list is "
            "applied after the matrix universe so stale-source exclusions match live."
        ),
    )
    parser.add_argument("--start", default="2026-07-08T00:00:00Z")
    parser.add_argument("--end-exclusive", default="2026-07-11T00:00:00Z")
    parser.add_argument(
        "--default-barrier-pct",
        type=float,
        default=0.02,
        help="Fallback label barrier used only when no causal label history exists.",
    )
    parser.add_argument(
        "--evidence-scope",
        choices=("oof_oos", "frozen_backcast_diagnostic"),
        default="oof_oos",
    )
    parser.add_argument(
        "--output-filename",
        default="frozen_predictions.parquet",
    )
    parser.add_argument(
        "--symbol-limit",
        type=int,
        default=0,
        help="Deterministic alphabetic symbol cap for smoke tests; zero means all.",
    )
    parser.add_argument(
        "--base-only-backcast",
        action="store_true",
        help=(
            "Run a frozen base-stage top-k diagnostic backcast. This is the "
            "honest fallback when the frozen meta contract has no historical "
            "coverage for one or more selected inputs."
        ),
    )
    parser.add_argument(
        "--backcast-admission-frac",
        type=float,
        default=0.10,
        help="Full-universe base fraction replayed in base-only backcast mode.",
    )
    parser.add_argument(
        "--backcast-outcome-source",
        choices=("auto", "execution_1m", "hourly_close_proxy"),
        default="auto",
        help=(
            "Outcome source for diagnostic historical base backcasts. auto uses "
            "the canonical 1-minute execution store when available and otherwise "
            "falls back to a cost-aware hourly-close proxy."
        ),
    )
    parser.add_argument(
        "--backcast-proxy-horizon-hours",
        type=int,
        default=24,
        help="Forward horizon for the explicitly diagnostic hourly-close proxy.",
    )
    parser.add_argument(
        "--backcast-policy-bar-minutes",
        type=int,
        default=15,
        help="Native policy-geometry bar duration used by max_bars_to_mfe.",
    )
    parser.add_argument(
        "--backcast-include-observable-features",
        action="store_true",
        help=(
            "Persist the broad observable regime basket and frozen AE/GMM outputs "
            "alongside diagnostic backcast rows for failure-state discovery."
        ),
    )
    parser.add_argument("--path-len", type=int, default=96)
    parser.add_argument(
        "--rescore-observed",
        action="store_true",
        help=(
            "Rebuild every requested timestamp from the frozen feature store, "
            "including timestamps already present in the label artifact. This "
            "is intended for replay/live parity audits; the default remains an "
            "incremental missing-timestamp backfill."
        ),
    )
    parser.add_argument(
        "--context-warmup-hours",
        type=int,
        default=336,
        help=(
            "Causal feature/AE-GMM context retained before --start. These rows "
            "are scored only to initialize temporal state outputs and are never "
            "written to the requested output window."
        ),
    )
    parser.add_argument(
        "--persist-meta-input",
        action="store_true",
        help="Persist the exact ordered meta-model matrix for parity diagnosis.",
    )
    parser.add_argument(
        "--allow-unresolved-outcomes",
        action="store_true",
        help=(
            "Allow prediction-parity scoring before the full future execution "
            "path exists. Prediction and policy columns are still materialized; "
            "unresolved outcome metrics remain diagnostic-only."
        ),
    )
    parser.add_argument(
        "--stop-after-base",
        action="store_true",
        help="Write base scores/model inputs and stop before meta materialization.",
    )
    parser.add_argument(
        "--stop-after-meta",
        action="store_true",
        help="Write meta scores/model inputs and stop before postprocessing/replay.",
    )
    parser.add_argument(
        "--compact-meta-output",
        action="store_true",
        help=(
            "With --stop-after-meta, persist the compact candidate ledger rather "
            "than the full ordered base/meta model matrices. The compact output "
            "keeps frozen scores, observable policy archetypes, and AE/GMM state "
            "outputs for historical diagnostics."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.environ.setdefault("EPM_SIMPLE_POLICY_15M_DOWNLOAD", "1")
    os.environ.setdefault("EPM_SIMPLE_POLICY_15M_CHART_ONLY", "1")
    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end_exclusive)
    if start.tzinfo is None or end.tzinfo is None:
        raise ValueError("Backcast boundaries must be explicit UTC timestamps")
    start = start.tz_convert("UTC")
    end = end.tz_convert("UTC")
    if end <= start:
        raise ValueError("--end-exclusive must be after --start")
    if int(args.context_warmup_hours) < 0:
        raise ValueError("--context-warmup-hours must be non-negative")
    if args.labels_dir is None and args.evidence_scope != "frozen_backcast_diagnostic":
        raise ValueError(
            "Label-free scoring must be explicitly marked "
            "--evidence-scope frozen_backcast_diagnostic"
        )

    packaged_model_root = args.model_artifact_root
    base_models_by_side: dict[str, Any] = {}
    meta_models_by_side: dict[str, Any] = {}
    if packaged_model_root is not None:
        trained_state_path = packaged_model_root / "models" / "trained_state.pkl"
        meta_feature_contract_path = (
            packaged_model_root / "meta_oof" / "meta_feature_contract.json"
        )
        if not trained_state_path.exists():
            raise FileNotFoundError(trained_state_path)
        if not meta_feature_contract_path.exists():
            raise FileNotFoundError(meta_feature_contract_path)
        with trained_state_path.open("rb") as handle:
            packaged_state = pickle.load(handle)
        packaged_bundle = (
            packaged_state.get("bundle", packaged_state)
            if isinstance(packaged_state, dict)
            else {}
        )
        packaged_alpha = packaged_bundle.get("alpha_models", {})
        if not isinstance(packaged_alpha, dict):
            raise RuntimeError("Packaged model artifact has no alpha_models mapping")
        base_columns_by_side: dict[str, list[str]] = {}
        for side in ("long", "short"):
            strategy_key = f"{side}_s52_meta_threshold_handoff"
            model_info = packaged_alpha.get(strategy_key)
            if not isinstance(model_info, dict) or model_info.get("model") is None:
                raise RuntimeError(
                    f"Packaged model artifact is missing {strategy_key} base model"
                )
            columns = [str(name) for name in model_info.get("feat_cols", [])]
            if not columns:
                raise RuntimeError(f"Packaged {strategy_key} has no feature contract")
            base_models_by_side[side] = model_info["model"]
            base_columns_by_side[side] = columns
        base_columns = list(
            dict.fromkeys(
                name
                for side in ("long", "short")
                for name in base_columns_by_side[side]
            )
        )
        base_contract = {
            "feature_names": base_columns,
            "feature_names_by_side": base_columns_by_side,
            "feature_contract_hash": (
                _load_json(packaged_model_root / "manifest.json")
                .get("feature_contract", {})
                .get("base_feature_hash")
            ),
        }
        packaged_meta_contract = _load_json(meta_feature_contract_path)
        model_contracts = packaged_meta_contract.get("meta_models", {}) or {}
        meta_contract = {
            "feature_names_by_model": {
                f"base_soft_label_{side}": list(
                    model_contracts[f"{side}_s52_meta_threshold_handoff"][
                        "feature_columns"
                    ]
                )
                for side in ("long", "short")
            },
            "feature_contract_hash": next(
                (
                    row.get("feature_contract_hash")
                    for row in model_contracts.values()
                    if isinstance(row, dict) and row.get("feature_contract_hash")
                ),
                None,
            ),
        }
        loaded_meta_models = load_meta_models_from_pickle(str(trained_state_path))
        for side in ("long", "short"):
            key = f"{side}_s52_meta_threshold_handoff"
            model = loaded_meta_models.get(key)
            if model is None:
                raise RuntimeError(
                    f"Packaged model artifact is missing meta model {key}"
                )
            meta_models_by_side[side] = model
    else:
        if args.base_model_dir is None or args.meta_model_dir is None:
            raise ValueError(
                "Provide --model-artifact-root or both --base-model-dir and "
                "--meta-model-dir"
            )
        base_contract = _load_json(args.base_model_dir / "columns.json")
        meta_contract = _load_json(args.meta_model_dir / "columns.json")
        base_columns = list(base_contract["feature_names"])
    meta_columns_by_model = {
        str(label): list(columns)
        for label, columns in (
            meta_contract.get("feature_names_by_model", {}) or {}
        ).items()
    }
    meta_columns = list(
        dict.fromkeys(
            column for columns in meta_columns_by_model.values() for column in columns
        )
    ) or list(meta_contract["feature_names"])
    meta_ood = [name for name in meta_columns if name.startswith(OOD_PREFIX)]
    meta_pre_ood = [name for name in meta_columns if name not in meta_ood]
    packaged_manifest = (
        _load_json(packaged_model_root / "manifest.json")
        if packaged_model_root is not None
        else {}
    )
    packaged_alignment_mode = str(
        (packaged_manifest.get("meta_model_override") or {}).get("score_alignment", "")
    ).strip()
    identity_score_domain = packaged_alignment_mode.startswith("identity_")
    meta_score_alignment = None
    if args.meta_score_alignment is not None and not args.disable_meta_score_alignment:
        meta_score_alignment = _load_json(args.meta_score_alignment)
    if (
        meta_score_alignment is None
        and packaged_model_root is not None
        and not args.disable_meta_score_alignment
        and not identity_score_domain
    ):
        packaged_alignment = (
            packaged_model_root / "policy_params" / "s52_meta_score_alignment.json"
        )
        if packaged_alignment.exists():
            meta_score_alignment = _load_json(packaged_alignment)
    with args.ae_gmm_state.open("rb") as handle:
        ae_gmm_state = pickle.load(handle)

    labels = _load_optional_label_period(
        args.labels_dir,
        start=start,
        end=end,
    )
    allowed_symbols: set[str] | None = None
    allowed_matrix_feature_columns: list[str] = []
    allowed_matrix_values: pd.DataFrame | None = None
    if args.allowed_symbols_matrix is not None:
        allowed_matrix = pd.read_parquet(args.allowed_symbols_matrix)
        allowed_matrix_feature_columns = [
            str(column) for column in allowed_matrix.columns if str(column)
        ]
        if "symbol" in allowed_matrix.columns:
            allowed_symbols = set(allowed_matrix["symbol"].dropna().astype(str))
        else:
            allowed_symbols = set(allowed_matrix.index.dropna().astype(str))
        allowed_matrix_values = allowed_matrix.copy()
        allowed_matrix_values["__symbol__"] = allowed_matrix_values.get(
            "symbol", allowed_matrix_values.index
        ).astype(str)
        allowed_matrix_values = allowed_matrix_values.set_index("__symbol__", drop=True)
        if not labels.empty:
            labels = labels.loc[
                labels["__symbol__"].astype(str).isin(allowed_symbols)
            ].reset_index(drop=True)
        if args.labels_dir is not None and labels.empty:
            raise RuntimeError(
                "Allowed-symbol matrix has no overlap with the supplied labels"
            )
    if args.allowed_symbols_report is not None:
        source_parity = _load_json(args.allowed_symbols_report)
        source_accepted = {
            str(symbol)
            for symbol in source_parity.get("accepted_symbols", [])
            if str(symbol)
        }
        if not source_accepted:
            raise RuntimeError("Allowed-symbol report has no accepted_symbols entries")
        allowed_symbols = (
            source_accepted
            if allowed_symbols is None
            else allowed_symbols.intersection(source_accepted)
        )
        if not labels.empty:
            labels = labels.loc[
                labels["__symbol__"].astype(str).isin(allowed_symbols)
            ].reset_index(drop=True)
        if args.labels_dir is not None and labels.empty:
            raise RuntimeError(
                "Allowed-symbol source report has no overlap with supplied labels"
            )
    symbols = (
        sorted(labels["__symbol__"].astype(str).unique())
        if not labels.empty
        else _feature_symbols(args.feature_root)
    )
    if allowed_symbols is not None:
        symbols = sorted(set(symbols).intersection(allowed_symbols))
    if int(args.symbol_limit) > 0:
        symbols = symbols[: int(args.symbol_limit)]
    if not symbols:
        raise RuntimeError("No symbols are available for the requested scoring scope")

    expected_timestamps = pd.date_range(start, end - pd.Timedelta(hours=1), freq="h")
    context_start = start - pd.Timedelta(hours=int(args.context_warmup_hours))
    context_timestamps = pd.date_range(
        context_start, end - pd.Timedelta(hours=1), freq="h"
    )
    observed_timestamps = pd.DatetimeIndex(labels["__ts__"].dropna().unique())
    tail_timestamps = (
        context_timestamps
        if args.rescore_observed
        else expected_timestamps.difference(observed_timestamps)
    )
    if tail_timestamps.empty:
        raise RuntimeError(
            "No missing hourly label batches were found in the requested scope"
        )
    backcast_observable_features: list[str] = []
    if bool(args.backcast_include_observable_features):
        from extreme_price_movements.unsupervised_regime_learning.feature_registry import (
            UNSUPERVISED_REGIME_PRIMITIVE_FEATURES,
        )

        backcast_observable_features = list(UNSUPERVISED_REGIME_PRIMITIVE_FEATURES)
    synthetic_raw = _read_tail_feature_rows(
        args.feature_root,
        symbols=symbols,
        timestamps=tail_timestamps,
        columns=_store_input_columns(
            list(ae_gmm_state["feature_columns"]),
            [*list(base_columns), *backcast_observable_features],
            allowed_matrix_feature_columns,
        ),
    )
    # The frozen live matrix contains cross-sectional features that are
    # materialized after the per-symbol stores are read.  Merely adding their
    # names to the store request leaves them all-NaN and changes source-regime
    # scores before the frozen AE/GMM transform.  Hydrate the exact matrix
    # values for this parity timestamp, keyed by symbol, while retaining store
    # values for every ordinary feature.
    if allowed_matrix_values is not None and len(tail_timestamps) == 1:
        matrix_symbols = pd.Index(
            allowed_matrix_values.index.astype(str),
            name="__symbol__",
        )
        if allowed_symbols is not None:
            matrix_symbols = matrix_symbols[
                matrix_symbols.isin(sorted(allowed_symbols))
            ]
        observed_symbols = set(synthetic_raw["__symbol__"].astype(str))
        missing_symbols = [
            symbol for symbol in matrix_symbols if symbol not in observed_symbols
        ]
        if missing_symbols:
            missing_rows = allowed_matrix_values.reindex(missing_symbols).copy()
            missing_rows["__symbol__"] = missing_rows.index.astype(str)
            missing_rows["__ts__"] = pd.Timestamp(tail_timestamps[0])
            synthetic_raw = pd.concat(
                [synthetic_raw, missing_rows.reset_index(drop=True)],
                ignore_index=True,
                sort=False,
                copy=False,
            )
            print(
                "Canonical latest matrix supplied exact-hour rows missing from "
                f"the partition store: n={len(missing_symbols)} "
                f"sample={missing_symbols[:8]}"
            )
        matrix_aligned = allowed_matrix_values.reindex(
            synthetic_raw["__symbol__"].astype(str)
        )
        for column in allowed_matrix_feature_columns:
            if column not in matrix_aligned.columns:
                continue
            matrix_values = pd.to_numeric(
                matrix_aligned[column], errors="coerce"
            ).to_numpy(dtype=np.float32, copy=False)
            if column not in synthetic_raw.columns:
                synthetic_raw[column] = matrix_values
                continue
            store_values = pd.to_numeric(
                synthetic_raw[column], errors="coerce"
            ).to_numpy(dtype=np.float32, copy=True)
            finite_matrix = np.isfinite(matrix_values)
            store_values[finite_matrix] = matrix_values[finite_matrix]
            synthetic_raw[column] = store_values
    synthetic_raw = _hydrate_live_gated_inputs(
        synthetic_raw,
        data_root=args.feature_root.parents[1],
        symbols=symbols,
        timestamps=tail_timestamps,
        required_columns=list(ae_gmm_state["feature_columns"]),
    )
    synthetic_parts = []
    for side, side_value in (("long", 1.0), ("short", -1.0)):
        part = synthetic_raw.copy()
        part["side_name"] = side
        part["side"] = np.float32(side_value)
        part["__side__"] = np.float32(side_value)
        barrier_history = labels.loc[
            labels["side_name"].eq(side),
            ["__ts__", "__symbol__", "__barrier_pct__"],
        ].sort_values(["__ts__", "__symbol__"], kind="stable")
        if barrier_history.empty:
            part["__barrier_pct__"] = np.float32(args.default_barrier_pct)
        else:
            part = pd.merge_asof(
                part.sort_values(["__ts__", "__symbol__"], kind="stable"),
                barrier_history,
                on="__ts__",
                by="__symbol__",
                direction="backward",
                allow_exact_matches=True,
            )
        part["__barrier_pct__"] = (
            pd.to_numeric(part["__barrier_pct__"], errors="coerce")
            .fillna(float(args.default_barrier_pct))
            .astype(np.float32)
        )
        part["_synthetic_tail"] = True
        synthetic_parts.append(part)
    synthetic = pd.concat(synthetic_parts, ignore_index=True, copy=False)
    overlay_columns = set(base_columns).union(ae_gmm_state["feature_columns"])
    synthetic_batches: list[pd.DataFrame] = []
    for side, batch in synthetic.groupby("side_name", sort=True):
        # Materialize the complete monthly side sequence in one call. Besides
        # avoiding a registry rebuild for every hour, this preserves the causal
        # within-symbol history required by prior_recent_source_strength().
        # Live inference still calls the same helper with one scalar timestamp.
        indexed = batch.sort_values(
            ["__symbol__", "__ts__"], kind="stable"
        ).reset_index(drop=True)
        enriched = materialize_live_source_regime_features(
            indexed,
            side=str(side),
            signal_bar_ts=None,
            required_columns=overlay_columns,
        )
        synthetic_batches.append(enriched)
    synthetic = pd.concat(synthetic_batches, ignore_index=True, copy=False)
    # Keep the complete label history only through synthetic-row construction,
    # where it supplies the causal per-symbol barrier lookup.  Scoring and
    # feature hydration must operate on the requested audit window; otherwise a
    # one-hour parity check needlessly expands the entire monthly label table.
    labels = labels.loc[labels["__ts__"].ge(start) & labels["__ts__"].lt(end)].copy()
    if args.rescore_observed:
        labels = labels.loc[~labels["__ts__"].isin(tail_timestamps)].copy()
    labels["_synthetic_tail"] = False
    full = pd.concat([labels, synthetic], ignore_index=True, sort=False, copy=False)
    full = full.sort_values(
        ["__ts__", "__symbol__", "side_name"], kind="stable"
    ).reset_index(drop=True)
    complete_case_attrition: dict[str, Any] = {}

    ae_raw_complete, ae_raw_missing = _finite_complete_case_mask(
        full, list(ae_gmm_state["feature_columns"])
    )
    complete_case_attrition["base_ae_gmm_raw_inputs"] = _attrition_summary(
        full, ae_raw_complete, ae_raw_missing
    )

    ae_transform_scope = (
        os.environ.get("EPM_REPLAY_AE_GMM_TRANSFORM_SCOPE", "historical_full")
        .strip()
        .lower()
    )
    if ae_transform_scope == "historical_full":
        valid_ae = full.loc[ae_raw_complete]
        generated = transform_ae_gmm_features(
            valid_ae.reindex(columns=ae_gmm_state["feature_columns"]),
            ae_gmm_state,
            index=valid_ae.index,
        ).reindex(full.index)
    elif ae_transform_scope == "timestamp_side":
        generated_parts: list[pd.DataFrame] = []
        for (_, _side), batch in full.groupby(["__ts__", "side_name"], sort=True):
            batch_complete = ae_raw_complete.reindex(batch.index).fillna(False)
            valid_batch = batch.loc[batch_complete]
            if valid_batch.empty:
                continue
            generated_parts.append(
                transform_ae_gmm_features(
                    valid_batch.reindex(columns=ae_gmm_state["feature_columns"]),
                    ae_gmm_state,
                    index=valid_batch.index,
                )
            )
        generated = (
            pd.concat(generated_parts, axis=0).reindex(full.index)
            if generated_parts
            else pd.DataFrame(index=full.index)
        )
    else:
        raise ValueError(
            f"Unsupported EPM_REPLAY_AE_GMM_TRANSFORM_SCOPE={ae_transform_scope!r}"
        )
    for name in generated.columns:
        full[name] = generated[name].to_numpy(copy=False)
    # The production base contract is broader than the AE/GMM input basket.
    # Hydrate every selected base feature from the same frozen feature store
    # before scoring; otherwise LightGBM silently treats omitted columns as
    # missing and the historical replay diverges before the meta handoff.
    full, base_store_coverage = _fill_store_features(
        full,
        args.feature_root,
        base_columns,
        prefer_existing_finite=allowed_matrix_values is not None,
    )
    full["score"] = np.nan
    if base_models_by_side:
        for side in ("long", "short"):
            side_mask = full["side_name"].astype(str).str.lower().eq(side)
            columns = list(base_contract["feature_names_by_side"][side])
            side_complete, side_missing = _finite_complete_case_mask(
                full.loc[side_mask], columns
            )
            side_complete &= ae_raw_complete.loc[side_mask]
            accepted_index = side_complete.index[side_complete]
            complete_case_attrition[f"base_model_{side}"] = _attrition_summary(
                full.loc[side_mask], side_complete, side_missing
            )
            if len(accepted_index):
                matrix = full.loc[accepted_index].reindex(columns=columns)
                scores, attributions = _predict_with_family_attributions(
                    base_models_by_side[side], matrix
                )
                full.loc[accepted_index, "score"] = scores
                for name in attributions.columns:
                    if name not in full.columns:
                        full[name] = np.float32(np.nan)
                    full.loc[accepted_index, name] = attributions[name]
    else:
        base_model = joblib.load(args.base_model_dir / "base_model.joblib")
        base_complete, base_missing = _finite_complete_case_mask(full, base_columns)
        base_complete &= ae_raw_complete
        complete_case_attrition["base_model_global"] = _attrition_summary(
            full, base_complete, base_missing
        )
        accepted_index = base_complete.index[base_complete]
        if len(accepted_index):
            matrix = full.loc[accepted_index].reindex(columns=base_columns)
            scores, attributions = _predict_with_family_attributions(base_model, matrix)
            full.loc[accepted_index, "score"] = scores
            for name in attributions.columns:
                if name not in full.columns:
                    full[name] = np.float32(np.nan)
                full.loc[accepted_index, name] = attributions[name]

    in_scope = full.loc[full["__ts__"].ge(start) & full["__ts__"].lt(end)].copy()
    if packaged_model_root is not None:
        keep_indices: list[int] = []
        for _, batch in in_scope.groupby(["__ts__", "side_name"], sort=True):
            ranked = pd.to_numeric(batch["score"], errors="coerce").dropna()
            top_n = max(1, int(np.ceil(len(ranked) * float(args.base_top_frac))))
            keep_indices.extend(
                ranked.sort_values(ascending=False).head(top_n).index.tolist()
            )
        valid = in_scope.loc[keep_indices].copy()
        base_cutoff = float("nan")
    else:
        base_reference = pd.read_parquet(
            args.base_reference, columns=["score", "selected_top30"]
        )
        base_cutoff = float(
            pd.to_numeric(
                base_reference.loc[
                    base_reference["selected_top30"].astype(bool), "score"
                ],
                errors="coerce",
            ).min()
        )
        valid = in_scope.loc[
            pd.to_numeric(in_scope["score"], errors="coerce").ge(base_cutoff)
        ].copy()
    valid["selected_top30"] = True

    if args.base_only_backcast:
        if args.evidence_scope != "frozen_backcast_diagnostic":
            raise ValueError("--base-only-backcast is diagnostic evidence only")
        if not 0.0 < float(args.backcast_admission_frac) <= float(args.base_top_frac):
            raise ValueError(
                "--backcast-admission-frac must be positive and no larger than "
                "--base-top-frac"
            )
        classifier = load_live_policy_archetype_classifier(
            data_root="data_perp", run_id=args.native_run_id
        )
        return _run_base_only_backcast(
            valid,
            args=args,
            classifier=classifier,
            complete_case_attrition=complete_case_attrition,
            symbols=symbols,
            start=start,
            end=end,
        )

    if args.stop_after_base:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        diagnostic = pd.concat(
            [
                valid[KEYS + ["score"]].reset_index(drop=True),
                valid.reindex(columns=base_columns)
                .reset_index(drop=True)
                .add_prefix("base_input__"),
                valid.reindex(columns=list(ae_gmm_state["feature_columns"]))
                .reset_index(drop=True)
                .add_prefix("ae_input__"),
            ],
            axis=1,
        )
        output_path = args.output_dir / "base_model_input.parquet"
        diagnostic.to_parquet(output_path, index=False, compression="zstd")
        manifest = {
            "schema": "complete_july_base_input_diagnostic_v1",
            "start": start.isoformat(),
            "end_exclusive": end.isoformat(),
            "context_start": context_start.isoformat(),
            "context_warmup_hours": int(args.context_warmup_hours),
            "rescore_observed": bool(args.rescore_observed),
            "rows": int(len(valid)),
            "base_top_frac": float(args.base_top_frac),
            "base_feature_store_coverage": base_store_coverage,
            # A zero-row diagnostic must explain which strict input contract
            # rejected the rows.  Without this, historical parity failures are
            # indistinguishable from a genuine empty candidate universe.
            "complete_case_attrition": complete_case_attrition,
            "output": str(output_path),
        }
        (args.output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, default=str), encoding="utf-8"
        )
        print(json.dumps(manifest, indent=2, default=str))
        return 0

    source_contract = _load_json(args.source_manifest)["source_contract"]
    source_edges = source_contract.get("edges")
    if source_edges:
        valid["source_tag"] = _source_tags(
            valid["score"], valid["side_name"], source_edges
        ).to_numpy()
    elif packaged_model_root is not None:
        # Packaged inference rebuilds source_tag from the frozen base-score
        # reliability contract immediately below. The newer semantic source
        # manifest intentionally has no global score-bin edges.
        valid["source_tag"] = ""
    else:
        raise KeyError("Legacy exported-model replay requires source_contract.edges")
    valid, store_coverage = _fill_store_features(
        valid,
        args.feature_root,
        list(dict.fromkeys(meta_pre_ood)),
        prefer_existing_finite=allowed_matrix_values is not None,
    )
    valid, derived_meta_coverage = _hydrate_derived_meta_features(
        valid,
        feature_root=args.feature_root,
        requested=meta_pre_ood,
    )
    store_coverage.update(derived_meta_coverage)

    train = pd.DataFrame()
    if packaged_model_root is None:
        joined = _load_joined_frame(
            args.meta_handoff_dir / "train_meta_regime_handoff.parquet",
            args.meta_handoff_dir / "s52_trailing_regime_scored_ledger.parquet",
            "top30",
        )
        joined["__ts__"] = _utc(joined["__ts__"])
        train = joined.loc[
            joined["__ts__"].lt(pd.Timestamp("2026-07-01", tz="UTC"))
        ].reset_index(drop=True)
        train, _ = _fill_store_features(train, args.feature_root, meta_pre_ood)
    valid = valid.reset_index(drop=True)
    if packaged_model_root is not None:
        reliability_payload = _load_json(
            packaged_model_root / "policy_params" / "meta_reliability_priors.json"
        )
        reliability_parts: list[pd.DataFrame] = []
        valid["_replay_row_id"] = np.arange(len(valid), dtype=np.int64)
        for (timestamp, side), side_rows in valid.groupby(
            ["__ts__", "side_name"], sort=False
        ):
            side = str(side).lower()
            side_rows = side_rows.copy()
            # The live helper expects one row per symbol. Historical replay must
            # therefore call it per decision batch, not once over many timestamps.
            side_rows = side_rows.drop(
                columns=[
                    name
                    for name in (
                        "source_tag",
                        "__source_tag__",
                        "archetype_source_tag",
                    )
                    if name in side_rows.columns
                ]
            )
            side_rows = side_rows.set_index("__symbol__", drop=False)
            if not side_rows.index.is_unique:
                raise RuntimeError(
                    "Reliability replay batch contains duplicate symbols: "
                    f"timestamp={timestamp} side={side}"
                )
            base_predictions = {
                str(symbol): {"base_pred": float(score)}
                for symbol, score in zip(
                    side_rows.index,
                    pd.to_numeric(side_rows["score"], errors="coerce"),
                )
            }
            side_rows = apply_live_meta_reliability_priors(
                side_rows,
                side=side,
                base_predictions=base_predictions,
                prior_payload=reliability_payload,
            )
            reliability_parts.append(side_rows.reset_index(drop=True))
        if not reliability_parts:
            attrition = json.dumps(complete_case_attrition, indent=2, default=str)
            raise RuntimeError(
                "No finite base candidates reached packaged meta reliability "
                f"materialization. Complete-case attrition:\n{attrition}"
            )
        valid = (
            pd.concat(reliability_parts, ignore_index=True, copy=False)
            .sort_values("_replay_row_id", kind="stable")
            .drop(columns="_replay_row_id")
            .reset_index(drop=True)
        )
    else:
        train, valid = _add_fold_base_prior_features(
            train, valid, selected_col="selected_top30"
        )
        train, valid = _add_fold_reliability_features(train, valid)
    valid_matrix = pd.DataFrame(
        index=valid.index, columns=meta_columns, dtype=np.float32
    )
    valid["score_meta_base_soft_label"] = np.nan
    side_model_contract = {
        side: meta_columns_by_model.get(f"base_soft_label_{side}", [])
        for side in ("long", "short")
    }
    if all(side_model_contract.values()):
        for side in ("long", "short"):
            columns = side_model_contract[side]
            pre_ood = [name for name in columns if not name.startswith(OOD_PREFIX)]
            valid_mask = valid["side_name"].astype(str).str.lower().eq(side)
            meta_model = meta_models_by_side.get(side)
            if meta_model is None:
                meta_model = joblib.load(
                    args.meta_model_dir / f"base_soft_label_{side}.joblib"
                )
            if packaged_model_root is not None:
                side_matrix = append_s52_meta_ood_features(
                    valid.loc[valid_mask],
                    getattr(meta_model, "s52_meta_ood_reference_", None),
                    output_features=[
                        name for name in columns if name.startswith(OOD_PREFIX)
                    ],
                ).reindex(columns=columns)
            else:
                train_mask = train["side_name"].astype(str).str.lower().eq(side)
                ood_reference = _fit_meta_post_selection_ood_reference(
                    train.loc[train_mask], pre_ood
                )
                side_matrix = _append_meta_post_selection_ood_features(
                    valid.loc[valid_mask, pre_ood],
                    valid.loc[valid_mask],
                    ood_reference,
                ).reindex(columns=columns)
            meta_complete, meta_missing = _finite_complete_case_mask(
                side_matrix, columns
            )
            complete_case_attrition[f"meta_model_{side}"] = _attrition_summary(
                valid.loc[valid_mask], meta_complete, meta_missing
            )
            accepted_index = meta_complete.index[meta_complete]
            if len(accepted_index):
                valid.loc[accepted_index, "score_meta_base_soft_label"] = (
                    meta_model.predict(side_matrix.loc[accepted_index, columns]).astype(
                        np.float32
                    )
                )
            if meta_score_alignment is not None:
                valid.loc[accepted_index, "score_meta_base_soft_label"] = (
                    apply_s52_meta_score_alignment(
                        valid.loc[accepted_index, "score_meta_base_soft_label"],
                        meta_score_alignment,
                        side=side,
                    )
                )
            valid_matrix.loc[valid_mask, columns] = side_matrix.to_numpy(
                dtype=np.float32, copy=False
            )
    else:
        ood_reference = _fit_meta_post_selection_ood_reference(train, meta_pre_ood)
        valid_matrix = _append_meta_post_selection_ood_features(
            valid.reindex(columns=meta_pre_ood), valid, ood_reference
        ).reindex(columns=meta_columns)
        meta_model = joblib.load(args.meta_model_dir / "base_soft_label.joblib")
        meta_complete, meta_missing = _finite_complete_case_mask(
            valid_matrix, meta_columns
        )
        complete_case_attrition["meta_model_global"] = _attrition_summary(
            valid, meta_complete, meta_missing
        )
        accepted_index = meta_complete.index[meta_complete]
        valid.loc[accepted_index, "score_meta_base_soft_label"] = meta_model.predict(
            valid_matrix.loc[accepted_index, meta_columns]
        ).astype(np.float32)
        if meta_score_alignment is not None:
            for side in ("long", "short"):
                mask = valid.index.isin(accepted_index) & valid["side_name"].astype(
                    str
                ).str.lower().eq(side)
                valid.loc[mask, "score_meta_base_soft_label"] = (
                    apply_s52_meta_score_alignment(
                        valid.loc[mask, "score_meta_base_soft_label"],
                        meta_score_alignment,
                        side=side,
                    )
                )

    meta_scored = np.isfinite(
        pd.to_numeric(valid["score_meta_base_soft_label"], errors="coerce").to_numpy(
            dtype=np.float64, copy=False
        )
    )
    complete_case_attrition["meta_scored_output"] = _attrition_summary(
        valid,
        pd.Series(meta_scored, index=valid.index),
        {"score_meta_base_soft_label": int((~meta_scored).sum())}
        if bool((~meta_scored).any())
        else {},
    )
    valid = valid.loc[meta_scored].copy()
    valid_matrix = valid_matrix.loc[valid.index].copy()

    # Policy archetypes are observable pre-entry context. Materialize them even
    # for a raw-meta backcast so downstream diagnostics remain side x archetype
    # aware without applying V9/MLP/recent-performance postprocessing.
    if args.stop_after_meta:
        classifier = load_live_policy_archetype_classifier(
            data_root="data_perp", run_id=args.native_run_id
        )
        if "__archetype_policy_key__" not in valid:
            valid["__archetype_policy_key__"] = ""
        else:
            valid["__archetype_policy_key__"] = (
                valid["__archetype_policy_key__"].fillna("").astype(str)
            )
        side_defaults = dict(classifier.get("side_defaults") or {})
        assignment_mask = valid["__archetype_policy_key__"].eq("")
        for idx in valid.index[assignment_mask]:
            side = str(valid.at[idx, "side_name"])
            predicted = predict_observable_policy_archetype(
                side=side,
                candidate_feature_row=valid.loc[[idx]],
                meta_model_input_row=valid_matrix.loc[[idx]],
            )
            if not predicted:
                predicted = predict_live_policy_archetype(
                    side=side,
                    payload=classifier,
                    candidate_feature_row=valid.loc[[idx]],
                    meta_model_input_row=valid_matrix.loc[[idx]],
                )
            if not predicted:
                predicted = str(side_defaults.get(side, ""))
            prefix = f"{side}__"
            valid.at[idx, "__archetype_policy_key__"] = (
                predicted[len(prefix) :] if predicted.startswith(prefix) else predicted
            )
        valid["archetype_policy_key"] = valid["__archetype_policy_key__"].astype(str)

    if args.persist_meta_input or args.stop_after_meta:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        if args.compact_meta_output:
            state_columns = sorted(
                name
                for name in valid.columns
                if name.startswith(("gmm_", "dae_", "AE_"))
            )
            compact_columns = [
                *KEYS,
                "side",
                "source_tag",
                "source_family",
                "archetype_policy_key",
                "score",
                "score_meta_base_soft_label",
                "selected_top30",
                *state_columns,
            ]
            diagnostic = valid[
                [name for name in dict.fromkeys(compact_columns) if name in valid.columns]
            ].reset_index(drop=True)
        else:
            base_input = valid.reindex(columns=base_columns).reset_index(drop=True)
            diagnostic = pd.concat(
                [
                    valid[
                        KEYS
                        + [
                            name
                            for name in (
                                "source_tag",
                                "archetype_policy_key",
                                "score",
                                "score_meta_base_soft_label",
                            )
                            if name in valid.columns
                        ]
                    ].reset_index(drop=True),
                    base_input.add_prefix("base_input__"),
                    valid_matrix.reset_index(drop=True).add_prefix("meta_input__"),
                ],
                axis=1,
            )
        meta_input_path = args.output_dir / str(args.output_filename)
        diagnostic.to_parquet(
            meta_input_path,
            index=False,
            compression="zstd",
        )
    if args.stop_after_meta:
        if packaged_model_root is not None:
            # Packaged inference artifacts serialize both side-specific base
            # and meta heads in one immutable state file.  The legacy
            # ``model_state_meta.pkl`` path is not part of that contract.
            meta_model_paths = [packaged_model_root / "models" / "trained_state.pkl"]
            base_model_provenance = _file_provenance(
                packaged_model_root / "models" / "trained_state.pkl"
            )
            base_columns_provenance = _file_provenance(
                packaged_model_root / "models" / "trained_state.pkl"
            )
            meta_columns_provenance = _file_provenance(
                packaged_model_root / "meta_oof" / "meta_feature_contract.json"
            )
        else:
            meta_model_paths = sorted(
                args.meta_model_dir.glob("base_soft_label*.joblib")
            )
            base_model_provenance = _file_provenance(
                args.base_model_dir / "base_model.joblib"
            )
            base_columns_provenance = _file_provenance(
                args.base_model_dir / "columns.json"
            )
            meta_columns_provenance = _file_provenance(
                args.meta_model_dir / "columns.json"
            )
        manifest = {
            "schema": (
                "frozen_meta_compact_candidate_backcast_v1"
                if args.compact_meta_output
                else "complete_july_meta_input_diagnostic_v1"
            ),
            "start": start.isoformat(),
            "end_exclusive": end.isoformat(),
            "context_start": context_start.isoformat(),
            "context_warmup_hours": int(args.context_warmup_hours),
            "rescore_observed": bool(args.rescore_observed),
            "rows": int(len(valid)),
            "base_cutoff": float(base_cutoff),
            "base_feature_contract_hash": base_contract.get("feature_contract_hash"),
            "meta_feature_contract_hash": meta_contract.get("feature_contract_hash"),
            "model_provenance": {
                "base_model": base_model_provenance,
                "meta_models": [_file_provenance(path) for path in meta_model_paths],
                "base_columns": base_columns_provenance,
                "meta_columns": meta_columns_provenance,
                "ae_gmm_state": _file_provenance(args.ae_gmm_state),
                "meta_score_alignment": (
                    _file_provenance(args.meta_score_alignment)
                    if args.meta_score_alignment is not None
                    else None
                ),
            },
            "feature_store_coverage": store_coverage,
            "base_feature_store_coverage": base_store_coverage,
            # Persist the same audit for meta-only backcasts.  This is the
            # intended diagnostic mode for expanding frozen-contract history.
            "complete_case_attrition": complete_case_attrition,
            "output": str(meta_input_path),
            "compact_meta_output": bool(args.compact_meta_output),
            "postprocessing_applied": False,
            "evidence_scope": str(args.evidence_scope),
        }
        (args.output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, default=str), encoding="utf-8"
        )
        print(json.dumps(manifest, indent=2, default=str))
        return 0

    classifier = load_live_policy_archetype_classifier(
        data_root="data_perp", run_id=args.native_run_id
    )
    if "__archetype_policy_key__" not in valid:
        valid["__archetype_policy_key__"] = ""
    else:
        valid["__archetype_policy_key__"] = (
            valid["__archetype_policy_key__"].fillna("").astype(str)
        )
    synthetic_mask = valid["_synthetic_tail"].fillna(False).astype(bool)
    assignment_mask = synthetic_mask | valid["__archetype_policy_key__"].eq("")
    side_defaults = dict(classifier.get("side_defaults") or {})
    for idx in valid.index[assignment_mask]:
        side = str(valid.at[idx, "side_name"])
        predicted = predict_observable_policy_archetype(
            side=side,
            candidate_feature_row=valid.loc[[idx]],
            meta_model_input_row=valid_matrix.loc[[idx]],
        )
        if not predicted:
            predicted = predict_live_policy_archetype(
                side=side,
                payload=classifier,
                candidate_feature_row=valid.loc[[idx]],
                meta_model_input_row=valid_matrix.loc[[idx]],
            )
        if not predicted:
            predicted = str(side_defaults.get(side, ""))
        prefix = f"{side}__"
        valid.at[idx, "__archetype_policy_key__"] = (
            predicted[len(prefix) :] if predicted.startswith(prefix) else predicted
        )
    archetype_complete = valid["__archetype_policy_key__"].astype(str).str.len().gt(0)
    complete_case_attrition["policy_archetype"] = _attrition_summary(
        valid,
        archetype_complete,
        {"__archetype_policy_key__": int((~archetype_complete).sum())},
    )
    valid = valid.loc[archetype_complete].copy()
    valid_matrix = valid_matrix.loc[valid.index].copy()
    if valid.empty:
        raise RuntimeError(
            "No rows received a live-predictable side/archetype assignment"
        )
    valid["archetype_policy_key"] = valid["__archetype_policy_key__"].astype(str)

    # Production replaces the shared meta-backbone score with the frozen
    # side-local base-residual expert rank before V9/MLP. Historical scoring
    # must do the same; otherwise identical raw meta inputs still diverge at
    # the first post-base score.
    if packaged_model_root is not None:
        portfolio_config_path = (
            packaged_model_root
            / "policy_params"
            / "optimized_portfolio_policy_config.json"
        )
        portfolio_config = (
            _load_json(portfolio_config_path) if portfolio_config_path.exists() else {}
        )
        if bool(portfolio_config.get("side_residual_expert_enabled", False)):
            expert_path = Path(
                portfolio_config.get("side_residual_expert_artifact_path")
                or packaged_model_root / "policy_params" / "side_residual_expert.joblib"
            )
            expert = SideResidualExpertBundle.load(expert_path)
            expert_required = expert.required_input_features()
            expert_store_features = [
                name
                for name in expert_required
                if name
                not in {
                    "side_name",
                    "archetype_policy_key",
                    "score",
                    "score_base",
                }
            ]
            valid, expert_coverage = _fill_store_features(
                valid,
                args.feature_root,
                expert_store_features,
                prefer_existing_finite=allowed_matrix_values is not None,
            )
            valid["score_base"] = pd.to_numeric(valid["score"], errors="coerce").astype(
                np.float32
            )
            expert_output = expert.transform(valid)
            expert_complete = expert_output[
                "meta_residual_expert_complete_case"
            ].astype(bool)
            expert_missing: dict[str, int] = {}
            for side in ("long", "short"):
                side_mask = valid["side_name"].astype(str).str.lower().eq(side)
                columns = list(expert.payload["feature_contract"][side])
                _, missing = _finite_complete_case_mask(valid.loc[side_mask], columns)
                for name, count in missing.items():
                    expert_missing[name] = expert_missing.get(name, 0) + int(count)
            complete_case_attrition["side_residual_expert"] = _attrition_summary(
                valid, expert_complete, expert_missing
            )
            valid = valid.loc[expert_complete].copy()
            valid_matrix = valid_matrix.loc[valid.index].copy()
            expert_output = expert_output.loc[valid.index]
            valid["score_meta_backbone"] = valid["score_meta_base_soft_label"].astype(
                np.float32
            )
            valid["score_meta_base_soft_label"] = pd.to_numeric(
                expert_output["score_base_residual_ev_rank_train_reference"],
                errors="coerce",
            ).astype(np.float32)
            for column in expert_output.columns:
                valid[column] = expert_output[column]
            store_coverage.update(
                {
                    f"side_residual::{key}": value
                    for key, value in expert_coverage.items()
                }
            )

    residual_train = pd.read_parquet(args.residual_train_reference)
    residual_train["__ts__"] = _utc(residual_train["__ts__"])
    residual_train = residual_train.loc[
        residual_train["__ts__"].lt(pd.Timestamp("2026-07-01", tz="UTC"))
    ].reset_index(drop=True)
    # Validation rows intentionally have no realized outcome. The shared
    # training helper only needs a column-shaped placeholder here; every
    # validation prior is computed from residual_train.
    valid["clean_exec_label"] = np.float32(0.0)
    _residual_train_enriched, valid = _add_reference_fold_features(
        residual_train, valid
    )

    if args.residual_event_state is None or args.regime_ev_calibration is None:
        raise RuntimeError(
            "Canonical July scoring requires the V9 predecessor, residual-event "
            "state, and MLP/hierarchical-EV artifact"
        )
    canonical_postprocessor = CanonicalMetaPostprocessor.load(
        predecessor_bundle_path=args.residual_bundle,
        residual_event_state_path=args.residual_event_state,
        regime_ev_artifact_path=args.regime_ev_calibration,
    )
    valid["score_base"] = pd.to_numeric(valid["score"], errors="coerce").astype(
        np.float32
    )
    residual_required = canonical_postprocessor.required_input_features()
    valid, residual_coverage = _fill_store_features(
        valid,
        args.feature_root,
        residual_required,
        prefer_existing_finite=allowed_matrix_values is not None,
    )
    regime_artifact: dict[str, Any] = {}
    regime_required: list[str] = []
    regime_feature_coverage: dict[str, float] = {}
    residual_state_coverage: dict[str, float] = {}
    if args.residual_event_state is not None:
        residual_state = canonical_postprocessor.residual_event_state
        residual_state_inputs: list[str] = []
        for model in residual_state.local_models.values():
            residual_state_inputs.extend(model.feature_columns)
        if residual_state.market_model is not None:
            residual_state_inputs.extend(residual_state.market_model.feature_columns)
        residual_state_inputs = list(dict.fromkeys(residual_state_inputs))
        valid, residual_state_coverage = _fill_store_features(
            valid,
            args.feature_root,
            residual_state_inputs,
            prefer_existing_finite=allowed_matrix_values is not None,
        )

    if args.regime_ev_calibration is not None:
        regime_artifact = dict(canonical_postprocessor.regime_ev_artifact)
        regime_required = required_feature_columns(regime_artifact)
        valid, regime_feature_coverage = _fill_store_features(
            valid,
            args.feature_root,
            regime_required,
            prefer_existing_finite=allowed_matrix_values is not None,
        )
        missing = sorted(name for name in regime_required if name not in valid.columns)
        if missing:
            raise RuntimeError(
                "Frozen V9+MLP calibration inputs are missing: " + ", ".join(missing)
            )
        postprocessor_complete = canonical_postprocessor.complete_case_report(valid)
        postprocessor_mask = postprocessor_complete["complete_case"].astype(bool)
        missing_counts: dict[str, int] = {}
        for names in postprocessor_complete.loc[
            ~postprocessor_mask, "missing_features"
        ].astype(str):
            for name in (item for item in names.split(",") if item):
                missing_counts[name] = missing_counts.get(name, 0) + 1
        complete_case_attrition["canonical_meta_postprocessor"] = _attrition_summary(
            valid, postprocessor_mask, missing_counts
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        pd.concat([valid[KEYS], postprocessor_complete], axis=1).to_parquet(
            args.output_dir / "complete_case_postprocessor_report.parquet",
            index=False,
            compression="zstd",
        )
        valid = valid.loc[postprocessor_mask].copy()
        if valid.empty:
            raise RuntimeError(
                "No rows satisfy the strict V9/residual/MLP complete-case contract"
            )
        # Persist the exact complete-case frame passed to the frozen V9 chain.
        # This is intentionally before transform so historical/live parity can
        # identify the first input divergence without reconstructing discarded
        # feature columns from the reduced prediction artifact.
        valid.to_parquet(
            args.output_dir / "canonical_postprocessor_input.parquet",
            index=False,
            compression="zstd",
        )
        valid = canonical_postprocessor.transform(
            valid,
            copy=False,
        )
        generated_output_columns = [
            name for name in valid.columns if name.startswith("resid_event_")
        ] + [
            "historical_rank",
            "score_regime_calibrated",
            "expected_net_ev_after_1pct",
            "expected_ev_rank_score",
        ]
        generated_complete, generated_missing = _finite_complete_case_mask(
            valid, generated_output_columns
        )
        complete_case_attrition["canonical_meta_postprocessor_outputs"] = (
            _attrition_summary(valid, generated_complete, generated_missing)
        )
        valid = valid.loc[generated_complete].copy()

    outcome_cols = [
        "ev_after_1pct",
        "clean_exec",
        "first_touch_bad_mae_1r",
        "full_path_bad_mae_1r",
        "timeout",
    ]
    if args.old_labels_dir is not None and args.labels_dir is not None:
        outcomes = _outcomes_from_labels(args.old_labels_dir, args.labels_dir)
    else:
        outcomes = pd.DataFrame(columns=[*KEYS, *outcome_cols])
    valid = valid.drop(columns=[name for name in outcome_cols if name in valid.columns])
    valid = valid.merge(
        outcomes[KEYS + outcome_cols], on=KEYS, how="left", validate="one_to_one"
    )
    valid["exec_margin"] = np.nan
    valid["dirty_positive"] = np.nan

    policy_manifest = _load_json(args.policy_manifest)
    path_stats: dict[str, Any] = {}
    synthetic_mask = valid["_synthetic_tail"].fillna(False).astype(bool)
    replay_mask = synthetic_mask | valid["ev_after_1pct"].isna()
    if args.allow_unresolved_outcomes:
        path_stats["prediction_parity"] = {
            "status": "outcome_replay_skipped",
            "unresolved_rows": int(replay_mask.sum()),
        }
    else:
        for side in ("long", "short"):
            mask = replay_mask.to_numpy() & valid["side_name"].eq(side).to_numpy()
            if not bool(mask.any()):
                continue
            rows = valid.loc[mask].copy().reset_index(drop=True)
            capture, side_path_stats = _capture_for_policy_keys(
                rows,
                side=side,
                policy_keys=rows["archetype_policy_key"],
                policy_manifest=policy_manifest,
                data_root=Path("data_perp"),
                path_len=int(args.path_len),
                allow_partial_paths=True,
            )
            path_stats[side] = side_path_stats
            captured = _capture_outcomes(capture)
            for name in captured.columns:
                valid.loc[mask, name] = captured[name].to_numpy(copy=False)

    valid["score_current_reference"] = valid["score_meta_base_soft_label"].astype(
        np.float32
    )
    valid["evidence_scope"] = str(args.evidence_scope)
    valid["prediction_evidence"] = (
        "frozen_research_contract_complete_backfill"
        if args.evidence_scope == "oof_oos"
        else "frozen_model_historical_backcast_diagnostic_not_oos"
    )
    keep = [
        *KEYS,
        "__barrier_pct__",
        "archetype_policy_key",
        "score_current_reference",
        "score_meta_base_soft_label",
        "score_shock_adjusted",
        "score_lifecycle_only",
        "score_residual_overlay",
        "shock_composite_raw",
        "shock_composite_local",
        "hit_probability",
        "historical_rank",
        "ev_after_1pct",
        "exec_margin",
        "clean_exec",
        "dirty_positive",
        "first_touch_bad_mae_1r",
        "full_path_bad_mae_1r",
        "timeout",
        "evidence_scope",
        "prediction_evidence",
        "_synthetic_tail",
        "calibrated_score",
        "score_regime_calibrated",
        "regime_ev_risk_score",
        "regime_ev_effect_count",
        "market_state_mlp_score_correction",
        "expected_net_ev_after_1pct",
        "expected_ev_rank_score",
        "regime_ev_blacklisted",
        "meta_postprocessor_policy_id",
        *regime_required,
    ]
    keep = list(dict.fromkeys(name for name in keep if name in valid.columns))
    complete = valid[keep].copy()
    complete = complete.sort_values(KEYS, kind="stable").drop_duplicates(
        KEYS, keep="last"
    )

    threshold_policy_id = None
    if args.threshold_policy is not None:
        threshold_policy = load_threshold_basis_policy(args.threshold_policy)
        if not threshold_policy:
            raise FileNotFoundError(
                f"Threshold policy could not be loaded: {args.threshold_policy}"
            )
        decisions = [
            {
                "timestamp": timestamp,
                "symbol": symbol,
                "side_name": side,
                "policy_archetype": archetype,
                "strategy_id": f"{side}_s52_meta_threshold_handoff",
                "expected_net_ev_after_1pct_side_archetype": float(mapped_ev),
                "expected_ev_rank_score": float(ev_rank),
                "v9_tail95_predecessor_rank": float(parent_rank),
                "policy_rank_pct": float(ev_rank),
            }
            for (
                timestamp,
                symbol,
                side,
                archetype,
                mapped_ev,
                ev_rank,
                parent_rank,
            ) in complete[
                [
                    "__ts__",
                    "__symbol__",
                    "side_name",
                    "archetype_policy_key",
                    "expected_net_ev_after_1pct",
                    "expected_ev_rank_score",
                    "historical_rank",
                ]
            ].itertuples(index=False, name=None)
        ]
        apply_threshold_basis_policy_to_decisions(decisions, policy=threshold_policy)
        complete["threshold_basis_selected"] = np.fromiter(
            (bool(row.get("threshold_basis_selected", False)) for row in decisions),
            dtype=bool,
            count=len(decisions),
        )
        for output_col, decision_key, default in (
            ("threshold_basis_rank_score", "threshold_basis_rank_score", np.nan),
            ("threshold_basis_multiplier", "threshold_basis_ev_target_multiplier", 1.0),
            (
                "threshold_basis_local_support",
                "threshold_basis_ev_target_local_support",
                0.0,
            ),
            (
                "threshold_basis_mapped_expected_ev_side_archetype",
                "threshold_basis_mapped_expected_ev_side_archetype",
                np.nan,
            ),
            (
                "threshold_basis_side_archetype_recent_ev_correction",
                "threshold_basis_side_archetype_recent_ev_correction",
                0.0,
            ),
            (
                "threshold_basis_corrected_expected_ev",
                "threshold_basis_corrected_expected_ev",
                np.nan,
            ),
            (
                "threshold_basis_corrected_expected_ev_rank",
                "threshold_basis_corrected_expected_ev_rank",
                np.nan,
            ),
            (
                "threshold_basis_dynamic_ev_target",
                "threshold_basis_dynamic_ev_target",
                np.nan,
            ),
            (
                "threshold_basis_dynamic_score_threshold",
                "threshold_basis_dynamic_score_threshold",
                np.nan,
            ),
        ):
            complete[output_col] = np.fromiter(
                (float(row.get(decision_key, default)) for row in decisions),
                dtype=np.float32,
                count=len(decisions),
            )
        complete["threshold_basis_global_fallback"] = np.fromiter(
            (
                bool(row.get("threshold_basis_ev_target_global_fallback", False))
                for row in decisions
            ),
            dtype=bool,
            count=len(decisions),
        )
        threshold_policy_id = threshold_policy.get("policy_id")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / str(args.output_filename)
    complete.to_parquet(output_path, index=False, compression="zstd")
    hour_counts = (
        complete.assign(day=complete["__ts__"].dt.strftime("%Y-%m-%d"))
        .groupby("day", observed=True)
        .agg(
            rows=("__ts__", "size"),
            hours=("__ts__", "nunique"),
            outcomes=("ev_after_1pct", "count"),
        )
        .reset_index()
    )
    hour_counts.to_csv(args.output_dir / "coverage_by_day.csv", index=False)
    if packaged_model_root is not None:
        base_model_provenance = _file_provenance(
            packaged_model_root / "models" / "trained_state.pkl"
        )
        meta_model_paths = [packaged_model_root / "models" / "trained_state.pkl"]
        base_columns_provenance = _file_provenance(
            packaged_model_root / "models" / "trained_state.pkl"
        )
        meta_columns_provenance = _file_provenance(
            packaged_model_root / "meta_oof" / "meta_feature_contract.json"
        )
    else:
        base_model_provenance = _file_provenance(
            args.base_model_dir / "base_model.joblib"
        )
        meta_model_paths = sorted(args.meta_model_dir.glob("base_soft_label*.joblib"))
        base_columns_provenance = _file_provenance(args.base_model_dir / "columns.json")
        meta_columns_provenance = _file_provenance(args.meta_model_dir / "columns.json")
    manifest = {
        "schema": "frozen_period_predictions_v2",
        "start": start.isoformat(),
        "end_exclusive": end.isoformat(),
        "context_start": context_start.isoformat(),
        "context_warmup_hours": int(args.context_warmup_hours),
        "evidence_scope": str(args.evidence_scope),
        "labels_available": bool(args.labels_dir is not None),
        "symbol_source": (
            "monthly_labels"
            if args.labels_dir is not None
            else "canonical_feature_store"
        ),
        "symbol_count": int(len(symbols)),
        "default_barrier_pct": float(args.default_barrier_pct),
        "base_cutoff": base_cutoff,
        "base_feature_contract_hash": base_contract.get("feature_contract_hash"),
        "meta_feature_contract_hash": meta_contract.get("feature_contract_hash"),
        "rescore_observed": bool(args.rescore_observed),
        "ae_gmm_transform_scope": ae_transform_scope,
        "meta_score_alignment_contract": {
            "effective": meta_score_alignment is not None,
            "disabled_by_cli": bool(args.disable_meta_score_alignment),
            "package_declared_mode": packaged_alignment_mode or None,
            "identity_score_domain": bool(identity_score_domain),
        },
        "model_provenance": {
            "base_model": base_model_provenance,
            "meta_models": [_file_provenance(path) for path in meta_model_paths],
            "base_columns": base_columns_provenance,
            "meta_columns": meta_columns_provenance,
            "ae_gmm_state": _file_provenance(args.ae_gmm_state),
            "meta_score_alignment": (
                _file_provenance(args.meta_score_alignment)
                if args.meta_score_alignment is not None
                else None
            ),
            "v9_predecessor": _file_provenance(args.residual_bundle),
            "residual_event_state": _file_provenance(args.residual_event_state),
            "mlp_hierarchical_ev": _file_provenance(args.regime_ev_calibration),
        },
        "rows": int(len(complete)),
        "hours": int(complete["__ts__"].nunique()),
        "outcome_rows": int(complete["ev_after_1pct"].notna().sum()),
        "feature_store_coverage": store_coverage,
        "base_feature_store_coverage": base_store_coverage,
        "residual_feature_coverage": residual_coverage,
        "residual_state_feature_coverage": residual_state_coverage,
        "regime_feature_coverage": regime_feature_coverage,
        "residual_event_state": str(args.residual_event_state)
        if args.residual_event_state is not None
        else None,
        "residual_train_reference": str(args.residual_train_reference),
        "regime_ev_calibration": str(args.regime_ev_calibration)
        if args.regime_ev_calibration is not None
        else None,
        "threshold_policy": str(args.threshold_policy)
        if args.threshold_policy is not None
        else None,
        "threshold_policy_id": threshold_policy_id,
        "threshold_selected_rows": int(
            complete.get(
                "threshold_basis_selected", pd.Series(False, index=complete.index)
            ).sum()
        ),
        "complete_case_contract": {
            "enabled": True,
            "rule": "reject_row_before_scoring_if_any_applicable_model_input_is_non_finite",
            "attrition": complete_case_attrition,
        },
        "allowed_symbols_matrix": str(args.allowed_symbols_matrix)
        if args.allowed_symbols_matrix is not None
        else None,
        "allowed_symbol_count": len(allowed_symbols)
        if allowed_symbols is not None
        else None,
        "synthetic_tail_path_coverage": path_stats,
        "output": str(output_path),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )
    (args.output_dir / "complete_case_attrition.json").write_text(
        json.dumps(complete_case_attrition, indent=2, default=str),
        encoding="utf-8",
    )
    print(hour_counts.to_string(index=False))
    print(json.dumps(manifest, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
