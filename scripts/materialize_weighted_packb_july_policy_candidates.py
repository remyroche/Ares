#!/usr/bin/env python3
"""Materialize frozen weighted Pack-B July rows for fixed-EV policy replay.

The only fitted objects in this script are monotone EV maps trained on the
explicitly supplied, strictly pre-July reference.  July prediction rows are
never used to fit ranks, curves, thresholds, or support statistics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_FIXED_TARGET_NET_EV = 0.007
DEFAULT_MIN_SIDE_ROWS = 240
DEFAULT_MIN_LOCAL_ROWS = 80
DEFAULT_SIDE_SHRINK_ROWS = 960.0
DEFAULT_LOCAL_SHRINK_ROWS = 240.0
DEFAULT_LOCAL_WEIGHT_CAP = 0.85

SCORE_CANDIDATES = (
    "score_meta_base_soft_label",
    "meta_score_oof",
    "score_meta",
    "score",
)
TARGET_CANDIDATES = (
    "ev_after_1pct",
    "__u_policy_net__",
    "__first_touch_capture_net__",
    "u_policy_net",
)
TIMESTAMP_CANDIDATES = ("__ts__", "timestamp", "signal_bar_ts")
SYMBOL_CANDIDATES = ("__symbol__", "symbol")
SIDE_CANDIDATES = ("side_name", "side")
ARCHETYPE_FALLBACK = (
    "archetype_label_family",
    "policy_archetype",
    "local_side_archetype",
    "source_archetype",
)
ARCHETYPE_ALIASES = {
    "archetype_label_family": "__archetype_label_family__",
    "policy_archetype": "__archetype_policy_key__",
}
CONTEXT_COLUMNS = (
    "candidate_id",
    "strategy_id",
    "archetype_label_family",
    "__archetype_label_family__",
    "policy_archetype",
    "local_side_archetype",
    "source_archetype",
    "archetype_policy_key",
    "__archetype_policy_key__",
    "base_rank_pct_timestamp_side",
    "base_rank_within_timestamp_side",
    "selected_top30",
    "base_cutoff_score_timestamp_side",
    "__barrier_pct__",
    "barrier_pct",
    "median_spread_bps",
    "expected_spread_bps",
    "policy_spread_bps",
    "spread_bps",
    "gmm_cluster_id",
    "gmm_entropy",
    "cluster_entropy_norm",
    "gmm_ood_score",
    "meta_hit_probability_uncertainty_p1mp",
    "mahalanobis_distance",
    "AE_reconstruction_error",
    "ae_reconstruction_error",
    "dae_reconstruction_error",
    "latent_speed",
    "latent_acceleration",
    "policy_sl_mult",
    "policy_trailing_activation_mult",
    "policy_trailing_power",
    "policy_trailing_squash_divisor",
    "policy_giveback_beta",
    "policy_target_holding_hours",
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parquet_columns(path: Path) -> set[str]:
    try:
        import pyarrow.parquet as pq

        return set(pq.ParquetFile(path).schema.names)
    except ImportError:
        return set(pd.read_parquet(path).columns)


def _first_present(columns: Iterable[str], available: set[str], *, role: str) -> str:
    for column in columns:
        if column in available:
            return column
    raise ValueError(f"{role} lacks one of: {', '.join(columns)}")


def _read_compact(path: Path, *, reference: bool) -> tuple[pd.DataFrame, dict[str, str]]:
    available = _parquet_columns(path)
    timestamp = _first_present(TIMESTAMP_CANDIDATES, available, role=str(path))
    symbol = _first_present(SYMBOL_CANDIDATES, available, role=str(path))
    side = _first_present(SIDE_CANDIDATES, available, role=str(path))
    score = _first_present(SCORE_CANDIDATES, available, role=str(path))
    target = _first_present(TARGET_CANDIDATES, available, role=str(path)) if reference else ""
    columns = {
        timestamp,
        symbol,
        side,
        score,
        *[column for column in ARCHETYPE_FALLBACK if column in available],
        *[alias for alias in ARCHETYPE_ALIASES.values() if alias in available],
        *([] if reference else [column for column in CONTEXT_COLUMNS if column in available]),
    }
    if reference:
        columns.add(target)
        for column in (
            "__first_touch_round_trip_cost__",
            "first_touch_round_trip_cost",
            "round_trip_cost",
            "first_touch_gross",
            "gross_return",
        ):
            if column in available:
                columns.add(column)
    frame = pd.read_parquet(path, columns=sorted(columns))
    return frame, {
        "timestamp": timestamp,
        "symbol": symbol,
        "side": side,
        "score": score,
        "target": target,
    }


def _valid_text(values: pd.Series) -> pd.Series:
    text = values.astype("string").str.strip()
    return text.notna() & ~text.str.lower().isin(("", "nan", "none", "null", "missing"))


def _normalise_side(values: pd.Series) -> pd.Series:
    text = values.astype("string").str.lower().str.strip()
    numeric = pd.to_numeric(values, errors="coerce")
    result = pd.Series(pd.NA, index=values.index, dtype="string")
    result = result.mask(numeric < 0.0, "short")
    result = result.mask(numeric > 0.0, "long")
    result = result.mask(text.isin(("short", "-1", "-1.0")), "short")
    result = result.mask(text.isin(("long", "1", "1.0")), "long")
    invalid = ~result.isin(("long", "short"))
    if bool(invalid.any()):
        raise ValueError(f"unsupported side values: {sorted(text.loc[invalid].dropna().unique())[:10]}")
    return result.astype("string")


def _base_archetype(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    value = pd.Series(pd.NA, index=frame.index, dtype="string")
    source = pd.Series("missing", index=frame.index, dtype="string")
    for column in ARCHETYPE_FALLBACK:
        actual = column if column in frame.columns else ARCHETYPE_ALIASES.get(column)
        if actual not in frame.columns:
            continue
        candidate = frame[actual].astype("string").str.strip()
        use = value.isna() & _valid_text(candidate)
        value = value.mask(use, candidate)
        source = source.mask(use, column)
    value = value.fillna("missing")
    # Policy labels sometimes include the side in the value.  Keep the base
    # identity side-independent because side is a separate hierarchy level.
    for side in ("long", "short"):
        value = value.str.removeprefix(f"{side}__")
    return value.astype("string"), source.astype("string")


def _normalise_frame(frame: pd.DataFrame, columns: Mapping[str, str], *, reference: bool) -> pd.DataFrame:
    out = frame.copy()
    out["timestamp"] = pd.to_datetime(out[columns["timestamp"]], utc=True, errors="coerce")
    out["symbol"] = out[columns["symbol"]].astype("string").str.strip()
    out["side_name"] = _normalise_side(out[columns["side"]])
    out["raw_meta_score"] = pd.to_numeric(out[columns["score"]], errors="coerce")
    out["base_archetype"], out["base_archetype_source"] = _base_archetype(out)
    if reference:
        out["target_ev_after_1pct"] = pd.to_numeric(out[columns["target"]], errors="coerce")
    valid = (
        out["timestamp"].notna()
        & out["symbol"].notna()
        & out["symbol"].ne("")
        & np.isfinite(out["raw_meta_score"])
    )
    if reference:
        valid &= np.isfinite(out["target_ev_after_1pct"])
    out = out.loc[valid].copy()
    if out.empty:
        raise ValueError("no finite rows remain after input normalization")
    keys = ["timestamp", "symbol", "side_name"]
    if bool(out.duplicated(keys).any()):
        raise ValueError(f"duplicate timestamp/symbol/side rows: {int(out.duplicated(keys).sum())}")
    return out.sort_values(keys, kind="stable").reset_index(drop=True)


def _assert_fee_contract(reference: pd.DataFrame, target_col: str) -> str:
    cost_cols = (
        "__first_touch_round_trip_cost__",
        "first_touch_round_trip_cost",
        "round_trip_cost",
    )
    for column in cost_cols:
        if column not in reference.columns:
            continue
        values = pd.to_numeric(reference[column], errors="coerce").dropna()
        if values.empty:
            continue
        if not np.allclose(values.to_numpy(dtype=np.float64), 0.01, rtol=0.0, atol=1e-6):
            raise ValueError(f"{column} must be exactly 0.01; target EV must include one 1% fee")
        return f"verified:{column}=0.01"
    if "after_1pct" in target_col:
        return f"declared_by_target_name:{target_col}"
    raise ValueError(
        "cannot verify the 1% embedded-fee contract; provide ev_after_1pct or a "
        "reference with __first_touch_round_trip_cost__ == 0.01"
    )


def _fit_iso(score: np.ndarray, target: np.ndarray) -> IsotonicRegression:
    model = IsotonicRegression(increasing=True, out_of_bounds="clip", y_min=-0.25, y_max=0.25)
    q80, q90 = np.quantile(score, (0.80, 0.90))
    weights = np.where(score >= q90, 4.0, np.where(score >= q80, 2.0, 0.25))
    model.fit(score, target, sample_weight=weights)
    return model


def _curve(model: IsotonicRegression) -> dict[str, list[float]]:
    return {
        "x": np.asarray(model.X_thresholds_, dtype=np.float64).tolist(),
        "y": np.asarray(model.y_thresholds_, dtype=np.float64).tolist(),
    }


def _fit_mapping(
    reference: pd.DataFrame,
    *,
    min_side_rows: int,
    min_local_rows: int,
    side_shrink_rows: float,
    local_shrink_rows: float,
    local_weight_cap: float,
) -> dict[str, Any]:
    score = reference["raw_meta_score"].to_numpy(dtype=np.float64, copy=False)
    target = reference["target_ev_after_1pct"].to_numpy(dtype=np.float64, copy=False)
    if len(reference) < 100 or np.unique(score).size < 8:
        raise ValueError("calibration reference needs at least 100 rows and 8 distinct scores")
    global_model = _fit_iso(score, target)
    side_models: dict[str, IsotonicRegression] = {}
    side_meta: dict[str, dict[str, Any]] = {}
    for side, group in reference.groupby("side_name", sort=True, observed=True):
        x = group["raw_meta_score"].to_numpy(dtype=np.float64, copy=False)
        y = group["target_ev_after_1pct"].to_numpy(dtype=np.float64, copy=False)
        support = int(len(group))
        if support >= min_side_rows and np.unique(x).size >= 8:
            model = _fit_iso(x, y)
            side_models[str(side)] = model
            weight = min(float(local_weight_cap), support / (support + float(side_shrink_rows)))
            side_meta[str(side)] = {"support": support, "weight": weight, **_curve(model)}
        else:
            side_meta[str(side)] = {"support": support, "weight": 0.0, "fallback": "global"}

    def side_expected(side_values: np.ndarray, values: np.ndarray) -> np.ndarray:
        result = np.asarray(global_model.predict(values), dtype=np.float64)
        for side, model in side_models.items():
            mask = side_values == side
            if mask.any():
                weight = float(side_meta[side]["weight"])
                result[mask] = (1.0 - weight) * result[mask] + weight * model.predict(values[mask])
        return result

    local_models: dict[tuple[str, str], IsotonicRegression] = {}
    local_meta: dict[str, dict[str, Any]] = {}
    for (side, archetype), group in reference.groupby(["side_name", "base_archetype"], sort=True, observed=True):
        x = group["raw_meta_score"].to_numpy(dtype=np.float64, copy=False)
        y = group["target_ev_after_1pct"].to_numpy(dtype=np.float64, copy=False)
        support = int(len(group))
        key = (str(side), str(archetype))
        payload: dict[str, Any] = {"support": support}
        if support >= min_local_rows and np.unique(x).size >= 8:
            model = _fit_iso(x, y)
            local_models[key] = model
            payload.update(
                weight=min(float(local_weight_cap), support / (support + float(local_shrink_rows))),
                **_curve(model),
            )
        else:
            payload.update(weight=0.0, fallback="side")
        local_meta[f"{key[0]}||{key[1]}"] = payload

    sides = reference["side_name"].astype(str).to_numpy(copy=False)
    archetypes = reference["base_archetype"].astype(str).to_numpy(copy=False)
    mapped = side_expected(sides, score)
    for key, model in local_models.items():
        mask = (sides == key[0]) & (archetypes == key[1])
        if mask.any():
            weight = float(local_meta[f"{key[0]}||{key[1]}"]["weight"])
            mapped[mask] = (1.0 - weight) * mapped[mask] + weight * model.predict(score[mask])
    raw_reference = np.sort(score.astype(np.float32, copy=False))
    ev_reference = np.sort(mapped.astype(np.float32, copy=False))
    return {
        "schema": "weighted_packb_hierarchical_monotonic_ev_v1",
        "unit": "net_return_after_1pct",
        "hierarchy": "global -> side -> side_x_base_archetype",
        "global": _curve(global_model),
        "side": side_meta,
        "side_x_base_archetype": local_meta,
        "raw_score_rank_reference": raw_reference.tolist(),
        "mapped_expected_ev_rank_reference": ev_reference.tolist(),
        "min_side_rows": int(min_side_rows),
        "min_local_rows": int(min_local_rows),
        "side_shrink_rows": float(side_shrink_rows),
        "local_shrink_rows": float(local_shrink_rows),
        "local_weight_cap": float(local_weight_cap),
    }


def _apply_mapping(rows: pd.DataFrame, mapping: Mapping[str, Any]) -> pd.DataFrame:
    out = rows.copy()
    score = out["raw_meta_score"].to_numpy(dtype=np.float64, copy=False)
    global_curve = mapping["global"]
    global_ev = np.interp(score, global_curve["x"], global_curve["y"])
    mapped = global_ev.copy()
    side_support = np.zeros(len(out), dtype=np.int32)
    side_weight = np.zeros(len(out), dtype=np.float32)
    local_support = np.zeros(len(out), dtype=np.int32)
    local_weight = np.zeros(len(out), dtype=np.float32)
    scope = np.full(len(out), "global", dtype=object)
    sides = out["side_name"].astype(str).to_numpy(copy=False)
    archetypes = out["base_archetype"].astype(str).to_numpy(copy=False)
    for side, curve in dict(mapping["side"]).items():
        mask = sides == str(side)
        if not mask.any():
            continue
        support = int(curve.get("support", 0))
        weight = float(curve.get("weight", 0.0))
        side_support[mask] = support
        side_weight[mask] = weight
        if weight > 0.0 and curve.get("x"):
            local = np.interp(score[mask], curve["x"], curve["y"])
            mapped[mask] = (1.0 - weight) * mapped[mask] + weight * local
            scope[mask] = "side"
    for joined, curve in dict(mapping["side_x_base_archetype"]).items():
        side, archetype = str(joined).split("||", 1)
        mask = (sides == side) & (archetypes == archetype)
        if not mask.any():
            continue
        support = int(curve.get("support", 0))
        weight = float(curve.get("weight", 0.0))
        local_support[mask] = support
        local_weight[mask] = weight
        if weight > 0.0 and curve.get("x"):
            local = np.interp(score[mask], curve["x"], curve["y"])
            mapped[mask] = (1.0 - weight) * mapped[mask] + weight * local
            scope[mask] = "side_x_base_archetype"
    raw_reference = np.asarray(mapping["raw_score_rank_reference"], dtype=np.float32)
    ev_reference = np.asarray(mapping["mapped_expected_ev_rank_reference"], dtype=np.float32)
    out["raw_meta_score_rank"] = (
        np.searchsorted(raw_reference, score, side="right") / float(len(raw_reference))
    ).astype(np.float32)
    out["mapped_expected_ev"] = mapped.astype(np.float32)
    out["mapped_expected_ev_rank"] = (
        np.searchsorted(ev_reference, mapped, side="right") / float(len(ev_reference))
    ).astype(np.float32)
    out["mapped_expected_ev_global"] = global_ev.astype(np.float32)
    out["ev_map_scope"] = pd.Series(scope, index=out.index, dtype="string")
    out["ev_map_side_support"] = side_support
    out["ev_map_side_weight"] = side_weight
    out["ev_map_local_support"] = local_support
    out["ev_map_local_weight"] = local_weight
    return out


def _coalesce_numeric(frame: pd.DataFrame, columns: Iterable[str]) -> pd.Series:
    result = pd.Series(np.nan, index=frame.index, dtype="float64")
    for column in columns:
        if column in frame.columns:
            result = result.where(result.notna(), pd.to_numeric(frame[column], errors="coerce"))
    return result


def _global_top_fraction_mask(values: pd.Series, fraction: float) -> pd.Series:
    """Return a stable pooled top-fraction mask for OOS diagnostics only."""
    numeric = pd.to_numeric(values, errors="coerce")
    valid = numeric.dropna()
    selected = pd.Series(False, index=values.index)
    if valid.empty:
        return selected
    count = max(1, int(math.ceil(len(valid) * float(fraction))))
    chosen = valid.sort_values(ascending=False, kind="mergesort").index[:count]
    selected.loc[chosen] = True
    return selected


def _candidate_table(rows: pd.DataFrame, *, fixed_target_net_ev: float, mapping: Mapping[str, Any]) -> pd.DataFrame:
    out = rows.copy()
    admitted = out["mapped_expected_ev"].ge(float(fixed_target_net_ev))
    # These two masks are retrospective OOS layer diagnostics.  They must be
    # the actual pooled top 10% of this comparison scope, not a >=0.90 cutoff
    # against the pre-July rank reference.  Deployable admission remains the
    # fixed, train-derived EV threshold below.
    out["raw_global_top10_selected"] = _global_top_fraction_mask(
        out["raw_meta_score"], 0.10
    )
    out["ev_mapped_global_top10_selected"] = _global_top_fraction_mask(
        out["mapped_expected_ev"], 0.10
    )
    eligible_reference = np.asarray(mapping["mapped_expected_ev_rank_reference"], dtype=np.float32)
    eligible_reference = eligible_reference[eligible_reference >= float(fixed_target_net_ev)]
    # The current fixed-EV admission contract maps admitted rows into the
    # portfolio's top-10 rank band without imposing a new top-k quota.
    admission_rank = np.zeros(len(out), dtype=np.float32)
    if eligible_reference.size:
        values = out.loc[admitted, "mapped_expected_ev"].to_numpy(dtype=np.float64, copy=False)
        conditional = np.searchsorted(eligible_reference, values, side="right") / float(len(eligible_reference))
        admission_rank[admitted.to_numpy()] = (0.90 + 0.10 * conditional).astype(np.float32)
    else:
        admission_rank[admitted.to_numpy()] = np.float32(0.90)
    out["policy_admitted_before_portfolio"] = admitted.astype(bool)
    out["replay_union_selected"] = (
        out["raw_global_top10_selected"]
        | out["ev_mapped_global_top10_selected"]
        | out["policy_admitted_before_portfolio"]
    )
    out["policy_admission_reason"] = np.where(admitted, "fixed_net_ev_ge_0.007", "below_fixed_net_ev")
    out["policy_fixed_target_net_ev"] = np.float32(fixed_target_net_ev)
    out["policy_admission_rank"] = admission_rank
    # Canonical policy/replay aliases retain all four layers without asking a
    # downstream caller to reinterpret the Pack-B-specific names.
    out["expected_net_ev_after_1pct"] = out["mapped_expected_ev"].astype(np.float32)
    out["expected_net_ev_after_1pct_side_archetype"] = out["mapped_expected_ev"].astype(np.float32)
    out["expected_ev_rank_score"] = out["mapped_expected_ev_rank"].astype(np.float32)
    out["threshold_basis_selected"] = admitted.astype(bool)
    out["threshold_basis_corrected_expected_ev"] = out["mapped_expected_ev"].astype(np.float32)
    out["threshold_basis_corrected_expected_ev_rank"] = admission_rank
    out["threshold_basis_rank_score"] = admission_rank
    out["threshold_basis_reason"] = out["policy_admission_reason"].astype("string")
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    out["side"] = np.where(out["side_name"].eq("short"), -1.0, 1.0).astype(np.float32)
    out["strategy_id"] = out["side_name"].astype(str) + "_weighted_packb_meta_policy"
    out["calibrated_score"] = out["mapped_expected_ev"].astype(np.float32)
    out["rank_pct"] = admission_rank
    out["normalized_rank_score"] = admission_rank
    out["strategy_rank_pct"] = admission_rank
    out["base_strategy_threshold"] = np.float32(0.90)
    out["barrier_pct"] = _coalesce_numeric(out, ("barrier_pct", "__barrier_pct__")).astype(np.float32)
    spread = _coalesce_numeric(out, ("expected_spread_bps", "policy_spread_bps", "median_spread_bps", "spread_bps"))
    out["expected_spread_bps"] = spread.astype(np.float32)
    out["policy_spread_bps"] = spread.astype(np.float32)
    out["expected_half_spread_bps"] = (spread / 2.0).astype(np.float32)
    out["spread_cost_bps"] = (spread / 2.0).astype(np.float32)
    out["exit_quote_half_spread_bps"] = (spread / 2.0).astype(np.float32)
    out["exit_spread_cost_bps"] = (spread / 2.0).astype(np.float32)
    out["target_ev_embedded_round_trip_fee_bps"] = np.float32(100.0)
    out["target_ev_includes_spread"] = False
    out["policy_archetype"] = out["side_name"].astype(str) + "__" + out["base_archetype"].astype(str)
    out["local_side_archetype"] = out["policy_archetype"]
    out["archetype_policy_key"] = out["base_archetype"].astype("string")
    return out


def _downcast(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in out.select_dtypes(include=["float64"]).columns:
        out[column] = out[column].astype(np.float32)
    for column in out.select_dtypes(include=["int64"]).columns:
        out[column] = pd.to_numeric(out[column], downcast="integer")
    for column in out.select_dtypes(include=["object"]).columns:
        out[column] = out[column].astype("string")
    return out


def materialize(
    *,
    predictions_path: Path,
    calibration_reference_path: Path,
    output_dir: Path,
    fixed_target_net_ev: float = DEFAULT_FIXED_TARGET_NET_EV,
    min_side_rows: int = DEFAULT_MIN_SIDE_ROWS,
    min_local_rows: int = DEFAULT_MIN_LOCAL_ROWS,
    side_shrink_rows: float = DEFAULT_SIDE_SHRINK_ROWS,
    local_shrink_rows: float = DEFAULT_LOCAL_SHRINK_ROWS,
    local_weight_cap: float = DEFAULT_LOCAL_WEIGHT_CAP,
) -> dict[str, Any]:
    """Write July candidates, a frozen map, and provenance without replaying them."""
    predictions_path = predictions_path.resolve()
    calibration_reference_path = calibration_reference_path.resolve()
    prediction_raw, prediction_cols = _read_compact(predictions_path, reference=False)
    reference_raw, reference_cols = _read_compact(calibration_reference_path, reference=True)
    predictions = _normalise_frame(prediction_raw, prediction_cols, reference=False)
    reference = _normalise_frame(reference_raw, reference_cols, reference=True)
    fee_verification = _assert_fee_contract(reference, reference_cols["target"])
    if not reference["timestamp"].max() < predictions["timestamp"].min():
        raise ValueError(
            "calibration reference must be strictly before frozen July predictions: "
            f"reference_end={reference['timestamp'].max()}, prediction_start={predictions['timestamp'].min()}"
        )
    mapping = _fit_mapping(
        reference,
        min_side_rows=int(min_side_rows),
        min_local_rows=int(min_local_rows),
        side_shrink_rows=float(side_shrink_rows),
        local_shrink_rows=float(local_shrink_rows),
        local_weight_cap=float(local_weight_cap),
    )
    mapped = _apply_mapping(predictions, mapping)
    candidates = _downcast(_candidate_table(mapped, fixed_target_net_ev=float(fixed_target_net_ev), mapping=mapping))
    candidates = candidates.sort_values(["timestamp", "symbol", "side_name"], kind="stable").reset_index(drop=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = output_dir / "weighted_packb_july_policy_candidates.parquet"
    mapping_path = output_dir / "weighted_packb_july_hierarchical_ev_map.json"
    manifest_path = output_dir / "manifest.json"
    candidates.to_parquet(candidate_path, index=False, compression="zstd")
    mapping_path.write_text(json.dumps(_json_safe(mapping), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "schema": "weighted_packb_july_policy_candidates_v1",
        "predictions": str(predictions_path),
        "predictions_sha256": _sha256(predictions_path),
        "calibration_reference": str(calibration_reference_path),
        "calibration_reference_sha256": _sha256(calibration_reference_path),
        "output_candidates": str(candidate_path),
        "ev_mapping": str(mapping_path),
        "prediction_score_column": prediction_cols["score"],
        "reference_score_column": reference_cols["score"],
        "reference_target_column": reference_cols["target"],
        "reference_rows": int(len(reference)),
        "reference_start": reference["timestamp"].min(),
        "reference_end": reference["timestamp"].max(),
        "prediction_start": predictions["timestamp"].min(),
        "prediction_end": predictions["timestamp"].max(),
        "rows": int(len(candidates)),
        "policy_admitted_before_portfolio_rows": int(candidates["policy_admitted_before_portfolio"].sum()),
        "fixed_admission_contract": {
            "selection_mode": "fixed_corrected_ev_threshold",
            "fixed_target_net_ev": float(fixed_target_net_ev),
            "portfolio_rank_band": "[0.90, 1.00] for admitted rows",
        },
        "archetype_fallback_order": list(ARCHETYPE_FALLBACK),
        "cost_contract": {
            "target_ev": "already net of exactly one 1% round-trip fee",
            "fee_verification": fee_verification,
            "fee_deducted_by_materializer": False,
            "spread": "preserved only as separate entry/exit half-spread execution context",
        },
        "leakage_contract": "mapping, raw-score ranks, mapped-EV ranks, and support statistics are fitted only on the strictly pre-July reference",
    }
    manifest_path.write_text(json.dumps(_json_safe(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True, help="Frozen weighted Pack-B July meta OOS predictions.")
    parser.add_argument("--calibration-reference", type=Path, required=True, help="Resolved, strictly pre-July OOF reference with net EV after the 1%% fee.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fixed-target-net-ev", type=float, default=DEFAULT_FIXED_TARGET_NET_EV)
    parser.add_argument("--min-side-rows", type=int, default=DEFAULT_MIN_SIDE_ROWS)
    parser.add_argument("--min-local-rows", type=int, default=DEFAULT_MIN_LOCAL_ROWS)
    parser.add_argument("--side-shrink-rows", type=float, default=DEFAULT_SIDE_SHRINK_ROWS)
    parser.add_argument("--local-shrink-rows", type=float, default=DEFAULT_LOCAL_SHRINK_ROWS)
    parser.add_argument("--local-weight-cap", type=float, default=DEFAULT_LOCAL_WEIGHT_CAP)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = materialize(
        predictions_path=args.predictions,
        calibration_reference_path=args.calibration_reference,
        output_dir=args.output_dir,
        fixed_target_net_ev=args.fixed_target_net_ev,
        min_side_rows=args.min_side_rows,
        min_local_rows=args.min_local_rows,
        side_shrink_rows=args.side_shrink_rows,
        local_shrink_rows=args.local_shrink_rows,
        local_weight_cap=args.local_weight_cap,
    )
    print(json.dumps(_json_safe(manifest), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
