#!/usr/bin/env python3
"""Forward-test executable-quality targets and causal trajectory features.

This is intentionally a compact complement to the AE/GMM state-discovery
runner.  It tests whether the two economic distinctions that matter to the
policy are learnable before they are promoted into a larger state model:

* ``correct_direction``: the first-touch/clean direction succeeds;
* ``good_trade``: it remains positive after costs without adverse-path damage.

For each OOS month, shallow side x archetype heads are fitted only on prior
rows in the train-fitted global top-20 score population.  They re-rank only
the same OOS top-20 pool and retain exactly half of it, preserving the original
top-10 activity budget.  The script never uses realized OOS fields as inputs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.residual_event_archetypes import (  # noqa: E402
    RESIDUAL_EVENT_PREFIX,
    RESIDUAL_EVENT_TRAJECTORY_SUFFIXES,
    ResidualEventArchetypeConfig,
    _TRAJECTORY_SOURCE_ALIASES,
    _executable_quality_targets,
    add_residual_event_temporal_context,
)
from extreme_price_movements.static_feature_store import read_static_features  # noqa: E402

DEFAULT_ROOT = ROOT / (
    "data_perp/reports/train_meta_residual_archetype_enhancement_20260711_v1/"
    "champion_frozen_single_source_202501_20260710/candidate_shards"
)
DEFAULT_OUTPUT = ROOT / "data_perp/reports/executable_quality_transition_ablation_20260719_v1"
DEFAULT_FEATURE_STORE_ID = "20260711_070000"
KEYS = ("__ts__", "__symbol__", "side_name", "archetype_policy_key")
OUTCOMES = (
    "ev_after_1pct",
    "exec_margin",
    "clean_exec",
    "dirty_positive",
    "first_touch_bad_mae_1r",
    "full_path_bad_mae_1r",
    "timeout",
)

# The existing candidate stream already carries these causal lifecycle/GMM
# values.  The added transition block is calculated from them without looking
# at outcomes or recent performance.
FEATURE_HINTS = (
    "gmm_",
    "dae_",
    "mahal",
    "reconstruction",
    "oi_",
    "funding",
    "breadth",
    "liquid",
    "cover",
    "delever",
    "flush",
    "vol",
    "shock",
    "ret",
    "dislocation",
    "spectral",
    "dispersion",
    "correlation",
    "breakout",
    "pullback",
    "support",
)
STRUCTURAL = {
    *KEYS,
    "row_id",
    "source_tag",
    "side",
    "selected_top30",
    "score",
    *OUTCOMES,
}


def _safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (pd.Timestamp, Path)):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_safe(payload), indent=2, sort_keys=True), encoding="utf-8")


def _parse_months(value: str) -> list[pd.Period]:
    result = [pd.Period(item.strip(), freq="M") for item in value.split(",") if item.strip()]
    if not result:
        raise ValueError("at least one OOS month is required")
    return result


def _load(root: Path, end: pd.Timestamp) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted(root.glob("candidates_*.parquet")):
        month = pd.Period(path.stem.removeprefix("candidates_"), freq="M")
        if pd.Timestamp(month.start_time, tz="UTC") >= end:
            continue
        columns = pq.ParquetFile(path).schema_arrow.names
        wanted = [name for name in columns if name not in {"__first_touch_target_soft__", "target_soft"}]
        part = pd.read_parquet(path, columns=wanted)
        part["__ts__"] = pd.to_datetime(part["__ts__"], utc=True, errors="coerce")
        part = part.loc[part["__ts__"].notna()].copy()
        frames.append(part)
    if not frames:
        raise FileNotFoundError(f"No candidate shards before {end} under {root}")
    data = pd.concat(frames, ignore_index=True, copy=False)
    data = data.sort_values(["__ts__", "__symbol__", "side_name"], kind="stable")
    data = data.drop_duplicates(list(KEYS), keep="last").reset_index(drop=True)
    for name in data.select_dtypes(include=["float64"]).columns:
        data[name] = pd.to_numeric(data[name], downcast="float")
    return data


def _feature_columns(frame: pd.DataFrame) -> list[str]:
    chosen = ["score"]
    for name in frame.columns:
        lower = str(name).lower()
        if name in STRUCTURAL or lower.startswith(("resid_target_", "resid_event_")):
            continue
        if not pd.api.types.is_numeric_dtype(frame[name]):
            continue
        if any(token in lower for token in FEATURE_HINTS):
            if float(pd.to_numeric(frame[name], errors="coerce").notna().mean()) >= 0.70:
                chosen.append(str(name))
    return list(dict.fromkeys(chosen))


def _transition_source_keys() -> list[str]:
    """Return every portable source accepted by the production transition block."""

    return list(
        dict.fromkeys(
            source
            for aliases in _TRAJECTORY_SOURCE_ALIASES.values()
            for source in aliases
        )
    )


def _hydrate_transition_sources(
    frame: pd.DataFrame,
    *,
    feature_store_id: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Rehydrate lifecycle sources from the canonical read-only feature store.

    Candidate ledgers intentionally remain compact, but several newer market
    lifecycle fields were absent when the frozen shards were written.  This
    reads exactly those observable columns through ``read_static_features`` -
    the same endpoint used by training, replay, and inference - and aligns on
    UTC timestamp plus symbol.  It never materializes outcomes or derives a
    fallback from future rows.
    """

    source_keys = _transition_source_keys()
    out = frame.copy(deep=False)
    requested = pd.MultiIndex.from_frame(out.loc[:, ["__ts__", "__symbol__"]])
    symbols = sorted(out["__symbol__"].astype(str).dropna().unique().tolist())
    start_ts = pd.Timestamp(out["__ts__"].min())
    end_ts = pd.Timestamp(out["__ts__"].max())
    store_ts = pd.to_datetime(str(feature_store_id), format="%Y%m%d_%H%M%S", utc=True)
    loaded = read_static_features(
        feature_store_ts=store_ts,
        data_root=ROOT / "data_perp",
        feature_keys=source_keys,
        symbols=symbols,
        start_ts=start_ts,
        end_ts=end_ts,
        output_layout="panels",
    )
    coverage: dict[str, float] = {}
    replaced: list[str] = []
    if loaded is None or not hasattr(loaded, "get"):
        raise RuntimeError(f"Canonical static feature store {feature_store_id} returned no panels")
    for key in source_keys:
        panel = loaded.get(key)
        if not isinstance(panel, pd.DataFrame) or panel.empty:
            coverage[key] = 0.0
            continue
        panel = panel.copy(deep=False)
        panel.index = pd.to_datetime(panel.index, utc=True, errors="coerce")
        # The store preserves the canonical symbols, but coercing here makes
        # joins robust to Parquet categorical/object restoration.
        panel.columns = panel.columns.astype(str)
        values = panel.stack(dropna=False)
        values.index = values.index.set_names(["__ts__", "__symbol__"])
        values = values.reindex(requested)
        candidate_values = pd.to_numeric(out.get(key), errors="coerce") if key in out else None
        if candidate_values is None:
            final_values = values.to_numpy(dtype=np.float32, copy=False)
        else:
            final_values = candidate_values.to_numpy(dtype=np.float32, copy=True)
            missing = ~np.isfinite(final_values)
            static_values = values.to_numpy(dtype=np.float32, copy=False)
            final_values[missing] = static_values[missing]
        out[key] = final_values
        coverage[key] = float(np.isfinite(final_values).mean())
        if candidate_values is None or float(candidate_values.notna().mean()) < coverage[key]:
            replaced.append(key)
    mechanism_sources = {
        mechanism: next(
            (key for key in aliases if coverage.get(key, 0.0) >= 0.70),
            None,
        )
        for mechanism, aliases in _TRAJECTORY_SOURCE_ALIASES.items()
    }
    absent = [mechanism for mechanism, key in mechanism_sources.items() if key is None]
    if absent:
        raise RuntimeError(
            "Canonical feature store lacks enough coverage for transition mechanisms: "
            + ", ".join(absent)
        )
    return out, {
        "feature_store_id": str(feature_store_id),
        "source_coverage": coverage,
        "rehydrated_sources": replaced,
        "mechanism_sources": mechanism_sources,
        "requested_rows": int(len(out)),
    }


def _add_trajectory_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Use the production causal transition builder without outcome inputs."""

    generated = pd.DataFrame(index=frame.index)
    for index in range(7):
        generated[f"{RESIDUAL_EVENT_PREFIX}gmm_cluster_posterior_{index}"] = 0.0
    generated[f"{RESIDUAL_EVENT_PREFIX}gmm_ood_score"] = 0.0
    generated[f"{RESIDUAL_EVENT_PREFIX}dae_reconstruction_error_zscore"] = 0.0
    observable = frame.drop(columns=[name for name in OUTCOMES if name in frame], errors="ignore")
    context = add_residual_event_temporal_context(
        generated,
        observable,
        ResidualEventArchetypeConfig(enable_executable_quality_targets=False),
    )
    columns = [
        f"{RESIDUAL_EVENT_PREFIX}{suffix}"
        for suffix in RESIDUAL_EVENT_TRAJECTORY_SUFFIXES
        if f"{RESIDUAL_EVENT_PREFIX}{suffix}" in context
    ]
    columns.extend(
        name
        for name in (
            f"{RESIDUAL_EVENT_PREFIX}posterior_switch_pressure",
            f"{RESIDUAL_EVENT_PREFIX}posterior_entropy_delta",
        )
        if name in context
    )
    out = frame.copy(deep=False)
    for name in columns:
        out[name] = pd.to_numeric(context[name], errors="coerce").fillna(0.0).astype(np.float32)
    return out, columns


def _tail_mask(frame: pd.DataFrame, cutoff: float) -> np.ndarray:
    return pd.to_numeric(frame["score"], errors="coerce").fillna(-np.inf).to_numpy() >= cutoff


def _local_key(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["side_name"].astype(str).str.lower()
        + "|"
        + frame["archetype_policy_key"].astype(str)
    )


def _fit_score_value_map(
    reference: pd.DataFrame,
    *,
    value_col: str,
    bins: int = 8,
) -> dict[str, Any]:
    """Fit a shrunk score-to-outcome map from resolved history only."""

    score = pd.to_numeric(reference["score"], errors="coerce").to_numpy(dtype=np.float32)
    value = pd.to_numeric(reference[value_col], errors="coerce").to_numpy(dtype=np.float32)
    valid = np.isfinite(score) & np.isfinite(value)
    if int(valid.sum()) < 500:
        return {"edges": np.array([0.0, 1.0], dtype=np.float32), "global": np.zeros(1, dtype=np.float32), "counts": np.zeros(1, dtype=np.float32), "local": {}}
    edges = np.unique(np.nanquantile(score[valid], np.linspace(0.0, 1.0, bins + 1))).astype(np.float32)
    if len(edges) < 3:
        edges = np.array([float(np.nanmin(score[valid])) - 1e-6, float(np.nanmax(score[valid])) + 1e-6], dtype=np.float32)
    bucket = np.clip(np.searchsorted(edges[1:-1], score, side="right"), 0, len(edges) - 2)
    n_bins = len(edges) - 1
    counts = np.bincount(bucket[valid], minlength=n_bins).astype(np.float32)
    sums = np.bincount(bucket[valid], weights=value[valid], minlength=n_bins).astype(np.float64)
    global_mean = float(np.nanmean(value[valid]))
    global_rates = ((sums + 120.0 * global_mean) / np.maximum(counts + 120.0, 1.0)).astype(np.float32)
    local: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    keys = _local_key(reference).to_numpy(dtype=object)
    for key in np.unique(keys[valid]):
        mask = valid & (keys == key)
        if int(mask.sum()) < 80:
            continue
        c = np.bincount(bucket[mask], minlength=n_bins).astype(np.float32)
        s = np.bincount(bucket[mask], weights=value[mask], minlength=n_bins).astype(np.float64)
        local[str(key)] = (s.astype(np.float32), c)
    return {"edges": edges, "global": global_rates, "counts": counts, "local": local}


def _expected_ev(frame: pd.DataFrame, state: dict[str, Any]) -> np.ndarray:
    score = pd.to_numeric(frame["score"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    edges = np.asarray(state["edges"], dtype=np.float32)
    global_rates = np.asarray(state["global"], dtype=np.float32)
    bucket = np.clip(np.searchsorted(edges[1:-1], score, side="right"), 0, len(global_rates) - 1)
    out = global_rates[bucket].astype(np.float32, copy=True)
    for key, positions in _local_key(frame).groupby(_local_key(frame), sort=False).groups.items():
        values = state["local"].get(str(key))
        if values is None:
            continue
        sums, counts = values
        pos = np.asarray(positions, dtype=np.int64)
        b = bucket[pos]
        local_rate = sums[b] / np.maximum(counts[b], 1.0)
        weight = counts[b] / np.maximum(counts[b] + 120.0, 1.0)
        out[pos] = weight * local_rate + (1.0 - weight) * global_rates[b]
    return out


def _residual_thresholds(residual: pd.DataFrame) -> tuple[float, dict[str, float]]:
    values = pd.to_numeric(residual["__residual__"], errors="coerce")
    finite = values[np.isfinite(values)]
    global_cut = float(np.quantile(finite, 0.20)) if len(finite) >= 100 else -0.005
    local: dict[str, float] = {}
    for key, group in residual.groupby("__local_key__", observed=True, sort=False):
        local_values = pd.to_numeric(group["__residual__"], errors="coerce").dropna()
        if len(local_values) < 80:
            continue
        raw = float(np.quantile(local_values, 0.20))
        weight = len(local_values) / (len(local_values) + 120.0)
        local[str(key)] = float(weight * raw + (1.0 - weight) * global_cut)
    return global_cut, local


def _causal_residual_target(
    train: pd.DataFrame,
    *,
    value_col: str = "ev_after_1pct",
    label_col: str = "__negative_residual_event__",
) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
    """Build a negative-EV-residual target from chronological prior data.

    Four contiguous timestamp blocks are used.  The first is warm-up; every
    later block receives its score-to-EV expectation from earlier blocks only.
    A row becomes a residual-risk event only if its realized EV is in the
    locally shrunk lower residual tail known before that block.
    """

    ordered = train.sort_values(["__ts__", "__symbol__", "side_name"], kind="stable").reset_index(drop=True)
    pieces = [part for part in np.array_split(np.arange(len(ordered), dtype=np.int64), 4) if len(part)]
    history: list[pd.DataFrame] = []
    labelled: list[pd.DataFrame] = []
    residual_history: list[pd.DataFrame] = []
    for block_index, positions in enumerate(pieces):
        # The score-map implementation uses compact NumPy positions.  Reset
        # each chronological block so pandas group labels cannot be mistaken
        # for positions from the full training frame.
        block = ordered.iloc[positions].reset_index(drop=True)
        if block_index == 0:
            history.append(block)
            continue
        reference = pd.concat(history, ignore_index=True, copy=False)
        state = _fit_score_value_map(reference, value_col=value_col)
        evaluation = block.loc[:, [*KEYS, "score", value_col]].copy()
        evaluation["__residual__"] = pd.to_numeric(block[value_col], errors="coerce").to_numpy(dtype=np.float32) - _expected_ev(block, state)
        evaluation["__local_key__"] = _local_key(block).to_numpy(dtype=object)
        if residual_history:
            global_cut, local_cut = _residual_thresholds(pd.concat(residual_history, ignore_index=True, copy=False))
            threshold = np.asarray([local_cut.get(key, global_cut) for key in evaluation["__local_key__"]], dtype=np.float32)
            candidate = block.copy(deep=False)
            candidate[label_col] = (evaluation["__residual__"].to_numpy(dtype=np.float32) <= threshold).astype(np.float32)
            labelled.append(candidate)
        residual_history.append(evaluation)
        history.append(block)
    if not labelled or not residual_history:
        return ordered.iloc[0:0].copy(), np.empty(0, dtype=np.float32), {"global_cut": -0.005, "local_cuts": {}, "state": _fit_score_value_map(ordered, value_col=value_col)}
    reference_residuals = pd.concat(residual_history, ignore_index=True, copy=False)
    global_cut, local_cut = _residual_thresholds(reference_residuals)
    labelled_frame = pd.concat(labelled, ignore_index=True, copy=False)
    return labelled_frame, labelled_frame[label_col].to_numpy(dtype=np.float32), {
        "global_cut": global_cut,
        "local_cuts": local_cut,
        "state": _fit_score_value_map(ordered, value_col=value_col),
    }


def _negative_residual_event(
    frame: pd.DataFrame,
    state: dict[str, Any],
    *,
    value_col: str = "ev_after_1pct",
) -> np.ndarray:
    residual = pd.to_numeric(frame[value_col], errors="coerce").to_numpy(dtype=np.float32) - _expected_ev(frame, state["state"])
    keys = _local_key(frame).to_numpy(dtype=object)
    threshold = np.asarray([state["local_cuts"].get(str(key), state["global_cut"]) for key in keys], dtype=np.float32)
    return (residual <= threshold).astype(np.float32)


def _fit_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    features: list[str] | dict[str, list[str]],
    target: np.ndarray,
    seed: int,
    sample_weight: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    result = np.full(len(test), np.nan, dtype=np.float32)
    report: dict[str, Any] = {"local_models": 0, "side_fallbacks": 0, "global_fallback": 0}
    if isinstance(features, dict):
        feature_by_side = {
            str(side).lower(): list(dict.fromkeys(str(feature) for feature in values))
            for side, values in features.items()
        }
        all_features = list(dict.fromkeys(feature for values in feature_by_side.values() for feature in values))
    else:
        all_features = list(dict.fromkeys(str(feature) for feature in features))
        feature_by_side = {}
    median = train.reindex(columns=all_features).median(numeric_only=True).reindex(all_features).fillna(0.0)
    target = np.asarray(target, dtype=np.float32)
    if len(target) != len(train):
        raise ValueError("target length must match train rows")
    if sample_weight is None:
        weights = np.ones(len(train), dtype=np.float32)
    else:
        weights = np.asarray(sample_weight, dtype=np.float32)
        if len(weights) != len(train):
            raise ValueError("sample_weight length must match train rows")
        weights = np.nan_to_num(weights, nan=1.0, posinf=1.0, neginf=1.0)
        weights = np.clip(weights, 0.05, 20.0)
    global_prior = float(np.average(target, weights=weights)) if len(target) else 0.5

    def _matrix(part: pd.DataFrame, selected_features: list[str]) -> np.ndarray:
        return (
            part.reindex(columns=selected_features)
            .apply(pd.to_numeric, errors="coerce")
            .fillna(median.reindex(selected_features).fillna(0.0))
            .to_numpy(dtype=np.float32)
        )

    train_groups = train.groupby(["side_name", "archetype_policy_key"], observed=True, sort=False).groups
    side_groups = train.groupby("side_name", observed=True, sort=False).groups
    for (side, archetype), test_idx in test.groupby(["side_name", "archetype_policy_key"], observed=True, sort=False).groups.items():
        selected_features = feature_by_side.get(str(side).lower(), all_features)
        if not selected_features:
            result[np.asarray(test_idx, dtype=np.int64)] = global_prior
            report["global_fallback"] += 1
            continue
        local_idx = train_groups.get((side, archetype))
        fallback = False
        if local_idx is None or len(local_idx) < 800 or np.unique(target[local_idx]).size < 2:
            local_idx = side_groups.get(side)
            fallback = True
        min_rows = 1_500 if fallback else 800
        if local_idx is None or len(local_idx) < min_rows or np.unique(target[local_idx]).size < 2:
            result[test_idx] = global_prior
            report["global_fallback"] += 1
            continue
        local_target = target[np.asarray(local_idx, dtype=np.int64)]
        dataset = lgb.Dataset(
            _matrix(train.iloc[local_idx], selected_features),
            label=local_target,
            weight=weights[np.asarray(local_idx, dtype=np.int64)],
            feature_name=selected_features,
            free_raw_data=True,
        )
        model = lgb.train(
            {
                "objective": "binary",
                "metric": "binary_logloss",
                "learning_rate": 0.035,
                "num_leaves": 15,
                "max_depth": 3,
                "min_data_in_leaf": 120,
                "lambda_l1": 1.0,
                "lambda_l2": 5.0,
                "feature_fraction": 0.85,
                "bagging_fraction": 0.85,
                "bagging_freq": 1,
                "seed": int(seed),
                # Keep comparative meta-head ablations invariant to the set
                # of auxiliary arms evaluated in the same process.  LightGBM
                # otherwise derives several sampling seeds implicitly, which
                # can make an unchanged M1 fit drift after an unrelated state
                # recognizer consumes RNG state.
                "feature_fraction_seed": int(seed + 101),
                "bagging_seed": int(seed + 211),
                "data_random_seed": int(seed + 307),
                "deterministic": True,
                "force_col_wise": True,
                "num_threads": 2,
                "verbosity": -1,
            },
            dataset,
            num_boost_round=140,
        )
        result[np.asarray(test_idx, dtype=np.int64)] = model.predict(
            _matrix(test.iloc[test_idx], selected_features), num_iteration=model.best_iteration
        ).astype(np.float32)
        # Native LightGBM allocations are not reliably reclaimed merely by
        # rebinding `model` in a large side × archetype loop.
        model.free_dataset()
        del model, dataset
        report["local_models"] += int(not fallback)
        report["side_fallbacks"] += int(fallback)
    report["feature_count_by_side"] = {
        side: len(values) for side, values in feature_by_side.items()
    } or {"shared": len(all_features)}
    return np.nan_to_num(result, nan=global_prior), report


def _select_top10(frame: pd.DataFrame, score: np.ndarray, baseline_tail: np.ndarray) -> pd.DataFrame:
    work = frame.loc[baseline_tail].copy()
    work["__selection_score__"] = score[baseline_tail]
    # Preserve a true global top-10 opportunity budget at each decision time:
    # start from the base rank's top-20 candidate stream and keep its best
    # half per timestamp.  Sorting only across a whole month lets one side
    # consume the entire tail when raw score scales drift across sides.
    work = work.sort_values(
        ["__selection_score__", "score", "__ts__", "__symbol__", "side_name"],
        ascending=[False, False, True, True, True],
        kind="stable",
    )
    work["__rank_in_timestamp__"] = work.groupby("__ts__", observed=True, sort=False).cumcount()
    count = work.groupby("__ts__", observed=True, sort=False)["__selection_score__"].transform("size")
    return work.loc[work["__rank_in_timestamp__"] < np.ceil(count * 0.50)].drop(columns="__rank_in_timestamp__").copy()


def _metrics(frame: pd.DataFrame, arm: str) -> dict[str, Any]:
    if frame.empty:
        return {"arm": arm, "rows": 0}
    work = frame.copy(deep=False)
    work["week"] = pd.to_datetime(work["__ts__"], utc=True).dt.to_period("W-MON").astype(str)
    work["month"] = pd.to_datetime(work["__ts__"], utc=True).dt.strftime("%Y-%m")
    daily = work.groupby(pd.to_datetime(work["__ts__"], utc=True).dt.floor("D"), observed=True)["ev_after_1pct"].mean()
    weekly = work.groupby("week", observed=True)["ev_after_1pct"].mean()
    monthly = work.groupby("month", observed=True)["ev_after_1pct"].mean()
    return {
        "arm": arm,
        "rows": int(len(work)),
        "trades_per_day": float(len(work) / max(work["__ts__"].dt.floor("D").nunique(), 1)),
        "mean_ev_after_1pct": float(work["ev_after_1pct"].mean()),
        "sum_ev_after_1pct": float(work["ev_after_1pct"].sum()),
        "clean_exec_precision": float(work["clean_exec"].mean()),
        "negative_executable_ev_rate": float((work["ev_after_1pct"] <= 0.0).mean()),
        "first_touch_bad_mae_rate": float(work["first_touch_bad_mae_1r"].mean()),
        "full_path_bad_mae_rate": float(work["full_path_bad_mae_1r"].mean()),
        "timeout_rate": float(work["timeout"].mean()),
        "dirty_positive_rate": float(work["dirty_positive"].mean()),
        "worst_week_ev": float(weekly.min()),
        "worst_month_ev": float(monthly.min()),
        "negative_ev_days": int((daily <= 0.0).sum()),
        "negative_ev_day_rate": float((daily <= 0.0).mean()),
    }


def _breakdown(frame: pd.DataFrame, arm: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    work = frame.copy(deep=False)
    work["month"] = pd.to_datetime(work["__ts__"], utc=True).dt.strftime("%Y-%m")
    return (
        work.groupby(["month", "side_name", "archetype_policy_key"], observed=True)
        .agg(
            rows=("__ts__", "size"),
            mean_ev_after_1pct=("ev_after_1pct", "mean"),
            clean_exec_precision=("clean_exec", "mean"),
            negative_executable_ev_rate=("ev_after_1pct", lambda value: float((value <= 0.0).mean())),
            full_path_bad_mae_rate=("full_path_bad_mae_1r", "mean"),
            timeout_rate=("timeout", "mean"),
        )
        .reset_index()
        .assign(arm=arm)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--months", default="2026-04,2026-05,2026-06")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--feature-store-id", default=DEFAULT_FEATURE_STORE_ID)
    parser.add_argument(
        "--hydrate-transition-sources",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Read missing lifecycle inputs from the canonical static store.",
    )
    args = parser.parse_args()

    months = _parse_months(args.months)
    end = pd.Timestamp((max(months) + 1).start_time, tz="UTC")
    data = _load(args.candidate_root, end)
    hydration: dict[str, Any] = {"enabled": False}
    if args.hydrate_transition_sources:
        data, hydration = _hydrate_transition_sources(
            data,
            feature_store_id=str(args.feature_store_id),
        )
        hydration["enabled"] = True
    data, transition_columns = _add_trajectory_features(data)
    static_features = _feature_columns(data)
    trajectory_features = list(dict.fromkeys([*static_features, *transition_columns]))
    rows: list[dict[str, Any]] = []
    detail: list[pd.DataFrame] = []
    diagnostics: list[dict[str, Any]] = []
    for fold, month in enumerate(months):
        start = pd.Timestamp(month.start_time, tz="UTC")
        end = pd.Timestamp((month + 1).start_time, tz="UTC")
        train = data.loc[data["__ts__"].lt(start)].reset_index(drop=True)
        test = data.loc[data["__ts__"].ge(start) & data["__ts__"].lt(end)].reset_index(drop=True)
        if len(train) < 25_000 or len(test) < 1_000:
            continue
        cutoff = float(np.nanquantile(pd.to_numeric(train["score"], errors="coerce"), 0.80))
        train_tail = train.loc[_tail_mask(train, cutoff)].reset_index(drop=True)
        test_tail = _tail_mask(test, cutoff)
        targets = _executable_quality_targets(train_tail, ResidualEventArchetypeConfig())
        # This is deliberately conditional: among rows that got the direction
        # right, distinguish a good executable trade from a bad path/net
        # realization.  A broad negative-EV target is dominated by ordinary
        # low-score misses and did not answer the decision problem.
        damage_target = targets["correct_direction_bad_trade"].astype(np.float32)
        correct_target = targets["correct_direction"].astype(np.float32)
        # Fit shared static heads once per fold.  The former arm loop retrained
        # exactly the same correct-direction/damage heads for every selector.
        correct_static, correct_static_report = _fit_predict(
            train_tail, test, features=static_features, target=correct_target, seed=int(args.seed + fold)
        )
        damage_static, damage_static_report = _fit_predict(
            train_tail, test, features=static_features, target=damage_target, seed=int(args.seed + fold + 10_000)
        )
        # Path risk is conditional on getting the direction right.  This is
        # the actual ambiguity left after the directional head: will a row
        # that otherwise looks correct experience a stop/MAE/timeout path?
        path_train = train_tail.loc[correct_target > 0.5].reset_index(drop=True)
        path_target = _executable_quality_targets(
            path_train, ResidualEventArchetypeConfig()
        )["executable_adverse_path_event"].astype(np.float32)
        path_static, path_static_report = _fit_predict(
            path_train, test, features=static_features, target=path_target, seed=int(args.seed + fold + 20_000)
        )
        residual_train, residual_target, residual_state = _causal_residual_target(train_tail)
        residual_static, residual_static_report = _fit_predict(
            residual_train,
            test,
            features=static_features,
            target=residual_target,
            seed=int(args.seed + fold + 25_000),
        )
        hit_residual_train, hit_residual_target, hit_residual_state = _causal_residual_target(
            train_tail,
            value_col="clean_exec",
            label_col="__negative_hit_residual_event__",
        )
        hit_residual_static, hit_residual_static_report = _fit_predict(
            hit_residual_train,
            test,
            features=static_features,
            target=hit_residual_target,
            seed=int(args.seed + fold + 27_500),
        )
        correct_trajectory, correct_trajectory_report = _fit_predict(
            train_tail, test, features=trajectory_features, target=correct_target, seed=int(args.seed + fold + 30_000)
        )
        damage_trajectory, damage_trajectory_report = _fit_predict(
            train_tail, test, features=trajectory_features, target=damage_target, seed=int(args.seed + fold + 40_000)
        )
        static_quality = correct_static * (1.0 - damage_static)
        trajectory_quality = correct_trajectory * (1.0 - damage_trajectory)
        arm_scores = {
            "baseline_score": test["score"].to_numpy(dtype=np.float32),
            "correct_direction_static": correct_static,
            "quality_static": static_quality,
            "quality_static_path_risk_050": static_quality * (1.0 - 0.50 * path_static),
            "quality_static_path_risk_100": static_quality * (1.0 - path_static),
            "quality_static_negative_residual_050": static_quality * (1.0 - 0.50 * residual_static),
            "quality_static_negative_residual_100": static_quality * (1.0 - residual_static),
            "quality_static_negative_hit_residual_050": static_quality * (1.0 - 0.50 * hit_residual_static),
            "quality_static_negative_hit_residual_100": static_quality * (1.0 - hit_residual_static),
            "quality_static_plus_trajectory": trajectory_quality,
        }
        observed_damage = (
            (pd.to_numeric(test["clean_exec"], errors="coerce").fillna(0.0).to_numpy() > 0.5)
            & (
                (pd.to_numeric(test["ev_after_1pct"], errors="coerce").fillna(0.0).to_numpy() <= 0.0)
                | (pd.to_numeric(test["full_path_bad_mae_1r"], errors="coerce").fillna(0.0).to_numpy() > 0.5)
                | (pd.to_numeric(test["timeout"], errors="coerce").fillna(0.0).to_numpy() > 0.5)
            )
        ).astype(np.float32)
        observed_path = (
            (pd.to_numeric(test["full_path_bad_mae_1r"], errors="coerce").fillna(0.0).to_numpy() > 0.5)
            | (pd.to_numeric(test["timeout"], errors="coerce").fillna(0.0).to_numpy() > 0.5)
        ).astype(np.float32)
        observed_negative_residual = _negative_residual_event(test, residual_state)
        observed_negative_hit_residual = _negative_residual_event(
            test,
            hit_residual_state,
            value_col="clean_exec",
        )
        for arm, selection_score in arm_scores.items():
            selected = _select_top10(test, selection_score, test_tail)
            rows.append({"month": str(month), **_metrics(selected, arm)})
            detail.append(_breakdown(selected, arm))
            diagnostics.append(
                {
                    "month": str(month),
                    "arm": arm,
                    "features": len(trajectory_features) if arm.endswith("trajectory") else len(static_features),
                    "damage_ap": float(average_precision_score(observed_damage, damage_trajectory if arm.endswith("trajectory") else damage_static)),
                    "damage_auc": float(roc_auc_score(observed_damage, damage_trajectory if arm.endswith("trajectory") else damage_static)),
                    "path_ap": float(average_precision_score(observed_path, path_static)),
                    "path_auc": float(roc_auc_score(observed_path, path_static)),
                    "negative_residual_ap": float(average_precision_score(observed_negative_residual, residual_static)),
                    "negative_residual_auc": float(roc_auc_score(observed_negative_residual, residual_static)),
                    "negative_hit_residual_ap": float(average_precision_score(observed_negative_hit_residual, hit_residual_static)),
                    "negative_hit_residual_auc": float(roc_auc_score(observed_negative_hit_residual, hit_residual_static)),
                    "correct_ap": float(average_precision_score(pd.to_numeric(test["clean_exec"], errors="coerce").fillna(0.0).clip(0.0, 1.0), correct_trajectory if arm.endswith("trajectory") else correct_static)),
                    "correct_auc": float(roc_auc_score(pd.to_numeric(test["clean_exec"], errors="coerce").fillna(0.0).clip(0.0, 1.0), correct_trajectory if arm.endswith("trajectory") else correct_static)),
                    "correct_models": correct_trajectory_report if arm.endswith("trajectory") else correct_static_report,
                    "damage_models": damage_trajectory_report if arm.endswith("trajectory") else damage_static_report,
                    "path_models": path_static_report,
                    "negative_residual_models": residual_static_report,
                    "negative_residual_target_rows": int(len(residual_train)),
                    "negative_residual_target_rate": float(np.mean(residual_target)) if len(residual_target) else None,
                    "negative_hit_residual_models": hit_residual_static_report,
                    "negative_hit_residual_target_rows": int(len(hit_residual_train)),
                    "negative_hit_residual_target_rate": float(np.mean(hit_residual_target)) if len(hit_residual_target) else None,
                }
            )
    args.output.mkdir(parents=True, exist_ok=True)
    scorecard = pd.DataFrame(rows)
    scorecard.to_csv(args.output / "oos_scorecard_by_month.csv", index=False)
    if not scorecard.empty:
        aggregate = scorecard.groupby("arm", observed=True).mean(numeric_only=True).reset_index()
        baseline = aggregate.loc[aggregate["arm"].eq("baseline_score")].iloc[0]
        for column in ("mean_ev_after_1pct", "worst_week_ev", "worst_month_ev", "negative_ev_day_rate", "full_path_bad_mae_rate", "timeout_rate"):
            aggregate[f"delta_{column}_vs_baseline"] = aggregate[column] - float(baseline[column])
        aggregate.to_csv(args.output / "oos_scorecard_aggregate.csv", index=False)
    pd.concat(detail, ignore_index=True).to_csv(args.output / "oos_side_archetype_breakdown.csv", index=False) if detail else pd.DataFrame().to_csv(args.output / "oos_side_archetype_breakdown.csv", index=False)
    pd.DataFrame(diagnostics).to_json(args.output / "predictor_diagnostics.json", orient="records", indent=2)
    _write_json(
        args.output / "manifest.json",
        {
            "schema": "executable_quality_transition_ablation_v1",
            "months": [str(item) for item in months],
            "candidate_root": str(args.candidate_root),
            "static_hydration": hydration,
            "static_feature_count": len(static_features),
            "trajectory_feature_count": len(transition_columns),
            "trajectory_features": transition_columns,
            "selection_contract": "train-fitted score top20; retain best half OOS to preserve top10 activity",
            "leakage_contract": "outcomes define train labels and OOS metrics only; all predictor inputs are pre-entry columns",
        },
    )
    print(json.dumps({"event": "complete", "output": str(args.output), "rows": len(rows)}), flush=True)


if __name__ == "__main__":
    main()
