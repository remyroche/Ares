#!/usr/bin/env python3
"""Materialize a frozen market/cross-sectional AE/GMM state for meta training.

The state deliberately excludes asset-local price direction and outcomes.  It
uses market/cross-sectional feature families plus *causal*, completed-trade
summaries of the base stream.  The output is keyed by timestamp/symbol and can
therefore be joined to both long and short meta candidates without refitting.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from pathlib import Path
from typing import Iterable

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.features_gmm_ae import (  # noqa: E402
    ae_gmm_feature_columns,
    fit_ae_gmm_state,
    transform_ae_gmm_features,
)


KEYS = ("__ts__", "__symbol__", "side_name")
PREFIX = "meta_market_aegmm_"
ROLLING_WINDOWS = (5, 10, 20)
MARKET_PREFIXES = (
    "mkt_",
    "market_",
    "xs_",
    "cs_",
    "q_iqr__",
    "state_spectral_",
    "cross_asset_",
    "pct_assets_",
)
MARKET_SUBSTRINGS = ("_iqr", "_xs_", "_cs_")
BASE_STATE_COLUMNS = (
    "dae_reconstruction_error_zscore",
    "gmm_entropy",
    "gmm_ood_score",
    "gmm_unknown_probability",
    "gmm_posterior_margin",
    "min_mahalanobis",
)


def _quoted(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def _market_feature_columns(feature_dir: Path) -> list[str]:
    sample = next(feature_dir.glob("symbol=*.parquet"), None)
    if sample is None:
        raise FileNotFoundError(f"No feature shards in {feature_dir}")
    names = pq.ParquetFile(sample).schema_arrow.names
    result = []
    for name in names:
        lower = str(name).lower()
        if "target" in lower or "label" in lower or "future" in lower:
            continue
        if lower.startswith(MARKET_PREFIXES) or any(token in lower for token in MARKET_SUBSTRINGS):
            result.append(str(name))
    return list(dict.fromkeys(result))


def _load_base_stream(handoff: Path, ledger: Path) -> pd.DataFrame:
    h = str(handoff.resolve()).replace("'", "''")
    s = str(ledger.resolve()).replace("'", "''")
    state_select = ", ".join(
        f"CAST(h.{_quoted(name)} AS FLOAT) AS {_quoted(name)}" for name in BASE_STATE_COLUMNS
    )
    query = f"""
        SELECT
            h.__ts__, h.__symbol__, lower(h.side_name) AS side_name,
            h.__label_path_end_ts__,
            coalesce(nullif(h.archetype_policy_key, ''), nullif(h.archetype_label_family, ''), 'unknown')
                AS archetype_policy_key,
            CAST(h.score AS FLOAT) AS score,
            CAST(s.clean_exec AS FLOAT) AS clean_exec,
            CAST(s.ev_after_1pct AS FLOAT) AS ev_after_1pct,
            {state_select}
        FROM read_parquet('{h}') h
        INNER JOIN read_parquet('{s}') s
          ON h.__ts__ = s.__ts__
         AND h.__symbol__ = s.__symbol__
         AND lower(h.side_name) = lower(s.side_name)
        ORDER BY h.__ts__, h.__symbol__, side_name
    """
    frame = duckdb.sql(query).df()
    for column in ("__ts__", "__label_path_end_ts__"):
        frame[column] = pd.to_datetime(frame[column], utc=True, errors="coerce")
    if frame[["__ts__", "__label_path_end_ts__"]].isna().any().any():
        raise ValueError("Market AE/GMM source has non-finite timestamps")
    frame["side_name"] = frame["side_name"].astype(str).str.lower()
    frame["side_name"] = frame["side_name"].astype("category")
    frame["archetype_policy_key"] = frame["archetype_policy_key"].astype(str).astype("category")
    frame["__symbol__"] = frame["__symbol__"].astype(str).astype("category")
    return frame


def _spearman(x: Iterable[float], y: Iterable[float]) -> float:
    x_arr = np.asarray(list(x), dtype=np.float64)
    y_arr = np.asarray(list(y), dtype=np.float64)
    if len(x_arr) < 5 or np.nanstd(x_arr) <= 1e-12 or np.nanstd(y_arr) <= 1e-12:
        return 0.0
    return float(pd.Series(x_arr).rank().corr(pd.Series(y_arr).rank(), method="pearson") or 0.0)


def _rolling_completed_trade_context(frame: pd.DataFrame) -> pd.DataFrame:
    """Build per-row causal last-N resolved-trade summaries, then aggregate by timestamp."""
    work = frame.copy(deep=False)
    work["__group__"] = work["side_name"].astype(str) + "__" + work["archetype_policy_key"].astype(str)
    # Statistics are identical for all rows with the same timestamp and
    # side/archetype.  Materializing them at candidate-row granularity would
    # create a large, unnecessary 1.3m x 12 intermediate.
    targets = (
        work.groupby(["__group__", "__ts__"], observed=True, sort=False)
        .size()
        .rename("candidate_count")
        .reset_index()
    )
    summary_parts: list[pd.DataFrame] = []
    for group_key, group in work.groupby("__group__", observed=True, sort=False):
        events = group.sort_values(["__label_path_end_ts__", "__ts__", "score"], kind="stable")
        events = events.reset_index(drop=True)
        # Rolling Pearson correlation of global ranks is an efficient Spearman
        # approximation here: the ranks preserve ordering and each local window
        # is tiny (5/10/20 completed trades).
        score_rank = events["score"].rank(method="average")
        ev_rank = events["ev_after_1pct"].rank(method="average")
        for n in ROLLING_WINDOWS:
            events[f"market_base_completed_hit_n{n}"] = events["clean_exec"].rolling(n, min_periods=1).mean()
            events[f"market_base_completed_ev_n{n}"] = events["ev_after_1pct"].rolling(n, min_periods=1).mean()
            # Constant short windows have undefined rank correlation on some
            # pandas versions and can yield +/-inf.  IC is a bounded statistic;
            # encode unavailable windows as neutral before it reaches the state
            # scaler/AE rather than relying on downstream imputation.
            events[f"market_base_completed_ic_n{n}"] = (
                score_rank.rolling(n, min_periods=5)
                .corr(ev_rank)
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0.0)
                .clip(-1.0, 1.0)
            )
            events[f"market_base_completed_support_n{n}"] = np.minimum(np.arange(1, len(events) + 1), n).astype(np.float32)
        # State after a whole completion cohort.  The strict merge below keeps
        # it unavailable to same-timestamp candidate rows.
        hist = events.groupby("__label_path_end_ts__", observed=True, sort=True).tail(1)
        hist = hist.loc[
            :, ["__label_path_end_ts__", *[c for c in events if c.startswith("market_base_")]]
        ]
        if hist.empty:
            continue
        hist["__label_path_end_ts__"] = pd.to_datetime(
            hist["__label_path_end_ts__"], utc=True
        ).astype("datetime64[ns, UTC]")
        target = targets.loc[targets["__group__"].eq(group_key), ["__ts__", "candidate_count"]].copy()
        target = target.sort_values("__ts__", kind="stable")
        target["__ts__"] = pd.to_datetime(target["__ts__"], utc=True).astype(
            "datetime64[ns, UTC]"
        )
        aligned = pd.merge_asof(
            target,
            hist,
            left_on="__ts__",
            right_on="__label_path_end_ts__",
            direction="backward",
            allow_exact_matches=False,
        )
        summary_parts.append(aligned.drop(columns="__label_path_end_ts__", errors="ignore"))
    if not summary_parts:
        return pd.DataFrame({"__ts__": work["__ts__"].drop_duplicates()})
    joined = pd.concat(summary_parts, ignore_index=True, copy=False)
    metric_columns = [column for column in joined if column.startswith("market_base_")]
    # Candidate-weighted timestamp aggregates represent the current opportunity
    # basket. Add IQR only for quality measures; support uses a weighted mean.
    grouped = joined.groupby("__ts__", observed=True)
    result = pd.DataFrame({"__ts__": joined["__ts__"].drop_duplicates().sort_values().to_numpy()})
    total_weight = grouped["candidate_count"].sum().rename("__weight__").reset_index()
    result = result.merge(total_weight, on="__ts__", how="left", validate="one_to_one")
    for column in metric_columns:
        weighted = (joined[column].fillna(0.0) * joined["candidate_count"]).groupby(joined["__ts__"], observed=True).sum()
        result = result.merge(
            (weighted / total_weight.set_index("__ts__")["__weight__"]).rename(column).reset_index(),
            on="__ts__",
            how="left",
            validate="one_to_one",
        )
    result = result.drop(columns="__weight__")
    for n in ROLLING_WINDOWS:
        for base in (f"market_base_completed_hit_n{n}", f"market_base_completed_ev_n{n}", f"market_base_completed_ic_n{n}"):
            q = grouped[base].quantile([0.25, 0.75]).unstack()
            result = result.merge(
                (q[0.75] - q[0.25]).rename(f"{base}_iqr").reset_index(),
                on="__ts__",
                how="left",
                validate="one_to_one",
            )
    return result


def _timestamp_base_state_context(frame: pd.DataFrame) -> pd.DataFrame:
    values = frame.loc[:, ["__ts__", *BASE_STATE_COLUMNS]].copy()
    grouped = values.groupby("__ts__", observed=True)
    output = grouped[list(BASE_STATE_COLUMNS)].mean().reset_index()
    output = output.rename(columns={name: f"market_base_{name}_mean" for name in BASE_STATE_COLUMNS})
    output["market_base_candidate_count"] = grouped.size().to_numpy(dtype=np.float32)
    for name in ("gmm_ood_score", "gmm_entropy", "dae_reconstruction_error_zscore"):
        q = grouped[name].quantile([0.50, 0.90]).unstack()
        output[f"market_base_{name}_p50"] = q[0.50].to_numpy(dtype=np.float32)
        output[f"market_base_{name}_p90"] = q[0.90].to_numpy(dtype=np.float32)
    return output


def _symbol_path(feature_dir: Path, symbol: str) -> Path:
    return feature_dir / f"symbol={symbol.replace('/', '_')}.parquet"


def _feature_store_symbols(feature_dir: Path, max_symbols: int) -> np.ndarray:
    """Return a deterministic feature-store universe, independent of candidates.

    The market-state cross section must not depend on the later handoff's
    candidate eligibility.  A shard only contributes where it has a finite
    point-in-time value, so symbols listed after an historical timestamp do
    not affect that timestamp.
    """
    symbols = np.asarray(
        sorted(path.stem.removeprefix("symbol=").replace("_", "/") for path in feature_dir.glob("symbol=*.parquet")),
        dtype=object,
    )
    if len(symbols) == 0:
        raise FileNotFoundError(f"No feature shards in {feature_dir}")
    if len(symbols) > max_symbols:
        symbols = symbols[np.unique(np.linspace(0, len(symbols) - 1, max_symbols, dtype=np.int64))]
    return symbols


def _materialize_market_timestamp_inputs(
    timestamps_source: pd.Series,
    feature_dir: Path,
    raw_columns: list[str],
    timestamp_context: pd.DataFrame,
    sample_symbols: np.ndarray,
) -> tuple[pd.DataFrame, int]:
    """Aggregate observable cross-asset inputs into one row per timestamp.

    The features in this block are intended to describe the market, not an
    individual trade.  A deterministic, evenly spaced sample of eligible
    symbols keeps the cross-section representative without reading the entire
    1.3m-row candidate matrix for every ablation.
    """
    timestamps = pd.DatetimeIndex(pd.to_datetime(timestamps_source, utc=True).drop_duplicates().sort_values())
    n_rows, n_cols = len(timestamps), len(raw_columns)
    sums = np.zeros((n_rows, n_cols), dtype=np.float64)
    counts = np.zeros((n_rows, n_cols), dtype=np.uint16)
    timestamp_index = pd.Index(timestamps)
    for symbol in sample_symbols.tolist():
        path = _symbol_path(feature_dir, str(symbol))
        if not path.exists():
            continue
        schema = set(pq.ParquetFile(path).schema_arrow.names)
        available = [name for name in raw_columns if name in schema]
        if not available:
            continue
        # Direct file access retains the stored ``ts`` column.  ``pq.read_table``
        # interprets ``symbol=...`` as a hive partition path and can replace the
        # physical schema with the partition schema on some pyarrow builds.
        table = pq.ParquetFile(path).read(columns=["ts", "__symbol__", *available])
        # Feature shards preserve ``ts`` as their pandas index.  Reset it so
        # the point-in-time join is explicit and independent of pandas metadata.
        part = table.to_pandas().reset_index()
        part["__ts__"] = pd.to_datetime(part.pop("ts"), utc=True, errors="coerce")
        positions = timestamp_index.get_indexer(part["__ts__"])
        keep = positions >= 0
        if not bool(keep.any()):
            continue
        positions = positions[keep]
        for idx, name in enumerate(raw_columns):
            if name not in part:
                continue
            values = pd.to_numeric(part.loc[keep, name], errors="coerce").to_numpy(dtype=np.float64, copy=False)
            finite = np.isfinite(values)
            if finite.any():
                np.add.at(sums[:, idx], positions[finite], values[finite])
                np.add.at(counts[:, idx], positions[finite], 1)
    means = np.divide(sums, counts, out=np.full_like(sums, np.nan), where=counts > 0).astype(np.float32)
    output = pd.DataFrame(means, columns=[f"market_input__{name}" for name in raw_columns])
    output.insert(0, "__ts__", timestamps)
    output["market_input_cross_section_coverage"] = counts.mean(axis=1).astype(np.float32)
    output = output.merge(timestamp_context, on="__ts__", how="left", validate="one_to_one")
    return output, int(len(sample_symbols))


def _time_spread_sample(frame: pd.DataFrame, count: int) -> pd.DataFrame:
    if len(frame) <= count:
        return frame
    order = frame.sort_values([column for column in ("__ts__", "__symbol__") if column in frame], kind="stable").reset_index(drop=True)
    positions = np.linspace(0, len(order) - 1, num=count, dtype=np.int64)
    return order.iloc[np.unique(positions)].copy()


def _varying_state_columns(generated: pd.DataFrame) -> list[str]:
    """Keep only meaningful nonconstant, nonduplicate state outputs for meta.

    Fixed-width AE/GMM schemas reserve unused component slots and include
    temporal derivatives.  This market-state block is row-independent, so
    those derivatives are structurally zero.  Excluding them makes the meta
    feature contract smaller without discarding any live-computable signal.
    """
    kept: list[str] = []
    signatures: set[tuple[float, ...]] = set()
    for column in generated.columns:
        values = pd.to_numeric(generated[column], errors="coerce").to_numpy(dtype=np.float32, copy=False)
        finite = np.isfinite(values)
        if not finite.any() or np.nanmax(values) - np.nanmin(values) <= 1e-8:
            continue
        signature = tuple(np.nan_to_num(values, nan=np.float32(-9999.0)).tolist())
        if signature in signatures:
            continue
        signatures.add(signature)
        kept.append(column)
    return kept


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--handoff", type=Path, required=True)
    parser.add_argument("--scored-ledger", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--fit-end", default="2026-04-01")
    parser.add_argument("--ae-rows", type=int, default=50_000)
    parser.add_argument("--gmm-rows", type=int, default=100_000)
    parser.add_argument("--market-sample-symbols", type=int, default=32)
    parser.add_argument("--reuse-materialized-inputs", action="store_true")
    args = parser.parse_args()

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    cached_inputs = out / "market_aegmm_inputs_by_timestamp.parquet"
    if args.reuse_materialized_inputs and cached_inputs.exists():
        market_inputs = pd.read_parquet(cached_inputs)
        timestamp_context = pd.read_parquet(out / "market_timestamp_causal_context.parquet")
        raw_columns = [c.removeprefix("market_input__") for c in market_inputs if c.startswith("market_input__")]
        market_sample_symbols = int(args.market_sample_symbols)
        print(f"[market-aegmm] reused timestamp inputs={len(market_inputs):,}", flush=True)
    else:
        print("[market-aegmm] loading causal base stream", flush=True)
        frame = _load_base_stream(args.handoff, args.scored_ledger)
        print(f"[market-aegmm] base stream rows={len(frame):,}", flush=True)
        timestamp_base = _timestamp_base_state_context(frame)
        print(f"[market-aegmm] base-state timestamp rows={len(timestamp_base):,}", flush=True)
        rolling_columns = ["__ts__", "__label_path_end_ts__", "side_name", "archetype_policy_key", "score", "clean_exec", "ev_after_1pct"]
        rolling_frame = frame.loc[:, rolling_columns].copy()
        timestamps_source = frame["__ts__"].copy()
        del frame
        gc.collect()
        print("[market-aegmm] deriving completed-trade causal summaries", flush=True)
        timestamp_context = timestamp_base.merge(
            _rolling_completed_trade_context(rolling_frame), on="__ts__", how="left", validate="one_to_one"
        )
        del rolling_frame, timestamp_base
        gc.collect()
        print(f"[market-aegmm] timestamp context rows={len(timestamp_context):,}", flush=True)
        timestamp_context.to_parquet(out / "market_timestamp_causal_context.parquet", index=False, compression="zstd")
        raw_columns = _market_feature_columns(args.feature_dir)
        print(f"[market-aegmm] raw market columns={len(raw_columns)}", flush=True)
        sample_symbols = _feature_store_symbols(args.feature_dir, max(4, int(args.market_sample_symbols)))
        market_inputs, market_sample_symbols = _materialize_market_timestamp_inputs(
            timestamps_source,
            args.feature_dir,
            raw_columns,
            timestamp_context,
            sample_symbols=sample_symbols,
        )
        del timestamps_source
        gc.collect()
        print(f"[market-aegmm] market timestamp inputs={len(market_inputs):,}", flush=True)
        market_inputs.to_parquet(cached_inputs, index=False, compression="zstd")
    input_columns = [column for column in market_inputs.columns if column != "__ts__"]
    fit_end = pd.Timestamp(args.fit_end, tz="UTC")
    train = market_inputs.loc[market_inputs["__ts__"].lt(fit_end), ["__ts__", *input_columns]].copy()
    print(f"[market-aegmm] train candidates={len(train):,}", flush=True)
    train["__ts__"] = pd.to_datetime(train["__ts__"], utc=True)
    train = _time_spread_sample(train, max(args.gmm_rows, args.ae_rows))
    x = train.reindex(columns=input_columns).astype(np.float32)
    state = fit_ae_gmm_state(
        x,
        timestamps=train["__ts__"],
        random_state=20260722,
        max_train_rows=int(args.ae_rows),
        gmm_max_train_rows=int(args.gmm_rows),
        ae_max_iter=32,
        # This is a controlled content ablation, not a representation HPO.
        # Hold the density family to one incumbent-like configuration so the
        # downstream comparison measures market-context value rather than an
        # additional encoder search.
        cluster_candidates=(6,),
        reg_covar_candidates=(0.003,),
        covariance_type_candidates=("diag",),
        smooth_lambda_candidates=(0.0,),
        enhanced_search=False,
        outcome_free=True,
        temporal_feature_contract="row_independent_v1",
    )
    print(f"[market-aegmm] fitted components={state.get('gmm_n_components')}", flush=True)
    if not state.get("enabled", False):
        raise RuntimeError(f"Market AE/GMM fit failed: {state.get('reason')}")
    state_path = out / "meta_market_aegmm_state.pkl"
    pd.to_pickle(state, state_path)
    output_columns = ae_gmm_feature_columns(PREFIX)
    generated = transform_ae_gmm_features(
        market_inputs.reindex(columns=input_columns), state, index=market_inputs.index, prefix=PREFIX
    )
    generated[PREFIX + "active"] = np.float32(1.0)
    active_output_columns = _varying_state_columns(generated)
    active_output_columns.append(PREFIX + "active")
    state_by_timestamp = pd.concat([market_inputs[["__ts__"]].reset_index(drop=True), generated.reset_index(drop=True)], axis=1)
    state_by_timestamp.to_parquet(out / "meta_market_aegmm_state_by_timestamp.parquet", index=False, compression="zstd")
    print("[market-aegmm] transformed market state", flush=True)
    output_path = str((out / "meta_market_aegmm_state_by_timestamp.parquet").resolve()).replace("'", "''")
    handoff = str(args.handoff.resolve()).replace("'", "''")
    state_select = ", ".join(f"m.{_quoted(c)}" for c in active_output_columns)
    augmented = out / "train_meta_regime_handoff_with_market_aegmm.parquet"
    augmented_sql = str(augmented.resolve()).replace("'", "''")
    duckdb.sql(
        f"COPY (SELECT h.*, {state_select} FROM read_parquet('{handoff}') h "
        f"LEFT JOIN read_parquet('{output_path}') m ON h.__ts__=m.__ts__) "
        f"TO '{augmented_sql}' (FORMAT PARQUET, COMPRESSION ZSTD)"
    )
    print("[market-aegmm] wrote augmented handoff", flush=True)
    manifest = {
        "schema_version": "meta_market_aegmm_v1",
        "purpose": "pre-entry cross-asset/market-state AE/GMM plus causal completed-trade base-stream reliability",
        "fit_end_exclusive": fit_end.isoformat(),
        "input_feature_count": len(input_columns),
        "raw_market_feature_count": len(raw_columns),
        "market_sample_symbols": int(locals().get("market_sample_symbols", args.market_sample_symbols)),
        "market_symbol_sampling_contract": "feature_store_universe_even_time_invariant_sample_v1",
        "causal_base_aggregate_feature_count": len(timestamp_context.columns) - 1,
        "raw_market_feature_columns": raw_columns,
        "causal_context_columns": [c for c in timestamp_context.columns if c != "__ts__"],
        "state_path": str(state_path),
        "augmented_handoff": str(augmented),
        "generated_prefix": PREFIX,
        "generated_columns": [*output_columns, PREFIX + "active"],
        "meta_active_generated_columns": active_output_columns,
        "state_config": state.get("selected_config", {}),
        "actual_state_config": {
            "n_components": int(state.get("gmm_n_components", 0)),
            "covariance_type": str(state.get("gmm_covariance_type", "")),
            "reg_covar": float(state.get("gmm_reg_covar", float("nan"))),
            "latent_dimension": len(state.get("latent_columns", [])),
        },
        "temporal_contract": "row_independent_v1",
        "representation_fit_contract": "Unlabeled AE/GMM fit. Inputs include pre-entry market features and causal completed-trade aggregates; no outcome target is passed to state selection.",
        "causal_outcome_derived_input_columns": [c for c in input_columns if c.startswith("market_base_completed_")],
        "leakage_contract": "Market features are pre-entry. Completed-trade aggregates admit only label paths ending strictly before the signal timestamp.",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
if __name__ == "__main__":
    main()
