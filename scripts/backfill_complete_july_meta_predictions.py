#!/usr/bin/env python3
"""Rebuild complete July 8-10 frozen base/meta predictions and outcomes."""

from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from extreme_price_movements.data_store import read_symbol_features
from extreme_price_movements.features_gmm_ae import transform_ae_gmm_features
from extreme_price_movements.inference.live_meta_feature_overlays import (
    materialize_live_source_regime_features,
)
from extreme_price_movements.inference.live_policy_archetype import (
    load_live_policy_archetype_classifier,
    predict_live_policy_archetype,
)
from extreme_price_movements.lgbm_pipeline import (
    _append_meta_post_selection_ood_features,
    _fit_meta_post_selection_ood_reference,
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
from scripts.score_compare_meta_residual_july_oos import _append_store_features

KEYS = ["__ts__", "__symbol__", "side_name"]
OOD_PREFIX = "meta_sel_ood_"


def _utc(values: Any) -> pd.Series:
    return pd.to_datetime(values, utc=True, errors="coerce")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _feature_symbols(feature_root: Path) -> list[str]:
    symbols: list[str] = []
    for path in feature_root.glob("symbol=*.parquet"):
        encoded = path.name[len("symbol=") : -len(".parquet")]
        if "_USD:USD" not in encoded:
            continue
        symbols.append(encoded.replace("_USD:USD", "/USD:USD"))
    return sorted(set(symbols))


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
        missing = ~np.isfinite(current)
        current[missing] = values[missing]
        loaded[name] = current
    return loaded, coverage


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
    finite_path = np.isfinite(paths[0]).all(axis=1) & (paths[0][:, 0] > 0.0)
    for path in paths[1:]:
        finite_path &= np.isfinite(path).all(axis=1)
    missing_columns = [name for name in KEYS if name in rows.columns]
    stats["missing_paths"] = (
        rows.loc[~finite_path, missing_columns].astype(str).to_dict(orient="records")
    )
    # Prediction coverage must be complete. Outcome repair also includes rows
    # already known to lack an executable Kraken chart, so require a diagnostic
    # floor while retaining every missing path explicitly in the manifest.
    if float(stats.get("finite_path_coverage", 0.0)) < 0.75:
        raise RuntimeError(f"Synthetic tail path coverage is too low: {stats}")
    policies = _policy_lookup(policy_manifest)
    captures: dict[str, pd.DataFrame] = {}
    for key in sorted(policy_keys.unique()):
        policy = policies.get(key)
        if policy is None:
            raise KeyError(f"No S59 geometry found for policy key {key!r}")
        captures[key] = _first_touch_capture_outcome(
            rows,
            paths,
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
    return output.infer_objects(copy=False), stats


def _capture_outcomes(capture: pd.DataFrame) -> pd.DataFrame:
    net = pd.to_numeric(capture["capture_net"], errors="coerce")
    valid = pd.to_numeric(capture["capture_valid_path"], errors="coerce").gt(0.5)
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
    out["ev_after_1pct"] = (net - 0.01).where(valid)
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--labels-dir", type=Path, required=True)
    parser.add_argument("--old-labels-dir", type=Path, required=True)
    parser.add_argument("--base-reference", type=Path, required=True)
    parser.add_argument("--base-model-dir", type=Path, required=True)
    parser.add_argument("--meta-model-dir", type=Path, required=True)
    parser.add_argument("--ae-gmm-state", type=Path, required=True)
    parser.add_argument("--meta-handoff-dir", type=Path, required=True)
    parser.add_argument("--residual-bundle", type=Path, required=True)
    parser.add_argument("--native-run-id", required=True)
    parser.add_argument("--policy-manifest", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start", default="2026-07-08T00:00:00Z")
    parser.add_argument("--end-exclusive", default="2026-07-11T00:00:00Z")
    parser.add_argument("--path-len", type=int, default=96)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.environ.setdefault("EPM_SIMPLE_POLICY_15M_DOWNLOAD", "1")
    os.environ.setdefault("EPM_SIMPLE_POLICY_15M_CHART_ONLY", "1")
    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end_exclusive)

    base_contract = _load_json(args.base_model_dir / "columns.json")
    meta_contract = _load_json(args.meta_model_dir / "columns.json")
    base_columns = list(base_contract["feature_names"])
    meta_columns = list(meta_contract["feature_names"])
    meta_ood = [name for name in meta_columns if name.startswith(OOD_PREFIX)]
    meta_pre_ood = [name for name in meta_columns if name not in meta_ood]
    with args.ae_gmm_state.open("rb") as handle:
        ae_gmm_state = pickle.load(handle)

    label_parts = []
    for side in ("long", "short"):
        path = args.labels_dir / f"train_global_{side}_5_2026_07.parquet"
        frame = pd.read_parquet(path)
        frame["__ts__"] = _utc(frame["__ts__"])
        label_parts.append(frame)
    labels = pd.concat(label_parts, ignore_index=True, copy=False)
    symbols = sorted(labels["__symbol__"].astype(str).unique())

    expected_timestamps = pd.date_range(start, end - pd.Timedelta(hours=1), freq="h")
    observed_timestamps = pd.DatetimeIndex(labels["__ts__"].dropna().unique())
    tail_timestamps = expected_timestamps.difference(observed_timestamps)
    if tail_timestamps.empty:
        raise RuntimeError(
            "No missing hourly label batches were found in the requested scope"
        )
    synthetic_raw = _read_tail_feature_rows(
        args.feature_root,
        symbols=symbols,
        timestamps=tail_timestamps,
        columns=list(ae_gmm_state["feature_columns"]),
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
            .fillna(0.02)
            .astype(np.float32)
        )
        part["_synthetic_tail"] = True
        synthetic_parts.append(part)
    synthetic = pd.concat(synthetic_parts, ignore_index=True, copy=False)
    synthetic_batches: list[pd.DataFrame] = []
    overlay_columns = set(base_columns).union(ae_gmm_state["feature_columns"])
    for (timestamp, side), batch in synthetic.groupby(
        ["__ts__", "side_name"], sort=True
    ):
        indexed = batch.set_index("__symbol__", drop=False)
        enriched = materialize_live_source_regime_features(
            indexed,
            side=str(side),
            signal_bar_ts=timestamp,
            required_columns=overlay_columns,
        )
        synthetic_batches.append(enriched.reset_index(drop=True))
    synthetic = pd.concat(synthetic_batches, ignore_index=True, copy=False)
    labels["_synthetic_tail"] = False
    full = pd.concat([labels, synthetic], ignore_index=True, sort=False, copy=False)
    full = full.sort_values(
        ["__ts__", "__symbol__", "side_name"], kind="stable"
    ).reset_index(drop=True)

    generated = transform_ae_gmm_features(
        full.reindex(columns=ae_gmm_state["feature_columns"]),
        ae_gmm_state,
        index=full.index,
    )
    for name in generated.columns:
        full[name] = generated[name].to_numpy(copy=False)
    base_model = joblib.load(args.base_model_dir / "base_model.joblib")
    full["score"] = base_model.predict(
        full.reindex(columns=base_columns).replace([np.inf, -np.inf], np.nan)
    ).astype(np.float32)

    base_reference = pd.read_parquet(
        args.base_reference, columns=["score", "selected_top30"]
    )
    base_cutoff = float(
        pd.to_numeric(
            base_reference.loc[base_reference["selected_top30"].astype(bool), "score"],
            errors="coerce",
        ).min()
    )
    valid = full.loc[
        full["__ts__"].ge(start)
        & full["__ts__"].lt(end)
        & pd.to_numeric(full["score"], errors="coerce").ge(base_cutoff)
    ].copy()
    valid["selected_top30"] = True

    source_contract = _load_json(args.source_manifest)["source_contract"]
    valid["source_tag"] = _source_tags(
        valid["score"], valid["side_name"], source_contract["edges"]
    ).to_numpy()
    valid, store_coverage = _fill_store_features(
        valid,
        args.feature_root,
        list(dict.fromkeys(meta_pre_ood)),
    )

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
    train, valid = _add_fold_base_prior_features(
        train, valid.reset_index(drop=True), selected_col="selected_top30"
    )
    train, valid = _add_fold_reliability_features(train, valid)
    ood_reference = _fit_meta_post_selection_ood_reference(train, meta_pre_ood)
    valid_matrix = _append_meta_post_selection_ood_features(
        valid.reindex(columns=meta_pre_ood), valid, ood_reference
    ).reindex(columns=meta_columns)
    meta_model = joblib.load(args.meta_model_dir / "base_soft_label.joblib")
    valid["score_meta_base_soft_label"] = meta_model.predict(
        valid_matrix.replace([np.inf, -np.inf], np.nan)
    ).astype(np.float32)

    classifier = load_live_policy_archetype_classifier(
        data_root="data_perp", run_id=args.native_run_id
    )
    synthetic_mask = valid["_synthetic_tail"].fillna(False).astype(bool)
    for idx in valid.index[synthetic_mask]:
        side = str(valid.at[idx, "side_name"])
        predicted = predict_live_policy_archetype(
            side=side,
            payload=classifier,
            candidate_feature_row=valid.loc[[idx]],
            meta_model_input_row=valid_matrix.loc[[idx]],
        )
        prefix = f"{side}__"
        valid.at[idx, "__archetype_policy_key__"] = (
            predicted[len(prefix) :] if predicted.startswith(prefix) else predicted
        )
    valid["archetype_policy_key"] = valid["__archetype_policy_key__"].astype(str)

    residual_bundle = joblib.load(args.residual_bundle)
    residual_required = list(
        dict.fromkeys(
            residual_bundle.required_input_features()
            + list(residual_bundle.raw_selected_features)
        )
    )
    valid, residual_coverage = _fill_store_features(
        valid, args.feature_root, residual_required
    )
    residual_predictions = residual_bundle.predict(valid)
    valid[residual_predictions.columns] = residual_predictions.to_numpy(
        dtype=np.float32, copy=False
    )

    outcomes = _outcomes_from_labels(args.old_labels_dir, args.labels_dir)
    outcome_cols = [
        "ev_after_1pct",
        "clean_exec",
        "first_touch_bad_mae_1r",
        "full_path_bad_mae_1r",
        "timeout",
    ]
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
        )
        path_stats[side] = side_path_stats
        captured = _capture_outcomes(capture)
        for name in captured.columns:
            valid.loc[mask, name] = captured[name].to_numpy(copy=False)

    valid["score_current_reference"] = valid["score_meta_base_soft_label"].astype(
        np.float32
    )
    valid["prediction_evidence"] = "frozen_research_contract_complete_backfill"
    keep = [
        *KEYS,
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
        "prediction_evidence",
        "_synthetic_tail",
    ]
    complete = valid[[name for name in keep if name in valid.columns]].copy()
    complete = complete.sort_values(KEYS, kind="stable").drop_duplicates(
        KEYS, keep="last"
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "july_08_10_complete_predictions.parquet"
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
    manifest = {
        "schema": "complete_july_08_10_frozen_predictions_v1",
        "start": start.isoformat(),
        "end_exclusive": end.isoformat(),
        "base_cutoff": base_cutoff,
        "base_feature_contract_hash": base_contract.get("feature_contract_hash"),
        "meta_feature_contract_hash": meta_contract.get("feature_contract_hash"),
        "rows": int(len(complete)),
        "hours": int(complete["__ts__"].nunique()),
        "outcome_rows": int(complete["ev_after_1pct"].notna().sum()),
        "feature_store_coverage": store_coverage,
        "residual_feature_coverage": residual_coverage,
        "synthetic_tail_path_coverage": path_stats,
        "output": str(output_path),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )
    print(hour_counts.to_string(index=False))
    print(json.dumps(manifest, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
