#!/usr/bin/env python3
"""C3 PCA→HDBSCAN causal-market-state control for supportive-label research.

This completes the named HDBSCAN causal-regime arm without substituting a
different clustering method.  The clusterer sees only point-in-time market
context.  Its state probabilities are mapped to expected policy net using
resolved training outcomes after the state is defined; no realised H12 value
is used for held scoring.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import hdbscan
from hdbscan import prediction as hdb_prediction
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler

from run_strict_r3_long_supportive_label_funnel import (
    DEFAULT_LABELS, DEFAULT_LEDGER, FOLDS, MAX_TRAIN_ROWS, SEED,
    _score_eligible, _finite, _joined_population, _ledger_fields, _matrix,
    _quality_metrics, _sample_month_balanced, _sha256,
)
from run_strict_r3_long_supportive_label_causal_joint import (
    HISTORY_START, _market_fields, _month_balanced_timestamps, _state_panel,
)


SCHEMA = "strict_r3_long_supportive_label_hdbscan_regime_v1"
MAX_STATE_FIT_ROWS = 20_000
# Chosen by a target-free density-support scan on the first outer-fold's
# pre-held market-state panel.  The earlier 120/40 floor returned 100% noise;
# 30/5 gives four persistent broad states with 24.3% explicit noise.
MIN_CLUSTER_SIZE = 30
MIN_SAMPLES = 5
STATE_PRIOR_ROWS = 500


def _fit_transform(state_train: pd.DataFrame, fields: list[str]) -> tuple[pd.Series, np.ndarray, np.ndarray, RobustScaler, PCA, np.ndarray]:
    sample = _month_balanced_timestamps(state_train, MAX_STATE_FIT_ROWS, seed=SEED)
    matrix, medians = _matrix(sample, fields)
    lower = np.nanquantile(matrix, 0.005, axis=0).astype(np.float32)
    upper = np.nanquantile(matrix, 0.995, axis=0).astype(np.float32)
    scaler = RobustScaler(quantile_range=(10.0, 90.0), unit_variance=True)
    scaled = scaler.fit_transform(np.clip(matrix, lower, upper))
    pca = PCA(n_components=min(12, scaled.shape[1], max(2, scaled.shape[0] - 1)), random_state=SEED, svd_solver="randomized")
    return medians, lower, upper, scaler, pca, pca.fit_transform(scaled).astype(np.float32)


def _transform(frame: pd.DataFrame, fields: list[str], medians: pd.Series, lower: np.ndarray, upper: np.ndarray, scaler: RobustScaler, pca: PCA) -> np.ndarray:
    matrix, _ = _matrix(frame, fields, medians=medians)
    return pca.transform(scaler.transform(np.clip(matrix, lower, upper))).astype(np.float32)


def _probability(model: hdbscan.HDBSCAN, latent: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    labels, strength = hdb_prediction.approximate_predict(model, latent)
    n_clusters = int(len(model.cluster_persistence_))
    if n_clusters < 1:
        raise RuntimeError("HDBSCAN selected no persistent clusters")
    # ``membership_vector`` in hdbscan 0.8.44 can raise a KeyError for a large
    # out-of-sample query set under Python 3.12.  The supported approximate
    # predict API returns a cluster label and membership strength.  Preserve
    # that strength as a soft two-way cluster/noise mixture rather than hiding
    # the uncertainty by hard-assigning every state.
    result = np.zeros((len(labels), n_clusters + 1), dtype=np.float32)
    noise = np.ones(len(labels), dtype=np.float32)
    usable = (labels >= 0) & (labels < n_clusters)
    clipped = np.clip(np.asarray(strength, dtype=np.float32), 0.0, 1.0)
    rows = np.flatnonzero(usable)
    result[rows, labels[usable].astype(int)] = clipped[usable]
    noise[usable] = 1.0 - clipped[usable]
    result[:, -1] = noise
    return result, noise


def _attach(frame: pd.DataFrame, state: pd.DataFrame, probability: np.ndarray) -> np.ndarray:
    columns = [f"q_{i:02d}" for i in range(probability.shape[1])]
    side = pd.DataFrame(probability, columns=columns)
    side.insert(0, "__decision_ts__", state["__decision_ts__"].to_numpy())
    joined = frame[["candidate_id", "__decision_ts__"]].merge(side, on="__decision_ts__", how="left", validate="many_to_one")
    if joined.loc[:, columns].isna().any(axis=None):
        raise AssertionError("candidate missing HDBSCAN state probability")
    return joined.loc[:, columns].to_numpy(np.float32)


def _state_map(y: np.ndarray, probability: np.ndarray) -> np.ndarray:
    usable = np.isfinite(y)
    y = y[usable]
    probability = probability[usable]
    if len(y) < 1_000:
        raise AssertionError("HDBSCAN state map lacks resolved train-only policy support")
    hard = probability.argmax(axis=1)
    global_mean = float(np.mean(y))
    values = np.full(probability.shape[1], global_mean, dtype=np.float64)
    for index in range(len(values)):
        local = y[hard == index]
        values[index] = (local.sum() + STATE_PRIOR_ROWS * global_mean) / (len(local) + STATE_PRIOR_ROWS)
    return values


def run(*, ledger: Path, labels_root: Path, out: Path, max_train_rows: int) -> Path:
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=False)
    fields = list(_ledger_fields(ledger)); market_fields = list(_market_fields(fields))
    all_metrics: list[dict[str, Any]] = []; audits: list[dict[str, Any]] = []; structural: list[dict[str, Any]] = []
    prediction_root = out / "hdbscan_oof_predictions"; prediction_root.mkdir()
    for ordinal, fold in enumerate(FOLDS):
        train_raw = _joined_population(ledger, labels_root, start=HISTORY_START, end=fold.start, fields=fields, p1_fields=())
        held_raw = _joined_population(ledger, labels_root, start=fold.start, end=fold.end, fields=fields, p1_fields=())
        train = _score_eligible(train_raw); held = _score_eligible(held_raw)
        train = _sample_month_balanced(train, max_train_rows, seed=SEED + ordinal)
        state_train = _state_panel(train_raw, market_fields); state_held = _state_panel(held_raw, market_fields)
        del train_raw, held_raw; gc.collect()
        medians, lower, upper, scaler, pca, sample_latent = _fit_transform(state_train, market_fields)
        model = hdbscan.HDBSCAN(
            min_cluster_size=MIN_CLUSTER_SIZE, min_samples=MIN_SAMPLES,
            metric="euclidean", cluster_selection_method="eom", prediction_data=True,
        ).fit(sample_latent)
        train_p, train_noise = _probability(model, _transform(state_train, market_fields, medians, lower, upper, scaler, pca))
        held_p, held_noise = _probability(model, _transform(state_held, market_fields, medians, lower, upper, scaler, pca))
        q_train = _attach(train, state_train, train_p); q_held = _attach(held, state_held, held_p)
        expected = _state_map(_finite(train["policy_net_bps"]).to_numpy(float), q_train)
        score = q_held @ expected
        arm = "C3_hdbscan_state_expected_ev"
        all_metrics.extend(_quality_metrics(fold=fold, arm=arm, feature_mode="market_context_only", score=score, held=held))
        hard = train_p.argmax(axis=1)
        supports = pd.Series(hard).value_counts().sort_index().to_dict()
        structural.append({
            "fold": fold.name, "cohort": fold.cohort, "clusters_excluding_noise": int(train_p.shape[1] - 1),
            "noise_state_fraction": float((hard == train_p.shape[1] - 1).mean()),
            "mean_noise_membership": float(train_noise.mean()), "min_cluster_size": MIN_CLUSTER_SIZE,
            "min_samples": MIN_SAMPLES, "relative_validity": float(getattr(model, "relative_validity_", np.nan)),
            "mean_cluster_persistence": float(np.mean(getattr(model, "cluster_persistence_", np.array([np.nan])))),
            "hard_support": json.dumps({str(k): int(v) for k, v in supports.items()}),
        })
        pd.DataFrame({
            "candidate_id": held["candidate_id"].to_numpy(), "__decision_ts__": held["__decision_ts__"].to_numpy(),
            "fold": fold.name, "cohort": fold.cohort, "arm": arm, "predicted_policy_net_bps": score.astype(np.float32),
            "realised_policy_net_bps": _finite(held["policy_net_bps"]).to_numpy(np.float32), "hdbscan_noise_mass": q_held[:, -1].astype(np.float32),
        }).to_parquet(prediction_root / f"fold={ordinal:02d}_{fold.name}.parquet", index=False, compression="zstd")
        audits.append({"fold": fold.name, "status": "ok", "train_rows": int(len(train)), "held_rows": int(len(held)), "train_label_cutoff": str(fold.start), "embargo_hours": 12, "state_train_timestamps": int(len(state_train)), "state_held_timestamps": int(len(state_held))})
        print(json.dumps(audits[-1]), flush=True)
    metric_frame = pd.DataFrame(all_metrics)
    metric_frame.to_parquet(out / "hdbscan_metrics.parquet", index=False, compression="zstd")
    metric_frame.groupby(["arm", "cohort", "metric"], as_index=False).agg(mean_value=("value", "mean"), worst_value=("value", "min"), folds=("fold", "nunique")).to_parquet(out / "hdbscan_summary.parquet", index=False, compression="zstd")
    pd.DataFrame(audits).to_parquet(out / "hdbscan_fold_audit.parquet", index=False, compression="zstd")
    pd.DataFrame(structural).to_parquet(out / "hdbscan_structural_audit.parquet", index=False, compression="zstd")
    (out / "run_manifest.json").write_text(json.dumps({
        "schema": SCHEMA, "scope": "offline long-only research only; no live mutation", "ledger": str(ledger.resolve()), "ledger_sha256": _sha256(ledger),
        "labels_root": str(labels_root.resolve()), "market_context_only_state_fields": market_fields,
        "hdbscan": {"min_cluster_size": MIN_CLUSTER_SIZE, "min_samples": MIN_SAMPLES, "explicit_noise_state": True},
        "causality": "state inputs are point-in-time only; state-to-policy map uses strictly pre-held resolved policy labels; 12h embargo applies to supervised maps",
    }, indent=2) + "\n")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER); parser.add_argument("--labels-root", type=Path, default=DEFAULT_LABELS); parser.add_argument("--out", type=Path, required=True); parser.add_argument("--max-train-rows", type=int, default=MAX_TRAIN_ROWS)
    args = parser.parse_args(); print(json.dumps({"out": str(run(ledger=args.ledger.resolve(), labels_root=args.labels_root.resolve(), out=args.out.resolve(), max_train_rows=args.max_train_rows))}))


if __name__ == "__main__":
    main()
