#!/usr/bin/env python3
"""Fit a compact Beta--Binomial residual-event trust model on strict R3.

This is a bounded alternative to the continuous ideal-lambda Bayesian target.
It learns two lower-noise, economically explicit events on resolved training
rows only:

* policy net exceeds its causal mapped expectation by at least 50 bps;
* policy net underperforms that expectation by at least 100 bps.

The output is a train-CDF-normalised posterior score suitable only for the
existing bounded score-correction and causal-admission evaluators.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


SEED = 20260812
RAW_K9_PREFIX = "k09__cluster_"
IDENTITY = {
    "candidate_id", "__decision_ts__", "__symbol__", "side_name",
    "policy_path_valid", "policy_label_available_ts", "policy_net_bps",
    "policy_gross_bps", "policy_exit_reason", "policy_exit_bar_15m",
    "policy_entry_price", "policy_exit_price",
}
EXCLUDE_PREFIXES = ("causal_", "ev_mapping_", "calibration_")
EXCLUDE_DIRECT = {
    "conversion_bundle_sha256", "geometry_bundle_sha256",
    "ev_score_family_id", "upstream_bundle_sha256", "stack_is_prequential",
    "policy_outcome_source", "severe_affects_final_score",
    # This is the residual target anchor, not an inference feature.  Including
    # it would make a correction that is itself passed into the EV map
    # self-referential.
    "raw_expected_bps",
}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _timestamp_top30(frame: pd.DataFrame) -> np.ndarray:
    ordered = frame.sort_values(
        ["__decision_ts__", "final_score", "candidate_id"],
        ascending=[True, False, True], kind="stable",
    )
    position = ordered.groupby("__decision_ts__", sort=False).cumcount().to_numpy()
    count = ordered.groupby("__decision_ts__", sort=False)["candidate_id"].transform("size").to_numpy()
    active = position < np.maximum(1, np.ceil(count * 0.30).astype(int))
    return pd.Series(active, index=ordered.index).reindex(frame.index).to_numpy(bool)


def _eligible_fields(train: pd.DataFrame) -> list[str]:
    output: list[str] = []
    for field in train.columns:
        if (
            field in IDENTITY or field in EXCLUDE_DIRECT
            or field.startswith(EXCLUDE_PREFIXES) or field.startswith(RAW_K9_PREFIX)
        ):
            continue
        values = pd.to_numeric(train[field], errors="coerce")
        if values.notna().mean() >= 0.90 and values.var() > 1e-12:
            output.append(field)
    if len(output) < 12:
        raise ValueError(f"only {len(output)} causal fields pass coverage/variance")
    return output


def _bin_edges(values: pd.Series, bins: int) -> np.ndarray:
    finite = pd.to_numeric(values, errors="coerce").dropna().to_numpy(float)
    if len(finite) < bins:
        raise ValueError("insufficient finite values for binning")
    edges = np.unique(np.quantile(finite, np.linspace(0.0, 1.0, bins + 1)))
    if len(edges) < 3:
        raise ValueError("feature has insufficient unique values for binning")
    return edges[1:-1]


def _codes(values: pd.Series, edges: np.ndarray) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce")
    fill = float(numeric.median()) if numeric.notna().any() else 0.0
    return np.searchsorted(edges, numeric.fillna(fill).to_numpy(float), side="right").astype(np.int16)


def _binary_mi(codes: np.ndarray, target: np.ndarray) -> float:
    table = np.zeros((int(codes.max(initial=0)) + 1, 2), dtype=float)
    np.add.at(table, (codes, target.astype(int)), 1.0)
    joint = table / max(1.0, table.sum())
    px = joint.sum(axis=1, keepdims=True)
    py = joint.sum(axis=0, keepdims=True)
    valid = joint > 0.0
    return float(np.sum(joint[valid] * np.log(joint[valid] / (px * py)[valid])))


def _cdf(reference: np.ndarray, value: np.ndarray) -> np.ndarray:
    reference = np.sort(np.asarray(reference, dtype=float)[np.isfinite(reference)])
    if len(reference) < 100:
        return np.full(len(value), 0.5, dtype=np.float32)
    return (np.searchsorted(reference, np.asarray(value, dtype=float), side="right") / len(reference)).astype(np.float32)


def _logit(probability: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(probability, dtype=float), 1e-4, 1.0 - 1e-4)
    return np.log(p / (1.0 - p))


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -30.0, 30.0)))


def _fit_predict(
    train: pd.DataFrame, held: pd.DataFrame, fields: list[str], *, bins: int, prior: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, object]]]:
    residual = (
        pd.to_numeric(train["policy_net_bps"], errors="raise").to_numpy(float)
        - pd.to_numeric(train["raw_expected_bps"], errors="raise").to_numpy(float)
    )
    success = residual >= 50.0
    adverse = residual <= -100.0
    # Field selection jointly rewards uplift-event information and adverse
    # information, on training labels only.
    selection: list[dict[str, object]] = []
    definitions: dict[str, np.ndarray] = {}
    for field in fields:
        try:
            edges = _bin_edges(train[field], bins)
        except ValueError:
            # The coarse coverage/variance screen is intentionally broad.
            # A field that cannot form at least two train-only quantile bins
            # has no usable Beta--Binomial cell representation for this fold.
            continue
        code = _codes(train[field], edges)
        mi_success = _binary_mi(code, success)
        mi_adverse = _binary_mi(code, adverse)
        selection.append({
            "field": field, "selection_mi": mi_success + 0.5 * mi_adverse,
            "success_mi": mi_success, "adverse_mi": mi_adverse,
        })
        definitions[field] = edges
    selection.sort(key=lambda value: (-float(value["selection_mi"]), str(value["field"])))
    selected = selection[:12]
    if len(selected) < 12:
        raise ValueError(f"only {len(selected)} fields formed stable train-only bins")
    base_success = float(success.mean())
    base_adverse = float(adverse.mean())
    success_effects_train: list[np.ndarray] = []
    success_effects_held: list[np.ndarray] = []
    adverse_effects_train: list[np.ndarray] = []
    adverse_effects_held: list[np.ndarray] = []
    weights_train: list[np.ndarray] = []
    weights_held: list[np.ndarray] = []
    for item in selected:
        field = str(item["field"])
        edges = definitions[field]
        train_code = _codes(train[field], edges)
        held_code = _codes(held[field], edges)
        size = max(int(train_code.max(initial=0)), int(held_code.max(initial=0))) + 1
        support = np.bincount(train_code, minlength=size).astype(float)
        success_total = np.bincount(train_code, weights=success.astype(float), minlength=size)
        adverse_total = np.bincount(train_code, weights=adverse.astype(float), minlength=size)
        p_success = (success_total + prior * base_success) / (support + prior)
        p_adverse = (adverse_total + prior * base_adverse) / (support + prior)
        success_effects_train.append(_logit(p_success[train_code]) - _logit(np.array([base_success]))[0])
        success_effects_held.append(_logit(p_success[held_code]) - _logit(np.array([base_success]))[0])
        adverse_effects_train.append(_logit(p_adverse[train_code]) - _logit(np.array([base_adverse]))[0])
        adverse_effects_held.append(_logit(p_adverse[held_code]) - _logit(np.array([base_adverse]))[0])
        weights_train.append(support[train_code] / (support[train_code] + prior))
        weights_held.append(support[held_code] / (support[held_code] + prior))
    def combine(effects: list[np.ndarray], weights: list[np.ndarray], base: float) -> np.ndarray:
        effect = np.vstack(effects)
        weight = np.vstack(weights)
        average = (effect * weight).sum(axis=0) / np.maximum(weight.sum(axis=0), 1e-12)
        return _sigmoid(_logit(np.array([base]))[0] + average)
    return (
        combine(success_effects_train, weights_train, base_success),
        combine(adverse_effects_train, weights_train, base_adverse),
        combine(success_effects_held, weights_held, base_success),
        combine(adverse_effects_held, weights_held, base_adverse),
        selected,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--train-start", required=True)
    parser.add_argument("--cutoff", required=True)
    parser.add_argument("--held-end", required=True)
    parser.add_argument("--bins", type=int, default=8)
    parser.add_argument("--prior", type=float, default=100.0)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable output exists: {args.out_dir}")
    if args.bins < 4 or args.prior <= 0.0:
        raise ValueError("bins must be >= 4 and prior must be positive")
    train_start = pd.Timestamp(args.train_start, tz="UTC")
    cutoff = pd.Timestamp(args.cutoff, tz="UTC")
    held_end = pd.Timestamp(args.held_end, tz="UTC")
    frame = pd.read_parquet(args.source)
    frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True, errors="raise")
    frame["policy_label_available_ts"] = pd.to_datetime(frame["policy_label_available_ts"], utc=True, errors="raise")
    if frame["candidate_id"].duplicated().any() or not frame["side_name"].astype(str).str.lower().eq("long").all():
        raise AssertionError("source must be one unique long-only candidate ledger")
    if not frame["stack_is_prequential"].fillna(False).astype(bool).all():
        raise AssertionError("source contains non-prequential scores")
    usable = (
        frame["policy_path_valid"].fillna(False).astype(bool)
        & frame["policy_net_bps"].notna()
        & frame["causal_21d_side_expected_net_bps"].notna()
    )
    train_all = frame.loc[
        frame["__decision_ts__"].ge(train_start) & frame["__decision_ts__"].lt(cutoff)
        & frame["policy_label_available_ts"].lt(cutoff) & usable
    ].copy()
    held = frame.loc[frame["__decision_ts__"].ge(cutoff) & frame["__decision_ts__"].lt(held_end)].copy()
    train_mask = _timestamp_top30(train_all)
    train = train_all.loc[train_mask].copy()
    if len(train) < 2_000 or held.empty:
        raise ValueError(f"insufficient train/held support {len(train)=} {len(held)=}")
    train["raw_expected_bps"] = pd.to_numeric(train["causal_21d_side_expected_net_bps"], errors="raise")
    held["raw_expected_bps"] = pd.to_numeric(held["causal_21d_side_expected_net_bps"], errors="coerce")
    candidate_fields = _eligible_fields(train)
    train_success, train_adverse, held_success, held_adverse, selected = _fit_predict(
        train, held, candidate_fields, bins=int(args.bins), prior=float(args.prior),
    )
    output = held.loc[:, [
        "candidate_id", "__decision_ts__", "__symbol__", "side_name", "final_score",
        "policy_path_valid", "policy_label_available_ts", "policy_gross_bps", "policy_net_bps",
        "policy_exit_reason", "geometry_bundle_sha256",
    ]].copy().reset_index(drop=True)
    output["posterior_expected_bps"] = held_success.astype(np.float32)
    output["p_ev_positive"] = held_success.astype(np.float32)
    output["p_adverse_tail"] = held_adverse.astype(np.float32)
    output["posterior_expected_rank_train"] = _cdf(train_success, held_success)
    output["posterior_adverse_rank_train"] = _cdf(train_adverse, held_adverse)
    output["timestamp_top30"] = _timestamp_top30(held)
    output["bayes_available"] = True
    output["arm"] = "beta_residual_event_mi12"
    args.out_dir.mkdir(parents=True)
    output.to_parquet(args.out_dir / "predictions.parquet", index=False, compression="zstd")
    correctness = {
        "unique_candidate_ids": bool(not output["candidate_id"].duplicated().any()),
        "strictly_prequential_source": True,
        "train_labels_resolved_before_cutoff": bool((train_all["policy_label_available_ts"] < cutoff).all()),
        "held_labels_after_decision": bool((output["policy_label_available_ts"] > output["__decision_ts__"]).all()),
        "raw_k9_memberships_used": False,
        "train_held_disjoint": bool(set(train_all["candidate_id"]).isdisjoint(set(output["candidate_id"]))),
    }
    (args.out_dir / "correctness_report.json").write_text(json.dumps(correctness, indent=2) + "\n")
    manifest = {
        "schema": "strict_r3_beta_residual_event_fold_v1",
        "source": str(args.source), "source_sha256": _sha(args.source),
        "train_start": str(train_start), "cutoff": str(cutoff), "held_end": str(held_end),
        "train_rows_before_timestamp_top30": int(len(train_all)), "train_rows": int(len(train)),
        "held_rows": int(len(held)), "candidate_field_count": len(candidate_fields),
        "selected_fields": selected, "bins": int(args.bins), "prior": float(args.prior),
        "target": "policy_net_bps - causal_21d_side_expected_net_bps: success >= +50 bps; adverse <= -100 bps",
        "model": "per-field train-quantile Beta-Binomial posterior, support-weighted additive logit effects",
        "causality": "strict prequential source; train labels resolve before cutoff; held labels excluded from fitting",
        "seed": SEED,
    }
    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"event": "complete", "rows": len(output), "out_dir": str(args.out_dir)}))


if __name__ == "__main__":
    main()
