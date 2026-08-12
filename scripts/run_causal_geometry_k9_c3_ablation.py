#!/usr/bin/env python3
"""Matched C3 geometry/K9 cadence ablation for the strict-R3 long stack.

This research runner deliberately separates two state contracts:

* **market geometry/K9** is base-independent.  It is an unsupervised, causal
  K=9 representation of the frozen raw 120-field market feature contract;
* **leaf/path health** is base-dependent.  A 64-round R3-clear reference model
  is refit for every held month and contributes only aggregate active-leaf
  support/OOD features.  Raw leaf identities never cross a monthly boundary.

The C3 arms enforce the important training restriction: a safety/residual
model may train only on rows *after* the frozen geometry/K9 burn-in ends.  It
therefore never mixes incompatible K9 meanings in one supervised fit.

The upstream strict-R3 base+ten-head consensus handoff is held fixed.  This is
intentional: the experiment isolates representation cadence and its downstream
safety/Correctness layers, rather than re-opening base or consensus HPO.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from lightgbm import LGBMClassifier, LGBMRanker
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import OneHotEncoder

from extreme_price_movements.strict_r3_canonical_current import (
    _structural_geometry_breaks,
    _structural_projection,
    _weighted_moments,
)


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_tp6_sl4_exact170_canonical_consensus import BASE_PARAMS, _load_contract  # noqa: E402


SEED = 20260810
K = 9
GEOMETRY_CAP = 100_000
MODEL_CAP = 240_000
TAILS = (0.005, 0.01, 0.02, 0.05, 0.10)
FEATURE_PATH = ROOT / (
    "data_perp/artifacts/strict_r3_exact_h12_2025_2026_v16_approved_15m_proxy_features/"
    "canonical120_features.parquet"
)
UPSTREAM_PATH = ROOT / "data_perp/artifacts/strict_r3_full_inference_2025_2026_v2/predictions.parquet"
POLICY_PATH = ROOT / "data_perp/artifacts/strict_r3_simple_policy_15m_2025_2026_20260809_v3/candidate_policy_outcomes.parquet"
LABEL_ROOT = ROOT / "data_perp/artifacts/stage_i_packb_tp6_sl4_h12_r3_20260803_v1"


def _month_start(value: str | pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(value, tz="UTC").normalize().replace(day=1)


def _months(spec: str) -> list[str]:
    values: list[str] = []
    for section in str(spec).split(","):
        start, end = section.strip().split(":", 1)
        values.extend(pd.period_range(start, end, freq="M").astype(str).tolist())
    return values


def _month_add(value: pd.Timestamp, offset: int) -> pd.Timestamp:
    return (value.to_period("M") + offset).to_timestamp().tz_localize("UTC")


def _pct(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    ref = np.sort(np.asarray(reference, dtype=float)[np.isfinite(reference)], kind="stable")
    result = np.full(len(values), 0.5, dtype=np.float32)
    good = np.isfinite(values)
    if len(ref) >= 2 and good.any():
        left = np.searchsorted(ref, values[good], side="left")
        right = np.searchsorted(ref, values[good], side="right")
        result[good] = (0.5 * (left + right) + 0.5) / len(ref)
    return np.clip(result, 0.0, 1.0).astype(np.float32)


def _numeric(frame: pd.DataFrame, fields: Sequence[str], medians: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    values = frame.loc[:, list(fields)].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if medians is None:
        medians = values.median().fillna(0.0).to_numpy(dtype=np.float32)
    filled = values.fillna(pd.Series(medians, index=list(fields))).fillna(0.0)
    return filled.to_numpy(dtype=np.float32), np.asarray(medians, dtype=np.float32)


def _equal_month_sample(frame: pd.DataFrame, cap: int, *, seed: int) -> pd.DataFrame:
    if len(frame) <= cap:
        return frame.copy()
    timestamp_column = "__decision_ts__" if "__decision_ts__" in frame else "__ts__"
    month = pd.to_datetime(frame[timestamp_column], utc=True).dt.to_period("M")
    chunks = list(frame.assign(__month__=month).groupby("__month__", sort=True))
    quota = max(1, int(math.ceil(cap / len(chunks))))
    result = pd.concat(
        [block.sample(min(len(block), quota), random_state=seed + index) for index, (_, block) in enumerate(chunks)],
        ignore_index=True,
    ).drop(columns="__month__")
    if len(result) > cap:
        result = result.sample(cap, random_state=seed + 991)
    order = [timestamp_column]
    if "candidate_id" in result:
        order.append("candidate_id")
    return result.sort_values(order, kind="stable")


def _feature_fields() -> list[str]:
    # Read schema metadata only: loading this multi-year raw panel merely to
    # discover names would waste several gigabytes before the ablation starts.
    names = pq.ParquetFile(FEATURE_PATH).schema.names
    fields = [str(value) for value in _load_contract()["long"]]
    missing = sorted(set(fields) - set(names))
    if missing:
        raise ValueError(f"raw feature panel lacks frozen fields: {missing[:12]}")
    if len(fields) != 120:
        raise ValueError(f"expected the frozen 120-field raw contract, found {len(fields)}")
    return fields


def _read_features(start: pd.Timestamp, end: pd.Timestamp, fields: Sequence[str]) -> pd.DataFrame:
    columns = ["__ts__", "__symbol__", *fields]
    frame = pd.read_parquet(
        FEATURE_PATH,
        columns=columns,
        filters=[("__ts__", ">=", start), ("__ts__", "<", end)],
    )
    frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True)
    return frame.sort_values(["__ts__", "__symbol__"], kind="stable").reset_index(drop=True)


def _load_labels() -> pd.DataFrame:
    columns = [
        "candidate_id", "__ts__", "__decision_ts__", "__label_available_at__",
        "label_valid", "target_invalid", "t2_tp6_sl4_event", "robust_clear_event_b25",
    ]
    paths = sorted(LABEL_ROOT.glob("parts/month=*/side=long.parquet"))
    if not paths:
        raise FileNotFoundError(f"no long R3 labels under {LABEL_ROOT}")
    frame = pd.concat([pd.read_parquet(path, columns=columns) for path in paths], ignore_index=True)
    for column in ("__ts__", "__decision_ts__", "__label_available_at__"):
        frame[column] = pd.to_datetime(frame[column], utc=True)
    valid = frame["label_valid"].fillna(False).astype(bool) & ~frame["target_invalid"].fillna(True).astype(bool)
    event = pd.to_numeric(frame["t2_tp6_sl4_event"], errors="coerce")
    clear = pd.to_numeric(frame["robust_clear_event_b25"], errors="coerce")
    frame["r3_clear"] = (valid & event.ne(1) & clear.eq(1)).astype(np.int8)
    frame["r3_label_valid"] = valid
    return frame[["candidate_id", "__ts__", "__decision_ts__", "__label_available_at__", "r3_clear", "r3_label_valid"]]


def _load_main(fields: Sequence[str], months: Sequence[str]) -> pd.DataFrame:
    policy = pd.read_parquet(POLICY_PATH)
    policy["__ts__"] = pd.to_datetime(policy["__ts__"], utc=True)
    policy["__decision_ts__"] = pd.to_datetime(policy["__decision_ts__"], utc=True)
    policy = policy.loc[
        policy["month"].isin(months)
        & policy["policy_path_valid"].fillna(False).astype(bool)
        & np.isfinite(pd.to_numeric(policy["policy_net_bps"], errors="coerce"))
    ].copy()
    # The policy replay carries the source score only to identify candidates.
    # The upstream handoff below is authoritative and supplies the complete
    # base/consensus field set without pandas suffix ambiguity.
    policy = policy.drop(columns=["final_score"], errors="ignore")
    upstream = pd.read_parquet(
        UPSTREAM_PATH,
        columns=[
            "candidate_id", "__ts__", "base_score", "base_anchor_bps", "base_rank",
            "consensus_rank", "final_score", "exact_h12_net_bps", "evaluation_exact_label_valid",
        ],
    )
    upstream["__ts__"] = pd.to_datetime(upstream["__ts__"], utc=True)
    main = policy.merge(upstream, on=["candidate_id", "__ts__"], how="inner", validate="one_to_one")
    main = main.loc[main["evaluation_exact_label_valid"].fillna(False).astype(bool)].copy()
    start = main["__ts__"].min().floor("D")
    end = (main["__ts__"].max() + pd.Timedelta(days=1)).ceil("D")
    raw = _read_features(start, end, fields)
    main = main.merge(raw, on=["__ts__", "__symbol__"], how="inner", validate="one_to_one")
    labels = _load_labels()
    main = main.merge(labels, on=["candidate_id", "__ts__", "__decision_ts__"], how="left", validate="one_to_one")
    finite = main.loc[:, list(fields)].replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
    main = main.loc[finite].copy()
    main["label_available_ts"] = main["__decision_ts__"] + pd.Timedelta(hours=12)
    main["month"] = main["__decision_ts__"].dt.to_period("M").astype(str)
    if main.candidate_id.duplicated().any():
        raise ValueError("matched main panel has duplicate candidate identities")
    return main.sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)


@dataclass
class RawK9Bundle:
    bundle_id: str
    fit_start: pd.Timestamp
    fit_end: pd.Timestamp
    fields: tuple[str, ...]
    medians: np.ndarray
    scale: np.ndarray
    kmeans: MiniBatchKMeans
    permutation: np.ndarray
    temperature: float
    fit_rows: int
    source_kind: str
    temperature_scale: float = 1.0
    structural_projection: np.ndarray | None = None
    structural_mean: np.ndarray | None = None
    structural_covariance: np.ndarray | None = None
    structural_correlation: np.ndarray | None = None
    cluster_structural_mean: np.ndarray | None = None
    cluster_structural_covariance: np.ndarray | None = None
    cluster_structural_correlation: np.ndarray | None = None
    cluster_structural_support: np.ndarray | None = None

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        x, _ = _numeric(frame, self.fields, self.medians)
        z = np.clip((x - self.medians) / self.scale, -8.0, 8.0)
        distances = self.kmeans.transform(z)[:, self.permutation].astype(np.float32)
        # Membership scaling is part of the frozen representation. Deriving
        # it from the transformed frame would let held rows rescale their own
        # inputs even though the K9 centres themselves are causal.
        effective_temperature = float(self.temperature) * float(self.temperature_scale)
        logits = -distances / max(effective_temperature, 1e-6)
        logits -= logits.max(axis=1, keepdims=True)
        membership = np.exp(logits, dtype=np.float32)
        membership /= np.maximum(membership.sum(axis=1, keepdims=True), 1e-12)
        result: dict[str, np.ndarray] = {}
        for cluster in range(K):
            prefix = f"k09__cluster_{cluster:02d}"
            result[f"{prefix}__membership"] = membership[:, cluster]
            result[f"{prefix}__negative_distance"] = -distances[:, cluster]
            result[f"{prefix}__confidence"] = membership[:, cluster] ** 2
        ordered = pd.DataFrame(result, index=frame.index, dtype=np.float32)
        ordered["k9_entropy"] = (-membership * np.log(np.clip(membership, 1e-12, 1.0))).sum(axis=1)
        ordered["k9_top2_margin"] = np.partition(membership, -2, axis=1)[:, -1] - np.partition(membership, -2, axis=1)[:, -2]
        ordered["k9_ood_distance"] = distances.min(axis=1)
        if self.structural_projection is not None:
            ordered = pd.concat(
                [ordered, _structural_geometry_breaks(frame, z, membership, self)],
                axis=1,
            )
        return ordered.astype(np.float32)


def _fit_raw_k9(
    fit: pd.DataFrame,
    fields: Sequence[str],
    *,
    bundle_id: str,
    fit_start: pd.Timestamp,
    fit_end: pd.Timestamp,
    source_kind: str,
    previous: RawK9Bundle | None,
    temperature_scale: float = 1.0,
) -> tuple[RawK9Bundle, dict[str, object]]:
    if fit.empty:
        raise ValueError(f"{bundle_id}: empty geometry warm-up")
    sample = _equal_month_sample(fit, GEOMETRY_CAP, seed=SEED + len(bundle_id))
    x, medians = _numeric(sample, fields)
    q25 = np.quantile(x, 0.25, axis=0)
    q75 = np.quantile(x, 0.75, axis=0)
    scale = np.maximum(q75 - q25, 1e-4).astype(np.float32)
    z = np.clip((x - medians) / scale, -8.0, 8.0)
    model = MiniBatchKMeans(n_clusters=K, batch_size=4096, n_init=5, random_state=SEED + len(bundle_id))
    model.fit(z)
    permutation = np.arange(K, dtype=int)
    mean_cosine = float("nan")
    mean_distance = float("nan")
    if previous is not None:
        old = previous.kmeans.cluster_centers_[previous.permutation]
        new = model.cluster_centers_
        old_norm = old / np.maximum(np.linalg.norm(old, axis=1, keepdims=True), 1e-12)
        new_norm = new / np.maximum(np.linalg.norm(new, axis=1, keepdims=True), 1e-12)
        cosine = old_norm @ new_norm.T
        row, col = linear_sum_assignment(1.0 - cosine)
        permutation = np.empty(K, dtype=int)
        permutation[row] = col
        mean_cosine = float(cosine[row, col].mean())
        mean_distance = float(np.linalg.norm(old - new[permutation], axis=1).mean())
    digest = hashlib.sha256()
    digest.update(np.asarray(model.cluster_centers_[permutation], dtype=np.float32).tobytes())
    fit_distances = model.transform(z)[:, permutation]
    temperature = max(float(np.median(fit_distances.min(axis=1))), 1e-3)
    if not 0.0 < float(temperature_scale) <= 1.0:
        raise ValueError("temperature_scale must lie in (0, 1]")
    effective_temperature = temperature * float(temperature_scale)
    logits = -fit_distances / max(effective_temperature, 1e-6)
    logits -= logits.max(axis=1, keepdims=True)
    fit_membership = np.exp(logits, dtype=np.float32)
    fit_membership /= np.maximum(fit_membership.sum(axis=1, keepdims=True), 1e-12)
    projection = _structural_projection(fields)
    projected = np.asarray(z @ projection, dtype=np.float64)
    structural_mean, structural_covariance, structural_correlation = _weighted_moments(
        projected, None,
    )
    cluster_mean: list[np.ndarray] = []
    cluster_covariance: list[np.ndarray] = []
    cluster_correlation: list[np.ndarray] = []
    cluster_support: list[float] = []
    for cluster in range(K):
        mean, covariance, correlation = _weighted_moments(
            projected, fit_membership[:, cluster],
        )
        cluster_mean.append(mean)
        cluster_covariance.append(covariance)
        cluster_correlation.append(correlation)
        cluster_support.append(float(fit_membership[:, cluster].sum()))
    digest.update(
        np.asarray([temperature, temperature_scale], dtype=np.float32).tobytes(),
    )
    digest.update(np.asarray(projection, dtype=np.float32).tobytes())
    digest.update(np.asarray(structural_covariance, dtype=np.float32).tobytes())
    digest.update(np.asarray(cluster_covariance, dtype=np.float32).tobytes())
    bundle = RawK9Bundle(
        bundle_id=bundle_id, fit_start=fit_start, fit_end=fit_end, fields=tuple(fields),
        medians=medians, scale=scale, kmeans=model, permutation=permutation,
        temperature=temperature, fit_rows=len(sample), source_kind=source_kind,
        temperature_scale=float(temperature_scale),
        structural_projection=np.asarray(projection, dtype=np.float32),
        structural_mean=np.asarray(structural_mean, dtype=np.float32),
        structural_covariance=np.asarray(structural_covariance, dtype=np.float32),
        structural_correlation=np.asarray(structural_correlation, dtype=np.float32),
        cluster_structural_mean=np.asarray(cluster_mean, dtype=np.float32),
        cluster_structural_covariance=np.asarray(cluster_covariance, dtype=np.float32),
        cluster_structural_correlation=np.asarray(cluster_correlation, dtype=np.float32),
        cluster_structural_support=np.asarray(cluster_support, dtype=np.float32),
    )
    audit = {
        "bundle_id": bundle_id, "source_kind": source_kind,
        "fit_start": fit_start, "fit_end_exclusive": fit_end,
        "fit_rows": int(len(sample)), "bundle_sha256": digest.hexdigest(),
        "membership_temperature": float(temperature),
        "membership_temperature_scale": float(temperature_scale),
        "effective_membership_temperature": float(effective_temperature),
        "membership_temperature_source": "geometry_fit_population_only",
        "previous_bundle_id": None if previous is None else previous.bundle_id,
        "matched_center_cosine": mean_cosine, "matched_center_distance": mean_distance,
        "fit_uses_outcomes": False, "base_independent": True,
    }
    return bundle, audit


@dataclass
class LeafReference:
    """One current-base R3 reference, shared by leaf health and coupled K9."""

    fields: tuple[str, ...]
    medians: np.ndarray
    model: LGBMClassifier
    train_leaves: np.ndarray
    support_counts: tuple[np.ndarray, ...]
    train_rows: int
    leaf_values: tuple[np.ndarray, ...] | None = None

    def leaves(self, frame: pd.DataFrame) -> np.ndarray:
        x, _ = _numeric(frame, self.fields, self.medians)
        leaves = np.asarray(self.model.predict(x, pred_leaf=True), dtype=np.int32)
        if leaves.shape[1] != 64:
            raise AssertionError("base-coupled leaf reference must contain 64 trees")
        return leaves


@dataclass
class BaseCoupledK9Bundle:
    """K9 over current base-model leaf paths; deliberately in-sample for meta."""

    bundle_id: str
    fit_start: pd.Timestamp
    fit_end: pd.Timestamp
    lookback_months: int
    leaf_reference: LeafReference
    one_hot: OneHotEncoder
    kmeans: MiniBatchKMeans
    permutation: np.ndarray
    temperature: float
    fit_rows: int

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        leaves = self.leaf_reference.leaves(frame)
        encoded = self.one_hot.transform(leaves)
        distances = self.kmeans.transform(encoded)[:, self.permutation].astype(np.float32)
        logits = -distances / max(float(self.temperature), 1e-6)
        logits -= logits.max(axis=1, keepdims=True)
        membership = np.exp(logits, dtype=np.float32)
        membership /= np.maximum(membership.sum(axis=1, keepdims=True), 1e-12)
        result: dict[str, np.ndarray] = {}
        for cluster in range(K):
            prefix = f"k09__cluster_{cluster:02d}"
            result[f"{prefix}__membership"] = membership[:, cluster]
            result[f"{prefix}__negative_distance"] = -distances[:, cluster]
            result[f"{prefix}__confidence"] = membership[:, cluster] ** 2
        output = pd.DataFrame(result, index=frame.index, dtype=np.float32)
        output["k9_entropy"] = (-membership * np.log(np.clip(membership, 1e-12, 1.0))).sum(axis=1)
        output["k9_top2_margin"] = (
            np.partition(membership, -2, axis=1)[:, -1]
            - np.partition(membership, -2, axis=1)[:, -2]
        )
        output["k9_ood_distance"] = distances.min(axis=1)
        return output.astype(np.float32)


def _one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", dtype=np.float32, sparse_output=True)
    except TypeError:  # pragma: no cover - older sklearn
        return OneHotEncoder(handle_unknown="ignore", dtype=np.float32, sparse=True)


def _fit_base_coupled_k9(
    reference: LeafReference,
    fit: pd.DataFrame,
    *,
    bundle_id: str,
    fit_start: pd.Timestamp,
    fit_end: pd.Timestamp,
    lookback_months: int,
) -> tuple[BaseCoupledK9Bundle, dict[str, object]]:
    """Fit K9 on recent, in-sample leaf paths of the current R3 reference.

    This arm is intentionally not a C3 representation: the leaf encoder uses
    labelled data and the recent K9-fitting rows are also available to the
    downstream fit. It is a bounded diagnostic of the former base-coupled
    geometry idea, with every dependency recorded in the manifest.
    """
    if fit.empty:
        raise ValueError(f"{bundle_id}: empty base-coupled geometry fit")
    sample = _equal_month_sample(fit, GEOMETRY_CAP, seed=SEED + len(bundle_id))
    one_hot = _one_hot_encoder()
    one_hot.fit(reference.train_leaves)
    fit_leaves = reference.leaves(sample)
    encoded_fit = one_hot.transform(fit_leaves)
    model = MiniBatchKMeans(
        n_clusters=K, batch_size=4096, n_init=5, random_state=SEED + len(bundle_id),
    ).fit(encoded_fit)
    centre_hashes = [
        hashlib.sha256(np.asarray(centre, dtype=np.float32).tobytes()).hexdigest()
        for centre in model.cluster_centers_
    ]
    permutation = np.argsort(np.asarray(centre_hashes), kind="stable").astype(int)
    distances = model.transform(encoded_fit)[:, permutation]
    temperature = max(float(np.median(distances.min(axis=1))), 1e-3)
    digest = hashlib.sha256()
    digest.update(np.asarray(model.cluster_centers_[permutation], dtype=np.float32).tobytes())
    digest.update(np.asarray(reference.train_leaves[:1], dtype=np.int32).tobytes())
    bundle = BaseCoupledK9Bundle(
        bundle_id=bundle_id,
        fit_start=fit_start,
        fit_end=fit_end,
        lookback_months=int(lookback_months),
        leaf_reference=reference,
        one_hot=one_hot,
        kmeans=model,
        permutation=permutation,
        temperature=temperature,
        fit_rows=int(len(sample)),
    )
    audit = {
        "bundle_id": bundle_id,
        "source_kind": "same_base_recent_leaf_paths_in_sample_meta",
        "fit_start": fit_start,
        "fit_end_exclusive": fit_end,
        "fit_rows": int(len(sample)),
        "lookback_months": int(lookback_months),
        "bundle_sha256": digest.hexdigest(),
        "matched_center_cosine": float("nan"),
        "matched_center_distance": float("nan"),
        "fit_uses_outcomes": True,
        "base_independent": False,
        "same_leaf_reference_for_k9_and_state": True,
        "in_sample_meta_rows_allowed": True,
    }
    return bundle, audit

def _dynamic_k9_state(frame: pd.DataFrame, k9: pd.DataFrame) -> pd.DataFrame:
    membership_fields = [f"k09__cluster_{index:02d}__membership" for index in range(K)]
    event = pd.concat([frame[["__decision_ts__"]].reset_index(drop=True), k9[membership_fields].reset_index(drop=True)], axis=1)
    state = event.groupby("__decision_ts__", sort=True)[membership_fields].sum().sort_index()
    prior = state.shift(1).fillna(0.0)
    support = prior.rolling("28D", min_periods=1).sum()
    current_prob = state.div(state.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)
    ref_prob = support.div(support.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)
    mean = prior.rolling("28D", min_periods=4).mean()
    std = prior.rolling("28D", min_periods=8).std().replace(0.0, np.nan)
    z = (state - mean) / std
    # Map timestamp aggregates back to the original candidate rows.  The
    # state table has one row per decision time, whereas the downstream model
    # has many candidates at each time.
    out = pd.DataFrame(index=frame.index)
    support_rows = frame[["__decision_ts__"]].merge(
        support.reset_index(), on="__decision_ts__", how="left", validate="many_to_one",
    )[membership_fields].fillna(0.0).to_numpy(float)
    memberships = k9[membership_fields].to_numpy(float)
    total = np.maximum(support_rows.sum(axis=1, keepdims=True), 1.0)
    marginal_surprise = -np.log(
        np.clip((support_rows + 1.0) / (total + K), 1e-12, 1.0),
    )
    out["k9_path_support_effective_28d"] = (memberships * support_rows).sum(axis=1)
    out["k9_path_support_adequate_fraction"] = (
        memberships * (support_rows >= 30.0)
    ).sum(axis=1)
    out["k9_path_ood_marginal"] = (memberships * marginal_surprise).sum(axis=1)
    out["k9_model_ood_marginal"] = frame[["__decision_ts__"]].merge(z.abs().mean(axis=1).rename("_v").reset_index(), on="__decision_ts__", how="left")["_v"].fillna(0.0).to_numpy(float)
    out["k9_model_drift_psi"] = frame[["__decision_ts__"]].merge((((current_prob-ref_prob)*np.log(np.clip(current_prob,1e-12,None)/np.clip(ref_prob,1e-12,None))).sum(axis=1)).rename("_v").reset_index(), on="__decision_ts__", how="left")["_v"].fillna(0.0).to_numpy(float)
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)


def _leaf_value_arrays(
    model: LGBMClassifier,
    support_counts: Sequence[np.ndarray],
) -> tuple[np.ndarray, ...]:
    output: list[np.ndarray] = []
    for tree_index, tree in enumerate(model.booster_.dump_model()["tree_info"]):
        values = np.zeros(len(support_counts[tree_index]), dtype=np.float32)

        def visit(node: dict[str, object]) -> None:
            if "leaf_index" in node:
                leaf_index = int(node["leaf_index"])
                if leaf_index < len(values):
                    values[leaf_index] = float(node.get("leaf_value", 0.0))
                return
            visit(node["left_child"])
            visit(node["right_child"])

        visit(tree["tree_structure"])
        output.append(values)
    return tuple(output)


def _fit_leaf_reference(train: pd.DataFrame, fields: Sequence[str]) -> tuple[LeafReference, dict[str, object]]:
    fit = train.loc[
        train["r3_label_valid"].fillna(False).astype(bool) & train["r3_clear"].notna()
    ].copy()
    fit = _equal_month_sample(fit, MODEL_CAP, seed=SEED + 171)
    if len(fit) < 1_000 or fit["r3_clear"].nunique() < 2:
        raise ValueError("leaf-state base has insufficient R3 support")
    x_train, medians = _numeric(fit, fields)
    params = {
        **BASE_PARAMS,
        "objective": "binary",
        "n_estimators": 64,
        "num_class": None,
        "random_state": SEED + 173,
    }
    params.pop("num_class", None)
    model = LGBMClassifier(**params).fit(x_train, fit["r3_clear"].to_numpy(np.int8))
    leaf_train = np.asarray(model.predict(x_train, pred_leaf=True), dtype=np.int32)
    support_counts = tuple(
        np.bincount(leaf_train[:, tree]).astype(np.float32)
        for tree in range(leaf_train.shape[1])
    )
    reference = LeafReference(
        fields=tuple(fields),
        medians=medians,
        model=model,
        train_leaves=leaf_train,
        support_counts=support_counts,
        train_rows=int(len(fit)),
        leaf_values=_leaf_value_arrays(model, support_counts),
    )
    audit = {
        "leaf_base_fit_rows": int(len(fit)),
        "leaf_base_trees": int(leaf_train.shape[1]),
        "leaf_base_target": "R3 robust clear",
        "leaf_state_refit": True,
    }
    return reference, audit


def _leaf_state_from_reference(reference: LeafReference, score: pd.DataFrame) -> pd.DataFrame:
    leaf_score = reference.leaves(score)
    support = np.zeros_like(leaf_score, dtype=np.float32)
    coverage = np.zeros_like(leaf_score, dtype=bool)
    contribution = np.zeros_like(leaf_score, dtype=np.float32)
    for tree, counts in enumerate(reference.support_counts):
        token = leaf_score[:, tree]
        good = token < len(counts)
        support[good, tree] = counts[token[good]]
        coverage[:, tree] = good
        values = None if reference.leaf_values is None else reference.leaf_values[tree]
        if values is not None:
            value_good = token < len(values)
            contribution[value_good, tree] = np.abs(values[token[value_good]])
    surprise = -np.log(
        np.clip((support + 1.0) / (reference.train_rows + 1.0), 1e-12, 1.0)
    )
    contribution_total = contribution.sum(axis=1)
    contribution_weighted = np.divide(
        (support * contribution).sum(axis=1),
        contribution_total,
        out=support.mean(axis=1),
        where=contribution_total > 1e-12,
    )
    contribution_weighted_log = np.divide(
        (np.log1p(support) * contribution).sum(axis=1),
        contribution_total,
        out=np.log1p(support).mean(axis=1),
        where=contribution_total > 1e-12,
    )
    high_contribution = contribution >= np.quantile(
        contribution, 0.75, axis=1, keepdims=True,
    )
    high_contribution_support = np.where(high_contribution, support, np.nan)
    effective_contributors = np.divide(
        contribution_total**2,
        np.square(contribution).sum(axis=1),
        out=np.zeros_like(contribution_total),
        where=np.square(contribution).sum(axis=1) > 1e-12,
    )
    return pd.DataFrame(
        {
            "leaf_support_effective": support.mean(axis=1),
            "leaf_support_p05": np.quantile(support, 0.05, axis=1),
            "leaf_support_p50": np.quantile(support, 0.50, axis=1),
            "leaf_support_p95": np.quantile(support, 0.95, axis=1),
            "leaf_support_contribution_weighted": contribution_weighted,
            "leaf_support_contribution_weighted_log": contribution_weighted_log,
            "leaf_support_high_contribution_min": np.nanmin(
                high_contribution_support, axis=1,
            ),
            "leaf_contributor_effective_n": effective_contributors,
            "leaf_support_adequate_fraction": (support >= 30.0).mean(axis=1),
            "leaf_support_leaf_coverage": coverage.mean(axis=1),
            "leaf_ood_marginal": surprise.mean(axis=1),
            "leaf_ood_joint": surprise.sum(axis=1),
            "leaf_ood_joint_rms": np.sqrt(np.mean(np.square(surprise), axis=1)),
        },
        index=score.index,
    ).astype(np.float32)


def _leaf_state(
    train: pd.DataFrame, score: pd.DataFrame, fields: Sequence[str],
) -> tuple[pd.DataFrame, dict[str, object]]:
    reference, audit = _fit_leaf_reference(train, fields)
    return _leaf_state_from_reference(reference, score), audit



def _state_features(frame: pd.DataFrame, k9: pd.DataFrame, leaf: pd.DataFrame) -> pd.DataFrame:
    dynamic = _dynamic_k9_state(frame, k9)
    return pd.concat([k9.reset_index(drop=True), dynamic.reset_index(drop=True), leaf.reset_index(drop=True)], axis=1).astype(np.float32)


def _impute_pair(train: pd.DataFrame, score: pd.DataFrame, fields: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    x_train, med = _numeric(train, fields)
    x_score, _ = _numeric(score, fields, med)
    return x_train, x_score


def _fit_safety(
    train: pd.DataFrame,
    score: pd.DataFrame,
    fields: Sequence[str],
) -> tuple[np.ndarray, dict[str, object]]:
    target_column = "exact_h12_net_bps"
    threshold_bps = -200.0
    target_values = pd.to_numeric(train[target_column], errors="coerce")
    if not np.isfinite(target_values).all():
        raise ValueError(f"safety target {target_column} contains non-finite rows")
    target = target_values.le(float(threshold_bps)).astype(np.int8).to_numpy()
    x_train, x_score = _impute_pair(train, score, fields)
    model = LGBMClassifier(
        objective="binary", n_estimators=35, learning_rate=0.0444772418995553,
        max_depth=5, num_leaves=15, min_child_samples=max(103, int(0.01 * len(train))),
        colsample_bytree=0.7393319822815638, subsample=0.7853518403594505,
        subsample_freq=1, reg_alpha=0.02534130367151813, reg_lambda=16.57892339556902,
        max_bin=127, random_state=SEED + 199, n_jobs=4, verbosity=-1,
    ).fit(x_train, target)
    return model.predict_proba(x_score)[:, 1].astype(np.float32), {
        "safety_fit_rows": int(len(train)),
        "safety_positive_rate": float(target.mean()),
        "safety_features": int(len(fields)),
        "safety_target": target_column,
        "safety_threshold_bps": float(threshold_bps),
    }


def _groups(frame: pd.DataFrame) -> np.ndarray:
    key = pd.to_datetime(frame["__decision_ts__"], utc=True).dt.floor("4h").astype("int64")
    return key.groupby(key, sort=False).size().to_numpy(np.int32)


def _fit_correctness(train: pd.DataFrame, score: pd.DataFrame, fields: Sequence[str]) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    target = (pd.to_numeric(train["policy_net_bps"], errors="coerce") - pd.to_numeric(train["base_anchor_bps"], errors="coerce") > 100.0).astype(np.int8).to_numpy()
    ordered = train.assign(_query=pd.to_datetime(train["__decision_ts__"], utc=True).dt.floor("4h")).sort_values(["_query", "__decision_ts__", "candidate_id"], kind="stable")
    x_train, x_score = _impute_pair(ordered, score, fields)
    model = LGBMRanker(
        objective="lambdarank", n_estimators=120, learning_rate=0.035, max_depth=4,
        num_leaves=15, min_child_samples=max(120, int(0.03 * len(ordered))),
        colsample_bytree=0.80, subsample=0.82, subsample_freq=1, reg_alpha=0.05,
        reg_lambda=5.0, max_bin=127, label_gain=[0, 1], lambdarank_truncation_level=10,
        random_state=SEED + 211, n_jobs=4, verbosity=-1,
    )
    y = (pd.to_numeric(ordered["policy_net_bps"], errors="coerce") - pd.to_numeric(ordered["base_anchor_bps"], errors="coerce") > 100.0).astype(np.int8).to_numpy()
    model.fit(x_train, y, group=_groups(ordered))
    raw_train = model.predict(x_train)
    raw_score = model.predict(x_score)
    return raw_score.astype(np.float32), _pct(raw_train, raw_score), {"correctness_fit_rows": int(len(ordered)), "correctness_positive_rate": float(y.mean()), "correctness_features": int(len(fields)), "correctness_query": "4h UTC"}


def _geometry_window(arm: str, cutoff: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    """Return [fit start, fit end) and earliest allowed residual/meta row."""
    if arm == "c3_frozen":
        # The approved raw-contract archive has no 2024 observations.  These
        # are therefore the first three complete available months; all
        # downstream rows begin after this 2025-01:03 burn-in.
        start, end = pd.Timestamp("2025-01-01", tz="UTC"), pd.Timestamp("2025-04-01", tz="UTC")
    elif arm == "c3_quarterly":
        quarter_start = cutoff.to_period("Q").start_time.tz_localize("UTC")
        start, end = _month_add(quarter_start, -12), _month_add(quarter_start, -9)
    elif arm == "c3_rolling":
        start, end = _month_add(cutoff, -12), _month_add(cutoff, -9)
    else:
        raise ValueError(f"unknown causal arm {arm}")
    return start, end, end


def _base_coupled_lookback_months(arm: str) -> int | None:
    """Return the recent in-sample K9 horizon encoded by a diagnostic arm."""
    prefix = "basecoupled_in_sample_"
    if not arm.startswith(prefix) or not arm.endswith("m"):
        return None
    value = arm[len(prefix):-1]
    if not value.isdigit() or int(value) not in {3, 6, 9}:
        raise ValueError(f"unsupported base-coupled K9 arm: {arm}")
    return int(value)


def _metrics(predictions: pd.DataFrame, score: str, arm: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    monthly: list[dict[str, object]] = []
    for scope, frame in [("global", predictions), *[(month, block) for month, block in predictions.groupby("month", sort=True)]]:
        valid = frame.loc[np.isfinite(frame[score]) & np.isfinite(frame["policy_net_bps"])].copy()
        if valid.empty:
            continue
        for tail in TAILS:
            count = max(1, int(math.ceil(tail * len(valid))))
            selected = valid.nlargest(count, score, keep="first")
            payload = {
                "arm": arm, "scope": scope, "tail": tail, "rows": int(len(valid)), "trades": int(len(selected)),
                "net_bps_per_trade": float(selected["policy_net_bps"].mean()),
                "gross_bps_per_trade": float(selected["policy_gross_bps"].mean()),
                "net_pnl_bps": float(selected["policy_net_bps"].sum()),
                "days": int(selected["__decision_ts__"].dt.floor("D").nunique()),
                "trades_per_day": float(len(selected) / max(selected["__decision_ts__"].dt.floor("D").nunique(), 1)),
            }
            (rows if scope == "global" else monthly).append(payload)
        ic = spearmanr(valid[score], valid["policy_net_bps"], nan_policy="omit").statistic
        payload = {"arm": arm, "scope": scope, "tail": -1.0, "rows": int(len(valid)), "trades": int(len(valid)), "net_bps_per_trade": float(ic), "gross_bps_per_trade": np.nan, "net_pnl_bps": np.nan, "days": int(valid["__decision_ts__"].dt.floor("D").nunique()), "trades_per_day": np.nan}
        (rows if scope == "global" else monthly).append(payload)
    return pd.DataFrame(rows), pd.DataFrame(monthly)


def _run_raw_arm(
    arm: str,
    main: pd.DataFrame,
    fields: Sequence[str],
    held_months: Sequence[str],
    *,
    raw_cache: dict[tuple[pd.Timestamp, pd.Timestamp], pd.DataFrame],
    previous_by_arm: dict[str, RawK9Bundle],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    output: list[pd.DataFrame] = []
    audits: list[dict[str, object]] = []
    bundle_audits: list[dict[str, object]] = []
    for held_month in held_months:
        cutoff = _month_start(held_month)
        held = main.loc[main["month"].eq(held_month)].copy()
        reference = main.loc[
            main["__decision_ts__"].ge(cutoff - pd.Timedelta(days=42)) & main["__decision_ts__"].lt(cutoff)
        ].copy()
        if arm == "in_sample_k9":
            fit_start = max(main["__decision_ts__"].min(), cutoff - pd.DateOffset(months=9))
            fit_end = cutoff
            geometry_source = main.loc[(main["__decision_ts__"] >= fit_start) & (main["__decision_ts__"] < fit_end)].copy()
            allowed_start = fit_start
            source_kind = "in_sample_meta_training_rows"
        else:
            fit_start, fit_end, allowed_start = _geometry_window(arm, cutoff)
            cache_key = (fit_start, fit_end)
            if cache_key not in raw_cache:
                raw_cache[cache_key] = _read_features(fit_start, fit_end, fields)
            geometry_source = raw_cache[cache_key]
            source_kind = "raw_market_burn_in_only"
        if held.empty or reference.empty:
            audits.append({"arm": arm, "held_month": held_month, "status": "skipped_empty_held_or_reference"})
            continue
        bundle_id = f"{arm}__{fit_start:%Y%m}__{fit_end:%Y%m}"
        previous = previous_by_arm.get(arm)
        bundle, bundle_audit = _fit_raw_k9(
            geometry_source, fields, bundle_id=bundle_id, fit_start=fit_start, fit_end=fit_end,
            source_kind=source_kind, previous=previous,
        )
        previous_by_arm[arm] = bundle
        bundle_audit.update({"arm": arm, "held_month": held_month})
        bundle_audits.append(bundle_audit)
        meta_train = main.loc[
            main["label_available_ts"].lt(cutoff)
            & main["__decision_ts__"].ge(allowed_start)
        ].copy()
        if not meta_train.empty and not meta_train["__decision_ts__"].ge(allowed_start).all():
            raise AssertionError("downstream meta training crossed the geometry/K9 bundle boundary")
        if len(meta_train) < 5_000:
            audits.append({"arm": arm, "held_month": held_month, "status": "skipped_insufficient_post_bundle_meta_rows", "meta_train_rows": int(len(meta_train)), "bundle_id": bundle_id})
            continue
        # The base-dependent state refits every held month; its training span is
        # independent of K9, while every downstream meta row obeys allowed_start.
        leaf_train = main.loc[
            main["__label_available_at__"].lt(cutoff)
            & main["r3_label_valid"].fillna(False).astype(bool)
        ].copy()
        score_population = pd.concat([meta_train, reference, held], ignore_index=True).drop_duplicates("candidate_id", keep="last")
        leaf_values, leaf_audit = _leaf_state(leaf_train, score_population, fields)
        k9_values = bundle.transform(score_population)
        state = _state_features(score_population, k9_values, leaf_values)
        state.index = score_population["candidate_id"].to_numpy()
        state_fields = ["base_score", "base_anchor_bps", "base_rank", "consensus_rank", "final_score", *state.columns.tolist()]
        meta_state = meta_train[["candidate_id"]].join(state, on="candidate_id")
        ref_state = reference[["candidate_id"]].join(state, on="candidate_id")
        held_state = held[["candidate_id"]].join(state, on="candidate_id")
        meta_fit = meta_train.join(meta_state.drop(columns="candidate_id"))
        ref_fit = reference.join(ref_state.drop(columns="candidate_id"))
        held_fit = held.join(held_state.drop(columns="candidate_id"))
        meta_fit = _equal_month_sample(meta_fit, MODEL_CAP, seed=SEED + 241)
        score_fit = pd.concat([ref_fit.assign(_role="reference"), held_fit.assign(_role="held")], ignore_index=True)
        severe, severe_audit = _fit_safety(meta_fit, score_fit, state_fields)
        score_fit["severe200_probability"] = severe
        score_fit["raw_severe"] = score_fit["final_score"].to_numpy(float) * (1.0 - 0.5 * severe)
        correctness_raw, correctness_pct, correctness_audit = _fit_correctness(meta_fit, score_fit, state_fields)
        score_fit["correctness_raw"] = correctness_raw
        score_fit["correctness_percentile"] = correctness_pct
        score_fit["raw_correctness_demote"] = score_fit["raw_severe"] * (0.25 + 0.75 * correctness_pct)
        ref_mask = score_fit["_role"].eq("reference").to_numpy()
        score_fit["c3_score"] = _pct(score_fit.loc[ref_mask, "raw_correctness_demote"].to_numpy(float), score_fit["raw_correctness_demote"].to_numpy(float))
        held_output = score_fit.loc[~ref_mask].copy()
        held_output["arm"] = arm
        held_output["geometry_bundle_id"] = bundle_id
        held_output["geometry_fit_start"] = fit_start
        held_output["geometry_fit_end"] = fit_end
        held_output["meta_training_start"] = allowed_start
        output.append(held_output)
        audits.append({
            "arm": arm, "held_month": held_month, "status": "complete", "bundle_id": bundle_id,
            "held_rows": int(len(held)), "reference_rows": int(len(reference)), "meta_train_rows": int(len(meta_fit)),
            "meta_training_start": allowed_start, "meta_training_old_rows_forbidden": True,
            "reference_pre_bundle_rows": int((reference["__decision_ts__"] < allowed_start).sum()),
            **leaf_audit, **severe_audit, **correctness_audit,
        })
    if not output:
        raise RuntimeError(f"{arm}: no held-month predictions completed")
    return pd.concat(output, ignore_index=True), pd.DataFrame(audits), pd.DataFrame(bundle_audits)



def _run_base_coupled_arm(
    arm: str,
    main: pd.DataFrame,
    fields: Sequence[str],
    held_months: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the deliberately in-sample, same-base leaf-geometry challenger."""
    lookback_months = _base_coupled_lookback_months(arm)
    if lookback_months is None:
        raise ValueError(f"{arm} is not a base-coupled diagnostic arm")
    output: list[pd.DataFrame] = []
    audits: list[dict[str, object]] = []
    bundle_audits: list[dict[str, object]] = []
    all_start = main["__decision_ts__"].min()

    for held_month in held_months:
        cutoff = _month_start(held_month)
        held = main.loc[main["month"].eq(held_month)].copy()
        reference = main.loc[
            main["__decision_ts__"].ge(cutoff - pd.Timedelta(days=42))
            & main["__decision_ts__"].lt(cutoff)
        ].copy()
        meta_train = main.loc[main["label_available_ts"].lt(cutoff)].copy()
        if held.empty or reference.empty or len(meta_train) < 5_000:
            audits.append({
                "arm": arm,
                "held_month": held_month,
                "status": "skipped_insufficient_held_reference_or_meta",
                "meta_train_rows": int(len(meta_train)),
            })
            continue

        # This fitted R3 reference is shared exactly by the active-leaf health
        # block and the K9 leaf-path geometry. K9 fit rows are intentionally
        # retained in the downstream meta fit for this diagnostic.
        leaf_train = main.loc[
            main["__label_available_at__"].lt(cutoff)
            & main["r3_label_valid"].fillna(False).astype(bool)
        ].copy()
        leaf_reference, leaf_audit = _fit_leaf_reference(leaf_train, fields)
        fit_start = cutoff - pd.DateOffset(months=lookback_months)
        coupled_fit = leaf_train.loc[
            leaf_train["__decision_ts__"].ge(fit_start)
            & leaf_train["__decision_ts__"].lt(cutoff)
        ].copy()
        if len(coupled_fit) < 1_000:
            audits.append({
                "arm": arm,
                "held_month": held_month,
                "status": "skipped_insufficient_recent_base_geometry_rows",
                "geometry_rows": int(len(coupled_fit)),
            })
            continue

        bundle_id = f"{arm}__{fit_start:%Y%m}__{cutoff:%Y%m}"
        bundle, bundle_audit = _fit_base_coupled_k9(
            leaf_reference,
            coupled_fit,
            bundle_id=bundle_id,
            fit_start=fit_start,
            fit_end=cutoff,
            lookback_months=lookback_months,
        )
        bundle_audit.update({"arm": arm, "held_month": held_month})
        bundle_audits.append(bundle_audit)

        score_population = (
            pd.concat([meta_train, reference, held], ignore_index=True)
            .drop_duplicates("candidate_id", keep="last")
        )
        leaf_values = _leaf_state_from_reference(leaf_reference, score_population)
        k9_values = bundle.transform(score_population)
        state = _state_features(score_population, k9_values, leaf_values)
        state.index = score_population["candidate_id"].to_numpy()
        state_fields = [
            "base_score",
            "base_anchor_bps",
            "base_rank",
            "consensus_rank",
            "final_score",
            *state.columns.tolist(),
        ]

        meta_state = meta_train[["candidate_id"]].join(state, on="candidate_id")
        ref_state = reference[["candidate_id"]].join(state, on="candidate_id")
        held_state = held[["candidate_id"]].join(state, on="candidate_id")
        meta_fit = meta_train.join(meta_state.drop(columns="candidate_id"))
        ref_fit = reference.join(ref_state.drop(columns="candidate_id"))
        held_fit = held.join(held_state.drop(columns="candidate_id"))
        meta_fit = _equal_month_sample(meta_fit, MODEL_CAP, seed=SEED + 241)
        score_fit = pd.concat(
            [ref_fit.assign(_role="reference"), held_fit.assign(_role="held")],
            ignore_index=True,
        )

        severe, severe_audit = _fit_safety(meta_fit, score_fit, state_fields)
        score_fit["severe200_probability"] = severe
        score_fit["raw_severe"] = (
            score_fit["final_score"].to_numpy(float) * (1.0 - 0.5 * severe)
        )
        correctness_raw, correctness_pct, correctness_audit = _fit_correctness(
            meta_fit, score_fit, state_fields,
        )
        score_fit["correctness_raw"] = correctness_raw
        score_fit["correctness_percentile"] = correctness_pct
        score_fit["raw_correctness_demote"] = (
            score_fit["raw_severe"] * (0.25 + 0.75 * correctness_pct)
        )
        reference_mask = score_fit["_role"].eq("reference").to_numpy()
        score_fit["c3_score"] = _pct(
            score_fit.loc[reference_mask, "raw_correctness_demote"].to_numpy(float),
            score_fit["raw_correctness_demote"].to_numpy(float),
        )
        held_output = score_fit.loc[~reference_mask].copy()
        held_output["arm"] = arm
        held_output["geometry_bundle_id"] = bundle_id
        held_output["geometry_fit_start"] = fit_start
        held_output["geometry_fit_end"] = cutoff
        held_output["meta_training_start"] = all_start
        held_output["geometry_uses_base_labels"] = True
        held_output["geometry_lookback_months"] = lookback_months
        output.append(held_output)

        audits.append({
            "arm": arm,
            "held_month": held_month,
            "status": "complete",
            "bundle_id": bundle_id,
            "held_rows": int(len(held)),
            "reference_rows": int(len(reference)),
            "meta_train_rows": int(len(meta_fit)),
            "meta_training_start": all_start,
            "meta_training_old_rows_forbidden": False,
            "reference_pre_bundle_rows": 0,
            "geometry_uses_base_labels": True,
            "geometry_lookback_months": lookback_months,
            "same_leaf_reference_for_k9_and_state": True,
            "in_sample_meta_rows_allowed": True,
            **leaf_audit,
            **severe_audit,
            **correctness_audit,
        })

    if not output:
        raise RuntimeError(f"{arm}: no held-month predictions completed")
    return (
        pd.concat(output, ignore_index=True),
        pd.DataFrame(audits),
        pd.DataFrame(bundle_audits),
    )


def _run_arm(
    arm: str,
    main: pd.DataFrame,
    fields: Sequence[str],
    held_months: Sequence[str],
    *,
    raw_cache: dict[tuple[pd.Timestamp, pd.Timestamp], pd.DataFrame],
    previous_by_arm: dict[str, RawK9Bundle],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if _base_coupled_lookback_months(arm) is not None:
        return _run_base_coupled_arm(arm, main, fields, held_months)
    return _run_raw_arm(
        arm,
        main,
        fields,
        held_months,
        raw_cache=raw_cache,
        previous_by_arm=previous_by_arm,
    )

def _write_report(out: Path, global_metrics: pd.DataFrame, monthly: pd.DataFrame, audit: pd.DataFrame, bundles: pd.DataFrame) -> None:
    def _table(frame: pd.DataFrame, *, index: bool = False) -> str:
        view = frame.reset_index() if index else frame.reset_index(drop=True)
        columns = [str(column) for column in view.columns]
        header = "| " + " | ".join(columns) + " |"
        rule = "| " + " | ".join(["---"] * len(columns)) + " |"
        values = []
        for row in view.itertuples(index=False, name=None):
            values.append("| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |")
        return "\n".join([header, rule, *values])

    lines = [
        "# C3 Causal Geometry/K9 Cadence Ablation", "",
        "## Contract", "",
        "- Upstream strict-R3 base plus ten-head consensus handoff is held fixed.",
        "- K9 is unsupervised and base-independent: it uses only the raw frozen 120-field causal contract.",
        "- `basecoupled_in_sample_{3,6,9}m` is a separate, deliberately riskier diagnostic: K9 clusters the recent leaf paths of the same current R3 reference model and lets those K9-fit rows remain in meta fitting.",
        "- The 64-round leaf-state model is refit for each held month and yields only aggregate support/OOD; raw leaf identities never cross folds.",
        "- For causal C3 arms, Severe and Correctness training rows begin strictly after that arm's geometry burn-in ends.",
        "- Policy evaluation uses the fixed SL3 / activation 0.5 ATR / giveback 0.25 ATR / H12 / 100-bps-once replay.",
        "",
        "## Global policy-net metrics", "",
    ]
    view = global_metrics.loc[global_metrics["tail"].ge(0)].copy()
    if not view.empty:
        pivot = view.pivot(index="arm", columns="tail", values="net_bps_per_trade").rename(columns=lambda x: f"Top {x:.1%}")
        lines += [_table(pivot.round(2), index=True), ""]
    ic = global_metrics.loc[global_metrics["tail"].eq(-1.0), ["arm", "net_bps_per_trade"]].rename(columns={"net_bps_per_trade": "policy_net_rank_ic"})
    lines += ["## Rank IC", "", _table(ic.round(4)), "", "## Fold lineage", ""]
    cols = [column for column in ("arm", "held_month", "bundle_id", "geometry_fit_start", "geometry_fit_end", "meta_training_start", "meta_train_rows", "status") if column in audit]
    lines += [_table(audit.loc[:, cols]), "", "## Geometry bundle stability", ""]
    if not bundles.empty:
        cols = ["arm", "held_month", "bundle_id", "fit_start", "fit_end_exclusive", "fit_rows", "matched_center_cosine", "matched_center_distance"]
        lines += [_table(bundles.loc[:, [c for c in cols if c in bundles]].round(4)), ""]
    lines += [
        "## Interpretation guardrails", "",
        "This is a matched downstream-state ablation, not an end-to-end replacement of the persisted historical canonical artifact: the upstream base/consensus handoff is intentionally reused. A positive C3 result supports the causal state cadence and post-burn-in restriction. A positive base-coupled result establishes only a bounded same-base diagnostic because its recent K9 fit rows remain in the downstream training set; it is not production geometry evidence.",
    ]
    (out / "CAUSAL_GEOMETRY_K9_C3_ABLATION_REPORT.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--months", default="2026-04:2026-07")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data_perp/artifacts/causal_geometry_k9_c3_ablation_20260810_v1")
    parser.add_argument("--arms", default="in_sample_k9,c3_frozen,c3_quarterly,c3_rolling")
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable output exists: {args.out_dir}")
    held_months = _months(args.months)
    all_months = pd.period_range("2025-02", max(held_months), freq="M").astype(str).tolist()
    fields = _feature_fields()
    main_panel = _load_main(fields, all_months)
    args.out_dir.mkdir(parents=True)
    raw_cache: dict[tuple[pd.Timestamp, pd.Timestamp], pd.DataFrame] = {}
    prior: dict[str, RawK9Bundle] = {}
    predictions: list[pd.DataFrame] = []
    audits: list[pd.DataFrame] = []
    bundle_audits: list[pd.DataFrame] = []
    metric_global: list[pd.DataFrame] = []
    metric_monthly: list[pd.DataFrame] = []
    for arm in [token.strip() for token in args.arms.split(",") if token.strip()]:
        print(json.dumps({"event": "arm_start", "arm": arm, "months": held_months}), flush=True)
        arm_prediction, arm_audit, arm_bundles = _run_arm(arm, main_panel, fields, held_months, raw_cache=raw_cache, previous_by_arm=prior)
        # The upstream score appears exactly once as the matched control on the
        # same successfully-scored candidate rows for each arm.
        arm_prediction["upstream_core_score"] = arm_prediction["final_score"].astype(np.float32)
        predictions.append(arm_prediction)
        audits.append(arm_audit)
        bundle_audits.append(arm_bundles)
        for score, name in (("upstream_core_score", f"{arm}__upstream_control"), ("c3_score", arm)):
            global_part, monthly_part = _metrics(arm_prediction, score, name)
            metric_global.append(global_part)
            metric_monthly.append(monthly_part)
        print(json.dumps({"event": "arm_complete", "arm": arm, "rows": len(arm_prediction)}), flush=True)
    pred = pd.concat(predictions, ignore_index=True)
    audit = pd.concat(audits, ignore_index=True)
    bundles = pd.concat(bundle_audits, ignore_index=True)
    global_metrics = pd.concat(metric_global, ignore_index=True)
    monthly_metrics = pd.concat(metric_monthly, ignore_index=True)
    pred.to_parquet(args.out_dir / "predictions.parquet", index=False, compression="zstd")
    audit.to_parquet(args.out_dir / "fold_audit.parquet", index=False)
    bundles.to_parquet(args.out_dir / "geometry_bundle_audit.parquet", index=False)
    global_metrics.to_parquet(args.out_dir / "metrics_global.parquet", index=False)
    monthly_metrics.to_parquet(args.out_dir / "metrics_monthly.parquet", index=False)
    _write_report(args.out_dir, global_metrics, monthly_metrics, audit, bundles)
    manifest = {
        "schema": "causal_geometry_k9_c3_ablation_v1",
        "held_months": held_months, "arms": args.arms,
        "upstream": str(UPSTREAM_PATH), "policy_outcomes": str(POLICY_PATH),
        "feature_path": str(FEATURE_PATH), "raw_feature_count": len(fields),
        "market_geometry": "C3 arms: base-independent unsupervised raw-120 K9; basecoupled arms: current-R3 leaf-path K9 with in-sample meta rows allowed",
        "leaf_state": "monthly refit 64-round R3-clear model; aggregate support/OOD only",
        "meta_restriction": "causal arms train Severe and Correctness strictly after geometry burn-in",
        "entry": "signal close + 1h first 15m open",
        "exit": "SL3 ATR; trail activation 0.5 ATR; giveback 0.25 ATR; H12",
        "cost_bps_once": 100.0,
        "rows": int(len(pred)),
    }
    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    print(json.dumps({"event": "complete", "output": str(args.out_dir), "rows": len(pred)}))


if __name__ == "__main__":
    main()
