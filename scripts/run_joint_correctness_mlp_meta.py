#!/usr/bin/env python3
"""Discover causal latent joint-correctness states and test an MLP meta layer.

This experiment is deliberately self contained around the frozen long-side
structural sidecar.  It rebuilds ten side-local LambdaRank heads (five causal
feature caps x ordinary/equal-month weighting), materialises leaf-derived
correctness probabilities, discovers joint activation states from resolved
history, and trains an MLP on inference-available inputs.  The MLP is tested
both as a replacement score and as a weight on the normalized head consensus.

The outer sidecar partitions are respected as follows:

* ``meta_train`` is split chronologically into ``head_fit`` and ``state_train``;
* heads and leaf correctness maps are fit only on ``head_fit``;
* latent state discovery and MLP fitting use only ``state_train``;
* isotonic EV maps are fit only on ``meta_calibration``;
* ``test`` is scored once and is never used for target/state/model selection.

All reported tails are pooled global tails over the outer test rows, not
per-timestamp top-k.  The source sidecar is long-only; no short-side result is
implied by this runner.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Iterable
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import rankdata, spearmanr
from sklearn.cluster import KMeans
from sklearn.feature_selection import mutual_info_classif
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss, silhouette_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", message="X does not have valid feature names")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCHEMA = "joint_correctness_mlp_meta_v1"
SIDE = "long"
TAILS = (0.005, 0.01, 0.02, 0.05, 0.10, 0.20)
CAPS = (40, 60, 80, 100, 120)
WEIGHTS = ("ordinary", "equal_month")
TARGET_ARMS = ("residual_ordinal", "net_ordinal", "clear_binary", "query_rank_ordinal")
STATE_KS = (3, 4, 5, 6)

DEFAULT_SIDECAR = ROOT / "data_perp/artifacts/long_structural_tree_meta_sidecar_20260804_v4/tree_meta_candidate_sidecar.parquet"
DEFAULT_OUT = ROOT / "data_perp/artifacts/joint_correctness_mlp_meta_20260806_v1"

IDENTITY = {
    "candidate_id", "__ts__", "side_name", "gross_bps", "net_bps", "label_valid",
    "barrier_relevance_0_5", "mfe_mae_label_valid", "atr_bps", "label_available_ts",
    "query_id", "meta_partition", "fold", "feature_contract_sha256", "base_raw_score",
    "base_expected_bps", "month", "year", "era",
}


@dataclass
class HeadFit:
    name: str
    model: lgb.LGBMRanker
    fields: list[str]
    medians: pd.Series
    leaf_maps: list[dict[int, tuple[float, int]]]
    train_score: np.ndarray
    train_score_quantiles: np.ndarray
    train_outcome_quantiles: np.ndarray
    target_name: str
    weight_name: str


def _jsonable(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    raise TypeError(type(value).__name__)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, default=_jsonable) + "\n")


def _digest(values: Iterable[str]) -> str:
    return hashlib.sha256(json.dumps(list(values), separators=(",", ":")).encode()).hexdigest()


def _finite_rank(values: np.ndarray, reference: np.ndarray | None = None) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    ref = arr if reference is None else np.asarray(reference, dtype=float)
    ref = ref[np.isfinite(ref)]
    if len(ref) < 2:
        return np.full(len(arr), 0.5, dtype=float)
    # Empirical CDF using training-only reference values.  This is stable for
    # ties and does not fit a future-period normalization.
    ordered = np.sort(ref)
    positions = np.searchsorted(ordered, arr, side="right")
    out = positions.astype(float) / float(len(ordered))
    out[~np.isfinite(arr)] = 0.5
    return np.clip(out, 0.0, 1.0)


def _grade(values: np.ndarray, kind: str, query: pd.Series | None = None) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    if kind == "clear_binary":
        return (x > 50.0).astype(np.int32)
    if kind == "residual_ordinal" or kind == "net_ordinal":
        return np.select(
            [x <= -150.0, x <= -50.0, x < 50.0, x < 150.0],
            [0, 1, 2, 3], default=4,
        ).astype(np.int32)
    if kind == "query_rank_ordinal":
        if query is None:
            raise ValueError("query_rank_ordinal requires query IDs")
        out = np.zeros(len(x), dtype=np.int32)
        q = pd.Series(query).astype(str).to_numpy()
        for key in pd.unique(q):
            idx = np.flatnonzero(q == key)
            finite = np.isfinite(x[idx])
            if finite.sum() < 2:
                out[idx] = 2
                continue
            ranks = rankdata(x[idx][finite], method="average") / float(finite.sum())
            bins = np.minimum(4, (ranks * 5.0).astype(int))
            out[idx[finite]] = bins
            out[idx[~finite]] = 2
        return out
    raise ValueError(f"unknown target arm {kind}")


def _target(frame: pd.DataFrame, kind: str) -> np.ndarray:
    net = pd.to_numeric(frame["net_bps"], errors="coerce").to_numpy(float)
    base = pd.to_numeric(frame["base_expected_bps"], errors="coerce").to_numpy(float)
    if kind == "residual_ordinal":
        return _grade(net - base, kind)
    if kind == "net_ordinal" or kind == "clear_binary":
        return _grade(net, kind)
    return _grade(net - base, kind, frame["query_id"])


def _feature_columns(frame: pd.DataFrame) -> list[str]:
    """Extract the frozen causal raw/AE-GMM block, excluding derived health."""
    cols = list(frame.columns)
    try:
        start = cols.index("atr_bps") + 1
        end = cols.index("label_available_ts")
    except ValueError as exc:
        raise ValueError("sidecar does not expose the expected causal feature block") from exc
    fields = []
    for name in cols[start:end]:
        if name in IDENTITY or name.startswith("base_reasoning__") or name.startswith("base_structural_family__"):
            continue
        if name.startswith("structural_health__") or name.startswith("target__"):
            continue
        values = pd.to_numeric(frame[name], errors="coerce")
        if values.notna().mean() >= 0.90 and values.nunique(dropna=True) > 1:
            fields.append(name)
    if len(fields) < 40:
        raise ValueError(f"causal feature contract unexpectedly small: {len(fields)}")
    return fields


def _matrix(frame: pd.DataFrame, fields: list[str], medians: pd.Series | None = None) -> tuple[np.ndarray, pd.Series]:
    values = frame.loc[:, fields].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    med = values.median() if medians is None else medians.reindex(fields)
    med = med.fillna(0.0)
    values = values.fillna(med).clip(-1e6, 1e6)
    return values.to_numpy("float32"), med


def _feature_order(train: pd.DataFrame, fields: list[str], target: np.ndarray) -> list[str]:
    ranked: list[tuple[float, str]] = []
    y = np.asarray(target, dtype=float)
    for field in fields:
        x = pd.to_numeric(train[field], errors="coerce").to_numpy(float)
        valid = np.isfinite(x) & np.isfinite(y)
        if valid.sum() < 200 or np.unique(x[valid]).size < 2:
            continue
        corr = spearmanr(x[valid], y[valid]).statistic
        ranked.append((abs(float(corr)) if np.isfinite(corr) else -np.inf, field))
    ranked.sort(key=lambda pair: (-pair[0], pair[1]))
    if len(ranked) < max(CAPS):
        # Preserve the contract even when a particular target is weak.
        remaining = [field for field in fields if field not in {name for _, name in ranked}]
        ranked.extend((0.0, field) for field in remaining)
    return [field for _, field in ranked]


def _query_sort(frame: pd.DataFrame, labels: np.ndarray) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    q = frame["query_id"].astype(str)
    order = np.lexsort((frame["candidate_id"].astype(str).to_numpy(), frame["__ts__"].to_numpy(), q.to_numpy()))
    sorted_frame = frame.iloc[order].copy()
    y = np.asarray(labels)[order]
    groups = sorted_frame.groupby(sorted_frame["query_id"].astype(str), sort=False, observed=True).size().to_numpy(dtype=np.int32)
    return sorted_frame, y, groups


def _sample_weights(frame: pd.DataFrame, mode: str) -> np.ndarray:
    if mode == "ordinary":
        return np.ones(len(frame), dtype=np.float32)
    month = pd.to_datetime(frame["__ts__"], utc=True).dt.strftime("%Y-%m")
    counts = month.value_counts()
    weight = month.map((len(counts) / counts).to_dict()).to_numpy(float)
    weight = weight / max(weight.mean(), 1e-9)
    return np.clip(weight, 0.25, 4.0).astype(np.float32)


def _fit_ranker(train: pd.DataFrame, fields: list[str], labels: np.ndarray, weight_mode: str, seed: int) -> tuple[lgb.LGBMRanker, pd.Series]:
    ordered, y, groups = _query_sort(train, labels)
    x, med = _matrix(ordered, fields)
    weights = _sample_weights(ordered, weight_mode)
    model = lgb.LGBMRanker(
        objective="lambdarank", metric="ndcg", lambdarank_truncation_level=10,
        n_estimators=55, learning_rate=0.045, num_leaves=15, max_depth=4,
        min_child_samples=max(100, int(0.003 * len(ordered))),
        min_sum_hessian_in_leaf=1.0, feature_fraction=0.8, bagging_fraction=0.8,
        bagging_freq=1, lambda_l1=0.1, lambda_l2=8.0, max_bin=63,
        random_state=seed, n_jobs=2, verbosity=-1,
    )
    model.fit(x, y, group=groups, sample_weight=weights)
    return model, med


def _predict(model: lgb.LGBMRanker, frame: pd.DataFrame, fields: list[str], medians: pd.Series, *, leaf: bool = False):
    x, _ = _matrix(frame, fields, medians)
    return model.predict(x, pred_leaf=leaf)


def _fit_leaf_maps(model: lgb.LGBMRanker, train: pd.DataFrame, fields: list[str], medians: pd.Series, target: np.ndarray) -> tuple[list[dict[int, tuple[float, int]]], np.ndarray, np.ndarray, np.ndarray]:
    score = np.asarray(_predict(model, train, fields, medians), dtype=float)
    score_pct = _finite_rank(score)
    outcome = pd.to_numeric(train["net_bps"], errors="coerce").to_numpy(float) - pd.to_numeric(train["base_expected_bps"], errors="coerce").to_numpy(float)
    outcome_pct = _finite_rank(outcome)
    correctness = np.exp(-np.abs(score_pct - outcome_pct) / 0.25)
    # A stricter correctness bit is retained for state semantics and metrics.
    correct_bit = (np.abs(score_pct - outcome_pct) <= 0.25).astype(float)
    leaf_values = np.asarray(_predict(model, train, fields, medians, leaf=True), dtype=np.int32)
    maps: list[dict[int, tuple[float, int]]] = []
    for column in range(leaf_values.shape[1]):
        table: dict[int, tuple[float, int]] = {}
        for leaf_id in np.unique(leaf_values[:, column]):
            idx = leaf_values[:, column] == leaf_id
            count = int(idx.sum())
            # Beta-like shrinkage toward the head's training mean.  The map is
            # fitted only on resolved head-fit outcomes.
            mean = float(correctness[idx].mean()) if count else float(correctness.mean())
            shrunk = (count * mean + 20.0 * float(correctness.mean())) / (count + 20.0)
            table[int(leaf_id)] = (shrunk, count)
        maps.append(table)
    return maps, score, score_pct, correct_bit


def _leaf_correctness(model: lgb.LGBMRanker, frame: pd.DataFrame, fields: list[str], medians: pd.Series, maps: list[dict[int, tuple[float, int]]]) -> np.ndarray:
    leaf_values = np.asarray(_predict(model, frame, fields, medians, leaf=True), dtype=np.int32)
    out = np.zeros(len(frame), dtype=np.float32)
    for col, table in enumerate(maps):
        out += np.asarray([table.get(int(value), (0.5, 0))[0] for value in leaf_values[:, col]], dtype=np.float32)
    return out / max(1, len(maps))


def _make_head_fit(name: str, train: pd.DataFrame, fields: list[str], target_name: str, weight_name: str, seed: int) -> HeadFit:
    labels = _target(train, target_name)
    order = _feature_order(train, fields, labels)
    cap = int(name.split("_")[0].replace("cap", ""))
    selected = order[:cap]
    model, medians = _fit_ranker(train, selected, labels, weight_name, seed)
    maps, score, score_pct, correct_bit = _fit_leaf_maps(model, train, selected, medians, labels)
    outcome = pd.to_numeric(train["net_bps"], errors="coerce").to_numpy(float) - pd.to_numeric(train["base_expected_bps"], errors="coerce").to_numpy(float)
    return HeadFit(name, model, selected, medians, maps, score, np.sort(score[np.isfinite(score)]), np.sort(outcome[np.isfinite(outcome)]), target_name, weight_name)


def _head_features(head: HeadFit, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    score = np.asarray(_predict(head.model, frame, head.fields, head.medians), dtype=float)
    score_pct = _finite_rank(score, head.train_score_quantiles)
    correctness = _leaf_correctness(head.model, frame, head.fields, head.medians, head.leaf_maps)
    return score_pct.astype(np.float32), correctness.astype(np.float32)


def _pair_indices(c: np.ndarray) -> list[tuple[int, int]]:
    active = c >= np.nanmedian(c, axis=0, keepdims=True)
    p = active.mean(axis=0)
    pairs: list[tuple[float, int, int]] = []
    for i in range(c.shape[1]):
        for j in range(i + 1, c.shape[1]):
            joint = float((active[:, i] & active[:, j]).mean())
            lift = joint / max(float(p[i] * p[j]), 1e-6)
            pairs.append((lift * joint, i, j))
    pairs.sort(reverse=True)
    return [(i, j) for _, i, j in pairs[: min(12, len(pairs))]]


def _joint_matrix(correctness: np.ndarray, pairs: list[tuple[int, int]]) -> np.ndarray:
    c = np.asarray(correctness, dtype=float)
    active = (c >= np.nanmedian(c, axis=0, keepdims=True)).astype(float)
    summaries = np.column_stack([
        c.mean(axis=1), c.std(axis=1), c.min(axis=1), c.max(axis=1),
        active.sum(axis=1) / max(1, c.shape[1]),
    ])
    pair_values = np.column_stack([c[:, i] * c[:, j] for i, j in pairs]) if pairs else np.empty((len(c), 0))
    return np.column_stack([c, active, summaries, pair_values]).astype(np.float32)


def _choose_states(matrix: np.ndarray, net: np.ndarray, seed: int) -> tuple[KMeans, pd.DataFrame, list[tuple[int, int]], int]:
    pairs = _pair_indices(matrix)
    joint = _joint_matrix(matrix, pairs)
    scale = StandardScaler().fit(joint)
    z = scale.transform(joint)
    candidates = []
    for k in STATE_KS:
        if len(z) < k * 20:
            continue
        model = KMeans(n_clusters=k, n_init=10, random_state=seed)
        labels = model.fit_predict(z)
        counts = np.bincount(labels, minlength=k)
        if counts.min() < max(20, int(0.01 * len(labels))):
            continue
        sil = float(silhouette_score(z, labels, sample_size=min(10000, len(z)), random_state=seed))
        means = np.array([np.mean(net[labels == state]) for state in range(k)])
        separation = float(np.std(means) / max(np.std(net), 1.0))
        score = sil + 0.20 * separation
        candidates.append((score, k, model, labels, means, counts, sil, separation))
    if not candidates:
        k = 3
        model = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(z)
        labels = model.labels_
        means = np.array([np.mean(net[labels == state]) for state in range(k)])
        counts = np.bincount(labels, minlength=k)
        sil, separation = np.nan, np.nan
    else:
        _, k, model, labels, means, counts, sil, separation = max(candidates, key=lambda row: (row[0], -row[1]))
    # Order state IDs from adverse to favorable to make reports comparable.
    order = np.argsort(means)
    remap = {int(old): int(new) for new, old in enumerate(order)}
    labels = np.array([remap[int(value)] for value in labels], dtype=np.int32)
    model.labels_ = labels
    state_rows = []
    for state, old in enumerate(order):
        subset = net[labels == state]
        state_rows.append({
            "state": state, "rows": int(len(subset)), "mean_net_bps": float(np.mean(subset)) if len(subset) else np.nan,
            "median_net_bps": float(np.median(subset)) if len(subset) else np.nan,
            "mean_correctness": float(np.mean(matrix[labels == state])) if len(subset) else np.nan,
            "cluster_silhouette": sil, "state_separation": separation, "selected_k": int(k),
        })
    state_scale = {"mean": scale.mean_.tolist(), "scale": scale.scale_.tolist()}
    model._joint_scale = state_scale  # type: ignore[attr-defined]
    model._joint_pairs = pairs  # type: ignore[attr-defined]
    model._state_remap = remap  # type: ignore[attr-defined]
    return model, pd.DataFrame(state_rows), pairs, int(k)


def _state_predict(model: KMeans, matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pairs = getattr(model, "_joint_pairs")
    state_scale = getattr(model, "_joint_scale")
    joint = _joint_matrix(matrix, pairs)
    dtype = np.dtype(getattr(model, "cluster_centers_").dtype)
    z = (joint.astype(dtype) - np.asarray(state_scale["mean"], dtype=dtype)) / np.maximum(np.asarray(state_scale["scale"], dtype=dtype), dtype.type(1e-8))
    raw = model.predict(z)
    remap = getattr(model, "_state_remap")
    labels = np.array([remap.get(int(value), int(value)) for value in raw], dtype=np.int32)
    distance = model.transform(z)
    # Soft posterior from inverse distance; no future labels are required.
    inv = 1.0 / np.maximum(distance, 1e-6)
    posterior = inv / inv.sum(axis=1, keepdims=True)
    reordered = np.zeros_like(posterior)
    for old, new in remap.items():
        reordered[:, new] = posterior[:, old]
    return labels, reordered


def _recent_features(all_rows: pd.DataFrame, head_correct: np.ndarray, windows=(7, 14, 28)) -> pd.DataFrame:
    """Build prior-only rolling head correctness from resolved head-fit rows."""
    times = pd.to_datetime(all_rows["__ts__"], utc=True).astype("int64").to_numpy()
    resolved = times + int(pd.Timedelta(hours=12).value)
    order = np.argsort(resolved, kind="stable")
    event_times = resolved[order]
    values = np.asarray(head_correct, dtype=float)[order]
    cumulative = np.vstack([np.zeros((1, values.shape[1])), np.cumsum(values, axis=0)])
    output = {}
    query_times = times
    for days in windows:
        width = int(pd.Timedelta(days=days).value)
        end = np.searchsorted(event_times, query_times, side="left")
        start = np.searchsorted(event_times, query_times - width, side="left")
        count = np.maximum(end - start, 0)
        sums = cumulative[end] - cumulative[start]
        output.update({f"head_{i:02d}_recent_correct_{days}d": (sums[:, i] / np.maximum(count, 1)).astype(np.float32) for i in range(values.shape[1])})
    return pd.DataFrame(output, index=all_rows.index)


def _context_matrix(frame: pd.DataFrame, score_pct: np.ndarray, correctness: np.ndarray, recent: pd.DataFrame, context_fields: list[str], medians: pd.Series) -> tuple[np.ndarray, list[str], pd.Series]:
    values = frame.loc[:, context_fields].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    med = values.median() if medians is None else medians.reindex(context_fields)
    med = med.fillna(0.0)
    values = values.fillna(med).clip(-1e6, 1e6)
    recent = recent.reset_index(drop=True)
    diagnostics = pd.DataFrame({
        **{f"head_{i:02d}_score_pct": score_pct[:, i] for i in range(score_pct.shape[1])},
        **{f"head_{i:02d}_leaf_correctness": correctness[:, i] for i in range(correctness.shape[1])},
        "consensus_median": np.median(score_pct, axis=1),
        "consensus_mean": np.mean(score_pct, axis=1),
        "consensus_std": np.std(score_pct, axis=1),
        "consensus_min": np.min(score_pct, axis=1),
        "consensus_max": np.max(score_pct, axis=1),
        "head_agreement_topquartile": (score_pct >= 0.75).mean(axis=1),
        "head_agreement_bottomquartile": (score_pct <= 0.25).mean(axis=1),
        "leaf_correctness_mean": np.mean(correctness, axis=1),
        "leaf_correctness_std": np.std(correctness, axis=1),
    })
    diagnostics = pd.concat([diagnostics, recent], axis=1)
    x = pd.concat([values.reset_index(drop=True), diagnostics.reset_index(drop=True)], axis=1)
    names = list(x.columns)
    x = x.replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy("float32")
    return x, names, med


def _select_context(train: pd.DataFrame, fields: list[str], state: np.ndarray, cap: int = 64) -> list[str]:
    if not fields:
        return []
    sample = np.linspace(0, len(train) - 1, min(len(train), 20000), dtype=int)
    x = train.iloc[sample].loc[:, fields].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy("float32")
    y = state[sample]
    if np.unique(y).size < 2:
        return fields[:cap]
    try:
        mi = mutual_info_classif(x, y, discrete_features=False, random_state=20260806)
        order = np.argsort(-np.nan_to_num(mi, nan=-np.inf))
        return [fields[int(i)] for i in order[:min(cap, len(fields))]]
    except Exception:
        return fields[:cap]


def _fit_mlp(train_x: np.ndarray, train_state: np.ndarray, seed: int) -> tuple[MLPClassifier, StandardScaler]:
    scaler = StandardScaler().fit(train_x)
    x = scaler.transform(train_x)
    model = MLPClassifier(
        hidden_layer_sizes=(48, 24), activation="relu", solver="adam", alpha=2e-3,
        batch_size=512, learning_rate_init=1e-3, max_iter=220, early_stopping=False,
        random_state=seed, shuffle=False, tol=1e-4, n_iter_no_change=30,
    )
    model.fit(x, train_state)
    return model, scaler


def _aligned_proba(model: MLPClassifier, values: np.ndarray, n_classes: int) -> np.ndarray:
    """Expand MLP probabilities to the frozen latent-state class contract."""
    raw = np.asarray(model.predict_proba(values), dtype=float)
    out = np.full((len(values), n_classes), 1e-6, dtype=float)
    for column, cls in enumerate(model.classes_.astype(int)):
        if 0 <= cls < n_classes:
            out[:, cls] = raw[:, column]
    out /= out.sum(axis=1, keepdims=True)
    return out


def _tail_rows(frame: pd.DataFrame, score: str) -> list[dict]:
    rows = []
    ordered = frame.sort_values([score, "candidate_id"], ascending=[False, True], kind="stable")
    for fraction in TAILS:
        n = max(1, int(math.ceil(len(ordered) * fraction)))
        selected = ordered.head(n)
        rows.append({
            "score": score, "period": "pooled", "tail": fraction, "trades": int(n),
            "gross_bps_per_trade": float(selected["gross_bps"].mean()),
            "net_bps_per_trade": float(selected["net_bps"].mean()),
            "win_rate_net": float((selected["net_bps"] > 0).mean()),
        })
    for month, block in frame.assign(month=pd.to_datetime(frame["__ts__"], utc=True).dt.strftime("%Y-%m")).groupby("month", observed=True):
        ordered_month = block.sort_values([score, "candidate_id"], ascending=[False, True], kind="stable")
        for fraction in TAILS:
            n = max(1, int(math.ceil(len(ordered_month) * fraction)))
            selected = ordered_month.head(n)
            rows.append({
                "score": score, "period": str(month), "tail": fraction, "trades": int(n),
                "gross_bps_per_trade": float(selected["gross_bps"].mean()),
                "net_bps_per_trade": float(selected["net_bps"].mean()),
                "win_rate_net": float((selected["net_bps"] > 0).mean()),
            })
    return rows


def _load(path: Path) -> pd.DataFrame:
    schema = list(map(str, pq.ParquetFile(path).schema.names))
    required = ["candidate_id", "__ts__", "side_name", "gross_bps", "net_bps", "label_available_ts", "query_id", "meta_partition", "fold", "base_expected_bps", "atr_bps"]
    missing = sorted(set(required).difference(schema))
    if missing:
        raise ValueError(f"sidecar missing columns: {missing}")
    try:
        start = schema.index("atr_bps") + 1
        end = schema.index("label_available_ts")
        feature_block = schema[start:end]
    except ValueError as exc:
        raise ValueError("sidecar does not expose an ordered causal feature block") from exc
    columns = [name for name in schema if name in set(required).union(feature_block)]
    frame = pd.read_parquet(path, columns=columns)
    for field in feature_block:
        frame[field] = pd.to_numeric(frame[field], errors="coerce").astype("float32")
    frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True, errors="coerce")
    frame["label_available_ts"] = pd.to_datetime(frame["label_available_ts"], utc=True, errors="coerce")
    if frame[["__ts__", "label_available_ts"]].isna().any().any():
        raise ValueError("sidecar contains invalid timestamps")
    if not frame["label_available_ts"].ge(frame["__ts__"]).all():
        raise ValueError("label availability precedes decision")
    if not frame["side_name"].astype(str).str.lower().eq(SIDE).all():
        raise ValueError("this runner is explicitly long-only")
    if frame.duplicated(["fold", "candidate_id"]).any():
        raise ValueError("duplicate candidate inside fold")
    return frame.sort_values(["fold", "__ts__", "candidate_id"], kind="stable").reset_index(drop=True)


def _fold_run(
    block: pd.DataFrame,
    fold: str,
    fields: list[str],
    seed: int,
    out: Path,
    target_arms: tuple[str, ...] = TARGET_ARMS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = block[block.meta_partition.eq("meta_train")].sort_values(["__ts__", "candidate_id"], kind="stable").copy()
    calibration = block[block.meta_partition.eq("meta_calibration")].sort_values(["__ts__", "candidate_id"], kind="stable").copy()
    test = block[block.meta_partition.eq("test")].sort_values(["__ts__", "candidate_id"], kind="stable").copy()
    if min(len(train), len(calibration), len(test)) < 100:
        raise ValueError(f"{fold}: insufficient partition rows")
    if train["label_available_ts"].max() >= calibration["__ts__"].min():
        # The sidecar's train partition may include a boundary row whose H12
        # path resolves inside calibration.  Purge it rather than weakening the
        # contract.
        train = train[train["label_available_ts"] < calibration["__ts__"].min()].copy()
    if calibration["label_available_ts"].max() >= test["__ts__"].min():
        raise ValueError(f"{fold}: calibration labels overlap test start")
    head_cut = train["__ts__"].quantile(0.60)
    head_fit_rows = train[train["__ts__"] <= head_cut].copy()
    state_train = train[train["__ts__"] > head_cut].copy()
    if min(len(head_fit_rows), len(state_train)) < 200:
        raise ValueError(f"{fold}: chronological head/state split too small")

    all_target_predictions: list[pd.DataFrame] = []
    all_state_rows: list[dict] = []
    all_head_rows: list[dict] = []
    all_comparison: list[dict] = []
    all_state_metrics: list[dict] = []

    for target_index, target_name in enumerate(target_arms):
        # Fit/extract one head at a time.  Keeping ten full LightGBM model
        # objects resident alongside the wide sidecar is unnecessarily costly;
        # the latent-state layer needs only their ranks, leaf posteriors, and
        # the cap-120 raw control score.
        score_train_parts: list[np.ndarray] = []
        correct_train_parts: list[np.ndarray] = []
        score_cal_parts: list[np.ndarray] = []
        correct_cal_parts: list[np.ndarray] = []
        score_test_parts: list[np.ndarray] = []
        correct_test_parts: list[np.ndarray] = []
        head_fit_correct_parts: list[np.ndarray] = []
        cap120_cal_raw: np.ndarray | None = None
        cap120_test_raw: np.ndarray | None = None
        head_audit_for_target: list[dict] = []
        for cap_index, cap in enumerate(CAPS):
            for weight_index, weight_name in enumerate(WEIGHTS):
                head_name = f"cap{cap}_{weight_name}"
                head = _make_head_fit(head_name, head_fit_rows, fields, target_name, weight_name, seed + target_index * 1000 + cap_index * 10 + weight_index)
                fit_correct = _leaf_correctness(head.model, head_fit_rows, head.fields, head.medians, head.leaf_maps)
                tr_score, tr_correct = _head_features(head, state_train)
                ca_score, ca_correct = _head_features(head, calibration)
                te_score, te_correct = _head_features(head, test)
                head_fit_correct_parts.append(fit_correct)
                score_train_parts.append(tr_score)
                correct_train_parts.append(tr_correct)
                score_cal_parts.append(ca_score)
                correct_cal_parts.append(ca_correct)
                score_test_parts.append(te_score)
                correct_test_parts.append(te_correct)
                if head_name == "cap120_equal_month":
                    cap120_cal_raw = np.asarray(_predict(head.model, calibration, head.fields, head.medians), dtype=float)
                    cap120_test_raw = np.asarray(_predict(head.model, test, head.fields, head.medians), dtype=float)
                head_audit_for_target.append({"fold": str(fold), "target_arm": target_name, "head_name": head.name, "feature_count": len(head.fields), "target": head.target_name, "weight": head.weight_name, "feature_digest": _digest(head.fields)})
                del head
        score_train = np.column_stack(score_train_parts)
        correct_train = np.column_stack(correct_train_parts)
        score_cal = np.column_stack(score_cal_parts)
        correct_cal = np.column_stack(correct_cal_parts)
        score_test = np.column_stack(score_test_parts)
        correct_test = np.column_stack(correct_test_parts)
        if cap120_cal_raw is None or cap120_test_raw is None:
            raise RuntimeError("cap120_equal_month head was not materialised")
        all_head_rows.extend(head_audit_for_target)

        state_model, state_definitions, pairs, selected_k = _choose_states(
            correct_train, state_train.net_bps.to_numpy(float), seed + target_index * 100,
        )
        state_train_labels, state_train_post = _state_predict(state_model, correct_train)
        cal_labels, cal_post = _state_predict(state_model, correct_cal)
        test_labels, test_post = _state_predict(state_model, correct_test)
        state_defs = state_definitions.copy()
        state_defs["fold"] = str(fold)
        state_defs["target_arm"] = target_name
        state_defs.to_parquet(out / f"state_definitions_{fold}_{target_name}.parquet", index=False)
        all_state_rows.extend(state_defs.to_dict("records"))

        # Causal, prior-only recent correctness.  For each outer row, the
        # event table contains only head-fit observations, so state/cal/test
        # rows never contribute their own or future outcomes.
        recent_source = pd.concat([head_fit_rows, state_train, calibration, test], ignore_index=True)
        source_correct = np.vstack([
            np.column_stack(head_fit_correct_parts),
            correct_train, correct_cal, correct_test,
        ])
        recent = _recent_features(recent_source, source_correct)
        recent_train = recent.iloc[len(head_fit_rows):len(head_fit_rows) + len(state_train)].reset_index(drop=True)
        recent_cal = recent.iloc[len(head_fit_rows) + len(state_train):len(head_fit_rows) + len(state_train) + len(calibration)].reset_index(drop=True)
        recent_test = recent.iloc[-len(test):].reset_index(drop=True)

        context_fields = [field for field in fields if field in state_train.columns]
        selected_context = _select_context(state_train, context_fields, state_train_labels, cap=64)
        context_medians = state_train[selected_context].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).median().fillna(0.0)
        x_train, context_names, _ = _context_matrix(state_train, score_train, correct_train, recent_train, selected_context, context_medians)
        x_cal, _, _ = _context_matrix(calibration, score_cal, correct_cal, recent_cal, selected_context, context_medians)
        x_test, _, _ = _context_matrix(test, score_test, correct_test, recent_test, selected_context, context_medians)
        mlp, scaler = _fit_mlp(x_train, state_train_labels, seed + 5000 + target_index)
        post_train = _aligned_proba(mlp, scaler.transform(x_train), selected_k)
        post_cal = _aligned_proba(mlp, scaler.transform(x_cal), selected_k)
        post_test = _aligned_proba(mlp, scaler.transform(x_test), selected_k)
        state_values = state_train.groupby(state_train_labels).net_bps.mean().reindex(range(selected_k)).fillna(state_train.net_bps.mean()).to_numpy(float)
        mlp_train_raw = post_train @ state_values
        mlp_cal_raw = post_cal @ state_values
        mlp_test_raw = post_test @ state_values
        iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
        iso.fit(mlp_cal_raw, calibration.net_bps.to_numpy(float))
        mlp_cal_ev = iso.predict(mlp_cal_raw)
        mlp_test_ev = iso.predict(mlp_test_raw)
        for split_name, true_state, posterior, net_values, ev_values in (
            ("state_train", state_train_labels, post_train, state_train.net_bps.to_numpy(float), mlp_train_raw),
            ("calibration", cal_labels, post_cal, calibration.net_bps.to_numpy(float), mlp_cal_raw),
            ("test", test_labels, post_test, test.net_bps.to_numpy(float), mlp_test_raw),
        ):
            pred_state = np.argmax(posterior, axis=1)
            hard_accuracy = float(np.mean(pred_state == np.asarray(true_state, dtype=int)))
            state_logloss = float(log_loss(np.asarray(true_state, dtype=int), posterior, labels=list(range(selected_k))))
            state_rank_ic = spearmanr(ev_values, net_values).statistic
            all_state_metrics.append({
                "fold": str(fold), "target_arm": target_name, "split": split_name,
                "rows": int(len(net_values)), "state_k": int(selected_k),
                "accuracy": hard_accuracy, "logloss": state_logloss,
                "state_ev_rank_ic": float(state_rank_ic) if np.isfinite(state_rank_ic) else np.nan,
                "state_mean_net_bps": float(np.mean(net_values)),
            })

        # Matched cap-120 policy correction: same head-fit rows, same target,
        # same causal calibration and same outer test rows.
        cap_iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
        cap_iso.fit(cap120_cal_raw, calibration.net_bps.to_numpy(float))
        cap120_cal_ev = cap_iso.predict(cap120_cal_raw)
        cap120_test_ev = cap_iso.predict(cap120_test_raw)

        consensus_cal = np.median(score_cal, axis=1)
        consensus_test = np.median(score_test, axis=1)
        consensus_iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
        consensus_iso.fit(consensus_cal, calibration.net_bps.to_numpy(float))
        consensus_cal_ev = consensus_iso.predict(consensus_cal)
        consensus_test_ev = consensus_iso.predict(consensus_test)
        mlp_cal_pct = _finite_rank(mlp_cal_ev)
        mlp_test_pct = _finite_rank(mlp_test_ev, mlp_cal_ev)
        # MLP-as-weight: preserve consensus as the primary score and use the
        # latent-state posterior to down-weight consensus when the state model
        # is uncertain or economically adverse.
        confidence_cal = np.max(post_cal, axis=1)
        confidence_test = np.max(post_test, axis=1)
        state_positive_cal = _finite_rank(mlp_cal_ev)
        state_positive_test = _finite_rank(mlp_test_ev, mlp_cal_ev)
        weighted_cal = consensus_cal * (0.5 + 0.5 * state_positive_cal) * (0.75 + 0.25 * confidence_cal)
        weighted_test = consensus_test * (0.5 + 0.5 * state_positive_test) * (0.75 + 0.25 * confidence_test)
        blend25_cal = 0.75 * consensus_cal + 0.25 * state_positive_cal
        blend25_test = 0.75 * consensus_test + 0.25 * state_positive_test
        blend50_cal = 0.50 * consensus_cal + 0.50 * state_positive_cal
        blend50_test = 0.50 * consensus_test + 0.50 * state_positive_test

        output = test.loc[:, ["candidate_id", "__ts__", "side_name", "gross_bps", "net_bps"]].copy()
        output["fold"] = str(fold)
        output["target_arm"] = target_name
        output["mlp_state_score"] = mlp_test_ev
        output["cap120_policy_correction"] = cap120_test_ev
        output["consensus_ev"] = consensus_test_ev
        output["mlp_weighted_consensus"] = weighted_test
        output["mlp_blend25_consensus"] = blend25_test
        output["mlp_blend50_consensus"] = blend50_test
        output["mlp_state_entropy"] = -np.sum(post_test * np.log(np.maximum(post_test, 1e-9)), axis=1)
        output["mlp_state_confidence"] = confidence_test
        output["latent_state"] = test_labels
        for index, head_name in enumerate([f"cap{cap}_{weight}" for cap in CAPS for weight in WEIGHTS]):
            output[f"{head_name}__score_pct"] = score_test[:, index]
            output[f"{head_name}__leaf_correctness"] = correct_test[:, index]

        all_target_predictions.append(output)
        # Score metrics are created after all folds so tails are pooled globally.
        for score_name in ("mlp_state_score", "cap120_policy_correction", "consensus_ev", "mlp_weighted_consensus", "mlp_blend25_consensus", "mlp_blend50_consensus"):
            all_comparison.extend(_tail_rows(output, score_name))
        state_defs = state_defs.assign(fold=str(fold), target_arm=target_name)

        # Persist fold-level causal audit including all state/MLP inputs.
        audit = {
            "fold": str(fold), "target_arm": target_name, "side": SIDE,
            "head_fit_rows": len(head_fit_rows), "state_train_rows": len(state_train),
            "calibration_rows": len(calibration), "test_rows": len(test),
            "head_fit_end": head_fit_rows.__ts__.max().isoformat(),
            "state_train_start": state_train.__ts__.min().isoformat(),
            "calibration_start": calibration.__ts__.min().isoformat(),
            "test_start": test.__ts__.min().isoformat(),
            "selected_state_k": int(selected_k), "joint_pair_count": len(pairs),
            "selected_context_fields": selected_context,
            "context_feature_count": len(context_names),
            "mlp_iterations": int(mlp.n_iter_), "mlp_final_loss": float(mlp.loss_),
            "target_distribution_state_train": {str(k): int(v) for k, v in zip(*np.unique(state_train_labels, return_counts=True))},
            "calibration_isotonic": True,
            "forbidden_outcome_inputs_at_inference": ["net_bps", "gross_bps", "label_available_ts", "latent_state"],
        }
        _write_json(out / f"fold_audit_{fold}_{target_name}.json", audit)

    predictions = pd.concat(all_target_predictions, ignore_index=True)
    # Recompute pooled tails from all outer test rows instead of combining fold
    # tail means.  This is the required global-top-k evaluation basis.
    metric_rows: list[dict] = []
    for target_name, group in predictions.groupby("target_arm", observed=True):
        for score_name in ("mlp_state_score", "cap120_policy_correction", "consensus_ev", "mlp_weighted_consensus", "mlp_blend25_consensus", "mlp_blend50_consensus"):
            block = group.copy()
            for row in _tail_rows(block, score_name):
                metric_rows.append({**row, "target_arm": target_name})
    predictions.to_parquet(out / "mlp_and_consensus_oos_predictions.parquet", index=False, compression="zstd")
    pd.DataFrame(metric_rows).to_parquet(out / "comparison_metrics.parquet", index=False, compression="zstd")
    pd.DataFrame(all_head_rows).drop_duplicates().to_parquet(out / "head_contract_audit.parquet", index=False, compression="zstd")
    pd.DataFrame(all_state_rows).to_parquet(out / "joint_state_definitions.parquet", index=False, compression="zstd")
    state_metrics = pd.DataFrame(all_state_metrics)
    state_metrics.to_parquet(out / "mlp_state_metrics.parquet", index=False, compression="zstd")
    pd.DataFrame(all_comparison).to_parquet(out / "fold_diagnostic_metrics.parquet", index=False, compression="zstd")
    return predictions, pd.DataFrame(metric_rows), pd.DataFrame(all_head_rows), pd.DataFrame(all_state_rows), state_metrics


def run(
    sidecar: Path = DEFAULT_SIDECAR,
    out: Path = DEFAULT_OUT,
    folds: tuple[str, ...] | None = None,
    target_arms: tuple[str, ...] = TARGET_ARMS,
) -> Path:
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to overwrite populated output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    _write_json(out / "run_manifest.json", {"schema": SCHEMA, "status": "RUNNING", "sidecar": str(sidecar), "side": SIDE})
    frame = _load(sidecar)
    fields = _feature_columns(frame)
    fold_ids = sorted(frame["fold"].astype(str).unique())
    if len(fold_ids) < 2:
        raise ValueError("expected at least two chronological outer folds")
    all_predictions = []
    all_metrics = []
    all_heads = []
    all_states = []
    all_state_metrics = []
    fold_reports = []
    selected_folds = fold_ids if folds is None else [fold for fold in fold_ids if fold in set(folds)]
    if not selected_folds:
        raise ValueError(f"none of requested folds are present: {folds}")
    for fold_index, fold in enumerate(selected_folds):
        block = frame[frame["fold"].astype(str).eq(fold)].copy()
        predictions, metrics, heads, states, state_metrics = _fold_run(block, fold, fields, 20260806 + fold_index * 10000, out, target_arms)
        all_predictions.append(predictions); all_metrics.append(metrics); all_heads.append(heads); all_states.append(states); all_state_metrics.append(state_metrics)
        fold_reports.append({"fold": fold, "rows": len(block), "test_rows": int((block.meta_partition == "test").sum())})
    predictions = pd.concat(all_predictions, ignore_index=True)
    # Global pooled metrics are recomputed across all folds and are the
    # authoritative comparison table.
    metric_rows: list[dict] = []
    for target_name, group in predictions.groupby("target_arm", observed=True):
        for score_name in ("mlp_state_score", "cap120_policy_correction", "consensus_ev", "mlp_weighted_consensus", "mlp_blend25_consensus", "mlp_blend50_consensus"):
            for row in _tail_rows(group, score_name):
                metric_rows.append({**row, "target_arm": target_name})
    metrics = pd.DataFrame(metric_rows)
    predictions.to_parquet(out / "mlp_and_consensus_oos_predictions.parquet", index=False, compression="zstd")
    metrics.to_parquet(out / "comparison_metrics.parquet", index=False, compression="zstd")
    pd.concat(all_heads, ignore_index=True).drop_duplicates().to_parquet(out / "head_contract_audit.parquet", index=False, compression="zstd")
    pd.concat(all_states, ignore_index=True).drop_duplicates().to_parquet(out / "joint_state_definitions.parquet", index=False, compression="zstd")
    pd.concat(all_state_metrics, ignore_index=True).to_parquet(out / "mlp_state_metrics.parquet", index=False, compression="zstd")
    manifest = {
        "schema": SCHEMA, "status": "complete", "side": SIDE, "sidecar": str(sidecar),
        "outer_folds": fold_reports, "causal_feature_count": len(fields),
        "causal_feature_digest": _digest(fields), "feature_fields": fields,
        "heads": [f"cap{cap}_{weight}" for cap in CAPS for weight in WEIGHTS],
        "target_arms": list(target_arms), "state_k_candidates": list(STATE_KS),
        "head_target_contract": "target-specific LambdaRank relevance grades; leaf correctness maps fit on head_fit outcomes only",
        "joint_state_contract": "enhanced co-activation vector: ten soft leaf-correctness values, binary activation, summaries, top coactivation-lift pairs",
        "mlp_contract": "causal context + head score ranks + leaf correctness + consensus dispersion/agreement + prior-only 7/14/28d correctness history",
        "mlp_meta_arms": ["mlp_state_score", "mlp_weighted_consensus", "mlp_blend25_consensus", "mlp_blend50_consensus"],
        "control": "matched cap120_equal_month LambdaRank policy-correction score with calibration-only isotonic EV map",
        "execution": "source sidecar fixed-policy net labels; costs already represented once in net_bps",
        "ranking": "pooled global top-k over outer test rows; not per-timestamp",
        "leakage": "head_fit -> state_train -> meta_calibration -> test; test labels never used in state/MLP/iso selection",
        "limitations": ["long-only source sidecar", "structural head scores are rebuilt because ten-head persisted scores were unavailable", "AE/GMM fields follow the frozen representation-selection exception"],
    }
    _write_json(out / "run_manifest.json", manifest)
    report = out / "JOINT_CORRECTNESS_MLP_REPORT.md"
    report.write_text(
        "# Joint-correctness MLP meta experiment\n\n"
        f"Side: `{SIDE}`. Heads: {len(CAPS) * len(WEIGHTS)}. Target arms: {', '.join(TARGET_ARMS)}.\n\n"
        "The authoritative results are in `comparison_metrics.parquet`; pooled rows are recomputed across all outer test folds. "
        "The MLP replacement and consensus-weighting arms are compared with the matched cap-120 equal-month policy-correction control.\n\n"
        "This is a long-only research result, not a production promotion.\n"
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sidecar", type=Path, default=DEFAULT_SIDECAR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--fold", dest="folds", action="append", default=None)
    parser.add_argument("--target", dest="targets", action="append", choices=list(TARGET_ARMS), default=None)
    args = parser.parse_args()
    print(run(args.sidecar, args.out, tuple(args.folds) if args.folds else None, tuple(args.targets) if args.targets else TARGET_ARMS))


if __name__ == "__main__":
    main()
