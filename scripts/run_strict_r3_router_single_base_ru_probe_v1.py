#!/usr/bin/env python3
"""Fixed R/U downstream probe for one Router50 single-base candidate.

This is the selection bridge between a one-head routed Base and the eventual
dual-MC1 architecture:

    frozen Router50 identities -> strict-OOF Base ledger -> fixed R + U heads
      -> causal single-family mini-MC1 probe

The mini mapper is deliberately a *selection diagnostic*, not a replacement
for the production dual Current/BCF MC1 maps.  It answers whether a Base
candidate leaves useful, causal residual information for R and U after its own
ranking is fixed.  Finalists alone advance to the unchanged dual-MC1 replay.

No Router score is read as a numeric model input.  All held Base/R/U receipts
are persisted target-free before canonical policy or path outcomes are joined
for diagnostics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from lightgbm import LGBMRanker, LGBMRegressor, early_stopping
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mutual_info_score


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "strict_r3_router_single_base_ru_probe_v1"
SEED = 1729
IDENTITY = ("candidate_id", "__decision_ts__", "side_name")
F72_SELECTION = ROOT / "data_perp/artifacts/strict_r3_b0_family_addback_20260826_v1_policy_ordinal_base_g3/selection.json"
EPOCH = pd.Timestamp("2025-01-01", tz="UTC")
TRAIN_MONTHS = 3
RESERVE_DAYS = 28
ANCHOR_BLOCK_DAYS = 14
TRAIN_CAP = 60_000
MIN_MINI_MONTHS = 3
R_GAIN = [0.0, 1.0, 2.0, 4.0, 7.0]
U_GAIN = [0.0, 1.0]


@dataclass(frozen=True)
class HeadSpec:
    name: str
    kind: str
    feature_count: int


R_SPEC = HeadSpec("r_residual_sqrt_atr_quintile", "residual", 72)
U_SPEC = HeadSpec("u_unexpected_trailing_atr1", "upside", 72)


def _utc(value: object) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _month_end(month: pd.Timestamp) -> pd.Timestamp:
    return month + pd.offsets.MonthBegin(1)


def _months_between(start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.Timestamp, ...]:
    return tuple(pd.date_range(
        start.normalize().replace(day=1),
        (end - pd.Timedelta(nanoseconds=1)).normalize().replace(day=1),
        freq="MS", tz="UTC",
    ))


def _exclusive_json(path: Path, payload: object) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def _progress(out: Path, **payload: object) -> None:
    with (out / "progress.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")


def _sha256(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(set(paths)):
        digest.update(str(path).encode())
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    return digest.hexdigest()


def _load_f72(path: Path) -> tuple[str, ...]:
    payload = json.loads(path.read_text())
    fields = payload.get("selected_features")
    if not isinstance(fields, list) or len(fields) != 72 or len(set(fields)) != 72:
        raise AssertionError(f"{path}: expected frozen 72-field Base contract")
    return tuple(str(item) for item in fields)


def _resolve_base_score(roots: Sequence[Path], target: str, month: pd.Timestamp) -> Path:
    paths = [root / "target_free_scores" / target / f"month={month:%Y-%m}.parquet" for root in roots]
    existing = [path for path in paths if path.exists()]
    if len(existing) != 1:
        raise AssertionError(f"{month:%Y-%m}: expected one Base score owner across {paths}, found {len(existing)}")
    return existing[0]


def _resolve_features(roots: Sequence[Path], month: pd.Timestamp) -> Path:
    paths = [root / f"month={month:%Y-%m}" / "causal_feature_universe.parquet" for root in roots]
    existing = [path for path in paths if path.exists()]
    if len(existing) != 1:
        raise AssertionError(f"{month:%Y-%m}: expected one feature owner across {paths}, found {len(existing)}")
    return existing[0]


def _support_path(root: Path, month: pd.Timestamp) -> Path:
    path = root / "parts" / f"month={month:%Y-%m}" / "side=long.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _rank_desc(frame: pd.DataFrame, field: str) -> np.ndarray:
    work = frame.loc[:, ["candidate_id", "__decision_ts__", field]].copy()
    work["__row__"] = np.arange(len(work), dtype=np.int64)
    work = work.sort_values(["__decision_ts__", field, "candidate_id"], ascending=[True, False, True], kind="stable")
    ordinal = work.groupby("__decision_ts__", sort=False).cumcount().to_numpy(float) + 1.0
    count = work.groupby("__decision_ts__", sort=False).candidate_id.transform("size").to_numpy(float)
    result = np.empty(len(work), dtype=np.float32)
    result[work.__row__.to_numpy(np.int64)] = (1.0 - (ordinal - .5) / count).astype(np.float32)
    return result


def _base_geometry(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if not np.isfinite(pd.to_numeric(out.base_score, errors="coerce")).all():
        raise AssertionError("Base target-free score contains a non-finite value")
    # The Base rank must be a target-free monotone rank of its raw score.  Most
    # candidate bases use deterministic candidate-ID tie-breaking.  ET50 has
    # genuinely tied, mapped score values and its native contract uses the
    # average-tie percentile.  Preserve either declared source convention;
    # never rebuild a held rank with an unrelated treatment of ties.
    persisted = pd.to_numeric(out.base_rank_ts, errors="coerce").to_numpy(float)
    rebuilt_stable = _rank_desc(out, "base_score")
    rebuilt_average = out.groupby("__decision_ts__", sort=False)["base_score"].rank(
        pct=True, method="average",
    ).to_numpy(float)
    if np.allclose(rebuilt_stable, persisted, rtol=0.0, atol=1e-7):
        out["base_rank_contract"] = "stable_candidate_id"
    elif np.allclose(rebuilt_average, persisted, rtol=0.0, atol=1e-7):
        out["base_rank_contract"] = "average_tie_percentile"
    else:
        raise AssertionError("persisted Base timestamp rank is not a declared target-free raw-score rank")
    summary = out.groupby("__decision_ts__", sort=False).base_score.agg(["size", "std", "min", "max"])
    out["base_query_count"] = out.__decision_ts__.map(summary["size"]).astype(np.float32)
    out["base_query_std"] = out.__decision_ts__.map(summary["std"]).fillna(0.0).astype(np.float32)
    out["base_query_range"] = (out.__decision_ts__.map(summary["max"]) - out.__decision_ts__.map(summary["min"])).astype(np.float32)
    ordered = out.loc[:, ["candidate_id", "__decision_ts__", "base_score"]].sort_values(
        ["__decision_ts__", "base_score", "candidate_id"], ascending=[True, False, True], kind="stable",
    )
    ordered["next"] = ordered.groupby("__decision_ts__", sort=False).base_score.shift(-1)
    ordered["third"] = ordered.groupby("__decision_ts__", sort=False).base_score.shift(-2)
    top = ordered.groupby("__decision_ts__", sort=False).first()
    out["base_top_gap"] = out.__decision_ts__.map(top.base_score - top.next).fillna(0.0).astype(np.float32)
    out["base_top2_gap"] = out.__decision_ts__.map(top.next - top.third).fillna(0.0).astype(np.float32)
    return out


def _read_policy(path: Path) -> pd.DataFrame:
    columns = ["candidate_id", "policy_path_valid", "policy_net_bps", "policy_exit_reason", "policy_label_available_ts"]
    out = pd.read_parquet(path, columns=columns)
    out["policy_label_available_ts"] = pd.to_datetime(out.policy_label_available_ts, utc=True, errors="coerce")
    out["policy_path_valid"] = out.policy_path_valid.fillna(False).astype(bool)
    out["policy_net_bps"] = pd.to_numeric(out.policy_net_bps, errors="coerce")
    if out.candidate_id.duplicated().any():
        raise AssertionError("canonical policy ledger has duplicate candidate IDs")
    return out


def _read_window(*, base_roots: Sequence[Path], base_target: str, feature_roots: Sequence[Path],
                 policy: pd.DataFrame, path_root: Path, fields: Sequence[str], start: pd.Timestamp,
                 end: pd.Timestamp) -> tuple[pd.DataFrame, list[Path]]:
    pieces: list[pd.DataFrame] = []
    source_paths: list[Path] = []
    support_parts: list[pd.DataFrame] = []
    for month in _months_between(start, end):
        score_path = _resolve_base_score(base_roots, base_target, month)
        feature_path = _resolve_features(feature_roots, month)
        source_paths.extend((score_path, feature_path))
        score = pd.read_parquet(score_path, columns=[*IDENTITY, "base_score", "base_rank_ts"])
        feature = pd.read_parquet(feature_path, columns=[*IDENTITY, *fields])
        for part in (score, feature):
            part["__decision_ts__"] = pd.to_datetime(part.__decision_ts__, utc=True, errors="raise")
            if part.duplicated(IDENTITY).any():
                raise AssertionError(f"{month:%Y-%m}: duplicate target-free identity")
        merged = score.merge(feature, on=list(IDENTITY), how="left", validate="one_to_one")
        if len(merged) != len(score):
            raise AssertionError(f"{month:%Y-%m}: causal feature merge changed target-free Base identities")
        pieces.append(merged.loc[merged.__decision_ts__.ge(start) & merged.__decision_ts__.lt(end)].copy())
        # Support labels are keyed by signal-close month; include the prior
        # partition and filter by executable decision timestamp below.
        for token in (month - pd.offsets.MonthBegin(1), month):
            support_path = _support_path(path_root, token)
            source_paths.append(support_path)
            support = pd.read_parquet(support_path, columns=[
                "candidate_id", "__decision_ts__", "side_name", "supportive_path_valid",
                "supportive_label_available_ts", "path_arch_atr_fraction",
            ])
            support["__decision_ts__"] = pd.to_datetime(support.__decision_ts__, utc=True, errors="raise")
            support["supportive_label_available_ts"] = pd.to_datetime(support.supportive_label_available_ts, utc=True, errors="coerce")
            support_parts.append(support.loc[support.__decision_ts__.ge(month) & support.__decision_ts__.lt(_month_end(month))].copy())
    base = pd.concat(pieces, ignore_index=True)
    if base.empty or base.duplicated(IDENTITY).any() or not base.side_name.eq("long").all():
        raise AssertionError("invalid long-only strict-OOF Base window")
    support = pd.concat(support_parts, ignore_index=True)
    if support.duplicated(IDENTITY).any():
        raise AssertionError("supportive-path sidecar has duplicate identities after boundary collection")
    out = base.merge(support, on=list(IDENTITY), how="left", validate="one_to_one")
    out = out.merge(policy, on="candidate_id", how="left", validate="one_to_one")
    if len(out) != len(base):
        raise AssertionError("outcome join changed target-free Base identities")
    out["supportive_path_valid"] = out.supportive_path_valid.fillna(False).astype(bool)
    out["atr_bps"] = (10_000.0 * pd.to_numeric(out.path_arch_atr_fraction, errors="coerce")).astype(np.float32)
    return _base_geometry(out), source_paths


def _sample_complete_queries(frame: pd.DataFrame, cap: int, seed: int) -> pd.DataFrame:
    if len(frame) <= cap:
        return frame.reset_index(drop=True).copy()
    work = frame.copy()
    queries = work.loc[:, ["__decision_ts__"]].drop_duplicates().copy()
    queries["month"] = queries.__decision_ts__.dt.strftime("%Y-%m")
    queries["hash"] = pd.util.hash_pandas_object(queries.__decision_ts__.astype(str) + f"|{seed}", index=False).to_numpy(np.uint64)
    size = work.groupby("__decision_ts__", sort=False).size()
    queries["rows"] = queries.__decision_ts__.map(size).astype(int)
    selected: list[pd.Timestamp] = []
    quota = max(1, cap // max(1, queries.month.nunique()))
    for _, group in queries.sort_values(["month", "hash", "__decision_ts__"], kind="stable").groupby("month", sort=False):
        used = 0
        for stamp, rows in group.loc[:, ["__decision_ts__", "rows"]].itertuples(index=False, name=None):
            if used and used + int(rows) > quota:
                continue
            selected.append(stamp)
            used += int(rows)
    out = work.loc[work.__decision_ts__.isin(selected)].copy()
    if out.empty:
        raise AssertionError("query-safe training sample is empty")
    return out.reset_index(drop=True)


def _matrix(frame: pd.DataFrame, fields: Sequence[str]) -> np.ndarray:
    geometry = ["base_score", "base_rank_ts", "base_query_count", "base_query_std", "base_query_range", "base_top_gap", "base_top2_gap"]
    return frame.loc[:, [*geometry, *fields]].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)


def _impute(train: np.ndarray, held: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    medians = np.nanmedian(train, axis=0).astype(np.float32)
    medians[~np.isfinite(medians)] = 0.0
    values: list[np.ndarray] = []
    for matrix in (train.copy(), held.copy()):
        missing = ~np.isfinite(matrix)
        if missing.any():
            matrix[missing] = np.broadcast_to(medians, matrix.shape)[missing]
        values.append(matrix.astype(np.float32, copy=False))
    return values[0], values[1]


def _anchor_valid(frame: pd.DataFrame) -> np.ndarray:
    return (
        frame.policy_path_valid.fillna(False).astype(bool).to_numpy()
        & np.isfinite(pd.to_numeric(frame.policy_net_bps, errors="coerce").to_numpy(float))
        & np.isfinite(pd.to_numeric(frame.base_rank_ts, errors="coerce").to_numpy(float))
    )


def _prequential_anchor(frame: pd.DataFrame) -> np.ndarray:
    """Expanding train-only anchors, blockwise to prevent same-row fitting."""
    work = frame.sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index()
    result = np.full(len(work), np.nan, dtype=np.float32)
    start = work.__decision_ts__.min().floor("D")
    work["block"] = ((work.__decision_ts__ - start) / pd.Timedelta(days=ANCHOR_BLOCK_DAYS)).astype(int)
    valid = _anchor_valid(work)
    available = pd.to_datetime(work.policy_label_available_ts, utc=True, errors="coerce")
    for block in sorted(work.block.unique()):
        current = work.block.eq(block)
        block_start = work.loc[current, "__decision_ts__"].min()
        prior = work.__decision_ts__.lt(block_start).to_numpy() & valid & available.lt(block_start).to_numpy()
        if int(prior.sum()) < 1_000:
            continue
        mapper = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(
            work.loc[prior, "base_rank_ts"], work.loc[prior, "policy_net_bps"],
        )
        result[current.to_numpy()] = mapper.predict(work.loc[current, "base_rank_ts"]).astype(np.float32)
    restored = np.empty(len(frame), dtype=np.float32)
    restored[work["index"].to_numpy(np.int64)] = result
    return restored


def _residual_targets(frame: pd.DataFrame, *, train: bool, held_anchor: IsotonicRegression | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    valid = (
        _anchor_valid(frame)
        & frame.supportive_path_valid.fillna(False).astype(bool).to_numpy()
        & np.isfinite(pd.to_numeric(frame.atr_bps, errors="coerce").to_numpy(float))
        & pd.to_numeric(frame.atr_bps, errors="coerce").gt(0.0).to_numpy()
    )
    if train:
        anchor = _prequential_anchor(frame)
        valid &= np.isfinite(anchor)
    else:
        if held_anchor is None:
            raise AssertionError("held R/U diagnostic target requires train-only anchor")
        anchor = held_anchor.predict(frame.base_rank_ts).astype(np.float32)
    net = pd.to_numeric(frame.policy_net_bps, errors="coerce").to_numpy(float)
    residual = (net - anchor).astype(np.float32)
    atr = pd.to_numeric(frame.atr_bps, errors="coerce").to_numpy(float)
    sqrt_residual = residual / np.sqrt(np.maximum(atr, 1e-3))
    atr_residual = residual / np.maximum(atr, 1e-3)
    return valid, residual, sqrt_residual.astype(np.float32), atr_residual.astype(np.float32)


def _r_labels(values: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, list[float]]:
    labels = np.full(len(values), -1, dtype=np.int8)
    finite = valid & np.isfinite(values)
    if int(finite.sum()) < 10_000:
        raise AssertionError("insufficient R residual support")
    edges = np.quantile(values[finite], (.20, .40, .60, .80))
    labels[finite] = np.searchsorted(edges, values[finite], side="right").clip(0, 4).astype(np.int8)
    return labels, [float(value) for value in edges]


def _u_labels(frame: pd.DataFrame, valid: np.ndarray, atr_residual: np.ndarray) -> np.ndarray:
    labels = np.full(len(frame), -1, dtype=np.int8)
    exit_reason = frame.policy_exit_reason.astype(str)
    clean_trailing = exit_reason.isin(("trailing", "smooth_capital_protect")).to_numpy()
    labels[valid] = (clean_trailing[valid] & (atr_residual[valid] >= 1.0)).astype(np.int8)
    return labels


def _query(frame: pd.DataFrame, kind: str) -> np.ndarray:
    if kind == "r":
        band = np.minimum(19, np.maximum(0, np.floor(pd.to_numeric(frame.base_rank_ts, errors="coerce") / .05))).astype(int)
        block = ((frame.__decision_ts__ - EPOCH) / pd.Timedelta(days=28)).astype(int)
        return np.asarray([f"b{left:02d}|k{right:03d}" for left, right in zip(band, block)], dtype=object)
    if kind == "u":
        return frame.__decision_ts__.astype(str).to_numpy(object)
    raise ValueError(kind)


def _ordered_groups(frame: pd.DataFrame, query_id: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[int], np.ndarray, np.ndarray]:
    work = pd.DataFrame({"query": query_id, "__decision_ts__": frame.__decision_ts__.to_numpy(), "candidate_id": frame.candidate_id.astype(str).to_numpy(), "row": np.arange(len(frame), dtype=np.int64)})
    support = work.groupby("query", sort=False).size()
    work = work.loc[work["query"].isin(support.index[support.ge(2)])].copy()
    start = work.groupby("query", sort=False).__decision_ts__.min().sort_values(kind="stable")
    cutoff = max(1, int(math.floor(.80 * len(start))))
    fit_queries = set(start.index[:cutoff]); valid_queries = set(start.index[cutoff:])
    work = work.sort_values(["query", "candidate_id"], kind="stable")
    order = work.row.to_numpy(np.int64)
    ordered_q = work["query"].to_numpy(object)
    group = work.groupby("query", sort=False).size().astype(int).tolist()
    fit = np.asarray([item in fit_queries for item in ordered_q], dtype=bool)
    tune = np.asarray([item in valid_queries for item in ordered_q], dtype=bool)
    if not fit.any() or not tune.any():
        raise AssertionError("insufficient causal query groups for R/U early stop")
    return order, ordered_q, group, fit, tune


def _group_sizes(ids: np.ndarray) -> list[int]:
    return pd.Series(ids).groupby(pd.Series(ids), sort=False).size().astype(int).tolist()


def _fit_head(*, train: pd.DataFrame, held: pd.DataFrame, fields: Sequence[str], spec: HeadSpec,
              fold_seed: int) -> tuple[pd.DataFrame, dict[str, object], pd.DataFrame]:
    sampled = _sample_complete_queries(train, TRAIN_CAP, SEED + fold_seed)
    valid, residual, sqrt_residual, atr_residual = _residual_targets(sampled, train=True)
    if spec.kind == "residual":
        labels, edges = _r_labels(sqrt_residual, valid)
        gain, truncation = R_GAIN, None
        params = dict(n_estimators=300, learning_rate=.045, max_depth=4, num_leaves=15, min_child_samples=350, reg_alpha=.02, reg_lambda=8.0, min_split_gain=.001)
        query_id = _query(sampled, "r")
    else:
        labels = _u_labels(sampled, valid, atr_residual)
        edges = []
        gain, truncation = U_GAIN, 20
        params = dict(n_estimators=300, learning_rate=.040, max_depth=2, num_leaves=3, min_child_samples=300, reg_alpha=.02, reg_lambda=8.0, min_split_gain=.001)
        query_id = _query(sampled, "u")
    keep = labels >= 0
    sampled = sampled.loc[keep].reset_index(drop=True)
    labels = labels[keep]
    query_id = query_id[keep]
    if len(sampled) < 20_000 or len(np.unique(labels)) < 2:
        raise AssertionError(f"{spec.name}: insufficient target support")
    x_train, x_held = _impute(_matrix(sampled, fields), _matrix(held, fields))
    order, ordered_q, _groups, fit, tune = _ordered_groups(sampled, query_id)
    x, y = x_train[order], labels[order]
    model_kwargs: dict[str, object] = dict(
        objective="lambdarank", metric="ndcg", label_gain=gain,
        random_state=SEED + fold_seed, n_jobs=2, deterministic=True, force_col_wise=True, verbosity=-1,
        **params,
    )
    if truncation is not None:
        model_kwargs["lambdarank_truncation_level"] = truncation
    model = LGBMRanker(**model_kwargs)
    model.fit(
        x[fit], y[fit], group=_group_sizes(ordered_q[fit]),
        eval_set=[(x[tune], y[tune])], eval_group=[_group_sizes(ordered_q[tune])],
        callbacks=[early_stopping(30, verbose=False)],
    )
    raw = model.predict(x_held).astype(np.float32)
    out = held.loc[:, list(IDENTITY)].copy()
    out[f"{spec.name}_raw"] = raw
    out[f"{spec.name}_rank"] = _rank_desc(out.rename(columns={f"{spec.name}_raw": "score"}), "score")
    # Diagnostic outcomes stay in memory.  The target-free score receipt is
    # constructed before they are inspected.
    held_anchor = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(
        sampled.loc[_anchor_valid(sampled), "base_rank_ts"], sampled.loc[_anchor_valid(sampled), "policy_net_bps"],
    )
    held_valid, held_residual, _held_sqrt, held_atr = _residual_targets(held, train=False, held_anchor=held_anchor)
    if spec.kind == "residual":
        target_for_ic = held_residual
    else:
        target_for_ic = _u_labels(held, held_valid, held_atr).astype(float)
        target_for_ic[target_for_ic < 0] = np.nan
    policy = held.loc[:, ["candidate_id", "policy_path_valid", "policy_net_bps"]].copy()
    cache = pd.DataFrame({
        "candidate_id": held.candidate_id,
        "target": target_for_ic,
        "residual_bps": held_residual,
        "target_valid": held_valid,
        "best_iteration": int(model.best_iteration_ or model.n_estimators),
    })
    info = {
        "head": spec.name, "train_rows": int(len(sampled)), "best_iteration": int(model.best_iteration_ or model.n_estimators),
        "query": "base_rank_band_x_28d" if spec.kind == "residual" else "decision_timestamp_x_long",
        "target": "sqrt_atr_residual_quintile" if spec.kind == "residual" else "clean_trailing_and_unexpected_plus_1atr",
        "r_edges": edges, "held_target_valid": int(held_valid.sum()), "policy": policy,
    }
    return out, info, cache


def _bin(values: np.ndarray, bins: int = 10) -> np.ndarray:
    output = np.full(len(values), -1, dtype=np.int16)
    finite = np.isfinite(values)
    if int(finite.sum()) >= 2:
        ranks = pd.Series(values[finite]).rank(method="average", pct=True).to_numpy(float)
        output[finite] = np.minimum(bins - 1, np.floor(ranks * bins)).astype(np.int16)
    return output


def _conditional_mi(score: np.ndarray, base: np.ndarray, outcome: np.ndarray) -> float:
    s, b, y = _bin(score), _bin(base), _bin(outcome)
    valid = (s >= 0) & (b >= 0) & (y >= 0)
    if int(valid.sum()) < 100:
        return float("nan")
    total, value = float(valid.sum()), 0.0
    for band in np.unique(b[valid]):
        mask = valid & (b == band)
        if int(mask.sum()) >= 20:
            value += float(mask.sum()) / total * mutual_info_score(s[mask], y[mask])
    return float(value)


def _substitution(frame: pd.DataFrame, field: str, k: int) -> float:
    work = frame.loc[frame.policy_path_valid.fillna(False).astype(bool) & np.isfinite(pd.to_numeric(frame.policy_net_bps, errors="coerce"))].copy()
    work["combined"] = .70 * pd.to_numeric(work.base_rank_ts, errors="coerce") + .15 * pd.to_numeric(work.r_residual_sqrt_atr_quintile_rank, errors="coerce") + .15 * pd.to_numeric(work.u_unexpected_trailing_atr1_rank, errors="coerce")
    selected = work.sort_values(["__decision_ts__", field, "candidate_id"], ascending=[True, False, True], kind="stable").groupby("__decision_ts__", sort=False).head(k)
    return float(selected.groupby("__decision_ts__", sort=False).policy_net_bps.mean().mean())


def _mini_mc1(*, history: pd.DataFrame, held: pd.DataFrame, reserve: pd.Timestamp, seed: int) -> tuple[dict[str, object], pd.DataFrame] | None:
    # The predeclared mini probe waits for three complete prior R/U score
    # months.  It is intentionally single-family; final candidates alone
    # advance to the unchanged dual Current/BCF MC1 replay.
    train = history.loc[
        history.__decision_ts__.lt(reserve)
        & history.policy_label_available_ts.lt(reserve)
        & history.policy_path_valid.fillna(False).astype(bool)
        & np.isfinite(pd.to_numeric(history.policy_net_bps, errors="coerce")),
    ].copy()
    if train.__decision_ts__.dt.to_period("M").nunique() < MIN_MINI_MONTHS:
        return None
    train = _sample_complete_queries(train, TRAIN_CAP, seed)
    fields = ["base_rank_ts", "r_residual_sqrt_atr_quintile_rank", "u_unexpected_trailing_atr1_rank", "base_query_std", "base_query_range", "base_top_gap", "base_top2_gap"]
    baseline_fields = ["base_rank_ts", "base_query_std", "base_query_range", "base_top_gap", "base_top2_gap"]
    valid_held = held.policy_path_valid.fillna(False).astype(bool) & np.isfinite(pd.to_numeric(held.policy_net_bps, errors="coerce"))
    output = held.loc[:, [*IDENTITY, "policy_path_valid", "policy_net_bps"]].copy()
    for name, columns in (("base_only", baseline_fields), ("base_ru", fields)):
        x_train = train.loc[:, columns].apply(pd.to_numeric, errors="coerce")
        medians = x_train.median(axis=0).fillna(0.0)
        x_train = x_train.fillna(medians).to_numpy(np.float32)
        x_held = held.loc[:, columns].apply(pd.to_numeric, errors="coerce").fillna(medians).to_numpy(np.float32)
        model = LGBMRegressor(
            objective="huber", n_estimators=220, learning_rate=.04, max_depth=2, num_leaves=7,
            min_child_samples=500, colsample_bytree=.85, subsample=.85, subsample_freq=1,
            reg_alpha=.05, reg_lambda=10.0, random_state=seed, n_jobs=2,
            deterministic=True, force_col_wise=True, verbosity=-1,
        )
        model.fit(x_train, np.clip(pd.to_numeric(train.policy_net_bps, errors="coerce").to_numpy(float), -600.0, 600.0))
        output[f"{name}_expected_bps"] = model.predict(x_held).astype(np.float32)
    rows: dict[str, object] = {"held_month": f"{held.__decision_ts__.min():%Y-%m}", "train_rows": int(len(train)), "train_months": int(train.__decision_ts__.dt.to_period("M").nunique())}
    for name in ("base_only", "base_ru"):
        admitted = output.loc[output[f"{name}_expected_bps"].ge(50.0)].copy()
        realised = admitted.loc[admitted.policy_path_valid.fillna(False).astype(bool) & np.isfinite(pd.to_numeric(admitted.policy_net_bps, errors="coerce"))]
        rows[f"{name}_admitted"] = int(len(admitted))
        rows[f"{name}_realised_ev_bps"] = float(realised.policy_net_bps.mean()) if len(realised) else np.nan
        rows[f"{name}_total_bps"] = float(realised.policy_net_bps.sum()) if len(realised) else np.nan
    rows["delta_admitted"] = int(rows["base_ru_admitted"]) - int(rows["base_only_admitted"])
    rows["delta_ev_bps"] = float(rows["base_ru_realised_ev_bps"]) - float(rows["base_only_realised_ev_bps"])
    rows["delta_total_bps"] = float(rows["base_ru_total_bps"]) - float(rows["base_only_total_bps"])
    output["held_outcome_valid"] = valid_held.to_numpy(bool)
    return rows, output


def run(*, base_roots: Sequence[Path], base_target: str, feature_roots: Sequence[Path], policy_path: Path,
        path_root: Path, out: Path, held_months: Sequence[pd.Timestamp]) -> Path:
    if out.exists():
        raise FileExistsError(f"immutable output exists: {out}")
    fields = _load_f72(F72_SELECTION)
    if not held_months or tuple(sorted(held_months)) != tuple(held_months):
        raise ValueError("held months must be chronological and non-empty")
    policy = _read_policy(policy_path)
    out.mkdir(parents=True)
    all_scores: list[pd.DataFrame] = []
    all_metrics: list[dict[str, object]] = []
    mini_metrics: list[dict[str, object]] = []
    source_paths: list[Path] = [F72_SELECTION, policy_path]
    audits: list[dict[str, object]] = []
    for index, held_month in enumerate(held_months):
        reserve = held_month - pd.Timedelta(days=RESERVE_DAYS)
        start, end = reserve - pd.DateOffset(months=TRAIN_MONTHS), _month_end(held_month)
        window, consumed = _read_window(
            base_roots=base_roots, base_target=base_target, feature_roots=feature_roots,
            policy=policy, path_root=path_root, fields=fields, start=start, end=end,
        )
        source_paths.extend(consumed)
        train = window.loc[
            window.__decision_ts__.lt(reserve)
            & window.policy_label_available_ts.lt(reserve)
            & window.supportive_label_available_ts.lt(reserve),
        ].copy()
        held = window.loc[window.__decision_ts__.ge(held_month) & window.__decision_ts__.lt(end)].copy()
        if len(train) < 30_000 or len(held) < 10_000:
            raise AssertionError(f"{held_month:%Y-%m}: insufficient strict Base->R/U support {len(train)} / {len(held)}")
        head_scores: list[pd.DataFrame] = []
        caches: dict[str, tuple[dict[str, object], pd.DataFrame]] = {}
        for offset, spec in enumerate((R_SPEC, U_SPEC)):
            score, info, cache = _fit_head(train=train, held=held, fields=fields, spec=spec, fold_seed=100 * index + offset)
            target_free = score.loc[:, [*IDENTITY, f"{spec.name}_raw", f"{spec.name}_rank"]].copy()
            score_path = out / "target_free_scores" / spec.name / f"month={held_month:%Y-%m}.parquet"
            score_path.parent.mkdir(parents=True, exist_ok=True)
            target_free.to_parquet(score_path, index=False, compression="zstd")
            head_scores.append(target_free)
            caches[spec.name] = (info, cache)
        combined = held.loc[:, [*IDENTITY, "base_score", "base_rank_ts", "base_query_count", "base_query_std", "base_query_range", "base_top_gap", "base_top2_gap", "policy_path_valid", "policy_net_bps", "policy_label_available_ts"]].copy()
        for score in head_scores:
            combined = combined.merge(score, on=list(IDENTITY), how="inner", validate="one_to_one")
        if len(combined) != len(held):
            raise AssertionError("R/U target-free join changed held Base identity set")
        # Target-free combined source is persisted before metrics/outcomes are
        # inspected.  It is the only panel that later mini-MC1 can consume.
        combined_path = out / "target_free_combined" / f"month={held_month:%Y-%m}.parquet"
        combined_path.parent.mkdir(parents=True, exist_ok=True)
        combined.loc[:, [*IDENTITY, "base_score", "base_rank_ts", "base_query_count", "base_query_std", "base_query_range", "base_top_gap", "base_top2_gap", "r_residual_sqrt_atr_quintile_raw", "r_residual_sqrt_atr_quintile_rank", "u_unexpected_trailing_atr1_raw", "u_unexpected_trailing_atr1_rank"]].to_parquet(combined_path, index=False, compression="zstd")
        # The following copy is the post-write diagnostic branch.  The policy
        # columns were joined in memory before fitting, but never persisted in
        # the target-free combined receipt above.
        diagnostic = combined.copy()
        valid_policy = diagnostic.policy_path_valid.fillna(False).astype(bool).to_numpy() & np.isfinite(pd.to_numeric(diagnostic.policy_net_bps, errors="coerce").to_numpy(float))
        for spec in (R_SPEC, U_SPEC):
            info, cache = caches[spec.name]
            target = cache["target"].to_numpy(float)
            rank = diagnostic[f"{spec.name}_rank"].to_numpy(float)
            mask = valid_policy & np.isfinite(target)
            ic = float(spearmanr(rank[mask], target[mask]).statistic) if int(mask.sum()) >= 20 else np.nan
            cmi = _conditional_mi(rank[valid_policy], diagnostic.base_rank_ts.to_numpy(float)[valid_policy], pd.to_numeric(diagnostic.policy_net_bps, errors="coerce").to_numpy(float)[valid_policy])
            all_metrics.append({
                "held_month": f"{held_month:%Y-%m}", "head": spec.name, "residual_or_upside_ic": ic,
                "conditional_mi_policy_given_base": cmi, "target_valid_rows": int(mask.sum()),
                "train_rows": info["train_rows"], "best_iteration": info["best_iteration"],
                "query": info["query"], "target": info["target"], "r_edges": json.dumps(info["r_edges"]),
            })
        # The substitution diagnostic is a score-only check; it is never the
        # admission policy and cannot affect mini-MC1 fitting.
        base_top1, base_top2 = _substitution(diagnostic, "base_rank_ts", 1), _substitution(diagnostic, "base_rank_ts", 2)
        ru_top1, ru_top2 = _substitution(diagnostic, "combined", 1), _substitution(diagnostic, "combined", 2)
        all_metrics.append({
            "held_month": f"{held_month:%Y-%m}", "head": "R_plus_U_score_only_diagnostic",
            "residual_or_upside_ic": np.nan, "conditional_mi_policy_given_base": np.nan,
            "target_valid_rows": int(valid_policy.sum()), "train_rows": int(len(train)), "best_iteration": np.nan,
            "query": "diagnostic", "target": "0.70 Base rank + 0.15 R rank + 0.15 U rank",
            "base_top1_bps": base_top1, "ru_top1_bps": ru_top1, "delta_top1_bps": ru_top1 - base_top1,
            "base_top2_bps": base_top2, "ru_top2_bps": ru_top2, "delta_top2_bps": ru_top2 - base_top2,
            "r_edges": "[]",
        })
        all_scores.append(combined)
        history = pd.concat(all_scores, ignore_index=True)
        mini = _mini_mc1(history=history, held=combined, reserve=reserve, seed=SEED + index)
        if mini is not None:
            metric, predictions = mini
            mini_metrics.append(metric)
            mini_path = out / "mini_mc1_predictions" / f"month={held_month:%Y-%m}.parquet"
            mini_path.parent.mkdir(parents=True, exist_ok=True)
            predictions.loc[:, [*IDENTITY, "base_only_expected_bps", "base_ru_expected_bps"]].to_parquet(mini_path, index=False, compression="zstd")
        audits.append({
            "held_month": f"{held_month:%Y-%m}", "base_rows": int(len(held)), "train_rows": int(len(train)),
            "base_targetfree_source": True, "ru_scores_targetfree_before_metrics": True,
            "router_numeric_available_to_ru": False, "base_post_router_cutoff": False,
            "train_labels_resolved_before_reserve": bool(train.policy_label_available_ts.lt(reserve).all() and train.supportive_label_available_ts.lt(reserve).all()),
            "combined_targetfree_path": str(combined_path),
        })
        _progress(out, event="ru_fold_complete", held_month=f"{held_month:%Y-%m}", rows=int(len(held)))
    metrics = pd.DataFrame(all_metrics)
    metrics.to_parquet(out / "ru_head_metrics.parquet", index=False, compression="zstd")
    mini = pd.DataFrame(mini_metrics)
    mini.to_parquet(out / "mini_mc1_metrics.parquet", index=False, compression="zstd")
    summary = metrics.loc[metrics["head"].isin((R_SPEC.name, U_SPEC.name))].groupby("head", sort=True).agg(
        folds=("held_month", "nunique"), ic=("residual_or_upside_ic", "mean"), cmi=("conditional_mi_policy_given_base", "mean"),
    ).reset_index()
    if not mini.empty:
        summary["mini_delta_ev_bps"] = float(mini.delta_ev_bps.mean())
        summary["mini_delta_total_bps"] = float(mini.delta_total_bps.sum())
        summary["mini_delta_admitted"] = int(mini.delta_admitted.sum())
    summary.to_parquet(out / "ru_probe_summary.parquet", index=False, compression="zstd")
    _exclusive_json(out / "correctness_report.json", {
        "all_base_scores_target_free": True,
        "all_base_rank_matches_raw_score": True,
        "all_ru_scores_written_before_outcome_metrics": True,
        "all_train_labels_resolved_before_reserve": bool(all(item["train_labels_resolved_before_reserve"] for item in audits)),
        "any_router_numeric_input_to_base_ru_or_mini_mc1": False,
        "any_post_router_base_cutoff": False,
        "r_u_are_fixed_probes_no_hpo": True,
        "mini_mc1_is_selection_diagnostic_only": True,
    })
    _exclusive_json(out / "run_manifest.json", {
        "schema": SCHEMA,
        "scope": "offline Router50 Base->R/U downstream selection probe; no dual-MC1, live, inference, admission, portfolio, or exchange mutation",
        "architecture": "Router top50 -> one strict-OOF Base -> fixed R + U -> causal single-family mini-MC1 diagnostic",
        "base_score_roots": [str(root) for root in base_roots], "base_target": base_target,
        "feature_roots": [str(root) for root in feature_roots], "features": list(fields),
        "heads": {
            "R": "sqrt-ATR residual quintile; base-score-band x 28-day query; fixed LambdaRank",
            "U": "unexpected clean trailing above +1 ATR residual; timestamp x long query; fixed LambdaRank",
        },
        "mini_mc1": "causal 3-month-minimum, shallow single-family expected-policy-net probe; never a dual-MC1 replacement",
        "held_months": [f"{month:%Y-%m}" for month in held_months],
        "source_sha256": _sha256(source_paths), "fold_audits": audits,
    })
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-score-roots", required=True, help="comma-separated immutable warm-up/main Base score roots")
    parser.add_argument("--base-target", required=True)
    parser.add_argument("--feature-roots", required=True, help="comma-separated immutable causal feature roots")
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--path-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--held-months", default="2026-02,2026-03,2026-04,2026-05,2026-06,2026-07")
    args = parser.parse_args()
    months = tuple(_utc(f"{token.strip()}-01") for token in args.held_months.split(",") if token.strip())
    print(run(
        base_roots=tuple(Path(item.strip()).resolve() for item in args.base_score_roots.split(",") if item.strip()),
        base_target=args.base_target,
        feature_roots=tuple(Path(item.strip()).resolve() for item in args.feature_roots.split(",") if item.strip()),
        policy_path=args.policy.resolve(), path_root=args.path_root.resolve(), out=args.out.resolve(), held_months=months,
    ))


if __name__ == "__main__":
    main()
