#!/usr/bin/env python3
"""Strict-OOF full-feature selection for a frozen P8u Meta family head.

The selector is intentionally staged so a long full-universe read can be
audited and resumed without changing any prior result:

``prescreen``
    full target-free universe -> hygiene -> cross-era univariate conditional
    IC/CMI -> numerical redundancy representatives;
``subspace``
    randomized shallow ranker subspaces, including gain and tail-SHAP;
``mda``
    group permutation MDA, semantic-family accounting, a 120..25 ladder, and
    three frozen candidate contracts.

All supervision is restricted to the pre-reserve training population.  Held
features and scores are target-free through fitting; policy/path outcomes join
only to evaluate the designated development folds.  This producer does not
fit MC1, change a live bundle, or contact an exchange.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from lightgbm import LGBMRanker
from scipy.stats import spearmanr
from sklearn.cluster import MiniBatchKMeans

import run_strict_r3_p8u_meta_target_query_grid_v1 as screen


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "strict_r3_p8u_meta_fullfeatures_selection_v1"
IDENTITY = screen.IDENTITY
GEOMETRY = (
    "base_score", "base_rank_ts", "base_query_count", "base_query_mean",
    "base_query_std", "base_query_range", "base_score_z_ts", "base_top_gap",
    "base_top2_gap",
)
# ``GEOMETRY`` is the frozen causal Base Explanation V1 contract.  The CMI
# prescreen must assess whether a candidate feature adds policy information
# *beyond this full description*, not merely beyond ``base_rank_ts``.
BASE_EXPLANATION_V1 = GEOMETRY
SEED = 1729
# The historical default is retained for backwards-compatible invocations.
# New research must pass a predeclared arm explicitly; feature evidence is
# target-specific and must be compared to that same Meta -> MC1 baseline.
ARM_NAME = "under_bps100__timestamp"
DEFAULT_HELD_MONTHS = ("2026-01", "2026-02", "2026-03")


@dataclasses.dataclass
class Fold:
    held_month: pd.Timestamp
    start: pd.Timestamp
    reserve: pd.Timestamp
    end: pd.Timestamp
    train: pd.DataFrame
    held: pd.DataFrame
    labels: np.ndarray
    groups: list[int]
    held_labelled: pd.DataFrame
    held_anchor: Any


def _once(path: Path, payload: object) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def _progress(root: Path, **payload: object) -> None:
    with (root / "progress.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    members = sorted(path.rglob("*.parquet")) if path.is_dir() else [path]
    for member in members:
        digest.update(str(member).encode())
        with member.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    return digest.hexdigest()


def _months(values: Sequence[str]) -> tuple[pd.Timestamp, ...]:
    parsed = tuple(screen._utc_month(value) for value in values)
    if len(parsed) < 3 or tuple(sorted(parsed)) != parsed:
        raise ValueError("need at least three ascending strict-OOF held months")
    return parsed


def _arm(spec: screen.Spec, arm_name: str | None = None) -> screen.Arm:
    """Resolve one predeclared target without changing the default survivor.

    The historical default remains ``under_bps100__timestamp``.  Research
    callers may select a declared Under, Over, or Magnitude target; this
    prevents a feature-selection run from silently substituting an objective.
    """
    name = ARM_NAME if arm_name is None else str(arm_name)
    item = next((value for value in screen._arm_specs(spec.raw, None) if value.name == name), None)
    if item is None:
        raise AssertionError(f"missing declared target {name}")
    if item.family not in {"under", "over", "magnitude", "state"}:
        raise AssertionError(f"full selection does not support family {item.family}")
    return item


def _semantic_family(field: str) -> str:
    """Conservative accounting family, never a feature eligibility rule."""
    token = str(field).lower()
    for prefix, family in (
        ("mkt_", "market"), ("regime_", "regime"), ("transition", "transition"),
        ("state_", "state"), ("eigen_", "eigen"), ("iqr_", "iqr"),
        ("ae_", "ae"), ("gmm_", "gmm"), ("k9_", "k9"), ("reliability_", "reliability"),
        ("oi_", "oi"), ("funding_", "funding"), ("ob_", "orderbook"),
    ):
        if token.startswith(prefix) or f"__{prefix}" in token:
            return family
    head = token.split("__", 1)[0].split("_", 2)
    return "_".join(head[:2]) if len(head) >= 2 else head[0]


def _source(spec: screen.Spec) -> tuple[Path, tuple[Path, ...], Path, Path]:
    raw = spec.source
    return (
        ROOT / str(raw["base_target_free_root"]),
        tuple(ROOT / str(item) for item in raw["full_feature_roots"]),
        ROOT / str(raw["policy_labels"]),
        ROOT / str(raw["path_labels"]),
    )


def _source_months(held_months: Sequence[pd.Timestamp], *, train_months: int, reserve_days: int) -> tuple[pd.Timestamp, ...]:
    start = held_months[0] - pd.Timedelta(days=reserve_days) - pd.DateOffset(months=train_months)
    end = screen._month_end(held_months[-1])
    return screen._month_range(start, end)


def _full_fields(roots: Sequence[Path], months: Sequence[pd.Timestamp]) -> tuple[str, ...]:
    common: set[str] | None = None
    forbidden = set(IDENTITY) | {"__ts__", "__symbol__"} | set(screen.PROHIBITED) | set(GEOMETRY)
    for month in months:
        path = screen._full_path(roots, month)
        screen._assert_target_free(path)
        names = set(pq.ParquetFile(path).schema_arrow.names)
        current = {name for name in names if name not in forbidden}
        common = current if common is None else common.intersection(current)
    result = tuple(sorted(common or ()))
    # The versioned F120 + SHAP/state stack is deliberately narrower than the
    # original 900+ raw field universe, but each field has explicit causal
    # provenance.  Keep a real breadth guard without forcing the stale size.
    if len(result) < 200:
        raise AssertionError(f"full Meta universe unexpectedly small: {len(result)}")
    return result


def _hygiene(roots: Sequence[Path], months: Sequence[pd.Timestamp], fields: Sequence[str]) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for month in months:
        owner = screen._full_path(roots, month).parent / "feature_coverage.parquet"
        if owner.exists():
            part = pd.read_parquet(owner, columns=["feature", "finite_fraction", "n_unique"])
        else:
            # The append-only F120 + SHAP/state materializer stores one
            # immutable coverage ledger at the artifact root, with an
            # explicit month. Select only this month; never infer coverage
            # from later data.
            owner = screen._full_path(roots, month).parents[1] / "feature_coverage.parquet"
            if not owner.exists():
                raise FileNotFoundError(owner)
            part = pd.read_parquet(owner, columns=["month", "feature", "finite_fraction", "nunique"])
            part = part.loc[part["month"].astype(str).eq(f"{month:%Y-%m}")].drop(columns="month")
            part = part.rename(columns={"nunique": "n_unique"})
        part = part.loc[part.feature.isin(fields)].copy()
        part["month"] = f"{month:%Y-%m}"
        parts.append(part)
    joined = pd.concat(parts, ignore_index=True)
    summary = joined.groupby("feature", sort=True).agg(
        observed_months=("month", "nunique"),
        min_coverage=("finite_fraction", "min"),
        median_coverage=("finite_fraction", "median"),
        min_unique=("n_unique", "min"),
    ).reset_index()
    summary["family"] = summary.feature.map(_semantic_family)
    summary["hygiene_keep"] = (
        summary.observed_months.eq(len(months))
        & summary.min_coverage.ge(.90)
        & summary.min_unique.gt(1)
    )
    return summary.sort_values(["hygiene_keep", "min_coverage", "feature"], ascending=[False, False, True], kind="stable")


def _base_only(*, base_root: Path, feature_roots: Sequence[Path], start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Read the immutable Base panel without inventing a dummy feature field."""
    pieces: list[pd.DataFrame] = []
    for month in screen._month_range(start, end):
        base_path, feature_path = screen._base_path(base_root, month), screen._full_path(feature_roots, month)
        screen._assert_target_free(base_path); screen._assert_target_free(feature_path)
        base = pd.read_parquet(base_path, columns=list(screen.BASE_COLUMNS))
        feature_identity = pd.read_parquet(feature_path, columns=list(IDENTITY))
        for piece in (base, feature_identity):
            piece["__decision_ts__"] = pd.to_datetime(piece["__decision_ts__"], utc=True, errors="raise")
            if piece.duplicated(IDENTITY).any():
                raise AssertionError(f"{month:%Y-%m}: duplicate target-free identity")
        joined = base.merge(feature_identity, on=list(IDENTITY), how="left", validate="one_to_one", indicator=True)
        if len(joined) != len(base) or not joined._merge.eq("both").all():
            raise AssertionError(f"{month:%Y-%m}: Base/full-feature identity mismatch")
        pieces.append(joined.loc[joined.__decision_ts__.ge(start) & joined.__decision_ts__.lt(end), list(screen.BASE_COLUMNS)].copy())
    result = pd.concat(pieces, ignore_index=True).sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)
    if result.duplicated(IDENTITY).any() or not result.side_name.eq("long").all():
        raise AssertionError("invalid target-free P8u Base population")
    return screen._add_base_geometry(result)


def _context(
    *, held_month: pd.Timestamp, base_root: Path, feature_roots: Sequence[Path], policy: pd.DataFrame,
    path_root: Path, spec: screen.Spec, arm: screen.Arm, seed: int,
) -> Fold:
    reserve_days, train_months = int(spec.folds["resolved_label_reserve_days"]), int(spec.folds["train_months"])
    reserve = held_month - pd.Timedelta(days=reserve_days)
    start, end = reserve - pd.DateOffset(months=train_months), screen._month_end(held_month)
    # Geometry comes from the immutable target-free Base receipt.  No full
    # feature needs to be held in memory until a bounded screen/probe block.
    base = _base_only(base_root=base_root, feature_roots=feature_roots, start=start, end=end)
    train_tf = base.loc[base.__decision_ts__.lt(reserve)].copy()
    held = base.loc[base.__decision_ts__.ge(held_month)].copy()
    train_labelled = screen._labelled(train_tf, policy, path_root, start, reserve)
    train_labelled = train_labelled.loc[screen._valid_label(train_labelled, reserve)].copy().reset_index(drop=True)
    if len(train_labelled) < 30_000 or len(held) < 10_000:
        raise AssertionError(f"{held_month:%Y-%m}: insufficient strict Meta support")
    anchors = screen._prequential_anchor(train_labelled, block_days=int(spec.folds["anchor_block_days"]))
    labels, residual, target_info = screen._train_target(train_labelled, arm, anchor=anchors)
    valid = labels >= 0
    sampled = screen._sample_queries(train_labelled.loc[valid].copy(), int(spec.folds["max_train_rows"]), seed)
    label_frame = train_labelled.loc[:, list(IDENTITY)].copy()
    label_frame["label"] = labels
    label_frame["prequential_residual_bps"] = residual
    sampled = sampled.merge(label_frame, on=list(IDENTITY), how="left", validate="one_to_one")
    if sampled.label.isna().any():
        raise AssertionError("query sample lost its strict-prequential target")
    order, _qid, groups = screen._bounded_queries(sampled, screen._query_ids(sampled, arm.query), int(spec.folds["max_query_rows"]))
    sampled = sampled.iloc[order].reset_index(drop=True)
    y = sampled.label.to_numpy(np.int32)
    if len(sampled) < 20_000 or len(np.unique(y)) < 2:
        raise AssertionError(f"{held_month:%Y-%m}: inadequate under-confidence training classes")
    held_labelled = screen._labelled(held, policy, path_root, held_month, end)
    held_anchor = screen._fit_anchor(train_labelled, np.ones(len(train_labelled), dtype=bool))
    sampled.attrs["target_info"] = target_info
    return Fold(held_month, start, reserve, end, sampled, held.reset_index(drop=True), y, groups, held_labelled, held_anchor)


def _contexts(spec: screen.Spec, arm: screen.Arm, months: Sequence[pd.Timestamp]) -> list[Fold]:
    base_root, roots, policy_path, path_root = _source(spec)
    policy = screen._read_policy(policy_path)
    return [
        _context(
            held_month=month, base_root=base_root, feature_roots=roots, policy=policy, path_root=path_root,
            spec=spec, arm=arm, seed=SEED + index,
        )
        for index, month in enumerate(months)
    ]


def _with_fields(fold: Fold, fields: Sequence[str], *, spec: screen.Spec) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_root, roots, _policy_path, _path_root = _source(spec)
    panel = screen._read_base_features(base_root=base_root, feature_roots=roots, start=fold.start, end=fold.end, fields=fields)
    train_panel = panel.loc[panel.__decision_ts__.lt(fold.reserve), list(IDENTITY) + list(fields)]
    held_panel = panel.loc[panel.__decision_ts__.ge(fold.held_month), list(IDENTITY) + list(fields)]
    train = fold.train.merge(train_panel, on=list(IDENTITY), how="left", validate="one_to_one")
    held = fold.held.merge(held_panel, on=list(IDENTITY), how="left", validate="one_to_one")
    if len(train) != len(fold.train) or len(held) != len(fold.held):
        raise AssertionError("feature join changed P8u Base identities")
    if fields and (train.loc[:, list(fields)].isna().all(axis=None) or held.loc[:, list(fields)].isna().all(axis=None)):
        raise AssertionError("selected full-causal fields have no identity coverage")
    return train, held


def _impute_matrix(train: pd.DataFrame, held: pd.DataFrame, fields: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    x_train = screen._matrix(train, fields)
    x_held = screen._matrix(held, fields)
    return screen._impute(x_train, x_held)


def _model(*, classes: int, seed: int, n_jobs: int) -> LGBMRanker:
    return LGBMRanker(
        objective="rank_xendcg", metric="ndcg", label_gain=list((0, 1, 2, 4, 7, 11, 16, 24)[:classes]),
        n_estimators=260, learning_rate=.045, max_depth=4, num_leaves=15, min_child_samples=350,
        min_split_gain=.001, colsample_bytree=.80, subsample=.82, subsample_freq=1,
        reg_alpha=.02, reg_lambda=8.0, random_state=seed, n_jobs=n_jobs, verbosity=-1,
    )


def _score(
    *, fold: Fold, fields: Sequence[str], spec: screen.Spec, seed: int, n_jobs: int,
    held_take: np.ndarray | None = None,
) -> tuple[pd.DataFrame, Mapping[str, Any], LGBMRanker, np.ndarray, pd.DataFrame, np.ndarray]:
    train, held = _with_fields(fold, fields, spec=spec)
    x_train, x_held = _impute_matrix(train, held, fields)
    model = _model(classes=int(np.nanmax(fold.labels)) + 1, seed=seed, n_jobs=n_jobs)
    model.fit(x_train, fold.labels, group=fold.groups)
    if held_take is not None:
        held = held.iloc[held_take].reset_index(drop=True)
        x_held = x_held[held_take]
        held_labelled = fold.held_labelled.merge(held.loc[:, list(IDENTITY)], on=list(IDENTITY), how="inner", validate="one_to_one")
    else:
        held_labelled = fold.held_labelled
    raw = np.asarray(model.predict(x_held), dtype=np.float32)
    score = held.loc[:, list(IDENTITY) + ["base_score", "base_rank_ts"]].copy()
    score["meta_raw_score"] = raw
    rank_frame = score.loc[:, ["candidate_id", "__decision_ts__"]].copy(); rank_frame["value"] = raw
    score["meta_rank_ts"] = screen._rank_desc(rank_frame, "value")
    weekly, _bands, metrics = screen._metrics(score=score, held_labelled=held_labelled, held_anchor=fold.held_anchor, spec=spec)
    return score, metrics, model, x_held, held, raw


def _metric_table(records: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(records)
    if frame.empty:
        raise AssertionError("empty full-feature selector metrics")
    return frame


def _held_target(fold: Fold, arm: screen.Arm) -> tuple[np.ndarray, np.ndarray]:
    anchor = fold.held_anchor.predict(fold.held_labelled.base_rank_ts).astype(np.float32)
    labels, residual, _info = screen._train_target(fold.held_labelled, arm, anchor=anchor)
    return labels, residual


def _rank_bins(values: np.ndarray, *, bins: int = 10) -> np.ndarray:
    """Return deterministic within-fold quantile bins, with -1 for missing."""
    out = np.full(len(values), -1, dtype=np.int16)
    valid = np.isfinite(values)
    if int(valid.sum()) < 20:
        return out
    rank = pd.Series(values[valid]).rank(method="average", pct=True).to_numpy(float)
    out[valid] = np.minimum(bins - 1, np.floor(rank * bins)).astype(np.int16)
    return out


def _base_explanation_strata(frame: pd.DataFrame, evidence_rows: np.ndarray) -> np.ndarray:
    """Compress Base Explanation V1 into outcome-free, deterministic strata.

    Directly conditioning a binned CMI on nine fields would create mostly
    singleton cells.  A small fixed MiniBatchKMeans representation preserves
    the joint Base score/context geometry while leaving all policy/path labels
    out of the construction.  It is fitted independently per held fold and
    only on the predeclared top-Base evidence population.
    """
    if len(frame) != len(evidence_rows):
        raise AssertionError("Base Explanation rows do not align to evidence mask")
    raw = frame.loc[:, list(BASE_EXPLANATION_V1)].apply(pd.to_numeric, errors="coerce")
    rows = np.asarray(evidence_rows, dtype=bool)
    if int(rows.sum()) < 500:
        return np.full(len(frame), -1, dtype=np.int16)
    values = raw.to_numpy(float)
    # Rank-normalisation makes this a geometry representation rather than a
    # scale-sensitive score transform.  All medians are feature-only and fit
    # within the fold's evidence rows.
    normalized = np.empty_like(values, dtype=np.float32)
    active: list[int] = []
    for column in range(values.shape[1]):
        current = values[:, column]
        support = current[rows & np.isfinite(current)]
        if len(support) < 500 or float(np.nanmax(support) - np.nanmin(support)) <= 1e-12:
            continue
        median = float(np.nanmedian(support))
        filled = np.where(np.isfinite(current), current, median)
        normalized[:, column] = pd.Series(filled).rank(method="average", pct=True).to_numpy(np.float32)
        active.append(column)
    if len(active) < 2:
        return np.full(len(frame), -1, dtype=np.int16)
    matrix = normalized[np.ix_(rows, active)]
    # Twelve strata keep every conditional cell well supported on the
    # 15-percent Base tail while retaining joint score/context distinctions.
    clusters = min(12, max(4, int(np.sqrt(len(matrix) / 300.0))))
    model = MiniBatchKMeans(
        n_clusters=clusters, random_state=SEED, n_init=3,
        batch_size=min(4096, len(matrix)), max_iter=100,
    )
    result = np.full(len(frame), -1, dtype=np.int16)
    result[rows] = model.fit_predict(matrix).astype(np.int16)
    return result


def _conditional_mi_binned(feature: np.ndarray, condition: np.ndarray, outcome: np.ndarray) -> float:
    """Estimate I(feature; outcome | condition) with fixed support guards."""
    x, y = _rank_bins(np.asarray(feature, dtype=float)), _rank_bins(np.asarray(outcome, dtype=float))
    z = np.asarray(condition, dtype=np.int16)
    valid = (x >= 0) & (y >= 0) & (z >= 0)
    if int(valid.sum()) < 100:
        return float("nan")
    total, result = float(valid.sum()), 0.0
    for state in np.unique(z[valid]):
        local = valid & (z == state)
        if int(local.sum()) >= 20:
            result += float(local.sum()) / total * screen.mutual_info_score(x[local], y[local])
    return float(result)


def _feature_base_explanation_mi(feature: np.ndarray, condition: np.ndarray) -> float:
    """Record explanatory redundancy without making it a hard veto."""
    x, z = _rank_bins(np.asarray(feature, dtype=float)), np.asarray(condition, dtype=np.int16)
    valid = (x >= 0) & (z >= 0)
    return float(screen.mutual_info_score(x[valid], z[valid])) if int(valid.sum()) >= 100 else float("nan")


def _screen_block(
    *, fold: Fold, fields: Sequence[str], spec: screen.Spec, arm: screen.Arm, top_base_fraction: float,
) -> list[dict[str, Any]]:
    _train, held = _with_fields(fold, fields, spec=spec)
    labels, residual = _held_target(fold, arm)
    net = pd.to_numeric(fold.held_labelled.policy_net_bps, errors="coerce").to_numpy(float)
    if not 0.0 < float(top_base_fraction) <= 1.0:
        raise ValueError("top_base_fraction must be in (0, 1]")
    # ``base_rank_ts`` is descending: the strongest candidate has rank near
    # one.  Restrict only the CMI/feature-evidence population, not later
    # model training or held scoring, to the requested Base top tail.
    top_cut = 1.0 - float(top_base_fraction)
    valid = (
        screen._valid_label(fold.held_labelled) & (labels >= 0)
        & np.isfinite(residual) & np.isfinite(net)
        & (fold.held.base_rank_ts.to_numpy(float) >= top_cut)
    )
    base = fold.held.base_rank_ts.to_numpy(float)
    base_explanation_strata = _base_explanation_strata(held, valid)
    records: list[dict[str, Any]] = []
    for field in fields:
        values = pd.to_numeric(held[field], errors="coerce").to_numpy(float)
        rows = valid & np.isfinite(values) & np.isfinite(base)
        if int(rows.sum()) < 500:
            records.append({"feature": field, "family": _semantic_family(field), "held_month": f"{fold.held_month:%Y-%m}", "rows": int(rows.sum())})
            continue
        records.append({
            "feature": field, "family": _semantic_family(field), "held_month": f"{fold.held_month:%Y-%m}",
            "rows": int(rows.sum()),
            # Keep the legacy ``under_ic`` column for historical-reader
            # compatibility, but publish the target-neutral name as the
            # authoritative value for Under, Over, and Magnitude runs.
            "target_ic": float(spearmanr(values[rows], labels[rows]).statistic),
            "under_ic": float(spearmanr(values[rows], labels[rows]).statistic),
            "residual_ic": float(spearmanr(values[rows], residual[rows]).statistic),
            "cmi_policy_given_base": screen._conditional_mi(values[rows], base[rows], net[rows]),
            "cmi_policy_given_base_explanation_v1": _conditional_mi_binned(
                values[rows], base_explanation_strata[rows], net[rows],
            ),
            "mi_feature_base_explanation_v1": _feature_base_explanation_mi(
                values[rows], base_explanation_strata[rows],
            ),
            "coverage": float(np.mean(np.isfinite(values))),
        })
    return records


def _summarise_screen(records: pd.DataFrame) -> pd.DataFrame:
    result = records.groupby(["feature", "family"], sort=True).agg(
        folds=("held_month", "nunique"),
        target_ic_median=("target_ic", "median"), target_ic_q25=("target_ic", lambda x: float(np.nanquantile(np.abs(x), .25))),
        residual_ic_median=("residual_ic", "median"), residual_ic_q25=("residual_ic", lambda x: float(np.nanquantile(np.abs(x), .25))),
        cmi_median=("cmi_policy_given_base", "median"), cmi_q25=("cmi_policy_given_base", lambda x: float(np.nanquantile(x, .25))),
        cmi_base_explanation_median=("cmi_policy_given_base_explanation_v1", "median"),
        cmi_base_explanation_q25=("cmi_policy_given_base_explanation_v1", lambda x: float(np.nanquantile(x, .25))),
        base_explanation_redundancy_median=("mi_feature_base_explanation_v1", "median"),
        min_coverage=("coverage", "min"),
    ).reset_index()
    stable = (
        result.target_ic_q25.rank(pct=True) + result.residual_ic_q25.rank(pct=True) + result.cmi_base_explanation_q25.rank(pct=True)
    ) / 3.0
    central = (
        result.target_ic_median.abs().rank(pct=True) + result.residual_ic_median.abs().rank(pct=True) + result.cmi_base_explanation_median.rank(pct=True)
    ) / 3.0
    result["prescreen_score"] = .60 * stable + .40 * central
    return result.sort_values(["prescreen_score", "feature"], ascending=[False, True], kind="stable")


def _veto_values(*, fields: Sequence[str], folds: Sequence[Fold], spec: screen.Spec, cap: int) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    per_fold = max(1, cap // max(1, len(folds)))
    for number, fold in enumerate(folds):
        _train, held = _with_fields(fold, fields, spec=spec)
        token = held.candidate_id.astype(str) + f"|{SEED + number}"
        order = pd.util.hash_pandas_object(token, index=False).to_numpy(np.uint64).argsort(kind="stable")[:per_fold]
        parts.append(held.iloc[order].loc[:, list(fields)].copy())
    return pd.concat(parts, ignore_index=True)


def _redundancy(summary: pd.DataFrame, values: pd.DataFrame, *, ceiling: float, limit: int) -> pd.DataFrame:
    ordered = summary.feature.astype(str).tolist()
    numeric = values.loc[:, ordered].apply(pd.to_numeric, errors="coerce")
    correlation = numeric.rank(method="average", pct=True).corr(method="pearson").abs()
    retained: list[str] = []; representative: dict[str, str] = {}
    for field in ordered:
        if len(retained) >= limit:
            break
        prior = next((item for item in retained if np.isfinite(correlation.loc[field, item]) and correlation.loc[field, item] >= ceiling), None)
        if prior is None:
            retained.append(field)
        else:
            representative[field] = prior
    audit = summary.loc[:, ["feature", "family", "prescreen_score"]].copy()
    audit["kept_after_redundancy"] = audit.feature.isin(retained)
    audit["redundancy_representative"] = audit.feature.map(representative)
    return audit.sort_values(["kept_after_redundancy", "prescreen_score", "feature"], ascending=[False, False, True], kind="stable")


def _subspace(folds: Sequence[Fold], fields: Sequence[str], spec: screen.Spec, *, probes: int, width: int, n_jobs: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    records: list[dict[str, Any]] = []; evidence: list[dict[str, Any]] = []
    width = min(width, len(fields))
    for fold_index, fold in enumerate(folds):
        for probe in range(probes):
            rng = np.random.default_rng(SEED + 100_000 * fold_index + probe)
            subset = tuple(sorted(rng.choice(np.asarray(fields, dtype=object), width, replace=False).astype(str).tolist()))
            _score_frame, metrics, model, x_held, held, _raw = _score(
                fold=fold, fields=subset, spec=spec, seed=SEED + 10_000 * fold_index + probe, n_jobs=n_jobs,
            )
            gain = model.booster_.feature_importance(importance_type="gain")[len(GEOMETRY):]
            tail = np.flatnonzero(held.base_rank_ts.to_numpy(float) >= .90)[:4_000]
            shap = np.zeros(len(subset), dtype=float)
            if len(tail):
                shap = np.mean(np.abs(model.booster_.predict(x_held[tail], pred_contrib=True)[:, len(GEOMETRY):-1]), axis=0)
            records.append({"held_month": f"{fold.held_month:%Y-%m}", "probe": probe, "fields": list(subset), **metrics})
            for index, field in enumerate(subset):
                evidence.append({
                    "held_month": f"{fold.held_month:%Y-%m}", "probe": probe, "feature": field,
                    "gain": float(gain[index]), "tail_abs_shap": float(shap[index]),
                    "sstable_meta": float(metrics["sstable_meta"]), "conditional_mi": float(metrics["conditional_mi_meta_policy_given_base"]),
                    "top2_substitution": float(metrics["mean_top2_substitution_ev_bps"]),
                })
            del model, x_held
    return pd.DataFrame(records), pd.DataFrame(evidence)


def _subspace_summary(evidence: pd.DataFrame, fields: Sequence[str]) -> pd.DataFrame:
    quality = evidence.loc[:, ["held_month", "probe", "sstable_meta", "conditional_mi", "top2_substitution"]].drop_duplicates()
    quality["quality"] = (
        quality.groupby("held_month", sort=False).sstable_meta.rank(pct=True)
        + quality.groupby("held_month", sort=False).conditional_mi.rank(pct=True)
        + quality.groupby("held_month", sort=False).top2_substitution.rank(pct=True)
    ) / 3.0
    work = evidence.merge(quality.loc[:, ["held_month", "probe", "quality"]], on=["held_month", "probe"], how="left", validate="many_to_one")
    rows: list[dict[str, Any]] = []
    for field in fields:
        present = work.loc[work.feature.eq(field), "quality"]
        absent = quality.merge(work.loc[work.feature.eq(field), ["held_month", "probe"]], on=["held_month", "probe"], how="left", indicator=True)
        absent_quality = absent.loc[absent._merge.eq("left_only"), "quality"]
        field_rows = work.loc[work.feature.eq(field)]
        rows.append({
            "feature": field,
            "random_subspace_inclusion_uplift": float(present.mean() - absent_quality.mean()) if len(present) and len(absent_quality) else float("nan"),
            "random_subspace_inclusion_q25": float(np.quantile(present, .25) - np.quantile(absent_quality, .25)) if len(present) >= 2 and len(absent_quality) >= 2 else float("nan"),
            "gain_median": float(field_rows.gain.median()), "tail_shap_median": float(field_rows.tail_abs_shap.median()),
            "subspace_frequency": int(len(field_rows)),
        })
    return pd.DataFrame(rows)


def _permutation(frame: pd.DataFrame, seed: int) -> np.ndarray:
    work = frame.loc[:, ["candidate_id", "__decision_ts__"]].copy().reset_index(drop=True)
    work["row"] = np.arange(len(work), dtype=np.int64)
    work["hash"] = pd.util.hash_pandas_object(work.candidate_id.astype(str) + f"|{seed}", index=False).to_numpy(np.uint64)
    source = np.arange(len(work), dtype=np.int64)
    for _timestamp, group in work.sort_values(["__decision_ts__", "hash", "candidate_id"], kind="stable").groupby("__decision_ts__", sort=False):
        destination = group.row.to_numpy(np.int64)
        source[destination] = np.roll(destination, 1)
    return source


def _mda_groups(fields: Sequence[str], values: pd.DataFrame, *, ceiling: float = .94) -> dict[str, tuple[str, ...]]:
    numeric = values.loc[:, list(fields)].apply(pd.to_numeric, errors="coerce")
    correlation = numeric.rank(method="average", pct=True).corr(method="pearson").abs()
    groups: dict[str, list[str]] = {}
    for field in fields:
        found = next((rep for rep, members in groups.items() if any(np.isfinite(correlation.loc[field, member]) and correlation.loc[field, member] >= ceiling for member in members)), None)
        if found is None:
            groups[field] = [field]
        else:
            groups[found].append(field)
    return {key: tuple(value) for key, value in groups.items()}


def _mda(
    *, folds: Sequence[Fold], fields: Sequence[str], spec: screen.Spec, values: pd.DataFrame,
    held_cap: int, n_jobs: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    groups = _mda_groups(fields, values)
    result: list[dict[str, Any]] = []
    base_rows: list[dict[str, Any]] = []
    per_group: dict[str, list[float]] = {key: [] for key in groups}
    for fold_index, fold in enumerate(folds):
        token = fold.held.candidate_id.astype(str) + f"|mda|{SEED + fold_index}"
        order = pd.util.hash_pandas_object(token, index=False).to_numpy(np.uint64).argsort(kind="stable")
        # Sampling entire timestamp queries avoids an artefactual partial-query metric.
        sampled = screen._sample_queries(fold.held.iloc[order].copy(), held_cap, SEED + fold_index)
        lookup = pd.Series(np.arange(len(fold.held), dtype=np.int64), index=pd.MultiIndex.from_frame(fold.held.loc[:, list(IDENTITY)]))
        take = lookup.loc[pd.MultiIndex.from_frame(sampled.loc[:, list(IDENTITY)])].to_numpy(np.int64)
        base_score, base_metrics, model, x_held, held, _raw = _score(
            fold=fold, fields=fields, spec=spec, seed=SEED + 50_000 + fold_index, n_jobs=n_jobs, held_take=take,
        )
        base_rows.append({"held_month": f"{fold.held_month:%Y-%m}", **base_metrics})
        source = _permutation(held, SEED + 60_000 + fold_index)
        index = {field: number + len(GEOMETRY) for number, field in enumerate(fields)}
        label_sample = fold.held_labelled.merge(held.loc[:, list(IDENTITY)], on=list(IDENTITY), how="inner", validate="one_to_one")
        for group_index, (representative, members) in enumerate(groups.items()):
            altered = x_held.copy()
            positions = [index[field] for field in members]
            altered[:, positions] = x_held[source][:, positions]
            raw = np.asarray(model.predict(altered), dtype=np.float32)
            changed = held.loc[:, list(IDENTITY) + ["base_score", "base_rank_ts"]].copy()
            changed["meta_raw_score"] = raw
            rank_frame = changed.loc[:, ["candidate_id", "__decision_ts__"]].copy(); rank_frame["value"] = raw
            changed["meta_rank_ts"] = screen._rank_desc(rank_frame, "value")
            _weekly, _bands, metric = screen._metrics(score=changed, held_labelled=label_sample, held_anchor=fold.held_anchor, spec=spec)
            delta = float(base_metrics["sstable_meta"] - metric["sstable_meta"])
            per_group[representative].append(delta)
            result.append({
                "held_month": f"{fold.held_month:%Y-%m}", "representative": representative,
                "members": list(members), "mda_delta_sstable": delta,
                "mda_delta_top2_ev_bps": float(base_metrics["mean_top2_substitution_ev_bps"] - metric["mean_top2_substitution_ev_bps"]),
            })
            del altered
        del model, x_held
    summary = pd.DataFrame([
        {"representative": representative, "members": list(members), "mda_delta_sstable_median": float(np.median(per_group[representative])), "mda_positive_folds": int(np.sum(np.asarray(per_group[representative]) > 0.0))}
        for representative, members in groups.items()
    ])
    return pd.DataFrame(result), summary


def _scale(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    spread = float(numeric.max() - numeric.min())
    return pd.Series(0.0, index=numeric.index) if spread <= 1e-12 else (numeric - numeric.min()) / spread


def _write_contract(
    path: Path, *, fields: Sequence[str], ranked: pd.DataFrame, size: int, sstable: float, arm: screen.Arm,
) -> None:
    chosen = list(fields[:size])
    _once(path, {
        "schema": "strict_r3_p8u_meta_fullfeature_contract_v1",
        "scope": "offline full-causal P8u Meta candidate; no policy/path outcome enters inference inputs",
        "base_contract": "P8U_RAW_BPS_CATBOOST_QUERYRMSE_F72_TAIL125",
        "arm": arm.name, "family": arm.family, "query": arm.query, "objective": "rank_xendcg", "feature_count": len(chosen),
        "selected_features": chosen, "feature_sha256": hashlib.sha256("\n".join(chosen).encode()).hexdigest(),
        "selection": "full-universe hygiene + Base-top-tail conditional IC/CMI + redundancy + randomized subspace gain/tail-SHAP + group-MDA + bounded SStableMeta ladder",
        "development_sstable_meta": float(sstable),
        "feature_families": ranked.loc[ranked.feature.isin(chosen)].groupby("family", sort=True).size().to_dict(),
        "next_gate": "strict-prequential Meta-to-MC1, independent dual-MC1 mapping, and chronological constrained portfolio on later months",
    })


def _stage_prescreen(
    *, out: Path, spec: screen.Spec, arm: screen.Arm, folds: Sequence[Fold], fields: Sequence[str],
    block_size: int, veto_rows: int, top_base_fraction: float,
) -> None:
    if (out / "prescreen_summary.parquet").exists():
        raise FileExistsError("prescreen output already sealed")
    records: list[dict[str, Any]] = []
    for begin in range(0, len(fields), block_size):
        block = fields[begin:begin + block_size]
        for fold in folds:
            records.extend(_screen_block(
                fold=fold, fields=block, spec=spec, arm=arm, top_base_fraction=top_base_fraction,
            ))
        _progress(out, stage="prescreen_block", begin=begin, fields=len(block))
    observations = pd.DataFrame(records)
    summary = _summarise_screen(observations)
    candidate = summary.head(min(260, len(summary))).feature.astype(str).tolist()
    values = _veto_values(fields=candidate, folds=folds, spec=spec, cap=veto_rows)
    veto = _redundancy(summary.loc[summary.feature.isin(candidate)].copy(), values, ceiling=.985, limit=160)
    retained = veto.loc[veto.kept_after_redundancy, "feature"].astype(str).tolist()
    if len(retained) < 120:
        raise AssertionError(f"strict full prescreen retained only {len(retained)} fields")
    observations.to_parquet(out / "prescreen_observations.parquet", index=False, compression="zstd")
    summary.to_parquet(out / "prescreen_summary.parquet", index=False, compression="zstd")
    veto.to_parquet(out / "redundancy_veto.parquet", index=False, compression="zstd")
    values.to_parquet(out / "redundancy_sample.parquet", index=False, compression="zstd")
    _once(out / "prescreen_contract.json", {
        "schema": SCHEMA, "arm": arm.name, "family": arm.family,
        "candidate_fields": retained, "candidate_count": len(retained),
        "base_top_fraction_for_cmi": float(top_base_fraction),
        "selection": "top-260 cross-era full-causal Base-top-tail conditional-MI screen, then .985 redundancy representatives",
    })


def _stage_subspace(*, out: Path, spec: screen.Spec, folds: Sequence[Fold], probes: int, width: int, n_jobs: int) -> None:
    if (out / "subspace_summary.parquet").exists():
        raise FileExistsError("subspace output already sealed")
    candidate = tuple(json.loads((out / "prescreen_contract.json").read_text())["candidate_fields"])
    metrics, evidence = _subspace(folds, candidate, spec, probes=probes, width=width, n_jobs=n_jobs)
    summary = _subspace_summary(evidence, candidate)
    merged = pd.read_parquet(out / "prescreen_summary.parquet").merge(summary, on="feature", how="inner", validate="one_to_one")
    merged["subspace_rank_score"] = (
        .50 * merged.random_subspace_inclusion_uplift.fillna(-1e9).rank(pct=True)
        + .20 * merged.random_subspace_inclusion_q25.fillna(-1e9).rank(pct=True)
        + .15 * merged.gain_median.fillna(0.0).rank(pct=True)
        + .15 * merged.tail_shap_median.fillna(0.0).rank(pct=True)
    )
    merged["pre_mda_score"] = .60 * merged.prescreen_score.rank(pct=True) + .40 * merged.subspace_rank_score
    merged = merged.sort_values(["pre_mda_score", "feature"], ascending=[False, True], kind="stable")
    selected = merged.head(120).feature.astype(str).tolist()
    metrics.drop(columns="fields").to_parquet(out / "random_subspace_metrics.parquet", index=False, compression="zstd")
    evidence.to_parquet(out / "random_subspace_feature_evidence.parquet", index=False, compression="zstd")
    merged.to_parquet(out / "subspace_summary.parquet", index=False, compression="zstd")
    _once(out / "subspace_contract.json", {"schema": SCHEMA, "candidate_fields": selected, "candidate_count": len(selected), "selection": "randomized shallow strict-OOF ranker subspaces with cross-era inclusion, gain, and tail-SHAP evidence"})


def _stage_mda(*, out: Path, spec: screen.Spec, arm: screen.Arm, folds: Sequence[Fold], n_jobs: int, held_cap: int) -> None:
    if (out / "final_selection_summary.parquet").exists():
        raise FileExistsError("MDA/ladder output already sealed")
    candidate = tuple(json.loads((out / "subspace_contract.json").read_text())["candidate_fields"])
    values = pd.read_parquet(out / "redundancy_sample.parquet", columns=list(candidate))
    detail, mda = _mda(folds=folds, fields=candidate, spec=spec, values=values, held_cap=held_cap, n_jobs=n_jobs)
    representative = {member: row.representative for row in mda.itertuples(index=False) for member in row.members}
    ranked = pd.read_parquet(out / "subspace_summary.parquet")
    ranked = ranked.loc[ranked.feature.isin(candidate)].copy()
    ranked["mda_representative"] = ranked.feature.map(representative)
    ranked = ranked.merge(mda.loc[:, ["representative", "mda_delta_sstable_median", "mda_positive_folds"]], left_on="mda_representative", right_on="representative", how="left", validate="many_to_one")
    ranked["final_selection_score"] = (
        .45 * _scale(ranked.prescreen_score) + .25 * _scale(ranked.subspace_rank_score)
        + .25 * _scale(ranked.mda_delta_sstable_median) + .05 * _scale(ranked.mda_positive_folds)
    )
    ranked = ranked.sort_values(["final_selection_score", "feature"], ascending=[False, True], kind="stable").reset_index(drop=True)
    ladder_rows: list[dict[str, Any]] = []
    fields = ranked.feature.astype(str).tolist()
    for size in (120, 90, 70, 50, 35, 25):
        subset = fields[:size]
        fold_metrics: list[dict[str, Any]] = []
        for index, fold in enumerate(folds):
            _score_frame, metrics, model, x_held, _held, _raw = _score(fold=fold, fields=subset, spec=spec, seed=SEED + 80_000 + 100 * size + index, n_jobs=n_jobs)
            fold_metrics.append(metrics)
            del model, x_held
        aggregate = pd.DataFrame(fold_metrics).mean(numeric_only=True).to_dict()
        ladder_rows.append({"feature_count": size, **aggregate})
        _progress(out, stage="ladder_complete", feature_count=size, sstable_meta=aggregate["sstable_meta"])
    ladder = pd.DataFrame(ladder_rows).sort_values(["sstable_meta", "conditional_mi_meta_policy_given_base", "feature_count"], ascending=[False, False, True], kind="stable")
    ladder["rank"] = np.arange(1, len(ladder) + 1, dtype=int)
    # Top three ladder contracts deliberately go to the independent later
    # MC1/portfolio gate; this selector never promotes a model alone.
    contracts = out / "contracts"; contracts.mkdir(parents=True, exist_ok=True)
    for row in ladder.head(3).itertuples(index=False):
        _write_contract(
            contracts / f"{arm.family}_f{int(row.feature_count)}.json", fields=fields,
            ranked=ranked, size=int(row.feature_count), sstable=float(row.sstable_meta), arm=arm,
        )
    detail.to_parquet(out / "group_mda_detail.parquet", index=False, compression="zstd")
    mda.to_parquet(out / "group_mda_summary.parquet", index=False, compression="zstd")
    ranked.to_parquet(out / "final_selection_summary.parquet", index=False, compression="zstd")
    ladder.to_parquet(out / "feature_ladder_metrics.parquet", index=False, compression="zstd")
    _once(out / "selection_decision.json", {
        "schema": SCHEMA, "selection": "top three MDA/ladder contracts proceed to independently prequential MC1 portfolio evaluation",
        "candidate_contracts": [str(path) for path in sorted(contracts.glob(f"{arm.family}_f*.json"))],
        "selected_sizes": [int(row.feature_count) for row in ladder.head(3).itertuples(index=False)],
        "stop_rule": "No smaller ladder contracts are generated after the predeclared 25-field floor; the downstream gate selects only among the recorded top three.",
    })
    _once(out / "correctness_report.json", {
        "full_target_free_universe_used": True,
        "cross_year_month_coverage_used": True,
        "all_feature_hygiene_is_predeclared_and_month_complete": True,
        "univariate_cmi_is_conditional_on_frozen_base_rank": True,
        "randomized_subspaces_use_strict_oof_folds": True,
        "group_mda_permutations_are_within_timestamp_only": True,
        "group_mda_never_refits_on_held_outcomes": True,
        "feature_ladder_is_scored_under_the_same_target_query_objective": True,
        "only_top_three_contracts_may_reach_the_later_mc1_gate": True,
        "no_policy_or_path_field_in_model_inputs": True,
        "train_labels_resolved_before_reserve": True,
        "no_mc1_admission_portfolio_live_or_exchange_mutation": True,
    })


def run(args: argparse.Namespace) -> Path:
    raw, applied_source_override = screen._apply_source_override(
        json.loads(args.config.read_text()), args.source_override.resolve() if args.source_override else None,
    )
    spec = screen.Spec(raw=raw, config_path=args.config)
    arm = _arm(spec, args.arm); months = _months(tuple(item.strip() for item in args.held_months.split(",") if item.strip()))
    base_root, roots, policy_path, path_root = _source(spec)
    all_months = _source_months(months, train_months=int(spec.folds["train_months"]), reserve_days=int(spec.folds["resolved_label_reserve_days"]))
    fields = _full_fields(roots, all_months)
    if args.stage == "prescreen":
        if args.out.exists():
            raise FileExistsError(args.out)
        args.out.mkdir(parents=True)
        preflight = screen._preflight(spec, (), all_months)
        hygiene = _hygiene(roots, all_months, fields)
        _once(args.out / "run_manifest.json", {
            "schema": SCHEMA,
            "scope": "offline strict-OOF P8u full-feature Meta selection; no MC1/admission/portfolio/live/exchange mutation",
            "base_contract": spec.raw["base_contract"], "arm": dataclasses.asdict(arm),
            "held_months": [f"{month:%Y-%m}" for month in months], "cross_year_source_months": [f"{month:%Y-%m}" for month in all_months],
            "source": spec.source, "source_hashes": {"base": _sha(base_root), "policy": _sha(policy_path), "path": _sha(path_root)},
            "source_override": str(args.source_override) if args.source_override else None,
            "source_override_sha256": _sha(args.source_override) if args.source_override else None,
            "source_override_payload": applied_source_override,
            "full_feature_count_before_hygiene": len(fields), "feature_selection_only": True,
            "base_top_fraction_for_cmi": float(args.top_base_fraction),
            "causality": {"features": "target-free exact-identity panels", "train": "only resolved labels before 28-day reserve", "held": "scores never use outcomes", "base": "stored P8u Base target-free score/rank"},
        })
        preflight.to_parquet(args.out / "source_coverage_audit.parquet", index=False, compression="zstd")
        hygiene.to_parquet(args.out / "hygiene.parquet", index=False, compression="zstd")
        eligible = tuple(hygiene.loc[hygiene.hygiene_keep, "feature"].astype(str))
        if len(eligible) < 120:
            raise AssertionError(f"only {len(eligible)} full features pass strict hygiene")
        folds = _contexts(spec, arm, months)
        _stage_prescreen(
            out=args.out, spec=spec, arm=arm, folds=folds, fields=eligible,
            block_size=args.screen_block_size, veto_rows=args.veto_rows,
            top_base_fraction=float(args.top_base_fraction),
        )
        _once(args.out / "prescreen_correctness_report.json", {
            "full_target_free_universe_used": True, "cross_year_month_coverage_used": True,
            "no_policy_or_path_field_in_model_inputs": True, "train_labels_resolved_before_reserve": True,
            "strict_prequential_base_residual_anchor": True, "held_outcomes_only_after_target_free_identity": True,
            "no_mc1_admission_portfolio_live_or_exchange_mutation": True,
        })
    else:
        if not args.out.exists() or not (args.out / "run_manifest.json").exists():
            raise FileNotFoundError("resume stage requires a sealed prescreen root")
        folds = _contexts(spec, arm, months)
        if args.stage == "subspace":
            _stage_subspace(out=args.out, spec=spec, folds=folds, probes=args.probes, width=args.subspace_width, n_jobs=args.n_jobs)
        elif args.stage == "mda":
            _stage_mda(out=args.out, spec=spec, arm=arm, folds=folds, n_jobs=args.n_jobs, held_cap=args.mda_held_cap)
        else:  # pragma: no cover
            raise ValueError(args.stage)
    return args.out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stage", choices=("prescreen", "subspace", "mda"), required=True)
    parser.add_argument("--held-months", default=",".join(DEFAULT_HELD_MONTHS))
    parser.add_argument("--arm", default=None, help="one declared Under/Over/Magnitude/State target; default preserves the historical Under arm")
    parser.add_argument("--top-base-fraction", type=float, default=1.0, help="Base top fraction used only for feature-evidence IC/CMI")
    parser.add_argument("--screen-block-size", type=int, default=80)
    parser.add_argument("--veto-rows", type=int, default=45_000)
    parser.add_argument("--probes", type=int, default=6)
    parser.add_argument("--subspace-width", type=int, default=60)
    parser.add_argument("--mda-held-cap", type=int, default=20_000)
    parser.add_argument("--n-jobs", type=int, default=3)
    parser.add_argument("--source-override", type=Path, help="immutable source-only binding receipt")
    args = parser.parse_args()
    if args.screen_block_size < 20 or args.veto_rows < 5_000 or args.probes < 4 or args.subspace_width < 20 or args.mda_held_cap < 5_000 or args.n_jobs < 1:
        raise ValueError("invalid bounded full-feature selection contract")
    print(run(args))


if __name__ == "__main__":
    main()
