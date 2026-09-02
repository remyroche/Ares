#!/usr/bin/env python3
"""Select full-causal feature contracts for the incumbent E/T meta heads.

The incumbent upstream is immutable throughout this producer:

``incumbent_upstream_bps = 0.50 * efficiency_bps + 0.50 * timing_bps``.

This is deliberately a *selection* producer, not a score, MC1, admission,
portfolio, inference, or exchange producer.  It starts with the target-free
full causal feature receipt (about 1,400 fields), validates hygiene over every
development month, screens fields in bounded parquet column blocks, and then
runs shallow randomized feature-subspace rankers.  The final output is a
small ladder of explicit immutable contracts for each already-selected target
family.  Downstream MC1 is reserved for the final contract comparison.

The screen uses only strict-prequential training labels.  Held feature values
are read from target-free receipts before their policy/path outcomes are joined
for diagnostics.  Invalid path/outcome rows are never converted to losses.

Research only.  It cannot alter the live trader or reach an exchange.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from lightgbm import LGBMRanker, early_stopping
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import mutual_info_score


ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT, ROOT / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import run_strict_r3_incumbent_meta_target_query_grid_v1 as meta_grid  # noqa: E402


SCHEMA = "strict_r3_incumbent_meta_fullfeatures_selection_v1"
SEED = 1729
IDENTITY = meta_grid.IDENTITY
DEFAULT_SOURCE_ROOT = meta_grid.DEFAULT_SOURCE_ROOT
DEFAULT_POLICY = meta_grid.DEFAULT_POLICY
DEFAULT_PATH_ROOT = meta_grid.DEFAULT_PATH_ROOT
DEFAULT_ARM_CONFIG = ROOT / "config/strict_r3_incumbent_meta_fullfeature_candidates_20260827_v1.json"
DEFAULT_CURRENT_ROOT = ROOT / "data_perp/artifacts/strict_r3_incumbent_meta_causal_features_20260827_v1"
DEFAULT_PREAUG_ROOT = ROOT / "data_perp/artifacts/strict_r3_incumbent_meta_causal_features_20260827_preaug_v1"
DEFAULT_HELD_MONTHS = "2025-10,2026-01,2026-04"
GEOMETRY_COLUMNS = (
    "enhanced_base_bps", "efficiency_bps", "timing_bps", "inc_base_rank_ts", "inc_query_count",
    "inc_query_std", "inc_query_range", "inc_top_gap", "inc_top2_gap", "inc_e_minus_t",
    "inc_e_t_mean", "inc_e_t_abs_gap", "base_component_std",
)


def _exclusive_json(path: Path, payload: object) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _progress(root: Path, **payload: object) -> None:
    with (root / "progress.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")


def _months(raw: str) -> tuple[pd.Timestamp, ...]:
    return meta_grid._parse_months(raw)


def _month_end(month: pd.Timestamp) -> pd.Timestamp:
    return meta_grid._month_end(month)


def _months_between(start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.Timestamp, ...]:
    return meta_grid._months_between(start, end)


def _family(field: str) -> str:
    """A conservative, name-only family for diversity accounting.

    This is *not* a semantic router.  It merely prevents a short list from
    being dominated by near-identical engineered aliases before the explicit
    numerical redundancy veto runs.
    """
    token = str(field).split("__", 1)[0]
    parts = token.split("_")
    return "_".join(parts[:2]) if len(parts) >= 2 else token


def _load_arms(path: Path) -> tuple[meta_grid.Arm, ...]:
    return meta_grid._custom_arms(path)


def _all_full_fields(roots: Sequence[Path], months: Sequence[pd.Timestamp]) -> tuple[str, ...]:
    common: set[str] | None = None
    for month in months:
        source = meta_grid._full_feature_path(roots, month)
        names = pq.ParquetFile(source).schema_arrow.names
        leaked = sorted(meta_grid.PROHIBITED.intersection(names))
        if leaked:
            raise AssertionError(f"{source}: target-free source leaks {leaked}")
        current = {
            name for name in names
            if name not in set(IDENTITY) | {"__ts__", "__symbol__"}
        }
        common = current if common is None else common.intersection(current)
    if not common:
        raise AssertionError("no common full-causal fields across development months")
    return tuple(sorted(common))


def _hygiene(roots: Sequence[Path], months: Sequence[pd.Timestamp], fields: Sequence[str]) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for month in months:
        source = meta_grid._full_feature_path(roots, month).parent / "feature_coverage.parquet"
        if not source.exists():
            raise FileNotFoundError(source)
        part = pd.read_parquet(source, columns=["feature", "finite_fraction", "n_unique"])
        part = part.loc[part.feature.isin(fields)].copy()
        part["month"] = f"{month:%Y-%m}"
        pieces.append(part)
    joined = pd.concat(pieces, ignore_index=True)
    expected = len(months)
    summary = joined.groupby("feature", sort=True).agg(
        observed_months=("month", "nunique"),
        min_coverage=("finite_fraction", "min"),
        median_coverage=("finite_fraction", "median"),
        min_unique=("n_unique", "min"),
    ).reset_index()
    summary["family"] = summary.feature.map(_family)
    summary["pass"] = (
        summary.observed_months.eq(expected)
        & summary.min_coverage.ge(.90)
        & summary.min_unique.gt(1)
    )
    return summary.sort_values(["pass", "min_coverage", "feature"], ascending=[False, False, True], kind="stable")


def _fold_base(
    *, source_root: Path, policy: pd.DataFrame, path_root: Path, held_months: Sequence[pd.Timestamp]
) -> tuple[meta_grid.Fold, ...]:
    """Load small incumbent/label frames once; no full feature matrix here."""
    return meta_grid._prepare_folds(
        source_root=source_root,
        policy=policy,
        path_root=path_root,
        held_months=held_months,
    )


def _read_feature_block(
    roots: Sequence[Path], frame: pd.DataFrame, fields: Sequence[str]
) -> pd.DataFrame:
    """Read a bounded target-free block and preserve exact supplied identities."""
    start = pd.Timestamp(frame.__decision_ts__.min())
    end = pd.Timestamp(frame.__decision_ts__.max()) + pd.Timedelta(nanoseconds=1)
    # Month ownership and target-free validation live in the shared reader.
    raw = meta_grid._read_full_features(roots, start, end, fields)
    expected = frame.loc[:, list(IDENTITY)]
    joined = expected.merge(raw, on=list(IDENTITY), how="left", validate="one_to_one")
    if len(joined) != len(frame) or joined.duplicated(IDENTITY).any():
        raise AssertionError("full feature block changed target-free identity population")
    return joined.loc[:, list(fields)]


def _bins(values: np.ndarray, bins: int = 10) -> np.ndarray:
    result = np.full(len(values), -1, dtype=np.int16)
    finite = np.isfinite(values)
    if int(finite.sum()) < 2:
        return result
    rank = rankdata(values[finite], method="average") / float(finite.sum())
    result[finite] = np.minimum(bins - 1, np.floor(rank * bins)).astype(np.int16)
    return result


def _conditional_mi(feature: np.ndarray, base: np.ndarray, outcome: np.ndarray) -> float:
    f, b, y = _bins(feature), _bins(base), _bins(outcome)
    valid = (f >= 0) & (b >= 0) & (y >= 0)
    if int(valid.sum()) < 200:
        return float("nan")
    total = float(valid.sum())
    value = 0.0
    for band in np.unique(b[valid]):
        rows = valid & (b == band)
        if int(rows.sum()) >= 30:
            value += float(rows.sum()) / total * float(mutual_info_score(f[rows], y[rows]))
    return float(value)


def _screen_direction(arm: meta_grid.Arm, labels: np.ndarray, residual: np.ndarray) -> np.ndarray:
    """Return the prospective final-score target used by the cheap screen."""
    if arm.family in {"magnitude", "state"}:
        return np.asarray(residual, dtype=float)
    final = np.asarray(labels, dtype=float)
    final[final < 0] = np.nan
    # The over head is inverted after model prediction; its desirable final
    # direction is therefore *absence* of unexpected severe adverse movement.
    return -final if arm.family == "over" else final


def _screen_block(
    *, folds: Sequence[meta_grid.Fold], roots: Sequence[Path], fields: Sequence[str], arm: meta_grid.Arm
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for fold in folds:
        train_labels, _train_residual, _info = meta_grid._target(fold.train, arm, train=True)
        held_anchor = meta_grid._fit_anchor(fold.train.loc[train_labels >= 0].copy())
        held_labels, held_residual, _held_info = meta_grid._target(
            fold.held, arm, train=False, held_anchor=held_anchor
        )
        valid_policy = (
            fold.held.policy_path_valid.fillna(False).to_numpy(bool)
            & np.isfinite(pd.to_numeric(fold.held.policy_net_bps, errors="coerce").to_numpy(float))
            & (held_labels >= 0)
        )
        target = _screen_direction(arm, held_labels, held_residual)
        valid = valid_policy & np.isfinite(target)
        block = _read_feature_block(roots, fold.held, fields)
        base = pd.to_numeric(fold.held.inc_base_rank_ts, errors="coerce").to_numpy(float)
        outcome = pd.to_numeric(fold.held.policy_net_bps, errors="coerce").to_numpy(float)
        for field in fields:
            values = pd.to_numeric(block[field], errors="coerce").to_numpy(float)
            rows = valid & np.isfinite(values) & np.isfinite(base) & np.isfinite(outcome)
            if int(rows.sum()) < 500:
                records.append({
                    "feature": field, "family": _family(field), "held_month": f"{fold.held_month:%Y-%m}",
                    "rows": int(rows.sum()), "spearman_abs": float("nan"), "cmi_policy_given_base": float("nan"),
                })
                continue
            rho = float(spearmanr(values[rows], target[rows]).statistic)
            records.append({
                "feature": field, "family": _family(field), "held_month": f"{fold.held_month:%Y-%m}",
                "rows": int(rows.sum()), "spearman": rho, "spearman_abs": abs(rho),
                "cmi_policy_given_base": _conditional_mi(values[rows], base[rows], outcome[rows]),
                "coverage": float(np.mean(np.isfinite(values))),
            })
    return records


def _screen_summary(records: pd.DataFrame) -> pd.DataFrame:
    summary = records.groupby(["feature", "family"], sort=True).agg(
        folds=("held_month", "nunique"),
        ic_median=("spearman", "median"),
        ic_abs_median=("spearman_abs", "median"),
        ic_abs_q25=("spearman_abs", lambda x: float(np.nanquantile(x, .25))),
        cmi_median=("cmi_policy_given_base", "median"),
        cmi_q25=("cmi_policy_given_base", lambda x: float(np.nanquantile(x, .25))),
        min_coverage=("coverage", "min"),
    ).reset_index()
    stable = (summary.ic_abs_q25.rank(pct=True) + summary.cmi_q25.rank(pct=True)) / 2.0
    central = (summary.ic_abs_median.rank(pct=True) + summary.cmi_median.rank(pct=True)) / 2.0
    summary["screen_score"] = .60 * stable + .40 * central
    return summary.sort_values(["screen_score", "feature"], ascending=[False, True], kind="stable")


def _sample_veto_values(
    *, folds: Sequence[meta_grid.Fold], roots: Sequence[Path], fields: Sequence[str], cap: int
) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for fold_index, fold in enumerate(folds):
        block = _read_feature_block(roots, fold.held, fields)
        # Deterministic candidate-ID sampling prevents a single dense month
        # from defining correlation representation.
        key = fold.held.candidate_id.astype(str) + f"|{SEED + fold_index}"
        order = pd.util.hash_pandas_object(key, index=False).to_numpy(np.uint64).argsort(kind="stable")
        take = order[: min(len(order), max(1, cap // max(1, len(folds))))]
        pieces.append(block.iloc[take].copy())
    return pd.concat(pieces, ignore_index=True)


def _redundancy_veto(summary: pd.DataFrame, values: pd.DataFrame, *, ceiling: float, keep_limit: int) -> pd.DataFrame:
    ordered = summary.feature.astype(str).tolist()
    selected: list[str] = []
    representative: dict[str, str] = {}
    numeric = values.loc[:, ordered].apply(pd.to_numeric, errors="coerce")
    # Compute every rank correlation once.  Re-sorting a 60k-row vector for
    # each candidate pair makes the otherwise small diversity veto dominate
    # the selection runtime.  The matrix is bounded (at most 240 fields) and
    # is exactly equivalent to the prior pairwise Spearman checks.
    correlation = numeric.rank(method="average", pct=True).corr(method="pearson").abs()
    for field in ordered:
        if len(selected) >= keep_limit:
            break
        reject = next((previous for previous in selected if np.isfinite(correlation.loc[field, previous]) and correlation.loc[field, previous] >= ceiling), None)
        if reject is None:
            selected.append(field)
        else:
            representative[field] = reject
    audit = summary.loc[:, ["feature", "family", "screen_score"]].copy()
    audit["kept_after_redundancy"] = audit.feature.isin(selected)
    audit["redundancy_representative"] = audit.feature.map(representative)
    return audit.sort_values(["kept_after_redundancy", "screen_score", "feature"], ascending=[False, False, True], kind="stable")


def _geometry_matrix(frame: pd.DataFrame, block: pd.DataFrame, fields: Sequence[str]) -> np.ndarray:
    geometry = frame.loc[:, list(GEOMETRY_COLUMNS)].apply(pd.to_numeric, errors="coerce")
    data = pd.concat([geometry.reset_index(drop=True), block.loc[:, list(fields)].reset_index(drop=True)], axis=1)
    return data.to_numpy(np.float32)


def _fit_probe(
    *, fold: meta_grid.Fold, roots: Sequence[Path], fields: Sequence[str], arm: meta_grid.Arm,
    seed: int, train_cap: int, n_jobs: int
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Fit one deliberately shallow strict-OOF subspace probe."""
    train_labels, _train_residual, _info = meta_grid._target(fold.train, arm, train=True)
    valid_train = train_labels >= 0
    train = fold.train.loc[valid_train].reset_index(drop=True)
    labels = train_labels[valid_train]
    if len(train) < 20_000 or len(np.unique(labels)) < 2:
        raise AssertionError(f"{arm.name} {fold.held_month:%Y-%m}: insufficient probe target support")
    held_anchor = meta_grid._fit_anchor(train)
    held_labels, held_residual, _held_info = meta_grid._target(fold.held, arm, train=False, held_anchor=held_anchor)
    train = meta_grid._sample_queries(train, train_cap, seed)
    # sampling changes row order; recreate valid labels by strict identity join
    original = fold.train.loc[valid_train, list(IDENTITY)].copy()
    label_frame = original.copy(); label_frame["label"] = labels
    train = train.merge(label_frame, on=list(IDENTITY), how="left", validate="one_to_one")
    labels = train.pop("label").to_numpy(np.int32)
    train_block = _read_feature_block(roots, train, fields)
    held_block = _read_feature_block(roots, fold.held, fields)
    train_x, held_x = meta_grid._impute(
        _geometry_matrix(train, train_block, fields), _geometry_matrix(fold.held, held_block, fields)
    )
    order, query_ids, _groups = meta_grid._ordered_query(train, meta_grid._query_ids(train, arm.query))
    x = train_x[order]; y = labels[order]
    unique = pd.Index(query_ids).drop_duplicates()
    cut = max(1, int(math.floor(.80 * len(unique))))
    fit_queries, valid_queries = set(unique[:cut]), set(unique[cut:])
    fit = np.asarray([item in fit_queries for item in query_ids], dtype=bool)
    tune = np.asarray([item in valid_queries for item in query_ids], dtype=bool)
    if not fit.any() or not tune.any():
        raise AssertionError("probe has insufficient causal early-stop queries")
    fit_groups = pd.Series(query_ids[fit]).groupby(pd.Series(query_ids[fit]), sort=False).size().astype(int).tolist()
    valid_groups = pd.Series(query_ids[tune]).groupby(pd.Series(query_ids[tune]), sort=False).size().astype(int).tolist()
    gain = meta_grid._gain(labels, arm.gain_schedule)
    model = LGBMRanker(
        objective="lambdarank", metric="ndcg", label_gain=gain,
        n_estimators=260, learning_rate=.065, max_depth=3, num_leaves=15,
        min_child_samples=350, min_split_gain=.003, colsample_bytree=.80,
        subsample=.80, reg_alpha=.05, reg_lambda=10.0, random_state=seed,
        n_jobs=n_jobs, verbosity=-1,
    )
    model.fit(x[fit], y[fit], group=fit_groups, eval_set=[(x[tune], y[tune])], eval_group=[valid_groups], callbacks=[early_stopping(25, verbose=False)])
    raw = np.asarray(model.predict(held_x), dtype=float)
    if arm.family == "over":
        raw *= -1.0
    rank = meta_grid._rank_desc(pd.DataFrame({
        "candidate_id": fold.held.candidate_id, "__decision_ts__": fold.held.__decision_ts__, "score": raw,
    }), "score")
    policy = pd.to_numeric(fold.held.policy_net_bps, errors="coerce").to_numpy(float)
    valid_policy = fold.held.policy_path_valid.fillna(False).to_numpy(bool) & np.isfinite(policy) & (held_labels >= 0)
    residual = np.asarray(held_residual, dtype=float)
    ic = float(spearmanr(rank[valid_policy], residual[valid_policy]).statistic) if int(valid_policy.sum()) >= 100 else float("nan")
    cmi = _conditional_mi(rank[valid_policy], fold.held.inc_base_rank_ts.to_numpy(float)[valid_policy], policy[valid_policy])
    sub1 = meta_grid._substitution(fold.held, score=rank, policy=policy, k=1)[2]
    sub2 = meta_grid._substitution(fold.held, score=rank, policy=policy, k=2)[2]
    importance = model.booster_.feature_importance(importance_type="gain")
    # Tail SHAP is bounded and diagnostic: it samples high-base-rank held rows
    # only; it does not enter a label or route calculation.
    tail = np.flatnonzero(fold.held.inc_base_rank_ts.to_numpy(float) >= .90)[:4_000]
    shap = np.zeros(len(GEOMETRY_COLUMNS) + len(fields), dtype=float)
    if len(tail):
        contrib = model.booster_.predict(held_x[tail], pred_contrib=True)
        shap = np.mean(np.abs(contrib[:, :-1]), axis=0)
    rows: list[dict[str, Any]] = []
    for offset, field in enumerate(fields):
        index = len(GEOMETRY_COLUMNS) + offset
        rows.append({
            "feature": field, "held_month": f"{fold.held_month:%Y-%m}", "probe_seed": int(seed),
            "gain_importance": float(importance[index]), "tail_abs_shap": float(shap[index]),
            "probe_ic": ic, "probe_cmi": cmi, "probe_sub_top1": sub1, "probe_sub_top2": sub2,
            "best_iteration": int(model.best_iteration_ or model.n_estimators),
        })
    metric = {
        "held_month": f"{fold.held_month:%Y-%m}", "probe_seed": int(seed), "fields": int(len(fields)),
        "ic": ic, "cmi": cmi, "substitution_top1": sub1, "substitution_top2": sub2,
        "best_iteration": int(model.best_iteration_ or model.n_estimators),
    }
    return metric, pd.DataFrame(rows)


def _subspace_probes(
    *, folds: Sequence[meta_grid.Fold], roots: Sequence[Path], fields: Sequence[str], arm: meta_grid.Arm,
    probes: int, subspace_size: int, train_cap: int, n_jobs: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(fields) < subspace_size:
        subspace_size = len(fields)
    metrics: list[dict[str, Any]] = []
    rows: list[pd.DataFrame] = []
    arm_token = int(hashlib.sha256(arm.name.encode()).hexdigest()[:8], 16) % 97
    for fold_index, fold in enumerate(folds):
        for probe in range(probes):
            rng = np.random.default_rng(SEED + 10_000 * fold_index + 100 * probe + arm_token)
            subset = sorted(rng.choice(np.asarray(fields, dtype=object), size=subspace_size, replace=False).astype(str).tolist())
            metric, evidence = _fit_probe(
                fold=fold, roots=roots, fields=subset, arm=arm,
                seed=SEED + 10_000 * fold_index + probe, train_cap=train_cap, n_jobs=n_jobs,
            )
            metric["probe"] = probe
            metrics.append(metric); rows.append(evidence)
    return pd.DataFrame(metrics), pd.concat(rows, ignore_index=True)


def _subspace_summary(evidence: pd.DataFrame, all_fields: Sequence[str]) -> pd.DataFrame:
    # Inclusion probability is constant by construction within each fold.  A
    # field's stable contribution is the median probe quality of models which
    # selected it, relative to models which did not, using only the same held
    # month.  It finds interacting field groups, unlike univariate CMI alone.
    keys = ["held_month", "probe_seed"]
    quality = evidence.loc[:, [*keys, "probe_ic", "probe_cmi", "probe_sub_top2"]].drop_duplicates()
    # Probe outcomes have materially different levels across held months.  A
    # stability selector must compare inclusion against other probes from the
    # *same* held month, never reward a feature merely because it happened to
    # occur in an easy month.
    quality["quality"] = (
        quality.groupby("held_month", sort=False).probe_ic.rank(pct=True)
        + quality.groupby("held_month", sort=False).probe_cmi.rank(pct=True)
        + quality.groupby("held_month", sort=False).probe_sub_top2.rank(pct=True)
    ) / 3.0
    work = evidence.merge(quality.loc[:, [*keys, "quality"]], on=keys, how="left", validate="many_to_one")
    present = work.groupby("feature", sort=True).agg(
        included_models=("quality", "size"),
        inclusion_quality=("quality", "mean"),
        gain_median=("gain_importance", "median"),
        tail_shap_median=("tail_abs_shap", "median"),
    )
    total = quality.groupby("held_month", sort=True).size().sum()
    # Estimate a fold-aware background from probes without a field.  This is
    # computed by reconstructing field memberships from the evidence table.
    memberships = work.loc[:, ["feature", *keys]].drop_duplicates()
    results: list[dict[str, Any]] = []
    for field in all_fields:
        included = memberships.loc[memberships.feature.eq(field), keys]
        all_keys = quality.loc[:, keys]
        absent = all_keys.merge(included.assign(_in=True), on=keys, how="left")
        absent = absent.loc[absent._in.isna(), keys]
        in_quality = quality.merge(included.assign(_in=True), on=keys, how="inner").quality
        out_quality = quality.merge(absent.assign(_out=True), on=keys, how="inner").quality
        entry = present.loc[field] if field in present.index else None
        results.append({
            "feature": field,
            "included_models": int(len(in_quality)), "excluded_models": int(len(out_quality)),
            "inclusion_uplift": float(in_quality.mean() - out_quality.mean()) if len(in_quality) and len(out_quality) else float("nan"),
            "inclusion_q25": float(np.quantile(in_quality, .25) - np.quantile(out_quality, .25)) if len(in_quality) >= 2 and len(out_quality) >= 2 else float("nan"),
            "gain_median": float(entry.gain_median) if entry is not None else 0.0,
            "tail_shap_median": float(entry.tail_shap_median) if entry is not None else 0.0,
            "model_fraction": float(len(in_quality) / max(1, total)),
        })
    return pd.DataFrame(results)


def _final_rank(screen: pd.DataFrame, subspace: pd.DataFrame) -> pd.DataFrame:
    result = screen.merge(subspace, on="feature", how="left", validate="one_to_one")
    result["subspace_score"] = (
        .55 * result.inclusion_uplift.fillna(-1e9).rank(pct=True)
        + .25 * result.inclusion_q25.fillna(-1e9).rank(pct=True)
        + .10 * result.gain_median.fillna(0.0).rank(pct=True)
        + .10 * result.tail_shap_median.fillna(0.0).rank(pct=True)
    )
    result["final_selection_score"] = .55 * result.screen_score.rank(pct=True) + .45 * result.subspace_score
    return result.sort_values(["final_selection_score", "feature"], ascending=[False, True], kind="stable")


def _contracts(
    *, out: Path, arm: meta_grid.Arm, ranked: pd.DataFrame, redundancy: pd.DataFrame
) -> dict[str, Any]:
    allowed = redundancy.loc[redundancy.kept_after_redundancy, "feature"].astype(str).tolist()
    pool = ranked.loc[ranked.feature.isin(allowed)].copy()
    if len(pool) < 35:
        raise AssertionError(f"{arm.name}: redundancy-veto pool has only {len(pool)} fields")
    directory = out / "contracts" / arm.family
    directory.mkdir(parents=True, exist_ok=True)
    written: dict[str, Any] = {}
    for size in (120, 90, 70, 50, 35):
        if len(pool) < size:
            continue
        fields = pool.head(size).feature.astype(str).tolist()
        payload = {
            "schema": "strict_r3_incumbent_meta_fullfeature_contract_v1",
            "scope": "offline selected full causal fields for strict-prequential incumbent meta score; no policy/outcome field is an inference input",
            "incumbent_upstream": "0.50 * efficiency_bps + 0.50 * timing_bps",
            "arm": arm.name, "family": arm.family, "feature_count": size,
            "features": fields,
            "selection": "hygiene + strict-OOF conditional IC/CMI + redundancy veto + randomized shallow subspace gain/tail-SHAP stability",
            "feature_sha256": hashlib.sha256("\n".join(fields).encode()).hexdigest(),
        }
        path = directory / f"{arm.family}_f{size}.json"
        _exclusive_json(path, payload)
        written[f"f{size}"] = str(path)
    return written


def run(args: argparse.Namespace) -> None:
    out = args.out
    if out.exists():
        raise FileExistsError(f"{out}: immutable selection output already exists")
    roots = tuple(Path(item.strip()) for item in args.feature_roots.split(",") if item.strip())
    if len(roots) < 2:
        raise ValueError("--feature-roots must provide the predecessor and current immutable roots")
    held_months = _months(args.held_months)
    policy = meta_grid._read_policy(args.policy)
    folds = _fold_base(source_root=args.source_root, policy=policy, path_root=args.path_root, held_months=held_months)
    all_months = tuple(sorted({month for fold in folds for month in _months_between(fold.train.__decision_ts__.min(), _month_end(fold.held_month))}))
    fields = _all_full_fields(roots, all_months)
    hygiene = _hygiene(roots, all_months, fields)
    eligible = hygiene.loc[hygiene["pass"], "feature"].astype(str).tolist()
    if len(eligible) < 300:
        raise AssertionError(f"only {len(eligible)} full features pass strict all-month hygiene")
    arms = _load_arms(args.arm_config)
    out.mkdir(parents=True)
    _exclusive_json(out / "run_manifest.json", {
        "schema": SCHEMA,
        "scope": "offline strict-prequential full-causal incumbent meta feature selection; no score/admission/portfolio/inference/live/exchange mutation",
        "incumbent_upstream": "0.50 * efficiency_bps + 0.50 * timing_bps",
        "source_root": str(args.source_root), "feature_roots": [str(root) for root in roots],
        "feature_manifest_sha256": {str(root): _sha_file(root / "run_manifest.json") for root in roots},
        "policy": str(args.policy), "path_root": str(args.path_root),
        "held_months": [f"{month:%Y-%m}" for month in held_months],
        "full_hygiene_months": [f"{month:%Y-%m}" for month in all_months],
        "arms": [arm.name for arm in arms], "screen_block_size": int(args.screen_block_size),
        "probes_per_fold": int(args.probes), "subspace_size": int(args.subspace_size),
        "causality": {
            "candidates_features": "immutable target-free point-in-time panels; labels/outcomes only join after identity is fixed",
            "base": "stored canonical incumbent route; 50/50 E/T arithmetic checked by shared reader",
            "folds": "four complete train months with a 28-day resolved-label reserve; held diagnostics only after score calculation",
            "selection": "development folds span multiple 2025/2026 months; downstream MC1 decides among written contracts",
        },
    })
    hygiene.to_parquet(out / "hygiene.parquet", index=False, compression="zstd")
    _progress(out, event="hygiene_complete", raw_fields=len(fields), eligible_fields=len(eligible), folds=len(folds))
    for arm_index, arm in enumerate(arms):
        arm_root = out / arm.family
        arm_root.mkdir(parents=True)
        screen_records: list[dict[str, Any]] = []
        for begin in range(0, len(eligible), args.screen_block_size):
            block = eligible[begin: begin + args.screen_block_size]
            screen_records.extend(_screen_block(folds=folds, roots=roots, fields=block, arm=arm))
            _progress(out, event="screen_block_complete", arm=arm.name, begin=begin, fields=len(block))
        observations = pd.DataFrame(screen_records)
        summary = _screen_summary(observations)
        # Retain a deliberately over-complete candidate pool; numeric redundancy
        # and randomized probes, rather than a one-dimensional screen, decide
        # the eventual 35--70-field contracts.
        candidate = summary.head(min(240, len(summary))).feature.astype(str).tolist()
        veto_values = _sample_veto_values(folds=folds, roots=roots, fields=candidate, cap=args.veto_rows)
        veto = _redundancy_veto(summary.loc[summary.feature.isin(candidate)].copy(), veto_values, ceiling=args.redundancy_ceiling, keep_limit=180)
        screened = veto.loc[veto.kept_after_redundancy, "feature"].astype(str).tolist()[: min(120, int(veto.kept_after_redundancy.sum()))]
        if len(screened) < 35:
            raise AssertionError(f"{arm.name}: only {len(screened)} screened features")
        metrics, probe_evidence = _subspace_probes(
            folds=folds, roots=roots, fields=screened, arm=arm,
            probes=args.probes, subspace_size=args.subspace_size, train_cap=args.probe_train_cap, n_jobs=args.n_jobs,
        )
        subspace = _subspace_summary(probe_evidence, screened)
        ranked = _final_rank(summary.loc[summary.feature.isin(screened)].copy(), subspace)
        contracts = _contracts(out=out, arm=arm, ranked=ranked, redundancy=veto)
        observations.to_parquet(arm_root / "screen_observations.parquet", index=False, compression="zstd")
        summary.to_parquet(arm_root / "screen_summary.parquet", index=False, compression="zstd")
        veto.to_parquet(arm_root / "redundancy_veto.parquet", index=False, compression="zstd")
        metrics.to_parquet(arm_root / "random_subspace_metrics.parquet", index=False, compression="zstd")
        probe_evidence.to_parquet(arm_root / "random_subspace_feature_evidence.parquet", index=False, compression="zstd")
        ranked.to_parquet(arm_root / "final_feature_ranking.parquet", index=False, compression="zstd")
        _exclusive_json(arm_root / "selection_summary.json", {
            "arm": arm.name, "family": arm.family, "screened_features": screened,
            "contract_paths": contracts, "top_features": ranked.head(20).feature.astype(str).tolist(),
        })
        _progress(out, event="arm_complete", arm=arm.name, family=arm.family, contracts=contracts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--feature-roots", default=f"{DEFAULT_PREAUG_ROOT},{DEFAULT_CURRENT_ROOT}")
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--path-root", type=Path, default=DEFAULT_PATH_ROOT)
    parser.add_argument("--arm-config", type=Path, default=DEFAULT_ARM_CONFIG)
    parser.add_argument("--held-months", default=DEFAULT_HELD_MONTHS)
    parser.add_argument("--screen-block-size", type=int, default=64)
    parser.add_argument("--probes", type=int, default=12)
    parser.add_argument("--subspace-size", type=int, default=50)
    parser.add_argument("--probe-train-cap", type=int, default=80_000)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--veto-rows", type=int, default=60_000)
    parser.add_argument("--redundancy-ceiling", type=float, default=.985)
    args = parser.parse_args()
    if args.screen_block_size < 8 or args.probes < 4 or args.subspace_size < 10 or args.n_jobs < 1:
        raise ValueError("screen block, probes, and subspace size are below the predeclared minimum")
    if not .90 <= args.redundancy_ceiling < 1.0:
        raise ValueError("redundancy ceiling must be in [.90, 1.0)")
    run(args)


if __name__ == "__main__":
    main()
