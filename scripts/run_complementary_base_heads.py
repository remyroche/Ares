#!/usr/bin/env python3
"""Sequential, downstream-selected complementary base-head ablation.

This is intentionally a funnel, not a factorial search.  Each candidate base
head first chooses an economically meaningful target and a causal query using
development folds, then freezes a side-local 60/80/100 feature contract and
receives a bounded 500-round LambdaRank HPO.  A new head is retained only if
its *OOF downstream residual* economics improve; the search stops after three
consecutive failures and never considers more than seven candidate heads.

The final 2024 monthly replay is held out from all target/query/feature/model
selection.  It is a confirmation replay, not a source of promotion decisions.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.complementary_base_heads import (  # noqa: E402
    AGREEMENT_FEATURES,
    agreement_features,
    causal_rank_norm,
    global_tail_metrics,
)
from extreme_price_movements.query_candidate_definitions import (  # noqa: E402
    QueryDefinition,
    assign_query_ids,
    base_head_query_definitions,
)
from extreme_price_movements.residual_lambdarank_hpo import (  # noqa: E402
    make_pruned_study,
    portability_score,
    suggest_base_lambdarank_params,
)


LEDGER = ROOT / "data_perp/artifacts/tp6_sl4_h12_two_year_ledger_20260806_v1/ledger.parquet"
ARCHIVE_SELECTION = ROOT / "data_perp/artifacts/stage_i_base_selection_20260803_v5"
DEFAULT_OUT = ROOT / "data_perp/artifacts/complementary_base_heads_20260808_v1"
SEED = 20260808
TAILS = (0.01, 0.02, 0.05)
DEV_FOLDS = (
    ("dev_2023_07_08", "2023-07-01", "2023-09-01"),
    ("dev_2023_09_10", "2023-09-01", "2023-11-01"),
    ("dev_2023_11_12", "2023-11-01", "2024-01-01"),
)
FINAL_MONTHS = tuple(f"2024-{month:02d}" for month in range(1, 9))
FEATURE_CAPS = (60, 80, 100)
MAX_BASE_TRAIN_ROWS = 18_000
MIN_TRAIN_ROWS = 500


@dataclass(frozen=True)
class TargetSpec:
    name: str
    description: str
    labels: Callable[[pd.DataFrame], np.ndarray]


def _r3(frame: pd.DataFrame) -> np.ndarray:
    return pd.to_numeric(frame["r3_class"], errors="coerce").fillna(1).clip(0, 2).to_numpy(np.int32)


def _soft_clear(frame: pd.DataFrame) -> np.ndarray:
    # Economic memberships are continuous in source; retain ordinal ordering
    # while respecting LightGBM LambdaRank's discrete gain semantics.
    raw = pd.to_numeric(frame["robust_clear_soft_b25_t50"], errors="coerce").fillna(0.5).clip(0.0, 1.0)
    return np.rint(raw.to_numpy(float) * 5.0).astype(np.int32)


def _net_binary(hurdle: float) -> Callable[[pd.DataFrame], np.ndarray]:
    return lambda frame: np.where(pd.to_numeric(frame["exact_net_bps"], errors="coerce").to_numpy(float) > hurdle, 5, 0).astype(np.int32)


def _ordinal_net(frame: pd.DataFrame) -> np.ndarray:
    values = pd.to_numeric(frame["exact_net_bps"], errors="coerce").fillna(-1000.0).to_numpy(float)
    return np.digitize(values, [-200.0, 0.0, 50.0, 150.0]).astype(np.int32)


def _tp_path(frame: pd.DataFrame) -> np.ndarray:
    event = pd.to_numeric(frame["t2_tp6_sl4_event"], errors="coerce").fillna(0).to_numpy(int)
    # 0 timeout, 1 adverse, 2 TP-first.  Preserve the economic timeout middle.
    return np.select((event == 1, event == 0, event == 2), (0, 2, 5), default=1).astype(np.int32)


TARGETS = {
    "r3_hard_b25": TargetSpec("r3_hard_b25", "R3 adverse / weak / robust-clear b25", _r3),
    "r3_soft_b25": TargetSpec("r3_soft_b25", "soft robust-clear b25, five ordinal levels", _soft_clear),
    "net_gt50": TargetSpec("net_gt50", "exact H12 net > +50 bps", _net_binary(50.0)),
    "net_gt100": TargetSpec("net_gt100", "exact H12 net > +100 bps", _net_binary(100.0)),
    "ordinal_net": TargetSpec("ordinal_net", "H12 net bands [-200, 0, +50, +150] bps", _ordinal_net),
    "tp_path": TargetSpec("tp_path", "TP6 / timeout / SL4 path ordering", _tp_path),
}

# The menus are intentionally role-specific.  They permit target HPO while
# preventing seven seed replicas of the same economic hypothesis.
HEAD_ROLES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("opportunity", ("r3_hard_b25", "r3_soft_b25", "tp_path")),
    ("cost_clear", ("net_gt50", "net_gt100", "ordinal_net")),
    ("soft_path", ("r3_soft_b25", "tp_path", "ordinal_net")),
    ("margin", ("net_gt100", "ordinal_net", "net_gt50")),
    ("robust_clear", ("r3_hard_b25", "r3_soft_b25", "net_gt50")),
    ("path_economics", ("tp_path", "ordinal_net", "net_gt100")),
    ("recall", ("r3_hard_b25", "net_gt50", "tp_path")),
)


def _utc(value: str | pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(value, tz="UTC") if not isinstance(value, pd.Timestamp) else value.tz_convert("UTC")


def _sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def _archive_feature_universe(side: str, columns: set[str]) -> list[str]:
    manifest = json.loads((ARCHIVE_SELECTION / side / "manifest.json").read_text())
    fields = [str(x) for x in manifest["input_feature_contract"] if str(x) in columns]
    if len(fields) < max(FEATURE_CAPS):
        raise ValueError(f"{side}: archival causal base universe has only {len(fields)} compatible fields")
    return list(dict.fromkeys(fields))


def _read_ledger(path: Path) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    import pyarrow.parquet as pq

    columns = set(pq.ParquetFile(path).schema.names)
    universes = {side: _archive_feature_universe(side, columns) for side in ("long", "short")}
    need = [
        "candidate_id", "__ts__", "side_name", "label_available_ts", "r3_class",
        "robust_clear_soft_b25_t50", "t2_tp6_sl4_event", "exact_net_bps", "exact_gross_bps",
        "prequential_base_expected_net_bps", "r3_p_adverse", "r3_p_weak", "r3_p_clear",
    ] + sorted(set(universes["long"]) | set(universes["short"]))
    frame = pd.read_parquet(path, columns=need)
    frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True, errors="raise")
    frame["label_available_ts"] = pd.to_datetime(frame["label_available_ts"], utc=True, errors="raise")
    frame["side_name"] = frame["side_name"].astype(str).str.lower()
    frame = frame[frame["side_name"].isin(["long", "short"])].copy()
    valid = np.isfinite(pd.to_numeric(frame.exact_net_bps, errors="coerce")) & np.isfinite(pd.to_numeric(frame.exact_gross_bps, errors="coerce"))
    frame = frame.loc[valid].sort_values(["__ts__", "candidate_id"], kind="stable").reset_index(drop=True)
    if frame.candidate_id.duplicated().any():
        raise ValueError("candidate IDs must be unique in the base-head ledger")
    return frame, universes


def _sample(frame: pd.DataFrame, cap: int, seed: int) -> pd.DataFrame:
    if len(frame) <= cap:
        return frame.copy()
    per_month = max(1, int(math.ceil(cap / max(frame.__ts__.dt.to_period("M").nunique(), 1))))
    sampled = []
    for _month, group in frame.groupby(frame.__ts__.dt.to_period("M"), sort=True):
        n = min(len(group), per_month)
        sampled.append(group.sample(n=n, random_state=seed + len(sampled)))
    out = pd.concat(sampled, ignore_index=True)
    return out.sample(n=min(cap, len(out)), random_state=seed).sort_values(["__ts__", "candidate_id"], kind="stable").copy()


def _query_order(frame: pd.DataFrame, definition: QueryDefinition) -> tuple[np.ndarray, np.ndarray]:
    query = assign_query_ids(frame, definition)
    raw = pd.DataFrame({"query": query.astype(str), "candidate_id": frame.candidate_id.astype(str), "row": np.arange(len(frame))})
    raw = raw.sort_values(["query", "candidate_id"], kind="stable")
    counts = raw.groupby("query", sort=False).size()
    raw = raw[raw["query"].isin(counts.index[counts.ge(2)])]
    group = raw.groupby("query", sort=False).size().to_numpy(np.int32)
    return raw.row.to_numpy(np.int64), group


def _matrix(train: pd.DataFrame, held: pd.DataFrame, fields: Sequence[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    x = train.loc[:, fields].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    med = x.median().fillna(0.0)
    return x.fillna(med).astype("float32"), held.loc[:, fields].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(med).astype("float32")


def _ranker_params(base: Mapping[str, Any]) -> dict[str, Any]:
    params = dict(base)
    params.pop("min_child_samples_fraction", None)
    params.pop("label_gain_name", None)
    params["verbosity"] = -1
    params["random_state"] = SEED
    params["n_jobs"] = 2
    params["bagging_freq"] = 1
    return params


def _fit_head(
    train: pd.DataFrame,
    held: pd.DataFrame,
    fields: Sequence[str],
    target: np.ndarray,
    query: QueryDefinition,
    params: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit a base ranker with a chronological internal early-stop slice."""
    if len(train) < MIN_TRAIN_ROWS:
        raise ValueError("too few base training rows")
    target = np.asarray(target, dtype=np.int32)
    if len(target) != len(train):
        raise ValueError("target must align with sampled training frame")
    if len(train) > MAX_BASE_TRAIN_ROWS:
        # Attach the label before deterministic month-stratified sampling so
        # the target cannot drift from its row after a cap is applied.
        sampled = train.copy()
        sampled["__target_for_fit__"] = target
        train = _sample(sampled, MAX_BASE_TRAIN_ROWS, SEED)
        target = train.pop("__target_for_fit__").to_numpy(np.int32)
    # The final 20% is a time-respecting early-stopping validation slice.
    split = max(int(len(train) * 0.80), MIN_TRAIN_ROWS)
    split = min(split, len(train) - 100)
    fit, valid = train.iloc[:split].copy(), train.iloc[split:].copy()
    yfit, yval = target[:split], target[split:]
    fit_order, fit_groups = _query_order(fit, query)
    val_order, val_groups = _query_order(valid, query)
    if len(fit_groups) == 0 or len(val_groups) == 0:
        raise ValueError(f"{query.name}: insufficient rankable query support")
    xfit, xheld = _matrix(fit, held, fields)
    xval = valid.loc[:, fields].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(xfit.median()).astype("float32")
    model = lgb.LGBMRanker(**_ranker_params(params))
    model.fit(
        xfit.iloc[fit_order], yfit[fit_order], group=fit_groups,
        eval_set=[(xval.iloc[val_order], yval[val_order])], eval_group=[val_groups],
        callbacks=[lgb.early_stopping(30, verbose=False)],
    )
    raw_fit = np.asarray(model.predict(xfit), dtype=np.float32)
    raw_held = np.asarray(model.predict(xheld), dtype=np.float32)
    raw_full = np.full(len(train), np.nan, dtype=np.float32)
    raw_full[:split] = raw_fit
    # Validation scores are required for a causal score-to-value map, but the
    # ranker did not fit their labels beyond early stopping.
    raw_full[split:] = np.asarray(model.predict(xval), dtype=np.float32)
    del model, xfit, xheld, xval
    gc.collect()
    return raw_full, raw_held, target


def _pava_map(train_score: np.ndarray, train_net: np.ndarray, held_score: np.ndarray) -> np.ndarray:
    valid = np.isfinite(train_score) & np.isfinite(train_net)
    if valid.sum() < 50:
        return np.full(len(held_score), float(np.nanmean(train_net)), dtype=np.float32)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=-3000.0, y_max=3000.0)
    iso.fit(train_score[valid], train_net[valid])
    return iso.predict(held_score).astype(np.float32)


def _fixed_screen_params() -> dict[str, Any]:
    return {
        "objective": "lambdarank", "metric": "ndcg", "n_estimators": 180,
        "learning_rate": 0.04, "max_depth": 4, "num_leaves": 15,
        "min_child_samples": 180, "feature_fraction": 0.80, "bagging_fraction": 0.80,
        "lambda_l1": 0.1, "lambda_l2": 8.0, "max_bin": 63,
        "lambdarank_truncation_level": 10, "label_gain": [0.0, .10, 1.0, 3.0, 7.0, 12.0],
    }


def _feature_selection(
    train: pd.DataFrame,
    universe: Sequence[str],
    target: np.ndarray,
    query: QueryDefinition,
    *,
    seed: int,
) -> tuple[list[str], pd.DataFrame]:
    """Coverage -> univariate/Relief proxy -> alias prune -> MDA selection.

    This mirrors the Stage-I selector's order and records the intermediate
    evidence.  It is deliberately frozen once per head/side, not reselected
    in each OOF month.
    """
    rows: list[dict[str, Any]] = []
    values = train.loc[:, universe].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    net = pd.to_numeric(train.exact_net_bps, errors="coerce").to_numpy(float)
    y = np.asarray(target, dtype=float)
    high = y >= np.quantile(y, .70)
    low = y <= np.quantile(y, .30)
    for field in universe:
        value = values[field].to_numpy(float)
        finite = np.isfinite(value)
        coverage = float(finite.mean())
        variance = float(np.nanvar(value)) if finite.any() else 0.0
        if finite.sum() > 40 and variance > 1e-12:
            target_ic = spearmanr(value[finite], y[finite]).statistic
            net_ic = spearmanr(value[finite], net[finite]).statistic
            scale = float(np.nanstd(value[finite])) or 1.0
            relief = abs(float(np.nanmean(value[high & finite]) - np.nanmean(value[low & finite]))) / scale
        else:
            target_ic = net_ic = relief = 0.0
        rows.append({"feature": field, "coverage": coverage, "variance": variance, "target_spearman": float(np.nan_to_num(target_ic)), "net_spearman": float(np.nan_to_num(net_ic)), "relief_proxy": float(np.nan_to_num(relief))})
    audit = pd.DataFrame(rows)
    eligible = audit.query("coverage >= 0.90 and variance > 1e-12").copy()
    if len(eligible) < max(FEATURE_CAPS):
        raise ValueError(f"feature selection has only {len(eligible)} eligible fields")
    eligible["pre_score"] = eligible.target_spearman.abs() + .35 * eligible.net_spearman.abs() + .05 * eligible.relief_proxy
    eligible = eligible.sort_values(["pre_score", "feature"], ascending=[False, True], kind="stable")
    # Hard aliases are removed before MDA, retaining the pre-screen winner.
    representatives: list[str] = []
    for field in eligible.feature.tolist():
        if len(representatives) >= 180:
            break
        vector = values[field].to_numpy(float)
        alias = False
        for selected in representatives:
            other = values[selected].to_numpy(float)
            valid = np.isfinite(vector) & np.isfinite(other)
            if valid.sum() >= 100 and abs(float(np.corrcoef(vector[valid], other[valid])[0, 1])) >= .95:
                alias = True
                break
        if not alias:
            representatives.append(field)
    # One time-spread calibration interval supplies a cheap but real MDA pass.
    cut = max(int(len(train) * .75), MIN_TRAIN_ROWS)
    fit, cal = train.iloc[:cut].copy(), train.iloc[cut:].copy()
    yfit = np.asarray(target[:cut], dtype=np.int32)
    ycal = np.asarray(target[cut:], dtype=np.int32)
    mda = {field: 0.0 for field in representatives}
    try:
        raw_fit, raw_cal, _ = _fit_head(fit, cal, representatives, yfit, query, _fixed_screen_params())
        base_ic = spearmanr(raw_cal, pd.to_numeric(cal.exact_net_bps, errors="coerce")).statistic
        rng = np.random.default_rng(seed)
        xfit, xcal = _matrix(fit, cal, representatives)
        # Refit once for MDA to retain the exact input transform, then permute
        # each field only in the held calibration matrix.
        order, groups = _query_order(fit, query)
        model = lgb.LGBMRanker(**_ranker_params(_fixed_screen_params())).fit(xfit.iloc[order], yfit[order], group=groups)
        for field in representatives:
            altered = xcal.copy()
            altered[field] = rng.permutation(altered[field].to_numpy())
            score = model.predict(altered)
            perm_ic = spearmanr(score, pd.to_numeric(cal.exact_net_bps, errors="coerce")).statistic
            mda[field] = float(np.nan_to_num(base_ic) - np.nan_to_num(perm_ic))
        del model, xfit, xcal, raw_fit, raw_cal
    except Exception as exc:  # audited rather than silently changing selectors
        audit["mda_error"] = str(exc)
    audit["alias_representative"] = audit.feature.isin(representatives)
    audit["mda_net_rank_ic_drop"] = audit.feature.map(mda).fillna(0.0)
    audit["selection_score"] = audit.pre_score if "pre_score" in audit else 0.0
    audit.loc[audit.alias_representative, "selection_score"] += .75 * audit.loc[audit.alias_representative, "mda_net_rank_ic_drop"]
    selected = audit[audit.alias_representative].sort_values(["selection_score", "feature"], ascending=[False, True], kind="stable").feature.tolist()
    if len(selected) < max(FEATURE_CAPS):
        selected += [x for x in eligible.feature.tolist() if x not in selected]
    audit["selected_rank"] = audit.feature.map({field: i + 1 for i, field in enumerate(selected)})
    return selected, audit


def _base_fold_predictions(
    frame: pd.DataFrame,
    fields_by_side: Mapping[str, Sequence[str]],
    target_spec: TargetSpec,
    query: QueryDefinition,
    params: Mapping[str, Any],
    start: pd.Timestamp,
    end: pd.Timestamp,
    head_id: str,
) -> pd.DataFrame:
    test = frame[(frame.__ts__ >= start) & (frame.__ts__ < end)].copy()
    pieces: list[pd.DataFrame] = []
    for side in ("long", "short"):
        train = frame[(frame["side_name"] == side) & (frame["__ts__"] < start) & (frame["label_available_ts"] < start)].copy()
        held = test[test["side_name"] == side].copy()
        if train.empty or held.empty:
            continue
        y = target_spec.labels(train)
        raw_train, raw_held, _ = _fit_head(train, held, fields_by_side[side], y, query, params)
        rank = causal_rank_norm(raw_train, raw_held)
        bps = _pava_map(raw_train, pd.to_numeric(train.exact_net_bps, errors="coerce").to_numpy(float), raw_held)
        out = held[["candidate_id", "__ts__", "side_name", "label_available_ts", "exact_net_bps", "exact_gross_bps", "prequential_base_expected_net_bps", "r3_p_adverse", "r3_p_weak", "r3_p_clear"]].copy()
        out[f"{head_id}__rank_norm"] = rank
        out[f"{head_id}__expected_net_bps"] = bps
        pieces.append(out)
    return pd.concat(pieces, ignore_index=True)


def _merge_heads(head_predictions: Sequence[pd.DataFrame], weights: Mapping[str, float]) -> pd.DataFrame:
    if not head_predictions:
        raise ValueError("at least one base head is required")
    base = head_predictions[0].copy()
    ids = ["candidate_id", "__ts__", "side_name", "label_available_ts", "exact_net_bps", "exact_gross_bps", "prequential_base_expected_net_bps", "r3_p_adverse", "r3_p_weak", "r3_p_clear"]
    for nxt in head_predictions[1:]:
        extra = [c for c in nxt.columns if c not in ids]
        base = base.merge(nxt[["candidate_id", *extra]], on="candidate_id", validate="one_to_one")
    ranks = sorted(c for c in base if c.endswith("__rank_norm"))
    bps = sorted(c for c in base if c.endswith("__expected_net_bps"))
    if len(ranks) != len(bps):
        raise ValueError("base head score/bps panel is incomplete")
    rank_weights = {column: weights.get(column.removesuffix("__rank_norm"), 1.0) for column in ranks}
    extra = agreement_features(base, ranks, weights=rank_weights)
    base = pd.concat([base, extra], axis=1)
    w = np.asarray([rank_weights[c] for c in ranks], dtype=float); w = w / w.sum()
    base["base_committee_rank_norm"] = (base[ranks].to_numpy(float) @ w).astype(np.float32)
    base["base_committee_expected_net_bps"] = (base[bps].to_numpy(float) @ w).astype(np.float32)
    return base


def _residual_params() -> dict[str, Any]:
    return {
        "objective": "lambdarank", "metric": "ndcg", "n_estimators": 180, "learning_rate": .035,
        "max_depth": 4, "num_leaves": 15, "min_child_samples": 220,
        "feature_fraction": .85, "bagging_fraction": .85, "lambda_l1": .1, "lambda_l2": 8.,
        "max_bin": 63, "lambdarank_truncation_level": 10, "label_gain": [0., .25, 1., 3., 7.],
    }


def _residual_score(train: pd.DataFrame, held: pd.DataFrame) -> np.ndarray:
    ranks = sorted(c for c in train if c.endswith("__rank_norm"))
    bps = sorted(c for c in train if c.endswith("__expected_net_bps"))
    fields = ["base_committee_rank_norm", "base_committee_expected_net_bps", *AGREEMENT_FEATURES, "prequential_base_expected_net_bps", "r3_p_adverse", "r3_p_weak", "r3_p_clear", *ranks, *bps]
    fields = [f for f in fields if f in train and f in held]
    residual = pd.to_numeric(train.exact_net_bps, errors="coerce").to_numpy(float) - train.base_committee_expected_net_bps.to_numpy(float)
    # Keep the residual target aligned when its ranker needs the same bounded
    # month-stratified training cap as a base head.
    if len(train) > MAX_BASE_TRAIN_ROWS:
        sampled = train.copy()
        sampled["__residual_for_fit__"] = residual
        train = _sample(sampled, MAX_BASE_TRAIN_ROWS, SEED + 91)
        residual = train.pop("__residual_for_fit__").to_numpy(float)
    grade = np.digitize(residual, [-150., -50., 50., 150.]).astype(np.int32)
    query = QueryDefinition("residual_q4h_side", "cycle", cycle_hours=4)
    raw_train, raw_held, _ = _fit_head(train, held, fields, grade, query, _residual_params())
    correction = _pava_map(raw_train, residual, raw_held)
    return (held.base_committee_expected_net_bps.to_numpy(float) + correction).astype(np.float32)


def _evaluate_downstream(fold_panels: Mapping[str, list[pd.DataFrame]], weights: Mapping[str, float]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    predictions: list[pd.DataFrame] = []
    preceding: list[pd.DataFrame] = []
    for fold_name in sorted(fold_panels):
        combined = _merge_heads(fold_panels[fold_name], weights)
        if preceding:
            train = pd.concat(preceding, ignore_index=True)
            combined["score"] = _residual_score(train, combined)
        else:
            combined["score"] = combined.base_committee_expected_net_bps
        combined["fold"] = fold_name
        predictions.append(combined)
        preceding.append(combined.drop(columns=["score"], errors="ignore"))
    pred = pd.concat(predictions, ignore_index=True)
    metrics_rows: list[dict[str, Any]] = []
    for fold, part in pred.groupby("fold", sort=True):
        metrics_rows.append({"fold": fold, **global_tail_metrics(part, score_column="score")})
    fold_metrics = pd.DataFrame(metrics_rows)
    aggregate = global_tail_metrics(pred, score_column="score")
    era_top5 = fold_metrics.top5_net_bps.tolist() if "top5_net_bps" in fold_metrics else []
    aggregate["portability_score_top5"] = portability_score(era_top5) if era_top5 else float("-inf")
    aggregate["worst_fold_top5_net_bps"] = float(min(era_top5)) if era_top5 else float("nan")
    return pred, fold_metrics, aggregate


def _query_prescreen(frame: pd.DataFrame, definition: QueryDefinition) -> dict[str, Any]:
    q = assign_query_ids(frame, definition)
    sizes = q.value_counts(sort=False)
    return {
        "query": definition.name, "rows": int(len(frame)), "groups": int(len(sizes)),
        "median_group_size": float(sizes.median()), "p10_group_size": float(sizes.quantile(.10)),
        "singleton_fraction": float((sizes == 1).sum() / max(len(sizes), 1)),
        "rankable_row_fraction": float(sizes[sizes >= 2].sum() / max(len(frame), 1)),
    }


def _development_panels(
    frame: pd.DataFrame,
    fields_by_side: Mapping[str, Sequence[str]],
    target: TargetSpec,
    query: QueryDefinition,
    params: Mapping[str, Any],
    head_id: str,
) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for name, start, end in DEV_FOLDS:
        out[name] = _base_fold_predictions(frame, fields_by_side, target, query, params, _utc(start), _utc(end), head_id)
    return out


def _downstream_with_candidate(
    accepted: Mapping[str, Mapping[str, pd.DataFrame]],
    candidate: Mapping[str, pd.DataFrame],
    weights: Mapping[str, float],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    panels: dict[str, list[pd.DataFrame]] = {}
    for name, _start, _end in DEV_FOLDS:
        panels[name] = [result[name] for result in accepted.values()] + [candidate[name]]
    return _evaluate_downstream(panels, weights)


def _screen_target_query(
    frame: pd.DataFrame,
    role: str,
    target_names: Sequence[str],
    accepted: Mapping[str, Mapping[str, pd.DataFrame]],
    weights: Mapping[str, float],
    screen_fields: Mapping[str, Sequence[str]],
    out: Path,
) -> tuple[TargetSpec, QueryDefinition, dict[str, pd.DataFrame], pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    prescreen = [_query_prescreen(frame[frame.__ts__ < _utc("2024-01-01")], q) for q in base_head_query_definitions()]
    pd.DataFrame(prescreen).to_parquet(out / f"{role}_query_prescreen.parquet", index=False)
    for target_name in target_names:
        target = TARGETS[target_name]
        for query in base_head_query_definitions():
            head_id = f"head_{role}"
            try:
                candidate = _development_panels(frame, screen_fields, target, query, _fixed_screen_params(), head_id)
                _, fold_metrics, aggregate = _downstream_with_candidate(accepted, candidate, {**weights, head_id: 1.0})
                rows.append({"role": role, "target": target_name, "query": query.name, "status": "complete", **aggregate, "fold_top5": json.dumps(dict(zip(fold_metrics["fold"], fold_metrics["top5_net_bps"])))})
            except Exception as exc:
                rows.append({"role": role, "target": target_name, "query": query.name, "status": "failed", "error": str(exc), "portability_score_top5": -np.inf})
    trials = pd.DataFrame(rows)
    trials.to_parquet(out / f"{role}_target_query_trials.parquet", index=False)
    completed = trials[trials.status.eq("complete")].copy()
    if completed.empty:
        raise RuntimeError(f"{role}: no target/query trial completed")
    winner = completed.sort_values(["portability_score_top5", "top5_net_bps", "top2_net_bps", "top1_net_bps"], ascending=False, kind="stable").iloc[0]
    target, query = TARGETS[str(winner["target"])], next(q for q in base_head_query_definitions() if q.name == winner["query"])
    candidate = _development_panels(frame, screen_fields, target, query, _fixed_screen_params(), f"head_{role}")
    return target, query, candidate, trials


def _select_caps(
    frame: pd.DataFrame,
    target: TargetSpec,
    query: QueryDefinition,
    accepted: Mapping[str, Mapping[str, pd.DataFrame]],
    weights: Mapping[str, float],
    role: str,
    out: Path,
) -> tuple[dict[str, list[str]], int, pd.DataFrame]:
    cutoff = _utc("2023-07-01")
    selected_by_side: dict[str, list[str]] = {}
    for side in ("long", "short"):
        train = frame[(frame["side_name"] == side) & (frame["__ts__"] < cutoff) & (frame["label_available_ts"] < cutoff)].copy()
        universe = _archive_feature_universe(side, set(frame.columns))
        selected, audit = _feature_selection(train, universe, target.labels(train), query, seed=SEED + (1 if side == "long" else 2))
        audit.to_parquet(out / f"{role}_{side}_feature_selection.parquet", index=False)
        selected_by_side[side] = selected
    rows: list[dict[str, Any]] = []
    for cap in FEATURE_CAPS:
        fields = {side: selected_by_side[side][:cap] for side in ("long", "short")}
        candidate = _development_panels(frame, fields, target, query, _fixed_screen_params(), f"head_{role}")
        _, fold_metrics, metrics = _downstream_with_candidate(accepted, candidate, {**weights, f"head_{role}": 1.0})
        rows.append({"cap": cap, **metrics, "fold_top5": json.dumps(dict(zip(fold_metrics["fold"], fold_metrics["top5_net_bps"])))})
    caps = pd.DataFrame(rows)
    caps.to_parquet(out / f"{role}_feature_cap_trials.parquet", index=False)
    winner = caps.sort_values(["portability_score_top5", "top5_net_bps", "top1_net_bps"], ascending=False, kind="stable").iloc[0]
    cap = int(winner.cap)
    return {side: values[:cap] for side, values in selected_by_side.items()}, cap, caps


def _hpo(
    frame: pd.DataFrame,
    target: TargetSpec,
    query: QueryDefinition,
    fields: Mapping[str, Sequence[str]],
    accepted: Mapping[str, Mapping[str, pd.DataFrame]],
    weights: Mapping[str, float],
    role: str,
    out: Path,
    trials: int,
) -> tuple[dict[str, Any], dict[str, pd.DataFrame], pd.DataFrame]:
    study = make_pruned_study(seed=SEED + len(accepted), n_startup_trials=min(4, max(2, trials // 3)), n_warmup_steps=1)

    def objective(trial: optuna.Trial) -> float:
        params = suggest_base_lambdarank_params(trial, retained_fraction=.05, median_candidates_per_query=8.0, max_boost_rounds=500)
        # Fold-specific conversion occurs only after rows are materialised.
        params["min_child_samples"] = max(80, int(params.pop("min_child_samples_fraction") * 8_000))
        candidate = _development_panels(frame, fields, target, query, params, f"head_{role}")
        _pred, fold_metrics, metrics = _downstream_with_candidate(accepted, candidate, {**weights, f"head_{role}": 1.0})
        for key, value in metrics.items():
            trial.set_user_attr(key, value)
        for index, value in enumerate(fold_metrics.top5_net_bps.tolist()):
            trial.report(float(value), step=index)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(metrics["portability_score_top5"])

    study.optimize(objective, n_trials=trials, show_progress_bar=False, gc_after_trial=True)
    records: list[dict[str, Any]] = []
    for trial in study.trials:
        records.append({"role": role, "trial": trial.number, "state": trial.state.name, "value": trial.value, **trial.params, **{f"metric_{k}": v for k, v in trial.user_attrs.items()}})
    audit = pd.DataFrame(records)
    audit.to_parquet(out / f"{role}_model_hpo_trials.parquet", index=False)
    complete = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE and trial.value is not None]
    if not complete:
        raise RuntimeError(f"{role}: all HPO trials failed/pruned")
    best = max(complete, key=lambda trial: float(trial.value))
    params = suggest_base_lambdarank_params(best, retained_fraction=.05, median_candidates_per_query=8.0, max_boost_rounds=500)
    params["min_child_samples"] = max(80, int(params.pop("min_child_samples_fraction") * 8_000))
    candidate = _development_panels(frame, fields, target, query, params, f"head_{role}")
    return params, candidate, audit


def _committee_weight(candidate: Mapping[str, pd.DataFrame], accepted: Mapping[str, Mapping[str, pd.DataFrame]], head_id: str) -> float:
    """Freeze a diversity weight from score correlation only, never outcomes."""
    if not accepted:
        return 1.0
    correlations: list[float] = []
    for fold in candidate:
        source = candidate[fold][["candidate_id", f"{head_id}__rank_norm"]]
        for prior_id, prior in accepted.items():
            merged = source.merge(prior[fold][["candidate_id", f"{prior_id}__rank_norm"]], on="candidate_id", validate="one_to_one")
            corr = merged.iloc[:, 1].corr(merged.iloc[:, 2], method="spearman")
            if np.isfinite(corr):
                correlations.append(abs(float(corr)))
    return float(np.clip(1.0 - (max(correlations) if correlations else 0.0), .10, 1.0))


def _final_replay(
    frame: pd.DataFrame,
    accepted_specs: Sequence[Mapping[str, Any]],
    accepted_dev: Mapping[str, Mapping[str, pd.DataFrame]],
    weights: Mapping[str, float],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    panels: dict[str, list[pd.DataFrame]] = {name: [result[name] for result in accepted_dev.values()] for name, _start, _end in DEV_FOLDS}
    monthly_rows: list[pd.DataFrame] = []
    for month in FINAL_MONTHS:
        start = _utc(f"{month}-01")
        end = start + pd.offsets.MonthBegin(1)
        current: list[pd.DataFrame] = []
        for spec in accepted_specs:
            current.append(_base_fold_predictions(frame, spec["fields"], TARGETS[spec["target"]], spec["query"], spec["params"], start, end, spec["head_id"]))
        panels[month] = current
        monthly_rows.append(_merge_heads(current, weights).assign(fold=month))
    # The residual sees only prior OOF development/months.  It never trains on
    # the current month's base predictions or outcomes.
    prior = [_merge_heads(panels[name], weights) for name, _start, _end in DEV_FOLDS]
    scored: list[pd.DataFrame] = []
    monthly_metric_rows: list[dict[str, Any]] = []
    for base in monthly_rows:
        held = base.copy()
        held["score"] = _residual_score(pd.concat(prior, ignore_index=True), held)
        scored.append(held)
        month = str(held["fold"].iloc[0])
        monthly_metric_rows.append({"month": month, **global_tail_metrics(held, score_column="score")})
        prior.append(held.drop(columns=["score"], errors="ignore"))
    predictions = pd.concat(scored, ignore_index=True)
    overall = pd.DataFrame([{**global_tail_metrics(predictions, score_column="score"), "months": len(FINAL_MONTHS)}])
    return predictions, pd.DataFrame(monthly_metric_rows), overall


def _finalize_one_month(
    out: Path, frame: pd.DataFrame, accepted_specs: Sequence[Mapping[str, Any]],
    accepted_dev: Mapping[str, Mapping[str, pd.DataFrame]], weights: Mapping[str, float], month: str,
) -> None:
    """Score one final confirmation month, preserving prior OOF-only meta fit."""
    if month not in FINAL_MONTHS:
        raise ValueError(f"final month must be one of {list(FINAL_MONTHS)}")
    start = _utc(f"{month}-01")
    end = start + pd.offsets.MonthBegin(1)
    existing_path = out / "final_oos_predictions.parquet"
    existing = pd.read_parquet(existing_path) if existing_path.is_file() else pd.DataFrame()
    if not existing.empty and str(existing["fold"].iloc[0]) == month and existing[existing["fold"].eq(month)].candidate_id.nunique() > 0:
        return
    current = [
        _base_fold_predictions(frame, spec["fields"], TARGETS[spec["target"]], spec["query"], spec["params"], start, end, spec["head_id"])
        for spec in accepted_specs
    ]
    held = _merge_heads(current, weights).assign(fold=month)
    prior = [_merge_heads([result[name] for result in accepted_dev.values()], weights) for name, _start, _end in DEV_FOLDS]
    if not existing.empty:
        prior.append(existing[existing["__ts__"] < start].drop(columns=["score"], errors="ignore"))
    held["score"] = _residual_score(pd.concat(prior, ignore_index=True), held)
    combined = pd.concat([existing[existing["fold"].ne(month)] if not existing.empty else existing, held], ignore_index=True)
    combined.to_parquet(existing_path, index=False, compression="zstd")
    monthly = []
    for value, group in combined.groupby("fold", sort=True):
        monthly.append({"month": str(value), **global_tail_metrics(group, score_column="score")})
    pd.DataFrame(monthly).to_parquet(out / "final_oos_monthly_metrics.parquet", index=False)
    pd.DataFrame([{**global_tail_metrics(combined, score_column="score"), "months": int(combined.fold.nunique())}]).to_parquet(out / "final_oos_global_metrics.parquet", index=False)


def _report(out: Path, manifest: Mapping[str, Any], all_metrics: pd.DataFrame) -> None:
    summary_path = out / "head_summary.parquet"
    summary = pd.read_parquet(summary_path) if summary_path.is_file() else pd.DataFrame(manifest.get("head_summary", []))
    monthly_path = out / "final_oos_monthly_metrics.parquet"
    monthly = pd.read_parquet(monthly_path) if monthly_path.is_file() else pd.DataFrame()
    lines = [
        "# Complementary base-head ablation", "",
        "## Contract", "",
        "- Source: exact TP6/SL4/H12 two-year ledger (2022-09 through 2024-08).",
        "- Development selection: Jul–Dec 2023 chronological OOF folds.",
        "- Final confirmation: Jan–Aug 2024 monthly expanding OOF; never used for target/query/feature/HPO selection.",
        "- Base target/query screen is selected on downstream residual top-1/top-2/top-5 global economics, with top-5 portability as the primary score.",
        "- Base HPO: bounded LambdaRank, 500 boosting-round ceiling and chronological early stopping after 30 rounds.",
        "- Agreement features use training-CDF rank normalisation and are passed to the residual only; no held-period rank or outcome enters them.",
        "",
        "## Search expansion", "",
        "- Each role used three economically distinct target definitions and the six predeclared side-local 1/2/4/6/8/12-hour query constructions: 18 target/query cells per candidate head.",
        "- Every target/query cell was scored by downstream residual top-1/top-2/top-5 economics across three chronological development folds; the primary selector is top-5 portability (median less dispersion and negative-worst-fold penalty).",
        "- The selected cell received side-local feature selection at 60/80/100 fields, following coverage, univariate/Relief proxy, Spearman alias pruning, and held-calibration permutation MDA evidence.",
        "- Each selected feature cap then received 12 bounded LambdaRank trials (500 rounds maximum, 30-round chronological early stopping, aggressive median pruning).",
        "",
        "## Head-by-head downstream decision", "",
        "| Head | Role | Target | Query | Cap | Accepted | Top-1 | Top-2 | Top-5 | Portability | Worst development fold |", "|---|---|---|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary.sort_values("head_id", kind="stable").to_dict("records") if not summary.empty else []:
        lines.append(
            f"| {row.get('head_id', '')} | {row.get('role', '')} | {row.get('target', '')} | {row.get('query', '')} | "
            f"{int(row.get('feature_cap', 0) or 0)} | {bool(row.get('accepted', False))} | "
            f"{float(row.get('top1_net_bps', np.nan)):.2f} | {float(row.get('top2_net_bps', np.nan)):.2f} | "
            f"{float(row.get('top5_net_bps', np.nan)):.2f} | {float(row.get('portability_score_top5', np.nan)):.2f} | "
            f"{float(row.get('worst_fold_top5_net_bps', np.nan)):.2f} |"
        )
    lines += ["", "## Frozen retained head configuration", ""]
    for spec in manifest.get("accepted_heads", []):
        contract_path = out / spec["head_id"] / "head_contract.json"
        contract = json.loads(contract_path.read_text()) if contract_path.is_file() else spec
        params = contract.get("params", {})
        hpo_path = out / spec["head_id"] / f"{contract.get('role', '')}_model_hpo_trials.parquet"
        hpo = pd.read_parquet(hpo_path) if hpo_path.is_file() else pd.DataFrame()
        completed = int(hpo.state.eq("COMPLETE").sum()) if "state" in hpo else 0
        pruned = int(hpo.state.eq("PRUNED").sum()) if "state" in hpo else 0
        lines += [
            f"### {spec['head_id']}", "",
            f"- Target: `{contract.get('target')}` — {contract.get('target_description')}",
            f"- Query: `{contract.get('query', {}).get('name')}`; feature cap: {contract.get('feature_cap')} per side; diversity weight: {float(contract.get('committee_weight', 1.0)):.3f}.",
            f"- HPO: {completed} completed / {pruned} median-pruned trials. Winner: `{json.dumps(params, sort_keys=True)}`.",
            f"- Feature hashes: `{_sha(contract.get('features', {}))}`; the exact feature lists are in its head contract and the two side feature-selection parquet audits.",
            "",
        ]
    lines += ["", "## Final OOS economics", ""]
    final_prediction_path = out / "final_oos_predictions.parquet"
    final_predictions = pd.read_parquet(final_prediction_path) if final_prediction_path.is_file() else pd.DataFrame()
    if not all_metrics.empty:
        row = all_metrics.iloc[0]
        lines += [f"- Global top-1%: {row.top1_net_bps:.2f} net bps/trade.", f"- Global top-2%: {row.top2_net_bps:.2f} net bps/trade.", f"- Global top-5%: {row.top5_net_bps:.2f} net bps/trade."]
    if not final_predictions.empty:
        lines += ["", "| Side | Top-1 net | Top-2 net | Top-5 net |", "|---|---:|---:|---:|"]
        for side, group in final_predictions.groupby("side_name", sort=True):
            values = global_tail_metrics(group, score_column="score")
            lines.append(f"| {side} | {values['top1_net_bps']:.2f} | {values['top2_net_bps']:.2f} | {values['top5_net_bps']:.2f} |")
    if not monthly.empty:
        lines += ["", "| Month | Top-1 net | Top-2 net | Top-5 net |", "|---|---:|---:|---:|"]
        for row in monthly.sort_values("month", kind="stable").to_dict("records"):
            lines.append(f"| {row['month']} | {float(row['top1_net_bps']):.2f} | {float(row['top2_net_bps']):.2f} | {float(row['top5_net_bps']):.2f} |")
    lines += ["", "All target/query trials, feature-selection/MDA audits, HPO trials, development panels, final predictions, and month metrics are retained next to this report."]
    (out / "COMPLEMENTARY_BASE_HEADS_REPORT.md").write_text("\n".join(lines) + "\n")


def _state_payload(
    accepted_specs: Sequence[Mapping[str, Any]], head_rows: Sequence[Mapping[str, Any]],
    weights: Mapping[str, float], baseline_score: float, no_improvement: int,
) -> dict[str, Any]:
    """Persist only deterministic, replayable head-boundary state."""
    return {
        "schema": "complementary_base_heads_state_v1",
        "accepted_heads": [
            {
                **{key: value for key, value in spec.items() if key != "query"},
                "query": spec["query"].manifest(),
            }
            for spec in accepted_specs
        ],
        "head_rows": list(head_rows), "weights": dict(weights),
        "baseline_score": float(baseline_score), "no_improvement": int(no_improvement),
    }


def _load_state(out: Path) -> tuple[dict[str, dict[str, pd.DataFrame]], list[dict[str, Any]], dict[str, float], list[dict[str, Any]], float, int]:
    state = json.loads((out / "state.json").read_text())
    if state.get("schema") != "complementary_base_heads_state_v1":
        raise ValueError("unsupported complementary-base-head state")
    accepted_specs: list[dict[str, Any]] = []
    accepted_dev: dict[str, dict[str, pd.DataFrame]] = {}
    for spec in state.get("accepted_heads", []):
        item = dict(spec)
        item["query"] = QueryDefinition(**dict(item["query"]))
        head_id = str(item["head_id"])
        dev_path = out / head_id / "development_oof_predictions.parquet"
        if not dev_path.is_file():
            raise ValueError(f"state references missing development OOF panel: {dev_path}")
        raw = pd.read_parquet(dev_path)
        accepted_dev[head_id] = {str(fold): group.drop(columns=["fold"]).reset_index(drop=True) for fold, group in raw.groupby("fold", sort=True)}
        accepted_specs.append(item)
    return accepted_dev, accepted_specs, {str(k): float(v) for k, v in state.get("weights", {}).items()}, list(state.get("head_rows", [])), float(state.get("baseline_score", -np.inf)), int(state.get("no_improvement", 0))


def run(out: Path, *, hpo_trials: int = 12, max_heads: int = 7,
        head_index: int | None = None, finalize: bool = False, final_month: str | None = None) -> Path:
    """Run the complete funnel or one resumable head/finalisation checkpoint."""
    state_path = out / "state.json"
    if out.exists() and any(out.iterdir()):
        if not state_path.is_file():
            raise FileExistsError(f"refusing to overwrite unrelated/non-resumable run: {out}")
        accepted_dev, accepted_specs, weights, head_rows, baseline_score, no_improvement = _load_state(out)
    else:
        out.mkdir(parents=True, exist_ok=False)
        accepted_dev, accepted_specs, weights, head_rows, baseline_score, no_improvement = {}, [], {}, [], -np.inf, 0
    frame, universes = _read_ledger(LEDGER)
    # Fast target/query screening uses deterministic archival causal prefixes;
    # target-specific 60/80/100 MDA selection occurs only after that decision.
    screen_fields = {side: values[:60] for side, values in universes.items()}
    if final_month is not None:
        if not accepted_specs:
            raise RuntimeError("cannot score final month without at least one accepted head")
        _finalize_one_month(out, frame, accepted_specs, accepted_dev, weights, final_month)
        return out
    if finalize:
        if not accepted_specs:
            raise RuntimeError("cannot finalise without at least one accepted head")
        if (out / "final_oos_predictions.parquet").is_file():
            predictions = pd.read_parquet(out / "final_oos_predictions.parquet")
            if set(predictions["fold"].astype(str)) != set(FINAL_MONTHS):
                raise ValueError("cannot publish final result until every final confirmation month is scored")
            monthly = pd.read_parquet(out / "final_oos_monthly_metrics.parquet")
            overall = pd.DataFrame([{**global_tail_metrics(predictions, score_column="score"), "months": len(FINAL_MONTHS)}])
        else:
            predictions, monthly, overall = _final_replay(frame, accepted_specs, accepted_dev, weights)
        predictions.to_parquet(out / "final_oos_predictions.parquet", index=False, compression="zstd")
        monthly.to_parquet(out / "final_oos_monthly_metrics.parquet", index=False)
        overall.to_parquet(out / "final_oos_global_metrics.parquet", index=False)
        pd.DataFrame(head_rows).to_parquet(out / "head_summary.parquet", index=False)
        manifest = {
            "schema": "complementary_base_heads_v1", "status": "complete", "ledger": str(LEDGER),
            "ledger_rows": int(len(frame)), "ledger_time_range": [str(frame.__ts__.min()), str(frame.__ts__.max())],
            "development_folds": [{"name": n, "start": s, "end_exclusive": e} for n, s, e in DEV_FOLDS],
            "final_confirmation_months": list(FINAL_MONTHS), "target_space": {k: v.description for k, v in TARGETS.items()},
            "query_space": [q.manifest() for q in base_head_query_definitions()], "feature_selection": "coverage >=90% + univariate/Relief proxy + Spearman alias representatives + held-calibration permutation MDA; frozen side/head-specific 60/80/100 caps",
            "base_hpo": "Optuna TPE + aggressive MedianPruner; 500 trees maximum; chronological early stopping 30", "residual": "ordinal policy-net correction with q4h x side LambdaRank; base-head ranks, expected-net maps, and agreement/disagreement fields", "agreement_features": list(AGREEMENT_FEATURES),
            "stop_rule": "stop after three consecutive candidate heads without >=1.0 bps development portability improvement; at most seven candidates", "accepted_heads": [{**{k: v for k, v in spec.items() if k not in {"query", "fields", "params"}}, "query": spec["query"].manifest(), "feature_hash_by_side": {side: _sha(values) for side, values in spec["fields"].items()}, "params": spec["params"]} for spec in accepted_specs], "head_summary": head_rows,
        }
        (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
        _report(out, manifest, overall)
        return out
    roles = list(enumerate(HEAD_ROLES[:max_heads], start=1))
    if head_index is not None:
        roles = [(index, item) for index, item in roles if index == int(head_index)]
        if not roles:
            raise ValueError(f"head_index must be in 1..{max_heads}")
    for index, (role, target_names) in roles:
        if any(str(row.get("head_id")) == f"head_{index:02d}_{role}" for row in head_rows):
            continue
        head_dir = out / f"head_{index:02d}_{role}"
        head_dir.mkdir()
        head_id = f"head_{index:02d}_{role}"
        try:
            target, query, _screen_candidate, _screen_trials = _screen_target_query(frame, role, target_names, accepted_dev, weights, screen_fields, head_dir)
            fields, cap, _cap_trials = _select_caps(frame, target, query, accepted_dev, weights, role, head_dir)
            params, candidate, _hpo_trials = _hpo(frame, target, query, fields, accepted_dev, weights, role, head_dir, hpo_trials)
            candidate = {fold: panel.rename(columns={f"head_{role}__rank_norm": f"{head_id}__rank_norm", f"head_{role}__expected_net_bps": f"{head_id}__expected_net_bps"}) for fold, panel in candidate.items()}
            provisional = {**weights, head_id: 1.0}
            _pred, fold_metrics, metrics = _downstream_with_candidate(accepted_dev, candidate, provisional)
            weight = _committee_weight(candidate, accepted_dev, head_id)
            provisional[head_id] = weight
            _pred, fold_metrics, metrics = _downstream_with_candidate(accepted_dev, candidate, provisional)
            improved = not accepted_specs or float(metrics["portability_score_top5"]) >= baseline_score + 1.0
            row = {"head_id": head_id, "role": role, "target": target.name, "query": query.name, "feature_cap": cap, "committee_weight": weight, "accepted": improved, "no_improvement_streak": no_improvement, **metrics}
            head_rows.append(row)
            pd.DataFrame([row]).to_parquet(head_dir / "head_result.parquet", index=False)
            (head_dir / "head_contract.json").write_text(json.dumps({"head_id": head_id, "role": role, "target": target.name, "target_description": target.description, "query": query.manifest(), "features": fields, "feature_cap": cap, "params": params, "committee_weight": weight, "development_metrics": metrics, "accepted": improved}, indent=2, default=str) + "\n")
            if improved:
                saved = pd.concat([panel.assign(fold=fold) for fold, panel in candidate.items()], ignore_index=True)
                saved.to_parquet(head_dir / "development_oof_predictions.parquet", index=False, compression="zstd")
                accepted_dev[head_id] = candidate
                weights[head_id] = weight
                accepted_specs.append({"head_id": head_id, "role": role, "target": target.name, "query": query, "fields": fields, "feature_cap": cap, "params": params, "committee_weight": weight, "development_metrics": metrics})
                baseline_score = float(metrics["portability_score_top5"])
                no_improvement = 0
            else:
                no_improvement += 1
        except Exception as exc:
            no_improvement += 1
            row = {"head_id": head_id, "role": role, "accepted": False, "error": str(exc), "no_improvement_streak": no_improvement}
            head_rows.append(row)
            pd.DataFrame([row]).to_parquet(head_dir / "head_result.parquet", index=False)
        state_path.write_text(json.dumps(_state_payload(accepted_specs, head_rows, weights, baseline_score, no_improvement), indent=2, default=str) + "\n")
        if no_improvement >= 3:
            break
    if not accepted_specs:
        raise RuntimeError("no base head completed")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--hpo-trials", type=int, default=12)
    parser.add_argument("--max-heads", type=int, default=7)
    parser.add_argument("--head-index", type=int)
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument("--final-month", choices=FINAL_MONTHS)
    args = parser.parse_args()
    print(run(args.out, hpo_trials=args.hpo_trials, max_heads=args.max_heads, head_index=args.head_index, finalize=args.finalize, final_month=args.final_month))
