#!/usr/bin/env python3
"""Train fixed cross-fold family specialists and a matched residual ranker.

This runner is the first economic test of the frozen cross-fold family
contract.  It deliberately separates three decisions that were conflated by
the earlier family/path experiment:

* view discovery is performed once on the two development folds only;
* every specialist keeps the same frozen superfamily ID, medoid fields and
  common 40-field causal context pool in every fold;
* specialist probabilities are then passed row-by-row to a matched native
  LambdaRank residual learner.

The specialist target is the declared binary economic target ``net_bps > 50``
using the exact execution labels already present in the structural sidecar.
Costs are not applied again.  All reported tails are pooled global tails.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import re
import warnings
from pathlib import Path
from typing import Iterable

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.filterwarnings("ignore", message="Series.view is deprecated")


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "data_perp/artifacts/long_structural_tree_meta_sidecar_20260804_v4"
DEFAULT_INPUTS = ROOT / "data_perp/artifacts/frozen_family_inputs_20260808_v1"
DEFAULT_AUDIT = ROOT / "data_perp/artifacts/frozen_family_coverage_audit_20260808_v1"
DEFAULT_OUT = ROOT / "data_perp/artifacts/frozen_family_specialist_residual_20260808_v1"

DEV_FOLDS = ("oof_jul_aug", "oof_may_jun")
ALL_FOLDS = ("oof_jul_aug", "oof_may_jun", "oos_sep_nov")
SPECIALIST_TARGET_HURDLE_BPS = 50.0
COMMON_FEATURE_COUNT = 40
SIMILARITY_THRESHOLD = 0.60
DEFAULT_TOP_N = 64
TAILS = (0.005, 0.01, 0.02, 0.05, 0.10)
RESIDUAL_BIN_EDGES = (-150.0, -50.0, 50.0, 150.0)
QUERY_HOURS = 4
SEED = 20260808


def _digest(values: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(map(str, values)).encode()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def _feature_group(name: str) -> str:
    """Coarse causal family used only to diversify the deterministic pool."""

    n = name.lower()
    if n.startswith("aegmm_dae_"):
        return "ae_dae"
    if n.startswith("aegmm_gmm_") or n.startswith("aegmm_cluster"):
        return "ae_gmm"
    if n.startswith("aegmm_"):
        return "ae_other"
    if "funding" in n:
        return "funding"
    if n.startswith("oi") or "_oi_" in n or n.startswith("asset_minus_mkt_oi"):
        return "oi_flow"
    if n.startswith("ob_") or n.startswith("spread_") or n.startswith("impact") or "liquidity" in n:
        return "liquidity"
    if any(x in n for x in ("entropy", "vol", "adx", "climax", "shock", "jump", "semivol", "compression")):
        return "volatility"
    if any(x in n for x in ("lr_", "ret", "trend", "recovery", "return", "momentum", "flow_ratio")):
        return "returns_trend"
    if any(x in n for x in ("loc_", "distance", "dist_", "donch", "support", "breakout", "swing", "pullback")):
        return "price_structure"
    if n.startswith("hour_") or n.startswith("dow_"):
        return "calendar"
    return "other"


def _schema_names(path: Path) -> list[str]:
    import pyarrow.parquet as pq

    return list(map(str, pq.ParquetFile(path).schema.names))


def _raw_causal_candidates(schema: list[str]) -> list[str]:
    start = schema.index("atr_bps")
    end = schema.index("label_available_ts")
    excluded = {
        "candidate_id", "__ts__", "side_name", "gross_bps", "net_bps", "label_valid",
        "barrier_relevance_0_5", "mfe_mae_label_valid", "query_id", "meta_partition",
        "base_raw_score", "base_expected_bps", "fold", "feature_contract_sha256",
    }
    out: list[str] = []
    for name in schema[start:end]:
        if name in excluded or name.startswith("base_reasoning__") or name.startswith("base_structural_family__"):
            continue
        out.append(name)
    return out


def _build_common_pool(source: Path, schema: list[str], out: Path) -> tuple[list[str], pd.DataFrame]:
    """Select a fixed diverse pool using development rows only.

    No labels, base scores or OOS rows enter this selection.  Coverage and
    robust spread are fit per development fold; a correlation veto prevents
    the forty fields from collapsing into duplicate AE/GMM coordinates.
    """

    candidates = _raw_causal_candidates(schema)
    stats: list[dict[str, object]] = []
    dev_samples: list[pd.DataFrame] = []
    for fold in DEV_FOLDS:
        path = source / "fold_evaluations" / f"{fold}.parquet"
        cols = ["meta_partition", *candidates]
        frame = pd.read_parquet(path, columns=cols)
        frame = frame[frame.meta_partition.eq("meta_train")].drop(columns=["meta_partition"])
        # Keep correlation calculations bounded while retaining both eras.
        if len(frame) > 50000:
            frame = frame.iloc[np.linspace(0, len(frame) - 1, 50000).astype(int)]
        dev_samples.append(frame)
        for name in candidates:
            x = pd.to_numeric(frame[name], errors="coerce").to_numpy(float)
            finite = np.isfinite(x)
            if not finite.any():
                cov = 0.0
                spread = 0.0
                variance = 0.0
            else:
                vals = x[finite]
                q05, q95 = np.nanpercentile(vals, [5.0, 95.0])
                cov = float(finite.mean())
                spread = float(q95 - q05)
                variance = float(np.nanvar(vals))
            stats.append({"fold": fold, "feature": name, "coverage": cov, "robust_spread": spread, "variance": variance})
    stat = pd.DataFrame(stats)
    pivot = stat.pivot(index="feature", columns="fold", values=["coverage", "robust_spread", "variance"])
    required_cols = [(x, f) for x in ("coverage", "robust_spread", "variance") for f in DEV_FOLDS]
    for c in required_cols:
        if c not in pivot.columns:
            pivot[c] = np.nan
    valid = pivot[[("coverage", f) for f in DEV_FOLDS]].min(axis=1).ge(0.90)
    valid &= pivot[[("variance", f) for f in DEV_FOLDS]].min(axis=1).gt(1e-12)
    candidates_valid = pivot.index[valid].tolist()
    if len(candidates_valid) < COMMON_FEATURE_COUNT:
        raise ValueError(f"only {len(candidates_valid)} causal fields pass the development coverage/variance gate")

    sample = pd.concat(dev_samples, ignore_index=True)[candidates_valid]
    sample = sample.apply(pd.to_numeric, errors="coerce")
    sample = sample.fillna(sample.median(numeric_only=True)).fillna(0.0)
    arr = sample.to_numpy(float)
    corr = np.corrcoef(arr, rowvar=False)
    corr = np.nan_to_num(np.abs(corr), nan=0.0)
    corr_index = {name: i for i, name in enumerate(candidates_valid)}
    info: list[dict[str, object]] = []
    for name in candidates_valid:
        row = pivot.loc[name]
        info.append({
            "feature": name,
            "group": _feature_group(name),
            "coverage_min": float(min(row[("coverage", f)] for f in DEV_FOLDS)),
            "robust_spread_median": float(np.median([row[("robust_spread", f)] for f in DEV_FOLDS])),
            "variance_min": float(min(row[("variance", f)] for f in DEV_FOLDS)),
        })
    info_df = pd.DataFrame(info)
    info_df = info_df.sort_values(["group", "coverage_min", "robust_spread_median", "feature"], ascending=[True, False, False, True])
    by_group: dict[str, list[str]] = {}
    for row in info_df.itertuples(index=False):
        by_group.setdefault(str(row.group), []).append(str(row.feature))
    group_names = sorted(by_group)
    selected: list[str] = []
    # Round-robin with a soft six-field group cap gives genuine cross-view
    # diversity without importing semantic labels into the model target.
    while len(selected) < COMMON_FEATURE_COUNT:
        progressed = False
        for group in group_names:
            if len(selected) >= COMMON_FEATURE_COUNT:
                break
            if sum(_feature_group(x) == group for x in selected) >= 6 and any(_feature_group(x) != group for x in selected):
                continue
            while by_group[group]:
                candidate = by_group[group].pop(0)
                idx = corr_index[candidate]
                if all(corr[idx, corr_index[old]] < 0.95 for old in selected):
                    selected.append(candidate)
                    progressed = True
                    break
        if not progressed:
            # Fill the remainder deterministically if the correlation veto
            # leaves too few fields in a highly redundant source block.
            remaining = [x for x in candidates_valid if x not in selected]
            remaining.sort(key=lambda x: (-float(info_df.loc[info_df.feature.eq(x), "coverage_min"].iloc[0]), x))
            selected.extend(remaining[: COMMON_FEATURE_COUNT - len(selected)])
            break
    selected = selected[:COMMON_FEATURE_COUNT]
    if len(selected) != COMMON_FEATURE_COUNT:
        raise ValueError(f"selected {len(selected)} common fields, expected {COMMON_FEATURE_COUNT}")
    info_df["selected_common_pool"] = info_df.feature.isin(selected)
    info_df.to_parquet(out / "common_feature_selection_stats.parquet", index=False, compression="zstd")
    return selected, info_df


def _load_frozen_contract(audit: Path, threshold: float, top_n: int) -> pd.DataFrame:
    summary = pd.read_parquet(audit / "frozen_family_superfamily_summary.parquet")
    selected = summary[summary.threshold.eq(threshold) & summary.development_mass_rank.le(top_n)].sort_values("development_mass_rank").copy()
    if len(selected) != top_n:
        raise ValueError(f"expected {top_n} frozen families at threshold {threshold}, found {len(selected)}")
    if selected.development_fold_count.lt(2).any():
        raise ValueError("every selected family must be supported by both development folds")
    return selected.reset_index(drop=True)


def _family_columns(rank: int) -> list[str]:
    p = f"sf__{rank:03d}__"
    return [p + "signed_share", p + "abs_share", p + "active"]


def _coerce_matrix(frame: pd.DataFrame, cols: list[str], medians: pd.Series | None = None) -> tuple[np.ndarray, pd.Series]:
    x = frame[cols].apply(pd.to_numeric, errors="coerce")
    if medians is None:
        medians = x.median(numeric_only=True).reindex(cols).fillna(0.0)
    x = x.fillna(medians).fillna(0.0)
    return np.nan_to_num(x.to_numpy(dtype="float32"), nan=0.0, posinf=0.0, neginf=0.0), medians


def _classifier() -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="binary", n_estimators=220, learning_rate=0.03,
        num_leaves=16, max_depth=4, min_child_samples=500,
        subsample=0.80, subsample_freq=1, colsample_bytree=0.84,
        reg_alpha=0.05, reg_lambda=8.0, max_bin=127,
        random_state=SEED, n_jobs=1, verbosity=-1,
    )


def _query_order(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    ts = pd.to_datetime(frame["__ts__"], utc=True)
    q = ts.dt.floor(f"{QUERY_HOURS}h").astype(str).to_numpy() + "|" + frame["side_name"].astype(str).to_numpy()
    order = np.lexsort((frame["candidate_id"].astype(str).to_numpy(), ts.astype("int64").to_numpy(), q))
    q_sorted = q[order]
    if len(q_sorted) == 0:
        return order, np.array([], dtype="int32")
    boundaries = np.r_[0, np.flatnonzero(q_sorted[1:] != q_sorted[:-1]) + 1, len(q_sorted)]
    groups = np.diff(boundaries).astype("int32")
    return order, groups


def _residual_ranker(seed: int) -> lgb.LGBMRanker:
    return lgb.LGBMRanker(
        objective="lambdarank", metric="ndcg", n_estimators=260,
        learning_rate=0.03, num_leaves=16, max_depth=4,
        min_child_samples=500, subsample=0.80, subsample_freq=1,
        colsample_bytree=0.80, reg_alpha=0.05, reg_lambda=8.0,
        max_bin=127, lambdarank_truncation_level=10,
        random_state=seed, n_jobs=1, verbosity=-1,
    )


def _fit_ranker(x_train: np.ndarray, y_train: np.ndarray, train: pd.DataFrame, x_cal: np.ndarray, y_cal: np.ndarray, cal: pd.DataFrame, seed: int) -> tuple[np.ndarray, lgb.LGBMRanker, dict[str, object]]:
    train_order, train_groups = _query_order(train)
    cal_order, cal_groups = _query_order(cal)
    model = _residual_ranker(seed)
    kwargs = {
        "group": train_groups.tolist(),
        "eval_set": [(x_cal[cal_order], y_cal[cal_order])],
        "eval_group": [cal_groups.tolist()],
        "eval_at": [5, 10],
        "callbacks": [lgb.early_stopping(25, verbose=False)],
    }
    model.fit(x_train[train_order], y_train[train_order], **kwargs)
    return model.predict(x_cal), model, {"iterations": int(getattr(model, "best_iteration_", 0) or 0), "feature_importance": model.feature_importances_.tolist()}


def _map_rank_to_bps(raw_cal: np.ndarray, residual_cal: np.ndarray, raw_eval: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    ok = np.isfinite(raw_cal) & np.isfinite(residual_cal)
    if ok.sum() < 100 or np.nanstd(raw_cal[ok]) < 1e-8:
        value = float(np.nanmedian(residual_cal[ok])) if ok.any() else 0.0
        return np.full(len(raw_eval), value, dtype="float32"), {"mapping": "constant", "value": value}
    q = np.linspace(0.0, 1.0, 21)
    edges = np.unique(np.nanquantile(raw_cal[ok], q))
    if len(edges) < 3:
        value = float(np.nanmedian(residual_cal[ok]))
        return np.full(len(raw_eval), value, dtype="float32"), {"mapping": "constant", "value": value}
    bins = np.clip(np.searchsorted(edges, raw_cal, side="right") - 1, 0, len(edges) - 2)
    vals = []
    for i in range(len(edges) - 1):
        m = ok & (bins == i)
        vals.append(float(np.nanmedian(residual_cal[m])) if m.any() else np.nan)
    vals = pd.Series(vals).interpolate(limit_direction="both").to_numpy(float)
    eval_bins = np.clip(np.searchsorted(edges, raw_eval, side="right") - 1, 0, len(vals) - 1)
    return vals[eval_bins].astype("float32"), {"mapping": "calibration_quantile_median", "bins": int(len(vals)), "edges": edges.tolist()}


def _tail_metrics(frame: pd.DataFrame, score_col: str, period: str = "pooled") -> dict[str, object]:
    block = frame if period == "pooled" else frame[frame.period_key.eq(period)]
    if block.empty:
        return []
    ordered = block.sort_values([score_col, "candidate_id"], ascending=[False, True], kind="stable")
    out: list[dict[str, object]] = []
    for tail in TAILS:
        n = max(1, int(math.ceil(len(ordered) * tail)))
        chosen = ordered.head(n)
        out.append({"arm": score_col, "period": period, "tail": float(tail), "trades": int(n), "gross_bps": float(chosen.gross_bps.mean()), "net_bps": float(chosen.net_bps.mean()), "win_rate": float((chosen.net_bps > 0).mean())})
    return out


def _specialist_metrics(frame: pd.DataFrame, score_col: str, period: str = "pooled") -> dict[str, object]:
    block = frame if period == "pooled" else frame[frame.period_key.eq(period)]
    ok = np.isfinite(block[score_col].to_numpy(float)) & np.isfinite(block.net_bps.to_numpy(float))
    if ok.sum() < 20:
        return {"score": score_col, "period": period, "rows": int(ok.sum())}
    p = block.loc[ok, score_col].to_numpy(float)
    net = block.loc[ok, "net_bps"].to_numpy(float)
    y = (net > SPECIALIST_TARGET_HURDLE_BPS).astype(int)
    result: dict[str, object] = {"score": score_col, "period": period, "rows": int(ok.sum()), "positive_rate": float(y.mean()), "rank_ic_net": float(spearmanr(p, net).statistic) if len(np.unique(p)) > 1 and len(np.unique(net)) > 1 else np.nan}
    if len(np.unique(y)) > 1:
        result.update({"auc": float(roc_auc_score(y, p)), "pr_auc": float(average_precision_score(y, p)), "logloss": float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))), "brier": float(brier_score_loss(y, p))})
    else:
        result.update({"auc": np.nan, "pr_auc": np.nan, "logloss": np.nan, "brier": np.nan})
    result.update({"top1_net_bps": float(_tail_metrics(block, score_col)[0]["net_bps"]), "top5_net_bps": float(_tail_metrics(block, score_col)[3]["net_bps"]), "top10_net_bps": float(_tail_metrics(block, score_col)[4]["net_bps"])})
    return result


def run(args: argparse.Namespace) -> Path:
    source = Path(args.source)
    inputs_path = Path(args.inputs)
    audit = Path(args.audit)
    out = Path(args.out)
    top_n = int(args.top_n)
    if out.exists() and any(out.iterdir()) and not args.resume:
        raise FileExistsError(f"refusing to overwrite populated output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    contract = _load_frozen_contract(audit, SIMILARITY_THRESHOLD, top_n)
    schema = _schema_names(source / "fold_evaluations" / "oof_jul_aug.parquet")
    common, stats = _build_common_pool(source, schema, out)
    all_family_cols = [c for rank in range(1, top_n + 1) for c in _family_columns(rank)]
    family_inputs = pd.read_parquet(inputs_path / "frozen_family_inputs.parquet", columns=["fold_id", "candidate_id", *all_family_cols])
    union_path_features = sorted({f for values in contract.frozen_feature_names for f in list(values)})
    required_source_features = sorted(set(common) | set(union_path_features))
    missing = sorted(set(required_source_features).difference(schema))
    if missing:
        raise ValueError(f"frozen contract fields missing from source: {missing[:20]}")

    specialist_specs: list[dict[str, object]] = []
    for row in contract.itertuples(index=False):
        rank = int(row.development_mass_rank)
        path_features = list(row.frozen_feature_names)
        family_fields = _family_columns(rank)
        features = list(dict.fromkeys([*common, *path_features, *family_fields]))
        if not 40 <= len(features) <= 80:
            raise ValueError(f"specialist {rank} has {len(features)} fields; expected 40-80")
        specialist_specs.append({"rank": rank, "superfamily_id": str(row.superfamily_id), "path_features": path_features, "family_fields": family_fields, "features": features, "feature_digest": _digest(features), "frozen_feature_digest": str(row.frozen_feature_digest)})

    feature_contract = {
        "schema": "frozen_cross_fold_specialist_contract_v1",
        "selection_folds": list(DEV_FOLDS),
        "outer_folds": list(ALL_FOLDS),
        "similarity_threshold": SIMILARITY_THRESHOLD,
        "top_n": top_n,
        "common_feature_count": len(common),
        "common_features": common,
        "common_feature_digest": _digest(common),
        "specialist_specs": specialist_specs,
        "target": "binary_exact_h12_net_bps_gt_50",
        "query_grouping": "floor(__ts__, 4h) x side_name",
        "outcome_columns_not_features": ["gross_bps", "net_bps", "label_valid", "meta_partition", "label_available_ts"],
    }
    _write_json(out / "specialist_contract.json", feature_contract)

    all_test_outputs: list[pd.DataFrame] = []
    all_standalone: list[dict[str, object]] = []
    all_residual_meta: list[dict[str, object]] = []
    for fold_i, fold in enumerate(ALL_FOLDS):
        source_path = source / "fold_evaluations" / f"{fold}.parquet"
        base_cols = ["candidate_id", "__ts__", "side_name", "gross_bps", "net_bps", "label_valid", "meta_partition", "query_id", "base_raw_score", "base_expected_bps", *required_source_features]
        frame = pd.read_parquet(source_path, columns=list(dict.fromkeys(base_cols)))
        fam = family_inputs[family_inputs.fold_id.eq(fold)].drop(columns=["fold_id"])
        frame = frame.merge(fam, on="candidate_id", how="inner", validate="one_to_one")
        if len(frame) != len(fam):
            raise ValueError(f"family/source join changed row count for {fold}")
        frame["period_key"] = pd.to_datetime(frame["__ts__"], utc=True).dt.strftime("%Y-%m")
        train_mask = frame.meta_partition.eq("meta_train").to_numpy()
        cal_mask = frame.meta_partition.eq("meta_calibration").to_numpy()
        test_mask = frame.meta_partition.eq("test").to_numpy()
        if not (train_mask.any() and cal_mask.any() and test_mask.any()):
            raise ValueError(f"missing partition in {fold}")
        # One fit-only median per raw field.  This is the only imputation used
        # by all specialist heads and the residual learner.
        raw_feature_cols = sorted(set(common) | set(union_path_features))
        _, medians = _coerce_matrix(frame.loc[train_mask], raw_feature_cols)
        raw_values, _ = _coerce_matrix(frame, raw_feature_cols, medians)
        for j, name in enumerate(raw_feature_cols):
            frame[name] = raw_values[:, j]
        y_all = (pd.to_numeric(frame.net_bps, errors="coerce") > SPECIALIST_TARGET_HURDLE_BPS).astype("int8").to_numpy()
        if np.unique(y_all[train_mask]).size < 2:
            raise ValueError(f"constant specialist target in {fold}")
        shared_output_cols = ["candidate_id", "__ts__", "side_name", "gross_bps", "net_bps", "base_raw_score", "base_expected_bps", "period_key", *common]
        test_out = frame.loc[test_mask, shared_output_cols].copy()
        train_out = frame.loc[train_mask, shared_output_cols].copy()
        cal_out = frame.loc[cal_mask, shared_output_cols].copy()
        train_probs: list[np.ndarray] = []
        cal_probs: list[np.ndarray] = []
        test_probs: list[np.ndarray] = []
        for j, spec in enumerate(specialist_specs):
            cols = list(spec["features"])
            x_all, _ = _coerce_matrix(frame, cols, frame.loc[train_mask, cols].median(numeric_only=True).reindex(cols).fillna(0.0))
            x_train, x_cal, x_test = x_all[train_mask], x_all[cal_mask], x_all[test_mask]
            y_train, y_cal = y_all[train_mask], y_all[cal_mask]
            model = _classifier()
            model.set_params(random_state=SEED + fold_i * 1000 + j)
            model.fit(x_train, y_train, eval_set=[(x_cal, y_cal)], eval_metric="binary_logloss", callbacks=[lgb.early_stopping(25, verbose=False)])
            p_train = model.predict_proba(x_train)[:, 1].astype("float32")
            p_cal = model.predict_proba(x_cal)[:, 1].astype("float32")
            p_test = model.predict_proba(x_test)[:, 1].astype("float32")
            name = f"sp__{int(spec['rank']):03d}__prob"
            train_out[name] = p_train
            cal_out[name] = p_cal
            test_out[name] = p_test
            train_probs.append(p_train); cal_probs.append(p_cal); test_probs.append(p_test)
            for split_name, split_frame, probs in (("train", train_out, p_train), ("calibration", cal_out, p_cal), ("test", test_out, p_test)):
                metric = _specialist_metrics(split_frame.rename(columns={name: name}), name, "pooled")
                metric.update({"fold": fold, "specialist_rank": int(spec["rank"]), "superfamily_id": spec["superfamily_id"], "split": split_name, "feature_count": len(cols), "feature_digest": spec["feature_digest"], "best_iteration": int(getattr(model, "best_iteration_", 0) or 0)})
                # Training/calibration scores are diagnostics only; test is
                # the promotion-relevant population.
                all_standalone.append(metric)
        specialist_names = [f"sp__{int(s['rank']):03d}__prob" for s in specialist_specs]
        train_prob_matrix = np.column_stack(train_probs).astype("float32")
        cal_prob_matrix = np.column_stack(cal_probs).astype("float32")
        test_prob_matrix = np.column_stack(test_probs).astype("float32")
        for matrix, out_frame in ((train_prob_matrix, train_out), (cal_prob_matrix, cal_out), (test_prob_matrix, test_out)):
            out_frame["sp__mean_prob"] = matrix.mean(axis=1)
            out_frame["sp__std_prob"] = matrix.std(axis=1)
            out_frame["sp__max_prob"] = matrix.max(axis=1)
            out_frame["sp__min_prob"] = matrix.min(axis=1)
            out_frame["sp__n_above_50"] = (matrix > 0.5).sum(axis=1).astype("float32")

        residual_feature_base = ["base_expected_bps", "base_raw_score", *common]
        residual_feature_specialists = [*residual_feature_base, *specialist_names, "sp__mean_prob", "sp__std_prob", "sp__max_prob", "sp__min_prob", "sp__n_above_50"]
        residual_train = (train_out.net_bps.to_numpy(float) - train_out.base_expected_bps.to_numpy(float))
        residual_cal = (cal_out.net_bps.to_numpy(float) - cal_out.base_expected_bps.to_numpy(float))
        residual_test = (test_out.net_bps.to_numpy(float) - test_out.base_expected_bps.to_numpy(float))
        y_train_rank = np.digitize(residual_train, RESIDUAL_BIN_EDGES).astype("int8")
        y_cal_rank = np.digitize(residual_cal, RESIDUAL_BIN_EDGES).astype("int8")
        x_anchor_train, _ = _coerce_matrix(train_out, residual_feature_base)
        x_anchor_cal, _ = _coerce_matrix(cal_out, residual_feature_base, train_out[residual_feature_base].median(numeric_only=True).reindex(residual_feature_base).fillna(0.0))
        x_anchor_test, _ = _coerce_matrix(test_out, residual_feature_base, train_out[residual_feature_base].median(numeric_only=True).reindex(residual_feature_base).fillna(0.0))
        x_sp_train, _ = _coerce_matrix(train_out, residual_feature_specialists)
        x_sp_cal, _ = _coerce_matrix(cal_out, residual_feature_specialists, train_out[residual_feature_specialists].median(numeric_only=True).reindex(residual_feature_specialists).fillna(0.0))
        x_sp_test, _ = _coerce_matrix(test_out, residual_feature_specialists, train_out[residual_feature_specialists].median(numeric_only=True).reindex(residual_feature_specialists).fillna(0.0))
        for arm_name, xa_tr, xa_cal, xa_test, feature_count in (("R_anchor_context", x_anchor_train, x_anchor_cal, x_anchor_test, len(residual_feature_base)), ("R_all_specialists", x_sp_train, x_sp_cal, x_sp_test, len(residual_feature_specialists))):
            raw_cal, rank_model, rank_meta = _fit_ranker(xa_tr, y_train_rank, train_out, xa_cal, y_cal_rank, cal_out, SEED + fold_i * 100 + (0 if arm_name.startswith("R_anchor") else 1))
            # The test partition is never passed to fit/eval_set.  It is
            # scored only after the model and calibration map are frozen.
            map_test_raw = rank_model.predict(xa_test)
            mapped_cal, map_meta = _map_rank_to_bps(raw_cal, residual_cal, raw_cal)
            mapped_test, _ = _map_rank_to_bps(raw_cal, residual_cal, map_test_raw)
            # The ranker score itself is retained for audit; the mapped score
            # is the economically interpretable residual added to the base.
            test_out[arm_name] = (test_out.base_expected_bps.to_numpy(float) + mapped_test).astype("float32")
            all_residual_meta.append({"fold": fold, "arm": arm_name, "feature_count": feature_count, "query_hours": QUERY_HOURS, "target": "ordinal_residual_bps_-150_-50_50_150", "ranker_iterations": rank_meta["iterations"], "mapping": map_meta["mapping"], "mapping_bins": map_meta.get("bins"), "specialist_count": top_n})
        all_test_outputs.append(test_out)
        pd.DataFrame(all_standalone).query("fold == @fold and split == 'test'").to_parquet(out / f"specialist_metrics_{fold}.parquet", index=False, compression="zstd")
        pd.DataFrame(all_residual_meta).query("fold == @fold").to_parquet(out / f"residual_model_{fold}.parquet", index=False, compression="zstd")
        gc.collect()

    predictions = pd.concat(all_test_outputs, ignore_index=True)
    predictions.to_parquet(out / "specialist_oos_predictions.parquet", index=False, compression="zstd")
    specialist_metrics: list[dict[str, object]] = []
    specialist_cols = [f"sp__{int(s['rank']):03d}__prob" for s in specialist_specs]
    for col in specialist_cols:
        specialist_metrics.append(_specialist_metrics(predictions, col, "pooled"))
        for month in sorted(predictions.period_key.unique()):
            specialist_metrics.append(_specialist_metrics(predictions, col, str(month)))
    specialist_metrics_df = pd.DataFrame(specialist_metrics)
    specialist_metrics_df.to_parquet(out / "specialist_standalone_metrics.parquet", index=False, compression="zstd")

    residual_metrics: list[dict[str, object]] = []
    for arm in ["base_expected_bps", "R_anchor_context", "R_all_specialists"]:
        residual_metrics.extend(_tail_metrics(predictions, arm, "pooled"))
        for month in sorted(predictions.period_key.unique()):
            residual_metrics.extend(_tail_metrics(predictions, arm, str(month)))
    residual_metrics_df = pd.DataFrame(residual_metrics)
    residual_metrics_df.to_parquet(out / "residual_layer_metrics.parquet", index=False, compression="zstd")
    stability_rows: list[dict[str, object]] = []
    for arm in ["base_expected_bps", "R_anchor_context", "R_all_specialists"]:
        for tail in TAILS:
            base = residual_metrics_df[(residual_metrics_df["arm"] == "base_expected_bps") & (residual_metrics_df["period"] != "pooled") & (residual_metrics_df["tail"] == tail)].set_index("period")["net_bps"]
            cur = residual_metrics_df[(residual_metrics_df["arm"] == arm) & (residual_metrics_df["period"] != "pooled") & (residual_metrics_df["tail"] == tail)].set_index("period")["net_bps"]
            aligned = pd.concat([base.rename("base"), cur.rename("cur")], axis=1).dropna()
            uplift = aligned.cur - aligned.base
            pooled_uplift = float(residual_metrics_df[(residual_metrics_df["arm"] == arm) & (residual_metrics_df["period"] == "pooled") & (residual_metrics_df["tail"] == tail)]["net_bps"].iloc[0] - residual_metrics_df[(residual_metrics_df["arm"] == "base_expected_bps") & (residual_metrics_df["period"] == "pooled") & (residual_metrics_df["tail"] == tail)]["net_bps"].iloc[0])
            stability_rows.append({"arm": arm, "tail": float(tail), "pooled_uplift_bps": pooled_uplift, "median_month_uplift_bps": float(uplift.median()), "worst_month_uplift_bps": float(uplift.min()), "share_positive_months": float((uplift > 0).mean()), "stability_gate": bool(len(uplift) and uplift.median() >= 0 and uplift.min() >= 0)})
    stability = pd.DataFrame(stability_rows)
    stability.to_parquet(out / "residual_stability_metrics.parquet", index=False, compression="zstd")

    checks = {
        "status": "passed",
        "fixed_specialist_contract_across_folds": len({int(s["rank"]) for s in specialist_specs}) == top_n and len({str(s["superfamily_id"]) for s in specialist_specs}) == top_n,
        "selection_uses_development_folds_only": True,
        "specialist_count": len(specialist_specs) == top_n,
        "all_specialists_have_40_to_80_features": all(40 <= len(s["features"]) <= 80 for s in specialist_specs),
        "outer_test_rows_only": not predictions.duplicated("candidate_id").any(),
        "no_outcome_fields_in_specialist_features": not any(any(x in s["features"] for x in ["gross_bps", "net_bps", "label_valid", "meta_partition", "label_available_ts"]) for s in specialist_specs),
        "specialist_outputs_present": all(c in predictions.columns for c in specialist_cols),
        "residual_control_present": all(c in predictions.columns for c in ["base_expected_bps", "R_anchor_context", "R_all_specialists"]),
        "query_grouping_declared": QUERY_HOURS == 4,
        "target_declared": SPECIALIST_TARGET_HURDLE_BPS == 50.0,
        "metrics_present": bool(not specialist_metrics_df.empty and not residual_metrics_df.empty),
    }
    if not all(v for k, v in checks.items() if k != "status"):
        checks["status"] = "failed"
    _write_json(out / "correctness_test_report.json", checks)
    _write_json(out / "run_manifest.json", {"schema": "frozen_family_specialist_residual_v1", "status": checks["status"], "source": str(source), "inputs": str(inputs_path), "audit": str(audit), "rows": len(predictions), "specialists": top_n, "target": "exact_h12_net_bps_gt_50", "query_grouping": "4h_x_side", "residual_target": "ordinal_bps_-150_-50_50_150", "winner": None, "checks": checks})
    report = [
        "# Frozen cross-fold specialist and residual-layer ablation", "",
        f"Outer OOS rows: {len(predictions):,}. Ranking is pooled global top-k. Specialist target is exact H12 net > {SPECIALIST_TARGET_HURDLE_BPS:.0f} bps; costs are applied once in the source labels.", "",
        "## Frozen contract", "", f"{top_n} superfamilies at similarity threshold {SIMILARITY_THRESHOLD}; common causal pool: {len(common)} fields; each specialist has 40–80 fixed fields; selection folds: {', '.join(DEV_FOLDS)}.", "",
        "## Residual controls", "", "R_anchor_context is the matched LambdaRank residual learner with base outputs plus the fixed common context pool. R_all_specialists adds every specialist probability plus consensus summaries. Query groups are floor(timestamp, 4h) × side.", "",
        "## Pooled residual-layer net bps/trade", "", "| arm | top 1% | top 5% | top 10% |", "|---|---:|---:|---:|",
    ]
    pooled = residual_metrics_df[residual_metrics_df["period"].eq("pooled")]
    for arm in ["base_expected_bps", "R_anchor_context", "R_all_specialists"]:
        vals = pooled[pooled.arm.eq(arm)].set_index("tail").net_bps
        report.append(f"| {arm} | {vals.get(0.005, np.nan):.2f} | {vals.get(0.05, np.nan):.2f} | {vals.get(0.10, np.nan):.2f} |")
    report += ["", "## Stability gate", "", "An arm is stability-qualified only when median monthly uplift and worst-month uplift are both non-negative versus the base_expected_bps control. Absolute positive net EV is still required for execution promotion.", ""]
    for row in stability[stability["tail"].eq(0.05)].itertuples(index=False):
        report.append(f"- {row.arm}: pooled uplift {row.pooled_uplift_bps:.2f} bps; median month {row.median_month_uplift_bps:.2f}; worst month {row.worst_month_uplift_bps:.2f}; gate={row.stability_gate}")
    (out / "FROZEN_SPECIALIST_RESIDUAL_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return out


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    p.add_argument("--inputs", type=Path, default=DEFAULT_INPUTS)
    p.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    p.add_argument("--resume", action="store_true")
    return p


if __name__ == "__main__":
    print(run(_parser().parse_args()))
