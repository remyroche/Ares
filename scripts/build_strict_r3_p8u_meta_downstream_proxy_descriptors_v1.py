#!/usr/bin/env python3
"""Build strict-OOF descriptors for the learned P8u Meta downstream proxy.

This is deliberately *not* a hand-weighted HPO selector.  It takes already
persisted target-free Meta trial scores, verifies their identity against the
frozen Under F120 control, and only then opens the held policy/path labels to
produce inexpensive trial descriptors.  The resulting table is an input to a
separate learned PriorityProxy/GateProxy fit; it has no direct score, MC1,
admission, portfolio, or live authority.

The descriptors are fold-local and trial-level.  They preserve target, loss,
query, HPO, sample-weight, and feature-contract lineage so later grouped
validation can leave an entire target/loss/feature family out.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import reblend_strict_r3_p8u_meta_hpo_authority_v1 as authority
import run_strict_r3_p8u_meta_target_query_grid_v1 as screen


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "strict_r3_p8u_meta_downstream_proxy_descriptors_v1"
IDENTITY = screen.IDENTITY
HELD_MONTHS = tuple(screen._utc_month(value) for value in (
    "2026-01", "2026-02", "2026-03", "2026-04", "2026-05", "2026-06", "2026-07",
))
UTILITY_LOW, UTILITY_HIGH = -400.0, 400.0
BASE_HIGH, META_HIGH = 0.80, 0.80
CORRECTION_BINS = (0.00, 0.05, 0.15, 0.30, np.inf)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    members = sorted(path.rglob("*.parquet")) if path.is_dir() else [path]
    for member in members:
        digest.update(str(member).encode())
        with member.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    return digest.hexdigest()


def _once(path: Path, payload: object) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _score_path(root: Path, trial: str, month: pd.Timestamp) -> Path:
    path = root / "target_free_scores" / trial / f"month={month:%Y-%m}.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _control_path(root: Path, month: pd.Timestamp) -> Path:
    path = root / "target_free_scores" / "current" / f"month={month:%Y-%m}.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _read_candidate_score(path: Path) -> pd.DataFrame:
    screen._assert_target_free(path)
    required = [*IDENTITY, "base_score", "base_rank_ts", "meta_raw_score", "meta_rank_ts", "trial", "target_free"]
    columns = set(pd.read_parquet(path, columns=None).columns)
    missing = sorted(set(required).difference(columns))
    if missing:
        raise AssertionError(f"{path}: missing target-free score fields {missing}")
    score = pd.read_parquet(path, columns=required)
    score["__decision_ts__"] = pd.to_datetime(score["__decision_ts__"], utc=True, errors="raise")
    if score.duplicated(IDENTITY).any() or not score.side_name.eq("long").all() or not score.target_free.fillna(False).astype(bool).all():
        raise AssertionError(f"{path}: invalid target-free candidate score receipt")
    return score.sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)


def _read_control_score(path: Path) -> pd.DataFrame:
    screen._assert_target_free(path)
    required = [*IDENTITY, "base_rank42", "conditional_consensus_rank", "upstream", "final_score"]
    columns = set(pd.read_parquet(path, columns=None).columns)
    missing = sorted(set(required).difference(columns))
    if missing:
        raise AssertionError(f"{path}: missing frozen control score fields {missing}")
    score = pd.read_parquet(path, columns=required)
    score["__decision_ts__"] = pd.to_datetime(score["__decision_ts__"], utc=True, errors="raise")
    if score.duplicated(IDENTITY).any() or not score.side_name.eq("long").all():
        raise AssertionError(f"{path}: invalid frozen control target-free receipt")
    return score.sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)


def _validate_target_free_pair(candidate: pd.DataFrame, control: pd.DataFrame, *, trial: str, month: pd.Timestamp) -> None:
    if len(candidate) != len(control):
        raise AssertionError(f"{trial} {month:%Y-%m}: candidate/control target-free row count differs")
    left = candidate.loc[:, list(IDENTITY)].reset_index(drop=True)
    right = control.loc[:, list(IDENTITY)].reset_index(drop=True)
    if not left.equals(right):
        raise AssertionError(f"{trial} {month:%Y-%m}: candidate/control target-free identities differ")
    if not np.allclose(candidate.base_rank_ts.to_numpy(float), control.base_rank42.to_numpy(float), rtol=0.0, atol=1e-6):
        raise AssertionError(f"{trial} {month:%Y-%m}: candidate Base rank differs from frozen control")


def _rank_top(frame: pd.DataFrame, column: str, k: int) -> pd.DataFrame:
    return frame.sort_values(["__decision_ts__", column, "candidate_id"], ascending=[True, False, True], kind="stable").groupby("__decision_ts__", sort=False).head(k)


def _top_fraction(frame: pd.DataFrame, column: str, fraction: float = 0.20) -> pd.DataFrame:
    ordered = frame.sort_values(["__decision_ts__", column, "candidate_id"], ascending=[True, False, True], kind="stable").copy()
    ordered["__ordinal__"] = ordered.groupby("__decision_ts__", sort=False).cumcount()
    ordered["__size__"] = ordered.groupby("__decision_ts__", sort=False).candidate_id.transform("size")
    return ordered.loc[ordered.__ordinal__.lt(np.ceil(fraction * ordered.__size__))].copy()


def _timestamp_mean(frame: pd.DataFrame, value: str = "policy_net_bps") -> pd.Series:
    return frame.groupby("__decision_ts__", sort=False)[value].mean()


def _mean(series: pd.Series | np.ndarray) -> float:
    values = np.asarray(series, dtype=float)
    return float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")


def _q(series: pd.Series | np.ndarray, quantile: float) -> float:
    values = np.asarray(series, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.quantile(values, quantile)) if len(values) else float("nan")


def _set_delta_values(candidate: pd.DataFrame, control: pd.DataFrame, column: str, k: int) -> pd.DataFrame:
    c = _rank_top(candidate, column, k).loc[:, ["__decision_ts__", "candidate_id", "policy_net_bps"]]
    b = _rank_top(control, "control_combined_rank", k).loc[:, ["__decision_ts__", "candidate_id", "policy_net_bps"]]
    rows: list[dict[str, Any]] = []
    for timestamp, cgroup in c.groupby("__decision_ts__", sort=False):
        bgroup = b.loc[b.__decision_ts__.eq(timestamp)]
        cids, bids = set(cgroup.candidate_id), set(bgroup.candidate_id)
        only_c = cgroup.loc[~cgroup.candidate_id.isin(bids), "policy_net_bps"]
        only_b = bgroup.loc[~bgroup.candidate_id.isin(cids), "policy_net_bps"]
        rows.append({
            "__decision_ts__": timestamp,
            "candidate_only_n": int(len(only_c)), "control_only_n": int(len(only_b)),
            "candidate_only_ev": _mean(only_c), "control_only_ev": _mean(only_b),
            "candidate_minus_control_only_ev": _mean(only_c) - _mean(only_b) if len(only_c) and len(only_b) else float("nan"),
        })
    return pd.DataFrame(rows)


def _jaccard_by_timestamp(left: pd.DataFrame, right: pd.DataFrame, left_column: str, right_column: str, k: int) -> float:
    a = _rank_top(left, left_column, k).groupby("__decision_ts__", sort=False).candidate_id.agg(set)
    b = _rank_top(right, right_column, k).groupby("__decision_ts__", sort=False).candidate_id.agg(set)
    values = [len(a[t].intersection(b[t])) / max(1, len(a[t].union(b[t]))) for t in a.index.intersection(b.index)]
    return _mean(values)


def _conditional_mi(frame: pd.DataFrame) -> float:
    return float(screen._conditional_mi(
        frame.meta_rank_ts.to_numpy(float), frame.base_rank_ts.to_numpy(float), frame.policy_net_bps.to_numpy(float),
    ))


def _metric_row(
    *, candidate: pd.DataFrame, control: pd.DataFrame, labelled: pd.DataFrame, anchor: Any,
    trial: Mapping[str, Any], root_name: str, feature_contract: str,
    feature_count: int, held_month: pd.Timestamp,
) -> tuple[dict[str, Any], pd.DataFrame]:
    # Held outcomes are joined only here, after both source score receipts have
    # been read and exact identity/Base-rank checks have passed.
    outcome = labelled.loc[:, list(IDENTITY) + ["policy_path_valid", "policy_net_bps", "supportive_path_valid"]].copy()
    work = candidate.merge(outcome, on=list(IDENTITY), how="left", validate="one_to_one")
    control_work = control.merge(outcome, on=list(IDENTITY), how="left", validate="one_to_one")
    # Keep exactly the same complete-policy/path validity definition as the
    # OOF screen.  In particular, a path row without a valid decision-time
    # ATR is not silently promoted into the proxy training population.
    valid = screen._valid_label(labelled)
    work, control_work = work.loc[valid].copy(), control_work.loc[valid].copy()
    if len(work) < 1000 or len(work) != len(control_work):
        raise AssertionError(f"{trial['name']} {held_month:%Y-%m}: inadequate matched valid held policy support")
    work["policy_net_bps"] = pd.to_numeric(work.policy_net_bps, errors="raise").astype(float)
    control_work["policy_net_bps"] = pd.to_numeric(control_work.policy_net_bps, errors="raise").astype(float)
    work["utility_bps"] = work.policy_net_bps.clip(UTILITY_LOW, UTILITY_HIGH)
    control_work["utility_bps"] = control_work.policy_net_bps.clip(UTILITY_LOW, UTILITY_HIGH)
    work["control_combined_rank"] = control_work.upstream.to_numpy(float)
    control_work["control_combined_rank"] = control_work.upstream.to_numpy(float)
    work["candidate_combined_rank"] = .75 * work.base_rank_ts.to_numpy(float) + .25 * work.meta_rank_ts.to_numpy(float)
    work["base_anchor_bps"] = anchor.predict(work.base_rank_ts).astype(float)
    work["residual_bps"] = work.policy_net_bps - work.base_anchor_bps
    work["rank_correction"] = work.meta_rank_ts - work.base_rank_ts
    work["abs_rank_correction"] = work.rank_correction.abs()

    base = work.loc[:, [*IDENTITY, "base_rank_ts", "policy_net_bps", "utility_bps"]].copy()
    base["base_only_rank"] = base.base_rank_ts
    candidate_selection = work.loc[:, [*IDENTITY, "candidate_combined_rank", "policy_net_bps", "utility_bps"]].copy()
    candidate_selection["selection_rank"] = candidate_selection.candidate_combined_rank
    control_selection = control_work.loc[:, [*IDENTITY, "control_combined_rank", "policy_net_bps", "utility_bps"]].copy()

    top: dict[str, float] = {}
    weekly: list[pd.DataFrame] = []
    for k in (1, 2, 5):
        selected = _rank_top(candidate_selection, "selection_rank", k)
        control_selected = _rank_top(control_selection, "control_combined_rank", k)
        base_selected = _rank_top(base, "base_only_rank", k)
        candidate_ts, control_ts = _timestamp_mean(selected), _timestamp_mean(control_selected)
        base_ts = _timestamp_mean(base_selected)
        top[f"meta_top{k}_ev"] = _mean(candidate_ts)
        top[f"control_top{k}_ev"] = _mean(control_ts)
        top[f"base_top{k}_ev"] = _mean(base_ts)
        top[f"probe_delta_top{k}_ev"] = _mean(candidate_ts - base_ts)
        top[f"candidate_minus_control_top{k}_ev"] = _mean(candidate_ts - control_ts)
        if k in {1, 2}:
            delta = _set_delta_values(candidate_selection, control_selection, "selection_rank", k)
            top[f"top{k}_candidate_only_minus_control_only_ev"] = _mean(delta.candidate_minus_control_only_ev)
            weekly.append(pd.DataFrame({
                "__decision_ts__": candidate_ts.index,
                f"candidate_top{k}_ev": candidate_ts.to_numpy(float),
                f"control_top{k}_ev": control_ts.reindex(candidate_ts.index).to_numpy(float),
                f"base_top{k}_ev": base_ts.reindex(candidate_ts.index).to_numpy(float),
            }))

    candidate_admit = _top_fraction(candidate_selection, "selection_rank")
    base_admit = _top_fraction(base, "base_only_rank")
    control_admit = _top_fraction(control_selection, "control_combined_rank")
    candidate_admit_ts = _timestamp_mean(candidate_admit, "utility_bps")
    control_admit_ts = _timestamp_mean(control_admit, "utility_bps")
    base_admit_ts = _timestamp_mean(base_admit, "utility_bps")
    precision50 = candidate_admit.policy_net_bps.gt(50.0).mean()
    precision100 = candidate_admit.policy_net_bps.gt(100.0).mean()
    utility_sep50 = (
        _mean(candidate_admit.loc[candidate_admit.policy_net_bps.gt(50.0), "utility_bps"])
        - _mean(candidate_admit.loc[candidate_admit.policy_net_bps.le(50.0), "utility_bps"])
    )
    utility_sep100 = (
        _mean(candidate_admit.loc[candidate_admit.policy_net_bps.gt(100.0), "utility_bps"])
        - _mean(candidate_admit.loc[candidate_admit.policy_net_bps.le(100.0), "utility_bps"])
    )

    bands = {
        "ic_base_0_5": (0.95, 1.001), "ic_base_5_10": (0.90, 0.95),
        "ic_base_10_20": (0.80, 0.90), "ic_base_20_30": (0.70, 0.80),
    }
    band_values: dict[str, float] = {}
    for name, (low, high) in bands.items():
        local = work.loc[work.base_rank_ts.ge(low) & work.base_rank_ts.lt(high)]
        band_values[name] = float(spearmanr(local.meta_rank_ts, local.residual_bps).statistic) if len(local) >= 40 else float("nan")

    geometry_masks = {
        "ev_base_high_meta_high": work.base_rank_ts.ge(BASE_HIGH) & work.meta_rank_ts.ge(META_HIGH),
        "ev_base_high_meta_low": work.base_rank_ts.ge(BASE_HIGH) & work.meta_rank_ts.lt(META_HIGH),
        "ev_base_low_meta_high": work.base_rank_ts.lt(BASE_HIGH) & work.meta_rank_ts.ge(META_HIGH),
        "ev_base_low_meta_low": work.base_rank_ts.lt(BASE_HIGH) & work.meta_rank_ts.lt(META_HIGH),
    }
    geometry = {name: _mean(work.loc[mask, "policy_net_bps"]) for name, mask in geometry_masks.items()}
    correction_band = pd.cut(work.abs_rank_correction, bins=list(CORRECTION_BINS), right=False, include_lowest=True)
    for index, label in enumerate(("0_005", "005_015", "015_030", "030_plus")):
        geometry[f"ev_correction_band_{label}"] = _mean(work.loc[correction_band.cat.codes.eq(index), "policy_net_bps"])

    weekly_panel = pd.concat(weekly, axis=1)
    weekly_panel = weekly_panel.loc[:, ~weekly_panel.columns.duplicated()].copy()
    weekly_panel["week"] = pd.to_datetime(weekly_panel.__decision_ts__, utc=True).dt.to_period("W-SUN").astype(str)
    weekly_panel["top2_delta_ev"] = weekly_panel.candidate_top2_ev - weekly_panel.control_top2_ev
    week_metric = weekly_panel.groupby("week", sort=True).top2_delta_ev.mean()
    # Month-level stability is attached after all held folds are assembled.
    # A single held-fold row is not itself a monthly distribution.

    model = dict(trial.get("model", {}))
    row: dict[str, Any] = {
        "score_root": root_name, "trial": str(trial["name"]), "held_month": f"{held_month:%Y-%m}",
        "target": trial.get("target"), "arm_name": trial.get("arm_name"), "target_family": str(trial.get("arm_name", "")).split("__", 1)[0],
        # ``parent_contract`` is deliberately shared by append-only SHAP
        # variants.  It is not the grouping key for a leave-feature-contract-
        # out proxy validation.  The immutable score-root contract is.
        "feature_contract": feature_contract,
        "parent_feature_contract": trial.get("parent_contract"),
        "feature_count": int(feature_count), "feature_family": trial.get("additive_feature_family"),
        "feature_mode": trial.get("feature_mode"), "sample_weight_profile": json.dumps(trial.get("sample_weight"), sort_keys=True, default=str),
        "loss": model.get("objective"), "query_contract": work.get("query_contract", pd.Series([trial.get("arm_name")])).iloc[0],
        "gain": json.dumps(trial.get("gain"), default=str), "truncation": trial.get("truncation"), "sigmoid": trial.get("sigmoid"),
        "rows": int(len(work)), "queries": int(work.__decision_ts__.nunique()),
        "residual_ic": float(spearmanr(work.meta_rank_ts, work.residual_bps).statistic),
        "conditional_mi_given_base": _conditional_mi(work),
        "base_meta_rank_correlation": float(spearmanr(work.base_rank_ts, work.meta_rank_ts).statistic),
        "top1_overlap": _jaccard_by_timestamp(work, work, "base_rank_ts", "meta_rank_ts", 1),
        "top2_overlap": _jaccard_by_timestamp(work, work, "base_rank_ts", "meta_rank_ts", 2),
        "top5_overlap": _jaccard_by_timestamp(work, work, "base_rank_ts", "meta_rank_ts", 5),
        "median_abs_rank_correction": _q(work.abs_rank_correction, .50), "p90_rank_correction": _q(work.abs_rank_correction, .90),
        "upgrade_fraction": float(work.rank_correction.gt(0).mean()), "downgrade_fraction": float(work.rank_correction.lt(0).mean()),
        "ev_upgrades": _mean(work.loc[work.rank_correction.gt(0), "policy_net_bps"]),
        "ev_downgrades": _mean(work.loc[work.rank_correction.lt(0), "policy_net_bps"]),
        "useful_upgrade_ev": _mean(work.loc[work.rank_correction.gt(0) & work.policy_net_bps.gt(50.0), "policy_net_bps"]),
        "false_upgrade_ev": _mean(work.loc[work.rank_correction.gt(0) & work.policy_net_bps.le(0.0), "policy_net_bps"]),
        "useful_upgrade_fraction": float((work.rank_correction.gt(0) & work.policy_net_bps.gt(50.0)).mean()),
        "false_upgrade_fraction": float((work.rank_correction.gt(0) & work.policy_net_bps.le(0.0)).mean()),
        "utility_separation_gt50": utility_sep50, "utility_separation_gt100": utility_sep100,
        "selected_precision_gt50": float(precision50), "selected_precision_gt100": float(precision100),
        "probe_delta_admitted_utility": _mean(candidate_admit_ts - base_admit_ts),
        "probe_delta_gt50_precision": float(precision50 - base_admit.policy_net_bps.gt(50.0).mean()),
        "candidate_minus_control_admitted_utility": _mean(candidate_admit_ts - control_admit_ts),
        "weekly_mean": _mean(week_metric), "weekly_q25": _q(week_metric, .25), "weekly_q10": _q(week_metric, .10), "weekly_q5": _q(week_metric, .05),
        **top, **band_values, **geometry,
    }
    for name, value in model.items():
        if isinstance(value, (int, float, bool, str)) or value is None:
            row[f"hpo__{name}"] = value
    weekly_panel["score_root"] = root_name; weekly_panel["trial"] = str(trial["name"]); weekly_panel["held_month"] = f"{held_month:%Y-%m}"
    return row, weekly_panel


def _bootstrap_summary(fold: pd.DataFrame, *, iterations: int, seed: int) -> pd.DataFrame:
    # Keep every diagnostic consumed by the learned Priority/Gate proxies.
    # ``ic_base_*`` and useful/false-upgrade measures were already calculated
    # per fold but accidentally omitted from the trial summary, making the
    # proxy fitter refuse its declared feature contract.
    numeric = [name for name in fold.columns if name.startswith((
        "meta_top", "probe_", "candidate_", "residual_", "conditional_", "utility_", "selected_",
        "weekly_", "monthly_", "ev_", "top", "base_meta", "median_", "p90_", "upgrade_", "downgrade_",
        "ic_", "useful_", "false_",
    ))]
    results: list[dict[str, Any]] = []
    for (root, trial), group in fold.groupby(["score_root", "trial"], sort=True):
        means = group[numeric].mean(numeric_only=True)
        rng = np.random.default_rng(seed + int.from_bytes(hashlib.sha256(f"{root}|{trial}".encode()).digest()[:4], "little"))
        boot = np.empty((iterations, len(numeric)), dtype=float)
        values = group[numeric].to_numpy(float)
        for index in range(iterations):
            boot[index] = np.nanmean(values[rng.integers(0, len(values), size=len(values))], axis=0)
        item: dict[str, Any] = {"score_root": root, "trial": trial, "folds": int(len(group))}
        for column, value in means.items():
            item[column] = float(value)
            pos = numeric.index(column)
            item[f"{column}__bootstrap_se"] = float(np.nanstd(boot[:, pos], ddof=1))
        for column in ("target", "arm_name", "target_family", "feature_contract", "parent_feature_contract", "feature_count", "feature_family", "feature_mode", "sample_weight_profile", "loss", "query_contract", "gain", "truncation", "sigmoid"):
            item[column] = group[column].iloc[0]
        results.append(item)
    return pd.DataFrame(results)


def _attach_cross_fold_stability(summary: pd.DataFrame, fold: pd.DataFrame, weekly: pd.DataFrame) -> pd.DataFrame:
    """Attach true cross-fold month/week stability after all OOF folds exist."""
    output = summary.copy()
    stability: list[dict[str, Any]] = []
    for (root, trial), group in fold.groupby(["score_root", "trial"], sort=True):
        monthly = pd.to_numeric(group["candidate_minus_control_top2_ev"], errors="coerce")
        later = weekly.loc[weekly.score_root.eq(root) & weekly.trial.eq(trial)].copy()
        later["delta"] = pd.to_numeric(later.candidate_top2_ev, errors="coerce") - pd.to_numeric(later.control_top2_ev, errors="coerce")
        per_week = later.groupby(pd.to_datetime(later.__decision_ts__, utc=True).dt.to_period("W-SUN").astype(str), sort=True).delta.mean()
        stability.append({
            "score_root": root, "trial": trial,
            "weekly_mean": _mean(per_week), "weekly_q25": _q(per_week, .25),
            "weekly_q10": _q(per_week, .10), "weekly_q5": _q(per_week, .05),
            "monthly_median": _q(monthly, .50), "monthly_q25": _q(monthly, .25),
            "positive_month_fraction": float(monthly.gt(0).mean()), "worst_fold": _q(monthly, 0.0),
        })
    replacement = pd.DataFrame(stability)
    drop = [column for column in replacement.columns if column in output.columns and column not in {"score_root", "trial"}]
    return output.drop(columns=drop).merge(replacement, on=["score_root", "trial"], how="left", validate="one_to_one")


def _root_contract(root: Path) -> tuple[str, int, list[dict[str, Any]]]:
    """Return the actual immutable score-root feature contract and trials.

    Trial-local ``parent_contract`` values identify the shared F120 ancestor
    and therefore cannot be used to distinguish append-only feature sets.
    """
    manifest = _read_json(root / "run_manifest.json")
    contract, count, trials = (
        manifest.get("meta_feature_contract"),
        manifest.get("meta_feature_count"),
        manifest.get("trials"),
    )
    if not isinstance(contract, str) or not contract:
        raise AssertionError(f"{root}: missing actual meta_feature_contract")
    if not isinstance(count, int) or count <= 0:
        raise AssertionError(f"{root}: missing actual meta_feature_count")
    if not isinstance(trials, list) or not trials:
        raise AssertionError(f"{root}: missing trial receipt")
    return contract, int(count), [dict(trial) for trial in trials]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proxy-spec", type=Path, required=True)
    parser.add_argument("--hpo-config", type=Path, required=True)
    parser.add_argument("--canonical-root", type=Path, required=True)
    parser.add_argument("--score-root", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=500)
    parser.add_argument("--bootstrap-seed", type=int, default=1729)
    parser.add_argument(
        "--source-override", type=Path,
        help="immutable source-only binding receipt for a causal ledger successor",
    )
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"immutable output exists: {args.out}")
    if args.bootstrap_iterations < 100:
        raise ValueError("bootstrap iterations must be at least 100")

    proxy_spec = _read_json(args.proxy_spec.resolve())
    if proxy_spec.get("schema") != "strict_r3_p8u_meta_learned_downstream_proxy_v1":
        raise AssertionError("unexpected learned-proxy specification")
    raw, applied_source_override = screen._apply_source_override(
        _read_json(args.hpo_config.resolve()),
        args.source_override.resolve() if args.source_override else None,
    )
    control_root = args.canonical_root.resolve()
    roots = [path.resolve() for path in args.score_root]
    root_contracts: dict[Path, tuple[str, int, list[dict[str, Any]]]] = {}
    for root in roots:
        if not (root / "run_manifest.json").exists() or not (root / "objective_summary.parquet").exists():
            raise AssertionError(f"{root}: incomplete strict-OOF score root")
        root_contracts[root] = _root_contract(root)

    policy_path = (ROOT / str(raw["source"]["policy_labels"])).resolve()
    path_root = (ROOT / str(raw["source"]["path_labels"])).resolve()
    base_root = (ROOT / str(raw["source"]["base_target_free_root"])).resolve()
    policy = screen._read_policy(policy_path)
    args.out.mkdir(parents=True)
    _once(args.out / "run_manifest.json", {
        "schema": SCHEMA,
        "scope": "offline descriptors only; no hand-weighted selection, no MC1 fit, no admission, no portfolio, no live or exchange mutation",
        "proxy_spec": str(args.proxy_spec.resolve()), "proxy_spec_sha256": _sha(args.proxy_spec.resolve()),
        "hpo_config": str(args.hpo_config.resolve()), "hpo_config_sha256": _sha(args.hpo_config.resolve()),
        "source_override": str(args.source_override.resolve()) if args.source_override else None,
        "source_override_sha256": _sha(args.source_override.resolve()) if args.source_override else None,
        "source_override_payload": applied_source_override,
        "canonical_control": str(control_root),
        "score_roots": [str(root) for root in roots],
        "score_root_feature_contracts": {
            root.name: {"feature_contract": root_contracts[root][0], "feature_count": root_contracts[root][1]}
            for root in roots
        },
        "held_months": [f"{month:%Y-%m}" for month in HELD_MONTHS],
        "source": {"base": str(base_root), "policy": str(policy_path), "path": str(path_root)},
        "causality": "candidate and canonical target-free score receipts are read and exact-identity/Base-rank validated before any held outcome/path label is opened; anchors use only pre-resolved labels",
        "selection_authority": "none; descriptors feed a later learned PriorityProxy/GateProxy only",
    })

    # Cache each fold's later-opened labels and prequential anchor once.  This
    # prevents any trial-specific label materialisation from changing the
    # comparison population.
    anchors = {
        month: authority._held_anchor(raw=raw, base_root=base_root, policy=policy, path_root=path_root, held_month=month)
        for month in HELD_MONTHS
    }
    labelled: dict[pd.Timestamp, pd.DataFrame] = {}
    fold_rows: list[dict[str, Any]] = []
    weekly_rows: list[pd.DataFrame] = []
    audits: list[dict[str, Any]] = []
    for root in roots:
        feature_contract, feature_count, trials = root_contracts[root]
        for trial in trials:
            trial_name = str(trial["name"])
            for month in HELD_MONTHS:
                candidate = _read_candidate_score(_score_path(root, trial_name, month))
                control = _read_control_score(_control_path(control_root, month))
                _validate_target_free_pair(candidate, control, trial=trial_name, month=month)
                # Only now is the cached held outcome panel opened.
                if month not in labelled:
                    labelled[month] = screen._labelled(control, policy, path_root, month, screen._month_end(month))
                row, weekly = _metric_row(
                    candidate=candidate, control=control, labelled=labelled[month], anchor=anchors[month],
                    trial=trial, root_name=root.name, feature_contract=feature_contract,
                    feature_count=feature_count, held_month=month,
                )
                fold_rows.append(row); weekly_rows.append(weekly)
                audits.append({
                    "score_root": root.name, "trial": trial_name, "held_month": f"{month:%Y-%m}",
                    "target_free_candidate_score_validated_before_outcome_join": True,
                    "target_free_control_score_validated_before_outcome_join": True,
                    "candidate_control_identity_exact": True, "candidate_base_rank_matches_control": True,
                    "held_anchor_is_pre_resolved_only": True, "labels_excluded_from_score_producer": True,
                    "actual_score_root_feature_contract": feature_contract,
                    "actual_score_root_feature_count": feature_count,
                })

    fold = pd.DataFrame(fold_rows)
    weekly_frame = pd.concat(weekly_rows, ignore_index=True)
    summary = _bootstrap_summary(fold, iterations=args.bootstrap_iterations, seed=args.bootstrap_seed)
    summary = _attach_cross_fold_stability(summary, fold, weekly_frame)
    fold.to_parquet(args.out / "trial_fold_descriptors.parquet", index=False, compression="zstd")
    summary.to_parquet(args.out / "trial_descriptor_summary.parquet", index=False, compression="zstd")
    weekly_frame.to_parquet(args.out / "trial_weekly_descriptors.parquet", index=False, compression="zstd")
    pd.DataFrame(audits).to_parquet(args.out / "correctness_audit.parquet", index=False, compression="zstd")
    _once(args.out / "correctness_report.json", {
        "target_free_scores_checked_before_held_outcomes_opened": True,
        "target_free_score_inputs_contain_no_policy_or_path_fields": True,
        "candidate_control_candidate_id_timestamp_side_identity_is_exact": True,
        "candidate_base_rank_matches_frozen_control": True,
        "feature_contract_uses_actual_score_root_not_shared_parent_reference": True,
        "held_outcomes_used_only_for_post_score_descriptors": True,
        "held_anchor_uses_only_prior_resolved_labels": True,
        "descriptors_have_no_selection_authority": True,
        "no_mc1_admission_portfolio_live_or_exchange_mutation": True,
    })
    print(args.out)


if __name__ == "__main__":
    main()
