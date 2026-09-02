#!/usr/bin/env python3
"""Strict-OOF SHAP-derived feature audit for routed E/T Base heads.

This producer is deliberately downstream of the full-universe screen.  It
does *not* run CMI or IC over the raw causal feature universe.  Instead it:

  1. fits each Screen120 E/T parent only on labels resolved before an outer
     fold reserve;
  2. writes target-free held parent scores and SHAP-derived attribution /
     interaction-transform features; and only then
  3. joins held policy outcomes to audit the new derived fields with
     conditional MI, timestamp-local IC, and two-sided timestamp Top-10 EV.

The output is a diagnostic and feature-engineering receipt.  It has no live,
MC1, admission, portfolio, or exchange authority.  Derived fields must pass
their own strict OOF economic gate before they can enter a future Base or Meta
feature-selection funnel.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMRegressor


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
import run_strict_r3_routed_et_fulluniverse_screen as screen  # noqa: E402


SCHEMA = "strict_r3_p8u_shapderived_feature_pipeline_v1"
SEED = 1729
IDENTITY = screen.IDENTITY
# ``target`` is deliberately *not* a prohibited token: causal structural
# fields such as ``reversion_target_distance`` are legitimate decision-time
# inputs.  Outcome-derived prefixes are prohibited at the source-contract
# boundary instead of relying on an overly broad substring test over the
# generated explanatory field names.
PROHIBITED_SOURCE_PREFIXES = ("policy_", "supportive_", "h12_", "label_")


def _utc_month(value: str) -> pd.Timestamp:
    return pd.Timestamp(f"{value.strip()}-01", tz="UTC")


def _write_once(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def _hash_lines(values: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(map(str, values)).encode("utf-8")).hexdigest()


def _progress(root: Path, **payload: object) -> None:
    with (root / "progress.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")


def _head_fields(screen_root: Path, head: str, width: int) -> tuple[str, ...]:
    contract = json.loads((screen_root / f"{head.lower()}_screen120_contract.json").read_text())
    fields = tuple(map(str, contract["feature_contract"]))
    if not 20 <= len(fields) <= 120 or len(fields) != len(set(fields)):
        raise AssertionError(f"{head}: invalid Screen120 feature contract")
    invalid = [field for field in fields if field.lower().startswith(PROHIBITED_SOURCE_PREFIXES)]
    if invalid:
        raise AssertionError(f"{head}: Screen120 contract contains noncausal source fields: {invalid[:5]}")
    summary = pd.read_parquet(screen_root / f"{head.lower()}_screen_feature_summary.parquet")
    ranked = summary.loc[summary.feature.isin(fields)].copy()
    ranked["__shap__"] = ranked[["global_shap", "precision_shap"]].max(axis=1)
    ordered = ranked.sort_values(["__shap__", "screen_score", "feature"], ascending=[False, False, True], kind="stable").feature.astype(str).tolist()
    selected = tuple(ordered[: min(width, len(ordered))])
    if len(selected) < 8:
        raise AssertionError(f"{head}: insufficient stable SHAP contribution fields")
    return fields, selected


def _bins(values: np.ndarray, bins: int) -> np.ndarray:
    """Deterministic quantile bins with a one-bin constant fallback."""
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    result = np.full(len(values), -1, dtype=np.int16)
    if finite.sum() < max(32, bins * 3):
        return result
    rank = pd.Series(values[finite]).rank(method="first", pct=True).to_numpy(float)
    result[finite] = np.minimum(bins - 1, np.floor(rank * bins).astype(np.int16))
    return result


def _conditional_mi(x: np.ndarray, y: np.ndarray, condition: np.ndarray, bins: int = 10) -> float:
    """Plug-in I(X;Y|Z), used only for post-score diagnostics."""
    xb, yb, zb = _bins(x, bins), _bins(y, bins), _bins(condition, bins)
    valid = (xb >= 0) & (yb >= 0) & (zb >= 0)
    if valid.sum() < 256:
        return float("nan")
    work = pd.DataFrame({"x": xb[valid], "y": yb[valid], "z": zb[valid]})
    n = float(len(work))
    xyz = work.groupby(["x", "y", "z"], sort=False).size().rename("n").reset_index()
    xz = work.groupby(["x", "z"], sort=False).size().rename("nxz").reset_index()
    yz = work.groupby(["y", "z"], sort=False).size().rename("nyz").reset_index()
    z = work.groupby("z", sort=False).size().rename("nz").reset_index()
    merged = xyz.merge(xz, on=["x", "z"], validate="many_to_one").merge(yz, on=["y", "z"], validate="many_to_one").merge(z, on="z", validate="many_to_one")
    probability = merged.n.to_numpy(float) / n
    ratio = (merged.n.to_numpy(float) * merged.nz.to_numpy(float)) / (merged.nxz.to_numpy(float) * merged.nyz.to_numpy(float))
    return float(np.sum(probability * np.log(np.maximum(ratio, 1e-12))))


def _timestamp_ic(frame: pd.DataFrame, field: str) -> float:
    rows: list[float] = []
    for _, part in frame.groupby("__decision_ts__", sort=False):
        x = pd.to_numeric(part[field], errors="coerce")
        y = pd.to_numeric(part.policy_net_bps, errors="coerce")
        if len(part) >= 5 and x.nunique(dropna=True) >= 3 and y.nunique(dropna=True) >= 3:
            value = x.corr(y, method="spearman")
            if np.isfinite(value):
                rows.append(float(value))
    return float(np.mean(rows)) if rows else float("nan")


def _two_sided_top10(frame: pd.DataFrame, field: str) -> tuple[float, float]:
    work = frame.loc[:, ["__decision_ts__", "candidate_id", field, "policy_net_bps"]].copy()
    work[field] = pd.to_numeric(work[field], errors="coerce")
    results: list[float] = []
    for ascending in (False, True):
        ordered = work.sort_values(["__decision_ts__", field, "candidate_id"], ascending=[True, ascending, True], kind="stable")
        ordered["__ord__"] = ordered.groupby("__decision_ts__", sort=False).cumcount() + 1
        ordered["__size__"] = ordered.groupby("__decision_ts__", sort=False).candidate_id.transform("size")
        selected = ordered.loc[ordered.__ord__.le(np.ceil(ordered.__size__.to_numpy(float) * .10))]
        results.append(float(selected.groupby("__decision_ts__", sort=False).policy_net_bps.mean().mean()))
    return tuple(results)


def _stable_pair_indices(model: LGBMRegressor, x_train: np.ndarray, *, sample_cap: int, pairs: int, seed: int) -> list[tuple[int, int]]:
    """Discover pair definitions on training data only with bounded TreeSHAP."""
    if len(x_train) < 128:
        return []
    sample = screen._stratified_index(  # type: ignore[arg-type]
        pd.DataFrame({"candidate_id": np.arange(len(x_train)).astype(str), "__decision_ts__": pd.date_range("2000-01-01", periods=len(x_train), freq="h", tz="UTC")}),
        min(sample_cap, len(x_train)), seed=seed,
    )
    values = np.asarray(x_train[sample], dtype=np.float32)
    explainer = shap.TreeExplainer(model.booster_)
    interactions = np.asarray(explainer.shap_interaction_values(values), dtype=np.float32)
    if interactions.ndim == 4:  # defensive for classifiers; this producer uses regressors.
        interactions = interactions[0]
    if interactions.ndim != 3 or interactions.shape[1] != interactions.shape[2]:
        raise AssertionError(f"unexpected TreeSHAP interaction shape={interactions.shape}")
    strength = np.mean(np.abs(interactions), axis=0)
    np.fill_diagonal(strength, 0.0)
    order = np.dstack(np.unravel_index(np.argsort(strength.ravel())[::-1], strength.shape))[0]
    selected: list[tuple[int, int]] = []
    used: set[tuple[int, int]] = set()
    for first, second in order:
        pair = (int(min(first, second)), int(max(first, second)))
        if pair[0] == pair[1] or pair in used or strength[pair] <= 0.0:
            continue
        selected.append(pair)
        used.add(pair)
        if len(selected) >= pairs:
            break
    return selected


def _derived(
    *, head: str, fields: Sequence[str], contribution_fields: Sequence[str], pairs: Sequence[tuple[int, int]],
    x_train: np.ndarray, x_held: np.ndarray, contribution: np.ndarray,
) -> pd.DataFrame:
    """Build inference-available attribution summaries and SHAP-selected pairs."""
    signed = np.asarray(contribution[:, :-1], dtype=np.float32)
    absolute = np.abs(signed)
    total = absolute.sum(axis=1)
    safe_total = np.maximum(total, 1e-8)
    proportions = absolute / safe_total[:, None]
    entropy = -np.sum(np.where(proportions > 0.0, proportions * np.log(np.maximum(proportions, 1e-12)), 0.0), axis=1) / math.log(max(2, absolute.shape[1]))
    top = np.partition(absolute, -min(3, absolute.shape[1]), axis=1)[:, -min(3, absolute.shape[1]):]
    result: dict[str, np.ndarray] = {
        f"shap_{head.lower()}_abs_total": total,
        f"shap_{head.lower()}_signed_total": signed.sum(axis=1),
        f"shap_{head.lower()}_positive_total": np.maximum(signed, 0.0).sum(axis=1),
        f"shap_{head.lower()}_negative_total": np.minimum(signed, 0.0).sum(axis=1),
        f"shap_{head.lower()}_top1_abs_share": absolute.max(axis=1) / safe_total,
        f"shap_{head.lower()}_top3_abs_share": top.sum(axis=1) / safe_total,
        f"shap_{head.lower()}_entropy": entropy,
    }
    index = {field: offset for offset, field in enumerate(fields)}
    for field in contribution_fields:
        result[f"shap_{head.lower()}_contrib__{field}"] = signed[:, index[field]]
    median = np.nanmedian(x_train, axis=0)
    q25, q75 = np.nanquantile(x_train, [.25, .75], axis=0)
    scale = np.maximum(q75 - q25, 1e-6)
    z = np.clip((x_held - median) / scale, -8.0, 8.0)
    for first, second in pairs:
        left, right = fields[first], fields[second]
        stem = f"shap_{head.lower()}_pair__{left}__{right}"
        result[f"{stem}__product"] = z[:, first] * z[:, second]
        result[f"{stem}__agreement"] = -np.abs(z[:, first] - z[:, second])
    frame = pd.DataFrame(result)
    if frame.columns.duplicated().any() or not len(frame.columns):
        raise AssertionError("invalid SHAP-derived feature names")
    return frame.replace([np.inf, -np.inf], np.nan).astype(np.float32)


def _fold(
    *, head: str, fields: Sequence[str], contribution_fields: Sequence[str], held_month: pd.Timestamp,
    feature_root: Path, router_root: Path, labels_root: Path, policy: pd.DataFrame,
    train_months: int, reserve_days: int, train_cap: int, held_cap: int, n_jobs: int,
    interaction_sample: int, interaction_pairs: int, out: Path, base_labels_root: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    spec = screen.HEADS[head]
    target, direction = str(spec["target"]), float(spec["direction"])
    reserve = held_month - pd.Timedelta(days=reserve_days)
    start = reserve - pd.DateOffset(months=train_months)
    window = screen._joined(feature_root=feature_root, router_root=router_root, labels_root=labels_root, policy=policy, start=start, end=screen._month_end(held_month), fields=(), route_fraction=.50, base_labels_root=base_labels_root)
    train = screen._strict_train(window.loc[window.__decision_ts__.lt(reserve)].copy(), reserve, target, train_cap)
    held = screen._time_balanced_sample(screen._held_eval(window.loc[window.__decision_ts__.ge(held_month)].copy(), target), held_cap, seed=SEED + held_month.month)
    if len(train) < 8_000 or len(held) < 1_000:
        raise AssertionError(f"{head}/{held_month:%Y-%m}: insufficient strict support")
    selected = pd.concat([train, held], ignore_index=True)
    matrix = screen._selected_feature_matrix(feature_root, selected, fields)
    matrix, _ = screen._impute_from_train(matrix, len(train))
    x_train, x_held = matrix[:len(train)], matrix[len(train):]
    model = LGBMRegressor(**screen._params(seed=SEED + 1000 * held_month.month + (0 if head == "E" else 100_000), n_jobs=n_jobs))
    model.fit(x_train, pd.to_numeric(train[target], errors="coerce").to_numpy(float))
    raw_score = direction * model.predict(x_held)
    contribution = direction * np.asarray(model.predict(x_held, pred_contrib=True), dtype=np.float32)
    pairs = _stable_pair_indices(model, x_train, sample_cap=interaction_sample, pairs=interaction_pairs, seed=SEED + held_month.month)
    derived = _derived(head=head, fields=fields, contribution_fields=contribution_fields, pairs=pairs, x_train=x_train, x_held=x_held, contribution=contribution)
    target_free = held.loc[:, list(IDENTITY)].reset_index(drop=True).copy()
    target_free[f"shap_{head.lower()}_parent_score"] = np.asarray(raw_score, dtype=np.float32)
    target_free = pd.concat([target_free, derived.reset_index(drop=True)], axis=1)
    if any(any(f"__{prefix}" in field.lower() for prefix in PROHIBITED_SOURCE_PREFIXES) for field in derived.columns):
        raise AssertionError("prohibited source field flowed into SHAP-derived feature name")
    target_path = out / "target_free_derived" / f"head={head}" / f"month={held_month:%Y-%m}.parquet"
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_free.to_parquet(target_path, index=False, compression="zstd")
    # Outcome metrics are intentionally calculated after the target-free
    # receipt is durable.  Derived-field CMI/IC is diagnostic only.
    evaluated = target_free.copy()
    evaluated["policy_net_bps"] = pd.to_numeric(held.policy_net_bps, errors="coerce").to_numpy(float)
    rows: list[dict[str, object]] = []
    parent = evaluated[f"shap_{head.lower()}_parent_score"].to_numpy(float)
    outcome = evaluated.policy_net_bps.to_numpy(float)
    for field in derived.columns:
        high, low = _two_sided_top10(evaluated, field)
        rows.append({
            "head": head, "held_month": f"{held_month:%Y-%m}", "feature": field,
            "conditional_mi_policy_given_parent_score": _conditional_mi(evaluated[field].to_numpy(float), outcome, parent),
            "timestamp_spearman_ic_policy": _timestamp_ic(evaluated, field),
            "ts_top10_ev_high": high, "ts_top10_ev_low": low,
            "rows": len(evaluated), "parent_ts_top10_ev": screen._metric_suite(held.assign(__score__=raw_score), "__score__")["ts_top10_ev"],
        })
    provenance = {
        "head": head, "held_month": f"{held_month:%Y-%m}", "train_rows": len(train), "held_rows": len(held),
        "target_free_receipt": str(target_path), "interaction_pairs": [[fields[a], fields[b]] for a, b in pairs],
        "target_free_persisted_before_metrics": True,
        "strict_labels_before_reserve": True,
    }
    return pd.DataFrame(rows), target_free, provenance


def run(args: argparse.Namespace) -> None:
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.mkdir(parents=True)
    held_months = tuple(_utc_month(value) for value in args.held_months.split(",") if value.strip())
    if len(held_months) != 3 or tuple(sorted(held_months)) != held_months:
        raise ValueError("require exactly three ordered held months")
    policy = screen._read_policy(args.policy_path, held_months)
    manifest = {
        "schema": SCHEMA,
        "scope": "offline strict-OOF SHAP-derived diagnostic; no live/MC1/admission/portfolio/exchange mutation",
        "screen_root": str(args.screen_root), "feature_root": str(args.feature_root), "router_root": str(args.router_root),
        "labels_root": str(args.labels_root), "base_labels_root": str(args.base_labels_root) if args.base_labels_root else None, "policy_path": str(args.policy_path),
        "held_months": [f"{month:%Y-%m}" for month in held_months],
        "train_months": args.train_months, "reserve_days": args.reserve_days,
        "diagnostics": "CMI and timestamp IC are computed only for newly generated shap_* fields and only after target-free receipt persistence",
        "raw_feature_cmi_or_ic": False,
        "interaction": {"tree_shap_train_only": True, "sample_cap": args.interaction_sample, "pairs": args.interaction_pairs},
    }
    _write_once(args.out / "run_manifest.json", manifest)
    metrics: list[pd.DataFrame] = []
    provenance: list[dict[str, object]] = []
    contracts: dict[str, object] = {}
    for head in args.heads:
        fields, contribution_fields = _head_fields(args.screen_root, head, args.contribution_fields)
        contracts[head] = {"raw_screen120_fields": list(fields), "raw_screen120_sha256": _hash_lines(fields), "contribution_fields": list(contribution_fields)}
        for month in held_months:
            metric, _, receipt = _fold(
                head=head, fields=fields, contribution_fields=contribution_fields, held_month=month,
                feature_root=args.feature_root, router_root=args.router_root, labels_root=args.labels_root, policy=policy,
                train_months=args.train_months, reserve_days=args.reserve_days, train_cap=args.train_cap, held_cap=args.held_cap,
                n_jobs=args.n_jobs, interaction_sample=args.interaction_sample, interaction_pairs=args.interaction_pairs, out=args.out, base_labels_root=args.base_labels_root,
            )
            metrics.append(metric); provenance.append(receipt)
            _progress(args.out, stage="fold_complete", **receipt)
    detail = pd.concat(metrics, ignore_index=True)
    summary = detail.groupby(["head", "feature"], sort=False).agg(
        cmi_median=("conditional_mi_policy_given_parent_score", "median"),
        ic_mean=("timestamp_spearman_ic_policy", "mean"),
        ic_min=("timestamp_spearman_ic_policy", "min"),
        high_top10_ev_mean=("ts_top10_ev_high", "mean"),
        low_top10_ev_mean=("ts_top10_ev_low", "mean"),
        positive_ic_folds=("timestamp_spearman_ic_policy", lambda x: int((x > 0).sum())),
        folds=("held_month", "nunique"),
    ).reset_index()
    summary["best_two_sided_top10_ev"] = summary[["high_top10_ev_mean", "low_top10_ev_mean"]].max(axis=1)
    detail.to_parquet(args.out / "shap_derived_oof_metrics.parquet", index=False, compression="zstd")
    summary.sort_values(["cmi_median", "ic_mean", "feature"], ascending=[False, False, True], kind="stable").to_parquet(args.out / "shap_derived_summary.parquet", index=False, compression="zstd")
    _write_once(args.out / "derived_feature_contract.json", {"schema": SCHEMA, "heads": contracts, "fold_provenance": provenance, "inference_note": "All shap_* values depend solely on the frozen causal parent model and decision-time raw features; policy outcomes are metric-only."})
    _write_once(args.out / "correctness_report.json", {
        "raw_feature_cmi_or_ic_not_run": True,
        "derived_only_cmi_ic": True,
        "all_held_receipts_persisted_before_metrics": True,
        "all_parent_models_strict_prequential_to_outer_reserve": True,
        "all_derived_names_free_of_outcome_tokens": True,
        "tree_shap_pair_discovery_train_only": True,
        "live_or_exchange_mutation": False,
    })
    _progress(args.out, stage="complete", derived_features=int(summary.feature.nunique()), heads=list(args.heads))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen-root", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--router-root", type=Path, required=True)
    parser.add_argument("--labels-root", type=Path, required=True)
    parser.add_argument("--policy-path", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--heads", nargs="+", choices=("B0", "E", "T"), default=("E", "T"))
    parser.add_argument("--base-labels-root", type=Path, help="required only when auditing B0 policy-ordinal target")
    parser.add_argument("--held-months", default="2025-05,2025-06,2025-07")
    parser.add_argument("--train-months", type=int, default=4)
    parser.add_argument("--reserve-days", type=int, default=28)
    parser.add_argument("--train-cap", type=int, default=120_000)
    parser.add_argument("--held-cap", type=int, default=25_000)
    parser.add_argument("--contribution-fields", type=int, default=16)
    parser.add_argument("--interaction-sample", type=int, default=1_000)
    parser.add_argument("--interaction-pairs", type=int, default=8)
    parser.add_argument("--n-jobs", type=int, default=min(6, os.cpu_count() or 1))
    args = parser.parse_args()
    if args.out.exists() or args.train_months < 2 or args.reserve_days < 12 or args.interaction_pairs < 1:
        raise ValueError("invalid immutable SHAP-derived run contract")
    if "B0" in args.heads and args.base_labels_root is None:
        raise ValueError("--base-labels-root is required when auditing B0")
    run(args)


if __name__ == "__main__":
    main()
