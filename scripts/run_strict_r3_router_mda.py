#!/usr/bin/env python3
"""Strict-OOF MDA, subset and add-back selection for the research router.

This runner deliberately operates only on the base-router research contract.
It never writes a live bundle, never builds a score-time target feature, and
never uses a held-period policy/path outcome before the held target-free score
has been generated.  Every evaluation unit is one decision timestamp.

Stages are intentionally resumable and immutable by output file:

``mda``
    Fits the frozen ranker on three chronological OOF folds and two seeds.
    Permutes individual features and training-derived Spearman families within
    held timestamp queries, then records economic-recall loss.
``subsets``
    Strictly refits the 100/80/70/60/50/40 MDA contracts on the same folds.
``addback``
    Refits the chosen compact contract plus one frozen correlation family.

The final JSON is a development-selection receipt only.  A later full ledger
and a later untouched period remain required before any promotion decision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
import run_strict_r3_economic_recall_router as rr  # noqa: E402


SCHEMA = "strict_r3_router_mda_v1"
SEEDS = (1729, 2718)
FOLDS = ("2025-11", "2026-03", "2026-07")
ROUTES = (0.30, 0.40, 0.50)
ROUTE_WEIGHTS = {0.30: .25, 0.40: .35, 0.50: .40}
BOUNDARIES = ((.20, .40), (.30, .50), (.40, .60))
FEATURE_COUNTS = (100, 80, 70, 60, 50, 40)
PERMUTATION_SEED = 918273

HPO_RANKER: dict[str, object] = {
    "objective": "rank_xendcg", "n_estimators": 2000,
    "learning_rate": .05675711241798078, "max_depth": 4,
    "num_leaves": 15, "min_child_fraction": .017037648187522316,
    "min_child_floor": 500, "min_split_gain": .003215384442187362,
    "subsample": .7279092484163134, "feature_fraction": .7873548972006517,
    "l1": .014167459964217931, "l2": .21674583268126038,
    "max_bin": 127, "truncation": 12, "label_gains": [0, 1, 2, 4, 7, 11],
    "row_weight_scheme": "uniform", "row_weight_floor_bps": 100.0,
    "row_weight_cap_bps": 250.0, "early_stopping_rounds": 30,
    "inner_validation_fraction": .20, "n_jobs": min(4, os.cpu_count() or 1),
    "deterministic": True,
}


def _utc_month(token: str) -> pd.Timestamp:
    return pd.Timestamp(token + "-01", tz="UTC")


def _write_json(path: Path, payload: object) -> None:
    if path.exists():
        raise FileExistsError(path)
    rr._write_json_exclusive(path, payload)


def _sha256(path: Path) -> str:
    return rr._sha256_file(path)


def _parse_tokens(value: str) -> tuple[str, ...]:
    items = tuple(part.strip() for part in value.split(",") if part.strip())
    if not items:
        raise argparse.ArgumentTypeError("at least one comma-separated month is required")
    return items


def _window(args: argparse.Namespace, fields: tuple[str, ...], held_month: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    month = _utc_month(held_month)
    reserve = month - pd.Timedelta(days=args.reserve_days)
    start = reserve - pd.DateOffset(months=args.train_months)
    held_end = rr._month_end(month)
    cols = ("candidate_id", "__decision_ts__", "side_name", *fields)
    train_features = rr._window_features(args.feature_root, start, reserve, cols)
    held = rr._window_features(args.feature_root, month, held_end, cols)
    train_features = rr._deterministic_query_cap(train_features, cap=args.train_cap)
    ids = set(train_features["candidate_id"].astype(str))
    aux = rr._window_aux(args.aux_root, start, reserve)
    aux = aux.loc[aux["candidate_id"].astype(str).isin(ids)].copy()
    policy = rr._policy_window(args.policy_path, start, held_end)
    train = rr._prepare_train(train_features, aux, policy, reserve)
    if len(train) < 20_000 or len(held) < 5_000:
        raise AssertionError(f"{held_month}: inadequate strict support train={len(train)} held={len(held)}")
    return train, held, policy


def _within_timestamp_permute(frame: pd.DataFrame, fields: Sequence[str], token: str) -> dict[str, np.ndarray]:
    """Return deterministic permutations that preserve each timestamp cross-section."""
    seed = int(hashlib.sha256(f"{PERMUTATION_SEED}|{token}".encode()).hexdigest()[:16], 16) % (2**32)
    rng = np.random.default_rng(seed)
    order = np.argsort(frame["__decision_ts__"].to_numpy(), kind="stable")
    stamps = frame["__decision_ts__"].to_numpy()[order]
    starts = np.r_[0, np.flatnonzero(stamps[1:] != stamps[:-1]) + 1, len(order)]
    output: dict[str, np.ndarray] = {}
    for field in fields:
        original = frame[field].to_numpy(copy=True)
        shuffled = original.copy()
        for left, right in zip(starts[:-1], starts[1:]):
            rows = order[left:right]
            if len(rows) > 1:
                shuffled[rows] = original[rng.permutation(rows)]
        output[field] = shuffled
    return output


def _timestamp_percentile(frame: pd.DataFrame, score: np.ndarray) -> np.ndarray:
    work = pd.DataFrame({"__decision_ts__": frame["__decision_ts__"].to_numpy(), "candidate_id": frame["candidate_id"].astype(str).to_numpy(), "score": score})
    work["__row__"] = np.arange(len(work), dtype=np.int64)
    work = work.sort_values(["__decision_ts__", "score", "candidate_id"], ascending=[True, False, True], kind="stable")
    work["__rank__"] = work.groupby("__decision_ts__", sort=False).cumcount().to_numpy(float)
    work["__size__"] = work.groupby("__decision_ts__", sort=False)["candidate_id"].transform("size").to_numpy(float)
    value = (work["__rank__"].to_numpy(float) + .5) / work["__size__"].to_numpy(float)
    return pd.Series(value, index=work["__row__"].to_numpy()).reindex(np.arange(len(frame))).to_numpy(float)


def _metric_rows(
    held: pd.DataFrame, policy: pd.DataFrame, score: np.ndarray, baseline_pct: np.ndarray, *, label: str, held_month: str, seed: int,
) -> list[dict[str, object]]:
    joined = held.loc[:, ["candidate_id", "__decision_ts__"]].merge(policy, on="candidate_id", how="left", validate="one_to_one")
    net = pd.to_numeric(joined["policy_net_bps"], errors="coerce").to_numpy(float)
    valid = joined["policy_path_valid"].fillna(False).to_numpy(bool) & np.isfinite(net)
    positive50 = valid & (net > 50.0)
    positive100 = valid & (net > 100.0)
    positive200 = valid & (net > 200.0)
    excess = np.where(valid, np.maximum(net - 50.0, 0.0), 0.0)
    rank_frame = held.loc[:, ["candidate_id", "__decision_ts__"]].copy()
    rank_frame["__score__"] = np.asarray(score, dtype=np.float32)
    rows: list[dict[str, object]] = []
    for route in ROUTES:
        selected = rr._route_rank(rank_frame, "__score__", route)
        work = pd.DataFrame({
            "ts": held["__decision_ts__"].to_numpy(), "selected": selected, "valid": valid,
            "excess": excess, "p100": positive100, "p200": positive200,
        })
        work["sel_excess"] = work["selected"].to_numpy(bool) * work["excess"].to_numpy(float)
        work["sel_p100"] = work["selected"].to_numpy(bool) * work["p100"].to_numpy(bool)
        work["sel_p200"] = work["selected"].to_numpy(bool) * work["p200"].to_numpy(bool)
        grouped = work.groupby("ts", sort=False).sum(numeric_only=True)
        er50 = np.divide(grouped["sel_excess"], grouped["excess"], out=np.full(len(grouped), np.nan), where=grouped["excess"].to_numpy(float) > 0)
        r100 = np.divide(grouped["sel_p100"], grouped["p100"], out=np.full(len(grouped), np.nan), where=grouped["p100"].to_numpy(float) > 0)
        r200 = np.divide(grouped["sel_p200"], grouped["p200"], out=np.full(len(grouped), np.nan), where=grouped["p200"].to_numpy(float) > 0)
        row: dict[str, object] = {
            "label": label, "held_month": held_month, "seed": seed, "route": route,
            "timestamps": int(len(grouped)), "er50": float(np.nanmean(er50)),
            "recall100": float(np.nanmean(r100)), "recall200": float(np.nanmean(r200)),
        }
        for low, high in BOUNDARIES:
            mask = (baseline_pct >= low) & (baseline_pct < high)
            bw = work.loc[mask].copy()
            if bw.empty:
                value = np.nan
            else:
                bgroup = bw.groupby("ts", sort=False).sum(numeric_only=True)
                value = float(np.nanmean(np.divide(bgroup["sel_excess"], bgroup["excess"], out=np.full(len(bgroup), np.nan), where=bgroup["excess"].to_numpy(float) > 0)))
            row[f"boundary_er50_{int(low*100)}_{int(high*100)}"] = value
        rows.append(row)
    return rows


def _wide_summary(rows: pd.DataFrame, *, baseline_label: str) -> pd.DataFrame:
    metrics = ["er50", "recall100", "recall200", *[f"boundary_er50_{int(a*100)}_{int(b*100)}" for a, b in BOUNDARIES]]
    result: list[dict[str, object]] = []
    baseline = rows.loc[rows["label"].eq(baseline_label)]
    for label, work in rows.groupby("label", sort=False):
        out: dict[str, object] = {"label": label, "observations": int(len(work)), "folds": int(work["held_month"].nunique())}
        score_values: list[float] = []
        for metric in metrics:
            value = pd.to_numeric(work[metric], errors="coerce")
            out[f"mean_{metric}"] = float(value.mean())
            out[f"worst_{metric}"] = float(value.groupby(work["held_month"]).mean().min())
            if label != baseline_label:
                base_lookup = baseline.loc[:, ["held_month", "seed", "route", metric]].rename(columns={metric: "base"})
                delta = work.merge(base_lookup, on=["held_month", "seed", "route"], how="left", validate="one_to_one")[metric].to_numpy(float) - work.merge(base_lookup, on=["held_month", "seed", "route"], how="left", validate="one_to_one")["base"].to_numpy(float)
                out[f"mean_delta_{metric}"] = float(np.nanmean(delta))
                out[f"worst_delta_{metric}"] = float(np.nanmin(delta))
            else:
                out[f"mean_delta_{metric}"] = 0.0
                out[f"worst_delta_{metric}"] = 0.0
        # Predeclared selection: main route-weighted ER50 loss carries 50%;
        # recall100/200 @30 and the three boundary bands share the remainder.
        for route in ROUTES:
            value = work.loc[work["route"].eq(route), "er50"].mean()
            score_values.append(ROUTE_WEIGHTS[route] * float(value))
        primary = float(sum(score_values))
        secondary = float(np.nanmean([
            work.loc[work["route"].eq(.30), "recall100"].mean(),
            work.loc[work["route"].eq(.30), "recall200"].mean(),
            *(work[column].mean() for column in metrics[3:]),
        ]))
        fold_primary = []
        for _, fold in work.groupby(["held_month", "seed"], sort=False):
            fold_primary.append(sum(ROUTE_WEIGHTS[route] * float(fold.loc[fold["route"].eq(route), "er50"].iloc[0]) for route in ROUTES))
        out["primary_er50_score"] = primary
        out["selection_score"] = .5 * primary + .5 * secondary - .20 * float(np.nanstd(fold_primary))
        out["positive_fold_count"] = int(np.sum(np.asarray(fold_primary) > 0.0))
        result.append(out)
    frame = pd.DataFrame(result)
    baseline_score = float(frame.loc[frame["label"].eq(baseline_label), "selection_score"].iloc[0])
    frame["mean_delta_selection_score"] = frame["selection_score"] - baseline_score
    frame["worst_delta_selection_score"] = frame["mean_delta_selection_score"]
    return frame.sort_values("selection_score", ascending=False, kind="stable").reset_index(drop=True)


def _families(train: pd.DataFrame, fields: tuple[str, ...]) -> dict[str, object]:
    sample = train.loc[:, list(fields)].replace([np.inf, -np.inf], np.nan)
    if len(sample) > 50_000:
        sample = sample.sample(n=50_000, random_state=SEEDS[0])
    corr = sample.corr(method="spearman", min_periods=500).fillna(0.0).abs().to_numpy(float)
    parent = list(range(len(fields)))
    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(i: int, j: int) -> None:
        left, right = find(i), find(j)
        if left != right:
            parent[right] = left
    for i in range(len(fields)):
        for j in range(i + 1, len(fields)):
            if corr[i, j] >= .50:
                union(i, j)
    buckets: dict[int, list[str]] = {}
    for i, field in enumerate(fields):
        buckets.setdefault(find(i), []).append(field)
    components = [sorted(names) for names in buckets.values()]
    components.sort(key=lambda names: (-len(names), names))
    return {"source": "first OOF fold train only", "threshold_abs_spearman": .50, "sample_rows": int(len(sample)), "families": {f"family_{number:03d}": names for number, names in enumerate(components)}}


def _fit_score(train: pd.DataFrame, held: pd.DataFrame, fields: tuple[str, ...], seed: int):
    target = rr._primary_target(train, "P8u_floor100_cap250")
    model = rr._fit_target(train, fields, "P8u_floor100_cap250", "main", False, target, seed, HPO_RANKER)
    _, score = model.score(held)
    return model, score


def _mda(args: argparse.Namespace, fields: tuple[str, ...], out: Path) -> None:
    raw_rows: list[dict[str, object]] = []
    family_contract: dict[str, object] | None = None
    for fold_index, held_month in enumerate(args.folds):
        train, held, policy = _window(args, fields, held_month)
        if family_contract is None:
            family_contract = _families(train, fields)
            _write_json(out / "spearman_feature_families.json", family_contract)
        permutation = _within_timestamp_permute(held, fields, held_month)
        family_items = list(family_contract["families"].items())
        for seed in SEEDS:
            model, baseline_score = _fit_score(train, held, fields, seed)
            baseline_pct = _timestamp_percentile(held, baseline_score)
            raw_rows.extend(_metric_rows(held, policy, baseline_score, baseline_pct, label="baseline", held_month=held_month, seed=seed))
            for number, field in enumerate(fields, start=1):
                original = held[field].to_numpy(copy=True)
                held[field] = permutation[field]
                try:
                    _, score = model.score(held)
                    raw_rows.extend(_metric_rows(held, policy, score, baseline_pct, label=f"feature::{field}", held_month=held_month, seed=seed))
                finally:
                    held[field] = original
                if number % 20 == 0:
                    print(json.dumps({"event": "mda_feature_progress", "fold": held_month, "seed": seed, "completed": number, "total": len(fields)}), flush=True)
            for number, (name, family) in enumerate(family_items, start=1):
                originals = {field: held[field].to_numpy(copy=True) for field in family}
                for field in family:
                    held[field] = permutation[field]
                try:
                    _, score = model.score(held)
                    raw_rows.extend(_metric_rows(held, policy, score, baseline_pct, label=f"group::{name}", held_month=held_month, seed=seed))
                finally:
                    for field, values in originals.items():
                        held[field] = values
                if number % 10 == 0:
                    print(json.dumps({"event": "mda_family_progress", "fold": held_month, "seed": seed, "completed": number, "total": len(family_items)}), flush=True)
        print(json.dumps({"event": "mda_fold_complete", "fold": held_month, "train_rows": len(train), "held_rows": len(held)}), flush=True)
    rows = pd.DataFrame(raw_rows)
    rows.to_parquet(out / "mda_timestamp_metrics.parquet", index=False, compression="zstd")
    feature_summary = _wide_summary(rows.loc[rows["label"].eq("baseline") | rows["label"].str.startswith("feature::")], baseline_label="baseline")
    feature_summary = feature_summary.loc[feature_summary["label"].str.startswith("feature::")].copy()
    feature_summary["feature"] = feature_summary["label"].str.removeprefix("feature::")
    feature_summary["mda_loss"] = -feature_summary["mean_delta_selection_score"]
    feature_summary = feature_summary.sort_values("mda_loss", ascending=False, kind="stable").reset_index(drop=True)
    feature_summary.to_parquet(out / "individual_feature_mda.parquet", index=False, compression="zstd")
    family_summary = _wide_summary(rows.loc[rows["label"].eq("baseline") | rows["label"].str.startswith("group::")], baseline_label="baseline")
    family_summary = family_summary.loc[family_summary["label"].str.startswith("group::")].copy()
    family_summary["family"] = family_summary["label"].str.removeprefix("group::")
    family_summary["mda_loss"] = -family_summary["mean_delta_selection_score"]
    family_summary.to_parquet(out / "family_mda.parquet", index=False, compression="zstd")
    baseline = _wide_summary(rows.loc[rows["label"].eq("baseline")], baseline_label="baseline")
    baseline.to_parquet(out / "baseline_oof_metrics.parquet", index=False, compression="zstd")
    # Contracts preserve the canonical base-feature input order, never MDA rank order.
    importance = feature_summary.set_index("feature")["mda_loss"].to_dict()
    ranked = sorted(fields, key=lambda field: (-float(importance.get(field, -np.inf)), field))
    contracts = {"120": list(fields)}
    contracts.update({str(n): [field for field in fields if field in set(ranked[:n])] for n in FEATURE_COUNTS})
    _write_json(out / "mda_feature_contracts.json", {"schema": SCHEMA, "base_feature_hash": rr.FEATURE_HASH, "ranked_by": "strict OOF within-timestamp permutation loss", "contracts": contracts})
    _write_json(out / "mda_manifest.json", {"schema": SCHEMA, "status": "complete_mda", "folds": list(args.folds), "seeds": list(SEEDS), "primary_target": "P8u_floor100_cap250", "ranker": HPO_RANKER, "selection_unit": "equal timestamp", "held_score_contract": "target-free; policy joined only inside metric calculation", "permutation": "within timestamp query", "feature_count": len(fields), "families": len(family_contract["families"]), "outputs": ["baseline_oof_metrics.parquet", "individual_feature_mda.parquet", "family_mda.parquet", "mda_feature_contracts.json"]})


def _subset_rows(args: argparse.Namespace, fields: tuple[str, ...], held_month: str, seed: int, label: str) -> list[dict[str, object]]:
    train, held, policy = _window(args, fields, held_month)
    _, score = _fit_score(train, held, fields, seed)
    pct = _timestamp_percentile(held, score)
    return _metric_rows(held, policy, score, pct, label=label, held_month=held_month, seed=seed)


def _subsets(args: argparse.Namespace, base_fields: tuple[str, ...], out: Path) -> None:
    contracts = json.loads((out / "mda_feature_contracts.json").read_text())["contracts"]
    rows: list[dict[str, object]] = []
    for count, selected in contracts.items():
        fields = tuple(field for field in base_fields if field in set(selected))
        for held_month in args.folds:
            for seed in SEEDS:
                rows.extend(_subset_rows(args, fields, held_month, seed, f"subset_{count}"))
            print(json.dumps({"event": "subset_fold_complete", "subset": count, "fold": held_month}), flush=True)
    frame = pd.DataFrame(rows)
    frame.to_parquet(out / "subset_timestamp_metrics.parquet", index=False, compression="zstd")
    summary = _wide_summary(frame, baseline_label="subset_120")
    summary.to_parquet(out / "subset_oof_metrics.parquet", index=False, compression="zstd")
    winner = summary.loc[summary["label"].ne("subset_120")].iloc[0]
    count = winner["label"].removeprefix("subset_")
    selected = [field for field in base_fields if field in set(contracts[count])]
    _write_json(out / "selected_compact_feature_contract.json", {"schema": SCHEMA, "status": "compact_development_winner", "selection_score": float(winner["selection_score"]), "selected_count": len(selected), "feature_contract": selected, "feature_contract_sha256": rr._feature_hash(selected), "source": "strict OOF subset comparison; add-back still pending"})
    print(json.dumps({"event": "subsets_complete", "winner": count, "selection_score": float(winner["selection_score"])}), flush=True)


def _addback(args: argparse.Namespace, base_fields: tuple[str, ...], out: Path) -> None:
    compact = json.loads((out / "selected_compact_feature_contract.json").read_text())["feature_contract"]
    families = json.loads((out / "spearman_feature_families.json").read_text())["families"]
    family_summary = pd.read_parquet(out / "family_mda.parquet").sort_values("mda_loss", ascending=False, kind="stable")
    chosen_name = None
    extra: list[str] = []
    selected_set = set(compact)
    for name in family_summary["family"].astype(str):
        candidate = [field for field in families[name] if field not in selected_set]
        if candidate:
            chosen_name, extra = name, candidate
            break
    if chosen_name is None:
        raise AssertionError("no unselected family is available for required group add-back")
    arms = {"compact": compact, f"compact_plus_{chosen_name}": [field for field in base_fields if field in selected_set | set(extra)]}
    rows: list[dict[str, object]] = []
    for label, selected in arms.items():
        fields = tuple(selected)
        for held_month in args.folds:
            for seed in SEEDS:
                rows.extend(_subset_rows(args, fields, held_month, seed, label))
            print(json.dumps({"event": "addback_fold_complete", "arm": label, "fold": held_month}), flush=True)
    frame = pd.DataFrame(rows)
    frame.to_parquet(out / "addback_timestamp_metrics.parquet", index=False, compression="zstd")
    summary = _wide_summary(frame, baseline_label="compact")
    summary.to_parquet(out / "addback_oof_metrics.parquet", index=False, compression="zstd")
    winner = summary.iloc[0]
    winner_fields = arms[str(winner["label"])]
    _write_json(out / "selected_router_feature_contract.json", {"schema": SCHEMA, "status": "development feature winner; full-ledger and untouched validation pending", "selection_score": float(winner["selection_score"]), "source_arm": str(winner["label"]), "added_family": chosen_name if str(winner["label"]) != "compact" else None, "feature_contract": winner_fields, "feature_contract_sha256": rr._feature_hash(winner_fields), "base_feature_contract_sha256": rr.FEATURE_HASH})
    (out / "MDA_ROUTER_REPORT.md").write_text("\n".join([
        "# Strict-R3 router feature-selection receipt", "",
        "Research-only. Held scores were target-free, every evaluation quantity is timestamp-averaged, and MDA permutations were within held timestamp queries.", "",
        f"Selected development arm: `{winner['label']}` ({len(winner_fields)} fields).", "",
        "A full chronological ledger and later untouched-period test remain required before promotion.", "",
    ]))
    manifest = json.loads((out / "mda_manifest.json").read_text())
    manifest.update({"status": "complete", "compact_winner": str(winner["label"]), "addback_family_tested": chosen_name, "final_feature_count": len(winner_fields)})
    (out / "mda_manifest.json").unlink()
    _write_json(out / "mda_manifest.json", manifest)
    print(json.dumps({"event": "addback_complete", "winner": str(winner["label"]), "feature_count": len(winner_fields)}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("mda", "subsets", "addback", "all"), default="all")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, default=rr.DEFAULT_FEATURE_ROOT)
    parser.add_argument("--aux-root", type=Path, default=ROOT / "data_perp/artifacts/strict_r3_o3v2_recall_router_aux_labels_20250401_20260731_extatr_20260825_v1")
    parser.add_argument("--policy-path", type=Path, default=rr.DEFAULT_POLICY)
    parser.add_argument("--bundle", type=Path, default=rr.DEFAULT_BUNDLE)
    parser.add_argument("--folds", type=_parse_tokens, default=FOLDS)
    parser.add_argument("--train-months", type=int, default=6)
    parser.add_argument("--reserve-days", type=int, default=28)
    parser.add_argument("--train-cap", type=int, default=240_000)
    args = parser.parse_args()
    if args.train_months < 2 or args.reserve_days < 12 or args.train_cap < 20_000:
        raise ValueError("invalid strict historical support")
    fields = rr._base_fields(args.bundle)
    if args.stage in {"mda", "all"}:
        if args.out.exists():
            raise FileExistsError(args.out)
        args.out.mkdir(parents=True)
        _mda(args, fields, args.out)
    if args.stage in {"subsets", "all"}:
        _subsets(args, fields, args.out)
    if args.stage in {"addback", "all"}:
        _addback(args, fields, args.out)


if __name__ == "__main__":
    main()
