#!/usr/bin/env python3
"""Strict-OOF state-Meta probe for target-free market-state representations.

The market-state lattice is timestamp-global.  It must therefore not be
treated as a standalone cross-sectional candidate ranker: that would merely
tie every candidate within a decision timestamp.  This probe gives it the
role it can causally support: predicting the policy-residual quality of the
two candidates already selected by the target-free Base rank at that
timestamp.  It retains the requested shallow random-subspace, pair-synergy,
and beam-search funnel, but measures timestamp selection rather than a
spurious within-timestamp ordering.

This is historical research only.  It neither mutates the P8U contract nor
opens any admission, portfolio, live, or exchange path.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from scipy.stats import spearmanr

import materialize_strict_r3_p8u_meta_base_state_v1 as base_state
import screen_strict_r3_p8u_market_state_transition_v1 as screen


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "strict_r3_p8u_market_state_timestamp_synergy_probe_v1"
SEED = 1729
BASE_CONTEXT = (
    "state_base_top1_rank",
    "state_base_top2_rank_mean",
    "state_base_top1_score",
    "state_base_top2_score_mean",
    "state_base_top2_score_gap",
    "state_base_top20_score_iqr",
    "state_base_top20_candidate_n",
)


def _once(path: Path, payload: object) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def _month(token: str) -> pd.Timestamp:
    return pd.Timestamp(f"{token}-01", tz="UTC")


def _rank_desc(frame: pd.DataFrame, column: str) -> np.ndarray:
    order = frame.loc[:, ["__decision_ts__", column]].copy()
    order["row"] = np.arange(len(order))
    order = order.sort_values([column, "__decision_ts__"], ascending=[False, True], kind="stable")
    rank = np.arange(len(order), dtype=float) + 1.0
    output = np.empty(len(order), dtype=np.float32)
    output[order.row.to_numpy(int)] = (1.0 - (rank - .5) / len(order)).astype(np.float32)
    return output


def _timestamp_inputs(base: pd.DataFrame, state: pd.DataFrame) -> pd.DataFrame:
    """Create target-free Base descriptors then exact join the state lattice."""
    work = base.loc[base.base_rank_ts.ge(screen.TOP20_START), [
        "candidate_id", "__decision_ts__", "base_rank_ts", "base_score",
    ]].copy()
    if work.empty:
        raise AssertionError("no Base top-20% rows")
    work = work.sort_values(["__decision_ts__", "base_rank_ts", "candidate_id"], ascending=[True, False, True], kind="stable")
    work["position"] = work.groupby("__decision_ts__", sort=False).cumcount()
    top = work.loc[work.position.lt(2)].copy()
    top1 = top.loc[top.position.eq(0)].set_index("__decision_ts__")
    top2 = top.groupby("__decision_ts__", sort=True).agg(
        state_base_top2_rank_mean=("base_rank_ts", "mean"),
        state_base_top2_score_mean=("base_score", "mean"),
    )
    pool = work.groupby("__decision_ts__", sort=True).agg(
        state_base_top20_score_iqr=("base_score", lambda value: float(value.quantile(.75) - value.quantile(.25))),
        state_base_top20_candidate_n=("candidate_id", "size"),
    )
    output = top2.join(pool, how="inner")
    output["state_base_top1_rank"] = top1.base_rank_ts
    output["state_base_top1_score"] = top1.base_score
    output["state_base_top2_score_gap"] = top1.base_score - output.state_base_top2_score_mean
    output = output.reset_index()
    result = output.merge(state, on="__decision_ts__", how="inner", validate="one_to_one")
    if len(result) != len(output):
        raise AssertionError("timestamp state lattice is missing Base decisions")
    return result.sort_values("__decision_ts__", kind="stable").reset_index(drop=True)


def _timestamp_labels(base: pd.DataFrame, policy: pd.DataFrame) -> pd.DataFrame:
    """Join policy labels only after target-free timestamp inputs are fixed."""
    events = base_state._policy_events(base, policy).merge(
        base.loc[:, ["candidate_id", "base_rank_ts"]], on="candidate_id", how="left", validate="one_to_one",
    )
    events = events.loc[events.base_rank_ts.ge(screen.TOP20_START)].copy()
    events = events.sort_values(["__decision_ts__", "base_rank_ts", "candidate_id"], ascending=[True, False, True], kind="stable")
    events["position"] = events.groupby("__decision_ts__", sort=False).cumcount()
    top = events.loc[events.position.lt(2)]
    labels = top.groupby("__decision_ts__", sort=True).agg(
        label_available_ts=("available", "max"),
        top2_residual_bps=("residual_bps", "mean"),
        top2_policy_net_bps=("policy_net_bps", "mean"),
        label_top2_n=("candidate_id", "size"),
    ).reset_index()
    return labels.loc[labels.label_top2_n.eq(2)].drop(columns="label_top2_n")


def _load_frame(state_root: Path, screen_root: Path, early_base: Path, later_base: Path, policy_path: Path) -> tuple[pd.DataFrame, list[str]]:
    receipt = json.loads((state_root / "correctness_report.json").read_text())
    if not all(value is True or key in {"schema", "fast_slow_pairs_predeclared"} for key, value in receipt.items()):
        raise AssertionError("market-state representation receipt is not clean")
    selected = pd.read_parquet(screen_root / "selected_top30_preprobe.parquet")
    states = selected.feature.tolist()
    state = pd.read_parquet(state_root / "market_state_hourly.parquet", columns=["__decision_ts__", *states])
    state["__decision_ts__"] = pd.to_datetime(state["__decision_ts__"], utc=True, errors="raise")
    start = pd.Timestamp(year=state.__decision_ts__.min().year, month=state.__decision_ts__.min().month, day=1, tz="UTC")
    finish = state.__decision_ts__.max() + pd.Timedelta(hours=1)
    end = pd.Timestamp(year=finish.year, month=finish.month, day=1, tz="UTC")
    base = screen._read_base(early_base, later_base, start, end)
    inputs = _timestamp_inputs(base, state)
    policy = screen._read_policy(policy_path)
    labels = _timestamp_labels(base, policy)
    # Inputs are materialised and identity-checked before the outcome join.
    frame = inputs.merge(labels, on="__decision_ts__", how="inner", validate="one_to_one")
    if frame.loc[:, states].isna().all(axis=None) or not frame.label_available_ts.le(frame.__decision_ts__ + pd.Timedelta(days=30)).all():
        raise AssertionError("state coverage or label timestamp failure")
    return frame.sort_values("__decision_ts__", kind="stable").reset_index(drop=True), states


@dataclass(frozen=True)
class Fold:
    token: str
    train_index: np.ndarray
    held_index: np.ndarray


class Probe:
    def __init__(self, frame: pd.DataFrame, states: Sequence[str], held_months: Sequence[str], *, seed: int, max_train_rows: int) -> None:
        self.frame = frame.reset_index(drop=True)
        self.states = tuple(states)
        self.seed = int(seed)
        self.max_train_rows = int(max_train_rows)
        self.folds = self._folds(tuple(held_months))
        self.cache: dict[tuple[str, ...], tuple[dict[str, object], pd.DataFrame]] = {}

    def _folds(self, held_months: tuple[str, ...]) -> tuple[Fold, ...]:
        folds: list[Fold] = []
        for token in held_months:
            start = _month(token); end = start + pd.offsets.MonthBegin(1)
            train = np.flatnonzero((self.frame.__decision_ts__.lt(start) & self.frame.label_available_ts.lt(start)).to_numpy())
            held = np.flatnonzero((self.frame.__decision_ts__.ge(start) & self.frame.__decision_ts__.lt(end)).to_numpy())
            if len(train) >= 500 and len(held) >= 200:
                folds.append(Fold(token, train, held))
        if len(folds) < 3:
            raise AssertionError("need three strict-OOF timestamp-state folds")
        return tuple(folds)

    def _sample(self, values: np.ndarray, fold_index: int) -> np.ndarray:
        if len(values) <= self.max_train_rows:
            return values
        rng = np.random.default_rng(self.seed + fold_index * 1013)
        return np.sort(rng.choice(values, size=self.max_train_rows, replace=False))

    @staticmethod
    def _economic(held: pd.DataFrame, score: np.ndarray) -> tuple[float, int]:
        work = held.loc[:, ["__decision_ts__", "top2_policy_net_bps"]].copy()
        work["score"] = score
        count = max(1, int(math.ceil(len(work) * .20)))
        chosen = work.sort_values(["score", "__decision_ts__"], ascending=[False, True], kind="stable").head(count)
        return float(chosen.top2_policy_net_bps.mean() - work.top2_policy_net_bps.mean()), count

    def _one(self, subset: tuple[str, ...], fold: Fold, fold_index: int) -> tuple[dict[str, object], pd.DataFrame]:
        train = self.frame.iloc[self._sample(fold.train_index, fold_index)]
        held = self.frame.iloc[fold.held_index].copy()
        columns = [*BASE_CONTEXT, *subset]
        x_train = train.loc[:, columns].replace([np.inf, -np.inf], np.nan)
        x_held = held.loc[:, columns].replace([np.inf, -np.inf], np.nan)
        target = train.top2_residual_bps.clip(-500., 500.).to_numpy(float)
        depth = 2 if (fold_index + len(subset)) % 2 == 0 else 3
        model = LGBMRegressor(
            objective="huber", n_estimators=150, learning_rate=.045, max_depth=depth,
            num_leaves=7 if depth == 2 else 15, min_child_samples=96,
            min_split_gain=.002, feature_fraction=.85, bagging_fraction=.80,
            reg_lambda=12.0, reg_alpha=.05, random_state=self.seed + fold_index,
            n_jobs=1, verbosity=-1,
        )
        model.fit(x_train, target)
        prediction = model.predict(x_held)
        ic = float(spearmanr(prediction, held.top2_residual_bps).statistic)
        # A constant held prediction has no rank information.  Treat its IC
        # as zero rather than allowing pandas' mean to silently drop the
        # entire held month from a portability aggregate.
        if not np.isfinite(ic):
            ic = 0.0
        spread, selected = self._economic(held, prediction)
        return {
            "fold": fold.token, "residual_ic": ic, "economic_spread_bps": spread,
            "selected_timestamps": selected, "probe_score": .5 * ic + .5 * spread / 100.,
        }, held.assign(state_prediction=prediction)

    def evaluate(self, subset: Iterable[str], *, retain_predictions: bool = False) -> tuple[dict[str, object], pd.DataFrame]:
        key = tuple(sorted(subset))
        if key in self.cache:
            cached, predictions = self.cache[key]
            return cached, predictions.copy() if retain_predictions else pd.DataFrame()
        rows: list[dict[str, object]] = []; predictions: list[pd.DataFrame] = []
        for index, fold in enumerate(self.folds):
            row, values = self._one(key, fold, index); rows.append(row)
            if retain_predictions:
                values["fold"] = fold.token; predictions.append(values)
        folds = pd.DataFrame(rows)
        result: dict[str, object] = {
            "features": "|".join(key), "feature_count": len(key),
            "mean_probe_score": float(folds.probe_score.mean()), "mean_residual_ic": float(folds.residual_ic.mean()),
            "mean_economic_spread_bps": float(folds.economic_spread_bps.mean()),
            "positive_economic_folds": int(folds.economic_spread_bps.gt(0).sum()),
            "worst_economic_spread_bps": float(folds.economic_spread_bps.min()), "fold_count": int(len(folds)),
            "fold_metrics": folds.to_dict(orient="records"),
        }
        self.cache[key] = (result, pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame())
        return result, self.cache[key][1].copy() if retain_predictions else pd.DataFrame()


def _random_subspaces(features: Sequence[str], *, count: int = 80) -> list[tuple[str, ...]]:
    rng = np.random.default_rng(SEED); output: set[tuple[str, ...]] = set()
    while len(output) < count:
        width = int(rng.integers(3, min(9, len(features) + 1)))
        output.add(tuple(sorted(rng.choice(features, size=width, replace=False).tolist())))
    return sorted(output)


def _parallel(probe: Probe, subsets: Sequence[tuple[str, ...]], workers: int) -> list[dict[str, object]]:
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        return [future.result()[0] for future in [pool.submit(probe.evaluate, subset) for subset in subsets]]


def _inclusion(rows: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    out = []
    for feature in features:
        included = rows.features.str.split("|").map(lambda values: feature in values if values != [""] else False)
        yes, no = rows.loc[included, "mean_probe_score"], rows.loc[~included, "mean_probe_score"]
        out.append({"feature": feature, "included_trials": len(yes), "excluded_trials": len(no), "inclusion_uplift": float(yes.mean() - no.mean())})
    return pd.DataFrame(out).sort_values(["inclusion_uplift", "feature"], ascending=[False, True], kind="stable")


def _pairs(probe: Probe, states: Sequence[str], base: dict[str, object], workers: int) -> pd.DataFrame:
    singles = {state: probe.evaluate((state,))[0] for state in states}
    pairs = list(combinations(states, 2)); both = _parallel(probe, [tuple(pair) for pair in pairs], workers)
    base_fold = {str(row["fold"]): float(row["probe_score"]) for row in base["fold_metrics"]}
    rows = []
    for (left, right), result in zip(pairs, both, strict=True):
        both_fold = {str(row["fold"]): float(row["probe_score"]) for row in result["fold_metrics"]}
        left_fold = {str(row["fold"]): float(row["probe_score"]) for row in singles[left]["fold_metrics"]}
        right_fold = {str(row["fold"]): float(row["probe_score"]) for row in singles[right]["fold_metrics"]}
        synergy = [both_fold[key] - left_fold[key] - right_fold[key] + base_fold[key] for key in both_fold]
        rows.append({"left": left, "right": right, "pair": f"{left}|{right}", "score_base": base["mean_probe_score"],
                     "score_pair": result["mean_probe_score"], "delta_base": float(result["mean_probe_score"] - base["mean_probe_score"]),
                     "synergy": float(np.mean(synergy)), "positive_synergy_folds": int(np.sum(np.asarray(synergy) > 0)),
                     "worst_synergy": float(np.min(synergy))})
    return pd.DataFrame(rows).sort_values(["delta_base", "synergy", "pair"], ascending=[False, False, True], kind="stable")


def _beam(probe: Probe, candidates: Sequence[str], pairs: pd.DataFrame, base_score: float, *, width: int = 6) -> pd.DataFrame:
    starts = [tuple(sorted(row)) for row in pairs.loc[pairs.delta_base.gt(0)].head(width)[["left", "right"]].itertuples(index=False, name=None)]
    if not starts:
        starts = [(value,) for value in candidates[:width]]
    seen = set(starts); beam = starts; output = []
    for size in range(min(map(len, beam)), min(8, len(candidates)) + 1):
        proposals = list(beam) if size == min(map(len, beam)) else []
        if size > min(map(len, starts)):
            for current in beam:
                for extra in candidates:
                    proposal = tuple(sorted(set(current) | {extra}))
                    if len(proposal) == size and proposal not in seen:
                        proposals.append(proposal); seen.add(proposal)
        rows = [probe.evaluate(value)[0] for value in sorted(set(proposals))]
        table = pd.DataFrame(rows)
        table = table.loc[(table.mean_probe_score.gt(base_score)) & table.positive_economic_folds.ge(math.ceil(table.fold_count.iloc[0] * .6))]
        if table.empty:
            break
        table = table.sort_values(["mean_probe_score", "worst_economic_spread_bps", "features"], ascending=[False, False, True], kind="stable")
        output.append(table.head(width).assign(block_size=size)); beam = [tuple(value.split("|")) for value in table.head(width).features]
    return pd.concat(output, ignore_index=True) if output else pd.DataFrame()


def _write(root: Path, *, base: dict[str, object], subspace: pd.DataFrame, inclusion: pd.DataFrame, singles: pd.DataFrame, pairs: pd.DataFrame, beam: pd.DataFrame, predictions: pd.DataFrame, source: dict[str, object]) -> None:
    root.mkdir(parents=True, exist_ok=False)
    pd.DataFrame([base]).to_parquet(root / "base_context_probe.parquet", index=False)
    subspace.to_parquet(root / "random_subspace_probes.parquet", index=False)
    inclusion.to_parquet(root / "inclusion_uplift.parquet", index=False)
    singles.to_parquet(root / "single_feature_probes.parquet", index=False)
    pairs.to_parquet(root / "pair_synergy.parquet", index=False)
    beam.to_parquet(root / "beam_blocks.parquet", index=False)
    predictions.to_parquet(root / "best_state_block_oof_predictions.parquet", index=False)
    correctness = {
        "schema": SCHEMA, "state_inputs_target_free": True, "base_context_target_free": True,
        "policy_labels_joined_only_after_timestamp_input_identity": True,
        "training_labels_resolved_before_each_held_month": True,
        "held_prediction_matrix_excludes_held_labels": True,
        "held_prediction_constructed_before_held_label_metrics": True,
        "state_role_is_timestamp_calibration_not_standalone_candidate_rank": True,
        "no_meta_mc1_admission_portfolio_live_or_exchange_mutation": True,
    }
    _once(root / "correctness_report.json", correctness)
    _once(root / "run_manifest.json", {"schema": SCHEMA, "scope": "offline strict-OOF timestamp State Meta probe", "source": source, "correctness": correctness})


def _write_confirmation(root: Path, *, base: dict[str, object], challenger: dict[str, object], base_predictions: pd.DataFrame, challenger_predictions: pd.DataFrame, source: dict[str, object]) -> None:
    """Write a frozen-block OOS confirmation without reselecting features."""
    root.mkdir(parents=True, exist_ok=False)
    # These are intentionally target-free score receipts.  The compact fold
    # metric dictionaries are derived after the state predictions; realised
    # outcome columns never persist in the OOS score files.
    for name, values in (("base", base_predictions), ("state", challenger_predictions)):
        prediction_column = "state_prediction" if "state_prediction" in values else "probe_prediction"
        columns = ["__decision_ts__", "fold", prediction_column]
        score = values.loc[:, [column for column in columns if column in values]].copy()
        score = score.rename(columns={prediction_column: "state_prediction"})
        score.to_parquet(root / f"target_free_{name}_timestamp_scores.parquet", index=False)
    pd.DataFrame([base, challenger]).to_parquet(root / "confirmation_metrics.parquet", index=False)
    correctness = {
        "schema": SCHEMA, "frozen_pre2026_feature_block_only": True,
        "state_inputs_target_free": True, "base_context_target_free": True,
        "training_labels_resolved_before_each_held_month": True,
        "held_prediction_matrix_excludes_held_labels": True,
        "held_prediction_constructed_before_held_label_metrics": True,
        "no_reselection_on_2026": True,
        "no_meta_mc1_admission_portfolio_live_or_exchange_mutation": True,
    }
    _once(root / "correctness_report.json", correctness)
    _once(root / "run_manifest.json", {"schema": SCHEMA, "scope": "offline frozen-block 2026 State Meta confirmation", "source": source, "correctness": correctness})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-root", required=True); parser.add_argument("--screen-root", required=True)
    parser.add_argument("--early-base-root", required=True); parser.add_argument("--later-base-root", required=True)
    parser.add_argument("--policy-labels", required=True); parser.add_argument("--held-months", default="2025-05,2025-06,2025-07,2025-08,2025-09,2025-10,2025-11,2025-12")
    parser.add_argument("--workers", type=int, default=4); parser.add_argument("--max-train-rows", type=int, default=6000); parser.add_argument("--out", required=True)
    parser.add_argument("--frozen-contract", help="JSON pre-2026 state-block contract; enables confirmation-only mode")
    args = parser.parse_args()
    state, screen_root, early, later, policy, out = (ROOT / args.state_root, ROOT / args.screen_root, ROOT / args.early_base_root, ROOT / args.later_base_root, ROOT / args.policy_labels, ROOT / args.out)
    if out.exists(): raise FileExistsError(out)
    frame, states = _load_frame(state, screen_root, early, later, policy)
    frozen_features: tuple[str, ...] | None = None
    if args.frozen_contract:
        payload = json.loads((ROOT / args.frozen_contract).read_text())
        frozen_features = tuple(str(value) for value in payload.get("selected_features", ()))
        if len(frozen_features) < 1 or len(frozen_features) != len(set(frozen_features)) or not set(frozen_features).issubset(states):
            raise AssertionError("invalid frozen market-state feature contract")
    probe = Probe(frame, states, tuple(item.strip() for item in args.held_months.split(",") if item.strip()), seed=SEED, max_train_rows=args.max_train_rows)
    base, _ = probe.evaluate(())
    if frozen_features is not None:
        base, base_predictions = probe.evaluate((), retain_predictions=True)
        challenger, challenger_predictions = probe.evaluate(frozen_features, retain_predictions=True)
        _write_confirmation(out, base=base, challenger=challenger, base_predictions=base_predictions, challenger_predictions=challenger_predictions, source={
            "state_root": str(state.relative_to(ROOT)), "screen_root": str(screen_root.relative_to(ROOT)), "early_base_root": str(early.relative_to(ROOT)),
            "later_base_root": str(later.relative_to(ROOT)), "policy_labels": str(policy.relative_to(ROOT)), "held_months": args.held_months,
            "workers": args.workers, "max_train_rows": args.max_train_rows, "frozen_contract": str(args.frozen_contract),
        })
        print(json.dumps({"out": str(out), "folds": len(probe.folds), "features": list(frozen_features), "base_score": base["mean_probe_score"], "state_score": challenger["mean_probe_score"]}, sort_keys=True))
        return
    subspace = pd.DataFrame(_parallel(probe, _random_subspaces(states), max(1, args.workers)))
    inclusion = _inclusion(subspace, states)
    singles = pd.DataFrame([probe.evaluate((state,))[0] for state in states])
    pairs = _pairs(probe, states, base, max(1, args.workers))
    candidates = inclusion.head(12).feature.tolist()
    beam = _beam(probe, candidates, pairs.loc[pairs.left.isin(candidates) & pairs.right.isin(candidates)].copy(), float(base["mean_probe_score"]))
    best = tuple(str(beam.iloc[0].features).split("|")) if not beam.empty else ()
    _, predictions = probe.evaluate(best, retain_predictions=True)
    _write(out, base=base, subspace=subspace, inclusion=inclusion, singles=singles, pairs=pairs, beam=beam, predictions=predictions, source={
        "state_root": str(state.relative_to(ROOT)), "screen_root": str(screen_root.relative_to(ROOT)), "early_base_root": str(early.relative_to(ROOT)),
        "later_base_root": str(later.relative_to(ROOT)), "policy_labels": str(policy.relative_to(ROOT)), "held_months": args.held_months,
        "workers": args.workers, "max_train_rows": args.max_train_rows,
    })
    print(json.dumps({"out": str(out), "folds": len(probe.folds), "states": len(states), "best": best, "base_score": base["mean_probe_score"]}, sort_keys=True))


if __name__ == "__main__":
    main()
