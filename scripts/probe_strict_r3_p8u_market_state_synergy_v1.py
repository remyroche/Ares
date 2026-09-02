#!/usr/bin/env python3
"""Shallow strict-OOF subspace, pair-synergy, and beam probes for market state.

This is deliberately a *research probe*, not a replacement Meta model.  It
answers whether timestamp-global market transitions improve the conversion of
the Base top-20% region into the best two opportunities per timestamp.  Every
fit uses only labels resolved before its held month; held policy net is used
only for the residual IC and economic-spread metrics.
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

import screen_strict_r3_p8u_market_state_transition_v1 as screen
import materialize_strict_r3_p8u_meta_base_state_v1 as base_state


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "strict_r3_p8u_market_state_synergy_probe_v1"
SEED = 1729


def _once(path: Path, payload: object) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    members = sorted(path.rglob("*.parquet")) if path.is_dir() else [path]
    for member in members:
        digest.update(str(member).encode())
        with member.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    return digest.hexdigest()


def _month(token: str) -> pd.Timestamp:
    return pd.Timestamp(f"{token}-01", tz="UTC")


def _rank_desc(frame: pd.DataFrame, column: str) -> np.ndarray:
    order = frame.loc[:, ["candidate_id", "__decision_ts__", column]].copy()
    order["row"] = np.arange(len(order))
    order = order.sort_values(["__decision_ts__", column, "candidate_id"], ascending=[True, False, True], kind="stable")
    rank = order.groupby("__decision_ts__", sort=False).cumcount().to_numpy(float) + 1.0
    size = order.groupby("__decision_ts__", sort=False).candidate_id.transform("size").to_numpy(float)
    output = np.empty(len(order), dtype=np.float32)
    output[order.row.to_numpy(int)] = (1.0 - (rank - .5) / size).astype(np.float32)
    return output


@dataclass(frozen=True)
class Fold:
    token: str
    train_index: np.ndarray
    held_index: np.ndarray


class Probe:
    def __init__(self, frame: pd.DataFrame, features: Sequence[str], held_months: Sequence[str], *, seed: int, max_train_rows: int) -> None:
        self.frame = frame.reset_index(drop=True)
        self.features = tuple(features)
        self.seed = int(seed)
        self.max_train_rows = int(max_train_rows)
        self.folds = self._folds(tuple(held_months))
        self.cache: dict[tuple[str, ...], tuple[dict[str, float], pd.DataFrame]] = {}

    def _folds(self, held_months: tuple[str, ...]) -> tuple[Fold, ...]:
        folds: list[Fold] = []
        decision = self.frame.__decision_ts__
        available = self.frame.available
        for token in held_months:
            held_start = _month(token)
            held_end = held_start + pd.offsets.MonthBegin(1)
            train = np.flatnonzero((decision < held_start).to_numpy() & (available < held_start).to_numpy())
            held = np.flatnonzero((decision >= held_start).to_numpy() & (decision < held_end).to_numpy())
            if len(train) < 25_000 or len(held) < 1_000:
                continue
            folds.append(Fold(token, train, held))
        if len(folds) < 3:
            raise AssertionError("need at least three strict-OOF probe folds")
        return tuple(folds)

    def _sample_train(self, index: np.ndarray, fold_index: int) -> np.ndarray:
        if len(index) <= self.max_train_rows:
            return index
        rng = np.random.default_rng(self.seed + fold_index * 1009)
        # Uniform row sampling is deliberately conservative here: timestamp
        # weighting belongs to the later Meta fit, not the representation probe.
        return np.sort(rng.choice(index, size=self.max_train_rows, replace=False))

    @staticmethod
    def _economic(frame: pd.DataFrame, score: np.ndarray) -> tuple[float, float, int]:
        work = frame.loc[:, ["candidate_id", "__decision_ts__", "policy_net_bps", "residual_bps"]].copy()
        work["score"] = score
        work = work.sort_values(["__decision_ts__", "score", "candidate_id"], ascending=[True, False, True], kind="stable")
        work["position"] = work.groupby("__decision_ts__", sort=False).cumcount()
        selected = work.loc[work.position.lt(2)]
        selected_ev = selected.groupby("__decision_ts__", sort=False).policy_net_bps.mean()
        pool_ev = work.groupby("__decision_ts__", sort=False).policy_net_bps.mean()
        selected_residual = selected.groupby("__decision_ts__", sort=False).residual_bps.mean()
        pool_residual = work.groupby("__decision_ts__", sort=False).residual_bps.mean()
        return float((selected_ev - pool_ev).mean()), float((selected_residual - pool_residual).mean()), int(len(selected))

    def _one(self, subset: tuple[str, ...], fold: Fold, fold_index: int) -> tuple[dict[str, float], pd.DataFrame]:
        held = self.frame.iloc[fold.held_index].copy()
        if not subset:
            score = held.base_rank_ts.to_numpy(float)
            ic = float(spearmanr(score, held.residual_bps).statistic)
            ev_spread, residual_spread, selected = self._economic(held, score)
            return {"fold": fold.token, "residual_ic": ic, "economic_spread_bps": ev_spread, "residual_spread_bps": residual_spread, "selected_rows": selected, "probe_score": .5 * ic + .5 * ev_spread / 100.0}, held.assign(probe_prediction=score, probe_score=score)
        train_index = self._sample_train(fold.train_index, fold_index)
        train = self.frame.iloc[train_index]
        columns = ["base_rank_ts", "base_score", *subset]
        x_train = train.loc[:, columns].replace([np.inf, -np.inf], np.nan)
        x_held = held.loc[:, columns].replace([np.inf, -np.inf], np.nan)
        target = train.residual_bps.clip(-500.0, 500.0).to_numpy(float)
        depth = 2 if (fold_index + len(subset)) % 2 == 0 else 3
        model = LGBMRegressor(
            objective="huber", n_estimators=150, learning_rate=.045, max_depth=depth,
            num_leaves=7 if depth == 2 else 15, min_child_samples=750,
            min_split_gain=.002, feature_fraction=.85, bagging_fraction=.80,
            reg_lambda=12.0, reg_alpha=.05, random_state=self.seed + fold_index,
            n_jobs=1, verbosity=-1,
        )
        model.fit(x_train, target)
        prediction = model.predict(x_held)
        meta_rank = _rank_desc(held.assign(__probe__=prediction), "__probe__")
        score = .75 * held.base_rank_ts.to_numpy(float) + .25 * meta_rank
        ic = float(spearmanr(prediction, held.residual_bps).statistic)
        ev_spread, residual_spread, selected = self._economic(held, score)
        return {"fold": fold.token, "residual_ic": ic, "economic_spread_bps": ev_spread, "residual_spread_bps": residual_spread, "selected_rows": selected, "probe_score": .5 * ic + .5 * ev_spread / 100.0}, held.assign(probe_prediction=prediction, probe_score=score)

    def evaluate(self, subset: Iterable[str], *, retain_predictions: bool = False) -> tuple[dict[str, float], pd.DataFrame]:
        key = tuple(sorted(subset))
        if key in self.cache:
            result, predictions = self.cache[key]
            return result, predictions.copy() if retain_predictions else pd.DataFrame()
        rows: list[dict[str, float]] = []
        predictions: list[pd.DataFrame] = []
        for index, fold in enumerate(self.folds):
            row, prediction = self._one(key, fold, index)
            rows.append(row)
            if retain_predictions:
                prediction["fold"] = fold.token
                predictions.append(prediction)
        fold_frame = pd.DataFrame(rows)
        result = {
            "features": "|".join(key), "feature_count": len(key),
            "mean_probe_score": float(fold_frame.probe_score.mean()),
            "mean_residual_ic": float(fold_frame.residual_ic.mean()),
            "mean_economic_spread_bps": float(fold_frame.economic_spread_bps.mean()),
            "mean_residual_spread_bps": float(fold_frame.residual_spread_bps.mean()),
            "positive_score_folds": int(fold_frame.probe_score.gt(0).sum()),
            "positive_economic_folds": int(fold_frame.economic_spread_bps.gt(0).sum()),
            "worst_economic_spread_bps": float(fold_frame.economic_spread_bps.min()),
            "fold_count": int(len(fold_frame)),
            "fold_metrics": fold_frame.to_dict(orient="records"),
        }
        self.cache[key] = (result, pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame())
        return result, self.cache[key][1].copy() if retain_predictions else pd.DataFrame()


def _load_frame(state_root: Path, screen_root: Path, early_base: Path, later_base: Path, policy_path: Path) -> tuple[pd.DataFrame, list[str]]:
    state_receipt = json.loads((state_root / "correctness_report.json").read_text())
    if not all(value is True or key in {"schema", "fast_slow_pairs_predeclared"} for key, value in state_receipt.items()):
        raise AssertionError("state receipt is not clean")
    selected = pd.read_parquet(screen_root / "selected_top30_preprobe.parquet")
    features = selected.feature.tolist()
    state = pd.read_parquet(state_root / "market_state_hourly.parquet", columns=["__decision_ts__", *features])
    state["__decision_ts__"] = pd.to_datetime(state["__decision_ts__"], utc=True, errors="raise")
    start, end = state.__decision_ts__.min(), state.__decision_ts__.max() + pd.Timedelta(hours=1)
    start = pd.Timestamp(year=start.year, month=start.month, day=1, tz="UTC")
    # `_read_base` treats `end` as exclusive.  The hourly state series ends
    # at the final completed observation, so its next calendar-month boundary
    # is already the correct exclusive bound.  Advancing it again requests an
    # unmaterialised future month and, more importantly, obscures the exact
    # historical panel actually used by the probe.
    end = pd.Timestamp(year=end.year, month=end.month, day=1, tz="UTC")
    base = screen._read_base(early_base, later_base, start, end)
    policy = screen._read_policy(policy_path)
    events = base_state._policy_events(base, policy).merge(
        base.loc[:, ["candidate_id", "base_score", "base_rank_ts"]], on="candidate_id", how="left", validate="one_to_one",
    )
    events = events.loc[events.base_rank_ts.ge(screen.TOP20_START)].copy()
    frame = events.merge(state, on="__decision_ts__", how="inner", validate="many_to_one")
    if len(frame) != len(events) or frame.loc[:, features].isna().all(axis=None):
        raise AssertionError("market state / residual identity or coverage failure")
    return frame.sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True), features


def _random_subspaces(features: Sequence[str], *, seed: int, count: int = 80) -> list[tuple[str, ...]]:
    if len(features) <= 12:
        explicit = [tuple(item) for width in range(3, min(8, len(features)) + 1) for item in combinations(features, width)]
        return explicit[:count]
    rng = np.random.default_rng(seed)
    result: set[tuple[str, ...]] = set()
    while len(result) < count:
        width = int(rng.integers(3, min(9, len(features) + 1)))
        result.add(tuple(sorted(rng.choice(features, size=width, replace=False).tolist())))
    return sorted(result)


def _parallel_evaluate(probe: Probe, subsets: Sequence[tuple[str, ...]], workers: int) -> list[dict[str, float]]:
    # LightGBM is pinned to one thread per model; a bounded thread pool lets
    # the independent shallow probes use the available cores without copying
    # the full labelled panel to child processes.
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(probe.evaluate, subset) for subset in subsets]
        return [future.result()[0] for future in futures]


def _inclusion(rows: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    values = []
    for feature in features:
        contains = rows.features.str.split("|").map(lambda x: feature in x if x != [""] else False)
        yes, no = rows.loc[contains, "mean_probe_score"], rows.loc[~contains, "mean_probe_score"]
        values.append({"feature": feature, "included_trials": int(len(yes)), "excluded_trials": int(len(no)), "inclusion_uplift": float(yes.mean() - no.mean()), "included_score": float(yes.mean()), "excluded_score": float(no.mean())})
    return pd.DataFrame(values).sort_values(["inclusion_uplift", "feature"], ascending=[False, True], kind="stable").reset_index(drop=True)


def _pair_synergy(probe: Probe, features: Sequence[str], base_score: float, workers: int) -> pd.DataFrame:
    singles = {feature: probe.evaluate((feature,))[0] for feature in features}
    pairs = list(combinations(features, 2))
    paired_rows = _parallel_evaluate(probe, [tuple(pair) for pair in pairs], workers)
    rows = []
    for (left, right), both in zip(pairs, paired_rows, strict=True):
        per_fold = {item["fold"]: item["probe_score"] for item in both["fold_metrics"]}
        left_fold = {item["fold"]: item["probe_score"] for item in singles[left]["fold_metrics"]}
        right_fold = {item["fold"]: item["probe_score"] for item in singles[right]["fold_metrics"]}
        # Base fold score is included inside the no-state cache to preserve a
        # genuine per-fold synergy, rather than only a pooled arithmetic one.
        base = probe.evaluate(())[0]
        base_fold = {item["fold"]: item["probe_score"] for item in base["fold_metrics"]}
        synergy_folds = [per_fold[key] - left_fold[key] - right_fold[key] + base_fold[key] for key in per_fold]
        rows.append({
            "left": left, "right": right, "pair": f"{left}|{right}",
            "score_base": base_score, "score_left": singles[left]["mean_probe_score"], "score_right": singles[right]["mean_probe_score"],
            "score_pair": both["mean_probe_score"], "synergy": float(np.mean(synergy_folds)),
            "positive_synergy_folds": int(np.sum(np.asarray(synergy_folds) > 0.0)), "fold_count": len(synergy_folds),
            "worst_synergy": float(np.min(synergy_folds)),
        })
    return pd.DataFrame(rows).sort_values(["synergy", "positive_synergy_folds", "pair"], ascending=[False, False, True], kind="stable").reset_index(drop=True)


def _beam(probe: Probe, candidates: Sequence[str], pairs: pd.DataFrame, *, width: int = 6) -> pd.DataFrame:
    starts = [tuple(sorted(item)) for item in pairs.loc[pairs.positive_synergy_folds.ge(max(2, int(pairs.fold_count.iloc[0] * .6)))].head(width)[["left", "right"]].itertuples(index=False, name=None)]
    if not starts:
        starts = [(value,) for value in candidates[:width]]
    seen = set(starts)
    output: list[dict[str, object]] = []
    beam = starts
    for size in range(min(len(item) for item in beam), min(8, len(candidates)) + 1):
        candidates_at_size = beam if size == min(len(item) for item in beam) else []
        if size > min(len(item) for item in starts):
            for current in beam:
                for extra in candidates:
                    proposal = tuple(sorted(set(current) | {extra}))
                    if len(proposal) == size and proposal not in seen:
                        candidates_at_size.append(proposal)
                        seen.add(proposal)
        scored = []
        for proposal in sorted(set(candidates_at_size)):
            result, _ = probe.evaluate(proposal)
            scored.append(result)
        if not scored:
            break
        table = pd.DataFrame(scored)
        table = table.loc[table.positive_economic_folds.ge(math.ceil(table.fold_count.iloc[0] * .6))].copy()
        if table.empty:
            break
        table = table.sort_values(["mean_probe_score", "worst_economic_spread_bps", "features"], ascending=[False, False, True], kind="stable")
        output.append(table.head(width).assign(block_size=size))
        beam = [tuple(item.split("|")) for item in table.head(width).features]
    return pd.concat(output, ignore_index=True) if output else pd.DataFrame()


def _write(root: Path, *, subspace: pd.DataFrame, inclusion: pd.DataFrame, singles: pd.DataFrame, pairs: pd.DataFrame, beam: pd.DataFrame, best_predictions: pd.DataFrame, source: dict[str, str]) -> None:
    root.mkdir(parents=True, exist_ok=False)
    subspace.to_parquet(root / "random_subspace_probes.parquet", index=False)
    inclusion.to_parquet(root / "inclusion_uplift.parquet", index=False)
    singles.to_parquet(root / "single_feature_probes.parquet", index=False)
    pairs.to_parquet(root / "pair_synergy.parquet", index=False)
    beam.to_parquet(root / "beam_blocks.parquet", index=False)
    best_predictions.to_parquet(root / "best_block_oof_predictions.parquet", index=False)
    correctness = {
        "schema": SCHEMA,
        "state_inputs_target_free": True,
        "base_inputs_target_free_before_label_join": True,
        "training_labels_resolved_before_each_held_month": True,
        "held_policy_outcomes_metric_only": True,
        "shallow_probe_depth_2_or_3": True,
        "base_top20_gate_before_probe": True,
        "top2_per_timestamp_economic_metric": True,
        "no_meta_mc1_admission_portfolio_live_or_exchange_mutation": True,
    }
    _once(root / "correctness_report.json", correctness)
    _once(root / "run_manifest.json", {"schema": SCHEMA, "scope": "offline strict-OOF representation probe only", "source": source, "correctness": correctness})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-root", required=True)
    parser.add_argument("--screen-root", required=True)
    parser.add_argument("--early-base-root", required=True)
    parser.add_argument("--later-base-root", required=True)
    parser.add_argument("--policy-labels", required=True)
    parser.add_argument("--held-months", default="2025-05,2025-06,2025-07,2025-08,2025-09,2025-10,2025-11,2025-12")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-train-rows", type=int, default=60000)
    parser.add_argument("--feature-limit", type=int, default=30, help="bounded smoke/debug limit; canonical probe uses all 30")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    state_root, screen_root, early, later, policy, output = (ROOT / args.state_root, ROOT / args.screen_root, ROOT / args.early_base_root, ROOT / args.later_base_root, ROOT / args.policy_labels, ROOT / args.out)
    if output.exists():
        raise FileExistsError(output)
    frame, features = _load_frame(state_root, screen_root, early, later, policy)
    features = features[: int(args.feature_limit)]
    if len(features) < 3:
        raise AssertionError("need at least three state features for the probe")
    held_months = tuple(value.strip() for value in args.held_months.split(",") if value.strip())
    probe = Probe(frame, features, held_months, seed=SEED, max_train_rows=args.max_train_rows)
    base, _ = probe.evaluate(())
    random_rows = _parallel_evaluate(probe, _random_subspaces(features, seed=SEED), max(1, args.workers))
    subspace = pd.DataFrame(random_rows)
    inclusion = _inclusion(subspace, features)
    singles = pd.DataFrame([probe.evaluate((feature,))[0] for feature in features])
    pairs = _pair_synergy(probe, features, base["mean_probe_score"], max(1, args.workers))
    beam_candidates = inclusion.head(12).feature.tolist()
    beam = _beam(probe, beam_candidates, pairs.loc[pairs.left.isin(beam_candidates) & pairs.right.isin(beam_candidates)].copy())
    best_features: tuple[str, ...]
    if beam.empty:
        best_features = tuple(inclusion.head(3).feature)
    else:
        best_features = tuple(str(beam.sort_values(["mean_probe_score", "worst_economic_spread_bps"], ascending=False).iloc[0].features).split("|"))
    _, best_predictions = probe.evaluate(best_features, retain_predictions=True)
    _write(output, subspace=subspace, inclusion=inclusion, singles=singles, pairs=pairs, beam=beam, best_predictions=best_predictions, source={
        "state_root": str(state_root.relative_to(ROOT)), "screen_root": str(screen_root.relative_to(ROOT)), "early_base_root": str(early.relative_to(ROOT)), "later_base_root": str(later.relative_to(ROOT)), "policy_labels": str(policy.relative_to(ROOT)), "held_months": list(held_months), "workers": str(args.workers), "max_train_rows": str(args.max_train_rows),
    })
    print(json.dumps({"out": str(output), "folds": len(probe.folds), "features": len(features), "pairs": len(pairs), "best_features": best_features}, sort_keys=True))


if __name__ == "__main__":
    main()
