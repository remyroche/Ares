#!/usr/bin/env python3
"""Chronological target-specific MDA for the short policy-conversion base.

The October--December 2024 OOS block is deliberately never opened.  Three
earlier chronological folds choose a stable prefix from the 115 target-free
coverage-valid fields for the frozen P1/K32 policy-conversion ranker.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_short_base_target_objective_funnel import _coverage_fields, _feature_fields, _load_candidates, _load_features, _sha256  # noqa: E402
from scripts.run_short_policy_conversion_funnel import (  # noqa: E402
    GAIN_FAMILIES, PolicySpec, _load_policy_ledger, _matrix, _query_order,
    _targets, _valid_policy,
)
from scripts.run_strict_r3_ordinal_base_target_ablation import FROZEN_BASE_PARAMS  # noqa: E402


SEED = 17
FEATURE_SIZES = (15, 30, 60, 90, 115)
PRE_SCREEN_MAX = 72
MDA_REPEATS = 3
TRAIN_CAP = 20_000
VALIDATION_CAP = 8_000
FOLDS = (
    ("2024-05_06", "2023-10-01T00:00:00Z", "2024-05-01T00:00:00Z", "2024-07-01T00:00:00Z"),
    ("2024-07_08", "2023-10-01T00:00:00Z", "2024-07-01T00:00:00Z", "2024-09-01T00:00:00Z"),
    ("2024-09", "2023-10-01T00:00:00Z", "2024-09-01T00:00:00Z", "2024-10-01T00:00:00Z"),
)
SPEC = PolicySpec("P1_policy_bps_k32_linear", "P1 winner", "policy_bps", truncation=32, gain_family="linear")


def _utc(value: str) -> pd.Timestamp:
    result = pd.Timestamp(value)
    return result.tz_localize("UTC") if result.tzinfo is None else result.tz_convert("UTC")


def _sample_queries(frame: pd.DataFrame, cap: int) -> pd.DataFrame:
    """Deterministic time-spread, whole-query sampler."""
    sizes = frame.groupby("__ts__", sort=True).size()
    if len(frame) <= cap:
        return frame.sort_values(["__ts__", "candidate_id"], kind="stable").reset_index(drop=True)
    median = max(1, int(sizes.median()))
    target = max(2, cap // median)
    picks = np.unique(np.linspace(0, len(sizes) - 1, min(target, len(sizes)), dtype=np.int64))
    timestamps = sizes.index.take(picks)
    return frame.loc[frame.__ts__.isin(timestamps)].sort_values(["__ts__", "candidate_id"], kind="stable").reset_index(drop=True)


def _params() -> dict[str, Any]:
    result = dict(FROZEN_BASE_PARAMS)
    result.pop("num_class", None)
    result.update({"objective": "lambdarank", "lambdarank_norm": True, "lambdarank_truncation_level": 32, "label_gain": GAIN_FAMILIES["linear"], "seed": SEED, "random_state": SEED})
    return result


def _eligible(frame: pd.DataFrame, before: pd.Timestamp) -> pd.DataFrame:
    target = _targets(frame, SPEC)
    keep = np.isfinite(target) & frame.policy_label_available_at.lt(before).to_numpy(bool)
    output = frame.loc[keep].copy()
    output["_target"] = target[keep].astype(np.int8)
    count = output.groupby("__ts__", sort=False).size()
    return output.loc[output.__ts__.isin(count.index[count.ge(2)])].copy()


def _fit(train: pd.DataFrame, fields: list[str]) -> tuple[lgb.LGBMRanker, pd.Series]:
    ordered, groups, label = _query_order(train, train._target.to_numpy(float))
    medians = ordered.loc[:, fields].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).median()
    model = lgb.LGBMRanker(**_params())
    model.fit(_matrix(ordered, fields, medians), label, group=groups)
    return model, medians


def _utility(frame: pd.DataFrame, score: np.ndarray) -> float:
    local = frame.loc[_valid_policy(frame)].copy()
    local["_score"] = np.asarray(score, dtype=float)[_valid_policy(frame)]
    items: dict[int, list[float]] = {1: [], 2: [], 4: []}
    for _, group in local.groupby("__ts__", sort=False):
        if len(group) < 2:
            continue
        outcome = pd.to_numeric(group.p0_canonical_net_bps, errors="coerce")
        baseline = float(outcome.mean())
        ordered = group.sort_values(["_score", "candidate_id"], ascending=[False, True], kind="stable")
        for k in items:
            items[k].append(float(pd.to_numeric(ordered.iloc[:min(k, len(ordered))].p0_canonical_net_bps, errors="coerce").mean()) - baseline)
    return float(.45 * np.mean(items[1]) + .35 * np.mean(items[2]) + .20 * np.mean(items[4])) if all(items.values()) else float("nan")


def _gain_screen(train: pd.DataFrame, fields: list[str]) -> list[str]:
    sample = _sample_queries(train, TRAIN_CAP)
    model, _ = _fit(sample, fields)
    gain = model.booster_.feature_importance(importance_type="gain")
    ordered = sorted(zip(fields, gain, strict=True), key=lambda item: (-float(item[1]), item[0]))
    return [name for name, _ in ordered[:PRE_SCREEN_MAX]]


def _mda(model: lgb.LGBMRanker, medians: pd.Series, validation: pd.DataFrame, fields: list[str], fold: str) -> list[dict[str, Any]]:
    sample = _sample_queries(validation, VALIDATION_CAP)
    matrix = _matrix(sample, fields, medians).to_numpy(np.float32, copy=True)
    baseline = _utility(sample, model.predict(matrix))
    output: list[dict[str, Any]] = []
    for offset, feature in enumerate(fields):
        original = matrix[:, offset].copy()
        drops = []
        for repeat in range(MDA_REPEATS):
            rng = np.random.default_rng(SEED + offset * 1009 + repeat * 37 + len(fold))
            matrix[:, offset] = original[rng.permutation(len(original))]
            drops.append(baseline - _utility(sample, model.predict(matrix)))
        matrix[:, offset] = original
        output.append({"fold": fold, "feature": feature, "mda_rows": int(len(sample)), "baseline_utility_bps": baseline, "mda_drop_mean_bps": float(np.mean(drops)), "mda_drop_std_bps": float(np.std(drops, ddof=1)), "repeats": MDA_REPEATS})
    return output


def _prefix_metrics(train: pd.DataFrame, validation: pd.DataFrame, ranked: list[str], fold: str) -> list[dict[str, Any]]:
    result = []
    for size in FEATURE_SIZES:
        chosen = ranked[:size]
        model, medians = _fit(_sample_queries(train, TRAIN_CAP), chosen)
        result.append({"fold": fold, "feature_size": size, "utility_bps": _utility(validation, model.predict(_matrix(validation, chosen, medians)))})
    return result


def run(*, out: Path, policies: Path, features_path: Path, candidates_path: Path) -> Path:
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    start, selection_end = _utc("2023-10-01T00:00:00Z"), _utc("2024-10-01T00:00:00Z")
    fields120 = _feature_fields("short")
    candidates = _load_candidates(candidates_path, "short")
    candidates = candidates.loc[candidates.__ts__.ge(start) & candidates.__ts__.lt(selection_end)].copy()
    features = _load_features(features_path, fields120, candidates, "short")
    policy = _load_policy_ledger(policies, start, selection_end)
    ledger = features.merge(policy, on=["candidate_id", "__ts__", "__decision_ts__", "__symbol__", "side_name"], how="left", validate="one_to_one")
    kept, coverage = _coverage_fields(ledger, fields120)
    if len(kept) != 115:
        raise ValueError(f"expected 115 coverage-valid fields, found {len(kept)}")
    ledger = ledger.loc[ledger.entry_executable.astype(bool)].copy()
    screens: list[list[str]] = []
    mda: list[dict[str, Any]] = []
    folds: list[tuple[str, pd.DataFrame, pd.DataFrame]] = []
    for name, train_start, valid_start, valid_end in FOLDS:
        train_start_ts, valid_start_ts, valid_end_ts = map(_utc, (train_start, valid_start, valid_end))
        train = _eligible(ledger.loc[ledger.__ts__.ge(train_start_ts) & ledger.__ts__.lt(valid_start_ts)], valid_start_ts)
        valid = ledger.loc[ledger.__ts__.ge(valid_start_ts) & ledger.__ts__.lt(valid_end_ts)].copy()
        if train.empty or valid.empty:
            raise ValueError(f"empty MDA fold {name}")
        screen = _gain_screen(train, kept)
        screens.append(screen); folds.append((name, train, valid))
    candidates72 = sorted(set().union(*[set(values[:48]) for values in screens]))
    # Preserve the exact 72-field cap while favouring repeatedly selected gain fields.
    frequency = {field: sum(field in values for values in screens) for field in kept}
    gain_order = {field: sum((len(values) - values.index(field)) if field in values else 0 for values in screens) for field in kept}
    candidates72 = sorted(candidates72, key=lambda field: (-frequency[field], -gain_order[field], field))[:PRE_SCREEN_MAX]
    for name, train, valid in folds:
        model, medians = _fit(_sample_queries(train, TRAIN_CAP), candidates72)
        mda.extend(_mda(model, medians, valid, candidates72, name))
    mda_frame = pd.DataFrame(mda)
    aggregate = mda_frame.groupby("feature", as_index=False).agg(mean_drop_bps=("mda_drop_mean_bps", "mean"), median_drop_bps=("mda_drop_mean_bps", "median"), std_drop_bps=("mda_drop_mean_bps", "std"), fold_count=("fold", "nunique"))
    aggregate["portable_mda_bps"] = aggregate.mean_drop_bps - .5 * aggregate.std_drop_bps.fillna(0.0)
    mda_ranked = aggregate.sort_values(["portable_mda_bps", "mean_drop_bps", "feature"], ascending=[False, False, True], kind="stable").feature.tolist()
    remainder = [field for field in sorted(kept, key=lambda field: (-frequency[field], -gain_order[field], field)) if field not in set(mda_ranked)]
    ranked = mda_ranked + remainder
    prefix = []
    for name, train, valid in folds:
        prefix.extend(_prefix_metrics(train, valid, ranked, name))
    prefix_frame = pd.DataFrame(prefix)
    summary = prefix_frame.groupby("feature_size", as_index=False).agg(mean_utility_bps=("utility_bps", "mean"), std_utility_bps=("utility_bps", "std"), folds=("fold", "nunique"))
    summary["se_utility_bps"] = summary.std_utility_bps / np.sqrt(summary.folds)
    best = summary.loc[summary.mean_utility_bps.idxmax()]
    recommended = int(summary.loc[summary.mean_utility_bps.ge(best.mean_utility_bps - best.se_utility_bps), "feature_size"].min())
    selected = {str(size): ranked[:size] for size in FEATURE_SIZES}
    mda_frame.to_parquet(out / "chronological_policy_mda.parquet", index=False, compression="zstd")
    aggregate.to_parquet(out / "feature_mda_summary.parquet", index=False, compression="zstd")
    prefix_frame.to_parquet(out / "feature_size_development_metrics.parquet", index=False, compression="zstd")
    payload = {"schema": "strict_r3_short_policy_conversion_mda_v1", "status": "complete", "side": "short", "target": SPEC.name, "selection_window": "[2023-10-01, 2024-10-01)", "final_oos_excluded": "[2024-10-01, 2025-01-01)", "coverage_valid_field_count": len(kept), "coverage": {field: float(coverage[field]) for field in kept}, "prescreen_fields": candidates72, "ranked_features": ranked, "feature_sets": selected, "recommended_feature_size_development_only": recommended, "selection_rule": "smallest feature size within one standard error of best three-fold chronological utility", "folds": [name for name, _, _ in folds]}
    (out / "selected_features.json").write_text(json.dumps(payload, indent=2) + "\n")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--policies", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    args = parser.parse_args()
    print(run(out=args.out.resolve(), policies=args.policies.resolve(), features_path=args.features.resolve(), candidates_path=args.candidates.resolve()))


if __name__ == "__main__":
    main()
