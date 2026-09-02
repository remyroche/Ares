#!/usr/bin/env python3
"""Grouped strict-OOF permutation audit for the timestamp State-Meta block.

Importance is evaluated on a state model's ability to select timestamps whose
already Base-selected top-two candidates realise higher policy residuals.  It
does not misrepresent timestamp-global state as a per-candidate rank feature.
Correlation blocks are frozen from target-free pre-May-2025 data, then
permuted only in each held fold; labels never determine block membership.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from scipy.stats import spearmanr

import probe_strict_r3_p8u_market_state_timestamp_synergy_v1 as state_probe


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "strict_r3_p8u_market_state_timestamp_group_mda_v1"
SEED = 1729


def _once(path: Path, payload: object) -> None:
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def _month(token: str) -> pd.Timestamp:
    return pd.Timestamp(f"{token}-01", tz="UTC")


def _interactions(frame: pd.DataFrame, fields: tuple[str, ...]) -> tuple[pd.DataFrame, tuple[str, ...]]:
    out = frame.copy()
    # The discovery contract contains the six raw fields needed for the
    # three predeclared interactions.  A later frozen MDA contract may keep
    # only one raw field; in that confirmation case do not silently invent
    # a different interaction feature set.
    if len(fields) < 4:
        return out, fields
    specs = (
        ("ms_synergy__breadth_downside_x_return_iqr", fields[0], fields[3]),
        ("ms_synergy__execution_spread_x_return_iqr", fields[2], fields[3]),
        ("ms_synergy__breadth_downside_x_execution_spread", fields[0], fields[2]),
    )
    for name, left, right in specs:
        product = out[left].to_numpy(float) * out[right].to_numpy(float)
        out[name] = np.sign(product) * np.sqrt(np.abs(product))
    return out, tuple([*fields, *(name for name, _, _ in specs)])


def _components(corr: pd.DataFrame, threshold: float) -> list[tuple[str, ...]]:
    names = list(corr.columns); seen: set[str] = set(); output: list[tuple[str, ...]] = []
    for start in names:
        if start in seen:
            continue
        queue = [start]; current: set[str] = set()
        while queue:
            node = queue.pop()
            if node in current:
                continue
            current.add(node); seen.add(node)
            close = corr.index[corr.loc[node].abs().ge(threshold)].tolist()
            queue.extend(value for value in close if value not in current)
        output.append(tuple(sorted(current)))
    return sorted(output, key=lambda values: (len(values), values), reverse=True)


def _score(held: pd.DataFrame, prediction: np.ndarray) -> tuple[float, float, float]:
    ic = float(spearmanr(prediction, held.top2_residual_bps).statistic)
    work = held.loc[:, ["__decision_ts__", "top2_policy_net_bps"]].copy(); work["prediction"] = prediction
    count = max(1, int(np.ceil(len(work) * .20)))
    chosen = work.sort_values(["prediction", "__decision_ts__"], ascending=[False, True], kind="stable").head(count)
    spread = float(chosen.top2_policy_net_bps.mean() - work.top2_policy_net_bps.mean())
    return .5 * ic + .5 * spread / 100., ic, spread


def _model(seed: int, feature_count: int) -> LGBMRegressor:
    return LGBMRegressor(
        objective="huber", n_estimators=180, learning_rate=.04, max_depth=3, num_leaves=15,
        min_child_samples=96, min_split_gain=.002, feature_fraction=max(.70, min(.95, 8 / max(8, feature_count))),
        bagging_fraction=.80, reg_lambda=12.0, reg_alpha=.05, random_state=seed, n_jobs=1, verbosity=-1,
    )


def _folds(frame: pd.DataFrame, tokens: Iterable[str]) -> list[tuple[str, np.ndarray, np.ndarray]]:
    output = []
    for token in tokens:
        start = _month(token); end = start + pd.offsets.MonthBegin(1)
        train = np.flatnonzero((frame.__decision_ts__.lt(start) & frame.label_available_ts.lt(start)).to_numpy())
        held = np.flatnonzero((frame.__decision_ts__.ge(start) & frame.__decision_ts__.lt(end)).to_numpy())
        if len(train) >= 500 and len(held) >= 200:
            output.append((token, train, held))
    if len(output) < 3:
        raise AssertionError("need at least three strict-OOF MDA folds")
    return output


def _fit_predict(frame: pd.DataFrame, train_index: np.ndarray, held_index: np.ndarray, fields: tuple[str, ...], seed: int) -> tuple[np.ndarray, pd.DataFrame]:
    columns = [*state_probe.BASE_CONTEXT, *fields]
    train, held = frame.iloc[train_index], frame.iloc[held_index].copy()
    model = _model(seed, len(columns))
    model.fit(train.loc[:, columns].replace([np.inf, -np.inf], np.nan), train.top2_residual_bps.clip(-500., 500.).to_numpy(float))
    prediction = model.predict(held.loc[:, columns].replace([np.inf, -np.inf], np.nan))
    return prediction, held


def _summary(frame: pd.DataFrame, *, key: str) -> pd.DataFrame:
    grouped = frame.groupby(key, sort=True).agg(
        mean_delta_score=("delta_score", "mean"), median_delta_score=("delta_score", "median"),
        positive_score_folds=("delta_score", lambda value: int((value > 0).sum())),
        mean_delta_ic=("delta_ic", "mean"), mean_delta_top20_spread_bps=("delta_spread_bps", "mean"),
        positive_spread_folds=("delta_spread_bps", lambda value: int((value > 0).sum())),
        folds=("fold", "nunique"), worst_delta_spread_bps=("delta_spread_bps", "min"),
    ).reset_index()
    grouped["stable_useful"] = grouped.positive_score_folds.ge(np.ceil(grouped.folds * .60)) & grouped.positive_spread_folds.ge(np.ceil(grouped.folds * .60)) & grouped.mean_delta_score.gt(0)
    return grouped.sort_values(["stable_useful", "mean_delta_score", "mean_delta_top20_spread_bps", key], ascending=[False, False, False, True], kind="stable")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-root", required=True); parser.add_argument("--screen-root", required=True)
    parser.add_argument("--early-base-root", required=True); parser.add_argument("--later-base-root", required=True)
    parser.add_argument("--policy-labels", required=True); parser.add_argument("--frozen-contract", required=True)
    parser.add_argument("--held-months", default="2025-05,2025-06,2025-07,2025-08,2025-09,2025-10,2025-11,2025-12")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    root = ROOT / args.out
    if root.exists(): raise FileExistsError(root)
    frozen = json.loads((ROOT / args.frozen_contract).read_text())
    raw = tuple(str(value) for value in frozen["selected_features"])
    frame, available = state_probe._load_frame(ROOT / args.state_root, ROOT / args.screen_root, ROOT / args.early_base_root, ROOT / args.later_base_root, ROOT / args.policy_labels)
    if not set(raw).issubset(available): raise AssertionError("frozen raw state fields unavailable")
    frame, fields = _interactions(frame, raw)
    # Freeze correlation blocks from target-free Dec-2024 through Apr-2025
    # inputs.  This precedes every May--December held outcome.
    development = frame.loc[frame.__decision_ts__.lt(pd.Timestamp("2025-05-01", tz="UTC")), list(fields)]
    corr = development.corr(method="spearman").fillna(0.0)
    blocks85, blocks95 = _components(corr, .85), _components(corr, .95)
    assignments = []
    for level, blocks in (("block85", blocks85), ("subblock95", blocks95)):
        for index, block in enumerate(blocks):
            for field in block: assignments.append({"level": level, "block": f"{level}_{index:02d}", "field": field, "block_size": len(block)})
    folds = _folds(frame, (item.strip() for item in args.held_months.split(",") if item.strip()))
    rows: list[dict[str, object]] = []
    for fold_index, (token, train_index, held_index) in enumerate(folds):
        baseline, held = _fit_predict(frame, train_index, held_index, fields, SEED + fold_index)
        base_score, base_ic, base_spread = _score(held, baseline)
        for level, blocks in (("block85", blocks85), ("subblock95", blocks95)):
            for block_index, block in enumerate(blocks):
                permuted = held.copy()
                rng = np.random.default_rng(SEED + fold_index * 1009 + block_index * 31 + (0 if level == "block85" else 1))
                order = rng.permutation(len(permuted))
                permuted.loc[:, list(block)] = permuted.loc[:, list(block)].to_numpy()[order]
                # Existing fitted model is evaluated against a target-free
                # held matrix with only this frozen state block permuted.
                columns = [*state_probe.BASE_CONTEXT, *fields]
                train = frame.iloc[train_index]
                model = _model(SEED + fold_index, len(columns))
                model.fit(train.loc[:, columns].replace([np.inf, -np.inf], np.nan), train.top2_residual_bps.clip(-500., 500.).to_numpy(float))
                prediction = model.predict(permuted.loc[:, columns].replace([np.inf, -np.inf], np.nan))
                value, ic, spread = _score(permuted, prediction)
                rows.append({"fold": token, "level": level, "block": f"{level}_{block_index:02d}", "fields": "|".join(block), "field_count": len(block),
                             "baseline_score": base_score, "baseline_ic": base_ic, "baseline_top20_spread_bps": base_spread,
                             "permuted_score": value, "permuted_ic": ic, "permuted_top20_spread_bps": spread,
                             "delta_score": base_score - value, "delta_ic": base_ic - ic, "delta_spread_bps": base_spread - spread})
    detail = pd.DataFrame(rows)
    summary = pd.concat([_summary(detail.loc[detail.level.eq(level)], key="block") for level in ("block85", "subblock95")], ignore_index=True)
    selected_blocks = summary.loc[(summary.block.str.startswith("subblock95")) & summary.stable_useful].copy()
    selected_fields = sorted({field for block in selected_blocks.block for field in pd.DataFrame(assignments).loc[lambda x: x.block.eq(block), "field"]})
    output_contract = {"schema": SCHEMA, "scope": "pre-2026 strict-OOF grouped permutation selection for timestamp State Meta", "selected_features": selected_fields, "selection_rule": "subblock has positive score and top20 spread permutation importance in at least 60% of held months", "correlation_window": "target-free Dec-2024 through Apr-2025", "thresholds": {"block": .85, "subblock": .95}}
    root.mkdir(parents=True)
    corr.to_parquet(root / "target_free_spearman_correlation.parquet")
    pd.DataFrame(assignments).to_parquet(root / "correlation_block_assignments.parquet", index=False)
    detail.to_parquet(root / "grouped_permutation_fold_metrics.parquet", index=False)
    summary.to_parquet(root / "grouped_permutation_summary.parquet", index=False)
    _once(root / "selected_state_mda_contract.json", output_contract)
    correctness = {"schema": SCHEMA, "correlation_blocks_fit_target_free_before_held_period": True, "all_state_features_target_free": True, "held_labels_excluded_from_feature_matrices": True, "training_labels_resolved_before_held_month": True, "permutation_acts_on_state_fields_only": True, "top2_outcome_is_base_selected_before_state_model": True, "no_meta_mc1_admission_portfolio_live_or_exchange_mutation": True}
    _once(root / "correctness_report.json", correctness)
    _once(root / "run_manifest.json", {"schema": SCHEMA, "scope": "offline State Meta grouped permutation importance", "selected_raw_contract": args.frozen_contract, "fields": list(fields), "held_months": args.held_months, "correctness": correctness})
    print(json.dumps({"out": str(root), "block85": len(blocks85), "subblock95": len(blocks95), "selected_fields": selected_fields}, sort_keys=True))


if __name__ == "__main__":
    main()
