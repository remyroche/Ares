#!/usr/bin/env python3
"""Bounded strict-OOS HPO for the selected VWAP pairwise entry challenger.

This is deliberately narrower than an entry-policy search.  It keeps the
target-free candidate universe, the 20--30 bps reserve band, two-entry
incumbent capacity, BCF priority, and the +50-bps replacement requirement
fixed.  It varies only the pairwise quantile model geometry after causal
feature selection.  June--July choose a configuration; August is written as
an untouched holdout and is never used in ranking.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.p8u_15m_features import FIFTEEN_MINUTE_FEATURE_KEYS, VWAP_15M_FEATURE_KEYS
from scripts import run_strict_r3_p8u_15m_entry_feature_contract_ablation as study
from scripts import run_strict_r3_p8u_15m_entry_pairwise_replacement_ablation as base


FEATURE_STUDY = ROOT / "data_perp/artifacts/strict_r3_p8u_15m_entry_feature_contract_20260830_v2"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/strict_r3_p8u_15m_entry_postfs_hpo_20260830_v1"
SELECTION_END = pd.Timestamp("2026-08-01", tz="UTC")
SEED = 1729

# The baseline is explicitly present.  The other arms vary one model-family
# dimension at a time and keep tree support comparatively high for portability.
SPECS: dict[str, dict[str, float | int]] = {
    "H0_q50_d3_l7_baseline": {"alpha": .50, "max_depth": 3, "num_leaves": 7, "min_child_fraction": .03, "reg_lambda": 8., "learning_rate": .03, "n_estimators": 350},
    "H1_q35_d3_l7": {"alpha": .35, "max_depth": 3, "num_leaves": 7, "min_child_fraction": .03, "reg_lambda": 8., "learning_rate": .03, "n_estimators": 350},
    "H2_q65_d3_l7": {"alpha": .65, "max_depth": 3, "num_leaves": 7, "min_child_fraction": .03, "reg_lambda": 8., "learning_rate": .03, "n_estimators": 350},
    "H3_q50_d2_l3_strict": {"alpha": .50, "max_depth": 2, "num_leaves": 3, "min_child_fraction": .04, "reg_lambda": 12., "learning_rate": .03, "n_estimators": 350},
    "H4_q50_d3_l7_leaf5_reg16": {"alpha": .50, "max_depth": 3, "num_leaves": 7, "min_child_fraction": .05, "reg_lambda": 16., "learning_rate": .03, "n_estimators": 350},
    "H5_q50_d4_l15_leaf5_reg16": {"alpha": .50, "max_depth": 4, "num_leaves": 15, "min_child_fraction": .05, "reg_lambda": 16., "learning_rate": .025, "n_estimators": 420},
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _features_by_month(path: Path, variant: str, held_months: tuple[pd.Timestamp, ...]) -> dict[pd.Timestamp, tuple[str, ...]]:
    selected = pd.read_parquet(path)
    required = {"variant", "held_month", "feature", "position"}
    missing = required.difference(selected.columns)
    if missing:
        raise ValueError(f"feature-selection receipt lacks {sorted(missing)}")
    scoped = selected.loc[selected["variant"].eq(variant)].copy()
    result: dict[pd.Timestamp, tuple[str, ...]] = {}
    for held in held_months:
        rows = scoped.loc[scoped["held_month"].eq(held.strftime("%Y-%m"))].sort_values("position", kind="stable")
        features = tuple(rows["feature"].astype(str))
        if not 30 <= len(features) <= 45 or len(set(features)) != len(features):
            raise AssertionError(f"{variant} selection for {held:%Y-%m} is not a 30--45 unique-field contract")
        result[held] = features
    return result


def _fit(train: pd.DataFrame, features: tuple[str, ...], spec: dict[str, float | int]) -> lgb.LGBMRegressor:
    child = max(8, int(np.ceil(len(train) * float(spec["min_child_fraction"]))))
    model = lgb.LGBMRegressor(
        objective="quantile", alpha=float(spec["alpha"]),
        n_estimators=int(spec["n_estimators"]), learning_rate=float(spec["learning_rate"]),
        max_depth=int(spec["max_depth"]), num_leaves=int(spec["num_leaves"]), min_child_samples=child,
        subsample=.80, colsample_bytree=.80, reg_lambda=float(spec["reg_lambda"]),
        random_state=SEED, n_jobs=2, verbosity=-1,
    )
    position = np.clip(
        (pd.to_numeric(train.reserve_dual_mc1_min_bps, errors="raise").to_numpy(float) - base.RESERVE_FLOOR)
        / (base.CORE_FLOOR - base.RESERVE_FLOOR), 0.0, 1.0,
    )
    model.fit(train.loc[:, features], pd.to_numeric(train.pair_advantage_bps, errors="raise"), sample_weight=1.0 + position)
    return model


def _scope_replay(selection: pd.DataFrame, labels: pd.DataFrame, arm: str, output: Path) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    for scope, frame in (
        ("selection_jun_jul", selection.loc[pd.to_datetime(selection.__decision_ts__, utc=True).lt(SELECTION_END)].copy()),
        ("august_holdout", selection.loc[pd.to_datetime(selection.__decision_ts__, utc=True).ge(SELECTION_END)].copy()),
        ("all_oos", selection),
    ):
        if frame.empty:
            continue
        metrics = base._replay(frame, labels, f"{arm}__{scope}", output)
        metrics["model_arm"], metrics["evaluation_scope"] = arm, scope
        summaries.append(metrics)
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-study", type=Path, default=FEATURE_STUDY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-months", type=int, default=4)
    parser.add_argument("--held-month", action="append", help="repeatable YYYY-MM; default Jun--Aug 2026")
    parser.add_argument("--spec", choices=tuple(SPECS), action="append", default=[], help="repeatable bounded model arm")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"immutable output already exists: {output}")
    if args.train_months < 2:
        raise ValueError("strict training needs at least two complete preceding calendar months")
    held_months = tuple(pd.Timestamp(f"{value}-01", tz="UTC") for value in args.held_month) if args.held_month else tuple(pd.date_range("2026-06-01", "2026-08-01", freq="MS", tz="UTC"))
    study_root = args.feature_study.resolve()
    selected_path = study_root / "stable_selected_features.parquet"
    panel = study._candidate_frame(study._load_panel(study.OLD_PANEL, study.VWAP_PANEL))
    labels = base._labels(study.LABEL_ROOT)
    labelled = panel.merge(labels, on="candidate_id", how="inner", validate="one_to_one")
    labelled = labelled.loc[labelled.policy_path_valid.fillna(False)].copy()
    labelled["policy_label_available_ts"] = pd.to_datetime(labelled.policy_label_available_ts, utc=True, errors="raise")
    raw_features = (*FIFTEEN_MINUTE_FEATURE_KEYS, *study.SCORE_FEATURES, *VWAP_15M_FEATURE_KEYS)
    feature_map = _features_by_month(selected_path, "E3_vwap_fs", held_months)
    specs = {name: SPECS[name] for name in (args.spec or list(SPECS))}
    output.mkdir(parents=True, exist_ok=False)
    selections: dict[str, list[pd.DataFrame]] = {name: [] for name in specs}
    controls: list[pd.DataFrame] = []
    trace: list[pd.DataFrame] = []
    for held in held_months:
        end, start = held + pd.offsets.MonthBegin(1), held - pd.DateOffset(months=args.train_months)
        train_raw = labelled.loc[labelled.__decision_ts__.ge(start) & labelled.__decision_ts__.lt(held)].copy()
        train_pairs = study._pairs(train_raw, raw_features, require_labels=True)
        train_pairs = train_pairs.loc[pd.to_datetime(train_pairs.pair_label_available_ts, utc=True).lt(held)].copy() if not train_pairs.empty else train_pairs
        test = panel.loc[panel.__decision_ts__.ge(held) & panel.__decision_ts__.lt(end)].copy()
        test_pairs = study._pairs(test, raw_features, require_labels=False)
        required = {(held - pd.DateOffset(months=1)).strftime("%Y-%m"), (held - pd.DateOffset(months=2)).strftime("%Y-%m")}
        observed = set(pd.to_datetime(train_pairs.__decision_ts__, utc=True).dt.strftime("%Y-%m")) if not train_pairs.empty else set()
        if not required.issubset(observed) or len(train_pairs) < 100 or test.empty:
            raise RuntimeError(f"strict pair support missing for {held:%Y-%m}")
        controls.append(base._incumbent_top2(test).assign(held_month=held.strftime("%Y-%m")))
        features = feature_map[held]
        missing = set(features).difference(train_pairs.columns) | set(features).difference(test_pairs.columns)
        if missing:
            raise AssertionError(f"selected feature contract is absent from pair panel: {sorted(missing)}")
        for name, spec in specs.items():
            model = _fit(train_pairs, features, spec)
            predicted = test_pairs.loc[:, ["reserve_candidate_id", "incumbent_candidate_id", "__decision_ts__", "__symbol__", "reserve_bcf_mc1_expected_bps", "incumbent_bcf_mc1_expected_bps"]].copy()
            predicted["pair_lcb_advantage_bps"] = model.predict(test_pairs.loc[:, features])
            chosen, proposals = base._apply_replacement(test, predicted, 50.0)
            chosen["held_month"], chosen["hpo_arm"] = held.strftime("%Y-%m"), name
            selections[name].append(chosen)
            proposals["held_month"], proposals["hpo_arm"] = held.strftime("%Y-%m"), name
            trace.append(proposals)
    summaries: list[dict[str, object]] = []
    for name, frames in selections.items():
        selected = pd.concat(frames, ignore_index=True)
        if selected.candidate_id.duplicated().any():
            raise AssertionError(f"{name} duplicated a strict-OOS candidate")
        selected.to_parquet(output / f"{name}_selection_target_free.parquet", index=False, compression="zstd")
        summaries.extend(_scope_replay(selected, labels, name, output))
    control = pd.concat(controls, ignore_index=True)
    if control.candidate_id.duplicated().any():
        raise AssertionError("B0 control duplicated a strict-OOS candidate")
    control.to_parquet(output / "B0_bcf_top2_selection_target_free.parquet", index=False, compression="zstd")
    summaries.extend(_scope_replay(control, labels, "B0_bcf_top2", output))
    summary = pd.DataFrame(summaries)
    summary["total_ev_per_abs_drawdown"] = summary.total_policy_net_bps / summary.max_drawdown.abs().replace(0.0, np.nan)
    for scope, group in summary.groupby("evaluation_scope", sort=False):
        baseline = group.loc[group.model_arm.eq("B0_bcf_top2")]
        if len(baseline) != 1:
            raise AssertionError(f"missing B0 for {scope}")
        for metric in ("portfolio_accepted", "policy_net_bps_per_trade", "total_policy_net_bps", "max_drawdown", "worst_week", "total_ev_per_abs_drawdown"):
            summary.loc[group.index, f"delta_vs_B0_{metric}"] = group[metric] - baseline.iloc[0][metric]
    summary.to_parquet(output / "portfolio_summary.parquet", index=False)
    selection = summary.loc[summary.evaluation_scope.eq("selection_jun_jul") & ~summary.model_arm.eq("B0_bcf_top2")].sort_values(["total_ev_per_abs_drawdown", "policy_net_bps_per_trade", "worst_week"], ascending=[False, False, False], kind="stable")
    selection.to_parquet(output / "selection_ranking_jun_jul.parquet", index=False)
    pd.concat(trace, ignore_index=True).to_parquet(output / "replacement_proposals.parquet", index=False, compression="zstd")
    (output / "run_manifest.json").write_text(json.dumps({
        "schema": "strict-r3-p8u-entry-postfs-hpo-v1", "scope": "offline strict-OOS research only; no live/canonical mutation",
        "feature_study": str(study_root), "feature_selection_variant": "E3_vwap_fs", "feature_selection_sha256": _sha256(selected_path),
        "fold": f"up to {args.train_months} trailing complete prior calendar months; pair labels resolved before each held boundary",
        "selection_period": "2026-06 through 2026-07 only; August untouched", "held_months": [f"{x:%Y-%m}" for x in held_months],
        "authority": "fixed: 20--30 bps reserve may replace only the marginal of ordinary BCF-priority top-two 30-bps incumbents; +50 bps required; no capacity expansion",
        "specs": specs, "seed": SEED,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
