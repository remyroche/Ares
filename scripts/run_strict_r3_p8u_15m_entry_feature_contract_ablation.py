#!/usr/bin/env python3
"""Strict-OOS feature-contract comparison for the capacity-bounded entry arm.

The entry authority is fixed throughout: a reserve (20--30 bps dual-MC1)
candidate can replace at most the marginal of the ordinary BCF-priority top
two 30-bps incumbents at its timestamp.  It cannot manufacture capacity.
April--July 2026 ranks contracts by total policy EV per absolute drawdown;
August is written separately and is never used for that selection.
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
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.p8u_15m_features import FIFTEEN_MINUTE_FEATURE_KEYS, VWAP_15M_FEATURE_KEYS
from scripts import run_strict_r3_p8u_15m_entry_pairwise_replacement_ablation as base


OLD_PANEL = ROOT / "data_perp/artifacts/strict_r3_p8u_15m_ordinal_mc1_threshold_observed25h_20260830_v4_manifested_results/target_free_15m_features.parquet"
VWAP_PANEL = ROOT / "data_perp/artifacts/strict_r3_p8u_15m_entry_vwap_target_free_20260830_v1/target_free_15m_features.parquet"
LABEL_ROOT = ROOT / "data_perp/artifacts/strict_r3_p8u_15m_rich_policy_labels_20260830_v1_control"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/strict_r3_p8u_15m_entry_feature_contract_20260830_v1"
SEED = 1729
SELECTION_END = pd.Timestamp("2026-08-01", tz="UTC")

SCORE_FEATURES = ("bcf_final_score", "bcf_mc1_expected_bps", "current_mc1_expected_bps", "dual_mc1_min_bps")
LEGACY_MARGIN_FEATURES = base.MARGIN_FEATURES
MANDATORY = (
    *SCORE_FEATURES,
    "margin__bcf_final_score",
    "margin__bcf_mc1_expected_bps",
    "margin__current_mc1_expected_bps",
    "margin__dual_mc1_min_bps",
    "incumbent_bcf_mc1_expected_bps",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_panel(old_path: Path, vwap_path: Path) -> pd.DataFrame:
    old = pd.read_parquet(old_path)
    vwap = pd.read_parquet(vwap_path)
    for frame in (old, vwap):
        frame["candidate_id"] = frame["candidate_id"].astype(str)
        frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True, errors="raise")
    if old.candidate_id.duplicated().any() or vwap.candidate_id.duplicated().any():
        raise AssertionError("target-free entry feature identity is not unique")
    ids = old.candidate_id.to_numpy()
    common = old.set_index("candidate_id").loc[ids, list(FIFTEEN_MINUTE_FEATURE_KEYS)]
    regenerated = vwap.set_index("candidate_id").loc[ids, list(FIFTEEN_MINUTE_FEATURE_KEYS)]
    if not np.allclose(common.to_numpy(float), regenerated.to_numpy(float), rtol=0.0, atol=0.0, equal_nan=True):
        raise AssertionError("VWAP enrichment changed a frozen legacy 15m feature value")
    overlay = vwap.loc[:, ["candidate_id", *VWAP_15M_FEATURE_KEYS]].copy()
    panel = old.merge(overlay, on="candidate_id", how="inner", validate="one_to_one")
    if len(panel) != len(old):
        raise AssertionError("VWAP target-free panel does not cover the legacy feature identity")
    return panel


def _candidate_frame(features: pd.DataFrame) -> pd.DataFrame:
    required = {
        "candidate_id", "__decision_ts__", "__symbol__", "dual_mc1_min_bps", "bcf_mc1_expected_bps", "bcf_final_score",
        "finite_15m_feature_count", *FIFTEEN_MINUTE_FEATURE_KEYS, *VWAP_15M_FEATURE_KEYS,
    }
    missing = required.difference(features.columns)
    if missing:
        raise ValueError(f"feature panel lacks {sorted(missing)}")
    frame = features.copy()
    frame["candidate_id"] = frame.candidate_id.astype(str)
    frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True, errors="raise")
    frame = frame.loc[
        pd.to_numeric(frame.dual_mc1_min_bps, errors="coerce").ge(base.RESERVE_FLOOR)
        & pd.to_numeric(frame.finite_15m_feature_count, errors="coerce").ge(50)
    ].copy()
    if frame.candidate_id.duplicated().any():
        raise AssertionError("candidate target-free input duplicated identity")
    return frame


def _pairs(frame: pd.DataFrame, raw_features: tuple[str, ...], *, require_labels: bool) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for timestamp, group in frame.groupby("__decision_ts__", sort=True):
        incumbent = base._marginal_incumbent(group)
        if incumbent is None:
            continue
        reserves = group.loc[
            pd.to_numeric(group.dual_mc1_min_bps, errors="coerce").ge(base.RESERVE_FLOOR)
            & pd.to_numeric(group.dual_mc1_min_bps, errors="coerce").lt(base.CORE_FLOOR)
        ]
        for _, reserve in reserves.iterrows():
            row: dict[str, object] = {
                "reserve_candidate_id": str(reserve.candidate_id), "incumbent_candidate_id": str(incumbent.candidate_id),
                "__decision_ts__": timestamp, "__symbol__": str(reserve.__symbol__),
                "reserve_bcf_mc1_expected_bps": float(reserve.bcf_mc1_expected_bps),
                "reserve_dual_mc1_min_bps": float(reserve.dual_mc1_min_bps),
                "incumbent_bcf_mc1_expected_bps": float(incumbent.bcf_mc1_expected_bps),
            }
            for feature in raw_features:
                row[feature] = float(reserve[feature])
            for feature in LEGACY_MARGIN_FEATURES:
                row[f"margin__{feature}"] = float(reserve[feature]) - float(incumbent[feature])
            if any(feature in raw_features for feature in VWAP_15M_FEATURE_KEYS):
                for feature in VWAP_15M_FEATURE_KEYS:
                    row[f"margin__{feature}"] = float(reserve[feature]) - float(incumbent[feature])
            if require_labels:
                row["pair_advantage_bps"] = float(reserve.policy_net_bps) - float(incumbent.policy_net_bps)
                row["pair_label_available_ts"] = max(pd.Timestamp(reserve.policy_label_available_ts), pd.Timestamp(incumbent.policy_label_available_ts))
            records.append(row)
    return pd.DataFrame(records)


def _fit(train: pd.DataFrame, features: tuple[str, ...]) -> lgb.LGBMRegressor:
    model = lgb.LGBMRegressor(
        objective="quantile", alpha=0.50, n_estimators=350, learning_rate=0.03,
        max_depth=3, num_leaves=7, min_child_samples=max(8, int(np.ceil(len(train) * 0.03))),
        subsample=0.8, colsample_bytree=0.8, reg_lambda=8.0, random_state=SEED, n_jobs=2, verbosity=-1,
    )
    position = np.clip((pd.to_numeric(train.reserve_dual_mc1_min_bps, errors="raise").to_numpy(float) - base.RESERVE_FLOOR) / (base.CORE_FLOOR - base.RESERVE_FLOOR), 0.0, 1.0)
    model.fit(train.loc[:, features], pd.to_numeric(train.pair_advantage_bps, errors="raise"), sample_weight=1.0 + position)
    return model


def _impute(frame: pd.DataFrame, columns: tuple[str, ...]) -> np.ndarray:
    values = frame.loc[:, columns].apply(pd.to_numeric, errors="coerce")
    return values.fillna(values.median(numeric_only=True)).fillna(0.0).to_numpy(float)


def _sample(frame: pd.DataFrame, maximum: int, seed: int) -> pd.DataFrame:
    if len(frame) <= maximum:
        return frame
    return frame.sample(n=maximum, random_state=seed).sort_values("__decision_ts__", kind="stable")


def _subspace_ranking(train: pd.DataFrame, mandatory: tuple[str, ...], optional: tuple[str, ...], runs: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = train.sort_values("__decision_ts__", kind="stable")
    cut = ordered["__decision_ts__"].quantile(.80)
    fit, valid = ordered.loc[ordered["__decision_ts__"].lt(cut)], ordered.loc[ordered["__decision_ts__"].ge(cut)]
    if len(fit) < 200 or len(valid) < 50:
        raise RuntimeError("insufficient temporal pair rows for feature selection")
    fit, valid = _sample(fit, 15_000, SEED), _sample(valid, 6_000, SEED + 1)
    rng = np.random.default_rng(SEED)
    rows: list[dict[str, object]] = []
    for run in range(runs):
        width = max(1, int(round(len(optional) * rng.uniform(.30, .60))))
        chosen = tuple(sorted(rng.choice(np.asarray(optional, dtype=object), size=width, replace=False).tolist()))
        model = _fit(fit, (*mandatory, *chosen))
        predicted = model.predict(valid.loc[:, (*mandatory, *chosen)])
        top = valid.assign(_prediction=predicted).nlargest(max(1, int(np.ceil(len(valid) * .20))), "_prediction")
        utility = float(top.pair_advantage_bps.mean())
        temporal = float((top.assign(_month=pd.to_datetime(top.__decision_ts__, utc=True).dt.strftime("%Y-%m")).groupby("_month").pair_advantage_bps.mean() > 0.0).mean())
        top_features = pd.Series(model.feature_importances_, index=(*mandatory, *chosen)).nlargest(min(12, len(mandatory) + len(chosen))).index
        for feature in chosen:
            rows.append({"feature": feature, "run": run, "utility": utility, "temporal_positive_fraction": temporal, "selected_topk": feature in set(top_features)})
    detail = pd.DataFrame(rows)
    median = float(detail.drop_duplicates("run").utility.median())
    ranking = detail.groupby("feature", as_index=False).agg(selection_frequency=("selected_topk", "mean"), mean_subspace_utility=("utility", "mean"), temporal_stability=("temporal_positive_fraction", "mean"), inclusion_runs=("run", "nunique"))
    ranking["performance_weight"] = 1.0 / (1.0 + np.exp(-(ranking.mean_subspace_utility - median) / 50.0))
    ranking["stability_score"] = ranking.selection_frequency * ranking.performance_weight * ranking.temporal_stability
    return ranking.sort_values(["stability_score", "selection_frequency"], ascending=False, kind="stable"), detail


def _utility(train: pd.DataFrame, features: tuple[str, ...]) -> float:
    ordered = train.sort_values("__decision_ts__", kind="stable")
    cut = ordered["__decision_ts__"].quantile(.80)
    fit, valid = ordered.loc[ordered["__decision_ts__"].lt(cut)], ordered.loc[ordered["__decision_ts__"].ge(cut)]
    if len(fit) < 150 or len(valid) < 40:
        return float("-inf")
    model = _fit(fit, features)
    proposed = valid.assign(_prediction=model.predict(valid.loc[:, features])).nlargest(max(1, int(np.ceil(len(valid) * .20))), "_prediction")
    return float(proposed.pair_advantage_bps.mean())


def _select(train: pd.DataFrame, all_features: tuple[str, ...], runs: int) -> tuple[tuple[str, ...], pd.DataFrame, pd.DataFrame]:
    mandatory = tuple(name for name in MANDATORY if name in all_features)
    optional = tuple(name for name in all_features if name not in mandatory)
    ranking, detail = _subspace_ranking(train, mandatory, optional, runs)
    selected = list(mandatory)
    base_utility = _utility(train, tuple(selected))
    stale = 0
    trace: list[dict[str, object]] = []
    for _, row in ranking.iterrows():
        feature = str(row.feature)
        sample = _sample(train, 10_000, SEED + len(selected))
        y = pd.to_numeric(sample.pair_advantage_bps, errors="coerce").fillna(0.0).to_numpy(float)
        x = _impute(sample, (feature,))[:, 0]
        if selected:
            context = _impute(sample, tuple(selected))
            y = y - Ridge(alpha=4.0).fit(context, y).predict(context)
            x = x - Ridge(alpha=4.0).fit(context, x).predict(context)
        cmi = float(mutual_info_regression(x.reshape(-1, 1), y, n_neighbors=5, random_state=SEED)[0]) if np.nanstd(x) > 1e-12 and np.nanstd(y) > 1e-12 else 0.0
        candidate = tuple((*selected, feature))
        utility = _utility(train, candidate)
        delta = utility - base_utility
        accept = bool(cmi >= .002 and delta >= -2.0)
        trace.append({**row.to_dict(), "conditional_mi": cmi, "incremental_oof_pair_utility": delta, "accepted": accept, "feature_count_after": len(candidate) if accept else len(selected)})
        if accept:
            selected.append(feature); base_utility = utility; stale = 0
        else:
            stale += 1
        if len(selected) >= 45 or (len(selected) >= 30 and stale >= 6):
            break
    for feature in ranking.feature:
        if len(selected) >= 30:
            break
        if feature not in selected:
            selected.append(str(feature))
    return tuple(selected), pd.DataFrame(trace), ranking


def _predict_selection(frame: pd.DataFrame, train_pairs: pd.DataFrame, test_pairs: pd.DataFrame, features: tuple[str, ...], threshold: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    model = _fit(train_pairs, features)
    predicted = test_pairs.loc[:, ["reserve_candidate_id", "incumbent_candidate_id", "__decision_ts__", "__symbol__", "reserve_bcf_mc1_expected_bps", "incumbent_bcf_mc1_expected_bps"]].copy()
    predicted["pair_lcb_advantage_bps"] = model.predict(test_pairs.loc[:, features])
    selection, log = base._apply_replacement(frame, predicted, threshold)
    return selection, log


def _run_variant(target_free: pd.DataFrame, labelled: pd.DataFrame, variant: str, raw_features: tuple[str, ...], pair_features: tuple[str, ...], *, select: bool, subspace_runs: int, held_months: tuple[pd.Timestamp, ...], train_months: int) -> tuple[pd.DataFrame, list[pd.DataFrame], list[pd.DataFrame], list[pd.DataFrame], pd.DataFrame]:
    selections: list[pd.DataFrame] = []
    proposals: list[pd.DataFrame] = []
    feature_rows: list[pd.DataFrame] = []
    traces: list[pd.DataFrame] = []
    rankings: list[pd.DataFrame] = []
    controls: list[pd.DataFrame] = []
    for held in held_months:
        start, end = held - pd.DateOffset(months=train_months), held + pd.offsets.MonthBegin(1)
        train_raw = labelled.loc[labelled.__decision_ts__.ge(start) & labelled.__decision_ts__.lt(held)].copy()
        train_pairs = _pairs(train_raw, raw_features, require_labels=True)
        train_pairs = train_pairs.loc[pd.to_datetime(train_pairs.pair_label_available_ts, utc=True).lt(held)].copy() if not train_pairs.empty else train_pairs
        test = target_free.loc[target_free.__decision_ts__.ge(held) & target_free.__decision_ts__.lt(end)].copy()
        test_pairs = _pairs(test, raw_features, require_labels=False)
        needed = {(held - pd.DateOffset(months=1)).strftime("%Y-%m"), (held - pd.DateOffset(months=2)).strftime("%Y-%m")}
        seen = set(pd.to_datetime(train_pairs.__decision_ts__, utc=True).dt.strftime("%Y-%m")) if not train_pairs.empty else set()
        if not needed.issubset(seen) or len(train_pairs) < 100 or test.empty:
            raise RuntimeError(f"{variant} lacks strict-OOS pair data for {held:%Y-%m}")
        controls.append(base._incumbent_top2(test).assign(held_month=held.strftime("%Y-%m")))
        selected_features = pair_features
        if select:
            selected_features, trace, ranking = _select(train_pairs, pair_features, subspace_runs)
            feature_rows.append(pd.DataFrame({"variant": variant, "held_month": held.strftime("%Y-%m"), "feature": selected_features, "position": np.arange(len(selected_features))}))
            trace["variant"] = variant; trace["held_month"] = held.strftime("%Y-%m"); traces.append(trace)
            ranking["variant"] = variant; ranking["held_month"] = held.strftime("%Y-%m"); rankings.append(ranking)
        selection, proposed = _predict_selection(test, train_pairs, test_pairs, selected_features, 50.0)
        selection["held_month"] = held.strftime("%Y-%m"); selection["variant"] = variant; selections.append(selection)
        proposed["held_month"] = held.strftime("%Y-%m"); proposed["variant"] = variant; proposals.append(proposed)
    result = pd.concat(selections, ignore_index=True)
    control = pd.concat(controls, ignore_index=True)
    if result.candidate_id.duplicated().any() or control.candidate_id.duplicated().any():
        raise AssertionError("entry selection duplicated a strict-OOS candidate")
    return result, feature_rows, traces, rankings, control


def _scope_replays(selection: pd.DataFrame, labels: pd.DataFrame, arm: str, output: Path) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    for scope, scoped in (
        ("selection_apr_jul", selection.loc[pd.to_datetime(selection.__decision_ts__, utc=True).lt(SELECTION_END)].copy()),
        ("august_holdout", selection.loc[pd.to_datetime(selection.__decision_ts__, utc=True).ge(SELECTION_END)].copy()),
        ("all_oos", selection),
    ):
        if scoped.empty:
            continue
        result = base._replay(scoped, labels, f"{arm}__{scope}", output)
        result["model_arm"] = arm; result["evaluation_scope"] = scope
        summaries.append(result)
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-panel", type=Path, default=OLD_PANEL)
    parser.add_argument("--vwap-panel", type=Path, default=VWAP_PANEL)
    parser.add_argument("--labels-root", type=Path, default=LABEL_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--subspace-runs", type=int, default=200)
    parser.add_argument("--train-months", type=int, default=4, help="maximum trailing resolved calendar months for each strict-OOS fit")
    parser.add_argument("--held-month", action="append", help="optional repeatable YYYY-MM; default Jun--Aug 2026 (earlier reserve-pair support is insufficient for stable selection)")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"immutable output exists: {output}")
    panel = _candidate_frame(_load_panel(args.old_panel.resolve(), args.vwap_panel.resolve()))
    held_months = tuple(pd.Timestamp(f"{value}-01", tz="UTC") for value in args.held_month) if args.held_month else tuple(pd.date_range("2026-06-01", "2026-08-01", freq="MS", tz="UTC"))
    labels = base._labels(args.labels_root.resolve())
    labelled = panel.merge(labels, on="candidate_id", how="inner", validate="one_to_one")
    labelled = labelled.loc[labelled.policy_path_valid.fillna(False)].copy()
    labelled["policy_label_available_ts"] = pd.to_datetime(labelled.policy_label_available_ts, utc=True, errors="raise")
    old_raw_features = (*FIFTEEN_MINUTE_FEATURE_KEYS, *SCORE_FEATURES)
    vwap_raw_features = (*old_raw_features, *VWAP_15M_FEATURE_KEYS)
    old_features = base.PAIR_FEATURES
    vwap_features = (*old_features, *VWAP_15M_FEATURE_KEYS, *(f"margin__{feature}" for feature in VWAP_15M_FEATURE_KEYS))
    if set(vwap_features).difference(panel.columns) - {"incumbent_bcf_mc1_expected_bps", *(f"margin__{name}" for name in LEGACY_MARGIN_FEATURES), *(f"margin__{name}" for name in VWAP_15M_FEATURE_KEYS)}:
        raise AssertionError("base pair feature contract is not available in the target-free panel")
    output.mkdir(parents=True, exist_ok=False)
    variants = {
        "E0_old_all": (old_raw_features, old_features, False),
        "E1_old_fs": (old_raw_features, old_features, True),
        "E2_vwap_all": (vwap_raw_features, vwap_features, False),
        "E3_vwap_fs": (vwap_raw_features, vwap_features, True),
    }
    summaries: list[dict[str, object]] = []
    selections: dict[str, pd.DataFrame] = {}
    all_selected: list[pd.DataFrame] = []
    all_traces: list[pd.DataFrame] = []
    all_rankings: list[pd.DataFrame] = []
    control: pd.DataFrame | None = None
    for name, (raw_features, features, select) in variants.items():
        chosen, selected, traces, rankings, proposed_control = _run_variant(panel, labelled, name, raw_features, features, select=select, subspace_runs=args.subspace_runs, held_months=held_months, train_months=args.train_months)
        selections[name] = chosen
        summaries.extend(_scope_replays(chosen, labels, name, output))
        all_selected.extend(selected); all_traces.extend(traces); all_rankings.extend(rankings)
        if control is None:
            control = proposed_control
        elif not control.candidate_id.equals(proposed_control.candidate_id):
            raise AssertionError("feature variant changed the target-free incumbent control")
    assert control is not None
    summaries.extend(_scope_replays(control, labels, "B0_bcf_top2", output))
    summary = pd.DataFrame(summaries)
    summary["total_ev_per_abs_drawdown"] = summary.total_policy_net_bps / summary.max_drawdown.abs().replace(0.0, np.nan)
    for scope, group in summary.groupby("evaluation_scope", sort=False):
        baseline = group.loc[group.model_arm.eq("B0_bcf_top2")]
        if len(baseline) != 1:
            raise AssertionError(f"missing B0 control for {scope}")
        for metric in ("portfolio_accepted", "policy_net_bps_per_trade", "total_policy_net_bps", "max_drawdown", "worst_week", "total_ev_per_abs_drawdown"):
            summary.loc[group.index, f"delta_vs_B0_{metric}"] = group[metric] - baseline.iloc[0][metric]
    summary.to_parquet(output / "portfolio_summary.parquet", index=False)
    selection = summary.loc[summary.evaluation_scope.eq("selection_apr_jul") & ~summary.model_arm.eq("B0_bcf_top2")].sort_values(["total_ev_per_abs_drawdown", "policy_net_bps_per_trade", "worst_week"], ascending=[False, False, False], kind="stable")
    selection.to_parquet(output / "selection_ranking_apr_jul.parquet", index=False)
    monthly = pd.concat([pd.read_parquet(path) for path in output.glob("*__all_oos_monthly.parquet")], ignore_index=True)
    monthly.to_parquet(output / "monthly_metrics.parquet", index=False)
    for name, frame in selections.items():
        frame.to_parquet(output / f"{name}_selection_target_free.parquet", index=False, compression="zstd")
    control.to_parquet(output / "B0_bcf_top2_selection_target_free.parquet", index=False, compression="zstd")
    if all_selected:
        pd.concat(all_selected, ignore_index=True).to_parquet(output / "stable_selected_features.parquet", index=False)
        pd.concat(all_traces, ignore_index=True).to_parquet(output / "selection_trace.parquet", index=False)
        pd.concat(all_rankings, ignore_index=True).to_parquet(output / "subspace_stability.parquet", index=False)
    manifest = {
        "schema": "strict-r3-p8u-entry-feature-contract-ablation-v1",
        "scope": "offline strict-OOS challenger only; no live/canonical mutation",
        "entry_authority": "one 20--30 bps reserve may replace only the timestamp marginal 30-bps incumbent; no capacity expansion",
        "model": "LightGBM median (q50) pairwise advantage, 2x upper-reserve weighting, +50 bps replacement margin",
        "selection_order": "total policy net bps / absolute max drawdown, then policy EV/trade, then worst week",
        "selection_period": "2026-06 through 2026-07; April--May have insufficient strictly prior 20--30 reserve/incumbent pairs for stable random-subspace selection, and August is an untouched holdout",
        "fold": f"up to {args.train_months} trailing complete preceding calendar months; pair labels must resolve before the held boundary",
        "feature_selection": "200 random subspaces, 30--60% optional features, stability score then conditional-MI/OOF greedy selection; 30--45 final features",
        "old_panel": str(args.old_panel.resolve()), "old_panel_sha256": _sha256(args.old_panel.resolve()),
        "vwap_panel": str(args.vwap_panel.resolve()), "vwap_panel_sha256": _sha256(args.vwap_panel.resolve()),
        "labels_root": str(args.labels_root.resolve()),
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
