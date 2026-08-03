#!/usr/bin/env python3
"""Immutable fixed nonlinear alpha-tail, cost-clearing hurdle diagnostic.

This research-only ablation consumes the strict A-grade identity intersections
and sealed v5 residual/linear candidate ledgers.  It adds no HPO and no
portfolio replay: the only challenger is a fixed side-aware logistic hurdle
over deterministic timestamp-by-side alpha ranks, ventiles and tail hinges.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import run_a_grade_cost_clearing_conversion_ablation as v5

OUT = ROOT / "data_perp/artifacts/nonlinear_alpha_tail_cost_clearing_hurdle_20260730_v1"
V5 = ROOT / "data_perp/artifacts/a_grade_cost_clearing_conversion_ablation_20260730_v5"
SCHEMA = "nonlinear_alpha_tail_cost_clearing_hurdle_v1"
ARMS = ("baseline_residual_ev", "hurdle_alpha", "nonlinear_alpha_tail_hurdle")
TOP, VENTILES = 0.10, 20
TAIL_HINGES = (0.80, 0.90, 0.95)
ID = v5.ID
NET, GROSS, COST, SCORE, ALPHA = v5.NET, v5.GROSS, v5.COST, v5.SCORE, v5.ALPHA


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for part in iter(lambda: handle.read(1 << 20), b""):
            digest.update(part)
    return digest.hexdigest()


def safe(value: Any) -> Any:
    if isinstance(value, dict): return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [safe(v) for v in value]
    if isinstance(value, (Path, pd.Timestamp)): return str(value)
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, float) and not np.isfinite(value): return None
    return value


def write_json(path: Path, payload: Any) -> None:
    partial = path.with_name(f".{path.name}.{os.getpid()}.partial")
    partial.write_text(json.dumps(safe(payload), indent=2, sort_keys=True) + "\n")
    os.replace(partial, path)


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    partial = path.with_name(f".{path.name}.{os.getpid()}.partial")
    frame.to_parquet(partial, index=False)
    os.replace(partial, path)


def verify_v5() -> dict[str, Any]:
    manifest_path, seal = V5 / "manifest.json", V5 / "manifest.sha256"
    if not manifest_path.is_file() or not seal.is_file():
        raise ValueError("sealed v5 artifact is missing")
    if seal.read_text().split()[0] != sha(manifest_path):
        raise ValueError("v5 manifest seal mismatch")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "a_grade_cost_clearing_conversion_ablation_v2_manifest":
        raise ValueError("unexpected v5 schema")
    current_runner = sha(ROOT / "scripts/run_a_grade_cost_clearing_conversion_ablation.py")
    if manifest.get("runner_sha256") != current_runner:
        raise ValueError("v5 runner hash differs from current cost-clearing runner")
    for name, expected in manifest.get("outputs_sha256", {}).items():
        if sha(V5 / name) != expected:
            raise ValueError(f"v5 output hash mismatch: {name}")
    return {"artifact": str(V5), "manifest_sha256": sha(manifest_path),
            "current_runner_sha256": current_runner, "outputs_verified": len(manifest["outputs_sha256"])}


def alpha_rank_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Rank alpha inside timestamp×side with one stable, cross-era definition."""
    work = frame.copy()
    work["__alpha_row__"] = np.arange(len(work), dtype=np.int64)
    ordered = work.sort_values(
        ["__ts__", "side_name", ALPHA, "__symbol__", "candidate_id"],
        ascending=[True, True, False, True, True], kind="stable",
    ).copy()
    grouped = ordered.groupby(["__ts__", "side_name"], sort=False, observed=True)
    ordered["alpha_rank_timestamp_side"] = grouped.cumcount().add(1).astype(int)
    ordered["alpha_rank_population_timestamp_side"] = grouped["candidate_id"].transform("size").astype(int)
    # High alpha is high percentile.  Mid-rank convention avoids zero/one
    # values while preserving exact rank order and repeated-timestamp semantics.
    ordered["alpha_percentile_timestamp_side"] = 1.0 - (
        (ordered["alpha_rank_timestamp_side"] - 0.5) /
        ordered["alpha_rank_population_timestamp_side"]
    )
    ordered["alpha_ventile_timestamp_side"] = np.minimum(
        VENTILES,
        np.ceil(ordered["alpha_rank_timestamp_side"] * VENTILES /
                ordered["alpha_rank_population_timestamp_side"]).astype(int),
    )
    for hinge in TAIL_HINGES:
        ordered[f"alpha_tail_hinge_{int(hinge * 100):02d}"] = np.maximum(
            0.0, ordered["alpha_percentile_timestamp_side"] - hinge,
        )
    return ordered.sort_values("__alpha_row__", kind="stable").drop(columns="__alpha_row__")


def nonlinear_x(frame: pd.DataFrame) -> np.ndarray:
    pct = frame["alpha_percentile_timestamp_side"].to_numpy(float)
    side = frame.side_name.eq("long").astype(float).to_numpy()
    ventile = frame["alpha_ventile_timestamp_side"].to_numpy(int)
    dummy = np.column_stack([(ventile == value).astype(float) for value in range(1, VENTILES + 1)])
    tails = np.column_stack([frame[f"alpha_tail_hinge_{int(h * 100):02d}"].to_numpy(float) for h in TAIL_HINGES])
    nonlinear = np.column_stack([pct, tails, dummy])
    # Side is explicit and every rank-shape term has a side interaction.
    return np.column_stack([frame[SCORE].to_numpy(float), side, nonlinear, nonlinear * side[:, None]])


def nonlinear_score(train: pd.DataFrame, test: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    target = train[NET].gt(0).astype(int).to_numpy()
    if target.min() == target.max():
        probability = np.full(len(test), float(target.mean()))
    else:
        model = make_pipeline(StandardScaler(), LogisticRegression(
            C=.25, max_iter=300, class_weight="balanced", random_state=20260730,
        ))
        model.fit(nonlinear_x(train), target)
        probability = model.predict_proba(nonlinear_x(test))[:, 1]
    positive, negative = train.loc[train[NET].gt(0), NET], train.loc[train[NET].le(0), NET]
    fallback = (float(positive.mean()), float(negative.mean()))
    payoffs: dict[str, tuple[float, float]] = {}
    for side in test.side_name.drop_duplicates():
        local = train.loc[train.side_name.eq(side), NET]
        payoffs[str(side)] = (
            float(local.loc[local.gt(0)].mean()) if local.gt(0).any() else fallback[0],
            float(local.loc[local.le(0)].mean()) if local.le(0).any() else fallback[1],
        )
    payoff = np.asarray([payoffs[str(side)] for side in test.side_name])
    return probability * payoff[:, 0] + (1.0 - probability) * payoff[:, 1], {
        "model": "fixed_standardized_logistic_side_aware_alpha_rank_ventile_tail_hinges_plus_train_side_payoff",
        "positive_rate_train": float(target.mean()), "feature_contract": "residual|side|timestamp_side_alpha_percentile|20_ventiles|tail_80_90_95|side_interactions",
    }


def causal_map(prior: pd.DataFrame, score: np.ndarray, start: pd.Timestamp) -> tuple[np.ndarray, dict[str, Any]]:
    ref = prior.loc[pd.to_datetime(prior.execution_label_end_utc, utc=True) < start].copy()
    if len(ref) < v5.MIN_MAP or ref.raw_score.nunique() < 2:
        return np.full(len(score), np.nan), {"map_eligible": False, "map_reference_rows": int(len(ref)), "latest_reference_label_end_utc": ref.execution_label_end_utc.max() if len(ref) else None}
    mapper = IsotonicRegression(out_of_bounds="clip").fit(ref.raw_score.to_numpy(float), ref[NET].to_numpy(float))
    return mapper.predict(score), {"map_eligible": True, "map_reference_rows": int(len(ref)), "latest_reference_label_end_utc": ref.execution_label_end_utc.max()}


def score_oof_nonlinear(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = alpha_rank_features(frame).sort_values(["__ts__", "candidate_id"], kind="stable")
    prior: list[pd.DataFrame] = []
    rows: list[pd.DataFrame] = []
    audit: list[dict[str, Any]] = []
    for block in v5.block_starts(frame):
        test = frame.loc[frame.__ts__.ge(block) & frame.__ts__.lt(block + pd.Timedelta(days=v5.BLOCK_DAYS))].copy()
        train = frame.loc[(frame.__ts__ < block) & (frame.execution_label_end_utc < block)].copy()
        if len(train) < v5.MIN_TRAIN:
            audit.append({"lineage_id": frame.lineage_id.iloc[0], "outer_block_start_utc": block, "status": "warmup_unscored", "train_rows": len(train), "test_rows": len(test), "causal_train_max_label_end_utc": train.execution_label_end_utc.max() if len(train) else None})
            continue
        historical = pd.concat(prior, ignore_index=True) if prior else pd.DataFrame(columns=["execution_label_end_utc", "raw_score", NET])
        raw, model = nonlinear_score(train, test)
        mapped, mapping = causal_map(historical, raw, block)
        out = test.loc[:, [*ID, "lineage_id", "evidence_grade", "execution_label_end_utc", GROSS, COST, NET, "execution_exit_reason", "alpha_percentile_timestamp_side", "alpha_ventile_timestamp_side"]].copy()
        out["arm"] = "nonlinear_alpha_tail_hurdle"; out["outer_block_start_utc"] = block
        out["raw_score"] = raw; out["mapped_ev"] = mapped; out["map_eligible"] = mapping["map_eligible"]; out["map_reference_rows"] = mapping["map_reference_rows"]
        out["evaluation_role"] = "BLOCKED_2025_OOF" if frame["__ts__"].dt.year.iloc[0] == 2025 else "WITHIN_LINEAGE_CHRONOLOGICAL_OOF_DIAGNOSTIC"
        rows.append(out); prior.append(out.loc[:, ["execution_label_end_utc", "raw_score", NET]])
        audit.append({"lineage_id": frame.lineage_id.iloc[0], "outer_block_start_utc": block, "status": "scored", "train_rows": len(train), "test_rows": len(test), "causal_train_max_label_end_utc": train.execution_label_end_utc.max(), **model, **mapping})
    return (pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(), pd.DataFrame(audit))


def same_ids(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    return left.loc[:, list(ID)].sort_values(list(ID), kind="stable").reset_index(drop=True).equals(right.loc[:, list(ID)].sort_values(list(ID), kind="stable").reset_index(drop=True))


def fractional_top(frame: pd.DataFrame, score: str, fraction: float = TOP) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Pooled top-k with exact cutoff-tie mass, never a side quota."""
    work = frame.sort_values([score, "__symbol__", "candidate_id"], ascending=[False, True, True], kind="stable").reset_index(drop=True)
    target = len(work) * fraction
    weights = np.zeros(len(work), dtype=float)
    cumulative = 0.0; cutoff_rows = 0; cutoff_score = None
    for value, index in work.groupby(score, sort=False, dropna=False).groups.items():
        positions = work.index.get_indexer(index)
        count = len(positions)
        if cumulative + count <= target + 1e-12:
            weights[positions] = 1.0; cumulative += count
        elif cumulative < target - 1e-12:
            weights[positions] = (target - cumulative) / count
            cutoff_rows, cutoff_score = count, value
            cumulative = target
            break
        else:
            break
    work["selection_weight"] = weights
    chosen = work.loc[work.selection_weight.gt(0)].copy()
    return chosen, {"candidate_rows": len(work), "target_selection_mass": target, "realized_selection_mass": float(chosen.selection_weight.sum()), "cutoff_tie_rows": cutoff_rows, "cutoff_tie_fraction": cutoff_rows / len(work) if len(work) else 0.0, "cutoff_score": cutoff_score, "selection_rule": "one_pooled_global_top10_exact_fractional_cutoff_tie_no_side_quota"}


def rank_ic(frame: pd.DataFrame, score: str) -> float:
    if len(frame) < 2 or frame[score].nunique() < 2 or frame[NET].nunique() < 2: return float("nan")
    return float(frame[score].corr(frame[NET], method="spearman"))


def weighted_mean(frame: pd.DataFrame, column: str) -> float:
    return float(np.average(frame[column], weights=frame.selection_weight)) if len(frame) and frame.selection_weight.sum() else float("nan")


def select_and_reports(scores: pd.DataFrame, role: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    eligible = scores.loc[scores.map_eligible].copy(); eligible["month"] = eligible.__ts__.dt.strftime("%Y-%m")
    selected: list[pd.DataFrame] = []; metrics: list[dict[str, Any]] = []; ties: list[dict[str, Any]] = []; tails: list[dict[str, Any]] = []
    for (lineage, arm, month), group in eligible.groupby(["lineage_id", "arm", "month"], sort=True, observed=True):
        top, tie = fractional_top(group, "mapped_ev")
        top["selection_month"] = month; selected.append(top)
        ties.append({"evaluation_role": role, "lineage_id": lineage, "arm": arm, "period": month, **tie})
        metrics.append({"evaluation_role": role, "lineage_id": lineage, "arm": arm, "period": month, "candidate_rows": len(group), "selected_mass": float(top.selection_weight.sum()), "selected_rows_with_fraction": len(top), "mapped_top10_raw_ic": rank_ic(top, "raw_score"), "mapped_top10_mapped_ic": rank_ic(top, "mapped_ev"), "mean_gross_bps": weighted_mean(top, GROSS)*1e4, "mean_cost_bps": weighted_mean(top, COST)*1e4, "mean_net_bps": weighted_mean(top, NET)*1e4, "waterfall_reconciliation_bps": (weighted_mean(top, GROSS)-weighted_mean(top, COST)-weighted_mean(top, NET))*1e4})
        for fraction in (.01, .05, .10, .20):
            for basis in ("raw_score", "mapped_ev"):
                tail, audit = fractional_top(group, basis, fraction)
                tails.append({"evaluation_role": role, "lineage_id": lineage, "arm": arm, "month": month, "tail_fraction": fraction, "ranking_basis": basis, "candidate_rows": len(group), "selection_mass": float(tail.selection_weight.sum()), "raw_score_ic": rank_ic(tail, "raw_score"), "mapped_ev_ic": rank_ic(tail, "mapped_ev"), "mean_gross_bps": weighted_mean(tail, GROSS)*1e4, "mean_cost_bps": weighted_mean(tail, COST)*1e4, "mean_net_bps": weighted_mean(tail, NET)*1e4, "waterfall_reconciliation_bps": (weighted_mean(tail, GROSS)-weighted_mean(tail, COST)-weighted_mean(tail, NET))*1e4, **{f"tie_{k}": v for k, v in audit.items() if k in ("cutoff_tie_rows", "cutoff_tie_fraction")}})
    return pd.concat(selected, ignore_index=True), pd.DataFrame(metrics), pd.DataFrame(ties), pd.DataFrame(tails)


def decile_calibration(scores: pd.DataFrame, role: str) -> pd.DataFrame:
    rows = []
    for (lineage, arm), group in scores.groupby(["lineage_id", "arm"], sort=True, observed=True):
        for basis in ("raw_score", "mapped_ev"):
            order = group.sort_values([basis, "__symbol__", "candidate_id"], ascending=[False, True, True], kind="stable").copy()
            order["decile"] = np.minimum(10, np.ceil((np.arange(len(order)) + 1) * 10 / len(order)).astype(int))
            for decile, part in order.groupby("decile", sort=True):
                pos, neg = part.loc[part[NET].gt(0), NET], part.loc[part[NET].le(0), NET]
                rows.append({"evaluation_role": role, "lineage_id": lineage, "arm": arm, "score_basis": basis, "decile": decile, "rows": len(part), "mean_gross_bps": part[GROSS].mean()*1e4, "mean_cost_bps": part[COST].mean()*1e4, "mean_net_bps": part[NET].mean()*1e4, "positive_rate": part[NET].gt(0).mean(), "conditional_positive_net_bps": pos.mean()*1e4 if len(pos) else float("nan"), "conditional_nonpositive_net_bps": neg.mean()*1e4 if len(neg) else float("nan")})
    return pd.DataFrame(rows)


def attribution(selected: pd.DataFrame, role: str) -> pd.DataFrame:
    rows=[]
    for dimensions in (("side_name",), ("side_name", "execution_exit_reason")):
        for key, group in selected.groupby(["evaluation_role", "lineage_id", "arm", *dimensions], sort=True, observed=True):
            keys = key if isinstance(key, tuple) else (key,)
            row = dict(zip(["evaluation_role", "lineage_id", "arm", *dimensions], keys))
            row.update({"selection_mass": float(group.selection_weight.sum()), "mean_gross_bps": weighted_mean(group, GROSS)*1e4, "mean_cost_bps": weighted_mean(group, COST)*1e4, "mean_net_bps": weighted_mean(group, NET)*1e4, "reconciliation_bps": (weighted_mean(group, GROSS)-weighted_mean(group, COST)-weighted_mean(group, NET))*1e4})
            rows.append(row)
    return pd.DataFrame(rows)


def numeric_quantile_migration(scores: pd.DataFrame, cuts: np.ndarray) -> pd.DataFrame:
    frame = scores.copy()
    frame["fixed_numeric_alpha_ventile"] = np.searchsorted(cuts[1:-1], frame[ALPHA].to_numpy(float), side="right") + 1
    rows=[]
    for (role, lineage, arm, numeric, ranked), group in frame.groupby(["evaluation_role", "lineage_id", "arm", "fixed_numeric_alpha_ventile", "alpha_ventile_timestamp_side"], observed=True, sort=True):
        rows.append({"evaluation_role":role,"lineage_id":lineage,"arm":arm,"fixed_numeric_alpha_ventile":numeric,"timestamp_side_rank_ventile":ranked,"rows":len(group),"mean_net_bps":group[NET].mean()*1e4})
    return pd.DataFrame(rows)


def gate_table(forward_metrics: pd.DataFrame, forward_selected: pd.DataFrame, ties: pd.DataFrame, parity: pd.DataFrame, causal: pd.DataFrame) -> pd.DataFrame:
    nl = forward_metrics.loc[forward_metrics.arm.eq("nonlinear_alpha_tail_hurdle")]
    gates=[]
    gates.append({"gate":"causal_legality", "pass": bool(causal.causal_pass.all()), "detail":"every 2025 train/map label end strictly precedes fold start"})
    gates.append({"gate":"identity_parity", "pass": bool(parity.identity_parity.all()), "detail":"same eligible IDs across residual, v5 linear and nonlinear arms"})
    gates.append({"gate":"global_selection", "pass": True, "detail":"one pooled global top-k after arm-local mapping; no side quota"})
    gates.append({"gate":"positive_mapped_top10_aggregate_latest_worst", "pass": bool((nl.mean_net_bps > 0).all()) and len(nl)>0, "detail":"all forward monthly aggregate/latest/worst are positive (one row per month)"})
    improve = True
    # Explicit per-control lookup avoids any implicit metric pooling.
    for (_, period), row in nl.set_index(["lineage_id", "period"]).iterrows():
        lineage = row.name[0]
        for arm in ("baseline_residual_ev", "hurdle_alpha"):
            base = forward_metrics.loc[(forward_metrics.lineage_id.eq(lineage)) & (forward_metrics.period.eq(period)) & forward_metrics.arm.eq(arm), "mean_net_bps"]
            improve = improve and len(base) == 1 and float(row.mean_net_bps) > float(base.iloc[0])
    gates.append({"gate":"improves_identically_mapped_residual_and_v5_linear", "pass": bool(improve), "detail":"nonlinear mapped top10 net exceeds each control in every forward month"})
    side = forward_selected.loc[forward_selected.arm.eq("nonlinear_alpha_tail_hurdle")].groupby("side_name", observed=True).apply(lambda x: weighted_mean(x, NET)*1e4)
    gates.append({"gate":"both_sides_positive", "pass": bool(set(side.index)=={"long","short"} and (side > 0).all()), "detail":"weighted mapped top10 net is positive for both sides"})
    gates.append({"gate":"tie_at_most_5pct", "pass": bool((ties.cutoff_tie_fraction <= .05).all()), "detail":"every pooled cutoff tie is at most five percent of its candidate population"})
    gates.append({"gate":"frozen_2026_forward_legality", "pass": True, "detail":"2026 scored with frozen 2025 full fit and 2025 blocked-OOF map; no 2026 labels in fit/map"})
    return pd.DataFrame(gates)


def run(output: Path = OUT) -> dict[str, Any]:
    if output.exists(): raise FileExistsError(output)
    staging = output.with_name(f".{output.name}.{os.getpid()}.partial")
    if staging.exists(): raise FileExistsError(staging)
    staging.mkdir(parents=True)
    try:
        verification = verify_v5()
        lineages, sources = v5.load_lineages()
        historical_name = next(name for name in lineages if name.startswith("canonical_marapr"))
        current_name = next(name for name in lineages if name.startswith("current_mayjul"))
        historical, current = alpha_rank_features(lineages[historical_name]), alpha_rank_features(lineages[current_name])
        oof_nl, causal = score_oof_nonlinear(historical)
        v5_oof = pd.read_parquet(V5 / "within_lineage_candidate_scores.parquet")
        controls_oof = v5_oof.loc[v5_oof.arm.isin(("baseline_residual_ev", "hurdle_alpha")) & v5_oof.lineage_id.eq(historical_name)].copy()
        for item in (controls_oof,):
            item["evaluation_role"] = "BLOCKED_2025_OOF"
        # v5 scores have no alpha-rank columns; attach the immutable feature ledger by identity.
        feature_columns = [*ID, ALPHA, "alpha_percentile_timestamp_side", "alpha_ventile_timestamp_side", "execution_exit_reason"]
        controls_oof = controls_oof.merge(historical.loc[:, feature_columns], on=list(ID), how="inner", validate="many_to_one")
        assert same_ids(controls_oof.loc[controls_oof.arm.eq("baseline_residual_ev")], oof_nl)
        assert same_ids(controls_oof.loc[controls_oof.arm.eq("hurdle_alpha")], oof_nl)
        oof = pd.concat([controls_oof, oof_nl.assign(**{ALPHA: oof_nl.merge(historical.loc[:, [*ID, ALPHA]], on=list(ID), how="left", validate="one_to_one")[ALPHA].to_numpy()})], ignore_index=True)
        # The final 2025 fit is allowed to use resolved 2025 labels; map support
        # remains blocked-OOF predictions only and is frozen before 2026 scoring.
        raw26, model26 = nonlinear_score(historical, current)
        nl_prior = oof_nl.loc[:, ["execution_label_end_utc", "raw_score", NET]].copy()
        mapper = IsotonicRegression(out_of_bounds="clip").fit(nl_prior.raw_score, nl_prior[NET])
        forward_nl = current.loc[:, [*ID, "lineage_id", "evidence_grade", "execution_label_end_utc", GROSS, COST, NET, "execution_exit_reason", ALPHA, "alpha_percentile_timestamp_side", "alpha_ventile_timestamp_side"]].copy()
        forward_nl["arm"] = "nonlinear_alpha_tail_hurdle"; forward_nl["raw_score"] = raw26; forward_nl["mapped_ev"] = mapper.predict(raw26); forward_nl["map_eligible"] = True; forward_nl["map_reference_rows"] = len(nl_prior); forward_nl["outer_block_start_utc"] = pd.Timestamp("2026-01-01T00:00:00Z"); forward_nl["evaluation_role"] = "FROZEN_2025_FIT_AND_BLOCKED_OOF_MAP_TO_2026"
        controls_forward = pd.read_parquet(V5 / "strict_forward_2026_candidate_scores.parquet")
        controls_forward = controls_forward.loc[controls_forward.arm.isin(("baseline_residual_ev", "hurdle_alpha"))].copy()
        controls_forward["evaluation_role"] = "FROZEN_2025_FIT_AND_BLOCKED_OOF_MAP_TO_2026"
        controls_forward = controls_forward.merge(current.loc[:, feature_columns], on=list(ID), how="inner", validate="many_to_one")
        assert same_ids(controls_forward.loc[controls_forward.arm.eq("baseline_residual_ev")], forward_nl)
        assert same_ids(controls_forward.loc[controls_forward.arm.eq("hurdle_alpha")], forward_nl)
        forward = pd.concat([controls_forward, forward_nl], ignore_index=True)
        # Same eligible identity intersection is a hard, separately auditable contract.
        parity_rows=[]
        for role, table in (("oof",oof),("forward",forward)):
            sets={arm:set(table.loc[table.arm.eq(arm)&table.map_eligible,"candidate_id"]) for arm in ARMS}
            parity_rows.append({"evaluation_role":role,"identity_parity":sets[ARMS[0]]==sets[ARMS[1]]==sets[ARMS[2]], **{f"{arm}_eligible_rows":len(ids) for arm,ids in sets.items()}})
        parity=pd.DataFrame(parity_rows)
        causal["causal_pass"] = causal.apply(lambda x: x.status != "scored" or pd.Timestamp(x.causal_train_max_label_end_utc) < pd.Timestamp(x.outer_block_start_utc), axis=1)
        oof_selected, oof_metrics, oof_ties, oof_tails = select_and_reports(oof, "BLOCKED_2025_OOF")
        fwd_selected, fwd_metrics, fwd_ties, fwd_tails = select_and_reports(forward, "FROZEN_2025_FIT_AND_BLOCKED_OOF_MAP_TO_2026")
        selected = pd.concat([oof_selected, fwd_selected], ignore_index=True)
        selected["evaluation_role"] = selected.get("evaluation_role", "")
        selected.loc[selected.lineage_id.eq(historical_name), "evaluation_role"] = "BLOCKED_2025_OOF"
        selected.loc[selected.lineage_id.eq(current_name), "evaluation_role"] = "FROZEN_2025_FIT_AND_BLOCKED_OOF_MAP_TO_2026"
        all_scores=pd.concat([oof,forward],ignore_index=True)
        cuts=np.quantile(historical[ALPHA].to_numpy(float), np.linspace(0,1,VENTILES+1))
        migration=numeric_quantile_migration(all_scores, cuts)
        calibration=pd.concat([decile_calibration(oof,"BLOCKED_2025_OOF"), decile_calibration(forward,"FROZEN_2025_FIT_AND_BLOCKED_OOF_MAP_TO_2026")],ignore_index=True)
        attr=attribution(selected, "all")
        gates=gate_table(fwd_metrics,fwd_selected,fwd_ties,parity,causal)
        availability=pd.DataFrame([{"arm":"baseline_residual_ev","status":"sealed_v5_control_available","fit_labels":"v5 sealed 2025","map_labels":"v5 sealed blocked OOF 2025"},{"arm":"hurdle_alpha","status":"sealed_v5_linear_control_available","fit_labels":"v5 sealed 2025","map_labels":"v5 sealed blocked OOF 2025"},{"arm":"nonlinear_alpha_tail_hurdle","status":"strict_forward_available","fit_labels":"2025 only","map_labels":"2025 blocked OOF only","model":model26["model"]}])
        outputs={"oof_candidate_scores.parquet":oof,"forward_2026_candidate_scores.parquet":forward,"oof_fold_causal_audit.csv":causal,"identity_parity.csv":parity,"oof_mapped_top10_metrics.csv":oof_metrics,"forward_2026_mapped_top10_metrics.csv":fwd_metrics,"tail_local_ic_waterfall.csv":pd.concat([oof_tails,fwd_tails],ignore_index=True),"fixed_numeric_vs_quantile_migration.csv":migration,"decile_calibration_conditional_payoff.csv":calibration,"side_exit_attribution_reconciliation.csv":attr,"tie_audit.csv":pd.concat([oof_ties,fwd_ties],ignore_index=True),"arm_availability.csv":availability,"gates.csv":gates,"selected_fractional_top10.parquet":selected}
        for name, table in outputs.items():
            (atomic_parquet(table, staging/name) if name.endswith(".parquet") else table.to_csv(staging/name,index=False))
        input_hashes={str(path):sha(path) for path in (v5.EXACT_2025,v5.EXACT_2026,v5.CONTEXT_2025,v5.REGIME_2026,v5.TRANSITION_2026,V5/"manifest.json")}
        report={"schema":SCHEMA,"status":"SEALED_DIAGNOSTIC_NON_PROMOTION","promotion_eligible":False,"v5_verification":verification,"input_sha256":input_hashes,"fixed_contract":{"rank":"timestamp×side descending score_base_alpha; ties ascending symbol then candidate_id; stable sort","rank_features":"mid-rank high-alpha percentile, 20 ventiles, fixed 80/90/95 percentile hinges","linear_control":"sealed v5 hurdle_alpha candidate scores","mapping":"OOF arm-local isotonic uses only earlier blocked-OOF rows with execution_label_end_utc strictly before fold start; 2026 map frozen from 2025 blocked OOF","selection":"one pooled global top10 after mapping; no side quota; exact fractional cutoff ties","scope":"research-only; no portfolio replay"},"numeric_alpha_ventile_cutpoints_from_2025":cuts.tolist(),"gates":gates.to_dict("records"),"outputs_sha256":{path.name:sha(path) for path in staging.iterdir() if path.is_file()}}
        write_json(staging/"report.json",report)
        manifest={"schema":SCHEMA+"_manifest","status":"SEALED_DIAGNOSTIC_NON_PROMOTION","runner_sha256":sha(Path(__file__)),"input_sha256":input_hashes,"outputs_sha256":{path.name:sha(path) for path in staging.iterdir() if path.is_file()}}
        write_json(staging/"manifest.json",manifest); (staging/"manifest.sha256").write_text(f"{sha(staging/'manifest.json')}  manifest.json\n")
        staging.replace(output); return manifest
    except Exception:
        shutil.rmtree(staging,ignore_errors=True); raise


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--output",type=Path,default=OUT)
    args=parser.parse_args(); print(json.dumps(safe(run(args.output)),indent=2))


if __name__ == "__main__": main()
