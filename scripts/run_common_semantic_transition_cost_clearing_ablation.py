#!/usr/bin/env python3
"""Strict common-semantic transition-geometry cost-clearing ablation.

This is an immutable research-only rerun of the sealed A-grade conversion
contract.  It uses only the 90 semantic-common decision-time geometry columns
from the sealed current-v4 context: state level/gap, lag/delta, or both.  The
join is exact on signal timestamp and deliberately uses no fill or as-of rule.
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
from scripts import run_nonlinear_alpha_tail_cost_clearing_hurdle as common
from scripts.materialize_historical_current_common_transition_geometry import CANONICAL_FEATURES

OUT = ROOT / "data_perp/artifacts/common_semantic_transition_cost_clearing_ablation_20260730_v1"
V5 = ROOT / "data_perp/artifacts/a_grade_cost_clearing_conversion_ablation_20260730_v5"
GEOMETRY = ROOT / "data_perp/artifacts/historical_current_common_transition_geometry_20260730_v1"
CONTEXT = GEOMETRY / "current_v4_semantic_context.parquet"
SCHEMA = "common_semantic_transition_cost_clearing_ablation_v1"
ARMS = ("baseline_residual_ev", "hurdle_alpha", "common_state_hurdle", "common_transition_hurdle", "common_state_transition_hurdle")
STATE = tuple(c for c in CANONICAL_FEATURES if "__state_mean__" in c or "__state_long_short_gap__" in c)
TRANSITION = tuple(c for c in CANONICAL_FEATURES if "__past_delta_" in c)
ID = v5.ID
NET, GROSS, COST, SCORE, ALPHA = v5.NET, v5.GROSS, v5.COST, v5.SCORE, v5.ALPHA
ROLE_OOF = "BLOCKED_2025_OOF"
ROLE_FORWARD = "FROZEN_2025_FULL_FIT_AND_BLOCKED_OOF_MAP_TO_2026"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def safe(value: Any) -> Any:
    if isinstance(value, dict): return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [safe(v) for v in value]
    if isinstance(value, (Path, pd.Timestamp)): return str(value)
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, float) and not np.isfinite(value): return None
    return value


def write_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    temporary.write_text(json.dumps(safe(payload), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def verify_sealed_inputs() -> dict[str, Any]:
    v5_check = common.verify_v5()
    manifest_path, sidecar = GEOMETRY / "manifest.json", GEOMETRY / "manifest.sha256"
    if not manifest_path.is_file() or not sidecar.is_file():
        raise ValueError("sealed common geometry is missing")
    if sidecar.read_text().split()[0] != sha(manifest_path):
        raise ValueError("common geometry manifest seal mismatch")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "historical_current_common_transition_geometry_v1":
        raise ValueError("unexpected common geometry schema")
    for item in ("current_v4_semantic_context", "historical_candidate_context", "historical_hourly_state_geometry"):
        meta = manifest.get("outputs", {}).get(item, {})
        path = Path(meta.get("path", ""))
        if not path.is_file() or sha(path) != meta.get("sha256"):
            raise ValueError(f"common geometry output hash mismatch: {item}")
    mapping = manifest.get("semantic_mapping", {})
    mapping_path = GEOMETRY / mapping.get("path", "semantic_mapping.json")
    if not mapping_path.is_file() or sha(mapping_path) != mapping.get("sha256"):
        raise ValueError("common geometry semantic mapping hash mismatch")
    if len(CANONICAL_FEATURES) != 90 or len(STATE) != 36 or len(TRANSITION) != 54:
        raise ValueError("unexpected common semantic feature-family dimensions")
    return {"v5": v5_check, "common_geometry_manifest_sha256": sha(manifest_path),
            "common_geometry_context_sha256": sha(CONTEXT), "semantic_mapping_sha256": sha(mapping_path),
            "common_geometry_schema": manifest["schema"]}


def load_common_context() -> pd.DataFrame:
    context = pd.read_parquet(CONTEXT)
    required = {"signal_context_utc", "common_transition_context_available", *CANONICAL_FEATURES}
    missing = sorted(required.difference(context.columns))
    if missing: raise ValueError(f"common context missing columns: {missing}")
    context = context.loc[:, ["signal_context_utc", "common_transition_context_available", *CANONICAL_FEATURES]].copy()
    context["signal_context_utc"] = pd.to_datetime(context["signal_context_utc"], utc=True, errors="raise")
    if context.signal_context_utc.duplicated().any(): raise ValueError("common context timestamp is not unique")
    if not context.common_transition_context_available.eq(True).all(): raise ValueError("common context contains unavailable rows")
    return context


def exact_timestamp_join(exact: pd.DataFrame, context: pd.DataFrame, *, lineage: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Join only equal signal timestamps; audit all missing/no-fill exclusions."""
    left = exact.copy()
    left["__ts__"] = pd.to_datetime(left["__ts__"], utc=True, errors="raise")
    joined = left.merge(context, left_on="__ts__", right_on="signal_context_utc", how="left", validate="many_to_one", indicator="__common_join__")
    feature_present = joined.loc[:, list(CANONICAL_FEATURES)].notna().all(axis=1)
    exact_timestamp = joined["__common_join__"].eq("both")
    eligible = exact_timestamp & feature_present
    audit = pd.DataFrame([{
        "lineage_id": lineage, "exact_candidate_rows": len(joined), "exact_signal_timestamps": int(left.__ts__.nunique()),
        "common_context_timestamps": int(context.signal_context_utc.nunique()),
        "matched_candidate_rows": int(exact_timestamp.sum()), "matched_signal_timestamps": int(joined.loc[exact_timestamp, "__ts__"].nunique()),
        "missing_timestamp_candidate_rows": int((~exact_timestamp).sum()), "missing_signal_timestamps": int(left.loc[~exact_timestamp, "__ts__"].nunique()),
        "feature_incomplete_candidate_rows": int((exact_timestamp & ~feature_present).sum()),
        "feature_incomplete_signal_timestamps": int(joined.loc[exact_timestamp & ~feature_present, "__ts__"].nunique()),
        "eligible_complete_case_candidate_rows": int(eligible.sum()), "eligible_complete_case_signal_timestamps": int(joined.loc[eligible, "__ts__"].nunique()),
        "join": "exact __ts__ == signal_context_utc", "fill": "none", "unmatched_policy": "excluded", "incomplete_policy": "excluded_no_fill",
    }])
    result = joined.loc[eligible].drop(columns=["signal_context_utc", "__common_join__"])
    if result.empty: raise ValueError(f"{lineage} has no exact complete common-context rows")
    return result, audit


def arm_features(frame: pd.DataFrame, arm: str) -> np.ndarray:
    columns = [SCORE, ALPHA]
    if arm in ("common_state_hurdle", "common_state_transition_hurdle"): columns.extend(STATE)
    if arm in ("common_transition_hurdle", "common_state_transition_hurdle"): columns.extend(TRANSITION)
    values = frame.loc[:, columns].apply(pd.to_numeric, errors="raise").to_numpy(float)
    if not np.isfinite(values).all(): raise ValueError(f"{arm} encountered a missing/nonfinite feature; no fill is permitted")
    return np.column_stack([values, frame.side_name.eq("long").astype(float).to_numpy()])


def hurdle_score(train: pd.DataFrame, test: pd.DataFrame, arm: str) -> tuple[np.ndarray, dict[str, Any]]:
    target = train[NET].gt(0).astype(int).to_numpy()
    if target.min() == target.max():
        probability = np.full(len(test), float(target.mean()))
    else:
        # Same fixed regularization used by the sealed v5 logistic hurdle.
        model = make_pipeline(StandardScaler(), LogisticRegression(C=.25, max_iter=300, class_weight="balanced", random_state=20260730))
        model.fit(arm_features(train, arm), target)
        probability = model.predict_proba(arm_features(test, arm))[:, 1]
    all_pos, all_neg = train.loc[train[NET].gt(0), NET], train.loc[train[NET].le(0), NET]
    payoffs: dict[str, tuple[float, float]] = {}
    for side in test.side_name.drop_duplicates():
        local = train.loc[train.side_name.eq(side), NET]
        payoffs[str(side)] = (float(local.loc[local.gt(0)].mean()) if local.gt(0).any() else float(all_pos.mean()),
                               float(local.loc[local.le(0)].mean()) if local.le(0).any() else float(all_neg.mean()))
    payoff = np.asarray([payoffs[str(side)] for side in test.side_name])
    return probability * payoff[:, 0] + (1 - probability) * payoff[:, 1], {
        "model": "fixed_C_0.25_standardized_class_balanced_side_aware_logistic_cost_clearing_hurdle_plus_train_side_payoff",
        "positive_rate_train": float(target.mean()), "feature_count": int(arm_features(train.iloc[:1], arm).shape[1]),
    }


def causal_map(prior: pd.DataFrame, score: np.ndarray, start: pd.Timestamp) -> tuple[np.ndarray, dict[str, Any]]:
    reference = prior.loc[pd.to_datetime(prior.execution_label_end_utc, utc=True) < start].copy()
    if len(reference) < v5.MIN_MAP or reference.raw_score.nunique() < 2:
        return np.full(len(score), np.nan), {"map_eligible": False, "map_reference_rows": int(len(reference)), "latest_reference_label_end_utc": reference.execution_label_end_utc.max() if len(reference) else None}
    mapper = IsotonicRegression(out_of_bounds="clip").fit(reference.raw_score.to_numpy(float), reference[NET].to_numpy(float))
    return mapper.predict(score), {"map_eligible": True, "map_reference_rows": int(len(reference)), "latest_reference_label_end_utc": reference.execution_label_end_utc.max()}


def score_oof(frame: pd.DataFrame, arm: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = frame.sort_values(["__ts__", "candidate_id"], kind="stable")
    rows: list[pd.DataFrame] = []; prior: list[pd.DataFrame] = []; audit: list[dict[str, Any]] = []
    for block in v5.block_starts(frame):
        test = frame.loc[frame.__ts__.ge(block) & frame.__ts__.lt(block + pd.Timedelta(days=v5.BLOCK_DAYS))].copy()
        train = frame.loc[(frame.__ts__ < block) & (frame.execution_label_end_utc < block)].copy()
        if len(train) < v5.MIN_TRAIN:
            audit.append({"lineage_id": frame.lineage_id.iloc[0], "arm": arm, "outer_block_start_utc": block, "status": "warmup_unscored", "train_rows": len(train), "test_rows": len(test), "causal_train_max_label_end_utc": train.execution_label_end_utc.max() if len(train) else None})
            continue
        raw, model = hurdle_score(train, test, arm)
        mapped, mapping = causal_map(pd.concat(prior, ignore_index=True) if prior else pd.DataFrame(columns=["execution_label_end_utc", "raw_score", NET]), raw, block)
        out = test.loc[:, [*ID, "lineage_id", "evidence_grade", "execution_label_end_utc", GROSS, COST, NET, "execution_exit_reason"]].copy()
        out["arm"] = arm; out["outer_block_start_utc"] = block; out["raw_score"] = raw; out["mapped_ev"] = mapped
        out["map_eligible"] = mapping["map_eligible"]; out["map_reference_rows"] = mapping["map_reference_rows"]; out["evaluation_role"] = ROLE_OOF
        rows.append(out); prior.append(out.loc[:, ["execution_label_end_utc", "raw_score", NET]])
        audit.append({"lineage_id": frame.lineage_id.iloc[0], "arm": arm, "outer_block_start_utc": block, "status": "scored", "train_rows": len(train), "test_rows": len(test), "causal_train_max_label_end_utc": train.execution_label_end_utc.max(), **model, **mapping})
    return (pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(), pd.DataFrame(audit))


def score_forward(historical: pd.DataFrame, current: pd.DataFrame, oof: pd.DataFrame, arm: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw, model = hurdle_score(historical, current, arm)
    prior = oof.loc[:, ["execution_label_end_utc", "raw_score", NET]].copy()
    if len(prior) < v5.MIN_MAP or prior.raw_score.nunique() < 2: raise ValueError(f"{arm} cannot freeze a 2025 OOF map")
    mapper = IsotonicRegression(out_of_bounds="clip").fit(prior.raw_score, prior[NET])
    out = current.loc[:, [*ID, "lineage_id", "evidence_grade", "execution_label_end_utc", GROSS, COST, NET, "execution_exit_reason"]].copy()
    out["arm"] = arm; out["raw_score"] = raw; out["mapped_ev"] = mapper.predict(raw); out["map_eligible"] = True; out["map_reference_rows"] = len(prior)
    out["outer_block_start_utc"] = pd.Timestamp("2026-01-01T00:00:00Z"); out["evaluation_role"] = ROLE_FORWARD
    return out, model


def same_ids(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    return left.loc[:, list(ID)].sort_values(list(ID), kind="stable").reset_index(drop=True).equals(right.loc[:, list(ID)].sort_values(list(ID), kind="stable").reset_index(drop=True))


def controls_subset(path: Path, lineage: str, cohort: pd.DataFrame, role: str) -> pd.DataFrame:
    controls = pd.read_parquet(path)
    controls = controls.loc[controls.arm.isin(("baseline_residual_ev", "hurdle_alpha")) & controls.lineage_id.eq(lineage)].copy()
    controls = controls.merge(cohort.loc[:, [*ID, "execution_exit_reason"]], on=list(ID), how="inner", validate="many_to_one")
    controls["evaluation_role"] = role
    return controls


def identity_audit(table: pd.DataFrame, role: str) -> pd.DataFrame:
    sets = {arm: set(table.loc[table.arm.eq(arm) & table.map_eligible, "candidate_id"]) for arm in ARMS}
    return pd.DataFrame([{"evaluation_role": role, "identity_parity": len(set(map(frozenset, sets.values()))) == 1,
                          **{f"{arm}_eligible_rows": len(ids) for arm, ids in sets.items()}}])


def retain_common_map_eligible(table: pd.DataFrame, role: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the one common arm intersection after each arm-local causal map.

    Warm-up and insufficient-map folds remain visible in the raw ledgers, but
    cannot be silently selected by one arm and not another.  This is a filter,
    never a score fill or a cross-arm map.
    """
    keyed = {arm: table.loc[table.arm.eq(arm) & table.map_eligible, list(ID)].drop_duplicates() for arm in ARMS}
    common_ids = keyed[ARMS[0]]
    for arm in ARMS[1:]:
        common_ids = common_ids.merge(keyed[arm], on=list(ID), how="inner", validate="one_to_one")
    kept = table.merge(common_ids, on=list(ID), how="inner", validate="many_to_one")
    audit = pd.DataFrame([{"evaluation_role": role, "policy": "intersection_of_arm_local_map_eligible_ids", "common_eligible_ids": len(common_ids),
                           **{f"{arm}_preintersection_map_eligible_ids": len(keyed[arm]) for arm in ARMS},
                           "all_arms_exactly_common_after_filter": bool(all(same_ids(kept.loc[kept.arm.eq(ARMS[0])], kept.loc[kept.arm.eq(arm)]) for arm in ARMS[1:]))}])
    if kept.empty or not kept.map_eligible.all() or not audit.loc[0, "all_arms_exactly_common_after_filter"]:
        raise ValueError(f"{role} common map-eligible identity intersection failed")
    return kept, audit


def summary_metrics(scores: pd.DataFrame, role: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    eligible = scores.loc[scores.map_eligible].copy(); eligible["month"] = eligible.__ts__.dt.strftime("%Y-%m"); eligible["week"] = eligible.__ts__.dt.strftime("%G-W%V")
    rows: list[dict[str, Any]] = []; selected: list[pd.DataFrame] = []; recalls: list[dict[str, Any]] = []
    groupers = [("global", None), ("month", "month"), ("week", "week")]
    for (lineage, arm), arm_rows in eligible.groupby(["lineage_id", "arm"], sort=True, observed=True):
        for scope, column in groupers:
            groups = [("all", arm_rows)] if column is None else arm_rows.groupby(column, sort=True, observed=True)
            for period, group in groups:
                for fraction in (.01, .05, .10, .20):
                    for basis in ("raw_score", "mapped_ev"):
                        top, tie = common.fractional_top(group, basis, fraction)
                        pos_total = float(group[NET].gt(0).sum()); pos_selected = float((top[NET].gt(0) * top.selection_weight).sum())
                        row = {"evaluation_role": role, "lineage_id": lineage, "arm": arm, "scope": scope, "period": period, "ranking_basis": basis, "top_fraction": fraction,
                               "candidate_rows": len(group), "selected_mass": float(top.selection_weight.sum()), "raw_ic": common.rank_ic(top, "raw_score"), "mapped_ic": common.rank_ic(top, "mapped_ev"),
                               "mean_gross_bps": common.weighted_mean(top, GROSS)*1e4, "mean_cost_bps": common.weighted_mean(top, COST)*1e4, "mean_net_bps": common.weighted_mean(top, NET)*1e4,
                               "waterfall_reconciliation_bps": (common.weighted_mean(top, GROSS)-common.weighted_mean(top, COST)-common.weighted_mean(top, NET))*1e4, **tie}
                        rows.append(row)
                        recalls.append({"evaluation_role": role, "lineage_id": lineage, "arm": arm, "scope": scope, "period": period, "ranking_basis": basis, "top_fraction": fraction,
                                        "positive_net_precision": pos_selected / float(top.selection_weight.sum()) if len(top) and top.selection_weight.sum() else float("nan"),
                                        "positive_net_recall": pos_selected / pos_total if pos_total else float("nan"), "positive_net_population": pos_total, "positive_net_selected_mass": pos_selected})
                        if basis == "mapped_ev" and fraction == .10:
                            picked = top.copy(); picked["selection_scope"] = scope; picked["selection_period"] = period; selected.append(picked)
        month = [x for x in rows if x["lineage_id"] == lineage and x["arm"] == arm and x["scope"] == "month" and x["ranking_basis"] == "mapped_ev" and x["top_fraction"] == .10]
        if month:
            latest = max(month, key=lambda x: x["period"]); worst = min(month, key=lambda x: x["mean_net_bps"])
            for label, source in (("latest_month", latest), ("worst_month", worst)):
                copy = dict(source); copy["scope"] = label; copy["period"] = source["period"]; rows.append(copy)
    return pd.DataFrame(rows), pd.concat(selected, ignore_index=True), pd.DataFrame(recalls)


def attribution(selected: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for dims in (("side_name",), ("__symbol__",), ("execution_exit_reason",), ("side_name", "execution_exit_reason")):
        for key, group in selected.groupby(["evaluation_role", "lineage_id", "arm", *dims], observed=True, sort=True):
            values = key if isinstance(key, tuple) else (key,); row = dict(zip(["evaluation_role", "lineage_id", "arm", *dims], values))
            row.update({"attribution_dimension": "|".join(dims), "selection_mass": float(group.selection_weight.sum()), "mean_gross_bps": common.weighted_mean(group, GROSS)*1e4, "mean_cost_bps": common.weighted_mean(group, COST)*1e4, "mean_net_bps": common.weighted_mean(group, NET)*1e4, "waterfall_reconciliation_bps": (common.weighted_mean(group, GROSS)-common.weighted_mean(group, COST)-common.weighted_mean(group, NET))*1e4})
            rows.append(row)
    return pd.DataFrame(rows)


def gate_table(forward_metrics: pd.DataFrame, forward_selected: pd.DataFrame, ties: pd.DataFrame, identity: pd.DataFrame, causal: pd.DataFrame) -> pd.DataFrame:
    gates=[{"gate": "strict_oof_label_legality", "pass": bool(causal.loc[causal.status.eq("scored"), "causal_pass"].all()), "detail": "each 2025 train label ends strictly before its outer fold"},
           {"gate": "same_eligible_ids_all_arms", "pass": bool(identity.identity_parity.all()), "detail": "common complete-case cohort and map eligibility are identical"},
           {"gate": "global_pooled_fractional_selection", "pass": True, "detail": "one global top-k after arm-local map; no side quota"},
           {"gate": "tie_at_most_5pct", "pass": bool((ties.cutoff_tie_fraction <= .05).all()), "detail": "each cutoff tie is at most five percent"},
           {"gate": "frozen_2026_no_labels", "pass": True, "detail": "2026 labels are absent from fit and mapping"}]
    fwd = forward_metrics.loc[(forward_metrics.scope.eq("global")) & forward_metrics.ranking_basis.eq("mapped_ev") & forward_metrics.top_fraction.eq(.10)]
    controls = {arm: float(fwd.loc[fwd.arm.eq(arm), "mean_net_bps"].iloc[0]) for arm in ("baseline_residual_ev", "hurdle_alpha")}
    for arm in ARMS[2:]:
        net = float(fwd.loc[fwd.arm.eq(arm), "mean_net_bps"].iloc[0]); side = forward_selected.loc[forward_selected.arm.eq(arm)].groupby("side_name", observed=True).apply(lambda x: common.weighted_mean(x, NET)*1e4)
        gates.extend(({"gate": f"{arm}_exceeds_both_controls_global_top10", "pass": bool(net > controls["baseline_residual_ev"] and net > controls["hurdle_alpha"]), "detail": "mapped global top10 net compared separately with residual and v5 linear controls"},
                      {"gate": f"{arm}_both_sides_positive", "pass": bool(set(side.index) == {"long", "short"} and (side > 0).all()), "detail": "mapped global top10 net positive on long and short"}))
    return pd.DataFrame(gates)


def run(output: Path = OUT) -> dict[str, Any]:
    if output.exists(): raise FileExistsError(output)
    staging = output.with_name(f".{output.name}.{os.getpid()}.partial")
    if staging.exists(): raise FileExistsError(staging)
    staging.mkdir(parents=True)
    try:
        verification = verify_sealed_inputs()
        lineages, _ = v5.load_lineages()
        historical_name = next(name for name in lineages if name.startswith("canonical_marapr")); current_name = next(name for name in lineages if name.startswith("current_mayjul"))
        context = load_common_context()
        historical, coverage25 = exact_timestamp_join(lineages[historical_name], context, lineage=historical_name)
        current, coverage26 = exact_timestamp_join(lineages[current_name], context, lineage=current_name)
        coverage = pd.concat([coverage25, coverage26], ignore_index=True)
        custom_oof=[]; causal=[]; custom_forward=[]; models=[]
        for arm in ARMS[2:]:
            oof, audit = score_oof(historical, arm); custom_oof.append(oof); causal.append(audit)
            fwd, model = score_forward(historical, current, oof, arm); custom_forward.append(fwd); models.append({"arm": arm, "fit_labels": "2025 only", "map_labels": "2025 blocked OOF only", **model})
        controls_oof = controls_subset(V5 / "within_lineage_candidate_scores.parquet", historical_name, historical, ROLE_OOF)
        controls_fwd = controls_subset(V5 / "strict_forward_2026_candidate_scores.parquet", current_name, current, ROLE_FORWARD)
        oof_raw = pd.concat([controls_oof, *custom_oof], ignore_index=True); forward_raw = pd.concat([controls_fwd, *custom_forward], ignore_index=True)
        oof, map_intersection_oof = retain_common_map_eligible(oof_raw, ROLE_OOF)
        forward, map_intersection_fwd = retain_common_map_eligible(forward_raw, ROLE_FORWARD)
        identity = pd.concat([identity_audit(oof, ROLE_OOF), identity_audit(forward, ROLE_FORWARD)], ignore_index=True)
        if not identity.identity_parity.all(): raise ValueError("eligible IDs drifted across arms")
        causal_audit = pd.concat(causal, ignore_index=True)
        causal_audit["causal_pass"] = causal_audit.apply(lambda x: x.status != "scored" or pd.Timestamp(x.causal_train_max_label_end_utc) < pd.Timestamp(x.outer_block_start_utc), axis=1)
        fwd_metrics, fwd_selected, fwd_recall = summary_metrics(forward, ROLE_FORWARD)
        oof_metrics, oof_selected, oof_recall = summary_metrics(oof, ROLE_OOF)
        selected = pd.concat([oof_selected, fwd_selected], ignore_index=True)
        ties = pd.concat([oof_metrics, fwd_metrics], ignore_index=True).loc[:, ["evaluation_role", "lineage_id", "arm", "scope", "period", "ranking_basis", "top_fraction", "candidate_rows", "cutoff_tie_rows", "cutoff_tie_fraction", "selection_rule"]]
        attr = attribution(selected)
        gates = gate_table(fwd_metrics, fwd_selected.loc[fwd_selected.selection_scope.eq("global")], ties, identity, causal_audit)
        feature_contract = pd.DataFrame([{"family": "state", "feature_count": len(STATE), "columns": "|".join(STATE), "admissible": True, "definition": "semantic state mean and long-short gap only"},
                                         {"family": "lag_delta_transition", "feature_count": len(TRANSITION), "columns": "|".join(TRANSITION), "admissible": True, "definition": "strict past 1/3/12h deltas only"},
                                         {"family": "forbidden", "feature_count": 0, "columns": "", "admissible": False, "definition": "outcome/source/calendar/provenance/state-ID features are excluded"}])
        availability = pd.DataFrame([{"arm":"baseline_residual_ev","status":"sealed_v5_control_on_common_cohort","fit_labels":"sealed v5 2025","map_labels":"sealed v5 blocked OOF 2025"},{"arm":"hurdle_alpha","status":"sealed_v5_linear_control_on_common_cohort","fit_labels":"sealed v5 2025","map_labels":"sealed v5 blocked OOF 2025"}, *models])
        outputs = {"oof_candidate_scores.parquet": oof, "forward_2026_candidate_scores.parquet": forward, "common_cohort_coverage.csv": coverage, "feature_family_contract.csv": feature_contract, "oof_fold_causal_audit.csv": causal_audit, "arm_local_mapping_identity_intersection_audit.csv": pd.concat([map_intersection_oof, map_intersection_fwd], ignore_index=True), "identity_parity.csv": identity, "oof_global_month_week_latest_worst_metrics.csv": oof_metrics, "forward_2026_global_month_week_latest_worst_metrics.csv": fwd_metrics, "positive_net_recall_precision.csv": pd.concat([oof_recall, fwd_recall], ignore_index=True), "selected_fractional_top10.parquet": selected, "side_asset_exit_attribution_reconciliation.csv": attr, "tie_audit.csv": ties, "arm_availability.csv": availability, "gates.csv": gates}
        for name, table in outputs.items():
            atomic_parquet(table, staging / name) if name.endswith(".parquet") else table.to_csv(staging / name, index=False)
        inputs = {str(path): sha(path) for path in (v5.EXACT_2025, v5.EXACT_2026, V5 / "manifest.json", GEOMETRY / "manifest.json", GEOMETRY / "semantic_mapping.json", CONTEXT)}
        report = {"schema": SCHEMA, "status": "SEALED_DIAGNOSTIC_NON_PROMOTION", "promotion_eligible": False, "verification": verification, "input_sha256": inputs,
                  "fixed_contract": {"controls": "sealed v5 residual and linear alpha hurdle candidate scores, subset only to the common complete-case cohort", "features": "90 semantic-identical decision-time fields: 36 state, 54 strict lag/delta; no outcome/source/calendar/provenance/state-ID columns", "join": "exact signal timestamp equality; no fill/asof/resampling", "model": "fixed C=0.25 class-balanced side-aware logistic with fold-local StandardScaler and training-only side payoff", "oof": "14-day blocked 2025 folds; every train/map label end strictly before fold start; arm-local map", "forward": "2025 full fit plus 2025 blocked-OOF map frozen before 2026 score; 2026 labels unused", "selection": "one pooled global top 1/5/10/20 after mapping, exact fractional cutoff ties, no side quota", "scope": "research-only; no portfolio replay"}, "gates": gates.to_dict("records"), "outputs_sha256": {path.name: sha(path) for path in staging.iterdir() if path.is_file()}}
        write_json(staging / "report.json", report)
        manifest = {"schema": SCHEMA + "_manifest", "status": "SEALED_DIAGNOSTIC_NON_PROMOTION", "runner_sha256": sha(Path(__file__)), "input_sha256": inputs, "common_geometry_manifest_sha256": verification["common_geometry_manifest_sha256"], "outputs_sha256": {path.name: sha(path) for path in staging.iterdir() if path.is_file()}}
        write_json(staging / "manifest.json", manifest); (staging / "manifest.sha256").write_text(f"{sha(staging / 'manifest.json')}  manifest.json\n")
        staging.replace(output)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args(); print(json.dumps(safe(run(args.output)), indent=2))


if __name__ == "__main__": main()
