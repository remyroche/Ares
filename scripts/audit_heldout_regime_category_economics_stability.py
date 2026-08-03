#!/usr/bin/env python3
"""Audit held-out global-book economics by state and transition taxonomy.

Selection is fixed *before* any taxonomy is joined: every source-month gets one
pooled global top decile across sides and timestamps.  States are decision-time
context. Transition phase is explicitly ex-post attribution only, never a
selection input, model feature, or policy gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "data_perp/artifacts"
OUT = ART / "heldout_regime_category_economics_stability_20260730_v1"
SUPPORT_SHRINKAGE_DAYS = 20.0
MIN_HELDOUT_DAYS = 5
MIN_ERAS = 3
MEANINGFUL_BPS = 5.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def safe(value: Any) -> Any:
    if isinstance(value, (Path, pd.Timestamp)):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def select_global_top10(frame: pd.DataFrame, score: str) -> pd.Series:
    """One deterministic global book, with no category/side/time allocation."""
    valid = frame.loc[pd.to_numeric(frame[score], errors="coerce").notna()]
    selected = pd.Series(False, index=frame.index)
    count = max(1, math.ceil(len(valid) * 0.10))
    chosen = valid.sort_values([score, "candidate_id"], ascending=[False, True], kind="stable").index[:count]
    selected.loc[chosen] = True
    return selected


def select_by_lineage_month(frame: pd.DataFrame) -> pd.Series:
    selected = pd.Series(False, index=frame.index)
    for _, group in frame.groupby(["lineage", "month"], sort=True):
        selected.loc[group.index] = select_global_top10(group, "score")
    return selected


def _source(path: Path, *, lineage: str, cohort: str, score: str, columns: dict[str, str], end: str | None = None) -> pd.DataFrame:
    needed = ["candidate_id", "__ts__", "side_name", "execution_net_ev_12h", score]
    frame = pd.read_parquet(path, columns=needed).rename(columns={score: "score", "execution_net_ev_12h": "net"})
    frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True, errors="raise")
    if end is not None:
        frame = frame.loc[frame["__ts__"] < pd.Timestamp(end, tz="UTC")].copy()
    frame["side_name"] = frame["side_name"].astype(str).str.lower()
    frame["lineage"] = lineage
    frame["economics_cohort"] = cohort
    frame["era"] = frame["__ts__"].dt.year.astype(str)
    frame["month"] = frame["__ts__"].dt.strftime("%Y-%m")
    return frame


def load_candidates(artifacts: Path) -> tuple[pd.DataFrame, dict[str, Path]]:
    historical = artifacts / "reconstructed_base_residual_stack_2022_2024_20260730_v3/oof_scores.parquet"
    canonical = artifacts / "historical_causal_score_economics_mapping_20260729_v1/canonical_residual__score_residual_expected_ev/causal_mapped_candidates.parquet"
    current = artifacts / "current_exact_policy_global_book_mapping_source_20260730_v3/causal_mapped_candidates.parquet"
    reconstructed = _source(historical, lineage="reconstructed_2022_2024", cohort="frozen_pf_spread_counterfactual", score="score_residual_expected_ev", columns={})
    # The inverse-PI H1 population is explicitly incompatible with the frozen
    # population even though it shares a counterfactual spread construction.
    reconstructed.loc[reconstructed["__ts__"] < pd.Timestamp("2022-08-01", tz="UTC"), "economics_cohort"] = "inverse_pi_separate_population"
    canonical_rows = _source(canonical, lineage="canonical_2025", cohort="exact_usd_linear_policy", score="score_residual_expected_ev", columns={})
    current_rows = _source(current, lineage="current_2026", cohort="exact_usd_linear_policy", score="catboost__residual__without_hpo__all_features", columns={}, end="2026-07-11")
    result = pd.concat([reconstructed, canonical_rows, current_rows], ignore_index=True)
    if result.duplicated(["candidate_id", "__ts__", "side_name"], keep=False).any():
        raise ValueError("candidate identities duplicate across sources")
    return result, {"reconstructed": historical, "canonical": canonical, "current": current}


def load_context(artifacts: Path) -> tuple[pd.DataFrame, dict[str, Path]]:
    state_path = artifacts / "regime_episode_ledger_2022_2026_20260730_v1/hourly_state_calendar.parquet"
    phase_path = artifacts / "transition_pattern_catalogue_20260730_v6/adaptive_phase_labels.parquet"
    state = pd.read_parquet(state_path, columns=["source_utc", "target__pooled_state"])
    phase = pd.read_parquet(phase_path, columns=["source_utc", "target__pattern_phase", "target__pattern_phase_available_utc"])
    state["source_utc"] = pd.to_datetime(state["source_utc"], utc=True, errors="raise")
    phase["source_utc"] = pd.to_datetime(phase["source_utc"], utc=True, errors="raise")
    if state.source_utc.duplicated().any() or phase.source_utc.duplicated().any():
        raise ValueError("state/phase timestamp must be unique")
    context = state.merge(phase, on="source_utc", how="inner", validate="one_to_one")
    context = context.rename(columns={"target__pooled_state": "regime_state", "target__pattern_phase": "transition_phase"})
    return context, {"state": state_path, "phase": phase_path}


def _estimate(days: pd.Series, prior_mean: float, prior_se: float) -> dict[str, float]:
    values = pd.to_numeric(days, errors="coerce").dropna().to_numpy(float)
    n = len(values)
    raw = float(np.mean(values)) if n else np.nan
    se = float(np.std(values, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    weight = float(n / (n + SUPPORT_SHRINKAGE_DAYS))
    mean = weight * raw + (1.0 - weight) * prior_mean if n else np.nan
    uncertainty = math.sqrt((weight * se) ** 2 + ((1.0 - weight) * prior_se) ** 2) if np.isfinite(se) and np.isfinite(prior_se) else np.nan
    return {"n_days": n, "raw_mean_net_bps": raw, "raw_se_bps": se, "shrinkage_weight": weight, "shrunken_mean_net_bps": mean, "ci95_low_bps": mean - 1.96 * uncertainty if np.isfinite(uncertainty) else np.nan, "ci95_high_bps": mean + 1.96 * uncertainty if np.isfinite(uncertainty) else np.nan}


def make_taxonomies(selected: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for taxonomy, category in (
        ("regime_state_at_decision", selected["regime_state"].astype("Int64").astype(str)),
        ("transition_phase_ex_post", selected["transition_phase"].astype(str)),
        ("state_x_transition_phase_ex_post", selected["regime_state"].astype("Int64").astype(str) + "|" + selected["transition_phase"].astype(str)),
    ):
        part = selected.loc[:, ["candidate_id", "__ts__", "lineage", "economics_cohort", "era", "side_name", "net"]].copy()
        part["taxonomy"] = taxonomy
        part["category"] = category.to_numpy()
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def daily_category_book(taxonomy_rows: pd.DataFrame) -> pd.DataFrame:
    copy = taxonomy_rows.copy(); copy["day"] = copy["__ts__"].dt.floor("D")
    return copy.groupby(["taxonomy", "category", "economics_cohort", "era", "side_name", "day"], as_index=False).agg(candidate_rows=("net", "size"), net_bps=("net", lambda x: float(np.mean(x) * 1e4)))


def cell_summary(daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in daily.groupby(["taxonomy", "category", "economics_cohort", "era", "side_name"], sort=True):
        prior_days = daily.loc[daily.economics_cohort.eq(keys[2])].groupby("day", sort=False).net_bps.mean()
        prior = _estimate(prior_days, float(prior_days.mean()), float(prior_days.std(ddof=1) / np.sqrt(len(prior_days))))
        estimate = _estimate(group.net_bps, prior["raw_mean_net_bps"], prior["raw_se_bps"])
        rows.append({"taxonomy": keys[0], "category": keys[1], "economics_cohort": keys[2], "era": keys[3], "side_name": keys[4], "candidate_rows": int(group.candidate_rows.sum()), **estimate})
    return pd.DataFrame(rows)


def leave_era_out(daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["taxonomy", "category", "economics_cohort", "era", "side_name"]
    for values, held in daily.groupby(keys, sort=True):
        taxonomy, category, cohort, era, side = values
        train = daily.loc[(daily.taxonomy.eq(taxonomy)) & (daily.category.eq(category)) & (daily.economics_cohort.eq(cohort)) & ~daily.era.eq(era)]
        prior_train = daily.loc[(daily.economics_cohort.eq(cohort)) & ~daily.era.eq(era)].groupby("day", sort=False).net_bps.mean()
        prior_mean = float(prior_train.mean()) if len(prior_train) else np.nan
        prior_se = float(prior_train.std(ddof=1) / np.sqrt(len(prior_train))) if len(prior_train) > 1 else np.nan
        train_estimate = _estimate(train.net_bps, prior_mean, prior_se)
        held_estimate = _estimate(held.net_bps, prior_mean, prior_se)
        good = bool(train_estimate["ci95_low_bps"] >= MEANINGFUL_BPS and held_estimate["ci95_low_bps"] > 0.0)
        poor = bool(train_estimate["ci95_high_bps"] <= -MEANINGFUL_BPS and held_estimate["ci95_high_bps"] < 0.0)
        rows.append({"taxonomy": taxonomy, "category": category, "economics_cohort": cohort, "heldout_era": era, "side_name": side, "heldout_candidate_rows": int(held.candidate_rows.sum()), "train_candidate_rows": int(train.candidate_rows.sum()), **{f"train_{key}": value for key, value in train_estimate.items()}, **{f"heldout_{key}": value for key, value in held_estimate.items()}, "good_sign_confirmed": good, "poor_sign_confirmed": poor, "qualified_cell_support": bool(train_estimate["n_days"] >= SUPPORT_SHRINKAGE_DAYS and held_estimate["n_days"] >= MIN_HELDOUT_DAYS)})
    return pd.DataFrame(rows)


def qualify(loo: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in loo.groupby(["taxonomy", "category", "economics_cohort"], sort=True):
        sides, eras = group.side_name.nunique(), group.heldout_era.nunique()
        support = group.qualified_cell_support.all()
        good = bool(sides == 2 and eras >= MIN_ERAS and support and group.good_sign_confirmed.all())
        poor = bool(sides == 2 and eras >= MIN_ERAS and support and group.poor_sign_confirmed.all())
        reasons = []
        if sides != 2: reasons.append("not_observed_on_both_sides")
        if eras < MIN_ERAS: reasons.append(f"fewer_than_{MIN_ERAS}_eras_in_comparable_cohort")
        if not support: reasons.append("insufficient_leave_era_out_day_support")
        if not good and not poor: reasons.append("leave_era_out_uncertainty_or_sign_not_consistent")
        rows.append({"taxonomy": keys[0], "category": keys[1], "economics_cohort": keys[2], "observed_sides": sides, "observed_eras": eras, "stable_good_net_ev": good, "stable_poor_net_ev": poor, "promotion_eligible": False, "qualification": "research_only_stable" if good or poor else "not_stable", "reasons": "|".join(reasons) if reasons else "stable_only_as_research_diagnostic"})
    return pd.DataFrame(rows)


def run(*, artifacts: Path = ART, output: Path = OUT) -> dict[str, Any]:
    if output.exists(): raise FileExistsError(output)
    candidates, source_paths = load_candidates(artifacts)
    candidates["selected_global_top10"] = select_by_lineage_month(candidates)
    selected = candidates.loc[candidates.selected_global_top10].copy()
    context, context_paths = load_context(artifacts)
    selected = selected.merge(context, left_on="__ts__", right_on="source_utc", how="left", validate="many_to_one")
    coverage = selected.assign(context_covered=selected.regime_state.notna() & selected.transition_phase.notna()).groupby(["lineage", "economics_cohort", "era"], as_index=False).agg(selected_rows=("candidate_id", "size"), context_covered_rows=("context_covered", "sum"))
    coverage["context_coverage_fraction"] = coverage.context_covered_rows / coverage.selected_rows
    attributed = selected.loc[selected.regime_state.notna() & selected.transition_phase.notna()].copy()
    taxonomies = make_taxonomies(attributed); daily = daily_category_book(taxonomies); cells = cell_summary(daily); loo = leave_era_out(daily); qualification = qualify(loo)
    stage = output.parent / f".{output.name}.{uuid.uuid4().hex}.stage"; stage.mkdir(parents=True, exist_ok=False)
    try:
        selected.to_parquet(stage / "selected_global_top10_context.parquet", index=False, compression="zstd")
        coverage.to_csv(stage / "context_coverage.csv", index=False)
        daily.to_parquet(stage / "category_daily_economics.parquet", index=False, compression="zstd")
        cells.to_csv(stage / "category_hierarchical_cell_summary.csv", index=False)
        loo.to_csv(stage / "category_leave_era_out.csv", index=False)
        qualification.to_csv(stage / "category_stability_qualification.csv", index=False)
        stable = qualification.loc[qualification.stable_good_net_ev | qualification.stable_poor_net_ev]
        summary = {"schema": "heldout_regime_category_economics_stability_v1", "research_only": True, "promotion_eligible": False, "selection_contract": "one pooled global top10 per lineage x UTC month before any category join; no side/timestamp/category quota", "taxonomy_contract": {"regime_state_at_decision": "decision-time attribution only", "transition_phase_ex_post": "ex-post attribution only; prohibited as policy/model input or gate", "state_x_transition_phase_ex_post": "diagnostic joint attribution only"}, "economics_contract": "cohorts are never pooled across incompatible candidate/economic lineages", "shrinkage_contract": {"unit": "UTC-day category mean net EV", "support_shrinkage_days": SUPPORT_SHRINKAGE_DAYS, "uncertainty": "normal 95% interval with prior uncertainty"}, "leave_era_out_contract": {"minimum_eras": MIN_ERAS, "minimum_heldout_days": MIN_HELDOUT_DAYS, "both_sides_required": True, "good_threshold_bps": MEANINGFUL_BPS, "poor_threshold_bps": -MEANINGFUL_BPS}, "counts": {"candidate_rows": len(candidates), "selected_rows": len(selected), "context_attributed_rows": len(attributed), "stable_categories": len(stable)}, "stable_categories": stable.to_dict("records"), "inputs_sha256": {name: sha256(path) for name, path in {**source_paths, **context_paths}.items()}, "outputs_sha256": {path.name: sha256(path) for path in stage.iterdir() if path.is_file()}}
        (stage / "audit_summary.json").write_text(json.dumps(safe(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (stage / "manifest.json").write_text(json.dumps(safe(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (stage / "manifest.sha256").write_text(f"{sha256(stage / 'manifest.json')}  manifest.json\n", encoding="utf-8")
        os.replace(stage, output)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True); raise
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--artifacts", type=Path, default=ART); parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args(argv); print(json.dumps(safe(run(artifacts=args.artifacts, output=args.output)), sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
