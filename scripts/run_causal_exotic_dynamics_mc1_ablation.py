#!/usr/bin/env python3
"""Strict temporal MC1 ablation for causal transition/dynamics feature families.

This is an offline research runner.  It preserves the paired BCF/current-v5
score families, source-aligned parent-policy outcomes, dual +50-bps admission
and BCF-EV global portfolio auction.  2025 is chronological discovery: each
month uses only the immediately preceding three calendar months with resolved
labels.  The recurring 2025 field contract is then frozen; 2026 is a pure
confirmation period and cannot alter fields, target, HGB geometry, threshold,
or portfolio policy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.causal_exotic_dynamics import FEATURE_COLUMNS
from extreme_price_movements.portfolio_policy_replay import replay_candidates
from scripts.ablate_strict_r3_bcf_current_v5_agreement_blend import POLICY_COLUMNS, _to_candidates
from scripts.report_strict_r3_mc1_d2_controlled_portfolio import CAUSAL_AUCTION_CURVE, _metrics, _params
from scripts.run_strict_r3_mc1_d2_controlled_ablation import CORE, _causal_shifts, _day_balanced, _fit_hgb


BCF = ROOT / "data_perp/artifacts/strict_r3_score_family_matched_mc1_canonical_policy_20260817_v7_bcf/predictions_bcf_mc1_d2.parquet"
CURRENT = ROOT / "data_perp/artifacts/strict_r3_score_family_matched_mc1_canonical_policy_20260817_v7_current/predictions_current_v5_mc1_d2.parquet"
DYNAMIC = ROOT / "data_perp/artifacts/causal_exotic_dynamics_2025train_2026confirm_20260831_v3_expanded"
ASSESSMENT = ROOT / "data_perp/artifacts/causal_exotic_dynamics_assessment_2025_20260831_v2_expanded"
OUT = ROOT / "data_perp/artifacts/causal_exotic_dynamics_mc1_ablation_2025oof_2026confirm_20260831_v1"

ADMISSION_BPS = 50.0
TRAIN_MONTHS = 3
START = pd.Timestamp("2025-07-01T00:00:00Z")
END = pd.Timestamp("2026-08-01T00:00:00Z")
FAMILIES = ("CP", "SP", "WV", "EN", "DS")
REPRESENTATIONS = ("raw", "head", "raw_head")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _utc(values: object) -> pd.Series:
    return pd.to_datetime(values, utc=True, errors="raise")


def _family(field: str) -> str:
    return field.split("_", 1)[0].upper()


def _read_family(path: Path, prefix: str) -> pd.DataFrame:
    cols = [
        "candidate_id", "__decision_ts__", "__symbol__", *CORE, "score_band",
        "mc1_expected_bps", *POLICY_COLUMNS,
    ]
    frame = pd.read_parquet(path, columns=cols)
    frame["candidate_id"] = frame.candidate_id.astype(str)
    frame["__decision_ts__"] = _utc(frame["__decision_ts__"])
    frame["policy_label_available_ts"] = _utc(frame["policy_label_available_ts"])
    rename = {field: f"{prefix}_{field}" for field in (*CORE, "score_band", "mc1_expected_bps")}
    return frame.rename(columns=rename)


def _assert_policy_equal(left: pd.DataFrame, right: pd.DataFrame) -> None:
    for field in ("__decision_ts__", *POLICY_COLUMNS):
        a, b = left[field], right[field]
        if pd.api.types.is_numeric_dtype(a):
            same = np.isclose(pd.to_numeric(a, errors="coerce"), pd.to_numeric(b, errors="coerce"), equal_nan=True).all()
        else:
            same = a.fillna("__null__").astype(str).equals(b.fillna("__null__").astype(str))
        if not same:
            raise AssertionError(f"paired BCF/current policy contract differs: {field}")


def _load_panel(dynamic: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    bcf, current = _read_family(BCF, "bcf"), _read_family(CURRENT, "current")
    # The two frozen score ledgers retain their family-specific unrouted rows.
    # Their shared candidate IDs are the contractual paired universe; the
    # policy labels on that intersection must be identical.  Do not compare
    # incidental source row order, or admit a family-only row into a paired
    # mapper.  The target-free dynamics materialisation is built on precisely
    # this common universe.
    common_ids = pd.Index(bcf.candidate_id).intersection(pd.Index(current.candidate_id), sort=False)
    if common_ids.empty:
        raise AssertionError("BCF/current score ledgers have no common candidate universe")
    bcf = bcf.loc[bcf.candidate_id.isin(common_ids)].sort_values("candidate_id").reset_index(drop=True)
    current = current.loc[current.candidate_id.isin(common_ids)].sort_values("candidate_id").reset_index(drop=True)
    _assert_policy_equal(bcf, current)
    score = bcf.loc[:, ["candidate_id", "__decision_ts__", "__symbol__", *[f"bcf_{x}" for x in (*CORE, "score_band", "mc1_expected_bps")], *POLICY_COLUMNS]].merge(
        current.loc[:, ["candidate_id", *[f"current_{x}" for x in (*CORE, "score_band", "mc1_expected_bps")]]],
        on="candidate_id", validate="one_to_one",
    )
    score = score.loc[score.__decision_ts__.ge(pd.Timestamp("2025-04-01", tz="UTC")) & score.__decision_ts__.lt(END)].copy()
    table = ds.dataset(str(dynamic / "feature_parts"), format="parquet", partitioning="hive")
    states = table.to_table(columns=["candidate_id", "dynamic_source_status", *FEATURE_COLUMNS]).to_pandas()
    states.candidate_id = states.candidate_id.astype(str)
    if states.candidate_id.duplicated().any():
        raise AssertionError("dynamic feature partitions duplicate candidate identity")
    full = score.merge(states, on="candidate_id", how="inner", validate="one_to_one")
    if len(full) != len(score):
        raise AssertionError("dynamic state matrix does not cover the exact paired score route")
    for family in FAMILIES:
        family_fields = [field for field in FEATURE_COLUMNS if _family(field) == family]
        full[f"{family.lower()}_state_available"] = full.loc[:, family_fields].notna().any(axis=1).astype("int8")
    full["policy_path_valid"] = full.policy_path_valid.fillna(False).astype(bool)
    full["valid_label"] = full.policy_path_valid & np.isfinite(pd.to_numeric(full.policy_net_bps, errors="coerce"))
    full["m0_expected_bps"] = (full.bcf_mc1_expected_bps + full.current_mc1_expected_bps) / 2.0
    full["m0_residual_bps"] = pd.to_numeric(full.policy_net_bps, errors="coerce") - full.m0_expected_bps
    target_free = full.drop(columns=list(POLICY_COLUMNS) + ["valid_label", "m0_residual_bps"]).copy()
    return full, target_free


def _contracts(assessment: Path) -> tuple[dict[str, tuple[str, ...]], dict[tuple[str, str], tuple[str, ...]]]:
    frozen = pd.read_parquet(assessment / "frozen_2025_family_feature_contract.parquet")
    frozen_fields = {
        family: tuple(group.loc[group.stable_selected, "feature_name"].astype(str))
        for family, group in frozen.groupby("family", sort=True)
    }
    trace = pd.read_parquet(assessment / "family_probe_trace_2025.parquet")
    fold_fields: dict[tuple[str, str], tuple[str, ...]] = {}
    for row in trace.loc[trace.status.eq("scored")].itertuples(index=False):
        fold_fields[(str(row.family), str(row.held_month))] = tuple(json.loads(row.selected_features))
    if set(FAMILIES).difference(frozen_fields):
        raise AssertionError("frozen 2025 contract is missing a feature family")
    return frozen_fields, fold_fields


def _fields_for(family: str, held: pd.Timestamp, frozen: dict[str, tuple[str, ...]], folds: dict[tuple[str, str], tuple[str, ...]]) -> tuple[str, ...]:
    if held.year == 2025:
        fields = folds.get((family, f"{held:%Y-%m}"), ())
        if fields:
            return fields
    return frozen[family]


def _month_starts(start: pd.Timestamp, end: pd.Timestamp) -> Iterable[pd.Timestamp]:
    yield from pd.date_range(start, end, freq="MS", tz="UTC", inclusive="left")


def _prior_train(full: pd.DataFrame, held: pd.Timestamp) -> pd.DataFrame:
    start = held - pd.DateOffset(months=TRAIN_MONTHS)
    return full.loc[
        full.__decision_ts__.ge(start) & full.__decision_ts__.lt(held)
        & full.valid_label & full.policy_label_available_ts.lt(held)
    ].copy()


def _balanced(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Apply the fixed MC1 daily substrate using that score family's rank."""
    work = frame.copy()
    work["day"] = work.__decision_ts__.dt.floor("1d")
    work["final_score"] = pd.to_numeric(work[f"{prefix}_final_score"], errors="raise")
    work["score_band"] = pd.to_numeric(work[f"{prefix}_score_band"], errors="raise")
    return _day_balanced(work)


def _specialist_oof(full: pd.DataFrame, family: str, frozen: dict[str, tuple[str, ...]], folds: dict[tuple[str, str], tuple[str, ...]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    pieces, audit = [], []
    for held in _month_starts(START, END):
        fields = _fields_for(family, held, frozen, folds)
        train = _prior_train(full, held)
        test = full.loc[full.__decision_ts__.ge(held) & full.__decision_ts__.lt(held + pd.offsets.MonthBegin(1))].copy()
        if len(train) < 1_000 or test.empty or not fields:
            audit.append({"family": family, "held_month": f"{held:%Y-%m}", "status": "insufficient_prior_resolved_support", "train_rows": len(train), "fields": json.dumps(fields)})
            continue
        working = train.copy()
        working["policy_net_bps"] = working.m0_residual_bps
        model, medians, _, clip = _fit_hgb(_balanced(working, "bcf"), list(fields))
        values = model.predict(test.loc[:, list(fields)].apply(pd.to_numeric, errors="coerce").fillna(medians))
        pieces.append(pd.DataFrame({"candidate_id": test.candidate_id.astype(str), f"{family.lower()}_specialist_residual_bps": np.clip(values, *clip), "specialist_fold": held}))
        audit.append({"family": family, "held_month": f"{held:%Y-%m}", "status": "scored", "train_rows": len(train), "fields": json.dumps(fields), "clip_low_bps": float(clip[0]), "clip_high_bps": float(clip[1])})
    return (pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()), pd.DataFrame(audit)


def _map_one_family(
    full: pd.DataFrame, target_free: pd.DataFrame, *, family: str, representation: str,
    frozen: dict[str, tuple[str, ...]], folds: dict[tuple[str, str], tuple[str, ...]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Produce separate BCF/current monthly maps with no held-label use."""
    pieces, audit = [], []
    head = f"{family.lower()}_specialist_residual_bps"
    for held in _month_starts(START, END):
        raw = _fields_for(family, held, frozen, folds)
        available = f"{family.lower()}_state_available"
        extras = [*raw, available] if representation == "raw" else [head, available] if representation == "head" else [*raw, head, available]
        train = _prior_train(full, held)
        test = target_free.loc[target_free.__decision_ts__.ge(held) & target_free.__decision_ts__.lt(held + pd.offsets.MonthBegin(1))].copy()
        # Head representations require strictly prequential head values for
        # their train rows.  We do not fill a missing OOF head from its own
        # training target; early months simply remain unavailable for that arm.
        if head in extras:
            train = train.loc[train[head].notna()].copy()
        if len(train) < 1_000 or test.empty:
            audit.append({"family": family, "representation": representation, "held_month": f"{held:%Y-%m}", "status": "insufficient_prior_resolved_support", "train_rows": len(train), "extras": json.dumps(extras)})
            continue
        result = test.loc[:, ["candidate_id", "__decision_ts__", "__symbol__"]].copy()
        for prefix in ("bcf", "current"):
            core = [f"{prefix}_{field}" for field in CORE]
            fields = [*core, *extras]
            # ``final_score`` is part of CORE.  Preserve a single copy so
            # pandas returns a Series (rather than a duplicate-column frame)
            # when the daily MC1 substrate reads it.
            working_columns = list(dict.fromkeys([
                *fields,
                "policy_net_bps",
                f"{prefix}_final_score",
                f"{prefix}_score_band",
                "__decision_ts__",
                "policy_label_available_ts",
                "policy_path_valid",
                "candidate_id",
            ]))
            working = train.loc[:, working_columns].copy()
            substrate = _balanced(working, prefix)
            model, medians, curve, clip = _fit_hgb(substrate, fields)
            values = model.predict(test.loc[:, fields].apply(pd.to_numeric, errors="coerce").fillna(medians))
            shift_frame = train.loc[:, ["__decision_ts__", "policy_label_available_ts", "policy_path_valid", "policy_net_bps", f"{prefix}_score_band"]].rename(columns={f"{prefix}_score_band": "score_band"})
            buckets = test.__decision_ts__.dt.floor("1d")
            shifts = _causal_shifts(shift_frame, curve, pd.DatetimeIndex(buckets.unique()), "1d")
            result[f"{prefix}_mapped_expected_bps"] = np.clip(values, *clip) + buckets.map(shifts).fillna(0.0).to_numpy(float)
            result[f"{prefix}_static_expected_bps"] = np.clip(values, *clip)
            result[f"{prefix}_recent_shift_bps"] = buckets.map(shifts).fillna(0.0).to_numpy(float)
        result["family"] = family
        result["representation"] = representation
        result["fold_start"] = held
        pieces.append(result)
        audit.append({"family": family, "representation": representation, "held_month": f"{held:%Y-%m}", "status": "scored", "train_rows": len(train), "extras": json.dumps(extras)})
    return (pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()), pd.DataFrame(audit)


def _map_combo(
    full: pd.DataFrame, target_free: pd.DataFrame, *, arm: str,
    specification: tuple[tuple[str, str], ...],
    frozen: dict[str, tuple[str, ...]], folds: dict[tuple[str, str], tuple[str, ...]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Strict-OOS joint mapper for a predeclared set of family representations.

    ``specification`` is frozen before this runner is called and contains only
    representations already evaluated in 2025 strict OOS.  Each specialist is
    itself prequential; missing head rows are never backfilled from their own
    label.  This is intentionally a small addition funnel rather than a broad
    combination search.
    """
    pieces, audit = [], []
    for held in _month_starts(START, END):
        extras: list[str] = []
        required_heads: list[str] = []
        for family, representation in specification:
            if representation not in REPRESENTATIONS:
                raise ValueError(f"{arm}: unsupported representation {representation}")
            raw = list(_fields_for(family, held, frozen, folds))
            head = f"{family.lower()}_specialist_residual_bps"
            if representation in ("raw", "raw_head"):
                extras.extend(raw)
            if representation in ("head", "raw_head"):
                extras.append(head)
                required_heads.append(head)
            extras.append(f"{family.lower()}_state_available")
        extras = list(dict.fromkeys(extras))
        required_heads = list(dict.fromkeys(required_heads))
        train = _prior_train(full, held)
        test = target_free.loc[
            target_free.__decision_ts__.ge(held)
            & target_free.__decision_ts__.lt(held + pd.offsets.MonthBegin(1))
        ].copy()
        if required_heads:
            train = train.loc[train.loc[:, required_heads].notna().all(axis=1)].copy()
        if len(train) < 1_000 or test.empty:
            audit.append({
                "arm": arm, "held_month": f"{held:%Y-%m}",
                "status": "insufficient_prior_resolved_support", "train_rows": len(train),
                "specification": json.dumps(specification), "extras": json.dumps(extras),
            })
            continue
        result = test.loc[:, ["candidate_id", "__decision_ts__", "__symbol__"]].copy()
        for prefix in ("bcf", "current"):
            core = [f"{prefix}_{field}" for field in CORE]
            fields = [*core, *extras]
            working_columns = list(dict.fromkeys([
                *fields,
                "policy_net_bps",
                f"{prefix}_final_score",
                f"{prefix}_score_band",
                "__decision_ts__",
                "policy_label_available_ts",
                "policy_path_valid",
                "candidate_id",
            ]))
            working = train.loc[:, working_columns].copy()
            model, medians, curve, clip = _fit_hgb(_balanced(working, prefix), fields)
            values = model.predict(test.loc[:, fields].apply(pd.to_numeric, errors="coerce").fillna(medians))
            shift_frame = train.loc[:, [
                "__decision_ts__", "policy_label_available_ts", "policy_path_valid",
                "policy_net_bps", f"{prefix}_score_band",
            ]].rename(columns={f"{prefix}_score_band": "score_band"})
            buckets = test.__decision_ts__.dt.floor("1d")
            shifts = _causal_shifts(shift_frame, curve, pd.DatetimeIndex(buckets.unique()), "1d")
            result[f"{prefix}_mapped_expected_bps"] = np.clip(values, *clip) + buckets.map(shifts).fillna(0.0).to_numpy(float)
            result[f"{prefix}_static_expected_bps"] = np.clip(values, *clip)
            result[f"{prefix}_recent_shift_bps"] = buckets.map(shifts).fillna(0.0).to_numpy(float)
        result["family"] = arm
        result["representation"] = "combo"
        result["fold_start"] = held
        pieces.append(result)
        audit.append({
            "arm": arm, "held_month": f"{held:%Y-%m}", "status": "scored",
            "train_rows": len(train), "specification": json.dumps(specification),
            "extras": json.dumps(extras),
        })
    return (pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()), pd.DataFrame(audit)


def _risk_metrics(equity: pd.DataFrame, decisions: pd.DataFrame) -> dict[str, float]:
    if equity.empty or "wallet" not in equity:
        return {"ulcer_index": np.nan, "daily_cvar5": np.nan, "daily_sortino": np.nan, "time_underwater_fraction": np.nan}
    work = equity.copy()
    work["timestamp"] = _utc(work.timestamp)
    wallet = pd.to_numeric(work.set_index("timestamp").sort_index()["wallet"], errors="coerce").dropna().resample("1h").last().ffill()
    drawdown = wallet / wallet.cummax() - 1.0
    daily = wallet.resample("1d").last().pct_change().dropna()
    downside = np.sqrt(np.mean(np.minimum(daily.to_numpy(float), 0.0) ** 2)) if len(daily) else np.nan
    q05 = daily.quantile(.05) if len(daily) else np.nan
    return {
        "ulcer_index": float(np.sqrt(np.mean(drawdown.to_numpy(float) ** 2))),
        "daily_cvar5": float(daily.loc[daily.le(q05)].mean()) if len(daily) else np.nan,
        "daily_sortino": float(daily.mean() / downside * np.sqrt(365.0)) if np.isfinite(downside) and downside > 0 else np.nan,
        "time_underwater_fraction": float(drawdown.lt(0.0).mean()),
    }


def _replay(panel: pd.DataFrame, policy: pd.DataFrame, arm: str, out: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    target_free = panel.copy()
    target_free["dual_admitted"] = target_free.bcf_mapped_expected_bps.ge(ADMISSION_BPS) & target_free.current_mapped_expected_bps.ge(ADMISSION_BPS)
    target_free["auction_priority_bps"] = target_free.bcf_mapped_expected_bps
    outcome = target_free.merge(policy, on="candidate_id", how="left", validate="one_to_one")
    if len(outcome) != len(target_free):
        raise AssertionError(f"{arm}: outcome join altered target-free identity")
    candidates = _to_candidates(outcome, admission=outcome.dual_admitted, priority=outcome.auction_priority_bps)
    rows, all_decisions = [], []
    for scope, start, end in (
        ("2025_strict_oof", pd.Timestamp("2025-07-01", tz="UTC"), pd.Timestamp("2026-01-01", tz="UTC")),
        ("2026_confirmation", pd.Timestamp("2026-01-01", tz="UTC"), END),
        ("2026_jun_jul", pd.Timestamp("2026-06-01", tz="UTC"), END),
    ):
        subset = candidates.loc[candidates.timestamp.ge(start) & candidates.timestamp.lt(end)].copy()
        decisions, equity, _ = replay_candidates(subset, _params(), mode="global_auction", ev_curve=CAUSAL_AUCTION_CURVE, market_mode="perps", initial_wallet=1000.0)
        if decisions.empty:
            decisions["policy_outcome_available"] = pd.Series(dtype=bool)
        else:
            provenance = subset.loc[:, ["candidate_id", "policy_outcome_available"]].reset_index(drop=True)
            provenance.index.name = "candidate_index"
            decisions = decisions.merge(provenance, on="candidate_index", how="left", validate="many_to_one")
        decisions.to_parquet(out / f"{arm}_{scope}_decisions.parquet", index=False, compression="zstd")
        equity.to_parquet(out / f"{arm}_{scope}_equity.parquet", index=False, compression="zstd")
        metric = _metrics(decisions, equity, arm, scope)
        metric.update(_risk_metrics(equity, decisions))
        metric["dual_admitted_rows"] = int(target_free.loc[target_free.__decision_ts__.ge(start) & target_free.__decision_ts__.lt(end), "dual_admitted"].sum())
        rows.append(metric)
        decisions["arm"], decisions["scope"] = arm, scope
        all_decisions.append(decisions)
    return pd.DataFrame(rows), pd.concat(all_decisions, ignore_index=True) if all_decisions else pd.DataFrame()


def run(args: argparse.Namespace) -> Path:
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(f"immutable output exists: {out}")
    full, target_free = _load_panel(args.dynamic.resolve())
    frozen, folds = _contracts(args.assessment.resolve())
    out.mkdir(parents=True, exist_ok=False)
    policy = full.loc[:, ["candidate_id", *POLICY_COLUMNS]].copy()
    # Baseline uses the frozen source maps; it never consumes a dynamic field.
    baseline = target_free.loc[target_free.__decision_ts__.ge(START)].copy()
    baseline["bcf_mapped_expected_bps"] = baseline.bcf_mc1_expected_bps
    baseline["current_mapped_expected_bps"] = baseline.current_mc1_expected_bps
    summary, _ = _replay(baseline, policy, "M0_frozen_pair_control", out)
    summaries = [summary]
    specialist_audits, mapper_audits = [], []
    for family in FAMILIES:
        head, audit = _specialist_oof(full, family, frozen, folds)
        specialist_audits.append(audit)
        full = full.merge(head.loc[:, ["candidate_id", f"{family.lower()}_specialist_residual_bps"]], on="candidate_id", how="left", validate="one_to_one")
        target_free = target_free.merge(head.loc[:, ["candidate_id", f"{family.lower()}_specialist_residual_bps"]], on="candidate_id", how="left", validate="one_to_one")
        for representation in REPRESENTATIONS:
            mapped, audit = _map_one_family(full, target_free, family=family, representation=representation, frozen=frozen, folds=folds)
            mapper_audits.append(audit)
            if mapped.empty:
                continue
            arm = f"{family}_{representation}"
            mapped.to_parquet(out / f"{arm}_target_free_scores.parquet", index=False, compression="zstd")
            # A prequential specialist cannot exist in its first held month.
            # Evaluate its frozen M0 comparator over exactly the same decision
            # support rather than allowing an early unavailable month to
            # masquerade as a selection difference.
            arm_start = mapped["__decision_ts__"].min().floor("1h")
            matched_control = baseline.loc[baseline.__decision_ts__.ge(arm_start)].copy()
            control_arm = f"M0_frozen_pair_control_{arm}_matched"
            control_metrics, _ = _replay(matched_control, policy, control_arm, out)
            control_metrics["comparison_control_arm"] = control_arm
            control_metrics["evaluation_start"] = arm_start
            summaries.append(control_metrics)
            metrics, _ = _replay(mapped, policy, arm, out)
            metrics["comparison_control_arm"] = control_arm
            metrics["evaluation_start"] = arm_start
            summaries.append(metrics)
    result = pd.concat(summaries, ignore_index=True)
    result["comparison_control_arm"] = result.get("comparison_control_arm", "M0_frozen_pair_control").fillna("M0_frozen_pair_control")
    for index, row in result.iterrows():
        reference = result.loc[
            result.arm.eq(row.comparison_control_arm) & result.period.eq(row.period)
        ]
        if len(reference) != 1:
            continue
        for field in ("accepted_rows", "net_ev_bps_per_realised_trade", "net_sum_bps_realised", "worst_month_bps", "worst_week_bps", "max_drawdown", "ulcer_index", "daily_cvar5", "time_underwater_fraction"):
            result.loc[index, f"delta_vs_m0_{field}"] = row[field] - reference.iloc[0][field]
    result.to_parquet(out / "portfolio_summary.parquet", index=False)
    result.to_csv(out / "portfolio_summary.csv", index=False)
    pd.concat(specialist_audits, ignore_index=True).to_parquet(out / "specialist_oof_audit.parquet", index=False)
    pd.concat(mapper_audits, ignore_index=True).to_parquet(out / "mapper_oof_audit.parquet", index=False)
    manifest = {
        "schema": "causal-exotic-dynamics-mc1-ablation-v1",
        "scope": "offline strict temporal ablation; no live/canonical/policy/execution mutation",
        "selection": "2025 prior-three-month nested feature selection; stable recurring contract frozen before 2026 confirmation",
        "representations": list(REPRESENTATIONS), "families": list(FAMILIES),
        "target": "source-aligned parent-policy net bps; specialist target is parent-policy net bps minus frozen paired-MC1 mean",
        "mapper": "separate BCF/current HGB depth2/80 trees/lr .04/L2 20/min leaf 100 plus prior-resolved daily score-band shift",
        "admission": "dual BCF/current >= +50 bps; BCF mapped EV priority", "portfolio": "fixed global 7x/10%-slot, 2-new, 8-concurrent, 80%-wallet auction; invalid outcomes excluded before capacity",
        "dynamic_manifest_sha256": _sha256(args.dynamic.resolve() / "run_manifest.json"),
        "assessment_manifest_sha256": _sha256(args.assessment.resolve() / "run_manifest.json"),
        "no_exchange_calls": True,
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dynamic", type=Path, default=DYNAMIC)
    parser.add_argument("--assessment", type=Path, default=ASSESSMENT)
    parser.add_argument("--out", type=Path, default=OUT)
    print(run(parser.parse_args()))


if __name__ == "__main__":
    main()
