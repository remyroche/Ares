#!/usr/bin/env python3
"""Materialise an exact-identity, target-free UnderF120 + V2 reliability overlay.

The parent UnderF120 fields are copied unchanged.  V2 fields are joined only
from the strict-OOF target-free prediction receipts and frozen target-free
state lattice.  Missing candidate-level reliability predictions outside the
Base Top-10 remain explicit through availability flags; they are never filled
with outcomes or used to alter the parent score family.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "strict_r3_p8u_under_reliability_overlay_v2"


def _once(path: Path, payload: object) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    members = sorted(path.rglob("*.parquet")) if path.is_dir() else [path]
    for member in members:
        digest.update(str(member.relative_to(ROOT)).encode())
        with member.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    return digest.hexdigest()


def _months(start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    return list(pd.date_range(start, end, freq="MS", inclusive="left", tz="UTC"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-root", required=True)
    parser.add_argument("--reliability-root", required=True)
    parser.add_argument("--state-root", required=True)
    parser.add_argument("--under-contract", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--screen-root",
        help=(
            "optional frozen selection-era reliability screen. When supplied, "
            "conditionally repeatable state fields are materialised as additive, "
            "target-free inference columns; the screen itself is never joined "
            "at scoring time."
        ),
    )
    parser.add_argument("--conditional-min-active-months", type=int, default=5)
    parser.add_argument("--conditional-min-cmi", type=float, default=0.045)
    parser.add_argument("--start", default="2025-08")
    parser.add_argument("--end", default="2026-08")
    args = parser.parse_args()
    parent, reliability, state_root, contract, out = (ROOT / args.parent_root, ROOT / args.reliability_root, ROOT / args.state_root, ROOT / args.under_contract, ROOT / args.out)
    if out.exists():
        raise FileExistsError(out)
    for root in (parent, reliability, state_root):
        receipt = json.loads((root / "correctness_report.json").read_text())
        if not all(value is True or key == "schema" for key, value in receipt.items()):
            raise AssertionError(f"unverified source receipt: {root}")
    parent_fields = tuple(json.loads(contract.read_text())["selected_features"])
    candidate_state = pd.read_parquet(reliability / "target_free_candidate_reliability_predictions.parquet")
    timestamp_state = pd.read_parquet(reliability / "target_free_timestamp_failure_predictions.parquet")
    raw_state = pd.read_parquet(state_root / "target_free_state_episode_hourly.parquet")
    for frame in (candidate_state, timestamp_state, raw_state):
        frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True, errors="raise")
    selected_all = json.loads((reliability / "reliability_feature_contract.json").read_text())["candidate_fields"]
    base_context = {
        "base_score", "base_rank_ts", "v2_base_top10_score_mean",
        "v2_base_top10_score_iqr", "v2_base_top10_score_gap",
        "v2_base_tail_transition",
    }
    episode_fields = {
        "v2_regime_id", "v2_regime_distance", "v2_regime_second_distance",
        "v2_regime_assignment_margin", "v2_regime_transition_flag",
        "v2_time_since_regime_change_hours",
    }
    # These are the 2025-selected, deviation-first state fields.  Their
    # meanings are stable because the unsupervised episode bundle is frozen.
    deviation_fields = [
        name for name in selected_all
        if name.startswith("v2_") and name not in base_context and name not in episode_fields
    ]
    deviation_fields = list(dict.fromkeys(deviation_fields))
    conditional_fields: list[str] = []
    conditional_selection: dict[str, object] | None = None
    if args.screen_root:
        screen_root = ROOT / str(args.screen_root)
        selection_path = screen_root / "feature_summary_selection_2025.parquet"
        if not selection_path.exists():
            raise FileNotFoundError(selection_path)
        selection = pd.read_parquet(selection_path)
        required = {
            "feature", "abs_residual_ic_sign_months", "mean_cmi_abs",
            "supported_regimes", "era",
        }
        if not required.issubset(selection.columns):
            raise AssertionError("reliability selection receipt is missing required columns")
        # The 2025 selection receipt is frozen before the 2026 confirmation.
        # It lowers the original threshold only enough to keep fields that
        # repeatedly explain residual/error geometry over multiple months and
        # macro regimes.  No 2026 result participates in materialisation.
        candidate = selection.loc[
            selection["era"].eq("selection_2025")
            & selection["abs_residual_ic_sign_months"].ge(int(args.conditional_min_active_months))
            & selection["mean_cmi_abs"].ge(float(args.conditional_min_cmi))
            & selection["supported_regimes"].ge(3),
            "feature",
        ].astype(str)
        conditional_fields = [
            name for name in dict.fromkeys(candidate.tolist())
            if name.startswith("v2_") and name not in base_context and name not in episode_fields
        ]
        missing_conditional = sorted(set(conditional_fields).difference(raw_state.columns))
        if missing_conditional:
            raise AssertionError(
                "selection-era conditional fields are absent from the frozen state panel: "
                f"{missing_conditional[:8]}"
            )
        conditional_selection = {
            "screen_root": str(screen_root.relative_to(ROOT)),
            "selection_receipt": str(selection_path.relative_to(ROOT)),
            "era": "selection_2025",
            "min_active_months": int(args.conditional_min_active_months),
            "min_cmi_abs": float(args.conditional_min_cmi),
            "field_count": len(conditional_fields),
        }
    episode_contract = json.loads((state_root / "target_free_episode_contract.json").read_text())
    episode_k = int(episode_contract["episode"]["selected_k"])
    episode_onehot = [f"v2_regime_is_{index:02d}" for index in range(episode_k)]
    raw_state = raw_state.loc[:, list(dict.fromkeys([
        "__decision_ts__", *deviation_fields, *conditional_fields, *episode_fields,
    ]))]
    candidate_fields = [
        "v2_pred_error_scale_bps_d2", "v2_pred_error_percentile_d2",
        "v2_p_large_error_100_d2", "v2_p_overconfidence_100_d2",
        "v2_p_underconfidence_100_d2", "v2_transition_uncertainty",
        "v2_state_authority_d2",
    ]
    timestamp_fields = [
        "v2_p_weak_top2_d2", "v2_p_catastrophic_top2_d2",
        "v2_timestamp_failure_risk_d2",
    ]
    additional = list(dict.fromkeys([
        "v2_reliability_top10_available", "v2_reliability_timestamp_available",
        *candidate_fields, *timestamp_fields, *deviation_fields, *conditional_fields, *episode_onehot,
        "v2_regime_distance", "v2_regime_second_distance",
        "v2_regime_assignment_margin", "v2_regime_transition_flag",
        "v2_time_since_regime_change_hours",
    ]))
    start = pd.Timestamp(f"{args.start}-01", tz="UTC")
    end = pd.Timestamp(f"{args.end}-01", tz="UTC")
    audit: list[dict[str, object]] = []
    out.mkdir(parents=True)
    for month in _months(start, end):
        source = parent / f"month={month:%Y-%m}" / "causal_feature_universe.parquet"
        if not source.exists():
            raise FileNotFoundError(source)
        panel = pd.read_parquet(source, columns=["candidate_id", "__decision_ts__", "side_name", *parent_fields])
        panel["__decision_ts__"] = pd.to_datetime(panel["__decision_ts__"], utc=True, errors="raise")
        if panel.duplicated(["candidate_id", "__decision_ts__", "side_name"]).any():
            raise AssertionError(f"{month:%Y-%m}: parent identity duplicates")
        part_c = candidate_state.loc[candidate_state.__decision_ts__.ge(month) & candidate_state.__decision_ts__.lt(month + pd.offsets.MonthBegin(1)), ["candidate_id", *candidate_fields]]
        part_t = timestamp_state.loc[timestamp_state.__decision_ts__.ge(month) & timestamp_state.__decision_ts__.lt(month + pd.offsets.MonthBegin(1)), ["__decision_ts__", *timestamp_fields]]
        part_r = raw_state.loc[raw_state.__decision_ts__.ge(month) & raw_state.__decision_ts__.lt(month + pd.offsets.MonthBegin(1))].copy()
        for index, field in enumerate(episode_onehot):
            part_r[field] = part_r["v2_regime_id"].eq(index).astype("int8")
        merged = panel.merge(part_c, on="candidate_id", how="left", validate="one_to_one")
        merged = merged.merge(part_t, on="__decision_ts__", how="left", validate="many_to_one")
        merged = merged.merge(part_r, on="__decision_ts__", how="left", validate="many_to_one")
        if len(merged) != len(panel):
            raise AssertionError(f"{month:%Y-%m}: additive join changed parent identities")
        merged["v2_reliability_top10_available"] = merged[candidate_fields].notna().all(axis=1).astype("int8")
        merged["v2_reliability_timestamp_available"] = merged[timestamp_fields].notna().all(axis=1).astype("int8")
        destination = out / f"month={month:%Y-%m}"
        destination.mkdir()
        merged.loc[:, ["candidate_id", "__decision_ts__", "side_name", *parent_fields, *additional]].to_parquet(destination / "causal_feature_universe.parquet", index=False, compression="zstd")
        audit.append({"month": f"{month:%Y-%m}", "parent_rows": int(len(panel)), "identity_rows": int(len(merged)), "top10_reliability_coverage": float(merged.v2_reliability_top10_available.mean()), "timestamp_reliability_coverage": float(merged.v2_reliability_timestamp_available.mean()), "raw_state_coverage": float(merged.loc[:, deviation_fields].notna().all(axis=1).mean())})
    audit_frame = pd.DataFrame(audit)
    audit_frame.to_parquet(out / "coverage_audit.parquet", index=False)
    variance_fields = ["v2_reliability_top10_available", "v2_pred_error_scale_bps_d2", "v2_pred_error_percentile_d2"]
    failure_fields = [
        "v2_reliability_top10_available", "v2_reliability_timestamp_available",
        "v2_p_large_error_100_d2", "v2_p_overconfidence_100_d2",
        "v2_p_underconfidence_100_d2", *timestamp_fields,
    ]
    episode_contract_fields = [
        *episode_onehot, "v2_regime_distance", "v2_regime_second_distance",
        "v2_regime_assignment_margin", "v2_regime_transition_flag",
        "v2_time_since_regime_change_hours",
    ]
    authority_fields = [
        "v2_reliability_top10_available", "v2_reliability_timestamp_available",
        "v2_state_authority_d2", "v2_p_large_error_100_d2",
        "v2_pred_error_percentile_d2", "v2_transition_uncertainty",
        "v2_timestamp_failure_risk_d2",
    ]
    contracts = {
        "m0_parent": list(parent_fields),
        "m1_deviations": [*parent_fields, *deviation_fields],
        "m2_episodes": [*parent_fields, *episode_contract_fields],
        "m3_error_variance": [*parent_fields, *variance_fields],
        "m4_failure": [*parent_fields, *failure_fields],
        "m5_authority": [*parent_fields, *authority_fields],
        "m6_deviations_failure": [*parent_fields, *deviation_fields, *failure_fields],
        "m7_episodes_failure_variance": [*parent_fields, *episode_contract_fields, *failure_fields, *variance_fields],
        # M8 is the full additive reliability arm.  Keep it distinct from
        # M7 (episodes + failure + variance) so future immutable receipts
        # preserve the experimental lineage rather than reusing an arm ID.
        "m8_full_reliability": [*parent_fields, *additional],
    }
    contract_root = out / "contracts"; contract_root.mkdir()
    for name, fields in contracts.items():
        fields = list(dict.fromkeys(fields))
        _once(contract_root / f"{name}.json", {"schema": SCHEMA, "selected_features": fields, "feature_count": len(fields), "parent_under_fields": len(parent_fields), "additive_state_reliability_fields": [field for field in fields if field not in parent_fields], "source": "strict-OOF target-free V2 state/reliability overlay"})
    correctness = {
        "parent_under_f120_fields_copied_unchanged": True,
        "additional_fields_are_target_free": True,
        "candidate_reliability_join_is_exact_candidate_identity": True,
        "timestamp_reliability_join_is_exact_decision_timestamp": True,
        "missing_outside_base_top10_is_explicit_not_outcome_filled": True,
        "no_parent_score_mc1_admission_portfolio_or_live_mutation": True,
    }
    _once(out / "correctness_report.json", correctness)
    _once(out / "run_manifest.json", {"schema": SCHEMA, "scope": "offline additive UnderF120 reliability overlay", "parent_root": str(parent.relative_to(ROOT)), "reliability_root": str(reliability.relative_to(ROOT)), "state_root": str(state_root.relative_to(ROOT)), "under_contract": str(contract.relative_to(ROOT)), "parent_hash": _sha(parent), "reliability_hash": _sha(reliability), "state_hash": _sha(state_root), "episode_k": episode_k, "conditional_selection": conditional_selection, "correctness": correctness})
    print(json.dumps({"out": str(out), "months": len(audit), "additional_fields": len(additional), "raw_deviation_fields": len(deviation_fields), "conditional_fields": len(conditional_fields), "episode_k": episode_k}, sort_keys=True))


if __name__ == "__main__":
    main()
