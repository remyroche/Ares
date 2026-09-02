#!/usr/bin/env python3
"""Strict-prequential MC1 combination ablation for incumbent meta heads.

Each candidate meta rank is a target-free OOF receipt.  This runner first
persists their joins to the immutable Current/BCF score families, then joins
the canonical policy ledger only for prior-resolved MC1 fitting and portfolio
diagnostics.  It tests all non-empty combinations of the selected roles and
reports leave-one-role-out deltas (Delta_R/U/O/C) plus pairwise Top-1/2
substitution economics.

Meta heads are inputs to the two MC1 expected-EV mappers only.  They never
directly rerank, route, admit, or trade candidates.  Current and BCF maps are
separate; both must clear the unchanged gate before the common chronological
portfolio auction.

Research only.  It has no live, inference, or exchange authority.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT, ROOT / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import run_strict_r3_incumbent_meta_mc1_screen_v1 as mc1  # noqa: E402


SCHEMA = "strict_r3_incumbent_meta_mc1_combinations_v1"
IDENTITY = mc1.IDENTITY
DEFAULT_PARENT = mc1.DEFAULT_PARENT
DEFAULT_POLICY = mc1.DEFAULT_POLICY


def _exclusive_json(path: Path, payload: object) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def _sha(path: Path) -> str:
    return mc1._sha(path)


def _parse_months(raw: str) -> tuple[pd.Timestamp, ...]:
    return mc1._parse_months(raw)


def _safe(name: str) -> str:
    return "_".join(part for part in name.replace("/", "_").split() if part)


def _read_meta(meta_root: Path, arm: str, month: pd.Timestamp) -> pd.DataFrame:
    data = mc1._read_meta(meta_root, arm, month)
    return data.loc[:, [*IDENTITY, "meta_rank_ts"]].rename(columns={"meta_rank_ts": f"meta__{arm}"})


def _target_free_panels(
    *, parent_root: Path, meta_root: Path, arms: Sequence[str], months: Sequence[pd.Timestamp], out: Path,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    panels: dict[str, list[pd.DataFrame]] = {"current": [], "bcf": []}
    audits: list[dict[str, Any]] = []
    for month in months:
        meta_parts = [_read_meta(meta_root, arm, month) for arm in arms]
        reference = meta_parts[0]
        for part in meta_parts[1:]:
            if not reference.loc[:, list(IDENTITY)].equals(part.loc[:, list(IDENTITY)]):
                raise AssertionError(f"{month:%Y-%m}: meta target-free identities differ by arm")
        meta = reference
        for part in meta_parts[1:]:
            meta = meta.merge(part, on=list(IDENTITY), how="inner", validate="one_to_one")
        if len(meta) != len(reference) or meta.duplicated(IDENTITY).any():
            raise AssertionError(f"{month:%Y-%m}: meta combination identity merge changed rows")
        for family in panels:
            parent = mc1._read_parent(parent_root, family, month)
            merged = parent.merge(meta, on=list(IDENTITY), how="inner", validate="one_to_one")
            if len(merged) != len(meta):
                raise AssertionError(f"{month:%Y-%m} {family}: parent misses target-free meta rows")
            if not merged.enhanced_base_routed.fillna(False).astype(bool).all():
                raise AssertionError(f"{month:%Y-%m} {family}: meta score exists outside canonical base route")
            path = out / "target_free_panels" / family / f"month={month:%Y-%m}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            merged.to_parquet(path, index=False, compression="zstd")
            panels[family].append(merged)
            audits.append({
                "family": family, "month": f"{month:%Y-%m}", "rows": int(len(merged)),
                "meta_arms": list(arms), "target_free_before_policy_join": True, "path": str(path),
            })
    return {family: pd.concat(parts, ignore_index=True) for family, parts in panels.items()}, pd.DataFrame(audits)


def _one_combination(
    *, labelled: dict[str, pd.DataFrame], combo: tuple[str, ...], role_by_arm: dict[str, str], out: Path,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    label = "__".join(role_by_arm[arm] for arm in combo) or "control_no_meta"
    features = [*mc1.PARENT_FEATURES, *(f"meta__{arm}" for arm in combo)]
    predicted: dict[str, pd.DataFrame] = {}
    audit_rows: list[pd.DataFrame] = []
    months = tuple(
        sorted(
            pd.to_datetime(labelled["current"]["__decision_ts__"], utc=True)
            .dt.to_period("M")
            .dt.to_timestamp()
            .dt.tz_localize("UTC")
            .unique()
        )
    )
    for family, panel in labelled.items():
        pred, audit = mc1._mc1_predict(panel, family=family, features=features, months=months)
        predicted[family] = pred.rename(columns={"mc1_expected_bps": f"{family}_mc1_expected_bps"})
        audit["combo"] = label; audit["family"] = family; audit_rows.append(audit)
    current = predicted["current"]
    bcf = predicted["bcf"]
    left = current.loc[:, [
        "candidate_id", "__decision_ts__", "policy_path_valid", "policy_gross_bps", "policy_net_bps",
        "policy_exit_bar_15m", "policy_entry_price", "policy_exit_price", "policy_exit_reason",
        "current_mc1_expected_bps",
    ]]
    right = bcf.loc[:, ["candidate_id", "__decision_ts__", "bcf_mc1_expected_bps"]]
    combined = left.merge(right, on=["candidate_id", "__decision_ts__"], how="inner", validate="one_to_one")
    metric, admitted = mc1._portfolio(combined, arm=f"combo__{_safe(label)}", out=out)
    metric.update({"combo": label, "arms": list(combo), "roles": [role_by_arm[arm] for arm in combo], "feature_count": len(features)})
    admitted = admitted.merge(
        combined.loc[:, ["candidate_id", "__decision_ts__", "policy_net_bps"]],
        on=["candidate_id", "__decision_ts__"], how="left", validate="one_to_one",
    )
    admitted["combo"] = label
    audit = pd.concat(audit_rows, ignore_index=True)
    return metric, admitted, audit


def _top_substitution(admissions: pd.DataFrame, *, combo: str, comparator: str, k: int) -> dict[str, Any]:
    """Post-hoc Top-k substitution economics among already MC1-admitted rows.

    This never enters model fitting or portfolio decisions.  It documents
    whether a role changes the immediate best candidates rather than merely
    shifting a pooled metric.
    """
    left = admissions.loc[admissions.combo.eq(combo)].copy()
    right = admissions.loc[admissions.combo.eq(comparator)].copy()
    if left.empty or right.empty:
        return {"combo": combo, "comparator": comparator, "k": k, "common_timestamps": 0}
    # BCF expected EV is the canonical pre-portfolio priority.  Realised
    # policy net is joined only after these predeclared selections are fixed,
    # making this a diagnostic—not an input to selection or MC1 fitting.
    def choose(frame: pd.DataFrame) -> pd.DataFrame:
        ordered = frame.sort_values(["__decision_ts__", "bcf_mc1_expected_bps", "candidate_id"], ascending=[True, False, True], kind="stable")
        return ordered.groupby("__decision_ts__", sort=False).head(k).loc[:, ["candidate_id", "__decision_ts__"]]
    selected_l, selected_r = choose(left), choose(right)
    selected_l = selected_l.merge(left.loc[:, ["candidate_id", "__decision_ts__", "policy_net_bps"]], on=["candidate_id", "__decision_ts__"], how="left", validate="one_to_one")
    selected_r = selected_r.merge(right.loc[:, ["candidate_id", "__decision_ts__", "policy_net_bps"]], on=["candidate_id", "__decision_ts__"], how="left", validate="one_to_one")
    common = selected_l.loc[:, ["candidate_id", "__decision_ts__"]].merge(selected_r.loc[:, ["candidate_id", "__decision_ts__"]], on=["candidate_id", "__decision_ts__"], how="outer", indicator=True)
    per_ts_left = selected_l.groupby("__decision_ts__", sort=False).policy_net_bps.mean()
    per_ts_right = selected_r.groupby("__decision_ts__", sort=False).policy_net_bps.mean()
    aligned = pd.concat([per_ts_left.rename("combo"), per_ts_right.rename("comparator")], axis=1).dropna()
    return {
        "combo": combo, "comparator": comparator, "k": k,
        "combo_only_selected": int((common._merge == "left_only").sum()),
        "comparator_only_selected": int((common._merge == "right_only").sum()),
        "shared_selected": int((common._merge == "both").sum()),
        "common_timestamps": int(min(selected_l["__decision_ts__"].nunique(), selected_r["__decision_ts__"].nunique())),
        "combo_mean_policy_net_bps": float(aligned.combo.mean()) if len(aligned) else float("nan"),
        "comparator_mean_policy_net_bps": float(aligned.comparator.mean()) if len(aligned) else float("nan"),
        "substitution_delta_policy_net_bps": float((aligned.combo - aligned.comparator).mean()) if len(aligned) else float("nan"),
        "post_selection_policy_diagnostic_only": True,
    }


def run(args: argparse.Namespace) -> None:
    if args.out.exists():
        raise FileExistsError(f"{args.out}: immutable output already exists")
    arms = tuple(item.strip() for item in args.arms.split(",") if item.strip())
    roles = tuple(item.strip() for item in args.roles.split(",") if item.strip())
    if len(arms) != len(roles) or not arms or len(set(arms)) != len(arms) or len(set(roles)) != len(roles):
        raise ValueError("--arms and --roles must have the same non-zero unique length")
    role_by_arm = dict(zip(arms, roles, strict=True))
    months = _parse_months(args.months)
    if months[0] > mc1.EVALUATION_START - pd.DateOffset(months=mc1.MC1_MONTHS):
        raise ValueError("months omit the strict six-month MC1 history")
    args.out.mkdir(parents=True)
    _exclusive_json(args.out / "run_manifest.json", {
        "schema": SCHEMA,
        "scope": "offline strict-prequential incumbent meta-to-MC1 combination research; no live/inference/admission/exchange mutation",
        "incumbent_upstream": "0.50 * efficiency_bps + 0.50 * timing_bps",
        "meta_authority": "additional Current/BCF MC1 coordinates only; no direct route/rank/admission authority",
        "parent_root": str(args.parent_root), "meta_root": str(args.meta_root), "policy": str(args.policy),
        "arms": list(arms), "roles": list(roles), "months": [f"{month:%Y-%m}" for month in months],
        "mc1": {"separate_current_bcf": True, "fit_months": mc1.MC1_MONTHS, "shift_days": mc1.SHIFT_DAYS},
        "admission": {"dual_gate_bps": mc1.ADMISSION_BPS, "priority": "bcf_mc1_expected_bps"},
        "source_hashes": {"parent": _sha(args.parent_root), "meta": _sha(args.meta_root), "policy": _sha(args.policy)},
        "causality": "all meta/parent panels persist target-free before one later policy join; MC1 uses only prior resolved labels",
    })
    parent, audit = _target_free_panels(parent_root=args.parent_root, meta_root=args.meta_root, arms=arms, months=months, out=args.out)
    audit.to_parquet(args.out / "target_free_panel_audit.parquet", index=False, compression="zstd")
    policy = mc1._read_policy(args.policy)
    labelled = {family: panel.merge(policy, on="candidate_id", how="left", validate="one_to_one") for family, panel in parent.items()}
    if any(len(labelled[family]) != len(parent[family]) for family in parent):
        raise AssertionError("policy join changed target-free identity count")
    results: list[dict[str, Any]] = []
    admitted: list[pd.DataFrame] = []
    fit_audits: list[pd.DataFrame] = []
    combos = ((), *(combo for size in range(1, len(arms) + 1) for combo in itertools.combinations(arms, size)))
    for combo in combos:
        metric, cohort, fit_audit = _one_combination(labelled=labelled, combo=combo, role_by_arm=role_by_arm, out=args.out)
        results.append(metric); admitted.append(cohort); fit_audits.append(fit_audit)
    metrics = pd.DataFrame(results)
    all_combo = "__".join(roles)
    deltas: list[dict[str, Any]] = []
    if all_combo in set(metrics.combo):
        full = metrics.set_index("combo").loc[all_combo]
        for arm, role in role_by_arm.items():
            leave = tuple(item for item in arms if item != arm)
            leave_name = "__".join(role_by_arm[item] for item in leave)
            if leave_name not in set(metrics.combo):
                continue
            reduced = metrics.set_index("combo").loc[leave_name]
            row: dict[str, Any] = {"role": role, "full_combo": all_combo, "without_combo": leave_name}
            for field in ("accepted_rows", "net_ev_bps_per_realised_trade", "net_sum_bps_realised", "worst_month_bps", "worst_week_bps", "max_drawdown", "candidate_admitted_rows"):
                if field in full and field in reduced:
                    row[f"Delta_{role}_{field}"] = float(full[field]) - float(reduced[field])
            deltas.append(row)
    all_admitted = pd.concat(admitted, ignore_index=True)
    substitutions: list[dict[str, Any]] = []
    if all_combo in set(all_admitted.combo):
        for row in deltas:
            for k in (1, 2):
                substitutions.append(_top_substitution(all_admitted, combo=all_combo, comparator=str(row["without_combo"]), k=k))
    metrics.to_parquet(args.out / "mc1_combination_metrics.parquet", index=False, compression="zstd")
    pd.DataFrame(deltas).to_parquet(args.out / "mc1_role_deltas.parquet", index=False, compression="zstd")
    all_admitted.to_parquet(args.out / "mc1_admission_provenance.parquet", index=False, compression="zstd")
    pd.DataFrame(substitutions).to_parquet(args.out / "top_substitution_targetfree_selection.parquet", index=False, compression="zstd")
    pd.concat(fit_audits, ignore_index=True).to_parquet(args.out / "mc1_fit_audit.parquet", index=False, compression="zstd")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--meta-root", type=Path, required=True)
    parser.add_argument("--arms", required=True, help="four comma-separated selected meta-arm names")
    parser.add_argument("--roles", default="R,U,O,C", help="same-length role names, used for Delta_R/U/O/C")
    parser.add_argument("--months", default="2025-09,2025-10,2025-11,2025-12,2026-01,2026-02,2026-03,2026-04,2026-05,2026-06,2026-07")
    parser.add_argument("--parent-root", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
