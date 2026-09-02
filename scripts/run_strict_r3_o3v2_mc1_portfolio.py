#!/usr/bin/env python3
"""Strict downstream MC1/portfolio test for O3-v2 consensus candidates.

Research only.  This adapter leaves every live artifact untouched.  It
combines target-free O3 consensus receipts with the pre-existing target-free
current/BCF score families, then fits the exact shallow MC1 class strictly
prequentially.  Policy outcomes are joined only after the combined target-free
panel has passed its identity and leakage checks.

The first valid evaluation starts in April 2026: October 2025--March 2026 are
required to form the six complete prior O3 score months for the MC1 fit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
for item in (ROOT, ROOT / "scripts"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

import run_strict_r3_enhanced_base_live_stack_challenger as parent  # noqa: E402
import run_strict_r3_orthogonal_meta_mc1 as mc1  # noqa: E402


SCHEMA = "strict_r3_o3v2_mc1_portfolio_v4"
DEFAULT_LEDGER_MONTHS = tuple(pd.date_range("2025-10-01", "2026-07-01", freq="MS", tz="UTC"))
DEFAULT_EVALUATION_MONTHS = tuple(pd.date_range("2026-04-01", "2026-07-01", freq="MS", tz="UTC"))
HEADS = ("cap100_ordinary", "cap80_ordinary", "cap120_equal_month", "cap40_equal_month", "cap60_equal_month")
PROHIBITED = frozenset({
    "policy_path_valid", "policy_gross_bps", "policy_net_bps", "policy_exit_bar_15m",
    "policy_entry_price", "policy_exit_price", "policy_exit_reason", "policy_label_available_ts",
    "policy_cost_bps", "semantic_path_valid", "semantic_sequence", "semantic_speed_bin",
    "semantic_persistence_bin", "semantic_pre_adverse_bin", "semantic_policy_conversion_bin",
    "semantic_exit_reason", "semantic_composite", "semantic_tbm_event",
})


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    paths = sorted(path.rglob("*.parquet")) if path.is_dir() else [path]
    for child in paths:
        digest.update(str(child).encode())
        with child.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    return digest.hexdigest()


def _parse_arm_sources(values: list[str]) -> dict[str, tuple[Path, ...]]:
    """Parse ``ARM=ROOT|ROOT`` mappings, allowing a late-funnel source split."""
    result: dict[str, tuple[Path, ...]] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"expected ARM=ROOT[|ROOT], got {value!r}")
        arm, raw = value.split("=", 1)
        roots = tuple(Path(part) for part in raw.split("|") if part)
        if not arm or not roots:
            raise ValueError(f"invalid arm source {value!r}")
        result[arm] = roots
    if not result:
        raise ValueError("at least one --arm-source is required")
    return result


def _source_path(roots: tuple[Path, ...], arm: str, month: pd.Timestamp) -> Path:
    """Resolve one immutable target-free receipt for ``arm`` and ``month``.

    Earlier target-funnel outputs stored scores beneath ``<arm>/``.  The
    support funnel intentionally namespaces a receipt as
    ``<arm>__<selected-support>/`` so a strict MC1 ledger cannot silently mix
    weights.  Accept either representation, but require one and only one
    physical source across all declared roots.  Ambiguity remains a hard
    failure rather than an arbitrary glob choice.
    """
    name = f"month={month:%Y-%m}.parquet"
    available: list[Path] = []
    for root in roots:
        score_root = root / "target_free_scores"
        direct = score_root / arm / name
        if direct.exists():
            available.append(direct)
        available.extend(sorted(score_root.glob(f"{arm}__*/{name}")))
    found = [path for path in available if path.exists()]
    if len(found) != 1:
        raise FileNotFoundError(f"{arm} {month:%Y-%m}: expected exactly one source, found {found}")
    return found[0]


PARENT_FEATURE_MODES = {
    # Historical diagnostic: preserves every incumbent correction coordinate
    # and appends experimental O3 fields.  This is not the replacement
    # architecture and remains available only for continuity/reconciliation.
    "additive": tuple(parent.MC1_FEATURES),
    # Canonical challenger: the new O3 heads replace the incumbent consensus
    # inputs.  Only base-calibration and causal correctness coordinates remain
    # from the parent.  In particular, this excludes final_score, upstream,
    # conditional_consensus_rank, ordinary_shadow_consensus_rank, and every
    # incumbent per-head rank derived from the prior correction stack.
    "replace_correction": (
        "base_rank42",
        "base_anchor_bps",
        "correctness_rank",
    ),
}


def _load_physical_slots(path: Path | None, arms: tuple[str, ...]) -> tuple[dict[str, str] | None, str | None]:
    """Load the one-physical-head-per-target successor contract.

    A missing contract is permitted solely for the explicit same-parent
    control.  Any O3 challenger arm must declare a selected physical head so
    it cannot reconstitute a five-slot target ensemble downstream.
    """
    if not arms:
        if path is not None:
            raise ValueError("a physical-slot selection is meaningless for the parent-only control")
        return None, None
    if path is None:
        raise ValueError("O3 challenger arms require --physical-slot-selection")
    payload = json.loads(path.read_text())
    if payload.get("schema") != "strict_r3_o3v2_physical_slot_selection_v1":
        raise AssertionError("unknown physical-slot selection schema")
    slots = payload.get("selected_slots")
    if not isinstance(slots, dict):
        raise AssertionError("physical-slot selection has no selected_slots mapping")
    missing = sorted(set(arms) - set(slots))
    if missing:
        raise AssertionError(f"physical-slot selection misses O3 arms: {missing}")
    selected = {arm: str(slots[arm]) for arm in arms}
    invalid = sorted(set(selected.values()) - set(HEADS))
    if invalid:
        raise AssertionError(f"physical-slot selection names unknown slots: {invalid}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return selected, digest


def _feature_names(
    arms: tuple[str, ...], mode: str, parent_feature_mode: str,
    physical_slots: dict[str, str] | None,
) -> tuple[str, ...]:
    if parent_feature_mode not in PARENT_FEATURE_MODES:
        raise ValueError(f"unsupported parent feature mode: {parent_feature_mode}")
    fields = list(PARENT_FEATURE_MODES[parent_feature_mode])
    for arm in arms:
        prefix = f"o3__{arm.lower()}__"
        if mode in {"aggregate", "full"}:
            # Each target now contributes one physical correction head.  Its
            # former within-target ensemble dispersion is identically zero
            # and is intentionally absent rather than injected as a fake
            # feature.
            fields.extend((f"{prefix}consensus_rank", f"{prefix}combined_rank"))
            # This delta explicitly refers to the incumbent consensus; it is
            # useful for an additive diagnostic but would violate a genuine
            # replacement-head architecture.
            if parent_feature_mode == "additive":
                fields.append(f"{prefix}delta_parent_consensus")
        if mode in {"heads", "full"}:
            if physical_slots is None:
                raise AssertionError("O3 head features require a frozen physical-slot contract")
            fields.append(f"{prefix}{physical_slots[arm]}_rank")
    return tuple(fields)


def _load_family(
    p2_root: Path,
    arm_sources: dict[str, tuple[Path, ...]],
    family: str,
    months: tuple[pd.Timestamp, ...],
    parent_feature_mode: str,
    physical_slots: dict[str, str] | None,
) -> pd.DataFrame:
    """Build an all-target-free, routed O3 panel for one parent score family."""
    if parent_feature_mode not in PARENT_FEATURE_MODES:
        raise ValueError(f"unsupported parent feature mode: {parent_feature_mode}")
    parent_columns = (
        "candidate_id", "__decision_ts__", "side_name", "enhanced_base_routed", "enhanced_base_bps",
        *PARENT_FEATURE_MODES[parent_feature_mode],
    )
    pieces: list[pd.DataFrame] = []
    arms = tuple(arm_sources)
    if arms and physical_slots is None:
        raise AssertionError("O3 source loading requires the frozen physical-slot contract")
    for month in months:
        token = f"{month:%Y-%m}"
        parent_path = p2_root / "target_free_scores" / family / f"month={token}.parquet"
        if not parent_path.exists():
            raise FileNotFoundError(parent_path)
        frame = pd.read_parquet(parent_path, columns=parent_columns)
        frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True, errors="raise")
        # Never trust a persisted route flag at this downstream boundary.
        # Reconstruct the canonical deterministic timestamp-local top-30%
        # route from the sealed upstream score.  This prevents a malformed or
        # legacy-tie receipt from exposing an otherwise eligible candidate to
        # either experimental consensus features or MC1.
        canonical_route = parent._exact_timestamp_top_fraction(
            frame, "enhanced_base_bps", parent.BASE_ROUTE,
        )
        frame = frame.loc[canonical_route].copy()
        frame["enhanced_base_routed"] = True
        if frame.empty:
            raise AssertionError(f"{family} {token}: canonical base route is empty")
        for arm in arms:
            source = _source_path(arm_sources[arm], arm, month)
            probe = pd.read_parquet(source)
            leaked = sorted(PROHIBITED.intersection(probe.columns))
            if leaked:
                raise AssertionError(f"{source}: outcome fields in target-free score receipt: {leaked}")
            slot = physical_slots[arm] if physical_slots is not None else None
            required = ["candidate_id", "base_rank_ts", f"head__{slot}__rank"]
            missing = sorted(set(required) - set(probe.columns))
            if missing:
                raise AssertionError(f"{source}: missing O3 fields {missing}")
            meta = probe.loc[:, required].copy()
            if meta["candidate_id"].duplicated().any():
                raise AssertionError(f"{source}: duplicate candidate identities")
            prefix = f"o3__{arm.lower()}__"
            rank = pd.to_numeric(meta[f"head__{slot}__rank"], errors="coerce")
            base_rank = pd.to_numeric(meta["base_rank_ts"], errors="coerce")
            if rank.isna().any() or base_rank.isna().any():
                raise AssertionError(f"{source}: selected {slot} head or base rank is incomplete")
            # Rebuild the target-family aggregate from the sole selected
            # physical slot.  Do not consume a persisted five-slot median.
            meta = meta.loc[:, ["candidate_id"]].copy()
            meta[f"{prefix}consensus_rank"] = rank.astype(np.float32)
            meta[f"{prefix}combined_rank"] = (.75 * base_rank + .25 * rank).astype(np.float32)
            meta[f"{prefix}{slot}_rank"] = rank.astype(np.float32)
            # An inner merge deliberately removes candidates outside the O3
            # base route.  Missing O3 outputs are not treated as a model value.
            frame = frame.merge(meta, on="candidate_id", how="inner", validate="one_to_one")
            if parent_feature_mode == "additive":
                frame[f"{prefix}delta_parent_consensus"] = (
                    pd.to_numeric(frame[f"{prefix}consensus_rank"], errors="coerce")
                    - pd.to_numeric(frame["conditional_consensus_rank"], errors="coerce")
                )
        if frame["candidate_id"].duplicated().any():
            raise AssertionError(f"{family} {token}: duplicate merged identities")
        if not frame["enhanced_base_routed"].all():
            raise AssertionError(f"{family} {token}: MC1 input escaped canonical base route")
        if parent_feature_mode == "replace_correction":
            replacement_columns = [f"o3__{arm.lower()}__combined_rank" for arm in arms]
            if not replacement_columns:
                raise AssertionError("replacement correction mode requires at least one O3 head")
            values = frame.loc[:, replacement_columns].apply(pd.to_numeric, errors="coerce")
            if values.isna().any(axis=None):
                raise AssertionError(f"{family} {token}: replacement score lacks an O3 combined rank")
            # The same replacement score is used solely as the causal
            # score-band and portfolio-priority coordinate in the MC1 helper.
            # It is the equal-weight combination of target-family scores,
            # each of which uses exactly its frozen physical head, not a
            # hidden legacy or within-target ensemble score.
            frame["final_score"] = values.mean(axis=1).astype(np.float32)
        pieces.append(frame)
    output = pd.concat(pieces, ignore_index=True)
    if output["candidate_id"].duplicated().any():
        raise AssertionError(f"{family}: duplicate candidate identities across monthly receipts")
    return output


def _load_policy(path: Path) -> pd.DataFrame:
    columns = [
        "candidate_id", "policy_path_valid", "policy_gross_bps", "policy_net_bps", "policy_exit_bar_15m",
        "policy_entry_price", "policy_exit_price", "policy_exit_reason", "policy_label_available_ts", "policy_cost_bps",
    ]
    policy = pd.read_parquet(path, columns=columns)
    if policy["candidate_id"].duplicated().any():
        raise AssertionError("canonical policy ledger has duplicate candidate IDs")
    policy["policy_label_available_ts"] = pd.to_datetime(policy["policy_label_available_ts"], utc=True, errors="coerce")
    return policy


def _baseline(current_path: Path, bcf_path: Path, policy: pd.DataFrame) -> pd.DataFrame:
    # Some valid offline parent receipts derive the symbol from candidate_id
    # instead of redundantly persisting ``__symbol__``.  The portfolio
    # comparison needs the same identity information, not a particular
    # storage layout; derive it only when the immutable receipt lacks it.
    probe_columns = set(pq.ParquetFile(current_path).schema_arrow.names)
    fields = ["candidate_id", "__decision_ts__", "side_name", "final_score", "mc1_expected_bps"]
    if "__symbol__" in probe_columns:
        fields.insert(2, "__symbol__")
    current = pd.read_parquet(current_path, columns=fields).rename(columns={
        "final_score": "current_final_score", "mc1_expected_bps": "current_mc1_expected_bps",
    })
    bcf = pd.read_parquet(bcf_path, columns=["candidate_id", "__decision_ts__", "final_score", "mc1_expected_bps"]).rename(columns={
        "final_score": "bcf_final_score", "mc1_expected_bps": "bcf_mc1_expected_bps",
    })
    for frame in (current, bcf):
        frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True, errors="raise")
    if "__symbol__" not in current.columns:
        current["__symbol__"] = current["candidate_id"].astype(str).str.split("|", n=1, expand=True)[0]
    current["enhanced_base_routed"] = True
    return current.merge(bcf, on=["candidate_id", "__decision_ts__"], how="inner", validate="one_to_one").merge(
        policy, on="candidate_id", how="left", validate="one_to_one",
    )


def _run_metrics(frame: pd.DataFrame, label: str, evaluation_name: str, out: Path, threshold: float) -> dict[str, object]:
    previous = parent.MC1_THRESHOLD_BPS
    try:
        parent.MC1_THRESHOLD_BPS = float(threshold)
        metrics = parent._portfolio_metrics(frame, label, evaluation_name, out)
    finally:
        parent.MC1_THRESHOLD_BPS = previous
    metrics["threshold_bps"] = float(threshold)
    return metrics


def run(
    *, p2_root: Path, policy_path: Path, live_current: Path, live_bcf: Path,
    out: Path, arm_sources: dict[str, tuple[Path, ...]], thresholds: tuple[float, ...], feature_mode: str, mc1_seed: int,
    parent_feature_mode: str = "additive",
    ledger_months: tuple[pd.Timestamp, ...] = DEFAULT_LEDGER_MONTHS,
    evaluation_months: tuple[pd.Timestamp, ...] = DEFAULT_EVALUATION_MONTHS,
    physical_slot_selection: Path | None = None,
) -> None:
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    policy = _load_policy(policy_path)
    arms = tuple(arm_sources)
    physical_slots, physical_slot_selection_sha256 = _load_physical_slots(physical_slot_selection, arms)
    features = _feature_names(arms, feature_mode, parent_feature_mode, physical_slots)
    # Reuse the frozen implementation and its model class/hyperparameters;
    # narrow its month set to the valid O3 score ledger.
    old_months = mc1.SCORE_MONTHS
    old_seed = mc1.SEED
    if not evaluation_months or not set(evaluation_months).issubset(set(ledger_months)):
        raise ValueError("evaluation months must be a non-empty subset of ledger months")
    if min(evaluation_months) - pd.DateOffset(months=6) < min(ledger_months):
        raise ValueError("each emitted MC1 month requires six complete preceding O3 score months in the ledger")
    # All earlier score months remain in ``panel`` as strict prior support.
    # Emit only the caller-declared months, rejecting implicit warm-up output.
    mc1.SCORE_MONTHS = evaluation_months
    mc1.SEED = int(mc1_seed)
    try:
        family_predictions: dict[str, pd.DataFrame] = {}
        audits: list[pd.DataFrame] = []
        for family in ("current", "bcf"):
            target_free = _load_family(
                p2_root, arm_sources, family, ledger_months, parent_feature_mode, physical_slots,
            )
            if PROHIBITED.intersection(target_free.columns):
                raise AssertionError(f"{family}: target-free merge contains policy or semantic outcomes")
            panel = target_free.merge(policy, on="candidate_id", how="left", validate="one_to_one")
            prediction, audit = mc1._predictions(panel, features, family, out)
            family_predictions[family] = prediction
            audits.append(audit)
    finally:
        mc1.SCORE_MONTHS = old_months
        mc1.SEED = old_seed
    current, bcf = family_predictions["current"], family_predictions["bcf"]
    challenger = mc1._combine(current, bcf)
    start = min(evaluation_months)
    end = max(evaluation_months) + pd.offsets.MonthBegin(1)
    evaluation_name = f"{start:%Y%m}_{(end - pd.Timedelta(days=1)):%Y%m}"
    challenger = challenger.loc[challenger["__decision_ts__"].ge(start) & challenger["__decision_ts__"].lt(end)].copy()
    baseline = _baseline(live_current, live_bcf, policy)
    baseline = baseline.loc[baseline["__decision_ts__"].ge(start) & baseline["__decision_ts__"].lt(end)].copy()
    common = pd.Index(challenger["candidate_id"].astype(str)).intersection(pd.Index(baseline["candidate_id"].astype(str)), sort=False)
    challenger_matched = challenger.loc[challenger["candidate_id"].astype(str).isin(common)].copy()
    baseline_matched = baseline.loc[baseline["candidate_id"].astype(str).isin(common)].copy()
    if challenger_matched.empty:
        raise AssertionError("no common candidate identities with current-live control")
    # An empty arm set is an explicit *same-parent score-only* control.  It
    # is useful because a comparison to a later live bundle can otherwise
    # confound an O3-head effect with a parent-score-coordinate change.
    arm_label = "same_parent_control" if not arms else "o3v2"
    results: list[dict[str, object]] = []
    for threshold in thresholds:
        results.append(_run_metrics(challenger, f"{arm_label}_full_{int(threshold)}", evaluation_name, out, threshold))
        results.append(_run_metrics(challenger_matched, f"{arm_label}_matched_{int(threshold)}", evaluation_name, out, threshold))
        results.append(_run_metrics(baseline_matched, f"live_control_matched_{int(threshold)}", evaluation_name, out, threshold))
    metrics = pd.DataFrame(results)
    metrics.to_parquet(out / "portfolio_metrics.parquet", index=False, compression="zstd")
    deltas: list[dict[str, object]] = []
    for threshold in thresholds:
        new = metrics.loc[metrics["arm"].eq(f"{arm_label}_matched_{int(threshold)}")].iloc[0]
        old = metrics.loc[metrics["arm"].eq(f"live_control_matched_{int(threshold)}")].iloc[0]
        row: dict[str, object] = {"threshold_bps": threshold}
        for field in ("accepted_rows", "realised_rows", "net_ev_bps_per_realised_trade", "net_sum_bps_realised", "worst_month_bps", "worst_week_bps", "max_drawdown"):
            if field in new and field in old:
                row[f"delta_{field}"] = float(new[field]) - float(old[field])
        deltas.append(row)
    pd.DataFrame(deltas).to_parquet(out / "delta_vs_live_control.parquet", index=False, compression="zstd")
    pd.concat(audits, ignore_index=True).to_parquet(out / "mc1_fit_audit.parquet", index=False, compression="zstd")
    manifest = {
        "schema": SCHEMA,
        "scope": "offline challenger only; no live artifact, configuration, or process was changed",
        "arms": list(arms), "feature_mode": feature_mode, "parent_feature_mode": parent_feature_mode,
        "physical_slot_selection": str(physical_slot_selection) if physical_slot_selection else None,
        "physical_slot_selection_sha256": physical_slot_selection_sha256,
        "selected_physical_slots": physical_slots,
        "mc1_seed": int(mc1_seed), "mc1_features": list(features), "thresholds_bps": list(thresholds),
        "O3_ledger_months": [f"{month:%Y-%m}" for month in ledger_months],
        "emitted_MC1_months": [f"{month:%Y-%m}" for month in evaluation_months],
        "evaluation": {evaluation_name: [str(start), str(end)]},
        "comparison_population": {"challenger_rows": int(len(challenger)), "baseline_rows": int(len(baseline)), "common_ids": int(len(common)), "matched_challenger_rows": int(len(challenger_matched))},
        "causality": {
            "O3_features": (
                "target-free strict OOF consensus receipts only"
                if arms else "none; same-parent MC1 score-only control"
            ),
            "parent_feature_mode": (
                "new O3 target families replace incumbent correction features; each family contributes exactly one development-frozen physical head and their mean combined rank defines MC1 score bands and auction priority; base-only calibration and correctness coordinates retained"
                if parent_feature_mode == "replace_correction"
                else "incumbent correction coordinates retained and O3 features appended"
            ),
            "MC1_fit": "six complete prior calendar months, using policy labels resolved before held month",
            "admission": "same dual current/BCF MC1 threshold and parent constrained auction as control",
            "policy": "canonical reconciled rich-policy ledger shared by challenger and baseline",
        },
        "source_hashes": {
            "p2_root": _hash(p2_root), "policy": _hash(policy_path), "live_current": _hash(live_current), "live_bcf": _hash(live_bcf),
            **{f"arm_{arm}": ":".join(_hash(root) for root in roots) for arm, roots in arm_sources.items()},
        },
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p2-root", type=Path, required=True)
    parser.add_argument("--policy-path", type=Path, required=True)
    parser.add_argument("--live-current", type=Path, required=True)
    parser.add_argument("--live-bcf", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--arm-source", action="append", default=[], help="ARM=ROOT[|ROOT]; may be given twice")
    parser.add_argument(
        "--physical-slot-selection", type=Path,
        help="frozen one-physical-head-per-target contract; mandatory whenever --arm-source is used",
    )
    parser.add_argument("--parent-only-control", action="store_true", help="run the same-parent MC1 control with no O3 specialist inputs")
    parser.add_argument("--thresholds", default="30,50", help="dual MC1 admission thresholds in bps")
    parser.add_argument("--feature-mode", choices=("aggregate", "heads", "full"), default="full")
    parser.add_argument(
        "--parent-feature-mode", choices=tuple(PARENT_FEATURE_MODES), default="additive",
        help="whether O3 heads augment or replace the incumbent correction coordinates",
    )
    parser.add_argument("--mc1-seed", type=int, default=1729)
    parser.add_argument("--ledger-months", help="comma-separated YYYY-MM; must include six complete months before evaluation")
    parser.add_argument("--evaluation-months", help="comma-separated YYYY-MM emitted as strict MC1 held months")
    args = parser.parse_args()
    ledger_months = tuple(pd.Timestamp(f"{value}-01", tz="UTC") for value in args.ledger_months.split(",")) if args.ledger_months else DEFAULT_LEDGER_MONTHS
    evaluation_months = tuple(pd.Timestamp(f"{value}-01", tz="UTC") for value in args.evaluation_months.split(",")) if args.evaluation_months else DEFAULT_EVALUATION_MONTHS
    if args.parent_only_control and args.arm_source:
        parser.error("--parent-only-control cannot be combined with --arm-source")
    arm_sources = {} if args.parent_only_control else _parse_arm_sources(args.arm_source)
    run(
        p2_root=args.p2_root, policy_path=args.policy_path, live_current=args.live_current,
        live_bcf=args.live_bcf, out=args.out, arm_sources=arm_sources,
        thresholds=tuple(float(value) for value in args.thresholds.split(",") if value), feature_mode=args.feature_mode,
        mc1_seed=args.mc1_seed, parent_feature_mode=args.parent_feature_mode,
        ledger_months=ledger_months, evaluation_months=evaluation_months,
        physical_slot_selection=args.physical_slot_selection,
    )


if __name__ == "__main__":
    main()
