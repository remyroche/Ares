#!/usr/bin/env python3
"""Extend the retained direct MC1 S/R/E2 input study to archived August scores.

This is an offline, partial-August continuation of the June--July study.  It
uses the same two score families, absolute policy-EV target, 21-day causal
shift, dual +50-bps admission, BCF-EV priority and controlled global auction.
The August score archive ends at 2026-08-18 21:00 UTC, so it intentionally
does not claim a full-calendar-month or live-policy result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_canonical_sr_e2_mc1_input_ablation as base
from scripts import run_causal_sr_mc1_residual_ablation as sr
from scripts.run_strict_r3_mc1_d2_controlled_ablation import _score_bands


HISTORY_CURRENT = ROOT / "data_perp/artifacts/strict_r3_score_family_matched_mc1_canonical_policy_20260817_v7_current/predictions_current_v5_mc1_d2.parquet"
HISTORY_BCF = ROOT / "data_perp/artifacts/strict_r3_score_family_matched_mc1_canonical_policy_20260817_v7_bcf/predictions_bcf_mc1_d2.parquet"
HISTORY_FEATURE_CACHE = ROOT / "data_perp/artifacts/canonical_sr_e2_mc1_input_ablation_20260831_v2/target_free_15m_features.parquet"
DEFAULT_PREPARED = ROOT / "data_perp/artifacts/canonical_sr_e2_mc1_august_extension_inputs_20260831_v1"
DEFAULT_FROZEN_MAP = ROOT / "data_perp/artifacts/strict_r3_live_contract_dual30_august_replay_20260819_v1/dual_mapping_audit.parquet"
DEFAULT_OUT = ROOT / "data_perp/artifacts/canonical_sr_e2_mc1_input_ablation_august_20260831_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_aug_scores(path: Path, *, family: str) -> pd.DataFrame:
    columns = ["candidate_id", "__decision_ts__", "__symbol__", "side_name", *base.CORE]
    frame = pd.read_parquet(path, columns=columns)
    frame["candidate_id"] = frame["candidate_id"].astype(str)
    frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True, errors="raise")
    if frame["candidate_id"].duplicated().any():
        raise AssertionError(f"August {family} scores duplicate identity")
    if not frame.loc[:, list(base.CORE)].apply(pd.to_numeric, errors="coerce").notna().all(axis=1).all():
        raise AssertionError(f"August {family} source was not score-complete")
    frame["score_band"] = _score_bands(frame)
    return frame.loc[:, ["candidate_id", "__decision_ts__", "__symbol__", "side_name", *base.CORE, "score_band"]]


def _load_aug_labels(path: Path, candidate_ids: pd.Index) -> pd.DataFrame:
    fields = ["candidate_id", *base.POLICY_COLUMNS]
    raw = pd.read_parquet(path)
    required_source_fields = [field for field in fields if field != "policy_outcome_source"]
    missing = sorted(set(required_source_fields).difference(raw.columns))
    if missing:
        raise ValueError(f"August labels lack required policy fields: {missing}")
    labels = raw.loc[:, required_source_fields].copy()
    labels["candidate_id"] = labels["candidate_id"].astype(str)
    if labels["candidate_id"].duplicated().any():
        raise AssertionError("August labels duplicate candidate identity")
    labels["policy_label_available_ts"] = pd.to_datetime(labels["policy_label_available_ts"], utc=True, errors="raise")
    labels = labels.loc[labels["candidate_id"].isin(candidate_ids)].copy()
    # Explicit provenance is required by the shared candidate adapter but is
    # not a model input.  It only names this separately materialised extension.
    labels["policy_outcome_source"] = "august_same_parent_policy_15m_extension"
    return labels.loc[:, fields]


def _append_family(history: base.FamilySource, aug_scores: pd.DataFrame, aug_labels: pd.DataFrame, frozen: pd.Series) -> base.FamilySource:
    scores = pd.concat([history.scores, aug_scores], ignore_index=True)
    labels = pd.concat([history.labels, aug_labels], ignore_index=True)
    if scores["candidate_id"].duplicated().any() or labels["candidate_id"].duplicated().any():
        raise AssertionError("historical/August extension duplicated candidate identity")
    frozen_values = pd.concat([history.frozen_map, frozen.reset_index(drop=True)], ignore_index=True)
    if len(frozen_values) != len(scores) or not np.isfinite(frozen_values.to_numpy(float)).all():
        raise AssertionError("frozen-map extension is incomplete or non-finite")
    return base.FamilySource(name=history.name, scores=scores, labels=labels, frozen_map=frozen_values)


def _feature_cache(
    route: pd.DataFrame, *, out: Path, reuse_august_feature_cache: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    history = pd.read_parquet(HISTORY_FEATURE_CACHE)
    history["candidate_id"] = history["candidate_id"].astype(str)
    august_start = pd.Timestamp("2026-08-01T00:00:00Z")
    historical_route = route.loc[route["__decision_ts__"].lt(august_start), ["candidate_id"]]
    if not historical_route["candidate_id"].reset_index(drop=True).equals(history["candidate_id"].reset_index(drop=True)):
        raise AssertionError("reused February--July feature cache does not match retained score-union identity")
    august_route = route.loc[route["__decision_ts__"].ge(august_start)].copy().reset_index(drop=True)
    aug_cache = out / "august_target_free_15m_features.parquet"
    if reuse_august_feature_cache is not None:
        august_features = pd.read_parquet(reuse_august_feature_cache)
        if not august_route["candidate_id"].astype(str).reset_index(drop=True).equals(
            august_features["candidate_id"].astype(str).reset_index(drop=True)
        ):
            raise AssertionError("supplied August feature cache changed target-free identity")
        coverage = pd.DataFrame([{
            "status": "supplied_verified_cache",
            "rows": int(len(august_features)),
            "path": str(reuse_august_feature_cache),
            "sha256": _sha256(reuse_august_feature_cache),
        }])
    elif aug_cache.exists():
        august_features = pd.read_parquet(aug_cache)
        if not august_route["candidate_id"].astype(str).reset_index(drop=True).equals(
            august_features["candidate_id"].astype(str).reset_index(drop=True)
        ):
            raise AssertionError("reused August feature cache changed target-free identity")
        coverage = pd.DataFrame([{"status": "reused", "rows": int(len(august_features))}])
    else:
        # The shared feature helper caches every symbol inside one call.  That
        # is efficient for a small panel but can retain the full 164-symbol
        # history simultaneously.  Bounded groups leave the target-free
        # feature equation untouched while releasing each local bar cache
        # before the next group.
        parts: list[pd.DataFrame] = []
        grouped = list(august_route.groupby("symbol", sort=True))
        for offset in range(0, len(grouped), 8):
            batch = pd.concat([frame for _, frame in grouped[offset:offset + 8]], ignore_index=True)
            part = base._materialize_target_free_features(batch)
            parts.append(part)
            print(json.dumps({"event": "august_15m_feature_batch", "symbols_complete": min(offset + 8, len(grouped)), "symbols_total": len(grouped)}), flush=True)
        raw = pd.concat(parts, ignore_index=True)
        august_features = august_route.loc[:, ["candidate_id"]].merge(raw, on="candidate_id", how="left", validate="one_to_one")
        if len(august_features) != len(august_route) or august_features["candidate_id"].duplicated().any():
            raise AssertionError("bounded August feature materialisation changed target-free identity")
        aug_cache.parent.mkdir(parents=True, exist_ok=True)
        august_features.to_parquet(aug_cache, index=False, compression="zstd")
        coverage = august_features.groupby("feature_source_status", dropna=False, sort=True).size().rename("rows").reset_index()
    full = pd.concat([history, august_features], ignore_index=True)
    if not route["candidate_id"].reset_index(drop=True).equals(full["candidate_id"].astype(str).reset_index(drop=True)):
        raise AssertionError("full causal 15m feature cache changed score-union identity")
    full.to_parquet(out / "target_free_15m_features_feb_aug.parquet", index=False, compression="zstd")
    return full, coverage


def _monthly_from_decisions(decisions: pd.DataFrame, *, arm: str) -> pd.DataFrame:
    accepted = decisions.loc[decisions.get("accepted", pd.Series(index=decisions.index, dtype=bool)).fillna(False).astype(bool)].copy()
    if accepted.empty:
        return pd.DataFrame(columns=["arm", "month", "portfolio_accepted_trades", "net_ev_bps_per_trade", "net_sum_bps"])
    accepted["timestamp"] = pd.to_datetime(accepted["timestamp"], utc=True, errors="raise")
    accepted["month"] = accepted["timestamp"].dt.strftime("%Y-%m")
    accepted["net_bps"] = pd.to_numeric(accepted["position_net_return"], errors="coerce") * 10_000.0
    result = accepted.groupby("month", sort=True).agg(
        portfolio_accepted_trades=("net_bps", "size"), net_ev_bps_per_trade=("net_bps", "mean"), net_sum_bps=("net_bps", "sum"),
    ).reset_index()
    result.insert(0, "arm", arm)
    return result


def _replay(
    target_free: pd.DataFrame, outcome: pd.DataFrame, *, arm: str, out: Path, period: str,
) -> tuple[dict[str, object], pd.DataFrame]:
    metric, _ = base._replay(target_free, outcome, arm, out)
    metric["period"] = period
    decisions = pd.read_parquet(out / f"{arm}_portfolio_decisions.parquet")
    return metric, _monthly_from_decisions(decisions, arm=arm)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, default=DEFAULT_PREPARED)
    parser.add_argument("--policy-labels", type=Path, required=True)
    parser.add_argument("--frozen-map", type=Path, default=DEFAULT_FROZEN_MAP)
    parser.add_argument("--history-current", type=Path, default=HISTORY_CURRENT)
    parser.add_argument("--history-bcf", type=Path, default=HISTORY_BCF)
    parser.add_argument(
        "--replay-start", type=pd.Timestamp, default=pd.Timestamp("2026-08-01T00:00:00Z"),
        help="inclusive monthly replay boundary; score history remains strictly prior",
    )
    parser.add_argument(
        "--replay-end", type=pd.Timestamp, default=pd.Timestamp("2026-08-19T00:00:00Z"),
        help="exclusive monthly replay boundary",
    )
    parser.add_argument(
        "--skip-e2", action="store_true",
        help="avoid unused 15-minute E2 materialisation when no selected arm consumes E2",
    )
    parser.add_argument("--sr-root", type=Path, default=base.SR_ROOT)
    parser.add_argument("--profile-root", type=Path, help="optional 2025-trained causal profile/channel OOF head root")
    parser.add_argument("--anchor-root", type=Path, help="optional 2025-selected causal Anchor Discovery OOF head root")
    parser.add_argument("--anchor-variant", help="must equal the source's 2025-selected variant")
    parser.add_argument(
        "--reuse-august-feature-cache", type=Path,
        help="existing verified target-free August 15m feature cache; identity is rechecked before reuse",
    )
    parser.add_argument(
        "--component-ablation", action="store_true",
        help="add the causal S/R demotion-only and support-only decomposition arms",
    )
    parser.add_argument(
        "--only-arms",
        help="comma-separated subset of non-frozen arms; keeps a confirmation run bounded",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    prepared, policy_labels, frozen_map, out = (args.prepared.resolve(), args.policy_labels.resolve(), args.frozen_map.resolve(), args.out.resolve())
    history_current, history_bcf = args.history_current.resolve(), args.history_bcf.resolve()
    start, end = (pd.Timestamp(args.replay_start), pd.Timestamp(args.replay_end))
    start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
    end = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")
    if end <= start:
        raise ValueError("--replay-end must be after --replay-start")
    if out.exists():
        raise FileExistsError(f"immutable output exists: {out}")
    out.mkdir(parents=True)
    current_history = base._load_family(history_current, "current_v5")
    bcf_history = base._load_family(history_bcf, "bcf")
    current_aug = _load_aug_scores(prepared / "current_scores_core_complete.parquet", family="current_v5")
    bcf_aug = _load_aug_scores(prepared / "bcf_scores_core_complete.parquet", family="bcf")
    if not current_aug["candidate_id"].equals(bcf_aug["candidate_id"]):
        raise AssertionError("prepared August BCF/current score identities are not aligned")
    labels_aug = _load_aug_labels(policy_labels, pd.Index(current_aug["candidate_id"]))
    mapping = pd.read_parquet(frozen_map, columns=["candidate_id", "bcf_mc1_expected_bps", "current_v5_mc1_expected_bps", "bcf_mc1_available", "current_v5_mc1_available"])
    mapping["candidate_id"] = mapping["candidate_id"].astype(str)
    mapping = mapping.set_index("candidate_id").loc[current_aug["candidate_id"]].reset_index()
    if not (mapping["bcf_mc1_available"].fillna(False) & mapping["current_v5_mc1_available"].fillna(False)).all():
        raise AssertionError("prepared score-complete rows include an unavailable frozen map")
    current = _append_family(current_history, current_aug, labels_aug, pd.to_numeric(mapping["current_v5_mc1_expected_bps"], errors="raise"))
    bcf = _append_family(bcf_history, bcf_aug, labels_aug, pd.to_numeric(mapping["bcf_mc1_expected_bps"], errors="raise"))
    labels = base._candidate_labels(current, bcf)
    union = base._score_union(current, bcf)
    reuse_august_feature_cache = (
        args.reuse_august_feature_cache.resolve() if args.reuse_august_feature_cache else None
    )
    if args.skip_e2:
        e2 = union.loc[:, ["candidate_id"]].copy()
        e2[base.E2_OUTPUT] = np.nan
        e2[base.E2_AVAILABLE] = np.int8(0)
        e2_audit = pd.DataFrame([{"status": "skipped_unused"}])
        feature_coverage = pd.DataFrame([{"status": "skipped_unused", "rows": int(len(e2))}])
    else:
        feature_route = union.loc[
            union["__decision_ts__"].ge(base.FEATURE_START) & union["__decision_ts__"].lt(pd.Timestamp("2026-08-19T00:00:00Z"))
        ].copy().reset_index(drop=True)
        features, feature_coverage = _feature_cache(
            feature_route, out=out, reuse_august_feature_cache=reuse_august_feature_cache,
        )
        e2, e2_audit = base._prequential_e2(
            union.loc[union["__decision_ts__"].ge(base.FEATURE_START)].copy(), labels, features,
            start=base.FEATURE_START, end=pd.Timestamp("2026-08-19T00:00:00Z"),
        )
    sr_probe, sr_coverage = sr._merge_causal_sr(current.scores.copy(), args.sr_root)
    del sr_probe
    _current_scores, current_full = base._augment_family(current, args.sr_root, e2, profile_root=args.profile_root, anchor_root=args.anchor_root, anchor_variant=args.anchor_variant)
    _bcf_scores, bcf_full = base._augment_family(bcf, args.sr_root, e2, profile_root=args.profile_root, anchor_root=args.anchor_root, anchor_variant=args.anchor_variant)
    del _current_scores, _bcf_scores
    arms: dict[str, tuple[str, ...]] = {
        "C0_refit_core_postfeb": (),
        "C1_refit_core_plus_causal_sr": (*sr.SR_FEATURES, base.SR_AVAILABLE),
        "C2_refit_core_plus_15m_e2": (base.E2_OUTPUT, base.E2_AVAILABLE),
        "C3_refit_core_plus_causal_sr_15m_e2": (*sr.SR_FEATURES, base.SR_AVAILABLE, base.E2_OUTPUT, base.E2_AVAILABLE),
    }
    if args.component_ablation:
        arms.update({
            "C1a_refit_core_plus_sr_demotion": (*base.SR_DEMOTION_FEATURES, base.SR_AVAILABLE),
            "C1b_refit_core_plus_sr_support": (*base.SR_SUPPORT_FEATURES, base.SR_AVAILABLE),
        })
    if args.profile_root is not None:
        arms["C5_refit_core_plus_causal_sr_profile_geometry"] = (
            *sr.SR_FEATURES, base.SR_AVAILABLE,
            *base.PROFILE_GEOMETRY_HEADS, base.PROFILE_GEOMETRY_AVAILABLE,
        )
    if args.anchor_root is not None:
        arms["C6_refit_core_plus_causal_anchor"] = (*base.ANCHOR_ENTRY_HEADS, base.ANCHOR_AVAILABLE)
        arms["C7_refit_core_plus_causal_sr_anchor"] = (
            *sr.SR_FEATURES, base.SR_AVAILABLE, *base.ANCHOR_ENTRY_HEADS, base.ANCHOR_AVAILABLE,
        )
    if args.only_arms:
        requested = tuple(name.strip() for name in args.only_arms.split(",") if name.strip())
        unknown = set(requested).difference(arms)
        if unknown:
            raise ValueError(f"unknown --only-arms entries: {sorted(unknown)}")
        arms = {name: arms[name] for name in requested}
    metric_rows: list[dict[str, object]] = []
    monthly_rows: list[pd.DataFrame] = []
    folds: list[pd.DataFrame] = []
    frozen_tf, frozen_outcome = base._frozen_control(current, bcf, labels)
    frozen_tf = frozen_tf.loc[frozen_tf["__decision_ts__"].ge(start) & frozen_tf["__decision_ts__"].lt(end)].copy()
    frozen_outcome = frozen_outcome.loc[frozen_outcome["__decision_ts__"].ge(start) & frozen_outcome["__decision_ts__"].lt(end)].copy()
    frozen_tf.to_parquet(out / "C0_frozen_retained_target_free_admission.parquet", index=False, compression="zstd")
    period = f"{start:%Y-%m-%dT%H:%MZ}_to_{end:%Y-%m-%dT%H:%MZ}"
    metric, monthly = _replay(frozen_tf, frozen_outcome, arm="C0_frozen_retained", out=out, period=period)
    metric_rows.append(metric); monthly_rows.append(monthly)
    for arm, extras in arms.items():
        current_pred, current_audit = base._refit_family(current_full, family="current_v5", extras=extras, start=start, end=end)
        bcf_pred, bcf_audit = base._refit_family(bcf_full, family="bcf", extras=extras, start=start, end=end)
        current_pred.to_parquet(out / f"{arm}_current_mc1_predictions.parquet", index=False, compression="zstd")
        bcf_pred.to_parquet(out / f"{arm}_bcf_mc1_predictions.parquet", index=False, compression="zstd")
        target_free, outcome = base._combine_predictions(current_pred, bcf_pred, labels)
        target_free.to_parquet(out / f"{arm}_target_free_admission.parquet", index=False, compression="zstd")
        metric, monthly = _replay(target_free, outcome, arm=arm, out=out, period=period)
        metric_rows.append(metric); monthly_rows.append(monthly)
        audit = pd.concat([current_audit, bcf_audit], ignore_index=True); audit.insert(0, "arm", arm); folds.append(audit)
    summary = base._append_delta(pd.DataFrame(metric_rows), "C0_refit_core_postfeb")
    summary.to_csv(out / "portfolio_summary.csv", index=False)
    summary.to_parquet(out / "portfolio_summary.parquet", index=False, compression="zstd")
    pd.concat(monthly_rows, ignore_index=True).to_parquet(out / "monthly_metrics.parquet", index=False, compression="zstd")
    pd.concat(folds, ignore_index=True).to_parquet(out / "mc1_fold_audit.parquet", index=False, compression="zstd")
    feature_coverage.to_parquet(out / "august_target_free_15m_feature_coverage.parquet", index=False, compression="zstd")
    e2_audit.to_parquet(out / "e2_prequential_audit.parquet", index=False, compression="zstd")
    pd.read_parquet(prepared / "target_free_availability.parquet").to_parquet(out / "source_target_free_availability.parquet", index=False, compression="zstd")
    manifest = {
        "schema": "canonical_sr_e2_mc1_input_ablation_august_extension_v1",
        "scope": "offline partial-August matched MC1-input challenger; no live or canonical mutation and no exchange calls",
        "period": {"start": start.isoformat(), "end_exclusive": end.isoformat(), "coverage_note": "archived paired score source ends 2026-08-18 21:00 UTC"},
        "score_history": {"current": {"path": str(history_current), "sha256": _sha256(history_current)}, "bcf": {"path": str(history_bcf), "sha256": _sha256(history_bcf)}},
        "august_target_free_source": {"prepared": str(prepared), "manifest_sha256": _sha256(prepared / "run_manifest.json"), "frozen_map": {"path": str(frozen_map), "sha256": _sha256(frozen_map)}},
        "outcomes": {"path": str(policy_labels), "sha256": _sha256(policy_labels), "policy": "same 15-minute +1h source-aligned parent geometry as retained v7; cost 100 bps once"},
        "mc1": "family-specific absolute policy_net_bps HGB d2/80/.04/L2=20/min_leaf=100 seed=1729; 21d 10%-trimmed prior-resolved score-band shift",
        "admission": "BCF MC1 >= +50 AND current-v5 MC1 >= +50; BCF EV auction priority",
        "portfolio": "controlled global long-only 7x/10%-slot, 2-new, 8-concurrent, 80%-wallet; invalid outcomes excluded before capacity",
        "sr": {"path": str(args.sr_root), "manifest_sha256": _sha256(args.sr_root / "run_manifest.json")},
        "profile_geometry": None if args.profile_root is None else {
            "path": str(args.profile_root), "manifest_sha256": _sha256(args.profile_root / "run_manifest.json"),
            "fields": list(base.PROFILE_GEOMETRY_HEADS),
            "contract": "2025-trained causal profile/channel heads; optional availability remains a mapper field only",
        },
        "reused_august_feature_cache": None if reuse_august_feature_cache is None else {
            "path": str(reuse_august_feature_cache), "sha256": _sha256(reuse_august_feature_cache),
        },
        "e2": "skipped_unused" if args.skip_e2 else "prequential_target_free_15m_features",
        "arms": {arm: list(extras) for arm, extras in arms.items()},
        "status": "complete",
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"out": str(out), "rows": {"august_score_complete": int(len(current_aug)), "august_policy_labels": int(len(labels_aug))}}, sort_keys=True))


if __name__ == "__main__":
    main()
