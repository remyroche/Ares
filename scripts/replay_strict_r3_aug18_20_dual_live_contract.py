#!/usr/bin/env python3
"""Replay a frozen BCF/current dual-MC1 live route over a bounded period.

The producer deliberately separates score/admission from outcome materialisation:

    frozen target-free scores -> strict prior-resolved MC1 maps -> dual route
    -> exact 1m parent-policy replay where the 12h path is complete

Rows whose exact 12-hour path has not yet elapsed at ``--as-of`` are retained
as ``pending_horizon``.  They are never converted to a zero return and never
enter realised portfolio metrics.  This is offline research only: it has no
exchange client and no order authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.strict_r3_bcf_mc1_mapper import BCFMC1D2Bundle
from extreme_price_movements.strict_r3_mc1_mapper import MC1D2Bundle, _robust_mean, score_bands
from scripts.replay_strict_r3_bcf_exact5m_1m import (
    _exact_labels,
    _policy,
    _portfolio_candidates,
    _run_portfolio,
)


CURRENT_FEATURES = (
    "final_score",
    "base_rank42",
    "conditional_consensus_rank",
    "upstream",
    "ordinary_shadow_consensus_rank",
    "correctness_rank",
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc(value: object) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _read(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path).copy()
    frame["candidate_id"] = frame["candidate_id"].astype(str)
    frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True)
    if frame["candidate_id"].duplicated().any():
        raise ValueError(f"duplicate candidate IDs in {path}")
    return frame


def _read_selected(path: Path, columns: list[str]) -> pd.DataFrame:
    """Read the minimum declared schema from a large reserve panel."""
    frame = pd.read_parquet(path, columns=columns).copy()
    frame["candidate_id"] = frame["candidate_id"].astype(str)
    if "__decision_ts__" in frame:
        frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True)
    if frame["candidate_id"].duplicated().any():
        raise ValueError(f"duplicate candidate IDs in {path}")
    return frame


def _prepend_reserve(
    frame: pd.DataFrame,
    *,
    reserve: Path | None,
    columns: list[str],
) -> pd.DataFrame:
    """Append an immutable prior reserve, preferring the main panel on ID."""
    if reserve is None:
        return frame
    prior = _read_selected(reserve, columns)
    return pd.concat([prior, frame], ignore_index=True, sort=False).drop_duplicates(
        "candidate_id", keep="last"
    ).copy()


def _labelled(scores: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    out = scores.merge(
        labels.loc[:, [
            "candidate_id", "policy_path_valid", "policy_net_bps",
            "policy_label_available_ts",
        ]],
        on="candidate_id", how="left", validate="one_to_one",
    )
    out["policy_path_valid"] = out["policy_path_valid"].fillna(False).astype(bool)
    out["policy_label_available_ts"] = pd.to_datetime(
        out["policy_label_available_ts"], utc=True, errors="coerce"
    )
    return out


def _history(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.loc[:, [
        "candidate_id", "__decision_ts__", "final_score", "policy_path_valid",
        "policy_net_bps", "policy_label_available_ts",
    ]].copy()
    out = out.loc[
        out["policy_path_valid"].fillna(False).astype(bool)
        & pd.to_numeric(out["policy_net_bps"], errors="coerce").notna()
        & out["policy_label_available_ts"].notna()
    ].copy()
    out["score_band"] = score_bands(out)
    return out


def _map(
    held: pd.DataFrame,
    history: pd.DataFrame,
    *,
    bundle: Any,
    prefix: str,
    threshold: float,
) -> pd.DataFrame:
    """Apply the sealed model plus a strictly prior-resolved 21-day shift."""
    features = list(bundle.manifest["features_ordered"])
    curve = np.asarray(bundle.payload["structural_curve_bps"], dtype=float)
    labels_ns = history["policy_label_available_ts"].astype("int64").to_numpy()
    decisions_ns = history["__decision_ts__"].astype("int64").to_numpy()
    residual = (
        pd.to_numeric(history["policy_net_bps"], errors="raise").to_numpy(float)
        - curve[history["score_band"].to_numpy(int)]
    )
    window = int(pd.Timedelta(days=21).value)
    pieces: list[pd.DataFrame] = []
    for decision, group in held.groupby("__decision_ts__", sort=True):
        now = pd.Timestamp(decision).value
        # Strict inequality intentionally excludes labels that resolve at this
        # decision boundary.
        shift = _robust_mean(
            residual[(labels_ns < now) & (decisions_ns >= now - window)], trim=0.10
        )
        matrix = group.loc[:, features].apply(pd.to_numeric, errors="coerce")
        available = np.isfinite(matrix.to_numpy(float)).all(axis=1) & np.isfinite(shift)
        value = np.full(len(group), np.nan, dtype=float)
        if available.any():
            value[available] = bundle.payload["model"].predict(
                matrix.loc[available, features]
            ) + shift
        pieces.append(pd.DataFrame({
            "candidate_id": group["candidate_id"].astype(str).to_numpy(),
            f"{prefix}_expected_bps": value,
            f"{prefix}_recent_shift_bps": shift,
            f"{prefix}_available": available,
            f"{prefix}_admitted_ge_30bps": available & np.isfinite(value) & (value >= threshold),
        }))
    return pd.concat(pieces, ignore_index=True)


def _append_hourly_suffix(
    frame: pd.DataFrame,
    *,
    glob_pattern: str,
    relative: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """Append point-in-time current-hour receipts absent from a full prefix."""
    existing = set(frame["candidate_id"].astype(str))
    pieces = [frame]
    for root in sorted(ROOT.glob(glob_pattern)):
        path = root / relative
        if not path.exists():
            continue
        item = _read(path)
        item = item.loc[item["__decision_ts__"].between(start, end, inclusive="both")]
        item = item.loc[~item["candidate_id"].isin(existing)].copy()
        if not item.empty:
            pieces.append(item)
            existing.update(item["candidate_id"].astype(str))
    return pd.concat(pieces, ignore_index=True)


def _daily(admission: pd.DataFrame, decisions: pd.DataFrame) -> pd.DataFrame:
    result = admission.copy()
    result["day"] = result["__decision_ts__"].dt.floor("D")
    result = result.groupby("day", as_index=False).agg(
        scored_candidates=("candidate_id", "size"),
        dual_admitted=("dual_admitted", "sum"),
        exact_resolved=(
            "outcome_status",
            lambda x: int(x.astype(str).isin(["resolved", "resolved_15m_proxy"]).sum()),
        ),
        exact_pending=("outcome_status", lambda x: int((x == "pending_horizon").sum())),
        exact_invalid=("outcome_status", lambda x: int((x == "invalid_source").sum())),
    )
    if decisions.empty:
        result["portfolio_accepted"] = 0
        result["net_ev_bps_per_trade"] = np.nan
        result["net_sum_bps"] = 0.0
        return result
    accepted = decisions.loc[decisions["accepted"].fillna(False).astype(bool)].copy()
    accepted["day"] = pd.to_datetime(accepted["timestamp"], utc=True).dt.floor("D")
    metrics = accepted.groupby("day", as_index=False).agg(
        portfolio_accepted=("accepted", "size"),
        net_ev_bps_per_trade=("position_net_return", lambda x: float(x.mean() * 10_000.0)),
        net_sum_bps=("position_net_return", lambda x: float(x.sum() * 10_000.0)),
    )
    return result.merge(metrics, on="day", how="left").fillna({"portfolio_accepted": 0, "net_sum_bps": 0.0})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current-predictions", type=Path, required=True)
    parser.add_argument("--bcf-predictions", type=Path, required=True)
    parser.add_argument("--feature-panel", type=Path, required=True)
    parser.add_argument("--resolved-labels", type=Path, required=True)
    parser.add_argument(
        "--current-score-reserve", type=Path,
        help="Immutable prior current-score reserve for 21-day MC1 warm-up.",
    )
    parser.add_argument(
        "--bcf-score-reserve", type=Path,
        help="Immutable prior BCF-score reserve for 21-day MC1 warm-up.",
    )
    parser.add_argument(
        "--bcf-reserve-context", type=Path,
        help="Current-score context used to complete BCF reserve MC1 fields.",
    )
    parser.add_argument(
        "--feature-reserve", type=Path,
        help="Immutable prior candidate identity/symbol reserve for BCF history.",
    )
    parser.add_argument(
        "--label-reserve", type=Path,
        help="Immutable prior resolved policy-label reserve for MC1 warm-up.",
    )
    parser.add_argument(
        "--reconstruct-missing-base-route-top30",
        action="store_true",
        help=(
            "Recover an absent timestamp-local top-30%% route only after exact "
            "validation against persisted route timestamps."
        ),
    )
    parser.add_argument("--current-mc1-bundle", type=Path, required=True)
    parser.add_argument("--bcf-mc1-bundle", type=Path, required=True)
    parser.add_argument("--policy-json", type=Path, required=True)
    parser.add_argument(
        "--atr-source",
        choices=("canonical_hourly", "canonical_15m_aggregated", "minute_aggregated_100h"),
        default="canonical_hourly",
        help="Causal ATR source for exact-one-minute policy outcomes only.",
    )
    parser.add_argument(
        "--policy-proxy-labels", type=Path,
        help=(
            "Optional canonical 15-minute parent-policy labels for an explicitly "
            "labelled outcome fallback when exact 1m coverage is incomplete."
        ),
    )
    parser.add_argument("--data-root", type=Path, default=Path("data_perp"))
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True, help="inclusive decision hour")
    parser.add_argument("--as-of", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    start, end, as_of = _utc(args.start), _utc(args.end), _utc(args.as_of)
    if end < start:
        raise ValueError("end precedes start")
    if args.out_dir.exists():
        raise FileExistsError(f"immutable output exists: {args.out_dir}")
    args.out_dir.mkdir(parents=True)

    # The final 21-day map needs earlier August scores, not merely the report
    # period.  The panel is already target-free and frozen before outcome joins.
    history_start = start - pd.Timedelta(days=21)
    features = _read(args.feature_panel)
    features = _prepend_reserve(
        features,
        reserve=args.feature_reserve,
        columns=["candidate_id", "__decision_ts__", "__symbol__"],
    )
    features = features.loc[features["__decision_ts__"].between(history_start, end, inclusive="both")].copy()
    features = _append_hourly_suffix(
        features,
        glob_pattern="data_perp/artifacts/strict_r3_successor_v104_live_*",
        relative="features/canonical120_features.parquet",
        start=history_start,
        end=end,
    )
    current = _read(args.current_predictions)
    current = _prepend_reserve(
        current,
        reserve=args.current_score_reserve,
        columns=["candidate_id", "__decision_ts__", *CURRENT_FEATURES],
    )
    current = current.loc[current["__decision_ts__"].between(history_start, end, inclusive="both")].copy()
    current = _append_hourly_suffix(
        current,
        glob_pattern="data_perp/artifacts/strict_r3_successor_v104_live_*",
        relative="cycle/score/predictions.parquet",
        start=history_start,
        end=end,
    )
    route_reconstruction: dict[str, Any] = {"enabled": False}
    if args.reconstruct_missing_base_route_top30:
        route_name = "base_route_timestamp_top30"
        if route_name not in current or "base_score" not in current:
            raise ValueError("cannot reconstruct base route without base_score and route field")
        persisted = current[route_name].fillna(False).astype(bool)
        derived = current.groupby("__decision_ts__")["base_score"].rank(
            method="first", pct=True, ascending=True
        ).gt(0.70)
        group_has_persisted = persisted.groupby(current["__decision_ts__"]).transform("any")
        validation = group_has_persisted.to_numpy(bool)
        mismatches = int((persisted.loc[validation] != derived.loc[validation]).sum())
        if mismatches:
            raise ValueError(
                f"top-30 route reconstruction failed persisted-route validation: {mismatches} mismatches"
            )
        reconstructed = ~group_has_persisted
        current["base_route_reconstruction"] = np.where(
            reconstructed, "reconstructed_base_score_rank_gt_70pct", "persisted"
        )
        current[route_name] = np.where(reconstructed, derived, persisted)
        route_reconstruction = {
            "enabled": True,
            "definition": "timestamp-local base_score rank(method=first, pct=True) > 0.70",
            "persisted_validation_rows": int(validation.sum()),
            "persisted_validation_mismatches": mismatches,
            "reconstructed_rows": int(reconstructed.sum()),
            "reconstructed_timestamps": int(current.loc[reconstructed, "__decision_ts__"].nunique()),
        }
    bcf = _read(args.bcf_predictions)
    if args.bcf_score_reserve is not None:
        bcf_reserve = _read_selected(
            args.bcf_score_reserve,
            ["candidate_id", "__decision_ts__", "final_score", "base_rank42", "upstream"],
        )
        if args.bcf_reserve_context is not None:
            context = _read_selected(
                args.bcf_reserve_context,
                [
                    "candidate_id", "conditional_consensus_rank",
                    "ordinary_shadow_consensus_rank", "correctness_rank",
                ],
            )
            bcf_reserve = bcf_reserve.merge(
                context,
                on="candidate_id",
                how="left",
                validate="one_to_one",
            )
        bcf = pd.concat([bcf_reserve, bcf], ignore_index=True, sort=False).drop_duplicates(
            "candidate_id", keep="last"
        ).copy()
    bcf = bcf.loc[bcf["__decision_ts__"].between(history_start, end, inclusive="both")].copy()
    bcf = _append_hourly_suffix(
        bcf,
        glob_pattern="data_perp/artifacts/strict_r3_successor_v104_live_*",
        relative="cycle/bcf_score/predictions.parquet",
        start=history_start,
        end=end,
    )
    bcf = bcf.merge(
        features.loc[:, ["candidate_id", "__symbol__"]], on="candidate_id", how="inner", validate="one_to_one"
    )
    label_required = {"candidate_id", "policy_path_valid", "policy_net_bps", "policy_label_available_ts"}
    labels = _read(args.resolved_labels)
    labels = _prepend_reserve(
        labels,
        reserve=args.label_reserve,
        columns=sorted(label_required | {"__decision_ts__"}),
    )
    if missing := sorted(label_required.difference(labels.columns)):
        raise ValueError(f"resolved labels lack {missing}")
    labels = labels.loc[:, list(label_required)].drop_duplicates("candidate_id", keep="last")
    labels["policy_label_available_ts"] = pd.to_datetime(labels["policy_label_available_ts"], utc=True)

    current_l = _labelled(current, labels)
    bcf_l = _labelled(bcf, labels)
    current_bundle = MC1D2Bundle.load(args.current_mc1_bundle)
    bcf_bundle = BCFMC1D2Bundle.load(args.bcf_mc1_bundle)
    current_map = _map(
        current_l, _history(current_l), bundle=current_bundle,
        prefix="current_v5_mc1", threshold=30.0,
    )
    bcf_map = _map(
        bcf_l, _history(bcf_l), bundle=bcf_bundle,
        prefix="bcf_mc1", threshold=30.0,
    )
    held = current_l.loc[current_l["__decision_ts__"].between(start, end, inclusive="both")].copy()
    held = held.merge(current_map, on="candidate_id", how="left", validate="one_to_one")
    held = held.merge(bcf_map, on="candidate_id", how="left", validate="one_to_one")
    complete = held.get("frozen_base_contract_complete", pd.Series(False, index=held.index)).fillna(False).astype(bool)
    routed = held.get("base_route_timestamp_top30", pd.Series(False, index=held.index)).fillna(False).astype(bool)
    held["dual_admitted"] = (
        complete & routed
        & held["current_v5_mc1_admitted_ge_30bps"].fillna(False).astype(bool)
        & held["bcf_mc1_admitted_ge_30bps"].fillna(False).astype(bool)
    )
    held["dual_auction_priority_bps"] = held["bcf_mc1_expected_bps"]
    selected = held.loc[held["dual_admitted"]].copy()
    selected["__ts__"] = selected["__decision_ts__"] - pd.Timedelta(hours=1)
    selected.to_parquet(args.out_dir / "target_free_dual_admission.parquet", index=False, compression="zstd")

    # Outcome materialisation begins only after the target-free selection is
    # immutable.  It is intentionally limited to the selected identities.
    policy = _policy(args.policy_json)
    exact = _exact_labels(
        # The shared exact-label helper owns the identity join itself.  Its
        # request side must therefore contain only the timestamp/symbol pair.
        selected.loc[:, ["__decision_ts__", "__symbol__"]],
        selected.loc[:, ["candidate_id", "__decision_ts__", "__symbol__"]],
        data_root=args.data_root,
        policy=policy,
        atr_source=str(args.atr_source),
        entry_delay_minutes=5,
    ) if len(selected) else pd.DataFrame()
    if len(selected):
        exact["horizon_end_ts"] = pd.to_datetime(exact["delayed_entry_ts"], utc=True) + pd.Timedelta(hours=12)
        exact["outcome_status"] = np.where(
            exact["policy_path_valid"].fillna(False).astype(bool),
            "resolved",
            np.where(exact["horizon_end_ts"].gt(as_of), "pending_horizon", "invalid_source"),
        )
        exact.to_parquet(args.out_dir / "exact1m_outcomes.parquet", index=False, compression="zstd")
        held = held.merge(exact.loc[:, ["candidate_id", "outcome_status"]], on="candidate_id", how="left")
    else:
        held["outcome_status"] = pd.Series(dtype="string")
    held["outcome_status"] = held["outcome_status"].fillna("not_dual_admitted")

    # When supplied, the canonical 15m parent policy is a transparent fallback
    # only for realised-outcome reporting.  It never feeds mapping, routing,
    # or auction priority.  Exact 1m takes precedence whenever it exists.
    realised_labels = exact.copy()
    realised_delay_minutes = 5
    outcome_contract_name = "exact_1m_tplus5_parent_policy"
    if args.policy_proxy_labels is not None:
        proxy = _read(args.policy_proxy_labels)
        required_proxy = {
            "candidate_id", "__decision_ts__", "__symbol__", "policy_path_valid",
            "policy_gross_bps", "policy_net_bps", "policy_exit_bar_15m",
            "policy_entry_price", "policy_exit_price", "policy_exit_reason",
            "policy_label_available_ts", "policy_cost_bps",
        }
        if missing := sorted(required_proxy.difference(proxy.columns)):
            raise ValueError(f"policy proxy labels lack {missing}")
        proxy = proxy.loc[proxy["candidate_id"].isin(selected["candidate_id"])].copy()
        proxy["policy_label_available_ts"] = pd.to_datetime(
            proxy["policy_label_available_ts"], utc=True
        )
        proxy["delayed_entry_ts"] = proxy["__decision_ts__"]
        proxy["policy_outcome_source"] = "canonical_15m_parent_policy_proxy"
        proxy["policy_exit_minutes"] = (
            pd.to_numeric(proxy["policy_exit_bar_15m"], errors="coerce") + 1.0
        ) * 15.0
        proxy["policy_exit_timestamp"] = (
            proxy["__decision_ts__"] + pd.to_timedelta(proxy["policy_exit_minutes"], unit="min")
        )
        proxy.loc[~proxy["policy_path_valid"].fillna(False).astype(bool), "policy_exit_timestamp"] = pd.NaT
        proxy["outcome_status"] = np.where(
            proxy["policy_path_valid"].fillna(False).astype(bool)
            & proxy["policy_label_available_ts"].le(as_of),
            "resolved_15m_proxy",
            np.where(
                (proxy["__decision_ts__"] + pd.Timedelta(hours=12)).gt(as_of),
                "pending_horizon", "invalid_source",
            ),
        )
        # The fallback is deliberately declared at the row level as well as
        # in the manifest.  It replaces only missing exact outcomes.
        if len(exact):
            exact_valid = exact.loc[exact["policy_path_valid"].fillna(False).astype(bool)].copy()
            proxy = proxy.loc[~proxy["candidate_id"].isin(exact_valid["candidate_id"])].copy()
            realised_labels = pd.concat([exact_valid, proxy], ignore_index=True, sort=False)
        else:
            realised_labels = proxy
        status = realised_labels.loc[:, ["candidate_id", "outcome_status"]].drop_duplicates("candidate_id")
        held = held.drop(columns=["outcome_status"], errors="ignore").merge(status, on="candidate_id", how="left")
        held["outcome_status"] = held["outcome_status"].fillna("not_dual_admitted")
        realised_delay_minutes = 0
        outcome_contract_name = "canonical_15m_parent_policy_proxy"

    # Realised portfolio metrics have no outcome/survivorship selection:
    # selected but unresolved rows remain in the audit above and are withheld
    # from realised arithmetic only.
    decisions = pd.DataFrame()
    equity = pd.DataFrame()
    metrics: dict[str, Any] = {}
    if len(selected) and len(realised_labels):
        resolved_ids = set(realised_labels.loc[
            realised_labels["outcome_status"].astype(str).isin(["resolved", "resolved_15m_proxy"]),
            "candidate_id",
        ].astype(str))
        resolved_labels = realised_labels.loc[realised_labels["candidate_id"].astype(str).isin(resolved_ids)].copy()
        bcf_eval = selected.loc[selected["candidate_id"].astype(str).isin(resolved_ids)].copy()
        current_eval = bcf_eval.copy()
        bcf_eval["mc1_expected_bps"] = bcf_eval["bcf_mc1_expected_bps"]
        current_eval["mc1_expected_bps"] = current_eval["current_v5_mc1_expected_bps"]
        candidates = _portfolio_candidates(
            resolved_labels, bcf_eval, current_eval, threshold_bps=30.0,
            entry_delay_minutes=realised_delay_minutes,
        )
        if not candidates.empty:
            decisions, equity, metrics = _run_portfolio(candidates)
            candidates.to_parquet(args.out_dir / "resolved_portfolio_candidates.parquet", index=False, compression="zstd")
            decisions.to_parquet(args.out_dir / "resolved_portfolio_decisions.parquet", index=False, compression="zstd")
            equity.to_parquet(args.out_dir / "resolved_portfolio_equity.parquet", index=False, compression="zstd")
    held.to_parquet(args.out_dir / "score_admission_outcome_ledger.parquet", index=False, compression="zstd")
    daily = _daily(held, decisions)
    daily.to_parquet(args.out_dir / "daily_metrics_and_pending.parquet", index=False, compression="zstd")

    manifest = {
        "schema": "strict_r3_aug18_20_dual_live_contract_replay_v1",
        "purpose": "offline target-free dual-route replay with explicit incomplete-horizon rows",
        "exchange_calls": 0,
        "order_submission_enabled": False,
        "score_contract": {
            "base_route": "timestamp-local top 30% current-v5 base route",
            "base_route_reconstruction": route_reconstruction,
            "admission": "BCF MC1 >= 30 bps AND current-v5 MC1 >= 30 bps",
            "auction_priority": "BCF MC1 expected bps",
            "shift": "strictly prior-resolved 21-day, 10% trimmed residual mean",
        },
        "outcome_contract": {
            "entry": (
                "Kraken Futures exact 1m open at decision +5 minutes"
                if outcome_contract_name.startswith("exact") else
                "canonical 15m decision-open proxy"
            ),
            "exit": "frozen SimplePolicyOptimiser parent (no Adaptive Exit V1 historical overlay)",
            "source": outcome_contract_name,
            "exact_atr_source": str(args.atr_source),
            "cost": "100 bps exactly once",
            "unresolved": "retained as pending_horizon and excluded from realised metrics",
        },
        "range": {"start": start.isoformat(), "end_inclusive": end.isoformat(), "as_of": as_of.isoformat()},
        "counts": {
            "current_scored": int(len(held)),
            "dual_admitted": int(held["dual_admitted"].sum()),
            "resolved": int(
                held["outcome_status"].astype(str).isin(["resolved", "resolved_15m_proxy"]).sum()
            ),
            "pending_horizon": int((held["outcome_status"] == "pending_horizon").sum()),
            "invalid_source": int((held["outcome_status"] == "invalid_source").sum()),
            "portfolio_accepted_resolved": int(decisions.get("accepted", pd.Series(dtype=bool)).fillna(False).sum()),
        },
        "realised_portfolio_metrics": metrics,
        "inputs": {str(p): _sha(p) for p in [
            args.current_predictions, args.bcf_predictions, args.feature_panel,
            args.resolved_labels, args.policy_json,
            args.current_mc1_bundle / "run_manifest.json", args.bcf_mc1_bundle / "run_manifest.json",
            *[
                path for path in [
                    args.current_score_reserve, args.bcf_score_reserve,
                    args.bcf_reserve_context, args.feature_reserve, args.label_reserve,
                ] if path is not None
            ],
        ]},
    }
    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps({"event": "complete", "out_dir": str(args.out_dir), **manifest["counts"]}, sort_keys=True))


if __name__ == "__main__":
    main()
