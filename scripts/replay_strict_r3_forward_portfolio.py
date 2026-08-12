#!/usr/bin/env python3
"""Apply canonical causal EV admission, then the global portfolio auction."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.portfolio_policy_replay import normalise_candidate_table  # noqa: E402
from extreme_price_movements.strict_r3_canonical_current import (  # noqa: E402
    OptimizedPolicyContract,
    SCHEMA as CURRENT_SCHEMA,
    apply_current_admission_by_geometry,
)
from extreme_price_movements.strict_r3_n5_canonical import (  # noqa: E402
    load_canonical_n5_bundle,
    score_canonical_n5_bundle,
)
from extreme_price_movements.stage_i_causal_admission import Causal21dAdmissionSpec  # noqa: E402
from extreme_price_movements.strict_r3_canonical_v2 import (  # noqa: E402
    SCHEMA as LEGACY_SCHEMA,
    apply_schema_v2_admission,
    require_single_geometry_hash,
)
from extreme_price_movements.strict_r3_cell_day_admission import (  # noqa: E402
    CELL_DAY_TRIM_15_CALIBRATION_MODE,
)
from extreme_price_movements.strict_r3_cell_day_trust import (  # noqa: E402
    load_cell_day_residual_trust_bundle,
)
from scripts.replay_strict_r3_policy_portfolio_2025_2026 import _run  # noqa: E402


# Canonical admission already maps every candidate into common net-bps space.
# The portfolio auction therefore needs no second, outcome-fitted EV curve.
# This frozen curve preserves the rank-surplus priority formula while making
# it impossible for held policy outcomes to affect candidate ordering.
CAUSAL_AUCTION_CURVE = {
    "schema": "monotone_ev_curve_v1",
    "x": [0.0, 1.0],
    "y": [0.0, 1.0],
    "ev_span": 1.0,
    "n_rows": 0,
    "source": "fixed_after_causal_expected_net_mapping",
}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _attach_producer_lineage(
    ledger: pd.DataFrame,
    lineage_path: Path | None,
) -> pd.DataFrame:
    """Require exact conversion × upstream provenance for current-v5 replay."""
    if "upstream_bundle_sha256" in ledger and "ev_score_family_id" in ledger:
        return ledger
    if lineage_path is None:
        raise ValueError(
            "current-v5 replay requires --producer-lineage when its OOF ledger "
            "does not persist per-row upstream producer hashes",
        )
    lineage = pd.read_parquet(lineage_path)
    required = {
        "candidate_id", "__decision_ts__", "conversion_bundle_sha256",
        "geometry_bundle_sha256", "upstream_bundle_sha256",
        "ev_score_family_id",
    }
    missing = sorted(required.difference(lineage.columns))
    if missing:
        raise ValueError(f"producer lineage lacks {missing}")
    if lineage["candidate_id"].duplicated().any():
        raise ValueError("producer lineage has duplicate candidate IDs")
    merged = ledger.merge(
        lineage.loc[:, sorted(required)], on="candidate_id", how="left",
        validate="one_to_one", suffixes=("", "__lineage"),
    )
    if merged["upstream_bundle_sha256"].isna().any():
        raise ValueError("producer lineage does not cover the entire scored ledger")
    for column in (
        "__decision_ts__", "conversion_bundle_sha256", "geometry_bundle_sha256",
    ):
        source = merged[column]
        sidecar = merged[f"{column}__lineage"]
        if not source.astype(str).eq(sidecar.astype(str)).all():
            raise ValueError(f"producer lineage conflicts on {column}")
        merged = merged.drop(columns=f"{column}__lineage")
    if "ev_score_family_id" in ledger:
        if not merged["ev_score_family_id"].astype(str).eq(
            merged["ev_score_family_id__lineage"].astype(str),
        ).all():
            raise ValueError("producer lineage conflicts on ev_score_family_id")
        merged = merged.drop(columns="ev_score_family_id__lineage")
    return merged


def _resolve_policy(
    schema: str, policy_json: Path | None,
) -> tuple[dict[str, float], str, object]:
    payload = json.loads(policy_json.read_text()) if policy_json else None
    if payload is not None:
        values = {
            key: float(payload["winner"][key])
            for key in ("sl_mult", "trailing_activation_mult", "fixed_trailing_gap_mult")
        }
        engine = payload.get("engine", "extreme_price_movements.simple_policy_optimiser")
        selection_period = payload.get("development_period")
        if schema == "current-v5":
            canonical = OptimizedPolicyContract()
            expected = {
                "sl_mult": canonical.stop_loss_atr,
                "trailing_activation_mult": canonical.trailing_activation_atr,
                "fixed_trailing_gap_mult": canonical.trailing_giveback_atr,
            }
            if any(not np.isclose(values[key], value) for key, value in expected.items()):
                raise ValueError(
                    "current-v5 outcomes must use the frozen pre-2025 "
                    "SimplePolicyOptimiser winner; retrain for another policy"
                )
        return values, str(engine), selection_period
    if schema == "current-v5":
        canonical = OptimizedPolicyContract()
        return (
            {
                "sl_mult": canonical.stop_loss_atr,
                "trailing_activation_mult": canonical.trailing_activation_atr,
                "fixed_trailing_gap_mult": canonical.trailing_giveback_atr,
            },
            canonical.source,
            "strict-prequential pre-2025 development only",
        )
    return (
        {
            "sl_mult": 3.0,
            "trailing_activation_mult": 0.5,
            "fixed_trailing_gap_mult": 0.25,
        },
        "frozen canonical SimplePolicyOptimiser mechanics",
        None,
    )


def _auction_candidates(
    frame: pd.DataFrame, *, strategy_prefix: str = "strict_r3_current_v5",
) -> pd.DataFrame:
    posterior_mode = "trust_posterior_admitted_ge_50bps" in frame
    admission_field = (
        "trust_posterior_admitted_ge_50bps"
        if posterior_mode else "causal_21d_side_admitted_ge_50bps"
    )
    expected_field = (
        "trust_posterior_expected_bps"
        if posterior_mode else "causal_21d_side_expected_net_bps"
    )
    admitted = frame.loc[
        frame[admission_field].fillna(False).astype(bool)
        & np.isfinite(pd.to_numeric(frame[expected_field], errors="coerce"))
    ].copy()
    # Admission itself is always determined by the causal EV map.  A bounded
    # research adjustment may only refine auction ordering after that gate;
    # it cannot add a candidate or rewrite the mapped EV retained for audit.
    adjustment = pd.Series(0.0, index=admitted.index)
    if not posterior_mode:
        adjustment = pd.to_numeric(
            admitted.get(
                "auction_rank_adjustment_bps",
                pd.Series(0.0, index=admitted.index),
            ),
            errors="coerce",
        ).fillna(0.0)
    admitted["auction_expected_net_bps"] = pd.to_numeric(
        admitted[expected_field], errors="coerce",
    ) + adjustment
    admitted = admitted.sort_values(
        ["__decision_ts__", "auction_expected_net_bps", "candidate_id"],
        ascending=[True, False, True], kind="stable",
    )
    admitted["auction_rank"] = admitted.groupby("__decision_ts__", sort=False)[
        "auction_expected_net_bps"
    ].rank(pct=True, method="average")
    entry_ts = pd.to_datetime(admitted["__decision_ts__"], utc=True)
    outcome_available = (
        admitted["policy_path_valid"].fillna(False).astype(bool)
        & np.isfinite(pd.to_numeric(admitted["policy_net_bps"], errors="coerce"))
        & np.isfinite(pd.to_numeric(admitted["policy_gross_bps"], errors="coerce"))
        & np.isfinite(pd.to_numeric(admitted["policy_exit_bar_15m"], errors="coerce"))
    )
    # Outcome availability is evaluation-only information.  An admitted row
    # without a realised path must still consume a portfolio slot, otherwise a
    # replay can replace it with a lower-ranked candidate using future
    # knowledge.  Reserve that slot to the H12 timeout and keep a separate
    # provenance flag; the neutral placeholder is never an economic label.
    exit_bar = pd.to_numeric(
        admitted["policy_exit_bar_15m"], errors="coerce",
    ).where(outcome_available, 47).astype(int)
    exit_ts = entry_ts + pd.to_timedelta(
        exit_bar.add(1) * 15,
        unit="min",
    )
    side = admitted["side_name"].astype(str).str.lower()
    strategy = strategy_prefix + "_" + side
    output = pd.DataFrame({
        "timestamp": entry_ts,
        "symbol": admitted["__symbol__"].astype(str),
        "side": side,
        "strategy_id": strategy,
        "policy_archetype": strategy,
        "normalized_rank_score": admitted["auction_rank"].to_numpy(float),
        "strategy_rank_pct": admitted["auction_rank"].to_numpy(float),
        "base_strategy_threshold": 0.0,
        # The monotone EV map is intentionally binned, so multiple candidates
        # can share one mapped-bps value.  A producer-local final CDF may be
        # supplied solely as a deterministic *secondary* ordering inside that
        # exact EV tie.  It never changes admission, the mapped expected bps,
        # rank-based sizing or the portfolio-priority value.
        "calibrated_score": pd.to_numeric(
            admitted.get("auction_tie_break_score", admitted["auction_rank"]),
            errors="coerce",
        ).fillna(admitted["auction_rank"]).to_numpy(float),
        "entry_price": pd.to_numeric(
            admitted["policy_entry_price"], errors="coerce",
        ).where(outcome_available, 1.0),
        "exit_timestamp": exit_ts,
        "exit_price": pd.to_numeric(
            admitted["policy_exit_price"], errors="coerce",
        ).where(outcome_available, 1.0),
        "net_return": pd.to_numeric(
            admitted["policy_net_bps"], errors="coerce",
        ).where(outcome_available, 0.0) / 10_000.0,
        "gross_return": pd.to_numeric(
            admitted["policy_gross_bps"], errors="coerce",
        ).where(outcome_available, 0.0) / 10_000.0,
        "holding_bars": exit_bar.add(1),
        "simple_policy_exit_reason": admitted["policy_exit_reason"].where(
            outcome_available, "OUTCOME_UNAVAILABLE_RESERVED_H12",
        ).astype(str),
        "fees_bps": 100.0, "slippage_bps": 0.0, "expected_friction_bps": 100.0,
        "price_gap_bps": 0.0, "liquidity_capacity_weight": 1.0,
        "source_month": entry_ts.dt.strftime("%Y-%m"),
        "candidate_id": admitted["candidate_id"].astype(str),
        "mapped_expected_net_bps": admitted[expected_field].to_numpy(float),
        "auction_expected_net_bps": admitted["auction_expected_net_bps"].to_numpy(float),
        "auction_rank_adjustment_bps": adjustment.reindex(admitted.index).to_numpy(float),
        "policy_outcome_available": outcome_available.to_numpy(bool),
        "policy_outcome_proxy_for_constraints": (~outcome_available).to_numpy(bool),
        "policy_outcome_source": admitted.get(
            "policy_outcome_source",
            pd.Series("unspecified", index=admitted.index),
        ).fillna("unspecified").astype(str).to_numpy(),
        "portfolio_size_multiplier": pd.to_numeric(
            admitted.get(
                "portfolio_size_multiplier",
                pd.Series(1.0, index=admitted.index),
            ),
            errors="coerce",
        ).fillna(1.0).to_numpy(float),
    })
    return normalise_candidate_table(output)


def _load_authoritative_cell_day_provenance(
    *, ledger: pd.DataFrame, path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load an independently materialised canonical Cell-day map.

    The promoted Cell-day estimator is intentionally not equivalent to the
    older hierarchical row-weighted map.  Recomputing the latter and then
    comparing it with Cell-day provenance is therefore invalid.  Accept the
    persisted map only when its candidate identity, producer lineage, mapping
    status and strictly-prior audit all match the score ledger exactly.
    """
    provenance = pd.read_parquet(path)
    required = {
        "candidate_id", "causal_21d_side_expected_net_bps",
        "causal_21d_side_admitted_ge_50bps",
        "causal_21d_side_mapping_status",
    }
    missing = sorted(required.difference(provenance.columns))
    if missing:
        raise ValueError(f"canonical Cell-day provenance lacks: {missing}")
    if provenance["candidate_id"].duplicated().any():
        raise ValueError("canonical Cell-day provenance has duplicate candidate IDs")
    if (
        len(provenance) != len(ledger)
        or set(provenance["candidate_id"]) != set(ledger["candidate_id"])
    ):
        raise ValueError("canonical Cell-day provenance does not exactly cover the score ledger")
    if not provenance["causal_21d_side_mapping_status"].astype(str).eq(
        CELL_DAY_TRIM_15_CALIBRATION_MODE,
    ).all():
        raise ValueError("admission provenance is not the canonical Cell-day trim-15 map")
    expected_flag = (
        np.isfinite(pd.to_numeric(
            provenance["causal_21d_side_expected_net_bps"], errors="coerce",
        ))
        & pd.to_numeric(
            provenance["causal_21d_side_expected_net_bps"], errors="coerce",
        ).ge(50.0)
    )
    if not np.array_equal(
        expected_flag.to_numpy(bool),
        provenance["causal_21d_side_admitted_ge_50bps"].fillna(False).to_numpy(bool),
    ):
        raise ValueError("canonical Cell-day admission flags disagree with the +50-bps rule")
    lineage = (
        "__decision_ts__", "conversion_bundle_sha256", "geometry_bundle_sha256",
        "upstream_bundle_sha256", "ev_score_family_id", "stack_is_prequential",
    )
    present = [column for column in lineage if column in provenance.columns]
    if len(present) != len(lineage):
        raise ValueError("canonical Cell-day provenance lacks complete score-producer lineage")
    check = ledger.loc[:, ["candidate_id", *lineage]].merge(
        provenance.loc[:, ["candidate_id", *lineage]], on="candidate_id",
        how="inner", validate="one_to_one", suffixes=("__score", "__map"),
    )
    for column in lineage:
        left, right = check[f"{column}__score"], check[f"{column}__map"]
        if not left.astype(str).eq(right.astype(str)).all():
            raise ValueError(f"canonical Cell-day provenance changed lineage field: {column}")
    audit_path = path.parent / "cell_day_admission_audit.parquet"
    manifest_path = path.parent / "run_manifest.json"
    if not audit_path.exists() or not manifest_path.exists():
        raise FileNotFoundError("canonical Cell-day provenance requires its audit and manifest")
    audit = pd.read_parquet(audit_path)
    if "strictly_prior_resolved" not in audit or not audit[
        "strictly_prior_resolved"
    ].fillna(False).astype(bool).all():
        raise ValueError("canonical Cell-day provenance did not pass strictly-prior resolution")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("mapping") != CELL_DAY_TRIM_15_CALIBRATION_MODE:
        raise ValueError("canonical Cell-day provenance manifest has the wrong mapping contract")
    map_columns = [
        column for column in provenance.columns
        if column.startswith("causal_") or column.startswith("cell_day_")
        or column.startswith("ev_mapping_") or column == "ev_bridge_bundle_identity"
    ]
    base = ledger.drop(columns=[column for column in map_columns if column in ledger], errors="ignore")
    admitted = base.merge(
        provenance.loc[:, ["candidate_id", *map_columns]], on="candidate_id",
        how="inner", validate="one_to_one",
    )
    return admitted, audit


def _weekly(decisions: pd.DataFrame) -> pd.DataFrame:
    accepted = decisions.loc[decisions["accepted"].fillna(False)].copy()
    accepted["week"] = pd.to_datetime(accepted["timestamp"], utc=True).dt.to_period("W-SUN").astype(str)
    accepted["net_bps"] = pd.to_numeric(accepted["position_net_return"], errors="coerce") * 10_000.0
    accepted["gross_bps"] = pd.to_numeric(accepted["position_gross_return"], errors="coerce") * 10_000.0
    return accepted.groupby("week", as_index=False).agg(
        # The portfolio normaliser is free to rename/drop the upstream identity.
        # Weekly trade count is a row count, so bind it to the outcome column that
        # is required for every accepted decision instead of a schema-specific ID.
        trades=("net_bps", "size"), gross_bps_per_trade=("gross_bps", "mean"),
        net_bps_per_trade=("net_bps", "mean"), net_sum_bps=("net_bps", "sum"),
        positive_rate=("net_bps", lambda value: float((value > 0.0).mean())),
    )


def _wallet_periods(
    equity: pd.DataFrame,
    *,
    frequency: str,
    initial_wallet: float,
    evaluation_start: pd.Timestamp,
    evaluation_end: pd.Timestamp,
) -> pd.DataFrame:
    """Return non-overlapping realised wallet changes from the replay ledger."""
    frame = (
        equity.loc[:, ["timestamp", "wallet"]].copy()
        if not equity.empty
        else pd.DataFrame(columns=["timestamp", "wallet"])
    )
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame["wallet"] = pd.to_numeric(frame["wallet"], errors="coerce")
    frame = frame.dropna(subset=["timestamp", "wallet"]).sort_values("timestamp", kind="stable")
    last_inclusive = evaluation_end - pd.Timedelta(nanoseconds=1)
    start_naive = evaluation_start.tz_convert(None)
    last_naive = last_inclusive.tz_convert(None)
    if frequency == "month":
        frame["period"] = frame["timestamp"].dt.strftime("%Y-%m")
        periods = pd.period_range(start_naive, last_naive, freq="M").astype(str)
    elif frequency == "week":
        frame["period"] = frame["timestamp"].dt.to_period("W-SUN").astype(str)
        periods = pd.period_range(start_naive, last_naive, freq="W-SUN").astype(str)
    else:
        raise ValueError(f"unsupported wallet frequency: {frequency}")
    ends = frame.groupby("period", sort=True)["wallet"].last().reindex(periods)
    ends = ends.ffill().fillna(float(initial_wallet))
    starts = ends.shift(1)
    if len(starts):
        starts.iloc[0] = float(initial_wallet)
    out = pd.DataFrame({"period": ends.index, "wallet_start": starts.values, "wallet_end": ends.values})
    out["wallet_pnl"] = out["wallet_end"] - out["wallet_start"]
    out["wallet_return_pct"] = 100.0 * out["wallet_pnl"] / out["wallet_start"]
    return out.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--schema", choices=("current-v5", "legacy-v2"), default="current-v5",
        help="Frozen-geometry current-v5 is canonical; legacy-v2 is reconciliation-only.",
    )
    parser.add_argument("--scored-label-ledger", type=Path, required=True)
    parser.add_argument(
        "--producer-lineage", type=Path,
        help=(
            "Immutable candidate-keyed conversion/upstream lineage sidecar for "
            "a pre-repair current-v5 OOF ledger."
        ),
    )
    parser.add_argument(
        "--geometry-mode", choices=("frozen", "episode-isolated"), default="frozen",
        help=(
            "frozen is the one-bundle canonical contract. episode-isolated is "
            "the research-only periodic-K9 arm and resets EV-map support at "
            "every immutable geometry boundary."
        ),
    )
    parser.add_argument(
        "--admission-provenance", type=Path,
        help=(
            "Optional target-free causal EV-map provenance.  When supplied, "
            "the replay requires bit-level agreement with its independently "
            "recomputed admission map."
        ),
    )
    parser.add_argument(
        "--admission-score-ledger", type=Path,
        help=(
            "Optional candidate-identical current-v5 ledger used solely to fit/apply "
            "the causal EV-admission map. The primary scored ledger remains the "
            "auction-ranking source; this is for isolated map-correction research."
        ),
    )
    parser.add_argument(
        "--auction-adjustment-ledger", type=Path,
        help=(
            "Optional candidate-identical sidecar with a bounded bps adjustment "
            "for post-admission auction ordering only. It cannot alter the EV "
            "map or its admission flags."
        ),
    )
    parser.add_argument(
        "--cell-day-trust-bundle-dir", type=Path,
        help=(
            "Frozen R5 residual-trust bundle. With the canonical posterior "
            "integration it replaces raw Cell-day admission and auction value; "
            "without that explicit contract it is historical demotion-only."
        ),
    )
    parser.add_argument(
        "--cell-day-trust-oof-predictions", type=Path,
        help=(
            "Candidate-keyed chronological R5 decomposition for historical replay. "
            "It must cover the evaluated population exactly."
        ),
    )
    parser.add_argument(
        "--cell-day-trust-integration", type=Path,
        help=(
            "Explicit R5 integration contract. The canonical 9-month contract "
            "uses posterior expected policy net for fail-closed admission and ordering."
        ),
    )
    parser.add_argument("--evaluation-start", required=True)
    parser.add_argument("--evaluation-end", required=True, help="exclusive UTC bound")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--initial-wallet", type=float, default=1000.0)
    parser.add_argument("--perp-leverage", type=float, default=7.0)
    parser.add_argument("--margin-slot-wallet-fraction", type=float, default=0.10)
    parser.add_argument(
        "--policy-json", type=Path,
        help=(
            "Frozen SimplePolicyOptimiser winner used to materialise policy outcomes; "
            "required by current-v5."
        ),
    )
    parser.add_argument(
        "--n5-bundle-dir",
        type=Path,
        help="Optional shadow LDF sizing bundle for the evaluation cutoff.",
    )
    parser.add_argument(
        "--n5-features",
        type=Path,
        help="Target-free causal N5 feature sidecar keyed by candidate_id.",
    )
    parser.add_argument(
        "--n5-oof-predictions",
        type=Path,
        help=(
            "Candidate-keyed block-OOF LDF multipliers for a historical replay. "
            "They replace a single static N5 bundle and are merged only after "
            "causal admission."
        ),
    )
    parser.add_argument(
        "--disable-canonical-n5",
        action="store_true",
        help="Deprecated compatibility alias for canonical unit relative sizing.",
    )
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable output already exists: {args.out_dir}")
    if args.schema == "current-v5" and args.policy_json is None:
        parser.error(
            "--policy-json is required for current-v5 so the selected-policy "
            "contract is explicit and hashable"
        )
    n5_requested = (
        args.n5_oof_predictions is not None
        or args.n5_bundle_dir is not None
        or args.n5_features is not None
    )
    if args.schema == "current-v5" and n5_requested and not args.disable_canonical_n5:
        static_n5 = args.n5_bundle_dir is not None or args.n5_features is not None
        if args.n5_oof_predictions is not None and static_n5:
            parser.error(
                "use either --n5-oof-predictions or --n5-bundle-dir/--n5-features, not both",
            )
        if args.n5_oof_predictions is None and (
            args.n5_bundle_dir is None or args.n5_features is None
        ):
            parser.error(
                "LDF shadow replay requires --n5-bundle-dir and --n5-features, "
                "or historical --n5-oof-predictions"
            )
    if args.n5_oof_predictions is not None and args.schema != "current-v5":
        parser.error("--n5-oof-predictions is defined only for current-v5")
    if args.cell_day_trust_bundle_dir is not None and args.cell_day_trust_oof_predictions is not None:
        parser.error("use a frozen R5 bundle or OOF R5 predictions, not both")
    posterior_integration: dict[str, object] | None = None
    if args.cell_day_trust_integration is not None:
        posterior_integration = json.loads(args.cell_day_trust_integration.read_text())
        if posterior_integration.get("schema") != "strict_r3_cell_day_residual_trust_posterior_28d_challenger_v1":
            parser.error("--cell-day-trust-integration is not the canonical R5 posterior contract")
        if args.cell_day_trust_bundle_dir is None and args.cell_day_trust_oof_predictions is None:
            parser.error("posterior integration requires a frozen bundle or OOF predictions")
    if (
        args.cell_day_trust_bundle_dir is not None
        or args.cell_day_trust_oof_predictions is not None
    ) and args.auction_adjustment_ledger is not None:
        parser.error("canonical R5 owns the post-admission auction adjustment")
    policy_values, policy_engine, policy_selection_period = _resolve_policy(
        args.schema, args.policy_json,
    )
    ledger = pd.read_parquet(args.scored_label_ledger)
    if args.schema == "current-v5":
        ledger = _attach_producer_lineage(ledger, args.producer_lineage)
        required = {
            "conversion_bundle_sha256", "geometry_bundle_sha256",
            "upstream_bundle_sha256", "ev_score_family_id",
            "stack_is_prequential", "severe_affects_final_score",
        }
        missing = sorted(required - set(ledger.columns))
        if missing:
            raise ValueError(f"current-v5 scored ledger lacks {missing}")
        if ledger["severe_affects_final_score"].fillna(True).astype(bool).any():
            raise ValueError("current-v5 portfolio replay prohibits active Severe demotion")
        if args.geometry_mode == "frozen":
            geometry_identity: object = require_single_geometry_hash(ledger)
        else:
            required_episode = {"geometry_episode_start", "geometry_episode_end_exclusive"}
            missing_episode = sorted(required_episode.difference(ledger.columns))
            if missing_episode:
                raise ValueError(
                    "episode-isolated replay lacks geometry episode provenance: "
                    f"{missing_episode}",
                )
            geometry_identity = sorted(
                ledger["geometry_bundle_sha256"].dropna().astype(str).unique().tolist(),
            )
            if len(geometry_identity) < 2:
                raise ValueError("episode-isolated replay requires multiple geometry identities")
            timestamp = pd.to_datetime(ledger["__decision_ts__"], utc=True)
            episode_start = pd.to_datetime(ledger["geometry_episode_start"], utc=True)
            episode_end = pd.to_datetime(ledger["geometry_episode_end_exclusive"], utc=True)
            if not timestamp.ge(episode_start).all() or not timestamp.lt(episode_end).all():
                raise ValueError("episodic score rows lie outside their declared geometry bundle")
        map_ledger = ledger
        if args.admission_score_ledger is not None:
            map_ledger = _attach_producer_lineage(
                pd.read_parquet(args.admission_score_ledger), args.producer_lineage,
            )
            if map_ledger["candidate_id"].duplicated().any() or (
                len(map_ledger) != len(ledger)
                or set(map_ledger["candidate_id"]) != set(ledger["candidate_id"])
            ):
                raise ValueError("admission-score ledger must cover primary candidates exactly")
            check_columns = [
                "__decision_ts__", "policy_label_available_ts", "policy_net_bps",
                "geometry_bundle_sha256", "conversion_bundle_sha256",
                "upstream_bundle_sha256", "ev_score_family_id", "stack_is_prequential",
            ]
            missing = sorted(set(check_columns).difference(map_ledger.columns))
            if missing:
                raise ValueError(f"admission-score ledger lacks lineage columns: {missing}")
            aligned = ledger.loc[:, ["candidate_id", *check_columns]].merge(
                map_ledger.loc[:, ["candidate_id", *check_columns]], on="candidate_id",
                how="inner", validate="one_to_one", suffixes=("__primary", "__map"),
            )
            for column in check_columns:
                left, right = aligned[f"{column}__primary"], aligned[f"{column}__map"]
                if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
                    if not np.allclose(left.to_numpy(float), right.to_numpy(float), rtol=0.0, atol=1e-10, equal_nan=True):
                        raise ValueError(f"admission-score ledger changed non-score column: {column}")
                elif not left.astype(str).eq(right.astype(str)).all():
                    raise ValueError(f"admission-score ledger changed non-score column: {column}")
        authoritative_cell_day = False
        if args.admission_provenance is not None:
            provenance = pd.read_parquet(args.admission_provenance)
            authoritative_cell_day = bool(
                "causal_21d_side_mapping_status" in provenance
                and provenance["causal_21d_side_mapping_status"].astype(str).eq(
                    CELL_DAY_TRIM_15_CALIBRATION_MODE,
                ).all()
            )
        if authoritative_cell_day:
            if args.admission_score_ledger is not None:
                raise ValueError(
                    "canonical Cell-day provenance owns its score contract; "
                    "--admission-score-ledger is not permitted",
                )
            admitted, admission_audit = _load_authoritative_cell_day_provenance(
                ledger=ledger, path=args.admission_provenance,
            )
        else:
            mapped, admission_audit = apply_current_admission_by_geometry(
                map_ledger, geometry_mode=args.geometry_mode,
            )
            if args.admission_score_ledger is None:
                admitted = mapped
            else:
                map_columns = [
                    column for column in mapped.columns
                    if column.startswith("causal_") or column.startswith("ev_mapping_")
                ]
                admitted = ledger.merge(
                    mapped.loc[:, ["candidate_id", *map_columns]], on="candidate_id",
                    how="inner", validate="one_to_one",
                )
        if args.admission_provenance is not None and not authoritative_cell_day:
            if "raw_expected_bps" not in provenance:
                if "causal_21d_side_expected_net_bps" not in provenance:
                    raise ValueError(
                        "admission provenance lacks raw_expected_bps and canonical "
                        "causal_21d_side_expected_net_bps",
                    )
                provenance = provenance.rename(columns={
                    "causal_21d_side_expected_net_bps": "raw_expected_bps",
                })
            if "mapped_ev_available" not in provenance:
                provenance["mapped_ev_available"] = np.isfinite(
                    pd.to_numeric(provenance["raw_expected_bps"], errors="coerce"),
                )
            required_provenance = {"candidate_id", "raw_expected_bps", "mapped_ev_available"}
            missing_provenance = sorted(required_provenance.difference(provenance.columns))
            if missing_provenance:
                raise ValueError(f"admission provenance lacks: {missing_provenance}")
            if provenance["candidate_id"].duplicated().any():
                raise ValueError("admission provenance has duplicate candidate IDs")
            if (
                len(provenance) != len(admitted)
                or set(provenance["candidate_id"]) != set(admitted["candidate_id"])
            ):
                raise ValueError("admission provenance does not cover the replay score ledger exactly")
            check = admitted.loc[:, [
                "candidate_id", "causal_21d_side_expected_net_bps",
                "causal_21d_side_admitted_ge_50bps",
            ]].merge(
                provenance.loc[:, ["candidate_id", "raw_expected_bps", "mapped_ev_available"]],
                on="candidate_id", how="left", validate="one_to_one",
            )
            if check["raw_expected_bps"].isna().sum() != admitted[
                "causal_21d_side_expected_net_bps"
            ].isna().sum():
                raise ValueError("admission provenance does not cover the replay score ledger")
            expected = pd.to_numeric(
                check["causal_21d_side_expected_net_bps"], errors="coerce",
            ).to_numpy(float)
            observed = pd.to_numeric(check["raw_expected_bps"], errors="coerce").to_numpy(float)
            if not np.allclose(expected, observed, rtol=0.0, atol=1e-8, equal_nan=True):
                raise ValueError("recomputed causal admission differs from its persisted provenance")
            admitted_flag = check["causal_21d_side_admitted_ge_50bps"].fillna(False).to_numpy(bool)
            provenance_flag = check["mapped_ev_available"].fillna(False).to_numpy(bool) & np.isfinite(observed) & (observed >= 50.0)
            if not np.array_equal(admitted_flag, provenance_flag):
                raise ValueError("recomputed causal admission flags differ from provenance")
        schema_name = CURRENT_SCHEMA
        strategy_prefix = "strict_r3_current_v5"
        score_description = "final policy-correctness prior-28-day CDF"
        geometry_cadence = (
            "one frozen Oct-Dec 2024 geometry/K9 view; never refit"
            if args.geometry_mode == "frozen"
            else "research-only immutable K9 episodes; admission support reset by bundle"
        )
    else:
        geometry_identity = require_single_geometry_hash(ledger)
        admitted, admission_audit = apply_schema_v2_admission(
            ledger,
            score_column="final_score",
            net_column="policy_net_bps",
            label_available_column="policy_label_available_ts",
            spec=Causal21dAdmissionSpec(mode="hierarchical_tail_side_shrinkage_v2"),
        )
        schema_name = LEGACY_SCHEMA
        strategy_prefix = "strict_r3_schema_v2"
        score_description = "final Severe prior-28-day CDF"
        geometry_cadence = "fit once on 2024-10-01 through 2025-01-01, then frozen"
    start = pd.to_datetime(args.evaluation_start, utc=True)
    end = pd.to_datetime(args.evaluation_end, utc=True)
    evaluation = admitted.loc[
        pd.to_datetime(admitted["__decision_ts__"], utc=True).ge(start)
        & pd.to_datetime(admitted["__decision_ts__"], utc=True).lt(end)
    ].copy()
    auction_adjustment_manifest: dict[str, object] = {
        "enabled": False,
        "role": "raw causal EV-map ordering",
    }
    trust_manifest: dict[str, object] = {
        "enabled": False,
        "role": "no residual-trust auction demotion",
    }
    admission_before_trust = evaluation[
        "causal_21d_side_admitted_ge_50bps"
    ].fillna(False).to_numpy(bool)
    if args.cell_day_trust_oof_predictions is not None:
        trust = pd.read_parquet(args.cell_day_trust_oof_predictions)
        posterior_source = next(
            (
                field for field in (
                    "trust_posterior_expected_bps", "posterior_expected_bps",
                ) if field in trust
            ),
            None,
        )
        required_trust = {"candidate_id"}
        if posterior_integration is None:
            required_trust.add("auction_rank_adjustment_bps")
        elif posterior_source is None:
            required_trust.add("trust_posterior_expected_bps")
        missing_trust = sorted(required_trust.difference(trust.columns))
        if missing_trust:
            raise ValueError(f"R5 OOF predictions lack: {missing_trust}")
        if trust["candidate_id"].duplicated().any():
            raise ValueError("R5 OOF predictions contain duplicate candidate IDs")
        if len(trust) != len(evaluation) or set(trust["candidate_id"]) != set(evaluation["candidate_id"]):
            raise ValueError("R5 OOF predictions must exactly cover the evaluated population")
        if posterior_source == "posterior_expected_bps":
            trust = trust.rename(columns={
                "posterior_expected_bps": "trust_posterior_expected_bps",
            })
        trust_columns = [
            column for column in trust.columns
            if column == "candidate_id" or column.startswith("trust_")
            or column == "auction_rank_adjustment_bps"
        ]
        evaluation = evaluation.merge(
            trust.loc[:, trust_columns], on="candidate_id", how="inner", validate="one_to_one",
        )
        trust_manifest = {
            "enabled": True,
            "mode": "chronological_block_oof",
            "path": str(args.cell_day_trust_oof_predictions),
            "sha256": _sha(args.cell_day_trust_oof_predictions),
        }
    elif args.cell_day_trust_bundle_dir is not None:
        bundle = load_cell_day_residual_trust_bundle(args.cell_day_trust_bundle_dir)
        trust_input = evaluation.loc[:, ["candidate_id", *bundle.fields]].copy()
        trust_input["raw_expected_bps"] = pd.to_numeric(
            evaluation["causal_21d_side_expected_net_bps"], errors="coerce",
        ).to_numpy(float)
        trust = bundle.score(trust_input)
        evaluation = evaluation.merge(
            trust, on="candidate_id", how="inner", validate="one_to_one",
        )
        trust_manifest = {
            "enabled": True,
            "mode": "frozen_bundle",
            "bundle_dir": str(args.cell_day_trust_bundle_dir),
            "bundle_cutoff": bundle.cutoff.isoformat(),
            "bundle_sha256": bundle.manifest.get("bundle_sha256"),
        }
    if trust_manifest["enabled"]:
        if posterior_integration is not None:
            posterior = pd.to_numeric(
                evaluation["trust_posterior_expected_bps"], errors="coerce",
            )
            evaluation["trust_posterior_admitted_ge_50bps"] = (
                np.isfinite(posterior) & posterior.ge(50.0)
            )
            trust_manifest.update({
                "integration_contract": str(args.cell_day_trust_integration),
                "integration_contract_sha256": _sha(args.cell_day_trust_integration),
                "role": "R5 9-month posterior expected-net admission and ordering",
                "missing_posterior": "fail_closed",
                "posterior_available_rows": int(np.isfinite(posterior).sum()),
                "posterior_admitted_rows": int(
                    evaluation["trust_posterior_admitted_ge_50bps"].sum()
                ),
                "cell_day_admission_changed_rows": int(np.sum(
                    admission_before_trust
                    != evaluation["trust_posterior_admitted_ge_50bps"].to_numpy(bool)
                )),
            })
            auction_adjustment_manifest = {
                "enabled": False,
                "role": "posterior expected net is the direct auction value",
            }
        else:
            adjustment = pd.to_numeric(
                evaluation["auction_rank_adjustment_bps"], errors="raise",
            )
            if (~np.isfinite(adjustment) | adjustment.gt(1e-8)).any():
                raise ValueError("canonical R5 adjustment must be finite and demotion-only")
            if not np.array_equal(
                admission_before_trust,
                evaluation["causal_21d_side_admitted_ge_50bps"].fillna(False).to_numpy(bool),
            ):
                raise AssertionError("canonical R5 changed Cell-day admission")
            auction_adjustment_manifest = {
                "enabled": True,
                "role": "historical R5 post-admission corroborated demotion only",
                "min_bps": float(adjustment.min()),
                "max_bps": float(adjustment.max()),
            }
    if args.auction_adjustment_ledger is not None:
        adjustment = pd.read_parquet(args.auction_adjustment_ledger)
        required_adjustment = {"candidate_id", "auction_rank_adjustment_bps"}
        missing_adjustment = sorted(required_adjustment.difference(adjustment.columns))
        if missing_adjustment:
            raise ValueError(f"auction-adjustment ledger lacks {missing_adjustment}")
        if adjustment["candidate_id"].duplicated().any():
            raise ValueError("auction-adjustment ledger has duplicate candidate IDs")
        if len(adjustment) != len(ledger) or set(adjustment["candidate_id"]) != set(ledger["candidate_id"]):
            raise ValueError("auction-adjustment ledger must cover primary candidates exactly")
        values = pd.to_numeric(adjustment["auction_rank_adjustment_bps"], errors="coerce")
        if (~np.isfinite(values)).any():
            raise ValueError("auction rank adjustments must be finite")
        evaluation = evaluation.merge(
            adjustment.loc[:, ["candidate_id", "auction_rank_adjustment_bps"]],
            on="candidate_id", how="left", validate="one_to_one",
        )
        if evaluation["auction_rank_adjustment_bps"].isna().any():
            raise ValueError("auction-adjustment ledger does not cover every evaluated candidate")
        auction_adjustment_manifest = {
            "enabled": True,
            "role": "post-admission auction ordering only; causal EV map unchanged",
            "ledger": str(args.auction_adjustment_ledger),
            "ledger_sha256": _sha(args.auction_adjustment_ledger),
            "min_bps": float(values.min()),
            "max_bps": float(values.max()),
        }
    n5_enabled = args.schema == "current-v5" and n5_requested and not args.disable_canonical_n5
    n5_manifest: dict[str, object] = {
        "enabled": False,
        "role": "canonical_unit_size" if not n5_enabled else "shadow_ldf_requested",
    }
    if n5_enabled and args.n5_oof_predictions is not None:
        oof = pd.read_parquet(args.n5_oof_predictions)
        required_oof = {
            "candidate_id", "trust_size_multiplier", "n5_available", "n5_bundle_cutoff",
        }
        missing_oof = sorted(required_oof.difference(oof.columns))
        if missing_oof:
            raise ValueError(f"N5 OOF prediction ledger lacks: {missing_oof}")
        if oof["candidate_id"].duplicated().any():
            raise ValueError("N5 OOF prediction ledger has duplicate candidate IDs")
        oof = oof.loc[:, list(required_oof)].copy()
        evaluation = evaluation.merge(
            oof, on="candidate_id", how="left", validate="one_to_one",
        )
        if evaluation["trust_size_multiplier"].isna().any():
            raise ValueError("N5 OOF prediction ledger does not cover every evaluated candidate")
        multiplier = pd.to_numeric(evaluation["trust_size_multiplier"], errors="coerce")
        if (~np.isfinite(multiplier) | multiplier.le(0.0)).any():
            raise ValueError("N5 OOF size multipliers must be finite and positive")
        evaluation["portfolio_size_multiplier"] = multiplier.to_numpy(float)
        n5_manifest = {
            "enabled": True,
            "mode": "block_oof_historical_replay",
            "oof_predictions": str(args.n5_oof_predictions),
            "oof_predictions_sha256": _sha(args.n5_oof_predictions),
            "available_rows": int(evaluation["n5_available"].fillna(False).astype(bool).sum()),
            "unit_size_warmup_rows": int((~evaluation["n5_available"].fillna(False).astype(bool)).sum()),
            "integration": "after causal admission; relative position size only; each multiplier is block-OOF",
        }
    elif n5_enabled:
        bundle = load_canonical_n5_bundle(args.n5_bundle_dir)
        features = pd.read_parquet(args.n5_features)
        if "candidate_id" not in features or features["candidate_id"].duplicated().any():
            raise ValueError("N5 feature sidecar requires unique candidate_id")
        resident_fields = [field for field in bundle.fields if field in evaluation.columns]
        feature_columns = [
            column for column in features.columns
            if column != "candidate_id" and column not in resident_fields and column != "final_score"
        ]
        n5_input = evaluation.loc[:, ["candidate_id", "final_score", *resident_fields]].merge(
            features.loc[:, ["candidate_id", *feature_columns]],
            on="candidate_id",
            how="left",
            validate="one_to_one",
        )
        # The only economic input at scoring is the causal prior-resolved EV
        # map.  Realised policy outcomes remain outside the N5 feature frame.
        n5_input["raw_expected_bps"] = pd.to_numeric(
            evaluation["causal_21d_side_expected_net_bps"], errors="coerce",
        ).to_numpy(float)
        n5_score = score_canonical_n5_bundle(bundle, n5_input)
        evaluation = evaluation.merge(n5_score, on="candidate_id", how="left", validate="one_to_one")
        evaluation["portfolio_size_multiplier"] = pd.to_numeric(
            evaluation["portfolio_size_multiplier"], errors="coerce",
        ).fillna(1.0)
        n5_manifest = {
            "enabled": True,
            "schema": bundle.schema,
            "bundle_dir": str(args.n5_bundle_dir),
            "bundle_cutoff": bundle.cutoff.isoformat(),
            "feature_sidecar": str(args.n5_features),
            "feature_sidecar_sha256": _sha(args.n5_features),
            "integration": "after causal admission; relative position size only",
        }
    else:
        evaluation["portfolio_size_multiplier"] = 1.0
    candidates = _auction_candidates(evaluation, strategy_prefix=strategy_prefix)
    if candidates.empty:
        decisions = pd.DataFrame(columns=[
            "timestamp", "accepted", "position_net_return",
            "position_gross_return",
        ])
        equity = pd.DataFrame(columns=["timestamp", "wallet"])
        monthly = pd.DataFrame(columns=[
            "month", "trades", "net_bps_per_trade", "gross_bps_per_trade",
            "net_sum_bps", "positive_rate", "threshold",
        ])
        summary = {
            "accepted_trades": 0,
            "candidate_trades": 0,
            "net_bps_per_trade": np.nan,
            "gross_bps_per_trade": np.nan,
            "net_sum_bps": 0.0,
            "trades_per_day": 0.0,
            "wallet_start": float(args.initial_wallet),
            "wallet_end": float(args.initial_wallet),
            "wallet_pnl": 0.0,
            "wallet_return_pct": 0.0,
            "zero_admission_fail_closed": True,
        }
    else:
        decisions, equity, monthly, summary = _run(
            candidates, 0.0, f"{start.isoformat()}_to_{end.isoformat()}_{args.schema}",
            initial_wallet=float(args.initial_wallet),
            perp_leverage=float(args.perp_leverage),
            margin_slot_wallet_fraction=float(args.margin_slot_wallet_fraction),
            ev_curve=CAUSAL_AUCTION_CURVE,
        )
    # The shared replay summary reports trades/day over the span from the
    # first accepted trade to the last.  That is useful as an active-period
    # density, but it overstates executable cadence when causal admission
    # deliberately fails closed for whole periods.  Canonical reporting
    # reporting uses the complete declared evaluation calendar.
    evaluation_days = max((end - start).total_seconds() / 86_400.0, 1.0)
    summary["active_span_trades_per_day"] = float(summary.get("trades_per_day", np.nan))
    summary["evaluation_calendar_days"] = float(evaluation_days)
    summary["trades_per_day"] = float(summary["accepted_trades"] / evaluation_days)
    monthly_wallet = _wallet_periods(
        equity, frequency="month", initial_wallet=float(args.initial_wallet),
        evaluation_start=start, evaluation_end=end,
    ).rename(columns={"period": "month"})
    monthly = monthly.merge(monthly_wallet, on="month", how="outer", validate="one_to_one")
    monthly["trades"] = monthly["trades"].fillna(0).astype(int)
    monthly["net_sum_bps"] = monthly["net_sum_bps"].fillna(0.0)
    monthly["threshold"] = monthly["threshold"].fillna(0.0)
    weekly = _weekly(decisions)
    weekly_wallet = _wallet_periods(
        equity, frequency="week", initial_wallet=float(args.initial_wallet),
        evaluation_start=start, evaluation_end=end,
    ).rename(columns={"period": "week"})
    weekly = weekly.merge(weekly_wallet, on="week", how="outer", validate="one_to_one")
    weekly["trades"] = weekly["trades"].fillna(0).astype(int)
    weekly["net_sum_bps"] = weekly["net_sum_bps"].fillna(0.0)
    args.out_dir.mkdir(parents=True)
    admitted.to_parquet(args.out_dir / "score_and_21d_admission_provenance.parquet", index=False, compression="zstd")
    evaluation.loc[
        :,
        [
            column for column in (
                "candidate_id", "__decision_ts__", "final_score",
                "causal_21d_side_expected_net_bps", "causal_21d_side_admitted_ge_50bps",
                "trust_posterior_expected_bps", "trust_posterior_admitted_ge_50bps",
                "n5_expected_bps", "n5_predictive_sd_bps", "n5_shrinkage_lambda",
                "n5_effective_support", "n5_p_ev_positive", "n5_p_adverse_200",
                "portfolio_size_multiplier", "n5_bundle_cutoff", "n5_schema",
            )
            if column in evaluation.columns
        ],
    ].to_parquet(args.out_dir / "n5_sizing_decomposition.parquet", index=False, compression="zstd")
    admission_audit.to_parquet(args.out_dir / "causal_21d_admission_audit.parquet", index=False)
    decisions.to_parquet(args.out_dir / "portfolio_decisions.parquet", index=False, compression="zstd")
    equity.to_parquet(args.out_dir / "portfolio_equity.parquet", index=False, compression="zstd")
    monthly.to_parquet(args.out_dir / "monthly_metrics.parquet", index=False)
    weekly.to_parquet(args.out_dir / "weekly_metrics.parquet", index=False)
    pd.DataFrame([summary]).to_parquet(args.out_dir / "portfolio_summary.parquet", index=False)
    manifest = {
        "schema": f"{schema_name}_admission_portfolio",
        "score": score_description,
        "admission": (
            "R5 nine-month posterior expected policy net >= +50 bps; missing posterior fail-closed"
            if posterior_integration is not None else
            "same exact-producer fixed score cells; one equal-weight policy-net "
            "mean per UTC day x cell over 28 days; symmetric 15% day trimming; "
            "monotone expected-net curve >= +50 bps"
            if args.admission_provenance is not None and authoritative_cell_day else
            "causal prior-resolved hierarchical 21/42/84-day uneven-tail side-shrunk map >= +50 bps"
        ),
        "insufficient_support": "fail_closed",
        "auction_order": (
            "R5 posterior expected policy net bps, then same-producer final score"
            if posterior_integration is not None else
            "mapped expected net bps among currently actionable admitted candidates"
        ),
        "auction_ev_curve": CAUSAL_AUCTION_CURVE,
        "retrospective_percentile_threshold": None,
        "portfolio": "global auction; 8 concurrent; 2 new per 15m bar; 1 per asset; 80% margin cap",
        "exit_policy": {
            "entry": "signal close + one hour",
            "sl_atr": policy_values["sl_mult"],
            "trailing_activation_atr": policy_values["trailing_activation_mult"],
            "trailing_giveback_atr": policy_values["fixed_trailing_gap_mult"],
            "timeout_hours": 12, "cost_bps_once": 100.0,
            "engine": policy_engine,
            "selection_period": policy_selection_period,
            "contract_path": str(args.policy_json) if args.policy_json else None,
            "contract_sha256": _sha(args.policy_json) if args.policy_json else None,
        },
        "initial_wallet": args.initial_wallet, "leverage": args.perp_leverage,
        "margin_slot_wallet_fraction": args.margin_slot_wallet_fraction,
        "evaluation_start": start.isoformat(), "evaluation_end_exclusive": end.isoformat(),
        "geometry_bundle_sha256": geometry_identity,
        "geometry_refit_cadence": geometry_cadence,
        "producer_lineage": (
            None if args.schema != "current-v5" else {
                "mode": "strict_full_producer_vintage_fail_closed_v2",
                "sidecar": (
                    str(args.producer_lineage)
                    if args.producer_lineage is not None else "embedded_in_scored_ledger"
                ),
                "sidecar_sha256": (
                    _sha(args.producer_lineage)
                    if args.producer_lineage is not None else None
                ),
            }
        ),
        "n5_forest_support_sizing": n5_manifest,
        "cell_day_residual_trust_overlay": trust_manifest,
        "auction_ranking_adjustment": auction_adjustment_manifest,
        "admission_provenance": (
            None if args.admission_provenance is None else {
                "path": str(args.admission_provenance),
                "sha256": _sha(args.admission_provenance),
                "verified_equal_to_recomputed_map": not authoritative_cell_day,
                "verified_authoritative_cell_day_contract": authoritative_cell_day,
            }
        ),
        "admission_score_ledger": (
            None if args.admission_score_ledger is None else {
                "path": str(args.admission_score_ledger),
                "sha256": _sha(args.admission_score_ledger),
                "role": "causal EV-map score only; auction ranking remains primary ledger final_score",
            }
        ),
    }
    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"event": "complete", "output": str(args.out_dir), **summary}, default=str))


if __name__ == "__main__":
    main()
