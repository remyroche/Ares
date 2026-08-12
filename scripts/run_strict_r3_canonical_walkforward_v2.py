#!/usr/bin/env python3
"""Fit and score the long-only schema-v2 canonical stack month by month.

The source panel may contain evaluation outcomes, but each scoring frame is
reduced to the declared decision-time feature contract before it is passed to
the scorer.  Monthly bundles consume only strict prequential ledger rows that
precede the held cutoff.  The one persisted October-December 2024 geometry/K9
bundle is loaded once and embedded unchanged in every monthly bundle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.strict_r3_canonical_v2 import (  # noqa: E402
    BASE_TRAIN_CAP,
    SCHEMA,
    _equal_month_sample,
    assert_scoring_frame_is_target_free,
    load_geometry_bundle,
    load_monthly_bundle,
    persist_monthly_bundle,
    require_single_geometry_hash,
    score_same_model_reference,
    train_monthly_bundle,
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _contracts(path: Path) -> tuple[list[str], list[str]]:
    payload = json.loads(path.read_text())
    return (
        [str(value) for value in payload["base_fields_by_side"]["long"]],
        [str(value) for value in payload["severe_context_fields"]],
    )


def _read_window(
    path: Path,
    *,
    columns: list[str],
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Read only one chronological score/training window from parquet.

    The previous runner materialised the entire source panel and ledger before
    filtering a single fold.  Besides being memory-heavy, that made a normal
    one-month replay needlessly fragile.  Parquet predicate pushdown keeps a
    fold's working set bounded without changing its chronological contract.
    """
    filters: list[tuple[str, str, object]] = []
    if start is not None:
        filters.append(("__decision_ts__", ">=", start))
    if end is not None:
        filters.append(("__decision_ts__", "<", end))
    frame = pd.read_parquet(path, columns=list(dict.fromkeys(columns)), filters=filters or None)
    frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True)
    return frame


def _compact_prequential_ledger(
    ledger: pd.DataFrame,
    *,
    cutoff: pd.Timestamp,
    base_train_cap: int,
    downstream_cap: int,
) -> pd.DataFrame:
    """Retain the exact prequential support each supervised layer can use.

    The base keeps its canonical latest 240k resolved rows.  The policy-map /
    residual and Severe learners each receive a deterministic equal-month
    sample of *only* rows whose labels were available before the cutoff.  The
    union is strictly prequential and avoids retaining several full-history
    copies merely to later apply their established training caps.
    """
    for column in (
        "r3_label_available_ts", "policy_label_available_ts", "h12_label_available_ts",
    ):
        ledger[column] = pd.to_datetime(ledger[column], utc=True, errors="coerce")
    base = ledger.loc[
        ledger["r3_label_available_ts"].lt(cutoff) & ledger["r3_class"].notna()
    ].sort_values("r3_label_available_ts", kind="stable").tail(base_train_cap)
    mapped = ledger.loc[
        ledger["policy_label_available_ts"].lt(cutoff)
        & pd.to_numeric(ledger["policy_net_bps"], errors="coerce").notna()
        & pd.to_numeric(ledger["prequential_base_rank42"], errors="coerce").notna()
    ]
    mapped = _equal_month_sample(mapped, downstream_cap, seed=20260817 + 211)
    severe = ledger.loc[
        ledger["h12_label_available_ts"].lt(cutoff)
        & ledger["h12_label_valid"].fillna(False).astype(bool)
        & pd.to_numeric(ledger["h12_tp6_sl4_net_bps"], errors="coerce").notna()
        & ~ledger["__decision_ts__"].between(
            pd.Timestamp("2024-10-01", tz="UTC"),
            pd.Timestamp("2025-01-01", tz="UTC"), inclusive="left",
        )
    ]
    severe = _equal_month_sample(severe, downstream_cap, seed=20260817 + 301)
    selected = base.index.union(mapped.index).union(severe.index)
    compact = ledger.loc[selected].copy().sort_values(
        ["__decision_ts__", "candidate_id"], kind="stable",
    )
    if compact["candidate_id"].duplicated().any():
        raise AssertionError("compact prequential ledger changed candidate identity")
    return compact


def _score_frame(
    panel: pd.DataFrame,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    base_fields: list[str],
    context_fields: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    block = panel.loc[
        panel["__decision_ts__"].ge(start) & panel["__decision_ts__"].lt(end)
    ].copy()
    identity = block.loc[
        :, ["candidate_id", "__ts__", "__decision_ts__", "__symbol__", "side_name"]
    ].copy()
    selected = list(dict.fromkeys([
        "candidate_id", "__decision_ts__", "side_name", *base_fields, *context_fields,
    ]))
    score = block.loc[:, selected].copy()
    assert_scoring_frame_is_target_free(score)
    return score, identity


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-panel", type=Path, required=True)
    parser.add_argument("--prequential-ledger", type=Path, required=True)
    parser.add_argument("--geometry-bundle", type=Path, required=True)
    parser.add_argument("--feature-contract", type=Path, required=True)
    parser.add_argument(
        "--downstream-train-cap", type=int, default=80_000,
        help="Equal-month prequential rows retained for policy-map/residual and Severe training.",
    )
    parser.add_argument(
        "--train-cap", type=int, default=BASE_TRAIN_CAP,
        help="Maximum strict-prequential training rows per base/map/Severe learner.",
    )
    parser.add_argument("--first-held-month", required=True)
    parser.add_argument("--last-held-month", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    base_fields, context_fields = _contracts(args.feature_contract)
    if len(base_fields) != 120 or len(context_fields) != 73:
        raise ValueError("schema-v2 requires 120 base and 73 Severe-context fields")
    geometry = load_geometry_bundle(args.geometry_bundle)
    geometry_hash = geometry.bundle_sha256
    months = pd.date_range(
        pd.Timestamp(args.first_held_month, tz="UTC"),
        pd.Timestamp(args.last_held_month, tz="UTC"),
        freq="MS",
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    source_hashes = {
        "source_panel": _sha(args.source_panel),
        "prequential_ledger": _sha(args.prequential_ledger),
        "geometry_manifest": _sha(args.geometry_bundle / "run_manifest.json"),
        "feature_contract": _sha(args.feature_contract),
    }
    score_columns = list(dict.fromkeys([
        "candidate_id", "__ts__", "__decision_ts__", "__symbol__", "side_name",
        # Evaluation-only columns are carried in the source window but are
        # deliberately excluded by _score_frame before any model call.  They
        # are joined back only after immutable predictions are produced.
        "policy_path_valid", "policy_label_available_ts", "policy_net_bps",
        "policy_gross_bps", "policy_exit_reason", "policy_exit_bar_15m",
        "h12_label_valid", "h12_label_available_ts", "h12_tp6_sl4_net_bps",
        *base_fields, *context_fields,
    ]))
    ledger_columns = list(dict.fromkeys([
        "candidate_id", "__decision_ts__", "__symbol__", "side_name",
        "stack_is_prequential", "r3_class", "r3_label_available_ts",
        "policy_label_available_ts", "policy_net_bps",
        "h12_label_available_ts", "h12_label_valid", "h12_tp6_sl4_net_bps",
        "prequential_base_rank42", "prequential_base_anchor_bps",
        "prequential_consensus_rank", "prequential_residual_rank", "prequential_upstream",
        *base_fields, *context_fields,
    ]))
    monthly_predictions: list[pd.DataFrame] = []
    monthly_audits: list[pd.DataFrame] = []
    rows: list[dict[str, object]] = []
    for cutoff in months:
        month = cutoff.strftime("%Y-%m")
        end = cutoff + pd.offsets.MonthBegin(1)
        bundle_dir = args.out_dir / "bundles" / f"month={month}"
        score_dir = args.out_dir / "scores" / f"month={month}"
        score_window = _read_window(
            args.source_panel, columns=score_columns,
            start=cutoff - pd.Timedelta(days=42), end=end,
        )
        score_window = score_window.loc[
            score_window["side_name"].astype(str).str.lower().eq("long")
        ].copy()
        score_window["__ts__"] = pd.to_datetime(score_window["__ts__"], utc=True)
        held_score, held_identity = _score_frame(
            score_window, start=cutoff, end=end,
            base_fields=base_fields, context_fields=context_fields,
        )
        reference_score, reference_identity = _score_frame(
            score_window, start=cutoff - pd.Timedelta(days=42), end=cutoff,
            base_fields=base_fields, context_fields=context_fields,
        )
        if held_score.empty or reference_score.empty:
            rows.append({"held_month": month, "status": "skipped_empty_score_population"})
            continue
        if bundle_dir.exists():
            bundle = load_monthly_bundle(bundle_dir)
        else:
            prior_ledger = _read_window(
                args.prequential_ledger, columns=ledger_columns, end=cutoff,
            )
            prior_ledger = prior_ledger.loc[
                prior_ledger["side_name"].astype(str).str.lower().eq("long")
            ].copy()
            prior_ledger = _compact_prequential_ledger(
                prior_ledger, cutoff=cutoff, base_train_cap=args.train_cap,
                downstream_cap=args.downstream_train_cap,
            )
            bundle = train_monthly_bundle(
                cutoff=cutoff,
                training_ledger=prior_ledger,
                frozen_geometry=geometry,
                base_fields=base_fields,
                context_fields=context_fields,
                train_cap=args.train_cap,
                source_hashes=source_hashes,
            )
            persist_monthly_bundle(bundle, bundle_dir)
        if bundle.geometry.bundle_sha256 != geometry_hash:
            raise AssertionError("monthly bundle changed the frozen geometry/K9 identity")
        if (score_dir / "predictions.parquet").exists():
            held_predictions = pd.read_parquet(score_dir / "predictions.parquet")
            audit = pd.read_parquet(score_dir / "same_model_reference_replay_audit.parquet")
        else:
            scored, audit = score_same_model_reference(
                bundle, reference=reference_score, held=held_score,
            )
            score_dir.mkdir(parents=True, exist_ok=False)
            reference_predictions = scored.loc[scored["__score_role__"].eq("reference")].copy()
            held_predictions = scored.loc[scored["__score_role__"].eq("held")].copy()
            held_predictions = held_identity.merge(
                held_predictions.drop(columns="__score_role__"),
                on=["candidate_id", "__decision_ts__", "side_name"],
                validate="one_to_one",
            )
            held_outcomes = score_window.loc[
                score_window["__decision_ts__"].ge(cutoff)
                & score_window["__decision_ts__"].lt(end),
                [
                    "candidate_id", "__decision_ts__", "side_name",
                    "policy_path_valid", "policy_label_available_ts", "policy_net_bps",
                    "policy_gross_bps", "policy_exit_reason", "policy_exit_bar_15m",
                    "h12_label_valid", "h12_label_available_ts", "h12_tp6_sl4_net_bps",
                ],
            ]
            held_predictions = held_predictions.merge(
                held_outcomes,
                on=["candidate_id", "__decision_ts__", "side_name"],
                how="left", validate="one_to_one",
            )
            reference_predictions = reference_identity.merge(
                reference_predictions.drop(columns="__score_role__"),
                on=["candidate_id", "__decision_ts__", "side_name"],
                validate="one_to_one",
            )
            held_predictions.to_parquet(
                score_dir / "predictions.parquet", index=False, compression="zstd",
            )
            reference_predictions.to_parquet(
                score_dir / "same_model_prior42_reference_scores.parquet",
                index=False, compression="zstd",
            )
            audit.to_parquet(
                score_dir / "same_model_reference_replay_audit.parquet", index=False,
            )
            (score_dir / "run_manifest.json").write_text(json.dumps({
                "schema": f"{SCHEMA}_monthly_score_output",
                "held_month": month,
                "side_name": "long",
                "held_rows": len(held_predictions),
                "reference_rows": len(reference_predictions),
                "bundle_sha256": bundle.manifest["bundle_sha256"],
                "geometry_bundle_sha256": geometry_hash,
                "same_bundle_for_reference_and_held": True,
                "held_percentile_operations": 0,
                "outcome_columns_consumed": [],
            }, indent=2))
        membership_path = score_dir / "k9_membership_internal.parquet"
        if not membership_path.exists():
            # This is an internal, target-free lineage sidecar.  It exists
            # solely to construct prequential soft-K9 correctness/residual
            # aggregates after outcomes have been joined.  Raw memberships are
            # never copied into walkforward_predictions or an N5 contract.
            membership_state = bundle.geometry.transform(held_score)
            membership_fields = [
                column for column in membership_state.columns
                if column.startswith("k09__cluster_") and column.endswith("__membership")
            ]
            if len(membership_fields) != 9:
                raise AssertionError("frozen K9 internal sidecar requires nine memberships")
            membership_sidecar = held_score.loc[
                :, ["candidate_id", "__decision_ts__", "side_name"]
            ].reset_index(drop=True)
            membership_sidecar["geometry_bundle_sha256"] = geometry_hash
            membership_sidecar = pd.concat(
                [membership_sidecar, membership_state.loc[:, membership_fields].reset_index(drop=True)],
                axis=1,
            )
            membership_sidecar.to_parquet(membership_path, index=False, compression="zstd")
        monthly_predictions.append(held_predictions)
        monthly_audits.append(audit.assign(held_month=month))
        rows.append({
            "held_month": month, "status": "complete",
            "held_rows": len(held_predictions), "reference_rows": len(reference_score),
            "bundle_sha256": bundle.manifest["bundle_sha256"],
            "geometry_bundle_sha256": geometry_hash,
        })
        print(json.dumps({"event": "month_complete", **rows[-1]}), flush=True)
    if not monthly_predictions:
        raise RuntimeError("no monthly predictions were produced")
    predictions = pd.concat(monthly_predictions, ignore_index=True).sort_values(
        ["__decision_ts__", "candidate_id"], kind="stable",
    ).reset_index(drop=True)
    require_single_geometry_hash(predictions)
    predictions.to_parquet(
        args.out_dir / "walkforward_predictions.parquet", index=False, compression="zstd",
    )
    pd.concat(monthly_audits, ignore_index=True).to_parquet(
        args.out_dir / "same_model_reference_replay_audit.parquet", index=False,
    )
    pd.DataFrame(rows).to_parquet(args.out_dir / "monthly_run_audit.parquet", index=False)
    manifest = {
        "schema": f"{SCHEMA}_walkforward_long",
        "side_name": "long", "rows": len(predictions),
        "first_held_month": args.first_held_month,
        "last_held_month": args.last_held_month,
        "monthly_bundles": len(monthly_predictions),
        "training_cap": int(args.train_cap),
        "downstream_training_cap": int(args.downstream_train_cap),
        "geometry_bundle_sha256": geometry_hash,
        "geometry_refit_cadence": "never; one Oct-Dec 2024 definition",
        "held_percentile_operations": 0,
        "outcome_columns_consumed_during_scoring": [],
        "source_hashes": source_hashes,
    }
    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"event": "complete", "output": str(args.out_dir), **manifest}), flush=True)


if __name__ == "__main__":
    main()
