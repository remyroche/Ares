#!/usr/bin/env python3
"""Build the minimal causal early-history feature panels for F72.

This producer exists solely to make the frozen F72 base head evaluable under
the declared strict-OFF chronology.  It creates a target-free feature panel
for the union of:

* the immutable 72-field F72 B-head contract; and
* the immutable 30-field full-universe router contract.

The B0 target-label parquet is read *only* as a pre-existing candidate
identity ledger.  Its outcome columns are never selected, joined to the
output, or used to decide whether a candidate receives features.  Every row
in the ledger, including rows with invalid or unresolved targets, is carried
into the materialisation attempt.  Downstream scorers retain their separate
strict label-availability gates.

Research only.  No inference, admission, portfolio, exchange, or live state
is read or changed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_tp6_sl4_exact170_canonical_consensus import (  # noqa: E402
    FROZEN_GENERATION_DEPENDENCIES,
    materialize_features,
)


# This is the retained 72-field Raw-bps Base contract.  The historical
# direct-head winner was superseded, and no longer exists in the artifact
# store.  This producer only needs the frozen ordered causal field list; it
# never reads a target, fitted booster, or score from this selection file.
DEFAULT_B_WINNER = ROOT / "data_perp/artifacts/strict_r3_b0_family_addback_20260826_v1_policy_ordinal_base_g3/selection.json"
DEFAULT_ROUTER_CONTRACT = ROOT / "data_perp/artifacts/strict_r3_fulluniverse_recall_selector_20260826_v2/selected_full_feature_contract.json"
DEFAULT_IDENTITY_ROOT = ROOT / "data_perp/artifacts/strict_r3_b0_replacement_target_labels_20260826_v3"
CANONICAL_PANEL_MATERIALISER = ROOT / "scripts/run_tp6_sl4_exact170_canonical_consensus.py"
SOURCE_PRECEDENCE_CONTRACT = "cell_local_15m_cache_official_legacy_v2"

# ``rv_48h`` is stored after a causal normalization by ``rv_120h`` in the
# frozen feature engine.  Full-universe production materialisation computes
# this parent implicitly; a minimal requested-field call must declare it
# explicitly or it changes the meaning of a selected F72 field.  The parent
# is generation-only and is never exposed as a Base/Router model feature.
GENERATION_ONLY_DEPENDENCIES = ("rv_120h",)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_fields(path: Path, keys: tuple[str, ...]) -> tuple[list[str], str]:
    payload: Any = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object with one of {keys!r}")
    key = next((candidate for candidate in keys if isinstance(payload.get(candidate), list)), None)
    values = payload.get(key) if key is not None else None
    if not isinstance(values, list) or not values:
        raise ValueError(f"{path}: expected non-empty list under one of {keys!r}")
    fields = [str(value) for value in values]
    if len(fields) != len(set(fields)):
        raise ValueError(f"{path}: duplicate {key!r} fields")
    return fields, key


def _candidate_identities(
    identity_root: Path,
    month: pd.Timestamp,
    *,
    allow_missing_predecessor_partition: bool = False,
) -> tuple[pd.DataFrame, dict[str, int | bool | list[str]]]:
    # The target-label partitions follow the signal-close month, while this
    # score ledger follows the next-bar decision timestamp.  Read the two
    # adjacent signal partitions and retain the exact decision-time month;
    # this includes the required first decision at 00:00 and excludes the
    # following month's first decision without touching any target field.
    partitions = (month - pd.offsets.MonthBegin(1), month)
    columns = [
        "candidate_id", "__decision_ts__", "side_name",
        "policy_ordinal_base_valid", "policy_net_bps",
    ]
    parts = []
    paths = []
    missing_predecessor: list[str] = []
    for position, token in enumerate(partitions):
        path = identity_root / f"month={token:%Y-%m}" / "b0_replacement_targets.parquet"
        if not path.exists():
            if position == 0 and allow_missing_predecessor_partition:
                # This is allowed only for an explicitly declared earliest
                # *training-history* month.  The retained current partition
                # must still cover the entire decision-time month below; no
                # missing prior source is imputed or synthesized.
                missing_predecessor.append(str(path))
                continue
            raise FileNotFoundError(path)
        parts.append(pd.read_parquet(path, columns=columns))
        paths.append(path)
    if not parts:
        raise AssertionError(f"{month:%Y-%m}: no identity source partition available")
    raw = pd.concat(parts, ignore_index=True)
    raw["__decision_ts__"] = pd.to_datetime(raw["__decision_ts__"], utc=True, errors="raise")
    if raw.duplicated(["candidate_id", "__decision_ts__", "side_name"]).any():
        raise AssertionError(f"{paths}: duplicate candidate identity")
    raw = raw.loc[
        raw.__decision_ts__.ge(month)
        & raw.__decision_ts__.lt(month + pd.offsets.MonthBegin(1))
    ].copy()
    # A candidate ID is a deterministic decision-time identity:
    # ``symbol|side|signal-close``.  The source signal timestamp is one H1
    # bar before the declared decision timestamp.  Do not read target values
    # after the following audit counts are calculated.
    symbol = raw["candidate_id"].astype(str).str.split("|", n=2, expand=True)[0]
    identities = pd.DataFrame({
        "candidate_id": raw["candidate_id"].astype(str),
        "__decision_ts__": raw["__decision_ts__"],
        "side_name": raw["side_name"].astype(str),
        "__ts__": raw["__decision_ts__"] - pd.Timedelta(hours=1),
        "__symbol__": symbol.astype(str),
    })
    if not identities.side_name.eq("long").all():
        raise AssertionError("F72 early-history ledger must be long-only")
    if not identities.__decision_ts__.ge(month).all() or not identities.__decision_ts__.lt(month + pd.offsets.MonthBegin(1)).all():
        raise AssertionError(f"{paths}: rows outside declared calendar month")
    expected_first_decision = month + pd.Timedelta(hours=1)
    if missing_predecessor and identities.__decision_ts__.min() > expected_first_decision:
        raise AssertionError(
            f"{month:%Y-%m}: missing predecessor partition also removed the first decision-time coverage"
        )
    audit = {
        "identity_rows": int(len(identities)),
        "invalid_or_unresolved_target_rows_retained": int((~raw["policy_ordinal_base_valid"].fillna(False).astype(bool)).sum()),
        "null_policy_net_rows_retained": int(pd.to_numeric(raw["policy_net_bps"], errors="coerce").isna().sum()),
        "missing_predecessor_identity_partition": bool(missing_predecessor),
        "missing_predecessor_identity_paths": missing_predecessor,
    }
    return identities, audit


def _candidate_identities_from_grid(
    grid_root: Path,
    month: pd.Timestamp,
) -> tuple[pd.DataFrame, dict[str, int | bool | list[str] | str]]:
    """Read an immutable target-free forward grid without opening labels.

    This is intentionally separate from ``_candidate_identities``: the
    historical identity-only ledger is retained for its existing early-history
    receipts, whereas an append-only forward extension must not even open a
    target-labelled parquet merely to recover candidate IDs.
    """
    source = grid_root / "target_free_candidate_population.parquet"
    manifest_path = grid_root / "run_manifest.json"
    if not manifest_path.exists():
        manifest_path = grid_root / "target_free_candidate_population.manifest.json"
    if not source.exists() or not manifest_path.exists():
        raise FileNotFoundError(f"{grid_root}: expected target-free grid and manifest")
    manifest = json.loads(manifest_path.read_text())
    schema = manifest.get("schema")
    if schema == "strict_r3_canonical_forward_v2_target_free_hourly_grid":
        if manifest.get("future_path_columns_consumed") != []:
            raise AssertionError(f"{grid_root}: forward grid consumed future-path fields")
    elif schema == "strict_r3_recall_target_free_candidate_grid_v1":
        if bool(manifest.get("outcome_fields_read")) or bool(manifest.get("score_fields_read")):
            raise AssertionError(f"{grid_root}: generic grid is not target-free")
    else:
        raise AssertionError(f"{grid_root}: unsupported target-free grid schema {schema!r}")
    columns = ["candidate_id", "__decision_ts__", "side_name", "__ts__", "__symbol__"]
    raw = pd.read_parquet(source, columns=columns)
    raw["__decision_ts__"] = pd.to_datetime(raw["__decision_ts__"], utc=True, errors="raise")
    raw["__ts__"] = pd.to_datetime(raw["__ts__"], utc=True, errors="raise")
    identities = raw.loc[
        raw.__decision_ts__.ge(month)
        & raw.__decision_ts__.lt(month + pd.offsets.MonthBegin(1)),
        columns,
    ].copy()
    if identities.empty:
        raise AssertionError(f"{grid_root}: no target-free identities for {month:%Y-%m}")
    if identities.duplicated(["candidate_id", "__decision_ts__", "side_name"]).any():
        raise AssertionError(f"{grid_root}: duplicate target-free candidate identity")
    if identities.duplicated(["__ts__", "__symbol__"]).any():
        raise AssertionError(f"{grid_root}: duplicate target-free native identity")
    if not identities.side_name.eq("long").all():
        raise AssertionError(f"{grid_root}: F72 forward grid must be long-only")
    if not identities.__decision_ts__.eq(identities.__ts__ + pd.Timedelta(hours=1)).all():
        raise AssertionError(f"{grid_root}: noncanonical target-free decision timestamp")
    return identities, {
        "identity_rows": int(len(identities)),
        "invalid_or_unresolved_target_rows_retained": -1,
        "null_policy_net_rows_retained": -1,
        "missing_predecessor_identity_partition": False,
        "missing_predecessor_identity_paths": [],
        "identity_source": "target_free_grid",
        "outcome_columns_read": False,
    }


def _candidate_identities_from_policy_ledger(
    policy_ledger: Path,
    month: pd.Timestamp,
) -> tuple[pd.DataFrame, dict[str, int | bool | list[str] | str]]:
    """Recover historical candidate identities without opening outcome values.

    The canonical reconciled policy ledger is an outcome *container*, but its
    source manifest explicitly guarantees that its candidate population was
    fixed before the policy replay and retains invalid paths.  This adapter
    deliberately reads only the identity columns; it is therefore suitable
    solely for rebuilding a lost historical target-free feature panel.  It
    must never be used if the ledger's own manifest does not make that
    population guarantee.
    """
    if not policy_ledger.is_file():
        raise FileNotFoundError(policy_ledger)
    manifest_path = policy_ledger.parent / "run_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"{policy_ledger}: missing source manifest")
    manifest = json.loads(manifest_path.read_text())
    candidate_contract = str(manifest.get("candidate_contract", ""))
    if "target-free" not in candidate_contract or "invalid" not in candidate_contract:
        raise AssertionError(
            f"{policy_ledger}: source manifest does not guarantee a target-free, invalid-row-retaining population"
        )
    columns = ("candidate_id", "__decision_ts__", "__symbol__", "side_name")
    raw = pd.read_parquet(policy_ledger, columns=list(columns))
    raw["__decision_ts__"] = pd.to_datetime(raw["__decision_ts__"], utc=True, errors="raise")
    raw = raw.loc[
        raw["__decision_ts__"].ge(month)
        & raw["__decision_ts__"].lt(month + pd.offsets.MonthBegin(1)),
        list(columns),
    ].copy()
    if raw.empty or raw.duplicated(["candidate_id", "__decision_ts__", "side_name"]).any():
        raise AssertionError(f"{policy_ledger}: missing or duplicate historical candidate identity for {month:%Y-%m}")
    if not raw["side_name"].astype(str).str.lower().eq("long").all():
        raise AssertionError(f"{policy_ledger}: early F72 identity source is not long-only")
    identities = raw.assign(__ts__=raw["__decision_ts__"] - pd.Timedelta(hours=1))
    if not identities["__decision_ts__"].eq(identities["__ts__"] + pd.Timedelta(hours=1)).all():
        raise AssertionError(f"{policy_ledger}: noncanonical decision-time identity")
    return identities.loc[:, ["candidate_id", "__decision_ts__", "side_name", "__ts__", "__symbol__"]], {
        "identity_rows": int(len(identities)),
        "invalid_or_unresolved_target_rows_retained": -1,
        "null_policy_net_rows_retained": -1,
        "missing_predecessor_identity_partition": False,
        "missing_predecessor_identity_paths": [],
        "identity_source": "canonical_policy_ledger_identity_only",
        "outcome_columns_read": False,
        "candidate_contract": candidate_contract,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--months", default="2025-01,2025-02,2025-03")
    # Match the retained full-universe F72 source.  Several selected
    # stateful/long-lookback fields have a materially different early-window
    # transient under a 90-day context, so shortening this to save runtime
    # would change Base feature semantics rather than merely reduce history.
    parser.add_argument("--warmup-days", type=int, default=180)
    parser.add_argument("--b-winner", type=Path, default=DEFAULT_B_WINNER)
    parser.add_argument("--router-contract", type=Path, default=DEFAULT_ROUTER_CONTRACT)
    parser.add_argument("--identity-root", type=Path, default=DEFAULT_IDENTITY_ROOT)
    parser.add_argument(
        "--candidate-grid",
        type=Path,
        help=(
            "Immutable target-free forward candidate grid. When supplied, it "
            "replaces the historical label-derived identity ledger for every "
            "requested month."
        ),
    )
    parser.add_argument(
        "--identity-policy-ledger",
        type=Path,
        help=(
            "Canonical reconciled policy ledger used only for historical candidate identity columns. "
            "Its manifest must prove that the population was target-free and retains invalid paths."
        ),
    )
    parser.add_argument(
        "--allow-missing-predecessor-identity-partition",
        action="store_true",
        help=(
            "allow only the prior signal-month identity partition to be absent "
            "for an explicitly declared earliest training-history month; the "
            "current partition must still cover the whole decision-time month"
        ),
    )
    parser.add_argument(
        "--batch-contiguous-months",
        action="store_true",
        help=(
            "materialize one contiguous causal panel for all requested months, "
            "then split target-free identity rows by month.  This avoids "
            "rebuilding the expensive graph for each month."
        ),
    )
    parser.add_argument(
        "--split-shared-panel",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()
    if args.out.exists() and not args.split_shared_panel:
        raise FileExistsError(args.out)
    if args.split_shared_panel and not args.out.exists():
        raise FileNotFoundError(f"{args.out}: shared batch output does not exist")
    if args.warmup_days < 30:
        raise ValueError("warmup-days must be at least 30")
    months = tuple(pd.Timestamp(f"{value.strip()}-01", tz="UTC") for value in args.months.split(",") if value.strip())
    if not months or tuple(sorted(months)) != months or len(set(months)) != len(months):
        raise ValueError("months must be unique chronological YYYY-MM values")
    if sum(item is not None for item in (args.candidate_grid, args.identity_policy_ledger)) > 1:
        raise ValueError("--candidate-grid and --identity-policy-ledger are mutually exclusive")

    def load_identities(month: pd.Timestamp) -> tuple[pd.DataFrame, dict[str, int | bool | list[str] | str]]:
        if args.candidate_grid is not None:
            return _candidate_identities_from_grid(args.candidate_grid, month)
        if args.identity_policy_ledger is not None:
            return _candidate_identities_from_policy_ledger(args.identity_policy_ledger, month)
        return _candidate_identities(
            args.identity_root,
            month,
            allow_missing_predecessor_partition=bool(args.allow_missing_predecessor_identity_partition),
        )

    # ``selected_features`` is the canonical F72 selection schema; accepting
    # ``features`` keeps this research-only producer compatible with a sealed
    # winner file while making the exact consumed key auditable.
    b_fields, b_field_key = _read_fields(args.b_winner, ("selected_features", "features"))
    router_fields, router_field_key = _read_fields(args.router_contract, ("feature_contract",))
    fields = list(dict.fromkeys([*b_fields, *router_fields]))
    generation_fields = list(dict.fromkeys([*fields, *GENERATION_ONLY_DEPENDENCIES]))
    if len(fields) > 120:
        raise AssertionError("minimal F72/router union unexpectedly exceeds 120 fields")
    args.out.mkdir(parents=True, exist_ok=args.split_shared_panel)
    manifest = {
        "schema": "strict_r3_f72_early_router_feature_universe_v1",
        "scope": "offline research-only target-free early-history feature materialisation; no live, inference, admission, portfolio, execution, or exchange mutation",
        "candidate_contract": (
            "immutable target-free forward grid; no outcome column is opened"
            if args.candidate_grid is not None
            else "canonical reconciled policy ledger identity-only; source manifest proves target-free candidates and invalid-path retention; no outcome column is opened"
            if args.identity_policy_ledger is not None
            else "B0 target-label source is identity-only; invalid and unresolved target rows are retained, and no target/outcome column enters feature materialisation"
        ),
        "candidate_grid": str(args.candidate_grid) if args.candidate_grid is not None else None,
        "identity_policy_ledger": str(args.identity_policy_ledger) if args.identity_policy_ledger is not None else None,
        "feature_contract": fields,
        "feature_contract_sha256": hashlib.sha256("\n".join(fields).encode()).hexdigest(),
        "f72_b_winner": {
            "path": str(args.b_winner), "sha256": _sha(args.b_winner),
            "feature_key": b_field_key, "fields": len(b_fields),
        },
        "router_contract": {
            "path": str(args.router_contract), "sha256": _sha(args.router_contract),
            "feature_key": router_field_key, "fields": len(router_fields),
        },
        "generation_only_dependencies": list(GENERATION_ONLY_DEPENDENCIES),
        "generation_dependencies": list(FROZEN_GENERATION_DEPENDENCIES),
        "canonical_panel_materialiser": str(CANONICAL_PANEL_MATERIALISER),
        "canonical_panel_materialiser_sha256": _sha(CANONICAL_PANEL_MATERIALISER),
        "source_precedence_contract": SOURCE_PRECEDENCE_CONTRACT,
        "warmup_days": int(args.warmup_days),
        "months": [f"{month:%Y-%m}" for month in months],
        "allow_missing_predecessor_identity_partition": bool(args.allow_missing_predecessor_identity_partition),
    }
    manifest_path = args.out / "run_manifest.json"
    if args.split_shared_panel:
        existing_manifest = json.loads(manifest_path.read_text())
        if existing_manifest.get("feature_contract_sha256") != manifest["feature_contract_sha256"]:
            raise AssertionError("shared batch feature contract differs from split contract")
        manifest = existing_manifest
    else:
        with manifest_path.open("x", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
    audits: list[dict[str, Any]] = []
    contract = {"long": generation_fields, "short": []}
    if args.batch_contiguous_months:
        expected = tuple(pd.date_range(months[0], months[-1], freq="MS", tz="UTC"))
        if months != expected:
            raise ValueError("--batch-contiguous-months requires contiguous calendar months")
        identities_by_month: dict[pd.Timestamp, pd.DataFrame] = {}
        audit_by_month: dict[pd.Timestamp, dict[str, int]] = {}
        for month in months:
            identities, audit = load_identities(month)
            identities_by_month[month] = identities
            audit_by_month[month] = audit
        all_identities = pd.concat(list(identities_by_month.values()), ignore_index=True)
        if all_identities.duplicated(["candidate_id", "__decision_ts__", "side_name"]).any():
            raise AssertionError("batched early history has duplicate candidate identities")
        if all_identities.duplicated(["__ts__", "__symbol__"]).any():
            raise AssertionError("batched early history cannot map identities one-to-one to causal feature keys")
        start = months[0] - pd.Timedelta(days=args.warmup_days)
        end = months[-1] + pd.offsets.MonthBegin(1)
        shared_root = args.out / "shared_contiguous_causal_panel"
        shared_path = shared_root / "causal_feature_universe.parquet"
        if not args.split_shared_panel:
            generated_path = materialize_features(
                shared_root,
                all_identities,
                contract,
                start,
                end,
                full_feature_universe=False,
            )
            os.replace(generated_path, shared_path)
            # ``materialize_features`` has a deliberately broad, expensive
            # graph.  Re-exec before the cheap split so its peak allocations
            # cannot coexist with the parquet read/merge.  This is a memory
            # optimisation only; the materialised panel and all causal inputs
            # are immutable and unchanged.
            manifest["materialization_mode"] = "single_contiguous_causal_panel"
            manifest["shared_context_start"] = start.isoformat()
            manifest["shared_context_end_exclusive"] = end.isoformat()
            manifest["split_state"] = "pending_fresh_process"
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
            os.execv(
                sys.executable,
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--out", str(args.out),
                    "--months", args.months,
                    "--warmup-days", str(args.warmup_days),
                    "--b-winner", str(args.b_winner),
                    "--router-contract", str(args.router_contract),
                    "--identity-root", str(args.identity_root),
                    *(["--candidate-grid", str(args.candidate_grid)] if args.candidate_grid is not None else []),
                    *(["--identity-policy-ledger", str(args.identity_policy_ledger)] if args.identity_policy_ledger is not None else []),
                    "--batch-contiguous-months",
                    "--split-shared-panel",
                ],
            )
        if not shared_path.exists():
            raise FileNotFoundError(f"{shared_path}: missing shared causal panel")
        generated = pd.read_parquet(shared_path, columns=["__ts__", "__symbol__", *fields])
        native_identity = ["__ts__", "__symbol__"]
        if any(column not in generated.columns for column in native_identity):
            raise AssertionError("batched causal generator dropped its native identity")
        for month in months:
            identities = identities_by_month[month]
            restored = identities.loc[:, ["candidate_id", "__decision_ts__", "side_name", *native_identity]].merge(
                generated,
                on=native_identity,
                how="inner",
                validate="one_to_one",
            )
            if len(restored) != len(identities) or restored.candidate_id.duplicated().any():
                raise AssertionError(f"{month:%Y-%m}: batched causal generator changed target-free identities")
            month_root = args.out / f"month={month:%Y-%m}"
            month_root.mkdir()
            feature_path = month_root / "causal_feature_universe.parquet"
            restored.to_parquet(feature_path, index=False, compression="zstd")
            audit = audit_by_month[month]
            audit.update({
                "month": f"{month:%Y-%m}",
                "context_start": start.isoformat(),
                "context_end_exclusive": end.isoformat(),
                "feature_rows": int(len(restored)),
                "feature_columns": int(len(restored.columns)),
                "target_fields_in_output": False,
                "materialization_mode": "single_contiguous_causal_panel",
            })
            audits.append(audit)
            print(json.dumps({"event": "month_complete", **audit}), flush=True)
        manifest["materialization_mode"] = "single_contiguous_causal_panel"
        manifest["shared_context_start"] = start.isoformat()
        manifest["shared_context_end_exclusive"] = end.isoformat()
        manifest["split_state"] = "complete"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        pd.DataFrame(audits).to_parquet(args.out / "identity_and_coverage_audit.parquet", index=False, compression="zstd")
        return
    for month in months:
        identities, audit = load_identities(month)
        month_root = args.out / f"month={month:%Y-%m}"
        start = month - pd.Timedelta(days=args.warmup_days)
        end = month + pd.offsets.MonthBegin(1)
        feature_path = materialize_features(
            month_root, identities, contract, start, end, full_feature_universe=False,
        )
        # The existing F72/router readers use this canonical target-free file
        # name.  The underlying producer was deliberately asked only for the
        # frozen union above, rather than its 1,400-field research universe.
        canonical_path = month_root / "causal_feature_universe.parquet"
        os.replace(feature_path, canonical_path)
        feature_path = canonical_path
        # ``materialize_features`` intentionally writes only its native
        # timestamp/symbol key.  Restore the declared candidate identity from
        # the target-free, identity-only ledger before this panel is exposed
        # to the strict OOF scorer.  There is exactly one long candidate for
        # each (signal timestamp, symbol) in this contract; a one-to-one merge
        # makes any unexpected universe multiplication a hard failure.
        generated = pd.read_parquet(feature_path, columns=["__ts__", "__symbol__", *fields])
        native_identity = ["__ts__", "__symbol__"]
        if any(column not in generated.columns for column in native_identity):
            raise AssertionError(f"{month:%Y-%m}: feature generator dropped its native identity")
        restored = identities.loc[:, ["candidate_id", "__decision_ts__", "side_name", *native_identity]].merge(
            generated,
            on=native_identity,
            how="inner",
            validate="one_to_one",
        )
        if len(restored) != len(identities):
            raise AssertionError(f"{month:%Y-%m}: causal generator did not cover every target-free identity")
        if restored.candidate_id.duplicated().any():
            raise AssertionError(f"{month:%Y-%m}: restored feature panel has duplicate candidate IDs")
        restored.to_parquet(feature_path, index=False, compression="zstd")
        source = restored.loc[:, ["candidate_id", "__decision_ts__", "side_name"]].copy()
        source["__decision_ts__"] = pd.to_datetime(source["__decision_ts__"], utc=True, errors="raise")
        if len(source) != len(identities) or set(source.candidate_id) != set(identities.candidate_id):
            raise AssertionError(f"{month:%Y-%m}: target-free feature identity mismatch")
        audit.update({
            "month": f"{month:%Y-%m}",
            "context_start": start.isoformat(),
            "context_end_exclusive": end.isoformat(),
            "feature_rows": int(len(source)),
            "feature_columns": int(len(pd.read_parquet(feature_path, columns=None).columns)),
            "target_fields_in_output": False,
            "materialization_mode": "per_month_causal_panel",
        })
        audits.append(audit)
        print(json.dumps({"event": "month_complete", **audit}), flush=True)
    pd.DataFrame(audits).to_parquet(args.out / "identity_and_coverage_audit.parquet", index=False, compression="zstd")


if __name__ == "__main__":
    main()
