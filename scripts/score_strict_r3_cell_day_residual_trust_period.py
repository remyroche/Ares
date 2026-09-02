#!/usr/bin/env python3
"""Score a side-local period with monthly strict-OOS R5 trust bundles.

Each bundle is fitted at a monthly cutoff, then scores only that following
calendar month.  Causal EV-map provenance is joined by candidate identity;
unmapped rows are retained with a false admission flag instead of borrowing a
different bundle or being removed from the candidate population.
"""

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

from extreme_price_movements.strict_r3_cell_day_trust import (  # noqa: E402
    MAP_FIELD,
    load_cell_day_residual_trust_bundle,
)


SOURCE_MAP_FIELD = "causal_21d_side_expected_net_bps"
MAP_COLUMNS = (
    "candidate_id", "__decision_ts__", SOURCE_MAP_FIELD,
    "causal_21d_side_mapping_status",
)
BASE_COLUMNS = (
    "candidate_id", "__decision_ts__", "side_name", "policy_path_valid",
    "policy_net_bps", "policy_label_available_ts", "final_score",
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc(value: object) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prequential-ledger", type=Path, required=True)
    parser.add_argument("--cell-day-provenance", type=Path, required=True)
    parser.add_argument("--bundle-dir", type=Path, action="append", required=True)
    parser.add_argument("--side", choices=("long", "short"), required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable output exists: {args.out_dir}")

    bundles = [load_cell_day_residual_trust_bundle(path) for path in args.bundle_dir]
    if not bundles:
        raise ValueError("at least one R5 bundle is required")
    if any(bundle.side != args.side for bundle in bundles):
        raise ValueError("R5 bundle side does not match --side")
    fields = bundles[0].fields
    if any(bundle.fields != fields for bundle in bundles[1:]):
        raise ValueError("monthly R5 bundles have different frozen feature contracts")
    cutoffs = [_utc(bundle.cutoff) for bundle in bundles]
    if len(set(cutoffs)) != len(cutoffs):
        raise ValueError("duplicate monthly R5 cutoff")
    ordered = sorted(zip(cutoffs, bundles, args.bundle_dir), key=lambda item: item[0])

    ledger_columns = list(dict.fromkeys([*BASE_COLUMNS, *fields]))
    ledger = pd.read_parquet(args.prequential_ledger, columns=ledger_columns)
    mapped = pd.read_parquet(args.cell_day_provenance, columns=list(MAP_COLUMNS))
    if ledger["candidate_id"].duplicated().any() or mapped["candidate_id"].duplicated().any():
        raise ValueError("R5 period scorer requires unique candidate identities")
    observed = ledger["side_name"].astype(str).str.lower()
    if not observed.eq(args.side).all():
        raise ValueError("prequential ledger is not side-local")
    ledger["__decision_ts__"] = pd.to_datetime(ledger["__decision_ts__"], utc=True, errors="raise")
    ledger["policy_label_available_ts"] = pd.to_datetime(
        ledger["policy_label_available_ts"], utc=True, errors="raise",
    )
    mapped["__decision_ts__"] = pd.to_datetime(mapped["__decision_ts__"], utc=True, errors="raise")
    joined = ledger.merge(
        mapped.rename(columns={
            "__decision_ts__": "__map_decision_ts__",
            SOURCE_MAP_FIELD: MAP_FIELD,
        }),
        on="candidate_id", how="left", validate="one_to_one",
    )
    overlap = joined["__map_decision_ts__"].notna()
    if not joined.loc[overlap, "__decision_ts__"].eq(joined.loc[overlap, "__map_decision_ts__"]).all():
        raise ValueError("R5 period scorer map candidate/timestamp mismatch")

    parts: list[pd.DataFrame] = []
    audit_rows: list[dict[str, object]] = []
    for cutoff, bundle, directory in ordered:
        end = cutoff + pd.offsets.MonthBegin(1)
        held = joined.loc[
            joined["__decision_ts__"].ge(cutoff)
            & joined["__decision_ts__"].lt(end),
        ].copy()
        if held.empty:
            continue
        output = held.loc[:, list(dict.fromkeys([
            *BASE_COLUMNS, MAP_FIELD, "causal_21d_side_mapping_status",
        ]))].copy()
        output["r5_bundle_cutoff"] = cutoff
        output["r5_bundle_dir"] = str(directory)
        output["r5_bundle_sha256"] = str(bundle.manifest.get("bundle_sha256", ""))
        output["trust_posterior_expected_bps"] = np.nan
        output["trust_posterior_predictive_q10_bps"] = np.nan
        output["trust_p_ev_positive"] = np.nan
        output["trust_p_adverse_200bps"] = np.nan
        output["trust_effective_support"] = np.nan
        output["trust_residual_q25_bps"] = np.nan
        output["trust_p_map_overestimate_100bps"] = np.nan
        output["trust_risk_corroborated"] = False
        output["trust_authority"] = 0.0
        output["trust_corrected_expected_net_bps"] = np.nan
        output["auction_rank_adjustment_bps"] = np.nan
        output["trust_posterior_admitted_ge_50bps"] = False
        mapped_mask = np.isfinite(pd.to_numeric(held[MAP_FIELD], errors="coerce"))
        if mapped_mask.any():
            scored = bundle.score(held.loc[mapped_mask].copy())
            scored = scored.set_index("candidate_id")
            positions = output.index[mapped_mask]
            for column in scored.columns:
                if column == "candidate_id":
                    continue
                output.loc[positions, column] = scored.reindex(
                    output.loc[positions, "candidate_id"],
                )[column].to_numpy()
            posterior = pd.to_numeric(output["trust_posterior_expected_bps"], errors="coerce")
            output["trust_posterior_admitted_ge_50bps"] = posterior.ge(50.0).fillna(False)
        output["trust_missing_fail_closed"] = ~mapped_mask.to_numpy()
        parts.append(output)
        audit_rows.append({
            "cutoff": cutoff,
            "end_exclusive": end,
            "rows": int(len(output)),
            "mapped_rows": int(mapped_mask.sum()),
            "unmapped_rows_fail_closed": int((~mapped_mask).sum()),
            "posterior_admitted_rows": int(output["trust_posterior_admitted_ge_50bps"].sum()),
            "bundle_dir": str(directory),
            "bundle_sha256": str(bundle.manifest.get("bundle_sha256", "")),
        })
    if not parts:
        raise ValueError("no held rows overlap the supplied R5 monthly bundles")
    result = pd.concat(parts, ignore_index=True).sort_values(
        ["__decision_ts__", "candidate_id"], kind="stable",
    )
    if result["candidate_id"].duplicated().any():
        raise AssertionError("R5 monthly scoring assigned a candidate twice")
    args.out_dir.mkdir(parents=True)
    result.to_parquet(args.out_dir / "short_r5_oof_predictions.parquet", index=False, compression="zstd")
    audit = pd.DataFrame(audit_rows)
    audit.to_parquet(args.out_dir / "short_r5_oof_audit.parquet", index=False)
    manifest = {
        "schema": "strict_r3_short_r5_monthly_oof_score_v1",
        "side": args.side,
        "prequential_ledger": str(args.prequential_ledger),
        "prequential_ledger_sha256": _sha(args.prequential_ledger),
        "cell_day_provenance": str(args.cell_day_provenance),
        "cell_day_provenance_sha256": _sha(args.cell_day_provenance),
        "field_count": len(fields),
        "bundle_count": len(ordered),
        "rows": int(len(result)),
        "mapped_rows": int((~result["trust_missing_fail_closed"]).sum()),
        "unmapped_rows_fail_closed": int(result["trust_missing_fail_closed"].sum()),
        "posterior_admitted_rows": int(result["trust_posterior_admitted_ge_50bps"].sum()),
        "contract": "one monthly bundle scores only its following held month; R5 missing map is fail closed",
    }
    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"event": "complete", "out_dir": str(args.out_dir), **manifest}))


if __name__ == "__main__":
    main()
