#!/usr/bin/env python3
"""Publish May--July exact-policy rows in the canonical mapping schema.

The existing recent-EV artifact supplies the frozen raw score and its strict
outer-OOF/frozen-forward provenance.  Its published EV map was calibrated on
an older outcome ledger and is therefore deliberately discarded here.  The
raw score is remapped once per UTC day against only corrected exact-policy
outcomes that resolved during the previous 21 days.  This makes the resulting
common-unit score suitable for the same global-book before/after materializer
used by the February--April research panel.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MAPPED = ROOT / (
    "data_perp/artifacts/execution_ev_context_clean_recent_mapping_forward_"
    "july19_20260726_v1/mapped_oof.parquet"
)
DEFAULT_MAPPING_REPORT = DEFAULT_MAPPED.parent / "report.json"
DEFAULT_EXACT_POLICY = ROOT / (
    "data_perp/artifacts/execution_ev_canonical_exact_policy_regime_input_"
    "20260727_v3/joined.parquet"
)
DEFAULT_OUTPUT = ROOT / (
    "data_perp/artifacts/current_exact_policy_global_book_mapping_source_"
    "20260730_v3"
)

SCHEMA = "causal_score_economics_conversion_mapping_v1"
IDENTITY = ("candidate_id", "__ts__", "__symbol__", "side_name")
RAW_SCORE = "catboost__residual__without_hpo__all_features"
MAPPED_SCORE = "causal_recent_side_isotonic_ev"
GLOBAL_MAPPED_SCORE = "causal_recent_isotonic_ev"
OOF_FLAG = f"{MAPPED_SCORE}__is_oof"
FORWARD_FLAG = f"{MAPPED_SCORE}__is_forward_oos"
WINDOW_DAYS = 21
MINIMUM_REFERENCE_ROWS = 500


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe(value: Any) -> Any:
    if isinstance(value, (Path, pd.Timestamp, pd.Timedelta)):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _normalise_identity(frame: pd.DataFrame, *, name: str) -> pd.DataFrame:
    missing = sorted(set(IDENTITY).difference(frame.columns))
    if missing:
        raise ValueError(f"{name} lacks exact identity fields: {missing}")
    result = frame.copy()
    result["candidate_id"] = result["candidate_id"].astype(str)
    result["__ts__"] = pd.to_datetime(
        result["__ts__"], utc=True, errors="raise"
    )
    result["__symbol__"] = result["__symbol__"].astype(str)
    result["side_name"] = result["side_name"].astype(str).str.lower()
    if not result["side_name"].isin(("long", "short")).all():
        raise ValueError(f"{name} contains noncanonical sides")
    if result.duplicated(list(IDENTITY), keep=False).any():
        raise ValueError(f"{name} contains duplicate candidate identities")
    return result


def _normalise_exit_class(values: pd.Series) -> pd.Series:
    result = values.astype(str).str.lower().replace({"full_sl": "full_stop"})
    allowed = {"trailing", "timeout", "full_stop", "adverse_exit"}
    invalid = sorted(set(result.unique()).difference(allowed))
    if invalid:
        raise ValueError(f"unsupported current exit reasons: {invalid}")
    return result


def _load_exact_policy_economics(source: pd.DataFrame) -> pd.DataFrame:
    required = {
        *IDENTITY,
        "execution_decision_utc",
        "execution_label_end_utc",
        "execution_gross_ev_12h",
        "execution_cost_return",
        "execution_net_ev_12h",
        "execution_mfe_return_12h",
        "execution_mae_return_12h",
        "execution_exit_reason",
        "execution_exit_hour",
    }
    missing = sorted(required.difference(source.columns))
    if missing:
        raise ValueError(f"exact-policy source lacks exact economics: {missing}")
    result = _normalise_identity(
        source.loc[:, sorted(required)], name="exact-policy source"
    )
    for column in ("execution_decision_utc", "execution_label_end_utc"):
        result[column] = pd.to_datetime(
            result[column], utc=True, errors="raise"
        )
    return result


def _reference_audit(
    population: pd.DataFrame,
    *,
    window_days: int,
    minimum_reference_rows: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reconstruct the mapper's resolved-before-day support metadata."""

    result = population.copy()
    result["map_reference_rows"] = 0
    result["map_side_reference_rows"] = 0
    result["map_cell_reference_rows"] = 0
    decision = result["execution_decision_utc"]
    resolved = result["execution_label_end_utc"]
    raw = result[RAW_SCORE].to_numpy(float)
    target = result["execution_net_ev_12h"].to_numpy(float)
    sides = result["side_name"].to_numpy(str)
    audits: list[dict[str, Any]] = []
    for snapshot, indices in result.groupby(decision.dt.floor("D"), sort=True).groups.items():
        snapshot = pd.Timestamp(snapshot)
        positions = result.index.get_indexer(indices)
        reference = (
            resolved.lt(snapshot)
            & resolved.ge(snapshot - pd.Timedelta(days=int(window_days)))
            & np.isfinite(raw)
            & np.isfinite(target)
        ).to_numpy()
        reference_positions = np.flatnonzero(reference)
        reference_rows = int(len(reference_positions))
        reference_max = (
            resolved.iloc[reference_positions].max()
            if reference_rows
            else pd.NaT
        )
        side_counts = {
            side: int((sides[reference_positions] == side).sum())
            for side in ("long", "short")
        }
        result.iloc[
            positions, result.columns.get_loc("map_reference_rows")
        ] = reference_rows
        for side in ("long", "short"):
            local = positions[sides[positions] == side]
            result.iloc[
                local, result.columns.get_loc("map_side_reference_rows")
            ] = side_counts[side]
            # The authoritative current mapper has global and side-local
            # isotonic cells; its finest cell is therefore the side.
            result.iloc[
                local, result.columns.get_loc("map_cell_reference_rows")
            ] = side_counts[side]
        audits.append(
            {
                "snapshot_utc": snapshot,
                "reference_window_start_utc": snapshot
                - pd.Timedelta(days=int(window_days)),
                "reference_window_end_utc": snapshot,
                "reference_rows": reference_rows,
                "long_reference_rows": side_counts["long"],
                "short_reference_rows": side_counts["short"],
                "reference_label_end_max_utc": reference_max,
                "current_rows": int(len(positions)),
                "mapping_available": reference_rows
                >= int(minimum_reference_rows),
            }
        )
    return result, pd.DataFrame.from_records(audits)


def build_current_mapping_source(
    mapped: pd.DataFrame,
    exact_policy: pd.DataFrame,
    mapping_report: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    required = {
        *IDENTITY,
        RAW_SCORE,
        OOF_FLAG,
        FORWARD_FLAG,
        "promotion_eligible",
        "evaluation_origin",
    }
    missing = sorted(required.difference(mapped.columns))
    if missing:
        raise ValueError(f"mapped source lacks current-lineage fields: {missing}")
    work = _normalise_identity(mapped.loc[:, sorted(required)], name="mapped source")
    oof = work[OOF_FLAG].fillna(False).astype(bool)
    forward = work[FORWARD_FLAG].fillna(False).astype(bool)
    if (oof & forward).any():
        raise ValueError("mapped OOF and forward flags overlap")
    unflagged = ~(oof | forward)
    if (
        unflagged
        & ~work["evaluation_origin"].astype(str).eq("historical_outer_oof")
    ).any():
        raise ValueError(
            "only historical mapping-warmup rows may lack a mapping flag"
        )
    if work.loc[forward, "promotion_eligible"].fillna(False).astype(bool).any():
        raise ValueError("forward-OOS mapping rows must remain nonpromotable")

    economics = _load_exact_policy_economics(exact_policy)
    merged = work.merge(
        economics,
        on=list(IDENTITY),
        how="left",
        validate="one_to_one",
    )
    if merged["execution_cost_return"].isna().any():
        raise ValueError("rich handoffs do not cover every mapped candidate")
    if not merged["execution_decision_utc"].eq(
        merged["__ts__"] + pd.Timedelta(hours=1)
    ).all():
        raise ValueError("exact policy violates source-to-decision timing")
    if not merged["execution_label_end_utc"].eq(
        merged["execution_decision_utc"] + pd.Timedelta(hours=12)
    ).all():
        raise ValueError("exact policy violates the 12-hour resolution contract")
    for column in (
        RAW_SCORE,
        "execution_gross_ev_12h",
        "execution_cost_return",
        "execution_net_ev_12h",
    ):
        merged[column] = pd.to_numeric(merged[column], errors="coerce")
    if not np.allclose(
        merged["execution_gross_ev_12h"] - merged["execution_cost_return"],
        merged["execution_net_ev_12h"],
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("current spread-aware gross-cost-net accounting fails")
    merged["execution_exit_class"] = _normalise_exit_class(
        merged["execution_exit_reason"]
    )
    # Refit only the causal score-to-EV map.  The raw score and all model
    # predictions remain frozen; corrected exact-policy outcomes replace the
    # semantically obsolete legacy mapping target.
    try:
        from scripts.run_execution_ev_recent_mapping_ablation import (
            causal_mappings,
        )
    except ModuleNotFoundError:
        from run_execution_ev_recent_mapping_ablation import causal_mappings

    merged, mapping_audit = causal_mappings(
        merged.reset_index(drop=True),
        score_col=RAW_SCORE,
        window_days=WINDOW_DAYS,
        min_reference_rows=MINIMUM_REFERENCE_ROWS,
        side_support_target=500.0,
    )
    merged["mapped_eligible"] = (
        np.isfinite(merged[MAPPED_SCORE])
        & (oof.to_numpy() | forward.to_numpy())
    )
    merged["mapped_direct_net"] = merged[MAPPED_SCORE]
    merged["mapped_global_direct_net"] = merged[GLOBAL_MAPPED_SCORE]
    merged["candidate_month"] = merged[
        "execution_decision_utc"
    ].dt.strftime("%Y-%m")
    merged["opportunity_gross_above_cost_0bps"] = (
        merged["execution_gross_ev_12h"]
        > merged["execution_cost_return"]
    ).astype(float)
    merged["opportunity_gross_above_cost_25bps"] = (
        merged["execution_gross_ev_12h"]
        > merged["execution_cost_return"] + 0.0025
    ).astype(float)
    merged, audit = _reference_audit(
        merged,
        window_days=WINDOW_DAYS,
        minimum_reference_rows=MINIMUM_REFERENCE_ROWS,
    )
    available = merged["mapped_eligible"]
    if not merged.loc[available, "map_reference_rows"].ge(
        MINIMUM_REFERENCE_ROWS
    ).all():
        raise ValueError("authoritative mapped rows lack required causal support")

    contract = dict(mapping_report.get("contract", {}))
    if (
        int(contract.get("window_days", -1)) != WINDOW_DAYS
        or int(contract.get("min_reference_rows", -1))
        != MINIMUM_REFERENCE_ROWS
        or bool(contract.get("per_timestamp_quota", True))
        or contract.get("ranking_scope")
        != "global pooled across timestamps and sides"
    ):
        raise ValueError("current mapping report contract changed")
    report_audit = {
        pd.Timestamp(row["snapshot"]): row
        for row in mapping_report.get("daily_audit", [])
    }
    for row in audit.itertuples(index=False):
        expected = report_audit.get(pd.Timestamp(row.snapshot_utc))
        if expected is None:
            if bool(row.mapping_available):
                raise ValueError("mapping report omits an available snapshot")
            continue
        if int(expected["reference_rows"]) != int(row.reference_rows):
            raise ValueError("reconstructed global mapping support differs")
        if int(expected["long_reference_rows"]) != int(row.long_reference_rows):
            raise ValueError("reconstructed long mapping support differs")
        if int(expected["short_reference_rows"]) != int(row.short_reference_rows):
            raise ValueError("reconstructed short mapping support differs")
    recomputed_audit = {
        pd.Timestamp(row["snapshot"]): row for row in mapping_audit
    }
    for row in audit.itertuples(index=False):
        expected = recomputed_audit.get(pd.Timestamp(row.snapshot_utc))
        if bool(row.mapping_available) != (expected is not None):
            raise ValueError("exact-policy mapper availability differs from audit")
        if expected is not None and int(expected["reference_rows"]) != int(
            row.reference_rows
        ):
            raise ValueError("exact-policy mapper support differs from audit")

    output_columns = [
        "candidate_id",
        "__symbol__",
        "side_name",
        "__ts__",
        "execution_decision_utc",
        "execution_label_end_utc",
        "candidate_month",
        "mapped_eligible",
        "mapped_direct_net",
        "mapped_global_direct_net",
        "map_reference_rows",
        "map_side_reference_rows",
        "map_cell_reference_rows",
        "execution_gross_ev_12h",
        "execution_cost_return",
        "execution_net_ev_12h",
        "execution_mfe_return_12h",
        "execution_mae_return_12h",
        "execution_exit_class",
        "execution_exit_hour",
        "opportunity_gross_above_cost_0bps",
        "opportunity_gross_above_cost_25bps",
        RAW_SCORE,
        GLOBAL_MAPPED_SCORE,
        MAPPED_SCORE,
        OOF_FLAG,
        FORWARD_FLAG,
        "promotion_eligible",
        "evaluation_origin",
    ]
    result = merged.loc[:, output_columns].sort_values(
        ["execution_decision_utc", "candidate_id"], kind="stable"
    ).reset_index(drop=True)
    stats = {
        "input_rows": int(len(result)),
        "mapped_eligible_rows": int(result["mapped_eligible"].sum()),
        "warmup_unmapped_rows": int((~result["mapped_eligible"]).sum()),
        "strict_oof_score_lineage_rows": int(
            result["evaluation_origin"].eq("historical_outer_oof").sum()
        ),
        "strict_mapped_oof_rows": int(result[OOF_FLAG].sum()),
        "forward_oos_rows": int(result[FORWARD_FLAG].sum()),
        "mapped_start_utc": result.loc[
            result["mapped_eligible"], "execution_decision_utc"
        ].min(),
        "mapped_end_utc": result.loc[
            result["mapped_eligible"], "execution_decision_utc"
        ].max(),
    }
    return result, audit, stats


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite immutable output {output}")
    source_paths = {
        "mapped": Path(args.mapped),
        "mapping_report": Path(args.mapping_report),
        "exact_policy": Path(args.exact_policy),
    }
    report = json.loads(
        source_paths["mapping_report"].read_text(encoding="utf-8")
    )
    mapped, audit, stats = build_current_mapping_source(
        pd.read_parquet(source_paths["mapped"]),
        pd.read_parquet(source_paths["exact_policy"]),
        report,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(dir=output.parent, prefix=f".{output.name}.")
    )
    mapped_path = temporary / "causal_mapped_candidates.parquet"
    audit_path = temporary / "causal_snapshot_audit.parquet"
    mapped.to_parquet(mapped_path, index=False, compression="zstd")
    audit.to_parquet(audit_path, index=False, compression="zstd")
    manifest = {
        "schema": SCHEMA,
        "status": "CURRENT_LINEAGE_OOF_AND_FROZEN_FORWARD_MAPPING_BOUND",
        "lineage": (
            "May--July 2026 frozen current execution-EV raw score remapped "
            "against corrected exact-policy outcomes; strict outer OOF "
            "followed by frozen final-fit forward OOS"
        ),
        "population_audit": stats,
        "causal_contract": {
            "window_days": WINDOW_DAYS,
            "minimum_reference_rows": MINIMUM_REFERENCE_ROWS,
            "reference_rule": "execution_label_end_utc < snapshot",
            "snapshot_frequency": "UTC day",
            "score": RAW_SCORE,
            "mapping": {
                "global": GLOBAL_MAPPED_SCORE,
                "side_shrunk": MAPPED_SCORE,
                "canonical_existing_alias": "mapped_direct_net",
                "global_alias": "mapped_global_direct_net",
            },
            "legacy_mapping_reused": False,
            "exact_policy_target_remap": True,
        },
        "selection_contract": {
            "primary": "one pooled global top-k",
            "not_per_timestamp": True,
            "not_per_side": True,
            "tie_break": "candidate_id ascending",
        },
        "economics_contract": {
            "target": "exact deployed spread-aware policy execution_net_ev_12h",
            "identity_join": list(IDENTITY),
            "gross_minus_cost_equals_net": True,
            "full_sl_normalized_to": "full_stop",
        },
        "promotion_contract": {
            "strict_oof_rows": "eligible diagnostic history",
            "forward_oos_rows": "frozen and nonpromotable",
            "combined_use": (
                "source-separated regime-transition research only; never "
                "misrepresented as one strict-OOF promotion panel"
            ),
        },
        "source_sha256": {
            name: {"path": str(path), "sha256": sha256(path)}
            for name, path in source_paths.items()
        },
        "outputs": {
            "mapped": {
                "path": "causal_mapped_candidates.parquet",
                "sha256": sha256(mapped_path),
            },
            "audit": {
                "path": "causal_snapshot_audit.parquet",
                "sha256": sha256(audit_path),
            },
        },
    }
    (temporary / "manifest.json").write_text(
        json.dumps(_safe(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (temporary / "manifest.sha256").write_text(
        f"{sha256(temporary / 'manifest.json')}  manifest.json\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
    return {"output": str(output), **stats}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--mapped", type=Path, default=DEFAULT_MAPPED)
    result.add_argument(
        "--mapping-report", type=Path, default=DEFAULT_MAPPING_REPORT
    )
    result.add_argument(
        "--exact-policy", type=Path, default=DEFAULT_EXACT_POLICY
    )
    result.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return result


def main() -> None:
    print(json.dumps(_safe(run(parser().parse_args())), sort_keys=True))


if __name__ == "__main__":
    main()
