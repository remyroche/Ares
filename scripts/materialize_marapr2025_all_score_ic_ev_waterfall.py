#!/usr/bin/env python3
"""Materialize the exact Mar--Apr 2025 all-score IC-to-EV waterfall.

This is deliberately a bounded, source-separated diagnostic.  It intersects
the canonical residual/current-spread exact-policy ledger with the historical
OOF direct q25 head on their complete four-field identity.  It neither reads
nor emits a mapped direct score, changes a model/policy, nor makes a promotion
decision.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from scripts.materialize_source_separated_ic_ev_waterfall import (
    IDENTITY_COLUMNS,
    cutoff_ties,
    fixed_composition,
    full_ic,
    response_20bin,
    safe,
    score_columns,
    score_compression,
    sha256,
    tail_metrics,
    validate_source,
    write_json,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LEDGER = ROOT / (
    "data_perp/artifacts/historical_score_economics_conversion_ledgers_20260729_v1/"
    "ledgers/canonical_residual_exact1m_current_spread_cf.parquet"
)
DEFAULT_LEDGER_MANIFEST = ROOT / (
    "data_perp/artifacts/historical_score_economics_conversion_ledgers_20260729_v1/"
    "manifest.json"
)
DEFAULT_DIRECT = ROOT / (
    "data_perp/artifacts/cross_era_direct_net_quantile_challenger_20260730_v1/"
    "historical_oof_winner.parquet"
)
DEFAULT_DIRECT_MANIFEST = ROOT / (
    "data_perp/artifacts/cross_era_direct_net_quantile_challenger_20260730_v1/"
    "manifest.json"
)
DEFAULT_OUTPUT = ROOT / (
    "data_perp/artifacts/marapr2025_all_score_ic_ev_waterfall_20260730_v1"
)

SOURCE_FAMILY = "marapr2025_canonical_residual_plus_raw_direct_q25"
MARCH_START = pd.Timestamp("2025-03-01T00:00:00Z")
MAY_START = pd.Timestamp("2025-05-01T00:00:00Z")
EXPECTED_ROWS = 140_682
DIRECT_BPS_COLUMN = "score_direct_q25_bps"
DIRECT_RETURN_COLUMN = "direct_q25_return"
DIRECT_NET_COLUMN = "direct_source_execution_net_ev_12h"
DIRECT_LABEL_END_COLUMN = "direct_label_resolution_utc"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _identity(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    missing = sorted(set(IDENTITY_COLUMNS).difference(frame.columns))
    if missing:
        raise ValueError(f"{source} missing identity columns: {missing}")
    result = frame.copy()
    result["candidate_id"] = result["candidate_id"].astype(str)
    result["side_name"] = result["side_name"].astype(str)
    result["__symbol__"] = result["__symbol__"].astype(str)
    result["__ts__"] = pd.to_datetime(result["__ts__"], utc=True, errors="raise")
    if result.duplicated(list(IDENTITY_COLUMNS)).any():
        raise ValueError(f"{source} has duplicate four-field identities")
    return result


def _require_ledger_manifest(path: Path, ledger_path: Path) -> dict[str, Any]:
    manifest = _read_json(path)
    if manifest.get("schema") != "historical_score_economics_conversion_ledgers_v1":
        raise ValueError("requires historical conversion-ledger v1 manifest")
    records = [
        record
        for record in manifest.get("ledgers", [])
        if record.get("source_family")
        == "canonical_residual_exact1m_current_spread_cf"
    ]
    if len(records) != 1:
        raise ValueError("canonical residual ledger manifest record is missing/ambiguous")
    record = records[0]
    if Path(str(record.get("path"))).resolve() != ledger_path.resolve():
        raise ValueError("canonical residual ledger path differs from manifest")
    if str(record.get("sha256")) != sha256(ledger_path):
        raise ValueError("canonical residual ledger hash mismatch")
    return manifest


def _require_direct_manifest(path: Path, direct_path: Path) -> dict[str, Any]:
    manifest = _read_json(path)
    if manifest.get("schema") != "cross_era_direct_net_quantile_challenger_v1":
        raise ValueError("requires cross-era direct q25 challenger v1 manifest")
    record = manifest.get("outputs", {}).get("historical_oof_winner", {})
    if not record:
        raise ValueError("direct historical OOF output is absent from manifest")
    declared = Path(str(record.get("path")))
    # The authoritative manifest stores this path relative to the Ares root.
    if not declared.is_absolute():
        declared = ROOT / declared
    if declared.resolve() != direct_path.resolve():
        raise ValueError("direct historical OOF path differs from manifest")
    if str(record.get("sha256")) != sha256(direct_path):
        raise ValueError("direct historical OOF hash mismatch")
    return manifest


def _marapr_direct(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        *IDENTITY_COLUMNS,
        "q25_net_bps",
        "execution_net_ev_12h",
        "label_resolution_utc",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"direct historical OOF is missing: {missing}")
    direct = _identity(frame, "direct historical OOF")
    direct = direct.loc[
        direct["__ts__"].ge(MARCH_START) & direct["__ts__"].lt(MAY_START)
    ].copy()
    if direct.empty:
        raise ValueError("direct historical OOF has no March-April 2025 rows")
    direct[DIRECT_BPS_COLUMN] = pd.to_numeric(
        direct["q25_net_bps"], errors="raise"
    )
    if not np.isfinite(direct[DIRECT_BPS_COLUMN].to_numpy(dtype=float)).all():
        raise ValueError("direct q25 bps score is non-finite")
    direct[DIRECT_NET_COLUMN] = pd.to_numeric(
        direct["execution_net_ev_12h"], errors="raise"
    )
    direct[DIRECT_LABEL_END_COLUMN] = pd.to_datetime(
        direct["label_resolution_utc"], utc=True, errors="raise"
    )
    return direct.loc[
        :,
        [
            *IDENTITY_COLUMNS,
            DIRECT_BPS_COLUMN,
            DIRECT_NET_COLUMN,
            DIRECT_LABEL_END_COLUMN,
        ],
    ]


def build_all_score_frame(
    ledger: pd.DataFrame,
    direct_historical_oof: pd.DataFrame,
    *,
    expected_rows: int = EXPECTED_ROWS,
) -> pd.DataFrame:
    """Return the exact Mar--Apr identity intersection, failing closed."""

    base = _identity(ledger, "canonical residual exact-policy ledger")
    direct = _marapr_direct(direct_historical_oof)
    base_month = pd.to_datetime(base["__ts__"], utc=True)
    if not (base_month.ge(MARCH_START) & base_month.lt(MAY_START)).all():
        raise ValueError("canonical residual ledger must contain March-April 2025 only")
    if int(len(base)) != int(expected_rows):
        raise ValueError(
            f"canonical residual ledger rows {len(base)} != expected {expected_rows}"
        )
    if int(len(direct)) != int(expected_rows):
        raise ValueError(
            f"direct March-April rows {len(direct)} != expected {expected_rows}"
        )
    joined = base.merge(
        direct,
        on=list(IDENTITY_COLUMNS),
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if not joined["_merge"].eq("both").all():
        counts = joined["_merge"].value_counts().to_dict()
        raise ValueError(f"exact four-field coverage failed: {counts}")
    joined = joined.drop(columns="_merge")
    if len(joined) != int(expected_rows):
        raise ValueError("joined all-score rows differ from expected coverage")
    if not np.array_equal(
        joined["execution_net_ev_12h"].to_numpy(dtype=float),
        joined[DIRECT_NET_COLUMN].to_numpy(dtype=float),
    ):
        raise ValueError("direct and canonical ledgers have different realized net outcomes")
    canonical_label_end = pd.to_datetime(
        joined["execution_label_end_utc"], utc=True, errors="raise"
    )
    if not canonical_label_end.equals(joined[DIRECT_LABEL_END_COLUMN]):
        raise ValueError("direct and canonical ledgers have different label horizons")
    joined = joined.drop(columns=[DIRECT_NET_COLUMN, DIRECT_LABEL_END_COLUMN])
    if any("mapped" in column.lower() for column in joined.columns):
        # The canonical ledger contains no mapped score.  Fail closed rather
        # than accidentally carrying a future mapped field into the bridge.
        raise ValueError("mapped fields are forbidden in all-score waterfall")
    joined[DIRECT_RETURN_COLUMN] = joined[DIRECT_BPS_COLUMN] / 10_000.0
    joined["source_family"] = SOURCE_FAMILY
    # Existing ledger flags are provenance, but the materializer adds the
    # exact evidence roles in its manifest rather than trying to infer OOF
    # status from score values.
    validate_source(joined, {"source_family": SOURCE_FAMILY})
    return joined.sort_values(list(IDENTITY_COLUMNS), kind="stable").reset_index(
        drop=True
    )


def _emit_diagnostics(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    parts: dict[str, list[pd.DataFrame]] = {
        key: []
        for key in (
            "full_ic",
            "tails",
            "compression",
            "response_cells",
            "response_summary",
            "cutoff_ties",
            "fixed_composition",
        )
    }
    for score in score_columns(frame):
        parts["full_ic"].append(full_ic(frame, source_family=SOURCE_FAMILY, score=score))
        parts["tails"].append(tail_metrics(frame, source_family=SOURCE_FAMILY, score=score))
        parts["compression"].append(
            score_compression(frame, source_family=SOURCE_FAMILY, score=score)
        )
        cells, summary = response_20bin(frame, source_family=SOURCE_FAMILY, score=score)
        parts["response_cells"].append(cells)
        parts["response_summary"].append(summary)
        parts["cutoff_ties"].append(
            cutoff_ties(frame, source_family=SOURCE_FAMILY, score=score)
        )
        parts["fixed_composition"].append(
            fixed_composition(frame, source_family=SOURCE_FAMILY, score=score)
        )
    return {
        name: pd.concat(values, ignore_index=True) if values else pd.DataFrame()
        for name, values in parts.items()
    }


def run(
    ledger_path: Path,
    ledger_manifest_path: Path,
    direct_path: Path,
    direct_manifest_path: Path,
    output_dir: Path,
    *,
    expected_rows: int = EXPECTED_ROWS,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    _require_ledger_manifest(ledger_manifest_path, ledger_path)
    _require_direct_manifest(direct_manifest_path, direct_path)
    frame = build_all_score_frame(
        pd.read_parquet(ledger_path),
        pd.read_parquet(
            direct_path,
            columns=[
                *IDENTITY_COLUMNS,
                "q25_net_bps",
                "execution_net_ev_12h",
                "label_resolution_utc",
            ],
        ),
        expected_rows=expected_rows,
    )
    diagnostics = _emit_diagnostics(frame)
    output_dir.mkdir(parents=True, exist_ok=False)
    outputs: dict[str, dict[str, Any]] = {}
    all_outputs = {"all_score_waterfall": frame, **diagnostics}
    for name, result in all_outputs.items():
        path = output_dir / f"{name}.parquet"
        result.to_parquet(path, index=False, compression="zstd")
        outputs[name] = {"path": str(path), "sha256": sha256(path), "rows": int(len(result))}
    report = {
        "schema": "marapr2025_all_score_ic_ev_waterfall_v1",
        "status": "DIAGNOSTIC_ONLY_NO_MAPPING_NO_PROMOTION",
        "rows": int(len(frame)),
        "identity": list(IDENTITY_COLUMNS),
        "period": {"start": MARCH_START.isoformat(), "end_exclusive": MAY_START.isoformat()},
        "inputs": {
            "canonical_residual_exact_policy": {
                "path": str(ledger_path),
                "sha256": sha256(ledger_path),
                "manifest_path": str(ledger_manifest_path),
                "manifest_sha256": sha256(ledger_manifest_path),
                "rows": int(len(frame)),
                "evidence_role": "strict_residual_OOF_plus_current_spread_exact_policy_labels",
            },
            "direct_q25": {
                "path": str(direct_path),
                "sha256": sha256(direct_path),
                "manifest_path": str(direct_manifest_path),
                "manifest_sha256": sha256(direct_manifest_path),
                "evidence_role": "historical_OOF_research_only",
                "raw_score_only": True,
            },
        },
        "score_contract": {
            "declared_diagnostic_scores": score_columns(frame),
            "units": {
                "score_base_alpha": "unitless legacy native-24h alpha score",
                "score_base_expected_ev": "return units; upstream base expected-EV stream",
                "score_residual_delta_ev": "return units; upstream residual component",
                "score_residual_expected_ev": "return units; upstream base-plus-residual expected-EV stream",
                DIRECT_BPS_COLUMN: "basis points; raw q25 direct exact-net score",
                DIRECT_RETURN_COLUMN: "return-unit conversion of raw q25; retained for comparison only, not a separately evaluated score",
            },
            "mapped_score_forbidden": True,
            "mapped_columns_emitted": [],
        },
        "contracts": {
            "join": "exact full four-field identity; 1:1 and complete after March-April direct filtering; realized net outcome and label-resolution timestamp must also be exactly equal",
            "economics": "canonical current-spread exact 1m frozen-policy 12h gross/cost/net; gross - explicit fee cost = net; spread is embedded in gross",
            "selection": "all reusable diagnostics rank each declared score descending with candidate-ID ascending ties and no additional mapping; global books remain pooled across timestamps and sides; upstream expected-EV streams retain their declared upstream semantics",
            "no_mapping": True,
            "no_promotion": True,
        },
        "outputs": outputs,
        "runner": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
        "promotion_eligible": False,
    }
    manifest_path = output_dir / "manifest.json"
    write_json(manifest_path, report)
    (output_dir / "manifest.sha256").write_text(sha256(manifest_path) + "\n", encoding="utf-8")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--ledger-manifest", type=Path, default=DEFAULT_LEDGER_MANIFEST)
    parser.add_argument("--direct", type=Path, default=DEFAULT_DIRECT)
    parser.add_argument("--direct-manifest", type=Path, default=DEFAULT_DIRECT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    print(
        json.dumps(
            safe(
                run(
                    args.ledger,
                    args.ledger_manifest,
                    args.direct,
                    args.direct_manifest,
                    args.output_dir,
                )
            ),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
