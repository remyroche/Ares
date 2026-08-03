#!/usr/bin/env python3
"""Compare raw, causal-global and causal-side recent-EV score mappings.

This consumes the corrected exact-policy v3 mapping source, retaining its
114,096-row strict outer-OOF mapped slice.  It performs no model fit or
remapping.  All arms rank the same pooled global population; side-local
outputs are attribution only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

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
DEFAULT_INPUT = ROOT / (
    "data_perp/artifacts/current_exact_policy_global_book_mapping_source_"
    "20260730_v3/causal_mapped_candidates.parquet"
)
DEFAULT_INPUT_MANIFEST = DEFAULT_INPUT.with_name("manifest.json")
DEFAULT_OUTPUT = ROOT / (
    "data_perp/artifacts/causal_mapping_ic_ev_waterfall_20260730_v1"
)
EXPECTED_ROWS = 114_096
SOURCE_FAMILY = "mayjul2026_strict_oof_causal_mapping_comparison"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_frame(
    source: pd.DataFrame, *, expected_rows: int = EXPECTED_ROWS
) -> pd.DataFrame:
    required = {
        *IDENTITY_COLUMNS,
        "execution_decision_utc",
        "execution_label_end_utc",
        "execution_gross_ev_12h",
        "execution_cost_return",
        "execution_net_ev_12h",
        "execution_mfe_return_12h",
        "execution_mae_return_12h",
        "execution_exit_class",
        "catboost__residual__without_hpo__all_features",
        "causal_recent_isotonic_ev",
        "causal_recent_side_isotonic_ev",
        "causal_recent_side_isotonic_ev__is_oof",
        "mapped_eligible",
        "evaluation_origin",
    }
    missing = sorted(required.difference(source.columns))
    if missing:
        raise ValueError(f"mapped comparison source missing: {missing}")
    frame = source.copy()
    strict = (
        frame["mapped_eligible"].fillna(False).astype(bool)
        & frame["causal_recent_side_isotonic_ev__is_oof"]
        .fillna(False)
        .astype(bool)
    )
    frame = frame.loc[strict].copy()
    if len(frame) != expected_rows:
        raise ValueError(f"strict mapped rows {len(frame)} != expected {expected_rows}")
    frame["candidate_id"] = frame["candidate_id"].astype(str)
    frame["side_name"] = frame["side_name"].astype(str)
    frame["__symbol__"] = frame["__symbol__"].astype(str)
    frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True, errors="raise")
    frame["execution_decision_utc"] = pd.to_datetime(
        frame["execution_decision_utc"], utc=True, errors="raise"
    )
    frame["execution_label_end_utc"] = pd.to_datetime(
        frame["execution_label_end_utc"], utc=True, errors="raise"
    )
    if frame.duplicated(list(IDENTITY_COLUMNS)).any():
        raise ValueError("mapped comparison has duplicate identities")
    if not frame["evaluation_origin"].eq("historical_outer_oof").all():
        raise ValueError("mapped comparison is not uniformly strict outer OOF")
    if not frame["execution_label_end_utc"].eq(
        frame["execution_decision_utc"] + pd.Timedelta(hours=12)
    ).all():
        raise ValueError("mapped comparison is not on the exact 12h horizon")

    frame["score_raw_execution_ev"] = pd.to_numeric(
        frame["catboost__residual__without_hpo__all_features"], errors="raise"
    )
    frame["score_causal_global_21d_ev"] = pd.to_numeric(
        frame["causal_recent_isotonic_ev"], errors="raise"
    )
    frame["score_causal_side_21d_ev"] = pd.to_numeric(
        frame["causal_recent_side_isotonic_ev"], errors="raise"
    )
    frame["candidate_month"] = frame["__ts__"].dt.strftime("%Y-%m")
    frame["opportunity_gross_above_cost_0bps"] = frame[
        "execution_net_ev_12h"
    ].gt(0.0)
    frame["source_family"] = SOURCE_FAMILY
    keep = [
        *IDENTITY_COLUMNS,
        "execution_decision_utc",
        "execution_label_end_utc",
        "execution_gross_ev_12h",
        "execution_cost_return",
        "execution_net_ev_12h",
        "execution_mfe_return_12h",
        "execution_mae_return_12h",
        "execution_exit_class",
        "candidate_month",
        "opportunity_gross_above_cost_0bps",
        "source_family",
        "evaluation_origin",
        "score_raw_execution_ev",
        "score_causal_global_21d_ev",
        "score_causal_side_21d_ev",
    ]
    frame = frame.loc[:, keep]
    for score in score_columns(frame):
        if not np.isfinite(frame[score].to_numpy(float)).all():
            raise ValueError(f"non-finite score: {score}")
    validate_source(frame, {"source_family": SOURCE_FAMILY})
    return frame.sort_values(list(IDENTITY_COLUMNS), kind="stable").reset_index(
        drop=True
    )


def _diagnostics(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
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
        name: pd.concat(values, ignore_index=True) for name, values in parts.items()
    }


def run(
    input_path: Path,
    input_manifest_path: Path,
    output_dir: Path,
    *,
    expected_rows: int = EXPECTED_ROWS,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    manifest = _read_json(input_manifest_path)
    if manifest.get("schema") != "causal_score_economics_conversion_mapping_v1":
        raise ValueError("requires corrected exact-policy mapping source")
    record = manifest.get("outputs", {}).get("mapped", {})
    declared = Path(str(record.get("path")))
    if not declared.is_absolute():
        declared = input_manifest_path.parent / declared
    if declared.resolve() != input_path.resolve() or str(
        record.get("sha256")
    ) != sha256(input_path):
        raise ValueError("diagnostic input hash mismatch")
    mapping = manifest.get("causal_contract", {}).get("mapping", {})
    if mapping.get("global") != "causal_recent_isotonic_ev" or mapping.get(
        "side_shrunk"
    ) != "causal_recent_side_isotonic_ev":
        raise ValueError("corrected source does not retain both mapping arms")
    selection = manifest.get("selection_contract", {})
    if selection.get("primary") != "one pooled global top-k":
        raise ValueError("source does not use the canonical pooled global book")

    frame = build_frame(pd.read_parquet(input_path), expected_rows=expected_rows)
    diagnostics = _diagnostics(frame)
    output_dir.mkdir(parents=True, exist_ok=False)
    frames = {"mapped_score_waterfall": frame, **diagnostics}
    outputs: dict[str, dict[str, Any]] = {}
    for name, result in frames.items():
        path = output_dir / f"{name}.parquet"
        result.to_parquet(path, index=False, compression="zstd")
        outputs[name] = {
            "path": str(path),
            "sha256": sha256(path),
            "rows": int(len(result)),
        }
    report = {
        "schema": "causal_mapping_ic_ev_waterfall_v1",
        "status": "DIAGNOSTIC_ONLY_EXISTING_STRICT_OOF_MAPS_NO_REFIT",
        "rows": int(len(frame)),
        "period": {
            "start": frame["__ts__"].min().isoformat(),
            "end": frame["__ts__"].max().isoformat(),
        },
        "contracts": {
            "identity": list(IDENTITY_COLUMNS),
            "population": "same 114096 jointly finite strict outer-OOF rows for every arm",
            "mapping": (
                "existing causal 21-day global and side isotonic maps; each "
                "snapshot uses only earlier label-resolved outcomes; no fit here"
            ),
            "selection": (
                "one month-level pooled global top 1/5/10/20 across timestamps "
                "and sides; side-local metrics are attribution only"
            ),
            "economics": "exact 1m decision+12h gross/cost/net; no second cost subtraction",
        },
        "input": {
            "path": str(input_path),
            "sha256": sha256(input_path),
            "manifest_path": str(input_manifest_path),
            "manifest_sha256": sha256(input_manifest_path),
            "upstream_exact_policy": manifest.get("source_sha256", {}).get(
                "exact_policy"
            ),
        },
        "score_contract": {
            "scores": score_columns(frame),
            "existing_maps_only": True,
            "refit": False,
        },
        "outputs": outputs,
        "runner": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256(Path(__file__).resolve()),
        },
        "promotion_eligible": False,
    }
    manifest_path = output_dir / "manifest.json"
    write_json(manifest_path, report)
    (output_dir / "manifest.sha256").write_text(
        sha256(manifest_path) + "\n", encoding="utf-8"
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--input-manifest", type=Path, default=DEFAULT_INPUT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    print(
        json.dumps(
            safe(run(args.input, args.input_manifest, args.output_dir)),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
