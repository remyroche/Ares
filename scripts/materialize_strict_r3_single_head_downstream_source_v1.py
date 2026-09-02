#!/usr/bin/env python3
"""Materialise a target-free one-head source for the strict-R3 downstream stack.

The one-head research challenger is deliberately *not* a B/E/T blend.  The
downstream stack has a fixed historical schema containing B/E/T coordinates;
this adapter supplies that schema without giving any second head authority:
all score coordinates are the one LambdaRank score and all disagreement fields
are exactly zero.  The 120 causal market fields are copied unchanged from the
immutable source panel.  Labels are never read by this program.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


UPSTREAM_COORDINATE = "single declared Base head; no E/T/R3 model input"
PROHIBITED_COLUMNS = frozenset({
    "policy_path_valid", "policy_net_bps", "policy_gross_bps",
    "policy_label_available_ts", "policy_exit_bar_15m", "policy_entry_price",
    "policy_exit_price", "policy_exit_reason", "policy_cost_bps",
})
IDENTITY_COLUMNS = ("candidate_id", "__decision_ts__", "side_name")
SCORE_COLUMNS = (*IDENTITY_COLUMNS, "head_score")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _month_range(start: pd.Timestamp, end: pd.Timestamp) -> Iterable[pd.Timestamp]:
    yield from pd.date_range(start, end, freq="MS", tz="UTC")


def _rank_and_adapt(
    scores: pd.DataFrame,
    features: pd.DataFrame,
    fields: tuple[str, ...],
    *,
    all_routed: bool = False,
) -> pd.DataFrame:
    """Join one model score to causal fields and build a zero-authority schema.

    This pure function intentionally makes the no-E/T property testable.  The
    rank is timestamp-local over the pre-routed population scored by the one
    model; a higher score therefore has a larger rank in ``[0, 1]``.
    """
    if set(SCORE_COLUMNS) - set(scores.columns):
        raise AssertionError("single-head score panel lacks required columns")
    required_features = set((*IDENTITY_COLUMNS, *fields))
    if required_features - set(features.columns):
        raise AssertionError("causal feature panel lacks frozen 120-field contract")
    if PROHIBITED_COLUMNS & set(features.columns):
        raise AssertionError("causal feature panel contains outcome columns")
    if scores["candidate_id"].duplicated().any() or features["candidate_id"].duplicated().any():
        raise AssertionError("candidate IDs must be unique before target-free join")
    work = scores.loc[:, list(SCORE_COLUMNS)].copy()
    work["__decision_ts__"] = pd.to_datetime(work["__decision_ts__"], utc=True, errors="raise")
    source = features.loc[:, [*IDENTITY_COLUMNS, *fields]].copy()
    source["__decision_ts__"] = pd.to_datetime(source["__decision_ts__"], utc=True, errors="raise")
    joined = work.merge(
        source,
        on=list(IDENTITY_COLUMNS),
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if not joined["_merge"].eq("both").all():
        raise AssertionError("one-head score identity is absent from causal feature panel")
    joined = joined.drop(columns="_merge")
    if joined.loc[:, list(fields)].isna().any().any():
        raise AssertionError("one-head score identity has incomplete causal feature values")
    score = pd.to_numeric(joined["head_score"], errors="coerce")
    if not np.isfinite(score).all():
        raise AssertionError("one-head scores must be finite")
    joined["base_rank_ts"] = joined.groupby("__decision_ts__", sort=False)["head_score"].rank(
        method="first", pct=True, ascending=True,
    ).astype(np.float32)
    # The historical downstream consumer expects this wire column even when
    # routing has already occurred upstream.  A Router50 -> one-Base
    # experiment must not silently reintroduce an additional Base top-30%
    # gate.  Keep the historical behaviour as the default for old adapters,
    # while allowing a declared all-routed contract for the routed Base path.
    joined["enhanced_base_routed"] = (
        True if all_routed else joined["base_rank_ts"].ge(0.70)
    )
    # These columns exist only because the frozen downstream consumer has a
    # historical fixed schema.  No E/T/B0 score is loaded or derived here.
    for column in ("enhanced_base_bps", "base_bps", "efficiency_bps", "timing_bps"):
        joined[column] = score.astype(np.float32)
    for column in ("e_minus_t", "e_minus_b0", "t_minus_b0", "base_component_std"):
        joined[column] = np.float32(0.0)
    output_columns = [
        *IDENTITY_COLUMNS,
        "enhanced_base_bps", "base_rank_ts", "enhanced_base_routed",
        "base_bps", "efficiency_bps", "timing_bps",
        "e_minus_t", "e_minus_b0", "t_minus_b0", "base_component_std",
        *fields,
    ]
    output = joined.loc[:, output_columns].copy()
    if output["candidate_id"].duplicated().any():
        raise AssertionError("target-free output has duplicate candidates")
    if output.loc[:, list(fields)].isna().any().any():
        raise AssertionError("target-free output has incomplete causal contract")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--single-head-root", type=Path, required=True)
    parser.add_argument("--causal-feature-root", type=Path, required=True)
    parser.add_argument("--feature-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--score-target",
        default=None,
        help=(
            "optional target directory for modern single-Base receipts; when "
            "set, reads target_free_scores/TARGET/month=YYYY-MM.parquet with "
            "a base_score column instead of the legacy month directory"
        ),
    )
    parser.add_argument(
        "--score-subroot",
        default=None,
        help=(
            "optional relative target-free score directory below --single-head-root, "
            "for example scheme=tail_linear_125/target_free_scores.  This reads "
            "month=YYYY-MM.parquet with base_score and preserves the root manifest."
        ),
    )
    parser.add_argument(
        "--upstream-coordinate",
        default=UPSTREAM_COORDINATE,
        help=(
            "human-readable immutable provenance for the one Base head; it "
            "does not affect score construction or downstream inputs"
        ),
    )
    parser.add_argument(
        "--all-routed",
        action="store_true",
        help="preserve every already Router-selected score row for downstream; never apply a Base top-30%% gate",
    )
    parser.add_argument("--start", default="2025-11-01")
    parser.add_argument("--end", default="2026-07-01")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    start = pd.Timestamp(args.start, tz="UTC")
    end = pd.Timestamp(args.end, tz="UTC")
    if start > end:
        raise ValueError("start must be no later than end")
    hpo_manifest = json.loads(args.feature_manifest.read_text())
    fields = tuple(hpo_manifest.get("features", ()))
    if len(fields) != 120 or len(set(fields)) != 120:
        raise AssertionError("feature manifest must contain exactly 120 unique causal fields")
    args.out.mkdir(parents=True)
    coverage: list[dict[str, object]] = []
    for month in _month_range(start, end):
        month_key = f"{month:%Y-%m}"
        if args.score_subroot:
            score_path = args.single_head_root / str(args.score_subroot) / f"month={month_key}.parquet"
            score_column = "base_score"
        elif args.score_target:
            score_path = (
                args.single_head_root / "target_free_scores" / str(args.score_target)
                / f"month={month_key}.parquet"
            )
            score_column = "base_score"
        else:
            score_path = args.single_head_root / "target_free_scores" / f"month={month_key}" / "target_free_scores.parquet"
            score_column = "head_score"
        source_path = args.causal_feature_root / "target_free_monthly" / f"month={month_key}" / "scores_features.parquet"
        if not score_path.exists() or not source_path.exists():
            raise FileNotFoundError(f"missing immutable source for {month_key}")
        source_columns = set(pq.ParquetFile(source_path).schema.names)
        leaked = sorted(PROHIBITED_COLUMNS & source_columns)
        if leaked:
            raise AssertionError(f"causal source is not target-free for {month_key}: {leaked}")
        scores = pd.read_parquet(score_path, columns=[*IDENTITY_COLUMNS, score_column]).rename(
            columns={score_column: "head_score"},
        )
        features = pd.read_parquet(source_path, columns=[*IDENTITY_COLUMNS, *fields])
        output = _rank_and_adapt(scores, features, fields, all_routed=bool(args.all_routed))
        target = args.out / f"month={month_key}"
        target.mkdir()
        output.to_parquet(target / "scores_features.parquet", index=False, compression="zstd")
        coverage.append({
            "month": month_key,
            "rows": int(len(output)),
            "timestamps": int(output["__decision_ts__"].nunique()),
            "feature_complete_fraction": float(output.loc[:, list(fields)].notna().all(axis=1).mean()),
            "routed_rows": int(output["enhanced_base_routed"].sum()),
            "score_sha256": _sha256(score_path),
            "feature_source_sha256": _sha256(source_path),
        })
    coverage_frame = pd.DataFrame(coverage)
    if coverage_frame["feature_complete_fraction"].lt(0.90).any():
        raise AssertionError("a monthly causal feature contract is below the 90% gate")
    coverage_frame.to_parquet(args.out / "coverage_audit.parquet", index=False, compression="zstd")
    manifest = {
        "schema": "strict_r3_single_head_downstream_source_v1",
        "upstream": {
            "coordinate": str(args.upstream_coordinate),
            "model_count": 1,
            "model_score": "head_score",
            "downstream_schema_adapter": {
                "score_coordinates": "all four historical score slots equal the one head score",
                "disagreement_coordinates": "all exactly zero; placeholders have no model authority",
            },
        },
        "feature_contract": {"count": len(fields), "ordered_fields": list(fields)},
        "source": {
            "single_head_root": str(args.single_head_root.resolve()),
            "causal_feature_root": str(args.causal_feature_root.resolve()),
            "feature_manifest": str(args.feature_manifest.resolve()),
            "single_head_manifest_sha256": _sha256(args.single_head_root / "run_manifest.json"),
            "feature_manifest_sha256": _sha256(args.feature_manifest),
        },
        "routing": {
            "all_router_selected_rows_retained": bool(args.all_routed),
            "score_target": args.score_target,
            "score_subroot": args.score_subroot,
        },
        "score_months": coverage_frame["month"].tolist(),
        "target_free": True,
        "prohibited_outcome_columns": sorted(PROHIBITED_COLUMNS),
        "coverage_min": float(coverage_frame["feature_complete_fraction"].min()),
    }
    (args.out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    correctness = {
        "single_head_only": True,
        "no_et_or_r3_score_loaded": True,
        "all_historical_disagreement_fields_zero": True,
        "no_post_router_base_cutoff": bool(args.all_routed),
        "target_free_output": True,
        "all_months_feature_complete": bool(coverage_frame["feature_complete_fraction"].ge(0.90).all()),
        "identity_preserving_left_join": True,
    }
    (args.out / "correctness_report.json").write_text(json.dumps(correctness, indent=2) + "\n")
    print(json.dumps({"event": "complete", "out": str(args.out), "months": len(coverage_frame), "rows": int(coverage_frame["rows"].sum())}), flush=True)


if __name__ == "__main__":
    main()
