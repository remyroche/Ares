#!/usr/bin/env python3
"""Materialize immutable, anchor-time context for economic-transition cohorts.

This deliberately separates the feature surface from
``canonical_economic_conversion_transition_labels_20260729_v1``.  Cohorts are
the same deterministic timestamp/side/base-score-decile groups as the label
artifact, but every value below is aggregated from rows available at the
anchor.  No execution outcome, causal mapping, or transition label is copied
into this artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.materialize_canonical_economic_conversion_transition_labels import (
    add_frozen_causal_score_deciles,
)


PANEL_SOURCE = ROOT / "data_perp/artifacts/canonical_opportunity_payoff_trust_panel_20260729_v2"
LABEL_SOURCE = ROOT / "data_perp/artifacts/canonical_economic_conversion_transition_labels_20260729_v1"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/canonical_economic_conversion_transition_context_20260729_v1"

SCHEMA = "canonical_economic_conversion_transition_context_v1"
IDENTITY = ("candidate_id", "side_name", "__symbol__", "__ts__")
COHORT_KEY = ("cohort_anchor_utc", "side_name", "frozen_base_score_decile")

# The exact compact, decision-observable surface.  The five core levels and
# eighteen deltas are fixed by the canonical v2 manifest; selected composites
# are intentionally small rather than the full historical regime catalogue.
BASE_CONTEXT_COLUMNS = (
    "base_oof_score",
    "base_rank_timestamp_side",
    "base_group_rows_timestamp_side",
    "base_rank_pct_timestamp_side",
    "base_score_z_timestamp_side",
    "base_rank_decile_timestamp_side",
    "selected_top40_timestamp_side",
    "base_top40_cutoff_timestamp_side",
    "base_margin_to_top40_cutoff",
    "base_margin_to_top40_cutoff_z",
    "base_rank_timestamp_global",
    "base_group_rows_timestamp_global",
    "base_rank_pct_timestamp_global",
    "base_score_z_timestamp_global",
)
CORE_MARKET_COLUMNS = (
    "range_24h_pct",
    "__meta_raw__volatility_zscore",
    "trend_r2_24",
    "jump_intensity",
    "__meta_raw__chop_score",
)
TRANSITION_COLUMNS = (
    "preentry_transition__range_24h_pct__delta_3h",
    "preentry_transition__range_24h_pct__delta_12h",
    "preentry_transition__meta_raw__volatility_zscore__delta_3h",
    "preentry_transition__meta_raw__volatility_zscore__delta_12h",
    "preentry_transition__trend_r2_24__delta_3h",
    "preentry_transition__trend_r2_24__delta_12h",
    "preentry_transition__jump_intensity__delta_3h",
    "preentry_transition__jump_intensity__delta_12h",
    "preentry_transition__meta_raw__chop_score__delta_3h",
    "preentry_transition__meta_raw__chop_score__delta_12h",
    "preentry_transition__regime_source_shock_impulse_score__delta_3h",
    "preentry_transition__regime_source_shock_impulse_score__delta_12h",
    "preentry_transition__regime_source_compression_score__delta_3h",
    "preentry_transition__regime_source_compression_score__delta_12h",
    "preentry_transition__regime_source_dirty_shock_avoid_score__delta_3h",
    "preentry_transition__regime_source_dirty_shock_avoid_score__delta_12h",
    "preentry_transition__regime_source_loud_breakout_impulse_score__delta_3h",
    "preentry_transition__regime_source_loud_breakout_impulse_score__delta_12h",
)
COMPACT_REGIME_COLUMNS = (
    "__regime_source_shock_impulse_score__",
    "__regime_source_execution_quality_score__",
    "__regime_source_execution_risk_score__",
    "__regime_source_oi_agreement_score__",
    "__regime_source_compression_score__",
    "__regime_source_loud_breakout_impulse_score__",
    "__regime_source_dirty_shock_avoid_score__",
    "__regime_source_clean_execution_context_score__",
)
DECISION_OBSERVABLE_COLUMNS = (
    *BASE_CONTEXT_COLUMNS,
    *CORE_MARKET_COLUMNS,
    *TRANSITION_COLUMNS,
    *COMPACT_REGIME_COLUMNS,
)
# Side and score-decile are score/identity context fixed at the anchor, never
# label-derived routing.  They are passed as compact numeric coordinates so
# the head can retain the canonical long/short and cohort distinction.
DERIVED_COHORT_CONTEXT_COLUMNS = (
    "context__side_sign",
    "context__frozen_base_score_decile",
)
PROHIBITED_PREFIXES = (
    "execution_",
    "opportunity_",
    "mapped_",
)
PROHIBITED_TOKENS = (
    "target",
    "label",
    "outcome",
    "exit",
    "mfe",
    "mae",
    "first_touch",
    "realized",
    "wait_action",
    "target_price",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _artifact_manifest(root: Path, required_schema: str) -> tuple[dict[str, Any], dict[str, str]]:
    paths = (root / "manifest.json", root / "manifest.sha256")
    if not all(path.is_file() for path in paths):
        raise FileNotFoundError(f"immutable artifact manifest is incomplete: {root}")
    actual = sha256(paths[0])
    expected = paths[1].read_text(encoding="utf-8").strip().split(maxsplit=1)
    if not expected or expected[0] != actual:
        raise ValueError(f"manifest checksum mismatch: {root}")
    manifest = json.loads(paths[0].read_text(encoding="utf-8"))
    if manifest.get("schema") != required_schema:
        raise ValueError(f"unexpected artifact schema at {root}: {manifest.get('schema')!r}")
    return manifest, {str(path): sha256(path) for path in paths}


def _panel_identity_sha256(frame: pd.DataFrame) -> str:
    ordered = frame.loc[:, list(IDENTITY)].copy()
    ordered["__ts__"] = pd.to_datetime(ordered["__ts__"], utc=True, errors="raise").astype(str)
    ordered = ordered.astype(str).sort_values(list(IDENTITY), kind="stable")
    return hashlib.sha256(ordered.to_csv(index=False, lineterminator="\n").encode()).hexdigest()


def _context_name(column: str) -> str:
    return f"context__{column.strip('_')}__mean"


def context_feature_columns() -> tuple[str, ...]:
    return (
        *(_context_name(column) for column in DECISION_OBSERVABLE_COLUMNS),
        *DERIVED_COHORT_CONTEXT_COLUMNS,
    )


def _validate_feature_surface(columns: Iterable[str]) -> None:
    columns = tuple(columns)
    bad = [
        column
        for column in columns
        if (
            any(column.lower().startswith(prefix) for prefix in PROHIBITED_PREFIXES)
            or any(token in column.lower() for token in PROHIBITED_TOKENS)
        )
    ]
    if bad:
        raise ValueError(f"non-observable or outcome-derived field entered context surface: {bad}")
    if len(set(columns)) != len(columns):
        raise ValueError("decision-observable context surface has duplicate columns")


def _normalise_panel(panel: pd.DataFrame) -> pd.DataFrame:
    required = set(IDENTITY) | set(DECISION_OBSERVABLE_COLUMNS)
    missing = sorted(required.difference(panel.columns))
    if missing:
        raise ValueError(f"canonical panel lacks context columns: {missing}")
    result = panel.loc[:, [*IDENTITY, *DECISION_OBSERVABLE_COLUMNS]].copy()
    result["candidate_id"] = result["candidate_id"].astype(str)
    result["side_name"] = result["side_name"].astype(str).str.lower()
    result["__symbol__"] = result["__symbol__"].astype(str)
    result["__ts__"] = pd.to_datetime(result["__ts__"], utc=True, errors="raise")
    if not result["side_name"].isin(("long", "short")).all():
        raise ValueError("canonical panel contains non-canonical sides")
    if not result["__ts__"].dt.floor("h").eq(result["__ts__"]).all():
        raise ValueError("canonical panel timestamps must be UTC-aligned to the hour")
    if result.duplicated(list(IDENTITY)).any() or result["candidate_id"].duplicated().any():
        raise ValueError("canonical panel identity is not one-to-one")
    for column in DECISION_OBSERVABLE_COLUMNS:
        result[column] = pd.to_numeric(result[column], errors="coerce")
        if np.isinf(result[column].to_numpy(dtype=float, na_value=np.nan)).any():
            raise ValueError(f"context column contains an infinity: {column}")
    return result


def _normalise_label_cohorts(labels: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(COHORT_KEY).difference(labels.columns))
    if missing:
        raise ValueError(f"transition labels lack cohort key columns: {missing}")
    result = labels.loc[:, list(COHORT_KEY)].copy()
    result["cohort_anchor_utc"] = pd.to_datetime(
        result["cohort_anchor_utc"], utc=True, errors="raise"
    )
    result["side_name"] = result["side_name"].astype(str).str.lower()
    result["frozen_base_score_decile"] = pd.to_numeric(
        result["frozen_base_score_decile"], errors="raise"
    ).astype(np.int8)
    if not result["side_name"].isin(("long", "short")).all():
        raise ValueError("transition labels contain non-canonical sides")
    if not result["frozen_base_score_decile"].between(0, 9).all():
        raise ValueError("transition labels contain invalid score deciles")
    return result.drop_duplicates().sort_values(list(COHORT_KEY), kind="stable").reset_index(drop=True)


def materialize_context(panel: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Aggregate exact anchor-time fields onto the immutable label cohorts."""

    _validate_feature_surface(DECISION_OBSERVABLE_COLUMNS)
    rows = add_frozen_causal_score_deciles(_normalise_panel(panel))
    rows = rows.rename(columns={"__ts__": "cohort_anchor_utc"})
    group_columns = ["cohort_anchor_utc", "side_name", "frozen_base_score_decile"]
    context = (
        rows.groupby(group_columns, observed=True, sort=True)
        .agg(
            anchor_candidate_support=("candidate_id", "size"),
            **{
                _context_name(column): (column, "mean")
                for column in DECISION_OBSERVABLE_COLUMNS
            },
        )
        .reset_index()
    )
    context["context__side_sign"] = np.where(
        context["side_name"].eq("long"), 1.0, -1.0
    ).astype(np.int8)
    context["context__frozen_base_score_decile"] = context[
        "frozen_base_score_decile"
    ].astype(np.int8)
    context["anchor_candidate_support"] = context["anchor_candidate_support"].astype(np.int32)
    cohorts = _normalise_label_cohorts(labels)
    joined = cohorts.merge(context, on=group_columns, how="left", validate="one_to_one")
    if joined["anchor_candidate_support"].isna().any():
        missing = joined.loc[joined["anchor_candidate_support"].isna(), group_columns].head(5)
        raise ValueError(f"canonical panel does not cover label cohorts: {missing.to_dict(orient='records')}")
    if context.duplicated(group_columns).any() or joined.duplicated(group_columns).any():
        raise ValueError("context cohort identity is not one-to-one")
    expected = context_feature_columns()
    if tuple(column for column in joined if column.startswith("context__")) != expected:
        raise ValueError("context feature order drifted from the fixed contract")
    return joined.sort_values(group_columns, kind="stable").reset_index(drop=True)


def _source_hashes(panel_source: Path, label_source: Path) -> tuple[dict[str, Any], dict[str, str]]:
    panel_manifest, panel_hashes = _artifact_manifest(
        panel_source, "canonical_opportunity_payoff_trust_panel_v2"
    )
    label_manifest, label_hashes = _artifact_manifest(
        label_source, "canonical_economic_conversion_transition_labels_v1"
    )
    panel_path = panel_source / "panel.parquet"
    label_path = label_source / "cohort_transition_labels.parquet"
    if not panel_path.is_file() or not label_path.is_file():
        raise FileNotFoundError("immutable source artifact lacks its material parquet")
    if label_manifest.get("source") != str(panel_source):
        raise ValueError("transition labels are not bound to the supplied canonical panel")
    expected_panel_hash = label_manifest.get("source_sha256", {}).get(str(panel_path))
    actual_panel_hash = sha256(panel_path)
    if expected_panel_hash != actual_panel_hash:
        raise ValueError("transition labels do not bind the exact supplied panel.parquet hash")
    source_hashes = {
        **panel_hashes,
        **label_hashes,
        str(panel_path): actual_panel_hash,
        str(label_path): sha256(label_path),
    }
    return {"panel": panel_manifest, "labels": label_manifest}, source_hashes


def plan(panel_source: Path, label_source: Path, output: Path) -> dict[str, Any]:
    manifests, hashes = _source_hashes(panel_source, label_source)
    return {
        "action": "PLAN_ONLY_NO_MATERIALIZATION",
        "schema": SCHEMA,
        "panel_source": str(panel_source),
        "label_source": str(label_source),
        "output": str(output),
        "source_sha256": hashes,
        "expected_panel_identity_sha256": manifests["panel"].get("identity_sha256"),
        "cohort_identity": "anchor UTC × side × frozen score-only base-score decile",
        "aggregation": "mean of exact anchor-time candidate values; support is audit-only",
        "feature_contract": {
            "base_score_and_context": list(BASE_CONTEXT_COLUMNS),
            "core_market_levels": list(CORE_MARKET_COLUMNS),
            "past_only_transition_deltas": list(TRANSITION_COLUMNS),
            "selected_compact_regime_composites": list(COMPACT_REGIME_COLUMNS),
            "derived_side_and_score_cohort_context": list(DERIVED_COHORT_CONTEXT_COLUMNS),
            "feature_columns": list(context_feature_columns()),
            "excluded": "all outcomes, labels, exit fields, causal mappings, and mapped fields",
        },
        "contracts": {
            "utc": "all persisted timestamps are timezone-aware UTC",
            "deciles": "identical score-descending, symbol/candidate-id tie-broken deterministic logic to canonical labels",
            "immutable_output": "refuse existing output; atomically publish a new immutable artifact",
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    panel_source = Path(args.panel_source)
    label_source = Path(args.label_source)
    output = Path(args.output_dir)
    if args.plan_only:
        return plan(panel_source, label_source, output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite immutable output {output}")
    manifests, hashes = _source_hashes(panel_source, label_source)
    panel_columns = [*IDENTITY, *DECISION_OBSERVABLE_COLUMNS]
    label_columns = list(COHORT_KEY)
    panel = pd.read_parquet(panel_source / "panel.parquet", columns=panel_columns)
    labels = pd.read_parquet(label_source / "cohort_transition_labels.parquet", columns=label_columns)
    actual_identity = _panel_identity_sha256(panel)
    if actual_identity != manifests["panel"].get("identity_sha256"):
        raise ValueError("canonical panel identity hash differs from its immutable manifest")
    context = materialize_context(panel, labels)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(dir=output.parent, prefix=f".{output.name}."))
    context.to_parquet(temporary / "cohort_transition_context.parquet", index=False, compression="zstd")
    coverage = (
        context.assign(month=context["cohort_anchor_utc"].dt.strftime("%Y-%m"))
        .groupby(["month", "side_name"], observed=True, sort=True)
        .agg(
            cohort_rows=("cohort_anchor_utc", "size"),
            anchor_candidate_rows=("anchor_candidate_support", "sum"),
            min_anchor_candidates=("anchor_candidate_support", "min"),
            max_anchor_candidates=("anchor_candidate_support", "max"),
        )
        .reset_index()
    )
    coverage.to_parquet(temporary / "coverage_by_month_side.parquet", index=False, compression="zstd")
    manifest = {
        "schema": SCHEMA,
        "status": "IMMUTABLE_DECISION_OBSERVABLE_CONTEXT_ARTIFACT",
        "rows": int(len(context)),
        "cohort_key": list(COHORT_KEY),
        "context_feature_columns": list(context_feature_columns()),
        "feature_sources": {
            "base_score_and_context": list(BASE_CONTEXT_COLUMNS),
            "core_market_levels": list(CORE_MARKET_COLUMNS),
            "past_only_transition_deltas": list(TRANSITION_COLUMNS),
            "selected_compact_regime_composites": list(COMPACT_REGIME_COLUMNS),
            "derived_side_and_score_cohort_context": list(DERIVED_COHORT_CONTEXT_COLUMNS),
        },
        "source_panel_identity_sha256": actual_identity,
        "source_artifacts_sha256": hashes,
        "source_manifest_schemas": {
            "panel": manifests["panel"].get("schema"),
            "labels": manifests["labels"].get("schema"),
        },
        "contracts": {
            "utc": "all persisted timestamps are timezone-aware UTC",
            "cohort": "exact label cohort membership recomputed from base_oof_score only at each anchor timestamp/side",
            "aggregation": "only anchor-time rows, arithmetic mean per explicitly whitelisted decision-observable source field",
            "forbidden": "outcomes, labels, exit fields, causal mappings, mapped fields, and conditional targets never enter feature columns",
            "immutable": "existing outputs are refused; manifest uses detached SHA256 sidecar",
        },
        "coverage_by_month_side": coverage.to_dict(orient="records"),
        "outputs_sha256": {
            path.name: sha256(path) for path in sorted(temporary.glob("*.parquet"))
        },
        "checksum_convention": "manifest.json is verified by detached manifest.sha256",
    }
    (temporary / "manifest.json").write_text(
        json.dumps(_safe(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (temporary / "manifest.sha256").write_text(
        f"{sha256(temporary / 'manifest.json')}  manifest.json\n", encoding="utf-8"
    )
    os.replace(temporary, output)
    return {"output": str(output), "rows": int(len(context)), "source_sha256": hashes}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--panel-source", type=Path, default=PANEL_SOURCE)
    result.add_argument("--label-source", type=Path, default=LABEL_SOURCE)
    result.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    result.add_argument(
        "--plan-only",
        action="store_true",
        help="Validate immutable manifests/hashes and print the contract without reading panel rows or writing output.",
    )
    return result


def main() -> None:
    print(json.dumps(_safe(run(parser().parse_args())), sort_keys=True))


if __name__ == "__main__":
    main()
