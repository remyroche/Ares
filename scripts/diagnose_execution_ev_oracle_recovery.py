#!/usr/bin/env python3
"""Diagnose OOS score recovery of the *non-tradable* exact-policy tail.

This is deliberately a diagnostic, not an oracle backtest or a promotion
criterion.  Each arm is ranked only by its already persisted, causally mapped
OOS score.  The future exact net label is used afterwards solely to measure
which profitable candidates the ranker recovered and which it missed.

For each independent forward block, selection is one pooled global book: no
timestamp, symbol, side, or week quota is ever applied.  Window, side, and
week rows are therefore breakdowns of that same selected book, rather than
locally re-ranked books.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


IDENTITY = ("__ts__", "__symbol__", "side_name", "candidate_id")
TARGET_COLUMNS = (
    "execution_gross_ev_12h",
    "execution_cost_return",
    "execution_net_ev_12h",
)
TOP_FRACTIONS = (0.10, 0.05, 0.02)
SURPLUS_BPS = (0.0, 25.0, 50.0, 100.0)
SCHEMA = "execution_ev_oracle_recovery_v1"


@dataclass(frozen=True)
class PredictionSource:
    """An immutable mapped-prediction artifact to be joined to exact labels."""

    name: str
    path: Path
    manifest: Path
    output_key: str
    mapping_stage: str | None = None


DEFAULT_TARGET = Path(
    "data_perp/artifacts/execution_ev_canonical_exact_policy_regime_input_20260727_v3/"
    "joined.parquet"
)
DEFAULT_TARGET_MANIFEST = DEFAULT_TARGET.with_name("manifest.json")
DEFAULT_SOURCES = (
    PredictionSource(
        "capture_support",
        Path("data_perp/artifacts/exact_policy_capture_support_ablation_20260727_v8/capture_support_predictions.parquet"),
        Path("data_perp/artifacts/exact_policy_capture_support_ablation_20260727_v8/manifest.json"),
        "predictions",
        "canonical_recent_ev_mapping",
    ),
    PredictionSource(
        "decomposed_hurdle",
        Path("data_perp/artifacts/exact_policy_decomposed_hurdle_ablation_20260727_v3/hurdle_predictions.parquet"),
        Path("data_perp/artifacts/exact_policy_decomposed_hurdle_ablation_20260727_v3/manifest.json"),
        "predictions",
    ),
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> Mapping[str, Any]:
    with path.open(encoding="utf-8") as handle:
        result = json.load(handle)
    if not isinstance(result, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return result


def _require(value: bool, message: str) -> None:
    if not value:
        raise ValueError(message)


def _canonical_identity(
    frame: pd.DataFrame, *, source: Path, check_duplicates: bool = True
) -> pd.DataFrame:
    missing = sorted(set(IDENTITY).difference(frame.columns))
    _require(not missing, f"identity missing from {source}: {missing}")
    out = frame.copy()
    out["__ts__"] = pd.to_datetime(out["__ts__"], utc=True, errors="raise")
    for column in ("__symbol__", "side_name", "candidate_id"):
        out[column] = out[column].astype(str)
    out["side_name"] = out["side_name"].str.lower()
    _require(out["side_name"].isin(("long", "short")).all(), f"bad side in {source}")
    if check_duplicates:
        _require(not out.duplicated(list(IDENTITY)).any(), f"duplicate identities in {source}")
    return out


def load_exact_targets(path: Path, manifest_path: Path) -> tuple[pd.DataFrame, Mapping[str, Any]]:
    """Hash-check canonical exact labels and enforce exact accounting."""
    manifest = _read_json(manifest_path)
    _require(manifest.get("schema") == "canonical_exact_policy_regime_input_v1",
             f"unexpected target schema: {manifest_path}")
    output = manifest.get("output")
    _require(isinstance(output, dict), f"missing target output lineage: {manifest_path}")
    _require(output.get("sha256") == _sha(path), f"target hash mismatch: {path}")
    _require(output.get("path") == str(path), f"target path mismatch: {manifest_path}")
    contract = manifest.get("contract")
    _require(isinstance(contract, dict), f"missing target contract: {manifest_path}")
    _require(contract.get("target_policy") == "one exact 1m deployed-policy replay source for all dates",
             f"target is not canonical exact policy: {manifest_path}")
    _require("gross - cost = net" in str(contract.get("accounting")),
             f"target accounting contract mismatch: {manifest_path}")
    target = pd.read_parquet(path, columns=list(IDENTITY + TARGET_COLUMNS))
    target = _canonical_identity(target, source=path)
    values = target.loc[:, TARGET_COLUMNS].to_numpy(dtype=float)
    _require(np.isfinite(values).all(), f"nonfinite exact target in {path}")
    _require(np.allclose(
        target["execution_gross_ev_12h"] - target["execution_cost_return"],
        target["execution_net_ev_12h"], rtol=0.0, atol=1e-10,
    ), f"gross-cost-net reconciliation failed: {path}")
    return target, manifest


def load_prediction_source(
    source: PredictionSource,
    *,
    target_manifest: Mapping[str, Any],
    expected_ids: set[tuple[Any, ...]],
) -> tuple[pd.DataFrame, Mapping[str, Any]]:
    """Fail closed on mapped-prediction and canonical-input provenance."""
    manifest = _read_json(source.manifest)
    output = manifest.get("outputs")
    _require(isinstance(output, dict) and isinstance(output.get(source.output_key), dict),
             f"missing prediction output lineage: {source.manifest}")
    _require(output[source.output_key].get("sha256") == _sha(source.path),
             f"prediction hash mismatch: {source.path}")
    inputs = manifest.get("inputs")
    _require(isinstance(inputs, dict) and isinstance(inputs.get("data"), dict),
             f"missing prediction target lineage: {source.manifest}")
    target_output = target_manifest["output"]
    _require(inputs["data"].get("sha256") == target_output["sha256"],
             f"prediction does not use canonical exact input: {source.manifest}")
    _require(inputs["data"].get("path") == target_output["path"],
             f"prediction target path mismatch: {source.manifest}")
    contract = manifest.get("contract")
    _require(isinstance(contract, dict), f"missing prediction contract: {source.manifest}")
    _require("one pooled global top" in str(contract.get("ranking", "")).lower(),
             f"prediction is not pooled-global ranked: {source.manifest}")

    columns = list(IDENTITY) + ["window", "arm", "canonical_recent_ev_score"]
    if source.mapping_stage is not None:
        columns.append("mapping_stage")
    predictions = pd.read_parquet(source.path, columns=columns)
    if source.mapping_stage is not None:
        predictions = predictions.loc[
            predictions["mapping_stage"].eq(source.mapping_stage)
        ].drop(columns="mapping_stage")
    predictions = _canonical_identity(
        predictions, source=source.path, check_duplicates=False
    )
    _require(predictions["window"].notna().all() and predictions["arm"].notna().all(),
             f"missing arm/window in {source.path}")
    _require(np.isfinite(predictions["canonical_recent_ev_score"].to_numpy(dtype=float)).all(),
             f"nonfinite mapped score in {source.path}")
    _require(not predictions.duplicated(list(IDENTITY) + ["window", "arm"]).any(),
             f"duplicate arm identity in {source.path}")
    ids = set(map(tuple, predictions.loc[:, IDENTITY].itertuples(index=False, name=None)))
    _require(ids == expected_ids,
             f"prediction identity coverage differs from expected OOS cohort: {source.path} "
             f"({len(ids)} versus {len(expected_ids)})")
    predictions["source"] = source.name
    predictions["diagnostic_arm"] = source.name + "::" + predictions["arm"].astype(str)
    return predictions, manifest


def select_global_topk(frame: pd.DataFrame, fraction: float) -> pd.Index:
    """Exact pooled-global top-k rule used by the existing mapped-arm metrics.

    ``execution_ev_metrics`` uses stable ascending ``argsort`` followed by the
    final k positions.  Retaining that input-order tie rule is essential here:
    isotonic mappings can create many equal scores, and an attractive new
    candidate-ID tiebreak would silently diagnose a different traded book.
    """
    if frame.empty:
        return frame.index
    count = max(1, int(math.ceil(len(frame) * float(fraction))))
    positions = np.argsort(
        frame["canonical_recent_ev_score"].to_numpy(dtype=float), kind="stable"
    )[-count:]
    return frame.index.take(positions)


def oracle_topk(frame: pd.DataFrame, fraction: float) -> pd.Index:
    """Same-k non-tradable future-net oracle, for post-hoc recovery only."""
    if frame.empty:
        return frame.index
    count = max(1, int(math.ceil(len(frame) * float(fraction))))
    positions = np.argsort(
        frame["execution_net_ev_12h"].to_numpy(dtype=float), kind="stable"
    )[-count:]
    return frame.index.take(positions)


def _safe_divide(num: float, den: float) -> float:
    return float(num / den) if den else float("nan")


def _display_groups(frame: pd.DataFrame) -> Iterable[tuple[str, str, pd.Series]]:
    yield "overall", "all", pd.Series(True, index=frame.index)
    for side in ("long", "short"):
        yield "side", side, frame["side_name"].eq(side)
    for week in sorted(frame["week"].unique()):
        yield "week", str(week), frame["week"].eq(week)
    latest_week = str(frame["week"].max())
    yield "latest_week", latest_week, frame["week"].eq(latest_week)


def recovery_rows(frame: pd.DataFrame, *, top_fraction: float, surplus_bps: float) -> list[dict[str, Any]]:
    """Recovery metrics for a global selected book, displayed in causal slices."""
    selected_index = select_global_topk(frame, top_fraction)
    oracle_index = oracle_topk(frame, top_fraction)
    selected = pd.Series(False, index=frame.index)
    selected.loc[selected_index] = True
    oracle = pd.Series(False, index=frame.index)
    oracle.loc[oracle_index] = True
    event = frame["execution_net_ev_12h"].gt(float(surplus_bps) / 10_000.0)
    rows: list[dict[str, Any]] = []
    for grouping, group_value, group_mask in _display_groups(frame):
        subset = frame.loc[group_mask]
        selected_mask = selected.loc[group_mask]
        oracle_mask = oracle.loc[group_mask]
        event_mask = event.loc[group_mask]
        selected_rows = int(selected_mask.sum())
        event_rows = int(event_mask.sum())
        true_positive = int((selected_mask & event_mask).sum())
        false_positive = int((selected_mask & ~event_mask).sum())
        missed = (~selected_mask) & event_mask
        selected_oracle = int((selected_mask & oracle_mask).sum())
        oracle_rows = int(oracle_mask.sum())
        union = int((selected_mask | oracle_mask).sum())
        fp = subset.loc[selected_mask & ~event_mask, "execution_net_ev_12h"] * 10_000.0
        missed_net = subset.loc[missed, "execution_net_ev_12h"] * 10_000.0
        baseline = _safe_divide(event_rows, len(subset))
        precision = _safe_divide(true_positive, selected_rows)
        rows.append({
            "top_fraction": float(top_fraction),
            "surplus_bps": float(surplus_bps),
            "grouping": grouping,
            "group_value": group_value,
            "candidate_rows": int(len(subset)),
            "selected_rows": selected_rows,
            "selected_rows_global": int(selected.sum()),
            "oracle_topk_rows": oracle_rows,
            "surplus_event_rows": event_rows,
            "true_positive_rows": true_positive,
            "false_positive_rows": false_positive,
            "missed_winner_rows": int(missed.sum()),
            "event_prevalence": baseline,
            "precision": precision,
            "surplus_recall": _safe_divide(true_positive, event_rows),
            "lift_vs_prevalence": _safe_divide(precision, baseline),
            "false_positive_rate_among_selected": _safe_divide(false_positive, selected_rows),
            "false_positive_mean_net_bps": float(fp.mean()) if len(fp) else float("nan"),
            "false_positive_mean_shortfall_bps": float((float(surplus_bps) - fp).mean()) if len(fp) else float("nan"),
            "false_positive_shortfall_sum_bps": float((float(surplus_bps) - fp).sum()),
            "missed_winner_mean_net_bps": float(missed_net.mean()) if len(missed_net) else float("nan"),
            "missed_winner_net_sum_bps": float(missed_net.sum()),
            "oracle_topk_overlap_rows": selected_oracle,
            "oracle_topk_recall": _safe_divide(selected_oracle, oracle_rows),
            "oracle_topk_precision": _safe_divide(selected_oracle, selected_rows),
            "oracle_topk_jaccard": _safe_divide(selected_oracle, union),
            "selected_mean_net_bps": float((subset.loc[selected_mask, "execution_net_ev_12h"] * 10_000.0).mean()) if selected_rows else float("nan"),
            "oracle_topk_mean_net_bps": float((subset.loc[oracle_mask, "execution_net_ev_12h"] * 10_000.0).mean()) if oracle_rows else float("nan"),
        })
    return rows


def coverage_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Evidence that every requested arm reaches both OOS blocks and the latest week."""
    rows: list[dict[str, Any]] = []
    for (arm, window), part in frame.groupby(["diagnostic_arm", "window"], sort=True):
        rows.append({
            "diagnostic_arm": arm,
            "window": window,
            "candidate_rows": int(len(part)),
            "first_timestamp_utc": part["__ts__"].min().isoformat(),
            "last_timestamp_utc": part["__ts__"].max().isoformat(),
            "weeks": int(part["week"].nunique()),
            "sides": int(part["side_name"].nunique()),
            "latest_week": str(part["week"].max()),
            "latest_week_rows": int(part["week"].eq(part["week"].max()).sum()),
        })
    return rows


def run(*, target: Path, target_manifest: Path, sources: Sequence[PredictionSource], output_dir: Path) -> Mapping[str, Any]:
    exact, exact_manifest = load_exact_targets(target, target_manifest)
    prediction_parts: list[pd.DataFrame] = []
    provenance: dict[str, Any] = {}
    expected_ids: set[tuple[Any, ...]] | None = None
    for source in sources:
        # The first verified source defines the current OOS cohort.  Every
        # subsequent arm must have exactly the same identities, not merely a
        # successful inner join.
        if expected_ids is None:
            raw_columns = list(IDENTITY)
            if source.mapping_stage is not None:
                raw_columns.append("mapping_stage")
            raw = pd.read_parquet(source.path, columns=raw_columns)
            if source.mapping_stage is not None:
                raw = raw.loc[raw["mapping_stage"].eq(source.mapping_stage)].drop(
                    columns="mapping_stage"
                )
            # The artifact has one copy per arm.  This first pass establishes
            # the common cohort only; duplicate identity across arms is
            # expected, whereas duplicate identity *within an arm* is checked
            # in load_prediction_source below.
            raw = raw.drop_duplicates(list(IDENTITY))
            raw = _canonical_identity(raw, source=source.path)
            expected_ids = set(map(tuple, raw.loc[:, IDENTITY].itertuples(index=False, name=None)))
        prediction, manifest = load_prediction_source(
            source, target_manifest=exact_manifest, expected_ids=expected_ids
        )
        prediction_parts.append(prediction)
        provenance[source.name] = {
            "prediction_path": str(source.path),
            "prediction_sha256": _sha(source.path),
            "manifest_path": str(source.manifest),
            "manifest_sha256": _sha(source.manifest),
            "arms": sorted(prediction["arm"].unique().tolist()),
            "mapping_stage": source.mapping_stage or "canonical_recent_ev_mapping",
        }
    _require(expected_ids is not None, "at least one prediction source is required")
    label_ids = set(map(tuple, exact.loc[:, IDENTITY].itertuples(index=False, name=None)))
    _require(expected_ids.issubset(label_ids), "OOS prediction identities missing canonical exact labels")
    predictions = pd.concat(prediction_parts, ignore_index=True)
    merged = predictions.merge(exact, on=list(IDENTITY), how="left", validate="many_to_one")
    _require(merged.loc[:, TARGET_COLUMNS].notna().all().all(), "missing exact targets after identity join")
    merged["week"] = merged["__ts__"].dt.tz_localize(None).dt.to_period("W-SUN").astype(str)

    metric_rows: list[dict[str, Any]] = []
    for (arm, window), part in merged.groupby(["diagnostic_arm", "window"], sort=True):
        for fraction in TOP_FRACTIONS:
            for surplus in SURPLUS_BPS:
                for row in recovery_rows(part, top_fraction=fraction, surplus_bps=surplus):
                    row["diagnostic_arm"] = arm
                    row["window"] = window
                    metric_rows.append(row)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "oracle_recovery_metrics.csv"
    coverage_path = output_dir / "coverage.csv"
    pd.DataFrame(metric_rows).to_csv(metrics_path, index=False)
    pd.DataFrame(coverage_rows(merged)).to_csv(coverage_path, index=False)
    manifest = {
        "schema": SCHEMA,
        "status": "completed_research_oos_diagnostic_not_promotion_evidence",
        "contract": {
            "selection": "one pooled global top-k within each independent forward block; no timestamp, side, symbol, or week quotas",
            "oracle": "future exact net is used only after score ranking for diagnosis; no hindsight action, threshold, HPO, or promotion",
            "events": "realized exact net strictly greater than 0/25/50/100 bps",
            "accounting": "canonical exact deployed-policy gross - cost = net",
        },
        "inputs": {
            "canonical_exact_targets": {
                "path": str(target), "sha256": _sha(target),
                "manifest": str(target_manifest), "manifest_sha256": _sha(target_manifest),
            },
            "mapped_prediction_sources": provenance,
        },
        "cohort": {"rows": int(len(expected_ids)), "windows": sorted(merged["window"].unique().tolist())},
        "outputs": {
            "metrics": {"path": str(metrics_path), "sha256": _sha(metrics_path)},
            "coverage": {"path": str(coverage_path), "sha256": _sha(coverage_path)},
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--target-manifest", type=Path, default=DEFAULT_TARGET_MANIFEST)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


if __name__ == "__main__":
    args = _parser().parse_args()
    print(json.dumps(run(target=args.target, target_manifest=args.target_manifest, sources=DEFAULT_SOURCES, output_dir=args.output_dir), indent=2))
