"""Pure artifact assembly for the base-model portability diagnosis.

This module intentionally sits *after* materialisation and diagnostics.  It
does not open a candidate store, construct labels, score a model, or refit
anything.  Its only jobs are to combine already-computed relationship, input
population, and paired-score diagnostics into an explicit gate scorecard and
to publish that decision as an immutable, hash-bound artifact.

Keeping this boundary strict matters: a portability report is evidence about
a frozen experiment, not another opportunity to tune it.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .base_monthly_drift_diagnosis import (
    BaseMonthlyDriftDiagnosticError,
    DriftAttributionThresholds,
    classify_drift_attribution,
)


SCHEMA = "base_portability_diagnosis_report_v1"
STATUS = "SEALED_BASE_PORTABILITY_DIAGNOSIS"
SCORECARD_NAME = "base_portability_gate_scorecard.parquet"
SUMMARY_NAME = "base_portability_summary.json"
REPORT_NAME = "BASE_PORTABILITY_DIAGNOSIS_REPORT.md"
MANIFEST_NAME = "run_manifest.json"


class BasePortabilityReportError(ValueError):
    """Raised when supplied diagnostic tables cannot support a sealed report."""


@dataclass(frozen=True)
class BasePortabilityGatePolicy:
    """Predeclared base portability goals and stability tolerances.

    The recall floors are deliberately above random-selection recall (30% and
    40% respectively).  Callers should record a different policy explicitly
    rather than silently changing what "high and stable" means after seeing
    an era result.
    """

    pooled_rank_ic_min: float = 0.02
    positive_era_fraction_strictly_greater_than: float = 0.50
    worst_era_rank_ic_min: float = 0.0
    top30_winner_recall_min: float = 0.40
    top40_winner_recall_min: float = 0.50
    top30_winner_recall_range_max: float = 0.15
    top40_winner_recall_range_max: float = 0.15
    top5_uplift_min: float = 0.0
    max_decile_adjacent_violations: int = 0


@dataclass(frozen=True)
class BasePortabilityReportResult:
    output_dir: Path
    scorecard_path: Path
    summary_path: Path
    report_path: Path
    manifest_path: Path
    portable: bool


def sha256_file(path: str | os.PathLike[str]) -> str:
    """Return a streaming SHA-256 digest for an already-written artifact."""
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataframe_sha256(frame: pd.DataFrame) -> str:
    """Hash a frame's schema and values deterministically, independent of index."""
    ordered = frame.copy().reset_index(drop=True)
    ordered = ordered.reindex(sorted(ordered.columns), axis=1)
    # ``to_json`` makes timestamps and missing values stable across pandas
    # versions more reliably than hashing Python object representations.
    payload = {
        "columns": list(ordered.columns),
        "dtypes": {column: str(dtype) for column, dtype in ordered.dtypes.items()},
        "rows": json.loads(ordered.to_json(orient="values", date_format="iso", date_unit="ns")),
    }
    return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _period_column(frame: pd.DataFrame, *, table: str) -> str:
    for candidate in ("month", "era", "period"):
        if candidate in frame.columns:
            return candidate
    raise BasePortabilityReportError(f"{table} needs one of month, era, or period")


def _require(frame: pd.DataFrame, columns: tuple[str, ...], *, table: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise BasePortabilityReportError(f"{table} lacks required columns: {missing}")
    if frame.empty:
        raise BasePortabilityReportError(f"{table} is empty")


def _finite(frame: pd.DataFrame, columns: tuple[str, ...], *, table: str) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    if not np.isfinite(result.loc[:, list(columns)].to_numpy(dtype=float)).all():
        raise BasePortabilityReportError(f"{table} requires finite numeric values for {list(columns)}")
    return result


def _relationship_rows(metrics: pd.DataFrame) -> pd.DataFrame:
    required = (
        "within_query_rank_ic", "top5_uplift", "top30_winner_recall",
        "top40_winner_recall", "decile_adjacent_violations",
    )
    period = _period_column(metrics, table="relationship_metrics")
    _require(metrics, required, table="relationship_metrics")
    result = _finite(metrics, required, table="relationship_metrics")
    result = result.rename(columns={period: "period"})
    if "scope" in result.columns:
        pooled = result["scope"].astype(str).eq("pooled")
        if pooled.any():
            result = result.loc[pooled].copy()
    if "scope_value" in result.columns:
        all_scope = result["scope_value"].astype(str).eq("all")
        if all_scope.any():
            result = result.loc[all_scope].copy()
    result["period"] = result["period"].astype(str)
    if result["period"].duplicated().any():
        raise BasePortabilityReportError("relationship_metrics must have one pooled row per period")
    columns = ["period", *required]
    if "n_rows" in result.columns:
        result["n_rows"] = pd.to_numeric(result["n_rows"], errors="coerce")
        if not np.isfinite(result["n_rows"].to_numpy(dtype=float)).all() or result["n_rows"].le(0.0).any():
            raise BasePortabilityReportError("relationship_metrics.n_rows must be finite and positive when supplied")
        columns.append("n_rows")
    return result.loc[:, columns].sort_values("period", kind="stable").reset_index(drop=True)


def _input_rows(metrics: pd.DataFrame) -> pd.DataFrame:
    required = ("max_feature_psi", "max_feature_extrapolation_rate")
    period = _period_column(metrics, table="input_drift_metrics")
    _require(metrics, required, table="input_drift_metrics")
    result = _finite(metrics, required, table="input_drift_metrics").rename(columns={period: "period"})
    result["period"] = result["period"].astype(str)
    # Feature-level tables are expected; the relevant population-drift risk is
    # the worst causal input in that period.
    return result.groupby("period", as_index=False, observed=True)[list(required)].max().sort_values("period", kind="stable")


def _score_rows(metrics: pd.DataFrame) -> pd.DataFrame:
    required = (
        "frozen_rank_ic", "refit_rank_ic", "score_spearman", "top_05_overlap_fraction",
        "train_rank_ic", "calibration_slope_shift",
    )
    period = _period_column(metrics, table="score_stability_metrics")
    _require(metrics, required, table="score_stability_metrics")
    result = _finite(metrics, required, table="score_stability_metrics").rename(columns={period: "period"})
    result["period"] = result["period"].astype(str)
    if result["period"].duplicated().any():
        raise BasePortabilityReportError("score_stability_metrics must have one paired-score row per period")
    return result.loc[:, ["period", *required]].sort_values("period", kind="stable").reset_index(drop=True)


def build_base_portability_scorecard(
    relationship_metrics: pd.DataFrame,
    input_drift_metrics: pd.DataFrame,
    score_stability_metrics: pd.DataFrame,
    *,
    policy: BasePortabilityGatePolicy = BasePortabilityGatePolicy(),
    attribution_thresholds: DriftAttributionThresholds = DriftAttributionThresholds(),
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build period and aggregate gates from precomputed diagnostic tables.

    No target or candidate rows are accepted here, deliberately preventing a
    report call from changing fitting, label, or population semantics.
    """
    relationship = _relationship_rows(relationship_metrics)
    inputs = _input_rows(input_drift_metrics)
    stability = _score_rows(score_stability_metrics)
    scorecard = relationship.merge(inputs, on="period", how="left", validate="one_to_one").merge(
        stability, on="period", how="left", validate="one_to_one"
    )
    if scorecard.isna().any().any():
        missing = scorecard.loc[scorecard.isna().any(axis=1), "period"].tolist()
        raise BasePortabilityReportError(f"every relationship period needs input and score diagnostics; missing={missing}")
    attribution_columns = (
        "frozen_rank_ic", "refit_rank_ic", "score_spearman", "top_05_overlap_fraction",
        "max_feature_psi", "max_feature_extrapolation_rate", "train_rank_ic", "calibration_slope_shift",
    )
    try:
        scorecard["drift_attribution"] = [
            classify_drift_attribution(row, thresholds=attribution_thresholds)
            for row in scorecard.loc[:, list(attribution_columns)].to_dict(orient="records")
        ]
    except BaseMonthlyDriftDiagnosticError as exc:  # pragma: no cover - defensive API boundary
        raise BasePortabilityReportError(str(exc)) from exc
    scorecard["rank_ic_positive"] = scorecard["within_query_rank_ic"].gt(0.0)
    scorecard["worst_era_rank_ic_gate"] = scorecard["within_query_rank_ic"].ge(policy.worst_era_rank_ic_min)
    scorecard["top5_uplift_gate"] = scorecard["top5_uplift"].gt(policy.top5_uplift_min)
    scorecard["top30_recall_gate"] = scorecard["top30_winner_recall"].ge(policy.top30_winner_recall_min)
    scorecard["top40_recall_gate"] = scorecard["top40_winner_recall"].ge(policy.top40_winner_recall_min)
    scorecard["decile_monotonic_gate"] = scorecard["decile_adjacent_violations"].le(policy.max_decile_adjacent_violations)
    scorecard["period_goal_pass"] = scorecard.loc[:, [
        "worst_era_rank_ic_gate", "top5_uplift_gate", "top30_recall_gate", "top40_recall_gate", "decile_monotonic_gate",
    ]].all(axis=1)

    weight_col = "n_rows" if "n_rows" in scorecard.columns else None
    if weight_col is not None:
        pooled_ic = float(np.average(scorecard["within_query_rank_ic"], weights=pd.to_numeric(scorecard[weight_col], errors="coerce")))
    else:
        pooled_ic = float(scorecard["within_query_rank_ic"].mean())
    positive_fraction = float(scorecard["rank_ic_positive"].mean())
    top30_range = float(scorecard["top30_winner_recall"].max() - scorecard["top30_winner_recall"].min())
    top40_range = float(scorecard["top40_winner_recall"].max() - scorecard["top40_winner_recall"].min())
    gate_values = {
        "pooled_rank_ic_gate": pooled_ic >= policy.pooled_rank_ic_min,
        "positive_era_fraction_gate": positive_fraction > policy.positive_era_fraction_strictly_greater_than,
        "worst_era_rank_ic_gate": bool(scorecard["worst_era_rank_ic_gate"].all()),
        "top5_uplift_each_era_gate": bool(scorecard["top5_uplift_gate"].all()),
        "top30_recall_each_era_gate": bool(scorecard["top30_recall_gate"].all()),
        "top40_recall_each_era_gate": bool(scorecard["top40_recall_gate"].all()),
        "top30_recall_stability_gate": top30_range <= policy.top30_winner_recall_range_max,
        "top40_recall_stability_gate": top40_range <= policy.top40_winner_recall_range_max,
        "decile_monotonic_each_era_gate": bool(scorecard["decile_monotonic_gate"].all()),
    }
    summary: dict[str, Any] = {
        "schema": SCHEMA,
        "periods": scorecard["period"].tolist(),
        "pooled_within_query_rank_ic": pooled_ic,
        "positive_era_fraction": positive_fraction,
        "worst_era_within_query_rank_ic": float(scorecard["within_query_rank_ic"].min()),
        "top30_winner_recall_range": top30_range,
        "top40_winner_recall_range": top40_range,
        "drift_attribution_counts": scorecard["drift_attribution"].value_counts(sort=False).to_dict(),
        "gates": gate_values,
        "portable": bool(all(gate_values.values())),
        "policy": asdict(policy),
        "attribution_thresholds": asdict(attribution_thresholds),
        "source_sha256": {
            "relationship_metrics": dataframe_sha256(relationship_metrics),
            "input_drift_metrics": dataframe_sha256(input_drift_metrics),
            "score_stability_metrics": dataframe_sha256(score_stability_metrics),
        },
    }
    return scorecard, summary


def render_base_portability_markdown(scorecard: pd.DataFrame, summary: Mapping[str, Any]) -> str:
    """Render a compact human audit; all values remain in the scorecard."""
    gate_rows = "\n".join(
        f"| {name} | {'PASS' if value else 'FAIL'} |"
        for name, value in dict(summary["gates"]).items()
    )
    period_rows = "\n".join(
        f"| {row.period} | {row.within_query_rank_ic:.4f} | {row.top5_uplift:.4f} | "
        f"{row.top30_winner_recall:.3f} | {row.top40_winner_recall:.3f} | "
        f"{int(row.decile_adjacent_violations)} | {row.drift_attribution} | "
        f"{'PASS' if row.period_goal_pass else 'FAIL'} |"
        for row in scorecard.itertuples(index=False)
    )
    return "\n".join((
        "# Base portability diagnosis", "",
        "This is a report over precomputed frozen diagnostics. It did not load candidates, labels, features, or models, and it did not refit a model.", "",
        "## Aggregate decision", "",
        f"Portable under the declared gate policy: **{'YES' if summary['portable'] else 'NO'}**", "",
        f"- Pooled within-query rank IC: `{summary['pooled_within_query_rank_ic']:.6f}`",
        f"- Positive-era fraction: `{summary['positive_era_fraction']:.3f}`",
        f"- Worst-era rank IC: `{summary['worst_era_within_query_rank_ic']:.6f}`",
        f"- Top-30 recall range: `{summary['top30_winner_recall_range']:.3f}`",
        f"- Top-40 recall range: `{summary['top40_winner_recall_range']:.3f}`", "",
        "## Gate scorecard", "", "| Gate | Result |", "|---|---|", gate_rows, "",
        "## Era / month diagnostics", "",
        "| Period | Rank IC | Top-5 uplift | Top-30 recall | Top-40 recall | Decile violations | Drift attribution | Period goal |",
        "|---|---:|---:|---:|---:|---:|---|---|", period_rows, "",
        "`MODEL_DRIFT`, `INPUT_POPULATION_DRIFT`, and `ECONOMIC_RELATIONSHIP_DRIFT` are descriptive evidence labels, not causal proof. Mixed labels mean multiple diagnostics fired.", "",
    ))


def write_immutable_base_portability_report(
    scorecard: pd.DataFrame,
    summary: Mapping[str, Any],
    output_dir: str | os.PathLike[str],
    *,
    provenance: Mapping[str, Any],
) -> BasePortabilityReportResult:
    """Atomically publish a hash-bound report; refuse to overwrite an artifact."""
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite immutable base portability report: {destination}")
    if not provenance:
        raise BasePortabilityReportError("immutable report requires non-empty provenance")
    destination.parent.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        scorecard_path = scratch / SCORECARD_NAME
        scorecard.to_parquet(scorecard_path, index=False)
        summary_path = scratch / SUMMARY_NAME
        summary_path.write_text(json.dumps(dict(summary), indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
        report_path = scratch / REPORT_NAME
        report_path.write_text(render_base_portability_markdown(scorecard, summary), encoding="utf-8")
        output_hashes = {
            SCORECARD_NAME: sha256_file(scorecard_path),
            SUMMARY_NAME: sha256_file(summary_path),
            REPORT_NAME: sha256_file(report_path),
        }
        manifest = {
            "schema": SCHEMA,
            "status": STATUS,
            "artifact_state": "COMPLETE",
            "immutable_output": True,
            "refit_or_data_loading_performed": False,
            "portable": bool(summary.get("portable", False)),
            "provenance": dict(provenance),
            "source_sha256": dict(summary.get("source_sha256", {})),
            "sha256": output_hashes,
        }
        manifest_path = scratch / MANIFEST_NAME
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
        os.replace(scratch, destination)
    except Exception:
        shutil.rmtree(scratch, ignore_errors=True)
        raise
    return BasePortabilityReportResult(
        output_dir=destination,
        scorecard_path=destination / SCORECARD_NAME,
        summary_path=destination / SUMMARY_NAME,
        report_path=destination / REPORT_NAME,
        manifest_path=destination / MANIFEST_NAME,
        portable=bool(summary.get("portable", False)),
    )


def verify_immutable_base_portability_report(output_dir: str | os.PathLike[str]) -> dict[str, Any]:
    """Verify the published report's declared outputs without recalculating it."""
    root = Path(output_dir)
    try:
        manifest = json.loads((root / MANIFEST_NAME).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BasePortabilityReportError(f"missing readable immutable manifest under {root}") from exc
    if manifest.get("schema") != SCHEMA or manifest.get("status") != STATUS or manifest.get("immutable_output") is not True:
        raise BasePortabilityReportError("manifest is not a sealed base portability report")
    hashes = manifest.get("sha256")
    if not isinstance(hashes, Mapping) or set(hashes) != {SCORECARD_NAME, SUMMARY_NAME, REPORT_NAME}:
        raise BasePortabilityReportError("manifest has incomplete output hashes")
    for relative, expected in hashes.items():
        path = root / str(relative)
        if not path.is_file() or sha256_file(path) != str(expected):
            raise BasePortabilityReportError(f"sealed output hash mismatch: {path}")
    return manifest


__all__ = [
    "BasePortabilityGatePolicy", "BasePortabilityReportError", "BasePortabilityReportResult",
    "build_base_portability_scorecard", "dataframe_sha256", "render_base_portability_markdown",
    "sha256_file", "verify_immutable_base_portability_report", "write_immutable_base_portability_report",
]
