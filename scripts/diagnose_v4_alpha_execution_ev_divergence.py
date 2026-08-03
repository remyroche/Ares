#!/usr/bin/env python3
"""Audit alpha-target versus executable 12h EV without mixing lineages.

The score rows are hourly decision rows.  Exact 1-minute data is only used
inside the future 12-hour outcome/replay path; it never changes the assessment
cadence or creates minute-level candidates.

This is deliberately a diagnosis, not a model or policy selector.  Historical
backcasts, inverse contracts, and strict current lineages remain separate in
every calculation.  In particular, no IC, top-decile book, PnL, or average is
ever computed across a lineage boundary.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
IDENTITY = ("candidate_id", "__ts__", "__symbol__", "side_name")
ALPHA_TARGET = "__first_touch_target_soft__"
GROSS = "execution_gross_ev_12h"
COST = "execution_cost_return"
NET = "execution_net_ev_12h"
TOP_FRACTION = 0.10
SCHEMA = "alpha_execution_ev_divergence_lineage_aware_v1"

DEFAULT_BACKFILL = ROOT / "data_perp/artifacts/reconstructed_base_residual_stack_2022_2024_20260730_v4"
DEFAULT_2025 = ROOT / "data_perp/artifacts/marapr2025_all_score_ic_ev_waterfall_20260730_v1"
DEFAULT_2026 = ROOT / "data_perp/artifacts/mayjul2026_exact_allscore_ic_ev_waterfall_20260730_v1"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/alpha_execution_ev_divergence_2022_2026_20260730_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.generic,)):
        return value.item()
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    temporary.write_text(json.dumps(json_safe(value), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _read_manifest(root: Path, expected_schema: str) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != expected_schema:
        raise ValueError(f"{root} schema {manifest.get('schema')!r} != {expected_schema!r}")
    sidecar = root / "manifest.sha256"
    if sidecar.exists() and sidecar.read_text().strip().split()[0] != sha256_file(manifest_path):
        raise ValueError(f"manifest sidecar hash mismatch: {root}")
    return manifest


def _validate_frame(frame: pd.DataFrame, *, source: str) -> pd.DataFrame:
    missing = sorted(set(IDENTITY + (ALPHA_TARGET, GROSS, COST, NET)).difference(frame.columns))
    if missing:
        raise ValueError(f"{source} missing required columns: {missing}")
    work = frame.copy()
    work["__ts__"] = pd.to_datetime(work["__ts__"], utc=True, errors="raise")
    if work.duplicated(list(IDENTITY)).any() or work["candidate_id"].duplicated().any():
        raise ValueError(f"{source} does not have one-to-one candidate identity")
    for column in (ALPHA_TARGET, GROSS, COST, NET, "alpha_score", "residual_score"):
        if column in work:
            work[column] = pd.to_numeric(work[column], errors="coerce")
    required_numeric = (ALPHA_TARGET, GROSS, COST, NET, "alpha_score", "residual_score")
    if work.loc[:, required_numeric].isna().any().any():
        raise ValueError(f"{source} has non-finite score/target/economic data")
    if not np.allclose(
        work[GROSS].to_numpy(float) - work[COST].to_numpy(float),
        work[NET].to_numpy(float), atol=1e-10, rtol=0.0,
    ):
        raise ValueError(f"{source} violates gross - cost = net")
    deltas = work["__ts__"].sort_values().diff().dropna().dt.total_seconds().div(3600)
    # A source can have candidate gaps, but no off-grid score timestamp is permitted.
    if not (work["__ts__"].dt.minute.eq(0) & work["__ts__"].dt.second.eq(0)).all():
        raise ValueError(f"{source} contains non-hourly decision timestamps")
    return work


def load_sources(backfill_root: Path, root_2025: Path, root_2026: Path) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    backfill_manifest = _read_manifest(backfill_root, "historical_base_residual_stack_calendar_block_oof_v1")
    bpath = backfill_root / "oof_scores.parquet"
    backfill = pd.read_parquet(bpath).rename(columns={
        "__reconstructed_soft_alpha_12h__": ALPHA_TARGET,
        "score_residual_alpha": "alpha_score",
        "score_residual_expected_ev": "residual_score",
    })
    backfill["lineage_id"] = backfill.pop("stack_lineage")
    backfill["evidence_grade"] = np.where(
        backfill["lineage_id"].eq("inverse_pi_2022_h1"),
        "C_RESEARCH_OOF_SEPARATE_POPULATION",
        "B_RESEARCH_RESIDUAL_OOF_BASE_BACKCAST",
    )
    backfill["alpha_score_origin"] = "score_residual_alpha"
    backfill["residual_score_origin"] = "score_residual_expected_ev"
    backfill["candidate_population"] = np.where(
        backfill["lineage_id"].eq("inverse_pi_2022_h1"),
        "inverse_pi_hourly_grid", "frozen_pf_base_monitor_population",
    )
    backfill = _validate_frame(backfill, source="historical reconstructed backfill")

    sources: list[tuple[Path, str, str, str, str]] = [
        (bpath, "historical_reconstructed", "RESEARCH_OOF_BACKFILL_COMPLETE", "historical base backcast / current-spread counterfactual where applicable", backfill_manifest.get("targets", {}).get("alpha", "")),
    ]
    parts = [backfill]

    manifest_2025 = _read_manifest(root_2025, "marapr2025_all_score_ic_ev_waterfall_v1")
    p2025 = root_2025 / "all_score_waterfall.parquet"
    strict_2025 = pd.read_parquet(p2025).assign(
        lineage_id="canonical_marapr2025_strict_residual_oof",
        evidence_grade="A_STRICT_OOF_EXACT_POLICY",
        alpha_score=lambda x: x["score_base_alpha"],
        residual_score=lambda x: x["score_residual_expected_ev"],
        alpha_score_origin="score_base_alpha (legacy native-24h alpha)",
        residual_score_origin="score_residual_expected_ev",
        candidate_population="canonical_febapr2025_top40",
    )
    strict_2025 = _validate_frame(strict_2025, source="March-April 2025 strict source")
    parts.append(strict_2025)
    sources.append((p2025, "canonical_marapr2025_strict_residual_oof", manifest_2025.get("status", ""), "strict residual OOF + exact 1m policy economics", "native first-touch soft target"))

    manifest_2026 = _read_manifest(root_2026, "mayjul2026_exact_allscore_ic_ev_waterfall_v1")
    p2026 = root_2026 / "allscore_waterfall.parquet"
    strict_2026 = pd.read_parquet(p2026).assign(
        lineage_id="current_mayjul2026_strict_residual_oof",
        evidence_grade="A_STRICT_OOF_EXACT_POLICY",
        alpha_score=lambda x: x["score_base_alpha"],
        residual_score=lambda x: x["score_residual_expected_ev"],
        alpha_score_origin="score_base_alpha",
        residual_score_origin="score_residual_expected_ev",
        candidate_population="current_packb31_8_top40",
    )
    strict_2026 = _validate_frame(strict_2026, source="May-July 2026 strict source")
    parts.append(strict_2026)
    sources.append((p2026, "current_mayjul2026_strict_residual_oof", manifest_2026.get("status", ""), "strict residual OOF + exact 1m policy economics", "native first-touch soft target"))

    output = pd.concat(parts, ignore_index=True, sort=False)
    if output.duplicated(["lineage_id", *IDENTITY]).any():
        raise ValueError("duplicate four-key identity inside a lineage")
    registry = []
    for path, lineage, status, evidence, target in sources:
        local = output.loc[output["lineage_id"].eq(lineage)] if lineage != "historical_reconstructed" else output.loc[output["evidence_grade"].str.startswith(("B_", "C_"))]
        for (actual_lineage, grade), group in local.groupby(["lineage_id", "evidence_grade"], sort=True, observed=True):
            registry.append({
                "lineage_id": actual_lineage,
                "evidence_grade": grade,
                "source_path": str(path.resolve()), "source_sha256": sha256_file(path),
                "source_manifest_sha256": sha256_file(path.parent / "manifest.json"),
                "source_status": status, "evidence": evidence,
                "alpha_target_contract": target,
                "rows": int(len(group)), "first_signal_utc": group["__ts__"].min(),
                "last_signal_utc": group["__ts__"].max(),
                "candidate_population": str(group["candidate_population"].iloc[0]),
                "alpha_score_origin": str(group["alpha_score_origin"].iloc[0]),
                "residual_score_origin": str(group["residual_score_origin"].iloc[0]),
                "assessment_cadence": "1h decision rows; exact 1m only inside the 12h label/replay path",
                "cross_lineage_pooling_allowed": False,
            })
    return output, registry


def spearman_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    """One rank pass per reporting group, rather than one pass per metric."""
    columns = ("alpha_score", "residual_score", ALPHA_TARGET, GROSS, COST, NET)
    local = frame.loc[:, columns].dropna()
    if len(local) < 2:
        return pd.DataFrame(np.nan, index=columns, columns=columns)
    # Pearson correlation of average ranks is Spearman correlation.  This keeps
    # a full weekly/monthly/side audit tractable without changing the metric.
    return local.rank(method="average").corr(method="pearson")


def matrix_value(matrix: pd.DataFrame, left: str, right: str) -> float:
    value = matrix.loc[left, right]
    return float(value) if np.isfinite(value) else float("nan")


def period_key(timestamp: pd.Series, cadence: str) -> pd.Series:
    if cadence == "month":
        return timestamp.dt.strftime("%Y-%m")
    if cadence == "week":
        return timestamp.dt.strftime("%G-W%V")
    raise ValueError(f"unsupported cadence {cadence}")


def stable_top_mask(frame: pd.DataFrame, score: str) -> pd.Series:
    """Pooled global top 10% across sides and timestamps within one period."""
    n = int(math.ceil(len(frame) * TOP_FRACTION))
    ordered = frame.sort_values([score, "candidate_id"], ascending=[False, True], kind="stable")
    selected = pd.Series(False, index=frame.index)
    selected.loc[ordered.index[:n]] = True
    return selected


def metric_row(
    frame: pd.DataFrame, *, selected: bool, population_rows: int | None = None,
) -> dict[str, Any]:
    if frame.empty:
        return {"candidate_rows": 0, "selected_rows": 0 if selected else None}
    rank = spearman_matrix(frame)
    row: dict[str, Any] = {
        "candidate_rows": int(population_rows if population_rows is not None else len(frame)),
        "selected_rows": int(len(frame)) if selected else None,
        "alpha_rank_ic": matrix_value(rank, "alpha_score", ALPHA_TARGET),
        "residual_rank_ic": matrix_value(rank, "residual_score", ALPHA_TARGET),
        "alpha_to_gross_rank_ic": matrix_value(rank, "alpha_score", GROSS),
        "alpha_to_cost_rank_ic": matrix_value(rank, "alpha_score", COST),
        "alpha_to_net_rank_ic": matrix_value(rank, "alpha_score", NET),
        "residual_to_gross_rank_ic": matrix_value(rank, "residual_score", GROSS),
        "residual_to_cost_rank_ic": matrix_value(rank, "residual_score", COST),
        "residual_to_net_rank_ic": matrix_value(rank, "residual_score", NET),
        "first_touch_to_gross_rank_ic": matrix_value(rank, ALPHA_TARGET, GROSS),
        "first_touch_to_cost_rank_ic": matrix_value(rank, ALPHA_TARGET, COST),
        "first_touch_to_net_rank_ic": matrix_value(rank, ALPHA_TARGET, NET),
        "mean_alpha_target": float(frame[ALPHA_TARGET].mean()),
        "mean_gross_bps": float(frame[GROSS].mean() * 1e4),
        "mean_cost_bps": float(frame[COST].mean() * 1e4),
        "mean_net_bps": float(frame[NET].mean() * 1e4),
        "positive_net_rate": float(frame[NET].gt(0).mean()),
        "gross_cost_net_reconciles": bool(np.allclose(frame[GROSS] - frame[COST], frame[NET], atol=1e-10, rtol=0.0)),
    }
    return row


def build_tables(rows: pd.DataFrame, cadences: Iterable[str] = ("month", "week")) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    period_rows: list[dict[str, Any]] = []
    decile_rows: list[dict[str, Any]] = []
    lineage_rows: list[dict[str, Any]] = []
    for (lineage, grade), lineage_frame in rows.groupby(["lineage_id", "evidence_grade"], sort=True, observed=True):
        lineage_rows.append({
            "lineage_id": lineage, "evidence_grade": grade,
            "candidate_rows": int(len(lineage_frame)), "first_signal_utc": lineage_frame["__ts__"].min(),
            "last_signal_utc": lineage_frame["__ts__"].max(),
            "assessment_cadence": "1h",
            **metric_row(lineage_frame, selected=False),
        })
        for cadence in cadences:
            local = lineage_frame.copy()
            local["period"] = period_key(local["__ts__"], cadence)
            for period, group in local.groupby("period", sort=True, observed=True):
                common = {
                    "lineage_id": lineage, "evidence_grade": grade, "cadence": cadence, "period": period,
                    "period_start_utc": group["__ts__"].min(), "period_end_utc": group["__ts__"].max(),
                    "selection_contract": "pooled_global_top10_across_sides_and_timestamps_within_lineage_period",
                    "assessment_cadence": "1h",
                }
                for score_family, score in (("alpha", "alpha_score"), ("residual", "residual_score")):
                    selected_mask = stable_top_mask(group, score)
                    selected_group = group.loc[selected_mask]
                    period_rows.append({**common, "score_family": score_family, "scope": "all_candidates", "side_name": "all", **metric_row(group, selected=False)})
                    period_rows.append({**common, "score_family": score_family, "scope": "pooled_global_top10", "side_name": "all", **metric_row(selected_group, selected=True, population_rows=len(group))})
                    for side, side_group in group.groupby("side_name", sort=True, observed=True):
                        period_rows.append({**common, "score_family": score_family, "scope": "all_candidates", "side_name": side, **metric_row(side_group, selected=False)})
                    for side, side_group in selected_group.groupby("side_name", sort=True, observed=True):
                        period_rows.append({**common, "score_family": score_family, "scope": "pooled_global_top10", "side_name": side, **metric_row(side_group, selected=True, population_rows=len(group))})
                    ranked = group.sort_values([score, "candidate_id"], kind="stable").copy()
                    ranked["score_decile"] = np.minimum(10, np.ceil(ranked[score].rank(method="first", pct=True) * 10).astype(int))
                    for decile, decile_group in ranked.groupby("score_decile", sort=True, observed=True):
                        decile_rows.append({**common, "score_family": score_family, "score_decile": int(decile), **metric_row(decile_group, selected=False)})
    return pd.DataFrame(period_rows), pd.DataFrame(decile_rows), pd.DataFrame(lineage_rows)


def write_artifact(output_root: Path, rows: pd.DataFrame, registry: list[dict[str, Any]]) -> None:
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {output_root}")
    temporary = output_root.with_name(f".{output_root.name}.{os.getpid()}.partial")
    if temporary.exists():
        raise FileExistsError(f"staging path already exists: {temporary}")
    temporary.mkdir(parents=True)
    try:
        period, deciles, lineage = build_tables(rows)
        period.to_parquet(temporary / "period_metrics.parquet", index=False)
        deciles.to_parquet(temporary / "score_deciles.parquet", index=False)
        lineage.to_parquet(temporary / "lineage_summary.parquet", index=False)
        pd.DataFrame(registry).to_csv(temporary / "evidence_registry.csv", index=False)
        key = ["lineage_id", "cadence", "period", "score_family", "scope", "side_name"]
        if period.duplicated(key).any():
            raise ValueError("period metrics failed unique lineage-period-score-scope-side contract")
        report = {
            "schema": SCHEMA, "status": "SEALED_DIAGNOSTIC_NON_PROMOTION", "promotion_eligible": False,
            "assessment_contract": {
                "candidate_cadence": "1h", "outcome_path": "exact 1m future replay over [decision, decision + 12h)",
                "selection": "top 10% is pooled global across sides and timestamps within each lineage and reporting period",
                "lineage_pooling": "forbidden: all metrics are source-lineage-local",
                "alpha_target": "soft first-touch target; target meaning differs only where declared in the evidence registry",
                "economics": "gross - explicit cost = net",
            },
            "rows": int(len(rows)), "lineages": registry,
            "outputs": {name: sha256_file(temporary / name) for name in ("period_metrics.parquet", "score_deciles.parquet", "lineage_summary.parquet", "evidence_registry.csv")},
            "interpretation": {
                "positive_alpha_ic_negative_net": "A positive alpha-to-first-touch rank IC with weak or negative alpha-to-net rank IC, plus weak first-touch-to-net relation, is evidence that the first-touch objective and the direct 12h executable policy outcome differ on these exact rows. It is not causal proof that any one label is wrong.",
                "cost": "Rank IC with cost is not a cost-level forecast claim; mean cost and exact gross-cost-net reconciliation are reported separately.",
                "no_cross_lineage_conclusion": "Do not average, sum, or compare PnL magnitudes across different evidence grades/populations as if they were one backtest.",
            },
        }
        atomic_json(temporary / "report.json", report)
        manifest = {
            "schema": f"{SCHEMA}_manifest", "status": "SEALED_DIAGNOSTIC_NON_PROMOTION",
            "runner": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__))},
            "inputs": registry,
            "outputs_sha256": {name: sha256_file(temporary / name) for name in ("period_metrics.parquet", "score_deciles.parquet", "lineage_summary.parquet", "evidence_registry.csv", "report.json")},
        }
        atomic_json(temporary / "manifest.json", manifest)
        (temporary / "manifest.sha256").write_text(f"{sha256_file(temporary / 'manifest.json')}  manifest.json\n")
        temporary.replace(output_root)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backfill-root", type=Path, default=DEFAULT_BACKFILL)
    parser.add_argument("--strict-2025-root", type=Path, default=DEFAULT_2025)
    parser.add_argument("--strict-2026-root", type=Path, default=DEFAULT_2026)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    rows, registry = load_sources(args.backfill_root, args.strict_2025_root, args.strict_2026_root)
    write_artifact(args.output_root, rows, registry)


if __name__ == "__main__":
    main()
