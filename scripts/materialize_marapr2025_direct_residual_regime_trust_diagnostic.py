#!/usr/bin/env python3
"""Materialise the causal March/April direct-versus-residual regime diagnosis.

This is a reused-month, diagnostic-only artifact.  It fits no trading model,
selects no gate, and changes no scores, labels, costs, exits, or action layer.
Every regime/transition value is joined from the candidate signal timestamp.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BRIDGE_ROOT = (
    ROOT / "data_perp/artifacts/marapr2025_identical_causal_score_bridge_20260730_v1"
)
SOFT_ROOT = (
    ROOT / "data_perp/artifacts/authoritative_soft_regime_transition_sidecars_20260730_v1"
)
TRAJECTORY_ROOT = (
    ROOT / "data_perp/artifacts/hourly_trajectory_transition_soft_sidecar_20260730_v1"
)
OUTPUT = (
    ROOT / "data_perp/artifacts/"
    "marapr2025_direct_residual_regime_trust_diagnostic_20260730_v1"
)

SCHEMA = "marapr2025_direct_residual_regime_trust_diagnostic_v1"
IDENTITY = ("candidate_id", "side_name", "__symbol__", "__ts__")
NET = "execution_net_ev_12h"
GROSS = "execution_gross_ev_12h"
COST = "execution_cost_return"
MFE = "execution_mfe_return_12h"
MAE = "execution_mae_return_12h"
SCORES = {
    "direct_q25": "score_raw_direct_q25_ev",
    "residual": "score_residual_expected_ev",
}
PERIODS = {
    "march03_19": (
        pd.Timestamp("2025-03-03T00:00:00Z"),
        pd.Timestamp("2025-03-20T00:00:00Z"),
    ),
    "march20_31": (
        pd.Timestamp("2025-03-20T00:00:00Z"),
        pd.Timestamp("2025-04-01T00:00:00Z"),
    ),
    "april": (
        pd.Timestamp("2025-04-01T00:00:00Z"),
        pd.Timestamp("2025-05-01T00:00:00Z"),
    ),
}
REFERENCE_START = pd.Timestamp("2024-09-03T00:00:00Z")
REFERENCE_END = pd.Timestamp("2025-03-03T00:00:00Z")

REGIME_FIELDS = (
    "bocpd__change_probability_mean",
    "bocpd__change_probability_max",
    "bocpd__run_length_mean",
    "bocpd__run_length_q05",
    "bocpd__run_length_entropy",
    "bocpd__signal_count",
    "bocpd__state_age_hours",
    "bocpd__is_persistent_24h",
    "bocpd__is_persistent_72h",
)
TRANSITION_FIELDS = (
    "lgbm_transition_probability",
    "lgbm_entropy",
    "lgbm_margin",
    "bocpd_stable_vs_transition_probability",
    "bocpd_onset_h1_probability",
    "bocpd_onset_h3_probability",
    "bocpd_onset_h6_probability",
    "bocpd_onset_h12_probability",
)
TRAJECTORY_SOURCE_FIELDS = (
    "trajectory_available",
    "trajectory_transition_probability",
    "probability_entropy",
    "top2_margin",
)
TRAJECTORY_RENAME = {
    "probability_entropy": "trajectory_probability_entropy",
    "top2_margin": "trajectory_top2_margin",
}
REGIME_PROVENANCE_FIELDS = (
    "provenance_partition_bocpd",
    "train_end_exclusive_utc_bocpd",
    "fit_label_resolution_max_utc_bocpd",
)
TRANSITION_PROVENANCE_FIELDS = (
    "provenance_partition_lgbm",
    "train_end_exclusive_utc_lgbm",
    "fit_label_resolution_max_utc_lgbm",
)
TRAJECTORY_PROVENANCE_FIELDS = (
    "oof_held_era",
    "provenance_partition",
    "fit_train_eras",
)
CONTEXT_FIELDS = (
    *REGIME_FIELDS,
    *TRANSITION_FIELDS,
    "trajectory_available",
    "trajectory_transition_probability",
    "trajectory_probability_entropy",
    "trajectory_top2_margin",
)


class DiagnosticError(RuntimeError):
    """Raised when a sealed-input or causal-alignment invariant fails."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(item) for item in value]
    if isinstance(value, (Path, pd.Timestamp)):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(safe(payload), indent=2, sort_keys=True) + "\n")


def identity_hash(frame: pd.DataFrame) -> str:
    values = frame.loc[:, list(IDENTITY)].copy()
    values["__ts__"] = pd.to_datetime(values["__ts__"], utc=True).astype(str)
    values = values.astype(str).sort_values(list(IDENTITY), kind="stable")
    return hashlib.sha256(values.to_csv(index=False).encode()).hexdigest()


def verify_artifact(root: Path, schema: str, output: str) -> tuple[dict[str, Any], Path]:
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise DiagnosticError(f"missing manifest: {manifest_path}")
    seal_path = root / "manifest.sha256"
    if seal_path.is_file() and sha256(manifest_path) != seal_path.read_text().split()[0]:
        raise DiagnosticError(f"manifest seal mismatch: {root}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != schema:
        raise DiagnosticError(f"schema mismatch at {root}: {manifest.get('schema')}")
    path = root / output
    expected = manifest.get("outputs_sha256", {}).get(output)
    if expected is None:
        expected = manifest.get("outputs", {}).get(output.removesuffix(".parquet"), {}).get("sha256")
    if not path.is_file() or expected != sha256(path):
        raise DiagnosticError(f"sealed output mismatch: {path}")
    return manifest, path


def assign_period(timestamp: pd.Series) -> pd.Series:
    values = pd.to_datetime(timestamp, utc=True, errors="raise")
    result = pd.Series(pd.NA, index=values.index, dtype="string")
    for name, (start, end) in PERIODS.items():
        result.loc[values.ge(start) & values.lt(end)] = name
    if result.isna().any():
        raise DiagnosticError("candidate timestamp is outside the frozen periods")
    return result


def add_candidate_coordinates(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for scope, groups in (
        ("timestamp_side", ["__ts__", "side_name"]),
        ("timestamp_global", ["__ts__"]),
    ):
        grouped = result.groupby(groups, sort=False, observed=True)
        result[f"candidate_group_rows_{scope}"] = grouped["candidate_id"].transform("size")
        for source, score in SCORES.items():
            rank = grouped[score].rank(method="first", ascending=False)
            count = result[f"candidate_group_rows_{scope}"].astype(float)
            result[f"{source}_rank_pct_{scope}"] = np.where(
                count.gt(1), (rank - 1.0) / (count - 1.0), 0.0
            )
            mean = grouped[score].transform("mean")
            std = grouped[score].transform("std").replace(0.0, np.nan)
            result[f"{source}_score_z_{scope}"] = (
                (result[score] - mean) / std
            ).fillna(0.0)
        result[f"direct_minus_residual_rank_{scope}"] = (
            result[f"residual_rank_pct_{scope}"]
            - result[f"direct_q25_rank_pct_{scope}"]
        )
    return result


def global_top(frame: pd.DataFrame, score: str, fraction: float = 0.10) -> pd.DataFrame:
    count = max(1, int(math.ceil(len(frame) * fraction)))
    return frame.sort_values(
        [score, "candidate_id", "side_name", "__symbol__", "__ts__"],
        ascending=[False, True, True, True, True],
        kind="mergesort",
    ).head(count).copy()


def rank_ic(left: pd.Series, right: pd.Series) -> float:
    pair = pd.DataFrame(
        {"left": pd.to_numeric(left), "right": pd.to_numeric(right)}
    ).dropna()
    if len(pair) < 3 or pair.left.nunique() < 2 or pair.right.nunique() < 2:
        return float("nan")
    return float(pair.left.corr(pair.right, method="spearman"))


def calendar_metrics(selected: pd.DataFrame) -> dict[str, float]:
    days = pd.to_datetime(selected["__ts__"], utc=True).dt.floor("D")
    shares = days.value_counts(normalize=True)
    return {
        "selected_days": float(len(shares)),
        "effective_selected_days": float(1.0 / np.square(shares).sum()),
        "max_day_share": float(shares.max()),
        "top3_day_share": float(shares.nlargest(3).sum()),
        "top5_day_share": float(shares.nlargest(5).sum()),
    }


def metric_record(
    population: pd.DataFrame,
    selected: pd.DataFrame,
    *,
    period: str,
    source: str,
) -> dict[str, Any]:
    return {
        "period": period,
        "source": source,
        "candidate_rows": len(population),
        "selected_rows": len(selected),
        "rank_ic_net": rank_ic(population[SCORES[source]], population[NET]),
        "rank_ic_gross": rank_ic(population[SCORES[source]], population[GROSS]),
        "net_bps": float(selected[NET].mean() * 1e4),
        "gross_bps": float(selected[GROSS].mean() * 1e4),
        "cost_bps": float(selected[COST].mean() * 1e4),
        "mfe_bps": float(selected[MFE].mean() * 1e4),
        "mae_bps": float(selected[MAE].mean() * 1e4),
        "positive_net_rate": float(selected[NET].gt(0).mean()),
        **calendar_metrics(selected),
    }


def side_records(selected: pd.DataFrame, *, period: str, source: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    total = len(selected)
    for side, rows in selected.groupby("side_name", sort=True, observed=True):
        records.append(
            {
                "period": period,
                "source": source,
                "side_name": side,
                "selected_rows": len(rows),
                "selected_share": float(len(rows) / total),
                "net_bps": float(rows[NET].mean() * 1e4),
                "gross_bps": float(rows[GROSS].mean() * 1e4),
                "cost_bps": float(rows[COST].mean() * 1e4),
                "portfolio_contribution_bps": float(rows[NET].sum() / total * 1e4),
                "positive_net_rate": float(rows[NET].gt(0).mean()),
            }
        )
    return records


def overlap_records(
    population: pd.DataFrame,
    direct: pd.DataFrame,
    residual: pd.DataFrame,
    *,
    period: str,
) -> list[dict[str, Any]]:
    keys = list(IDENTITY)
    direct_ids = set(map(tuple, direct[keys].itertuples(index=False, name=None)))
    residual_ids = set(map(tuple, residual[keys].itertuples(index=False, name=None)))
    labels = []
    for key in map(tuple, population[keys].itertuples(index=False, name=None)):
        in_direct = key in direct_ids
        in_residual = key in residual_ids
        labels.append(
            "shared" if in_direct and in_residual
            else "direct_only" if in_direct
            else "residual_only" if in_residual
            else "neither"
        )
    local = population.copy()
    local["membership"] = labels
    count = len(direct)
    records: list[dict[str, Any]] = []
    for membership in ("shared", "direct_only", "residual_only"):
        rows = local.loc[local["membership"].eq(membership)]
        sign = -1.0 if membership == "residual_only" else 1.0
        contribution = 0.0 if membership == "shared" else sign * rows[NET].sum() / count
        records.append(
            {
                "period": period,
                "membership": membership,
                "rows": len(rows),
                "mean_net_bps": float(rows[NET].mean() * 1e4) if len(rows) else np.nan,
                "direct_minus_residual_contribution_bps": float(contribution * 1e4),
                "direct_selected_rows": len(direct),
                "residual_selected_rows": len(residual),
                "selection_jaccard": float(
                    len(direct_ids & residual_ids) / len(direct_ids | residual_ids)
                ),
            }
        )
    delta = float(direct[NET].mean() - residual[NET].mean())
    reconciled = sum(r["direct_minus_residual_contribution_bps"] for r in records)
    if not np.isclose(reconciled, delta * 1e4, atol=1e-9):
        raise DiagnosticError("overlap attribution does not reconcile")
    return records


def quantile_edges(reference: pd.Series) -> np.ndarray:
    values = pd.to_numeric(reference, errors="raise").to_numpy(dtype=float)
    if not np.isfinite(values).all() or len(values) < 100:
        raise DiagnosticError("context reference is insufficient or non-finite")
    edges = np.quantile(values, [0.2, 0.4, 0.6, 0.8])
    return np.asarray(edges, dtype=float)


def context_bin_records(
    candidates: pd.DataFrame,
    books: pd.DataFrame,
    reference: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    bins: list[dict[str, Any]] = []
    shifts: list[dict[str, Any]] = []
    for feature in CONTEXT_FIELDS:
        edges = quantile_edges(reference[feature])
        q25, median, q75 = np.quantile(
            pd.to_numeric(reference[feature]).to_numpy(float), [0.25, 0.5, 0.75]
        )
        scale = max(float(q75 - q25), 1e-12)
        for period, rows in candidates.groupby("diagnostic_period", sort=False):
            values = pd.to_numeric(rows[feature]).to_numpy(float)
            shifts.append(
                {
                    "period": period,
                    "feature": feature,
                    "rows": len(rows),
                    "reference_median": float(median),
                    "reference_iqr": float(scale),
                    "period_median": float(np.median(values)),
                    "median_shift_reference_iqr": float(
                        (np.median(values) - median) / scale
                    ),
                    "period_p10": float(np.quantile(values, 0.10)),
                    "period_p90": float(np.quantile(values, 0.90)),
                }
            )
        for (period, source), rows in books.groupby(
            ["diagnostic_period", "selection_source"], sort=False
        ):
            local = rows.copy()
            local["context_bin"] = np.searchsorted(
                edges,
                pd.to_numeric(local[feature]).to_numpy(float),
                side="right",
            )
            for (side, context_bin), cell in local.groupby(
                ["side_name", "context_bin"], sort=True, observed=True
            ):
                bins.append(
                    {
                        "period": period,
                        "source": source,
                        "side_name": side,
                        "feature": feature,
                        "context_bin": int(context_bin),
                        "rows": len(cell),
                        "selected_share": float(len(cell) / len(rows)),
                        "net_bps": float(cell[NET].mean() * 1e4),
                        "gross_bps": float(cell[GROSS].mean() * 1e4),
                        "cost_bps": float(cell[COST].mean() * 1e4),
                        "positive_net_rate": float(cell[NET].gt(0).mean()),
                        "reference_q20": float(edges[0]),
                        "reference_q40": float(edges[1]),
                        "reference_q60": float(edges[2]),
                        "reference_q80": float(edges[3]),
                    }
                )
    return pd.DataFrame(bins), pd.DataFrame(shifts)


def _load_hourly(
    path: Path,
    fields: Iterable[str],
    *,
    rename: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    result = pd.read_parquet(path, columns=["source_utc", *fields])
    result["source_utc"] = pd.to_datetime(result["source_utc"], utc=True, errors="raise")
    if result["source_utc"].duplicated().any():
        raise DiagnosticError(f"duplicated hourly identity: {path}")
    return result.rename(columns=dict(rename or {}))


def run(
    bridge_root: Path = BRIDGE_ROOT,
    soft_root: Path = SOFT_ROOT,
    trajectory_root: Path = TRAJECTORY_ROOT,
    output: Path = OUTPUT,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite sealed output: {output}")

    bridge_manifest, bridge_path = verify_artifact(
        bridge_root,
        "marapr2025_identical_causal_score_bridge_v1",
        "identical_score_bridge.parquet",
    )
    soft_manifest, regime_path = verify_artifact(
        soft_root,
        "authoritative_soft_regime_transition_sidecars_v1",
        "soft_regime_hourly.parquet",
    )
    _, transition_path = verify_artifact(
        soft_root,
        "authoritative_soft_regime_transition_sidecars_v1",
        "soft_transition_hourly.parquet",
    )
    trajectory_manifest, trajectory_path = verify_artifact(
        trajectory_root,
        "hourly_trajectory_transition_soft_sidecar_v1",
        "hourly_trajectory_transition_soft_sidecar.parquet",
    )

    candidates = pd.read_parquet(bridge_path)
    candidates["__ts__"] = pd.to_datetime(candidates["__ts__"], utc=True, errors="raise")
    if candidates.duplicated(list(IDENTITY)).any() or len(candidates) != 136_074:
        raise DiagnosticError("bridge identity contract failed")
    candidates["diagnostic_period"] = assign_period(candidates["__ts__"])

    regime = _load_hourly(
        regime_path,
        (*REGIME_FIELDS, *REGIME_PROVENANCE_FIELDS),
        rename={
            "provenance_partition_bocpd": "regime_provenance",
            "train_end_exclusive_utc_bocpd": "regime_train_end_exclusive_utc",
            "fit_label_resolution_max_utc_bocpd": "regime_fit_label_resolution_max_utc",
        },
    )
    transition = _load_hourly(
        transition_path,
        (*TRANSITION_FIELDS, *TRANSITION_PROVENANCE_FIELDS),
        rename={
            "provenance_partition_lgbm": "transition_provenance",
            "train_end_exclusive_utc_lgbm": "transition_train_end_exclusive_utc",
            "fit_label_resolution_max_utc_lgbm": "transition_fit_label_resolution_max_utc",
        },
    )
    trajectory = _load_hourly(
        trajectory_path,
        (*TRAJECTORY_SOURCE_FIELDS, *TRAJECTORY_PROVENANCE_FIELDS),
        rename={
            **TRAJECTORY_RENAME,
            "oof_held_era": "trajectory_oof_held_era",
            "provenance_partition": "trajectory_provenance",
            "fit_train_eras": "trajectory_fit_train_eras",
        },
    )
    hourly = regime.merge(
        transition, on="source_utc", how="inner", validate="one_to_one"
    ).merge(trajectory, on="source_utc", how="inner", validate="one_to_one")
    reference = hourly.loc[
        hourly["source_utc"].ge(REFERENCE_START)
        & hourly["source_utc"].lt(REFERENCE_END)
    ].copy()
    candidates = candidates.merge(
        hourly,
        left_on="__ts__",
        right_on="source_utc",
        how="left",
        validate="many_to_one",
    )
    if candidates["source_utc"].isna().any():
        raise DiagnosticError("hourly context coverage is incomplete")
    candidates = candidates.drop(columns=["source_utc"])
    if not np.isfinite(candidates[list(CONTEXT_FIELDS)].to_numpy(float)).all():
        raise DiagnosticError("causal context contains non-finite values")
    if not candidates["trajectory_available"].astype(bool).all():
        raise DiagnosticError("trajectory context is unavailable on a candidate row")
    expected_text = {
        "regime_provenance": "blocked_oof_2022_2025",
        "transition_provenance": "blocked_oof_2022_2025",
        "trajectory_provenance": "blocked_era_oof",
        "trajectory_fit_train_eras": "2022,2023,2024",
    }
    for field, expected in expected_text.items():
        if set(candidates[field].astype(str).unique()) != {expected}:
            raise DiagnosticError(f"unexpected causal provenance in {field}")
    if set(pd.to_numeric(candidates["trajectory_oof_held_era"]).unique()) != {2025}:
        raise DiagnosticError("trajectory rows do not all hold out calendar era 2025")
    for prefix in ("regime", "transition"):
        train_end = pd.to_datetime(
            candidates[f"{prefix}_train_end_exclusive_utc"], utc=True, errors="raise"
        )
        label_end = pd.to_datetime(
            candidates[f"{prefix}_fit_label_resolution_max_utc"], utc=True, errors="raise"
        )
        if not train_end.lt(candidates["__ts__"]).all():
            raise DiagnosticError(f"{prefix} fit reaches a candidate timestamp")
        if not label_end.lt(candidates["__ts__"]).all():
            raise DiagnosticError(f"{prefix} fit label reaches a candidate timestamp")
    candidates = add_candidate_coordinates(candidates)

    metric_rows: list[dict[str, Any]] = []
    side_rows: list[dict[str, Any]] = []
    overlap_rows: list[dict[str, Any]] = []
    books: list[pd.DataFrame] = []
    for period in PERIODS:
        population = candidates.loc[candidates["diagnostic_period"].eq(period)].copy()
        selected: dict[str, pd.DataFrame] = {}
        for source, score in SCORES.items():
            book = global_top(population, score)
            book["selection_source"] = source
            books.append(book)
            selected[source] = book
            metric_rows.append(
                metric_record(population, book, period=period, source=source)
            )
            side_rows.extend(side_records(book, period=period, source=source))
        overlap_rows.extend(
            overlap_records(
                population,
                selected["direct_q25"],
                selected["residual"],
                period=period,
            )
        )
    selected_books = pd.concat(books, ignore_index=True)
    context_bins, context_shift = context_bin_records(
        candidates, selected_books, reference
    )

    provenance = pd.DataFrame(
        [
            {
                "source": "bridge",
                "rows": len(candidates),
                "minimum_source_utc": candidates["__ts__"].min(),
                "maximum_source_utc": candidates["__ts__"].max(),
                "fit_train_end_exclusive_utc": pd.NaT,
                "fit_label_resolution_max_utc": pd.NaT,
                "causal_before_march": True,
            },
            {
                "source": "authoritative_soft_regime_transition",
                "rows": len(hourly),
                "minimum_source_utc": hourly["source_utc"].min(),
                "maximum_source_utc": hourly["source_utc"].max(),
                "fit_train_end_exclusive_utc": pd.to_datetime(
                    candidates["regime_train_end_exclusive_utc"], utc=True
                ).max(),
                "fit_label_resolution_max_utc": pd.to_datetime(
                    candidates["regime_fit_label_resolution_max_utc"], utc=True
                ).max(),
                "causal_before_march": True,
            },
            {
                "source": "trajectory_soft_transition",
                "rows": len(hourly),
                "minimum_source_utc": hourly["source_utc"].min(),
                "maximum_source_utc": hourly["source_utc"].max(),
                "fit_train_end_exclusive_utc": pd.NaT,
                "fit_label_resolution_max_utc": pd.NaT,
                "causal_before_march": True,
            },
        ]
    )

    outputs = {
        "candidate_panel.parquet": candidates,
        "selected_books.parquet": selected_books,
        "period_metrics.parquet": pd.DataFrame(metric_rows),
        "side_metrics.parquet": pd.DataFrame(side_rows),
        "selection_overlap_attribution.parquet": pd.DataFrame(overlap_rows),
        "context_bin_metrics.parquet": context_bins,
        "context_shift.parquet": context_shift,
        "provenance_audit.parquet": provenance,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        output_hashes: dict[str, str] = {}
        output_rows: dict[str, int] = {}
        for name, frame in outputs.items():
            path = stage / name
            frame.to_parquet(path, index=False, compression="zstd")
            output_hashes[name] = sha256(path)
            output_rows[name] = len(frame)
        manifest = {
            "schema": SCHEMA,
            "status": "SEALED_REUSED_MONTH_CAUSAL_REGIME_TRUST_DIAGNOSTIC_NO_GATE",
            "promotion_eligible": False,
            "portfolio_replay_authorized": False,
            "contract": {
                "identity": list(IDENTITY),
                "evaluation_rows": len(candidates),
                "selection": "one pooled-global top10 per declared period and source; deterministic candidate identity tie-break; never per timestamp/side/asset",
                "periods": {key: [str(value[0]), str(value[1])] for key, value in PERIODS.items()},
                "context_alignment": "candidate __ts__ equals hourly source_utc; never execution_decision_utc",
                "context_reference": [str(REFERENCE_START), str(REFERENCE_END)],
                "context": "9 authoritative BOCPD regime + 8 authoritative transition + 4 trajectory soft/availability fields; OOD/state/destination/post-entry/action fields excluded",
                "candidate_coordinates": "complete bridge timestamp-side and timestamp-global ranks/z/group size, diagnostic only",
                "reuse": "March/April were used in prior direct-head research; this artifact is diagnosis only and cannot select or promote a trust gate",
                "actions": "timing, MAE, wait and target-price layers excluded",
            },
            "feature_columns": list(CONTEXT_FIELDS),
            "score_columns": SCORES,
            "outputs_sha256": output_hashes,
            "output_rows": output_rows,
            "sources": {
                "bridge_manifest_sha256": sha256(bridge_root / "manifest.json"),
                "bridge_sha256": sha256(bridge_path),
                "bridge_identity_sha256": identity_hash(candidates),
                "soft_manifest_sha256": sha256(soft_root / "manifest.json"),
                "regime_sha256": sha256(regime_path),
                "transition_sha256": sha256(transition_path),
                "soft_status": soft_manifest["status"],
                "trajectory_manifest_sha256": sha256(trajectory_root / "manifest.json"),
                "trajectory_sha256": sha256(trajectory_path),
                "trajectory_status": trajectory_manifest["status"],
            },
            "runner": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256(Path(__file__).resolve()),
            },
        }
        write_json(stage / "manifest.json", manifest)
        (stage / "manifest.sha256").write_text(
            f"{sha256(stage / 'manifest.json')}  manifest.json\n"
        )
        os.replace(stage, output)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return manifest


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--bridge-root", type=Path, default=BRIDGE_ROOT)
    result.add_argument("--soft-root", type=Path, default=SOFT_ROOT)
    result.add_argument("--trajectory-root", type=Path, default=TRAJECTORY_ROOT)
    result.add_argument("--output", type=Path, default=OUTPUT)
    return result


def main() -> None:
    args = parser().parse_args()
    print(
        json.dumps(
            safe(
                run(
                    bridge_root=args.bridge_root,
                    soft_root=args.soft_root,
                    trajectory_root=args.trajectory_root,
                    output=args.output,
                )
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
