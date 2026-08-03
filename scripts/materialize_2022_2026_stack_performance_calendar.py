#!/usr/bin/env python3
"""Materialize a lineage-aware 2022--2026 base+residual performance calendar.

Only strict residual-OOF rows with exact 12-hour deployed-policy economics are
used to estimate stack IC and EV.  Other historical lineages are registered as
diagnostic coverage or explicit gaps; they are never silently pooled into the
strict performance denominator.
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
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "stack_performance_calendar_2022_2026_v1"
START = pd.Timestamp("2022-01-01T00:00:00Z")
END_EXCLUSIVE = pd.Timestamp("2026-07-24T00:00:00Z")
SCORE = "score_residual_expected_ev"
ALPHA_TARGET = "__first_touch_target_soft__"
NET_TARGET = "execution_net_ev_12h"
GROSS = "execution_gross_ev_12h"
COST = "execution_cost_return"
IDENTITY = ("candidate_id", "__ts__", "__symbol__", "side_name")
TOP_FRACTION = 0.10

DEFAULT_2025 = ROOT / (
    "data_perp/artifacts/"
    "marapr2025_all_score_ic_ev_waterfall_20260730_v1"
)
DEFAULT_2026 = ROOT / (
    "data_perp/artifacts/"
    "mayjul2026_exact_allscore_ic_ev_waterfall_20260730_v1"
)
DEFAULT_BACKFILL = ROOT / (
    "data_perp/artifacts/"
    "reconstructed_base_residual_stack_2022_2024q1_20260730_v2"
)
DEFAULT_OUTPUT = ROOT / (
    "data_perp/artifacts/"
    "stack_performance_calendar_2022_2026_20260730_v2"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _read_manifest(root: Path, expected_schema: str) -> dict[str, Any]:
    path = root / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != expected_schema:
        raise ValueError(f"{root} has unexpected schema {manifest.get('schema')!r}")
    sidecar = root / "manifest.sha256"
    if sidecar.exists():
        expected = sidecar.read_text(encoding="utf-8").strip().split()[0]
        if expected != sha256_file(path):
            raise ValueError(f"{root} manifest hash sidecar mismatch")
    return manifest


def _validate_rows(frame: pd.DataFrame, *, source: str) -> pd.DataFrame:
    required = {
        *IDENTITY,
        SCORE,
        ALPHA_TARGET,
        NET_TARGET,
        GROSS,
        COST,
        "execution_label_end_utc",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{source} lacks calendar fields: {missing}")
    work = frame.copy()
    work["__ts__"] = pd.to_datetime(work["__ts__"], utc=True, errors="raise")
    work["execution_label_end_utc"] = pd.to_datetime(
        work["execution_label_end_utc"], utc=True, errors="raise"
    )
    if work.duplicated(list(IDENTITY)).any() or work["candidate_id"].duplicated().any():
        raise ValueError(f"{source} candidate identities are not one-to-one")
    numeric = (SCORE, ALPHA_TARGET, NET_TARGET, GROSS, COST)
    for column in numeric:
        work[column] = pd.to_numeric(work[column], errors="coerce")
    if work[list(numeric)].isna().any().any():
        raise ValueError(f"{source} has non-finite score/target/economic rows")
    if not np.allclose(
        work[GROSS].to_numpy(float) - work[COST].to_numpy(float),
        work[NET_TARGET].to_numpy(float),
        atol=1e-10,
        rtol=0.0,
    ):
        raise ValueError(f"{source} violates gross - cost = net")
    if (work["execution_label_end_utc"] <= work["__ts__"]).any():
        raise ValueError(f"{source} label resolution is not after signal time")
    return work


def load_strict_stack_rows(
    root_2025: Path, root_2026: Path,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    manifest_2025 = _read_manifest(
        root_2025, "marapr2025_all_score_ic_ev_waterfall_v1"
    )
    manifest_2026 = _read_manifest(
        root_2026, "mayjul2026_exact_allscore_ic_ev_waterfall_v1"
    )
    specifications = (
        (
            root_2025 / "all_score_waterfall.parquet",
            "canonical_marapr2025_strict_residual_oof",
            "A_STRICT_OOF_EXACT_POLICY",
            manifest_2025,
        ),
        (
            root_2026 / "allscore_waterfall.parquet",
            "current_mayjul2026_strict_residual_oof",
            "A_STRICT_OOF_EXACT_POLICY",
            manifest_2026,
        ),
    )
    parts: list[pd.DataFrame] = []
    registry: list[dict[str, Any]] = []
    for path, lineage, grade, manifest in specifications:
        part = _validate_rows(pd.read_parquet(path), source=lineage)
        part["lineage_id"] = lineage
        part["evidence_grade"] = grade
        part["strict_residual_oof"] = True
        parts.append(part)
        registry.append(
            {
                "lineage_id": lineage,
                "evidence_grade": grade,
                "first_signal_utc": part["__ts__"].min(),
                "last_signal_utc": part["__ts__"].max(),
                "rows": len(part),
                "candidate_population": (
                    "canonical_febapr2025_top40"
                    if "2025" in lineage
                    else "current_packb31_8_top40"
                ),
                "product_economics": "USD_linear_exact_1m_policy_12h",
                "score": SCORE,
                "alpha_target": ALPHA_TARGET,
                "economic_target": NET_TARGET,
                "strict_oof": True,
                "selection": "one_pooled_global_cross_side_cross_timestamp_top10_per_period",
                "source_path": str(path.resolve()),
                "source_sha256": sha256_file(path),
                "source_manifest_sha256": sha256_file(path.parent / "manifest.json"),
                "source_status": manifest.get("status"),
                "cross_lineage_net_pooling_allowed": False,
            }
        )
    combined = pd.concat(parts, ignore_index=True)
    if combined.duplicated(list(IDENTITY)).any():
        raise ValueError("strict stack source lineages overlap by exact identity")
    return combined, registry


def load_reconstructed_stack_rows(
    root: Path,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    manifest = _read_manifest(
        root, "historical_base_residual_stack_calendar_block_oof_v1"
    )
    if manifest.get("status") != "RESEARCH_OOF_BACKFILL_COMPLETE":
        raise ValueError(f"{root} is not an accepted completed backfill")
    if (root / "RESEARCH_INVALIDATION.json").exists():
        raise ValueError(f"{root} has been explicitly invalidated")
    path = root / "oof_scores.parquet"
    frame = pd.read_parquet(path)
    frame[ALPHA_TARGET] = pd.to_numeric(
        frame.pop("__reconstructed_soft_alpha_12h__"), errors="raise"
    )
    frame["execution_label_end_utc"] = (
        pd.to_datetime(frame["__ts__"], utc=True, errors="raise")
        + pd.Timedelta(hours=13)
    )
    frame = _validate_rows(frame, source="reconstructed_2022_2024q1")
    frame["lineage_id"] = frame.pop("stack_lineage")
    frame["evidence_grade"] = np.where(
        frame["lineage_id"].eq("inverse_pi_2022_h1"),
        "C_RESEARCH_OOF_SEPARATE_POPULATION",
        "B_RESEARCH_RESIDUAL_OOF_BASE_BACKCAST",
    )
    frame["strict_residual_oof"] = True
    registry: list[dict[str, Any]] = []
    for (lineage, grade), local in frame.groupby(
        ["lineage_id", "evidence_grade"], sort=True
    ):
        registry.append(
            {
                "lineage_id": lineage,
                "evidence_grade": grade,
                "first_signal_utc": local["__ts__"].min(),
                "last_signal_utc": local["__ts__"].max(),
                "rows": len(local),
                "candidate_population": (
                    "inverse_pi_hourly_grid"
                    if lineage == "inverse_pi_2022_h1"
                    else "frozen_pf_base_monitor_population"
                ),
                "product_economics": (
                    "inverse_quote_notional_current_spread_counterfactual"
                    if lineage == "inverse_pi_2022_h1"
                    else "exact_1m_current_spread_counterfactual"
                ),
                "score": SCORE,
                "alpha_target": ALPHA_TARGET,
                "economic_target": NET_TARGET,
                "strict_oof": False,
                "residual_oof": True,
                "selection": "one_pooled_global_cross_side_cross_timestamp_top10_per_period",
                "source_path": str(path.resolve()),
                "source_sha256": sha256_file(path),
                "source_manifest_sha256": sha256_file(root / "manifest.json"),
                "source_status": manifest.get("status"),
                "cross_lineage_net_pooling_allowed": False,
                "limitation": manifest["lineage_limitations"].get(lineage),
            }
        )
    return frame, registry


def stable_global_top_mask(
    frame: pd.DataFrame, score: Sequence[float], fraction: float = TOP_FRACTION
) -> np.ndarray:
    if not 0.0 < float(fraction) <= 1.0:
        raise ValueError("fraction must be in (0,1]")
    values = np.asarray(score, dtype=float)
    if len(values) != len(frame) or not np.isfinite(values).all():
        raise ValueError("score must be finite and aligned")
    order = pd.DataFrame(
        {
            "position": np.arange(len(frame)),
            "candidate_id": frame["candidate_id"].astype(str),
            "score": values,
        }
    ).sort_values(
        ["score", "candidate_id"],
        ascending=[False, True],
        kind="mergesort",
    )
    count = max(1, int(math.ceil(len(frame) * float(fraction))))
    mask = np.zeros(len(frame), dtype=bool)
    mask[order["position"].to_numpy()[:count]] = True
    return mask


def _rank_ic(left: Sequence[float], right: Sequence[float]) -> float:
    x = pd.Series(np.asarray(left, dtype=float)).rank(method="average")
    y = pd.Series(np.asarray(right, dtype=float)).rank(method="average")
    if x.nunique() < 2 or y.nunique() < 2:
        return float("nan")
    return float(x.corr(y))


def _period_key(timestamp: pd.Series, period_type: str) -> pd.Series:
    if period_type == "week":
        naive = timestamp.dt.tz_convert("UTC").dt.tz_localize(None)
        return naive.dt.to_period("W-SUN").dt.start_time.dt.tz_localize("UTC")
    if period_type == "month":
        return timestamp.dt.strftime("%Y-%m")
    raise ValueError("period_type must be week or month")


def _block_bootstrap(
    frame: pd.DataFrame,
    *,
    selected: np.ndarray,
    draws: int,
    seed: int,
) -> dict[str, float]:
    work = frame.reset_index(drop=True)
    day = work["__ts__"].dt.floor("D")
    days = pd.Index(day.unique())
    if len(days) < 2:
        return {
            "ic_alpha_lo80": float("nan"),
            "ic_alpha_lo95": float("nan"),
            "net_bps_lo80": float("nan"),
            "net_bps_lo95": float("nan"),
        }
    positions = {
        value: np.flatnonzero(day.eq(value).to_numpy()) for value in days
    }
    rng = np.random.default_rng(seed)
    ic_values: list[float] = []
    net_values: list[float] = []
    for _ in range(int(draws)):
        sampled = rng.choice(days.to_numpy(), size=len(days), replace=True)
        index = np.concatenate([positions[value] for value in sampled])
        ic_values.append(
            _rank_ic(
                work[SCORE].to_numpy(float)[index],
                work[ALPHA_TARGET].to_numpy(float)[index],
            )
        )
        chosen = index[selected[index]]
        if len(chosen):
            net_values.append(
                float(work[NET_TARGET].to_numpy(float)[chosen].mean() * 10_000.0)
            )
    ic = np.asarray(ic_values, dtype=float)
    net = np.asarray(net_values, dtype=float)
    return {
        "ic_alpha_lo80": float(np.nanquantile(ic, 0.10)),
        "ic_alpha_lo95": float(np.nanquantile(ic, 0.025)),
        "net_bps_lo80": float(np.nanquantile(net, 0.10)),
        "net_bps_lo95": float(np.nanquantile(net, 0.025)),
    }


def _period_metrics(
    frame: pd.DataFrame,
    *,
    period_type: str,
    period: Any,
    bootstrap_draws: int,
    seed: int,
) -> dict[str, Any]:
    local = frame.sort_values(
        [SCORE, "candidate_id"], ascending=[False, True], kind="mergesort"
    ).reset_index(drop=True)
    selected = stable_global_top_mask(local, local[SCORE])
    tail = local.loc[selected]
    observed_days = int(local["__ts__"].dt.floor("D").nunique())
    if period_type == "week":
        period_start = pd.Timestamp(period)
        period_end = period_start + pd.Timedelta(days=7)
        expected_days = int(
            max(
                0,
                (
                    min(period_end, END_EXCLUSIVE)
                    - max(period_start, START)
                ).days,
            )
        )
        complete_for_percentage = observed_days >= min(7, expected_days) and expected_days >= 4
    else:
        period_start = pd.Timestamp(f"{period}-01", tz="UTC")
        period_end = period_start + pd.offsets.MonthBegin(1)
        expected_days = int(
            (min(period_end, END_EXCLUSIVE) - max(period_start, START)).days
        )
        complete_for_percentage = observed_days >= expected_days and expected_days >= 14
    bootstrap = _block_bootstrap(
        local,
        selected=selected,
        draws=bootstrap_draws,
        seed=seed,
    )
    alpha_ic = _rank_ic(local[SCORE], local[ALPHA_TARGET])
    execution_ic = _rank_ic(local[SCORE], local[NET_TARGET])
    tail_ic = _rank_ic(tail[SCORE], tail[NET_TARGET])
    mean_net_bps = float(tail[NET_TARGET].mean() * 10_000.0)
    meaningful_ic = bool(
        np.isfinite(alpha_ic)
        and alpha_ic >= 0.05
        and bootstrap["ic_alpha_lo80"] > 0.0
    )
    meaningful_ev = bool(
        mean_net_bps >= 5.0 and bootstrap["net_bps_lo80"] > 0.0
    )
    return {
        "period_type": period_type,
        "period": str(period),
        "period_start_utc": period_start,
        "period_end_exclusive_utc": period_end,
        "candidate_rows": len(local),
        "selected_rows": int(selected.sum()),
        "observed_days": observed_days,
        "expected_days": expected_days,
        "coverage_fraction": (
            float(observed_days / expected_days) if expected_days else float("nan")
        ),
        "complete_for_percentage": bool(complete_for_percentage),
        "lineage_id": ",".join(sorted(local["lineage_id"].unique())),
        "evidence_grade": ",".join(sorted(local["evidence_grade"].unique())),
        "score": SCORE,
        "alpha_target": ALPHA_TARGET,
        "alpha_rank_ic": alpha_ic,
        "execution_net_rank_ic": execution_ic,
        "tail_execution_net_rank_ic": tail_ic,
        "mean_gross_bps": float(tail[GROSS].mean() * 10_000.0),
        "mean_cost_bps": float(tail[COST].mean() * 10_000.0),
        "mean_net_bps": mean_net_bps,
        "sum_net_return": float(tail[NET_TARGET].sum()),
        "positive_net_rate": float(tail[NET_TARGET].gt(0.0).mean()),
        "cvar05_net_bps": float(
            tail[NET_TARGET].nsmallest(max(1, int(math.ceil(len(tail) * 0.05)))).mean()
            * 10_000.0
        ),
        "long_share": float(tail["side_name"].eq("long").mean()),
        "distinct_assets": int(tail["__symbol__"].nunique()),
        "distinct_timestamps": int(tail["__ts__"].nunique()),
        **bootstrap,
        "meaningful_ic_rule": "alpha_rank_ic>=0.05 and UTC-day bootstrap p10>0",
        "meaningful_ev_rule": "top10_mean_net>=5bps and UTC-day bootstrap p10>0",
        "meaningfully_positive_ic": meaningful_ic,
        "meaningfully_positive_ev": meaningful_ev,
        "meaningfully_positive_ic_and_ev": bool(meaningful_ic and meaningful_ev),
        "point_positive_ic": bool(alpha_ic > 0.0),
        "point_positive_ev": bool(mean_net_bps > 0.0),
        "point_positive_ic_and_ev": bool(alpha_ic > 0.0 and mean_net_bps > 0.0),
        "selection_scope": "one_pooled_global_cross_side_cross_timestamp_top10_within_period",
    }


def materialize_period_metrics(
    rows: pd.DataFrame, *, bootstrap_draws: int, seed: int
) -> pd.DataFrame:
    work = rows.copy()
    outputs: list[dict[str, Any]] = []
    for period_type in ("week", "month"):
        work["__period__"] = _period_key(work["__ts__"], period_type)
        for position, (period, local) in enumerate(
            work.groupby("__period__", sort=True)
        ):
            outputs.append(
                _period_metrics(
                    local.drop(columns="__period__"),
                    period_type=period_type,
                    period=period,
                    bootstrap_draws=bootstrap_draws,
                    seed=seed + position + (10_000 if period_type == "month" else 0),
                )
            )
    return pd.DataFrame(outputs).sort_values(
        ["period_type", "period_start_utc"], kind="mergesort"
    )


def positive_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for period_type, group in metrics.groupby("period_type", sort=True):
        eligible = group.loc[group["complete_for_percentage"]].copy()
        denominator = len(eligible)
        rows.append(
            {
                "period_type": period_type,
                "eligible_complete_periods": denominator,
                "observed_partial_or_complete_periods": len(group),
                "meaningfully_positive_ic_periods": int(
                    eligible["meaningfully_positive_ic"].sum()
                ),
                "meaningfully_positive_ic_pct": (
                    float(eligible["meaningfully_positive_ic"].mean() * 100.0)
                    if denominator
                    else float("nan")
                ),
                "meaningfully_positive_ev_periods": int(
                    eligible["meaningfully_positive_ev"].sum()
                ),
                "meaningfully_positive_ev_pct": (
                    float(eligible["meaningfully_positive_ev"].mean() * 100.0)
                    if denominator
                    else float("nan")
                ),
                "meaningfully_positive_both_periods": int(
                    eligible["meaningfully_positive_ic_and_ev"].sum()
                ),
                "meaningfully_positive_both_pct": (
                    float(
                        eligible["meaningfully_positive_ic_and_ev"].mean() * 100.0
                    )
                    if denominator
                    else float("nan")
                ),
                "point_positive_both_pct": (
                    float(eligible["point_positive_ic_and_ev"].mean() * 100.0)
                    if denominator
                    else float("nan")
                ),
                "denominator_contract": (
                    "complete held-block residual-OOF or strict residual-OOF "
                    "exact-policy periods; evidence grade retained and unsupported "
                    "calendar periods reported separately"
                ),
            }
        )
    return pd.DataFrame(rows)


def period_distribution_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    """Summarize the complete weekly and monthly reporting distributions.

    This deliberately operates on already-materialized *period* metrics: a
    weekly Q10 is a quantile across complete weekly pooled-global books, not a
    quantile across candidate rows.  The same distinction applies to monthly
    Q10/Q50 and to every positive-period share.
    """

    records: list[dict[str, Any]] = []
    for period_type, group in metrics.groupby("period_type", sort=True):
        complete = group.loc[group["complete_for_percentage"]].copy()
        net = pd.to_numeric(complete["mean_net_bps"], errors="coerce")
        alpha_ic = pd.to_numeric(complete["alpha_rank_ic"], errors="coerce")
        execution_ic = pd.to_numeric(
            complete["execution_net_rank_ic"], errors="coerce"
        )
        records.append(
            {
                "period_type": str(period_type),
                "complete_periods": int(len(complete)),
                "net_ev_bps_q10": float(net.quantile(0.10)) if len(net) else float("nan"),
                "net_ev_bps_q50": float(net.quantile(0.50)) if len(net) else float("nan"),
                "alpha_rank_ic_q10": float(alpha_ic.quantile(0.10)) if len(alpha_ic) else float("nan"),
                "alpha_rank_ic_q50": float(alpha_ic.quantile(0.50)) if len(alpha_ic) else float("nan"),
                "execution_net_rank_ic_q10": (
                    float(execution_ic.quantile(0.10)) if len(execution_ic) else float("nan")
                ),
                "execution_net_rank_ic_q50": (
                    float(execution_ic.quantile(0.50)) if len(execution_ic) else float("nan")
                ),
                "point_positive_ev_periods": int(complete["point_positive_ev"].sum()),
                "point_positive_ev_period_share": (
                    float(complete["point_positive_ev"].mean()) if len(complete) else float("nan")
                ),
                "meaningfully_positive_ev_periods": int(
                    complete["meaningfully_positive_ev"].sum()
                ),
                "meaningfully_positive_ev_period_share": (
                    float(complete["meaningfully_positive_ev"].mean())
                    if len(complete)
                    else float("nan")
                ),
                "point_positive_ic_periods": int(complete["point_positive_ic"].sum()),
                "point_positive_ic_period_share": (
                    float(complete["point_positive_ic"].mean()) if len(complete) else float("nan")
                ),
                "meaningfully_positive_ic_periods": int(
                    complete["meaningfully_positive_ic"].sum()
                ),
                "meaningfully_positive_ic_period_share": (
                    float(complete["meaningfully_positive_ic"].mean())
                    if len(complete)
                    else float("nan")
                ),
                "point_positive_ic_and_ev_periods": int(
                    complete["point_positive_ic_and_ev"].sum()
                ),
                "point_positive_ic_and_ev_period_share": (
                    float(complete["point_positive_ic_and_ev"].mean())
                    if len(complete)
                    else float("nan")
                ),
                "meaningfully_positive_ic_and_ev_periods": int(
                    complete["meaningfully_positive_ic_and_ev"].sum()
                ),
                "meaningfully_positive_ic_and_ev_period_share": (
                    float(complete["meaningfully_positive_ic_and_ev"].mean())
                    if len(complete)
                    else float("nan")
                ),
                "selection_scope": "one pooled-global top10 within each reporting period",
                "quantile_unit": "complete reporting period, not candidate row",
            }
        )
    return pd.DataFrame.from_records(records)


def evidence_registry(strict_registry: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    strict = []
    for item in strict_registry:
        record = dict(item)
        record["stack_available"] = True
        record["reason_not_stack"] = None
        strict.append(record)
    return pd.DataFrame(strict)


def coverage_gaps(metrics: pd.DataFrame) -> pd.DataFrame:
    calendar_months = pd.date_range(
        START.floor("D"),
        END_EXCLUSIVE - pd.Timedelta(seconds=1),
        freq="MS",
        tz="UTC",
    )
    observed_months = set(
        metrics.loc[metrics["period_type"].eq("month"), "period"].astype(str)
    )
    records = []
    for month in calendar_months:
        key = month.strftime("%Y-%m")
        if key in observed_months:
            local = metrics.loc[
                metrics["period_type"].eq("month") & metrics["period"].eq(key)
            ].iloc[0]
            is_strict = str(local["evidence_grade"]) == "A_STRICT_OOF_EXACT_POLICY"
            status = (
                ("strict_stack_complete" if is_strict else "research_stack_complete")
                if bool(local["complete_for_percentage"])
                else ("strict_stack_partial" if is_strict else "research_stack_partial")
            )
            reason = None if status.endswith("_complete") else "partial month"
        else:
            status = "stack_calendar_missing"
            reason = "no authoritative strict base+residual exact-policy panel found"
        records.append(
            {
                "month": key,
                "status": status,
                "reason": reason,
                "stack_percentage_denominator_eligible": status.endswith("_complete"),
            }
        )
    return pd.DataFrame(records)


def run(args: argparse.Namespace) -> Path:
    if args.output.exists():
        raise FileExistsError(f"immutable output exists: {args.output}")
    rows, strict_registry = load_strict_stack_rows(args.source_2025, args.source_2026)
    reconstructed, reconstructed_registry = load_reconstructed_stack_rows(
        args.source_backfill
    )
    rows = pd.concat([reconstructed, rows], ignore_index=True)
    if rows.duplicated(list(IDENTITY)).any():
        raise ValueError("backfilled and strict stack sources overlap by exact identity")
    metrics = materialize_period_metrics(
        rows, bootstrap_draws=args.bootstrap_draws, seed=args.seed
    )
    summary = positive_summary(metrics)
    distribution = period_distribution_summary(metrics)
    registry = evidence_registry([*reconstructed_registry, *strict_registry])
    gaps = coverage_gaps(metrics)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(dir=args.output.parent, prefix=f".{args.output.name}.")
    )
    try:
        outputs = {
            "performance_period_metrics.parquet": metrics,
            "meaningful_positive_summary.csv": summary,
            "period_distribution_summary.csv": distribution,
            "atlas_evidence_registry.csv": registry,
            "atlas_month_coverage_gaps.csv": gaps,
        }
        hashes: dict[str, str] = {}
        for name, frame in outputs.items():
            path = temporary / name
            if path.suffix == ".parquet":
                frame.to_parquet(path, index=False, compression="zstd")
            else:
                frame.to_csv(path, index=False)
            hashes[name] = sha256_file(path)
        manifest = {
            "schema": SCHEMA,
            "status": "MATERIALIZED_LINEAGE_AWARE_BACKFILLED_PARTIAL_COVERAGE",
            "calendar_start_utc": START,
            "calendar_end_exclusive_utc": END_EXCLUSIVE,
            "strict_stack_rows": len(rows),
            "strict_stack_first_signal_utc": rows["__ts__"].min(),
            "strict_stack_last_signal_utc": rows["__ts__"].max(),
            "strict_stack_months": sorted(rows["__ts__"].dt.strftime("%Y-%m").unique()),
            "score": SCORE,
            "alpha_target": ALPHA_TARGET,
            "economic_target": NET_TARGET,
            "top_fraction": TOP_FRACTION,
            "meaningful_positive_contract": {
                "ic": "alpha rank IC >= 0.05 and UTC-day block-bootstrap 10th percentile > 0",
                "ev": "pooled-global top10 mean exact policy net >= +5 bps and UTC-day block-bootstrap 10th percentile > 0",
                "both": "both IC and EV conditions",
                "sensitivity": "point-positive IC and EV are also reported",
                "bootstrap_draws": args.bootstrap_draws,
                "seed": args.seed,
            },
            "period_distribution_contract": {
                "q10_q50_unit": "complete weekly/monthly pooled-global reporting period",
                "positive_period_share": "fraction of complete reporting periods satisfying the named point or meaningful rule",
                "candidate_level_quantiles_forbidden": True,
            },
            "denominator": (
                "complete held-block residual-OOF or strict residual-OOF periods; "
                "partial and still-missing periods excluded and explicitly listed; "
                "evidence grade retained on every period"
            ),
            "promotion_eligible": False,
            "research_only": True,
            "outputs_sha256": hashes,
            "runner_sha256": sha256_file(Path(__file__).resolve()),
        }
        manifest_path = temporary / "manifest.json"
        manifest_path.write_text(
            json.dumps(json_safe(manifest), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (temporary / "manifest.sha256").write_text(
            f"{sha256_file(manifest_path)}  manifest.json\n", encoding="utf-8"
        )
        os.replace(temporary, args.output)
        return args.output
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-2025", type=Path, default=DEFAULT_2025)
    parser.add_argument("--source-2026", type=Path, default=DEFAULT_2026)
    parser.add_argument("--source-backfill", type=Path, default=DEFAULT_BACKFILL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-draws", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260730)
    args = parser.parse_args(argv)
    if args.bootstrap_draws < 100:
        parser.error("--bootstrap-draws must be at least 100")
    return args


if __name__ == "__main__":
    run(parse_args())
