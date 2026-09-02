#!/usr/bin/env python3
"""Strict-prequential Stage-3 consensus screen on the frozen :00 B0 stack.

This is deliberately a *screen*, not a promoted stack.  It changes only the
combination of the already frozen ten residual-head rank outputs.  Scores are
computed target-free; policy outcomes are joined only after all variant scores
and ranks are final.  The downstream correctness/CDF/MC1 rebuild belongs to
the follow-on Stage-4 producer and is required before any arm can advance.

The producer implements C0--C5 from the supplied research specification:

* C0: frozen unconditional median;
* C1: declared query x weighting x capacity grouped median;
* C2: strictly-prior reliability-shrunk weighted consensus;
* C3: literal corroborated-group formula;
* C4: support/OOD/agreement-gated consensus authority;
* C5: consensus withheld from the upstream coordinate.

It is offline, long-only, and :00-only.  It neither reads a live bundle nor
writes an inference, canonical, exchange, admission, portfolio, or exit
artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.research.consensus_ensemble import (  # noqa: E402
    corroborated_recall,
    effective_head_count,
    grouped_median,
    reliability_authority,
    shrunk_reliability_weights,
)


CONTROL_ROOT = ROOT / (
    "data_perp/artifacts/strict_r3_score_family_current_v5_canonical_policy_"
    "reconstruction_2025_2026_20260816_v4"
)
DEFAULT_SCORES = ROOT / (
    "data_perp/artifacts/strict_r3_base_recall_residual2_consensus_research_"
    "20260822_v4_control_parity_full_v2_20260822/control_rescored_target_free.parquet"
)
DEFAULT_LABELS = ROOT / (
    "data_perp/artifacts/strict_r3_long_base_recall_funnel_2025dev_holdout_"
    "2026oos_20260822_v1/outcome_joined_recall_ledger.parquet"
)
DEFAULT_CONTRACT = ROOT / "config/strict_r3_conditional_consensus_v1.json"
DEFAULT_OUT = ROOT / (
    "data_perp/artifacts/strict_r3_stage3_consensus_screen_"
    "20260823_v1"
)

RESERVE_DAYS = 28
BASE_WEIGHT = 0.75
CONSENSUS_WEIGHT = 0.25
PERIODS = {
    "development_2025q1q3": ("2025-01-01", "2025-10-01"),
    "frozen_holdout_2025q4": ("2025-10-01", "2026-01-01"),
    "frozen_oos_2026jan_jul": ("2026-01-01", "2026-08-01"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc(value: object) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _head_columns(frame: pd.DataFrame) -> list[str]:
    columns = sorted(
        name for name in frame.columns
        if name.startswith("conditional_head__") and name.endswith("__rank")
    )
    if len(columns) != 10:
        raise AssertionError(f"expected the frozen ten head ranks, found {len(columns)}")
    return columns


def _load_metadata(contract: Path, head_columns: Iterable[str]) -> pd.DataFrame:
    payload = json.loads(contract.read_text())
    if payload.get("schema") != "strict_r3_conditional_consensus_v1":
        raise AssertionError("not the frozen strict-R3 consensus contract")
    rows: list[dict[str, object]] = []
    expected = set(head_columns)
    for raw in payload["heads"]:
        column = f"conditional_head__{raw['name']}__rank"
        if column not in expected:
            raise AssertionError(f"contract head is absent from score input: {column}")
        cap = int(raw["cap"])
        rows.append({
            "column": column,
            "name": str(raw["name"]),
            "query": str(raw["query"]),
            "weight_mode": str(raw["weight_mode"]),
            "capacity": cap,
            "capacity_group": "small" if cap <= 40 else "medium" if cap <= 80 else "large",
        })
    result = pd.DataFrame(rows).sort_values("column", kind="stable").reset_index(drop=True)
    if set(result["column"]) != expected:
        raise AssertionError("head metadata does not cover the exact frozen vector")
    return result


def _declared_groups(metadata: pd.DataFrame) -> dict[str, tuple[str, ...]]:
    """Build C1 groups from the required declared metadata, never outcomes."""

    groups: dict[str, tuple[str, ...]] = {}
    for key, part in metadata.groupby(["query", "weight_mode", "capacity_group"], sort=True):
        name = "|".join(map(str, key))
        groups[name] = tuple(part.sort_values("column", kind="stable")["column"])
    if not groups:
        raise AssertionError("C1 has no declared groups")
    return groups


def _nanmedian_columns(frame: pd.DataFrame, columns: Iterable[str]) -> np.ndarray:
    values = frame.loc[:, list(columns)].to_numpy(dtype=np.float32, copy=False)
    with np.errstate(all="ignore"):
        return np.nanmedian(values, axis=1).astype(np.float64)


def build_c0_c1_c3(
    frame: pd.DataFrame, metadata: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, dict[float, np.ndarray], dict[str, np.ndarray]]:
    """Return C0/C1 and the declared C3 eta grid from target-free head ranks."""

    heads = _head_columns(frame)
    c0 = _nanmedian_columns(frame, heads)
    groups = _declared_groups(metadata)
    group_scores = grouped_median(
        {column: frame[column].to_numpy(dtype=float, copy=False) for column in heads}, groups,
    )
    with np.errstate(all="ignore"):
        c1 = np.nanmedian(np.column_stack(list(group_scores.values())), axis=1)
    c3 = {eta: corroborated_recall(group_scores, eta=eta) for eta in (.25, .50, .75)}
    return c0, c1, c3, group_scores


def _timestamp_rank_ic(frame: pd.DataFrame, score: str, *, target: str = "policy_net_bps") -> float:
    valid = frame.loc[
        frame[score].notna() & frame[target].notna(),
        ["__decision_ts__", score, target],
    ].copy()
    if valid.empty:
        return float("nan")
    valid["__x"] = valid.groupby("__decision_ts__", sort=False)[score].rank(method="average")
    valid["__y"] = valid.groupby("__decision_ts__", sort=False)[target].rank(method="average")
    group = valid.groupby("__decision_ts__", sort=False)
    count = group["__x"].transform("count")
    mx = group["__x"].transform("mean")
    my = group["__y"].transform("mean")
    dx = valid["__x"] - mx
    dy = valid["__y"] - my
    numer = (dx * dy).groupby(valid["__decision_ts__"], sort=False).sum()
    denom = np.sqrt(
        (dx * dx).groupby(valid["__decision_ts__"], sort=False).sum()
        * (dy * dy).groupby(valid["__decision_ts__"], sort=False).sum()
    )
    values = (numer / denom).where(count.groupby(valid["__decision_ts__"], sort=False).first().ge(2))
    return float(values.replace([np.inf, -np.inf], np.nan).mean())


def _prior_reliability(
    prior: pd.DataFrame, head_columns: Iterable[str],
) -> tuple[dict[str, float], pd.DataFrame]:
    """C2 reliability uses only earlier resolved residual outcomes."""

    work = prior.loc[
        prior["policy_path_valid"].fillna(False).astype(bool)
        & prior["policy_net_bps"].notna()
        & prior["base_anchor_bps"].notna()
        & prior["base_route_timestamp_top30"].fillna(False).astype(bool),
    ].copy()
    work["__residual"] = work["policy_net_bps"] - work["base_anchor_bps"]
    rows: list[dict[str, object]] = []
    values: dict[str, float] = {}
    month_key = work["__decision_ts__"].dt.tz_localize(None).dt.to_period("M")
    for head in head_columns:
        monthly: list[float] = []
        for _, month in work.groupby(month_key, sort=True):
            ic = _timestamp_rank_ic(month, head, target="__residual")
            if np.isfinite(ic):
                monthly.append(float(ic))
        support = int(work[head].notna().sum())
        if monthly:
            median = float(np.median(monthly))
            positive = float(np.mean(np.asarray(monthly) > 0.0))
            minimum = float(np.min(monthly))
            reliability = float(np.clip(.50 + 2 * median + .20 * (positive - .50) + .50 * minimum, 0.0, 1.0))
        else:
            median = positive = minimum = float("nan")
            reliability = 0.0
        values[head] = reliability
        rows.append({
            "head_column": head,
            "prior_support": support,
            "prior_months": len(monthly),
            "median_monthly_residual_rank_ic": median,
            "positive_month_fraction": positive,
            "minimum_monthly_residual_rank_ic": minimum,
            "bounded_reliability": reliability,
        })
    return values, pd.DataFrame(rows)


def _c4_values(
    frame: pd.DataFrame,
    *,
    c0: np.ndarray,
    prior: pd.DataFrame,
    alpha_max: float,
    agreement_power: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Strict-prior C4 normalisers.  Missing telemetry deliberately fails closed."""

    heads = _head_columns(frame)
    matrix = frame.loc[:, heads].to_numpy(dtype=float, copy=False)
    with np.errstate(all="ignore"):
        q25, q75 = np.nanpercentile(matrix, [25, 75], axis=1)
    iqr = q75 - q25
    # The first chronological block has no earlier reconstructed head history.
    # C4 cannot calibrate its normalisers there, so it must assign zero
    # consensus authority rather than derive a held-period normalisation.
    if prior.empty:
        return np.zeros(len(frame), dtype=float), iqr, {
            "iqr_p90": float("nan"), "log_support_p95": float("nan"),
            "ood_p95": float("nan"),
        }
    p_heads = _head_columns(prior)
    p_matrix = prior.loc[:, p_heads].to_numpy(dtype=float, copy=False)
    with np.errstate(all="ignore"):
        p25, p75 = np.nanpercentile(p_matrix, [25, 75], axis=1)
    p_iqr = p75 - p25
    support_column = "rule_support_contribution_weighted"
    ood_column = "rule_ood_joint_factorised"
    scale_iqr = float(np.nanquantile(p_iqr[np.isfinite(p_iqr)], .90)) if np.isfinite(p_iqr).any() else float("nan")
    p_support = pd.to_numeric(prior[support_column], errors="coerce").to_numpy(float)
    p_ood = pd.to_numeric(prior[ood_column], errors="coerce").to_numpy(float)
    scale_support = float(np.nanquantile(np.log1p(np.clip(p_support[np.isfinite(p_support)], 0, None)), .95)) if np.isfinite(p_support).any() else float("nan")
    scale_ood = float(np.nanquantile(np.clip(p_ood[np.isfinite(p_ood)], 0, None), .95)) if np.isfinite(p_ood).any() else float("nan")
    if not np.isfinite(scale_iqr) or scale_iqr <= 0:
        agreement = np.full(len(frame), np.nan)
    else:
        agreement = np.clip(1.0 - iqr / scale_iqr, 0.0, 1.0)
    support = pd.to_numeric(frame[support_column], errors="coerce").to_numpy(float)
    ood = pd.to_numeric(frame[ood_column], errors="coerce").to_numpy(float)
    normal_support = np.full(len(frame), np.nan) if not np.isfinite(scale_support) or scale_support <= 0 else np.clip(np.log1p(np.clip(support, 0, None)) / scale_support, 0.0, 1.0)
    in_domain = np.full(len(frame), np.nan) if not np.isfinite(scale_ood) or scale_ood <= 0 else np.clip(1.0 - np.clip(ood, 0, None) / scale_ood, 0.0, 1.0)
    authority = reliability_authority(
        agreement=agreement,
        normalized_support=normal_support,
        in_domain_probability=in_domain,
        alpha_max=alpha_max,
        agreement_power=agreement_power,
    )
    return authority, iqr, {
        "iqr_p90": scale_iqr,
        "log_support_p95": scale_support,
        "ood_p95": scale_ood,
    }


def _load_scores(scores: Path, control_root: Path) -> pd.DataFrame:
    import pyarrow.parquet as pq

    schema_names = set(pq.read_schema(scores).names)
    head_columns = sorted(
        name for name in schema_names
        if name.startswith("conditional_head__") and name.endswith("__rank")
    )
    needed = [
        "candidate_id", "__decision_ts__", "side_name", "control_block", "base_score",
        "base_rank42", "base_anchor_bps", "conditional_consensus_rank", "upstream",
        "base_route_timestamp_top30", *head_columns,
    ]
    # The score screen must remain target-free.  This explicit construction
    # prevents later schema additions from quietly becoming inputs.
    raw = pd.read_parquet(scores, columns=needed)
    raw["__decision_ts__"] = pd.to_datetime(raw["__decision_ts__"], utc=True, errors="raise")
    phase_error = (raw["__decision_ts__"].dt.minute != 0) | (raw["__decision_ts__"].dt.second != 0)
    if phase_error.any() or raw["candidate_id"].duplicated().any():
        raise AssertionError("Stage 3 input is not a unique :00 candidate population")
    telemetry_parts: list[pd.DataFrame] = []
    fields = ["candidate_id", "rule_support_contribution_weighted", "rule_ood_joint_factorised"]
    for path in sorted((control_root / "scores").glob("block=*.parquet")):
        if path.name.endswith("_audit.parquet"):
            continue
        telemetry_parts.append(pd.read_parquet(path, columns=fields))
    telemetry = pd.concat(telemetry_parts, ignore_index=True)
    if telemetry["candidate_id"].duplicated().any():
        raise AssertionError("frozen telemetry has duplicate candidate identities")
    result = raw.merge(telemetry, on="candidate_id", how="left", validate="one_to_one")
    if result[["rule_support_contribution_weighted", "rule_ood_joint_factorised"]].isna().all(axis=1).any():
        raise AssertionError("frozen C4 telemetry does not cover every candidate")
    return result.sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)


def _load_labels(labels: Path) -> pd.DataFrame:
    result = pd.read_parquet(labels, columns=[
        "candidate_id", "policy_path_valid", "policy_net_bps", "policy_label_available_ts",
    ])
    result["policy_label_available_ts"] = pd.to_datetime(
        result["policy_label_available_ts"], utc=True, errors="coerce",
    )
    if result["candidate_id"].duplicated().any():
        raise AssertionError("policy labels have duplicate candidate identities")
    return result


def _block_cutoff(block: str) -> pd.Timestamp:
    token = str(block).removeprefix("block=").removesuffix("_finalcoverage")
    return pd.Timestamp(token)


def make_stage3_scores(scores: pd.DataFrame, labels: pd.DataFrame, metadata: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute C0--C5 only from frozen target-free inputs and earlier labels."""

    work = scores.copy()
    c0, c1, c3, _ = build_c0_c1_c3(work, metadata)
    # The canonical C0 is an exact reduction parity check, not an approximate
    # comparator.  It validates the stage's upstream input before challenges.
    frozen = work["conditional_consensus_rank"].to_numpy(float)
    if not np.allclose(c0, frozen, equal_nan=True, rtol=0.0, atol=0.0):
        raise AssertionError("C0 head median is not bit-identical to frozen conditional consensus")
    work["consensus__c0_median"] = c0
    work["consensus__c1_grouped"] = c1
    for eta, values in c3.items():
        work[f"consensus__c3_corroborated_eta{int(eta * 100):02d}"] = values
    for name in (
        "c0_median", "c1_grouped",
        "c3_corroborated_eta25", "c3_corroborated_eta50", "c3_corroborated_eta75",
    ):
        work[f"upstream__{name}"] = np.where(
            work["base_route_timestamp_top30"].fillna(False).to_numpy(bool),
            BASE_WEIGHT * work["base_rank42"].to_numpy(float)
            + CONSENSUS_WEIGHT * work[f"consensus__{name}"].to_numpy(float),
            np.nan,
        )
    work["upstream__c5_base_only"] = np.where(
        work["base_route_timestamp_top30"].fillna(False).to_numpy(bool),
        work["base_rank42"].to_numpy(float),
        np.nan,
    )
    outcome = work.loc[:, ["candidate_id", "__decision_ts__", "control_block", "base_anchor_bps", "base_route_timestamp_top30", *_head_columns(work), "rule_support_contribution_weighted", "rule_ood_joint_factorised"]].merge(
        labels, on="candidate_id", how="left", validate="one_to_one",
    )
    audit: list[dict[str, Any]] = []
    for block in sorted(work["control_block"].unique(), key=_block_cutoff):
        cutoff = _block_cutoff(str(block))
        reserve_start = cutoff - pd.Timedelta(days=RESERVE_DAYS)
        current = work["control_block"].eq(block).to_numpy()
        prior = outcome.loc[
            outcome["__decision_ts__"].lt(reserve_start)
            & outcome["policy_label_available_ts"].lt(reserve_start)
        ].copy()
        reliability, head_audit = _prior_reliability(prior, _head_columns(work))
        for rho in (.25, .50):
            for temp in (.02, .05):
                weights = shrunk_reliability_weights(reliability, rho=rho, temperature=temp)
                column = f"consensus__c2_r{int(rho * 100):02d}_t{int(temp * 100):02d}"
                head_count = effective_head_count(weights)
                if head_count < 3.0:
                    # A near-single-head consensus is not an admissible C2
                    # arm.  Preserve an explicit non-score rather than let a
                    # later screen/MC1 stage mistake it for a valid output.
                    values = np.full(current.sum(), np.nan, dtype=float)
                    eligible = False
                else:
                    values = np.zeros(current.sum(), dtype=float)
                    for head, weight in weights.items():
                        values += weight * work.loc[current, head].to_numpy(float)
                    eligible = True
                work.loc[current, column] = values
                work.loc[current, f"upstream__{column.removeprefix('consensus__')}"] = np.where(
                    work.loc[current, "base_route_timestamp_top30"].fillna(False).to_numpy(bool),
                    BASE_WEIGHT * work.loc[current, "base_rank42"].to_numpy(float) + CONSENSUS_WEIGHT * values,
                    np.nan,
                )
                audit.append({
                    "arm": column, "control_block": block, "cutoff": cutoff,
                    "reserve_start": reserve_start, "prior_label_rows": int(len(prior)),
                    "prior_label_max": prior["policy_label_available_ts"].max(),
                    "effective_head_count": head_count,
                    "effective_head_count_passed": eligible,
                    "weights": json.dumps(weights, sort_keys=True),
                    "strict_reserve": bool(prior["policy_label_available_ts"].lt(reserve_start).all()),
                    "head_reliability": head_audit.to_json(orient="records"),
                })
        for alpha in (.25, .40, .50):
            for power in (1.0, 2.0):
                authority, iqr, normalisers = _c4_values(
                    work.loc[current], c0=c0[current], prior=prior,
                    alpha_max=alpha, agreement_power=power,
                )
                key = f"c4_a{int(alpha * 100):02d}_p{int(power)}"
                work.loc[current, f"agreement__{key}_iqr"] = iqr
                work.loc[current, f"authority__{key}"] = authority
                upstream = work.loc[current, "base_rank42"].to_numpy(float) + authority * (
                    c0[current] - work.loc[current, "base_rank42"].to_numpy(float)
                )
                work.loc[current, f"upstream__{key}"] = np.where(
                    work.loc[current, "base_route_timestamp_top30"].fillna(False).to_numpy(bool), upstream, np.nan,
                )
                audit.append({
                    "arm": key, "control_block": block, "cutoff": cutoff,
                    "reserve_start": reserve_start, "prior_label_rows": int(len(prior)),
                    "prior_label_max": prior["policy_label_available_ts"].max(),
                    "strict_reserve": bool(prior["policy_label_available_ts"].lt(reserve_start).all()),
                    **normalisers,
                    "mean_authority": float(np.nanmean(authority)),
                    "nonzero_authority_fraction": float(np.mean(authority > 0)),
                })
    # Predicted score artifact must not contain labels/outcomes.
    forbidden = {"policy_path_valid", "policy_net_bps", "policy_label_available_ts"}
    if forbidden & set(work.columns):
        raise AssertionError("target-free Stage 3 score artifact contains policy labels")
    return work, pd.DataFrame(audit)


def _metrics(scores: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    work = scores.merge(labels, on="candidate_id", how="left", validate="one_to_one")
    upstreams = sorted(column for column in work if column.startswith("upstream__"))
    rows: list[dict[str, Any]] = []
    for arm in upstreams:
        for period, (start, end) in PERIODS.items():
            subset = work.loc[
                work["__decision_ts__"].ge(_utc(start))
                & work["__decision_ts__"].lt(_utc(end))
                & work["base_route_timestamp_top30"].fillna(False).astype(bool),
            ].copy()
            valid = subset["policy_path_valid"].fillna(False).astype(bool) & subset["policy_net_bps"].notna()
            timestamp_rank = subset.groupby("__decision_ts__", sort=False)[arm].rank(method="first", ascending=False)
            count = subset.groupby("__decision_ts__", sort=False)[arm].transform("count")
            for tail in (.10, .20, .30):
                selected = valid & timestamp_rank.le(np.ceil(tail * count))
                rows.append({
                    "arm": arm, "period": period, "tail_of_base_route": tail,
                    "candidate_rows": int(len(subset)), "selected_rows": int(selected.sum()),
                    "policy_net_mean_bps": float(subset.loc[selected, "policy_net_bps"].mean()) if selected.any() else float("nan"),
                    "policy_net_median_bps": float(subset.loc[selected, "policy_net_bps"].median()) if selected.any() else float("nan"),
                    "positive_fraction": float((subset.loc[selected, "policy_net_bps"] > 0).mean()) if selected.any() else float("nan"),
                    "within_timestamp_policy_rank_ic": _timestamp_rank_ic(subset, arm),
                })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, default=DEFAULT_SCORES)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--control-root", type=Path, default=CONTROL_ROOT)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable output already exists: {args.out_dir}")
    for path in (args.scores, args.labels, args.contract):
        if not path.is_file():
            raise FileNotFoundError(path)
    raw = _load_scores(args.scores, args.control_root)
    labels = _load_labels(args.labels)
    metadata = _load_metadata(args.contract, _head_columns(raw))
    target_free, audit = make_stage3_scores(raw, labels, metadata)
    metrics = _metrics(target_free, labels)
    args.out_dir.mkdir(parents=True)
    target_free.to_parquet(args.out_dir / "stage3_target_free_scores.parquet", index=False, compression="zstd")
    metadata.to_parquet(args.out_dir / "head_metadata.parquet", index=False)
    audit.to_parquet(args.out_dir / "strict_prequential_audit.parquet", index=False)
    metrics.to_parquet(args.out_dir / "consensus_screen_metrics.parquet", index=False)
    manifest = {
        "schema": "strict_r3_stage3_consensus_screen_v1",
        "scope": "offline long-only :00-only Stage-3 screen; no live/canonical/admission/portfolio/exit artifact modified",
        "input_scores": {"path": str(args.scores), "sha256": _sha256(args.scores)},
        "input_labels": {"path": str(args.labels), "sha256": _sha256(args.labels)},
        "consensus_contract": {"path": str(args.contract), "sha256": _sha256(args.contract)},
        "reserve_days": RESERVE_DAYS,
        "arms": ["C0", "C1", "C2 rho/temperature grid", "C3 eta grid (.25/.50/.75)", "C4 authority grid", "C5 base-only"],
        "label_use": "policy labels joined only after target-free variant outputs are final; earlier resolved labels only determine C2/C4 block normalisers",
        "next_required": "rebuild correctness/CDF then native current/BCF MC1 and constrained portfolio for finalists; this screen cannot advance an arm",
    }
    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"event": "stage3_screen_complete", "rows": len(target_free), "metrics": len(metrics)}, sort_keys=True))


if __name__ == "__main__":
    main()
