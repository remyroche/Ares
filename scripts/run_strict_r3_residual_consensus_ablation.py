#!/usr/bin/env python3
"""Offline C0--C4 consensus ablations for strict-R3 :00 residual heads.

Inputs are fixed out-of-fold layer-two predictions.  This producer never
re-trains a residual head or uses a held outcome as an inference feature.  All
reliability, support, OOD, and agreement normalisations are estimated from
strictly earlier, resolved rows before the current block's 28-day reserve.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

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
from extreme_price_movements.research.hierarchical_residual import bounded_rank_fusion  # noqa: E402


DEFAULT_L2 = ROOT / (
    "data_perp/artifacts/strict_r3_base_recall_residual2_consensus_research_"
    "20260822_v2/layer2_oof_predictions.parquet"
)
DEFAULT_B0 = ROOT / (
    "data_perp/artifacts/strict_r3_long_base_recall_funnel_2025dev_holdout_"
    "2026oos_20260822_v1/outcome_joined_recall_ledger.parquet"
)
DEFAULT_CONTROL = ROOT / (
    "data_perp/artifacts/strict_r3_score_family_current_v5_canonical_policy_"
    "reconstruction_2025_2026_20260816_v4"
)
DEFAULT_OUT = ROOT / (
    "data_perp/artifacts/strict_r3_base_recall_residual2_consensus_research_"
    "20260822_v2/consensus_c0_c4"
)
RESERVE_DAYS = 28
PERIODS = {
    "development_2025q1q3": ("2025-01-01", "2025-10-01"),
    "frozen_holdout_2025q4": ("2025-10-01", "2026-01-01"),
    "frozen_oos_2026jan_jul": ("2026-01-01", "2026-08-01"),
}


def _utc(value: object) -> pd.Timestamp:
    value = pd.Timestamp(value)
    return value.tz_localize("UTC") if value.tzinfo is None else value.tz_convert("UTC")


def _cutoff(block: str) -> pd.Timestamp:
    match = re.fullmatch(r"block=(\d{8}T\d{6}Z)(?:_finalcoverage)?", block)
    if match is None:
        raise ValueError(block)
    return pd.Timestamp(match.group(1))


def _head_columns(frame: pd.DataFrame) -> list[str]:
    names = sorted(
        column for column in frame
        if column.startswith("layer2_head__") and column.endswith("__rank")
    )
    if len(names) != 6:
        raise AssertionError(f"expected six R1 head ranks, found {len(names)}")
    return names


def _head_metadata(layer2_path: Path, head_columns: list[str]) -> pd.DataFrame:
    """Load the frozen R1 metadata emitted with the strict-OOF predictions.

    Consensus independence is declared by the residual-head contract, never
    reverse-engineered from a held outcome or score correlation.  The Stage-2
    runner persists this metadata beside the score file specifically so that
    later consensus arms can use stable query/weight/capacity identities.
    """

    audit_path = layer2_path.parent / "head_audit.parquet"
    if not audit_path.is_file():
        raise FileNotFoundError(f"missing immutable R1 head audit: {audit_path}")
    audit = pd.read_parquet(
        audit_path,
        columns=["head", "query", "weight_mode", "field_count"],
    ).drop_duplicates()
    rows: list[dict[str, object]] = []
    for column in head_columns:
        marker = column.removeprefix("layer2_head__").removesuffix("__rank")
        if not marker.startswith("r2_"):
            raise AssertionError(f"unexpected R1 head field: {column}")
        name = marker
        matched = audit.loc[audit["head"].eq(name)]
        if len(matched) != 1:
            raise AssertionError(f"R1 metadata is not unique for {name}: {len(matched)} rows")
        cap = re.search(r"cap(\d+)", name)
        if cap is None:
            raise AssertionError(f"R1 head name lacks declared feature capacity: {name}")
        item = matched.iloc[0]
        rows.append({
            "column": column,
            "head": name,
            "capacity": int(cap.group(1)),
            "query": str(item["query"]),
            "weight_mode": str(item["weight_mode"]),
            "field_count": int(item["field_count"]),
        })
    metadata = pd.DataFrame(rows).sort_values(["capacity", "query", "weight_mode", "head"], kind="stable")
    if set(metadata["column"]) != set(head_columns):
        raise AssertionError("R1 metadata does not cover the exact head vector")
    return metadata.reset_index(drop=True)


def _capacity_groups(metadata: pd.DataFrame) -> dict[str, tuple[str, ...]]:
    """Build independent groups from frozen head capacity families.

    The ten-head incumbent varies query and weighting within each capacity
    family.  Grouping by capacity prevents several variants of one model-size
    family from winning merely through multiplicity, while retaining their
    internal median as a stable family view.
    """

    groups = {
        f"cap{int(capacity)}": tuple(group["column"])
        for capacity, group in metadata.groupby("capacity", sort=True)
    }
    if len(groups) < 2 or any(not values for values in groups.values()):
        raise AssertionError(f"insufficient declared capacity groups: {groups}")
    return groups


def _head_reliability_from_prior(prior: pd.DataFrame, head_columns: list[str]) -> tuple[dict[str, float], pd.DataFrame]:
    """Summarise only strictly-prior monthly rank evidence for C2 weights."""

    rows: list[dict[str, object]] = []
    reliability: dict[str, float] = {}
    for field in head_columns:
        monthly: list[float] = []
        # Calendar-month aggregation is only a reporting bucket; strip the
        # UTC timezone explicitly to avoid an implicit, warning-producing
        # timezone discard while retaining the same UTC month identity.
        month_key = prior["__decision_ts__"].dt.tz_localize(None).dt.to_period("M")
        for _, month in prior.groupby(month_key, sort=True):
            value = _rank_ic(month, field)
            if np.isfinite(value):
                monthly.append(float(value))
        support = int(prior[field].notna().sum())
        if monthly:
            median_ic = float(np.median(monthly))
            positive_fraction = float(np.mean(np.asarray(monthly) > 0.0))
            minimum_ic = float(np.min(monthly))
            # Bounded, deliberately modest evidence coordinate.  The uniform
            # component in C2 remains substantial even for the strongest head.
            value = float(np.clip(
                0.50 + 2.0 * median_ic + 0.20 * (positive_fraction - .50) + 0.50 * minimum_ic,
                0.0,
                1.0,
            ))
        else:
            median_ic = positive_fraction = minimum_ic = float("nan")
            value = 0.0
        reliability[field] = value
        rows.append({
            "head_column": field,
            "prior_months": len(monthly),
            "prior_support": support,
            "median_monthly_residual_rank_ic": median_ic,
            "positive_ic_month_fraction": positive_fraction,
            "minimum_monthly_residual_rank_ic": minimum_ic,
            "bounded_reliability": value,
        })
    return reliability, pd.DataFrame(rows)


def _telemetry(control_root: Path, candidate_ids: pd.Series) -> pd.DataFrame:
    fields = [
        "candidate_id", "rule_support_contribution_weighted", "rule_support_effective",
        "path_support_effective_28d", "rule_ood_joint_factorised",
        "path_ood_conditioned", "model_ood_mahalanobis_diag",
    ]
    parts: list[pd.DataFrame] = []
    for path in sorted((control_root / "scores").glob("block=*.parquet")):
        present = pd.read_parquet(path, columns=fields)
        parts.append(present)
    output = pd.concat(parts, ignore_index=True)
    output = output.loc[output["candidate_id"].isin(candidate_ids)].copy()
    if output["candidate_id"].duplicated().any():
        raise AssertionError("telemetry score blocks have duplicate candidate identities")
    return output


def _empirical_cdf(reference: np.ndarray, values: np.ndarray, *, higher_is_better: bool) -> np.ndarray:
    reference = np.sort(reference[np.isfinite(reference)])
    out = np.full(len(values), np.nan, dtype=float)
    valid = np.isfinite(values)
    if len(reference) < 100:
        return out
    side = "right" if higher_is_better else "left"
    pos = np.searchsorted(reference, values[valid], side=side).astype(float) / float(len(reference))
    out[valid] = pos if higher_is_better else 1.0 - pos
    return np.clip(out, 0.0, 1.0)


def _rank_ic(frame: pd.DataFrame, field: str) -> float:
    valid = frame.loc[
        frame["policy_path_valid"].fillna(False).astype(bool)
        & frame["policy_net_bps"].notna() & frame[field].notna(),
        ["__decision_ts__", field, "policy_net_bps"],
    ].copy()
    if valid.empty:
        return float("nan")
    valid["x"] = valid.groupby("__decision_ts__", sort=False)[field].rank(method="average")
    valid["y"] = valid.groupby("__decision_ts__", sort=False)["policy_net_bps"].rank(method="average")
    values = valid.groupby("__decision_ts__", sort=False)[["x", "y"]].corr().iloc[0::2, -1]
    return float(values.mean()) if len(values) else float("nan")


def _metrics(frame: pd.DataFrame, fields: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for field in fields:
        for period, (start, end) in PERIODS.items():
            subset = frame.loc[
                frame["__decision_ts__"].ge(_utc(start)) & frame["__decision_ts__"].lt(_utc(end))
                & frame["base_route_timestamp_top30"].fillna(False).astype(bool)
            ].copy()
            valid = subset["policy_path_valid"].fillna(False).astype(bool) & subset["policy_net_bps"].notna()
            rank = subset.groupby("__decision_ts__", sort=False)[field].rank(method="first", ascending=False)
            count = subset.groupby("__decision_ts__", sort=False)[field].transform("count")
            for fraction in (.10, .20, .30):
                selected = rank.le(np.ceil(fraction * count)) & valid
                rows.append({
                    "arm": field, "period": period, "tail_of_base_route": fraction,
                    "selected_rows": int(selected.sum()),
                    "policy_net_mean_bps": float(subset.loc[selected, "policy_net_bps"].mean()) if selected.any() else float("nan"),
                    "policy_net_median_bps": float(subset.loc[selected, "policy_net_bps"].median()) if selected.any() else float("nan"),
                    "within_timestamp_policy_rank_ic": _rank_ic(subset, field),
                })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer2", type=Path, default=DEFAULT_L2)
    parser.add_argument("--b0", type=Path, default=DEFAULT_B0)
    parser.add_argument("--control-root", type=Path, default=DEFAULT_CONTROL)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable output exists: {args.out_dir}")
    frame = pd.read_parquet(args.layer2)
    b0 = pd.read_parquet(
        args.b0,
        columns=[
            "candidate_id", "policy_path_valid", "policy_net_bps",
            "policy_label_available_ts",
        ],
    )
    b0["policy_label_available_ts"] = pd.to_datetime(
        b0["policy_label_available_ts"], utc=True, errors="coerce",
    )
    frame = frame.merge(b0, on="candidate_id", how="left", validate="one_to_one")
    frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True, errors="raise")
    head_columns = _head_columns(frame)
    metadata = _head_metadata(args.layer2, head_columns)
    telemetry = _telemetry(args.control_root, frame["candidate_id"])
    frame = frame.merge(telemetry, on="candidate_id", how="left", validate="one_to_one")
    if len(frame) != len(b0):
        raise AssertionError("consensus input identity count changed")
    groups = _capacity_groups(metadata)
    score_rows: list[pd.DataFrame] = []
    audit_rows: list[dict[str, object]] = []
    reliability_rows: list[pd.DataFrame] = []
    metric_fields = ["C0_median", "C1_grouped_capacity"]
    for block, held in frame.groupby("control_block", sort=True):
        held = held.copy()
        cutoff = _cutoff(str(block))
        reserve = cutoff - pd.Timedelta(days=RESERVE_DAYS)
        prior = frame.loc[
            frame["__decision_ts__"].lt(reserve)
            & frame["policy_label_available_ts"].lt(reserve)
            & frame["policy_path_valid"].fillna(False).astype(bool)
            & frame["policy_net_bps"].notna(),
        ].copy()
        output = held.loc[:, ["candidate_id", "__decision_ts__", "control_block", "upstream", "base_route_timestamp_top30", *head_columns]].copy()
        # C0: simple median.
        output["C0_median"] = held["layer2_consensus_rank"].to_numpy(float)
        group_scores = grouped_median({name: held[name].to_numpy(float) for name in head_columns}, groups)
        for name, values in group_scores.items():
            output[f"group_{name}_median"] = values
        output["C1_grouped_capacity"] = np.nanmedian(np.column_stack(list(group_scores.values())), axis=1)
        for eta in (.25, .50, .75):
            field = f"C3_corroborated_eta{int(eta * 100):02d}"
            output[field] = corroborated_recall(group_scores, eta=eta)
            metric_fields.append(field)
        # C2: reliability may only use preceding resolved L2 outputs.  With no
        # prior score support, it falls back exactly to C0 rather than inventing
        # held-period reliability.
        reliability, reliability_audit = _head_reliability_from_prior(prior, head_columns)
        reliability_audit["control_block"] = block
        reliability_rows.append(reliability_audit)
        c2_audit: dict[str, object] = {}
        for rho in (.25, .50):
            for temperature in (.02, .05):
                suffix = f"rho{int(rho * 100):02d}_t{int(temperature * 100):02d}"
                field = f"C2_reliability_{suffix}"
                try:
                    weights = shrunk_reliability_weights(reliability, rho=rho, temperature=temperature)
                    n_eff = effective_head_count(weights)
                except ValueError:
                    weights, n_eff = {name: 1.0 / len(head_columns) for name in head_columns}, float(len(head_columns))
                if n_eff < 3.0 or len(prior) < 1000:
                    output[field] = output["C0_median"]
                    status = "fallback_C0_insufficient_prior_support_or_effective_heads"
                else:
                    output[field] = sum(weights[name] * held[name].to_numpy(float) for name in head_columns)
                    status = "scored"
                output[f"{field}__effective_head_count"] = n_eff
                output[f"{field}__status"] = status
                c2_audit[f"{field}__status"] = status
                c2_audit[f"{field}__effective_head_count"] = n_eff
                c2_audit.update({f"{field}__weight__{name}": value for name, value in weights.items()})
                metric_fields.append(field)
        # C4: support and OOD CDFs are based only on earlier score-block
        # telemetry.  Agreement is low head dispersion (higher is better).
        support_raw = np.log1p(pd.to_numeric(held["rule_support_contribution_weighted"], errors="coerce").to_numpy(float))
        prior_support = np.log1p(pd.to_numeric(prior["rule_support_contribution_weighted"], errors="coerce").to_numpy(float))
        ood_raw = pd.to_numeric(held["rule_ood_joint_factorised"], errors="coerce").to_numpy(float)
        prior_ood = pd.to_numeric(prior["rule_ood_joint_factorised"], errors="coerce").to_numpy(float)
        dispersion = pd.to_numeric(held["layer2_agreement__iqr"], errors="coerce").to_numpy(float)
        prior_dispersion = pd.to_numeric(prior["layer2_agreement__iqr"], errors="coerce").to_numpy(float)
        support = _empirical_cdf(prior_support, support_raw, higher_is_better=True)
        in_domain = _empirical_cdf(prior_ood, ood_raw, higher_is_better=False)
        agreement = _empirical_cdf(prior_dispersion, dispersion, higher_is_better=False)
        for alpha in (.25, .40, .50):
            for power in (1.0, 2.0):
                suffix = f"a{int(alpha * 100):02d}_p{int(power)}"
                authority = reliability_authority(
                    agreement=np.nan_to_num(agreement, nan=0.0),
                    normalized_support=np.nan_to_num(support, nan=0.0),
                    in_domain_probability=np.nan_to_num(in_domain, nan=0.0),
                    alpha_max=alpha,
                    agreement_power=power,
                )
                output[f"C4_authority_{suffix}"] = authority
                field = f"C4_support_ood_agreement_{suffix}"
                output[field] = bounded_rank_fusion(
                    held["upstream"].fillna(0.0), output["C1_grouped_capacity"].fillna(0.0), authority,
                )
                metric_fields.append(field)
        output["C4_prior_rows"] = len(prior)
        audit_rows.append({
            "block": block, "cutoff": cutoff.isoformat(), "reserve_start": reserve.isoformat(),
            "prior_resolved_rows": len(prior),
            **c2_audit,
        })
        score_rows.append(output)
    scores = pd.concat(score_rows, ignore_index=True)
    if scores["candidate_id"].duplicated().any():
        raise AssertionError("consensus score output has duplicate identities")
    args.out_dir.mkdir(parents=True)
    scores.to_parquet(args.out_dir / "consensus_oof_predictions.parquet", index=False, compression="zstd")
    pd.DataFrame(audit_rows).to_parquet(args.out_dir / "consensus_fit_audit.parquet", index=False)
    pd.concat(reliability_rows, ignore_index=True).to_parquet(
        args.out_dir / "head_reliability_prior_audit.parquet", index=False,
    )
    labelled = scores.merge(b0, on="candidate_id", how="left", validate="one_to_one")
    metrics = _metrics(labelled, list(dict.fromkeys(metric_fields)))
    metrics.to_parquet(args.out_dir / "consensus_metrics.parquet", index=False)
    head_metrics = _metrics(labelled, head_columns).merge(
        metadata.rename(columns={"column": "arm"}), on="arm", how="left", validate="many_to_one",
    )
    head_metrics.to_parquet(args.out_dir / "head_metrics.parquet", index=False)
    routed = scores.loc[scores["base_route_timestamp_top30"].fillna(False).astype(bool), head_columns]
    correlation = routed.corr(method="spearman").stack(dropna=False).rename("spearman_rank_correlation").reset_index()
    correlation.columns = ["head_left", "head_right", "spearman_rank_correlation"]
    correlation.to_parquet(args.out_dir / "head_correlations.parquet", index=False)
    manifest = {
        "schema": "strict_r3_residual_consensus_c0_c4_v2",
        "scope": "offline :00-only research; no live, canonical, MC1, admission, portfolio or execution change",
        "reserve_days": RESERVE_DAYS,
        "C0": "six-head simple median",
        "head_metadata": metadata.to_dict(orient="records"),
        "C1": "median of declared capacity-family medians; query/weight metadata retained in head audit",
        "C2": "strict-prior monthly-IC/support reliability; rho in {.25,.50}, temperature in {.02,.05}, effective-head-count >=3",
        "C3": "predeclared corroborated-recall eta in {.25,.50,.75}",
        "C4": "support/OOD/agreement authority alpha in {.25,.40,.50}, agreement power in {1,2}; all CDF references strict-prior",
        "labels": "joined only after target-free consensus score construction",
    }
    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"event": "complete", "rows": len(scores), "metrics": len(metrics)}, sort_keys=True))


if __name__ == "__main__":
    main()
