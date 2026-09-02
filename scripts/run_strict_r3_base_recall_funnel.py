#!/usr/bin/env python3
"""Strict-R3 long base-recall Funnel A, using frozen current-v5 bundles.

This is an offline research producer.  It first reconstructs the target-free
base probabilities from each persisted current-v5 monthly upstream bundle and
requires exact parity with the saved B0 score blocks.  Only then does it score
the B1 formula grid and B2 route-width controls.  Policy/R3 outcomes are joined
after scores and timestamp-local route membership have been finalized.

The producer deliberately stops before rebuilding a selected B1 downstream
stack.  That expensive work is allowed only after the screening metrics pass
their predeclared development/holdout gates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.strict_r3_canonical_current import (
    score_monthly_upstream_bundle,
)


DEFAULT_SOURCE = ROOT / (
    "data_perp/artifacts/strict_r3_schema_v2_prequential_ledger_targetfree_long_"
    "2024_2026_20260809_v1/prequential_stack_ledger.parquet"
)
DEFAULT_CONTROL = ROOT / (
    "data_perp/artifacts/strict_r3_score_family_current_v5_canonical_policy_"
    "reconstruction_2025_2026_20260816_v4"
)
DEFAULT_POLICY = ROOT / (
    "data_perp/artifacts/strict_r3_source_aligned_optimized_policy_outcomes_long_"
    "2024jan_jul2026_20260812_v1/candidate_policy_outcomes.parquet"
)

PERIODS = {
    "development_2025q1q3": ("2025-01-01", "2025-10-01"),
    "frozen_holdout_2025q4": ("2025-10-01", "2026-01-01"),
    "frozen_oos_2026jan_jul": ("2026-01-01", "2026-08-01"),
}
BASE_ROUTE_FRACTION = 0.30
EPS = 1e-9


@dataclass(frozen=True)
class Arm:
    name: str
    beta_weak: float
    alpha_adverse: float
    route_fraction: float
    family: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc(value: object) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def b1_score(
    p_clear: np.ndarray | pd.Series,
    p_weak: np.ndarray | pd.Series,
    p_adverse: np.ndarray | pd.Series,
    *,
    beta_weak: float,
    alpha_adverse: float,
) -> np.ndarray:
    """Return the predeclared B1 probability-space score."""
    return (
        np.asarray(p_clear, dtype=float)
        + float(beta_weak) * np.asarray(p_weak, dtype=float)
        - float(alpha_adverse) * np.asarray(p_adverse, dtype=float)
    )


def timestamp_route(
    frame: pd.DataFrame,
    score_column: str,
    *,
    fraction: float,
) -> np.ndarray:
    """Deterministic top-fraction route by decision timestamp and candidate ID."""
    if not 0.0 < float(fraction) <= 1.0:
        raise ValueError("route fraction must lie in (0, 1]")
    required = {"candidate_id", "__decision_ts__", score_column}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"timestamp route misses fields: {missing}")
    work = frame.loc[:, ["candidate_id", "__decision_ts__", score_column]].copy()
    work["__row__"] = np.arange(len(work), dtype=np.int64)
    work = work.sort_values(
        ["__decision_ts__", score_column, "candidate_id"],
        ascending=[True, False, True], kind="stable", na_position="last",
    )
    work["__rank__"] = work.groupby("__decision_ts__", sort=False).cumcount()
    count = work.groupby("__decision_ts__", sort=False)["candidate_id"].transform("size")
    work["__keep__"] = work["__rank__"].lt(np.ceil(float(fraction) * count).astype(int))
    return work.sort_values("__row__", kind="stable")["__keep__"].to_numpy(bool)


def _numeric_parity(
    left: pd.Series,
    right: pd.Series,
    *,
    rtol: float = 1e-4,
    atol: float = 1e-8,
) -> tuple[bool, dict[str, float]]:
    a = pd.to_numeric(left, errors="coerce").to_numpy(float)
    b = pd.to_numeric(right, errors="coerce").to_numpy(float)
    finite = np.isfinite(a) & np.isfinite(b)
    same_nan = np.isnan(a) & np.isnan(b)
    close = np.isclose(a, b, rtol=rtol, atol=atol, equal_nan=True)
    delta = np.abs(a[finite] - b[finite])
    relative = delta / np.maximum(np.abs(a[finite]), EPS)
    return bool(close.all()), {
        "rows": float(len(a)),
        "exact_or_tolerant_fraction": float(close.mean()),
        "same_nan_fraction": float(same_nan.mean()),
        "max_abs_delta": float(delta.max()) if len(delta) else 0.0,
        "p99_abs_delta": float(np.quantile(delta, .99)) if len(delta) else 0.0,
        "max_relative_delta": float(relative.max()) if len(relative) else 0.0,
    }


def _load_target_free_block(source: Path, bundle: object) -> pd.DataFrame:
    cutoff = _utc(bundle.cutoff)
    end = _utc(bundle.end_exclusive)
    start = cutoff - pd.Timedelta(days=28)
    columns = ["candidate_id", "__decision_ts__", "side_name", *bundle.base_fields]
    frame = pd.read_parquet(
        source,
        columns=columns,
        filters=[("__decision_ts__", ">=", start), ("__decision_ts__", "<", end)],
    )
    frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True, errors="raise")
    if frame.empty:
        raise ValueError(f"no target-free rows for frozen bundle {cutoff.isoformat()}")
    return frame


def reconstruct_b0(
    *,
    source: Path,
    control_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Re-score every frozen upstream block and require B0 parity."""
    rows: list[pd.DataFrame] = []
    audits: list[dict[str, object]] = []
    bundle_root = control_root / "bundles"
    score_root = control_root / "scores"
    if not bundle_root.is_dir() or not score_root.is_dir():
        raise FileNotFoundError("control root misses persisted upstream bundles or score blocks")
    for block_dir in sorted(bundle_root.glob("block=*")):
        upstream_path = block_dir / "upstream/monthly_upstream_bundle.joblib"
        score_path = score_root / f"{block_dir.name}.parquet"
        if not upstream_path.is_file() or not score_path.is_file():
            raise FileNotFoundError(f"incomplete frozen control block: {block_dir.name}")
        bundle = joblib.load(upstream_path)
        target_free = _load_target_free_block(source, bundle)
        cutoff = _utc(bundle.cutoff)
        rebuilt = score_monthly_upstream_bundle(
            bundle,
            target_free,
            allow_prior_reference=True,
            prior_reference_start=cutoff - pd.Timedelta(days=28),
            route_top_fraction=BASE_ROUTE_FRACTION,
        )
        held = rebuilt.loc[rebuilt["__decision_ts__"].ge(cutoff)].copy()
        stored_columns = [
            "candidate_id", "base_score", "base_rank42", "base_anchor_bps",
            "conditional_consensus_rank", "upstream",
        ]
        stored = pd.read_parquet(score_path, columns=stored_columns)
        joined = stored.merge(held, on="candidate_id", how="outer", indicator=True,
                              suffixes=("__stored", "__replayed"), validate="one_to_one")
        identity_counts = joined["_merge"].value_counts().to_dict()
        if int(identity_counts.get("left_only", 0)):
            raise AssertionError(
                f"B0 frozen candidates missing from reconstruction for {block_dir.name}: "
                f"{identity_counts}"
            )
        # The terminal final-coverage bundle predates a later target-free
        # source expansion.  Its persisted score block is the frozen universe
        # for this control.  Keep only those identities and retain an explicit
        # audit count rather than allowing new source rows into B0/B1 metrics.
        held = held.loc[held["candidate_id"].isin(stored["candidate_id"])].copy()
        joined = joined.loc[joined["_merge"].eq("both")].copy()
        audit: dict[str, object] = {
            "block": block_dir.name,
            "cutoff": cutoff.isoformat(),
            "held_rows": int(len(held)),
            "target_free_source_rows": int(len(target_free)),
            "source_extra_rows_excluded": int(identity_counts.get("right_only", 0)),
            "identity_parity": True,
        }
        for field in (
            "base_score", "base_rank42", "base_anchor_bps",
            "conditional_consensus_rank", "upstream",
        ):
            okay, result = _numeric_parity(
                joined[f"{field}__stored"], joined[f"{field}__replayed"],
            )
            audit[f"{field}_parity"] = okay
            for key, value in result.items():
                audit[f"{field}_{key}"] = value
            if not okay:
                raise AssertionError(f"B0 {field} parity failed for {block_dir.name}: {result}")
        held = held.loc[:, [
            "candidate_id", "__decision_ts__", "side_name", "p_adverse", "p_weak",
            "p_clear", "base_score", "base_rank42", "base_anchor_bps",
            "conditional_consensus_rank", "upstream", "ordinary_shadow_consensus_rank",
            "ordinary_shadow_upstream", "base_route_timestamp_top30",
        ]].copy()
        held["control_block"] = block_dir.name
        rows.append(held)
        audits.append(audit)
    output = pd.concat(rows, ignore_index=True).sort_values(
        ["__decision_ts__", "candidate_id"], kind="stable",
    ).reset_index(drop=True)
    if output["candidate_id"].duplicated().any():
        raise AssertionError("frozen B0 blocks overlap in held candidate identities")
    return output, pd.DataFrame(audits)


def _rank_ic(frame: pd.DataFrame, score_column: str) -> tuple[float, float]:
    valid = frame.loc[
        frame["policy_path_valid"].fillna(False).astype(bool)
        & pd.to_numeric(frame["policy_net_bps"], errors="coerce").notna()
        & pd.to_numeric(frame[score_column], errors="coerce").notna(),
        ["__decision_ts__", score_column, "policy_net_bps"],
    ].copy()
    if valid.empty:
        return float("nan"), float("nan")
    valid["__x__"] = valid.groupby("__decision_ts__", sort=False)[score_column].rank(method="average")
    valid["__y__"] = valid.groupby("__decision_ts__", sort=False)["policy_net_bps"].rank(method="average")
    grouped = valid.groupby("__decision_ts__", sort=False)
    n = grouped.size().astype(float)
    sx = grouped["__x__"].sum()
    sy = grouped["__y__"].sum()
    sxx = (valid["__x__"] ** 2).groupby(valid["__decision_ts__"], sort=False).sum()
    syy = (valid["__y__"] ** 2).groupby(valid["__decision_ts__"], sort=False).sum()
    sxy = (valid["__x__"] * valid["__y__"]).groupby(valid["__decision_ts__"], sort=False).sum()
    denominator = np.sqrt((n * sxx - sx * sx) * (n * syy - sy * sy))
    corr = (n * sxy - sx * sy) / denominator.replace(0.0, np.nan)
    corr = corr.replace([np.inf, -np.inf], np.nan).dropna()
    if corr.empty:
        return float("nan"), float("nan")
    return float(np.average(corr, weights=n.loc[corr.index])), float(corr.mean())


def _recall_metrics(frame: pd.DataFrame, selected: np.ndarray, label: pd.Series) -> tuple[float, float, int]:
    positive = label.fillna(False).to_numpy(bool)
    if not positive.any():
        return float("nan"), float("nan"), 0
    work = pd.DataFrame({
        "timestamp": frame["__decision_ts__"].to_numpy(),
        "positive": positive,
        "selected_positive": selected & positive,
    })
    total = int(work["positive"].sum())
    row = float(work["selected_positive"].sum() / total)
    grouped = work.groupby("timestamp", sort=False).sum(numeric_only=True)
    grouped = grouped.loc[grouped["positive"].gt(0)]
    equal = float((grouped["selected_positive"] / grouped["positive"]).mean())
    return row, equal, total


def attach_outcomes(scores: pd.DataFrame, *, source: Path, policy: Path) -> pd.DataFrame:
    """Attach labels only after every target-free B0/B1 route is frozen."""
    r3 = pd.read_parquet(source, columns=[
        "candidate_id", "r3_class", "r3_label_available_ts",
    ])
    policy_frame = pd.read_parquet(policy, columns=[
        "candidate_id", "policy_path_valid", "policy_net_bps",
        "policy_label_available_ts",
    ])
    if r3["candidate_id"].duplicated().any() or policy_frame["candidate_id"].duplicated().any():
        raise ValueError("source or policy contract has duplicate candidate identities")
    output = scores.merge(r3, on="candidate_id", how="left", validate="one_to_one")
    output = output.merge(policy_frame, on="candidate_id", how="left", validate="one_to_one")
    output["policy_path_valid"] = output["policy_path_valid"].fillna(False).astype(bool)
    output["policy_net_bps"] = pd.to_numeric(output["policy_net_bps"], errors="coerce")
    output["r3_label_available_ts"] = pd.to_datetime(output["r3_label_available_ts"], utc=True, errors="coerce")
    output["policy_label_available_ts"] = pd.to_datetime(output["policy_label_available_ts"], utc=True, errors="coerce")
    output["is_r3_clear"] = output["r3_class"].eq(2) & output["r3_label_available_ts"].notna()
    valid_policy = output["policy_path_valid"] & output["policy_net_bps"].notna()
    for threshold in (30, 50, 100, 200):
        output[f"policy_ge_{threshold}"] = valid_policy & output["policy_net_bps"].ge(float(threshold))
    outcome_order = output.loc[:, ["candidate_id", "__decision_ts__", "policy_net_bps"]].copy()
    outcome_order["__valid__"] = valid_policy.to_numpy(bool)
    outcome_order = outcome_order.loc[outcome_order["__valid__"]].sort_values(
        ["__decision_ts__", "policy_net_bps", "candidate_id"],
        ascending=[True, False, True], kind="stable",
    )
    outcome_order["__rank__"] = outcome_order.groupby("__decision_ts__", sort=False).cumcount()
    count = outcome_order.groupby("__decision_ts__", sort=False)["candidate_id"].transform("size")
    positive = outcome_order["policy_net_bps"].gt(0)
    for fraction, name in ((.20, "positive_top20"), (.10, "positive_top10")):
        keep = outcome_order["__rank__"].lt(np.ceil(fraction * count).astype(int)) & positive
        output = output.merge(
            outcome_order.loc[:, ["candidate_id"]].assign(**{name: keep.to_numpy(bool)}),
            on="candidate_id", how="left", validate="one_to_one",
        )
        output[name] = output[name].fillna(False).astype(bool)
    return output


def build_arms() -> tuple[Arm, ...]:
    arms = [Arm("B0_D2_route30", 0.0, .5, .30, "B0")]
    for beta in (0.0, .10, .25, .40):
        for alpha in (.25, .50, .75, 1.0):
            arms.append(Arm(f"B1_b{beta:g}_a{alpha:g}", beta, alpha, .30, "B1"))
    arms.extend([
        Arm("B2_D2_route35", 0.0, .5, .35, "B2"),
        Arm("B2_D2_route40", 0.0, .5, .40, "B2"),
    ])
    return tuple(arms)


def evaluate_arms(scored: pd.DataFrame, arms: Iterable[Arm]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return summary metrics and target-free route membership by arm."""
    work = scored.copy()
    work["B0_D2_score"] = pd.to_numeric(work["base_score"], errors="coerce")
    route_table = work.loc[:, ["candidate_id", "__decision_ts__"]].copy()
    summaries: list[dict[str, object]] = []
    for arm in arms:
        score_name = f"__score__{arm.name}"
        if arm.family == "B0" or arm.family == "B2":
            work[score_name] = work["B0_D2_score"]
        else:
            work[score_name] = b1_score(
                work["p_clear"], work["p_weak"], work["p_adverse"],
                beta_weak=arm.beta_weak, alpha_adverse=arm.alpha_adverse,
            )
        selected = timestamp_route(work, score_name, fraction=arm.route_fraction)
        route_table[arm.name] = selected
        for period, (start, end) in PERIODS.items():
            subset = work.loc[
                work["__decision_ts__"].ge(_utc(start))
                & work["__decision_ts__"].lt(_utc(end)),
            ].copy()
            selector = selected[subset.index.to_numpy()]
            row: dict[str, object] = {
                "arm": arm.name,
                "family": arm.family,
                "beta_weak": arm.beta_weak,
                "alpha_adverse": arm.alpha_adverse,
                "route_fraction": arm.route_fraction,
                "period": period,
                "candidate_rows": int(len(subset)),
                "decision_timestamps": int(subset["__decision_ts__"].nunique()),
                "routed_rows": int(selector.sum()),
                "routed_fraction": float(selector.mean()) if len(selector) else float("nan"),
            }
            valid = subset["policy_path_valid"] & subset["policy_net_bps"].notna()
            row["routed_policy_net_mean_bps"] = (
                float(subset.loc[selector & valid, "policy_net_bps"].mean())
                if (selector & valid).any() else float("nan")
            )
            score_column = score_name
            row["pooled_within_timestamp_rank_ic"], row["equal_timestamp_rank_ic"] = _rank_ic(subset, score_column)
            for label_name in ("is_r3_clear", "policy_ge_30", "policy_ge_50", "policy_ge_100", "policy_ge_200", "positive_top20", "positive_top10"):
                r, e, support = _recall_metrics(subset, selector, subset[label_name])
                row[f"row_recall__{label_name}"] = r
                row[f"equal_timestamp_recall__{label_name}"] = e
                row[f"support__{label_name}"] = support
            row["recall_composite"] = (
                .20 * row["row_recall__policy_ge_50"]
                + .30 * row["row_recall__policy_ge_100"]
                + .25 * row["row_recall__policy_ge_200"]
                + .15 * row["row_recall__positive_top20"]
                + .10 * row["row_recall__positive_top10"]
            )
            summaries.append(row)
    return pd.DataFrame(summaries), route_table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--control-root", type=Path, default=DEFAULT_CONTROL)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable output exists: {args.out_dir}")
    for path in (args.source, args.policy):
        if not path.is_file():
            raise FileNotFoundError(path)
    target_free, parity = reconstruct_b0(source=args.source, control_root=args.control_root)
    outcome = attach_outcomes(target_free, source=args.source, policy=args.policy)
    summary, routes = evaluate_arms(outcome, build_arms())
    b0_b1_equivalent = summary.loc[
        summary["arm"].eq("B1_b0_a0.5"),
        ["period", "recall_composite", "routed_policy_net_mean_bps", "pooled_within_timestamp_rank_ic"],
    ].merge(
        summary.loc[
            summary["arm"].eq("B0_D2_route30"),
            ["period", "recall_composite", "routed_policy_net_mean_bps", "pooled_within_timestamp_rank_ic"],
        ],
        on="period", suffixes=("__b1", "__b0"), validate="one_to_one",
    )
    b0_b1_equivalent["composite_delta"] = (
        b0_b1_equivalent["recall_composite__b1"] - b0_b1_equivalent["recall_composite__b0"]
    )
    if not np.allclose(b0_b1_equivalent["composite_delta"], 0.0, atol=1e-12, rtol=0.0):
        raise AssertionError("B1 beta=0 alpha=.5 must exactly reproduce B0 recall")
    args.out_dir.mkdir(parents=True)
    target_free.to_parquet(args.out_dir / "b0_target_free_reconstruction.parquet", index=False, compression="zstd")
    outcome.to_parquet(args.out_dir / "outcome_joined_recall_ledger.parquet", index=False, compression="zstd")
    routes.to_parquet(args.out_dir / "timestamp_route_membership.parquet", index=False, compression="zstd")
    parity.to_parquet(args.out_dir / "b0_block_parity.parquet", index=False)
    summary.to_parquet(args.out_dir / "base_recall_metrics.parquet", index=False)
    b0_b1_equivalent.to_parquet(args.out_dir / "b0_b1_equivalence.parquet", index=False)
    manifest = {
        "schema": "strict_r3_long_base_recall_funnel_v1",
        "purpose": "Stage 0 frozen B0 parity plus B0/B1/B2 base-recall screening; offline research only",
        "source": {"path": str(args.source), "sha256": _sha256(args.source)},
        "frozen_control": {"path": str(args.control_root), "score_blocks": int(len(parity))},
        "policy": {"path": str(args.policy), "sha256": _sha256(args.policy)},
        "target_free_scoring": {
            "base_probability_source": "persisted current-v5 monthly upstream bundles",
            "parity_tolerance": {"relative": 1e-4, "absolute": 1e-8},
            "outcomes_joined_after_scores_and_timestamp_routes": True,
        },
        "splits": PERIODS,
        "arms": [asdict(arm) for arm in build_arms()],
        "labels": {
            "r3_clear": "r3_class == 2 with resolved R3 label",
            "policy_thresholds": "canonical policy net bps after valid-path outcome join",
            "top_labels": "positive policy outcome in timestamp-local top 20% / 10% by realised policy net; diagnostic only",
        },
        "next_gate": "A B1 arm may only rebuild downstream maps/residuals after separate development and frozen-holdout advancement review; this producer performs no downstream fitting.",
    }
    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"event": "complete", "target_free_rows": int(len(target_free)), "metric_rows": int(len(summary)), "blocks": int(len(parity))}, sort_keys=True))


if __name__ == "__main__":
    main()
