#!/usr/bin/env python3
"""Three-month-block trust/distribution sizing ablations on strict-R3.

2025 is the development/OOF selection era.  The top three specifications per
pipeline are selected without consulting 2026; those frozen names can then be
confirmed on 2026 with identical causal admission and portfolio constraints.

Candidate ranking and causal EV admission remain frozen.  Trust models affect
relative position size only.  Raw K9 membership/archetype columns are excluded
because rolling Geometry/K9 bundle identities do not share stable semantics.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Iterable, Sequence

# On Apple Silicon, loading PyTorch after LightGBM/scipy's native runtimes can
# terminate the process inside the allocator/OpenMP runtime.  The MLP shard may
# opt into an early import so torch owns its runtime before Ares imports the
# larger feature/model stack.  Other pipelines remain free of the dependency.
if os.environ.get("ARES_TRUST_PRELOAD_TORCH") == "1":  # pragma: no cover - runtime guard
    import torch as _torch  # noqa: F401

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.stage_i_causal_admission import (  # noqa: E402
    Causal21dAdmissionSpec,
    apply_causal_21d_side_admission,
)
from extreme_price_movements.trust_sizing_ablation import (  # noqa: E402
    ParentExpectation,
    TrustModelSpec,
    TrustPrediction,
    causal_size_multiplier,
    catalogue,
    fit_trust_model,
    sizing_quality,
)
from scripts.replay_strict_r3_forward_portfolio import _auction_candidates  # noqa: E402
from scripts.replay_strict_r3_policy_portfolio_2025_2026 import _run  # noqa: E402
from scripts.run_strict_r3_c3_window_cadence_ablation import (  # noqa: E402
    _causal_reliability_context,
)


SEED = 20260810
TAILS = (0.005, 0.01, 0.02, 0.05, 0.10)
INPUTS = {
    2025: ROOT / "data_perp/artifacts/strict_r3_top30_k9_temperature_fullcap_long_2025_janjul_20260810_v1/predictions.parquet",
    2026: ROOT / "data_perp/artifacts/strict_r3_top30_k9_temperature_fullcap_long_2026_janjul_20260810_v1/predictions.parquet",
}
PERIODS = {
    2025: (pd.Timestamp("2025-01-01", tz="UTC"), pd.Timestamp("2025-08-01", tz="UTC")),
    2026: (pd.Timestamp("2026-01-01", tz="UTC"), pd.Timestamp("2026-08-01", tz="UTC")),
}
BASE_COLUMNS = (
    "candidate_id", "__ts__", "__decision_ts__", "__symbol__", "side_name",
    "base_score", "base_rank", "base_anchor_bps", "consensus_rank", "final_score",
    "correctness_raw", "correctness_rank", "correctness_gate_active",
    "raw_correctness_demote", "severe200_probability",
    "k9_entropy", "k9_top2_margin", "k9_ood_distance",
    "k9_path_support_effective_28d", "k9_path_support_adequate_fraction",
    "k9_model_ood_marginal", "k9_model_drift_psi",
    "leaf_support_effective", "leaf_support_p05", "leaf_support_p50",
    "leaf_support_p95", "leaf_support_adequate_fraction",
    "leaf_support_leaf_coverage", "leaf_ood_marginal", "leaf_ood_joint",
    "geometry_bundle_sha256", "geometry_bundle_id", "model_cutoff",
    "policy_path_valid", "policy_gross_bps", "policy_net_bps",
    "policy_exit_bar_15m", "policy_exit_reason", "policy_entry_price",
    "policy_exit_price", "policy_label_available_ts", "policy_outcome_source",
    "is_admission_warmup",
)
INVARIANT_TRUST_FIELDS = (
    "base_score", "base_rank", "base_anchor_bps", "consensus_rank", "final_score",
    "correctness_raw", "correctness_rank", "severe200_probability",
    "k9_entropy", "k9_top2_margin", "k9_ood_distance",
    "k9_path_support_effective_28d", "k9_path_support_adequate_fraction",
    "k9_model_ood_marginal", "k9_model_drift_psi",
    "leaf_support_effective", "leaf_support_p05", "leaf_support_p50",
    "leaf_support_p95", "leaf_support_adequate_fraction",
    "leaf_support_leaf_coverage", "leaf_ood_marginal", "leaf_ood_joint",
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    frame = pd.read_parquet(path, columns=list(BASE_COLUMNS))
    frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True)
    frame["policy_label_available_ts"] = pd.to_datetime(
        frame["policy_label_available_ts"], utc=True, errors="coerce",
    )
    frame = frame.sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)
    if frame["candidate_id"].duplicated().any():
        raise ValueError("source prediction ledger contains duplicate candidate IDs")
    admitted, admission_audit = apply_causal_21d_side_admission(
        frame,
        score_column="final_score",
        net_column="policy_net_bps",
        decision_column="__decision_ts__",
        label_available_column="policy_label_available_ts",
        identity_column="candidate_id",
        spec=Causal21dAdmissionSpec(mode="hierarchical_tail_side_shrinkage_v2"),
    )
    mapped = pd.to_numeric(
        admitted["causal_21d_side_expected_net_bps"], errors="coerce",
    )
    admitted["raw_expected_bps"] = mapped
    admitted["mapped_ev_available"] = mapped.notna()
    context, groups = _causal_reliability_context(admitted)
    context.index = admitted.index
    admitted = pd.concat([admitted, context], axis=1)
    fields = list(INVARIANT_TRUST_FIELDS)
    for group in ("cross_model", "global_recent", "covariance"):
        fields.extend(groups[group])
    fields = list(dict.fromkeys(field for field in fields if field in admitted))
    coverage = admitted.loc[:, fields].notna().mean()
    variance = admitted.loc[:, fields].apply(pd.to_numeric, errors="coerce").var()
    eligible = [field for field in fields if coverage[field] >= 0.90 and variance[field] > 1e-12]
    if len(eligible) < 20:
        raise ValueError(f"trust feature contract is too small: {len(eligible)}")
    audit = {
        "source_rows": len(frame),
        "mapped_ev_available_rows": int(admitted["mapped_ev_available"].sum()),
        "feature_candidates": len(fields),
        "eligible_features": len(eligible),
        "eligible_feature_names": eligible,
        "raw_k9_memberships_used": False,
        "geometry_bundle_count": int(admitted["geometry_bundle_sha256"].nunique()),
        "cluster_semantics_rule": (
            "raw cluster/archetype fields prohibited across bundles; only invariant "
            "entropy/margin/OOD/support/drift summaries used"
        ),
        "admission_audit_rows": len(admission_audit),
    }
    return admitted, eligible, audit


def _blocks(year: int) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    start, end = PERIODS[year]
    result: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    cursor = start
    while cursor < end:
        held_end = min(cursor + pd.DateOffset(months=3), end)
        result.append((cursor, held_end))
        cursor = held_end
    return result


def _sample_equal_month(frame: pd.DataFrame, cap: int) -> pd.DataFrame:
    if len(frame) <= cap:
        return frame.copy()
    work = frame.copy()
    month = work["__decision_ts__"].dt.to_period("M").astype(str)
    months = sorted(month.unique())
    quota = max(1, cap // max(len(months), 1))
    rng = np.random.default_rng(SEED)
    chosen: list[np.ndarray] = []
    for token in months:
        index = np.flatnonzero(month.eq(token).to_numpy())
        if len(index) > quota:
            index = np.sort(rng.choice(index, size=quota, replace=False))
        chosen.append(index)
    selected = np.concatenate(chosen)
    if len(selected) > cap:
        selected = np.sort(rng.choice(selected, size=cap, replace=False))
    return work.iloc[selected].sort_values(["__decision_ts__", "candidate_id"], kind="stable")


def _control_prediction(train: pd.DataFrame, score: pd.DataFrame) -> tuple[TrustPrediction, TrustPrediction]:
    realised = pd.to_numeric(train["policy_net_bps"], errors="coerce").to_numpy(float)
    expected = pd.to_numeric(train["raw_expected_bps"], errors="coerce").to_numpy(float)
    expected_score = pd.to_numeric(score["raw_expected_bps"], errors="coerce").to_numpy(float)
    sigma = max(float(np.sqrt(np.mean(np.clip(realised - expected, -2_000, 2_000) ** 2))), 25.0)

    def make(mean: np.ndarray, rows: int) -> TrustPrediction:
        predictive = np.full(rows, sigma)
        support = np.full(rows, len(train), dtype=float)
        q = student_t_quantile(sigma)
        return TrustPrediction(
            mean, np.ones(rows), predictive, predictive / np.sqrt(max(len(train), 1)),
            mean - q, mean, mean + q,
            1.0 - pd.Series((0.0 - mean) / predictive).map(
                lambda value: _student_cdf(float(value))
            ).to_numpy(),
            pd.Series((-200.0 - mean) / predictive).map(
                lambda value: _student_cdf(float(value))
            ).to_numpy(),
            support,
        )

    return make(expected, len(train)), make(expected_score, len(score))


def _student_cdf(value: float) -> float:
    from scipy.stats import t as student
    return float(student.cdf(value, df=5.0))


def student_t_quantile(scale: float) -> float:
    from scipy.stats import t as student
    return float(student.ppf(0.90, df=5.0) * scale)


def _period_tail_metrics(
    frame: pd.DataFrame,
    *,
    arm: str,
    period_kind: str,
) -> pd.DataFrame:
    if period_kind == "global":
        groups: Iterable[tuple[str, pd.DataFrame]] = [("all", frame)]
    elif period_kind == "month":
        token = frame["__decision_ts__"].dt.to_period("M").astype(str)
        groups = frame.groupby(token, sort=True)
    elif period_kind == "week":
        token = frame["__decision_ts__"].dt.to_period("W-SUN").astype(str)
        groups = frame.groupby(token, sort=True)
    else:
        raise ValueError(period_kind)
    rows: list[dict[str, Any]] = []
    for period, block in groups:
        score_population = block.loc[np.isfinite(pd.to_numeric(block["final_score"], errors="coerce"))]
        for tail in TAILS:
            count = max(1, int(math.ceil(tail * len(score_population))))
            selected = score_population.nlargest(count, "final_score", keep="first")
            valid = selected.loc[
                selected["policy_path_valid"].fillna(False).astype(bool)
                & np.isfinite(pd.to_numeric(selected["policy_net_bps"], errors="coerce"))
            ].copy()
            weight = pd.to_numeric(valid["trust_size_multiplier"], errors="coerce").fillna(1.0).clip(0.0)
            net = pd.to_numeric(valid["policy_net_bps"], errors="coerce")
            gross = pd.to_numeric(valid["policy_gross_bps"], errors="coerce")
            denominator = float(weight.sum())
            weighted_net = float(np.average(net, weights=weight)) if denominator > 0 else np.nan
            weighted_gross = float(np.average(gross, weights=weight)) if denominator > 0 else np.nan
            rows.append(
                {
                    "arm": arm,
                    "period_kind": period_kind,
                    "period": str(period),
                    "tail": tail,
                    "population_rows": len(score_population),
                    "selected_score_rows": len(selected),
                    "valid_outcomes": len(valid),
                    "outcome_coverage": len(valid) / max(len(selected), 1),
                    "unweighted_net_bps_per_trade": float(net.mean()),
                    "exposure_weighted_net_bps": weighted_net,
                    "exposure_weighted_gross_bps": weighted_gross,
                    "sizing_uplift_bps": weighted_net - float(net.mean()),
                    "mean_size_multiplier": float(weight.mean()),
                    "positive_rate": float(net.gt(0).mean()),
                }
            )
    return pd.DataFrame(rows)


def _portability(values: Sequence[float]) -> tuple[float, float, float, float]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if not len(x):
        return -np.inf, np.nan, np.nan, np.nan
    median = float(np.median(x))
    mad = float(np.median(np.abs(x - median)))
    worst = float(np.min(x))
    return median - 0.5 * mad - max(0.0, -worst), median, mad, worst


def _stability(monthly: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (arm, tail), block in monthly.groupby(["arm", "tail"], sort=True):
        score, median, mad, worst = _portability(block["exposure_weighted_net_bps"])
        rows.append(
            {
                "arm": arm, "tail": tail, "portability": score,
                "month_median_bps": median, "month_mad_bps": mad,
                "worst_month_bps": worst,
                "positive_months": int(block["exposure_weighted_net_bps"].gt(0).sum()),
                "months": len(block),
            }
        )
    return pd.DataFrame(rows)


def _selection(global_metrics: pd.DataFrame, stability: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for arm, block in global_metrics.groupby("arm", sort=True):
        values = block.set_index("tail")["exposure_weighted_net_bps"]
        weighted_tail = (
            float(values.get(0.01, np.nan))
            + 0.5 * float(values.get(0.02, np.nan))
            + 0.2 * float(values.get(0.05, np.nan))
            + 0.1 * float(values.get(0.10, np.nan))
        )
        stable = stability.loc[
            stability["arm"].eq(arm) & stability["tail"].isin([0.01, 0.02, 0.05])
        ]
        portability = float(stable["portability"].mean())
        worst = float(stable["worst_month_bps"].min())
        score = weighted_tail + 0.25 * portability - max(0.0, -worst)
        rows.append(
            {
                "arm": arm, "weighted_tail_score": weighted_tail,
                "mean_portability_top1_2_5": portability,
                "worst_month_top1_2_5": worst,
                "selection_score": score,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["selection_score", "mean_portability_top1_2_5", "weighted_tail_score"],
        ascending=False, kind="stable",
    )


def _weekly_portfolio(decisions: pd.DataFrame, arm: str) -> pd.DataFrame:
    accepted = decisions.loc[decisions["accepted"].fillna(False).astype(bool)].copy()
    accepted["week"] = pd.to_datetime(accepted["timestamp"], utc=True).dt.to_period("W-SUN").astype(str)
    accepted["net_bps"] = pd.to_numeric(accepted["position_net_return"], errors="coerce") * 10_000.0
    accepted["gross_bps"] = pd.to_numeric(accepted["position_gross_return"], errors="coerce") * 10_000.0
    return accepted.groupby("week", as_index=False).agg(
        trades=("net_bps", "size"), net_bps_per_trade=("net_bps", "mean"),
        gross_bps_per_trade=("gross_bps", "mean"), net_sum_bps=("net_bps", "sum"),
        positive_rate=("net_bps", lambda value: float((value > 0).mean())),
    ).assign(arm=arm)


def _portfolio(
    source: pd.DataFrame,
    output: pd.DataFrame,
    *,
    arm: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    size = output.loc[:, ["candidate_id", "trust_size_multiplier"]]
    evaluation = source.loc[source["__decision_ts__"].ge(start) & source["__decision_ts__"].lt(end)].copy()
    evaluation = evaluation.merge(size, on="candidate_id", how="left", validate="one_to_one")
    evaluation["trust_size_multiplier"] = evaluation["trust_size_multiplier"].fillna(1.0)
    try:
        candidates = _auction_candidates(evaluation, strategy_prefix="strict_r3_trust_sizing")
    except ValueError:
        return (
            pd.DataFrame([{"arm": arm, "accepted_trades": 0, "net_bps_per_trade": np.nan, "max_drawdown": np.nan}]),
            pd.DataFrame(), pd.DataFrame(),
        )
    candidates = candidates.merge(size, on="candidate_id", how="left", validate="one_to_one")
    candidates["portfolio_size_multiplier"] = candidates["trust_size_multiplier"].fillna(1.0)
    decisions, equity, monthly, summary = _run(
        candidates, 0.0, arm, initial_wallet=1_000.0,
        perp_leverage=7.0, margin_slot_wallet_fraction=0.10,
    )
    replay = summary.get("replay_metric_summary", {})
    if isinstance(replay, str):
        replay = json.loads(replay)
    summary_row = pd.DataFrame(
        [{
            "arm": arm,
            "accepted_trades": int(summary["accepted_trades"]),
            "trades_per_day": float(summary["accepted_trades"] / max((end - start).days, 1)),
            "gross_bps_per_trade": float(summary["gross_bps_per_trade"]),
            "net_bps_per_trade": float(summary["net_bps_per_trade"]),
            "positive_rate": float(summary["positive_rate"]),
            "portfolio_net_pnl": float(summary.get("portfolio_net_pnl", np.nan)),
            "final_wallet": float(summary.get("final_wallet", np.nan)),
            "max_drawdown": float(replay.get("max_drawdown", np.nan)),
        }]
    )
    return summary_row, monthly.assign(arm=arm), _weekly_portfolio(decisions, arm)


def _run_pipeline(
    frame: pd.DataFrame,
    fields: Sequence[str],
    specs: Sequence[TrustModelSpec],
    *,
    year: int,
    train_cap: int,
) -> tuple[
    dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame, pd.DataFrame,
    pd.DataFrame, pd.DataFrame, pd.DataFrame,
]:
    start, end = PERIODS[year]
    base_parts: list[pd.DataFrame] = []
    arm_parts: dict[str, list[pd.DataFrame]] = {spec.name: [] for spec in specs}
    fold_audits: list[dict[str, Any]] = []
    edge_rows: list[dict[str, Any]] = []
    for cutoff, held_end in _blocks(year):
        train_start = cutoff - pd.DateOffset(months=3)
        train_all = frame.loc[
            frame["__decision_ts__"].ge(train_start)
            & frame["__decision_ts__"].lt(cutoff)
            & frame["policy_label_available_ts"].lt(cutoff)
            & frame["policy_path_valid"].fillna(False).astype(bool)
            & frame["mapped_ev_available"].astype(bool)
            & np.isfinite(pd.to_numeric(frame["policy_net_bps"], errors="coerce"))
        ].copy()
        held = frame.loc[
            frame["__decision_ts__"].ge(cutoff) & frame["__decision_ts__"].lt(held_end)
        ].copy()
        if len(train_all) < 2_000 or held.empty:
            raise ValueError(
                f"insufficient three-month train/held support at {cutoff}: "
                f"{len(train_all)}/{len(held)}"
            )
        parent = ParentExpectation.fit(
            train_all["final_score"], train_all["policy_net_bps"],
        )
        train_all["parent_expected_bps"] = parent.predict(train_all["final_score"])
        held["parent_expected_bps"] = parent.predict(held["final_score"])
        train_floor = float(pd.to_numeric(train_all["final_score"], errors="coerce").quantile(0.70))
        train = train_all.loc[pd.to_numeric(train_all["final_score"], errors="coerce").ge(train_floor)].copy()
        train = _sample_equal_month(train, train_cap)
        held["trust_gate_active"] = (
            held["mapped_ev_available"].astype(bool)
            & pd.to_numeric(held["final_score"], errors="coerce").ge(train_floor)
        )
        base_parts.append(
            held.loc[:, [
                "candidate_id", "__decision_ts__", "__symbol__", "final_score",
                "policy_path_valid", "policy_gross_bps", "policy_net_bps",
                "policy_exit_reason", "geometry_bundle_sha256", "raw_expected_bps",
                "parent_expected_bps", "trust_gate_active",
            ]].copy()
        )
        for spec in specs:
            print(json.dumps({
                "event": "fit_start", "year": year, "pipeline": spec.pipeline,
                "arm": spec.name, "train_start": train_start.isoformat(),
                "cutoff": cutoff.isoformat(), "held_end": held_end.isoformat(),
                "train_rows": len(train), "held_rows": len(held),
            }), flush=True)
            if spec.sizing_mode == "equal" and spec.name.endswith("equal_control"):
                train_prediction, held_prediction = _control_prediction(train, held)
                audit: dict[str, Any] = {
                    "edge_count": 0, "selected_edges": [], "control": True,
                    "raw_k9_memberships_used": False,
                }
            else:
                train_prediction, held_prediction, audit = fit_trust_model(
                    train, held, fields, spec,
                )
            train_quality = sizing_quality(train_prediction, train, spec.sizing_mode)
            held_quality = sizing_quality(held_prediction, held, spec.sizing_mode)
            multiplier = causal_size_multiplier(train_quality, held_quality)
            multiplier = np.where(held["trust_gate_active"].to_numpy(bool), multiplier, 1.0)
            output = held_prediction.as_frame()
            output.insert(0, "candidate_id", held["candidate_id"].to_numpy())
            output["trust_size_multiplier"] = multiplier.astype(np.float32)
            output["arm"] = spec.name
            arm_parts[spec.name].append(output)
            fold_audits.append(
                {
                    "year": year, "arm": spec.name,
                    "train_start": train_start, "train_end_exclusive": cutoff,
                    "held_start": cutoff, "held_end_exclusive": held_end,
                    "train_rows_before_top30": len(train_all), "train_rows": len(train),
                    "held_rows": len(held), "train_score_floor": train_floor,
                    "held_active_fraction": float(held["trust_gate_active"].mean()),
                    "train_geometry_bundles": int(train["geometry_bundle_sha256"].nunique()),
                    "held_geometry_bundles": int(held["geometry_bundle_sha256"].nunique()),
                    "raw_k9_memberships_used": False,
                    "geometry_semantics_contract": "bundle-invariant aggregate state only",
                    **{key: value for key, value in audit.items() if key != "selected_edges"},
                }
            )
            for edge in audit.get("selected_edges", []):
                edge_rows.append(
                    {
                        "year": year, "arm": spec.name, "cutoff": cutoff,
                        **edge,
                    }
                )
            print(json.dumps({
                "event": "fit_complete", "year": year, "arm": spec.name,
                "cutoff": cutoff.isoformat(), "edge_count": audit.get("edge_count", 0),
            }), flush=True)
    base = pd.concat(base_parts, ignore_index=True)
    outputs: dict[str, pd.DataFrame] = {}
    global_parts: list[pd.DataFrame] = []
    month_parts: list[pd.DataFrame] = []
    week_parts: list[pd.DataFrame] = []
    for spec in specs:
        prediction = pd.concat(arm_parts[spec.name], ignore_index=True)
        result = base.merge(prediction, on="candidate_id", how="left", validate="one_to_one")
        outputs[spec.name] = result
        global_parts.append(_period_tail_metrics(result, arm=spec.name, period_kind="global"))
        month_parts.append(_period_tail_metrics(result, arm=spec.name, period_kind="month"))
        week_parts.append(_period_tail_metrics(result, arm=spec.name, period_kind="week"))
    global_metrics = pd.concat(global_parts, ignore_index=True)
    monthly = pd.concat(month_parts, ignore_index=True)
    weekly = pd.concat(week_parts, ignore_index=True)
    stability = _stability(monthly)
    return (
        outputs, global_metrics, monthly, weekly, stability,
        pd.DataFrame(fold_audits), pd.DataFrame(edge_rows),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline", choices=("bayesian", "gam", "nonlinear"), required=True)
    parser.add_argument("--year", type=int, choices=(2025, 2026), required=True)
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--selected-specs", type=Path)
    parser.add_argument(
        "--specs",
        help=(
            "Optional comma-separated development subset. This is intended for "
            "memory-isolated family shards; 2026 instead requires --selected-specs."
        ),
    )
    parser.add_argument("--train-cap", type=int, default=60_000)
    parser.add_argument("--ngboost-path", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable output already exists: {args.out_dir}")
    if args.ngboost_path is not None:
        # Core numerical packages have already been imported above.  This path
        # therefore supplies NGBoost only and cannot shadow the active NumPy /
        # pandas / sklearn runtime.
        sys.path.insert(0, str(args.ngboost_path))
    source_path = args.input or INPUTS[args.year]
    specs = catalogue()[args.pipeline]
    frozen_selected: list[str] | None = None
    if args.specs:
        if args.year != 2025:
            parser.error("--specs is development-only; use --selected-specs for 2026")
        requested_names = {value.strip() for value in args.specs.split(",") if value.strip()}
        known_names = {spec.name for spec in specs}
        unknown_names = requested_names - known_names
        if unknown_names:
            parser.error(f"unknown --specs values: {sorted(unknown_names)}")
        specs = [spec for spec in specs if spec.name in requested_names]
        if not specs:
            parser.error("--specs selected no arms")
    if args.year == 2026:
        if args.selected_specs is None:
            parser.error("2026 confirmation requires --selected-specs from the 2025 run")
        selected_payload = json.loads(args.selected_specs.read_text())
        frozen_selected = list(selected_payload["selected_top3"])
        selected_names = set(frozen_selected)
        control = next(spec for spec in specs if spec.name.endswith("_equal_control"))
        finalists = [spec for spec in specs if spec.name in selected_names]
        if len(finalists) != 3:
            raise ValueError(f"expected three frozen finalists, found {len(finalists)}")
        # The matched equal-size control is evaluated alongside, but is never
        # eligible to replace a finalist or enter the top-three prediction file.
        specs = [control, *finalists]
    frame, fields, source_audit = _load(source_path)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    (
        outputs, global_metrics, monthly, weekly, stability, fold_audit, edge_audit,
    ) = _run_pipeline(
        frame, fields, specs, year=args.year, train_cap=int(args.train_cap),
    )
    selection = _selection(global_metrics, stability)
    if args.year == 2025:
        selected = selection.head(3)["arm"].tolist()
    else:
        assert frozen_selected is not None
        selected = frozen_selected
    controls = [spec.name for spec in specs if spec.name.endswith("_equal_control")]
    portfolio_arms = [*controls, *selected]
    portfolio_summary: list[pd.DataFrame] = []
    portfolio_monthly: list[pd.DataFrame] = []
    portfolio_weekly: list[pd.DataFrame] = []
    start, end = PERIODS[args.year]
    for name in portfolio_arms:
        summary, month, week = _portfolio(
            frame, outputs[name], arm=name, start=start, end=end,
        )
        portfolio_summary.append(summary)
        portfolio_monthly.append(month)
        portfolio_weekly.append(week)
    top3_predictions = pd.concat(
        [outputs[name].assign(selection_rank=index + 1) for index, name in enumerate(selected)],
        ignore_index=True,
    )
    global_metrics.to_parquet(args.out_dir / "metrics_global.parquet", index=False)
    monthly.to_parquet(args.out_dir / "metrics_monthly.parquet", index=False)
    weekly.to_parquet(args.out_dir / "metrics_weekly.parquet", index=False)
    stability.to_parquet(args.out_dir / "stability.parquet", index=False)
    selection.to_parquet(args.out_dir / "selection_metrics.parquet", index=False)
    fold_audit.to_parquet(args.out_dir / "fold_audit.parquet", index=False)
    edge_audit.to_parquet(args.out_dir / "cmi_edge_audit.parquet", index=False)
    pd.concat(portfolio_summary, ignore_index=True).to_parquet(
        args.out_dir / "portfolio_summary.parquet", index=False,
    )
    pd.concat(portfolio_monthly, ignore_index=True).to_parquet(
        args.out_dir / "portfolio_monthly.parquet", index=False,
    )
    pd.concat(portfolio_weekly, ignore_index=True).to_parquet(
        args.out_dir / "portfolio_weekly.parquet", index=False,
    )
    top3_predictions.to_parquet(
        args.out_dir / "top3_predictions.parquet", index=False, compression="zstd",
    )
    (args.out_dir / "selected_top3.json").write_text(
        json.dumps(
            {
                "pipeline": args.pipeline, "selection_year": 2025,
                "selected_top3": selected,
                "selection_formula": (
                    "EV1 + 0.5*EV2 + 0.2*EV5 + 0.1*EV10 + "
                    "0.25*mean_portability_top1_2_5 - negative_worst_month_penalty"
                ),
            },
            indent=2,
        )
        + "\n"
    )
    manifest = {
        "schema": "strict_r3_three_month_trust_sizing_ablation_v1",
        "pipeline": args.pipeline,
        "year": args.year,
        "evaluation_role": (
            "development_oof_model_selection" if args.year == 2025
            else "frozen_2026_confirmation_not_used_for_selection"
        ),
        "source": str(source_path), "source_sha256": _sha(source_path),
        "three_month_training_blocks": True,
        "three_month_evaluation_blocks": True,
        "purge_embargo": "policy_label_available_ts < held block start; 12h outcome horizon",
        "ranking": "frozen canonical final_score; one pooled-global ranking",
        "trust_integration": "relative size only; no reranking",
        "admission": "unchanged causal hierarchical 21/42/84-day EV >= +50 bps",
        "portfolio": "8 concurrent, 2 new per bar, 1 per asset, 80% margin, 7x leverage",
        "cost": "selected SimplePolicyOptimiser policy net; 100 bps exactly once",
        "geometry_semantics": source_audit["cluster_semantics_rule"],
        "raw_k9_memberships_used": False,
        "train_cap": int(args.train_cap),
        "specs": [asdict(spec) for spec in specs],
        "selected_top3": selected,
        "source_audit": source_audit,
        "ngboost_runtime": str(args.ngboost_path) if args.ngboost_path else None,
        "seed": SEED,
    }
    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    print(json.dumps({"event": "complete", "output": str(args.out_dir), **manifest}, default=str))


if __name__ == "__main__":
    main()
