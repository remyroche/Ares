#!/usr/bin/env python3
"""Select and validate at most 30 causal extra base fields for Strict-R3.

The broad F1--F5 screens are allowed to be wider than the final contract only
as a discovery device.  This offline producer turns a named broad source
family into a stable <=30-field contract using development-only fold
importance, then re-runs it under the same strict D2/28-day-reserve protocol
over the frozen Q4 and 2026 periods.  It never changes a live or canonical
model/bundle.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.strict_r3_canonical_v2 import _fit_medians, _numeric_matrix  # noqa: E402
from scripts.run_strict_r3_base_f1_session_funnel import (  # noqa: E402
    _base_params, _d2_weights, _diagnose, _feature_contract, _strict_train,
)
from scripts.run_strict_r3_base_f2_f3_context_funnel import (  # noqa: E402
    _load_source as _load_f2_f3_source,
)
from scripts.run_strict_r3_base_f4_f5_context_funnel import (  # noqa: E402
    _load_source as _load_f4_f5_source,
)
from scripts.run_strict_r3_base_recall_funnel import (  # noqa: E402
    BASE_ROUTE_FRACTION, DEFAULT_CONTROL, DEFAULT_SOURCE, PERIODS, _utc, timestamp_route,
)


DEFAULT_B0 = ROOT / "data_perp/artifacts/strict_r3_long_base_recall_funnel_2025dev_holdout_2026oos_20260822_v1"
DEV_END = pd.Timestamp("2025-10-01T00:00:00Z")
MAX_EXTRA_FIELDS = 30


def _candidate_extra_fields(frame: pd.DataFrame, family: str) -> tuple[str, ...]:
    prefix = family.strip().lower() + "_"
    fields = tuple(sorted(name for name in frame.columns if name.startswith(prefix)))
    if not fields:
        raise ValueError(f"no derived source fields match family {family!r}")
    return fields


def _load_source_for_family(
    source: Path,
    fields: tuple[str, ...],
    family: str,
) -> pd.DataFrame:
    """Materialise one predeclared target-free feature family.

    F2/F3 and F4/F5 have intentionally separate primitive source contracts.
    Routing through the corresponding source loader prevents a later broad
    feature screen from accidentally deriving a family with a different input
    population or a future-aware fallback.
    """

    if family in {"f2", "f3"}:
        return _load_f2_f3_source(source, fields)
    if family in {"f4", "f5"}:
        return _load_f4_f5_source(source, fields)
    raise ValueError(f"unsupported feature family: {family!r}")


def _development_blocks(control_root: Path) -> list[Path]:
    blocks: list[Path] = []
    for path in sorted(control_root.glob("bundles/block=*/upstream/monthly_upstream_bundle.joblib")):
        bundle = joblib.load(path)
        if _utc(bundle.cutoff) < DEV_END:
            blocks.append(path)
    if len(blocks) < 3:
        raise AssertionError("need at least three development folds for stability selection")
    return blocks


def _stable_selection(
    source: pd.DataFrame,
    b0: pd.DataFrame,
    control_root: Path,
    base_fields: tuple[str, ...],
    extras: tuple[str, ...],
) -> pd.DataFrame:
    """Return dev-only gain stability, with no held/Q4/2026 outcome access."""

    source_index = source.set_index("candidate_id", drop=False)
    observations: list[pd.DataFrame] = []
    contract = (*base_fields, *extras)
    for path in _development_blocks(control_root):
        block = path.parents[1].name
        bundle = joblib.load(path)
        cutoff = _utc(bundle.cutoff)
        held_ids = b0.loc[b0["control_block"].eq(block), "candidate_id"]
        if held_ids.empty:
            continue
        train = _strict_train(source, cutoff)
        held = source_index.loc[held_ids.to_numpy()].copy().reset_index(drop=True)
        train_cov = train.loc[:, extras].notna().mean()
        held_cov = held.loc[:, extras].notna().mean()
        if float(min(train_cov.min(), held_cov.min())) < .90:
            raise AssertionError(f"development coverage fails in {block}")
        weights, _ = _d2_weights(train)
        medians = _fit_medians(train, contract)
        model = lgb.LGBMClassifier(**_base_params(bundle)).fit(
            _numeric_matrix(train, contract, medians), train["r3_class"].astype(int), sample_weight=weights,
        )
        importance = pd.DataFrame({
            "feature": contract,
            "gain": model.booster_.feature_importance(importance_type="gain").astype(float),
            "split": model.booster_.feature_importance(importance_type="split").astype(float),
            "control_block": block,
        })
        observations.append(importance.loc[importance["feature"].isin(extras)])
    all_importance = pd.concat(observations, ignore_index=True)
    all_importance["gain_share"] = all_importance.groupby("control_block", sort=False)["gain"].transform(
        lambda x: x / max(float(x.sum()), 1e-12)
    )
    all_importance["rank"] = all_importance.groupby("control_block", sort=False)["gain"].rank(
        method="first", ascending=False,
    )
    folds = int(all_importance["control_block"].nunique())
    summary = all_importance.groupby("feature", sort=True).agg(
        median_gain_share=("gain_share", "median"), mean_gain_share=("gain_share", "mean"),
        median_rank=("rank", "median"), top30_folds=("rank", lambda x: int((x <= MAX_EXTRA_FIELDS).sum())),
        nonzero_folds=("gain", lambda x: int((x > 0.0).sum())),
    ).reset_index()
    summary["folds"] = folds
    summary["top30_fraction"] = summary["top30_folds"] / float(folds)
    summary["nonzero_fraction"] = summary["nonzero_folds"] / float(folds)
    # A stable field must be useful in at least half of independent dev folds.
    summary["stable"] = (summary["top30_fraction"] >= .50) & (summary["nonzero_fraction"] >= .50)
    summary = summary.sort_values(
        ["stable", "top30_fraction", "median_gain_share", "mean_gain_share", "feature"],
        ascending=[False, False, False, False, True], kind="stable",
    ).reset_index(drop=True)
    summary["selected"] = False
    stable = summary.index[summary["stable"]].tolist()
    chosen = stable[:MAX_EXTRA_FIELDS]
    if len(chosen) < MAX_EXTRA_FIELDS:
        # This is recorded explicitly rather than silently selecting an
        # unstable tail.  The final contract may contain fewer than 30 extras.
        chosen = stable
    summary.loc[chosen, "selected"] = True
    return summary


def _score_selected(
    source: pd.DataFrame,
    b0: pd.DataFrame,
    control_root: Path,
    base_fields: tuple[str, ...],
    selected: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    source_index = source.set_index("candidate_id", drop=False)
    contract = (*base_fields, *selected)
    rows: list[pd.DataFrame] = []
    audit: list[dict[str, object]] = []
    for path in sorted(control_root.glob("bundles/block=*/upstream/monthly_upstream_bundle.joblib")):
        block = path.parents[1].name
        bundle = joblib.load(path)
        cutoff = _utc(bundle.cutoff)
        held_ids = b0.loc[b0["control_block"].eq(block), "candidate_id"]
        if held_ids.empty:
            continue
        train = _strict_train(source, cutoff)
        held = source_index.loc[held_ids.to_numpy()].copy().reset_index(drop=True)
        if selected and float(min(train.loc[:, selected].notna().mean().min(), held.loc[:, selected].notna().mean().min())) < .90:
            raise AssertionError(f"selected feature coverage fails in {block}")
        weights, weight_audit = _d2_weights(train)
        medians = _fit_medians(train, contract)
        model = lgb.LGBMClassifier(**_base_params(bundle)).fit(
            _numeric_matrix(train, contract, medians), train["r3_class"].astype(int), sample_weight=weights,
        )
        proba = model.predict_proba(_numeric_matrix(held, contract, medians))
        output = held.loc[:, ["candidate_id", "__decision_ts__"]].copy()
        output["p_adverse"] = proba[:, 0]
        output["p_weak"] = proba[:, 1]
        output["p_clear"] = proba[:, 2]
        output["score"] = proba[:, 2] - .5 * proba[:, 0]
        output["control_block"] = block
        rows.append(output)
        audit.append({
            "control_block": block, "cutoff": cutoff.isoformat(), "train_rows": len(train), "held_rows": len(held),
            "reserve_start": (cutoff - pd.Timedelta(days=28)).isoformat(), "d2_weight_audit_json": json.dumps(weight_audit, sort_keys=True),
        })
    result = pd.concat(rows, ignore_index=True)
    if len(result) != len(b0) or result["candidate_id"].duplicated().any():
        raise AssertionError("selected-contract scoring changed B0 candidate identities")
    return result, pd.DataFrame(audit)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--family", choices=("f2", "f3", "f4", "f5"), required=True,
        help="Predeclared target-free feature family to stability-select.",
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--control-root", type=Path, default=DEFAULT_CONTROL)
    parser.add_argument("--b0-root", type=Path, default=DEFAULT_B0)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable output exists: {args.out_dir}")
    base_fields = _feature_contract(args.control_root)
    source = _load_source_for_family(args.source, base_fields, args.family)
    extras = _candidate_extra_fields(source, args.family)
    b0 = pd.read_parquet(args.b0_root / "b0_target_free_reconstruction.parquet")
    b0["__decision_ts__"] = pd.to_datetime(b0["__decision_ts__"], utc=True, errors="raise")
    stability = _stable_selection(source, b0, args.control_root, base_fields, extras)
    selected = tuple(stability.loc[stability["selected"], "feature"])
    if len(selected) > MAX_EXTRA_FIELDS:
        raise AssertionError("feature selection exceeded 30-field final cap")
    prediction, audit = _score_selected(source, b0, args.control_root, base_fields, selected)
    # Outcome diagnostics happen strictly after target-free scores/routes.
    outcome = pd.read_parquet(args.b0_root / "outcome_joined_recall_ledger.parquet")
    scored = outcome.merge(prediction, on=["candidate_id", "__decision_ts__", "control_block"], how="inner", validate="one_to_one")
    if len(scored) != len(b0):
        raise AssertionError("post-score outcome join changed candidate identities")
    scored["B0_route"] = timestamp_route(scored, "base_score", fraction=BASE_ROUTE_FRACTION)
    scored["selected_route"] = timestamp_route(scored, "score", fraction=BASE_ROUTE_FRACTION)
    metrics: list[dict[str, object]] = []
    evaluation_slices: list[tuple[str, pd.DataFrame]] = []
    for label, (start, end) in PERIODS.items():
        subset = scored.loc[scored["__decision_ts__"].ge(_utc(start)) & scored["__decision_ts__"].lt(_utc(end))].copy()
        evaluation_slices.append((label, subset))
    evaluation_slices.extend(
        (str(quarter), group.copy())
        for quarter, group in scored.groupby(
            scored["__decision_ts__"].dt.to_period("Q"), sort=True,
        )
        if quarter >= pd.Period("2025Q4", freq="Q")
    )
    for label, subset in evaluation_slices:
        for arm, score, route in (("B0", "base_score", "B0_route"), (f"{args.family}_selected", "score", "selected_route")):
            row = _diagnose(subset, subset[route].to_numpy(bool), score, label)
            row["arm"] = arm
            metrics.append(row)
    metrics_frame = pd.DataFrame(metrics)
    selected_name = f"{args.family}_selected"
    wide = metrics_frame.pivot(index="label", columns="arm", values=["recall_composite", "routed_policy_net_mean_bps", "rank_ic"])
    gates: list[dict[str, object]] = []
    for label in ("frozen_holdout_2025q4", "frozen_oos_2026jan_jul"):
        gates.append({
            "period": label,
            "relative_recall_gain": float(wide.loc[label, ("recall_composite", selected_name)] / wide.loc[label, ("recall_composite", "B0")] - 1.0),
            "mean_policy_net_delta_bps": float(wide.loc[label, ("routed_policy_net_mean_bps", selected_name)] - wide.loc[label, ("routed_policy_net_mean_bps", "B0")]),
            "rank_ic_delta": float(wide.loc[label, ("rank_ic", selected_name)] - wide.loc[label, ("rank_ic", "B0")]),
        })
    gate = pd.DataFrame(gates)
    quarterly = metrics_frame.loc[
        metrics_frame["label"].str.match(r"^20\d\dQ[1-4]$", na=False)
    ].pivot(index="label", columns="arm", values="recall_policy_ge_100")
    policy_ge_100_no_quarterly_decline = bool(
        (quarterly[selected_name] >= quarterly["B0"]).all()
    )
    advances = bool(
        len(selected) > 0
        and gate["relative_recall_gain"].ge(.02).all()
        and gate["mean_policy_net_delta_bps"].ge(-5.0).all()
        and gate["rank_ic_delta"].ge(-.005).all()
        and policy_ge_100_no_quarterly_decline
    )
    args.out_dir.mkdir(parents=True)
    stability.to_parquet(args.out_dir / "feature_stability.parquet", index=False)
    prediction.to_parquet(args.out_dir / "target_free_selected_scores.parquet", index=False, compression="zstd")
    scored.to_parquet(args.out_dir / "outcome_joined_selected_audit.parquet", index=False, compression="zstd")
    audit.to_parquet(args.out_dir / "selected_block_training_audit.parquet", index=False)
    metrics_frame.to_parquet(args.out_dir / "selected_base_metrics.parquet", index=False)
    gate.to_parquet(args.out_dir / "selected_advancement_gate.parquet", index=False)
    (args.out_dir / "run_manifest.json").write_text(json.dumps({
        "schema": "strict_r3_base_feature_stability_selection_v1",
        "scope": "offline :00-only base research; no residual, MC1, admission, portfolio, execution, live, or canonical mutation",
        "family": args.family, "candidate_extra_fields": len(extras), "selected_extra_fields": list(selected),
        "selection": "development-only median gain-share and top30-fold stability >=50%; no unstable fill",
        "maximum_selected_extras": MAX_EXTRA_FIELDS, "advance_to_downstream_rebuild": advances,
        "policy_ge_100_no_quarterly_decline": policy_ge_100_no_quarterly_decline,
        "causality": "D2 base fits only labels resolved before each preceding 28-day reserve; scores and timestamp-local routes are target-free before outcome diagnostics",
    }, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"event": "complete", "family": args.family, "selected": len(selected), "advances": advances}, sort_keys=True))


if __name__ == "__main__":
    main()
