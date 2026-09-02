#!/usr/bin/env python3
"""Offline F1 session/calendar base-feature ablation for long Strict-R3.

The challenger appends the existing 23 timestamp-only session/calendar fields
to the frozen 120-field B0 contract.  It uses the exact frozen multiclass base
model geometry and strict 28-day prequential reserve.  Score, route and held
candidate identity are established before R3/policy outcomes are joined for
diagnostics.  This does not retrain residuals, MC1, BCF, portfolio or live
artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
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

from extreme_price_movements.features import SESSION_CALENDAR_FEATURE_KEYS, session_calendar_features  # noqa: E402
from extreme_price_movements.strict_r3_canonical_current import D2_SPEC  # noqa: E402
from extreme_price_movements.strict_r3_canonical_v2 import _fit_medians, _numeric_matrix  # noqa: E402
from extreme_price_movements.strict_r3_self_distillation import build_distillation_weights  # noqa: E402
from scripts.run_strict_r3_base_recall_funnel import (  # noqa: E402
    BASE_ROUTE_FRACTION,
    DEFAULT_CONTROL,
    DEFAULT_SOURCE,
    PERIODS,
    _utc,
    timestamp_route,
)
DEFAULT_B0_ROOT = ROOT / "data_perp/artifacts/strict_r3_long_base_recall_funnel_2025dev_holdout_2026oos_20260822_v1"
MAX_TRAIN_ROWS = 240_000


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _feature_contract(control_root: Path) -> tuple[str, ...]:
    paths = sorted(control_root.glob("bundles/block=*/upstream/monthly_upstream_bundle.joblib"))
    contracts = {tuple(joblib.load(path).base_fields) for path in paths}
    if len(contracts) != 1:
        raise AssertionError("F1 requires one frozen base feature contract")
    fields = next(iter(contracts))
    if len(fields) != 120:
        raise AssertionError("F1 expects exactly 120 frozen B0 fields")
    return fields


def _base_params(bundle: object) -> dict[str, object]:
    model = bundle.base_model
    allowed = (
        "boosting_type", "colsample_bytree", "learning_rate", "max_depth", "min_child_samples",
        "min_child_weight", "min_split_gain", "n_estimators", "num_leaves", "objective",
        "reg_alpha", "reg_lambda", "subsample", "subsample_for_bin", "subsample_freq", "num_class",
        "deterministic", "force_col_wise", "verbosity",
    )
    params = {key: model.get_params()[key] for key in allowed if key in model.get_params()}
    # Keep the saved single-thread deterministic construction.  A feature
    # ablation must not change the learner's thread/tie behaviour while it is
    # meant to isolate only an input block.
    params.update({
        "random_state": int(model.get_params()["random_state"]),
        "n_jobs": int(model.get_params().get("n_jobs", 1)),
    })
    return params


def _load_source(source: Path, fields: tuple[str, ...]) -> pd.DataFrame:
    frame = pd.read_parquet(source, columns=[
        "candidate_id", "__decision_ts__", "r3_class", "r3_label_available_ts",
        "prequential_base_rank42", *fields,
    ])
    frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True, errors="raise")
    frame["r3_label_available_ts"] = pd.to_datetime(frame["r3_label_available_ts"], utc=True, errors="coerce")
    session = session_calendar_features(pd.DatetimeIndex(frame["__decision_ts__"]))
    for name in SESSION_CALENDAR_FEATURE_KEYS:
        frame[name] = session[name]
    if frame["candidate_id"].duplicated().any():
        raise AssertionError("source candidate IDs must be unique")
    return frame


def _strict_train(frame: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    reserve_start = cutoff - pd.Timedelta(days=28)
    valid = frame.loc[
        frame["__decision_ts__"].lt(reserve_start)
        & frame["r3_label_available_ts"].lt(reserve_start)
        & frame["r3_class"].isin([0, 1, 2]),
    ].copy()
    if valid.empty:
        raise ValueError(f"no resolved R3 labels before {cutoff.isoformat()}")
    # Match the frozen D2 base contract: latest label-resolved rows up to the
    # canonical cap.  Do not day-sample or admit labels resolved in the 28-day
    # calibration reserve, as either would make a feature arm incomparable to
    # the B0 control.
    return valid.sort_values("r3_label_available_ts", kind="stable").tail(MAX_TRAIN_ROWS).reset_index(drop=True)


def _d2_weights(train: pd.DataFrame) -> tuple[np.ndarray, dict[str, object]]:
    return build_distillation_weights(
        train,
        teacher_rank_column="prequential_base_rank42",
        layer="base",
        spec=D2_SPEC,
    )


def _rank_ic(frame: pd.DataFrame, score: str) -> float:
    valid = frame.loc[
        frame["policy_path_valid"] & frame["policy_net_bps"].notna(),
        ["__decision_ts__", score, "policy_net_bps"],
    ].copy()
    if valid.empty:
        return float("nan")
    valid["x"] = valid.groupby("__decision_ts__", sort=False)[score].rank()
    valid["y"] = valid.groupby("__decision_ts__", sort=False)["policy_net_bps"].rank()
    corr = valid.groupby("__decision_ts__", sort=False)[["x", "y"]].corr().iloc[0::2, -1]
    return float(corr.mean())


def _diagnose(frame: pd.DataFrame, selected: np.ndarray, score: str, label: str) -> dict[str, object]:
    valid = frame["policy_path_valid"] & frame["policy_net_bps"].notna()
    result: dict[str, object] = {
        "label": label,
        "candidate_rows": int(len(frame)),
        "routed_rows": int(selected.sum()),
        "route_fraction": float(selected.mean()),
        "routed_policy_net_mean_bps": float(frame.loc[selected & valid, "policy_net_bps"].mean()) if (selected & valid).any() else float("nan"),
        "rank_ic": _rank_ic(frame, score),
    }
    for threshold in (50, 100, 200):
        target = frame[f"policy_ge_{threshold}"].to_numpy(bool)
        result[f"recall_policy_ge_{threshold}"] = float((selected & target).sum() / target.sum()) if target.any() else float("nan")
    for field in ("positive_top20", "positive_top10"):
        target = frame[field].to_numpy(bool)
        result[f"recall_{field}"] = float((selected & target).sum() / target.sum()) if target.any() else float("nan")
    result["recall_composite"] = (
        .20 * result["recall_policy_ge_50"] + .30 * result["recall_policy_ge_100"]
        + .25 * result["recall_policy_ge_200"] + .15 * result["recall_positive_top20"]
        + .10 * result["recall_positive_top10"]
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--control-root", type=Path, default=DEFAULT_CONTROL)
    parser.add_argument("--b0-root", type=Path, default=DEFAULT_B0_ROOT)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable output exists: {args.out_dir}")
    fields = _feature_contract(args.control_root)
    f1_fields = (*fields, *SESSION_CALENDAR_FEATURE_KEYS)
    b0 = pd.read_parquet(args.b0_root / "b0_target_free_reconstruction.parquet")
    b0["__decision_ts__"] = pd.to_datetime(b0["__decision_ts__"], utc=True, errors="raise")
    source = _load_source(args.source, fields)
    source_index = source.set_index("candidate_id", drop=False)
    predictions: list[pd.DataFrame] = []
    audit: list[dict[str, object]] = []
    for path in sorted(args.control_root.glob("bundles/block=*/upstream/monthly_upstream_bundle.joblib")):
        block = path.parents[1].name
        bundle = joblib.load(path)
        cutoff = _utc(bundle.cutoff)
        held_ids = b0.loc[b0["control_block"].eq(block), "candidate_id"]
        if held_ids.empty:
            continue
        held = source_index.loc[held_ids.to_numpy()].copy().reset_index(drop=True)
        train = _strict_train(source, cutoff)
        sample_weight, weight_audit = _d2_weights(train)
        medians = _fit_medians(train, f1_fields)
        model = lgb.LGBMClassifier(**_base_params(bundle)).fit(
            _numeric_matrix(train, f1_fields, medians),
            train["r3_class"].astype(int),
            sample_weight=sample_weight,
        )
        probabilities = model.predict_proba(_numeric_matrix(held, f1_fields, medians))
        out = held.loc[:, ["candidate_id", "__decision_ts__"]].copy()
        out["p_adverse_f1"] = probabilities[:, 0]
        out["p_weak_f1"] = probabilities[:, 1]
        out["p_clear_f1"] = probabilities[:, 2]
        out["f1_score"] = out["p_clear_f1"] - .5 * out["p_adverse_f1"]
        out["control_block"] = block
        predictions.append(out)
        audit.append({
            "block": block, "cutoff": cutoff.isoformat(), "held_rows": int(len(held)), "train_rows": int(len(train)),
            "reserve_start": (cutoff - pd.Timedelta(days=28)).isoformat(),
            "all_labels_before_reserve": bool(train["r3_label_available_ts"].lt(cutoff - pd.Timedelta(days=28)).all()),
            "d2_weight_audit_json": json.dumps(weight_audit, sort_keys=True),
        })
    scores = pd.concat(predictions, ignore_index=True).sort_values(["__decision_ts__", "candidate_id"], kind="stable")
    if len(scores) != len(b0) or scores["candidate_id"].duplicated().any():
        raise AssertionError("F1 predictions must preserve frozen B0 candidate identities")
    outcome = pd.read_parquet(args.b0_root / "outcome_joined_recall_ledger.parquet")
    outcome["__decision_ts__"] = pd.to_datetime(outcome["__decision_ts__"], utc=True, errors="raise")
    scored = outcome.merge(scores, on=["candidate_id", "__decision_ts__", "control_block"], how="inner", validate="one_to_one")
    if len(scored) != len(b0):
        raise AssertionError("F1 outcome join must preserve frozen B0 candidate identities")
    scored["f1_route_top30"] = timestamp_route(scored, "f1_score", fraction=BASE_ROUTE_FRACTION)
    scored["b0_route_top30"] = timestamp_route(scored, "base_score", fraction=BASE_ROUTE_FRACTION)
    rows: list[dict[str, object]] = []
    for period, (start, end) in PERIODS.items():
        subset = scored.loc[scored["__decision_ts__"].ge(_utc(start)) & scored["__decision_ts__"].lt(_utc(end))].copy()
        for name, route, score in (("B0", "b0_route_top30", "base_score"), ("F1_session_calendar", "f1_route_top30", "f1_score")):
            item = _diagnose(subset, subset[route].to_numpy(bool), score, period)
            item["arm"] = name
            rows.append(item)
    # Quarterly portability audit: the final partial 2026-Q3 interval remains
    # explicitly labelled rather than folded into Q2.
    for quarter, subset in scored.groupby(scored["__decision_ts__"].dt.to_period("Q"), sort=True):
        if quarter < pd.Period("2025Q4", freq="Q"):
            continue
        for name, route, score in (("B0", "b0_route_top30", "base_score"), ("F1_session_calendar", "f1_route_top30", "f1_score")):
            item = _diagnose(subset, subset[route].to_numpy(bool), score, str(quarter))
            item["arm"] = name
            rows.append(item)
    metrics = pd.DataFrame(rows)
    wide = metrics.pivot(index="label", columns="arm", values=["recall_composite", "recall_policy_ge_100", "routed_policy_net_mean_bps", "rank_ic"])
    gates = []
    for label in ("frozen_holdout_2025q4", "frozen_oos_2026jan_jul"):
        row = wide.loc[label]
        control_comp = float(row[("recall_composite", "B0")])
        f1_comp = float(row[("recall_composite", "F1_session_calendar")])
        gates.append({
            "period": label,
            "relative_recall_gain": f1_comp / control_comp - 1.0,
            "mean_policy_net_delta_bps": float(row[("routed_policy_net_mean_bps", "F1_session_calendar")] - row[("routed_policy_net_mean_bps", "B0")]),
            "rank_ic_delta": float(row[("rank_ic", "F1_session_calendar")] - row[("rank_ic", "B0")]),
        })
    gate_frame = pd.DataFrame(gates)
    quarter = metrics.loc[metrics["label"].str.match(r"^20\d\dQ[1-4]$")].pivot(index="label", columns="arm", values="recall_policy_ge_100")
    advance = bool(
        gate_frame["relative_recall_gain"].ge(.02).all()
        and gate_frame["mean_policy_net_delta_bps"].ge(-5.0).all()
        and gate_frame["rank_ic_delta"].ge(-.005).all()
        and (quarter["F1_session_calendar"] >= quarter["B0"]).all()
    )
    args.out_dir.mkdir(parents=True)
    scores.to_parquet(args.out_dir / "f1_target_free_scores.parquet", index=False, compression="zstd")
    scored.to_parquet(args.out_dir / "f1_outcome_joined_audit.parquet", index=False, compression="zstd")
    pd.DataFrame(audit).to_parquet(args.out_dir / "f1_block_training_audit.parquet", index=False)
    metrics.to_parquet(args.out_dir / "f1_base_metrics.parquet", index=False)
    gate_frame.to_parquet(args.out_dir / "f1_advancement_gate.parquet", index=False)
    (args.out_dir / "run_manifest.json").write_text(json.dumps({
        "schema": "strict_r3_long_f1_session_calendar_v2_canonical_d2",
        "scope": "offline base-only feature block screening; no downstream or live artifact modified",
        "source": {"path": str(args.source), "sha256": _sha256(args.source)},
        "b0_root": str(args.b0_root),
        "feature_contract": {"b0_count": len(fields), "f1_count": len(f1_fields), "session_fields": list(SESSION_CALENDAR_FEATURE_KEYS)},
        "causality": "session fields are timestamp-only; supervised R3 fitting requires label availability before the fully excluded 28-day calibration reserve",
        "base_training_contract": "same 240k latest-label-resolved cap, train-fold median imputation, and D2 teacher weighting as B0",
        "advance_to_downstream_rebuild": advance,
    }, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"event": "complete", "rows": int(len(scored)), "advance": advance}, sort_keys=True))


if __name__ == "__main__":
    main()
