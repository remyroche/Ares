#!/usr/bin/env python3
"""Fit a bank-wise challenger-relative Meta-HPO shortlist proxy.

This utility is deliberately offline.  It turns *matched* strict-MC1 portfolio
receipts into a challenger-versus-fixed-incumbent label table, then performs
leave-one-HPO-bank-out falsification.  It never opens target-free score panels,
changes Meta/MC1/live artifacts, or promotes a candidate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "strict_r3_p8u_meta_challenger_relative_gateproxy_v1"
CORE_FIELDS = (
    "residual_ic", "conditional_mi_given_base", "ic_base_5_10", "ic_base_10_20", "ic_base_20_30",
    "meta_top1_ev", "meta_top2_ev", "top1_candidate_only_minus_control_only_ev",
    "top2_candidate_only_minus_control_only_ev", "false_upgrade_ev", "useful_upgrade_ev",
    "base_meta_rank_correlation", "median_abs_rank_correction", "weekly_q10",
    "probe_delta_top2_ev", "probe_delta_admitted_utility",
)
STRUCTURAL_NUMERIC = ("feature_count", "truncation", "sigmoid")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _once(path: Path, payload: object) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def _read_descriptor(root: Path, trial: str) -> pd.Series:
    correctness = json.loads((root / "correctness_report.json").read_text())
    if not all(value is True for value in correctness.values() if isinstance(value, bool)):
        raise AssertionError(f"{root}: descriptor correctness receipt is incomplete")
    table = pd.read_parquet(root / "trial_descriptor_summary.parquet")
    row = table.loc[table.trial.astype(str).eq(trial)]
    if len(row) != 1:
        raise AssertionError(f"{root}: expected exactly one descriptor row for {trial}, found {len(row)}")
    return row.iloc[0].copy()


def _read_summary(path: Path, trial: str) -> pd.Series:
    table = pd.read_parquet(path)
    row = table.loc[table.trial.astype(str).eq(trial)]
    if len(row) != 1:
        raise AssertionError(f"{path}: expected exactly one summary row for {trial}, found {len(row)}")
    return row.iloc[0].copy()


def _summary_rows(path: Path) -> pd.DataFrame:
    table = pd.read_parquet(path)
    required = {
        "trial", "period", "base_identity_exact", "base_rank_max_abs_delta", "accepted_rows",
        "net_ev_bps_per_realised_trade", "net_sum_bps_realised", "worst_month_bps",
        "worst_week_bps", "max_drawdown", "admission_threshold_bps", "max_new_entries_per_bar",
    }
    missing = sorted(required.difference(table.columns))
    if missing:
        raise AssertionError(f"{path}: summary missing {missing}")
    if not table.base_identity_exact.fillna(False).all() or not table.base_rank_max_abs_delta.fillna(np.inf).eq(0.0).all():
        raise AssertionError(f"{path}: base identity / rank alignment is not exact")
    return table.copy()


def _metric_delta(candidate: pd.Series, incumbent: pd.Series) -> dict[str, float]:
    return {
        "delta_ev_per_trade_bps": float(candidate.net_ev_bps_per_realised_trade - incumbent.net_ev_bps_per_realised_trade),
        "total_net_ratio": float(candidate.net_sum_bps_realised / incumbent.net_sum_bps_realised) if incumbent.net_sum_bps_realised else np.nan,
        "delta_worst_month_bps": float(candidate.worst_month_bps - incumbent.worst_month_bps),
        "delta_worst_week_bps": float(candidate.worst_week_bps - incumbent.worst_week_bps),
        "delta_max_drawdown_pp": float(100.0 * (candidate.max_drawdown - incumbent.max_drawdown)),
        "entry_ratio": float(candidate.accepted_rows / incumbent.accepted_rows) if incumbent.accepted_rows else np.nan,
    }


def _label(delta: dict[str, float], policy: dict[str, Any]) -> tuple[bool, str]:
    standard = dict(policy["all_conditions"])
    standard_ok = (
        delta["delta_ev_per_trade_bps"] >= float(standard["net_ev_per_trade_delta_bps_min"])
        and delta["total_net_ratio"] >= float(standard["total_net_bps_ratio_to_incumbent_min"])
        and delta["delta_worst_week_bps"] >= float(standard["worst_week_delta_bps_min"])
        and delta["delta_max_drawdown_pp"] >= float(standard["max_drawdown_delta_percentage_points_min"])
        and delta["entry_ratio"] >= float(standard["constrained_entries_ratio_to_incumbent_min"])
    )
    exception = dict(policy["material_efficiency_exception"])
    exception_ok = (
        delta["delta_ev_per_trade_bps"] >= float(exception["when_net_ev_per_trade_delta_bps_at_least"])
        and delta["total_net_ratio"] >= float(exception["total_net_bps_ratio_to_incumbent_min"])
        and delta["delta_worst_week_bps"] >= float(standard["worst_week_delta_bps_min"])
        and delta["delta_max_drawdown_pp"] >= float(standard["max_drawdown_delta_percentage_points_min"])
        and delta["entry_ratio"] >= float(exception["constrained_entries_ratio_to_incumbent_min"])
    )
    return bool(standard_ok or exception_ok), "standard" if standard_ok else ("material_efficiency_exception" if exception_ok else "not_beat")


def _margin(delta: dict[str, float]) -> float:
    # The numerical target is a diagnostic ranking margin only.  Hard safety
    # gates remain in BeatIncumbent and cannot be offset by other components.
    return float(
        delta["delta_ev_per_trade_bps"]
        + 100.0 * (delta["total_net_ratio"] - 1.0)
        + delta["delta_worst_week_bps"]
        + 10.0 * delta["delta_max_drawdown_pp"]
        + 50.0 * (delta["entry_ratio"] - 1.0)
    )


def _relative_features(candidate: pd.Series, incumbent: pd.Series) -> dict[str, float]:
    result: dict[str, float] = {}
    for field in CORE_FIELDS + STRUCTURAL_NUMERIC:
        left = pd.to_numeric(pd.Series([candidate.get(field)]), errors="coerce").iloc[0]
        right = pd.to_numeric(pd.Series([incumbent.get(field)]), errors="coerce").iloc[0]
        result[f"candidate__{field}"] = float(left) if pd.notna(left) else np.nan
        result[f"delta__{field}"] = float(left - right) if pd.notna(left) and pd.notna(right) else np.nan
        result[f"relative_delta__{field}"] = float((left - right) / (abs(right) + 1e-6)) if pd.notna(left) and pd.notna(right) else np.nan
    return result


def _pipeline() -> Pipeline:
    return Pipeline((
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", RobustScaler(quantile_range=(10.0, 90.0))),
        ("model", LogisticRegression(C=0.10, class_weight="balanced", max_iter=20_000, random_state=1729)),
    ))


def _margin_pipeline() -> Pipeline:
    return Pipeline((
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", RobustScaler(quantile_range=(10.0, 90.0))),
        ("model", Ridge(alpha=30.0, random_state=1729)),
    ))


def _bankwise_predictions(table: pd.DataFrame, fields: list[str]) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    predictions: list[pd.DataFrame] = []
    audits: list[dict[str, Any]] = []
    for bank_id in sorted(table.bank_id.unique()):
        train = table.loc[table.bank_id.ne(bank_id)].copy()
        test = table.loc[table.bank_id.eq(bank_id)].copy()
        classes = sorted(train.beat_incumbent.unique().tolist())
        audit: dict[str, Any] = {"held_bank": bank_id, "train_rows": len(train), "test_rows": len(test), "train_positive": int(train.beat_incumbent.sum()), "train_negative": int((~train.beat_incumbent).sum())}
        if len(classes) < 2:
            audit["status"] = "unsupported_training_single_class"
            audits.append(audit)
            continue
        probability = _pipeline().fit(train[fields], train.beat_incumbent.astype(int)).predict_proba(test[fields])[:, 1]
        margin_model = _margin_pipeline().fit(train[fields], train.margin_of_victory).predict(test[fields])
        output = test.loc[:, ["bank_id", "trial", "beat_incumbent", "margin_of_victory"]].copy()
        output["p_beat_incumbent"] = probability
        output["predicted_margin_of_victory"] = margin_model
        output["shortlist_score"] = np.maximum(0.0, probability * margin_model)
        output["shortlist_rank"] = output.shortlist_score.rank(method="first", ascending=False).astype(int)
        best_index = int(np.argmax(test.margin_of_victory.to_numpy(float)))
        top = output.nsmallest(min(3, len(output)), "shortlist_rank")
        audit.update({
            "status": "scored",
            "winner_in_top3": bool(test.iloc[best_index].trial in set(top.trial)),
            "regret_at3": float(test.margin_of_victory.iloc[best_index] - test.loc[test.trial.isin(top.trial), "margin_of_victory"].max()),
            "any_beat_in_top3": bool(top.beat_incumbent.any()),
        })
        predictions.append(output); audits.append(audit)
    return (pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame(), audits)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--bank-spec", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    protocol_path, spec_path, out = args.protocol.resolve(), args.bank_spec.resolve(), args.out.resolve()
    if out.exists():
        raise FileExistsError(out)
    protocol, bank_spec = json.loads(protocol_path.read_text()), json.loads(spec_path.read_text())
    if protocol.get("schema") != "strict_r3_p8u_meta_challenger_relative_gateproxy_protocol_v1":
        raise AssertionError("invalid challenger-relative protocol")
    if bank_spec.get("schema") != "strict_r3_p8u_meta_challenger_relative_historical_banks_v1":
        raise AssertionError("invalid bank specification")
    rows: list[dict[str, Any]] = []
    bank_audit: list[dict[str, Any]] = []
    for spec in bank_spec["banks"]:
        candidate_summary_path = (ROOT / str(spec["candidate_summary"])).resolve()
        incumbent_summary_path = (ROOT / str(spec["incumbent_summary"])).resolve()
        candidate_root = (ROOT / str(spec["candidate_descriptor_root"])).resolve()
        incumbent_root = (ROOT / str(spec["incumbent_descriptor_root"])).resolve()
        incumbent_trial = str(spec["incumbent_trial"])
        candidates = _summary_rows(candidate_summary_path)
        incumbent_summary = _read_summary(incumbent_summary_path, incumbent_trial)
        incumbent_descriptor = _read_descriptor(incumbent_root, incumbent_trial)
        candidate_periods = set(candidates.period.astype(str)); candidate_gates = set(pd.to_numeric(candidates.admission_threshold_bps)); candidate_caps = set(pd.to_numeric(candidates.max_new_entries_per_bar))
        if candidate_periods != {str(incumbent_summary.period)} or candidate_gates != {float(incumbent_summary.admission_threshold_bps)} or candidate_caps != {int(incumbent_summary.max_new_entries_per_bar)}:
            raise AssertionError(f"{spec['bank_id']}: candidate/incumbent MC1 contract mismatch")
        candidates = candidates.loc[~candidates.trial.astype(str).eq(incumbent_trial)].copy()
        for _, candidate_summary in candidates.iterrows():
            trial = str(candidate_summary.trial)
            candidate_descriptor = _read_descriptor(candidate_root, trial)
            delta = _metric_delta(candidate_summary, incumbent_summary)
            beat, rationale = _label(delta, protocol["beat_incumbent_label"])
            row: dict[str, Any] = {
                "bank_id": str(spec["bank_id"]), "trial": trial, "incumbent_trial": incumbent_trial,
                "period": str(candidate_summary.period), "beat_incumbent": beat, "beat_rationale": rationale,
                "margin_of_victory": _margin(delta), **delta,
            }
            row.update(_relative_features(candidate_descriptor, incumbent_descriptor))
            for field in ("target_family", "loss", "query_contract", "feature_contract", "sample_weight_profile"):
                row[f"candidate_{field}"] = str(candidate_descriptor.get(field, ""))
                row[f"incumbent_{field}"] = str(incumbent_descriptor.get(field, ""))
                row[f"same_{field}"] = row[f"candidate_{field}"] == row[f"incumbent_{field}"]
            rows.append(row)
        bank_audit.append({"bank_id": str(spec["bank_id"]), "period": str(incumbent_summary.period), "incumbent_trial": incumbent_trial, "candidate_count": len(candidates), "candidate_summary_sha256": _sha256(candidate_summary_path), "incumbent_summary_sha256": _sha256(incumbent_summary_path), "candidate_descriptor_sha256": _sha256(candidate_root / "trial_descriptor_summary.parquet"), "incumbent_descriptor_sha256": _sha256(incumbent_root / "trial_descriptor_summary.parquet")})
    table = pd.DataFrame(rows)
    if table.empty:
        raise AssertionError("no challenger rows")
    feature_fields = [column for column in table.columns if column.startswith(("candidate__", "delta__", "relative_delta__"))]
    predictions, cv_audit = _bankwise_predictions(table, feature_fields)
    eligible_for_fit = table.bank_id.nunique() >= 3 and table.beat_incumbent.nunique() == 2 and int(table.beat_incumbent.sum()) >= 3 and int((~table.beat_incumbent).sum()) >= 3
    out.mkdir(parents=True)
    table.to_parquet(out / "challenger_relative_training_table.parquet", index=False, compression="zstd")
    pd.DataFrame(bank_audit).to_parquet(out / "bank_audit.parquet", index=False, compression="zstd")
    pd.DataFrame(cv_audit).to_parquet(out / "leave_one_bank_out_audit.parquet", index=False, compression="zstd")
    if not predictions.empty:
        predictions.to_parquet(out / "leave_one_bank_out_predictions.parquet", index=False, compression="zstd")
    if eligible_for_fit:
        classifier = _pipeline().fit(table[feature_fields], table.beat_incumbent.astype(int))
        regressor = _margin_pipeline().fit(table[feature_fields], table.margin_of_victory)
        joblib.dump({"schema": SCHEMA, "fields": feature_fields, "classifier": classifier, "margin_regressor": regressor, "protocol_sha256": _sha256(protocol_path)}, out / "challenger_relative_gateproxy.joblib")
    _once(out / "run_manifest.json", {
        "schema": SCHEMA,
        "scope": "offline historical challenger-relative proxy bootstrap only; no HPO selection, promotion, score, MC1, admission, portfolio, live, or exchange mutation",
        "protocol": str(protocol_path), "protocol_sha256": _sha256(protocol_path),
        "bank_spec": str(spec_path), "bank_spec_sha256": _sha256(spec_path),
        "banks": bank_audit, "feature_fields": feature_fields,
        "eligible_for_final_fit": eligible_for_fit,
        "training_support": {"banks": int(table.bank_id.nunique()), "challengers": int(len(table)), "positive_beats": int(table.beat_incumbent.sum()), "negative_beats": int((~table.beat_incumbent).sum())},
        "decision": "Historical bootstrap only. A future independent HPO bank is required before this protocol may shortlist a new candidate bank.",
    })
    _once(out / "correctness_report.json", {
        "all_banks_use_matched_period_gate_capacity_and_exact_base_identity": True,
        "incumbent_is_external_to_each_challenger_pool": True,
        "outcome_fields_are_opened_only_for_historical_label_construction": True,
        "leave_one_hpo_bank_out_never_splits_nearby_trials": True,
        "no_proxy_prediction_has_hpo_promotion_or_live_authority": True,
        "insufficient_support_fails_closed_without_serializing_a_final_model": not eligible_for_fit,
    })
    print(out)


if __name__ == "__main__":
    main()
