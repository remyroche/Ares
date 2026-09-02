#!/usr/bin/env python3
"""Sequential short-base target/objective ablation funnel.

Round 1 deliberately changes only the base supervision/objective (plus the
single R3 equal-timestamp control).  It uses one strict chronological split,
the frozen 120-field short contract, target-free candidate identities, and
only labels resolved before the held window.  It is not a policy HPO or a
promotion script.

The runner is also reusable for the later funnel stages through ``--spec``
and ``--weight-mode`` / window arguments.  It never turns invalid future paths
into economic failures and only opens exact 1-minute paths after score
selections are frozen.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Literal

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    mean_absolute_error,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_strict_r3_ordinal_base_target_ablation import (  # noqa: E402
    FROZEN_BASE_PARAMS,
    _coverage_fields,
    _feature_fields,
    _load_candidates,
    _load_features,
    _r3_target,
    _sha256,
    _short_policy_outcomes,
    _side_paths,
    _valid_label,
)


SEED = 17
TAILS = (0.001, 0.0025, 0.005, 0.01, 0.02, 0.05)
POLICY_TAILS = TAILS
WeightMode = Literal["ordinary", "equal_timestamp", "equal_month_timestamp", "recency_equal_month_timestamp"]
Family = Literal["r3", "regression", "rank", "binary", "ordinal"]


@dataclass(frozen=True)
class Spec:
    name: str
    family: Family
    description: str
    target: str
    objective: str
    weight_mode: WeightMode = "ordinary"
    relevance: str | None = None


ROUND1: tuple[Spec, ...] = (
    Spec("A0_r3_ordinary", "r3", "Canonical R3 multiclass control.", "r3", "multiclass"),
    Spec("A1_r3_equal_timestamp", "r3", "Canonical R3 with equal timestamp weight.", "r3", "multiclass", "equal_timestamp"),
    Spec("B1_net_huber", "regression", "Clipped H12 net regression with Huber loss.", "net_clip500", "huber", "equal_timestamp"),
    Spec("C1_net_lambdarank_standard", "rank", "Economic H12-net LambdaRank, symmetric relevance.", "net", "lambdarank", "ordinary", "standard_net"),
    Spec("C2_net_lambdarank_tail", "rank", "Tail-focused economic H12-net LambdaRank.", "net", "lambdarank", "ordinary", "tail_net"),
    Spec("D1_gross_lambdarank", "rank", "H12 gross LambdaRank; cost is applied downstream.", "gross", "lambdarank", "ordinary", "standard_gross"),
    Spec("E1_p_net_gt_0", "binary", "P(H12 net > 0 bps).", "net_gt_0", "binary", "equal_timestamp"),
    Spec("E2_p_net_gt_100", "binary", "P(H12 net > +100 bps).", "net_gt_100", "binary", "equal_timestamp"),
    Spec("F1_ordinal_economic", "ordinal", "Six-class ordinal economic classifier.", "ordinal_economic", "multiclass", "equal_timestamp"),
)

ORDINAL_VALUES = np.asarray([-300.0, -125.0, 0.0, 100.0, 225.0, 450.0], dtype=np.float32)
LABEL_GAINS: dict[str, list[float]] = {
    "standard_net": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    "standard_gross": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    "tail_net": [0.0, 1.0, 2.0, 4.0, 8.0, 16.0],
}


def _utc(values: pd.Series | pd.Index) -> pd.Series:
    return pd.to_datetime(values, utc=True, errors="raise")


def _load_labels(root: Path, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for month in pd.date_range(start.normalize().replace(day=1), end, freq="MS", inclusive="left"):
        path = root / "parts" / f"month={month:%Y-%m}" / "side=short.parquet"
        if not path.exists():
            raise FileNotFoundError(path)
        pieces.append(pd.read_parquet(path))
    frame = pd.concat(pieces, ignore_index=True)
    for column in ("__ts__", "__decision_ts__", "__label_available_at__"):
        frame[column] = _utc(frame[column])
    if frame.candidate_id.duplicated().any() or not frame.side_name.astype(str).str.lower().eq("short").all():
        raise ValueError("short labels have invalid identities or side")
    return frame


def _target(frame: pd.DataFrame, spec: Spec) -> np.ndarray:
    valid = _valid_label(frame).to_numpy(dtype=bool)
    net = pd.to_numeric(frame["t4_tp6_sl4_net_bps"], errors="coerce").to_numpy(dtype=float)
    gross = pd.to_numeric(frame["t4_tp6_sl4_gross_bps"], errors="coerce").to_numpy(dtype=float)
    if spec.target == "r3":
        return _r3_target(frame).to_numpy(dtype=float)
    if spec.target == "net_clip500":
        result = np.clip(net, -500.0, 500.0)
        return np.where(valid, result, np.nan)
    if spec.target == "net":
        return np.where(valid, net, np.nan)
    if spec.target == "gross":
        return np.where(valid, gross, np.nan)
    if spec.target == "net_gt_0":
        return np.where(valid, (net > 0.0).astype(float), np.nan)
    if spec.target == "net_gt_100":
        return np.where(valid, (net > 100.0).astype(float), np.nan)
    if spec.target == "ordinal_economic":
        # [-inf,-200], (-200,-50], (-50,50], (50,150], (150,300], (300,inf)
        result = np.digitize(net, [-200.0, -50.0, 50.0, 150.0, 300.0], right=True).astype(float)
        return np.where(valid, result, np.nan)
    raise ValueError(f"unknown target {spec.target}")


def _relevance(values: np.ndarray, name: str) -> np.ndarray:
    if name == "standard_net":
        return np.digitize(values, [-200.0, -100.0, 0.0, 100.0, 200.0, 400.0], right=True).astype(np.int32)
    if name == "standard_gross":
        # Same economic partitions shifted by the fixed 100-bps cost.
        return np.digitize(values, [-100.0, 0.0, 100.0, 200.0, 300.0, 500.0], right=True).astype(np.int32)
    if name == "tail_net":
        return np.digitize(values, [0.0, 100.0, 200.0, 300.0, 500.0], right=True).astype(np.int32)
    raise ValueError(f"unknown relevance geometry {name}")


def _weights(train: pd.DataFrame, mode: WeightMode, *, reference_end: pd.Timestamp) -> np.ndarray:
    if mode == "ordinary":
        return np.ones(len(train), dtype=np.float64)
    group_n = train.groupby("__ts__", sort=False)["__ts__"].transform("size").to_numpy(dtype=float)
    raw = 1.0 / np.maximum(group_n, 1.0)
    if mode in {"equal_month_timestamp", "recency_equal_month_timestamp"}:
        month = train["__ts__"].dt.strftime("%Y-%m")
        n_month = month.map(month.value_counts()).to_numpy(dtype=float)
        raw = raw / np.maximum(n_month, 1.0)
    if mode == "recency_equal_month_timestamp":
        age_days = (reference_end - train["__ts__"]).dt.total_seconds().to_numpy(dtype=float) / 86400.0
        raw *= np.power(0.5, np.maximum(age_days, 0.0) / 90.0)
    raw /= float(np.mean(raw))
    if not np.isfinite(raw).all() or (raw <= 0.0).any():
        raise ValueError("training weights are invalid")
    return raw.astype(np.float64)


def _matrix(frame: pd.DataFrame, fields: list[str], medians: pd.Series) -> pd.DataFrame:
    x = frame.loc[:, fields].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    x = x.fillna(medians)
    if x.isna().any().any():
        raise AssertionError("training-only median imputation left missing model inputs")
    return x.astype(np.float32)


def _groups(frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    ordered = frame.sort_values(["__ts__", "candidate_id"], kind="stable").reset_index(drop=True)
    group = ordered.groupby("__ts__", sort=False).size().to_numpy(dtype=np.int32)
    if int(group.sum()) != len(ordered) or (group <= 0).any():
        raise AssertionError("invalid LambdaRank query groups")
    return ordered, group


def _model_params(family: Family, objective: str, relevance: str | None) -> dict[str, Any]:
    params = dict(FROZEN_BASE_PARAMS)
    params.pop("num_class", None)
    params["objective"] = objective
    if family in {"r3", "ordinal"}:
        params["num_class"] = 3 if family == "r3" else 6
    if family == "rank":
        params.update({
            "lambdarank_truncation_level": 8,
            "label_gain": LABEL_GAINS[str(relevance)],
            "lambdarank_norm": True,
        })
    if objective == "quantile":
        params["alpha"] = 0.5
    return params


def _target_audit(y: np.ndarray, spec: Spec) -> dict[str, Any]:
    """Persist compact target support without serialising every raw net bps."""
    values = pd.Series(y, dtype="float64")
    audit: dict[str, Any] = {
        "target_train_summary": {
            "rows": int(len(values)),
            "unique_values": int(values.nunique(dropna=True)),
            "min": float(values.min()),
            "p01": float(values.quantile(0.01)),
            "p50": float(values.median()),
            "p99": float(values.quantile(0.99)),
            "max": float(values.max()),
        },
    }
    if spec.family == "rank":
        relevance = _relevance(y, str(spec.relevance))
        audit["rank_relevance_counts"] = {
            str(key): int(value)
            for key, value in pd.Series(relevance).value_counts().sort_index().items()
        }
    else:
        audit["target_class_counts"] = {
            str(key): int(value)
            for key, value in values.value_counts().sort_index().items()
        }
    return audit


def _fit_predict(
    train: pd.DataFrame, test: pd.DataFrame, fields: list[str], medians: pd.Series, spec: Spec, *, train_end: pd.Timestamp,
) -> tuple[np.ndarray, dict[str, np.ndarray], Any, dict[str, Any]]:
    weights = _weights(train, spec.weight_mode, reference_end=train_end)
    y = _target(train, spec)
    if not np.isfinite(y).all():
        raise AssertionError(f"{spec.name} has unresolved training labels")
    x_test = _matrix(test, fields, medians)
    payload: dict[str, np.ndarray] = {}
    if spec.family == "rank":
        ordered, groups = _groups(train)
        y_rank = _relevance(_target(ordered, spec), str(spec.relevance))
        model = lgb.LGBMRanker(**_model_params(spec.family, spec.objective, spec.relevance))
        model.fit(_matrix(ordered, fields, medians), y_rank, group=groups, sample_weight=_weights(ordered, spec.weight_mode, reference_end=train_end))
        score = model.predict(x_test).astype(np.float32)
        payload["relevance"] = _relevance(_target(test, spec), str(spec.relevance)).astype(np.int8)
    elif spec.family == "regression":
        model = lgb.LGBMRegressor(**_model_params(spec.family, spec.objective, spec.relevance))
        model.fit(_matrix(train, fields, medians), y, sample_weight=weights)
        score = model.predict(x_test).astype(np.float32)
    else:
        model = lgb.LGBMClassifier(**_model_params(spec.family, spec.objective, spec.relevance))
        model.fit(_matrix(train, fields, medians), y.astype(int), sample_weight=weights)
        probabilities = np.asarray(model.predict_proba(x_test), dtype=np.float32)
        if spec.family == "r3":
            score = probabilities[:, 2] - 0.5 * probabilities[:, 0]
        elif spec.family == "ordinal":
            score = probabilities @ ORDINAL_VALUES
        else:
            score = probabilities[:, 1]
        for index in range(probabilities.shape[1]):
            payload[f"p{index}"] = probabilities[:, index]
    audit = {
        "weight_mode": spec.weight_mode,
        "weight_min": float(weights.min()), "weight_mean": float(weights.mean()), "weight_max": float(weights.max()),
    }
    audit.update(_target_audit(y, spec))
    return np.asarray(score, dtype=np.float32), payload, model, audit


def _safe_spearman(left: pd.Series, right: pd.Series) -> float:
    good = left.notna() & right.notna() & np.isfinite(left) & np.isfinite(right)
    return float(left.loc[good].corr(right.loc[good], method="spearman")) if int(good.sum()) > 1 else float("nan")


def _query_metrics(frame: pd.DataFrame) -> dict[str, float]:
    values: list[tuple[int, float, float]] = []
    for _, group in frame.groupby("__ts__", sort=False):
        if len(group) < 2:
            continue
        values.append((
            len(group),
            _safe_spearman(group.score, pd.to_numeric(group.t4_tp6_sl4_net_bps, errors="coerce")),
            _safe_spearman(group.score, pd.to_numeric(group.t4_tp6_sl4_gross_bps, errors="coerce")),
        ))
    query = pd.DataFrame(values, columns=["rows", "net_ic", "gross_ic"])
    if query.empty:
        return {"query_count": 0.0, "query_net_ic_mean": float("nan"), "query_net_ic_median": float("nan"), "query_net_ic_positive_fraction": float("nan"), "query_gross_ic_mean": float("nan")}
    return {
        "query_count": float(len(query)),
        "query_net_ic_mean": float(np.average(query.net_ic.fillna(0.0), weights=query.rows)),
        "query_net_ic_median": float(query.net_ic.median()),
        "query_net_ic_positive_fraction": float((query.net_ic > 0.0).mean()),
        "query_gross_ic_mean": float(np.average(query.gross_ic.fillna(0.0), weights=query.rows)),
    }


def _aggregate_metrics(frame: pd.DataFrame, spec: Spec, scope: str) -> dict[str, Any]:
    valid = _valid_label(frame)
    resolved = frame.loc[valid].copy()
    record: dict[str, Any] = {
        "spec": spec.name, "family": spec.family, "objective": spec.objective, "weight_mode": spec.weight_mode,
        "scope": scope, "scored_rows": int(len(frame)), "resolved_rows": int(len(resolved)),
        "resolved_fraction": float(len(resolved) / max(len(frame), 1)),
        "score_net_spearman": _safe_spearman(resolved.score, pd.to_numeric(resolved.t4_tp6_sl4_net_bps, errors="coerce")),
        "score_gross_spearman": _safe_spearman(resolved.score, pd.to_numeric(resolved.t4_tp6_sl4_gross_bps, errors="coerce")),
    }
    record.update(_query_metrics(resolved))
    y = _target(resolved, spec)
    if spec.family in {"r3", "ordinal"}:
        pcols = [f"p{i}" for i in range(3 if spec.family == "r3" else 6)]
        p = resolved[pcols].to_numpy(dtype=float)
        p = p / np.maximum(p.sum(axis=1, keepdims=True), 1e-12)
        record["target_log_loss"] = float(log_loss(y.astype(int), p, labels=list(range(p.shape[1]))))
        record["target_macro_f1"] = float(f1_score(y.astype(int), p.argmax(axis=1), average="macro"))
    elif spec.family == "binary":
        p = resolved.p1.to_numpy(dtype=float)
        record["target_auc"] = float(roc_auc_score(y.astype(int), p))
        record["target_pr_auc"] = float(average_precision_score(y.astype(int), p))
        record["target_brier"] = float(brier_score_loss(y.astype(int), p))
    elif spec.family == "regression":
        record["target_mae_bps"] = float(mean_absolute_error(y, resolved.score.to_numpy(dtype=float)))
    else:
        record["target_relevance_spearman"] = _safe_spearman(resolved.score, resolved.relevance)
    return record


def _tail_metrics(frame: pd.DataFrame, spec: Spec, scope: str) -> list[dict[str, Any]]:
    ordered = frame.sort_values("score", ascending=False, kind="stable")
    result: list[dict[str, Any]] = []
    for tail in TAILS:
        selected = ordered.iloc[:max(1, int(math.ceil(len(ordered) * tail)))]
        valid = selected.loc[_valid_label(selected)].copy()
        net = pd.to_numeric(valid.t4_tp6_sl4_net_bps, errors="coerce")
        q10 = float(net.quantile(0.10)) if len(net) else float("nan")
        cvar = float(net.loc[net.le(q10)].mean()) if len(net) else float("nan")
        result.append({
            "spec": spec.name, "family": spec.family, "objective": spec.objective, "weight_mode": spec.weight_mode,
            "scope": scope, "tail_fraction": tail, "requested_rows": int(len(selected)), "resolved_rows": int(len(valid)),
            "coverage": float(len(valid) / max(len(selected), 1)), "mean_score": float(selected.score.mean()),
            "h12_gross_bps": float(pd.to_numeric(valid.t4_tp6_sl4_gross_bps, errors="coerce").mean()),
            "h12_net_bps": float(net.mean()), "h12_net_median_bps": float(net.median()), "h12_net_p10_bps": q10,
            "h12_net_cvar10_bps": cvar, "fraction_net_lt_n200": float((net < -200.0).mean()),
            "fraction_net_lt_n400": float((net < -400.0).mean()),
        })
    return result


def _scope_selected_ids(predictions: dict[str, pd.DataFrame]) -> set[str]:
    chosen: set[str] = set()
    for frame in predictions.values():
        for _, scope in [("all", frame), *[(month, g) for month, g in frame.groupby(frame.__ts__.dt.strftime("%Y-%m"), sort=True)]]:
            for tail in POLICY_TAILS:
                chosen.update(scope.nlargest(max(1, int(math.ceil(len(scope) * tail))), "score").candidate_id.astype(str))
    return chosen


def _policy_metrics(
    predictions: dict[str, pd.DataFrame], policy: pd.DataFrame, specs: tuple[Spec, ...],
) -> pd.DataFrame:
    by_id = policy.set_index("candidate_id")
    records: list[dict[str, Any]] = []
    for spec in specs:
        frame = predictions[spec.name]
        scopes = [("oos", frame), *[(month, group) for month, group in frame.groupby(frame.__ts__.dt.strftime("%Y-%m"), sort=True)]]
        for scope, rows in scopes:
            ordered = rows.sort_values("score", ascending=False, kind="stable")
            for tail in POLICY_TAILS:
                selected = ordered.iloc[:max(1, int(math.ceil(len(ordered) * tail)))]
                joined = selected.loc[:, ["candidate_id"]].join(by_id, on="candidate_id", how="left")
                valid = joined.loc[joined.policy_path_valid.fillna(False).astype(bool)]
                net = pd.to_numeric(valid.policy_net_bps, errors="coerce")
                q10 = float(net.quantile(0.10)) if len(net) else float("nan")
                records.append({
                    "spec": spec.name, "family": spec.family, "objective": spec.objective, "weight_mode": spec.weight_mode,
                    "scope": scope, "tail_fraction": tail, "requested_rows": int(len(selected)), "resolved_rows": int(len(valid)),
                    "coverage": float(len(valid) / max(len(selected), 1)),
                    "policy_gross_bps": float(pd.to_numeric(valid.policy_gross_bps, errors="coerce").mean()),
                    "policy_net_bps": float(net.mean()), "policy_net_median_bps": float(net.median()),
                    "policy_net_p10_bps": q10, "policy_net_cvar10_bps": float(net.loc[net.le(q10)].mean()) if len(net) else float("nan"),
                    "fraction_net_lt_n200": float((net < -200.0).mean()), "fraction_net_lt_n400": float((net < -400.0).mean()),
                })
    return pd.DataFrame(records)


def run(
    *, out: Path, train_start: pd.Timestamp, oos_start: pd.Timestamp, oos_end: pd.Timestamp,
    features_path: Path | None = None, candidates_path: Path | None = None, labels_root: Path | None = None,
    specs: tuple[Spec, ...] = ROUND1, fields: list[str] | None = None,
    feature_selection_audit: dict[str, Any] | None = None,
) -> Path:
    if out.exists():
        raise FileExistsError(f"immutable output exists: {out}")
    if not (train_start < oos_start < oos_end):
        raise ValueError("invalid chronological windows")
    paths = _side_paths("short")
    features_path = (features_path or paths["features"]).resolve()
    candidates_path = (candidates_path or paths["candidates"]).resolve()
    labels_root = (labels_root or paths["labels"]).resolve()
    out.mkdir(parents=True)
    fields = list(fields or _feature_fields("short"))
    selection_audit = feature_selection_audit or {
        "mode": "full_frozen_contract", "field_count": len(fields),
    }
    candidates = _load_candidates(candidates_path, "short")
    candidates = candidates.loc[candidates.__ts__.ge(train_start) & candidates.__ts__.lt(oos_end)].copy()
    features = _load_features(features_path, fields, candidates, "short")
    labels = _load_labels(labels_root, train_start, oos_end)
    ledger = features.merge(labels, on=["candidate_id", "__ts__", "__decision_ts__", "__symbol__", "side_name"], how="left", validate="one_to_one")
    if len(ledger) != len(features):
        raise AssertionError("future-label join changed candidate cardinality")
    ledger["entry_executable"] = ledger.entry_executable.astype(bool)
    train_population = ledger.loc[ledger.__ts__.ge(train_start) & ledger.__ts__.lt(oos_start)]
    kept, coverage = _coverage_fields(train_population, fields)
    if set(fields).difference(kept):
        raise ValueError("frozen short feature contract fails target-free train coverage")
    train = ledger.loc[
        ledger.__ts__.ge(train_start) & ledger.__ts__.lt(oos_start) & ledger.entry_executable
        & _valid_label(ledger) & ledger.__label_available_at__.lt(oos_start)
    ].copy()
    test = ledger.loc[ledger.__ts__.ge(oos_start) & ledger.__ts__.lt(oos_end) & ledger.entry_executable].copy()
    if train.empty or test.empty:
        raise ValueError("empty strict train or OOS population")
    medians = train.loc[:, fields].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).median()
    if medians.isna().any():
        raise AssertionError("short frozen features lack train-only median")
    aggregate: list[dict[str, Any]] = []
    tails: list[dict[str, Any]] = []
    predictions: dict[str, pd.DataFrame] = {}
    audits: dict[str, Any] = {}
    for spec in specs:
        print(f"fitting {spec.name}", flush=True)
        score, extra, model, audit = _fit_predict(train, test, fields, medians, spec, train_end=oos_start)
        prediction = test.loc[:, [
            "candidate_id", "__ts__", "__decision_ts__", "__symbol__", "side_name", "label_valid", "target_invalid", "invalid_reason",
            "tp6_sl4_entry_price", "atr_1h", "t4_tp6_sl4_gross_bps", "t4_tp6_sl4_net_bps", "t2_tp6_sl4_event", "robust_clear_event_b25",
        ]].copy()
        prediction["score"] = score
        for name, values in extra.items():
            prediction[name] = values
        prediction.to_parquet(out / f"oos_predictions_{spec.name}.parquet", index=False, compression="zstd")
        model.booster_.save_model(str(out / f"model_{spec.name}.txt"))
        aggregate.append(_aggregate_metrics(prediction, spec, "oos"))
        tails.extend(_tail_metrics(prediction, spec, "oos"))
        for month, group in prediction.groupby(prediction.__ts__.dt.strftime("%Y-%m"), sort=True):
            aggregate.append(_aggregate_metrics(group, spec, month))
            tails.extend(_tail_metrics(group, spec, month))
        predictions[spec.name] = prediction
        audits[spec.name] = audit
        del model, score, extra
        gc.collect()
    # All model score selections are frozen before accessing future paths.
    selected_ids = _scope_selected_ids(predictions)
    selected = test.loc[test.candidate_id.astype(str).isin(selected_ids)].copy()
    policy, policy_audit = _short_policy_outcomes(selected, train)
    policy.to_parquet(out / "exact1m_policy_selected_outcomes.parquet", index=False, compression="zstd")
    pd.DataFrame(aggregate).to_parquet(out / "metrics_by_scope.parquet", index=False, compression="zstd")
    pd.DataFrame(tails).to_parquet(out / "tail_metrics_by_scope.parquet", index=False, compression="zstd")
    _policy_metrics(predictions, policy, specs).to_parquet(out / "policy_metrics_by_scope_tail.parquet", index=False, compression="zstd")
    manifest = {
        "schema": "strict_r3_short_base_target_objective_funnel_v1",
        "status": "complete", "side": "short",
        "train_decision_window": f"[{train_start.isoformat()}, {oos_start.isoformat()})",
        "strict_label_availability_gate": f"label_available_at < {oos_start.isoformat()}",
        "oos_decision_window": f"[{oos_start.isoformat()}, {oos_end.isoformat()})",
        "entry": "signal close + one hour; exact decision-minute open", "label_horizon": "12 hours",
        "h12_contract": "TP +6 ATR / SL -4 ATR; adverse tie; 100 bps cost exactly once; R3 B25/T50",
        "feature_contract": "base_fields_by_side.short", "feature_count": len(fields), "feature_coverage_gate": ">=90% target-free train entry-executable rows",
        "selected_feature_contract": selection_audit,
        "selected_features": fields,
        "feature_coverage": {field: float(coverage[field]) for field in fields},
        "training_rows": int(len(train)), "oos_scored_rows": int(len(test)), "oos_resolved_rows": int(_valid_label(test).sum()),
        "specs": [asdict(spec) for spec in specs], "arm_audits": audits,
        "fixed_policy_diagnostic": policy_audit,
        "policy_selection": "score selections frozen before exact 1m paths; no portfolio/admission layer",
        "features_sha256": _sha256(features_path), "candidates_sha256": _sha256(candidates_path),
        "labels_manifest_sha256": _sha256(labels_root / "run_manifest.json"),
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    return out


def _parse_timestamp(value: str) -> pd.Timestamp:
    result = pd.Timestamp(value)
    return result.tz_localize("UTC") if result.tzinfo is None else result.tz_convert("UTC")


def _selected_short_feature_fields(contract_path: Path | None) -> tuple[list[str], dict[str, Any]]:
    """Return the immutable short feature subset, if one was selected upstream.

    A Stage-I selection artifact is permitted because it is fit before the
    held OOS window.  This loader deliberately accepts *only* a subset of the
    configured causal short base contract: it cannot smuggle an ad-hoc field,
    label, score, or future-path column into a later target comparison.
    """
    frozen = _feature_fields("short")
    if contract_path is None:
        return frozen, {"mode": "full_frozen_contract", "field_count": len(frozen)}
    resolved = contract_path.resolve()
    payload = json.loads(resolved.read_text())
    fields = payload.get("selected_features")
    selection_key = "selected_features"
    # The chronological MDA receipt deliberately stores several frozen
    # nested subsets and a predeclared recommended size instead of duplicating
    # one top-level list.  Resolve that exact recommended subset; never fall
    # back to the ranked superset, whose additional fields were not selected.
    if fields is None and isinstance(payload.get("feature_sets"), dict):
        size = payload.get("recommended_feature_size_development_only")
        if isinstance(size, int):
            fields = payload["feature_sets"].get(str(size))
            selection_key = f"feature_sets[{size}]"
    if not isinstance(fields, list) or not fields or not all(isinstance(value, str) for value in fields):
        raise ValueError("selected short feature contract has no string selected_features list")
    if len(fields) != len(set(fields)):
        raise ValueError("selected short feature contract contains duplicate fields")
    forbidden = sorted(set(fields).difference(frozen))
    if forbidden:
        raise ValueError(f"selected short feature contract is outside base_fields_by_side.short: {forbidden}")
    if len(fields) < 4:
        raise ValueError("selected short feature contract has fewer than four causal fields")
    return list(fields), {
        "mode": "stagei_selected_subset",
        "path": str(resolved),
        "sha256": _sha256(resolved),
        "selection_description": payload.get("selection"),
        "selection_training_window": payload.get("training_window"),
        "selection_key": selection_key,
        "field_count": len(fields),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--train-start", default="2024-01-01T00:00:00Z")
    parser.add_argument("--oos-start", default="2024-04-01T00:00:00Z")
    parser.add_argument("--oos-end", default="2024-07-01T00:00:00Z")
    parser.add_argument("--features", type=Path)
    parser.add_argument("--candidates", type=Path)
    parser.add_argument("--labels", type=Path)
    parser.add_argument(
        "--selected-feature-contract", type=Path,
        help="Immutable causal short Stage-I selected_features.json; must be a subset of config base_fields_by_side.short.",
    )
    parser.add_argument("--spec", action="append", default=[], help="run only named predeclared spec; repeatable")
    parser.add_argument(
        "--weight-mode", action="append", default=[], choices=(
            "ordinary", "equal_timestamp", "equal_month_timestamp", "recency_equal_month_timestamp",
        ), help="expand each selected spec over these predeclared training-weight modes",
    )
    args = parser.parse_args()
    specs = ROUND1
    if args.spec:
        wanted = set(args.spec)
        specs = tuple(spec for spec in ROUND1 if spec.name in wanted)
        if wanted != {spec.name for spec in specs}:
            raise ValueError(f"unknown spec(s): {sorted(wanted - {spec.name for spec in specs})}")
    if args.weight_mode:
        base_specs = specs
        specs = tuple(
            replace(spec, name=f"{spec.name}__w_{mode}", weight_mode=mode)
            for spec in base_specs for mode in args.weight_mode
        )
    selected_fields, selection_audit = _selected_short_feature_fields(args.selected_feature_contract)
    print(run(
        out=args.out.resolve(), train_start=_parse_timestamp(args.train_start), oos_start=_parse_timestamp(args.oos_start),
        oos_end=_parse_timestamp(args.oos_end), features_path=args.features, candidates_path=args.candidates,
        labels_root=args.labels, specs=specs, fields=selected_fields, feature_selection_audit=selection_audit,
    ))


if __name__ == "__main__":
    main()
