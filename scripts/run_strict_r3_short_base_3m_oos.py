#!/usr/bin/env python3
"""Fit one strict short-side R3 base probe on a chronological 3m/3m split.

This is deliberately a *base-layer* validation only.  It uses the same
exact H12 TP6/SL4 R3 label contract as the long base (exact next-hour
one-minute entry, adverse same-minute tie break, and 100 bps cost once), but
creates its own short candidate identities and its own short feature matrix.

No label-dependent feature selection, HPO, calibration, portfolio admission,
or meta/consensus score is included here.  The resulting OOS score is
``P(robust_clear) - 0.5 * P(adverse)``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, log_loss, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "data_perp/artifacts/strict_r3_short_base_3m_train_3m_oos_2024_20260820_v1"
DEFAULT_FEATURES = DEFAULT_OUT / "features/canonical120_features.parquet"
DEFAULT_LABELS = DEFAULT_OUT / "labels"
DEFAULT_CANDIDATES = DEFAULT_OUT / "short_target_free_candidate_population.parquet"
FEATURE_CONTRACT = ROOT / "config/strict_r3_canonical_v2_feature_contract.json"

TRAIN_START = pd.Timestamp("2024-01-01T00:00:00Z")
OOS_START = pd.Timestamp("2024-04-01T00:00:00Z")
OOS_END = pd.Timestamp("2024-07-01T00:00:00Z")
SEED = 17
TAILS = (0.01, 0.02, 0.05, 0.10, 0.20, 0.30)

# This is the frozen long R3-base HPO configuration.  The experiment freezes
# it unchanged: adapting side-specific parameters would turn this requested
# 3m/3m measurement into an HPO result.
FROZEN_BASE_PARAMS: dict[str, Any] = {
    "objective": "multiclass",
    "num_class": 3,
    "n_estimators": 140,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 350,
    "subsample": 0.80,
    "subsample_freq": 1,
    "colsample_bytree": 0.80,
    "reg_lambda": 8.0,
    "random_state": SEED,
    "n_jobs": 1,
    "deterministic": True,
    "force_col_wise": True,
    "verbosity": -1,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc(values: pd.Series | pd.Index) -> pd.Series:
    return pd.to_datetime(values, utc=True, errors="raise")


def _load_labels(root: Path) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for month in pd.date_range(TRAIN_START, OOS_END, freq="MS", inclusive="left"):
        path = root / "parts" / f"month={month:%Y-%m}" / "side=short.parquet"
        if not path.exists():
            raise FileNotFoundError(f"missing exact short label partition: {path}")
        parts.append(pd.read_parquet(path))
    out = pd.concat(parts, ignore_index=True)
    out["__ts__"] = _utc(out["__ts__"])
    out["__decision_ts__"] = _utc(out["__decision_ts__"])
    out["__label_available_at__"] = _utc(out["__label_available_at__"])
    if out.candidate_id.duplicated().any():
        raise ValueError("short label candidate IDs are not unique")
    if not out.side_name.astype(str).str.lower().eq("short").all():
        raise ValueError("label artifact contains a non-short row")
    return out


def _load_feature_contract(path: Path) -> list[str]:
    payload = json.loads(path.read_text())
    fields = [str(value) for value in payload["base_fields_by_side"]["short"]]
    if len(fields) != 120 or len(fields) != len(set(fields)):
        raise ValueError("short base feature contract must contain 120 unique fields")
    return fields


def _load_features(path: Path, fields: list[str]) -> pd.DataFrame:
    columns = ["candidate_id", "__ts__", "__decision_ts__", "__symbol__", "side_name", *fields]
    # The forward materializer attaches target-free identities.  Keeping this
    # contract candidate-keyed prevents an accidental signal/decision shift.
    frame = pd.read_parquet(path, columns=columns)
    frame["__ts__"] = _utc(frame["__ts__"])
    frame["__decision_ts__"] = _utc(frame["__decision_ts__"])
    if frame.candidate_id.duplicated().any():
        raise ValueError("short feature candidate IDs are not unique")
    if not frame.side_name.astype(str).str.lower().eq("short").all():
        raise ValueError("feature artifact contains a non-short row")
    return frame


def _load_candidates(path: Path) -> pd.DataFrame:
    """Load the complete target-free grid and retain only score-time facts.

    ``entry_executable`` is determined before any future H12 path is known.
    It is therefore the appropriate population for an inference-contract
    coverage gate: rejected rows remain in the target-free candidate/rejection
    artifacts, but cannot dilute the feature contract of models which never
    score them.
    """
    columns = [
        "candidate_id", "__ts__", "__decision_ts__", "__symbol__", "side_name",
        "entry_executable", "eligibility_reason",
    ]
    frame = pd.read_parquet(path, columns=columns)
    frame["__ts__"] = _utc(frame["__ts__"])
    frame["__decision_ts__"] = _utc(frame["__decision_ts__"])
    if frame.candidate_id.duplicated().any():
        raise ValueError("target-free candidate IDs are not unique")
    if not frame.side_name.astype(str).str.lower().eq("short").all():
        raise ValueError("target-free candidate artifact contains a non-short row")
    if frame.entry_executable.isna().any():
        raise ValueError("target-free candidate artifact has null entry_executable")
    return frame


def _causal_coverage_fields(
    candidate_population: pd.DataFrame,
    fields: list[str],
    *,
    minimum: float = 0.90,
) -> tuple[list[str], pd.Series]:
    """Choose fields on pre-decision, executable candidate coverage only.

    This explicitly excludes rows rejected on decision-time executability. It
    does *not* use ``label_valid``, outcomes, or path completeness, and hence
    cannot introduce future-path qualification into feature selection.
    """
    if not candidate_population["entry_executable"].any():
        raise ValueError("no entry-executable candidate exists for coverage")
    values = (
        candidate_population.loc[candidate_population["entry_executable"], fields]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
    )
    coverage = values.notna().mean()
    kept = [field for field in fields if float(coverage[field]) >= float(minimum)]
    return kept, coverage


def _r3_class(frame: pd.DataFrame) -> pd.Series:
    """Return canonical R3 class: adverse=0, weak=1, robust clear=2."""
    valid = (
        frame["label_valid"].astype("boolean").fillna(False).astype(bool)
        & ~frame["target_invalid"].astype("boolean").fillna(True).astype(bool)
    )
    event = pd.to_numeric(frame["t2_tp6_sl4_event"], errors="coerce")
    robust = pd.to_numeric(frame["robust_clear_event_b25"], errors="coerce").eq(1.0)
    result = pd.Series(np.nan, index=frame.index, dtype="float64")
    result.loc[valid] = 1.0
    result.loc[valid & event.eq(1.0)] = 0.0  # canonical adverse-first event
    result.loc[valid & robust] = 2.0
    return result


def _matrix(frame: pd.DataFrame, fields: list[str], medians: pd.Series) -> pd.DataFrame:
    matrix = frame.loc[:, fields].apply(pd.to_numeric, errors="coerce")
    matrix = matrix.replace([np.inf, -np.inf], np.nan).fillna(medians)
    if matrix.isna().any().any():
        raise ValueError("training-only median imputation left a non-finite base value")
    return matrix.astype("float32")


def _spearman(left: pd.Series, right: pd.Series) -> float:
    valid = left.notna() & right.notna() & np.isfinite(left) & np.isfinite(right)
    if int(valid.sum()) < 2:
        return float("nan")
    return float(left.loc[valid].corr(right.loc[valid], method="spearman"))


def _safe_auc(target: pd.Series, score: pd.Series) -> float:
    valid = target.notna() & score.notna() & np.isfinite(score)
    values = target.loc[valid].astype(int)
    if len(values) < 2 or values.nunique() < 2:
        return float("nan")
    return float(roc_auc_score(values, score.loc[valid]))


def _metrics(frame: pd.DataFrame, *, scope: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    valid = frame.loc[frame["r3_class"].notna()].copy()
    if valid.empty:
        raise ValueError(f"{scope}: no complete exact H12 labels")
    probs = valid[["p_adverse", "p_weak", "p_clear"]].to_numpy(np.float64)
    # Stored prediction probabilities are float32 for artifact size.  Restore
    # their exact simplex before probability-sensitive diagnostics; this does
    # not alter the raw score, ranks, classes, or economic tail metrics.
    probs /= probs.sum(axis=1, keepdims=True)
    metrics = {
        "scope": scope,
        "rows": int(len(valid)),
        "clear_rate": float(valid.r3_class.eq(2).mean()),
        "adverse_rate": float(valid.r3_class.eq(0).mean()),
        "multiclass_log_loss": float(log_loss(valid.r3_class.astype(int), probs, labels=[0, 1, 2])),
        "macro_f1": float(f1_score(valid.r3_class.astype(int), probs.argmax(axis=1), average="macro")),
        "clear_auc": _safe_auc(valid.r3_class.eq(2).astype(int), valid.p_clear),
        "adverse_auc": _safe_auc(valid.r3_class.eq(0).astype(int), valid.p_adverse),
        "score_r3_soft_spearman": _spearman(valid.base_score, valid.robust_clear_soft_b25_t50),
        "score_net_bps_spearman": _spearman(valid.base_score, valid.t4_tp6_sl4_net_bps),
    }
    rows: list[dict[str, Any]] = []
    ordered = valid.sort_values("base_score", ascending=False, kind="stable")
    for fraction in TAILS:
        take = max(1, int(math.ceil(len(ordered) * fraction)))
        tail = ordered.iloc[:take]
        rows.append({
            "scope": scope,
            "tail_fraction": fraction,
            "tail_rows": int(len(tail)),
            "mean_score": float(tail.base_score.mean()),
            "clear_rate": float(tail.r3_class.eq(2).mean()),
            "adverse_rate": float(tail.r3_class.eq(0).mean()),
            "mean_gross_bps": float(pd.to_numeric(tail.t4_tp6_sl4_gross_bps, errors="coerce").mean()),
            "mean_net_bps": float(pd.to_numeric(tail.t4_tp6_sl4_net_bps, errors="coerce").mean()),
            "median_net_bps": float(pd.to_numeric(tail.t4_tp6_sl4_net_bps, errors="coerce").median()),
        })
    return metrics, rows


def run(out: Path, feature_path: Path, labels_root: Path, candidate_path: Path) -> Path:
    prediction_path = out / "oos_predictions.parquet"
    if prediction_path.exists():
        raise FileExistsError(f"immutable OOS predictions already exist: {prediction_path}")
    out.mkdir(parents=True, exist_ok=True)
    fields = _load_feature_contract(FEATURE_CONTRACT)
    labels = _load_labels(labels_root)
    features = _load_features(feature_path, fields)
    candidates = _load_candidates(candidate_path)
    ledger = features.merge(labels, on=["candidate_id", "__ts__", "__decision_ts__", "__symbol__", "side_name"], how="left", validate="one_to_one")
    if len(ledger) != len(features):
        raise AssertionError("feature/label join changed point-in-time candidate identities")
    ledger = ledger.merge(
        candidates,
        on=["candidate_id", "__ts__", "__decision_ts__", "__symbol__", "side_name"],
        how="left",
        validate="one_to_one",
    )
    if len(ledger) != len(features) or ledger["entry_executable"].isna().any():
        raise AssertionError("feature ledger is not identical to target-free candidate identities")
    ledger["entry_executable"] = ledger["entry_executable"].astype(bool)
    ledger["r3_class"] = _r3_class(ledger)
    label_cutoff = OOS_START
    train_mask = (
        ledger["__ts__"].ge(TRAIN_START)
        & ledger["__ts__"].lt(OOS_START)
        & ledger["entry_executable"]
        & ledger["__label_available_at__"].lt(label_cutoff)
        & ledger["r3_class"].notna()
    )
    oos_mask = (
        ledger["__ts__"].ge(OOS_START)
        & ledger["__ts__"].lt(OOS_END)
        & ledger["entry_executable"]
    )
    train = ledger.loc[train_mask].copy()
    test = ledger.loc[oos_mask].copy()
    if train.empty or test.empty:
        raise ValueError("chronological train/OOS split is empty")
    # Feature availability is an inference-time property.  Fit it on the
    # complete *entry-executable* point-in-time candidate population, never on
    # rows whose future H12 labels later happened to resolve.  Rejected target-
    # free rows are retained in the candidate audit, but are not an inference
    # population because the live stack never scores them.
    train_candidate_population = ledger.loc[
        ledger["__decision_ts__"].ge(TRAIN_START)
        & ledger["__decision_ts__"].lt(OOS_START)
    ]
    kept, coverage = _causal_coverage_fields(
        train_candidate_population,
        fields,
        minimum=0.90,
    )
    if len(kept) != len(fields):
        missing = {
            field: float(coverage[field])
            for field in fields if field not in set(kept)
        }
        raise ValueError(
            "short frozen 120-field contract is incomplete on target-free "
            f"entry-executable training rows: {missing}"
        )
    medians = train.loc[:, kept].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).median()
    if medians.isna().any():
        raise ValueError("a kept short field has no training support")
    model = lgb.LGBMClassifier(**FROZEN_BASE_PARAMS).fit(
        _matrix(train, kept, medians), train.r3_class.astype(int).to_numpy(),
    )
    if model.classes_.tolist() != [0, 1, 2]:
        raise ValueError(f"unexpected R3 class order: {model.classes_.tolist()}")
    proba = np.asarray(model.predict_proba(_matrix(test, kept, medians)), dtype=np.float32)
    prediction = test.loc[:, [
        "candidate_id", "__ts__", "__decision_ts__", "__symbol__", "side_name",
        "label_valid", "target_invalid", "invalid_reason", "r3_class",
        "t2_tp6_sl4_event", "robust_clear_soft_b25_t50",
        "t4_tp6_sl4_gross_bps", "t4_tp6_sl4_net_bps",
    ]].copy()
    prediction["p_adverse"] = proba[:, 0]
    prediction["p_weak"] = proba[:, 1]
    prediction["p_clear"] = proba[:, 2]
    prediction["base_score"] = prediction.p_clear - 0.5 * prediction.p_adverse
    prediction.to_parquet(prediction_path, index=False, compression="zstd")
    model.booster_.save_model(str(out / "short_r3_base_frozen_long_hpo.model.txt"))

    metric_rows: list[dict[str, Any]] = []
    tail_rows: list[dict[str, Any]] = []
    total, tails = _metrics(prediction, scope="2024-04_to_2024-06")
    metric_rows.append(total)
    tail_rows.extend(tails)
    prediction["month"] = prediction["__ts__"].dt.strftime("%Y-%m")
    for month, group in prediction.groupby("month", sort=True):
        current, current_tails = _metrics(group, scope=str(month))
        metric_rows.append(current)
        tail_rows.extend(current_tails)
    pd.DataFrame(metric_rows).to_parquet(out / "metrics_by_scope.parquet", index=False, compression="zstd")
    pd.DataFrame(tail_rows).to_parquet(out / "metrics_by_scope_tail.parquet", index=False, compression="zstd")

    summary = {
        "schema": "strict_r3_short_r3_base_3m_train_3m_oos_v1",
        "status": "COMPLETE",
        "side": "short",
        "train_decision_window": f"[{TRAIN_START.isoformat()}, {OOS_START.isoformat()})",
        "strict_label_availability_gate": f"label_available_at < {OOS_START.isoformat()}",
        "oos_decision_window": f"[{OOS_START.isoformat()}, {OOS_END.isoformat()})",
        "target": "R3: adverse=0, weak=1, robust_clear=2; robust clear is pre-adverse MFE > fixed cost +25 bps, temperature 50 bps",
        "entry": "exact one-minute open at signal close +1 hour",
        "path": "H12 TP +6 ATR / SL -4 ATR; adverse wins a same-minute tie",
        "cost": "100 bps applied once to net labels",
        "frozen_hpo_source": "long R3 base frozen configuration; no short HPO was conducted",
        "model_params": FROZEN_BASE_PARAMS,
        "feature_contract": "short base_fields_by_side.short",
        "requested_feature_count": len(fields),
        "coverage_selected_feature_count": len(kept),
        "coverage_gate": (
            ">=90% on complete target-free, decision-time entry-executable "
            "training candidates; label validity/outcomes excluded"
        ),
        "coverage_population_rows": int(train_candidate_population.entry_executable.sum()),
        "coverage_by_feature": {field: float(coverage[field]) for field in fields},
        "selected_features": kept,
        "train_rows": int(len(train)),
        "oos_scored_rows": int(len(test)),
        "oos_label_valid_rows": int(prediction.r3_class.notna().sum()),
        "labels_sha256": _sha256(labels_root / "run_manifest.json"),
        "features_sha256": _sha256(feature_path),
        "feature_contract_sha256": _sha256(FEATURE_CONTRACT),
        "global_oos_metrics": total,
    }
    (out / "base_probe_manifest.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    args = parser.parse_args()
    print(run(args.out, args.features, args.labels, args.candidates))


if __name__ == "__main__":
    main()
