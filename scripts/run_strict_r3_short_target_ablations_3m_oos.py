#!/usr/bin/env python3
"""Compare causal short-base target families on one frozen 3m/3m split.

All arms use the same target-free executable candidates, 120-field short
contract, exact H12 outcome path and frozen base parameters.  The only moving
part is the supervised target.  The held April--June period is reported once
as target-selection evidence; no arm is promoted by this utility.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from scripts.run_strict_r3_short_base_3m_oos import (
    FEATURE_CONTRACT,
    FROZEN_BASE_PARAMS,
    OOS_END,
    OOS_START,
    TRAIN_START,
    _causal_coverage_fields,
    _load_candidates,
    _load_feature_contract,
    _load_features,
    _matrix,
    _safe_auc,
    _spearman,
    _utc,
)

DEFAULT_FEATURES = ROOT / "data_perp/artifacts/strict_r3_short_base_3m_train_3m_oos_2024_20260820_v5/features/canonical120_features.parquet"
DEFAULT_CANDIDATES = ROOT / "data_perp/artifacts/strict_r3_short_base_3m_train_3m_oos_2024_20260820_v1/short_target_free_candidate_population.parquet"
DEFAULT_LABELS = ROOT / "data_perp/artifacts/strict_r3_short_target_labels_2024_20260820_v1"
DEFAULT_OUT = ROOT / "data_perp/artifacts/strict_r3_short_target_ablations_3m_oos_2024_20260820_v1"

TAILS = (0.01, 0.02, 0.05, 0.10, 0.20, 0.30)
SEED = 17


@dataclass(frozen=True)
class TargetSpec:
    name: str
    family: str
    description: str
    hard_column: str | None = None
    soft_columns: tuple[str, ...] = ()
    ordinal_edges: tuple[float, float, float] | None = None


SPECS = (
    TargetSpec(
        "R0_r3_b25_hard_control", "multiclass_r3",
        "Existing hard R3 control: robust clear cost+25 / adverse-first / weak.",
    ),
    TargetSpec(
        "R1_cost_aware_clear_b50", "binary",
        "Cost-aware robust clear before meaningful adverse movement: cost +50 bps.",
        hard_column="r3_b50_robust_clear",
    ),
    TargetSpec(
        "R1_cost_aware_clear_b75", "binary",
        "Cost-aware robust clear before meaningful adverse movement: cost +75 bps.",
        hard_column="r3_b75_robust_clear",
    ),
    TargetSpec(
        "R3_soft_three_state_b50", "soft_r3",
        "Soft robust-clear/adverse-first/weak memberships; cost+50, margin, time-to-clear and same-bar ambiguity.",
        soft_columns=("r3_b50_soft_adverse", "r3_b50_soft_weak", "r3_b50_soft_clear"),
        hard_column="r3_b50_robust_clear",
    ),
    TargetSpec(
        "R3_soft_three_state_b75", "soft_r3",
        "Soft robust-clear/adverse-first/weak memberships; cost+75, margin, time-to-clear and same-bar ambiguity.",
        soft_columns=("r3_b75_soft_adverse", "r3_b75_soft_weak", "r3_b75_soft_clear"),
        hard_column="r3_b75_robust_clear",
    ),
    TargetSpec(
        "R4_binary_exact_net_gt50", "binary",
        "Direct economic control: exact H12 TP6/SL4 net outcome > +50 bps.",
        hard_column="__net_gt50__",
    ),
    TargetSpec(
        "R5_ordinal_n150_p25", "ordinal",
        "Ordinal net: <=-150, -150..0, 0..+25, >+25 bps.",
        ordinal_edges=(-150.0, 0.0, 25.0),
    ),
    TargetSpec(
        "R5_ordinal_n200_p50", "ordinal",
        "Ordinal net: <=-200, -200..0, 0..+50, >+50 bps.",
        ordinal_edges=(-200.0, 0.0, 50.0),
    ),
    TargetSpec(
        "R5_ordinal_n250_p75", "ordinal",
        "Ordinal net: <=-250, -250..0, 0..+75, >+75 bps.",
        ordinal_edges=(-250.0, 0.0, 75.0),
    ),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_labels(root: Path) -> pd.DataFrame:
    parts = []
    for month in pd.date_range(TRAIN_START, OOS_END, freq="MS", inclusive="left"):
        path = root / "parts" / f"month={month:%Y-%m}" / "side=short.parquet"
        if not path.exists():
            raise FileNotFoundError(path)
        parts.append(pd.read_parquet(path))
    out = pd.concat(parts, ignore_index=True)
    for column in ("__ts__", "__decision_ts__", "__label_available_at__"):
        out[column] = _utc(out[column])
    if out.candidate_id.duplicated().any() or not out.side_name.astype(str).str.lower().eq("short").all():
        raise ValueError("augmented short labels have invalid identities or side")
    return out


def _valid_label(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["label_valid"].astype("boolean").fillna(False).astype(bool)
        & ~frame["target_invalid"].astype("boolean").fillna(True).astype(bool)
    )


def _existing_r3(frame: pd.DataFrame) -> pd.Series:
    valid = _valid_label(frame)
    event = pd.to_numeric(frame["t2_tp6_sl4_event"], errors="coerce")
    robust = pd.to_numeric(frame["robust_clear_event_b25"], errors="coerce").eq(1.0)
    result = pd.Series(np.nan, index=frame.index, dtype="float64")
    result.loc[valid] = 1.0
    result.loc[valid & event.eq(1.0)] = 0.0
    result.loc[valid & robust] = 2.0
    return result


def _target_values(frame: pd.DataFrame, spec: TargetSpec) -> pd.DataFrame | pd.Series:
    valid = _valid_label(frame)
    net = pd.to_numeric(frame["t4_tp6_sl4_net_bps"], errors="coerce")
    if spec.family == "multiclass_r3":
        return _existing_r3(frame)
    if spec.family == "binary":
        if spec.hard_column == "__net_gt50__":
            target = net.gt(50.0)
        else:
            target = pd.to_numeric(frame[spec.hard_column], errors="coerce").eq(1.0)
        return target.where(valid, np.nan).astype("float64")
    if spec.family == "soft_r3":
        target = frame.loc[:, list(spec.soft_columns)].apply(pd.to_numeric, errors="coerce")
        target.columns = ["adverse", "weak", "clear"]
        target.loc[~valid, :] = np.nan
        sums = target.sum(axis=1)
        if not np.allclose(sums.loc[valid].to_numpy(float), 1.0, rtol=0.0, atol=2e-6):
            raise ValueError(f"{spec.name}: soft memberships do not sum to one")
        return target
    if spec.family == "ordinal":
        lower, middle, upper = spec.ordinal_edges or (np.nan, np.nan, np.nan)
        target = pd.Series(np.nan, index=frame.index, dtype="float64")
        target.loc[valid & net.le(lower)] = 0.0
        target.loc[valid & net.gt(lower) & net.le(middle)] = 1.0
        target.loc[valid & net.gt(middle) & net.le(upper)] = 2.0
        target.loc[valid & net.gt(upper)] = 3.0
        return target
    raise ValueError(f"unsupported target family: {spec.family}")


def _base_params(objective: str, *, num_class: int | None = None) -> dict[str, Any]:
    params = dict(FROZEN_BASE_PARAMS)
    params["objective"] = objective
    if num_class is None:
        params.pop("num_class", None)
    else:
        params["num_class"] = int(num_class)
    return params


def _fit_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    fields: list[str],
    medians: pd.Series,
    spec: TargetSpec,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    target = _target_values(train, spec)
    x_train = _matrix(train, fields, medians)
    x_test = _matrix(test, fields, medians)
    if spec.family in {"binary", "multiclass_r3", "ordinal"}:
        classes = 2 if spec.family == "binary" else (3 if spec.family == "multiclass_r3" else 4)
        if not isinstance(target, pd.Series):
            raise TypeError("hard target expected a Series")
        y = target.astype(int).to_numpy()
        model = lgb.LGBMClassifier(**_base_params("binary" if classes == 2 else "multiclass", num_class=None if classes == 2 else classes))
        model.fit(x_train, y)
        probabilities = np.asarray(model.predict_proba(x_test), dtype=np.float32)
        if classes == 2:
            score = probabilities[:, 1]
        elif classes == 3:
            score = probabilities[:, 2] - 0.5 * probabilities[:, 0]
        else:
            score = probabilities @ np.asarray([-2.0, -1.0, 0.5, 1.0], dtype=np.float32)
        return score.astype(np.float32), probabilities, {"model": model, "target": target}
    if spec.family == "soft_r3":
        if not isinstance(target, pd.DataFrame):
            raise TypeError("soft R3 target expected a DataFrame")
        outputs = []
        models = []
        for column in ("adverse", "weak", "clear"):
            model = lgb.LGBMRegressor(**_base_params("regression_l2"))
            model.fit(x_train, target[column].to_numpy(dtype=np.float32))
            outputs.append(np.clip(model.predict(x_test), 0.0, None))
            models.append(model)
        probabilities = np.column_stack(outputs).astype(np.float32)
        normalizer = probabilities.sum(axis=1, keepdims=True)
        probabilities /= np.where(normalizer > 1e-8, normalizer, 1.0)
        # a zero three-head output is an unequivocal model error, never a
        # hidden arbitrary class assignment.
        if np.any(normalizer[:, 0] <= 1e-8):
            raise ValueError(f"{spec.name}: soft R3 model produced an all-zero simplex")
        score = probabilities[:, 2] - 0.5 * probabilities[:, 0]
        return score.astype(np.float32), probabilities, {"models": models, "target": target}
    raise ValueError(spec.family)


def _metrics(
    frame: pd.DataFrame,
    spec: TargetSpec,
    *, scope: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    valid = _valid_label(frame)
    evaluation = frame.loc[valid].copy()
    target = _target_values(evaluation, spec)
    result: dict[str, Any] = {
        "spec": spec.name,
        "family": spec.family,
        "scope": scope,
        "scored_executable_rows": int(len(frame)),
        "resolved_rows": int(valid.sum()),
        "resolved_fraction": float(valid.mean()),
        "score_net_bps_spearman": _spearman(
            evaluation["score"], pd.to_numeric(evaluation["t4_tp6_sl4_net_bps"], errors="coerce"),
        ),
    }
    if spec.family == "binary":
        if not isinstance(target, pd.Series):
            raise TypeError("binary metric target")
        result["target_auc"] = _safe_auc(target, evaluation["score"])
        result["target_log_loss"] = float(log_loss(target.astype(int), np.clip(evaluation["p1"], 1e-6, 1.0 - 1e-6)))
        result["score_target_spearman"] = _spearman(evaluation["score"], target)
    elif spec.family == "soft_r3":
        if not isinstance(target, pd.DataFrame):
            raise TypeError("soft metric target")
        result["target_auc"] = _safe_auc(
            pd.to_numeric(evaluation[spec.hard_column], errors="coerce").eq(1.0).astype(int), evaluation["p_clear"],
        )
        result["target_log_loss"] = float("nan")
        result["score_target_spearman"] = _spearman(evaluation["score"], target["clear"] - 0.5 * target["adverse"])
    elif spec.family in {"multiclass_r3", "ordinal"}:
        if not isinstance(target, pd.Series):
            raise TypeError("multiclass metric target")
        probability_columns = (
            ["p_adverse", "p_weak", "p_clear"]
            if spec.family == "multiclass_r3"
            else sorted(
                [
                    column for column in evaluation.columns
                    if column.startswith("p") and column[1:].isdigit()
                ],
                key=lambda column: int(column[1:]),
            )
        )
        result["target_auc"] = float("nan")
        result["target_log_loss"] = float(log_loss(
            target.astype(int), evaluation[probability_columns].to_numpy(),
            labels=list(range(len(probability_columns))),
        ))
        result["score_target_spearman"] = _spearman(evaluation["score"], target)
    tails: list[dict[str, Any]] = []
    # Crucially rank every *scored executable* candidate first.  Resolve its
    # H12 outcome only afterwards, so a missing future path never changes the
    # contemporaneous candidate ranking or tail membership.
    ordered = frame.sort_values("score", ascending=False, kind="stable")
    for fraction in TAILS:
        selected = ordered.iloc[: max(1, int(math.ceil(len(ordered) * fraction)))]
        resolved = selected.loc[_valid_label(selected)]
        tails.append({
            "spec": spec.name,
            "family": spec.family,
            "scope": scope,
            "tail_fraction": fraction,
            "tail_rows_requested": int(len(selected)),
            "tail_rows_resolved": int(len(resolved)),
            "tail_label_coverage": float(len(resolved) / len(selected)),
            "mean_score": float(selected.score.mean()),
            "mean_gross_bps": float(pd.to_numeric(resolved.t4_tp6_sl4_gross_bps, errors="coerce").mean()),
            "mean_net_bps": float(pd.to_numeric(resolved.t4_tp6_sl4_net_bps, errors="coerce").mean()),
            "median_net_bps": float(pd.to_numeric(resolved.t4_tp6_sl4_net_bps, errors="coerce").median()),
            "clear_rate": float(pd.to_numeric(resolved.get("r3_b50_robust_clear"), errors="coerce").mean()),
        })
    return result, tails


def _prediction_columns(probabilities: np.ndarray) -> dict[str, np.ndarray]:
    if probabilities.shape[1] == 2:
        return {"p0": probabilities[:, 0], "p1": probabilities[:, 1]}
    if probabilities.shape[1] == 3:
        return {
            "p_adverse": probabilities[:, 0], "p_weak": probabilities[:, 1], "p_clear": probabilities[:, 2],
        }
    return {f"p{column}": probabilities[:, column] for column in range(probabilities.shape[1])}


def run(*, features_path: Path, labels_root: Path, candidates_path: Path, out: Path) -> Path:
    if out.exists():
        raise FileExistsError(f"output must be new: {out}")
    out.mkdir(parents=True)
    fields = _load_feature_contract(FEATURE_CONTRACT)
    features = _load_features(features_path, fields)
    candidates = _load_candidates(candidates_path)
    features = features.merge(
        candidates,
        on=["candidate_id", "__ts__", "__decision_ts__", "__symbol__", "side_name"],
        how="left", validate="one_to_one",
    )
    if features.entry_executable.isna().any():
        raise AssertionError("feature panel is not identical to target-free candidates")
    features["entry_executable"] = features.entry_executable.astype(bool)
    features = features.loc[
        features.__ts__.ge(TRAIN_START) & features.__ts__.lt(OOS_END) & features.entry_executable
    ].copy()
    labels = _load_labels(labels_root)
    ledger = features.merge(
        labels,
        on=["candidate_id", "__ts__", "__decision_ts__", "__symbol__", "side_name"],
        how="left", validate="one_to_one",
    )
    if len(ledger) != len(features):
        raise AssertionError("label merge changed target-free executable identities")
    coverage_population = ledger.loc[ledger.__decision_ts__.ge(TRAIN_START) & ledger.__decision_ts__.lt(OOS_START)]
    kept, coverage = _causal_coverage_fields(coverage_population, fields)
    if kept != fields:
        missing = {field: float(coverage[field]) for field in fields if field not in set(kept)}
        raise ValueError(f"frozen 120-field short coverage contract incomplete: {missing}")
    train = ledger.loc[
        ledger.__ts__.ge(TRAIN_START)
        & ledger.__ts__.lt(OOS_START)
        & _valid_label(ledger)
        & ledger.__label_available_at__.lt(OOS_START)
    ].copy()
    test = ledger.loc[ledger.__ts__.ge(OOS_START) & ledger.__ts__.lt(OOS_END)].copy()
    if train.empty or test.empty:
        raise ValueError("empty strict train/OOS population")
    medians = train.loc[:, fields].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).median()
    if medians.isna().any():
        raise ValueError("full short contract has a field without a training median")
    summary: list[dict[str, Any]] = []
    tails: list[dict[str, Any]] = []
    for spec in SPECS:
        score, probabilities, fitted = _fit_predict(train, test, fields, medians, spec)
        prediction = test.loc[:, [
            "candidate_id", "__ts__", "__decision_ts__", "__symbol__", "side_name",
            "label_valid", "target_invalid", "invalid_reason", "t4_tp6_sl4_gross_bps", "t4_tp6_sl4_net_bps",
            "t2_tp6_sl4_event", "robust_clear_event_b25",
            "r3_b50_robust_clear", "r3_b75_robust_clear",
            "r3_b50_soft_adverse", "r3_b50_soft_weak", "r3_b50_soft_clear",
            "r3_b75_soft_adverse", "r3_b75_soft_weak", "r3_b75_soft_clear",
        ]].copy()
        prediction["score"] = score
        for name, values in _prediction_columns(probabilities).items():
            prediction[name] = values
        prediction.to_parquet(out / f"oos_predictions_{spec.name}.parquet", index=False, compression="zstd")
        if "model" in fitted:
            fitted["model"].booster_.save_model(str(out / f"model_{spec.name}.txt"))
        else:
            for name, model in zip(("adverse", "weak", "clear"), fitted["models"], strict=True):
                model.booster_.save_model(str(out / f"model_{spec.name}_{name}.txt"))
        total, total_tails = _metrics(prediction, spec, scope="2024-04_to_2024-06")
        summary.append(total)
        tails.extend(total_tails)
        prediction["month"] = prediction.__ts__.dt.strftime("%Y-%m")
        for month, group in prediction.groupby("month", sort=True):
            current, current_tails = _metrics(group, spec, scope=str(month))
            summary.append(current)
            tails.extend(current_tails)
    pd.DataFrame(summary).to_parquet(out / "metrics_by_scope.parquet", index=False, compression="zstd")
    pd.DataFrame(tails).to_parquet(out / "metrics_by_scope_tail.parquet", index=False, compression="zstd")
    manifest = {
        "schema": "strict_r3_short_target_ablations_3m_oos_v1",
        "status": "complete",
        "side": "short",
        "train_decision_window": f"[{TRAIN_START.isoformat()}, {OOS_START.isoformat()})",
        "strict_label_availability_gate": f"label_available_at < {OOS_START.isoformat()}",
        "oos_decision_window": f"[{OOS_START.isoformat()}, {OOS_END.isoformat()})",
        "entry": "exact one-minute open at signal close + one hour",
        "outcome": "exact H12 TP6/SL4 gross and net; 100 bps cost once",
        "feature_contract": "short base_fields_by_side.short",
        "feature_count": len(fields),
        "coverage_gate": ">=90% on target-free entry-executable training candidates only",
        "coverage_by_feature": {field: float(coverage[field]) for field in fields},
        "training_rows": int(len(train)),
        "oos_scored_executable_rows": int(len(test)),
        "frozen_model_parameters": FROZEN_BASE_PARAMS,
        "specifications": [spec.__dict__ for spec in SPECS],
        "features_sha256": _sha256(features_path),
        "labels_manifest_sha256": _sha256(labels_root / "run_manifest.json"),
        "candidates_sha256": _sha256(candidates_path),
        "feature_contract_sha256": _sha256(FEATURE_CONTRACT),
        "selection_note": "April-June is the sole held comparison; no LambdaRank or production promotion is implied.",
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    print(run(features_path=args.features, labels_root=args.labels, candidates_path=args.candidates, out=args.out))


if __name__ == "__main__":
    main()
