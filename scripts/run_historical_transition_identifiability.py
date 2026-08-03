#!/usr/bin/env python3
"""Assess whether existing 2022--23 regime transitions are identifiable.

This is a non-walk-forward grouped-OOF diagnostic.  It is deliberately
separate from entry EV: a transition classifier earns diagnostic/controller
status only through transition-label discrimination and calibrated alert
behaviour, never through a mechanical blend into candidate ranking.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data_perp/artifacts/failure_2022_2023_pf_exact1m_transition_context_continuation_20260730_v1/frozen_v3_market_spine/hourly_transition_dataset.parquet"
OUT = ROOT / "data_perp/artifacts/historical_transition_identifiability_20260731_v1"
TARGETS = ("target__transition_active", "target__onset_within_3h", "target__onset_within_6h", "target__onset_within_12h")
META = {"source_utc", "execution_decision_utc", "segment_id", "target__pooled_state", "target__event_id", "target__phase", "target__destination_state", "target__transition_archetype", "target__time_to_onset_hours", "target__available_utc", *TARGETS}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def dump(path: Path, data: object) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(temporary, path)


def feature_sets(columns: list[str]) -> dict[str, list[str]]:
    numeric = [c for c in columns if c not in META]
    state = [c for c in numeric if c.startswith("state_context__")]
    regime = [c for c in numeric if c.startswith("mkt_regime_change__")]
    dynamic = [c for c in numeric if c.startswith("transition_new__")]
    breadth = [c for c in dynamic if any(t in c for t in ("breadth", "dispersion", "correlation", "fragmented", "relative_strength"))]
    flow = [c for c in dynamic if c not in breadth]
    static = [c for c in numeric if c not in set(state + regime + dynamic)]
    groups = {"state_geometry": state, "regime_change": regime, "breadth_correlation": breadth, "flow_recovery": flow, "static_mechanisms": static, "all_causal_context": numeric}
    if any(not x for x in groups.values()):
        raise ValueError("one of the predeclared mechanism groups is empty")
    return groups


def arrays(train: pd.DataFrame, test: pd.DataFrame, columns: list[str]) -> tuple[np.ndarray, np.ndarray]:
    x = train[columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    med = x.median().fillna(0.0)
    return x.fillna(med).to_numpy(np.float32), test[columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(med).to_numpy(np.float32)


def predict(train: pd.DataFrame, test: pd.DataFrame, columns: list[str], target: str, model: str, seed: int) -> np.ndarray:
    x, z = arrays(train, test, columns)
    y = train[target].astype(int).to_numpy()
    if model == "logistic":
        scale = StandardScaler().fit(x)
        estimator = LogisticRegression(C=.25, class_weight="balanced", max_iter=1500, random_state=seed).fit(scale.transform(x), y)
        return estimator.predict_proba(scale.transform(z))[:, 1]
    estimator = lgb.LGBMClassifier(n_estimators=180, learning_rate=.035, num_leaves=12, min_child_samples=80, colsample_bytree=.75, reg_lambda=8., class_weight="balanced", random_state=seed, n_jobs=4, verbosity=-1).fit(x, y)
    return estimator.predict_proba(z)[:, 1]


def metrics(frame: pd.DataFrame, target: str, group: str, model: str) -> dict[str, object]:
    y = frame["target_value"].astype(int).to_numpy()
    p = frame["probability"].to_numpy(float)
    count = max(1, int(np.ceil(len(frame) * .1)))
    selected = frame.sort_values(["probability", "source_utc"], ascending=[False, True], kind="stable").head(count)
    # An onset is caught when an alert occurs in the preceding three hours.
    actual = frame.loc[frame["target_value"].eq(1), "source_utc"]
    alerts = selected["source_utc"]
    caught = sum(alerts.between(t - pd.Timedelta(hours=3), t, inclusive="both").any() for t in actual)
    days = max((frame.source_utc.max() - frame.source_utc.min()) / pd.Timedelta(days=1), 1.0)
    false = sum(not actual.between(t, t + pd.Timedelta(hours=3), inclusive="both").any() for t in alerts)
    return {"target": target, "feature_group": group, "model": model, "rows": int(len(frame)), "positive_rows": int(y.sum()), "prevalence": float(y.mean()), "roc_auc": float(roc_auc_score(y, p)), "pr_auc": float(average_precision_score(y, p)), "brier": float(brier_score_loss(y, p)), "log_loss": float(log_loss(y, np.clip(p, 1e-8, 1 - 1e-8))), "top10_precision": float(selected["target_value"].mean()), "top10_lift": float(selected["target_value"].mean() / y.mean()) if y.mean() else float("nan"), "event_recall_at_top10": float(caught / len(actual)) if len(actual) else float("nan"), "false_alerts_per_30d": float(false / days * 30.0)}


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    data = pd.read_parquet(SOURCE)
    data["source_utc"] = pd.to_datetime(data["source_utc"], utc=True, errors="raise")
    data["target__available_utc"] = pd.to_datetime(data["target__available_utc"], utc=True, errors="raise")
    if (data.target__available_utc < data.source_utc).any():
        raise ValueError("a transition target resolves before the source timestamp")
    groups = feature_sets(list(data.columns))
    data["week_group"] = data.source_utc.dt.to_period("W-SUN").astype(str)
    splitter = GroupKFold(n_splits=5)
    splits = list(splitter.split(data, groups=data.week_group))
    output: list[pd.DataFrame] = []
    folds: list[dict[str, object]] = []
    for feature_group, columns in groups.items():
        for target in TARGETS:
            if data[target].nunique() != 2:
                continue
            for model in ("logistic", "lightgbm"):
                prediction = np.full(len(data), np.nan)
                for fold, (tr, ev) in enumerate(splits):
                    train, evaluate = data.iloc[tr], data.iloc[ev]
                    if train[target].nunique() != 2 or evaluate[target].nunique() != 2:
                        raise ValueError(f"non-stratified weekly fold for {target}")
                    prediction[ev] = predict(train, evaluate, columns, target, model, 20260731 + fold)
                    folds.append({"feature_group": feature_group, "target": target, "model": model, "fold": fold, "train_rows": int(len(tr)), "evaluation_rows": int(len(ev)), "train_positive": int(train[target].sum()), "evaluation_positive": int(evaluate[target].sum()), "validation": "non_walk_forward_week_group_oof"})
                if not np.isfinite(prediction).all():
                    raise ValueError("OOF coverage incomplete")
                output.append(data.loc[:, ["source_utc"]].assign(target=target, target_value=data[target].astype(np.int8), feature_group=feature_group, model=model, probability=prediction))
    predictions = pd.concat(output, ignore_index=True)
    report = []
    for (target, group, model), local in predictions.groupby(["target", "feature_group", "model"], sort=True):
        report.append(metrics(local, target, group, model))
    temp = Path(tempfile.mkdtemp(dir=OUT.parent, prefix=f".{OUT.name}."))
    try:
        predictions.to_parquet(temp / "oof_predictions.parquet", index=False)
        pd.DataFrame(folds).to_csv(temp / "fold_provenance.csv", index=False)
        pd.DataFrame(report).to_csv(temp / "metrics.csv", index=False)
        contract = {"evidence_scope": "non_walk_forward_grouped_oof_transition_identifiability", "promotion_eligible": False, "features": "all decision-known fields from frozen historical geometry; target columns and future geometry excluded", "targets": {target: "derived future-confirmed transition label; label availability must be after source timestamp" for target in TARGETS}, "use_rule": "diagnostic/controller only; never feed into exact-H12 entry EV without a separate successful incremental economic ablation", "groups": {name: len(columns) for name, columns in groups.items()}}
        dump(temp / "contract.json", contract)
        files = [temp / x for x in ("oof_predictions.parquet", "fold_provenance.csv", "metrics.csv", "contract.json")]
        manifest = {"schema": "historical_transition_identifiability_v1", "status": "COMPLETE_RESEARCH_ONLY", "source": {"path": str(SOURCE), "sha256": sha(SOURCE)}, "rows": int(len(data)), "coverage": [str(data.source_utc.min()), str(data.source_utc.max())], "outputs_sha256": {x.name: sha(x) for x in files}, **contract}
        dump(temp / "manifest.json", manifest)
        (temp / "manifest.sha256").write_text(f"{sha(temp / 'manifest.json')}  manifest.json\n")
        os.replace(temp, OUT)
    except Exception:
        shutil.rmtree(temp, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
