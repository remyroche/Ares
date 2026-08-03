#!/usr/bin/env python3
"""Freeze a transition model on comparable 2022--25 evidence, then test 2026 once.

Transition probabilities and lifecycle probabilities are distinct from the
current-regime state.  Ex-post phase is a target/attribution only: it is never
an input, selection quota, trading gate, or model-selection field.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, average_precision_score, brier_score_loss, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "data_perp/artifacts"
CATALOGUE = ART / "transition_pattern_catalogue_20260730_v6/adaptive_phase_labels.parquet"
CURRENT = ART / "current_exact_policy_global_book_mapping_source_20260730_v3/causal_mapped_candidates.parquet"
OUT = ART / "strict_forward_transition_evaluation_20260730_v1"
TRAIN_END = pd.Timestamp("2026-01-01", tz="UTC")
MAX_FEATURES = 48


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def safe(value: Any) -> Any:
    if isinstance(value, (Path, pd.Timestamp)): return str(value)
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, dict): return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [safe(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value): return None
    return value


def global_top10(frame: pd.DataFrame, score: str) -> pd.Series:
    selected = pd.Series(False, index=frame.index)
    valid = frame.loc[pd.to_numeric(frame[score], errors="coerce").notna()]
    count = max(1, math.ceil(len(valid) * .10))
    selected.loc[valid.sort_values([score, "candidate_id"], ascending=[False, True], kind="stable").index[:count]] = True
    return selected


def causal_feature_columns(frame: pd.DataFrame, train: pd.DataFrame) -> list[str]:
    excluded = {"source_utc", "execution_decision_utc", "calendar_segment_id", "source_segment_id", "segment_id"}
    candidates = [column for column in frame.columns if column not in excluded and not column.startswith(("target__", "state_context__", "source_artifact")) and pd.api.types.is_numeric_dtype(frame[column])]
    numeric = train.loc[:, candidates].apply(pd.to_numeric, errors="coerce")
    usable = [column for column in candidates if numeric[column].notna().mean() >= .80 and np.isfinite(numeric[column].var()) and numeric[column].var() > 1e-12]
    return sorted(usable, key=lambda column: (-float(numeric[column].var()), column))[:MAX_FEATURES]


def label_available(frame: pd.DataFrame) -> pd.Series:
    floor = frame["source_utc"] + pd.Timedelta(hours=12)
    available = pd.to_datetime(frame["target__available_utc"], utc=True, errors="coerce").fillna(floor)
    phase_available = pd.to_datetime(frame["target__pattern_phase_available_utc"], utc=True, errors="coerce").fillna(floor)
    return pd.Series(np.maximum(available.to_numpy("datetime64[ns]"), phase_available.to_numpy("datetime64[ns]")), index=frame.index).dt.tz_localize("UTC")


def ece(y: pd.Series, probability: pd.Series, bins: int = 10) -> float:
    data = pd.DataFrame({"y": y, "p": probability}).dropna()
    if data.empty: return np.nan
    data["bin"] = pd.cut(data.p.clip(0, 1), bins=np.linspace(0, 1, bins + 1), include_lowest=True)
    grouped = data.groupby("bin", observed=True).agg(n=("y", "size"), observed=("y", "mean"), predicted=("p", "mean"))
    return float((grouped.n / len(data) * (grouped.observed - grouped.predicted).abs()).sum())


def binary_metrics(frame: pd.DataFrame, *, scope: str) -> dict[str, Any]:
    data = frame.loc[pd.to_numeric(frame["target__transition_active"], errors="coerce").notna()].copy()
    y = pd.to_numeric(data["target__transition_active"], errors="coerce").astype(int)
    p = data["transition_probability"].astype(float)
    return {"scope": scope, "rows": len(data), "positive_rate": float(y.mean()), "roc_auc": float(roc_auc_score(y, p)) if y.nunique() == 2 else np.nan, "average_precision": float(average_precision_score(y, p)) if y.nunique() == 2 else np.nan, "brier": float(brier_score_loss(y, p)), "ece10": ece(y, p)}


def run(*, catalogue: Path = CATALOGUE, current: Path = CURRENT, output: Path = OUT, regime_forward: Path | None = None) -> dict[str, Any]:
    if output.exists(): raise FileExistsError(output)
    frame = pd.read_parquet(catalogue).copy(); frame["source_utc"] = pd.to_datetime(frame.source_utc, utc=True, errors="raise")
    latest_candidate = pd.read_parquet(current, columns=["__ts__"])["__ts__"].max(); latest_candidate = pd.to_datetime(latest_candidate, utc=True)
    resolved = label_available(frame)
    train = frame.loc[frame.source_utc.lt(TRAIN_END) & resolved.lt(TRAIN_END)].copy()
    test = frame.loc[frame.source_utc.ge(TRAIN_END) & frame.source_utc.le(latest_candidate)].copy()
    features = causal_feature_columns(frame, train)
    if not features: raise ValueError("no causal transition features")
    imputer = SimpleImputer(strategy="median"); x_train = imputer.fit_transform(train[features]); x_test = imputer.transform(test[features])
    active_y = pd.to_numeric(train.target__transition_active, errors="coerce").fillna(0).astype(int)
    active = HistGradientBoostingClassifier(max_iter=120, max_leaf_nodes=15, learning_rate=.06, l2_regularization=2.0, random_state=20260730).fit(x_train, active_y)
    active_prob = active.predict_proba(x_test)[:, list(active.classes_).index(1)]
    phase_train = train.loc[train.target__pattern_phase.notna()].copy(); phase_x = imputer.transform(phase_train[features]); phase_y = phase_train.target__pattern_phase.astype(str)
    lifecycle = HistGradientBoostingClassifier(max_iter=120, max_leaf_nodes=15, learning_rate=.06, l2_regularization=2.0, random_state=20260731).fit(phase_x, phase_y)
    lifecycle_prob = lifecycle.predict_proba(x_test); classes = lifecycle.classes_.astype(str)
    predicted = test.loc[:, ["source_utc", "target__transition_active", "target__pattern_phase", "target__transition_archetype"]].copy()
    predicted["transition_probability"] = active_prob
    predicted["transition_entropy"] = -(np.clip(active_prob, 1e-12, 1) * np.log(np.clip(active_prob, 1e-12, 1)) + np.clip(1-active_prob, 1e-12, 1) * np.log(np.clip(1-active_prob, 1e-12, 1))) / math.log(2)
    predicted["transition_confidence"] = 1 - predicted.transition_entropy
    predicted["lifecycle_predicted_phase"] = classes[np.argmax(lifecycle_prob, axis=1)]
    predicted["lifecycle_probability"] = lifecycle_prob.max(axis=1)
    predicted["lifecycle_entropy"] = -(np.clip(lifecycle_prob, 1e-12, 1) * np.log(np.clip(lifecycle_prob, 1e-12, 1))).sum(axis=1) / math.log(len(classes))
    for pos, label in enumerate(classes): predicted[f"lifecycle_probability__{label}"] = lifecycle_prob[:, pos]
    metrics = [binary_metrics(predicted, scope="all_2026")] + [binary_metrics(group, scope=f"month::{month}") for month, group in predicted.assign(month=predicted.source_utc.dt.strftime("%Y-%m")).groupby("month", sort=True)]
    phase_eval = predicted.loc[predicted.target__pattern_phase.notna()]
    phase_metric = pd.DataFrame([{ "scope": "all_2026", "rows": len(phase_eval), "accuracy": accuracy_score(phase_eval.target__pattern_phase.astype(str), phase_eval.lifecycle_predicted_phase), "macro_f1": f1_score(phase_eval.target__pattern_phase.astype(str), phase_eval.lifecycle_predicted_phase, average="macro", zero_division=0)}])
    phase_attribution = predicted.groupby("target__pattern_phase", dropna=False, as_index=False).agg(hours=("source_utc", "size"), observed_transition_rate=("target__transition_active", "mean"), mean_transition_probability=("transition_probability", "mean"), mean_transition_entropy=("transition_entropy", "mean"), mean_lifecycle_confidence=("lifecycle_probability", "mean"))
    candidates = pd.read_parquet(current, columns=["candidate_id", "__ts__", "side_name", "execution_net_ev_12h", "catboost__residual__without_hpo__all_features"])
    candidates["__ts__"] = pd.to_datetime(candidates["__ts__"], utc=True); candidates = candidates.loc[candidates.__ts__.le(predicted.source_utc.max())].copy(); candidates["month"] = candidates.__ts__.dt.strftime("%Y-%m")
    candidates["selected_global_top10"] = False
    for _, group in candidates.groupby("month", sort=True): candidates.loc[group.index, "selected_global_top10"] = global_top10(group, "catboost__residual__without_hpo__all_features")
    econ = candidates.loc[candidates.selected_global_top10].merge(predicted, left_on="__ts__", right_on="source_utc", how="inner", validate="many_to_one")
    econ["risk_decile"] = pd.qcut(econ.transition_probability.rank(method="first"), q=10, labels=False, duplicates="drop")
    economic = econ.groupby(["month", "risk_decile"], as_index=False).agg(selected_rows=("candidate_id", "size"), mean_net_bps=("execution_net_ev_12h", lambda x: float(x.mean()*1e4)), mean_transition_probability=("transition_probability", "mean"), observed_transition_rate=("target__transition_active", "mean"))
    support = pd.DataFrame([{ "split": "train", "start": train.source_utc.min(), "end": train.source_utc.max(), "rows": len(train), "resolved_rows": len(train), "features": len(features)}, {"split": "untouched_2026", "start": test.source_utc.min(), "end": test.source_utc.max(), "rows": len(test), "resolved_rows": int(test.target__transition_active.notna().sum()), "features": len(features)}, {"split": "economic_global_top10", "start": econ.__ts__.min(), "end": econ.__ts__.max(), "rows": len(econ), "resolved_rows": len(econ), "features": len(features)}])
    combined = pd.DataFrame([{ "status": "PENDING_REGIME_FORWARD_ARTIFACT", "reason": "combined regime+transition read-only ablation requires the separate regime forward output" }])
    if regime_forward is not None and regime_forward.exists():
        combined = pd.DataFrame([{ "status": "NOT_RUN_IN_TRANSITION_ONLY_SEAL", "reason": "regime artifact supplied after transition freeze; combine only in a new immutable follow-up artifact" }])
    stage = output.parent / f".{output.name}.{uuid.uuid4().hex}.stage"; stage.mkdir(parents=True, exist_ok=False)
    try:
        predicted.to_parquet(stage / "forward_transition_predictions.parquet", index=False, compression="zstd")
        pd.DataFrame(metrics).to_csv(stage / "discrimination_calibration_monthly.csv", index=False); phase_metric.to_csv(stage / "lifecycle_discrimination.csv", index=False); phase_attribution.to_csv(stage / "phase_path_attribution.csv", index=False); economic.to_csv(stage / "global_top10_economic_attribution.csv", index=False); support.to_csv(stage / "support.csv", index=False); combined.to_csv(stage / "combined_regime_transition_readiness.csv", index=False); (stage / "selected_features.json").write_text(json.dumps(features, indent=2)+"\n")
        manifest = {"schema": "strict_forward_transition_evaluation_v1", "research_only": True, "promotion_eligible": False, "train_contract": "only 2022-08-30 through 2025-12-31 comparable-lineage causal hourly fields; labels resolved strictly before 2026-01-01; fixed model after fitting", "test_contract": f"untouched 2026 through transition-label/economic overlap {predicted.source_utc.max()}", "separation_contract": "transition probability, uncertainty and lifecycle outputs exclude current-regime state fields; ex-post phase is target/attribution only and never an input, gate, or selection quota", "economic_contract": "one pooled global top10 per UTC month before joining transition outputs; diagnostic only", "combined_ablation": combined.iloc[0].to_dict(), "inputs_sha256": {"catalogue": sha256(catalogue), "current_candidates": sha256(current)}, "outputs_sha256": {path.name: sha256(path) for path in stage.iterdir() if path.is_file()}, "counts": {"train_rows": len(train), "test_rows": len(test), "economic_selected_rows": len(econ), "features": len(features)}}
        (stage / "manifest.json").write_text(json.dumps(safe(manifest), indent=2, sort_keys=True)+"\n"); (stage / "manifest.sha256").write_text(f"{sha256(stage/'manifest.json')}  manifest.json\n"); os.replace(stage, output)
    except Exception: shutil.rmtree(stage, ignore_errors=True); raise
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--catalogue", type=Path, default=CATALOGUE); parser.add_argument("--current", type=Path, default=CURRENT); parser.add_argument("--output", type=Path, default=OUT); parser.add_argument("--regime-forward", type=Path)
    args = parser.parse_args(argv); print(json.dumps(safe(run(catalogue=args.catalogue, current=args.current, output=args.output, regime_forward=args.regime_forward)), sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
