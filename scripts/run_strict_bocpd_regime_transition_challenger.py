#!/usr/bin/env python3
"""Resumable strict online-BOCPD challenger.

Modes are deliberately independent and bounded:
``context`` builds compact per-signal × horizon × split checkpoints;
``head`` selects exactly one supervised head from those checkpoints; and
``seal`` only merges frozen head bundles and assesses untouched 2026.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.bayesian_changepoint import BOCPDConfig, bocpd_student_t_run_summary
from extreme_price_movements.regime_transition_changepoint import CHANGEPOINT_INPUT_COLUMNS
from scripts.run_strict_forward_transition_evaluation import ART, CATALOGUE, CURRENT, TRAIN_END, ece, global_top10, safe, sha256

OUT = ART / "strict_bocpd_regime_transition_challenger_20260730_v2"
CHECKPOINTS = ART / "strict_bocpd_regime_transition_challenger_20260730_v2_checkpoints"
FOLDS = tuple(pd.Timestamp(value, tz="UTC") for value in ("2024-01-01", "2024-07-01", "2025-01-01", "2025-07-01"))
HEADS = (("stable_vs_transition", "target__transition_active"), ("onset_h1", "target__onset_within_1h"), ("onset_h3", "target__onset_within_3h"), ("onset_h6", "target__onset_within_6h"), ("onset_h12", "target__onset_within_12h"))
HORIZONS = (24, 48)
CS = (0.1, 1.0)
SPLITS = tuple([f"fold_{index:02d}" for index in range(len(FOLDS))] + ["forward"])
SUMMARY = ("change_probability", "run_length_mean", "run_length_q05", "run_length_entropy")


def _metric(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    y, p = np.asarray(y, dtype=int), np.asarray(p, dtype=float)
    return {"ap": float(average_precision_score(y, p)) if np.unique(y).size == 2 else np.nan, "auc": float(roc_auc_score(y, p)) if np.unique(y).size == 2 else np.nan, "brier": float(brier_score_loss(y, p)), "ece10": float(ece(pd.Series(y), pd.Series(p)))}


def _runs(frame: pd.DataFrame) -> list[np.ndarray]:
    segment = frame["source_segment_id"]
    times = pd.to_datetime(frame.source_utc, utc=True)
    result: list[np.ndarray] = []
    for _, positions in segment.groupby(segment, sort=False).groups.items():
        index = np.asarray(list(positions), dtype=int)
        index = index[np.argsort(times.iloc[index].to_numpy())]
        begin = 0
        for end in range(1, len(index) + 1):
            if end == len(index) or times.iloc[index[end]] - times.iloc[index[end - 1]] != pd.Timedelta(hours=1):
                result.append(index[begin:end]); begin = end
    return result


def _label_available(frame: pd.DataFrame) -> pd.Series:
    floor = frame.source_utc + pd.Timedelta(hours=12)
    target = pd.to_datetime(frame["target__available_utc"], utc=True, errors="coerce").fillna(floor)
    phase = pd.to_datetime(frame["target__pattern_phase_available_utc"], utc=True, errors="coerce").fillna(floor)
    return pd.Series(np.maximum(target.to_numpy("datetime64[ns]"), phase.to_numpy("datetime64[ns]")), index=frame.index).dt.tz_localize("UTC")


def _load(catalogue: Path, current: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    targets = [target for _, target in HEADS]
    columns = ["source_utc", "source_segment_id", "target__available_utc", "target__pattern_phase_available_utc", *targets, *CHANGEPOINT_INPUT_COLUMNS]
    frame = pd.read_parquet(catalogue, columns=columns)
    frame["source_utc"] = pd.to_datetime(frame.source_utc, utc=True)
    latest = pd.to_datetime(pd.read_parquet(current, columns=["__ts__"])["__ts__"].max(), utc=True)
    available = _label_available(frame)
    train = frame.loc[frame.source_utc.lt(TRAIN_END) & available.lt(TRAIN_END)].copy()
    test = frame.loc[frame.source_utc.ge(TRAIN_END) & available.le(latest)].copy()
    for _, target in HEADS:
        train[target] = pd.to_numeric(train[target], errors="coerce").fillna(0).astype(int)
        test[target] = pd.to_numeric(test[target], errors="coerce").fillna(0).astype(int)
    return frame, train.reset_index(drop=True), test.reset_index(drop=True)


def _scale(reference: pd.Series) -> tuple[float, float]:
    finite = pd.to_numeric(reference, errors="coerce").to_numpy(float)
    finite = finite[np.isfinite(finite)]
    if len(finite) < 64:
        return 0.0, 1.0
    q25, q75 = np.quantile(finite, (.25, .75))
    return float(np.median(finite)), max(float(q75 - q25), 1e-4)


def _signal_context(reference: pd.DataFrame, score: pd.DataFrame, *, signal: str, horizon: int) -> pd.DataFrame:
    """Use the tested BOCPD primitive on one signal, bounded to one vector."""
    median, scale = _scale(reference[signal])
    values = pd.to_numeric(score[signal], errors="coerce").to_numpy(float)
    values = np.clip((np.nan_to_num(values, nan=median) - median) / scale, -8.0, 8.0)
    result = np.full((len(score), 4), np.nan, dtype=np.float32)
    config = BOCPDConfig(expected_run_hours=horizon, max_run_hours=horizon * 2)
    for run in _runs(score):
        result[run] = bocpd_student_t_run_summary(values[run], config)
    prefix = f"bocpd__{signal}__"
    return pd.DataFrame({"source_utc": score.source_utc.to_numpy(), **{prefix + name: result[:, index] for index, name in enumerate(SUMMARY)}})


def _split_context(train: pd.DataFrame, test: pd.DataFrame, split: str) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    if split == "forward":
        return train, pd.concat([train, test], ignore_index=True), {"reference_end_exclusive": str(TRAIN_END), "score_scope": "train_plus_untouched_2026"}
    index = int(split.rsplit("_", 1)[1])
    start = FOLDS[index]
    stop = start + pd.DateOffset(months=6)
    reference = train.loc[train.source_utc.lt(start)]
    history = train.loc[train.source_utc.lt(stop)]
    return reference, history, {"reference_end_exclusive": str(start), "score_end_exclusive": str(stop), "score_scope": "fold_history"}


def _checkpoint_dir(root: Path, horizon: int, split: str) -> Path:
    return root / f"h{horizon}" / split


def build_context(*, catalogue: Path, current: Path, root: Path, horizon: int, split: str, signal: str | None = None) -> dict[str, Any]:
    if horizon not in HORIZONS or split not in SPLITS:
        raise ValueError("invalid predeclared BOCPD horizon or split")
    _, train, test = _load(catalogue, current)
    reference, score, contract = _split_context(train, test, split)
    folder = _checkpoint_dir(root, horizon, split)
    folder.mkdir(parents=True, exist_ok=True)
    selected_signals = (signal,) if signal is not None else CHANGEPOINT_INPUT_COLUMNS
    if signal is not None and signal not in CHANGEPOINT_INPUT_COLUMNS:
        raise ValueError("signal must be one predeclared BOCPD input")
    for name in selected_signals:
        path, sidecar = folder / f"{name}.parquet", folder / f"{name}.json"
        expected = {"schema": "strict_bocpd_context_signal_v2", "horizon": horizon, "split": split, "signal": name, "catalogue_sha256": sha256(catalogue), "current_sha256": sha256(current), "rows": len(score), "reference_rows": len(reference), **contract}
        if path.exists() and sidecar.exists() and json.loads(sidecar.read_text()) == safe(expected):
            continue
        stage = Path(tempfile.mkdtemp(dir=folder, prefix=f".{name}."))
        try:
            context = _signal_context(reference, score, signal=name, horizon=horizon)
            output = stage / path.name
            context.to_parquet(output, index=False, compression="zstd")
            (stage / sidecar.name).write_text(json.dumps(safe(expected), indent=2, sort_keys=True) + "\n")
            os.replace(output, path); os.replace(stage / sidecar.name, sidecar)
        finally:
            shutil.rmtree(stage, ignore_errors=True)
    rows: list[dict[str, Any]] = []
    for name in CHANGEPOINT_INPUT_COLUMNS:
        path, sidecar = folder / f"{name}.parquet", folder / f"{name}.json"
        if path.exists() and sidecar.exists():
            rows.append({**json.loads(sidecar.read_text()), "context_sha256": sha256(path)})
    summary = {"schema": "strict_bocpd_context_split_v2", "horizon": horizon, "split": split, "model_sample_cadence": "1h", "signals": rows, "complete": len(rows) == len(CHANGEPOINT_INPUT_COLUMNS)}
    (folder / "manifest.json").write_text(json.dumps(safe(summary), indent=2, sort_keys=True) + "\n")
    (folder / "manifest.sha256").write_text(f"{sha256(folder / 'manifest.json')}  manifest.json\n")
    return summary


def _combined_checkpoint(root: Path, *, horizon: int, split: str) -> pd.DataFrame:
    folder = _checkpoint_dir(root, horizon, split)
    manifest = json.loads((folder / "manifest.json").read_text())
    if not manifest.get("complete"):
        raise RuntimeError(f"missing BOCPD context checkpoint: h{horizon}/{split}")
    frames = [pd.read_parquet(folder / f"{signal}.parquet") for signal in CHANGEPOINT_INPUT_COLUMNS]
    result = frames[0]
    for frame in frames[1:]:
        result = result.merge(frame, on="source_utc", how="inner", validate="one_to_one")
    if len(result) != int(manifest["signals"][0]["rows"]):
        raise RuntimeError("checkpoint merge lost hourly rows")
    probability = result[[f"bocpd__{signal}__change_probability" for signal in CHANGEPOINT_INPUT_COLUMNS]]
    run_mean = result[[f"bocpd__{signal}__run_length_mean" for signal in CHANGEPOINT_INPUT_COLUMNS]]
    run_q05 = result[[f"bocpd__{signal}__run_length_q05" for signal in CHANGEPOINT_INPUT_COLUMNS]]
    entropy = result[[f"bocpd__{signal}__run_length_entropy" for signal in CHANGEPOINT_INPUT_COLUMNS]]
    result["bocpd__change_probability_mean"] = probability.mean(axis=1)
    result["bocpd__change_probability_max"] = probability.max(axis=1)
    result["bocpd__run_length_mean"] = run_mean.mean(axis=1)
    result["bocpd__run_length_q05"] = run_q05.mean(axis=1)
    result["bocpd__run_length_entropy"] = entropy.mean(axis=1)
    result["bocpd__signal_count"] = float(len(CHANGEPOINT_INPUT_COLUMNS))
    return result.loc[:, ["source_utc", *[name for name in result if name.startswith("bocpd__")]]]


def _features(context: pd.DataFrame, reference_rows: int) -> tuple[pd.DataFrame, float]:
    result = context.copy()
    threshold = float(np.nanquantile(result.iloc[:reference_rows]["bocpd__change_probability_max"], .95))
    changed = result["bocpd__change_probability_max"].ge(threshold).to_numpy()
    age, state = np.zeros(len(result), dtype=np.float32), np.zeros(len(result), dtype=np.int32)
    active, state_id = 0, 0
    for row, flag in enumerate(changed):
        if bool(flag): active, state_id = 0, state_id + 1
        else: active += 1
        age[row], state[row] = active, state_id
    result["bocpd__state_age_hours"] = age
    result["bocpd__persistent_state_id"] = state
    result["bocpd__is_persistent_24h"] = (age >= 24).astype(np.float32)
    result["bocpd__is_persistent_72h"] = (age >= 72).astype(np.float32)
    return result, threshold


def _fit(train: pd.DataFrame, test: pd.DataFrame, *, target: str, c: float) -> np.ndarray:
    features = [name for name in train if name.startswith("bocpd__") and name != "bocpd__persistent_state_id"]
    imputer = SimpleImputer(strategy="median")
    x, z = imputer.fit_transform(train[features]), imputer.transform(test[features])
    model = LogisticRegression(C=float(c), class_weight="balanced", max_iter=300, random_state=20260730).fit(x, train[target].astype(int))
    return model.predict_proba(z)[:, 1]


def _head_context(train: pd.DataFrame, context: pd.DataFrame, *, start: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    joined = train.loc[:, ["source_utc", *[target for _, target in HEADS]]].merge(context, on="source_utc", how="inner", validate="one_to_one")
    return joined.loc[joined.source_utc.lt(start)].reset_index(drop=True), joined.loc[joined.source_utc.ge(start)].reset_index(drop=True)


def run_head(*, catalogue: Path, current: Path, root: Path, head: str) -> dict[str, Any]:
    targets = dict(HEADS)
    if head not in targets:
        raise ValueError("head must be exactly one predeclared BOCPD head")
    folder = root / "heads"; folder.mkdir(parents=True, exist_ok=True)
    bundle, hpo_path, winner_path = folder / f"{head}.parquet", folder / f"{head}_hpo.csv", folder / f"{head}_winner.json"
    if bundle.exists() and hpo_path.exists() and winner_path.exists():
        return {"head": head, "status": "reused", "bundle_sha256": sha256(bundle)}
    _, train, test = _load(catalogue, current); target = targets[head]
    rows: list[dict[str, Any]] = []
    for horizon in HORIZONS:
        contexts: list[tuple[int, pd.DataFrame, pd.DataFrame]] = []
        for fold, start in enumerate(FOLDS):
            context, _ = _features(_combined_checkpoint(root, horizon=horizon, split=f"fold_{fold:02d}"), len(train.loc[train.source_utc.lt(start)]))
            contexts.append((fold, *_head_context(train.loc[train.source_utc.lt(start + pd.DateOffset(months=6))], context, start=start)))
        for c in CS:
            oof = []
            for fold, fit, score in contexts:
                if fit[target].nunique() == 2 and not score.empty:
                    probability = _fit(fit, score, target=target, c=c)
                    oof.append(pd.DataFrame({"fold": fold, "y": score[target], "probability": probability}))
            joined = pd.concat(oof, ignore_index=True)
            scores = pd.DataFrame([_metric(group.y, group.probability) for _, group in joined.groupby("fold")])
            rows.append({"head": head, "target": target, "expected_run_hours": horizon, "max_run_hours": horizon * 2, "logistic_c": c, "oof_rows": len(joined), "mean_ap": scores.ap.mean(), "mean_auc": scores.auc.mean(), "mean_brier": scores.brier.mean(), "mean_ece10": scores.ece10.mean(), "mean_composite": (scores.ap - scores.brier).mean(), "min_fold_composite": (scores.ap - scores.brier).min()})
    hpo = pd.DataFrame(rows)
    winner = hpo.sort_values(["mean_composite", "min_fold_composite", "mean_ap", "expected_run_hours"], ascending=[False, False, False, True], kind="stable").iloc[0].to_dict()
    horizon = int(winner["expected_run_hours"])
    final = _combined_checkpoint(root, horizon=horizon, split="forward")
    context, threshold = _features(final, len(train))
    joined = pd.concat([train.loc[:, ["source_utc", *targets.values()]], test.loc[:, ["source_utc", *targets.values()]]], ignore_index=True).merge(context, on="source_utc", how="inner", validate="one_to_one")
    probability = _fit(joined.iloc[:len(train)], joined.iloc[len(train):], target=target, c=float(winner["logistic_c"]))
    output = joined.iloc[len(train):].loc[:, ["source_utc", *[name for name in context if name.startswith("bocpd__")]]].copy()
    output["head"], output["target"], output["probability"] = head, test[target].to_numpy(), probability
    winner["frozen_change_threshold_q95"] = threshold
    stage = Path(tempfile.mkdtemp(dir=folder, prefix=f".{head}."))
    try:
        output.to_parquet(stage / bundle.name, index=False, compression="zstd")
        hpo.to_csv(stage / hpo_path.name, index=False)
        (stage / winner_path.name).write_text(json.dumps(safe(winner), indent=2, sort_keys=True) + "\n")
        os.replace(stage / bundle.name, bundle); os.replace(stage / hpo_path.name, hpo_path); os.replace(stage / winner_path.name, winner_path)
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    return {"head": head, "status": "built", "bundle_sha256": sha256(bundle), "winner": winner}


def _run_lengths(state: np.ndarray, times: pd.Series) -> np.ndarray:
    values, start = [], 0; ts = pd.to_datetime(times, utc=True).to_numpy()
    for end in range(1, len(state) + 1):
        if end == len(state) or state[end] != state[end - 1] or ts[end] - ts[end - 1] != np.timedelta64(1, "h"):
            values.append(end - start); start = end
    return np.asarray(values, dtype=int)


def _economics(current: Path, forward: pd.DataFrame, head: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidates = pd.read_parquet(current, columns=["candidate_id", "__ts__", "side_name", "execution_net_ev_12h", "catboost__residual__without_hpo__all_features"])
    candidates["__ts__"] = pd.to_datetime(candidates["__ts__"], utc=True)
    candidates = candidates.loc[candidates.__ts__.le(forward.source_utc.max())].copy(); candidates["month"] = candidates.__ts__.dt.strftime("%Y-%m"); candidates["side_name"] = candidates.side_name.astype(str).str.lower(); candidates["selected_global_top10"] = False
    for _, group in candidates.groupby("month", sort=True): candidates.loc[group.index, "selected_global_top10"] = global_top10(group, "catboost__residual__without_hpo__all_features")
    local_forward = forward.loc[
        forward["head"].eq(head), ["source_utc", "probability", "target"]
    ].copy()
    local_forward["head"] = head
    merged = candidates.loc[candidates.selected_global_top10].merge(local_forward, left_on="__ts__", right_on="source_utc", how="inner", validate="many_to_one")
    merged["risk_decile"] = merged.groupby(["month", "side_name"])["probability"].transform(lambda values: pd.qcut(values.rank(method="first"), 10, labels=False, duplicates="drop"))
    economics = merged.groupby(["head", "month", "side_name", "risk_decile"], as_index=False).agg(selected_rows=("candidate_id", "size"), mean_net_bps=("execution_net_ev_12h", lambda values: float(values.mean() * 1e4)), mean_probability=("probability", "mean"), observed_target=("target", "mean"))
    support = merged.groupby(["head", "month", "side_name"], as_index=False).agg(selected_rows=("candidate_id", "size"), exact_economic_rows=("execution_net_ev_12h", lambda values: int(values.notna().sum())), mean_net_bps=("execution_net_ev_12h", lambda values: float(values.mean() * 1e4)))
    return economics, support


def seal(*, catalogue: Path, current: Path, root: Path, output: Path) -> dict[str, Any]:
    if output.exists(): raise FileExistsError(output)
    _, train, test = _load(catalogue, current)
    bundles = root / "heads"
    missing = [head for head, _ in HEADS if not (bundles / f"{head}.parquet").exists()]
    if missing: raise RuntimeError(f"cannot seal before one-head bundles exist: {missing}")
    forward = pd.concat([pd.read_parquet(bundles / f"{head}.parquet") for head, _ in HEADS], ignore_index=True)
    hpo = pd.concat([pd.read_csv(bundles / f"{head}_hpo.csv") for head, _ in HEADS], ignore_index=True)
    winners = pd.DataFrame([json.loads((bundles / f"{head}_winner.json").read_text()) for head, _ in HEADS])
    metrics, support = [], []
    for head, local in forward.groupby("head", sort=True):
        pieces = [("all_2026", local), *[(f"month::{month}", group) for month, group in local.assign(month=local.source_utc.dt.strftime("%Y-%m")).groupby("month")]]
        for scope, piece in pieces:
            metrics.append({"head": head, "scope": scope, **_metric(piece.target, piece.probability)})
            support.append({"head": head, "scope": scope, "rows": len(piece), "positives": int(piece.target.sum()), "prevalence": float(piece.target.mean())})
    economics, sides = [], []
    for head, _ in HEADS:
        economic, side = _economics(current, forward, head); economics.append(economic); sides.append(side)
    state_context = forward.loc[forward["head"].eq("onset_h3")].copy(); dwell = []
    for month, group in state_context.assign(month=state_context.source_utc.dt.strftime("%Y-%m")).groupby("month"):
        runs = _run_lengths(group.bocpd__persistent_state_id.to_numpy(int), group.source_utc)
        dwell.append({"method": "bocpd_persistent_state", "month": month, "rows": len(group), "states": int(group.bocpd__persistent_state_id.nunique()), "median_dwell_hours": float(np.median(runs)), "mean_dwell_hours": float(np.mean(runs)), "state_change_fraction": float(group.bocpd__persistent_state_id.ne(group.bocpd__persistent_state_id.shift()).iloc[1:].mean())})
    comparison = pd.DataFrame(metrics); comparison["method"] = "bocpd_logistic_calibrator"
    for path, name in ((ART / "strict_forward_regime_only_2022aug_2025_to_2026_20260730_v3/2026_monthly_coverage_stability.csv", "diagonal_gmm_v3"), (ART / "strict_forward_sticky_fullcov_regime_challenger_2022aug_2025_to_2026_20260730_v1/2026_monthly_coverage_stability.csv", "sticky_fullcov_gmm_v1"), (ART / "strict_forward_dae_gmm_regime_challenger_2022aug_2025_to_2026_20260730_v1/2026_monthly_coverage_stability.csv", "dae_gmm_v1")):
        if path.exists():
            baseline = pd.read_csv(path); baseline["method"], baseline["head"], baseline["scope"] = name, "persistent_state", "month::" + baseline.month.astype(str); comparison = pd.concat([comparison, baseline], ignore_index=True, sort=False)
    stage = Path(tempfile.mkdtemp(dir=output.parent, prefix=f".{output.name}."))
    try:
        hpo.to_csv(stage / "train_only_bocpd_hpo.csv", index=False); winners.to_csv(stage / "frozen_bocpd_winners.csv", index=False); forward.to_parquet(stage / "forward_bocpd_regime_transition.parquet", index=False, compression="zstd"); pd.DataFrame(metrics).to_csv(stage / "untouched_2026_discrimination_calibration.csv", index=False); pd.DataFrame(support).to_csv(stage / "untouched_2026_monthly_support.csv", index=False); pd.concat(economics, ignore_index=True).to_csv(stage / "global_top10_economic_attribution.csv", index=False); pd.concat(sides, ignore_index=True).to_csv(stage / "global_top10_candidate_side_support.csv", index=False); pd.DataFrame(dwell).to_csv(stage / "bocpd_dwell_switching_2026.csv", index=False); comparison.to_csv(stage / "strict_method_comparison_2026.csv", index=False)
        manifest = {"schema": "strict_bocpd_regime_transition_challenger_v2", "status": "SEALED_STRICT_RESUMABLE_BOCPD", "research_only": True, "promotion_eligible": False, "model_sample_cadence": "1h", "assessment_sample_cadence": "1h", "exact_replay_bar_cadence": "1m_labels_only", "train_contract": "per-signal BOCPD scaling, 24/48h horizon, logistic calibration and alert threshold are selected on resolved 2022-2025 only", "test_contract": f"untouched 2026 through {forward.source_utc.max()}", "resumable_contract": "context checkpoint per signal×horizon×split; exactly one head/HPO bundle per invocation; merge-only final seal", "separation": "current GMM state, transition labels/phase, identity and post-entry outcomes are not BOCPD inputs", "economic_contract": "one pooled global top10 per UTC month before joining BOCPD outputs; attribution only", "inputs_sha256": {"catalogue": sha256(catalogue), "current": sha256(current)}, "checkpoint_manifests": {str(path.relative_to(root)): sha256(path) for path in sorted(root.glob("h*/**/manifest.json"))}, "head_bundles": {head: sha256(bundles / f"{head}.parquet") for head, _ in HEADS}, "outputs_sha256": {path.name: sha256(path) for path in stage.iterdir() if path.is_file()}, "counts": {"heads": len(HEADS), "signals": len(CHANGEPOINT_INPUT_COLUMNS), "train_rows": len(train), "test_rows_per_head": len(test)}}
        manifest_path = stage / "manifest.json"; manifest_path.write_text(json.dumps(safe(manifest), indent=2, sort_keys=True) + "\n"); (stage / "manifest.sha256").write_text(f"{sha256(manifest_path)}  manifest.json\n"); os.replace(stage, output)
        return manifest
    except Exception:
        shutil.rmtree(stage, ignore_errors=True); raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("context", "head", "seal"), required=True)
    parser.add_argument("--catalogue", type=Path, default=CATALOGUE); parser.add_argument("--current", type=Path, default=CURRENT); parser.add_argument("--checkpoints", type=Path, default=CHECKPOINTS); parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--horizon", type=int, choices=HORIZONS); parser.add_argument("--split", choices=SPLITS); parser.add_argument("--signal", choices=CHANGEPOINT_INPUT_COLUMNS); parser.add_argument("--head", choices=[head for head, _ in HEADS])
    args = parser.parse_args()
    if args.mode == "context":
        if args.horizon is None or args.split is None: parser.error("context requires --horizon and --split")
        result = build_context(catalogue=args.catalogue, current=args.current, root=args.checkpoints, horizon=args.horizon, split=args.split, signal=args.signal)
    elif args.mode == "head":
        if args.head is None: parser.error("head requires exactly one --head")
        result = run_head(catalogue=args.catalogue, current=args.current, root=args.checkpoints, head=args.head)
    else:
        result = seal(catalogue=args.catalogue, current=args.current, root=args.checkpoints, output=args.output)
    print(json.dumps(safe(result), sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
