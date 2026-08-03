#!/usr/bin/env python3
"""Materialize causal OOF clean-opportunity and adverse-risk probabilities.

This is a research-only sidecar for the four shared context architectures.  It
uses exact historical 1m labels, but makes no claim that those frozen-backcast
labels are execution-parity evidence.  Every emitted probability is generated
by a side-local model fit only on labels resolved before its evaluation fold.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
IDENTITY = ("candidate_id", "__ts__", "__symbol__", "side_name")
ARCHITECTURES = ("baseline", "regime_only", "transition_only", "regime_plus_transition")
HEADS = ("clean_opportunity", "adverse_competing_risk")
TARGETS = {
    "clean_opportunity": "__opportunity_occurred_12h__",
    "adverse_competing_risk": "__adverse_competing_risk_12h__",
}
SCORE = "score_residual_expected_ev"
MAPPED_SCORE = "causal_recent_ev_mapped_score"
MAPPING_AVAILABLE = "causal_recent_ev_mapping_available"
LABEL_AVAILABLE = "__label_available_at__"
SCHEMA = "clean_competing_risk_probability_oof_sidecar_v1"
ACTION_TOKENS = ("timing", "time_to", "mae", "wait", "target_price", "targetprice", "entry_price", "suggested_price", "action_")
DEFAULT_STATE = ROOT / "data_perp/artifacts/reconstructed_2023apr_2024_candidate_oof_regime_transition_20260730_v1/candidate_oof_regime_transition.parquet"
DEFAULT_SCORES = ROOT / "data_perp/artifacts/reconstructed_base_residual_stack_2022_2024_20260730_v3/oof_scores.parquet"
DEFAULT_2023_LABELS = ROOT / "data_perp/artifacts/failure_2022_2023_pf_exact1m_multitask_labels_20260730_v1/joined_multitask_labels.parquet"
DEFAULT_2024_LABELS = ROOT / "data_perp/artifacts/failure_2024_exact1m_multitask_labels_20260730_v1/joined_multitask_labels.parquet"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/clean_competing_risk_probability_oof_2023_2024_20260730_v1"


def _safe(value: Any) -> Any:
    if isinstance(value, (np.generic,)):
        return value.item()
    if isinstance(value, (pd.Timestamp, Path)):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(frame: pd.DataFrame, *, source: str) -> pd.DataFrame:
    missing = sorted(set(IDENTITY).difference(frame.columns))
    if missing:
        raise ValueError(f"{source} lacks exact identity: {missing}")
    result = frame.copy()
    result["__ts__"] = pd.to_datetime(result["__ts__"], utc=True, errors="raise")
    result["side_name"] = result["side_name"].astype(str).str.lower()
    if not result["side_name"].isin(("long", "short")).all() or result.duplicated(list(IDENTITY)).any():
        raise ValueError(f"{source} has duplicate/noncanonical exact identities")
    return result


def forbid_action_features(features: Sequence[str]) -> tuple[str, ...]:
    names = tuple(map(str, features))
    bad = [name for name in names if any(token in name.lower() for token in ACTION_TOKENS)]
    if bad:
        raise ValueError("clean/competing probability heads exclude timing/MAE/target-price/wait action fields: " + ", ".join(sorted(bad)))
    return names


def architecture_feature_sets(columns: Sequence[str]) -> dict[str, tuple[str, ...]]:
    """Four explicit, nested feature pools; all keep the score/mapping anchor."""

    available = set(map(str, columns))
    anchor = (SCORE, MAPPED_SCORE, MAPPING_AVAILABLE)
    missing = sorted(set(anchor).difference(available))
    if missing:
        raise KeyError(f"probability sidecar lacks shared score/mapping anchor: {missing}")
    regime = tuple(sorted(name for name in available if name.startswith("regime_state_p__") or name in {
        "regime_state_entropy", "regime_state_margin", "regime_state_uncertainty", "regime_state_ood_score",
    }))
    transition = tuple(sorted(name for name in available if name.startswith("transition_state_p__") or name in {
        "transition_active_probability", "transition_state_entropy", "transition_state_margin", "transition_state_uncertainty", "transition_state_ood_score",
    }))
    if not regime or not transition:
        raise ValueError("reconstructed causal regime and transition feature sets are both required")
    pools = {
        "baseline": anchor,
        "regime_only": (*anchor, *regime),
        "transition_only": (*anchor, *transition),
        "regime_plus_transition": (*anchor, *regime, *transition),
    }
    return {name: forbid_action_features(values) for name, values in pools.items()}


def add_causal_recent_ev_mapping(frame: pd.DataFrame, *, lookback_days: int = 21, minimum_rows: int = 300) -> pd.DataFrame:
    """Daily side-local isotonic map using only already-resolved prior labels."""

    if lookback_days < 1 or minimum_rows < 10:
        raise ValueError("mapping lookback/minimum support is invalid")
    result = frame.copy()
    result[MAPPED_SCORE] = np.nan
    result[MAPPING_AVAILABLE] = np.int8(0)
    timestamps = pd.to_datetime(result["__ts__"], utc=True, errors="raise")
    available = pd.to_datetime(result[LABEL_AVAILABLE], utc=True, errors="raise")
    if not available.ge(timestamps).all():
        raise ValueError("a historical label resolves before its decision timestamp")
    for side in ("long", "short"):
        side_positions = np.flatnonzero(result["side_name"].eq(side).to_numpy())
        if not len(side_positions):
            continue
        ordered = side_positions[np.argsort(timestamps.iloc[side_positions].to_numpy(), kind="stable")]
        days = timestamps.iloc[ordered].dt.floor("D")
        for day in pd.Index(days.unique()).sort_values():
            predict = ordered[days.eq(day).to_numpy()]
            fit_mask = (
                timestamps.iloc[ordered].ge(day - pd.Timedelta(days=lookback_days)).to_numpy()
                & available.iloc[ordered].lt(day).to_numpy()
            )
            fit = ordered[fit_mask]
            if len(fit) < minimum_rows:
                continue
            x = result.iloc[fit][SCORE].to_numpy(float)
            y = result.iloc[fit]["execution_net_ev_12h"].to_numpy(float)
            finite = np.isfinite(x) & np.isfinite(y)
            if finite.sum() < minimum_rows or np.unique(x[finite]).size < 2:
                continue
            model = IsotonicRegression(out_of_bounds="clip")
            model.fit(x[finite], y[finite])
            values = result.iloc[predict][SCORE].to_numpy(float)
            usable = np.isfinite(values)
            result.loc[result.index[predict[usable]], MAPPED_SCORE] = model.predict(values[usable])
            result.loc[result.index[predict[usable]], MAPPING_AVAILABLE] = 1
    return result


def chronological_folds(frame: pd.DataFrame, *, first_evaluation: str, last_evaluation: str, frequency: str = "QS", minimum_train_months: int = 2) -> list[tuple[str, np.ndarray, np.ndarray, pd.Timestamp]]:
    source = pd.to_datetime(frame["__ts__"], utc=True, errors="raise")
    available = pd.to_datetime(frame[LABEL_AVAILABLE], utc=True, errors="raise")
    result: list[tuple[str, np.ndarray, np.ndarray, pd.Timestamp]] = []
    for start in pd.date_range(first_evaluation, last_evaluation, freq=frequency, tz="UTC"):
        end = start + pd.tseries.frequencies.to_offset(frequency)
        train = np.flatnonzero(available.lt(start).to_numpy())
        evaluation = np.flatnonzero(source.ge(start).to_numpy() & source.lt(end).to_numpy())
        if not len(evaluation):
            continue
        months = source.iloc[train].dt.tz_localize(None).dt.to_period("M")
        if len(train) == 0 or months.nunique() < minimum_train_months:
            continue
        if available.iloc[train].max() >= start:
            raise AssertionError("chronological head fold includes unresolved training labels")
        result.append((f"{start:%Y%m%d}_{end:%Y%m%d}", train, evaluation, start))
    return result


def _select_train_features(matrix: pd.DataFrame, target: np.ndarray, *, maximum: int = 24) -> list[str]:
    """Small deterministic target-specific screen fitted on a fold only."""

    y = np.asarray(target, dtype=float)
    scores: list[tuple[float, str]] = []
    for name in matrix.columns:
        values = pd.to_numeric(matrix[name], errors="coerce")
        good = values.notna().to_numpy() & np.isfinite(y)
        if good.sum() < 20 or values.iloc[np.flatnonzero(good)].nunique() < 2:
            score = -np.inf
        else:
            score = abs(float(pd.Series(values.to_numpy()[good]).corr(pd.Series(y[good]), method="spearman")))
            if not np.isfinite(score):
                score = -np.inf
        scores.append((score, str(name)))
    selected = [name for score, name in sorted(scores, key=lambda item: (-item[0], item[1]))[:maximum] if np.isfinite(score)]
    if not selected:
        raise ValueError("no finite train-only probability-head features")
    return selected


def _probability_metrics(target: np.ndarray, probability: np.ndarray) -> tuple[dict[str, float], pd.DataFrame]:
    y, p = np.asarray(target, dtype=int), np.clip(np.asarray(probability, dtype=float), 1e-8, 1 - 1e-8)
    bins = np.minimum((p * 10).astype(int), 9)
    rows = []
    for bucket in range(10):
        mask = bins == bucket
        if mask.any():
            rows.append({"bin": bucket, "rows": int(mask.sum()), "mean_prediction": float(p[mask].mean()), "event_rate": float(y[mask].mean()), "absolute_gap": float(abs(p[mask].mean() - y[mask].mean()))})
    calibration = pd.DataFrame(rows)
    ece = float((calibration["rows"] / len(y) * calibration["absolute_gap"]).sum()) if len(calibration) else float("nan")
    return {
        "rows": int(len(y)), "prevalence": float(y.mean()), "auc": float(roc_auc_score(y, p)) if y.min() != y.max() else float("nan"),
        "ap": float(average_precision_score(y, p)) if y.sum() else float("nan"), "brier": float(np.mean((p - y) ** 2)), "ece10": ece,
    }, calibration


def _stable_top(frame: pd.DataFrame, score: str, fraction: float) -> pd.DataFrame:
    count = max(1, int(math.ceil(len(frame) * fraction)))
    return frame.sort_values([score, "candidate_id"], ascending=[False, True], kind="stable").head(count)


def _economic_tail(frame: pd.DataFrame, score: str, *, architecture: str, fold_id: str) -> list[dict[str, Any]]:
    result = []
    for fraction in (0.01, 0.05, 0.10):
        selected = _stable_top(frame, score, fraction)
        net = selected["execution_net_ev_12h"].to_numpy(float)
        gross = selected["execution_gross_ev_12h"].to_numpy(float)
        cost = selected["execution_cost_return"].to_numpy(float)
        if not np.allclose(gross - cost, net, atol=1e-7, rtol=0.0):
            raise ValueError("economic tail breaks exact gross-cost=net")
        result.append({"fold_id": fold_id, "architecture": architecture, "score": score, "fraction": fraction, "population_rows": len(frame), "selected_rows": len(selected), "net_ev_bps": float(net.mean() * 1e4), "gross_ev_bps": float(gross.mean() * 1e4), "cost_bps": float(cost.mean() * 1e4), "positive_net_rate": float((net > 0).mean()), "clean_rate": float(selected[TARGETS['clean_opportunity']].mean()), "adverse_rate": float(selected[TARGETS['adverse_competing_risk']].mean()), "cvar5_bps": float(np.sort(net)[:max(1, int(math.ceil(.05 * len(net))))].mean() * 1e4)})
    return result


def fit_oof_probability_heads(frame: pd.DataFrame, *, first_evaluation: str = "2023-10-01", last_evaluation: str = "2024-10-01", frequency: str = "QS", minimum_train_months: int = 2, seed: int = 20260730) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit side-local clean/adverse heads on strict chronological folds."""

    pools = architecture_feature_sets(frame.columns)
    folds = chronological_folds(frame, first_evaluation=first_evaluation, last_evaluation=last_evaluation, frequency=frequency, minimum_train_months=minimum_train_months)
    if not folds:
        raise ValueError("no eligible strict chronological probability-head folds")
    prediction_parts: list[pd.DataFrame] = []
    selections: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    for fold_number, (fold_id, train, evaluation, start) in enumerate(folds):
        out = frame.iloc[evaluation][list(IDENTITY)].copy().reset_index(drop=True)
        out["probability_fold_id"] = fold_id
        out["probability_evaluation_start_utc"] = start
        for side_number, side in enumerate(("long", "short")):
            side_train = train[frame.iloc[train]["side_name"].eq(side).to_numpy()]
            side_eval = evaluation[frame.iloc[evaluation]["side_name"].eq(side).to_numpy()]
            if len(side_train) < 500 or len(side_eval) < 20:
                raise ValueError(f"{fold_id}/{side} lacks strict probability-head support")
            for architecture, candidates in pools.items():
                for head_number, head in enumerate(HEADS):
                    target_name = TARGETS[head]
                    target = frame.iloc[side_train][target_name].to_numpy(int)
                    if np.unique(target).size != 2:
                        raise ValueError(f"{fold_id}/{side}/{head} has one-class train support")
                    selected = _select_train_features(frame.iloc[side_train].loc[:, candidates], target)
                    model = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(), LogisticRegression(C=0.25, max_iter=300, solver="lbfgs", random_state=seed + fold_number * 1000 + side_number * 100 + head_number))
                    model.fit(frame.iloc[side_train].loc[:, selected], target)
                    values = model.predict_proba(frame.iloc[side_eval].loc[:, selected])[:, 1]
                    output_positions = np.flatnonzero(out["side_name"].eq(side).to_numpy())
                    out.loc[output_positions, f"{head}_p__{architecture}"] = values
                    selections.append({"fold_id": fold_id, "side_name": side, "architecture": architecture, "head": head, "candidate_features": json.dumps(list(candidates)), "selected_features": json.dumps(selected), "train_rows": len(side_train), "train_label_available_max": pd.to_datetime(frame.iloc[side_train][LABEL_AVAILABLE], utc=True).max(), "evaluation_start_utc": start})
        probability_columns = [name for name in out if name.endswith(tuple(ARCHITECTURES)) and "_p__" in name]
        if not np.isfinite(out[probability_columns].to_numpy(float)).all():
            raise AssertionError("strict OOF probability sidecar has missing values")
        prediction_parts.append(out)
        audits.append({"fold_id": fold_id, "evaluation_start_utc": start, "train_rows": len(train), "evaluation_rows": len(evaluation), "train_label_available_max": pd.to_datetime(frame.iloc[train][LABEL_AVAILABLE], utc=True).max(), "strict_label_availability": True})
    predictions = pd.concat(prediction_parts, ignore_index=True)
    diagnostics = predictions.merge(frame[[*IDENTITY, *TARGETS.values(), "execution_gross_ev_12h", "execution_cost_return", "execution_net_ev_12h", SCORE]], on=list(IDENTITY), how="left", validate="one_to_one")
    metrics: list[dict[str, Any]] = []
    calibration: list[pd.DataFrame] = []
    economics: list[dict[str, Any]] = []
    for fold_id, local in diagnostics.groupby("probability_fold_id", observed=True, sort=True):
        for architecture in ARCHITECTURES:
            clean = f"clean_opportunity_p__{architecture}"; adverse = f"adverse_competing_risk_p__{architecture}"
            local = local.copy(); local["joint_clean_net_of_adverse"] = local[clean] * (1.0 - local[adverse]); local["negative_adverse_probability"] = -local[adverse]
            for head, column in (("clean_opportunity", clean), ("adverse_competing_risk", adverse)):
                for scope, scoped in (("global", local), *[(side, local.loc[local.side_name.eq(side)]) for side in ("long", "short")]):
                    values, bins = _probability_metrics(scoped[TARGETS[head]].to_numpy(int), scoped[column].to_numpy(float))
                    metrics.append({"fold_id": fold_id, "architecture": architecture, "head": head, "scope": scope, **values})
                    if scope == "global":
                        bins.insert(0, "head", head); bins.insert(0, "architecture", architecture); bins.insert(0, "fold_id", fold_id); calibration.append(bins)
            for score in (clean, "negative_adverse_probability", "joint_clean_net_of_adverse"):
                economics.extend(_economic_tail(local, score, architecture=architecture, fold_id=fold_id))
    return predictions, pd.DataFrame(selections), pd.DataFrame(audits), pd.concat([pd.DataFrame(metrics), pd.DataFrame(economics), pd.concat(calibration, ignore_index=True)], keys=["metrics", "economics", "calibration"], names=["kind", "row"]).reset_index(level="kind").reset_index(drop=True)


def load_historical_panel(state_path: Path, scores_path: Path, labels_2023: Path, labels_2024: Path) -> pd.DataFrame:
    state = _canonical(pd.read_parquet(state_path), source="causal regime/transition OOF")
    scores = _canonical(pd.read_parquet(scores_path), source="reconstructed residual OOF")
    if "residual_is_oof" not in scores or not scores["residual_is_oof"].astype(bool).all():
        raise ValueError("reconstructed score anchor is not fully OOF")
    labels = pd.concat([_canonical(pd.read_parquet(path), source=f"exact labels {path.name}") for path in (labels_2023, labels_2024)], ignore_index=True)
    required = {*TARGETS.values(), "__label_available_at__", "execution_gross_ev_12h", "execution_cost_return", "execution_net_ev_12h"}
    if missing := sorted(required.difference(labels.columns)):
        raise ValueError(f"exact historical labels lack required clean/adverse/economic fields: {missing}")
    labels = labels.loc[:, [*IDENTITY, *required]].copy(); labels[LABEL_AVAILABLE] = pd.to_datetime(labels[LABEL_AVAILABLE], utc=True, errors="raise")
    result = state.merge(scores[[*IDENTITY, SCORE]], on=list(IDENTITY), how="inner", validate="one_to_one").merge(labels, on=list(IDENTITY), how="inner", validate="one_to_one")
    if result.empty:
        raise ValueError("exact labels, causal state OOF, and residual OOF anchor have no common identities")
    for prefix, available in (("regime_", "regime_available_utc"), ("transition_", "transition_available_utc")):
        if available not in result:
            raise ValueError(f"causal state OOF lacks {available}")
        if not pd.to_datetime(result[available], utc=True, errors="raise").le(result["__ts__"]).all():
            raise ValueError(f"{prefix} state is not available by candidate timestamp")
    if not np.allclose(result.execution_gross_ev_12h.to_numpy(float) - result.execution_cost_return.to_numpy(float), result.execution_net_ev_12h.to_numpy(float), atol=1e-7, rtol=0.0):
        raise ValueError("historical exact economics violates gross-cost=net")
    return add_causal_recent_ev_mapping(result)


def materialize(*, state_path: Path = DEFAULT_STATE, scores_path: Path = DEFAULT_SCORES, labels_2023: Path = DEFAULT_2023_LABELS, labels_2024: Path = DEFAULT_2024_LABELS, output_dir: Path = DEFAULT_OUTPUT, **fit_kwargs: Any) -> dict[str, Any]:
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"immutable output already exists: {output_dir}")
    frame = load_historical_panel(Path(state_path), Path(scores_path), Path(labels_2023), Path(labels_2024))
    predictions, selections, audit, packed = fit_oof_probability_heads(frame, **fit_kwargs)
    metrics = packed.loc[packed["kind"].eq("metrics")].drop(columns="kind"); economics = packed.loc[packed["kind"].eq("economics")].drop(columns="kind"); calibration = packed.loc[packed["kind"].eq("calibration")].drop(columns="kind")
    temporary = Path(tempfile.mkdtemp(dir=output_dir.parent, prefix=f".{output_dir.name}."))
    try:
        outputs = {"clean_competing_probability_oof_sidecar.parquet": predictions, "feature_provenance.parquet": selections, "fold_audit.parquet": audit, "probability_metrics.parquet": metrics, "calibration.parquet": calibration, "economic_tail_diagnostics.parquet": economics}
        hashes = {}
        for name, data in outputs.items():
            path = temporary / name; data.to_parquet(path, index=False, compression="zstd"); hashes[name] = sha256(path)
        report = {"schema": SCHEMA, "status": "COMPLETED_RESEARCH_ONLY_STRICT_CHRONOLOGICAL_OOF", "promotion_eligible": False, "research_only_reason": "historical exact labels are frozen backcast diagnostic labels; OOF head predictions do not change that evidence scope", "architectures": list(ARCHITECTURES), "heads": {head: {"target": TARGETS[head], "semantics": "separate binary probability; not a simplex and never cross-relabeled"} for head in HEADS}, "feature_contract": {"shared_anchor": [SCORE, MAPPED_SCORE, MAPPING_AVAILABLE], "regime": "regime_state_p__* plus entropy/margin/uncertainty/OOD", "transition": "transition_state_p__* plus active probability, entropy/margin/uncertainty/OOD", "action_exclusion": "timing, MAE, target-price and wait fields rejected", "selection": "side/head/fold-specific deterministic train-only Spearman screen"}, "oof_contract": {"folds": "expanding chronological quarterly", "label_availability": "__label_available_at__ < evaluation_start_utc", "state_availability": "regime_available_utc and transition_available_utc <= candidate __ts__", "mapping": "daily side-local isotonic recent-EV map from labels resolved before UTC-day snapshot only"}, "sources": {str(path): sha256(Path(path)) for path in (state_path, scores_path, labels_2023, labels_2024)}, "rows": {"joined_exact_rows": len(frame), "oof_sidecar_rows": len(predictions)}, "outputs_sha256": hashes}
        manifest = temporary / "manifest.json"; manifest.write_text(json.dumps(_safe(report), indent=2, sort_keys=True) + "\n", encoding="utf-8"); (temporary / "manifest.sha256").write_text(f"{sha256(manifest)}  manifest.json\n", encoding="utf-8"); os.replace(temporary, output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True); raise
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE); parser.add_argument("--scores", type=Path, default=DEFAULT_SCORES); parser.add_argument("--labels-2023", type=Path, default=DEFAULT_2023_LABELS); parser.add_argument("--labels-2024", type=Path, default=DEFAULT_2024_LABELS); parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--first-evaluation", default="2023-10-01"); parser.add_argument("--last-evaluation", default="2024-10-01"); parser.add_argument("--frequency", default="QS"); parser.add_argument("--minimum-train-months", type=int, default=2)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    report = materialize(state_path=args.state, scores_path=args.scores, labels_2023=args.labels_2023, labels_2024=args.labels_2024, output_dir=args.output_dir, first_evaluation=args.first_evaluation, last_evaluation=args.last_evaluation, frequency=args.frequency, minimum_train_months=args.minimum_train_months)
    print(json.dumps(_safe(report), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
