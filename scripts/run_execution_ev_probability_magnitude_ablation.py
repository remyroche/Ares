#!/usr/bin/env python3
"""Exact-policy OOF probability × conditional win/loss magnitude EV heads."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.execution_ev_model_ablation import (  # noqa: E402
    ExecutionEVModelAblationConfig,
    _materialize_feature_matrix,
    apply_execution_ev_causal_recent_ev_correction,
    chronological_purged_splits,
    validate_execution_ev_model_ablation_contract,
)
from scripts.run_execution_ev_model_ablation import _load_provenance  # noqa: E402

IDENTITY = ("__ts__", "__symbol__", "side_name", "candidate_id")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _classifier(*, seed: int, iterations: int, threads: int) -> CatBoostClassifier:
    return CatBoostClassifier(
        iterations=int(iterations),
        depth=6,
        learning_rate=0.04,
        loss_function="Logloss",
        eval_metric="Logloss",
        l2_leaf_reg=6.0,
        random_seed=int(seed),
        thread_count=int(threads),
        verbose=False,
        allow_writing_files=False,
    )


def _regressor(*, seed: int, iterations: int, threads: int) -> CatBoostRegressor:
    return CatBoostRegressor(
        iterations=int(iterations),
        depth=6,
        learning_rate=0.04,
        loss_function="RMSE",
        l2_leaf_reg=8.0,
        random_seed=int(seed),
        thread_count=int(threads),
        verbose=False,
        allow_writing_files=False,
    )


def _fit_regression(
    x: pd.DataFrame,
    target: np.ndarray,
    train: np.ndarray,
    valid: np.ndarray,
    *,
    seed: int,
    iterations: int,
    threads: int,
    positive_mask: np.ndarray | None = None,
    scale: float = 0.01,
) -> np.ndarray:
    fit = train if positive_mask is None else train[positive_mask[train]]
    if len(fit) < 100:
        raise ValueError("conditional magnitude fit has insufficient support")
    values = np.maximum(target[fit], 0.0)
    cap = float(np.quantile(values, 0.995))
    transformed = np.log1p(np.minimum(values, cap) / float(scale))
    model = _regressor(seed=seed, iterations=iterations, threads=threads)
    model.fit(x.iloc[fit], transformed)
    prediction = np.maximum(
        np.expm1(model.predict(x.iloc[valid])) * float(scale),
        0.0,
    )
    return prediction


def _fit_outer_side(
    x: pd.DataFrame,
    net_ev: np.ndarray,
    train: np.ndarray,
    valid: np.ndarray,
    *,
    seed: int,
    iterations: int,
    threads: int,
    temperature: float,
) -> dict[str, np.ndarray]:
    hard = (net_ev > 0.0).astype(np.int8)
    hard_model = _classifier(seed=seed, iterations=iterations, threads=threads)
    hard_model.fit(x.iloc[train], hard[train])
    p_hard = hard_model.predict_proba(x.iloc[valid])[:, 1]

    magnitude_weight = np.clip(
        np.abs(net_ev[train]) / max(float(np.median(np.abs(net_ev[train]))), 1e-4),
        0.25,
        5.0,
    )
    weighted_model = _classifier(
        seed=seed + 1, iterations=iterations, threads=threads
    )
    weighted_model.fit(x.iloc[train], hard[train], sample_weight=magnitude_weight)
    p_weighted = weighted_model.predict_proba(x.iloc[valid])[:, 1]

    soft_target = 1.0 / (
        1.0 + np.exp(-np.clip(net_ev / float(temperature), -30.0, 30.0))
    )
    soft_model = _regressor(seed=seed + 2, iterations=iterations, threads=threads)
    soft_model.fit(x.iloc[train], soft_target[train])
    p_soft = np.clip(soft_model.predict(x.iloc[valid]), 0.0, 1.0)

    barrier_probabilities: dict[str, np.ndarray] = {}
    barrier_specs = (
        ("win_ge_005", net_ev >= 0.005),
        ("win_ge_010", net_ev >= 0.010),
        ("win_ge_020", net_ev >= 0.020),
        ("loss_le_005", net_ev <= -0.005),
        ("loss_le_010", net_ev <= -0.010),
        ("loss_le_020", net_ev <= -0.020),
        ("loss_le_040", net_ev <= -0.040),
    )
    for offset, (name, label) in enumerate(barrier_specs, start=10):
        model = _classifier(
            seed=seed + offset,
            iterations=iterations,
            threads=threads,
        )
        model.fit(x.iloc[train], label[train].astype(np.int8))
        barrier_probabilities[name] = model.predict_proba(x.iloc[valid])[:, 1]

    win = np.maximum(net_ev, 0.0)
    loss = np.maximum(-net_ev, 0.0)
    win_cond = _fit_regression(
        x,
        win,
        train,
        valid,
        seed=seed + 3,
        iterations=iterations,
        threads=threads,
        positive_mask=hard.astype(bool),
    )
    loss_cond = _fit_regression(
        x,
        loss,
        train,
        valid,
        seed=seed + 4,
        iterations=iterations,
        threads=threads,
        positive_mask=~hard.astype(bool),
    )
    win_unconditional = _fit_regression(
        x,
        win,
        train,
        valid,
        seed=seed + 5,
        iterations=iterations,
        threads=threads,
    )
    loss_unconditional = _fit_regression(
        x,
        loss,
        train,
        valid,
        seed=seed + 6,
        iterations=iterations,
        threads=threads,
    )
    positive_tail = (
        0.005 * barrier_probabilities["win_ge_005"]
        + 0.005 * barrier_probabilities["win_ge_010"]
        + 0.010 * barrier_probabilities["win_ge_020"]
    )
    negative_tail = (
        0.005 * barrier_probabilities["loss_le_005"]
        + 0.005 * barrier_probabilities["loss_le_010"]
        + 0.010 * barrier_probabilities["loss_le_020"]
        + 0.020 * barrier_probabilities["loss_le_040"]
    )
    return {
        "hard_probability": p_hard,
        "weighted_probability": p_weighted,
        "soft_probability": p_soft,
        "conditional_win_magnitude": win_cond,
        "conditional_loss_magnitude": loss_cond,
        "hard_decomposed_ev": p_hard * win_cond - (1.0 - p_hard) * loss_cond,
        "weighted_decomposed_ev": (
            p_weighted * win_cond - (1.0 - p_weighted) * loss_cond
        ),
        "soft_decomposed_ev": p_soft * win_cond - (1.0 - p_soft) * loss_cond,
        "unconditional_contribution_ev": win_unconditional - loss_unconditional,
        "unconditional_win_contribution": win_unconditional,
        "unconditional_loss_contribution": loss_unconditional,
        **{f"probability_{name}": value for name, value in barrier_probabilities.items()},
        "barrier_tail_ev": positive_tail - negative_tail,
        "strong_win_minus_severe_loss": (
            barrier_probabilities["win_ge_010"]
            + barrier_probabilities["win_ge_020"]
            - barrier_probabilities["loss_le_020"]
            - 2.0 * barrier_probabilities["loss_le_040"]
        ),
        "strong_win_risk_gate": (
            barrier_probabilities["win_ge_010"]
            * (1.0 - barrier_probabilities["loss_le_020"])
            * (1.0 - barrier_probabilities["loss_le_040"])
        ),
    }


def _economic_metrics(
    score: np.ndarray,
    net_ev: np.ndarray,
    gross_ev: np.ndarray,
    mask: np.ndarray,
    *,
    top_fraction: float,
) -> dict[str, Any]:
    positions = np.flatnonzero(mask & np.isfinite(score))
    take = int(np.ceil(float(top_fraction) * len(positions)))
    ranked = positions[np.argsort(-score[positions], kind="stable")[:take]]
    return {
        "rows": int(len(positions)),
        "top_k_rows": int(len(ranked)),
        "top_k_mean_net_ev": float(net_ev[ranked].mean()),
        "top_k_mean_gross_ev": float(gross_ev[ranked].mean()),
        "top_k_positive_rate": float((net_ev[ranked] > 0.0).mean()),
        "top_k_sum_net_ev": float(net_ev[ranked].sum()),
    }


def _probability_metrics(
    probability: np.ndarray,
    net_ev: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    valid = mask & np.isfinite(probability)
    actual = net_ev[valid] > 0.0
    score = probability[valid]
    return {
        "rows": int(valid.sum()),
        "auc": float(roc_auc_score(actual, score)),
        "average_precision": float(average_precision_score(actual, score)),
        "brier": float(brier_score_loss(actual, score)),
    }


def _parse_families(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--additional-input-families",
        type=_parse_families,
        default=(),
    )
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--min-train-rows", type=int, default=10_000)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--threads", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.01)
    parser.add_argument("--top-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=20260725)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    frame = pd.read_parquet(args.input)
    frame["execution_decision_utc"] = pd.to_datetime(
        frame["execution_decision_utc"], utc=True, errors="raise"
    )
    frame["execution_label_end_utc"] = pd.to_datetime(
        frame["execution_label_end_utc"], utc=True, errors="raise"
    )
    provenance, payload = _load_provenance(args.provenance)
    config = ExecutionEVModelAblationConfig(
        n_splits=args.n_splits,
        min_train_rows=args.min_train_rows,
        decision_time_col="execution_decision_utc",
        label_end_time_col="execution_label_end_utc",
        additional_input_families=tuple(args.additional_input_families),
        recent_ev_correction_routes=("catboost_predicted_archetype",),
        top_k_fraction=float(args.top_fraction),
    )
    raw_columns, archetype_levels = validate_execution_ev_model_ablation_contract(
        frame,
        provenance,
        decision_time_col=config.decision_time_col,
        side_col=config.side_col,
        catboost_archetype_col=config.catboost_archetype_col,
        additional_input_families=config.additional_input_families,
    )
    x = _materialize_feature_matrix(
        frame,
        raw_columns,
        catboost_archetype_col=config.catboost_archetype_col,
        archetype_levels=archetype_levels,
    )
    net_ev = pd.to_numeric(frame["execution_net_ev_12h"], errors="raise").to_numpy(
        dtype=np.float64
    )
    gross_ev = pd.to_numeric(
        frame["execution_gross_ev_12h"], errors="raise"
    ).to_numpy(dtype=np.float64)
    side = frame["side_name"].astype(str).str.lower().to_numpy()
    folds = chronological_purged_splits(
        frame,
        n_splits=args.n_splits,
        min_train_size=args.min_train_rows,
        decision_time_col=config.decision_time_col,
        label_end_time_col=config.label_end_time_col,
        horizon_hours=24.0,
        embargo_hours=12.0,
    )
    prediction_names = (
        "hard_probability",
        "weighted_probability",
        "soft_probability",
        "conditional_win_magnitude",
        "conditional_loss_magnitude",
        "hard_decomposed_ev",
        "weighted_decomposed_ev",
        "soft_decomposed_ev",
        "unconditional_contribution_ev",
        "unconditional_win_contribution",
        "unconditional_loss_contribution",
        "probability_win_ge_005",
        "probability_win_ge_010",
        "probability_win_ge_020",
        "probability_loss_le_005",
        "probability_loss_le_010",
        "probability_loss_le_020",
        "probability_loss_le_040",
        "barrier_tail_ev",
        "strong_win_minus_severe_loss",
        "strong_win_risk_gate",
    )
    predictions = {
        name: np.full(len(frame), np.nan, dtype=np.float64)
        for name in prediction_names
    }
    fold_id = np.full(len(frame), np.nan, dtype=np.float64)
    audits: list[dict[str, Any]] = []
    for split in folds:
        for side_name in ("long", "short"):
            train = np.asarray(
                [index for index in split.train_indices if side[index] == side_name],
                dtype=int,
            )
            valid = np.asarray(
                [
                    index
                    for index in split.validation_indices
                    if side[index] == side_name
                ],
                dtype=int,
            )
            result = _fit_outer_side(
                x,
                net_ev,
                train,
                valid,
                seed=args.seed + split.fold * 100 + (0 if side_name == "long" else 50),
                iterations=args.iterations,
                threads=args.threads,
                temperature=args.temperature,
            )
            for name, values in result.items():
                predictions[name][valid] = values
            fold_id[valid] = split.fold
            audits.append(
                {
                    "fold": int(split.fold),
                    "side": side_name,
                    "train_rows": int(len(train)),
                    "validation_rows": int(len(valid)),
                    "train_positive_rate": float((net_ev[train] > 0.0).mean()),
                }
            )
            print(
                f"[probability-magnitude] fold={split.fold} side={side_name} "
                f"train={len(train)} valid={len(valid)}",
                flush=True,
            )
    shared = np.isfinite(fold_id)
    for values in predictions.values():
        shared &= np.isfinite(values)
    if not shared.any():
        raise ValueError("probability-magnitude ablation has no shared OOF rows")

    correction_reports: dict[str, Any] = {}
    for name in (
        "hard_decomposed_ev",
        "weighted_decomposed_ev",
        "soft_decomposed_ev",
        "unconditional_contribution_ev",
        "barrier_tail_ev",
    ):
        corrected, report = apply_execution_ev_causal_recent_ev_correction(
            frame,
            predictions[name],
            net_ev,
            provenance,
            route="catboost_predicted_archetype",
            config=config,
        )
        corrected_name = f"{name}__recent_ev"
        predictions[corrected_name] = corrected
        correction_reports[name] = report

    economics = {
        name: _economic_metrics(
            values,
            net_ev,
            gross_ev,
            shared,
            top_fraction=args.top_fraction,
        )
        for name, values in predictions.items()
        if name
        in {
            "hard_probability",
            "weighted_probability",
            "soft_probability",
            "hard_decomposed_ev",
            "weighted_decomposed_ev",
            "soft_decomposed_ev",
            "unconditional_contribution_ev",
            "hard_decomposed_ev__recent_ev",
            "weighted_decomposed_ev__recent_ev",
            "soft_decomposed_ev__recent_ev",
            "unconditional_contribution_ev__recent_ev",
            "barrier_tail_ev",
            "barrier_tail_ev__recent_ev",
            "strong_win_minus_severe_loss",
            "strong_win_risk_gate",
        }
    }
    probability = {
        name: _probability_metrics(predictions[name], net_ev, shared)
        for name in ("hard_probability", "weighted_probability", "soft_probability")
    }
    barrier_probability = {}
    barrier_actual = {
        "probability_win_ge_005": net_ev >= 0.005,
        "probability_win_ge_010": net_ev >= 0.010,
        "probability_win_ge_020": net_ev >= 0.020,
        "probability_loss_le_005": net_ev <= -0.005,
        "probability_loss_le_010": net_ev <= -0.010,
        "probability_loss_le_020": net_ev <= -0.020,
        "probability_loss_le_040": net_ev <= -0.040,
    }
    for name, actual in barrier_actual.items():
        valid = shared & np.isfinite(predictions[name])
        score = predictions[name][valid]
        label = actual[valid]
        barrier_probability[name] = {
            "rows": int(valid.sum()),
            "prevalence": float(label.mean()),
            "auc": float(roc_auc_score(label, score)),
            "average_precision": float(average_precision_score(label, score)),
            "brier": float(brier_score_loss(label, score)),
        }
    output = frame.loc[:, list(IDENTITY)].copy()
    output["oof_fold"] = fold_id
    for name, values in predictions.items():
        output[name] = values
    output.to_parquet(
        args.output_dir / "oof_predictions.parquet",
        index=False,
        compression="zstd",
    )
    summary = {
        "schema": "execution_ev_probability_magnitude_ablation_v1",
        "status": "strict_side_local_outer_oof_not_promoted",
        "architecture": (
            "P(net>0) x E[net|win] - P(net<=0) x E[-net|loss], plus "
            "soft/magnitude-weighted probability and unconditional contribution arms"
        ),
        "feature_columns": raw_columns,
        "additional_input_families": list(args.additional_input_families),
        "shared_oof_rows": int(shared.sum()),
        "probability_metrics": probability,
        "barrier_probability_metrics": barrier_probability,
        "economic_metrics": economics,
        "fold_audit": audits,
        "recent_ev_correction": correction_reports,
        "sources": {
            "input": {"path": str(args.input), "sha256": _sha256(args.input)},
            "provenance": {
                "path": str(args.provenance),
                "sha256": _sha256(args.provenance),
                "schema": payload.get("schema"),
            },
        },
    }
    _write_json(args.output_dir / "summary.json", summary)
    return summary


def main() -> None:
    summary = run(_parser().parse_args())
    print(json.dumps(_safe(summary["economic_metrics"]), indent=2))


if __name__ == "__main__":
    main()
