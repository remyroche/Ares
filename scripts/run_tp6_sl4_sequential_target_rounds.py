#!/usr/bin/env python3
"""Sequential TP6/SL4 H12 target/weight runner for rounds 2--4.

This runner is intentionally *not* a factorial search.  One invocation fits
one named candidate on a supplied, frozen per-side feature contract:

  round 2: B5 terminal-margin, B6 peak-MFE, B7 adverse-MAE
  round 3: B8 cost-aware net, B9 gross/net headroom
  round 4: B10 convex combinations of prior candidates, with one BW policy

``--candidate B10`` requires explicit component names and weights.  A caller
must select those from an earlier development result before opening a later
evaluation window.  Future-path sidecar fields are accepted only to construct
resolved training labels and are rejected from model features.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from extreme_price_movements.tp6_sl4_target_weights import (  # noqa: E402
    TP6SL4Columns, TargetParameters, WeightParameters, assert_simplex,
    build_target, build_weight, target_manifest,
)


ROUND_CANDIDATES = {
    2: {"B5", "B6", "B7"},
    3: {"B8", "B9"},
    4: {"B10"},
}
WEIGHTS = {"BW0", "BW1", "BW2", "BW3", "BW4", "BW5", "BW6", "BW8"}
TOP_FRACTIONS = (.01, .05, .10, .20)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _coerce_columns(value: dict[str, Any] | None) -> TP6SL4Columns:
    if value is None:
        return TP6SL4Columns()
    valid = set(TP6SL4Columns.__dataclass_fields__)
    extra = sorted(set(value) - valid)
    if extra:
        raise ValueError(f"unknown sidecar-column keys: {extra}")
    return TP6SL4Columns(**value)


def _coerce_dataclass(value: dict[str, Any] | None, kind: type[TargetParameters] | type[WeightParameters]):
    if value is None:
        return kind()
    valid = set(kind.__dataclass_fields__)
    extra = sorted(set(value) - valid)
    if extra:
        raise ValueError(f"unknown {kind.__name__} keys: {extra}")
    return kind(**value)


def _feature_contract(payload: Any) -> dict[str, list[str]]:
    """Accept ``{long:[...],short:[...]}`` or a shared list."""
    if isinstance(payload, list):
        result = {"long": list(payload), "short": list(payload)}
    elif isinstance(payload, dict) and {"long", "short"}.issubset(payload):
        result = {"long": list(payload["long"]), "short": list(payload["short"])}
    else:
        raise ValueError("features JSON must be a list or contain long and short lists")
    for side, columns in result.items():
        if not columns or len(columns) != len(set(columns)):
            raise ValueError(f"{side} feature contract must be non-empty and unique")
    return result


def _frozen_base_contract(root: Path) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for side in ("long", "short"):
        payload = _load_json(root / side / "target_family_manifest.json")
        fields = payload.get("feature_contract", {}).get(f"T2_soft_barrier|tp3_sl2|{side}", [])
        if not isinstance(fields, list) or len(fields) < 30:
            raise ValueError(f"missing frozen {side} base feature contract")
        result[side] = fields
    return result


def _read_joined(panel: Path, sidecar: Path, base_columns: list[str], label_columns: list[str]) -> pd.DataFrame:
    """Join causal base fields to resolved TP6/SL4 labels by candidate ID."""
    base_parts = sorted((panel / "parts").glob("*.parquet"))
    label_parts = sorted((sidecar / "parts").glob("*.parquet"))
    if not base_parts or not label_parts:
        raise FileNotFoundError("panel or TP6/SL4 sidecar parts are missing")
    base = pd.concat([pd.read_parquet(part, columns=base_columns) for part in base_parts], ignore_index=True)
    labels = pd.concat([pd.read_parquet(part, columns=label_columns) for part in label_parts], ignore_index=True)
    if base.candidate_id.duplicated().any() or labels.candidate_id.duplicated().any():
        raise ValueError("candidate IDs must be unique before TP6/SL4 join")
    return base.merge(labels, on="candidate_id", how="inner", validate="one_to_one")


def _matrix(frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    values = frame.loc[:, features].replace([np.inf, -np.inf], np.nan)
    # The runner does not silently admit a poorly-covered feature contract.
    coverage = 1.0 - values.isna().mean()
    if (coverage < .90).any():
        failed = coverage[coverage < .90].to_dict()
        raise ValueError(f"feature coverage below 90%: {failed}")
    return values.fillna(0.).to_numpy(np.float32)


def _model() -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        objective="huber", alpha=.90, n_estimators=50, learning_rate=.05,
        num_leaves=24, min_child_samples=400, colsample_bytree=.8,
        subsample=.8, reg_lambda=10., random_state=20260808, n_jobs=1,
        verbosity=-1,
    )


def _probability_models(train: pd.DataFrame, evaluation: pd.DataFrame, features: list[str], target: np.ndarray, weight: np.ndarray) -> np.ndarray:
    x_train, x_evaluation = _matrix(train, features), _matrix(evaluation, features)
    predictions = np.column_stack([
        np.maximum(_model().fit(x_train, target[:, klass], sample_weight=weight).predict(x_evaluation), 0.)
        for klass in range(3)
    ])
    predictions /= np.maximum(predictions.sum(axis=1, keepdims=True), 1e-12)
    assert_simplex(predictions)
    return predictions


def _score(probability: np.ndarray, target: np.ndarray, train_net_bps: np.ndarray, weight: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Causal in-fold target-conditional net values, then expected net score."""
    means = (target * train_net_bps[:, None] * weight[:, None]).sum(axis=0) / np.maximum((target * weight[:, None]).sum(axis=0), 1e-12)
    return probability @ means, means


def _metrics(frame: pd.DataFrame, score: np.ndarray) -> list[dict[str, Any]]:
    ranked = frame.assign(score_bps=score).sort_values(["score_bps", "candidate_id"], ascending=[False, True], kind="mergesort")
    rows: list[dict[str, Any]] = []
    for fraction in TOP_FRACTIONS:
        selected = ranked.head(int(np.ceil(len(ranked) * fraction)))
        for side, subset in (("all", selected), ("long", selected[selected.side_name.eq("long")]), ("short", selected[selected.side_name.eq("short")])):
            rows.append({"top_fraction": fraction, "attribution_side": side, "n": int(len(subset)),
                         "gross_bps": float(subset.gross_bps.mean()), "net_bps": float(subset.net_bps.mean())})
    return rows


def _candidate_target(frame: pd.DataFrame, candidate: str, *, b10_components: list[str], b10_weights: np.ndarray, columns: TP6SL4Columns, parameters: TargetParameters) -> np.ndarray:
    if candidate != "B10":
        return build_target(frame, candidate, columns=columns, parameters=parameters)
    if len(b10_components) < 2 or len(b10_components) != len(b10_weights):
        raise ValueError("B10 requires at least two components and one weight per component")
    allowed = {"B5", "B6", "B7", "B8", "B9"}
    if not set(b10_components).issubset(allowed) or len(set(b10_components)) != len(b10_components):
        raise ValueError("B10 components must be distinct members of B5,B6,B7,B8,B9")
    if not np.isfinite(b10_weights).all() or (b10_weights < 0.).any() or b10_weights.sum() <= 0.:
        raise ValueError("B10 weights must be finite, non-negative, and sum positive")
    weights = b10_weights / b10_weights.sum()
    target = sum(weight * build_target(frame, name, columns=columns, parameters=parameters) for name, weight in zip(b10_components, weights, strict=True))
    assert_simplex(target)
    return target


def _target_columns(name: str, columns: TP6SL4Columns, b10_components: list[str]) -> set[str]:
    if name == "B10":
        return set().union(*(_target_columns(component, columns, []) for component in b10_components))
    required = {"B5": {columns.event, columns.terminal_atr}, "B6": {columns.event, columns.mfe_atr},
                "B7": {columns.event, columns.mae_atr}, "B8": {columns.event, columns.gross_bps},
                "B9": {columns.event, columns.net_bps, columns.gross_bps}}
    return required[name]


def _weight_columns(name: str, columns: TP6SL4Columns) -> set[str]:
    required = {"BW0": set(), "BW1": {columns.event}, "BW2": {columns.event, columns.exit_minute},
                "BW3": {columns.event, columns.contract_consensus}, "BW4": {columns.event, columns.gross_bps},
                "BW5": {columns.event}, "BW6": {columns.event}, "BW8": {columns.event}}
    return required[name]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--panel", type=Path, required=True, help="Causal original full-universe panel with parts/")
    p.add_argument("--sidecar", type=Path, required=True, help="Exact TP6/SL4 resolved-label sidecar with parts/")
    p.add_argument("--consensus-sidecar", type=Path, help="Exact nearby-geometry consensus label sidecar; required for BW3")
    p.add_argument("--features-json", type=Path, help="Frozen side-local feature contract: list or {long,short}")
    p.add_argument("--base-contract", type=Path, help="Base HPO artifact used to load frozen side-local 36-feature lists")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--round", type=int, choices=(2, 3, 4), required=True)
    p.add_argument("--candidate", required=True, help="B5,B6,B7 (R2); B8,B9 (R3); B10 (R4)")
    p.add_argument("--weight", choices=sorted(WEIGHTS), default="BW0")
    p.add_argument("--train-start", required=True)
    p.add_argument("--eval-start", required=True)
    p.add_argument("--eval-end", required=True)
    p.add_argument("--columns-json", type=Path, help="Optional JSON overrides for TP6SL4Columns")
    p.add_argument("--target-parameters-json", type=Path, help="Optional TargetParameters JSON")
    p.add_argument("--target-parameters", help="Inline TargetParameters JSON; mutually exclusive with --target-parameters-json")
    p.add_argument("--weight-parameters-json", type=Path, help="Optional WeightParameters JSON")
    p.add_argument("--weight-parameters", help="Inline WeightParameters JSON; mutually exclusive with --weight-parameters-json")
    p.add_argument("--b10-components", default="", help="Comma-separated B5,B6,B7,B8,B9 for B10 only")
    p.add_argument("--b10-weights", default="", help="Comma-separated non-negative B10 weights")
    a = p.parse_args()
    candidate = str(a.candidate).upper()
    if candidate not in ROUND_CANDIDATES[a.round]:
        raise ValueError(f"{candidate} is not an allowed round-{a.round} candidate: {sorted(ROUND_CANDIDATES[a.round])}")
    if a.out.exists():
        raise FileExistsError(a.out)
    columns = _coerce_columns(_load_json(a.columns_json) if a.columns_json else None)
    if not a.columns_json:
        columns = replace(columns, terminal_atr="t4_tp6_sl4_terminal_pnl_atr")
    if a.target_parameters_json and a.target_parameters:
        raise ValueError("use only one target-parameter source")
    target_parameters = _coerce_dataclass(_load_json(a.target_parameters_json) if a.target_parameters_json else (json.loads(a.target_parameters) if a.target_parameters else None), TargetParameters)
    if a.weight_parameters_json and a.weight_parameters:
        raise ValueError("use only one weight-parameter source")
    weight_parameters = _coerce_dataclass(_load_json(a.weight_parameters_json) if a.weight_parameters_json else (json.loads(a.weight_parameters) if a.weight_parameters else None), WeightParameters)
    if bool(a.features_json) == bool(a.base_contract):
        raise ValueError("provide exactly one of --features-json or --base-contract")
    features = _feature_contract(_load_json(a.features_json)) if a.features_json else _frozen_base_contract(a.base_contract)
    b10_components = [item.strip().upper() for item in a.b10_components.split(",") if item.strip()]
    b10_weights = np.asarray([float(item) for item in a.b10_weights.split(",") if item.strip()], dtype=float)
    if candidate != "B10" and (b10_components or len(b10_weights)):
        raise ValueError("B10 components/weights are forbidden unless --candidate B10")
    start, evaluation_start, evaluation_end = (pd.Timestamp(value, tz="UTC") for value in (a.train_start, a.eval_start, a.eval_end))
    if not start < evaluation_start < evaluation_end:
        raise ValueError("require train-start < eval-start < eval-end")
    sidecar_fields = list(asdict(columns).values())
    label_fields = _target_columns(candidate, columns, b10_components) | _weight_columns(a.weight, columns) | {columns.gross_bps, columns.net_bps}
    leaked = sorted((set(features["long"]) | set(features["short"])) & set(sidecar_fields))
    if leaked:
        raise ValueError(f"resolved TP6/SL4 label fields are forbidden model features: {leaked}")
    path_fields = {columns.mfe_atr, columns.mae_atr}
    if a.weight == "BW3" and not a.consensus_sidecar:
        raise ValueError("BW3 requires --consensus-sidecar with exact 3×3 nearby-geometry labels")
    base_fields = list(dict.fromkeys(["candidate_id", "__ts__", "side_name", *path_fields, *features["long"], *features["short"]]))
    derived_training_fields = {columns.contract_consensus} if a.weight == "BW3" else set()
    resolved_fields = list(dict.fromkeys(["candidate_id", "__label_available_at__", *(set(label_fields) - path_fields - derived_training_fields)]))
    raw = _read_joined(a.panel, a.sidecar, base_fields, resolved_fields)
    if a.weight == "BW3":
        consensus_parts = sorted((a.consensus_sidecar / "parts").glob("*.parquet"))
        if not consensus_parts:
            raise FileNotFoundError("consensus sidecar has no parts")
        consensus = pd.concat([pd.read_parquet(part, columns=["candidate_id", "tp6_sl4_contract_consensus", "__label_available_at__"]) for part in consensus_parts], ignore_index=True)
        if consensus.candidate_id.duplicated().any():
            raise ValueError("consensus sidecar candidate IDs must be unique")
        raw = raw.merge(consensus, on="candidate_id", how="inner", validate="one_to_one", suffixes=("", "_consensus"))
        if not raw["__label_available_at__"].eq(raw["__label_available_at___consensus"]).all():
            raise ValueError("central and consensus label availability must match")
        raw = raw.drop(columns=["__label_available_at___consensus"])
    raw["__ts__"] = pd.to_datetime(raw["__ts__"], utc=True)
    raw["__label_available_at__"] = pd.to_datetime(raw["__label_available_at__"], utc=True)
    # Keep original sidecar names for target builders, and add neutral output
    # aliases rather than mutating the label schema.
    raw["gross_bps"] = raw[columns.gross_bps]
    raw["net_bps"] = raw[columns.net_bps]
    # The panel contract must resolve labels after their candidate timestamp;
    # we require it explicitly rather than trusting a caller's window string.
    if not raw["__label_available_at__"].gt(raw["__ts__"]).all():
        raise ValueError("TP6/SL4 labels must become available strictly after candidate time")
    train = raw[raw["__ts__"].ge(start) & raw["__ts__"].lt(evaluation_start) & raw["__label_available_at__"].lt(evaluation_start)].copy()
    evaluation = raw[raw["__ts__"].ge(evaluation_start) & raw["__ts__"].lt(evaluation_end)].copy()
    if train.empty or evaluation.empty:
        raise ValueError("empty train/evaluation split")
    outputs: list[pd.DataFrame] = []
    summary: dict[str, Any] = {"sides": {}}
    for side in ("long", "short"):
        tr, ev = train[train.side_name.eq(side)].copy(), evaluation[evaluation.side_name.eq(side)].copy()
        if min(len(tr), len(ev)) < 1000:
            raise ValueError(f"insufficient {side} rows")
        target = _candidate_target(tr, candidate, b10_components=b10_components, b10_weights=b10_weights, columns=columns, parameters=target_parameters)
        weight = build_weight(tr, a.weight, columns=columns, target=target, target_parameters=target_parameters, parameters=weight_parameters)
        probability = _probability_models(tr, ev, features[side], target, weight)
        score, means = _score(probability, target, tr.net_bps.to_numpy(float), weight)
        outputs.append(ev[["candidate_id", "__ts__", "side_name", "gross_bps", "net_bps"]].assign(score_bps=score, p_upper=probability[:, 0], p_lower=probability[:, 1], p_timeout=probability[:, 2]))
        summary["sides"][side] = {"train_rows": len(tr), "evaluation_rows": len(ev), "feature_count": len(features[side]), "conditional_net_means_bps": means.tolist(), "weight_min": float(weight.min()), "weight_max": float(weight.max())}
    output = pd.concat(outputs, ignore_index=True)
    metrics = _metrics(output, output.score_bps.to_numpy(float))
    summary.update({"global_metrics": metrics, "score_net_spearman": float(spearmanr(output.score_bps, output.net_bps).statistic)})
    a.out.mkdir(parents=True)
    output.to_parquet(a.out / "predictions.parquet", index=False)
    pd.DataFrame(metrics).to_parquet(a.out / "metrics.parquet", index=False)
    target_contract = dict(target_manifest("B5" if candidate == "B10" else candidate, a.weight, columns=columns, target_parameters=target_parameters, weight_parameters=weight_parameters))
    if candidate == "B10":
        target_contract["target"] = "B10"
        target_contract["b10"] = {"components": b10_components, "normalised_weights": (b10_weights / b10_weights.sum()).tolist()}
    manifest = {"schema": "tp6_sl4_sequential_target_rounds_v1", "round": a.round, "candidate": candidate,
                "b10_components": b10_components, "b10_weights": b10_weights.tolist(),
                "weight": a.weight, "target_contract": target_contract,
                "windows": {"train_start": str(start), "eval_start": str(evaluation_start), "eval_end": str(evaluation_end), "purge": "train labels must resolve strictly before eval_start"},
                "feature_contract": features, "inference_guard": "all sidecar outcome/path columns are rejected from features", "summary": summary}
    (a.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"candidate": candidate, "weight": a.weight, "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
