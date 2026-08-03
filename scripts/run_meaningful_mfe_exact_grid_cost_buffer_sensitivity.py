#!/usr/bin/env python3
"""Diagnostic exact-grid row-cost-plus-buffer opportunity/capture sensitivity.

This deliberately *does not* tune a new model.  It holds the model geometry
from the authoritative ``meaningful_mfe_exact_grid_reset_20260730_v2`` May
selection fixed and asks whether an executable, row-cost-aware definition of
opportunity transfers better than the ATR barrier.  For every buffer in
``0/25/50/100`` bp it fits, separately for long and short:

* ``opportunity``: the executable 12h MFE can cover that row's execution cost
  plus the buffer; and
* ``capture_given_opportunity``: the frozen exact execution policy actually
  produces net EV at least equal to the buffer, conditional on opportunity.

The deployable ranking is only their probability product.  All results are
diagnostic/non-promotable: this is a bounded label sensitivity, not policy HPO.
Feature screening is deliberately repeated inside each train split and side;
only geometry is frozen from the primary report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_july_exact_preentry_heads import IDENTITY
from scripts.run_historical_to_july_meaningful_mfe_gate_challenger import (
    classification_metrics,
    fit_model,
    predict_model,
    select_features_nested,
    sha256,
)
from scripts.run_meaningful_mfe_exact_grid_reset import (
    MODEL_GRIDS,
    SIDES,
    TRANSFER_SPECS,
    _base_masks,
    july_grouped_day_folds,
    load_panel,
    stable_top,
)


SCHEMA = "meaningful_mfe_exact_grid_cost_buffer_sensitivity_v1"
BUFFERS_BPS = (0, 25, 50, 100)
FROZEN_RESET_REPORT = Path(
    "data_perp/artifacts/meaningful_mfe_exact_grid_reset_20260730_v2/report.json"
)
FROZEN_RESET_RUNNER = Path("scripts/run_meaningful_mfe_exact_grid_reset.py")


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
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def buffer_column(buffer_bps: int, prefix: str) -> str:
    if buffer_bps not in BUFFERS_BPS:
        raise ValueError(f"unsupported buffer {buffer_bps} bps")
    return f"{prefix}_{buffer_bps}bps"


def derive_cost_buffer_targets(panel: pd.DataFrame, buffer_bps: int) -> pd.DataFrame:
    """Create row-cost labels without subtracting execution cost a second time."""

    required = {
        "execution_mfe_return_12h",
        "execution_cost_return",
        "execution_gross_ev_12h",
        "execution_net_ev_12h",
    }
    missing = sorted(required.difference(panel.columns))
    if missing:
        raise ValueError(f"panel misses exact-policy economics: {missing}")
    work = panel.copy()
    mfe = pd.to_numeric(work["execution_mfe_return_12h"], errors="raise").to_numpy(float)
    cost = pd.to_numeric(work["execution_cost_return"], errors="raise").to_numpy(float)
    gross = pd.to_numeric(work["execution_gross_ev_12h"], errors="raise").to_numpy(float)
    net = pd.to_numeric(work["execution_net_ev_12h"], errors="raise").to_numpy(float)
    if not all(np.isfinite(values).all() for values in (mfe, cost, gross, net)):
        raise ValueError("exact-policy MFE/cost/gross/net fields must be finite")
    if not np.allclose(gross - cost, net, atol=1e-7, rtol=0.0):
        raise ValueError("exact gross-cost-net identity failed before buffer labels")
    buffer_return = float(buffer_bps) / 1e4
    # Strictness is intentional.  The zero-buffer label remains the canonical
    # ``MFE > cost`` / ``net > 0`` rule; no tolerance may turn a tie into a win.
    opportunity = mfe > cost + buffer_return
    capture = net > buffer_return
    if bool((capture & ~opportunity).any()):
        raise ValueError("exact-policy capture must imply executable opportunity")
    work[buffer_column(buffer_bps, "opportunity")] = opportunity.astype(np.int8)
    work[buffer_column(buffer_bps, "capture")] = capture.astype(np.int8)
    work["cost_buffer_return"] = buffer_return
    return work


def compose_opportunity_capture(
    opportunity_probability: np.ndarray | pd.Series,
    capture_given_opportunity_probability: np.ndarray | pd.Series,
) -> np.ndarray:
    opportunity = np.asarray(opportunity_probability, dtype=float)
    capture = np.asarray(capture_given_opportunity_probability, dtype=float)
    if opportunity.shape != capture.shape:
        raise ValueError("conditional probability vectors differ in shape")
    if not np.isfinite(opportunity).all() or not np.isfinite(capture).all():
        raise ValueError("conditional probability vectors must be finite")
    if ((opportunity < 0.0) | (opportunity > 1.0)).any() or (
        (capture < 0.0) | (capture > 1.0)
    ).any():
        raise ValueError("conditional probability vectors must be bounded")
    return opportunity * capture


def _classification(target: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    result = classification_metrics(target, prediction)
    positives = int(np.asarray(target, dtype=int).sum())
    result.update({"positive_rows": positives, "negative_rows": int(len(target) - positives)})
    values = np.asarray(target, dtype=int)
    order = np.argsort(-np.asarray(prediction), kind="stable")
    for percent in (1, 5, 10, 20):
        count = max(1, int(math.ceil(len(target) * percent / 100.0)))
        selected = values[order[:count]]
        result[f"top{percent}_rows"] = count
        result[f"top{percent}_precision"] = float(selected.mean())
        result[f"top{percent}_recall"] = (
            float(selected.sum() / positives) if positives else float("nan")
        )
    return result


def _frozen_task_for(head: str) -> str:
    if head == "opportunity":
        return "any_touch"
    if head == "capture_given_opportunity":
        return "capture_given_touch"
    raise ValueError(f"unexpected head {head}")


def load_frozen_geometry(
    report_path: Path = FROZEN_RESET_REPORT,
    runner_path: Path = FROZEN_RESET_RUNNER,
) -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, Any]]:
    """Load only primary-v2 geometry, proving the report and runner provenance."""

    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("schema") != "meaningful_mfe_exact_grid_reset_v1":
        raise ValueError("frozen source is not the exact-grid-reset report")
    if report.get("status") != "COMPLETED_DIAGNOSTIC_EXACT_GRID_NO_PROMOTION":
        raise ValueError("frozen source report has unexpected status")
    source_runner = report.get("runner", {})
    expected_hash = source_runner.get("sha256")
    if not isinstance(expected_hash, str) or sha256(runner_path) != expected_hash:
        raise ValueError("frozen primary runner hash does not match source report")
    source_winners = report.get("frozen_winners")
    if not isinstance(source_winners, Mapping):
        raise ValueError("frozen source lacks winner table")
    geometry: dict[str, dict[str, dict[str, Any]]] = {}
    for family in MODEL_GRIDS:
        geometry[family] = {}
        for side in SIDES:
            source_side = source_winners.get(family, {}).get(side, {})
            geometry[family][side] = {}
            for head in ("opportunity", "capture_given_opportunity"):
                source = source_side.get(_frozen_task_for(head))
                if not isinstance(source, Mapping) or not isinstance(source.get("params"), Mapping):
                    raise ValueError(f"frozen primary geometry missing {family}/{side}/{head}")
                selected = source.get("selected_features")
                if not isinstance(selected, list) or not selected:
                    raise ValueError(f"frozen primary feature-count provenance missing {family}/{side}/{head}")
                geometry[family][side][head] = {
                    "params": dict(source["params"]),
                    # Do not reuse outcome-selected source features: this runner
                    # reselects them inside every allowed training split.
                    "feature_count": len(selected),
                    "source_task": _frozen_task_for(head),
                }
    provenance = {
        "report_path": report_path,
        "report_sha256": sha256(report_path),
        "primary_runner_path": runner_path,
        "primary_runner_sha256": sha256(runner_path),
        "source_status": report["status"],
        "geometry_only": True,
        "features_reselected_train_only": True,
    }
    return geometry, provenance


def economic_metrics(
    frame: pd.DataFrame,
    score_column: str,
    *,
    buffer_bps: int,
    fraction: float,
    scope: str,
    side: str,
) -> dict[str, Any]:
    local = frame if side == "pooled" else frame.loc[frame["side_name"].astype(str).eq(side)]
    selected = stable_top(local, score_column, fraction=fraction)
    net = pd.to_numeric(selected["execution_net_ev_12h"], errors="raise")
    gross = pd.to_numeric(selected["execution_gross_ev_12h"], errors="raise")
    cost = pd.to_numeric(selected["execution_cost_return"], errors="raise")
    if not np.allclose(gross.to_numpy(float) - cost.to_numpy(float), net.to_numpy(float), atol=1e-7, rtol=0.0):
        raise ValueError("exact cost identity failed in global top-k evaluation")
    opportunity_column = buffer_column(buffer_bps, "opportunity")
    capture_column = buffer_column(buffer_bps, "capture")
    selected_opportunity = selected[opportunity_column].astype(bool)
    selected_capture = selected[capture_column].astype(bool)
    population_opportunity = int(local[opportunity_column].astype(bool).sum())
    population_capture = int(local[capture_column].astype(bool).sum())
    opportunity_prevalence = population_opportunity / len(local)
    capture_prevalence = population_capture / len(local)
    opportunity_precision = float(selected_opportunity.mean())
    capture_precision = float(selected_capture.mean())
    buffer_return = float(buffer_bps) / 1e4
    return {
        "evaluation": scope,
        "side": side,
        "score": score_column,
        "buffer_bps": buffer_bps,
        "selected_fraction": fraction,
        "population_rows": len(local),
        "selected_rows": len(selected),
        "net_ev_bps": float(net.mean() * 1e4),
        "net_minus_buffer_bps": float((net.mean() - buffer_return) * 1e4),
        "gross_ev_bps": float(gross.mean() * 1e4),
        "cost_bps": float(cost.mean() * 1e4),
        "positive_net_rate": float((net > 0.0).mean()),
        "loss_rate": float((net < 0.0).mean()),
        "opportunity_prevalence": opportunity_prevalence,
        "opportunity_precision": opportunity_precision,
        "opportunity_recall": (
            float(selected_opportunity.sum() / population_opportunity)
            if population_opportunity
            else float("nan")
        ),
        "opportunity_lift": (
            float(opportunity_precision / opportunity_prevalence)
            if opportunity_prevalence > 0.0
            else float("nan")
        ),
        "capture_prevalence": capture_prevalence,
        "capture_precision": capture_precision,
        "capture_recall": (
            float(selected_capture.sum() / population_capture)
            if population_capture
            else float("nan")
        ),
        "capture_lift": (
            float(capture_precision / capture_prevalence)
            if capture_prevalence > 0.0
            else float("nan")
        ),
        "cvar5_bps": float(net.nsmallest(max(1, math.ceil(len(net) * 0.05))).mean() * 1e4),
        "asset_count": int(selected["__symbol__"].nunique()),
        "largest_asset_share": float(selected["__symbol__"].value_counts(normalize=True).max()),
        "long_selected_rows": int(selected["side_name"].astype(str).eq("long").sum()),
        "short_selected_rows": int(selected["side_name"].astype(str).eq("short").sum()),
        "long_selected_share": float(selected["side_name"].astype(str).eq("long").mean()),
    }


def _fit_predict(
    panel: pd.DataFrame,
    matrix: pd.DataFrame,
    train: np.ndarray,
    evaluation: np.ndarray,
    geometry: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    buffer_bps: int,
    split_name: str,
    seed: int,
    validation_days: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    work = derive_cost_buffer_targets(panel, buffer_bps)
    opportunity_target = buffer_column(buffer_bps, "opportunity")
    capture_target = buffer_column(buffer_bps, "capture")
    keep = [
        *IDENTITY, "label_resolution_utc", "execution_net_ev_12h",
        "execution_gross_ev_12h", "execution_cost_return", "execution_mfe_return_12h",
        opportunity_target, capture_target,
    ]
    scored = work.iloc[evaluation][keep].copy().reset_index(drop=True)
    metric_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    selection_cache: dict[tuple[str, int, str, int], tuple[list[str], pd.DataFrame]] = {}
    for family_index, family in enumerate(MODEL_GRIDS):
        for side_index, side in enumerate(SIDES):
            side_train = train[work.iloc[train]["side_name"].astype(str).eq(side).to_numpy()]
            side_evaluation = evaluation[work.iloc[evaluation]["side_name"].astype(str).eq(side).to_numpy()]
            output_positions = np.flatnonzero(scored["side_name"].astype(str).eq(side).to_numpy())
            if len(side_evaluation) != len(output_positions):
                raise AssertionError("side output position mismatch")
            for head_index, (head, target, conditional) in enumerate((
                ("opportunity", opportunity_target, False),
                ("capture_given_opportunity", capture_target, True),
            )):
                task_train = side_train if not conditional else side_train[work.iloc[side_train][opportunity_target].astype(bool).to_numpy()]
                task_evaluation = side_evaluation if not conditional else side_evaluation[work.iloc[side_evaluation][opportunity_target].astype(bool).to_numpy()]
                if len(task_train) < 500 or len(task_evaluation) < 100:
                    raise ValueError(f"insufficient support {split_name}/{buffer_bps}/{family}/{side}/{head}: train={len(task_train)}, eval={len(task_evaluation)}")
                train_classes = work.iloc[task_train][target].value_counts()
                evaluation_classes = work.iloc[task_evaluation][target].value_counts()
                if (
                    set(train_classes.index.astype(int)) != {0, 1}
                    or int(train_classes.min()) < 100
                    or set(evaluation_classes.index.astype(int)) != {0, 1}
                    or int(evaluation_classes.min()) < 25
                ):
                    raise ValueError(
                        f"insufficient class support {split_name}/{buffer_bps}/"
                        f"{family}/{side}/{head}: train={train_classes.to_dict()}, "
                        f"eval={evaluation_classes.to_dict()}"
                    )
                frozen = geometry[family][side][head]
                # Selection is target/split/side local.  Cache only the exact
                # same (side, buffer, head, feature count) request across model
                # families; it can never cross a label or side boundary.
                cache_key = (side, buffer_bps, head, int(frozen["feature_count"]))
                cached = selection_cache.get(cache_key)
                if cached is None:
                    cached = select_features_nested(
                        matrix, work[target].to_numpy(float), task_train,
                        int(frozen["feature_count"]),
                    )
                    selection_cache[cache_key] = cached
                selected, screen = cached
                model = fit_model(
                    family, frozen["params"], matrix.iloc[task_train][selected],
                    work.iloc[task_train][target].to_numpy(float), soft=False,
                    seed=seed + family_index * 100_000 + side_index * 10_000 + head_index * 100,
                )
                prediction = predict_model(model, family, matrix.iloc[side_evaluation][selected])
                column = f"p_{family}_{head}_{buffer_bps}bps"
                scored.loc[output_positions, column] = prediction
                lookup = pd.Series(np.arange(len(side_evaluation)), index=side_evaluation)
                metric_positions = lookup.loc[task_evaluation].to_numpy(int)
                metric_rows.append({
                    "evaluation": split_name, "buffer_bps": buffer_bps, "family": family,
                    "side": side, "head": head, "train_rows": len(task_train),
                    "evaluation_rows": len(task_evaluation),
                    "training_label_resolution_max": work.iloc[task_train]["label_resolution_utc"].max(),
                    "validation_days": "|".join(validation_days) if validation_days else "",
                    **_classification(work.iloc[task_evaluation][target].to_numpy(int), prediction[metric_positions]),
                })
                selection_rows.append({
                    "evaluation": split_name, "buffer_bps": buffer_bps, "family": family,
                    "side": side, "head": head, "selected_feature_count": len(selected),
                    "selected_features": json.dumps(selected),
                    "screen_top20": json.dumps(_safe(screen.head(20).to_dict("records")), sort_keys=True),
                    "frozen_source_task": frozen["source_task"], "frozen_params": json.dumps(frozen["params"], sort_keys=True),
                })
    for family in MODEL_GRIDS:
        score = f"score_{family}_opportunity_capture_{buffer_bps}bps"
        scored[score] = compose_opportunity_capture(
            scored[f"p_{family}_opportunity_{buffer_bps}bps"],
            scored[f"p_{family}_capture_given_opportunity_{buffer_bps}bps"],
        )
    scored["evaluation"] = split_name
    scored["buffer_bps"] = buffer_bps
    return scored, metric_rows, selection_rows


def _aggregate_july_head_metrics(
    scored: pd.DataFrame,
    buffer_bps: int,
) -> list[dict[str, Any]]:
    opportunity_target = buffer_column(buffer_bps, "opportunity")
    capture_target = buffer_column(buffer_bps, "capture")
    rows: list[dict[str, Any]] = []
    for family in MODEL_GRIDS:
        for side in SIDES:
            local = scored.loc[scored["side_name"].astype(str).eq(side)]
            for head, target, conditional in (
                ("opportunity", opportunity_target, False),
                ("capture_given_opportunity", capture_target, True),
            ):
                metric_local = (
                    local
                    if not conditional
                    else local.loc[local[opportunity_target].astype(bool)]
                )
                prediction = metric_local[
                    f"p_{family}_{head}_{buffer_bps}bps"
                ].to_numpy(float)
                rows.append(
                    {
                        "evaluation": "july_grouped_oof",
                        "buffer_bps": buffer_bps,
                        "family": family,
                        "side": side,
                        "head": head,
                        "train_rows": np.nan,
                        "evaluation_rows": len(metric_local),
                        "training_label_resolution_max": pd.NaT,
                        "validation_days": "five_contiguous_two_day_blocks",
                        **_classification(
                            metric_local[target].to_numpy(int),
                            prediction,
                        ),
                    }
                )
    return rows


def _economics(scored: pd.DataFrame, evaluation: str) -> list[dict[str, Any]]:
    scores = [column for column in scored if column.startswith("score_")]
    buffers = scored["buffer_bps"].unique()
    if len(buffers) != 1:
        raise ValueError("economic evaluation must contain exactly one buffer")
    buffer_bps = int(buffers[0])
    return [
        economic_metrics(
            scored,
            score,
            buffer_bps=buffer_bps,
            fraction=fraction,
            scope=evaluation,
            side=side,
        )
        for score in scores for fraction in (0.01, 0.05, 0.10, 0.20)
        for side in ("pooled", *SIDES)
    ]


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    panel, matrix, raw_features, lineage = load_panel(
        args.features, args.feature_manifest, args.grid, args.grid_manifest, grid_name=args.grid_name
    )
    geometry, provenance = load_frozen_geometry(args.frozen_report, args.frozen_runner)
    all_scored: list[pd.DataFrame] = []
    all_metrics: list[dict[str, Any]] = []
    all_selections: list[dict[str, Any]] = []
    all_economics: list[dict[str, Any]] = []
    splits: list[dict[str, Any]] = []
    for buffer_index, buffer_bps in enumerate(BUFFERS_BPS):
        for spec_index, spec in enumerate(TRANSFER_SPECS):
            train, evaluation = _base_masks(panel, spec)
            scored, metrics, selections = _fit_predict(
                panel, matrix, train, evaluation, geometry, buffer_bps=buffer_bps,
                split_name=spec.name, seed=args.seed + buffer_index * 10_000_000 + spec_index * 1_000_000,
            )
            all_scored.append(scored); all_metrics.extend(metrics); all_selections.extend(selections)
            all_economics.extend(_economics(scored, spec.name))
            splits.append({"name": spec.name, "buffer_bps": buffer_bps, "train_rows": len(train), "evaluation_rows": len(evaluation), "source_forward_split_promotable": spec.promotable, "promotion_eligible": False, "note": "COST_BUFFER_LABEL_SENSITIVITY_DIAGNOSTIC_NONPROMOTABLE"})
        july_scored: list[pd.DataFrame] = []
        for fold_index, (name, train, evaluation, days) in enumerate(july_grouped_day_folds(panel)):
            scored, metrics, selections = _fit_predict(
                panel, matrix, train, evaluation, geometry, buffer_bps=buffer_bps,
                split_name=name, seed=args.seed + buffer_index * 10_000_000 + 5_000_000 + fold_index * 1_000_000,
                validation_days=days,
            )
            july_scored.append(scored); all_metrics.extend(metrics); all_selections.extend(selections)
            splits.append({"name": name, "buffer_bps": buffer_bps, "train_rows": len(train), "evaluation_rows": len(evaluation), "validation_days": days, "source_forward_split_promotable": False, "promotion_eligible": False, "note": "GROUPED_JULY_OOF_COST_BUFFER_DIAGNOSTIC_NONPROMOTABLE"})
        july_oof = pd.concat(july_scored, ignore_index=True)
        if july_oof.duplicated(list(IDENTITY)).any():
            raise ValueError("July grouped OOF produced duplicate identities")
        july_oof["evaluation"] = "july_grouped_oof"
        all_scored.append(july_oof)
        all_metrics.extend(_aggregate_july_head_metrics(july_oof, buffer_bps))
        all_economics.extend(_economics(july_oof, "july_grouped_oof"))
    outputs: dict[str, Any] = {}
    for name, frame in (("predictions", pd.concat(all_scored, ignore_index=True)), ("head_metrics", pd.DataFrame(all_metrics)), ("feature_selections", pd.DataFrame(all_selections)), ("economics", pd.DataFrame(all_economics))):
        path = args.output_dir / f"{name}.parquet"; frame.to_parquet(path, index=False)
        outputs[name] = {"path": path, "rows": len(frame), "sha256": sha256(path)}
    report = {
        "schema": SCHEMA, "status": "COMPLETED_DIAGNOSTIC_COST_BUFFER_NO_PROMOTION", "promotion_eligible": False,
        "runner": {"path": Path(__file__).resolve(), "sha256": sha256(Path(__file__).resolve())},
        "lineage": lineage, "frozen_geometry_provenance": provenance, "raw_feature_count": len(raw_features), "splits": splits,
        "contracts": {
            "opportunity": "execution_mfe_return_12h > execution_cost_return + buffer_return",
            "capture": "execution_net_ev_12h > buffer_return, trained only conditional on declared opportunity",
            "composition": "p(opportunity) * p(capture | opportunity)",
            "buffers_bps": list(BUFFERS_BPS),
            "selection": "features reselected train-only per split/side/head; primary-v2 model geometry frozen, never re-HPOed",
            "economics": "one pooled global top 1/5/10/20% per score; long/short are diagnostics; exact cost recorded once",
            "validation": "May->June, June->July, reverse June diagnostics, five grouped July OOF folds with source +/-12h purge",
        }, "outputs": outputs,
    }
    report_path = args.output_dir / "report.json"; _write_json(report_path, report)
    _write_json(args.output_dir / "manifest.json", {"schema": SCHEMA, "status": report["status"], "promotion_eligible": False, "report": {"path": report_path, "sha256": sha256(report_path)}, "outputs": outputs})
    return report


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--features", type=Path, default=Path("data_perp/artifacts/exact_policy_capture_feature_universe_20260727_v2/capture_feature_universe.parquet"))
    value.add_argument("--feature-manifest", type=Path, default=Path("data_perp/artifacts/exact_policy_capture_feature_universe_20260727_v2/manifest.json"))
    value.add_argument("--grid", type=Path, default=Path("data_perp/artifacts/meaningful_mfe_exact_policy_label_grid_20260727_v1/meaningful_mfe_label_grid.parquet"))
    value.add_argument("--grid-manifest", type=Path, default=Path("data_perp/artifacts/meaningful_mfe_exact_policy_label_grid_20260727_v1/manifest.json"))
    value.add_argument("--output-dir", type=Path, default=Path("data_perp/artifacts/meaningful_mfe_exact_grid_cost_buffer_sensitivity_20260730_v1"))
    value.add_argument("--frozen-report", type=Path, default=FROZEN_RESET_REPORT)
    value.add_argument("--frozen-runner", type=Path, default=FROZEN_RESET_RUNNER)
    value.add_argument("--grid-name", choices=("h12_u1p5atr", "h12_u2p0atr"), default="h12_u1p5atr")
    value.add_argument("--seed", type=int, default=20260730)
    return value


if __name__ == "__main__":
    run(parser().parse_args())
