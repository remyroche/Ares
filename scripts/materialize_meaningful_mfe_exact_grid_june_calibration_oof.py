#!/usr/bin/env python3
"""Materialise the strict June OOF seed for June-to-July causal mapping.

This deliberately does *not* fit an EV map.  It produces the score-aligned,
strictly prior June OOF ledger that a later mapping evaluator needs to apply a
21-day causal global or side-shrunk isotonic score-to-net-EV map to the frozen
July 1--10 predictions from the exact-grid reset v2.

The reset's model geometry and winner feature-count contracts are loaded from
its immutable report.  No HPO is rerun.  Feature screening remains inside
each earlier-June training fold, exactly as it is in the reset transfer
runner; consequently every OOF score is made by a model that did not fit its
row.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_meaningful_mfe_exact_grid_reset import (  # noqa: E402
    IDENTITY,
    JULY_START,
    JUNE_START,
    _fit_predict_split,
    load_panel,
    sha256,
)


SCHEMA = "meaningful_mfe_exact_grid_june_calibration_oof_v1"
RESET_SCHEMA = "meaningful_mfe_exact_grid_reset_v1"
DEFAULT_RESET_DIR = ROOT / "data_perp/artifacts/meaningful_mfe_exact_grid_reset_20260730_v2"
DEFAULT_OUTPUT = (
    ROOT / "data_perp/artifacts/meaningful_mfe_exact_grid_june_calibration_oof_20260730_v1"
)
JUNE_FOLD_STARTS = (
    pd.Timestamp("2026-06-10T00:00:00Z"),
    pd.Timestamp("2026-06-17T00:00:00Z"),
    pd.Timestamp("2026-06-24T00:00:00Z"),
)
PREDECLARED_SCORES = tuple(
    [
        f"p_{family}_{event}"
        for family in ("logistic", "lightgbm", "catboost")
        for event in ("any_touch", "clean_first", "soft_triple_barrier")
    ]
    + [
        f"score_{family}_{composition}"
        for family in ("logistic", "lightgbm", "catboost")
        for composition in ("touch_capture", "clean_capture")
    ]
)
EVENT_COLUMNS = ("any_touch", "clean_first", "positive_net", "timeout")
EXACT_COLUMNS = (
    "execution_net_ev_12h",
    "execution_gross_ev_12h",
    "execution_cost_return",
    "execution_exit_reason",
    "execution_mfe_return_12h",
    "execution_mae_return_12h",
)


@dataclass(frozen=True)
class JuneFold:
    """One earlier-June train / seven-day validation OOF fold."""

    name: str
    start: pd.Timestamp
    end: pd.Timestamp


def _safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, (Path, pd.Timestamp)):
        return str(value)
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


def _canonical_hash(value: Mapping[str, Any]) -> str:
    payload = json.dumps(_safe(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def june_folds() -> tuple[JuneFold, ...]:
    ends = (*JUNE_FOLD_STARTS[1:], JULY_START)
    return tuple(
        JuneFold(f"june_oof_fold_{index}", start, end)
        for index, (start, end) in enumerate(zip(JUNE_FOLD_STARTS, ends))
    )


def june_fold_masks(panel: pd.DataFrame, fold: JuneFold) -> tuple[np.ndarray, np.ndarray]:
    """Return exactly earlier-June purged training and validation positions."""

    required = {"__ts__", "execution_decision_utc", "label_resolution_utc"}
    missing = sorted(required.difference(panel.columns))
    if missing:
        raise ValueError(f"June fold panel lacks {missing}")
    signal = pd.to_datetime(panel["__ts__"], utc=True, errors="raise")
    decision = pd.to_datetime(
        panel["execution_decision_utc"], utc=True, errors="raise"
    )
    resolved = pd.to_datetime(
        panel["label_resolution_utc"], utc=True, errors="raise"
    )
    if not (JUNE_START <= fold.start < fold.end <= JULY_START):
        raise ValueError(f"{fold.name} is outside the canonical June window")
    train = np.flatnonzero(
        signal.ge(JUNE_START).to_numpy()
        & signal.lt(fold.start).to_numpy()
        & resolved.lt(fold.start).to_numpy()
        & decision.lt(fold.start - pd.Timedelta(hours=12)).to_numpy()
    )
    validation = np.flatnonzero(
        signal.ge(fold.start).to_numpy() & signal.lt(fold.end).to_numpy()
    )
    if not len(train) or not len(validation):
        raise ValueError(f"{fold.name} has empty train or validation support")
    if not bool((resolved.iloc[train] < fold.start).all()):
        raise ValueError(f"{fold.name} has unresolved training labels")
    if not bool(
        (decision.iloc[train] < fold.start - pd.Timedelta(hours=12)).all()
    ):
        raise ValueError(f"{fold.name} violates the 12-hour decision purge")
    return train, validation


def load_frozen_winners(report_path: Path) -> tuple[dict[str, Any], str, dict[str, Any]]:
    """Load, rather than recreate, the v2 HPO/feature winner contract."""

    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("schema") != RESET_SCHEMA:
        raise ValueError("frozen winner report has an unexpected schema")
    if report.get("status") != "COMPLETED_DIAGNOSTIC_EXACT_GRID_NO_PROMOTION":
        raise ValueError("frozen winner report is not the completed exact reset")
    winners = report.get("frozen_winners")
    if not isinstance(winners, dict):
        raise ValueError("frozen winner report lacks frozen_winners")
    for family in ("logistic", "lightgbm", "catboost"):
        for side in ("long", "short"):
            for task in (
                "any_touch",
                "clean_first",
                "capture_given_touch",
                "capture_given_clean",
                "soft_triple_barrier",
            ):
                item = winners.get(family, {}).get(side, {}).get(task, {})
                if not isinstance(item.get("selected_features"), list) or not item.get(
                    "params"
                ):
                    raise ValueError(f"frozen winner missing {family}/{side}/{task}")
    runner = report.get("runner", {})
    runner_path = Path(str(runner.get("path", "")))
    runner_sha = str(runner.get("sha256", ""))
    if not runner_path.is_file() or sha256(runner_path) != runner_sha:
        raise ValueError("frozen reset runner binding does not match current code")
    recipe = {
        "winners": winners,
        "reset_runner_sha256": runner_sha,
        "grid_name": report.get("lineage", {})
        .get("labels", {})
        .get("grid_name"),
    }
    return winners, _canonical_hash(recipe), report


def _select_scores(frame: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(PREDECLARED_SCORES).difference(frame.columns))
    if missing:
        raise ValueError(f"scored frame lacks predeclared scores: {missing}")
    timing = [
        column
        for column in ("execution_decision_utc", "label_resolution_utc")
        if column in frame.columns
    ]
    result = frame.loc[
        :,
        [
            *IDENTITY,
            *timing,
            *EXACT_COLUMNS,
            *EVENT_COLUMNS,
            *PREDECLARED_SCORES,
        ],
    ].copy()
    for column in PREDECLARED_SCORES:
        values = pd.to_numeric(result[column], errors="coerce").to_numpy(float)
        if not np.isfinite(values).all():
            raise ValueError(f"non-finite {column} in calibration ledger")
    return result


def _stamp(
    frame: pd.DataFrame,
    *,
    source_partition: str,
    is_oof: bool,
    fold: str,
    model_available_at: pd.Timestamp,
    training_decision_cutoff: pd.Timestamp,
    training_label_resolution_max: pd.Timestamp,
    recipe_hash: str,
) -> pd.DataFrame:
    result = frame.copy()
    result["execution_decision_utc"] = pd.to_datetime(
        result["execution_decision_utc"], utc=True, errors="raise"
    )
    result["label_resolution_utc"] = pd.to_datetime(
        result["label_resolution_utc"], utc=True, errors="raise"
    )
    result["prediction_available_at"] = result["execution_decision_utc"]
    result["model_available_at"] = pd.Timestamp(model_available_at)
    result["training_decision_cutoff"] = pd.Timestamp(training_decision_cutoff)
    result["training_label_resolution_max"] = pd.Timestamp(
        training_label_resolution_max
    )
    result["source_partition"] = source_partition
    result["is_oof"] = bool(is_oof)
    result["is_frozen_forward_oos"] = not bool(is_oof)
    result["fold"] = str(fold)
    result["score_recipe_hash"] = str(recipe_hash)
    result["execution_label_end_utc"] = result["label_resolution_utc"]
    if not bool(
        result["label_resolution_utc"].eq(
            result["execution_decision_utc"] + pd.Timedelta(hours=12)
        ).all()
    ):
        raise ValueError("stamped rows violate the exact 12-hour label contract")
    if not bool(
        result["prediction_available_at"].le(result["execution_decision_utc"])
        .all()
    ):
        raise ValueError("a score is unavailable at its execution decision")
    if not bool(
        result["model_available_at"].le(result["execution_decision_utc"]).all()
    ):
        raise ValueError("a fitted model is unavailable at its execution decision")
    if is_oof:
        if not bool(
            result["training_label_resolution_max"].lt(
                result["execution_decision_utc"]
            ).all()
        ):
            raise ValueError("OOF score training outcomes reach its decision")
        if not bool(
            result["training_decision_cutoff"].lt(
                result["execution_decision_utc"]
            ).all()
        ):
            raise ValueError("OOF score training decisions reach its decision")
    return result


def _attach_panel_contract(
    scored: pd.DataFrame, panel: pd.DataFrame
) -> pd.DataFrame:
    """Attach immutable exact timing and prove economics against the signed panel."""

    required = {
        *IDENTITY,
        "execution_decision_utc",
        "label_resolution_utc",
        *EXACT_COLUMNS,
        *EVENT_COLUMNS,
    }
    missing = sorted(required.difference(panel.columns))
    if missing:
        raise ValueError(f"signed panel lacks {missing}")
    source = panel.loc[:, list(required)].copy()
    if source.duplicated(list(IDENTITY)).any():
        raise ValueError("signed panel has duplicate identities")
    merged = scored.merge(source, on=list(IDENTITY), how="left", suffixes=("", "__panel"), validate="one_to_one")
    if merged["execution_decision_utc"].isna().any():
        raise ValueError("scored rows are not covered by the signed panel")
    for column in (*EXACT_COLUMNS, *EVENT_COLUMNS):
        left = merged[column]
        right = merged[f"{column}__panel"]
        if column == "execution_exit_reason":
            same = left.astype(str).eq(right.astype(str))
        else:
            same = np.isclose(
                pd.to_numeric(left, errors="raise").to_numpy(float),
                pd.to_numeric(right, errors="raise").to_numpy(float),
                rtol=0.0,
                atol=1e-10,
            )
        if not bool(np.all(same)):
            raise ValueError(f"scored rows differ from signed panel: {column}")
        merged = merged.drop(columns=f"{column}__panel")
    return merged


def build_june_oof(
    panel: pd.DataFrame,
    matrix: pd.DataFrame,
    winners: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    recipe_hash: str,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit only the three predefined earlier-June OOF validation blocks."""

    ledgers: list[pd.DataFrame] = []
    audits: list[dict[str, Any]] = []
    for fold_index, fold in enumerate(june_folds()):
        train, validation = june_fold_masks(panel, fold)
        scored, _, _ = _fit_predict_split(
            panel,
            matrix,
            train,
            validation,
            winners,
            split_name=fold.name,
            seed=int(seed) + fold_index * 1_000_000,
            validation_days=None,
        )
        selected = _select_scores(_attach_panel_contract(scored, panel))
        label_max = pd.to_datetime(
            panel.iloc[train]["label_resolution_utc"], utc=True, errors="raise"
        ).max()
        decision_cutoff = fold.start - pd.Timedelta(hours=12)
        ledgers.append(
            _stamp(
                selected,
                source_partition="june_calibration_oof",
                is_oof=True,
                fold=fold.name,
                model_available_at=fold.start,
                training_decision_cutoff=decision_cutoff,
                training_label_resolution_max=label_max,
                recipe_hash=recipe_hash,
            )
        )
        audits.append(
            {
                "fold": fold.name,
                "validation_start": fold.start,
                "validation_end": fold.end,
                "train_rows": int(len(train)),
                "validation_rows": int(len(validation)),
                "training_decision_cutoff": decision_cutoff,
                "training_label_resolution_max": label_max,
            }
        )
    result = pd.concat(ledgers, ignore_index=True)
    if result.duplicated(list(IDENTITY)).any():
        raise ValueError("June OOF folds overlap")
    return result, pd.DataFrame(audits)


def append_frozen_july(
    frozen_predictions: pd.DataFrame,
    panel: pd.DataFrame,
    report: Mapping[str, Any],
    *,
    recipe_hash: str,
) -> pd.DataFrame:
    """Append the existing June-trained, July-forward frozen scores unchanged."""

    if "evaluation" not in frozen_predictions.columns:
        raise ValueError("reset predictions lack evaluation provenance")
    frozen = frozen_predictions.loc[
        frozen_predictions["evaluation"].astype(str).eq("june_to_july")
    ].copy()
    if frozen.empty:
        raise ValueError("reset predictions lack june_to_july rows")
    selected = _select_scores(_attach_panel_contract(frozen, panel))
    spec = next(
        (item for item in report.get("splits", []) if item.get("name") == "june_to_july"),
        None,
    )
    if not isinstance(spec, Mapping) or bool(spec.get("promotion_eligible")) is not True:
        raise ValueError("reset report lacks promotable June-to-July split provenance")
    available_at = pd.Timestamp(spec["evaluation_start"])
    cutoff = available_at - pd.Timedelta(hours=12)
    label_max = pd.Timestamp(spec["training_label_resolution_max"])
    return _stamp(
        selected,
        source_partition="june_to_july_frozen_forward_oos",
        is_oof=False,
        fold="frozen_june_to_july",
        model_available_at=available_at,
        training_decision_cutoff=cutoff,
        training_label_resolution_max=label_max,
        recipe_hash=recipe_hash,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite immutable output {args.output_dir}")
    winners, recipe_hash, report = load_frozen_winners(args.reset_report)
    panel, matrix, _, lineage = load_panel(
        args.features,
        args.feature_manifest,
        args.grid,
        args.grid_manifest,
    )
    june_oof, fold_audit = build_june_oof(
        panel, matrix, winners, recipe_hash=recipe_hash, seed=args.seed
    )
    frozen = append_frozen_july(
        pd.read_parquet(args.reset_predictions), panel, report, recipe_hash=recipe_hash
    )
    ledger = pd.concat([june_oof, frozen], ignore_index=True)
    if ledger.duplicated(list(IDENTITY)).any():
        raise ValueError("OOF seed and frozen July predictions overlap")
    ledger = ledger.sort_values(
        ["execution_decision_utc", "candidate_id", "__symbol__", "side_name"],
        kind="stable",
    ).reset_index(drop=True)
    if not bool(ledger.loc[ledger["is_oof"], "__ts__"].lt(JULY_START).all()):
        raise ValueError("June calibration OOF contains post-June signals")
    if not bool(ledger.loc[~ledger["is_oof"], "__ts__"].ge(JULY_START).all()):
        raise ValueError("frozen July segment contains pre-July signals")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    ledger_path = args.output_dir / "calibration_ledger.parquet"
    audit_path = args.output_dir / "fold_audit.parquet"
    ledger.to_parquet(ledger_path, index=False, compression="zstd")
    fold_audit.to_parquet(audit_path, index=False, compression="zstd")
    manifest = {
        "schema": SCHEMA,
        "status": "COMPLETED_JUNE_OOF_SEED_AND_FROZEN_JULY_NO_EV_MAPPING",
        "promotion_eligible": False,
        "contract": {
            "purpose": "June-to-July causal 21-day mapping seed only; no map fit here",
            "scores": list(PREDECLARED_SCORES),
            "folds": [
                {"name": fold.name, "start": fold.start, "end": fold.end}
                for fold in june_folds()
            ],
            "train_rule": (
                "June-only signal < fold start, label_resolution < fold start, "
                "and execution_decision < fold start - 12h"
            ),
            "frozen_winners": "loaded from exact reset v2 report; HPO not rerun",
            "frozen_july": "existing june_to_july predictions appended unchanged",
            "exact_economics": "gross - row cost = net; no second cost subtraction",
        },
        "lineage": {
            "reset_report": {"path": str(args.reset_report), "sha256": sha256(args.reset_report)},
            "reset_predictions": {"path": str(args.reset_predictions), "sha256": sha256(args.reset_predictions)},
            "winner_recipe_hash": recipe_hash,
            "panel": lineage,
        },
        "outputs": {
            "ledger": {"path": str(ledger_path), "sha256": sha256(ledger_path), "rows": int(len(ledger))},
            "fold_audit": {"path": str(audit_path), "sha256": sha256(audit_path), "rows": int(len(fold_audit))},
        },
        "runner": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
    }
    _write_json(args.output_dir / "manifest.json", manifest)
    return manifest


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--reset-report", type=Path, default=DEFAULT_RESET_DIR / "report.json")
    value.add_argument("--reset-predictions", type=Path, default=DEFAULT_RESET_DIR / "predictions.parquet")
    value.add_argument("--features", type=Path, default=ROOT / "data_perp/artifacts/exact_policy_capture_feature_universe_20260727_v2/capture_feature_universe.parquet")
    value.add_argument("--feature-manifest", type=Path, default=ROOT / "data_perp/artifacts/exact_policy_capture_feature_universe_20260727_v2/manifest.json")
    value.add_argument("--grid", type=Path, default=ROOT / "data_perp/artifacts/meaningful_mfe_exact_policy_label_grid_20260727_v1/meaningful_mfe_label_grid.parquet")
    value.add_argument("--grid-manifest", type=Path, default=ROOT / "data_perp/artifacts/meaningful_mfe_exact_policy_label_grid_20260727_v1/manifest.json")
    value.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    value.add_argument("--seed", type=int, default=20260730)
    return value


if __name__ == "__main__":
    run(parser().parse_args())
