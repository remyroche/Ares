#!/usr/bin/env python3
"""Create compact, fold-local multiview panels for regime and transition use.

This is a selection/materialization stage only: it does not fit a regime or a
transition classifier, and therefore does not emit probabilities.  The manifest
reserves distinct namespaces for those future model outputs and documents the
lossless adapter used by interaction discovery.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Sequence

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.regime_multiview_selection import (  # noqa: E402
    MultiviewSelectionConfig,
    select_fold_local_multiview_features,
)
from scripts.materialize_multiview_regime_panel import sha256  # noqa: E402
from scripts.run_regime_transition_active_head_chronological_oos import (  # noqa: E402
    conservative_label_available_utc,
)


DEFAULT_PANEL = ROOT / "data_perp/artifacts/regime_multiview_panel_2022_2026_20260730_v1"
DEFAULT_LEDGER = ROOT / "data_perp/artifacts/regime_episode_ledger_2022_2026_20260730_v1"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/fold_local_multiview_selection_2022_2026_20260730_v2"
SCHEMA = "fold_local_multiview_selection_materialization_v1"


def _safe(value: Any) -> Any:
    if isinstance(value, (Path, pd.Timestamp, pd.Timedelta)):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _verify_manifest(root: Path) -> dict[str, Any]:
    manifest, signature = root / "manifest.json", root / "manifest.sha256"
    if not manifest.exists() or not signature.exists():
        raise FileNotFoundError(f"signed source manifest required under {root}")
    if signature.read_text(encoding="utf-8").strip().split()[0] != sha256(manifest):
        raise ValueError(f"source manifest checksum fails: {manifest}")
    return json.loads(manifest.read_text(encoding="utf-8"))


def chronological_period_folds(
    labels: pd.DataFrame,
    *,
    first_evaluation: str,
    last_evaluation: str,
    minimum_train_months: int,
    frequency: str = "QS",
) -> list[tuple[pd.Timestamp, pd.Timestamp, np.ndarray, np.ndarray]]:
    """Expanding folds whose training labels resolve strictly before evaluation."""

    source = pd.to_datetime(labels["source_utc"], utc=True, errors="raise")
    available = conservative_label_available_utc(labels)
    first = pd.Timestamp(first_evaluation, tz="UTC")
    last = pd.Timestamp(last_evaluation, tz="UTC")
    result: list[tuple[pd.Timestamp, pd.Timestamp, np.ndarray, np.ndarray]] = []
    for start in pd.date_range(first, last, freq=frequency, tz="UTC"):
        end = min(start + pd.tseries.frequencies.to_offset(frequency), last + pd.offsets.MonthBegin(1))
        evaluation = np.flatnonzero(source.ge(start).to_numpy() & source.lt(end).to_numpy())
        train = np.flatnonzero(available.lt(start).to_numpy())
        if not len(evaluation):
            continue
        months = source.iloc[train].dt.tz_localize(None).dt.to_period("M")
        if months.nunique() < int(minimum_train_months):
            continue
        if len(train) and available.iloc[train].max() >= start:
            raise AssertionError("fold includes a label unavailable at evaluation start")
        result.append((start, end, train, evaluation))
    return result


def _load_joined(panel_root: Path, ledger_root: Path) -> tuple[pd.DataFrame, list[str]]:
    panel_path = panel_root / "multiview_regime_features.parquet"
    ledger_path = ledger_root / "hourly_state_calendar.parquet"
    if not panel_path.exists() or not ledger_path.exists():
        raise FileNotFoundError("multiview feature panel and hourly ledger are both required")
    panel = pd.read_parquet(panel_path)
    ledger = pd.read_parquet(
        ledger_path,
        columns=[
            "source_utc", "calendar_segment_id", "target__pooled_state",
            "target__transition_active", "target__available_utc",
        ],
    )
    keys = ["source_utc", "calendar_segment_id"]
    for frame in (panel, ledger):
        frame["source_utc"] = pd.to_datetime(frame["source_utc"], utc=True, errors="raise")
        if frame.duplicated(keys).any():
            raise ValueError("panel/ledger source identity must be unique")
    merged = panel.merge(ledger, on=keys, how="inner", validate="one_to_one")
    if len(merged) != len(panel) or len(merged) != len(ledger):
        raise ValueError("multiview panel and ledger do not have exact source identity coverage")
    features = [name for name in panel.columns if name.startswith("mv__")]
    if not features:
        raise ValueError("multiview panel has no mv__ feature fields")
    # Downstream fold positions must be derived from this exact joined order,
    # rather than assuming a parquet merge happens to retain ledger ordering.
    return merged, features


def _panel_for_folds(
    rows: pd.DataFrame,
    *,
    features: list[str],
    folds: list[tuple[pd.Timestamp, pd.Timestamp, np.ndarray, np.ndarray]],
    config_kwargs: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    regime_parts: list[pd.DataFrame] = []
    transition_parts: list[pd.DataFrame] = []
    regime_selection: list[pd.DataFrame] = []
    transition_selection: list[pd.DataFrame] = []
    audits: list[dict[str, Any]] = []
    for fold_number, (start, end, train, evaluation) in enumerate(folds):
        fold_id = f"{start:%Y%m%d}_{end:%Y%m%d}"
        train_rows = rows.iloc[train]
        config = MultiviewSelectionConfig(fold_id=fold_id, **config_kwargs)
        result = select_fold_local_multiview_features(
            train_rows.loc[:, features],
            config=config,
            regime_train_labels=train_rows["target__pooled_state"],
            transition_train_labels=train_rows["target__transition_active"],
            fold_training_row_ids=train_rows.index,
        )
        identity = rows.iloc[evaluation]["source_utc"].to_frame().reset_index(drop=True)
        identity["calendar_segment_id"] = rows.iloc[evaluation]["calendar_segment_id"].to_numpy()
        identity["fold_id"] = fold_id
        identity["evaluation_start_utc"] = start
        identity["evaluation_end_exclusive_utc"] = end
        regime_parts.append(pd.concat([identity, rows.iloc[evaluation][result.regime_features].reset_index(drop=True)], axis=1))
        transition_parts.append(pd.concat([identity, rows.iloc[evaluation][result.transition_features].reset_index(drop=True)], axis=1))
        for selection, kind, output in (
            (result.lineage.loc[result.lineage["regime_selected"]], "regime", regime_selection),
            (result.lineage.loc[result.lineage["transition_selected"]], "transition", transition_selection),
        ):
            local = selection.copy()
            local.insert(0, "fold_id", fold_id)
            local.insert(1, "selection_kind", kind)
            local.insert(2, "train_rows", int(len(train)))
            local.insert(3, "evaluation_rows", int(len(evaluation)))
            output.append(local)
        audits.append(
            {
                "fold_id": fold_id, "evaluation_start_utc": start, "evaluation_end_exclusive_utc": end,
                "train_rows": int(len(train)), "evaluation_rows": int(len(evaluation)),
                "train_latest_label_available_utc": conservative_label_available_utc(train_rows).max(),
                "regime_features": int(len(result.regime_features)),
                "transition_features": int(len(result.transition_features)),
                "unsupervised_features": int(len(result.unsupervised_features)),
                "regime_selection_train_only": bool(result.diagnostics["regime_supervised_labels_used"]),
                "transition_selection_train_only": bool(result.diagnostics["transition_supervised_labels_used"]),
            }
        )
    return (
        pd.concat(regime_parts, ignore_index=True),
        pd.concat(transition_parts, ignore_index=True),
        pd.concat(regime_selection, ignore_index=True),
        pd.concat(transition_selection, ignore_index=True),
        audits,
    )


def materialize_fold_local_multiview_selection(
    *,
    panel_root: Path = DEFAULT_PANEL,
    ledger_root: Path = DEFAULT_LEDGER,
    output_dir: Path = DEFAULT_OUTPUT,
    first_evaluation: str = "2023-10-01",
    last_evaluation: str = "2026-07-01",
    minimum_train_months: int = 12,
    frequency: str = "QS",
    config_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Materialize compact OOF feature panels, with selection fit only on train rows."""

    panel_root, ledger_root, output_dir = Path(panel_root), Path(ledger_root), Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"immutable output exists: {output_dir}")
    panel_manifest, ledger_manifest = _verify_manifest(panel_root), _verify_manifest(ledger_root)
    rows, features = _load_joined(panel_root, ledger_root)
    folds = chronological_period_folds(
        rows, first_evaluation=first_evaluation, last_evaluation=last_evaluation,
        minimum_train_months=minimum_train_months, frequency=frequency,
    )
    if not folds:
        raise ValueError("no eligible fold-local selection folds")
    defaults = {
        "max_correlation_rows": 4_000,
        "max_candidates_per_family_before_redundancy": 192,
        "family_caps": {
            "distribution_dynamics": 24, "volatility": 16, "liquidity_proxy": 16,
            "dependence_covariance": 16, "other": 8,
        },
        # The unsupervised panel deliberately keeps a broad, family-balanced
        # candidate set.  Each downstream task then receives a smaller,
        # independently label-ranked subset; using the same caps for both
        # stages would only reproduce the broad panel twice.
        "supervised_family_caps": {
            "distribution_dynamics": 12, "volatility": 8, "liquidity_proxy": 8,
            "dependence_covariance": 8, "other": 4,
        },
        "random_state": 20260730,
    }
    if config_kwargs:
        defaults.update(config_kwargs)
    regime, transition, regime_selection, transition_selection, audits = _panel_for_folds(
        rows, features=features, folds=folds, config_kwargs=defaults
    )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(dir=output_dir.parent, prefix=f".{output_dir.name}."))
    try:
        outputs = {
            "regime_oof_features.parquet": regime,
            "transition_oof_features.parquet": transition,
            "regime_fold_selection.parquet": regime_selection,
            "transition_fold_selection.parquet": transition_selection,
            "fold_audit.parquet": pd.DataFrame(audits),
        }
        hashes: dict[str, str] = {}
        for name, frame in outputs.items():
            path = temporary / name
            frame.to_parquet(path, index=False, compression="zstd")
            hashes[name] = sha256(path)
        report = {
            "schema": SCHEMA,
            "research_only": True,
            "promotion_evidence": False,
            "sources": {
                "multiview_panel": {"path": str(panel_root), "manifest_sha256": sha256(panel_root / "manifest.json"), "feature_count": len(features)},
                "ledger": {"path": str(ledger_root), "manifest_sha256": sha256(ledger_root / "manifest.json")},
                "source_manifest_schemas": {"multiview": panel_manifest.get("schema"), "ledger": ledger_manifest.get("schema")},
            },
            "fold_contract": {
                "type": "expanding chronological", "frequency": frequency,
                "first_evaluation": first_evaluation, "last_evaluation": last_evaluation,
                "minimum_train_months": int(minimum_train_months),
                "label_availability": "train max(max(source_utc+12h, target__available_utc)) < evaluation_start_utc",
                "selection": "coverage/variance/redundancy and supervised ranking fit only on each fold train rows",
            },
            "probability_namespace_contract": {
                "canonical_regime_model_output": "regime_state_p__*",
                "canonical_transition_model_output": "transition_state_p__*",
                "interaction_discovery_regime_input": "regime_prob__*",
                "interaction_discovery_transition_input": "transition_prob__*",
                "adapter": {"regime_state_p__": "regime_prob__", "transition_state_p__": "transition_prob__"},
                "invariant": "the adapter only renames a single namespace after OOF prediction; it never merges, relabels, or cross-uses regime and transition probabilities",
                "probabilities_materialized_here": False,
            },
            "counts": {
                "folds": len(audits), "input_rows": len(rows), "regime_oof_rows": len(regime),
                "transition_oof_rows": len(transition), "regime_oof_feature_union": len([c for c in regime if c.startswith("mv__")]),
                "transition_oof_feature_union": len([c for c in transition if c.startswith("mv__")]),
            },
            "outputs_sha256": hashes,
        }
        manifest = temporary / "manifest.json"
        manifest.write_text(json.dumps(_safe(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (temporary / "manifest.sha256").write_text(f"{sha256(manifest)}  manifest.json\n", encoding="utf-8")
        os.replace(temporary, output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-root", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--ledger-root", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--first-evaluation", default="2023-10-01")
    parser.add_argument("--last-evaluation", default="2026-07-01")
    parser.add_argument("--minimum-train-months", type=int, default=12)
    parser.add_argument("--frequency", default="QS")
    parser.add_argument("--max-correlation-rows", type=int, default=4_000)
    parser.add_argument("--max-candidates-per-family-before-redundancy", type=int, default=192)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    report = materialize_fold_local_multiview_selection(
        panel_root=args.panel_root, ledger_root=args.ledger_root, output_dir=args.output_dir,
        first_evaluation=args.first_evaluation, last_evaluation=args.last_evaluation,
        minimum_train_months=args.minimum_train_months, frequency=args.frequency,
        config_kwargs={
            "max_correlation_rows": args.max_correlation_rows,
            "max_candidates_per_family_before_redundancy": args.max_candidates_per_family_before_redundancy,
        },
    )
    print(json.dumps(_safe(report), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
