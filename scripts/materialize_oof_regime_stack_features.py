#!/usr/bin/env python3
"""Materialize fold-local OOF regime and optional transition soft-state features.

The input panels must be point-in-time, outcome-free and candidate-keyed.  A
calendar-block fold uses only rows strictly before its block (less a purge), so
every emitted feature has a train-end timestamp strictly before its candidate
decision.  Transition morphology is an independent optional panel: absence is
recorded as unavailable and is never replaced by regime-state probabilities.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.regime_oof_stack import (  # noqa: E402
    IDENTITY_COLUMNS,
    STATE_OOD_COLUMN,
    STATE_PROBABILITY_PREFIX,
    TRANSITION_OOD_COLUMN,
    TRANSITION_PROBABILITY_PREFIX,
    RegimeOOFStackError,
    assert_outcome_free,
    derive_soft_state_fields,
    validate_candidate_identity,
    validate_regime_output_frame,
    validate_transition_output_frame,
)


SCHEMA = "oof_regime_stack_features_v1"
DEFAULT_CONFIG = ROOT / "configs/regime_oof_stack_2022_2026_v1.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "regime_oof_stack_contract_v1":
        raise RegimeOOFStackError("frozen regime OOF stack config is required")
    return payload


def _read_panel(path: Path, *, name: str) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    assert_outcome_free(frame)
    return validate_candidate_identity(frame)


def _exact_features(candidates: pd.DataFrame, panel: pd.DataFrame, *, name: str) -> pd.DataFrame:
    value_columns = [column for column in panel.columns if column not in IDENTITY_COLUMNS]
    collisions = sorted(set(value_columns).intersection(candidates.columns))
    if collisions:
        raise RegimeOOFStackError(f"{name} panel collides with candidate fields: {collisions[:12]}")
    joined = candidates.merge(
        panel.loc[:, [*IDENTITY_COLUMNS, *value_columns]],
        on=list(IDENTITY_COLUMNS),
        how="left",
        validate="one_to_one",
        sort=False,
    )
    if len(joined) != len(candidates) or joined.duplicated(list(IDENTITY_COLUMNS)).any():
        raise RegimeOOFStackError(f"{name} exact feature join changed candidate cardinality")
    if value_columns and joined[value_columns].isna().all(axis=1).any():
        raise RegimeOOFStackError(f"{name} panel lacks exact candidate coverage")
    return joined


def _feature_columns(frame: pd.DataFrame, *, max_features: int, train_mask: np.ndarray) -> list[str]:
    excluded = set(IDENTITY_COLUMNS)
    numeric = [
        column
        for column in frame.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(frame[column])
    ]
    if not numeric:
        raise RegimeOOFStackError("multiview panel has no numeric pre-entry features")
    train = frame.loc[train_mask, numeric].apply(pd.to_numeric, errors="coerce")
    coverage = train.notna().mean()
    variance = train.var(skipna=True)
    selected = [
        column
        for column in numeric
        if coverage.get(column, 0.0) >= 0.80 and np.isfinite(variance.get(column, np.nan)) and variance.get(column, 0.0) > 1e-12
    ]
    selected = sorted(selected, key=lambda column: (-float(variance[column]), str(column)))[: int(max_features)]
    if len(selected) < 2:
        raise RegimeOOFStackError("fewer than two train-supported multiview features remain")
    return selected


def _fit_gmm_features(
    frame: pd.DataFrame,
    *,
    train_mask: np.ndarray,
    evaluation_mask: np.ndarray,
    prefix: str,
    ood_column: str,
    n_components: int,
    max_features: int,
    pca_components: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    features = _feature_columns(frame, max_features=max_features, train_mask=train_mask)
    train = frame.loc[train_mask, features].apply(pd.to_numeric, errors="coerce")
    evaluate = frame.loc[evaluation_mask, features].apply(pd.to_numeric, errors="coerce")
    if len(train) < max(int(n_components) * 8, 32):
        raise RegimeOOFStackError("insufficient train rows for requested soft-state GMM")
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    train_x = scaler.fit_transform(imputer.fit_transform(train))
    eval_x = scaler.transform(imputer.transform(evaluate))
    width = min(int(pca_components), train_x.shape[1], max(1, len(train_x) - 1))
    pca = PCA(n_components=width, random_state=int(seed))
    train_z = pca.fit_transform(train_x)
    eval_z = pca.transform(eval_x)
    model = GaussianMixture(
        n_components=int(n_components), covariance_type="diag", random_state=int(seed), reg_covar=1e-5, max_iter=200
    ).fit(train_z)
    # Component labels are aligned *within the outer fold* by the first train
    # PCA coordinate.  The materializer records this explicitly; a later
    # cross-fold semantic alignment step must not overwrite posterior columns.
    order = np.argsort(model.means_[:, 0], kind="stable")
    probability = model.predict_proba(eval_z)[:, order].astype(np.float32)
    output = pd.DataFrame(
        {f"{prefix}{index}": probability[:, index] for index in range(probability.shape[1])},
        index=frame.index[evaluation_mask],
    )
    output[ood_column] = np.maximum(0.0, -model.score_samples(eval_z)).astype(np.float32)
    return output, {
        "selected_features": features,
        "feature_count": int(len(features)),
        "pca_components": int(width),
        "gmm_components": int(n_components),
        "component_alignment": "train_only_pca0_center_order",
        "train_rows": int(train_mask.sum()),
        "evaluation_rows": int(evaluation_mask.sum()),
    }


def _calendar_blocks(timestamp: pd.Series, *, frequency: str, evaluation_start: pd.Timestamp) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    values = pd.to_datetime(timestamp, utc=True, errors="raise")
    if frequency == "month":
        periods = values.dt.to_period("M")
        starts = sorted(pd.Timestamp(period.start_time, tz="UTC") for period in periods.unique())
        return [
            (start, start + pd.offsets.MonthBegin(1))
            for start in starts
            if start + pd.offsets.MonthBegin(1) > evaluation_start
        ]
    if frequency == "week":
        naive = values.dt.tz_convert("UTC").dt.tz_localize(None)
        starts = sorted(pd.Timestamp(value, tz="UTC") for value in naive.dt.to_period("W-SUN").dt.start_time.unique())
        return [
            (start, start + pd.Timedelta(days=7))
            for start in starts
            if start + pd.Timedelta(days=7) > evaluation_start
        ]
    raise RegimeOOFStackError("frequency must be 'week' or 'month'")


def materialize(
    *,
    candidates_path: Path,
    regime_panel_path: Path,
    output_dir: Path,
    evaluation_start: str,
    transition_panel_path: Path | None = None,
    config_path: Path = DEFAULT_CONFIG,
    frequency: str = "month",
    purge_hours: int = 12,
    n_components: int = 5,
    max_features: int = 64,
    pca_components: int = 12,
    seed: int = 52,
    require_transition: bool = False,
) -> Path:
    """Write an exact-coverage, candidate-keyed OOF soft-state ledger."""

    destination = Path(output_dir)
    if destination.exists():
        raise RegimeOOFStackError(f"refusing to overwrite existing output: {destination}")
    config = _read_json(Path(config_path))
    candidates = _read_panel(Path(candidates_path), name="candidates")
    regime_panel = _read_panel(Path(regime_panel_path), name="regime")
    regime = _exact_features(candidates, regime_panel, name="regime")
    transition_available = transition_panel_path is not None
    if require_transition and not transition_available:
        raise RegimeOOFStackError("transition panel is required; regime features cannot substitute for it")
    transition = None
    if transition_available:
        transition_panel = _read_panel(Path(transition_panel_path), name="transition")
        transition = _exact_features(candidates, transition_panel, name="transition")

    start = pd.to_datetime(evaluation_start, utc=True, errors="raise")
    eligible = candidates["__ts__"].ge(start).to_numpy()
    if not eligible.any():
        raise RegimeOOFStackError("evaluation_start leaves no eligible candidate rows")
    blocks = _calendar_blocks(candidates.loc[eligible, "__ts__"], frequency=frequency, evaluation_start=start)
    outputs: list[pd.DataFrame] = []
    folds: list[dict[str, Any]] = []
    purge = pd.Timedelta(hours=int(purge_hours))
    for fold_number, (block_start, block_end) in enumerate(blocks, start=1):
        evaluation = (
            eligible
            & candidates["__ts__"].ge(block_start).to_numpy()
            & candidates["__ts__"].lt(block_end).to_numpy()
        )
        train = candidates["__ts__"].lt(block_start - purge).to_numpy()
        if not evaluation.any():
            continue
        if int(train.sum()) < max(int(n_components) * 8, 32):
            raise RegimeOOFStackError(
                f"fold {block_start.isoformat()} has insufficient pre-block train support; move evaluation_start later"
            )
        base = candidates.loc[evaluation, list(IDENTITY_COLUMNS)].copy()
        train_end = candidates.loc[train, "__ts__"].max()
        base["regime_fold_id"] = f"{frequency}_{fold_number:03d}_{block_start.strftime('%Y%m%d')}"
        base["regime_train_end_utc"] = train_end
        base["regime_available_utc"] = base["__ts__"]
        regime_features, regime_audit = _fit_gmm_features(
            regime,
            train_mask=train,
            evaluation_mask=evaluation,
            prefix=STATE_PROBABILITY_PREFIX,
            ood_column=STATE_OOD_COLUMN,
            n_components=n_components,
            max_features=max_features,
            pca_components=pca_components,
            seed=seed + fold_number,
        )
        base = base.join(regime_features)
        base = derive_soft_state_fields(base)
        transition_audit: dict[str, Any] = {"status": "UNAVAILABLE_FAIL_CLOSED"}
        if transition is not None:
            transition_features, transition_audit = _fit_gmm_features(
                transition,
                train_mask=train,
                evaluation_mask=evaluation,
                prefix=TRANSITION_PROBABILITY_PREFIX,
                ood_column=TRANSITION_OOD_COLUMN,
                n_components=n_components,
                max_features=max_features,
                pca_components=pca_components,
                seed=seed + 10_000 + fold_number,
            )
            base["transition_fold_id"] = f"{frequency}_{fold_number:03d}_{block_start.strftime('%Y%m%d')}"
            base["transition_train_end_utc"] = train_end
            base["transition_available_utc"] = base["__ts__"]
            base = base.join(transition_features)
            base = derive_soft_state_fields(base, probability_prefix=TRANSITION_PROBABILITY_PREFIX)
            transition_audit["status"] = "MATERIALIZED"
        outputs.append(base)
        folds.append(
            {
                "fold_id": base["regime_fold_id"].iloc[0],
                "evaluation_start_utc": block_start.isoformat(),
                "evaluation_end_exclusive_utc": block_end.isoformat(),
                "purge_hours": int(purge_hours),
                "regime": regime_audit,
                "transition": transition_audit,
            }
        )
    if not outputs:
        raise RegimeOOFStackError("no OOF regime outputs were materialized")
    ledger = pd.concat(outputs, ignore_index=True).sort_values(["__ts__", "candidate_id"], kind="stable")
    expected = candidates.loc[eligible, list(IDENTITY_COLUMNS)].sort_values(["__ts__", "candidate_id"], kind="stable")
    actual = ledger.loc[:, list(IDENTITY_COLUMNS)].sort_values(["__ts__", "candidate_id"], kind="stable")
    if len(actual) != len(expected) or not actual.reset_index(drop=True).equals(expected.reset_index(drop=True)):
        raise RegimeOOFStackError("OOF ledger does not exactly cover every eligible candidate identity")
    validate_regime_output_frame(ledger)
    if transition is not None:
        validate_transition_output_frame(ledger)

    destination.mkdir(parents=True)
    ledger_path = destination / "oof_regime_stack_features.parquet"
    ledger.to_parquet(ledger_path, index=False, compression="zstd")
    manifest = {
        "schema": SCHEMA,
        "status": "MATERIALIZED_OOF_REGIME" if transition is None else "MATERIALIZED_OOF_REGIME_AND_TRANSITION",
        "config": {"path": str(Path(config_path).resolve()), "sha256": _sha256(Path(config_path))},
        "inputs": {
            "candidates": {"path": str(Path(candidates_path).resolve()), "sha256": _sha256(Path(candidates_path))},
            "regime_panel": {"path": str(Path(regime_panel_path).resolve()), "sha256": _sha256(Path(regime_panel_path))},
            "transition_panel": ({"path": str(Path(transition_panel_path).resolve()), "sha256": _sha256(Path(transition_panel_path))} if transition_panel_path else None),
        },
        "validation_contract": {
            "folding": f"expanding pre-block {frequency} folds",
            "purge_hours": int(purge_hours),
            "train_end": "strictly before every candidate decision",
            "availability": "at or before every candidate decision",
            "feature_selection": "train-only numeric coverage/variance screen",
            "dimensionality_reduction": "train-only median imputer, scaler, PCA",
            "state_model": "train-only diagonal GMM with train-only PCA0 center-order component alignment",
            "transition_fail_closed": "no transition fields emitted when transition panel is unavailable",
        },
        "coverage": {"eligible_rows": int(len(expected)), "ledger_rows": int(len(ledger)), "exact_identity_coverage": True},
        "transition": {"available": bool(transition is not None), "status": "MATERIALIZED" if transition is not None else "UNAVAILABLE_FAIL_CLOSED"},
        "folds": folds,
        "outputs": {"oof_regime_stack_features.parquet": {"sha256": _sha256(ledger_path), "rows": int(len(ledger))}},
        "config_method_families": {"regime": config["regime_state_layer"]["method_families"], "transition": config["transition_state_layer"]["method_families"]},
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--regime-panel", type=Path, required=True)
    parser.add_argument("--transition-panel", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--evaluation-start", required=True, help="UTC timestamp; earlier candidate rows are pre-block training only")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--frequency", choices=("week", "month"), default="month")
    parser.add_argument("--purge-hours", type=int, default=12)
    parser.add_argument("--n-components", type=int, default=5)
    parser.add_argument("--max-features", type=int, default=64)
    parser.add_argument("--pca-components", type=int, default=12)
    parser.add_argument("--seed", type=int, default=52)
    parser.add_argument("--require-transition", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output = materialize(
        candidates_path=args.candidates,
        regime_panel_path=args.regime_panel,
        transition_panel_path=args.transition_panel,
        output_dir=args.output_dir,
        evaluation_start=args.evaluation_start,
        config_path=args.config,
        frequency=args.frequency,
        purge_hours=args.purge_hours,
        n_components=args.n_components,
        max_features=args.max_features,
        pca_components=args.pca_components,
        seed=args.seed,
        require_transition=args.require_transition,
    )
    print(json.dumps({"output_dir": str(output), "status": "ok"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
