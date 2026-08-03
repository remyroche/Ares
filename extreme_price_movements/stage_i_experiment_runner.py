"""Bounded sequential runner for the four Stage-I selection cells.

This module deliberately does not materialise data or start a large run by
itself.  A caller supplies four in-memory jobs and a strict-OOF generator.  It
then enforces the only permitted sequence: two side-local base selections,
strict same-side base expected-net OOF scores, two shared residual selections,
strict residual OOF scores, and common-bps reconstruction.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from .stage_i_feature_selection import (
    STAGE_I_ACTIVE_CONTRACTS,
    STAGE_I_META_BASE_OOF_HANDOFF_FEATURES,
    StageIHeadContract,
    run_stage_i_head_selection,
)
from .stage_i_causal_admission import (
    Causal21dAdmissionSpec,
    apply_causal_21d_side_admission,
    pooled_global_admission_comparison,
)


@dataclass(frozen=True)
class StageIExperimentJob:
    contract: StageIHeadContract
    frame: pd.DataFrame
    candidate_ids: Sequence[Any]
    candidate_kwargs: Mapping[str, Any]
    # Base targets are supplied; meta targets are always reconstructed here.
    target: Any | None = None


StrictOOFGenerator = Callable[[StageIExperimentJob, Mapping[str, Any]], Mapping[str, Any]]
TrainCandidate = Callable[..., Mapping[str, Any] | None]


def _as_ids(ids: Sequence[Any], n: int, *, label: str) -> np.ndarray:
    out = np.asarray(ids, dtype=object).reshape(-1)
    if len(out) != n or len(pd.unique(out)) != n:
        raise ValueError(f"{label} candidate_ids must be unique and row-aligned")
    return out


def _exact_net(job: StageIExperimentJob) -> np.ndarray:
    value = job.candidate_kwargs.get("exact_net_bps")
    units = str(job.candidate_kwargs.get("exact_net_units", "")).lower()
    arr = np.asarray(value, dtype=np.float32).reshape(-1) if value is not None else np.array([])
    if units != "bps" or len(arr) != len(job.frame) or not np.isfinite(arr).all():
        raise ValueError(f"{job.contract.artifact_key} requires aligned exact_net_bps")
    return arr


def _is_explicit_true(value: Any) -> bool:
    """Only accept literal bools or finite numeric 0/1 provenance flags."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(np.isfinite(value) and float(value) == 1.0)
    return False


def _base_handoff(
    payload: Mapping[str, Any], prediction: np.ndarray, job: StageIExperimentJob
) -> dict[str, np.ndarray]:
    """Validate the direct, same-side R3 output contract for meta selection."""
    raw = payload.get("base_oof_handoff")
    if not isinstance(raw, Mapping):
        raise ValueError(
            f"{job.contract.artifact_key} strict base OOF must provide base_oof_handoff"
        )
    handoff: dict[str, np.ndarray] = {}
    for feature in STAGE_I_META_BASE_OOF_HANDOFF_FEATURES:
        value = raw.get(feature)
        array = np.asarray(value, dtype=np.float32).reshape(-1)
        if len(array) != len(job.frame) or not np.isfinite(array).all():
            raise ValueError(
                f"{job.contract.artifact_key} base handoff {feature!r} is missing/misaligned"
            )
        handoff[feature] = array
    simplex = np.column_stack(
        [
            handoff["r3_p_adverse"],
            handoff["r3_p_weak"],
            handoff["r3_p_clear"],
        ]
    )
    if (simplex < 0.0).any() or not np.allclose(simplex.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError(f"{job.contract.artifact_key} base handoff is not an R3 probability simplex")
    if not np.allclose(
        handoff["r3_opportunity_score"],
        handoff["r3_p_clear"] - handoff["r3_p_adverse"],
        rtol=0.0,
        atol=1e-6,
    ):
        raise ValueError(
            f"{job.contract.artifact_key} r3_opportunity_score must be direct P(clear)-P(adverse)"
        )
    if not np.array_equal(handoff["prequential_base_expected_net_bps"], prediction):
        raise ValueError(
            f"{job.contract.artifact_key} prediction must equal its direct prequential base expected-net handoff"
        )
    return handoff


def _strict_oof(
    generator: StrictOOFGenerator,
    job: StageIExperimentJob,
    result: Mapping[str, Any],
) -> tuple[np.ndarray, Mapping[str, Any], Mapping[str, np.ndarray] | None]:
    payload = generator(job, result)
    prediction = np.asarray(payload.get("prediction"), dtype=np.float32).reshape(-1)
    provenance = payload.get("provenance")
    if len(prediction) != len(job.frame) or not np.isfinite(prediction).all():
        raise ValueError(f"{job.contract.artifact_key} strict OOF prediction is missing/misaligned")
    if not isinstance(provenance, Mapping):
        raise ValueError(f"{job.contract.artifact_key} strict OOF provenance is required")
    if not _is_explicit_true(provenance.get("strict_oof", provenance.get("is_oof", False))):
        raise ValueError(f"{job.contract.artifact_key} prediction is not strict OOF")
    if str(provenance.get("side", "")).lower() != job.contract.side:
        raise ValueError(f"{job.contract.artifact_key} strict OOF side provenance mismatch")
    if str(provenance.get("units", "bps")).lower() != "bps":
        raise ValueError(f"{job.contract.artifact_key} strict OOF score must be in bps")
    expected_semantics = (
        "prequential_base_expected_net_bps"
        if job.contract.layer == "base"
        else "raw_predicted_residual_bps"
    )
    if str(provenance.get("score_semantics", "")) != expected_semantics:
        raise ValueError(
            f"{job.contract.artifact_key} strict OOF must declare "
            f"score_semantics={expected_semantics!r}; mapped/calibrated scores "
            "cannot be handed to the residual learner"
        )
    folds = provenance.get("folds")
    if not isinstance(folds, Sequence) or isinstance(folds, (str, bytes)) or not folds:
        raise ValueError(f"{job.contract.artifact_key} strict OOF fold lineage is required")
    for index, fold in enumerate(folds):
        if not isinstance(fold, Mapping):
            raise ValueError(f"{job.contract.artifact_key} fold {index} lineage is invalid")
        train_available = pd.Timestamp(fold.get("train_max_label_available_ts"))
        validation_start = pd.Timestamp(fold.get("validation_start_ts"))
        if pd.isna(train_available) or pd.isna(validation_start):
            raise ValueError(f"{job.contract.artifact_key} fold {index} lacks availability boundaries")
        if train_available.tzinfo is None:
            train_available = train_available.tz_localize("UTC")
        else:
            train_available = train_available.tz_convert("UTC")
        if validation_start.tzinfo is None:
            validation_start = validation_start.tz_localize("UTC")
        else:
            validation_start = validation_start.tz_convert("UTC")
        if not train_available < validation_start:
            raise ValueError(
                f"{job.contract.artifact_key} fold {index} is not prior-resolved: "
                f"{train_available} >= {validation_start}"
            )
    handoff = _base_handoff(payload, prediction, job) if job.contract.layer == "base" else None
    return prediction, dict(provenance), handoff


def _layer_ledger(
    *,
    candidate_ids: Sequence[Any],
    timestamps: Sequence[Any],
    score: np.ndarray,
    exact_net_bps: np.ndarray,
    side: str,
    layer: str,
    label_available_timestamps: Sequence[Any],
) -> pd.DataFrame:
    """Build the narrow score/economics ledger used by pooled-global metrics."""
    ts = pd.to_datetime(pd.Series(timestamps), utc=True, errors="coerce")
    available = pd.to_datetime(
        pd.Series(label_available_timestamps), utc=True, errors="coerce"
    )
    ids = np.asarray(candidate_ids, dtype=object).reshape(-1)
    if (
        ts.isna().any()
        or available.isna().any()
        or len(ts) != len(score)
        or len(available) != len(score)
        or len(ids) != len(score)
    ):
        raise ValueError("Stage-I layer ledger requires aligned finite row inputs")
    return pd.DataFrame(
        {
            "candidate_id": ids,
            "candidate_key": [f"{side}::{value}" for value in ids],
            "side_name": str(side),
            "layer": str(layer),
            "signal_close_ts": ts,
            "decision_ts": ts + pd.Timedelta(hours=1),
            "label_available_ts": available,
            "score_bps": np.asarray(score, dtype=np.float32),
            "net_bps": np.asarray(exact_net_bps, dtype=np.float32),
        }
    )


def _pooled_global_layer_metrics(
    ledger: pd.DataFrame,
    *,
    top_fractions: Sequence[float] = (0.01, 0.05, 0.10, 0.20),
) -> list[dict[str, Any]]:
    """Select once over the pooled OOS population, then attribute by month/side.

    This deliberately does not rerank within a month, timestamp or side.  The
    grouped rows are contributions from the identical globally selected set.
    """
    if ledger.empty or ledger["candidate_key"].duplicated().any():
        raise ValueError("pooled-global Stage-I metrics require a non-empty unique ledger")
    work = ledger.copy()
    work["month"] = work["signal_close_ts"].dt.strftime("%Y-%m")
    order = work.sort_values(
        ["score_bps", "candidate_key"], ascending=[False, True], kind="stable"
    )
    rows: list[dict[str, Any]] = []
    for fraction in top_fractions:
        k = max(1, int(np.ceil(float(fraction) * len(order))))
        selected = order.head(k)
        common = {
            "layer": str(work["layer"].iloc[0]),
            "selection": "pooled_global_once_no_timestamp_or_side_rerank",
            "top_fraction": float(fraction),
            "candidate_rows": int(len(work)),
            "selected_global_rows": int(len(selected)),
        }
        rows.append(
            {
                **common,
                "scope": "pooled_global",
                "month": "__all__",
                "side": "__all__",
                "selected_rows": int(len(selected)),
                "net_bps_per_trade": float(selected["net_bps"].mean()),
                "gross_bps_per_trade": float(selected["net_bps"].mean() + 100.0),
            }
        )
        for (month, side), group in selected.groupby(["month", "side_name"], sort=True):
            rows.append(
                {
                    **common,
                    "scope": "selected_contribution",
                    "month": str(month),
                    "side": str(side),
                    "selected_rows": int(len(group)),
                    "net_bps_per_trade": float(group["net_bps"].mean()),
                    "gross_bps_per_trade": float(group["net_bps"].mean() + 100.0),
                }
            )
    return rows


def run_stage_i_sequential_funnel(
    jobs: Sequence[StageIExperimentJob],
    *,
    cfg: Mapping[str, Any],
    report_root: str | Path,
    train_candidate: TrainCandidate,
    strict_oof_generator: StrictOOFGenerator,
) -> dict[str, Any]:
    """Run exactly the approved Stage-I funnel, with no implicit HPO cells."""
    by_contract = {job.contract: job for job in jobs}
    expected = set(STAGE_I_ACTIVE_CONTRACTS)
    if set(by_contract) != expected or len(jobs) != len(expected):
        raise ValueError("Stage-I sequential funnel requires exactly the four active contracts")
    outputs: dict[str, Any] = {
        "schema": "stage_i_sequential_funnel_v1",
        "cells": {},
        "pooled_global_layer_metrics": [],
    }
    base_scores: dict[
        str, tuple[np.ndarray, np.ndarray, Mapping[str, Any], Mapping[str, np.ndarray]]
    ] = {}
    base_ledgers: list[pd.DataFrame] = []
    meta_ledgers: list[pd.DataFrame] = []

    for contract in (cell for cell in STAGE_I_ACTIVE_CONTRACTS if cell.layer == "base"):
        job = by_contract[contract]
        if job.target is None:
            raise ValueError(f"{contract.artifact_key} base target is required")
        ids = _as_ids(job.candidate_ids, len(job.frame), label=contract.artifact_key)
        result = run_stage_i_head_selection(
            job.frame, job.target, contract=contract, cfg=cfg, report_root=report_root,
            train_candidate=train_candidate, candidate_kwargs=job.candidate_kwargs,
        )
        if result is None:
            raise RuntimeError(f"{contract.artifact_key} selection returned no candidate")
        score, provenance, handoff = _strict_oof(strict_oof_generator, job, result)
        assert handoff is not None
        base_scores[contract.side] = (ids, score, provenance, handoff)
        outputs["cells"][contract.artifact_key] = {
            "selection": result,
            "strict_oof": provenance,
            "candidate_ids": ids.tolist(),
            "prequential_base_expected_net_bps": score,
            "base_oof_handoff_columns": list(STAGE_I_META_BASE_OOF_HANDOFF_FEATURES),
        }
        base_ledgers.append(
            _layer_ledger(
                candidate_ids=ids,
                timestamps=job.candidate_kwargs.get("timestamps", []),
                label_available_timestamps=job.candidate_kwargs.get(
                    "label_available_timestamps", []
                ),
                score=score,
                exact_net_bps=_exact_net(job),
                side=contract.side,
                layer="base",
            )
        )

    base_ledger = pd.concat(base_ledgers, ignore_index=True)
    outputs["pooled_global_layer_metrics"].extend(
        _pooled_global_layer_metrics(base_ledger)
    )

    for contract in (cell for cell in STAGE_I_ACTIVE_CONTRACTS if cell.layer == "meta"):
        job = by_contract[contract]
        ids = _as_ids(job.candidate_ids, len(job.frame), label=contract.artifact_key)
        base_ids, base_score, base_provenance, base_handoff = base_scores[contract.side]
        # Indexer join is O(n), avoids a temporary wide pandas reference-store
        # frame, and preserves the exact base OOF row values without mapping.
        take = pd.Index(base_ids).get_indexer(pd.Index(ids))
        if np.any(take < 0):
            raise ValueError(f"{contract.artifact_key} candidates do not align with same-side base OOF")
        offset = np.asarray(base_score, dtype=np.float32)[take]
        meta_frame = job.frame.copy()
        for feature in STAGE_I_META_BASE_OOF_HANDOFF_FEATURES:
            direct_value = np.asarray(base_handoff[feature], dtype=np.float32)[take]
            if feature in meta_frame.columns:
                supplied = pd.to_numeric(meta_frame[feature], errors="coerce").to_numpy(
                    dtype=np.float32
                )
                if not np.array_equal(supplied, direct_value):
                    raise ValueError(
                        f"{contract.artifact_key} supplied {feature!r} differs from the exact same-side base OOF handoff"
                    )
            else:
                meta_frame[feature] = direct_value
        exact_net = _exact_net(job)
        residual_target = exact_net - offset
        kwargs = dict(job.candidate_kwargs)
        kwargs.update(
            {
                "base_oof_provenance": dict(base_provenance),
                "frozen_base_expected_net_bps": offset,
                "frozen_base_expected_net_units": "bps",
            }
        )
        result = run_stage_i_head_selection(
            meta_frame, residual_target, contract=contract, cfg=cfg, report_root=report_root,
            train_candidate=train_candidate, candidate_kwargs=kwargs,
        )
        if result is None:
            raise RuntimeError(f"{contract.artifact_key} selection returned no candidate")
        # The strict OOF generator must receive the same immutable feature
        # contract that selection/HPO saw, including the direct same-side base
        # OOF values injected above.  Passing ``job`` here would silently let
        # a generator use the raw meta frame instead.
        meta_oof_job = replace(job, frame=meta_frame)
        residual_oof, provenance, _ = _strict_oof(
            strict_oof_generator, meta_oof_job, result
        )
        reconstructed = offset + residual_oof
        outputs["cells"][contract.artifact_key] = {
            "selection": result,
            "strict_oof": provenance,
            "target": "exact_net_bps_minus_frozen_causal_base_expected_net_bps",
            "reconstruction": "frozen_base_expected_net_bps_plus_predicted_residual_bps",
            "candidate_ids": ids.tolist(),
            "base_expected_net_bps": offset,
            "residual_oof_bps": residual_oof,
            "reconstructed_common_bps": reconstructed,
            "same_side_base_oof_handoff_features": list(
                STAGE_I_META_BASE_OOF_HANDOFF_FEATURES
            ),
            "frozen_meta_feature_contract": list(
                result.get("stage_i_selected_feature_contract", result.get("selected_feature_names", []))
            ),
        }
        meta_ledgers.append(
            _layer_ledger(
                candidate_ids=ids,
                timestamps=job.candidate_kwargs.get("timestamps", []),
                label_available_timestamps=job.candidate_kwargs.get(
                    "label_available_timestamps", []
                ),
                score=reconstructed,
                exact_net_bps=exact_net,
                side=contract.side,
                layer="meta_residual_reconstructed",
            )
        )
    meta_ledger = pd.concat(meta_ledgers, ignore_index=True)
    outputs["pooled_global_layer_metrics"].extend(
        _pooled_global_layer_metrics(meta_ledger)
    )
    admitted, admission_audit = apply_causal_21d_side_admission(
        meta_ledger,
        score_column="score_bps",
        net_column="net_bps",
        decision_column="decision_ts",
        label_available_column="label_available_ts",
        identity_column="candidate_key",
        spec=Causal21dAdmissionSpec(),
    )
    outputs["causal_21d_admission_metrics"] = pooled_global_admission_comparison(
        admitted,
        raw_score_column="score_bps",
        net_column="net_bps",
        identity_column="candidate_key",
        top_fractions=(0.01, 0.05, 0.10, 0.20),
    ).to_dict(orient="records")
    outputs["causal_21d_admission_audit"] = admission_audit.to_dict(orient="records")
    outputs["causal_21d_admission_candidates"] = admitted
    return outputs


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage-I bounded sequential-funnel preflight")
    parser.add_argument("--show-contract", action="store_true", help="print the four authorised cells; does not run training")
    args = parser.parse_args(argv)
    if args.show_contract:
        for contract in STAGE_I_ACTIVE_CONTRACTS:
            print(contract.artifact_key)
    else:
        parser.error("this bounded runner is library-driven; supply jobs and strict OOF generator from a run manifest")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
