#!/usr/bin/env python3
"""Install a side x archetype EV map and causal EV-unit admission policy."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from extreme_price_movements.supervised_market_state_calibration import (
    fit_hierarchical_ev_calibrator,
    hierarchical_ev_calibrator_payload,
)


POLICY_ID = "side_archetype_hier_ev_fixed70_trim10_21d_v1"
POLICY_NAME = "s52_v9_tail95_mlp_hierev_sidearch_ev70_trim10_21d_v1"
FAMILY = "side_archetype_expected_ev_recent_correction"
DEFAULT_FIXED_TARGET_NET_EV = 0.007
DEFAULT_ROBUST_DAILY_TRIM_FRACTION = 0.10
DEFAULT_WINDOW_DAYS = 21
DEFAULT_CANONICAL_PARENT = Path(
    "data_perp/artifacts/"
    "s59_s52_finalfit_meta_v9_exact55_tail95_mlp_hierev_evtarget28d_20260713_v1"
)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalise_archetypes(rows: pd.DataFrame) -> pd.DataFrame:
    out = rows.copy()
    out["side_name"] = out["side_name"].astype(str).str.lower()
    arch_col = next(
        (
            name
            for name in (
                "archetype_policy_key",
                "policy_archetype",
                "local_side_archetype",
                "archetype_label_family",
            )
            if name in out.columns
        ),
        None,
    )
    if arch_col is None:
        raise ValueError("OOF rows have no policy archetype column")
    values = out[arch_col].fillna("missing").astype(str)
    for side in ("long", "short"):
        prefix = f"{side}__"
        mask = out["side_name"].eq(side) & values.str.startswith(prefix, na=False)
        values.loc[mask] = values.loc[mask].str[len(prefix) :]
    out["archetype_policy_key"] = values
    return out


def _load_oos(path: Path) -> pd.DataFrame:
    rows = pd.read_parquet(path)
    rows = _normalise_archetypes(rows)
    timestamp_col = "__ts__" if "__ts__" in rows.columns else "timestamp"
    symbol_col = "__symbol__" if "__symbol__" in rows.columns else "symbol"
    rows["timestamp"] = pd.to_datetime(rows[timestamp_col], utc=True, errors="coerce")
    rows["symbol"] = rows[symbol_col].astype(str)
    required = [
        "timestamp",
        "symbol",
        "side_name",
        "archetype_policy_key",
        "rank_mlp_direct",
        "ev_after_1pct",
    ]
    missing = [name for name in required if name not in rows.columns]
    if missing:
        raise ValueError(f"OOF source is missing columns: {missing}")
    rows = rows.dropna(subset=required).sort_values("timestamp", kind="stable")
    if rows.empty:
        raise ValueError("OOF source has no usable rows")
    return rows


def _patch_policy_pointer(payload: dict[str, Any], relative_path: str) -> None:
    payload.update(
        {
            "policy_name": POLICY_NAME,
            "status": "promoted_default_threshold_basis_ev70_trim10_21d",
            "archetype_dynamic_layer": (
                "side_archetype_hierarchical_ev + "
                "causal_21d_trim10_recent_ev_correction + "
                "fixed_0.70pct_net_ev_admission"
            ),
            "threshold_basis_policy_enabled": True,
            "threshold_basis_policy_id": POLICY_ID,
            "threshold_basis_policy_path": relative_path,
            "threshold_basis_family": FAMILY,
            "threshold_basis_window_days": DEFAULT_WINDOW_DAYS,
            "threshold_basis_selection_mode": "fixed_corrected_ev_threshold",
            "threshold_basis_fixed_target_net_ev": DEFAULT_FIXED_TARGET_NET_EV,
            "threshold_basis_robust_daily_residual_trim_fraction": (
                DEFAULT_ROBUST_DAILY_TRIM_FRACTION
            ),
            "source_threshold_basis_policy": relative_path,
        }
    )
    selection = payload.get("selection")
    if isinstance(selection, dict):
        _patch_policy_pointer(selection, relative_path)


def _patch_canonical_paths(payload: dict[str, Any], bundle: Path) -> None:
    policy_dir = bundle / "policy_params"
    payload.update(
        {
            "regime_ev_predecessor_bundle_path": str(
                policy_dir / "v9_tail95_predecessor_bundle.joblib"
            ),
            "regime_ev_residual_event_state_path": str(
                policy_dir / "residual_event_state.joblib"
            ),
            "regime_ev_calibration_artifact_path": str(
                policy_dir / "composite_policy_regime_ev_calibration.json"
            ),
        }
    )
    selection = payload.get("selection")
    if isinstance(selection, dict):
        _patch_canonical_paths(selection, bundle)


def _refresh_meta_score_reference(
    policy_dir: Path, meta_training_source: Path | None
) -> int:
    priors_path = policy_dir / "meta_reliability_priors.json"
    if not priors_path.is_file():
        raise FileNotFoundError(priors_path)
    priors = _read_json(priors_path)
    existing = priors.get("score_reference_quantiles") or []
    if meta_training_source is None:
        if len(existing) < 257:
            raise ValueError(
                "Meta reliability priors need a frozen train-score reference; "
                "pass --meta-training-source"
            )
        return int(len(existing))
    source_path = meta_training_source
    if source_path.is_dir():
        source_path = source_path / "train_meta_regime_handoff.parquet"
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    columns = ["score", "selected_top30"]
    try:
        rows = pd.read_parquet(source_path, columns=columns)
    except (KeyError, ValueError):
        rows = pd.read_parquet(source_path, columns=["score"])
    if "selected_top30" in rows.columns:
        rows = rows.loc[rows["selected_top30"].fillna(False).astype(bool)]
    score = pd.to_numeric(rows["score"], errors="coerce").dropna().to_numpy(
        dtype=np.float64, copy=False
    )
    if score.size < 1_000:
        raise ValueError(f"Insufficient meta train scores: {score.size}")
    priors["score_reference_quantiles"] = np.quantile(
        score, np.linspace(0.0, 1.0, 4097)
    ).astype(float).tolist()
    priors["score_reference_source"] = str(source_path)
    priors["score_reference_rows"] = int(score.size)
    _write_json(priors_path, priors)
    return 4097


def promote(
    bundle: Path,
    oos_predictions: Path,
    *,
    outcome_horizon_hours: int,
    canonical_parent: Path,
    predecessor_bundle: Path | None,
    meta_training_source: Path | None,
) -> dict[str, Any]:
    policy_dir = bundle / "policy_params"
    score_reference_count = _refresh_meta_score_reference(
        policy_dir, meta_training_source
    )
    postprocessor_path = policy_dir / "composite_policy_regime_ev_calibration.json"
    canonical_policy_dir = canonical_parent / "policy_params"
    canonical_postprocessor = (
        canonical_policy_dir / "composite_policy_regime_ev_calibration.json"
    )
    if not canonical_postprocessor.is_file():
        raise FileNotFoundError(canonical_postprocessor)
    policy_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(canonical_postprocessor, postprocessor_path)
    canonical_models = canonical_policy_dir / "policy_models"
    if not canonical_models.is_dir():
        raise FileNotFoundError(canonical_models)
    shutil.copytree(
        canonical_models,
        policy_dir / "policy_models",
        dirs_exist_ok=True,
    )
    rows = _load_oos(oos_predictions)
    score = pd.to_numeric(rows["rank_mlp_direct"], errors="coerce").to_numpy(
        dtype=np.float64, copy=False
    )
    realized = pd.to_numeric(rows["ev_after_1pct"], errors="coerce").to_numpy(
        dtype=np.float64, copy=False
    )
    calibrator = fit_hierarchical_ev_calibrator(
        rows,
        score,
        realized,
        shrink_rows=2_000.0,
        min_local_rows=600,
        local_weight_cap=0.50,
        tail_weight_top10=4.0,
        rank_blend=1.0,
    )
    mapping = hierarchical_ev_calibrator_payload(calibrator)
    mapping.update(
        {
            "mapping_scope": "side_x_archetype_shrunk_to_global",
            "rank_reference_scope": "hierarchical_side_x_archetype_expected_ev",
            "production_fit_source": str(oos_predictions),
            "production_fit_rows": int(len(rows)),
        }
    )
    postprocessor = _read_json(postprocessor_path)
    postprocessor["predecessor_policy_id"] = str(
        postprocessor.get("predecessor_policy_id")
        or postprocessor.get("predecessor")
        or ""
    )
    postprocessor["expected_ev_mapping"] = mapping
    postprocessor["expected_ev_col"] = "expected_net_ev_after_1pct"
    postprocessor["expected_ev_rank_col"] = "expected_ev_rank_score"
    postprocessor["expected_ev_contract"] = (
        "Expected EV is first mapped by side x archetype and shrunk to the global "
        "curve. The admission policy may then add only a causal recent side x "
        "archetype realized-minus-mapped EV correction."
    )
    _write_json(postprocessor_path, postprocessor)

    predecessor_source = predecessor_bundle or (
        canonical_parent / "policy_params/v9_tail95_predecessor_bundle.joblib"
    )
    if not predecessor_source.is_file():
        raise FileNotFoundError(
            "Canonical V9 predecessor bundle is missing: "
            f"{predecessor_source}"
        )
    predecessor_path = policy_dir / "v9_tail95_predecessor_bundle.joblib"
    shutil.copy2(predecessor_source, predecessor_path)

    mapped_col = "expected_net_ev_after_1pct_mlp_direct"
    if mapped_col not in rows.columns:
        raise ValueError(
            "OOF source must contain fold-local side x archetype expected EV: "
            + mapped_col
        )
    reference = rows.loc[
        :,
        [
            "timestamp",
            "symbol",
            "side_name",
            "archetype_policy_key",
            "rank_mlp_direct",
            mapped_col,
            "ev_after_1pct",
        ],
    ].rename(
        columns={
            "archetype_policy_key": "policy_archetype",
            mapped_col: "mapped_expected_ev",
        }
    )
    reference["outcome_resolved_at"] = reference["timestamp"] + pd.Timedelta(
        hours=int(outcome_horizon_hours)
    )
    reference_path = policy_dir / "threshold_basis_reference_sidearch_ev21d.parquet"
    reference.to_parquet(reference_path, index=False, compression="zstd")
    policy = {
        "schema_version": "threshold_basis_policy_v3",
        "policy_id": POLICY_ID,
        "policy_name": POLICY_NAME,
        "enabled": True,
        "family": FAMILY,
        "window_days": DEFAULT_WINDOW_DAYS,
        "selection_mode": "fixed_corrected_ev_threshold",
        "fixed_target_net_ev": DEFAULT_FIXED_TARGET_NET_EV,
        "recalibration_frequency": "1d_at_00_utc",
        "robust_daily_residual_trim_fraction": DEFAULT_ROBUST_DAILY_TRIM_FRACTION,
        "robust_daily_residual_normalization": "median_iqr",
        "top_fraction": 0.10,
        "min_reference_rows": 40,
        "side_support_target": 320.0,
        "local_support_target": 160.0,
        "recent_ev_correction_cap": 0.03,
        "ev_rank_blend_weight": 1.0,
        "rank_blend_parent_col": "v9_tail95_predecessor_rank",
        "mapped_expected_ev_col": "expected_net_ev_after_1pct_side_archetype",
        "reference_mapped_expected_ev_col": "mapped_expected_ev",
        "live_score_col": "expected_ev_rank_score",
        "return_col": "ev_after_1pct",
        "reference_candidates_path": reference_path.name,
        "reference_columns": list(reference.columns),
        "reference_rows": int(len(reference)),
        "meta_score_reference_quantiles": int(score_reference_count),
        "reference_timestamp_min": reference["timestamp"].min().isoformat(),
        "reference_timestamp_max": reference["timestamp"].max().isoformat(),
        "outcome_horizon_hours": int(outcome_horizon_hours),
        "formula": (
            "corrected_expected_ev = side_archetype_mapped_expected_ev + "
            "causal_21d_robust_trimmed_side_archetype_recent_ev_residual; "
            "admit when corrected_expected_ev >= 0.007"
        ),
        "cost_contract": (
            "mapped_expected_ev and ev_after_1pct are net of the sole 1% round-trip "
            "cost; no additional fee is subtracted"
        ),
        "causal_contract": (
            "At t, reference rows require timestamp < t and outcome_resolved_at < t; "
            "recent residual outcome days use [day(t)-21d,day(t)) only; daily "
            "residual means receive symmetric 10% median/IQR trimming"
        ),
    }
    policy_path = policy_dir / "threshold_basis_policy_sidearch_ev70_trim10_21d.json"
    _write_json(policy_path, policy)
    # Inference and replay resolve this canonical filename by default. Keep the
    # versioned policy for provenance, but install the promoted policy here too.
    canonical_policy_path = policy_dir / "threshold_basis_policy.json"
    shutil.copy2(policy_path, canonical_policy_path)

    relative_policy = str(policy_path.relative_to(bundle.parent.parent.parent))
    for name in (
        "optimized_portfolio_policy_config.json",
        "promoted_policy_manifest.json",
    ):
        path = policy_dir / name
        if not path.exists():
            continue
        payload = _read_json(path)
        _patch_policy_pointer(payload, relative_policy)
        _patch_canonical_paths(payload, bundle)
        payload["side_archetype_expected_ev_policy"] = {
            "policy_id": POLICY_ID,
            "formula": policy["formula"],
            "mapped_expected_ev_is_side_archetype_specific": True,
        }
        _write_json(path, payload)

    pointer_path = bundle / "meta_postprocessor_pointer.json"
    pointer = _read_json(pointer_path) if pointer_path.exists() else {}
    pointer.update(
        {
            "schema": "meta_postprocessor_pointer_v1",
            "policy_id": "meta_residual_v9_tail95_market_state_mlp_hier_ev_v1",
            "artifact_path": str(postprocessor_path),
            "predecessor_bundle_path": str(predecessor_path),
            "residual_event_state_path": str(
                policy_dir / "residual_event_state.joblib"
            ),
            "rolling_8d_modulator_enabled": False,
        }
    )
    _write_json(pointer_path, pointer)

    manifest = {
        "schema": "side_archetype_expected_ev_policy_promotion_v1",
        "bundle": str(bundle),
        "policy_id": POLICY_ID,
        "policy_name": POLICY_NAME,
        "oos_source": str(oos_predictions),
        "oos_rows": int(len(rows)),
        "side_archetype_curves": int(len(mapping.get("local") or {})),
        "reference_rows": int(len(reference)),
        "postprocessor_sha256": _sha256(postprocessor_path),
        "predecessor_sha256": _sha256(predecessor_path),
        "policy_sha256": _sha256(policy_path),
        "canonical_policy_sha256": _sha256(canonical_policy_path),
        "status": "candidate_pending_frozen_replay",
    }
    _write_json(policy_dir / "side_archetype_expected_ev_policy_manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--oos-predictions", type=Path, required=True)
    parser.add_argument("--outcome-horizon-hours", type=int, default=12)
    parser.add_argument(
        "--canonical-parent", type=Path, default=DEFAULT_CANONICAL_PARENT
    )
    parser.add_argument(
        "--predecessor-bundle",
        type=Path,
        help=(
            "V9 predecessor bundle fitted to the deployed meta score domain. "
            "Defaults to the canonical parent bundle."
        ),
    )
    parser.add_argument("--meta-training-source", type=Path)
    args = parser.parse_args()
    manifest = promote(
        args.bundle.resolve(),
        args.oos_predictions.resolve(),
        outcome_horizon_hours=int(args.outcome_horizon_hours),
        canonical_parent=args.canonical_parent.resolve(),
        predecessor_bundle=(
            args.predecessor_bundle.resolve()
            if args.predecessor_bundle is not None
            else None
        ),
        meta_training_source=(
            args.meta_training_source.resolve()
            if args.meta_training_source is not None
            else None
        ),
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
