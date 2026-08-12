"""Production OOS writer for a frozen explicit-target Stage-I winner."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from .stage_i_adapter_strict_oof import (
    StageIAdapterStrictOOFPlan,
    generate_stage_i_adapter_strict_oof,
)
from .stage_i_adapter_winner_bundle import StageIAdapterWinnerBundle
from .stage_i_causal_admission import (
    Causal21dAdmissionSpec,
    apply_causal_21d_side_admission,
    pooled_global_admission_comparison,
)
from .stage_i_ranking import stable_stage_i_rank_frame
from .stage_i_target_adapter import bind_target_contract, canonical_sha256, file_sha256


SCHEMA = "stage_i_production_target_adapter_oos_v2"


@dataclass(frozen=True)
class StageIAdapterProductionInput:
    side: str
    frame: pd.DataFrame
    contract_frame: pd.DataFrame
    candidate_ids: Sequence[Any]
    decision_timestamps: Sequence[Any]
    label_available_timestamps: Sequence[Any]
    base_target: Sequence[float]
    exact_gross_bps: Sequence[float]
    exact_net_bps: Sequence[float]
    target_valid: Sequence[bool]
    sample_weight: Sequence[float]
    panel_manifest: Mapping[str, Any]
    panel_manifest_sha256: str
    base_target_column: str
    meta_basis_column: str
    candidate_fraction: float = 0.30
    n_validation_folds: int = 4
    min_train_rows: int = 500


def _tail_metrics(frame: pd.DataFrame, score_column: str, layer: str) -> pd.DataFrame:
    work = frame.loc[frame[score_column].notna() & frame.target_valid.astype(bool)].copy()
    if work.empty:
        return pd.DataFrame()
    work["month"] = pd.to_datetime(work.decision_ts, utc=True).dt.strftime("%Y-%m")
    ordered = stable_stage_i_rank_frame(work, score_column=score_column)
    rows = []
    for fraction in (0.01, 0.05, 0.10, 0.20):
        selected = ordered.head(max(1, int(math.ceil(fraction * len(ordered)))))
        rows.append({
            "layer": layer, "scope": "pooled_global", "top_fraction": fraction,
            "selected_rows": len(selected), "gross_bps_per_trade": float(selected.exact_gross_bps.mean()),
            "net_bps_per_trade": float(selected.exact_net_bps.mean()),
            "long_rows": int(selected.side_name.eq("long").sum()),
            "short_rows": int(selected.side_name.eq("short").sum()),
            "worst_month_net_bps_per_trade": float(selected.groupby("month").exact_net_bps.mean().min()),
        })
        for (side, month), local in selected.groupby(["side_name", "month"], observed=True):
            rows.append({
                "layer": layer, "scope": "side_month", "side": side, "month": month,
                "top_fraction": fraction, "selected_rows": len(local),
                "gross_bps_per_trade": float(local.exact_gross_bps.mean()),
                "net_bps_per_trade": float(local.exact_net_bps.mean()),
            })
    return pd.DataFrame(rows)


def run_stage_i_adapter_production_oos(
    *, bundle: StageIAdapterWinnerBundle,
    inputs: Sequence[StageIAdapterProductionInput], output_dir: str | Path,
    fit_model: Callable[..., Any],
    admission_spec: Causal21dAdmissionSpec = Causal21dAdmissionSpec(),
) -> dict[str, Any]:
    if {item.side for item in inputs} != {"long", "short"} or len(inputs) != 2:
        raise ValueError("production target-adapter OOS requires long and short")
    results = []
    source_manifest = []
    for source in inputs:
        cell = bundle.cell(source.side)
        if canonical_sha256(dict(source.panel_manifest)) != source.panel_manifest_sha256:
            raise ValueError(f"{source.side}: production panel manifest hash drift")
        contract = source.contract_frame.copy()
        runtime_base = bind_target_contract(
            contract, family=cell.base_target_contract.family, layer="base",
            target_name=cell.base_target_contract.target_name,
            geometry=cell.base_target_contract.geometry,
            target_columns=(source.base_target_column,),
            economics_columns=("gross_bps", "net_bps"),
            validity_column="target_valid", weight_column="sample_weight",
            metadata=dict(cell.base_target_contract.metadata),
        )
        runtime_meta = bind_target_contract(
            contract, family=cell.meta_target_contract.family, layer="meta",
            target_name=cell.meta_target_contract.target_name,
            geometry=cell.meta_target_contract.geometry,
            target_columns=(source.meta_basis_column,),
            economics_columns=("gross_bps", "net_bps"),
            validity_column="target_valid", weight_column="meta_sample_weight",
            metadata=dict(cell.meta_target_contract.metadata),
        )
        result = generate_stage_i_adapter_strict_oof(
            StageIAdapterStrictOOFPlan(
                side=source.side, frame=source.frame, contract_frame=contract,
                candidate_ids=source.candidate_ids,
                decision_timestamps=source.decision_timestamps,
                label_available_timestamps=source.label_available_timestamps,
                base_target=source.base_target, exact_gross_bps=source.exact_gross_bps,
                exact_net_bps=source.exact_net_bps, target_valid=source.target_valid,
                sample_weight=source.sample_weight,
                base_target_contract=cell.base_target_contract,
                meta_target_contract=cell.meta_target_contract,
                runtime_base_target_contract=runtime_base,
                runtime_meta_target_contract=runtime_meta,
                base_feature_names=cell.base_features, meta_feature_names=cell.meta_features,
                base_params=cell.base_params, meta_params=cell.meta_params,
                candidate_selected=None, candidate_fraction=source.candidate_fraction,
                meta_sample_weight=contract.meta_sample_weight.to_numpy(np.float32),
                n_validation_folds=source.n_validation_folds,
                min_train_rows=source.min_train_rows,
            ),
            fit_model=fit_model,
        )
        results.append(result)
        source_manifest.append({
            "side": source.side, "panel_manifest": dict(source.panel_manifest),
            "panel_manifest_sha256": source.panel_manifest_sha256,
            "runtime_base_target_contract_sha256": runtime_base.sha256,
            "runtime_meta_target_contract_sha256": runtime_meta.sha256,
        })
    all_predictions = pd.concat([result.predictions for result in results], ignore_index=True)
    all_predictions["candidate_key"] = all_predictions.side_name + "::" + all_predictions.candidate_id.astype(str)
    base_metrics = _tail_metrics(all_predictions, "prequential_base_expected_net_bps", "base")
    meta_metrics = _tail_metrics(all_predictions, "reconstructed_expected_net_bps", "meta")
    reference = all_predictions.loc[
        all_predictions.mapping_reference_eligible.astype(bool)
    ].rename(columns={"exact_net_bps": "net_bps"})
    admitted, admission_audit = apply_causal_21d_side_admission(
        reference, score_column="reconstructed_expected_net_bps", net_column="net_bps",
        decision_column="decision_ts", label_available_column="label_available_ts",
        identity_column="candidate_key", spec=admission_spec,
    )
    # Only the base candidate stream can become an action. Noncandidate rows
    # remain reference support and cannot be resurrected by calibration.
    admitted["action_eligible"] = admitted.candidate_selected.astype(bool)
    admitted["causal_21d_side_admitted_ge_50bps"] &= admitted.action_eligible
    admission_metrics = pooled_global_admission_comparison(
        admitted, raw_score_column="reconstructed_expected_net_bps",
        net_column="net_bps", gross_column="exact_gross_bps",
        identity_column="candidate_key", top_fractions=(0.01, 0.05, 0.10, 0.20),
    )
    root = Path(output_dir)
    if root.exists():
        raise FileExistsError(f"production target-adapter OOS output exists: {root}")
    root.mkdir(parents=True)
    all_predictions.to_parquet(root / "strict_oof_predictions.parquet", index=False, compression="zstd")
    pd.concat([result.fold_provenance for result in results], ignore_index=True).to_parquet(root / "fold_provenance.parquet", index=False, compression="zstd")
    pd.concat([base_metrics, meta_metrics], ignore_index=True).to_parquet(root / "layer_metrics.parquet", index=False, compression="zstd")
    admitted.to_parquet(root / "causal_21d_admission.parquet", index=False, compression="zstd")
    admission_audit.to_parquet(root / "causal_21d_admission_audit.parquet", index=False, compression="zstd")
    admission_metrics.to_parquet(root / "causal_21d_admission_metrics.parquet", index=False, compression="zstd")
    artifact_paths = [path for path in root.iterdir() if path.is_file()]
    manifest = {
        "schema": SCHEMA, "status": "complete", "winner_bundle_sha256": bundle.sha256,
        "winner_bundle": bundle.to_dict(), "source_manifests": source_manifest,
        "rows": len(all_predictions), "strict_meta_rows": int(all_predictions.strict_oof_available.sum()),
        "ranking": "pooled global after common-bps mapping; never per timestamp",
        "meta_training": "candidate-only; full valid rows reference-only for mapping",
        "admission": "side-local robust causal 21-day map >=50bps then pooled-global action ranking",
        "artifacts": {path.name: file_sha256(path) for path in artifact_paths},
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


__all__ = [
    "SCHEMA", "StageIAdapterProductionInput", "run_stage_i_adapter_production_oos",
]
