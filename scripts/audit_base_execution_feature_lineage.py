#!/usr/bin/env python3
"""Fail-closed feature and OOF-lineage contracts for base/execution research.

This module intentionally does not train, rank, calibrate, or replay a model.
It answers two narrower questions which must be settled before those steps:

* can a named input belong to the base or execution-EV layer; and
* is a supplied score a one-to-one, strictly OOF prediction available at the
  receiving layer's decision time?

The base layer may consume only causal/raw inputs.  The execution layer may
also consume explicitly declared upstream model outputs, but only after the
corresponding prediction table passes :func:`audit_oof_prediction_lineage`.
Timing, MAE, target-price and wait decisions remain action-layer inputs, not
execution-EV inputs.  The functions are deliberately generic so that they can
consume the existing raw-feature, side-local feature-contract, and lineage
artifacts without introducing a second feature registry.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "base_execution_feature_lineage_audit_v1"
IDENTITY = ("candidate_id",)
PREDICTION_AUDIT_COLUMNS = (
    "model_layer", "rows", "candidate_identity", "prediction_columns",
    "prediction_timestamp_column", "fit_end_timestamp_column", "oof_flag_column",
    "availability_rule", "fit_end_rule", "status",
)

# These are deliberately semantic rather than an exhaustive block list.  A
# name which describes a realised path or an action is never rescued by an OOF
# flag; it belongs in the target/action contract, not a base or execution-EV
# feature list.
REALIZED_PATH_TOKENS = (
    "future", "realized", "realised", "outcome", "label", "mfe", "mae",
    "giveback", "exit_reason", "exit_hour", "first_event", "timeout",
    "postcost", "retained", "adverse", "favorable", "favourable",
    "bars_before", "price_stops_decreasing", "slope_atr_per_hour",
)
DIRECT_ECONOMIC_TARGET_TOKENS = (
    "gross_h12", "net_h12", "execution_adjusted_gross", "row_cost",
    "known_row_cost", "realized_cost", "realised_cost", "fee_bps",
    "spread_bps", "slippage_bps", "total_cost", "delta_continue",
    "continue_better", "action_value", "net_continue", "net_exit",
)
ACTION_ONLY_TOKENS = (
    "entry_timing", "time_to_first", "target_price", "wait_action",
    "wait_price", "wait_for", "exit_now", "entry_delay", "mae",
)
IDENTITY_TOKENS = {
    "candidate_id", "decision_ts", "feature_cutoff_ts", "entry_ts", "label_end_ts",
    "label_available_ts", "symbol", "source_symbol", "side", "side_name",
    "__ts__", "timestamp", "execution_policy_id", "cost_model_id", "path_source_id",
}
MODEL_DERIVED_TOKENS = (
    "score_", "_score", "prediction", "predicted", "oof", "expected_ev",
    "mapped_ev", "model_output", "base_alpha", "residual_alpha", "leaf_",
    "posterior", "probability",
)


class FeatureEligibilityError(ValueError):
    """A named input is incompatible with the receiving model layer."""


class PredictionLineageError(ValueError):
    """A prediction table cannot be shown to be strict OOF at decision time."""


@dataclass(frozen=True)
class PredictionLineageAudit:
    """Validated row-level and aggregate OOF provenance for one score source."""

    summary: dict[str, Any]
    rows: pd.DataFrame


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (pd.Timestamp, Path)):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _normalise_utc(frame: pd.DataFrame, column: str, *, context: str) -> pd.Series:
    if column not in frame:
        raise PredictionLineageError(f"{context} is missing required timestamp column {column!r}")
    value = pd.to_datetime(frame[column], utc=True, errors="coerce")
    if value.isna().any():
        raise PredictionLineageError(f"{context}.{column} contains invalid or missing UTC timestamps")
    return value


def _strict_bool(values: pd.Series, *, context: str) -> pd.Series:
    """Accept bool/0/1 only; notably, string ``'false'`` is never truthy."""
    if pd.api.types.is_bool_dtype(values):
        return values.astype(bool)
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.isna().any() or not numeric.isin((0, 1)).all():
        raise PredictionLineageError(f"{context} must contain only boolean or 0/1 OOF flags")
    return numeric.astype(bool)


def _first_present(frame: pd.DataFrame, candidates: Sequence[str], *, description: str) -> str:
    for name in candidates:
        if name in frame.columns:
            return name
    raise PredictionLineageError(f"prediction frame is missing {description}; tried {list(candidates)}")


def _feature_rows(value: Any, *, side: str = "all") -> list[dict[str, str]]:
    """Read the common existing contract shapes without inventing a new one."""
    if isinstance(value, (list, tuple)):
        return [
            {"model_side": side, "feature_name": str(name)}
            for name in value
            if isinstance(name, str) and name.strip()
        ]
    if isinstance(value, Mapping):
        rows: list[dict[str, str]] = []
        for key, item in value.items():
            # Side-local contracts usually look like {"long": [...], "short": [...]}.
            child_side = str(key) if str(key) in {"long", "short", "all", "shared"} else side
            rows.extend(_feature_rows(item, side=child_side))
        return rows
    return []


def feature_contract_rows(payload: Mapping[str, Any]) -> pd.DataFrame:
    """Extract ordered named features from existing raw or side-local contracts.

    Supported top-level keys intentionally mirror existing artifacts:
    ``raw_feature_columns``, ``feature_columns``, ``feature_names``,
    ``features``, ``feature_contract``, and the two common ``*_by_side`` keys.
    ``feature_contract`` is allowed to be either a list or a side mapping.
    """
    keys = (
        "raw_feature_columns", "feature_columns", "feature_names", "features",
        "feature_contract", "features_by_side", "selected_features_by_side",
        "selected_features",
    )
    rows: list[dict[str, str]] = []
    for key in keys:
        if key in payload:
            rows.extend(_feature_rows(payload[key]))
    if not rows:
        raise FeatureEligibilityError(f"feature contract has none of the supported feature keys: {list(keys)}")
    out = pd.DataFrame(rows).drop_duplicates(["model_side", "feature_name"], keep="first")
    if out.feature_name.duplicated().any() and out.model_side.eq("all").any():
        # A shared list can co-exist with per-side lists; retain both because a
        # manifest should describe actual layer/side use rather than flatten it.
        out = out.reset_index(drop=True)
    return out


def load_feature_contract(path: Path) -> pd.DataFrame:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise FeatureEligibilityError(f"feature contract {path} must be a JSON object")
    out = feature_contract_rows(payload)
    out["contract_path"] = str(path)
    out["contract_sha256"] = sha256(path)
    return out


def _metadata_index(feature_lineage: pd.DataFrame | None) -> pd.DataFrame:
    if feature_lineage is None or feature_lineage.empty:
        return pd.DataFrame(index=pd.Index([], name="feature_name"))
    if "feature_name" not in feature_lineage:
        raise FeatureEligibilityError("feature lineage must have a feature_name column")
    if feature_lineage.feature_name.astype(str).duplicated().any():
        raise FeatureEligibilityError("feature lineage has duplicate feature_name entries")
    out = feature_lineage.copy()
    out["feature_name"] = out.feature_name.astype(str)
    return out.set_index("feature_name", drop=False)


def _truth_or_unknown(value: Any) -> str:
    if value is True or value == 1:
        return "TRUE"
    if value is False or value == 0:
        return "FALSE"
    return "UNDECLARED"


def _semantic_disposition(name: str) -> tuple[str, str]:
    key = name.lower()
    if key in IDENTITY_TOKENS:
        return "REJECT_IDENTITY", "row identity/timestamp is not a model feature"
    if any(token in key for token in ACTION_ONLY_TOKENS):
        return "REJECT_ACTION_LAYER_ONLY", "timing/MAE/target-price/wait decision belongs to the action layer"
    if any(token in key for token in DIRECT_ECONOMIC_TARGET_TOKENS):
        return "REJECT_DIRECT_TARGET_OR_COST", "direct realised economic target or realised cost"
    # A predicted path-head output is not the realised path itself.  It may be
    # an execution context only after an exact-ID OOF audit; its unprefixed
    # counterpart below remains forbidden.  This is what permits e.g. a
    # predicted peak-MFE or future-slope head without admitting MFE/slope
    # labels from the evaluation row.
    if any(token in key for token in MODEL_DERIVED_TOKENS):
        return "MODEL_DERIVED", "upstream model output requires strict OOF lineage"
    if any(token in key for token in REALIZED_PATH_TOKENS):
        return "REJECT_REALIZED_PATH", "realised/future path or path-derived target"
    return "CAUSAL_RAW", "no prohibited target/action/model-derived semantic token"


def build_feature_eligibility_manifest(
    *,
    base_contract: pd.DataFrame,
    execution_contract: pd.DataFrame,
    feature_lineage: pd.DataFrame | None = None,
    declared_oof_prediction_features: Iterable[str] = (),
) -> pd.DataFrame:
    """Build a layer-specific, machine-readable eligibility manifest.

    ``declared_oof_prediction_features`` is intentionally an explicit allow
    list rather than a name-pattern bypass.  Even allowed outputs remain
    conditional until their candidate-level table passes
    :func:`audit_oof_prediction_lineage`.
    """
    lineage = _metadata_index(feature_lineage)
    oof_names = {str(name) for name in declared_oof_prediction_features}
    rows: list[dict[str, Any]] = []
    for layer, contract in (("base", base_contract), ("execution", execution_contract)):
        required = {"feature_name", "model_side", "contract_path", "contract_sha256"}
        missing = sorted(required.difference(contract.columns))
        if missing:
            raise FeatureEligibilityError(f"{layer} contract rows missing columns: {missing}")
        for item in contract.itertuples(index=False):
            name = str(item.feature_name)
            kind, reason = _semantic_disposition(name)
            meta = lineage.loc[name] if name in lineage.index else pd.Series(dtype=object)
            pit = _truth_or_unknown(meta.get("point_in_time_safe"))
            live = _truth_or_unknown(meta.get("live_reproducible"))
            if pit == "FALSE":
                status = "REJECT_LINEAGE_NOT_POINT_IN_TIME_SAFE"
                eligible_now = eligible_if_audited = False
                reason = "feature lineage explicitly marks input not point-in-time safe"
            elif kind != "CAUSAL_RAW" and kind != "MODEL_DERIVED":
                status = kind
                eligible_now = eligible_if_audited = False
            elif kind == "MODEL_DERIVED" and layer == "base":
                status = "REJECT_BASE_MODEL_DERIVED_INPUT"
                eligible_now = eligible_if_audited = False
                reason = "base layer cannot consume model outputs or recursive scores"
            elif kind == "MODEL_DERIVED" and name not in oof_names:
                status = "REJECT_UNDECLARED_OOF_PREDICTION"
                eligible_now = eligible_if_audited = False
                reason = "execution layer model output is absent from explicit OOF prediction allow list"
            elif kind == "MODEL_DERIVED":
                status = "CONDITIONAL_OOF_LINEAGE_REQUIRED"
                eligible_now, eligible_if_audited = False, True
                reason = "allowed execution context only after exact-ID strict-OOF lineage audit"
            else:
                status = "ELIGIBLE_RESEARCH_CAUSAL"
                eligible_now = eligible_if_audited = True
            rows.append({
                "model_layer": layer,
                "model_side": str(item.model_side),
                "feature_name": name,
                "contract_path": str(item.contract_path),
                "contract_sha256": str(item.contract_sha256),
                "semantic_class": kind,
                "eligibility_status": status,
                "eligible_now": bool(eligible_now),
                "eligible_if_prediction_lineage_audited": bool(eligible_if_audited),
                "requires_prediction_lineage_audit": bool(status == "CONDITIONAL_OOF_LINEAGE_REQUIRED"),
                "point_in_time_safe": pit,
                "live_reproducible": live,
                "production_live_status": (
                    "VERIFIED" if eligible_now and pit == "TRUE" and live == "TRUE"
                    else "NOT_VERIFIED" if eligible_if_audited else "INELIGIBLE"
                ),
                "reason": reason,
            })
    return pd.DataFrame(rows).sort_values(["model_layer", "model_side", "feature_name"], kind="stable").reset_index(drop=True)


def assert_layer_eligibility(
    manifest: pd.DataFrame,
    *,
    permit_conditional_oof: bool = False,
    require_live_reproducibility: bool = False,
) -> None:
    """Fail before fitting if a selected layer input has no valid contract."""
    required = {"feature_name", "eligibility_status", "eligible_now", "eligible_if_prediction_lineage_audited", "production_live_status"}
    missing = sorted(required.difference(manifest.columns))
    if missing:
        raise FeatureEligibilityError(f"eligibility manifest missing columns: {missing}")
    valid = manifest.eligible_now.astype(bool)
    if permit_conditional_oof:
        valid = valid | manifest.eligible_if_prediction_lineage_audited.astype(bool)
    if require_live_reproducibility:
        valid &= manifest.production_live_status.eq("VERIFIED")
    if not valid.all():
        blocked = manifest.loc[~valid, ["model_layer", "model_side", "feature_name", "eligibility_status"]]
        raise FeatureEligibilityError(f"ineligible layer features: {blocked.to_dict(orient='records')}")


def audit_oof_prediction_lineage(
    candidates: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    model_layer: str,
    prediction_columns: Sequence[str],
    candidate_id_col: str = "candidate_id",
    decision_ts_col: str = "decision_ts",
    feature_cutoff_ts_col: str = "feature_cutoff_ts",
    prediction_ts_col: str | None = None,
    fit_end_ts_col: str | None = None,
    oof_flag_col: str | None = None,
) -> PredictionLineageAudit:
    """Verify strict candidate-level OOF provenance for base or execution scores.

    Base scores must be available no later than the feature cutoff.  Execution
    scores may be produced at the decision timestamp, but never after it.  In
    both cases the row-specific training ``fit_end`` must be strictly earlier
    than the prediction timestamp, preventing same-row/future fitted scores
    from masquerading as OOF evidence.
    """
    if model_layer not in {"base", "execution"}:
        raise PredictionLineageError("model_layer must be 'base' or 'execution'")
    for frame, label in ((candidates, "candidates"), (predictions, "predictions")):
        if candidate_id_col not in frame:
            raise PredictionLineageError(f"{label} missing {candidate_id_col!r}")
        if frame[candidate_id_col].isna().any() or frame[candidate_id_col].astype(str).duplicated().any():
            raise PredictionLineageError(f"{label} {candidate_id_col} must be non-null and one-to-one")
    missing_scores = sorted(set(prediction_columns).difference(predictions.columns))
    if missing_scores:
        raise PredictionLineageError(f"prediction frame missing score columns: {missing_scores}")
    decision = _normalise_utc(candidates, decision_ts_col, context="candidates")
    cutoff = _normalise_utc(candidates, feature_cutoff_ts_col, context="candidates")
    if not cutoff.le(decision).all():
        raise PredictionLineageError("candidate feature_cutoff_ts must be no later than decision_ts")
    prediction_ts_col = prediction_ts_col or _first_present(
        predictions, ("prediction_ts", "score_ts", "__ts__", "feature_cutoff_ts"), description="prediction timestamp",
    )
    fit_end_ts_col = fit_end_ts_col or _first_present(
        predictions, ("fit_end_ts", "fit_end", "source_model_fit_end", "train_max_label_available"), description="fit-end timestamp",
    )
    oof_flag_col = oof_flag_col or _first_present(
        predictions, ("is_oof", "residual_is_oof", "prediction_is_oof", "oof"), description="OOF flag",
    )
    prediction_ts = _normalise_utc(predictions, prediction_ts_col, context="predictions")
    fit_end = _normalise_utc(predictions, fit_end_ts_col, context="predictions")
    is_oof = _strict_bool(predictions[oof_flag_col], context=f"predictions.{oof_flag_col}")
    numeric_predictions = predictions.loc[:, prediction_columns].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric_predictions.to_numpy(float)).all():
        raise PredictionLineageError("prediction score columns must be finite")

    left = candidates.loc[:, [candidate_id_col]].copy()
    left["_decision_ts"] = decision.to_numpy()
    left["_feature_cutoff_ts"] = cutoff.to_numpy()
    right = predictions.loc[:, [candidate_id_col]].copy()
    right["_prediction_ts"] = prediction_ts.to_numpy()
    right["_fit_end_ts"] = fit_end.to_numpy()
    right["_is_oof"] = is_oof.to_numpy()
    joined = left.merge(right, on=candidate_id_col, how="left", validate="one_to_one", indicator=True)
    if not joined["_merge"].eq("both").all():
        missing = joined.loc[~joined["_merge"].eq("both"), candidate_id_col].astype(str).head(10).tolist()
        raise PredictionLineageError(f"prediction candidate join is incomplete; examples: {missing}")
    # Predictions must not contain rows absent from this candidate population.
    extra = predictions.loc[~predictions[candidate_id_col].astype(str).isin(candidates[candidate_id_col].astype(str)), candidate_id_col]
    if len(extra):
        raise PredictionLineageError("prediction candidate join has rows outside candidate population")

    available_by = joined["_feature_cutoff_ts"] if model_layer == "base" else joined["_decision_ts"]
    joined["prediction_at_or_before_layer_cutoff"] = joined["_prediction_ts"].le(available_by)
    joined["fit_end_strictly_before_prediction"] = joined["_fit_end_ts"].lt(joined["_prediction_ts"])
    joined["strict_oof"] = joined["_is_oof"]
    joined["lineage_pass"] = (
        joined.prediction_at_or_before_layer_cutoff
        & joined.fit_end_strictly_before_prediction
        & joined.strict_oof
    )
    if not joined.lineage_pass.all():
        failed = joined.loc[
            ~joined.lineage_pass,
            [candidate_id_col, "prediction_at_or_before_layer_cutoff", "fit_end_strictly_before_prediction", "strict_oof"],
        ].head(10).to_dict(orient="records")
        raise PredictionLineageError(f"strict OOF lineage failed: {failed}")
    output = joined.drop(columns=["_merge"])
    summary = {
        "model_layer": model_layer,
        "rows": int(len(output)),
        "candidate_identity": [candidate_id_col],
        "prediction_columns": list(prediction_columns),
        "prediction_timestamp_column": prediction_ts_col,
        "fit_end_timestamp_column": fit_end_ts_col,
        "oof_flag_column": oof_flag_col,
        "availability_rule": "prediction_ts <= feature_cutoff_ts" if model_layer == "base" else "prediction_ts <= decision_ts",
        "fit_end_rule": "fit_end_ts < prediction_ts",
        "status": "STRICT_OOF_LINEAGE_VERIFIED",
    }
    return PredictionLineageAudit(summary=summary, rows=output)


def audit_base_to_execution_handoff(
    candidates: pd.DataFrame,
    base_predictions: pd.DataFrame,
    *,
    prediction_columns: Sequence[str],
    **kwargs: Any,
) -> PredictionLineageAudit:
    """Explicit alias for the only permitted model-output handoff into execution.

    Callers should run this before marking a ``score_*`` feature as eligible in
    an execution contract.  The base-layer availability rule is intentionally
    retained: a base score delivered after the feature cutoff is not a valid
    execution input, even when it happens before the final decision timestamp.
    """
    return audit_oof_prediction_lineage(
        candidates, base_predictions, model_layer="base", prediction_columns=prediction_columns, **kwargs,
    )


def run(
    *,
    base_contract_path: Path,
    execution_contract_path: Path,
    output: Path,
    feature_lineage_path: Path | None = None,
    declared_oof_prediction_features: Iterable[str] = (),
    candidates_path: Path | None = None,
    base_predictions_path: Path | None = None,
    base_prediction_columns: Sequence[str] = (),
    execution_predictions_path: Path | None = None,
    execution_prediction_columns: Sequence[str] = (),
) -> dict[str, Any]:
    """Materialise only audit evidence; no model/policy data is changed."""
    if output.exists():
        raise FileExistsError(output)
    base = load_feature_contract(base_contract_path)
    execution = load_feature_contract(execution_contract_path)
    lineage = pd.read_parquet(feature_lineage_path) if feature_lineage_path else None
    eligibility = build_feature_eligibility_manifest(
        base_contract=base,
        execution_contract=execution,
        feature_lineage=lineage,
        declared_oof_prediction_features=declared_oof_prediction_features,
    )
    prediction_audits: list[PredictionLineageAudit] = []
    if any((base_predictions_path, execution_predictions_path)) and candidates_path is None:
        raise PredictionLineageError("candidates_path is required when auditing prediction lineage")
    candidates = pd.read_parquet(candidates_path) if candidates_path else None
    if base_predictions_path:
        if not base_prediction_columns:
            raise PredictionLineageError("base_prediction_columns is required with base_predictions_path")
        prediction_audits.append(audit_oof_prediction_lineage(
            candidates, pd.read_parquet(base_predictions_path), model_layer="base", prediction_columns=base_prediction_columns,
        ))
    if execution_predictions_path:
        if not execution_prediction_columns:
            raise PredictionLineageError("execution_prediction_columns is required with execution_predictions_path")
        prediction_audits.append(audit_oof_prediction_lineage(
            candidates, pd.read_parquet(execution_predictions_path), model_layer="execution", prediction_columns=execution_prediction_columns,
        ))

    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        eligibility.to_parquet(stage / "feature_eligibility_manifest.parquet", index=False, compression="zstd")
        # Preserve a readable zero-row schema when this is intentionally a
        # feature-only audit.  A zero-column dataframe cannot be written as a
        # portable Parquet file by every engine.
        prediction_summary = pd.DataFrame(
            [audit.summary for audit in prediction_audits], columns=PREDICTION_AUDIT_COLUMNS,
        )
        prediction_summary.to_parquet(stage / "prediction_lineage_audit.parquet", index=False, compression="zstd")
        if prediction_audits:
            rows = pd.concat(
                [audit.rows.assign(model_layer=audit.summary["model_layer"]) for audit in prediction_audits],
                ignore_index=True,
            )
            rows.to_parquet(stage / "prediction_lineage_rows.parquet", index=False, compression="zstd")
        inputs = {str(base_contract_path): sha256(base_contract_path), str(execution_contract_path): sha256(execution_contract_path)}
        for path in (feature_lineage_path, candidates_path, base_predictions_path, execution_predictions_path):
            if path:
                inputs[str(path)] = sha256(path)
        outputs = {
            name: sha256(stage / name)
            for name in ("feature_eligibility_manifest.parquet", "prediction_lineage_audit.parquet")
        }
        if prediction_audits:
            outputs["prediction_lineage_rows.parquet"] = sha256(stage / "prediction_lineage_rows.parquet")
        manifest = {
            "schema": SCHEMA,
            "status": "COMPLETE_READ_ONLY_CONTRACT_AUDIT",
            "promotion_eligible": False,
            "base_layer_rule": "causal/raw inputs only; model-derived inputs rejected",
            "execution_layer_rule": "causal/raw inputs plus explicitly declared upstream outputs after strict OOF lineage audit",
            "action_layer_exclusion": "timing, MAE, target-price and wait inputs rejected from base and execution-EV",
            "declared_oof_prediction_features": sorted(str(name) for name in declared_oof_prediction_features),
            "feature_counts": eligibility.groupby(["model_layer", "eligibility_status"], sort=True).size().rename("rows").reset_index().to_dict(orient="records"),
            "prediction_audits": [audit.summary for audit in prediction_audits],
            "inputs_sha256": inputs,
            "outputs_sha256": outputs,
            "runner": {"path": str(Path(__file__).relative_to(ROOT)), "sha256": sha256(Path(__file__))},
        }
        _write_json(stage / "run_manifest.json", manifest)
        os.replace(stage, output)
        return manifest
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-contract", type=Path, required=True)
    parser.add_argument("--execution-contract", type=Path, required=True)
    parser.add_argument("--feature-lineage", type=Path)
    parser.add_argument("--oof-feature", action="append", default=[], help="Explicit execution input that requires a strict OOF audit; repeatable.")
    parser.add_argument("--candidates", type=Path)
    parser.add_argument("--base-predictions", type=Path)
    parser.add_argument("--base-prediction-column", action="append", default=[])
    parser.add_argument("--execution-predictions", type=Path)
    parser.add_argument("--execution-prediction-column", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = run(
        base_contract_path=args.base_contract,
        execution_contract_path=args.execution_contract,
        feature_lineage_path=args.feature_lineage,
        declared_oof_prediction_features=args.oof_feature,
        candidates_path=args.candidates,
        base_predictions_path=args.base_predictions,
        base_prediction_columns=args.base_prediction_column,
        execution_predictions_path=args.execution_predictions,
        execution_prediction_columns=args.execution_prediction_column,
        output=args.output,
    )
    print(json.dumps(_json_safe(manifest), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
