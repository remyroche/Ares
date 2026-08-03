#!/usr/bin/env python3
"""Exact Primary100 config-routed competing-risk context add/drop runner.

This is deliberately a *new* diagnostic lineage.  It joins the 134,889-row
exact Primary100 competing-risk label panel, resolves the existing config
base/meta pools, and tests whether strictly cross-fitted event/payoff context
improves a final direct-net residual head.  It is not a policy, timing, MAE,
wait, or target-price action model.

The important leakage boundary is two-stage: inside every outer training
window the base score and every context channel are chronological OOF; the
final residual head is fitted only on their common OOF rows.  On the outer
evaluation window it sees only predictions from models fitted on the outer
training window.  All joins bind the outer split and exact identity and retain
the frozen row order.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_execution_ev_competing_risk_simplex_ablation import (  # noqa: E402
    CLASS_COLUMN,
    CLASS_NAMES,
    PRIMARY_DIR,
    load_competing_risk_panel,
)
from scripts.run_meaningful_mfe_exact_grid_base_residual_sensitivity import (  # noqa: E402
    _crossfit_plan,
    config_routed_feature_pools,
)
from scripts.run_meaningful_mfe_exact_grid_reset import (  # noqa: E402
    IDENTITY,
    SIDES,
    TRANSFER_SPECS,
    _base_masks,
    july_grouped_day_folds,
    stable_top,
)
from scripts.materialize_primary100_context_sidecar import (  # noqa: E402
    CANDIDATE_FIELDS as SIDECAR_CANDIDATE_FIELDS,
    DAE_FIELDS,
    DEFAULT_OUTPUT as DEFAULT_CONTEXT_SIDECAR_DIR,
    EXPECTED_ROWS as SIDECAR_EXPECTED_ROWS,
    GMM_GEOMETRY_FIELDS,
    GMM_POSTERIOR_FIELDS,
    GMM_RISK_FIELDS,
    REPRESENTATION_AVAILABILITY,
    REPRESENTATION_FIELDS,
    SCHEMA as SIDECAR_SCHEMA,
    TRANSITION_OUTPUT_FIELDS,
)


SCHEMA = "execution_ev_competing_risk_context_add_drop_v1"
EXPECTED_PRIMARY100_IDENTITIES = SIDECAR_EXPECTED_ROWS
HORIZON = pd.Timedelta(hours=12)
TOP_FRACTIONS = (0.01, 0.05, 0.10, 0.20)

# The materializer's broad geometry list also contains cluster/posterior-risk
# summaries.  The initial geometry arm uses only distances to centres and
# Mahalanobis distances, plus availability.  Posterior and compact risk fields
# are explicitly forbidden until an incremental gate proves their value.
GMM_DISTANCE_GEOMETRY_FIELDS = tuple(
    field
    for field in GMM_GEOMETRY_FIELDS
    if field.startswith("gmm_dist_center_") or field.startswith("gmm_mahal_")
)
DAE_BLOCK_FIELDS = (*DAE_FIELDS, *GMM_RISK_FIELDS, REPRESENTATION_AVAILABILITY)
GMM_GEOMETRY_BLOCK_FIELDS = (*GMM_DISTANCE_GEOMETRY_FIELDS, REPRESENTATION_AVAILABILITY)
RAW_TRANSITION_BLOCK_FIELDS = tuple(TRANSITION_OUTPUT_FIELDS)
FORBIDDEN_REPRESENTATION_FIELDS = (*GMM_POSTERIOR_FIELDS, *tuple(
    field for field in GMM_GEOMETRY_FIELDS if field not in GMM_DISTANCE_GEOMETRY_FIELDS
),)

# Context is deliberately prediction-only.  These names name model outputs,
# never observed outer labels.  The arms make incremental claims testable.
CONTEXT_CHANNELS = (
    "p_timeout",
    "p_adverse",
    "p_clean",
    "pred_clean_gross",
    "pred_clean_value",
    "pred_clean_rank",
)
CANDIDATE_CONTEXT_FIELDS: Mapping[str, tuple[str, ...]] = {
    "cutoff_context": (
        "base_margin_to_cutoff",
        "base_margin_to_cutoff_z",
    ),
    "timestamp_relative_context": (
        "base_candidate_rank_pct_timestamp_side",
        "base_score_z_timestamp_side",
    ),
    "archetype_relative_context": ("base_signal_zscore_within_archetype",),
    "rank_group_context": ("base_rank_decile", "base_candidate_group_rows"),
}
DIAGNOSTIC_ONLY_CANDIDATE_FIELDS = ("base_candidate_rank_timestamp_side",)
ALL_CANDIDATE_CONTEXT = tuple(
    field for values in CANDIDATE_CONTEXT_FIELDS.values() for field in values
)
ALLOWED_SIDECAR_MODEL_FIELDS = (
    *ALL_CANDIDATE_CONTEXT,
    *DAE_BLOCK_FIELDS,
    *GMM_GEOMETRY_BLOCK_FIELDS,
    *RAW_TRANSITION_BLOCK_FIELDS,
)
ARMS: Mapping[str, tuple[str, ...]] = {
    "base_only": (),
    "direct_meta_only": (),
    "direct_meta_plus_alpha": (),
    "plus_clean_probability": ("p_clean",),
    "plus_competing_risk": ("p_timeout", "p_adverse", "p_clean"),
    "plus_clean_payoff": ("p_clean", "pred_clean_gross"),
    "plus_clean_value": ("p_clean", "pred_clean_gross", "pred_clean_value"),
    "plus_clean_value_rank": (
        "p_clean",
        "pred_clean_gross",
        "pred_clean_value",
        "pred_clean_rank",
    ),
    # Candidate-context fields are frozen per-candidate OOF context, not
    # outcomes.  They are ablated individually before the joint arm.
    "plus_cutoff_context": (),
    "plus_timestamp_relative_context": (),
    "plus_archetype_relative_context": (),
    "plus_rank_group_context": (),
    "plus_candidate_context_all": (),
    "plus_candidate_context_and_clean_value_rank": (
        "p_clean",
        "pred_clean_gross",
        "pred_clean_value",
        "pred_clean_rank",
    ),
    "plus_dae": (),
    "plus_gmm_geometry": (),
    "plus_raw_transition": (),
    "plus_candidate_context_clean_value_rank_dae": (
        "p_clean", "pred_clean_gross", "pred_clean_value", "pred_clean_rank",
    ),
    "plus_candidate_context_clean_value_rank_gmm_geometry": (
        "p_clean", "pred_clean_gross", "pred_clean_value", "pred_clean_rank",
    ),
    "plus_candidate_context_clean_value_rank_raw_transition": (
        "p_clean", "pred_clean_gross", "pred_clean_value", "pred_clean_rank",
    ),
}
ARM_INCLUDE_ALPHA: Mapping[str, bool] = {
    "base_only": True,
    "direct_meta_only": False,
    "direct_meta_plus_alpha": True,
    "plus_clean_probability": True,
    "plus_competing_risk": True,
    "plus_clean_payoff": True,
    "plus_clean_value": True,
    "plus_clean_value_rank": True,
    "plus_cutoff_context": True,
    "plus_timestamp_relative_context": True,
    "plus_archetype_relative_context": True,
    "plus_rank_group_context": True,
    "plus_candidate_context_all": True,
    "plus_candidate_context_and_clean_value_rank": True,
    "plus_dae": True,
    "plus_gmm_geometry": True,
    "plus_raw_transition": True,
    "plus_candidate_context_clean_value_rank_dae": True,
    "plus_candidate_context_clean_value_rank_gmm_geometry": True,
    "plus_candidate_context_clean_value_rank_raw_transition": True,
}
ARM_CANDIDATE_CONTEXT: Mapping[str, tuple[str, ...]] = {
    "base_only": (),
    "direct_meta_only": (),
    "direct_meta_plus_alpha": (),
    "plus_clean_probability": (),
    "plus_competing_risk": (),
    "plus_clean_payoff": (),
    "plus_clean_value": (),
    "plus_clean_value_rank": (),
    "plus_cutoff_context": CANDIDATE_CONTEXT_FIELDS["cutoff_context"],
    "plus_timestamp_relative_context": CANDIDATE_CONTEXT_FIELDS["timestamp_relative_context"],
    "plus_archetype_relative_context": CANDIDATE_CONTEXT_FIELDS["archetype_relative_context"],
    "plus_rank_group_context": CANDIDATE_CONTEXT_FIELDS["rank_group_context"],
    "plus_candidate_context_all": ALL_CANDIDATE_CONTEXT,
    "plus_candidate_context_and_clean_value_rank": ALL_CANDIDATE_CONTEXT,
    "plus_dae": DAE_BLOCK_FIELDS,
    "plus_gmm_geometry": GMM_GEOMETRY_BLOCK_FIELDS,
    "plus_raw_transition": RAW_TRANSITION_BLOCK_FIELDS,
    "plus_candidate_context_clean_value_rank_dae": (*ALL_CANDIDATE_CONTEXT, *DAE_BLOCK_FIELDS),
    "plus_candidate_context_clean_value_rank_gmm_geometry": (*ALL_CANDIDATE_CONTEXT, *GMM_GEOMETRY_BLOCK_FIELDS),
    "plus_candidate_context_clean_value_rank_raw_transition": (*ALL_CANDIDATE_CONTEXT, *RAW_TRANSITION_BLOCK_FIELDS),
}
DEFAULT_CONTEXT_SIDECAR = DEFAULT_CONTEXT_SIDECAR_DIR / "context.parquet"

# Action-layer fields are explicitly out of scope.  The assertion applies to
# every config-routed feature list before any model matrix is constructed.
ACTION_EXCLUSION_TOKENS = (
    "timing",
    "time_to",
    "mae",
    "wait",
    "target_price",
    "targetprice",
    "entry_price",
    "suggested_price",
    "action_",
)

RESIDUAL_PARAMS: Mapping[str, Any] = {
    "iterations": 260,
    "learning_rate": 0.030,
    "depth": 5,
    "l2_leaf_reg": 12.0,
}
BASE_ALPHA_COLUMN = "base_oof_score"
OBSERVED_OUTCOME_COLUMNS = {
    CLASS_COLUMN,
    *CLASS_NAMES,
    "execution_gross_ev_12h",
    "execution_cost_return",
    "execution_net_ev_12h",
}


def _safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_safe(item) for item in value]
    if isinstance(value, (np.generic,)):
        return value.item()
    if isinstance(value, (Path, pd.Timestamp)):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(_safe(payload), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def forbid_action_features(features: Sequence[str]) -> list[str]:
    """Fail closed rather than silently feeding action-layer fields upstream."""

    result = [str(feature) for feature in features]
    forbidden = [
        feature
        for feature in result
        if any(token in feature.lower() for token in ACTION_EXCLUSION_TOKENS)
    ]
    if forbidden:
        raise ValueError(
            "context runner excludes timing/MAE/wait/target-price action features: "
            + ", ".join(sorted(forbidden))
        )
    return result


def validate_primary100_contract(panel: pd.DataFrame, *, expected_rows: int | None = EXPECTED_PRIMARY100_IDENTITIES) -> None:
    """Validate the exact, one-row-per-identity Primary100 handoff."""

    required = {
        *IDENTITY,
        "execution_decision_utc",
        "label_resolution_utc",
        "execution_gross_ev_12h",
        "execution_cost_return",
        "execution_net_ev_12h",
        BASE_ALPHA_COLUMN,
        CLASS_COLUMN,
        *CLASS_NAMES,
    }
    missing = sorted(required.difference(panel.columns))
    if missing:
        raise ValueError(f"Primary100 context panel lacks required columns: {missing}")
    if expected_rows is not None and len(panel) != int(expected_rows):
        raise ValueError(
            f"Primary100 context panel must contain exactly {expected_rows} identities; got {len(panel)}"
        )
    if panel.duplicated(list(IDENTITY)).any():
        raise ValueError("Primary100 context panel has duplicate exact identities")
    if not panel["side_name"].astype(str).str.lower().isin(SIDES).all():
        raise ValueError("Primary100 context panel has noncanonical sides")
    decision = pd.to_datetime(panel["execution_decision_utc"], utc=True, errors="raise")
    resolution = pd.to_datetime(panel["label_resolution_utc"], utc=True, errors="raise")
    if not resolution.eq(decision + HORIZON).all():
        raise ValueError("Primary100 context labels must resolve exactly 12h after decision")
    gross = panel["execution_gross_ev_12h"].to_numpy(float)
    cost = panel["execution_cost_return"].to_numpy(float)
    net = panel["execution_net_ev_12h"].to_numpy(float)
    if not np.isfinite(gross).all() or not np.isfinite(cost).all() or not np.isfinite(net).all():
        raise ValueError("Primary100 context economics must be finite")
    if not np.allclose(gross - cost, net, atol=1e-7, rtol=0.0):
        raise ValueError("Primary100 context gross-cost=net identity failed")
    simplex = panel[list(CLASS_NAMES)].to_numpy(int)
    hard = panel[CLASS_COLUMN].to_numpy(int)
    if not np.array_equal(simplex.sum(axis=1), np.ones(len(panel), dtype=int)) or not np.array_equal(hard, simplex.argmax(axis=1)):
        raise ValueError("Primary100 competing-risk class is not a closed hard simplex")
    if not np.isfinite(panel[BASE_ALPHA_COLUMN].to_numpy(float)).all():
        raise ValueError("Primary100 context panel lacks a finite frozen alpha OOF score")


def validate_context_sidecar_source(path: Path) -> dict[str, Any]:
    """Bind the exact outcome-free sidecar to its manifest and report."""

    path = Path(path).resolve()
    manifest_path = path.parent / "manifest.json"
    if not path.is_file() or not manifest_path.is_file():
        raise ValueError("context sidecar and sibling manifest must exist")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != SIDECAR_SCHEMA or manifest.get("status") != "MATERIALIZED_EXACT_OUTCOME_FREE_PRIMARY100_CONTEXT":
        raise ValueError("context sidecar manifest has unexpected schema or non-outcome-free status")
    output = manifest.get("output", {})
    bound_path = Path(str(output.get("path", ""))).resolve()
    if bound_path != path or output.get("sha256") != _sha256(path):
        raise ValueError("context sidecar manifest does not bind requested parquet hash/path")
    if int(output.get("rows", -1)) != EXPECTED_PRIMARY100_IDENTITIES:
        raise ValueError("context sidecar manifest does not bind the exact Primary100 row count")
    report = manifest.get("report", {})
    report_path = Path(str(report.get("path", ""))).resolve()
    if not report_path.is_file() or report.get("sha256") != _sha256(report_path):
        raise ValueError("context sidecar manifest does not bind its report hash/path")
    return {
        "path": path,
        "sha256": _sha256(path),
        "manifest": manifest_path,
        "manifest_sha256": _sha256(manifest_path),
        "report": report_path,
        "report_sha256": _sha256(report_path),
        "manifest_schema": manifest["schema"],
        "manifest_status": manifest["status"],
        "source_rows": int(output["rows"]),
        "candidate_identity_sha256": output.get("candidate_identity_sha256"),
    }


def validate_sidecar_representation_missingness(context: pd.DataFrame, fields: Sequence[str] = REPRESENTATION_FIELDS) -> None:
    """Allow native representation NaNs only where materialized availability=0."""

    missing = sorted(set([REPRESENTATION_AVAILABILITY, *fields]).difference(context.columns))
    if missing:
        raise ValueError(f"context sidecar lacks representation fields: {missing}")
    availability = pd.to_numeric(context[REPRESENTATION_AVAILABILITY], errors="coerce")
    if not availability.isin((0.0, 1.0)).all():
        raise ValueError("context sidecar representation availability must be binary 0/1")
    values = context.loc[:, list(fields)].apply(pd.to_numeric, errors="coerce")
    if np.isinf(values.to_numpy(float)).any():
        raise ValueError("context sidecar representation values contain infinities")
    if (values.isna().any(axis=1) & availability.eq(1.0)).any():
        raise ValueError("representation NaNs are permitted only where availability=0")


def load_exact_context_sidecar(panel: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Join the hash-bound, outcome-free 134,889-row sidecar in frozen order."""

    validate_context_sidecar_source(path)
    required = [
        *IDENTITY,
        *SIDECAR_CANDIDATE_FIELDS,
        "candidate_prediction_source",
        "candidate_selected_top40",
        REPRESENTATION_AVAILABILITY,
        *REPRESENTATION_FIELDS,
        *TRANSITION_OUTPUT_FIELDS,
    ]
    source = pd.read_parquet(path, columns=required)
    if source.duplicated(list(IDENTITY)).any() or len(source) != EXPECTED_PRIMARY100_IDENTITIES:
        raise ValueError("context sidecar must contain exactly one row per Primary100 identity")
    if not source["candidate_prediction_source"].astype(str).eq("outer_oof_fold_model").all() or not source["candidate_selected_top40"].astype(bool).all():
        raise ValueError("context sidecar candidate stream is not the frozen selected outer-OOF source")
    if not np.isfinite(source[list(SIDECAR_CANDIDATE_FIELDS)].to_numpy(float)).all() or not np.isfinite(source[list(TRANSITION_OUTPUT_FIELDS)].to_numpy(float)).all():
        raise ValueError("context sidecar candidate/transition fields must be finite")
    validate_sidecar_representation_missingness(source)
    source["__exact_sidecar_match__"] = 1
    context = panel[list(IDENTITY)].merge(source, on=list(IDENTITY), how="left", validate="one_to_one", sort=False)
    if len(context) != len(panel) or not context["__exact_sidecar_match__"].eq(1).all():
        raise ValueError("context sidecar lacks complete exact Primary100 identity coverage")
    if not context[list(IDENTITY)].equals(panel[list(IDENTITY)].reset_index(drop=True)):
        raise ValueError("context-sidecar join changed frozen Primary100 identity order")
    if not np.allclose(context[BASE_ALPHA_COLUMN].to_numpy(float), panel[BASE_ALPHA_COLUMN].to_numpy(float), atol=1e-12, rtol=0.0):
        raise ValueError("context sidecar frozen alpha does not match Primary100 handoff")
    return context.drop(columns="__exact_sidecar_match__").reset_index(drop=True)


def _identity_frame(panel: pd.DataFrame, positions: np.ndarray, outer_split: str) -> pd.DataFrame:
    result = panel.iloc[np.asarray(positions, dtype=int)][list(IDENTITY)].copy().reset_index(drop=True)
    result["outer_split"] = str(outer_split)
    return result


def safe_oof_join(
    anchor: pd.DataFrame,
    predicted: pd.DataFrame,
    *,
    value_columns: Sequence[str],
) -> pd.DataFrame:
    """Join OOF context by split+identity while proving order and coverage.

    A bare identity is intentionally insufficient: an identity occurs in more
    than one diagnostic outer evaluation.  This rejects duplicate predictions
    rather than accidentally selecting a later/reverse-fold value.
    """

    key = ["outer_split", *IDENTITY]
    for name, frame, required in (("anchor", anchor, key), ("predicted", predicted, [*key, *value_columns])):
        missing = sorted(set(required).difference(frame.columns))
        if missing:
            raise ValueError(f"{name} OOF join lacks fields: {missing}")
        if frame.duplicated(key).any():
            raise ValueError(f"{name} OOF join has duplicate split/identity keys")
    joined = anchor.merge(predicted[[*key, *value_columns]], on=key, how="left", validate="one_to_one", sort=False)
    if len(joined) != len(anchor) or joined[list(value_columns)].isna().any().any():
        raise ValueError("OOF join is incomplete")
    if not joined[key].equals(anchor[key].reset_index(drop=True)):
        raise ValueError("OOF join changed frozen anchor identity order")
    return joined


def train_only_empirical_cdf(reference_prediction: np.ndarray, evaluation_prediction: np.ndarray) -> np.ndarray:
    """Rank values against *training-clean* predictions, never eval outcomes."""

    reference = np.sort(np.asarray(reference_prediction, dtype=float))
    evaluation = np.asarray(evaluation_prediction, dtype=float)
    if len(reference) < 2 or not np.isfinite(reference).all() or not np.isfinite(evaluation).all():
        raise ValueError("within-clean CDF needs finite train-only support of at least two rows")
    return np.searchsorted(reference, evaluation, side="right").astype(float) / float(len(reference))


def _fit_direct_context(X: pd.DataFrame, y: np.ndarray, *, seed: int, iterations: int) -> Any:
    # Import lazily: pure contract tests remain independent of CatBoost.
    from scripts.run_meaningful_mfe_catboost_v2_ablation import _fit_catboost
    if int(iterations) < 1:
        raise ValueError("residual iterations must be positive")
    params = dict(RESIDUAL_PARAMS) | {"iterations": int(iterations)}
    return _fit_catboost("quality", X, y, params, seed=seed)


def _predict_direct_context(model: Any, X: pd.DataFrame) -> np.ndarray:
    result = np.asarray(model.predict(X), dtype=float)
    if result.shape != (len(X),) or not np.isfinite(result).all():
        raise ValueError("direct context regressor emitted invalid predictions")
    return result


def _fit_event_classifier(X: pd.DataFrame, y: np.ndarray, *, seed: int) -> Any:
    classes = np.asarray(y, dtype=int)
    if set(np.unique(classes)) != {0, 1, 2}:
        raise ValueError("each OOF classifier fit requires all competing-risk classes")
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(C=0.30, max_iter=400, solver="lbfgs", multi_class="multinomial", random_state=seed),
    )
    model.fit(X, classes)
    return model


def _predict_event_classifier(model: Any, X: pd.DataFrame) -> np.ndarray:
    result = np.asarray(model.predict_proba(X), dtype=float)
    if result.shape != (len(X), 3) or not np.isfinite(result).all() or not np.allclose(result.sum(axis=1), 1.0, atol=1e-6, rtol=0.0):
        raise ValueError("event classifier emitted an invalid competing-risk simplex")
    return result


def _fit_clean_payoff(X: pd.DataFrame, y: np.ndarray) -> Any:
    values = np.asarray(y, dtype=float)
    if len(values) < 20 or not np.isfinite(values).all():
        raise ValueError("conditional clean payoff fit lacks finite clean-only support")
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=10.0)).fit(X, values)


def _predict_clean_payoff(model: Any, X: pd.DataFrame) -> np.ndarray:
    result = np.asarray(model.predict(X), dtype=float)
    if result.shape != (len(X),) or not np.isfinite(result).all():
        raise ValueError("conditional clean payoff regressor emitted invalid predictions")
    return result


def _channel_frame(
    panel: pd.DataFrame,
    meta_X: pd.DataFrame,
    *,
    fit_positions: np.ndarray,
    prediction_positions: np.ndarray,
    seed: int,
) -> pd.DataFrame:
    """Fit channels on one resolved training block and predict another block."""

    fit_positions = np.asarray(fit_positions, dtype=int)
    prediction_positions = np.asarray(prediction_positions, dtype=int)
    classes = panel[CLASS_COLUMN].to_numpy(int)
    clean = fit_positions[classes[fit_positions] == 2]
    if len(clean) < 20:
        raise ValueError("conditional clean payoff needs at least 20 observed clean training rows")
    event = _fit_event_classifier(meta_X.iloc[fit_positions], classes[fit_positions], seed=seed)
    probability = _predict_event_classifier(event, meta_X.iloc[prediction_positions])
    payoff = _fit_clean_payoff(
        meta_X.iloc[clean], panel.iloc[clean]["execution_gross_ev_12h"].to_numpy(float)
    )
    predicted_clean = _predict_clean_payoff(payoff, meta_X.iloc[prediction_positions])
    # The rank reference is generated by the model on the *fit clean rows*,
    # then fixed before any prediction/evaluation rows are touched.
    fit_clean_prediction = _predict_clean_payoff(payoff, meta_X.iloc[clean])
    rank = train_only_empirical_cdf(fit_clean_prediction, predicted_clean)
    return pd.DataFrame(
        {
            "p_timeout": probability[:, 0],
            "p_adverse": probability[:, 1],
            "p_clean": probability[:, 2],
            "pred_clean_gross": predicted_clean,
            "pred_clean_value": probability[:, 2] * predicted_clean,
            "pred_clean_rank": rank,
        }
    )


def _crossfit_context(
    panel: pd.DataFrame,
    meta_X: pd.DataFrame,
    outer_train: np.ndarray,
    *,
    outer_split: str,
    seed: int,
    min_crossfit_train_rows: int,
    min_crossfit_validation_rows: int,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Return strictly chronological OOF competing-risk context predictions."""

    local = panel.iloc[outer_train].reset_index(drop=True)
    rows: list[pd.DataFrame] = []
    reports: list[dict[str, Any]] = []
    for item in _crossfit_plan(
        local,
        min_train_rows=min_crossfit_train_rows,
        min_validation_rows=min_crossfit_validation_rows,
    ):
        train_local = np.asarray(item["train"], dtype=int)
        validation_local = np.asarray(item["validation"], dtype=int)
        # This function is called per side, hence every label-based channel
        # fit is side-local.  The frozen base alpha score is already present
        # in the panel; it is not retrained against the direct-net target.
        channel = _channel_frame(
            panel,
            meta_X,
            fit_positions=outer_train[train_local],
            prediction_positions=outer_train[validation_local],
            seed=seed + 10_000 + int(item["fold"]),
        )
        output = _identity_frame(panel, outer_train[validation_local], f"{outer_split}__train_oof")
        output = pd.concat([output, channel], axis=1)
        rows.append(output)
        reports.append({
            key: value for key, value in item.items() if key not in {"train", "validation"}
        } | {"train_rows": int(len(train_local)), "validation_rows": int(len(validation_local))})
    if not rows:
        raise ValueError("no chronological OOF folds met support requirements")
    result = pd.concat(rows, ignore_index=True)
    if result.duplicated(["outer_split", *IDENTITY]).any():
        raise ValueError("cross-fitted context emitted duplicate exact identities")
    if not np.isfinite(result[list(CONTEXT_CHANNELS)].to_numpy(float)).all():
        raise ValueError("cross-fitted context contains nonfinite model predictions")
    return result, reports


def _direct_residual_prediction(
    panel: pd.DataFrame,
    meta_X: pd.DataFrame,
    candidate_context: pd.DataFrame,
    *,
    outer_train: np.ndarray,
    evaluation: np.ndarray,
    oof: pd.DataFrame,
    outer_split: str,
    channels: Sequence[str],
    candidate_fields: Sequence[str],
    include_alpha: bool,
    seed: int,
    residual_iterations: int,
) -> tuple[np.ndarray, int]:
    # ``oof`` deliberately covers only later chronological validation blocks;
    # using all outer-train rows here would turn early missing OOF scores into
    # either an accidental imputation or a leakage path.  Its own frozen key
    # order is the residual training order.
    anchor = oof[["outer_split", *IDENTITY]].copy()
    joined = safe_oof_join(anchor, oof, value_columns=CONTEXT_CHANNELS)
    # The final direct-EV head sees a frozen OOF alpha score, never an
    # in-sample base target.  Only predicted context channels may enter.
    index_lookup = _identity_frame(panel, outer_train, f"{outer_split}__train_oof")
    index_lookup["__panel_position__"] = np.asarray(outer_train, dtype=int)
    resolved = joined.merge(index_lookup, on=["outer_split", *IDENTITY], how="left", validate="one_to_one", sort=False)
    if resolved["__panel_position__"].isna().any():
        raise ValueError("OOF context identity escaped the declared outer training set")
    positions = resolved["__panel_position__"].to_numpy(int)
    features = build_final_context_matrix(
        meta_X.iloc[positions].reset_index(drop=True),
        panel.iloc[positions][BASE_ALPHA_COLUMN].to_numpy(float),
        joined[list(CONTEXT_CHANNELS)],
        channels=channels,
        include_alpha=include_alpha,
        candidate_context=candidate_context.iloc[positions].reset_index(drop=True),
        candidate_fields=candidate_fields,
    )
    # The residual/context head has the frozen alpha score as a feature, but
    # directly learns exact net EV.  It therefore stays a cost-aware final
    # handoff rather than forcing incompatible alpha and return scales into a
    # subtraction target.
    target = panel.iloc[positions]["execution_net_ev_12h"].to_numpy(float)
    model = _fit_direct_context(features, target, seed=seed + 20_000, iterations=residual_iterations)

    # Evaluation context is generated from full outer-train models only.
    channel_eval = _channel_frame(
        panel, meta_X, fit_positions=outer_train, prediction_positions=evaluation, seed=seed + 30_000
    )
    evaluation_features = build_final_context_matrix(
        meta_X.iloc[evaluation].reset_index(drop=True),
        panel.iloc[evaluation][BASE_ALPHA_COLUMN].to_numpy(float),
        channel_eval,
        channels=channels,
        include_alpha=include_alpha,
        candidate_context=candidate_context.iloc[evaluation].reset_index(drop=True),
        candidate_fields=candidate_fields,
    )
    return _predict_direct_context(model, evaluation_features), int(len(features))


def build_final_context_matrix(
    meta: pd.DataFrame,
    frozen_base_alpha: np.ndarray,
    predicted_channels: pd.DataFrame,
    *,
    channels: Sequence[str],
    include_alpha: bool = True,
    candidate_context: pd.DataFrame | None = None,
    candidate_fields: Sequence[str] = (),
) -> pd.DataFrame:
    """Build final direct-EV inputs without exposing any observed outcome."""

    if set(meta.columns).intersection(OBSERVED_OUTCOME_COLUMNS):
        raise ValueError("final direct context matrix may not contain observed labels/payoffs")
    selected = tuple(map(str, channels))
    if not set(selected).issubset(CONTEXT_CHANNELS):
        raise ValueError("final direct context matrix received an undeclared channel")
    alpha = np.asarray(frozen_base_alpha, dtype=float)
    if len(meta) != len(alpha) or len(meta) != len(predicted_channels) or (include_alpha and not np.isfinite(alpha).all()):
        raise ValueError("final direct context inputs have incompatible frozen-alpha support")
    missing = sorted(set(selected).difference(predicted_channels.columns))
    if missing or (selected and not np.isfinite(predicted_channels[list(selected)].to_numpy(float)).all()):
        raise ValueError("final direct context requires finite predicted channels only")
    result = meta.copy().reset_index(drop=True)
    if include_alpha:
        result["__frozen_base_alpha_oof__"] = alpha
    for channel in selected:
        result[f"__ctx_{channel}__"] = predicted_channels[channel].to_numpy(float)
    fields = tuple(map(str, candidate_fields))
    if not set(fields).issubset(ALLOWED_SIDECAR_MODEL_FIELDS):
        raise ValueError("final direct context matrix received an undeclared or excluded sidecar field")
    forbidden_representation = sorted(set(fields).intersection(FORBIDDEN_REPRESENTATION_FIELDS))
    if forbidden_representation:
        raise ValueError("GMM posterior/compact-risk fields are excluded pending an incremental gate")
    if fields:
        if candidate_context is None or len(candidate_context) != len(result):
            raise ValueError("final direct context candidate fields lack aligned support")
        missing_candidate = sorted(set(fields).difference(candidate_context.columns))
        if missing_candidate:
            raise ValueError("final direct context requires requested sidecar fields")
        representation = tuple(field for field in fields if field in REPRESENTATION_FIELDS)
        if representation:
            validate_sidecar_representation_missingness(candidate_context, representation)
        nonrepresentation = tuple(field for field in fields if field not in REPRESENTATION_FIELDS)
        if nonrepresentation and not np.isfinite(candidate_context[list(nonrepresentation)].to_numpy(float)).all():
            raise ValueError("final direct context requires finite non-representation sidecar fields")
        for field in fields:
            result[f"__candidate_{field}__"] = candidate_context[field].to_numpy(float)
    return result


def evaluate_global_topk(frame: pd.DataFrame, score_column: str, *, evaluation: str) -> list[dict[str, Any]]:
    """One pooled global deterministic top-k evaluation, including month cover."""

    rows: list[dict[str, Any]] = []
    for fraction in TOP_FRACTIONS:
        selected = stable_top(frame, score_column, fraction=fraction)
        gross = selected["execution_gross_ev_12h"].to_numpy(float)
        cost = selected["execution_cost_return"].to_numpy(float)
        net = selected["execution_net_ev_12h"].to_numpy(float)
        if not np.allclose(gross - cost, net, atol=1e-7, rtol=0.0):
            raise ValueError("global top-k context economics violates exact cost identity")
        row: dict[str, Any] = {
            "evaluation": evaluation,
            "score": score_column,
            "selected_fraction": fraction,
            "population_rows": int(len(frame)),
            "selected_rows": int(len(selected)),
            "net_ev_bps": float(net.mean() * 1e4),
            "gross_ev_bps": float(gross.mean() * 1e4),
            "cost_bps": float(cost.mean() * 1e4),
            "positive_net_rate": float((net > 0.0).mean()),
            "long_share": float(selected["side_name"].eq("long").mean()),
            "asset_count": int(selected["__symbol__"].nunique()),
        }
        month = pd.to_datetime(selected["__ts__"], utc=True).dt.strftime("%Y-%m")
        for value in sorted(month.unique()):
            local = selected.loc[month.eq(value), "execution_net_ev_12h"].to_numpy(float)
            row[f"month_{value}_rows"] = int(len(local))
            row[f"month_{value}_net_bps"] = float(local.mean() * 1e4)
        rows.append(row)
    return rows


def score_split(
    panel: pd.DataFrame,
    matrix: pd.DataFrame,
    base_by_side: Mapping[str, Sequence[str]],
    meta_features: Sequence[str],
    candidate_context: pd.DataFrame,
    *,
    train: np.ndarray,
    evaluation: np.ndarray,
    name: str,
    seed: int,
    min_crossfit_train_rows: int,
    min_crossfit_validation_rows: int,
    arm_names: Sequence[str],
    residual_iterations: int,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Score one outer split.  Every fit and context channel is side-local."""

    keep = [*IDENTITY, "execution_gross_ev_12h", "execution_cost_return", "execution_net_ev_12h", CLASS_COLUMN, *CLASS_NAMES]
    scored = panel.iloc[evaluation][keep].copy().reset_index(drop=True)
    architecture: list[dict[str, Any]] = []
    for side_index, side in enumerate(SIDES):
        side_train = train[panel.iloc[train]["side_name"].eq(side).to_numpy()]
        side_eval = evaluation[panel.iloc[evaluation]["side_name"].eq(side).to_numpy()]
        output = np.flatnonzero(scored["side_name"].eq(side).to_numpy())
        if len(side_train) < min_crossfit_train_rows or len(side_eval) < 20:
            raise ValueError(f"{name}/{side} lacks side-local context support")
        base_features = forbid_action_features(base_by_side[side])
        meta = forbid_action_features(meta_features)
        oof, folds = _crossfit_context(
            panel, matrix[meta], side_train, outer_split=name,
            seed=seed + side_index * 100_000,
            min_crossfit_train_rows=min_crossfit_train_rows,
            min_crossfit_validation_rows=min_crossfit_validation_rows,
        )
        # Base-only is the frozen config-routed alpha OOF score.  It remains
        # a benchmark rather than being refit to a different direct-EV target.
        scored.loc[output, "score_base_only"] = panel.iloc[side_eval][BASE_ALPHA_COLUMN].to_numpy(float)
        for arm in arm_names:
            channels = ARMS[arm]
            if arm == "base_only":
                continue
            prediction, residual_rows = _direct_residual_prediction(
                panel, matrix[meta], candidate_context, outer_train=side_train, evaluation=side_eval,
                oof=oof, outer_split=name, channels=channels, candidate_fields=ARM_CANDIDATE_CONTEXT[arm], include_alpha=ARM_INCLUDE_ALPHA[arm],
                seed=seed + side_index * 100_000 + 10_000 + list(ARMS).index(arm), residual_iterations=residual_iterations,
            )
            scored.loc[output, f"score_{arm}"] = prediction
            architecture.append({
                "evaluation": name, "side": side, "arm": arm,
                "base_feature_count": len(base_features), "meta_feature_count": len(meta),
                "context_channels": json.dumps(list(channels)),
                "includes_frozen_base_alpha": ARM_INCLUDE_ALPHA[arm],
                "sidecar_fields": json.dumps(list(ARM_CANDIDATE_CONTEXT[arm])),
                "oof_rows": int(len(oof)), "residual_train_rows": residual_rows,
                "crossfit_folds": json.dumps(_safe(folds)),
                "final_head": "side_local_direct_net_head_on_config_meta_and_predicted_context_only",
            })
    scored["evaluation"] = name
    return scored, architecture


def _split_rows(scored: pd.DataFrame, name: str, arm_names: Sequence[str]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for arm in arm_names:
        result.extend(evaluate_global_topk(scored, f"score_{arm}", evaluation=name))
    return result


def begin_atomic_output(target: Path) -> Path:
    """Refuse overwrite and write a complete diagnostic atomically."""

    target = Path(target)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite existing context artifact: {target}")
    partial = target.with_name(f".{target.name}.partial-{os.getpid()}")
    if partial.exists():
        raise FileExistsError(f"existing partial context artifact requires explicit inspection: {partial}")
    partial.mkdir(parents=True, exist_ok=False)
    return partial


def resolve_runtime_controls(
    requested_arms: Sequence[str] | None,
    requested_evaluations: Sequence[str] | None,
    *,
    available_evaluations: Sequence[str],
    grouped_july_folds: Sequence[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Validate bounded rerun controls without changing default experiment scope."""

    arms = tuple(ARMS) if not requested_arms else tuple(dict.fromkeys(map(str, requested_arms)))
    unknown_arms = sorted(set(arms).difference(ARMS))
    if unknown_arms:
        raise ValueError(f"unknown context arms: {unknown_arms}")
    evaluations = (
        tuple(available_evaluations)
        if not requested_evaluations
        else tuple(dict.fromkeys(map(str, requested_evaluations)))
    )
    unknown_evaluations = sorted(set(evaluations).difference(available_evaluations))
    if unknown_evaluations:
        raise ValueError(f"unknown context evaluations: {unknown_evaluations}")
    if "july_grouped_oof" in evaluations and not set(grouped_july_folds).issubset(evaluations):
        raise ValueError("july_grouped_oof requires every constituent grouped-day July fold")
    return arms, evaluations


def run(args: argparse.Namespace) -> dict[str, Any]:
    final_output_dir = Path(args.output_dir)
    if int(args.residual_iterations) < 1:
        raise ValueError("--residual-iterations must be positive")
    arm_names = tuple(ARMS) if not args.arms else tuple(dict.fromkeys(map(str, args.arms)))
    unknown_arms = sorted(set(arm_names).difference(ARMS))
    if unknown_arms:
        raise ValueError(f"unknown context arms: {unknown_arms}")
    panel, matrix, raw_features, lineage = load_competing_risk_panel(
        args.features, args.feature_manifest, args.grid, args.grid_manifest,
        args.label_dir, label_kind="primary_floor", buffer_bps=100,
    )
    validate_primary100_contract(panel)
    base_by_side, meta, feature_contract = config_routed_feature_pools(matrix.columns)
    forbid_action_features([*base_by_side["long"], *base_by_side["short"], *meta])
    sidecar_source = validate_context_sidecar_source(args.context_sidecar)
    candidate_context = load_exact_context_sidecar(panel, args.context_sidecar)
    lineage["context_sidecar"] = sidecar_source
    july_folds = list(july_grouped_day_folds(panel))
    grouped_fold_names = tuple(row[0] for row in july_folds)
    available_evaluations = (
        "may_to_june", "june_to_july", "july_to_june_matched", *grouped_fold_names, "july_grouped_oof",
    )
    arm_names, evaluation_names = resolve_runtime_controls(
        args.arms, args.evaluations,
        available_evaluations=available_evaluations,
        grouped_july_folds=grouped_fold_names,
    )
    output_dir = begin_atomic_output(final_output_dir)
    predictions: list[pd.DataFrame] = []
    architecture: list[dict[str, Any]] = []
    economics: list[dict[str, Any]] = []
    splits: list[dict[str, Any]] = []
    for index, spec in enumerate(TRANSFER_SPECS):
        if spec.name not in evaluation_names or spec.name not in {"may_to_june", "june_to_july", "july_to_june_matched"}:
            continue
        train, evaluation = _base_masks(panel, spec)
        scored, detail = score_split(panel, matrix, base_by_side, meta, candidate_context, train=train, evaluation=evaluation, name=spec.name, seed=args.seed + index * 1_000_000, min_crossfit_train_rows=args.min_crossfit_train_rows, min_crossfit_validation_rows=args.min_crossfit_validation_rows, arm_names=arm_names, residual_iterations=args.residual_iterations)
        predictions.append(scored); architecture.extend(detail); economics.extend(_split_rows(scored, spec.name, arm_names))
        splits.append({"name": spec.name, "train_rows": len(train), "evaluation_rows": len(evaluation), "promotion_eligible": False, "note": spec.note})
    july_parts: list[pd.DataFrame] = []
    for index, (name, train, evaluation, days) in enumerate(july_folds):
        if name not in evaluation_names:
            continue
        scored, detail = score_split(panel, matrix, base_by_side, meta, candidate_context, train=train, evaluation=evaluation, name=name, seed=args.seed + 10_000_000 + index * 1_000_000, min_crossfit_train_rows=args.min_crossfit_train_rows, min_crossfit_validation_rows=args.min_crossfit_validation_rows, arm_names=arm_names, residual_iterations=args.residual_iterations)
        july_parts.append(scored); architecture.extend(detail); economics.extend(_split_rows(scored, name, arm_names))
        splits.append({"name": name, "train_rows": len(train), "evaluation_rows": len(evaluation), "validation_days": list(days), "promotion_eligible": False})
    if "july_grouped_oof" in evaluation_names:
        grouped = pd.concat(july_parts, ignore_index=True)
        if grouped.duplicated(list(IDENTITY)).any():
            raise ValueError("grouped-July context OOF contains duplicate identities")
        grouped["evaluation"] = "july_grouped_oof"
        predictions.append(grouped); economics.extend(_split_rows(grouped, "july_grouped_oof", arm_names))
    if not predictions:
        raise ValueError("requested evaluation subset selected no evaluation output")
    outputs: dict[str, Any] = {}
    for name, data in (("predictions", pd.concat(predictions, ignore_index=True)), ("architecture", pd.DataFrame(architecture)), ("economics", pd.DataFrame(economics))):
        path = output_dir / f"{name}.parquet"; data.to_parquet(path, index=False)
        outputs[name] = {"path": final_output_dir / f"{name}.parquet", "rows": len(data), "sha256": _sha256(path)}
    report = {
        "schema": SCHEMA,
        "status": "COMPLETED_DIAGNOSTIC_CONFIG_ROUTED_CONTEXT_NO_PROMOTION",
        "promotion_eligible": False,
        "runner": {"path": Path(__file__).resolve(), "sha256": _sha256(Path(__file__).resolve())},
        "lineage": lineage,
        "raw_feature_count": len(raw_features),
        "feature_contract": feature_contract,
        "context_sidecar_source": sidecar_source,
        "runtime_controls": {"arms": list(arm_names), "evaluations": list(evaluation_names), "residual_iterations": int(args.residual_iterations)},
        "splits": splits,
        "contracts": {
            "panel": "exact 134,889 Primary100 identities; exact 12h decision-to-resolution; gross-cost=net",
            "architecture": "frozen config-routed alpha OOF score -> side-local direct-net head on config meta fields; this is direct net prediction, not a residual-subtraction target",
            "context": "OOF clean/adverse/timeout probabilities; conditional clean gross payoff trained only on clean rows; train-clean-prediction CDF payoff rank; final head consumes predictions only",
            "action_exclusion": "timing, MAE, wait, target-price and related action fields are rejected before fitting",
            "economics": "one pooled global deterministic top 1/5/10/20% ranking; per-month coverage persisted; no side/timestamp quota",
        },
        "geometry": "fixed, bounded diagnostic geometry only: multinomial logistic event probabilities, clean-only Ridge payoff, and frozen CatBoost residual params; no HPO claim or promotion eligibility",
        "arms": {name: {"predicted_channels": list(ARMS[name]), "sidecar_fields": list(ARM_CANDIDATE_CONTEXT[name]), "includes_frozen_base_alpha": ARM_INCLUDE_ALPHA[name]} for name in arm_names},
        "outputs": outputs,
    }
    report_path = output_dir / "report.json"; _write_json(report_path, report)
    _write_json(output_dir / "manifest.json", {"schema": SCHEMA, "status": report["status"], "promotion_eligible": False, "report": {"path": final_output_dir / "report.json", "sha256": _sha256(report_path)}, "inputs": lineage, "outputs": outputs})
    os.replace(output_dir, final_output_dir)
    return report


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--features", type=Path, default=ROOT / "data_perp/artifacts/exact_policy_capture_feature_universe_20260727_v2/capture_feature_universe.parquet")
    value.add_argument("--feature-manifest", type=Path, default=ROOT / "data_perp/artifacts/exact_policy_capture_feature_universe_20260727_v2/manifest.json")
    value.add_argument("--grid", type=Path, default=ROOT / "data_perp/artifacts/meaningful_mfe_exact_policy_label_grid_20260727_v1/meaningful_mfe_label_grid.parquet")
    value.add_argument("--grid-manifest", type=Path, default=ROOT / "data_perp/artifacts/meaningful_mfe_exact_policy_label_grid_20260727_v1/manifest.json")
    value.add_argument("--label-dir", type=Path, default=PRIMARY_DIR)
    value.add_argument("--context-sidecar", type=Path, default=DEFAULT_CONTEXT_SIDECAR)
    value.add_argument("--output-dir", type=Path, required=True)
    value.add_argument("--seed", type=int, default=20260730)
    value.add_argument("--min-crossfit-train-rows", type=int, default=2_000)
    value.add_argument("--min-crossfit-validation-rows", type=int, default=500)
    value.add_argument("--arms", nargs="*", default=None, choices=tuple(ARMS))
    value.add_argument("--evaluations", nargs="*", default=None)
    value.add_argument("--residual-iterations", type=int, default=int(RESIDUAL_PARAMS["iterations"]))
    return value


if __name__ == "__main__":
    arguments = parser().parse_args()
    print(json.dumps(run(arguments), indent=2, default=str))
