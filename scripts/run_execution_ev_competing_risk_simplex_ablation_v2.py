#!/usr/bin/env python3
"""Bounded exact 1m competing-risk plus conditional-payoff ablation.

The sole decision target is the exact deployed 12-hour net outcome.  The
classifier learns the mutually-exclusive full-path first-touch simplex
``timeout/adverse-first/clean-economic-favourable-first``; separate,
side-local regressors estimate deployed *gross* execution payoff inside each
observed class.  The composed score is consequently

    sum(class_probability * conditional_gross_payoff) - row_execution_cost

exactly once.  This is a standalone PIT-simplex diagnostic, not a
config-routed base->residual/context architecture or a policy search. Geometry
is chosen only on the purged May inner split, and the reported forward, reverse
and July OOF evaluations use frozen geometry. The frozen base score is retained
only for the IC-to-EV bridge; it is never a feature of this runner.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import HuberRegressor, LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_july_exact_preentry_heads import IDENTITY
from scripts.run_historical_to_july_meaningful_mfe_gate_challenger import (
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


SCHEMA = "execution_ev_competing_risk_simplex_ablation_v2"
CLASS_NAMES = ("timeout", "adverse_first", "clean_economic_favorable_first")
CLASS_COLUMN = "competing_risk_class"
FAMILIES = ("logistic", "lightgbm", "catboost")
FEATURE_COUNT = 48
PRIMARY_BUFFERS = (0, 100)
NOFLOOR_BUFFERS = (0, 25, 50, 100)
PRIMARY_DIR = ROOT / "data_perp/artifacts/execution_ev_cost_aware_competing_risk_1m_labels_20260730_v1"
NOFLOOR_DIR = ROOT / "data_perp/artifacts/execution_ev_cost_aware_competing_risk_1m_labels_nofloor_20260730_v1"
LABEL_FILE = "execution_ev_cost_aware_competing_risk_labels.parquet"
LABEL_SCHEMA = "execution_ev_cost_aware_competing_risk_1m_labels_v1"
HORIZON = pd.Timedelta(hours=12)
CALIBRATION_MIN_ROWS = 100
CALIBRATION_MIN_CLASS_MASS = 20.0
OFFSET_TEMPERATURE_L2 = 1e-3
# The no-floor 50-bps grouped-July timeout class has a predeclared minimum of
# 212 train rows (fold 4 short) across the complete five-fold matrix.  A 200
# row gate therefore keeps every fold, leaves a 12-row safety margin, and
# remains materially above the 100-row calibration/evaluation floor.  V1's
# arbitrary 250-row gate excluded this otherwise valid predeclared ablation.
MIN_CONDITIONAL_PAYOFF_ROWS = 200

# This is deliberately much smaller than an open HPO surface.  Every family
# receives two comparable geometries and selection is frozen by the May-only
# inner split.  The values are architecture, not trading-policy, parameters.
SIMPLEX_GRIDS: Mapping[str, tuple[Mapping[str, Any], ...]] = {
    "logistic": ({"C": 0.10}, {"C": 1.0}),
    "lightgbm": tuple(MODEL_GRIDS["lightgbm"]),
    "catboost": tuple(MODEL_GRIDS["catboost"]),
}

# Conditional payoff models solve a different problem from the simplex
# classifier.  Keep their two bounded alternatives genuinely independent from
# classifier capacity, including robust-linear Huber geometry.
PAYOFF_GRIDS: Mapping[str, tuple[Mapping[str, Any], ...]] = {
    "logistic": (
        {"huber_epsilon": 1.20, "huber_alpha": 1e-4},
        {"huber_epsilon": 1.75, "huber_alpha": 1e-3},
    ),
    "lightgbm": (
        {
            "num_leaves": 15,
            "max_depth": 5,
            "min_child_samples": 250,
            "reg_lambda": 8.0,
        },
        {
            "num_leaves": 31,
            "max_depth": 7,
            "min_child_samples": 150,
            "reg_lambda": 12.0,
        },
    ),
    "catboost": (
        {"depth": 5, "l2_leaf_reg": 8.0},
        {"depth": 7, "l2_leaf_reg": 12.0},
    ),
}


def _safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
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
    temporary.write_text(json.dumps(_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _resolved_path(value: str | Path, *, relative_to: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (relative_to / path).resolve()


def geometry_contract(label_kind: str, buffer_bps: int) -> dict[str, Any]:
    """Declare, rather than silently skip, inert primary-floor geometries."""

    if label_kind == "primary_floor":
        if buffer_bps == 25:
            return {"included": False, "reason": "exact_duplicate_of_primary_floor_0bps"}
        if buffer_bps == 50:
            return {"included": False, "reason": "only_42_of_156202_hard_labels_change_vs_primary_floor_0bps"}
        if buffer_bps in PRIMARY_BUFFERS:
            return {"included": True, "reason": "predeclared_primary_floor_geometry"}
    if label_kind == "nofloor" and buffer_bps in NOFLOOR_BUFFERS:
        return {"included": True, "reason": "predeclared_nofloor_geometry"}
    raise ValueError(f"unsupported competing-risk geometry: {label_kind}/{buffer_bps}bps")


def _validate_label_artifact(label_dir: Path, *, expect_floor: bool) -> tuple[Path, Path, dict[str, Any]]:
    manifest_path = label_dir / "manifest.json"
    labels_path = label_dir / LABEL_FILE
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != LABEL_SCHEMA:
        raise ValueError("unexpected competing-risk label schema")
    if manifest.get("status") != "completed_exact_1m_target_only_not_model_evidence":
        raise ValueError("competing-risk labels have unexpected status")
    if bool(manifest.get("event_contract", {}).get("upper_return_floor_included")) != expect_floor:
        raise ValueError("competing-risk label floor contract disagrees with requested geometry")
    if int(manifest.get("coverage", {}).get("complete_rows", -1)) != int(manifest.get("coverage", {}).get("rows", -2)) or not math.isclose(float(manifest.get("coverage", {}).get("rate", 0.0)), 1.0):
        raise ValueError("competing-risk label artifact lacks 100% exact 1m coverage")
    output = manifest.get("outputs", {}).get("labels", {})
    if output.get("sha256") != sha256(labels_path):
        raise ValueError("competing-risk labels hash is not bound by its manifest")
    bound = _resolved_path(str(output.get("path", "")), relative_to=ROOT)
    if bound != labels_path.resolve():
        raise ValueError("competing-risk manifest labels path disagrees with requested input")
    if manifest.get("event_contract", {}).get("label_resolution") != "execution_decision_utc + 720m":
        raise ValueError("competing-risk label manifest does not bind 12h resolution")
    runner = manifest.get("runner", {})
    runner_path = Path(str(runner.get("path", "")))
    if not runner_path.is_file() or runner.get("sha256") != sha256(runner_path):
        raise ValueError("competing-risk label manifest does not bind its materializer runner")
    return labels_path, manifest_path, manifest


def _require_identity(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    missing = sorted(set(IDENTITY).difference(frame.columns))
    if missing:
        raise ValueError(f"{name} lacks exact identity: {missing}")
    work = frame.copy()
    work["__ts__"] = pd.to_datetime(work["__ts__"], utc=True, errors="raise")
    work["side_name"] = work["side_name"].astype(str).str.lower()
    if not work["side_name"].isin(SIDES).all() or work.duplicated(list(IDENTITY)).any():
        raise ValueError(f"{name} has noncanonical or duplicate identities")
    return work


def load_competing_risk_panel(
    feature_path: Path,
    feature_manifest: Path,
    grid_path: Path,
    grid_manifest: Path,
    label_dir: Path,
    *,
    label_kind: str,
    buffer_bps: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict[str, Any]]:
    """Join exact PIT features, reset anchors and one economics-label geometry."""

    contract = geometry_contract(label_kind, int(buffer_bps))
    if not contract["included"]:
        raise ValueError(f"redundant geometry is explicitly excluded: {contract['reason']}")
    labels_path, manifest_path, label_manifest = _validate_label_artifact(
        label_dir, expect_floor=label_kind == "primary_floor"
    )
    panel, matrix, raw_features, lineage = load_panel(
        feature_path, feature_manifest, grid_path, grid_manifest
    )
    # The materializer deliberately contains one row per exact identity *per
    # buffer*.  Canonicalise before filtering, then require uniqueness on the
    # selected geometry rather than falsely rejecting the audited source grid.
    labels = pd.read_parquet(labels_path)
    if "cost_buffer_bps" not in labels.columns:
        raise ValueError("competing-risk labels lack cost_buffer_bps")
    labels["__ts__"] = pd.to_datetime(labels["__ts__"], utc=True, errors="raise")
    labels["side_name"] = labels["side_name"].astype(str).str.lower()
    if not labels["side_name"].isin(SIDES).all():
        raise ValueError("competing-risk labels have noncanonical sides")
    labels = _require_identity(
        labels.loc[labels["cost_buffer_bps"].eq(int(buffer_bps))].copy(),
        "selected competing-risk buffer",
    )
    panel_ids = set(map(tuple, panel[list(IDENTITY)].itertuples(False, None)))
    label_ids = set(map(tuple, labels[list(IDENTITY)].itertuples(False, None)))
    if not panel_ids.issubset(label_ids):
        raise ValueError(f"selected competing-risk buffer lacks exact feature-panel coverage: {len(panel_ids - label_ids)} missing")
    # Label artifacts intentionally extend beyond the frozen feature panel;
    # retain precisely the common decision universe rather than conflating
    # harmless extra materialized labels with missing model labels.
    labels = labels.merge(panel[list(IDENTITY)], on=list(IDENTITY), how="inner", validate="one_to_one")
    if labels.duplicated(list(IDENTITY)).any():
        raise ValueError("selected competing-risk buffer has duplicate identities")
    required = [
        *IDENTITY, "execution_decision_utc", "execution_label_end_utc", "label_resolution_utc",
        "execution_gross_ev_12h", "execution_cost_return", "execution_net_ev_12h",
        CLASS_COLUMN, *CLASS_NAMES,
        "first_favorable_minute", "first_adverse_minute",
        "timeout_soft_clean_economic_favorable_viability",
        "timeout_soft_adverse_viability", "timeout_soft_timeout_viability",
    ]
    missing = sorted(set(required).difference(labels.columns))
    if missing:
        raise ValueError(f"competing-risk labels lack required fields: {missing}")
    # The reset panel carries the same policy anchors and some older barrier
    # labels.  Prove the immutable policy anchors agree, then deliberately
    # replace the old barrier classes with the competing-risk labels rather
    # than accidentally retaining a same-named previous target.
    anchors = ("execution_decision_utc", "execution_label_end_utc", "label_resolution_utc", "execution_gross_ev_12h", "execution_cost_return", "execution_net_ev_12h")
    for anchor in anchors:
        if anchor in panel.columns:
            left = panel[[*IDENTITY, anchor]].merge(labels[[*IDENTITY, anchor]], on=list(IDENTITY), suffixes=("__panel", "__cr"), validate="one_to_one")
            if anchor.endswith("_utc"):
                same = pd.to_datetime(left[f"{anchor}__panel"], utc=True, errors="raise").eq(pd.to_datetime(left[f"{anchor}__cr"], utc=True, errors="raise"))
            else:
                same = np.isclose(left[f"{anchor}__panel"].to_numpy(float), left[f"{anchor}__cr"].to_numpy(float), atol=1e-7, rtol=0.0)
            if not bool(np.all(same)):
                raise ValueError(f"reset-panel and competing-risk exact anchor differ: {anchor}")
    panel = panel.drop(columns=[name for name in required if name not in IDENTITY and name in panel.columns])
    joined = panel.merge(labels[required], on=list(IDENTITY), how="inner", validate="one_to_one")
    if len(joined) != len(panel):
        raise ValueError("exact PIT/competing-risk identity join is incomplete")
    if list(map(tuple, joined[list(IDENTITY)].itertuples(False, None))) != list(map(tuple, panel[list(IDENTITY)].itertuples(False, None))):
        raise ValueError("competing-risk join changed frozen feature-panel identity order")
    for column in ("execution_decision_utc", "execution_label_end_utc", "label_resolution_utc"):
        joined[column] = pd.to_datetime(joined[column], utc=True, errors="raise")
    expected_end = joined["execution_decision_utc"] + HORIZON
    if not joined["execution_label_end_utc"].eq(expected_end).all() or not joined["label_resolution_utc"].eq(expected_end).all():
        raise ValueError("competing-risk exact resolution is not decision + 12h")
    for name in ("execution_gross_ev_12h", "execution_cost_return", "execution_net_ev_12h"):
        if not np.isfinite(pd.to_numeric(joined[name], errors="raise").to_numpy(float)).all():
            raise ValueError(f"competing-risk {name} must be finite")
    if not np.allclose(
        joined["execution_gross_ev_12h"].to_numpy(float) - joined["execution_cost_return"].to_numpy(float),
        joined["execution_net_ev_12h"].to_numpy(float), atol=1e-7, rtol=0.0,
    ):
        raise ValueError("competing-risk gross-cost=net identity failed")
    hard = joined[list(CLASS_NAMES)].to_numpy(int)
    classes = joined[CLASS_COLUMN].to_numpy(int)
    if not np.array_equal(hard.sum(axis=1), np.ones(len(joined), dtype=int)) or not np.array_equal(classes, hard.argmax(axis=1)):
        raise ValueError("competing-risk hard simplex is not exhaustive and mutually exclusive")
    timeout = joined["timeout"].to_numpy(int).astype(bool)
    soft = joined[[
        "timeout_soft_timeout_viability", "timeout_soft_adverse_viability", "timeout_soft_clean_economic_favorable_viability",
    ]].to_numpy(float)
    if timeout.any() and (not np.isfinite(soft[timeout]).all() or not np.allclose(soft[timeout].sum(axis=1), 1.0, atol=1e-6, rtol=0.0)):
        raise ValueError("timeout soft viability does not close to a simplex")
    if (~timeout).any() and np.isfinite(soft[~timeout]).any():
        raise ValueError("observed hits may not receive a soft timeout target")
    lineage["competing_risk_labels"] = {
        "kind": label_kind, "buffer_bps": int(buffer_bps), "path": labels_path,
        "sha256": sha256(labels_path), "manifest": manifest_path,
        "manifest_sha256": sha256(manifest_path), "runner": label_manifest.get("runner"),
        "coverage": label_manifest.get("coverage"), "geometry_contract": contract,
    }
    return joined.reset_index(drop=True), matrix.reset_index(drop=True), raw_features, lineage


def soft_simplex_targets(frame: pd.DataFrame) -> np.ndarray:
    """One-hot observed hits; terminal viability only for genuinely timed-out rows."""

    hard = frame[list(CLASS_NAMES)].to_numpy(float)
    result = hard.copy()
    timeout = frame["timeout"].to_numpy(int).astype(bool)
    if timeout.any():
        result[timeout] = frame[[
            "timeout_soft_timeout_viability", "timeout_soft_adverse_viability", "timeout_soft_clean_economic_favorable_viability",
        ]].to_numpy(float)[timeout]
    if not np.isfinite(result).all() or not np.allclose(result.sum(axis=1), 1.0, atol=1e-6, rtol=0.0):
        raise ValueError("soft simplex targets must be finite and close to one")
    return result


def temperature_scale_probabilities(probability: np.ndarray, temperature: float) -> np.ndarray:
    values = np.asarray(probability, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all() or (values <= 0.0).any():
        raise ValueError("temperature scaling requires finite positive three-class probabilities")
    if not np.allclose(values.sum(axis=1), 1.0, atol=1e-6, rtol=0.0) or not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature scaling needs closed probabilities and positive temperature")
    logits = np.log(np.clip(values, 1e-12, 1.0)) / float(temperature)
    logits -= logits.max(axis=1, keepdims=True)
    result = np.exp(logits)
    return result / result.sum(axis=1, keepdims=True)


def _simplex_target_matrix(target: np.ndarray) -> np.ndarray:
    """Canonical hard or fractional three-class targets for proper scoring."""

    values = np.asarray(target)
    if values.ndim == 1:
        hard = values.astype(int)
        if not np.isin(hard, (0, 1, 2)).all():
            raise ValueError("hard simplex calibration targets must be class codes 0..2")
        return np.eye(3, dtype=float)[hard]
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("simplex calibration targets must be N hard codes or N x 3 weights")
    result = values.astype(float)
    if not np.isfinite(result).all() or (result < 0.0).any() or not np.allclose(result.sum(axis=1), 1.0, atol=1e-6, rtol=0.0):
        raise ValueError("fractional simplex calibration targets must be finite nonnegative closed weights")
    return result


def _simplex_cross_entropy(target: np.ndarray, probability: np.ndarray) -> float:
    y = _simplex_target_matrix(target)
    p = np.asarray(probability, dtype=float)
    if p.shape != y.shape:
        raise ValueError("simplex scoring target/probability shapes disagree")
    if not np.isfinite(p).all() or (p <= 0.0).any() or not np.allclose(p.sum(axis=1), 1.0, atol=1e-6, rtol=0.0):
        raise ValueError("simplex scoring probabilities must be finite positive and closed")
    return float(-np.mean(np.sum(y * np.log(np.clip(p, 1e-12, 1.0)), axis=1)))


def _calibration_support(target: np.ndarray) -> tuple[bool, str, np.ndarray]:
    y = _simplex_target_matrix(target)
    masses = y.sum(axis=0)
    if len(y) < CALIBRATION_MIN_ROWS:
        return False, f"insufficient_rows_{len(y)}_lt_{CALIBRATION_MIN_ROWS}", masses
    if (masses < CALIBRATION_MIN_CLASS_MASS).any():
        return False, "insufficient_effective_class_mass", masses
    return True, "ok", masses


def fit_temperature(probability: np.ndarray, target: np.ndarray) -> float:
    supported, _, _ = _calibration_support(target)
    if not supported:
        return 1.0
    candidates = np.linspace(0.50, 3.00, 51)
    scores = [_simplex_cross_entropy(target, temperature_scale_probabilities(probability, float(t))) for t in candidates]
    return float(candidates[int(np.argmin(scores))])


def fit_offset_temperature_calibrator(
    probability: np.ndarray,
    target: np.ndarray,
    *,
    l2: float = OFFSET_TEMPERATURE_L2,
) -> dict[str, Any]:
    """Fit a small, train-only multiclass prior/temperature correction.

    A single temperature corrects only common over/under-confidence.  The two
    anchored offsets below additionally correct class-prior/intercept drift,
    while deliberately avoiding an over-parameterized 3x3 vector scaler.
    """

    p = np.asarray(probability, dtype=float)
    y = _simplex_target_matrix(target)
    if p.shape != y.shape or p.ndim != 2 or p.shape[1] != 3:
        raise ValueError("offset-temperature calibration requires matching N x 3 inputs")
    if not math.isfinite(float(l2)) or l2 < 0.0:
        raise ValueError("offset-temperature calibration l2 must be finite and nonnegative")
    supported, reason, masses = _calibration_support(y)
    base: dict[str, Any] = {
        "available": bool(supported),
        "reason": reason,
        "rows": int(len(y)),
        "effective_class_mass": [float(value) for value in masses],
        "temperature": 1.0,
        "offsets": [0.0, 0.0, 0.0],
        "l2": float(l2),
        "objective": float("nan"),
    }
    if not supported:
        return base
    if not np.isfinite(p).all() or (p <= 0.0).any() or not np.allclose(p.sum(axis=1), 1.0, atol=1e-6, rtol=0.0):
        raise ValueError("offset-temperature calibration needs finite positive closed probabilities")
    log_probability = np.log(np.clip(p, 1e-12, 1.0))

    def objective(theta: np.ndarray) -> float:
        log_temperature, offset0, offset1 = map(float, theta)
        logits = log_probability / math.exp(log_temperature)
        logits = logits + np.array([offset0, offset1, 0.0], dtype=float)
        logits = logits - logits.max(axis=1, keepdims=True)
        scaled = np.exp(logits)
        scaled /= scaled.sum(axis=1, keepdims=True)
        return _simplex_cross_entropy(y, scaled) + float(l2) * float(np.dot(theta, theta))

    fitted = minimize(
        objective,
        x0=np.zeros(3, dtype=float),
        method="L-BFGS-B",
        bounds=[(math.log(0.50), math.log(3.00)), (-4.0, 4.0), (-4.0, 4.0)],
        options={"maxiter": 200, "ftol": 1e-12},
    )
    if not fitted.success or not np.isfinite(fitted.x).all() or not math.isfinite(float(fitted.fun)):
        base["available"] = False
        base["reason"] = f"optimizer_failure_{str(fitted.message).replace(' ', '_')}"
        return base
    temperature = float(math.exp(float(fitted.x[0])))
    offsets = [float(fitted.x[1]), float(fitted.x[2]), 0.0]
    calibrated = offset_temperature_scale_probabilities(p, temperature, offsets)
    base.update(
        {
            "available": True,
            "reason": "ok",
            "temperature": temperature,
            "offsets": offsets,
            "objective": float(_simplex_cross_entropy(y, calibrated)),
        }
    )
    return base


def offset_temperature_scale_probabilities(
    probability: np.ndarray,
    temperature: float,
    offsets: Sequence[float],
) -> np.ndarray:
    """Apply the anchored three-class offset plus common-temperature map."""

    values = np.asarray(probability, dtype=float)
    adjustment = np.asarray(offsets, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3 or adjustment.shape != (3,):
        raise ValueError("offset-temperature scaling requires N x 3 probabilities and three offsets")
    if not np.isfinite(adjustment).all() or not np.isfinite(values).all() or (values <= 0.0).any():
        raise ValueError("offset-temperature scaling requires finite positive values")
    if not np.allclose(values.sum(axis=1), 1.0, atol=1e-6, rtol=0.0) or not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("offset-temperature scaling needs closed probabilities and positive temperature")
    logits = np.log(np.clip(values, 1e-12, 1.0)) / float(temperature)
    logits = logits + adjustment[None, :]
    logits -= logits.max(axis=1, keepdims=True)
    result = np.exp(logits)
    return result / result.sum(axis=1, keepdims=True)


def apply_offset_temperature_calibrator(probability: np.ndarray, calibrator: Mapping[str, Any]) -> np.ndarray:
    """Fail closed to raw probabilities while retaining a machine-readable reason."""

    values = np.asarray(probability, dtype=float)
    if not bool(calibrator.get("available", False)):
        # Validate even the fallback so a missing calibration fit cannot turn
        # an invalid model simplex into a silently scored utility value.
        return temperature_scale_probabilities(values, 1.0)
    return offset_temperature_scale_probabilities(
        values,
        float(calibrator["temperature"]),
        calibrator["offsets"],
    )


def select_simplex_features(matrix: pd.DataFrame, target: np.ndarray, positions: np.ndarray, count: int) -> tuple[list[str], pd.DataFrame]:
    """Train-only stable one-vs-rest screen; class codes are never ordinal."""

    train = matrix.iloc[positions]
    y = np.asarray(target, dtype=int)[positions]
    midpoint = len(train) // 2
    rows: list[dict[str, Any]] = []
    for feature in matrix.columns:
        values = pd.to_numeric(train[feature], errors="coerce")
        coverage = float(values.notna().mean())
        variance = float(values.var()) if coverage else 0.0
        class_scores: list[float] = []
        for class_index in range(3):
            binary = (y == class_index).astype(float)
            def _ic(part: pd.Series, outcome: np.ndarray) -> float:
                good = np.isfinite(part.to_numpy(float)) & np.isfinite(outcome)
                if int(good.sum()) < 100 or np.unique(part.to_numpy(float)[good]).size < 2 or np.unique(outcome[good]).size < 2:
                    return 0.0
                value = spearmanr(part.to_numpy(float)[good], outcome[good]).statistic
                return float(value) if np.isfinite(value) else 0.0
            early = _ic(values.iloc[:midpoint], binary[:midpoint])
            late = _ic(values.iloc[midpoint:], binary[midpoint:])
            full = _ic(values, binary)
            class_scores.append(min(abs(early), abs(late)) if early * late > 0.0 else .10 * abs(full))
        rows.append({"feature": feature, "coverage": coverage, "variance": variance, "one_vs_rest_stable_score": float(max(class_scores)), "timeout_stable_score": class_scores[0], "adverse_stable_score": class_scores[1], "clean_stable_score": class_scores[2]})
    screen = pd.DataFrame(rows).loc[lambda data: data["coverage"].ge(.99) & data["variance"].gt(1e-12)].sort_values(["one_vs_rest_stable_score", "feature"], ascending=[False, True], kind="stable")
    selected: list[str] = []
    for feature in screen["feature"]:
        if len(selected) >= int(count):
            break
        candidate = pd.to_numeric(train[feature], errors="coerce")
        if any(abs(candidate.corr(pd.to_numeric(train[prior], errors="coerce"))) >= .95 for prior in selected):
            continue
        selected.append(str(feature))
    if len(selected) < min(8, count):
        raise ValueError("one-vs-rest simplex feature selector returned insufficient features")
    screen["selected"] = screen["feature"].isin(selected)
    return selected, screen


def winsor_bounds(target: np.ndarray) -> tuple[float, float]:
    values = np.asarray(target, dtype=float)
    if len(values) < 100 or not np.isfinite(values).all():
        raise ValueError("winsorization needs finite train-only target support")
    low, high = np.quantile(values, [.01, .99])
    if not np.isfinite(low) or not np.isfinite(high) or low > high:
        raise ValueError("invalid train-only payoff winsorization bounds")
    return float(low), float(high)


def winsorize_train_target(target: np.ndarray, bounds: tuple[float, float]) -> tuple[np.ndarray, float]:
    values = np.asarray(target, dtype=float); clipped = np.clip(values, *bounds)
    return clipped, float(np.mean(~np.isclose(values, clipped, atol=0.0, rtol=0.0)))


def conditional_class_positions(panel: pd.DataFrame, positions: np.ndarray, class_index: int) -> np.ndarray:
    """Return only observed rows of one hard class for conditional payoff fit."""

    if class_index not in (0, 1, 2):
        raise ValueError("conditional payoff class must be one of the hard simplex classes")
    source = np.asarray(positions, dtype=int)
    result = source[panel.iloc[source][CLASS_COLUMN].eq(class_index).to_numpy()]
    if len(result) and not panel.iloc[result][CLASS_COLUMN].eq(class_index).all():
        raise AssertionError("conditional payoff rows escaped their declared hard class")
    return result


def fit_simplex_model(family: str, params: Mapping[str, Any], matrix: pd.DataFrame, target: np.ndarray, *, seed: int, soft: bool = False) -> Any:
    target_array = np.asarray(target)
    if soft:
        if family != "logistic":
            raise ValueError("soft-simplex challenger is intentionally logistic-only")
        weights = np.asarray(target_array, dtype=float)
        if weights.ndim != 2 or weights.shape[1] != 3:
            raise ValueError("soft-simplex requires N x 3 replicated target weights")
        repeated = pd.concat([matrix, matrix, matrix], ignore_index=True)
        labels = np.repeat(np.arange(3), len(matrix))
        flat_weights = weights.T.ravel()
        keep = flat_weights > 1e-8
        model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=float(params["C"]), max_iter=400, solver="lbfgs", multi_class="multinomial", random_state=seed))
        model.fit(repeated.iloc[np.flatnonzero(keep)], labels[keep], logisticregression__sample_weight=flat_weights[keep])
        return model
    y = target_array.astype(int)
    if set(np.unique(y)) != {0, 1, 2}:
        raise ValueError("all three hard simplex classes require training support")
    if family == "logistic":
        model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=float(params["C"]), max_iter=400, solver="lbfgs", multi_class="multinomial", random_state=seed))
        # Do not inverse-prevalence weight a model whose probabilities enter
        # expected utility. Such weights change the fitted class prior, and a
        # scalar temperature cannot restore natural P(class | x).
        model.fit(matrix, y)
        return model
    if family == "lightgbm":
        import lightgbm as lgb
        model = lgb.LGBMClassifier(objective="multiclass", num_class=3, n_estimators=240, learning_rate=0.04, subsample=0.80, subsample_freq=1, colsample_bytree=0.75, reg_alpha=0.5, n_jobs=4, verbosity=-1, random_state=seed, **dict(params))
        model.fit(matrix, y)
        return model
    if family == "catboost":
        from catboost import CatBoostClassifier
        model = CatBoostClassifier(loss_function="MultiClass", iterations=240, learning_rate=0.04, random_seed=seed, thread_count=4, verbose=False, allow_writing_files=False, **dict(params))
        model.fit(matrix, y)
        return model
    raise ValueError(f"unknown simplex family: {family}")


def predict_simplex(model: Any, matrix: pd.DataFrame) -> np.ndarray:
    values = np.asarray(model.predict_proba(matrix), dtype=float)
    if values.shape != (len(matrix), 3):
        raise ValueError("multiclass model did not return a three-class simplex")
    values = np.clip(values, 1e-8, 1.0)
    values /= values.sum(axis=1, keepdims=True)
    return values


def fit_payoff_model(family: str, params: Mapping[str, Any], matrix: pd.DataFrame, target: np.ndarray, *, seed: int) -> Any:
    if family == "logistic":
        # Huber is the robust baseline; Ridge is retained as fallback for a
        # degenerate constant response which Huber cannot fit.
        if set(params).difference({"huber_epsilon", "huber_alpha"}):
            raise ValueError("linear payoff geometry contains classifier-only parameters")
        estimator: Any = HuberRegressor(
            epsilon=float(params["huber_epsilon"]),
            alpha=float(params["huber_alpha"]),
            max_iter=300,
        )
        if np.nanstd(np.asarray(target, dtype=float)) < 1e-10:
            estimator = Ridge(alpha=1.0)
        model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), estimator)
        model.fit(matrix, np.asarray(target, dtype=float))
        return model
    if family == "lightgbm":
        import lightgbm as lgb
        model = lgb.LGBMRegressor(objective="huber", n_estimators=240, learning_rate=0.04, subsample=0.80, subsample_freq=1, colsample_bytree=0.75, reg_alpha=0.5, n_jobs=4, verbosity=-1, random_state=seed, **dict(params))
        model.fit(matrix, np.asarray(target, dtype=float))
        return model
    if family == "catboost":
        from catboost import CatBoostRegressor
        model = CatBoostRegressor(loss_function="Huber:delta=0.01", iterations=240, learning_rate=0.04, random_seed=seed, thread_count=4, verbose=False, allow_writing_files=False, **dict(params))
        model.fit(matrix, np.asarray(target, dtype=float))
        return model
    raise ValueError(f"unknown payoff family: {family}")


def _predict_regression(model: Any, matrix: pd.DataFrame) -> np.ndarray:
    result = np.asarray(model.predict(matrix), dtype=float)
    if result.shape != (len(matrix),) or not np.isfinite(result).all():
        raise ValueError("conditional payoff regressor returned invalid predictions")
    return result


def compose_expected_net(probability: np.ndarray, conditional_gross: np.ndarray, row_cost: np.ndarray) -> np.ndarray:
    probability = np.asarray(probability, dtype=float)
    payoff = np.asarray(conditional_gross, dtype=float)
    cost = np.asarray(row_cost, dtype=float)
    if probability.ndim != 2 or payoff.shape != probability.shape or probability.shape[1] != 3 or cost.shape != (len(probability),):
        raise ValueError("composition inputs have incompatible shapes")
    if not np.isfinite(probability).all() or not np.isfinite(payoff).all() or not np.isfinite(cost).all() or not np.allclose(probability.sum(axis=1), 1.0, atol=1e-6, rtol=0.0):
        raise ValueError("composition requires a finite closed simplex, finite payoff and one row cost")
    return (probability * payoff).sum(axis=1) - cost


def true_class_oracle_score(
    evaluation_true_class: np.ndarray,
    train_side_class_gross_means: np.ndarray,
    row_cost: np.ndarray,
) -> np.ndarray:
    """A permanent nonpredictive ceiling: reveal only the evaluation class.

    The class payoff itself remains a mean fitted from the training side/class
    rows.  This diagnostic must never be used as a model input, selection
    target, HPO score, calibration reference, or deployable ranking.
    """

    classes = np.asarray(evaluation_true_class, dtype=int)
    means = np.asarray(train_side_class_gross_means, dtype=float)
    cost = np.asarray(row_cost, dtype=float)
    if means.shape != (3,) or classes.shape != cost.shape or not np.isfinite(means).all() or not np.isfinite(cost).all() or not np.isin(classes, (0, 1, 2)).all():
        raise ValueError("true-class oracle requires three train-only class means and valid evaluation classes/costs")
    return means[classes] - cost


def multiclass_metrics(target: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    y = np.asarray(target, dtype=int)
    p = np.asarray(probability, dtype=float)
    result: dict[str, Any] = {
        "rows": int(len(y)), "nll": float(log_loss(y, p, labels=[0, 1, 2])),
        "rps": float(np.mean(np.sum((np.cumsum(p, axis=1) - (np.arange(3)[None, :] == y[:, None]).cumsum(axis=1)) ** 2, axis=1) / 2.0)),
        "simplex_error_max": float(np.abs(p.sum(axis=1) - 1.0).max()),
    }
    for class_index, name in enumerate(CLASS_NAMES):
        binary = (y == class_index).astype(int)
        pred = p[:, class_index]
        bins = np.minimum((pred * 10).astype(int), 9)
        result[f"{name}_prevalence"] = float(binary.mean())
        result[f"{name}_brier"] = float(np.mean((pred - binary) ** 2))
        result[f"{name}_ece10"] = float(sum((bins == index).mean() * abs(pred[bins == index].mean() - binary[bins == index].mean()) for index in np.unique(bins)))
        result[f"{name}_auc"] = float(roc_auc_score(binary, pred)) if binary.min() != binary.max() else float("nan")
        result[f"{name}_ap"] = float(average_precision_score(binary, pred)) if binary.sum() else float("nan")
    return result


def payoff_metrics(target: np.ndarray, prediction: np.ndarray, baseline: float) -> dict[str, float]:
    y = np.asarray(target, dtype=float)
    p = np.asarray(prediction, dtype=float)
    delta = y - p
    huber = np.where(np.abs(delta) <= 0.01, 0.5 * delta ** 2, 0.01 * (np.abs(delta) - 0.005)).mean()
    base_delta = y - float(baseline)
    base_huber = np.where(np.abs(base_delta) <= 0.01, 0.5 * base_delta ** 2, 0.01 * (np.abs(base_delta) - 0.005)).mean()
    ic = spearmanr(y, p).statistic if len(y) >= 3 and np.unique(p).size > 1 else float("nan")
    return {"mae": float(np.mean(np.abs(delta))), "huber": float(huber), "ic": float(ic) if np.isfinite(ic) else float("nan"), "class_side_median": float(baseline), "median_mae": float(np.mean(np.abs(base_delta))), "median_huber": float(base_huber)}


def economic_metrics(frame: pd.DataFrame, score: str, *, fraction: float, evaluation: str) -> dict[str, Any]:
    selected = stable_top(frame, score, fraction=fraction)
    gross = selected["execution_gross_ev_12h"].to_numpy(float)
    cost = selected["execution_cost_return"].to_numpy(float)
    net = selected["execution_net_ev_12h"].to_numpy(float)
    if not np.allclose(gross - cost, net, atol=1e-7, rtol=0.0):
        raise ValueError("global top-k exact economics has a double-cost or anchor error")
    favorable_touch = selected["first_favorable_minute"].notna().to_numpy()
    population_favorable = int(frame["first_favorable_minute"].notna().sum())
    clean = selected["clean_economic_favorable_first"].to_numpy(int).astype(bool)
    population_clean = int(frame["clean_economic_favorable_first"].sum())
    adverse = selected["adverse_first"].to_numpy(int).astype(bool)
    timeout = selected["timeout"].to_numpy(int).astype(bool)
    score_values = selected[score].to_numpy(float)
    within_tail_net_ic = (
        spearmanr(score_values, net).statistic
        if len(selected) >= 3 and np.unique(score_values).size > 1 and np.unique(net).size > 1
        else float("nan")
    )
    return {
        "evaluation": evaluation,
        "score": score,
        "selected_fraction": fraction,
        "population_rows": len(frame),
        "selected_rows": len(selected),
        "net_ev_bps": float(net.mean() * 1e4),
        "gross_ev_bps": float(gross.mean() * 1e4),
        "cost_bps": float(cost.mean() * 1e4),
        "mfe_bps": float(selected["execution_mfe_return_12h"].mean() * 1e4),
        "positive_net_rate": float((net > 0).mean()),
        "favorable_touch_rate": float(favorable_touch.mean()),
        "favorable_touch_recall": float(favorable_touch.sum() / population_favorable) if population_favorable else float("nan"),
        "clean_first_rate": float(clean.mean()),
        "clean_first_recall": float(clean.sum() / population_clean) if population_clean else float("nan"),
        "adverse_first_rate": float(adverse.mean()),
        "timeout_rate": float(timeout.mean()),
        "within_tail_net_ic": float(within_tail_net_ic) if np.isfinite(within_tail_net_ic) else float("nan"),
        "score_std": float(np.std(score_values)),
        "score_unique": int(np.unique(score_values).size),
        "cutoff_tie_rows": int(np.isclose(frame[score].to_numpy(float), selected[score].min(), atol=0.0, rtol=0.0).sum()),
        "cvar5_bps": float(np.sort(net)[:max(1, math.ceil(len(net) * .05))].mean() * 1e4),
        "long_share": float(selected["side_name"].eq("long").mean()),
        "long_rows": int(selected["side_name"].eq("long").sum()),
        "short_rows": int(selected["side_name"].eq("short").sum()),
        "asset_count": int(selected["__symbol__"].nunique()),
    }


def score_bridge_metrics(frame: pd.DataFrame, score: str, *, evaluation: str, scope: str) -> dict[str, Any]:
    """Bind one frozen score to the exact opportunity-to-net waterfall."""

    values = frame[score].to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError(f"IC-to-EV bridge score is nonfinite: {score}/{evaluation}/{scope}")
    result: dict[str, Any] = {
        "evaluation": evaluation,
        "scope": scope,
        "score": score,
        "rows": int(len(frame)),
        "score_std": float(np.std(values)),
        "score_unique": int(np.unique(values).size),
    }
    outcomes = {
        "mfe": frame["execution_mfe_return_12h"].to_numpy(float),
        "gross": frame["execution_gross_ev_12h"].to_numpy(float),
        "net": frame["execution_net_ev_12h"].to_numpy(float),
        "favorable_touch": frame["first_favorable_minute"].notna().to_numpy(int),
        "clean_first": frame["clean_economic_favorable_first"].to_numpy(int),
        "adverse_first": frame["adverse_first"].to_numpy(int),
        "timeout": frame["timeout"].to_numpy(int),
    }
    for name, target in outcomes.items():
        statistic = (
            spearmanr(values, target).statistic
            if len(frame) >= 3 and np.unique(values).size > 1 and np.unique(target).size > 1
            else float("nan")
        )
        result[f"{name}_rank_ic"] = float(statistic) if np.isfinite(statistic) else float("nan")
    return result


def _inner_calibration_split(panel: pd.DataFrame, positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ordered = positions[np.argsort(panel.iloc[positions]["__ts__"].to_numpy(), kind="stable")]
    boundary = pd.Timestamp(panel.iloc[ordered[max(1, int(len(ordered) * .80))]]["__ts__"])
    fit = ordered[(panel.iloc[ordered]["label_resolution_utc"] < boundary).to_numpy()]
    calibration = ordered[(panel.iloc[ordered]["__ts__"] >= boundary).to_numpy()]
    if len(fit) < 500 or len(calibration) < 100:
        raise ValueError("insufficient chronological inner temperature-calibration support")
    if not (panel.iloc[fit]["label_resolution_utc"] < panel.iloc[calibration]["__ts__"].min()).all():
        raise ValueError("inner temperature calibration violates 12h purge")
    return fit, calibration


def joint_hpo_combinations(
    classifier_params: Sequence[Mapping[str, Any]],
    payoff_params: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    """Enumerate the bounded classifier x three-independent-payoff surface."""

    if set(payoff_params) != set(CLASS_NAMES):
        raise ValueError("joint HPO needs exactly one payoff grid for every simplex class")
    combinations: list[dict[str, Any]] = []
    for classifier, choices in itertools.product(
        classifier_params,
        itertools.product(*(payoff_params[name] for name in CLASS_NAMES)),
    ):
        combinations.append(
            {
                "classifier_params": dict(classifier),
                "payoff_params": {
                    name: dict(params) for name, params in zip(CLASS_NAMES, choices)
                },
            }
        )
    expected = len(classifier_params) * math.prod(len(payoff_params[name]) for name in CLASS_NAMES)
    if len(combinations) != expected or not combinations:
        raise AssertionError("joint HPO candidate enumeration is incomplete")
    return combinations


def _select_geometry(panel: pd.DataFrame, matrix: pd.DataFrame, *, side: str, family: str, seed: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Select a whole probability/payoff composition on the purged May split.

    Each family/side searches exactly 2 classifier choices x 2 choices for
    each of the three conditional gross heads: 16 compositions.  Feature
    screens, winsorization and every fitted model are limited to May 1--24;
    May 25--31 supplies proper scores and exact-net composition economics.
    """

    from scripts.run_meaningful_mfe_exact_grid_reset import _inner_may_masks

    train, validation = _inner_may_masks(panel)
    side_train = train[panel.iloc[train]["side_name"].eq(side).to_numpy()]
    side_validation = validation[panel.iloc[validation]["side_name"].eq(side).to_numpy()]
    y_train = panel.iloc[side_train][CLASS_COLUMN].to_numpy(int)
    y_validation = panel.iloc[side_validation][CLASS_COLUMN].to_numpy(int)
    simplex_features, _ = select_simplex_features(matrix, panel[CLASS_COLUMN].to_numpy(int), side_train, FEATURE_COUNT)
    classifier_results: dict[str, tuple[np.ndarray, dict[str, Any]]] = {}
    for classifier_index, params in enumerate(SIMPLEX_GRIDS[family]):
        model = fit_simplex_model(
            family,
            params,
            matrix.iloc[side_train][simplex_features],
            y_train,
            seed=seed + classifier_index,
        )
        probability = predict_simplex(model, matrix.iloc[side_validation][simplex_features])
        classifier_results[json.dumps(dict(params), sort_keys=True)] = (
            probability,
            multiclass_metrics(y_validation, probability),
        )

    payoff_features: dict[str, list[str]] = {}
    payoff_bounds: dict[str, tuple[float, float]] = {}
    payoff_results: dict[str, dict[str, tuple[np.ndarray, dict[str, Any]]]] = {
        name: {} for name in CLASS_NAMES
    }
    for class_index, class_name in enumerate(CLASS_NAMES):
        class_train = conditional_class_positions(panel, side_train, class_index)
        if len(class_train) < MIN_CONDITIONAL_PAYOFF_ROWS:
            raise ValueError(f"insufficient May conditional payoff support {family}/{side}/{class_name}")
        selected, _ = select_features_nested(
            matrix,
            panel["execution_gross_ev_12h"].to_numpy(float),
            class_train,
            FEATURE_COUNT,
        )
        bounds = winsor_bounds(panel.iloc[class_train]["execution_gross_ev_12h"].to_numpy(float))
        target, _ = winsorize_train_target(
            panel.iloc[class_train]["execution_gross_ev_12h"].to_numpy(float), bounds
        )
        payoff_features[class_name] = selected
        payoff_bounds[class_name] = bounds
        observed = side_validation[
            panel.iloc[side_validation][CLASS_COLUMN].eq(class_index).to_numpy()
        ]
        if not len(observed):
            raise ValueError(f"May validation lacks conditional payoff support {family}/{side}/{class_name}")
        baseline = float(np.median(panel.iloc[class_train]["execution_gross_ev_12h"].to_numpy(float)))
        for payoff_index, params in enumerate(PAYOFF_GRIDS[family]):
            model = fit_payoff_model(
                family,
                params,
                matrix.iloc[class_train][selected],
                target,
                seed=seed + 10_000 + class_index * 100 + payoff_index,
            )
            prediction = np.clip(
                _predict_regression(model, matrix.iloc[side_validation][selected]), *bounds
            )
            payoff_results[class_name][json.dumps(dict(params), sort_keys=True)] = (
                prediction,
                payoff_metrics(
                    panel.iloc[observed]["execution_gross_ev_12h"].to_numpy(float),
                    prediction[
                        panel.iloc[side_validation][CLASS_COLUMN].to_numpy(int)
                        == class_index
                    ],
                    baseline,
                ),
            )

    candidates: list[dict[str, Any]] = []
    for candidate_index, combination in enumerate(
        joint_hpo_combinations(
            SIMPLEX_GRIDS[family],
            {name: PAYOFF_GRIDS[family] for name in CLASS_NAMES},
        )
    ):
        classifier_key = json.dumps(combination["classifier_params"], sort_keys=True)
        probability, classifier_metrics = classifier_results[classifier_key]
        conditional = np.column_stack(
            [
                payoff_results[name][json.dumps(combination["payoff_params"][name], sort_keys=True)][0]
                for name in CLASS_NAMES
            ]
        )
        payoff_metric_by_name = {
            name: payoff_results[name][json.dumps(combination["payoff_params"][name], sort_keys=True)][1]
            for name in CLASS_NAMES
        }
        score = compose_expected_net(
            probability,
            conditional,
            panel.iloc[side_validation]["execution_cost_return"].to_numpy(float),
        )
        exact_net = panel.iloc[side_validation]["execution_net_ev_12h"].to_numpy(float)
        composed_ic = (
            spearmanr(score, exact_net).statistic
            if np.unique(score).size > 1 and np.unique(exact_net).size > 1
            else float("nan")
        )
        ranked = panel.iloc[side_validation].copy()
        ranked["__inner_composed_score"] = score
        top10_net = float(
            stable_top(ranked, "__inner_composed_score", fraction=.10)[
                "execution_net_ev_12h"
            ].mean()
            * 1e4
        )
        mean_huber = float(np.mean([payoff_metric_by_name[name]["huber"] for name in CLASS_NAMES]))
        mean_mae = float(np.mean([payoff_metric_by_name[name]["mae"] for name in CLASS_NAMES]))
        # This is a documented bounded diagnostic objective, not a trading
        # policy: proper simplex loss + proper conditional-payoff losses +
        # composed within-side exact-net rank and deterministic top-decile EV.
        objective = float(
            classifier_metrics["nll"]
            + classifier_metrics["rps"]
            + classifier_metrics["clean_economic_favorable_first_brier"]
            - .10 * classifier_metrics["clean_economic_favorable_first_ap"]
            + mean_huber
            + .25 * mean_mae
            - .01 * (composed_ic if np.isfinite(composed_ic) else 0.0)
            - .0001 * np.clip(top10_net, -500.0, 500.0)
        )
        row: dict[str, Any] = {
            "candidate_index": candidate_index,
            "classifier_params": combination["classifier_params"],
            "payoff_params": combination["payoff_params"],
            "objective": objective,
            "objective_contract": "nll+rps+clean_brier-0.10*clean_ap+mean_payoff_huber+0.25*mean_payoff_mae-0.01*composed_net_ic-0.0001*clipped_side_local_top10_net_bps",
            "classifier_nll": classifier_metrics["nll"],
            "classifier_rps": classifier_metrics["rps"],
            "classifier_clean_brier": classifier_metrics["clean_economic_favorable_first_brier"],
            "classifier_clean_ap": classifier_metrics["clean_economic_favorable_first_ap"],
            "composed_net_ic": float(composed_ic) if np.isfinite(composed_ic) else float("nan"),
            "side_local_top10_net_bps": top10_net,
            "classifier_selected_feature_count": len(simplex_features),
        }
        for class_name in CLASS_NAMES:
            payoff_metric = payoff_metric_by_name[class_name]
            row[f"{class_name}_params"] = combination["payoff_params"][class_name]
            row[f"{class_name}_selected_feature_count"] = len(payoff_features[class_name])
            row[f"{class_name}_winsor_low"] = payoff_bounds[class_name][0]
            row[f"{class_name}_winsor_high"] = payoff_bounds[class_name][1]
            for metric_name in ("mae", "huber", "ic", "median_mae", "median_huber"):
                row[f"{class_name}_{metric_name}"] = payoff_metric[metric_name]
        candidates.append(row)
    winner = min(
        candidates,
        key=lambda row: (
            float(row["objective"]),
            float(row["classifier_nll"]),
            -float(row["side_local_top10_net_bps"]),
            json.dumps(row["classifier_params"], sort_keys=True),
            json.dumps(row["payoff_params"], sort_keys=True),
        ),
    )
    return winner, candidates


def _select_direct_geometry(panel: pd.DataFrame, matrix: pd.DataFrame, *, side: str, family: str, seed: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """May-only direct-net control selection, independent of simplex labels."""

    from scripts.run_meaningful_mfe_exact_grid_reset import _inner_may_masks
    train, validation = _inner_may_masks(panel)
    side_train = train[panel.iloc[train]["side_name"].eq(side).to_numpy()]
    side_validation = validation[panel.iloc[validation]["side_name"].eq(side).to_numpy()]
    selected, _ = select_features_nested(matrix, panel["execution_net_ev_12h"].to_numpy(float), side_train, FEATURE_COUNT)
    bounds = winsor_bounds(panel.iloc[side_train]["execution_net_ev_12h"].to_numpy(float))
    target, _ = winsorize_train_target(panel.iloc[side_train]["execution_net_ev_12h"].to_numpy(float), bounds)
    candidates: list[dict[str, Any]] = []
    for index, params in enumerate(PAYOFF_GRIDS[family]):
        model = fit_payoff_model(family, params, matrix.iloc[side_train][selected], target, seed=seed + index)
        prediction = np.clip(_predict_regression(model, matrix.iloc[side_validation][selected]), *bounds)
        y = panel.iloc[side_validation]["execution_net_ev_12h"].to_numpy(float)
        metrics = payoff_metrics(y, prediction, float(np.median(panel.iloc[side_train]["execution_net_ev_12h"].to_numpy(float))))
        ranked = panel.iloc[side_validation].copy(); ranked["__inner_direct_score"] = prediction
        top10_net = float(stable_top(ranked, "__inner_direct_score", fraction=.10)["execution_net_ev_12h"].mean() * 1e4)
        objective = float(metrics["huber"] + metrics["mae"] - .01 * (metrics["ic"] if math.isfinite(metrics["ic"]) else 0.0) - .0001 * np.clip(top10_net, -500.0, 500.0))
        candidates.append({"params": dict(params), "objective": objective, "mae": metrics["mae"], "huber": metrics["huber"], "ic": metrics["ic"], "top10_net_bps": top10_net, "selected_feature_count": len(selected)})
    winner = min(candidates, key=lambda row: (float(row["objective"]), float(row["huber"]), float(row["mae"]), json.dumps(row["params"], sort_keys=True)))
    return winner, candidates


def _fit_score_split(panel: pd.DataFrame, matrix: pd.DataFrame, train: np.ndarray, evaluation: np.ndarray, geometries: Mapping[str, Mapping[str, Any]], *, seed: int, include_soft: bool) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    keep = [
        *IDENTITY,
        "base_oof_score",
        "execution_mfe_return_12h",
        "execution_gross_ev_12h",
        "execution_cost_return",
        "execution_net_ev_12h",
        "first_favorable_minute",
        "first_adverse_minute",
        CLASS_COLUMN,
        *CLASS_NAMES,
    ]
    scored = panel.iloc[evaluation][keep].copy().reset_index(drop=True)
    if not np.isfinite(scored["base_oof_score"].to_numpy(float)).all():
        raise ValueError("frozen base OOF score is unavailable on an evaluation split")
    scored["score_frozen_base_alpha"] = scored["base_oof_score"].to_numpy(float)
    metrics: list[dict[str, Any]] = []
    features_rows: list[dict[str, Any]] = []
    payoff_rows: list[dict[str, Any]] = []
    selection_cache: dict[tuple[str, str], tuple[list[str], pd.DataFrame]] = {}
    for family_index, family in enumerate(FAMILIES):
        for side_index, side in enumerate(SIDES):
            side_train = train[panel.iloc[train]["side_name"].eq(side).to_numpy()]
            side_eval = evaluation[panel.iloc[evaluation]["side_name"].eq(side).to_numpy()]
            output = np.flatnonzero(scored["side_name"].eq(side).to_numpy())
            if len(side_train) < 1000 or len(side_eval) < 100:
                raise ValueError(f"insufficient side support for {family}/{side}")
            geometry = geometries[family][side]["simplex"]
            direct_geometry = geometries[family][side]["direct_net"]
            simplex_key = (side, "simplex")
            if simplex_key not in selection_cache:
                selection_cache[simplex_key] = select_simplex_features(matrix, panel[CLASS_COLUMN].to_numpy(int), side_train, FEATURE_COUNT)
            selected, screen = selection_cache[simplex_key]
            model = fit_simplex_model(family, geometry["classifier_params"], matrix.iloc[side_train][selected], panel.iloc[side_train][CLASS_COLUMN].to_numpy(int), seed=seed + family_index * 10000 + side_index * 1000)
            raw = predict_simplex(model, matrix.iloc[side_eval][selected])
            cal_fit, cal_eval = _inner_calibration_split(panel, side_train)
            calibration_key = (side, "temperature_calibration_simplex")
            if calibration_key not in selection_cache:
                selection_cache[calibration_key] = select_simplex_features(matrix, panel[CLASS_COLUMN].to_numpy(int), cal_fit, FEATURE_COUNT)
            cal_selected, _ = selection_cache[calibration_key]
            calibration_model = fit_simplex_model(family, geometry["classifier_params"], matrix.iloc[cal_fit][cal_selected], panel.iloc[cal_fit][CLASS_COLUMN].to_numpy(int), seed=seed + family_index * 10000 + side_index * 1000 + 77)
            calibration_probability = predict_simplex(
                calibration_model, matrix.iloc[cal_eval][cal_selected]
            )
            calibration_target = panel.iloc[cal_eval][CLASS_COLUMN].to_numpy(int)
            temperature = fit_temperature(calibration_probability, calibration_target)
            calibrated = temperature_scale_probabilities(raw, temperature)
            offset_temperature = fit_offset_temperature_calibrator(
                calibration_probability, calibration_target
            )
            offset_calibrated = apply_offset_temperature_calibrator(
                raw, offset_temperature
            )
            conditional = np.empty((len(side_eval), 3), dtype=float)
            train_class_means = np.empty(3, dtype=float)
            for class_index, class_name in enumerate(CLASS_NAMES):
                class_train = conditional_class_positions(panel, side_train, class_index)
                if len(class_train) < MIN_CONDITIONAL_PAYOFF_ROWS:
                    raise ValueError(f"insufficient conditional payoff support {family}/{side}/{class_name}")
                train_class_means[class_index] = float(panel.iloc[class_train]["execution_gross_ev_12h"].mean())
                payoff_key = (side, f"gross_given_{class_name}")
                if payoff_key not in selection_cache:
                    selection_cache[payoff_key] = select_features_nested(matrix, panel["execution_gross_ev_12h"].to_numpy(float), class_train, FEATURE_COUNT)
                features, payoff_screen = selection_cache[payoff_key]
                payoff_target, target_clipped_rate = winsorize_train_target(panel.iloc[class_train]["execution_gross_ev_12h"].to_numpy(float), winsor_bounds(panel.iloc[class_train]["execution_gross_ev_12h"].to_numpy(float)))
                payoff_bounds = winsor_bounds(panel.iloc[class_train]["execution_gross_ev_12h"].to_numpy(float))
                payoff_model = fit_payoff_model(family, geometry["payoff_params"][class_name], matrix.iloc[class_train][features], payoff_target, seed=seed + family_index * 10000 + side_index * 1000 + class_index * 100)
                raw_payoff_prediction = _predict_regression(payoff_model, matrix.iloc[side_eval][features])
                conditional[:, class_index] = np.clip(raw_payoff_prediction, *payoff_bounds)
                observed = side_eval[panel.iloc[side_eval][CLASS_COLUMN].eq(class_index).to_numpy()]
                if len(observed):
                    baseline = float(np.median(panel.iloc[class_train]["execution_gross_ev_12h"].to_numpy(float)))
                    payoff_rows.append({"family": family, "side": side, "class_name": class_name, "train_rows": len(class_train), "evaluation_rows": len(observed), "winsor_low": payoff_bounds[0], "winsor_high": payoff_bounds[1], "train_target_clipped_rate": target_clipped_rate, "evaluation_prediction_clipped_rate": float(np.mean(~np.isclose(raw_payoff_prediction, conditional[:, class_index], atol=0.0, rtol=0.0))), **payoff_metrics(panel.iloc[observed]["execution_gross_ev_12h"].to_numpy(float), conditional[panel.iloc[side_eval][CLASS_COLUMN].to_numpy(int) == class_index, class_index], baseline)})
                features_rows.append({"family": family, "side": side, "task": f"gross_given_{class_name}", "selected_feature_count": len(features), "selected_features": json.dumps(features), "screen_top20": json.dumps(_safe(payoff_screen.head(20).to_dict("records")), sort_keys=True)})
            direct_key = (side, "direct_net")
            if direct_key not in selection_cache:
                selection_cache[direct_key] = select_features_nested(matrix, panel["execution_net_ev_12h"].to_numpy(float), side_train, FEATURE_COUNT)
            direct_features, direct_screen = selection_cache[direct_key]
            direct_bounds = winsor_bounds(panel.iloc[side_train]["execution_net_ev_12h"].to_numpy(float))
            direct_target, direct_target_clipped_rate = winsorize_train_target(panel.iloc[side_train]["execution_net_ev_12h"].to_numpy(float), direct_bounds)
            direct = fit_payoff_model(family, direct_geometry["params"], matrix.iloc[side_train][direct_features], direct_target, seed=seed + family_index * 10000 + side_index * 1000 + 999)
            raw_direct_prediction = _predict_regression(direct, matrix.iloc[side_eval][direct_features])
            direct_prediction = np.clip(raw_direct_prediction, *direct_bounds)
            calibration_rows = int(len(calibration_target))
            for mode, probability, calibration_detail in (
                ("raw", raw, {"reason": "not_applicable", "offsets": [0.0, 0.0, 0.0]}),
                ("scalar_temperature", calibrated, {"reason": "ok", "offsets": [0.0, 0.0, 0.0]}),
                ("offset_temperature", offset_calibrated, offset_temperature),
            ):
                for class_index, class_name in enumerate(CLASS_NAMES):
                    scored.loc[output, f"p_{family}_{mode}_{class_name}"] = probability[:, class_index]
                metrics.append({
                    "family": family,
                    "side": side,
                    "mode": mode,
                    "temperature": 1.0 if mode == "raw" else (temperature if mode == "scalar_temperature" else float(calibration_detail["temperature"])),
                    "calibration_rows": calibration_rows,
                    "calibration_target": "hard_first_touch",
                    "calibration_available": bool(calibration_detail.get("available", mode != "offset_temperature")),
                    "calibration_reason": str(calibration_detail["reason"]),
                    "calibration_offsets": json.dumps(_safe(calibration_detail["offsets"])),
                    "calibration_l2": float(calibration_detail.get("l2", 0.0)),
                    "calibration_objective": float(calibration_detail.get("objective", float("nan"))),
                    **multiclass_metrics(panel.iloc[side_eval][CLASS_COLUMN].to_numpy(int), probability),
                })
                scored.loc[output, f"score_{family}_{mode}_composed"] = compose_expected_net(probability, conditional, panel.iloc[side_eval]["execution_cost_return"].to_numpy(float))
            scored.loc[output, f"score_{family}_direct_net"] = direct_prediction
            # Identical train-only class means are used by each family, but we
            # write a single family-independent ceiling once to avoid treating
            # a revealed true class as an architecture comparison.
            if family == FAMILIES[0]:
                scored.loc[output, "score_true_class_oracle"] = true_class_oracle_score(
                    panel.iloc[side_eval][CLASS_COLUMN].to_numpy(int),
                    train_class_means,
                    panel.iloc[side_eval]["execution_cost_return"].to_numpy(float),
                )
            payoff_rows.append({"family": family, "side": side, "class_name": "direct_net", "train_rows": len(side_train), "evaluation_rows": len(side_eval), "winsor_low": direct_bounds[0], "winsor_high": direct_bounds[1], "train_target_clipped_rate": direct_target_clipped_rate, "evaluation_prediction_clipped_rate": float(np.mean(~np.isclose(raw_direct_prediction, direct_prediction, atol=0.0, rtol=0.0))), **payoff_metrics(panel.iloc[side_eval]["execution_net_ev_12h"].to_numpy(float), direct_prediction, float(np.median(panel.iloc[side_train]["execution_net_ev_12h"].to_numpy(float))))})
            features_rows.extend([
                {"family": family, "side": side, "task": "simplex", "selected_feature_count": len(selected), "selected_features": json.dumps(selected), "screen_top20": json.dumps(_safe(screen.head(20).to_dict("records")), sort_keys=True)},
                {"family": family, "side": side, "task": "direct_net", "selected_feature_count": len(direct_features), "selected_features": json.dumps(direct_features), "screen_top20": json.dumps(_safe(direct_screen.head(20).to_dict("records")), sort_keys=True)},
            ])
            if include_soft and family == "logistic":
                soft_model = fit_simplex_model("logistic", geometry["classifier_params"], matrix.iloc[side_train][selected], soft_simplex_targets(panel.iloc[side_train]), seed=seed + side_index * 1000 + 500, soft=True)
                soft_probability = predict_simplex(soft_model, matrix.iloc[side_eval][selected])
                soft_calibration_model = fit_simplex_model(
                    "logistic",
                    geometry["classifier_params"],
                    matrix.iloc[cal_fit][cal_selected],
                    soft_simplex_targets(panel.iloc[cal_fit]),
                    seed=seed + side_index * 1000 + 577,
                    soft=True,
                )
                soft_calibration_probability = predict_simplex(
                    soft_calibration_model, matrix.iloc[cal_eval][cal_selected]
                )
                soft_calibration_target = soft_simplex_targets(panel.iloc[cal_eval])
                soft_temperature = fit_temperature(
                    soft_calibration_probability, soft_calibration_target
                )
                soft_scalar_calibrated = temperature_scale_probabilities(
                    soft_probability, soft_temperature
                )
                soft_offset_temperature = fit_offset_temperature_calibrator(
                    soft_calibration_probability, soft_calibration_target
                )
                soft_offset_calibrated = apply_offset_temperature_calibrator(
                    soft_probability, soft_offset_temperature
                )
                for mode, probability, calibration_detail in (
                    ("raw", soft_probability, {"reason": "not_applicable", "offsets": [0.0, 0.0, 0.0]}),
                    ("scalar_temperature", soft_scalar_calibrated, {"reason": "ok", "offsets": [0.0, 0.0, 0.0]}),
                    ("offset_temperature", soft_offset_calibrated, soft_offset_temperature),
                ):
                    for class_index, class_name in enumerate(CLASS_NAMES):
                        scored.loc[output, f"p_soft_logistic_{mode}_{class_name}"] = probability[:, class_index]
                    scored.loc[output, f"score_soft_logistic_{mode}_composed"] = compose_expected_net(probability, conditional, panel.iloc[side_eval]["execution_cost_return"].to_numpy(float))
                    metrics.append({
                        "family": "soft_logistic",
                        "side": side,
                        "mode": mode,
                        "temperature": 1.0 if mode == "raw" else (soft_temperature if mode == "scalar_temperature" else float(calibration_detail["temperature"])),
                        "calibration_rows": int(len(soft_calibration_target)),
                        "calibration_target": "timeout_soft_fractional",
                        "calibration_available": bool(calibration_detail.get("available", mode != "offset_temperature")),
                        "calibration_reason": str(calibration_detail["reason"]),
                        "calibration_offsets": json.dumps(_safe(calibration_detail["offsets"])),
                        "calibration_l2": float(calibration_detail.get("l2", 0.0)),
                        "calibration_objective": float(calibration_detail.get("objective", float("nan"))),
                        **multiclass_metrics(panel.iloc[side_eval][CLASS_COLUMN].to_numpy(int), probability),
                    })
    return scored, metrics, features_rows, payoff_rows


def _payoff_transfer(panel: pd.DataFrame) -> pd.DataFrame:
    work = panel.copy(); work["month"] = pd.to_datetime(work["__ts__"], utc=True).dt.strftime("%Y-%m")
    return work.groupby(["month", "side_name", CLASS_COLUMN], observed=True).agg(rows=("candidate_id", "size"), gross_mean=("execution_gross_ev_12h", "mean"), net_mean=("execution_net_ev_12h", "mean"), gross_median=("execution_gross_ev_12h", "median"), net_median=("execution_net_ev_12h", "median")).reset_index()


def aggregate_grouped_july_metrics(scored: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for family in FAMILIES:
        for side in SIDES:
            local = scored.loc[scored["side_name"].eq(side)]
            y = local[CLASS_COLUMN].to_numpy(int)
            for mode in ("raw", "scalar_temperature", "offset_temperature"):
                probability = local[[f"p_{family}_{mode}_{name}" for name in CLASS_NAMES]].to_numpy(float)
                rows.append({"evaluation": "july_grouped_oof", "family": family, "side": side, "mode": mode, "validation_days": "five_contiguous_two_day_blocks", **multiclass_metrics(y, probability)})
    for mode in ("raw", "scalar_temperature", "offset_temperature"):
        soft_columns = [f"p_soft_logistic_{mode}_{name}" for name in CLASS_NAMES]
        if set(soft_columns).issubset(scored.columns):
            for side in SIDES:
                local = scored.loc[scored["side_name"].eq(side)]
                rows.append({"evaluation": "july_grouped_oof", "family": "soft_logistic", "side": side, "mode": mode, "validation_days": "five_contiguous_two_day_blocks", **multiclass_metrics(local[CLASS_COLUMN].to_numpy(int), local[soft_columns].to_numpy(float))})
    return rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    contract = geometry_contract(args.label_kind, int(args.buffer_bps))
    if not contract["included"]:
        raise ValueError(f"requested geometry is excluded: {contract['reason']}")
    if args.label_dir is None:
        args.label_dir = PRIMARY_DIR if args.label_kind == "primary_floor" else NOFLOOR_DIR
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite competing-risk output: {args.output_dir}")
    panel, matrix, raw_features, lineage = load_competing_risk_panel(args.features, args.feature_manifest, args.grid, args.grid_manifest, args.label_dir, label_kind=args.label_kind, buffer_bps=args.buffer_bps)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    geometries: dict[str, dict[str, dict[str, Any]]] = {family: {} for family in FAMILIES}
    geometry_candidates: list[dict[str, Any]] = []
    for family_index, family in enumerate(FAMILIES):
        for side_index, side in enumerate(SIDES):
            simplex_winner, simplex_candidates = _select_geometry(panel, matrix, side=side, family=family, seed=args.seed + family_index * 1000 + side_index * 100)
            direct_winner, direct_candidates = _select_direct_geometry(panel, matrix, side=side, family=family, seed=args.seed + family_index * 1000 + side_index * 100 + 50_000)
            geometries[family][side] = {"simplex": simplex_winner, "direct_net": direct_winner}
            geometry_candidates.extend([{"family": family, "side": side, "task": "simplex", "candidate_index": index, "selected": candidate == simplex_winner, **candidate} for index, candidate in enumerate(simplex_candidates)])
            geometry_candidates.extend([{"family": family, "side": side, "task": "direct_net", "candidate_index": index, "selected": candidate == direct_winner, **candidate} for index, candidate in enumerate(direct_candidates)])
    all_scored: list[pd.DataFrame] = []; all_metrics: list[dict[str, Any]] = []; all_features: list[dict[str, Any]] = []; all_payoff: list[dict[str, Any]] = []; splits: list[dict[str, Any]] = []
    for split_index, spec in enumerate(TRANSFER_SPECS):
        train, evaluation = _base_masks(panel, spec)
        scored, metrics, selections, payoffs = _fit_score_split(panel, matrix, train, evaluation, geometries, seed=args.seed + split_index * 1_000_000, include_soft=args.label_kind == "nofloor" and args.buffer_bps == 50)
        scored["evaluation"] = spec.name; scored["fold_name"] = spec.name; all_scored.append(scored); all_metrics.extend([{**row, "evaluation": spec.name} for row in metrics]); all_features.extend([{**row, "evaluation": spec.name} for row in selections]); all_payoff.extend([{**row, "evaluation": spec.name} for row in payoffs]); splits.append({"name": spec.name, "train_rows": len(train), "evaluation_rows": len(evaluation), "source_forward_split_promotable": bool(spec.promotable), "promotion_eligible": False, "note": spec.note})
    july_parts: list[pd.DataFrame] = []
    for fold_index, (name, train, evaluation, days) in enumerate(july_grouped_day_folds(panel)):
        scored, metrics, selections, payoffs = _fit_score_split(panel, matrix, train, evaluation, geometries, seed=args.seed + 9_000_000 + fold_index * 1_000_000, include_soft=args.label_kind == "nofloor" and args.buffer_bps == 50)
        scored["evaluation"] = "july_grouped_oof"; scored["fold_name"] = name; july_parts.append(scored); all_metrics.extend([{**row, "evaluation": name, "validation_days": "|".join(days)} for row in metrics]); all_features.extend([{**row, "evaluation": name} for row in selections]); all_payoff.extend([{**row, "evaluation": name} for row in payoffs]); splits.append({"name": name, "train_rows": len(train), "evaluation_rows": len(evaluation), "validation_days": days, "promotion_eligible": False, "note": "GROUPED_JULY_OOF_NONPROMOTABLE"})
    july = pd.concat(july_parts, ignore_index=True)
    if july.duplicated(list(IDENTITY)).any():
        raise ValueError("grouped July OOF has duplicate exact identities")
    all_scored.append(july)
    all_metrics.extend(aggregate_grouped_july_metrics(july))
    predictions = pd.concat(all_scored, ignore_index=True)
    economics_rows: list[dict[str, Any]] = []
    for evaluation, group in predictions.groupby("evaluation", sort=True):
        for score in (name for name in group.columns if name.startswith("score_")):
            for fraction in (.01, .05, .10, .20):
                economics_rows.append(
                    {
                        **economic_metrics(group, score, fraction=fraction, evaluation=str(evaluation)),
                        "evaluation_scope": "aggregate",
                        "fold_name": "all",
                    }
                )
    # Keep the five independent grouped-July OOF books materialized.  The
    # aggregate pooled book remains the primary diagnostic, while no single
    # two-day holdout may disappear into its average.
    july_predictions = predictions.loc[predictions["evaluation"].eq("july_grouped_oof")]
    for fold_name, group in july_predictions.groupby("fold_name", sort=True):
        for score in (name for name in group.columns if name.startswith("score_")):
            for fraction in (.01, .05, .10, .20):
                economics_rows.append(
                    {
                        **economic_metrics(group, score, fraction=fraction, evaluation="july_grouped_oof"),
                        "evaluation_scope": "grouped_july_fold",
                        "fold_name": str(fold_name),
                    }
                )
    economics = pd.DataFrame(economics_rows)
    bridge_rows: list[dict[str, Any]] = []
    for evaluation, group in predictions.groupby("evaluation", sort=True):
        score_columns = [name for name in group.columns if name.startswith("score_")]
        for scope, local in (
            ("pooled_global", group),
            ("long", group.loc[group["side_name"].eq("long")]),
            ("short", group.loc[group["side_name"].eq("short")]),
        ):
            for score in score_columns:
                bridge_rows.append(score_bridge_metrics(local, score, evaluation=str(evaluation), scope=scope))
    ic_ev_bridge = pd.DataFrame(bridge_rows)
    payoff_metrics_frame = pd.DataFrame(all_payoff)
    conditional_support = payoff_metrics_frame.loc[
        payoff_metrics_frame["class_name"].isin(CLASS_NAMES), "train_rows"
    ]
    if conditional_support.empty or int(conditional_support.min()) < MIN_CONDITIONAL_PAYOFF_ROWS:
        raise ValueError("persisted conditional-payoff support violates the v2 minimum")
    outputs: dict[str, Any] = {}
    for name, frame in (("predictions", predictions), ("multiclass_metrics", pd.DataFrame(all_metrics)), ("feature_selections", pd.DataFrame(all_features)), ("geometry_candidates", pd.DataFrame(geometry_candidates)), ("conditional_payoff_metrics", payoff_metrics_frame), ("payoff_transfer", _payoff_transfer(panel)), ("economics", economics), ("ic_ev_bridge", ic_ev_bridge)):
        path = args.output_dir / f"{name}.parquet"; frame.to_parquet(path, index=False); outputs[name] = {"path": path, "rows": len(frame), "sha256": sha256(path)}
    report = {"schema": SCHEMA, "status": "COMPLETED_DIAGNOSTIC_COMPETING_RISK_SIMPLEX_NO_PROMOTION", "promotion_eligible": False, "runner": {"path": Path(__file__).resolve(), "sha256": sha256(Path(__file__).resolve())}, "lineage": lineage, "raw_feature_count": len(raw_features), "architecture_scope": "STANDALONE_PIT_SIMPLEX_ONLY: config-routed base->residual/context is not implemented here; frozen base_oof_score is IC-to-EV bridge-only and excluded from features", "minimum_conditional_payoff_train_rows": int(conditional_support.min()), "geometries": geometries, "splits": splits, "contracts": {"classification": "three mutually exclusive full-horizon first-touch classes, fitted separately by side and unweighted when probabilities enter expected utility", "payoff": f"side/family/class-specific gross mean regressor on observed class rows only; direct net comparator is separate; every fit requires at least {MIN_CONDITIONAL_PAYOFF_ROWS} train rows", "composition": "sum_k P(k)*E[gross|k]-row_execution_cost exactly once", "true_class_oracle": "permanently nonpromotable ceiling: evaluation true class revealed, train side/class gross mean only, row cost subtracted once; never a feature, HPO score, calibration input or predictive comparison", "calibration": "raw, scalar-temperature and regularized two-offset-plus-temperature modes fit only on chronological 12h-purged training rows; inadequate class support is explicitly raw fallback", "soft_challenger": "logistic weighted replicated simplex only for nofloor/50bps; hard hits one-hot, timeout-only terminal viability; scalar and offset-temperature calibrators fit fractional timeout targets while outer metrics retain hard observed classes", "selection": "simplex uses stable one-vs-rest screening; each conditional gross class and direct net independently selected train-only", "geometry": "per family/side, joint May HPO enumerates 2 classifier x 2 timeout-payoff x 2 adverse-payoff x 2 clean-payoff candidates; objective persists proper classification/payoff losses plus composed side-local exact-net IC and top10 economics; direct-net geometry is independently selected", "economics": "pooled global top 1/5/10/20%, deterministic identity ties, no side quotas; grouped-July aggregate and every two-day OOF fold are emitted", "ic_ev_bridge": "every evaluation persists the frozen base OOF score beside exact 12h MFE/gross/net and competing-risk rank IC; native base-label IC belongs to the source-separated historical waterfall because that target is not present in this current-lineage product", "diagnostic": "all output is nonpromotable; source-forward status is retained only as split metadata; reverse and grouped July folds are regime diagnostics"}, "outputs": outputs}
    report_path = args.output_dir / "report.json"; _write_json(report_path, report)
    _write_json(args.output_dir / "manifest.json", {"schema": SCHEMA, "status": report["status"], "promotion_eligible": False, "report": {"path": report_path, "sha256": sha256(report_path)}, "inputs": lineage, "outputs": outputs})
    return report


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--features", type=Path, default=ROOT / "data_perp/artifacts/exact_policy_capture_feature_universe_20260727_v2/capture_feature_universe.parquet")
    value.add_argument("--feature-manifest", type=Path, default=ROOT / "data_perp/artifacts/exact_policy_capture_feature_universe_20260727_v2/manifest.json")
    value.add_argument("--grid", type=Path, default=ROOT / "data_perp/artifacts/meaningful_mfe_exact_policy_label_grid_20260727_v1/meaningful_mfe_label_grid.parquet")
    value.add_argument("--grid-manifest", type=Path, default=ROOT / "data_perp/artifacts/meaningful_mfe_exact_policy_label_grid_20260727_v1/manifest.json")
    value.add_argument("--label-kind", choices=("primary_floor", "nofloor"), required=True)
    value.add_argument("--buffer-bps", type=int, required=True)
    value.add_argument("--label-dir", type=Path, default=None)
    value.add_argument("--output-dir", type=Path, required=True)
    value.add_argument("--seed", type=int, default=20260730)
    return value


if __name__ == "__main__":
    arguments = parser().parse_args()
    if arguments.label_dir is None:
        arguments.label_dir = PRIMARY_DIR if arguments.label_kind == "primary_floor" else NOFLOOR_DIR
    print(json.dumps(run(arguments), indent=2, default=str))
