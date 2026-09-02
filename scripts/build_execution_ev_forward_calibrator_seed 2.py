#!/usr/bin/env python3
"""Build the frozen 21-day operational EV-mapping seed before a forward block.

The serialized final direct/capture heads are scored on already-resolved,
pre-block rows.  Their fixed base-margin interaction is mapped to exact net EV
with a side-local isotonic map using only labels resolved in the trailing
21-day window.  The seed history and threshold arrays are persisted so future
decisions can be scored without consulting their outcomes.

This operational final-fit calibration is not OOS performance evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_exact_policy_capture_support_ablation import (  # noqa: E402
    load_frozen_base_margin_interaction,
)


SCHEMA = "execution_ev_forward_calibrator_seed_v1"
IDENTITY = ("candidate_id", "__ts__", "__symbol__", "side_name")
SIDES = ("long", "short")
TARGET = "execution_net_ev_12h"
BASELINE = "existing_alpha_ev"
DECISION = "execution_decision_utc"
RESOLUTION = "execution_label_end_utc"
DEFAULT_INPUT = Path(
    "data_perp/artifacts/execution_ev_canonical_exact_policy_regime_input_20260727_v3/"
    "joined.parquet"
)
DEFAULT_HEAD_ROOT = Path(
    "data_perp/artifacts/execution_ev_forward_final_heads_20260728_v1"
)
DEFAULT_SCREEN = Path(
    "data_perp/artifacts/execution_ev_false_positive_feature_diagnosis_20260727_v2/"
    "frozen_screens.csv"
)
DEFAULT_OUTPUT = Path(
    "data_perp/artifacts/execution_ev_forward_calibrator_seed_20260728_v1"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _utc(value: object, *, name: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ValueError(f"{name} must be timezone-aware UTC")
    return timestamp.tz_convert("UTC")


def interaction_score(
    direct: np.ndarray,
    capture: np.ndarray,
    margin: np.ndarray,
    *,
    contract: Mapping[str, Any],
    direct_center: float | None = None,
    direct_scale: float | None = None,
    capture_center: float | None = None,
    capture_scale: float | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    direct_values = np.asarray(direct, dtype=float)
    capture_values = np.asarray(capture, dtype=float)
    margin_values = np.asarray(margin, dtype=float)
    if (
        not np.isfinite(direct_values).all()
        or not np.isfinite(capture_values).all()
        or not np.isfinite(margin_values).all()
    ):
        raise ValueError("interaction inputs must be finite")
    d_center = float(np.mean(direct_values)) if direct_center is None else float(direct_center)
    d_scale = (
        max(float(np.std(direct_values)), 1e-8)
        if direct_scale is None
        else float(direct_scale)
    )
    c_center = (
        float(np.mean(capture_values))
        if capture_center is None
        else float(capture_center)
    )
    c_scale = (
        max(float(np.std(capture_values)), 1e-8)
        if capture_scale is None
        else float(capture_scale)
    )
    if d_scale <= 0.0 or c_scale <= 0.0:
        raise ValueError("interaction standardization scales must be positive")
    direct_z = (direct_values - d_center) / d_scale
    capture_z = (capture_values - c_center) / c_scale
    distance = (
        float(contract["direction"])
        * (margin_values - float(contract["threshold"]))
        / float(contract["robust_scale"])
    )
    gate = 1.0 / (1.0 + np.exp(-np.clip(distance, -40.0, 40.0)))
    confidence = np.maximum(0.0, 0.5 * (direct_z + capture_z))
    score = direct_z + float(contract["interaction_weight"]) * (
        2.0 * gate - 1.0
    ) * confidence
    return score, {
        "direct_center": d_center,
        "direct_scale": d_scale,
        "capture_center": c_center,
        "capture_scale": c_scale,
    }


def select_seed_history(
    frame: pd.DataFrame,
    *,
    first_decision_exclusive: pd.Timestamp,
    lookback_days: int,
) -> pd.DataFrame:
    cutoff = _utc(first_decision_exclusive, name="first_decision_exclusive")
    resolution = pd.to_datetime(frame[RESOLUTION], utc=True, errors="raise")
    lower = cutoff - pd.Timedelta(days=int(lookback_days))
    selected = frame.loc[resolution.lt(cutoff) & resolution.ge(lower)].copy()
    if selected.empty:
        raise ValueError("causal calibrator seed has no resolved trailing history")
    if pd.to_datetime(selected[RESOLUTION], utc=True).max() >= cutoff:
        raise AssertionError("calibrator seed includes a forward-block outcome")
    return selected


def _load_final_head_models(
    root: Path,
    manifest: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    from catboost import CatBoostClassifier, CatBoostRegressor

    output: dict[str, dict[str, Any]] = {}
    for side in SIDES:
        side_manifest = manifest["sides"][side]
        direct_record = side_manifest["models"]["direct_exact_net_residual"]
        capture_record = side_manifest["models"]["capture_probability"]
        direct_path = Path(direct_record["path"])
        capture_path = Path(capture_record["path"])
        if not direct_path.is_absolute():
            direct_path = ROOT / direct_path
        if not capture_path.is_absolute():
            capture_path = ROOT / capture_path
        if _sha256(direct_path) != direct_record["sha256"]:
            raise ValueError(f"{side} direct final-head hash mismatch")
        if _sha256(capture_path) != capture_record["sha256"]:
            raise ValueError(f"{side} capture final-head hash mismatch")
        direct = CatBoostRegressor()
        capture = CatBoostClassifier()
        direct.load_model(direct_path)
        capture.load_model(capture_path)
        output[side] = {"direct": direct, "capture": capture}
    return output


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    cutoff = _utc(args.first_decision_exclusive, name="first_decision_exclusive")
    head_manifest_path = args.head_root / "manifest.json"
    head_contract_path = args.head_root / "feature_contract.json"
    head_manifest = json.loads(head_manifest_path.read_text(encoding="utf-8"))
    if head_manifest.get("schema") != "execution_ev_forward_final_heads_v1":
        raise ValueError("unexpected final-head manifest schema")
    feature_contract = json.loads(head_contract_path.read_text(encoding="utf-8"))
    if _sha256(head_contract_path) != head_manifest["feature_contract"]["sha256"]:
        raise ValueError("final-head feature contract hash mismatch")
    interaction = load_frozen_base_margin_interaction(args.base_margin_screen)
    frame = pd.read_parquet(args.input)
    if frame.duplicated(list(IDENTITY)).any():
        raise ValueError("calibrator input contains duplicate identities")
    for column in (DECISION, RESOLUTION):
        values = frame[column]
        if not isinstance(values.dtype, pd.DatetimeTZDtype):
            raise ValueError(f"{column} must be stored timezone-aware")
        frame[column] = values.dt.tz_convert("UTC")
    seed = select_seed_history(
        frame,
        first_decision_exclusive=cutoff,
        lookback_days=args.lookback_days,
    )
    models = _load_final_head_models(args.head_root, head_manifest)
    parts: list[pd.DataFrame] = []
    side_states: dict[str, Any] = {}
    for side in SIDES:
        local = seed.loc[seed["side_name"].astype(str).eq(side)].copy()
        if len(local) < args.minimum_side_rows:
            raise ValueError(f"{side} calibrator seed has insufficient support")
        features = list(feature_contract["feature_columns_by_side"][side])
        for column in features:
            prefix = "catboost_archetype__"
            if column.startswith(prefix) and column not in local:
                level = column[len(prefix) :]
                local[column] = (
                    local["catboost_archetype"].astype(str).eq(level).astype("float32")
                )
        missing = sorted(set(features).difference(local.columns))
        if missing:
            raise ValueError(f"{side} calibrator features missing: {missing}")
        x = local.loc[:, features].apply(pd.to_numeric, errors="raise")
        values = x.to_numpy(dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError(f"{side} calibrator features are not finite")
        direct = local[BASELINE].to_numpy(dtype=float) + np.asarray(
            models[side]["direct"].predict(x), dtype=float
        )
        capture = np.asarray(
            models[side]["capture"].predict_proba(x)[:, 1], dtype=float
        )
        score, standardization = interaction_score(
            direct,
            capture,
            local["base_margin_to_cutoff_z"].to_numpy(dtype=float),
            contract=interaction,
        )
        target = local[TARGET].to_numpy(dtype=float)
        mapper = IsotonicRegression(out_of_bounds="clip", increasing=True)
        mapper.fit(score, target)
        local["final_direct_net_raw"] = direct
        local["final_capture_probability"] = capture
        local["frozen_margin_capture_interaction_raw"] = score
        local["seed_mapped_execution_ev"] = mapper.predict(score)
        parts.append(
            local.loc[
                :,
                [
                    *IDENTITY,
                    DECISION,
                    RESOLUTION,
                    TARGET,
                    "final_direct_net_raw",
                    "final_capture_probability",
                    "frozen_margin_capture_interaction_raw",
                    "seed_mapped_execution_ev",
                ],
            ]
        )
        side_states[side] = {
            "rows": int(len(local)),
            "decision_min_utc": local[DECISION].min(),
            "decision_max_utc": local[DECISION].max(),
            "resolved_label_min_utc": local[RESOLUTION].min(),
            "resolved_label_max_utc": local[RESOLUTION].max(),
            "standardization": standardization,
            "isotonic": {
                "x_thresholds": mapper.X_thresholds_.tolist(),
                "y_thresholds": mapper.y_thresholds_.tolist(),
                "out_of_bounds": "clip",
                "increasing": True,
            },
        }
    history = pd.concat(parts, ignore_index=True).sort_values(
        [RESOLUTION, "candidate_id"], kind="stable"
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    history_path = args.output_dir / "seed_history.parquet"
    history.to_parquet(history_path, index=False, compression="zstd")
    state_path = args.output_dir / "causal_recent_ev_state.json"
    state = {
        "schema": SCHEMA,
        "status": "frozen_operational_seed_not_oos_evidence",
        "mapping": "causal_recent_side_isotonic_ev_21d",
        "lookback_days": int(args.lookback_days),
        "first_decision_exclusive_utc": cutoff,
        "resolved_label_max_utc": history[RESOLUTION].max(),
        "sequential_updates_only_after_resolution": True,
        "update_rule": (
            "before each decision t, retain only seed/forward rows with "
            "execution_label_end_utc < t and >= t-21d; refit the same side-local "
            "increasing isotonic map; never change features, models, interaction, "
            "lookback, admission floors, or ranking scope"
        ),
        "interaction": interaction,
        "sides": side_states,
        "sources": {
            "input": {"path": args.input, "sha256": _sha256(args.input)},
            "head_manifest": {
                "path": head_manifest_path,
                "sha256": _sha256(head_manifest_path),
            },
            "feature_contract": {
                "path": head_contract_path,
                "sha256": _sha256(head_contract_path),
            },
            "base_margin_screen": {
                "path": args.base_margin_screen,
                "sha256": _sha256(args.base_margin_screen),
            },
        },
        "history": {
            "path": history_path,
            "sha256": _sha256(history_path),
            "rows": int(len(history)),
        },
    }
    _write_json(state_path, state)
    manifest = {
        "schema": "execution_ev_forward_calibrator_seed_manifest_v1",
        "status": state["status"],
        "state": {"path": state_path, "sha256": _sha256(state_path)},
        "history": state["history"],
        "resolved_label_max_utc": state["resolved_label_max_utc"],
        "side_rows": {side: side_states[side]["rows"] for side in SIDES},
    }
    _write_json(args.output_dir / "manifest.json", manifest)
    return manifest


def _parser(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--head-root", type=Path, default=DEFAULT_HEAD_ROOT)
    parser.add_argument("--base-margin-screen", type=Path, default=DEFAULT_SCREEN)
    parser.add_argument(
        "--first-decision-exclusive",
        default="2026-07-27T23:59:59.999999Z",
    )
    parser.add_argument("--lookback-days", type=int, default=21)
    parser.add_argument("--minimum-side-rows", type=int, default=500)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


if __name__ == "__main__":
    result = run(_parser())
    print(json.dumps(result, indent=2, default=str))
