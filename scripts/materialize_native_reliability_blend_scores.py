#!/usr/bin/env python3
"""Materialize reliability-blend scores from persisted native components.

This is the deployable scoring path for the researched reliability blend.  It
expects a component feature ledger containing the same feature columns recorded
in the native component bundle.  The distilled student scorer remains an
audit/fallback tool and is intentionally not used here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_fixed_tpsl_blend_simple_policy_optimiser import STRATEGY_IDS  # noqa: E402

BLEND_NEW_HARD = "B2_new_period_hard_qfail"
BLEND_NEW_SOFT = "B3_new_period_soft_qfail"
HEAD_BY_STRATEGY_ID = {strategy_id: head for head, strategy_id in STRATEGY_IDS.items()}


def _shape_rank(values: np.ndarray, *, power: float, side: str) -> np.ndarray:
    x = np.clip(np.asarray(values, dtype=np.float32), 0.0, 1.0)
    p = min(2.0, max(0.5, float(power) if np.isfinite(float(power)) else 1.0))
    if str(side) == "low":
        return np.power(1.0 - x, p).astype(np.float32, copy=False)
    return np.power(x, p).astype(np.float32, copy=False)


def _blend_score(
    anchor_rank: np.ndarray,
    period_rank: np.ndarray,
    qfail_rank: np.ndarray,
    alpha: float,
    beta: float,
    *,
    period_power: float = 1.0,
    period_side: str = "high",
    qfail_power: float = 1.0,
    qfail_side: str = "high",
) -> np.ndarray:
    return (
        np.asarray(anchor_rank, dtype=np.float32)
        + float(alpha) * _shape_rank(period_rank, power=period_power, side=period_side)
        + float(beta) * _shape_rank(qfail_rank, power=qfail_power, side=qfail_side)
    ).astype(np.float32, copy=False)


def _component_ablation_scores(
    anchor_rank: np.ndarray,
    period_rank: np.ndarray,
    qfail_rank: np.ndarray,
    config: dict[str, Any],
) -> dict[str, np.ndarray]:
    alpha = float(config.get("alpha", 0.0))
    beta = float(config.get("beta", 0.0))
    period_power = float(config.get("period_power", 1.0))
    period_side = str(config.get("period_side", "high"))
    qfail_power = float(config.get("qfail_power", 1.0))
    qfail_side = str(config.get("qfail_side", "high"))
    zero = np.zeros(len(anchor_rank), dtype=np.float32)
    return {
        "reliability_anchor_only_score": np.asarray(anchor_rank, dtype=np.float32),
        "reliability_anchor_qfail_score": _blend_score(
            anchor_rank,
            zero,
            qfail_rank,
            0.0,
            beta,
            period_power=period_power,
            period_side=period_side,
            qfail_power=qfail_power,
            qfail_side=qfail_side,
        ),
        "reliability_anchor_period_score": _blend_score(
            anchor_rank,
            period_rank,
            zero,
            alpha,
            0.0,
            period_power=period_power,
            period_side=period_side,
            qfail_power=qfail_power,
            qfail_side=qfail_side,
        ),
        "reliability_blend_score": _blend_score(
            anchor_rank,
            period_rank,
            qfail_rank,
            alpha,
            beta,
            period_power=period_power,
            period_side=period_side,
            qfail_power=qfail_power,
            qfail_side=qfail_side,
        ),
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(float(value)) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _file_sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _scores_to_reference_rank(scores: np.ndarray, reference: dict[str, Any]) -> np.ndarray:
    ref = np.asarray(reference.get("scores", []), dtype=np.float64)
    ref = ref[np.isfinite(ref)]
    ref.sort()
    out = np.full(len(scores), np.nan, dtype=np.float32)
    finite = np.isfinite(scores)
    if ref.size == 0:
        return out
    out[finite] = (
        np.searchsorted(ref, np.asarray(scores, dtype=np.float64)[finite], side="right")
        / float(ref.size)
    ).astype(np.float32, copy=False)
    return out


def _score_distribution(values: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {
            "n": int(arr.size),
            "finite": 0,
            "min": None,
            "max": None,
            "std": None,
            "unique_rounded_6": 0,
            "collapsed": True,
        }
    std = float(np.nanstd(finite))
    unique = int(pd.Series(finite).round(6).nunique(dropna=True))
    return {
        "n": int(arr.size),
        "finite": int(finite.size),
        "min": float(np.nanmin(finite)),
        "max": float(np.nanmax(finite)),
        "std": std,
        "unique_rounded_6": unique,
        "q01": float(np.nanquantile(finite, 0.01)),
        "q50": float(np.nanquantile(finite, 0.50)),
        "q99": float(np.nanquantile(finite, 0.99)),
        "collapsed": bool(std < 1e-7 or unique <= 1),
    }


def _window_rank_debug(scores: np.ndarray) -> np.ndarray:
    return pd.Series(scores).rank(method="average", pct=True).to_numpy(dtype=np.float32)


def _prepare_feature_frame(rows: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    missing = [c for c in columns if c not in rows.columns]
    if missing:
        raise RuntimeError(f"component feature ledger missing required columns: {missing[:30]}")
    out = pd.DataFrame(index=rows.index)
    for col in columns:
        out[col] = pd.to_numeric(rows[col], errors="coerce").replace([np.inf, -np.inf], np.nan).astype("float32")
    return out


def _ensure_head_column(rows: pd.DataFrame) -> pd.DataFrame:
    out = rows.copy()
    if "head" in out.columns and out["head"].notna().any():
        out["head"] = out["head"].astype(str)
        return out
    if "strategy_id" not in out.columns:
        raise RuntimeError("component feature ledger missing 'head' and 'strategy_id'")
    out["strategy_id"] = out["strategy_id"].astype(str)
    out["head"] = out["strategy_id"].map(HEAD_BY_STRATEGY_ID)
    missing = out["head"].isna()
    if bool(missing.any()):
        sample = sorted(out.loc[missing, "strategy_id"].dropna().astype(str).unique().tolist())[:10]
        raise RuntimeError(
            "component feature ledger contains strategy_id values that cannot be mapped to a reliability head: "
            f"{sample}"
        )
    return out


def _select_component_model(
    models: list[dict[str, Any]],
    *,
    component: str,
    allow_oof_fold_models: bool,
) -> dict[str, Any] | None:
    candidates = [m for m in models if str(m.get("component")) == str(component)]
    full_fit = [
        m
        for m in candidates
        if str(m.get("fold", "")).lower() == "full_fit"
        or str(m.get("model_scope", "")).lower() == "full_fit"
    ]
    if full_fit:
        return full_fit[-1]
    if candidates and allow_oof_fold_models:
        return candidates[-1]
    return None


def _predict_component(
    rows: pd.DataFrame,
    artifact: dict[str, Any] | None,
    *,
    component_score_col: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    if component_score_col in rows.columns:
        score = pd.to_numeric(rows[component_score_col], errors="coerce").to_numpy(dtype=np.float32)
        return score, {"score_source": "precomputed_component_score", "component_score_col": component_score_col}
    if artifact is None:
        raise RuntimeError(
            f"No full-fit model artifact and no precomputed {component_score_col!r} column."
        )
    if str(artifact.get("backend")) == "constant":
        fill = float(artifact.get("fill_value", np.nan))
        return np.full(len(rows), fill, dtype=np.float32), {
            "score_source": "constant_component_artifact",
            "backend": artifact.get("backend"),
        }
    model = artifact.get("model")
    if model is None:
        raise RuntimeError(f"component artifact for {component_score_col} has no model")
    columns = [str(c) for c in (artifact.get("input_feature_columns") or [])]
    x = _prepare_feature_frame(rows, columns)
    pred = np.asarray(model.predict(x), dtype=np.float32)
    pred = np.clip(pred, 0.0, 1.0).astype(np.float32, copy=False)
    return pred, {
        "score_source": "native_component_model",
        "backend": artifact.get("backend"),
        "feature_count": int(len(columns)),
        "selected_feature_count": int(len(artifact.get("selected_features") or [])),
    }


def _default_config_by_head(bundle: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in bundle.get("default_deployable_config_by_head", []) or []:
        if isinstance(row, dict) and row.get("head"):
            out[str(row["head"])] = row
    if out:
        return out
    for row in bundle.get("blend_winners", []) or []:
        if not isinstance(row, dict) or not row.get("head"):
            continue
        if str(row.get("variant")) == BLEND_NEW_SOFT:
            out.setdefault(str(row["head"]), row)
    if out:
        return out
    for row in bundle.get("default_soft_qfail_config_by_head", []) or []:
        if isinstance(row, dict) and row.get("head"):
            out[str(row["head"])] = row
    if out:
        return out
    for row in bundle.get("blend_winners", []) or []:
        if not isinstance(row, dict) or not row.get("head"):
            continue
        if str(row.get("variant")) in {BLEND_NEW_SOFT, BLEND_NEW_HARD}:
            out.setdefault(str(row["head"]), row)
    return out


def _component_names_for_variant(variant: str) -> tuple[str, str]:
    if str(variant) == BLEND_NEW_HARD:
        return "period_new_score", "qfail_hard_score"
    # The production default is soft q_fail.  Old-period variants may still be
    # used if their precomputed score columns are supplied in the feature ledger.
    if "old_period" in str(variant):
        period = "period_old_score"
    else:
        period = "period_new_score"
    qfail = "qfail_soft_score" if "soft_qfail" in str(variant) else "qfail_hard_score"
    return period, qfail


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component-model-bundle", type=Path, required=True)
    parser.add_argument("--component-feature-ledger", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-oof-fold-models", action="store_true")
    parser.add_argument("--allow-window-rank-debug", action="store_true")
    parser.add_argument(
        "--fail-on-collapsed-components",
        action="store_true",
        help="Fail closed if period/qfail native component scores or persisted-reference ranks are collapsed.",
    )
    args = parser.parse_args()

    bundle = joblib.load(args.component_model_bundle)
    rows = pd.read_parquet(args.component_feature_ledger).copy()
    if rows.empty:
        raise RuntimeError("component feature ledger is empty")
    rows = _ensure_head_column(rows)
    for col in ("head", "timestamp", "symbol"):
        if col not in rows.columns:
            raise RuntimeError(f"component feature ledger missing {col!r}")
    rows["timestamp"] = pd.to_datetime(rows["timestamp"], utc=True, errors="coerce")
    configs = _default_config_by_head(bundle)
    output_frames: list[pd.DataFrame] = []
    diagnostics: list[dict[str, Any]] = []

    for head, group in rows.groupby(rows["head"].astype(str), sort=True):
        head_bundle = dict((bundle.get("heads", {}) or {}).get(str(head), {}) or {})
        config = configs.get(str(head))
        if not config:
            diagnostics.append({"head": head, "status": "missing_blend_config"})
            continue
        variant = str(config.get("variant", ""))
        period_col, qfail_col = _component_names_for_variant(variant)
        model_artifacts = list(head_bundle.get("models", []) or [])
        period_artifact = _select_component_model(
            model_artifacts,
            component="new_period",
            allow_oof_fold_models=bool(args.allow_oof_fold_models),
        )
        qfail_artifact = _select_component_model(
            model_artifacts,
            component="qfail_soft" if qfail_col == "qfail_soft_score" else "qfail_hard",
            allow_oof_fold_models=bool(args.allow_oof_fold_models),
        )
        if period_col == "period_old_score":
            period_artifact = None
        period_score, period_diag = _predict_component(
            group,
            period_artifact,
            component_score_col=period_col,
        )
        qfail_score, qfail_diag = _predict_component(
            group,
            qfail_artifact,
            component_score_col=qfail_col,
        )
        if "anchor_score" not in group.columns:
            raise RuntimeError("component feature ledger missing 'anchor_score'")
        anchor_score = pd.to_numeric(group["anchor_score"], errors="coerce").to_numpy(dtype=np.float32)
        refs = dict(head_bundle.get("component_rank_references", {}) or {})
        if refs:
            anchor_rank = _scores_to_reference_rank(anchor_score, refs.get("anchor_score", {}))
            period_rank = _scores_to_reference_rank(period_score, refs.get(period_col, {}))
            qfail_rank = _scores_to_reference_rank(qfail_score, refs.get(qfail_col, {}))
            rank_source = "persisted_component_score_reference"
        elif args.allow_window_rank_debug:
            anchor_rank = _window_rank_debug(anchor_score)
            period_rank = _window_rank_debug(period_score)
            qfail_rank = _window_rank_debug(qfail_score)
            rank_source = "window_rank_debug_not_deployable"
        else:
            raise RuntimeError(
                f"Missing persisted component rank references for {head}. "
                "Refusing to rank over the current ledger."
            )
        arm_scores = _component_ablation_scores(anchor_rank, period_rank, qfail_rank, config)
        score = arm_scores["reliability_blend_score"]
        finite = (
            np.isfinite(anchor_rank)
            & np.isfinite(period_rank)
            & np.isfinite(qfail_rank)
            & np.isfinite(score)
        )
        if not bool(np.all(finite)):
            raise RuntimeError(f"Non-finite native reliability score inputs for {head}: {int((~finite).sum())} rows")
        out = group[["timestamp", "symbol", "head"]].copy()
        if "strategy_id" in group.columns:
            out["strategy_id"] = group["strategy_id"].astype(str).to_numpy()
        out["anchor_score"] = anchor_score
        out["anchor_component_rank"] = anchor_rank
        out["period_component_score"] = period_score
        out["period_component_rank"] = period_rank
        out["qfail_component_score"] = qfail_score
        out["qfail_component_rank"] = qfail_rank
        for arm_col, arm_score in arm_scores.items():
            out[arm_col] = arm_score
        out["blend_variant"] = variant
        out["score_source"] = "native_component_reliability_blend"
        out["component_rank_source"] = rank_source
        output_frames.append(out)
        score_distribution = {
            "anchor_score": _score_distribution(anchor_score),
            "anchor_rank": _score_distribution(anchor_rank),
            "period_score": _score_distribution(period_score),
            "period_rank": _score_distribution(period_rank),
            "qfail_score": _score_distribution(qfail_score),
            "qfail_rank": _score_distribution(qfail_rank),
            "anchor_only_score": _score_distribution(arm_scores["reliability_anchor_only_score"]),
            "anchor_qfail_score": _score_distribution(arm_scores["reliability_anchor_qfail_score"]),
            "anchor_period_score": _score_distribution(arm_scores["reliability_anchor_period_score"]),
            "blend_score": _score_distribution(arm_scores["reliability_blend_score"]),
        }
        if bool(args.fail_on_collapsed_components):
            collapsed = [
                name
                for name in ("period_score", "period_rank", "qfail_score", "qfail_rank")
                if bool(score_distribution.get(name, {}).get("collapsed", False))
            ]
            if collapsed:
                raise RuntimeError(
                    f"Collapsed native reliability component outputs for {head}: {collapsed}. "
                    "Refusing deployable B0 scoring."
                )
        diagnostics.append(
            {
                "head": head,
                "status": "ok",
                "rows": int(len(out)),
                "variant": variant,
                "period_component": period_col,
                "qfail_component": qfail_col,
                "period_diag": period_diag,
                "qfail_diag": qfail_diag,
                "component_rank_source": rank_source,
                "allow_oof_fold_models": bool(args.allow_oof_fold_models),
                "score_distribution": score_distribution,
            }
        )

    if not output_frames:
        raise RuntimeError("No native reliability-blend scores were produced")
    scores = pd.concat(output_frames, axis=0, ignore_index=True)
    scores = scores.sort_values(["timestamp", "head", "symbol"], kind="mergesort")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    score_path = args.output_dir / "native_reliability_blend_scores.parquet"
    scores.to_parquet(score_path, index=False)
    diag_path = args.output_dir / "native_reliability_blend_score_diagnostics.csv"
    pd.DataFrame(diagnostics).to_csv(diag_path, index=False)
    manifest = {
        "generated_by": "materialize_native_reliability_blend_scores",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "score_path": str(score_path),
        "diagnostics_path": str(diag_path),
        "component_model_bundle": str(args.component_model_bundle),
        "component_model_bundle_sha256": _file_sha256(args.component_model_bundle),
        "component_feature_ledger": str(args.component_feature_ledger),
        "component_feature_ledger_sha256": _file_sha256(args.component_feature_ledger),
        "rows": int(len(scores)),
        "timestamp_min": scores["timestamp"].min().isoformat(),
        "timestamp_max": scores["timestamp"].max().isoformat(),
        "deployment_status": (
            "native_component_path"
            if not args.allow_oof_fold_models
            else "audit_only_oof_fold_component_models"
        ),
        "rank_policy": "persisted component score references; no current-window rank unless explicit debug flag",
        "score_arms": [
            "reliability_anchor_only_score",
            "reliability_anchor_qfail_score",
            "reliability_anchor_period_score",
            "reliability_blend_score",
        ],
        "diagnostics": diagnostics,
    }
    (args.output_dir / "native_reliability_blend_score_manifest.json").write_text(
        json.dumps(_json_safe(manifest), indent=2) + "\n"
    )
    print(json.dumps(_json_safe(manifest), indent=2)[:6000])


if __name__ == "__main__":
    main()
