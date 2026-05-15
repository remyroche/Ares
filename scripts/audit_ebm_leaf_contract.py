#!/usr/bin/env python
from __future__ import annotations

import argparse
import glob
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from extreme_price_movements.ebm_on_lgbm import (  # noqa: E402
    _compute_oof_bundle_tree_frame,
    _compute_soft_tree_features_ebm,
    iter_ebm_models,
    summarize_ebm_leaf_contract,
)

OOF_PRED_COLUMNS = [
    "oof_pred",
    "clf",
    "oof_prob",
    "oof_prob_ebm_raw",
    "oof_prob_uncertainty_weighted",
]


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        out = float(value)
        return out if np.isfinite(out) else None
    if value is None or isinstance(value, (str, int, float)):
        return value
    return str(value)


def _sample(values: set[str] | list[str], n: int = 20) -> list[str]:
    return sorted(map(str, values))[:n]


def _synthetic_raw_frame(model: Any, rows: int = 3) -> pd.DataFrame:
    raw = [str(c) for c in (getattr(model, "raw_selected_features", []) or [])]
    return pd.DataFrame(np.ones((rows, len(raw)), dtype=np.float32), columns=raw)


def _compute_tree_frame(
    model: Any, x_raw: pd.DataFrame
) -> tuple[pd.DataFrame, str | None]:
    tree_names = [str(c) for c in (getattr(model, "tree_feature_names", []) or [])]
    tree_config = getattr(model, "tree_feature_config", {}) or {}
    try:
        if isinstance(tree_config, dict) and tree_config.get("oof_tree_features"):
            return (
                _compute_oof_bundle_tree_frame(
                    tree_config,
                    x_raw,
                    selected_tree_names=tree_names,
                ),
                None,
            )
        tree_models = list(getattr(model, "tree_models", []) or [])
        if not tree_models:
            return pd.DataFrame(index=x_raw.index), None
        arr, emitted_names, _ = _compute_soft_tree_features_ebm(
            tree_models,
            x_raw.to_numpy(dtype=np.float32),
            getattr(model, "tree_feature_scales", None),
            selected_names=set(tree_names),
        )
        return pd.DataFrame(arr, columns=emitted_names, index=x_raw.index), None
    except Exception as exc:  # audit must report the contract error, not hide it
        return pd.DataFrame(index=x_raw.index), f"{exc.__class__.__name__}: {exc}"


def _probe_raw_zero_fill(model: Any) -> dict[str, Any]:
    raw = [str(c) for c in (getattr(model, "raw_selected_features", []) or [])]
    out: dict[str, Any] = {
        "raw_missing_probe_silent_zero_fill": None,
        "raw_missing_probe_feature": None,
        "raw_missing_probe_error": None,
    }
    if not raw:
        return out
    missing_col = raw[0]
    out["raw_missing_probe_feature"] = missing_col
    x_full = _synthetic_raw_frame(model)
    x_missing = x_full.drop(columns=[missing_col])
    try:
        frame_full = model._frame(x_full)
        frame_missing = model._frame(x_missing)
        finite = bool(np.isfinite(frame_missing.to_numpy(dtype=np.float32)).all())
        differs = frame_full.shape != frame_missing.shape or not np.allclose(
            frame_full.to_numpy(dtype=np.float32),
            frame_missing.to_numpy(dtype=np.float32),
            equal_nan=True,
        )
        zero_col = missing_col in frame_missing.columns and bool(
            np.allclose(frame_missing[missing_col].to_numpy(dtype=np.float32), 0.0)
        )
        out["raw_missing_probe_silent_zero_fill"] = bool(
            finite and (zero_col or differs)
        )
    except Exception as exc:
        out["raw_missing_probe_silent_zero_fill"] = False
        out["raw_missing_probe_error"] = f"{exc.__class__.__name__}: {exc}"
    return out


def _probe_policy_oof(
    model: Any,
    model_path: str,
    artifact_dir: Path,
    max_rows: int,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "policy_oof_replay_possible": False,
        "spearman_live_vs_oof": None,
        "mean_abs_diff_live_vs_oof": None,
        "live_pred_std": None,
        "oof_pred_std": None,
    }
    raw = [str(c) for c in (getattr(model, "raw_selected_features", []) or [])]
    meta_oof_dir = artifact_dir / "meta_oof"
    paths = sorted(glob.glob(str(meta_oof_dir / "meta_oof_*.parquet")))
    if not paths:
        out["policy_oof_interpretation"] = "no matching meta_oof parquet found"
        return out

    tokens = [
        t
        for t in model_path.replace("'", "").replace("]", "").split("[")
        if len(t) >= 2
    ]
    ranked = sorted(
        paths,
        key=lambda p: any(token in Path(p).name for token in tokens),
        reverse=True,
    )
    for parquet_path in ranked:
        try:
            df = pd.read_parquet(parquet_path).head(max_rows)
        except Exception as exc:
            out["policy_oof_interpretation"] = f"could not read {parquet_path}: {exc}"
            continue
        pred_col = next((c for c in OOF_PRED_COLUMNS if c in df.columns), None)
        if pred_col is None:
            continue
        missing_raw = [c for c in raw if c not in df.columns]
        if missing_raw:
            out.update(
                {
                    "missing_raw_features_in_meta_oof_count": len(missing_raw),
                    "missing_raw_features_in_meta_oof_sample": missing_raw[:20],
                    "policy_oof_interpretation": (
                        "simple_policy_optimiser cannot prove live EBM _frame parity from this parquet"
                    ),
                }
            )
            return out
        try:
            live_pred = np.asarray(model.predict(df[raw]), dtype=np.float32)
            oof_pred = pd.to_numeric(df[pred_col], errors="coerce").to_numpy(
                dtype=np.float32
            )
            n = min(len(live_pred), len(oof_pred))
            if n == 0:
                continue
            live_pred = live_pred[:n]
            oof_pred = oof_pred[:n]
            out["policy_oof_replay_possible"] = True
            out["policy_oof_path"] = str(parquet_path)
            out["policy_oof_pred_column"] = pred_col
            out["spearman_live_vs_oof"] = float(
                pd.Series(live_pred).corr(pd.Series(oof_pred), method="spearman")
            )
            out["mean_abs_diff_live_vs_oof"] = float(
                np.mean(np.abs(live_pred - oof_pred))
            )
            out["live_pred_std"] = float(np.std(live_pred))
            out["oof_pred_std"] = float(np.std(oof_pred))
            return out
        except Exception as exc:
            out[
                "policy_oof_interpretation"
            ] = f"live replay failed for {parquet_path}: {exc}"
            return out
    out[
        "policy_oof_interpretation"
    ] = "no usable OOF prediction column found in meta_oof parquet files"
    return out


def audit_model(
    model_path: str, model: Any, artifact_dir: Path, max_rows: int
) -> dict[str, Any]:
    row = summarize_ebm_leaf_contract(model_path, model)
    row.update(_probe_raw_zero_fill(model))

    raw = [str(c) for c in (getattr(model, "raw_selected_features", []) or [])]
    selected = [str(c) for c in (getattr(model, "selected_features", []) or [])]
    tree_names = [str(c) for c in (getattr(model, "tree_feature_names", []) or [])]
    x_full = _synthetic_raw_frame(model)
    x_raw = x_full.reindex(columns=raw, fill_value=0.0)
    tree_df, tree_error = _compute_tree_frame(model, x_raw)
    requested = set(tree_names)
    emitted = set(map(str, tree_df.columns))
    missing_tree = requested - emitted

    materializable = set(raw) | emitted
    selected_missing = set(selected) - materializable
    leaf_state_missing = False
    if row["selected_lgbm_features_n"] > 0:
        leaf_state_missing = bool(
            row["raw_selected_features_n"] == 0
            or row["tree_feature_names_n"] == 0
            or (row["tree_models_n"] == 0 and not row["has_oof_tree_features"])
            or (not row["has_tree_feature_scales"] and not row["has_oof_tree_features"])
        )

    row.update(
        {
            "tree_features_requested_n": len(requested),
            "tree_features_emitted_n": len(emitted),
            "tree_features_missing_count": len(missing_tree),
            "tree_features_missing_sample": _sample(missing_tree),
            "tree_features_extra_count": len(emitted - requested),
            "tree_feature_coverage": 1.0 - (len(missing_tree) / max(len(requested), 1)),
            "tree_feature_regeneration_error": tree_error,
            "selected_missing_after_frame_count": 0,
            "selected_missing_before_zero_fill_count": len(selected_missing),
            "selected_missing_before_zero_fill_sample": _sample(selected_missing),
            "leaf_transform_state_missing": leaf_state_missing,
            "orchestrator_contract_enforced": True,
        }
    )

    try:
        frame = model._frame(x_full)
        row["selected_missing_after_frame_count"] = len(
            set(selected) - set(map(str, frame.columns))
        )
    except Exception as exc:
        row["frame_materialization_error"] = f"{exc.__class__.__name__}: {exc}"

    row.update(_probe_policy_oof(model, model_path, artifact_dir, max_rows))

    fail_reasons: list[str] = []
    warning_reasons: list[str] = []
    if leaf_state_missing:
        fail_reasons.append("missing_leaf_transform_state")
    if tree_names and missing_tree:
        fail_reasons.append("selected_tree_features_not_regenerated")
    if selected_missing:
        fail_reasons.append("selected_features_only_materializable_by_zero_fill")
    if row.get("raw_missing_probe_silent_zero_fill"):
        warning_reasons.append("raw_missing_feature_silent_zero_fill")
    if not row.get("policy_oof_replay_possible"):
        warning_reasons.append("policy_oof_replay_unavailable")
    row["fail_reasons"] = fail_reasons
    row["warning_reasons"] = warning_reasons
    row["status"] = "fail" if fail_reasons else ("warning" if warning_reasons else "ok")
    return row


def run_audit(data_root: Path, run_id: str, max_rows: int) -> dict[str, Any]:
    artifact_dir = data_root / "artifacts" / run_id
    state_path = artifact_dir / "models" / "trained_state.pkl"
    if not state_path.exists():
        raise FileNotFoundError(f"trained_state.pkl not found: {state_path}")
    with state_path.open("rb") as fh:
        state = pickle.load(fh)
    models = list(iter_ebm_models(state))
    rows = [audit_model(path, model, artifact_dir, max_rows) for path, model in models]
    return {
        "run_id": run_id,
        "state_path": str(state_path),
        "models_found": len(rows),
        "models": rows,
        "status": "fail" if any(r.get("status") == "fail" for r in rows) else "ok",
    }


def print_table(report: dict[str, Any]) -> None:
    rows = report.get("models", [])
    print(
        f"EBM leaf-contract audit: run_id={report.get('run_id')} models={len(rows)} status={report.get('status')}"
    )
    if not rows:
        return
    header = f"{'status':8} {'path':48} {'raw':>5} {'tree':>5} {'miss_tree':>9} {'miss_sel':>8} {'state':>6}"
    print(header)
    print("-" * len(header))
    for row in rows:
        path = str(row.get("model_path", ""))[-48:]
        print(
            f"{row.get('status',''):8} {path:48} "
            f"{row.get('raw_selected_features_n', 0):5} "
            f"{row.get('tree_feature_names_n', 0):5} "
            f"{row.get('tree_features_missing_count', 0):9} "
            f"{row.get('selected_missing_before_zero_fill_count', 0):8} "
            f"{str(not row.get('leaf_transform_state_missing')):>6}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit persisted EBM-on-LGBM leaf feature contracts."
    )
    parser.add_argument("--data-root", default="data", type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--max-rows", default=256, type=int)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    report = run_audit(
        args.data_root, args.run_id, max(1, min(int(args.max_rows), 256))
    )
    print_table(report)
    out_path = args.json_out or (
        args.data_root
        / "artifacts"
        / args.run_id
        / "diagnostics"
        / "ebm_leaf_contract_audit.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n")
    print(f"Wrote {out_path}")
    return 1 if report.get("status") == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
