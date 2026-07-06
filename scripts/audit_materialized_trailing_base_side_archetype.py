#!/usr/bin/env python3
"""Audit a materialized trailing-label base HPO winner by side x archetype."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_first_touch_label_training_smoke import _target_from_frame  # noqa: E402
from scripts.run_label_quality_proxy_diagnostics import (  # noqa: E402
    DEFAULT_FEATURE_DIR,
    DEFAULT_FEATURE_LIST_CSV,
    _json_safe,
    _rank_top_indices,
    _safe_mean,
    _safe_quantile,
)
from scripts.run_label_weighted_proxy_ablation import WEIGHT_ARMS, _effective_sample_size, _weight_series  # noqa: E402
from scripts.run_materialized_trailing_label_topk_lgbm_hpo import (  # noqa: E402
    TOP_FRACS,
    _cap_rows,
    _fit_predict_lgbm,
    _parse_csv,
    _prepare_folds,
)


DEFAULT_OUTPUT_DIR = Path("data_perp/reports/materialized_trailing_label_base_side_archetype_audit_v1")
PARAM_KEYS = (
    "n_estimators",
    "learning_rate",
    "num_leaves",
    "max_depth",
    "min_child_samples",
    "subsample",
    "colsample_bytree",
    "reg_alpha",
    "reg_lambda",
    "target_mode",
    "weight_arm",
)


def _safe_numeric(values: Any) -> pd.Series:
    if isinstance(values, pd.Series):
        return pd.to_numeric(values, errors="coerce")
    return pd.to_numeric(pd.Series(values), errors="coerce")


def _side_name(frame: pd.DataFrame, metrics: pd.DataFrame) -> pd.Series:
    if "side_name" in frame.columns:
        return frame["side_name"].astype(str).replace({"nan": ""}).where(frame["side_name"].notna(), "")
    if "__side__" in frame.columns:
        side = _safe_numeric(frame["__side__"]).fillna(1.0)
    elif "side" in frame.columns:
        side = _safe_numeric(frame["side"]).fillna(1.0)
    else:
        side = _safe_numeric(metrics.get("side", pd.Series(1.0, index=frame.index))).fillna(1.0)
    return pd.Series(np.where(side.to_numpy(dtype=float) < 0.0, "short", "long"), index=frame.index)


def _archetype(frame: pd.DataFrame) -> pd.Series:
    for col in ("__archetype_label_family__", "__regime_family__"):
        if col in frame.columns:
            return frame[col].astype(str).fillna("unknown")
    return pd.Series("unknown", index=frame.index, dtype=object)


def _group_metrics(group: pd.DataFrame, *, scope: str, month: str, top_frac: float) -> dict[str, Any]:
    net = _safe_numeric(group["first_touch_net"]).fillna(0.0)
    cost = _safe_numeric(group.get("round_trip_cost", pd.Series(0.0, index=group.index))).fillna(0.0)
    gross = (net + cost).clip(lower=0.0)
    clean = _safe_numeric(group["clean_first_touch_exec"]).fillna(0.0).clip(0.0, 1.0)
    denom = float(gross.sum())
    symbols = int(group["symbol"].nunique(dropna=True)) if "symbol" in group.columns else 0
    top_symbol_share = (
        float(group["symbol"].astype(str).value_counts(normalize=True).iloc[0])
        if len(group) and "symbol" in group.columns
        else float("nan")
    )
    return {
        "scope": str(scope),
        "month": str(month),
        "top_frac": float(top_frac),
        "side": str(group["side_name"].iloc[0]) if len(group) else "",
        "archetype": str(group["archetype"].iloc[0]) if len(group) else "",
        "selected_rows": int(len(group)),
        "selected_symbols": symbols,
        "top_symbol_share": top_symbol_share,
        "gross_ev_weighted_clean_precision": float((clean * gross).sum() / denom) if denom > 0.0 else float("nan"),
        "clean_precision": _safe_mean(clean),
        "mean_first_touch_net": _safe_mean(net),
        "mean_first_touch_gross": _safe_mean(net + cost),
        "q10_first_touch_net": _safe_quantile(net, 0.10),
        "hit_first_touch_net": _safe_mean(net > 0.0),
        "first_touch_stop_rate": _safe_mean(group["first_touch_stop"]),
        "first_touch_timeout_rate": _safe_mean(group["first_touch_timeout"]),
        "first_touch_bad_mae_to_sl_rate": _safe_mean(_safe_numeric(group["first_touch_mae_to_sl"]).ge(1.0)),
        "p90_first_touch_mae_to_sl": _safe_quantile(group["first_touch_mae_to_sl"], 0.90),
        "p90_first_touch_bar": _safe_quantile(group["first_touch_bar"], 0.90),
        "mean_score": _safe_mean(group["score"]),
        "q10_score": _safe_quantile(group["score"], 0.10),
    }


def _availability_metrics(frame: pd.DataFrame, *, scope: str, month: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if frame.empty:
        return rows
    for (side, archetype), group in frame.groupby(["side_name", "archetype"], dropna=False):
        net = _safe_numeric(group["first_touch_net"]).fillna(0.0)
        clean = _safe_numeric(group["clean_first_touch_exec"]).fillna(0.0).clip(0.0, 1.0)
        rows.append(
            {
                "scope": str(scope),
                "month": str(month),
                "side": str(side),
                "archetype": str(archetype),
                "available_rows": int(len(group)),
                "available_symbols": int(group["symbol"].nunique(dropna=True)),
                "available_clean_rate": _safe_mean(clean),
                "available_mean_first_touch_net": _safe_mean(net),
                "available_q10_first_touch_net": _safe_quantile(net, 0.10),
                "available_timeout_rate": _safe_mean(group["first_touch_timeout"]),
                "available_bad_mae_to_sl_rate": _safe_mean(_safe_numeric(group["first_touch_mae_to_sl"]).ge(1.0)),
            }
        )
    return rows


def _extract_params(best: dict[str, Any]) -> dict[str, Any]:
    params = {key: best[key] for key in PARAM_KEYS if key in best}
    missing = sorted(set(PARAM_KEYS).difference(params))
    if missing:
        raise RuntimeError(f"Best params file missing keys: {missing}")
    for key in ("n_estimators", "num_leaves", "max_depth", "min_child_samples"):
        params[key] = int(float(params[key]))
    for key in ("learning_rate", "subsample", "colsample_bytree", "reg_alpha", "reg_lambda"):
        params[key] = float(params[key])
    params["target_mode"] = str(params["target_mode"])
    params["weight_arm"] = str(params["weight_arm"])
    return params


def run_audit(
    *,
    labels_path: Path,
    hpo_output_dir: Path,
    output_dir: Path,
    feature_dir: Path,
    feature_list_csv: Path,
    months: list[str],
    max_feature_store_features: int | None,
    max_train_rows: int,
    seed: int,
    include_ae_gmm_state_features: bool,
    ae_gmm_state_feature_max_train_rows: int,
    ae_gmm_state_feature_max_iter: int,
) -> dict[str, Any]:
    best_path = hpo_output_dir / "topk_lgbm_hpo_best.json"
    if not best_path.exists():
        raise FileNotFoundError(f"Missing HPO best params: {best_path}")
    best = json.loads(best_path.read_text(encoding="utf-8"))
    params = _extract_params(best)
    trial_number = int(float(best.get("trial_number", 0)))
    if params["weight_arm"] not in WEIGHT_ARMS:
        raise ValueError(f"Unknown weight arm: {params['weight_arm']}")

    output_dir.mkdir(parents=True, exist_ok=True)
    folds, prep_manifest = _prepare_folds(
        labels_path=labels_path,
        feature_dir=feature_dir,
        feature_list_csv=feature_list_csv,
        max_feature_store_features=max_feature_store_features,
        months=months,
        include_ae_gmm_state_features=include_ae_gmm_state_features,
        ae_gmm_state_feature_max_train_rows=ae_gmm_state_feature_max_train_rows,
        ae_gmm_state_feature_max_iter=ae_gmm_state_feature_max_iter,
        seed=seed,
    )
    if not folds:
        raise RuntimeError("No valid month-forward folds prepared")

    ledger_frames: list[pd.DataFrame] = []
    diagnostic_rows: list[dict[str, Any]] = []
    for fold_id, fold in enumerate(folds):
        train_target = _target_from_frame(fold["train"], fold["train_metrics"], target_mode=params["target_mode"])
        weights = _weight_series(
            frame=fold["train"],
            metrics=fold["train_metrics"],
            target=train_target,
            arm=params["weight_arm"],
        )
        idx = _cap_rows(len(fold["x_train"]), int(max_train_rows), int(seed) + 1009 * int(trial_number) + fold_id)
        pred = _fit_predict_lgbm(
            x_train=fold["x_train"].iloc[idx].reset_index(drop=True),
            y_train=train_target["target_soft"].iloc[idx].reset_index(drop=True),
            w_train=weights.iloc[idx].reset_index(drop=True),
            x_valid=fold["x_valid"],
            params=params,
            seed=int(seed) + 1000 * int(trial_number) + fold_id,
        )
        valid = fold["valid"].reset_index(drop=True)
        metrics = fold["valid_metrics"].reset_index(drop=True)
        ledger = pd.DataFrame(
            {
                "month": str(fold["month"]),
                "timestamp": valid["__ts__"].astype(str) if "__ts__" in valid.columns else "",
                "symbol": valid["__symbol__"].astype(str) if "__symbol__" in valid.columns else "",
                "side_name": _side_name(valid, metrics).reset_index(drop=True),
                "archetype": _archetype(valid).reset_index(drop=True),
                "policy_key": valid.get("__archetype_policy_key__", pd.Series("", index=valid.index)).astype(str),
                "score": _safe_numeric(pred).reset_index(drop=True).astype(np.float32),
                "first_touch_net": _safe_numeric(metrics["first_touch_net"]).reset_index(drop=True).astype(np.float32),
                "round_trip_cost": _safe_numeric(metrics["round_trip_cost"]).reset_index(drop=True).astype(np.float32),
                "clean_first_touch_exec": _safe_numeric(metrics["clean_first_touch_exec"]).reset_index(drop=True).astype(
                    np.float32
                ),
                "first_touch_stop": _safe_numeric(metrics["first_touch_stop"]).reset_index(drop=True).astype(np.float32),
                "first_touch_timeout": _safe_numeric(metrics["first_touch_timeout"]).reset_index(drop=True).astype(
                    np.float32
                ),
                "first_touch_mae_to_sl": _safe_numeric(metrics["first_touch_mae_to_sl"]).reset_index(drop=True).astype(
                    np.float32
                ),
                "first_touch_bar": _safe_numeric(metrics["first_touch_bar"]).reset_index(drop=True).astype(np.float32),
            }
        )
        for frac in TOP_FRACS:
            selected = np.zeros(len(ledger), dtype=np.int8)
            selected[_rank_top_indices(ledger["score"], float(frac))] = 1
            ledger[f"selected_top{int(round(float(frac) * 100))}"] = selected
        ledger_frames.append(ledger)
        diagnostic_rows.append(
            {
                "month": str(fold["month"]),
                "train_rows": int(len(idx)),
                "train_rows_uncapped": int(len(fold["x_train"])),
                "valid_rows": int(len(fold["x_valid"])),
                "weight_effective_frac": _effective_sample_size(weights) / max(float(len(weights)), 1.0),
                "ae_gmm_generated_features": int(fold["ae_gmm_generated_features"]),
                "ae_gmm_status": fold.get("ae_gmm_status"),
                **params,
            }
        )

    ledger_all = pd.concat(ledger_frames, ignore_index=True)
    availability_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    for month, month_frame in ledger_all.groupby("month", dropna=False):
        availability_rows.extend(_availability_metrics(month_frame, scope="month", month=str(month)))
        for frac in TOP_FRACS:
            col = f"selected_top{int(round(float(frac) * 100))}"
            selected = month_frame[month_frame[col].eq(1)]
            for (_side_value, _archetype_value), group in selected.groupby(["side_name", "archetype"], dropna=False):
                selected_rows.append(_group_metrics(group, scope="month", month=str(month), top_frac=float(frac)))
    availability_rows.extend(_availability_metrics(ledger_all, scope="all", month="all"))
    for frac in TOP_FRACS:
        col = f"selected_top{int(round(float(frac) * 100))}"
        selected = ledger_all[ledger_all[col].eq(1)]
        for (_side_value, _archetype_value), group in selected.groupby(["side_name", "archetype"], dropna=False):
            selected_rows.append(_group_metrics(group, scope="all", month="all", top_frac=float(frac)))

    availability = pd.DataFrame(availability_rows)
    selected_audit = pd.DataFrame(selected_rows)
    diagnostics = pd.DataFrame(diagnostic_rows)
    paths = {
        "ledger": output_dir / "base_best_scored_ledger.parquet",
        "availability": output_dir / "base_side_archetype_availability.csv",
        "selected_audit": output_dir / "base_side_archetype_selected_topk.csv",
        "diagnostics": output_dir / "base_side_archetype_audit_diagnostics.csv",
        "manifest": output_dir / "manifest.json",
        "report": output_dir / "base_side_archetype_audit.md",
    }
    ledger_all.to_parquet(paths["ledger"], index=False)
    availability.to_csv(paths["availability"], index=False)
    selected_audit.to_csv(paths["selected_audit"], index=False)
    diagnostics.to_csv(paths["diagnostics"], index=False)

    def table(frame: pd.DataFrame, cols: list[str], limit: int | None = None) -> str:
        if frame.empty:
            return "No rows."
        view = frame[[col for col in cols if col in frame.columns]].copy()
        if limit is not None:
            view = view.head(limit)
        for col in view.columns:
            if pd.api.types.is_float_dtype(view[col]):
                view[col] = view[col].map(lambda v: f"{float(v):.4f}" if pd.notna(v) else "")
        return view.to_markdown(index=False)

    report_lines = [
        "# Base Side x Archetype Audit",
        "",
        f"Labels: `{labels_path}`",
        f"HPO output: `{hpo_output_dir}`",
        f"Winner: `{best.get('trial_name', '')}` / target `{params['target_mode']}` / weight `{params['weight_arm']}`",
        "",
        "## Selected Top-k by Side x Archetype",
        "",
        table(
            selected_audit[
                selected_audit["scope"].eq("all") & selected_audit["top_frac"].isin([0.10, 0.20, 0.30])
            ].sort_values(["top_frac", "side", "archetype"]),
            [
                "top_frac",
                "side",
                "archetype",
                "selected_rows",
                "selected_symbols",
                "gross_ev_weighted_clean_precision",
                "clean_precision",
                "mean_first_touch_net",
                "q10_first_touch_net",
                "first_touch_timeout_rate",
                "first_touch_bad_mae_to_sl_rate",
                "top_symbol_share",
            ],
        ),
        "",
        "## Available Base Stream by Side x Archetype",
        "",
        table(
            availability[availability["scope"].eq("all")].sort_values(["side", "archetype"]),
            [
                "side",
                "archetype",
                "available_rows",
                "available_symbols",
                "available_clean_rate",
                "available_mean_first_touch_net",
                "available_timeout_rate",
                "available_bad_mae_to_sl_rate",
            ],
        ),
        "",
        "## Outputs",
        "",
        f"- Ledger: `{paths['ledger']}`",
        f"- Availability: `{paths['availability']}`",
        f"- Selected audit: `{paths['selected_audit']}`",
        f"- Diagnostics: `{paths['diagnostics']}`",
        f"- Manifest: `{paths['manifest']}`",
    ]
    paths["report"].write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    manifest = {
        "scope": "materialized_trailing_base_side_archetype_audit",
        "labels_path": str(labels_path),
        "hpo_output_dir": str(hpo_output_dir),
        "output_dir": str(output_dir),
        "best_params": best,
        "replayed_params": params,
        "trial_number": int(trial_number),
        "max_train_rows": int(max_train_rows),
        "seed": int(seed),
        "fold_manifest": prep_manifest,
        "outputs": {key: str(value) for key, value in paths.items()},
    }
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-path", type=Path, required=True)
    parser.add_argument("--hpo-output-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--feature-list-csv", type=Path, default=DEFAULT_FEATURE_LIST_CSV)
    parser.add_argument("--months", default="2026-04,2026-05,2026-06")
    parser.add_argument("--max-feature-store-features", type=int, default=None)
    parser.add_argument("--max-train-rows", type=int, default=80_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-ae-gmm-state-features", action="store_true")
    parser.add_argument("--ae-gmm-state-feature-max-train-rows", type=int, default=50_000)
    parser.add_argument("--ae-gmm-state-feature-max-iter", type=int, default=64)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = run_audit(
        labels_path=args.labels_path,
        hpo_output_dir=args.hpo_output_dir,
        output_dir=args.output_dir,
        feature_dir=args.feature_dir,
        feature_list_csv=args.feature_list_csv,
        months=_parse_csv(args.months, ()),
        max_feature_store_features=args.max_feature_store_features,
        max_train_rows=int(args.max_train_rows),
        seed=int(args.seed),
        include_ae_gmm_state_features=not bool(args.no_ae_gmm_state_features),
        ae_gmm_state_feature_max_train_rows=int(args.ae_gmm_state_feature_max_train_rows),
        ae_gmm_state_feature_max_iter=int(args.ae_gmm_state_feature_max_iter),
    )
    print(json.dumps(_json_safe(manifest), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
