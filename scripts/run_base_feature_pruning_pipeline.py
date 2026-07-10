#!/usr/bin/env python3
"""Cheap-first pruning diagnostics for the S59/A0bis base feature set.

The script uses cached A0bis fold payloads, fits fixed-parameter diagnostic
models once per fold, then runs inference-only permutation diagnostics to
identify redundant or harmful selected features. It writes conservative,
balanced, and aggressive pruned feature CSVs, and can optionally confirm them
through the existing materialized top-k runner.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_label_quality_proxy_diagnostics import _json_safe  # noqa: E402
from scripts.run_label_weighted_proxy_ablation import WEIGHT_ARMS, _weight_series  # noqa: E402
from scripts.run_materialized_trailing_label_topk_lgbm_hpo import (  # noqa: E402
    TOP_FRACS,
    _feature_contract_hash,
    _feature_selection_family,
    _fit_lgbm_model,
    _load_fixed_params,
    _objective_from_rows,
    _safe_artifact_stem,
    _safe_numeric,
    _selection_metrics,
    _target_from_frame,
    _time_spread_cap_rows,
)


DEFAULT_SOURCE_REPORT = Path(
    "data_perp/reports/base_archetype_anchor_location_ablation_20260709/"
    "A0bis_atr_normalized_momentum_inputs"
)
DEFAULT_OUTPUT_DIR = Path(
    "data_perp/reports/base_feature_pruning_a0bis_20260709"
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _folds_from_cache(source_report: Path, months: Iterable[str] | None = None) -> list[dict[str, Any]]:
    cache = source_report / "_fold_cache"
    wanted = {str(m) for m in (months or []) if str(m).strip()}
    folds: list[dict[str, Any]] = []
    for fold_dir in sorted(cache.iterdir()):
        if not fold_dir.is_dir():
            continue
        fold = fold_dir.name
        if wanted and fold not in wanted:
            continue
        required = {
            "train": fold_dir / "train.parquet",
            "valid": fold_dir / "valid.parquet",
            "train_metrics": fold_dir / "train_metrics.parquet",
            "valid_metrics": fold_dir / "valid_metrics.parquet",
            "x_train": fold_dir / "x_train.parquet",
            "x_valid": fold_dir / "x_valid.parquet",
        }
        missing = [str(p) for p in required.values() if not p.exists()]
        if missing:
            raise FileNotFoundError(f"Fold {fold} is missing cached payloads: {missing[:5]}")
        start = pd.Timestamp(f"{fold}-01", tz="UTC")
        end = pd.Timestamp((pd.Period(fold) + 1).start_time, tz="UTC")
        folds.append(
            {
                "fold": fold,
                "month": fold,
                "valid_start": start,
                "valid_end": end,
                "payload_paths": {key: str(path) for key, path in required.items()},
            }
        )
    if not folds:
        raise RuntimeError(f"No cached folds found under {cache}")
    return folds


def _read_payload(fold: dict[str, Any], columns: list[str] | None = None) -> dict[str, pd.DataFrame]:
    payload: dict[str, pd.DataFrame] = {}
    for key, path_text in fold["payload_paths"].items():
        path = Path(path_text)
        if key in {"x_train", "x_valid"} and columns is not None:
            frame = pd.read_parquet(path, columns=columns).astype(np.float32, copy=False)
        else:
            frame = pd.read_parquet(path)
            if key in {"x_train", "x_valid"}:
                frame = frame.astype(np.float32, copy=False)
        payload[key] = frame
    return payload


def _topk_objective(
    *,
    valid: pd.DataFrame,
    metrics: pd.DataFrame,
    target: pd.DataFrame,
    pred: np.ndarray,
    fold: str,
) -> tuple[float, list[dict[str, Any]]]:
    rows = [
        _selection_metrics(
            valid=valid,
            metrics=metrics,
            target=target,
            pred=pd.Series(pred.astype(np.float32, copy=False)),
            month=str(fold),
            top_frac=float(frac),
            trial_name="feature_pruning",
        )
        for frac in TOP_FRACS
    ]
    return float(_objective_from_rows(rows)), rows


def _predict_model(model: Any, values: np.ndarray, features: list[str]) -> np.ndarray:
    try:
        booster = getattr(model, "booster_", None)
        if booster is not None:
            return np.asarray(booster.predict(values), dtype=np.float32)
    except Exception:
        pass
    return np.asarray(
        model.predict(pd.DataFrame(values, columns=features)),
        dtype=np.float32,
    )


def _psi(train_values: np.ndarray, valid_values: np.ndarray, bins: int = 10) -> float:
    tr = np.asarray(train_values, dtype=np.float64)
    va = np.asarray(valid_values, dtype=np.float64)
    tr = tr[np.isfinite(tr)]
    va = va[np.isfinite(va)]
    if len(tr) < 100 or len(va) < 100:
        return float("nan")
    edges = np.unique(np.nanquantile(tr, np.linspace(0.0, 1.0, int(bins) + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0] = -np.inf
    edges[-1] = np.inf
    tr_counts, _ = np.histogram(tr, bins=edges)
    va_counts, _ = np.histogram(va, bins=edges)
    eps = 1e-6
    tr_p = tr_counts.astype(np.float64) / max(float(tr_counts.sum()), 1.0)
    va_p = va_counts.astype(np.float64) / max(float(va_counts.sum()), 1.0)
    return float(np.sum((va_p - tr_p) * np.log((va_p + eps) / (tr_p + eps))))


def _corr_groups(sample: pd.DataFrame, threshold: float) -> pd.DataFrame:
    features = [str(c) for c in sample.columns]
    n = len(features)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    ranked = sample.rank(method="average", pct=True).astype(np.float32, copy=False)
    corr = ranked.corr(method="pearson").abs().fillna(0.0).to_numpy(dtype=np.float32, copy=False)
    for i in range(n):
        for j in range(i + 1, n):
            if float(corr[i, j]) >= float(threshold):
                union(i, j)
    groups: dict[int, list[str]] = {}
    for i, feature in enumerate(features):
        groups.setdefault(find(i), []).append(feature)
    rows: list[dict[str, Any]] = []
    for group_idx, members in enumerate(sorted(groups.values(), key=lambda g: (len(g), g[0]), reverse=True), start=1):
        gid = f"corr_{group_idx:03d}"
        for member in members:
            rows.append(
                {
                    "feature": member,
                    "corr_group_id": gid,
                    "corr_group_size": int(len(members)),
                    "corr_group_members": "|".join(members),
                }
            )
    return pd.DataFrame(rows)


def _permutation_records(
    *,
    model: Any,
    x_eval: pd.DataFrame,
    valid_eval: pd.DataFrame,
    metrics_eval: pd.DataFrame,
    target_eval: pd.DataFrame,
    feature_groups: dict[str, list[str]],
    fold: str,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], float]:
    features = [str(c) for c in x_eval.columns]
    feature_to_idx = {f: i for i, f in enumerate(features)}
    base_values = x_eval.to_numpy(dtype=np.float32, copy=True)
    baseline_pred = _predict_model(model, base_values, features).astype(np.float32, copy=False)
    baseline_obj, baseline_rows = _topk_objective(
        valid=valid_eval,
        metrics=metrics_eval,
        target=target_eval,
        pred=baseline_pred,
        fold=fold,
    )
    rng = np.random.default_rng(int(seed))
    records: list[dict[str, Any]] = []
    for group_name, members in feature_groups.items():
        idx = [feature_to_idx[m] for m in members if m in feature_to_idx]
        if not idx:
            continue
        perm_values = base_values.copy()
        order = rng.permutation(perm_values.shape[0])
        perm_values[:, idx] = perm_values[order[:, None], idx]
        perm_pred = _predict_model(model, perm_values, features).astype(np.float32, copy=False)
        perm_obj, perm_rows = _topk_objective(
            valid=valid_eval,
            metrics=metrics_eval,
            target=target_eval,
            pred=perm_pred,
            fold=fold,
        )
        delta = float(baseline_obj - perm_obj)
        top10_base = next((r for r in baseline_rows if float(r["top_frac"]) == 0.10), {})
        top10_perm = next((r for r in perm_rows if float(r["top_frac"]) == 0.10), {})
        records.append(
            {
                "fold": str(fold),
                "group": str(group_name),
                "members": "|".join(members),
                "member_count": int(len(idx)),
                "baseline_objective": float(baseline_obj),
                "permuted_objective": float(perm_obj),
                "mda_drop": delta,
                "harm_score": float(perm_obj - baseline_obj),
                "baseline_top10_net": float(top10_base.get("mean_first_touch_net", np.nan)),
                "permuted_top10_net": float(top10_perm.get("mean_first_touch_net", np.nan)),
                "baseline_top10_clean_precision": float(top10_base.get("clean_precision", np.nan)),
                "permuted_top10_clean_precision": float(top10_perm.get("clean_precision", np.nan)),
                "baseline_top10_stop_rate": float(top10_base.get("first_touch_stop_rate", np.nan)),
                "permuted_top10_stop_rate": float(top10_perm.get("first_touch_stop_rate", np.nan)),
            }
        )
        del perm_values, perm_pred
    return records, baseline_rows, baseline_obj


def _feature_set_csv(path: Path, features: list[str], diagnostics: pd.DataFrame, label: str) -> None:
    selected = set(features)
    rows = []
    for rank, feature in enumerate(features, start=1):
        rows.append({"feature": str(feature), "selected": True, "rank": int(rank), "set": str(label)})
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    diagnostics.assign(selected_for_set=diagnostics["feature"].astype(str).isin(selected)).to_csv(
        path.with_suffix(".diagnostics.csv"),
        index=False,
    )


def _ranked_feature_sets(feature_diag: pd.DataFrame, original_features: list[str]) -> dict[str, list[str]]:
    diag = feature_diag.copy()
    max_mda = float(np.nanmax(np.abs(diag["mda_mean"].to_numpy(dtype=np.float64)))) if len(diag) else 0.0
    if not math.isfinite(max_mda) or max_mda <= 0.0:
        max_mda = 1.0
    diag["drift_penalty"] = np.maximum(pd.to_numeric(diag["psi_mean"], errors="coerce").fillna(0.0) - 0.25, 0.0)
    diag["harm_penalty"] = pd.to_numeric(diag["harmful_fold_count"], errors="coerce").fillna(0.0)
    diag["protected"] = diag["feature"].astype(str).eq("side")
    diag["prune_score"] = (
        pd.to_numeric(diag["mda_mean"], errors="coerce").fillna(0.0)
        - 0.20 * max_mda * diag["drift_penalty"]
        - 0.10 * max_mda * diag["harm_penalty"]
        + np.where(diag["is_corr_group_best"], 0.025 * max_mda, 0.0)
        + np.where(diag["protected"], 10.0 * max_mda, 0.0)
    )
    order = diag.sort_values(
        ["protected", "prune_score", "mda_mean"],
        ascending=[False, False, False],
        kind="mergesort",
    )["feature"].astype(str).tolist()
    original_order = [f for f in original_features if f in set(order)]
    ranked = [f for f in order if f in original_order]
    n = len(original_features)
    targets = {
        "conservative": max(1, min(n, max(80, int(round(0.75 * n))))),
        "balanced": max(1, min(n, max(55, int(round(0.55 * n))))),
        "aggressive": max(1, min(n, max(35, int(round(0.35 * n))))),
    }
    sets: dict[str, list[str]] = {}
    for name, target in targets.items():
        chosen = ranked[:target]
        if "side" in original_order and "side" not in chosen:
            chosen = ["side"] + chosen[:-1]
        chosen_set = set(chosen)
        sets[name] = [feature for feature in original_features if feature in chosen_set]
    return sets


def _confirmation_command(
    *,
    runner: Path,
    baseline_manifest: dict[str, Any],
    source_report: Path,
    output_dir: Path,
    feature_csv: Path,
    months: list[str],
    confirm_max_train_rows: int,
) -> list[str]:
    cmd = [
        sys.executable,
        "-u",
        str(runner),
        "--labels-path",
        str(baseline_manifest["labels_path"]),
        "--feature-dir",
        str(baseline_manifest["feature_dir"]),
        "--feature-list-csv",
        str(baseline_manifest["feature_list_csv"]),
        "--output-dir",
        str(output_dir),
        "--fixed-params-json",
        str(source_report / "topk_lgbm_hpo_best.json"),
        "--fixed-selected-features-csv",
        str(feature_csv),
        "--ae-gmm-input-features-csv",
        str(baseline_manifest["ae_gmm_input_features_csv"]),
        "--train-window-days",
        str(int(baseline_manifest.get("train_window_days", 150) or 150)),
        "--ae-gmm-anchor-days",
        str(int(baseline_manifest.get("ae_gmm_anchor_days", 30) or 30)),
        "--ae-gmm-state-feature-max-train-rows",
        str(int(baseline_manifest.get("ae_gmm_state_ae_max_train_rows", 15000) or 15000)),
        "--ae-gmm-state-feature-gmm-max-train-rows",
        str(int(baseline_manifest.get("ae_gmm_state_gmm_max_train_rows", 100000) or 100000)),
        "--max-train-rows",
        str(int(confirm_max_train_rows)),
    ]
    if months:
        cmd.extend(["--months", ",".join(str(month) for month in months)])
    if bool(baseline_manifest.get("ae_gmm_refit_per_window", False)):
        cmd.append("--refit-ae-gmm-per-window")
    return cmd


def _ledger_metrics(ledger_path: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    ledger = pd.read_parquet(ledger_path)
    score_col = "score" if "score" in ledger.columns else "pred"
    if score_col not in ledger.columns:
        score_candidates = [c for c in ledger.columns if "pred" in c.lower() or "score" in c.lower()]
        if not score_candidates:
            return {"rows": int(len(ledger)), "error": "no_score_column"}, pd.DataFrame()
        score_col = score_candidates[0]
    ret_col = "first_touch_net" if "first_touch_net" in ledger.columns else "ret_net"
    if ret_col not in ledger.columns:
        ret_candidates = [c for c in ledger.columns if "net" in c.lower()]
        ret_col = ret_candidates[0] if ret_candidates else ""
    ts = pd.to_datetime(ledger.get("__ts__", ledger.get("timestamp")), utc=True, errors="coerce")
    rows: list[dict[str, Any]] = []
    out: dict[str, Any] = {"rows": int(len(ledger)), "score_col": score_col, "ret_col": ret_col}
    for frac in TOP_FRACS:
        n = int(math.ceil(len(ledger) * float(frac)))
        idx = pd.to_numeric(ledger[score_col], errors="coerce").nlargest(max(n, 1)).index
        sel = ledger.loc[idx]
        ret = pd.to_numeric(sel[ret_col], errors="coerce") if ret_col else pd.Series(dtype=float)
        tag = f"top{int(round(float(frac) * 100))}"
        out[f"{tag}_rows"] = int(len(sel))
        out[f"{tag}_mean_net"] = float(ret.mean()) if len(ret) else float("nan")
        out[f"{tag}_net_pnl"] = float(ret.sum()) if len(ret) else float("nan")
        if len(ts):
            out[f"{tag}_trades_per_day"] = float(len(sel) / max(int(ts.dt.date.nunique()), 1))
        for week, sub_idx in sel.groupby(ts.loc[idx].dt.to_period("W").astype(str)).groups.items():
            sub = ledger.loc[list(sub_idx)]
            sub_ret = pd.to_numeric(sub[ret_col], errors="coerce") if ret_col else pd.Series(dtype=float)
            rows.append(
                {
                    "week_start": str(week),
                    "top_frac": float(frac),
                    "selected_rows": int(len(sub)),
                    "avg_net_return_per_trade": float(sub_ret.mean()) if len(sub_ret) else float("nan"),
                    "net_pnl": float(sub_ret.sum()) if len(sub_ret) else float("nan"),
                }
            )
    week_df = pd.DataFrame(rows)
    if not week_df.empty:
        q10 = week_df[week_df["top_frac"].eq(0.10)]["avg_net_return_per_trade"].quantile(0.10)
        out["q10_week_ev_top10"] = float(q10)
    return out, week_df


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-report", type=Path, default=DEFAULT_SOURCE_REPORT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--months", nargs="*", default=["2026-04", "2026-05", "2026-06"])
    parser.add_argument("--diagnostic-max-train-rows", type=int, default=300_000)
    parser.add_argument("--diagnostic-max-valid-rows", type=int, default=80_000)
    parser.add_argument("--corr-max-rows-per-fold", type=int, default=25_000)
    parser.add_argument("--corr-threshold", type=float, default=0.92)
    parser.add_argument("--seed", type=int, default=7331)
    parser.add_argument("--run-confirmation", action="store_true")
    parser.add_argument("--confirmation-sets", nargs="*", default=["conservative", "balanced", "aggressive"])
    parser.add_argument("--confirm-max-train-rows", type=int, default=0)
    args = parser.parse_args()

    source_report = args.source_report
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_set_dir = output_dir / "feature_sets"
    confirm_dir = output_dir / "confirmation_runs"
    source_manifest = _load_json(source_report / "manifest.json")
    params = _load_fixed_params(source_report / "topk_lgbm_hpo_best.json")
    if str(params["weight_arm"]) not in WEIGHT_ARMS:
        raise ValueError(f"Unknown weight arm in fixed params: {params['weight_arm']}")
    folds = _folds_from_cache(source_report, args.months)
    first_payload = _read_payload(folds[0])
    features = [str(c) for c in first_payload["x_train"].columns]
    del first_payload

    corr_samples: list[pd.DataFrame] = []
    feature_records: list[dict[str, Any]] = []
    baseline_rows_all: list[dict[str, Any]] = []
    drift_rows: list[dict[str, Any]] = []

    for fold_idx, fold in enumerate(folds):
        print(f"[prune] fold={fold['fold']} load", flush=True)
        payload = _read_payload(fold, columns=features)
        train_target = _target_from_frame(payload["train"], payload["train_metrics"], target_mode=str(params["target_mode"]))
        valid_target = _target_from_frame(payload["valid"], payload["valid_metrics"], target_mode=str(params["target_mode"]))
        weights = _weight_series(
            frame=payload["train"],
            metrics=payload["train_metrics"],
            target=train_target,
            arm=str(params["weight_arm"]),
        )
        fit_idx = _time_spread_cap_rows(len(payload["x_train"]), int(args.diagnostic_max_train_rows))
        eval_idx = _time_spread_cap_rows(len(payload["x_valid"]), int(args.diagnostic_max_valid_rows))
        print(
            f"[prune] fold={fold['fold']} fit_rows={len(fit_idx)} eval_rows={len(eval_idx)} features={len(features)}",
            flush=True,
        )
        model = _fit_lgbm_model(
            x_train=payload["x_train"].iloc[fit_idx].reset_index(drop=True),
            y_train=train_target["target_soft"].iloc[fit_idx].reset_index(drop=True),
            w_train=weights.iloc[fit_idx].reset_index(drop=True),
            params=params,
            seed=int(args.seed) + fold_idx,
        )
        corr_idx = _time_spread_cap_rows(len(payload["x_train"]), int(args.corr_max_rows_per_fold))
        corr_samples.append(payload["x_train"].iloc[corr_idx].reset_index(drop=True))
        x_eval = payload["x_valid"].iloc[eval_idx].reset_index(drop=True)
        valid_eval = payload["valid"].iloc[eval_idx].reset_index(drop=True)
        metrics_eval = payload["valid_metrics"].iloc[eval_idx].reset_index(drop=True)
        target_eval = valid_target.iloc[eval_idx].reset_index(drop=True)

        feature_groups = {feature: [feature] for feature in features}
        records, baseline_rows, _baseline_obj = _permutation_records(
            model=model,
            x_eval=x_eval,
            valid_eval=valid_eval,
            metrics_eval=metrics_eval,
            target_eval=target_eval,
            feature_groups=feature_groups,
            fold=str(fold["fold"]),
            seed=int(args.seed) + 10_000 + fold_idx,
        )
        feature_records.extend(records)
        for row in baseline_rows:
            row = dict(row)
            row["fold"] = str(fold["fold"])
            row["fit_rows"] = int(len(fit_idx))
            row["eval_rows"] = int(len(eval_idx))
            baseline_rows_all.append(row)

        train_drift_idx = _time_spread_cap_rows(len(payload["x_train"]), int(args.corr_max_rows_per_fold))
        valid_drift_idx = _time_spread_cap_rows(len(payload["x_valid"]), int(args.corr_max_rows_per_fold))
        train_drift = payload["x_train"].iloc[train_drift_idx]
        valid_drift = payload["x_valid"].iloc[valid_drift_idx]
        for feature in features:
            drift_rows.append(
                {
                    "fold": str(fold["fold"]),
                    "feature": str(feature),
                    "psi": _psi(train_drift[feature].to_numpy(), valid_drift[feature].to_numpy()),
                    "train_mean": float(np.nanmean(train_drift[feature].to_numpy(dtype=np.float64))),
                    "valid_mean": float(np.nanmean(valid_drift[feature].to_numpy(dtype=np.float64))),
                }
            )
        del payload, model, train_target, valid_target, weights, x_eval, valid_eval, metrics_eval, target_eval

    print("[prune] correlation groups", flush=True)
    corr_sample = pd.concat(corr_samples, ignore_index=True).astype(np.float32, copy=False)
    corr_df = _corr_groups(corr_sample, float(args.corr_threshold))
    corr_df.to_csv(output_dir / "correlation_groups.csv", index=False)

    corr_group_map = corr_df.groupby("corr_group_id")["feature"].apply(list).to_dict()
    group_feature_groups = {
        group_id: members
        for group_id, members in corr_group_map.items()
        if len(members) > 1
    }
    group_records: list[dict[str, Any]] = []
    if group_feature_groups:
        for fold_idx, fold in enumerate(folds):
            print(f"[prune] group_mda fold={fold['fold']} groups={len(group_feature_groups)}", flush=True)
            payload = _read_payload(fold, columns=features)
            train_target = _target_from_frame(payload["train"], payload["train_metrics"], target_mode=str(params["target_mode"]))
            valid_target = _target_from_frame(payload["valid"], payload["valid_metrics"], target_mode=str(params["target_mode"]))
            weights = _weight_series(
                frame=payload["train"],
                metrics=payload["train_metrics"],
                target=train_target,
                arm=str(params["weight_arm"]),
            )
            fit_idx = _time_spread_cap_rows(len(payload["x_train"]), int(args.diagnostic_max_train_rows))
            eval_idx = _time_spread_cap_rows(len(payload["x_valid"]), int(args.diagnostic_max_valid_rows))
            model = _fit_lgbm_model(
                x_train=payload["x_train"].iloc[fit_idx].reset_index(drop=True),
                y_train=train_target["target_soft"].iloc[fit_idx].reset_index(drop=True),
                w_train=weights.iloc[fit_idx].reset_index(drop=True),
                params=params,
                seed=int(args.seed) + 20_000 + fold_idx,
            )
            valid_target = valid_target.iloc[eval_idx].reset_index(drop=True)
            records, _baseline_rows, _baseline_obj = _permutation_records(
                model=model,
                x_eval=payload["x_valid"].iloc[eval_idx].reset_index(drop=True),
                valid_eval=payload["valid"].iloc[eval_idx].reset_index(drop=True),
                metrics_eval=payload["valid_metrics"].iloc[eval_idx].reset_index(drop=True),
                target_eval=valid_target,
                feature_groups=group_feature_groups,
                fold=str(fold["fold"]),
                seed=int(args.seed) + 30_000 + fold_idx,
            )
            group_records.extend(records)
            del payload, model, train_target, valid_target, weights

    feature_mda = pd.DataFrame(feature_records)
    feature_mda.to_csv(output_dir / "feature_permutation_mda_by_fold.csv", index=False)
    baseline_df = pd.DataFrame(baseline_rows_all)
    baseline_df.to_csv(output_dir / "diagnostic_baseline_metrics_by_fold.csv", index=False)
    group_mda = pd.DataFrame(group_records)
    group_mda.to_csv(output_dir / "correlation_group_permutation_mda_by_fold.csv", index=False)
    drift_df = pd.DataFrame(drift_rows)
    drift_df.to_csv(output_dir / "feature_drift_by_fold.csv", index=False)

    feature_agg = (
        feature_mda.assign(feature=feature_mda["group"].astype(str))
        .groupby("feature", as_index=False)
        .agg(
            mda_mean=("mda_drop", "mean"),
            mda_std=("mda_drop", "std"),
            mda_min=("mda_drop", "min"),
            harmful_fold_count=("harm_score", lambda s: int((pd.to_numeric(s, errors="coerce") > 0.0).sum())),
            fold_count=("fold", "nunique"),
        )
    )
    drift_agg = (
        drift_df.groupby("feature", as_index=False)
        .agg(psi_mean=("psi", "mean"), psi_max=("psi", "max"))
    )
    diag = (
        pd.DataFrame({"feature": features, "original_rank": np.arange(1, len(features) + 1, dtype=np.int32)})
        .merge(feature_agg, on="feature", how="left")
        .merge(drift_agg, on="feature", how="left")
        .merge(corr_df, on="feature", how="left")
    )
    diag["feature_family"] = diag["feature"].map(_feature_selection_family)
    group_mean_map: dict[str, float] = {}
    if not group_mda.empty:
        group_mean_map = group_mda.groupby("group")["mda_drop"].mean().to_dict()
    diag["corr_group_mda_mean"] = diag["corr_group_id"].map(group_mean_map).astype(float)
    diag["corr_group_size"] = pd.to_numeric(diag["corr_group_size"], errors="coerce").fillna(1).astype(int)
    diag["mda_mean"] = pd.to_numeric(diag["mda_mean"], errors="coerce").fillna(0.0)
    diag["mda_std"] = pd.to_numeric(diag["mda_std"], errors="coerce").fillna(0.0)
    diag["psi_mean"] = pd.to_numeric(diag["psi_mean"], errors="coerce").fillna(0.0)
    diag["harmful_fold_count"] = pd.to_numeric(diag["harmful_fold_count"], errors="coerce").fillna(0).astype(int)
    diag["is_corr_group_best"] = False
    for _gid, part in diag.groupby("corr_group_id", dropna=False):
        idx = part["mda_mean"].astype(float).idxmax()
        if pd.notna(idx):
            diag.loc[idx, "is_corr_group_best"] = True
    diag = diag.sort_values(["mda_mean", "original_rank"], ascending=[False, True], kind="mergesort")
    diag.to_csv(output_dir / "feature_pruning_diagnostics.csv", index=False)

    feature_sets = _ranked_feature_sets(diag, features)
    feature_set_paths: dict[str, str] = {}
    for name, selected_features in feature_sets.items():
        path = feature_set_dir / f"{name}_features.csv"
        _feature_set_csv(path, selected_features, diag, name)
        feature_set_paths[name] = str(path)
        print(f"[prune] feature_set={name} count={len(selected_features)} path={path}", flush=True)

    confirmation_results: list[dict[str, Any]] = []
    if bool(args.run_confirmation):
        runner = Path("scripts/run_materialized_trailing_label_topk_lgbm_hpo.py")
        for set_name in args.confirmation_sets:
            if set_name not in feature_set_paths:
                print(f"[confirm] skip unknown set={set_name}", flush=True)
                continue
            out = confirm_dir / f"{set_name}_fixedparams"
            cmd = _confirmation_command(
                runner=runner,
                baseline_manifest=source_manifest,
                source_report=source_report,
                output_dir=out,
                feature_csv=Path(feature_set_paths[set_name]),
                months=list(args.months),
                confirm_max_train_rows=int(args.confirm_max_train_rows),
            )
            env = dict(os.environ)
            env.setdefault("PYTHONPATH", ".")
            env.setdefault("EPM_LGBM_AE_GMM_INPUT_POLICY", "a0bis")
            log_path = out.with_suffix(".log")
            out.parent.mkdir(parents=True, exist_ok=True)
            print(f"[confirm] start set={set_name} cmd={' '.join(cmd)}", flush=True)
            with log_path.open("w", encoding="utf-8") as log:
                proc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
            result = {
                "set": set_name,
                "returncode": int(proc.returncode),
                "output_dir": str(out),
                "log_path": str(log_path),
                "feature_csv": feature_set_paths[set_name],
            }
            ledger_path = out / "best_oos_scored_ledger.parquet"
            if proc.returncode == 0 and ledger_path.exists():
                metrics, week_df = _ledger_metrics(ledger_path)
                result.update(metrics)
                if not week_df.empty:
                    week_df.insert(0, "set", set_name)
                    week_df.to_csv(out / "week_metrics_from_ledger.csv", index=False)
            confirmation_results.append(result)
            print(f"[confirm] done set={set_name} returncode={proc.returncode}", flush=True)
            if proc.returncode != 0:
                print(f"[confirm] failed set={set_name}; see {log_path}", flush=True)

    if confirmation_results:
        pd.DataFrame(confirmation_results).to_csv(output_dir / "confirmation_summary.csv", index=False)
    manifest = {
        "schema": "base_feature_pruning_pipeline_v1",
        "source_report": str(source_report),
        "source_manifest": str(source_report / "manifest.json"),
        "output_dir": str(output_dir),
        "months": list(args.months),
        "features_in": int(len(features)),
        "feature_contract_hash_in": _feature_contract_hash(features),
        "diagnostic_max_train_rows": int(args.diagnostic_max_train_rows),
        "diagnostic_max_valid_rows": int(args.diagnostic_max_valid_rows),
        "corr_threshold": float(args.corr_threshold),
        "params": _json_safe(params),
        "feature_set_paths": feature_set_paths,
        "run_confirmation": bool(args.run_confirmation),
        "confirmation_results": _json_safe(confirmation_results),
        "outputs": {
            "feature_pruning_diagnostics": str(output_dir / "feature_pruning_diagnostics.csv"),
            "feature_permutation_mda_by_fold": str(output_dir / "feature_permutation_mda_by_fold.csv"),
            "correlation_groups": str(output_dir / "correlation_groups.csv"),
            "group_mda": str(output_dir / "correlation_group_permutation_mda_by_fold.csv"),
            "drift": str(output_dir / "feature_drift_by_fold.csv"),
            "diagnostic_baseline_metrics": str(output_dir / "diagnostic_baseline_metrics_by_fold.csv"),
            "confirmation_summary": str(output_dir / "confirmation_summary.csv"),
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    print(json.dumps(_json_safe(manifest), indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
