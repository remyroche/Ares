#!/usr/bin/env python3
"""Diagnose whether clean oracle rows are inside label-proxy candidate pools.

This is a proxy-only containment diagnostic. It does not train base/meta models
or optimize policy geometry. It asks whether wider causal label-proxy candidate
pools contain enough oracle and strict-clean oracle rows to make later veto or
rerank stages plausible.
"""

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

from scripts.diagnose_label_matched_clean_dirty_feature_gap import (  # noqa: E402
    DEFAULT_LABELS_PATH,
    _build_frame,
    _parse_csv,
    _parse_float_csv,
)
from scripts.report_label_proxy_quality_with_economic_limits import (  # noqa: E402
    DEFAULT_FEATURE_LIST_CSV,
    DEFAULT_PROXY_TOP_K,
    _score_proxy,
    _top_gate,
)
from scripts.run_label_adverse_path_proxy_gate_ablation import _path_targets, _table  # noqa: E402
from scripts.run_label_confusion_veto_proxy_ablation import DEFAULT_LABEL_ARMS, DEFAULT_TOP_FRACS  # noqa: E402
from scripts.run_label_economic_proxy_ablation import _label_targets  # noqa: E402
from scripts.run_label_quality_proxy_diagnostics import (  # noqa: E402
    DEFAULT_FEATURE_DIR,
    _feature_columns,
    _json_safe,
    _safe_mean,
    _safe_quantile,
    _spearman,
)
from scripts.run_soft_label_economic_proxy_ablation import (  # noqa: E402
    DEFAULT_EVENT_FEATURE_STORE_FEATURES,
    DEFAULT_PRIOR_WINDOWS_DAYS,
    DEFAULT_STATE_PATH_PRIOR_FEATURES,
)


DEFAULT_OUTPUT_DIR = Path("data_perp/reports/label_candidate_pool_oracle_containment_v1")
DEFAULT_CANDIDATE_MULTS = (1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0)


def _safe_numeric(values: Any) -> pd.Series:
    if isinstance(values, pd.Series):
        return pd.to_numeric(values, errors="coerce")
    return pd.to_numeric(pd.Series(values), errors="coerce")


def _mfe_mae(metrics: pd.DataFrame) -> pd.Series:
    out = _safe_numeric(metrics["mfe_norm"]) / _safe_numeric(metrics["mae_norm"]).clip(lower=0.25)
    return out.replace([np.inf, -np.inf], np.nan).clip(upper=10.0)


def _mask_rate(mask: pd.Series, denom: pd.Series) -> float:
    denom_n = int(denom.sum())
    return float((mask & denom).sum() / denom_n) if denom_n else 0.0


def _candidate_metrics(
    *,
    metrics: pd.DataFrame,
    target: pd.DataFrame,
    label_score: pd.Series,
    strict_clean: pd.Series,
    bounded: pd.Series,
    dirty: pd.Series,
    top_frac: float,
    candidate_mult: float,
) -> dict[str, Any]:
    label = _safe_numeric(label_score).reset_index(drop=True)
    metrics = metrics.reset_index(drop=True)
    target = target.reset_index(drop=True)
    strict_clean = strict_clean.reset_index(drop=True).astype(bool)
    bounded = bounded.reset_index(drop=True).astype(bool)
    dirty = dirty.reset_index(drop=True).astype(bool)

    oracle = _top_gate(target["target_soft"], float(top_frac)).reset_index(drop=True).astype(bool)
    candidate_frac = min(1.0, max(float(top_frac), float(top_frac) * float(candidate_mult)))
    candidate = _top_gate(label, candidate_frac).reset_index(drop=True).astype(bool)
    strict_oracle = oracle & strict_clean
    bounded_oracle = oracle & bounded
    candidate_rows = int(candidate.sum())
    oracle_rows = int(oracle.sum())
    strict_oracle_rows = int(strict_oracle.sum())
    bounded_oracle_rows = int(bounded_oracle.sum())
    recovered_oracle = candidate & oracle
    recovered_strict_oracle = candidate & strict_oracle
    recovered_bounded_oracle = candidate & bounded_oracle
    selected_metrics = metrics.loc[candidate]
    mfe_mae = _mfe_mae(selected_metrics) if candidate_rows else pd.Series(dtype=float)
    return {
        "candidate_mult": float(candidate_mult),
        "candidate_frac": float(candidate_frac),
        "candidate_rows": candidate_rows,
        "oracle_rows": oracle_rows,
        "oracle_recovered_rows": int(recovered_oracle.sum()),
        "oracle_recovery_rate": float(recovered_oracle.sum() / oracle_rows) if oracle_rows else 0.0,
        "strict_oracle_rows": strict_oracle_rows,
        "strict_oracle_recovered_rows": int(recovered_strict_oracle.sum()),
        "strict_oracle_recovery_rate": (
            float(recovered_strict_oracle.sum() / strict_oracle_rows) if strict_oracle_rows else 0.0
        ),
        "bounded_oracle_rows": bounded_oracle_rows,
        "bounded_oracle_recovered_rows": int(recovered_bounded_oracle.sum()),
        "bounded_oracle_recovery_rate": (
            float(recovered_bounded_oracle.sum() / bounded_oracle_rows) if bounded_oracle_rows else 0.0
        ),
        "candidate_oracle_density": float(recovered_oracle.sum() / candidate_rows) if candidate_rows else 0.0,
        "candidate_strict_oracle_density": (
            float(recovered_strict_oracle.sum() / candidate_rows) if candidate_rows else 0.0
        ),
        "candidate_mean_return_net": _safe_mean(selected_metrics.get("ret_net", pd.Series(dtype=float))),
        "candidate_hit_return_net": _safe_mean(selected_metrics.get("ret_net", pd.Series(dtype=float)) > 0.0),
        "candidate_bad_mae_1r_rate": _safe_mean(selected_metrics.get("mae_norm", pd.Series(dtype=float)) >= 1.0),
        "candidate_p90_mae_norm": _safe_quantile(selected_metrics.get("mae_norm", pd.Series(dtype=float)), 0.90),
        "candidate_wide_25bps_rate": _safe_mean(selected_metrics.get("barrier", pd.Series(dtype=float)) > 0.025),
        "candidate_timeout_rate": _safe_mean(selected_metrics.get("is_timeout", pd.Series(dtype=float)).astype(float))
        if candidate_rows and "is_timeout" in selected_metrics
        else float("nan"),
        "candidate_strict_clean_rate": _safe_mean(strict_clean.loc[candidate]),
        "candidate_bounded_rate": _safe_mean(bounded.loc[candidate]),
        "candidate_dirty_rate": _safe_mean(dirty.loc[candidate]),
        "candidate_mean_mfe_mae_ratio": _safe_mean(mfe_mae),
    }


def _write_markdown(
    output_dir: Path,
    rows: pd.DataFrame,
    manifest: dict[str, Any],
) -> Path:
    path = output_dir / "label_candidate_pool_oracle_containment.md"
    cols = [
        "month",
        "label_arm",
        "top_frac",
        "candidate_mult",
        "candidate_rows",
        "oracle_recovery_rate",
        "strict_oracle_recovery_rate",
        "bounded_oracle_recovery_rate",
        "candidate_oracle_density",
        "candidate_mean_return_net",
        "candidate_bad_mae_1r_rate",
        "candidate_p90_mae_norm",
        "candidate_strict_clean_rate",
        "candidate_timeout_rate",
    ]
    best_cols = [
        "month",
        "label_arm",
        "top_frac",
        "candidate_mult",
        "oracle_recovery_rate",
        "strict_oracle_recovery_rate",
        "bounded_oracle_recovery_rate",
        "candidate_oracle_density",
        "candidate_strict_oracle_density",
        "candidate_bad_mae_1r_rate",
        "candidate_p90_mae_norm",
    ]
    best = (
        rows.sort_values(["month", "label_arm", "top_frac", "strict_oracle_recovery_rate", "oracle_recovery_rate"])
        .groupby(["month", "label_arm", "top_frac"], observed=True, dropna=False)
        .tail(1)
        .sort_values(["month", "label_arm", "top_frac"])
        if not rows.empty
        else rows
    )
    lines = [
        "# Label Candidate-Pool Oracle Containment",
        "",
        "Scope: proxy-only diagnostic. No LightGBM, Optuna, policy geometry, or base/meta training is run.",
        "",
        "This report measures whether wider causal label-proxy candidate pools contain the oracle and strict-clean oracle rows. Low containment means the label proxy itself is the bottleneck; high containment with poor selected recovery means the downstream veto/rerank objective is the bottleneck.",
        "",
        f"Labels: `{', '.join(manifest['label_arms'])}`",
        f"Top fractions: `{manifest['top_fracs']}`",
        f"Candidate multipliers: `{manifest['candidate_mults']}`",
        f"Rows: `{manifest['rows']}`",
        f"Feature count: `{manifest['feature_count']}`",
        "",
        "## Best Containment Per Month/Head",
        "",
        _table(best, best_cols, limit=80),
        "",
        "## Full Containment Grid",
        "",
        _table(rows.sort_values(["month", "label_arm", "top_frac", "candidate_mult"]), cols, limit=220),
        "",
        "## Outputs",
        "",
        f"- Rows: `{manifest['outputs']['rows']}`",
        f"- Manifest: `{manifest['outputs']['manifest']}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_report(
    *,
    labels_path: Path,
    output_dir: Path,
    feature_dir: Path,
    feature_list_csv: Path,
    max_feature_store_features: int | None,
    label_arms: list[str],
    top_fracs: list[float],
    candidate_mults: list[float],
    proxy_top_k: int,
    include_causal_outcome_priors: bool,
    include_causal_state_path_priors: bool,
    include_event_confirmation_features: bool,
    include_adverse_path_composites: bool,
    prior_windows_days: list[float],
    prior_embargo_hours: float,
    state_path_prior_features: list[str],
    event_feature_store_features: list[str],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame, metrics, reports = _build_frame(
        labels_path=labels_path,
        feature_dir=feature_dir,
        feature_list_csv=feature_list_csv,
        max_feature_store_features=max_feature_store_features,
        include_causal_outcome_priors=include_causal_outcome_priors,
        include_causal_state_path_priors=include_causal_state_path_priors,
        include_event_confirmation_features=include_event_confirmation_features,
        include_adverse_path_composites=include_adverse_path_composites,
        prior_windows_days=prior_windows_days,
        prior_embargo_hours=prior_embargo_hours,
        state_path_prior_features=state_path_prior_features,
        event_feature_store_features=event_feature_store_features,
    )
    features = _feature_columns(frame)
    targets = _label_targets(frame, metrics)
    unknown = sorted(set(label_arms).difference(targets))
    if unknown:
        raise ValueError(f"Unknown label arms: {unknown}")
    path_targets = _path_targets(metrics)
    month_period = frame["__ts__"].dt.to_period("M").astype(str)
    months = sorted(m for m in month_period.dropna().unique().tolist() if m >= "2026-04")
    rows: list[dict[str, Any]] = []
    for month in months:
        train_mask = month_period < str(month)
        valid_mask = month_period == str(month)
        if int(train_mask.sum()) < 500 or int(valid_mask.sum()) < 100:
            continue
        train = frame.loc[train_mask].copy()
        valid = frame.loc[valid_mask].copy()
        valid_metrics = metrics.loc[valid_mask].copy().reset_index(drop=True)
        valid_strict = path_targets["strict_clean"].loc[valid_mask].reset_index(drop=True).gt(0.5)
        valid_bounded = path_targets["bounded"].loc[valid_mask].reset_index(drop=True).gt(0.5)
        valid_dirty = path_targets["dirty"].loc[valid_mask].reset_index(drop=True).gt(0.5)
        for label_arm in label_arms:
            target = targets[label_arm]
            label_score, label_diag = _score_proxy(
                train=train,
                valid=valid,
                features=features,
                y_train=target.loc[train_mask, "target_soft"],
                proxy_top_k=int(proxy_top_k),
            )
            valid_target = target.loc[valid_mask].copy().reset_index(drop=True)
            label_score = label_score.reset_index(drop=True)
            for top_frac in top_fracs:
                for candidate_mult in candidate_mults:
                    row = {
                        "month": str(month),
                        "label_arm": str(label_arm),
                        "top_frac": float(top_frac),
                        "score_ic_u": _spearman(label_score, valid_metrics["u_policy_net"]),
                        "score_ic_target": _spearman(label_score, valid_target["target_soft"]),
                        "proxy_features": ",".join(label_diag.get("proxy_features", [])),
                    }
                    row.update(
                        _candidate_metrics(
                            metrics=valid_metrics,
                            target=valid_target,
                            label_score=label_score,
                            strict_clean=valid_strict,
                            bounded=valid_bounded,
                            dirty=valid_dirty,
                            top_frac=float(top_frac),
                            candidate_mult=float(candidate_mult),
                        )
                    )
                    rows.append(row)

    out = pd.DataFrame(rows)
    paths = {
        "rows": output_dir / "label_candidate_pool_oracle_containment.csv",
        "manifest": output_dir / "manifest.json",
    }
    out.to_csv(paths["rows"], index=False)
    manifest = {
        "scope": "proxy_only_label_candidate_pool_oracle_containment",
        "labels_path": str(labels_path),
        "output_dir": str(output_dir),
        "rows": int(len(frame)),
        "timestamp_min": frame["__ts__"].min(),
        "timestamp_max": frame["__ts__"].max(),
        "symbols": int(frame["__symbol__"].nunique(dropna=True)),
        "feature_dir": str(feature_dir),
        "feature_list_csv": str(feature_list_csv),
        "max_feature_store_features": max_feature_store_features,
        "feature_count": int(len(features)),
        "label_arms": list(label_arms),
        "top_fracs": [float(v) for v in top_fracs],
        "candidate_mults": [float(v) for v in candidate_mults],
        "proxy_top_k": int(proxy_top_k),
        "include_causal_outcome_priors": bool(include_causal_outcome_priors),
        "include_causal_state_path_priors": bool(include_causal_state_path_priors),
        "include_event_confirmation_features": bool(include_event_confirmation_features),
        "include_adverse_path_composites": bool(include_adverse_path_composites),
        "prior_windows_days": [float(v) for v in prior_windows_days],
        "prior_embargo_hours": float(prior_embargo_hours),
        "months": months,
        "outputs": {key: str(value) for key, value in paths.items()},
        **reports,
    }
    markdown = _write_markdown(output_dir, out, manifest)
    manifest["outputs"]["markdown"] = str(markdown)
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-path", type=Path, default=DEFAULT_LABELS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--feature-list-csv", type=Path, default=DEFAULT_FEATURE_LIST_CSV)
    parser.add_argument("--max-feature-store-features", type=int, default=None)
    parser.add_argument("--label-arms", type=lambda value: _parse_csv(value, DEFAULT_LABEL_ARMS), default=",".join(DEFAULT_LABEL_ARMS))
    parser.add_argument("--top-fracs", type=_parse_float_csv, default=",".join(str(v) for v in DEFAULT_TOP_FRACS))
    parser.add_argument("--candidate-mults", type=_parse_float_csv, default=",".join(str(v) for v in DEFAULT_CANDIDATE_MULTS))
    parser.add_argument("--proxy-top-k", type=int, default=DEFAULT_PROXY_TOP_K)
    parser.add_argument("--include-causal-outcome-priors", action="store_true")
    parser.add_argument("--include-causal-state-path-priors", action="store_true")
    parser.add_argument("--include-event-confirmation-features", action="store_true")
    parser.add_argument("--include-adverse-path-composites", action="store_true")
    parser.add_argument(
        "--prior-windows-days",
        type=_parse_float_csv,
        default=",".join(str(v) for v in DEFAULT_PRIOR_WINDOWS_DAYS),
    )
    parser.add_argument("--prior-embargo-hours", type=float, default=24.0)
    parser.add_argument(
        "--state-path-prior-features",
        type=lambda value: _parse_csv(value, DEFAULT_STATE_PATH_PRIOR_FEATURES),
        default=",".join(DEFAULT_STATE_PATH_PRIOR_FEATURES),
    )
    parser.add_argument(
        "--event-feature-store-features",
        type=lambda value: _parse_csv(value, DEFAULT_EVENT_FEATURE_STORE_FEATURES),
        default=",".join(DEFAULT_EVENT_FEATURE_STORE_FEATURES),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = run_report(
        labels_path=args.labels_path,
        output_dir=args.output_dir,
        feature_dir=args.feature_dir,
        feature_list_csv=args.feature_list_csv,
        max_feature_store_features=args.max_feature_store_features,
        label_arms=list(args.label_arms),
        top_fracs=[float(v) for v in args.top_fracs],
        candidate_mults=[float(v) for v in args.candidate_mults],
        proxy_top_k=int(args.proxy_top_k),
        include_causal_outcome_priors=bool(args.include_causal_outcome_priors),
        include_causal_state_path_priors=bool(args.include_causal_state_path_priors),
        include_event_confirmation_features=bool(args.include_event_confirmation_features),
        include_adverse_path_composites=bool(args.include_adverse_path_composites),
        prior_windows_days=[float(v) for v in args.prior_windows_days],
        prior_embargo_hours=float(args.prior_embargo_hours),
        state_path_prior_features=list(args.state_path_prior_features),
        event_feature_store_features=list(args.event_feature_store_features),
    )
    print(json.dumps(_json_safe(manifest), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
