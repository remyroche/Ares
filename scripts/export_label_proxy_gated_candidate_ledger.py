#!/usr/bin/env python3
"""Export causal path-risk-gated proxy candidate ledgers.

This script is pre-training only. For each OOT month it learns two weighted
feature proxies from earlier months:

- candidate score: the requested soft label and sample-weight design;
- path-risk score: future high-MAE / poor-path target.

It then selects from the candidate score only after suppressing rows with high
prior-month-learned path risk.
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

from scripts.export_label_proxy_candidate_ledger import (
    _aggregate_summary,
    _period_summary,
    _selected_ledger,
    _weekly_summary,
    _write_markdown,
)
from scripts.run_label_economic_proxy_ablation import LABEL_ARMS, _label_targets
from scripts.run_label_quality_proxy_diagnostics import (
    DEFAULT_FEATURE_DIR,
    DEFAULT_FEATURE_LIST_CSV,
    DEFAULT_LABELS_DIR,
    _feature_columns,
    _json_safe,
    _load_feature_store_columns,
    _load_labels,
    _path_metrics,
    _rank_top_indices,
    _read_feature_list,
    _safe_mean,
    _safe_quantile,
    _sigmoid,
    _spearman,
)
from scripts.run_label_weighted_proxy_ablation import (
    WEIGHT_ARMS,
    _effective_sample_size,
    _weight_series,
    _weighted_proxy_score,
)
from scripts.run_soft_label_economic_proxy_ablation import (
    DEFAULT_EVENT_FEATURE_STORE_FEATURES,
    DEFAULT_PRIOR_WINDOWS_DAYS,
    DEFAULT_STATE_PATH_PRIOR_FEATURES,
    _causal_outcome_prior_features,
    _causal_state_path_prior_features,
    _event_confirmation_features,
)


DEFAULT_OUTPUT_DIR = Path("data_perp/reports/label_proxy_gated_candidate_ledger_s14_w12_v1")
RISK_GATES = (0.30, 0.50, 0.70)
TOP_FRACS = (0.005, 0.01, 0.02)


def _safe_numeric(values: Any) -> pd.Series:
    if isinstance(values, pd.Series):
        return pd.to_numeric(values, errors="coerce")
    return pd.to_numeric(pd.Series(values), errors="coerce")


def _risk_target(metrics: pd.DataFrame, kind: str) -> pd.Series:
    mae_norm = _safe_numeric(metrics["mae_norm"]).fillna(0.0)
    bars_to_mfe = _safe_numeric(metrics["bars_to_mfe"]).fillna(24.0)
    barrier = _safe_numeric(metrics["barrier"]).fillna(0.0)
    utility = _safe_numeric(metrics["u_policy_net"]).fillna(-0.02)
    if kind == "bad_mae":
        raw = (mae_norm - 1.50) / 0.70
    elif kind == "bad_path":
        raw = (
            0.90 * (mae_norm - 1.50)
            + 0.25 * np.log1p(bars_to_mfe)
            + 12.0 * (barrier - 0.025).clip(lower=0.0)
            - 15.0 * utility.clip(lower=-0.02, upper=0.04)
        )
    elif kind == "tail_drawdown":
        raw = (
            1.20 * (mae_norm - 2.00)
            + 20.0 * (barrier - 0.030).clip(lower=0.0)
            - 10.0 * utility.clip(lower=-0.02, upper=0.04)
        )
    else:
        raise ValueError(f"Unknown risk target: {kind}")
    return pd.Series(_sigmoid(raw), index=metrics.index).clip(0.0, 1.0)


def _target_frame(series: pd.Series) -> pd.DataFrame:
    soft = _safe_numeric(series).clip(0.0, 1.0)
    return pd.DataFrame(
        {
            "target_soft": soft,
            "target_hard": (soft >= 0.65).astype(float),
        },
        index=soft.index,
    )


def _low_risk_mask(risk_score: pd.Series, keep_frac: float) -> pd.Series:
    risk = _safe_numeric(risk_score)
    ranks = risk.rank(method="first", pct=True)
    return ranks <= float(keep_frac)


def _gated_score(
    candidate_score: pd.Series,
    risk_score: pd.Series,
    *,
    risk_keep_frac: float,
) -> pd.Series:
    return candidate_score.where(_low_risk_mask(risk_score, risk_keep_frac))


def _monthly_run(
    *,
    frame: pd.DataFrame,
    metrics: pd.DataFrame,
    targets: dict[str, pd.DataFrame],
    features: list[str],
    month: str,
    label_arm: str,
    weight_arm: str,
    risk_kind: str,
    risk_keep_frac: float,
    top_frac: float,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    month_period = frame["__ts__"].dt.to_period("M").astype(str)
    train_mask = month_period < month
    valid_mask = month_period == month
    train = frame.loc[train_mask].copy()
    valid = frame.loc[valid_mask].copy().reset_index(drop=True)
    train_metrics = metrics.loc[train_mask].copy()
    valid_metrics = metrics.loc[valid_mask].copy().reset_index(drop=True)
    target_train = targets[label_arm].loc[train_mask].copy()
    target_valid = targets[label_arm].loc[valid_mask].copy().reset_index(drop=True)
    weights = _weight_series(
        frame=train,
        metrics=train_metrics,
        target=target_train,
        arm=weight_arm,
    )
    candidate_score, candidate_diag = _weighted_proxy_score(
        train,
        frame.loc[valid_mask].copy(),
        features,
        target_train["target_soft"],
        weights,
    )
    risk_train = _risk_target(train_metrics, risk_kind)
    risk_score, risk_diag = _weighted_proxy_score(
        train,
        frame.loc[valid_mask].copy(),
        features,
        risk_train,
        weights,
    )
    candidate_score = candidate_score.reset_index(drop=True)
    risk_score = risk_score.reset_index(drop=True)
    score = _gated_score(
        candidate_score,
        risk_score,
        risk_keep_frac=risk_keep_frac,
    )
    proxy_features = list(candidate_diag.get("proxy_features", []))
    ledger = _selected_ledger(
        valid=valid,
        valid_metrics=valid_metrics,
        target_valid=target_valid,
        score=score,
        month=str(month),
        top_frac=top_frac,
        proxy_features=proxy_features,
    )
    selected_pos = ledger["__valid_pos__"].to_numpy(dtype=np.int64, copy=False)
    ledger["candidate_score"] = candidate_score.iloc[selected_pos].to_numpy(dtype=np.float64, copy=False)
    ledger["risk_score"] = risk_score.iloc[selected_pos].to_numpy(dtype=np.float64, copy=False)
    ledger["risk_kind"] = risk_kind
    ledger["risk_keep_frac"] = float(risk_keep_frac)
    ledger["candidate_score_ic_u_month"] = _spearman(candidate_score, valid_metrics["u_policy_net"])
    ledger["risk_score_ic_bad_mae_month"] = _spearman(risk_score, (valid_metrics["mae_norm"] >= 1.0).astype(float))
    ledger["weight_effective_frac_train"] = _effective_sample_size(weights) / float(len(weights))
    monthly = _period_summary(
        frame=valid,
        metrics=valid_metrics,
        target=target_valid,
        score=score,
        label_arm=label_arm,
        weight_arm=weight_arm,
        period=str(month),
        top_frac=top_frac,
    )
    monthly["risk_kind"] = risk_kind
    monthly["risk_keep_frac"] = float(risk_keep_frac)
    monthly["candidate_score_ic_u"] = _spearman(candidate_score, valid_metrics["u_policy_net"])
    monthly["gated_score_ic_u"] = _spearman(score, valid_metrics["u_policy_net"])
    monthly["risk_score_ic_bad_mae"] = _spearman(risk_score, (valid_metrics["mae_norm"] >= 1.0).astype(float))
    monthly["risk_score_ic_u"] = _spearman(risk_score, valid_metrics["u_policy_net"])
    monthly["candidate_proxy_features"] = ",".join(proxy_features)
    monthly["risk_proxy_features"] = ",".join(risk_diag.get("proxy_features", []))
    weekly = _weekly_summary(
        valid=valid,
        valid_metrics=valid_metrics,
        target_valid=target_valid,
        score=score,
        ledger=ledger,
        label_arm=label_arm,
        weight_arm=weight_arm,
        month=str(month),
        top_frac=top_frac,
    )
    weekly["risk_kind"] = risk_kind
    weekly["risk_keep_frac"] = float(risk_keep_frac)
    return ledger, monthly, weekly


def _aggregate_grid(monthly: pd.DataFrame, weekly: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    groups = monthly.groupby(["risk_kind", "risk_keep_frac", "top_frac"], dropna=False, observed=True)
    for key, group in groups:
        risk_kind, risk_keep_frac, top_frac = key
        weeks = weekly[
            weekly["risk_kind"].eq(risk_kind)
            & weekly["risk_keep_frac"].eq(risk_keep_frac)
            & weekly["top_frac"].eq(top_frac)
        ].copy()
        selected_weeks = weeks[_safe_numeric(weeks["selected_rows"]) > 0]
        mean_u = _safe_numeric(group["mean_u"])
        week_mean = _safe_numeric(selected_weeks["mean_u"])
        week_count = len(weeks[["month", "week"]].drop_duplicates()) if len(weeks) else 0
        rows.append(
            {
                "risk_kind": risk_kind,
                "risk_keep_frac": float(risk_keep_frac),
                "top_frac": float(top_frac),
                "months": int(group["period"].nunique()),
                "positive_months": int((mean_u > 0.0).sum()),
                "mean_u": _safe_mean(mean_u),
                "worst_month_mean_u": _safe_quantile(mean_u, 0.0),
                "weeks": int(week_count),
                "selected_weeks": int(len(selected_weeks)),
                "positive_selected_weeks": int((week_mean > 0.0).sum()),
                "q25_week_mean_u": _safe_quantile(week_mean, 0.25),
                "worst_week_mean_u": _safe_quantile(week_mean, 0.0),
                "hit_u": _safe_mean(group["hit_u"]),
                "q10_u": _safe_mean(group["q10_u"]),
                "bad_mae_1r_rate": _safe_mean(group["bad_mae_1r_rate"]),
                "wide_barrier_25bps_rate": _safe_mean(group["wide_barrier_25bps_rate"]),
                "mean_selected_rows_month": _safe_mean(group["selected_rows"]),
                "min_selected_rows_month": int(_safe_numeric(group["selected_rows"]).min()),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["mean_u", "q25_week_mean_u", "worst_month_mean_u"],
        ascending=[False, False, False],
    )


def _write_grid_markdown(
    *,
    output_dir: Path,
    aggregate: pd.DataFrame,
    manifest: dict[str, Any],
) -> Path:
    path = output_dir / "label_proxy_gated_candidate_ledger.md"

    def table(frame: pd.DataFrame, cols: list[str], limit: int | None = None) -> str:
        if frame.empty:
            return "No rows."
        view = frame[[c for c in cols if c in frame.columns]].copy()
        if limit is not None:
            view = view.head(limit)
        for col in view.columns:
            if pd.api.types.is_float_dtype(view[col]):
                view[col] = view[col].map(lambda v: f"{float(v):.4f}" if pd.notna(v) else "")
        return view.to_markdown(index=False)

    lines = [
        "# Label Proxy Gated Candidate Ledger",
        "",
        "Scope: causal prior-month path-risk gating of a no-training label proxy.",
        "",
        "## Aggregate Grid",
        "",
        table(
            aggregate,
            [
                "risk_kind",
                "risk_keep_frac",
                "top_frac",
                "positive_months",
                "mean_u",
                "worst_month_mean_u",
                "selected_weeks",
                "positive_selected_weeks",
                "q25_week_mean_u",
                "worst_week_mean_u",
                "hit_u",
                "q10_u",
                "bad_mae_1r_rate",
                "wide_barrier_25bps_rate",
                "mean_selected_rows_month",
            ],
            limit=40,
        ),
        "",
        "## Outputs",
        "",
        f"- Ledger: `{manifest['outputs']['ledger']}`",
        f"- Monthly: `{manifest['outputs']['monthly']}`",
        f"- Weekly: `{manifest['outputs']['weekly']}`",
        f"- Aggregate: `{manifest['outputs']['aggregate']}`",
        f"- Manifest: `{manifest['outputs']['manifest']}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_export(
    *,
    labels_path: Path,
    output_dir: Path,
    feature_dir: Path,
    feature_list_csv: Path,
    max_feature_store_features: int | None,
    label_arm: str,
    weight_arm: str,
    risk_gates: tuple[float, ...],
    top_fracs: tuple[float, ...],
    risk_kinds: tuple[str, ...],
    include_causal_outcome_priors: bool,
    include_causal_state_path_priors: bool,
    include_event_confirmation_features: bool,
    prior_windows_days: tuple[float, ...],
    prior_embargo_hours: float,
    state_path_prior_features: tuple[str, ...],
    event_feature_store_features: tuple[str, ...],
) -> dict[str, Any]:
    if label_arm not in LABEL_ARMS:
        raise ValueError(f"label_arm must be one of {LABEL_ARMS}")
    if weight_arm not in WEIGHT_ARMS:
        raise ValueError(f"weight_arm must be one of {WEIGHT_ARMS}")
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = _load_labels(labels_path)
    selected_features = _read_feature_list(feature_list_csv, max_features=max_feature_store_features)
    if include_event_confirmation_features:
        selected_features = list(dict.fromkeys(list(selected_features) + list(event_feature_store_features)))
    feature_matrix, feature_store_report = _load_feature_store_columns(
        frame,
        feature_dir=feature_dir,
        selected_features=selected_features,
    )
    if not feature_matrix.empty:
        frame = pd.concat(
            [
                frame.drop(columns=[col for col in feature_matrix.columns if col in frame.columns]),
                feature_matrix.astype(np.float32, copy=False),
            ],
            axis=1,
        ).copy()
    metrics = _path_metrics(frame)

    prior_reports: dict[str, Any] = {
        "causal_outcome_priors": {"enabled": False},
        "causal_state_path_priors": {"enabled": False},
        "event_confirmation_features": {"enabled": False},
    }
    if include_causal_outcome_priors:
        prior_features, prior_reports["causal_outcome_priors"] = _causal_outcome_prior_features(
            frame,
            metrics,
            windows_days=[float(v) for v in prior_windows_days],
            embargo_hours=float(prior_embargo_hours),
        )
        frame = pd.concat([frame, prior_features.astype(np.float32, copy=False)], axis=1).copy()
    if include_causal_state_path_priors:
        state_prior_features, prior_reports["causal_state_path_priors"] = _causal_state_path_prior_features(
            frame,
            metrics,
            state_features=list(state_path_prior_features),
            windows_days=[float(v) for v in prior_windows_days],
            embargo_hours=float(prior_embargo_hours),
        )
        frame = pd.concat([frame, state_prior_features.astype(np.float32, copy=False)], axis=1).copy()
    if include_event_confirmation_features:
        event_features, prior_reports["event_confirmation_features"] = _event_confirmation_features(
            frame,
            event_features=list(event_feature_store_features),
        )
        frame = pd.concat([frame, event_features.astype(np.float32, copy=False)], axis=1).copy()

    features = _feature_columns(frame)
    targets = _label_targets(frame, metrics)
    months = sorted(frame["__ts__"].dt.to_period("M").dropna().astype(str).unique())

    ledgers: list[pd.DataFrame] = []
    monthly_rows: list[dict[str, Any]] = []
    weekly_frames: list[pd.DataFrame] = []
    for risk_kind in risk_kinds:
        for risk_keep_frac in risk_gates:
            for top_frac in top_fracs:
                for month in months[1:]:
                    month_period = frame["__ts__"].dt.to_period("M").astype(str)
                    if int((month_period < month).sum()) < 100 or int((month_period == month).sum()) < 50:
                        continue
                    ledger, monthly, weekly = _monthly_run(
                        frame=frame,
                        metrics=metrics,
                        targets=targets,
                        features=features,
                        month=str(month),
                        label_arm=label_arm,
                        weight_arm=weight_arm,
                        risk_kind=risk_kind,
                        risk_keep_frac=float(risk_keep_frac),
                        top_frac=float(top_frac),
                    )
                    ledgers.append(ledger)
                    monthly_rows.append(monthly)
                    weekly_frames.append(weekly)

    ledger_df = pd.concat(ledgers, ignore_index=True) if ledgers else pd.DataFrame()
    monthly = pd.DataFrame(monthly_rows)
    weekly = pd.concat(weekly_frames, ignore_index=True) if weekly_frames else pd.DataFrame()
    aggregate = _aggregate_grid(monthly, weekly) if not monthly.empty else pd.DataFrame()
    paths = {
        "ledger": output_dir / "selected_ledger.csv",
        "monthly": output_dir / "monthly_summary.csv",
        "weekly": output_dir / "weekly_summary.csv",
        "aggregate": output_dir / "aggregate_summary.csv",
        "manifest": output_dir / "manifest.json",
    }
    ledger_df.to_csv(paths["ledger"], index=False)
    monthly.to_csv(paths["monthly"], index=False)
    weekly.to_csv(paths["weekly"], index=False)
    aggregate.to_csv(paths["aggregate"], index=False)
    manifest = {
        "labels_path": str(labels_path),
        "output_dir": str(output_dir),
        "label_arm": label_arm,
        "weight_arm": weight_arm,
        "risk_gates": list(risk_gates),
        "top_fracs": list(top_fracs),
        "risk_kinds": list(risk_kinds),
        "rows": int(len(frame)),
        "timestamp_min": frame["__ts__"].min(),
        "timestamp_max": frame["__ts__"].max(),
        "symbols": int(frame["__symbol__"].nunique(dropna=True)),
        "feature_count": int(len(features)),
        "feature_store": feature_store_report,
        "feature_dir": str(feature_dir),
        "feature_list_csv": str(feature_list_csv),
        "include_causal_outcome_priors": bool(include_causal_outcome_priors),
        "include_causal_state_path_priors": bool(include_causal_state_path_priors),
        "include_event_confirmation_features": bool(include_event_confirmation_features),
        "prior_windows_days": [float(v) for v in prior_windows_days],
        "prior_embargo_hours": float(prior_embargo_hours),
        "state_path_prior_features": list(state_path_prior_features),
        "event_feature_store_features": list(event_feature_store_features),
        **prior_reports,
        "outputs": {key: str(value) for key, value in paths.items()},
    }
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    markdown = _write_grid_markdown(
        output_dir=output_dir,
        aggregate=aggregate,
        manifest=manifest,
    )
    manifest["outputs"]["markdown"] = str(markdown)
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    return manifest


def _parse_csv_floats(value: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in str(value).split(",") if part.strip())


def _parse_csv_strings(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in str(value).split(",") if part.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-path", type=Path, default=DEFAULT_LABELS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--feature-list-csv", type=Path, default=DEFAULT_FEATURE_LIST_CSV)
    parser.add_argument("--max-feature-store-features", type=int, default=None)
    parser.add_argument("--label-arm", default="S14_policy_net_path_blend")
    parser.add_argument("--weight-arm", default="W12_tail_timestamp_balanced")
    parser.add_argument("--risk-gates", default="0.3,0.5,0.7")
    parser.add_argument("--top-fracs", default="0.005,0.01,0.02")
    parser.add_argument("--risk-kinds", default="bad_mae,bad_path,tail_drawdown")
    parser.add_argument("--include-causal-outcome-priors", action="store_true")
    parser.add_argument("--include-causal-state-path-priors", action="store_true")
    parser.add_argument("--include-event-confirmation-features", action="store_true")
    parser.add_argument(
        "--prior-windows-days",
        default=",".join(str(v) for v in DEFAULT_PRIOR_WINDOWS_DAYS),
    )
    parser.add_argument("--prior-embargo-hours", type=float, default=24.0)
    parser.add_argument(
        "--state-path-prior-features",
        default=",".join(DEFAULT_STATE_PATH_PRIOR_FEATURES),
    )
    parser.add_argument(
        "--event-feature-store-features",
        default=",".join(DEFAULT_EVENT_FEATURE_STORE_FEATURES),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = run_export(
        labels_path=args.labels_path,
        output_dir=args.output_dir,
        feature_dir=args.feature_dir,
        feature_list_csv=args.feature_list_csv,
        max_feature_store_features=args.max_feature_store_features,
        label_arm=str(args.label_arm),
        weight_arm=str(args.weight_arm),
        risk_gates=_parse_csv_floats(args.risk_gates),
        top_fracs=_parse_csv_floats(args.top_fracs),
        risk_kinds=_parse_csv_strings(args.risk_kinds),
        include_causal_outcome_priors=bool(args.include_causal_outcome_priors),
        include_causal_state_path_priors=bool(args.include_causal_state_path_priors),
        include_event_confirmation_features=bool(args.include_event_confirmation_features),
        prior_windows_days=_parse_csv_floats(args.prior_windows_days),
        prior_embargo_hours=float(args.prior_embargo_hours),
        state_path_prior_features=_parse_csv_strings(args.state_path_prior_features),
        event_feature_store_features=_parse_csv_strings(args.event_feature_store_features),
    )
    print(json.dumps(_json_safe(manifest), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
