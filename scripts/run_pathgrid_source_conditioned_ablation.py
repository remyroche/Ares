#!/usr/bin/env python3
"""Source-conditioned proxy ablation for Stage108 path-grid labels.

This is a proxy-only test. It does not train LightGBM or tune policy geometry.
It tests whether the decisive Stage108 label becomes learnable when the proxy
is fit inside decision-time source families, then evaluated OOS with both
source-local and whole-portfolio selection budgets.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.diagnose_stage108_pathgrid_recoverability import (  # noqa: E402
    _choose_candidates,
    _dirty,
    _spec_from_row,
    _strict_clean,
)
from scripts.report_label_proxy_quality_with_economic_limits import (  # noqa: E402
    _add_delta,
    _baseline,
    _fit_holdout_summary,
    _slice_week_positions,
    _table,
)
from scripts.run_label_quality_proxy_diagnostics import (  # noqa: E402
    DEFAULT_FEATURE_DIR,
    _decile_diagnostics,
    _feature_columns,
    _json_safe,
    _rank_top_indices,
    _selection_metrics,
    _spearman,
)
from scripts.run_path_aware_label_target_grid import (  # noqa: E402
    DEFAULT_FEATURE_LIST_CSV,
    _target_for_spec,
)
from scripts.run_pathgrid_separator_gate_ablation import (  # noqa: E402
    DEFAULT_FIT_MONTHS,
    DEFAULT_HOLDOUT_MONTH,
    DEFAULT_MONTHS,
    DEFAULT_STAGE_DIR,
    DEFAULT_TOP_FRACS,
    _candidate_summary,
    _parse_csv,
    _parse_float_csv,
    _safe_numeric,
)
from scripts.run_pathgrid_source_gate_ablation import _gate_mask_frame  # noqa: E402
from scripts.run_soft_label_candidate_source_ablation import (  # noqa: E402
    _build_sources,
    _source_context,
)
from scripts.run_soft_label_economic_proxy_ablation import (  # noqa: E402
    DEFAULT_EVENT_FEATURE_STORE_FEATURES,
    DEFAULT_PRIOR_WINDOWS_DAYS,
    DEFAULT_STATE_PATH_PRIOR_FEATURES,
    _proxy_score,
)
from scripts.diagnose_label_matched_clean_dirty_feature_gap import _build_frame  # noqa: E402


DEFAULT_OUTPUT_DIR = Path("data_perp/reports/pathgrid_source_conditioned_stage113_v1")
DEFAULT_SOURCES = (
    "all",
    "exclude_rebound_or_lowbarrier",
    "exclude_rebound_or_confirmed_event_family",
    "loud_event_no_rebound",
    "quiet_mid_no_rebound",
    "clean_breakout_no_rebound",
    "run_entry_loud_event_no_rebound",
    "rebound_mid",
    "run_entry_rebound_mid",
)


def _source_masks(
    *,
    frame: pd.DataFrame,
    source_context: pd.DataFrame,
    run_gap_hours: float,
) -> dict[str, pd.Series]:
    raw = _build_sources(frame, source_context, run_gap_hours=float(run_gap_hours))
    gates = _gate_mask_frame(raw)
    out = {**raw, **gates}
    return {name: mask.fillna(False).astype(bool) for name, mask in out.items()}


def _rank_top_indices_with_budget(score: pd.Series, k: int) -> np.ndarray:
    score = _safe_numeric(score)
    valid = score.notna().to_numpy(dtype=bool, copy=False)
    if not bool(valid.any()) or int(k) <= 0:
        return np.array([], dtype=np.int64)
    valid_idx = np.flatnonzero(valid)
    order = np.argsort(-score.iloc[valid_idx].to_numpy(dtype=np.float64), kind="mergesort")
    return valid_idx[order[: min(int(k), len(order))]].astype(np.int64, copy=False)


def _score_period_selected(
    *,
    frame: pd.DataFrame,
    metrics: pd.DataFrame,
    target: pd.DataFrame,
    score: pd.Series,
    selected_idx: np.ndarray,
    period_type: str,
    period: str,
    month: str,
    selector: str,
    label_arm: str,
    economic_arm: str,
    top_frac: float,
    label_score: pd.Series | None,
    label_proxy_features: str,
) -> dict[str, Any]:
    score = _safe_numeric(score).reset_index(drop=True)
    frame = frame.reset_index(drop=True)
    metrics = metrics.reset_index(drop=True)
    target = target.reset_index(drop=True)
    row = _selection_metrics(
        frame=frame,
        metrics=metrics,
        target=target,
        score=score,
        arm=f"{selector}::{label_arm}::{economic_arm}",
        selector=selector,
        period=period,
        top_frac=float(top_frac),
        selected_idx=np.asarray(selected_idx, dtype=np.int64),
    )
    _add_delta(row, _baseline(metrics))
    row.update(
        {
            "period_type": period_type,
            "month": month,
            "label_arm": label_arm,
            "economic_arm": economic_arm,
            "label_proxy_features": label_proxy_features,
            "economic_proxy_features": "",
            "score_ic_u": _spearman(score, metrics["u_policy_net"]),
            "score_ic_label": _spearman(score, label_score.reset_index(drop=True))
            if label_score is not None
            else float("nan"),
            "score_ic_economic": float("nan"),
        }
    )
    row.update(_decile_diagnostics(score, metrics["u_policy_net"]))
    return row


def _confusion_row_selected(
    *,
    valid: pd.DataFrame,
    valid_metrics: pd.DataFrame,
    valid_target: pd.DataFrame,
    candidate: str,
    month: str,
    selector: str,
    top_frac: float,
    score: pd.Series,
    selected_idx: np.ndarray,
) -> dict[str, Any]:
    selected_mask = pd.Series(False, index=valid.index)
    if len(selected_idx):
        selected_mask.iloc[np.asarray(selected_idx, dtype=np.int64)] = True
    oracle_mask = pd.Series(False, index=valid.index)
    oracle_idx = _rank_top_indices(valid_target["target_soft"], float(top_frac))
    if len(oracle_idx):
        oracle_mask.iloc[oracle_idx] = True
    strict_clean = _strict_clean(valid_metrics).reset_index(drop=True)
    dirty = _dirty(valid_metrics).reset_index(drop=True)
    selected_metrics = valid_metrics.loc[selected_mask].copy()
    selected_target = valid_target.loc[selected_mask].copy()
    return {
        "candidate": candidate,
        "month": str(month),
        "selector": selector,
        "top_frac": float(top_frac),
        "selected_rows": int(selected_mask.sum()),
        "oracle_recovery_rate": float((selected_mask & oracle_mask).sum() / max(int(oracle_mask.sum()), 1)),
        "target_hard_rate": float(pd.to_numeric(selected_target.get("target_hard"), errors="coerce").mean())
        if int(selected_mask.sum())
        else float("nan"),
        "strict_clean_rate": float(strict_clean.loc[selected_mask].mean()) if int(selected_mask.sum()) else float("nan"),
        "dirty_rate": float(dirty.loc[selected_mask].mean()) if int(selected_mask.sum()) else float("nan"),
        "mean_return_net": float(pd.to_numeric(selected_metrics.get("ret_net"), errors="coerce").mean())
        if int(selected_mask.sum())
        else float("nan"),
        "mean_u": float(pd.to_numeric(selected_metrics.get("u_policy_net"), errors="coerce").mean())
        if int(selected_mask.sum())
        else float("nan"),
        "bad_mae_1r_rate": float((pd.to_numeric(selected_metrics.get("mae_norm"), errors="coerce") >= 1.0).mean())
        if int(selected_mask.sum())
        else float("nan"),
        "timeout_rate": float(pd.to_numeric(selected_metrics.get("is_timeout"), errors="coerce").mean())
        if int(selected_mask.sum())
        else float("nan"),
        "p90_mae_norm": float(pd.to_numeric(selected_metrics.get("mae_norm"), errors="coerce").quantile(0.90))
        if int(selected_mask.sum())
        else float("nan"),
        "score_ic_u": _spearman(score, valid_metrics["u_policy_net"]),
    }


def _selected_ledger_rows_selected(
    *,
    valid: pd.DataFrame,
    valid_metrics: pd.DataFrame,
    valid_target: pd.DataFrame,
    candidate: str,
    month: str,
    selector: str,
    top_frac: float,
    score: pd.Series,
    selected_idx: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not len(selected_idx):
        return rows
    mfe_mae = (
        valid_metrics["mfe_norm"] / valid_metrics["mae_norm"].clip(lower=0.25)
    ).replace([np.inf, -np.inf], np.nan).clip(upper=10.0)
    for pos in np.asarray(selected_idx, dtype=np.int64):
        rows.append(
            {
                "candidate": candidate,
                "month": str(month),
                "selector": selector,
                "top_frac": float(top_frac),
                "position": int(pos),
                "__ts__": valid["__ts__"].iloc[pos],
                "__symbol__": valid["__symbol__"].iloc[pos],
                "side": valid_metrics["side"].iloc[pos],
                "score": score.iloc[pos],
                "target_soft": valid_target["target_soft"].iloc[pos],
                "target_hard": valid_target["target_hard"].iloc[pos],
                "u_policy_net": valid_metrics["u_policy_net"].iloc[pos],
                "ret_net": valid_metrics["ret_net"].iloc[pos],
                "mae_norm": valid_metrics["mae_norm"].iloc[pos],
                "mfe_norm": valid_metrics["mfe_norm"].iloc[pos],
                "mfe_mae": mfe_mae.iloc[pos],
                "barrier": valid_metrics["barrier"].iloc[pos],
                "is_timeout": bool(valid_metrics["is_timeout"].iloc[pos]),
                "bars_to_mfe": valid_metrics["bars_to_mfe"].iloc[pos],
            }
        )
    return rows


def _period_rows_for_selector(
    *,
    valid: pd.DataFrame,
    valid_metrics: pd.DataFrame,
    valid_target: pd.DataFrame,
    score: pd.Series,
    candidate: str,
    month: str,
    source: str,
    top_frac: float,
    proxy_features: str,
    budget_mode: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    selector = f"source_{budget_mode}_{source}"
    period_slices = [("month", str(month), np.arange(len(valid), dtype=np.int64))]
    period_slices.extend(("week", week, pos) for week, pos in _slice_week_positions(valid))
    for period_type, period_name, pos in period_slices:
        local_score = _safe_numeric(score.iloc[pos]).reset_index(drop=True)
        if budget_mode == "local":
            selected_idx = _rank_top_indices(local_score, float(top_frac))
        elif budget_mode == "portfolio":
            selected_idx = _rank_top_indices_with_budget(
                local_score,
                max(1, int(math.ceil(float(top_frac) * len(pos)))),
            )
        else:
            raise ValueError(f"Unknown budget mode: {budget_mode}")
        rows.append(
            _score_period_selected(
                frame=valid.iloc[pos].reset_index(drop=True),
                metrics=valid_metrics.iloc[pos].reset_index(drop=True),
                target=valid_target.iloc[pos].reset_index(drop=True),
                score=local_score,
                selected_idx=selected_idx,
                period_type=period_type,
                period=period_name,
                month=str(month),
                selector=selector,
                label_arm=candidate,
                economic_arm="pathgrid_source_conditioned",
                top_frac=float(top_frac),
                label_score=valid_target["target_soft"].iloc[pos].reset_index(drop=True),
                label_proxy_features=proxy_features,
            )
        )
    return rows


def _source_profile_rows(
    *,
    month: str,
    candidate: str,
    source_masks: dict[str, pd.Series],
    valid_metrics: pd.DataFrame,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source, mask in source_masks.items():
        mask = mask.fillna(False).astype(bool).reset_index(drop=True)
        selected = valid_metrics.loc[mask]
        rows.append(
            {
                "month": str(month),
                "candidate": candidate,
                "source": source,
                "source_rows": int(mask.sum()),
                "source_frac": float(mask.mean()) if len(mask) else 0.0,
                "source_mean_return_net": float(pd.to_numeric(selected.get("ret_net"), errors="coerce").mean())
                if len(selected)
                else float("nan"),
                "source_bad_mae_1r_rate": float((pd.to_numeric(selected.get("mae_norm"), errors="coerce") >= 1.0).mean())
                if len(selected)
                else float("nan"),
                "source_timeout_rate": float(pd.to_numeric(selected.get("is_timeout"), errors="coerce").mean())
                if len(selected)
                else float("nan"),
            }
        )
    return rows


def _write_report(
    *,
    output_dir: Path,
    fit_holdout: pd.DataFrame,
    period_rows: pd.DataFrame,
    proxy_diag: pd.DataFrame,
    confusion: pd.DataFrame,
    source_profile: pd.DataFrame,
    manifest: dict[str, Any],
) -> Path:
    path = output_dir / "pathgrid_source_conditioned_ablation.md"
    monthly = period_rows[period_rows["period_type"].astype(str).eq("month")].copy() if not period_rows.empty else pd.DataFrame()
    lines = [
        "# Path-Grid Source-Conditioned Ablation",
        "",
        "Scope: proxy-only source-conditioned Stage108 hard-binary path-grid labels. No model training, Optuna, or policy optimisation.",
        "",
        f"Stage dir: `{manifest['stage_dir']}`",
        f"Months: `{', '.join(manifest['months'])}`. Fit months: `{', '.join(manifest['fit_months'])}`. Holdout: `{manifest['holdout_month']}`.",
        f"Candidates: `{', '.join(manifest['candidates'])}`",
        f"Sources: `{', '.join(manifest['sources'])}`",
        f"Top fractions: `{manifest['top_fracs']}`. Min source train rows: `{manifest['min_source_train_rows']}`.",
        "",
        "## Fit/Holdout Summary",
        "",
        _table(
            fit_holdout,
            [
                "trainworthy_pass",
                "fit_economic_pass",
                "holdout_economic_pass",
                "selector",
                "label_arm",
                "top_frac",
                "fit_mean_return_net",
                "holdout_mean_return_net",
                "fit_bad_mae_1r_rate",
                "holdout_bad_mae_1r_rate",
                "fit_p90_mae_norm",
                "holdout_p90_mae_norm",
                "fit_timeout_rate",
                "holdout_timeout_rate",
                "fit_score_ic_u",
                "holdout_score_ic_u",
                "fit_selected_rows",
                "holdout_selected_rows",
            ],
            limit=120,
        ),
        "",
        "## Monthly OOS Rows",
        "",
        _table(
            monthly.sort_values(["month", "label_arm", "top_frac", "selector"]) if not monthly.empty else monthly,
            [
                "month",
                "selector",
                "label_arm",
                "top_frac",
                "selected_rows",
                "mean_return_net",
                "bad_mae_1r_rate",
                "p90_mae_norm",
                "timeout_rate",
                "wide_barrier_25bps_rate",
                "score_ic_u",
                "strict_clean_row_rate",
                "target_top_hard_rate",
            ],
            limit=180,
        ),
        "",
        "## Proxy Diagnostics",
        "",
        _table(
            proxy_diag.sort_values(["month", "candidate", "source"]) if not proxy_diag.empty else proxy_diag,
            [
                "month",
                "candidate",
                "source",
                "train_rows",
                "valid_source_rows",
                "proxy_candidate_count",
                "proxy_mean_train_target_ic",
                "proxy_mean_train_utility_ic",
                "proxy_mean_train_bad_mae_ic",
                "proxy_mean_train_timeout_ic",
                "proxy_features",
            ],
            limit=160,
        ),
        "",
        "## Selector Confusion",
        "",
        _table(
            confusion.sort_values(["month", "candidate", "top_frac", "selector"]) if not confusion.empty else confusion,
            [
                "month",
                "candidate",
                "selector",
                "top_frac",
                "selected_rows",
                "oracle_recovery_rate",
                "target_hard_rate",
                "strict_clean_rate",
                "dirty_rate",
                "mean_return_net",
                "bad_mae_1r_rate",
                "timeout_rate",
                "p90_mae_norm",
                "score_ic_u",
            ],
            limit=180,
        ),
        "",
        "## Source Profile",
        "",
        _table(
            source_profile.sort_values(["month", "candidate", "source"]) if not source_profile.empty else source_profile,
            [
                "month",
                "candidate",
                "source",
                "source_rows",
                "source_frac",
                "source_mean_return_net",
                "source_bad_mae_1r_rate",
                "source_timeout_rate",
            ],
            limit=160,
        ),
        "",
        "## Outputs",
        "",
        f"- Period rows: `{manifest['outputs']['period_rows']}`",
        f"- Fit/holdout: `{manifest['outputs']['fit_holdout']}`",
        f"- Proxy diagnostics: `{manifest['outputs']['proxy_diag']}`",
        f"- Confusion: `{manifest['outputs']['confusion']}`",
        f"- Source profile: `{manifest['outputs']['source_profile']}`",
        f"- Selected ledger: `{manifest['outputs']['selected_ledger']}`",
        f"- Manifest: `{manifest['outputs']['manifest']}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_report(
    *,
    stage_dir: Path,
    output_dir: Path,
    candidates: list[str],
    max_candidates: int,
    months: list[str],
    fit_months: list[str],
    holdout_month: str,
    top_fracs: list[float],
    sources: list[str],
    min_train_rows: int | None,
    min_valid_rows: int | None,
    min_source_train_rows: int,
    min_source_valid_rows: int,
    run_gap_hours: float,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stage_manifest = json.loads((stage_dir / "manifest.json").read_text(encoding="utf-8"))
    candidate_summary = pd.read_csv(stage_dir / "path_aware_label_target_candidate_summary.csv")
    selected_candidates = _choose_candidates(candidate_summary, candidates, max_candidates=max_candidates)
    spec_rows = candidate_summary.set_index("candidate").loc[selected_candidates].reset_index()
    specs = [_spec_from_row(row) for _, row in spec_rows.iterrows()]

    reports = stage_manifest.get("reports", {})
    frame, metrics, load_reports = _build_frame(
        labels_path=Path(stage_manifest["labels_path"]),
        feature_dir=Path(stage_manifest.get("feature_dir", DEFAULT_FEATURE_DIR)),
        feature_list_csv=Path(stage_manifest.get("feature_list_csv", DEFAULT_FEATURE_LIST_CSV)),
        max_feature_store_features=stage_manifest.get("max_feature_store_features"),
        include_causal_outcome_priors=bool(reports.get("causal_outcome_priors", {}).get("enabled", False)),
        include_causal_state_path_priors=bool(reports.get("causal_state_path_priors", {}).get("enabled", False)),
        include_event_confirmation_features=bool(reports.get("event_confirmation_features", {}).get("enabled", False)),
        include_adverse_path_composites=bool(reports.get("adverse_path_composites", {}).get("enabled", False)),
        prior_windows_days=[float(v) for v in stage_manifest.get("prior_windows_days", DEFAULT_PRIOR_WINDOWS_DAYS)],
        prior_embargo_hours=float(stage_manifest.get("prior_embargo_hours", 24.0)),
        state_path_prior_features=list(stage_manifest.get("state_path_prior_features", DEFAULT_STATE_PATH_PRIOR_FEATURES)),
        event_feature_store_features=list(stage_manifest.get("event_feature_store_features", DEFAULT_EVENT_FEATURE_STORE_FEATURES)),
    )
    source_context = _source_context(frame)
    overlap = [col for col in source_context.columns if col in frame.columns]
    if overlap:
        frame = frame.drop(columns=overlap)
    frame = pd.concat([frame, source_context.astype(np.float32, copy=False)], axis=1).copy()
    all_sources = _source_masks(frame=frame, source_context=source_context, run_gap_hours=float(run_gap_hours))
    missing_sources = sorted(set(sources).difference(all_sources))
    if missing_sources:
        raise ValueError(f"Unknown sources: {missing_sources}")

    features = _feature_columns(frame)
    month_period = frame["__ts__"].dt.to_period("M").astype(str)
    target_mode = stage_manifest.get("grid", {}).get("target_mode", "hard_binary")
    target_rank_weight = float(stage_manifest.get("grid", {}).get("target_rank_weight", 0.30))
    target_soft_power = float(stage_manifest.get("grid", {}).get("target_soft_power", 1.0))
    min_train = int(min_train_rows if min_train_rows is not None else stage_manifest.get("min_train_rows", 500))
    min_valid = int(min_valid_rows if min_valid_rows is not None else stage_manifest.get("min_valid_rows", 100))

    period_rows: list[dict[str, Any]] = []
    proxy_diag_rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []
    source_profile_rows: list[dict[str, Any]] = []
    ledger_rows: list[dict[str, Any]] = []

    for spec_idx, spec in enumerate(specs, start=1):
        target = _target_for_spec(
            metrics.reset_index(drop=True),
            spec,
            frame["__ts__"].reset_index(drop=True),
            target_mode=target_mode,
            target_rank_weight=target_rank_weight,
            target_soft_power=target_soft_power,
        )
        target.index = frame.index
        for month in months:
            train_mask = month_period.lt(str(month))
            valid_mask = month_period.eq(str(month))
            if int(train_mask.sum()) < min_train or int(valid_mask.sum()) < min_valid:
                continue
            valid_source = frame.loc[valid_mask].copy()
            valid = valid_source.reset_index(drop=True)
            valid_metrics = metrics.loc[valid_mask].copy().reset_index(drop=True)
            valid_target = target.loc[valid_mask].copy().reset_index(drop=True)
            valid_source_masks = {
                name: all_sources[name].loc[valid_mask].copy().reset_index(drop=True)
                for name in sources
            }
            source_profile_rows.extend(
                _source_profile_rows(
                    month=str(month),
                    candidate=spec.name,
                    source_masks=valid_source_masks,
                    valid_metrics=valid_metrics,
                )
            )
            for source in sources:
                train_source_mask = (train_mask & all_sources[source]).fillna(False).astype(bool)
                valid_source_mask = valid_source_masks[source].fillna(False).astype(bool)
                source_train_rows = int(train_source_mask.sum())
                source_valid_rows = int(valid_source_mask.sum())
                if source_train_rows < int(min_source_train_rows) or source_valid_rows < int(min_source_valid_rows):
                    proxy_score = pd.Series(np.nan, index=valid_source.index, dtype=np.float32)
                    proxy_diag = {
                        "proxy_features": [],
                        "proxy_candidate_count": 0,
                        "proxy_mean_train_target_ic": float("nan"),
                        "proxy_mean_train_utility_ic": float("nan"),
                        "proxy_mean_train_bad_mae_ic": float("nan"),
                        "proxy_mean_train_timeout_ic": float("nan"),
                    }
                else:
                    proxy_score, proxy_diag = _proxy_score(
                        train=frame.loc[train_source_mask].copy(),
                        valid=valid_source,
                        features=features,
                        target_train=target.loc[train_source_mask, "target_soft"],
                        metrics_train=metrics.loc[train_source_mask].copy(),
                        top_k=int(stage_manifest.get("proxy_top_k", 4)),
                        proxy_objective=str(stage_manifest.get("proxy_objective", "economic_ic")),
                        min_target_ic=float(stage_manifest.get("proxy_min_target_ic", 0.0)),
                        min_utility_ic=float(stage_manifest.get("proxy_min_utility_ic", 0.0)),
                        max_bad_mae_ic=float(stage_manifest.get("proxy_max_bad_mae_ic", 0.0)),
                        max_wide_ic=float(stage_manifest.get("proxy_max_wide_ic", 0.0)),
                        max_timeout_ic=float(stage_manifest.get("proxy_max_timeout_ic", 0.0)),
                        utility_weight=float(stage_manifest.get("proxy_utility_weight", 1.0)),
                        bad_mae_weight=float(stage_manifest.get("proxy_bad_mae_weight", 1.0)),
                        wide_weight=float(stage_manifest.get("proxy_wide_weight", 0.5)),
                        timeout_weight=float(stage_manifest.get("proxy_timeout_weight", 0.5)),
                    )
                score = _safe_numeric(proxy_score).reset_index(drop=True)
                score.loc[~valid_source_mask.reset_index(drop=True)] = np.nan
                proxy_features = ",".join(proxy_diag.get("proxy_features", []))
                proxy_diag_rows.append(
                    {
                        "candidate": spec.name,
                        "month": str(month),
                        "source": source,
                        "train_rows": source_train_rows,
                        "valid_source_rows": source_valid_rows,
                        "proxy_candidate_count": proxy_diag.get("proxy_candidate_count", 0),
                        "proxy_features": proxy_features,
                        "proxy_mean_train_target_ic": proxy_diag.get("proxy_mean_train_target_ic"),
                        "proxy_mean_train_utility_ic": proxy_diag.get("proxy_mean_train_utility_ic"),
                        "proxy_mean_train_bad_mae_ic": proxy_diag.get("proxy_mean_train_bad_mae_ic"),
                        "proxy_mean_train_timeout_ic": proxy_diag.get("proxy_mean_train_timeout_ic"),
                    }
                )
                for top_frac in top_fracs:
                    for budget_mode in ("local", "portfolio"):
                        period_rows.extend(
                            _period_rows_for_selector(
                                valid=valid,
                                valid_metrics=valid_metrics,
                                valid_target=valid_target,
                                score=score,
                                candidate=spec.name,
                                month=str(month),
                                source=source,
                                top_frac=float(top_frac),
                                proxy_features=proxy_features,
                                budget_mode=budget_mode,
                            )
                        )
                        selector = f"source_{budget_mode}_{source}"
                        if budget_mode == "local":
                            selected_idx = _rank_top_indices(score, float(top_frac))
                        else:
                            selected_idx = _rank_top_indices_with_budget(
                                score,
                                max(1, int(math.ceil(float(top_frac) * len(valid)))),
                            )
                        confusion_rows.append(
                            _confusion_row_selected(
                                valid=valid,
                                valid_metrics=valid_metrics,
                                valid_target=valid_target,
                                candidate=spec.name,
                                month=str(month),
                                selector=selector,
                                top_frac=float(top_frac),
                                score=score,
                                selected_idx=selected_idx,
                            )
                        )
                        ledger_rows.extend(
                            _selected_ledger_rows_selected(
                                valid=valid,
                                valid_metrics=valid_metrics,
                                valid_target=valid_target,
                                candidate=spec.name,
                                month=str(month),
                                selector=selector,
                                top_frac=float(top_frac),
                                score=score,
                                selected_idx=selected_idx,
                            )
                        )
        print(json.dumps({"progress": f"{spec_idx}/{len(specs)}", "candidate": spec.name}, sort_keys=True))

    period_frame = pd.DataFrame(period_rows)
    fit_holdout = _candidate_summary(
        _fit_holdout_summary(
            period_frame,
            fit_months=[str(v) for v in fit_months],
            holdout_month=str(holdout_month),
            min_week_rows=10,
            max_timeout_rate=stage_manifest.get("max_timeout_rate"),
        )
    )
    proxy_diag = pd.DataFrame(proxy_diag_rows)
    confusion = pd.DataFrame(confusion_rows)
    source_profile = pd.DataFrame(source_profile_rows)
    ledger = pd.DataFrame(ledger_rows)

    paths = {
        "period_rows": output_dir / "pathgrid_source_conditioned_period_rows.csv",
        "fit_holdout": output_dir / "pathgrid_source_conditioned_fit_holdout.csv",
        "proxy_diag": output_dir / "pathgrid_source_conditioned_proxy_diag.csv",
        "confusion": output_dir / "pathgrid_source_conditioned_confusion.csv",
        "source_profile": output_dir / "pathgrid_source_conditioned_profile.csv",
        "selected_ledger": output_dir / "pathgrid_source_conditioned_selected_ledger.csv",
        "manifest": output_dir / "manifest.json",
    }
    period_frame.to_csv(paths["period_rows"], index=False)
    fit_holdout.to_csv(paths["fit_holdout"], index=False)
    proxy_diag.to_csv(paths["proxy_diag"], index=False)
    confusion.to_csv(paths["confusion"], index=False)
    source_profile.to_csv(paths["source_profile"], index=False)
    ledger.to_csv(paths["selected_ledger"], index=False)

    manifest = {
        "scope": "pathgrid_source_conditioned_ablation",
        "stage_dir": str(stage_dir),
        "output_dir": str(output_dir),
        "labels_path": stage_manifest["labels_path"],
        "rows": int(len(frame)),
        "feature_count": int(len(features)),
        "candidates": [spec.name for spec in specs],
        "candidate_specs": [asdict(spec) for spec in specs],
        "months": [str(v) for v in months],
        "fit_months": [str(v) for v in fit_months],
        "holdout_month": str(holdout_month),
        "top_fracs": [float(v) for v in top_fracs],
        "sources": list(sources),
        "min_source_train_rows": int(min_source_train_rows),
        "min_source_valid_rows": int(min_source_valid_rows),
        "run_gap_hours": float(run_gap_hours),
        "stage_proxy_objective": stage_manifest.get("proxy_objective"),
        "stage_proxy_top_k": stage_manifest.get("proxy_top_k"),
        "stage_target_mode": target_mode,
        "outputs": {key: str(value) for key, value in paths.items()},
        "load_reports": load_reports,
    }
    markdown = _write_report(
        output_dir=output_dir,
        fit_holdout=fit_holdout,
        period_rows=period_frame,
        proxy_diag=proxy_diag,
        confusion=confusion,
        source_profile=source_profile,
        manifest={**manifest, "outputs": {**manifest["outputs"], "markdown": str(output_dir / "pathgrid_source_conditioned_ablation.md")}},
    )
    manifest["outputs"]["markdown"] = str(markdown)
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-dir", type=Path, default=DEFAULT_STAGE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--candidates", type=lambda value: _parse_csv(value), default="")
    parser.add_argument("--max-candidates", type=int, default=6)
    parser.add_argument("--months", type=lambda value: _parse_csv(value, DEFAULT_MONTHS), default=",".join(DEFAULT_MONTHS))
    parser.add_argument(
        "--fit-months",
        type=lambda value: _parse_csv(value, DEFAULT_FIT_MONTHS),
        default=",".join(DEFAULT_FIT_MONTHS),
    )
    parser.add_argument("--holdout-month", default=DEFAULT_HOLDOUT_MONTH)
    parser.add_argument("--top-fracs", type=_parse_float_csv, default=",".join(str(v) for v in DEFAULT_TOP_FRACS))
    parser.add_argument(
        "--sources",
        type=lambda value: _parse_csv(value, DEFAULT_SOURCES),
        default=",".join(DEFAULT_SOURCES),
    )
    parser.add_argument("--min-train-rows", type=int, default=None)
    parser.add_argument("--min-valid-rows", type=int, default=None)
    parser.add_argument("--min-source-train-rows", type=int, default=200)
    parser.add_argument("--min-source-valid-rows", type=int, default=30)
    parser.add_argument("--run-gap-hours", type=float, default=6.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = run_report(
        stage_dir=args.stage_dir,
        output_dir=args.output_dir,
        candidates=list(args.candidates),
        max_candidates=int(args.max_candidates),
        months=list(args.months),
        fit_months=list(args.fit_months),
        holdout_month=str(args.holdout_month),
        top_fracs=[float(v) for v in args.top_fracs],
        sources=list(args.sources),
        min_train_rows=args.min_train_rows,
        min_valid_rows=args.min_valid_rows,
        min_source_train_rows=int(args.min_source_train_rows),
        min_source_valid_rows=int(args.min_source_valid_rows),
        run_gap_hours=float(args.run_gap_hours),
    )
    print(json.dumps(_json_safe({k: v for k, v in manifest.items() if k != "load_reports"}), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
