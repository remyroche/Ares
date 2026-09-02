#!/usr/bin/env python3
"""Diagnose soft-label oracle winners missed by causal proxy scores."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.diagnose_label_proxy_feature_gap import (  # noqa: E402
    _feature_contrasts,
    _group_metrics,
    _row_extract,
    _top_mask,
)
from scripts.run_label_quality_proxy_diagnostics import (  # noqa: E402
    DEFAULT_FEATURE_DIR,
    DEFAULT_FEATURE_LIST_CSV,
    DEFAULT_LABELS_DIR,
    _feature_columns,
    _json_safe,
    _load_feature_store_columns,
    _load_labels,
    _path_metrics,
    _read_feature_list,
    _safe_mean,
    _spearman,
)
from scripts.run_label_economic_proxy_ablation import (  # noqa: E402
    LABEL_ARMS as ECONOMIC_LABEL_ARMS,
    _label_targets as _economic_label_targets,
)
from scripts.run_soft_label_economic_proxy_ablation import (  # noqa: E402
    DEFAULT_EVENT_FEATURE_STORE_FEATURES,
    DEFAULT_PRIOR_WINDOWS_DAYS,
    DEFAULT_STATE_PATH_PRIOR_FEATURES,
    _all_targets,
    _causal_outcome_prior_features,
    _causal_state_path_prior_features,
    _event_confirmation_features,
    _parse_csv,
    _parse_float_csv,
    _proxy_score,
)


DEFAULT_OUTPUT_DIR = Path("data_perp/reports/soft_label_proxy_feature_gap_v1")


def _write_markdown(
    *,
    output_dir: Path,
    group_summary: pd.DataFrame,
    feature_contrast: pd.DataFrame,
    manifest: dict[str, Any],
) -> Path:
    path = output_dir / "soft_label_proxy_feature_gap.md"

    def table(frame: pd.DataFrame, cols: list[str], limit: int | None = None) -> str:
        if frame.empty:
            return "No rows."
        view = frame[[col for col in cols if col in frame.columns]].copy()
        if limit is not None:
            view = view.head(limit)
        for col in view.columns:
            if pd.api.types.is_float_dtype(view[col]):
                view[col] = view[col].map(lambda value: f"{float(value):.4f}" if pd.notna(value) else "")
        return view.to_markdown(index=False)

    lines = [
        "# Soft Label Proxy Feature Gap",
        "",
        "Scope: no model training. Compares soft-label oracle winners with causal feature-proxy selections.",
        "",
        f"Month: `{manifest['month']}`",
        f"Label arm: `{manifest['label_arm']}`",
        f"Oracle basis: `{manifest['oracle_basis']}`",
        f"Top fraction: `{manifest['top_frac']}`",
        f"Proxy top-k: `{manifest['proxy_top_k']}`",
        f"Outcome priors: `{manifest['include_causal_outcome_priors']}`",
        f"State-path priors: `{manifest['include_causal_state_path_priors']}`",
        f"Event confirmation features: `{manifest['include_event_confirmation_features']}`",
        "",
        "## Group Summary",
        "",
        table(
            group_summary,
            [
                "group",
                "selected_rows",
                "group_frac",
                "mean_u",
                "hit_u",
                "q10_u",
                "bad_mae_1r_rate",
                "wide_barrier_25bps_rate",
                "timeout_rate",
                "top_symbol_share",
            ],
        ),
        "",
        "## Missed Winners Vs False Positives",
        "",
        table(
            feature_contrast,
            [
                "feature",
                "missed_rank_mean",
                "false_positive_rank_mean",
                "missed_minus_false_positive_rank",
                "robust_effect_iqr",
                "missed_median",
                "false_positive_median",
            ],
            limit=40,
        ),
        "",
        "## Proxy Features",
        "",
        ", ".join(manifest.get("proxy_features", [])) or "No proxy features.",
        "",
        "## Outputs",
        "",
        f"- Group summary: `{manifest['outputs']['group_summary']}`",
        f"- Feature contrast: `{manifest['outputs']['feature_contrast']}`",
        f"- Missed winners: `{manifest['outputs']['missed_winners']}`",
        f"- False positives: `{manifest['outputs']['false_positives']}`",
        f"- Manifest: `{manifest['outputs']['manifest']}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_diagnostic(
    *,
    labels_path: Path,
    output_dir: Path,
    feature_dir: Path,
    feature_list_csv: Path,
    max_feature_store_features: int | None,
    month: str,
    label_arm: str,
    top_frac: float,
    proxy_top_k: int,
    oracle_basis: str,
    include_causal_outcome_priors: bool,
    include_causal_state_path_priors: bool,
    include_event_confirmation_features: bool,
    prior_windows_days: list[float],
    prior_embargo_hours: float,
    state_path_prior_features: list[str],
    event_feature_store_features: list[str],
    row_limit: int,
) -> dict[str, Any]:
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
    outcome_prior_report: dict[str, Any] = {"enabled": False}
    if include_causal_outcome_priors:
        prior_features, outcome_prior_report = _causal_outcome_prior_features(
            frame,
            metrics,
            windows_days=prior_windows_days,
            embargo_hours=prior_embargo_hours,
        )
        frame = pd.concat([frame, prior_features.astype(np.float32, copy=False)], axis=1).copy()

    state_path_prior_report: dict[str, Any] = {"enabled": False}
    if include_causal_state_path_priors:
        state_prior_features, state_path_prior_report = _causal_state_path_prior_features(
            frame,
            metrics,
            state_features=state_path_prior_features,
            windows_days=prior_windows_days,
            embargo_hours=prior_embargo_hours,
        )
        frame = pd.concat([frame, state_prior_features.astype(np.float32, copy=False)], axis=1).copy()

    event_confirmation_report: dict[str, Any] = {"enabled": False}
    if include_event_confirmation_features:
        event_features, event_confirmation_report = _event_confirmation_features(
            frame,
            event_features=event_feature_store_features,
        )
        frame = pd.concat([frame, event_features.astype(np.float32, copy=False)], axis=1).copy()

    targets, descriptions = _all_targets(frame, metrics)
    economic_targets = _economic_label_targets(frame, metrics)
    targets.update(economic_targets)
    descriptions.update({arm: f"economic label arm {arm}" for arm in ECONOMIC_LABEL_ARMS})
    if label_arm not in targets:
        raise ValueError(f"Unknown label arm {label_arm!r}; available arms: {sorted(targets)}")

    features = _feature_columns(frame)
    month_period = frame["__ts__"].dt.to_period("M").astype(str)
    train_mask = month_period < str(month)
    valid_mask = month_period == str(month)
    if int(train_mask.sum()) < 500 or int(valid_mask.sum()) < 100:
        raise ValueError(f"Insufficient train/valid rows for month={month}")

    train = frame.loc[train_mask].copy()
    valid = frame.loc[valid_mask].copy().reset_index(drop=True)
    valid_metrics = metrics.loc[valid_mask].copy().reset_index(drop=True)
    target = targets[label_arm]
    target_train = target.loc[train_mask, "target_soft"]
    target_valid = target.loc[valid_mask].copy().reset_index(drop=True)
    score, score_diag = _proxy_score(
        train=train,
        valid=valid,
        features=features,
        target_train=target_train,
        top_k=proxy_top_k,
    )
    score = score.reset_index(drop=True)
    if oracle_basis == "target_soft":
        oracle_score = target_valid["target_soft"]
    elif oracle_basis == "utility":
        oracle_score = valid_metrics["u_policy_net"]
    else:
        raise ValueError("oracle_basis must be target_soft or utility")

    oracle_mask = _top_mask(oracle_score, top_frac)
    proxy_mask = _top_mask(score, top_frac)
    recovered_mask = oracle_mask & proxy_mask
    missed_mask = oracle_mask & ~proxy_mask
    false_positive_mask = proxy_mask & ~oracle_mask
    neither_mask = ~(oracle_mask | proxy_mask)

    group_summary = pd.DataFrame(
        [
            _group_metrics(
                frame=valid,
                metrics=valid_metrics,
                score=score,
                target=target_valid,
                mask=mask,
                group=group,
                top_frac=top_frac,
            )
            for group, mask in [
                ("all_valid_month", pd.Series(True, index=valid.index)),
                ("oracle_top", oracle_mask),
                ("proxy_top", proxy_mask),
                ("recovered_winners", recovered_mask),
                ("missed_winners", missed_mask),
                ("false_positives", false_positive_mask),
                ("neither", neither_mask),
            ]
        ]
    )
    feature_contrast = _feature_contrasts(
        frame=valid,
        features=features,
        missed_mask=missed_mask,
        false_positive_mask=false_positive_mask,
        oracle_mask=oracle_mask,
        proxy_mask=proxy_mask,
    )
    proxy_features = list(score_diag.get("proxy_features", []))
    row_features = list(dict.fromkeys(proxy_features + feature_contrast["feature"].head(20).astype(str).tolist()))
    missed_winners = _row_extract(
        frame=valid,
        metrics=valid_metrics,
        score=score,
        target=target_valid,
        mask=missed_mask,
        features=row_features,
        limit=row_limit,
    )
    false_positives = _row_extract(
        frame=valid,
        metrics=valid_metrics,
        score=score,
        target=target_valid,
        mask=false_positive_mask,
        features=row_features,
        limit=row_limit,
    )

    paths = {
        "group_summary": output_dir / "group_summary.csv",
        "feature_contrast": output_dir / "feature_contrast_missed_vs_false_positive.csv",
        "missed_winners": output_dir / "missed_winners.csv",
        "false_positives": output_dir / "false_positives.csv",
        "manifest": output_dir / "manifest.json",
    }
    group_summary.to_csv(paths["group_summary"], index=False)
    feature_contrast.to_csv(paths["feature_contrast"], index=False)
    missed_winners.to_csv(paths["missed_winners"], index=False)
    false_positives.to_csv(paths["false_positives"], index=False)

    manifest = {
        "labels_path": str(labels_path),
        "output_dir": str(output_dir),
        "month": str(month),
        "label_arm": str(label_arm),
        "label_description": descriptions.get(label_arm, ""),
        "top_frac": float(top_frac),
        "proxy_top_k": int(proxy_top_k),
        "oracle_basis": str(oracle_basis),
        "include_causal_outcome_priors": bool(include_causal_outcome_priors),
        "include_causal_state_path_priors": bool(include_causal_state_path_priors),
        "include_event_confirmation_features": bool(include_event_confirmation_features),
        "prior_windows_days": [float(v) for v in prior_windows_days],
        "prior_embargo_hours": float(prior_embargo_hours),
        "rows": int(len(frame)),
        "train_rows": int(train_mask.sum()),
        "valid_rows": int(valid_mask.sum()),
        "feature_count": int(len(features)),
        "proxy_features": proxy_features,
        "proxy_top_abs_ic": score_diag.get("proxy_top_abs_ic"),
        "proxy_mean_top_abs_ic": score_diag.get("proxy_mean_top_abs_ic"),
        "score_ic_u_valid": _spearman(score, valid_metrics["u_policy_net"]),
        "score_ic_label_valid": _spearman(score, target_valid["target_soft"]),
        "recovered_winners": int(recovered_mask.sum()),
        "missed_winners": int(missed_mask.sum()),
        "false_positives": int(false_positive_mask.sum()),
        "oracle_top_rows": int(oracle_mask.sum()),
        "proxy_top_rows": int(proxy_mask.sum()),
        "oracle_recovery_rate": float(recovered_mask.sum() / oracle_mask.sum()) if int(oracle_mask.sum()) else 0.0,
        "feature_store": feature_store_report,
        "causal_outcome_priors": outcome_prior_report,
        "causal_state_path_priors": state_path_prior_report,
        "event_confirmation_features": event_confirmation_report,
        "outputs": {key: str(value) for key, value in paths.items()},
    }
    markdown = _write_markdown(
        output_dir=output_dir,
        group_summary=group_summary,
        feature_contrast=feature_contrast,
        manifest=manifest,
    )
    manifest["outputs"]["markdown"] = str(markdown)
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-path", type=Path, default=DEFAULT_LABELS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--feature-list-csv", type=Path, default=DEFAULT_FEATURE_LIST_CSV)
    parser.add_argument("--max-feature-store-features", type=int, default=None)
    parser.add_argument("--month", default="2026-06")
    parser.add_argument("--label-arm", default="E9_low_mae_mfe_ratio")
    parser.add_argument("--top-frac", type=float, default=0.005)
    parser.add_argument("--proxy-top-k", type=int, default=4)
    parser.add_argument("--oracle-basis", choices=("target_soft", "utility"), default="target_soft")
    parser.add_argument("--include-causal-outcome-priors", action="store_true")
    parser.add_argument("--include-causal-state-path-priors", action="store_true")
    parser.add_argument("--include-event-confirmation-features", action="store_true")
    parser.add_argument("--prior-windows-days", type=_parse_float_csv, default=",".join(str(v) for v in DEFAULT_PRIOR_WINDOWS_DAYS))
    parser.add_argument("--prior-embargo-hours", type=float, default=24.0)
    parser.add_argument(
        "--state-path-prior-features",
        type=_parse_csv,
        default=",".join(DEFAULT_STATE_PATH_PRIOR_FEATURES),
    )
    parser.add_argument(
        "--event-feature-store-features",
        type=_parse_csv,
        default=",".join(DEFAULT_EVENT_FEATURE_STORE_FEATURES),
    )
    parser.add_argument("--row-limit", type=int, default=100)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = run_diagnostic(
        labels_path=args.labels_path,
        output_dir=args.output_dir,
        feature_dir=args.feature_dir,
        feature_list_csv=args.feature_list_csv,
        max_feature_store_features=args.max_feature_store_features,
        month=str(args.month),
        label_arm=str(args.label_arm),
        top_frac=float(args.top_frac),
        proxy_top_k=int(args.proxy_top_k),
        oracle_basis=str(args.oracle_basis),
        include_causal_outcome_priors=bool(args.include_causal_outcome_priors),
        include_causal_state_path_priors=bool(args.include_causal_state_path_priors),
        include_event_confirmation_features=bool(args.include_event_confirmation_features),
        prior_windows_days=list(args.prior_windows_days),
        prior_embargo_hours=float(args.prior_embargo_hours),
        state_path_prior_features=list(args.state_path_prior_features),
        event_feature_store_features=list(args.event_feature_store_features),
        row_limit=int(args.row_limit),
    )
    print(
        json.dumps(
            _json_safe(
                {
                    key: value
                    for key, value in manifest.items()
                    if key not in {"feature_store", "causal_state_path_priors"}
                }
            ),
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
