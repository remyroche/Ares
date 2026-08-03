#!/usr/bin/env python3
"""Lock a decision-time false-positive screen on June, then test it in July.

This is deliberately a *bounded diagnostic*, not a new predictor or a policy
search.  It starts from the existing temporal-OOS direct-net and capture-only
scores, uses one pooled global direct top-10% book, and labels that book only
after the fact with the exact 12-hour return.  The June forward-control rows
are the sole discovery sample.  A field must show the same sign in both sides
there before it can be put in a frozen, equal-weight family composite.  July is
read only after that frozen list, centring, scale and threshold have been
written.

No realized-path field, target/support label, calendar field, action, exit or
portfolio field may enter the screen.  The permitted inputs are exactly the
numeric live-aligned columns in the canonical current meta input plus a small,
auditable candidate-context set derived at decision time from the already-OOS
direct/capture scores (their contemporaneous timestamp ranks and disagreement).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


IDENTITY = ("__ts__", "__symbol__", "side_name", "candidate_id")
TARGET = "execution_net_ev_12h"
DIRECT_ARM = "direct_net"
CAPTURE_ARM = "capture_only"
MAPPING_STAGE = "canonical_recent_ev_mapping"
CONTROL_WINDOW = "may_to_june_forward_control"
LATER_WINDOW = "later_july_forward"
TOP_FRACTION = 0.10
HIGH_SURPLUS_BPS = 50.0
MIN_CLASS_ROWS = 100
MIN_SIDE_CLASS_ROWS = 20
MIN_SIDE_EFFECT = 0.05
MIN_POOLED_EFFECT = 0.12
MAX_FAMILIES = 4

DEFAULT_JOINED = Path(
    "data_perp/artifacts/execution_ev_canonical_exact_policy_regime_input_20260727_v3/joined.parquet"
)
DEFAULT_PREDICTIONS = Path(
    "data_perp/artifacts/exact_policy_capture_support_ablation_20260727_v8/capture_support_predictions.parquet"
)
DEFAULT_OUTPUT = Path(
    "data_perp/artifacts/execution_ev_false_positive_feature_diagnosis_20260727_v2"
)

# These are realized outcomes, support-label materializations, policy actions,
# or calendar shortcuts.  The prefix guard below also protects future schema
# additions.  Existing OOF predictions (for example pred_peak_MFE_12h_ATR) are
# allowed because they are frozen decision-time values in the live meta input,
# not realised labels.
FORBIDDEN_EXACT = {
    TARGET,
    "execution_gross_ev_12h",
    "execution_cost_return",
    "execution_exit_reason",
    "execution_exit_hour",
    "execution_mfe_return_12h",
    "execution_mae_return_12h",
    "execution_decision_utc",
    "execution_label_end_utc",
    "oof_entry_atr_fraction",
}
FORBIDDEN_TOKENS = (
    "target",
    "label",
    "realized",
    "exit",
    "mfe_return",
    "mae_return",
    "time_to",
    "bar_before",
    "action",
    "portfolio",
    "hour_",
    "dow_",
    "weekday",
    "month",
    "calendar",
)


def _sha(path: Path) -> str:
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
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(_safe(payload), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _numeric(values: Any) -> pd.Series:
    return pd.to_numeric(values, errors="coerce")


def _family(column: str) -> str:
    if column.startswith("ctx_"):
        return "candidate_score_context"
    if column == "existing_alpha_ev":
        return "alpha_ev"
    if column.startswith("pred_peak_") or column.startswith("oof_clean_"):
        return "frozen_auxiliary_prediction"
    if column.startswith("catboost_p_") or column == "catboost_entropy":
        return "catboost_geometry"
    if column.startswith("alpha_prediction_") or column.startswith("alpha_leaf_"):
        return "alpha_confidence_support"
    if column.startswith("base_oof_") or column.startswith("base_margin_"):
        return "base_candidate_context"
    if column.startswith("base_archetype_label__"):
        return "base_archetype"
    return "live_meta_input"


def _is_allowed_feature(column: str, series: pd.Series) -> bool:
    lower = column.lower()
    if column in IDENTITY or column in FORBIDDEN_EXACT:
        return False
    if any(token in lower for token in FORBIDDEN_TOKENS):
        return False
    return pd.api.types.is_numeric_dtype(series)


def allowed_live_features(frame: pd.DataFrame) -> list[str]:
    """Return only current decision-time numeric fields, never targets/actions."""
    return [
        column
        for column in frame.columns
        if _is_allowed_feature(column, frame[column])
    ]


def add_decision_time_context(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach contemporaneous-only score context without any calendar shortcut."""
    work = frame.copy()
    direct = _numeric(work["direct_mapped_score"])
    capture = _numeric(work["capture_mapped_score"])
    for name, values in (("direct", direct), ("capture", capture)):
        # Each timestamp's eligible candidates are observable together at
        # inference.  This is not a rank over a future day/week/month.
        work[f"ctx_{name}_timestamp_rank"] = values.groupby(work["__ts__"], dropna=False).rank(
            method="average", pct=True
        )
    work["ctx_direct_minus_capture"] = direct - capture
    # Difference of contemporaneous ranks is scale-free and decision-time.
    work["ctx_direct_minus_capture_timestamp_rank"] = (
        work["ctx_direct_timestamp_rank"] - work["ctx_capture_timestamp_rank"]
    )
    work["ctx_candidate_group_size"] = (
        work.groupby("__ts__", dropna=False)["candidate_id"].transform("size").astype(float)
    )
    return work


def add_classes(frame: pd.DataFrame, *, top_fraction: float = TOP_FRACTION) -> pd.DataFrame:
    """Form an exhaustive, post-outcome four-cell diagnostic partition."""
    work = frame.copy()
    work["direct_selected_global_top10"] = False
    count = max(1, int(math.ceil(top_fraction * len(work))))
    selected = work.nlargest(count, "direct_mapped_score", keep="first").index
    work.loc[selected, "direct_selected_global_top10"] = True
    work["capture_selected_global_top10"] = False
    capture_selected = work.nlargest(count, "capture_mapped_score", keep="first").index
    work.loc[capture_selected, "capture_selected_global_top10"] = True
    work["high_surplus_50bps"] = _numeric(work[TARGET]).ge(HIGH_SURPLUS_BPS / 10_000.0)
    selected_mask = work["direct_selected_global_top10"]
    high_mask = work["high_surplus_50bps"]
    work["diagnostic_class"] = np.select(
        [selected_mask & high_mask, selected_mask & ~high_mask, ~selected_mask & high_mask],
        ["true_positive", "false_positive", "missed_high_surplus_winner"],
        default="true_negative",
    )
    return work


def _pooled_std(left: pd.Series, right: pd.Series) -> float:
    combined = pd.concat([left, right], ignore_index=True).dropna()
    if len(combined) < 3:
        return float("nan")
    scale = float(combined.std(ddof=0))
    return scale if math.isfinite(scale) and scale > 1e-12 else float("nan")


def _effect(true_positive: pd.Series, false_positive: pd.Series) -> tuple[float, int, int]:
    left = _numeric(true_positive).dropna()
    right = _numeric(false_positive).dropna()
    scale = _pooled_std(left, right)
    effect = (float(left.mean()) - float(right.mean())) / scale if math.isfinite(scale) else float("nan")
    return effect, int(len(left)), int(len(right))


def _robust_center_scale(values: pd.Series) -> tuple[float, float]:
    clean = _numeric(values).dropna()
    if clean.empty:
        return float("nan"), float("nan")
    center = float(clean.median())
    scale = float(clean.quantile(0.75) - clean.quantile(0.25))
    if not math.isfinite(scale) or scale <= 1e-12:
        scale = float(clean.std(ddof=0))
    return center, scale if math.isfinite(scale) and scale > 1e-12 else float("nan")


def control_contrasts(control: pd.DataFrame, features: Iterable[str]) -> pd.DataFrame:
    """Compute side-preserving true-positive minus false-positive effects."""
    selected = control.loc[control["direct_selected_global_top10"]].copy()
    rows: list[dict[str, Any]] = []
    for feature in features:
        true_positive = selected.loc[selected["diagnostic_class"].eq("true_positive"), feature]
        false_positive = selected.loc[selected["diagnostic_class"].eq("false_positive"), feature]
        pooled, n_tp, n_fp = _effect(true_positive, false_positive)
        row: dict[str, Any] = {
            "feature": feature,
            "family": _family(feature),
            "control_pooled_effect_tp_minus_fp": pooled,
            "control_true_positive_rows": n_tp,
            "control_false_positive_rows": n_fp,
            "control_true_positive_mean": float(_numeric(true_positive).mean()),
            "control_false_positive_mean": float(_numeric(false_positive).mean()),
            "control_nonmissing_fraction": float(_numeric(selected[feature]).notna().mean()),
        }
        side_effects: list[float] = []
        for side in ("long", "short"):
            side_selected = selected.loc[selected["side_name"].eq(side)]
            effect, side_tp, side_fp = _effect(
                side_selected.loc[side_selected["diagnostic_class"].eq("true_positive"), feature],
                side_selected.loc[side_selected["diagnostic_class"].eq("false_positive"), feature],
            )
            row[f"control_{side}_effect_tp_minus_fp"] = effect
            row[f"control_{side}_true_positive_rows"] = side_tp
            row[f"control_{side}_false_positive_rows"] = side_fp
            side_effects.append(effect)
        valid = [value for value in side_effects if math.isfinite(value)]
        row["control_side_sign_agreement"] = bool(
            len(valid) == 2 and valid[0] * valid[1] > 0.0
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        "control_pooled_effect_tp_minus_fp", key=lambda series: series.abs(), ascending=False
    ).reset_index(drop=True)


def freeze_screens(control: pd.DataFrame, contrasts: pd.DataFrame) -> pd.DataFrame:
    """Select at most one predeclared robust field per family on control only."""
    eligible = contrasts.copy()
    side_effects = [
        "control_long_effect_tp_minus_fp",
        "control_short_effect_tp_minus_fp",
    ]
    eligible = eligible.loc[
        eligible["control_pooled_effect_tp_minus_fp"].abs().ge(MIN_POOLED_EFFECT)
        & eligible["control_side_sign_agreement"]
        & eligible[side_effects].abs().min(axis=1).ge(MIN_SIDE_EFFECT)
        & eligible["control_true_positive_rows"].ge(MIN_CLASS_ROWS)
        & eligible["control_false_positive_rows"].ge(MIN_CLASS_ROWS)
        & eligible[[
            "control_long_true_positive_rows",
            "control_long_false_positive_rows",
            "control_short_true_positive_rows",
            "control_short_false_positive_rows",
        ]].min(axis=1).ge(MIN_SIDE_CLASS_ROWS)
        & eligible["control_nonmissing_fraction"].ge(0.95)
    ].copy()
    eligible["abs_effect"] = eligible["control_pooled_effect_tp_minus_fp"].abs()
    eligible = eligible.sort_values(["family", "abs_effect", "feature"], ascending=[True, False, True])
    frozen = eligible.groupby("family", as_index=False, sort=True).head(1).copy()
    frozen = frozen.sort_values(["abs_effect", "feature"], ascending=[False, True]).head(MAX_FAMILIES)
    selected_book = control.loc[control["direct_selected_global_top10"]]
    rows: list[dict[str, Any]] = []
    for _, value in frozen.iterrows():
        feature = str(value["feature"])
        center, scale = _robust_center_scale(control[feature])
        threshold, _ = _robust_center_scale(selected_book[feature])
        if not math.isfinite(center) or not math.isfinite(scale) or not math.isfinite(threshold):
            continue
        # A fixed 50% selected-book retention rule: the median is outcome-free
        # conditional on the *already selected* control candidate set.  It is
        # not a threshold optimized for June economic performance.
        rows.append(
            {
                "feature": feature,
                "family": str(value["family"]),
                "direction_tp_over_fp": float(np.sign(value["control_pooled_effect_tp_minus_fp"])),
                "control_pooled_effect_tp_minus_fp": float(value["control_pooled_effect_tp_minus_fp"]),
                "frozen_control_center": center,
                "frozen_control_scale": scale,
                "frozen_selected_book_median": threshold,
                "screen_rule": "retain_if_directional_value_at_or_above_control_selected_median",
            }
        )
    columns = (
        "feature",
        "family",
        "direction_tp_over_fp",
        "control_pooled_effect_tp_minus_fp",
        "frozen_control_center",
        "frozen_control_scale",
        "frozen_selected_book_median",
        "screen_rule",
    )
    result = pd.DataFrame(rows, columns=columns)
    if result.empty:
        return result
    return result.sort_values(
        "control_pooled_effect_tp_minus_fp", key=lambda s: s.abs(), ascending=False
    ).reset_index(drop=True)


def apply_frozen_screens(frame: pd.DataFrame, screens: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    score_columns: list[str] = []
    pass_columns: list[str] = []
    for _, screen in screens.iterrows():
        feature = str(screen["feature"])
        name = "screen_" + hashlib.sha1(feature.encode()).hexdigest()[:10]
        directional = float(screen["direction_tp_over_fp"]) * (
            (_numeric(work[feature]) - float(screen["frozen_control_center"]))
            / float(screen["frozen_control_scale"])
        )
        threshold = float(screen["direction_tp_over_fp"]) * (
            float(screen["frozen_selected_book_median"]) - float(screen["frozen_control_center"])
        ) / float(screen["frozen_control_scale"])
        work[f"{name}_score"] = directional
        work[f"{name}_pass"] = directional.ge(threshold)
        score_columns.append(f"{name}_score")
        pass_columns.append(f"{name}_pass")
    if score_columns:
        work["frozen_equal_weight_composite"] = work[score_columns].mean(axis=1, skipna=True)
        # Composite threshold is deliberately zero after robust control
        # centring, not outcome-optimized.  It is reported only as a screen.
        work["frozen_equal_weight_composite_pass"] = work[
            "frozen_equal_weight_composite"
        ].ge(0.0)
        work["frozen_component_pass_count"] = work[pass_columns].sum(axis=1)
    else:
        work["frozen_equal_weight_composite"] = np.nan
        work["frozen_equal_weight_composite_pass"] = False
        work["frozen_component_pass_count"] = 0
    return work


def _class_counts(frame: pd.DataFrame, *, window: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group_name, group in (("all_candidates", frame), ("direct_selected", frame.loc[frame["direct_selected_global_top10"]])):
        for label in (
            "true_positive",
            "false_positive",
            "missed_high_surplus_winner",
            "true_negative",
        ):
            rows.append({
                "window": window,
                "scope": group_name,
                "diagnostic_class": label,
                "rows": int(group["diagnostic_class"].eq(label).sum()),
                "fraction": float(group["diagnostic_class"].eq(label).mean()) if len(group) else float("nan"),
            })
        rows.append({
            "window": window,
            "scope": group_name,
            "diagnostic_class": "capture_direct_top10_overlap",
            "rows": int((group["direct_selected_global_top10"] & group["capture_selected_global_top10"]).sum()),
            "fraction": float((group["direct_selected_global_top10"] & group["capture_selected_global_top10"]).mean()) if len(group) else float("nan"),
        })
    return rows


def screen_metrics(frame: pd.DataFrame, screens: pd.DataFrame, *, window: str) -> pd.DataFrame:
    """Assess the locked June fields/composite; July never feeds selection."""
    selected = frame.loc[frame["direct_selected_global_top10"]].copy()
    rows: list[dict[str, Any]] = []
    candidates: list[tuple[str, pd.Series, pd.Series]] = []
    for _, screen in screens.iterrows():
        feature = str(screen["feature"])
        token = "screen_" + hashlib.sha1(feature.encode()).hexdigest()[:10]
        candidates.append((feature, selected[f"{token}_score"], selected[f"{token}_pass"]))
    if len(screens) >= 2:
        candidates.append(
            (
                "frozen_equal_weight_composite",
                selected["frozen_equal_weight_composite"],
                selected["frozen_equal_weight_composite_pass"],
            )
        )
    for name, score, passed in candidates:
        positive = selected.loc[selected["diagnostic_class"].eq("true_positive"), score.name]
        negative = selected.loc[selected["diagnostic_class"].eq("false_positive"), score.name]
        effect, n_tp, n_fp = _effect(positive, negative)
        keep = selected.loc[passed.fillna(False)]
        drop = selected.loc[~passed.fillna(False)]
        rows.append({
            "window": window,
            "screen": name,
            "selected_rows": int(len(selected)),
            "screen_keep_rows": int(len(keep)),
            "screen_keep_fraction": float(len(keep) / len(selected)) if len(selected) else float("nan"),
            "tp_minus_fp_effect": effect,
            "true_positive_rows_for_effect": n_tp,
            "false_positive_rows_for_effect": n_fp,
            "screen_keep_high_surplus_rate": float(keep["high_surplus_50bps"].mean()) if len(keep) else float("nan"),
            "screen_drop_high_surplus_rate": float(drop["high_surplus_50bps"].mean()) if len(drop) else float("nan"),
            "screen_keep_net_bps": float(_numeric(keep[TARGET]).mean() * 1e4) if len(keep) else float("nan"),
            "screen_drop_net_bps": float(_numeric(drop[TARGET]).mean() * 1e4) if len(drop) else float("nan"),
            "screen_keep_capture_selected_rate": float(keep["capture_selected_global_top10"].mean()) if len(keep) else float("nan"),
        })
    return pd.DataFrame(rows)


def stability_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    """Separate locked classification stability from economic stability.

    A positive later-July class contrast alone is insufficient.  Economic
    stability requires positive high-surplus and exact-net lifts in each
    window.  This remains diagnostic evidence, never a promotion decision.
    """
    columns = [
        "screen", "control_effect", "later_july_effect", "effect_sign_stable",
        "control_high_surplus_lift", "later_july_high_surplus_lift",
        "high_surplus_lift_stable", "control_net_lift_bps",
        "later_july_net_lift_bps", "economic_lift_stable", "status",
    ]
    if metrics.empty:
        return pd.DataFrame(columns=columns)
    pivot = metrics.pivot(index="screen", columns="window")
    rows: list[dict[str, Any]] = []
    for screen in pivot.index:
        try:
            control_effect = float(pivot.loc[screen, ("tp_minus_fp_effect", CONTROL_WINDOW)])
            later_effect = float(pivot.loc[screen, ("tp_minus_fp_effect", LATER_WINDOW)])
            control_high_lift = float(
                pivot.loc[screen, ("screen_keep_high_surplus_rate", CONTROL_WINDOW)]
                - pivot.loc[screen, ("screen_drop_high_surplus_rate", CONTROL_WINDOW)]
            )
            later_high_lift = float(
                pivot.loc[screen, ("screen_keep_high_surplus_rate", LATER_WINDOW)]
                - pivot.loc[screen, ("screen_drop_high_surplus_rate", LATER_WINDOW)]
            )
            control_net_lift = float(
                pivot.loc[screen, ("screen_keep_net_bps", CONTROL_WINDOW)]
                - pivot.loc[screen, ("screen_drop_net_bps", CONTROL_WINDOW)]
            )
            later_net_lift = float(
                pivot.loc[screen, ("screen_keep_net_bps", LATER_WINDOW)]
                - pivot.loc[screen, ("screen_drop_net_bps", LATER_WINDOW)]
            )
        except KeyError:
            continue
        sign_stable = bool(
            math.isfinite(control_effect)
            and math.isfinite(later_effect)
            and control_effect * later_effect > 0.0
        )
        high_stable = bool(control_high_lift > 0.0 and later_high_lift > 0.0)
        economic_stable = bool(high_stable and control_net_lift > 0.0 and later_net_lift > 0.0)
        rows.append({
            "screen": screen,
            "control_effect": control_effect,
            "later_july_effect": later_effect,
            "effect_sign_stable": sign_stable,
            "control_high_surplus_lift": control_high_lift,
            "later_july_high_surplus_lift": later_high_lift,
            "high_surplus_lift_stable": high_stable,
            "control_net_lift_bps": control_net_lift,
            "later_july_net_lift_bps": later_net_lift,
            "economic_lift_stable": economic_stable,
            "status": (
                "locked_diagnostic_economic_lift_only"
                if economic_stable else "not_economically_stable"
            ),
        })
    return pd.DataFrame(rows, columns=columns)


def _load(joined_path: Path, prediction_path: Path) -> pd.DataFrame:
    joined = pd.read_parquet(joined_path)
    predictions = pd.read_parquet(prediction_path)
    for frame in (joined, predictions):
        frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True, errors="raise")
        frame["side_name"] = frame["side_name"].astype(str).str.lower()
        frame["__symbol__"] = frame["__symbol__"].astype(str)
        frame["candidate_id"] = frame["candidate_id"].astype(str)
    if joined.duplicated(list(IDENTITY)).any():
        raise ValueError("canonical joined input has duplicate identities")
    predictions = predictions.loc[
        predictions["window"].isin((CONTROL_WINDOW, LATER_WINDOW))
        & predictions["arm"].isin((DIRECT_ARM, CAPTURE_ARM))
        & predictions["mapping_stage"].eq(MAPPING_STAGE)
    ].copy()
    if predictions.empty:
        raise ValueError("missing required current mapped direct/capture score rows")
    pivot = predictions.pivot(index=list(IDENTITY), columns="arm", values="canonical_recent_ev_score").reset_index()
    pivot.columns.name = None
    pivot = pivot.rename(columns={DIRECT_ARM: "direct_mapped_score", CAPTURE_ARM: "capture_mapped_score"})
    window_check = predictions.loc[:, [*IDENTITY, "window"]].drop_duplicates()
    if window_check.duplicated(list(IDENTITY)).any():
        raise ValueError("an identity appears in more than one forward window")
    pivot = pivot.merge(window_check, on=list(IDENTITY), how="left", validate="one_to_one")
    required = {"direct_mapped_score", "capture_mapped_score"}
    if missing := sorted(required - set(pivot)):
        raise ValueError("score pivot missing arms: " + ", ".join(missing))
    merged = pivot.merge(joined, on=list(IDENTITY), how="left", validate="one_to_one")
    if len(merged) != len(pivot) or merged[TARGET].isna().any():
        raise ValueError("incomplete mapped-score to canonical-input/target join")
    if not np.isfinite(merged[["direct_mapped_score", "capture_mapped_score", TARGET]].to_numpy(float)).all():
        raise ValueError("nonfinite mapped score or exact net target")
    return merged


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    raw = _load(args.joined, args.predictions)
    raw = add_decision_time_context(raw)
    features = allowed_live_features(raw)
    # Scores are permitted decision-time candidate context.  The exact target
    # is absent from `features` and is used solely below to form diagnostic
    # classes/metrics after score selection.
    context = [
        "direct_mapped_score",
        "capture_mapped_score",
        "ctx_direct_timestamp_rank",
        "ctx_capture_timestamp_rank",
        "ctx_direct_minus_capture",
        "ctx_direct_minus_capture_timestamp_rank",
        "ctx_candidate_group_size",
    ]
    features = [*features, *context]
    features = list(dict.fromkeys(column for column in features if column not in FORBIDDEN_EXACT))
    windows = {
        window: add_classes(group.reset_index(drop=True))
        for window, group in raw.groupby("window", sort=False)
    }
    control = windows.get(CONTROL_WINDOW)
    later = windows.get(LATER_WINDOW)
    if control is None or later is None:
        raise ValueError("both May-to-June control and later-July forward windows are required")
    contrasts = control_contrasts(control, features)
    screens = freeze_screens(control, contrasts)
    control_scored = apply_frozen_screens(control, screens)
    later_scored = apply_frozen_screens(later, screens)
    assignments = pd.concat([control_scored, later_scored], ignore_index=True)
    metrics = pd.concat(
        [
            screen_metrics(control_scored, screens, window=CONTROL_WINDOW),
            screen_metrics(later_scored, screens, window=LATER_WINDOW),
        ],
        ignore_index=True,
    )
    stability = stability_summary(metrics)
    class_counts = pd.DataFrame(
        _class_counts(control_scored, window=CONTROL_WINDOW)
        + _class_counts(later_scored, window=LATER_WINDOW)
    )
    args.output_dir.mkdir(parents=True)
    paths = {
        "control_feature_contrasts": args.output_dir / "control_feature_contrasts.csv",
        "frozen_screens": args.output_dir / "frozen_screens.csv",
        "screen_metrics": args.output_dir / "screen_metrics.csv",
        "stability_summary": args.output_dir / "stability_summary.csv",
        "class_counts": args.output_dir / "class_counts.csv",
        "assignments": args.output_dir / "candidate_assignments.parquet",
    }
    contrasts.to_csv(paths["control_feature_contrasts"], index=False)
    screens.to_csv(paths["frozen_screens"], index=False)
    metrics.to_csv(paths["screen_metrics"], index=False)
    stability.to_csv(paths["stability_summary"], index=False)
    class_counts.to_csv(paths["class_counts"], index=False)
    assignment_columns = [
        *IDENTITY,
        "window",
        TARGET,
        "direct_mapped_score",
        "capture_mapped_score",
        "direct_selected_global_top10",
        "capture_selected_global_top10",
        "high_surplus_50bps",
        "diagnostic_class",
        "frozen_equal_weight_composite",
        "frozen_equal_weight_composite_pass",
        "frozen_component_pass_count",
    ]
    assignments.loc[:, [column for column in assignment_columns if column in assignments]].to_parquet(
        paths["assignments"], index=False, compression="zstd"
    )
    report = args.output_dir / "REPORT.md"
    report.write_text(
        "# Frozen June false-positive feature diagnostic\n\n"
        "This is a bounded decision-time diagnostic, not a promoted gate, model or HPO result. "
        "The direct top-10% book and a 50-bps realized-surplus class define post-outcome "
        "diagnostic groups. Feature selection, signs, robust scales and thresholds use only "
        "the May-to-June OOS control. Later July is a locked stability check.\n\n"
        "- `true_positive`: direct-selected and realized exact net >= 50 bps.\n"
        "- `false_positive`: direct-selected and below 50 bps.\n"
        "- `missed_high_surplus_winner`: not direct-selected and >= 50 bps.\n"
        "- `true_negative`: not direct-selected and below 50 bps.\n\n"
        "A field needs >= 100 control true/false positives, >= 20 rows in every side/class cell, "
        ">=95% selected-book coverage, a pooled TP-minus-FP standardized effect of at least 0.12, "
        "and matching long/short signs with each absolute effect >=0.05. At most one field per family and four families "
        "are frozen. The retention rule is a non-economic control selected-book median; it is "
        "not tuned to June return.\n\n"
        "See `frozen_screens.csv`, `screen_metrics.csv` and `stability_summary.csv`; July must preserve direction and "
        "high-surplus/net lift before any subsequent predeclared ablation is authorized. The June long selected-book support "
        "is explicitly limited and does not justify promotion.\n",
        encoding="utf-8",
    )
    paths["report"] = report
    manifest = {
        "schema": "execution_ev_false_positive_feature_diagnosis_v1",
        "status": "completed_diagnostic_not_model_or_promotion_evidence",
        "contract": {
            "discovery": "May-to-June forward OOS control only",
            "locked_assessment": "later-July forward OOS only; no July feature/composite/threshold selection",
            "selection": "one pooled global direct-score top 10%; no time/side/asset quota",
            "classes": "post-outcome exact net >= 50bps used for diagnostic labels only",
            "features": "current live-aligned canonical meta inputs plus decision-time direct/capture context only",
            "forbidden": "realized outcomes/support labels, calendar shortcuts, policy actions, exits and portfolio fields",
            "model_hpo": "none",
        },
        "parameters": {
            "high_surplus_bps": HIGH_SURPLUS_BPS,
            "top_fraction": TOP_FRACTION,
            "min_class_rows": MIN_CLASS_ROWS,
            "min_side_class_rows": MIN_SIDE_CLASS_ROWS,
            "min_pooled_effect": MIN_POOLED_EFFECT,
            "min_side_effect": MIN_SIDE_EFFECT,
            "max_families": MAX_FAMILIES,
        },
        "inputs": {
            "joined": {"path": str(args.joined), "sha256": _sha(args.joined)},
            "mapped_predictions": {"path": str(args.predictions), "sha256": _sha(args.predictions)},
            "mapped_arms": [DIRECT_ARM, CAPTURE_ARM],
            "mapping_stage": MAPPING_STAGE,
        },
        "feature_inventory": {
            "allowed_count": len(features),
            "allowed_features": features,
            "frozen_count": len(screens),
            "frozen_features": screens["feature"].tolist(),
        },
        "outputs": {name: {"path": str(path), "sha256": _sha(path)} for name, path in paths.items()},
    }
    _write_json(args.output_dir / "manifest.json", manifest)
    return manifest


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--joined", type=Path, default=DEFAULT_JOINED)
    result.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    result.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    print(json.dumps(run(args), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
