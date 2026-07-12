#!/usr/bin/env python3
"""Compare current, PCA-overlay, and retrained residual-meta arms on aligned OOS rows."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.report_meta_residual_historical_rank import (  # noqa: E402
    FRACTIONS,
    SCOPES,
    _calendar,
    _record,
)
from scripts.run_meta_residual_retrained_blend_ablation import (  # noqa: E402
    BLEND_ARM,
    FORCED_ARM,
)
from scripts.run_train_meta_residual_archetype_enhancement import (
    DEFAULT_OUT_DIR,  # noqa: E402
)
from scripts.run_train_meta_residual_surprise_head_retrained import (
    ARM as RETRAINED_ARM,  # noqa: E402
)

PCA_ARM = "lifecycle_residual_aware_ae_gmm_overlay_pca8_clip8_baseline_globaloverlay"
KEYS = ["__ts__", "__symbol__", "side_name", "archetype_policy_key"]


def _source_path(root: Path, arm: str) -> Path:
    return (
        root / f"historical_rank_oos_{arm}" / "oos_predictions_historical_rank.parquet"
    )


def _load(root: Path, arm: str) -> pd.DataFrame:
    frame = pd.read_parquet(_source_path(root, arm))
    frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True, errors="coerce")
    frame["calendar_month"] = frame["calendar_month"].astype(str)
    return frame


def _aligned_sources(root: Path) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    arms = [PCA_ARM, RETRAINED_ARM, FORCED_ARM, BLEND_ARM]
    sources = {arm: _load(root, arm) for arm in arms}
    key_frames = {arm: frame[KEYS].drop_duplicates() for arm, frame in sources.items()}
    overlap = key_frames[arms[0]]
    for arm in arms[1:]:
        overlap = overlap.merge(key_frames[arm], on=KEYS, how="inner")
    sources = {
        arm: frame.merge(overlap, on=KEYS, how="inner", validate="one_to_one")
        for arm, frame in sources.items()
    }
    alignment = {"overlap_rows": int(len(overlap))}
    for arm in arms:
        alignment[f"{arm}__rows"] = int(len(key_frames[arm]))
        alignment[f"{arm}__overlap_rate"] = float(
            len(overlap) / max(len(key_frames[arm]), 1)
        )
    return sources, alignment


def _metric_rows(
    frame: pd.DataFrame,
    *,
    selector: str,
    rank_col: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scope, group_cols in SCOPES.items():
        grouped: Iterable[tuple[Any, pd.DataFrame]] = (
            [((), frame)]
            if not group_cols
            else frame.groupby(group_cols, dropna=False, sort=True)
        )
        for key, group in grouped:
            values = key if isinstance(key, tuple) else (key,)
            rank = pd.to_numeric(group[rank_col], errors="coerce")
            for fraction in FRACTIONS:
                row = _record(group, rank.ge(1.0 - fraction), selector, fraction)
                row["scope"] = scope
                for name, value in zip(group_cols, values, strict=False):
                    row[name] = value
                rows.append(row)
    return rows


def _selector_summary(metrics: pd.DataFrame, selector: str) -> dict[str, Any]:
    overall = metrics[
        metrics["scope"].eq("overall")
        & metrics["fraction"].eq(0.10)
        & metrics["selector"].eq(selector)
    ].iloc[0]
    weeks = metrics[
        metrics["scope"].eq("week")
        & metrics["fraction"].eq(0.10)
        & metrics["selector"].eq(selector)
    ]
    months = metrics[
        metrics["scope"].eq("month")
        & metrics["fraction"].eq(0.10)
        & metrics["selector"].eq(selector)
    ]
    return {
        "selected_rows": int(overall["selected_rows"]),
        "mean_ev_after_1pct": float(overall["mean_ev_after_1pct"]),
        "clean_exec_precision": float(overall["clean_exec_precision"]),
        "full_path_bad_mae_rate": float(overall["full_path_bad_mae_rate"]),
        "timeout_rate": float(overall["timeout_rate"]),
        "worst_week_ev": float(weeks["mean_ev_after_1pct"].min()),
        "worst_month_ev": float(months["mean_ev_after_1pct"].min()),
        "positive_weeks": int(weeks["mean_ev_after_1pct"].gt(0.0).sum()),
        "weeks": int(len(weeks)),
    }


def _autocorr(frame: pd.DataFrame, selector: str) -> tuple[float, pd.DataFrame]:
    _calendar_rows, autocorr, comparison = _calendar(frame, selector)
    values = pd.to_numeric(
        autocorr.loc[autocorr["selector"].eq(selector), "surprise_autocorr_lag1"],
        errors="coerce",
    )
    return float(values.abs().mean()), comparison


def main() -> None:
    root = DEFAULT_OUT_DIR
    out_dir = root / "final_report"
    out_dir.mkdir(parents=True, exist_ok=True)
    sources, alignment = _aligned_sources(root)
    retrained = sources[RETRAINED_ARM]
    rows: list[dict[str, Any]] = []
    rows.extend(
        _metric_rows(
            retrained,
            selector="current_reference",
            rank_col="historical_rank_current_reference",
        )
    )
    for arm, frame in sources.items():
        rows.extend(
            _metric_rows(
                frame,
                selector=arm,
                rank_col="historical_rank_alternative",
            )
        )
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / "retrained_aligned_metrics_by_scope.csv", index=False)

    arm_ac: dict[str, float] = {}
    for arm, frame in sources.items():
        ac, events = _autocorr(frame, arm)
        arm_ac[arm] = ac
        events.assign(selector=arm).to_csv(
            out_dir / f"retrained_comparison_{arm}_events.csv", index=False
        )
    current_calendar = pd.read_csv(
        root
        / f"historical_rank_oos_{RETRAINED_ARM}"
        / "hit_surprise_autocorrelation.csv"
    )
    current_values = pd.to_numeric(
        current_calendar.loc[
            current_calendar["selector"].eq("current_reference"),
            "surprise_autocorr_lag1",
        ],
        errors="coerce",
    )
    current_ac = float(current_values.abs().mean())
    summaries = {
        selector: _selector_summary(metrics, selector)
        for selector in ("current_reference", *sources.keys())
    }
    summaries["current_reference"]["mean_abs_surprise_autocorr_lag1"] = current_ac
    for arm, ac in arm_ac.items():
        summaries[arm]["mean_abs_surprise_autocorr_lag1"] = ac
    summary_frame = pd.DataFrame(
        [{"selector": selector, **values} for selector, values in summaries.items()]
    )
    summary_frame.to_csv(out_dir / "retrained_aligned_top10_summary.csv", index=False)
    manifest = {
        "schema": "meta_residual_retrained_aligned_comparison_v1",
        "alignment": alignment,
        "selectors": summaries,
        "cost_contract": "ev_after_1pct includes 1% round-trip cost",
        "rank_contract": "expanding prior score CDF by side",
        "evaluation_months": ["2026-04", "2026-05", "2026-06"],
    }
    (out_dir / "retrained_aligned_comparison_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
