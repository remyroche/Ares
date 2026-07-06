#!/usr/bin/env python3
"""Run strict-OOS repair validation only for frozen profiles.

This orchestrates two diagnostic-only steps:

1. build a month-forward repair-ranker ledger for each pre-registered profile,
   using singleton source/proxy/top-frac/feature/selection settings;
2. validate the frozen profiles against hard guards.

It intentionally avoids the broad repair-ranker grid so later untouched periods
can be evaluated without profile re-selection.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_label_quality_proxy_diagnostics import (  # noqa: E402
    DEFAULT_FEATURE_DIR,
    DEFAULT_FEATURE_LIST_CSV,
    DEFAULT_LABELS_DIR,
    _json_safe,
)
from scripts.run_strict_oos_repair_ranker_ablation import (  # noqa: E402
    DEFAULT_EVENT_ROWS,
    DEFAULT_PREDICTIONS,
    DEFAULT_QUALITY_LABELS,
    DEFAULT_OUTPUT_DIR as DEFAULT_BROAD_REPAIR_OUTPUT_DIR,
    run_report as run_repair_report,
)
from scripts.validate_strict_oos_repair_ranker_frozen_profiles import (  # noqa: E402
    DEFAULT_OUTPUT_DIR as DEFAULT_VALIDATION_OUTPUT_DIR,
    DEFAULT_PROFILE_MANIFEST,
    PROFILE_KEYS,
    load_profile_manifest,
    run_validation,
)


DEFAULT_OUTPUT_DIR = Path(
    "data_perp/reports/source_quality_label_walkforward_ablation_v1/"
    "strict_oos_repair_ranker_frozen_profile_run"
)
DEFAULT_REFERENCE_MONTHLY = DEFAULT_BROAD_REPAIR_OUTPUT_DIR / "strict_oos_repair_ranker_monthly.csv"
REFERENCE_TOLERANCE = 1e-9


def _parse_csv(value: str | None) -> list[str]:
    if value is None or not str(value).strip():
        return []
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")
    return slug[:120] or "profile"


def planned_months(
    *,
    profile_manifest: dict[str, Any],
    validation_periods: list[str] | None = None,
    history_periods: list[str] | None = None,
) -> list[str]:
    """Return ordered months to materialize.

    History periods are needed because the repair ranker trains each validation
    month from earlier strict-OOS oracle/proxy mistakes. The default uses the
    manifest's non-promotion periods as frozen history and appends validation
    periods.
    """
    history = history_periods if history_periods is not None else profile_manifest.get("non_promotion_periods", [])
    validation = validation_periods if validation_periods is not None else profile_manifest.get("validation_periods", [])
    seen: set[str] = set()
    out: list[str] = []
    for period in [*history, *validation]:
        period_s = str(period)
        if period_s and period_s not in seen:
            seen.add(period_s)
            out.append(period_s)
    return out


def _singleton_profile_manifest(profile_manifest: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    return {
        **profile_manifest,
        "profiles": [profile],
    }


def _combine_csv(paths: list[Path], out_path: Path) -> int:
    frames: list[pd.DataFrame] = []
    for path in paths:
        if not path.exists():
            continue
        try:
            frames.append(pd.read_csv(path))
        except pd.errors.EmptyDataError:
            continue
    if not frames:
        pd.DataFrame().to_csv(out_path, index=False)
        return 0
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(out_path, index=False)
    return int(len(combined))


def _equalish(left: Any, right: Any, *, tolerance: float) -> tuple[bool, float]:
    if pd.isna(left) and pd.isna(right):
        return True, 0.0
    left_num = pd.to_numeric(pd.Series([left]), errors="coerce").iloc[0]
    right_num = pd.to_numeric(pd.Series([right]), errors="coerce").iloc[0]
    if pd.notna(left_num) and pd.notna(right_num):
        diff = abs(float(left_num) - float(right_num))
        return diff <= tolerance, diff
    return str(left) == str(right), 0.0 if str(left) == str(right) else float("nan")


def build_reference_consistency_audit(
    frozen_monthly: pd.DataFrame,
    reference_monthly: pd.DataFrame,
    *,
    tolerance: float = REFERENCE_TOLERANCE,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compare singleton frozen-profile monthly rows to a broad-grid reference."""
    key_cols = ["period", *PROFILE_KEYS]
    missing_frozen = sorted(set(key_cols).difference(frozen_monthly.columns))
    missing_reference = sorted(set(key_cols).difference(reference_monthly.columns))
    if missing_frozen or missing_reference:
        raise ValueError(
            "Consistency audit missing key columns: "
            f"frozen={missing_frozen}, reference={missing_reference}"
        )

    common_cols = [
        col
        for col in frozen_monthly.columns
        if col in reference_monthly.columns and col not in key_cols
    ]
    rows: list[dict[str, Any]] = []
    reference_dupes = reference_monthly.duplicated(key_cols, keep=False)
    reference_index = {
        key: group
        for key, group in reference_monthly.groupby(key_cols, dropna=False, observed=True)
    }
    for _, frozen_row in frozen_monthly.iterrows():
        key = tuple(frozen_row[col] for col in key_cols)
        ref_group = reference_index.get(key)
        base = {col: frozen_row[col] for col in key_cols}
        if ref_group is None or ref_group.empty:
            rows.append(
                {
                    **base,
                    "consistency_status": "missing_reference_row",
                    "mismatch_columns": "",
                    "max_abs_diff": float("nan"),
                    "compared_columns": int(len(common_cols)),
                }
            )
            continue
        if len(ref_group) > 1:
            rows.append(
                {
                    **base,
                    "consistency_status": "duplicate_reference_rows",
                    "mismatch_columns": "",
                    "max_abs_diff": float("nan"),
                    "compared_columns": int(len(common_cols)),
                }
            )
            continue
        ref_row = ref_group.iloc[0]
        mismatches: list[str] = []
        diffs: list[float] = []
        for col in common_cols:
            ok, diff = _equalish(frozen_row[col], ref_row[col], tolerance=tolerance)
            if not ok:
                mismatches.append(col)
            if pd.notna(diff):
                diffs.append(float(diff))
        rows.append(
            {
                **base,
                "consistency_status": "matches_reference" if not mismatches else "differs_from_reference",
                "mismatch_columns": ",".join(mismatches),
                "max_abs_diff": max(diffs) if diffs else float("nan"),
                "compared_columns": int(len(common_cols)),
            }
        )
    audit = pd.DataFrame(rows)
    status_counts = (
        audit["consistency_status"].value_counts().to_dict()
        if not audit.empty and "consistency_status" in audit.columns
        else {}
    )
    summary = {
        "rows_checked": int(len(audit)),
        "reference_duplicate_key_rows": int(reference_dupes.sum()),
        "tolerance": float(tolerance),
        "status_counts": status_counts,
        "passes": bool(len(audit) > 0 and set(status_counts).issubset({"matches_reference"})),
    }
    return audit, summary


def run_frozen_profile_workflow(
    *,
    profile_manifest_path: Path,
    quality_labels_path: Path,
    labels_path: Path,
    predictions_path: Path,
    event_rows_path: Path,
    feature_dir: Path,
    feature_list_csv: Path,
    output_dir: Path,
    validation_output_dir: Path,
    validation_periods: list[str] | None,
    history_periods: list[str] | None,
    reference_monthly_path: Path | None,
    max_features: int,
    min_train_class_rows: int,
    min_valid_scope_rows: int,
    seed: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    profile_manifest = load_profile_manifest(profile_manifest_path)
    months = planned_months(
        profile_manifest=profile_manifest,
        validation_periods=validation_periods,
        history_periods=history_periods,
    )
    if not months:
        raise ValueError("No months to run; provide validation periods in manifest or CLI")

    profile_run_manifests: list[dict[str, Any]] = []
    monthly_paths: list[Path] = []
    aggregate_paths: list[Path] = []
    diagnostics_paths: list[Path] = []
    selected_rows_paths: list[Path] = []

    for idx, profile in enumerate(profile_manifest["profiles"], start=1):
        profile_name = str(profile.get("name") or f"profile_{idx}")
        profile_output = output_dir / f"{idx:02d}_{_slug(profile_name)}"
        manifest = run_repair_report(
            quality_labels_path=quality_labels_path,
            labels_path=labels_path,
            predictions_path=predictions_path,
            event_rows_path=event_rows_path,
            feature_dir=feature_dir,
            feature_list_csv=feature_list_csv,
            output_dir=profile_output,
            months=months,
            source_buckets=[str(profile["source_bucket"])],
            proxy_cols=[str(profile["proxy_col"])],
            top_fracs=[float(profile["top_frac"])],
            feature_modes=[str(profile["feature_mode"])],
            selection_methods=[str(profile["selection_method"])],
            max_features=max_features,
            min_train_class_rows=min_train_class_rows,
            min_valid_scope_rows=min_valid_scope_rows,
            seed=seed,
        )
        profile_run_manifests.append(manifest)
        outputs = manifest.get("outputs", {})
        if "monthly" in outputs:
            monthly_paths.append(Path(outputs["monthly"]))
        if "aggregate" in outputs:
            aggregate_paths.append(Path(outputs["aggregate"]))
        if "diagnostics" in outputs:
            diagnostics_paths.append(Path(outputs["diagnostics"]))
        if "selected_rows" in outputs:
            selected_rows_paths.append(Path(outputs["selected_rows"]))

    combined_paths = {
        "monthly": output_dir / "strict_oos_repair_ranker_frozen_profile_monthly.csv",
        "aggregate": output_dir / "strict_oos_repair_ranker_frozen_profile_aggregate.csv",
        "diagnostics": output_dir / "strict_oos_repair_ranker_frozen_profile_diagnostics.csv",
        "selected_rows": output_dir / "strict_oos_repair_ranker_frozen_profile_selected_rows.csv",
        "reference_consistency": output_dir / "strict_oos_repair_ranker_frozen_profile_reference_consistency.csv",
        "profile_manifest_effective": output_dir / "strict_oos_repair_ranker_frozen_profiles_effective.json",
    }
    combined_rows = {
        "monthly": _combine_csv(monthly_paths, combined_paths["monthly"]),
        "aggregate": _combine_csv(aggregate_paths, combined_paths["aggregate"]),
        "diagnostics": _combine_csv(diagnostics_paths, combined_paths["diagnostics"]),
        "selected_rows": _combine_csv(selected_rows_paths, combined_paths["selected_rows"]),
    }

    effective_manifest = {
        **profile_manifest,
        "validation_periods": validation_periods or profile_manifest.get("validation_periods", []),
        "non_promotion_periods": history_periods
        if history_periods is not None
        else profile_manifest.get("non_promotion_periods", []),
        "runner_months": months,
    }
    combined_paths["profile_manifest_effective"].write_text(
        json.dumps(_json_safe(effective_manifest), indent=2),
        encoding="utf-8",
    )

    reference_consistency_summary: dict[str, Any] = {
        "enabled": reference_monthly_path is not None,
        "reference_monthly_path": str(reference_monthly_path) if reference_monthly_path is not None else None,
        "status": "skipped",
    }
    if reference_monthly_path is not None:
        if reference_monthly_path.exists():
            try:
                frozen_monthly = pd.read_csv(combined_paths["monthly"])
            except pd.errors.EmptyDataError:
                frozen_monthly = pd.DataFrame()
            try:
                reference_monthly = pd.read_csv(reference_monthly_path)
            except pd.errors.EmptyDataError:
                reference_monthly = pd.DataFrame()
            if frozen_monthly.empty:
                pd.DataFrame().to_csv(combined_paths["reference_consistency"], index=False)
                reference_consistency_summary.update({"status": "no_frozen_rows", "passes": False})
            elif reference_monthly.empty:
                pd.DataFrame().to_csv(combined_paths["reference_consistency"], index=False)
                reference_consistency_summary.update({"status": "empty_reference", "passes": False})
            else:
                audit, summary = build_reference_consistency_audit(
                    frozen_monthly,
                    reference_monthly,
                    tolerance=REFERENCE_TOLERANCE,
                )
                audit.to_csv(combined_paths["reference_consistency"], index=False)
                reference_consistency_summary.update({"status": "checked", **summary})
        else:
            pd.DataFrame().to_csv(combined_paths["reference_consistency"], index=False)
            reference_consistency_summary.update({"status": "missing_reference", "passes": False})

    validation_manifest = run_validation(
        profile_manifest_path=combined_paths["profile_manifest_effective"],
        monthly_path=combined_paths["monthly"],
        output_dir=validation_output_dir,
        validation_periods=validation_periods,
    )

    workflow_manifest = {
        "scope": "strict_oos_repair_ranker_frozen_profile_run",
        "profile_manifest_path": str(profile_manifest_path),
        "quality_labels_path": str(quality_labels_path),
        "labels_path": str(labels_path),
        "predictions_path": str(predictions_path),
        "event_rows_path": str(event_rows_path),
        "feature_dir": str(feature_dir),
        "feature_list_csv": str(feature_list_csv),
        "output_dir": str(output_dir),
        "validation_output_dir": str(validation_output_dir),
        "months": months,
        "validation_periods": validation_periods or profile_manifest.get("validation_periods", []),
        "history_periods": history_periods
        if history_periods is not None
        else profile_manifest.get("non_promotion_periods", []),
        "profile_count": int(len(profile_manifest["profiles"])),
        "profile_keys": PROFILE_KEYS,
        "max_features": int(max_features),
        "min_train_class_rows": int(min_train_class_rows),
        "min_valid_scope_rows": int(min_valid_scope_rows),
        "seed": int(seed),
        "combined_rows": combined_rows,
        "profile_run_outputs": profile_run_manifests,
        "reference_consistency": reference_consistency_summary,
        "validation_manifest": validation_manifest,
        "outputs": {key: str(value) for key, value in combined_paths.items()},
    }
    workflow_manifest["outputs"]["manifest"] = str(output_dir / "manifest.json")
    Path(workflow_manifest["outputs"]["manifest"]).write_text(
        json.dumps(_json_safe(workflow_manifest), indent=2),
        encoding="utf-8",
    )
    return workflow_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-manifest", type=Path, default=DEFAULT_PROFILE_MANIFEST)
    parser.add_argument("--quality-labels-path", type=Path, default=DEFAULT_QUALITY_LABELS)
    parser.add_argument("--labels-path", type=Path, default=DEFAULT_LABELS_DIR)
    parser.add_argument("--predictions-path", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--event-rows-path", type=Path, default=DEFAULT_EVENT_ROWS)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--feature-list-csv", type=Path, default=DEFAULT_FEATURE_LIST_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--validation-output-dir", type=Path, default=DEFAULT_VALIDATION_OUTPUT_DIR)
    parser.add_argument("--reference-monthly-path", type=Path, default=DEFAULT_REFERENCE_MONTHLY)
    parser.add_argument(
        "--skip-reference-audit",
        action="store_true",
        help="Skip consistency comparison against the broad repair-ranker monthly ledger.",
    )
    parser.add_argument("--validation-periods", type=str, default=None)
    parser.add_argument("--history-periods", type=str, default=None)
    parser.add_argument("--max-features", type=int, default=96)
    parser.add_argument("--min-train-class-rows", type=int, default=10)
    parser.add_argument("--min-valid-scope-rows", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = run_frozen_profile_workflow(
        profile_manifest_path=args.profile_manifest,
        quality_labels_path=args.quality_labels_path,
        labels_path=args.labels_path,
        predictions_path=args.predictions_path,
        event_rows_path=args.event_rows_path,
        feature_dir=args.feature_dir,
        feature_list_csv=args.feature_list_csv,
        output_dir=args.output_dir,
        validation_output_dir=args.validation_output_dir,
        validation_periods=_parse_csv(args.validation_periods) or None,
        history_periods=_parse_csv(args.history_periods) or None,
        reference_monthly_path=None if args.skip_reference_audit else args.reference_monthly_path,
        max_features=args.max_features,
        min_train_class_rows=args.min_train_class_rows,
        min_valid_scope_rows=args.min_valid_scope_rows,
        seed=args.seed,
    )
    print(json.dumps(_json_safe(manifest), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
