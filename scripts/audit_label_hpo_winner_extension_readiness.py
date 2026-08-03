#!/usr/bin/env python3
"""Audit how far the frozen label-HPO winner can be extended safely.

This is deliberately read-only.  It distinguishes the source-data ceiling from
the already scored/policy-replayed ceiling, verifies the side-local model-head
bundle, and records the one unavoidable recovery step: the original ablation
did not serialize the base boosters.
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_base_residual_label_ablation import (  # noqa: E402
    DEFAULT_FEATURE_STORE,
    DEFAULT_LABELS,
    DEFAULT_PATH_LABELS,
)

DEFAULT_FROZEN = ROOT / "data_perp/artifacts/base_residual_label_ablation_20260725_v2"
DEFAULT_EXTENSION = ROOT / "data_perp/artifacts/label_hpo_policy_replay_20260725_v1"
DEFAULT_OHLCV = ROOT / "data_perp/exchanges/krakenfutures/raw/ohlcv_15m"
SIDES = ("long", "short")
BASE_RESOLUTION_HOURS = 25


def _safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _utc(value: object | None) -> pd.Timestamp | None:
    if value is None or pd.isna(value):
        return None
    timestamp = pd.Timestamp(value)
    return (
        timestamp.tz_localize("UTC")
        if timestamp.tzinfo is None
        else timestamp.tz_convert("UTC")
    )


def _json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _query_one(connection: Any, query: str, params: list[Any]) -> tuple[Any, ...]:
    return tuple(connection.execute(query, params).fetchone())


def scan_sources(
    *, labels: Path, path_labels: Path, feature_store: Path, ohlcv_15m: Path
) -> dict[str, Any]:
    """Read timestamp/row bounds only; no feature matrix or model is loaded."""

    import duckdb

    connection = duckdb.connect(database=":memory:")
    sides: dict[str, Any] = {}
    try:
        for side in SIDES:
            base = labels / f"train_global_{side}_5_2026_07.parquet"
            path = path_labels / f"train_global_{side}_3.parquet"
            if not base.is_file() or not path.is_file():
                sides[side] = {
                    "available": False,
                    "missing": [str(item) for item in (base, path) if not item.is_file()],
                }
                continue
            base_rows, base_min, base_max = _query_one(
                connection,
                "SELECT count(*), min(__ts__), max(__ts__) FROM read_parquet(?)",
                [str(base)],
            )
            joined_rows, valid_rows, paired_max, paired_resolution_max = _query_one(
                connection,
                """
                SELECT
                    count(*),
                    sum(CASE WHEN a.__path_auxiliary_target_valid__ = 1 THEN 1 ELSE 0 END),
                    max(CASE WHEN a.__path_auxiliary_target_valid__ = 1 THEN b.__ts__ END),
                    max(CASE WHEN a.__path_auxiliary_target_valid__ = 1 THEN a.__label_end_ts__ END)
                FROM read_parquet(?) AS b
                INNER JOIN read_parquet(?) AS a USING (candidate_id)
                """,
                [str(base), str(path)],
            )
            sides[side] = {
                "available": True,
                "base_label_rows": int(base_rows),
                "base_signal_min": _utc(base_min),
                "base_signal_max": _utc(base_max),
                "base_label_resolution_max": (
                    _utc(base_max) + pd.Timedelta(hours=BASE_RESOLUTION_HOURS)
                    if _utc(base_max) is not None
                    else None
                ),
                "joined_rows": int(joined_rows),
                "valid_paired_path_rows": int(valid_rows or 0),
                "paired_signal_max": _utc(paired_max),
                "paired_path_resolution_max": _utc(paired_resolution_max),
                "base_label_path": base,
                "path_target_path": path,
            }

        feature_glob = str(feature_store / "symbol=*.parquet")
        try:
            _, feature_max = _query_one(
                connection,
                """
                SELECT count(*), max(ts)
                FROM read_parquet(?, union_by_name=true)
                WHERE ts IS NOT NULL
                """,
                [feature_glob],
            )
            feature_max = _utc(feature_max)
        except Exception as exc:  # schema evolution must be reported, not hidden
            feature_max = None
            feature_error = str(exc)
        else:
            feature_error = None

        raw_glob = str(ohlcv_15m / "*.parquet")
        try:
            _, raw_max = _query_one(
                connection,
                "SELECT count(*), max(__index_level_0__) FROM read_parquet(?)",
                [raw_glob],
            )
            raw_max = _utc(raw_max)
        except Exception as exc:
            raw_max = None
            raw_error = str(exc)
        else:
            raw_error = None
    finally:
        connection.close()
    return {
        "sides": sides,
        "feature_store_max": feature_max,
        "feature_store_error": feature_error,
        "raw_15m_ohlcv_max": raw_max,
        "raw_15m_ohlcv_error": raw_error,
    }


def scan_artifacts(*, frozen: Path, extension: Path) -> dict[str, Any]:
    summary = _json(frozen / "summary.json")
    extension_manifest = _json(extension / "manifest.json")
    policy_manifest = _json(extension / "simple_policy_optimizer/manifest.json")
    portfolio_summary = _json(extension / "portfolio_replay/summary.json")
    sides: dict[str, Any] = {}
    for side in SIDES:
        side_summary = dict(summary.get("sides", {}).get(side, {}))
        score_path = extension / f"{side}_extended_scores.parquet"
        score_max = None
        score_rows = 0
        if score_path.is_file():
            score_times = pd.read_parquet(score_path, columns=["__ts__"])
            score_times["__ts__"] = pd.to_datetime(
                score_times["__ts__"], utc=True, errors="coerce"
            )
            score_rows = int(len(score_times))
            score_max = _utc(score_times["__ts__"].max())
        side_root = frozen / side
        heads = {
            "base_booster_serialized": (side_root / "base_model.txt").is_file(),
            "residual_booster_serialized": (side_root / "residual_model.txt").is_file(),
            "base_ev_map_serialized": (side_root / "base_ev_map.joblib").is_file(),
            "admission_calibrator_serialized": (
                side_root / "admission_calibrator.joblib"
            ).is_file(),
        }
        sides[side] = {
            "winner_recipe": side_summary.get("winner_recipe"),
            "selected_feature_count": len(side_summary.get("features", [])),
            "selected_features": side_summary.get("features", []),
            "heads": heads,
            "base_recovery_required": not heads["base_booster_serialized"],
            "extended_score_rows": score_rows,
            "extended_score_max": score_max,
        }
    return {
        "frozen_summary_present": bool(summary),
        "extension_manifest": extension_manifest,
        "policy_manifest": policy_manifest,
        "portfolio_summary": portfolio_summary,
        "sides": sides,
    }


def build_readiness(
    source: Mapping[str, Any],
    artifacts: Mapping[str, Any],
    *,
    requested_end_inclusive: object,
) -> dict[str, Any]:
    """Combine bounded scans into a machine-readable, conservative verdict."""

    requested = _utc(requested_end_inclusive)
    paired_ends = [
        _utc(item.get("paired_signal_max"))
        for item in source.get("sides", {}).values()
        if _utc(item.get("paired_signal_max")) is not None
    ]
    source_ceiling = min(paired_ends) if len(paired_ends) == len(SIDES) else None
    feature_ceiling = _utc(source.get("feature_store_max"))
    raw_ceiling = _utc(source.get("raw_15m_ohlcv_max"))
    scoreable_ceiling = min(
        [item for item in (source_ceiling, feature_ceiling) if item is not None],
        default=None,
    )
    exact_policy_ceiling = min(
        [item for item in (scoreable_ceiling, raw_ceiling) if item is not None],
        default=None,
    )
    blockers: list[str] = []
    if source_ceiling is None:
        blockers.append("paired_canonical_base_and_path_labels_missing")
    if feature_ceiling is None:
        blockers.append("feature_store_timestamp_scan_failed")
    if raw_ceiling is None:
        blockers.append("raw_15m_ohlcv_timestamp_scan_failed")
    if requested is not None and (
        exact_policy_ceiling is None or requested > exact_policy_ceiling
    ):
        blockers.append("requested_end_exceeds_locally_available_causal_inputs")

    side_readiness: dict[str, Any] = {}
    for side in SIDES:
        artifact = dict(artifacts.get("sides", {}).get(side, {}))
        heads = dict(artifact.get("heads", {}))
        missing_heads = [name for name, present in heads.items() if not present]
        if artifact.get("base_recovery_required"):
            missing_heads = [
                name for name in missing_heads if name != "base_booster_serialized"
            ]
        side_readiness[side] = {
            **artifact,
            "missing_nonrecoverable_heads": missing_heads,
            "frozen_scoring_without_refit_available": not bool(
                artifact.get("base_recovery_required")
            )
            and not missing_heads,
            "deterministic_recovery_allowed": bool(
                artifact.get("base_recovery_required")
            )
            and all(
                heads.get(name, False)
                for name in (
                    "residual_booster_serialized",
                    "base_ev_map_serialized",
                    "admission_calibrator_serialized",
                )
            ),
        }

    recovered_score_ends = [
        _utc(item.get("extended_score_max"))
        for item in side_readiness.values()
        if _utc(item.get("extended_score_max")) is not None
    ]
    existing_score_ceiling = (
        min(recovered_score_ends) if len(recovered_score_ends) == len(SIDES) else None
    )
    gaps: list[dict[str, str]] = []
    if scoreable_ceiling is not None and existing_score_ceiling is not None:
        if existing_score_ceiling < scoreable_ceiling:
            gaps.append(
                {
                    "stage": "post_recovery_feature_matrix_and_score_export",
                    "available_through": existing_score_ceiling.isoformat(),
                    "source_ceiling": scoreable_ceiling.isoformat(),
                    "action": (
                        "Run the deterministic recovery extension into a new output "
                        "directory; it must reproduce Apr-Jun scores before accepting "
                        "new July rows."
                    ),
                }
            )
    if artifacts.get("policy_manifest"):
        holdout = dict(artifacts["policy_manifest"].get("holdout", {}))
        policy_max = _utc(holdout.get("holdout_max_ts"))
        if policy_max is not None and (
            existing_score_ceiling is None or policy_max < existing_score_ceiling
        ):
            gaps.append(
                {
                    "stage": "simple_policy_path_materialisation",
                    "available_through": policy_max.isoformat(),
                    "source_ceiling": (
                        existing_score_ceiling.isoformat()
                        if existing_score_ceiling is not None
                        else "unknown"
                    ),
                    "action": "Regenerate the policy handoff/path bundles after scoring is extended.",
                }
            )

    recovery_output = ROOT / "data_perp/artifacts/label_hpo_policy_replay_20260725_extension_next"
    commands = [
        f"python3 {shlex.quote(str(Path(__file__).relative_to(ROOT)))} --requested-end-inclusive {requested.isoformat() if requested else '2026-07-23T23:59:59Z'}",
        (
            "python3 scripts/extend_label_hpo_winner_for_policy_replay.py "
            f"--output {shlex.quote(str(recovery_output))}"
        ),
    ]
    return {
        "schema": "label_hpo_winner_extension_readiness_v1",
        "requested_end_inclusive": requested,
        "source_data_ceiling_signal_timestamp": source_ceiling,
        "feature_store_ceiling_timestamp": feature_ceiling,
        "raw_15m_ohlcv_ceiling_timestamp": raw_ceiling,
        "maximum_scoreable_signal_timestamp": scoreable_ceiling,
        "maximum_exact_policy_signal_timestamp": exact_policy_ceiling,
        "existing_recovered_score_ceiling": existing_score_ceiling,
        "requested_end_is_currently_available": not blockers,
        "blockers": blockers,
        "sides": side_readiness,
        "missing_stages": gaps,
        "safe_next_commands": commands,
        "contract": (
            "No HPO, feature selection, residual fitting, EV-map fitting, or admission "
            "calibration may be repeated. The missing base booster may only be recovered "
            "with the frozen training rows/recipe/seed and an exact Apr-Jun parity gate."
        ),
    }


def _markdown(report: Mapping[str, Any]) -> str:
    def timestamp(key: str) -> str:
        value = report.get(key)
        return value.isoformat() if isinstance(value, pd.Timestamp) else "unavailable"

    lines = [
        "# Label-HPO winner July-extension readiness",
        "",
        f"- Requested end: `{timestamp('requested_end_inclusive')}`",
        f"- Maximum scoreable signal: `{timestamp('maximum_scoreable_signal_timestamp')}`",
        f"- Maximum exact-policy signal: `{timestamp('maximum_exact_policy_signal_timestamp')}`",
        f"- Existing recovered-score end: `{timestamp('existing_recovered_score_ceiling')}`",
        f"- Requested end currently available: `{report.get('requested_end_is_currently_available')}`",
        "",
        "## Missing stages",
        "",
    ]
    gaps = list(report.get("missing_stages", []))
    if gaps:
        lines.extend(
            f"- `{item['stage']}`: {item['available_through']} -> {item['source_ceiling']}; {item['action']}"
            for item in gaps
        )
    else:
        lines.append("- None.")
    lines.extend(["", "## Safe commands", ""])
    lines.extend(f"- `{command}`" for command in report.get("safe_next_commands", []))
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> dict[str, Any]:
    requested = _utc(args.requested_end_inclusive)
    source = scan_sources(
        labels=args.labels,
        path_labels=args.path_labels,
        feature_store=args.feature_store,
        ohlcv_15m=args.ohlcv_15m,
    )
    artifacts = scan_artifacts(frozen=args.frozen, extension=args.extension)
    report = build_readiness(source, artifacts, requested_end_inclusive=requested)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "readiness.json").write_text(
        json.dumps(_safe(report), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output / "readiness.md").write_text(_markdown(report), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--path-labels", type=Path, default=DEFAULT_PATH_LABELS)
    parser.add_argument("--feature-store", type=Path, default=DEFAULT_FEATURE_STORE)
    parser.add_argument("--ohlcv-15m", type=Path, default=DEFAULT_OHLCV)
    parser.add_argument("--frozen", type=Path, default=DEFAULT_FROZEN)
    parser.add_argument("--extension", type=Path, default=DEFAULT_EXTENSION)
    parser.add_argument(
        "--requested-end-inclusive", default="2026-07-23T23:59:59+00:00"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data_perp/artifacts/label_hpo_winner_extension_readiness_20260725_v1",
    )
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps(_safe(result), indent=2, sort_keys=True))
