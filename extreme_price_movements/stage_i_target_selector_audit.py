"""Reproducible Stage-I target-information versus selector-sparsity audit.

This is a development diagnostic.  It uses resolved selector outcomes to ask
whether authorised entry-time features contain stable univariate information
about the base target, and whether that information has the same relationship
to exact policy economics.  It never produces inference features or a model
winner.
"""

from __future__ import annotations

from hashlib import sha256
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .stage_i_timestamp_contract import resolve_stage_i_timestamp_contract


SCHEMA = "stage_i_target_selector_information_audit_v1"


class StageITargetSelectorAuditError(ValueError):
    """Raised when the selector audit would compare misaligned populations."""


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_spearman(left: pd.Series, right: pd.Series) -> float:
    valid = left.notna() & right.notna()
    if int(valid.sum()) < 3 or left.loc[valid].nunique() < 2 or right.loc[valid].nunique() < 2:
        return float("nan")
    value = left.loc[valid].corr(right.loc[valid], method="spearman")
    return float(value) if value is not None and math.isfinite(float(value)) else float("nan")


def _broad_eras(timestamps: pd.Series) -> np.ndarray:
    ranks = pd.to_datetime(timestamps, utc=True, errors="raise").rank(method="first")
    return pd.qcut(ranks, 3, labels=False).to_numpy(dtype=np.int8)


def audit_stage_i_target_selector_information(
    ledger: pd.DataFrame,
    feature_frame: pd.DataFrame,
    *,
    side_feature_universes: Mapping[str, Sequence[str]],
) -> tuple[pd.DataFrame, pd.DataFrame, Mapping[str, Any]]:
    """Return feature stability, target composition, and compact side summary."""
    if len(ledger) != len(feature_frame) or not ledger.index.equals(feature_frame.index):
        raise StageITargetSelectorAuditError(
            "selector ledger and feature matrix must have identical row order"
        )
    required = {
        "side_name", "decision_ts", "r3_class", "r3_metric_target",
        "robust_clear_soft_b25_t50", "t2_tp6_sl4_event", "exact_net_bps",
    }
    missing = sorted(required.difference(ledger.columns))
    if missing:
        raise StageITargetSelectorAuditError(f"selector ledger lacks fields: {missing}")
    resolve_stage_i_timestamp_contract(ledger)
    feature_rows: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    for side in ("long", "short"):
        mask = ledger["side_name"].astype(str).str.lower().eq(side).to_numpy()
        local_ledger = ledger.loc[mask].reset_index(drop=True)
        local_features = feature_frame.loc[mask].reset_index(drop=True)
        if local_ledger.empty:
            raise StageITargetSelectorAuditError(f"selector contains no {side} rows")
        universe = tuple(dict.fromkeys(map(str, side_feature_universes.get(side, ()))))
        if not universe:
            raise StageITargetSelectorAuditError(f"authorised {side} feature universe is empty")
        absent = sorted(set(universe).difference(local_features.columns))
        if absent:
            raise StageITargetSelectorAuditError(
                f"authorised {side} features absent from selector matrix: {absent[:12]}"
            )
        target = pd.to_numeric(local_ledger["r3_metric_target"], errors="coerce")
        net = pd.to_numeric(local_ledger["exact_net_bps"], errors="coerce")
        if target.isna().any() or net.isna().any():
            raise StageITargetSelectorAuditError("selector target/economics must be finite")
        era = _broad_eras(local_ledger["decision_ts"])
        for feature in universe:
            values = pd.to_numeric(local_features[feature], errors="coerce")
            target_rho = _safe_spearman(values, target)
            net_rho = _safe_spearman(values, net)
            era_rho = [
                _safe_spearman(values.loc[era == era_id], target.loc[era == era_id])
                for era_id in range(3)
            ]
            finite_era = np.asarray([value for value in era_rho if math.isfinite(value)])
            target_sign = np.sign(target_rho) if math.isfinite(target_rho) else 0.0
            sign_consistent = bool(
                len(finite_era) == 3
                and target_sign != 0.0
                and np.all(np.sign(finite_era) == target_sign)
            )
            feature_rows.append({
                "side": side,
                "feature": feature,
                "coverage": float(values.notna().mean()),
                "unique_values": int(values.nunique(dropna=True)),
                "spearman_r3_metric_target": target_rho,
                "spearman_exact_net_bps": net_rho,
                "era_early_spearman_target": era_rho[0],
                "era_middle_spearman_target": era_rho[1],
                "era_late_spearman_target": era_rho[2],
                "worst_absolute_era_target_spearman": (
                    float(np.min(np.abs(finite_era))) if len(finite_era) == 3 else float("nan")
                ),
                "all_era_target_sign_consistent": sign_consistent,
            })
        for r3_class, group in local_ledger.groupby("r3_class", sort=True):
            group_net = pd.to_numeric(group["exact_net_bps"], errors="raise")
            target_rows.append({
                "side": side,
                "slice": "r3_class",
                "slice_value": str(int(r3_class)),
                "rows": int(len(group)),
                "prevalence": float(len(group) / len(local_ledger)),
                "exact_net_mean_bps": float(group_net.mean()),
                "exact_net_median_bps": float(group_net.median()),
                "positive_net_rate": float((group_net > 0.0).mean()),
            })
        for event, group in local_ledger.groupby("t2_tp6_sl4_event", sort=True):
            group_net = pd.to_numeric(group["exact_net_bps"], errors="raise")
            target_rows.append({
                "side": side,
                "slice": "first_touch_event",
                "slice_value": str(int(event)),
                "rows": int(len(group)),
                "prevalence": float(len(group) / len(local_ledger)),
                "exact_net_mean_bps": float(group_net.mean()),
                "exact_net_median_bps": float(group_net.median()),
                "positive_net_rate": float((group_net > 0.0).mean()),
            })
        soft = pd.to_numeric(
            local_ledger["robust_clear_soft_b25_t50"], errors="raise"
        )
        oracle: dict[str, Any] = {}
        tie = np.arange(len(local_ledger), dtype=np.int64)
        for fraction in (0.01, 0.05, 0.10, 0.20):
            count = max(1, int(math.ceil(fraction * len(local_ledger))))
            order = np.lexsort((tie, soft.to_numpy(dtype=float)))
            chosen = order[-count:]
            oracle[f"top_{int(fraction * 100):02d}"] = {
                "rows": int(count),
                "exact_net_mean_bps": float(net.iloc[chosen].mean()),
            }
        local_audit = pd.DataFrame(
            [row for row in feature_rows if row["side"] == side]
        )
        summaries[side] = {
            "rows": int(len(local_ledger)),
            "authorised_feature_count": int(len(universe)),
            "absolute_target_spearman_counts": {
                str(threshold): int(
                    local_audit["spearman_r3_metric_target"].abs().ge(threshold).sum()
                )
                for threshold in (0.01, 0.02, 0.03, 0.05)
            },
            "stable_all_era_sign_and_min_abs_0_01": int(
                (
                    local_audit["all_era_target_sign_consistent"]
                    & local_audit["worst_absolute_era_target_spearman"].ge(0.01)
                ).sum()
            ),
            "cross_feature_target_vs_net_spearman": _safe_spearman(
                local_audit["spearman_r3_metric_target"],
                local_audit["spearman_exact_net_bps"],
            ),
            "soft_target_unique_values": int(soft.nunique()),
            "soft_target_spearman_exact_net_bps": _safe_spearman(soft, net),
            "soft_target_oracle": oracle,
        }
    return pd.DataFrame(feature_rows), pd.DataFrame(target_rows), summaries


def publish_stage_i_target_selector_audit(
    selector_dir: str | Path,
    output_dir: str | Path,
    *,
    side_feature_universes: Mapping[str, Sequence[str]],
) -> Mapping[str, Any]:
    source = Path(selector_dir).resolve()
    destination = Path(output_dir).resolve()
    if destination.exists():
        raise FileExistsError(f"selector target audit already exists: {destination}")
    ledger_path = source / "selector_ledger.parquet"
    features_path = source / "selector_features.parquet"
    manifest_path = source / "manifest.json"
    if not ledger_path.is_file() or not features_path.is_file() or not manifest_path.is_file():
        raise StageITargetSelectorAuditError("completed selector payload is incomplete")
    source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if source_manifest.get("status") != "complete":
        raise StageITargetSelectorAuditError("selector source is not complete")
    ledger = pd.read_parquet(ledger_path).reset_index(drop=True)
    features = pd.read_parquet(features_path).reset_index(drop=True)
    feature_audit, target_audit, summaries = audit_stage_i_target_selector_information(
        ledger, features, side_feature_universes=side_feature_universes
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent))
    try:
        feature_path = temporary / "feature_univariate_stability.parquet"
        target_path = temporary / "target_outcome_composition.parquet"
        feature_audit.to_parquet(feature_path, index=False, compression="zstd")
        target_audit.to_parquet(target_path, index=False, compression="zstd")
        manifest = {
            "schema": SCHEMA,
            "status": "complete",
            "research_status": "development_diagnostic_not_oos",
            "selector_source": str(source),
            "selector_manifest_sha256": _file_sha256(manifest_path),
            "selector_ledger_sha256": _file_sha256(ledger_path),
            "selector_features_sha256": _file_sha256(features_path),
            "feature_univariate_stability_sha256": _file_sha256(feature_path),
            "target_outcome_composition_sha256": _file_sha256(target_path),
            "side_feature_universes": {
                side: list(map(str, values))
                for side, values in side_feature_universes.items()
            },
            "summary": summaries,
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    except BaseException:
        import shutil

        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return manifest


__all__ = [
    "SCHEMA",
    "StageITargetSelectorAuditError",
    "audit_stage_i_target_selector_information",
    "publish_stage_i_target_selector_audit",
]
