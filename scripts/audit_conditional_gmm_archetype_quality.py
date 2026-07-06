#!/usr/bin/env python3
"""Audit whether conditional-GMM archetypes are economically and feature-wise distinct.

This is an in-sample diagnostic for the archetype layer. It does not make an
OOS trading claim; it checks whether recent archetype artifacts satisfy the
intended contract:

- clusters group different outcome/path states,
- clusters are separated by feature signatures,
- selected GMM config is stable enough over time,
- selected features can be more informative about archetype membership than
  about the global economic target.
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

from extreme_price_movements.features_gmm_ae import fit_ae_gmm_state, transform_ae_gmm_features  # noqa: E402
from scripts.report_conditional_gmm_archetypes import (  # noqa: E402
    _gmm_economic_targets,
    _load_selected_feature_frame,
    _read_selected_features,
    _sample_frame,
)
from scripts.run_conditional_gmm_feature_selection import build_side_aware_targets  # noqa: E402


DEFAULT_SELECTION_DIR = Path(
    "data_perp/reports/conditional_gmm_feature_selection_20260702_lowcost_strict_econ_target_wide_sidebalanced_hpo"
)
DEFAULT_LABELS_PATH = Path(
    "data_perp/artifacts/"
    "20260702_211500_single_head_monthly_walkforward_bidirectional_sideaware_"
    "lowcost_strict_economic_target_labels/labels"
)
DEFAULT_FEATURE_DIR = Path("data_perp/features/20260629_050000")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        out = float(value)
        return out if math.isfinite(out) else None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def _spearman(a: Any, b: Any) -> float:
    x = pd.to_numeric(pd.Series(a), errors="coerce")
    y = pd.to_numeric(pd.Series(b), errors="coerce")
    mask = x.notna() & y.notna()
    if int(mask.sum()) < 20:
        return float("nan")
    if float(x.loc[mask].std(ddof=0)) <= 0.0 or float(y.loc[mask].std(ddof=0)) <= 0.0:
        return float("nan")
    return _safe_float(x.loc[mask].corr(y.loc[mask], method="spearman"))


def _timestamp_column(frame: pd.DataFrame) -> str | None:
    for col in ("timestamp", "__ts__", "ts", "entry_ts"):
        if col in frame.columns:
            return col
    return None


def _feature_family(feature: str) -> str:
    lower = feature.lower()
    if any(token in lower for token in ("oi", "funding", "open_interest")):
        return "open_interest"
    if any(token in lower for token in ("vol", "atr", "range", "shock", "compression")):
        return "volatility"
    if any(token in lower for token in ("adx", "ema", "trend", "breakout", "slope")):
        return "trend"
    if any(token in lower for token in ("spread", "liquid", "depth", "amihud")):
        return "liquidity"
    if any(token in lower for token in ("btc", "xasset", "peer", "dispersion", "assets")):
        return "cross_asset"
    if any(token in lower for token in ("entropy", "eig", "spectral")):
        return "entropy_state"
    if any(token in lower for token in ("dist", "loc", "pullback", "zscore", "vwap")):
        return "location_reversion"
    return "other"


def _cluster_summary(
    work: pd.DataFrame,
    x: pd.DataFrame,
    clusters: np.ndarray,
) -> pd.DataFrame:
    targets, _ = build_side_aware_targets(work)
    utility = pd.to_numeric(work["__u_econ_net__"], errors="coerce")
    side_col = "side" if "side" in work.columns else "__side__" if "__side__" in work.columns else ""
    side = pd.to_numeric(work[side_col], errors="coerce").fillna(1.0) if side_col else pd.Series(1.0, index=work.index)
    xz = (x - x.mean()) / x.std(ddof=0).replace(0.0, np.nan)
    rows: list[dict[str, Any]] = []
    for cluster in sorted(set(int(v) for v in clusters)):
        mask = clusters == cluster
        zmeans = xz.loc[mask].mean().sort_values(key=lambda s: s.abs(), ascending=False).head(12)
        rows.append(
            {
                "cluster": int(cluster),
                "rows": int(mask.sum()),
                "share": float(mask.mean()),
                "u_econ_net_mean": float(utility.loc[mask].mean()),
                "u_econ_net_q10": float(utility.loc[mask].quantile(0.10)),
                "u_econ_hit": float((utility.loc[mask] > 0.0).mean()),
                "side_short_share": float((side.loc[mask] < 0.0).mean()),
                "bad_MAE_mean": float(targets.loc[mask, "bad_MAE"].mean()),
                "timeout_mean": float(targets.loc[mask, "timeout"].mean()),
                "adverse_excursion_mean": float(targets.loc[mask, "adverse_excursion"].mean()),
                "favorable_excursion_mean": float(targets.loc[mask, "favorable_excursion"].mean()),
                "lower_tail_utility_mean": float(targets.loc[mask, "lower_tail_utility"].mean()),
                "top_feature_z_deviations": "; ".join(
                    f"{feature}:{value:+.2f}" for feature, value in zmeans.items()
                ),
                "top_feature_families": ",".join(
                    sorted({_feature_family(str(feature)) for feature in zmeans.index})
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("u_econ_net_mean", ascending=False).reset_index(drop=True)


def _cluster_pairwise_feature_distance(cluster_summary: pd.DataFrame) -> pd.DataFrame:
    parsed: dict[int, dict[str, float]] = {}
    for _, row in cluster_summary.iterrows():
        vec: dict[str, float] = {}
        for part in str(row.get("top_feature_z_deviations", "")).split(";"):
            item = part.strip()
            if not item or ":" not in item:
                continue
            name, value = item.rsplit(":", 1)
            try:
                vec[str(name)] = float(value)
            except Exception:
                continue
        parsed[int(row["cluster"])] = vec
    rows: list[dict[str, Any]] = []
    clusters = sorted(parsed)
    for i, a in enumerate(clusters):
        for b in clusters[i + 1 :]:
            keys_a = set(parsed[a])
            keys_b = set(parsed[b])
            union = keys_a | keys_b
            shared = keys_a & keys_b
            all_keys = sorted(union)
            distance = math.sqrt(
                sum((parsed[a].get(key, 0.0) - parsed[b].get(key, 0.0)) ** 2 for key in all_keys)
            )
            rows.append(
                {
                    "cluster_a": int(a),
                    "cluster_b": int(b),
                    "top_feature_jaccard": float(len(shared) / len(union)) if union else 0.0,
                    "shared_top_features": ",".join(sorted(shared)),
                    "top_z_euclidean": float(distance),
                }
            )
    return pd.DataFrame(rows)


def _monthly_stability(
    work: pd.DataFrame,
    clusters: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ts_col = _timestamp_column(work)
    if ts_col is None:
        return pd.DataFrame(), pd.DataFrame()
    ts = pd.to_datetime(work[ts_col], utc=True, errors="coerce")
    period = ts.dt.to_period("M").astype(str)
    utility = pd.to_numeric(work["__u_econ_net__"], errors="coerce")
    local = pd.DataFrame({"period": period, "cluster": clusters.astype(int), "u": utility})
    local = local[local["period"].notna()].copy()
    total_by_period = local.groupby("period", sort=False).size()
    rows: list[dict[str, Any]] = []
    for (month, cluster), group in local.groupby(["period", "cluster"], sort=False):
        total = int(total_by_period.loc[month])
        rows.append(
            {
                "period": str(month),
                "cluster": int(cluster),
                "rows": int(len(group)),
                "share": float(len(group) / max(total, 1)),
                "mean_u": float(group["u"].mean()),
                "hit_u": float((group["u"] > 0.0).mean()),
            }
        )
    monthly = pd.DataFrame(rows)
    summary_rows: list[dict[str, Any]] = []
    for cluster, group in monthly.groupby("cluster", sort=False):
        share = pd.to_numeric(group["share"], errors="coerce")
        mean_u = pd.to_numeric(group["mean_u"], errors="coerce")
        summary_rows.append(
            {
                "cluster": int(cluster),
                "months": int(group["period"].nunique()),
                "min_month_share": float(share.min()),
                "max_month_share": float(share.max()),
                "share_cv": float(share.std(ddof=0) / max(float(share.mean()), 1e-12)),
                "positive_months": int((mean_u > 0.0).sum()),
                "negative_months": int((mean_u < 0.0).sum()),
                "mean_u_min": float(mean_u.min()),
                "mean_u_max": float(mean_u.max()),
            }
        )
    return monthly, pd.DataFrame(summary_rows)


def _feature_archetype_ic(
    work: pd.DataFrame,
    x: pd.DataFrame,
    clusters: np.ndarray,
) -> pd.DataFrame:
    utility = pd.to_numeric(work["__u_econ_net__"], errors="coerce")
    y_soft = pd.to_numeric(work["__y_econ_soft__"], errors="coerce")
    rows: list[dict[str, Any]] = []
    for feature in x.columns:
        values = pd.to_numeric(x[feature], errors="coerce")
        best_abs = float("nan")
        best_cluster = None
        best_signed = float("nan")
        for cluster in sorted(set(int(v) for v in clusters)):
            membership = (clusters == cluster).astype(np.float32)
            ic = _spearman(values, membership)
            if math.isfinite(ic) and (not math.isfinite(best_abs) or abs(ic) > best_abs):
                best_abs = abs(ic)
                best_signed = ic
                best_cluster = int(cluster)
        utility_ic = _spearman(values, utility)
        soft_ic = _spearman(values, y_soft)
        target_abs = max(
            abs(utility_ic) if math.isfinite(utility_ic) else 0.0,
            abs(soft_ic) if math.isfinite(soft_ic) else 0.0,
        )
        rows.append(
            {
                "feature": str(feature),
                "family": _feature_family(str(feature)),
                "best_cluster": best_cluster,
                "best_cluster_membership_ic": best_signed,
                "best_cluster_membership_abs_ic": best_abs,
                "utility_abs_ic": abs(utility_ic) if math.isfinite(utility_ic) else float("nan"),
                "soft_target_abs_ic": abs(soft_ic) if math.isfinite(soft_ic) else float("nan"),
                "max_global_target_abs_ic": target_abs,
                "archetype_ic_minus_target_ic": (
                    float(best_abs - target_abs) if math.isfinite(best_abs) else float("nan")
                ),
                "archetype_ic_ratio_vs_target": (
                    float(best_abs / max(target_abs, 1e-6)) if math.isfinite(best_abs) else float("nan")
                ),
                "archetype_specific_low_target_ic": bool(
                    math.isfinite(best_abs) and best_abs >= 0.08 and target_abs <= 0.04
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["best_cluster_membership_abs_ic", "archetype_ic_minus_target_ic"],
        ascending=[False, False],
    )


def _outcome_separation(cluster_summary: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "u_econ_net_mean",
        "u_econ_hit",
        "u_econ_net_q10",
        "bad_MAE_mean",
        "timeout_mean",
        "adverse_excursion_mean",
        "favorable_excursion_mean",
        "lower_tail_utility_mean",
        "side_short_share",
    ]
    rows = []
    for metric in metrics:
        values = pd.to_numeric(cluster_summary[metric], errors="coerce")
        rows.append(
            {
                "metric": metric,
                "min": float(values.min()),
                "max": float(values.max()),
                "range": float(values.max() - values.min()),
                "std": float(values.std(ddof=0)),
            }
        )
    return pd.DataFrame(rows)


def run_audit(
    *,
    selection_dir: Path,
    labels_path: Path,
    feature_dir: Path,
    output_dir: Path,
    max_rows: int,
    max_train_rows: int,
    ae_max_iter: int,
    random_state: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_features = _read_selected_features(selection_dir)
    frame, available, feature_report = _load_selected_feature_frame(
        labels_path,
        feature_dir,
        selected_features,
    )
    work = _sample_frame(frame, max_rows=max_rows)
    x = (
        work.reindex(columns=available)
        .apply(pd.to_numeric, errors="coerce")
        .astype(np.float32)
    )
    side_col = "side" if "side" in work.columns else "__side__" if "__side__" in work.columns else ""
    economic_targets, _targets = _gmm_economic_targets(work)
    state = fit_ae_gmm_state(
        x,
        economic_targets=economic_targets,
        random_state=int(random_state),
        max_train_rows=int(max_train_rows),
        ae_max_iter=int(ae_max_iter),
        require_both_sides=bool(side_col),
    )
    transformed = transform_ae_gmm_features(x, state)
    clusters = (
        pd.to_numeric(transformed["gmm_cluster_id"], errors="coerce")
        .fillna(0)
        .astype(np.int32)
        .to_numpy()
    )
    cluster_summary = _cluster_summary(work, x, clusters)
    pairwise = _cluster_pairwise_feature_distance(cluster_summary)
    outcome = _outcome_separation(cluster_summary)
    monthly, monthly_summary = _monthly_stability(work, clusters)
    feature_ic = _feature_archetype_ic(work, x, clusters)

    selected_config = dict(state.get("selected_config", {}) or {})
    feature_specific = feature_ic["archetype_specific_low_target_ic"].astype(bool)
    pairwise_distance = pd.to_numeric(pairwise.get("top_z_euclidean"), errors="coerce")
    pairwise_jaccard = pd.to_numeric(pairwise.get("top_feature_jaccard"), errors="coerce")
    outcome_range = outcome.set_index("metric")["range"].to_dict()
    checks = {
        "state_enabled": bool(state.get("enabled", False)),
        "contract_columns_finite": bool(
            transformed.size and np.isfinite(transformed.to_numpy(dtype=np.float32)).all()
        ),
        "occupancy_ok": bool(selected_config.get("occupancy_ok", False)),
        "side_coverage_ok": bool(selected_config.get("side_coverage_ok", False)),
        "temporal_stability_ok": bool(
            _safe_float(selected_config.get("temporal_stability_score")) >= 0.80
            and _safe_float(selected_config.get("switch_rate")) <= 0.10
        ),
        "feature_signature_separation_ok": bool(
            pairwise_distance.notna().any()
            and float(pairwise_distance.median()) >= 2.0
            and float(pairwise_jaccard.mean()) <= 0.35
        ),
        "outcome_utility_separation_ok": bool(
            abs(float(outcome_range.get("u_econ_net_mean", 0.0))) >= 0.001
        ),
        "outcome_risk_separation_weak": bool(
            float(outcome_range.get("bad_MAE_mean", 0.0)) < 0.08
        ),
        "feature_to_archetype_ic_ok": bool(
            int(feature_specific.sum()) >= 10
            and float(feature_ic["best_cluster_membership_abs_ic"].head(20).median()) >= 0.10
        ),
    }
    checks["overall_read"] = (
        "pass_with_risk_outcome_caveat"
        if checks["state_enabled"]
        and checks["temporal_stability_ok"]
        and checks["feature_signature_separation_ok"]
        and checks["feature_to_archetype_ic_ok"]
        else "fail_or_incomplete"
    )

    paths = {
        "cluster_summary": output_dir / "archetype_quality_cluster_summary.csv",
        "pairwise_feature_distance": output_dir / "archetype_quality_pairwise_feature_distance.csv",
        "outcome_separation": output_dir / "archetype_quality_outcome_separation.csv",
        "monthly": output_dir / "archetype_quality_monthly.csv",
        "monthly_summary": output_dir / "archetype_quality_monthly_summary.csv",
        "feature_ic": output_dir / "archetype_quality_feature_ic.csv",
        "manifest": output_dir / "archetype_quality_manifest.json",
        "markdown": output_dir / "archetype_quality_report.md",
    }
    cluster_summary.to_csv(paths["cluster_summary"], index=False)
    pairwise.to_csv(paths["pairwise_feature_distance"], index=False)
    outcome.to_csv(paths["outcome_separation"], index=False)
    monthly.to_csv(paths["monthly"], index=False)
    monthly_summary.to_csv(paths["monthly_summary"], index=False)
    feature_ic.to_csv(paths["feature_ic"], index=False)

    manifest = {
        "selection_dir": str(selection_dir),
        "labels_path": str(labels_path),
        "feature_dir": str(feature_dir),
        "output_dir": str(output_dir),
        "sample_rows": int(len(work)),
        "selected_feature_count": int(len(selected_features)),
        "available_feature_count": int(len(available)),
        "feature_store": feature_report,
        "state_enabled": bool(state.get("enabled", False)),
        "selected_config": selected_config,
        "hpo_report_count": int(state.get("hpo_report_count", 0) or 0),
        "checks": checks,
        "counts": {
            "clusters": int(cluster_summary["cluster"].nunique()),
            "feature_ic_rows": int(len(feature_ic)),
            "archetype_specific_low_target_ic_features": int(feature_specific.sum()),
            "unique_top_feature_count": int(
                len(
                    {
                        item.rsplit(":", 1)[0].strip()
                        for text in cluster_summary["top_feature_z_deviations"].astype(str)
                        for item in text.split(";")
                        if ":" in item
                    }
                )
            ),
        },
        "outputs": {key: str(path) for key, path in paths.items()},
    }
    paths["manifest"].write_text(
        json.dumps(_json_safe(manifest), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_markdown(paths["markdown"], manifest, cluster_summary, outcome, pairwise, monthly_summary, feature_ic)
    return manifest


def _write_markdown(
    path: Path,
    manifest: dict[str, Any],
    cluster_summary: pd.DataFrame,
    outcome: pd.DataFrame,
    pairwise: pd.DataFrame,
    monthly_summary: pd.DataFrame,
    feature_ic: pd.DataFrame,
) -> None:
    checks = manifest["checks"]
    lines = [
        "# Conditional GMM Archetype Quality Audit",
        "",
        f"Selection dir: `{manifest['selection_dir']}`",
        f"Rows sampled: `{manifest['sample_rows']}`",
        f"Selected features: `{manifest['selected_feature_count']}`",
        f"Overall read: `{checks['overall_read']}`",
        "",
        "## Checks",
        "",
        pd.DataFrame(
            [{"check": key, "value": value} for key, value in checks.items()]
        ).to_markdown(index=False),
        "",
        "## Cluster Summary",
        "",
        cluster_summary.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Outcome Separation",
        "",
        outcome.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Pairwise Feature Signature Distance",
        "",
        pairwise.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Monthly Stability",
        "",
        monthly_summary.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Top Feature-To-Archetype IC",
        "",
        feature_ic.head(40).to_markdown(index=False, floatfmt=".6f"),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-dir", type=Path, default=DEFAULT_SELECTION_DIR)
    parser.add_argument("--labels-path", type=Path, default=DEFAULT_LABELS_PATH)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_SELECTION_DIR / "archetype_quality_audit",
    )
    parser.add_argument("--max-rows", type=int, default=10000)
    parser.add_argument("--max-train-rows", type=int, default=5000)
    parser.add_argument("--ae-max-iter", type=int, default=8)
    parser.add_argument("--random-state", type=int, default=913)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = run_audit(
        selection_dir=args.selection_dir,
        labels_path=args.labels_path,
        feature_dir=args.feature_dir,
        output_dir=args.output_dir,
        max_rows=int(args.max_rows),
        max_train_rows=int(args.max_train_rows),
        ae_max_iter=int(args.ae_max_iter),
        random_state=int(args.random_state),
    )
    print(json.dumps(_json_safe(manifest), indent=2, sort_keys=True))
    return 0 if manifest["checks"]["overall_read"] != "fail_or_incomplete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
