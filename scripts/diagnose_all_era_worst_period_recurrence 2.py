#!/usr/bin/env python3
"""All-era, diagnostic-only recurrence analysis of worst strict-stack weeks.

Worst weeks are defined *within each evidence lineage*, never against a pooled
2022--2026 threshold.  The inferential unit is the complete UTC week.  The
multiview feature set is selected structurally before seeing performance,
keeping feature/covariance/interaction tests bounded and avoiding an outcome
selected feature screen.  Nothing emitted here is a trading gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_2022_2026_stack_regime_failures import bh_fdr, exact_label_permutation_pvalues  # noqa: E402
from scripts.materialize_2022_2026_stack_performance_calendar import _period_key, json_safe, sha256_file  # noqa: E402


SCHEMA = "all_era_worst_period_multiview_recurrence_v1"
DEFAULT_CALENDAR = ROOT / "data_perp/artifacts/stack_performance_calendar_2022_2026_20260730_v3"
DEFAULT_MULTIVIEW = ROOT / "data_perp/artifacts/regime_multiview_panel_2022_2026_20260730_v1/multiview_regime_features.parquet"
DEFAULT_STATE = ROOT / "data_perp/artifacts/regime_episode_ledger_2022_2026_20260730_v1/hourly_state_calendar.parquet"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/all_era_worst_period_multiview_recurrence_20260730_v1"
FORBIDDEN = ("target", "label", "outcome", "future", "post_entry", "realized", "mfe", "mae", "pnl", "net_ev", "gross_ev", "exit", "timeout")


def compact_feature_columns(columns: Iterable[str], *, maximum: int = 64) -> list[str]:
    """Select a deterministic causal compact set without performance labels."""
    eligible = [str(c) for c in columns if str(c).startswith("mv__") and not any(token in str(c).lower() for token in FORBIDDEN)]
    preferences = ("__robust_z_24h", "__delta_24h", "__realized_vol_24h", "__iqr_24h", "__vol_of_vol_24h")
    chosen: list[str] = []
    seen_roots: set[str] = set()
    for suffix in preferences:
        for name in sorted(c for c in eligible if c.endswith(suffix)):
            parts = name.split("__")
            root = "__".join(parts[1:3]) if len(parts) > 3 and parts[1] == "transition_new" else (parts[1] if len(parts) > 1 else name)
            if root not in seen_roots:
                seen_roots.add(root); chosen.append(name)
                if len(chosen) >= maximum:
                    return chosen
    for name in sorted(eligible):
        if name not in chosen:
            chosen.append(name)
            if len(chosen) >= maximum:
                break
    if len(chosen) < 8:
        raise ValueError("insufficient causal multiview fields for bounded recurrence analysis")
    return chosen


def identify_worst_weeks_by_era(performance: pd.DataFrame, *, quantile: float = .25) -> pd.DataFrame:
    weeks = performance.loc[(performance["period_type"] == "week") & performance["complete_for_percentage"].astype(bool)].copy()
    records = []
    for era, local in weeks.groupby("lineage_id", sort=True, observed=True):
        local = local.sort_values(["mean_net_bps", "period_start_utc"], kind="stable").copy()
        local["era_id"] = str(era)
        local["worst_week"] = False
        if len(local) >= 8:
            count = max(2, int(math.ceil(len(local) * quantile)))
            local.iloc[:count, local.columns.get_loc("worst_week")] = True
            local["worst_definition"] = f"bottom {count}/{len(local)} complete weeks within lineage by mean_net_bps"
            local["era_status"] = "eligible_pending_multiview_coverage"
        else:
            local["worst_definition"] = "insufficient complete weeks (<8)"
            local["era_status"] = "insufficient_performance_week_support"
        records.append(local)
    return pd.concat(records, ignore_index=True) if records else weeks.assign(era_id=pd.Series(dtype=str))


def _robust_z(values: pd.DataFrame, regular: np.ndarray) -> pd.DataFrame:
    ref = values.loc[regular]
    center = ref.median()
    scale = (ref - center).abs().median() * 1.4826
    scale = scale.where(scale.gt(1e-9), ref.std()).fillna(1.0)
    return (values - center) / scale


def _feature_family(name: str) -> str:
    if "dependence" in name:
        return "dependence_covariance"
    if "transition_new" in name or "mkt_regime_change" in name:
        return "transition_dynamics"
    if "liquidity" in name:
        return "liquidity"
    return "market_state_composite"


def _shifts(weekly: pd.DataFrame, labels: np.ndarray, *, era: str) -> pd.DataFrame:
    numeric = weekly.drop(columns=["dominant_regime"], errors="ignore").select_dtypes(include="number")
    numeric = numeric.loc[:, numeric.notna().all() & numeric.nunique().gt(1)]
    if numeric.empty:
        return pd.DataFrame()
    z = _robust_z(numeric, ~labels)
    matrix = z.to_numpy(float)
    p = exact_label_permutation_pvalues(matrix, labels)
    delta = matrix[labels].mean(0) - matrix[~labels].mean(0)
    output = pd.DataFrame({"era_id": era, "diagnostic_kind": "feature_shift", "feature": numeric.columns, "feature_family": [_feature_family(c) for c in numeric.columns], "worst_minus_regular_z": delta, "direction": np.sign(delta).astype(int), "permutation_p": p, "worst_weeks": int(labels.sum()), "regular_weeks": int((~labels).sum())})
    output["bh_q"] = bh_fdr(output["permutation_p"])
    output["era_significant"] = output["bh_q"].le(.10)
    return output


def _pair_tests(hourly: pd.DataFrame, weekly: pd.DataFrame, labels: np.ndarray, *, era: str, features: list[str]) -> pd.DataFrame:
    available = [f for f in features if f in hourly and hourly[f].notna().all() and hourly[f].nunique() > 1]
    if len(available) < 2:
        return pd.DataFrame()
    # A bounded, structurally selected subset.  Scale using regular-week hours.
    hour_labels = hourly["week_start_utc"].map(dict(zip(weekly.index, labels))).to_numpy(bool)
    z = _robust_z(hourly[available], ~hour_labels)
    pairs = [(left, right) for i, left in enumerate(available) for right in available[i + 1:]]
    cov, product = [], []
    for week in weekly.index:
        block = z.loc[hourly["week_start_utc"].eq(week), available]
        matrix = block.cov()
        cov.append([float(matrix.loc[a, b]) for a, b in pairs])
        product.append([float((block[a] * block[b]).mean()) for a, b in pairs])
    records = []
    for kind, matrix in (("covariance", np.asarray(cov)), ("standardized_interaction", np.asarray(product))):
        valid = np.isfinite(matrix).all(axis=0)
        p = np.full(len(pairs), np.nan)
        p[valid] = exact_label_permutation_pvalues(matrix[:, valid], labels)
        q = bh_fdr(p)
        for i, (left, right) in enumerate(pairs):
            if valid[i]:
                delta = float(matrix[labels, i].mean() - matrix[~labels, i].mean())
                records.append({"era_id": era, "diagnostic_kind": kind, "feature": f"{left} × {right}", "left_feature": left, "right_feature": right, "worst_minus_regular": delta, "direction": int(np.sign(delta)), "permutation_p": p[i], "bh_q": q[i], "era_significant": bool(q[i] <= .10), "worst_weeks": int(labels.sum()), "regular_weeks": int((~labels).sum())})
    return pd.DataFrame(records)


def _conditional_importance(weekly: pd.DataFrame, labels: np.ndarray, *, era: str, features: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if "dominant_regime" not in weekly:
        return pd.DataFrame()
    for state, local in weekly.groupby("dominant_regime", dropna=True, observed=True):
        mask = local.index.map(dict(zip(weekly.index, labels))).to_numpy(bool)
        values = local.loc[:, [f for f in features if f in local]].select_dtypes(include="number")
        valid = values.notna().all() & values.nunique().gt(1)
        values = values.loc[:, valid]
        if mask.sum() < 2 or (~mask).sum() < 2 or values.empty:
            rows.append({"era_id": era, "dominant_regime": str(state), "support_status": "insufficient_worst_or_regular_week_support", "weeks": int(len(local)), "worst_weeks": int(mask.sum()), "regular_weeks": int((~mask).sum())})
            continue
        z = _robust_z(values, ~mask).to_numpy(float)
        p = exact_label_permutation_pvalues(z, mask)
        q = bh_fdr(p)
        delta = z[mask].mean(0) - z[~mask].mean(0)
        for i, feature in enumerate(values.columns):
            rows.append({"era_id": era, "dominant_regime": str(state), "support_status": "tested", "feature": feature, "conditional_permutation_importance": abs(float(delta[i])), "direction": int(np.sign(delta[i])), "permutation_p": p[i], "bh_q": q[i], "era_significant": bool(q[i] <= .10), "weeks": int(len(local)), "worst_weeks": int(mask.sum()), "regular_weeks": int((~mask).sum())})
    return pd.DataFrame(rows)


def recurrence_summary(tables: Sequence[pd.DataFrame], *, expected_eras: Sequence[str]) -> pd.DataFrame:
    source = pd.concat([t for t in tables if not t.empty], ignore_index=True) if any(not t.empty for t in tables) else pd.DataFrame()
    if source.empty:
        return pd.DataFrame(columns=["diagnostic_kind", "feature", "evidence_status"])
    source = source.loc[source["era_significant"].notna()].copy()
    rows = []
    for (kind, feature), local in source.groupby(["diagnostic_kind", "feature"], observed=True, sort=True):
        tested = sorted(local["era_id"].unique())
        significant = local.loc[local["era_significant"].astype(bool)]
        pos = sorted(significant.loc[significant["direction"].gt(0), "era_id"].unique())
        neg = sorted(significant.loc[significant["direction"].lt(0), "era_id"].unique())
        direction, recurrent = ("positive", len(pos) >= 2) if len(pos) >= len(neg) else ("negative", len(neg) >= 2)
        status = "recurrent_across_separated_eras" if recurrent else ("no_recurrent_driver_found" if len(tested) >= 2 else "missing_cross_era_evidence")
        rows.append({"diagnostic_kind": kind, "feature": feature, "tested_eras": ",".join(tested), "significant_eras": ",".join(sorted(significant["era_id"].unique())), "same_direction_eras": ",".join(pos if direction == "positive" else neg), "recurrent_direction": direction, "recurrent": recurrent, "evidence_status": status, "uncovered_calendar_eras": ",".join(sorted(set(expected_eras) - set(tested)))})
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> Path:
    if args.output.exists():
        raise FileExistsError(f"immutable output exists: {args.output}")
    performance = pd.read_parquet(args.performance_calendar / "performance_period_metrics.parquet")
    calendar = identify_worst_weeks_by_era(performance, quantile=args.worst_quantile)
    schema = pq.ParquetFile(args.multiview).schema.names
    compact = compact_feature_columns(schema, maximum=args.compact_features)
    mv = pd.read_parquet(args.multiview, columns=["source_utc", "calendar_segment_id", *compact])
    mv["source_utc"] = pd.to_datetime(mv["source_utc"], utc=True)
    states = pd.read_parquet(args.state, columns=["source_utc", "state_context__current_state"])
    states["source_utc"] = pd.to_datetime(states["source_utc"], utc=True)
    hourly = mv.merge(states, on="source_utc", how="left", validate="one_to_one")
    hourly["week_start_utc"] = _period_key(hourly["source_utc"], "week")
    support = hourly.groupby("week_start_utc", observed=True).size().rename("multiview_hours")
    feature_tables: list[pd.DataFrame] = []; pair_tables: list[pd.DataFrame] = []; conditional_tables: list[pd.DataFrame] = []; coverage_rows: list[dict[str, Any]] = []
    interaction_features = compact_feature_columns(compact, maximum=args.interaction_features)
    for era, era_weeks in calendar.groupby("era_id", sort=True, observed=True):
        era_weeks = era_weeks.copy(); era_weeks["week_start_utc"] = pd.to_datetime(era_weeks["period_start_utc"], utc=True)
        required = era_weeks["week_start_utc"].tolist()
        observed = support.reindex(required).fillna(0).astype(int)
        complete = observed.ge(168).to_numpy()
        base_ok = era_weeks["era_status"].iloc[0] == "eligible_pending_multiview_coverage"
        if not base_ok or not complete.all():
            coverage_rows.append({"era_id": era, "era_status": "missing_multiview_evidence" if base_ok else era_weeks["era_status"].iloc[0], "calendar_complete_weeks": int(len(era_weeks)), "multiview_complete_weeks": int(complete.sum()), "missing_week_starts": ",".join(str(x) for x in np.asarray(required)[~complete])})
            continue
        local_hourly = hourly.loc[hourly["week_start_utc"].isin(required)].copy()
        weekly = local_hourly.groupby("week_start_utc", observed=True)[compact].mean().reindex(required)
        weekly["dominant_regime"] = local_hourly.groupby("week_start_utc", observed=True)["state_context__current_state"].agg(lambda x: x.mode().iloc[0] if x.notna().any() else np.nan).reindex(required)
        labels = era_weeks.set_index("week_start_utc").loc[required, "worst_week"].to_numpy(bool)
        coverage_rows.append({"era_id": era, "era_status": "tested", "calendar_complete_weeks": int(len(era_weeks)), "multiview_complete_weeks": int(complete.sum()), "missing_week_starts": ""})
        feature_tables.append(_shifts(weekly, labels, era=era))
        pair_tables.append(_pair_tests(local_hourly, weekly.drop(columns="dominant_regime"), labels, era=era, features=interaction_features))
        conditional_tables.append(_conditional_importance(weekly, labels, era=era, features=compact))
    features = pd.concat([t for t in feature_tables if not t.empty], ignore_index=True) if any(not t.empty for t in feature_tables) else pd.DataFrame()
    pairs = pd.concat([t for t in pair_tables if not t.empty], ignore_index=True) if any(not t.empty for t in pair_tables) else pd.DataFrame()
    conditional = pd.concat([t for t in conditional_tables if not t.empty], ignore_index=True) if any(not t.empty for t in conditional_tables) else pd.DataFrame()
    coverage = pd.DataFrame(coverage_rows)
    recurrence = recurrence_summary([features, pairs], expected_eras=calendar["era_id"].unique())
    output = args.output; output.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(dir=output.parent, prefix=f".{output.name}."))
    try:
        frames = {"era_week_calendar.csv": calendar, "era_coverage.csv": coverage, "compact_feature_list.csv": pd.DataFrame({"feature": compact}), "weekly_feature_shifts.csv": features, "weekly_covariance_and_interactions.csv": pairs, "regime_conditional_permutation_importance.csv": conditional, "recurrence_summary.csv": recurrence}
        hashes = {}
        for name, frame in frames.items():
            path = temp / name; frame.to_csv(path, index=False); hashes[name] = sha256_file(path)
        tested_eras = coverage.loc[coverage["era_status"].eq("tested"), "era_id"].tolist()
        manifest: dict[str, Any] = {"schema": SCHEMA, "status": "DIAGNOSTIC_ONLY_NO_GATE_PROMOTION", "performance_calendar": str(args.performance_calendar.resolve()), "multiview": str(args.multiview.resolve()), "state_timeline": str(args.state.resolve()), "inference_unit": "complete UTC week", "worst_contract": "bottom quantile independently within each evidence lineage", "compact_feature_contract": "structural causal selection before performance labels", "bh_contract": "BH q<=0.10 within era and diagnostic family", "recurrence_contract": "same signed significant effect in >=2 separated tested eras", "tested_eras": tested_eras, "missing_evidence_eras": coverage.loc[coverage["era_status"].ne("tested"), "era_id"].tolist(), "promotion_eligible": False, "outputs_sha256": hashes, "runner_sha256": sha256_file(Path(__file__).resolve())}
        manifest_path = temp / "manifest.json"; manifest_path.write_text(json.dumps(json_safe(manifest), indent=2, sort_keys=True) + "\n")
        (temp / "manifest.sha256").write_text(f"{sha256_file(manifest_path)}  manifest.json\n")
        os.replace(temp, output); return output
    except Exception:
        shutil.rmtree(temp, ignore_errors=True); raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--performance-calendar", type=Path, default=DEFAULT_CALENDAR)
    parser.add_argument("--multiview", type=Path, default=DEFAULT_MULTIVIEW)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--worst-quantile", type=float, default=.25)
    parser.add_argument("--compact-features", type=int, default=64)
    parser.add_argument("--interaction-features", type=int, default=12)
    args = parser.parse_args(argv)
    if not .10 <= args.worst_quantile <= .50: parser.error("worst quantile must be in [.10, .50]")
    return args


if __name__ == "__main__":
    run(parse_args())
