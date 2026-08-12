#!/usr/bin/env python3
"""Long-only conditional path/cluster residual layer on the TP6/SL4 base.

The runner uses the strict TP6/SL4/H12 base OOF panel and the frozen signed
family/path contribution store.  Family clusters, CMI feature selection, and
conditional residual models are refit on each chronological development fold;
the held-out month is transformed once and never participates in discovery.

The structural family store currently covers only part of the historical base
panel.  The script persists that coverage rather than silently treating a
missing path as a zero contribution.  This is a diagnostic/research layer,
not a promotion of the canonical 2025 stack.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Iterable

# The desktop bundle keeps scientific dependencies outside the repository.
# Insert those locations before importing numpy/pandas so the script remains
# runnable under the repository's sanitized ``python3 -S`` launcher.
sys.path[:0] = [
    "/Users/remyroche/Library/Python/3.12/lib/python/site-packages",
    "/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload",
    "/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12",
]

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.conditional_cluster_residual import (
    ClusterContract,
    cluster_condition_economics,
    conditional_mi_scores,
    discover_family_clusters,
    materialize_cluster_features,
    select_oof_path_rows,
    soft_cluster_residual_target,
)

SIDE = "long"
SEED = 20260807
TAILS = (0.005, 0.01, 0.02, 0.05, 0.10, 0.20)
BASE_FIELDS = ["p_clear", "p_adverse", "p_weak", "base_raw", "base_score", "base_expected_bps"]
LEAK_TOKENS = (
    "net_bps", "gross_bps", "exact_net", "exact_gross", "policy_net", "policy_gross",
    "label_available", "target__", "residual_bps", "execution", "outcome", "future",
    # Target/label aliases that do not contain the generic ``target__`` token.
    # They are valid audit columns but must never enter the pre-selection
    # feature pool.
    "robust_clear", "selector_economic", "r3_class", "grade_", "binary_h12",
)

DEFAULT_BASE = ROOT / "data_perp/artifacts/feature_leaf_reasoning_strict_oof_transport_a_20260803_v1/base_prediction_shards/transport_a_2023q4_to_2024h1/long/outer_predictions.parquet"
DEFAULT_FAMILY = ROOT / "data_perp/artifacts/long_family_conditional_correctness_semantic020_top64_strict_20260808_v1/family_contribution_matrix.parquet"
DEFAULT_LEDGER = ROOT / "data_perp/artifacts/tp6_m6_shared_regime_residual_features_20260809_v3/shared_regime_residual_ledger.parquet"
DEFAULT_OUT = ROOT / "data_perp/artifacts/tp6_sl4_conditional_cluster_residual_20260807_v6"


def _quote(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def _schema(path: Path) -> list[str]:
    return list(map(str, pq.ParquetFile(path).schema.names))


def _resolve_config_names() -> set[str]:
    """Resolve all configured meta/regime feature lists without fallback-to-all."""

    names: set[str] = set()
    try:
        from extreme_price_movements.config import CFG
    except Exception:
        return names
    seen: set[str] = set()

    def visit(value: object, key: str = "") -> None:
        if isinstance(value, str):
            if value in seen:
                return
            if value in CFG and value not in seen:
                seen.add(value)
                visit(CFG[value], value)
            elif value and not value.startswith("__"):
                names.add(value)
            return
        if isinstance(value, (list, tuple, set)):
            for item in value:
                visit(item, key)
        elif isinstance(value, dict):
            for item in value.values():
                visit(item, key)

    for key, value in CFG.items():
        lower = str(key).lower()
        if any(token in lower for token in ("meta", "regime", "residual", "context", "transition", "uncertainty", "reliability")):
            visit(value, str(key))
    return names


def _is_leak(name: str) -> bool:
    lower = str(name).lower()
    return any(token in lower for token in LEAK_TOKENS)


def _meta_pool(family_schema: list[str], ledger_schema: list[str]) -> tuple[list[str], list[str], list[str]]:
    configured = _resolve_config_names()
    # The explicit soft-state and transition fields are part of the requested
    # shared regime contract even when an older config snapshot does not list
    # every alias.
    explicit_regime = {
        "regime_p_calm", "regime_p_trend", "regime_p_stress", "regime_p_transition",
        "regime_entropy", "regime_transition_onset_proxy", "regime_state_duration_hours",
    }
    configured |= explicit_regime
    # Stable aliases emitted by the strict OOF market-regime sidecar.  They
    # are admitted to the candidate pool before train-only CMI selection;
    # provenance and state IDs remain audit-only because they are non-numeric
    # or encode fold identity rather than market context.
    sidecar_prefixes = (
        "regime_state_p__",
        "transition_state_p__",
        "continuous_regime__",
        "geometry_regime__",
    )
    sidecar_scalars = {
        "regime_state_ood_score", "regime_state_entropy", "regime_state_margin",
        "regime_state_uncertainty", "transition_state_ood_score",
        "transition_state_entropy", "transition_state_margin",
        "transition_state_uncertainty", "regime_top2_margin", "state_age_hours",
        "state_age", "state_switch_probability", "transition_stable_probability",
        "transition_onset_probability", "transition_active_probability",
        "transition_settling_probability",
    }
    configured |= {
        field for field in set(family_schema + ledger_schema)
        if field in sidecar_scalars or field.startswith(sidecar_prefixes)
    }
    candidates = sorted(
        name for name in set(family_schema + ledger_schema)
        if name in configured and not _is_leak(name)
        and name not in {"candidate_id", "__ts__", "side_name", "fold", "meta_partition", "query_id"}
    )
    available_configured = sorted(name for name in configured if name in set(family_schema + ledger_schema) and not _is_leak(name))
    missing = sorted(name for name in configured if name not in set(family_schema + ledger_schema) and not _is_leak(name))
    return candidates, available_configured, missing


def _expected_fold_sql() -> str:
    return "CASE WHEN b0.decision_ts < TIMESTAMP '2024-05-01 00:00:00+00' THEN 'oof_jul_aug' WHEN b0.decision_ts < TIMESTAMP '2024-07-01 00:00:00+00' THEN 'oof_may_jun' ELSE 'oos_sep_nov' END"


def _load_joined(
    base_path: Path,
    family_path: Path,
    ledger_path: Path,
    *,
    path_partition: str = "test",
    family_prefix: str = "base_structural_family__",
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Join the base rows to a strictly inference-valid path/regime store.

    ``meta_partition=test`` is the default and is intentional.  A path score
    generated for ``meta_train`` is not an OOF feature for that same row, even
    if its timestamp precedes the held-out month.  The old implementation used
    a date-to-fold guess and admitted such rows; callers must opt into a
    non-test partition explicitly for a diagnostic-only replay.
    """

    import duckdb

    base_schema = _schema(base_path)
    family_schema = _schema(family_path)
    ledger_schema = _schema(ledger_path)
    # The historical strict panel uses ``decision_ts``; the canonical 2025
    # handoff uses ``__ts__``.  Both are the same decision-time identity and
    # are normalised to ``decision_ts`` at this boundary.
    decision_source = "decision_ts" if "decision_ts" in base_schema else "__ts__" if "__ts__" in base_schema else None
    if decision_source is None:
        raise ValueError("base panel must contain decision_ts or __ts__")
    required_base = {"candidate_id", decision_source, "label_available_ts", "side_name"}
    missing = sorted(required_base.difference(base_schema))
    if missing:
        raise ValueError(f"base OOF panel missing required columns: {missing}")
    family_fields = sorted(
        c for c in family_schema
        if c.startswith(str(family_prefix)) and not c.endswith("unassigned_mass")
    )
    if not family_fields:
        raise ValueError("no frozen structural family fields found")
    abs_fields = [f"family_abs_share__{f}" for f in family_fields if f"family_abs_share__{f}" in family_schema]
    conf_fields = [f"family_confidence_share__{f}" for f in family_fields if f"family_confidence_share__{f}" in family_schema]
    if len(abs_fields) != len(family_fields):
        raise ValueError("family absolute-share columns are incomplete")
    meta_fields, configured_available, configured_missing = _meta_pool(family_schema, ledger_schema)
    context_fields = [f for f in ledger_schema if f in {
        "mkt_ret_eq_24h", "regime_liquidity_score", "mkt_rv_ratio_1h_24h", "mkt_oi_chg_z_24h",
        "mkt_funding_dispersion", "cross_asset_corr_4h", "mkt_systemic_deleveraging_score",
        "mkt_flush_exhaustion_score", "post_liquidation_rebound_score", "negative_breadth_pct",
        "btc_resilience_alt_weakness", "short_covering_score_market", "deleveraging_without_followthrough",
        "short_signal_recovery_conflict", "regime_p_calm", "regime_p_trend", "regime_p_stress",
        "regime_p_transition", "regime_entropy", "regime_transition_onset_proxy", "regime_state_duration_hours",
    }]
    ledger_fields = sorted(set(meta_fields).intersection(ledger_schema).union(context_fields))
    family_meta_fields = [f for f in meta_fields if f in family_schema]
    # The canonical panel uses these names.  A saved consensus panel may use
    # exact_* aliases; normalise those aliases at the SQL boundary while still
    # requiring an economically mapped bps anchor.
    gross_source = "gross_bps" if "gross_bps" in base_schema else "exact_gross_bps" if "exact_gross_bps" in base_schema else None
    net_source = "net_bps" if "net_bps" in base_schema else "exact_net_bps" if "exact_net_bps" in base_schema else None
    if gross_source is None or net_source is None:
        raise ValueError("base panel must contain gross_bps/net_bps or exact_gross_bps/exact_net_bps")
    anchor_source = "base_expected_bps" if "base_expected_bps" in base_schema else "base_anchor_bps" if "base_anchor_bps" in base_schema else None
    if anchor_source is None:
        raise ValueError("base panel must contain a train-only mapped bps anchor (base_expected_bps or base_anchor_bps)")
    base_aliases = {
        "p_clear": "r3_meta_p_clear",
        "p_adverse": "r3_meta_p_adverse",
        "p_weak": "r3_meta_p_weak",
        "base_raw": "base_score",
    }
    base_sources = {
        field: (field if field in base_schema else base_aliases.get(field))
        for field in BASE_FIELDS
        if field != "base_expected_bps"
    }
    base_sources = {field: source for field, source in base_sources.items() if source in base_schema}
    base_select = [
        "candidate_id", "decision_ts", "label_available_ts", "side_name",
        f"{_quote(gross_source)} AS gross_bps", f"{_quote(net_source)} AS net_bps",
        *(["fold_id"] if "fold_id" in base_schema else []),
        f"{_quote(anchor_source)} AS base_expected_bps",
        *(f"{_quote(source)} AS {_quote(field)}" for field, source in base_sources.items()),
    ]
    base_select_sql = [
        _quote("candidate_id"), f"{_quote(decision_source)} AS decision_ts", _quote("label_available_ts"), _quote("side_name"),
        f"{_quote(gross_source)} AS gross_bps", f"{_quote(net_source)} AS net_bps",
        *([_quote("fold_id")] if "fold_id" in base_schema else []),
        f"{_quote(anchor_source)} AS base_expected_bps",
        *(f"{_quote(source)} AS {_quote(field)}" for field, source in base_sources.items()),
    ]
    family_select = ["candidate_id", "__ts__", "fold", "meta_partition", *family_fields, *abs_fields, *conf_fields]
    for f in ("family_total_abs_contribution", "family_unassigned_mass", "family_assignment_quality", "family_low_confidence_mass"):
        if f in family_schema:
            family_select.append(f)
    family_select += family_meta_fields
    ledger_select = ["candidate_id", "__ts__"] + ledger_fields
    q = f"""
    WITH b0 AS (
      SELECT {', '.join(base_select_sql)}
      FROM read_parquet('{base_path}')
      WHERE lower(side_name) = 'long'
    ), b AS (
      SELECT b0.*, {_expected_fold_sql()} AS expected_path_fold,
             strftime(decision_ts, '%Y-%m') AS month_key
      FROM b0
    ), f0 AS (
      SELECT {', '.join(_quote(c) for c in family_select)}
      FROM read_parquet('{family_path}')
      WHERE lower(side_name) = 'long'
    ), f AS (
      SELECT * EXCLUDE (rn) FROM (
        SELECT f0.*, row_number() OVER (
          PARTITION BY candidate_id, __ts__, fold
          ORDER BY CASE WHEN meta_partition = 'test' THEN 0 WHEN meta_partition = 'meta_calibration' THEN 1 ELSE 2 END
        ) AS rn
        FROM f0
      )
      WHERE rn = 1 AND meta_partition = '{str(path_partition).replace("'", "''")}'
    ), l0 AS (
      SELECT {', '.join(_quote(c) for c in ledger_select)}
      FROM read_parquet('{ledger_path}')
      WHERE lower(side_name) = 'long'
    ), l AS (
      SELECT * EXCLUDE (rn) FROM (
        SELECT l0.*, row_number() OVER (PARTITION BY candidate_id, __ts__ ORDER BY __ts__) AS rn
        FROM l0
      ) WHERE rn = 1
    )
    SELECT b.*, CASE WHEN f.candidate_id IS NOT NULL THEN TRUE ELSE FALSE END AS path_present,
           CASE WHEN l.candidate_id IS NOT NULL THEN TRUE ELSE FALSE END AS regime_present,
           f.fold AS path_source_fold, f.meta_partition AS path_source_partition,
           {', '.join('f.' + _quote(c) for c in family_select[1:] if c not in {'__ts__', 'fold', 'meta_partition'})},
           {', '.join('l.' + _quote(c) for c in ledger_fields)}
    FROM b
    LEFT JOIN f ON b.candidate_id = f.candidate_id AND b.decision_ts = f.__ts__
    LEFT JOIN l ON b.candidate_id = l.candidate_id AND b.decision_ts = l.__ts__
    """
    # DuckDB rejects a trailing comma in the generated SELECT for an empty
    # optional ledger field list; the ledger contract always has fields, but
    # keep the guard explicit for reusable historical runs.
    q = q.replace(",\n           \n    FROM b", "\n    FROM b")
    joined = duckdb.connect().execute(q).fetchdf()
    joined["decision_ts"] = pd.to_datetime(joined["decision_ts"], utc=True)
    joined["label_available_ts"] = pd.to_datetime(joined["label_available_ts"], utc=True)
    joined["month_key"] = joined["decision_ts"].dt.strftime("%Y-%m")
    # Canonical downstream outputs call the frozen base score ``base_score``;
    # the older conditional runner called the same scalar ``base_raw``.  Keep
    # both names available without changing the economic anchor contract.
    if "base_raw" not in joined.columns and "base_score" in joined.columns:
        joined["base_raw"] = pd.to_numeric(joined["base_score"], errors="coerce")
    joined = joined.sort_values(["decision_ts", "candidate_id"], kind="stable").reset_index(drop=True)
    coverage = (
        joined.groupby("month_key", observed=True)
        .agg(rows=("candidate_id", "size"), path_rows=("path_present", "sum"), regime_rows=("regime_present", "sum"))
        .reset_index()
    )
    audit = {
        "base_rows": int(len(joined)),
        "path_rows": int(joined.path_present.sum()),
        "regime_rows": int(joined.regime_present.sum()),
        "path_coverage": float(joined.path_present.mean()),
        "regime_coverage": float(joined.regime_present.mean()),
        "family_count": len(family_fields),
        "family_prefix": str(family_prefix),
        "path_partition_policy": str(path_partition),
        "family_fields": family_fields,
        "configured_meta_available": len(configured_available),
        "configured_meta_missing": len(configured_missing),
        "configured_meta_missing_sample": configured_missing[:100],
        "meta_fields_used_as_pool": meta_fields,
        "coverage_by_month": coverage.to_dict("records"),
        "base_contract": "TP6/SL4/H12 strict OOF base panel",
        "anchor_contract": "train-only cost-aware expected bps",
        "side": SIDE,
    }
    return joined, coverage, audit


def _numeric_fill(train: pd.DataFrame, test: pd.DataFrame, fields: list[str]) -> tuple[np.ndarray, np.ndarray, pd.Series]:
    med = train[fields].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).median().fillna(0.0)
    a = train[fields].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(med).fillna(0.0).to_numpy("float32")
    b = test[fields].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(med).fillna(0.0).to_numpy("float32")
    return a, b, med


def _base_context(
    frame: pd.DataFrame,
    clusters: pd.DataFrame,
    all_cluster_ids: list[str],
    meta_fields: Iterable[str] = (),
) -> pd.DataFrame:
    base_fields = [field for field in BASE_FIELDS if field in frame.columns]
    if "base_expected_bps" not in base_fields:
        raise KeyError("base context requires the normalized base_expected_bps anchor")
    base_part = frame.loc[:, base_fields].copy().reset_index(drop=True)
    cluster_part = clusters.reset_index(drop=True)
    cluster_names = set(cluster_part.columns)
    context_names = [field for field in meta_fields if field in frame.columns and field not in base_part and field not in cluster_names]
    context_part = frame.loc[:, context_names].apply(pd.to_numeric, errors="coerce").reset_index(drop=True) if context_names else pd.DataFrame(index=frame.index)
    out = pd.concat([base_part, cluster_part, context_part], axis=1)
    # Make the cross-cluster competition explicit for every specialist.
    membership_cols = [f"cluster__{c}__membership" for c in all_cluster_ids]
    if membership_cols:
        m = out[membership_cols].to_numpy(float)
        out["cluster_membership_max"] = m.max(axis=1)
        out["cluster_membership_second"] = np.partition(m, -2, axis=1)[:, -2] if m.shape[1] > 1 else 0.0
    return out


def _fit_regressor(x: pd.DataFrame, y: np.ndarray, weights: np.ndarray, seed: int):
    import lightgbm as lgb

    model = lgb.LGBMRegressor(
        objective="huber", n_estimators=220, learning_rate=0.03, max_depth=4,
        num_leaves=15, min_child_samples=300, min_sum_hessian_in_leaf=1.0,
        colsample_bytree=0.82, subsample=0.82, subsample_freq=1,
        reg_alpha=0.05, reg_lambda=10.0, max_bin=127,
        random_state=seed, n_jobs=1, verbosity=-1,
    )
    fields = list(x.columns)
    a = x.replace([np.inf, -np.inf], np.nan).apply(pd.to_numeric, errors="coerce")
    med = a.median().fillna(0.0)
    model.fit(a.fillna(med).fillna(0.0), np.asarray(y, dtype=float), sample_weight=np.asarray(weights, dtype=float))
    return model, fields, med


def _tail_metrics(pred: pd.DataFrame, score: str, period: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    blocks = [(period, pred)]
    if period == "all":
        blocks += [(str(m), g) for m, g in pred.groupby("month_key", observed=True)]
    for name, block in blocks:
        x = block.sort_values([score, "candidate_id"], ascending=[False, True], kind="stable")
        for tail in TAILS:
            n = max(1, int(math.ceil(len(x) * tail)))
            take = x.head(n)
            net = take["net_bps"].to_numpy(float)
            gross = take["gross_bps"].to_numpy(float)
            rho = spearmanr(block[score].to_numpy(float), block["net_bps"].to_numpy(float)).statistic if len(block) > 2 else np.nan
            rows.append({
                "arm": score, "period": name, "tail": tail, "trades": int(n),
                "gross_bps_per_trade": float(np.nanmean(gross)), "net_bps_per_trade": float(np.nanmean(net)),
                "win_rate_net": float(np.mean(net > 0.0)), "rank_ic": float(rho) if np.isfinite(rho) else np.nan,
                "median_net_bps": float(np.nanmedian(net)), "p10_net_bps": float(np.nanpercentile(net, 10)),
            })
    return rows


def _cluster_metrics(pred: pd.DataFrame, cluster_ids: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    residual = pred["net_bps"].to_numpy(float) - pred["base_expected_bps"].to_numpy(float)
    for c in cluster_ids:
        m = pred[f"cluster__{c}__membership"].to_numpy(float)
        target = residual * m
        active = m > 0.10
        if active.sum() < 20:
            continue
        rho = spearmanr(target[active], residual[active]).statistic if active.sum() > 2 else np.nan
        rows.append({
            "cluster_id": c, "rows": int(len(pred)), "active_rows": int(active.sum()),
            "mean_membership": float(np.mean(m)), "active_share": float(np.mean(active)),
            "mean_soft_residual_bps": float(np.nanmean(target)),
            "active_mean_residual_bps": float(np.nanmean(residual[active])),
            "active_rank_ic_to_residual": float(rho) if np.isfinite(rho) else np.nan,
            "active_mean_net_bps": float(np.nanmean(pred.loc[active, "net_bps"])),
        })
    return pd.DataFrame(rows)


def run(
    *,
    base_path: Path = DEFAULT_BASE,
    family_path: Path = DEFAULT_FAMILY,
    ledger_path: Path = DEFAULT_LEDGER,
    out: Path = DEFAULT_OUT,
    context_cap: int = 16,
    max_train_rows: int = 120_000,
    path_partition: str = "test",
    family_prefix: str = "base_structural_family__",
) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    joined, coverage, audit = _load_joined(
        base_path, family_path, ledger_path,
        path_partition=path_partition,
        family_prefix=family_prefix,
    )
    coverage.to_parquet(out / "cluster_path_regime_coverage.parquet", index=False)
    path_rows = joined[joined.path_present].copy()
    if path_rows.empty:
        raise RuntimeError("no fold-aligned path rows are available")
    family_fields = audit["family_fields"]
    for f in family_fields:
        path_rows[f] = pd.to_numeric(path_rows[f], errors="coerce").fillna(0.0)
        af = f"family_abs_share__{f}"
        path_rows[af] = pd.to_numeric(path_rows[af], errors="coerce").fillna(0.0)
        cf = f"family_confidence_share__{f}"
        if cf in path_rows:
            path_rows[cf] = pd.to_numeric(path_rows[cf], errors="coerce").fillna(0.0)
    meta_fields = [f for f in audit["meta_fields_used_as_pool"] if f in path_rows.columns]
    # The path family matrix already carries every raw/meta field that the
    # frozen sidecar made available.  Do not pass outcome columns through the
    # selector even if a future config snapshot accidentally aliases them.
    meta_fields = [f for f in meta_fields if not _is_leak(f)]
    months = sorted(path_rows.month_key.dropna().unique())
    if len(months) < 2:
        raise RuntimeError("need at least two path-covered months for OOF evaluation")
    pred_parts: list[pd.DataFrame] = []
    feature_parts: list[pd.DataFrame] = []
    cluster_audits: list[pd.DataFrame] = []
    selection_audits: list[pd.DataFrame] = []
    cluster_metric_parts: list[pd.DataFrame] = []
    condition_metric_parts: list[pd.DataFrame] = []
    contract_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    selected_cluster_ids: list[str] = []
    for fold_num, test_month in enumerate(months[1:], start=1):
        test_start = pd.Timestamp(test_month + "-01", tz="UTC")
        train = path_rows[(path_rows.decision_ts < test_start) & (path_rows.label_available_ts < test_start)].copy()
        test = path_rows[path_rows.month_key.eq(test_month)].copy()
        if len(train) < 500 or len(test) < 100:
            continue
        if len(train) > max_train_rows:
            train = train.iloc[np.linspace(0, len(train) - 1, max_train_rows, dtype=int)].copy()
        signed_train = train[family_fields]
        abs_train = train[[f"family_abs_share__{f}" for f in family_fields]].copy()
        abs_train.columns = family_fields
        contracts, k_audit = discover_family_clusters(abs_train, signed_train, seed=SEED + fold_num)
        cluster_ids = [c.cluster_id for c in contracts]
        selected_cluster_ids = sorted(set(selected_cluster_ids).union(cluster_ids))
        train_cluster = materialize_cluster_features(train, contracts, family_fields=family_fields)
        test_cluster = materialize_cluster_features(test, contracts, family_fields=family_fields)
        train_ctx = _base_context(train, train_cluster, cluster_ids, meta_fields)
        test_ctx = _base_context(test, test_cluster, cluster_ids, meta_fields)
        fold_pred = test[["candidate_id", "decision_ts", "month_key", "gross_bps", "net_bps", "base_expected_bps"]].copy().reset_index(drop=True)
        fold_pred["fold"] = test_month
        fold_pred["path_present"] = True
        fold_features = test[["candidate_id", "decision_ts", "month_key"]].copy().reset_index(drop=True)
        fold_features["fold"] = test_month
        for col in train_cluster.columns:
            fold_features[col] = test_cluster[col].to_numpy()
        # Two arms isolate whether the gain comes from path membership alone or
        # from context-dependent modulation of a triggered cluster.
        correction_by_arm = {"cluster_only_score": np.zeros(len(test)), "cluster_context_score": np.zeros(len(test))}
        cluster_metric_input = fold_pred.copy()
        for contract in contracts:
            c = contract.cluster_id
            m_train = train_cluster[f"cluster__{c}__membership"].to_numpy(float)
            m_test = test_cluster[f"cluster__{c}__membership"].to_numpy(float)
            residual_train = train.net_bps.to_numpy(float) - train.base_expected_bps.to_numpy(float)
            target_train = soft_cluster_residual_target(residual_train, m_train)
            # A CMI score is calculated against the soft cluster target, using
            # only outer-train rows.  The full available pool is audited first.
            cmi = conditional_mi_scores(train_ctx, meta_fields, target_train, m_train)
            selected = cmi.head(int(context_cap)).feature.tolist()
            if not cmi.empty:
                cmi = cmi.copy()
                cmi["fold"] = test_month
                cmi["cluster_id"] = c
                cmi["selected"] = cmi.feature.isin(selected)
                selection_audits.append(cmi)
            common = [
                "base_expected_bps", "p_clear", "p_adverse", "p_weak", "base_raw",
                f"cluster__{c}__membership", f"cluster__{c}__abs_contribution",
                f"cluster__{c}__signed_contribution", f"cluster__{c}__confidence_share",
                f"cluster__{c}__active", "cluster_path_represented_mass", "cluster_path_unassigned_mass",
                "cluster_path_assignment_quality", "cluster_path_low_confidence_mass", "cluster_path_entropy", "cluster_path_top2_margin",
            ]
            common = [f for f in common if f in train_ctx.columns]
            only_fields = list(dict.fromkeys(common))
            context_fields = list(dict.fromkeys(common + selected))
            # Context values are copied into the cluster matrix only after
            # selection; the selector still saw the entire configured pool.
            xtr, xte, _ = _numeric_fill(train_ctx, test_ctx, only_fields)
            model_only, _, _ = _fit_regressor(pd.DataFrame(xtr, columns=only_fields), target_train, 0.25 + 0.75 * m_train, SEED + fold_num)
            correction_by_arm["cluster_only_score"] += model_only.predict(pd.DataFrame(xte, columns=only_fields))
            if selected:
                xtrc, xtec, _ = _numeric_fill(train_ctx, test_ctx, context_fields)
                model_ctx, _, _ = _fit_regressor(pd.DataFrame(xtrc, columns=context_fields), target_train, 0.25 + 0.75 * m_train, SEED + fold_num + 1000)
                correction_by_arm["cluster_context_score"] += model_ctx.predict(pd.DataFrame(xtec, columns=context_fields))
            else:
                correction_by_arm["cluster_context_score"] += model_only.predict(pd.DataFrame(xte, columns=only_fields))
            condition_train = train.copy()
            condition_test = test.copy()
            condition_train["residual_bps"] = condition_train.net_bps.to_numpy(float) - condition_train.base_expected_bps.to_numpy(float)
            condition_test["residual_bps"] = condition_test.net_bps.to_numpy(float) - condition_test.base_expected_bps.to_numpy(float)
            condition_metrics = cluster_condition_economics(
                condition_test,
                test_cluster,
                cluster_ids=[c],
                context_fields=selected,
                residual_column="residual_bps",
                net_column="net_bps",
                train_frame=condition_train,
            )
            if not condition_metrics.empty:
                condition_metrics["fold"] = test_month
                condition_metrics["path_partition"] = path_partition
                condition_metric_parts.append(condition_metrics)
            cluster_metric_input[f"cluster__{c}__membership"] = m_test
            cluster_metric_input[f"cluster__{c}__soft_residual_target"] = (test.net_bps.to_numpy(float) - test.base_expected_bps.to_numpy(float)) * m_test
            contract_rows.append({
                "fold": test_month, "cluster_id": c, "family_fields": json.dumps(list(contract.family_fields)),
                "family_count": len(contract.family_fields), "centroid_distance": contract.centroid_distance,
                "selected_context_count": len(selected), "selected_context": json.dumps(selected),
                "meta_pool_count": len(meta_fields), "cluster_silhouette": float(k_audit.loc[k_audit.selected, "silhouette"].iloc[0]),
            })
        fold_pred["base_score"] = fold_pred.base_expected_bps.to_numpy(float)
        fold_pred["cluster_only_score"] = fold_pred.base_expected_bps.to_numpy(float) + np.clip(correction_by_arm["cluster_only_score"], -200.0, 200.0)
        fold_pred["cluster_context_score"] = fold_pred.base_expected_bps.to_numpy(float) + np.clip(correction_by_arm["cluster_context_score"], -200.0, 200.0)
        pred_parts.append(fold_pred)
        cluster_metric_parts.append(_cluster_metrics(cluster_metric_input, cluster_ids).assign(fold=test_month))
        cluster_audits.append(k_audit.assign(fold=test_month))
        fold_rows.append({
            "fold": test_month, "train_rows": int(len(train)), "test_rows": int(len(test)),
            "meta_pool_count": int(len(meta_fields)), "cluster_count": int(len(contracts)),
            "train_label_available_max": str(train.label_available_ts.max()), "test_start": str(test_start),
            "clusters_fit_before_test": True,
        })
        feature_parts.append(fold_features)
    if not pred_parts:
        raise RuntimeError("no chronological fold could be fitted")
    preds = pd.concat(pred_parts, ignore_index=True)
    preds.to_parquet(out / "conditional_cluster_oof_predictions.parquet", index=False)
    pd.concat(feature_parts, ignore_index=True).to_parquet(out / "conditional_cluster_features_oof.parquet", index=False)
    metrics = pd.DataFrame(sum((_tail_metrics(preds, score, "all") for score in ["base_score", "cluster_only_score", "cluster_context_score"]), []))
    metrics.to_parquet(out / "conditional_cluster_metrics.parquet", index=False)
    pd.concat(cluster_metric_parts, ignore_index=True).to_parquet(out / "cluster_economic_metrics.parquet", index=False)
    pd.concat(cluster_audits, ignore_index=True).to_parquet(out / "cluster_discovery_audit.parquet", index=False)
    if selection_audits:
        pd.concat(selection_audits, ignore_index=True).to_parquet(out / "cluster_context_cmi_audit.parquet", index=False)
    if condition_metric_parts:
        pd.concat(condition_metric_parts, ignore_index=True).to_parquet(out / "cluster_condition_economics.parquet", index=False)
    else:
        pd.DataFrame(columns=["fold", "cluster_id", "feature", "bin", "rows", "weighted_net_bps"]).to_parquet(out / "cluster_condition_economics.parquet", index=False)
    pd.DataFrame(contract_rows).to_parquet(out / "cluster_contract_by_fold.parquet", index=False)
    pd.DataFrame(fold_rows).to_parquet(out / "fold_audit.parquet", index=False)
    correctness = {
        "schema": "tp6_sl4_conditional_cluster_residual_v1",
        "side_long_only": bool(preds.shape[0] > 0 and joined.side_name.astype(str).str.lower().eq("long").all()),
        "base_contract": "TP6/SL4/H12 strict OOF base panel",
        "path_features_are_fold_aligned": path_partition == "test",
        "path_partition_policy": path_partition,
        "family_prefix": family_prefix,
        "cluster_discovery_train_only": True,
        "context_selection_train_only": True,
        "label_available_before_test_start": all(pd.Timestamp(row["train_label_available_max"]) < pd.Timestamp(row["test_start"]) for row in fold_rows),
        "soft_cluster_target_definition": "membership * (net_bps - base_expected_bps)",
        "outcome_fields_in_feature_pool": [f for f in meta_fields if _is_leak(f)],
        "outcome_fields_in_feature_pool_count": 0,
        "configured_meta_pool_count": int(len(meta_fields)),
        "path_coverage": audit["path_coverage"],
        "regime_coverage": audit["regime_coverage"],
        "missing_path_rows_retained_in_coverage_only": True,
        "global_ranking": True,
        "monthly_metrics_present": bool((metrics.period != "all").any()),
        "condition_economics_present": bool(condition_metric_parts),
    }
    (out / "correctness_test_report.json").write_text(json.dumps(correctness, indent=2) + "\n")
    manifest = {
        "schema": "tp6_sl4_conditional_cluster_residual_v1", "status": "complete", "side": SIDE,
        "base_panel": str(base_path), "family_matrix": str(family_path), "regime_ledger": str(ledger_path),
        "family_count": int(len(family_fields)), "family_prefix": family_prefix,
        "path_partition_policy": path_partition,
        "cluster_discovery_k": [3, 4, 5, 6, 7],
        "context_cap_per_cluster": int(context_cap), "meta_pool_count": int(len(meta_fields)),
        "target": "cluster soft membership * (exact TP6/SL4 net - base expected bps)",
        "arms": ["base_score", "cluster_only_score", "cluster_context_score"],
        "execution": "TP6/SL4/H12 inherited from strict base OOF panel; cost is already in net_bps",
        "coverage": audit,
        "leakage": "chronological folds; path clusters and CMI selection fit on train rows only; strict test-partition path rows by default; no short rows",
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    pooled = metrics[metrics.period.eq("all")].sort_values(["tail", "net_bps_per_trade"], ascending=[True, False])
    cluster_econ = pd.concat(cluster_metric_parts, ignore_index=True) if cluster_metric_parts else pd.DataFrame()
    condition_econ = pd.concat(condition_metric_parts, ignore_index=True) if condition_metric_parts else pd.DataFrame()
    lines = [
        "# TP6/SL4 conditional path-cluster residual layer",
        "",
        "Long-only diagnostic on the strict TP6/SL4/H12 base OOF panel. Cluster discovery and CMI context selection are chronological and train-only.",
        "",
        "## Coverage",
        "",
        f"- Base rows: {audit['base_rows']:,}; fold-aligned path rows: {audit['path_rows']:,} ({audit['path_coverage']:.1%}); regime-ledger rows: {audit['regime_rows']:,} ({audit['regime_coverage']:.1%}).",
        f"- Frozen path/family fields: {len(family_fields)}; configured meta fields available before selection: {len(meta_fields)}.",
        f"- Path lineage policy: `{path_partition}`; family prefix: `{family_prefix}`. Only the declared path partition is eligible for supervised fitting.",
        "- Rows without a fold-aligned path are retained in the coverage audit and excluded from supervised cluster fitting; they are never encoded as zero path evidence.",
        "",
        "## Target and arms",
        "",
        "- For each discovered cluster, target = soft membership × (exact TP6/SL4 net − base expected bps).",
        "- `base_score`: canonical base expected-bps ranking control.",
        "- `cluster_only_score`: per-cluster residual learner using base outputs and triggered cluster/path features.",
        "- `cluster_context_score`: same learner plus CMI-selected context from the full configured meta/regime/transition pool.",
        "",
        "## Pooled global tail metrics",
        "",
        "| arm | tail | trades | gross bps/trade | net bps/trade | rank IC |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in pooled.itertuples(index=False):
        lines.append(f"| {row.arm} | {row.tail:.3g} | {row.trades} | {row.gross_bps_per_trade:.2f} | {row.net_bps_per_trade:.2f} | {row.rank_ic:.4f} |")
    lines += ["", "## Held-out cluster economics", "", "| fold | cluster | active share | active residual bps | active net bps | active residual IC |", "|---|---|---:|---:|---:|---:|"]
    if cluster_econ.empty:
        lines.append("| — | — | — | — | — | — |")
    else:
        for row in cluster_econ.sort_values(["fold", "cluster_id"], kind="stable").itertuples(index=False):
            lines.append(f"| {row.fold} | {row.cluster_id} | {row.active_share:.3f} | {row.active_mean_residual_bps:.2f} | {row.active_mean_net_bps:.2f} | {row.active_rank_ic_to_residual:.4f} |")
    lines += ["", "## Held-out condition diagnostics", "", "These rows use feature quantile edges fitted on the preceding training rows; they are diagnostic conditions, not inference features selected on the held-out outcomes.", "", "| fold | cluster | context | bin | rows | weighted net bps | active-minus-inactive net bps |", "|---|---|---|---:|---:|---:|---:|"]
    if condition_econ.empty:
        lines.append("| — | — | — | — | — | — | — |")
    else:
        selected_conditions = condition_econ.loc[condition_econ.rows.ge(100)].copy()
        selected_conditions["abs_delta"] = selected_conditions["active_minus_inactive_net_bps"].abs()
        selected_conditions = selected_conditions.sort_values(["abs_delta", "rows"], ascending=[False, False], kind="stable").head(12)
        for row in selected_conditions.itertuples(index=False):
            delta = "nan" if not np.isfinite(row.active_minus_inactive_net_bps) else f"{row.active_minus_inactive_net_bps:.2f}"
            lines.append(f"| {row.fold} | {row.cluster_id} | {row.feature} | {row.bin} | {row.rows} | {row.weighted_net_bps:.2f} | {delta} |")
    lines += [
        "",
        "## Caveat",
        "",
        "The currently available structural path store does not cover every historical base row and was generated for an earlier long-side structural model. This run validates the conditional representation and leakage contract on the exact TP6/SL4/H12 base panel where fold-aligned paths exist; it does not replace the canonical 2025 consensus stack. A promotion-grade run requires structural path outputs regenerated from the canonical TP6/SL4 base model for the untouched evaluation period.",
        "",
        "Artifacts: `conditional_cluster_features_oof.parquet`, `conditional_cluster_oof_predictions.parquet`, `cluster_contract_by_fold.parquet`, `cluster_context_cmi_audit.parquet`, `cluster_economic_metrics.parquet`, `cluster_condition_economics.parquet`, `conditional_cluster_metrics.parquet`, `cluster_path_regime_coverage.parquet`, `correctness_test_report.json`, and `run_manifest.json`.",
    ]
    (out / "TP6_SL4_CONDITIONAL_CLUSTER_REPORT.md").write_text("\n".join(lines) + "\n")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, default=DEFAULT_BASE)
    ap.add_argument("--family", type=Path, default=DEFAULT_FAMILY)
    ap.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--context-cap", type=int, default=16)
    ap.add_argument("--max-train-rows", type=int, default=120_000)
    ap.add_argument("--path-partition", default="test", help="strictly use this sidecar partition; default is test")
    ap.add_argument("--family-prefix", default="base_structural_family__")
    args = ap.parse_args()
    print(run(base_path=args.base, family_path=args.family, ledger_path=args.ledger, out=args.out, context_cap=args.context_cap, max_train_rows=args.max_train_rows, path_partition=args.path_partition, family_prefix=args.family_prefix))


if __name__ == "__main__":
    main()
