#!/usr/bin/env python3
"""Frozen cross-fold specialist retraining and residual-input ablations.

The legacy broad-multiview runner rediscovered ``data_view_*`` independently
for every fold and side.  This runner discovers one causal template from the
pre-transport development population, freezes the exact field membership and
ordering, then refits each specialist inside every transport fold.

Specialist target is fixed to ``H12 net > +50 bps``.  The residual target is
the existing per-row ordinal net residual, trained with native LambdaRank.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import re
import sys
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.features_gmm_ae import AE_GMM_FEATURE_COLUMNS
from extreme_price_movements.multiview_specialists import (
    apply_synergy_features,
    discover_broad_opportunity_views,
    opportunity_conditioned_synergy,
)
from extreme_price_movements.packb_static_point_feature_loader import (
    _provenance_backed_raw_allowlist,
)
from extreme_price_movements.specialist_head_selection import (
    select_complementary_heads,
)
from scripts.run_broad_multiview_specialist_lambdarank import (
    DELAY,
    LEDGER,
    LONG_HISTORY_FOLDS,
    MAX_PROXY_ROWS,
    MAX_TRAIN_ROWS,
    SEED,
    STORE,
    _base,
    _metric,
    _rank_target,
    _ranker,
    _sample,
    _store_rows,
    _utc,
    fit_residual_calibration,
)

SPECIALIST_COUNT = 7
OUT = ROOT / "data_perp/artifacts/frozen_multiview_specialist_input_ablation_20260805_v1"
TAILS = (.01, .05, .10)


def _schema() -> list[str]:
    con = duckdb.connect()
    rows = con.execute("DESCRIBE SELECT * FROM read_parquet(?)", [str(STORE)]).fetchall()
    con.close()
    excluded = {"candidate_id", "__ts__", "__symbol__", "__decision_ts__", "side_name"}
    registry_allowlist, _, _, _ = _provenance_backed_raw_allowlist()
    cols = [str(r[0]) for r in rows if str(r[0]) not in excluded]
    selected = [c for c in cols if c in registry_allowlist]
    if not selected:
        raise ValueError("no provenance-backed specialist fields in store schema")
    return sorted(dict.fromkeys(selected))


def _store_columns() -> set[str]:
    con = duckdb.connect()
    rows = con.execute("DESCRIBE SELECT * FROM read_parquet(?)", [str(STORE)]).fetchall()
    con.close()
    return {str(r[0]) for r in rows}


def _template_rows(base: pd.DataFrame, fold_names: list[str]) -> pd.DataFrame:
    folds = [f for f in LONG_HISTORY_FOLDS if f.name in fold_names]
    if not folds:
        raise ValueError("no valid folds selected")
    earliest_test = min(_utc(f.test_start) for f in folds)
    # The template is fit only before the earliest transport test period.  It
    # therefore uses cross-fold development rows without looking into any
    # transport test month.
    rows = base[base.__ts__.lt(earliest_test) & base.label_available_ts.lt(earliest_test)].copy()
    if rows.empty:
        raise ValueError("empty pre-transport template population")
    return rows


def _eligible_template_fields(frame: pd.DataFrame, fields: list[str]) -> tuple[list[str], pd.DataFrame]:
    records: list[dict[str, object]] = []
    eligible: list[str] = []
    for field in fields:
        value = pd.to_numeric(frame[field], errors="coerce").to_numpy(float)
        finite = np.isfinite(value)
        coverage = float(finite.mean())
        median = float(np.nanmedian(value)) if finite.any() else np.nan
        scale = float(np.nanmedian(np.abs(value - median)) * 1.4826) if finite.any() else 0.0
        ok = coverage >= 0.90 and np.isfinite(scale) and scale > 1e-8
        if ok:
            eligible.append(field)
        records.append({"feature": field, "template_coverage": coverage, "template_robust_scale": scale, "eligible": ok})
    return eligible, pd.DataFrame(records)


def _family_pick(frame: pd.DataFrame, candidates: list[str], available: set[str]) -> str:
    choices = [x for x in candidates if x in available]
    if not choices:
        raise ValueError(f"none of the requested context candidates are available: {candidates}")
    scored: list[tuple[float, str]] = []
    for field in choices:
        value = pd.to_numeric(frame[field], errors="coerce").to_numpy(float)
        finite = np.isfinite(value)
        scale = float(np.nanmedian(np.abs(value[finite] - np.nanmedian(value[finite]))) * 1.4826) if finite.any() else 0.0
        scored.append((float(finite.mean()) if np.isfinite(scale) and scale > 1e-8 else -1.0, field))
    scored.sort(key=lambda x: (-x[0], x[1]))
    if scored[0][0] < 0.90:
        raise ValueError(f"no {candidates} field has >=90% template coverage/variance")
    return scored[0][1]


def _select_six_context_fields(frame: pd.DataFrame, available: set[str]) -> list[str]:
    return [
        _family_pick(frame, ["trend_slope_48h", "ema20_slope_5h", "ema_slope_norm", "trend_strength_percentile"], available),
        _family_pick(frame, ["volume_percentile", "rvol_z_peer_resid", "mkt_volume_z_24h", "volume_trend_48"], available),
        _family_pick(frame, ["funding_z", "funding_rate", "funding_per_hour", "funding_abs_z"], available),
        _family_pick(frame, ["oi_chg_z_4h", "oi_value_log_1d_robust_z", "mkt_oi_chg_z_24h", "oi_z"], available),
        _family_pick(frame, ["rv_24h", "mkt_rv_4h", "volatility_ratio_short_long", "atr_percentile"], available),
        _family_pick(frame, ["vwap_zone_1d_atr", "mkt_close_location_1h", "distance_to_support_daily_vwap_atr", "distance_to_resistance_daily_vwap_atr"], available),
    ]


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def _template(base: pd.DataFrame, available: list[str], folds: list[str], out: Path, contract_path: Path | None = None) -> tuple[dict[str, dict[str, list[str]]], pd.DataFrame, list[str], list[str]]:
    if contract_path is not None:
        contract = json.loads(contract_path.read_text())
        if contract.get("specialist_target") != "exact_h12_net_bps_gt_50":
            raise ValueError("external view contract has the wrong specialist target")
        views_by_side = {str(side): {str(name): list(map(str, fields)) for name, fields in views.items()} for side, views in contract["views_by_side"].items()}
        if set(views_by_side) != {"long", "short"} or any(len(views) != SPECIALIST_COUNT for views in views_by_side.values()):
            raise ValueError("external view contract must contain seven views per side")
        missing = sorted({field for views in views_by_side.values() for fields in views.values() for field in fields}.difference(available))
        if missing:
            raise ValueError(f"external view contract fields absent from current causal store schema: {missing[:10]}")
        records = [
            {"side": side, "specialist": name, "feature": field, "audit": "external_frozen_membership", "selected": True}
            for side, views in views_by_side.items() for name, fields in views.items() for field in fields
        ]
        ae_fields = [str(field) for field in contract.get("ae_gmm_fields", []) if str(field) in _store_columns()]
        six = [str(field) for field in contract.get("selected_context_fields", []) if str(field) in available]
        _write_json(out / "frozen_view_contract.json", {**contract, "reused_from": str(contract_path)})
        return views_by_side, pd.DataFrame(records), ae_fields, six
    template_base = _template_rows(base, folds)
    audit_frames: list[pd.DataFrame] = []
    views_by_side: dict[str, dict[str, list[str]]] = {}
    for side in ("long", "short"):
        side_base = template_base[template_base.side_name.eq(side)].copy()
        proxy = _sample(side_base, max(MAX_PROXY_ROWS, 30_000))
        proxy = proxy.merge(_store_rows(proxy, available), on="candidate_id", validate="one_to_one")
        eligible, audit = _eligible_template_fields(proxy, available)
        if len(eligible) < SPECIALIST_COUNT * 40:
            raise ValueError(f"{side}: only {len(eligible)} eligible template fields for {SPECIALIST_COUNT} specialists")
        proxy["binary_h12_net50"] = (proxy.net_bps > 50.0).astype(np.int8)
        min_per_view = max(40, min(80, len(eligible) // SPECIALIST_COUNT))
        views, field_audit, edges = discover_broad_opportunity_views(
            proxy,
            eligible,
            base_score_column="base_score",
            label_column="binary_h12_net50",
            specialist_count=SPECIALIST_COUNT,
            min_features_per_view=min_per_view,
            max_features_per_view=min(80, max(min_per_view, len(eligible) // SPECIALIST_COUNT)),
            max_proxy_features=min(560, len(eligible)),
            min_joint_rows=80,
        )
        for name, fields in views.items():
            if len(fields) < 40:
                raise ValueError(f"{side}/{name}: frozen view has only {len(fields)} fields")
        views_by_side[side] = views
        audit["side"] = side
        audit["audit"] = "template_coverage"
        field_audit["side"] = side
        field_audit["audit"] = "template_activation"
        edges["side"] = side
        edges["audit"] = "template_joint_synergy"
        audit_frames.extend([audit, field_audit, edges])
    all_fields = sorted({field for side_views in views_by_side.values() for fields in side_views.values() for field in fields})
    available_set = set(available)
    ae_fields = [x for x in AE_GMM_FEATURE_COLUMNS if x in _store_columns()]
    six = _select_six_context_fields(template_base.merge(_store_rows(_sample(template_base, 30_000), available), on="candidate_id", validate="one_to_one"), available_set)
    template_cutoff = min(_utc(f.test_start) for f in LONG_HISTORY_FOLDS if f.name in folds)
    contract = {
        "schema": "frozen_cross_fold_specialist_template_v1",
        "specialist_target": "exact_h12_net_bps_gt_50",
        "template_cutoff": str(template_cutoff),
        "source_folds": folds,
        "specialist_count": SPECIALIST_COUNT,
        "views_by_side": views_by_side,
        "view_feature_counts": {side: {name: len(fields) for name, fields in views.items()} for side, views in views_by_side.items()},
        "ae_gmm_fields": ae_fields,
        "selected_context_fields": six,
        "all_specialist_fields_sha256": hashlib.sha256("\n".join(all_fields).encode()).hexdigest(),
    }
    _write_json(out / "frozen_view_contract.json", contract)
    return views_by_side, pd.concat(audit_frames, ignore_index=True), ae_fields, six


def _regime_context_fields(frame: pd.DataFrame) -> list[str]:
    exclude = {"target__exact_net_residual_bps", "target__soft_regime_centered_residual_bps", "target__soft_regime_standardized_residual"}
    prefixes = ("regime_", "regime_relative__", "regime_z__", "soft_regime_", "regime_transition_", "regime_state_duration")
    out = [c for c in frame.columns if c not in exclude and c.startswith(prefixes)]
    return [c for c in out if pd.api.types.is_numeric_dtype(frame[c])]


def _month_metric(frame: pd.DataFrame, score_col: str, fold: str, arm: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    x = frame.copy()
    x["month"] = pd.to_datetime(x["__ts__"], utc=True).dt.to_period("M").astype(str)
    for month, q in x.groupby("month", sort=True):
        for side, subset in [("pooled", q), *[(s, q[q.side_name.eq(s)]) for s in ("long", "short")]]:
            if len(subset) == 0:
                continue
            for tail in TAILS:
                n = max(1, int(np.ceil(len(subset) * tail)))
                top = subset.sort_values([score_col, "candidate_id"], ascending=[False, True], kind="stable").head(n)
                rows.append({"fold": fold, "month": month, "side": side, "arm": arm, "tail": tail, "rows": len(subset), "tail_rows": n, "net_bps": float(top.net_bps.mean()), "gross_bps": float(top.gross_bps.mean()), "rank_ic": float(subset[score_col].rank().corr(subset.net_bps.rank()))})
    return rows


def _fit_specialists(train: pd.DataFrame, cal: pd.DataFrame, test: pd.DataFrame, views: dict[str, list[str]]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    fit = _sample(train, MAX_TRAIN_ROWS)
    out_fit: dict[str, np.ndarray] = {}
    out_cal: dict[str, np.ndarray] = {}
    out_test: dict[str, np.ndarray] = {}
    for view, fields in views.items():
        fitx = fit.merge(_store_rows(fit, fields), on="candidate_id", validate="one_to_one")
        calx = cal.merge(_store_rows(cal, fields), on="candidate_id", validate="one_to_one")
        testx = test.merge(_store_rows(test, fields), on="candidate_id", validate="one_to_one")
        med = fitx[fields].median()
        X = fitx[fields].fillna(med).astype(np.float32)
        C = calx[fields].fillna(med).astype(np.float32)
        T = testx[fields].fillna(med).astype(np.float32)
        target = (fitx.net_bps.to_numpy(float) > 50.0).astype(np.int8)
        clf = lgb.LGBMClassifier(objective="binary", n_estimators=180, learning_rate=.04, num_leaves=20, min_child_samples=400, colsample_bytree=.8, reg_lambda=20., random_state=SEED, n_jobs=1, verbosity=-1).fit(X, target)
        out_fit[view] = clf.predict_proba(X)[:, 1]
        out_cal[view] = clf.predict_proba(C)[:, 1]
        out_test[view] = clf.predict_proba(T)[:, 1]
        del fitx, calx, testx, X, C, T, clf
        gc.collect()
    return out_fit, out_cal, out_test


def _fold(base: pd.DataFrame, views_by_side: dict[str, dict[str, list[str]]], ae_fields: list[str], six_fields: list[str], fold, metrics: list[dict[str, object]], predictions: list[pd.DataFrame], discovery: list[pd.DataFrame], *, max_meta_heads: int) -> None:
    a, b, c, e = map(_utc, (fold.train_start, fold.calibration_start, fold.test_start, fold.test_end))
    tr = base[base.__ts__.between(a, b, inclusive="left") & base.label_available_ts.lt(b)]
    ca = base[base.__ts__.between(b, c, inclusive="left") & base.label_available_ts.lt(c)]
    te = base[base.__ts__.between(c, e, inclusive="left")]
    for side in ("long", "short"):
        train, cal, test = (x[x.side_name.eq(side)].copy() for x in (tr, ca, te))
        _, cal_scores, test_scores = _fit_specialists(train, cal, test, views_by_side[side])
        mapping = {view: "mv__" + view for view in views_by_side[side]}
        cal_s, test_s = cal.copy(), test.copy()
        for view, col in mapping.items():
            cal_s[col] = cal_scores[view]
            test_s[col] = test_scores[view]
            z = test[["candidate_id", "__ts__", "side_name", "net_bps", "gross_bps"]].copy()
            z["score"] = test_scores[view]
            z["fold"] = fold.name
            z["specialist"] = view
            predictions.append(z)
            metrics.extend(row for row in _metric(test, test_scores[view], fold.name, "stable_specialist", "binary_h12_net50", "standalone", view) if row["side"] == side)
        # Discover synergy/routing only after the stable specialist columns are
        # aligned.  Fit it on the earlier half of calibration and apply it to
        # the residual-training half and OOS test.
        cal_s = cal_s.sort_values(["__ts__", "candidate_id"], kind="stable").reset_index(drop=True)
        select_rows = max(1, len(cal_s) // 2)
        select_frame = cal_s.iloc[:select_rows].copy()
        select_frame["binary_h12_net50"] = (select_frame.net_bps > 50.0).astype(np.int8)
        diag, _ = opportunity_conditioned_synergy(select_frame, mapping, base_score_column="base_score", label_column="binary_h12_net50")
        _, pair_template = opportunity_conditioned_synergy(select_frame, mapping, base_score_column="base_score", label_column="binary_h12_net50")
        # Recompute the applied pair columns from the train-only diagnostics.
        cal_after = cal_s.iloc[select_rows:].copy()
        cal_after_pairs = apply_synergy_features(cal_after, mapping, diag, base_score_column="base_score")
        test_pairs = apply_synergy_features(test_s, mapping, diag, base_score_column="base_score")
        for col in cal_after_pairs.columns:
            cal_after[col] = cal_after_pairs[col]
            test_s[col] = test_pairs[col] if col in test_pairs else 0.0
        pair_fields = list(cal_after_pairs.columns)
        specialist_fields = list(mapping.values())
        select_frame["residual_grade"] = _rank_target(
            select_frame.net_bps.to_numpy(float) - select_frame.prequential_base_expected_net_bps.to_numpy(float)
        )
        selected_specialists, head_audit = select_complementary_heads(
            select_frame, specialist_fields, target_column="residual_grade",
            base_score_column="base_score", max_heads=max_meta_heads, minimum_cmi=.001,
        )
        discovery.append(head_audit.assign(fold=fold.name, side=side, audit="matched_cmi_head_selection"))
        selected_pairs = [
            col for col in pair_fields
            if any(f"__{name.removeprefix('mv__')}__" in col for name in selected_specialists)
        ]
        base_fields = ["p_clear", "p_adverse", "p_weak", "base_score", "prequential_base_expected_net_bps"]
        regime_fields = [f for f in _regime_context_fields(cal_after) if f not in base_fields]
        # Join the frozen AE/GMM and selected context fields once, then reuse
        # the exact aligned columns for every residual-input arm.
        extra_fields = list(dict.fromkeys(ae_fields + six_fields))
        if extra_fields:
            cal_extra = _store_rows(cal_after, extra_fields)
            test_extra = _store_rows(test_s, extra_fields)
            cal_after = cal_after.merge(cal_extra, on="candidate_id", validate="one_to_one")
            test_s = test_s.merge(test_extra, on="candidate_id", validate="one_to_one")
        # All-seven is the historical frozen-contract control.  CMI-six uses
        # only scores selected on the earlier, strictly prior calibration half.
        arms = {
            "all7_heads_only": specialist_fields,
            "all7_plus_ae_gmm": specialist_fields + ae_fields,
            "all7_full_context": base_fields + specialist_fields + pair_fields + regime_fields + six_fields,
            "cmi6_heads_only": selected_specialists,
            "cmi6_plus_ae_gmm": selected_specialists + ae_fields,
            "cmi6_full_context": base_fields + selected_specialists + selected_pairs + regime_fields + six_fields,
        }
        # Keep a fixed order and only numeric columns present in both frames.
        for arm, fields in arms.items():
            fields = list(dict.fromkeys(f for f in fields if f in cal_after.columns and f in test_s.columns and pd.api.types.is_numeric_dtype(cal_after[f])))
            residual = cal_after.net_bps.to_numpy(float) - cal_after.prequential_base_expected_net_bps.to_numpy(float)
            meta_frame = pd.concat([cal_after[["__ts__"]], cal_after[fields]], axis=1)
            model, usable = _ranker(meta_frame, _rank_target(residual))
            raw_cal = model.predict(cal_after[usable].fillna(0.0))
            raw_test = model.predict(test_s[usable].fillna(0.0))
            iso = fit_residual_calibration(raw_cal, residual)
            score = test_s.prequential_base_expected_net_bps.to_numpy(float) + np.clip(iso.predict(raw_test), -50., 50.)
            z = test[["candidate_id", "__ts__", "side_name", "net_bps", "gross_bps", "prequential_base_expected_net_bps"]].copy()
            z["score"] = score
            z["fold"] = fold.name
            z["arm"] = arm
            z["input_count"] = len(usable)
            predictions.append(z)
            metrics.extend(_metric(test, score, fold.name, "stable_residual", "ordinal_net_residual", "meta_impact"))
            metrics.extend(_month_metric(z, "score", fold.name, arm))
            del model, iso, meta_frame, raw_cal, raw_test
            gc.collect()
        discovery.append(diag.assign(fold=fold.name, side=side, audit="frozen_template_routing"))


def run(out: Path = OUT, fold_names: list[str] | None = None, *, contract_path: Path | None = None, max_meta_heads: int = 6) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    folds = [f for f in LONG_HISTORY_FOLDS[3:] if fold_names is None or f.name in fold_names]
    if not folds:
        raise ValueError("no transport folds selected")
    base = _base()
    available = _schema()
    names = [f.name for f in folds]
    views_by_side, template_audit, ae_fields, six_fields = _template(base, available, names, out, contract_path)
    metrics: list[dict[str, object]] = []
    predictions: list[pd.DataFrame] = []
    discovery: list[pd.DataFrame] = [template_audit]
    for fold in folds:
        _fold(base, views_by_side, ae_fields, six_fields, fold, metrics, predictions, discovery, max_meta_heads=max_meta_heads)
        pd.DataFrame(metrics).to_parquet(out / "metrics.checkpoint.parquet", index=False)
        if predictions:
            pd.concat(predictions, ignore_index=True).to_parquet(out / "predictions.checkpoint.parquet", index=False)
        pd.concat(discovery, ignore_index=True).to_parquet(out / "view_discovery.checkpoint.parquet", index=False)
        _write_json(out / "progress.json", {"status": "running", "completed_fold": fold.name})
    pd.DataFrame(metrics).to_parquet(out / "metrics.parquet", index=False)
    pd.concat(predictions, ignore_index=True).to_parquet(out / "predictions.parquet", index=False)
    pd.concat(discovery, ignore_index=True).to_parquet(out / "view_discovery.parquet", index=False)
    _write_json(out / "manifest.json", {"schema": "frozen_multiview_specialist_input_ablation_v2", "folds": names, "specialist_target": "exact_h12_net_bps_gt_50", "residual_target": "ordinalized_per_row_net_residual_bps", "view_contract": "external or generated cross-fold pre-transport template; exact fields frozen by side", "max_meta_heads": max_meta_heads, "input_arms": ["all7_heads_only", "all7_plus_ae_gmm", "all7_full_context", "cmi6_heads_only", "cmi6_plus_ae_gmm", "cmi6_full_context"], "ae_gmm_fields": ae_fields, "selected_context_fields": six_fields})
    _write_json(out / "progress.json", {"status": "complete", "completed_folds": names})
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--fold", action="append", dest="folds", default=None)
    parser.add_argument("--contract", type=Path, default=None, help="reuse an audited frozen specialist contract")
    parser.add_argument("--max-meta-heads", type=int, default=6)
    args = parser.parse_args()
    print(run(args.out, args.folds, contract_path=args.contract, max_meta_heads=args.max_meta_heads))
