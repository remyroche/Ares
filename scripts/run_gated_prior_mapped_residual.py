#!/usr/bin/env python3
"""Prior-mapped base score plus OOF-gated residual repair.

This runner fixes the previous semantic mismatch:

    raw R3 score -> side-local prior-resolved expected-net map
    residual target = realised net - mapped base value
    final score = mapped base value + OOF-selected gated residual correction

The base map and residual feature contract are frozen before test scoring.  A
validation tail of each calibration period selects the side-local lambda and
score-region gates; if no lambda beats the no-op residual baseline, the fold or
region is explicitly rejected and the residual is disabled there.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.prequential_r3_value_map import (
    PrequentialR3ValueMapConfig,
    prequential_same_side_r3_value_map,
)
from scripts.run_broad_multiview_specialist_lambdarank import (
    LONG_HISTORY_FOLDS,
    _base,
    _rank_target,
    _ranker,
    _utc,
    fit_residual_calibration,
)
from scripts.run_frozen_multiview_specialist_input_ablation import (
    _fit_specialists,
    _schema,
    _select_six_context_fields,
    _store_columns,
    _store_rows,
)

OUT = ROOT / "data_perp/artifacts/gated_prior_mapped_residual_20260805_v1"
SEED = 20260805
LAMBDAS = (0.0, 0.125, 0.25, 0.5, 0.75, 1.0)
MAP_BINS = 20
MAP_SHRINK = 64.0
TAILS = (.01, .05, .10)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def _load_frozen_views(path: Path, side: str) -> dict[str, list[str]]:
    payload = json.loads((path / "frozen_view_contract.json").read_text())
    views = payload["views_by_side"][side]
    if sorted(views) != [f"data_view_{i:02d}" for i in range(7)]:
        raise ValueError("frozen specialist contract is not the expected seven-view contract")
    if any(len(fields) != 68 for fields in views.values()):
        raise ValueError("frozen specialist contract changed field count")
    return {str(k): [str(v) for v in values] for k, values in views.items()}


def _base_map(base: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    mapped: list[pd.DataFrame] = []
    audits: list[pd.DataFrame] = []
    config = {
        "bins": MAP_BINS,
        "min_global_rows": 32,
        "bin_shrink_rows": MAP_SHRINK,
        "mapping_mode": "monotone_pava",
        "monotone_min_bin_rows": 1,
    }
    for side in ("long", "short"):
        q = base[base.side_name.eq(side)].sort_values(["__ts__", "candidate_id"], kind="stable")
        values, audit, provenance = prequential_same_side_r3_value_map(
            exact_net_bps=q.net_bps.to_numpy(float),
            decision_timestamps=q.__ts__,
            label_available_timestamps=q.label_available_ts,
            side=side,
            score=q.base_score.to_numpy(float),
            config=PrequentialR3ValueMapConfig(side=side, **config),
        )
        out = q[["candidate_id", "__ts__", "side_name"]].copy()
        out["causal_base_map_bps"] = values.astype(np.float32)
        out["map_prior_global_support"] = audit.prior_resolved_global_support.to_numpy(np.int32)
        out["map_prior_bin_support"] = audit.prior_resolved_bin_support.to_numpy(np.int32)
        out["map_neutral_fallback"] = audit.value_map_fallback.astype(str).isin([
            "neutral_no_prior_resolved_support",
            "global_prior_fallback_empty_bin",
            "monotone_global_prior_fallback_sparse_bin",
        ]).astype(np.float32).to_numpy()
        out["map_score_bin"] = audit.r3_score_bin.to_numpy(np.int16)
        mapped.append(out)
        audit_out = out.copy()
        audit_out["r3_opportunity_score"] = audit.r3_opportunity_score.to_numpy(np.float32)
        audit_out["value_map_fallback"] = audit.value_map_fallback.astype(str).to_numpy()
        audits.append(audit_out)
    mapped_frame = pd.concat(mapped, ignore_index=True)
    audit_frame = pd.concat(audits, ignore_index=True)
    provenance = {
        "schema": "prior_resolved_side_local_monotone_r3_base_map_v1",
        "config": config,
        "score_semantics": "P(clear)-P(adverse)",
        "output_semantics": "causal_base_map_bps",
        "strict_boundary": "label_available_ts < decision_timestamp",
        "sides": ["long", "short"],
        "map_rows": int(len(mapped_frame)),
    }
    return mapped_frame, audit_frame, provenance


def _select_regime_context_fields(template: pd.DataFrame, store_cols: set[str]) -> list[str]:
    """Select a compact, causal regime/context ledger for the residual model.

    The base artifact is intentionally compact and does not carry the full
    store.  Select the regime/transition fields after a store join so they are
    actually present in every residual fold, rather than merely naming them in
    a contract that cannot be materialised.
    """
    prefixes = (
        "regime_",
        "market_state_transition_",
        "mkt_regime_change__",
        "soft_regime_",
        "regime_transition_",
        "regime_state_duration",
    )
    semantic = (
        "chop_score",
        "choppiness_index",
        "volatility_autocorr",
        "volatility_of_volatility",
        "liquidity_",
        "funding_",
        "oi_",
        "open_interest",
    )
    selected: list[str] = []
    # Preserve one representative of each trust/regime family even when the
    # broader OI/funding pool would otherwise fill the cap lexicographically.
    mandatory = (
        "regime_liquidity_score",
        "market_state_transition_entropy_5d",
        "chop_score_surprise",
        "choppiness_index_20",
        "funding_abs_z",
        "funding_persistence",
        "mkt_oi_chg_1h_rz",
        "mkt_oi_chg_4h_rz",
        "xasset_ob_liquidity_ts_resid",
        "xasset_ob_liquidity_peer_resid",
        "xs_dispersion__volatility_zscore",
    )
    for col in mandatory:
        if col in template and col in store_cols and pd.api.types.is_numeric_dtype(template[col]):
            selected.append(col)
    for col in sorted(store_cols):
        if col not in template or not pd.api.types.is_numeric_dtype(template[col]):
            continue
        lower = col.lower()
        if (col.startswith(prefixes) or any(token in lower for token in semantic)) and col not in selected:
            selected.append(col)
    # Keep the contract broad enough to represent regime and financing state,
    # but bounded so residual fitting remains memory-efficient.
    return selected[:80]


def _fixed_residual_contract(base: pd.DataFrame, views: dict[str, list[str]], six_fields: list[str], regime_fields: list[str]) -> dict[str, list[str]]:
    specialist = ["mv__" + name for name in views]
    base_fields = ["base_score", "p_clear", "p_adverse", "p_weak", "causal_base_map_bps", "map_prior_global_support", "map_prior_bin_support", "map_neutral_fallback"]
    regime = sorted(regime_fields)
    fields = list(dict.fromkeys(base_fields + specialist + six_fields + regime))
    # These are ledger inputs; specialist raw fields are joined separately.
    ledger_fields = [c for c in fields if c not in specialist]
    return {"all_fields": fields, "ledger_fields": ledger_fields, "specialist_fields": specialist, "regime_fields": regime, "selected_context_fields": six_fields}


def _attach_inputs(frame: pd.DataFrame, specialist_scores: dict[str, np.ndarray], views: dict[str, list[str]], store_fields: list[str], contract: dict[str, list[str]]) -> pd.DataFrame:
    out = frame.copy()
    for view in views:
        out["mv__" + view] = specialist_scores[view]
    if store_fields:
        out = out.merge(_store_rows(out, store_fields), on="candidate_id", validate="one_to_one")
    # Freeze and validate the exact residual field contract.  Missing values
    # are allowed for a field, but a missing column is a contract failure.
    for field in contract["all_fields"]:
        if field not in out:
            raise ValueError(f"residual contract field missing: {field}")
    return out


def _quantile_edges(values: np.ndarray) -> np.ndarray:
    finite = values[np.isfinite(values)]
    if len(finite) < 20:
        return np.array([-np.inf, np.inf], dtype=float)
    edges = np.unique(np.quantile(finite, [0.0, .2, .4, .6, .8, 1.0]))
    if len(edges) < 2:
        return np.array([-np.inf, np.inf], dtype=float)
    edges[0], edges[-1] = -np.inf, np.inf
    return edges.astype(float)


def _region_index(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.clip(np.searchsorted(edges, values, side="right") - 1, 0, len(edges) - 2).astype(np.int16)


def _choose_gate(cal_val: pd.DataFrame, pred_val: np.ndarray, *, side: str, edges: np.ndarray) -> tuple[float, np.ndarray, list[dict[str, object]]]:
    base = cal_val.causal_base_map_bps.to_numpy(float)
    residual = cal_val.net_bps.to_numpy(float) - base
    region = _region_index(base, edges)
    finite = np.isfinite(base) & np.isfinite(residual) & np.isfinite(pred_val)
    def top_net(values: np.ndarray, net: np.ndarray, fraction: float = .10) -> float:
        good = np.isfinite(values) & np.isfinite(net)
        if int(good.sum()) < 20:
            return -np.inf
        count = max(1, int(np.ceil(float(good.sum()) * fraction)))
        order = np.argsort(-values[good], kind="stable")[:count]
        return float(np.mean(net[good][order]))

    noop_top10 = top_net(base, cal_val.net_bps.to_numpy(float))
    best_lambda = 0.0
    best_economic = noop_top10
    best_mse = float(np.mean(residual[finite] ** 2)) if finite.any() else np.inf
    lambda_rows: list[dict[str, object]] = []
    for lam in LAMBDAS:
        mse = float(np.mean((residual[finite] - float(lam) * pred_val[finite]) ** 2)) if finite.any() else np.inf
        economic = top_net(base + float(lam) * pred_val, cal_val.net_bps.to_numpy(float))
        lambda_rows.append({"side": side, "lambda": float(lam), "rows": int(finite.sum()), "mse": mse, "economic_top10_net_bps": economic, "noop_top10_net_bps": noop_top10, "beats_noop": bool(economic > noop_top10)})
        if economic > best_economic + 1e-9 or (np.isclose(economic, best_economic) and mse < best_mse):
            best_economic = economic
            best_mse = mse
            best_lambda = float(lam)
    # Region gates are independently compared with the no-op *ranking* in the
    # region. If no region beats the no-op top-tail economics, correction is
    # disabled there even when its residual MSE improves.
    gates = np.zeros(len(edges) - 1, dtype=bool)
    audit = lambda_rows
    for idx in range(len(gates)):
        mask = finite & (region == idx)
        base_mse = float(np.mean(residual[mask] ** 2)) if mask.any() else np.inf
        gated_mse = float(np.mean((residual[mask] - best_lambda * pred_val[mask]) ** 2)) if mask.any() else np.inf
        region_noop = top_net(base[mask], cal_val.net_bps.to_numpy(float)[mask], .20)
        region_gated = top_net(base[mask] + best_lambda * pred_val[mask], cal_val.net_bps.to_numpy(float)[mask], .20)
        gates[idx] = bool(best_lambda > 0.0 and int(mask.sum()) >= 200 and region_gated > region_noop)
        audit.append({"side": side, "region": int(idx), "lower": float(edges[idx]), "upper": float(edges[idx + 1]), "rows": int(mask.sum()), "lambda": float(best_lambda), "mse": gated_mse, "baseline_mse": base_mse, "economic_top20_net_bps": region_gated, "noop_top20_net_bps": region_noop, "beats_noop": bool(gates[idx]), "gate_on": bool(gates[idx])})
    if not gates.any():
        best_lambda = 0.0
    return best_lambda, gates, audit


def _metrics(frame: pd.DataFrame, score_col: str, system: str, fold: str, period_kind: str = "fold") -> list[dict[str, object]]:
    z = frame.copy()
    z["month"] = pd.to_datetime(z.__ts__, utc=True).dt.strftime("%Y-%m")
    periods = [(fold, z)] if period_kind == "fold" else [(str(m), q) for m, q in z.groupby("month", sort=True)]
    rows: list[dict[str, object]] = []
    for period, q in periods:
        for side, sub in [("pooled", q), *[(s, q[q.side_name.eq(s)]) for s in ("long", "short")]]:
            if len(sub) == 0:
                continue
            for tail in TAILS:
                n = max(1, int(np.ceil(len(sub) * tail)))
                top = sub.sort_values([score_col, "candidate_id"], ascending=[False, True], kind="stable").head(n)
                rows.append({"system": system, "fold": fold, "period": period, "side": side, "tail": tail, "rows": len(sub), "trades": n, "net_bps": float(top.net_bps.mean()), "gross_bps": float(top.gross_bps.mean()), "rank_ic": float(sub[score_col].rank().corr(sub.net_bps.rank()))})
    return rows


def run(out: Path = OUT, frozen_artifact: Path | None = None) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    frozen_artifact = frozen_artifact or ROOT / "data_perp/artifacts/frozen_multiview_specialist_input_ablation_20260805_v1"
    folds = LONG_HISTORY_FOLDS[3:]
    base = _base()
    available = _schema()
    map_frame, map_audit, map_provenance = _base_map(base)
    base = base.merge(map_frame, on=["candidate_id", "__ts__", "side_name"], validate="one_to_one")
    map_audit.to_parquet(out / "base_map_audit.parquet", index=False)
    _write_json(out / "base_map_manifest.json", map_provenance)
    views_by_side = {side: _load_frozen_views(frozen_artifact, side) for side in ("long", "short")}
    template_probe = base.iloc[: min(30_000, len(base))].copy().merge(_store_rows(base.iloc[: min(30_000, len(base))], available), on="candidate_id", validate="one_to_one")
    six_fields = _select_six_context_fields(template_probe, set(available))
    regime_fields = _select_regime_context_fields(template_probe, set(available))
    contract = _fixed_residual_contract(base, views_by_side["long"], six_fields, regime_fields)
    contract["contract_sha256"] = hashlib.sha256("\n".join(contract["all_fields"]).encode()).hexdigest()
    contract["specialist_contract_source"] = str(frozen_artifact / "frozen_view_contract.json")
    _write_json(out / "residual_feature_contract.json", contract)
    metrics: list[dict[str, object]] = []
    predictions: list[pd.DataFrame] = []
    gate_rows: list[dict[str, object]] = []
    for fold in folds:
        a, b, c, e = map(_utc, (fold.train_start, fold.calibration_start, fold.test_start, fold.test_end))
        tr = base[base.__ts__.between(a, b, inclusive="left") & base.label_available_ts.lt(b)]
        ca = base[base.__ts__.between(b, c, inclusive="left") & base.label_available_ts.lt(c)]
        te = base[base.__ts__.between(c, e, inclusive="left")]
        fold_val: dict[str, tuple[pd.DataFrame, np.ndarray, np.ndarray]] = {}
        fold_test: dict[str, tuple[pd.DataFrame, np.ndarray]] = {}
        for side in ("long", "short"):
            train, cal, test = (x[x.side_name.eq(side)].copy() for x in (tr, ca, te))
            _, cal_scores, test_scores = _fit_specialists(train, cal, test, views_by_side[side])
            # Join raw selected context fields only; regime/context fields are
            # already frozen ledger columns in the contract.
            store_context = list(dict.fromkeys(six_fields + regime_fields))
            calx = _attach_inputs(cal, cal_scores, views_by_side[side], store_context, contract)
            testx = _attach_inputs(test, test_scores, views_by_side[side], store_context, contract)
            calx = calx.sort_values(["__ts__", "candidate_id"], kind="stable").reset_index(drop=True)
            split = max(1, int(len(calx) * .60))
            fit = calx.iloc[:split].copy()
            val = calx.iloc[split:].copy()
            target_fit = fit.net_bps.to_numpy(float) - fit.causal_base_map_bps.to_numpy(float)
            fields = contract["all_fields"]
            model_fit, usable = _ranker(pd.concat([fit[["__ts__"]], fit[fields]], axis=1), _rank_target(target_fit))
            raw_val = model_fit.predict(val[usable].fillna(0.0))
            residual_val = val.net_bps.to_numpy(float) - val.causal_base_map_bps.to_numpy(float)
            calibrator = fit_residual_calibration(raw_val, residual_val)
            pred_val = calibrator.predict(raw_val)
            edges = _quantile_edges(fit.causal_base_map_bps.to_numpy(float))
            lam, gates, audit = _choose_gate(val, pred_val, side=side, edges=edges)
            for row in audit:
                row.update({"fold": fold.name, "side": side, "contract_sha256": contract["contract_sha256"]})
            gate_rows.extend(audit)
            # Fit the final residual model on all calibration rows with the
            # frozen contract; lambda/gates remain selected on OOF validation.
            final_model, final_usable = _ranker(pd.concat([calx[["__ts__"]], calx[fields]], axis=1), _rank_target(calx.net_bps.to_numpy(float) - calx.causal_base_map_bps.to_numpy(float)))
            raw_test = final_model.predict(testx[final_usable].fillna(0.0))
            # The validation calibrator is strictly OOF relative to its fit
            # model and is reused unchanged for the untouched test period.
            residual_test = calibrator.predict(raw_test)
            test_region = _region_index(testx.causal_base_map_bps.to_numpy(float), edges)
            gate_mask = (test_region < len(gates)) & gates[np.clip(test_region, 0, len(gates) - 1)]
            final_score = testx.causal_base_map_bps.to_numpy(float) + np.where(gate_mask, lam * residual_test, 0.0)
            out_test = testx[["candidate_id", "__ts__", "side_name", "net_bps", "gross_bps", "base_score", "causal_base_map_bps"]].copy()
            out_test["score"] = final_score
            out_test["no_op_score"] = out_test.causal_base_map_bps
            out_test["residual_prediction_bps"] = residual_test
            out_test["residual_applied"] = gate_mask
            out_test["lambda"] = lam
            out_test["fold"] = fold.name
            predictions.append(out_test)
            fold_test[side] = (out_test, final_score)
            fold_val[side] = (val, pred_val, edges)
            del model_fit, final_model, calibrator
        combined = pd.concat([x[0] for x in fold_test.values()], ignore_index=True)
        metrics.extend(_metrics(combined, "score", "gated_prior_mapped_residual", fold.name, "fold"))
        metrics.extend(_metrics(combined, "score", "gated_prior_mapped_residual", fold.name, "month"))
        metrics.extend(_metrics(combined, "no_op_score", "no_op_prior_mapped_base", fold.name, "fold"))
        metrics.extend(_metrics(combined, "no_op_score", "no_op_prior_mapped_base", fold.name, "month"))
        pd.DataFrame(metrics).to_parquet(out / "metrics.checkpoint.parquet", index=False)
        pd.concat(predictions, ignore_index=True).to_parquet(out / "predictions.checkpoint.parquet", index=False)
        pd.DataFrame(gate_rows).to_parquet(out / "gate_audit.checkpoint.parquet", index=False)
        _write_json(out / "progress.json", {"status": "running", "completed_fold": fold.name})
    metric_frame = pd.DataFrame(metrics)
    metric_frame.to_parquet(out / "metrics.parquet", index=False)
    pd.concat(predictions, ignore_index=True).to_parquet(out / "predictions.parquet", index=False)
    pd.DataFrame(gate_rows).to_parquet(out / "gate_audit.parquet", index=False)
    _write_json(out / "manifest.json", {"schema": "gated_prior_mapped_residual_v3", "folds": [f.name for f in folds], "base_map": map_provenance, "residual_target": "exact_net_bps - side_local_prior_resolved_causal_base_map_bps", "lambda_grid": list(LAMBDAS), "gate_rule": "OOF validation top-tail net improvement over no-op, by side and mapped-base score region; residual MSE is secondary", "no_op_rejection": "lambda=0 or gate off whenever validation top-tail economics do not beat no-op", "ranking": "pooled global after common expected-net-bps mapping", "residual_contract_sha256": contract["contract_sha256"], "regime_context_contract": "store-joined causal regime/transition/funding/OI/liquidity fields; fixed per run"})
    _write_json(out / "progress.json", {"status": "complete", "completed_folds": [f.name for f in folds]})
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--frozen-artifact", type=Path, default=None)
    args = parser.parse_args()
    print(run(args.out, args.frozen_artifact))
