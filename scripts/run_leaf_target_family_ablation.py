#!/usr/bin/env python3
"""Chronological OOF ablation of posterior-only leaf targets.

The runner keeps the 0.70 posterior representation contract fixed.  It tests
five discovery targets at four requested horizons, evaluates every target
family separately, then forms a 20-feature combined arm only from families
which improve either pooled global inner top-1% or top-5% net EV.  No outer
test rows determine a family or feature admission decision.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import log_loss

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.performance_regimes.correctness_leaf_targets import TARGET_FAMILIES
from extreme_price_movements.config import CFG
from scripts.run_correctness_leaf_regime_oof import _represent, _rules, _screen, _target
from scripts.run_two_year_leaf_regime_top20_meta import _folds, _matrix, _model, _ord, _score_metrics


INPUT = ROOT / "data_perp/artifacts/correctness_leaf_regime_two_year_input_20260803_v2/input.parquet"
AVAILABILITY = ROOT / "data_perp/artifacts/correctness_leaf_regime_two_year_input_20260803_v2/feature_availability.parquet"
OUT = ROOT / "data_perp/artifacts/leaf_target_family_ablation_posterior070_20260803_v1"
HORIZONS = {"row": None, "period12h": 12, "period24h": 24, "period72h": 72}
POSTERIOR_SUFFIXES = ("__G0_geometric", "__G1_weighted_geometric", "__G2_generalized_pminus2", "__G3_softmin")
META_EXCLUDE = {
    "candidate_id", "__ts__", "__symbol__", "decision_ts", "label_available_ts", "side_name", "era",
    "gross_bps", "net_bps", "prequential_base_expected_net_bps",
}
ENTROPY_DIRECT_FIELDS = {"r3_p_adverse", "r3_p_weak", "r3_p_clear", "base_entropy", "base_top2_margin", "base_max_probability"}
DISCOVERY_POOL_SIZE = 40
MDA_CANDIDATE_CAP = 64
BASE_META_PROBABILITY_FIELDS = ("r3_p_adverse", "r3_p_weak", "r3_p_clear", "r3_opportunity_score", "base_entropy", "base_top2_margin", "base_max_probability")


def _diverse_discovery_pool(raw: list[str]) -> list[str]:
    """A fixed, configured, diverse causal pool for leaf discovery.

    This deliberately does not rescan the entire feature store for each of
    twenty target/horizon combinations.  It draws from the pre-existing meta
    contract: residual, ordinary meta context, market-regime context and
    model-trust/uncertainty fields.  Target-specific ``_screen`` below still
    ranks this full pool using only the discovery segment.
    """
    available = set(raw)
    groups = (
        ("RESIDUAL_META_FEATURE_KEYS", 10),
        ("meta_shared_feature_keys", 12),
        ("MODEL_REGIME_CONTEXT_META_FEATURE_KEYS", 10),
        ("META_MODEL_UNCERTAINTY_FEATURE_KEYS", 8),
    )
    selected: list[str] = []
    for group, quota in groups:
        for field in CFG.get(group, []):
            if field in available and field not in selected:
                selected.append(field)
                if sum(item in set(CFG.get(group, [])) for item in selected) >= quota:
                    break
    # Some historically materialised inputs predate parts of the modern
    # contract.  Fill only with remaining configured meta fields, never an
    # unqualified store-wide fallback.
    configured = [*CFG.get("meta_shared_feature_keys", []), *CFG.get("RESIDUAL_META_FEATURE_KEYS", []), *CFG.get("MODEL_REGIME_CONTEXT_META_FEATURE_KEYS", []), *CFG.get("META_MODEL_UNCERTAINTY_FEATURE_KEYS", [])]
    for field in configured:
        if field in available and field not in selected:
            selected.append(field)
            if len(selected) >= DISCOVERY_POOL_SIZE:
                break
    if len(selected) < DISCOVERY_POOL_SIZE:
        raise ValueError(f"need {DISCOVERY_POOL_SIZE} usable configured causal discovery fields; found {len(selected)}")
    return selected[:DISCOVERY_POOL_SIZE]


def _tail(frame: pd.DataFrame, score: np.ndarray, fraction: float) -> float:
    ordered = frame.assign(__score__=score).sort_values(["__score__", "candidate_id"], ascending=[False, True], kind="stable")
    return float(ordered.head(max(1, int(np.ceil(len(ordered) * fraction)))).net_bps.mean())


def _select_family(it: pd.DataFrame, iv: pd.DataFrame, raw: list[str], candidates: list[str], rng: np.random.Generator) -> tuple[list[str], list[dict]]:
    """Nested phantom-MDA selection, separately for each target family."""
    if not candidates:
        return [], []
    # Inner-train-only breadth cap before the costly permutation stage.  This
    # is a deterministic *candidate* cap (64 > final 20), not an outer-test
    # feature choice; it keeps the five-by-four target comparison tractable.
    if len(candidates) > MDA_CANDIDATE_CAP:
        outcome = it.net_bps.to_numpy(float) - it.prequential_base_expected_net_bps.to_numpy(float)
        ranked = []
        for field in candidates:
            value = pd.to_numeric(it[field], errors="coerce").to_numpy(float)
            valid = np.isfinite(value) & np.isfinite(outcome)
            score = spearmanr(value[valid], outcome[valid]).statistic if valid.sum() >= 200 else np.nan
            ranked.append((abs(float(score)) if np.isfinite(score) else -np.inf, field))
        candidates = [field for _, field in sorted(ranked, key=lambda item: (-item[0], item[1]))[:MDA_CANDIDATE_CAP]]
    phantoms = []
    it = it.copy()
    iv = iv.copy()
    for index in range(20):
        name = f"__phantom_{index:02d}"
        it[name] = rng.normal(size=len(it))
        iv[name] = rng.normal(size=len(iv))
        phantoms.append(name)
    fields = list(dict.fromkeys([*raw, *candidates, *phantoms]))
    p, _, model, median = _model(it, iv, fields)
    y = _ord(iv.net_bps.to_numpy(float) - iv.prequential_base_expected_net_bps.to_numpy(float))
    base_loss = log_loss(y, p, labels=[0, 1, 2])
    values = iv[fields].replace([np.inf, -np.inf], np.nan).fillna(median).to_numpy("float32")
    position = {name: index for index, name in enumerate(fields)}
    mda: dict[str, float] = {}
    for field in [*candidates, *phantoms]:
        index = position[field]
        original = values[:, index].copy()
        values[:, index] = rng.permutation(original)
        pp = np.clip(model.predict_proba(values), 1e-6, 1.0)
        pp /= pp.sum(axis=1, keepdims=True)
        mda[field] = float(log_loss(y, pp, labels=[0, 1, 2]) - base_loss)
        values[:, index] = original
    threshold = float(np.quantile([mda[field] for field in phantoms], .95))
    ordered = [field for field in sorted(candidates, key=lambda field: -mda[field]) if mda[field] > threshold]
    selected: list[str] = []
    audit: list[dict] = []
    for rank, field in enumerate(ordered, start=1):
        corr = max((abs(float(pd.Series(it[field]).corr(pd.Series(it[chosen])))) for chosen in selected), default=0.0)
        reason = "selected"
        if len(selected) >= 20:
            reason = "slot_cap"
        elif np.isfinite(corr) and corr > .80:
            reason = "activation_correlation_above_080"
        else:
            selected.append(field)
        audit.append({"feature": field, "mda_logloss": mda[field], "phantom_q95": threshold, "mda_excess_over_phantom": mda[field] - threshold, "max_activation_correlation": corr, "candidate_rank_by_mda": rank, "accepted": reason == "selected", "rejection_reason": reason})
    for field in candidates:
        if field not in ordered:
            audit.append({"feature": field, "mda_logloss": mda[field], "phantom_q95": threshold, "mda_excess_over_phantom": mda[field] - threshold, "max_activation_correlation": np.nan, "candidate_rank_by_mda": np.nan, "accepted": False, "rejection_reason": "mda_at_or_below_phantom_q95"})
    return selected, audit


def _combine_cap(it: pd.DataFrame, chosen: dict[str, list[str]], feature_mda: dict[str, float], families=TARGET_FAMILIES) -> list[str]:
    """Apply the original 20-feature and 0.80-correlation limits to admitted families."""
    selected: list[str] = []
    fields = [field for family in families for field in chosen.get(family, [])]
    for field in sorted(fields, key=lambda item: -feature_mda.get(item, -np.inf)):
        correlation = max((abs(float(pd.Series(it[field]).corr(pd.Series(it[other])))) for other in selected), default=0.0)
        if len(selected) < 20 and (not np.isfinite(correlation) or correlation <= .80):
            selected.append(field)
    return selected


def _prediction(frame: pd.DataFrame, train: pd.DataFrame, raw: list[str], fields: list[str]) -> np.ndarray:
    base, means, _, _ = _model(train, frame, raw)
    if not fields:
        adjusted = base
    else:
        probability, _, _, _ = _model(train, frame, [*raw, *fields])
        adjusted = probability
    return frame.prequential_base_expected_net_bps.to_numpy(float) + adjusted @ means


def run(out: Path = OUT, input_path: Path = INPUT, availability_path: Path = AVAILABILITY, *, end_ts: str | None = "2026-04-01", history_days: float | None = None, similarity_threshold: float = .70, families=TARGET_FAMILIES, transition_sidecar: Path | None = None, transition_to_correctness: bool = False, transition_to_meta: bool = False) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps({"status": "RUNNING"}, indent=2) + "\n")
    availability = pd.read_parquet(availability_path)
    store_raw = [name for name in availability.loc[availability.usable_90pct_nonconstant, "feature"].astype(str) if name not in META_EXCLUDE]
    discovery_pool = _diverse_discovery_pool(store_raw)
    # The conversion meta learner receives only a proper layer-specific
    # subset, rather than every physically present store field.  Keep the
    # frozen base simplex/margins alongside the diverse causal context pool.
    raw = list(dict.fromkeys([*discovery_pool, *(field for field in BASE_META_PROBABILITY_FIELDS if field in store_raw)]))
    required = ["candidate_id", "__ts__", "label_available_ts", "side_name", "era", "gross_bps", "net_bps", "prequential_base_expected_net_bps", "r3_p_adverse", "r3_p_weak", "r3_p_clear", *raw]
    data = pd.read_parquet(input_path, columns=list(dict.fromkeys(required)))
    data["__ts__"] = pd.to_datetime(data["__ts__"], utc=True)
    data["label_available_ts"] = pd.to_datetime(data["label_available_ts"], utc=True)
    transition_features: list[str] = []
    if transition_sidecar is not None:
        availability_file = transition_sidecar.parent / "feature_availability.parquet"
        transition_features = pd.read_parquet(availability_file).query("usable_90pct_nonconstant").feature.astype(str).tolist()
        timeline = pd.read_parquet(transition_sidecar, columns=["source_utc", *transition_features]).sort_values("source_utc")
        timeline["source_utc"] = pd.to_datetime(timeline.source_utc, utc=True)
        data = pd.merge_asof(data.sort_values("__ts__"), timeline, left_on="__ts__", right_on="source_utc", direction="backward", allow_exact_matches=True).drop(columns="source_utc")
        if transition_to_correctness:
            discovery_pool = [*discovery_pool, *transition_features]
        if transition_to_meta:
            raw = [*raw, *transition_features]
    if end_ts is not None:
        data = data[data.__ts__ < pd.Timestamp(end_ts, tz="UTC")].copy()
    data = data[np.isfinite(data.net_bps) & np.isfinite(data.prequential_base_expected_net_bps)].sort_values(["__ts__", "candidate_id"]).reset_index(drop=True)
    data["fold"] = _folds(data)
    metric_rows: list[dict] = []
    inner_rows: list[dict] = []
    selection_rows: list[dict] = []
    target_rows: list[dict] = []
    all_predictions: list[pd.DataFrame] = []
    for fold in (2, 3, 4):
        test = data[data.fold.eq(fold)].copy()
        start = test.__ts__.min()
        history = data[data.label_available_ts < start].copy()
        if history_days is not None:
            history = history[history.__ts__ >= start - pd.Timedelta(days=float(history_days))].copy()
        discovery_cut = history.__ts__.quantile(.60)
        discovery = history[history.__ts__ <= discovery_cut].copy()
        meta_train = history[history.__ts__ > discovery_cut].copy()
        inner_cut = meta_train.__ts__.quantile(.50)
        side_records: list[dict] = []
        for side in ("long", "short"):
            disc = discovery[discovery.side_name.eq(side)].copy()
            mt0 = meta_train[meta_train.side_name.eq(side)].copy()
            te0 = test[test.side_name.eq(side)].copy()
            if min(len(disc), len(mt0), len(te0)) < 300:
                continue
            combined = pd.concat([mt0, te0], ignore_index=True)
            candidates_by_family: dict[str, list[str]] = {family: [] for family in families}
            for family in families:
                for horizon_name, horizon in HORIZONS.items():
                    target_name = f"{family}__{horizon_name}"
                    labelled = _target(pd.concat([disc, mt0, te0], ignore_index=True), disc, horizon, family)
                    target_train = labelled.iloc[:len(disc)].copy()
                    target_train = target_train[np.isfinite(target_train.target_value)].copy()
                    discovery_fields = [field for field in discovery_pool if not (family == "entropy" and field in ENTROPY_DIRECT_FIELDS)]
                    chosen = _screen(target_train, discovery_fields, target_train.target_value.to_numpy(float))
                    median = target_train[chosen].median().fillna(0.0)
                    iqr = (target_train[chosen].quantile(.75) - target_train[chosen].quantile(.25)).replace(0, 1).fillna(1.0)
                    x = ((target_train[chosen].fillna(median) - median) / iqr).clip(-8, 8).to_numpy("float32")
                    import lightgbm as lgb
                    model = lgb.LGBMRegressor(objective="regression_l2", n_estimators=80, learning_rate=.04, num_leaves=16, max_depth=4, min_child_samples=max(80, int(.01 * len(target_train))), colsample_bytree=.8, reg_lambda=20.0, random_state=20260803 + fold, n_jobs=1, verbosity=-1).fit(x, target_train.target_value.to_numpy(float))
                    reference = combined.iloc[:len(mt0)].copy()
                    reference.loc[:, chosen] = ((reference[chosen].fillna(median) - median) / iqr).clip(-8, 8)
                    normalized = combined.copy()
                    normalized.loc[:, chosen] = ((normalized[chosen].fillna(median) - median) / iqr).clip(-8, 8)
                    rules, memberships = _rules(model, chosen, reference, 0.0)
                    representation, _, _, output_fields, _ = _represent(normalized, rules, memberships, side, target_name, fold, minimum_similarity=similarity_threshold)
                    output_fields = [field for field in output_fields if field.endswith(POSTERIOR_SUFFIXES)]
                    for field in output_fields:
                        combined[field] = representation[field].to_numpy(float)
                    candidates_by_family[family].extend(output_fields)
                    values = labelled.iloc[len(disc):len(disc) + len(mt0)].target_value.dropna()
                    target_rows.append({"fold": fold, "side_name": side, "target_family": family, "horizon": horizon_name, "discovery_rows": len(target_train), "selection_rows": len(values), "target_mean": float(values.mean()) if len(values) else np.nan, "target_std": float(values.std()) if len(values) else np.nan, "target_iqr": float(values.quantile(.75) - values.quantile(.25)) if len(values) else np.nan, "representation_candidates": len(output_fields)})
            mt = combined.iloc[:len(mt0)].copy()
            te = combined.iloc[len(mt0):].copy()
            it = mt[mt.__ts__ <= inner_cut].copy()
            iv = mt[mt.__ts__ > inner_cut].copy()
            if min(len(it), len(iv)) < 200:
                continue
            selected_by_family: dict[str, list[str]] = {}
            mda_by_feature: dict[str, float] = {}
            for index, family in enumerate(families):
                candidates = [field for field in candidates_by_family[family] if field in mt and mt[field].notna().any() and mt[field].nunique(dropna=True) > 1]
                selected, audit = _select_family(it, iv, raw, candidates, np.random.default_rng(20260803 + fold * 101 + index + (0 if side == "long" else 1000)))
                selected_by_family[family] = selected
                for row in audit:
                    row.update({"fold": fold, "side_name": side, "target_family": family})
                    mda_by_feature[row["feature"]] = row["mda_logloss"]
                    selection_rows.append(row)
            side_records.append({"side": side, "mt": mt, "te": te, "it": it, "iv": iv, "selected": selected_by_family, "mda": mda_by_feature})
        if len(side_records) != 2:
            continue
        inner = []
        for record in side_records:
            frame = record["iv"][["candidate_id", "__ts__", "side_name", "gross_bps", "net_bps", "prequential_base_expected_net_bps"]].copy()
            frame["baseline"] = _prediction(record["iv"], record["it"], raw, [])
            for family in families:
                frame[family] = _prediction(record["iv"], record["it"], raw, record["selected"][family])
            inner.append(frame)
        inner_frame = pd.concat(inner, ignore_index=True)
        admitted: list[str] = []
        baseline_inner = {fraction: _tail(inner_frame, inner_frame["baseline"].to_numpy(float), fraction) for fraction in (.01, .05)}
        for family in families:
            values = {fraction: _tail(inner_frame, inner_frame[family].to_numpy(float), fraction) for fraction in (.01, .05)}
            accept = values[.01] > baseline_inner[.01] or values[.05] > baseline_inner[.05]
            if accept:
                admitted.append(family)
            for fraction in (.01, .05):
                inner_rows.append({"fold": fold, "target_family": family, "top_fraction": fraction, "baseline_net_bps": baseline_inner[fraction], "family_net_bps": values[fraction], "delta_net_bps": values[fraction] - baseline_inner[fraction], "admitted_to_combination": accept})
        final = []
        for record in side_records:
            frame = record["te"][["candidate_id", "__ts__", "side_name", "gross_bps", "net_bps", "prequential_base_expected_net_bps"]].copy()
            frame["baseline_score_bps"] = _prediction(record["te"], record["mt"], raw, [])
            for family in families:
                frame[f"{family}_score_bps"] = _prediction(record["te"], record["mt"], raw, record["selected"][family])
            permitted = {family: record["selected"][family] for family in admitted}
            combined_fields = _combine_cap(record["it"], permitted, record["mda"], families)
            frame["combined_score_bps"] = _prediction(record["te"], record["mt"], raw, combined_fields)
            frame["fold"] = fold
            final.append(frame)
            selection_rows.extend({"fold": fold, "side_name": record["side"], "target_family": "combination", "feature": field, "mda_logloss": record["mda"].get(field, np.nan), "phantom_q95": np.nan, "mda_excess_over_phantom": np.nan, "max_activation_correlation": np.nan, "candidate_rank_by_mda": rank, "accepted": True, "rejection_reason": "inner_admitted_family_combination"} for rank, field in enumerate(combined_fields, start=1))
        outer = pd.concat(final, ignore_index=True)
        all_predictions.append(outer)
        metric_rows.extend(_score_metrics(outer, outer.baseline_score_bps.to_numpy(float), fold, "baseline_all_meta"))
        for family in families:
            metric_rows.extend(_score_metrics(outer, outer[f"{family}_score_bps"].to_numpy(float), fold, f"target_{family}"))
        metric_rows.extend(_score_metrics(outer, outer.combined_score_bps.to_numpy(float), fold, "combination_inner_admitted"))
    pd.DataFrame(metric_rows).to_parquet(out / "target_family_oof_comparison.parquet", index=False)
    pd.DataFrame(inner_rows).to_parquet(out / "inner_target_admission.parquet", index=False)
    pd.DataFrame(selection_rows).to_parquet(out / "target_family_selection.parquet", index=False)
    pd.DataFrame(target_rows).to_parquet(out / "target_family_label_audit.parquet", index=False)
    pd.concat(all_predictions, ignore_index=True).to_parquet(out / "oof_predictions.parquet", index=False)
    manifest = {"schema": "leaf_target_family_ablation_v3", "status": "COMPLETED", "input": str(input_path), "availability": str(availability_path), "end_ts_exclusive": end_ts, "history_policy": "expanding_all_prior_resolved_rows" if history_days is None else "rolling", "history_window_days": history_days, "similarity_threshold": similarity_threshold, "representation_contract": "posterior-only G0/G1/G2/G3; cluster similarity 0.70", "meta_feature_contract": {"count": len(raw), "fields": raw, "policy": "diverse configured causal pool plus frozen base probability/uncertainty fields"}, "discovery_feature_contract": {"count": len(discovery_pool), "fields": discovery_pool, "policy": "configured diverse causal meta pool; target-specific inner discovery ranking only"}, "mda_prefilter": {"candidate_cap": MDA_CANDIDATE_CAP, "policy": "inner-train residual rank-IC only; final phantom-MDA/correlation gate remains unchanged"}, "targets": {"correctness": "side-local clipped signed residual, .5 at zero", "positive": "soft residual membership: 0 through +50 bps, 1 at +75 bps", "negative": "soft residual membership: 0 through -50 bps, 1 at -75 bps", "entropy": "normalized Shannon entropy of frozen OOF R3 adverse/weak/clear simplex", "surprise": "1[absolute residual > 50 bps]"}, "horizons": HORIZONS, "label_availability": {"correctness": "realised label_available_ts", "positive": "realised label_available_ts", "negative": "realised label_available_ts", "surprise": "realised label_available_ts", "entropy": "decision timestamp; frozen OOF base simplex"}, "selection": "per-side, target-family inner rank-IC candidate cap 64, phantom-MDA q95 then correlation <=.80, maximum 20; combined arm admits a family only when pooled-global inner top1 or top5 net exceeds baseline and again caps at 20 per side", "ranking": "pooled global top-k after common-bps side class map"}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--availability", type=Path, default=AVAILABILITY)
    parser.add_argument("--end-ts", default="2026-04-01")
    parser.add_argument("--history-days", type=float, default=0.0, help="0 means all prior label-resolved history")
    parser.add_argument("--similarity-threshold", type=float, default=.70)
    args = parser.parse_args()
    print(run(args.out, args.input, args.availability, end_ts=args.end_ts, history_days=None if args.history_days <= 0 else args.history_days, similarity_threshold=args.similarity_threshold))
