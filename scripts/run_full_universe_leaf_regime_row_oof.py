#!/usr/bin/env python3
"""Full-universe, per-candidate walk-forward validation of leaf regimes.

Period aggregation is used *only* to discover a stable correctness leaf
dictionary.  Every conversion/meta observation, probability and score is a
candidate row; there is no per-timestamp cohort or outcome-stratified sample.
The final September--November block is kept as a separate untouched OOS block.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import log_loss

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_correctness_leaf_regime_oof import _represent, _rules, _screen, _target


LEDGER = ROOT / "data_perp/artifacts/tp6_m6_shared_regime_residual_features_20260809_v3/shared_regime_residual_ledger.parquet"
OUT = ROOT / "data_perp/artifacts/full_universe_leaf_regime_row_oof_20260804_v1"
TARGETS = {"row": None, "period12h": 12, "period24h": 24, "period72h": 72}
FOLDS = (
    ("oof_2024_05_06", "2024-05-01", "2024-07-01", "oof"),
    ("oof_2024_07_08", "2024-07-01", "2024-09-01", "oof"),
    ("oos_2024_09_11", "2024-09-01", "2024-12-01", "oos"),
)
EPS = 1e-7


def _ordinal(residual: np.ndarray) -> np.ndarray:
    """Per-row conversion target: overestimate / accurate / underestimate."""
    return np.where(residual <= -50.0, 0, np.where(residual >= 50.0, 2, 1)).astype("int8")


def _matrix(train: pd.DataFrame, test: pd.DataFrame, fields: list[str]) -> tuple[np.ndarray, np.ndarray]:
    median = train.loc[:, fields].replace([np.inf, -np.inf], np.nan).median().fillna(0.0)
    x = train.loc[:, fields].replace([np.inf, -np.inf], np.nan).fillna(median).to_numpy("float32")
    z = test.loc[:, fields].replace([np.inf, -np.inf], np.nan).fillna(median).to_numpy("float32")
    return x, z


def _fit_predict(train: pd.DataFrame, test: pd.DataFrame, fields: list[str], seed: int) -> tuple[np.ndarray, np.ndarray]:
    residual = train.net_bps.to_numpy(float) - train.prequential_base_expected_net_bps.to_numpy(float)
    y = _ordinal(residual)
    counts = np.bincount(y, minlength=3).astype(float)
    weight = np.sqrt(len(y) / np.maximum(3.0 * counts[y], 1.0))
    weight = np.clip(weight / weight.mean(), 0.5, 2.0)
    x, z = _matrix(train, test, fields)
    model = lgb.LGBMClassifier(
        objective="multiclass", num_class=3, n_estimators=140, learning_rate=.035,
        num_leaves=24, max_depth=5, min_child_samples=max(150, int(.01 * len(train))),
        colsample_bytree=.8, reg_lambda=20.0, random_state=seed, n_jobs=2, verbosity=-1,
    ).fit(x, y, sample_weight=weight)
    probability = np.zeros((len(test), 3), dtype=float)
    probability[:, model.classes_.astype(int)] = model.predict_proba(z)
    probability = np.clip(probability, EPS, 1.0)
    probability /= probability.sum(axis=1, keepdims=True)
    means = np.asarray([
        residual[y == label].mean() if np.any(y == label) else 0.0 for label in range(3)
    ], dtype=float)
    return probability, means


def _invariant_leaf(name: str) -> bool:
    """Keep only representations whose semantic meaning transports by design."""
    return (
        name.startswith("cluster_state_")
        or "__signed_contribution" in name
        or "__positive_contribution" in name
        or "__negative_contribution" in name
        or "__absolute_contribution" in name
        or "__total_contribution_share" in name
        or "__historical_" in name
        or "__structural_stability" in name
        or "__instability" in name
        or "__rule_count" in name
        or name.endswith("__G1_weighted_geometric")
        or any(token in name for token in ("__velocity_", "__acceleration", "__smoothed_membership", "__hours_active_", "__activation_"))
    )


def _select_leaf_features(later: pd.DataFrame, raw: list[str], candidates: list[str], *, seed: int) -> tuple[list[str], pd.DataFrame]:
    """Nested chronological, phantom-gated compact feature selection.

    A train-only rank screen reduces hundreds of related representations to 48;
    only those then face the inner held-out permutation test, so full-universe
    validation remains computationally tractable without using OOS labels.
    """
    times = pd.Index(later.__ts__.drop_duplicates().sort_values())
    cut = times[int(len(times) * .55)]
    inner_train = later[later.__ts__ < cut].copy()
    inner_valid = later[later.__ts__ >= cut].copy()
    residual = inner_train.net_bps.to_numpy(float) - inner_train.prequential_base_expected_net_bps.to_numpy(float)
    ranked = []
    for field in candidates:
        value = pd.to_numeric(inner_train[field], errors="coerce")
        valid = value.notna().to_numpy() & np.isfinite(residual)
        if valid.sum() < 500 or value[valid].nunique() < 2:
            continue
        corr = spearmanr(value.to_numpy(float)[valid], residual[valid]).statistic
        if np.isfinite(corr):
            ranked.append((abs(float(corr)), str(field)))
    screened = [field for _, field in sorted(ranked, key=lambda x: (-x[0], x[1]))[:48]]
    if not screened:
        return [], pd.DataFrame()
    rng = np.random.default_rng(seed)
    phantoms = [f"phantom_{index:02d}" for index in range(20)]
    train_aug, valid_aug = inner_train.copy(), inner_valid.copy()
    for index, phantom in enumerate(phantoms):
        source = screened[index % len(screened)]
        train_aug[phantom] = train_aug[source].to_numpy(float)[rng.permutation(len(train_aug))]
        valid_aug[phantom] = valid_aug[source].to_numpy(float)[rng.permutation(len(valid_aug))]
    fields = [*raw, *screened, *phantoms]
    probability, _means = _fit_predict(train_aug, valid_aug, fields, seed)
    y = _ordinal(valid_aug.net_bps.to_numpy(float) - valid_aug.prequential_base_expected_net_bps.to_numpy(float))
    baseline = log_loss(y, probability, labels=[0, 1, 2])
    x_train, x_valid = _matrix(train_aug, valid_aug, fields)
    residual_train = train_aug.net_bps.to_numpy(float) - train_aug.prequential_base_expected_net_bps.to_numpy(float)
    target = _ordinal(residual_train)
    counts = np.bincount(target, minlength=3).astype(float)
    weight = np.sqrt(len(target) / np.maximum(3.0 * counts[target], 1.0)); weight = np.clip(weight / weight.mean(), .5, 2.)
    model = lgb.LGBMClassifier(objective="multiclass", num_class=3, n_estimators=140, learning_rate=.035, num_leaves=24, max_depth=5, min_child_samples=max(150, int(.01 * len(train_aug))), colsample_bytree=.8, reg_lambda=20., random_state=seed, n_jobs=2, verbosity=-1).fit(x_train, target, sample_weight=weight)
    index = {name: i for i, name in enumerate(fields)}
    rows = []
    for field in [*screened, *phantoms]:
        position = index[field]; original = x_valid[:, position].copy(); x_valid[:, position] = rng.permutation(original)
        p = np.zeros((len(x_valid), 3), dtype=float); p[:, model.classes_.astype(int)] = model.predict_proba(x_valid); p = np.clip(p, EPS, 1.0); p /= p.sum(axis=1, keepdims=True)
        mda = float(log_loss(y, p, labels=[0, 1, 2]) - baseline)
        x_valid[:, position] = original
        rows.append({"feature": field, "is_phantom": field in phantoms, "mda_logloss": mda})
    audit = pd.DataFrame(rows)
    threshold = float(audit[audit.is_phantom].mda_logloss.quantile(.95))
    ordered = audit[~audit.is_phantom].query("mda_logloss > @threshold").sort_values(["mda_logloss", "feature"], ascending=[False, True])
    selected = []
    for field in ordered.feature.astype(str):
        correlation = max((abs(float(inner_train[field].corr(inner_train[kept], method="spearman"))) for kept in selected if np.isfinite(inner_train[field].corr(inner_train[kept], method="spearman"))), default=0.0)
        if correlation <= .80 and len(selected) < 20:
            selected.append(field)
    audit["phantom_q95"] = threshold
    audit["selected"] = audit.feature.isin(selected)
    return selected, audit


def _feature_fields(data: pd.DataFrame) -> list[str]:
    forbidden = {
        "candidate_id", "__ts__", "side_name", "gross_bps", "net_bps", "base_raw",
        "source", "event", "era", "m6_contract_complete", "shared_regime_contract_complete",
        "state_reference_cutoff_utc", "residual_reference_cutoff_utc", "label_available_ts",
        "target__exact_net_residual_bps", "target__soft_regime_centered_residual_bps",
        "target__soft_regime_standardized_residual",
    }
    output = []
    for name in data.columns:
        if name in forbidden or name.startswith("target__"):
            continue
        value = pd.to_numeric(data[name], errors="coerce")
        if value.notna().mean() >= .90 and value.nunique(dropna=True) > 1:
            output.append(name)
    return output


def _row_metrics(frame: pd.DataFrame, probability: np.ndarray, *, fold: str, split: str, side: str, arm: str) -> dict:
    residual = frame.net_bps.to_numpy(float) - frame.prequential_base_expected_net_bps.to_numpy(float)
    y = _ordinal(residual)
    onehot = np.eye(3)[y]
    expected = probability @ np.asarray([-100.0, 0.0, 100.0])
    return {
        "fold": fold, "split": split, "side_name": side, "arm": arm, "rows": len(frame),
        "target_logloss": float(log_loss(y, probability, labels=[0, 1, 2])),
        "target_brier": float(np.square(probability - onehot).sum(axis=1).mean() / 3.0),
        "row_residual_rank_ic": float(spearmanr(expected, residual).statistic),
        "row_net_rank_ic": float(spearmanr(frame.prequential_base_expected_net_bps.to_numpy(float) + expected, frame.net_bps.to_numpy(float)).statistic),
    }


def _tail_metrics(frame: pd.DataFrame, score: str, *, fold: str, split: str, arm: str) -> list[dict]:
    rows = []
    for fraction in (.01, .05, .10):
        n = max(1, int(np.ceil(len(frame) * fraction)))
        selected = frame.sort_values([score, "candidate_id"], ascending=[False, True], kind="stable").head(n)
        for side, part in list(selected.groupby("side_name", observed=True)) + [("pooled", selected)]:
            rows.append({
                "fold": fold, "split": split, "arm": arm, "selection": "pooled_global_top_k",
                "top_fraction": fraction, "side_name": side, "trades": len(part),
                "gross_bps_per_trade": float(part.gross_bps.mean()), "net_bps_per_trade": float(part.net_bps.mean()),
                "score_mean_bps": float(part[score].mean()),
            })
    return rows


def _build_leaf_surface(history: pd.DataFrame, test: pd.DataFrame, raw: list[str], *, side: str, fold_index: int) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[dict], list[pd.DataFrame]]:
    """Fit period-aware dictionaries on early resolved history; apply by row."""
    discovery_end = history.__ts__.quantile(.60)
    later = history[history.__ts__ > discovery_end].copy()
    if later.empty:
        raise ValueError("empty post-discovery meta history")
    combined = pd.concat([later, test], ignore_index=True)
    discovery = history[history.__ts__ <= discovery_end].copy()
    # This subset is also used to fit the correctness scale.  The old sampled
    # runner only filtered the eventual dictionary rows, which left the scale
    # itself able to see the final unresolved H12 observations of discovery.
    # Both the target normalizer and the fitted rules must be prior-resolved.
    resolved_discovery = discovery[discovery.label_available_ts < later.__ts__.min()].copy()
    candidates: list[str] = []
    audits: list[dict] = []
    rule_tables: list[pd.DataFrame] = []
    for target_name, horizon in TARGETS.items():
        # The aggregation itself may cover its declared period.  A discovery
        # row is legal only once that period's final H12 label is available
        # before the later-history model/application boundary.
        labelled = _target(pd.concat([history, test], ignore_index=True), resolved_discovery, horizon)
        dictionary = labelled.iloc[:len(history)].copy()
        dictionary = dictionary[(dictionary.__ts__ <= discovery_end) & (dictionary.target_available_ts < later.__ts__.min())]
        dictionary = dictionary[dictionary.side_name.eq(side) & np.isfinite(dictionary.target_value)].copy()
        local_later = later[later.side_name.eq(side)].copy()
        local_test = test[test.side_name.eq(side)].copy()
        if min(len(dictionary), len(local_later), len(local_test)) < 500:
            continue
        chosen = _screen(dictionary, raw, dictionary.target_value.to_numpy(float))
        median = dictionary.loc[:, chosen].median().fillna(0.0)
        scale = (dictionary.loc[:, chosen].quantile(.75) - dictionary.loc[:, chosen].quantile(.25)).replace(0.0, 1.0).fillna(1.0)
        x = ((dictionary.loc[:, chosen].fillna(median) - median) / scale).clip(-8, 8).to_numpy("float32")
        model = lgb.LGBMRegressor(
            objective="regression_l2", n_estimators=80, learning_rate=.04, num_leaves=16,
            max_depth=4, min_child_samples=max(100, int(.01 * len(dictionary))),
            colsample_bytree=.8, reg_lambda=20.0, random_state=20_260_804 + fold_index, n_jobs=2, verbosity=-1,
        ).fit(x, dictionary.target_value.to_numpy(float))
        reference = local_later.copy()
        reference.loc[:, chosen] = ((reference.loc[:, chosen].fillna(median) - median) / scale).clip(-8, 8)
        applied = pd.concat([local_later, local_test], ignore_index=True)
        applied.loc[:, chosen] = ((applied.loc[:, chosen].fillna(median) - median) / scale).clip(-8, 8)
        rules, memberships = _rules(model, chosen, reference, 0.0)
        represented, rule_rows, similarity, outputs, lineage = _represent(applied, rules, memberships, side, target_name, fold_index, minimum_similarity=.70)
        keep = [field for field in outputs if field in represented and represented[field].nunique(dropna=True) > 1]
        for field in keep:
            combined.loc[combined.side_name.eq(side), field] = represented[field].to_numpy("float32")
        candidates.extend(keep)
        audits.append({
            "side_name": side, "target": target_name, "horizon_hours": horizon or 0,
            "discovery_rows": len(dictionary), "later_rows": len(local_later), "test_rows": len(local_test),
            "raw_feature_count": len(chosen), "rule_count": len(rules), "representation_count": len(keep),
            "discovery_max_target_available_utc": dictionary.target_available_ts.max().isoformat(),
            "later_start_utc": later.__ts__.min().isoformat(),
            "strict_discovery_availability": bool(dictionary.target_available_ts.lt(later.__ts__.min()).all()),
        })
        if not rule_rows.empty:
            rule_tables.append(rule_rows.assign(target=target_name, fold=fold_index, side_name=side))
        if not lineage.empty:
            rule_tables.append(lineage.assign(target=target_name, fold=fold_index, side_name=side, _table="lineage"))
        if not similarity.empty:
            rule_tables.append(similarity.assign(target=target_name, fold=fold_index, side_name=side, _table="similarity"))
    return combined, test, list(dict.fromkeys(candidates)), audits, rule_tables


def run(out: Path = OUT, ledger: Path = LEDGER) -> Path:
    if out.exists():
        raise FileExistsError(f"refusing to overwrite immutable output {out}")
    out.mkdir(parents=True)
    (out / "run_manifest.json").write_text(json.dumps({"status": "RUNNING"}, indent=2) + "\n")
    data = pd.read_parquet(ledger)
    data["__ts__"] = pd.to_datetime(data["__ts__"], utc=True)
    data["label_available_ts"] = data["__ts__"] + pd.Timedelta(hours=12)
    data = data[data.shared_regime_contract_complete.fillna(False) & np.isfinite(data.net_bps) & np.isfinite(data.prequential_base_expected_net_bps)].copy()
    data = data.sort_values(["__ts__", "candidate_id"], kind="stable").reset_index(drop=True)
    raw = _feature_fields(data)
    predictions: list[pd.DataFrame] = []
    row_metrics: list[dict] = []
    tail_metrics: list[dict] = []
    feature_rows: list[dict] = []
    rules: list[pd.DataFrame] = []
    selection_audits: list[pd.DataFrame] = []
    for fold_index, (fold, start_s, end_s, split) in enumerate(FOLDS):
        start, end = pd.Timestamp(start_s, tz="UTC"), pd.Timestamp(end_s, tz="UTC")
        test_all = data[(data.__ts__ >= start) & (data.__ts__ < end)].copy()
        history_all = data[data.label_available_ts < start].copy()
        if test_all.empty or history_all.empty:
            raise ValueError(f"{fold}: missing history or test rows")
        pieces = []
        for side_index, side in enumerate(("long", "short")):
            history = history_all[history_all.side_name.eq(side)].copy()
            test = test_all[test_all.side_name.eq(side)].copy()
            surface, _original_test, leaf, audit, rule_tables = _build_leaf_surface(history, test, raw, side=side, fold_index=fold_index)
            rules.extend(rule_tables)
            later = surface.iloc[:len(history[history.__ts__ > history.__ts__.quantile(.60)])].copy()
            # _build_leaf_surface returns post-discovery history followed by test.
            n_later = int((history.__ts__ > history.__ts__.quantile(.60)).sum())
            later, scored = surface.iloc[:n_later].copy(), surface.iloc[n_later:].copy()
            leaf = [field for field in leaf if field in later and later[field].nunique(dropna=True) > 1]
            baseline_p, baseline_means = _fit_predict(later, scored, raw, 20_260_804 + 100 * fold_index + side_index)
            all_p, all_means = _fit_predict(later, scored, [*raw, *leaf], 20_260_904 + 100 * fold_index + side_index)
            selected_all, all_audit = _select_leaf_features(later, raw, leaf, seed=20_261_004 + 100 * fold_index + side_index)
            invariant = [field for field in leaf if _invariant_leaf(field)]
            selected_invariant, invariant_audit = _select_leaf_features(later, raw, invariant, seed=20_261_104 + 100 * fold_index + side_index)
            selected_p, selected_means = _fit_predict(later, scored, [*raw, *selected_all], 20_261_204 + 100 * fold_index + side_index)
            invariant_p, invariant_means = _fit_predict(later, scored, [*raw, *selected_invariant], 20_261_304 + 100 * fold_index + side_index)
            selection_audits.extend([
                all_audit.assign(fold=fold, split=split, side_name=side, arm="selected_all_leaf"),
                invariant_audit.assign(fold=fold, split=split, side_name=side, arm="selected_invariant_leaf"),
            ])
            output = scored.loc[:, ["candidate_id", "__ts__", "side_name", "gross_bps", "net_bps", "prequential_base_expected_net_bps"]].copy()
            output["fold"], output["split"] = fold, split
            output["baseline_score_bps"] = output.prequential_base_expected_net_bps.to_numpy(float) + baseline_p @ baseline_means
            output["all_leaf_regime_score_bps"] = output.prequential_base_expected_net_bps.to_numpy(float) + all_p @ all_means
            output["selected_leaf_score_bps"] = output.prequential_base_expected_net_bps.to_numpy(float) + selected_p @ selected_means
            output["selected_invariant_leaf_score_bps"] = output.prequential_base_expected_net_bps.to_numpy(float) + invariant_p @ invariant_means
            for label, probability in (("baseline_raw_context", baseline_p), ("all_leaf_regime_representations", all_p), ("selected_all_leaf", selected_p), ("selected_invariant_leaf", invariant_p)):
                row_metrics.append(_row_metrics(scored, probability, fold=fold, split=split, side=side, arm=label))
            feature_rows.extend([{"fold": fold, "split": split, "side_name": side, "feature": field, "kind": "raw_context", "used_in_all_representation_arm": True} for field in raw])
            feature_rows.extend([{"fold": fold, "split": split, "side_name": side, "feature": field, "kind": "leaf_representation", "used_in_all_representation_arm": True} for field in leaf])
            feature_rows.extend([{"fold": fold, "split": split, "side_name": side, "feature": field, "kind": "selected_leaf", "used_in_all_representation_arm": True} for field in selected_all])
            feature_rows.extend([{"fold": fold, "split": split, "side_name": side, "feature": field, "kind": "selected_invariant_leaf", "used_in_all_representation_arm": True} for field in selected_invariant])
            feature_rows.extend([{**row, "fold": fold, "split": split} for row in audit])
            pieces.append(output)
        combined = pd.concat(pieces, ignore_index=True)
        predictions.append(combined)
        tail_metrics.extend(_tail_metrics(combined, "baseline_score_bps", fold=fold, split=split, arm="baseline_raw_context"))
        tail_metrics.extend(_tail_metrics(combined, "all_leaf_regime_score_bps", fold=fold, split=split, arm="all_leaf_regime_representations"))
        tail_metrics.extend(_tail_metrics(combined, "selected_leaf_score_bps", fold=fold, split=split, arm="selected_all_leaf"))
        tail_metrics.extend(_tail_metrics(combined, "selected_invariant_leaf_score_bps", fold=fold, split=split, arm="selected_invariant_leaf"))
    pd.concat(predictions, ignore_index=True).to_parquet(out / "row_level_oof_oos_predictions.parquet", index=False)
    pd.DataFrame(row_metrics).to_parquet(out / "per_row_prediction_metrics.parquet", index=False)
    pd.DataFrame(tail_metrics).to_parquet(out / "pooled_global_tail_metrics.parquet", index=False)
    pd.DataFrame(feature_rows).to_parquet(out / "feature_and_discovery_audit.parquet", index=False)
    pd.concat(selection_audits, ignore_index=True).to_parquet(out / "fold_local_selection_audit.parquet", index=False)
    pd.concat(rules, ignore_index=True).to_parquet(out / "leaf_rule_and_lineage_audit.parquet", index=False) if rules else pd.DataFrame().to_parquet(out / "leaf_rule_and_lineage_audit.parquet", index=False)
    manifest = {
        "status": "COMPLETED", "input": str(ledger), "rows_after_complete_contract": len(data),
        "period_aggregation": "used only to fit row/12h/24h/72h leaf dictionaries; each period label must resolve before later history begins",
        "meta_assessment": "every full-universe candidate row; no per-timestamp selection, no outcome-stratified selector sample",
        "meta_target": "three-class per-row exact-net residual: <=-50 bps overestimate, (-50,+50) accurate, >=+50 bps underestimate",
        "leaf_feature_policy": "compare all materialised outputs with two compact, fold-local phantom-gated selections: all leaf outputs and invariant summaries only",
        "strictness": "history labels available strictly before outer fold start; leaf-dictionary period labels available strictly before later-history boundary",
        "folds": [{"name": n, "start": s, "end_exclusive": e, "kind": k} for n, s, e, k in FOLDS],
        "ranking": "scores are mapped to side-local training residual bps, then ranked pooled globally for tail summaries",
        "raw_causal_context_feature_count": len(raw),
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return out


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--ledger", type=Path, default=LEDGER)
    args = parser.parse_args()
    print(run(args.out, args.ledger))
