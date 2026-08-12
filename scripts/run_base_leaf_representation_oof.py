#!/usr/bin/env python3
"""Run the base-native, leak-safe leaf-representation ablation.

This is a deliberately sequential comparison, not an HPO sweep:

1. Earlier resolved R3 rows build a frozen rule dictionary.
2. Later history selects only target-free soft memberships with two internal
   chronological phantom-MDA checks.
3. A control and an augmented R3 base are fit on that same later history.
4. The experimental simplex is handed directly to the same per-row,
   quantile-residual meta construction and only then mapped to side-local
   common bps and pooled globally.

No leaf is a raw fitted tree ID, and no base tree can see residual/meta/trust
features, expected-net values, or realised economics beyond the R3 class.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.base_leaf_representations import (
    BaseLeafConfig,
    BaseLeafRepresentationError,
    cap_support_diverse,
    fit_dictionary,
    strict_dictionary_split,
    support_bucket,
    target_values,
)
from extreme_price_movements.prequential_r3_value_map import (
    PrequentialR3ValueMapConfig,
    base_oof_trust_features,
    prequential_same_side_r3_value_map,
)
from extreme_price_movements.stage_i_causal_admission import (
    Causal21dAdmissionSpec,
    apply_causal_21d_side_admission,
    pooled_global_admission_comparison,
)
from extreme_price_movements.stage_i_nested_lgbm_hooks import (
    FixedLGBMContract,
    fixed_lgbm_meta_predictor,
    fold_local_meta_feature_selector,
)
from extreme_price_movements.stage_i_ranking import (
    RANKING_POLICY,
    stable_stage_i_topk_positions,
)


IDENTITY = ("candidate_id", "__ts__", "__symbol__")
TOP_FRACTIONS = (.01, .05, .10)
TARGETS: dict[str, int | None] = {"row": None, "period12h": 12, "period24h": 24}
BASE_DIRECT = (
    "r3_p_adverse", "r3_p_weak", "r3_p_clear", "r3_opportunity_score",
    "base_r3_max_probability", "base_r3_top2_margin", "base_r3_entropy",
)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _valid_rows(ledger: pd.DataFrame) -> np.ndarray:
    label = ledger.get("label_valid", pd.Series(False, index=ledger.index)).fillna(False).astype(bool)
    net = pd.to_numeric(ledger.get("exact_net_bps"), errors="coerce")
    return (label & np.isfinite(net)).to_numpy(bool)


def _strict_blocks(frame: pd.DataFrame, *, minimum_history_rows: int, count: int) -> list[np.ndarray]:
    order = np.argsort(frame.decision_ts.to_numpy(dtype="datetime64[ns]"), kind="stable")
    decision = frame.decision_ts.to_numpy(dtype="datetime64[ns]")[order]
    starts = np.r_[0, np.flatnonzero(decision[1:] != decision[:-1]) + 1]
    groups = [order[start:stop] for start, stop in zip(starts, np.r_[starts[1:], len(order)], strict=True)]
    first = next(
        (
            index for index, group in enumerate(groups)
            if int(frame.label_available_ts.lt(frame.decision_ts.iloc[group].min()).sum()) >= int(minimum_history_rows)
        ),
        None,
    )
    if first is None:
        raise ValueError("no base-representation OOF fold has the required resolved history")
    pieces = np.array_split(np.arange(first, len(groups)), min(int(count), len(groups) - first))
    return [np.concatenate([groups[int(item)] for item in piece]).astype(np.int32) for piece in pieces if len(piece)]


def _probability(model: Any, x: pd.DataFrame) -> np.ndarray:
    raw = np.asarray(model.predict_proba(x), dtype=float)
    result = np.zeros((len(x), 3), dtype=float)
    result[:, np.asarray(model.classes_, dtype=int)] = raw
    if not np.isfinite(result).all() or (result < 0).any() or not np.allclose(result.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("LightGBM did not produce a valid R3 probability simplex")
    return result


def _lgbm(params: dict[str, Any]):
    from lightgbm import LGBMClassifier
    clean = dict(params)
    clean.update({"objective": "multiclass", "num_class": 3, "n_jobs": min(3, int(clean.get("n_jobs", 3))), "verbosity": -1})
    return LGBMClassifier(**clean)


def _loss(y: np.ndarray, probability: np.ndarray) -> float:
    return float(-np.log(np.clip(probability[np.arange(len(y)), y.astype(int)], 1e-12, 1.0)).mean())


def _support(frame: pd.DataFrame, columns: Iterable[str]) -> dict[str, float]:
    return {column: float((pd.to_numeric(frame[column], errors="coerce").to_numpy(float) >= .60).mean()) for column in columns}


def _phantoms(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    candidates: list[str],
    support: dict[str, float],
    *,
    seed: int,
    count_per_bucket: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    """Create support-matched, independently permuted null features."""
    rng = np.random.default_rng(seed)
    x_train, x_valid = pd.DataFrame(index=train.index), pd.DataFrame(index=valid.index)
    bucket_sources: dict[str, list[str]] = {}
    for column in candidates:
        bucket_sources.setdefault(support_bucket(support[column]), []).append(column)
    lookup: dict[str, str] = {}
    for bucket, sources in sorted(bucket_sources.items()):
        if bucket == "below_p05":
            continue
        for number in range(count_per_bucket):
            source = sources[number % len(sources)]
            name = f"__phantom__{bucket}__{number:02d}"
            x_train[name] = train[source].to_numpy(float)[rng.permutation(len(train))]
            x_valid[name] = valid[source].to_numpy(float)[rng.permutation(len(valid))]
            lookup[name] = bucket
    return x_train, x_valid, lookup


def _mda_selection(
    later: pd.DataFrame,
    *,
    control_features: list[str],
    candidates: list[str],
    side: str,
    fold_id: int,
) -> tuple[list[str], pd.DataFrame]:
    """Two later-history chronological checks, with support-matched phantoms."""
    if not candidates:
        return [], pd.DataFrame(columns=["feature", "selected"])
    times = pd.Index(later.decision_ts.drop_duplicates().sort_values())
    thirds = np.array_split(np.arange(len(times)), 3)
    support = _support(later, candidates)
    rows: list[dict[str, Any]] = []
    for block, (train_groups, valid_groups) in enumerate(((thirds[0], thirds[1]), (np.r_[thirds[0], thirds[1]], thirds[2]))):
        if not len(train_groups) or not len(valid_groups):
            continue
        valid_start = times[int(valid_groups[0])]
        train = later[later.decision_ts.isin(times[train_groups]) & later.label_available_ts.lt(valid_start)].copy()
        valid = later[later.decision_ts.isin(times[valid_groups])].copy()
        if len(train) < 500 or len(valid) < 200 or set(train.r3_class.astype(int)) != {0, 1, 2}:
            continue
        phantom_train, phantom_valid, phantom_bucket = _phantoms(
            train, valid, candidates, support, seed=20_260_804 + 100 * fold_id + block,
        )
        columns = [*control_features, *candidates, *phantom_train.columns]
        model = _lgbm({
            "n_estimators": 180, "learning_rate": .035, "num_leaves": 16,
            "max_depth": 4, "min_child_samples": max(80, int(.02 * len(train))),
            "colsample_bytree": .8, "subsample": .85, "subsample_freq": 1,
            "reg_lambda": 10.0, "random_state": 20_260_804 + fold_id + block,
        })
        x_train = pd.concat([train.loc[:, [*control_features, *candidates]], phantom_train], axis=1)
        x_valid = pd.concat([valid.loc[:, [*control_features, *candidates]], phantom_valid], axis=1)
        model.fit(x_train, train.r3_class.to_numpy(int))
        baseline = _loss(valid.r3_class.to_numpy(int), _probability(model, x_valid))
        for column in [*candidates, *phantom_train.columns]:
            shuffled = x_valid.copy()
            # Python's ``hash`` is intentionally process-salted.  The MDA
            # permutation has to be reproducible across reruns, so derive its
            # small seed component from a stable digest instead.
            column_seed = int(sha256(column.encode("utf-8")).hexdigest()[:8], 16) % 10_000
            rng = np.random.default_rng(7_000_000 + 20_000 * fold_id + 100 * block + column_seed)
            shuffled[column] = shuffled[column].to_numpy(float)[rng.permutation(len(shuffled))]
            mda = _loss(valid.r3_class.to_numpy(int), _probability(model, shuffled)) - baseline
            is_phantom = column in phantom_bucket
            bucket = phantom_bucket[column] if is_phantom else support_bucket(support[column])
            rows.append({
                "side_name": side, "fold_id": fold_id, "inner_block": block,
                "feature": column, "is_phantom": is_phantom, "support_bucket": bucket,
                "active_share": np.nan if is_phantom else support[column],
                "base_logloss": baseline, "mda_logloss": float(mda),
                "train_rows": len(train), "validation_rows": len(valid),
                "validation_start_utc": valid_start.isoformat(),
                "train_max_label_available_utc": train.label_available_ts.max().isoformat(),
                "strict_prior_resolved": bool(train.label_available_ts.lt(valid_start).all()),
            })
    table = pd.DataFrame(rows)
    if table.empty:
        return [], table
    threshold = (
        table[table.is_phantom]
        .groupby(["inner_block", "support_bucket"], observed=True).mda_logloss
        .quantile(.95).rename("phantom_p95").reset_index()
    )
    actual = table[~table.is_phantom].merge(threshold, on=["inner_block", "support_bucket"], how="left")
    actual["passes_block"] = actual.mda_logloss.gt(actual.phantom_p95.fillna(np.inf)) & actual.mda_logloss.gt(0.0)
    summary = actual.groupby("feature", observed=True).agg(
        blocks=("inner_block", "nunique"),
        passed_blocks=("passes_block", "sum"),
        min_block_mda=("mda_logloss", "min"),
        active_share=("active_share", "first"),
    ).reset_index()
    summary["selected_pre_redundancy"] = summary.blocks.ge(2) & summary.passed_blocks.ge(2) & summary.min_block_mda.gt(0.0)
    candidates_ranked = summary[summary.selected_pre_redundancy].sort_values(["min_block_mda", "feature"], ascending=[False, True], kind="stable")
    decollinear: list[dict[str, Any]] = []
    accepted: list[str] = []
    for row in candidates_ranked.itertuples(index=False):
        column = str(row.feature)
        correlated = any(
            abs(float(later[column].corr(later[kept], method="spearman"))) >= .80
            for kept in accepted
            if np.isfinite(later[column].corr(later[kept], method="spearman"))
        )
        if not correlated:
            accepted.append(column)
            decollinear.append(row._asdict())
    selected = cap_support_diverse(pd.DataFrame(decollinear), maximum_total=20) if decollinear else []
    summary["selected"] = summary.feature.isin(selected)
    return selected, table.merge(summary.loc[:, ["feature", "selected", "selected_pre_redundancy"]], on="feature", how="left")


def _tail_metrics(frame: pd.DataFrame, score: str, *, layer: str) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for fraction in TOP_FRACTIONS:
        count = max(1, int(math.ceil(fraction * len(frame))))
        take = stable_stage_i_topk_positions(
            frame[score].to_numpy(float), candidate_ids=frame.candidate_id.to_numpy(object),
            side_names=frame.side_name.to_numpy(object), decision_timestamps=frame.decision_ts,
            signal_timestamps=frame["__ts__"], symbols=frame["__symbol__"].to_numpy(object), count=count,
        )
        selected = frame.iloc[take]
        output.append({
            "layer": layer, "top_fraction": fraction, "trades": len(selected),
            "net_bps_per_trade": float(selected.exact_net_bps.mean()),
            "worst_month_net_bps_per_trade": float(selected.groupby(selected.decision_ts.dt.strftime("%Y-%m")).exact_net_bps.mean().min()),
            "ranking_tie_policy": RANKING_POLICY,
        })
    return output


def _base_context(frame: pd.DataFrame, probability: np.ndarray) -> pd.DataFrame:
    output = frame.copy()
    output["r3_p_adverse"], output["r3_p_weak"], output["r3_p_clear"] = probability[:, 0], probability[:, 1], probability[:, 2]
    output["r3_opportunity_score"] = probability[:, 2] - probability[:, 0]
    output["base_r3_max_probability"] = probability.max(axis=1)
    output["base_r3_top2_margin"] = np.partition(probability, -2, axis=1)[:, -1] - np.partition(probability, -2, axis=1)[:, -2]
    output["base_r3_entropy"] = -(np.clip(probability, 1e-12, 1.0) * np.log(np.clip(probability, 1e-12, 1.0))).sum(axis=1)
    return output


def _quantile_state(residual: np.ndarray, *, support: float = 50.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    edges = np.quantile(residual, (1.0 / 3.0, 2.0 / 3.0), method="linear")
    if not edges[0] < edges[1]:
        raise ValueError("quantile residual meta target is degenerate")
    labels = np.digitize(residual, edges, right=True).astype(np.int8)
    clip = np.quantile(residual, (.05, .95), method="linear")
    winsor = np.clip(residual, clip[0], clip[1])
    global_mean = float(winsor.mean())
    locations = np.asarray([
        (winsor[labels == value].sum() + support * global_mean) / (int((labels == value).sum()) + support)
        for value in range(3)
    ], dtype=float)
    return labels, edges, locations


def _run_meta(
    base_oof: pd.DataFrame,
    *,
    side: str,
    meta_context: list[str],
    params: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-row direct-simplex residual meta refit on experimental base OOF."""
    value = base_oof.copy().sort_values(["decision_ts", "candidate_id"], kind="stable").reset_index(drop=True)
    mapped, map_audit, _ = prequential_same_side_r3_value_map(
        exact_net_bps=value.exact_net_bps.to_numpy(float), decision_timestamps=value.decision_ts,
        label_available_timestamps=value.label_available_ts, side=side,
        score=value.r3_opportunity_score.to_numpy(float), config=PrequentialR3ValueMapConfig(side=side),
    )
    value["prequential_base_expected_net_bps"] = mapped
    trust = base_oof_trust_features(value.loc[:, ["r3_p_adverse", "r3_p_weak", "r3_p_clear"]].to_numpy(float), map_audit)
    for column in trust:
        value[column] = trust[column].to_numpy(float)
    all_context = [*BASE_DIRECT, *[column for column in meta_context if column in value.columns]]
    all_context = list(dict.fromkeys(all_context))
    meta_fold_rows: list[pd.DataFrame] = []
    provenance: list[dict[str, Any]] = []
    selector = fold_local_meta_feature_selector(FixedLGBMContract(base_params={"objective": "multiclass", "num_class": 3}, meta_params=params, meta_feature_cap=40))
    predictor = fixed_lgbm_meta_predictor(FixedLGBMContract(base_params={"objective": "multiclass", "num_class": 3}, meta_params=params, meta_feature_cap=40))
    for fold_id in sorted(value.fold_id.unique()):
        if int(fold_id) <= int(value.fold_id.min()):
            continue
        valid = value[(value.fold_id == fold_id) & np.isfinite(value.prequential_base_expected_net_bps)].copy()
        if valid.empty:
            continue
        start = valid.decision_ts.min()
        train = value[(value.fold_id < fold_id) & value.label_available_ts.lt(start) & np.isfinite(value.prequential_base_expected_net_bps)].copy()
        if len(train) < 500:
            continue
        residual = train.exact_net_bps.to_numpy(float) - train.prequential_base_expected_net_bps.to_numpy(float)
        labels, edges, locations = _quantile_state(residual)
        selected, selection = selector(train, labels, all_context, BASE_DIRECT, _MetaSpec())
        probability = predictor(train.loc[:, list(selected)], labels, np.ones(len(train)), valid.loc[:, list(selected)], _MetaSpec())
        prior = np.bincount(labels, minlength=3).astype(float) / float(len(labels))
        correction = np.clip((probability - prior) @ locations, -200.0, 200.0)
        piece = valid.copy()
        piece["meta_score"] = piece.prequential_base_expected_net_bps.to_numpy(float) + correction
        piece["meta_p_lower_residual_tercile"], piece["meta_p_middle_residual_tercile"], piece["meta_p_upper_residual_tercile"] = probability[:, 0], probability[:, 1], probability[:, 2]
        residual_valid = valid.exact_net_bps.to_numpy(float) - valid.prequential_base_expected_net_bps.to_numpy(float)
        target_valid = np.digitize(residual_valid, edges, right=True)
        meta_fold_rows.append(piece)
        provenance.append({
            "side_name": side, "fold_id": int(fold_id), "train_rows": len(train), "validation_rows": len(valid),
            "validation_start_utc": start.isoformat(), "train_max_label_available_utc": train.label_available_ts.max().isoformat(),
            "strict_prior_resolved": bool(train.label_available_ts.lt(start).all()), "per_row_base_handoff": True,
            "selected_meta_features": list(selected), "meta_feature_count": len(selected), "meta_selection": selection,
            "residual_q33_bps": float(edges[0]), "residual_q67_bps": float(edges[1]),
            "target_logloss": _loss(target_valid, probability),
            "target_brier": float(np.square(probability - np.eye(3)[target_valid]).sum(axis=1).mean() / 3.0),
        })
    if not meta_fold_rows:
        raise ValueError(f"{side}: no strict per-row meta OOF support")
    return pd.concat(meta_fold_rows, ignore_index=True), pd.DataFrame(provenance)


class _MetaSpec:
    """The hook needs only the fixed familiar target metadata."""
    arm_id = "T3Q_fold_quantile_ordinal_residual"
    family = "quantile_ordinal_residual"
    residual_clip_bps = 200.0
    shrinkage_support = 50.0
    hurdle_bps = 0.0
    veto_probability = .5


def _load_side(
    *, selector_dir: Path, base_selection_dir: Path, meta_selection_dir: Path, side: str,
) -> tuple[pd.DataFrame, list[str], list[str], list[str], dict[str, Any], dict[str, Any]]:
    base_manifest = _json(base_selection_dir / side / "manifest.json")
    meta_manifest = _json(meta_selection_dir / side / "manifest.json")
    base_input = list(map(str, base_manifest["input_feature_contract"]))
    base_control = list(map(str, base_manifest["selected_feature_contract"]))
    requested_meta_context = list(map(str, meta_manifest.get("selected_feature_contract", meta_manifest.get("selected_features", ()))))
    ledger = pd.read_parquet(selector_dir / "selector_ledger.parquet")
    # The frozen meta selection contains its derived direct-base handoff and
    # prequential-map fields.  They are rebuilt from the experimental simplex,
    # never read from the raw selector matrix.  Ask Parquet only for genuine
    # materialised decision-time context fields.
    from pyarrow.parquet import ParquetFile
    available_matrix = set(ParquetFile(selector_dir / "selector_features.parquet").schema.names)
    meta_context = [
        column for column in requested_meta_context
        if column in available_matrix
        and column not in BASE_DIRECT
        and not column.startswith("prequential_")
        and "expected_net" not in column
    ]
    required = set(base_input) | set(base_control) | set(meta_context)
    matrix = pd.read_parquet(selector_dir / "selector_features.parquet", columns=[*IDENTITY, *sorted(required)])
    if not ledger.loc[:, list(IDENTITY)].reset_index(drop=True).equals(matrix.loc[:, list(IDENTITY)].reset_index(drop=True)):
        raise ValueError("selector ledger/features identity ordering differs")
    local = ledger.side_name.astype(str).str.lower().eq(side).to_numpy()
    valid = local & _valid_rows(ledger)
    raw = pd.concat([
        ledger.loc[valid, list(IDENTITY) + ["side_name", "r3_class", "exact_net_bps", "label_available_ts", "decision_ts"]].reset_index(drop=True),
        matrix.loc[valid].reset_index(drop=True).drop(columns=list(IDENTITY)),
    ], axis=1)
    raw["decision_ts"] = pd.to_datetime(raw.decision_ts, utc=True, errors="raise")
    raw["label_available_ts"] = pd.to_datetime(raw.label_available_ts, utc=True, errors="raise")
    if not raw.label_available_ts.eq(raw.decision_ts + pd.Timedelta(hours=12)).all():
        raise ValueError(f"{side}: base label availability is not executable entry + H12")
    forbidden = {column for column in base_input if column.startswith("r3_") or "expected_net" in column or "residual" in column or "meta" in column or "trust" in column}
    if forbidden:
        raise ValueError(f"{side}: forbidden non-base input escaped base contract: {sorted(forbidden)}")
    if not set(base_control).issubset(base_input):
        raise ValueError(f"{side}: frozen base control escapes declared base input contract")
    missing = sorted(required.difference(raw.columns))
    if missing:
        raise ValueError(f"{side}: selector matrix lacks contract fields {missing[:10]}")
    return raw.sort_values(["decision_ts", "candidate_id"], kind="stable").reset_index(drop=True), base_input, base_control, meta_context, base_manifest, meta_manifest


def _run_side(
    *, selector_dir: Path, base_selection_dir: Path, meta_selection_dir: Path, side: str,
    folds: int, min_history_rows: int, output_dir: Path,
) -> dict[str, pd.DataFrame]:
    (output_dir / side).mkdir(parents=True, exist_ok=True)
    raw, base_input, base_control, meta_context, base_manifest, meta_manifest = _load_side(
        selector_dir=selector_dir, base_selection_dir=base_selection_dir, meta_selection_dir=meta_selection_dir, side=side,
    )
    blocks = _strict_blocks(raw, minimum_history_rows=min_history_rows, count=folds)
    base_records: dict[str, list[pd.DataFrame]] = {"control": [], "base_leaf": []}
    metrics: list[dict[str, Any]] = []
    leaf_lineage: list[pd.DataFrame] = []
    rule_rows: list[dict[str, Any]] = []
    selection_rows: list[pd.DataFrame] = []
    fold_rows: list[dict[str, Any]] = []
    for fold_id, valid_idx in enumerate(blocks):
        test = raw.iloc[valid_idx].copy()
        start = test.decision_ts.min()
        history = raw[raw.label_available_ts.lt(start)].copy()
        dictionary, later = strict_dictionary_split(history)
        candidate_columns: list[str] = []
        later_features = later.copy()
        test_features = test.copy()
        for target_name, horizon in TARGETS.items():
            label = target_values(dictionary, horizon_hours=horizon)
            label = label[label.base_leaf_target_available_ts.lt(later.decision_ts.min())].copy()
            try:
                frozen = fit_dictionary(
                    label, side=side, fold_id=fold_id, target_name=target_name,
                    legal_features=base_input, applied_from_decision=later.decision_ts.min(), config=BaseLeafConfig(),
                )
            except BaseLeafRepresentationError as exc:
                rule_rows.append({"side_name": side, "fold_id": fold_id, "target": target_name, "status": "not_fit", "reason": str(exc)})
                continue
            later_rep, later_lineage = frozen.apply(later)
            test_rep, test_lineage = frozen.apply(test)
            features = [column for column in later_rep.columns if column.startswith("baseleaf__")]
            later_features = later_features.join(later_rep.loc[:, features])
            test_features = test_features.join(test_rep.loc[:, features])
            candidate_columns.extend(features)
            leaf_lineage.extend([later_lineage.assign(applied_population="later_history"), test_lineage.assign(applied_population="outer_test")])
            for cluster_id, cluster in enumerate(frozen.clusters):
                for rule in cluster:
                    rule_rows.append({
                        "side_name": side, "fold_id": fold_id, "target": target_name, "status": "fit",
                        "cluster": cluster_id, "rule_id": rule.rule_id, "economic_effect": rule.economic_effect,
                        "conditions_json": json.dumps(rule.conditions), "dictionary_rows": frozen.dictionary_rows,
                        "dictionary_max_label_available_utc": frozen.dictionary_max_label_available_utc,
                        "applied_from_decision_utc": frozen.applied_from_decision_utc,
                    })
        candidate_columns = list(dict.fromkeys(candidate_columns))
        selected, selection = _mda_selection(later_features, control_features=base_control, candidates=candidate_columns, side=side, fold_id=fold_id)
        if not selection.empty:
            selection_rows.append(selection)
        for arm, features in (("control", base_control), ("base_leaf", [*base_control, *selected])):
            if set(later_features.r3_class.astype(int)) != {0, 1, 2}:
                raise ValueError(f"{side}/fold{fold_id}: later base fit lacks an R3 class")
            model = _lgbm(dict(base_manifest["best_params"]))
            model.fit(later_features.loc[:, features], later_features.r3_class.to_numpy(int))
            probability = _probability(model, test_features.loc[:, features])
            piece = _base_context(test_features, probability)
            piece["fold_id"] = fold_id
            piece["base_arm"] = arm
            base_records[arm].append(piece)
            y = test.r3_class.to_numpy(int)
            one_hot = np.eye(3)[y]
            metrics.append({
                "side_name": side, "fold_id": fold_id, "arm": arm, "layer": "base", "rows": len(piece),
                "multiclass_logloss": _loss(y, probability), "multiclass_brier": float(np.square(probability - one_hot).sum(axis=1).mean() / 3.0),
                "selected_leaf_features": list(selected), "selected_leaf_count": len(selected),
            })
            for item in _tail_metrics(piece, "r3_opportunity_score", layer="base"):
                metrics.append({"side_name": side, "fold_id": fold_id, "arm": arm, **item})
        fold_rows.append({
            "side_name": side, "fold_id": fold_id, "outer_validation_rows": len(test),
            "outer_validation_start_utc": start.isoformat(), "history_rows": len(history),
            "dictionary_rows": len(dictionary), "later_base_fit_rows": len(later),
            "dictionary_max_label_available_utc": dictionary.label_available_ts.max().isoformat(),
            "later_min_decision_utc": later.decision_ts.min().isoformat(),
            "dictionary_precedes_later": bool(dictionary.label_available_ts.lt(later.decision_ts.min()).all()),
            "base_fit_excludes_dictionary_rows": True, "base_input_feature_count": len(base_input),
            "frozen_control_feature_count": len(base_control), "selected_leaf_count": len(selected),
        })
    result: dict[str, pd.DataFrame] = {}
    meta_params = dict(meta_manifest.get("params", meta_manifest.get("best_params", {})))
    if not meta_params:
        raise ValueError(f"{side}: no frozen meta parameters")
    for arm, parts in base_records.items():
        base = pd.concat(parts, ignore_index=True).sort_values(["decision_ts", "candidate_id"], kind="stable").reset_index(drop=True)
        meta, meta_lineage = _run_meta(base, side=side, meta_context=meta_context, params=meta_params)
        base.to_parquet(output_dir / side / f"{arm}_base_oof.parquet", index=False)
        meta.to_parquet(output_dir / side / f"{arm}_meta_oof.parquet", index=False)
        meta_lineage.to_parquet(output_dir / side / f"{arm}_meta_fold_provenance.parquet", index=False)
        result[f"{arm}_base"] = base
        result[f"{arm}_meta"] = meta
        for item in _tail_metrics(meta, "meta_score", layer="meta"):
            metrics.append({"side_name": side, "fold_id": "pooled_meta_oof", "arm": arm, **item})
    # Fold-level base rows and the aggregated meta row deliberately share one
    # metrics table; use an explicit textual fold key rather than letting
    # Arrow infer an inconsistent int/string column from construction order.
    metrics_frame = pd.DataFrame(metrics)
    if "fold_id" in metrics_frame:
        metrics_frame["fold_id"] = metrics_frame["fold_id"].astype(str)
    metrics_frame.to_parquet(output_dir / side / "metrics.parquet", index=False)
    pd.DataFrame(fold_rows).to_parquet(output_dir / side / "fold_provenance.parquet", index=False)
    pd.concat(leaf_lineage, ignore_index=True).to_parquet(output_dir / side / "leaf_lineage.parquet", index=False) if leaf_lineage else pd.DataFrame().to_parquet(output_dir / side / "leaf_lineage.parquet", index=False)
    pd.DataFrame(rule_rows).to_parquet(output_dir / side / "rule_clusters.parquet", index=False)
    pd.concat(selection_rows, ignore_index=True).to_parquet(output_dir / side / "selection_mda.parquet", index=False) if selection_rows else pd.DataFrame().to_parquet(output_dir / side / "selection_mda.parquet", index=False)
    _write_json(output_dir / side / "manifest.json", {
        "schema": "stage_i_base_leaf_representation_oof_v1", "status": "complete", "side": side,
        "base_target": "R3 TP6/SL4/H12: adverse=-1, weak=0, robust_clear=+1 for leaf discovery; frozen multiclass R3 for base fit",
        "dictionary": {"targets": TARGETS, **asdict(BaseLeafConfig()), "early_resolved_only": True, "base_fit_excludes_dictionary_rows": True},
        "base_contract": {"input_feature_count": len(base_input), "control_feature_count": len(base_control), "selection": str(base_selection_dir / side / "manifest.json")},
        "meta_contract": {"per_row_direct_simplex_handoff": True, "target": "quantile ordinal exact-net residual", "context_feature_count": len(meta_context), "selection": str(meta_selection_dir / side / "manifest.json")},
    })
    return result


def _pooled(arms: dict[str, dict[str, pd.DataFrame]], output_dir: Path) -> pd.DataFrame:
    (output_dir / "pooled_global").mkdir(parents=True, exist_ok=True)
    metrics: list[pd.DataFrame] = []
    for arm in ("control", "base_leaf"):
        pieces: list[pd.DataFrame] = []
        audits: list[pd.DataFrame] = []
        for side in ("long", "short"):
            frame = arms[side][f"{arm}_meta"].copy()
            frame["candidate_key"] = frame.side_name.astype(str) + "::" + frame.candidate_id.astype(str) + "::" + frame["__ts__"].astype(str)
            mapped, audit = apply_causal_21d_side_admission(
                frame, score_column="meta_score", net_column="exact_net_bps", decision_column="decision_ts",
                label_available_column="label_available_ts", identity_column="candidate_key", spec=Causal21dAdmissionSpec(),
            )
            mapped["arm"] = arm
            pieces.append(mapped)
            audits.append(audit.assign(side_name=side, arm=arm))
            mapped.to_parquet(output_dir / "pooled_global" / f"{arm}_{side}_mapped_meta_oof.parquet", index=False)
        combined = pd.concat(pieces, ignore_index=True)
        combined.to_parquet(output_dir / "pooled_global" / f"{arm}_mapped_meta_oof.parquet", index=False)
        pd.concat(audits, ignore_index=True).to_parquet(output_dir / "pooled_global" / f"{arm}_admission_audit.parquet", index=False)
        comparison = pooled_global_admission_comparison(
            combined, raw_score_column="meta_score", net_column="exact_net_bps", identity_column="candidate_key", top_fractions=TOP_FRACTIONS,
        )
        comparison["arm"] = arm
        metrics.append(comparison)
    output = pd.concat(metrics, ignore_index=True)
    output.to_parquet(output_dir / "pooled_global" / "pooled_global_metrics.parquet", index=False)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selector-dir", type=Path, default=ROOT / "data_perp/artifacts/stage_i_selector_sample_20260803_v5")
    parser.add_argument("--base-selection-dir", type=Path, default=ROOT / "data_perp/artifacts/stage_i_base_selection_20260803_v7")
    parser.add_argument("--meta-selection-dir", type=Path, default=ROOT / "data_perp/artifacts/stage_i_meta_tercile_target_20260803_v1")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data_perp/artifacts/stage_i_base_leaf_representation_oof_20260804_v4")
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--min-history-rows", type=int, default=8_000)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite immutable output: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    _write_json(args.output_dir / "manifest.json", {"schema": "stage_i_base_leaf_representation_oof_v1", "status": "RUNNING"})
    try:
        arms = {
            side: _run_side(
                selector_dir=args.selector_dir, base_selection_dir=args.base_selection_dir, meta_selection_dir=args.meta_selection_dir,
                side=side, folds=args.folds, min_history_rows=args.min_history_rows, output_dir=args.output_dir,
            )
            for side in ("long", "short")
        }
        pooled = _pooled(arms, args.output_dir)
        _write_json(args.output_dir / "manifest.json", {
            "schema": "stage_i_base_leaf_representation_oof_v1", "status": "complete",
            "selector_dir": str(args.selector_dir), "selector_manifest_sha256": _sha(args.selector_dir / "manifest.json"),
            "base_selection_dir": str(args.base_selection_dir), "meta_selection_dir": str(args.meta_selection_dir),
            "folds": args.folds, "minimum_resolved_history_rows": args.min_history_rows,
            "arms": ["control", "base_leaf"],
            "base_features": "only per-side frozen base input/selected contracts; meta/residual/trust fields denied",
            "base_leaf_target": "native R3 signed opportunity only", "meta_handoff": "direct same-side experimental R3 simplex, per-row",
            "ranking": "causal side-local 21-day common-bps mapping then one pooled-global ranking",
            "pooled_metric_rows": int(len(pooled)),
        })
    except Exception as exc:
        _write_json(args.output_dir / "manifest.json", {"schema": "stage_i_base_leaf_representation_oof_v1", "status": "failed", "error": repr(exc)})
        raise
    print(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
