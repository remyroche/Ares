#!/usr/bin/env python3
"""Long-only executable-net LambdaRank experiment.

This is the deliberately narrow first implementation of the ranking framework.
It consumes the completed long Stage-I MDA contract, reads those features from
the canonical point-in-time store, and never adds leaf/GMM/DAE/regime-output
representations.  Every outer test block is chronologically later than both
the base and residual-ranker fitting/calibration rows.

The score is ranked globally only at evaluation time.  Timestamp x side groups
are used exclusively for the LambdaRank loss, never for portfolio selection.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.packb_static_point_feature_loader import (
    FrozenFeatureContract,
    _feature_contract_digest,
    discover_causal_feature_universe,
    freeze_feature_contract,
)
from extreme_price_movements.stage_i_production_data_adapter import (
    make_static_pit_feature_loader,
)


SCHEMA = "long_only_executable_net_lambdarank_v1"
SIDE = "long"
HORIZON_HOURS = 12
FORBIDDEN_REPRESENTATION_TOKENS = (
    "leafreg", "leaf_", "cluster", "posterior", "gmm", "dae",
    "regime_p_", "regime_relative", "regime_z_", "state_reference",
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")


def _month(value: pd.Series) -> pd.Series:
    return pd.to_datetime(value, utc=True).dt.strftime("%Y-%m")


def _subset_contract(contract: FrozenFeatureContract, fields: list[str]) -> FrozenFeatureContract:
    """Reuse the source provenance but freeze this exact 92-field projection."""
    fields = sorted(map(str, fields))
    return replace(
        contract,
        feature_columns=tuple(fields),
        feature_contract_sha256=_feature_contract_digest(
            feature_columns=fields,
            candidate_universe_sha256=contract.candidate_universe_sha256,
            source_schema_sha256=contract.source_schema_sha256,
            raw_allowlist_sha256=contract.raw_allowlist_sha256,
            generator_registry_sha256=contract.generator_registry_sha256,
            store_scan_manifest_sha256=contract.store_scan_manifest_sha256,
            coverage_profile_sha256=contract.coverage_profile_sha256,
            min_exact_key_coverage=contract.min_exact_key_coverage,
            min_non_null_feature_coverage=contract.min_non_null_feature_coverage,
            max_feature_columns=contract.max_feature_columns,
            coverage_admission_rejections=contract.coverage_admission_rejections,
        ),
    )


def _read_contract_features(mda_manifest: Path) -> list[str]:
    data = json.loads(mda_manifest.read_text())
    if data.get("status") != "complete":
        raise ValueError("long MDA manifest is not complete")
    fields = list(map(str, data.get("selected_feature_contract", [])))
    if len(fields) < 30 or len(fields) != len(set(fields)):
        raise ValueError("long MDA feature contract is unexpectedly small or duplicated")
    forbidden = [
        field for field in fields
        if any(token in field.lower() for token in FORBIDDEN_REPRESENTATION_TOKENS)
    ]
    if forbidden:
        raise ValueError(f"MDA contract contains prohibited representation fields: {forbidden}")
    # Frozen feature contracts are canonically sorted.  Column ordering has no
    # semantic meaning to a tree, while keeping this ordering lets the PIT
    # loader verify the exact projection hash.
    return sorted(fields)


def _load_ledger(path: Path) -> pd.DataFrame:
    columns = [
        "candidate_id", "__ts__", "side_name", "gross_bps", "net_bps",
        "m6_contract_complete", "shared_regime_contract_complete",
        "prequential_base_expected_net_bps",
    ]
    frame = pd.read_parquet(path, columns=columns)
    frame = frame.loc[frame["side_name"].astype(str).str.lower().eq(SIDE)].copy()
    frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True)
    frame["label_available_ts"] = frame["__ts__"] + pd.Timedelta(hours=HORIZON_HOURS)
    complete = frame["m6_contract_complete"].fillna(False) & frame["shared_regime_contract_complete"].fillna(False)
    finite = np.isfinite(pd.to_numeric(frame["gross_bps"], errors="coerce")) & np.isfinite(pd.to_numeric(frame["net_bps"], errors="coerce"))
    frame = frame.loc[complete & finite].copy()
    frame["current_base_bps"] = pd.to_numeric(
        frame["prequential_base_expected_net_bps"], errors="coerce"
    ).fillna(0.0).astype("float32")
    if frame.empty or frame["candidate_id"].duplicated().any():
        raise ValueError("long ledger is empty or has duplicate candidate identities")
    return frame.sort_values(["__ts__", "candidate_id"], kind="stable").reset_index(drop=True)


def _materialize_features(
    ledger: pd.DataFrame, *, fields: list[str], production_contract: FrozenFeatureContract,
    store: str, cache_path: Path,
) -> pd.DataFrame:
    """Exact causal read with a resumable, identity-checked narrow cache."""
    identity = ledger.loc[:, ["candidate_id", "__ts__"]].copy()
    # The feature store needs the canonical symbol.  It is intentionally read
    # from the separately verified candidate population below before calling.
    if cache_path.exists():
        cached = pd.read_parquet(cache_path)
        if list(cached["candidate_id"]) != list(ledger["candidate_id"]):
            raise ValueError("existing feature cache does not match the long candidate ledger")
        missing = sorted(set(fields).difference(cached.columns))
        if missing:
            raise ValueError(f"existing feature cache misses selected features: {missing}")
        return cached.loc[:, ["candidate_id", *fields]].copy()
    if "__symbol__" not in ledger.columns:
        raise ValueError("long ledger requires canonical __symbol__ before PIT materialisation")
    contract = _subset_contract(production_contract, fields)
    loader = make_static_pit_feature_loader(
        feature_store_dir=store, feature_contract=contract,
        max_rows_per_batch=4_000, max_columns_per_read=92, verify_frozen_schema=True,
    )
    loaded = loader(ledger.loc[:, ["candidate_id", "__ts__", "__symbol__"]], fields)
    if list(loaded["candidate_id"]) != list(ledger["candidate_id"]):
        raise ValueError("PIT feature read changed candidate order")
    result = loaded.loc[:, ["candidate_id", *fields]].copy()
    result.to_parquet(cache_path, index=False, compression="zstd")
    return result


def _attach_symbols(ledger: pd.DataFrame, candidate_panel: Path) -> pd.DataFrame:
    """Use the previously materialised full universe only as an identity map."""
    import duckdb
    panel = duckdb.sql(
        "SELECT candidate_id, any_value(__symbol__) AS __symbol__ "
        "FROM read_parquet(?) GROUP BY candidate_id",
        params=[str(candidate_panel)],
    ).df()
    panel = panel.drop_duplicates("candidate_id")
    result = ledger.merge(panel, on="candidate_id", how="left", validate="one_to_one", sort=False)
    if result["__symbol__"].isna().any():
        raise ValueError("full-universe identity map is missing symbols for long candidates")
    # Feature-store files retain the canonical slash form while a legacy
    # evaluation panel used underscores in the same market identifier.
    result["__symbol__"] = result["__symbol__"].astype(str).str.replace(
        "_USD:USD", "/USD:USD", n=1, regex=False
    )
    return result


def _group_sizes(frame: pd.DataFrame) -> np.ndarray:
    return frame.groupby("query_id", sort=False, observed=True).size().to_numpy(dtype=np.int32)


def _relevance(values: np.ndarray, groups: pd.Series, *, margin: float, classes: int) -> np.ndarray:
    """Fixed weak bins plus within-query ranks for opportunity/residual labels."""
    if classes < 3:
        raise ValueError("at least three relevance grades required")
    out = np.zeros(len(values), dtype=np.int8)
    frame = pd.DataFrame({"value": values, "query_id": groups.to_numpy()})
    for _, index in frame.groupby("query_id", sort=False).groups.items():
        idx = np.asarray(list(index), dtype=np.int64)
        vals = values[idx]
        weak = (vals > 0.0) & (vals <= margin)
        out[idx[weak]] = 1
        strong_idx = idx[vals > margin]
        if len(strong_idx):
            ranks = pd.Series(values[strong_idx]).rank(method="average", pct=True).to_numpy()
            out[strong_idx] = np.clip(2 + np.floor(ranks * (classes - 2)).astype(int), 2, classes - 1)
    return out


def _ordered_ranker_frame(frame: pd.DataFrame, features: list[str], label: np.ndarray) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    work = frame.loc[:, ["query_id", *features]].copy()
    work.loc[:, features] = work.loc[:, features].replace([np.inf, -np.inf], np.nan)
    # LightGBM handles NaNs, but all-null columns are a failed materialisation.
    all_null = [column for column in features if work[column].notna().sum() == 0]
    if all_null:
        raise ValueError(f"all-null model features: {all_null}")
    work = work.sort_values("query_id", kind="stable").reset_index(drop=True)
    order = work.index.to_numpy()
    # index is reset; recover ordering independently so labels remain aligned.
    source = frame.loc[:, ["query_id"]].copy()
    source["__row__"] = np.arange(len(source), dtype=np.int64)
    source = source.sort_values("query_id", kind="stable")
    order = source["__row__"].to_numpy(dtype=np.int64)
    return work.loc[:, features], label[order], _group_sizes(work)


def _fit_ranker(frame: pd.DataFrame, features: list[str], label: np.ndarray, *, seed: int):
    from lightgbm import LGBMRanker
    X, y, group = _ordered_ranker_frame(frame, features, label)
    min_child = max(50, int(np.ceil(0.015 * len(X))))
    model = LGBMRanker(
        objective="lambdarank", metric="ndcg", label_gain=list(range(6)),
        n_estimators=500, learning_rate=0.04, num_leaves=24, max_depth=-1,
        min_child_samples=min_child, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.8, reg_alpha=1.5, reg_lambda=4.0,
        lambdarank_truncation_level=10, random_state=seed, n_jobs=4, verbosity=-1,
    )
    model.fit(X, y, group=group)
    return model, {"rows": int(len(X)), "groups": int(len(group)), "min_child_samples": min_child}


def _predict_ranker(model, frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    values = frame.loc[:, features].replace([np.inf, -np.inf], np.nan)
    return np.asarray(model.predict(values), dtype=np.float64)


def _fit_bps_map(raw: np.ndarray, realized_net: np.ndarray):
    """Monotonic validation-only map from raw rank score to common net bps."""
    from sklearn.isotonic import IsotonicRegression
    finite = np.isfinite(raw) & np.isfinite(realized_net)
    if finite.sum() < 100:
        raise ValueError("insufficient calibration rows")
    return IsotonicRegression(out_of_bounds="clip", y_min=-500.0, y_max=500.0).fit(raw[finite], realized_net[finite])


def _tail_metrics(frame: pd.DataFrame, score: str) -> list[dict]:
    rows: list[dict] = []
    for label, sub in [("pooled", frame), *[(month, g) for month, g in frame.groupby("month", observed=True)]]:
        ordered = sub.sort_values(score, ascending=False, kind="stable")
        for tail in (0.01, 0.03, 0.05, 0.10):
            n = max(1, int(np.ceil(len(ordered) * tail)))
            chosen = ordered.iloc[:n]
            rows.append({
                "score": score, "period": label, "tail": tail, "trades": n,
                "gross_bps_per_trade": float(chosen["gross_bps"].mean()),
                "net_bps_per_trade": float(chosen["net_bps"].mean()),
                "win_rate_net": float((chosen["net_bps"] > 0).mean()),
            })
    return rows


def _folds() -> list[tuple[str, str, str]]:
    return [
        ("oof_may_jun", "2024-05-01", "2024-07-01"),
        ("oof_jul_aug", "2024-07-01", "2024-09-01"),
        ("oos_sep_nov", "2024-09-01", "2024-12-01"),
    ]


def run(args: argparse.Namespace) -> None:
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    fields = _read_contract_features(Path(args.mda_manifest))
    selector_contract = FrozenFeatureContract.from_mapping(json.loads(Path(args.selector_contract).read_text()))
    absent = sorted(set(fields).difference(selector_contract.feature_columns))
    if absent:
        raise ValueError(f"MDA fields not admitted by frozen PIT contract: {absent}")
    ledger = _attach_symbols(_load_ledger(Path(args.ledger)), Path(args.identity_panel))
    # The selected columns remain frozen by the completed MDA manifest.  The
    # store/registry evidence is refreshed separately because the historical
    # selector contract intentionally fails closed after registry evolution.
    # This discovery is schema/availability-only: it has no outcomes, targets,
    # importance, or HPO input.
    universe = discover_causal_feature_universe(
        ledger.loc[:, ["candidate_id", "__ts__", "__symbol__"]],
        feature_store_dir=args.feature_store,
    )
    fresh_universe_contract = freeze_feature_contract(
        universe, min_exact_key_coverage=0.0, min_non_null_feature_coverage=0.0,
        max_feature_columns=None,
    )
    missing_fresh = sorted(set(fields).difference(fresh_universe_contract.feature_columns))
    if missing_fresh:
        raise ValueError(f"selected long MDA fields absent from fresh causal store: {missing_fresh}")
    fresh_contract = _subset_contract(fresh_universe_contract, fields)
    _write_json(output / "fresh_mda92_feature_contract.json", fresh_contract.to_dict())
    feature_cache = output / "long_mda92_features.parquet"
    features = _materialize_features(
        ledger, fields=fields, production_contract=fresh_contract,
        store=args.feature_store, cache_path=feature_cache,
    )
    if not features["candidate_id"].equals(ledger["candidate_id"]):
        raise ValueError("materialized feature cache identity drift")
    coverage = features.loc[:, fields].notna().mean().rename("coverage").reset_index()
    coverage.columns = ["feature", "coverage"]
    coverage.to_parquet(output / "feature_coverage.parquet", index=False)
    low_coverage = coverage.loc[coverage["coverage"] < 0.90, "feature"].tolist()
    if low_coverage:
        raise ValueError(f"selected MDA fields fall below 90% full-universe coverage: {low_coverage}")
    work = pd.concat([ledger.reset_index(drop=True), features.loc[:, fields].reset_index(drop=True)], axis=1)
    work["query_id"] = work["__ts__"].astype(str) + "|long"
    work["month"] = _month(work["__ts__"])
    predictions: list[pd.DataFrame] = []
    query_audit: list[dict] = []
    manifests: list[dict] = []
    for fold_num, (name, start, end) in enumerate(_folds(), start=1):
        start_ts, end_ts = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
        test = work.loc[(work["__ts__"] >= start_ts) & (work["__ts__"] < end_ts)].copy()
        history = work.loc[work["label_available_ts"] < start_ts].copy()
        if test.empty or len(history) < 10_000:
            raise ValueError(f"{name}: insufficient chronological data")
        # Phase A trains base.  Phase B is an OOF base/meta training interval;
        # Phase C calibrates the residual map.  All resolve before test start.
        cuts = history["__ts__"].quantile([0.60, 0.80]).to_list()
        base_train = history.loc[history["__ts__"] <= cuts[0]].copy()
        meta_train = history.loc[(history["__ts__"] > cuts[0]) & (history["__ts__"] <= cuts[1])].copy()
        meta_cal = history.loc[history["__ts__"] > cuts[1]].copy()
        base_label = _relevance(base_train["net_bps"].to_numpy(), base_train["query_id"], margin=35.0, classes=6)
        base_model, base_audit = _fit_ranker(base_train, fields, base_label, seed=20260804 + fold_num)
        for block in (meta_train, meta_cal, test):
            block["new_base_raw"] = _predict_ranker(base_model, block, fields)
        base_map = _fit_bps_map(meta_cal["new_base_raw"].to_numpy(), meta_cal["net_bps"].to_numpy())
        for block in (meta_train, meta_cal, test):
            block["new_base_bps"] = base_map.predict(block["new_base_raw"])
        # Two residual rankers: ranking correction for current base and for the
        # new base.  This is the directly comparable long-only three-way core.
        meta_outputs = {}
        for anchor in ("current_base_bps", "new_base_bps"):
            meta_features = [*fields, anchor]
            residual = meta_train["net_bps"].to_numpy() - meta_train[anchor].to_numpy()
            meta_label = _relevance(residual, meta_train["query_id"], margin=50.0, classes=5)
            model, audit = _fit_ranker(meta_train, meta_features, meta_label, seed=20260900 + fold_num)
            meta_cal_raw = _predict_ranker(model, meta_cal, meta_features)
            residual_map = _fit_bps_map(meta_cal_raw, meta_cal["net_bps"].to_numpy() - meta_cal[anchor].to_numpy())
            test_raw = _predict_ranker(model, test, meta_features)
            meta_outputs[anchor] = residual_map.predict(test_raw)
            base_audit[f"meta_{anchor}"] = audit
        out = test.loc[:, ["candidate_id", "__ts__", "month", "gross_bps", "net_bps", "current_base_bps", "new_base_bps"]].copy()
        out["fold"] = name
        out["current_base_plus_rankmeta_bps"] = out["current_base_bps"] + meta_outputs["current_base_bps"]
        out["new_base_plus_rankmeta_bps"] = out["new_base_bps"] + meta_outputs["new_base_bps"]
        predictions.append(out)
        query_audit.extend([
            {"fold": name, "split": split, "rows": int(len(block)), "query_groups": int(block["query_id"].nunique()), "min_group": int(block.groupby("query_id").size().min()), "max_group": int(block.groupby("query_id").size().max())}
            for split, block in (("base_train", base_train), ("meta_train", meta_train), ("meta_cal", meta_cal), ("test", test))
        ])
        manifests.append({"fold": name, "test_start": start, "test_end": end, "history_labels_resolved_before": start, "base_train_end": str(cuts[0]), "meta_train_end": str(cuts[1]), "base_audit": base_audit})
    pred = pd.concat(predictions, ignore_index=True)
    pred.to_parquet(output / "oos_rank_predictions.parquet", index=False, compression="zstd")
    pd.DataFrame(query_audit).to_parquet(output / "query_group_audit.parquet", index=False)
    metrics: list[dict] = []
    for score in ("current_base_bps", "new_base_bps", "current_base_plus_rankmeta_bps", "new_base_plus_rankmeta_bps"):
        metrics.extend(_tail_metrics(pred, score))
    pd.DataFrame(metrics).to_parquet(output / "four_way_comparison.parquet", index=False)
    _write_json(output / "fold_manifest.json", {"schema": SCHEMA, "side": SIDE, "folds": manifests})
    _write_json(output / "run_manifest.json", {
        "schema": SCHEMA, "status": "complete", "side": SIDE,
        "feature_count": len(fields), "feature_contract": fields,
        "prohibited_representation_tokens": list(FORBIDDEN_REPRESENTATION_TOKENS),
        "feature_store": args.feature_store, "horizon_hours": HORIZON_HOURS,
        "fresh_causal_contract_sha256": fresh_contract.feature_contract_sha256,
        "ranking_semantics": "query loss is timestamp x long; evaluation is pooled-global top-k", 
        "outputs": ["feature_coverage.parquet", "query_group_audit.parquet", "oos_rank_predictions.parquet", "four_way_comparison.parquet"],
    })


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, default=Path("data_perp/artifacts/tp6_m6_shared_regime_residual_features_20260809_v3/shared_regime_residual_ledger.parquet"))
    parser.add_argument("--identity-panel", type=Path, default=Path("data_perp/artifacts/full_universe_t2_t4_panel_20260801_v3/parts/*.parquet"))
    parser.add_argument("--mda-manifest", type=Path, default=Path("data_perp/artifacts/stage_i_base_selection_R3_tp6sl4_coverage90_20260804_v1/long/manifest.json"))
    parser.add_argument("--selector-contract", type=Path, default=Path("data_perp/artifacts/stage_i_selector_sample_20260803_v5/selector_feature_contract.json"))
    parser.add_argument("--feature-store", default="/Users/remyroche/Documents/Ares/data_perp/features/20260711_070000")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
