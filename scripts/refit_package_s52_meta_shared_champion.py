#!/usr/bin/env python3
"""Final-fit and package the retained S52 shared-meta champion for inference.

Feature selection and HPO are deliberately not run here.  The model is refit
through the latest resolved label using the frozen 51-feature contract, while
the native base/AE-GMM state and v9 tail95/MLP policy postprocessors are copied
from their validated artifacts.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import pickle
import shutil
import subprocess
import sys
import tempfile
import os
from pathlib import Path
from typing import Any, Sequence

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from extreme_price_movements.inference.s52_meta_ood import (
    append_s52_meta_ood_features,
    fit_s52_meta_ood_reference,
)
from extreme_price_movements.inference.s52_meta_score_alignment import (
    apply_s52_meta_score_alignment,
    fit_s52_meta_score_alignment,
    fit_paired_s52_meta_score_alignment,
)
from scripts.run_s52_train_meta_regime_handoff_smoke import (
    _add_fold_base_prior_features,
    _add_fold_reliability_features,
    _base_soft_label_target,
    _base_style_weights_for_soft_label,
    _candidate_column,
    _feature_contract_hash,
    _load_fixed_model_params,
    _load_fixed_selected_features,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HANDOFF = ROOT / (
    "data_perp/reports/s59_h5_fullthroughjul10_base_configfull_freshmda_"
    "fixedparams_wf30_20260713/meta_handoff_top30_allsafe_aegmmfull_20260713"
)
DEFAULT_CONTRACT = (
    ROOT
    / "extreme_price_movements"
    / "config"
    / "meta_v9_anchor_oldparams_residual_backbone_v1.json"
)
DEFAULT_NATIVE_PARENT = ROOT / "data_perp/artifacts/s59_s52_frozen_native_shadow_20260709"
DEFAULT_POSTPROCESSOR_PARENT = ROOT / "data_perp/artifacts/s59_s52_frozen_inference_bundle_v9_tail95_mlp_hierev_20260713"
DEFAULT_OUTPUT = ROOT / (
    "data_perp/artifacts/"
    "s59_s52_finalfit_meta_v9_anchor_oldparams_residual_backbone_v1_20260713"
)
DEFAULT_CHAMPION_OOF = ROOT / (
    "data_perp/reports/meta_v9_recovery_20260713/"
    "anchored_oldparams_fullhistory_oos_v1/"
    "s52_train_meta_regime_handoff_smoke_predictions.parquet"
)
DEFAULT_RESIDUAL_EVENT_STATE = ROOT / (
    "data_perp/reports/residual_event_archetype_true_base_oof_compactlocal_market_"
    "20260712_v3/revisions/2026-07-01/residual_event_state.joblib"
)


_FINAL_FIT_GENERATED_FEATURE_PREFIXES = (
    "meta_sel_ood_",
    "rel_rankband_",
    "rel_marginband_",
    "support_",
    "base_arch_hit_",
)

_FINAL_FIT_GENERATED_FEATURE_NAMES = {
    "base_margin_to_cutoff",
    "base_margin_to_cutoff_z",
    "base_signal_zscore_within_archetype",
    "base_score_rank_pct_train_prior",
    "base_rank_band",
    "base_margin_band",
}


def _selected_features_required_from_handoff(selected: list[str]) -> list[str]:
    """Return observable selected inputs that must exist in the source parquet.

    Fold priors, reliability values, hit-surprise values, and post-selection OOD
    summaries are generated after loading. Every other selected input is an
    observable feature contract and must not be silently synthesized as zero.
    """
    return [
        name
        for name in selected
        if name not in _FINAL_FIT_GENERATED_FEATURE_NAMES
        and not name.startswith(_FINAL_FIT_GENERATED_FEATURE_PREFIXES)
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _copytree(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    shutil.copytree(src, dst)


def _symlink(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.symlink_to(src.resolve(), target_is_directory=True)


def _build_final_matrix(
    data: pd.DataFrame,
    selected_features: list[str],
    *,
    ood_reference: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], pd.DataFrame]:
    """Build a causal-in-row training matrix and frozen OOD reference."""
    empty_valid = data.iloc[:0].copy()
    train, _ = _add_fold_base_prior_features(
        data, empty_valid, selected_col=_candidate_column("top30")
    )
    train, _ = _add_fold_reliability_features(train, empty_valid)
    # The retained 51-feature contract does not select support-drift or
    # hit-surprise columns. Avoid materializing their large intermediate
    # matrices during a final fit; this does not alter the fitted input matrix.
    ood_features = [c for c in selected_features if c.startswith("meta_sel_ood_")]
    pre_ood_features = [c for c in selected_features if c not in ood_features]
    # This is the numeric branch of ``_make_xy`` for a final fit with no
    # validation frame. The selected shared contract has no categorical inputs.
    x_train = (
        train.reindex(columns=pre_ood_features)
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
    )
    medians = x_train.median(numeric_only=True).fillna(0.0)
    x_train = x_train.fillna(medians).fillna(0.0).astype(np.float32)
    reference = (
        dict(ood_reference)
        if isinstance(ood_reference, dict) and ood_reference.get("enabled")
        else fit_s52_meta_ood_reference(x_train, pre_ood_features)
    )
    x_train = append_s52_meta_ood_features(
        x_train.reindex(columns=pre_ood_features),
        reference,
        output_features=ood_features,
    ).reindex(columns=selected_features, fill_value=0.0)
    return train, x_train.astype(np.float32, copy=False), reference, data


def _read_final_fit_frame(
    handoff_path: Path,
    ledger_path: Path,
    selected: list[str],
    *,
    allowed_missing_observable: Sequence[str] = (),
    min_joint_observable_coverage: float = 0.90,
    coverage_warmup_days: int = 30,
) -> pd.DataFrame:
    """Read only the immutable champion feature/label contract.

    The generic research join intentionally loads the entire exploratory
    universe. That is appropriate for feature selection, but it is wasteful
    and can exceed memory during a fixed-contract final refit.
    """
    import pyarrow.parquet as pq

    key_columns = ["__ts__", "__symbol__", "side_name"]
    ood = {name for name in selected if name.startswith("meta_sel_ood_")}
    handoff_schema = set(pq.read_schema(handoff_path).names)
    required_observable = _selected_features_required_from_handoff(selected)
    allowed_missing = {str(name) for name in allowed_missing_observable}
    missing_observable = sorted(
        set(required_observable).difference(handoff_schema).difference(allowed_missing)
    )
    if missing_observable:
        preview = ", ".join(missing_observable[:12])
        suffix = "" if len(missing_observable) <= 12 else f", ... (+{len(missing_observable) - 12})"
        raise RuntimeError(
            "Final-fit handoff is missing selected observable features; refusing "
            "to replace them with all-zero columns. "
            f"handoff={handoff_path} missing={preview}{suffix}"
        )
    explicitly_missing = sorted(
        set(required_observable).difference(handoff_schema).intersection(allowed_missing)
    )
    if explicitly_missing:
        print(
            "[final-fit] explicitly retaining constant-missing compatibility "
            f"features: {', '.join(explicitly_missing)}",
            flush=True,
        )
    handoff_columns = set(key_columns) | {"score", "selected_top30"}
    handoff_columns.update(name for name in selected if name not in ood)
    handoff_columns.update(
        {
            "source_tag",
            "source_semantic_family",
            "policy_archetype",
            "archetype_policy_key",
            "archetype_label_family",
            "__archetype_policy_key__",
        }
    )
    handoff = pd.read_parquet(
        handoff_path, columns=sorted(handoff_columns.intersection(handoff_schema))
    )
    ledger_schema = set(pq.read_schema(ledger_path).names)
    ledger_columns = set(key_columns) | {
        "__first_touch_target_soft__",
        "clean_exec",
        "full_path_bad_mae_1r",
        "timeout",
        "dirty_positive",
        "exec_margin",
        "u_policy_net",
        "mae_norm",
        "first_touch_full_path_mae_norm",
        "mfe_norm",
        "first_touch_full_path_mfe_norm",
        "first_touch_bar",
        "__archetype_policy_max_barrier__",
    }
    ledger = pd.read_parquet(
        ledger_path, columns=sorted(ledger_columns.intersection(ledger_schema))
    )
    aligned = len(handoff) == len(ledger) and all(
        handoff[col].reset_index(drop=True).equals(ledger[col].reset_index(drop=True))
        for col in key_columns
    )
    if aligned:
        for col in ledger.columns:
            if col not in key_columns and col not in handoff.columns:
                handoff[col] = ledger[col].to_numpy(copy=False)
        out = handoff
    else:
        out = handoff.merge(ledger, on=key_columns, how="left", validate="one_to_one")
    out["selected_top30"] = out["selected_top30"].fillna(False).astype(bool)
    coverage_columns = sorted(
        set(required_observable)
        .intersection(out.columns)
        .difference(allowed_missing)
    )
    coverage_rows = out.loc[out["selected_top30"]]
    if "__ts__" in coverage_rows.columns and int(coverage_warmup_days) > 0:
        coverage_ts = pd.to_datetime(
            coverage_rows["__ts__"], utc=True, errors="coerce"
        )
        first_ts = coverage_ts.min()
        if pd.notna(first_ts):
            coverage_rows = coverage_rows.loc[
                coverage_ts.ge(first_ts + pd.Timedelta(days=int(coverage_warmup_days)))
            ]
    if coverage_columns and not coverage_rows.empty:
        numeric = coverage_rows[coverage_columns].apply(
            pd.to_numeric, errors="coerce"
        ).replace([np.inf, -np.inf], np.nan)
        individual = numeric.notna().mean().sort_values()
        complete = numeric.notna().all(axis=1)
        joint_coverage = float(complete.mean())
        minimum = float(np.clip(min_joint_observable_coverage, 0.0, 1.0))
        period_coverage = pd.Series(dtype="float64")
        if "__ts__" in coverage_rows.columns:
            periods = pd.to_datetime(
                coverage_rows["__ts__"], utc=True, errors="coerce"
            ).dt.strftime("%Y-%m")
            period_coverage = complete.groupby(periods, dropna=True).mean().sort_index()
        print(
            "[final-fit] observable feature coverage "
            f"rows={len(numeric):,} features={len(coverage_columns)} "
            f"joint={joint_coverage:.6f} required={minimum:.6f}",
            flush=True,
        )
        failed_periods = period_coverage.loc[period_coverage.lt(minimum)]
        if joint_coverage < minimum or not failed_periods.empty:
            worst = ", ".join(
                f"{name}={value:.3%}"
                for name, value in individual.head(12).items()
            )
            period_text = ", ".join(
                f"{period}={value:.3%}"
                for period, value in failed_periods.items()
            )
            raise RuntimeError(
                "Final-fit handoff fails the joint observable-feature coverage "
                "contract after the warm-up exclusion. Refusing to median-fill "
                "a structurally incomplete feature join. "
                f"joint={joint_coverage:.3%} required={minimum:.3%}; "
                f"failed calendar months: {period_text or 'none'}; "
                f"worst individual coverage: {worst}"
            )
    out["clean_exec_label"] = (
        pd.to_numeric(out.get("clean_exec"), errors="coerce").fillna(0.0).gt(0.5)
    ).astype(np.float32)
    out["positive_exec_margin"] = (
        pd.to_numeric(out.get("exec_margin"), errors="coerce").fillna(0.0).gt(0.0)
    ).astype(np.float32)
    return out


def _compact_reliability_priors(frame: pd.DataFrame, *, shrinkage_k: float = 60.0) -> dict[str, Any]:
    """Build the live reliability payload without copying the full handoff.

    It is algebraically the same side/archetype/band aggregation as the live
    helper, but vectorized for a 1M+ row final refit.
    """
    side = frame["side_name"].astype(str).str.lower()
    arch = frame.get("source_tag", side + "__unknown").astype(str)
    score = pd.to_numeric(frame["score"], errors="coerce")
    selected = frame["selected_top30"].astype(bool)
    clean = pd.to_numeric(frame.get("clean_exec"), errors="coerce").fillna(0.0).clip(0.0, 1.0)
    bad = pd.to_numeric(frame.get("full_path_bad_mae_1r"), errors="coerce").fillna(0.0).clip(0.0, 1.0)
    timeout = pd.to_numeric(frame.get("timeout"), errors="coerce").fillna(0.0).clip(0.0, 1.0)
    dirty = pd.to_numeric(frame.get("dirty_positive"), errors="coerce").fillna(0.0).clip(0.0, 1.0)
    exec_margin = pd.to_numeric(frame.get("exec_margin"), errors="coerce").fillna(0.0)
    global_cutoff = float(score.loc[selected & score.notna()].min())
    global_mean = float(score.mean())
    global_std = float(score.std())
    if not np.isfinite(global_std) or global_std <= 1e-12:
        global_std = 1.0
    key = side + "|" + arch
    stats_source = pd.DataFrame({"key": key, "score": score, "selected": selected})
    all_stats = stats_source.groupby("key", sort=False)["score"].agg(["size", "mean", "std"])
    selected_min = stats_source.loc[selected].groupby("key", sort=False)["score"].min()
    group_q70 = stats_source.groupby("key", sort=False)["score"].quantile(0.70)
    all_stats["cutoff"] = selected_min.reindex(all_stats.index)
    all_stats["cutoff"] = all_stats["cutoff"].fillna(group_q70.reindex(all_stats.index)).fillna(global_cutoff)
    all_stats["mean"] = all_stats["mean"].fillna(global_mean)
    all_stats["std"] = all_stats["std"].where(all_stats["std"] > 1e-12, global_std).fillna(global_std)
    cutoff = key.map(all_stats["cutoff"]).astype(np.float64)
    margin = score - cutoff
    score_edges = [float(v) for v in np.unique(score.dropna().quantile([0.2, 0.4, 0.6, 0.8]).to_numpy())]
    score_reference_quantiles = np.quantile(
        score.dropna().to_numpy(dtype=np.float64, copy=False),
        np.linspace(0.0, 1.0, 4097),
    ).astype(float).tolist()
    margin_edges = [float(v) for v in np.unique(margin.dropna().quantile([0.2, 0.4, 0.6, 0.8]).to_numpy())]
    rank_band = pd.Series(
        "base_rank_band__q" + np.searchsorted(score_edges, score.fillna(-np.inf).to_numpy(), side="right").astype(str),
        index=frame.index,
    )
    margin_band = pd.Series(
        "base_margin_band__q" + np.searchsorted(margin_edges, margin.fillna(-np.inf).to_numpy(), side="right").astype(str),
        index=frame.index,
    )
    global_prior = {
        "cutoff": global_cutoff,
        "score_mean": global_mean,
        "score_std": global_std,
        "clean_rate": float(clean.mean()),
        "bad_mae_rate": float(bad.mean()),
        "timeout_rate": float(timeout.mean()),
        "dirty_positive_rate": float(dirty.mean()),
        "exec_margin_mean": float(exec_margin.mean()),
    }
    stat = pd.DataFrame({
        "side": side, "arch": arch, "rank_band": rank_band, "margin_band": margin_band,
        "clean": clean, "bad": bad, "timeout": timeout, "dirty": dirty, "exec": exec_margin,
    })

    def aggregate(prefix: str, band: str) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for grouping, wildcard in ((["side", "arch", band], False), (["side", band], True), ([band], True)):
            grouped = stat.groupby(grouping, sort=False).agg(
                rows=("clean", "size"), clean_rate=("clean", "mean"), bad_mae_rate=("bad", "mean"),
                timeout_rate=("timeout", "mean"), dirty_positive_rate=("dirty", "mean"), exec_margin_mean=("exec", "mean"),
            )
            for values, row in grouped.iterrows():
                values = values if isinstance(values, tuple) else (values,)
                if len(grouping) == 3:
                    side_value, arch_value, band_value = map(str, values)
                elif len(grouping) == 2:
                    side_value, band_value = map(str, values); arch_value = "*"
                else:
                    band_value = str(values[0]); side_value = arch_value = "*"
                rows = float(row["rows"])
                weight = rows / (rows + float(shrinkage_k))
                values_out = {"rows": rows}
                for metric, global_name in (
                    ("clean_rate", "clean_rate"), ("bad_mae_rate", "bad_mae_rate"),
                    ("timeout_rate", "timeout_rate"), ("dirty_positive_rate", "dirty_positive_rate"),
                    ("exec_margin_mean", "exec_margin_mean"),
                ):
                    values_out[metric] = weight * float(row[metric]) + (1.0 - weight) * float(global_prior[global_name])
                out[f"{prefix}|{side_value}|{arch_value}|{band_value}"] = values_out
        return out

    side_arch_priors = {
        str(name): {
            "cutoff": float(row["cutoff"]), "mean": float(row["mean"]), "std": float(row["std"]), "rows": int(row["size"]),
        }
        for name, row in all_stats.iterrows()
    }
    groups = aggregate("rel_rankband", "rank_band")
    groups.update(aggregate("rel_marginband", "margin_band"))
    return {
        "schema": "s52_meta_reliability_priors_v1",
        "rows": int(len(frame)), "selected_col": "selected_top30", "shrinkage_k": float(shrinkage_k),
        "feature_names": [
            "rel_rankband_rows_log1p", "rel_rankband_clean_rate", "rel_rankband_bad_mae_rate", "rel_rankband_timeout_rate", "rel_rankband_dirty_positive_rate", "rel_rankband_exec_margin_mean", "rel_rankband_edge",
            "rel_marginband_rows_log1p", "rel_marginband_clean_rate", "rel_marginband_bad_mae_rate", "rel_marginband_timeout_rate", "rel_marginband_dirty_positive_rate", "rel_marginband_exec_margin_mean", "rel_marginband_edge",
        ],
        "score_quantile_edges": score_edges,
        "score_reference_quantiles": score_reference_quantiles,
        "margin_quantile_edges": margin_edges,
        "source_tag_score_thresholds": {}, "global_prior": global_prior,
        "side_arch_priors": side_arch_priors, "groups": groups,
        "leakage_contract": {"fit_scope": "final resolved training rows only", "oos_usage": "frozen priors", "realized_outcomes": "never read from live rows"},
    }


def _write_contract(
    output: Path, selected: list[str], model_keys: list[str], source: str
) -> None:
    mapping = {f"f{i}": name for i, name in enumerate(selected)}
    payload = {
        "schema_version": "meta_feature_contract_v1",
        "run_id": output.name,
        "generated_by": "refit_package_s52_meta_shared_champion",
        "meta_models": {
            key: {
                "model_key": key,
                "feature_columns": selected,
                "n_features": len(selected),
                "feature_contract_hash": _feature_contract_hash(selected),
                "positional_feature_mapping": mapping,
                "source": source,
            }
            for key in model_keys
        },
    }
    target = output / "meta_oof/meta_feature_contract.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _fit_native_final_booster(
    values: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    feature_names: list[str],
    shards: list[tuple[int, int]],
    params: dict[str, Any],
    *,
    seed: int,
) -> lgb.Booster:
    """Fit the exact shared-head learner while releasing raw data before boost."""
    if len(labels) < 50:
        raise RuntimeError("Insufficient valid soft-label rows for final fit")
    native_params = {
        "objective": "regression",
        "learning_rate": float(params["learning_rate"]),
        "num_leaves": int(params["num_leaves"]),
        "max_depth": int(params["max_depth"]),
        "min_child_samples": int(params["min_child_samples"]),
        "feature_fraction": float(params["colsample_bytree"]),
        "bagging_fraction": float(params["subsample"]),
        "bagging_freq": 1,
        "lambda_l1": float(params["reg_alpha"]),
        "lambda_l2": float(params["reg_lambda"]),
        "seed": int(seed),
        "num_threads": 1,
        "verbosity": -1,
        # Avoid LightGBM's parallel auto-probe allocating an extra histogram
        # matrix at this final full-data fit scale.
        "force_col_wise": True,
        # These are resource controls, not HPO dimensions. They keep every
        # training row while bounding the full-data histogram allocation.
        "max_bin": 63,
        "histogram_pool_size": 8,
    }
    if not shards:
        shards = [(0, len(labels))]
    total_trees = int(params["n_estimators"])
    base_trees, remainder = divmod(total_trees, len(shards))
    booster: lgb.Booster | None = None
    checkpoint = Path(tempfile.gettempdir()) / f"s52_finalfit_booster_{seed}.txt"
    for idx, (start, end) in enumerate(shards):
        trees = base_trees + (1 if idx < remainder else 0)
        if trees <= 0 or end <= start:
            continue
        print(
            f"[final-fit] boosting shard {idx + 1}/{len(shards)} rows={end - start:,} trees={trees}",
            flush=True,
        )
        dataset = lgb.Dataset(
            values[start:end],
            label=labels[start:end],
            weight=weights[start:end],
            feature_name=feature_names,
            free_raw_data=True,
        )
        booster = lgb.train(
            native_params,
            dataset,
            num_boost_round=trees,
            init_model=booster,
            # Retain the predictor only. The next shard can still use it as
            # ``init_model`` while the previous shard's Dataset is released.
            keep_training_booster=False,
        )
        del dataset
        # Native LightGBM can retain Dataset state through an ``init_model``
        # chain. Reloading a compact text model releases that state between
        # chronological shards without changing the accumulated trees.
        booster.save_model(str(checkpoint))
        del booster
        gc.collect()
        booster = lgb.Booster(model_file=str(checkpoint))
    checkpoint.unlink(missing_ok=True)
    if booster is None:
        raise RuntimeError("Chronological final refit did not fit any booster shard")
    return booster


def _native_final_params(params: dict[str, Any], *, seed: int) -> dict[str, Any]:
    """Return the fixed champion parameters plus bounded-fit resource controls."""
    return {
        "objective": "regression",
        "learning_rate": float(params["learning_rate"]),
        "num_leaves": int(params["num_leaves"]),
        "max_depth": int(params["max_depth"]),
        "min_child_samples": int(params["min_child_samples"]),
        "feature_fraction": float(params["colsample_bytree"]),
        "bagging_fraction": float(params["subsample"]),
        "bagging_freq": 1,
        "lambda_l1": float(params["reg_alpha"]),
        "lambda_l2": float(params["reg_lambda"]),
        "seed": int(seed),
        "num_threads": 1,
        "verbosity": -1,
        "force_col_wise": True,
        "max_bin": 63,
        "histogram_pool_size": 8,
    }


def _fit_native_chunk(args: argparse.Namespace) -> None:
    """Fit one chronological chunk in a separate process.

    LightGBM holds native histogram/Dataset memory outside Python's garbage
    collector. Running each chunk in its own process is the reliable way to
    keep the all-row final fit bounded on the production workstation.
    """
    shape = tuple(int(value) for value in args.matrix_shape.split(","))
    values = np.lib.format.open_memmap(
        args.matrix_path, mode="r", dtype=np.float32, shape=shape
    )
    labels = np.load(args.labels_path, mmap_mode="r")
    weights = np.load(args.weights_path, mmap_mode="r")
    feature_names = json.loads(args.features_path.read_text(encoding="utf-8"))
    params = json.loads(args.params_path.read_text(encoding="utf-8"))
    start, end = int(args.chunk_start), int(args.chunk_end)
    dataset = lgb.Dataset(
        values[start:end],
        label=labels[start:end],
        weight=weights[start:end],
        feature_name=feature_names,
        free_raw_data=True,
    )
    init_model = str(args.checkpoint) if args.checkpoint.exists() else None
    booster = lgb.train(
        params,
        dataset,
        num_boost_round=int(args.chunk_trees),
        init_model=init_model,
        keep_training_booster=False,
    )
    booster.save_model(str(args.checkpoint))


def _fit_native_final_booster_isolated(
    values_path: Path,
    values_shape: tuple[int, int],
    labels: np.ndarray,
    weights: np.ndarray,
    feature_names: list[str],
    shards: list[tuple[int, int]],
    params: dict[str, Any],
    *,
    seed: int,
) -> lgb.Booster:
    """Train all chronological shards with process-level native-memory isolation."""
    if len(labels) < 50:
        raise RuntimeError("Insufficient valid soft-label rows for final fit")
    work_dir = Path(tempfile.mkdtemp(prefix=f"s52_finalfit_{seed}_"))
    labels_path = work_dir / "labels.npy"
    weights_path = work_dir / "weights.npy"
    features_path = work_dir / "features.json"
    params_path = work_dir / "params.json"
    checkpoint = work_dir / "booster.txt"
    np.save(labels_path, np.asarray(labels, dtype=np.float32))
    np.save(weights_path, np.asarray(weights, dtype=np.float32))
    features_path.write_text(json.dumps(feature_names), encoding="utf-8")
    params_path.write_text(
        json.dumps(_native_final_params(params, seed=seed), sort_keys=True),
        encoding="utf-8",
    )
    total_trees = int(params["n_estimators"])
    base_trees, remainder = divmod(total_trees, len(shards))
    try:
        for idx, (start, end) in enumerate(shards):
            trees = base_trees + (1 if idx < remainder else 0)
            if trees <= 0 or end <= start:
                continue
            print(
                f"[final-fit] isolated shard {idx + 1}/{len(shards)} rows={end - start:,} trees={trees}",
                flush=True,
            )
            command = [
                sys.executable, str(Path(__file__).resolve()), "--fit-chunk",
                "--matrix-path", str(values_path),
                "--matrix-shape", f"{values_shape[0]},{values_shape[1]}",
                "--labels-path", str(labels_path),
                "--weights-path", str(weights_path),
                "--features-path", str(features_path),
                "--params-path", str(params_path),
                "--checkpoint", str(checkpoint),
                "--chunk-start", str(start), "--chunk-end", str(end),
                "--chunk-trees", str(trees),
            ]
            child_env = os.environ.copy()
            current_pythonpath = child_env.get("PYTHONPATH", "")
            child_env["PYTHONPATH"] = (
                f"{ROOT}{os.pathsep}{current_pythonpath}"
                if current_pythonpath
                else str(ROOT)
            )
            child_env.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
            subprocess.run(command, check=True, cwd=ROOT, env=child_env)
        if not checkpoint.exists():
            raise RuntimeError("Isolated final refit did not produce a booster checkpoint")
        return lgb.Booster(model_file=str(checkpoint))
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def _resolve_final_fit_shards(
    *,
    mode: str,
    row_count: int,
    chronological_shards: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Choose the statistical fit contract independently of memory isolation."""
    if str(mode) == "full_dataset":
        return [(0, int(row_count))]
    if str(mode) == "chronological_shards_legacy":
        return list(chronological_shards)
    raise ValueError(f"Unsupported final fit mode: {mode}")


def _install_model(
    native_parent: Path,
    output: Path,
    model: Any,
    selected: list[str],
    ood: dict[str, Any],
    score_alignment: dict[str, Any] | None = None,
    contract_id: str = "meta_v9_recovered_global_hpo150_v1",
) -> None:
    models_dir = output / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    with (native_parent / "models/trained_state.pkl").open("rb") as handle:
        state = pickle.load(handle)
    # The final contract is one shared global head. Keep a single model object
    # under both routing keys instead of deep-copying LightGBM at peak memory.
    bundle = dict(state.get("bundle", {}))
    meta_models = dict(bundle.get("meta_models", {}))
    model.selected_features = list(selected)
    model.feature_columns = list(selected)
    model.input_feature_names = list(selected)
    model.s52_meta_ood_reference_ = dict(ood)
    model.s52_meta_ood_enabled_ = bool(ood.get("enabled"))
    model.s52_meta_ood_input_features_ = [
        feature for feature in selected if not feature.startswith("meta_sel_ood_")
    ]
    if score_alignment is not None:
        model.s52_meta_score_alignment_ = dict(score_alignment)
    for key in ("long_s52_meta_threshold_handoff", "short_s52_meta_threshold_handoff"):
        meta_models[key] = model
        meta_dir = models_dir / "meta" / key
        meta_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, meta_dir / "base_soft_label.joblib", compress=3)
    bundle["meta_models"] = meta_models
    state["bundle"] = bundle
    state["run_id"] = output.name
    state["source_run_ids"] = [native_parent.name]
    with (models_dir / "trained_state.pkl").open("wb") as handle:
        pickle.dump(state, handle, protocol=pickle.HIGHEST_PROTOCOL)
    joblib.dump(
        {"run_id": output.name, "bundle": {"meta_models": meta_models}},
        models_dir / "model_state_meta.pkl",
        compress=3,
    )
    _write_contract(
        output,
        selected,
        sorted(meta_models),
        source=f"{contract_id} final refit",
    )


def _load_champion_oof_scores(path: Path) -> dict[str, np.ndarray]:
    """Return retained champion OOF raw-score references by side."""
    raw = pd.read_parquet(path, columns=["side_name", "score_meta_base_soft_label"])
    out: dict[str, np.ndarray] = {}
    for side, group in raw.groupby(raw["side_name"].astype(str).str.lower(), sort=False):
        values = pd.to_numeric(
            group["score_meta_base_soft_label"], errors="coerce"
        ).to_numpy(dtype=np.float32, copy=False)
        values = values[np.isfinite(values)]
        if values.size:
            out[str(side)] = values
    if not out:
        raise RuntimeError(f"Champion OOF source contains no finite scores: {path}")
    return out


def _load_champion_oof_frame(path: Path) -> pd.DataFrame:
    """Load keyed champion OOF scores for a same-row alignment calibration."""
    columns = ["__ts__", "__symbol__", "side_name", "score_meta_base_soft_label"]
    raw = pd.read_parquet(path, columns=columns)
    raw["__ts__"] = pd.to_datetime(raw["__ts__"], utc=True, errors="coerce")
    raw["__symbol__"] = raw["__symbol__"].astype(str)
    raw["side_name"] = raw["side_name"].astype(str).str.lower()
    raw["score_meta_base_soft_label"] = pd.to_numeric(
        raw["score_meta_base_soft_label"], errors="coerce"
    )
    raw = raw.loc[
        raw["__ts__"].notna()
        & np.isfinite(raw["score_meta_base_soft_label"].to_numpy(dtype=np.float64))
    ]
    return raw.drop_duplicates(
        ["__ts__", "__symbol__", "side_name"], keep="last"
    ).reset_index(drop=True)


def _score_matrix_chunked(model: Any, matrix: Any, *, chunk_rows: int = 50_000) -> np.ndarray:
    scores = np.empty(len(matrix), dtype=np.float32)
    for start in range(0, len(matrix), int(chunk_rows)):
        end = min(start + int(chunk_rows), len(matrix))
        chunk = matrix.iloc[start:end] if isinstance(matrix, pd.DataFrame) else matrix[start:end]
        values = (
            chunk.to_numpy(dtype=np.float32, copy=False)
            if isinstance(chunk, pd.DataFrame)
            else np.asarray(chunk, dtype=np.float32)
        )
        scores[start:end] = model.predict(
            values
        ).astype(np.float32, copy=False)
    return scores


def _persist_score_alignment(
    output: Path,
    *,
    model: Any,
    matrix: pd.DataFrame,
    sides: pd.Series,
    champion_oof: Path,
    knot_count: int = 4097,
) -> dict[str, Any]:
    """Fit and persist the score bridge plus live empirical rank references."""
    champion = _load_champion_oof_scores(champion_oof)
    final_scores = _score_matrix_chunked(model, matrix)
    side_values = sides.astype(str).str.lower().to_numpy(dtype=str, copy=False)
    final_by_side = {
        side: final_scores[side_values == side]
        for side in sorted(set(side_values))
    }
    alignment = fit_s52_meta_score_alignment(
        final_by_side,
        champion,
        knot_count=int(knot_count),
    )
    alignment["champion_oof_source"] = str(champion_oof)
    alignment["final_fit_source"] = "all_resolved_top30_rows"
    alignment_path = output / "policy_params/s52_meta_score_alignment.json"
    alignment_path.parent.mkdir(parents=True, exist_ok=True)
    alignment_path.write_text(
        json.dumps(_json_safe(alignment), indent=2, sort_keys=True), encoding="utf-8"
    )
    meta_oof = output / "meta_oof"
    meta_oof.mkdir(parents=True, exist_ok=True)
    for side, values in champion.items():
        # ``_historical_prediction_rank_pct`` uses this same OOF score domain
        # after ModelOrchestrator applies the bridge above.
        reference = pd.DataFrame({"score": np.asarray(values, dtype=np.float32)})
        reference.to_parquet(
            meta_oof / f"meta_score_reference_{side}_s52_meta_threshold_handoff.parquet",
            index=False,
        )
    return alignment


def _persist_oos_score_alignment(
    output: Path,
    *,
    source_rows: pd.DataFrame,
    source_scores: np.ndarray,
    champion_oof: Path,
    train_end_exclusive: pd.Timestamp,
    knot_count: int = 4097,
) -> dict[str, Any]:
    """Fit the production bridge from genuinely OOS checkpoint predictions.

    Mapping final-fit *training* scores into an OOF score domain compresses live
    predictions whenever the final learner has an in-sample score shift.  A
    checkpoint trained strictly before ``train_end_exclusive`` provides the
    score distribution the final refit is expected to emit on unseen rows.
    The target uses the champion OOF score on the exact same rows.
    """
    keys = ["__ts__", "__symbol__", "side_name"]
    source = source_rows.loc[:, keys].copy()
    source["checkpoint_oos_score"] = np.asarray(source_scores, dtype=np.float32)
    source["__ts__"] = pd.to_datetime(source["__ts__"], utc=True, errors="coerce")
    source["__symbol__"] = source["__symbol__"].astype(str)
    source["side_name"] = source["side_name"].astype(str).str.lower()
    champion_frame = _load_champion_oof_frame(champion_oof)
    paired = source.merge(champion_frame, on=keys, how="inner", validate="one_to_one")
    finite = (
        np.isfinite(pd.to_numeric(paired["checkpoint_oos_score"], errors="coerce"))
        & np.isfinite(
            pd.to_numeric(paired["score_meta_base_soft_label"], errors="coerce")
        )
    )
    paired = paired.loc[finite].reset_index(drop=True)
    if len(paired) < 128:
        raise RuntimeError(
            "OOS score alignment has insufficient same-row champion overlap: "
            f"rows={len(paired)}"
        )
    source_by_side: dict[str, np.ndarray] = {}
    target_by_side: dict[str, np.ndarray] = {}
    for side, group in paired.groupby("side_name", sort=True):
        source_by_side[str(side)] = group["checkpoint_oos_score"].to_numpy(
            dtype=np.float32, copy=False
        )
        target_by_side[str(side)] = group["score_meta_base_soft_label"].to_numpy(
            dtype=np.float32, copy=False
        )
    alignment = fit_s52_meta_score_alignment(
        source_by_side,
        target_by_side,
        knot_count=int(knot_count),
    )
    alignment["champion_oof_source"] = str(champion_oof)
    alignment["final_fit_source"] = "pre_cutoff_checkpoint_oos_same_row"
    alignment["source_train_end_exclusive"] = train_end_exclusive.isoformat()
    alignment["source_oos_start"] = paired["__ts__"].min().isoformat()
    alignment["source_oos_end"] = paired["__ts__"].max().isoformat()
    alignment["source_oos_rows"] = int(len(paired))
    alignment["leakage_contract"] = {
        "source_model_fit": f"rows strictly before {train_end_exclusive.isoformat()}",
        "source_scores": "checkpoint predictions on rows at/after cutoff",
        "target_scores": "matching-row champion OOF predictions",
        "final_refit_scores_used_for_alignment": False,
    }
    alignment_path = output / "policy_params/s52_meta_score_alignment.json"
    alignment_path.parent.mkdir(parents=True, exist_ok=True)
    alignment_path.write_text(
        json.dumps(_json_safe(alignment), indent=2, sort_keys=True), encoding="utf-8"
    )
    # Rank references remain the full champion OOF distributions.  Only the
    # bridge source changes from in-sample final-fit scores to checkpoint OOS.
    full_champion = _load_champion_oof_scores(champion_oof)
    meta_oof = output / "meta_oof"
    meta_oof.mkdir(parents=True, exist_ok=True)
    for side, values in full_champion.items():
        pd.DataFrame({"score": np.asarray(values, dtype=np.float32)}).to_parquet(
            meta_oof / f"meta_score_reference_{side}_s52_meta_threshold_handoff.parquet",
            index=False,
        )
    paired.to_parquet(
        meta_oof / "meta_score_alignment_oos_pairs.parquet",
        index=False,
        compression="zstd",
    )
    return alignment


def _persist_paired_score_alignment(
    output: Path,
    *,
    source_rows: pd.DataFrame,
    source_scores: np.ndarray,
    champion_oof: Path,
    calibration_end_exclusive: pd.Timestamp,
) -> dict[str, Any]:
    """Fit a same-row monotonic bridge and audit it on later paired rows."""
    keys = ["__ts__", "__symbol__", "side_name"]
    source = source_rows.loc[:, keys].copy()
    source["final_refit_score"] = np.asarray(source_scores, dtype=np.float32)
    source["__ts__"] = pd.to_datetime(source["__ts__"], utc=True, errors="coerce")
    source["__symbol__"] = source["__symbol__"].astype(str)
    source["side_name"] = source["side_name"].astype(str).str.lower()
    paired = source.merge(
        _load_champion_oof_frame(champion_oof),
        on=keys,
        how="inner",
        validate="one_to_one",
    )
    finite = np.isfinite(paired["final_refit_score"].to_numpy(dtype=np.float64)) & np.isfinite(
        paired["score_meta_base_soft_label"].to_numpy(dtype=np.float64)
    )
    paired = paired.loc[finite].sort_values(keys, kind="mergesort").reset_index(drop=True)
    fit = paired.loc[paired["__ts__"].lt(calibration_end_exclusive)]
    audit = paired.loc[paired["__ts__"].ge(calibration_end_exclusive)].copy()
    if len(fit) < 1_000 or len(audit) < 128:
        raise RuntimeError(
            "Paired score alignment has insufficient chronological support: "
            f"fit={len(fit)} audit={len(audit)} cutoff={calibration_end_exclusive}"
        )
    source_by_side = {
        str(side): group["final_refit_score"].to_numpy(dtype=np.float32, copy=False)
        for side, group in fit.groupby("side_name", sort=True)
    }
    target_by_side = {
        str(side): group["score_meta_base_soft_label"].to_numpy(dtype=np.float32, copy=False)
        for side, group in fit.groupby("side_name", sort=True)
    }
    alignment = fit_paired_s52_meta_score_alignment(source_by_side, target_by_side)
    alignment.update(
        {
            "champion_oof_source": str(champion_oof),
            "final_fit_source": "final_refit_same_row_champion_oof_distillation",
            "calibration_end_exclusive": calibration_end_exclusive.isoformat(),
            "calibration_rows": int(len(fit)),
            "audit_rows": int(len(audit)),
            "leakage_contract": {
                "bridge_fit_rows": f"timestamps before {calibration_end_exclusive.isoformat()}",
                "bridge_audit_rows": f"timestamps at/after {calibration_end_exclusive.isoformat()}",
                "target_scores": "matching-row champion OOF predictions; no realized outcomes",
                "source_scores": "frozen final-refit model predictions",
            },
        }
    )
    mapped = np.empty(len(audit), dtype=np.float32)
    for side, index in audit.groupby("side_name", sort=False).groups.items():
        loc = audit.index.get_indexer(index)
        mapped[loc] = apply_s52_meta_score_alignment(
            audit.loc[index, "final_refit_score"].to_numpy(dtype=np.float32),
            alignment,
            side=str(side),
        )
    audit["aligned_score"] = mapped
    target = audit["score_meta_base_soft_label"].to_numpy(dtype=np.float64)
    raw = audit["final_refit_score"].to_numpy(dtype=np.float64)
    aligned = audit["aligned_score"].to_numpy(dtype=np.float64)
    audit_metrics = {
        "rows": int(len(audit)),
        "raw_mae": float(np.mean(np.abs(raw - target))),
        "aligned_mae": float(np.mean(np.abs(aligned - target))),
        "raw_spearman": float(pd.Series(raw).corr(pd.Series(target), method="spearman")),
        "aligned_spearman": float(pd.Series(aligned).corr(pd.Series(target), method="spearman")),
    }
    alignment["chronological_audit"] = audit_metrics
    alignment_path = output / "policy_params/s52_meta_score_alignment.json"
    alignment_path.parent.mkdir(parents=True, exist_ok=True)
    alignment_path.write_text(
        json.dumps(_json_safe(alignment), indent=2, sort_keys=True), encoding="utf-8"
    )
    meta_oof = output / "meta_oof"
    meta_oof.mkdir(parents=True, exist_ok=True)
    full_champion = _load_champion_oof_scores(champion_oof)
    for side, values in full_champion.items():
        pd.DataFrame({"score": np.asarray(values, dtype=np.float32)}).to_parquet(
            meta_oof / f"meta_score_reference_{side}_s52_meta_threshold_handoff.parquet",
            index=False,
        )
    audit.to_parquet(
        meta_oof / "meta_score_alignment_paired_audit.parquet",
        index=False,
        compression="zstd",
    )
    return alignment


def _attach_score_alignment_to_artifact(
    output: Path, alignment: dict[str, Any]
) -> None:
    """Attach one immutable score bridge to both shared routing aliases."""
    models_dir = output / "models"
    state_path = models_dir / "trained_state.pkl"
    with state_path.open("rb") as handle:
        state = pickle.load(handle)
    bundle = dict(state.get("bundle", {}))
    meta_models = dict(bundle.get("meta_models", {}))
    route_keys = (
        "long_s52_meta_threshold_handoff",
        "short_s52_meta_threshold_handoff",
    )
    shared_model = next(
        (meta_models[key] for key in route_keys if key in meta_models), None
    )
    if shared_model is None:
        raise RuntimeError("Final artifact is missing the shared S52 meta routes")
    shared_model.s52_meta_score_alignment_ = dict(alignment)
    for key in route_keys:
        meta_models[key] = shared_model
        target = models_dir / "meta" / key / "base_soft_label.joblib"
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(shared_model, target, compress=3)
    bundle["meta_models"] = meta_models
    state["bundle"] = bundle
    with state_path.open("wb") as handle:
        pickle.dump(state, handle, protocol=pickle.HIGHEST_PROTOCOL)
    joblib.dump(
        {"run_id": output.name, "bundle": {"meta_models": meta_models}},
        models_dir / "model_state_meta.pkl",
        compress=3,
    )


def _refresh_score_alignment_only(args: argparse.Namespace) -> None:
    """Build score-domain references without retraining the final booster."""
    output = args.output
    state_path = output / "models/trained_state.pkl"
    if not state_path.exists():
        raise RuntimeError(f"Final artifact is missing: {state_path}")
    selected = _load_fixed_selected_features(args.contract)
    with state_path.open("rb") as handle:
        state = pickle.load(handle)
    model = (state.get("bundle", {}).get("meta_models", {}) or {}).get(
        "long_s52_meta_threshold_handoff"
    )
    if model is None:
        raise RuntimeError("Final artifact has no long shared S52 meta model")
    handoff = args.handoff_dir / "train_meta_regime_handoff.parquet"
    ledger = args.handoff_dir / "s52_trailing_regime_scored_ledger.parquet"
    print("[score-alignment] rebuilding exact fixed-contract final-fit matrix", flush=True)
    data = _read_final_fit_frame(
        handoff,
        ledger,
        list(selected),
        allowed_missing_observable=args.allow_missing_observable_feature,
    )
    data = data.loc[data["selected_top30"].fillna(False).astype(bool)].copy()
    data["__ts__"] = pd.to_datetime(data["__ts__"], utc=True, errors="coerce")
    data = data.loc[data["__ts__"].notna()].sort_values(
        ["__ts__", "__symbol__", "side_name"], kind="mergesort"
    ).reset_index(drop=True)
    train, matrix, _, _ = _build_final_matrix(
        data,
        list(selected),
        ood_reference=getattr(model, "s52_meta_ood_reference_", None),
    )
    if args.score_alignment_paired_cutoff:
        cutoff = pd.Timestamp(args.score_alignment_paired_cutoff)
        cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
        final_scores = _score_matrix_chunked(model, matrix)
        alignment = _persist_paired_score_alignment(
            output,
            source_rows=data,
            source_scores=final_scores,
            champion_oof=args.champion_oof,
            calibration_end_exclusive=cutoff,
        )
        del final_scores
    elif args.score_alignment_oos_start:
        cutoff = pd.Timestamp(args.score_alignment_oos_start)
        cutoff = (
            cutoff.tz_localize("UTC")
            if cutoff.tzinfo is None
            else cutoff.tz_convert("UTC")
        )
        train_mask = train["__ts__"].lt(cutoff).to_numpy(dtype=bool)
        oos_mask = ~train_mask
        if int(train_mask.sum()) < 1_000 or int(oos_mask.sum()) < 128:
            raise RuntimeError(
                "OOS score alignment split has insufficient support: "
                f"train={int(train_mask.sum())} oos={int(oos_mask.sum())} "
                f"cutoff={cutoff}"
            )
        # The frame is chronologically sorted, so the split remains contiguous
        # and the checkpoint sees no calibration-period labels.
        train_end = int(np.flatnonzero(oos_mask)[0])
        checkpoint_train = train.iloc[:train_end]
        checkpoint_matrix = matrix.iloc[:train_end]
        target, _ = _base_soft_label_target(checkpoint_train)
        if not bool(target.notna().all()):
            raise RuntimeError(
                "OOS alignment checkpoint requires resolved pre-cutoff labels"
            )
        weight_columns = [
            "__ts__", "u_policy_net", "exec_margin", "mae_norm",
            "first_touch_full_path_mae_norm", "mfe_norm",
            "first_touch_full_path_mfe_norm", "first_touch_bar",
            "__archetype_policy_max_barrier__", "timeout", "side_name",
        ]
        checkpoint_weights = _base_style_weights_for_soft_label(
            checkpoint_train.reindex(columns=weight_columns), target
        ).to_numpy(dtype=np.float32, copy=False)
        half_month = (
            checkpoint_train["__ts__"].dt.strftime("%Y-%m-").to_numpy(dtype=str)
            + np.where(
                checkpoint_train["__ts__"].dt.day.to_numpy() <= 15, "01", "16"
            )
        )
        change_points = np.flatnonzero(half_month[1:] != half_month[:-1]) + 1
        starts = np.r_[0, change_points]
        ends = np.r_[change_points, len(half_month)]
        shards = [
            (int(start), int(end))
            for start, end in zip(starts, ends)
            if end > start
        ]
        params = _load_fixed_model_params(args.contract)
        print(
            "[score-alignment] fitting pre-cutoff checkpoint: "
            f"train={train_end:,} oos={int(oos_mask.sum()):,} cutoff={cutoff}",
            flush=True,
        )
        checkpoint_model = _fit_native_final_booster(
            checkpoint_matrix,
            target.to_numpy(dtype=np.float32, copy=False),
            checkpoint_weights,
            list(selected),
            shards,
            params["regressor"],
            seed=int(args.seed),
        )
        checkpoint_oos_scores = _score_matrix_chunked(
            checkpoint_model, matrix.iloc[train_end:]
        )
        alignment = _persist_oos_score_alignment(
            output,
            source_rows=data.iloc[train_end:],
            source_scores=checkpoint_oos_scores,
            champion_oof=args.champion_oof,
            train_end_exclusive=cutoff,
            knot_count=int(args.score_alignment_knots),
        )
        del checkpoint_model, checkpoint_matrix, checkpoint_train
        del checkpoint_weights, checkpoint_oos_scores, target
    else:
        alignment = _persist_score_alignment(
            output,
            model=model,
            matrix=matrix,
            sides=data["side_name"],
            champion_oof=args.champion_oof,
            knot_count=int(args.score_alignment_knots),
        )
    del matrix, train, data
    gc.collect()
    _attach_score_alignment_to_artifact(output, alignment)
    archetype_contract = _copy_native_live_policy_archetype_contract(
        args.native_parent, output
    )
    _write_global_inference_policy_contract(output)
    parity_contract_paths = _refresh_training_live_parity_contract(
        output,
        feature_source_run_id=str(args.feature_source_run_id).strip()
        or _feature_source_run_id_from_handoff(args.handoff_dir),
    )
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    manifest["score_alignment"] = {
        "mode": alignment["mode"],
        "champion_oof_source": str(args.champion_oof),
        "knot_count_requested": int(args.score_alignment_knots),
        "sides": {
            side: {
                "final_train_rows": int(details["final_train_rows"]),
                "champion_oof_rows": int(details["champion_oof_rows"]),
                "final_train_p50": float(details["final_train_quantiles"]["p50"]),
                "champion_oof_p50": float(details["champion_oof_quantiles"]["p50"]),
            }
            for side, details in alignment["sides"].items()
        },
    }
    manifest["training_live_parity_contracts"] = parity_contract_paths
    manifest.setdefault("frozen_state", {})[
        "live_policy_archetype_contract"
    ] = archetype_contract
    manifest["status"] = "candidate_native_bundle_pending_shadow_replay"
    manifest_path.write_text(json.dumps(_json_safe(manifest), indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(_json_safe(manifest["score_alignment"]), indent=2), flush=True)


def _feature_source_run_id_from_handoff(handoff_dir: Path) -> str:
    manifest_path = handoff_dir / "manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(
            f"Cannot bind live feature parity without handoff manifest: {manifest_path}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    feature_dir = str(
        manifest.get("feature_dir")
        or (manifest.get("source_manifest") or {}).get("feature_dir")
        or ""
    ).strip()
    run_id = Path(feature_dir).name if feature_dir else ""
    if not run_id or not run_id[:8].isdigit():
        raise RuntimeError(
            "Handoff manifest does not identify a timestamped feature source: "
            f"feature_dir={feature_dir!r}"
        )
    return run_id


def _refresh_training_live_parity_contract(
    output: Path,
    *,
    feature_source_run_id: str,
) -> list[str]:
    """Write hash-bound parity manifests after final model serialization.

    The parent policy's parity manifest intentionally cannot be copied: its
    model/feature hashes point at the pre-refit artifact.  Build this only
    after every final artifact has been written.
    """
    from extreme_price_movements.inference.model_orchestrator import ModelOrchestrator
    from extreme_price_movements.inference.training_live_parity_contract import (
        build_training_live_parity_contract,
        persist_training_live_parity_contract,
    )

    with (output / "models/trained_state.pkl").open("rb") as handle:
        state = pickle.load(handle)
    deployment_path = output / "simple_policy_optimiser/deployment/best_policy_params_perps.json"
    if not deployment_path.exists():
        deployment_path = output / "simple_policy_optimiser/deployment/best_policy_params.json"
    portfolio_path = output / "policy_params/optimized_portfolio_policy_config.json"
    deployment = json.loads(deployment_path.read_text(encoding="utf-8")) if deployment_path.exists() else {}
    portfolio = json.loads(portfolio_path.read_text(encoding="utf-8")) if portfolio_path.exists() else {}
    orchestrator = ModelOrchestrator(state, {})
    strategy_ids = sorted(
        str(key)
        for key in (state.get("bundle", {}).get("alpha_models", {}) or {})
        if str(key)
    )
    if not strategy_ids:
        raise RuntimeError("Final artifact has no alpha strategy routing keys for parity contract")
    contract = build_training_live_parity_contract(
        data_root="data_perp",
        run_id=output.name,
        market_mode="perps",
        orchestrator=orchestrator,
        model_bundle=state,
        strategy_ids=strategy_ids,
        deployment_payload=deployment,
        portfolio_payload=portfolio,
        feature_source_run_id=feature_source_run_id,
        feature_source_data_root="data_perp",
    )
    return [
        str(path)
        for path in persist_training_live_parity_contract(
            contract, data_root="data_perp", run_id=output.name
        )
    ]


def _write_global_inference_policy_contract(output: Path) -> Path:
    """Declare that the global S52 route has no legacy pre-base mask gate."""
    promoted_path = output / "policy_params/promoted_policy_manifest.json"
    promoted = (
        json.loads(promoted_path.read_text(encoding="utf-8"))
        if promoted_path.exists()
        else {}
    )
    payload = {
        "schema": "global_s52_inference_policy_contract_v1",
        "policy_id": str(promoted.get("policy_id") or ""),
        "policy_name": str(promoted.get("policy_name") or ""),
        "selection_rules": {
            "candidate_source": "global_base_model_then_meta_top30_then_policy_top10",
            "requires_lgbm_regime_mask_contract": False,
            "reason": (
                "The global S52 base/meta path is side- and archetype-aware; "
                "it has no legacy strategy_id pre-base regime-mask gate."
            ),
        },
    }
    path = output / "policy_params/global_inference_policy_contract.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _copy_native_live_policy_archetype_contract(
    native_parent: Path, output: Path
) -> list[str]:
    """Copy the frozen classifier that produces live base policy archetypes."""
    src_dir = native_parent / "policy_params"
    dst_dir = output / "policy_params"
    copied: list[str] = []
    for name in (
        "live_policy_archetype_classifier.joblib",
        "live_policy_archetype_classifier_manifest.json",
    ):
        src, dst = src_dir / name, dst_dir / name
        if src.exists():
            shutil.copy2(src, dst)
            copied.append(str(dst))
    if len(copied) != 2:
        raise RuntimeError(
            "Native parent is missing the frozen live policy-archetype classifier contract"
        )
    return copied


def _copy_frozen_residual_event_state_contract(
    residual_state: Path,
    output: Path,
) -> dict[str, Any]:
    """Package the exact train-frozen V9 residual-event AE/GMM state.

    The state contains only train-derived transforms and priors.  It is copied
    into the run-scoped policy directory because the V9 regime-calibration
    artifact requires its generated fields at inference.
    """

    source = residual_state.resolve()
    source_manifest = source.with_name("manifest.json")
    if not source.exists() or not source_manifest.exists():
        raise RuntimeError(
            "Missing frozen residual-event state or manifest: "
            f"state={source} manifest={source_manifest}"
        )
    destination_dir = output / "policy_params"
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / "residual_event_state.joblib"
    destination_manifest = destination_dir / "residual_event_state_manifest.json"
    shutil.copy2(source, destination)
    shutil.copy2(source_manifest, destination_manifest)
    state = joblib.load(destination)
    state_manifest = state.manifest()
    contract = {
        "schema": "frozen_residual_event_state_contract_v1",
        "source_state": str(source),
        "source_manifest": str(source_manifest),
        "packaged_state": str(destination),
        "packaged_manifest": str(destination_manifest),
        "state_sha256": _sha256(destination),
        "manifest_sha256": _sha256(destination_manifest),
        "state_train_start": state_manifest.get("train_start"),
        "state_train_end": state_manifest.get("train_end"),
        "local_model_count": state_manifest.get("local_model_count"),
        "market_secondary": state_manifest.get("market_secondary"),
        "generated_feature_count": len(state_manifest.get("generated_features") or []),
        "inference_contract": (
            "live transform receives observable pre-entry features plus frozen "
            "meta score and policy archetype routing only"
        ),
    }
    (destination_dir / "residual_event_state_contract.json").write_text(
        json.dumps(_json_safe(contract), indent=2, sort_keys=True), encoding="utf-8"
    )
    return contract


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit-chunk", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--matrix-path", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--matrix-shape", help=argparse.SUPPRESS)
    parser.add_argument("--labels-path", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--weights-path", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--features-path", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--params-path", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--checkpoint", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--chunk-start", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--chunk-end", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--chunk-trees", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--refresh-parity-only", action="store_true")
    parser.add_argument("--build-score-alignment-only", action="store_true")
    parser.add_argument("--handoff-dir", type=Path, default=DEFAULT_HANDOFF)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--native-parent", type=Path, default=DEFAULT_NATIVE_PARENT)
    parser.add_argument("--postprocessor-parent", type=Path, default=DEFAULT_POSTPROCESSOR_PARENT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--feature-source-run-id",
        default="",
        help=(
            "Timestamped feature-store run used by training and historical replay. "
            "Defaults to the handoff manifest feature_dir basename."
        ),
    )
    parser.add_argument("--champion-oof", type=Path, default=DEFAULT_CHAMPION_OOF)
    parser.add_argument(
        "--residual-event-state", type=Path, default=DEFAULT_RESIDUAL_EVENT_STATE
    )
    parser.add_argument("--score-alignment-knots", type=int, default=4097)
    parser.add_argument(
        "--final-fit-mode",
        choices=("full_dataset", "chronological_shards_legacy"),
        default="full_dataset",
        help=(
            "Train every tree on the complete final-fit matrix by default. The legacy "
            "chronological mode is retained only for controlled diagnostics because "
            "splitting trees across shards changes the fitted objective."
        ),
    )
    parser.add_argument(
        "--score-alignment-oos-start",
        default=None,
        help=(
            "Optional UTC cutoff for leakage-safe score-domain alignment. A "
            "temporary checkpoint is fit strictly before the cutoff and its "
            "post-cutoff scores are mapped to matching champion OOF scores."
        ),
    )
    parser.add_argument(
        "--score-alignment-paired-cutoff",
        default=None,
        help=(
            "Optional UTC cutoff for a same-row monotonic final-refit to champion-OOF "
            "score bridge. The bridge is fit before the cutoff and audited after it."
        ),
    )
    parser.add_argument(
        "--allow-missing-observable-feature",
        action="append",
        default=[],
        help=(
            "Explicit compatibility exception for a selected observable feature "
            "that was constant-missing in the retained historical contract. May "
            "be repeated; all other missing selected inputs remain fatal."
        ),
    )
    parser.add_argument(
        "--min-joint-observable-coverage",
        type=float,
        default=0.90,
        help=(
            "Minimum complete-case coverage across selected observable features "
            "after excluding the initial warm-up period."
        ),
    )
    parser.add_argument(
        "--coverage-warmup-days",
        type=int,
        default=30,
        help="Initial calendar days excluded from the joint-coverage audit.",
    )
    parser.add_argument("--seed", type=int, default=20260713)
    args = parser.parse_args()
    feature_source_run_id = str(args.feature_source_run_id or "").strip()
    if not feature_source_run_id:
        feature_source_run_id = _feature_source_run_id_from_handoff(args.handoff_dir)

    if args.fit_chunk:
        required = (
            args.matrix_path, args.matrix_shape, args.labels_path, args.weights_path,
            args.features_path, args.params_path, args.checkpoint, args.chunk_start,
            args.chunk_end, args.chunk_trees,
        )
        if any(value is None for value in required):
            raise RuntimeError("Incomplete isolated final-fit chunk arguments")
        _fit_native_chunk(args)
        return

    if args.refresh_parity_only:
        residual_contract = _copy_frozen_residual_event_state_contract(
            args.residual_event_state, args.output
        )
        _write_global_inference_policy_contract(args.output)
        paths = _refresh_training_live_parity_contract(
            args.output,
            feature_source_run_id=feature_source_run_id,
        )
        manifest_path = args.output / "manifest.json"
        manifest = (
            json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest_path.exists()
            else {}
        )
        manifest.setdefault("frozen_state", {})[
            "residual_event_state_contract"
        ] = residual_contract
        manifest["training_live_parity_contracts"] = paths
        manifest_path.write_text(
            json.dumps(_json_safe(manifest), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "training_live_parity_contracts": paths,
                    "residual_event_state_contract": residual_contract,
                },
                indent=2,
            ),
            flush=True,
        )
        return
    if args.build_score_alignment_only:
        _refresh_score_alignment_only(args)
        return

    contract_payload = json.loads(args.contract.read_text(encoding="utf-8"))
    contract_id = str(contract_payload.get("contract_id") or args.contract.stem)
    selected = _load_fixed_selected_features(args.contract)
    params = _load_fixed_model_params(args.contract)
    if not selected or not params:
        raise RuntimeError("Champion contract must contain selected features and model parameters")
    handoff = args.handoff_dir / "train_meta_regime_handoff.parquet"
    ledger = args.handoff_dir / "s52_trailing_regime_scored_ledger.parquet"
    print("[final-fit] loading frozen champion feature/label contract", flush=True)
    data = _read_final_fit_frame(
        handoff,
        ledger,
        list(selected),
        allowed_missing_observable=args.allow_missing_observable_feature,
        min_joint_observable_coverage=args.min_joint_observable_coverage,
        coverage_warmup_days=args.coverage_warmup_days,
    )
    data = data.loc[data["selected_top30"].fillna(False).astype(bool)].copy()
    data["__ts__"] = pd.to_datetime(data["__ts__"], utc=True, errors="coerce")
    data = data.loc[data["__ts__"].notna()].sort_values(
        ["__ts__", "__symbol__", "side_name"], kind="mergesort"
    ).reset_index(drop=True)
    fit_through = str(data["__ts__"].max())
    priors = _compact_reliability_priors(data)
    gc.collect()
    print(f"[final-fit] selected top30 rows={len(data):,}; building causal matrix", flush=True)
    train, matrix, ood, source_data = _build_final_matrix(data, list(selected))
    # ``data`` is the raw selected handoff. All live priors are frozen above;
    # retain only the causal training frame and compact float32 matrix for fit.
    del source_data, data
    gc.collect()
    print(f"[final-fit] fitting shared LightGBM rows={len(train):,} features={matrix.shape[1]}", flush=True)
    target, target_name = _base_soft_label_target(train)
    train_rows = int(len(train))
    print(
        f"[final-fit] resolved soft-label coverage={float(target.notna().mean()):.6f}",
        flush=True,
    )
    weight_columns = [
        "__ts__", "u_policy_net", "exec_margin", "mae_norm",
        "first_touch_full_path_mae_norm", "mfe_norm",
        "first_touch_full_path_mfe_norm", "first_touch_bar",
        "__archetype_policy_max_barrier__", "timeout", "side_name",
    ]
    weight_frame = train.reindex(columns=weight_columns).copy()
    # Retain chronological boundaries only for the explicit legacy diagnostic.
    # The production default fits every tree against the complete matrix; feeding
    # successive time shards through ``init_model`` changes the learning objective
    # and is not a memory-equivalent full-data fit.
    half_month = (
        train["__ts__"].dt.strftime("%Y-%m-").to_numpy(dtype=str)
        + np.where(train["__ts__"].dt.day.to_numpy() <= 15, "01", "16")
    )
    change_points = np.flatnonzero(half_month[1:] != half_month[:-1]) + 1
    starts = np.r_[0, change_points]
    ends = np.r_[change_points, len(half_month)]
    shards = [(int(start), int(end)) for start, end in zip(starts, ends) if end > start]
    valid_target = target.notna().to_numpy(dtype=bool)
    if not bool(valid_target.all()):
        raise RuntimeError(
            "Final fixed-contract refit requires resolved labels for every retained top30 row"
        )
    if bool(valid_target.all()):
        weights = _base_style_weights_for_soft_label(
            weight_frame, target
        ).to_numpy(dtype=np.float32, copy=False)
        fit_labels = target.to_numpy(dtype=np.float32, copy=False)
    else:
        weights = _base_style_weights_for_soft_label(
            weight_frame.loc[valid_target], target.loc[valid_target]
        ).to_numpy(dtype=np.float32, copy=False)
        fit_labels = target.loc[valid_target].to_numpy(dtype=np.float32, copy=False)
    feature_names = list(matrix.columns)
    stage_path = Path(tempfile.gettempdir()) / (
        f"s52_finalfit_matrix_{args.seed}_{len(matrix)}x{matrix.shape[1]}.npy"
    )
    staging = np.lib.format.open_memmap(
        stage_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(matrix), matrix.shape[1]),
    )
    for start in range(0, len(matrix), 50_000):
        end = min(start + 50_000, len(matrix))
        staging[start:end] = matrix.iloc[start:end].to_numpy(
            dtype=np.float32, copy=False
        )
    staging.flush()
    del staging
    parity_matrix = matrix.iloc[:2048].to_numpy(dtype=np.float32, copy=True)
    # LightGBM only consumes ``matrix`` and the weighting helper only consumes
    # ``weight_frame``. Drop the wide causal feature frame before fitting.
    fit_sides = train["side_name"].astype(str).copy()
    del train, matrix, target, weight_frame
    gc.collect()
    fit_values = np.load(stage_path, mmap_mode="r")
    fit_shards = _resolve_final_fit_shards(
        mode=args.final_fit_mode,
        row_count=len(fit_labels),
        chronological_shards=shards,
    )
    model = _fit_native_final_booster_isolated(
        stage_path,
        (len(fit_labels), len(feature_names)),
        fit_labels,
        weights,
        feature_names,
        fit_shards,
        params["regressor"],
        seed=int(args.seed),
    )

    output = args.output
    print(f"[final-fit] packaging native candidate at {output}", flush=True)
    output.mkdir(parents=True, exist_ok=True)
    _copytree(args.native_parent / "ae_gmm_state", output / "ae_gmm_state")
    _copytree(args.postprocessor_parent / "policy_params", output / "policy_params")
    residual_event_state_contract = _copy_frozen_residual_event_state_contract(
        args.residual_event_state, output
    )
    live_policy_archetype_contract = _copy_native_live_policy_archetype_contract(
        args.native_parent, output
    )
    _symlink(args.native_parent / "live_state", output / "live_state")
    if (args.native_parent / "simple_policy_optimiser").exists():
        _copytree(args.native_parent / "simple_policy_optimiser", output / "simple_policy_optimiser")
    if (args.postprocessor_parent / "meta_postprocessor_pointer.json").exists():
        shutil.copy2(
            args.postprocessor_parent / "meta_postprocessor_pointer.json",
            output / "meta_postprocessor_pointer.json",
        )
    score_alignment = _persist_score_alignment(
        output,
        model=model,
        matrix=fit_values,
        sides=fit_sides,
        champion_oof=args.champion_oof,
        knot_count=int(args.score_alignment_knots),
    )
    # The fit matrix is several hundred MB. It is no longer needed once the
    # model and frozen priors exist; release it before serializing state.
    del fit_values, fit_labels, weights, fit_sides
    stage_path.unlink(missing_ok=True)
    gc.collect()
    (output / "policy_params/meta_reliability_priors.json").write_text(
        json.dumps(_json_safe(priors), indent=2, sort_keys=True), encoding="utf-8"
    )
    _install_model(
        args.native_parent,
        output,
        model,
        list(selected),
        ood,
        score_alignment,
        contract_id,
    )
    _write_global_inference_policy_contract(output)
    parity_contract_paths = _refresh_training_live_parity_contract(
        output,
        feature_source_run_id=feature_source_run_id,
    )

    direct = model.predict(parity_matrix).astype(np.float32)
    persisted = joblib.load(
        output / "models/meta/long_s52_meta_threshold_handoff/base_soft_label.joblib"
    ).predict(parity_matrix).astype(np.float32)
    ae_path = output / "ae_gmm_state/ae_gmm_state.pkl"
    parent_ae_path = args.native_parent / "ae_gmm_state/ae_gmm_state.pkl"
    manifest = {
        "schema": "s52_final_meta_native_bundle_v1",
        "status": "candidate_native_bundle_pending_shadow_replay",
        "run_id": output.name,
        "fit_through": fit_through,
        "train_rows": train_rows,
        "target": target_name,
        "frontier": "top30",
        "feature_contract": {
            "contract_id": contract_id,
            "feature_count": len(selected),
            "feature_hash": _feature_contract_hash(list(selected)),
            "selection_and_hpo": "frozen; not rerun during final refit",
            "explicit_constant_missing_compatibility_features": sorted(
                set(str(name) for name in args.allow_missing_observable_feature)
            ),
            "params": params,
            "final_fit_resource_overrides": {
                "num_threads": 1,
                "force_col_wise": True,
                "max_bin": 63,
                "histogram_pool_size_mb": 8,
                "fit_mode": str(args.final_fit_mode),
                "shards": len(fit_shards),
                "trees_total": int(params["regressor"]["n_estimators"]),
            },
        },
        "causal_training_features": {
            "base_priors": "full labelled training reference",
            "reliability": "leave-one-out on fit rows",
            "support_drift": "fit-row support only",
            "hit_surprise": "strictly earlier resolved rows only",
            "ood": "frozen s52_meta_post_selection_ood_v1 reference",
        },
        "frozen_state": {
            "ae_gmm_source": str(parent_ae_path),
            "ae_gmm_sha256": _sha256(ae_path),
            "ae_gmm_matches_parent": _sha256(ae_path) == _sha256(parent_ae_path),
            "live_state_source": str(args.native_parent / "live_state"),
            "live_state_mode": "symlinked immutable feature-transform state",
            "postprocessor_source": str(args.postprocessor_parent),
            "postprocessor_policy_id": "meta_residual_v9_tail95_market_state_mlp_hier_ev_v1",
            "live_policy_archetype_contract": live_policy_archetype_contract,
            "residual_event_state_contract": residual_event_state_contract,
        },
        "serialization_parity": {
            "rows": int(len(direct)),
            "model_prediction_max_abs_diff": float(np.max(np.abs(direct - persisted))),
            "pass": bool(np.max(np.abs(direct - persisted)) <= 1e-7),
        },
        "score_alignment": {
            "mode": score_alignment["mode"],
            "champion_oof_source": str(args.champion_oof),
            "knot_count_requested": int(args.score_alignment_knots),
            "sides": {
                side: {
                    "final_train_rows": int(details["final_train_rows"]),
                    "champion_oof_rows": int(details["champion_oof_rows"]),
                    "final_train_p50": float(details["final_train_quantiles"]["p50"]),
                    "champion_oof_p50": float(details["champion_oof_quantiles"]["p50"]),
                }
                for side, details in score_alignment["sides"].items()
            },
        },
        "training_live_parity_contracts": parity_contract_paths,
        "leakage_contract": {
            "base_model": "frozen; not retrained",
            "feature_selection_hpo": "retained champion contract only",
            "model_fit": "resolved labels through fit_through only",
            "ae_gmm": "copied frozen state; never refitted",
            "postprocessors": "copied from current v9 tail95 MLP policy artifact without retuning",
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(_json_safe(manifest), indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(_json_safe(manifest), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
