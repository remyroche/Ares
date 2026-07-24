#!/usr/bin/env python3
"""Reconstruct the frozen pre-July Pack-B base fold and score canonical rows.

This tool is intentionally narrow.  It does not refit feature selection, HPO,
AE/GMM, base or meta models on new outcomes.  It reconstructs the exact July
base fold from its compact cache, validates it against the cached OOS ledger,
then scores an externally supplied canonical pre-entry frame using:

* the cycle-frozen AE/GMM state;
* timestamp x side base top-30 admission;
* the saved July Pack-B weighted meta long/short models.

The supplied frame must be produced by the canonical feature pipeline and must
contain the complete observable source columns.  This script fails closed for
missing columns and non-finite base/AE-GMM inputs; it does not synthesize a
feature contract from labels or outcomes.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.features_gmm_ae import (  # noqa: E402
    load_ae_gmm_state_artifact,
    transform_ae_gmm_features,
)
from extreme_price_movements.feature_transform_contract import (  # noqa: E402
    FLOAT16_CLIPPED_THEN_FLOAT32_V1,
    apply_model_input_numeric_contract,
)
from extreme_price_movements.inference.s52_meta_ood import (  # noqa: E402
    append_s52_meta_ood_features,
    fit_s52_meta_ood_reference,
)
from extreme_price_movements.inference.live_policy_archetype import (  # noqa: E402
    OBSERVABLE_REGIME_FAMILY_SCORE_COLUMNS,
    predict_observable_policy_archetype,
)
from extreme_price_movements.inference.live_meta_feature_overlays import (  # noqa: E402
    materialize_live_source_regime_features,
)
from extreme_price_movements.meta_input_contract import (  # noqa: E402
    materialize_legacy_constant_zeros,
    require_encoded_meta_matrix,
    require_resolved_meta_input_contract,
    resolve_meta_input_contract,
)
from extreme_price_movements.static_feature_store import read_static_features  # noqa: E402
from extreme_price_movements.inference.side_residual_expert import (  # noqa: E402
    SideResidualExpertBundle,
)
from extreme_price_movements.inference.canonical_meta_postprocessor import (  # noqa: E402
    V9TailPostprocessor,
)
from scripts.run_materialized_trailing_label_topk_lgbm_hpo import (  # noqa: E402
    _fit_lgbm_models,
    _predict_lgbm_models,
    _timestamp_side_ranks,
)
from scripts.run_s52_train_meta_regime_handoff_smoke import (  # noqa: E402
    ALL_META_POST_SELECTION_OOD_FEATURE_NAMES,
    OUTCOME_COLUMNS,
    _add_fold_base_prior_features,
    _add_fold_hit_surprise_features,
    _add_fold_reliability_features,
    _add_fold_support_drift_features,
    _feature_contract_hash,
    _load_joined_frame,
    _make_xy,
)
from scripts.report_s52_trailing_regime_meta_handoff import (  # noqa: E402
    _apply_edges,
    _build_regime_columns,
    _quantile_edges,
)
from scripts.backfill_complete_july_meta_predictions import (  # noqa: E402
    _hydrate_live_gated_inputs,
)


DEFAULT_BASE_RUN = ROOT / "data_perp/reports/s59_h5_signalclose_causal_stagec_packb_sliding365_wf30_20260721_v1"
DEFAULT_FOLD = DEFAULT_BASE_RUN / "_fold_cache/2026-06-30_2026-07-30"
DEFAULT_META_MODELS = (
    ROOT
    / "data_perp/reports/s59_h5_signalclose_causal_stagec_packb_sliding365_meta_hpo150_wf30_20260721_v1"
    / "best_full_oos/models/2026-07-01_2026-07-31"
)
DEFAULT_META_TRAIN = (
    ROOT
    / "data_perp/reports/s59_h5_signalclose_causal_stagec_packb_sliding365_wf30_20260721_v1"
    / "meta_handoff_top30/train_meta_regime_handoff.parquet"
)
DEFAULT_META_LEDGER = (
    ROOT
    / "data_perp/reports/s59_h5_signalclose_causal_stagec_packb_sliding365_wf30_20260721_v1"
    / "meta_handoff_top30/s52_trailing_regime_scored_ledger.parquet"
)
DEFAULT_AE_GMM_STATE = (
    ROOT
    / "data_perp/reports/s59_h5_signalclose_causal_base_incumbentmda_hpo150_wf30_20260720_v2"
    / "_feature_selection_phase/ae_gmm_states/cycle__global_state.pkl"
)
DEFAULT_OUTPUT = ROOT / "data_perp/reports/weighted_packb_july_frozen_oos_scoring_v1"

JULY_FOLD_ID = 6
JULY_TRAIN_END = pd.Timestamp("2026-07-01", tz="UTC")
JULY_HPO_TRIAL = 135
JULY_BASE_SEED = 42 + 1000 * JULY_HPO_TRIAL + JULY_FOLD_ID
BASE_TOP_FRAC = 0.30
KEY_COLUMNS = ("__ts__", "__symbol__", "side_name")
META_PRIOR_REQUIRED_OUTCOMES = (
    "clean_exec",
    "clean_exec_label",
    "dirty_positive",
    "full_path_bad_mae_1r",
    "first_touch_bad_mae_1r",
    "timeout",
    "exec_margin",
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise TypeError(f"Expected object JSON: {path}")
    return loaded


def _utc(series: pd.Series, *, name: str) -> pd.Series:
    values = pd.to_datetime(series, utc=True, errors="coerce")
    if values.isna().any():
        raise ValueError(f"{name} contains {int(values.isna().sum())} invalid UTC timestamps")
    return values


def _normalize_side(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "side_name" in out.columns:
        side = out["side_name"].astype(str).str.lower()
    else:
        raw = pd.to_numeric(out.get("side", out.get("__side__")), errors="coerce")
        if raw.isna().any():
            raise ValueError("Canonical frame needs side_name or finite side/__side__")
        side = pd.Series(np.where(raw.to_numpy() < 0.0, "short", "long"), index=out.index)
    invalid = ~side.isin(("long", "short"))
    if invalid.any():
        raise ValueError(f"Unsupported side_name values: {sorted(side.loc[invalid].unique())[:10]}")
    out["side_name"] = side.astype(str)
    out["side"] = np.where(side.eq("short"), -1, 1).astype(np.int8)
    out["__side__"] = out["side"]
    return out


def _canonicalize(frame: pd.DataFrame, *, role: str) -> pd.DataFrame:
    missing = [name for name in ("__ts__", "__symbol__") if name not in frame.columns]
    if missing:
        raise ValueError(f"{role} misses identity column(s): {missing}")
    out = _normalize_side(frame)
    out["__ts__"] = _utc(out["__ts__"], name=f"{role}.__ts__")
    out["__symbol__"] = out["__symbol__"].astype(str)
    if out["__symbol__"].eq("").any():
        raise ValueError(f"{role} contains empty symbols")
    duplicate = out.duplicated(list(KEY_COLUMNS), keep=False)
    if duplicate.any():
        raise ValueError(
            f"{role} has {int(duplicate.sum())} duplicate UTC timestamp/symbol/side rows"
        )
    return out.sort_values(list(KEY_COLUMNS), kind="mergesort").reset_index(drop=True)


def _validate_meta_prior_training_contract(frame: pd.DataFrame) -> dict[str, Any]:
    """Fail closed when train-derived meta priors lack their outcome sources."""

    missing = [name for name in META_PRIOR_REQUIRED_OUTCOMES if name not in frame.columns]
    if missing:
        raise ValueError(
            "Meta prior training frame is missing supervised outcome column(s): "
            f"{missing}. Load it through the canonical handoff + outcome-ledger join; "
            "the reliability helpers otherwise replace missing targets with zero."
        )
    coverage: dict[str, float] = {}
    for name in META_PRIOR_REQUIRED_OUTCOMES:
        values = pd.to_numeric(frame[name], errors="coerce")
        finite = np.isfinite(values.to_numpy(dtype=np.float64, copy=False))
        coverage[name] = float(finite.mean()) if len(values) else 0.0
        if not finite.any():
            raise ValueError(f"Meta prior training outcome {name!r} has no finite values")
    return {
        "required_outcomes": list(META_PRIOR_REQUIRED_OUTCOMES),
        "finite_coverage": coverage,
    }


def _load_fold_payload(fold_dir: Path) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    manifest = _read_json(fold_dir / "fold_manifest.json")
    paths = {key: ROOT / value for key, value in dict(manifest["payload_paths"]).items()}
    required = ("x_train", "train_target", "train_weight", "train_side", "x_valid", "valid")
    missing = [key for key in required if key not in paths or not paths[key].is_file()]
    if missing:
        raise FileNotFoundError(f"Fold cache is incomplete: {missing}")
    payload = {key: pd.read_parquet(paths[key]) for key in required}
    return manifest, payload


def _base_params(base_run: Path) -> dict[str, Any]:
    best = _read_json(base_run / "topk_lgbm_hpo_best.json")
    required = (
        "learning_rate", "n_estimators", "num_leaves", "max_depth", "min_child_samples",
        "subsample", "colsample_bytree", "reg_alpha", "reg_lambda", "loss_function",
        "target_mode", "weight_arm", "model_side_scope",
    )
    missing = [key for key in required if key not in best]
    if missing:
        raise ValueError(f"Base HPO artifact misses parameters: {missing}")
    return best


def _fit_base_fold(
    *,
    payload: Mapping[str, pd.DataFrame],
    fold_manifest: Mapping[str, Any],
    params: Mapping[str, Any],
    fold_id: int,
) -> tuple[Any, dict[str, list[str]], int]:
    x_train = payload["x_train"].astype(np.float32, copy=False)
    target = payload["train_target"]
    weights = payload["train_weight"]
    sides = payload["train_side"]
    if "target_soft" not in target.columns or "sample_weight" not in weights.columns:
        raise ValueError("Compact fold cache is missing target_soft or sample_weight")
    if "side_name" not in sides.columns:
        raise ValueError("Compact fold cache is missing train_side.side_name")
    selected = dict(fold_manifest.get("selected_features_by_side", {}) or {})
    seed = 42 + 1000 * JULY_HPO_TRIAL + int(fold_id)
    models, contracts = _fit_lgbm_models(
        x_train=x_train,
        y_train=target["target_soft"].astype(np.float32),
        w_train=weights["sample_weight"].astype(np.float32),
        train_sides=sides["side_name"].astype(str).str.lower().to_numpy(),
        params=dict(params),
        seed=seed,
        model_side_scope=str(params["model_side_scope"]),
        features_by_side=selected,
    )
    return models, contracts, seed


def _save_base_models(
    *,
    output_dir: Path,
    models: Any,
    contracts: Mapping[str, Sequence[str]],
    params: Mapping[str, Any],
    seed: int,
    fold_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    model_dir = output_dir / "base_fold_models"
    model_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    for side, model in dict(models).items():
        path = model_dir / f"base_model_{side}.joblib"
        joblib.dump(model, path, compress=3)
        paths[str(side)] = str(path)
    contract = {str(side): [str(name) for name in names] for side, names in contracts.items()}
    columns_path = model_dir / "columns.json"
    columns_path.write_text(json.dumps({
        "schema": "weighted_packb_frozen_july_base_feature_contract_v1",
        "feature_names_by_side": contract,
        "feature_count_by_side": {side: len(names) for side, names in contract.items()},
        "feature_contract_hash": _feature_contract_hash(contract),
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "schema": "weighted_packb_frozen_july_base_fold_model_v1",
        "fold": str(fold_manifest["fold"]),
        "train_start": str(fold_manifest["train_start"]),
        "valid_start": str(fold_manifest["valid_start"]),
        "valid_end": str(fold_manifest["valid_end"]),
        "hpo_trial_number": JULY_HPO_TRIAL,
        "seed_before_side_offsets": int(seed),
        "models": paths,
        "columns_path": str(columns_path),
        "params": _json_safe(dict(params)),
        "leakage_contract": "fit uses only cached pre-July compact train payload",
    }
    (model_dir / "manifest.json").write_text(
        json.dumps(_json_safe(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def _load_saved_base_models(model_dir: Path) -> tuple[dict[str, Any], dict[str, list[str]], int]:
    columns = _read_json(model_dir / "columns.json")
    manifest_path = model_dir / "manifest.json"
    manifest = _read_json(manifest_path) if manifest_path.is_file() else {}
    contracts = {
        str(side): [str(name) for name in names]
        for side, names in dict(columns.get("feature_names_by_side", {})).items()
    }
    shared_path = model_dir / "base_model.joblib"
    if not shared_path.is_file():
        declared_shared = dict(manifest.get("models", {}) or {}).get("shared")
        if declared_shared:
            declared_path = Path(str(declared_shared))
            shared_path = (
                declared_path
                if declared_path.is_absolute()
                else ROOT / declared_path
            )
        elif (model_dir / "base_model_shared.joblib").is_file():
            shared_path = model_dir / "base_model_shared.joblib"
    if shared_path.is_file():
        loaded = joblib.load(shared_path)
        # Current hybrid artifacts package the independently trained long and
        # short heads in a single file.  Preserve their side-local contracts;
        # treating this mapping as a shared estimator both breaks ``predict``
        # and discards the intended feature separation.
        if isinstance(loaded, dict) and {"long", "short"}.issubset(loaded):
            if not contracts.get("long") or not contracts.get("short"):
                raise ValueError(
                    f"Side-model checkpoint has no long/short feature contracts: {model_dir}"
                )
            models = {side: loaded[side] for side in ("long", "short")}
        else:
            feature_names = [str(name) for name in columns.get("feature_names", [])]
            if not feature_names:
                feature_names = list(contracts.get("shared", []))
            if not feature_names:
                raise ValueError(f"Shared base checkpoint has no feature contract: {model_dir}")
            contracts = {"shared": feature_names}
            models = {"shared": loaded}
    else:
        models = {
            side: joblib.load(model_dir / f"base_model_{side}.joblib")
            for side in ("long", "short")
        }
    return models, contracts, int(
        manifest.get("seed", columns.get("seed_before_side_offsets", JULY_BASE_SEED))
    )


def _score_base(
    *,
    models: Any,
    contracts: Mapping[str, Sequence[str]],
    frame: pd.DataFrame,
    params: Mapping[str, Any],
) -> pd.DataFrame:
    missing = sorted(set(name for names in contracts.values() for name in names if name not in frame.columns))
    if missing:
        raise ValueError(f"Canonical frame misses {len(missing)} base feature(s): {missing[:20]}")
    out = _canonicalize(frame, role="base scoring frame")
    feature_union = list(dict.fromkeys(name for names in contracts.values() for name in names))
    numeric = out.reindex(columns=feature_union).apply(pd.to_numeric, errors="coerce")
    # The base contract is side-local.  Requiring every row to be finite over
    # the union of long and short features rejects a valid short solely because
    # a long-only input is absent (and vice versa).  Keep strict complete-case
    # scoring, but apply it against the row's own side contract.
    finite = np.zeros(len(out), dtype=bool)
    side_values = out["side_name"].astype("string")
    if "shared" in contracts:
        finite = np.isfinite(
            numeric.loc[:, list(contracts["shared"])].to_numpy(
                dtype=np.float32, copy=False
            )
        ).all(axis=1)
    else:
        for side, side_features in contracts.items():
            side_mask = side_values.eq(str(side)).to_numpy()
            if not side_mask.any():
                continue
            side_numeric = numeric.loc[side_mask, list(side_features)]
            finite[side_mask] = np.isfinite(
                side_numeric.to_numpy(dtype=np.float32, copy=False)
            ).all(axis=1)
    out["base_input_complete"] = finite.astype(np.int8)
    out["score"] = np.nan
    if finite.any():
        # Fold models were fitted from x_train/x_valid cache payloads persisted
        # as clip -> float16. Reproduce that numerical boundary before scoring;
        # the exported ledger/context columns themselves remain float32.
        x = apply_model_input_numeric_contract(
            numeric.loc[finite].astype(np.float32, copy=False),
            FLOAT16_CLIPPED_THEN_FLOAT32_V1,
            # The union matrix contains inactive opposite-side columns. Those
            # may remain missing exactly as in the fold cache; completeness is
            # enforced above against each row's active side contract.
            require_finite=False,
        )
        if "shared" in contracts:
            prediction = pd.Series(
                models["shared"].predict(x.loc[:, contracts["shared"]]),
                index=x.index,
                dtype=np.float32,
            )
        else:
            prediction = _predict_lgbm_models(
                models=models,
                x_valid=x,
                valid_sides=out.loc[finite, "side_name"].to_numpy(),
                model_side_scope=str(params["model_side_scope"]),
                feature_contracts=contracts,
            )
        out.loc[finite, "score"] = prediction.to_numpy(dtype=np.float32)
    ranks = _timestamp_side_ranks(
        out.loc[finite].reset_index(drop=True),
        out.loc[finite, "score"].reset_index(drop=True),
        out.loc[finite, "side"].reset_index(drop=True),
    ) if finite.any() else pd.DataFrame()
    for column in ("base_rank_within_timestamp_side", "base_rank_pct_timestamp_side", "base_cutoff_score_timestamp_side"):
        out[column] = np.nan
    out["selected_top30"] = False
    if not ranks.empty:
        positions = np.flatnonzero(finite)
        group_rows = ranks["group_rows"].to_numpy(dtype=np.int64)
        selected = ranks["rank"].to_numpy(dtype=np.int64) <= np.ceil(group_rows * BASE_TOP_FRAC)
        cutoff = pd.DataFrame({
            "ts": out.loc[finite, "__ts__"].to_numpy(),
            "side": out.loc[finite, "side_name"].to_numpy(),
            "score": out.loc[finite, "score"].to_numpy(),
            "selected": selected,
        }).assign(score_selected=lambda x: x["score"].where(x["selected"])).groupby(
            ["ts", "side"], sort=False
        )["score_selected"].transform("min")
        out.loc[finite, "base_rank_within_timestamp_side"] = ranks["rank"].to_numpy(dtype=np.int32)
        out.loc[finite, "base_rank_pct_timestamp_side"] = ranks["rank_pct"].to_numpy(dtype=np.float32)
        out.loc[finite, "base_cutoff_score_timestamp_side"] = cutoff.to_numpy(dtype=np.float32)
        out.loc[finite, "selected_top30"] = selected
    out["candidate_handoff_rank_scope"] = "timestamp_side"
    return out


def _append_candidate_base_score_context(frame: pd.DataFrame) -> pd.DataFrame:
    """Reproduce score context on the selected handoff population."""

    out = frame.copy(deep=False)
    score = pd.to_numeric(out["score"], errors="coerce")
    timestamp = pd.to_datetime(out["__ts__"], utc=True, errors="coerce")
    side = out["side_name"].astype(str).str.lower()
    out["base_rank_pct_by_timestamp"] = (
        score.groupby(timestamp).rank(pct=True).astype(np.float32)
    )
    out["base_rank_pct_by_timestamp_side"] = (
        score.groupby([timestamp, side]).rank(pct=True).astype(np.float32)
    )
    mean_timestamp = score.groupby(timestamp).transform("mean")
    std_timestamp = score.groupby(timestamp).transform("std").replace(0.0, np.nan)
    out["base_score_z_by_timestamp"] = (
        (score - mean_timestamp) / std_timestamp
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
    mean_timestamp_side = score.groupby([timestamp, side]).transform("mean")
    std_timestamp_side = (
        score.groupby([timestamp, side]).transform("std").replace(0.0, np.nan)
    )
    out["base_score_z_by_timestamp_side"] = (
        (score - mean_timestamp_side) / std_timestamp_side
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
    return out


def _append_frozen_base_train_prior_rank(
    train: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    fit_end_exclusive: pd.Timestamp,
) -> pd.DataFrame:
    """Map candidate scores through the same frozen, tie-aware train CDF."""

    reference_mask = pd.to_datetime(
        train["__ts__"], utc=True, errors="coerce"
    ).lt(fit_end_exclusive)
    reference = pd.to_numeric(
        train.loc[reference_mask, "score"], errors="coerce"
    ).to_numpy(dtype=np.float32)
    reference = np.sort(reference[np.isfinite(reference)])
    if not len(reference):
        raise ValueError("Frozen base-rank prior has no finite reference scores")
    values = pd.to_numeric(candidates["score"], errors="coerce").to_numpy(
        dtype=np.float32
    )
    finite = np.isfinite(values)
    ranks = np.full(len(values), 0.5, dtype=np.float32)
    left = np.searchsorted(reference, values[finite], side="left")
    right = np.searchsorted(reference, values[finite], side="right")
    ranks[finite] = ((left + right) / (2.0 * len(reference))).astype(np.float32)
    out = candidates.copy(deep=False)
    out["base_score_rank_pct_train_prior"] = ranks
    return out


def _apply_frozen_ae_gmm(frame: pd.DataFrame, state: Mapping[str, Any]) -> pd.DataFrame:
    inputs = [str(name) for name in state.get("feature_columns", [])]
    missing = [name for name in inputs if name not in frame.columns]
    if missing:
        raise ValueError(f"Canonical frame misses {len(missing)} AE/GMM inputs: {missing[:20]}")
    fill_map = {str(key): float(value) for key, value in dict(state.get("cycle_input_fill_values", {}) or {}).items()}
    source = frame.reindex(columns=inputs).apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    missing_before_fill = int(source.isna().sum().sum())
    for name in inputs:
        source[name] = source[name].fillna(fill_map.get(name, np.nan))
    invalid = ~np.isfinite(source.to_numpy(dtype=np.float32, copy=False)).all(axis=1)
    out = frame.copy()
    out["ae_gmm_input_complete"] = (~invalid).astype(np.int8)
    if (~invalid).any():
        generated = transform_ae_gmm_features(
            source.loc[~invalid].astype(np.float32, copy=False), dict(state),
            index=out.index[~invalid],
        )
        for column in generated.columns:
            out[column] = np.nan
            out.loc[~invalid, column] = generated[column].to_numpy(dtype=np.float32)
    out.attrs["ae_gmm_missing_values_filled"] = missing_before_fill
    return out


def _frozen_ae_gmm_output_names(state: Mapping[str, Any]) -> set[str]:
    """Derive the complete emitted contract, including legacy unprefixed aliases."""

    inputs = [str(name) for name in state.get("feature_columns", []) or []]
    fill = dict(state.get("cycle_input_fill_values", {}) or {})
    if not inputs:
        return set()
    probe = pd.DataFrame(
        [{name: float(fill.get(name, 0.0)) for name in inputs}],
        dtype=np.float32,
    )
    return set(
        transform_ae_gmm_features(probe, dict(state), index=probe.index).columns
    )


def _required_meta_sources(columns: Mapping[str, Any]) -> set[str]:
    entries = list(dict(columns.get("input_feature_contract", {}) or {}).get("entries", []) or [])
    generated = set(ALL_META_POST_SELECTION_OOD_FEATURE_NAMES)
    sources: set[str] = set()
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        source = str(entry.get("source_column") or "")
        kind = str(entry.get("source_type") or "")
        if source and kind != "fold_or_post_selection_generated" and source not in generated:
            sources.add(source)
    return sources


def _materialize_static_oos_frame(
    *,
    feature_root: Path,
    feature_keys: Sequence[str],
    symbols: Sequence[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """Read canonical static blocks in bounded symbol batches.

    The static endpoint is the parity authority, but a single 600-column read
    across the full eligible universe transiently builds several wide panels.
    Materialising independent symbol blocks keeps the values and timestamps
    identical while avoiding a multi-gigabyte fan-in for forward replays.
    """

    feature_store_ts = pd.to_datetime(
        feature_root.name, format="%Y%m%d_%H%M%S", utc=True
    )
    requested = list(dict.fromkeys(str(name) for name in feature_keys))
    symbol_values = list(dict.fromkeys(str(name) for name in symbols))
    if not symbol_values:
        raise RuntimeError("No symbols supplied to shared static OOS materialization")
    # ``read_static_features(..., output_layout='panels')`` holds its raw
    # column buffers behind a lazy proxy.  Eight symbols keeps the peak below
    # the replay host limit even for the 600-column AE/GMM input contract.
    batch_size = 8
    frames: list[pd.DataFrame] = []
    available: set[str] = set()
    for offset in range(0, len(symbol_values), batch_size):
        batch_symbols = symbol_values[offset : offset + batch_size]
        loaded = read_static_features(
            feature_store_ts=feature_store_ts,
            data_root=feature_root.parents[1],
            feature_keys=requested,
            symbols=batch_symbols,
            start_ts=start,
            end_ts=end - pd.Timedelta(nanoseconds=1),
            output_layout="panels",
        )
        if not loaded:
            continue
        available.update(str(name) for name in loaded)
        first_name = next(iter(loaded))
        timestamps = pd.DatetimeIndex(loaded[first_name].index)
        timestamps = timestamps[(timestamps >= start) & (timestamps < end)]
        present_symbols = [name for name in batch_symbols if name in loaded[first_name].columns]
        if timestamps.empty or not present_symbols:
            continue
        data: dict[str, Any] = {
            "__ts__": np.repeat(timestamps.to_numpy(), len(present_symbols)),
            "__symbol__": np.tile(np.asarray(present_symbols, dtype=object), len(timestamps)),
        }
        for name in loaded:
            if name not in requested:
                continue
            panel = loaded[name].reindex(index=timestamps, columns=present_symbols)
            data[name] = panel.to_numpy(dtype=np.float32, copy=False).reshape(-1)
        frames.append(pd.DataFrame(data))
        # Release the LazyFeatureDict raw buffers before loading the next
        # block.  ``frames`` contains only flattened float32 copies.
        del panel, data, loaded
        gc.collect()
    if not frames:
        raise RuntimeError("Shared static endpoint returned no OOS feature rows")
    missing = sorted(set(requested) - available)
    allowed_derived_missing = {
        name
        for name in missing
        if name.startswith("__regime_source_")
        or name.startswith("__meta_raw__")
        or name.startswith("base_arch_hit_")
        or name.startswith("base_margin_to_cutoff")
        or name.startswith("base_rank_pct_")
        or name.startswith("base_score_")
        or name in {
            "base_margin_band",
            "base_rank_band",
            "source_tag",
            "support_mean_log_count",
        }
        or name == "base_signal_zscore_within_archetype"
        or name.startswith("rel_rankband_")
        or name.startswith("rel_marginband_")
        or name.startswith("regime_")
        or name in {"side", "support_min_frequency"}
        or "_G_VOL_" in name
        or "_G_TREND_" in name
    }
    missing_static = sorted(set(missing) - allowed_derived_missing)
    if missing_static:
        raise ValueError(
            f"Shared static store misses {len(missing_static)} required source features: "
            f"{missing_static[:20]}"
        )
    one_side = pd.concat(frames, ignore_index=True, copy=False)
    long = one_side.copy(deep=False)
    long["side_name"] = "long"
    long["side"] = np.int8(1)
    short = one_side.copy(deep=False)
    short["side_name"] = "short"
    short["side"] = np.int8(-1)
    return pd.concat([long, short], ignore_index=True, copy=False)


def _materialize_observable_overlays(
    frame: pd.DataFrame, *, required_columns: Sequence[str]
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for side in ("long", "short"):
        batch = frame.loc[frame["side_name"].eq(side)].copy()
        if batch.empty:
            continue
        parts.append(
            materialize_live_source_regime_features(
                batch,
                side=side,
                signal_bar_ts=None,
                required_columns=required_columns,
                overwrite_existing=False,
            )
        )
    if not parts:
        raise RuntimeError("Observable overlay materialization received no side rows")
    return pd.concat(parts, ignore_index=True, copy=False)


def _prepare_meta_matrix(
    *,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    columns: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], pd.DataFrame]:
    train, valid = _add_fold_base_prior_features(train, valid, selected_col="selected_top30")
    train, valid = _add_fold_reliability_features(train, valid)
    train, valid = _add_fold_support_drift_features(train, valid)
    train, valid = _add_fold_hit_surprise_features(train, valid)
    selected = [str(name) for name in columns.get("feature_names", [])]
    required_sources = _required_meta_sources(columns)
    missing = sorted(source for source in required_sources if source not in valid.columns or source not in train.columns)
    if missing:
        raise ValueError(f"Meta input is missing {len(missing)} frozen source feature(s): {missing[:20]}")
    numeric = [str(name) for name in dict(columns["preprocessing_state"]).get("numeric_columns", [])]
    categorical = [str(name) for name in dict(columns["preprocessing_state"]).get("categorical_source_columns", [])]
    ood = set(ALL_META_POST_SELECTION_OOD_FEATURE_NAMES)
    core = [name for name in selected if name not in ood]
    train_x, valid_x, _ = _make_xy(
        train,
        valid,
        numeric_cols=numeric,
        categorical_cols=categorical,
        selected_features=core,
    )
    reference = fit_s52_meta_ood_reference(train_x, core)
    required_ood = [name for name in selected if name in ood]
    train_x = append_s52_meta_ood_features(train_x, reference, output_features=required_ood)
    valid_x = append_s52_meta_ood_features(valid_x, reference, output_features=required_ood)
    contract = dict(columns.get("input_feature_contract", {}) or {})
    require_resolved_meta_input_contract(contract, role="frozen July Pack-B meta scoring")
    train_x = materialize_legacy_constant_zeros(train_x, contract)
    valid_x = materialize_legacy_constant_zeros(valid_x, contract)
    train_x = require_encoded_meta_matrix(train_x, feature_names=selected, role="frozen Pack-B meta train matrix")
    valid_x = require_encoded_meta_matrix(valid_x, feature_names=selected, role="frozen Pack-B meta scoring matrix")
    return (
        train_x,
        valid_x,
        {"ood_reference": reference, "selected_features": selected},
        valid,
    )


def _score_meta(
    *,
    train: pd.DataFrame,
    candidates: pd.DataFrame,
    meta_model_dir: Path,
    source_contract: Mapping[str, Any],
    regime_fit_months: Sequence[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    columns = _read_json(meta_model_dir / "columns.json")
    models = {
        "long": joblib.load(meta_model_dir / "base_soft_label_long.joblib"),
        "short": joblib.load(meta_model_dir / "base_soft_label_short.joblib"),
    }
    candidates = _assign_frozen_source_score_tags(
        candidates, source_contract=source_contract
    )
    # The historical handoff filters the scored ledger to selected_top30 before
    # computing timestamp and timestamp-side rank/z-score context.
    candidates = _append_candidate_base_score_context(candidates)
    candidates = _apply_recovered_handoff_base_context(train, candidates)
    combined = pd.concat(
        [train.assign(__meta_split__="train"), candidates.assign(__meta_split__="oos")],
        ignore_index=True,
        copy=False,
    )
    fit_ts = pd.to_datetime(combined["__ts__"], utc=True, errors="coerce")
    fit_month_values = tuple(str(month) for month in regime_fit_months)
    if not fit_month_values:
        raise ValueError("Frozen meta handoff has no recorded regime fit months")
    fit_mask = combined["__meta_split__"].eq("train") & fit_ts.dt.strftime(
        "%Y-%m"
    ).isin(fit_month_values)
    if not bool(fit_mask.any()):
        raise ValueError(
            "Frozen meta regime fit window has no rows after training cutoff: "
            f"months={list(fit_month_values)}"
        )

    # Reproduce the handoff materialization order exactly.  The semantic source
    # tag is an input to the later fold-local base priors, while the supervised
    # regime heads were originally fit after the Jan-Mar base-context transform
    # and before those fold-local priors were applied.  Replacing source tags
    # with score-decile tags or applying the July priors first changes both
    # feature families materially.
    stored_train = combined["__meta_split__"].eq("train")
    reference_columns = {
        name: combined.loc[stored_train, name].reset_index(drop=True).copy()
        for name in (
            "source_tag",
            "base_margin_to_cutoff_z",
            "base_signal_zscore_within_archetype",
            "regime_bad_mae_score",
            "regime_first_touch_bad_mae_score",
            "regime_dirty_positive_score",
        )
        if name in combined.columns
    }
    combined, regime_specs = _build_regime_columns(combined, fit_mask=fit_mask)

    upstream_parity: dict[str, Any] = {}
    for name, expected in reference_columns.items():
        actual = combined.loc[stored_train, name].reset_index(drop=True)
        if name == "source_tag":
            match = actual.astype(str).eq(expected.astype(str))
            upstream_parity[name] = {
                "rows": int(len(match)),
                "exact_match_rate": float(match.mean()) if len(match) else 1.0,
            }
            if not bool(match.all()):
                raise RuntimeError(
                    "Frozen handoff source-tag reconstruction failed: "
                    f"match_rate={float(match.mean()):.8f}"
                )
            continue
        actual_num = pd.to_numeric(actual, errors="coerce").to_numpy(dtype=np.float64)
        expected_num = pd.to_numeric(expected, errors="coerce").to_numpy(dtype=np.float64)
        delta = np.abs(actual_num - expected_num)
        finite = np.isfinite(delta)
        max_delta = float(np.max(delta[finite])) if finite.any() else 0.0
        mean_delta = float(np.mean(delta[finite])) if finite.any() else 0.0
        upstream_parity[name] = {
            "rows": int(len(delta)),
            "finite_rows": int(finite.sum()),
            "mean_abs_delta": mean_delta,
            "max_abs_delta": max_delta,
        }
        if max_delta > 1e-6:
            raise RuntimeError(
                f"Frozen handoff transform reconstruction failed for {name}: "
                f"max_abs_delta={max_delta:.9g}"
            )

    train_regime = combined.loc[combined["__meta_split__"].eq("train")].drop(
        columns="__meta_split__"
    ).reset_index(drop=True)
    candidates_regime = combined.loc[combined["__meta_split__"].eq("oos")].drop(
        columns="__meta_split__"
    ).reset_index(drop=True)
    train_x, valid_x, matrix_info, prepared_candidates = _prepare_meta_matrix(
        train=train_regime, valid=candidates_regime, columns=columns
    )
    names_by_model = dict(columns.get("feature_names_by_model", {}) or {})
    # Keep the exact train-derived prior/reliability fields used by the encoded
    # matrix. Frozen residual experts consume these observable fields directly.
    out = prepared_candidates.copy()
    out["score_meta_base_soft_label"] = np.nan
    for side in ("long", "short"):
        label = f"base_soft_label_{side}"
        names = [str(name) for name in names_by_model.get(label, [])]
        if not names:
            raise ValueError(f"Meta columns contract lacks {label} feature names")
        mask = out["side_name"].eq(side).to_numpy()
        if mask.any():
            out.loc[mask, "score_meta_base_soft_label"] = models[side].predict(
                valid_x.loc[mask, names]
            ).astype(np.float32)
    if out["score_meta_base_soft_label"].isna().any():
        raise RuntimeError("Frozen meta scoring emitted non-finite scores")
    all_scores = out["score_meta_base_soft_label"].to_numpy(dtype=np.float64)
    rank = pd.Series(all_scores).rank(method="first", ascending=False, pct=True).astype(np.float32)
    out["meta_rank_global"] = rank.to_numpy()
    out["meta_global_rank_order"] = pd.Series(all_scores).rank(method="first", ascending=False).astype(np.int32).to_numpy()
    return out, {
        "meta_model_dir": str(meta_model_dir),
        "meta_feature_contract_hash": columns.get("feature_contract_hash"),
        "meta_feature_count": int(columns.get("feature_count", 0)),
        "meta_feature_count_by_model": {name: len(values) for name, values in names_by_model.items()},
        "ood_reference": matrix_info["ood_reference"],
        "regime_fit_rows": int(fit_mask.sum()),
        "regime_fit_months": list(fit_month_values),
        "regime_specs": regime_specs,
        "source_contract": dict(source_contract),
        "upstream_transform_parity": upstream_parity,
    }


def _validation_parity(
    *,
    payload: Mapping[str, pd.DataFrame],
    models: Any,
    contracts: Mapping[str, Sequence[str]],
    params: Mapping[str, Any],
    cached_scored_path: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    valid = _canonicalize(payload["valid"], role="cached July validation rows")
    x_valid = payload["x_valid"].astype(np.float32, copy=False)
    if len(valid) != len(x_valid):
        raise RuntimeError("Cached July valid and x_valid row counts differ")
    prediction = _predict_lgbm_models(
        models=models,
        x_valid=x_valid,
        valid_sides=valid["side_name"].to_numpy(),
        model_side_scope=str(params["model_side_scope"]),
        feature_contracts=contracts,
    ).to_numpy(dtype=np.float32)
    actual = valid.copy()
    actual["refit_score"] = prediction
    refit_ranks = _timestamp_side_ranks(actual, prediction, actual["side"])
    actual["refit_rank"] = refit_ranks["rank"].to_numpy(dtype=np.int32)
    actual["refit_top30"] = (
        refit_ranks["rank"].to_numpy(dtype=np.int64)
        <= np.ceil(refit_ranks["group_rows"].to_numpy(dtype=np.float64) * BASE_TOP_FRAC)
    )
    cached = _canonicalize(pd.read_parquet(cached_scored_path), role="cached July scored ledger")
    required = ("score", "base_rank_within_timestamp_side", "selected_top30")
    missing = [name for name in required if name not in cached.columns]
    if missing:
        raise ValueError(f"Cached scored ledger misses parity fields: {missing}")
    overlap = actual.merge(
        cached.loc[:, [*KEY_COLUMNS, "score", "base_rank_within_timestamp_side", "selected_top30"]],
        on=list(KEY_COLUMNS), how="inner", validate="one_to_one",
    )
    if len(overlap) != len(actual):
        raise RuntimeError(f"Parity join lost rows: refit={len(actual)} joined={len(overlap)}")
    overlap["score_abs_delta"] = np.abs(
        overlap["refit_score"].to_numpy(dtype=np.float64) - overlap["score"].to_numpy(dtype=np.float64)
    )
    overlap["rank_match"] = overlap["refit_rank"].astype(int).eq(overlap["base_rank_within_timestamp_side"].astype(int))
    overlap["top30_match"] = overlap["refit_top30"].astype(bool).eq(overlap["selected_top30"].astype(bool))
    summary = {
        "rows": int(len(overlap)),
        "max_abs_score_delta": float(overlap["score_abs_delta"].max()),
        "mean_abs_score_delta": float(overlap["score_abs_delta"].mean()),
        "rank_exact_match_rate": float(overlap["rank_match"].mean()),
        "top30_overlap_rate": float(overlap["top30_match"].mean()),
        "top30_jaccard": float(
            (
                overlap["refit_top30"].astype(bool) & overlap["selected_top30"].astype(bool)
            ).sum()
            / max(
                (
                    overlap["refit_top30"].astype(bool) | overlap["selected_top30"].astype(bool)
                ).sum(),
                1,
            )
        ),
        "seed_before_side_offsets": JULY_BASE_SEED,
        "cached_scored_path": str(cached_scored_path),
    }
    return overlap, summary


def _exclude_oos_outcomes(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    protected = set(OUTCOME_COLUMNS) | {
        "target_soft", "target_hard", "first_touch_net", "first_touch_gross",
        "ev_after_1pct", "exec_margin", "clean_exec", "dirty_positive",
        "timeout", "full_path_bad_mae_1r", "first_touch_bad_mae_1r",
    }
    present = sorted(name for name in protected if name in frame.columns)
    return frame.drop(columns=present, errors="ignore"), present


def _assign_observable_policy_archetypes(frame: pd.DataFrame) -> pd.DataFrame:
    """Assign the same pre-entry side/family policy key used by inference."""

    out = frame.copy()
    keys: list[str] = []
    for idx in out.index:
        side = str(out.at[idx, "side_name"]).lower()
        predicted = predict_observable_policy_archetype(
            side=side,
            candidate_feature_row=out.loc[[idx]],
            meta_model_input_row=None,
        )
        prefix = f"{side}__"
        key = predicted[len(prefix) :] if predicted.startswith(prefix) else predicted
        if not key:
            raise RuntimeError(
                "Observable policy archetype assignment failed for "
                f"timestamp={out.at[idx, '__ts__']} symbol={out.at[idx, '__symbol__']} "
                f"side={side}"
            )
        keys.append(key)
    out["__archetype_policy_key__"] = keys
    out["archetype_policy_key"] = keys
    out["archetype_label_family"] = keys
    out["__archetype_label_family__"] = keys
    out["policy_archetype"] = keys
    out["local_side_archetype"] = [
        f"{side}__{key}" for side, key in zip(out["side_name"], keys)
    ]
    out["policy_archetype_assignment_source"] = "observable_regime"
    return out


def _assign_frozen_source_score_tags(
    candidates: pd.DataFrame, *, source_contract: Mapping[str, Any]
) -> pd.DataFrame:
    """Materialize the pre-prior source tags used by the July meta fold."""

    if str(source_contract.get("source_tag_mode")) != "fallback_side_score_intensity":
        raise ValueError(f"Unsupported frozen source-tag contract: {source_contract}")
    raw_edges = list(source_contract.get("edges", []) or [])
    if len(raw_edges) != 11:
        raise ValueError("Frozen source-tag contract must contain 11 decile edges")
    edges = np.asarray(
        [
            -np.inf if idx == 0 and value is None else
            np.inf if idx == len(raw_edges) - 1 and value is None else
            float(value)
            for idx, value in enumerate(raw_edges)
        ],
        dtype=np.float64,
    )
    bucket = _apply_edges(candidates["score"], edges, "source_score_decile")
    intensity = bucket.map(
        {
            "source_score_decile__q9": "model_frontier_top10",
            "source_score_decile__q8": "model_frontier_top20",
            "source_score_decile__q7": "model_frontier_top30",
        }
    ).fillna("model_candidate_background").reset_index(drop=True)
    out = candidates.copy()
    side = out["side_name"].astype(str).str.lower().reset_index(drop=True)
    out["source_tag"] = (side + "__" + intensity.astype(str)).to_numpy()
    out["source_family"] = (side + "__s52_score_intensity").to_numpy()
    return out


def _apply_recovered_handoff_base_context(
    train: pd.DataFrame, candidates: pd.DataFrame
) -> pd.DataFrame:
    """Apply the exact pre-regime base context encoded in the handoff.

    The original context state was not serialized, but its group parameters are
    algebraically identifiable from the stored score, margin and z-score
    columns.  Recovering them is exact and avoids refitting on the already
    filtered top-30 handoff.
    """

    required = (
        "score",
        "side_name",
        "source_tag",
        "base_margin_to_cutoff",
        "base_margin_to_cutoff_z",
        "base_signal_zscore_within_archetype",
    )
    missing = [name for name in required if name not in train.columns]
    if missing:
        raise ValueError(f"Meta handoff misses recoverable base-context fields: {missing}")
    work = train.loc[:, required].copy()
    score = pd.to_numeric(work["score"], errors="coerce")
    margin = pd.to_numeric(work["base_margin_to_cutoff"], errors="coerce")
    margin_z = pd.to_numeric(work["base_margin_to_cutoff_z"], errors="coerce")
    signal_z = pd.to_numeric(
        work["base_signal_zscore_within_archetype"], errors="coerce"
    )
    std = (margin / margin_z.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
    work["_cutoff"] = score - margin
    work["_std"] = std
    work["_mean"] = score - signal_z * std
    group_cols = ["side_name", "source_tag"]
    state = (
        work.groupby(group_cols, observed=True, dropna=False)[["_cutoff", "_mean", "_std"]]
        .median()
        .reset_index()
    )
    global_cutoff = float(work["_cutoff"].median())
    global_mean = float(work["_mean"].median())
    global_std = float(work["_std"].median())
    out = candidates.merge(state, on=group_cols, how="left", validate="many_to_one")
    cutoff = pd.to_numeric(out.pop("_cutoff"), errors="coerce").fillna(global_cutoff)
    mean = pd.to_numeric(out.pop("_mean"), errors="coerce").fillna(global_mean)
    std = (
        pd.to_numeric(out.pop("_std"), errors="coerce")
        .replace(0.0, np.nan)
        .fillna(global_std)
    )
    candidate_score = pd.to_numeric(out["score"], errors="coerce")
    out["base_margin_to_cutoff"] = (candidate_score - cutoff).astype(np.float32)
    out["base_margin_to_cutoff_z"] = (
        (candidate_score - cutoff) / std
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
    out["base_signal_zscore_within_archetype"] = (
        (candidate_score - mean) / std
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
    return out


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-features", type=Path, default=None, help="Canonical pre-entry parquet to score.")
    parser.add_argument(
        "--feature-root",
        type=Path,
        default=None,
        help="Shared static feature-store directory used when no materialized parquet is supplied.",
    )
    parser.add_argument(
        "--symbols-file",
        type=Path,
        default=None,
        help="Optional newline/comma-separated symbol allowlist for static-store scoring.",
    )
    parser.add_argument(
        "--symbols-from",
        type=Path,
        default=None,
        help=(
            "Optional historical ledger used only to recover the frozen symbol "
            "universe. Unlike --row-universe it does not filter forward rows."
        ),
    )
    parser.add_argument(
        "--row-universe",
        type=Path,
        default=None,
        help="Optional parquet whose timestamp/symbol/side keys define eligible scoring rows.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--base-run", type=Path, default=DEFAULT_BASE_RUN)
    parser.add_argument(
        "--base-model-dir",
        type=Path,
        default=None,
        help="Reuse a previously parity-validated reconstructed July base-fold model directory.",
    )
    parser.add_argument("--fold-cache", type=Path, default=DEFAULT_FOLD)
    parser.add_argument("--meta-model-dir", type=Path, default=DEFAULT_META_MODELS)
    parser.add_argument("--meta-train-frame", type=Path, default=DEFAULT_META_TRAIN)
    parser.add_argument(
        "--meta-ledger-frame",
        type=Path,
        default=DEFAULT_META_LEDGER,
        help="Outcome ledger joined to the frozen handoff to reproduce train-only reliability targets.",
    )
    parser.add_argument("--ae-gmm-state", type=Path, default=DEFAULT_AE_GMM_STATE)
    parser.add_argument(
        "--residual-bundle",
        type=Path,
        default=None,
        help=(
            "Optional frozen side residual bundle. Its feature contract is "
            "loaded from the shared store and scored without refitting."
        ),
    )
    parser.add_argument(
        "--v9-policy-root",
        type=Path,
        default=None,
        help=(
            "Optional V9-only policy_params directory. Its frozen overlay "
            "input contract is included in the static-store request so the "
            "output can proceed through strict V9 plus hierarchical-EV scoring."
        ),
    )
    parser.add_argument("--cached-july-scored", type=Path, default=DEFAULT_BASE_RUN / "_scored_fold_cache/2026-06-30_2026-07-30.parquet")
    parser.add_argument("--oos-start", default="2026-07-01T00:00:00Z")
    parser.add_argument("--oos-end", default="2026-07-31T00:00:00Z")
    parser.add_argument("--fold-id", type=int, default=JULY_FOLD_ID)
    parser.add_argument("--skip-validation-parity", action="store_true")
    parser.add_argument(
        "--residual-only",
        action="store_true",
        help=(
            "Skip the legacy shared-meta checkpoint and apply the supplied "
            "direct base-residual bundle to the corrected base top-30 handoff."
        ),
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Refit and validate the exact cached July fold, then stop before external scoring.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    if (args.fold_cache / "fold_manifest.json").is_file():
        fold_manifest, payload = _load_fold_payload(args.fold_cache)
    elif args.base_model_dir is not None:
        saved = _read_json(args.base_model_dir / "manifest.json")
        saved_columns = _read_json(args.base_model_dir / "columns.json")
        saved_contracts = dict(saved_columns.get("feature_names_by_side", {}) or {})
        ae_outputs = sorted(
            {
                str(name)
                for names in saved_contracts.values()
                for name in names
                if str(name).startswith(("dae_", "gmm_", "cluster_", "ae_"))
            }
        )
        fold_manifest = {
            "fold": saved["fold"],
            "train_start": saved.get("train_start", ""),
            "valid_start": saved["valid_start"],
            "valid_end": saved["valid_end"],
            "ae_gmm_context_features": ae_outputs,
        }
        payload = {}
    else:
        raise FileNotFoundError(
            f"Fold cache is unavailable and no saved base model was supplied: {args.fold_cache}"
        )
    # A serialized side-model bundle already contains the exact fitted heads;
    # only the side-routing mode is needed for scoring.  Its fold manifest is
    # the authoritative source, avoiding an unrelated base-run HPO artifact.
    if args.base_model_dir is not None:
        saved_params = dict(fold_manifest.get("params", {}) or {})
        params = {**saved_params, "model_side_scope": "separate"}
    else:
        params = _base_params(args.base_run)
    if args.base_model_dir is not None:
        models, contracts, seed = _load_saved_base_models(args.base_model_dir)
    else:
        models, contracts, seed = _fit_base_fold(
            payload=payload, fold_manifest=fold_manifest, params=params, fold_id=int(args.fold_id)
        )
    base_model_manifest = _save_base_models(
        output_dir=output, models=models, contracts=contracts, params=params,
        seed=seed, fold_manifest=fold_manifest,
    )
    parity: dict[str, Any] = {"status": "skipped"}
    if not args.skip_validation_parity:
        if not payload:
            raise ValueError(
                "Validation parity requires the compact fold cache; pass "
                "--skip-validation-parity only for an already parity-validated saved base model."
            )
        detail, parity = _validation_parity(
            payload=payload, models=models, contracts=contracts, params=params,
            cached_scored_path=args.cached_july_scored,
        )
        detail.to_parquet(output / "base_refit_validation_parity_rows.parquet", index=False, compression="zstd")
        if parity["max_abs_score_delta"] > 1e-6 or parity["top30_overlap_rate"] < 1.0:
            raise RuntimeError(f"Exact base fold parity failed: {parity}")

    if args.validate_only:
        (output / "manifest.json").write_text(
            json.dumps(_json_safe({
                "schema": "weighted_packb_frozen_july_oos_scoring_v1",
                "status": "base_validation_only_completed",
                "base_model": base_model_manifest,
                "base_validation_parity": parity,
                "leakage_contract": "refit uses only the compact pre-July July-fold cache",
            }), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(_json_safe(parity), sort_keys=True))
        return
    start = pd.Timestamp(args.oos_start)
    end = pd.Timestamp(args.oos_end)
    start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
    end = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")
    state = load_ae_gmm_state_artifact(args.ae_gmm_state)
    meta_columns = (
        {}
        if args.residual_only
        else _read_json(args.meta_model_dir / "columns.json")
    )
    residual_bundle = (
        SideResidualExpertBundle.load(args.residual_bundle)
        if args.residual_bundle is not None
        else None
    )
    residual_required = set(
        residual_bundle.required_input_features()
        if residual_bundle is not None
        else []
    )
    v9_required: set[str] = set()
    if args.v9_policy_root is not None:
        v9_policy = V9TailPostprocessor.load(
            predecessor_bundle_path=args.v9_policy_root / "v9_tail95_predecessor_bundle.joblib",
            residual_event_state_path=args.v9_policy_root / "residual_event_state.joblib",
            hierarchical_ev_artifact_path=args.v9_policy_root / "composite_policy_regime_ev_calibration.json",
        )
        v9_required = set(v9_policy.required_input_features())
    required_model_sources = set(state.get("feature_columns", []) or [])
    required_model_sources.update(name for values in contracts.values() for name in values)
    required_model_sources.update(_required_meta_sources(meta_columns))
    required_model_sources.update(residual_required)
    required_model_sources.update(v9_required)
    required_model_sources.update(OBSERVABLE_REGIME_FAMILY_SCORE_COLUMNS.values())
    if args.canonical_features is not None:
        raw = _canonicalize(
            pd.read_parquet(args.canonical_features), role="external canonical features"
        )
        canonical_source = str(args.canonical_features)
        canonical_sha256 = _sha256_file(args.canonical_features)
    elif args.feature_root is not None:
        if args.symbols_file is not None:
            symbol_text = args.symbols_file.read_text(encoding="utf-8")
            symbols = [
                value.strip()
                for line in symbol_text.splitlines()
                for value in line.split(",")
                if value.strip()
            ]
        elif args.symbols_from is not None:
            symbols = sorted(
                pd.read_parquet(
                    args.symbols_from, columns=["__symbol__"]
                )["__symbol__"].astype(str).unique()
            )
        elif args.row_universe is not None:
            symbols = sorted(
                pd.read_parquet(
                    args.row_universe, columns=["__symbol__"]
                )["__symbol__"].astype(str).unique()
            )
        elif (args.fold_cache / "valid.parquet").is_file():
            valid_identity = pd.read_parquet(
                args.fold_cache / "valid.parquet", columns=["__symbol__"]
            )
            symbols = sorted(valid_identity["__symbol__"].astype(str).unique())
        else:
            # A frozen side-model bundle may be promoted without preserving a
            # reconstructible fold cache.  The canonical static store remains
            # the source-of-truth universe for a forward replay in that case.
            symbols = sorted(
                path.name[len("symbol=") : -len(".parquet")].replace("_", "/")
                for path in args.feature_root.glob("symbol=*.parquet")
            )
        ae_outputs = _frozen_ae_gmm_output_names(state)
        generated = {
            "score", "side", "__side__", "side_name", "side_name_long",
            "side_name_short", "__ts__", "__symbol__", "selected_top30",
            *ae_outputs,
            *ALL_META_POST_SELECTION_OOD_FEATURE_NAMES,
            *OBSERVABLE_REGIME_FAMILY_SCORE_COLUMNS.values(),
        }
        generated.update(
            name
            for name in required_model_sources
            if str(name).startswith(
                ("dae_", "gmm_", "cluster_", "ae_", "reconstruction_")
            )
        )
        derived_residual = {
            "archetype_policy_key",
            "score_base",
            "base_score_rank_pct_train_prior",
            "base_margin_to_cutoff",
            "base_margin_to_cutoff_z",
            "base_signal_zscore_within_archetype",
        }
        source_features = set(state.get("feature_columns", []) or [])
        source_features.update(
            name for values in contracts.values() for name in values if name not in ae_outputs
        )
        source_features.update(
            name for name in _required_meta_sources(meta_columns) if name not in generated
        )
        source_features.update(
            name
            for name in residual_required
            if name not in generated and name not in derived_residual
        )
        source_features.update(
            name
            for name in v9_required
            if name not in generated and name not in derived_residual
        )
        source_features.update(
            name[len("__meta_raw__") :]
            for name in list(source_features)
            if name.startswith("__meta_raw__")
        )
        raw = _canonicalize(
            _materialize_static_oos_frame(
                feature_root=args.feature_root,
                feature_keys=sorted(source_features),
                symbols=symbols,
                start=start,
                end=end,
            ),
            role="shared static OOS features",
        )
        canonical_source = f"static_feature_store:{args.feature_root}"
        canonical_sha256 = None
    else:
        raise ValueError(
            "Provide --canonical-features or --feature-root unless --validate-only is set"
        )
    raw = raw.loc[raw["__ts__"].ge(start) & raw["__ts__"].lt(end)].reset_index(drop=True)
    if args.row_universe is not None:
        eligible = _canonicalize(
            pd.read_parquet(
                args.row_universe,
                columns=["__ts__", "__symbol__", "side_name"],
            ),
            role="external scoring row universe",
        )
        eligible = eligible.loc[
            eligible["__ts__"].ge(start) & eligible["__ts__"].lt(end),
            ["__ts__", "__symbol__", "side_name"],
        ].drop_duplicates()
        raw = raw.merge(
            eligible,
            on=["__ts__", "__symbol__", "side_name"],
            how="inner",
            validate="one_to_one",
        )
    if raw.empty:
        raise ValueError("No canonical rows fall inside the requested OOS interval")
    raw = _hydrate_live_gated_inputs(
        raw,
        data_root=args.feature_root.parents[1] if args.feature_root is not None else ROOT / "data_perp",
        symbols=sorted(raw["__symbol__"].astype(str).unique()),
        timestamps=pd.DatetimeIndex(sorted(raw["__ts__"].unique())),
        required_columns=sorted(required_model_sources),
    )
    raw = _materialize_observable_overlays(
        raw, required_columns=sorted(required_model_sources)
    )
    with_aegmm = _apply_frozen_ae_gmm(raw, state)
    scored = _score_base(models=models, contracts=contracts, frame=with_aegmm, params=params)
    scored.to_parquet(output / "base_scored_all.parquet", index=False, compression="zstd")
    candidates = scored.loc[
        scored["base_input_complete"].eq(1)
        & scored["ae_gmm_input_complete"].eq(1)
        & scored["selected_top30"].astype(bool)
    ].copy()
    # Use the trainer's canonical join. It materializes train-only aliases such
    # as clean_exec_label and bad_path_label that reliability/hit-surprise
    # priors require. Reading either parquet alone silently changes those
    # priors because the helper functions fall back to zero targets.
    train = _canonicalize(
        _load_joined_frame(
            args.meta_train_frame,
            args.meta_ledger_frame,
            "top30",
        ),
        role="meta prior training frame",
    )
    meta_prior_contract = _validate_meta_prior_training_contract(train)
    handoff_manifest_path = args.meta_train_frame.parent / "manifest.json"
    handoff_manifest = _read_json(handoff_manifest_path)
    source_contract = dict(handoff_manifest.get("source_contract", {}) or {})
    if not source_contract:
        raise ValueError(
            f"Meta handoff manifest has no frozen source contract: {handoff_manifest_path}"
        )
    # Freeze all train-derived priors and regime references at the same cutoff
    # used by the serialized July models. A replay start later in July must not
    # move the model's information set forward.
    train = train.loc[train["__ts__"].lt(JULY_TRAIN_END)].copy()
    if "__label_path_end_ts__" in train.columns:
        resolution = _utc(train["__label_path_end_ts__"], name="meta prior label end")
        train = train.loc[resolution.lt(JULY_TRAIN_END)].copy()
    if train.empty:
        raise ValueError("No resolved pre-OOS rows remain for frozen meta priors")
    candidates = _assign_observable_policy_archetypes(candidates)
    candidates, excluded_outcome_columns = _exclude_oos_outcomes(candidates)
    if args.residual_only:
        if residual_bundle is None:
            raise ValueError("--residual-only requires --residual-bundle")
        meta_scored = _assign_frozen_source_score_tags(
            candidates, source_contract=source_contract
        )
        meta_scored = _append_candidate_base_score_context(meta_scored)
        meta_scored = _apply_recovered_handoff_base_context(train, meta_scored)
        prior_cutoff = pd.Timestamp(
            residual_bundle.payload["feature_selection_fit_end_exclusive"]
        )
        prior_cutoff = (
            prior_cutoff.tz_localize("UTC")
            if prior_cutoff.tzinfo is None
            else prior_cutoff.tz_convert("UTC")
        )
        meta_scored = _append_frozen_base_train_prior_rank(
            train, meta_scored, fit_end_exclusive=prior_cutoff
        )
        meta_scored["score_base"] = pd.to_numeric(
            meta_scored["score"], errors="coerce"
        ).astype(np.float32)
        meta_info = {
            "mode": "direct_base_residual_only",
            "legacy_meta_checkpoint_applied": False,
        }
    else:
        regime_fit_months = list(handoff_manifest.get("fit_months", []) or [])
        meta_scored, meta_info = _score_meta(
            train=train,
            candidates=candidates,
            meta_model_dir=args.meta_model_dir,
            source_contract=source_contract,
            regime_fit_months=regime_fit_months,
        )
    if residual_bundle is not None:
        if "score_base" not in meta_scored:
            meta_scored["score_base"] = pd.to_numeric(
                meta_scored["score"], errors="coerce"
            ).astype(np.float32)
        residual_scores = residual_bundle.transform(meta_scored)
        for name in residual_scores.columns:
            meta_scored[name] = residual_scores[name].to_numpy(copy=False)
        meta_info["residual_bundle"] = str(args.residual_bundle)
        meta_info["residual_complete_rows"] = int(
            residual_scores["meta_residual_expert_complete_case"].sum()
        )
    meta_scored.to_parquet(output / "weighted_packb_meta_candidates.parquet", index=False, compression="zstd")
    manifest = {
        "schema": "weighted_packb_frozen_july_oos_scoring_v1",
        "status": "completed",
        "oos_start": start,
        "oos_end_exclusive": end,
        "base_model": base_model_manifest,
        "base_validation_parity": parity,
        "base_top30_scope": "timestamp_side",
        "base_top30_fraction": BASE_TOP_FRAC,
        "canonical_feature_frame": canonical_source,
        "canonical_feature_frame_sha256": canonical_sha256,
        "ae_gmm_state": str(args.ae_gmm_state),
        "ae_gmm_state_sha256": _sha256_file(args.ae_gmm_state),
        "ae_gmm_input_feature_count": int(len(state.get("feature_columns", []))),
        "v9_policy_root": str(args.v9_policy_root) if args.v9_policy_root else None,
        "v9_required_input_feature_count": int(len(v9_required)),
        "meta_train_frame": str(args.meta_train_frame),
        "meta_train_frame_sha256": _sha256_file(args.meta_train_frame),
        "meta_ledger_frame": str(args.meta_ledger_frame),
        "meta_ledger_frame_sha256": _sha256_file(args.meta_ledger_frame),
        "meta_handoff_manifest": str(handoff_manifest_path),
        "meta_train_rows_resolved_pre_oos": int(len(train)),
        "meta_prior_training_contract": meta_prior_contract,
        "meta": _json_safe(meta_info),
        "rows": {
            "canonical": int(len(raw)),
            "base_complete": int(scored["base_input_complete"].sum()),
            "ae_gmm_complete": int(scored["ae_gmm_input_complete"].sum()),
            "base_top30_timestamp_side": int(len(candidates)),
            "meta_scored": int(len(meta_scored)),
        },
        "outcome_columns_excluded_from_oos_meta_input": excluded_outcome_columns,
        "leakage_contract": {
            "base": "reconstructed only from compact pre-July fold cache with exact trial and seed",
            "ae_gmm": "serialized cycle-frozen state only; no refit on OOS rows",
            "meta": "saved July long/short models; train-derived priors use the canonical handoff + outcome-ledger join and resolved pre-OOS rows only",
            "oos": "all scored canonical rows lie in [oos_start, oos_end_exclusive)",
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(_json_safe(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(_json_safe(manifest["rows"]), sort_keys=True))
    print(f"[complete] output={output}")


if __name__ == "__main__":
    main()
