"""
Model Orchestrator for Inference.

This module orchestrates the full inference chain:
2. Alpha model predictions (long_mr, long_tf, short_mr, short_tf)
3. Compute disagreement features
4. Meta model predictions
5. Ridge position sizing
6. Entry policy (Limit Offset Optimizer)

Returns full prediction chain results for each candidate.
"""

import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from extreme_price_movements.engine import _calculate_disagreement_features
from extreme_price_movements.entry_policy import (
    compute_entry_policy_decision,
    flatten_bucket_policy,
)
from extreme_price_movements.inference.feature_generator import (
    get_features_for_candidates,
    is_model_derived_feature_key,
)
from extreme_price_movements.inference.parity import (
    LIVE_UNAVAILABLE_FEATURES,
    strategy_core_id,
    strategy_id_matches,
    strategy_side,
)
from extreme_price_movements.regime_adaptor import (
    apply_regime_adaptor,
    regime_adaptor_inference_enabled,
)
from extreme_price_movements.utils import tprint


def _extract_ebm_contract_model(model: Any) -> Any:
    """Return the EBM contract-bearing model nested in a meta wrapper, if any."""
    if model is None:
        return None
    if model.__class__.__name__ == "EBMOnLGBMModel":
        return model
    best_model = getattr(model, "best_model", None)
    if best_model is not None and best_model.__class__.__name__ == "EBMOnLGBMModel":
        return best_model
    ebm_model = getattr(model, "ebm_model", None)
    if ebm_model is not None and ebm_model.__class__.__name__ == "EBMOnLGBMModel":
        return ebm_model
    return None


def _missing_ebm_raw_contract(model: Any, features: pd.DataFrame) -> list[str]:
    """List required raw EBM features absent from a live inference frame."""
    ebm_model = _extract_ebm_contract_model(model)
    if ebm_model is None or not isinstance(features, pd.DataFrame):
        return []
    raw_features = [
        str(c) for c in (getattr(ebm_model, "raw_selected_features", []) or [])
    ]
    if not raw_features:
        return []
    available = set(map(str, features.columns))
    missing = [name for name in raw_features if name not in available]
    positional_mapping = (
        getattr(ebm_model, "positional_feature_mapping", None)
        or getattr(ebm_model, "meta_positional_feature_mapping_", None)
        or getattr(model, "positional_feature_mapping", None)
        or getattr(model, "meta_positional_feature_mapping_", None)
        or {}
    )
    if missing and isinstance(positional_mapping, dict) and positional_mapping:
        still_missing: list[str] = []
        for raw_name in missing:
            real_name = str(positional_mapping.get(raw_name, ""))
            if not real_name or real_name not in available:
                still_missing.append(real_name or raw_name)
        return still_missing
    return missing


def _synthetic_ebm_raw_features(model: Any) -> list[str]:
    """Return an EBM f0/f1/... raw contract when the model uses one."""
    ebm_model = _extract_ebm_contract_model(model)
    if ebm_model is None:
        return []
    raw_features = [
        str(c) for c in (getattr(ebm_model, "raw_selected_features", []) or [])
    ]
    if raw_features and all(re.fullmatch(r"f\d+", name) for name in raw_features):
        return raw_features
    return []


def _alpha_prediction_frame_for_model(
    model: Any,
    aligned_features: pd.DataFrame,
    feat_cols: List[str],
) -> pd.DataFrame:
    """Return the actual prediction frame expected by a persisted alpha model.

    Most alpha model bundles keep ``feat_cols`` as the real feature contract.
    Some older ModelRace/LGBM bundles persisted the inner stability model after
    feature selection with synthetic ``fN`` names, where ``N`` is the position
    in ``feat_cols``. Build that frame explicitly so inference, debug dumps, and
    replay all exercise the same model input contract.
    """
    if aligned_features.empty or not feat_cols:
        return aligned_features

    feat_cols = [str(c) for c in feat_cols]
    X = aligned_features.reindex(columns=feat_cols, fill_value=0.0).fillna(0.0)
    inner = getattr(model, "best_model", model)
    selected = [str(c) for c in (getattr(inner, "selected_features", []) or [])]
    input_features = [
        str(c) for c in (getattr(inner, "input_feature_names", []) or [])
    ]
    has_named_aliases = (
        len(input_features) == len(selected)
        and bool(input_features)
        and input_features != selected
    )
    synthetic_selected = bool(selected) and all(
        re.fullmatch(r"f\d+", name) is not None for name in selected
    )
    if synthetic_selected and not has_named_aliases:
        mapped = pd.DataFrame(index=X.index)
        for name in selected:
            pos = int(name[1:])
            real_name = feat_cols[pos] if pos < len(feat_cols) else ""
            mapped[name] = X[real_name] if real_name in X.columns else 0.0
        return mapped.replace([np.inf, -np.inf], 0.0).fillna(0.0)

    synthetic_raw = _synthetic_ebm_raw_features(model)
    if synthetic_raw and len(synthetic_raw) == len(feat_cols):
        X = X.copy()
        X.columns = synthetic_raw
    return X


class ModelOrchestrator:
    """Orchestrates model inference pipeline with proper prediction order."""

    def __init__(
        self,
        model_bundle: Dict[str, Any],
        runtime_cfg: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the model orchestrator.

        Args:
            model_bundle: Loaded model bundle from model_loader
            runtime_cfg: Runtime configuration (optional, contains entry_policy_config, etc.)
        """
        self.cfg = runtime_cfg or {}
        self.full_state = (
            model_bundle
            if isinstance(model_bundle, dict) and "bundle" in model_bundle
            else {}
        )
        loaded_bundle = (
            model_bundle.get("bundle", {})
            if isinstance(model_bundle, dict) and "bundle" in model_bundle
            else (model_bundle or {})
        )
        runtime_bundle = (
            self.cfg.get("model_bundle", {}) if isinstance(self.cfg, dict) else {}
        )
        if (
            isinstance(model_bundle, dict)
            and "bundle" in model_bundle
            and isinstance(runtime_bundle, dict)
            and runtime_bundle
        ):
            self.bundle = dict(loaded_bundle)
            for key, value in runtime_bundle.items():
                if value:
                    self.bundle[key] = value
        else:
            self.bundle = loaded_bundle

        # Extract models from bundle
        self.alpha_models = self.bundle.get("alpha_models", {})
        self.alpha_by_strategy = self._build_alpha_strategy_index()
        self.meta_models = self.bundle.get("meta_models", {})
        self.spike_models = self.bundle.get("spike_models", {})  # GMM models
        self.ridge_weights = self.bundle.get("ridge_weights", {})
        self.bucket_params = (
            self.full_state.get("bucket_params", {})
            if isinstance(self.full_state, dict)
            and self.full_state.get("bucket_params")
            else self.bundle.get("bucket_params", {})
        )
        self.ridge_sizer = (
            self.full_state.get("ridge_sizer")
            if isinstance(self.full_state, dict)
            else None
        )
        self.booster_bundles = (
            self.full_state.get("booster_bundles", {})
            if isinstance(self.full_state, dict)
            else {}
        )
        self.regime_adaptors = (
            self.full_state.get("regime_adaptors", {})
            if isinstance(self.full_state, dict)
            else {}
        )
        self.ridge_params_per_bucket = {}
        if isinstance(self.ridge_weights, dict):
            self.ridge_params_per_bucket = (
                self.ridge_weights.get("params_per_bucket", {}) or {}
            )
            self.ridge_weight_map = self.ridge_weights.get("weights", {}) or {}
        else:
            self.ridge_weight_map = {}
        self._last_results: Dict[str, Any] = {}

        # Entry policy config from runtime_cfg or bucket_params
        self.entry_policy_config = self.cfg.get(
            "entry_policy_config"
        ) or self.bucket_params.get("entry_policy")

        # Extract feature columns from alpha models
        self.feature_columns = self._extract_feature_columns()

    def _build_alpha_strategy_index(self) -> Dict[str, Dict[str, Any]]:
        """Normalize flat and nested alpha bundle layouts."""
        out: Dict[str, Dict[str, Any]] = {}
        if not isinstance(self.alpha_models, dict):
            return out
        for key, value in self.alpha_models.items():
            if not isinstance(value, dict):
                continue
            if "model" in value or "feat_cols" in value:
                out[str(key)] = value
                continue
            for nested_key, model_info in value.items():
                if isinstance(model_info, dict):
                    out[f"{key}_{nested_key}"] = model_info
        return out

    def available_strategies(
        self, side: str, allowed: Optional[set[str]] = None
    ) -> List[str]:
        """Return loaded strategies for a side after optional selection filtering."""
        side_l = str(side).lower()
        selected: List[str] = []
        for sid in sorted(self.alpha_by_strategy.keys()):
            inferred = strategy_side(sid)
            if inferred and inferred != side_l:
                continue
            if not strategy_id_matches(sid, allowed):
                continue
            selected.append(sid)
        return selected

    def _normalize_bucket_key(self, side: str, kind: str) -> str:
        return f"{str(side).lower()}_{str(kind).lower()}"

    def _policy_bucket_key(self, side: str, kind: str) -> str:
        core = strategy_core_id(str(kind or ""))
        if core and core not in {"mr", "tf", "none"}:
            return core
        return self._normalize_bucket_key(side, kind)

    def _align_alpha_feature_contract(
        self,
        features: pd.DataFrame,
        feat_cols: List[str],
    ) -> pd.DataFrame:
        """Return a contract-aligned feature frame for an alpha model.

        The alpha bundles were trained on a fixed feature contract. At
        inference time we first try to synthesize missing gated features from
        the shared market columns, then fall back to zero-filled reindexing so
        the model always receives the expected width.
        """
        if features.empty or not feat_cols:
            return features

        aligned = features.copy()

        if "G_VOL" in feat_cols and "G_VOL" not in aligned.columns:
            if {"mkt_rv", "mkt_rv_med"}.issubset(aligned.columns):
                aligned["G_VOL"] = (
                    aligned["mkt_rv"].astype(float)
                    > aligned["mkt_rv_med"].astype(float)
                ).astype(np.float32)

        if "G_TREND" in feat_cols and "G_TREND" not in aligned.columns:
            if {"mkt_ret24h", "mkt_rv"}.issubset(aligned.columns):
                daily_vol = aligned["mkt_rv"].astype(float) * np.sqrt(24.0)
                dyn_thr = np.maximum(daily_vol * 1.5, 0.005)
                aligned["G_TREND"] = (
                    aligned["mkt_ret24h"].astype(float).abs() > dyn_thr
                ).astype(np.float32)

        for gate_name in ("G_VOL", "G_TREND"):
            if gate_name not in aligned.columns:
                continue
            gate_series = aligned[gate_name].astype(np.float32)
            for feat_name in feat_cols:
                if feat_name in aligned.columns or f"_{gate_name}_" not in feat_name:
                    continue
                base_part, state_part = feat_name.rsplit(f"_{gate_name}_", 1)
                if state_part not in {"0", "1"} or base_part not in aligned.columns:
                    continue
                base_vals = aligned[base_part].astype(np.float32)
                if state_part == "1":
                    aligned[feat_name] = (base_vals * gate_series).astype(np.float32)
                else:
                    aligned[feat_name] = (base_vals * (1.0 - gate_series)).astype(
                        np.float32
                    )

        return aligned.reindex(columns=feat_cols, fill_value=0.0).fillna(0.0)

    def _get_bucket_policy(self, side: str, kind: str) -> Dict[str, Any]:
        bucket_key = self._policy_bucket_key(side, kind)
        bucket_cfg = {}
        if isinstance(self.ridge_params_per_bucket, dict):
            bucket_cfg = self.ridge_params_per_bucket.get(bucket_key, {}) or {}
        if not bucket_cfg and isinstance(self.bucket_params, dict):
            buckets = (
                self.bucket_params.get("buckets", {})
                if "buckets" in self.bucket_params
                else {}
            )
            bucket_cfg = (
                buckets.get(bucket_key.upper(), {})
                or buckets.get(bucket_key, {})
                or self.bucket_params.get(bucket_key, {})
                or {}
            )
        if isinstance(bucket_cfg, dict):
            return bucket_cfg
        return {}

    def _materialize_symbol_features(
        self,
        symbol: str,
        features: Any,
    ) -> pd.DataFrame:
        """Return a single-row feature frame for a symbol from either a DataFrame or feature dict."""
        if isinstance(features, pd.Series):
            features = features.to_frame().T
        if isinstance(features, pd.DataFrame):
            if symbol in features.index:
                return features.loc[[symbol]].copy()
            return features.copy()
        if isinstance(features, dict):
            df = get_features_for_candidates(features, [symbol])
            if isinstance(df, pd.DataFrame) and not df.empty:
                return df
        return pd.DataFrame(index=[symbol])

    def _latest_panel_price(self, symbol: str, panel: Any) -> float:
        if not isinstance(panel, dict):
            return 1.0
        close = panel.get("close")
        if not isinstance(close, pd.DataFrame) or symbol not in close.columns:
            return 1.0
        series = close[symbol].dropna()
        if series.empty:
            return 1.0
        price = float(series.iloc[-1])
        return price if np.isfinite(price) and price > 0.0 else 1.0

    def _latest_atr_frac(self, features: pd.DataFrame) -> float:
        for col in ("atr_pct", "atr_pct_base", "realized_volatility_24h"):
            if col not in features.columns:
                continue
            val = float(features[col].iloc[0])
            if np.isfinite(val) and val > 0.0:
                return val
        return 0.01

    def _extract_feature_columns(self) -> Dict[str, List[str]]:
        """Extract feature column names from all loaded alpha models.

        Returns:
            Dictionary mapping ``"{side}_{kind}"`` to feature columns.
        """
        columns = {}

        if self.alpha_by_strategy:
            for sid, model_info in self.alpha_by_strategy.items():
                if isinstance(model_info, dict):
                    columns[sid] = model_info.get("feat_cols", [])
            return columns

        for side in ["long", "short"]:
            if side not in self.alpha_models:
                continue

            side_models = self.alpha_models.get(side, {})
            if not isinstance(side_models, dict):
                continue

            for kind, model_info in side_models.items():
                if not isinstance(model_info, dict):
                    continue
                feat_cols = model_info.get("feat_cols", [])
                columns[f"{side}_{kind}"] = feat_cols

        return columns

    # =========================================================================
    # STEP 1: GMM Spike Quality Filter
    # =========================================================================

    def predict_alpha(
        self,
        features: pd.DataFrame,
        side: str,
        kind: str = "mr",
    ) -> pd.Series:
        """Run alpha model prediction (Step 2: Base/Alpha model predictions).

        Args:
            features: Feature DataFrame (symbols as index)
            side: "long" or "short"
            kind: "mr" (mean reversion) or "tf" (trend following)

        Returns:
            Series of predictions indexed by symbol
        """
        key = str(kind)
        model_info = self.alpha_by_strategy.get(key)
        if model_info is None:
            nested_key = f"{side}_{kind}"
            model_info = self.alpha_by_strategy.get(nested_key)
            key = nested_key if model_info is not None else key
        if model_info is None:
            tprint(f"Warning: Alpha model not found for {key}")
            return pd.Series(dtype=float)

        model = model_info.get("model")
        feat_cols = model_info.get("feat_cols", [])

        if model is None:
            tprint(f"Warning: Model not loaded for {key}")
            return pd.Series(dtype=float)

        aligned_features = self._align_alpha_feature_contract(features, feat_cols)

        if aligned_features.empty:
            tprint(f"Warning: No matching features for {key}")
            return pd.Series(dtype=float)

        # Get feature matrix
        X = _alpha_prediction_frame_for_model(model, aligned_features, feat_cols)

        # Predict
        try:
            preds = model.predict(X)
            return pd.Series(preds, index=aligned_features.index)
        except Exception as e:
            tprint(f"Error predicting alpha for {key}: {e}")
            return pd.Series(dtype=float)

    def predict_alpha_all_horizons(
        self,
        features: pd.DataFrame,
        side: str,
    ) -> Dict[str, pd.Series]:
        """Run all loaded alpha model predictions for a side.

        Args:
            features: Feature DataFrame
            side: "long" or "short"

        Returns:
            Dictionary with predictions for each strategy kind (e.g. long_compression_ratio, ...).
        """
        results = {}

        for kind in self.available_strategies(side):
            preds = self.predict_alpha(features, side, kind)
            # Defensive check: ensure preds is a DataFrame/Series
            if isinstance(preds, pd.DataFrame):
                if not preds.empty:
                    results[kind] = preds
            elif isinstance(preds, pd.Series):
                if not preds.empty:
                    results[kind] = preds
            else:
                # Skip if preds is not a DataFrame or Series
                continue

        return results

    def predict_alpha_all_kinds(
        self,
        features: pd.DataFrame,
        side: str,
    ) -> Dict[str, pd.Series]:
        """Compatibility alias for code that expects a generic strategy fanout."""
        return self.predict_alpha_all_horizons(features, side)

    def get_last_results(self) -> Dict[str, Any]:
        return dict(self._last_results) if isinstance(self._last_results, dict) else {}

    # =========================================================================
    # STEP 3: Compute Disagreement Features
    # =========================================================================

    def compute_disagreement_features(
        self,
        meta_data: pd.DataFrame,
        mr_preds: pd.Series,
        tf_preds: pd.Series,
        kind_name: str = "mr",
    ) -> pd.Series:
        """Step 3: Compute disagreement features between MR and TF predictions.

        Replicates the _calculate_disagreement_features from engine.py.

        Args:
            meta_data: Meta features DataFrame
            mr_preds: Mean reversion predictions
            tf_preds: Trend following predictions
            kind_name: Name for logging

        Returns:
            Series of disagreement features
        """
        try:
            return _calculate_disagreement_features(
                meta_data=meta_data,
                h_preds={"mr": mr_preds, "tf": tf_preds},
                kind_name=kind_name,
            )
        except Exception as e:
            tprint(f"Error computing disagreement features: {e}")
            return pd.Series(0.0, index=meta_data.index)

    def _materialize_meta_model_derived_features(
        self,
        features: pd.DataFrame,
        meta_model: Any,
        *,
        side: str,
        kind: str,
    ) -> pd.DataFrame:
        """Build deterministic live values for train-time model-derived meta keys.

        Raw market features must already be present in ``features``. This helper
        only materializes columns derived from the base prediction itself. For
        historical recent-effectiveness diagnostics, live cannot know the future
        label at decision time, so we use the explicit neutral value instead of
        letting EBM positional mapping silently consume unrelated columns.
        """
        if not isinstance(features, pd.DataFrame) or features.empty:
            return features
        feat_cols = [str(c) for c in (getattr(meta_model, "feature_columns", []) or [])]
        if not feat_cols:
            return features

        out = features.copy()
        kind_s = str(kind)
        core = strategy_core_id(kind_s)
        core_no_head = re.sub(r"_(?:clf|reg|tbm_clf|early_inval)$", "", core)
        kind_no_head = re.sub(r"_(?:clf|reg|tbm_clf|early_inval)$", "", kind_s)
        base_series: pd.Series | None = None
        candidate_cols = [
            kind_s,
            kind_no_head,
            core,
            core_no_head,
            f"{side}_{core}",
            f"{side}_{core_no_head}",
            getattr(meta_model, "meta_feature_contract_", {}).get(
                "base_probability_column", ""
            )
            if isinstance(getattr(meta_model, "meta_feature_contract_", {}), dict)
            else "",
        ]
        candidate_cols.extend([c for c in feat_cols if re.match(r"^pred_.*_H\d+$", c)])
        candidate_cols.extend([c for c in feat_cols if re.match(r"^pred_H\d+$", c)])
        for col in candidate_cols:
            if col and col in out.columns:
                base_series = pd.to_numeric(out[col], errors="coerce").astype(float)
                break
        if base_series is None:
            return out

        base_prob = base_series.clip(1e-6, 1.0 - 1e-6).astype(float)
        base_logit = np.log(base_prob / (1.0 - base_prob))
        added = 0
        for col in feat_cols:
            if col in out.columns:
                continue
            value: pd.Series | float | None = None
            if re.match(r"^pred_logit(?:_H\d+)?$", col):
                value = base_logit
            elif re.match(r"^pred(?:_.*)?_H\d+(?:_ebm_raw|_ebm_en|_ebm_uncertainty_weighted)?$", col):
                value = base_prob
            elif re.match(r"^base_H\d+_ebm_(?:raw|en|uncertainty_weighted)$", col):
                value = base_prob
            elif col in {"base_model_score", "base_med_pred"}:
                value = base_prob
            elif col == "base_model_margin":
                value = (base_prob - 0.5).abs()
            elif col == "base_model_score_pct":
                value = 0.5
            elif col.startswith("base_prob_x_"):
                src = col.removeprefix("base_prob_x_")
                if src in out.columns:
                    value = base_prob * pd.to_numeric(out[src], errors="coerce").fillna(0.0)
                else:
                    value = 0.0
            elif col.startswith("base_med_x_"):
                src = col.removeprefix("base_med_x_")
                if src in out.columns:
                    value = base_prob * pd.to_numeric(out[src], errors="coerce").fillna(0.0)
                else:
                    value = 0.0
            elif is_model_derived_feature_key(col):
                value = 0.0

            if value is not None:
                out[col] = value
                added += 1
        if added and not getattr(self, "_meta_model_derived_warned", False):
            tprint(
                "Meta inference: materialized model-derived contract columns "
                f"from base prediction ({added} columns for {kind})."
            )
            self._meta_model_derived_warned = True
        return out

    # =========================================================================
    # STEP 4: Meta Model Prediction
    # =========================================================================

    def predict_meta(
        self,
        features: pd.DataFrame,
        side: str,
        kind: str = "mr",
    ) -> pd.Series:
        """Step 4: Meta model prediction.

        Args:
            features: Feature DataFrame (with disagreement features and alpha preds)
            side: "long" or "short"
            kind: "mr" or "tf"

        Returns:
            Series of meta predictions
        """
        requested_kind = str(kind)
        key = str(kind)
        if key not in self.meta_models:
            side_key = f"{side}_{kind}"
            clf_key = f"{key}_clf"
            tbm_key = f"{key}_tbm_clf"
            if side_key in self.meta_models:
                key = side_key
            elif clf_key in self.meta_models:
                key = clf_key
            elif tbm_key in self.meta_models:
                key = tbm_key

        if key not in self.meta_models:
            tprint(f"Warning: Meta model not found for {key}")
            return pd.Series(dtype=float)

        meta_model = self.meta_models[key]

        if meta_model is None:
            return pd.Series(dtype=float)

        # Get feature columns from meta model
        try:
            if hasattr(meta_model, "feature_columns"):
                feat_cols = meta_model.feature_columns
            else:
                feat_cols = list(features.columns)
            feat_cols = [str(c) for c in (feat_cols or [])]

            features = self._materialize_meta_model_derived_features(
                features,
                meta_model,
                side=side,
                kind=requested_kind,
            )

            missing_ebm_raw = _missing_ebm_raw_contract(meta_model, features)
            if missing_ebm_raw:
                reason = "missing_ebm_feature_contract"
                self._last_results["meta_contract_error"] = {
                    "key": key,
                    "reason": reason,
                    "missing_raw_features_count": len(missing_ebm_raw),
                    "missing_raw_features_sample": missing_ebm_raw[:20],
                }
                tprint(
                    f"Error predicting meta for {key}: {reason} "
                    f"({len(missing_ebm_raw)} missing raw EBM features)."
                )
                return pd.Series(dtype=float)

            available_cols = [c for c in feat_cols if c in features.columns]

            if not available_cols:
                return pd.Series(dtype=float)

            ebm_contract_model = _extract_ebm_contract_model(meta_model)
            if ebm_contract_model is not None:
                X = features.reindex(columns=feat_cols, fill_value=0.0).fillna(0)
            else:
                X = features[available_cols].fillna(0)
            preds = meta_model.predict(X)

            return pd.Series(preds, index=features.index)
        except Exception as e:
            tprint(f"Error predicting meta for {key}: {e}")
            return pd.Series(dtype=float)

    def _find_strategy_id_for_bucket(self, bucket_key: str) -> str:
        buckets = (
            self.bucket_params.get("buckets", {})
            if isinstance(self.bucket_params, dict)
            else {}
        )
        for sid, cfg in buckets.items():
            if not isinstance(cfg, dict):
                continue
            side = str(cfg.get("side", "")).lower()
            if side and side not in bucket_key:
                continue
            return sid
        return ""

    def _predict_booster(
        self,
        bundle: Dict[str, Any],
        features: pd.DataFrame,
    ) -> Tuple[np.ndarray, np.ndarray]:
        try:
            from extreme_price_movements.simple_position_sizer import (
                clean_and_standardize,
            )
        except ImportError:
            return np.array([]), np.array([])
        feature_keys = bundle.get("feature_keys", [])
        fold_models = bundle.get("fold_models", [])
        if not feature_keys or not fold_models:
            return np.array([]), np.array([])
        feat_cols = list(features.columns)
        feat_idx = [feat_cols.index(k) for k in feature_keys if k in feat_cols]
        if not feat_idx:
            return np.array([]), np.array([])
        X_raw = features.iloc[:, feat_idx].to_numpy(dtype=np.float64)
        winner = bundle.get("winner", "")
        n = X_raw.shape[0]
        pred_sum = np.zeros(n, dtype=np.float64)
        count = 0
        proba_sum = (
            np.zeros((n, 3), dtype=np.float64)
            if winner == "ridge_plus_lgbm_clf"
            else None
        )
        for fd in fold_models:
            if isinstance(fd, dict):
                model = fd["model"]
                medians = fd.get("medians")
                scaler = fd.get("scaler")
                c1d = fd.get("center_1d")
                s1d = fd.get("scale_1d")
            else:
                model = fd
                medians = scaler = c1d = s1d = None
            X_clean, _, _, _, _ = clean_and_standardize(
                X_raw, fit_medians=medians, scaler=scaler, center_1d=c1d, scale_1d=s1d
            )
            if winner == "ridge_plus_lgbm_clf":
                proba = np.asarray(model.predict_proba(X_clean), dtype=np.float32)
                score = proba[:, 0] - proba[:, 2]
                pred_sum += score
                if proba_sum is not None:
                    proba_sum += proba
            else:
                pred_sum += np.asarray(model.predict(X_clean), dtype=np.float64)
            count += 1
        if count == 0:
            return np.array([]), np.array([])
        booster_raw = pred_sum / count
        confidence = np.ones(n, dtype=np.float32)
        if proba_sum is not None and count > 1:
            p_mean = proba_sum / count
            p_mean = np.clip(p_mean, 1e-12, 1.0)
            entropy = -np.sum(p_mean * np.log(p_mean), axis=1)
            max_entropy = np.log(3.0)
            confidence = np.clip(1.0 - entropy / max_entropy, 0.7, 1.3).astype(
                np.float32
            )
        return booster_raw.astype(np.float32), confidence

    # =========================================================================
    # STEP 5: Ridge Position Sizing
    # =========================================================================

    def compute_ridge_position_size(
        self,
        features: pd.DataFrame,
        side: str,
        kind: str = "mr",
    ) -> Tuple[pd.Series, Dict[str, float]]:
        """Step 5: Ridge position sizer with optional booster mix.

        Args:
            features: Feature DataFrame (with meta predictions)
            side: "long" or "short"
            kind: "mr" or "tf"

        Returns:
            Tuple of (position_sizes Series, confidence dict)
        """
        bucket_key = self._policy_bucket_key(side, kind)
        ridge_preds = None
        skipped_unsafe_sizer = False

        if self.ridge_sizer is not None:
            try:
                model_names = getattr(self.ridge_sizer, "feature_names", None)
                if model_names is None:
                    model_names = getattr(self.ridge_sizer, "model_names_", [])
                model_names = list(model_names)
                unavailable = sorted(set(model_names) & LIVE_UNAVAILABLE_FEATURES)
                if unavailable:
                    skipped_unsafe_sizer = True
                    tprint(
                        "Ignoring legacy ridge position sizer for live inference; "
                        f"it requires target-derived fields: {unavailable}"
                    )
                    model_names = []

                if model_names:
                    for col in model_names:
                        if col not in features.columns:
                            features[col] = 0.0

                    if hasattr(self.ridge_sizer, "predict"):
                        ridge_preds = np.asarray(
                            self.ridge_sizer.predict(features), dtype=float
                        )
                    else:
                        model = getattr(self.ridge_sizer, "model", None)
                        if model is not None and hasattr(model, "predict"):
                            X = features[model_names].to_numpy(dtype=float)
                            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
                            ridge_preds = np.asarray(model.predict(X), dtype=float)
            except Exception as e:
                if "live-unavailable" in str(e):
                    raise
                tprint(f"Warning: ridge_sizer.predict failed for {bucket_key}: {e}")

        if ridge_preds is None:
            if not isinstance(self.ridge_weight_map, dict) or not self.ridge_weight_map:
                tprint(f"Warning: Ridge weights not found for {bucket_key}")
                return pd.Series(0.0, index=features.index), {"confidence": 0.0}

            prefix = f"{bucket_key}_"
            bucket_weights = {
                k[len(prefix) :]: float(v)
                for k, v in self.ridge_weight_map.items()
                if isinstance(k, str) and k.startswith(prefix)
            }
            if not bucket_weights:
                fallback_size = self._policy_fallback_position_size(bucket_key)
                if fallback_size > 0.0:
                    if skipped_unsafe_sizer:
                        tprint(
                            f"Using policy fallback sizing for {bucket_key}: "
                            f"{fallback_size:.6g}"
                        )
                    return pd.Series(fallback_size, index=features.index), {
                        "confidence": min(1.0, fallback_size)
                    }
                tprint(f"Warning: No flattened ridge weights found for {bucket_key}")
                return pd.Series(0.0, index=features.index), {"confidence": 0.0}

            feature_names = list(bucket_weights.keys())
            unavailable = sorted(set(feature_names) & LIVE_UNAVAILABLE_FEATURES)
            if unavailable:
                fallback_size = self._policy_fallback_position_size(bucket_key)
                if fallback_size > 0.0:
                    tprint(
                        "Ignoring flattened ridge weights for live inference; "
                        f"they require target-derived fields: {unavailable}. "
                        f"Using policy fallback sizing for {bucket_key}: "
                        f"{fallback_size:.6g}"
                    )
                    return pd.Series(fallback_size, index=features.index), {
                        "confidence": min(1.0, fallback_size)
                    }
                tprint(
                    "Ignoring flattened ridge weights for live inference; "
                    f"they require target-derived fields: {unavailable}"
                )
                return pd.Series(0.0, index=features.index), {"confidence": 0.0}
            X = (
                features.reindex(columns=feature_names, fill_value=0.0)
                .fillna(0.0)
                .to_numpy(dtype=float)
            )
            coefs_array = np.asarray(
                [bucket_weights[c] for c in feature_names], dtype=float
            )
            ridge_preds = np.dot(X, coefs_array)

        # --- Apply booster mix if available ---
        final_preds = ridge_preds.copy()
        mix_meta: Dict[str, float] = {}
        if self.booster_bundles and isinstance(self.bucket_params, dict):
            strategy_id = bucket_key or self._find_strategy_id_for_bucket(bucket_key)
            booster_bundle = None
            if strategy_id:
                booster_bundle = self.booster_bundles.get(strategy_id)
            if not booster_bundle:
                for sid, bb in self.booster_bundles.items():
                    if isinstance(bb, dict) and bb.get("winner"):
                        booster_bundle = bb
                        strategy_id = sid
                        break
            if booster_bundle is not None:
                bucket_cfg = self.bucket_params.get("buckets", {}).get(strategy_id, {})
                if isinstance(bucket_cfg, dict):
                    mix_ridge_w = float(bucket_cfg.get("sizer_mix_ridge_w", 1.0))
                    mix_booster_w = float(bucket_cfg.get("sizer_mix_booster_w", 0.0))
                    mix_conf_mult = float(bucket_cfg.get("sizer_mix_conf_mult", 1.0))
                else:
                    mix_ridge_w = 1.0
                    mix_booster_w = 0.0
                    mix_conf_mult = 1.0
                if mix_booster_w > 0:
                    booster_raw, booster_conf = self._predict_booster(
                        booster_bundle, features
                    )
                    if len(booster_raw) == len(ridge_preds):
                        final_preds = (
                            mix_ridge_w * ridge_preds
                            + mix_booster_w
                            * booster_raw
                            * (booster_conf * mix_conf_mult)
                        )
                        mix_meta["booster_winner"] = 1.0
                        mix_meta["mix_ridge_w"] = mix_ridge_w
                        mix_meta["mix_booster_w"] = mix_booster_w
                        mix_meta["mix_conf_mult"] = mix_conf_mult

        strategy_id_for_regime = bucket_key or self._find_strategy_id_for_bucket(
            bucket_key
        )
        adaptor = None
        if isinstance(self.regime_adaptors, dict):
            adaptor = self.regime_adaptors.get(strategy_id_for_regime)
            if adaptor is None:
                for sid, candidate in self.regime_adaptors.items():
                    if strategy_id_matches(str(sid), {str(strategy_id_for_regime)}):
                        adaptor = candidate
                        break
        if isinstance(adaptor, dict) and regime_adaptor_inference_enabled(
            self.cfg, adaptor
        ):
            try:
                if "symbol" in features.columns:
                    symbols = features["symbol"].astype(str).to_numpy()
                else:
                    symbols = features.index.astype(str).to_numpy()
                applied = apply_regime_adaptor(
                    features,
                    final_preds,
                    adaptor,
                    timestamps=features.index,
                    symbols=symbols,
                )
                regime_weight = np.asarray(
                    applied.get("regime_weight", np.ones(len(final_preds))), dtype=float
                )
                eligible = np.asarray(
                    applied.get("eligible", np.ones(len(final_preds), dtype=bool)),
                    dtype=bool,
                )
                if (
                    "combined_score" in applied
                    and "deployment_score_pre_rank" in applied
                ):
                    # Rolling RegimeAdaptor integration emits only a pre-rank score here.
                    # Portfolio/global or per-side rank normalization must happen downstream
                    # after all strategy × symbol candidates are assembled.
                    final_preds = np.asarray(
                        applied["deployment_score_pre_rank"], dtype=float
                    )
                else:
                    final_preds = final_preds * np.clip(regime_weight, 0.75, 1.20)
                final_preds = np.where(eligible, final_preds, 0.0)
                mix_meta["regime_adaptor_enabled"] = float(
                    bool(np.any(applied.get("regime_adjustment_enabled", [True])))
                )
                mix_meta["regime_eligible_share"] = float(np.mean(eligible))
                mix_meta["regime_weight_mean"] = float(np.mean(regime_weight))
                for key in (
                    "p_bad_regime_global_3d",
                    "p_bad_regime_global_5d",
                    "p_bad_regime_asset_3d",
                    "p_bad_regime_asset_5d",
                    "combined_global_bad_regime_score",
                    "combined_asset_bad_regime_score",
                    "bad_regime_offset",
                    "combined_score",
                    "deployment_score_pre_rank",
                    "local_batch_rank",
                    "score_delta_from_regime_adjustment",
                    "live_required_columns_available",
                ):
                    if key in applied:
                        arr = np.asarray(applied[key], dtype=float)
                        mix_meta[key] = float(np.nanmean(arr)) if len(arr) else 0.0
                for key in (
                    "selected_combination_params",
                    "rank_scope",
                    "regime_disabled_reason",
                    "missing_live_p_bad_regime_columns",
                ):
                    if key in applied:
                        vals = np.asarray(applied[key]).astype(str)
                        mix_meta[key] = vals[0] if len(vals) else ""
            except Exception as exc:
                tprint(f"Warning: regime adaptor failed for {bucket_key}: {exc}")

        conf = np.clip(np.abs(final_preds), 0.0, 1.0)
        return (
            pd.Series(final_preds, index=features.index),
            {"confidence": float(np.nanmean(conf)) if len(conf) else 0.0, **mix_meta},
        )

    def _policy_fallback_position_size(self, bucket_key: str) -> float:
        """Return a conservative live sizing fallback from persisted policy params."""
        if not isinstance(self.bucket_params, dict) or not bucket_key:
            return 0.0
        bucket_cfg = {}
        buckets = self.bucket_params.get("buckets", {})
        if isinstance(buckets, dict):
            bucket_cfg = buckets.get(bucket_key, {}) or buckets.get(
                f"{bucket_key}_tbm", {}
            )
        if not bucket_cfg:
            bucket_cfg = self.bucket_params.get(bucket_key, {}) or {}
        if not isinstance(bucket_cfg, dict):
            return 0.0

        raw = bucket_cfg.get("selection_frac")
        if raw is None:
            raw = bucket_cfg.get("position_size")
        if raw is None:
            return 0.0
        try:
            size = float(raw)
        except (TypeError, ValueError):
            return 0.0
        if not np.isfinite(size) or size <= 0.0:
            return 0.0
        return float(np.clip(size, 0.0, 0.30))

    # =========================================================================
    # STEP 6: Entry Policy (Limit Offset Optimizer)
    # =========================================================================

    def compute_entry_policy(
        self,
        symbol: str,
        side: str,
        meta_pred: float,
        features: pd.DataFrame,
        position_result: Dict[str, Any],
        kind: str = "mr",
        entry_price: float = 1.0,
        atr_frac: float = 0.02,
    ) -> Dict[str, Any]:
        """Step 6: Entry policy with limit offset optimizer.

        Uses compute_entry_policy_decision from entry_policy.py.

        Args:
            symbol: Trading symbol
            side: "long" or "short"
            meta_pred: Meta model prediction (score)
            features: Feature DataFrame
            position_result: Result from compute_ridge_position_size
            entry_price: Entry price (default 1.0 for normalized)
            atr_frac: ATR fraction for offset calculation

        Returns:
            Full entry decision dict with place_order, entry_px, etc.
        """
        # Get feature dict for entry policy
        feat_dict = {}
        if isinstance(features, pd.DataFrame) and not features.empty:
            # Extract key features for entry policy
            for col in features.columns:
                if col in [
                    "u_hat_z",
                    "mae_hat_z",
                    "mfe_hat_z",
                    "dur_hat_z",
                    "u_hat",
                    "mae_hat",
                    "mfe_hat",
                ]:
                    feat_dict[col] = (
                        float(features[col].iloc[0]) if len(features) > 0 else 0.0
                    )

        # Use meta prediction as score
        score = float(meta_pred) if np.isfinite(meta_pred) else 0.0

        # Get entry policy config from bucket_params or runtime_cfg
        bucket_cfg = (
            self.entry_policy_config
            or self._get_bucket_policy(side, kind)
            or self.bucket_params
        )

        # Flatten if needed
        if bucket_cfg:
            bucket_cfg = flatten_bucket_policy(bucket_cfg)

        try:
            decision = compute_entry_policy_decision(
                entry_px=entry_price,
                score=score,
                bucket_cfg=bucket_cfg,
                features=feat_dict,
                **{"atr_frac": atr_frac},
            )

            # Add metadata to decision
            decision["symbol"] = symbol
            decision["side"] = side
            decision["meta_score"] = score
            decision["position_size"] = position_result.get("size", 0.0)

            return decision

        except Exception as e:
            tprint(f"Error computing entry policy for {symbol}: {e}")
            return {
                "place_order": True,  # Default to placing order on error
                "entry_px_fill": entry_price,
                "offset_bps": 0.0,
                "symbol": symbol,
                "side": side,
                "error": str(e),
            }

    # =========================================================================
    # Full Chain Orchestration
    # =========================================================================

    def run_full_chain(
        self,
        symbol: str,
        side: str,
        features: Any,
        panel: Any = None,
        kind: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run predictions in proper order:

        1. Alpha (Base) Model Predictions
        2. Disagreement Features
        3. Meta Model Prediction
        4. Ridge Position Sizing
        5. Entry Policy (Limit Offset Optimizer)

        Args:
            symbol: Trading symbol
            side: "long" or "short"
            features: Feature DataFrame for the candidate

        Returns:
            Dictionary with all results including final action decision
        """
        results = {"symbol": symbol, "side": side}

        # Ensure features is a DataFrame with proper index
        features = self._materialize_symbol_features(symbol, features)
        if features.empty:
            results["action"] = "no_features"
            return results
        if symbol not in features.index:
            features.index = pd.Index([symbol] * len(features))

        if kind is None:
            strategies = self.available_strategies(side)
            kind = strategies[0] if strategies else "mr"
        strategy_id = strategy_core_id(str(kind))
        results["strategy_id"] = strategy_id

        # =====================================================================
        # STEP 2: Alpha (Base) Model Predictions
        # =====================================================================
        alpha_pred = self.predict_alpha(features, side, str(kind))
        alpha_preds = (
            {str(kind): alpha_pred}
            if isinstance(alpha_pred, (pd.DataFrame, pd.Series))
            and not alpha_pred.empty
            else {}
        )
        results["alpha_preds"] = alpha_preds

        if not alpha_preds:
            results["action"] = "no_alpha_predictions"
            return results

        # =====================================================================
        # STEP 3: Compute Disagreement Features
        # =====================================================================
        # Build meta base features
        meta_base = features.copy()

        # Add alpha predictions to meta features
        for pred_key, pred_series in alpha_preds.items():
            if (
                isinstance(pred_series, (pd.DataFrame, pd.Series))
                and not pred_series.empty
            ):
                meta_base[pred_key] = pred_series

        # Compute disagreement features
        key_mr = f"{side}_mr"
        key_tf = f"{side}_tf"

        if key_mr in alpha_preds and key_tf in alpha_preds:
            mr_preds = alpha_preds[key_mr]
            tf_preds = alpha_preds[key_tf]

            disagreement = self.compute_disagreement_features(
                meta_base, mr_preds, tf_preds, side
            )

            if (
                isinstance(disagreement, (pd.DataFrame, pd.Series))
                and not disagreement.empty
            ):
                meta_base["disagreement"] = disagreement
                results["disagreement_features"] = disagreement.to_dict()

        # =====================================================================
        # STEP 4: Meta Model Prediction
        # =====================================================================
        meta_pred = self.predict_meta(meta_base, side, kind)
        if not isinstance(meta_pred, (pd.DataFrame, pd.Series)) or meta_pred.empty:
            results["action"] = "no_meta_prediction"
            results["reason"] = "meta_prediction_missing_no_base_fallback"
            return results

        meta_pred_val = float(meta_pred.iloc[0]) if len(meta_pred) > 0 else 0.0
        if not np.isfinite(meta_pred_val):
            results["action"] = "no_meta_prediction"
            results["reason"] = "meta_prediction_non_finite_no_base_fallback"
            return results
        results["meta_pred"] = meta_pred_val

        # Merge Meta Model with Base Model Predictions
        # Final Prediction = Base Prediction + (Meta Prediction * Volatility Scale)
        base_key = str(kind)
        if base_key in alpha_preds:
            base_pred = alpha_preds[base_key]
            # Try to find vol scale, fallback to 1.0 if not found
            if "atr_pct" in meta_base.columns:
                vol_scale = meta_base["atr_pct"].astype(float).fillna(1.0)
            elif "realized_volatility_24h" in meta_base.columns:
                vol_scale = (
                    meta_base["realized_volatility_24h"].astype(float).fillna(1.0)
                )
            else:
                vol_scale = pd.Series(1.0, index=meta_base.index)

            # Reconstruct the calibrated regression prediction
            calibrated_reg_pred = base_pred + (meta_pred * vol_scale)
            results["calibrated_reg_pred"] = (
                float(calibrated_reg_pred.iloc[0])
                if len(calibrated_reg_pred) > 0
                else 0.0
            )

            meta_base["calibrated_reg_pred"] = calibrated_reg_pred

        # =====================================================================
        # STEP 5: Ridge Position Sizing
        # =====================================================================
        ridge_features = meta_base.copy()
        ridge_features["meta_pred"] = meta_pred
        if "calibrated_reg_pred" in meta_base.columns:
            ridge_features["calibrated_reg_pred"] = meta_base["calibrated_reg_pred"]

        position_size, confidence = self.compute_ridge_position_size(
            ridge_features, side, kind
        )

        position_val = float(position_size.iloc[0]) if len(position_size) > 0 else 0.0
        results["position_size"] = position_val
        results["ridge_confidence"] = confidence.get("confidence", 1.0)

        results["orchestrator_position_size"] = position_val
        if position_val <= 0:
            position_val = 0.05
            results["position_size"] = position_val
            results["sizing_source"] = "meta_policy_placeholder"
        else:
            results["sizing_source"] = "legacy_orchestrator_diagnostic"

        # =====================================================================
        # STEP 6: Entry Policy (Limit Offset Optimizer)
        # =====================================================================
        position_result = {
            "size": position_val,
            "confidence": confidence.get("confidence", 1.0),
        }

        entry_decision = self.compute_entry_policy(
            symbol=symbol,
            side=side,
            meta_pred=meta_pred_val,
            features=features,
            position_result=position_result,
            kind=kind,
            entry_price=self._latest_panel_price(symbol, panel),
            **{"atr_frac": self._latest_atr_frac(features)},
        )

        results["entry_policy"] = entry_decision

        # =====================================================================
        # Final Decision
        # =====================================================================
        if entry_decision.get("place_order", False):
            results["action"] = "enter"
            results["entry_px"] = entry_decision.get("entry_px_fill", 1.0)
            results["size"] = position_val
            results["stop_px"] = entry_decision.get("sl_distance_atr_eff")
            results["target_px"] = entry_decision.get("tp_distance_atr_eff")
            results["offset_bps"] = entry_decision.get("offset_bps", 0.0)
        else:
            results["action"] = "no_entry"
            results["reason"] = entry_decision.get("reason", "entry_policy_rejected")
        self._last_results = dict(results)
        return results

    def run_full_chain_batch(
        self,
        features_df: pd.DataFrame,
        side: str,
    ) -> List[Dict[str, Any]]:
        """Run full chain for multiple symbols.

        Args:
            features_df: DataFrame with features, indexed by symbol
            side: "long" or "short"

        Returns:
            List of results for each symbol
        """
        results = []

        for symbol in features_df.index:
            symbol_features = features_df.loc[symbol:symbol]
            result = self.run_full_chain(symbol, side, symbol_features)
            results.append(result)

        return results


def run_inference_chain(
    model_bundle: Dict[str, Any],
    runtime_cfg: Optional[Dict[str, Any]] = None,
    features: Optional[pd.DataFrame] = None,
    side: str = "long",
    symbol: Optional[str] = None,
) -> Dict[str, Any]:
    """Convenience function to run full inference chain.

    Args:
        model_bundle: Loaded model bundle
        runtime_cfg: Runtime configuration (optional)
        features: Input features DataFrame (optional)
        side: "long" or "short"
        symbol: Trading symbol (optional, used if features is a DataFrame)

    Returns:
        Dictionary with inference results
    """
    orchestrator = ModelOrchestrator(model_bundle, runtime_cfg)

    if features is not None and symbol is not None:
        return orchestrator.run_full_chain(symbol, side, features)
    elif features is not None:
        return orchestrator.run_full_chain_batch(features, side)
    else:
        raise ValueError(
            "Either (features and symbol) or just features must be provided"
        )
