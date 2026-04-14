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

from typing import Dict, List, Any, Optional, Tuple

import pandas as pd
import numpy as np

from extreme_price_movements.engine import _calculate_disagreement_features
from extreme_price_movements.entry_policy import compute_entry_policy_decision, flatten_bucket_policy
from extreme_price_movements.inference.feature_generator import get_features_for_candidates
from extreme_price_movements.utils import tprint


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
        self.full_state = model_bundle if isinstance(model_bundle, dict) and "bundle" in model_bundle else {}
        self.bundle = (
            model_bundle.get("bundle", {})
            if isinstance(model_bundle, dict) and "bundle" in model_bundle
            else (model_bundle or {})
        )

        # Extract models from bundle
        self.alpha_models = self.bundle.get("alpha_models", {})
        self.meta_models = self.bundle.get("meta_models", {})
        self.spike_models = self.bundle.get("spike_models", {})  # GMM models
        self.ridge_weights = self.bundle.get("ridge_weights", {})
        self.bucket_params = (
            self.full_state.get("bucket_params", {})
            if isinstance(self.full_state, dict) and self.full_state.get("bucket_params")
            else self.bundle.get("bucket_params", {})
        )
        self.ridge_sizer = self.full_state.get("ridge_sizer") if isinstance(self.full_state, dict) else None
        self.ridge_params_per_bucket = {}
        if isinstance(self.ridge_weights, dict):
            self.ridge_params_per_bucket = self.ridge_weights.get("params_per_bucket", {}) or {}
            self.ridge_weight_map = self.ridge_weights.get("weights", {}) or {}
        else:
            self.ridge_weight_map = {}
        self._last_results: Dict[str, Any] = {}

        # Entry policy config from runtime_cfg or bucket_params
        self.entry_policy_config = self.cfg.get("entry_policy_config") or self.bucket_params.get("entry_policy")
        
        # Extract feature columns from alpha models
        self.feature_columns = self._extract_feature_columns()

    def _normalize_bucket_key(self, side: str, kind: str) -> str:
        return f"{str(side).lower()}_{str(kind).lower()}"

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
                    aligned["mkt_rv"].astype(float) > aligned["mkt_rv_med"].astype(float)
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
                    aligned[feat_name] = (base_vals * (1.0 - gate_series)).astype(np.float32)

        return aligned.reindex(columns=feat_cols, fill_value=0.0).fillna(0.0)

    def _get_bucket_policy(self, side: str, kind: str) -> Dict[str, Any]:
        bucket_key = self._normalize_bucket_key(side, kind)
        bucket_cfg = {}
        if isinstance(self.ridge_params_per_bucket, dict):
            bucket_cfg = self.ridge_params_per_bucket.get(bucket_key, {}) or {}
        if not bucket_cfg and isinstance(self.bucket_params, dict):
            buckets = self.bucket_params.get("buckets", {}) if "buckets" in self.bucket_params else {}
            bucket_cfg = buckets.get(bucket_key.upper(), {}) or self.bucket_params.get(bucket_key, {}) or {}
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
    
    def _extract_feature_columns(self) -> Dict[str, List[str]]:
        """Extract feature column names from all loaded alpha models.

        Returns:
            Dictionary mapping ``"{side}_{kind}"`` to feature columns.
        """
        columns = {}

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
        key = f"{side}_{kind}"
        
        if side not in self.alpha_models or kind not in self.alpha_models[side]:
            tprint(f"Warning: Alpha model not found for {key}")
            return pd.Series(dtype=float)
        
        model_info = self.alpha_models[side][kind]
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
        X = aligned_features.reindex(columns=feat_cols, fill_value=0.0).fillna(0.0)
        
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

        side_models = self.alpha_models.get(side, {}) if isinstance(self.alpha_models, dict) else {}
        for kind in sorted(side_models.keys()):
            preds = self.predict_alpha(features, side, kind)
            # Defensive check: ensure preds is a DataFrame/Series
            if isinstance(preds, pd.DataFrame):
                if not preds.empty:
                    results[f"{side}_{kind}"] = preds
            elif isinstance(preds, pd.Series):
                if not preds.empty:
                    results[f"{side}_{kind}"] = preds
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
        key = f"{side}_{kind}"

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
            
            available_cols = [c for c in feat_cols if c in features.columns]
            
            if not available_cols:
                return pd.Series(dtype=float)
            
            X = features[available_cols].fillna(0)
            preds = meta_model.predict(X)
            
            return pd.Series(preds, index=features.index)
        except Exception as e:
            tprint(f"Error predicting meta for {key}: {e}")
            return pd.Series(dtype=float)
    
    # =========================================================================
    # STEP 5: Ridge Position Sizing
    # =========================================================================
    
    def compute_ridge_position_size(
        self,
        features: pd.DataFrame,
        side: str,
        kind: str = "mr",
    ) -> Tuple[pd.Series, Dict[str, float]]:
        """Step 5: Ridge position sizer with confidence.
        
        Args:
            features: Feature DataFrame (with meta predictions)
            side: "long" or "short"
            kind: "mr" or "tf"
            
        Returns:
            Tuple of (position_sizes Series, confidence dict)
        """
        bucket_key = self._normalize_bucket_key(side, kind)

        if self.ridge_sizer is not None:
            try:
                from extreme_price_movements.simple_position_sizer import clean_and_standardize

                # Fetch feature names configured in the artifact
                model_names = getattr(self.ridge_sizer, "feature_names", None)
                if model_names is None:
                    model_names = getattr(self.ridge_sizer, "model_names_", [])
                model_names = list(model_names)

                if model_names:
                    for col in model_names:
                        if col not in features.columns:
                            features[col] = 0.0

                    X = features[model_names].to_numpy(dtype=float)

                    # Robust standardizer parameters from train time
                    fit_medians = getattr(self.ridge_sizer, "fit_medians_", None)
                    scaler = getattr(self.ridge_sizer, "scaler_", None)
                    center_1d = getattr(self.ridge_sizer, "center_1d_", None)
                    scale_1d = getattr(self.ridge_sizer, "scale_1d_", None)

                    X_clean, _, _, _, _ = clean_and_standardize(
                        X,
                        fit_medians=fit_medians,
                        scaler=scaler,
                        center_1d=center_1d,
                        scale_1d=scale_1d
                    )

                    # Call predict on the underlying model (e.g. HuberRegressor or Ridge)
                    model = getattr(self.ridge_sizer, "model", self.ridge_sizer)
                    preds = np.asarray(model.predict(X_clean), dtype=float)
                    conf = np.clip(np.abs(preds), 0.0, 1.0)
                    return pd.Series(preds, index=features.index), {"confidence": float(np.nanmean(conf)) if len(conf) else 0.0}
            except Exception as e:
                tprint(f"Warning: ridge_sizer.predict failed for {bucket_key}: {e}")

        if not isinstance(self.ridge_weight_map, dict) or not self.ridge_weight_map:
            tprint(f"Warning: Ridge weights not found for {bucket_key}")
            return pd.Series(0.0, index=features.index), {"confidence": 0.0}

        prefix = f"{bucket_key}_"
        bucket_weights = {
            k[len(prefix):]: float(v)
            for k, v in self.ridge_weight_map.items()
            if isinstance(k, str) and k.startswith(prefix)
        }
        if not bucket_weights:
            tprint(f"Warning: No flattened ridge weights found for {bucket_key}")
            return pd.Series(0.0, index=features.index), {"confidence": 0.0}

        feature_names = list(bucket_weights.keys())
        X = features.reindex(columns=feature_names, fill_value=0.0).fillna(0.0).to_numpy(dtype=float)
        coefs_array = np.asarray([bucket_weights[c] for c in feature_names], dtype=float)
        positions = np.dot(X, coefs_array)
        conf = np.clip(np.abs(positions), 0.0, 1.0)
        return pd.Series(positions, index=features.index), {"confidence": float(np.nanmean(conf)) if len(conf) else 0.0}
    
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
                if col in ["u_hat_z", "mae_hat_z", "mfe_hat_z", "dur_hat_z",
                           "u_hat", "mae_hat", "mfe_hat"]:
                    feat_dict[col] = float(features[col].iloc[0]) if len(features) > 0 else 0.0
        
        # Use meta prediction as score
        score = float(meta_pred) if np.isfinite(meta_pred) else 0.0
        
        # Get entry policy config from bucket_params or runtime_cfg
        bucket_cfg = self.entry_policy_config or self._get_bucket_policy(side, kind) or self.bucket_params
        
        # Flatten if needed
        if bucket_cfg:
            bucket_cfg = flatten_bucket_policy(bucket_cfg)
        
        try:
            decision = compute_entry_policy_decision(
                entry_px=entry_price,
                atr_frac=atr_frac,
                score=score,
                bucket_cfg=bucket_cfg,
                features=feat_dict,
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
        
        # =====================================================================
        # STEP 2: Alpha (Base) Model Predictions
        # =====================================================================
        alpha_preds = self.predict_alpha_all_horizons(features, side)
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
            if isinstance(pred_series, (pd.DataFrame, pd.Series)) and not pred_series.empty:
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
            
            if isinstance(disagreement, (pd.DataFrame, pd.Series)) and not disagreement.empty:
                meta_base["disagreement"] = disagreement
                results["disagreement_features"] = disagreement.to_dict()
        
        # =====================================================================
        # STEP 4: Meta Model Prediction
        # =====================================================================
        if kind is None:
            side_models = self.alpha_models.get(side, {}) if isinstance(self.alpha_models, dict) else {}
            kind = "mr" if "mr" in side_models else (next(iter(sorted(side_models.keys())), "mr"))

        meta_pred = self.predict_meta(meta_base, side, kind)
        if not isinstance(meta_pred, (pd.DataFrame, pd.Series)) or meta_pred.empty:
            if not self.meta_models:
                fallback_key = f"{side}_{kind}"
                fallback_pred = alpha_preds.get(fallback_key)
                if isinstance(fallback_pred, (pd.DataFrame, pd.Series)) and not fallback_pred.empty:
                    meta_pred = fallback_pred if isinstance(fallback_pred, pd.Series) else fallback_pred.iloc[:, 0]
                    results["meta_pred_fallback"] = "alpha_only"
            if not isinstance(meta_pred, (pd.DataFrame, pd.Series)) or meta_pred.empty:
                results["action"] = "no_meta_prediction"
                return results
        
        meta_pred_val = float(meta_pred.iloc[0]) if len(meta_pred) > 0 else 0.0
        results["meta_pred"] = meta_pred_val
        
        # =====================================================================
        # STEP 5: Ridge Position Sizing
        # =====================================================================
        ridge_features = meta_base.copy()
        ridge_features["meta_pred"] = meta_pred
        
        position_size, confidence = self.compute_ridge_position_size(ridge_features, side, kind)
        
        position_val = float(position_size.iloc[0]) if len(position_size) > 0 else 0.0
        results["position_size"] = position_val
        results["ridge_confidence"] = confidence.get("confidence", 1.0)
        
        # Check minimum position size
        if position_val <= 0:
            results["action"] = "rejected_position_size"
            return results
        
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
        raise ValueError("Either (features and symbol) or just features must be provided")
