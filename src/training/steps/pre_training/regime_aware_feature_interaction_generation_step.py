"""
Regime-Aware Feature Interaction Generation Step.

This step derives regime-aware features from the calibrated regime
probabilities produced by the regime_ensemble_training component.

It:
- Loads the ensemble "tagged dataset" artifact produced by RegimeEnsembleTrainingComponent
- Extracts ensemble regime probabilities and labels
- Builds regime-aware summary features (probabilities, confidence, entropy,
  run-length within regime, transition indicators)
- Saves the resulting feature DataFrame via BaseStep artifact routing,
  which uses the HDF5-based versioned_artifacts system for tabular data.
"""

import logging
from typing import Any, Dict, Optional, List, Tuple

import numpy as np
import pandas as pd

from src.training.steps.base_step import BaseStep
from src.utils.logger import system_logger
from src.utils.tprint import (
    tprint,
    tprint_info,
    tprint_warning,
    tprint_error,
)
from src.utils.artifact_manager import ArtifactManager


logger = logging.getLogger(__name__)


class RegimeAwareFeatureInteractionGenerationStep(BaseStep):
    """Generate regime-aware features from ensemble regime probabilities.

    This step is intended to run *after* regime_ensemble_training.
    It does not retrain models; it only consumes the saved ensemble
    outputs and produces additional features for downstream models.
    """

    def __init__(self, step_name: str = "regime_aware_feature_interaction_generation_step") -> None:
        # Enable versioned artifacts so tabular outputs go to HDF5 store
        super().__init__(step_name, use_versioned_artifacts=True)
        self.logger = system_logger.getChild("RegimeAwareFeatureInteractionGeneration")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _load_tagged_dataset(
        self,
        symbol: str,
        exchange: str,
        regime_timeframe: str,
    ) -> Optional[pd.DataFrame]:
        """Load the tagged dataset produced by regime_ensemble_training.

        This uses the centralized ArtifactManager (not BaseStep router)
        because RegimeEnsembleTrainingComponent saves its artifacts via
        its own artifact manager under the component name
        "regime_ensemble_training".
        """
        tprint_info(
            f"📂 [REGIME_AWARE] Loading tagged_dataset from regime_ensemble_training "
            f"for {symbol}/{exchange} [{regime_timeframe}]",
        )

        artifact_manager = ArtifactManager(config={})
        artifact_manager.set_context(
            step_name="regime_ensemble_training",
            symbol=symbol,
            exchange=exchange,
            timeframe=regime_timeframe,
            direction="long",
            model="Analyst",
        )

        # The artifact name is "tagged_dataset" at component level
        tagged_artifact = artifact_manager.get_artifact(
            artifact_name="tagged_dataset",
            artifact_type="data",
            return_path=False,
        )

        if tagged_artifact is None:
            tprint_warning(
                "⚠️ [REGIME_AWARE] No tagged_dataset artifact found for "
                f"regime_ensemble_training ({symbol}/{exchange} [{regime_timeframe}])",
            )
            return None

        # The artifact structure can be either:
        # - DataFrame directly
        # - dict with "data" key holding the DataFrame
        # - dict with nested "tagged_dataset" → {"tagged_dataset": DataFrame, ...}
        if isinstance(tagged_artifact, pd.DataFrame):
            tprint_info(
                f"✅ [REGIME_AWARE] Loaded tagged_dataset DataFrame: {tagged_artifact.shape}",
            )
            return tagged_artifact

        if isinstance(tagged_artifact, dict):
            # Case 1: main artifacts dict with "data" key
            data_obj = tagged_artifact.get("data")
            if isinstance(data_obj, pd.DataFrame):
                tprint_info(
                    f"✅ [REGIME_AWARE] Loaded tagged_dataset['data'] DataFrame: {data_obj.shape}",
                )
                return data_obj

            # Case 2: individual tagged artifact with nested structure
            nested = tagged_artifact.get("tagged_dataset")
            if isinstance(nested, pd.DataFrame):
                tprint_info(
                    f"✅ [REGIME_AWARE] Loaded nested tagged_dataset DataFrame: {nested.shape}",
                )
                return nested
            if isinstance(nested, dict) and isinstance(nested.get("tagged_dataset"), pd.DataFrame):
                df = nested["tagged_dataset"]
                tprint_info(
                    f"✅ [REGIME_AWARE] Loaded nested['tagged_dataset']['tagged_dataset'] DataFrame: {df.shape}",
                )
                return df

        tprint_warning(
            "⚠️ [REGIME_AWARE] tagged_dataset artifact found but no DataFrame payload "
            "could be extracted",
        )
        return None

    def _build_regime_features(self, tagged_df: pd.DataFrame) -> pd.DataFrame:
        """Construct regime-aware summary features from ensemble outputs.

        Expected columns in tagged_df:
        - ensemble_regime_label (optional hard prediction)
        - ensemble_regime_*_probability (per-regime calibrated probabilities)
        """
        prob_cols = [
            c
            for c in tagged_df.columns
            if c.startswith("ensemble_regime_") and c.endswith("_probability")
        ]

        if not prob_cols:
            tprint_warning(
                "⚠️ [REGIME_AWARE] No ensemble_regime_*_probability columns found in tagged_dataset",
            )
            return pd.DataFrame(index=tagged_df.index.copy())

        probs = tagged_df[prob_cols].astype(float).to_numpy()
        n_samples, n_regimes = probs.shape

        # Numerical safety
        eps = 1e-12
        probs_safe = np.clip(probs, eps, 1.0 - eps)

        # Hard labels: prefer provided label, fallback to argmax of probs
        if "ensemble_regime_label" in tagged_df.columns:
            regime_ids = tagged_df["ensemble_regime_label"].astype(int).to_numpy()
        else:
            regime_ids = np.argmax(probs_safe, axis=1).astype(int)

        max_prob = probs_safe.max(axis=1)
        entropy = -np.sum(probs_safe * np.log(probs_safe), axis=1)

        regime_features = pd.DataFrame(index=tagged_df.index.copy())

        # Keep calibrated probabilities as-is
        for j, col in enumerate(prob_cols):
            regime_features[col] = probs_safe[:, j]

        # Summary features
        regime_features["ensemble_regime_id"] = regime_ids
        regime_features["ensemble_regime_max_prob"] = max_prob
        regime_features["ensemble_regime_entropy"] = entropy

        tprint_info(
            "✅ [REGIME_AWARE] Built regime-aware feature matrix: "
            f"{regime_features.shape[0]} rows × {regime_features.shape[1]} cols",
        )

        return regime_features

    def _load_generated_features_for_regime_aware(self, config: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """Load base generated features to be combined with regimes.

        Uses BaseStep's artifact routing to retrieve the latest
        `generated_features` artifact in the current context.
        """
        try:
            generated_features = self._get_artifact(
                artifact_name="generated_features",
                artifact_type="data",
                data_category="features",
            )

            if isinstance(generated_features, pd.DataFrame) and not generated_features.empty:
                tprint_info(
                    f"✅ [REGIME_AWARE] Loaded generated_features for regime-aware conditioning: "
                    f"{generated_features.shape}"
                )
                return generated_features

            tprint_warning(
                "⚠️ [REGIME_AWARE] generated_features artifact is missing or empty; "
                "regime-aware domain features will be skipped",
            )
            return None

        except Exception as e:
            tprint_warning(
                f"⚠️ [REGIME_AWARE] Failed to load generated_features artifact: {e}"
            )
            return None

    def _load_lookback_optimization_for_regime_aware(self) -> Optional[pd.DataFrame]:
        """Load lookback_optimization artifact for feature scoring.

        The underlying artifact is stored as a dict and materialized as a
        wide DataFrame with nested column names. We transform it to a
        long format with (feature_name, category, composite_score,
        optimal_lookback).
        """
        try:
            lookback_raw = self._get_artifact(
                artifact_name="lookback_optimization",
                artifact_type="data",
            )

            if not isinstance(lookback_raw, pd.DataFrame) or lookback_raw.empty:
                tprint_warning(
                    "⚠️ [REGIME_AWARE] lookback_optimization artifact is missing or empty; "
                    "cannot score base features for regime-aware conditioning",
                )
                return None

            tprint_info(
                f"✅ [REGIME_AWARE] Loaded lookback_optimization raw DataFrame: "
                f"{lookback_raw.shape}"
            )
            return lookback_raw

        except Exception as e:
            tprint_warning(
                f"⚠️ [REGIME_AWARE] Failed to load lookback_optimization artifact: {e}"
            )
            return None

    def _transform_lookback_optimization_data_simple(
        self, lookback_optimization: pd.DataFrame
    ) -> pd.DataFrame:
        """Transform wide lookback optimization data to long format.

        Extracts per-feature rows with columns:
        - feature_name
        - category (e.g. momentum, trend, volatility, oscillator)
        - composite_score
        - optimal_lookback
        """
        feature_data: List[Dict[str, Any]] = []

        feature_columns: List[str] = []
        for col in lookback_optimization.columns:
            if "category_optimizations" in col and col.endswith(".feature_name"):
                feature_columns.append(col)

        if not feature_columns:
            tprint_warning(
                "⚠️ [REGIME_AWARE] No feature_name columns found in lookback_optimization; "
                "skipping base feature scoring",
            )
            return pd.DataFrame()

        tprint_info(
            f"🔄 [REGIME_AWARE] Transforming lookback optimization data (" 
            f"{len(feature_columns)} feature columns)"
        )

        for col in feature_columns:
            try:
                parts = col.split(".")
                if len(parts) < 4:
                    continue

                # category_optimizations.<category>_features.<idx>.feature_name
                category_part = parts[2]
                category = category_part.replace("_features", "")

                value = lookback_optimization[col].iloc[0]
                feature_name = value if not pd.isna(value) else None
                if not feature_name:
                    continue

                base_col = col.replace(".feature_name", "")
                feature_info: Dict[str, Any] = {
                    "feature_name": feature_name,
                    "category": category,
                    "composite_score": 0.0,
                    "optimal_lookback": None,
                    "performance_score": 0.0,
                    "stability_score": 0.0,
                    "information_score": 0.0,
                }

                for metric in [
                    "composite_score",
                    "optimal_lookback",
                    "performance_score",
                    "stability_score",
                    "information_score",
                ]:
                    metric_col = f"{base_col}.{metric}"
                    if metric_col in lookback_optimization.columns:
                        metric_val = lookback_optimization[metric_col].iloc[0]
                        if not pd.isna(metric_val):
                            feature_info[metric] = metric_val

                feature_data.append(feature_info)

            except Exception as e:
                tprint_warning(
                    f"⚠️ [REGIME_AWARE] Error processing lookback column {col}: {e}"
                )
                continue

        if not feature_data:
            tprint_warning(
                "⚠️ [REGIME_AWARE] No per-feature records extracted from lookback_optimization"
            )
            return pd.DataFrame()

        df = pd.DataFrame(feature_data)
        if "feature_name" in df.columns:
            df = df.drop_duplicates(subset=["feature_name"], keep="first")

        tprint_info(
            f"✅ [REGIME_AWARE] Transformed lookback_optimization to long format: {df.shape}"
        )
        return df

    def _select_regime_aware_base_features(
        self,
        lookback_long: pd.DataFrame,
        generated_features: pd.DataFrame,
    ) -> Tuple[List[str], int, int]:
        """Select base features to convert into regime-aware variants.

        Implements a global top-40 selection with:
        - High MI proxy (information_score / composite_score)
        - High stability (stability_score threshold)
        - Low redundancy (pairwise correlation filter)

        Returns:
            (selected_feature_names, n_candidates_before_filter, n_selected_after_filter)
        """
        if lookback_long.empty or "feature_name" not in lookback_long.columns or "category" not in lookback_long.columns:
            tprint_warning(
                "⚠️ [REGIME_AWARE] lookback_optimization long format missing required columns; "
                "skipping base feature selection",
            )
            return [], 0, 0

        # Focus on categories that benefit most from regime awareness
        target_categories = {"momentum", "trend", "volatility", "oscillation", "oscillator"}

        lookback_long = lookback_long.copy()
        lookback_long["category"] = lookback_long["category"].astype(str)

        mask = lookback_long["category"].str.lower().isin(target_categories)
        df_interest = lookback_long[mask].copy()

        if df_interest.empty:
            tprint_warning(
                "⚠️ [REGIME_AWARE] No features found in target categories "
                "(momentum/trend/volatility/oscillator); skipping base feature selection",
            )
            return [], 0, 0

        # Apply quality filters: require stability and information scores when available
        if {"stability_score", "information_score"}.issubset(df_interest.columns):
            stability_threshold = 0.4
            info_threshold = 0.0
            before_quality = len(df_interest)
            df_interest = df_interest[
                (df_interest["stability_score"] >= stability_threshold)
                & (df_interest["information_score"] >= info_threshold)
            ]
            tprint_info(
                f"📊 [REGIME_AWARE] Quality filter: {before_quality} → {len(df_interest)} "
                f"rows after stability/information thresholds"
            )

        # Count unique candidates before redundancy filtering
        candidate_features = sorted(df_interest["feature_name"].astype(str).unique())
        n_candidates = len(candidate_features)

        tprint_info(
            f"📊 [REGIME_AWARE] Candidate base features for regime-aware variants "
            f"(before redundancy filter): {n_candidates}"
        )

        if n_candidates == 0:
            return [], 0, 0

        # Global ranking by composite_score (MI proxy × stability)
        if "composite_score" in df_interest.columns:
            df_ranked = df_interest.sort_values("composite_score", ascending=False)
        else:
            df_ranked = df_interest.copy()

        # Greedy redundancy reduction using correlation threshold
        max_features = 40
        corr_threshold = 0.9
        selected_names: List[str] = []

        # Ensure we work only with existing columns in generated_features
        available_cols = set(str(c) for c in generated_features.columns)

        for feature_name in df_ranked["feature_name"].astype(str).tolist():
            if len(selected_names) >= max_features:
                break

            if feature_name not in available_cols:
                continue

            series = generated_features[feature_name]

            # Skip constant or all-NaN features
            try:
                if hasattr(series, "nunique") and series.nunique(dropna=True) <= 1:
                    continue
            except Exception:
                continue

            if not selected_names:
                selected_names.append(feature_name)
                continue

            # Correlation with already selected features
            try:
                selected_df = generated_features[selected_names]
                corr = selected_df.corrwith(series.astype(float))
                max_abs_corr = float(corr.abs().max()) if not corr.isna().all() else 0.0
            except Exception:
                # If correlation computation fails, be conservative and skip
                continue

            if max_abs_corr < corr_threshold:
                selected_names.append(feature_name)

        n_selected = len(selected_names)

        tprint_info(
            f"📊 [REGIME_AWARE] Selected base features for regime-aware variants "
            f"(after redundancy filter & existence check): {n_selected}"
        )

        if n_selected == 0:
            tprint_warning(
                "⚠️ [REGIME_AWARE] No selected base features found in generated_features.columns; "
                "regime-aware domain features will be skipped",
            )

        return selected_names, n_candidates, n_selected

    def _build_regime_conditioned_features(
        self,
        base_features: pd.DataFrame,
        tagged_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Build regime-probability-weighted variants of base features.

        For each base feature f and each regime probability column p_k,
        creates a feature of the form f_x_ensemble_regime_k_probability.
        """
        if base_features is None or base_features.empty:
            return pd.DataFrame(index=tagged_df.index.copy())

        prob_cols = [
            c
            for c in tagged_df.columns
            if c.startswith("ensemble_regime_") and c.endswith("_probability")
        ]

        if not prob_cols:
            tprint_warning(
                "⚠️ [REGIME_AWARE] No regime probability columns available for "
                "conditioning base features",
            )
            return pd.DataFrame(index=tagged_df.index.copy())

        # Align base features to the tagged_df index
        aligned_base = base_features.reindex(tagged_df.index)

        probs = tagged_df[prob_cols].astype(float).to_numpy()
        n_samples, n_regimes = probs.shape

        eps = 1e-12
        probs_safe = np.clip(probs, eps, 1.0 - eps)

        tprint_info(
            f"📊 [REGIME_AWARE] Building regime-conditioned features for "
            f"{aligned_base.shape[1]} base features × {n_regimes} regimes"
        )

        conditioned_data: Dict[str, np.ndarray] = {}

        for feature_name in aligned_base.columns:
            values = aligned_base[feature_name].astype(float).to_numpy()
            if values.shape[0] != n_samples:
                # After reindex this should not happen, but guard anyway
                tprint_warning(
                    f"⚠️ [REGIME_AWARE] Base feature '{feature_name}' length mismatch; skipping"
                )
                continue

            for j, prob_col in enumerate(prob_cols):
                col_name = f"{feature_name}_x_{prob_col}"
                conditioned_data[col_name] = values * probs_safe[:, j]

        if not conditioned_data:
            tprint_warning(
                "⚠️ [REGIME_AWARE] No regime-conditioned features were created"
            )
            return pd.DataFrame(index=tagged_df.index.copy())

        conditioned_df = pd.DataFrame(conditioned_data, index=tagged_df.index.copy())
        tprint_info(
            f"✅ [REGIME_AWARE] Built regime-conditioned feature matrix: "
            f"{conditioned_df.shape[0]} rows × {conditioned_df.shape[1]} cols",
        )

        return conditioned_df

    # ------------------------------------------------------------------
    # BaseStep entry point
    # ------------------------------------------------------------------
    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute regime-aware feature generation.

        Config keys:
        - symbol: trading symbol (e.g., "ETHUSDT")
        - exchange: exchange name (e.g., "binance")
        - timeframe: trading timeframe for downstream features (e.g., "15m")
        - regime_timeframe: timeframe used for regime_ensemble_training (e.g., "1h")
        - direction: "long" or "short" (for context separation)
        - execution_mode: "blank", "light", or "full" (for logging only here)
        """
        import time

        start_time = time.time()

        symbol = config.get("symbol", "ETHUSDT")
        exchange = config.get("exchange", "binance")
        timeframe = config.get("timeframe", "15m")
        direction = config.get("direction", "long")
        regime_timeframe = config.get("regime_timeframe", timeframe)
        execution_mode = config.get("execution_mode", "blank")

        # Set BaseStep context so that outputs are stored under the
        # correct symbol/exchange/timeframe/direction/model in the
        # versioned_artifacts HDF5 store.
        self.set_context(
            symbol=symbol,
            exchange=exchange,
            timeframe=timeframe,
            direction=direction,
            model=config.get("model", "analyst"),
            use_versioned_artifacts=True,
        )

        tprint_info(
            f"🎯 [REGIME_AWARE] Starting regime-aware feature generation for "
            f"{symbol}/{exchange} timeframe={timeframe}, regime_timeframe={regime_timeframe}, "
            f"execution_mode={execution_mode}",
        )

        try:
            # 1) Load ensemble-tagged dataset from regime_ensemble_training
            tagged_df = self._load_tagged_dataset(symbol, exchange, regime_timeframe)
            if tagged_df is None or tagged_df.empty:
                error_msg = (
                    "No ensemble tagged_dataset available. Run regime_ensemble_training "
                    "before regime_aware_feature_interaction_generation_step."
                )
                tprint_error(f"❌ [REGIME_AWARE] {error_msg}")
                return {
                    "success": False,
                    "artifacts": {},
                    "metrics": {},
                    "error": error_msg,
                }

            # 2) Build regime-only features from ensemble outputs
            regime_features = self._build_regime_features(tagged_df)
            if regime_features.empty:
                error_msg = "Regime-aware feature matrix is empty (no probability columns found)."
                tprint_error(f"❌ [REGIME_AWARE] {error_msg}")
                return {
                    "success": False,
                    "artifacts": {},
                    "metrics": {},
                    "error": error_msg,
                }

            # 3) Load base features and lookback optimization to select
            #    which domain features to convert into regime-aware variants.
            generated_features = self._load_generated_features_for_regime_aware(config)
            lookback_raw = self._load_lookback_optimization_for_regime_aware()

            base_candidate_count = 0
            base_selected_count = 0
            regime_conditioned_features: Optional[pd.DataFrame] = None

            if generated_features is not None and lookback_raw is not None:
                lookback_long = self._transform_lookback_optimization_data_simple(lookback_raw)

                if not lookback_long.empty:
                    (
                        selected_base_features,
                        base_candidate_count,
                        base_selected_count,
                    ) = self._select_regime_aware_base_features(
                        lookback_long,
                        generated_features,
                    )

                    if selected_base_features:
                        base_subset = generated_features[selected_base_features]
                        regime_conditioned_features = self._build_regime_conditioned_features(
                            base_subset,
                            tagged_df,
                        )
                else:
                    tprint_warning(
                        "⚠️ [REGIME_AWARE] lookback_optimization long format is empty; "
                        "skipping regime-conditioned domain features",
                    )
            else:
                tprint_warning(
                    "⚠️ [REGIME_AWARE] Base features or lookback optimization not available; "
                    "regime-aware domain features will be limited to regime summaries",
                )

            # 4) Combine regime-only and regime-conditioned features
            if regime_conditioned_features is not None and not regime_conditioned_features.empty:
                combined_features = pd.concat([regime_features, regime_conditioned_features], axis=1)
            else:
                combined_features = regime_features

            # 5) Save regime-aware features via BaseStep artifact routing
            artifact_name = "regime_aware_features"
            artifact_path = self._save_artifact(
                data=combined_features,
                artifact_name=artifact_name,
                artifact_type="data",
                data_category="features",
                metadata={
                    "source_component": "regime_ensemble_training",
                    "description": "Regime-aware features derived from ensemble regime probabilities",
                    "symbol": symbol,
                    "exchange": exchange,
                    "timeframe": timeframe,
                    "regime_timeframe": regime_timeframe,
                },
            )

            elapsed = time.time() - start_time

            # Basic metrics for downstream diagnostics
            metrics: Dict[str, Any] = {
                "n_samples": int(combined_features.shape[0]),
                "n_regime_features_only": int(regime_features.shape[1]),
                "n_regime_conditioned_features": int(
                    regime_conditioned_features.shape[1]
                    if regime_conditioned_features is not None and not regime_conditioned_features.empty
                    else 0
                ),
                "n_regimes": int(len(np.unique(regime_features["ensemble_regime_id"].values))),
                "n_base_candidate_features": int(base_candidate_count),
                "n_base_selected_features": int(base_selected_count),
                "execution_time_seconds": float(elapsed),
            }

            tprint_info(
                f"✅ [REGIME_AWARE] Saved regime-aware features to {artifact_path} "
                f"in {elapsed:.2f}s",
            )

            return {
                "success": True,
                "artifacts": {artifact_name: artifact_path},
                "metrics": metrics,
            }

        except Exception as exc:
            error_msg = f"Regime-aware feature generation failed: {exc}"
            tprint_error(f"❌ [REGIME_AWARE] {error_msg}")
            self.logger.error(error_msg, exc_info=True)
            return {
                "success": False,
                "artifacts": {},
                "metrics": {},
                "error": error_msg,
            }
