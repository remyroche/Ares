"""Step 6: Feature Engineering - Refactored to use BaseStep.

This module implements comprehensive feature engineering including technical indicators,
interaction terms, and regime-aware features.
"""

from typing import Any, Dict, Tuple, Optional, List
from pathlib import Path
import pandas as pd
import numpy as np
import json
from datetime import datetime

from src.training.base_step import BaseStep
from src.utils.logger import system_logger
from src.core.decorators import handles_errors
from src.training.steps.feature_engineering.feature_components import (
from copy import copy
import asyncio

    TechnicalIndicatorEngine,
    FeatureInteractionEngine,
    RegimeAwareFeatureEngine
)


class FeatureEngineeringStep(BaseStep):
    """Step 6: Feature Engineering using standardized base class."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize feature engineering step.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config, "06", "feature_engineering")
        
        # Step-specific configuration
        self.feature_config = config.get("feature_engineering_config", {
            "use_technical_indicators": True,
            "use_interaction_features": True,
            "use_regime_features": True,
            "use_dynamic_lookback": True,
            "lookback_periods": {
                "short": [5, 10, 20],
                "medium": [50, 100],
                "long": [200]
            },
            "feature_selection": {
                "enabled": True,
                "max_features": 100,
                "importance_threshold": 0.01
            }
        })
        
        # Components
        self.technical_engine = None
        self.interaction_engine = None
        self.regime_engine = None
        
    def _initialize_step(self) -> None:
        """Initialize step-specific components."""
        try:
            # Initialize technical indicator engine
            self.technical_engine = TechnicalIndicatorEngine(
                self.feature_config.get("lookback_periods", {})
            )
            
            # Initialize interaction engine
            if self.feature_config.get("use_interaction_features", True):
                self.interaction_engine = FeatureInteractionEngine(
                    self.feature_config
                )
            
            # Initialize regime-aware feature engine
            if self.feature_config.get("use_regime_features", True):
                self.regime_engine = RegimeAwareFeatureEngine()
            
            self.logger.info("✅ Feature engineering components initialized")
            
        except ImportError as e:
            self.logger.warning(f"⚠️ Some feature components not available: {e}")
            # Will use basic feature engineering
    
    def validate_inputs(
        self, 
        training_input: Dict[str, Any], 
        pipeline_state: Dict[str, Any]
    ) -> Tuple[bool, list]:
        """Validate step inputs.
        
        Args:
            training_input: Training input parameters
            pipeline_state: Current pipeline state
            
        Returns:
            Tuple of (is_valid, errors)
        """
        errors = []
        
        # Check for labeled data
        if "labeled_data" not in pipeline_state:
            # Check for split data with labels
            if not all(f"{split}_data" in pipeline_state for split in ["train", "val", "test"]):
                errors.append("No labeled data from step 5")
        
        # Check for regime information if regime features are enabled
        if self.feature_config.get("use_regime_features", True):
            if "regime_labels" not in pipeline_state:
                self.logger.warning("Regime labels not available, will skip regime features")
        
        return len(errors) == 0, errors
    
    @handles_errors(
        exceptions=(Exception,),
        default_return={"success": False},
        context="feature engineering execution"
    )
    async def execute_logic(
        self,
        training_input: Dict[str, Any],
        pipeline_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute feature engineering logic.
        
        Args:
            training_input: Training input parameters
            pipeline_state: Current pipeline state
            
        Returns:
            Updated pipeline state
        """
        self.logger.info("🔧 Starting feature engineering...")
        
        # Get data to process
        data_dict = self._get_data_to_process(pipeline_state)
        
        # Process each data split
        engineered_data = {}
        feature_statistics = {}
        
        for split_name, data in data_dict.items():
            self.logger.info(f"📊 Engineering features for {split_name} split...")
            
            # Apply feature engineering
            engineered_split = await self._engineer_features_for_split(
                data, 
                pipeline_state
            )
            
            # Calculate feature statistics
            stats = self._calculate_feature_statistics(engineered_split)
            
            engineered_data[split_name] = engineered_split
            feature_statistics[split_name] = stats
        
        # Perform feature selection if enabled
        if self.feature_config.get("feature_selection", {}).get("enabled", True):
            self.logger.info("🎯 Performing feature selection...")
            engineered_data, selected_features = await self._perform_feature_selection(
                engineered_data,
                feature_statistics
            )
        else:
            selected_features = self._get_all_feature_columns(engineered_data)
        
        # Generate reports
        reports = self._generate_feature_reports(
            engineered_data,
            feature_statistics,
            selected_features
        )
        
        # Update pipeline state
        pipeline_state.update({
            "engineered_data": engineered_data,
            "feature_statistics": feature_statistics,
            "selected_features": selected_features,
            "feature_reports": reports,
            "feature_config": self.feature_config
        })
        
        # Update individual splits if they exist
        for split in ["train", "val", "test"]:
            if split in engineered_data and f"{split}_data" in pipeline_state:
                pipeline_state[f"{split}_data"] = engineered_data[split]
        
        # Save outputs
        await self._save_outputs(training_input, pipeline_state)
        
        return pipeline_state
    
    def validate_outputs(self, pipeline_state: Dict[str, Any]) -> Tuple[bool, list]:
        """Validate step outputs.
        
        Args:
            pipeline_state: Updated pipeline state
            
        Returns:
            Tuple of (is_valid, errors)
        """
        errors = []
        
        # Check if engineered data exists
        if "engineered_data" not in pipeline_state:
            errors.append("No engineered data in pipeline state")
            return False, errors
        
        engineered_data = pipeline_state["engineered_data"]
        
        # Check if features were added
        total_features = 0
        for split_name, data in engineered_data.items():
            if isinstance(data, pd.DataFrame):
                feature_cols = [col for col in data.columns if col.startswith("feature_")]
                total_features += len(feature_cols)
        
        if total_features == 0:
            errors.append("No features were engineered")
        
        # Check selected features
        if "selected_features" in pipeline_state:
            selected = pipeline_state["selected_features"]
            if len(selected) == 0:
                errors.append("No features were selected")
            
            # Check if selected features exist in data
            for split_data in engineered_data.values():
                if isinstance(split_data, pd.DataFrame):
                    missing_features = set(selected) - set(split_data.columns)
                    if missing_features:
                        errors.append(f"Selected features missing from data: {missing_features}")
                    break
        
        return len(errors) == 0, errors
    
    def _get_data_to_process(self, pipeline_state: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
        """Get data splits to process.
        
        Args:
            pipeline_state: Current pipeline state
            
        Returns:
            Dictionary of data splits
        """
        data_dict = {}
        
        # Check for individual splits first
        for split in ["train", "val", "test"]:
            if f"{split}_data" in pipeline_state:
                data_dict[split] = pipeline_state[f"{split}_data"].copy()
        
        # If no splits, use labeled data
        if not data_dict and "labeled_data" in pipeline_state:
            data_dict["all"] = pipeline_state["labeled_data"].copy()
        
        return data_dict
    
    async def _engineer_features_for_split(
        self,
        data: pd.DataFrame,
        pipeline_state: Dict[str, Any]
    ) -> pd.DataFrame:
        """Engineer features for a single data split.
        
        Args:
            data: Data to process
            pipeline_state: Pipeline state for context
            
        Returns:
            Data with engineered features
        """
        # Start with original data
        engineered = data.copy()
        
        # Apply technical indicators
        if self.feature_config.get("use_technical_indicators", True):
            if self.technical_engine:
                engineered = self.technical_engine.apply_indicators(engineered)
            else:
                engineered = self._apply_basic_indicators(engineered)
        
        # Apply interaction features
        if self.feature_config.get("use_interaction_features", True):
            if self.interaction_engine:
                engineered = await self.interaction_engine.create_interactions(engineered)
            else:
                engineered = self._create_basic_interactions(engineered)
        
        # Apply regime-aware features
        if self.feature_config.get("use_regime_features", True) and "regime_labels" in data.columns:
            if self.regime_engine:
                engineered = self.regime_engine.create_regime_features(
                    engineered,
                    pipeline_state.get("regime_characteristics", {})
                )
            else:
                engineered = self._create_basic_regime_features(engineered)
        
        # Add time-based features
        engineered = self._add_time_features(engineered)
        
        return engineered
    
    def _apply_basic_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply basic technical indicators.
        
        Args:
            data: Market data
            
        Returns:
            Data with indicators
        """
        # Simple moving averages
        for period in [10, 20, 50]:
            data[f"feature_sma_{period}"] = data["close"].rolling(period).mean()
            data[f"feature_sma_{period}_ratio"] = data["close"] / data[f"feature_sma_{period}"]
        
        # RSI
        delta = data["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        data["feature_rsi"] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        sma = data["close"].rolling(20).mean()
        std = data["close"].rolling(20).std()
        data["feature_bb_upper"] = sma + 2 * std
        data["feature_bb_lower"] = sma - 2 * std
        data["feature_bb_position"] = (data["close"] - data["feature_bb_lower"]) / (
            data["feature_bb_upper"] - data["feature_bb_lower"]
        )
        
        # Volume features
        if "volume" in data.columns:
            data["feature_volume_sma"] = data["volume"].rolling(20).mean()
            data["feature_volume_ratio"] = data["volume"] / data["feature_volume_sma"]
        
        return data
    
    def _create_basic_interactions(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create basic feature interactions.
        
        Args:
            data: Data with features
            
        Returns:
            Data with interaction features
        """
        feature_cols = [col for col in data.columns if col.startswith("feature_")]
        
        # Price-volume interactions
        if "feature_volume_ratio" in data.columns and "feature_returns" in data.columns:
            data["feature_price_volume_interaction"] = (
                data["feature_returns"] * data["feature_volume_ratio"]
            )
        
        # RSI-BB interactions
        if "feature_rsi" in data.columns and "feature_bb_position" in data.columns:
            data["feature_rsi_bb_interaction"] = (
                data["feature_rsi"] * data["feature_bb_position"]
            )
        
        # Momentum interactions
        if "feature_sma_10_ratio" in data.columns and "feature_sma_50_ratio" in data.columns:
            data["feature_momentum_interaction"] = (
                data["feature_sma_10_ratio"] - data["feature_sma_50_ratio"]
            )
        
        return data
    
    def _create_basic_regime_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create basic regime-aware features.
        
        Args:
            data: Data with regime labels
            
        Returns:
            Data with regime features
        """
        if "regime_label" in data.columns:
            # One-hot encode regimes
            regime_dummies = pd.get_dummies(
                data["regime_label"], 
                prefix="feature_regime"
            )
            data = pd.concat([data, regime_dummies], axis=1)
            
            # Regime transition features
            data["feature_regime_changed"] = (
                data["regime_label"] != data["regime_label"].shift(1)
            ).astype(int)
            
            # Time in regime
            data["feature_time_in_regime"] = data.groupby(
                (data["regime_label"] != data["regime_label"].shift()).cumsum()
            ).cumcount()
        
        return data
    
    def _add_time_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features.
        
        Args:
            data: Data with datetime index
            
        Returns:
            Data with time features
        """
        if hasattr(data.index, 'hour'):
            # Intraday features
            data["feature_hour"] = data.index.hour
            data["feature_minute"] = data.index.minute
            data["feature_hour_sin"] = np.sin(2 * np.pi * data.index.hour / 24)
            data["feature_hour_cos"] = np.cos(2 * np.pi * data.index.hour / 24)
        
        if hasattr(data.index, 'dayofweek'):
            # Day of week features
            data["feature_dayofweek"] = data.index.dayofweek
            data["feature_is_monday"] = (data.index.dayofweek == 0).astype(int)
            data["feature_is_friday"] = (data.index.dayofweek == 4).astype(int)
        
        return data
    
    def _calculate_feature_statistics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate statistics for engineered features.
        
        Args:
            data: Data with features
            
        Returns:
            Feature statistics
        """
        feature_cols = [col for col in data.columns if col.startswith("feature_")]
        
        stats = {
            "n_features": len(feature_cols),
            "feature_names": feature_cols,
            "missing_values": {},
            "zero_variance": [],
            "high_correlation_pairs": []
        }
        
        # Check missing values
        for col in feature_cols:
            missing_pct = data[col].isna().sum() / len(data) * 100
            if missing_pct > 0:
                stats["missing_values"][col] = missing_pct
        
        # Check zero variance features
        for col in feature_cols:
            if data[col].std() < 1e-10:
                stats["zero_variance"].append(col)
        
        # Check high correlations (sample for efficiency)
        if len(feature_cols) > 1:
            sample_size = min(1000, len(data))
            sample_data = data[feature_cols].sample(n=sample_size)
            corr_matrix = sample_data.corr()
            
            for i in range(len(feature_cols)):
                for j in range(i + 1, len(feature_cols)):
                    if abs(corr_matrix.iloc[i, j]) > 0.95:
                        stats["high_correlation_pairs"].append(
                            (feature_cols[i], feature_cols[j], corr_matrix.iloc[i, j])
                        )
        
        return stats
    
    async def _perform_feature_selection(
        self,
        engineered_data: Dict[str, pd.DataFrame],
        feature_statistics: Dict[str, Dict[str, Any]]
    ) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
        """Perform feature selection.
        
        Args:
            engineered_data: Dictionary of engineered data
            feature_statistics: Feature statistics
            
        Returns:
            Tuple of (filtered data, selected features)
        """
        # Use training data for feature selection
        train_data = engineered_data.get("train", next(iter(engineered_data.values())))
        
        # Get all feature columns
        all_features = [col for col in train_data.columns if col.startswith("feature_")]
        
        # Remove zero variance features
        zero_var_features = set()
        for stats in feature_statistics.values():
            zero_var_features.update(stats.get("zero_variance", []))
        
        valid_features = [f for f in all_features if f not in zero_var_features]
        
        # Remove highly correlated features
        to_remove = set()
        for stats in feature_statistics.values():
            for feat1, feat2, corr in stats.get("high_correlation_pairs", []):
                # Remove the second feature in the pair
                to_remove.add(feat2)
        
        valid_features = [f for f in valid_features if f not in to_remove]
        
        # Apply max features limit
        max_features = self.feature_config.get("feature_selection", {}).get("max_features", 100)
        if len(valid_features) > max_features:
            # Simple selection: take first max_features
            # In practice, you might want to use importance scores
            valid_features = valid_features[:max_features]
        
        self.logger.info(
            f"✅ Selected {len(valid_features)} features from {len(all_features)} total"
        )
        
        # Filter all data splits to selected features
        selected_data = {}
        base_columns = [col for col in train_data.columns if not col.startswith("feature_")]
        selected_columns = base_columns + valid_features
        
        for split_name, data in engineered_data.items():
            selected_data[split_name] = data[selected_columns]
        
        return selected_data, valid_features
    
    def _get_all_feature_columns(self, engineered_data: Dict[str, pd.DataFrame]) -> List[str]:
        """Get all feature columns from engineered data.
        
        Args:
            engineered_data: Dictionary of engineered data
            
        Returns:
            List of feature column names
        """
        all_features = set()
        for data in engineered_data.values():
            if isinstance(data, pd.DataFrame):
                features = [col for col in data.columns if col.startswith("feature_")]
                all_features.update(features)
        
        return sorted(list(all_features))
    
    def _generate_feature_reports(
        self,
        engineered_data: Dict[str, pd.DataFrame],
        feature_statistics: Dict[str, Dict[str, Any]],
        selected_features: List[str]
    ) -> Dict[str, str]:
        """Generate feature engineering reports.
        
        Args:
            engineered_data: Engineered data
            feature_statistics: Feature statistics
            selected_features: Selected feature names
            
        Returns:
            Dictionary of reports
        """
        reports = {}
        
        # Summary report
        summary_lines = [
            "Feature Engineering Summary",
            "=" * 40,
            f"Total features created: {len(selected_features)}",
            "",
            "Data splits:"
        ]
        
        for split_name, data in engineered_data.items():
            if isinstance(data, pd.DataFrame):
                summary_lines.append(
                    f"  {split_name}: {data.shape[0]} rows × {data.shape[1]} columns"
                )
        
        # Feature types breakdown
        feature_types = {
            "sma": 0, "rsi": 0, "bb": 0, "volume": 0, 
            "regime": 0, "interaction": 0, "time": 0, "other": 0
        }
        
        for feat in selected_features:
            categorized = False
            for key in feature_types:
                if key in feat.lower():
                    feature_types[key] += 1
                    categorized = True
                    break
            if not categorized:
                feature_types["other"] += 1
        
        summary_lines.extend([
            "",
            "Feature types:"
        ])
        for feat_type, count in feature_types.items():
            if count > 0:
                summary_lines.append(f"  {feat_type}: {count}")
        
        reports["summary"] = "\n".join(summary_lines)
        
        # Statistics report
        stats_lines = ["Feature Statistics", "=" * 40]
        
        for split_name, stats in feature_statistics.items():
            stats_lines.extend([
                "",
                f"{split_name.upper()} split:",
                f"  Total features: {stats.get('n_features', 0)}",
                f"  Features with missing values: {len(stats.get('missing_values', {}))}",
                f"  Zero variance features: {len(stats.get('zero_variance', []))}",
                f"  High correlation pairs: {len(stats.get('high_correlation_pairs', []))}"
            ])
        
        reports["statistics"] = "\n".join(stats_lines)
        
        return reports
    
    async def _save_outputs(
        self,
        training_input: Dict[str, Any],
        pipeline_state: Dict[str, Any]
    ) -> None:
        """Save step outputs to disk.
        
        Args:
            training_input: Training input parameters
            pipeline_state: Pipeline state with results
        """
        output_dir = Path(training_input.get("output_dir", "output")) / "step06_feature_engineering"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save engineered data
        if "engineered_data" in pipeline_state:
            for split_name, data in pipeline_state["engineered_data"].items():
                if isinstance(data, pd.DataFrame):
                    file_path = output_dir / f"{split_name}_engineered.parquet"
                    data.to_parquet(file_path)
                    self.logger.info(f"💾 Saved {split_name} engineered data to {file_path}")
        
        # Save selected features
        if "selected_features" in pipeline_state:
            features_path = output_dir / "selected_features.json"
            with open(features_path, 'w') as f:
                json.dump(pipeline_state["selected_features"], f, indent=2)
            self.logger.info(f"💾 Saved selected features to {features_path}")
        
        # Save feature statistics
        if "feature_statistics" in pipeline_state:
            stats_path = output_dir / "feature_statistics.json"
            with open(stats_path, 'w') as f:
                json.dump(pipeline_state["feature_statistics"], f, indent=2)
            self.logger.info(f"💾 Saved feature statistics to {stats_path}")
        
        # Save reports
        if "feature_reports" in pipeline_state:
            for report_name, content in pipeline_state["feature_reports"].items():
                report_path = output_dir / f"{report_name}_report.txt"
                with open(report_path, 'w') as f:
                    f.write(content)
                self.logger.info(f"💾 Saved {report_name} report")
    
    def get_required_inputs(self) -> list:
        """Get list of required inputs for this step."""
        return ["labeled_data or split data with labels"]
    
    def get_produced_outputs(self) -> list:
        """Get list of outputs produced by this step."""
        return [
            "engineered_data", "feature_statistics", 
            "selected_features", "feature_reports"
        ]
    
    def get_dependencies(self) -> list:
        """Get list of step dependencies."""
        return ["05_labeling"]