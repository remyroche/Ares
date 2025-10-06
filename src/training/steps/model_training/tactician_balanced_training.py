"""
Balanced Tactician Training Integration

This module integrates the comprehensive label balancing and sample weighting system
into the existing Tactician training pipeline. It provides a seamless integration
that maintains backward compatibility while adding advanced balancing capabilities.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass

# Import balancing system
try:
    from ..pre_training.label_balancing import ComprehensiveBalancingSystem
    BALANCING_SYSTEM_AVAILABLE = True
except ImportError:
    BALANCING_SYSTEM_AVAILABLE = False

# Import existing training components
try:
    from .tactician_training_step import TacticianTrainingStep, TacticianTrainingConfig
    from .tactician_pre_ml_orchestrator import TacticianPreMLOrchestrator, OrchestratorConfig
    TACTICIAN_TRAINING_AVAILABLE = True
except ImportError:
    TACTICIAN_TRAINING_AVAILABLE = False

# Import the new balancing system
try:
    from ..pre_training.label_balancing import (
        ComprehensiveBalancingSystem,
        BalancingConfig,
        WeightingConfig,
        RegimeConfig,
        ValidationFairnessConfig,
        DEFAULT_BALANCING_CONFIG,
        DEFAULT_WEIGHTING_CONFIG,
        DEFAULT_REGIME_CONFIG,
        DEFAULT_FAIRNESS_CONFIG
    )
    BALANCING_SYSTEM_AVAILABLE = True
except ImportError:
    BALANCING_SYSTEM_AVAILABLE = False

# Import utilities
try:
    from src.utils.tprint import tprint, tprint_info, tprint_warning, tprint_error, tprint_success
    from src.utils.common_operations import safe_divide, validate_dataframe
    TPRINT_AVAILABLE = True
except ImportError:
    TPRINT_AVAILABLE = False


@dataclass
class BalancedTrainingConfig:
    """Configuration for balanced training integration."""

    # Enable/disable balancing and weighting
    enable_balancing: bool = True
    enable_weighting: bool = True
    enable_regime_balancing: bool = True
    enable_validation_fairness: bool = True

    # Balancing configuration
    balancing_config: BalancingConfig = None

    # Weighting configuration
    weighting_config: WeightingConfig = None

    # Regime configuration
    regime_config: RegimeConfig = None

    # Fairness configuration
    fairness_config: ValidationFairnessConfig = None

    # Integration options
    apply_balancing_before_training: bool = True
    apply_balancing_per_model: bool = False  # If True, balance data for each model separately
    save_balancing_report: bool = True

    def __post_init__(self):
        """Post-initialization setup."""
        if self.balancing_config is None:
            self.balancing_config = DEFAULT_BALANCING_CONFIG

        if self.weighting_config is None:
            self.weighting_config = DEFAULT_WEIGHTING_CONFIG

        if self.regime_config is None:
            self.regime_config = DEFAULT_REGIME_CONFIG

        if self.fairness_config is None:
            self.fairness_config = DEFAULT_FAIRNESS_CONFIG


class BalancedTacticianTrainingStep:
    """
    Enhanced Tactician training step with integrated balancing and weighting.

    This class extends the existing TacticianTrainingStep with comprehensive
    label balancing and sample weighting capabilities while maintaining
    full backward compatibility.
    """

    def __init__(self, config: Optional[TacticianTrainingConfig] = None,
                 balanced_config: Optional[BalancedTrainingConfig] = None):
        """Initialize the balanced training step."""
        if TPRINT_AVAILABLE:
            tprint_info("🚀 Initializing Balanced Tactician Training Step")

        # Initialize base training step
        if TACTICIAN_TRAINING_AVAILABLE:
            self.base_trainer = TacticianTrainingStep(config)
        else:
            raise ImportError("TacticianTrainingStep not available")

        # Initialize balancing system
        if BALANCING_SYSTEM_AVAILABLE:
            self.balanced_config = balanced_config or BalancedTrainingConfig()
            self.balancing_system = ComprehensiveBalancingSystem(
                self.balanced_config.balancing_config,
                self.balanced_config.weighting_config,
                self.balanced_config.regime_config,
                self.balanced_config.fairness_config
            )

            if TPRINT_AVAILABLE:
                tprint_success("✅ Balancing system initialized")
        else:
            self.balancing_system = None
            if TPRINT_AVAILABLE:
                tprint_warning("⚠️ Balancing system not available")

        # Store configuration
        self.config = config
        self.balanced_config = balanced_config or BalancedTrainingConfig()

        if TPRINT_AVAILABLE:
            tprint_success("✅ BalancedTacticianTrainingStep initialized")

    async def train_tactician_models(
        self,
        analyst_signals: pd.DataFrame,
        market_data: pd.DataFrame,
        feature_names: List[str],
        **kwargs
    ) -> Any:
        """
        Train Tactician models with integrated balancing and weighting.

        Args:
            analyst_signals: DataFrame with Analyst signals and confidence scores
            market_data: Raw market data for feature generation
            feature_names: List of base feature names
            **kwargs: Additional parameters

        Returns:
            Training result with balancing information
        """
        if TPRINT_AVAILABLE:
            tprint_info("🚀 Starting balanced Tactician training")

        # Step 1: Run base training (includes pre-ML orchestration)
        if TPRINT_AVAILABLE:
            tprint_info("📊 Step 1: Running base training pipeline...")

        base_result = await self.base_trainer.train_tactician_models(
            analyst_signals, market_data, feature_names, **kwargs
        )

        if not base_result or not hasattr(base_result, 'orchestration_result'):
            if TPRINT_AVAILABLE:
                tprint_error("❌ Base training failed")
            return base_result

        # Step 2: Apply balancing and weighting if enabled
        if (self.balancing_system and
            self.balanced_config.enable_balancing and
            hasattr(base_result, 'orchestration_result') and
            base_result.orchestration_result):

            if TPRINT_AVAILABLE:
                tprint_info("⚖️ Step 2: Applying label balancing and sample weighting...")

            try:
                # Get training data from orchestration result
                training_data = base_result.orchestration_result.training_data

                if training_data is not None and not training_data.empty:
                    # Apply comprehensive balancing and weighting
                    balancing_result = await self._apply_balancing_and_weighting(training_data)

                    # Update the result with balancing information
                    base_result.balancing_applied = True
                    base_result.balanced_training_data = balancing_result.get('balanced_data')
                    base_result.balancing_report = balancing_result.get('balancing_report')

                    # Update sample counts
                    if balancing_result.get('balanced_data') is not None:
                        base_result.total_samples = len(balancing_result['balanced_data'])

                    if TPRINT_AVAILABLE:
                        tprint_success("✅ Balancing and weighting applied successfully")

                else:
                    if TPRINT_AVAILABLE:
                        tprint_warning("⚠️ No training data available for balancing")

            except Exception as e:
                if TPRINT_AVAILABLE:
                    tprint_warning(f"⚠️ Balancing failed: {e}, continuing with original data")

        # Step 3: Continue with base model training using balanced data
        if (hasattr(base_result, 'balanced_training_data') and
            base_result.balanced_training_data is not None):

            if TPRINT_AVAILABLE:
                tprint_info("📈 Step 3: Training models with balanced data...")

            # Use balanced data for model training
            balanced_data = base_result.balanced_training_data

            # Extract features and targets from balanced data
            selected_features = base_result.orchestration_result.selected_features
            target_columns = [col for col in balanced_data.columns if col.startswith('target_')]

            if selected_features and target_columns:
                # Train base models with balanced data
                base_training_result = await self._train_base_models_with_balanced_data(
                    balanced_data, selected_features, target_columns, base_result
                )

                # Update base result with balanced training results
                if base_training_result:
                    base_result.base_models = base_training_result.get('models', {})
                    base_result.base_training_metrics = base_training_result.get('metrics', {})

                    if TPRINT_AVAILABLE:
                        tprint_success("✅ Base models trained with balanced data")

        if TPRINT_AVAILABLE:
            tprint_success("✅ Balanced Tactician training completed")

        return base_result

    def balance_and_weight(self, X: pd.DataFrame, y: pd.Series,
                          sample_weight: Optional[pd.Series] = None,
                          additional_features: Optional[Dict[str, pd.Series]] = None) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """
        Apply comprehensive balancing and weighting.

        Args:
            X: Feature matrix
            y: Target labels
            sample_weight: Optional existing sample weights
            additional_features: Optional additional features

        Returns:
            Tuple of (balanced_X, balanced_y, final_weights)
        """
        if not self.balancing_system:
            return X, y, sample_weight or pd.Series(1.0, index=X.index)

        # Delegate to the balancing system
        return self.balancing_system.balance_and_weight(X, y, sample_weight, additional_features)

    async def _apply_balancing_and_weighting(self, training_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Apply comprehensive balancing and weighting to training data.

        Args:
            training_data: Original training data

        Returns:
            Dictionary with balanced data and balancing report
        """
        if not self.balancing_system:
            return {
                'balanced_data': training_data,
                'balancing_report': {'error': 'Balancing system not available'}
            }

        try:
            # Prepare data for balancing
            # Extract features (exclude target columns and metadata)
            exclude_cols = [col for col in training_data.columns if col.startswith('target_')]
            exclude_cols.extend(['sample_weight', 'timestamp', 'regime', 'analyst_signal'])

            feature_cols = [col for col in training_data.columns if col not in exclude_cols]
            X = training_data[feature_cols]

            # Extract targets
            target_cols = [col for col in training_data.columns if col.startswith('target_')]
            if target_cols:
                # For simplicity, use the first target for balancing
                # In practice, you might want to balance each target separately
                y = training_data[target_cols[0]]
            else:
                raise ValueError("No target columns found for balancing")

            # Extract existing sample weights if available
            sample_weight = training_data.get('sample_weight')
            if sample_weight is not None:
                sample_weight = sample_weight.copy()

            # Prepare additional features for weighting
            additional_features = {}

            # Add regime information if available
            if 'regime' in training_data.columns:
                additional_features['regime'] = training_data['regime']

            # Add analyst confidence if available
            analyst_cols = [col for col in training_data.columns if 'analyst' in col.lower()]
            if analyst_cols:
                additional_features['analyst_confidence'] = training_data[analyst_cols[0]]

            # Add volatility information if available
            volatility_cols = [col for col in training_data.columns if 'volatility' in col.lower()]
            if volatility_cols:
                additional_features['volatility'] = training_data[volatility_cols[0]]

            # Apply balancing and weighting
            X_balanced, y_balanced, final_weights = self.balancing_system.balance_and_weight(
                X, y, sample_weight, additional_features
            )

            # Reconstruct balanced training data
            balanced_data = X_balanced.copy()

            # Add back target columns (reconstructed from balanced y)
            for i, target_col in enumerate(target_cols):
                if i == 0:
                    # Use the balanced target for the first column
                    balanced_data[target_col] = y_balanced
                else:
                    # For multiple targets, use the original targets (simplified approach)
                    # In practice, you'd want to balance each target separately
                    balanced_data[target_col] = training_data[target_col].loc[X_balanced.index]

            # Add final sample weights
            balanced_data['sample_weight'] = final_weights

            # Generate balancing report
            balancing_report = self._generate_balancing_report(
                training_data, balanced_data, X, y, X_balanced, y_balanced
            )

            return {
                'balanced_data': balanced_data,
                'balancing_report': balancing_report,
                'original_samples': len(training_data),
                'balanced_samples': len(balanced_data)
            }

        except Exception as e:
            if TPRINT_AVAILABLE:
                tprint_error(f"❌ Balancing failed: {e}")
            return {
                'balanced_data': training_data,
                'balancing_report': {'error': str(e)},
                'original_samples': len(training_data),
                'balanced_samples': len(training_data)
            }

    def _generate_balancing_report(self, original_data: pd.DataFrame,
                                  balanced_data: pd.DataFrame,
                                  X: pd.DataFrame, y: pd.Series,
                                  X_balanced: pd.DataFrame, y_balanced: pd.Series) -> Dict[str, Any]:
        """Generate comprehensive balancing report."""
        report = {
            'balancing_applied': True,
            'original_dataset_info': {
                'total_samples': len(original_data),
                'class_distribution': y.value_counts().to_dict() if not y.empty else {},
                'class_balance_ratio': self._compute_balance_ratio(y)
            },
            'balanced_dataset_info': {
                'total_samples': len(balanced_data),
                'class_distribution': y_balanced.value_counts().to_dict() if not y_balanced.empty else {},
                'class_balance_ratio': self._compute_balance_ratio(y_balanced)
            },
            'balancing_effectiveness': {
                'sample_reduction_ratio': len(balanced_data) / len(original_data) if len(original_data) > 0 else 1.0,
                'minority_class_boost': self._compute_minority_boost(y, y_balanced)
            }
        }

        return report

    def _compute_balance_ratio(self, y: pd.Series) -> float:
        """Compute class balance ratio (0 = perfect balance, 1 = extreme imbalance)."""
        if y.empty:
            return 1.0

        class_counts = y.value_counts()
        if len(class_counts) <= 1:
            return 0.0

        max_ratio = class_counts.iloc[0] / class_counts.iloc[-1]
        return (max_ratio - 1) / (max_ratio + 1)  # Normalize to [0, 1]

    def _compute_minority_boost(self, original_y: pd.Series, balanced_y: pd.Series) -> Dict[int, float]:
        """Compute how much each minority class was boosted."""
        if original_y.empty or balanced_y.empty:
            return {}

        original_counts = original_y.value_counts()
        balanced_counts = balanced_y.value_counts()

        boost_ratios = {}
        for class_label in original_counts.index:
            original_count = original_counts.get(class_label, 0)
            balanced_count = balanced_counts.get(class_label, 0)

            if original_count > 0:
                boost_ratios[int(class_label)] = balanced_count / original_count
            else:
                boost_ratios[int(class_label)] = 0.0

        return boost_ratios

    async def _train_base_models_with_balanced_data(
        self, balanced_data: pd.DataFrame,
        selected_features: List[str],
        target_columns: List[str],
        base_result: Any
    ) -> Dict[str, Any]:
        """Train base models using balanced data."""
        try:
            if TPRINT_AVAILABLE:
                tprint_info("🔧 Training base models with balanced data...")

            # Use the same training logic as the base trainer but with balanced data
            training_config = {
                'model_type': 'ALL',  # Train all model types
                'training_data': balanced_data,
                'feature_columns': selected_features,
                'target_columns': target_columns,
                'sample_weight': balanced_data.get('sample_weight'),
                'save_models': self.config.save_models if self.config else True,
                'output_directory': f"{self.config.output_directory if self.config else 'generated/tactician_training'}/base_models/balanced"
            }

            # Call the base trainer's training method
            training_result = await self.base_trainer.base_trainer.train_tactician_models(
                **training_config
            )

            return training_result

        except Exception as e:
            if TPRINT_AVAILABLE:
                tprint_error(f"❌ Balanced base model training failed: {e}")
            return {}

    def check_validation_fairness(self, train_data: pd.DataFrame,
                                 val_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Check validation fairness between training and validation sets.

        Args:
            train_data: Training data
            val_data: Validation data

        Returns:
            Fairness report
        """
        if not self.balancing_system or not self.balanced_config.enable_validation_fairness:
            return {'fairness_check_disabled': True}

        # Prepare data for fairness check
        train_dict = {'y': train_data.filter(like='target_').iloc[:, 0] if not train_data.filter(like='target_').empty else None}
        val_dict = {'y': val_data.filter(like='target_').iloc[:, 0] if not val_data.filter(like='target_').empty else None}

        # Add regime information if available
        if 'regime' in train_data.columns:
            train_dict['regime'] = train_data['regime']
            val_dict['regime'] = val_data['regime']

        return self.balancing_system.check_validation_fairness(train_dict, val_dict)


# Convenience function for easy integration
def create_balanced_tactician_trainer(
    training_config: Optional[TacticianTrainingConfig] = None,
    balancing_config: Optional[BalancedTrainingConfig] = None
) -> BalancedTacticianTrainingStep:
    """
    Create a balanced Tactician trainer with default configurations.

    Args:
        training_config: Base training configuration
        balancing_config: Balancing configuration

    Returns:
        Configured balanced trainer
    """
    return BalancedTacticianTrainingStep(training_config, balancing_config)