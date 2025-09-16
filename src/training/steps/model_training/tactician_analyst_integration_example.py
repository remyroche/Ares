"""
Tactician-Analyst Integration Example

This example demonstrates how to integrate the Analyst and Tactician models
with the requested adjustments:

1. Only train Tactician on periods where Analyst gives green light
2. Include Analyst's outputs as input features for Tactician
3. Use optimal barrier settings for entry point finding
4. Single model for all regimes (not per-regime)
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
import logging

from src.training.steps.model_training.tactician_single_model_training import (
    TacticianSingleModelTrainingStep,
    create_tactician_single_model_training_step,
    execute_tactician_single_model_training
)
from src.training.steps.market_analysis.triple_barrier_labeling.tactician_barrier_config import (
    TacticianBarrierLabeler,
    create_tactician_barrier_labeler,
    apply_tactician_barrier_labeling
)
from src.utils.ml_common.config import TacticianTrainingConfig

logger = logging.getLogger(__name__)


class TacticianAnalystIntegration:
    """
    Integration class for Analyst and Tactician models.
    
    This class demonstrates the complete workflow:
    1. Analyst provides green light signals and model outputs
    2. Tactician uses these signals to filter training data
    3. Tactician includes Analyst outputs as features
    4. Tactician trains a single model for entry point optimization
    """
    
    def __init__(self, tactician_config: Optional[TacticianTrainingConfig] = None):
        """Initialize the integration."""
        self.tactician_config = tactician_config or TacticianTrainingConfig()
        self.logger = logger.getChild('TacticianAnalystIntegration')
        
        # Initialize components
        self.tactician_trainer = None
        self.barrier_labeler = None
        
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize the tactician trainer and barrier labeler."""
        try:
            # Initialize tactician trainer
            self.tactician_trainer = create_tactician_single_model_training_step(self.tactician_config)
            
            # Initialize barrier labeler for tactician
            self.barrier_labeler = create_tactician_barrier_labeler(
                profit_take_multiplier=0.0015,  # 0.15% - tight for entry optimization
                stop_loss_multiplier=0.0010,    # 0.10% - tight for entry optimization
                time_barrier_minutes=15,        # 15 minutes - short for entry timing
                entry_window_minutes=5,         # 5-minute entry window
                min_entry_confidence=0.6        # 60% minimum confidence
            )
            
            self.logger.info("✅ Tactician-Analyst integration components initialized")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize components: {e}")
            raise
    
    def prepare_tactician_training_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        regime_labels: np.ndarray,
        analyst_signals: np.ndarray,
        analyst_model_outputs: Optional[np.ndarray] = None,
        all_analyst_models_outputs: Optional[Dict[str, np.ndarray]] = None,
        hmm_regime_features: Optional[np.ndarray] = None,
        feature_names: Optional[list] = None
    ) -> Dict[str, Any]:
        """
        Prepare training data for Tactician with Analyst integration.
        
        Args:
            X: Base features (1m timeframe)
            y: Target values (from tactician barrier labeling)
            regime_labels: Regime labels
            analyst_signals: Binary signals from Analyst (green light indicators)
            analyst_model_outputs: Analyst model predictions
            all_analyst_models_outputs: All individual analyst ML model outputs
            hmm_regime_features: HMM regime features
            feature_names: Names of base features
            
        Returns:
            Dictionary containing prepared data and metrics
        """
        try:
            self.logger.info("🔄 Preparing Tactician training data with Analyst integration...")
            
            # Step 1: Filter to green light periods
            green_light_mask = analyst_signals == 1
            green_light_count = np.sum(green_light_mask)
            green_light_rate = green_light_count / len(analyst_signals)
            
            self.logger.info(f"📊 Analyst green light rate: {green_light_rate:.2%} ({green_light_count}/{len(analyst_signals)})")
            
            if green_light_count == 0:
                raise ValueError("No analyst green light signals found")
            
            # Step 2: Apply filtering
            X_filtered = X[green_light_mask]
            y_filtered = y[green_light_mask]
            regime_labels_filtered = regime_labels[green_light_mask]
            
            # Filter additional data if provided
            if analyst_model_outputs is not None:
                analyst_model_outputs = analyst_model_outputs[green_light_mask]
            
            if all_analyst_models_outputs is not None:
                all_analyst_models_outputs = {
                    model_name: outputs[green_light_mask]
                    for model_name, outputs in all_analyst_models_outputs.items()
                }
            
            if hmm_regime_features is not None:
                hmm_regime_features = hmm_regime_features[green_light_mask]
            
            # Step 3: Prepare combined features
            preparation_result = self._prepare_combined_features(
                X_filtered, feature_names, hmm_regime_features,
                analyst_model_outputs, all_analyst_models_outputs
            )
            
            # Step 4: Compile results
            result = {
                'X': preparation_result['X_combined'],
                'y': y_filtered,
                'regime_labels': regime_labels_filtered,
                'feature_names': preparation_result['feature_names'],
                'preparation_metrics': {
                    'original_samples': X.shape[0],
                    'green_light_samples': green_light_count,
                    'green_light_rate': green_light_rate,
                    'final_samples': X_filtered.shape[0],
                    'base_features': X.shape[1],
                    'total_features': preparation_result['X_combined'].shape[1],
                    'analyst_features_added': preparation_result['analyst_features_added'],
                    'hmm_features_added': preparation_result['hmm_features_added']
                }
            }
            
            self.logger.info(f"✅ Data preparation completed: {result['preparation_metrics']['final_samples']} samples, {result['preparation_metrics']['total_features']} features")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Data preparation failed: {e}")
            raise
    
    def _prepare_combined_features(
        self,
        X: np.ndarray,
        feature_names: Optional[list],
        hmm_regime_features: Optional[np.ndarray],
        analyst_model_outputs: Optional[np.ndarray],
        all_analyst_models_outputs: Optional[Dict[str, np.ndarray]]
    ) -> Dict[str, Any]:
        """Prepare combined features including Analyst outputs."""
        try:
            additional_features = []
            additional_feature_names = []
            analyst_features_added = 0
            hmm_features_added = 0
            
            # Add HMM regime features
            if hmm_regime_features is not None:
                additional_features.append(hmm_regime_features)
                additional_feature_names.extend([f"hmm_regime_{i}" for i in range(hmm_regime_features.shape[1])])
                hmm_features_added = hmm_regime_features.shape[1]
                self.logger.info(f"📊 Added {hmm_features_added} HMM regime features")
            
            # Add all analyst model outputs
            if all_analyst_models_outputs is not None:
                for model_name, model_outputs in all_analyst_models_outputs.items():
                    additional_features.append(model_outputs)
                    additional_feature_names.extend([f"analyst_{model_name}_{i}" for i in range(model_outputs.shape[1])])
                    analyst_features_added += model_outputs.shape[1]
                
                self.logger.info(f"📊 Added {analyst_features_added} analyst model features from {len(all_analyst_models_outputs)} models")
            
            # Add legacy analyst outputs
            if analyst_model_outputs is not None:
                additional_features.append(analyst_model_outputs)
                additional_feature_names.extend([f"analyst_legacy_{i}" for i in range(analyst_model_outputs.shape[1])])
                analyst_features_added += analyst_model_outputs.shape[1]
                self.logger.info(f"📊 Added {analyst_model_outputs.shape[1]} legacy analyst features")
            
            # Combine all features
            if additional_features:
                X_combined = np.column_stack([X] + additional_features)
                feature_names_combined = (feature_names or [f"feature_{i}" for i in range(X.shape[1])]) + additional_feature_names
            else:
                X_combined = X
                feature_names_combined = feature_names or [f"feature_{i}" for i in range(X.shape[1])]
            
            return {
                'X_combined': X_combined,
                'feature_names': feature_names_combined,
                'analyst_features_added': analyst_features_added,
                'hmm_features_added': hmm_features_added
            }
            
        except Exception as e:
            self.logger.error(f"❌ Feature combination failed: {e}")
            raise
    
    def train_tactician_with_analyst_integration(
        self,
        X: np.ndarray,
        y: np.ndarray,
        regime_labels: np.ndarray,
        analyst_signals: np.ndarray,
        analyst_model_outputs: Optional[np.ndarray] = None,
        all_analyst_models_outputs: Optional[Dict[str, np.ndarray]] = None,
        hmm_regime_features: Optional[np.ndarray] = None,
        feature_names: Optional[list] = None
    ) -> Dict[str, Any]:
        """
        Train Tactician model with full Analyst integration.
        
        This method implements all the requested adjustments:
        1. Only trains on Analyst green light periods
        2. Includes Analyst outputs as features
        3. Uses single model for all regimes
        4. Optimized for entry point finding
        """
        try:
            self.logger.info("🚀 Starting Tactician training with Analyst integration...")
            
            # Step 1: Prepare training data
            prepared_data = self.prepare_tactician_training_data(
                X, y, regime_labels, analyst_signals,
                analyst_model_outputs, all_analyst_models_outputs,
                hmm_regime_features, feature_names
            )
            
            # Step 2: Train tactician model
            training_result = self.tactician_trainer.execute(
                X=prepared_data['X'],
                y=prepared_data['y'],
                regime_labels=prepared_data['regime_labels'],
                feature_names=prepared_data['feature_names'],
                analyst_signals=analyst_signals,
                analyst_model_outputs=analyst_model_outputs,
                hmm_regime_features=hmm_regime_features,
                all_analyst_models_outputs=all_analyst_models_outputs
            )
            
            # Step 3: Add integration metadata
            training_result['integration_metadata'] = {
                'analyst_integration': True,
                'green_light_filtering': True,
                'analyst_features_included': True,
                'single_model_training': True,
                'entry_point_optimization': True,
                'preparation_metrics': prepared_data['preparation_metrics']
            }
            
            self.logger.info("✅ Tactician training with Analyst integration completed")
            
            return training_result
            
        except Exception as e:
            self.logger.error(f"❌ Tactician training with integration failed: {e}")
            raise
    
    def generate_tactician_labels(
        self,
        ohlc_data: pd.DataFrame,
        analyst_signals: np.ndarray
    ) -> Dict[str, Any]:
        """
        Generate tactician-specific labels using optimized barriers.
        
        Args:
            ohlc_data: OHLC data with timestamps
            analyst_signals: Binary signals from Analyst
            
        Returns:
            Dictionary containing labeled data and metrics
        """
        try:
            self.logger.info("🏷️ Generating tactician-specific labels...")
            
            # Apply tactician-specific barrier labeling
            result = self.barrier_labeler.apply_tactician_labeling(ohlc_data, analyst_signals)
            
            if result['success']:
                self.logger.info(f"✅ Label generation completed: {len(result['labeled_data'])} samples")
                if 'tactician_metrics' in result:
                    metrics = result['tactician_metrics']
                    self.logger.info(f"📊 High-confidence entries: {metrics.get('high_confidence_entries', 0)}")
                    self.logger.info(f"📊 Average confidence: {metrics.get('avg_confidence', 0):.3f}")
            else:
                self.logger.error(f"❌ Label generation failed: {result.get('error_message', 'Unknown error')}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Label generation failed: {e}")
            raise


def create_tactician_analyst_integration(
    tactician_config: Optional[TacticianTrainingConfig] = None
) -> TacticianAnalystIntegration:
    """Create a tactician-analyst integration instance."""
    return TacticianAnalystIntegration(tactician_config)


def demonstrate_tactician_analyst_integration():
    """Demonstrate the complete tactician-analyst integration workflow."""
    print("🎯 Tactician-Analyst Integration Demonstration")
    print("=" * 60)
    
    # Create sample data
    n_samples = 1000
    n_features = 50
    n_regimes = 3
    
    # Generate sample data
    X = np.random.randn(n_samples, n_features)
    y = np.random.randn(n_samples)
    regime_labels = np.random.choice(n_regimes, n_samples)
    
    # Generate analyst signals (20% green light rate)
    analyst_signals = np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
    
    # Generate analyst model outputs
    analyst_model_outputs = np.random.randn(n_samples, 5)  # 5 outputs per model
    all_analyst_models_outputs = {
        'model_1': np.random.randn(n_samples, 3),
        'model_2': np.random.randn(n_samples, 4),
        'model_3': np.random.randn(n_samples, 2)
    }
    
    # Generate HMM regime features
    hmm_regime_features = np.random.randn(n_samples, 8)
    
    # Create feature names
    feature_names = [f"feature_{i}" for i in range(n_features)]
    
    # Create OHLC data for labeling
    dates = pd.date_range('2024-01-01', periods=n_samples, freq='1min')
    ohlc_data = pd.DataFrame({
        'open': np.random.uniform(100, 110, n_samples),
        'high': np.random.uniform(105, 115, n_samples),
        'low': np.random.uniform(95, 105, n_samples),
        'close': np.random.uniform(100, 110, n_samples),
        'volume': np.random.uniform(1000, 10000, n_samples)
    }, index=dates)
    
    try:
        # Create integration instance
        integration = create_tactician_analyst_integration()
        
        print("\n📊 Step 1: Generating Tactician Labels")
        print("-" * 40)
        
        # Generate tactician labels
        label_result = integration.generate_tactician_labels(ohlc_data, analyst_signals)
        
        if label_result['success']:
            print(f"✅ Labels generated: {len(label_result['labeled_data'])} samples")
            if 'tactician_metrics' in label_result:
                metrics = label_result['tactician_metrics']
                print(f"📊 High-confidence entries: {metrics.get('high_confidence_entries', 0)}")
                print(f"📊 Average confidence: {metrics.get('avg_confidence', 0):.3f}")
        else:
            print(f"❌ Label generation failed: {label_result.get('error_message', 'Unknown error')}")
            return
        
        print("\n📊 Step 2: Training Tactician with Analyst Integration")
        print("-" * 40)
        
        # Train tactician with analyst integration
        training_result = integration.train_tactician_with_analyst_integration(
            X=X,
            y=y,
            regime_labels=regime_labels,
            analyst_signals=analyst_signals,
            analyst_model_outputs=analyst_model_outputs,
            all_analyst_models_outputs=all_analyst_models_outputs,
            hmm_regime_features=hmm_regime_features,
            feature_names=feature_names
        )
        
        if 'error' not in training_result:
            print("✅ Tactician training completed successfully")
            
            # Display integration metrics
            if 'integration_metadata' in training_result:
                metadata = training_result['integration_metadata']
                prep_metrics = metadata.get('preparation_metrics', {})
                
                print(f"📊 Original samples: {prep_metrics.get('original_samples', 0):,}")
                print(f"📊 Green light samples: {prep_metrics.get('green_light_samples', 0):,}")
                print(f"📊 Green light rate: {prep_metrics.get('green_light_rate', 0):.2%}")
                print(f"📊 Final training samples: {prep_metrics.get('final_samples', 0):,}")
                print(f"📊 Base features: {prep_metrics.get('base_features', 0)}")
                print(f"📊 Analyst features added: {prep_metrics.get('analyst_features_added', 0)}")
                print(f"📊 HMM features added: {prep_metrics.get('hmm_features_added', 0)}")
                print(f"📊 Total features: {prep_metrics.get('total_features', 0)}")
            
            # Display model performance
            if 'evaluation_results' in training_result:
                eval_results = training_result['evaluation_results']
                if 'single_model' in eval_results:
                    model_perf = eval_results['single_model']
                    print(f"📊 Model performance:")
                    for metric, value in model_perf.items():
                        print(f"   {metric}: {value:.4f}")
        else:
            print(f"❌ Training failed: {training_result.get('error_message', 'Unknown error')}")
        
        print("\n🎯 Integration Features Demonstrated:")
        print("✅ 1. Green light filtering - Only trains on Analyst green light periods")
        print("✅ 2. Analyst features - Includes Analyst model outputs as features")
        print("✅ 3. Single model - Uses one model for all regimes (not per-regime)")
        print("✅ 4. Entry optimization - Optimized barriers for entry point finding")
        print("✅ 5. 1m timeframe - Operates on 1-minute data for precise timing")
        
    except Exception as e:
        print(f"❌ Integration demonstration failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    demonstrate_tactician_analyst_integration()