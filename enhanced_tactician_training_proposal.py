"""
Enhanced Tactician Training with Comprehensive Feature Integration

This implementation shows how to properly integrate ALL Analyst and HMM outputs
for comprehensive Tactician training, not just binary green light signals.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
import logging

logger = logging.getLogger(__name__)

class EnhancedTacticianTraining:
    """
    Enhanced Tactician training that integrates ALL Analyst and HMM outputs,
    not just binary green light signals.
    """
    
    def __init__(self, confidence_threshold: float = 0.5):
        self.confidence_threshold = confidence_threshold
        self.logger = logger.getChild('EnhancedTacticianTraining')
    
    def prepare_comprehensive_training_data(
        self,
        # Base data
        X: np.ndarray,
        y: np.ndarray,
        regime_labels: np.ndarray,
        feature_names: Optional[List[str]] = None,
        
        # Analyst outputs (comprehensive)
        analyst_signals: Optional[np.ndarray] = None,
        analyst_confidence_scores: Optional[np.ndarray] = None,
        analyst_signal_strength: Optional[np.ndarray] = None,
        analyst_risk_score: Optional[np.ndarray] = None,
        analyst_regime_label: Optional[np.ndarray] = None,
        all_analyst_models_outputs: Optional[Dict[str, np.ndarray]] = None,
        analyst_ensemble_outputs: Optional[np.ndarray] = None,
        
        # HMM outputs (comprehensive)
        hmm_regime_features: Optional[np.ndarray] = None,
        hmm_model_outputs: Optional[np.ndarray] = None,
        hmm_transition_probs: Optional[np.ndarray] = None,
        hmm_state_sequence: Optional[np.ndarray] = None,
        hmm_states: Optional[np.ndarray] = None,
        
        # Timestamps for temporal alignment
        timestamps: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Prepare comprehensive training data with ALL Analyst and HMM outputs.
        
        This is the key enhancement: Instead of just using binary green light signals,
        we integrate ALL available information from Analyst and HMM models.
        """
        try:
            self.logger.info("🔄 Preparing comprehensive Tactician training data...")
            
            # Step 1: Apply confidence-based filtering (not just binary)
            filtered_data = self._apply_confidence_filtering(
                X, y, regime_labels, analyst_confidence_scores, timestamps
            )
            
            # Step 2: Integrate ALL Analyst outputs as features
            analyst_features = self._integrate_analyst_outputs(
                filtered_data['X'], filtered_data['y'], filtered_data['regime_labels'],
                analyst_signals, analyst_confidence_scores, analyst_signal_strength,
                analyst_risk_score, analyst_regime_label, all_analyst_models_outputs,
                analyst_ensemble_outputs, filtered_data['mask']
            )
            
            # Step 3: Integrate ALL HMM outputs as features
            hmm_features = self._integrate_hmm_outputs(
                analyst_features['X'], analyst_features['y'], analyst_features['regime_labels'],
                hmm_regime_features, hmm_model_outputs, hmm_transition_probs,
                hmm_state_sequence, hmm_states, filtered_data['mask']
            )
            
            # Step 4: Create comprehensive feature names
            feature_names_combined = self._create_comprehensive_feature_names(
                feature_names, analyst_features['feature_names'], hmm_features['feature_names']
            )
            
            # Step 5: Compile results
            result = {
                'X': hmm_features['X'],
                'y': hmm_features['y'],
                'regime_labels': hmm_features['regime_labels'],
                'feature_names': feature_names_combined,
                'timestamps': filtered_data['timestamps'],
                'preparation_metrics': {
                    'original_samples': X.shape[0],
                    'filtered_samples': filtered_data['X'].shape[0],
                    'confidence_filtering_rate': np.mean(filtered_data['mask']),
                    'base_features': X.shape[1],
                    'analyst_features_added': analyst_features['features_added'],
                    'hmm_features_added': hmm_features['features_added'],
                    'total_features': hmm_features['X'].shape[1],
                    'comprehensive_integration': True
                }
            }
            
            self.logger.info(f"✅ Comprehensive data preparation completed: {result['preparation_metrics']['total_features']} features")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Comprehensive data preparation failed: {e}")
            raise
    
    def _apply_confidence_filtering(
        self, X: np.ndarray, y: np.ndarray, regime_labels: np.ndarray,
        analyst_confidence_scores: Optional[np.ndarray], timestamps: Optional[np.ndarray]
    ) -> Dict[str, Any]:
        """Apply confidence-based filtering instead of binary green light."""
        
        if analyst_confidence_scores is not None:
            # Use confidence threshold filtering
            confidence_mask = analyst_confidence_scores >= self.confidence_threshold
            confidence_rate = np.mean(confidence_mask)
            
            self.logger.info(f"📊 Confidence filtering: {confidence_rate:.2%} above {self.confidence_threshold}")
            
            if confidence_rate < 0.1:  # Less than 10% data
                self.logger.warning("⚠️ Low confidence data rate, using all data")
                return {
                    'X': X, 'y': y, 'regime_labels': regime_labels,
                    'timestamps': timestamps, 'mask': np.ones(len(X), dtype=bool)
                }
            
            # Apply filtering
            return {
                'X': X[confidence_mask],
                'y': y[confidence_mask],
                'regime_labels': regime_labels[confidence_mask],
                'timestamps': timestamps[confidence_mask] if timestamps is not None else None,
                'mask': confidence_mask
            }
        else:
            # Fallback to all data
            self.logger.warning("⚠️ No confidence scores provided, using all data")
            return {
                'X': X, 'y': y, 'regime_labels': regime_labels,
                'timestamps': timestamps, 'mask': np.ones(len(X), dtype=bool)
            }
    
    def _integrate_analyst_outputs(
        self, X: np.ndarray, y: np.ndarray, regime_labels: np.ndarray,
        analyst_signals: Optional[np.ndarray], analyst_confidence_scores: Optional[np.ndarray],
        analyst_signal_strength: Optional[np.ndarray], analyst_risk_score: Optional[np.ndarray],
        analyst_regime_label: Optional[np.ndarray], all_analyst_models_outputs: Optional[Dict[str, np.ndarray]],
        analyst_ensemble_outputs: Optional[np.ndarray], mask: np.ndarray
    ) -> Dict[str, Any]:
        """Integrate ALL Analyst outputs as features."""
        
        additional_features = []
        additional_feature_names = []
        features_added = 0
        
        # 1. Binary signals (if available)
        if analyst_signals is not None:
            analyst_signals_filtered = analyst_signals[mask]
            additional_features.append(analyst_signals_filtered.reshape(-1, 1))
            additional_feature_names.append("analyst_binary_signal")
            features_added += 1
        
        # 2. Confidence scores (key missing piece!)
        if analyst_confidence_scores is not None:
            confidence_filtered = analyst_confidence_scores[mask]
            additional_features.append(confidence_filtered.reshape(-1, 1))
            additional_feature_names.append("analyst_confidence_score")
            features_added += 1
        
        # 3. Signal strength (key missing piece!)
        if analyst_signal_strength is not None:
            signal_strength_filtered = analyst_signal_strength[mask]
            additional_features.append(signal_strength_filtered.reshape(-1, 1))
            additional_feature_names.append("analyst_signal_strength")
            features_added += 1
        
        # 4. Risk score (key missing piece!)
        if analyst_risk_score is not None:
            risk_score_filtered = analyst_risk_score[mask]
            additional_features.append(risk_score_filtered.reshape(-1, 1))
            additional_feature_names.append("analyst_risk_score")
            features_added += 1
        
        # 5. Regime label (key missing piece!)
        if analyst_regime_label is not None:
            regime_label_filtered = analyst_regime_label[mask]
            additional_features.append(regime_label_filtered.reshape(-1, 1))
            additional_feature_names.append("analyst_regime_label")
            features_added += 1
        
        # 6. Individual analyst model outputs
        if all_analyst_models_outputs is not None:
            for model_name, model_outputs in all_analyst_models_outputs.items():
                model_outputs_filtered = model_outputs[mask]
                additional_features.append(model_outputs_filtered)
                
                for i in range(model_outputs_filtered.shape[1]):
                    additional_feature_names.append(f"analyst_{model_name}_output_{i}")
                
                features_added += model_outputs_filtered.shape[1]
        
        # 7. Analyst ensemble outputs
        if analyst_ensemble_outputs is not None:
            ensemble_filtered = analyst_ensemble_outputs[mask]
            additional_features.append(ensemble_filtered)
            
            for i in range(ensemble_filtered.shape[1]):
                additional_feature_names.append(f"analyst_ensemble_output_{i}")
            
            features_added += ensemble_filtered.shape[1]
        
        # Combine features
        if additional_features:
            X_combined = np.column_stack([X] + additional_features)
        else:
            X_combined = X
        
        self.logger.info(f"📊 Added {features_added} analyst features")
        
        return {
            'X': X_combined,
            'y': y,
            'regime_labels': regime_labels,
            'feature_names': additional_feature_names,
            'features_added': features_added
        }
    
    def _integrate_hmm_outputs(
        self, X: np.ndarray, y: np.ndarray, regime_labels: np.ndarray,
        hmm_regime_features: Optional[np.ndarray], hmm_model_outputs: Optional[np.ndarray],
        hmm_transition_probs: Optional[np.ndarray], hmm_state_sequence: Optional[np.ndarray],
        hmm_states: Optional[np.ndarray], mask: np.ndarray
    ) -> Dict[str, Any]:
        """Integrate ALL HMM outputs as features."""
        
        additional_features = []
        additional_feature_names = []
        features_added = 0
        
        # 1. HMM regime features (probabilities, characteristics)
        if hmm_regime_features is not None:
            hmm_regime_filtered = hmm_regime_features[mask]
            additional_features.append(hmm_regime_filtered)
            
            for i in range(hmm_regime_filtered.shape[1]):
                additional_feature_names.append(f"hmm_regime_feature_{i}")
            
            features_added += hmm_regime_filtered.shape[1]
        
        # 2. HMM model outputs (predictions, probabilities)
        if hmm_model_outputs is not None:
            hmm_outputs_filtered = hmm_model_outputs[mask]
            additional_features.append(hmm_outputs_filtered)
            
            for i in range(hmm_outputs_filtered.shape[1]):
                additional_feature_names.append(f"hmm_model_output_{i}")
            
            features_added += hmm_outputs_filtered.shape[1]
        
        # 3. HMM transition probabilities (key missing piece!)
        if hmm_transition_probs is not None:
            transition_filtered = hmm_transition_probs[mask]
            additional_features.append(transition_filtered)
            
            for i in range(transition_filtered.shape[1]):
                additional_feature_names.append(f"hmm_transition_prob_{i}")
            
            features_added += transition_filtered.shape[1]
        
        # 4. HMM state sequence (key missing piece!)
        if hmm_state_sequence is not None:
            state_sequence_filtered = hmm_state_sequence[mask]
            additional_features.append(state_sequence_filtered.reshape(-1, 1))
            additional_feature_names.append("hmm_state_sequence")
            features_added += 1
        
        # 5. HMM states
        if hmm_states is not None:
            hmm_states_filtered = hmm_states[mask]
            additional_features.append(hmm_states_filtered.reshape(-1, 1))
            additional_feature_names.append("hmm_state")
            features_added += 1
        
        # Combine features
        if additional_features:
            X_combined = np.column_stack([X] + additional_features)
        else:
            X_combined = X
        
        self.logger.info(f"📊 Added {features_added} HMM features")
        
        return {
            'X': X_combined,
            'y': y,
            'regime_labels': regime_labels,
            'feature_names': additional_feature_names,
            'features_added': features_added
        }
    
    def _create_comprehensive_feature_names(
        self, base_feature_names: Optional[List[str]], 
        analyst_feature_names: List[str], 
        hmm_feature_names: List[str]
    ) -> List[str]:
        """Create comprehensive feature names for all integrated features."""
        
        # Base features
        if base_feature_names is not None:
            base_names = [f"base_{name}" for name in base_feature_names]
        else:
            base_names = [f"base_feature_{i}" for i in range(50)]  # Assume 50 base features
        
        # Combine all feature names
        all_feature_names = base_names + analyst_feature_names + hmm_feature_names
        
        # Ensure uniqueness
        unique_names = []
        name_counts = {}
        for name in all_feature_names:
            if name in name_counts:
                name_counts[name] += 1
                unique_name = f"{name}_v{name_counts[name]:02d}"
            else:
                name_counts[name] = 0
                unique_name = name
            unique_names.append(unique_name)
        
        return unique_names


def demonstrate_comprehensive_tactician_training():
    """Demonstrate the enhanced Tactician training approach."""
    
    print("🎯 Enhanced Tactician Training Demonstration")
    print("=" * 60)
    
    # Create sample data
    n_samples = 1000
    n_base_features = 50
    
    # Base features
    X = np.random.randn(n_samples, n_base_features)
    y = np.random.randn(n_samples)
    regime_labels = np.random.choice(3, n_samples)
    
    # Analyst outputs (comprehensive)
    analyst_signals = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    analyst_confidence_scores = np.random.uniform(0.3, 0.9, n_samples)
    analyst_signal_strength = np.random.uniform(0.1, 1.0, n_samples)
    analyst_risk_score = np.random.uniform(0.1, 0.8, n_samples)
    analyst_regime_label = np.random.choice(3, n_samples)
    
    # Individual analyst model outputs
    all_analyst_models_outputs = {
        'model_1': np.random.randn(n_samples, 3),
        'model_2': np.random.randn(n_samples, 4),
        'model_3': np.random.randn(n_samples, 2)
    }
    
    # Analyst ensemble outputs
    analyst_ensemble_outputs = np.random.randn(n_samples, 5)
    
    # HMM outputs (comprehensive)
    hmm_regime_features = np.random.randn(n_samples, 8)
    hmm_model_outputs = np.random.randn(n_samples, 6)
    hmm_transition_probs = np.random.randn(n_samples, 9)  # 3x3 transition matrix flattened
    hmm_state_sequence = np.random.choice(3, n_samples)
    hmm_states = np.random.choice(3, n_samples)
    
    # Feature names
    feature_names = [f"feature_{i}" for i in range(n_base_features)]
    
    try:
        # Create enhanced training instance
        enhanced_trainer = EnhancedTacticianTraining(confidence_threshold=0.5)
        
        # Prepare comprehensive training data
        result = enhanced_trainer.prepare_comprehensive_training_data(
            X=X, y=y, regime_labels=regime_labels, feature_names=feature_names,
            analyst_signals=analyst_signals, analyst_confidence_scores=analyst_confidence_scores,
            analyst_signal_strength=analyst_signal_strength, analyst_risk_score=analyst_risk_score,
            analyst_regime_label=analyst_regime_label, all_analyst_models_outputs=all_analyst_models_outputs,
            analyst_ensemble_outputs=analyst_ensemble_outputs, hmm_regime_features=hmm_regime_features,
            hmm_model_outputs=hmm_model_outputs, hmm_transition_probs=hmm_transition_probs,
            hmm_state_sequence=hmm_state_sequence, hmm_states=hmm_states
        )
        
        print("✅ Enhanced Tactician training data prepared successfully")
        
        # Display comprehensive metrics
        metrics = result['preparation_metrics']
        print(f"📊 Original samples: {metrics['original_samples']:,}")
        print(f"📊 Filtered samples: {metrics['filtered_samples']:,}")
        print(f"📊 Confidence filtering rate: {metrics['confidence_filtering_rate']:.2%}")
        print(f"📊 Base features: {metrics['base_features']}")
        print(f"📊 Analyst features added: {metrics['analyst_features_added']}")
        print(f"📊 HMM features added: {metrics['hmm_features_added']}")
        print(f"📊 Total features: {metrics['total_features']}")
        print(f"📊 Comprehensive integration: {metrics['comprehensive_integration']}")
        
        print("\n🎯 Key Enhancements Demonstrated:")
        print("✅ 1. Confidence-based filtering (not just binary green light)")
        print("✅ 2. ALL Analyst outputs as features (confidence, signal strength, risk score)")
        print("✅ 3. ALL HMM outputs as features (transition probs, state sequence)")
        print("✅ 4. Individual analyst model outputs")
        print("✅ 5. Analyst ensemble outputs")
        print("✅ 6. Comprehensive feature naming and organization")
        
    except Exception as e:
        print(f"❌ Enhanced training demonstration failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    demonstrate_comprehensive_tactician_training()