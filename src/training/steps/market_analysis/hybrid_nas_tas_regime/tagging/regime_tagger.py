"""
Regime Tagger

Tags existing data with regime information based on TAS and NAS inputs.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
import time

from ..config.hybrid_config import HybridRegimeConfig


class RegimeTagger:
    """
    Regime tagger that tags existing data with regime information.
    
    This component:
    1. Tags existing data with regime information
    2. Provides regime labels and metadata
    3. Validates tag consistency
    4. Manages tag persistence
    """
    
    def __init__(self, config: HybridRegimeConfig):
        """
        Initialize Regime Tagger.
        
        Args:
            config: Hybrid regime configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize tagging components
        self.tag_history = []
        self.tag_metadata = {}
        self.tag_validation = {}
        
        self.logger.info("✅ Regime Tagger initialized")
        self.logger.info(f"🏷️ Tagging method: {config.tagging_method}")
        self.logger.info(f"📊 Confidence threshold: {config.tagging_confidence_threshold}")
        self.logger.info(f"🔍 Uncertainty threshold: {config.tagging_uncertainty_threshold}")
    
    def tag_data(self, 
                 data: Union[pd.DataFrame, np.ndarray],
                 regime_predictions: np.ndarray,
                 regime_probabilities: Optional[np.ndarray] = None,
                 regime_labels: Optional[List[str]] = None,
                 timestamps: Optional[np.ndarray] = None,
                 metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Tag data with regime information.
        
        Args:
            data: Data to tag
            regime_predictions: Regime predictions
            regime_probabilities: Regime probabilities
            regime_labels: Regime labels
            timestamps: Optional timestamps
            metadata: Optional metadata
            
        Returns:
            Dictionary with tagged data
        """
        start_time = time.time()
        self.logger.info("🏷️ Tagging data with regime information")
        
        try:
            # Prepare data for tagging
            prepared_data = self._prepare_data_for_tagging(data, timestamps)
            
            # Generate regime tags
            regime_tags = self._generate_regime_tags(
                regime_predictions, regime_probabilities, regime_labels
            )
            
            # Generate confidence tags
            confidence_tags = self._generate_confidence_tags(
                regime_probabilities, regime_predictions
            )
            
            # Generate uncertainty tags
            uncertainty_tags = self._generate_uncertainty_tags(
                regime_probabilities, regime_predictions
            )
            
            # Generate economic tags
            economic_tags = self._generate_economic_tags(
                regime_predictions, prepared_data
            )
            
            # Generate financial tags
            financial_tags = self._generate_financial_tags(
                regime_predictions, prepared_data
            )
            
            # Combine all tags
            combined_tags = self._combine_tags(
                regime_tags, confidence_tags, uncertainty_tags, 
                economic_tags, financial_tags
            )
            
            # Validate tags
            validation_results = self._validate_tags(combined_tags)
            
            # Create tagged data
            tagged_data = self._create_tagged_data(
                prepared_data, combined_tags, metadata
            )
            
            # Update tag history
            self._update_tag_history(tagged_data, validation_results)
            
            execution_time = time.time() - start_time
            
            self.logger.info(f"✅ Data tagging completed in {execution_time:.2f}s")
            self.logger.info(f"📊 Tagged {len(tagged_data)} samples")
            self.logger.info(f"🎯 Average confidence: {np.mean(confidence_tags):.3f}")
            self.logger.info(f"🔍 Average uncertainty: {np.mean(uncertainty_tags):.3f}")
            
            return {
                'success': True,
                'tagged_data': tagged_data,
                'regime_tags': regime_tags,
                'confidence_tags': confidence_tags,
                'uncertainty_tags': uncertainty_tags,
                'economic_tags': economic_tags,
                'financial_tags': financial_tags,
                'combined_tags': combined_tags,
                'validation_results': validation_results,
                'execution_time': execution_time,
                'metadata': {
                    'n_samples': len(tagged_data),
                    'n_regimes': len(set(regime_predictions)),
                    'tagging_method': self.config.tagging_method
                }
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"❌ Data tagging failed: {e}")
            
            return {
                'success': False,
                'tagged_data': None,
                'regime_tags': [],
                'confidence_tags': np.array([]),
                'uncertainty_tags': np.array([]),
                'economic_tags': [],
                'financial_tags': [],
                'combined_tags': {},
                'validation_results': {},
                'execution_time': execution_time,
                'error_message': str(e)
            }
    
    def _prepare_data_for_tagging(self, 
                                 data: Union[pd.DataFrame, np.ndarray],
                                 timestamps: Optional[np.ndarray]) -> Dict[str, Any]:
        """Prepare data for tagging."""
        self.logger.info("📊 Preparing data for tagging")
        
        # Convert to DataFrame if needed
        if isinstance(data, np.ndarray):
            if len(data.shape) == 1:
                data = data.reshape(-1, 1)
            df = pd.DataFrame(data, columns=[f"feature_{i}" for i in range(data.shape[1])])
        else:
            df = data.copy()
        
        # Add timestamps if provided
        if timestamps is not None:
            df['timestamp'] = timestamps
        else:
            df['timestamp'] = pd.date_range(start='2023-01-01', periods=len(df), freq='15T')
        
        return {
            'data': df,
            'n_samples': len(df),
            'n_features': len(df.columns),
            'feature_names': df.columns.tolist()
        }
    
    def _generate_regime_tags(self, 
                              regime_predictions: np.ndarray,
                              regime_probabilities: Optional[np.ndarray],
                              regime_labels: Optional[List[str]]) -> List[Dict[str, Any]]:
        """Generate regime tags."""
        self.logger.info("🏷️ Generating regime tags")
        
        regime_tags = []
        
        for i, regime_id in enumerate(regime_predictions):
            tag = {
                'regime_id': int(regime_id),
                'regime_label': regime_labels[regime_id] if regime_labels else f"regime_{regime_id}",
                'regime_probability': float(regime_probabilities[i, regime_id]) if regime_probabilities is not None else 1.0,
                'regime_confidence': float(regime_probabilities[i, regime_id]) if regime_probabilities is not None else 1.0
            }
            
            # Add regime characteristics
            tag['regime_type'] = self._classify_regime_type(regime_id)
            tag['regime_stability'] = self._calculate_regime_stability(regime_predictions, i)
            
            regime_tags.append(tag)
        
        return regime_tags
    
    def _generate_confidence_tags(self, 
                                  regime_probabilities: Optional[np.ndarray],
                                  regime_predictions: np.ndarray) -> np.ndarray:
        """Generate confidence tags."""
        self.logger.info("🎯 Generating confidence tags")
        
        if regime_probabilities is not None:
            # Use maximum probability as confidence
            confidence_tags = np.max(regime_probabilities, axis=1)
        else:
            # Use uniform confidence
            confidence_tags = np.ones(len(regime_predictions))
        
        return confidence_tags
    
    def _generate_uncertainty_tags(self, 
                                   regime_probabilities: Optional[np.ndarray],
                                   regime_predictions: np.ndarray) -> np.ndarray:
        """Generate uncertainty tags."""
        self.logger.info("🔍 Generating uncertainty tags")
        
        if regime_probabilities is not None:
            # Calculate entropy as uncertainty
            uncertainty_tags = np.zeros(len(regime_predictions))
            
            for i, probs in enumerate(regime_probabilities):
                # Normalize probabilities
                probs = probs / (np.sum(probs) + 1e-8)
                # Calculate entropy
                entropy = -np.sum(probs * np.log(probs + 1e-8))
                uncertainty_tags[i] = entropy
        else:
            # Use uniform uncertainty
            uncertainty_tags = np.zeros(len(regime_predictions))
        
        return uncertainty_tags
    
    def _generate_economic_tags(self, 
                                regime_predictions: np.ndarray,
                                prepared_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate economic tags."""
        self.logger.info("🏛️ Generating economic tags")
        
        economic_tags = []
        
        for i, regime_id in enumerate(regime_predictions):
            tag = {
                'economic_regime': self._classify_economic_regime(regime_id),
                'economic_significance': self._calculate_economic_significance(regime_id, prepared_data),
                'economic_stability': self._calculate_economic_stability(regime_predictions, i),
                'economic_volatility': self._calculate_economic_volatility(regime_id, prepared_data)
            }
            
            economic_tags.append(tag)
        
        return economic_tags
    
    def _generate_financial_tags(self, 
                                 regime_predictions: np.ndarray,
                                 prepared_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate financial tags."""
        self.logger.info("💰 Generating financial tags")
        
        financial_tags = []
        
        for i, regime_id in enumerate(regime_predictions):
            tag = {
                'financial_regime': self._classify_financial_regime(regime_id),
                'financial_significance': self._calculate_financial_significance(regime_id, prepared_data),
                'financial_stability': self._calculate_financial_stability(regime_predictions, i),
                'financial_volatility': self._calculate_financial_volatility(regime_id, prepared_data)
            }
            
            financial_tags.append(tag)
        
        return financial_tags
    
    def _combine_tags(self, 
                      regime_tags: List[Dict[str, Any]],
                      confidence_tags: np.ndarray,
                      uncertainty_tags: np.ndarray,
                      economic_tags: List[Dict[str, Any]],
                      financial_tags: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Combine all tags into a single structure."""
        self.logger.info("🔗 Combining tags")
        
        combined_tags = {
            'regime_tags': regime_tags,
            'confidence_tags': confidence_tags,
            'uncertainty_tags': uncertainty_tags,
            'economic_tags': economic_tags,
            'financial_tags': financial_tags,
            'n_samples': len(regime_tags),
            'tag_timestamp': time.time()
        }
        
        return combined_tags
    
    def _validate_tags(self, combined_tags: Dict[str, Any]) -> Dict[str, Any]:
        """Validate tag consistency and quality."""
        self.logger.info("✅ Validating tags")
        
        validation_results = {
            'consistency_check': self._check_tag_consistency(combined_tags),
            'quality_check': self._check_tag_quality(combined_tags),
            'completeness_check': self._check_tag_completeness(combined_tags)
        }
        
        return validation_results
    
    def _check_tag_consistency(self, combined_tags: Dict[str, Any]) -> Dict[str, Any]:
        """Check tag consistency."""
        regime_tags = combined_tags['regime_tags']
        confidence_tags = combined_tags['confidence_tags']
        uncertainty_tags = combined_tags['uncertainty_tags']
        
        # Check for consistent regime IDs
        regime_ids = [tag['regime_id'] for tag in regime_tags]
        unique_regimes = len(set(regime_ids))
        
        # Check confidence-uncertainty relationship
        confidence_uncertainty_correlation = np.corrcoef(confidence_tags, uncertainty_tags)[0, 1]
        
        return {
            'unique_regimes': unique_regimes,
            'confidence_uncertainty_correlation': confidence_uncertainty_correlation,
            'is_consistent': abs(confidence_uncertainty_correlation) < 0.5  # Should be negative
        }
    
    def _check_tag_quality(self, combined_tags: Dict[str, Any]) -> Dict[str, Any]:
        """Check tag quality."""
        confidence_tags = combined_tags['confidence_tags']
        uncertainty_tags = combined_tags['uncertainty_tags']
        
        # Quality metrics
        avg_confidence = np.mean(confidence_tags)
        avg_uncertainty = np.mean(uncertainty_tags)
        confidence_std = np.std(confidence_tags)
        uncertainty_std = np.std(uncertainty_tags)
        
        return {
            'avg_confidence': avg_confidence,
            'avg_uncertainty': avg_uncertainty,
            'confidence_std': confidence_std,
            'uncertainty_std': uncertainty_std,
            'is_high_quality': avg_confidence > self.config.tagging_confidence_threshold and 
                             avg_uncertainty < self.config.tagging_uncertainty_threshold
        }
    
    def _check_tag_completeness(self, combined_tags: Dict[str, Any]) -> Dict[str, Any]:
        """Check tag completeness."""
        regime_tags = combined_tags['regime_tags']
        economic_tags = combined_tags['economic_tags']
        financial_tags = combined_tags['financial_tags']
        
        # Check if all required tags are present
        required_fields = ['regime_id', 'regime_label', 'regime_probability']
        regime_complete = all(
            all(field in tag for field in required_fields) 
            for tag in regime_tags
        )
        
        economic_complete = all(
            all(field in tag for field in ['economic_regime', 'economic_significance'])
            for tag in economic_tags
        )
        
        financial_complete = all(
            all(field in tag for field in ['financial_regime', 'financial_significance'])
            for tag in financial_tags
        )
        
        return {
            'regime_complete': regime_complete,
            'economic_complete': economic_complete,
            'financial_complete': financial_complete,
            'is_complete': regime_complete and economic_complete and financial_complete
        }
    
    def _create_tagged_data(self, 
                           prepared_data: Dict[str, Any],
                           combined_tags: Dict[str, Any],
                           metadata: Optional[Dict[str, Any]]) -> pd.DataFrame:
        """Create tagged data DataFrame."""
        self.logger.info("📊 Creating tagged data")
        
        df = prepared_data['data'].copy()
        
        # Add regime tags
        regime_tags = combined_tags['regime_tags']
        df['regime_id'] = [tag['regime_id'] for tag in regime_tags]
        df['regime_label'] = [tag['regime_label'] for tag in regime_tags]
        df['regime_probability'] = [tag['regime_probability'] for tag in regime_tags]
        df['regime_confidence'] = [tag['regime_confidence'] for tag in regime_tags]
        
        # Add confidence and uncertainty tags
        df['confidence'] = combined_tags['confidence_tags']
        df['uncertainty'] = combined_tags['uncertainty_tags']
        
        # Add economic tags
        economic_tags = combined_tags['economic_tags']
        df['economic_regime'] = [tag['economic_regime'] for tag in economic_tags]
        df['economic_significance'] = [tag['economic_significance'] for tag in economic_tags]
        df['economic_stability'] = [tag['economic_stability'] for tag in economic_tags]
        df['economic_volatility'] = [tag['economic_volatility'] for tag in economic_tags]
        
        # Add financial tags
        financial_tags = combined_tags['financial_tags']
        df['financial_regime'] = [tag['financial_regime'] for tag in financial_tags]
        df['financial_significance'] = [tag['financial_significance'] for tag in financial_tags]
        df['financial_stability'] = [tag['financial_stability'] for tag in financial_tags]
        df['financial_volatility'] = [tag['financial_volatility'] for tag in financial_tags]
        
        # Add metadata
        if metadata:
            for key, value in metadata.items():
                df[f'metadata_{key}'] = value
        
        return df
    
    def _update_tag_history(self, tagged_data: pd.DataFrame, validation_results: Dict[str, Any]):
        """Update tag history."""
        tag_entry = {
            'timestamp': time.time(),
            'n_samples': len(tagged_data),
            'n_regimes': len(set(tagged_data['regime_id'])),
            'validation_results': validation_results,
            'tagged_data': tagged_data
        }
        
        self.tag_history.append(tag_entry)
        
        # Keep only recent history
        if len(self.tag_history) > self.config.tag_history_length:
            self.tag_history = self.tag_history[-self.config.tag_history_length:]
    
    def _classify_regime_type(self, regime_id: int) -> str:
        """Classify regime type based on regime ID."""
        regime_types = {
            0: "normal",
            1: "bull_market",
            2: "bear_market",
            3: "high_volatility",
            4: "low_volatility",
            5: "trending_up",
            6: "trending_down",
            7: "mean_reverting",
            8: "breakout",
            9: "consolidation",
            10: "crisis"
        }
        
        return regime_types.get(regime_id, "unknown")
    
    def _calculate_regime_stability(self, regime_predictions: np.ndarray, index: int) -> float:
        """Calculate regime stability for a specific index."""
        if len(regime_predictions) < 2:
            return 1.0
        
        window_size = min(10, len(regime_predictions) // 4)
        start_idx = max(0, index - window_size // 2)
        end_idx = min(len(regime_predictions), index + window_size // 2 + 1)
        
        window_regimes = regime_predictions[start_idx:end_idx]
        current_regime = regime_predictions[index]
        
        consistency = np.sum(window_regimes == current_regime) / len(window_regimes)
        return float(consistency)
    
    def _classify_economic_regime(self, regime_id: int) -> str:
        """Classify economic regime based on regime ID."""
        economic_regimes = {
            0: "normal",
            1: "expansion",
            2: "recession",
            3: "recovery",
            4: "stagnation",
            5: "inflation",
            6: "deflation",
            7: "stagflation",
            8: "boom",
            9: "bust"
        }
        
        return economic_regimes.get(regime_id, "unknown")
    
    def _calculate_economic_significance(self, regime_id: int, prepared_data: Dict[str, Any]) -> float:
        """Calculate economic significance for a regime."""
        # Base significance on regime type
        significance_map = {
            0: 0.5,  # normal
            1: 0.8,  # expansion
            2: 0.9,  # recession
            3: 0.7,  # recovery
            4: 0.6,  # stagnation
            5: 0.8,  # inflation
            6: 0.8,  # deflation
            7: 0.9,  # stagflation
            8: 0.8,  # boom
            9: 0.9   # bust
        }
        
        return significance_map.get(regime_id, 0.5)
    
    def _calculate_economic_stability(self, regime_predictions: np.ndarray, index: int) -> float:
        """Calculate economic stability for a specific index."""
        return self._calculate_regime_stability(regime_predictions, index)
    
    def _calculate_economic_volatility(self, regime_id: int, prepared_data: Dict[str, Any]) -> float:
        """Calculate economic volatility for a regime."""
        # Base volatility on regime type
        volatility_map = {
            0: 0.3,  # normal
            1: 0.4,  # expansion
            2: 0.8,  # recession
            3: 0.6,  # recovery
            4: 0.2,  # stagnation
            5: 0.7,  # inflation
            6: 0.7,  # deflation
            7: 0.9,  # stagflation
            8: 0.5,  # boom
            9: 0.9   # bust
        }
        
        return volatility_map.get(regime_id, 0.5)
    
    def _classify_financial_regime(self, regime_id: int) -> str:
        """Classify financial regime based on regime ID."""
        financial_regimes = {
            0: "normal",
            1: "risk_on",
            2: "risk_off",
            3: "liquidity_abundant",
            4: "liquidity_crunch",
            5: "credit_easy",
            6: "credit_tight",
            7: "flight_to_quality",
            8: "speculation",
            9: "crisis"
        }
        
        return financial_regimes.get(regime_id, "unknown")
    
    def _calculate_financial_significance(self, regime_id: int, prepared_data: Dict[str, Any]) -> float:
        """Calculate financial significance for a regime."""
        # Base significance on regime type
        significance_map = {
            0: 0.5,  # normal
            1: 0.7,  # risk_on
            2: 0.8,  # risk_off
            3: 0.6,  # liquidity_abundant
            4: 0.9,  # liquidity_crunch
            5: 0.6,  # credit_easy
            6: 0.8,  # credit_tight
            7: 0.8,  # flight_to_quality
            8: 0.7,  # speculation
            9: 0.9   # crisis
        }
        
        return significance_map.get(regime_id, 0.5)
    
    def _calculate_financial_stability(self, regime_predictions: np.ndarray, index: int) -> float:
        """Calculate financial stability for a specific index."""
        return self._calculate_regime_stability(regime_predictions, index)
    
    def _calculate_financial_volatility(self, regime_id: int, prepared_data: Dict[str, Any]) -> float:
        """Calculate financial volatility for a regime."""
        # Base volatility on regime type
        volatility_map = {
            0: 0.3,  # normal
            1: 0.5,  # risk_on
            2: 0.7,  # risk_off
            3: 0.4,  # liquidity_abundant
            4: 0.8,  # liquidity_crunch
            5: 0.4,  # credit_easy
            6: 0.7,  # credit_tight
            7: 0.6,  # flight_to_quality
            8: 0.8,  # speculation
            9: 0.9   # crisis
        }
        
        return volatility_map.get(regime_id, 0.5)
    
    def get_tag_summary(self) -> Dict[str, Any]:
        """Get summary of tagging results."""
        if not self.tag_history:
            return {"error": "No tagging performed yet"}
        
        latest_tags = self.tag_history[-1]
        
        return {
            "n_tagged_samples": latest_tags['n_samples'],
            "n_regimes": latest_tags['n_regimes'],
            "validation_results": latest_tags['validation_results'],
            "tag_history_length": len(self.tag_history),
            "latest_timestamp": latest_tags['timestamp']
        }
    
    def get_tagged_data(self, index: int = -1) -> Optional[pd.DataFrame]:
        """Get tagged data from history."""
        if not self.tag_history:
            return None
        
        return self.tag_history[index]['tagged_data']