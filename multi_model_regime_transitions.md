# Multi-Model Approach for Regime Transitions

## Strategy: Use Multiple Models During Uncertain Regimes

Instead of reducing position size, we'll run multiple regime-specific models simultaneously when regime confidence is low, then blend their predictions.

## Implementation in Strategist/Analyst

```python
class RegimeAwareStrategist:
    def __init__(self, config):
        self.regime_confidence_threshold = 0.75
        self.models_by_regime = {}  # Loaded trained models
        self.ensemble_weights = {
            'high_confidence': {
                'primary': 1.0,
                'others': 0.0
            },
            'medium_confidence': {
                'primary': 0.6,
                'adjacent': 0.3,
                'opposite': 0.1
            },
            'low_confidence': {
                'all': 0.33  # Equal weight to all regimes
            }
        }
        
    async def generate_predictions(self, data, current_regime, regime_confidence):
        """Generate predictions using single or multiple models based on confidence."""
        
        if regime_confidence >= self.regime_confidence_threshold:
            # High confidence: Use single regime model
            return await self._single_model_prediction(data, current_regime)
        else:
            # Low confidence: Use ensemble of models
            return await self._ensemble_prediction(data, current_regime, regime_confidence)
    
    async def _ensemble_prediction(self, data, primary_regime, confidence):
        """Run multiple models and blend predictions."""
        predictions = {}
        
        # Run all regime models
        for regime, model in self.models_by_regime.items():
            predictions[regime] = await model.predict(data)
        
        # Determine weights based on confidence and regime relationships
        weights = self._calculate_ensemble_weights(primary_regime, confidence)
        
        # Blend predictions
        final_prediction = np.zeros_like(predictions[primary_regime])
        for regime, weight in weights.items():
            if weight > 0:
                final_prediction += weight * predictions[regime]
        
        # Return blended prediction with metadata
        return {
            'prediction': final_prediction,
            'ensemble_used': True,
            'weights': weights,
            'individual_predictions': predictions,
            'confidence': confidence
        }
    
    def _calculate_ensemble_weights(self, primary_regime, confidence):
        """Calculate weights for each model based on regime relationships."""
        
        # Define regime adjacency
        regime_adjacency = {
            'bull': {'adjacent': 'sideways', 'opposite': 'bear'},
            'sideways': {'adjacent': ['bull', 'bear'], 'opposite': None},
            'bear': {'adjacent': 'sideways', 'opposite': 'bull'}
        }
        
        if confidence > 0.6:  # Medium confidence
            weights = {
                primary_regime: 0.6,
                regime_adjacency[primary_regime]['adjacent']: 0.3,
                regime_adjacency[primary_regime]['opposite']: 0.1
            }
        else:  # Low confidence
            # Equal weight to all
            weights = {
                'bull': 0.33,
                'sideways': 0.34,
                'bear': 0.33
            }
            
        return weights
```

## Analyst Implementation

```python
class MultiModelAnalyst:
    def __init__(self, config):
        self.analysts_by_regime = {
            'bull': BullMarketAnalyst(config),
            'bear': BearMarketAnalyst(config),
            'sideways': SidewaysMarketAnalyst(config)
        }
        self.transition_detector = RegimeTransitionDetector(config)
        
    async def analyze(self, market_data, regime_info):
        """Analyze using appropriate analyst(s) based on regime state."""
        
        # Check if we're in a transition
        transition_state = self.transition_detector.detect(
            market_data, 
            regime_info
        )
        
        if transition_state['in_transition']:
            # Use multiple analysts
            return await self._multi_analyst_analysis(
                market_data,
                transition_state
            )
        else:
            # Use single analyst
            return await self._single_analyst_analysis(
                market_data,
                regime_info['current_regime']
            )
    
    async def _multi_analyst_analysis(self, market_data, transition_state):
        """Run multiple analysts during transitions."""
        analyses = {}
        
        # Get list of relevant analysts
        relevant_regimes = transition_state['possible_regimes']
        
        # Run each analyst
        for regime in relevant_regimes:
            analyst = self.analysts_by_regime[regime]
            analyses[regime] = await analyst.analyze(market_data)
        
        # Combine analyses based on transition probabilities
        combined_analysis = self._combine_analyses(
            analyses,
            transition_state['regime_probabilities']
        )
        
        return {
            'analysis': combined_analysis,
            'multi_model': True,
            'regimes_considered': relevant_regimes,
            'transition_state': transition_state
        }
```

## Regime Transition Detector

```python
class RegimeTransitionDetector:
    def __init__(self, config):
        self.lookback_periods = 10  # Look at last 10 1h candles
        self.transition_indicators = [
            'regime_probability_variance',
            'regime_switch_frequency',
            'hmm_state_uncertainty'
        ]
        
    def detect(self, market_data, regime_info):
        """Detect if we're in a regime transition."""
        
        # Calculate transition metrics
        metrics = {
            'regime_stability': self._calculate_regime_stability(regime_info),
            'probability_concentration': self._calculate_probability_concentration(regime_info),
            'recent_switches': self._count_recent_switches(regime_info)
        }
        
        # Determine if in transition
        in_transition = (
            metrics['regime_stability'] < 0.7 or
            metrics['probability_concentration'] < 0.6 or
            metrics['recent_switches'] > 2
        )
        
        # Get possible regimes during transition
        if in_transition:
            possible_regimes = self._get_possible_regimes(regime_info)
            regime_probabilities = self._get_regime_probabilities(regime_info)
        else:
            possible_regimes = [regime_info['current_regime']]
            regime_probabilities = {regime_info['current_regime']: 1.0}
            
        return {
            'in_transition': in_transition,
            'metrics': metrics,
            'possible_regimes': possible_regimes,
            'regime_probabilities': regime_probabilities
        }
```