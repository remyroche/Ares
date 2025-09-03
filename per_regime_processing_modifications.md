# Per-Regime Processing Modifications for Steps 5+

## Overview
Ensure all processing from Step 5 onwards is regime-aware, with separate models, features, and performance tracking per regime.

## Step 5: Labeling Modifications

### Enhanced Regime-Aware Labeling
```python
class RegimeAwareLabelingStep:
    def __init__(self, config):
        self.config = config
        self.regime_specific_params = {
            'bull': {
                'profit_target': 0.02,  # 2% in bull markets
                'stop_loss': 0.01,      # Tighter stops in trends
                'time_barrier': 60      # 60 bars
            },
            'bear': {
                'profit_target': 0.015,  # More conservative
                'stop_loss': 0.015,      # Symmetric in bear
                'time_barrier': 30       # Faster exits
            },
            'sideways': {
                'profit_target': 0.01,   # Smaller targets
                'stop_loss': 0.01,       # Tight stops
                'time_barrier': 20       # Quick trades
            }
        }
    
    async def execute_per_regime_labeling(self, data, regime_labels):
        labeled_data_by_regime = {}
        
        for regime in ['bull', 'bear', 'sideways']:
            # Get regime-specific data
            regime_mask = regime_labels == regime
            regime_data = data[regime_mask].copy()
            
            if len(regime_data) < self.config['min_samples_per_regime']:
                self.logger.warning(f"Insufficient data for {regime} regime: {len(regime_data)} samples")
                continue
            
            # Apply regime-specific parameters
            params = self.regime_specific_params[regime]
            
            # Generate labels with regime-specific barriers
            labels = await self._generate_triple_barrier_labels(
                regime_data,
                profit_target=params['profit_target'],
                stop_loss=params['stop_loss'],
                time_barrier=params['time_barrier']
            )
            
            # Add regime confidence as a feature
            labels['regime_confidence'] = self._calculate_regime_confidence(
                regime_data, 
                regime_labels[regime_mask]
            )
            
            labeled_data_by_regime[regime] = labels
            
        return labeled_data_by_regime
```

## Step 6: Feature Engineering Modifications

### Regime-Specific Feature Engineering
```python
class RegimeAwareFeatureEngineering:
    def __init__(self, config):
        self.regime_features = {
            'bull': [
                'momentum_features',      # More important in trends
                'breakout_features',
                'trend_strength_features'
            ],
            'bear': [
                'support_resistance_features',  # Key in downtrends
                'volatility_features',
                'volume_divergence_features'
            ],
            'sideways': [
                'mean_reversion_features',     # Central to ranging
                'bollinger_band_features',
                'oscillator_features'
            ]
        }
    
    async def engineer_features_per_regime(self, data_by_regime):
        features_by_regime = {}
        
        for regime, regime_data in data_by_regime.items():
            # Get regime-specific feature set
            feature_types = self.regime_features[regime]
            
            # Engineer features specific to this regime
            features = await self._engineer_regime_features(
                regime_data, 
                feature_types,
                regime
            )
            
            # Add regime-specific indicators
            features = self._add_regime_indicators(features, regime)
            
            # Feature selection based on regime
            selected_features = await self._select_features_for_regime(
                features, 
                regime
            )
            
            features_by_regime[regime] = selected_features
            
            # Log feature importance by regime
            self._log_feature_importance(selected_features, regime)
            
        return features_by_regime
    
    def _add_regime_indicators(self, features, regime):
        if regime == 'bull':
            # Bull-specific indicators
            features['rsi_oversold_bounce'] = (features['rsi'] < 30).astype(int)
            features['momentum_acceleration'] = features['momentum'].diff()
            
        elif regime == 'bear':
            # Bear-specific indicators
            features['rsi_overbought_reversal'] = (features['rsi'] > 70).astype(int)
            features['support_break'] = self._calculate_support_breaks(features)
            
        elif regime == 'sideways':
            # Sideways-specific indicators
            features['bollinger_squeeze'] = self._calculate_bb_squeeze(features)
            features['range_position'] = self._calculate_range_position(features)
            
        return features
```

## Step 7: Enhanced Matrix Operations Modifications

### Regime-Aware Matrix Optimization
```python
class RegimeAwareMatrixOperations:
    def __init__(self, config):
        self.regime_matrix_configs = {
            'bull': {
                'correlation_threshold': 0.7,  # Allow more correlation in trends
                'pca_components': 0.95,        # Preserve more variance
                'regularization': 'low'        # Less regularization
            },
            'bear': {
                'correlation_threshold': 0.6,  # Stricter in bear markets
                'pca_components': 0.90,        # More dimension reduction
                'regularization': 'high'       # More regularization
            },
            'sideways': {
                'correlation_threshold': 0.5,  # Very strict
                'pca_components': 0.85,        # Aggressive reduction
                'regularization': 'medium'     # Balanced
            }
        }
    
    async def optimize_matrices_per_regime(self, features_by_regime):
        optimized_by_regime = {}
        
        for regime, features in features_by_regime.items():
            config = self.regime_matrix_configs[regime]
            
            # Remove highly correlated features per regime
            decorrelated = self._remove_correlations(
                features, 
                threshold=config['correlation_threshold']
            )
            
            # Apply PCA with regime-specific variance
            if config['pca_components'] < 1.0:
                reduced = self._apply_pca(
                    decorrelated,
                    n_components=config['pca_components']
                )
            else:
                reduced = decorrelated
            
            # Apply regime-specific regularization
            regularized = self._apply_regularization(
                reduced,
                level=config['regularization']
            )
            
            optimized_by_regime[regime] = regularized
            
        return optimized_by_regime
```

## Performance Tracking Modifications

### Regime-Specific Performance Tracking
```python
class RegimePerformanceTracker:
    def __init__(self):
        self.metrics_by_regime = {
            'bull': [],
            'bear': [],
            'sideways': []
        }
        
    def track_performance(self, predictions, actuals, regime_labels):
        for regime in ['bull', 'bear', 'sideways']:
            regime_mask = regime_labels == regime
            
            if regime_mask.sum() == 0:
                continue
                
            regime_predictions = predictions[regime_mask]
            regime_actuals = actuals[regime_mask]
            
            metrics = {
                'sharpe_ratio': self._calculate_sharpe(regime_predictions, regime_actuals),
                'win_rate': self._calculate_win_rate(regime_predictions, regime_actuals),
                'profit_factor': self._calculate_profit_factor(regime_predictions, regime_actuals),
                'max_drawdown': self._calculate_max_drawdown(regime_predictions, regime_actuals),
                'sample_count': regime_mask.sum(),
                'regime': regime
            }
            
            self.metrics_by_regime[regime].append(metrics)
            
        return self._generate_regime_report()
    
    def _generate_regime_report(self):
        report = {
            'summary': {},
            'recommendations': []
        }
        
        for regime, metrics_list in self.metrics_by_regime.items():
            if not metrics_list:
                continue
                
            # Average metrics for regime
            avg_metrics = {
                key: np.mean([m[key] for m in metrics_list if key in m])
                for key in ['sharpe_ratio', 'win_rate', 'profit_factor']
            }
            
            report['summary'][regime] = avg_metrics
            
            # Generate recommendations
            if avg_metrics['sharpe_ratio'] < 0.5:
                report['recommendations'].append(
                    f"Consider reducing position size in {regime} regime (low Sharpe)"
                )
            
            if avg_metrics['win_rate'] < 0.45:
                report['recommendations'].append(
                    f"Review feature engineering for {regime} regime (low win rate)"
                )
                
        return report
```

## Position Sizing Based on Regime Confidence

### Dynamic Position Sizing
```python
class RegimeBasedPositionSizer:
    def __init__(self, config):
        self.base_position_size = config['base_position_size']
        self.regime_multipliers = {
            'bull': 1.2,      # Larger positions in clear trends
            'bear': 0.8,      # Smaller in bear markets
            'sideways': 0.6   # Smallest in ranging markets
        }
    
    def calculate_position_size(self, regime, regime_confidence, model_confidence):
        # Base size adjusted by regime
        regime_adjusted = self.base_position_size * self.regime_multipliers[regime]
        
        # Further adjust by confidence levels
        confidence_multiplier = (regime_confidence * 0.5 + model_confidence * 0.5)
        
        # Additional safety during regime transitions
        if regime_confidence < 0.7:
            confidence_multiplier *= 0.5  # Half size during transitions
            
        final_size = regime_adjusted * confidence_multiplier
        
        # Apply limits
        return np.clip(final_size, 0.1, 1.0)
```

## Integration Example
```python
# Modified main pipeline
async def run_regime_aware_pipeline(data, config):
    # Step 3: Get regime labels
    regime_labels = await hmm_regime_discovery(data)
    
    # Step 5: Per-regime labeling
    labels_by_regime = await regime_aware_labeling(data, regime_labels)
    
    # Step 6: Per-regime features
    features_by_regime = await regime_aware_feature_engineering(labels_by_regime)
    
    # Step 7: Per-regime optimization
    optimized_by_regime = await regime_aware_matrix_ops(features_by_regime)
    
    # Train separate models per regime
    models_by_regime = {}
    for regime, features in optimized_by_regime.items():
        model = await train_model(features, labels_by_regime[regime])
        models_by_regime[regime] = model
    
    # Track performance per regime
    performance_tracker = RegimePerformanceTracker()
    for regime, model in models_by_regime.items():
        performance = await evaluate_model(model, test_data[regime])
        performance_tracker.track_performance(
            performance['predictions'],
            performance['actuals'],
            regime
        )
    
    return {
        'models': models_by_regime,
        'performance': performance_tracker.generate_report()
    }
```