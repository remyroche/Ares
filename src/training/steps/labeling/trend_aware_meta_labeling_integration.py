"""
Trend-Aware Meta-Labeling Integration for Training and Trading

This module provides the bridge between:
1. Feature Generation (labeling stage) - generates trend/zigzag/confluence features
2. Training Stage - trains ML models on these features
3. Trading Stage - uses trained models for live signal generation

Key Components:
- TrendAwareFeatureGenerator: Generates features for training data
- TrendAwareModelTrainer: Trains models that understand confluence/conflict
- TrendAwareTradingSignalGenerator: Generates live trading signals
- TrendAwareMetaLabelingStep: Pipeline step for integration

Usage Flow:
1. Training: OHLCV Data → TrendAwareFeatureGenerator → Features + Labels → Train Model
2. Trading: Live OHLCV → TrendAwareFeatureGenerator → Features → Trained Model → Signal
"""

import logging
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import json
import pickle

import numpy as np
import pandas as pd

# ML libraries
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import RobustScaler, StandardScaler
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, classification_report
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("scikit-learn not available")

try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

# Internal imports
from src.training.steps.base_step import BaseStep
from src.utils.logger import system_logger, get_logger
from src.utils.tprint import tprint

# Trend-aware meta-labeling imports
from src.feature_generation.utils.step06_labeling_components.trend_aware_meta_labeling import (
    TrendAwareMetaLabeler,
    TrendAwareTripleBarrierConfig,
    MultiTimeframeConfig,
    TrendConfluence,
    apply_trend_aware_meta_labeling,
    apply_multi_timeframe_labeling,
)

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class TrendAwareTrainingConfig:
    """Configuration for trend-aware model training."""
    
    # Feature generation settings
    include_bollinger_signals: bool = True
    include_obv_divergence: bool = True
    include_zigzag_features: bool = True
    include_multi_timeframe: bool = True
    
    # Multi-timeframe settings
    base_timeframe: str = "15min"
    higher_timeframes: List[Tuple[str, str]] = field(default_factory=lambda: [
        ("1H", "1H"),
        ("4H", "4H"),
    ])
    
    # Triple barrier settings
    profit_take_multiplier: float = 0.004
    stop_loss_multiplier: float = 0.003
    time_barrier_minutes: int = 30
    use_trend_adjusted_barriers: bool = True
    
    # Model training settings
    model_type: str = "lightgbm"  # "lightgbm", "xgboost", "random_forest"
    n_splits: int = 5  # TimeSeriesSplit folds
    use_sample_weights: bool = True  # Weight by signal_weight from trend analysis
    
    # Feature selection
    drop_categorical_features: bool = False
    feature_importance_threshold: float = 0.01
    
    # Model hyperparameters
    lgbm_params: Dict[str, Any] = field(default_factory=lambda: {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'n_estimators': 200,
        'early_stopping_rounds': 20,
    })
    
    xgb_params: Dict[str, Any] = field(default_factory=lambda: {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': 6,
        'learning_rate': 0.05,
        'n_estimators': 200,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'early_stopping_rounds': 20,
    })


@dataclass  
class TrendAwareTradingConfig:
    """Configuration for live trading signal generation."""
    
    # Signal thresholds
    probability_threshold: float = 0.55
    min_confluence_score: float = 0.3
    require_htf_alignment: bool = False
    
    # Position sizing based on confluence
    base_position_size: float = 0.1
    max_position_size: float = 0.3
    confluence_position_multiplier: float = 1.5
    conflict_position_multiplier: float = 0.5
    
    # Risk management
    max_positions: int = 3
    min_signal_weight: float = 0.5
    
    # Feature requirements
    required_features: List[str] = field(default_factory=lambda: [
        'trend_base',
        'mtf_confluence_score',
        'bb_squeeze_strength',
    ])


# =============================================================================
# FEATURE GENERATOR FOR TRAINING/TRADING
# =============================================================================

class TrendAwareFeatureGenerator:
    """
    Generates trend-aware features for both training and live trading.
    
    This class wraps the TrendAwareMetaLabeler to provide a consistent
    interface for feature generation across training and trading contexts.
    """
    
    def __init__(
        self,
        config: Optional[TrendAwareTrainingConfig] = None,
        labeler_config: Optional[TrendAwareTripleBarrierConfig] = None,
        mtf_config: Optional[MultiTimeframeConfig] = None
    ):
        """Initialize the feature generator.
        
        Args:
            config: Training configuration
            labeler_config: Trend-aware labeler configuration
            mtf_config: Multi-timeframe configuration
        """
        self.config = config or TrendAwareTrainingConfig()
        self.logger = get_logger('TrendAwareFeatureGenerator')
        
        # Create labeler configuration
        self.labeler_config = labeler_config or TrendAwareTripleBarrierConfig(
            base_profit_take_multiplier=self.config.profit_take_multiplier,
            base_stop_loss_multiplier=self.config.stop_loss_multiplier,
            time_barrier_minutes=self.config.time_barrier_minutes,
        )
        
        # Create multi-timeframe configuration
        self.mtf_config = mtf_config or MultiTimeframeConfig(
            base_timeframe=self.config.base_timeframe,
            higher_timeframes=self.config.higher_timeframes,
        )
        
        # Initialize labeler
        self.labeler = TrendAwareMetaLabeler(self.labeler_config)
        
        # Track feature names for consistency
        self.feature_names: List[str] = []
        self.categorical_features: List[str] = []
        
        self.logger.info("🔧 TrendAwareFeatureGenerator initialized")
    
    def generate_training_features(
        self,
        data: pd.DataFrame,
        include_labels: bool = True
    ) -> Tuple[pd.DataFrame, Optional[pd.Series], Optional[pd.Series]]:
        """Generate features for model training.
        
        Args:
            data: OHLCV DataFrame with DatetimeIndex
            include_labels: Whether to generate triple barrier labels
            
        Returns:
            Tuple of (features_df, labels_series, weights_series)
        """
        self.logger.info(f"📊 Generating training features for {len(data)} samples")
        
        # Generate all features
        result = self.labeler.generate_trend_aware_features(
            data,
            include_labels=include_labels,
            include_multi_timeframe=self.config.include_multi_timeframe,
            mtf_config=self.mtf_config
        )
        
        # Extract labels and weights
        labels = None
        weights = None
        
        if 'label' in result.columns:
            labels = result['label'].copy()
            result = result.drop(columns=['label'])
        
        if 'signal_weight' in result.columns:
            weights = result['signal_weight'].copy()
            # Don't drop - keep as feature too
        
        # Identify feature columns
        self._identify_features(result)
        
        # Handle categorical features
        if self.config.drop_categorical_features:
            cat_cols = [c for c in result.columns if result[c].dtype == 'category' or c.endswith('_cat')]
            result = result.drop(columns=cat_cols, errors='ignore')
        
        # Drop non-feature columns
        non_feature_cols = ['open', 'high', 'low', 'close', 'volume', 
                           'potential_profit_pct', 'barrier_hit_type',
                           'pt_multiplier_used', 'sl_multiplier_used']
        result = result.drop(columns=[c for c in non_feature_cols if c in result.columns])
        
        # Handle infinities and NaNs
        result = result.replace([np.inf, -np.inf], np.nan)
        result = result.fillna(0)
        
        self.logger.info(f"✅ Generated {len(result.columns)} features")
        
        return result, labels, weights
    
    def generate_trading_features(
        self,
        data: pd.DataFrame
    ) -> pd.DataFrame:
        """Generate features for live trading (no labels).
        
        Args:
            data: Recent OHLCV DataFrame with DatetimeIndex
            
        Returns:
            DataFrame with features for the most recent bars
        """
        self.logger.info(f"📊 Generating trading features for {len(data)} samples")
        
        # Generate features without labels
        result = self.labeler.generate_trend_aware_features(
            data,
            include_labels=False,
            include_multi_timeframe=self.config.include_multi_timeframe,
            mtf_config=self.mtf_config
        )
        
        # Apply same processing as training
        if self.config.drop_categorical_features:
            cat_cols = [c for c in result.columns if result[c].dtype == 'category' or c.endswith('_cat')]
            result = result.drop(columns=cat_cols, errors='ignore')
        
        non_feature_cols = ['open', 'high', 'low', 'close', 'volume']
        result = result.drop(columns=[c for c in non_feature_cols if c in result.columns])
        
        result = result.replace([np.inf, -np.inf], np.nan)
        result = result.fillna(0)
        
        # Ensure consistent feature order
        if self.feature_names:
            missing = [f for f in self.feature_names if f not in result.columns]
            for col in missing:
                result[col] = 0
            result = result[self.feature_names]
        
        return result
    
    def _identify_features(self, df: pd.DataFrame) -> None:
        """Identify and store feature names."""
        self.feature_names = [c for c in df.columns if df[c].dtype in ['float64', 'int64', 'int8', 'float32']]
        self.categorical_features = [c for c in df.columns if df[c].dtype == 'category' or c.endswith('_cat')]
        
        self.logger.info(f"   Numeric features: {len(self.feature_names)}")
        self.logger.info(f"   Categorical features: {len(self.categorical_features)}")


# =============================================================================
# MODEL TRAINER
# =============================================================================

class TrendAwareModelTrainer:
    """
    Trains ML models on trend-aware features.
    
    Supports:
    - LightGBM, XGBoost, RandomForest
    - Time-series cross-validation
    - Sample weighting based on signal_weight
    - Feature importance analysis
    """
    
    def __init__(self, config: Optional[TrendAwareTrainingConfig] = None):
        """Initialize the trainer.
        
        Args:
            config: Training configuration
        """
        self.config = config or TrendAwareTrainingConfig()
        self.logger = get_logger('TrendAwareModelTrainer')
        
        self.model = None
        self.scaler = None
        self.feature_names: List[str] = []
        self.feature_importance: Dict[str, float] = {}
        self.cv_scores: List[Dict[str, float]] = []
        
        self.logger.info(f"🎓 TrendAwareModelTrainer initialized (model: {self.config.model_type})")
    
    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weights: Optional[pd.Series] = None,
        validation_data: Optional[Tuple[pd.DataFrame, pd.Series]] = None
    ) -> Dict[str, Any]:
        """Train the model with time-series cross-validation.
        
        Args:
            X: Feature DataFrame
            y: Label Series
            sample_weights: Optional sample weights (e.g., signal_weight)
            validation_data: Optional (X_val, y_val) tuple
            
        Returns:
            Dictionary with training results and metrics
        """
        self.logger.info(f"🚀 Training {self.config.model_type} model on {len(X)} samples")
        
        self.feature_names = list(X.columns)
        
        # Scale features
        self.scaler = RobustScaler()
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X),
            index=X.index,
            columns=X.columns
        )
        
        # Convert labels to binary (1 for long, 0 for short/hold)
        y_binary = (y == 1).astype(int)
        
        # Prepare sample weights
        if sample_weights is not None and self.config.use_sample_weights:
            weights = sample_weights.values
        else:
            weights = None
        
        # Time-series cross-validation
        tscv = TimeSeriesSplit(n_splits=self.config.n_splits)
        self.cv_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_scaled)):
            X_train, X_val = X_scaled.iloc[train_idx], X_scaled.iloc[val_idx]
            y_train, y_val = y_binary.iloc[train_idx], y_binary.iloc[val_idx]
            
            if weights is not None:
                w_train = weights[train_idx]
            else:
                w_train = None
            
            # Train fold model
            fold_model = self._create_model()
            fold_model = self._fit_model(fold_model, X_train, y_train, X_val, y_val, w_train)
            
            # Evaluate fold
            y_pred_proba = self._predict_proba(fold_model, X_val)
            y_pred = (y_pred_proba > 0.5).astype(int)
            
            fold_scores = {
                'fold': fold,
                'accuracy': accuracy_score(y_val, y_pred),
                'precision': precision_score(y_val, y_pred, zero_division=0),
                'recall': recall_score(y_val, y_pred, zero_division=0),
                'f1': f1_score(y_val, y_pred, zero_division=0),
                'roc_auc': roc_auc_score(y_val, y_pred_proba) if len(np.unique(y_val)) > 1 else 0.5,
            }
            self.cv_scores.append(fold_scores)
            
            self.logger.info(f"   Fold {fold}: AUC={fold_scores['roc_auc']:.4f}, F1={fold_scores['f1']:.4f}")
        
        # Train final model on all data
        self.model = self._create_model()
        self.model = self._fit_model(self.model, X_scaled, y_binary, None, None, weights)
        
        # Extract feature importance
        self._extract_feature_importance()
        
        # Compute average CV scores
        avg_scores = {
            metric: np.mean([s[metric] for s in self.cv_scores])
            for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        }
        
        results = {
            'model_type': self.config.model_type,
            'n_samples': len(X),
            'n_features': len(self.feature_names),
            'cv_scores': self.cv_scores,
            'avg_scores': avg_scores,
            'feature_importance': self.feature_importance,
            'top_features': self._get_top_features(20),
        }
        
        self.logger.info(f"✅ Training complete: Avg AUC={avg_scores['roc_auc']:.4f}, F1={avg_scores['f1']:.4f}")
        
        return results
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class labels.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Array of predicted labels
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        X_scaled = pd.DataFrame(
            self.scaler.transform(X),
            index=X.index,
            columns=X.columns
        )
        
        proba = self._predict_proba(self.model, X_scaled)
        return (proba > 0.5).astype(int)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Array of probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        X_scaled = pd.DataFrame(
            self.scaler.transform(X),
            index=X.index,
            columns=X.columns
        )
        
        return self._predict_proba(self.model, X_scaled)
    
    def _create_model(self):
        """Create model based on configuration."""
        if self.config.model_type == "lightgbm" and LGBM_AVAILABLE:
            return lgb.LGBMClassifier(**self.config.lgbm_params)
        elif self.config.model_type == "xgboost" and XGB_AVAILABLE:
            return xgb.XGBClassifier(**self.config.xgb_params)
        elif SKLEARN_AVAILABLE:
            return RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        else:
            raise ValueError("No ML library available")
    
    def _fit_model(self, model, X_train, y_train, X_val, y_val, weights):
        """Fit model with appropriate parameters."""
        if self.config.model_type == "lightgbm" and LGBM_AVAILABLE:
            eval_set = [(X_val, y_val)] if X_val is not None else None
            model.fit(
                X_train, y_train,
                sample_weight=weights,
                eval_set=eval_set,
            )
        elif self.config.model_type == "xgboost" and XGB_AVAILABLE:
            eval_set = [(X_val, y_val)] if X_val is not None else None
            model.fit(
                X_train, y_train,
                sample_weight=weights,
                eval_set=eval_set,
                verbose=False,
            )
        else:
            model.fit(X_train, y_train, sample_weight=weights)
        
        return model
    
    def _predict_proba(self, model, X) -> np.ndarray:
        """Get probability predictions."""
        proba = model.predict_proba(X)
        if proba.ndim == 2:
            return proba[:, 1]
        return proba
    
    def _extract_feature_importance(self) -> None:
        """Extract feature importance from model."""
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            self.feature_importance = dict(zip(self.feature_names, importances))
    
    def _get_top_features(self, n: int = 20) -> List[Tuple[str, float]]:
        """Get top N most important features."""
        sorted_features = sorted(
            self.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_features[:n]
    
    def save(self, path: Union[str, Path]) -> None:
        """Save model and configuration.
        
        Args:
            path: Path to save directory
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = path / "model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'feature_importance': self.feature_importance,
                'config': self.config,
            }, f)
        
        # Save metadata
        metadata = {
            'model_type': self.config.model_type,
            'n_features': len(self.feature_names),
            'feature_names': self.feature_names,
            'cv_scores': self.cv_scores,
            'created_at': datetime.now().isoformat(),
        }
        with open(path / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"💾 Model saved to {path}")
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'TrendAwareModelTrainer':
        """Load model from disk.
        
        Args:
            path: Path to saved model directory
            
        Returns:
            Loaded TrendAwareModelTrainer instance
        """
        path = Path(path)
        
        with open(path / "model.pkl", 'rb') as f:
            data = pickle.load(f)
        
        trainer = cls(config=data['config'])
        trainer.model = data['model']
        trainer.scaler = data['scaler']
        trainer.feature_names = data['feature_names']
        trainer.feature_importance = data['feature_importance']
        
        return trainer


# =============================================================================
# TRADING SIGNAL GENERATOR
# =============================================================================

class TrendAwareTradingSignalGenerator:
    """
    Generates trading signals using trained model and trend-aware features.
    
    This class is used during live trading to:
    1. Generate features from recent OHLCV data
    2. Get model predictions
    3. Apply confluence/conflict rules for signal filtering
    4. Calculate position sizing based on trend alignment
    """
    
    def __init__(
        self,
        feature_generator: TrendAwareFeatureGenerator,
        model_trainer: TrendAwareModelTrainer,
        trading_config: Optional[TrendAwareTradingConfig] = None
    ):
        """Initialize the signal generator.
        
        Args:
            feature_generator: Trained feature generator
            model_trainer: Trained model
            trading_config: Trading configuration
        """
        self.feature_generator = feature_generator
        self.model = model_trainer
        self.config = trading_config or TrendAwareTradingConfig()
        self.logger = get_logger('TrendAwareTradingSignalGenerator')
        
        self.logger.info("🎯 TrendAwareTradingSignalGenerator initialized")
    
    def generate_signal(
        self,
        data: pd.DataFrame,
        return_details: bool = False
    ) -> Union[Dict[str, Any], Tuple[Dict[str, Any], pd.DataFrame]]:
        """Generate trading signal from recent market data.
        
        Args:
            data: Recent OHLCV DataFrame (need enough history for indicators)
            return_details: Whether to return full feature DataFrame
            
        Returns:
            Signal dictionary with trading decision, or tuple with details
        """
        # Generate features
        features = self.feature_generator.generate_trading_features(data)
        
        if len(features) == 0:
            return self._no_signal("No features generated")
        
        # Get latest features
        latest = features.iloc[-1:]
        
        # Get model prediction
        proba = self.model.predict_proba(latest)[0]
        
        # Extract trend information for filtering
        trend_info = self._extract_trend_info(latest)
        
        # Apply signal filtering rules
        signal = self._apply_signal_rules(proba, trend_info)
        
        # Calculate position size
        position_size = self._calculate_position_size(signal, trend_info)
        
        result = {
            'timestamp': data.index[-1],
            'signal': signal['direction'],  # 'long', 'short', 'none'
            'probability': proba,
            'confidence': signal['confidence'],
            'position_size': position_size,
            'trend_info': trend_info,
            'reason': signal['reason'],
        }
        
        if return_details:
            return result, features
        return result
    
    def _extract_trend_info(self, features: pd.DataFrame) -> Dict[str, Any]:
        """Extract trend information from features."""
        row = features.iloc[0]
        
        info = {
            'trend_base': row.get('trend_base', 0),
            'confluence_score': row.get('mtf_confluence_score', 0),
            'alignment_ratio': row.get('mtf_alignment_ratio', 0),
            'conflict': row.get('mtf_conflict', 0),
            'signal_weight': row.get('signal_weight', 1.0),
            'bb_squeeze': row.get('bb_is_squeeze', 0),
            'bb_squeeze_strength': row.get('bb_squeeze_strength', 0),
        }
        
        # Add higher timeframe trends if available
        for tf in ['1H', '4H', '1D']:
            col = f'trend_{tf}'
            if col in row:
                info[col] = row[col]
        
        return info
    
    def _apply_signal_rules(
        self,
        proba: float,
        trend_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply signal filtering rules based on probability and trend."""
        
        # Check probability threshold
        if proba < self.config.probability_threshold and proba > (1 - self.config.probability_threshold):
            return {
                'direction': 'none',
                'confidence': 0.0,
                'reason': f'Probability {proba:.3f} below threshold {self.config.probability_threshold}'
            }
        
        # Determine base direction from probability
        if proba >= self.config.probability_threshold:
            base_direction = 'long'
            confidence = proba
        else:
            base_direction = 'short'
            confidence = 1 - proba
        
        # Check confluence requirements
        confluence_score = trend_info.get('confluence_score', 0)
        if abs(confluence_score) < self.config.min_confluence_score:
            # Low confluence - reduce confidence but don't reject
            confidence *= 0.7
        
        # Check for HTF alignment if required
        if self.config.require_htf_alignment:
            conflict = trend_info.get('conflict', 0)
            if conflict:
                return {
                    'direction': 'none',
                    'confidence': 0.0,
                    'reason': 'Base trend conflicts with higher timeframes'
                }
        
        # Check signal weight threshold
        signal_weight = trend_info.get('signal_weight', 1.0)
        if signal_weight < self.config.min_signal_weight:
            confidence *= signal_weight / self.config.min_signal_weight
        
        # Validate direction alignment with trend
        trend_base = trend_info.get('trend_base', 0)
        if base_direction == 'long' and trend_base == -1:
            # Long signal in downtrend - reduce confidence
            confidence *= 0.6
        elif base_direction == 'short' and trend_base == 1:
            # Short signal in uptrend - reduce confidence
            confidence *= 0.6
        
        return {
            'direction': base_direction,
            'confidence': confidence,
            'reason': f'Signal aligned with trend (confluence: {confluence_score:.2f})'
        }
    
    def _calculate_position_size(
        self,
        signal: Dict[str, Any],
        trend_info: Dict[str, Any]
    ) -> float:
        """Calculate position size based on confluence and signal strength."""
        
        if signal['direction'] == 'none':
            return 0.0
        
        base_size = self.config.base_position_size
        
        # Adjust by confluence
        confluence_score = abs(trend_info.get('confluence_score', 0))
        if confluence_score > 0.5:
            base_size *= self.config.confluence_position_multiplier
        elif trend_info.get('conflict', 0):
            base_size *= self.config.conflict_position_multiplier
        
        # Adjust by confidence
        base_size *= signal['confidence']
        
        # Apply limits
        return min(base_size, self.config.max_position_size)
    
    def _no_signal(self, reason: str) -> Dict[str, Any]:
        """Return no signal result."""
        return {
            'timestamp': None,
            'signal': 'none',
            'probability': 0.5,
            'confidence': 0.0,
            'position_size': 0.0,
            'trend_info': {},
            'reason': reason,
        }


# =============================================================================
# PIPELINE STEP FOR INTEGRATION
# =============================================================================

class TrendAwareMetaLabelingStep(BaseStep):
    """
    Pipeline step for trend-aware meta-labeling integration.
    
    This step can be used in the training pipeline to:
    1. Generate trend-aware features from OHLCV data
    2. Create labels using trend-aware triple barrier
    3. Train models with confluence/conflict awareness
    4. Save trained models for trading use
    """
    
    def __init__(self, step_name: str = "trend_aware_meta_labeling"):
        """Initialize the step."""
        super().__init__(step_name)
        self.logger = system_logger.getChild('TrendAwareMetaLabeling')
    
    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the trend-aware meta-labeling step.
        
        Args:
            config: Configuration dictionary containing:
                - symbol: Trading symbol
                - timeframe: Base timeframe
                - data: OHLCV DataFrame (or path to load)
                - training_config: TrendAwareTrainingConfig params
                - save_model_path: Path to save trained model
                
        Returns:
            Dictionary with results and metrics
        """
        tprint("🚀 Starting Trend-Aware Meta-Labeling Step", "INFO")
        
        try:
            # Extract configuration
            symbol = config.get('symbol', 'UNKNOWN')
            timeframe = config.get('timeframe', '15m')
            data = config.get('data')
            
            if data is None:
                raise ValueError("No data provided in config")
            
            # Create training configuration
            training_config = TrendAwareTrainingConfig(
                base_timeframe=timeframe,
                **config.get('training_config', {})
            )
            
            # Initialize components
            feature_generator = TrendAwareFeatureGenerator(config=training_config)
            model_trainer = TrendAwareModelTrainer(config=training_config)
            
            # Generate features and labels
            tprint("📊 Generating trend-aware features...", "INFO")
            X, y, weights = feature_generator.generate_training_features(
                data,
                include_labels=True
            )
            
            if y is None or len(y) == 0:
                raise ValueError("No labels generated")
            
            tprint(f"   Generated {len(X.columns)} features for {len(X)} samples", "INFO")
            tprint(f"   Label distribution: LONG={(y==1).sum()}, SHORT={(y==-1).sum()}", "INFO")
            
            # Train model
            tprint("🎓 Training model...", "INFO")
            training_results = model_trainer.train(X, y, sample_weights=weights)
            
            # Log results
            avg_auc = training_results['avg_scores']['roc_auc']
            avg_f1 = training_results['avg_scores']['f1']
            tprint(f"✅ Training complete: AUC={avg_auc:.4f}, F1={avg_f1:.4f}", "SUCCESS")
            
            # Save model if path provided
            save_path = config.get('save_model_path')
            if save_path:
                model_trainer.save(save_path)
                tprint(f"💾 Model saved to {save_path}", "INFO")
            
            # Compile results
            results = {
                'symbol': symbol,
                'timeframe': timeframe,
                'n_samples': len(X),
                'n_features': len(X.columns),
                'feature_names': list(X.columns),
                'training_results': training_results,
                'top_features': training_results['top_features'],
                'status': 'success',
            }
            
            return results
            
        except Exception as e:
            self.logger.exception(f"Error in trend-aware meta-labeling: {e}")
            tprint(f"❌ Error: {e}", "ERROR")
            return {
                'status': 'error',
                'error': str(e),
            }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_trend_aware_training_pipeline(
    data: pd.DataFrame,
    config: Optional[TrendAwareTrainingConfig] = None,
    save_path: Optional[Union[str, Path]] = None
) -> Tuple[TrendAwareFeatureGenerator, TrendAwareModelTrainer, Dict[str, Any]]:
    """
    Create and run a complete training pipeline.
    
    Args:
        data: OHLCV DataFrame with DatetimeIndex
        config: Optional training configuration
        save_path: Optional path to save trained model
        
    Returns:
        Tuple of (feature_generator, model_trainer, results)
    """
    config = config or TrendAwareTrainingConfig()
    
    # Initialize components
    feature_generator = TrendAwareFeatureGenerator(config=config)
    model_trainer = TrendAwareModelTrainer(config=config)
    
    # Generate features
    X, y, weights = feature_generator.generate_training_features(data, include_labels=True)
    
    # Train model
    results = model_trainer.train(X, y, sample_weights=weights)
    
    # Save if requested
    if save_path:
        model_trainer.save(save_path)
    
    return feature_generator, model_trainer, results


def create_trading_signal_generator(
    model_path: Union[str, Path],
    training_config: Optional[TrendAwareTrainingConfig] = None,
    trading_config: Optional[TrendAwareTradingConfig] = None
) -> TrendAwareTradingSignalGenerator:
    """
    Create a trading signal generator from saved model.
    
    Args:
        model_path: Path to saved model
        training_config: Training configuration (for feature generator)
        trading_config: Trading configuration
        
    Returns:
        Configured TrendAwareTradingSignalGenerator
    """
    training_config = training_config or TrendAwareTrainingConfig()
    
    feature_generator = TrendAwareFeatureGenerator(config=training_config)
    model_trainer = TrendAwareModelTrainer.load(model_path)
    
    return TrendAwareTradingSignalGenerator(
        feature_generator=feature_generator,
        model_trainer=model_trainer,
        trading_config=trading_config
    )


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    tprint("=" * 70)
    tprint("Trend-Aware Meta-Labeling Integration Example")
    tprint("=" * 70)
    
    # Create sample data
    np.random.seed(42)
    n = 2000
    dates = pd.date_range('2024-01-01', periods=n, freq='15min')
    
    trend = np.cumsum(np.random.randn(n) * 0.001)
    price = 100 * np.exp(trend)
    
    data = pd.DataFrame({
        'open': price * (1 + np.random.uniform(-0.002, 0.002, n)),
        'high': price * (1 + np.random.uniform(0, 0.005, n)),
        'low': price * (1 + np.random.uniform(-0.005, 0, n)),
        'close': price,
        'volume': np.random.uniform(1000, 10000, n)
    }, index=dates)
    
    data['high'] = data[['open', 'high', 'close']].max(axis=1)
    data['low'] = data[['open', 'low', 'close']].min(axis=1)
    
    # Example 1: Full training pipeline
    tprint("\n" + "=" * 50)
    tprint("Example 1: Training Pipeline")
    tprint("=" * 50)
    
    config = TrendAwareTrainingConfig(
        include_multi_timeframe=True,
        model_type="random_forest",  # Use RF for demo (no LightGBM needed)
    )
    
    feature_gen, trainer, results = create_trend_aware_training_pipeline(
        data,
        config=config,
    )
    
    tprint(f"\nTraining Results:")
    tprint(f"   Samples: {results['n_samples']}")
    tprint(f"   Features: {results['n_features']}")
    tprint(f"   Avg AUC: {results['avg_scores']['roc_auc']:.4f}")
    tprint(f"   Avg F1: {results['avg_scores']['f1']:.4f}")
    
    tprint(f"\nTop 10 Features:")
    for feat, imp in results['top_features'][:10]:
        tprint(f"   {feat}: {imp:.4f}")
    
    # Example 2: Generate trading signal
    tprint("\n" + "=" * 50)
    tprint("Example 2: Trading Signal Generation")
    tprint("=" * 50)
    
    trading_config = TrendAwareTradingConfig(
        probability_threshold=0.55,
        min_confluence_score=0.2,
    )
    
    signal_gen = TrendAwareTradingSignalGenerator(
        feature_generator=feature_gen,
        model_trainer=trainer,
        trading_config=trading_config
    )
    
    # Get signal for latest data
    signal = signal_gen.generate_signal(data)
    
    tprint(f"\nLatest Signal:")
    tprint(f"   Direction: {signal['signal']}")
    tprint(f"   Probability: {signal['probability']:.4f}")
    tprint(f"   Confidence: {signal['confidence']:.4f}")
    tprint(f"   Position Size: {signal['position_size']:.4f}")
    tprint(f"   Reason: {signal['reason']}")
    
    if signal['trend_info']:
        tprint(f"\nTrend Info:")
        tprint(f"   Base Trend: {signal['trend_info'].get('trend_base', 'N/A')}")
        tprint(f"   Confluence: {signal['trend_info'].get('confluence_score', 'N/A'):.3f}")
        tprint(f"   Conflict: {signal['trend_info'].get('conflict', 'N/A')}")
    
    tprint("\n" + "=" * 70)
    tprint("✅ Example completed successfully!")
    tprint("=" * 70)
