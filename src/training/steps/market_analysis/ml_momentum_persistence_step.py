"""
ML Momentum Persistence Step (1.5-3% Range Specialist)

This step implements a momentum persistence model specifically designed
to identify and predict trend continuation opportunities in the 1.5-3%
return range, following de Prado's framework for diverse base models.

Key Features:
- Medium-term momentum persistence detection
- Volatility-adjusted momentum scoring
- Regime-aware momentum characteristics
- Cross-timeframe momentum alignment
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import logging
from datetime import datetime, timedelta

from src.utils.tprint import tprint_info, tprint_warning, tprint_error, tprint_success
from src.utils.versioned_artifacts import VersionedArtifactStore
from src.training.steps.market_analysis.ml_risk_regime_step import MLRiskRegimeStepHMM
from src.training.steps.market_analysis.specialist_diagnostics_mixin_enhanced import SpecialistDiagnosticsMixinEnhanced

logger = logging.getLogger(__name__)

class MLMomentumPersistenceStep(SpecialistDiagnosticsMixinEnhanced, MLRiskRegimeStepHMM):
    """
    Momentum Persistence Specialist for 1.5-3% Range Trading.
    
    This specialist focuses on identifying medium-term momentum
    persistence patterns that are predictive of 1.5-3% price moves.
    """
    
    def __init__(self, step_name: str = "ml_momentum_persistence_step"):
        """Initialize the momentum persistence step."""
        super().__init__(step_name)
        self.logger = logger.getChild("MLMomentumPersistenceStep")
        
    def _compute_momentum_persistence_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute momentum persistence features for 1.5-3% range prediction."""
        features = pd.DataFrame(index=df.index)
        
        # Medium-term momentum indicators (20-50 periods)
        for window in [20, 35, 50]:
            # Price momentum
            features[f'momentum_{window}'] = df['close'].pct_change(window)
            
            # Momentum persistence (autocorrelation)
            features[f'momentum_persistence_{window}'] = features[f'momentum_{window}'].rolling(window).apply(
                lambda x: x.autocorr() if len(x.dropna()) > 1 else 0, raw=False
            )
            
            # Momentum acceleration
            features[f'momentum_accel_{window}'] = features[f'momentum_{window}'].diff(5)
            
            # Volatility-adjusted momentum
            vol = df['close'].pct_change().rolling(window).std()
            features[f'momentum_vol_adj_{window}'] = features[f'momentum_{window}'] / (vol + 1e-8)
        
        # Cross-timeframe momentum alignment
        if 'high' in df.columns and 'low' in df.columns:
            # Intraday momentum patterns
            features['intraday_momentum'] = (df['close'] - df['open']) / df['open']
            features['intraday_persistence'] = features['intraday_momentum'].rolling(10).apply(
                lambda x: x.autocorr() if len(x.dropna()) > 1 else 0, raw=False
            )
        
        # Volume-weighted momentum
        if 'volume' in df.columns:
            vw_momentum = (df['close'] * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
            features['vw_momentum'] = vw_momentum.pct_change(20)
            features['vw_momentum_persistence'] = features['vw_momentum'].rolling(20).apply(
                lambda x: x.autocorr() if len(x.dropna()) > 1 else 0, raw=False
            )
        
        # Regime-aware momentum features
        features['momentum_regime_score'] = self._compute_momentum_regime_score(features)
        
        return features.fillna(0)
    
    def _compute_momentum_regime_score(self, features: pd.DataFrame) -> pd.Series:
        """Compute momentum regime score based on multiple indicators."""
        # Combine multiple momentum signals with different weights
        momentum_signals = []
        
        # Medium-term trend strength
        if 'momentum_35' in features.columns:
            momentum_signals.append(features['momentum_35'].abs())
        
        # Persistence strength
        if 'momentum_persistence_35' in features.columns:
            momentum_signals.append(features['momentum_persistence_35'].abs())
        
        # Acceleration magnitude
        if 'momentum_accel_35' in features.columns:
            momentum_signals.append(features['momentum_accel_35'].abs())
        
        if momentum_signals:
            # Normalize and combine
            combined = pd.concat(momentum_signals, axis=1)
            normalized = (combined - combined.mean()) / (combined.std() + 1e-8)
            return normalized.mean(axis=1)
        else:
            return pd.Series(0, index=features.index)
    
    def _create_1_5_3_percent_labels(self, df: pd.DataFrame, lookahead_bars: int = 48) -> pd.Series:
        """Create binary labels for 1.5-3% price moves within lookahead period."""
        returns = df['close'].pct_change(lookahead_bars)
        
        # Label 1 if return is between 1.5% and 3% (positive)
        labels = ((returns >= 0.015) & (returns <= 0.03)).astype(int)
        
        return labels.shift(-lookahead_bars)
    
    def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute momentum persistence specialist training."""
        try:
            tprint_info("🚀 Starting ML Momentum Persistence Step (1.5-3% Range Specialist)")
            
            # Extract configuration
            symbol = config.get('symbol', 'ETHUSDT')
            exchange = config.get('exchange', 'binance')
            timeframe = config.get('timeframe', '15m')
            direction = config.get('direction', 'long')
            
            # Load market data
            df = self._load_market_data(symbol, exchange, timeframe)
            if df is None or len(df) < 1000:
                tprint_error("❌ Insufficient data for momentum persistence training")
                return {"success": False, "error": "Insufficient data"}
            
            # Compute features
            tprint_info("📊 Computing momentum persistence features...")
            features = self._compute_momentum_persistence_features(df)
            
            # Create labels for 1.5-3% range
            tprint_info("🎯 Creating 1.5-3% range labels...")
            labels = self._create_1_5_3_percent_labels(df)
            
            # Align features and labels
            valid_idx = features.index.intersection(labels.index)
            features = features.loc[valid_idx]
            labels = labels.loc[valid_idx]
            
            # Remove NaN values
            mask = ~(features.isna().any(axis=1) | labels.isna())
            features = features[mask]
            labels = labels[mask]
            
            if len(features) < 500:
                tprint_error("❌ Insufficient valid samples after cleaning")
                return {"success": False, "error": "Insufficient valid samples"}
            
            # Train momentum persistence model
            tprint_info("🤖 Training momentum persistence model...")
            model, metrics = self._train_momentum_model(features, labels)
            
            # Generate predictions
            predictions = model.predict(features)
            probabilities = model.predict_proba(features)[:, 1]
            
            # Create output DataFrame
            output_df = features.copy()
            output_df['momentum_persistence_prediction'] = predictions
            output_df['momentum_persistence_probability'] = probabilities
            output_df['momentum_persistence_label'] = labels
            
            # Save artifacts
            artifact_name = f"ml_momentum_persistence_{timeframe}"
            self.artifact_router.save(
                artifact_name=artifact_name,
                data=output_df,
                metadata={
                    "symbol": symbol,
                    "exchange": exchange,
                    "timeframe": timeframe,
                    "direction": direction,
                    "model_type": "momentum_persistence",
                    "target_range": "1.5-3%",
                    "metrics": metrics,
                    "n_samples": len(output_df),
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            
            # Save model
            model_name = f"momentum_persistence_model_{timeframe}"
            self.artifact_router.save(
                artifact_name=model_name,
                data=model,
                metadata={
                    "symbol": symbol,
                    "exchange": exchange,
                    "timeframe": timeframe,
                    "direction": direction,
                    "model_type": "momentum_persistence",
                    "target_range": "1.5-3%",
                    "features": list(features.columns),
                    "metrics": metrics
                }
            )
            
            tprint_success(f"✅ Momentum persistence specialist completed: {metrics.get('auc', 0):.3f} AUC")
            
            return {
                "success": True,
                "metrics": metrics,
                "n_samples": len(output_df),
                "features": list(features.columns),
                "artifact_name": artifact_name,
                "model_name": model_name
            }
            
        except Exception as e:
            self.logger.exception(f"❌ Momentum persistence step failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _train_momentum_model(self, features: pd.DataFrame, labels: pd.Series) -> Tuple[lgb.LGBMClassifier, Dict[str, float]]:
        """Train LightGBM model for momentum persistence prediction."""
        # Time series split for validation
        n_splits = 5
        split_idx = len(features) // (n_splits + 1)
        
        auc_scores = []
        models = []
        
        for i in range(n_splits):
            train_start = i * split_idx
            train_end = (i + 2) * split_idx
            val_end = (i + 3) * split_idx
            
            if val_end > len(features):
                break
            
            X_train = features.iloc[train_start:train_end]
            y_train = labels.iloc[train_start:train_end]
            X_val = features.iloc[train_end:val_end]
            y_val = labels.iloc[train_end:val_end]
            
            # Train model
            model = lgb.LGBMClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                num_leaves=31,
                random_state=42 + i,
                verbose=-1
            )
            
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(10)])
            
            # Evaluate
            from sklearn.metrics import roc_auc_score
            val_pred = model.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, val_pred)
            auc_scores.append(auc)
            models.append(model)
        
        # Select best model
        best_idx = np.argmax(auc_scores)
        best_model = models[best_idx]
        
        metrics = {
            "auc": np.mean(auc_scores),
            "auc_std": np.std(auc_scores),
            "n_splits": len(auc_scores)
        }
        
        return best_model, metrics
    
    def _load_market_data(self, symbol: str, exchange: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Load market data for training."""
        try:
            market_data, _source = self.load_market_data_or_fail(
                {
                    "symbol": symbol,
                    "exchange": exchange,
                    "timeframe": timeframe,
                },
                pipeline_state={},
                allow_config_override=True,
            )
            return market_data
        except Exception as e:
            self.logger.error(f"Failed to load market data: {e}")
            return None
