"""
ML Volatility Burst Step (1.5-3% Range Specialist)

This step implements a volatility burst prediction model specifically
designed to identify volatility expansion opportunities that enable
1.5-3% price moves, following de Prado's framework for diverse base models.

Key Features:
- Volatility regime detection and prediction
- GARCH-based volatility forecasting
- Volume-volatility relationship analysis
- Volatility burst pattern recognition
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import logging
from datetime import datetime, timedelta
from arch import arch_model

from src.utils.tprint import tprint_info, tprint_warning, tprint_error, tprint_success
from src.utils.versioned_artifacts import VersionedArtifactStore
from src.training.steps.market_analysis.ml_risk_regime_step import MLRiskRegimeStepHMM
from src.training.steps.market_analysis.specialist_diagnostics_mixin import SpecialistDiagnosticsMixin

logger = logging.getLogger(__name__)

class MLVolatilityBurstStep(SpecialistDiagnosticsMixin, MLRiskRegimeStepHMM):
    """
    Volatility Burst Specialist for 1.5-3% Range Trading.
    
    This specialist focuses on identifying volatility expansion patterns
    that are predictive of 1.5-3% price moves.
    """
    
    def __init__(self, step_name: str = "ml_volatility_burst_step"):
        """Initialize the volatility burst step."""
        super().__init__(step_name)
        self.logger = logger.getChild("MLVolatilityBurstStep")
        
    def _compute_volatility_burst_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute volatility burst features for 1.5-3% range prediction."""
        features = pd.DataFrame(index=df.index)
        
        # Basic volatility measures
        returns = df['close'].pct_change()
        
        for window in [10, 20, 50]:
            # Rolling volatility
            features[f'volatility_{window}'] = returns.rolling(window).std()
            
            # Volatility of volatility (VoV)
            features[f'vol_of_vol_{window}'] = features[f'volatility_{window}'].rolling(window).std()
            
            # Volatility momentum
            features[f'vol_momentum_{window}'] = features[f'volatility_{window}'].pct_change(5)
            
            # Volatility persistence
            features[f'vol_persistence_{window}'] = features[f'volatility_{window}'].rolling(window).apply(
                lambda x: x.autocorr() if len(x.dropna()) > 1 else 0, raw=False
            )
        
        # GARCH-based volatility forecasts
        features['garch_forecast'] = self._compute_garch_forecast(returns)
        features['garch_momentum'] = features['garch_forecast'].pct_change(5)
        
        # Volume-volatility relationship
        if 'volume' in df.columns:
            volume_change = df['volume'].pct_change()
            
            for window in [20, 50]:
                # Volume-volatility correlation
                features[f'vol_vol_corr_{window}'] = (
                    returns.rolling(window).corr(volume_change)
                )
                
                # Volume surprise (unexpected volume)
                vol_avg = df['volume'].rolling(window).mean()
                features[f'volume_surprise_{window}'] = (df['volume'] - vol_avg) / vol_avg
                
                # Volatility response to volume
                features[f'vol_response_vol_{window}'] = (
                    features[f'volatility_{window}'] / (features[f'volume_surprise_{window}'].abs() + 1e-8)
                )
        
        # Volatility burst indicators
        features['vol_burst_score'] = self._compute_volatility_burst_score(features)
        features['vol_regime_transition'] = self._detect_volatility_regime_transitions(features)
        
        return features.fillna(0)
    
    def _compute_garch_forecast(self, returns: pd.Series, window: int = 100) -> pd.Series:
        """Compute GARCH(1,1) volatility forecasts."""
        forecasts = pd.Series(index=returns.index, dtype=float)

        if len(returns) > 5000:
            fast_forecast = returns.rolling(window).std()
            return fast_forecast.fillna(method='ffill').fillna(0.0)
        
        # Use rolling window for GARCH estimation
        for i in range(window, len(returns)):
            try:
                # Get window data
                window_returns = returns.iloc[i-window:i].dropna()
                
                if len(window_returns) < 50:
                    forecasts.iloc[i] = returns.iloc[i-window:i].std()
                    continue
                
                # Fit GARCH model
                am = arch_model(window_returns * 100, vol='Garch', p=1, q=1, rescale=False)
                res = am.fit(disp='off')
                
                # Forecast next period volatility
                forecast = res.forecast(horizon=1)
                forecasts.iloc[i] = np.sqrt(forecast.variance.iloc[-1, -1]) / 100
                
            except Exception:
                # Fallback to historical volatility
                forecasts.iloc[i] = returns.iloc[i-window:i].std()
        
        return forecasts.fillna(method='ffill')
    
    def _compute_volatility_burst_score(self, features: pd.DataFrame) -> pd.Series:
        """Compute composite volatility burst score."""
        burst_signals = []
        
        # Current volatility relative to recent average
        if 'volatility_20' in features.columns:
            vol_ratio = features['volatility_20'] / features['volatility_20'].rolling(50).mean()
            burst_signals.append(vol_ratio)
        
        # Volatility momentum
        if 'vol_momentum_20' in features.columns:
            burst_signals.append(features['vol_momentum_20'].abs())
        
        # Volatility of volatility
        if 'vol_of_vol_20' in features.columns:
            burst_signals.append(features['vol_of_vol_20'])
        
        # GARCH forecast momentum
        if 'garch_momentum' in features.columns:
            burst_signals.append(features['garch_momentum'].abs())
        
        if burst_signals:
            # Normalize and combine
            combined = pd.concat(burst_signals, axis=1)
            normalized = (combined - combined.mean()) / (combined.std() + 1e-8)
            return normalized.mean(axis=1)
        else:
            return pd.Series(0, index=features.index)
    
    def _detect_volatility_regime_transitions(self, features: pd.DataFrame) -> pd.Series:
        """Detect volatility regime transitions."""
        if 'volatility_20' not in features.columns:
            return pd.Series(0, index=features.index)
        
        # Simple regime detection based on volatility percentiles
        vol = features['volatility_20']
        vol_rolling = vol.rolling(100)
        
        # Define regimes based on percentiles
        low_threshold = vol_rolling.quantile(0.25)
        high_threshold = vol_rolling.quantile(0.75)
        
        # Detect transitions
        current_regime = pd.Series(0, index=vol.index)
        current_regime[vol > high_threshold] = 2  # High volatility
        current_regime[vol < low_threshold] = 0   # Low volatility
        current_regime[(vol >= low_threshold) & (vol <= high_threshold)] = 1  # Normal
        
        # Transition signals
        transitions = current_regime.diff().abs()
        
        return transitions.fillna(0)
    
    def _create_volatility_burst_labels(self, df: pd.DataFrame, lookahead_bars: int = 48) -> pd.Series:
        """Create binary labels for volatility bursts leading to 1.5-3% moves."""
        returns = df['close'].pct_change()
        volatility = returns.rolling(20).std()
        
        # Look for volatility expansion followed by 1.5-3% price move
        vol_expansion = volatility.pct_change(lookahead_bars//2) > 0.5  # 50% vol expansion
        price_move = (returns.shift(-lookahead_bars).abs() >= 0.015) & (returns.shift(-lookahead_bars).abs() <= 0.03)
        
        labels = (vol_expansion & price_move).astype(int)
        
        return labels.shift(-lookahead_bars)
    
    def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute volatility burst specialist training."""
        try:
            tprint_info("🚀 Starting ML Volatility Burst Step (1.5-3% Range Specialist)")
            
            # Extract configuration
            symbol = config.get('symbol', 'ETHUSDT')
            exchange = config.get('exchange', 'binance')
            timeframe = config.get('timeframe', '15m')
            direction = config.get('direction', 'long')
            
            # Load market data
            df = self._load_market_data(symbol, exchange, timeframe)
            if df is None or len(df) < 1000:
                tprint_error("❌ Insufficient data for volatility burst training")
                return {"success": False, "error": "Insufficient data"}
            
            # Compute features
            tprint_info("📊 Computing volatility burst features...")
            features = self._compute_volatility_burst_features(df)
            
            # Create labels for volatility bursts
            tprint_info("🎯 Creating volatility burst labels...")
            labels = self._create_volatility_burst_labels(df)
            
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
            
            # Train volatility burst model
            tprint_info("🤖 Training volatility burst model...")
            model, metrics = self._train_volatility_model(features, labels)
            
            # Generate predictions
            predictions = model.predict(features)
            probabilities = model.predict_proba(features)[:, 1]
            
            # Create output DataFrame
            output_df = features.copy()
            output_df['volatility_burst_prediction'] = predictions
            output_df['volatility_burst_probability'] = probabilities
            output_df['volatility_burst_label'] = labels
            
            # Save artifacts
            artifact_name = f"ml_volatility_burst_{timeframe}"
            self.artifact_router.save(
                artifact_name=artifact_name,
                data=output_df,
                metadata={
                    "symbol": symbol,
                    "exchange": exchange,
                    "timeframe": timeframe,
                    "direction": direction,
                    "model_type": "volatility_burst",
                    "target_range": "1.5-3%",
                    "metrics": metrics,
                    "n_samples": len(output_df),
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            
            # Save model
            model_name = f"volatility_burst_model_{timeframe}"
            self.artifact_router.save(
                artifact_name=model_name,
                data=model,
                metadata={
                    "symbol": symbol,
                    "exchange": exchange,
                    "timeframe": timeframe,
                    "direction": direction,
                    "model_type": "volatility_burst",
                    "target_range": "1.5-3%",
                    "features": list(features.columns),
                    "metrics": metrics
                }
            )
            
            tprint_success(f"✅ Volatility burst specialist completed: {metrics.get('auc', 0):.3f} AUC")
            
            return {
                "success": True,
                "metrics": metrics,
                "n_samples": len(output_df),
                "features": list(features.columns),
                "artifact_name": artifact_name,
                "model_name": model_name
            }
            
        except Exception as e:
            self.logger.exception(f"❌ Volatility burst step failed: {e}")
            return {"success": False, "error": str(e)}
    
    def run_diagnostics(self, symbol: str = 'ETHUSDT', exchange: str = 'binance', 
                       timeframe: str = '15m', direction: str = 'long') -> Dict[str, Any]:
        """Run independent diagnostics for volatility burst specialist."""
        return self.run_self_diagnostics(symbol, exchange, timeframe, direction)
    
    def _train_volatility_model(self, features: pd.DataFrame, labels: pd.Series) -> Tuple[lgb.LGBMClassifier, Dict[str, float]]:
        """Train LightGBM model for volatility burst prediction."""
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
