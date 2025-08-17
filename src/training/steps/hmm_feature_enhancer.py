# src/training/steps/hmm_feature_enhancer.py

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from src.utils.logger import system_logger
from src.utils.decorators import with_tracing_span, guard_dataframe_nulls


class HMMFeatureEnhancer:
    """Enhances HMM features with additional derived features for Step 5 compatibility."""
    
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.logger = system_logger.getChild("HMMFeatureEnhancer")
    
    @with_tracing_span("HMMFeatureEnhancer.enhance_hmm_features")
    @guard_dataframe_nulls(mode="warn", arg_index=0)
    def enhance_hmm_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Enhance HMM features with additional derived features.
        
        Args:
            features_df: DataFrame with existing HMM features
            
        Returns:
            Enhanced DataFrame with additional HMM features
        """
        try:
            self.logger.info("🔄 Enhancing HMM features with derived features...")
            
            enhanced_df = features_df.copy()
            
            # 1. Regime Transition Features
            enhanced_df = self._add_regime_transition_features(enhanced_df)
            
            # 2. Regime Stability Features
            enhanced_df = self._add_regime_stability_features(enhanced_df)
            
            # 3. Regime Interaction Features
            enhanced_df = self._add_regime_interaction_features(enhanced_df)
            
            # 4. Missing Technical Indicators (from Step 5 requirements)
            enhanced_df = self._add_missing_technical_indicators(enhanced_df)
            
            # 5. Regime-Enhanced Features
            enhanced_df = self._add_regime_enhanced_features(enhanced_df)
            
            self.logger.info(f"✅ Enhanced HMM features: {enhanced_df.shape[1]} total features")
            return enhanced_df
            
        except Exception as e:
            self.logger.error(f"🚨 HMM feature enhancement failed: {e}")
            return features_df
    
    def _add_regime_transition_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add regime transition and persistence features."""
        try:
            # Regime persistence (how long we've been in current regime)
            if 'composite_cluster_id' in df.columns:
                df['regime_persistence'] = self._calculate_regime_persistence(df['composite_cluster_id'])
                df['regime_transition_count'] = self._calculate_regime_transitions(df['composite_cluster_id'])
                df['regime_volatility'] = self._calculate_regime_volatility(df['composite_cluster_id'])
            
            # State transition probabilities
            state_columns = [col for col in df.columns if col.endswith('_p_state_')]
            if state_columns:
                # Max probability state
                df['dominant_state_prob'] = df[state_columns].max(axis=1)
                df['state_uncertainty'] = 1 - df['dominant_state_prob']
                
                # State entropy (measure of uncertainty)
                df['state_entropy'] = self._calculate_state_entropy(df[state_columns])
                
                # State stability (how much probabilities change)
                df['state_stability'] = self._calculate_state_stability(df[state_columns])
            
            self.logger.info("✅ Added regime transition features")
            return df
            
        except Exception as e:
            self.logger.warning(f"⚠️ Regime transition features failed: {e}")
            return df
    
    def _add_regime_stability_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add regime stability and consistency features."""
        try:
            # Regime consistency over different timeframes
            if 'composite_cluster_id' in df.columns:
                # Rolling regime consistency
                df['regime_consistency_5'] = df['composite_cluster_id'].rolling(5).apply(
                    lambda x: len(x.unique()) == 1, raw=False
                ).astype(float)
                
                df['regime_consistency_10'] = df['composite_cluster_id'].rolling(10).apply(
                    lambda x: len(x.unique()) == 1, raw=False
                ).astype(float)
                
                df['regime_consistency_20'] = df['composite_cluster_id'].rolling(20).apply(
                    lambda x: len(x.unique()) == 1, raw=False
                ).astype(float)
            
            # State probability stability
            state_columns = [col for col in df.columns if col.endswith('_p_state_')]
            if state_columns:
                # Rolling standard deviation of dominant state probability
                df['state_prob_volatility'] = df['dominant_state_prob'].rolling(10).std()
                
                # State probability trend
                df['state_prob_trend'] = df['dominant_state_prob'].rolling(5).mean() - df['dominant_state_prob'].rolling(20).mean()
            
            self.logger.info("✅ Added regime stability features")
            return df
            
        except Exception as e:
            self.logger.warning(f"⚠️ Regime stability features failed: {e}")
            return df
    
    def _add_regime_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add regime interaction and correlation features."""
        try:
            # Regime-momentum interactions
            if 'composite_cluster_id' in df.columns and 'momentum_strength' in df.columns:
                df['regime_momentum_interaction'] = df['composite_cluster_id'] * df['momentum_strength']
                df['regime_momentum_divergence'] = df['momentum_strength'] - df.groupby('composite_cluster_id')['momentum_strength'].transform('mean')
            
            # Regime-volatility interactions
            if 'composite_cluster_id' in df.columns and 'volume_volatility' in df.columns:
                df['regime_volatility_interaction'] = df['composite_cluster_id'] * df['volume_volatility']
                df['regime_volatility_divergence'] = df['volume_volatility'] - df.groupby('composite_cluster_id')['volume_volatility'].transform('mean')
            
            # Regime-liquidity interactions
            if 'composite_cluster_id' in df.columns and 'liquidity_score' in df.columns:
                df['regime_liquidity_interaction'] = df['composite_cluster_id'] * df['liquidity_score']
                df['regime_liquidity_divergence'] = df['liquidity_score'] - df.groupby('composite_cluster_id')['liquidity_score'].transform('mean')
            
            # Cross-regime correlations
            state_columns = [col for col in df.columns if col.endswith('_p_state_')]
            if len(state_columns) >= 2:
                # Create interaction features between different state probabilities
                for i, col1 in enumerate(state_columns[:3]):  # Limit to first 3 to avoid explosion
                    for col2 in state_columns[i+1:4]:
                        interaction_name = f"{col1.replace('_p_state_', '')}_{col2.replace('_p_state_', '')}_interaction"
                        df[interaction_name] = df[col1] * df[col2]
            
            self.logger.info("✅ Added regime interaction features")
            return df
            
        except Exception as e:
            self.logger.warning(f"⚠️ Regime interaction features failed: {e}")
            return df
    
    def _add_missing_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add missing technical indicators from Step 5 requirements."""
        try:
            # Check if we have OHLCV data to calculate missing indicators
            ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
            available_ohlcv = [col for col in ohlcv_cols if col in df.columns]
            
            if len(available_ohlcv) >= 4:  # Need at least OHLC
                # RSI (if not present)
                if 'rsi' not in df.columns and 'close' in df.columns:
                    df['rsi'] = self._calculate_rsi(df['close'])
                
                # MACD (if not present)
                if 'macd' not in df.columns and 'close' in df.columns:
                    df['macd'] = self._calculate_macd(df['close'])
                
                # Bollinger Bands position (if not present)
                if 'bb_position' not in df.columns and 'close' in df.columns:
                    df['bb_position'] = self._calculate_bb_position(df['close'])
                
                # ADX (if not present)
                if 'adx' not in df.columns and all(col in df.columns for col in ['high', 'low', 'close']):
                    df['adx'] = self._calculate_adx(df['high'], df['low'], df['close'])
                
                # CCI (if not present)
                if 'cci' not in df.columns and all(col in df.columns for col in ['high', 'low', 'close']):
                    df['cci'] = self._calculate_cci(df['high'], df['low'], df['close'])
                
                # MFI (if not present)
                if 'mfi' not in df.columns and all(col in df.columns for col in ['high', 'low', 'close', 'volume']):
                    df['mfi'] = self._calculate_mfi(df['high'], df['low'], df['close'], df['volume'])
                
                # ROC (if not present)
                if 'roc' not in df.columns and 'close' in df.columns:
                    df['roc'] = self._calculate_roc(df['close'])
                
                # SMA and EMA (if not present)
                if 'sma' not in df.columns and 'close' in df.columns:
                    df['sma'] = df['close'].rolling(20).mean()
                
                if 'ema' not in df.columns and 'close' in df.columns:
                    df['ema'] = df['close'].ewm(span=20).mean()
                
                # ATR (if not present)
                if 'atr' not in df.columns and all(col in df.columns for col in ['high', 'low', 'close']):
                    df['atr'] = self._calculate_atr(df['high'], df['low'], df['close'])
            
            self.logger.info("✅ Added missing technical indicators")
            return df
            
        except Exception as e:
            self.logger.warning(f"⚠️ Missing technical indicators failed: {e}")
            return df
    
    def _add_regime_enhanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add regime-enhanced versions of existing features."""
        try:
            # Regime-enhanced momentum
            if 'momentum_strength' in df.columns and 'composite_cluster_id' in df.columns:
                df['regime_enhanced_momentum'] = df['momentum_strength'] * (1 + df['dominant_state_prob'] * 0.5)
            
            # Regime-enhanced volatility
            if 'volume_volatility' in df.columns and 'composite_cluster_id' in df.columns:
                df['regime_enhanced_volatility'] = df['volume_volatility'] * (1 + df['state_uncertainty'] * 0.3)
            
            # Regime-enhanced liquidity
            if 'liquidity_score' in df.columns and 'composite_cluster_id' in df.columns:
                df['regime_enhanced_liquidity'] = df['liquidity_score'] * (1 + df['regime_consistency_10'] * 0.2)
            
            # Regime stress indicator
            if 'state_entropy' in df.columns and 'volume_volatility' in df.columns:
                df['regime_stress'] = df['state_entropy'] * df['volume_volatility']
            
            # Regime momentum divergence
            if 'momentum_strength' in df.columns and 'regime_momentum_divergence' in df.columns:
                df['regime_momentum_extreme'] = np.abs(df['regime_momentum_divergence']) > df['regime_momentum_divergence'].rolling(20).std() * 2
            
            self.logger.info("✅ Added regime-enhanced features")
            return df
            
        except Exception as e:
            self.logger.warning(f"⚠️ Regime-enhanced features failed: {e}")
            return df
    
    # Helper methods for calculations
    def _calculate_regime_persistence(self, regime_series: pd.Series) -> pd.Series:
        """Calculate how long we've been in the current regime."""
        persistence = pd.Series(index=regime_series.index, dtype=float)
        current_regime = None
        current_count = 0
        
        for i, regime in enumerate(regime_series):
            if regime == current_regime:
                current_count += 1
            else:
                current_regime = regime
                current_count = 1
            persistence.iloc[i] = current_count
        
        return persistence
    
    def _calculate_regime_transitions(self, regime_series: pd.Series) -> pd.Series:
        """Calculate number of regime transitions in rolling window."""
        transitions = (regime_series != regime_series.shift(1)).astype(int)
        return transitions.rolling(20).sum()
    
    def _calculate_regime_volatility(self, regime_series: pd.Series) -> pd.Series:
        """Calculate regime volatility (frequency of changes)."""
        changes = (regime_series != regime_series.shift(1)).astype(int)
        return changes.rolling(10).std()
    
    def _calculate_state_entropy(self, state_probs: pd.DataFrame) -> pd.Series:
        """Calculate entropy of state probabilities."""
        # Add small epsilon to avoid log(0)
        eps = 1e-10
        probs = state_probs + eps
        entropy = -(probs * np.log(probs)).sum(axis=1)
        return entropy
    
    def _calculate_state_stability(self, state_probs: pd.DataFrame) -> pd.Series:
        """Calculate stability of state probabilities."""
        return 1 - state_probs.rolling(5).std().sum(axis=1)
    
    # Technical indicator calculations
    def _calculate_rsi(self, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI."""
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
        """Calculate MACD."""
        ema_fast = close.ewm(span=fast).mean()
        ema_slow = close.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        return macd
    
    def _calculate_bb_position(self, close: pd.Series, period: int = 20, std_dev: float = 2) -> pd.Series:
        """Calculate Bollinger Bands position."""
        sma = close.rolling(period).mean()
        std = close.rolling(period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        bb_position = (close - lower_band) / (upper_band - lower_band)
        return bb_position
    
    def _calculate_adx(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate ADX (simplified version)."""
        # Simplified ADX calculation
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        return atr  # Simplified as ATR
    
    def _calculate_cci(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
        """Calculate CCI."""
        typical_price = (high + low + close) / 3
        sma_tp = typical_price.rolling(period).mean()
        mad = typical_price.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())))
        cci = (typical_price - sma_tp) / (0.015 * mad)
        return cci
    
    def _calculate_mfi(self, high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 14) -> pd.Series:
        """Calculate MFI."""
        typical_price = (high + low + close) / 3
        money_flow = typical_price * volume
        
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(period).sum()
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(period).sum()
        
        mfi = 100 - (100 / (1 + positive_flow / negative_flow))
        return mfi
    
    def _calculate_roc(self, close: pd.Series, period: int = 10) -> pd.Series:
        """Calculate ROC."""
        roc = ((close - close.shift(period)) / close.shift(period)) * 100
        return roc
    
    def _calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate ATR."""
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        return atr
