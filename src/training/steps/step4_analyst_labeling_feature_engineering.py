# src/training/steps/step4_analyst_labeling_feature_engineering.py

import asyncio
import json
import os
import pandas as pd
import pickle
import numpy as np
from typing import Any, Dict, Optional, List, Tuple
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from src.utils.error_handler import handle_errors
from src.utils.logger import system_logger


class AnalystLabelingFeatureEngineeringStep:
    """Step 4: Analyst Model Labeling & Feature Engineering."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = system_logger
        self.encoders = {}
        
    async def initialize(self) -> None:
        """Initialize the analyst labeling and feature engineering step."""
        try:
            self.logger.info("Initializing Analyst Labeling & Feature Engineering Step...")
            self.logger.info("Analyst Labeling & Feature Engineering Step initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing Analyst Labeling & Feature Engineering Step: {e}")
            raise
    
    async def execute(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute analyst labeling and feature engineering.
        
        Args:
            training_input: Training input parameters
            pipeline_state: Current pipeline state
            
        Returns:
            Dict containing labeling and feature engineering results
        """
        try:
            self.logger.info("🔄 Executing Analyst Labeling & Feature Engineering...")
            
            # Extract parameters
            symbol = training_input.get("symbol", "ETHUSDT")
            exchange = training_input.get("exchange", "BINANCE")
            data_dir = training_input.get("data_dir", "data/training")
            
            # Load regime data
            regime_data_dir = f"{data_dir}/regime_data"
            regime_data = {}
            
            # Load all regime data files
            for file in os.listdir(regime_data_dir):
                if file.startswith(f"{exchange}_{symbol}_") and file.endswith("_data.pkl"):
                    regime_name = file.replace(f"{exchange}_{symbol}_", "").replace("_data.pkl", "")
                    regime_file = os.path.join(regime_data_dir, file)
                    
                    with open(regime_file, 'rb') as f:
                        regime_data[regime_name] = pickle.load(f)
            
            if not regime_data:
                raise ValueError(f"No regime data found in {regime_data_dir}")
            
            # Process each regime
            labeled_data = {}
            feature_engineering_results = {}
            
            for regime_name, regime_df in regime_data.items():
                self.logger.info(f"Processing regime: {regime_name}")
                
                # Apply Triple Barrier Method for labeling
                labeled_regime_data = await self._apply_triple_barrier_labeling(regime_df, regime_name)
                
                # Perform feature engineering
                engineered_features = await self._perform_feature_engineering(labeled_regime_data, regime_name)
                
                # Train regime-specific encoders
                encoders = await self._train_regime_encoders(engineered_features, regime_name)
                
                labeled_data[regime_name] = labeled_regime_data
                feature_engineering_results[regime_name] = {
                    "features": engineered_features,
                    "encoders": encoders
                }
            
            # Save labeled data and feature engineering results
            labeled_data_dir = f"{data_dir}/labeled_data"
            os.makedirs(labeled_data_dir, exist_ok=True)
            
            for regime_name, data in labeled_data.items():
                labeled_file = f"{labeled_data_dir}/{exchange}_{symbol}_{regime_name}_labeled.pkl"
                with open(labeled_file, 'wb') as f:
                    pickle.dump(data, f)
            
            # Save feature engineering results
            feature_file = f"{data_dir}/{exchange}_{symbol}_feature_engineering.json"
            with open(feature_file, 'w') as f:
                json.dump(feature_engineering_results, f, indent=2)
            
            self.logger.info(f"✅ Analyst labeling and feature engineering completed. Results saved to {labeled_data_dir}")
            
            # Update pipeline state
            pipeline_state["labeled_data"] = labeled_data
            pipeline_state["feature_engineering_results"] = feature_engineering_results
            
            return {
                "labeled_data": labeled_data,
                "feature_engineering_results": feature_engineering_results,
                "duration": 0.0,  # Will be calculated in actual implementation
                "status": "SUCCESS"
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error in Analyst Labeling & Feature Engineering: {e}")
            return {
                "status": "FAILED",
                "error": str(e),
                "duration": 0.0
            }
    
    async def _apply_triple_barrier_labeling(self, data: pd.DataFrame, regime_name: str) -> pd.DataFrame:
        """
        Apply Triple Barrier Method for labeling.
        
        Args:
            data: Market data for the regime
            regime_name: Name of the regime
            
        Returns:
            DataFrame with labels added
        """
        try:
            self.logger.info(f"Applying Triple Barrier Method for regime: {regime_name}")
            
            # Ensure we have required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in required_columns:
                if col not in data.columns:
                    raise ValueError(f"Missing required column: {col}")
            
            # Calculate features for labeling
            data = self._calculate_features(data)
            
            # Apply Triple Barrier Method
            labeled_data = data.copy()
            
            # Define barrier parameters based on regime
            if regime_name.upper() in ['BULL', 'BEAR', 'SIDEWAYS']:
                # High-data regimes: more aggressive barriers
                profit_take_multiplier = 0.002  # 0.2%
                stop_loss_multiplier = 0.001    # 0.1%
                time_barrier_minutes = 30       # 30 minutes
            else:
                # Low-data regimes: more conservative barriers
                profit_take_multiplier = 0.001  # 0.1%
                stop_loss_multiplier = 0.0005   # 0.05%
                time_barrier_minutes = 15       # 15 minutes
            
            # Apply triple barrier labeling
            labels = []
            for i in range(len(data)):
                if i >= len(data) - 1:  # Skip last point
                    labels.append(0)
                    continue
                
                entry_price = data.iloc[i]['close']
                entry_time = data.index[i]
                
                # Calculate barriers
                profit_take_barrier = entry_price * (1 + profit_take_multiplier)
                stop_loss_barrier = entry_price * (1 - stop_loss_multiplier)
                time_barrier = entry_time + timedelta(minutes=time_barrier_minutes)
                
                # Check if any barrier is hit
                label = 0  # Neutral
                
                for j in range(i + 1, min(i + 100, len(data))):  # Look ahead up to 100 points
                    current_time = data.index[j]
                    current_price = data.iloc[j]['high']  # Use high for profit take
                    current_low = data.iloc[j]['low']     # Use low for stop loss
                    
                    # Check time barrier
                    if current_time > time_barrier:
                        label = 0  # Time barrier hit - neutral
                        break
                    
                    # Check profit take barrier
                    if current_price >= profit_take_barrier:
                        label = 1  # Profit take hit - positive
                        break
                    
                    # Check stop loss barrier
                    if current_low <= stop_loss_barrier:
                        label = -1  # Stop loss hit - negative
                        break
                
                labels.append(label)
            
            labeled_data['label'] = labels
            
            # Calculate label distribution
            label_counts = pd.Series(labels).value_counts()
            self.logger.info(f"Label distribution for {regime_name}: {dict(label_counts)}")
            
            return labeled_data
            
        except Exception as e:
            self.logger.error(f"Error in triple barrier labeling for {regime_name}: {e}")
            raise
    
    def _calculate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical features for the data.
        
        Args:
            data: Market data
            
        Returns:
            DataFrame with features added
        """
        try:
            # Calculate RSI
            data['rsi'] = self._calculate_rsi(data['close'])
            
            # Calculate MACD
            macd, signal = self._calculate_macd(data['close'])
            data['macd'] = macd
            data['macd_signal'] = signal
            data['macd_histogram'] = macd - signal
            
            # Calculate Bollinger Bands
            bb_upper, bb_lower = self._calculate_bollinger_bands(data['close'])
            data['bb_upper'] = bb_upper
            data['bb_lower'] = bb_lower
            data['bb_width'] = bb_upper - bb_lower
            data['bb_position'] = (data['close'] - bb_lower) / (bb_upper - bb_lower)
            
            # Calculate ATR
            data['atr'] = self._calculate_atr(data)
            
            # Calculate price-based features
            data['price_change'] = data['close'].pct_change()
            data['price_change_abs'] = data['price_change'].abs()
            data['high_low_ratio'] = data['high'] / data['low']
            data['volume_price_ratio'] = data['volume'] / data['close']
            
            # Calculate moving averages
            data['sma_5'] = data['close'].rolling(window=5).mean()
            data['sma_10'] = data['close'].rolling(window=10).mean()
            data['sma_20'] = data['close'].rolling(window=20).mean()
            data['ema_12'] = data['close'].ewm(span=12).mean()
            data['ema_26'] = data['close'].ewm(span=26).mean()
            
            # Calculate momentum features
            data['momentum_5'] = data['close'] / data['close'].shift(5) - 1
            data['momentum_10'] = data['close'] / data['close'].shift(10) - 1
            data['momentum_20'] = data['close'] / data['close'].shift(20) - 1
            
            # Fill NaN values
            data = data.fillna(method='bfill').fillna(0)
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error calculating features: {e}")
            raise
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series]:
        """Calculate MACD indicator."""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        return macd, signal_line
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: int = 2) -> Tuple[pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, lower_band
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift())
        low_close = np.abs(data['low'] - data['close'].shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        return atr
    
    async def _perform_feature_engineering(self, data: pd.DataFrame, regime_name: str) -> pd.DataFrame:
        """
        Perform feature engineering for the regime.
        
        Args:
            data: Labeled data with features
            regime_name: Name of the regime
            
        Returns:
            DataFrame with engineered features
        """
        try:
            self.logger.info(f"Performing feature engineering for regime: {regime_name}")
            
            # Select feature columns
            feature_columns = [
                'rsi', 'macd', 'macd_signal', 'macd_histogram',
                'bb_upper', 'bb_lower', 'bb_width', 'bb_position',
                'atr', 'price_change', 'price_change_abs', 'high_low_ratio',
                'volume_price_ratio', 'sma_5', 'sma_10', 'sma_20',
                'ema_12', 'ema_26', 'momentum_5', 'momentum_10', 'momentum_20'
            ]
            
            # Filter available features
            available_features = [col for col in feature_columns if col in data.columns]
            
            if not available_features:
                raise ValueError(f"No features available for regime {regime_name}")
            
            # Create feature matrix
            feature_data = data[available_features].copy()
            
            # Remove rows with NaN values
            feature_data = feature_data.dropna()
            
            # Standardize features
            scaler = StandardScaler()
            feature_data_scaled = scaler.fit_transform(feature_data)
            feature_data_scaled = pd.DataFrame(feature_data_scaled, columns=available_features, index=feature_data.index)
            
            # Add label column
            feature_data_scaled['label'] = data.loc[feature_data.index, 'label']
            
            self.logger.info(f"Feature engineering completed for {regime_name}: {len(feature_data_scaled)} samples, {len(available_features)} features")
            
            return feature_data_scaled
            
        except Exception as e:
            self.logger.error(f"Error in feature engineering for {regime_name}: {e}")
            raise
    
    async def _train_regime_encoders(self, data: pd.DataFrame, regime_name: str) -> Dict[str, Any]:
        """
        Train regime-specific encoders.
        
        Args:
            data: Feature-engineered data
            regime_name: Name of the regime
            
        Returns:
            Dict containing trained encoders
        """
        try:
            self.logger.info(f"Training encoders for regime: {regime_name}")
            
            # Separate features and labels
            feature_columns = [col for col in data.columns if col != 'label']
            X = data[feature_columns]
            y = data['label']
            
            # Train PCA encoder for dimensionality reduction
            pca = PCA(n_components=min(10, len(feature_columns)))
            X_pca = pca.fit_transform(X)
            
            # Train autoencoder for feature learning
            from sklearn.neural_network import MLPRegressor
            autoencoder = MLPRegressor(
                hidden_layer_sizes=(len(feature_columns), len(feature_columns)//2, len(feature_columns)),
                max_iter=1000,
                random_state=42
            )
            autoencoder.fit(X, X)
            
            # Store encoders
            encoders = {
                "pca": pca,
                "autoencoder": autoencoder,
                "feature_columns": feature_columns,
                "n_features": len(feature_columns),
                "n_samples": len(X)
            }
            
            self.logger.info(f"Encoders trained for {regime_name}: PCA components={pca.n_components_}, Autoencoder layers={autoencoder.n_layers_}")
            
            return encoders
            
        except Exception as e:
            self.logger.error(f"Error training encoders for {regime_name}: {e}")
            raise


# For backward compatibility with existing step structure
async def run_step(
    symbol: str,
    exchange: str = "BINANCE",
    data_dir: str = "data/training",
    **kwargs
) -> bool:
    """
    Run the analyst labeling and feature engineering step.
    
    Args:
        symbol: Trading symbol
        exchange: Exchange name
        data_dir: Data directory path
        **kwargs: Additional parameters
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create step instance
        config = {"symbol": symbol, "exchange": exchange, "data_dir": data_dir}
        step = AnalystLabelingFeatureEngineeringStep(config)
        await step.initialize()
        
        # Execute step
        training_input = {
            "symbol": symbol,
            "exchange": exchange,
            "data_dir": data_dir,
            **kwargs
        }
        
        pipeline_state = {}
        result = await step.execute(training_input, pipeline_state)
        
        return result.get("status") == "SUCCESS"
        
    except Exception as e:
        print(f"❌ Analyst labeling and feature engineering failed: {e}")
        return False


if __name__ == "__main__":
    # Test the step
    async def test():
        result = await run_step("ETHUSDT", "BINANCE", "data/training")
        print(f"Test result: {result}")
    
    asyncio.run(test())
