"""
Example: Adding a New Exchange to the Abstract Architecture

This example shows how easy it is to add support for a new exchange
using the abstract architecture.
"""
from datetime import datetime
from typing import List
import pandas as pd
import numpy as np

from .modular_components import (
    BaseExchangeDataSource,
    ExchangeDataSourceFactory,
    ModelTrainerFactory
)


# ============================================================================
# Example 1: Adding a new exchange (e.g., FTX, Bybit, etc.)
# ============================================================================

class BybitDataSource(BaseExchangeDataSource):
    """Bybit exchange data source implementation."""
    
    @property
    def exchange_name(self) -> str:
        return "bybit"
    
    def get_supported_symbols(self) -> List[str]:
        """Get Bybit trading pairs."""
        return ["BTCUSDT", "ETHUSDT", "SOLUSDT", "AVAXUSDT", "DOGEUSDT"]
    
    def get_supported_timeframes(self) -> List[str]:
        """Get Bybit supported timeframes."""
        return ["1", "3", "5", "15", "30", "60", "120", "240", "360", "720", "D", "W", "M"]
    
    async def fetch_data(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        """Fetch OHLCV data from Bybit."""
        # In real implementation, would use pybit or REST API
        # Example using pybit:
        # from pybit import HTTP
        # session = HTTP(
        #     testnet=self.testnet,
        #     api_key=self.api_key,
        #     api_secret=self.api_secret
        # )
        # result = session.query_kline(
        #     symbol=symbol,
        #     interval="60",  # 1 hour
        #     from=int(start.timestamp()),
        #     to=int(end.timestamp())
        # )
        
        # Mock implementation
        hours = int((end - start).total_seconds() / 3600)
        data = pd.DataFrame({
            'timestamp': pd.date_range(start, end, freq='1H')[:hours],
            'open': np.random.randn(hours) * 8 + 40000,
            'high': np.random.randn(hours) * 8 + 40500,
            'low': np.random.randn(hours) * 8 + 39500,
            'close': np.random.randn(hours) * 8 + 40000,
            'volume': np.random.randint(500, 5000, hours)
        })
        return data.set_index('timestamp')
    
    def _perform_connection_test(self) -> bool:
        """Test Bybit API connectivity."""
        # Would implement actual API test
        # try:
        #     session = HTTP(testnet=self.testnet)
        #     session.get_server_time()
        #     return True
        # except Exception:
        #     return False
        return True


class DeribitDataSource(BaseExchangeDataSource):
    """Deribit derivatives exchange data source."""
    
    @property
    def exchange_name(self) -> str:
        return "deribit"
    
    def get_supported_symbols(self) -> List[str]:
        """Get Deribit instruments."""
        # Deribit uses different naming convention
        return ["BTC-PERPETUAL", "ETH-PERPETUAL", "BTC-25MAR22", "ETH-25MAR22"]
    
    def get_supported_timeframes(self) -> List[str]:
        """Get Deribit supported timeframes."""
        return ["1", "3", "5", "10", "15", "30", "60", "120", "180", "360", "720", "1D"]
    
    async def fetch_data(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        """Fetch OHLCV data from Deribit."""
        # Real implementation would use deribit API
        # https://docs.deribit.com/
        
        hours = int((end - start).total_seconds() / 3600)
        data = pd.DataFrame({
            'timestamp': pd.date_range(start, end, freq='1H')[:hours],
            'open': np.random.randn(hours) * 20 + 45000,
            'high': np.random.randn(hours) * 20 + 45500,
            'low': np.random.randn(hours) * 20 + 44500,
            'close': np.random.randn(hours) * 20 + 45000,
            'volume': np.random.randint(1000, 10000, hours)
        })
        return data.set_index('timestamp')


# ============================================================================
# Example 2: Adding a new ML model type
# ============================================================================

from .modular_components import BaseModelTrainer, IModel
from typing import Optional


class CatBoostModel(IModel):
    """Wrapper for CatBoost model with standard interface."""
    
    def __init__(self, catboost_model):
        self.model = catboost_model
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities."""
        return self.model.predict_proba(X)
    
    def save(self, path: Path) -> None:
        """Save model to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(path))
    
    def load(self, path: Path) -> None:
        """Load model from disk."""
        self.model.load_model(str(path))


class CatBoostTrainer(BaseModelTrainer):
    """CatBoost model trainer."""
    
    @property
    def model_type(self) -> str:
        return "catboost"
    
    def _get_default_hyperparameters(self) -> Dict[str, Any]:
        """Get CatBoost default hyperparameters."""
        return {
            'iterations': 100,
            'learning_rate': 0.05,
            'depth': 6,
            'loss_function': 'Logloss',
            'verbose': False,
            'random_state': 42
        }
    
    def train(self, X: pd.DataFrame, y: pd.Series, validation_data: Optional[Tuple[pd.DataFrame, pd.Series]] = None) -> IModel:
        """Train CatBoost model."""
        from catboost import CatBoostClassifier
        
        # Create model
        self.model = CatBoostClassifier(**self.hyperparameters)
        
        # Prepare validation data if provided
        eval_set = None
        if validation_data is not None:
            X_val, y_val = validation_data
            eval_set = (X_val, y_val)
        
        # Train model
        self.model.fit(
            X, y,
            eval_set=eval_set,
            early_stopping_rounds=10
        )
        
        # Store feature importance
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance_ = pd.DataFrame({
                'feature': X.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
        
        return CatBoostModel(self.model)


# ============================================================================
# Registration and Usage
# ============================================================================

def register_new_components():
    """Register the new exchange and model implementations."""
    
    # Register new exchanges
    ExchangeDataSourceFactory.register_exchange('bybit', BybitDataSource)
    ExchangeDataSourceFactory.register_exchange('deribit', DeribitDataSource)
    
    # Register new model trainer
    ModelTrainerFactory.register_trainer('catboost', CatBoostTrainer)
    
    print("✅ New components registered successfully!")
    print(f"Available exchanges: {ExchangeDataSourceFactory.get_available_exchanges()}")
    print(f"Available models: {ModelTrainerFactory.get_available_models()}")


async def example_usage():
    """Example using the new components."""
    
    # Register new components
    register_new_components()
    
    # Create Bybit data source
    bybit = ExchangeDataSourceFactory.create(
        'bybit',
        api_key='your_api_key',
        api_secret='your_api_secret',
        testnet=True
    )
    
    # Fetch data
    data = await bybit.fetch_data(
        'BTCUSDT',
        datetime(2024, 1, 1),
        datetime(2024, 1, 2)
    )
    print(f"\nFetched {len(data)} hours of data from Bybit")
    print(f"Columns: {data.columns.tolist()}")
    
    # Create CatBoost trainer
    catboost_trainer = ModelTrainerFactory.create(
        'catboost',
        iterations=50,
        learning_rate=0.1,
        depth=4
    )
    
    # Prepare simple features and labels
    features = pd.DataFrame({
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100),
        'feature3': np.random.randn(100)
    })
    labels = pd.Series(np.random.randint(0, 2, 100))
    
    # Train model
    print("\nTraining CatBoost model...")
    model = catboost_trainer.train(features, labels)
    
    # Make predictions
    predictions = model.predict(features.iloc[:10])
    print(f"Sample predictions: {predictions}")
    
    # Get feature importance
    importance = catboost_trainer.get_feature_importance()
    print(f"\nFeature importance:\n{importance}")


# ============================================================================
# Configuration Example
# ============================================================================

EXAMPLE_CONFIG_WITH_NEW_COMPONENTS = {
    "name": "Pipeline_With_New_Components",
    "version": "1.0.0",
    
    "global_settings": {
        "data_source": {
            "type": "exchange",
            "exchange": "bybit",  # Use new exchange
            "api_key": "your_api_key",
            "api_secret": "your_api_secret",
            "testnet": True
        },
        "model": {
            "type": "catboost",  # Use new model
            "hyperparameters": {
                "iterations": 200,
                "learning_rate": 0.03,
                "depth": 8,
                "l2_leaf_reg": 3.0
            }
        }
    },
    
    "steps": {
        # ... rest of pipeline configuration
    }
}


if __name__ == "__main__":
    import asyncio
    asyncio.run(example_usage())