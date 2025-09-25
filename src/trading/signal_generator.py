"""
Signal Generator

Advanced signal generation system with comprehensive trading signal analysis,
optimization, and execution capabilities. Integrates with shared utilities for
enhanced performance and M1 hardware optimization.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import warnings

# Import shared utilities
from src.utils.common_operations import (
    safe_dataframe_operation, validate_dataframe_columns, 
    safe_convert_dtypes, calculate_data_quality_metrics,
    safe_merge_dataframes, create_summary_statistics,
    safe_drop_columns, safe_rename_columns,
    validate_timestamp_column, safe_timestamp_conversion,
    get_dataframe_info, create_data_quality_report,
    CommonUtilities, safe_divide, safe_log, safe_sqrt,
    safe_percentage_change, safe_weighted_average
)
from src.utils.math_validation import (
    validate_finite, validate_positive, validate_range,
    safe_correlation, safe_covariance, safe_mean, safe_std,
    safe_percentile, validate_correlation_matrix,
    safe_matrix_inverse, MathValidation, safe_kelly_calculation
)
from src.utils.tprint import tprint, tprint_info, tprint_warning, tprint_error, tprint_success
from src.utils.serialization_utils import JSONSerializer, PickleSerializer, UniversalSerializer
from src.utils.data.klines_parquet import KlinesParquetManager, get_klines_manager
from src.utils.ml_common.optimization.bayesian_tpe_optimizer import (
    BayesianTPEOptimizer, BayesianTPEConfig, OptimizationResult
)
from src.utils.hardware.m1_gpu_utils import get_m1_gpu_manager, is_m1_available, is_mps_available
from src.utils.hardware.m1_memory_optimizer import get_m1_memory_optimizer, optimize_dataframe_memory
from src.utils.hardware.m1_cpu_optimizer import get_m1_cpu_optimizer

# Optional ML imports with graceful fallback
try:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.linear_model import LogisticRegression, LinearRegression
    from sklearn.svm import SVC, SVR
    from sklearn.model_selection import cross_val_score, train_test_split
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("scikit-learn not available, some functionality will be limited")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

# Setup logging
logger = logging.getLogger(__name__)

class SignalType(Enum):
    """Signal types enumeration."""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    STRONG_BUY = "strong_buy"
    STRONG_SELL = "strong_sell"

class SignalStrength(Enum):
    """Signal strength enumeration."""
    WEAK = 1
    MODERATE = 2
    STRONG = 3
    VERY_STRONG = 4
    EXTREME = 5

@dataclass
class SignalConfig:
    """Configuration for signal generation."""
    
    # Model configuration
    model_type: str = 'random_forest'  # 'random_forest', 'xgboost', 'lightgbm', 'logistic_regression', 'svm'
    n_estimators: int = 100
    max_depth: Optional[int] = None
    random_state: int = 42
    
    # Signal thresholds
    buy_threshold: float = 0.6
    sell_threshold: float = 0.4
    strong_signal_threshold: float = 0.8
    weak_signal_threshold: float = 0.3
    
    # Risk management
    max_position_size: float = 0.1  # 10% of portfolio
    stop_loss_threshold: float = 0.05  # 5% stop loss
    take_profit_threshold: float = 0.15  # 15% take profit
    
    # Feature engineering
    feature_selection: bool = True
    feature_importance_threshold: float = 0.01
    lookback_periods: List[int] = field(default_factory=lambda: [5, 10, 20, 50])
    
    # Cross-validation
    cv_folds: int = 5
    test_size: float = 0.2
    
    # Optimization
    enable_hyperparameter_optimization: bool = True
    optimization_trials: int = 50
    
    # Performance
    enable_parallel_processing: bool = True
    n_jobs: int = -1
    
    # M1 optimization
    enable_m1_optimization: bool = True
    memory_limit_gb: Optional[float] = None
    
    # Logging
    verbose: bool = True
    log_level: str = 'INFO'

@dataclass
class TradingSignal:
    """Trading signal data structure."""
    
    # Signal information
    signal_type: SignalType
    signal_strength: SignalStrength
    confidence: float
    timestamp: datetime
    
    # Market data
    price: float
    volume: float
    symbol: str
    
    # Technical indicators
    technical_indicators: Dict[str, float] = field(default_factory=dict)
    
    # Risk metrics
    risk_score: float = 0.0
    position_size: float = 0.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    
    # Model information
    model_used: str = ""
    feature_importance: Dict[str, float] = field(default_factory=dict)
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SignalGenerationResult:
    """Result of signal generation."""
    
    # Generated signals
    signals: List[TradingSignal]
    
    # Performance metrics
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    
    # Model information
    model_type: str
    model_params: Dict[str, Any]
    training_time: float
    
    # Feature importance
    feature_importance: Dict[str, float]
    selected_features: List[str]
    
    # Cross-validation results
    cv_scores: List[float]
    cv_mean: float
    cv_std: float
    
    # Data quality
    data_quality_report: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    success: bool = True
    error_message: Optional[str] = None

class SignalGenerator:
    """
    Advanced signal generator with comprehensive trading signal analysis.
    
    Features:
    - Multiple ML algorithms for signal generation
    - Technical indicator integration
    - Risk management and position sizing
    - Hyperparameter optimization with Bayesian TPE
    - M1 hardware optimization
    - Comprehensive evaluation metrics
    - Real-time signal generation
    """
    
    def __init__(self, config: Optional[SignalConfig] = None):
        """Initialize signal generator."""
        self.config = config or SignalConfig()
        self.logger = logger.getChild('SignalGenerator')
        
        # Initialize utilities
        self.common_utils = CommonUtilities()
        self.math_validator = MathValidation()
        self.serializer = UniversalSerializer()
        
        # Initialize M1 optimizers if available
        self.m1_gpu_manager = None
        self.m1_memory_optimizer = None
        self.m1_cpu_optimizer = None
        
        if self.config.enable_m1_optimization:
            try:
                self.m1_gpu_manager = get_m1_gpu_manager()
                self.m1_memory_optimizer = get_m1_memory_optimizer(self.config.memory_limit_gb)
                self.m1_cpu_optimizer = get_m1_cpu_optimizer()
                
                if is_m1_available():
                    tprint_info("🧠 M1 optimization enabled")
                else:
                    tprint_warning("⚠️ M1 hardware not detected, using standard optimization")
            except Exception as e:
                tprint_warning(f"⚠️ M1 optimization setup failed: {e}")
        
        # Initialize model
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.feature_names = []
        self.is_trained = False
        
        # Signal history
        self.signal_history: List[TradingSignal] = []
        self.performance_metrics: Dict[str, Any] = {}
        
        tprint_info(f"📊 Signal Generator initialized (model: {self.config.model_type})")
    
    def _validate_input_data(self, X: Union[np.ndarray, pd.DataFrame], y: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Validate and preprocess input data."""
        try:
            # Convert to numpy arrays if needed
            if isinstance(X, pd.DataFrame):
                X_array = X.values
                self.feature_names = list(X.columns)
            else:
                X_array = np.array(X)
                self.feature_names = [f"feature_{i}" for i in range(X_array.shape[1])]
            
            # Validate X
            X_array = self.math_validator.validate_numeric_array(X_array, "X")
            
            # Validate y if provided
            y_array = None
            if y is not None:
                y_array = np.array(y)
                if y_array.ndim > 1:
                    y_array = y_array.flatten()
                y_array = self.math_validator.validate_numeric_array(y_array, "y")
            
            # M1 optimization
            if self.m1_memory_optimizer and isinstance(X, pd.DataFrame):
                X = self.m1_memory_optimizer.optimize_dataframe_memory(X)
            
            tprint_info(f"✅ Data validated: X shape {X_array.shape}, y shape {y_array.shape if y_array is not None else 'None'}")
            return X_array, y_array
            
        except Exception as e:
            tprint_error(f"❌ Data validation failed: {e}")
            raise
    
    def _create_model(self) -> Any:
        """Create the specified model."""
        try:
            if not SKLEARN_AVAILABLE:
                raise ImportError("scikit-learn is required for signal generation models")
            
            model_params = {
                'random_state': self.config.random_state,
                'n_jobs': self.config.n_jobs if self.config.enable_parallel_processing else 1
            }
            
            if self.config.model_type == 'random_forest':
                model = RandomForestClassifier(
                    n_estimators=self.config.n_estimators,
                    max_depth=self.config.max_depth,
                    **model_params
                )
            
            elif self.config.model_type == 'xgboost' and XGBOOST_AVAILABLE:
                model = xgb.XGBClassifier(
                    n_estimators=self.config.n_estimators,
                    max_depth=self.config.max_depth,
                    random_state=self.config.random_state,
                    n_jobs=self.config.n_jobs if self.config.enable_parallel_processing else 1
                )
            
            elif self.config.model_type == 'lightgbm' and LIGHTGBM_AVAILABLE:
                model = lgb.LGBMClassifier(
                    n_estimators=self.config.n_estimators,
                    max_depth=self.config.max_depth,
                    random_state=self.config.random_state,
                    n_jobs=self.config.n_jobs if self.config.enable_parallel_processing else 1,
                    verbose=-1
                )
            
            elif self.config.model_type == 'logistic_regression':
                model = LogisticRegression(
                    random_state=self.config.random_state,
                    max_iter=1000
                )
            
            elif self.config.model_type == 'svm':
                model = SVC(
                    random_state=self.config.random_state,
                    probability=True
                )
            
            else:
                raise ValueError(f"Unsupported model type: {self.config.model_type}")
            
            tprint_info(f"🤖 Created {self.config.model_type} model")
            return model
            
        except Exception as e:
            tprint_error(f"❌ Model creation failed: {e}")
            raise
    
    def _calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for signal generation."""
        try:
            df = data.copy()
            
            # Price-based indicators
            if 'close' in df.columns:
                # Moving averages
                for period in self.config.lookback_periods:
                    df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
                    df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
                
                # Price momentum
                df['price_change'] = df['close'].pct_change()
                df['price_momentum'] = df['close'] / df['close'].shift(1) - 1
                
                # Bollinger Bands
                if len(df) >= 20:
                    sma_20 = df['close'].rolling(window=20).mean()
                    std_20 = df['close'].rolling(window=20).std()
                    df['bb_upper'] = sma_20 + (std_20 * 2)
                    df['bb_lower'] = sma_20 - (std_20 * 2)
                    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / sma_20
                    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            # Volume indicators
            if 'volume' in df.columns:
                df['volume_sma'] = df['volume'].rolling(window=20).mean()
                df['volume_ratio'] = df['volume'] / df['volume_sma']
                df['volume_momentum'] = df['volume'].pct_change()
            
            # Volatility indicators
            if 'close' in df.columns:
                df['volatility'] = df['close'].rolling(window=20).std()
                df['volatility_ratio'] = df['volatility'] / df['volatility'].rolling(window=50).mean()
            
            # RSI
            if 'close' in df.columns and len(df) >= 14:
                delta = df['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                df['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            if 'close' in df.columns and len(df) >= 26:
                ema_12 = df['close'].ewm(span=12).mean()
                ema_26 = df['close'].ewm(span=26).mean()
                df['macd'] = ema_12 - ema_26
                df['macd_signal'] = df['macd'].ewm(span=9).mean()
                df['macd_histogram'] = df['macd'] - df['macd_signal']
            
            tprint_info(f"📈 Calculated {len([col for col in df.columns if col not in data.columns])} technical indicators")
            return df
            
        except Exception as e:
            tprint_warning(f"⚠️ Technical indicator calculation failed: {e}")
            return data
    
    def _select_features(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """Select important features for signal generation."""
        try:
            if not self.config.feature_selection:
                return X, self.feature_names
            
            # Create a temporary model for feature selection
            temp_model = self._create_model()
            temp_model.fit(X, y)
            
            # Get feature importance
            if hasattr(temp_model, 'feature_importances_'):
                importances = temp_model.feature_importances_
            else:
                # Fallback: use all features
                return X, self.feature_names
            
            # Select features above threshold
            selected_indices = np.where(importances >= self.config.feature_importance_threshold)[0]
            selected_features = [self.feature_names[i] for i in selected_indices]
            
            if len(selected_indices) == 0:
                tprint_warning("⚠️ No features selected, using all features")
                return X, self.feature_names
            
            X_selected = X[:, selected_indices]
            tprint_info(f"🔍 Selected {len(selected_features)} features from {len(self.feature_names)}")
            
            return X_selected, selected_features
            
        except Exception as e:
            tprint_warning(f"⚠️ Feature selection failed: {e}")
            return X, self.feature_names
    
    def _optimize_hyperparameters(self, X: np.ndarray, y: np.ndarray) -> Optional[OptimizationResult]:
        """Optimize hyperparameters using Bayesian TPE."""
        try:
            if not self.config.enable_hyperparameter_optimization:
                return None
            
            tprint_info("🎯 Starting hyperparameter optimization...")
            
            # Define search space based on model type
            if self.config.model_type == 'random_forest':
                search_space = {
                    'n_estimators': {'type': 'int', 'low': 50, 'high': 500},
                    'max_depth': {'type': 'int', 'low': 3, 'high': 20},
                    'min_samples_split': {'type': 'int', 'low': 2, 'high': 20},
                    'min_samples_leaf': {'type': 'int', 'low': 1, 'high': 10}
                }
            elif self.config.model_type == 'xgboost':
                search_space = {
                    'n_estimators': {'type': 'int', 'low': 50, 'high': 500},
                    'max_depth': {'type': 'int', 'low': 3, 'high': 15},
                    'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3, 'log': True},
                    'subsample': {'type': 'float', 'low': 0.6, 'high': 1.0}
                }
            else:
                tprint_warning(f"⚠️ Hyperparameter optimization not supported for {self.config.model_type}")
                return None
            
            # Create objective function
            def objective_function(params: Dict[str, Any], X: np.ndarray, y: np.ndarray) -> float:
                try:
                    # Create model with given parameters
                    model_params = {
                        'random_state': self.config.random_state,
                        'n_jobs': 1  # Use single job for optimization
                    }
                    model_params.update(params)
                    
                    if self.config.model_type == 'random_forest':
                        model = RandomForestClassifier(**model_params)
                    elif self.config.model_type == 'xgboost':
                        model = xgb.XGBClassifier(**model_params)
                    else:
                        return 0.0
                    
                    # Cross-validation score
                    scores = cross_val_score(model, X, y, cv=self.config.cv_folds, scoring='accuracy')
                    return float(np.mean(scores))
                    
                except Exception:
                    return 0.0
            
            # Configure optimization
            tpe_config = BayesianTPEConfig(
                n_trials=self.config.optimization_trials,
                enable_grid_search=True,
                coarse_grid_points=3,
                fine_grid_points=5
            )
            
            # Run optimization
            optimizer = BayesianTPEOptimizer(tpe_config)
            result = optimizer.optimize(
                objective_function=lambda params: objective_function(params, X, y),
                search_space=search_space
            )
            
            if result.success:
                tprint_info(f"✅ Hyperparameter optimization completed: best score {result.best_score:.4f}")
                return result
            else:
                tprint_warning(f"⚠️ Hyperparameter optimization failed: {result.error_message}")
                return None
                
        except Exception as e:
            tprint_warning(f"⚠️ Hyperparameter optimization failed: {e}")
            return None
    
    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: np.ndarray) -> 'SignalGenerator':
        """Train the signal generator."""
        try:
            tprint_info("🤖 Training Signal Generator...")
            start_time = datetime.now()
            
            # Validate input data
            X_array, y_array = self._validate_input_data(X, y)
            
            # Feature selection
            X_selected, selected_features = self._select_features(X_array, y_array)
            self.selected_features = selected_features
            
            # Hyperparameter optimization
            optimization_result = self._optimize_hyperparameters(X_selected, y_array)
            
            # Create final model
            self.model = self._create_model()
            
            # Apply optimized parameters if available
            if optimization_result and optimization_result.success:
                if hasattr(self.model, 'set_params'):
                    self.model.set_params(**optimization_result.best_params)
                tprint_info(f"🔧 Applied optimized parameters: {optimization_result.best_params}")
            
            # Train model
            self.model.fit(X_selected, y_array)
            
            # Store training metadata
            self.training_time = (datetime.now() - start_time).total_seconds()
            self.is_trained = True
            
            tprint_info(f"✅ Training completed in {self.training_time:.2f}s")
            return self
            
        except Exception as e:
            tprint_error(f"❌ Training failed: {e}")
            raise
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Make predictions."""
        try:
            if not self.is_trained:
                raise ValueError("Model must be trained before making predictions")
            
            # Validate input
            X_array, _ = self._validate_input_data(X)
            
            # Select same features as training
            if hasattr(self, 'selected_features') and len(self.selected_features) < len(self.feature_names):
                # Find indices of selected features
                selected_indices = [i for i, name in enumerate(self.feature_names) if name in self.selected_features]
                X_selected = X_array[:, selected_indices]
            else:
                X_selected = X_array
            
            # Make predictions
            predictions = self.model.predict(X_selected)
            
            return predictions
            
        except Exception as e:
            tprint_error(f"❌ Prediction failed: {e}")
            raise
    
    def generate_signals(self, market_data: pd.DataFrame, symbol: str = "ETHUSDT") -> List[TradingSignal]:
        """Generate trading signals from market data."""
        try:
            if not self.is_trained:
                raise ValueError("Signal generator must be trained before generating signals")
            
            tprint_info(f"📊 Generating signals for {symbol}...")
            
            # Calculate technical indicators
            data_with_indicators = self._calculate_technical_indicators(market_data)
            
            # Prepare features
            feature_columns = [col for col in data_with_indicators.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
            X = data_with_indicators[feature_columns].fillna(0)
            
            # Generate predictions
            predictions = self.predict(X)
            
            # Get prediction probabilities if available
            probabilities = None
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(X)
            
            # Convert predictions to signals
            signals = []
            for i, (idx, row) in enumerate(data_with_indicators.iterrows()):
                try:
                    # Determine signal type and strength
                    signal_type, signal_strength, confidence = self._classify_signal(
                        predictions[i], probabilities[i] if probabilities is not None else None
                    )
                    
                    # Calculate risk metrics
                    risk_score = self._calculate_risk_score(data_with_indicators.iloc[i])
                    position_size = self._calculate_position_size(confidence, risk_score)
                    
                    # Create signal
                    signal = TradingSignal(
                        signal_type=signal_type,
                        signal_strength=signal_strength,
                        confidence=confidence,
                        timestamp=idx if isinstance(idx, datetime) else datetime.now(),
                        price=row.get('close', 0.0),
                        volume=row.get('volume', 0.0),
                        symbol=symbol,
                        technical_indicators=self._extract_technical_indicators(row),
                        risk_score=risk_score,
                        position_size=position_size,
                        stop_loss=self._calculate_stop_loss(row.get('close', 0.0), signal_type),
                        take_profit=self._calculate_take_profit(row.get('close', 0.0), signal_type),
                        model_used=self.config.model_type,
                        feature_importance=self._get_feature_importance(),
                        metadata={
                            'prediction': predictions[i],
                            'feature_count': len(feature_columns)
                        }
                    )
                    
                    signals.append(signal)
                    
                except Exception as e:
                    tprint_warning(f"⚠️ Failed to create signal for row {i}: {e}")
                    continue
            
            # Store signals in history
            self.signal_history.extend(signals)
            
            tprint_success(f"✅ Generated {len(signals)} signals for {symbol}")
            return signals
            
        except Exception as e:
            tprint_error(f"❌ Signal generation failed: {e}")
            return []
    
    def _classify_signal(self, prediction: int, probabilities: Optional[np.ndarray] = None) -> Tuple[SignalType, SignalStrength, float]:
        """Classify prediction into signal type and strength."""
        try:
            # Map prediction to signal type
            if prediction == 0:  # Sell
                signal_type = SignalType.SELL
            elif prediction == 1:  # Buy
                signal_type = SignalType.BUY
            else:  # Hold
                signal_type = SignalType.HOLD
            
            # Calculate confidence
            if probabilities is not None:
                confidence = float(np.max(probabilities))
            else:
                confidence = 0.5  # Default confidence
            
            # Determine signal strength
            if confidence >= self.config.strong_signal_threshold:
                signal_strength = SignalStrength.VERY_STRONG
                if signal_type == SignalType.BUY:
                    signal_type = SignalType.STRONG_BUY
                elif signal_type == SignalType.SELL:
                    signal_type = SignalType.STRONG_SELL
            elif confidence >= self.config.buy_threshold:
                signal_strength = SignalStrength.STRONG
            elif confidence >= self.config.sell_threshold:
                signal_strength = SignalStrength.MODERATE
            else:
                signal_strength = SignalStrength.WEAK
            
            return signal_type, signal_strength, confidence
            
        except Exception as e:
            tprint_warning(f"⚠️ Signal classification failed: {e}")
            return SignalType.HOLD, SignalStrength.WEAK, 0.0
    
    def _calculate_risk_score(self, row: pd.Series) -> float:
        """Calculate risk score for a given data point."""
        try:
            risk_factors = []
            
            # Volatility risk
            if 'volatility' in row:
                volatility = row['volatility']
                if not np.isnan(volatility) and volatility > 0:
                    risk_factors.append(min(volatility * 10, 1.0))  # Scale volatility
            
            # Volume risk
            if 'volume_ratio' in row:
                volume_ratio = row['volume_ratio']
                if not np.isnan(volume_ratio) and volume_ratio > 0:
                    # High volume ratio can indicate risk
                    risk_factors.append(min(abs(volume_ratio - 1.0), 1.0))
            
            # Price momentum risk
            if 'price_momentum' in row:
                momentum = abs(row['price_momentum'])
                if not np.isnan(momentum):
                    risk_factors.append(min(momentum * 5, 1.0))  # Scale momentum
            
            # Calculate average risk
            if risk_factors:
                return float(np.mean(risk_factors))
            else:
                return 0.5  # Default moderate risk
                
        except Exception as e:
            tprint_warning(f"⚠️ Risk score calculation failed: {e}")
            return 0.5
    
    def _calculate_position_size(self, confidence: float, risk_score: float) -> float:
        """Calculate position size based on confidence and risk."""
        try:
            # Base position size from confidence
            base_size = confidence * self.config.max_position_size
            
            # Adjust for risk
            risk_adjustment = 1.0 - (risk_score * 0.5)  # Reduce size by up to 50% for high risk
            
            # Final position size
            position_size = base_size * risk_adjustment
            
            # Ensure within bounds
            return max(0.0, min(position_size, self.config.max_position_size))
            
        except Exception as e:
            tprint_warning(f"⚠️ Position size calculation failed: {e}")
            return 0.0
    
    def _extract_technical_indicators(self, row: pd.Series) -> Dict[str, float]:
        """Extract technical indicators from a data row."""
        try:
            indicators = {}
            
            # Common technical indicators
            indicator_names = ['rsi', 'macd', 'bb_position', 'bb_width', 'volatility', 'volume_ratio']
            
            for name in indicator_names:
                if name in row and not np.isnan(row[name]):
                    indicators[name] = float(row[name])
            
            return indicators
            
        except Exception as e:
            tprint_warning(f"⚠️ Technical indicator extraction failed: {e}")
            return {}
    
    def _calculate_stop_loss(self, price: float, signal_type: SignalType) -> Optional[float]:
        """Calculate stop loss price."""
        try:
            if signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
                return price * (1 - self.config.stop_loss_threshold)
            elif signal_type in [SignalType.SELL, SignalType.STRONG_SELL]:
                return price * (1 + self.config.stop_loss_threshold)
            else:
                return None
                
        except Exception as e:
            tprint_warning(f"⚠️ Stop loss calculation failed: {e}")
            return None
    
    def _calculate_take_profit(self, price: float, signal_type: SignalType) -> Optional[float]:
        """Calculate take profit price."""
        try:
            if signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
                return price * (1 + self.config.take_profit_threshold)
            elif signal_type in [SignalType.SELL, SignalType.STRONG_SELL]:
                return price * (1 - self.config.take_profit_threshold)
            else:
                return None
                
        except Exception as e:
            tprint_warning(f"⚠️ Take profit calculation failed: {e}")
            return None
    
    def _get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from the trained model."""
        try:
            if not self.is_trained or not hasattr(self.model, 'feature_importances_'):
                return {}
            
            return dict(zip(
                getattr(self, 'selected_features', self.feature_names),
                self.model.feature_importances_
            ))
            
        except Exception as e:
            tprint_warning(f"⚠️ Feature importance extraction failed: {e}")
            return {}
    
    def analyze_performance(self, X: Union[np.ndarray, pd.DataFrame], y: np.ndarray) -> SignalGenerationResult:
        """Analyze signal generation performance."""
        try:
            tprint_info("📊 Analyzing signal generation performance...")
            
            # Validate input data
            X_array, y_array = self._validate_input_data(X, y)
            
            # Data quality assessment
            if isinstance(X, pd.DataFrame):
                data_quality_report = create_data_quality_report(X)
            else:
                data_quality_report = {
                    'shape': X_array.shape,
                    'dtype': str(X_array.dtype),
                    'memory_usage_mb': X_array.nbytes / (1024 * 1024)
                }
            
            # Train model if not already trained
            if not self.is_trained:
                self.fit(X, y_array)
            
            # Cross-validation
            cv_scores = cross_val_score(
                self.model, X_array, y_array, 
                cv=self.config.cv_folds, 
                scoring='accuracy'
            )
            
            # Make predictions
            predictions = self.predict(X_array)
            
            # Calculate metrics
            accuracy = accuracy_score(y_array, predictions)
            
            # Get feature importance
            feature_importance = self._get_feature_importance()
            
            # Create result
            result = SignalGenerationResult(
                signals=[],  # Would be populated with actual signals
                accuracy=accuracy,
                precision=0.0,  # Would need classification_report for detailed metrics
                recall=0.0,
                f1_score=0.0,
                model_type=self.config.model_type,
                model_params=self.model.get_params() if hasattr(self.model, 'get_params') else {},
                training_time=getattr(self, 'training_time', 0.0),
                feature_importance=feature_importance,
                selected_features=getattr(self, 'selected_features', self.feature_names),
                cv_scores=cv_scores.tolist(),
                cv_mean=float(np.mean(cv_scores)),
                cv_std=float(np.std(cv_scores)),
                data_quality_report=data_quality_report
            )
            
            tprint_success(f"✅ Performance analysis completed: accuracy {accuracy:.4f}")
            return result
            
        except Exception as e:
            tprint_error(f"❌ Performance analysis failed: {e}")
            return SignalGenerationResult(
                signals=[],
                accuracy=0.0,
                precision=0.0,
                recall=0.0,
                f1_score=0.0,
                model_type=self.config.model_type,
                model_params={},
                training_time=0.0,
                feature_importance={},
                selected_features=[],
                cv_scores=[],
                cv_mean=0.0,
                cv_std=0.0,
                success=False,
                error_message=str(e)
            )
    
    def get_signal_history(self, limit: int = 100) -> List[TradingSignal]:
        """Get recent signal history."""
        return self.signal_history[-limit:] if self.signal_history else []
    
    def save_model(self, filepath: str) -> bool:
        """Save the trained model."""
        try:
            if not self.is_trained:
                raise ValueError("No trained model to save")
            
            model_data = {
                'model': self.model,
                'config': self.config,
                'feature_names': self.feature_names,
                'selected_features': getattr(self, 'selected_features', []),
                'is_trained': self.is_trained,
                'training_time': getattr(self, 'training_time', 0.0)
            }
            
            return self.serializer.save(model_data, filepath)
            
        except Exception as e:
            tprint_error(f"❌ Model saving failed: {e}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """Load a trained model."""
        try:
            model_data = self.serializer.load(filepath)
            if model_data is None:
                return False
            
            self.model = model_data['model']
            self.config = model_data['config']
            self.feature_names = model_data['feature_names']
            self.selected_features = model_data.get('selected_features', [])
            self.is_trained = model_data['is_trained']
            self.training_time = model_data.get('training_time', 0.0)
            
            tprint_success("✅ Model loaded successfully")
            return True
            
        except Exception as e:
            tprint_error(f"❌ Model loading failed: {e}")
            return False


# Convenience functions
def create_signal_generator(
    model_type: str = 'random_forest',
    enable_optimization: bool = True,
    **kwargs
) -> SignalGenerator:
    """Create a signal generator with specified configuration."""
    config = SignalConfig(
        model_type=model_type,
        enable_hyperparameter_optimization=enable_optimization,
        **kwargs
    )
    return SignalGenerator(config)


def generate_trading_signals(
    market_data: pd.DataFrame,
    model_type: str = 'random_forest',
    symbol: str = "ETHUSDT",
    **kwargs
) -> List[TradingSignal]:
    """Convenience function for signal generation."""
    generator = create_signal_generator(model_type=model_type, **kwargs)
    
    # This would need training data to work properly
    # For now, return empty list
    tprint_warning("⚠️ Signal generation requires training data")
    return []


# Export main classes and functions
__all__ = [
    'SignalGenerator',
    'SignalConfig',
    'TradingSignal',
    'SignalGenerationResult',
    'SignalType',
    'SignalStrength',
    'create_signal_generator',
    'generate_trading_signals'
]