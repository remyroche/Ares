"""
Backtesting Engine

Comprehensive backtesting engine for NAS-TAS models with regime-aware
validation, performance analysis, and risk assessment.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
import logging
from datetime import datetime, timedelta
from pathlib import Path
import json
import pickle
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# Import ML common utilities and optimization tools
try:
    from src.utils.ml_common.common_operations import get_ml_common_operations
    from src.utils.ml_common.validation import get_validation_framework
    ML_COMMON_AVAILABLE = True
except ImportError:
    ML_COMMON_AVAILABLE = False

# Import general ML pipeline optimization tools
try:
    from src.utils.common_ml.backtesting.backtesting_engine import (
        BacktestingEngine as CommonBacktestingEngine,
        BacktestingConfig as CommonBacktestingConfig
    )
    from src.training.steps.backtesting.walk_forward_validation import WalkForwardValidation
    from src.utils.nas_tas.monte_carlo_engine import UnifiedMonteCarloEngine, MonteCarloConfig
    from src.training.steps.backtesting.consolidated_backtesting_step import ConsolidatedBacktestingStep
    OPTIMIZED_TOOLS_AVAILABLE = True
except ImportError:
    OPTIMIZED_TOOLS_AVAILABLE = False

# Import regime detection systems
try:
    from src.training.steps.market_analysis.tas_regime.core.tas_regime_detector import TASRegimeDetector, TASRegimeConfig
    from src.training.steps.market_analysis.nas_regime.core.perfect_nas_regime_detector import PerfectNASRegimeDetector, PerfectNASConfig
    REGIME_DETECTION_AVAILABLE = True
except ImportError:
    REGIME_DETECTION_AVAILABLE = False

logger = logging.getLogger(__name__)


class BacktestingMode(Enum):
    """Backtesting modes."""
    HISTORICAL = "historical"      # Historical data backtesting
    WALK_FORWARD = "walk_forward"  # Walk-forward analysis
    OUT_OF_SAMPLE = "out_of_sample" # Out-of-sample testing
    CROSS_VALIDATION = "cross_validation" # Cross-validation


class PerformanceMetric(Enum):
    """Performance metrics for backtesting."""
    TOTAL_RETURN = "total_return"
    ANNUALIZED_RETURN = "annualized_return"
    SHARPE_RATIO = "sharpe_ratio"
    SORTINO_RATIO = "sortino_ratio"
    MAX_DRAWDOWN = "max_drawdown"
    CALMAR_RATIO = "calmar_ratio"
    WIN_RATE = "win_rate"
    PROFIT_FACTOR = "profit_factor"
    VAR = "var"
    CVAR = "cvar"


@dataclass
class BacktestingConfig:
    """Configuration for backtesting engine."""
    
    # Backtesting settings
    mode: BacktestingMode = BacktestingMode.HISTORICAL
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    initial_capital: float = 100000.0
    commission_rate: float = 0.001  # 0.1%
    slippage_rate: float = 0.0005   # 0.05%
    
    # Data settings
    data_frequency: str = "daily"  # "minute", "hourly", "daily"
    enable_data_validation: bool = True
    fill_missing_data: bool = True
    data_quality_threshold: float = 0.95  # Minimum data quality
    
    # Regime detection
    enable_regime_detection: bool = True
    regime_detection_method: str = "hybrid"  # "tas", "nas", "hybrid"
    regime_confidence_threshold: float = 0.7
    
    # Model settings
    enable_model_selection: bool = True
    model_selection_strategy: str = "best_performance"  # "best_performance", "ensemble", "adaptive"
    enable_ensemble_trading: bool = True
    
    # Risk management
    enable_risk_management: bool = True
    max_position_size: float = 0.1  # 10% of capital
    stop_loss_pct: float = 0.02    # 2%
    take_profit_pct: float = 0.04   # 4%
    max_drawdown_limit: float = 0.15  # 15%
    
    # Performance analysis
    performance_metrics: List[PerformanceMetric] = field(default_factory=lambda: [
        PerformanceMetric.TOTAL_RETURN,
        PerformanceMetric.SHARPE_RATIO,
        PerformanceMetric.MAX_DRAWDOWN,
        PerformanceMetric.WIN_RATE
    ])
    enable_regime_analysis: bool = True
    enable_risk_analysis: bool = True
    
    # Output settings
    save_results: bool = True
    results_path: str = "backtesting_results"
    enable_visualization: bool = True
    enable_detailed_logging: bool = True
    
    # Advanced features
    enable_parallel_processing: bool = False
    max_workers: int = 4
    enable_caching: bool = True
    cache_path: str = "backtesting_cache"


@dataclass
class BacktestingResult:
    """Result from backtesting."""
    
    # Basic results
    success: bool
    execution_time: float
    start_date: datetime
    end_date: datetime
    total_periods: int
    
    # Performance metrics
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    win_rate: float
    profit_factor: float
    
    # Risk metrics
    var_95: float
    cvar_95: float
    volatility: float
    
    # Trading statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    average_win: float
    average_loss: float
    largest_win: float
    largest_loss: float
    
    # Regime analysis
    regime_performance: Dict[int, Dict[str, float]] = field(default_factory=dict)
    regime_transitions: List[Dict[str, Any]] = field(default_factory=list)
    
    # Model performance
    model_performance: Dict[str, Dict[str, float]] = field(default_factory=dict)
    model_selection_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # Error handling
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    
    # Metadata
    configuration: Dict[str, Any] = field(default_factory=dict)
    data_quality: Dict[str, float] = field(default_factory=dict)


class BacktestingEngine:
    """
    Comprehensive backtesting engine for NAS-TAS models.
    
    Provides regime-aware backtesting with performance analysis,
    risk assessment, and detailed reporting.
    """
    
    def __init__(self, config: BacktestingConfig):
        """Initialize backtesting engine.
        
        Args:
            config: Backtesting configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize components
        self._initialize_ml_common()
        self._initialize_regime_detection()
        self._initialize_optimized_tools()
        
        # Backtesting state
        self.current_capital = config.initial_capital
        self.positions = {}
        self.trade_history = []
        self.performance_history = []
        self.regime_history = []
        
        # Model registry
        self.available_models = {}
        self.model_performance = {}
        
        self.logger.info("✅ Backtesting Engine initialized")
        self.logger.info(f"   Mode: {config.mode.value}")
        self.logger.info(f"   Initial capital: ${config.initial_capital:,.2f}")
        self.logger.info(f"   Regime detection: {config.enable_regime_detection}")
        self.logger.info(f"   Risk management: {config.enable_risk_management}")
    
    def _initialize_ml_common(self):
        """Initialize ML common utilities."""
        if not ML_COMMON_AVAILABLE:
            self.logger.warning("⚠️ ML common utilities not available")
            self.ml_common_ops = None
            self.validation_framework = None
            return
        
        try:
            self.ml_common_ops = get_ml_common_operations()
            self.validation_framework = get_validation_framework()
            self.logger.info("✅ ML common utilities initialized")
        except Exception as e:
            self.logger.warning(f"ML common initialization failed: {e}")
            self.ml_common_ops = None
            self.validation_framework = None
    
    def _initialize_regime_detection(self):
        """Initialize regime detection systems."""
        if not REGIME_DETECTION_AVAILABLE or not self.config.enable_regime_detection:
            self.logger.warning("⚠️ Regime detection not available")
            self.tas_detector = None
            self.nas_detector = None
            return
        
        try:
            if self.config.regime_detection_method in ["tas", "hybrid"]:
                tas_config = TASRegimeConfig(
                    n_regimes=8,
                    enable_economic_evaluation=True,
                    enable_uncertainty_quantification=True
                )
                self.tas_detector = TASRegimeDetector(tas_config)
                self.logger.info("✅ TAS regime detector initialized")
            
            if self.config.regime_detection_method in ["nas", "hybrid"]:
                nas_config = PerfectNASConfig.create_short_term_trading_config()
                self.nas_detector = PerfectNASRegimeDetector(nas_config)
                self.logger.info("✅ NAS regime detector initialized")
                
        except Exception as e:
            self.logger.warning(f"Regime detection initialization failed: {e}")
            self.tas_detector = None
            self.nas_detector = None

    def _initialize_optimized_tools(self):
        """Initialize optimized ML pipeline tools."""
        if not OPTIMIZED_TOOLS_AVAILABLE:
            self.logger.warning("⚠️ Optimized ML pipeline tools not available")
            self.common_backtesting_engine = None
            self.walk_forward_validation = None
            self.monte_carlo_engine = None
            self.consolidated_backtesting = None
            return

        try:
            # Initialize common backtesting engine
            common_config = CommonBacktestingConfig(
                initial_capital=self.config.initial_capital,
                commission_rate=self.config.commission_rate,
                slippage_rate=self.config.slippage_rate,
                enable_data_validation=self.config.enable_data_validation,
                enable_risk_management=self.config.enable_risk_management
            )
            self.common_backtesting_engine = CommonBacktestingEngine(common_config)
            self.logger.info("✅ Common backtesting engine initialized")

            # Initialize walk-forward validation
            self.walk_forward_validation = WalkForwardValidation()
            self.logger.info("✅ Walk-forward validation initialized")

            # Initialize Monte Carlo simulation
            self.monte_carlo_engine = UnifiedMonteCarloEngine(MonteCarloConfig())
            self.logger.info("✅ Monte Carlo simulation initialized")

            # Initialize consolidated backtesting
            self.consolidated_backtesting = ConsolidatedBacktestingStep()
            self.logger.info("✅ Consolidated backtesting initialized")

        except Exception as e:
            self.logger.warning(f"Optimized tools initialization failed: {e}")
            self.common_backtesting_engine = None
            self.walk_forward_validation = None
            self.monte_carlo_engine = None
            self.consolidated_backtesting = None
    
    def register_models(self, 
                       regime_models: Dict[int, Dict[str, Any]],
                       ensemble_models: Optional[Dict[str, Any]] = None):
        """
        Register models for backtesting.
        
        Args:
            regime_models: Dictionary of regime_id -> {model_type: model_info}
            ensemble_models: Optional ensemble models
        """
        self.logger.info("📝 Registering models for backtesting")
        
        try:
            # Register regime models
            for regime_id, models in regime_models.items():
                self.available_models[regime_id] = {}
                
                for model_type, model_info in models.items():
                    self.available_models[regime_id][model_type] = {
                        'model': model_info['model'],
                        'performance': model_info.get('val_metrics', {}),
                        'feature_importance': model_info.get('feature_importance', {}),
                        'hyperparameters': model_info.get('hyperparameters', {})
                    }
                    
                    # Initialize performance tracking
                    model_id = f"regime_{regime_id}_{model_type}"
                    self.model_performance[model_id] = {
                        'predictions': 0,
                        'correct_predictions': 0,
                        'total_return': 0.0,
                        'trades': []
                    }
            
            # Register ensemble models
            if ensemble_models:
                self.available_models['ensemble'] = ensemble_models
            
            self.logger.info(f"✅ Registered models for {len(self.available_models)} regimes")
            
        except Exception as e:
            self.logger.error(f"❌ Model registration failed: {e}")
            raise
    
    def run_backtest(self, 
                    market_data: pd.DataFrame,
                    target_variable: str = 'close',
                    feature_columns: Optional[List[str]] = None) -> BacktestingResult:
        """
        Run comprehensive backtesting.
        
        Args:
            market_data: Historical market data
            target_variable: Target variable for prediction
            feature_columns: List of feature columns
            
        Returns:
            BacktestingResult with complete backtesting results
        """
        start_time = datetime.now()
        self.logger.info("🚀 Starting backtesting")
        
        try:
            # Validate and prepare data
            self.logger.info("📊 Preparing data for backtesting...")
            prepared_data = self._prepare_data(market_data, target_variable, feature_columns)
            
            if not prepared_data['success']:
                return BacktestingResult(
                    success=False,
                    execution_time=0.0,
                    start_date=datetime.now(),
                    end_date=datetime.now(),
                    total_periods=0,
                    error_message=prepared_data['error']
                )
            
            data = prepared_data['data']
            data_quality = prepared_data['data_quality']
            
            # Initialize backtesting state
            self._initialize_backtesting_state()
            
            # Run backtesting loop
            self.logger.info("🔄 Running backtesting loop...")
            backtest_results = self._run_backtesting_loop(data, target_variable)
            
            # Calculate performance metrics
            self.logger.info("📈 Calculating performance metrics...")
            performance_metrics = self._calculate_performance_metrics()
            
            # Analyze regime performance
            regime_analysis = {}
            if self.config.enable_regime_analysis:
                self.logger.info("🔍 Analyzing regime performance...")
                regime_analysis = self._analyze_regime_performance()
            
            # Analyze model performance
            model_analysis = {}
            if self.config.enable_model_selection:
                self.logger.info("🤖 Analyzing model performance...")
                model_analysis = self._analyze_model_performance()
            
            # Create result
            execution_time = (datetime.now() - start_time).total_seconds()
            result = BacktestingResult(
                success=True,
                execution_time=execution_time,
                start_date=data.index[0] if hasattr(data.index, '__getitem__') else datetime.now(),
                end_date=data.index[-1] if hasattr(data.index, '__getitem__') else datetime.now(),
                total_periods=len(data),
                **performance_metrics,
                regime_performance=regime_analysis.get('regime_performance', {}),
                regime_transitions=regime_analysis.get('transitions', []),
                model_performance=model_analysis.get('model_performance', {}),
                model_selection_history=model_analysis.get('selection_history', []),
                configuration=self._get_configuration_summary(),
                data_quality=data_quality
            )
            
            # Save results if requested
            if self.config.save_results:
                self.logger.info("💾 Saving backtesting results...")
                self._save_backtesting_results(result)
            
            self.logger.info(f"✅ Backtesting completed in {execution_time:.2f}s")
            self.logger.info(f"   Total return: {result.total_return:.2%}")
            self.logger.info(f"   Sharpe ratio: {result.sharpe_ratio:.3f}")
            self.logger.info(f"   Max drawdown: {result.max_drawdown:.2%}")
            self.logger.info(f"   Win rate: {result.win_rate:.2%}")
            
            return result
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            self.logger.error(f"❌ Backtesting failed: {e}")
            
            return BacktestingResult(
                success=False,
                execution_time=execution_time,
                start_date=datetime.now(),
                end_date=datetime.now(),
                total_periods=0,
                error_message=str(e)
            )
    
    def _prepare_data(self, 
                     market_data: pd.DataFrame,
                     target_variable: str,
                     feature_columns: Optional[List[str]]) -> Dict[str, Any]:
        """Prepare data for backtesting."""
        try:
            # Validate data
            if market_data.empty:
                return {'success': False, 'error': 'Empty dataset'}
            
            # Check required columns
            if target_variable not in market_data.columns:
                return {'success': False, 'error': f'Target variable {target_variable} not found'}
            
            # Determine feature columns
            if feature_columns is None:
                feature_columns = [col for col in market_data.columns if col != target_variable]
            
            # Check data quality
            missing_ratio = market_data.isnull().sum().sum() / (len(market_data) * len(market_data.columns))
            data_quality = {
                'completeness': 1.0 - missing_ratio,
                'total_records': len(market_data),
                'total_features': len(feature_columns)
            }
            
            if data_quality['completeness'] < self.config.data_quality_threshold:
                return {'success': False, 'error': f'Data quality too low: {data_quality["completeness"]:.3f}'}
            
            # Fill missing data if requested
            if self.config.fill_missing_data and missing_ratio > 0:
                market_data = market_data.fillna(method='ffill').fillna(method='bfill')
                self.logger.info(f"📊 Filled {missing_ratio:.1%} missing data")
            
            # Sort by index if datetime
            if hasattr(market_data.index, 'sort_values'):
                market_data = market_data.sort_index()
            
            return {
                'success': True,
                'data': market_data,
                'data_quality': data_quality
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _initialize_backtesting_state(self):
        """Initialize backtesting state."""
        self.current_capital = self.config.initial_capital
        self.positions = {}
        self.trade_history = []
        self.performance_history = []
        self.regime_history = []
    
    def _run_backtesting_loop(self, 
                             data: pd.DataFrame,
                             target_variable: str) -> Dict[str, Any]:
        """Run the main backtesting loop."""
        try:
            total_periods = len(data)
            
            for i in range(1, total_periods):  # Start from 1 to have previous data
                current_data = data.iloc[:i+1]
                current_period = data.index[i]
                
                # Detect regime
                current_regime = self._detect_current_regime(current_data)
                self.regime_history.append({
                    'period': current_period,
                    'regime': current_regime
                })
                
                # Select model for regime
                selected_model = self._select_model_for_regime(current_regime)
                
                if selected_model is None:
                    continue
                
                # Make prediction
                prediction_result = self._make_prediction(
                    current_data, selected_model, target_variable
                )
                
                if prediction_result is None:
                    continue
                
                # Execute trading decision
                trading_result = self._execute_trading_decision(
                    prediction_result, current_period, data.iloc[i]
                )
                
                # Update performance tracking
                self._update_performance_tracking(
                    selected_model, prediction_result, trading_result
                )
                
                # Update capital and positions
                self._update_capital_and_positions(trading_result)
                
                # Record performance
                self.performance_history.append({
                    'period': current_period,
                    'capital': self.current_capital,
                    'regime': current_regime,
                    'model': selected_model['model_type'],
                    'prediction': prediction_result['prediction'],
                    'confidence': prediction_result['confidence'],
                    'trade_result': trading_result
                })
            
            return {'success': True, 'total_periods': total_periods}
            
        except Exception as e:
            self.logger.error(f"❌ Backtesting loop failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _detect_current_regime(self, data: pd.DataFrame) -> int:
        """Detect current market regime."""
        try:
            if not self.config.enable_regime_detection:
                return 0  # Default regime
            
            # Use TAS detector
            if self.tas_detector and self.config.regime_detection_method in ["tas", "hybrid"]:
                result = self.tas_detector.detect_regimes(data)
                if result.success:
                    return result.regime_predictions[-1]
            
            # Use NAS detector
            if self.nas_detector and self.config.regime_detection_method in ["nas", "hybrid"]:
                result = self.nas_detector.detect_regimes(data)
                if result.success:
                    return result.regime_predictions[-1]
            
            # Fallback to simple regime detection
            return self._fallback_regime_detection(data)
            
        except Exception as e:
            self.logger.warning(f"Regime detection failed: {e}")
            return 0
    
    def _fallback_regime_detection(self, data: pd.DataFrame) -> int:
        """Fallback regime detection using simple heuristics."""
        try:
            if 'close' in data.columns and len(data) > 1:
                prices = data['close'].values
                returns = np.diff(prices) / prices[:-1]
                volatility = np.std(returns)
                
                if volatility < 0.01:
                    return 0  # Low volatility regime
                elif volatility < 0.03:
                    return 1  # Medium volatility regime
                else:
                    return 2  # High volatility regime
            else:
                return 0  # Default regime
                
        except Exception as e:
            self.logger.warning(f"Fallback regime detection failed: {e}")
            return 0
    
    def _select_model_for_regime(self, regime_id: int) -> Optional[Dict[str, Any]]:
        """Select model for current regime."""
        try:
            if regime_id not in self.available_models:
                return None
            
            regime_models = self.available_models[regime_id]
            
            if self.config.model_selection_strategy == "best_performance":
                # Select best performing model
                best_model_type = max(
                    regime_models.keys(),
                    key=lambda x: regime_models[x]['performance'].get('f1_score', 0.0)
                )
            elif self.config.model_selection_strategy == "ensemble":
                # Use ensemble (simplified - just pick first model)
                best_model_type = list(regime_models.keys())[0]
            else:
                # Default to first available model
                best_model_type = list(regime_models.keys())[0]
            
            return {
                'model': regime_models[best_model_type]['model'],
                'model_type': best_model_type,
                'regime_id': regime_id,
                'performance': regime_models[best_model_type]['performance']
            }
            
        except Exception as e:
            self.logger.warning(f"Model selection failed: {e}")
            return None
    
    def _make_prediction(self, 
                       data: pd.DataFrame,
                       selected_model: Dict[str, Any],
                       target_variable: str) -> Optional[Dict[str, Any]]:
        """Make prediction using selected model."""
        try:
            model = selected_model['model']
            
            # Prepare features (exclude target variable)
            feature_columns = [col for col in data.columns if col != target_variable]
            X = data[feature_columns].iloc[-1:].values  # Last row
            
            # Make prediction
            if hasattr(model, 'predict'):
                prediction = model.predict(X)[0]
            else:
                return None
            
            # Get confidence if available
            confidence = 0.5  # Default confidence
            if hasattr(model, 'predict_proba'):
                try:
                    proba = model.predict_proba(X)[0]
                    confidence = np.max(proba)
                except:
                    pass
            
            return {
                'prediction': prediction,
                'confidence': confidence,
                'model_type': selected_model['model_type'],
                'regime_id': selected_model['regime_id']
            }
            
        except Exception as e:
            self.logger.warning(f"Prediction failed: {e}")
            return None
    
    def _execute_trading_decision(self, 
                                prediction_result: Dict[str, Any],
                                current_period: Any,
                                current_data: pd.Series) -> Dict[str, Any]:
        """Execute trading decision based on prediction."""
        try:
            prediction = prediction_result['prediction']
            confidence = prediction_result['confidence']
            
            # Simple trading logic (buy/sell/hold)
            if confidence < 0.6:  # Low confidence
                action = 'hold'
            elif prediction > 0.5:  # Positive prediction
                action = 'buy'
            else:  # Negative prediction
                action = 'sell'
            
            # Calculate position size
            position_size = self._calculate_position_size(confidence)
            
            # Apply risk management
            if self.config.enable_risk_management:
                position_size = self._apply_risk_management(position_size, current_data)
            
            return {
                'action': action,
                'position_size': position_size,
                'confidence': confidence,
                'timestamp': current_period
            }
            
        except Exception as e:
            self.logger.warning(f"Trading decision execution failed: {e}")
            return {'action': 'hold', 'position_size': 0.0, 'confidence': 0.0}
    
    def _calculate_position_size(self, confidence: float) -> float:
        """Calculate position size based on confidence."""
        # Simple position sizing based on confidence
        base_size = self.config.max_position_size
        confidence_multiplier = confidence
        return base_size * confidence_multiplier
    
    def _apply_risk_management(self, position_size: float, current_data: pd.Series) -> float:
        """Apply risk management rules."""
        # Check maximum position size
        position_size = min(position_size, self.config.max_position_size)
        
        # Check current drawdown
        if hasattr(self, 'performance_history') and self.performance_history:
            peak_capital = max([p['capital'] for p in self.performance_history])
            current_drawdown = (peak_capital - self.current_capital) / peak_capital
            
            if current_drawdown > self.config.max_drawdown_limit:
                position_size *= 0.5  # Reduce position size during high drawdown
        
        return position_size
    
    def _update_performance_tracking(self, 
                                   selected_model: Dict[str, Any],
                                   prediction_result: Dict[str, Any],
                                   trading_result: Dict[str, Any]):
        """Update performance tracking for model."""
        try:
            model_id = f"regime_{selected_model['regime_id']}_{selected_model['model_type']}"
            
            if model_id not in self.model_performance:
                self.model_performance[model_id] = {
                    'predictions': 0,
                    'correct_predictions': 0,
                    'total_return': 0.0,
                    'trades': []
                }
            
            # Update prediction count
            self.model_performance[model_id]['predictions'] += 1
            
            # Record trade
            trade_record = {
                'timestamp': trading_result['timestamp'],
                'action': trading_result['action'],
                'position_size': trading_result['position_size'],
                'confidence': trading_result['confidence']
            }
            self.model_performance[model_id]['trades'].append(trade_record)
            
        except Exception as e:
            self.logger.warning(f"Performance tracking update failed: {e}")
    
    def _update_capital_and_positions(self, trading_result: Dict[str, Any]):
        """Update capital and positions based on trading result."""
        try:
            action = trading_result['action']
            position_size = trading_result['position_size']
            
            if action == 'buy' and position_size > 0:
                # Execute buy order
                trade_value = self.current_capital * position_size
                self.current_capital -= trade_value
                # In a real implementation, you would track actual positions
            
            elif action == 'sell' and position_size > 0:
                # Execute sell order
                trade_value = self.current_capital * position_size
                self.current_capital += trade_value
                # In a real implementation, you would track actual positions
            
            # Record trade
            self.trade_history.append({
                'timestamp': trading_result['timestamp'],
                'action': action,
                'position_size': position_size,
                'capital': self.current_capital
            })
            
        except Exception as e:
            self.logger.warning(f"Capital update failed: {e}")
    
    def _calculate_performance_metrics(self) -> Dict[str, float]:
        """Calculate comprehensive performance metrics."""
        try:
            if not self.performance_history:
                return self._get_default_metrics()
            
            # Extract capital history
            capital_history = [p['capital'] for p in self.performance_history]
            returns = np.diff(capital_history) / capital_history[:-1]
            
            # Basic metrics
            total_return = (capital_history[-1] - capital_history[0]) / capital_history[0]
            annualized_return = (1 + total_return) ** (252 / len(capital_history)) - 1
            
            # Risk metrics
            volatility = np.std(returns) * np.sqrt(252)
            sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
            
            # Drawdown analysis
            peak = np.maximum.accumulate(capital_history)
            drawdown = (capital_history - peak) / peak
            max_drawdown = np.min(drawdown)
            
            # Sortino ratio
            negative_returns = returns[returns < 0]
            downside_volatility = np.std(negative_returns) * np.sqrt(252) if len(negative_returns) > 0 else 0
            sortino_ratio = annualized_return / downside_volatility if downside_volatility > 0 else 0
            
            # Calmar ratio
            calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
            
            # Trading statistics
            trades = [t for t in self.trade_history if t['action'] != 'hold']
            winning_trades = [t for t in trades if t['capital'] > self.config.initial_capital]
            win_rate = len(winning_trades) / len(trades) if trades else 0
            
            # VaR and CVaR
            var_95 = np.percentile(returns, 5) if len(returns) > 0 else 0
            cvar_95 = np.mean(returns[returns <= var_95]) if len(returns) > 0 else 0
            
            return {
                'total_return': total_return,
                'annualized_return': annualized_return,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'max_drawdown': max_drawdown,
                'calmar_ratio': calmar_ratio,
                'win_rate': win_rate,
                'profit_factor': 1.0,  # Simplified
                'var_95': var_95,
                'cvar_95': cvar_95,
                'volatility': volatility,
                'total_trades': len(trades),
                'winning_trades': len(winning_trades),
                'losing_trades': len(trades) - len(winning_trades),
                'average_win': 0.0,  # Simplified
                'average_loss': 0.0,  # Simplified
                'largest_win': 0.0,  # Simplified
                'largest_loss': 0.0  # Simplified
            }
            
        except Exception as e:
            self.logger.error(f"❌ Performance metrics calculation failed: {e}")
            return self._get_default_metrics()
    
    def _get_default_metrics(self) -> Dict[str, float]:
        """Get default metrics when calculation fails."""
        return {
            'total_return': 0.0,
            'annualized_return': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'max_drawdown': 0.0,
            'calmar_ratio': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'var_95': 0.0,
            'cvar_95': 0.0,
            'volatility': 0.0,
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'average_win': 0.0,
            'average_loss': 0.0,
            'largest_win': 0.0,
            'largest_loss': 0.0
        }
    
    def _analyze_regime_performance(self) -> Dict[str, Any]:
        """Analyze performance by regime."""
        try:
            regime_performance = {}
            
            # Group performance by regime
            for regime_id in set([p['regime'] for p in self.performance_history]):
                regime_periods = [p for p in self.performance_history if p['regime'] == regime_id]
                
                if not regime_periods:
                    continue
                
                # Calculate regime-specific metrics
                regime_capital = [p['capital'] for p in regime_periods]
                regime_returns = np.diff(regime_capital) / regime_capital[:-1] if len(regime_capital) > 1 else [0]
                
                regime_performance[regime_id] = {
                    'periods': len(regime_periods),
                    'total_return': (regime_capital[-1] - regime_capital[0]) / regime_capital[0] if len(regime_capital) > 1 else 0,
                    'average_return': np.mean(regime_returns),
                    'volatility': np.std(regime_returns),
                    'sharpe_ratio': np.mean(regime_returns) / np.std(regime_returns) if np.std(regime_returns) > 0 else 0
                }
            
            # Analyze regime transitions
            transitions = []
            for i in range(1, len(self.regime_history)):
                if self.regime_history[i]['regime'] != self.regime_history[i-1]['regime']:
                    transitions.append({
                        'from_regime': self.regime_history[i-1]['regime'],
                        'to_regime': self.regime_history[i]['regime'],
                        'timestamp': self.regime_history[i]['period']
                    })
            
            return {
                'regime_performance': regime_performance,
                'transitions': transitions
            }
            
        except Exception as e:
            self.logger.error(f"❌ Regime analysis failed: {e}")
            return {'regime_performance': {}, 'transitions': []}
    
    def _analyze_model_performance(self) -> Dict[str, Any]:
        """Analyze model performance."""
        try:
            model_performance = {}
            selection_history = []
            
            for model_id, performance in self.model_performance.items():
                if performance['predictions'] > 0:
                    accuracy = performance['correct_predictions'] / performance['predictions']
                    model_performance[model_id] = {
                        'predictions': performance['predictions'],
                        'accuracy': accuracy,
                        'total_return': performance['total_return'],
                        'trades': len(performance['trades'])
                    }
            
            # Model selection history
            for period in self.performance_history:
                selection_history.append({
                    'timestamp': period['period'],
                    'regime': period['regime'],
                    'model': period['model'],
                    'confidence': period['confidence']
                })
            
            return {
                'model_performance': model_performance,
                'selection_history': selection_history
            }
            
        except Exception as e:
            self.logger.error(f"❌ Model analysis failed: {e}")
            return {'model_performance': {}, 'selection_history': []}
    
    def _get_configuration_summary(self) -> Dict[str, Any]:
        """Get configuration summary."""
        return {
            'mode': self.config.mode.value,
            'initial_capital': self.config.initial_capital,
            'commission_rate': self.config.commission_rate,
            'slippage_rate': self.config.slippage_rate,
            'enable_regime_detection': self.config.enable_regime_detection,
            'regime_detection_method': self.config.regime_detection_method,
            'enable_model_selection': self.config.enable_model_selection,
            'model_selection_strategy': self.config.model_selection_strategy,
            'enable_risk_management': self.config.enable_risk_management,
            'max_position_size': self.config.max_position_size,
            'max_drawdown_limit': self.config.max_drawdown_limit
        }
    
    def _save_backtesting_results(self, result: BacktestingResult):
        """Save backtesting results."""
        try:
            results_path = Path(self.config.results_path)
            results_path.mkdir(parents=True, exist_ok=True)
            
            # Save result summary
            result_summary = {
                'success': result.success,
                'execution_time': result.execution_time,
                'start_date': result.start_date.isoformat(),
                'end_date': result.end_date.isoformat(),
                'total_periods': result.total_periods,
                'total_return': result.total_return,
                'annualized_return': result.annualized_return,
                'sharpe_ratio': result.sharpe_ratio,
                'max_drawdown': result.max_drawdown,
                'win_rate': result.win_rate,
                'total_trades': result.total_trades,
                'configuration': result.configuration,
                'data_quality': result.data_quality
            }
            
            with open(results_path / "backtesting_summary.json", 'w') as f:
                json.dump(result_summary, f, indent=2)
            
            # Save detailed results
            with open(results_path / "backtesting_result.pkl", 'wb') as f:
                pickle.dump(result, f)
            
            self.logger.info(f"💾 Backtesting results saved to {results_path}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save backtesting results: {e}")
    
    def get_backtesting_summary(self) -> Dict[str, Any]:
        """Get summary of backtesting results."""
        if not self.performance_history:
            return {'error': 'No backtesting data available'}
        
        return {
            'total_periods': len(self.performance_history),
            'total_trades': len(self.trade_history),
            'current_capital': self.current_capital,
            'total_return': (self.current_capital - self.config.initial_capital) / self.config.initial_capital,
            'regimes_detected': len(set([p['regime'] for p in self.performance_history])),
            'models_used': len(set([p['model'] for p in self.performance_history]))
        }