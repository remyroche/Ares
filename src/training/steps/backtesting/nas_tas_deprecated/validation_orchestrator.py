"""
Validation Orchestrator

Comprehensive validation orchestrator that coordinates backtesting,
walk-forward analysis, performance attribution, and scenario testing
for the NAS-TAS system.
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

# Import validation components
from .backtesting_engine import BacktestingEngine, BacktestingConfig, BacktestingResult
from .walk_forward_analyzer import WalkForwardAnalyzer, WalkForwardConfig, WalkForwardResult
from .performance_attribution import PerformanceAttributor, AttributionConfig, AttributionResult
from .scenario_tester import ScenarioTester, ScenarioConfig, ScenarioResult

# Import unified NAS/TAS tools
try:
    from src.nas_tas.data.data_processor import UnifiedDataProcessor, DataProcessingConfig
    from src.nas_tas.evaluation.unified_evaluator import UnifiedEvaluator, EvaluationConfig
    from src.nas_tas.config.base_config import UnifiedArchitectureConfig, create_comprehensive_config
    from src.nas_tas.error_handling import UnifiedErrorHandler
    from src.nas_tas.logging import UnifiedLogger, LoggingConfig
    UNIFIED_TOOLS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Unified NAS/TAS tools not available: {e}")
    UNIFIED_TOOLS_AVAILABLE = False

# Import VectorBT optimization tools
try:
    from src.feature_generation.utils.vectorbt_rolling_optimizer import get_vectorbt_rolling_optimizer
    from src.feature_generation.utils.vectorbt_optimization_integration import get_optimization_manager
    VECTORBT_AVAILABLE = True
except ImportError as e:
    print(f"Warning: VectorBT optimization tools not available: {e}")
    VECTORBT_AVAILABLE = False

logger = logging.getLogger(__name__)


class ValidationMode(Enum):
    """Validation modes."""
    COMPREHENSIVE = "comprehensive"  # All validation methods
    BACKTESTING_ONLY = "backtesting_only"  # Only backtesting
    WALK_FORWARD_ONLY = "walk_forward_only"  # Only walk-forward
    ATTRIBUTION_ONLY = "attribution_only"  # Only attribution
    SCENARIO_ONLY = "scenario_only"  # Only scenario testing


@dataclass
class ValidationConfig:
    """Configuration for validation orchestrator."""
    
    # Validation mode
    mode: ValidationMode = ValidationMode.COMPREHENSIVE
    
    # Component configurations
    backtesting_config: BacktestingConfig = field(default_factory=BacktestingConfig)
    walk_forward_config: WalkForwardConfig = field(default_factory=WalkForwardConfig)
    attribution_config: AttributionConfig = field(default_factory=AttributionConfig)
    scenario_config: ScenarioConfig = field(default_factory=ScenarioConfig)
    
    # Validation settings
    enable_backtesting: bool = True
    enable_walk_forward: bool = True
    enable_attribution: bool = True
    enable_scenario_testing: bool = True
    
    # Data settings
    data_validation: bool = True
    data_preprocessing: bool = True
    feature_engineering: bool = True
    
    # Performance settings
    performance_threshold: float = 0.6  # Minimum performance threshold
    risk_threshold: float = 0.15  # Maximum risk threshold
    enable_risk_adjustment: bool = True
    
    # Output settings
    save_results: bool = True
    results_path: str = "validation_results"
    enable_detailed_logging: bool = True
    enable_visualization: bool = True
    
    # Advanced features
    enable_parallel_processing: bool = False
    max_workers: int = 4
    enable_caching: bool = True
    cache_path: str = "validation_cache"


@dataclass
class ValidationResult:
    """Result from validation orchestration."""
    
    # Overall results
    success: bool
    execution_time: float
    mode: ValidationMode
    
    # Component results
    backtesting_result: Optional[BacktestingResult] = None
    walk_forward_result: Optional[WalkForwardResult] = None
    attribution_result: Optional[AttributionResult] = None
    scenario_result: Optional[ScenarioResult] = None
    
    # Validation metrics
    overall_performance: Dict[str, float] = field(default_factory=dict)
    risk_metrics: Dict[str, float] = field(default_factory=dict)
    stability_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Validation insights
    performance_insights: List[str] = field(default_factory=list)
    risk_insights: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    # Error handling
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    
    # Metadata
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    configuration: Optional[Dict[str, Any]] = None


class ValidationOrchestrator:
    """
    Validation orchestrator for NAS-TAS models.
    
    Coordinates comprehensive validation including backtesting,
    walk-forward analysis, performance attribution, and scenario testing.
    """
    
    def __init__(self, config: ValidationConfig):
        """Initialize validation orchestrator.
        
        Args:
            config: Validation configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize unified tools first
        self._initialize_unified_tools()
        
        # Initialize VectorBT optimization tools
        self._initialize_vectorbt_optimization()
        
        # Initialize components
        self._initialize_components()
        
        # Validation state
        self.validation_results = {}
        self.performance_history = []
        self.risk_history = []
        
        self.logger.info("✅ Validation Orchestrator initialized")
        self.logger.info(f"   Mode: {config.mode.value}")
        self.logger.info(f"   Components enabled:")
        self.logger.info(f"     - Backtesting: {config.enable_backtesting}")
        self.logger.info(f"     - Walk-forward: {config.enable_walk_forward}")
        self.logger.info(f"     - Attribution: {config.enable_attribution}")
        self.logger.info(f"     - Scenario testing: {config.enable_scenario_testing}")
        self.logger.info(f"   Unified tools available: {UNIFIED_TOOLS_AVAILABLE}")
        self.logger.info(f"   VectorBT optimization available: {VECTORBT_AVAILABLE}")
    
    def _initialize_vectorbt_optimization(self):
        """Initialize VectorBT optimization tools."""
        if not VECTORBT_AVAILABLE:
            self.logger.warning("⚠️ VectorBT optimization tools not available")
            self.rolling_optimizer = None
            self.optimization_manager = None
            return
        
        try:
            # Initialize VectorBT rolling optimizer
            self.rolling_optimizer = get_vectorbt_rolling_optimizer(
                enable_gpu=config.get('enable_gpu', False),
                enable_parallel=config.get('enable_parallel', True),
                memory_efficient=True,
                chunk_size=config.get('chunk_size', 1000),
                fast_fail=True,
                enable_logging=True
            )
            self.logger.info("✅ VectorBT rolling optimizer initialized")
            
            # Initialize VectorBT optimization manager
            self.optimization_manager = get_optimization_manager(
                enable_gpu=config.get('enable_gpu', False),
                enable_parallel=config.get('enable_parallel', True),
                memory_efficient=True,
                max_memory_gb=config.get('max_memory_gb', 8.0),
                chunk_size=config.get('chunk_size', 1000),
                enable_monitoring=True
            )
            self.logger.info("✅ VectorBT optimization manager initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize VectorBT optimization: {e}")
            self.rolling_optimizer = None
            self.optimization_manager = None
    
    def _initialize_unified_tools(self):
        """Initialize unified NAS/TAS tools."""
        if not UNIFIED_TOOLS_AVAILABLE:
            self.logger.warning("⚠️ Unified NAS/TAS tools not available")
            self.unified_data_processor = None
            self.unified_evaluator = None
            self.unified_error_handler = None
            self.unified_logger = None
            return
        
        try:
            # Initialize unified data processor
            data_config = DataProcessingConfig(
                handle_missing_values=True,
                missing_value_strategy="median",
                handle_outliers=True,
                outlier_method="iqr",
                enable_scaling=True,
                scaling_method="standard",
                enable_feature_engineering=True,
                validate_data=True,
                min_data_quality_score=0.8
            )
            self.unified_data_processor = UnifiedDataProcessor(data_config)
            self.logger.info("✅ Unified data processor initialized")
            
            # Initialize unified evaluator
            eval_config = EvaluationConfig(
                evaluation_type="comprehensive",
                calculate_performance_metrics=True,
                calculate_financial_metrics=True,
                calculate_regime_metrics=True,
                calculate_risk_metrics=True,
                financial_validation=True,
                enable_parallel_evaluation=True,
                max_workers=self.config.max_workers
            )
            self.unified_evaluator = UnifiedEvaluator(eval_config)
            self.logger.info("✅ Unified evaluator initialized")
            
            # Initialize unified error handler
            self.unified_error_handler = UnifiedErrorHandler()
            self.logger.info("✅ Unified error handler initialized")
            
            # Initialize unified logger
            logging_config = LoggingConfig(
                log_level="INFO",
                enable_file_logging=True,
                enable_console_logging=True,
                log_format="detailed"
            )
            self.unified_logger = UnifiedLogger(logging_config)
            self.logger.info("✅ Unified logger initialized")
            
        except Exception as e:
            self.logger.warning(f"Unified tools initialization failed: {e}")
            self.unified_data_processor = None
            self.unified_evaluator = None
            self.unified_error_handler = None
            self.unified_logger = None
    
    def _initialize_components(self):
        """Initialize validation components."""
        try:
            # Initialize backtesting engine
            if self.config.enable_backtesting:
                self.backtesting_engine = BacktestingEngine(self.config.backtesting_config)
                self.logger.info("✅ Backtesting engine initialized")
            else:
                self.backtesting_engine = None
            
            # Initialize walk-forward analyzer
            if self.config.enable_walk_forward:
                self.walk_forward_analyzer = WalkForwardAnalyzer(self.config.walk_forward_config)
                self.logger.info("✅ Walk-forward analyzer initialized")
            else:
                self.walk_forward_analyzer = None
            
            # Initialize performance attributor
            if self.config.enable_attribution:
                self.performance_attributor = PerformanceAttributor(self.config.attribution_config)
                self.logger.info("✅ Performance attributor initialized")
            else:
                self.performance_attributor = None
            
            # Initialize scenario tester
            if self.config.enable_scenario_testing:
                self.scenario_tester = ScenarioTester(self.config.scenario_config)
                self.logger.info("✅ Scenario tester initialized")
            else:
                self.scenario_tester = None
            
        except Exception as e:
            self.logger.error(f"❌ Component initialization failed: {e}")
            raise
    
    def run_validation(self, 
                      market_data: pd.DataFrame,
                      target_variable: str = 'close',
                      feature_columns: Optional[List[str]] = None,
                      regime_models: Optional[Dict[int, Dict[str, Any]]] = None,
                      ensemble_models: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """
        Run comprehensive validation.
        
        Args:
            market_data: Historical market data
            target_variable: Target variable for prediction
            feature_columns: List of feature columns
            regime_models: Trained regime models
            ensemble_models: Ensemble models
            
        Returns:
            ValidationResult with complete validation results
        """
        start_time = datetime.now()
        self.logger.info("🚀 Starting comprehensive validation")
        
        try:
            # Initialize result
            result = ValidationResult(
                success=False,
                execution_time=0.0,
                mode=self.config.mode,
                start_time=start_time
            )
            
            # Step 1: Data validation and preprocessing
            if self.config.data_validation or self.config.data_preprocessing:
                self.logger.info("📊 Validating and preprocessing data...")
                processed_data = self._validate_and_preprocess_data(
                    market_data, target_variable, feature_columns
                )
            else:
                processed_data = market_data
            
            # Step 2: Feature engineering
            if self.config.feature_engineering:
                self.logger.info("🔧 Performing feature engineering...")
                processed_data = self._perform_feature_engineering(processed_data, target_variable)
            
            # Step 3: Register models with components
            if regime_models:
                self.logger.info("📝 Registering models with validation components...")
                self._register_models_with_components(regime_models, ensemble_models)
            
            # Step 4: Run backtesting
            backtesting_result = None
            if self.config.enable_backtesting and self.backtesting_engine:
                self.logger.info("🔄 Running backtesting...")
                backtesting_result = self._run_backtesting(processed_data, target_variable, feature_columns)
                result.backtesting_result = backtesting_result
            
            # Step 5: Run walk-forward analysis
            walk_forward_result = None
            if self.config.enable_walk_forward and self.walk_forward_analyzer:
                self.logger.info("🔄 Running walk-forward analysis...")
                walk_forward_result = self._run_walk_forward_analysis(processed_data, target_variable, feature_columns)
                result.walk_forward_result = walk_forward_result
            
            # Step 6: Run performance attribution
            attribution_result = None
            if self.config.enable_attribution and self.performance_attributor:
                self.logger.info("📈 Running performance attribution...")
                attribution_result = self._run_performance_attribution(
                    backtesting_result, walk_forward_result, processed_data
                )
                result.attribution_result = attribution_result
            
            # Step 7: Run scenario testing
            scenario_result = None
            if self.config.enable_scenario_testing and self.scenario_tester:
                self.logger.info("🧪 Running scenario testing...")
                scenario_result = self._run_scenario_testing(processed_data, target_variable, feature_columns)
                result.scenario_result = scenario_result
            
            # Step 8: Analyze validation results
            self.logger.info("📊 Analyzing validation results...")
            analysis_results = self._analyze_validation_results(
                backtesting_result, walk_forward_result, attribution_result, scenario_result
            )
            
            # Update result with analysis
            result.overall_performance = analysis_results['overall_performance']
            result.risk_metrics = analysis_results['risk_metrics']
            result.stability_metrics = analysis_results['stability_metrics']
            result.performance_insights = analysis_results['performance_insights']
            result.risk_insights = analysis_results['risk_insights']
            result.recommendations = analysis_results['recommendations']
            
            # Step 9: Save results
            if self.config.save_results:
                self.logger.info("💾 Saving validation results...")
                self._save_validation_results(result)
            
            # Complete result
            end_time = datetime.now()
            result.end_time = end_time
            result.execution_time = (end_time - start_time).total_seconds()
            result.success = True
            result.configuration = self._get_configuration_summary()
            
            self.logger.info(f"✅ Validation completed in {result.execution_time:.2f}s")
            self.logger.info(f"   Overall performance: {result.overall_performance}")
            self.logger.info(f"   Risk metrics: {result.risk_metrics}")
            self.logger.info(f"   Recommendations: {len(result.recommendations)}")
            
            return result
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            self.logger.error(f"❌ Validation failed: {e}")
            
            return ValidationResult(
                success=False,
                execution_time=execution_time,
                mode=self.config.mode,
                error_message=str(e),
                start_time=start_time,
                end_time=datetime.now()
            )
    
    def _validate_and_preprocess_data(self, 
                                    market_data: pd.DataFrame,
                                    target_variable: str,
                                    feature_columns: Optional[List[str]]) -> pd.DataFrame:
        """Validate and preprocess market data."""
        try:
            # Check if target variable exists
            if target_variable not in market_data.columns:
                raise ValueError(f"Target variable '{target_variable}' not found in data")
            
            # Determine feature columns
            if feature_columns is None:
                feature_columns = [col for col in market_data.columns if col != target_variable]
            
            # Check for missing values
            missing_values = market_data.isnull().sum()
            if missing_values.any():
                self.logger.warning(f"⚠️ Found missing values: {missing_values[missing_values > 0].to_dict()}")
                # Fill missing values with forward fill
                market_data = market_data.fillna(method='ffill').fillna(method='bfill')
            
            # Check for infinite values
            inf_values = np.isinf(market_data.select_dtypes(include=[np.number])).sum()
            if inf_values.any():
                self.logger.warning(f"⚠️ Found infinite values: {inf_values[inf_values > 0].to_dict()}")
                # Replace infinite values with NaN and fill
                market_data = market_data.replace([np.inf, -np.inf], np.nan)
                market_data = market_data.fillna(method='ffill').fillna(method='bfill')
            
            self.logger.info(f"✅ Data validation completed - Shape: {market_data.shape}")
            return market_data
            
        except Exception as e:
            self.logger.error(f"❌ Data validation failed: {e}")
            raise
    
    def _perform_feature_engineering(self, 
                                   market_data: pd.DataFrame,
                                   target_variable: str) -> pd.DataFrame:
        """Perform feature engineering on market data."""
        try:
            # Create a copy to avoid modifying original data
            data = market_data.copy()
            
            # Technical indicators - use VectorBT optimization if available
            if 'close' in data.columns:
                # Price-based features
                data['price_change'] = data['close'].pct_change()
                
                if self.rolling_optimizer is not None:
                    # Use VectorBT rolling optimizer for better performance
                    data['price_volatility'] = self.rolling_optimizer.rolling_std(data['price_change'], window=20)
                    data['price_momentum'] = data['close'] / data['close'].shift(20)
                    
                    # Moving averages
                    data['ma_5'] = self.rolling_optimizer.rolling_mean(data['close'], window=5)
                    data['ma_20'] = self.rolling_optimizer.rolling_mean(data['close'], window=20)
                    data['ma_50'] = self.rolling_optimizer.rolling_mean(data['close'], window=50)
                    
                    # Price position using VectorBT rolling min/max
                    rolling_min = self.rolling_optimizer.rolling_min(data['close'], window=20)
                    rolling_max = self.rolling_optimizer.rolling_max(data['close'], window=20)
                    data['price_position_20'] = (data['close'] - rolling_min) / (rolling_max - rolling_min)
                else:
                    # Fallback to pandas rolling operations
                    data['price_volatility'] = data['price_change'].rolling(window=20).std()
                    data['price_momentum'] = data['close'] / data['close'].shift(20)
                    
                    # Moving averages
                    data['ma_5'] = data['close'].rolling(window=5).mean()
                    data['ma_20'] = data['close'].rolling(window=20).mean()
                    data['ma_50'] = data['close'].rolling(window=50).mean()
                    
                    # Price position
                    data['price_position_20'] = (data['close'] - data['close'].rolling(window=20).min()) / (data['close'].rolling(window=20).max() - data['close'].rolling(window=20).min())
            
            if 'volume' in data.columns:
                # Volume-based features
                data['volume_change'] = data['volume'].pct_change()
                
                if self.rolling_optimizer is not None:
                    # Use VectorBT rolling optimizer
                    data['volume_ma'] = self.rolling_optimizer.rolling_mean(data['volume'], window=20)
                else:
                    # Fallback to pandas
                    data['volume_ma'] = data['volume'].rolling(window=20).mean()
                
                data['volume_ratio'] = data['volume'] / data['volume_ma']
            
            if 'high' in data.columns and 'low' in data.columns:
                # Range-based features
                data['price_range'] = (data['high'] - data['low']) / data['close']
                
                if self.rolling_optimizer is not None:
                    # Use VectorBT rolling optimizer
                    data['range_volatility'] = self.rolling_optimizer.rolling_std(data['price_range'], window=20)
                else:
                    # Fallback to pandas
                    data['range_volatility'] = data['price_range'].rolling(window=20).std()
            
            # Time-based features
            if data.index.dtype == 'datetime64[ns]':
                data['hour'] = data.index.hour
                data['day_of_week'] = data.index.dayofweek
                data['month'] = data.index.month
            
            # Remove rows with NaN values created by rolling operations
            data = data.dropna()
            
            self.logger.info(f"✅ Feature engineering completed - New shape: {data.shape}")
            return data
            
        except Exception as e:
            self.logger.error(f"❌ Feature engineering failed: {e}")
            return market_data  # Return original data if engineering fails
    
    def _register_models_with_components(self, 
                                       regime_models: Dict[int, Dict[str, Any]],
                                       ensemble_models: Optional[Dict[str, Any]]):
        """Register models with validation components."""
        try:
            # Register with backtesting engine
            if self.backtesting_engine:
                self.backtesting_engine.register_models(regime_models, ensemble_models)
            
            # Register with walk-forward analyzer
            if self.walk_forward_analyzer:
                self.walk_forward_analyzer.register_models(regime_models, ensemble_models)
            
            # Register with scenario tester
            if self.scenario_tester:
                self.scenario_tester.register_models(regime_models, ensemble_models)
            
            self.logger.info("✅ Models registered with validation components")
            
        except Exception as e:
            self.logger.error(f"❌ Model registration failed: {e}")
            raise
    
    def _run_backtesting(self, 
                        market_data: pd.DataFrame,
                        target_variable: str,
                        feature_columns: Optional[List[str]]) -> BacktestingResult:
        """Run backtesting validation."""
        try:
            return self.backtesting_engine.run_backtest(
                market_data=market_data,
                target_variable=target_variable,
                feature_columns=feature_columns
            )
        except Exception as e:
            self.logger.error(f"❌ Backtesting failed: {e}")
            return BacktestingResult(
                success=False,
                execution_time=0.0,
                start_date=datetime.now(),
                end_date=datetime.now(),
                total_periods=0,
                error_message=str(e)
            )
    
    def _run_walk_forward_analysis(self, 
                                 market_data: pd.DataFrame,
                                 target_variable: str,
                                 feature_columns: Optional[List[str]]) -> WalkForwardResult:
        """Run walk-forward analysis."""
        try:
            return self.walk_forward_analyzer.run_walk_forward_analysis(
                market_data=market_data,
                target_variable=target_variable,
                feature_columns=feature_columns
            )
        except Exception as e:
            self.logger.error(f"❌ Walk-forward analysis failed: {e}")
            return WalkForwardResult(
                success=False,
                execution_time=0.0,
                total_folds=0,
                successful_folds=0,
                error_message=str(e)
            )
    
    def _run_performance_attribution(self, 
                                   backtesting_result: Optional[BacktestingResult],
                                   walk_forward_result: Optional[WalkForwardResult],
                                   market_data: pd.DataFrame) -> AttributionResult:
        """Run performance attribution analysis."""
        try:
            # Prepare performance data
            performance_history = []
            regime_history = []
            model_history = []
            
            # Extract data from backtesting result
            if backtesting_result and backtesting_result.success:
                performance_history = backtesting_result.performance_history if hasattr(backtesting_result, 'performance_history') else []
                regime_history = backtesting_result.regime_history if hasattr(backtesting_result, 'regime_history') else []
                model_history = backtesting_result.model_selection_history if hasattr(backtesting_result, 'model_selection_history') else []
            
            # Register performance data
            self.performance_attributor.register_performance_data(
                performance_history=performance_history,
                regime_history=regime_history,
                model_history=model_history,
                market_data=market_data
            )
            
            # Run attribution analysis
            return self.performance_attributor.run_attribution_analysis()
            
        except Exception as e:
            self.logger.error(f"❌ Performance attribution failed: {e}")
            return AttributionResult(
                success=False,
                execution_time=0.0,
                attribution_method=self.config.attribution_config.attribution_method,
                error_message=str(e)
            )
    
    def _run_scenario_testing(self, 
                            market_data: pd.DataFrame,
                            target_variable: str,
                            feature_columns: Optional[List[str]]) -> ScenarioResult:
        """Run scenario testing."""
        try:
            return self.scenario_tester.run_scenario_tests(
                market_data=market_data,
                target_variable=target_variable,
                feature_columns=feature_columns
            )
        except Exception as e:
            self.logger.error(f"❌ Scenario testing failed: {e}")
            return ScenarioResult(
                success=False,
                execution_time=0.0,
                total_scenarios=0,
                successful_scenarios=0,
                error_message=str(e)
            )
    
    def _analyze_validation_results(self, 
                                  backtesting_result: Optional[BacktestingResult],
                                  walk_forward_result: Optional[WalkForwardResult],
                                  attribution_result: Optional[AttributionResult],
                                  scenario_result: Optional[ScenarioResult]) -> Dict[str, Any]:
        """Analyze validation results."""
        try:
            analysis_results = {
                'overall_performance': {},
                'risk_metrics': {},
                'stability_metrics': {},
                'performance_insights': [],
                'risk_insights': [],
                'recommendations': []
            }
            
            # Analyze backtesting results
            if backtesting_result and backtesting_result.success:
                analysis_results['overall_performance'].update({
                    'total_return': backtesting_result.total_return,
                    'sharpe_ratio': backtesting_result.sharpe_ratio,
                    'max_drawdown': backtesting_result.max_drawdown,
                    'win_rate': backtesting_result.win_rate
                })
                
                analysis_results['risk_metrics'].update({
                    'var_95': backtesting_result.var_95,
                    'cvar_95': backtesting_result.cvar_95,
                    'volatility': backtesting_result.volatility
                })
            
            # Analyze walk-forward results
            if walk_forward_result and walk_forward_result.success:
                analysis_results['stability_metrics'].update({
                    'success_rate': walk_forward_result.successful_folds / walk_forward_result.total_folds,
                    'performance_consistency': self._calculate_performance_consistency(walk_forward_result)
                })
            
            # Analyze attribution results
            if attribution_result and attribution_result.success:
                analysis_results['performance_insights'].extend(
                    self._generate_attribution_insights(attribution_result)
                )
            
            # Analyze scenario results
            if scenario_result and scenario_result.success:
                analysis_results['risk_insights'].extend(
                    self._generate_scenario_insights(scenario_result)
                )
            
            # Generate recommendations
            analysis_results['recommendations'] = self._generate_recommendations(
                analysis_results['overall_performance'],
                analysis_results['risk_metrics'],
                analysis_results['stability_metrics']
            )
            
            return analysis_results
            
        except Exception as e:
            self.logger.error(f"❌ Results analysis failed: {e}")
            return {
                'overall_performance': {},
                'risk_metrics': {},
                'stability_metrics': {},
                'performance_insights': [],
                'risk_insights': [],
                'recommendations': []
            }
    
    def _calculate_performance_consistency(self, walk_forward_result: WalkForwardResult) -> float:
        """Calculate performance consistency from walk-forward results."""
        try:
            if not walk_forward_result.fold_performance:
                return 0.0
            
            # Calculate consistency based on performance variance
            f1_scores = [fold['performance_metrics']['f1_score'] for fold in walk_forward_result.fold_performance if fold['success']]
            
            if not f1_scores:
                return 0.0
            
            mean_f1 = np.mean(f1_scores)
            std_f1 = np.std(f1_scores)
            
            # Consistency = 1 - (std / mean)
            consistency = 1.0 - (std_f1 / (mean_f1 + 1e-8))
            return max(0.0, min(1.0, consistency))
            
        except Exception as e:
            self.logger.warning(f"Performance consistency calculation failed: {e}")
            return 0.0
    
    def _generate_attribution_insights(self, attribution_result: AttributionResult) -> List[str]:
        """Generate insights from attribution results."""
        insights = []
        
        try:
            # Regime attribution insights
            if attribution_result.regime_attribution:
                best_regime = max(attribution_result.regime_attribution.keys(), 
                                key=lambda x: attribution_result.regime_attribution[x].get('return_contribution', 0))
                insights.append(f"Best performing regime: {best_regime}")
            
            # Model attribution insights
            if attribution_result.model_attribution:
                best_model = max(attribution_result.model_attribution.keys(),
                               key=lambda x: attribution_result.model_attribution[x].get('return_contribution', 0))
                insights.append(f"Best performing model: {best_model}")
            
            # Factor attribution insights
            if attribution_result.factor_attribution:
                best_factor = max(attribution_result.factor_attribution.keys(),
                                key=lambda x: attribution_result.factor_attribution[x].get('return_contribution', 0))
                insights.append(f"Most influential factor: {best_factor}")
            
        except Exception as e:
            self.logger.warning(f"Attribution insights generation failed: {e}")
        
        return insights
    
    def _generate_scenario_insights(self, scenario_result: ScenarioResult) -> List[str]:
        """Generate insights from scenario results."""
        insights = []
        
        try:
            # Risk scenario insights
            if hasattr(scenario_result, 'risk_scenarios'):
                worst_scenario = min(scenario_result.risk_scenarios.keys(),
                                   key=lambda x: scenario_result.risk_scenarios[x].get('performance', 0))
                insights.append(f"Worst case scenario: {worst_scenario}")
            
            # Stress test insights
            if hasattr(scenario_result, 'stress_tests'):
                failed_stress_tests = [test for test, result in scenario_result.stress_tests.items() if not result.get('passed', False)]
                if failed_stress_tests:
                    insights.append(f"Failed stress tests: {', '.join(failed_stress_tests)}")
            
        except Exception as e:
            self.logger.warning(f"Scenario insights generation failed: {e}")
        
        return insights
    
    def _generate_recommendations(self, 
                                overall_performance: Dict[str, float],
                                risk_metrics: Dict[str, float],
                                stability_metrics: Dict[str, float]) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        try:
            # Performance recommendations
            if overall_performance.get('sharpe_ratio', 0) < 1.0:
                recommendations.append("Consider improving risk-adjusted returns through better model selection")
            
            if overall_performance.get('max_drawdown', 0) > 0.15:
                recommendations.append("Implement stricter risk management to reduce maximum drawdown")
            
            # Risk recommendations
            if risk_metrics.get('var_95', 0) < -0.05:
                recommendations.append("High Value at Risk detected - consider reducing position sizes")
            
            if risk_metrics.get('volatility', 0) > 0.2:
                recommendations.append("High volatility detected - consider diversification strategies")
            
            # Stability recommendations
            if stability_metrics.get('success_rate', 0) < 0.7:
                recommendations.append("Low success rate in walk-forward analysis - consider model improvements")
            
            if stability_metrics.get('performance_consistency', 0) < 0.5:
                recommendations.append("Low performance consistency - consider ensemble methods")
            
        except Exception as e:
            self.logger.warning(f"Recommendations generation failed: {e}")
        
        return recommendations
    
    def _get_configuration_summary(self) -> Dict[str, Any]:
        """Get configuration summary."""
        return {
            'mode': self.config.mode.value,
            'enable_backtesting': self.config.enable_backtesting,
            'enable_walk_forward': self.config.enable_walk_forward,
            'enable_attribution': self.config.enable_attribution,
            'enable_scenario_testing': self.config.enable_scenario_testing,
            'performance_threshold': self.config.performance_threshold,
            'risk_threshold': self.config.risk_threshold,
            'enable_risk_adjustment': self.config.enable_risk_adjustment
        }
    
    def _save_validation_results(self, result: ValidationResult):
        """Save validation results."""
        try:
            results_path = Path(self.config.results_path)
            results_path.mkdir(parents=True, exist_ok=True)
            
            # Save result summary
            result_summary = {
                'success': result.success,
                'execution_time': result.execution_time,
                'mode': result.mode.value,
                'overall_performance': result.overall_performance,
                'risk_metrics': result.risk_metrics,
                'stability_metrics': result.stability_metrics,
                'performance_insights': result.performance_insights,
                'risk_insights': result.risk_insights,
                'recommendations': result.recommendations,
                'start_time': result.start_time.isoformat(),
                'end_time': result.end_time.isoformat() if result.end_time else None,
                'configuration': result.configuration
            }
            
            with open(results_path / "validation_summary.json", 'w') as f:
                json.dump(result_summary, f, indent=2)
            
            # Save detailed results
            with open(results_path / "validation_result.pkl", 'wb') as f:
                pickle.dump(result, f)
            
            self.logger.info(f"💾 Validation results saved to {results_path}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save validation results: {e}")
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of validation results."""
        return {
            'components_initialized': {
                'backtesting': self.backtesting_engine is not None,
                'walk_forward': self.walk_forward_analyzer is not None,
                'attribution': self.performance_attributor is not None,
                'scenario': self.scenario_tester is not None
            },
            'validation_results': len(self.validation_results),
            'performance_history': len(self.performance_history),
            'risk_history': len(self.risk_history),
            'configuration': self._get_configuration_summary()
        }