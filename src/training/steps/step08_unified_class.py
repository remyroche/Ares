from ..standardized_parquet_handler import standardized_parquet_handler
"""
Unified Step08 Class Implementation - Part 2
"""

from src.utils.math_validation import (
    safe_divide, safe_log, safe_sqrt, safe_kelly_calculation,
    validate_positive, validate_range, MathValidationError
)
from src.utils.lookahead_bias_detector import (
import datetime
import logging
import numpy as np
import os
import pandas as pd
import pathlib as Path
import time
import typing

    get_global_detector, validate_no_future_data, LookaheadBiasError
)

class UnifiedStep08:
    """
    Unified Step08: Advanced Feature Selection with Regime Data Splitting and Financial Risk Assessment
    
    This class consolidates all Step08 functionality into a single, comprehensive module:
    - Regime data splitting with HMM composite clusters
    - Advanced feature selection with bias prevention
    - Financial metrics calculation (returns, volatility, Sharpe ratio, VaR)
    - Regime balance handling for imbalanced distributions
    - Comprehensive risk assessment with explicit risk metrics
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize unified Step08 with comprehensive configuration."""
        self.config = config
        self.logger = system_logger.getChild('UnifiedStep08')
        
        # Initialize components
        self._initialize_optimizations()
        self._initialize_configuration()
        self._initialize_metrics()
        
        self.logger.info('🚀 Unified Step08 initialized successfully')

    def _initialize_optimizations(self) -> None:
        """Initialize enhanced optimization components."""
        self.logger.info("🔧 Initializing enhanced optimization components...")
        
        # Initialize M1 optimizations if available
        if ENHANCED_OPTIMIZATIONS_AVAILABLE:
            try:
                self.m1_gpu_manager = get_m1_gpu_manager()
                self.m1_memory_optimizer = get_m1_memory_optimizer()
                self.m1_cpu_optimizer = get_m1_cpu_optimizer()
                self.pipeline_executor = OptimizedPipelineExecutor(max_concurrent_stages=6)
                self.matrix_operations = EnhancedMatrixOperations(
                    enable_gpu_acceleration=True,
                    enable_memory_optimization=True
                )
                self.optimization_selector = IntelligentOptimizationSelector()
                self.data_manager = OptimizedDataManager(
                    base_path=Path("data_cache"),
                    enable_compression=True,
                    enable_caching=True
                )
                self.error_handler = ErrorHandler(enable_recovery=True)
                self.logger.info("✅ Enhanced optimizations initialized")
            except Exception as e:
                self.logger.warning(f"⚠️ Enhanced optimizations failed: {e}")
                self._initialize_fallback_optimizations()
        else:
            self._initialize_fallback_optimizations()

    def _initialize_fallback_optimizations(self) -> None:
        """Initialize fallback optimization components."""
        self.m1_gpu_manager = None
        self.m1_memory_optimizer = None
        self.m1_cpu_optimizer = None
        self.pipeline_executor = None
        self.matrix_operations = None
        self.optimization_selector = None
        self.data_manager = None
        self.error_handler = None
        self.logger.info("✅ Fallback optimizations initialized")

    def _initialize_configuration(self) -> None:
        """Initialize configuration parameters."""
        self.step_config = self.config.get('step08_unified', {})
        
        # Feature selection parameters
        self.phase1_target_features = self.step_config.get('phase1_target_features', 150)
        self.phase2_targets = self.step_config.get('phase2_targets', [100, 80, 60])
        self.enable_mrmr = self.step_config.get('enable_mrmr', True)
        self.enable_rf_importance = self.step_config.get('enable_rf_importance', True)
        self.boruta_max_iter = self.step_config.get('boruta_max_iter', 100)
        self.boruta_alpha = self.step_config.get('boruta_alpha', 0.05)
        
        # Regime balance parameters
        self.min_regime_samples = self.step_config.get('min_regime_samples', 100)
        self.target_balance_ratio = self.step_config.get('target_balance_ratio', 0.8)
        self.enable_regime_rebalancing = self.step_config.get('enable_regime_rebalancing', True)
        self.rebalancing_method = self.step_config.get('rebalancing_method', 'oversample')
        
        # Financial metrics parameters
        self.risk_free_rate = self.step_config.get('risk_free_rate', 0.02)
        self.var_confidence_levels = self.step_config.get('var_confidence_levels', [0.95, 0.99])
        self.lookback_periods = self.step_config.get('lookback_periods', [30, 90, 252])
        
        # Risk assessment parameters
        self.model_risk_threshold = self.step_config.get('model_risk_threshold', 0.3)
        self.overfitting_threshold = self.step_config.get('overfitting_threshold', 0.1)
        self.feature_stability_threshold = self.step_config.get('feature_stability_threshold', 0.8)
        
        # Output directories
        self.output_dir = ensure_directory(self.step_config.get('output_dir', 'data/step08_unified'))
        self.reports_dir = ensure_directory(os.path.join(self.output_dir, 'reports'))
        self.artifacts_dir = ensure_directory(os.path.join(self.output_dir, 'artifacts'))
        self.metrics_dir = ensure_directory(os.path.join(self.output_dir, 'metrics'))

    def _initialize_metrics(self) -> None:
        """Initialize metrics tracking."""
        self.financial_metrics = FinancialMetrics()
        self.risk_metrics = RiskMetrics()
        self.regime_balance = RegimeBalanceMetrics()
        self.feature_validation = FeatureSelectionValidation()
        self.results = Step08Results()

    @with_tracing_span('step08_unified.execute', log_args=False)
    @handle_errors(exceptions=(Exception,), default_return={'success': False, 'error': 'Execution failed'}, context='step08_unified_execution')
    async def execute(self, training_input: Dict[str, Any] = None, pipeline_state: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute unified Step08 with comprehensive analysis."""
        try:
            start_time = datetime.now()
            self.logger.info('🚀 Starting Unified Step08 execution...')
            
            # Step 1: Load and validate data
            self.logger.info('📊 Step 1: Loading and validating data...')
            unified_data = await self._load_and_validate_data(training_input, pipeline_state)
            if unified_data is None:
                return {'success': False, 'error': 'Failed to load or validate data'}
            
            # Step 2: Regime balance analysis and handling
            self.logger.info('⚖️ Step 2: Analyzing and handling regime balance...')
            balanced_data = await self._handle_regime_balance(unified_data)
            
            # Step 3: Advanced feature selection with bias prevention
            self.logger.info('🔍 Step 3: Advanced feature selection with bias prevention...')
            selected_features = await self._advanced_feature_selection(balanced_data)
            
            # Step 4: Financial metrics calculation
            self.logger.info('💰 Step 4: Calculating financial metrics...')
            financial_metrics = await self._calculate_financial_metrics(balanced_data, selected_features)
            
            # Step 5: Risk assessment
            self.logger.info('⚠️ Step 5: Comprehensive risk assessment...')
            risk_metrics = await self._comprehensive_risk_assessment(balanced_data, selected_features, financial_metrics)
            
            # Step 6: Feature selection validation
            self.logger.info('✅ Step 6: Feature selection validation...')
            feature_validation = await self._validate_feature_selection(balanced_data, selected_features)
            
            # Step 7: Generate comprehensive results
            self.logger.info('📋 Step 7: Generating comprehensive results...')
            results = await self._generate_comprehensive_results(
                balanced_data, selected_features, financial_metrics, 
                risk_metrics, feature_validation, start_time
            )
            
            # Step 8: Save artifacts and reports
            self.logger.info('💾 Step 8: Saving artifacts and reports...')
            await self._save_artifacts_and_reports(results)
            
            self.logger.info('✅ Unified Step08 execution completed successfully')
            return {
                'success': True,
                'results': results,
                'execution_time': (datetime.now() - start_time).total_seconds()
            }
            
        except Exception as e:
            self.logger.exception(f'❌ Unified Step08 execution failed: {e}')
            return {'success': False, 'error': str(e)}

    async def _load_and_validate_data(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """Load and validate unified data with comprehensive checks."""
        try:
            # Load data using unified data loader
            if UNIFIED_DATA_LOADER_AVAILABLE:
                data_loader = UnifiedDataLoader(self.config)
                unified_data = await data_loader.load_unified_data(
                    symbol=self.config.get('symbol', 'ETHUSDT'),
                    exchange=self.config.get('exchange', 'BINANCE'),
                    timeframe=self.config.get('timeframe', '1m'),
                    data_dir=self.config.get('data_dir', 'data_cache')
                )
            else:
                # Fallback to pipeline state data
                if pipeline_state and 'dataframe' in pipeline_state:
                    unified_data = pipeline_state['dataframe']
                else:
                    self.logger.error('No data available and unified data loader not available')
                    return None
            
            if unified_data is None or len(unified_data) == 0:
                self.logger.error('Unified data is empty or None')
                return None
            
            # Validate required columns
            required_columns = ['timestamp', 'composite_cluster_id']
            missing_columns = [col for col in required_columns if col not in unified_data.columns]
            if missing_columns:
                self.logger.error(f'Missing required columns: {missing_columns}')
                return None
            
            # Validate regime data
            regime_data = unified_data['composite_cluster_id'].dropna()
            if regime_data.empty:
                self.logger.error('No valid regime data found')
                return None
            
            # Data quality validation
            unified_data = self._validate_and_fix_data_quality(unified_data)
            
            self.logger.info(f'✅ Loaded and validated data: {len(unified_data)} rows, {len(unified_data.columns)} columns')
            return unified_data
            
        except Exception as e:
            self.logger.error(f'Failed to load and validate data: {e}')
            return None

    def _validate_and_fix_data_quality(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate and fix data quality issues."""
        self.logger.info('🔍 Validating and fixing data quality...')
        
        # Remove duplicates
        if 'timestamp' in data.columns:
            duplicate_count = data['timestamp'].duplicated().sum()
            if duplicate_count > 0:
                self.logger.info(f'🗑️ Removing {duplicate_count} duplicate timestamps')
                data = data.drop_duplicates(subset=['timestamp'], keep='last')
        
        # Sort by timestamp
        if 'timestamp' in data.columns:
            if not data['timestamp'].is_monotonic_increasing:
                self.logger.info('📈 Sorting data by timestamp')
                data = data.sort_values('timestamp').reset_index(drop=True)
        
        # Set datetime index
        if 'timestamp' in data.columns and not isinstance(data.index, pd.DatetimeIndex):
            try:
                data['timestamp'] = pd.to_datetime(data['timestamp'])
                data = data.set_index('timestamp')
                self.logger.info('📅 Set datetime index')
            except Exception as e:
                self.logger.warning(f'⚠️ Could not set datetime index: {e}')
        
        # Handle missing values
        missing_before = data.isnull().sum().sum()
        if missing_before > 0:
            # Forward fill for regime data, drop for other columns
            if 'composite_cluster_id' in data.columns:
                data['composite_cluster_id'] = data['composite_cluster_id'].fillna(method='ffill')
            
            # Drop rows with missing values in critical columns
            critical_columns = ['open', 'high', 'low', 'close', 'volume']
            available_critical = [col for col in critical_columns if col in data.columns]
            if available_critical:
                data = data.dropna(subset=available_critical)
            
            missing_after = data.isnull().sum().sum()
            self.logger.info(f'🔧 Fixed missing values: {missing_before} → {missing_after}')
        
        return data

    async def _handle_regime_balance(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle regime balance for imbalanced distributions."""
        try:
            self.logger.info('⚖️ Analyzing regime balance...')
            
            # Calculate regime distribution
            regime_counts = data['composite_cluster_id'].value_counts().to_dict()
            total_samples = len(data)
            regime_percentages = {str(k): v/total_samples for k, v in regime_counts.items()}
            
            # Calculate balance score
            balance_score = self._calculate_balance_score(regime_percentages)
            
            # Determine imbalance severity
            imbalance_severity = self._assess_imbalance_severity(regime_percentages)
            
            # Update regime balance metrics
            self.regime_balance.regime_counts = {str(k): v for k, v in regime_counts.items()}
            self.regime_balance.regime_percentages = regime_percentages
            self.regime_balance.balance_score = balance_score
            self.regime_balance.imbalance_severity = imbalance_severity
            
            self.logger.info(f'📊 Regime balance analysis:')
            self.logger.info(f'   Balance score: {balance_score:.3f}')
            self.logger.info(f'   Imbalance severity: {imbalance_severity}')
            self.logger.info(f'   Regime distribution: {regime_percentages}')
            
            # Apply rebalancing if needed
            if self.enable_regime_rebalancing and imbalance_severity in ['moderate', 'severe']:
                self.logger.info('🔄 Applying regime rebalancing...')
                balanced_data = await self._apply_regime_rebalancing(data, regime_counts)
                self.regime_balance.rebalancing_applied = True
                self.regime_balance.rebalancing_method = self.rebalancing_method
                return balanced_data
            else:
                self.logger.info('✅ Regime balance is acceptable, no rebalancing needed')
                return data
                
        except Exception as e:
            self.logger.error(f'Failed to handle regime balance: {e}')
            return data

    def _calculate_balance_score(self, regime_percentages: Dict[str, float]) -> float:
        """Calculate regime balance score (0-1, higher is better)."""
        if not regime_percentages:
            return 0.0
        
        # Calculate Gini coefficient for balance assessment
        percentages = list(regime_percentages.values())
        n = len(percentages)
        if n <= 1:
            return 1.0
        
        # Sort percentages
        sorted_percentages = sorted(percentages)
        
        # Calculate Gini coefficient
        cumsum = np.cumsum(sorted_percentages)
        gini = (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n if cumsum[-1] > 0 else 0
        
        # Convert to balance score (1 - gini)
        balance_score = 1 - gini
        return max(0.0, min(1.0, balance_score))

    def _assess_imbalance_severity(self, regime_percentages: Dict[str, float]) -> str:
        """Assess the severity of regime imbalance."""
        if not regime_percentages:
            return 'none'
        
        percentages = list(regime_percentages.values())
        max_pct = max(percentages)
        min_pct = min(percentages)
        
        # Calculate imbalance ratio
        imbalance_ratio = max_pct / min_pct if min_pct > 0 else float('inf')
        
        if imbalance_ratio <= 2.0:
            return 'none'
        elif imbalance_ratio <= 5.0:
            return 'mild'
        elif imbalance_ratio <= 10.0:
            return 'moderate'
        else:
            return 'severe'

    async def _apply_regime_rebalancing(self, data: pd.DataFrame, regime_counts: Dict[int, int]) -> pd.DataFrame:
        """Apply regime rebalancing using specified method."""
        try:
            if self.rebalancing_method == 'oversample':
                return self._oversample_minority_regimes(data, regime_counts)
            elif self.rebalancing_method == 'undersample':
                return self._undersample_majority_regimes(data, regime_counts)
            elif self.rebalancing_method == 'smote':
                return await self._apply_smote_rebalancing(data, regime_counts)
            else:
                self.logger.warning(f'Unknown rebalancing method: {self.rebalancing_method}, using oversample')
                return self._oversample_minority_regimes(data, regime_counts)
                
        except Exception as e:
            self.logger.error(f'Failed to apply regime rebalancing: {e}')
            return data

    def _oversample_minority_regimes(self, data: pd.DataFrame, regime_counts: Dict[int, int]) -> pd.DataFrame:
        """Oversample minority regimes to balance the dataset."""
        # Find target sample size (median of regime counts)
        target_size = int(np.median(list(regime_counts.values())))
        
        balanced_data = []
        for regime_id, count in regime_counts.items():
            regime_data = data[data['composite_cluster_id'] == regime_id]
            
            if count < target_size:
                # Oversample minority regime
                n_samples = target_size - count
                oversampled = regime_data.sample(n=n_samples, replace=True, random_state=42)
                balanced_data.append(pd.concat([regime_data, oversampled]))
                self.logger.info(f'📈 Oversampled regime {regime_id}: {count} → {target_size}')
            else:
                balanced_data.append(regime_data)
        
        result = pd.concat(balanced_data, ignore_index=True)
        result = result.sort_values('timestamp' if 'timestamp' in result.columns else result.index.name or 'index')
        
        self.logger.info(f'✅ Regime rebalancing completed: {len(data)} → {len(result)} samples')
        return result

    def _undersample_majority_regimes(self, data: pd.DataFrame, regime_counts: Dict[int, int]) -> pd.DataFrame:
        """Undersample majority regimes to balance the dataset."""
        # Find target sample size (minimum regime count above threshold)
        min_count = min(regime_counts.values())
        target_size = max(min_count, self.min_regime_samples)
        
        balanced_data = []
        for regime_id, count in regime_counts.items():
            regime_data = data[data['composite_cluster_id'] == regime_id]
            
            if count > target_size:
                # Undersample majority regime
                undersampled = regime_data.sample(n=target_size, random_state=42)
                balanced_data.append(undersampled)
                self.logger.info(f'📉 Undersampled regime {regime_id}: {count} → {target_size}')
            else:
                balanced_data.append(regime_data)
        
        result = pd.concat(balanced_data, ignore_index=True)
        result = result.sort_values('timestamp' if 'timestamp' in result.columns else result.index.name or 'index')
        
        self.logger.info(f'✅ Regime rebalancing completed: {len(data)} → {len(result)} samples')
        return result

    async def _apply_smote_rebalancing(self, data: pd.DataFrame, regime_counts: Dict[int, int]) -> pd.DataFrame:
        """Apply SMOTE (Synthetic Minority Oversampling Technique) for regime rebalancing."""
        # This is a placeholder for SMOTE implementation
        # In practice, you would use imbalanced-learn library
        self.logger.warning('SMOTE rebalancing not implemented, using oversample instead')
        return self._oversample_minority_regimes(data, regime_counts)