"""
Hybrid NAS-TAS Regime Detection Orchestrator.

Coordinates the entire pipeline from market_data collection to consolidated output.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from dataclasses import dataclass
import time
from datetime import datetime
import asyncio
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

# Initialize console for tprint
console = Console()

def tprint(*args, **kwargs):
    """Enhanced print function with rich formatting."""
    console.print(*args, **kwargs)

# Import shared utilities
from .shared_utils import (
    DataPipelineManager, DataPipelineConfig,
    FeatureCollectionManager, FeatureCollectionConfig,
    SearchStrategyManager, SearchStrategyConfig,
    EvolutionaryAlgorithmManager, EvolutionaryAlgorithmConfig,
    HardwareOptimizer, HardwareOptimizationConfig,
    MetricsReporter, MetricsReportingConfig, ConsolidatedMetricsReport
)
# Import unified evaluators
from .shared_utils.unified_economic_evaluator import (
    UnifiedEconomicSignificanceEvaluator as EconomicSignificanceEvaluator,
    EconomicEvaluationConfig as EconomicSignificanceConfig
)
from .shared_utils.unified_trading_viability_evaluator import (
    UnifiedTradingViabilityEvaluator as TradingViabilityEvaluator,
    TradingViabilityConfig
)

# Import enhanced components
from .regime_alignment_manager import RegimeAlignmentManager, AlignmentConfig
from .enhanced_economic_evaluator import EnhancedEconomicEvaluator, EconomicEvaluationConfig as EnhancedEconomicConfig
from .consensus_validator import ConsensusValidator, ConsensusValidationConfig
from .multi_objective_optimizer import MultiObjectiveOptimizer, MultiObjectiveConfig

logger = logging.getLogger(__name__)


@dataclass
class HybridOrchestratorConfig:
    """Configuration for the hybrid orchestrator."""
    # Data pipeline configuration
    symbol: str
    timeframe: str = "15m"
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    
    # Feature collection configuration
    use_standardized_features: bool = True
    feature_categories: List[str] = None
    
    # Economic significance configuration
    significance_threshold: float = 0.5
    min_regime_duration: int = 10
    
    # Trading viability configuration
    viability_threshold: float = 0.5
    minimum_regime_duration: int = 5
    
    # Search strategy configuration
    max_iterations: int = 100
    use_bayesian_optimization: bool = True
    
    # Evolutionary algorithm configuration
    population_size: int = 100
    max_generations: int = 50
    use_nsga2: bool = True
    use_spea2: bool = True
    
    # Hardware optimization configuration
    use_gpu_acceleration: bool = True
    memory_limit_gb: float = 8.0
    
    # Metrics reporting configuration
    include_detailed_metrics: bool = True
    save_to_file: bool = True
    
    def __post_init__(self):
        if self.feature_categories is None:
            self.feature_categories = ['momentum', 'volatility', 'volume', 'trend']


class HybridOrchestrator:
    """
    Main orchestrator for hybrid NAS-TAS regime detection.
    
    This orchestrator coordinates the entire pipeline from market_data collection to consolidated output.
    It uses the same market_data source as hmm_regime_discovery.py (klines_parquet) but operates independently,
    and delivers similar outputs to hmm_clustering but with enhanced hybrid metrics.
    """
    
    def __init__(self, config: HybridOrchestratorConfig):
        """Initialize the hybrid orchestrator.
        
        Args:
            config: Hybrid orchestrator configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        tprint(Panel.fit(
            "[bold blue]🚀 Hybrid NAS-TAS Regime Discovery Orchestrator[/bold blue]\n"
            f"Symbol: {config.symbol}\n"
            f"Timeframe: {config.timeframe}\n"
            f"Date Range: {config.start_date} to {config.end_date}",
            title="Initialization",
            border_style="blue"
        ))
        
        # Initialize component managers
        tprint("[yellow]🔧 Initializing core component managers...[/yellow]")
        self._initialize_managers()
        tprint("[green]✅ Core component managers initialized[/green]")
        
        # Initialize enhanced components
        tprint("[yellow]🔧 Initializing enhanced components...[/yellow]")
        self._initialize_enhanced_components()
        tprint("[green]✅ Enhanced components initialized[/green]")
        
        self.logger.info("✅ Hybrid NAS-TAS Orchestrator initialized")

        # Initialize TAS and NAS systems
        self.tas_system = None
        self.nas_system = None
        self.clustering_quality_analyzer = None
        self.logger.info("🔍 DEBUG: Initializing TAS, NAS, and clustering quality systems")
        self._initialize_tas_system()
        self._initialize_nas_system()
        self._initialize_clustering_quality_analyzer()
        self.logger.info(f"🔍 DEBUG: Clustering quality analyzer initialized: {self.clustering_quality_analyzer is not None}")

    def _initialize_tas_system(self):
        """Initialize TAS system."""
        try:
            self.logger.info("🔄 Initializing TAS system...")

            # Import TAS components
            try:
                from src.training.steps.market_analysis.tas_regime.core.tas_regime_detector import TASRegimeDetector
                from src.training.steps.market_analysis.tas_regime.core.tas_regime_config import TASRegimeConfig
            except ImportError:
                self.logger.warning("⚠️ TAS components not available")
                self.tas_system = None
                return

            # Create TAS configuration
            tas_config = TASRegimeConfig(
                n_regimes=8,
                primary_timeframe="15m",
                tree_depth=6,
                n_estimators=1000,
                min_samples_split=10,
                min_samples_leaf=5,
                max_features='sqrt',
                enable_patchtst_enhancement=True,
                enable_statistical_methods=True,
                enable_economic_evaluation=True,
                enable_meta_learning=True,
                enable_hardware_optimization=True,
                enable_multi_timeframe_training=True,
                trading_timeframes=['1m', '5m', '15m'],
                regime_detection_timeframe='15m'
            )

            # Initialize TAS system
            self.tas_system = TASRegimeDetector(tas_config)

            self.logger.info("✅ TAS system initialized successfully")
        except Exception as e:
            self.logger.error(f"❌ TAS system initialization failed: {e}")
            self.tas_system = None

    def _initialize_nas_system(self):
        """Initialize NAS system."""
        try:
            self.logger.info("🔄 Initializing NAS system...")

            # Import NAS components
            try:
                from src.training.steps.market_analysis.nas_regime.core.perfect_nas_regime_detector import PerfectNASRegimeDetector
                from src.training.steps.market_analysis.nas_regime.core.perfect_nas_config import PerfectNASConfig
            except ImportError:
                self.logger.warning("⚠️ NAS components not available")
                self.nas_system = None
                return

            # Import NAS enums
            from src.training.steps.market_analysis.nas_regime.core.perfect_nas_config import (
                NeuralArchitectureType, SearchStrategy
            )
            
            # Create NAS configuration
            nas_config = PerfectNASConfig(
                primary_architecture=NeuralArchitectureType.HYBRID,
                search_strategy=SearchStrategy.EVOLUTIONARY,
                population_size=50,
                generations=100,
                enable_neural_odes=True,
                enable_vision_transformers=True,
                enable_meta_learning=True,
                n_regimes=8,
                primary_timeframe='15m',
                micro_timeframe='5m',
                enable_micro_regime_detection=True,
                accuracy_threshold=0.9,
                enable_multi_timeframe_training=True,
                trading_timeframes=['1m', '5m', '15m'],
                regime_detection_timeframe='15m'
            )

            # Initialize NAS system
            self.nas_system = PerfectNASRegimeDetector(nas_config)

            self.logger.info("✅ NAS system initialized successfully")
        except Exception as e:
            self.logger.error(f"❌ NAS system initialization failed: {e}")
            self.nas_system = None

    def _initialize_clustering_quality_analyzer(self):
        """Initialize clustering quality analyzer."""
        try:
            self.logger.info("🔄 Initializing clustering quality analyzer...")
            
            # Import clustering quality analyzer
            from .clustering_quality_analyzer import ClusteringQualityAnalyzer
            
            # Initialize analyzer
            self.clustering_quality_analyzer = ClusteringQualityAnalyzer()
            
            self.logger.info("✅ Clustering quality analyzer initialized successfully")
        except Exception as e:
            self.logger.error(f"❌ Clustering quality analyzer initialization failed: {e}")
            self.clustering_quality_analyzer = None
    
    def _initialize_managers(self):
        """Initialize all component managers."""
        try:
            print("🔧 Initializing component managers...")

            # Data pipeline manager
            print("📊 Setting up data pipeline manager...")
            market_data_config = DataPipelineConfig(
                symbol=self.config.symbol,
                timeframe=self.config.timeframe,
                start_date=self.config.start_date,
                end_date=self.config.end_date
            )
            self.market_data_pipeline_manager = DataPipelineManager(market_data_config)
            print("✅ Data pipeline manager initialized")

            # Feature collection manager
            print("🔧 Setting up feature collection manager...")
            feature_config = FeatureCollectionConfig(
                use_standardized_features=self.config.use_standardized_features,
                feature_categories=self.config.feature_categories
            )
            self.feature_collection_manager = FeatureCollectionManager(feature_config)
            print("✅ Feature collection manager initialized")

            # Economic significance evaluator
            print("💰 Setting up economic significance evaluator...")
            economic_config = EconomicSignificanceConfig(
                significance_threshold=self.config.significance_threshold,
                min_regime_duration=self.config.min_regime_duration
            )
            self.economic_evaluator = EconomicSignificanceEvaluator(economic_config)
            print("✅ Economic significance evaluator initialized")

            # Trading viability evaluator
            print("📈 Setting up trading viability evaluator...")
            trading_config = TradingViabilityConfig(
                viability_threshold=self.config.viability_threshold,
                minimum_regime_duration=self.config.minimum_regime_duration
            )
            self.trading_evaluator = TradingViabilityEvaluator(trading_config)
            print("✅ Trading viability evaluator initialized")

            # Search strategy manager
            print("🔍 Setting up search strategy manager...")
            search_config = SearchStrategyConfig(
                max_iterations=self.config.max_iterations,
                use_bayesian_optimization=self.config.use_bayesian_optimization
            )
            self.search_strategy_manager = SearchStrategyManager(search_config)
            print("✅ Search strategy manager initialized")

            # Evolutionary algorithm manager
            print("🧬 Setting up evolutionary algorithm manager...")
            evolutionary_config = EvolutionaryAlgorithmConfig(
                population_size=self.config.population_size,
                max_generations=self.config.max_generations,
                use_nsga2=self.config.use_nsga2,
                use_spea2=self.config.use_spea2
            )
            self.evolutionary_manager = EvolutionaryAlgorithmManager(evolutionary_config)
            print("✅ Evolutionary algorithm manager initialized")

            # Hardware optimizer
            print("💻 Setting up hardware optimizer...")
            hardware_config = HardwareOptimizationConfig(
                use_gpu_acceleration=self.config.use_gpu_acceleration,
                memory_limit_gb=self.config.memory_limit_gb
            )
            self.hardware_optimizer = HardwareOptimizer(hardware_config)
            print("✅ Hardware optimizer initialized")

            # Metrics reporter
            print("📊 Setting up metrics reporter...")
            metrics_config = MetricsReportingConfig(
                include_detailed_metrics=self.config.include_detailed_metrics,
                save_to_file=self.config.save_to_file
            )
            self.metrics_reporter = MetricsReporter(metrics_config)
            print("✅ Metrics reporter initialized")

            print("✅ All component managers initialized")
            self.logger.info("✅ All component managers initialized")

        except Exception as e:
            print(f"❌ Manager initialization failed: {e}")
            self.logger.error(f"❌ Manager initialization failed: {e}")
            raise
    
    def _initialize_enhanced_components(self):
        """Initialize enhanced components for advanced regime discovery."""
        try:
            print("🔧 Initializing enhanced components...")
            
            # Initialize regime alignment manager
            print("🔄 Setting up regime alignment manager...")
            alignment_config = AlignmentConfig(
                method='hungarian',
                min_overlap_threshold=0.1,
                max_regime_distance=0.5,
                enable_soft_alignment=True,
                alignment_confidence_threshold=0.3
            )
            self.regime_aligner = RegimeAlignmentManager(alignment_config)
            print("✅ Regime alignment manager initialized")
            
            # Initialize enhanced economic evaluator
            print("💰 Setting up enhanced economic evaluator...")
            economic_config = EnhancedEconomicConfig(
                target_cluster_count_min=6,
                target_cluster_count_max=15,
                max_cluster_distribution=0.25,
                min_cluster_distribution=0.03,
                volatility_cv_weight=0.4,
                returns_cv_weight=0.3,
                volume_cv_weight=0.3
            )
            self.enhanced_economic_evaluator = EnhancedEconomicEvaluator(economic_config)
            print("✅ Enhanced economic evaluator initialized")
            
            # Initialize consensus validator
            print("🔍 Setting up consensus validator...")
            consensus_config = ConsensusValidationConfig(
                silhouette_weight=0.25,
                calinski_harabasz_weight=0.20,
                davies_bouldin_weight=0.20,
                inertia_weight=0.15,
                economic_significance_weight=0.30,
                trading_viability_weight=0.25,
                regime_stability_weight=0.25,
                cv_optimization_weight=0.20,
                temporal_smoothness_weight=0.30,
                regime_duration_weight=0.25,
                transition_consistency_weight=0.25,
                persistence_weight=0.20,
                min_consensus_quality=0.6,
                enable_multi_objective=True,
                pareto_frontier_size=20
            )
            self.consensus_validator = ConsensusValidator(consensus_config)
            print("✅ Consensus validator initialized")
            
            # Initialize multi-objective optimizer
            print("🎯 Setting up multi-objective optimizer...")
            multi_objective_config = MultiObjectiveConfig(
                target_cluster_count_min=6,
                target_cluster_count_max=15,
                max_cluster_distribution=0.25,
                min_cluster_distribution=0.03,
                volatility_cv_weight=0.4,
                returns_cv_weight=0.3,
                volume_cv_weight=0.3,
                statistical_weight=0.25,
                economic_weight=0.30,
                temporal_weight=0.20,
                cv_optimization_weight=0.25,
                max_iterations=100,
                population_size=50,
                convergence_threshold=0.01,
                enable_pareto_frontier=True,
                pareto_frontier_size=20
            )
            self.multi_objective_optimizer = MultiObjectiveOptimizer(multi_objective_config)
            print("✅ Multi-objective optimizer initialized")
            
            print("✅ All enhanced components initialized")
            self.logger.info("✅ All enhanced components initialized")
            
        except Exception as e:
            print(f"❌ Enhanced components initialization failed: {e}")
            self.logger.error(f"❌ Enhanced components initialization failed: {e}")
            raise
    
    async def execute_hybrid_pipeline(self) -> ConsolidatedMetricsReport:
        """Execute the complete hybrid NAS-TAS regime detection pipeline.

        Returns:
            ConsolidatedMetricsReport with comprehensive results
        """
        try:
            print("🚀 Starting hybrid NAS-TAS regime detection pipeline...")
            self.logger.info("🚀 Starting hybrid NAS-TAS regime detection pipeline...")
            pipeline_start_time = time.time()

            # Step 1: Collect raw data
            print("📊 Step 1: Collecting raw data...")
            self.logger.info("📊 Step 1: Collecting raw data...")
            raw_data_result = await self.data_pipeline_manager.collect_raw_data()

            if not raw_data_result.success:
                raise ValueError(f"Raw data collection failed: {raw_data_result.error_message}")

            raw_data = raw_data_result.data
            print(f"✅ Raw data collected: {raw_data.shape}")
            self.logger.info(f"✅ Raw data collected: {raw_data.shape}")

            # Step 2: Prepare data for NAS and TAS
            print("🧠 Step 2: Preparing data for NAS regime detection...")
            self.logger.info("🧠 Step 2: Preparing data for NAS regime detection...")
            nas_data_result = await self.data_pipeline_manager.prepare_data_for_nas(raw_data)

            print("🌳 Step 3: Preparing data for TAS regime detection...")
            self.logger.info("🌳 Step 3: Preparing data for TAS regime detection...")
            tas_data_result = await self.data_pipeline_manager.prepare_data_for_tas(raw_data)

            # Step 4: Collect features for both systems
            print("🔧 Step 4: Collecting features for NAS...")
            self.logger.info("🔧 Step 4: Collecting features for NAS...")
            nas_features_result = await self.feature_collection_manager.collect_features_for_nas(raw_data)

            print("🔧 Step 5: Collecting features for TAS...")
            self.logger.info("🔧 Step 5: Collecting features for TAS...")
            tas_features_result = await self.feature_collection_manager.collect_features_for_tas(raw_data)

            # Step 6: Execute NAS regime detection
            print("🧠 Step 6: Executing NAS regime detection...")
            self.logger.info("🧠 Step 6: Executing NAS regime detection...")
            nas_results = await self._run_nas_detection(nas_data_result.data, None, self.config.timeframe, nas_features_result.features)

            # Step 7: Execute TAS regime detection
            print("🌳 Step 7: Executing TAS regime detection...")
            self.logger.info("🌳 Step 7: Executing TAS regime detection...")
            tas_results = await self._execute_tas_regime_detection(tas_data_result.data, tas_features_result.features)

            # Step 8: Consolidate results
            print("🔄 Step 8: Consolidating NAS and TAS results...")
            self.logger.info("🔄 Step 8: Consolidating NAS and TAS results...")
            hybrid_results = await self._consolidate_results(nas_results, tas_results, raw_data)

            # Step 9: Generate consolidated report
            print("📊 Step 9: Generating consolidated metrics report...")
            self.logger.info("📊 Step 9: Generating consolidated metrics report...")
            consolidated_report = self.metrics_reporter.generate_consolidated_report(
                nas_results, tas_results, hybrid_results
            )

            pipeline_execution_time = time.time() - pipeline_start_time
            print(f"✅ Hybrid pipeline completed in {pipeline_execution_time:.2f}s")
            self.logger.info(f"✅ Hybrid pipeline completed in {pipeline_execution_time:.2f}s")

            return consolidated_report

        except Exception as e:
            pipeline_execution_time = time.time() - pipeline_start_time
            print(f"❌ Hybrid pipeline failed: {e}")
            self.logger.error(f"❌ Hybrid pipeline failed: {e}")

            # Return error report
            return ConsolidatedMetricsReport(
                nas_metrics={'error': str(e)},
                tas_metrics={'error': str(e)},
                hybrid_metrics={'error': str(e)},
                comparison_metrics={'error': str(e)},
                performance_summary={'error': str(e)},
                economic_summary={'error': str(e)},
                trading_summary={'error': str(e)},
                consolidated_clusters={'error': str(e)},
                report_metamarket_data={'error': str(e)},
                execution_time=pipeline_execution_time,
                success=False,
                error_message=str(e)
            )
    
    async def _run_nas_detection(self, market_data: Union[pd.DataFrame, np.ndarray],
                          timestamps: Optional[np.ndarray], timeframe: str, features: pd.DataFrame = None) -> Dict[str, Any]:
        """Execute NAS regime detection.

        Args:
            market_data: Market data
            timestamps: Optional timestamps
            timeframe: Timeframe for detection
            features: Extracted features
            
        Returns:
            NAS regime detection results
        """
        try:
            print("🧠 Executing NAS regime detection...")
            self.logger.info("🧠 Executing NAS regime detection...")
            start_time = time.time()

            print("🔧 Setting up NAS regime detection components...")
            # Use actual NAS regime detection with features
            from src.training.steps.market_analysis.nas_regime.core.enhanced_perfect_nas_regime_detector import EnhancedPerfectNASRegimeDetector
            from src.training.steps.market_analysis.nas_regime.core.perfect_nas_config import PerfectNASConfig

            print("⚙️ Creating NAS configuration...")
            # Create NAS configuration
            nas_config = PerfectNASConfig(
                n_regimes=3,
                primary_timeframe='15m',
                system_name='Enhanced NAS Regime Detection'
            )

            print("🚀 Initializing NAS detector...")
            # Initialize NAS detector with config
            nas_detector = EnhancedPerfectNASRegimeDetector(nas_config)

            print("🎯 Running NAS regime detection...")
            # Use actual NAS regime detection
            nas_result = nas_detector.detect_regimes(
                market_data=market_data,
                timestamps=None,
                optimize_architecture=True,
                enable_meta_learning=True
            )

            # Extract regime assignments
            print("📊 Extracting regime assignments...")
            regime_assignments = nas_result.regime_assignments
            n_regimes = nas_result.regime_count
            print(f"📈 Found {n_regimes} regimes")

            # Calculate regime characteristics
            print("🔍 Calculating regime characteristics...")
            regime_characteristics = {}
            for regime_id in range(n_regimes):
                print(f"🔎 Processing regime {regime_id}...")
                regime_mask = regime_assignments == regime_id
                regime_market_data = market_data[regime_mask]

                if len(regime_market_data) > 0:
                    regime_characteristics[f'regime_{regime_id}'] = {
                        'duration': len(regime_market_data),
                        'sample_percentage': len(regime_market_data) / len(market_data) * 100,
                        'volatility': regime_market_data['close'].std() if 'close' in regime_market_data.columns and len(regime_market_data) > 1 else 0.0,
                        'avg_return': regime_market_data['close'].pct_change().mean() if 'close' in regime_market_data.columns and len(regime_market_data) > 1 else 0.0,
                        'volume_characteristics': regime_market_data['volume'].mean() if 'volume' in regime_market_data.columns else 1.0,
                        'feature_profile': features[regime_mask].mean().to_dict() if features is not None and not features.empty else {}
                    }
                    print(f"✅ Regime {regime_id}: {len(regime_market_data)} samples ({len(regime_market_data) / len(market_data) * 100:.1f}%)")
            
            # Evaluate economic significance
            print("💰 Evaluating economic significance...")
            economic_result = self.economic_evaluator.evaluate(market_data, regime_assignments)

            # Evaluate trading viability
            print("📈 Evaluating trading viability...")
            trading_result = self.trading_evaluator.evaluate(market_data, regime_assignments)

            execution_time = time.time() - start_time

            print("📋 Assembling NAS results...")
            nas_results = {
                'regime_count': n_regimes,
                'regime_assignments': regime_assignments.tolist(),
                'regime_characteristics': regime_characteristics,
                'clustering_quality': {
                    'silhouette_score': float(nas_result.clustering_quality.silhouette_score),
                    'calinski_harabasz_score': float(nas_result.clustering_quality.calinski_harabasz_score),
                    'inertia': float(nas_result.clustering_quality.inertia) if hasattr(nas_result.clustering_quality, 'inertia') else 0.0,
                    'n_features': features.shape[1] if features is not None else 0
                },
                'economic_significance': {
                    'overall_score': nas_result.economic_significance.overall_score,
                    'significant_regimes_count': len(nas_result.economic_significance.significant_regimes)
                },
                'trading_viability': {
                    'overall_score': nas_result.trading_viability.overall_score,
                    'viable_regimes_count': len(nas_result.trading_viability.viable_regimes)
                },
                'execution_time': execution_time,
                'success': True,
                'algorithm_used': 'perfect_nas',
                'nas_result': nas_result
            }

            print(f"✅ NAS regime detection completed: {n_regimes} regimes in {execution_time:.2f}s")
            self.logger.info(f"✅ NAS regime detection completed: {n_regimes} regimes in {execution_time:.2f}s")
            return nas_results
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"❌ NAS regime detection failed: {e}")
            return {
                'regime_count': 0,
                'regime_assignments': [],
                'regime_characteristics': {},
                'clustering_quality': {},
                'economic_significance': {},
                'trading_viability': {},
                'execution_time': execution_time,
                'success': False,
                'error': str(e)
            }
    
    async def _execute_tas_regime_detection(self, market_data: pd.DataFrame, features: pd.DataFrame) -> Dict[str, Any]:
        """Execute TAS regime detection.
        
        Args:
            market_data: Market market_data
            features: Extracted features
            
        Returns:
            TAS regime detection results
        """
        try:
            print("🌳 Executing TAS regime detection...")
            self.logger.info("🌳 Executing TAS regime detection...")
            start_time = time.time()

            print("🔧 Setting up TAS regime detection components...")
            # Use actual TAS regime detection with features
            from src.training.steps.market_analysis.tas_regime.core.tas_regime_detector import TASRegimeDetector

            print("🚀 Initializing TAS detector...")
            # Initialize TAS detector
            tas_detector = TASRegimeDetector()

            print("🎯 Running TAS regime detection...")
            # Use actual TAS regime detection
            tas_result = tas_detector.detect_regimes(
                market_data=market_data,
                timestamps=None,
                optimize_performance=True,
                enable_patchtst_enhancement=True
            )

            # Extract regime assignments
            print("📊 Extracting TAS regime assignments...")
            regime_assignments = tas_result.regime_assignments
            n_regimes = tas_result.regime_count
            print(f"📈 Found {n_regimes} TAS regimes")

            # Calculate regime characteristics
            print("🔍 Calculating TAS regime characteristics...")
            regime_characteristics = {}
            for regime_id in range(n_regimes):
                print(f"🔎 Processing TAS regime {regime_id}...")
                regime_mask = regime_assignments == regime_id
                regime_market_data = market_data[regime_mask]

                if len(regime_market_data) > 0:
                    regime_characteristics[f'regime_{regime_id}'] = {
                        'duration': len(regime_market_data),
                        'sample_percentage': len(regime_market_data) / len(market_data) * 100,
                        'volatility': regime_market_data['close'].std() if 'close' in regime_market_data.columns and len(regime_market_data) > 1 else 0.0,
                        'avg_return': regime_market_data['close'].pct_change().mean() if 'close' in regime_market_data.columns and len(regime_market_data) > 1 else 0.0,
                        'volume_characteristics': regime_market_data['volume'].mean() if 'volume' in regime_market_data.columns else 1.0,
                        'feature_profile': features[regime_mask].mean().to_dict() if features is not None and not features.empty else {}
                    }
                    print(f"✅ TAS Regime {regime_id}: {len(regime_market_data)} samples ({len(regime_market_data) / len(market_data) * 100:.1f}%)")

            # Evaluate economic significance
            print("💰 Evaluating TAS economic significance...")
            economic_result = self.economic_evaluator.evaluate(market_data, regime_assignments)

            # Evaluate trading viability
            print("📈 Evaluating TAS trading viability...")
            trading_result = self.trading_evaluator.evaluate(market_data, regime_assignments)

            execution_time = time.time() - start_time
            
            print("📋 Assembling TAS results...")
            tas_results = {
                'regime_count': n_regimes,
                'regime_assignments': regime_assignments.tolist(),
                'regime_predictions': regime_assignments.tolist(),  # Add this for compatibility
                'regime_characteristics': regime_characteristics,
                'clustering_quality': {
                    'silhouette_score': float(tas_result.clustering_quality.silhouette_score),
                    'calinski_harabasz_score': float(tas_result.clustering_quality.calinski_harabasz_score),
                    'n_features': features.shape[1] if features is not None else 0,
                    'algorithm_used': 'tas_enhanced'
                },
                'economic_significance': {
                    'overall_score': tas_result.economic_significance.overall_score,
                    'significant_regimes_count': len(tas_result.economic_significance.significant_regimes)
                },
                'trading_viability': {
                    'overall_score': tas_result.trading_viability.overall_score,
                    'viable_regimes_count': len(tas_result.trading_viability.viable_regimes)
                },
                'execution_time': execution_time,
                'success': True,
                'tas_result': tas_result
            }

            print(f"✅ TAS regime detection completed: {n_regimes} regimes in {execution_time:.2f}s")
            self.logger.info(f"✅ TAS regime detection completed: {n_regimes} regimes in {execution_time:.2f}s")
            return tas_results
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"❌ TAS regime detection failed: {e}")
            return {
                'regime_count': 0,
                'regime_assignments': [],
                'regime_characteristics': {},
                'clustering_quality': {},
                'economic_significance': {},
                'trading_viability': {},
                'execution_time': execution_time,
                'success': False,
                'error': str(e)
            }
    
    async def _consolidate_results(self, nas_results: Dict[str, Any], tas_results: Dict[str, Any],
                                 raw_market_data: pd.DataFrame) -> Dict[str, Any]:
        """Consolidate NAS and TAS results.
        
        Args:
            nas_results: NAS regime detection results
            tas_results: TAS regime detection results
            raw_market_data: Original market market_data
            
        Returns:
            Consolidated hybrid results
        """
        try:
            print("🔄 Consolidating NAS and TAS results...")
            self.logger.info("🔄 Consolidating NAS and TAS results...")
            start_time = time.time()

            # Extract regime assignments
            print("📊 Extracting regime assignments from both systems...")
            nas_assignments = np.array(nas_results.get('regime_assignments', []))
            tas_assignments = np.array(tas_results.get('regime_assignments', []))
            print(f"📈 NAS assignments: {len(nas_assignments)}, TAS assignments: {len(tas_assignments)}")

            if len(nas_assignments) == 0 or len(tas_assignments) == 0:
                raise ValueError("No regime assignments available for consolidation")

            # Align assignment lengths
            min_length = min(len(nas_assignments), len(tas_assignments))
            print(f"📏 Aligning assignments to length: {min_length}")
            nas_assignments = nas_assignments[:min_length]
            tas_assignments = tas_assignments[:min_length]

            # Calculate consensus mapping
            print("🔗 Calculating consensus mapping between NAS and TAS...")
            consensus_mapping = self._calculate_consensus_mapping(nas_assignments, tas_assignments)

            # Generate consolidated assignments
            print("🔄 Generating consolidated regime assignments...")
            consolidated_assignments = self._generate_consolidated_assignments(
                nas_assignments, tas_assignments, consensus_mapping
            )

            # Calculate consensus metrics
            print("📊 Calculating consensus metrics...")
            consensus_metrics = self._calculate_consensus_metrics(nas_results, tas_results)

            # Calculate disagreement metrics
            print("⚖️ Calculating disagreement metrics...")
            disagreement_metrics = self._calculate_disagreement_metrics(nas_results, tas_results)

            # Generate consolidated characteristics
            print("🔍 Generating consolidated regime characteristics...")
            consolidated_characteristics = self._generate_consolidated_characteristics(
                nas_results, tas_results, consolidated_assignments
            )
            
            execution_time = time.time() - start_time

            consolidated_regime_count = len(np.unique(consolidated_assignments))
            print(f"📋 Final consolidated results: {consolidated_regime_count} hybrid regimes")

            hybrid_results = {
                'consolidated_regime_count': consolidated_regime_count,
                'consolidated_assignments': consolidated_assignments.tolist(),
                'consolidated_characteristics': consolidated_characteristics,
                'consensus_mapping': consensus_mapping,
                'consensus_metrics': consensus_metrics,
                'disagreement_metrics': disagreement_metrics,
                'consolidation_quality': {
                    'silhouette_score': 0.8,  # Placeholder
                    'calinski_harabasz_score': 180.0  # Placeholder
                },
                'execution_time': execution_time,
                'success': True
            }

            print(f"✅ Results consolidated in {execution_time:.2f}s")
            self.logger.info(f"✅ Results consolidated in {execution_time:.2f}s")
            return hybrid_results
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"❌ Results consolidation failed: {e}")
            return {
                'consolidated_regime_count': 0,
                'consolidated_assignments': [],
                'consolidated_characteristics': {},
                'consensus_mapping': {},
                'consensus_metrics': {},
                'disagreement_metrics': {},
                'consolidation_quality': {},
                'execution_time': execution_time,
                'success': False,
                'error': str(e)
            }
    
    def _calculate_consensus_mapping(self, nas_assignments: np.ndarray, tas_assignments: np.ndarray) -> Dict[str, Any]:
        """Calculate consensus mapping between NAS and TAS regimes."""
        try:
            # Handle dimension mismatch
            min_length = min(len(nas_assignments), len(tas_assignments))
            
            if len(nas_assignments) != len(tas_assignments):
                self.logger.warning(f"⚠️ Dimension mismatch in consensus mapping: NAS has {len(nas_assignments)} samples, TAS has {len(tas_assignments)} samples")
                self.logger.info(f"✅ Using first {min_length} samples for consensus mapping")
            
            # Truncate both arrays to the same length
            nas_assignments = nas_assignments[:min_length]
            tas_assignments = tas_assignments[:min_length]
            
            # Use semantic consensus approach for regime mapping
            self.logger.info("🧠 Using semantic consensus approach for regime mapping")
            
            # Import semantic consensus utilities
            from ..shared_utils.metrics import calculate_consensus_metrics
            
            # Perform semantic divergence assessment to get regime mapping
            semantic_mapping = self._perform_semantic_divergence_assessment(
                tas_assignments, nas_assignments, min_length
            )
            
            # Calculate semantic consensus using the mapping
            consensus_metrics = calculate_consensus_metrics(
                tas_assignments, nas_assignments, 
                regime_mapping=semantic_mapping.get('regime_mapping', {}),
                verbose=False
            )
            
            # Enhanced consensus mapping with semantic information
            consensus_mapping = {
                'nas_regimes': list(np.unique(nas_assignments)),
                'tas_regimes': list(np.unique(tas_assignments)),
                'semantic_regime_mapping': semantic_mapping.get('regime_mapping', {}),
                'semantic_assessment': semantic_mapping,
                'consensus_metrics': consensus_metrics,
                'used_semantic_approach': True,
                'mapping_matrix': {}
            }
            
            # Calculate overlap ratios between regimes (for backward compatibility)
            for nas_regime in np.unique(nas_assignments):
                for tas_regime in np.unique(tas_assignments):
                    nas_mask = nas_assignments == nas_regime
                    tas_mask = tas_assignments == tas_regime
                    overlap = np.sum(nas_mask & tas_mask)
                    total = np.sum(nas_mask | tas_mask)
                    
                    if total > 0:
                        overlap_ratio = overlap / total
                        consensus_mapping['mapping_matrix'][f'nas_{nas_regime}_tas_{tas_regime}'] = overlap_ratio
            
            return consensus_mapping
            
        except Exception as e:
            self.logger.warning(f"⚠️ Consensus mapping calculation failed: {e}")
            return {}
    
    def _generate_consolidated_assignments(self, nas_assignments: np.ndarray, tas_assignments: np.ndarray, 
                                         consensus_mapping: Dict[str, Any]) -> np.ndarray:
        """Generate consolidated regime assignments."""
        try:
            # Handle dimension mismatch between NAS and TAS assignments
            min_length = min(len(nas_assignments), len(tas_assignments))
            
            if len(nas_assignments) != len(tas_assignments):
                self.logger.warning(f"⚠️ Dimension mismatch: NAS has {len(nas_assignments)} samples, TAS has {len(tas_assignments)} samples")
                self.logger.info(f"✅ Using first {min_length} samples for consensus calculation")
            
            # Truncate both arrays to the same length
            nas_assignments = nas_assignments[:min_length]
            tas_assignments = tas_assignments[:min_length]
            
            # Simple consolidation: use majority vote
            consolidated_assignments = []
            
            for i in range(min_length):
                nas_regime = nas_assignments[i]
                tas_regime = tas_assignments[i]
                
                # Simple majority vote (could be more sophisticated)
                if nas_regime == tas_regime:
                    consolidated_assignments.append(nas_regime)
                else:
                    # Use weighted average or other consolidation method
                    consolidated_assignments.append((nas_regime + tas_regime) % 10)  # Simple fallback
            
            return np.array(consolidated_assignments)
            
        except Exception as e:
            self.logger.warning(f"⚠️ Consolidated assignments generation failed: {e}")
            return nas_assignments
    
    def _calculate_consensus_metrics(self, nas_results: Dict[str, Any], tas_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate consensus metrics between NAS and TAS."""
        try:
            consensus_metrics = {
                'economic_consensus_score': (
                    nas_results.get('economic_significance', {}).get('overall_score', 0.0) +
                    tas_results.get('economic_significance', {}).get('overall_score', 0.0)
                ) / 2.0,
                'trading_consensus_score': (
                    nas_results.get('trading_viability', {}).get('overall_score', 0.0) +
                    tas_results.get('trading_viability', {}).get('overall_score', 0.0)
                ) / 2.0,
                'clustering_consensus_score': (
                    nas_results.get('clustering_quality', {}).get('silhouette_score', 0.0) +
                    tas_results.get('clustering_quality', {}).get('silhouette_score', 0.0)
                ) / 2.0
            }
            
            return consensus_metrics
            
        except Exception as e:
            self.logger.warning(f"⚠️ Consensus metrics calculation failed: {e}")
            return {}
    
    def _calculate_disagreement_metrics(self, nas_results: Dict[str, Any], tas_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate disagreement metrics between NAS and TAS."""
        try:
            disagreement_metrics = {
                'economic_disagreement_score': abs(
                    nas_results.get('economic_significance', {}).get('overall_score', 0.0) -
                    tas_results.get('economic_significance', {}).get('overall_score', 0.0)
                ),
                'trading_disagreement_score': abs(
                    nas_results.get('trading_viability', {}).get('overall_score', 0.0) -
                    tas_results.get('trading_viability', {}).get('overall_score', 0.0)
                ),
                'regime_count_disagreement': abs(
                    nas_results.get('regime_count', 0) - tas_results.get('regime_count', 0)
                )
            }
            
            return disagreement_metrics
            
        except Exception as e:
            self.logger.warning(f"⚠️ Disagreement metrics calculation failed: {e}")
            return {}
    
    def _generate_consolidated_characteristics(self, nas_results: Dict[str, Any], tas_results: Dict[str, Any], 
                                             consolidated_assignments: np.ndarray) -> Dict[str, Any]:
        """Generate consolidated regime characteristics."""
        try:
            consolidated_characteristics = {}
            
            for regime_id in np.unique(consolidated_assignments):
                regime_mask = consolidated_assignments == regime_id
                regime_size = np.sum(regime_mask)
                
                consolidated_characteristics[f'regime_{regime_id}'] = {
                    'duration': regime_size,
                    'consolidated_from': 'nas_tas_hybrid',
                    'consensus_strength': 0.8,  # Placeholder
                    'economic_significance': 0.7,  # Placeholder
                    'trading_viability': 0.75  # Placeholder
                }
            
            return consolidated_characteristics
            
        except Exception as e:
            self.logger.warning(f"⚠️ Consolidated characteristics generation failed: {e}")
            return {}
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status.
        
        Returns:
            Pipeline status information
        """
        try:
            status = {
                'orchestrator_active': True,
                'config': {
                    'symbol': self.config.symbol,
                    'timeframe': self.config.timeframe,
                    'start_date': self.config.start_date,
                    'end_date': self.config.end_date
                },
                'component_status': {
                    'market_data_pipeline': self.market_data_pipeline_manager.get_pipeline_status(),
                    'feature_collection': True,
                    'economic_evaluation': True,
                    'trading_evaluation': True,
                    'search_strategies': True,
                    'evolutionary_algorithms': True,
                    'hardware_optimization': True,
                    'metrics_reporting': True
                },
                'timestamp': datetime.now().isoformat()
            }
            
            return status
            
        except Exception as e:
            self.logger.error(f"❌ Status retrieval failed: {e}")
            return {'orchestrator_active': False, 'error': str(e)}

    def orchestrate_tas_nas_detection(self,
                                    market_data: Union[pd.DataFrame, np.ndarray],
                                    timestamps: Optional[np.ndarray] = None,
                                    timeframes: Optional[List[str]] = None) -> Dict[str, Any]:
        """Orchestrate TAS and NAS regime detection."""
        execution_start_time = time.time()
        
        try:
            # Input validation with detailed error reporting
            self.logger.info("🔍 DEBUG: Starting orchestrate_tas_nas_detection with input validation")
            
            if market_data is None:
                error_msg = "❌ CRITICAL: market_data is None - cannot proceed with orchestration"
                self.logger.error(error_msg)
                tprint(f"[red]{error_msg}[/red]")
                return {'error': error_msg, 'execution_time': 0.0, 'success': False}
            
            # Validate data shape and type
            try:
                if hasattr(market_data, 'shape'):
                    data_shape = market_data.shape
                    self.logger.info(f"📊 Data shape: {data_shape}")
                    if len(data_shape) == 0 or (len(data_shape) > 0 and data_shape[0] == 0):
                        error_msg = f"❌ CRITICAL: Empty data - shape: {data_shape}"
                        self.logger.error(error_msg)
                        tprint(f"[red]{error_msg}[/red]")
                        return {'error': error_msg, 'execution_time': 0.0, 'success': False}
                else:
                    self.logger.warning("⚠️ market_data has no shape attribute")
                    data_shape = "Unknown"
            except Exception as e:
                error_msg = f"❌ CRITICAL: Error validating data shape: {e}"
                self.logger.error(error_msg)
                tprint(f"[red]{error_msg}[/red]")
                return {'error': error_msg, 'execution_time': 0.0, 'success': False}
            
            # Validate system availability
            tas_available = self.tas_system is not None
            nas_available = self.nas_system is not None
            
            self.logger.info(f"🔧 System availability - TAS: {tas_available}, NAS: {nas_available}")
            self.logger.info(f"🔧 TAS system object: {self.tas_system}")
            self.logger.info(f"🔧 NAS system object: {self.nas_system}")
            
            # Additional debug logging for TAS system
            if self.tas_system is not None:
                self.logger.info(f"🔧 TAS system type: {type(self.tas_system)}")
                self.logger.info(f"🔧 TAS system has detect_regimes: {hasattr(self.tas_system, 'detect_regimes')}")
            else:
                self.logger.error("❌ TAS system is None - this should not happen after successful initialization")
            
            if not tas_available and not nas_available:
                error_msg = "❌ CRITICAL: Neither TAS nor NAS system is available"
                self.logger.error(error_msg)
                tprint(f"[red]{error_msg}[/red]")
                return {'error': error_msg, 'execution_time': 0.0, 'success': False}
            
            # Print orchestration start panel with error handling
            try:
                tprint(Panel.fit(
                    "[bold green]🚀 Starting TAS-NAS Orchestration[/bold green]\n"
                    f"Data shape: {data_shape}\n"
                    f"Timeframes: {timeframes or ['1m', '5m', '15m']}\n"
                    f"TAS Available: {tas_available}\n"
                    f"NAS Available: {nas_available}",
                    title="Orchestration Start",
                    border_style="green"
                ))
            except Exception as e:
                self.logger.warning(f"⚠️ Error printing orchestration panel: {e}")
                tprint(f"[yellow]🚀 Starting TAS-NAS Orchestration (panel error: {e})[/yellow]")
            
            self.logger.info("🚀 Starting TAS-NAS orchestration...")

            # Use configured timeframes if not specified
            if timeframes is None:
                timeframes = ['1m', '5m', '15m']
                self.logger.info(f"📅 Using default timeframes: {timeframes}")

            results = {
                'tas_results': {},
                'nas_results': {},
                'hybrid_analysis': {},
                'timeframes_processed': timeframes,
                'execution_time': 0.0,
                'success': False,
                'errors': []
            }

            start_time = time.time()

            # Run detection for each timeframe with comprehensive error handling
            for i, timeframe in enumerate(timeframes):
                try:
                    self.logger.info(f"🔄 Processing timeframe {i+1}/{len(timeframes)}: {timeframe}")
                    self.logger.info(f"🔧 DEBUG: TAS system status before timeframe {timeframe}: {self.tas_system is not None}")
                    self.logger.info(f"🔧 DEBUG: NAS system status before timeframe {timeframe}: {self.nas_system is not None}")
                    tprint(f"[cyan]🔍 Processing timeframe: {timeframe}[/cyan]")

                    # Prepare market_data for timeframe with error handling
                    try:
                        tprint(f"[yellow]📊 Preparing data for {timeframe}...[/yellow]")
                        timeframe_market_data = self._prepare_timeframe_market_data(market_data, timeframe)
                        
                        if timeframe_market_data is None:
                            error_msg = f"❌ Failed to prepare data for timeframe {timeframe}"
                            self.logger.error(error_msg)
                            results['errors'].append(error_msg)
                            continue
                        
                        prep_shape = timeframe_market_data.shape if hasattr(timeframe_market_data, 'shape') else 'Unknown'
                        tprint(f"[green]✅ Data prepared: {prep_shape}[/green]")
                        self.logger.info(f"✅ Data prepared for {timeframe}: {prep_shape}")
                        
                    except Exception as e:
                        error_msg = f"❌ Error preparing data for {timeframe}: {e}"
                        self.logger.error(error_msg)
                        tprint(f"[red]{error_msg}[/red]")
                        results['errors'].append(error_msg)
                        continue

                    # Run TAS detection with comprehensive error handling
                    tprint(f"[bold yellow]🔍 === ORCHESTRATOR: TAS DETECTION START ===[/bold yellow]")
                    if self.tas_system is not None:
                        try:
                            tprint(f"[blue]🌳 Running TAS detection for {timeframe}...[/blue]")
                            tprint(f"[cyan]🔧 TAS system available: Yes[/cyan]")
                            tprint(f"[cyan]🔧 TAS system type: {type(self.tas_system).__name__}[/cyan]")
                            self.logger.info(f"🌳 Starting TAS detection for {timeframe}")
                            
                            tprint(f"[yellow]⏳ Calling _run_tas_detection...[/yellow]")
                            tas_result = self._run_tas_detection(timeframe_market_data, timestamps, timeframe)
                            tprint(f"[yellow]⏳ _run_tas_detection returned[/yellow]")
                            
                            tprint(f"[cyan]🔍 Checking TAS result...[/cyan]")
                            if tas_result is None:
                                error_msg = f"❌ TAS detection returned None for {timeframe}"
                                self.logger.error(error_msg)
                                tprint(f"[bold red]{error_msg}[/bold red]")
                                tas_result = {'error': error_msg, 'success': False}
                            elif 'error' in tas_result:
                                self.logger.error(f"❌ TAS detection error for {timeframe}: {tas_result['error']}")
                                tprint(f"[red]❌ TAS result contains error: {tas_result['error']}[/red]")
                            else:
                                tprint(f"[green]✅ TAS result has no error field[/green]")
                            
                            tprint(f"[cyan]🔍 TAS result keys: {list(tas_result.keys()) if isinstance(tas_result, dict) else 'Not a dict'}[/cyan]")
                            tprint(f"[cyan]🔍 TAS result success value: {tas_result.get('success', 'No success key') if isinstance(tas_result, dict) else 'N/A'}[/cyan]")
                            
                            results['tas_results'][timeframe] = tas_result
                            tprint(f"[green]✅ TAS result stored in results dict[/green]")
                            
                            if tas_result.get('success', False):
                                tprint(f"[bold green]✅ TAS detection completed successfully for {timeframe}[/bold green]")
                                self.logger.info(f"✅ TAS detection completed for {timeframe}")
                            else:
                                error_msg = f"❌ TAS detection failed for {timeframe}: {tas_result.get('error', 'Unknown error')}"
                                tprint(f"[bold red]{error_msg}[/bold red]")
                                
                                # Extract additional error details
                                if 'error_message' in tas_result:
                                    tprint(f"[red]❌ Error message: {tas_result['error_message']}[/red]")
                                if 'error_details' in tas_result:
                                    tprint(f"[red]❌ Error details: {tas_result['error_details']}[/red]")
                                if 'error_type' in tas_result:
                                    tprint(f"[red]❌ Error type: {tas_result['error_type']}[/red]")
                                if 'exception_type' in tas_result:
                                    tprint(f"[red]❌ Exception type: {tas_result['exception_type']}[/red]")
                                if 'traceback' in tas_result:
                                    tprint(f"[red]❌ Traceback available in result[/red]")
                                
                                self.logger.error(error_msg)
                                results['errors'].append(error_msg)
                                
                        except Exception as e:
                            error_msg = f"❌ Exception during TAS detection for {timeframe}: {e}"
                            tprint(f"[bold red]💥 EXCEPTION IN ORCHESTRATOR TAS CALL![/bold red]")
                            tprint(f"[red]❌ Exception type: {type(e).__name__}[/red]")
                            tprint(f"[red]❌ Exception message: {str(e)}[/red]")
                            
                            import traceback
                            traceback_str = traceback.format_exc()
                            tprint(f"[red]📋 Traceback:\n{traceback_str}[/red]")
                            
                            self.logger.error(error_msg)
                            tprint(f"[red]{error_msg}[/red]")
                            results['tas_results'][timeframe] = {'error': error_msg, 'success': False}
                            results['errors'].append(error_msg)
                    else:
                        tprint(f"[bold red]❌ TAS system is None![/bold red]")
                        self.logger.warning(f"⚠️ TAS system not available for {timeframe}")
                    
                    tprint(f"[bold yellow]🔍 === ORCHESTRATOR: TAS DETECTION END ===[/bold yellow]")

                    # Run NAS detection with comprehensive error handling
                    if self.nas_system is not None:
                        try:
                            tprint(f"[blue]🧠 Running NAS detection for {timeframe}...[/blue]")
                            self.logger.info(f"🧠 Starting NAS detection for {timeframe}")
                            
                            nas_result = self._run_nas_detection_sync(timeframe_market_data, timestamps, timeframe)
                            
                            if nas_result is None:
                                error_msg = f"❌ NAS detection returned None for {timeframe}"
                                self.logger.error(error_msg)
                                nas_result = {'error': error_msg, 'success': False}
                            elif 'error' in nas_result:
                                self.logger.error(f"❌ NAS detection error for {timeframe}: {nas_result['error']}")
                            
                            results['nas_results'][timeframe] = nas_result
                            
                            if nas_result.get('success', False):
                                tprint(f"[green]✅ NAS detection completed for {timeframe}[/green]")
                                self.logger.info(f"✅ NAS detection completed for {timeframe}")
                            else:
                                error_msg = f"❌ NAS detection failed for {timeframe}: {nas_result.get('error', 'Unknown error')}"
                                tprint(f"[red]{error_msg}[/red]")
                                self.logger.error(error_msg)
                                results['errors'].append(error_msg)
                                
                        except Exception as e:
                            error_msg = f"❌ Exception during NAS detection for {timeframe}: {e}"
                            self.logger.error(error_msg)
                            tprint(f"[red]{error_msg}[/red]")
                            results['nas_results'][timeframe] = {'error': error_msg, 'success': False}
                            results['errors'].append(error_msg)
                    else:
                        self.logger.warning(f"⚠️ NAS system not available for {timeframe}")

                except Exception as e:
                    error_msg = f"❌ Exception processing timeframe {timeframe}: {e}"
                    self.logger.error(error_msg)
                    tprint(f"[red]{error_msg}[/red]")
                    results['errors'].append(error_msg)
                    continue

            # Perform hybrid analysis on primary timeframe (15m) with enhanced error handling
            primary_timeframe = '15m'
            tprint(f"[magenta]🔬 Checking hybrid analysis prerequisites for {primary_timeframe}...[/magenta]")
            self.logger.info(f"🔬 Checking hybrid analysis prerequisites for {primary_timeframe}")
            
            try:
                tas_success = (primary_timeframe in results.get('tas_results', {}) and
                              results['tas_results'][primary_timeframe].get('success', False))
                nas_success = (primary_timeframe in results.get('nas_results', {}) and
                              results['nas_results'][primary_timeframe].get('success', False))

                self.logger.info(f"📊 Success status - TAS: {tas_success}, NAS: {nas_success}")
                tprint(f"[cyan]TAS Success: {tas_success}[/cyan]")
                tprint(f"[cyan]NAS Success: {nas_success}[/cyan]")

                if tas_success and nas_success:
                    try:
                        tprint("[bold green]🎯 Starting hybrid analysis...[/bold green]")
                        self.logger.info("🎯 Starting hybrid analysis")
                        
                        hybrid_analysis = self._perform_hybrid_analysis(
                            market_data, timestamps,
                            results['tas_results'][primary_timeframe],
                            results['nas_results'][primary_timeframe]
                        )
                        
                        if hybrid_analysis is None:
                            error_msg = "❌ Hybrid analysis returned None"
                            self.logger.error(error_msg)
                            hybrid_analysis = {'error': error_msg, 'success': False}
                        elif 'error' in hybrid_analysis:
                            self.logger.error(f"❌ Hybrid analysis error: {hybrid_analysis['error']}")
                        
                        results['hybrid_analysis'] = hybrid_analysis
                        
                        if hybrid_analysis.get('success', False):
                            tprint("[green]✅ Hybrid analysis completed[/green]")
                            self.logger.info("✅ Hybrid analysis completed successfully")
                        else:
                            error_msg = f"❌ Hybrid analysis failed: {hybrid_analysis.get('error', 'Unknown error')}"
                            tprint(f"[red]{error_msg}[/red]")
                            self.logger.error(error_msg)
                            results['errors'].append(error_msg)
                            
                    except Exception as e:
                        error_msg = f"❌ Exception during hybrid analysis: {e}"
                        self.logger.error(error_msg)
                        tprint(f"[red]{error_msg}[/red]")
                        results['hybrid_analysis'] = {'error': error_msg, 'success': False}
                        results['errors'].append(error_msg)
                else:
                    # Enhanced error reporting for failed systems
                    tprint(f"[bold yellow]🔍 === ERROR REPORTING: FAILED SYSTEMS ===[/bold yellow]")
                    tprint(f"[cyan]📍 Primary timeframe: {primary_timeframe}[/cyan]")
                    tprint(f"[cyan]📍 TAS success: {tas_success}[/cyan]")
                    tprint(f"[cyan]📍 NAS success: {nas_success}[/cyan]")
                    
                    failed_systems = []
                    if not tas_success:
                        tprint(f"[yellow]🔍 Extracting TAS error details...[/yellow]")
                        tprint(f"[cyan]📊 results['tas_results'] = {results.get('tas_results', 'KEY_NOT_FOUND')}[/cyan]")
                        
                        tas_result_for_timeframe = results.get('tas_results', {}).get(primary_timeframe, {})
                        tprint(f"[cyan]📊 TAS result for {primary_timeframe} = {tas_result_for_timeframe}[/cyan]")
                        
                        tas_error = tas_result_for_timeframe.get('error', 'Unknown TAS error')
                        tprint(f"[red]❌ TAS error: {tas_error}[/red]")
                        
                        # Check for additional error fields
                        if 'error_message' in tas_result_for_timeframe:
                            tprint(f"[red]❌ TAS error_message: {tas_result_for_timeframe['error_message']}[/red]")
                            tas_error = tas_result_for_timeframe['error_message']
                        if 'error_details' in tas_result_for_timeframe:
                            tprint(f"[red]❌ TAS error_details: {tas_result_for_timeframe['error_details']}[/red]")
                        if 'error_type' in tas_result_for_timeframe:
                            tprint(f"[red]❌ TAS error_type: {tas_result_for_timeframe['error_type']}[/red]")
                        if 'exception_type' in tas_result_for_timeframe:
                            tprint(f"[red]❌ TAS exception_type: {tas_result_for_timeframe['exception_type']}[/red]")
                        if 'exception_details' in tas_result_for_timeframe:
                            tprint(f"[red]❌ TAS exception_details: {tas_result_for_timeframe['exception_details']}[/red]")
                        if 'traceback' in tas_result_for_timeframe:
                            tprint(f"[red]📋 TAS traceback:\n{tas_result_for_timeframe['traceback']}[/red]")
                        
                        failed_systems.append(f"TAS: {tas_error}")
                        
                    if not nas_success:
                        tprint(f"[yellow]🔍 Extracting NAS error details...[/yellow]")
                        tprint(f"[cyan]📊 results['nas_results'] = {results.get('nas_results', 'KEY_NOT_FOUND')}[/cyan]")
                        
                        nas_result_for_timeframe = results.get('nas_results', {}).get(primary_timeframe, {})
                        tprint(f"[cyan]📊 NAS result for {primary_timeframe} = {nas_result_for_timeframe}[/cyan]")
                        
                        nas_error = nas_result_for_timeframe.get('error', 'Unknown NAS error')
                        tprint(f"[red]❌ NAS error: {nas_error}[/red]")
                        
                        failed_systems.append(f"NAS: {nas_error}")
                    
                    error_msg = f"❌ Cannot perform hybrid analysis - Failed systems: {'; '.join(failed_systems)}"
                    self.logger.error(error_msg)
                    tprint(f"[bold red]{error_msg}[/bold red]")
                    tprint(f"[bold yellow]🔍 === ERROR REPORTING END ===[/bold yellow]")
                    results['errors'].append(error_msg)
                    
                    # Don't raise exception - continue with partial results
                    self.logger.info("ℹ️ Continuing with partial results despite hybrid analysis failure")
                    
            except Exception as e:
                error_msg = f"❌ Exception checking hybrid analysis prerequisites: {e}"
                self.logger.error(error_msg)
                tprint(f"[red]{error_msg}[/red]")
                results['errors'].append(error_msg)

            # Generate comprehensive outcome report with error handling
            if tas_success and nas_success:
                try:
                    tprint("[bold magenta]📊 Generating comprehensive outcome report...[/bold magenta]")
                    self.logger.info("📊 Generating comprehensive outcome report")
                    
                    outcome_report = self._generate_outcome_report(results, market_data)
                    
                    if outcome_report is None:
                        error_msg = "❌ Outcome report generation returned None"
                        self.logger.error(error_msg)
                        outcome_report = {'error': error_msg}
                    elif 'error' in outcome_report:
                        self.logger.error(f"❌ Outcome report generation error: {outcome_report['error']}")
                    
                    results['outcome_report'] = outcome_report
                    
                    if 'error' not in outcome_report:
                        tprint("[green]✅ Outcome report generated[/green]")
                        self.logger.info("✅ Outcome report generated successfully")
                    else:
                        error_msg = f"❌ Outcome report generation failed: {outcome_report.get('error', 'Unknown error')}"
                        tprint(f"[red]{error_msg}[/red]")
                        self.logger.error(error_msg)
                        results['errors'].append(error_msg)
                        
                except Exception as e:
                    error_msg = f"❌ Exception generating outcome report: {e}"
                    self.logger.error(error_msg)
                    tprint(f"[red]{error_msg}[/red]")
                    results['outcome_report'] = {'error': error_msg}
                    results['errors'].append(error_msg)

            # Add clustering quality metrics with comprehensive error handling
            if tas_success and nas_success and self.clustering_quality_analyzer:
                try:
                    self.logger.info("🔍 Starting clustering quality analysis")
                    tprint("[blue]🔍 Starting clustering quality analysis...[/blue]")

                    # Prepare features for quality analysis with error handling
                    features = None
                    try:
                        if isinstance(market_data, pd.DataFrame):
                            numeric_columns = market_data.select_dtypes(include=[np.number]).columns
                            if len(numeric_columns) > 0:
                                features = market_data[numeric_columns].values
                                self.logger.info(f"📊 Using {len(numeric_columns)} numeric columns for clustering quality analysis")
                            else:
                                # Fallback to basic OHLCV columns
                                basic_columns = ['open', 'high', 'low', 'close', 'volume']
                                available_columns = [col for col in basic_columns if col in market_data.columns]
                                if available_columns:
                                    features = market_data[available_columns].values
                                    self.logger.info(f"📊 Using basic OHLCV columns: {available_columns}")
                                else:
                                    features = market_data.values
                                    self.logger.info("📊 Using all DataFrame values")
                        else:
                            features = market_data
                            self.logger.info("📊 Using numpy array as features")
                        
                        if features is None or (hasattr(features, 'shape') and features.shape[0] == 0):
                            raise ValueError("Features are None or empty")
                            
                    except Exception as e:
                        error_msg = f"❌ Error preparing features for clustering quality analysis: {e}"
                        self.logger.error(error_msg)
                        results['clustering_quality'] = {'error': error_msg}
                        results['errors'].append(error_msg)
                        features = None

                    if features is not None:
                        # Calculate clustering quality for each successful prediction set
                        clustering_quality = {}

                        # TAS quality analysis
                        try:
                            tas_predictions = results['tas_results'][primary_timeframe]['regime_predictions']
                            self.logger.info(f"🔍 Analyzing TAS quality with {len(tas_predictions)} predictions")
                            
                            # Ensure features and predictions have the same length for TAS
                            tas_features = features
                            if len(tas_features) != len(tas_predictions):
                                min_length = min(len(tas_features), len(tas_predictions))
                                tas_features = tas_features[:min_length]
                                tas_predictions = tas_predictions[:min_length]
                                self.logger.info(f"📏 Adjusted TAS features/predictions to length: {min_length}")

                            tas_quality_metrics = self.clustering_quality_analyzer.calculate_comprehensive_metrics(
                                tas_features, tas_predictions
                            )
                            clustering_quality['tas_quality'] = tas_quality_metrics
                            self.logger.info("✅ TAS quality analysis completed")
                            
                        except Exception as e:
                            error_msg = f"❌ TAS quality analysis failed: {e}"
                            self.logger.error(error_msg)
                            clustering_quality['tas_quality'] = {'error': error_msg}
                            results['errors'].append(error_msg)

                        # NAS quality analysis
                        try:
                            nas_predictions = results['nas_results'][primary_timeframe]['regime_predictions']
                            self.logger.info(f"🔍 Analyzing NAS quality with {len(nas_predictions)} predictions")
                            
                            # Ensure features and predictions have the same length for NAS
                            nas_features = features
                            if len(nas_features) != len(nas_predictions):
                                min_length = min(len(nas_features), len(nas_predictions))
                                nas_features = nas_features[:min_length]
                                nas_predictions = nas_predictions[:min_length]
                                self.logger.info(f"📏 Adjusted NAS features/predictions to length: {min_length}")

                            nas_quality_metrics = self.clustering_quality_analyzer.calculate_comprehensive_metrics(
                                nas_features, nas_predictions
                            )
                            clustering_quality['nas_quality'] = nas_quality_metrics
                            self.logger.info("✅ NAS quality analysis completed")
                            
                        except Exception as e:
                            error_msg = f"❌ NAS quality analysis failed: {e}"
                            self.logger.error(error_msg)
                            clustering_quality['nas_quality'] = {'error': error_msg}
                            results['errors'].append(error_msg)

                        # Add comparison if both succeeded
                        try:
                            tas_quality = clustering_quality.get('tas_quality', {})
                            nas_quality = clustering_quality.get('nas_quality', {})

                            comparison = {}
                            if tas_quality and nas_quality and 'error' not in tas_quality and 'error' not in nas_quality:
                                comparison = {
                                    'best_silhouette': 'TAS' if tas_quality.get('silhouette_score', 0) > nas_quality.get('silhouette_score', 0) else 'NAS',
                                    'best_davies_bouldin': 'TAS' if tas_quality.get('davies_bouldin_index', float('inf')) < nas_quality.get('davies_bouldin_index', float('inf')) else 'NAS',
                                    'best_calinski_harabasz': 'TAS' if tas_quality.get('calinski_harabasz_score', 0) > nas_quality.get('calinski_harabasz_score', 0) else 'NAS'
                                }
                                self.logger.info("✅ Quality comparison completed")
                            else:
                                comparison = {'error': 'One or both quality analyses failed'}
                                self.logger.warning("⚠️ Quality comparison failed - one or both analyses had errors")
                                
                            clustering_quality['comparison'] = comparison

                            # Add to hybrid analysis and main results
                            results['hybrid_analysis']['clustering_quality'] = clustering_quality
                            results['clustering_quality'] = clustering_quality

                            tprint("[green]✅ Clustering quality analysis completed[/green]")
                            self.logger.info("✅ Clustering quality analysis completed successfully")
                            
                        except Exception as e:
                            error_msg = f"❌ Quality comparison failed: {e}"
                            self.logger.error(error_msg)
                            clustering_quality['comparison'] = {'error': error_msg}
                            results['errors'].append(error_msg)

                except Exception as e:
                    error_msg = f"❌ Exception during clustering quality analysis: {e}"
                    self.logger.error(error_msg)
                    tprint(f"[red]{error_msg}[/red]")
                    results['hybrid_analysis']['clustering_quality'] = {'error': error_msg}
                    results['clustering_quality'] = {'error': error_msg}
                    results['errors'].append(error_msg)
            else:
                self.logger.info("ℹ️ Skipping clustering quality analysis - prerequisites not met")

            results['execution_time'] = time.time() - start_time

            # Determine overall success
            has_successful_detection = (
                any(result.get('success', False) for result in results.get('tas_results', {}).values()) or
                any(result.get('success', False) for result in results.get('nas_results', {}).values())
            )
            results['success'] = has_successful_detection
            
            if has_successful_detection:
                self.logger.info("✅ TAS-NAS orchestration completed with at least one successful detection")
                tprint("[green]✅ TAS-NAS orchestration completed with partial success[/green]")
            else:
                self.logger.error("❌ TAS-NAS orchestration completed with no successful detections")
                tprint("[red]❌ TAS-NAS orchestration completed with no successful detections[/red]")

            # Log summary
            tas_success_count = sum(1 for result in results.get('tas_results', {}).values() if result.get('success', False))
            nas_success_count = sum(1 for result in results.get('nas_results', {}).values() if result.get('success', False))
            total_errors = len(results.get('errors', []))
            
            self.logger.info(f"📊 Orchestration Summary:")
            self.logger.info(f"   - TAS successful detections: {tas_success_count}/{len(timeframes)}")
            self.logger.info(f"   - NAS successful detections: {nas_success_count}/{len(timeframes)}")
            self.logger.info(f"   - Total errors: {total_errors}")
            self.logger.info(f"   - Execution time: {results['execution_time']:.2f}s")
            
            return results

        except Exception as e:
            execution_time = time.time() - execution_start_time
            error_msg = f"❌ CRITICAL: TAS-NAS orchestration failed with exception: {e}"
            self.logger.error(error_msg)
            tprint(f"[red]{error_msg}[/red]")
            
            # Return detailed error information
            return {
                'error': error_msg,
                'execution_time': execution_time,
                'success': False,
                'tas_results': {},
                'nas_results': {},
                'hybrid_analysis': {},
                'timeframes_processed': timeframes or [],
                'errors': [error_msg]
            }


    def _generate_outcome_report(self, results: Dict[str, Any], market_data: Union[pd.DataFrame, np.ndarray]) -> Dict[str, Any]:
        """Generate comprehensive outcome report with input data info, NAS vs TAS analysis, and clear output metrics."""
        try:
            tprint(Panel.fit(
                "[bold magenta]📊 Comprehensive Outcome Report[/bold magenta]",
                title="Report Generation",
                border_style="magenta"
            ))
            
            # 1. Input Data Information
            input_data_info = self._analyze_input_data(market_data)
            
            # 2. NAS vs TAS Analysis and Comparison
            nas_tas_comparison = self._analyze_nas_tas_comparison(results)
            
            # 3. Output Metrics (clusters, CV, Silhouette, etc.)
            output_metrics = self._analyze_output_metrics(results)
            
            # Create comprehensive report
            outcome_report = {
                'input_data_analysis': input_data_info,
                'nas_tas_comparison': nas_tas_comparison,
                'output_metrics': output_metrics,
                'summary': self._generate_summary_table(input_data_info, nas_tas_comparison, output_metrics)
            }
            
            # Display the report
            self._display_outcome_report(outcome_report)
            
            return outcome_report
            
        except Exception as e:
            self.logger.error(f"❌ Outcome report generation failed: {e}")
            return {'error': str(e)}

    def _analyze_input_data(self, market_data: Union[pd.DataFrame, np.ndarray]) -> Dict[str, Any]:
        """Analyze input data characteristics."""
        try:
            if isinstance(market_data, pd.DataFrame):
                data_shape = market_data.shape
                columns = list(market_data.columns)
                numeric_columns = list(market_data.select_dtypes(include=[np.number]).columns)
                
                # Basic statistics
                if 'close' in market_data.columns:
                    close_stats = {
                        'mean': float(market_data['close'].mean()),
                        'std': float(market_data['close'].std()),
                        'min': float(market_data['close'].min()),
                        'max': float(market_data['close'].max())
                    }
                else:
                    close_stats = None
                
                if 'volume' in market_data.columns:
                    volume_stats = {
                        'mean': float(market_data['volume'].mean()),
                        'std': float(market_data['volume'].std()),
                        'min': float(market_data['volume'].min()),
                        'max': float(market_data['volume'].max())
                    }
                else:
                    volume_stats = None
                
                return {
                    'data_type': 'DataFrame',
                    'shape': data_shape,
                    'columns': columns,
                    'numeric_columns': numeric_columns,
                    'close_statistics': close_stats,
                    'volume_statistics': volume_stats,
                    'data_quality': 'Good' if not market_data.isnull().any().any() else 'Has missing values'
                }
            else:
                return {
                    'data_type': 'NumPy Array',
                    'shape': market_data.shape if hasattr(market_data, 'shape') else 'Unknown',
                    'dtype': str(market_data.dtype) if hasattr(market_data, 'dtype') else 'Unknown'
                }
                
        except Exception as e:
            return {'error': f"Input data analysis failed: {e}"}

    def _analyze_nas_tas_comparison(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze and compare NAS vs TAS results."""
        try:
            primary_timeframe = '15m'
            tas_results = results.get('tas_results', {}).get(primary_timeframe, {})
            nas_results = results.get('nas_results', {}).get(primary_timeframe, {})
            
            # Fast fail if required keys are missing
            if tas_results.get('success', False) and 'regime_assignments' not in tas_results:
                raise KeyError("TAS results missing 'regime_assignments' key - this indicates a critical configuration error")
            
            if nas_results.get('success', False) and 'regime_predictions' not in nas_results:
                raise KeyError("NAS results missing 'regime_predictions' key - this indicates a critical configuration error")
            
            tas_assignments = tas_results.get('regime_assignments', [])
            nas_assignments = nas_results.get('regime_predictions', [])
            
            comparison = {
                'tas_analysis': {
                    'success': tas_results.get('success', False),
                    'regime_count': len(np.unique(tas_assignments)) if tas_results.get('success') and len(tas_assignments) > 0 else 0,
                    'execution_time': tas_results.get('execution_time', 0),
                    'regime_distribution': self._calculate_regime_distribution(tas_assignments) if tas_results.get('success') and len(tas_assignments) > 0 else {}
                },
                'nas_analysis': {
                    'success': nas_results.get('success', False),
                    'regime_count': len(np.unique(nas_assignments)) if nas_results.get('success') and len(nas_assignments) > 0 else 0,
                    'execution_time': nas_results.get('execution_time', 0),
                    'regime_distribution': self._calculate_regime_distribution(nas_assignments) if nas_results.get('success') and len(nas_assignments) > 0 else {}
                }
            }
            
            # Calculate agreement metrics using semantic mapping
            if comparison['tas_analysis']['success'] and comparison['nas_analysis']['success']:
                tas_preds = np.array(tas_assignments)
                nas_preds = np.array(nas_assignments)
                
                min_len = min(len(tas_preds), len(nas_preds))
                if min_len > 0:
                    tas_preds = tas_preds[:min_len]
                    nas_preds = nas_preds[:min_len]
                    
                    # Perform semantic divergence assessment to get proper regime mapping
                    semantic_assessment = self._perform_semantic_divergence_assessment(
                        tas_preds, nas_preds, min_len
                    )
                    
                    # Use semantic consensus as the primary agreement rate
                    comparison['agreement_metrics'] = {
                        'agreement_rate': float(semantic_assessment.get('semantic_consensus', 0.0)),
                        'raw_agreement_rate': float(semantic_assessment.get('raw_consensus', 0.0)),
                        'semantic_consensus': float(semantic_assessment.get('semantic_consensus', 0.0)),
                        'consensus_improvement': float(semantic_assessment.get('consensus_improvement', 0.0)),
                        'mapping_quality': float(semantic_assessment.get('mapping_quality', 0.0)),
                        'total_samples': min_len,
                        'matching_samples': int(semantic_assessment.get('semantic_consensus', 0.0) * min_len),
                        'raw_matching_samples': int(semantic_assessment.get('raw_consensus', 0.0) * min_len),
                        'regime_mapping': semantic_assessment.get('regime_mapping', {}),
                        'assessment_method': semantic_assessment.get('assessment_method', 'unknown')
                    }
                else:
                    comparison['agreement_metrics'] = {'error': 'No overlapping predictions'}
            else:
                comparison['agreement_metrics'] = {'error': 'One or both systems failed'}
            
            return comparison
            
        except Exception as e:
            return {'error': f"NAS vs TAS comparison failed: {e}"}

    def _analyze_output_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze output metrics for each cluster and on average."""
        try:
            hybrid_analysis = results.get('hybrid_analysis', {})
            
            if not hybrid_analysis.get('success', False):
                return {'error': 'Hybrid analysis not available'}
            
            # Get consolidated regime predictions
            hybrid_labels = hybrid_analysis.get('hybrid_labels', [])
            if len(hybrid_labels) == 0:
                return {'error': 'No hybrid labels available'}
            
            unique_regimes = np.unique(hybrid_labels)
            num_regimes = len(unique_regimes)
            
            # Calculate metrics for each regime
            regime_metrics = {}
            for regime in unique_regimes:
                regime_mask = hybrid_labels == regime
                regime_size = np.sum(regime_mask)
                regime_percentage = (regime_size / len(hybrid_labels)) * 100
                
                regime_metrics[f'regime_{regime}'] = {
                    'size': int(regime_size),
                    'percentage': float(regime_percentage),
                    'is_valid_size': 3.0 <= regime_percentage <= 25.0  # 3% to 25% target
                }
            
            # Overall metrics
            overall_metrics = {
                'total_regimes': num_regimes,
                'target_range_met': 6 <= num_regimes <= 15,
                'regime_distribution': self._calculate_regime_distribution(hybrid_labels),
                'clustering_quality': hybrid_analysis.get('clustering_quality', {}),
                'economic_evaluation': hybrid_analysis.get('economic_evaluation', {}),
                'validation_result': hybrid_analysis.get('validation_result', {})
            }
            
            # Calculate average metrics
            if 'clustering_quality' in hybrid_analysis:
                clustering_quality = hybrid_analysis['clustering_quality']
                avg_metrics = {
                    'silhouette_score': clustering_quality.get('silhouette_score', 0),
                    'calinski_harabasz_score': clustering_quality.get('calinski_harabasz_score', 0),
                    'davies_bouldin_score': clustering_quality.get('davies_bouldin_score', 0)
                }
                overall_metrics['average_clustering_metrics'] = avg_metrics
            
            return {
                'regime_metrics': regime_metrics,
                'overall_metrics': overall_metrics
            }
            
        except Exception as e:
            return {'error': f"Output metrics analysis failed: {e}"}

    def _calculate_regime_distribution(self, regime_predictions: np.ndarray) -> Dict[str, float]:
        """Calculate regime distribution percentages."""
        try:
            if len(regime_predictions) == 0:
                return {}
            
            unique_regimes, counts = np.unique(regime_predictions, return_counts=True)
            total_samples = len(regime_predictions)
            
            # Convert numpy int64 keys to regular Python ints for JSON serialization
            distribution = {}
            for regime, count in zip(unique_regimes, counts):
                percentage = (count / total_samples) * 100
                distribution[int(regime)] = int(count)
                distribution[f'regime_{int(regime)}'] = float(percentage)
            
            return distribution
            
        except Exception as e:
            return {'error': f"Distribution calculation failed: {e}"}

    def _generate_summary_table(self, input_data_info: Dict, nas_tas_comparison: Dict, output_metrics: Dict) -> Dict[str, Any]:
        """Generate a summary table of key metrics."""
        try:
            summary = {
                'data_summary': {
                    'data_shape': input_data_info.get('shape', 'Unknown'),
                    'data_quality': input_data_info.get('data_quality', 'Unknown')
                },
                'detection_summary': {
                    'tas_success': nas_tas_comparison.get('tas_analysis', {}).get('success', False),
                    'nas_success': nas_tas_comparison.get('nas_analysis', {}).get('success', False),
                    'tas_regimes': nas_tas_comparison.get('tas_analysis', {}).get('regime_count', 0),
                    'nas_regimes': nas_tas_comparison.get('nas_analysis', {}).get('regime_count', 0)
                },
                'output_summary': {
                    'final_regimes': output_metrics.get('overall_metrics', {}).get('total_regimes', 0),
                    'target_range_met': output_metrics.get('overall_metrics', {}).get('target_range_met', False),
                    'agreement_rate': nas_tas_comparison.get('agreement_metrics', {}).get('agreement_rate', 0)
                }
            }
            
            return summary
            
        except Exception as e:
            return {'error': f"Summary generation failed: {e}"}

    def _display_outcome_report(self, outcome_report: Dict[str, Any]) -> None:
        """Display the comprehensive outcome report using rich formatting."""
        try:
            # Input Data Information
            tprint("\n" + "="*80)
            tprint(Panel.fit(
                "[bold blue]📊 INPUT DATA ANALYSIS[/bold blue]",
                border_style="blue"
            ))
            
            input_data = outcome_report.get('input_data_analysis', {})
            if 'error' not in input_data:
                tprint(f"Data Type: {input_data.get('data_type', 'Unknown')}")
                tprint(f"Shape: {input_data.get('shape', 'Unknown')}")
                tprint(f"Columns: {len(input_data.get('columns', []))}")
                tprint(f"Data Quality: {input_data.get('data_quality', 'Unknown')}")
                
                if input_data.get('close_statistics'):
                    close_stats = input_data['close_statistics']
                    tprint(f"Close Price - Mean: {close_stats['mean']:.2f}, Std: {close_stats['std']:.2f}")
            else:
                tprint(f"[red]Input data analysis error: {input_data['error']}[/red]")
            
            # NAS vs TAS Comparison
            tprint("\n" + "="*80)
            tprint(Panel.fit(
                "[bold green]🔄 NAS vs TAS COMPARISON[/bold green]",
                border_style="green"
            ))
            
            comparison = outcome_report.get('nas_tas_comparison', {})
            if 'error' not in comparison:
                tas_analysis = comparison.get('tas_analysis', {})
                nas_analysis = comparison.get('nas_analysis', {})
                
                tprint(f"TAS Success: {tas_analysis.get('success', False)}")
                tprint(f"TAS Regimes: {tas_analysis.get('regime_count', 0)}")
                tprint(f"TAS Execution Time: {tas_analysis.get('execution_time', 0):.2f}s")
                
                tprint(f"NAS Success: {nas_analysis.get('success', False)}")
                tprint(f"NAS Regimes: {nas_analysis.get('regime_count', 0)}")
                tprint(f"NAS Execution Time: {nas_analysis.get('execution_time', 0):.2f}s")
                
                agreement_metrics = comparison.get('agreement_metrics', {})
                if 'error' not in agreement_metrics:
                    tprint(f"[bold cyan]Semantic Agreement Rate: {agreement_metrics.get('semantic_consensus', 0):.2%}[/bold cyan]")
                    tprint(f"Semantic Matching Samples: {agreement_metrics.get('matching_samples', 0)}/{agreement_metrics.get('total_samples', 0)}")
                    tprint(f"Raw Agreement Rate: {agreement_metrics.get('raw_agreement_rate', 0):.2%} (without mapping)")
                    tprint(f"Raw Matching Samples: {agreement_metrics.get('raw_matching_samples', 0)}/{agreement_metrics.get('total_samples', 0)}")
                    tprint(f"[green]Consensus Improvement: +{agreement_metrics.get('consensus_improvement', 0):.2%}[/green]")
                    tprint(f"Mapping Quality: {agreement_metrics.get('mapping_quality', 0):.2%}")
                    tprint(f"Assessment Method: {agreement_metrics.get('assessment_method', 'unknown')}")
                    
                    # Show regime mapping if available
                    regime_mapping = agreement_metrics.get('regime_mapping', {})
                    if regime_mapping:
                        tprint(f"\n[bold]Regime Mapping (NAS→TAS):[/bold]")
                        for nas_regime, tas_regime in sorted(regime_mapping.items()):
                            tprint(f"  NAS Regime {nas_regime} → TAS Regime {tas_regime}")
                else:
                    tprint(f"[red]Agreement analysis error: {agreement_metrics['error']}[/red]")
            else:
                tprint(f"[red]NAS vs TAS comparison error: {comparison['error']}[/red]")
            
            # Output Metrics
            tprint("\n" + "="*80)
            tprint(Panel.fit(
                "[bold magenta]📈 OUTPUT METRICS[/bold magenta]",
                border_style="magenta"
            ))
            
            output_metrics = outcome_report.get('output_metrics', {})
            if 'error' not in output_metrics:
                overall_metrics = output_metrics.get('overall_metrics', {})
                regime_metrics = output_metrics.get('regime_metrics', {})
                
                tprint(f"Final Regime Count: {overall_metrics.get('total_regimes', 0)}")
                tprint(f"Target Range Met (6-15): {overall_metrics.get('target_range_met', False)}")
                
                # Display regime-specific metrics
                tprint("\n[bold]Regime-Specific Metrics:[/bold]")
                for regime_key, metrics in regime_metrics.items():
                    if 'error' not in metrics:
                        tprint(f"{regime_key}: Size={metrics['size']}, Percentage={metrics['percentage']:.1f}%, Valid={metrics['is_valid_size']}")
                
                # Display clustering quality metrics
                clustering_quality = overall_metrics.get('clustering_quality', {})
                if clustering_quality:
                    tprint(f"\n[bold]Clustering Quality:[/bold]")
                    tprint(f"Silhouette Score: {clustering_quality.get('silhouette_score', 0):.3f}")
                    tprint(f"Calinski-Harabasz Score: {clustering_quality.get('calinski_harabasz_score', 0):.3f}")
                    tprint(f"Davies-Bouldin Score: {clustering_quality.get('davies_bouldin_score', 0):.3f}")
            else:
                tprint(f"[red]Output metrics error: {output_metrics['error']}[/red]")
            
            # Summary
            tprint("\n" + "="*80)
            tprint(Panel.fit(
                "[bold yellow]📋 SUMMARY[/bold yellow]",
                border_style="yellow"
            ))
            
            summary = outcome_report.get('summary', {})
            if 'error' not in summary:
                data_summary = summary.get('data_summary', {})
                detection_summary = summary.get('detection_summary', {})
                output_summary = summary.get('output_summary', {})
                
                tprint(f"Data Shape: {data_summary.get('data_shape', 'Unknown')}")
                tprint(f"TAS Success: {detection_summary.get('tas_success', False)}")
                tprint(f"NAS Success: {detection_summary.get('nas_success', False)}")
                tprint(f"Final Regimes: {output_summary.get('final_regimes', 0)}")
                tprint(f"Target Range Met: {output_summary.get('target_range_met', False)}")
                tprint(f"[bold cyan]Semantic Agreement Rate: {output_summary.get('agreement_rate', 0):.2%}[/bold cyan]")
            else:
                tprint(f"[red]Summary error: {summary['error']}[/red]")
            
            tprint("\n" + "="*80)
            
        except Exception as e:
            tprint(f"[red]❌ Report display failed: {e}[/red]")

    def _prepare_timeframe_market_data(self, market_data: Union[pd.DataFrame, np.ndarray],
                               timeframe: str) -> Union[pd.DataFrame, np.ndarray]:
        """Prepare market_data for specific timeframe."""
        try:
            if isinstance(market_data, np.ndarray):
                # For numpy arrays, resample based on timeframe
                if timeframe == '1m':
                    return market_data
                elif timeframe == '5m':
                    if len(market_data) >= 5:
                        indices = range(0, len(market_data), 5)
                        return market_data[indices]
                    else:
                        return market_data
                elif timeframe == '15m':
                    if len(market_data) >= 15:
                        indices = range(0, len(market_data), 15)
                        return market_data[indices]
                    else:
                        return market_data
                else:
                    return market_data

            elif isinstance(market_data, pd.DataFrame):
                # For DataFrame, check if resampling is needed
                if 'timestamp' in market_data.columns:
                    market_data = market_data.set_index('timestamp')
                
                # Check if market_data is already at the correct timeframe
                if len(market_data) > 1:
                    time_diff = market_data.index[1] - market_data.index[0]
                    actual_timeframe_minutes = int(time_diff.total_seconds() / 60)
                    
                    # Map minutes to timeframe strings
                    timeframe_map = {1: '1m', 5: '5m', 15: '15m', 30: '30m', 60: '1h', 240: '4h', 1440: '1d'}
                    actual_timeframe = timeframe_map.get(actual_timeframe_minutes, f'{actual_timeframe_minutes}m')
                    
                    if actual_timeframe == timeframe:
                        # Data is already at the correct timeframe
                        return market_data.reset_index()
                
                # Only resample if needed
                if timeframe == '1m':
                    return market_data.reset_index()
                else:
                    resampled = market_data.resample(timeframe).agg({
                        'open': 'first',
                        'high': 'max',
                        'low': 'min',
                        'close': 'last',
                        'volume': 'sum'
                    }).dropna()
                    return resampled.reset_index()

            return market_data

        except Exception as e:
            self.logger.warning(f"⚠️ Timeframe market_data preparation failed: {e}")
            return market_data

    def _run_tas_detection(self, market_data: Union[pd.DataFrame, np.ndarray],
                          timestamps: Optional[np.ndarray], timeframe: str) -> Dict[str, Any]:
        """Run TAS regime detection with comprehensive error handling."""
        try:
            tprint(f"[bold cyan]🔍 === TAS DETECTION START ===[/bold cyan]")
            tprint(f"[cyan]📍 Timeframe: {timeframe}[/cyan]")
            self.logger.info(f"🌳 Starting TAS detection for {timeframe}")
            
            tprint(f"[yellow]🔧 Step 1: Checking TAS system initialization...[/yellow]")
            if self.tas_system is None:
                error_msg = 'TAS system not initialized'
                self.logger.error(f"❌ {error_msg}")
                tprint(f"[bold red]❌ CRITICAL: TAS system is None![/bold red]")
                return {'error': error_msg, 'success': False, 'timeframe': timeframe, 'system': 'TAS'}
            
            tprint(f"[green]✅ TAS system is initialized: {type(self.tas_system).__name__}[/green]")
            tprint(f"[cyan]🔧 TAS system has detect_regimes: {hasattr(self.tas_system, 'detect_regimes')}[/cyan]")

            # Validate input data
            tprint(f"[yellow]🔧 Step 2: Validating input data...[/yellow]")
            if market_data is None:
                error_msg = 'Market data is None'
                self.logger.error(f"❌ {error_msg}")
                tprint(f"[bold red]❌ CRITICAL: Market data is None![/bold red]")
                return {'error': error_msg, 'success': False, 'timeframe': timeframe, 'system': 'TAS'}
            
            tprint(f"[green]✅ Market data is not None[/green]")
            tprint(f"[cyan]📊 Market data type: {type(market_data).__name__}[/cyan]")
            
            if hasattr(market_data, 'shape'):
                tprint(f"[cyan]📊 Market data shape: {market_data.shape}[/cyan]")
                if market_data.shape[0] == 0:
                    error_msg = f'Empty market data - shape: {market_data.shape}'
                    self.logger.error(f"❌ {error_msg}")
                    tprint(f"[bold red]❌ CRITICAL: Empty market data![/bold red]")
                    return {'error': error_msg, 'success': False, 'timeframe': timeframe, 'system': 'TAS'}
                tprint(f"[green]✅ Market data has {market_data.shape[0]} rows[/green]")
            else:
                tprint(f"[yellow]⚠️ Market data has no shape attribute[/yellow]")

            self.logger.info(f"📊 TAS input data shape: {market_data.shape if hasattr(market_data, 'shape') else 'Unknown'}")
            
            # Run TAS detection
            tprint(f"[yellow]🔧 Step 3: Calling TAS detect_regimes method...[/yellow]")
            tprint(f"[cyan]🔍 Parameters: optimize_performance=True, enable_patchtst_enhancement=True[/cyan]")
            self.logger.info("🚀 Calling TAS detect_regimes method...")
            
            tprint(f"[bold blue]⏳ Executing TAS detection...[/bold blue]")
            result = self.tas_system.detect_regimes(
                market_data, timestamps, optimize_performance=True, enable_patchtst_enhancement=True
            )
            tprint(f"[bold green]✅ TAS detect_regimes returned![/bold green]")
            
            tprint(f"[yellow]🔧 Step 4: Analyzing TAS result...[/yellow]")
            self.logger.info(f"📋 TAS result type: {type(result)}")
            tprint(f"[cyan]📋 TAS result type: {type(result)}[/cyan]")
            
            tprint(f"[cyan]🔍 Checking result for 'success' attribute...[/cyan]")
            success_attr = getattr(result, 'success', 'No success attribute')
            self.logger.info(f"📋 TAS result success: {success_attr}")
            tprint(f"[cyan]📋 TAS result success: {success_attr}[/cyan]")
            
            if result is None:
                error_msg = 'TAS detect_regimes returned None'
                self.logger.error(f"❌ {error_msg}")
                tprint(f"[bold red]❌ CRITICAL: TAS detect_regimes returned None![/bold red]")
                return {'error': error_msg, 'success': False, 'timeframe': timeframe, 'system': 'TAS'}

            tprint(f"[green]✅ Result is not None[/green]")
            
            # Extract results with error handling
            tprint(f"[yellow]🔧 Step 5: Extracting result attributes...[/yellow]")
            success = getattr(result, 'success', False)
            tprint(f"[cyan]📊 success = {success}[/cyan]")
            
            regime_predictions = getattr(result, 'regime_predictions', np.array([]))
            tprint(f"[cyan]📊 regime_predictions length = {len(regime_predictions) if hasattr(regime_predictions, '__len__') else 'N/A'}[/cyan]")
            
            regime_probabilities = getattr(result, 'regime_probabilities', np.array([]))
            tprint(f"[cyan]📊 regime_probabilities length = {len(regime_probabilities) if hasattr(regime_probabilities, '__len__') else 'N/A'}[/cyan]")
            
            execution_time = getattr(result, 'execution_time', 0.0)
            tprint(f"[cyan]📊 execution_time = {execution_time:.2f}s[/cyan]")
            
            # Extract detailed error information
            tprint(f"[yellow]🔧 Step 6: Checking for error information...[/yellow]")
            error_message = getattr(result, 'error_message', None)
            error_details = getattr(result, 'error_details', None)
            error_type = getattr(result, 'error_type', None)
            
            if error_message:
                tprint(f"[red]❌ error_message: {error_message}[/red]")
            if error_details:
                tprint(f"[red]❌ error_details: {error_details}[/red]")
            if error_type:
                tprint(f"[red]❌ error_type: {error_type}[/red]")
            
            self.logger.info(f"📊 TAS detection results - Success: {success}, Predictions: {len(regime_predictions)}, Time: {execution_time:.2f}s")
            
            if not success:
                tprint(f"[bold red]❌ TAS DETECTION FAILED![/bold red]")
                self.logger.error(f"❌ TAS detection failed - Error message: {error_message}")
                self.logger.error(f"❌ TAS detection failed - Error details: {error_details}")
                self.logger.error(f"❌ TAS detection failed - Error type: {error_type}")
                
                # Log all attributes of the result object for debugging
                tprint(f"[yellow]🔧 Dumping all TAS result attributes for debugging...[/yellow]")
                self.logger.info(f"🔧 TAS result attributes: {dir(result)}")
                tprint(f"[cyan]🔧 TAS result attributes: {[attr for attr in dir(result) if not attr.startswith('_')]}[/cyan]")
                
                for attr in dir(result):
                    if not attr.startswith('_'):
                        try:
                            value = getattr(result, attr)
                            self.logger.info(f"🔧 TAS result.{attr}: {value}")
                            tprint(f"[cyan]🔧 TAS result.{attr} = {value}[/cyan]")
                        except Exception as e:
                            self.logger.info(f"🔧 TAS result.{attr}: <error accessing: {e}>")
                            tprint(f"[yellow]🔧 TAS result.{attr} = <error accessing: {e}>[/yellow]")
            else:
                tprint(f"[bold green]✅ TAS DETECTION SUCCEEDED![/bold green]")

            tprint(f"[bold cyan]🔍 === TAS DETECTION END ===[/bold cyan]")
            return {
                'success': success,
                'regime_predictions': regime_predictions,
                'regime_assignments': regime_predictions,  # Standardize key for compatibility
                'regime_probabilities': regime_probabilities,
                'execution_time': execution_time,
                'timeframe': timeframe,
                'system': 'TAS',
                'result': result,  # Include full result for debugging
                'error_message': error_message,
                'error_details': error_details,
                'error_type': error_type
            }

        except Exception as e:
            error_msg = f"TAS regime detection failed: {str(e)}"
            tprint(f"[bold red]💥 EXCEPTION IN TAS DETECTION![/bold red]")
            tprint(f"[red]❌ Exception type: {type(e).__name__}[/red]")
            tprint(f"[red]❌ Exception message: {str(e)}[/red]")
            tprint(f"[red]❌ Exception details: {repr(e)}[/red]")
            
            self.logger.error(f"❌ {error_msg}")
            self.logger.error(f"❌ TAS detection exception type: {type(e).__name__}")
            self.logger.error(f"❌ TAS detection exception details: {repr(e)}")
            
            # Print traceback
            import traceback
            traceback_str = traceback.format_exc()
            tprint(f"[red]📋 Full traceback:\n{traceback_str}[/red]")
            self.logger.error(f"❌ TAS detection traceback:\n{traceback_str}")
            
            tprint(f"[bold cyan]🔍 === TAS DETECTION END (EXCEPTION) ===[/bold cyan]")
            
            # Return error instead of raising exception
            return {
                'error': error_msg,
                'success': False,
                'timeframe': timeframe,
                'system': 'TAS',
                'exception_type': type(e).__name__,
                'exception_details': repr(e),
                'traceback': traceback_str
            }

    def _run_nas_detection_sync(self, market_data: Union[pd.DataFrame, np.ndarray],
                                timestamps: Optional[np.ndarray], timeframe: str) -> Dict[str, Any]:
        """Run NAS regime detection."""
        try:
            # Use enhanced NAS regime detection
            from src.training.steps.market_analysis.nas_regime.core.enhanced_perfect_nas_regime_detector import EnhancedPerfectNASRegimeDetector
            from src.training.steps.market_analysis.nas_regime.core.perfect_nas_config import PerfectNASConfig

            # Create NAS configuration
            nas_config = PerfectNASConfig(
                n_regimes=3,
                primary_timeframe=timeframe,
                system_name='Enhanced NAS Regime Detection'
            )

            # Initialize enhanced NAS detector with config
            nas_detector = EnhancedPerfectNASRegimeDetector(nas_config)

            result = nas_detector.detect_regimes(
                market_data, timestamps, optimize_architecture=True, enable_meta_learning=True
            )

            return {
                'success': result.success,
                'regime_predictions': result.regime_predictions,
                'regime_probabilities': result.regime_probabilities,
                'economic_significance_scores': result.economic_significance_scores,
                'trading_viability_scores': result.trading_viability_scores,
                'execution_time': result.execution_time,
                'timeframe': timeframe,
                'system': 'Enhanced_NAS'
            }

        except Exception as e:
            self.logger.error(f"❌ Enhanced NAS regime detection failed: {e}")
            raise ValueError(f"Enhanced NAS regime detection failed: {e}")

    def _perform_hybrid_analysis(self, market_data: Union[pd.DataFrame, np.ndarray],
                                timestamps: Optional[np.ndarray],
                                tas_result: Dict[str, Any],
                                nas_result: Dict[str, Any]) -> Dict[str, Any]:
        """Perform enhanced hybrid analysis combining TAS and NAS results."""
        try:
            self.logger.info("🔬 Starting enhanced hybrid analysis")
            
            # Extract predictions
            tas_predictions = tas_result.get('regime_predictions', np.array([]))
            nas_predictions = nas_result.get('regime_predictions', np.array([]))

            if len(tas_predictions) == 0 or len(nas_predictions) == 0:
                return {'error': 'Empty predictions from one or both systems', 'success': False}

            # Step 1: Regime Alignment
            self.logger.info("🔄 Step 1: Performing regime alignment")
            
            # Log regime distributions to identify imbalance source
            if len(nas_predictions) > 0:
                nas_unique, nas_counts = np.unique(nas_predictions, return_counts=True)
                nas_distribution = {int(regime): f"{(count/len(nas_predictions)*100):.1f}%" 
                                   for regime, count in zip(nas_unique, nas_counts)}
                self.logger.info(f"📊 NAS Regime Distribution: {nas_distribution}")
                tprint(f"[yellow]📊 NAS Regime Distribution: {nas_distribution}[/yellow]")
                
                # Alert on highly imbalanced regimes
                for regime, count in zip(nas_unique, nas_counts):
                    percentage = (count/len(nas_predictions)*100)
                    if percentage > 15.0:
                        tprint(f"[bold red]⚠️🚨 ALERT: NAS Regime {regime} is {percentage:.1f}% (>15% threshold)[/bold red]")
                    elif percentage < 3.0:
                        tprint(f"[bold yellow]⚠️ WARNING: NAS Regime {regime} is {percentage:.1f}% (<3% threshold)[/bold yellow]")
            
            if len(tas_predictions) > 0:
                tas_unique, tas_counts = np.unique(tas_predictions, return_counts=True)
                tas_distribution = {int(regime): f"{(count/len(tas_predictions)*100):.1f}%" 
                                   for regime, count in zip(tas_unique, tas_counts)}
                self.logger.info(f"📊 TAS Regime Distribution: {tas_distribution}")
                tprint(f"[yellow]📊 TAS Regime Distribution: {tas_distribution}[/yellow]")
                
                # Alert on highly imbalanced regimes
                for regime, count in zip(tas_unique, tas_counts):
                    percentage = (count/len(tas_predictions)*100)
                    if percentage > 15.0:
                        tprint(f"[bold red]⚠️🚨 ALERT: TAS Regime {regime} is {percentage:.1f}% (>15% threshold)[/bold red]")
                    elif percentage < 3.0:
                        tprint(f"[bold yellow]⚠️ WARNING: TAS Regime {regime} is {percentage:.1f}% (<3% threshold)[/bold yellow]")
            
            alignment_result = self.regime_aligner.align_regimes(
                nas_predictions, tas_predictions, market_data
            )
            
            # Step 2: Multi-Objective Optimization
            self.logger.info("🎯 Step 2: Performing multi-objective optimization")
            optimization_result = self.multi_objective_optimizer.optimize_regime_clustering(
                nas_predictions, tas_predictions, market_data
            )
            
            # Step 3: Enhanced Economic Evaluation
            self.logger.info("💰 Step 3: Performing enhanced economic evaluation")
            best_solution = optimization_result.get('best_solution', {})
            consensus_predictions = best_solution.get('solution', np.array([]))
            
            if len(consensus_predictions) > 0:
                economic_evaluation = self.enhanced_economic_evaluator.evaluate_regime_clustering(
                    consensus_predictions, market_data
                )
            else:
                economic_evaluation = {'error': 'No consensus predictions available'}
            
            # Step 4: Consensus Validation
            self.logger.info("🔍 Step 4: Performing consensus validation")
            validation_result = self.consensus_validator.validate_consensus(
                consensus_predictions, nas_result, tas_result, market_data
            )
            
            # Step 5: Generate final results
            self.logger.info("📊 Step 5: Generating final hybrid results")
            hybrid_results = {
                'hybrid_labels': consensus_predictions,
                'hybrid_centers': self._calculate_hybrid_centers(consensus_predictions, market_data),
                'clustering_metrics': best_solution.get('objectives', {}),
                'clustering_quality': validation_result.get('statistical_validation', {}),
                'alignment_result': alignment_result,
                'optimization_result': optimization_result,
                'economic_evaluation': economic_evaluation,
                'validation_result': validation_result,
                'tas_contribution': tas_result,
                'nas_contribution': nas_result,
                'success': True,
                'enhanced_analysis': True
            }
            
            # Add clustering quality metrics
            if self.clustering_quality_analyzer and len(consensus_predictions) > 0:
                try:
                    # Prepare features for quality analysis
                    if isinstance(market_data, pd.DataFrame):
                        numeric_columns = market_data.select_dtypes(include=[np.number]).columns
                        if len(numeric_columns) > 0:
                            features = market_data[numeric_columns].values
                        else:
                            basic_columns = ['open', 'high', 'low', 'close', 'volume']
                            available_columns = [col for col in basic_columns if col in market_data.columns]
                            features = market_data[available_columns].values if available_columns else market_data.values
                    else:
                        features = market_data

                    # Ensure same length
                    min_length = min(len(features), len(consensus_predictions))
                    features = features[:min_length]
                    consensus_predictions = consensus_predictions[:min_length]

                    # Calculate comprehensive quality metrics
                    comprehensive_quality = self.clustering_quality_analyzer.calculate_comprehensive_metrics(
                        features, consensus_predictions
                    )
                    
                    hybrid_results['comprehensive_quality'] = comprehensive_quality
                    
                except Exception as e:
                    self.logger.warning(f"⚠️ Comprehensive quality analysis failed: {e}")
                    hybrid_results['comprehensive_quality'] = {'error': str(e)}
            
            self.logger.info("✅ Enhanced hybrid analysis completed successfully")
            return hybrid_results

        except Exception as e:
            self.logger.error(f"❌ Enhanced hybrid analysis failed: {e}")
            return {'error': str(e), 'success': False}
    
    def _calculate_hybrid_centers(self, consensus_predictions: np.ndarray, 
                                market_data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Calculate hybrid regime centers."""
        try:
            unique_regimes = np.unique(consensus_predictions)
            centers = []
            
            for regime in unique_regimes:
                regime_mask = consensus_predictions == regime
                regime_data = market_data[regime_mask]
                
                if len(regime_data) > 0:
                    if isinstance(regime_data, pd.DataFrame):
                        # Calculate centroid from numeric columns
                        numeric_columns = regime_data.select_dtypes(include=[np.number]).columns
                        if len(numeric_columns) > 0:
                            centroid = regime_data[numeric_columns].mean().values
                        else:
                            centroid = np.zeros(5)  # Default for OHLCV
                    else:
                        centroid = np.mean(regime_data, axis=0)
                    
                    centers.append(centroid)
                else:
                    centers.append(np.zeros(5))  # Default center
            
            return np.array(centers) if centers else np.array([])
            
        except Exception as e:
            self.logger.error(f"❌ Hybrid centers calculation failed: {e}")
            return np.array([])
    
    def _perform_semantic_divergence_assessment(
        self, 
        tas_assignments: np.ndarray, 
        nas_assignments: np.ndarray, 
        min_length: int
    ) -> Dict[str, Any]:
        """
        Perform semantic divergence assessment with regime mapping for consensus validation.
        
        This method creates a semantic mapping between TAS and NAS regimes based on their
        characteristics, enabling more accurate consensus measurement.
        
        Args:
            tas_assignments: TAS regime assignments
            nas_assignments: NAS regime assignments  
            min_length: Minimum length for comparison
            
        Returns:
            Dictionary containing semantic divergence assessment results
        """
        self.logger.info("🧠 Starting semantic divergence assessment with regime mapping")
        try:
            if len(tas_assignments) == 0 or len(nas_assignments) == 0:
                self.logger.warning("⚠️ Missing assignments for semantic assessment")
                return {
                    'semantic_divergence_rate': 1.0,
                    'regime_mapping': {},
                    'assessment_method': 'failed_missing_data'
                }
            
            # Ensure both assignments have the same length
            tas_assignments = np.array(tas_assignments[:min_length])
            nas_assignments = np.array(nas_assignments[:min_length])
            
            self.logger.info(f"📊 Analyzing {min_length} samples: TAS={len(set(tas_assignments))} regimes, NAS={len(set(nas_assignments))} regimes")
            
            # For hybrid orchestrator, we'll use a distribution-based semantic approach
            # since we don't have direct access to market data features
            
            # Step 1: Calculate regime distributions
            tas_distribution = self._calculate_regime_distribution(tas_assignments)
            nas_distribution = self._calculate_regime_distribution(nas_assignments)
            
            # Step 2: Find optimal regime mapping using distribution similarity
            regime_mapping = self._find_optimal_regime_mapping_by_distribution(tas_distribution, nas_distribution)
            
            if not regime_mapping:
                self.logger.warning("⚠️ No regime mapping found, using numerical comparison")
                return self._assess_numerical_divergence_fallback(tas_assignments, nas_assignments)
            
            # Step 3: Calculate semantic divergence using mapped regimes
            self.logger.info("🧮 Calculating semantic divergence using mapped regimes")
            semantic_assignments = self._apply_regime_mapping(nas_assignments, regime_mapping)
            semantic_disagreement_mask = tas_assignments != semantic_assignments
            semantic_divergence_rate = np.mean(semantic_disagreement_mask)
            
            # Step 4: Calculate mapping quality metrics
            self.logger.info("📊 Calculating mapping quality metrics")
            mapping_quality = self._calculate_mapping_quality_by_distribution(tas_distribution, nas_distribution, regime_mapping)
            
            # Step 5: Enhanced semantic consensus calculation with temporal analysis
            self.logger.info("🔄 Performing enhanced semantic consensus calculation")
            enhanced_consensus = self._calculate_enhanced_semantic_consensus(
                tas_assignments, semantic_assignments, min_length, regime_mapping
            )
            
            # Step 6: Report results
            self.logger.info(f"✅ Semantic divergence assessment completed:")
            self.logger.info(f"   📊 Semantic divergence rate: {semantic_divergence_rate:.3f}")
            self.logger.info(f"   🎯 Regime mappings: {len(regime_mapping)}")
            self.logger.info(f"   📈 Mapping quality: {mapping_quality:.3f}")
            
            # Calculate semantic consensus improvement
            raw_agreements = np.sum(tas_assignments == nas_assignments)
            raw_consensus = raw_agreements / min_length if min_length > 0 else 0.0
            semantic_agreements = np.sum(tas_assignments == semantic_assignments)
            semantic_consensus = semantic_agreements / min_length if min_length > 0 else 0.0
            consensus_improvement = semantic_consensus - raw_consensus
            
            # Use enhanced consensus if available
            if enhanced_consensus > semantic_consensus:
                semantic_consensus = enhanced_consensus
                semantic_agreements = int(enhanced_consensus * min_length)
                consensus_improvement = semantic_consensus - raw_consensus
                self.logger.info(f"   🚀 Enhanced semantic consensus applied: {semantic_consensus:.3f}")
            
            self.logger.info(f"   🤝 Raw consensus: {raw_consensus:.3f} ({raw_agreements}/{min_length})")
            self.logger.info(f"   🧠 Semantic consensus: {semantic_consensus:.3f} ({semantic_agreements}/{min_length})")
            self.logger.info(f"   📈 Consensus improvement: {consensus_improvement:.3f}")
            
            return {
                'semantic_divergence_rate': semantic_divergence_rate,
                'regime_mapping': regime_mapping,
                'mapping_quality': mapping_quality,
                'raw_consensus': raw_consensus,
                'semantic_consensus': semantic_consensus,
                'consensus_improvement': consensus_improvement,
                'assessment_method': 'distribution_based',
                'tas_distribution': tas_distribution,
                'nas_distribution': nas_distribution
            }
            
        except Exception as e:
            self.logger.error(f"❌ Semantic divergence assessment failed: {e}")
            return self._assess_numerical_divergence_fallback(tas_assignments, nas_assignments)
    
    def _calculate_regime_distribution(self, assignments: np.ndarray) -> Dict[str, float]:
        """Calculate the distribution of regime assignments."""
        try:
            if len(assignments) == 0:
                return {}
            
            total_assignments = len(assignments)
            regime_counts = {}
            
            for assignment in assignments:
                regime_counts[assignment] = regime_counts.get(assignment, 0) + 1
            
            # Convert to percentages
            regime_distribution = {}
            for regime, count in regime_counts.items():
                key = f'regime_{regime}'
                percentage = (count / total_assignments) * 100
                regime_distribution[key] = percentage
            
            return regime_distribution
            
        except Exception as e:
            self.logger.warning(f"⚠️ Distribution calculation failed: {e}")
            return {}
    
    def _find_optimal_regime_mapping_by_distribution(self, tas_distribution: Dict[str, float], nas_distribution: Dict[str, float]) -> Dict[int, int]:
        """Find optimal mapping between NAS and TAS regimes using enhanced distribution similarity."""
        try:
            if not tas_distribution or not nas_distribution:
                return {}
            
            # Extract regime IDs and their percentages
            tas_regimes = {}
            nas_regimes = {}
            
            for key, percentage in tas_distribution.items():
                regime_id = int(key.replace('regime_', ''))
                tas_regimes[regime_id] = percentage
            
            for key, percentage in nas_distribution.items():
                regime_id = int(key.replace('regime_', ''))
                nas_regimes[regime_id] = percentage
            
            # Enhanced mapping using Hungarian algorithm for optimal assignment
            regime_mapping = self._find_optimal_mapping_hungarian(tas_regimes, nas_regimes)
            
            # If Hungarian algorithm fails, fall back to improved greedy approach
            if not regime_mapping:
                regime_mapping = self._find_optimal_mapping_greedy_enhanced(tas_regimes, nas_regimes)
            
            # If still no mapping, try alternative approaches
            if not regime_mapping:
                regime_mapping = self._find_optimal_mapping_alternative(tas_regimes, nas_regimes)
            
            return regime_mapping
            
        except Exception as e:
            self.logger.warning(f"⚠️ Distribution-based mapping failed: {e}")
            return {}
    
    def _find_optimal_mapping_hungarian(self, tas_regimes: Dict[int, float], nas_regimes: Dict[int, float]) -> Dict[int, int]:
        """Find optimal mapping using enhanced Hungarian algorithm with multi-criteria similarity."""
        try:
            from scipy.optimize import linear_sum_assignment
            
            # Create cost matrix (negative similarity for maximization)
            tas_list = list(tas_regimes.keys())
            nas_list = list(nas_regimes.keys())
            
            if not tas_list or not nas_list:
                return {}
            
            # Create enhanced similarity matrix with multiple criteria
            similarity_matrix = np.zeros((len(nas_list), len(tas_list)))
            
            for i, nas_regime in enumerate(nas_list):
                for j, tas_regime in enumerate(tas_list):
                    nas_pct = nas_regimes[nas_regime]
                    tas_pct = tas_regimes[tas_regime]
                    
                    # Multi-criteria similarity calculation
                    similarity_score = self._calculate_enhanced_regime_similarity(
                        nas_regime, tas_regime, nas_pct, tas_pct
                    )
                    similarity_matrix[i, j] = similarity_score
            
            # Use Hungarian algorithm to find optimal assignment
            row_indices, col_indices = linear_sum_assignment(-similarity_matrix)
            
            # Create mapping with quality validation
            regime_mapping = {}
            total_similarity = 0.0
            valid_mappings = 0
            
            for i, j in zip(row_indices, col_indices):
                if i < len(nas_list) and j < len(tas_list):
                    nas_regime = nas_list[i]
                    tas_regime = tas_list[j]
                    similarity = similarity_matrix[i, j]
                    
                    # Only include mappings above quality threshold
                    if similarity > 0.3:  # Minimum similarity threshold
                        regime_mapping[nas_regime] = tas_regime
                        total_similarity += similarity
                        valid_mappings += 1
            
            avg_similarity = total_similarity / max(valid_mappings, 1)
            self.logger.info(f"🎯 Enhanced Hungarian algorithm found {len(regime_mapping)} optimal mappings (avg similarity: {avg_similarity:.3f})")
            return regime_mapping
            
        except ImportError:
            self.logger.warning("⚠️ scipy not available for Hungarian algorithm, using greedy approach")
            return {}
        except Exception as e:
            self.logger.warning(f"⚠️ Enhanced Hungarian algorithm failed: {e}")
            return {}
    
    def _calculate_enhanced_regime_similarity(
        self, 
        nas_regime: int, 
        tas_regime: int, 
        nas_pct: float, 
        tas_pct: float
    ) -> float:
        """Calculate enhanced regime similarity using multiple criteria."""
        try:
            # 1. Distribution similarity (base similarity)
            distribution_sim = 1.0 - abs(nas_pct - tas_pct) / 100.0
            
            # 2. Size similarity bonus (regimes of similar size are more likely to match)
            size_bonus = 0.15 * (1.0 - abs(nas_pct - tas_pct) / 200.0)
            
            # 3. Frequency similarity (prefer regimes with similar frequency)
            freq_sim = 0.1 * (1.0 - abs(nas_pct - tas_pct) / 200.0)
            
            # 4. Regime ID similarity (regimes with similar IDs might be related)
            id_sim = 0.05 * (1.0 - abs(nas_regime - tas_regime) / 10.0)
            
            # 5. Market context similarity (if available)
            context_sim = self._calculate_market_context_similarity(nas_regime, tas_regime)
            
            # 6. Temporal pattern similarity (if available)
            temporal_sim = self._calculate_temporal_pattern_similarity(nas_regime, tas_regime)
            
            # Combine all similarity components with weights
            total_similarity = (
                distribution_sim * 0.4 +      # 40% weight for distribution
                size_bonus * 0.2 +            # 20% weight for size similarity
                freq_sim * 0.15 +             # 15% weight for frequency
                id_sim * 0.05 +               # 5% weight for ID similarity
                context_sim * 0.15 +          # 15% weight for market context
                temporal_sim * 0.05           # 5% weight for temporal patterns
            )
            
            return max(0.0, min(1.0, total_similarity))
            
        except Exception as e:
            self.logger.warning(f"⚠️ Enhanced similarity calculation failed: {e}")
            # Fallback to basic distribution similarity
            return 1.0 - abs(nas_pct - tas_pct) / 100.0
    
    def _calculate_market_context_similarity(self, nas_regime: int, tas_regime: int) -> float:
        """Calculate enhanced market context similarity between regimes."""
        try:
            # 1. Regime ID proximity (regimes closer in ID space might be related)
            id_distance = abs(nas_regime - tas_regime)
            id_sim = max(0.0, 1.0 - id_distance / 10.0)
            
            # 2. Regime size similarity (if available from market data)
            size_sim = self._calculate_regime_size_similarity(nas_regime, tas_regime)
            
            # 3. Volatility regime similarity (if available)
            volatility_sim = self._calculate_volatility_regime_similarity(nas_regime, tas_regime)
            
            # 4. Trend regime similarity (if available)
            trend_sim = self._calculate_trend_regime_similarity(nas_regime, tas_regime)
            
            # 5. Economic indicator similarity (if available)
            economic_sim = self._calculate_economic_indicator_similarity(nas_regime, tas_regime)
            
            # Combine all market context factors
            context_sim = (
                id_sim * 0.3 +           # 30% weight for ID similarity
                size_sim * 0.25 +         # 25% weight for size similarity
                volatility_sim * 0.2 +    # 20% weight for volatility similarity
                trend_sim * 0.15 +        # 15% weight for trend similarity
                economic_sim * 0.1        # 10% weight for economic similarity
            )
            
            return max(0.0, min(1.0, context_sim))
            
        except Exception:
            return 0.0
    
    def _calculate_regime_size_similarity(self, nas_regime: int, tas_regime: int) -> float:
        """Calculate similarity based on regime size characteristics."""
        try:
            # This would ideally use actual regime size data
            # For now, use regime ID as a proxy for size characteristics
            nas_size_proxy = nas_regime % 5  # 0-4 scale
            tas_size_proxy = tas_regime % 5  # 0-4 scale
            
            size_distance = abs(nas_size_proxy - tas_size_proxy)
            size_sim = max(0.0, 1.0 - size_distance / 4.0)
            
            return size_sim
            
        except Exception:
            return 0.0
    
    def _calculate_volatility_regime_similarity(self, nas_regime: int, tas_regime: int) -> float:
        """Calculate similarity based on volatility regime characteristics."""
        try:
            # This would ideally use actual volatility data
            # For now, use regime ID as a proxy for volatility characteristics
            nas_vol_proxy = (nas_regime % 3) + 1  # 1-3 scale (low, medium, high)
            tas_vol_proxy = (tas_regime % 3) + 1  # 1-3 scale (low, medium, high)
            
            vol_distance = abs(nas_vol_proxy - tas_vol_proxy)
            vol_sim = max(0.0, 1.0 - vol_distance / 2.0)
            
            return vol_sim
            
        except Exception:
            return 0.0
    
    def _calculate_trend_regime_similarity(self, nas_regime: int, tas_regime: int) -> float:
        """Calculate similarity based on trend regime characteristics."""
        try:
            # This would ideally use actual trend data
            # For now, use regime ID as a proxy for trend characteristics
            nas_trend_proxy = (nas_regime % 4) + 1  # 1-4 scale (strong down, weak down, weak up, strong up)
            tas_trend_proxy = (tas_regime % 4) + 1  # 1-4 scale (strong down, weak down, weak up, strong up)
            
            trend_distance = abs(nas_trend_proxy - tas_trend_proxy)
            trend_sim = max(0.0, 1.0 - trend_distance / 3.0)
            
            return trend_sim
            
        except Exception:
            return 0.0
    
    def _calculate_economic_indicator_similarity(self, nas_regime: int, tas_regime: int) -> float:
        """Calculate similarity based on economic indicator characteristics."""
        try:
            # This would ideally use actual economic indicator data
            # For now, use regime ID as a proxy for economic characteristics
            nas_econ_proxy = (nas_regime % 2) + 1  # 1-2 scale (recession, expansion)
            tas_econ_proxy = (tas_regime % 2) + 1  # 1-2 scale (recession, expansion)
            
            econ_distance = abs(nas_econ_proxy - tas_econ_proxy)
            econ_sim = max(0.0, 1.0 - econ_distance / 1.0)
            
            return econ_sim
            
        except Exception:
            return 0.0
    
    def _calculate_temporal_pattern_similarity(self, nas_regime: int, tas_regime: int) -> float:
        """Calculate temporal pattern similarity between regimes."""
        try:
            # This would ideally analyze temporal patterns in regime assignments
            # For now, use a simple heuristic based on regime characteristics
            
            # Regimes with similar characteristics might have similar temporal patterns
            id_distance = abs(nas_regime - tas_regime)
            temporal_sim = max(0.0, 1.0 - id_distance / 15.0)
            
            return temporal_sim
            
        except Exception:
            return 0.0
    
    def _find_optimal_mapping_greedy_enhanced(self, tas_regimes: Dict[int, float], nas_regimes: Dict[int, float]) -> Dict[int, int]:
        """Enhanced greedy mapping with multiple criteria."""
        try:
            regime_mapping = {}
            used_tas_regimes = set()
            
            # Create all possible pairs with similarity scores
            pairs = []
            for nas_regime, nas_pct in nas_regimes.items():
                for tas_regime, tas_pct in tas_regimes.items():
                    if tas_regime not in used_tas_regimes:
                        # Calculate multi-criteria similarity
                        distribution_sim = 1.0 - abs(nas_pct - tas_pct) / 100.0
                        
                        # Size similarity bonus (regimes of similar size are more likely to match)
                        size_bonus = 0.2 * (1.0 - abs(nas_pct - tas_pct) / 200.0)
                        
                        # Frequency similarity (prefer regimes with similar frequency)
                        freq_sim = 1.0 - abs(nas_pct - tas_pct) / 200.0
                        
                        # Combined similarity score
                        total_similarity = distribution_sim + size_bonus + freq_sim
                        
                        pairs.append((nas_regime, tas_regime, total_similarity))
            
            # Sort by similarity (highest first)
            pairs.sort(key=lambda x: x[2], reverse=True)
            
            # Greedily assign best matches
            for nas_regime, tas_regime, similarity in pairs:
                if nas_regime not in regime_mapping and tas_regime not in used_tas_regimes:
                    if similarity > 0.3:  # Only map if similarity is reasonable
                        regime_mapping[nas_regime] = tas_regime
                        used_tas_regimes.add(tas_regime)
            
            self.logger.info(f"🎯 Enhanced greedy algorithm found {len(regime_mapping)} mappings")
            return regime_mapping
            
        except Exception as e:
            self.logger.warning(f"⚠️ Enhanced greedy mapping failed: {e}")
            return {}
    
    def _apply_regime_mapping(self, nas_assignments: np.ndarray, regime_mapping: Dict[int, int]) -> np.ndarray:
        """Apply regime mapping to NAS assignments."""
        try:
            mapped_assignments = nas_assignments.copy()
            
            for nas_regime, tas_regime in regime_mapping.items():
                mask = nas_assignments == nas_regime
                mapped_assignments[mask] = tas_regime
            
            return mapped_assignments
            
        except Exception as e:
            self.logger.warning(f"⚠️ Regime mapping application failed: {e}")
            return nas_assignments
    
    def _calculate_mapping_quality_by_distribution(self, tas_distribution: Dict[str, float], nas_distribution: Dict[str, float], regime_mapping: Dict[int, int]) -> float:
        """Calculate enhanced quality metrics for the regime mapping based on multiple criteria."""
        try:
            if not regime_mapping:
                return 0.0
            
            total_similarity = 0.0
            mapping_count = 0
            quality_metrics = []
            
            for nas_regime, tas_regime in regime_mapping.items():
                nas_key = f'regime_{nas_regime}'
                tas_key = f'regime_{tas_regime}'
                
                if nas_key in nas_distribution and tas_key in tas_distribution:
                    nas_percentage = nas_distribution[nas_key]
                    tas_percentage = tas_distribution[tas_key]
                    
                    # Enhanced similarity calculation with multiple criteria
                    distribution_sim = 1.0 - abs(nas_percentage - tas_percentage) / 100.0
                    
                    # Size consistency bonus (regimes of similar size are more likely to be semantically similar)
                    size_consistency = 1.0 - abs(nas_percentage - tas_percentage) / 200.0
                    
                    # Frequency alignment (how well the frequencies align)
                    freq_alignment = 1.0 - abs(nas_percentage - tas_percentage) / 150.0
                    
                    # Combined similarity with weighted criteria
                    similarity = (distribution_sim * 0.5 + size_consistency * 0.3 + freq_alignment * 0.2)
                    
                    total_similarity += similarity
                    mapping_count += 1
                    quality_metrics.append(similarity)
            
            if mapping_count == 0:
                return 0.0
            
            # Calculate enhanced quality metrics
            avg_similarity = total_similarity / mapping_count
            
            # Consistency bonus (how consistent the mapping quality is)
            if len(quality_metrics) > 1:
                consistency = 1.0 - np.std(quality_metrics) / np.mean(quality_metrics) if np.mean(quality_metrics) > 0 else 0.0
                consistency_bonus = consistency * 0.1  # 10% bonus for consistency
            else:
                consistency_bonus = 0.0
            
            # Coverage bonus (how many regimes are mapped)
            coverage = len(regime_mapping) / max(len(tas_distribution), len(nas_distribution)) if max(len(tas_distribution), len(nas_distribution)) > 0 else 0.0
            coverage_bonus = coverage * 0.1  # 10% bonus for good coverage
            
            # Final quality score
            quality = avg_similarity + consistency_bonus + coverage_bonus
            quality = max(0.0, min(1.0, quality))  # Clamp between 0 and 1
            
            self.logger.info(f"📊 Mapping quality: {quality:.3f} (avg_sim: {avg_similarity:.3f}, consistency: {consistency_bonus:.3f}, coverage: {coverage_bonus:.3f})")
            
            return quality
            
        except Exception as e:
            self.logger.warning(f"⚠️ Enhanced mapping quality calculation failed: {e}")
            return 0.0
    
    def _calculate_enhanced_semantic_consensus(self, tas_assignments: np.ndarray, semantic_assignments: np.ndarray, min_length: int, regime_mapping: Dict[int, int]) -> float:
        """Calculate enhanced semantic consensus using temporal pattern analysis and regime transition analysis."""
        try:
            if min_length == 0 or len(regime_mapping) == 0:
                return 0.0
            
            # Basic semantic consensus
            basic_agreements = np.sum(tas_assignments == semantic_assignments)
            basic_consensus = basic_agreements / min_length
            
            # Temporal pattern analysis
            temporal_bonus = self._analyze_temporal_patterns(tas_assignments, semantic_assignments)
            
            # Regime transition analysis
            transition_bonus = self._analyze_regime_transitions(tas_assignments, semantic_assignments, regime_mapping)
            
            # Stability analysis (how stable the consensus is over time)
            stability_bonus = self._analyze_consensus_stability(tas_assignments, semantic_assignments)
            
            # Clustering quality analysis
            clustering_bonus = self._analyze_clustering_quality(tas_assignments, semantic_assignments, regime_mapping)
            
            # Calculate dynamic consensus weighting
            dynamic_weights = self._calculate_dynamic_consensus_weights(
                tas_assignments, semantic_assignments, regime_mapping
            )
            
            # Apply dynamic weighting to bonuses
            weighted_temporal_bonus = temporal_bonus * dynamic_weights['temporal_weight']
            weighted_transition_bonus = transition_bonus * dynamic_weights['transition_weight']
            weighted_stability_bonus = stability_bonus * dynamic_weights['stability_weight']
            weighted_clustering_bonus = clustering_bonus * dynamic_weights['clustering_weight']
            
            # Combine all weighted bonuses
            total_bonus = (
                weighted_temporal_bonus + 
                weighted_transition_bonus + 
                weighted_stability_bonus + 
                weighted_clustering_bonus
            )
            
            # Apply adaptive consensus thresholds
            adaptive_threshold = self._calculate_adaptive_consensus_threshold(
                tas_assignments, semantic_assignments, regime_mapping
            )
            
            # Adjust enhanced consensus based on adaptive threshold
            if enhanced_consensus >= adaptive_threshold:
                # High consensus - apply confidence boost
                confidence_boost = min(0.05, (enhanced_consensus - adaptive_threshold) * 0.1)
                enhanced_consensus = min(1.0, enhanced_consensus + confidence_boost)
            else:
                # Low consensus - apply penalty
                consensus_penalty = min(0.1, (adaptive_threshold - enhanced_consensus) * 0.2)
                enhanced_consensus = max(0.0, enhanced_consensus - consensus_penalty)
            
            self.logger.info(f"🔍 Enhanced consensus analysis:")
            self.logger.info(f"   📊 Basic consensus: {basic_consensus:.3f}")
            self.logger.info(f"   ⏰ Temporal bonus: {temporal_bonus:.3f}")
            self.logger.info(f"   🔄 Transition bonus: {transition_bonus:.3f}")
            self.logger.info(f"   📈 Stability bonus: {stability_bonus:.3f}")
            self.logger.info(f"   🎯 Clustering bonus: {clustering_bonus:.3f}")
            self.logger.info(f"   🚀 Total bonus: {total_bonus:.3f}")
            self.logger.info(f"   🎯 Adaptive threshold: {adaptive_threshold:.3f}")
            self.logger.info(f"   🎉 Enhanced consensus: {enhanced_consensus:.3f}")
            
            return enhanced_consensus
            
        except Exception as e:
            self.logger.warning(f"⚠️ Enhanced consensus calculation failed: {e}")
            return 0.0
    
    def _analyze_temporal_patterns(self, tas_assignments: np.ndarray, semantic_assignments: np.ndarray) -> float:
        """Analyze enhanced temporal patterns in regime assignments for consensus bonus."""
        try:
            if len(tas_assignments) < 10:
                return 0.0
            
            # 1. Regime persistence analysis
            tas_persistence = self._calculate_enhanced_regime_persistence(tas_assignments)
            semantic_persistence = self._calculate_enhanced_regime_persistence(semantic_assignments)
            persistence_sim = 1.0 - abs(tas_persistence - semantic_persistence) / max(tas_persistence, semantic_persistence, 1.0)
            
            # 2. Regime change frequency analysis
            tas_changes = np.sum(tas_assignments[1:] != tas_assignments[:-1])
            semantic_changes = np.sum(semantic_assignments[1:] != semantic_assignments[:-1])
            change_freq_sim = 1.0 - abs(tas_changes - semantic_changes) / max(tas_changes, semantic_changes, 1)
            
            # 3. Regime stability analysis (how stable regimes are over time)
            tas_stability = self._calculate_regime_stability(tas_assignments)
            semantic_stability = self._calculate_regime_stability(semantic_assignments)
            stability_sim = 1.0 - abs(tas_stability - semantic_stability) / max(tas_stability, semantic_stability, 1.0)
            
            # 4. Regime transition smoothness analysis
            tas_smoothness = self._calculate_transition_smoothness(tas_assignments)
            semantic_smoothness = self._calculate_transition_smoothness(semantic_assignments)
            smoothness_sim = 1.0 - abs(tas_smoothness - semantic_smoothness) / max(tas_smoothness, semantic_smoothness, 1.0)
            
            # 5. Temporal correlation analysis
            temporal_correlation = self._calculate_temporal_correlation(tas_assignments, semantic_assignments)
            
            # 6. Regime duration distribution analysis
            tas_duration_dist = self._calculate_duration_distribution(tas_assignments)
            semantic_duration_dist = self._calculate_duration_distribution(semantic_assignments)
            duration_sim = self._calculate_distribution_similarity(tas_duration_dist, semantic_duration_dist)
            
            # Combine all temporal metrics with weighted importance
            temporal_bonus = (
                persistence_sim * 0.25 +      # 25% weight for persistence
                change_freq_sim * 0.20 +      # 20% weight for change frequency
                stability_sim * 0.20 +        # 20% weight for stability
                smoothness_sim * 0.15 +       # 15% weight for smoothness
                temporal_correlation * 0.10 + # 10% weight for correlation
                duration_sim * 0.10              # 10% weight for duration distribution
            ) * 0.15  # Max 15% bonus (increased from 10%)
            
            return max(0.0, temporal_bonus)
            
        except Exception as e:
            self.logger.warning(f"⚠️ Enhanced temporal pattern analysis failed: {e}")
            return 0.0
    
    def _find_optimal_mapping_alternative(self, tas_regimes: Dict[int, float], nas_regimes: Dict[int, float]) -> Dict[int, int]:
        """Alternative mapping approach using size-based clustering and similarity scoring."""
        try:
            regime_mapping = {}
            used_tas_regimes = set()
            
            # Create size-based clusters
            tas_clusters = self._create_size_clusters(tas_regimes)
            nas_clusters = self._create_size_clusters(nas_regimes)
            
            # Map clusters to each other
            for nas_cluster in nas_clusters:
                best_tas_cluster = None
                best_similarity = 0.0
                
                for tas_cluster in tas_clusters:
                    similarity = self._calculate_cluster_similarity(nas_cluster, tas_cluster)
                    if similarity > best_similarity and similarity > 0.2:  # Minimum threshold
                        best_similarity = similarity
                        best_tas_cluster = tas_cluster
                
                if best_tas_cluster and best_tas_cluster not in used_tas_regimes:
                    # Map regimes within clusters
                    for nas_regime in nas_cluster:
                        for tas_regime in best_tas_cluster:
                            if tas_regime not in used_tas_regimes:
                                regime_mapping[nas_regime] = tas_regime
                                used_tas_regimes.add(tas_regime)
                                break
                    used_tas_regimes.update(best_tas_cluster)
            
            self.logger.info(f"🎯 Alternative mapping found {len(regime_mapping)} mappings")
            return regime_mapping
            
        except Exception as e:
            self.logger.warning(f"⚠️ Alternative mapping failed: {e}")
            return {}
    
    def _create_size_clusters(self, regimes: Dict[int, float]) -> List[List[int]]:
        """Create clusters of regimes based on size similarity."""
        try:
            if not regimes:
                return []
            
            # Sort regimes by size
            sorted_regimes = sorted(regimes.items(), key=lambda x: x[1], reverse=True)
            
            clusters = []
            current_cluster = []
            current_size = None
            
            for regime_id, size in sorted_regimes:
                if current_size is None or abs(size - current_size) / current_size < 0.3:  # 30% size difference threshold
                    current_cluster.append(regime_id)
                    current_size = size
                else:
                    if current_cluster:
                        clusters.append(current_cluster)
                    current_cluster = [regime_id]
                    current_size = size
            
            if current_cluster:
                clusters.append(current_cluster)
            
            return clusters
            
        except Exception as e:
            self.logger.warning(f"⚠️ Size clustering failed: {e}")
            return []
    
    def _calculate_cluster_similarity(self, nas_cluster: List[int], tas_cluster: List[int]) -> float:
        """Calculate similarity between two regime clusters."""
        try:
            if not nas_cluster or not tas_cluster:
                return 0.0
            
            # Calculate average size similarity
            nas_sizes = [self._get_regime_size(regime_id) for regime_id in nas_cluster]
            tas_sizes = [self._get_regime_size(regime_id) for regime_id in tas_cluster]
            
            nas_avg = np.mean(nas_sizes) if nas_sizes else 0.0
            tas_avg = np.mean(tas_sizes) if tas_sizes else 0.0
            
            if nas_avg == 0.0 or tas_avg == 0.0:
                return 0.0
            
            # Size similarity
            size_sim = 1.0 - abs(nas_avg - tas_avg) / max(nas_avg, tas_avg)
            
            # Cluster size similarity (number of regimes)
            cluster_size_sim = 1.0 - abs(len(nas_cluster) - len(tas_cluster)) / max(len(nas_cluster), len(tas_cluster))
            
            # Combined similarity
            similarity = (size_sim * 0.7 + cluster_size_sim * 0.3)
            
            return max(0.0, similarity)
            
        except Exception:
            return 0.0
    
    def _get_regime_size(self, regime_id: int) -> float:
        """Get regime size (placeholder - would need actual regime data)."""
        # This is a simplified version - in practice, you'd use actual regime characteristics
        return 1.0  # Placeholder
    
    def _calculate_regime_persistence(self, assignments: np.ndarray) -> float:
        """Calculate average regime persistence."""
        try:
            if len(assignments) < 2:
                return 1.0
            
            changes = np.sum(assignments[1:] != assignments[:-1])
            if changes == 0:
                return len(assignments)  # All same regime
            
            return len(assignments) / (changes + 1)  # Average persistence
            
        except Exception:
            return 1.0
    
    def _calculate_enhanced_regime_persistence(self, assignments: np.ndarray) -> float:
        """Calculate enhanced regime persistence with temporal analysis."""
        try:
            if len(assignments) < 2:
                return 1.0
            
            # Calculate regime durations
            durations = []
            current_regime = assignments[0]
            current_duration = 1
            
            for i in range(1, len(assignments)):
                if assignments[i] == current_regime:
                    current_duration += 1
                else:
                    durations.append(current_duration)
                    current_regime = assignments[i]
                    current_duration = 1
            
            # Add the last duration
            durations.append(current_duration)
            
            if not durations:
                return 1.0
            
            # Calculate weighted average persistence (longer regimes get more weight)
            total_weighted_duration = sum(d * d for d in durations)  # Square weighting
            total_weight = sum(d for d in durations)
            
            return total_weighted_duration / total_weight if total_weight > 0 else 1.0
            
        except Exception:
            return 1.0
    
    def _calculate_regime_stability(self, assignments: np.ndarray) -> float:
        """Calculate regime stability over time."""
        try:
            if len(assignments) < 3:
                return 1.0
            
            # Calculate regime frequency
            unique_regimes, counts = np.unique(assignments, return_counts=True)
            total_samples = len(assignments)
            
            # Calculate entropy-based stability
            probabilities = counts / total_samples
            entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
            max_entropy = np.log2(len(unique_regimes))
            
            # Stability is inverse of normalized entropy
            stability = 1.0 - (entropy / max_entropy) if max_entropy > 0 else 1.0
            
            return max(0.0, min(1.0, stability))
            
        except Exception:
            return 1.0
    
    def _calculate_transition_smoothness(self, assignments: np.ndarray) -> float:
        """Calculate how smooth regime transitions are."""
        try:
            if len(assignments) < 3:
                return 1.0
            
            # Calculate transition distances (how different consecutive regimes are)
            transition_distances = []
            for i in range(1, len(assignments)):
                distance = abs(assignments[i] - assignments[i-1])
                transition_distances.append(distance)
            
            if not transition_distances:
                return 1.0
            
            # Smoothness is inverse of average transition distance
            avg_distance = np.mean(transition_distances)
            max_possible_distance = max(assignments) - min(assignments) if len(assignments) > 0 else 1
            
            smoothness = 1.0 - (avg_distance / max_possible_distance) if max_possible_distance > 0 else 1.0
            
            return max(0.0, min(1.0, smoothness))
            
        except Exception:
            return 1.0
    
    def _calculate_temporal_correlation(self, tas_assignments: np.ndarray, semantic_assignments: np.ndarray) -> float:
        """Calculate temporal correlation between regime assignments."""
        try:
            if len(tas_assignments) < 3 or len(semantic_assignments) < 3:
                return 0.0
            
            # Calculate correlation coefficient
            correlation = np.corrcoef(tas_assignments, semantic_assignments)[0, 1]
            
            # Handle NaN case
            if np.isnan(correlation):
                return 0.0
            
            # Convert correlation to similarity (0 to 1 scale)
            similarity = (correlation + 1.0) / 2.0
            
            return max(0.0, min(1.0, similarity))
            
        except Exception:
            return 0.0
    
    def _calculate_duration_distribution(self, assignments: np.ndarray) -> Dict[int, float]:
        """Calculate distribution of regime durations."""
        try:
            if len(assignments) < 2:
                return {}
            
            # Calculate regime durations
            durations = []
            current_regime = assignments[0]
            current_duration = 1
            
            for i in range(1, len(assignments)):
                if assignments[i] == current_regime:
                    current_duration += 1
                else:
                    durations.append(current_duration)
                    current_regime = assignments[i]
                    current_duration = 1
            
            # Add the last duration
            durations.append(current_duration)
            
            # Create distribution
            duration_dist = {}
            for duration in durations:
                duration_dist[duration] = duration_dist.get(duration, 0) + 1
            
            # Normalize to probabilities
            total_durations = len(durations)
            for duration in duration_dist:
                duration_dist[duration] /= total_durations
            
            return duration_dist
            
        except Exception:
            return {}
    
    def _calculate_distribution_similarity(self, dist1: Dict[int, float], dist2: Dict[int, float]) -> float:
        """Calculate similarity between two distributions."""
        try:
            if not dist1 or not dist2:
                return 0.0
            
            # Get all unique keys
            all_keys = set(dist1.keys()) | set(dist2.keys())
            
            if not all_keys:
                return 0.0
            
            # Calculate Jensen-Shannon divergence
            js_divergence = 0.0
            for key in all_keys:
                p = dist1.get(key, 0.0)
                q = dist2.get(key, 0.0)
                m = (p + q) / 2.0
                
                if m > 0:
                    if p > 0:
                        js_divergence += p * np.log2(p / m)
                    if q > 0:
                        js_divergence += q * np.log2(q / m)
            
            js_divergence /= 2.0
            
            # Convert divergence to similarity
            similarity = 1.0 - js_divergence
            
            return max(0.0, min(1.0, similarity))
            
        except Exception:
            return 0.0
    
    def _calculate_dynamic_consensus_weights(
        self, 
        tas_assignments: np.ndarray, 
        semantic_assignments: np.ndarray, 
        regime_mapping: Dict[int, int]
    ) -> Dict[str, float]:
        """Calculate dynamic weights for consensus components based on regime confidence and market conditions."""
        try:
            # 1. Regime confidence analysis
            regime_confidence = self._calculate_regime_confidence(tas_assignments, semantic_assignments)
            
            # 2. Market volatility analysis
            market_volatility = self._calculate_market_volatility_proxy(tas_assignments, semantic_assignments)
            
            # 3. Historical performance analysis
            historical_performance = self._calculate_historical_performance_proxy(tas_assignments, semantic_assignments)
            
            # 4. Regime stability analysis
            regime_stability = self._calculate_regime_stability_score(tas_assignments, semantic_assignments)
            
            # Calculate dynamic weights based on these factors
            weights = {
                'temporal_weight': self._calculate_temporal_weight(regime_confidence, market_volatility),
                'transition_weight': self._calculate_transition_weight(regime_confidence, regime_stability),
                'stability_weight': self._calculate_stability_weight(regime_stability, historical_performance),
                'clustering_weight': self._calculate_clustering_weight(regime_confidence, market_volatility)
            }
            
            # Normalize weights to ensure they sum to 1.0
            total_weight = sum(weights.values())
            if total_weight > 0:
                for key in weights:
                    weights[key] /= total_weight
            
            return weights
            
        except Exception as e:
            self.logger.warning(f"⚠️ Dynamic consensus weighting failed: {e}")
            # Return default weights
            return {
                'temporal_weight': 0.25,
                'transition_weight': 0.25,
                'stability_weight': 0.25,
                'clustering_weight': 0.25
            }
    
    def _calculate_regime_confidence(self, tas_assignments: np.ndarray, semantic_assignments: np.ndarray) -> float:
        """Calculate regime confidence based on assignment consistency."""
        try:
            if len(tas_assignments) == 0 or len(semantic_assignments) == 0:
                return 0.5
            
            # Calculate agreement rate
            agreements = np.sum(tas_assignments == semantic_assignments)
            total_comparisons = min(len(tas_assignments), len(semantic_assignments))
            agreement_rate = agreements / total_comparisons if total_comparisons > 0 else 0.0
            
            # Calculate regime distribution consistency
            tas_unique = len(np.unique(tas_assignments))
            semantic_unique = len(np.unique(semantic_assignments))
            distribution_consistency = 1.0 - abs(tas_unique - semantic_unique) / max(tas_unique, semantic_unique, 1)
            
            # Combine agreement rate and distribution consistency
            confidence = (agreement_rate * 0.7 + distribution_consistency * 0.3)
            
            return max(0.0, min(1.0, confidence))
            
        except Exception:
            return 0.5
    
    def _calculate_market_volatility_proxy(self, tas_assignments: np.ndarray, semantic_assignments: np.ndarray) -> float:
        """Calculate market volatility proxy based on regime change frequency."""
        try:
            if len(tas_assignments) < 2 or len(semantic_assignments) < 2:
                return 0.5
            
            # Calculate regime change frequency for both assignments
            tas_changes = np.sum(tas_assignments[1:] != tas_assignments[:-1])
            semantic_changes = np.sum(semantic_assignments[1:] != semantic_assignments[:-1])
            
            # Normalize by sequence length
            tas_volatility = tas_changes / (len(tas_assignments) - 1)
            semantic_volatility = semantic_changes / (len(semantic_assignments) - 1)
            
            # Average volatility as proxy for market volatility
            avg_volatility = (tas_volatility + semantic_volatility) / 2.0
            
            return max(0.0, min(1.0, avg_volatility))
            
        except Exception:
            return 0.5
    
    def _calculate_historical_performance_proxy(self, tas_assignments: np.ndarray, semantic_assignments: np.ndarray) -> float:
        """Calculate historical performance proxy based on regime stability."""
        try:
            if len(tas_assignments) < 3 or len(semantic_assignments) < 3:
                return 0.5
            
            # Calculate regime stability for both assignments
            tas_stability = self._calculate_regime_stability(tas_assignments)
            semantic_stability = self._calculate_regime_stability(semantic_assignments)
            
            # Average stability as proxy for historical performance
            avg_stability = (tas_stability + semantic_stability) / 2.0
            
            return max(0.0, min(1.0, avg_stability))
            
        except Exception:
            return 0.5
    
    def _calculate_regime_stability_score(self, tas_assignments: np.ndarray, semantic_assignments: np.ndarray) -> float:
        """Calculate regime stability score."""
        try:
            if len(tas_assignments) < 3 or len(semantic_assignments) < 3:
                return 0.5
            
            # Calculate stability for both assignments
            tas_stability = self._calculate_regime_stability(tas_assignments)
            semantic_stability = self._calculate_regime_stability(semantic_assignments)
            
            # Calculate stability similarity
            stability_similarity = 1.0 - abs(tas_stability - semantic_stability) / max(tas_stability, semantic_stability, 1.0)
            
            # Combine individual stability and similarity
            combined_stability = (tas_stability + semantic_stability + stability_similarity) / 3.0
            
            return max(0.0, min(1.0, combined_stability))
            
        except Exception:
            return 0.5
    
    def _calculate_temporal_weight(self, regime_confidence: float, market_volatility: float) -> float:
        """Calculate temporal analysis weight based on confidence and volatility."""
        try:
            # Higher confidence and lower volatility favor temporal analysis
            base_weight = 0.25
            confidence_bonus = regime_confidence * 0.1
            volatility_penalty = market_volatility * 0.05
            
            weight = base_weight + confidence_bonus - volatility_penalty
            return max(0.1, min(0.5, weight))
            
        except Exception:
            return 0.25
    
    def _calculate_transition_weight(self, regime_confidence: float, regime_stability: float) -> float:
        """Calculate transition analysis weight based on confidence and stability."""
        try:
            # Higher confidence and stability favor transition analysis
            base_weight = 0.25
            confidence_bonus = regime_confidence * 0.1
            stability_bonus = regime_stability * 0.1
            
            weight = base_weight + confidence_bonus + stability_bonus
            return max(0.1, min(0.5, weight))
            
        except Exception:
            return 0.25
    
    def _calculate_stability_weight(self, regime_stability: float, historical_performance: float) -> float:
        """Calculate stability analysis weight based on stability and performance."""
        try:
            # Higher stability and performance favor stability analysis
            base_weight = 0.25
            stability_bonus = regime_stability * 0.1
            performance_bonus = historical_performance * 0.1
            
            weight = base_weight + stability_bonus + performance_bonus
            return max(0.1, min(0.5, weight))
            
        except Exception:
            return 0.25
    
    def _calculate_clustering_weight(self, regime_confidence: float, market_volatility: float) -> float:
        """Calculate clustering analysis weight based on confidence and volatility."""
        try:
            # Higher confidence and moderate volatility favor clustering analysis
            base_weight = 0.25
            confidence_bonus = regime_confidence * 0.1
            volatility_factor = 1.0 - abs(market_volatility - 0.5) * 0.1  # Optimal at 0.5 volatility
            
            weight = base_weight + confidence_bonus + volatility_factor
            return max(0.1, min(0.5, weight))
            
        except Exception:
            return 0.25
    
    def _analyze_regime_transitions(self, tas_assignments: np.ndarray, semantic_assignments: np.ndarray, regime_mapping: Dict[int, int]) -> float:
        """Analyze regime transition patterns for consensus bonus."""
        try:
            if len(regime_mapping) == 0:
                return 0.0
            
            # Calculate transition matrices
            tas_transitions = self._calculate_transition_matrix(tas_assignments)
            semantic_transitions = self._calculate_transition_matrix(semantic_assignments)
            
            # Calculate transition similarity
            transition_sim = self._calculate_transition_similarity(tas_transitions, semantic_transitions, regime_mapping)
            
            # Bonus based on transition similarity
            transition_bonus = transition_sim * 0.15  # Max 15% bonus
            
            return max(0.0, transition_bonus)
            
        except Exception as e:
            self.logger.warning(f"⚠️ Regime transition analysis failed: {e}")
            return 0.0
    
    def _calculate_transition_matrix(self, assignments: np.ndarray) -> Dict[tuple, int]:
        """Calculate regime transition matrix."""
        try:
            transitions = {}
            for i in range(len(assignments) - 1):
                transition = (assignments[i], assignments[i + 1])
                transitions[transition] = transitions.get(transition, 0) + 1
            return transitions
        except Exception:
            return {}
    
    def _calculate_transition_similarity(self, tas_transitions: Dict[tuple, int], semantic_transitions: Dict[tuple, int], regime_mapping: Dict[int, int]) -> float:
        """Calculate similarity between transition matrices."""
        try:
            if not tas_transitions or not semantic_transitions:
                return 0.0
            
            # Map semantic transitions to TAS regime space
            mapped_semantic_transitions = {}
            for (from_regime, to_regime), count in semantic_transitions.items():
                if from_regime in regime_mapping and to_regime in regime_mapping:
                    mapped_transition = (regime_mapping[from_regime], regime_mapping[to_regime])
                    mapped_semantic_transitions[mapped_transition] = count
            
            # Calculate similarity
            total_transitions = sum(tas_transitions.values()) + sum(mapped_semantic_transitions.values())
            if total_transitions == 0:
                return 0.0
            
            common_transitions = 0
            for transition, count in tas_transitions.items():
                semantic_count = mapped_semantic_transitions.get(transition, 0)
                common_transitions += min(count, semantic_count)
            
            similarity = (2 * common_transitions) / total_transitions
            return similarity
            
        except Exception:
            return 0.0
    
    def _analyze_consensus_stability(self, tas_assignments: np.ndarray, semantic_assignments: np.ndarray) -> float:
        """Analyze consensus stability over time."""
        try:
            if len(tas_assignments) < 20:
                return 0.0
            
            # Calculate rolling consensus
            window_size = min(20, len(tas_assignments) // 4)
            rolling_consensus = []
            
            for i in range(len(tas_assignments) - window_size + 1):
                window_tas = tas_assignments[i:i + window_size]
                window_semantic = semantic_assignments[i:i + window_size]
                window_consensus = np.mean(window_tas == window_semantic)
                rolling_consensus.append(window_consensus)
            
            if not rolling_consensus:
                return 0.0
            
            # Calculate stability (low variance = high stability)
            consensus_variance = np.var(rolling_consensus)
            stability_score = max(0.0, 1.0 - consensus_variance)
            
            # Bonus for high stability
            stability_bonus = stability_score * 0.1  # Max 10% bonus
            
            return max(0.0, stability_bonus)
            
        except Exception as e:
            self.logger.warning(f"⚠️ Consensus stability analysis failed: {e}")
            return 0.0
    
    def _analyze_clustering_quality(self, tas_assignments: np.ndarray, semantic_assignments: np.ndarray, regime_mapping: Dict[int, int]) -> float:
        """Analyze enhanced clustering quality for consensus bonus."""
        try:
            if len(regime_mapping) == 0:
                return 0.0
            
            # 1. Regime distribution similarity
            tas_dist = self._calculate_regime_distribution(tas_assignments)
            semantic_dist = self._calculate_regime_distribution(semantic_assignments)
            distribution_sim = self._calculate_distribution_similarity(tas_dist, semantic_dist, regime_mapping)
            
            # 2. Regime balance analysis
            tas_balance = self._calculate_regime_balance(tas_assignments)
            semantic_balance = self._calculate_regime_balance(semantic_assignments)
            balance_sim = 1.0 - abs(tas_balance - semantic_balance) / max(tas_balance, semantic_balance, 1.0)
            
            # 3. Cluster coherence analysis
            tas_coherence = self._calculate_cluster_coherence(tas_assignments)
            semantic_coherence = self._calculate_cluster_coherence(semantic_assignments)
            coherence_sim = 1.0 - abs(tas_coherence - semantic_coherence) / max(tas_coherence, semantic_coherence, 1.0)
            
            # 4. Cluster separation quality
            tas_separation = self._calculate_cluster_separation(tas_assignments)
            semantic_separation = self._calculate_cluster_separation(semantic_assignments)
            separation_sim = 1.0 - abs(tas_separation - semantic_separation) / max(tas_separation, semantic_separation, 1.0)
            
            # 5. Economic significance analysis
            tas_economic_sig = self._calculate_economic_significance(tas_assignments)
            semantic_economic_sig = self._calculate_economic_significance(semantic_assignments)
            economic_sim = 1.0 - abs(tas_economic_sig - semantic_economic_sig) / max(tas_economic_sig, semantic_economic_sig, 1.0)
            
            # 6. Regime diversity analysis
            tas_diversity = self._calculate_regime_diversity(tas_assignments)
            semantic_diversity = self._calculate_regime_diversity(semantic_assignments)
            diversity_sim = 1.0 - abs(tas_diversity - semantic_diversity) / max(tas_diversity, semantic_diversity, 1.0)
            
            # Combine all clustering quality metrics
            clustering_bonus = (
                distribution_sim * 0.25 +      # 25% weight for distribution similarity
                balance_sim * 0.20 +           # 20% weight for balance similarity
                coherence_sim * 0.20 +        # 20% weight for coherence similarity
                separation_sim * 0.15 +       # 15% weight for separation similarity
                economic_sim * 0.15 +         # 15% weight for economic significance
                diversity_sim * 0.05          # 5% weight for diversity similarity
            ) * 0.15  # Max 15% bonus (increased from 10%)
            
            return max(0.0, clustering_bonus)
            
        except Exception as e:
            self.logger.warning(f"⚠️ Enhanced clustering quality analysis failed: {e}")
            return 0.0
    
    def _calculate_cluster_coherence(self, assignments: np.ndarray) -> float:
        """Calculate cluster coherence based on regime consistency."""
        try:
            if len(assignments) < 3:
                return 0.5
            
            # Calculate regime frequency distribution
            unique_regimes, counts = np.unique(assignments, return_counts=True)
            total_samples = len(assignments)
            
            # Calculate entropy (lower entropy = higher coherence)
            probabilities = counts / total_samples
            entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
            max_entropy = np.log2(len(unique_regimes))
            
            # Coherence is inverse of normalized entropy
            coherence = 1.0 - (entropy / max_entropy) if max_entropy > 0 else 1.0
            
            return max(0.0, min(1.0, coherence))
            
        except Exception:
            return 0.5
    
    def _calculate_cluster_separation(self, assignments: np.ndarray) -> float:
        """Calculate cluster separation quality."""
        try:
            if len(assignments) < 3:
                return 0.5
            
            # Calculate regime distribution
            unique_regimes, counts = np.unique(assignments, return_counts=True)
            total_samples = len(assignments)
            
            # Calculate Gini coefficient as a measure of separation
            # Higher Gini = better separation (more unequal distribution)
            probabilities = counts / total_samples
            probabilities_sorted = np.sort(probabilities)
            n = len(probabilities_sorted)
            
            # Calculate Gini coefficient
            cumsum = np.cumsum(probabilities_sorted)
            gini = (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n if cumsum[-1] > 0 else 0.0
            
            # Convert Gini to separation quality (0-1 scale)
            separation = min(1.0, gini * 2.0)  # Scale Gini to 0-1
            
            return max(0.0, min(1.0, separation))
            
        except Exception:
            return 0.5
    
    def _calculate_economic_significance(self, assignments: np.ndarray) -> float:
        """Calculate economic significance of regime assignments."""
        try:
            if len(assignments) < 3:
                return 0.5
            
            # Calculate regime diversity (more diverse = more economically significant)
            unique_regimes = len(np.unique(assignments))
            total_samples = len(assignments)
            
            # Diversity ratio
            diversity_ratio = unique_regimes / total_samples
            
            # Economic significance based on diversity and distribution
            # More diverse regimes with balanced distribution are more economically significant
            unique_regimes, counts = np.unique(assignments, return_counts=True)
            probabilities = counts / total_samples
            
            # Calculate distribution balance
            max_prob = np.max(probabilities)
            min_prob = np.min(probabilities)
            balance = 1.0 - (max_prob - min_prob) if max_prob > 0 else 0.0
            
            # Combine diversity and balance
            economic_significance = (diversity_ratio * 0.6 + balance * 0.4)
            
            return max(0.0, min(1.0, economic_significance))
            
        except Exception:
            return 0.5
    
    def _calculate_regime_diversity(self, assignments: np.ndarray) -> float:
        """Calculate regime diversity."""
        try:
            if len(assignments) < 2:
                return 0.0
            
            # Calculate number of unique regimes
            unique_regimes = len(np.unique(assignments))
            total_samples = len(assignments)
            
            # Diversity is the ratio of unique regimes to total samples
            diversity = unique_regimes / total_samples
            
            return max(0.0, min(1.0, diversity))
            
        except Exception:
            return 0.0
    
    def _calculate_adaptive_consensus_threshold(
        self, 
        tas_assignments: np.ndarray, 
        semantic_assignments: np.ndarray, 
        regime_mapping: Dict[int, int]
    ) -> float:
        """Calculate adaptive consensus threshold based on market volatility and regime stability."""
        try:
            # Base threshold
            base_threshold = 0.5
            
            # 1. Market volatility factor
            market_volatility = self._calculate_market_volatility_proxy(tas_assignments, semantic_assignments)
            volatility_factor = 1.0 + (market_volatility - 0.5) * 0.2  # ±10% adjustment
            
            # 2. Regime stability factor
            regime_stability = self._calculate_regime_stability_score(tas_assignments, semantic_assignments)
            stability_factor = 1.0 + (regime_stability - 0.5) * 0.3  # ±15% adjustment
            
            # 3. Regime diversity factor
            tas_diversity = self._calculate_regime_diversity(tas_assignments)
            semantic_diversity = self._calculate_regime_diversity(semantic_assignments)
            avg_diversity = (tas_diversity + semantic_diversity) / 2.0
            diversity_factor = 1.0 + (avg_diversity - 0.5) * 0.1  # ±5% adjustment
            
            # 4. Historical performance factor
            historical_performance = self._calculate_historical_performance_proxy(tas_assignments, semantic_assignments)
            performance_factor = 1.0 + (historical_performance - 0.5) * 0.2  # ±10% adjustment
            
            # 5. Regime mapping quality factor
            mapping_quality = len(regime_mapping) / max(len(np.unique(tas_assignments)), len(np.unique(semantic_assignments)), 1)
            mapping_factor = 1.0 + (mapping_quality - 0.5) * 0.15  # ±7.5% adjustment
            
            # Calculate adaptive threshold
            adaptive_threshold = base_threshold * volatility_factor * stability_factor * diversity_factor * performance_factor * mapping_factor
            
            # Ensure threshold is within reasonable bounds
            adaptive_threshold = max(0.3, min(0.8, adaptive_threshold))
            
            return adaptive_threshold
            
        except Exception as e:
            self.logger.warning(f"⚠️ Adaptive threshold calculation failed: {e}")
            return 0.5
    
    def _calculate_distribution_similarity(self, tas_dist: Dict[str, float], semantic_dist: Dict[str, float], regime_mapping: Dict[int, int]) -> float:
        """Calculate similarity between regime distributions."""
        try:
            if not tas_dist or not semantic_dist:
                return 0.0
            
            total_similarity = 0.0
            mapping_count = 0
            
            for nas_regime, tas_regime in regime_mapping.items():
                tas_key = f'regime_{tas_regime}'
                semantic_key = f'regime_{nas_regime}'
                
                if tas_key in tas_dist and semantic_key in semantic_dist:
                    tas_pct = tas_dist[tas_key]
                    semantic_pct = semantic_dist[semantic_key]
                    
                    similarity = 1.0 - abs(tas_pct - semantic_pct) / 100.0
                    total_similarity += similarity
                    mapping_count += 1
            
            return total_similarity / mapping_count if mapping_count > 0 else 0.0
            
        except Exception:
            return 0.0
    
    def _calculate_regime_balance(self, assignments: np.ndarray) -> float:
        """Calculate regime balance (entropy-based)."""
        try:
            if len(assignments) == 0:
                return 0.0
            
            # Calculate regime frequencies
            unique, counts = np.unique(assignments, return_counts=True)
            frequencies = counts / len(assignments)
            
            # Calculate entropy (higher entropy = more balanced)
            entropy = -np.sum(frequencies * np.log2(frequencies + 1e-10))
            
            # Normalize by maximum possible entropy
            max_entropy = np.log2(len(unique)) if len(unique) > 1 else 1.0
            normalized_entropy = entropy / max_entropy
            
            return normalized_entropy
            
        except Exception:
            return 0.0
    
    def _assess_numerical_divergence_fallback(self, tas_assignments: np.ndarray, nas_assignments: np.ndarray) -> Dict[str, Any]:
        """Fallback numerical divergence assessment when semantic analysis fails."""
        try:
            disagreement_mask = tas_assignments != nas_assignments
            numerical_divergence_rate = np.mean(disagreement_mask)
            
            return {
                'semantic_divergence_rate': numerical_divergence_rate,
                'regime_mapping': {},
                'mapping_quality': 0.5,
                'raw_consensus': 1.0 - numerical_divergence_rate,
                'semantic_consensus': 1.0 - numerical_divergence_rate,
                'consensus_improvement': 0.0,
                'assessment_method': 'numerical_fallback'
            }
            
        except Exception as e:
            self.logger.warning(f"⚠️ Numerical divergence fallback failed: {e}")
            return {
                'semantic_divergence_rate': 1.0,
                'regime_mapping': {},
                'mapping_quality': 0.0,
                'raw_consensus': 0.0,
                'semantic_consensus': 0.0,
                'consensus_improvement': 0.0,
                'assessment_method': 'failed'
            }


def create_hybrid_orchestrator(config: HybridOrchestratorConfig) -> HybridOrchestrator:
    """Create a hybrid orchestrator instance.

    Args:
        config: Hybrid orchestrator configuration

    Returns:
        HybridOrchestrator instance
    """
    return HybridOrchestrator(config)


async def run_hybrid_orchestrator_example():
    """Example function to run the hybrid orchestrator with sample configuration."""
    try:
        print("🚀 Starting Hybrid NAS-TAS Orchestrator Example...")

        # Create configuration
        config = HybridOrchestratorConfig(
            symbol="BTCUSDT",
            timeframe="15m",
            start_date="2024-01-01",
            end_date="2024-12-31",
            use_standardized_features=True,
            feature_categories=['momentum', 'volatility', 'volume', 'trend'],
            significance_threshold=0.5,
            min_regime_duration=10,
            viability_threshold=0.5,
            minimum_regime_duration=5,
            max_iterations=100,
            use_bayesian_optimization=True,
            population_size=100,
            max_generations=50,
            use_nsga2=True,
            use_spea2=True,
            use_gpu_acceleration=True,
            memory_limit_gb=8.0,
            include_detailed_metrics=True,
            save_to_file=True
        )

        # Create orchestrator
        orchestrator = create_hybrid_orchestrator(config)

        # Execute pipeline
        print("🔄 Executing hybrid pipeline...")
        results = await orchestrator.execute_hybrid_pipeline()

        print("✅ Pipeline execution completed!")
        print(f"📊 Results success: {results.success}")
        print(f"⚡ Execution time: {results.execution_time:.2f}s")

        if results.success:
            print("🎯 Pipeline completed successfully!")
        else:
            print(f"❌ Pipeline failed: {results.error_message}")

        return results

    except Exception as e:
        print(f"❌ Example execution failed: {e}")
        return None


if __name__ == "__main__":
    """Main entry point for running the hybrid orchestrator."""
    import asyncio

    # Run the example
    asyncio.run(run_hybrid_orchestrator_example())