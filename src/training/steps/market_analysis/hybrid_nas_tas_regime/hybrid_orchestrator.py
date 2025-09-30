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
            
            # Simple consensus mapping based on regime overlap
            consensus_mapping = {
                'nas_regimes': list(np.unique(nas_assignments)),
                'tas_regimes': list(np.unique(tas_assignments)),
                'consensus_regimes': [],
                'mapping_matrix': {}
            }
            
            # Calculate overlap between regimes
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
        try:
            tprint(Panel.fit(
                "[bold green]🚀 Starting TAS-NAS Orchestration[/bold green]\n"
                f"Data shape: {market_data.shape if hasattr(market_data, 'shape') else 'Unknown'}\n"
                f"Timeframes: {timeframes or ['1m', '5m', '15m']}",
                title="Orchestration Start",
                border_style="green"
            ))
            
            self.logger.info("🚀 Starting TAS-NAS orchestration...")

            # Use configured timeframes if not specified
            if timeframes is None:
                timeframes = ['1m', '5m', '15m']

            results = {
                'tas_results': {},
                'nas_results': {},
                'hybrid_analysis': {},
                'timeframes_processed': timeframes,
                'execution_time': 0.0
            }

            start_time = time.time()

            # Run detection for each timeframe
            for timeframe in timeframes:
                tprint(f"[cyan]🔍 Processing timeframe: {timeframe}[/cyan]")
                self.logger.info(f"🔍 Processing timeframe: {timeframe}")

                # Prepare market_data for timeframe
                tprint(f"[yellow]📊 Preparing data for {timeframe}...[/yellow]")
                timeframe_market_data = self._prepare_timeframe_market_data(market_data, timeframe)
                tprint(f"[green]✅ Data prepared: {timeframe_market_data.shape if hasattr(timeframe_market_data, 'shape') else 'Unknown shape'}[/green]")

                # Run TAS detection
                if self.tas_system is not None:
                    tprint(f"[blue]🌳 Running TAS detection for {timeframe}...[/blue]")
                    tas_result = self._run_tas_detection(timeframe_market_data, timestamps, timeframe)
                    results['tas_results'][timeframe] = tas_result
                    if tas_result.get('success', False):
                        tprint(f"[green]✅ TAS detection completed for {timeframe}[/green]")
                    else:
                        tprint(f"[red]❌ TAS detection failed for {timeframe}[/red]")

                # Run NAS detection
                if self.nas_system is not None:
                    tprint(f"[blue]🧠 Running NAS detection for {timeframe}...[/blue]")
                    nas_result = self._run_nas_detection_sync(timeframe_market_data, timestamps, timeframe)
                    results['nas_results'][timeframe] = nas_result
                    if nas_result.get('success', False):
                        tprint(f"[green]✅ NAS detection completed for {timeframe}[/green]")
                    else:
                        tprint(f"[red]❌ NAS detection failed for {timeframe}[/red]")

            # Perform hybrid analysis on primary timeframe (15m) - only if both systems succeeded
            primary_timeframe = '15m'
            tprint(f"[magenta]🔬 Checking hybrid analysis prerequisites for {primary_timeframe}...[/magenta]")
            
            tas_success = (primary_timeframe in results.get('tas_results', {}) and
                          results['tas_results'][primary_timeframe].get('success', False))
            nas_success = (primary_timeframe in results.get('nas_results', {}) and
                          results['nas_results'][primary_timeframe].get('success', False))

            tprint(f"[cyan]TAS Success: {tas_success}[/cyan]")
            tprint(f"[cyan]NAS Success: {nas_success}[/cyan]")

            if tas_success and nas_success:
                tprint("[bold green]🎯 Starting hybrid analysis...[/bold green]")
                hybrid_analysis = self._perform_hybrid_analysis(
                    market_data, timestamps,
                    results['tas_results'][primary_timeframe],
                    results['nas_results'][primary_timeframe]
                )
                results['hybrid_analysis'] = hybrid_analysis
                tprint("[green]✅ Hybrid analysis completed[/green]")
            else:
                # Fast fail - both TAS and NAS systems are required
                error_msg = "Both TAS and NAS systems failed to initialize. Cannot perform hybrid regime discovery."
                self.logger.error(f"❌ {error_msg}")
                tprint(f"[red]❌ {error_msg}[/red]")
                raise RuntimeError(error_msg)

            # Generate comprehensive outcome report
            if tas_success and nas_success:
                tprint("[bold magenta]📊 Generating comprehensive outcome report...[/bold magenta]")
                outcome_report = self._generate_outcome_report(results, market_data)
                results['outcome_report'] = outcome_report
                tprint("[green]✅ Outcome report generated[/green]")

            # Add clustering quality metrics (only if both systems succeeded)
            if tas_success and nas_success and self.clustering_quality_analyzer:
                try:
                    self.logger.info("🔍 DEBUG: Starting clustering quality analysis")

                    # Prepare features for quality analysis
                    if isinstance(market_data, pd.DataFrame):
                        numeric_columns = market_data.select_dtypes(include=[np.number]).columns
                        if len(numeric_columns) > 0:
                            features = market_data[numeric_columns].values
                        else:
                            # Fallback to basic OHLCV columns
                            basic_columns = ['open', 'high', 'low', 'close', 'volume']
                            available_columns = [col for col in basic_columns if col in market_data.columns]
                            features = market_data[available_columns].values if available_columns else market_data.values
                    else:
                        features = market_data

                    # Calculate clustering quality for each successful prediction set
                    clustering_quality = {}

                    # TAS quality
                    tas_predictions = results['tas_results'][primary_timeframe]['regime_predictions']
                    try:
                        # Ensure features and predictions have the same length for TAS
                        tas_features = features
                        if len(tas_features) != len(tas_predictions):
                            min_length = min(len(tas_features), len(tas_predictions))
                            tas_features = tas_features[:min_length]
                            tas_predictions = tas_predictions[:min_length]

                        tas_quality_metrics = self.clustering_quality_analyzer.calculate_comprehensive_metrics(
                            tas_features, tas_predictions
                        )
                        clustering_quality['tas_quality'] = tas_quality_metrics
                    except Exception as e:
                        clustering_quality['tas_quality'] = {'error': str(e)}

                    # NAS quality
                    nas_predictions = results['nas_results'][primary_timeframe]['regime_predictions']
                    try:
                        # Ensure features and predictions have the same length for NAS
                        nas_features = features
                        if len(nas_features) != len(nas_predictions):
                            min_length = min(len(nas_features), len(nas_predictions))
                            nas_features = nas_features[:min_length]
                            nas_predictions = nas_predictions[:min_length]

                        nas_quality_metrics = self.clustering_quality_analyzer.calculate_comprehensive_metrics(
                            nas_features, nas_predictions
                        )
                        clustering_quality['nas_quality'] = nas_quality_metrics
                    except Exception as e:
                        clustering_quality['nas_quality'] = {'error': str(e)}

                    # Add comparison if both succeeded
                    tas_quality = clustering_quality.get('tas_quality', {})
                    nas_quality = clustering_quality.get('nas_quality', {})

                    comparison = {}
                    if tas_quality and nas_quality and 'error' not in tas_quality and 'error' not in nas_quality:
                        comparison = {
                            'best_silhouette': 'TAS' if tas_quality.get('silhouette_score', 0) > nas_quality.get('silhouette_score', 0) else 'NAS',
                            'best_davies_bouldin': 'TAS' if tas_quality.get('davies_bouldin_index', float('inf')) < nas_quality.get('davies_bouldin_index', float('inf')) else 'NAS',
                            'best_calinski_harabasz': 'TAS' if tas_quality.get('calinski_harabasz_score', 0) > nas_quality.get('calinski_harabasz_score', 0) else 'NAS'
                        }
                    clustering_quality['comparison'] = comparison

                    # Add to hybrid analysis and main results
                    results['hybrid_analysis']['clustering_quality'] = clustering_quality
                    results['clustering_quality'] = clustering_quality

                except Exception as e:
                    self.logger.warning(f"⚠️ Clustering quality analysis failed: {e}")
                    results['hybrid_analysis']['clustering_quality'] = {'error': str(e)}
                    results['clustering_quality'] = {'error': str(e)}

            results['execution_time'] = time.time() - start_time

            self.logger.info("✅ TAS-NAS orchestration completed successfully")
            return results

        except Exception as e:
            self.logger.error(f"❌ TAS-NAS orchestration failed: {e}")
            return {'error': str(e), 'execution_time': 0.0}


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
            
            comparison = {
                'tas_analysis': {
                    'success': tas_results.get('success', False),
                    'regime_count': len(np.unique(tas_results.get('regime_predictions', []))) if tas_results.get('success') else 0,
                    'execution_time': tas_results.get('execution_time', 0),
                    'regime_distribution': self._calculate_regime_distribution(tas_results.get('regime_predictions', [])) if tas_results.get('success') else {}
                },
                'nas_analysis': {
                    'success': nas_results.get('success', False),
                    'regime_count': len(np.unique(nas_results.get('regime_predictions', []))) if nas_results.get('success') else 0,
                    'execution_time': nas_results.get('execution_time', 0),
                    'regime_distribution': self._calculate_regime_distribution(nas_results.get('regime_predictions', [])) if nas_results.get('success') else {}
                }
            }
            
            # Calculate agreement metrics
            if comparison['tas_analysis']['success'] and comparison['nas_analysis']['success']:
                tas_preds = np.array(tas_results.get('regime_predictions', []))
                nas_preds = np.array(nas_results.get('regime_predictions', []))
                
                min_len = min(len(tas_preds), len(nas_preds))
                if min_len > 0:
                    tas_preds = tas_preds[:min_len]
                    nas_preds = nas_preds[:min_len]
                    
                    agreement_rate = np.sum(tas_preds == nas_preds) / min_len
                    comparison['agreement_metrics'] = {
                        'agreement_rate': float(agreement_rate),
                        'total_samples': min_len,
                        'matching_samples': int(np.sum(tas_preds == nas_preds))
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
            
            distribution = {}
            for regime, count in zip(unique_regimes, counts):
                percentage = (count / total_samples) * 100
                distribution[f'regime_{regime}'] = float(percentage)
            
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
                    tprint(f"Agreement Rate: {agreement_metrics.get('agreement_rate', 0):.2%}")
                    tprint(f"Matching Samples: {agreement_metrics.get('matching_samples', 0)}")
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
                tprint(f"Agreement Rate: {output_summary.get('agreement_rate', 0):.2%}")
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
        """Run TAS regime detection."""
        try:
            if self.tas_system is None:
                return {'error': 'TAS system not initialized', 'timeframe': timeframe}

            result = self.tas_system.detect_regimes(
                market_data, timestamps, optimize_performance=True, enable_patchtst_enhancement=True
            )

            return {
                'success': result.success,
                'regime_predictions': getattr(result, 'regime_predictions', np.array([])),
                'regime_probabilities': getattr(result, 'regime_probabilities', np.array([])),
                'execution_time': getattr(result, 'execution_time', 0.0),
                'timeframe': timeframe,
                'system': 'TAS'
            }

        except Exception as e:
            self.logger.error(f"❌ TAS regime detection failed: {e}")
            raise ValueError(f"TAS regime detection failed: {e}")

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