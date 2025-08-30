# src/tactician/sr_breakout_predictor.py

# from src.analyst.unified_regime_classifier import UnifiedRegimeClassifier  # Temporarily commented due to syntax errors
from src.utils.logger import system_logger
from typing import Any
import numpy as np
import pandas as pd
import os
import json
from src.utils.error_handler import handle_errors, handle_specific_errors
from src.utils.centralized_decorators import validate_data_quality

# DBSCAN clustering for S/R level analysis
try:
    from sklearn.cluster import DBSCAN
    DBSCAN_AVAILABLE = True
except ImportError:
    DBSCAN_AVAILABLE = False
    print("Warning: sklearn not available, DBSCAN clustering will be disabled")

class SRBreakoutPredictor:
    """
    SR Breakout Predictor responsible for predicting support/resistance breakouts.
    This module handles all SR breakout prediction logic and feature engineering.
    Centralized S/R detection using multiple methods:
    - Fractal analysis for swing highs/lows
    - Volume-weighted price levels
    - Traditional pivot points (fallback)
    - ATR-based activation ranges
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """
        Initialize SR breakout predictor.

        Args:
            config: Configuration dictionary
        """
        self.config: dict[str, Any] = config
        self.logger = system_logger.getChild("SRBreakoutPredictor")

        # SR predictor state
        self.is_initialized: bool = False
        self.sr_predictions: dict[str, Any] = {}
        
        # Reporting system
        self.reporting_enabled: bool = self.sr_config.get("enable_detailed_reporting", True)
        self.report_directory: str = self.sr_config.get("report_directory", "reports/sr_analysis")
        self.report_format: str = self.sr_config.get("report_format", "json")  # json, csv, html
        self.report_retention_days: int = self.sr_config.get("report_retention_days", 30)
        self.metrics_history: list[dict[str, Any]] = []
        self.current_report_id: str = ""

        # Configuration
        self.sr_config: dict[str, Any] = self.config.get("sr_breakout_predictor", {})
        self.enable_sr_breakout_tactics: bool = self.sr_config.get(
            "enable_sr_breakout_tactics",
            True,
        )
        self.sr_proximity_threshold: float = self.sr_config.get(
            "sr_proximity_threshold",
            0.02,
        )
        self.breakout_confidence_threshold: float = self.sr_config.get(
            "breakout_confidence_threshold",
            0.6,
        )
        self.sr_detection_method: str = self.sr_config.get(
            "sr_detection_method",
            "fractal",
        )
        self.min_sr_strength: float = self.sr_config.get(
            "min_sr_strength",
            0.3,
        )
        self.max_sr_levels: int = self.sr_config.get(
            "max_sr_levels",
            10,
        )
        self.sr_lookback_periods: int = self.sr_config.get(
            "sr_lookback_periods",
            100,
        )
        self.volume_weight: float = self.sr_config.get(
            "volume_weight",
            0.7,
        )
        self.price_weight: float = self.sr_config.get(
            "price_weight",
            0.3,
        )
        self.atr_multiplier: float = self.sr_config.get(
            "atr_multiplier",
            1.5,
        )
        self.breakout_confirmation_periods: int = self.sr_config.get(
            "breakout_confirmation_periods",
            3,
        )
        self.false_breakout_filter: bool = self.sr_config.get(
            "false_breakout_filter",
            True,
        )

        # Zone multipliers
        self.support_zone_multiplier: float = self.sr_config.get(
            "support_zone_multiplier",
            0.8,
        )
        self.resistance_zone_multiplier: float = self.sr_config.get(
            "resistance_zone_multiplier",
            1.2,
        )
        self.sr_zone_threshold: float = self.sr_config.get(
            "sr_zone_threshold",
            0.01,
        )
        self.zone_expansion_factor: float = self.sr_config.get(
            "zone_expansion_factor",
            1.1,
        )
        self.zone_contraction_factor: float = self.sr_config.get(
            "zone_contraction_factor",
            0.9,
        )

        # Confidence thresholds
        self.min_sr_confidence: float = self.sr_config.get(
            "min_sr_confidence",
            0.4,
        )
        self.high_confidence_threshold: float = self.sr_config.get(
            "high_confidence_threshold",
            0.8,
        )
        self.confidence_decay_rate: float = self.sr_config.get(
            "confidence_decay_rate",
            0.95,
        )
        self.regime_confidence_boost: float = self.sr_config.get(
            "regime_confidence_boost",
            0.1,
        )
        self.ensemble_confidence_threshold: float = self.sr_config.get(
            "ensemble_confidence_threshold",
            0.7,
        )

        # Feature calculation parameters
        self.feature_config: dict[str, Any] = self.sr_config.get(
            "feature_calculation",
            {},
        )
        self.enable_comprehensive_features: bool = self.feature_config.get(
            "enable_comprehensive_features",
            True,
        )
        self.strength_score_weights: dict[str, float] = self.feature_config.get(
            "strength_score_weights",
            {
                "touch_count": 0.3,
                "total_volume": 0.2,
                "level_age": 0.2,
                "bounce_rate": 0.2,
                "isolation_score": 0.1,
            },
        )

        # LM Model Selection Configuration
        self.lm_config: dict[str, Any] = self.sr_config.get("lm_model_selection", {})
        self.enable_specialist_models: bool = self.lm_config.get(
            "enable_specialist_models",
            True,
        )
        self.sr_proximity_trigger_base: float = self.lm_config.get(
            "sr_proximity_trigger_base",
            0.006,
        )  # 0.6% base proximity
        self.sr_proximity_trigger_min: float = self.lm_config.get(
            "sr_proximity_trigger_min",
            0.003,
        )  # 0.3% minimum proximity
        self.sr_proximity_trigger_max: float = self.lm_config.get(
            "sr_proximity_trigger_max",
            0.015,
        )  # 1.5% maximum proximity
        self.proximity_decay_rate: float = self.lm_config.get(
            "proximity_decay_rate",
            0.98,
        )
        self.proximity_boost_factor: float = self.lm_config.get(
            "proximity_boost_factor",
            1.2,
        )

        # Model ensemble configuration
        self.ensemble_config: dict[str, Any] = self.sr_config.get("ensemble_config", {})
        self.enable_ensemble: bool = self.ensemble_config.get(
            "enable_ensemble",
            True,
        )
        self.ensemble_method: str = self.ensemble_config.get(
            "ensemble_method",
            "weighted_average",
        )
        self.model_weights: dict[str, float] = self.ensemble_config.get(
            "model_weights",
            {
                "fractal": 0.4,
                "volume": 0.3,
                "pivot": 0.2,
                "atr": 0.1,
            },
        )

        # Performance tracking
        self.performance_metrics: dict[str, Any] = {}
        self.prediction_history: list[dict[str, Any]] = []

        # DBSCAN clustering configuration
        self.dbscan_config: dict[str, Any] = self.sr_config.get("dbscan_clustering", {})
        self.enable_dbscan_clustering: bool = self.dbscan_config.get("enable_dbscan_clustering", True)
        self.dbscan_eps: float = self.dbscan_config.get("eps", 0.01)  # 1% of price for neighborhood
        self.dbscan_min_samples: int = self.dbscan_config.get("min_samples", 3)  # Minimum points for cluster
        self.dbscan_enable_noise_filtering: bool = self.dbscan_config.get("enable_noise_filtering", True)

        # Enhanced strength calculation configuration
        self.strength_config: dict[str, Any] = self.sr_config.get("strength_calculation", {})
        self.enable_enhanced_strength: bool = self.strength_config.get("enable_enhanced_strength", True)
        self.touch_count_lookback: int = self.strength_config.get("touch_count_lookback", 100)
        self.bounce_rate_threshold: float = self.strength_config.get("bounce_rate_threshold", 0.02)  # 2% bounce
        self.isolation_distance_threshold: float = self.strength_config.get("isolation_distance_threshold", 0.05)  # 5% distance
        self.age_decay_factor: float = self.strength_config.get("age_decay_factor", 0.95)  # 5% decay per period

        # Optimization integration
        self.optimized_params: Optional[dict[str, Any]] = None
        self.use_optimized_params: bool = self.sr_config.get("use_optimized_params", True)
        
        # Advanced S/R method configuration
        self.advanced_config: dict[str, Any] = self.sr_config.get("advanced_sr_methods", {})
        self.enable_fibonacci_analysis: bool = self.advanced_config.get("enable_fibonacci_analysis", True)
        self.enable_elliott_wave_analysis: bool = self.advanced_config.get("enable_elliott_wave_analysis", True)
        self.enable_order_flow_analysis: bool = self.advanced_config.get("enable_order_flow_analysis", True)
        
        # Advanced method parameters
        self.fibonacci_sensitivity: float = self.advanced_config.get("fibonacci_sensitivity", 0.7)
        self.elliott_confidence_threshold: float = self.advanced_config.get("elliott_confidence_threshold", 0.6)
        self.order_flow_hvn_threshold: float = self.advanced_config.get("order_flow_hvn_threshold", 1.5)
        
        # Multi-timeframe configuration
        self.timeframe_config: dict[str, Any] = self.sr_config.get("multi_timeframe", {})
        self.enable_multi_timeframe: bool = self.timeframe_config.get("enable_multi_timeframe", True)
        self.timeframe_weights: dict[str, float] = self.timeframe_config.get("timeframe_weights", {
            "1m": 0.05, "5m": 0.1, "15m": 0.15, "1h": 0.25, "4h": 0.25, "1d": 0.2
        })

    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid SR breakout predictor configuration"),
            AttributeError: (False, "Missing required SR parameters"),
            KeyError: (False, "Missing configuration keys"),
        },
        default_return=False,
        context="SR breakout predictor initialization",
    )
    async def initialize(self) -> bool:
        """Initialize the SR breakout predictor."""
        self.logger.info("Initializing SR breakout predictor...")

        try:
            # Validate configuration
            if not self._validate_configuration():
                return False

            # Initialize components
            if not await self._initialize_components():
                return False

            self.is_initialized = True
            self.logger.info("✅ SR breakout predictor initialized successfully")
            
            # Initialize reporting system
            if self.reporting_enabled:
                self._initialize_reporting_system()
            
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize SR breakout predictor: {e}")
            return False

    def _validate_configuration(self) -> bool:
        """Validate SR breakout predictor configuration."""
        try:
            required_keys = [
                "sr_proximity_threshold",
                "breakout_confidence_threshold",
                "min_sr_strength",
                "max_sr_levels",
            ]
            for key in required_keys:
                if not hasattr(self, key):
                    self.logger.error(f"Missing required configuration key: {key}")
                    return False

            # Validate values
            if self.sr_proximity_threshold <= 0:
                self.logger.error("Invalid sr_proximity_threshold")
                return False

            if self.breakout_confidence_threshold <= 0 or self.breakout_confidence_threshold >= 1:
                self.logger.error("Invalid breakout_confidence_threshold")
                return False

            if self.min_sr_strength <= 0 or self.min_sr_strength >= 1:
                self.logger.error("Invalid min_sr_strength")
                return False

            if self.max_sr_levels <= 0:
                self.logger.error("Invalid max_sr_levels")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Configuration validation failed: {e}")
            return False

    async def _initialize_components(self) -> bool:
        """Initialize SR breakout predictor components."""
        try:
            # Initialize regime classifier if needed
            # Note: regime_classifier is currently commented out due to import issues
            # if hasattr(self, "regime_classifier"):
            #     await self.regime_classifier.initialize()

            # Load optimized parameters if enabled
            if self.use_optimized_params:
                await self._load_optimized_parameters()

            self.logger.info("✅ SR breakout predictor components initialized")
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
            return False

    async def _load_optimized_parameters(self) -> None:
        """Load optimized parameters from optimization results."""
        try:
            # Try to load from optimization results file
            optimization_file = self.sr_config.get("optimization_results_file", "optimization_results.json")
            
            if os.path.exists(optimization_file):
                with open(optimization_file, 'r') as f:
                    data = json.load(f)
                
                if data.get("best_result"):
                    best_result = data["best_result"]
                    
                    # Apply optimized parameters
                    self.optimized_params = {
                        "method_weights": best_result.get("method_weights", {}),
                        "strength_weights": best_result.get("strength_weights", {}),
                        "dbscan_params": best_result.get("dbscan_params", {}),
                        "timeframe_weights": best_result.get("timeframe_weights", {}),
                        "advanced_params": best_result.get("advanced_params", {}),
                    }
                    
                    # Update current parameters with optimized values
                    await self._apply_optimized_parameters()
                    
                    self.logger.info("✅ Loaded and applied optimized parameters")
                else:
                    self.logger.warning("No best result found in optimization file")
            else:
                self.logger.info("No optimization results file found, using default parameters")
                
        except Exception as e:
            self.logger.error(f"Failed to load optimized parameters: {e}")

    async def _apply_optimized_parameters(self) -> None:
        """Apply optimized parameters to the S/R predictor."""
        try:
            if not self.optimized_params:
                return
            
            # Apply method weights
            method_weights = self.optimized_params.get("method_weights", {})
            if method_weights:
                self.model_weights.update(method_weights)
                self.logger.info(f"Applied optimized method weights: {method_weights}")
            
            # Apply strength weights
            strength_weights = self.optimized_params.get("strength_weights", {})
            if strength_weights:
                self.strength_score_weights.update(strength_weights)
                self.logger.info(f"Applied optimized strength weights: {strength_weights}")
            
            # Apply DBSCAN parameters
            dbscan_params = self.optimized_params.get("dbscan_params", {})
            if dbscan_params:
                if "eps" in dbscan_params:
                    self.dbscan_eps = dbscan_params["eps"]
                if "min_samples" in dbscan_params:
                    self.dbscan_min_samples = dbscan_params["min_samples"]
                self.logger.info(f"Applied optimized DBSCAN parameters: {dbscan_params}")
            
            # Apply advanced parameters
            advanced_params = self.optimized_params.get("advanced_params", {})
            if advanced_params:
                # Apply Fibonacci parameters
                if "fibonacci_sensitivity" in advanced_params:
                    self.fibonacci_sensitivity = advanced_params["fibonacci_sensitivity"]
                    self.logger.info(f"Applied optimized Fibonacci sensitivity: {self.fibonacci_sensitivity}")
                
                # Apply Elliott Wave parameters
                if "elliott_confidence_threshold" in advanced_params:
                    self.elliott_confidence_threshold = advanced_params["elliott_confidence_threshold"]
                    self.logger.info(f"Applied optimized Elliott confidence threshold: {self.elliott_confidence_threshold}")
                
                # Apply Order Flow parameters
                if "order_flow_hvn_threshold" in advanced_params:
                    self.order_flow_hvn_threshold = advanced_params["order_flow_hvn_threshold"]
                    self.logger.info(f"Applied optimized Order Flow HVN threshold: {self.order_flow_hvn_threshold}")
                
                self.logger.info(f"Applied optimized advanced parameters: {advanced_params}")
            
            # Apply timeframe weights
            timeframe_weights = self.optimized_params.get("timeframe_weights", {})
            if timeframe_weights:
                self.timeframe_weights = timeframe_weights
                self.logger.info(f"Applied optimized timeframe weights: {timeframe_weights}")
                
        except Exception as e:
            self.logger.error(f"Failed to apply optimized parameters: {e}")

    async def set_optimized_parameters(self, optimized_params: dict[str, Any]) -> None:
        """Set optimized parameters directly."""
        try:
            self.optimized_params = optimized_params
            await self._apply_optimized_parameters()
            self.logger.info("✅ Set optimized parameters directly")
        except Exception as e:
            self.logger.error(f"Failed to set optimized parameters: {e}")

    def get_current_parameters(self) -> dict[str, Any]:
        """Get current parameters for comparison."""
        return {
            "method_weights": self.model_weights,
            "strength_weights": self.strength_score_weights,
            "dbscan_params": {
                "eps": self.dbscan_eps,
                "min_samples": self.dbscan_min_samples,
            },
            "advanced_params": {
                "fibonacci_sensitivity": self.fibonacci_sensitivity,
                "elliott_confidence_threshold": self.elliott_confidence_threshold,
                "order_flow_hvn_threshold": self.order_flow_hvn_threshold,
            },
            "timeframe_weights": self.timeframe_weights,
        }

    def _initialize_reporting_system(self) -> None:
        """Initialize the reporting system."""
        try:
            import os
            from pathlib import Path
            
            # Create report directory if it doesn't exist
            report_path = Path(self.report_directory)
            report_path.mkdir(parents=True, exist_ok=True)
            
            # Create subdirectories for different report types
            (report_path / "json").mkdir(exist_ok=True)
            (report_path / "csv").mkdir(exist_ok=True)
            (report_path / "html").mkdir(exist_ok=True)
            (report_path / "metrics").mkdir(exist_ok=True)
            
            self.logger.info(f"📊 Reporting system initialized: {report_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize reporting system: {e}")

    def _generate_report_id(self) -> str:
        """Generate a unique report ID."""
        from datetime import datetime
        import uuid
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        return f"sr_report_{timestamp}_{unique_id}"

    def _calculate_comprehensive_metrics(self, market_data: pd.DataFrame, sr_context: dict[str, Any]) -> dict[str, Any]:
        """Calculate comprehensive metrics for reporting."""
        try:
            current_price = sr_context.get("current_price", market_data["close"].iloc[-1])
            
            # Basic market metrics
            market_metrics = {
                "data_points": len(market_data),
                "price_range": {
                    "min": float(market_data["low"].min()),
                    "max": float(market_data["high"].max()),
                    "current": float(current_price),
                    "volatility": float(market_data["close"].pct_change().std()),
                },
                "volume_metrics": {
                    "total_volume": float(market_data["volume"].sum()),
                    "avg_volume": float(market_data["volume"].mean()),
                    "volume_std": float(market_data["volume"].std()),
                    "volume_trend": float(market_data["volume"].iloc[-10:].mean() / market_data["volume"].iloc[-20:-10].mean() if len(market_data) >= 20 else 1.0),
                },
                "price_metrics": {
                    "price_change_1h": float(market_data["close"].pct_change().iloc[-1]),
                    "price_change_24h": float(market_data["close"].pct_change(24).iloc[-1]) if len(market_data) >= 24 else 0.0,
                    "price_trend": float(market_data["close"].iloc[-10:].mean() / market_data["close"].iloc[-20:-10].mean() if len(market_data) >= 20 else 1.0),
                }
            }
            
            # S/R level metrics
            support_levels = sr_context.get("support_levels", [])
            resistance_levels = sr_context.get("resistance_levels", [])
            
            sr_metrics = {
                "total_levels": len(support_levels) + len(resistance_levels),
                "support_levels": {
                    "count": len(support_levels),
                    "avg_strength": np.mean([level.get("enhanced_strength", level.get("strength", 0.5)) for level in support_levels]) if support_levels else 0.0,
                    "avg_price": np.mean([level.get("price", 0) for level in support_levels]) if support_levels else 0.0,
                    "price_range": {
                        "min": min([level.get("price", 0) for level in support_levels]) if support_levels else 0.0,
                        "max": max([level.get("price", 0) for level in support_levels]) if support_levels else 0.0,
                    }
                },
                "resistance_levels": {
                    "count": len(resistance_levels),
                    "avg_strength": np.mean([level.get("enhanced_strength", level.get("strength", 0.5)) for level in resistance_levels]) if resistance_levels else 0.0,
                    "avg_price": np.mean([level.get("price", 0) for level in resistance_levels]) if resistance_levels else 0.0,
                    "price_range": {
                        "min": min([level.get("price", 0) for level in resistance_levels]) if resistance_levels else 0.0,
                        "max": max([level.get("price", 0) for level in resistance_levels]) if resistance_levels else 0.0,
                    }
                },
                "proximity_metrics": {
                    "support_proximity": sr_context.get("support_proximity", 0.0),
                    "resistance_proximity": sr_context.get("resistance_proximity", 0.0),
                    "sr_zone_width": sr_context.get("sr_zone_width", 0.0),
                },
                "strength_metrics": {
                    "support_strength": sr_context.get("support_strength", 0.5),
                    "resistance_strength": sr_context.get("resistance_strength", 0.5),
                }
            }
            
            # Clustering metrics
            clustering_result = sr_context.get("clustering_result", {})
            clustering_metrics = {
                "total_clusters": clustering_result.get("n_clusters", 0),
                "noise_points": clustering_result.get("noise_points", 0),
                "total_points": clustering_result.get("total_points", 0),
                "clustering_quality": clustering_result.get("clustering_quality", "unknown"),
                "cluster_statistics": clustering_result.get("cluster_statistics", {})
            }
            
            # Advanced analysis metrics
            fibonacci_levels = sr_context.get("fibonacci_levels", {})
            elliott_wave_levels = sr_context.get("elliott_wave_levels", {})
            order_flow_analysis = sr_context.get("order_flow_analysis", {})
            
            advanced_metrics = {
                "fibonacci_analysis": {
                    "levels_detected": len(fibonacci_levels),
                    "level_types": list(fibonacci_levels.keys()) if fibonacci_levels else [],
                },
                "elliott_wave_analysis": {
                    "waves_detected": len(elliott_wave_levels.get("wave_levels", {})),
                    "wave_types": list(elliott_wave_levels.get("wave_levels", {}).keys()) if elliott_wave_levels.get("wave_levels") else [],
                    "trend_direction": elliott_wave_levels.get("trend_direction", "unknown"),
                },
                "order_flow_analysis": {
                    "poc_detected": bool(order_flow_analysis.get("volume_profile", {}).get("poc")),
                    "hvns_detected": len(order_flow_analysis.get("volume_profile", {}).get("high_volume_nodes", [])),
                    "imbalances_detected": len(order_flow_analysis.get("imbalances", [])),
                    "value_area": order_flow_analysis.get("volume_profile", {}).get("value_area", {})
                }
            }
            
            # Performance metrics
            performance_metrics = {
                "analysis_timestamp": pd.Timestamp.now().isoformat(),
                "data_quality_score": self._calculate_data_quality_score(market_data),
                "sr_confidence_score": self._calculate_sr_confidence_score(sr_context),
                "overall_analysis_quality": self._calculate_overall_quality_score(market_metrics, sr_metrics, clustering_metrics, advanced_metrics)
            }
            
            return {
                "market_metrics": market_metrics,
                "sr_metrics": sr_metrics,
                "clustering_metrics": clustering_metrics,
                "advanced_metrics": advanced_metrics,
                "performance_metrics": performance_metrics
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating comprehensive metrics: {e}")
            return {}

    def _calculate_data_quality_score(self, market_data: pd.DataFrame) -> float:
        """Calculate data quality score (0-1)."""
        try:
            score = 1.0
            
            # Check for missing data
            missing_ratio = market_data.isnull().sum().sum() / (len(market_data) * len(market_data.columns))
            score -= missing_ratio * 0.3
            
            # Check for sufficient data points
            if len(market_data) < 50:
                score -= 0.2
            elif len(market_data) < 100:
                score -= 0.1
            
            # Check for price anomalies
            price_changes = market_data["close"].pct_change().abs()
            anomaly_ratio = (price_changes > 0.1).sum() / len(price_changes)
            score -= anomaly_ratio * 0.2
            
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            self.logger.error(f"Error calculating data quality score: {e}")
            return 0.5

    def _calculate_sr_confidence_score(self, sr_context: dict[str, Any]) -> float:
        """Calculate S/R confidence score (0-1)."""
        try:
            score = 0.5  # Base score
            
            # Factor in number of levels
            total_levels = len(sr_context.get("support_levels", [])) + len(sr_context.get("resistance_levels", []))
            if total_levels >= 5:
                score += 0.2
            elif total_levels >= 3:
                score += 0.1
            
            # Factor in strength
            avg_strength = (sr_context.get("support_strength", 0.5) + sr_context.get("resistance_strength", 0.5)) / 2
            score += avg_strength * 0.2
            
            # Factor in clustering quality
            clustering_result = sr_context.get("clustering_result", {})
            if clustering_result.get("n_clusters", 0) > 0:
                score += 0.1
            
            return min(1.0, score)
            
        except Exception as e:
            self.logger.error(f"Error calculating SR confidence score: {e}")
            return 0.5

    def _calculate_overall_quality_score(self, market_metrics: dict, sr_metrics: dict, clustering_metrics: dict, advanced_metrics: dict) -> float:
        """Calculate overall analysis quality score (0-1)."""
        try:
            score = 0.5  # Base score
            
            # Market data quality
            if market_metrics.get("data_points", 0) >= 100:
                score += 0.1
            
            # S/R analysis quality
            if sr_metrics.get("total_levels", 0) >= 3:
                score += 0.1
            
            # Clustering quality
            if clustering_metrics.get("total_clusters", 0) > 0:
                score += 0.1
            
            # Advanced analysis quality
            if advanced_metrics.get("fibonacci_analysis", {}).get("levels_detected", 0) > 0:
                score += 0.1
            if advanced_metrics.get("elliott_wave_analysis", {}).get("waves_detected", 0) > 0:
                score += 0.1
            
            return min(1.0, score)
            
        except Exception as e:
            self.logger.error(f"Error calculating overall quality score: {e}")
            return 0.5

    async def _generate_detailed_report(self, market_data: pd.DataFrame, sr_context: dict[str, Any]) -> dict[str, Any]:
        """Generate detailed metrics report."""
        try:
            if not self.reporting_enabled:
                return {}
            
            # Generate report ID
            self.current_report_id = self._generate_report_id()
            
            # Calculate comprehensive metrics
            metrics = self._calculate_comprehensive_metrics(market_data, sr_context)
            
            # Create detailed report
            report = {
                "report_id": self.current_report_id,
                "report_timestamp": pd.Timestamp.now().isoformat(),
                "report_version": "1.0",
                "configuration": {
                    "sr_detection_method": self.sr_detection_method,
                    "sr_proximity_threshold": self.sr_proximity_threshold,
                    "breakout_confidence_threshold": self.breakout_confidence_threshold,
                    "min_sr_strength": self.min_sr_strength,
                    "max_sr_levels": self.max_sr_levels,
                    "enable_dbscan_clustering": DBSCAN_AVAILABLE,
                },
                "metrics": metrics,
                "sr_context_summary": {
                    "current_price": sr_context.get("current_price", 0.0),
                    "nearest_support": sr_context.get("nearest_support", 0.0),
                    "nearest_resistance": sr_context.get("nearest_resistance", 0.0),
                    "support_strength": sr_context.get("support_strength", 0.5),
                    "resistance_strength": sr_context.get("resistance_strength", 0.5),
                    "sr_zone_width": sr_context.get("sr_zone_width", 0.0),
                },
                "analysis_summary": {
                    "total_support_levels": len(sr_context.get("support_levels", [])),
                    "total_resistance_levels": len(sr_context.get("resistance_levels", [])),
                    "clusters_detected": sr_context.get("clustering_result", {}).get("n_clusters", 0),
                    "fibonacci_levels": len(sr_context.get("fibonacci_levels", {})),
                    "elliott_waves": len(sr_context.get("elliott_wave_levels", {}).get("wave_levels", {})),
                    "order_flow_imbalances": len(sr_context.get("order_flow_analysis", {}).get("imbalances", [])),
                }
            }
            
            # Store in history
            self.metrics_history.append(report)
            
            # Limit history size
            if len(self.metrics_history) > 100:
                self.metrics_history = self.metrics_history[-100:]
            
            # Save report to file
            await self._save_report_to_file(report)
            
            self.logger.info(f"📊 Detailed metrics report generated: {self.current_report_id}")
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating detailed report: {e}")
            return {}

    async def _save_report_to_file(self, report: dict[str, Any]) -> None:
        """Save report to file in specified format."""
        try:
            import os
            from pathlib import Path
            import json
            
            report_path = Path(self.report_directory)
            
            # Save JSON report
            json_file = report_path / "json" / f"{self.current_report_id}.json"
            with open(json_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            # Save CSV metrics
            csv_file = report_path / "csv" / f"{self.current_report_id}_metrics.csv"
            self._save_metrics_to_csv(report["metrics"], csv_file)
            
            # Save HTML report
            html_file = report_path / "html" / f"{self.current_report_id}.html"
            self._save_html_report(report, html_file)
            
            # Save latest metrics summary
            summary_file = report_path / "metrics" / "latest_metrics.json"
            with open(summary_file, 'w') as f:
                json.dump({
                    "last_report_id": self.current_report_id,
                    "last_report_timestamp": report["report_timestamp"],
                    "summary": report["analysis_summary"],
                    "quality_scores": {
                        "data_quality": report["metrics"]["performance_metrics"]["data_quality_score"],
                        "sr_confidence": report["metrics"]["performance_metrics"]["sr_confidence_score"],
                        "overall_quality": report["metrics"]["performance_metrics"]["overall_analysis_quality"]
                    }
                }, f, indent=2, default=str)
            
            self.logger.info(f"📁 Report saved: {self.current_report_id}")
            
        except Exception as e:
            self.logger.error(f"Error saving report to file: {e}")

    def _save_metrics_to_csv(self, metrics: dict[str, Any], file_path: Path) -> None:
        """Save metrics to CSV format."""
        try:
            import csv
            
            # Flatten metrics for CSV
            csv_data = []
            
            # Market metrics
            market_metrics = metrics.get("market_metrics", {})
            csv_data.append(["Category", "Metric", "Value"])
            csv_data.append(["Market", "Data Points", market_metrics.get("data_points", 0)])
            csv_data.append(["Market", "Current Price", market_metrics.get("price_range", {}).get("current", 0)])
            csv_data.append(["Market", "Volatility", market_metrics.get("price_range", {}).get("volatility", 0)])
            csv_data.append(["Market", "Total Volume", market_metrics.get("volume_metrics", {}).get("total_volume", 0)])
            
            # S/R metrics
            sr_metrics = metrics.get("sr_metrics", {})
            csv_data.append(["S/R", "Total Levels", sr_metrics.get("total_levels", 0)])
            csv_data.append(["S/R", "Support Levels", sr_metrics.get("support_levels", {}).get("count", 0)])
            csv_data.append(["S/R", "Resistance Levels", sr_metrics.get("resistance_levels", {}).get("count", 0)])
            csv_data.append(["S/R", "SR Zone Width", sr_metrics.get("proximity_metrics", {}).get("sr_zone_width", 0)])
            
            # Clustering metrics
            clustering_metrics = metrics.get("clustering_metrics", {})
            csv_data.append(["Clustering", "Total Clusters", clustering_metrics.get("total_clusters", 0)])
            csv_data.append(["Clustering", "Noise Points", clustering_metrics.get("noise_points", 0)])
            
            # Performance metrics
            performance_metrics = metrics.get("performance_metrics", {})
            csv_data.append(["Performance", "Data Quality Score", performance_metrics.get("data_quality_score", 0)])
            csv_data.append(["Performance", "SR Confidence Score", performance_metrics.get("sr_confidence_score", 0)])
            csv_data.append(["Performance", "Overall Quality Score", performance_metrics.get("overall_analysis_quality", 0)])
            
            with open(file_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(csv_data)
                
        except Exception as e:
            self.logger.error(f"Error saving metrics to CSV: {e}")

    def _save_html_report(self, report: dict[str, Any], file_path: Path) -> None:
        """Save HTML report."""
        try:
            html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>S/R Analysis Report - {report['report_id']}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
        .metric {{ margin: 10px 0; }}
        .score {{ font-weight: bold; color: #007bff; }}
        .quality-high {{ color: #28a745; }}
        .quality-medium {{ color: #ffc107; }}
        .quality-low {{ color: #dc3545; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>S/R Analysis Report</h1>
        <p><strong>Report ID:</strong> {report['report_id']}</p>
        <p><strong>Timestamp:</strong> {report['report_timestamp']}</p>
    </div>
    
    <div class="section">
        <h2>Analysis Summary</h2>
        <div class="metric"><strong>Support Levels:</strong> {report['analysis_summary']['total_support_levels']}</div>
        <div class="metric"><strong>Resistance Levels:</strong> {report['analysis_summary']['total_resistance_levels']}</div>
        <div class="metric"><strong>Clusters Detected:</strong> {report['analysis_summary']['clusters_detected']}</div>
        <div class="metric"><strong>Fibonacci Levels:</strong> {report['analysis_summary']['fibonacci_levels']}</div>
        <div class="metric"><strong>Elliott Waves:</strong> {report['analysis_summary']['elliott_waves']}</div>
    </div>
    
    <div class="section">
        <h2>Quality Scores</h2>
        <div class="metric">
            <strong>Data Quality:</strong> 
            <span class="score quality-{'high' if report['metrics']['performance_metrics']['data_quality_score'] > 0.7 else 'medium' if report['metrics']['performance_metrics']['data_quality_score'] > 0.4 else 'low'}">
                {report['metrics']['performance_metrics']['data_quality_score']:.3f}
            </span>
        </div>
        <div class="metric">
            <strong>SR Confidence:</strong> 
            <span class="score quality-{'high' if report['metrics']['performance_metrics']['sr_confidence_score'] > 0.7 else 'medium' if report['metrics']['performance_metrics']['sr_confidence_score'] > 0.4 else 'low'}">
                {report['metrics']['performance_metrics']['sr_confidence_score']:.3f}
            </span>
        </div>
        <div class="metric">
            <strong>Overall Quality:</strong> 
            <span class="score quality-{'high' if report['metrics']['performance_metrics']['overall_analysis_quality'] > 0.7 else 'medium' if report['metrics']['performance_metrics']['overall_analysis_quality'] > 0.4 else 'low'}">
                {report['metrics']['performance_metrics']['overall_analysis_quality']:.3f}
            </span>
        </div>
    </div>
    
    <div class="section">
        <h2>Market Metrics</h2>
        <div class="metric"><strong>Data Points:</strong> {report['metrics']['market_metrics']['data_points']}</div>
        <div class="metric"><strong>Current Price:</strong> {report['metrics']['market_metrics']['price_range']['current']:.4f}</div>
        <div class="metric"><strong>Volatility:</strong> {report['metrics']['market_metrics']['price_range']['volatility']:.4f}</div>
        <div class="metric"><strong>Total Volume:</strong> {report['metrics']['market_metrics']['volume_metrics']['total_volume']:.0f}</div>
    </div>
    
    <div class="section">
        <h2>S/R Context</h2>
        <div class="metric"><strong>Current Price:</strong> {report['sr_context_summary']['current_price']:.4f}</div>
        <div class="metric"><strong>Nearest Support:</strong> {report['sr_context_summary']['nearest_support']:.4f}</div>
        <div class="metric"><strong>Nearest Resistance:</strong> {report['sr_context_summary']['nearest_resistance']:.4f}</div>
        <div class="metric"><strong>SR Zone Width:</strong> {report['sr_context_summary']['sr_zone_width']:.4f}</div>
    </div>
</body>
</html>
            """
            
            with open(file_path, 'w') as f:
                f.write(html_content)
                
        except Exception as e:
            self.logger.error(f"Error saving HTML report: {e}")

    async def get_latest_report(self) -> dict[str, Any]:
        """Get the latest generated report."""
        if self.metrics_history:
            return self.metrics_history[-1]
        return {}

    async def get_report_history(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get recent report history."""
        return self.metrics_history[-limit:] if self.metrics_history else []

    async def cleanup_old_reports(self) -> None:
        """Clean up old reports based on retention policy."""
        try:
            import os
            from pathlib import Path
            from datetime import datetime, timedelta
            
            if not self.reporting_enabled:
                return
            
            report_path = Path(self.report_directory)
            cutoff_date = datetime.now() - timedelta(days=self.report_retention_days)
            
            for subdir in ["json", "csv", "html"]:
                subdir_path = report_path / subdir
                if subdir_path.exists():
                    for file_path in subdir_path.iterdir():
                        if file_path.is_file():
                            file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                            if file_time < cutoff_date:
                                file_path.unlink()
                                self.logger.info(f"Cleaned up old report: {file_path}")
            
            self.logger.info("🧹 Old reports cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error cleaning up old reports: {e}")

    async def generate_manual_report(self, market_data: pd.DataFrame, sr_context: dict[str, Any] = None) -> dict[str, Any]:
        """Manually generate a detailed report."""
        try:
            if not self.reporting_enabled:
                self.logger.warning("Reporting is disabled. Enable it in configuration to generate reports.")
                return {}
            
            if sr_context is None:
                # Generate SR context if not provided
                current_price = market_data["close"].iloc[-1]
                sr_context = await self.get_sr_context(market_data, current_price)
            
            report = await self._generate_detailed_report(market_data, sr_context)
            self.logger.info(f"📊 Manual report generated: {self.current_report_id}")
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating manual report: {e}")
            return {}

    def get_reporting_status(self) -> dict[str, Any]:
        """Get reporting system status."""
        return {
            "reporting_enabled": self.reporting_enabled,
            "report_directory": self.report_directory,
            "report_format": self.report_format,
            "report_retention_days": self.report_retention_days,
            "total_reports_generated": len(self.metrics_history),
            "current_report_id": self.current_report_id,
            "last_report_timestamp": self.metrics_history[-1]["report_timestamp"] if self.metrics_history else None
        }

    @validate_data_quality(
        required_columns=["open", "high", "low", "close", "volume"],
        min_rows=50,
        max_null_ratio=0.1,
        check_duplicates=True,
        check_timestamps=True,
        context="SR breakout prediction input validation"
    )
    @handle_specific_errors(
        error_handlers={
            ValueError: (None, "Invalid input data for SR breakout prediction"),
            AttributeError: (None, "Predictor not properly initialized"),
        },
        default_return=None,
        context="SR breakout prediction",
    )
    async def predict_sr_breakouts(
        self,
        market_data: pd.DataFrame,
        current_price: float,
    ) -> dict[str, Any]:
        """
        Predict support/resistance breakouts.

        Args:
            market_data: Market data DataFrame
            current_price: Current market price

        Returns:
            dict[str, Any]: SR breakout predictions
        """
        if not self.is_initialized:
            self.logger.error("SR breakout predictor not initialized")
            return {}

        try:
            self.logger.info("Predicting SR breakouts...")

            # Detect support and resistance levels
            support_levels = await self._detect_support_levels(market_data)
            resistance_levels = await self._detect_resistance_levels(market_data)

            # Calculate breakout probabilities
            breakout_probabilities = await self._calculate_breakout_probabilities(
                support_levels, resistance_levels, current_price,
            )

            # Calculate confidence scores
            confidence_scores = await self._calculate_confidence_scores(
                support_levels, resistance_levels, market_data,
            )

            # Generate SR features
            sr_features = await self._generate_sr_features(
                support_levels, resistance_levels, market_data,
            )

            # Create predictions
            predictions = {
                "support_levels": support_levels,
                "resistance_levels": resistance_levels,
                "breakout_probabilities": breakout_probabilities,
                "confidence_scores": confidence_scores,
                "sr_features": sr_features,
                "current_price": current_price,
                "timestamp": pd.Timestamp.now(),
            }
            
            # Generate detailed report for predictions
            if self.reporting_enabled:
                await self._generate_detailed_report(market_data, predictions)

            # Store predictions
            self.sr_predictions = predictions

            # Update performance metrics
            self._update_performance_metrics(predictions)

            self.logger.info("✅ SR breakout predictions generated")
            return predictions

        except Exception as e:
            self.logger.error(f"Error predicting SR breakouts: {e}")
            return {}

    @validate_data_quality(
        required_columns=["open", "high", "low", "close", "volume"],
        min_rows=50,
        max_null_ratio=0.1,
        check_duplicates=True,
        check_timestamps=True,
        context="SR context calculation input validation"
    )
    @handle_specific_errors(
        error_handlers={
            ValueError: (None, "Invalid input data for SR context calculation"),
            AttributeError: (None, "Predictor not properly initialized"),
        },
        default_return={},
        context="SR context calculation",
    )
    async def get_sr_context(
        self,
        market_data: pd.DataFrame,
        current_price: float,
    ) -> dict[str, Any]:
        """
        Get comprehensive S/R context for current market position.

        Args:
            market_data: Market data DataFrame
            current_price: Current market price

        Returns:
            dict[str, Any]: S/R context information
        """
        if not self.is_initialized:
            self.logger.error("SR breakout predictor not initialized")
            return {}

        try:
            # Detect support and resistance levels
            support_levels = await self._detect_support_levels(market_data)
            resistance_levels = await self._detect_resistance_levels(market_data)

            # Apply DBSCAN clustering to filter significant levels
            all_levels = support_levels + resistance_levels
            clustering_result = await self.cluster_sr_levels_dbscan(all_levels)
            clustered_levels = clustering_result.get('clustered_levels', all_levels)
            
            # Separate clustered levels back into support and resistance
            clustered_support = [level for level in clustered_levels if level.get('type', 'support') == 'support']
            clustered_resistance = [level for level in clustered_levels if level.get('type', 'resistance') == 'resistance']

            # Calculate enhanced strength for all levels
            enhanced_strength_support = await self.calculate_comprehensive_strength(market_data, clustered_support)
            enhanced_strength_resistance = await self.calculate_comprehensive_strength(market_data, clustered_resistance)

            # Update levels with enhanced strength
            for level in clustered_support:
                level_id = f"{level['price']:.4f}"
                if level_id in enhanced_strength_support:
                    level['enhanced_strength'] = enhanced_strength_support[level_id]['comprehensive_strength']
                    level['strength_factors'] = enhanced_strength_support[level_id]['factors']
                else:
                    level['enhanced_strength'] = level.get('strength', 0.5)
                    level['strength_factors'] = {}

            for level in clustered_resistance:
                level_id = f"{level['price']:.4f}"
                if level_id in enhanced_strength_resistance:
                    level['enhanced_strength'] = enhanced_strength_resistance[level_id]['comprehensive_strength']
                    level['strength_factors'] = enhanced_strength_resistance[level_id]['factors']
                else:
                    level['enhanced_strength'] = level.get('strength', 0.5)
                    level['strength_factors'] = {}

            # Find nearest levels using enhanced strength
            nearest_support = self._find_nearest_level(current_price, clustered_support, "support")
            nearest_resistance = self._find_nearest_level(current_price, clustered_resistance, "resistance")

            # Calculate proximity metrics
            support_proximity = self._calculate_proximity(current_price, nearest_support)
            resistance_proximity = self._calculate_proximity(current_price, nearest_resistance)

            # Get pivot levels
            pivot_levels = self._calculate_pivot_levels(market_data)

            # Get advanced S/R analysis
            fibonacci_levels = await self.calculate_fibonacci_levels(market_data)
            elliott_wave_levels = await self.detect_elliott_wave_levels(market_data)
            order_flow_analysis = await self.analyze_order_flow_levels(market_data)

            # Create context
            context = {
                "current_price": current_price,
                "nearest_support": nearest_support.get("price", current_price) if nearest_support else current_price,
                "nearest_resistance": nearest_resistance.get("price", current_price) if nearest_resistance else current_price,
                "support_strength": nearest_support.get("enhanced_strength", nearest_support.get("strength", 0.5)) if nearest_support else 0.5,
                "resistance_strength": nearest_resistance.get("enhanced_strength", nearest_resistance.get("strength", 0.5)) if nearest_resistance else 0.5,
                "support_proximity": support_proximity,
                "resistance_proximity": resistance_proximity,
                "pivot_levels": pivot_levels,
                "support_levels": clustered_support,  # Use clustered levels
                "resistance_levels": clustered_resistance,  # Use clustered levels
                "sr_zone_width": abs(nearest_resistance.get("price", current_price) - nearest_support.get("price", current_price)) / current_price if nearest_resistance and nearest_support else 0.0,
                
                # Enhanced Strength Analysis
                "enhanced_strength_support": enhanced_strength_support,
                "enhanced_strength_resistance": enhanced_strength_resistance,
                
                # DBSCAN Clustering Results
                "clustering_result": clustering_result,
                
                # Advanced S/R Analysis
                "fibonacci_levels": fibonacci_levels,
                "elliott_wave_levels": elliott_wave_levels,
                "order_flow_analysis": order_flow_analysis,
                
                # Generate detailed report
                "report_id": await self._generate_detailed_report(market_data, context),
                
                "timestamp": pd.Timestamp.now(),
            }

            return context

        except Exception as e:
            self.logger.error(f"Error getting S/R context: {e}")
            return {}

    async def _detect_support_levels(self, market_data: pd.DataFrame) -> list[dict[str, Any]]:
        """Detect support levels using configured method."""
        try:
            if self.sr_detection_method == "fractal":
                return await self._detect_fractal_support_levels(market_data)
            elif self.sr_detection_method == "volume":
                return await self._detect_volume_support_levels(market_data)
            elif self.sr_detection_method == "pivot":
                return await self._detect_pivot_support_levels(market_data)
            elif self.sr_detection_method == "atr":
                return await self._detect_atr_support_levels(market_data)
            else:
                self.logger.warning(f"Unknown SR detection method: {self.sr_detection_method}")
                return await self._detect_fractal_support_levels(market_data)

        except Exception as e:
            self.logger.error(f"Error detecting support levels: {e}")
            return []

    async def _detect_resistance_levels(self, market_data: pd.DataFrame) -> list[dict[str, Any]]:
        """Detect resistance levels using configured method."""
        try:
            if self.sr_detection_method == "fractal":
                return await self._detect_fractal_resistance_levels(market_data)
            elif self.sr_detection_method == "volume":
                return await self._detect_volume_resistance_levels(market_data)
            elif self.sr_detection_method == "pivot":
                return await self._detect_pivot_resistance_levels(market_data)
            elif self.sr_detection_method == "atr":
                return await self._detect_atr_resistance_levels(market_data)
            else:
                self.logger.warning(f"Unknown SR detection method: {self.sr_detection_method}")
                return await self._detect_fractal_resistance_levels(market_data)

        except Exception as e:
            self.logger.error(f"Error detecting resistance levels: {e}")
            return []

    async def _detect_fractal_support_levels(self, market_data: pd.DataFrame) -> list[dict[str, Any]]:
        """Detect support levels using fractal analysis."""
        try:
            # Implement fractal-based support level detection
            # This is a simplified implementation
            support_levels = []

            # Find local minima in price data
            low_prices = market_data['low'].rolling(window=5, center=True).min()

            # Identify significant support levels
            for i in range(2, len(market_data) - 2):
                if (market_data['low'].iloc[i] == low_prices.iloc[i] and
                    market_data['low'].iloc[i] < market_data['low'].iloc[i-1] and
                    market_data['low'].iloc[i] < market_data['low'].iloc[i+1]):

                    support_level = {
                        "price": market_data['low'].iloc[i],
                        "strength": self._calculate_level_strength(market_data, i, "support"),
                        "timestamp": market_data.index[i],
                        "method": "fractal",
                        "confidence": 0.7,
                    }
                    support_levels.append(support_level)

            return support_levels[:self.max_sr_levels]

        except Exception as e:
            self.logger.error(f"Error in fractal support detection: {e}")
            return []

    async def _detect_fractal_resistance_levels(self, market_data: pd.DataFrame) -> list[dict[str, Any]]:
        """Detect resistance levels using fractal analysis."""
        try:
            # Implement fractal-based resistance level detection
            resistance_levels = []

            # Find local maxima in price data
            high_prices = market_data['high'].rolling(window=5, center=True).max()

            # Identify significant resistance levels
            for i in range(2, len(market_data) - 2):
                if (market_data['high'].iloc[i] == high_prices.iloc[i] and
                    market_data['high'].iloc[i] > market_data['high'].iloc[i-1] and
                    market_data['high'].iloc[i] > market_data['high'].iloc[i+1]):

                    resistance_level = {
                        "price": market_data['high'].iloc[i],
                        "strength": self._calculate_level_strength(market_data, i, "resistance"),
                        "timestamp": market_data.index[i],
                        "method": "fractal",
                        "confidence": 0.7,
                    }
                    resistance_levels.append(resistance_level)

            return resistance_levels[:self.max_sr_levels]

        except Exception as e:
            self.logger.error(f"Error in fractal resistance detection: {e}")
            return []

    async def _detect_volume_support_levels(self, market_data: pd.DataFrame) -> list[dict[str, Any]]:
        """Detect support levels using volume-weighted analysis."""
        try:
            # Implement volume-weighted support level detection
            support_levels = []

            # Calculate volume-weighted average price
            vwap = (market_data['close'] * market_data['volume']).cumsum() / market_data['volume'].cumsum()

            # Find support levels near VWAP
            for i in range(len(market_data)):
                if market_data['low'].iloc[i] <= vwap.iloc[i] * 1.01:  # Within 1% of VWAP
                    support_level = {
                        "price": market_data['low'].iloc[i],
                        "strength": self._calculate_level_strength(market_data, i, "support"),
                        "timestamp": market_data.index[i],
                        "method": "volume",
                        "confidence": 0.6,
                    }
                    support_levels.append(support_level)

            return support_levels[:self.max_sr_levels]

        except Exception as e:
            self.logger.error(f"Error in volume support detection: {e}")
            return []

    async def _detect_volume_resistance_levels(self, market_data: pd.DataFrame) -> list[dict[str, Any]]:
        """Detect resistance levels using volume-weighted analysis."""
        try:
            # Implement volume-weighted resistance level detection
            resistance_levels = []

            # Calculate volume-weighted average price
            vwap = (market_data['close'] * market_data['volume']).cumsum() / market_data['volume'].cumsum()

            # Find resistance levels near VWAP
            for i in range(len(market_data)):
                if market_data['high'].iloc[i] >= vwap.iloc[i] * 0.99:  # Within 1% of VWAP
                    resistance_level = {
                        "price": market_data['high'].iloc[i],
                        "strength": self._calculate_level_strength(market_data, i, "resistance"),
                        "timestamp": market_data.index[i],
                        "method": "volume",
                        "confidence": 0.6,
                    }
                    resistance_levels.append(resistance_level)

            return resistance_levels[:self.max_sr_levels]

        except Exception as e:
            self.logger.error(f"Error in volume resistance detection: {e}")
            return []

    async def _detect_pivot_support_levels(self, market_data: pd.DataFrame) -> list[dict[str, Any]]:
        """Detect support levels using pivot point analysis."""
        try:
            # Implement pivot point support level detection
            support_levels = []

            # Calculate pivot points
            pivot = (market_data['high'] + market_data['low'] + market_data['close']) / 3
            s1 = 2 * pivot - market_data['high']
            s2 = pivot - (market_data['high'] - market_data['low'])

            # Find support levels
            for i in range(len(market_data)):
                support_level = {
                    "price": s1.iloc[i],
                    "strength": self._calculate_level_strength(market_data, i, "support"),
                    "timestamp": market_data.index[i],
                    "method": "pivot",
                    "confidence": 0.5,
                }
                support_levels.append(support_level)

            return support_levels[:self.max_sr_levels]

        except Exception as e:
            self.logger.error(f"Error in pivot support detection: {e}")
            return []

    async def _detect_pivot_resistance_levels(self, market_data: pd.DataFrame) -> list[dict[str, Any]]:
        """Detect resistance levels using pivot point analysis."""
        try:
            # Implement pivot point resistance level detection
            resistance_levels = []

            # Calculate pivot points
            pivot = (market_data['high'] + market_data['low'] + market_data['close']) / 3
            r1 = 2 * pivot - market_data['low']
            r2 = pivot + (market_data['high'] - market_data['low'])

            # Find resistance levels
            for i in range(len(market_data)):
                resistance_level = {
                    "price": r1.iloc[i],
                    "strength": self._calculate_level_strength(market_data, i, "resistance"),
                    "timestamp": market_data.index[i],
                    "method": "pivot",
                    "confidence": 0.5,
                }
                resistance_levels.append(resistance_level)

            return resistance_levels[:self.max_sr_levels]

        except Exception as e:
            self.logger.error(f"Error in pivot resistance detection: {e}")
            return []

    async def _detect_atr_support_levels(self, market_data: pd.DataFrame) -> list[dict[str, Any]]:
        """Detect support levels using ATR-based analysis."""
        try:
            # Implement ATR-based support level detection
            support_levels = []

            # Calculate ATR
            high_low = market_data['high'] - market_data['low']
            high_close = np.abs(market_data['high'] - market_data['close'].shift())
            low_close = np.abs(market_data['low'] - market_data['close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            atr = true_range.rolling(window=14).mean()

            # Find support levels
            for i in range(len(market_data)):
                support_level = {
                    "price": market_data['close'].iloc[i] - (atr.iloc[i] * self.atr_multiplier),
                    "strength": self._calculate_level_strength(market_data, i, "support"),
                    "timestamp": market_data.index[i],
                    "method": "atr",
                    "confidence": 0.4,
                }
                support_levels.append(support_level)

            return support_levels[:self.max_sr_levels]

        except Exception as e:
            self.logger.error(f"Error in ATR support detection: {e}")
            return []

    async def _detect_atr_resistance_levels(self, market_data: pd.DataFrame) -> list[dict[str, Any]]:
        """Detect resistance levels using ATR-based analysis."""
        try:
            # Implement ATR-based resistance level detection
            resistance_levels = []

            # Calculate ATR
            high_low = market_data['high'] - market_data['low']
            high_close = np.abs(market_data['high'] - market_data['close'].shift())
            low_close = np.abs(market_data['low'] - market_data['close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            atr = true_range.rolling(window=14).mean()

            # Find resistance levels
            for i in range(len(market_data)):
                resistance_level = {
                    "price": market_data['close'].iloc[i] + (atr.iloc[i] * self.atr_multiplier),
                    "strength": self._calculate_level_strength(market_data, i, "resistance"),
                    "timestamp": market_data.index[i],
                    "method": "atr",
                    "confidence": 0.4,
                }
                resistance_levels.append(resistance_level)

            return resistance_levels[:self.max_sr_levels]

        except Exception as e:
            self.logger.error(f"Error in ATR resistance detection: {e}")
            return []

    # ============================================================================
    # ADVANCED S/R DETECTION METHODS
    # ============================================================================

    @validate_data_quality(validation_level="WARNING")
    async def calculate_fibonacci_levels(self, market_data: pd.DataFrame) -> dict[str, float]:
        """Calculate Fibonacci retracement and extension levels using optimized sensitivity."""
        try:
            # Find swing high and low
            high = market_data['high'].max()
            low = market_data['low'].min()
            swing_range = high - low
            
            # Apply optimized sensitivity to filter levels
            sensitivity_threshold = swing_range * (1 - self.fibonacci_sensitivity)
            
            # Calculate Fibonacci levels with sensitivity filtering
            fib_levels = {}
            
            # Standard retracement levels
            retracement_levels = [0, 0.236, 0.382, 0.500, 0.618, 0.786, 1.0]
            for level in retracement_levels:
                fib_price = low + level * swing_range
                # Only include levels that meet sensitivity threshold
                if abs(fib_price - low) >= sensitivity_threshold or abs(fib_price - high) >= sensitivity_threshold:
                    fib_levels[f'fib_{int(level * 1000)}'] = fib_price
            
            # Extension levels (only if sensitivity allows)
            if self.fibonacci_sensitivity > 0.6:  # Only include extensions for higher sensitivity
                extension_levels = [1.272, 1.618, 2.618]
                for level in extension_levels:
                    fib_price = high + (level - 1) * swing_range
                    fib_levels[f'fib_{int(level * 1000)}'] = fib_price
            
            self.logger.info(f"✅ Calculated Fibonacci levels with sensitivity {self.fibonacci_sensitivity}: {len(fib_levels)} levels")
            return fib_levels
            
        except Exception as e:
            self.logger.error(f"Error calculating Fibonacci levels: {e}")
            return {}

    @validate_data_quality(validation_level="WARNING")
    async def detect_elliott_wave_levels(self, market_data: pd.DataFrame) -> dict[str, Any]:
        """Detect Elliott Wave patterns and associated S/R levels."""
        try:
            # Simple Elliott Wave detection (can be enhanced with more sophisticated algorithms)
            prices = market_data['close'].values
            highs = market_data['high'].values
            lows = market_data['low'].values
            
            # Find potential wave points
            wave_points = self._find_elliott_wave_points(prices, highs, lows)
            
            if len(wave_points) >= 5:
                # Calculate wave levels
                wave1_high = wave_points[1]['high']
                wave1_low = wave_points[0]['low']
                wave2_retracement = wave_points[2]['low']
                wave3_target = wave2_retracement + 1.618 * (wave1_high - wave1_low)
                wave4_retracement = wave_points[4]['low']
                wave5_target = wave4_retracement + 0.618 * (wave1_high - wave1_low)
                
                # Calculate confidence based on pattern quality and optimized threshold
                pattern_confidence = self._calculate_elliott_pattern_confidence(wave_points)
                
                elliott_levels = {
                    'wave1': {'high': wave1_high, 'low': wave1_low},
                    'wave2_retracement': wave2_retracement,
                    'wave3_target': wave3_target,
                    'wave4_retracement': wave4_retracement,
                    'wave5_target': wave5_target,
                    'pattern_type': 'impulse',
                    'confidence': pattern_confidence
                }
                
                # Only return high-confidence patterns based on optimized threshold
                if pattern_confidence >= self.elliott_confidence_threshold:
                    self.logger.info(f"✅ Detected Elliott Wave pattern with confidence {pattern_confidence:.3f} (threshold: {self.elliott_confidence_threshold})")
                else:
                    self.logger.info(f"⚠️ Elliott Wave pattern confidence {pattern_confidence:.3f} below threshold {self.elliott_confidence_threshold}")
            else:
                elliott_levels = {
                    'pattern_type': 'incomplete',
                    'confidence': 0.3
                }
            
            self.logger.info(f"✅ Detected Elliott Wave pattern: {elliott_levels.get('pattern_type', 'unknown')}")
            return elliott_levels
            
        except Exception as e:
            self.logger.error(f"Error detecting Elliott Wave levels: {e}")
            return {'pattern_type': 'error', 'confidence': 0.0}

    def _find_elliott_wave_points(self, prices: np.ndarray, highs: np.ndarray, lows: np.ndarray) -> list[dict[str, Any]]:
        """Find potential Elliott Wave points in price data."""
        wave_points = []
        
        # Simple peak and trough detection
        for i in range(2, len(prices) - 2):
            # Peak detection
            if highs[i] > highs[i-1] and highs[i] > highs[i-2] and highs[i] > highs[i+1] and highs[i] > highs[i+2]:
                wave_points.append({
                    'index': i,
                    'type': 'peak',
                    'high': highs[i],
                    'low': lows[i]
                })
            # Trough detection
            elif lows[i] < lows[i-1] and lows[i] < lows[i-2] and lows[i] < lows[i+1] and lows[i] < lows[i+2]:
                wave_points.append({
                    'index': i,
                    'type': 'trough',
                    'high': highs[i],
                    'low': lows[i]
                })
        
        return wave_points[:10]  # Limit to first 10 points

    def _calculate_elliott_pattern_confidence(self, wave_points: list[dict[str, Any]]) -> float:
        """Calculate confidence score for Elliott Wave pattern."""
        try:
            if len(wave_points) < 5:
                return 0.3
            
            # Calculate confidence based on wave relationships
            confidence_factors = []
            
            # Wave 2 should retrace 50-78.6% of wave 1
            wave1_range = wave_points[1]['high'] - wave_points[0]['low']
            wave2_retracement = (wave_points[1]['high'] - wave_points[2]['low']) / wave1_range
            if 0.5 <= wave2_retracement <= 0.786:
                confidence_factors.append(1.0)
            else:
                confidence_factors.append(0.5)
            
            # Wave 3 should be the longest (1.618x wave 1)
            wave3_range = wave_points[3]['high'] - wave_points[2]['low']
            wave3_ratio = wave3_range / wave1_range
            if wave3_ratio >= 1.618:
                confidence_factors.append(1.0)
            else:
                confidence_factors.append(0.7)
            
            # Wave 4 should retrace 23.6-38.2% of wave 3
            wave4_retracement = (wave_points[3]['high'] - wave_points[4]['low']) / wave3_range
            if 0.236 <= wave4_retracement <= 0.382:
                confidence_factors.append(1.0)
            else:
                confidence_factors.append(0.6)
            
            # Calculate average confidence
            return np.mean(confidence_factors) if confidence_factors else 0.3
            
        except Exception as e:
            self.logger.error(f"Error calculating Elliott pattern confidence: {e}")
            return 0.3

    @validate_data_quality(validation_level="WARNING")
    async def analyze_order_flow_levels(self, market_data: pd.DataFrame) -> dict[str, Any]:
        """Analyze order flow to identify institutional S/R levels (POC, HVN, etc.)."""
        try:
            # Volume Profile Analysis
            volume_profile = await self._calculate_volume_profile(market_data)
            
            # Point of Control (POC) - price level with highest volume
            poc_level = volume_profile['poc']
            
            # Value Area (70% of volume)
            value_area_high = volume_profile['value_area_high']
            value_area_low = volume_profile['value_area_low']
            
            # High Volume Nodes (HVN) - significant volume levels
            hvn_levels = volume_profile['hvn_levels']
            
            # Order Flow Imbalance
            imbalance_levels = await self._detect_order_imbalances(market_data)
            
            order_flow_analysis = {
                'poc': poc_level,
                'value_area': {'high': value_area_high, 'low': value_area_low},
                'hvn_levels': hvn_levels,
                'imbalances': imbalance_levels,
                'volume_nodes': volume_profile['volume_nodes'],
                'total_volume': market_data['volume'].sum(),
                'avg_volume': market_data['volume'].mean()
            }
            
            self.logger.info(f"✅ Order flow analysis complete: POC at {poc_level:.2f}, {len(hvn_levels)} HVN levels")
            return order_flow_analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing order flow levels: {e}")
            return {}

    async def _calculate_volume_profile(self, market_data: pd.DataFrame) -> dict[str, Any]:
        """Calculate volume profile for order flow analysis."""
        try:
            # Create price bins
            price_range = market_data['high'].max() - market_data['low'].min()
            num_bins = 50
            bin_size = price_range / num_bins
            
            # Initialize volume profile
            volume_profile = {}
            for i in range(num_bins):
                price_level = market_data['low'].min() + i * bin_size
                volume_profile[price_level] = 0
            
            # Calculate volume at each price level
            for idx, row in market_data.iterrows():
                price = row['close']
                volume = row['volume']
                
                # Find the appropriate bin
                bin_index = int((price - market_data['low'].min()) / bin_size)
                bin_index = max(0, min(bin_index, num_bins - 1))
                price_level = market_data['low'].min() + bin_index * bin_size
                
                volume_profile[price_level] += volume
            
            # Find POC (Point of Control)
            poc_level = max(volume_profile, key=volume_profile.get)
            
            # Calculate Value Area (70% of volume)
            total_volume = sum(volume_profile.values())
            target_volume = total_volume * 0.7
            sorted_levels = sorted(volume_profile.items(), key=lambda x: x[1], reverse=True)
            
            cumulative_volume = 0
            value_area_levels = []
            for level, volume in sorted_levels:
                cumulative_volume += volume
                value_area_levels.append(level)
                if cumulative_volume >= target_volume:
                    break
            
            value_area_high = max(value_area_levels)
            value_area_low = min(value_area_levels)
            
            # Find HVN (High Volume Nodes) using optimized threshold
            avg_volume = total_volume / len(volume_profile)
            hvn_levels = [
                {'price': level, 'volume': volume, 'strength': volume / avg_volume}
                for level, volume in volume_profile.items()
                if volume > avg_volume * self.order_flow_hvn_threshold  # Use optimized threshold
            ]
            
            # Sort HVN by strength
            hvn_levels.sort(key=lambda x: x['strength'], reverse=True)
            
            return {
                'poc': poc_level,
                'value_area_high': value_area_high,
                'value_area_low': value_area_low,
                'hvn_levels': hvn_levels[:10],  # Top 10 HVN
                'volume_nodes': volume_profile
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating volume profile: {e}")
            return {}

    async def _detect_order_imbalances(self, market_data: pd.DataFrame) -> list[dict[str, Any]]:
        """Detect order flow imbalances."""
        try:
            imbalances = []
            
            # Calculate bid/ask imbalance (simplified - using volume as proxy)
            for i in range(1, len(market_data)):
                current_volume = market_data['volume'].iloc[i]
                prev_volume = market_data['volume'].iloc[i-1]
                current_price = market_data['close'].iloc[i]
                prev_price = market_data['close'].iloc[i-1]
                
                # Volume spike
                if current_volume > prev_volume * 2:
                    imbalance = {
                        'type': 'volume_spike',
                        'price': current_price,
                        'volume_ratio': current_volume / prev_volume,
                        'timestamp': market_data.index[i],
                        'strength': min(current_volume / prev_volume, 5.0)
                    }
                    imbalances.append(imbalance)
                
                # Price gap
                price_change = abs(current_price - prev_price) / prev_price
                if price_change > 0.01:  # 1% gap
                    imbalance = {
                        'type': 'price_gap',
                        'price': current_price,
                        'gap_size': price_change,
                        'timestamp': market_data.index[i],
                        'strength': min(price_change * 100, 5.0)
                    }
                    imbalances.append(imbalance)
            
            return imbalances
            
        except Exception as e:
            self.logger.error(f"Error detecting order imbalances: {e}")
            return []

    @validate_data_quality(validation_level="WARNING")
    async def detect_multi_timeframe_confluence(self, market_data: dict[str, pd.DataFrame]) -> dict[str, Any]:
        """Detect S/R levels that appear across multiple timeframes using optimized weights."""
        try:
            confluence_levels = {}
            
            # Use optimized timeframe weights
            timeframes = list(self.timeframe_weights.keys())
            
            for tf in timeframes:
                if tf in market_data:
                    # Detect S/R levels for this timeframe
                    tf_support = await self._detect_support_levels(market_data[tf])
                    tf_resistance = await self._detect_resistance_levels(market_data[tf])
                    
                    # Add to confluence analysis
                    for level in tf_support:
                        level_key = f"{level['price']:.2f}"
                        if level_key not in confluence_levels:
                            confluence_levels[level_key] = {
                                'price': level['price'],
                                'type': 'support',
                                'timeframes': [],
                                'strength': 0,
                                'methods': []
                            }
                        
                        confluence_levels[level_key]['timeframes'].append(tf)
                        # Apply timeframe weight to strength calculation
                        tf_weight = self.timeframe_weights.get(tf, 0.1)
                        weighted_strength = level.get('strength', 0.5) * tf_weight
                        confluence_levels[level_key]['strength'] += weighted_strength
                        if level.get('method') not in confluence_levels[level_key]['methods']:
                            confluence_levels[level_key]['methods'].append(level.get('method', 'unknown'))
                    
                    for level in tf_resistance:
                        level_key = f"{level['price']:.2f}"
                        if level_key not in confluence_levels:
                            confluence_levels[level_key] = {
                                'price': level['price'],
                                'type': 'resistance',
                                'timeframes': [],
                                'strength': 0,
                                'methods': []
                            }
                        
                        confluence_levels[level_key]['timeframes'].append(tf)
                        # Apply timeframe weight to strength calculation
                        tf_weight = self.timeframe_weights.get(tf, 0.1)
                        weighted_strength = level.get('strength', 0.5) * tf_weight
                        confluence_levels[level_key]['strength'] += weighted_strength
                        if level.get('method') not in confluence_levels[level_key]['methods']:
                            confluence_levels[level_key]['methods'].append(level.get('method', 'unknown'))
            
            # Filter for strong confluence (appears in 3+ timeframes)
            strong_confluence = {
                k: v for k, v in confluence_levels.items() 
                if len(v['timeframes']) >= 3
            }
            
            # Sort by strength
            strong_confluence = dict(sorted(strong_confluence.items(), key=lambda x: x[1]['strength'], reverse=True))
            
            self.logger.info(f"✅ Multi-timeframe confluence analysis: {len(strong_confluence)} strong confluence levels")
            return strong_confluence
            
        except Exception as e:
            self.logger.error(f"Error detecting multi-timeframe confluence: {e}")
            return {}

    @validate_data_quality(validation_level="WARNING")
    async def get_comprehensive_sr_analysis(self, market_data: pd.DataFrame, multi_timeframe_data: dict[str, pd.DataFrame] = None) -> dict[str, Any]:
        """Get comprehensive S/R analysis including all advanced methods."""
        try:
            # Basic S/R context
            basic_context = await self.get_sr_context(market_data, market_data['close'].iloc[-1])
            
            # Multi-timeframe confluence (if data provided)
            mtf_confluence = {}
            if multi_timeframe_data:
                mtf_confluence = await self.detect_multi_timeframe_confluence(multi_timeframe_data)
            
            comprehensive_analysis = {
                **basic_context,
                "multi_timeframe_confluence": mtf_confluence,
                "analysis_timestamp": pd.Timestamp.now(),
                "analysis_methods": [
                    "fractal_analysis",
                    "volume_analysis", 
                    "pivot_points",
                    "atr_analysis",
                    "fibonacci_levels",
                    "elliott_wave",
                    "order_flow_analysis",
                    "multi_timeframe_confluence"
                ]
            }
            
            self.logger.info(f"✅ Comprehensive S/R analysis complete with {len(comprehensive_analysis['analysis_methods'])} methods")
            return comprehensive_analysis
            
        except Exception as e:
            self.logger.error(f"Error in comprehensive S/R analysis: {e}")
            return {}

    def _calculate_level_strength(self, market_data: pd.DataFrame, index: int, level_type: str) -> float:
        """Calculate the strength of a support/resistance level."""
        try:
            # Base strength calculation
            base_strength = 0.5

            # Volume factor
            volume_factor = min(market_data['volume'].iloc[index] / market_data['volume'].mean(), 2.0)
            base_strength *= (0.5 + 0.5 * volume_factor)

            # Price movement factor
            if level_type == "support":
                price_factor = 1.0 - (market_data['low'].iloc[index] - market_data['close'].iloc[index]) / market_data['close'].iloc[index]
            else:  # resistance
                price_factor = 1.0 - (market_data['close'].iloc[index] - market_data['high'].iloc[index]) / market_data['close'].iloc[index]

            base_strength *= max(0.1, price_factor)

            return min(1.0, max(0.0, base_strength))

        except Exception as e:
            self.logger.error(f"Error calculating level strength: {e}")
            return 0.5

    # ============================================================================
    # ENHANCED STRENGTH CALCULATION METHODS
    # ============================================================================

    @validate_data_quality(validation_level="WARNING")
    async def calculate_touch_count(self, market_data: pd.DataFrame, sr_levels: list[dict[str, Any]]) -> dict[str, int]:
        """Calculate touch count for each S/R level."""
        try:
            touch_counts = {}
            
            for level in sr_levels:
                level_price = level['price']
                level_id = f"{level_price:.4f}"
                touch_count = 0
                
                # Look back through market data to count touches
                lookback_data = market_data.tail(self.touch_count_lookback)
                
                for i in range(1, len(lookback_data)):
                    high = lookback_data['high'].iloc[i]
                    low = lookback_data['low'].iloc[i]
                    prev_high = lookback_data['high'].iloc[i-1]
                    prev_low = lookback_data['low'].iloc[i-1]
                    
                    # Check if price touched the level (candlestick crossed the level)
                    if (low <= level_price <= high) or (prev_low <= level_price <= prev_high):
                        # Additional check: price actually approached the level
                        if abs(high - level_price) / level_price < 0.01 or abs(low - level_price) / level_price < 0.01:
                            touch_count += 1
                
                touch_counts[level_id] = touch_count
            
            self.logger.info(f"✅ Calculated touch counts for {len(touch_counts)} S/R levels")
            return touch_counts
            
        except Exception as e:
            self.logger.error(f"Error calculating touch counts: {e}")
            return {}

    @validate_data_quality(validation_level="WARNING")
    async def calculate_level_age(self, market_data: pd.DataFrame, sr_levels: list[dict[str, Any]]) -> dict[str, float]:
        """Calculate age of each S/R level."""
        try:
            level_ages = {}
            
            for level in sr_levels:
                level_price = level['price']
                level_id = f"{level_price:.4f}"
                level_timestamp = level.get('timestamp', market_data.index[-1])
                
                # Calculate age in periods
                if isinstance(level_timestamp, pd.Timestamp):
                    age_periods = len(market_data) - market_data.index.get_loc(level_timestamp)
                else:
                    # If no timestamp, estimate age based on level strength
                    age_periods = int(level.get('strength', 0.5) * 50)  # Estimate 0-50 periods
                
                # Apply age decay factor
                age_score = self.age_decay_factor ** age_periods
                
                level_ages[level_id] = {
                    'age_periods': age_periods,
                    'age_score': age_score,
                    'is_recent': age_periods <= 10
                }
            
            self.logger.info(f"✅ Calculated age scores for {len(level_ages)} S/R levels")
            return level_ages
            
        except Exception as e:
            self.logger.error(f"Error calculating level ages: {e}")
            return {}

    @validate_data_quality(validation_level="WARNING")
    async def calculate_bounce_rate(self, market_data: pd.DataFrame, sr_levels: list[dict[str, Any]]) -> dict[str, float]:
        """Calculate bounce rate for each S/R level."""
        try:
            bounce_rates = {}
            
            for level in sr_levels:
                level_price = level['price']
                level_id = f"{level_price:.4f}"
                touches = 0
                bounces = 0
                
                # Look back through market data to analyze bounces
                lookback_data = market_data.tail(self.touch_count_lookback)
                
                for i in range(1, len(lookback_data)):
                    high = lookback_data['high'].iloc[i]
                    low = lookback_data['low'].iloc[i]
                    close = lookback_data['close'].iloc[i]
                    prev_close = lookback_data['close'].iloc[i-1]
                    
                    # Check if price touched the level
                    if low <= level_price <= high:
                        touches += 1
                        
                        # Check if it was a bounce (price moved away from level)
                        if level_price > prev_close:  # Support level
                            # Price bounced up from support
                            if close > level_price + (level_price * self.bounce_rate_threshold):
                                bounces += 1
                        else:  # Resistance level
                            # Price bounced down from resistance
                            if close < level_price - (level_price * self.bounce_rate_threshold):
                                bounces += 1
                
                # Calculate bounce rate - handle untested levels properly
                if touches == 0:
                    # Level hasn't been tested yet - give neutral score
                    bounce_rate = 0.5  # Neutral score for untested levels
                    bounce_strength = 1.0  # Neutral strength
                    is_untested = True
                else:
                    bounce_rate = bounces / touches
                    bounce_strength = bounce_rate * 2  # Scale to 0-2 range
                    is_untested = False
                
                bounce_rates[level_id] = {
                    'bounce_rate': bounce_rate,
                    'touches': touches,
                    'bounces': bounces,
                    'bounce_strength': bounce_strength,
                    'is_untested': is_untested
                }
            
            self.logger.info(f"✅ Calculated bounce rates for {len(bounce_rates)} S/R levels")
            return bounce_rates
            
        except Exception as e:
            self.logger.error(f"Error calculating bounce rates: {e}")
            return {}

    @validate_data_quality(validation_level="WARNING")
    async def calculate_isolation_score(self, sr_levels: list[dict[str, Any]]) -> dict[str, float]:
        """Calculate isolation score for each S/R level."""
        try:
            isolation_scores = {}
            
            for i, level in enumerate(sr_levels):
                level_price = level['price']
                level_id = f"{level_price:.4f}"
                
                # Calculate distance to nearest other level
                min_distance = float('inf')
                for j, other_level in enumerate(sr_levels):
                    if i != j:
                        distance = abs(level_price - other_level['price']) / level_price
                        min_distance = min(min_distance, distance)
                
                # Calculate isolation score (higher = more isolated)
                if min_distance == float('inf'):
                    isolation_score = 1.0  # Only level
                else:
                    # Normalize to 0-1 range, higher distance = higher isolation
                    isolation_score = min(1.0, min_distance / self.isolation_distance_threshold)
                
                isolation_scores[level_id] = {
                    'isolation_score': isolation_score,
                    'nearest_distance': min_distance if min_distance != float('inf') else 0.0,
                    'is_isolated': isolation_score > 0.7
                }
            
            self.logger.info(f"✅ Calculated isolation scores for {len(isolation_scores)} S/R levels")
            return isolation_scores
            
        except Exception as e:
            self.logger.error(f"Error calculating isolation scores: {e}")
            return {}

    @validate_data_quality(validation_level="WARNING")
    async def cluster_sr_levels_dbscan(self, sr_levels: list[dict[str, Any]]) -> dict[str, Any]:
        """Cluster S/R levels using DBSCAN to identify significant levels."""
        try:
            if not DBSCAN_AVAILABLE:
                self.logger.warning("DBSCAN not available, returning unclustered levels")
                return {
                    'clusters': {},
                    'n_clusters': 0,
                    'noise_points': 0,
                    'total_points': len(sr_levels),
                    'clustered_levels': sr_levels
                }
            
            if not self.enable_dbscan_clustering or len(sr_levels) < 3:
                return {
                    'clusters': {},
                    'n_clusters': 0,
                    'noise_points': 0,
                    'total_points': len(sr_levels),
                    'clustered_levels': sr_levels
                }
            
            # Extract prices for clustering
            prices = np.array([level['price'] for level in sr_levels])
            
            # Normalize prices for clustering (use percentage of price)
            price_mean = np.mean(prices)
            normalized_prices = (prices - price_mean) / price_mean
            
            # Apply DBSCAN clustering
            clustering = DBSCAN(
                eps=self.dbscan_eps, 
                min_samples=self.dbscan_min_samples
            ).fit(normalized_prices.reshape(-1, 1))
            
            # Process clustering results
            cluster_labels = clustering.labels_
            n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            noise_points = np.sum(cluster_labels == -1)
            
            # Group levels by cluster
            clustered_levels = {}
            significant_levels = []
            
            for i, level in enumerate(sr_levels):
                cluster_id = cluster_labels[i]
                
                if cluster_id == -1:
                    # Noise points (weak levels) - filter out if enabled
                    if not self.dbscan_enable_noise_filtering:
                        significant_levels.append(level)
                    continue
                    
                if cluster_id not in clustered_levels:
                    clustered_levels[cluster_id] = {
                        'levels': [],
                        'cluster_price': 0.0,
                        'cluster_strength': 0.0,
                        'cluster_volume': 0.0,
                        'touch_count': 0
                    }
                
                clustered_levels[cluster_id]['levels'].append(level)
                significant_levels.append(level)
            
            # Calculate cluster statistics
            for cluster_id, cluster_data in clustered_levels.items():
                levels = cluster_data['levels']
                
                # Calculate cluster center (weighted average by strength)
                total_strength = sum(level.get('strength', 0.5) for level in levels)
                if total_strength > 0:
                    cluster_price = sum(level['price'] * level.get('strength', 0.5) for level in levels) / total_strength
                else:
                    cluster_price = np.mean([level['price'] for level in levels])
                
                # Aggregate cluster metrics
                cluster_strength = sum(level.get('strength', 0.5) for level in levels) / len(levels)
                cluster_volume = sum(level.get('volume', 0) for level in levels)
                touch_count = sum(level.get('touch_count', 1) for level in levels)
                
                clustered_levels[cluster_id].update({
                    'cluster_price': cluster_price,
                    'cluster_strength': cluster_strength,
                    'cluster_volume': cluster_volume,
                    'touch_count': touch_count,
                    'level_count': len(levels)
                })
            
            self.logger.info(f"✅ DBSCAN clustering: {n_clusters} clusters, {noise_points} noise points filtered")
            return {
                'clusters': clustered_levels,
                'n_clusters': n_clusters,
                'noise_points': noise_points,
                'total_points': len(sr_levels),
                'clustered_levels': significant_levels
            }
            
        except Exception as e:
            self.logger.error(f"Error in DBSCAN clustering: {e}")
            return {
                'clusters': {},
                'n_clusters': 0,
                'noise_points': 0,
                'total_points': len(sr_levels),
                'clustered_levels': sr_levels
            }

    @validate_data_quality(validation_level="WARNING")
    async def calculate_comprehensive_strength(self, market_data: pd.DataFrame, sr_levels: list[dict[str, Any]]) -> dict[str, float]:
        """Calculate comprehensive strength using all factors."""
        try:
            if not self.enable_enhanced_strength:
                # Return basic strength calculation
                return {f"{level['price']:.4f}": level.get('strength', 0.5) for level in sr_levels}
            
            # Calculate all strength factors
            touch_counts = await self.calculate_touch_count(market_data, sr_levels)
            level_ages = await self.calculate_level_age(market_data, sr_levels)
            bounce_rates = await self.calculate_bounce_rate(market_data, sr_levels)
            isolation_scores = await self.calculate_isolation_score(sr_levels)
            
            comprehensive_strengths = {}
            
            for level in sr_levels:
                level_price = level['price']
                level_id = f"{level_price:.4f}"
                
                # Get base strength
                base_strength = level.get('strength', 0.5)
                
                # Get factor scores
                touch_count_data = touch_counts.get(level_id, {'touch_count': 1})
                age_data = level_ages.get(level_id, {'age_score': 0.5})
                bounce_data = bounce_rates.get(level_id, {'bounce_strength': 0.5, 'is_untested': False})
                isolation_data = isolation_scores.get(level_id, {'isolation_score': 0.5})
                
                # Calculate factor scores (normalize to 0-1 range)
                touch_factor = min(1.0, touch_count_data.get('touch_count', 1) / 10.0)  # Max 10 touches
                age_factor = age_data.get('age_score', 0.5)
                
                # Handle untested levels properly for bounce factor
                if bounce_data.get('is_untested', False):
                    bounce_factor = 0.5  # Neutral score for untested levels
                else:
                    bounce_factor = min(1.0, bounce_data.get('bounce_strength', 0.5) / 2.0)  # Max 2.0 strength
                
                isolation_factor = isolation_data.get('isolation_score', 0.5)
                volume_factor = min(1.0, level.get('volume', 0) / market_data['volume'].mean() if market_data['volume'].mean() > 0 else 0.5)
                
                # Apply weights from configuration
                weights = self.strength_score_weights
                comprehensive_strength = (
                    base_strength * 0.2 +  # Base strength gets 20% weight
                    touch_factor * weights.get('touch_count', 0.3) +
                    volume_factor * weights.get('total_volume', 0.2) +
                    age_factor * weights.get('level_age', 0.2) +
                    bounce_factor * weights.get('bounce_rate', 0.2) +
                    isolation_factor * weights.get('isolation_score', 0.1)
                )
                
                # Ensure strength is in 0-1 range
                comprehensive_strength = min(1.0, max(0.0, comprehensive_strength))
                
                comprehensive_strengths[level_id] = {
                    'comprehensive_strength': comprehensive_strength,
                    'base_strength': base_strength,
                    'touch_factor': touch_factor,
                    'volume_factor': volume_factor,
                    'age_factor': age_factor,
                    'bounce_factor': bounce_factor,
                    'isolation_factor': isolation_factor,
                    'factors': {
                        'touch_count': touch_count_data.get('touch_count', 1),
                        'age_periods': age_data.get('age_periods', 0),
                        'bounce_rate': bounce_data.get('bounce_rate', 0.0),
                        'isolation_score': isolation_data.get('isolation_score', 0.5),
                        'is_untested': bounce_data.get('is_untested', False)
                    }
                }
            
            self.logger.info(f"✅ Calculated comprehensive strength for {len(comprehensive_strengths)} S/R levels")
            return comprehensive_strengths
            
        except Exception as e:
            self.logger.error(f"Error calculating comprehensive strength: {e}")
            return {f"{level['price']:.4f}": level.get('strength', 0.5) for level in sr_levels}

    async def _calculate_breakout_probabilities(
        self,
        support_levels: list[dict[str, Any]],
        resistance_levels: list[dict[str, Any]],
        current_price: float,
    ) -> dict[str, float]:
        """Calculate breakout probabilities for support and resistance levels."""
        try:
            probabilities = {}

            # Calculate support breakout probabilities
            for i, level in enumerate(support_levels):
                distance = (current_price - level["price"]) / current_price
                if distance < 0:  # Price below support
                    prob = min(0.9, abs(distance) / self.sr_proximity_threshold)
                    probabilities[f"support_breakout_{i}"] = prob
                else:
                    probabilities[f"support_breakout_{i}"] = 0.0

            # Calculate resistance breakout probabilities
            for i, level in enumerate(resistance_levels):
                distance = (level["price"] - current_price) / current_price
                if distance < 0:  # Price above resistance
                    prob = min(0.9, abs(distance) / self.sr_proximity_threshold)
                    probabilities[f"resistance_breakout_{i}"] = prob
                else:
                    probabilities[f"resistance_breakout_{i}"] = 0.0

            return probabilities

        except Exception as e:
            self.logger.error(f"Error calculating breakout probabilities: {e}")
            return {}

    async def _calculate_confidence_scores(
        self,
        support_levels: list[dict[str, Any]],
        resistance_levels: list[dict[str, Any]],
        market_data: pd.DataFrame,
    ) -> dict[str, float]:
        """Calculate confidence scores for support and resistance levels."""
        try:
            confidence_scores = {}

            # Calculate support confidence scores
            for i, level in enumerate(support_levels):
                confidence = level.get("confidence", 0.5) * level.get("strength", 0.5)
                confidence_scores[f"support_confidence_{i}"] = confidence

            # Calculate resistance confidence scores
            for i, level in enumerate(resistance_levels):
                confidence = level.get("confidence", 0.5) * level.get("strength", 0.5)
                confidence_scores[f"resistance_confidence_{i}"] = confidence

            return confidence_scores

        except Exception as e:
            self.logger.error(f"Error calculating confidence scores: {e}")
            return {}

    async def _generate_sr_features(
        self,
        support_levels: list[dict[str, Any]],
        resistance_levels: list[dict[str, Any]],
        market_data: pd.DataFrame,
    ) -> dict[str, Any]:
        """Generate SR-related features for machine learning."""
        try:
            features = {}

            # Calculate proximity to nearest support and resistance
            if support_levels:
                nearest_support = min(support_levels, key=lambda x: abs(x["price"] - market_data['close'].iloc[-1]))
                features["support_proximity"] = abs(nearest_support["price"] - market_data['close'].iloc[-1]) / market_data['close'].iloc[-1]
                features["support_strength"] = nearest_support.get("strength", 0.5)
            else:
                features["support_proximity"] = 1.0
                features["support_strength"] = 0.0

            if resistance_levels:
                nearest_resistance = min(resistance_levels, key=lambda x: abs(x["price"] - market_data['close'].iloc[-1]))
                features["resistance_proximity"] = abs(nearest_resistance["price"] - market_data['close'].iloc[-1]) / market_data['close'].iloc[-1]
                features["resistance_strength"] = nearest_resistance.get("strength", 0.5)
            else:
                features["resistance_proximity"] = 1.0
                features["resistance_strength"] = 0.0

            # Calculate SR zone features
            features["sr_zone_width"] = features["resistance_proximity"] + features["support_proximity"]
            features["sr_zone_center"] = (features["resistance_proximity"] - features["support_proximity"]) / 2

            # Calculate level count features
            features["support_level_count"] = len(support_levels)
            features["resistance_level_count"] = len(resistance_levels)
            features["total_sr_levels"] = len(support_levels) + len(resistance_levels)

            return features

        except Exception as e:
            self.logger.error(f"Error generating SR features: {e}")
            return {}

    def _find_nearest_level(
        self,
        current_price: float,
        levels: list[dict[str, Any]],
        level_type: str,
    ) -> dict[str, Any] | None:
        """Find the nearest support or resistance level with enhanced strength consideration."""
        try:
            if not levels:
                return None

            nearest_level = None
            best_score = float('-inf')

            for level in levels:
                # Calculate distance score (closer is better)
                distance = abs(current_price - level["price"]) / current_price
                distance_score = 1.0 / (1.0 + distance)  # Convert to 0-1 score, higher is better
                
                # Get strength score (use enhanced strength if available)
                strength = level.get("enhanced_strength", level.get("strength", 0.5))
                
                # Combine distance and strength (weighted average)
                # Distance gets 60% weight, strength gets 40% weight
                combined_score = (distance_score * 0.6) + (strength * 0.4)
                
                if combined_score > best_score:
                    best_score = combined_score
                    nearest_level = level

            return nearest_level

        except Exception as e:
            self.logger.error(f"Error finding nearest {level_type} level: {e}")
            return None

    def _calculate_proximity(
        self,
        current_price: float,
        level: dict[str, Any] | None,
    ) -> float:
        """Calculate proximity to a level."""
        try:
            if not level:
                return 1.0

            distance = abs(current_price - level["price"]) / current_price
            return distance

        except Exception as e:
            self.logger.error(f"Error calculating proximity: {e}")
            return 1.0

    def _calculate_pivot_levels(self, market_data: pd.DataFrame) -> dict[str, Any]:
        """Calculate pivot point levels."""
        try:
            if len(market_data) < 1:
                return {}

            # Calculate pivot point
            high = market_data['high'].iloc[-1]
            low = market_data['low'].iloc[-1]
            close = market_data['close'].iloc[-1]

            pivot = (high + low + close) / 3
            r1 = 2 * pivot - low
            s1 = 2 * pivot - high

            return {
                "pivot": pivot,
                "r1": r1,
                "s1": s1,
                "nearest_strength": 0.5,
                "nearest_touches": 1,
            }

        except Exception as e:
            self.logger.error(f"Error calculating pivot levels: {e}")
            return {}

    async def _extract_outcome_features(
        self,
        market_data: pd.DataFrame,
        current_price: float,
        sr_context: dict[str, Any],
    ) -> dict[str, float]:
        """Extract features for S/R outcome prediction."""
        try:
            features: dict[str, float] = {}

            # Price-based features
            features["price_change_1m"] = (
                market_data["close"].pct_change().iloc[-1]
                if len(market_data) > 1
                else 0
            )
            features["price_change_5m"] = (
                market_data["close"].pct_change(5).iloc[-1]
                if len(market_data) > 5
                else 0
            )
            features["price_change_15m"] = (
                market_data["close"].pct_change(15).iloc[-1]
                if len(market_data) > 15
                else 0
            )
            features["price_volatility"] = (
                market_data["close"].rolling(20).std().iloc[-1]
                if len(market_data) >= 20
                else 0
            )

            # Volume-based features
            features["volume_ratio"] = (
                (
                    market_data["volume"].iloc[-1]
                    / market_data["volume"].rolling(20).mean().iloc[-1]
                )
                if len(market_data) >= 20
                else 1.0
            )
            features["volume_momentum"] = (
                market_data["volume"].pct_change().iloc[-1]
                if len(market_data) > 1
                else 0
            )

            # Technical indicators
            features["rsi"] = (
                self._calculate_rsi(market_data["close"]).iloc[-1]
                if len(market_data) >= 14
                else 50
            )
            features["macd"] = (
                self._calculate_macd(market_data["close"]).iloc[-1]
                if len(market_data) >= 26
                else 0
            )
            features["bb_position"] = (
                self._calculate_bb_position(market_data["close"]).iloc[-1]
                if len(market_data) >= 20
                else 0.5
            )

            # S/R-specific features
            if sr_context:
                nearest_support = sr_context.get("nearest_support", current_price)
                nearest_resistance = sr_context.get("nearest_resistance", current_price)

                features["distance_to_support"] = (
                    current_price - nearest_support
                ) / current_price
                features["distance_to_resistance"] = (
                    nearest_resistance - current_price
                ) / current_price
                features["support_strength"] = sr_context.get("support_strength", 0.5)
                features["resistance_strength"] = sr_context.get(
                    "resistance_strength", 0.5,
                )

                # Pivot level features
                pivot_levels = sr_context.get("pivot_levels", {})
                if pivot_levels:
                    features["nearest_pivot_strength"] = pivot_levels.get(
                        "nearest_strength", 0.5,
                    )
                    features["pivot_touches"] = pivot_levels.get("nearest_touches", 0)
                else:
                    features["nearest_pivot_strength"] = 0.5
                    features["pivot_touches"] = 0

            # Market context features
            features["market_trend"] = self._calculate_market_trend(market_data)
            features["momentum_strength"] = self._calculate_momentum_strength(
                market_data,
            )

            return features

        except Exception as e:
            self.logger.error(f"Error extracting outcome features: {e}")
            return {}

    def _predict_outcome_rules(
        self,
        features: dict[str, float],
        sr_context: dict[str, Any],
    ) -> str:
        """Predict S/R outcome using rule-based logic."""
        try:
            # Extract key features
            price_change_1m = features.get("price_change_1m", 0)
            price_change_5m = features.get("price_change_5m", 0)
            volume_ratio = features.get("volume_ratio", 1.0)
            rsi = features.get("rsi", 50)
            distance_to_support = features.get("distance_to_support", 0)
            distance_to_resistance = features.get("distance_to_resistance", 0)
            support_strength = features.get("support_strength", 0.5)
            resistance_strength = features.get("resistance_strength", 0.5)

            # Determine if near support or resistance
            is_near_support = abs(distance_to_support) < self.sr_proximity_threshold
            is_near_resistance = abs(distance_to_resistance) < self.sr_proximity_threshold

            # Breakout conditions
            if is_near_resistance and price_change_1m > 0.001 and volume_ratio > 1.2:
                return "breakout"
            elif is_near_support and price_change_1m < -0.001 and volume_ratio > 1.2:
                return "breakout"

            # Rebounce conditions
            elif is_near_resistance and price_change_1m < -0.001 and rsi > 70:
                return "rebounce"
            elif is_near_support and price_change_1m > 0.001 and rsi < 30:
                return "rebounce"

            # Default to consolidation
            else:
                return "consolidation"

        except Exception as e:
            self.logger.error(f"Error predicting outcome: {e}")
            return "consolidation"

    def _calculate_outcome_confidence(
        self,
        features: dict[str, float],
        sr_context: dict[str, Any],
    ) -> float:
        """Calculate confidence in S/R outcome prediction."""
        try:
            # Base confidence
            confidence = 0.5

            # Volume factor
            volume_ratio = features.get("volume_ratio", 1.0)
            if volume_ratio > 1.5:
                confidence += 0.2
            elif volume_ratio > 1.2:
                confidence += 0.1

            # Strength factor
            support_strength = features.get("support_strength", 0.5)
            resistance_strength = features.get("resistance_strength", 0.5)
            max_strength = max(support_strength, resistance_strength)
            confidence += max_strength * 0.2

            # Proximity factor
            support_proximity = sr_context.get("support_proximity", 1.0)
            resistance_proximity = sr_context.get("resistance_proximity", 1.0)
            min_proximity = min(support_proximity, resistance_proximity)
            if min_proximity < self.sr_proximity_threshold:
                confidence += 0.2

            # RSI factor
            rsi = features.get("rsi", 50)
            if rsi < 30 or rsi > 70:
                confidence += 0.1

            return min(1.0, confidence)

        except Exception as e:
            self.logger.error(f"Error calculating outcome confidence: {e}")
            return 0.5

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _calculate_macd(
        self, prices: pd.Series, fast: int = 12, slow: int = 26
    ) -> pd.Series:
        """Calculate MACD indicator."""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        return ema_fast - ema_slow

    def _calculate_bb_position(
        self, prices: pd.Series, period: int = 20, std: int = 2
    ) -> pd.Series:
        """Calculate Bollinger Band position."""
        sma = prices.rolling(window=period).mean()
        std_dev = prices.rolling(window=period).std()
        upper_band = sma + (std_dev * std)
        lower_band = sma - (std_dev * std)

        # Position within bands (0, at lower band, 1, at upper band)
        bb_position = (prices - lower_band) / (upper_band - lower_band)
        return bb_position.clip(0, 1)

    def _calculate_market_trend(self, market_data: pd.DataFrame) -> float:
        """Calculate market trend strength."""
        try:
            if len(market_data) < 20:
                return 0.0

            prices = market_data["close"].values
            x = np.arange(len(prices))
            slope = np.polyfit(x, prices, 1)[0]

            avg_price = np.mean(prices)
            normalized_slope = slope / avg_price if avg_price > 0 else 0

            return float(np.clip(normalized_slope * 100, -1, 1))
        except Exception as e:
            self.logger.error(f"Error calculating market trend: {e}")
            return 0.0

    def _calculate_momentum_strength(self, market_data: pd.DataFrame) -> float:
        """Calculate momentum strength."""
        try:
            if len(market_data) < 10:
                return 0.0

            short_momentum = (
                market_data["close"].pct_change(5).iloc[-1]
                if len(market_data) > 5
                else 0
            )
            long_momentum = (
                market_data["close"].pct_change(20).iloc[-1]
                if len(market_data) > 20
                else 0
            )

            momentum = short_momentum * 0.7 + long_momentum * 0.3

            return float(np.clip(momentum * 100, -1, 1))
        except Exception as e:
            self.logger.error(f"Error calculating momentum strength: {e}")
            return 0.0

    def _update_performance_metrics(self, predictions: dict[str, Any]) -> None:
        """Update performance metrics for SR breakout predictions."""
        try:
            # Store prediction in history
            self.prediction_history.append(predictions)

            # Keep only recent predictions
            if len(self.prediction_history) > 1000:
                self.prediction_history = self.prediction_history[-1000:]

            # Calculate basic metrics
            self.performance_metrics["total_predictions"] = len(self.prediction_history)
            self.performance_metrics["last_prediction_time"] = predictions.get("timestamp", pd.Timestamp.now())

        except Exception as e:
            self.logger.error(f"Error updating performance metrics: {e}")

    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid input data for S/R proximity check"),
            KeyError: (False, "Missing required S/R context data"),
        },
        default_return=False,
        context="S/R proximity check",
    )
    def is_near_sr_level(
        self,
        current_price: float,
        sr_context: dict[str, Any],
    ) -> bool:
        """
        Check if price is near significant S/R level.

        Args:
            current_price: Current market price
            sr_context: S/R context from get_sr_context

        Returns:
            bool: True if near S/R level
        """
        try:
            if not sr_context:
                return False

            # Check proximity to support and resistance
            support_proximity = sr_context.get("support_proximity", 1.0)
            resistance_proximity = sr_context.get("resistance_proximity", 1.0)

            # Consider near if within threshold
            is_near_support = support_proximity <= self.sr_proximity_threshold
            is_near_resistance = resistance_proximity <= self.sr_proximity_threshold

            return is_near_support or is_near_resistance

        except Exception as e:
            self.logger.error(f"Error checking S/R proximity: {e}")
            return False

    def get_sr_proximity_details(
        self,
        current_price: float,
        sr_context: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Get detailed proximity information to S/R levels.

        Args:
            current_price: Current market price
            sr_context: S/R context from get_sr_context

        Returns:
            dict[str, Any]: Detailed proximity information
        """
        try:
            if not sr_context:
                return {}

            details = {
                "current_price": current_price,
                "nearest_support": {
                    "price": sr_context.get("nearest_support", current_price),
                    "proximity": sr_context.get("support_proximity", 1.0),
                    "strength": sr_context.get("support_strength", 0.5),
                },
                "nearest_resistance": {
                    "price": sr_context.get("nearest_resistance", current_price),
                    "proximity": sr_context.get("resistance_proximity", 1.0),
                    "strength": sr_context.get("resistance_strength", 0.5),
                },
                "sr_zone_width": sr_context.get("sr_zone_width", 0.0),
                "is_near_sr": self.is_near_sr_level(current_price, sr_context),
                "closest_level_type": "support" if sr_context.get("support_proximity", 1.0) < sr_context.get("resistance_proximity", 1.0) else "resistance",
            }

            return details

        except Exception as e:
            self.logger.error(f"Error getting S/R proximity details: {e}")
            return {}

    @validate_data_quality(
        required_columns=["open", "high", "low", "close", "volume"],
        min_rows=20,
        max_null_ratio=0.1,
        check_duplicates=True,
        check_timestamps=True,
        context="S/R outcome prediction input validation"
    )
    @handle_specific_errors(
        error_handlers={
            ValueError: ({}, "Invalid input data for S/R outcome prediction"),
            KeyError: ({}, "Missing required S/R context data"),
            AttributeError: ({}, "Predictor not properly initialized"),
        },
        default_return={},
        context="S/R outcome prediction",
    )
    async def predict_sr_outcome(
        self,
        market_data: pd.DataFrame,
        current_price: float,
        sr_context: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Predict S/R outcome (breakout/rebounce/consolidation).

        Args:
            market_data: Market data DataFrame
            current_price: Current market price
            sr_context: S/R context from get_sr_context

        Returns:
            dict[str, Any]: S/R outcome prediction
        """
        if not self.is_initialized:
            self.logger.error("SR breakout predictor not initialized")
            return {}

        try:
            # Extract features for prediction
            features = await self._extract_outcome_features(market_data, current_price, sr_context)

            # Simple rule-based prediction (can be enhanced with ML model)
            outcome = self._predict_outcome_rules(features, sr_context)

            # Calculate confidence
            confidence = self._calculate_outcome_confidence(features, sr_context)

            result = {
                "outcome": outcome,
                "confidence": confidence,
                "features": features,
                "sr_context": sr_context,
                "current_price": current_price,
                "timestamp": pd.Timestamp.now(),
            }

            return result

        except Exception as e:
            self.logger.error(f"Error predicting S/R outcome: {e}")
            return {}

    @validate_data_quality(
        required_columns=["open", "high", "low", "close", "volume"],
        min_rows=50,
        max_null_ratio=0.1,
        check_duplicates=True,
        check_timestamps=True,
        context="S/R features calculation input validation"
    )
    @handle_specific_errors(
        error_handlers={
            ValueError: ({}, "Invalid input data for S/R features calculation"),
            AttributeError: ({}, "Predictor not properly initialized"),
        },
        default_return={},
        context="S/R features calculation",
    )
    async def calculate_sr_features(
        self,
        market_data: pd.DataFrame,
    ) -> dict[str, Any]:
        """
        Calculate SR-related features.

        Args:
            market_data: Market data DataFrame

        Returns:
            dict[str, Any]: SR features
        """
        if not self.is_initialized:
            self.logger.error("SR breakout predictor not initialized")
            return {}

        try:
            # Get current price
            current_price = market_data['close'].iloc[-1]

            # Get S/R context
            sr_context = await self.get_sr_context(market_data, current_price)

            # Extract comprehensive features
            features = await self._extract_outcome_features(market_data, current_price, sr_context)

            # Add proximity features
            features.update({
                "is_near_sr_level": self.is_near_sr_level(current_price, sr_context),
                "sr_zone_width": sr_context.get("sr_zone_width", 0.0),
                "support_proximity": sr_context.get("support_proximity", 1.0),
                "resistance_proximity": sr_context.get("resistance_proximity", 1.0),
            })

            self.logger.info("✅ SR features calculated")
            return features

        except Exception as e:
            self.logger.error(f"Error calculating SR features: {e}")
            return {}

    @validate_data_quality(
        required_columns=["open", "high", "low", "close", "volume"],
        min_rows=100,
        max_null_ratio=0.1,
        check_duplicates=True,
        check_timestamps=True,
        context="Comprehensive S/R features calculation input validation"
    )
    @handle_specific_errors(
        error_handlers={
            ValueError: ({}, "Invalid input data for comprehensive S/R features calculation"),
            AttributeError: ({}, "Predictor not properly initialized"),
        },
        default_return={},
        context="Comprehensive S/R features calculation",
    )
    async def calculate_comprehensive_sr_features(
        self,
        market_data: pd.DataFrame,
    ) -> dict[str, pd.Series]:
        """
        Calculate comprehensive S/R features with multiple timeframes.

        Args:
            market_data: Market data DataFrame

        Returns:
            dict[str, pd.Series]: Comprehensive S/R features
        """
        if not self.is_initialized:
            self.logger.error("SR breakout predictor not initialized")
            return {}

        try:
            features = {}
            current_price = market_data['close'].iloc[-1]
            
            # Get comprehensive S/R context
            sr_context = await self.get_sr_context(market_data, current_price)
            
            # Extract base outcome features
            base_features = await self._extract_outcome_features(market_data, current_price, sr_context)
            
            # Generate all required SR features based on sr_base_tokens
            features.update(await self._generate_comprehensive_sr_features(market_data, sr_context, base_features))
            
            # Calculate features for different lookback periods
            for lookback in [20, 50, 100]:
                if len(market_data) >= lookback:
                    lookback_data = market_data.tail(lookback)
                    lookback_price = lookback_data['close'].iloc[-1]
                    
                    # Get S/R context for this lookback
                    lookback_sr_context = await self.get_sr_context(lookback_data, lookback_price)
                    
                    # Extract features with lookback suffix
                    lookback_features = await self._extract_outcome_features(lookback_data, lookback_price, lookback_sr_context)
                    
                    # Add to features with lookback suffix
                    for feature_name, feature_value in lookback_features.items():
                        features[f"{feature_name}_{lookback}"] = pd.Series([feature_value] * len(market_data), index=market_data.index)

            self.logger.info(f"✅ Generated {len(features)} comprehensive SR features")
            return features

        except Exception as e:
            self.logger.error(f"Error calculating comprehensive SR features: {e}")
            return {}

    async def _generate_comprehensive_sr_features(
        self,
        market_data: pd.DataFrame,
        sr_context: dict[str, Any],
        base_features: dict[str, float]
    ) -> dict[str, pd.Series]:
        """Generate comprehensive SR features matching sr_base_tokens requirements."""
        try:
            features = {}
            current_price = market_data['close'].iloc[-1]
            
            # 1. Distance-based features
            nearest_support = sr_context.get("nearest_support", current_price)
            nearest_resistance = sr_context.get("nearest_resistance", current_price)
            
            # Calculate distances as percentages/returns
            support_distance_pct = (current_price - nearest_support) / current_price if current_price > 0 else 0.0
            resistance_distance_pct = (nearest_resistance - current_price) / current_price if current_price > 0 else 0.0
            
            features["sr_distance"] = pd.Series([support_distance_pct] * len(market_data), index=market_data.index)
            features["support_level"] = pd.Series([support_distance_pct] * len(market_data), index=market_data.index)  # Distance as percentage
            features["resistance_level"] = pd.Series([resistance_distance_pct] * len(market_data), index=market_data.index)  # Distance as percentage
            
            # 2. Proximity features (as percentages)
            features["proximity"] = pd.Series([min(support_distance_pct, resistance_distance_pct)] * len(market_data), index=market_data.index)
            features["sr_proximity"] = pd.Series([support_distance_pct] * len(market_data), index=market_data.index)
            features["sr_proximity_score"] = pd.Series([1.0 / (1.0 + support_distance_pct)] * len(market_data), index=market_data.index)
            
            # 3. Multi-timeframe SR score
            features["multi_timeframe_sr_score"] = pd.Series([self._calculate_multi_timeframe_sr_score(market_data)] * len(market_data), index=market_data.index)
            features["sr_multi_timeframe"] = pd.Series([self._calculate_multi_timeframe_sr_score(market_data)] * len(market_data), index=market_data.index)
            
            # 4. Normalized distance (as percentage)
            if nearest_resistance > nearest_support and current_price > 0:
                zone_width_pct = (nearest_resistance - nearest_support) / current_price
                normalized_distance_pct = support_distance_pct / zone_width_pct if zone_width_pct > 0 else 0.5
            else:
                normalized_distance_pct = 0.5
            features["normalized_distance"] = pd.Series([normalized_distance_pct] * len(market_data), index=market_data.index)
            
            # 5. Strength features
            features["strength_score"] = pd.Series([(sr_context.get("support_strength", 0.5) + sr_context.get("resistance_strength", 0.5)) / 2] * len(market_data), index=market_data.index)
            features["support_strength"] = pd.Series([sr_context.get("support_strength", 0.5)] * len(market_data), index=market_data.index)
            features["resistance_strength"] = pd.Series([sr_context.get("resistance_strength", 0.5)] * len(market_data), index=market_data.index)
            
            # 6. Clarity factor
            features["clarity_factor"] = pd.Series([self._calculate_clarity_factor(sr_context)] * len(market_data), index=market_data.index)
            
            # 7. Directional pressure
            features["directional_pressure"] = pd.Series([self._calculate_directional_pressure(market_data, sr_context)] * len(market_data), index=market_data.index)
            
            # 8. SR score
            features["sr_score"] = pd.Series([self._calculate_sr_score(sr_context)] * len(market_data), index=market_data.index)
            
            # 9. Delta SR score
            features["delta_sr_score"] = pd.Series([self._calculate_delta_sr_score(market_data, sr_context)] * len(market_data), index=market_data.index)
            
            # 10. Isolation score
            features["isolation_score"] = pd.Series([self._calculate_isolation_score(sr_context)] * len(market_data), index=market_data.index)
            
            # 11. SR level
            features["sr_level"] = pd.Series([self._determine_sr_level(current_price, sr_context)] * len(market_data), index=market_data.index)
            
            # 12. SR outcome
            features["sr_outcome"] = pd.Series([self._predict_sr_outcome(market_data, sr_context)] * len(market_data), index=market_data.index)
            
            # 13. Zone width
            features["sr_zone_width"] = pd.Series([sr_context.get("sr_zone_width", 0.0)] * len(market_data), index=market_data.index)
            
            # 14. Add base features
            for feature_name, feature_value in base_features.items():
                features[f"sr_{feature_name}"] = pd.Series([feature_value] * len(market_data), index=market_data.index)
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error generating comprehensive SR features: {e}")
            return {}

    def _calculate_multi_timeframe_sr_score(self, market_data: pd.DataFrame) -> float:
        """Calculate multi-timeframe SR score."""
        try:
            # Calculate SR strength across different timeframes
            timeframes = [20, 50, 100]
            scores = []
            
            for tf in timeframes:
                if len(market_data) >= tf:
                    tf_data = market_data.tail(tf)
                    # Simple SR strength calculation based on price action
                    high_low_ratio = tf_data['high'].max() / tf_data['low'].min()
                    volume_weight = tf_data['volume'].mean() / market_data['volume'].mean()
                    scores.append(high_low_ratio * volume_weight)
            
            return np.mean(scores) if scores else 1.0
            
        except Exception as e:
            self.logger.error(f"Error calculating multi-timeframe SR score: {e}")
            return 1.0

    def _calculate_clarity_factor(self, sr_context: dict[str, Any]) -> float:
        """Calculate SR clarity factor."""
        try:
            support_strength = sr_context.get("support_strength", 0.5)
            resistance_strength = sr_context.get("resistance_strength", 0.5)
            zone_width = sr_context.get("sr_zone_width", 0.0)
            
            # Clarity increases with strength and decreases with zone width
            clarity = (support_strength + resistance_strength) / 2
            if zone_width > 0:
                clarity *= (1.0 - min(zone_width, 0.5))
            
            return max(0.0, min(1.0, clarity))
            
        except Exception as e:
            self.logger.error(f"Error calculating clarity factor: {e}")
            return 0.5

    def _calculate_directional_pressure(self, market_data: pd.DataFrame, sr_context: dict[str, Any]) -> float:
        """Calculate directional pressure towards SR levels using percentages."""
        try:
            current_price = market_data['close'].iloc[-1]
            nearest_support = sr_context.get("nearest_support", current_price)
            nearest_resistance = sr_context.get("nearest_resistance", current_price)
            
            # Calculate pressure based on distance and momentum (using percentages)
            if current_price > 0:
                support_distance_pct = (current_price - nearest_support) / current_price
                resistance_distance_pct = (nearest_resistance - current_price) / current_price
            else:
                support_distance_pct = 1.0
                resistance_distance_pct = 1.0
            
            # Add momentum component (as percentage)
            momentum = market_data['close'].pct_change().iloc[-5:].mean()
            
            # Pressure towards support if price is falling, towards resistance if rising
            if momentum < 0:
                pressure = 1.0 / (1.0 + support_distance_pct)
            else:
                pressure = 1.0 / (1.0 + resistance_distance_pct)
            
            return max(0.0, min(1.0, pressure))
            
        except Exception as e:
            self.logger.error(f"Error calculating directional pressure: {e}")
            return 0.5

    def _calculate_sr_score(self, sr_context: dict[str, Any]) -> float:
        """Calculate overall SR score using percentages."""
        try:
            support_strength = sr_context.get("support_strength", 0.5)
            resistance_strength = sr_context.get("resistance_strength", 0.5)
            support_proximity = sr_context.get("support_proximity", 1.0)
            resistance_proximity = sr_context.get("resistance_proximity", 1.0)
            
            # Combine strength and proximity (proximity is already in percentage form)
            support_score = support_strength * (1.0 / (1.0 + support_proximity))
            resistance_score = resistance_strength * (1.0 / (1.0 + resistance_proximity))
            
            return (support_score + resistance_score) / 2
            
        except Exception as e:
            self.logger.error(f"Error calculating SR score: {e}")
            return 0.5

    def _calculate_delta_sr_score(self, market_data: pd.DataFrame, sr_context: dict[str, Any]) -> float:
        """Calculate change in SR score over time."""
        try:
            if len(market_data) < 20:
                return 0.0
            
            # Calculate SR score for current and previous periods
            current_price = market_data['close'].iloc[-1]
            prev_price = market_data['close'].iloc[-20]
            
            # Simplified delta calculation
            price_change = (current_price - prev_price) / prev_price
            return price_change
            
        except Exception as e:
            self.logger.error(f"Error calculating delta SR score: {e}")
            return 0.0

    def _calculate_isolation_score(self, sr_context: dict[str, Any]) -> float:
        """Calculate isolation score for SR levels."""
        try:
            # Use isolation data from enhanced strength calculation
            support_levels = sr_context.get("support_levels", [])
            resistance_levels = sr_context.get("resistance_levels", [])
            
            if not support_levels and not resistance_levels:
                return 0.5
            
            # Calculate average isolation score
            isolation_scores = []
            for level in support_levels + resistance_levels:
                if "strength_factors" in level and "isolation_score" in level["strength_factors"]:
                    isolation_scores.append(level["strength_factors"]["isolation_score"])
            
            return np.mean(isolation_scores) if isolation_scores else 0.5
            
        except Exception as e:
            self.logger.error(f"Error calculating isolation score: {e}")
            return 0.5

    def _determine_sr_level(self, current_price: float, sr_context: dict[str, Any]) -> float:
        """Determine current SR level position as percentage."""
        try:
            nearest_support = sr_context.get("nearest_support", current_price)
            nearest_resistance = sr_context.get("nearest_resistance", current_price)
            
            if nearest_resistance > nearest_support and current_price > 0:
                # Calculate position as percentage within the SR zone
                support_distance_pct = (current_price - nearest_support) / current_price
                zone_width_pct = (nearest_resistance - nearest_support) / current_price
                return support_distance_pct / zone_width_pct if zone_width_pct > 0 else 0.5
            else:
                return 0.5
                
        except Exception as e:
            self.logger.error(f"Error determining SR level: {e}")
            return 0.5

    def _predict_sr_outcome(self, market_data: pd.DataFrame, sr_context: dict[str, Any]) -> float:
        """Predict SR outcome based on current market conditions as percentage."""
        try:
            current_price = market_data['close'].iloc[-1]
            nearest_support = sr_context.get("nearest_support", current_price)
            nearest_resistance = sr_context.get("nearest_resistance", current_price)
            
            # Calculate outcome as percentage position within SR zone
            if nearest_resistance > nearest_support and current_price > 0:
                support_distance_pct = (current_price - nearest_support) / current_price
                zone_width_pct = (nearest_resistance - nearest_support) / current_price
                return support_distance_pct / zone_width_pct if zone_width_pct > 0 else 0.5
            else:
                return 0.5
                
        except Exception as e:
            self.logger.error(f"Error predicting SR outcome: {e}")
            return 0.5

    async def set_weights(self, weights: dict[str, float]) -> bool:
        """
        Set weights for S/R detection methods.

        Args:
            weights: Dictionary of weights for different detection methods

        Returns:
            bool: True if weights set successfully
        """
        try:
            # Update model weights
            if "fractal_weight" in weights:
                self.model_weights["fractal"] = weights["fractal_weight"]
            if "volume_weight" in weights:
                self.model_weights["volume"] = weights["volume_weight"]
            if "pivot_weight" in weights:
                self.model_weights["pivot"] = weights["pivot_weight"]
            if "atr_weight" in weights:
                self.model_weights["atr"] = weights["atr_weight"]

            # Update strength score weights
            if "touch_count_weight" in weights:
                self.strength_score_weights["touch_count"] = weights["touch_count_weight"]
            if "total_volume_weight" in weights:
                self.strength_score_weights["total_volume"] = weights["total_volume_weight"]
            if "level_age_weight" in weights:
                self.strength_score_weights["level_age"] = weights["level_age_weight"]
            if "bounce_rate_weight" in weights:
                self.strength_score_weights["bounce_rate"] = weights["bounce_rate_weight"]
            if "isolation_score_weight" in weights:
                self.strength_score_weights["isolation_score"] = weights["isolation_score_weight"]

            # Update advanced parameters
            if "fibonacci_sensitivity" in weights:
                self.fibonacci_sensitivity = weights["fibonacci_sensitivity"]
            if "elliott_confidence_threshold" in weights:
                self.elliott_confidence_threshold = weights["elliott_confidence_threshold"]
            if "order_flow_hvn_threshold" in weights:
                self.order_flow_hvn_threshold = weights["order_flow_hvn_threshold"]

            # Update timeframe weights
            timeframe_weights = {}
            for tf in ["1m", "5m", "15m", "1h", "4h", "1d"]:
                weight_key = f"tf_{tf}_weight"
                if weight_key in weights:
                    timeframe_weights[tf] = weights[weight_key]
            
            if timeframe_weights:
                self.timeframe_weights.update(timeframe_weights)

            self.logger.info(f"✅ S/R weights updated: {weights}")
            return True

        except Exception as e:
            self.logger.error(f"Error setting S/R weights: {e}")
            return False

    @validate_data_quality(
        required_columns=["open", "high", "low", "close", "volume"],
        min_rows=50,
        max_null_ratio=0.1,
        check_duplicates=True,
        check_timestamps=True,
        context="Breakout prediction input validation"
    )
    @handle_specific_errors(
        error_handlers={
            ValueError: (None, "Invalid input data for breakout prediction"),
            AttributeError: (None, "Predictor not properly initialized"),
        },
        default_return=None,
        context="Breakout prediction",
    )
    async def predict_breakout(self, market_data: pd.DataFrame) -> dict[str, Any] | None:
        """
        Predict breakout direction and confidence.

        Args:
            market_data: Market data DataFrame

        Returns:
            dict[str, Any]: Breakout prediction or None
        """
        try:
            current_price = market_data['close'].iloc[-1]
            
            # Get S/R context
            sr_context = await self.get_sr_context(market_data, current_price)
            
            # Predict outcome
            outcome = await self.predict_sr_outcome(market_data, current_price, sr_context)
            
            if not outcome:
                return None
            
            # Determine direction based on outcome
            direction = "none"
            if outcome.get("outcome") == "breakout":
                # Determine if breaking up or down
                if sr_context.get("resistance_proximity", 1.0) < sr_context.get("support_proximity", 1.0):
                    direction = "up"  # Breaking resistance
                else:
                    direction = "down"  # Breaking support
            
            return {
                "direction": direction,
                "confidence": outcome.get("confidence", 0.5),
                "price": current_price,
                "outcome": outcome.get("outcome", "consolidation"),
                "sr_context": sr_context
            }

        except Exception as e:
            self.logger.error(f"Error predicting breakout: {e}")
            return None

    async def stop(self) -> None:
        """Stop the SR breakout predictor."""
        try:
            self.logger.info("Stopping SR breakout predictor...")
            self.is_initialized = False
            self.logger.info("✅ SR breakout predictor stopped successfully")
        except Exception as e:
            self.logger.error(f"❌ Failed to stop SR breakout predictor: {e}")

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="SR breakout predictor cleanup",
    )
    async def cleanup(self) -> None:
        """Cleanup SR breakout predictor resources."""
        try:
            self.logger.info("Cleaning up SR breakout predictor...")
            await self.stop()
            self.sr_predictions.clear()
            self.prediction_history.clear()
            self.performance_metrics.clear()
            self.logger.info("✅ SR breakout predictor cleanup completed")
        except Exception as e:
            self.logger.error(f"Error cleaning up SR breakout predictor: {e}")


async def setup_sr_breakout_predictor(
    config: dict[str, Any] | None = None,
) -> SRBreakoutPredictor | None:
    """
    Setup and return a configured SRBreakoutPredictor instance with optimized parameters.

    Args:
        config: Configuration dictionary

    Returns:
        SRBreakoutPredictor: Configured SR breakout predictor instance
    """
    try:
        # Ensure optimized parameters are enabled
        sr_config = config.copy() if config else {}
        sr_config["sr_breakout_predictor"] = sr_config.get("sr_breakout_predictor", {})
        sr_config["sr_breakout_predictor"]["use_optimized_params"] = True
        
        predictor = SRBreakoutPredictor(sr_config)
        if await predictor.initialize():
            return predictor
        return None
    except Exception as e:
        system_logger.exception(f"Failed to setup SR Breakout Predictor: {e}")
        return None


def ensure_optimized_sr_config(config: dict[str, Any]) -> dict[str, Any]:
    """
    Ensure that the configuration has optimized S/R parameters enabled.
    
    Args:
        config: Original configuration dictionary
        
    Returns:
        dict: Configuration with optimized S/R parameters enabled
    """
    sr_config = config.copy()
    sr_config["sr_breakout_predictor"] = sr_config.get("sr_breakout_predictor", {})
    sr_config["sr_breakout_predictor"]["use_optimized_params"] = True
    return sr_config
