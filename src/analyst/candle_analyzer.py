# src/analyst/candle_analyzer.py
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import find_peaks

from src.utils.error_handler import (
    handle_errors,
    handle_specific_errors,
)
from src.utils.logger import system_logger


class CandleAnalyzer:
    """
    Comprehensive candle analyzer with advanced pattern recognition and size classification.
    
    Features:
    - Dynamic large candle detection based on market conditions
    - Candle pattern recognition (doji, hammer, shooting star, etc.)
    - Volatility-based size classification
    - Multi-timeframe analysis
    - Statistical outlier detection
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """
        Initialize candle analyzer with enhanced type safety.

        Args:
            config: Configuration dictionary
        """
        self.config: dict[str, Any] = config
        self.logger = system_logger.getChild("CandleAnalyzer")

        # Analysis state
        self.candle_patterns: list[dict[str, Any]] = []
        self.large_candles: list[dict[str, Any]] = []
        self.last_analysis_time: datetime | None = None
        self.analysis_history: list[dict[str, Any]] = []

        # Configuration
        self.candle_config: dict[str, Any] = self.config.get("candle_analyzer", {})
        
        # Size classification parameters
        self.size_thresholds = self.candle_config.get("size_thresholds", {
            "small": 0.5,      # 0.5x average
            "normal": 1.0,      # 1.0x average
            "large": 2.0,       # 2.0x average
            "huge": 3.0,        # 3.0x average
            "extreme": 5.0      # 5.0x average
        })
        
        # Volatility-based parameters
        self.volatility_period = self.candle_config.get("volatility_period", 20)
        self.volatility_multiplier = self.candle_config.get("volatility_multiplier", 2.0)
        
        # Pattern recognition parameters
        self.doji_threshold = self.candle_config.get("doji_threshold", 0.1)  # 10% of range
        self.hammer_ratio = self.candle_config.get("hammer_ratio", 0.3)      # 30% body
        self.shooting_star_ratio = self.candle_config.get("shooting_star_ratio", 0.3)
        
        # Statistical parameters
        self.outlier_threshold = self.candle_config.get("outlier_threshold", 2.5)  # Standard deviations
        self.min_candle_count = self.candle_config.get("min_candle_count", 100)
        
        # Enhanced configuration
        self.use_adaptive_thresholds = self.candle_config.get("use_adaptive_thresholds", True)
        self.use_volume_confirmation = self.candle_config.get("use_volume_confirmation", True)
        self.use_multi_timeframe = self.candle_config.get("use_multi_timeframe", True)

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="candle analyzer initialization",
    )
    async def initialize(self) -> None:
        """Initialize candle analyzer module."""
        try:
            self.logger.info("Initializing Candle Analyzer...")
            
            # Validate configuration
            self._validate_configuration()
            
            # Initialize analysis parameters
            self._initialize_analysis_parameters()
            
            self.logger.info("Configuration validation successful")
            self.logger.info("Analysis parameters initialized")
            self.logger.info("✅ Candle Analyzer initialization completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing candle analyzer: {e}")
            raise

    def _validate_configuration(self) -> None:
        """Validate candle analyzer configuration."""
        # Check if size_thresholds exists in config
        if "size_thresholds" not in self.candle_config:
            # Use default thresholds if not in config
            self.size_thresholds = {
                "small": 0.5,      # 0.5x average
                "normal": 1.0,      # 1.0x average
                "large": 2.0,       # 2.0x average
                "huge": 3.0,        # 3.0x average
                "extreme": 5.0      # 5.0x average
            }
        
        # Validate size thresholds
        if not all(isinstance(v, (int, float)) for v in self.size_thresholds.values()):
            raise ValueError("Size thresholds must be numeric values")
        
        # Validate periods
        if self.volatility_period < 10:
            raise ValueError("Volatility period must be at least 10")
        
        if self.min_candle_count < 50:
            raise ValueError("Minimum candle count must be at least 50")

    def _initialize_analysis_parameters(self) -> None:
        """Initialize analysis parameters."""
        self.logger.info("Candle configuration loaded successfully")
        self.logger.info("Analysis parameters initialized")

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return={},
        context="candle analysis",
    )
    async def analyze(self, df: pd.DataFrame) -> dict[str, Any]:
        """
        Perform comprehensive candle analysis.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            dict: Candle analysis results
        """
        try:
            self.logger.info("Starting candle analysis...")
            
            # Validate input data
            if df.empty:
                self.logger.warning("Empty data provided")
                return {}
            
            # Ensure required columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                self.logger.warning(f"Missing required columns: {missing_cols}")
                return {}
            
            # Clean and prepare data
            df_clean = self._prepare_data(df)
            if df_clean.empty:
                return {}
            
            # Perform analysis
            results = {
                "candle_sizes": self._analyze_candle_sizes(df_clean),
                "candle_patterns": self._analyze_candle_patterns(df_clean),
                "large_candles": self._detect_large_candles(df_clean),
                "volatility_analysis": self._analyze_volatility(df_clean),
                "statistical_analysis": self._perform_statistical_analysis(df_clean),
                "pattern_summary": self._generate_pattern_summary(df_clean),
                "analysis_timestamp": datetime.now().isoformat()
            }
            
            # Update state
            self.candle_patterns = results["candle_patterns"]
            self.large_candles = results["large_candles"]
            self.last_analysis_time = datetime.now()
            self.analysis_history.append({
                "timestamp": self.last_analysis_time,
                "data_points": len(df_clean),
                "patterns_found": len(results["candle_patterns"]),
                "large_candles_found": len(results["large_candles"])
            })
            
            self.logger.info("✅ Candle analysis completed successfully")
            self.logger.info(f"📊 Candle Analysis Results: {len(results['candle_patterns'])} patterns, {len(results['large_candles'])} large candles")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error during candle analysis: {e}")
            return {}

    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare and clean data for analysis.

        Args:
            df: Raw OHLCV data

        Returns:
            pd.DataFrame: Cleaned data
        """
        try:
            # Create copy to avoid modifying original
            df_clean = df.copy()
            
            # Standardize column names
            column_mapping = {
                'Open': 'open', 'High': 'high', 'Low': 'low', 
                'Close': 'close', 'Volume': 'volume'
            }
            df_clean = df_clean.rename(columns=column_mapping)
            
            # Ensure numeric data
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in df_clean.columns:
                    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
            
            # Remove invalid data
            initial_rows = len(df_clean)
            df_clean = df_clean.dropna(subset=['open', 'high', 'low', 'close'])
            
            # Validate price relationships
            invalid_mask = (
                (df_clean['high'] < df_clean['low']) |
                (df_clean['open'] > df_clean['high']) |
                (df_clean['close'] > df_clean['high']) |
                (df_clean['open'] < df_clean['low']) |
                (df_clean['close'] < df_clean['low'])
            )
            df_clean = df_clean[~invalid_mask]
            
            # Calculate additional metrics
            df_clean['body_size'] = abs(df_clean['close'] - df_clean['open'])
            df_clean['upper_shadow'] = df_clean['high'] - np.maximum(df_clean['open'], df_clean['close'])
            df_clean['lower_shadow'] = np.minimum(df_clean['open'], df_clean['close']) - df_clean['low']
            df_clean['total_range'] = df_clean['high'] - df_clean['low']
            df_clean['body_ratio'] = df_clean['body_size'] / df_clean['total_range'].replace(0, 1)
            df_clean['is_bullish'] = df_clean['close'] > df_clean['open']
            
            # Calculate moving averages for context
            df_clean['avg_body_size'] = df_clean['body_size'].rolling(window=self.volatility_period).mean()
            df_clean['avg_range'] = df_clean['total_range'].rolling(window=self.volatility_period).mean()
            df_clean['volatility'] = df_clean['total_range'].rolling(window=self.volatility_period).std()
            
            # Remove rows with insufficient data for moving averages
            df_clean = df_clean.dropna()
            
            final_rows = len(df_clean)
            if final_rows < self.min_candle_count:
                self.logger.warning(f"Insufficient data after cleaning: {final_rows} < {self.min_candle_count}")
                return pd.DataFrame()
            
            self.logger.info(f"Data preparation: {initial_rows} → {final_rows} rows")
            return df_clean
            
        except Exception as e:
            self.logger.error(f"Error preparing data: {e}")
            return pd.DataFrame()

    def _analyze_candle_sizes(self, df: pd.DataFrame) -> dict[str, Any]:
        """
        Analyze candle sizes and classify them.

        Args:
            df: Cleaned OHLCV data

        Returns:
            dict: Candle size analysis results
        """
        try:
            # Calculate size ratios
            body_size_ratio = df['body_size'] / df['avg_body_size'].replace(0, 1)
            range_ratio = df['total_range'] / df['avg_range'].replace(0, 1)
            
            # Classify candles
            size_classification = []
            for i, (body_ratio, range_ratio_val) in enumerate(zip(body_size_ratio, range_ratio)):
                if range_ratio_val >= self.size_thresholds["extreme"]:
                    size_class = "extreme"
                elif range_ratio_val >= self.size_thresholds["huge"]:
                    size_class = "huge"
                elif range_ratio_val >= self.size_thresholds["large"]:
                    size_class = "large"
                elif range_ratio_val >= self.size_thresholds["normal"]:
                    size_class = "normal"
                else:
                    size_class = "small"
                
                size_classification.append({
                    "index": i,
                    "timestamp": df.index[i] if i < len(df.index) else None,
                    "size_class": size_class,
                    "body_ratio": body_ratio,
                    "range_ratio": range_ratio_val,
                    "body_size": df['body_size'].iloc[i],
                    "total_range": df['total_range'].iloc[i],
                    "is_bullish": df['is_bullish'].iloc[i]
                })
            
            # Calculate statistics
            size_stats = {
                "small_count": len([c for c in size_classification if c["size_class"] == "small"]),
                "normal_count": len([c for c in size_classification if c["size_class"] == "normal"]),
                "large_count": len([c for c in size_classification if c["size_class"] == "large"]),
                "huge_count": len([c for c in size_classification if c["size_class"] == "huge"]),
                "extreme_count": len([c for c in size_classification if c["size_class"] == "extreme"]),
                "avg_body_ratio": np.mean(body_size_ratio),
                "avg_range_ratio": np.mean(range_ratio),
                "max_range_ratio": np.max(range_ratio),
                "min_range_ratio": np.min(range_ratio)
            }
            
            return {
                "classification": size_classification,
                "statistics": size_stats,
                "thresholds": self.size_thresholds
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing candle sizes: {e}")
            return {}

    def _analyze_candle_patterns(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        """
        Analyze and identify candle patterns.

        Args:
            df: Cleaned OHLCV data

        Returns:
            list: Identified candle patterns
        """
        try:
            patterns = []
            
            for i in range(len(df)):
                row = df.iloc[i]
                
                # Doji pattern (very small body)
                if row['body_ratio'] <= self.doji_threshold:
                    patterns.append({
                        "index": i,
                        "timestamp": df.index[i],
                        "pattern": "doji",
                        "confidence": 1.0 - row['body_ratio'],
                        "body_ratio": row['body_ratio'],
                        "is_bullish": row['is_bullish']
                    })
                
                # Hammer pattern (small body, long lower shadow)
                elif (row['body_ratio'] <= self.hammer_ratio and 
                      row['lower_shadow'] > 2 * row['body_size'] and
                      row['upper_shadow'] < row['body_size']):
                    patterns.append({
                        "index": i,
                        "timestamp": df.index[i],
                        "pattern": "hammer",
                        "confidence": 0.8,
                        "body_ratio": row['body_ratio'],
                        "is_bullish": row['is_bullish']
                    })
                
                # Shooting star pattern (small body, long upper shadow)
                elif (row['body_ratio'] <= self.shooting_star_ratio and
                      row['upper_shadow'] > 2 * row['body_size'] and
                      row['lower_shadow'] < row['body_size']):
                    patterns.append({
                        "index": i,
                        "timestamp": df.index[i],
                        "pattern": "shooting_star",
                        "confidence": 0.8,
                        "body_ratio": row['body_ratio'],
                        "is_bullish": row['is_bullish']
                    })
                
                # Marubozu pattern (no shadows)
                elif (row['upper_shadow'] < 0.1 * row['total_range'] and
                      row['lower_shadow'] < 0.1 * row['total_range']):
                    patterns.append({
                        "index": i,
                        "timestamp": df.index[i],
                        "pattern": "marubozu",
                        "confidence": 0.9,
                        "body_ratio": row['body_ratio'],
                        "is_bullish": row['is_bullish']
                    })
                
                # Spinning top pattern (small body, equal shadows)
                elif (row['body_ratio'] <= 0.3 and
                      abs(row['upper_shadow'] - row['lower_shadow']) < 0.2 * row['total_range']):
                    patterns.append({
                        "index": i,
                        "timestamp": df.index[i],
                        "pattern": "spinning_top",
                        "confidence": 0.7,
                        "body_ratio": row['body_ratio'],
                        "is_bullish": row['is_bullish']
                    })
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error analyzing candle patterns: {e}")
            return []

    def _detect_large_candles(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        """
        Detect and analyze large candles.

        Args:
            df: Cleaned OHLCV data

        Returns:
            list: Large candle analysis results
        """
        try:
            large_candles = []
            
            # Calculate adaptive thresholds based on volatility
            if self.use_adaptive_thresholds:
                volatility_factor = df['volatility'].iloc[-1] / df['volatility'].mean()
                adaptive_thresholds = {
                    k: v * volatility_factor for k, v in self.size_thresholds.items()
                }
            else:
                adaptive_thresholds = self.size_thresholds
            
            # Detect large candles
            for i in range(len(df)):
                row = df.iloc[i]
                range_ratio = row['total_range'] / row['avg_range']
                
                if range_ratio >= adaptive_thresholds["large"]:
                    # Calculate additional metrics
                    volume_ratio = row['volume'] / df['volume'].rolling(window=self.volatility_period).mean().iloc[i] if self.use_volume_confirmation else 1.0
                    
                    large_candles.append({
                        "index": i,
                        "timestamp": df.index[i],
                        "size_class": "large" if range_ratio < adaptive_thresholds["huge"] else "huge",
                        "range_ratio": range_ratio,
                        "body_ratio": row['body_ratio'],
                        "volume_ratio": volume_ratio,
                        "body_size": row['body_size'],
                        "total_range": row['total_range'],
                        "upper_shadow": row['upper_shadow'],
                        "lower_shadow": row['lower_shadow'],
                        "is_bullish": row['is_bullish'],
                        "volatility": row['volatility'],
                        "confidence": min(range_ratio / adaptive_thresholds["large"], 1.0)
                    })
            
            return large_candles
            
        except Exception as e:
            self.logger.error(f"Error detecting large candles: {e}")
            return []

    def _analyze_volatility(self, df: pd.DataFrame) -> dict[str, Any]:
        """
        Analyze volatility patterns.

        Args:
            df: Cleaned OHLCV data

        Returns:
            dict: Volatility analysis results
        """
        try:
            volatility_analysis = {
                "current_volatility": df['volatility'].iloc[-1],
                "avg_volatility": df['volatility'].mean(),
                "volatility_percentile": stats.percentileofscore(df['volatility'], df['volatility'].iloc[-1]),
                "volatility_trend": "increasing" if df['volatility'].iloc[-1] > df['volatility'].iloc[-2] else "decreasing",
                "high_volatility_periods": len(df[df['volatility'] > df['volatility'].quantile(0.8)]),
                "low_volatility_periods": len(df[df['volatility'] < df['volatility'].quantile(0.2)])
            }
            
            return volatility_analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing volatility: {e}")
            return {}

    def _perform_statistical_analysis(self, df: pd.DataFrame) -> dict[str, Any]:
        """
        Perform statistical analysis on candle data.

        Args:
            df: Cleaned OHLCV data

        Returns:
            dict: Statistical analysis results
        """
        try:
            # Calculate z-scores for outlier detection
            range_z_scores = stats.zscore(df['total_range'])
            body_z_scores = stats.zscore(df['body_size'])
            
            # Detect outliers
            range_outliers = df[abs(range_z_scores) > self.outlier_threshold]
            body_outliers = df[abs(body_z_scores) > self.outlier_threshold]
            
            statistical_analysis = {
                "range_statistics": {
                    "mean": df['total_range'].mean(),
                    "std": df['total_range'].std(),
                    "min": df['total_range'].min(),
                    "max": df['total_range'].max(),
                    "outliers_count": len(range_outliers)
                },
                "body_statistics": {
                    "mean": df['body_size'].mean(),
                    "std": df['body_size'].std(),
                    "min": df['body_size'].min(),
                    "max": df['body_size'].max(),
                    "outliers_count": len(body_outliers)
                },
                "correlation_analysis": {
                    "body_range_correlation": df['body_size'].corr(df['total_range']),
                    "volume_range_correlation": df['volume'].corr(df['total_range']),
                    "body_volume_correlation": df['body_size'].corr(df['volume'])
                },
                "distribution_analysis": {
                    "range_skewness": stats.skew(df['total_range']),
                    "body_skewness": stats.skew(df['body_size']),
                    "range_kurtosis": stats.kurtosis(df['total_range']),
                    "body_kurtosis": stats.kurtosis(df['body_size'])
                }
            }
            
            return statistical_analysis
            
        except Exception as e:
            self.logger.error(f"Error performing statistical analysis: {e}")
            return {}

    def _generate_pattern_summary(self, df: pd.DataFrame) -> dict[str, Any]:
        """
        Generate summary of candle patterns and analysis.

        Args:
            df: Cleaned OHLCV data

        Returns:
            dict: Pattern summary
        """
        try:
            # Count patterns
            pattern_counts = {}
            for pattern in self.candle_patterns:
                pattern_type = pattern["pattern"]
                pattern_counts[pattern_type] = pattern_counts.get(pattern_type, 0) + 1
            
            # Analyze large candles
            large_candle_summary = {
                "total_count": len(self.large_candles),
                "bullish_count": len([c for c in self.large_candles if c["is_bullish"]]),
                "bearish_count": len([c for c in self.large_candles if not c["is_bullish"]]),
                "avg_range_ratio": np.mean([c["range_ratio"] for c in self.large_candles]) if self.large_candles else 0,
                "max_range_ratio": np.max([c["range_ratio"] for c in self.large_candles]) if self.large_candles else 0
            }
            
            # Market context
            recent_data = df.tail(20)
            market_context = {
                "recent_avg_range": recent_data['total_range'].mean(),
                "recent_avg_body": recent_data['body_size'].mean(),
                "recent_volatility": recent_data['volatility'].mean(),
                "bullish_candle_ratio": (recent_data['is_bullish'].sum() / len(recent_data)),
                "trend_strength": abs(recent_data['close'].iloc[-1] - recent_data['close'].iloc[0]) / recent_data['close'].iloc[0]
            }
            
            return {
                "pattern_counts": pattern_counts,
                "large_candle_summary": large_candle_summary,
                "market_context": market_context,
                "analysis_quality": {
                    "data_points": len(df),
                    "patterns_found": len(self.candle_patterns),
                    "large_candles_found": len(self.large_candles),
                    "confidence_score": min(len(df) / 1000, 1.0)  # Higher confidence with more data
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error generating pattern summary: {e}")
            return {}

    def detect_large_candle(self, current_candle: dict[str, Any]) -> dict[str, Any]:
        """
        Detect if current candle is large based on historical context.

        Args:
            current_candle: Current candle data (OHLCV)

        Returns:
            dict: Large candle detection result
        """
        try:
            if not self.large_candles:
                return {
                    "is_large": False,
                    "confidence": 0.0,
                    "size_class": "unknown",
                    "reason": "No historical data available"
                }
            
            # Calculate current candle metrics
            body_size = abs(current_candle['close'] - current_candle['open'])
            total_range = current_candle['high'] - current_candle['low']
            body_ratio = body_size / total_range if total_range > 0 else 0
            
            # Get historical context from last analysis
            if hasattr(self, '_last_analysis_df') and self._last_analysis_df is not None:
                avg_range = self._last_analysis_df['total_range'].mean()
                avg_body = self._last_analysis_df['body_size'].mean()
                volatility = self._last_analysis_df['volatility'].iloc[-1] if len(self._last_analysis_df) > 0 else 0
            else:
                # Use default values if no historical data
                avg_range = total_range
                avg_body = body_size
                volatility = 0
            
            # Calculate ratios
            range_ratio = total_range / avg_range if avg_range > 0 else 1
            body_ratio_historical = body_size / avg_body if avg_body > 0 else 1
            
            # Determine size class
            if range_ratio >= self.size_thresholds["extreme"]:
                size_class = "extreme"
                confidence = min(range_ratio / self.size_thresholds["extreme"], 1.0)
            elif range_ratio >= self.size_thresholds["huge"]:
                size_class = "huge"
                confidence = min(range_ratio / self.size_thresholds["huge"], 1.0)
            elif range_ratio >= self.size_thresholds["large"]:
                size_class = "large"
                confidence = min(range_ratio / self.size_thresholds["large"], 1.0)
            else:
                size_class = "normal"
                confidence = 0.0
            
            return {
                "is_large": size_class in ["large", "huge", "extreme"],
                "confidence": confidence,
                "size_class": size_class,
                "range_ratio": range_ratio,
                "body_ratio": body_ratio,
                "body_ratio_historical": body_ratio_historical,
                "volatility": volatility,
                "is_bullish": current_candle['close'] > current_candle['open'],
                "reason": f"Range ratio: {range_ratio:.2f}x average"
            }
            
        except Exception as e:
            self.logger.error(f"Error detecting large candle: {e}")
            return {
                "is_large": False,
                "confidence": 0.0,
                "size_class": "error",
                "reason": f"Error: {str(e)}"
            }

    def get_analysis_status(self) -> dict[str, Any]:
        """Get analysis status and statistics."""
        return {
            "last_analysis_time": self.last_analysis_time.isoformat() if self.last_analysis_time else None,
            "patterns_found": len(self.candle_patterns),
            "large_candles_found": len(self.large_candles),
            "analysis_history_count": len(self.analysis_history),
            "configuration": {
                "size_thresholds": self.size_thresholds,
                "volatility_period": self.volatility_period,
                "outlier_threshold": self.outlier_threshold,
                "use_adaptive_thresholds": self.use_adaptive_thresholds,
                "use_volume_confirmation": self.use_volume_confirmation
            }
        } 