#    sr_analyzer.py
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import find_peaks


from src.utils.error_handler import (
    handle_errors,
    handle_specific_errors,
)
from src.utils.logger import system_logger


class SRLevelAnalyzer:
    """
    Enhanced support/resistance level analyzer with comprehensive error handling and type safety.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """
        Initialize SR level analyzer with enhanced type safety.

        Args:
            config: Configuration dictionary
        """
        self.config: dict[str, Any] = config
        self.logger = system_logger.getChild("SRLevelAnalyzer")

        # Analysis state
        self.support_levels: list[dict[str, Any]] = []
        self.resistance_levels: list[dict[str, Any]] = []
        self.last_analysis_time: datetime | None = None
        self.analysis_history: list[dict[str, Any]] = []

        # Configuration
        self.sr_config: dict[str, Any] = self.config.get("sr_analyzer", {})
        self.min_touch_count: int = self.sr_config.get("min_touch_count", 2)
        self.lookback_period: int = self.sr_config.get("lookback_period", 100)

        self.strength_weights: dict[str, float] = self.sr_config.get(
            "strength_weights", {"touches": 0.6, "recency": 0.4}
        )
        self.consolidation_tolerance: float = self.sr_config.get("consolidation_tolerance", 0.0075)  # 0.75% tolerance as requested
        
        # Enhanced configuration
        self.use_time_decay: bool = self.sr_config.get("use_time_decay", True)

    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid SR analyzer configuration"),
            AttributeError: (False, "Missing required SR parameters"),
            KeyError: (False, "Missing configuration keys"),
        },
        default_return=False,
        context="SR analyzer initialization",
    )
    async def initialize(self) -> bool:
        """
        Initialize SR analyzer with enhanced error handling.

        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            self.logger.info("Initializing SR Level Analyzer...")

            # Load SR configuration
            await self._load_sr_configuration()

            # Initialize analysis parameters
            await self._initialize_analysis_parameters()

            # Validate configuration
            if not self._validate_configuration():
                self.logger.error("Invalid configuration for SR analyzer")
                return False

            self.logger.info(
                "✅ SR Level Analyzer initialization completed successfully",
            )
            return True

        except Exception as e:
            self.logger.error(f"❌ SR Level Analyzer initialization failed: {e}")
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="SR configuration loading",
    )
    async def _load_sr_configuration(self) -> None:
        """Load SR analysis configuration."""
        try:
            # Set default SR parameters
            self.sr_config.setdefault("min_touch_count", 2)
            self.sr_config.setdefault("lookback_period", 100)
            self.sr_config.setdefault("price_tolerance", 0.0006)
            self.sr_config.setdefault(
                "strength_weights",
                {"touches": 0.6, "recency": 0.4},
            )
            self.logger.info("SR configuration loaded successfully")

        except Exception as e:
            self.logger.error(f"Error loading SR configuration: {e}")

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="analysis parameters initialization",
    )
    async def _initialize_analysis_parameters(self) -> None:
        """Initialize analysis parameters."""
        try:
            # Initialize analysis parameters
            self.min_touch_count = self.sr_config["min_touch_count"]
            self.lookback_period = self.sr_config["lookback_period"]
            self.logger.info("Analysis parameters initialized")

        except Exception as e:
            self.logger.error(f"Error initializing analysis parameters: {e}")

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="configuration validation",
    )
    def _validate_configuration(self) -> bool:
        """
        Validate SR analyzer configuration.

        Returns:
            bool: True if configuration is valid, False otherwise
        """
        try:
            required_keys = [
                "min_touch_count",
                "lookback_period",
            ]
            for key in required_keys:
                if key not in self.sr_config:
                    self.logger.error(f"Missing required SR configuration key: {key}")
                    return False

            # Validate parameter ranges
            if self.min_touch_count < 2:
                self.logger.error("min_touch_count must be at least 2")
                return False

            if self.lookback_period < 10:
                self.logger.error("lookback_period must be at least 10")
                return False

            self.logger.info("Configuration validation successful")
            return True

        except Exception as e:
            self.logger.error(f"Error validating configuration: {e}")
            return False

    @handle_specific_errors(
        error_handlers={
            ConnectionError: (None, "Failed to connect to data source"),
            TimeoutError: (None, "SR analysis timed out"),
            ValueError: (None, "Invalid market data"),
        },
        default_return=None,
        context="SR analysis",
    )
    async def analyze(self, data: pd.DataFrame) -> dict[str, Any] | None:
        """
        Analyze support and resistance levels with enhanced error handling.

        Args:
            data: Market data DataFrame with OHLCV columns

        Returns:
            Optional[Dict[str, Any]]: Analysis results or None if failed
        """
        try:
            if data.empty:
                self.logger.error("Empty data provided for SR analysis")
                return None

            self.logger.info("Starting SR level analysis...")

            # Validate data structure
            if not self._validate_data_structure(data):
                self.logger.error("Invalid data structure for SR analysis")
                return None

            # Perform enhanced SR analysis
            support_levels = await self._identify_support_levels(data)
            resistance_levels = await self._identify_resistance_levels(data)
            
            # Add enhanced detection methods
            all_levels = []
            
            # Add traditional levels
            if support_levels:
                all_levels.extend(support_levels)
            if resistance_levels:
                all_levels.extend(resistance_levels)
            
            # Use VPVR (Volume Profile Visible Range) for S/R detection
            vpvr_levels = self._calculate_vpvr_sr_levels(data)
            all_levels.extend(vpvr_levels)
            self.logger.info(f"SR: VPVR levels found: {len(vpvr_levels)}")
            
            # Log the actual VPVR price levels
            if vpvr_levels:
                self.logger.info("SR: VPVR Price Levels (ALL):")
                for i, level in enumerate(vpvr_levels):
                    self.logger.info(f"  {i+1:3d}. Price: ${level['price']:.2f}, Type: {level['type']}, Strength: {level['strength']:.3f}, Method: {level['method']}")
                self.logger.info(f"SR: Total VPVR levels found: {len(vpvr_levels)}")

            # Consolidate all detected levels
            all_levels = self._consolidate_levels(all_levels)
            self.logger.info(f"SR: Levels after consolidation: {len(all_levels)}")
            
            # Filter out lower and very low strength levels - keep only medium and high strength (>=0.4)
            all_levels = [level for level in all_levels if level.get('strength', 0) >= 0.4]
            self.logger.info(f"SR: Levels after strength filtering (>=0.4): {len(all_levels)}")
            
            # No top-N limit - let strength threshold determine the number
            self.logger.info(f"SR: Levels after strength-based filtering: {len(all_levels)}")
            
            # Skip time decay for now since volume-based levels are handled elsewhere
            # all_levels = self._apply_time_decay(all_levels, data)
            self.logger.info(f"SR: Levels after consolidation: {len(all_levels)} (time decay skipped)")
            
            # Log the final consolidated levels
            if all_levels:
                self.logger.info("SR: Final Consolidated Price Levels (ALL):")
                for i, level in enumerate(all_levels):
                    self.logger.info(f"  {i+1:3d}. Price: ${level['price']:.2f}, Type: {level['type']}, Strength: {level['strength']:.3f}")
                self.logger.info(f"SR: Total consolidated levels: {len(all_levels)}")
            
            # Separate levels by type (include VPVR levels)
            support_levels = [level for level in all_levels if level.get('type') == 'support']
            resistance_levels = [level for level in all_levels if level.get('type') == 'resistance']

            # Calculate confidence scores
            support_confidence = self._calculate_level_confidence(support_levels, data)
            resistance_confidence = self._calculate_level_confidence(
                resistance_levels,
                data,
            )

            # Generate analysis results
            analysis_result = {
                "support_levels": support_levels,
                "resistance_levels": resistance_levels,
                "support_confidence": support_confidence,
                "resistance_confidence": resistance_confidence,
                "analysis_time": datetime.now(),
                "data_points_analyzed": len(data),
                "lookback_period": self.lookback_period,
            }

            # Update state
            self.support_levels = support_levels
            self.resistance_levels = resistance_levels
            self.last_analysis_time = datetime.now()
            self.analysis_history.append(analysis_result)

            # Log the number of SR levels detected
            total_levels = len(self.support_levels) + len(self.resistance_levels)
            support_count = len(self.support_levels)
            resistance_count = len(self.resistance_levels)
            
            self.logger.info(f"✅ SR analysis completed successfully")
            self.logger.info(f"📊 SR Levels Detected: {total_levels} total ({support_count} support, {resistance_count} resistance)")
            
            return analysis_result

        except Exception as e:
            self.logger.error(f"Error performing SR analysis: {e}")
            return None

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="data structure validation",
    )
    def _validate_data_structure(self, data: pd.DataFrame) -> bool:
        """
        Validate data structure for SR analysis.

        Args:
            data: DataFrame to validate

        Returns:
            bool: True if valid, False otherwise
        """
        try:
            required_columns = ["open", "high", "low", "close"]
            missing_columns = [
                col for col in required_columns if col not in data.columns
            ]

            if missing_columns:
                self.logger.error(f"Missing required columns: {missing_columns}")
                return False

            # Check for sufficient data
            if len(data) < self.lookback_period:
                self.logger.warning(
                    f"Insufficient data: {len(data)} < {self.lookback_period}",
                )
                return False

            # Check for valid price data
            if (data[["open", "high", "low", "close"]] <= 0).any().any():
                self.logger.error("Invalid price data (non-positive values)")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Error validating data structure: {e}")
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="support level identification",
    )
    async def _identify_support_levels(
        self,
        data: pd.DataFrame,
    ) -> list[dict[str, Any]] | None:
        """
        Identify support levels in the data using peak detection.

        Args:
            data: Market data DataFrame

        Returns:
            Optional[List[Dict[str, Any]]]: Support levels or None
        """
        try:
            support_levels = []
            recent_data = data.tail(self.lookback_period)

            # Find local minima (valleys) by finding peaks in the negated 'low' series
            peaks, _ = find_peaks(-recent_data["low"], distance=40, prominence=0.014)  # Much more restrictive

            for i in peaks:
                price = recent_data.iloc[i]["low"]
                support_level = {
                    "price": price,
                    "timestamp": recent_data.index[i],
                    "touch_count": self._count_touches(
                        recent_data,
                        price,
                        "support",
                    ),
                    "strength": self._calculate_level_strength(
                        recent_data,
                        price,
                        "support",
                    ),
                }

                if support_level["touch_count"] >= self.min_touch_count:
                    support_levels.append(support_level)

            # Sort by strength
            support_levels.sort(key=lambda x: x["strength"], reverse=True)

            return support_levels

        except Exception as e:
            self.logger.error(f"Error identifying support levels: {e}")
            return None

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="resistance level identification",
    )
    async def _identify_resistance_levels(
        self,
        data: pd.DataFrame,
    ) -> list[dict[str, Any]] | None:
        """
        Identify resistance levels in the data using peak detection.

        Args:
            data: Market data DataFrame

        Returns:
            Optional[List[Dict[str, Any]]]: Resistance levels or None
        """
        try:
            resistance_levels = []
            recent_data = data.tail(self.lookback_period)

            # Find local maxima (peaks) in the 'high' series
            peaks, _ = find_peaks(recent_data["high"], distance=40, prominence=0.014)  # Much more restrictive

            for i in peaks:
                price = recent_data.iloc[i]["high"]
                resistance_level = {
                    "price": price,
                    "timestamp": recent_data.index[i],
                    "touch_count": self._count_touches(
                        recent_data,
                        price,
                        "resistance",
                    ),
                    "strength": self._calculate_level_strength(
                        recent_data,
                        price,
                        "resistance",
                    ),
                }

                if resistance_level["touch_count"] >= self.min_touch_count:
                    resistance_levels.append(resistance_level)

            # Sort by strength
            resistance_levels.sort(key=lambda x: x["strength"], reverse=True)

            return resistance_levels

        except Exception as e:
            self.logger.error(f"Error identifying resistance levels: {e}")
            return None

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=0,
        context="touch count calculation",
    )
    def _count_touches(
        self,
        data: pd.DataFrame,
        level_price: float,
        level_type: str,
    ) -> int:
        """
        Count how many times a price level was touched.

        Args:
            data: Market data
            level_price: Price level to check
            level_type: 'support' or 'resistance'

        Returns:
            int: Number of touches
        """
        try:
            tolerance = self.sr_config.get("price_tolerance", 0.001)
            touch_count = 0

            for _, row in data.iterrows():
                if level_type == "support":
                    # Check if low price touched the support level
                    if abs(row["low"] - level_price) <= tolerance * level_price:
                        touch_count += 1
                # Check if high price touched the resistance level
                elif abs(row["high"] - level_price) <= tolerance * level_price:
                    touch_count += 1

            return touch_count

        except Exception as e:
            self.logger.error(f"Error counting touches: {e}")
            return 0

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=0.0,
        context="level strength calculation",
    )
    def _calculate_level_strength(
        self,
        data: pd.DataFrame,
        level_price: float,
        level_type: str,
    ) -> float:
        """
        Calculate the strength of a support/resistance level based on touch count and recency.

        Args:
            data: Market data
            level_price: Price level
            level_type: 'support' or 'resistance'

        Returns:
            float: Level strength score
        """
        try:
            tolerance = self.sr_config.get("price_tolerance", 0.001)
            touch_count = 0
            recent_touches = 0

            for i, (_, row) in enumerate(data.iterrows()):
                if level_type == "support":
                    if abs(row["low"] - level_price) <= tolerance * level_price:
                        touch_count += 1
                        # Give more weight to recent touches
                        if i >= len(data) - 20:  # Last 20 periods
                            recent_touches += 1
                elif abs(row["high"] - level_price) <= tolerance * level_price:
                    touch_count += 1
                    # Give more weight to recent touches
                    if i >= len(data) - 20:  # Last 20 periods
                        recent_touches += 1

            # Calculate strength based on touch count and recency
            base_strength = touch_count * 0.5
            recency_bonus = recent_touches * 0.3
            strength = base_strength + recency_bonus

            return strength

        except Exception as e:
            self.logger.error(f"Error calculating level strength: {e}")
            return 0.0

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=0.0,
        context="confidence calculation",
    )
    def _calculate_level_confidence(
        self,
        levels: list[dict[str, Any]],
        data: pd.DataFrame,
    ) -> float:
        """
        Calculate confidence score for identified levels.

        Args:
            levels: List of support/resistance levels
            data: Market data

        Returns:
            float: Confidence score between 0 and 1
        """
        try:
            if not levels:
                return 0.0

            # Calculate average strength
            avg_strength = np.mean([level["strength"] for level in levels])

            # Calculate average touch count
            avg_touches = np.mean([level["touch_count"] for level in levels])

            # Normalize to 0-1 range
            strength_score = min(avg_strength / 10.0, 1.0)  # Normalize strength
            touch_score = min(avg_touches / 10.0, 1.0)  # Normalize touches

            # Weighted combination
            confidence = (strength_score * 0.6) + (touch_score * 0.4)

            return max(0.0, min(1.0, confidence))

        except Exception as e:
            self.logger.error(f"Error calculating confidence: {e}")
            return 0.0

    def get_support_levels(self) -> list[dict[str, Any]]:
        """
        Get current support levels.

        Returns:
            List[Dict[str, Any]]: Support levels
        """
        return self.support_levels.copy()

    def get_resistance_levels(self) -> list[dict[str, Any]]:
        """
        Get current resistance levels.

        Returns:
            List[Dict[str, Any]]: Resistance levels
        """
        return self.resistance_levels.copy()

    def get_last_analysis_time(self) -> datetime | None:
        """
        Get last analysis time.

        Returns:
            Optional[datetime]: Last analysis time or None
        """
        return self.last_analysis_time

    def get_analysis_history(self) -> list[dict[str, Any]]:
        """
        Get analysis history.

        Returns:
            List[Dict[str, Any]]: Analysis history
        """
        return self.analysis_history.copy()

    def _consolidate_levels(self, levels: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Consolidate nearby levels into a single, stronger level.
        
        Args:
            levels: List of S/R levels
            
        Returns:
            List of consolidated levels
        """
        if not levels:
            return []

        sorted_levels = sorted(levels, key=lambda x: x['price'])
        consolidated = []
        
        current_group = [sorted_levels[0]]
        
        for i in range(1, len(sorted_levels)):
            price_diff = abs(sorted_levels[i]['price'] - current_group[-1]['price']) / current_group[-1]['price']
            
            if price_diff <= self.consolidation_tolerance:
                current_group.append(sorted_levels[i])
            else:
                # Consolidate the current group
                if len(current_group) > 1:
                    # Weighted average price (by strength)
                    total_strength = sum(level.get('strength', 0) for level in current_group)
                    avg_price = sum(level['price'] * level.get('strength', 0) for level in current_group) / max(total_strength, 0.001)
                    
                    # Combine attributes
                    combined_level = {
                        'price': avg_price,
                        'type': current_group[0].get('type', 'traditional'),
                        'strength': total_strength,
                        'touch_count': sum(level.get('touch_count', 1) for level in current_group),
                        'consolidated': True,
                        'source_levels': current_group
                    }
                    consolidated.append(combined_level)
                else:
                    # Ensure the single level has all required fields
                    level = current_group[0].copy()
                    if 'type' not in level:
                        level['type'] = 'traditional'
                    if 'strength' not in level:
                        level['strength'] = 0.5
                    if 'touch_count' not in level:
                        level['touch_count'] = 1
                    consolidated.append(level)
                
                current_group = [sorted_levels[i]]
        
        # Consolidate the last group
        if len(current_group) > 1:
            total_strength = sum(level.get('strength', 0) for level in current_group)
            avg_price = sum(level['price'] * level.get('strength', 0) for level in current_group) / max(total_strength, 0.001)
            consolidated.append({
                'price': avg_price,
                'type': current_group[0].get('type', 'traditional'),
                'strength': total_strength,
                'touch_count': sum(level.get('touch_count', 1) for level in current_group),
                'consolidated': True,
                'source_levels': current_group
            })
        else:
            # Ensure the single level has all required fields
            level = current_group[0].copy()
            if 'type' not in level:
                level['type'] = 'traditional'
            if 'strength' not in level:
                level['strength'] = 0.5
            if 'touch_count' not in level:
                level['touch_count'] = 1
            consolidated.append(level)
            
        return consolidated

    def _calculate_vpvr_sr_levels(self, data: pd.DataFrame) -> list[dict[str, Any]]:
        """
        Calculate S/R levels using VPVR (Volume Profile Visible Range).
        
        Args:
            data: Market data DataFrame
            
        Returns:
            List of S/R levels based on VPVR
        """
        try:
            # Mocking this import as the actual file is not available
            # from src.analyst.data_utils import calculate_volume_profile
            
            def calculate_volume_profile(df, num_bins):
                # This is a mock function. Replace with your actual implementation.
                print("MOCK: Calculating volume profile...")
                poc = df['Close'].mode()[0]
                hvn_price = df['Close'].quantile(0.75)
                return {
                    'poc': poc,
                    'hvn_results': [{'price': hvn_price, 'strength': 0.75, 'volume_concentration': 0.1}]
                }

            # Prepare data for VPVR calculation
            vpvr_data = data.copy()
            vpvr_data.columns = [col.capitalize() for col in vpvr_data.columns]
            
            # Debug: Check the actual price data
            print(f"SR Debug - Data shape: {vpvr_data.shape}")
            print(f"SR Debug - Columns: {vpvr_data.columns.tolist()}")
            print(f"SR Debug - Price range: {vpvr_data['Low'].min():.2f} to {vpvr_data['High'].max():.2f}")
            print(f"SR Debug - Close range: {vpvr_data['Close'].min():.2f} to {vpvr_data['Close'].max():.2f}")
            print(f"SR Debug - Median Close: {vpvr_data['Close'].median():.2f}")
            print(f"SR Debug - Sample prices: {vpvr_data['Close'].head().tolist()}")
            
            # Fix corrupted price data - ETH should be around $3000-4000
            median_price = vpvr_data['Close'].median()
            if median_price > 10000:  # If median is too high, prices are corrupted
                print(f"ERROR - sr_analyzer.py/ _calculate_vpvr_sr_levels - Detected corrupted prices (median: {median_price:.2f})")
            
            # Calculate volume profile using data_utils with more bins for more granular detection
            volume_profile_result = calculate_volume_profile(vpvr_data, num_bins=150)
            
            vpvr_levels = []
            current_price = data['close'].iloc[-1]
            
            # Add Point of Control (POC) as a strong S/R level
            if not pd.isna(volume_profile_result['poc']):
                poc_price = volume_profile_result['poc']
                level_type = 'support' if poc_price < current_price else 'resistance'
                vpvr_levels.append({
                    'price': poc_price,
                    'type': level_type,
                    'strength': 0.9,  # POC is very strong
                    'method': 'vpvr_poc',
                    'touch_count': 0
                })
            
            # Add High Volume Nodes (HVNs) as S/R levels with strength information
            if 'hvn_results' in volume_profile_result:
                # Use the new detailed HVN results with strength
                for hvn_result in volume_profile_result['hvn_results']:
                    hvn_price = hvn_result['price']
                    level_type = 'support' if hvn_price < current_price else 'resistance'
                    vpvr_levels.append({
                        'price': hvn_price,
                        'type': level_type,
                        'strength': hvn_result['strength'],  # Use calculated strength
                        'method': 'vpvr_hvn',
                        'volume_concentration': hvn_result['volume_concentration'],
                        'touch_count': 0
                    })
            
            return vpvr_levels
            
        except Exception as e:
            self.logger.error(f"Error calculating VPVR S/R levels: {e}")
            return []

    def _apply_time_decay(self, levels: list[dict[str, Any]], data: pd.DataFrame) -> list[dict[str, Any]]:
        """
        Apply time decay to level strength based on recency.
        
        Args:
            levels: List of S/R levels
            data: Market data DataFrame
            
        Returns:
            List of levels with time-decayed strength
        """
        try:
            if not self.use_time_decay:
                return levels
                
            current_time = data.index[-1]
            decayed_levels = []
            
            for level in levels:
                if 'timestamp' in level:
                    time_diff = (current_time - level['timestamp']).total_seconds() / 3600  # hours
                    decay_factor = np.exp(-time_diff / 168)  # 1 week half-life
                    level['strength'] *= decay_factor
                    
                decayed_levels.append(level)
                
            return decayed_levels
            
        except Exception as e:
            self.logger.error(f"Error applying time decay: {e}")
            return levels

    def detect_sr_zone_proximity(
        self,
        current_price: float,
        tolerance_percent: float = 0.02,
    ) -> dict[str, Any]:
        """
        Detect if current price is near support/resistance levels with enhanced detection.
        
        Args:
            current_price: Current market price
            tolerance_percent: Percentage tolerance for proximity detection (default 2%)
            
        Returns:
            dict: Contains 'in_zone', 'nearest_level', 'distance_percent', 'level_type', 'level_method'
        """
        try:
            tolerance = tolerance_percent / 100.0
            nearest_distance = float('inf')
            nearest_level = None
            level_type = None
            
            # Check support levels
            for level in self.support_levels:
                distance = abs(current_price - level['price']) / level['price']
                if distance < tolerance and distance < nearest_distance:
                    nearest_distance = distance
                    nearest_level = level
                    level_type = 'support'
            
            # Check resistance levels
            for level in self.resistance_levels:
                distance = abs(current_price - level['price']) / level['price']
                if distance < tolerance and distance < nearest_distance:
                    nearest_distance = distance
                    nearest_level = level
                    level_type = 'resistance'
            
            if nearest_level is not None:
                return {
                    'in_zone': True,
                    'nearest_level': nearest_level,
                    'distance_percent': nearest_distance * 100,
                    'level_type': level_type,
                    'level_strength': nearest_level.get('strength', 0.0),
                    'touch_count': nearest_level.get('touch_count', 0),
                    'level_method': nearest_level.get('method', 'traditional'),
                    'confidence': min(nearest_level.get('strength', 0.0) / 2.0, 1.0)
                }
            else:
                return {
                    'in_zone': False,
                    'nearest_level': None,
                    'distance_percent': None,
                    'level_type': None,
                    'level_strength': None,
                    'touch_count': None,
                    'level_method': None,
                    'confidence': 0.0
                }
                
        except Exception as e:
            self.logger.error(f"Error detecting SR zone proximity: {e}")
            return {
                'in_zone': False,
                'nearest_level': None,
                'distance_percent': None,
                'level_type': None,
                'level_strength': None,
                'touch_count': None,
                'level_method': None,
                'confidence': 0.0
            }

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="SR analyzer cleanup",
    )
    async def stop(self) -> None:
        """Stop the SR analyzer component."""
        self.logger.info("🛑 Stopping SR Level Analyzer...")

        try:
            # Save analysis history
            if self.analysis_history:
                self.logger.info(
                    f"Saving {len(self.analysis_history)} analysis records",
                )

            # Clear current state
            self.support_levels = []
            self.resistance_levels = []
            self.last_analysis_time = None

            self.logger.info("✅ SR Level Analyzer stopped successfully")

        except Exception as e:
            self.logger.error(f"Error stopping SR analyzer: {e}")
