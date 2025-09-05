from typing import List
from typing import Dict
from typing import Any
'Enhanced Multi-Timeframe S/R Confluence Detection Module.\n\nThis module provides advanced multi-timeframe S/R confluence detection\nwith sophisticated weighting and validation algorithms.\n'
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')
from .core.decorators import handles_errors, traced
from .utils.logger import system_logger
from .core.decorators.errors import handles_errors
import numpy as np

@dataclass
class ConfluenceLevel:
    """Multi-timeframe S/R confluence level."""
    price: float
    strength: float
    type: str
    timeframes: List[str]
    timeframe_weights: Dict[str, float]
    confluence_score: float
    validation_score: float
    touch_count: int
    volume_confirmation: float
    age_bars: int
    metadata: Dict[str, Any]

@dataclass
class ConfluenceResult:
    """Result of multi-timeframe confluence analysis."""
    confluence_levels: List[ConfluenceLevel]
    total_levels: int
    high_confluence_levels: int
    avg_confluence_score: float
    timeframe_coverage: Dict[str, int]
    strength_distribution: Dict[str, float]
    validation_metrics: Dict[str, float]

class EnhancedSRConfluenceDetector:
    """Enhanced multi-timeframe S/R confluence detector."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize enhanced S/R confluence detector."""
        self.config = config
        self.logger = system_logger.getChild('EnhancedSRConfluenceDetector')
        self.timeframes = config.get('timeframes', ['1m', '5m', '15m', '30m', '1h', '4h'])
        self.timeframe_weights = config.get('timeframe_weights', {'1m': 0.1, '5m': 0.2, '15m': 0.3, '30m': 0.3, '1h': 0.2, '4h': 0.1})
        self.confluence_threshold = config.get('confluence_threshold', 0.001)
        self.min_timeframes = config.get('min_timeframes', 2)
        self.min_confluence_score = config.get('min_confluence_score', 0.6)
        self.validation_period = config.get('validation_period', 100)
        self.volume_confirmation_threshold = config.get('volume_confirmation_threshold', 1.5)

    @handles_errors(exceptions=(ValueError, AttributeError), default_return=None, context='detect multi-timeframe confluence')
    @traced(span_name='EnhancedSR.detect_confluence')
    def detect_confluence(self, multi_timeframe_levels: Dict[str, List[Any]]) -> Optional[ConfluenceResult]:
        """
        Detect multi-timeframe S/R confluence levels.
        
        Args:
            multi_timeframe_levels: Dictionary of S/R levels by timeframe
            
        Returns:
            Confluence analysis result
        """
        try:
            self.logger.info('🔍 Detecting multi-timeframe S/R confluence...')
            if not self._validate_input_data(multi_timeframe_levels):
                return None
            confluence_zones = self._find_confluence_zones(multi_timeframe_levels)
            confluence_levels = self._create_confluence_levels(confluence_zones, multi_timeframe_levels)
            validated_levels = self._validate_confluence_levels(confluence_levels, multi_timeframe_levels)
            confluence_metrics = self._calculate_confluence_metrics(validated_levels)
            result = ConfluenceResult(confluence_levels=validated_levels, total_levels=len(validated_levels), high_confluence_levels=len([l for l in validated_levels if l.confluence_score > 0.8]), avg_confluence_score=np.mean([l.confluence_score for l in validated_levels]) if validated_levels else 0.0, timeframe_coverage=confluence_metrics['timeframe_coverage'], strength_distribution=confluence_metrics['strength_distribution'], validation_metrics=confluence_metrics['validation_metrics'])
            self.logger.info(f'✅ Detected {len(validated_levels)} confluence levels')
            return result
        except Exception as e:
            self.logger.error(f'Multi-timeframe confluence detection failed: {e}')
            return None

    def _validate_input_data(self, multi_timeframe_levels: Dict[str, List[Any]]) -> bool:
        """Validate input data for confluence detection."""
        try:
            if not multi_timeframe_levels:
                self.logger.warning('No multi-timeframe levels provided')
                return False
            available_timeframes = [tf for tf in self.timeframes if tf in multi_timeframe_levels]
            if len(available_timeframes) < self.min_timeframes:
                self.logger.warning(f'Insufficient timeframes: {len(available_timeframes)} < {self.min_timeframes}')
                return False
            for timeframe, levels in multi_timeframe_levels.items():
                if not levels:
                    continue
                sample_level = levels[0]
                required_attrs = ['price', 'type', 'strength']
                for attr in required_attrs:
                    if not hasattr(sample_level, attr):
                        self.logger.warning(f'Level missing required attribute: {attr}')
                        return False
            return True
        except Exception as e:
            self.logger.warning(f'Input data validation failed: {e}')
            return False

    def _find_confluence_zones(self, multi_timeframe_levels: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """Find confluence zones across timeframes."""
        try:
            confluence_zones = []
            all_levels = []
            for timeframe, levels in multi_timeframe_levels.items():
                for level in levels:
                    all_levels.append({'price': getattr(level, 'price', 0.0), 'type': getattr(level, 'type', 'unknown'), 'strength': getattr(level, 'strength', 0.0), 'timeframe': timeframe, 'level': level})
            if not all_levels:
                return []
            used_indices = set()
            for i, level1 in enumerate(all_levels):
                if i in used_indices:
                    continue
                confluence_group = [level1]
                timeframes_in_group = {level1['timeframe']}
                for j, level2 in enumerate(all_levels[i + 1:], i + 1):
                    if j in used_indices:
                        continue
                    if self._are_levels_in_confluence(level1, level2):
                        confluence_group.append(level2)
                        timeframes_in_group.add(level2['timeframe'])
                        used_indices.add(j)
                if len(timeframes_in_group) >= self.min_timeframes:
                    confluence_zone = self._create_confluence_zone(confluence_group)
                    if confluence_zone:
                        confluence_zones.append(confluence_zone)
                used_indices.add(i)
            return confluence_zones
        except Exception as e:
            self.logger.warning(f'Confluence zone detection failed: {e}')
            return []

    def _are_levels_in_confluence(self, level1: Dict[str, Any], level2: Dict[str, Any]) -> bool:
        """Check if two levels are in confluence."""
        try:
            if level1['type'] != level2['type']:
                return False
            price_diff = abs(level1['price'] - level2['price']) / level1['price']
            if price_diff > self.confluence_threshold:
                return False
            if level1['timeframe'] == level2['timeframe']:
                return False
            return True
        except Exception as e:
            self.logger.warning(f'Confluence check failed: {e}')
            return False

    def _create_confluence_zone(self, confluence_group: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Create confluence zone from group of levels."""
        try:
            if not confluence_group:
                return None
            total_weight = 0.0
            weighted_price = 0.0
            for level in confluence_group:
                timeframe = level['timeframe']
                weight = self.timeframe_weights.get(timeframe, 0.1)
                strength = level['strength']
                combined_weight = weight * strength
                weighted_price += level['price'] * combined_weight
                total_weight += combined_weight
            if total_weight == 0:
                return None
            avg_price = weighted_price / total_weight
            timeframes = [level['timeframe'] for level in confluence_group]
            timeframe_weights = {level['timeframe']: self.timeframe_weights.get(level['timeframe'], 0.1) for level in confluence_group}
            confluence_score = self._calculate_confluence_score(confluence_group)
            return {'price': avg_price, 'type': confluence_group[0]['type'], 'timeframes': timeframes, 'timeframe_weights': timeframe_weights, 'confluence_score': confluence_score, 'levels': confluence_group, 'total_strength': sum((level['strength'] for level in confluence_group))}
        except Exception as e:
            self.logger.warning(f'Confluence zone creation failed: {e}')
            return None

    def _calculate_confluence_score(self, confluence_group: List[Dict[str, Any]]) -> float:
        """Calculate confluence score for a group of levels."""
        try:
            if not confluence_group:
                return 0.0
            timeframe_score = min(len(confluence_group) / len(self.timeframes), 1.0)
            avg_strength = np.mean([level['strength'] for level in confluence_group])
            strength_score = avg_strength
            weights = [self.timeframe_weights.get(level['timeframe'], 0.1) for level in confluence_group]
            weight_diversity = 1.0 - np.std(weights) / np.mean(weights) if np.mean(weights) > 0 else 0.0
            prices = [level['price'] for level in confluence_group]
            price_consistency = 1.0 - np.std(prices) / np.mean(prices) if np.mean(prices) > 0 else 0.0
            confluence_score = 0.4 * timeframe_score + 0.3 * strength_score + 0.2 * weight_diversity + 0.1 * price_consistency
            return min(1.0, max(0.0, confluence_score))
        except Exception as e:
            self.logger.warning(f'Confluence score calculation failed: {e}')
            return 0.0

    def _create_confluence_levels(self, confluence_zones: List[Dict[str, Any]], multi_timeframe_levels: Dict[str, List[Any]]) -> List[ConfluenceLevel]:
        """Create confluence levels from confluence zones."""
        try:
            confluence_levels = []
            for zone in confluence_zones:
                if zone['confluence_score'] < self.min_confluence_score:
                    continue
                touch_count = self._calculate_total_touches(zone, multi_timeframe_levels)
                volume_confirmation = self._calculate_volume_confirmation(zone, multi_timeframe_levels)
                age_bars = self._calculate_average_age(zone, multi_timeframe_levels)
                confluence_level = ConfluenceLevel(price=zone['price'], strength=zone['total_strength'] / len(zone['levels']), type=zone['type'], timeframes=zone['timeframes'], timeframe_weights=zone['timeframe_weights'], confluence_score=zone['confluence_score'], validation_score=0.0, touch_count=touch_count, volume_confirmation=volume_confirmation, age_bars=age_bars, metadata={'zone_id': f'zone_{len(confluence_levels)}', 'level_count': len(zone['levels']), 'timeframe_count': len(zone['timeframes']), 'creation_time': datetime.now().isoformat()})
                confluence_levels.append(confluence_level)
            return confluence_levels
        except Exception as e:
            self.logger.warning(f'Confluence level creation failed: {e}')
            return []

    def _calculate_total_touches(self, zone: Dict[str, Any], multi_timeframe_levels: Dict[str, List[Any]]) -> int:
        """Calculate total touches across all timeframes for confluence zone."""
        try:
            total_touches = 0
            for level_info in zone['levels']:
                timeframe = level_info['timeframe']
                level = level_info['level']
                if hasattr(level, 'touch_count'):
                    total_touches += level.touch_count
                else:
                    estimated_touches = max(1, int(level_info['strength'] * 5))
                    total_touches += estimated_touches
            return total_touches
        except Exception as e:
            self.logger.warning(f'Total touches calculation failed: {e}')
            return 0

    def _calculate_volume_confirmation(self, zone: Dict[str, Any], multi_timeframe_levels: Dict[str, List[Any]]) -> float:
        """Calculate volume confirmation across timeframes for confluence zone."""
        try:
            volume_scores = []
            for level_info in zone['levels']:
                timeframe = level_info['timeframe']
                level = level_info['level']
                if hasattr(level, 'volume_confirmation_score'):
                    volume_scores.append(level.volume_confirmation_score)
                else:
                    estimated_volume = min(1.0, level_info['strength'] * 1.2)
                    volume_scores.append(estimated_volume)
            return np.mean(volume_scores) if volume_scores else 0.0
        except Exception as e:
            self.logger.warning(f'Volume confirmation calculation failed: {e}')
            return 0.0

    def _calculate_average_age(self, zone: Dict[str, Any], multi_timeframe_levels: Dict[str, List[Any]]) -> int:
        """Calculate average age across timeframes for confluence zone."""
        try:
            ages = []
            for level_info in zone['levels']:
                level = level_info['level']
                if hasattr(level, 'age_bars'):
                    ages.append(level.age_bars)
                else:
                    timeframe = level_info['timeframe']
                    timeframe_minutes = self._get_timeframe_minutes(timeframe)
                    estimated_age = timeframe_minutes * 10
                    ages.append(estimated_age)
            return int(np.mean(ages)) if ages else 0
        except Exception as e:
            self.logger.warning(f'Average age calculation failed: {e}')
            return 0

    def _get_timeframe_minutes(self, timeframe: str) -> int:
        """Get timeframe in minutes."""
        timeframe_map = {'1m': 1, '5m': 5, '15m': 15, '30m': 30, '1h': 60, '4h': 240, '1d': 1440}
        return timeframe_map.get(timeframe, 1)

    def _validate_confluence_levels(self, confluence_levels: List[ConfluenceLevel], multi_timeframe_levels: Dict[str, List[Any]]) -> List[ConfluenceLevel]:
        """Validate confluence levels and calculate validation scores."""
        try:
            validated_levels = []
            for level in confluence_levels:
                validation_score = self._calculate_validation_score(level, multi_timeframe_levels)
                level.validation_score = validation_score
                if validation_score > 0.5:
                    validated_levels.append(level)
            return validated_levels
        except Exception as e:
            self.logger.warning(f'Confluence level validation failed: {e}')
            return confluence_levels

    def _calculate_validation_score(self, confluence_level: ConfluenceLevel, multi_timeframe_levels: Dict[str, List[Any]]) -> float:
        """Calculate validation score for confluence level."""
        try:
            base_score = confluence_level.confluence_score
            timeframe_boost = min(len(confluence_level.timeframes) / len(self.timeframes), 0.3)
            touch_boost = min(confluence_level.touch_count / 20.0, 0.2)
            volume_boost = confluence_level.volume_confirmation * 0.2
            age_boost = min(confluence_level.age_bars / 1000.0, 0.1)
            validation_score = base_score + timeframe_boost + touch_boost + volume_boost + age_boost
            return min(1.0, max(0.0, validation_score))
        except Exception as e:
            self.logger.warning(f'Validation score calculation failed: {e}')
            return 0.0

    def _calculate_confluence_metrics(self, confluence_levels: List[ConfluenceLevel]) -> Dict[str, Any]:
        """Calculate confluence analysis metrics."""
        try:
            if not confluence_levels:
                return {'timeframe_coverage': {}, 'strength_distribution': {}, 'validation_metrics': {}}
            timeframe_coverage = {}
            for level in confluence_levels:
                for timeframe in level.timeframes:
                    timeframe_coverage[timeframe] = timeframe_coverage.get(timeframe, 0) + 1
            strengths = [level.strength for level in confluence_levels]
            strength_distribution = {'mean': np.mean(strengths), 'std': np.std(strengths), 'min': np.min(strengths), 'max': np.max(strengths), 'median': np.median(strengths)}
            validation_scores = [level.validation_score for level in confluence_levels]
            confluence_scores = [level.confluence_score for level in confluence_levels]
            validation_metrics = {'avg_validation_score': np.mean(validation_scores), 'avg_confluence_score': np.mean(confluence_scores), 'high_quality_levels': len([s for s in validation_scores if s > 0.8]), 'total_levels': len(confluence_levels)}
            return {'timeframe_coverage': timeframe_coverage, 'strength_distribution': strength_distribution, 'validation_metrics': validation_metrics}
        except Exception as e:
            self.logger.warning(f'Confluence metrics calculation failed: {e}')
            return {'timeframe_coverage': {}, 'strength_distribution': {}, 'validation_metrics': {}}