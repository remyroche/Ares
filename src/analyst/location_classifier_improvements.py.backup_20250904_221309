# Improvements for the Fractal Location Classifier

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple

class LocationClassifierEnhancements:
    """Enhanced methods for the fractal location classifier."""
    
    def calculate_advanced_distance_metrics(
        self, 
        current_price: float, 
        support_levels: List[Dict], 
        resistance_levels: List[Dict],
        atr: float
    ) -> Dict[str, float]:
        """
        Calculate advanced distance metrics beyond simple nearest level.
        """
        metrics = {}
        
        # 1. Weighted Average Distance (considers multiple levels)
        # Closer levels have more weight
        support_distances = [(current_price - s['price']) / current_price for s in support_levels[:5]]
        resistance_distances = [(r['price'] - current_price) / current_price for r in resistance_levels[:5]]
        
        if support_distances:
            weights = [1/((i+1)**2) for i in range(len(support_distances))]  # 1, 1/4, 1/9, ...
            weight_sum = sum(weights)
            metrics['weighted_support_distance'] = sum(d*w for d,w in zip(support_distances, weights)) / weight_sum
        else:
            metrics['weighted_support_distance'] = 1.0
            
        if resistance_distances:
            weights = [1/((i+1)**2) for i in range(len(resistance_distances))]
            weight_sum = sum(weights)
            metrics['weighted_resistance_distance'] = sum(d*w for d,w in zip(resistance_distances, weights)) / weight_sum
        else:
            metrics['weighted_resistance_distance'] = 1.0
        
        # 2. Distance Velocity (how fast price is moving toward/away from levels)
        # Requires price history
        metrics['support_approach_velocity'] = 0.0  # Placeholder - implement with price history
        metrics['resistance_approach_velocity'] = 0.0
        
        # 3. Normalized Distance Score (0-1, considering typical market ranges)
        # Uses sigmoid transformation for smooth scaling
        metrics['normalized_support_score'] = 1 / (1 + np.exp(10 * metrics['weighted_support_distance']))
        metrics['normalized_resistance_score'] = 1 / (1 + np.exp(10 * metrics['weighted_resistance_distance']))
        
        # 4. Zone Density (how many levels in proximity)
        zone_threshold = 0.02  # 2% zone
        support_zone_count = sum(1 for s in support_levels if abs(current_price - s['price'])/current_price <= zone_threshold)
        resistance_zone_count = sum(1 for r in resistance_levels if abs(r['price'] - current_price)/current_price <= zone_threshold)
        
        metrics['support_zone_density'] = min(1.0, support_zone_count / 5.0)  # Normalize to 0-1
        metrics['resistance_zone_density'] = min(1.0, resistance_zone_count / 5.0)
        
        return metrics
    
    def calculate_dynamic_strength_metrics(
        self,
        levels: List[Dict],
        market_data: pd.DataFrame,
        lookback_periods: int = 100
    ) -> Dict[str, float]:
        """
        Calculate dynamic strength metrics that adapt to market conditions.
        """
        if not levels or market_data.empty:
            return {}
        
        metrics = {}
        current_volatility = market_data['close'].pct_change().std() * np.sqrt(252)
        
        # 1. Volatility-Adjusted Strength
        # Levels are stronger in low volatility environments
        volatility_factor = 1 / (1 + current_volatility)
        
        # 2. Recency-Weighted Strength
        # Recent touches are more important
        current_time = len(market_data)
        for level in levels:
            # Calculate time-decayed strength
            if 'last_touch_time' in level:
                time_decay = np.exp(-0.01 * (current_time - level['last_touch_time']))
                level['time_adjusted_strength'] = level['strength'] * time_decay
            else:
                level['time_adjusted_strength'] = level['strength'] * 0.5  # Penalty for no recent touch
        
        # 3. Volume-Confirmed Strength
        # Levels with high volume interactions are stronger
        avg_volume = market_data['volume'].mean()
        for level in levels:
            if 'volume_at_level' in level:
                volume_ratio = level['volume_at_level'] / avg_volume
                level['volume_adjusted_strength'] = level['strength'] * (1 + np.log1p(volume_ratio))
            else:
                level['volume_adjusted_strength'] = level['strength']
        
        # 4. Market Regime Strength Adjustment
        # Trending markets have weaker S/R, ranging markets have stronger S/R
        price_momentum = market_data['close'].pct_change(20).iloc[-1]
        trend_factor = 1 - abs(price_momentum)  # Stronger levels in non-trending markets
        
        metrics['volatility_factor'] = volatility_factor
        metrics['trend_factor'] = trend_factor
        metrics['market_adjusted_strength_multiplier'] = volatility_factor * trend_factor
        
        return metrics
    
    def calculate_price_action_context(
        self,
        current_price: float,
        market_data: pd.DataFrame,
        support_levels: List[Dict],
        resistance_levels: List[Dict]
    ) -> Dict[str, Any]:
        """
        Add price action context to improve location analysis.
        """
        context = {}
        
        # 1. Recent Price Behavior
        close_prices = market_data['close'].values
        recent_high = market_data['high'].iloc[-20:].max()
        recent_low = market_data['low'].iloc[-20:].min()
        
        # Price position within recent range
        if recent_high != recent_low:
            context['price_position_in_range'] = (current_price - recent_low) / (recent_high - recent_low)
        else:
            context['price_position_in_range'] = 0.5
        
        # 2. Momentum at Levels
        # Are we approaching levels with momentum or exhaustion?
        rsi = self._calculate_rsi(close_prices, 14)
        context['momentum_state'] = 'overbought' if rsi > 70 else 'oversold' if rsi < 30 else 'neutral'
        context['rsi_value'] = rsi
        
        # 3. Level Test History
        # How many times have we tested nearby levels recently?
        test_threshold = 0.002  # 0.2%
        recent_prices = close_prices[-50:]
        
        support_tests = 0
        resistance_tests = 0
        
        for price in recent_prices:
            for s_level in support_levels[:3]:  # Check top 3 support levels
                if abs(price - s_level['price']) / price <= test_threshold:
                    support_tests += 1
                    break
            
            for r_level in resistance_levels[:3]:  # Check top 3 resistance levels
                if abs(price - r_level['price']) / price <= test_threshold:
                    resistance_tests += 1
                    break
        
        context['recent_support_tests'] = support_tests
        context['recent_resistance_tests'] = resistance_tests
        
        # 4. Breakout/Breakdown Detection
        # Are we in the process of breaking levels?
        if support_levels and current_price < support_levels[0]['price']:
            context['potential_breakdown'] = True
            context['breakdown_magnitude'] = (support_levels[0]['price'] - current_price) / current_price
        else:
            context['potential_breakdown'] = False
            context['breakdown_magnitude'] = 0.0
        
        if resistance_levels and current_price > resistance_levels[0]['price']:
            context['potential_breakout'] = True
            context['breakout_magnitude'] = (current_price - resistance_levels[0]['price']) / current_price
        else:
            context['potential_breakout'] = False
            context['breakout_magnitude'] = 0.0
        
        # 5. Volume Profile at Current Price
        # Is current price area high or low volume?
        price_bins = pd.cut(market_data['close'], bins=50)
        volume_profile = market_data.groupby(price_bins)['volume'].sum()
        current_bin = pd.cut([current_price], bins=volume_profile.index.categories)[0]
        
        if current_bin in volume_profile.index:
            current_volume_profile = volume_profile[current_bin]
            avg_volume_profile = volume_profile.mean()
            context['volume_profile_ratio'] = current_volume_profile / avg_volume_profile if avg_volume_profile > 0 else 1.0
        else:
            context['volume_profile_ratio'] = 1.0
        
        return context
    
    def calculate_level_interaction_features(
        self,
        current_price: float,
        support_levels: List[Dict],
        resistance_levels: List[Dict]
    ) -> Dict[str, float]:
        """
        Calculate features based on interaction between multiple levels.
        """
        features = {}
        
        # 1. Level Spacing Analysis
        if len(support_levels) >= 2:
            support_spacing = [(support_levels[i]['price'] - support_levels[i+1]['price']) / support_levels[i]['price'] 
                             for i in range(min(4, len(support_levels)-1))]
            features['avg_support_spacing'] = np.mean(support_spacing) if support_spacing else 0.0
            features['support_spacing_std'] = np.std(support_spacing) if support_spacing else 0.0
        else:
            features['avg_support_spacing'] = 0.0
            features['support_spacing_std'] = 0.0
        
        if len(resistance_levels) >= 2:
            resistance_spacing = [(resistance_levels[i+1]['price'] - resistance_levels[i]['price']) / resistance_levels[i]['price'] 
                                for i in range(min(4, len(resistance_levels)-1))]
            features['avg_resistance_spacing'] = np.mean(resistance_spacing) if resistance_spacing else 0.0
            features['resistance_spacing_std'] = np.std(resistance_spacing) if resistance_spacing else 0.0
        else:
            features['avg_resistance_spacing'] = 0.0
            features['resistance_spacing_std'] = 0.0
        
        # 2. Confluence Zones
        # Identify areas where multiple levels cluster
        confluence_threshold = 0.005  # 0.5%
        
        support_clusters = self._find_level_clusters(support_levels, confluence_threshold)
        resistance_clusters = self._find_level_clusters(resistance_levels, confluence_threshold)
        
        features['support_confluence_zones'] = len(support_clusters)
        features['resistance_confluence_zones'] = len(resistance_clusters)
        
        # 3. Strength Distribution
        # How is strength distributed across levels?
        if support_levels:
            support_strengths = [s['strength'] for s in support_levels[:5]]
            features['support_strength_concentration'] = max(support_strengths) / sum(support_strengths) if sum(support_strengths) > 0 else 0.0
            features['support_strength_gradient'] = support_strengths[0] - support_strengths[-1] if len(support_strengths) > 1 else 0.0
        else:
            features['support_strength_concentration'] = 0.0
            features['support_strength_gradient'] = 0.0
        
        if resistance_levels:
            resistance_strengths = [r['strength'] for r in resistance_levels[:5]]
            features['resistance_strength_concentration'] = max(resistance_strengths) / sum(resistance_strengths) if sum(resistance_strengths) > 0 else 0.0
            features['resistance_strength_gradient'] = resistance_strengths[0] - resistance_strengths[-1] if len(resistance_strengths) > 1 else 0.0
        else:
            features['resistance_strength_concentration'] = 0.0
            features['resistance_strength_gradient'] = 0.0
        
        # 4. Price Squeeze Detection
        # Is price being squeezed between strong S/R?
        if support_levels and resistance_levels:
            nearest_support = support_levels[0]
            nearest_resistance = resistance_levels[0]
            
            price_range = nearest_resistance['price'] - nearest_support['price']
            price_range_pct = price_range / current_price
            
            features['price_squeeze_ratio'] = 1 / (1 + price_range_pct)  # Higher when squeezed
            features['squeeze_strength'] = (nearest_support['strength'] + nearest_resistance['strength']) / 2
            features['squeeze_score'] = features['price_squeeze_ratio'] * features['squeeze_strength']
        else:
            features['price_squeeze_ratio'] = 0.0
            features['squeeze_strength'] = 0.0
            features['squeeze_score'] = 0.0
        
        return features
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate RSI."""
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices)
        seed = deltas[:period+1]
        up = seed[seed >= 0].sum() / period
        down = -seed[seed < 0].sum() / period
        
        if down == 0:
            return 100.0
        
        rs = up / down
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _find_level_clusters(self, levels: List[Dict], threshold: float) -> List[List[Dict]]:
        """Find clusters of nearby levels."""
        if not levels:
            return []
        
        clusters = []
        current_cluster = [levels[0]]
        
        for i in range(1, len(levels)):
            if abs(levels[i]['price'] - current_cluster[-1]['price']) / current_cluster[-1]['price'] <= threshold:
                current_cluster.append(levels[i])
            else:
                if len(current_cluster) > 1:
                    clusters.append(current_cluster)
                current_cluster = [levels[i]]
        
        if len(current_cluster) > 1:
            clusters.append(current_cluster)
        
        return clusters
    
    def generate_ml_ready_features(self, all_metrics: Dict[str, Any]) -> pd.Series:
        """
        Generate final ML-ready feature vector from all metrics.
        """
        # Flatten all nested dictionaries
        flat_features = {}
        
        for key, value in all_metrics.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, (int, float, bool)):
                        flat_features[f"{key}_{sub_key}"] = float(sub_value)
            elif isinstance(value, (int, float, bool)):
                flat_features[key] = float(value)
            elif value is None:
                flat_features[key] = 0.0
        
        return pd.Series(flat_features)