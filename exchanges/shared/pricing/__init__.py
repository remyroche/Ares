"""
Pricing Utilities

Provides utilities for price fetching, OHLCV data management,
and market data aggregation.
"""

from .price_manager import PriceManager
from .ohlcv_manager import OHLCVManager
from .enhanced_ohlcv_manager import EnhancedOHLCVManager

# Market data aggregation class
class MarketDataAggregator:
    """
    Comprehensive market data aggregation and analysis.
    
    Provides functionality for collecting, processing, and analyzing market data
    from multiple sources with real-time updates and historical analysis.
    """
    
    def __init__(self, max_data_points: int = 10000):
        """
        Initialize the MarketDataAggregator.
        
        Args:
            max_data_points: Maximum number of data points to keep in memory
        """
        self.max_data_points = max_data_points
        self.market_data = {}  # {symbol: [data_points]}
        self.price_feeds = {}  # {symbol: price_feed_info}
        self.aggregated_data = {}  # {symbol: aggregated_metrics}
        self.subscribers = {}  # {symbol: [callback_functions]}
        self.data_sources = {}
        self.update_callbacks = []
    
    def add_data_source(self, source_name: str, source_config: dict):
        """
        Add a data source for market data.
        
        Args:
            source_name: Name of the data source
            source_config: Configuration for the data source
        """
        self.data_sources[source_name] = {
            'config': source_config,
            'last_update': None,
            'status': 'active',
            'data_count': 0
        }
    
    def subscribe_to_symbol(self, symbol: str, callback=None):
        """
        Subscribe to updates for a specific symbol.
        
        Args:
            symbol: Trading symbol to subscribe to
            callback: Callback function for updates (optional)
        """
        if symbol not in self.subscribers:
            self.subscribers[symbol] = []
        
        if callback:
            self.subscribers[symbol].append(callback)
    
    def add_price_data(self, symbol: str, price: float, volume: float = None,
                      timestamp: str = None, source: str = None, **kwargs):
        """
        Add price data for a symbol.
        
        Args:
            symbol: Trading symbol
            price: Price value
            volume: Trading volume (optional)
            timestamp: Data timestamp (optional)
            source: Data source (optional)
            **kwargs: Additional data fields
        """
        if timestamp is None:
            timestamp = self._get_timestamp()
        
        data_point = {
            'symbol': symbol,
            'price': price,
            'volume': volume,
            'timestamp': timestamp,
            'source': source,
            **kwargs
        }
        
        # Add to market data
        if symbol not in self.market_data:
            self.market_data[symbol] = []
        
        self.market_data[symbol].append(data_point)
        
        # Maintain max data points
        if len(self.market_data[symbol]) > self.max_data_points:
            self.market_data[symbol] = self.market_data[symbol][-self.max_data_points:]
        
        # Update aggregated data
        self._update_aggregated_data(symbol)
        
        # Notify subscribers
        self._notify_subscribers(symbol, data_point)
        
        # Update data source stats
        if source and source in self.data_sources:
            self.data_sources[source]['last_update'] = timestamp
            self.data_sources[source]['data_count'] += 1
    
    def get_latest_price(self, symbol: str) -> float:
        """Get the latest price for a symbol."""
        if symbol not in self.market_data or not self.market_data[symbol]:
            return None
        
        return self.market_data[symbol][-1]['price']
    
    def get_price_history(self, symbol: str, limit: int = None, 
                         start_time: str = None, end_time: str = None) -> list:
        """
        Get price history for a symbol.
        
        Args:
            symbol: Trading symbol
            limit: Maximum number of data points
            start_time: Start timestamp filter
            end_time: End timestamp filter
            
        Returns:
            List of price data points
        """
        if symbol not in self.market_data:
            return []
        
        data = self.market_data[symbol].copy()
        
        # Apply time filters
        if start_time:
            data = [d for d in data if d['timestamp'] >= start_time]
        
        if end_time:
            data = [d for d in data if d['timestamp'] <= end_time]
        
        # Apply limit
        if limit:
            data = data[-limit:]
        
        return data
    
    def get_aggregated_metrics(self, symbol: str) -> dict:
        """
        Get aggregated metrics for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dictionary with aggregated metrics
        """
        if symbol not in self.aggregated_data:
            return {}
        
        return self.aggregated_data[symbol].copy()
    
    def calculate_price_change(self, symbol: str, period: str = '1h') -> dict:
        """
        Calculate price change over a period.
        
        Args:
            symbol: Trading symbol
            period: Time period ('1h', '24h', '7d', etc.)
            
        Returns:
            Dictionary with price change metrics
        """
        if symbol not in self.market_data or not self.market_data[symbol]:
            return {'change': 0, 'change_percent': 0, 'period': period}
        
        data = self.market_data[symbol]
        current_price = data[-1]['price']
        
        # Calculate time threshold
        time_threshold = self._calculate_time_threshold(period)
        
        # Find data point at the start of the period
        start_price = current_price
        for point in reversed(data):
            if self._is_before_threshold(point['timestamp'], time_threshold):
                start_price = point['price']
                break
        
        # Calculate changes
        price_change = current_price - start_price
        change_percent = (price_change / start_price) * 100 if start_price != 0 else 0
        
        return {
            'current_price': current_price,
            'start_price': start_price,
            'change': price_change,
            'change_percent': change_percent,
            'period': period
        }
    
    def calculate_volume_metrics(self, symbol: str, period: str = '24h') -> dict:
        """
        Calculate volume metrics for a symbol.
        
        Args:
            symbol: Trading symbol
            period: Time period
            
        Returns:
            Dictionary with volume metrics
        """
        if symbol not in self.market_data:
            return {'total_volume': 0, 'avg_volume': 0, 'period': period}
        
        data = self.market_data[symbol]
        time_threshold = self._calculate_time_threshold(period)
        
        # Filter data by period
        period_data = [d for d in data if not self._is_before_threshold(d['timestamp'], time_threshold)]
        
        if not period_data:
            return {'total_volume': 0, 'avg_volume': 0, 'period': period}
        
        # Calculate volume metrics
        volumes = [d.get('volume', 0) for d in period_data if d.get('volume') is not None]
        
        if not volumes:
            return {'total_volume': 0, 'avg_volume': 0, 'period': period}
        
        total_volume = sum(volumes)
        avg_volume = total_volume / len(volumes)
        
        return {
            'total_volume': total_volume,
            'avg_volume': avg_volume,
            'data_points': len(volumes),
            'period': period
        }
    
    def get_top_symbols_by_volume(self, limit: int = 10, period: str = '24h') -> list:
        """
        Get top symbols by trading volume.
        
        Args:
            limit: Number of top symbols to return
            period: Time period for volume calculation
            
        Returns:
            List of symbols sorted by volume
        """
        symbol_volumes = []
        
        for symbol in self.market_data.keys():
            volume_metrics = self.calculate_volume_metrics(symbol, period)
            if volume_metrics['total_volume'] > 0:
                symbol_volumes.append({
                    'symbol': symbol,
                    'total_volume': volume_metrics['total_volume'],
                    'avg_volume': volume_metrics['avg_volume']
                })
        
        # Sort by total volume
        symbol_volumes.sort(key=lambda x: x['total_volume'], reverse=True)
        
        return symbol_volumes[:limit]
    
    def get_market_summary(self) -> dict:
        """Get overall market summary."""
        total_symbols = len(self.market_data)
        total_data_points = sum(len(data) for data in self.market_data.values())
        
        active_sources = len([s for s in self.data_sources.values() if s['status'] == 'active'])
        
        return {
            'total_symbols': total_symbols,
            'total_data_points': total_data_points,
            'active_sources': active_sources,
            'symbols': list(self.market_data.keys()),
            'data_sources': list(self.data_sources.keys())
        }
    
    def export_data(self, symbol: str, file_path: str, format: str = 'json'):
        """
        Export market data to file.
        
        Args:
            symbol: Trading symbol
            file_path: Path to export file
            format: Export format ('json' or 'csv')
        """
        import json
        import csv
        
        if symbol not in self.market_data:
            return
        
        data = self.market_data[symbol]
        
        if format.lower() == 'json':
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
        elif format.lower() == 'csv':
            if data:
                with open(file_path, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=data[0].keys())
                    writer.writeheader()
                    writer.writerows(data)
    
    def _update_aggregated_data(self, symbol: str):
        """Update aggregated data for a symbol."""
        if symbol not in self.market_data or not self.market_data[symbol]:
            return
        
        data = self.market_data[symbol]
        prices = [d['price'] for d in data]
        volumes = [d.get('volume', 0) for d in data if d.get('volume') is not None]
        
        # Calculate basic metrics
        latest_price = prices[-1]
        min_price = min(prices)
        max_price = max(prices)
        avg_price = sum(prices) / len(prices)
        
        # Calculate price volatility (standard deviation)
        price_variance = sum((p - avg_price) ** 2 for p in prices) / len(prices)
        price_volatility = price_variance ** 0.5
        
        self.aggregated_data[symbol] = {
            'latest_price': latest_price,
            'min_price': min_price,
            'max_price': max_price,
            'avg_price': avg_price,
            'price_volatility': price_volatility,
            'total_volume': sum(volumes) if volumes else 0,
            'avg_volume': sum(volumes) / len(volumes) if volumes else 0,
            'data_points': len(data),
            'last_update': data[-1]['timestamp']
        }
    
    def _notify_subscribers(self, symbol: str, data_point: dict):
        """Notify subscribers of new data."""
        if symbol in self.subscribers:
            for callback in self.subscribers[symbol]:
                try:
                    callback(symbol, data_point)
                except Exception as e:
                    print(f"Error in subscriber callback: {e}")
    
    def _calculate_time_threshold(self, period: str) -> int:
        """Calculate time threshold in seconds for a period."""
        period_map = {
            '1m': 60,
            '5m': 300,
            '15m': 900,
            '1h': 3600,
            '4h': 14400,
            '24h': 86400,
            '7d': 604800,
            '30d': 2592000
        }
        
        return period_map.get(period, 3600)  # Default to 1 hour
    
    def _is_before_threshold(self, timestamp: str, threshold_seconds: int) -> bool:
        """Check if timestamp is before the threshold."""
        from datetime import datetime, timedelta
        
        try:
            timestamp_dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            threshold_time = datetime.now() - timedelta(seconds=threshold_seconds)
            return timestamp_dt < threshold_time
        except:
            return False
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()

__all__ = [
    "PriceManager",
    "OHLCVManager",
    "EnhancedOHLCVManager",
    "MarketDataAggregator"
]