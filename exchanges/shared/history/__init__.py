"""
History Management Utilities

Provides utilities for trade history management and pagination.
"""

# History management classes
class TradeHistoryManager:
    """
    Comprehensive trade history management and analysis.
    
    Provides functionality for storing, retrieving, and analyzing trade history
    with support for filtering, sorting, and statistical analysis.
    """
    
    def __init__(self, max_history_size: int = 10000):
        """
        Initialize the TradeHistoryManager.
        
        Args:
            max_history_size: Maximum number of trades to keep in memory
        """
        self.max_history_size = max_history_size
        self.trades = []
        self.trade_index = {}  # {trade_id: index}
        self.symbol_index = {}  # {symbol: [trade_indices]}
        self.user_index = {}  # {user_id: [trade_indices]}
        self.statistics_cache = {}
    
    def add_trade(self, trade_id: str, symbol: str, side: str, quantity: float,
                  price: float, timestamp: str = None, user_id: str = None,
                  order_id: str = None, commission: float = 0.0, 
                  commission_asset: str = None, **kwargs):
        """
        Add a trade to the history.
        
        Args:
            trade_id: Unique trade identifier
            symbol: Trading symbol
            side: Trade side ('buy' or 'sell')
            quantity: Trade quantity
            price: Trade price
            timestamp: Trade timestamp (optional, defaults to current time)
            user_id: User identifier (optional)
            order_id: Associated order ID (optional)
            commission: Commission paid (optional)
            commission_asset: Commission asset (optional)
            **kwargs: Additional trade data
        """
        if timestamp is None:
            timestamp = self._get_timestamp()
        
        trade = {
            'trade_id': trade_id,
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'price': price,
            'timestamp': timestamp,
            'user_id': user_id,
            'order_id': order_id,
            'commission': commission,
            'commission_asset': commission_asset,
            'notional_value': quantity * price,
            **kwargs
        }
        
        # Add to main list
        self.trades.append(trade)
        trade_index = len(self.trades) - 1
        
        # Update indexes
        self.trade_index[trade_id] = trade_index
        
        if symbol not in self.symbol_index:
            self.symbol_index[symbol] = []
        self.symbol_index[symbol].append(trade_index)
        
        if user_id:
            if user_id not in self.user_index:
                self.user_index[user_id] = []
            self.user_index[user_id].append(trade_index)
        
        # Maintain max size
        if len(self.trades) > self.max_history_size:
            self._remove_oldest_trade()
        
        # Clear statistics cache
        self.statistics_cache.clear()
    
    def get_trade(self, trade_id: str) -> dict:
        """Get a specific trade by ID."""
        if trade_id in self.trade_index:
            index = self.trade_index[trade_id]
            return self.trades[index].copy()
        return None
    
    def get_trades_by_symbol(self, symbol: str, limit: int = None) -> list:
        """Get trades for a specific symbol."""
        if symbol not in self.symbol_index:
            return []
        
        indices = self.symbol_index[symbol]
        trades = [self.trades[i] for i in indices]
        
        if limit:
            trades = trades[-limit:]
        
        return trades.copy()
    
    def get_trades_by_user(self, user_id: str, limit: int = None) -> list:
        """Get trades for a specific user."""
        if user_id not in self.user_index:
            return []
        
        indices = self.user_index[user_id]
        trades = [self.trades[i] for i in indices]
        
        if limit:
            trades = trades[-limit:]
        
        return trades.copy()
    
    def get_trades_in_range(self, start_time: str, end_time: str, 
                           symbol: str = None, user_id: str = None) -> list:
        """
        Get trades within a time range.
        
        Args:
            start_time: Start timestamp (ISO format)
            end_time: End timestamp (ISO format)
            symbol: Filter by symbol (optional)
            user_id: Filter by user (optional)
            
        Returns:
            List of trades in the specified range
        """
        trades = self.trades.copy()
        
        # Filter by symbol
        if symbol:
            trades = [t for t in trades if t['symbol'] == symbol]
        
        # Filter by user
        if user_id:
            trades = [t for t in trades if t.get('user_id') == user_id]
        
        # Filter by time range
        trades = [t for t in trades if start_time <= t['timestamp'] <= end_time]
        
        return trades
    
    def get_recent_trades(self, limit: int = 100, symbol: str = None) -> list:
        """Get recent trades."""
        trades = self.trades.copy()
        
        if symbol:
            trades = [t for t in trades if t['symbol'] == symbol]
        
        return trades[-limit:] if limit else trades
    
    def search_trades(self, **filters) -> list:
        """
        Search trades with multiple filters.
        
        Args:
            **filters: Filter criteria (e.g., symbol='BTCUSDT', side='buy')
            
        Returns:
            List of matching trades
        """
        trades = self.trades.copy()
        
        for key, value in filters.items():
            if key in ['symbol', 'side', 'user_id', 'order_id']:
                trades = [t for t in trades if t.get(key) == value]
            elif key == 'min_quantity':
                trades = [t for t in trades if t['quantity'] >= value]
            elif key == 'max_quantity':
                trades = [t for t in trades if t['quantity'] <= value]
            elif key == 'min_price':
                trades = [t for t in trades if t['price'] >= value]
            elif key == 'max_price':
                trades = [t for t in trades if t['price'] <= value]
        
        return trades
    
    def get_trade_statistics(self, symbol: str = None, user_id: str = None) -> dict:
        """
        Get trade statistics.
        
        Args:
            symbol: Filter by symbol (optional)
            user_id: Filter by user (optional)
            
        Returns:
            Dictionary with trade statistics
        """
        cache_key = f"{symbol}_{user_id}"
        if cache_key in self.statistics_cache:
            return self.statistics_cache[cache_key]
        
        trades = self.trades.copy()
        
        if symbol:
            trades = [t for t in trades if t['symbol'] == symbol]
        
        if user_id:
            trades = [t for t in trades if t.get('user_id') == user_id]
        
        if not trades:
            return {
                'total_trades': 0,
                'total_volume': 0.0,
                'total_value': 0.0,
                'avg_price': 0.0,
                'buy_trades': 0,
                'sell_trades': 0,
                'total_commission': 0.0
            }
        
        total_trades = len(trades)
        total_volume = sum(t['quantity'] for t in trades)
        total_value = sum(t['notional_value'] for t in trades)
        avg_price = total_value / total_volume if total_volume > 0 else 0.0
        
        buy_trades = len([t for t in trades if t['side'] == 'buy'])
        sell_trades = len([t for t in trades if t['side'] == 'sell'])
        total_commission = sum(t.get('commission', 0.0) for t in trades)
        
        stats = {
            'total_trades': total_trades,
            'total_volume': total_volume,
            'total_value': total_value,
            'avg_price': avg_price,
            'buy_trades': buy_trades,
            'sell_trades': sell_trades,
            'total_commission': total_commission,
            'buy_volume': sum(t['quantity'] for t in trades if t['side'] == 'buy'),
            'sell_volume': sum(t['quantity'] for t in trades if t['side'] == 'sell')
        }
        
        self.statistics_cache[cache_key] = stats
        return stats
    
    def get_symbols_traded(self) -> list:
        """Get list of all symbols that have been traded."""
        return list(self.symbol_index.keys())
    
    def get_users_traded(self) -> list:
        """Get list of all users who have traded."""
        return list(self.user_index.keys())
    
    def export_trades(self, file_path: str, format: str = 'json', 
                     symbol: str = None, user_id: str = None):
        """
        Export trades to file.
        
        Args:
            file_path: Path to export file
            format: Export format ('json' or 'csv')
            symbol: Filter by symbol (optional)
            user_id: Filter by user (optional)
        """
        import json
        import csv
        
        trades = self.trades.copy()
        
        if symbol:
            trades = [t for t in trades if t['symbol'] == symbol]
        
        if user_id:
            trades = [t for t in trades if t.get('user_id') == user_id]
        
        if format.lower() == 'json':
            with open(file_path, 'w') as f:
                json.dump(trades, f, indent=2)
        elif format.lower() == 'csv':
            if trades:
                with open(file_path, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=trades[0].keys())
                    writer.writeheader()
                    writer.writerows(trades)
    
    def _remove_oldest_trade(self):
        """Remove the oldest trade to maintain max size."""
        if not self.trades:
            return
        
        oldest_trade = self.trades.pop(0)
        trade_id = oldest_trade['trade_id']
        symbol = oldest_trade['symbol']
        user_id = oldest_trade.get('user_id')
        
        # Remove from indexes
        if trade_id in self.trade_index:
            del self.trade_index[trade_id]
        
        if symbol in self.symbol_index:
            self.symbol_index[symbol].pop(0)
            if not self.symbol_index[symbol]:
                del self.symbol_index[symbol]
        
        if user_id and user_id in self.user_index:
            self.user_index[user_id].pop(0)
            if not self.user_index[user_id]:
                del self.user_index[user_id]
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()


class PaginationManager:
    """
    Manages pagination for large datasets.
    
    Provides functionality for paginating through large collections of data
    with support for different pagination strategies and metadata.
    """
    
    def __init__(self, default_page_size: int = 50, max_page_size: int = 1000):
        """
        Initialize the PaginationManager.
        
        Args:
            default_page_size: Default number of items per page
            max_page_size: Maximum allowed page size
        """
        self.default_page_size = default_page_size
        self.max_page_size = max_page_size
        self.pagination_cache = {}
    
    def paginate(self, data: list, page: int = 1, page_size: int = None, 
                sort_by: str = None, sort_order: str = 'asc') -> dict:
        """
        Paginate a list of data.
        
        Args:
            data: List of data to paginate
            page: Page number (1-based)
            page_size: Number of items per page
            sort_by: Field to sort by (optional)
            sort_order: Sort order ('asc' or 'desc')
            
        Returns:
            Dictionary with paginated data and metadata
        """
        if page_size is None:
            page_size = self.default_page_size
        
        # Validate page size
        page_size = min(max(1, page_size), self.max_page_size)
        page = max(1, page)
        
        # Sort data if requested
        if sort_by:
            data = self._sort_data(data, sort_by, sort_order)
        
        # Calculate pagination
        total_items = len(data)
        total_pages = (total_items + page_size - 1) // page_size
        start_index = (page - 1) * page_size
        end_index = start_index + page_size
        
        # Get page data
        page_data = data[start_index:end_index]
        
        # Build pagination metadata
        pagination_info = {
            'current_page': page,
            'page_size': page_size,
            'total_items': total_items,
            'total_pages': total_pages,
            'has_next': page < total_pages,
            'has_previous': page > 1,
            'next_page': page + 1 if page < total_pages else None,
            'previous_page': page - 1 if page > 1 else None,
            'start_index': start_index,
            'end_index': min(end_index, total_items),
            'items_on_page': len(page_data)
        }
        
        return {
            'data': page_data,
            'pagination': pagination_info
        }
    
    def paginate_with_cursor(self, data: list, cursor: str = None, 
                           page_size: int = None, cursor_field: str = 'id') -> dict:
        """
        Paginate data using cursor-based pagination.
        
        Args:
            data: List of data to paginate
            cursor: Cursor for current position
            page_size: Number of items per page
            cursor_field: Field to use as cursor
            
        Returns:
            Dictionary with paginated data and cursor info
        """
        if page_size is None:
            page_size = self.default_page_size
        
        page_size = min(max(1, page_size), self.max_page_size)
        
        # Sort data by cursor field
        data = sorted(data, key=lambda x: x.get(cursor_field, ''))
        
        # Find start position
        start_index = 0
        if cursor:
            for i, item in enumerate(data):
                if str(item.get(cursor_field, '')) == cursor:
                    start_index = i + 1
                    break
        
        # Get page data
        end_index = start_index + page_size
        page_data = data[start_index:end_index]
        
        # Get next cursor
        next_cursor = None
        if end_index < len(data):
            next_cursor = str(page_data[-1].get(cursor_field, ''))
        
        return {
            'data': page_data,
            'cursor': {
                'current': cursor,
                'next': next_cursor,
                'has_next': next_cursor is not None,
                'page_size': page_size,
                'total_items': len(data)
            }
        }
    
    def create_page_links(self, base_url: str, pagination_info: dict, 
                         additional_params: dict = None) -> dict:
        """
        Create page links for API responses.
        
        Args:
            base_url: Base URL for the API endpoint
            pagination_info: Pagination metadata
            additional_params: Additional query parameters
            
        Returns:
            Dictionary with page links
        """
        if additional_params is None:
            additional_params = {}
        
        links = {}
        
        # First page
        if pagination_info['has_previous']:
            first_params = {**additional_params, 'page': 1, 'page_size': pagination_info['page_size']}
            links['first'] = self._build_url(base_url, first_params)
        
        # Previous page
        if pagination_info['has_previous']:
            prev_params = {**additional_params, 'page': pagination_info['previous_page'], 
                          'page_size': pagination_info['page_size']}
            links['prev'] = self._build_url(base_url, prev_params)
        
        # Next page
        if pagination_info['has_next']:
            next_params = {**additional_params, 'page': pagination_info['next_page'], 
                          'page_size': pagination_info['page_size']}
            links['next'] = self._build_url(base_url, next_params)
        
        # Last page
        if pagination_info['has_next']:
            last_params = {**additional_params, 'page': pagination_info['total_pages'], 
                          'page_size': pagination_info['page_size']}
            links['last'] = self._build_url(base_url, last_params)
        
        return links
    
    def _sort_data(self, data: list, sort_by: str, sort_order: str) -> list:
        """Sort data by specified field and order."""
        reverse = sort_order.lower() == 'desc'
        
        try:
            return sorted(data, key=lambda x: x.get(sort_by, ''), reverse=reverse)
        except (TypeError, KeyError):
            # Fallback to string sorting if field doesn't exist or can't be compared
            return sorted(data, key=lambda x: str(x.get(sort_by, '')), reverse=reverse)
    
    def _build_url(self, base_url: str, params: dict) -> str:
        """Build URL with query parameters."""
        import urllib.parse
        
        if not params:
            return base_url
        
        query_string = urllib.parse.urlencode(params)
        separator = '&' if '?' in base_url else '?'
        return f"{base_url}{separator}{query_string}"
    
    def get_pagination_summary(self, pagination_info: dict) -> str:
        """Get a human-readable pagination summary."""
        current = pagination_info['current_page']
        total = pagination_info['total_pages']
        items = pagination_info['total_items']
        page_size = pagination_info['page_size']
        
        start_item = pagination_info['start_index'] + 1
        end_item = pagination_info['end_index']
        
        return f"Page {current} of {total} ({start_item}-{end_item} of {items} items, {page_size} per page)"

__all__ = [
    "TradeHistoryManager",
    "PaginationManager"
]
