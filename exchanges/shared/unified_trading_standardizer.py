"""
Unified Trading Standardizer

Ce module fournit une standardisation unifiée pour les données de trading
à travers différents exchanges.
"""

from typing import Dict, Any, Optional, Union, List
from datetime import datetime
from dataclasses import dataclass
from enum import Enum


class StandardizationError(Exception):
    """Exception levée pour les erreurs de standardisation."""
    pass


class StandardizationRule(Enum):
    """Règles de standardisation disponibles."""
    SYMBOL_MAPPING = "symbol_mapping"
    PRICE_PRECISION = "price_precision"
    QUANTITY_PRECISION = "quantity_precision"
    TIMESTAMP_FORMAT = "timestamp_format"
    ORDER_TYPE_MAPPING = "order_type_mapping"
    SIDE_MAPPING = "side_mapping"


@dataclass
class StandardizedOrder:
    """Ordre standardisé."""
    symbol: str
    side: str
    order_type: str
    quantity: float
    price: Optional[float]
    timestamp: datetime
    exchange: str
    exchange_order_id: Optional[str] = None
    status: Optional[str] = None
    filled_quantity: Optional[float] = None
    remaining_quantity: Optional[float] = None
    average_price: Optional[float] = None
    fees: Optional[float] = None


@dataclass
class StandardizedTicker:
    """Ticker standardisé."""
    symbol: str
    bid: float
    ask: float
    last: float
    volume: float
    timestamp: datetime
    exchange: str


@dataclass
class StandardizedTrade:
    """Trade standardisé."""
    symbol: str
    side: str
    quantity: float
    price: float
    timestamp: datetime
    exchange: str
    trade_id: Optional[str] = None
    fee: Optional[float] = None


@dataclass
class StandardizedBalance:
    """Balance standardisée."""
    asset: str
    free: float
    used: float
    total: float
    exchange: str


@dataclass
class StandardizedPosition:
    """Position standardisée."""
    symbol: str
    side: str
    quantity: float
    entry_price: float
    current_price: Optional[float]
    unrealized_pnl: Optional[float]
    realized_pnl: Optional[float]
    exchange: str
    timestamp: datetime


class UnifiedTradingStandardizer:
    """
    Standardiseur unifié pour les données de trading.
    
    Cette classe fournit des méthodes pour standardiser les données
    provenant de différents exchanges vers un format commun.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialise le standardiseur.
        
        Args:
            config: Configuration personnalisée pour la standardisation
        """
        self.config = config or {}
        self._symbol_mappings = self.config.get('symbol_mappings', {})
        self._price_precisions = self.config.get('price_precisions', {})
        self._quantity_precisions = self.config.get('quantity_precisions', {})
        
        # Mappages par défaut
        self._default_symbol_mappings = {
            'BTCUSDT': 'BTC/USDT',
            'ETHUSDT': 'ETH/USDT',
            'ADAUSDT': 'ADA/USDT',
        }
        
        self._default_order_type_mappings = {
            'market': 'market',
            'limit': 'limit',
            'stop': 'stop',
            'stop_limit': 'stop_limit',
        }
        
        self._default_side_mappings = {
            'buy': 'buy',
            'sell': 'sell',
            'long': 'buy',
            'short': 'sell',
        }
    
    def standardize_order(self, order_data: Dict[str, Any], exchange: str) -> StandardizedOrder:
        """
        Standardise un ordre.
        
        Args:
            order_data: Données brutes de l'ordre
            exchange: Nom de l'exchange source
            
        Returns:
            StandardizedOrder: Ordre standardisé
            
        Raises:
            StandardizationError: Si la standardisation échoue
        """
        try:
            # Standardisation du symbole
            symbol = self._standardize_symbol(order_data.get('symbol', ''), exchange)
            
            # Standardisation du côté
            side = self._standardize_side(order_data.get('side', ''))
            
            # Standardisation du type d'ordre
            order_type = self._standardize_order_type(order_data.get('order_type', ''))
            
            # Standardisation des quantités et prix
            quantity = float(order_data.get('quantity', 0))
            price = order_data.get('price')
            if price is not None:
                price = float(price)
                price = self._apply_price_precision(price, symbol, exchange)
            
            quantity = self._apply_quantity_precision(quantity, symbol, exchange)
            
            # Standardisation du timestamp
            timestamp = self._standardize_timestamp(order_data.get('timestamp'))
            
            return StandardizedOrder(
                symbol=symbol,
                side=side,
                order_type=order_type,
                quantity=quantity,
                price=price,
                timestamp=timestamp,
                exchange=exchange,
                exchange_order_id=order_data.get('id'),
                status=order_data.get('status'),
                filled_quantity=order_data.get('filled'),
                remaining_quantity=order_data.get('remaining'),
                average_price=order_data.get('average_price'),
                fees=order_data.get('fee')
            )
            
        except (ValueError, TypeError, KeyError) as e:
            raise StandardizationError(f"Erreur lors de la standardisation de l'ordre: {e}")
    
    def standardize_ticker(self, ticker_data: Dict[str, Any], exchange: str) -> StandardizedTicker:
        """
        Standardise un ticker.
        
        Args:
            ticker_data: Données brutes du ticker
            exchange: Nom de l'exchange source
            
        Returns:
            StandardizedTicker: Ticker standardisé
            
        Raises:
            StandardizationError: Si la standardisation échoue
        """
        try:
            symbol = self._standardize_symbol(ticker_data.get('symbol', ''), exchange)
            
            bid = float(ticker_data.get('bid', 0))
            ask = float(ticker_data.get('ask', 0))
            last = float(ticker_data.get('last', 0))
            volume = float(ticker_data.get('volume', 0))
            
            # Appliquer la précision de prix
            bid = self._apply_price_precision(bid, symbol, exchange)
            ask = self._apply_price_precision(ask, symbol, exchange)
            last = self._apply_price_precision(last, symbol, exchange)
            
            timestamp = self._standardize_timestamp(ticker_data.get('timestamp'))
            
            return StandardizedTicker(
                symbol=symbol,
                bid=bid,
                ask=ask,
                last=last,
                volume=volume,
                timestamp=timestamp,
                exchange=exchange
            )
            
        except (ValueError, TypeError, KeyError) as e:
            raise StandardizationError(f"Erreur lors de la standardisation du ticker: {e}")
    
    def standardize_trade(self, trade_data: Dict[str, Any], exchange: str) -> StandardizedTrade:
        """
        Standardise un trade.
        
        Args:
            trade_data: Données brutes du trade
            exchange: Nom de l'exchange source
            
        Returns:
            StandardizedTrade: Trade standardisé
            
        Raises:
            StandardizationError: Si la standardisation échoue
        """
        try:
            symbol = self._standardize_symbol(trade_data.get('symbol', ''), exchange)
            side = self._standardize_side(trade_data.get('side', ''))
            
            quantity = float(trade_data.get('quantity', 0))
            price = float(trade_data.get('price', 0))
            fee = trade_data.get('fee')
            if fee is not None:
                fee = float(fee)
            
            # Appliquer les précisions
            quantity = self._apply_quantity_precision(quantity, symbol, exchange)
            price = self._apply_price_precision(price, symbol, exchange)
            
            timestamp = self._standardize_timestamp(trade_data.get('timestamp'))
            
            return StandardizedTrade(
                symbol=symbol,
                side=side,
                quantity=quantity,
                price=price,
                timestamp=timestamp,
                exchange=exchange,
                trade_id=trade_data.get('id'),
                fee=fee
            )
            
        except (ValueError, TypeError, KeyError) as e:
            raise StandardizationError(f"Erreur lors de la standardisation du trade: {e}")
    
    def standardize_balance(self, balance_data: Dict[str, Any], exchange: str) -> StandardizedBalance:
        """
        Standardise une balance.
        
        Args:
            balance_data: Données brutes de la balance
            exchange: Nom de l'exchange source
            
        Returns:
            StandardizedBalance: Balance standardisée
            
        Raises:
            StandardizationError: Si la standardisation échoue
        """
        try:
            asset = balance_data.get('asset', '')
            free = float(balance_data.get('free', 0))
            used = float(balance_data.get('used', 0))
            total = float(balance_data.get('total', free + used))
            
            return StandardizedBalance(
                asset=asset,
                free=free,
                used=used,
                total=total,
                exchange=exchange
            )
            
        except (ValueError, TypeError, KeyError) as e:
            raise StandardizationError(f"Erreur lors de la standardisation de la balance: {e}")
    
    def standardize_position(self, position_data: Dict[str, Any], exchange: str) -> StandardizedPosition:
        """
        Standardise une position.
        
        Args:
            position_data: Données brutes de la position
            exchange: Nom de l'exchange source
            
        Returns:
            StandardizedPosition: Position standardisée
            
        Raises:
            StandardizationError: Si la standardisation échoue
        """
        try:
            symbol = self._standardize_symbol(position_data.get('symbol', ''), exchange)
            side = self._standardize_side(position_data.get('side', ''))
            
            quantity = float(position_data.get('quantity', 0))
            entry_price = float(position_data.get('entry_price', 0))
            current_price = position_data.get('current_price')
            unrealized_pnl = position_data.get('unrealized_pnl')
            realized_pnl = position_data.get('realized_pnl')
            
            if current_price is not None:
                current_price = float(current_price)
                current_price = self._apply_price_precision(current_price, symbol, exchange)
            
            if unrealized_pnl is not None:
                unrealized_pnl = float(unrealized_pnl)
            
            if realized_pnl is not None:
                realized_pnl = float(realized_pnl)
            
            # Appliquer la précision
            quantity = self._apply_quantity_precision(quantity, symbol, exchange)
            entry_price = self._apply_price_precision(entry_price, symbol, exchange)
            
            timestamp = self._standardize_timestamp(position_data.get('timestamp'))
            
            return StandardizedPosition(
                symbol=symbol,
                side=side,
                quantity=quantity,
                entry_price=entry_price,
                current_price=current_price,
                unrealized_pnl=unrealized_pnl,
                realized_pnl=realized_pnl,
                exchange=exchange,
                timestamp=timestamp
            )
            
        except (ValueError, TypeError, KeyError) as e:
            raise StandardizationError(f"Erreur lors de la standardisation de la position: {e}")
    
    def batch_standardize(self, items: List[Dict[str, Any]], exchange: str, 
                       item_type: str) -> List[Any]:
        """
        Standardise une liste d'items.
        
        Args:
            items: Liste d'items à standardiser
            exchange: Nom de l'exchange source
            item_type: Type d'items ('order', 'ticker', 'trade', 'balance', 'position')
            
        Returns:
            List[Any]: Liste d'items standardisés
            
        Raises:
            StandardizationError: Si la standardisation échoue
        """
        standardized_items = []
        
        for item in items:
            try:
                if item_type == 'order':
                    standardized = self.standardize_order(item, exchange)
                elif item_type == 'ticker':
                    standardized = self.standardize_ticker(item, exchange)
                elif item_type == 'trade':
                    standardized = self.standardize_trade(item, exchange)
                elif item_type == 'balance':
                    standardized = self.standardize_balance(item, exchange)
                elif item_type == 'position':
                    standardized = self.standardize_position(item, exchange)
                else:
                    raise StandardizationError(f"Type d'item non supporté: {item_type}")
                
                standardized_items.append(standardized)
                
            except StandardizationError as e:
                # Logger l'erreur mais continuer avec les autres items
                print(f"Erreur de standardisation pour un item: {e}")
                continue
        
        return standardized_items
    
    def get_standardization_rules(self) -> Dict[str, Any]:
        """
        Retourne les règles de standardisation actuelles.
        
        Returns:
            Dict[str, Any]: Règles de standardisation
        """
        return {
            'symbol_mappings': {**self._default_symbol_mappings, **self._symbol_mappings},
            'price_precisions': self._price_precisions,
            'quantity_precisions': self._quantity_precisions,
            'order_type_mappings': self._default_order_type_mappings,
            'side_mappings': self._default_side_mappings
        }
    
    def add_custom_standardization_rule(self, rule_type: str, rule_data: Dict[str, Any]) -> None:
        """
        Ajoute une règle de standardisation personnalisée.
        
        Args:
            rule_type: Type de règle (symbol_mapping, price_precision, etc.)
            rule_data: Données de la règle
        """
        if rule_type == 'symbol_mapping':
            self._symbol_mappings.update(rule_data)
        elif rule_type == 'price_precision':
            self._price_precisions.update(rule_data)
        elif rule_type == 'quantity_precision':
            self._quantity_precisions.update(rule_data)
        else:
            raise StandardizationError(f"Type de règle non supporté: {rule_type}")
    
    def _standardize_symbol(self, symbol: str, exchange: str) -> str:
        """Standardise un symbole."""
        # Appliquer les mappages personnalisés d'abord
        if symbol in self._symbol_mappings:
            return self._symbol_mappings[symbol]
        
        # Appliquer les mappages par défaut
        if symbol in self._default_symbol_mappings:
            return self._default_symbol_mappings[symbol]
        
        return symbol.upper()
    
    def _standardize_side(self, side: str) -> str:
        """Standardise un côté (buy/sell)."""
        side_lower = side.lower()
        return self._default_side_mappings.get(side_lower, side_lower)
    
    def _standardize_order_type(self, order_type: str) -> str:
        """Standardise un type d'ordre."""
        type_lower = order_type.lower()
        return self._default_order_type_mappings.get(type_lower, type_lower)
    
    def _standardize_timestamp(self, timestamp: Any) -> datetime:
        """Standardise un timestamp."""
        if timestamp is None:
            return datetime.now()
        
        if isinstance(timestamp, datetime):
            return timestamp
        
        if isinstance(timestamp, (int, float)):
            return datetime.fromtimestamp(timestamp)
        
        if isinstance(timestamp, str):
            # Essayer différents formats
            formats = [
                '%Y-%m-%d %H:%M:%S',
                '%Y-%m-%dT%H:%M:%S',
                '%Y-%m-%dT%H:%M:%SZ',
                '%Y-%m-%dT%H:%M:%S.%fZ',
            ]
            
            for fmt in formats:
                try:
                    return datetime.strptime(timestamp, fmt)
                except ValueError:
                    continue
        
        # Si tout échoue, utiliser maintenant
        return datetime.now()
    
    def _apply_price_precision(self, price: float, symbol: str, exchange: str) -> float:
        """Applique la précision de prix."""
        precision_key = f"{exchange}_{symbol}"
        precision = self._price_precisions.get(precision_key, 8)  # 8 décimales par défaut
        
        # Arrondir à la précision spécifiée
        return round(price, precision)
    
    def _apply_quantity_precision(self, quantity: float, symbol: str, exchange: str) -> float:
        """Applique la précision de quantité."""
        precision_key = f"{exchange}_{symbol}"
        precision = self._quantity_precisions.get(precision_key, 8)  # 8 décimales par défaut
        
        # Arrondir à la précision spécifiée
        return round(quantity, precision)


# Fonctions utilitaires pour la standardisation
def create_standardizer(config: Optional[Dict[str, Any]] = None) -> UnifiedTradingStandardizer:
    """
    Crée une instance de UnifiedTradingStandardizer.
    
    Args:
        config: Configuration optionnelle
        
    Returns:
        UnifiedTradingStandardizer: Instance du standardiseur
    """
    return UnifiedTradingStandardizer(config)


def standardize_data(data: Union[Dict[str, Any], List[Dict[str, Any]]],
                   exchange: str, data_type: str,
                   config: Optional[Dict[str, Any]] = None) -> Union[Any, List[Any]]:
    """
    Standardise des données de trading.
    
    Args:
        data: Données à standardiser
        exchange: Nom de l'exchange
        data_type: Type de données (order, ticker, trade, balance, position)
        config: Configuration optionnelle
        
    Returns:
        Union[Any, List[Any]]: Données standardisées
    """
    standardizer = create_standardizer(config)
    
    if isinstance(data, list):
        return standardizer.batch_standardize(data, exchange, data_type)
    else:
        if data_type == 'order':
            return standardizer.standardize_order(data, exchange)
        elif data_type == 'ticker':
            return standardizer.standardize_ticker(data, exchange)
        elif data_type == 'trade':
            return standardizer.standardize_trade(data, exchange)
        elif data_type == 'balance':
            return standardizer.standardize_balance(data, exchange)
        elif data_type == 'position':
            return standardizer.standardize_position(data, exchange)
        else:
            raise StandardizationError(f"Type de données non supporté: {data_type}")


# Export des classes et fonctions principales
__all__ = [
    'UnifiedTradingStandardizer',
    'StandardizedOrder',
    'StandardizedTicker', 
    'StandardizedTrade',
    'StandardizedBalance',
    'StandardizedPosition',
    'StandardizationError',
    'StandardizationRule',
    'create_standardizer',
    'standardize_data'
]