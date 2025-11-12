"""
Module exchanges pour le projet ARES.

Ce module contient les classes et fonctions pour la gestion des échanges,
le routage d'ordres et la réception de signaux de trading.
"""

# Import des énumérations
from .enums import (
    ExchangeStatus,
    OrderStatus,
    OrderType,
    OrderSide,
    TimeInForce,
    SignalStatus,
    ReceiverState,
    DispatchResult,
    RiskLevel,
    PositionSide,
    TradingSignal,
    RoutedOrder
)

# Import des classes principales (seront importées quand elles existeront)
try:
    from .exchange_dispatcher import ExchangeDispatcher
except ImportError:
    ExchangeDispatcher = None

try:
    from .order_router import OrderRouter
except ImportError:
    OrderRouter = None

try:
    from .trading_receiver import TradingReceiver
except ImportError:
    TradingReceiver = None

# Exporter les classes pour éviter les problèmes d'import circulaire
__all__ = [
    # Énumérations
    'ExchangeStatus',
    'OrderStatus',
    'OrderType',
    'OrderSide',
    'TimeInForce',
    'SignalStatus',
    'ReceiverState',
    'DispatchResult',
    'RiskLevel',
    'PositionSide',
    
    # Classes de données
    'TradingSignal',
    'RoutedOrder',
    
    # Classes principales
    'ExchangeDispatcher',
    'OrderRouter',
    'TradingReceiver'
]