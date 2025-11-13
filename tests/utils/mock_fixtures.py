"""
Utilitaires de Mock standardisés pour les tests du projet ARES

Ce module fournit des fixtures et utilitaires pour créer des mocks complexes
et gérer les dépendances manquantes de manière standardisée.
"""

import pytest
import asyncio
import uuid
from unittest.mock import Mock, AsyncMock, MagicMock, patch
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Callable
import pandas as pd
import numpy as np


class MockExchangeStatus:
    """
    Mock complet pour ExchangeStatus avec tous les attributs requis.
    
    Cette classe remplace l'enum ExchangeStatus manquant dans les tests.
    """
    ACTIVE = 'ACTIVE'
    DISABLED = 'DISABLED'
    INACTIVE = 'INACTIVE'
    MAINTENANCE = 'MAINTENANCE'
    ERROR = 'ERROR'
    
    @classmethod
    def all_statuses(cls) -> List[str]:
        """Retourne tous les statuts valides."""
        return [cls.ACTIVE, cls.DISABLED, cls.INACTIVE, cls.MAINTENANCE, cls.ERROR]
    
    @classmethod
    def is_valid(cls, status: str) -> bool:
        """Vérifie si un statut est valide."""
        return status.upper() in cls.all_statuses()


class MockOrderStatus:
    """
    Mock complet pour OrderStatus avec tous les attributs requis.
    """
    OPEN = 'OPEN'
    FILLED = 'FILLED'
    PARTIALLY_FILLED = 'PARTIALLY_FILLED'
    CANCELLED = 'CANCELLED'
    REJECTED = 'REJECTED'
    EXPIRED = 'EXPIRED'
    PENDING = 'PENDING'
    SUBMITTED = 'SUBMITTED'
    
    @classmethod
    def all_statuses(cls) -> List[str]:
        """Retourne tous les statuts valides."""
        return [
            cls.OPEN, cls.FILLED, cls.PARTIALLY_FILLED, cls.CANCELLED,
            cls.REJECTED, cls.EXPIRED, cls.PENDING, cls.SUBMITTED
        ]
    
    @classmethod
    def is_valid(cls, status: str) -> bool:
        """Vérifie si un statut est valide."""
        return status.upper() in cls.all_statuses()


class MockOrderType:
    """
    Mock complet pour OrderType avec tous les attributs requis.
    """
    MARKET = 'market'
    LIMIT = 'limit'
    STOP = 'stop'
    STOP_LIMIT = 'stop_limit'
    
    @classmethod
    def all_types(cls) -> List[str]:
        """Retourne tous les types valides."""
        return [cls.MARKET, cls.LIMIT, cls.STOP, cls.STOP_LIMIT]
    
    @classmethod
    def is_valid(cls, order_type: str) -> bool:
        """Vérifie si un type est valide."""
        return order_type.lower() in cls.all_types()


class MockOrderSide:
    """
    Mock complet pour OrderSide avec tous les attributs requis.
    """
    BUY = 'buy'
    SELL = 'sell'
    
    @classmethod
    def all_sides(cls) -> List[str]:
        """Retourne tous les côtés valides."""
        return [cls.BUY, cls.SELL]
    
    @classmethod
    def is_valid(cls, side: str) -> bool:
        """Vérifie si un côté est valide."""
        return side.lower() in cls.all_sides()


class MockSimulatorConfig:
    """
    Mock complet pour SimulatorConfig avec tous les attributs requis.
    """
    def __init__(self):
        self.default_taker_fee = 0.001
        self.default_maker_fee = 0.001
        self.max_slippage_pct = 0.0005
        self.slippage_model = MockSlippageModel.ORDERBOOK
        self.min_order_size = 0.001
        self.max_order_size = 1000.0
        self.enable_order_validation = True


class MockSlippageModel:
    """
    Mock complet pour SlippageModel avec tous les attributs requis.
    """
    ORDERBOOK = 'orderbook'
    FIXED = 'fixed'
    PERCENTAGE = 'percentage'
    
    @classmethod
    def all_models(cls) -> List[str]:
        """Retourne tous les modèles valides."""
        return [cls.ORDERBOOK, cls.FIXED, cls.PERCENTAGE]


class MockPaperTradingSimulator:
    """
    Mock complet pour PaperTradingSimulator avec méthodes asynchrones configurées.
    """
    def __init__(self, config=None, exchange="binance", initial_balance=10000.0, direction_constraint="both"):
        self.config = config or MockSimulatorConfig()
        self.exchange = exchange
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.direction_constraint = direction_constraint
        self.positions = []
        self.trade_history = []
        
        # Configuration des méthodes asynchrones
        self.simulate_order = AsyncMock(side_effect=self._simulate_order_side_effect)
        self.get_positions = Mock(return_value=self.positions)
        self.get_trade_history = Mock(return_value=self.trade_history)
        self.get_performance_metrics = Mock(return_value=self._get_default_metrics())
        self.generate_daily_report = AsyncMock(return_value=True)
    
    def _simulate_order_side_effect(self, symbol, side, order_type, quantity, price, order_book):
        """Side effect pour simulate_order."""
        order_id = f"order_{symbol}_{side}_{quantity}_{uuid.uuid4().hex[:8]}"
        
        if quantity > 1000:  # Solde insuffisant
            return {
                'status': MockOrderStatus.REJECTED,
                'symbol': symbol,
                'side': side.upper(),
                'quantity': quantity,
                'rejectedReason': 'Insufficient balance'
            }
        
        if quantity < 0:  # Quantité invalide
            return {
                'status': MockOrderStatus.REJECTED,
                'symbol': symbol,
                'side': side.upper(),
                'quantity': quantity,
                'rejectedReason': 'Invalid quantity'
            }
            
        if not symbol:  # Symbole vide
            return {
                'status': MockOrderStatus.REJECTED,
                'symbol': symbol,
                'side': side.upper(),
                'quantity': quantity,
                'rejectedReason': 'Invalid symbol'
            }
        
        # Ordre valide
        return {
            'status': MockOrderStatus.FILLED,
            'orderId': order_id,
            'symbol': symbol,
            'side': side.upper(),
            'order_type': order_type,
            'quantity': quantity,
            'price': price or 2000.0,
            'fee': quantity * (price or 2000.0) * self.config.default_taker_fee,
            'slippagePct': self.config.max_slippage_pct
        }
    
    def _get_default_metrics(self):
        """Retourne les métriques de performance par défaut."""
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'total_fees': 0.0,
            'current_balance': self.current_balance,
            'initial_balance': self.initial_balance
        }


class MockExchangeDispatcher:
    """
    Mock complet pour ExchangeDispatcher avec méthodes asynchrones configurées.
    """
    def __init__(self, exchange_registry=None):
        self.exchange_registry = exchange_registry or AsyncMock()
        self._running = False
        self._monitoring_task = None
        self.exchange_status = {}
        self.dispatch_history = []
        
        # Configuration des méthodes asynchrones
        self.start = AsyncMock(side_effect=self._start_side_effect)
        self.stop = AsyncMock(side_effect=self._stop_side_effect)
        self.dispatch_to_exchange = AsyncMock(side_effect=self._dispatch_to_exchange_side_effect)
        self.dispatch_to_best_exchange = AsyncMock(side_effect=self._dispatch_to_best_exchange_side_effect)
        self.dispatch_to_multiple_exchanges = AsyncMock(side_effect=self._dispatch_to_multiple_exchanges_side_effect)
        self.get_best_exchange = AsyncMock(side_effect=self._get_best_exchange_side_effect)
        self.get_exchange_status = AsyncMock(side_effect=self._get_exchange_status_side_effect)
        self.get_all_exchanges_status = AsyncMock(side_effect=self._get_all_exchanges_status_side_effect)
        self.update_exchange_status = AsyncMock(side_effect=self._update_exchange_status_side_effect)
        self.disable_exchange = AsyncMock(side_effect=self._disable_exchange_side_effect)
        self.enable_exchange = AsyncMock(side_effect=self._enable_exchange_side_effect)
        self.get_dispatch_history = AsyncMock(side_effect=self._get_dispatch_history_side_effect)
        self.get_statistics = AsyncMock(side_effect=self._get_statistics_side_effect)
        self.check_exchange_health = AsyncMock(side_effect=self._check_exchange_health_side_effect)
        self.get_exchange_load = AsyncMock(side_effect=self._get_exchange_load_side_effect)
        self.get_exchange_latency = AsyncMock(side_effect=self._get_exchange_latency_side_effect)
    
    def _start_side_effect(self):
        """Side effect pour start."""
        self._running = True
        return True
    
    def _stop_side_effect(self):
        """Side effect pour stop."""
        self._running = False
        self._monitoring_task = None
        return False
    
    def _dispatch_to_exchange_side_effect(self, exchange, symbol, side, order_type, quantity, price):
        """Side effect pour dispatch_to_exchange."""
        if exchange == 'nonexistent_exchange':
            return {'success': False, 'error': f'Exchange {exchange} not found'}
        return {
            'success': True,
            'order_id': f'order_{uuid.uuid4().hex[:8]}',
            'exchange': exchange,
            'status': 'submitted',
            'timestamp': datetime.now()
        }
    
    def _dispatch_to_best_exchange_side_effect(self, symbol, side, order_type, quantity):
        """Side effect pour dispatch_to_best_exchange."""
        return {
            'success': True,
            'order_id': f'order_{uuid.uuid4().hex[:8]}',
            'exchange': 'binance',
            'status': 'submitted'
        }
    
    def _dispatch_to_multiple_exchanges_side_effect(self, symbol, side, order_type, total_quantity, exchanges, allocation):
        """Side effect pour dispatch_to_multiple_exchanges."""
        if allocation and sum(allocation.values()) > 1.0:
            return {'success': False, 'error': 'Invalid allocation: total exceeds 1.0'}
        
        orders = []
        for exchange in exchanges:
            orders.append({
                'exchange': exchange,
                'order_id': f'order_{uuid.uuid4().hex[:8]}',
                'quantity': total_quantity * allocation.get(exchange, 0.5),
                'allocation': allocation.get(exchange, 0.5)
            })
        
        return {'success': True, 'orders': orders}
    
    def _get_best_exchange_side_effect(self, symbol, side, order_type):
        """Side effect pour get_best_exchange."""
        return 'binance' if side == 'buy' else 'okx'
    
    def _get_exchange_status_side_effect(self, exchange):
        """Side effect pour get_exchange_status."""
        if exchange == 'nonexistent_exchange':
            return {'success': False, 'error': f'Exchange {exchange} not found'}
        
        return {
            'success': True,
            'exchange': exchange,
            'status': MockExchangeStatus.ACTIVE,
            'last_check': datetime.now(),
            'latency': 50,
            'error_rate': 0.01
        }
    
    def _get_all_exchanges_status_side_effect(self):
        """Side effect pour get_all_exchanges_status."""
        return {
            'success': True,
            'exchanges': {
                'binance': {
                    'success': True,
                    'exchange': 'binance',
                    'status': MockExchangeStatus.ACTIVE,
                    'last_check': datetime.now(),
                    'latency': 50,
                    'error_rate': 0.01
                },
                'okx': {
                    'success': True,
                    'exchange': 'okx',
                    'status': MockExchangeStatus.ACTIVE,
                    'last_check': datetime.now(),
                    'latency': 60,
                    'error_rate': 0.02
                }
            }
        }
    
    def _update_exchange_status_side_effect(self, exchange, status, latency, error_rate):
        """Side effect pour update_exchange_status."""
        return {
            'success': True,
            'exchange': exchange,
            'status': status,
            'latency': latency,
            'error_rate': error_rate
        }
    
    def _disable_exchange_side_effect(self, exchange, reason='Maintenance'):
        """Side effect pour disable_exchange."""
        return {
            'success': True,
            'exchange': exchange,
            'status': MockExchangeStatus.DISABLED,
            'reason': reason
        }
    
    def _enable_exchange_side_effect(self, exchange):
        """Side effect pour enable_exchange."""
        return {
            'success': True,
            'exchange': exchange,
            'status': MockExchangeStatus.ACTIVE
        }
    
    def _get_dispatch_history_side_effect(self, exchange=None, symbol=None):
        """Side effect pour get_dispatch_history."""
        history = []
        for entry in self.dispatch_history:
            if exchange and entry.get('exchange') != exchange:
                continue
            if symbol and entry.get('symbol') != symbol:
                continue
            history.append(entry)
        
        return {'success': True, 'history': history, 'count': len(history)}
    
    def _get_statistics_side_effect(self):
        """Side effect pour get_statistics."""
        return {
            'success': True,
            'statistics': {
                'total_dispatches': len(self.dispatch_history),
                'successful_dispatches': len(self.dispatch_history),
                'failed_dispatches': 0,
                'by_exchange': {},
                'by_symbol': {},
                'by_side': {}
            }
        }
    
    def _check_exchange_health_side_effect(self, exchange):
        """Side effect pour check_exchange_health."""
        if exchange == 'binance':
            return {
                'success': True,
                'exchange': exchange,
                'healthy': True,
                'latency': 50,
                'timestamp': datetime.now()
            }
        else:
            return {
                'success': True,
                'exchange': exchange,
                'healthy': False,
                'latency': 1000,
                'timestamp': datetime.now(),
                'error': 'Connection timeout'
            }
    
    def _get_exchange_load_side_effect(self, exchange):
        """Side effect pour get_exchange_load."""
        return 0.8 if exchange == 'binance' else 0.3
    
    def _get_exchange_latency_side_effect(self, exchange):
        """Side effect pour get_exchange_latency."""
        return 20 if exchange == 'binance' else 50


class MockOrderManager:
    """
    Mock complet pour OrderManager avec méthodes asynchrones configurées.
    """
    def __init__(self, config=None):
        self.config = config or {}
        self.orders = []
        self.active_orders = []
        self.completed_orders = []
        
        # Configuration des méthodes asynchrones
        self.start = AsyncMock()
        self.stop = AsyncMock()
        self.create_order = AsyncMock(side_effect=self._create_order_side_effect)
        self.cancel_order = AsyncMock(side_effect=self._cancel_order_side_effect)
        self.get_order = AsyncMock(side_effect=self._get_order_side_effect)
        self.get_active_orders = AsyncMock(side_effect=self._get_active_orders_side_effect)
        self.get_completed_orders = AsyncMock(side_effect=self._get_completed_orders_side_effect)
        self.update_order_status = AsyncMock(side_effect=self._update_order_status_side_effect)
        self.get_order_statistics = AsyncMock(side_effect=self._get_order_statistics_side_effect)
        self.get_orders_by_symbol = AsyncMock(side_effect=self._get_orders_by_symbol_side_effect)
        self.get_orders_by_side = AsyncMock(side_effect=self._get_orders_by_side_side_effect)
        self.batch_create_orders = AsyncMock(side_effect=self._batch_create_orders_side_effect)
        self.batch_cancel_orders = AsyncMock(side_effect=self._batch_cancel_orders_side_effect)
    
    def _create_order_side_effect(self, symbol, side, order_type, quantity, price=None, stop_price=None):
        """Side effect pour create_order."""
        order_id = f'order_{uuid.uuid4().hex[:8]}'
        order = {
            'order_id': order_id,
            'symbol': symbol,
            'side': side,
            'order_type': order_type,
            'quantity': quantity,
            'price': price,
            'stop_price': stop_price,
            'status': MockOrderStatus.OPEN,
            'timestamp': datetime.now(),
            'filled_quantity': 0.0
        }
        
        # Validation basique
        if not symbol or 'INVALID' in symbol.upper():
            return {'success': False, 'error': 'Invalid symbol'}
        
        if side not in MockOrderSide.all_sides():
            return {'success': False, 'error': 'Invalid side'}
        
        if order_type not in MockOrderType.all_types():
            return {'success': False, 'error': 'Invalid order type'}
        
        if quantity <= 0:
            return {'success': False, 'error': 'Invalid quantity'}
        
        if quantity > 1000:  # Solde insuffisant
            return {'success': False, 'error': 'Insufficient balance'}
        
        self.orders.append(order)
        self.active_orders.append(order)
        
        return {'success': True, 'order': order}
    
    def _cancel_order_side_effect(self, order_id):
        """Side effect pour cancel_order."""
        for i, order in enumerate(self.active_orders):
            if order['order_id'] == order_id:
                order['status'] = MockOrderStatus.CANCELLED
                self.active_orders.pop(i)
                self.completed_orders.append(order)
                return {'success': True, 'order_id': order_id, 'status': MockOrderStatus.CANCELLED}
        
        return {'success': False, 'error': f'Order {order_id} not found'}
    
    def _get_order_side_effect(self, order_id):
        """Side effect pour get_order."""
        for order in self.orders:
            if order['order_id'] == order_id:
                return {'success': True, 'order': order}
        
        return {'success': False, 'error': f'Order {order_id} not found'}
    
    def _get_active_orders_side_effect(self, symbol=None, side=None, order_type=None):
        """Side effect pour get_active_orders."""
        orders = self.active_orders.copy()
        
        if symbol:
            orders = [o for o in orders if o['symbol'] == symbol]
        
        if side:
            orders = [o for o in orders if o['side'] == side]
        
        if order_type:
            orders = [o for o in orders if o['order_type'] == order_type]
        
        return {'success': True, 'orders': orders}
    
    def _get_completed_orders_side_effect(self):
        """Side effect pour get_completed_orders."""
        return {'success': True, 'orders': self.completed_orders}
    
    def _update_order_status_side_effect(self, order_id, new_status, filled_quantity=None):
        """Side effect pour update_order_status."""
        for order in self.orders:
            if order['order_id'] == order_id:
                old_status = order['status']
                order['status'] = new_status
                if filled_quantity is not None:
                    order['filled_quantity'] = filled_quantity
                
                return {
                    'success': True,
                    'order_id': order_id,
                    'old_status': old_status,
                    'new_status': new_status,
                    'filled_quantity': filled_quantity
                }
        
        return {'success': False, 'error': f'Order {order_id} not found'}
    
    def _get_order_statistics_side_effect(self):
        """Side effect pour get_order_statistics."""
        stats = {
            'total_orders': len(self.orders),
            'active_orders': len(self.active_orders),
            'completed_orders': len(self.completed_orders),
            'filled_orders': len([o for o in self.completed_orders if o['status'] == MockOrderStatus.FILLED]),
            'cancelled_orders': len([o for o in self.completed_orders if o['status'] == MockOrderStatus.CANCELLED]),
            'by_symbol': {},
            'by_order_type': {},
            'by_side': {}
        }
        
        return {'success': True, 'statistics': stats}
    
    def _get_orders_by_symbol_side_effect(self, symbol):
        """Side effect pour get_orders_by_symbol."""
        orders = [o for o in self.orders if o['symbol'] == symbol]
        return {'success': True, 'orders': orders}
    
    def _get_orders_by_side_side_effect(self, side):
        """Side effect pour get_orders_by_side."""
        orders = [o for o in self.orders if o['side'] == side]
        return {'success': True, 'orders': orders}
    
    def _batch_create_orders_side_effect(self, orders_data):
        """Side effect pour batch_create_orders."""
        results = {'orders': [], 'failed_orders': []}
        
        for order_data in orders_data:
            result = self._create_order_side_effect(
                order_data.get('symbol'),
                order_data.get('side'),
                order_data.get('order_type'),
                order_data.get('quantity'),
                order_data.get('price'),
                order_data.get('stop_price')
            )
            
            if result['success']:
                results['orders'].append(result)
            else:
                results['failed_orders'].append({
                    'order_data': order_data,
                    'error': result.get('error', 'Unknown error')
                })
        
        return {'success': True, **results}
    
    def _batch_cancel_orders_side_effect(self, order_ids):
        """Side effect pour batch_cancel_orders."""
        results = {'orders': [], 'failed_orders': []}
        
        for order_id in order_ids:
            result = self._cancel_order_side_effect(order_id)
            
            if result['success']:
                results['orders'].append(result)
            else:
                results['failed_orders'].append({
                    'order_id': order_id,
                    'error': result.get('error', 'Unknown error')
                })
        
        return {'success': True, **results}


class MockHelpers:
    """
    Classe utilitaire pour aider à la configuration des mocks complexes.
    """
    
    @staticmethod
    def configure_async_mock_with_side_effect(mock_obj: AsyncMock, side_effect_func: Callable) -> AsyncMock:
        """
        Configure un AsyncMock avec une fonction de side_effect.
        
        Args:
            mock_obj: L'objet AsyncMock à configurer
            side_effect_func: La fonction de side_effect à appliquer
            
        Returns:
            L'objet AsyncMock configuré
        """
        mock_obj.side_effect = side_effect_func
        return mock_obj
    
    @staticmethod
    def configure_mock_attributes(mock_obj: Mock, attributes: Dict[str, Any]) -> Mock:
        """
        Configure les attributs d'un Mock.
        
        Args:
            mock_obj: L'objet Mock à configurer
            attributes: Dictionnaire des attributs à configurer
            
        Returns:
            L'objet Mock configuré
        """
        for attr_name, attr_value in attributes.items():
            setattr(mock_obj, attr_name, attr_value)
        return mock_obj
    
    @staticmethod
    def create_mock_with_methods(methods: Dict[str, Any]) -> Mock:
        """
        Crée un Mock avec des méthodes préconfigurées.
        
        Args:
            methods: Dictionnaire des méthodes à configurer
            
        Returns:
            L'objet Mock configuré
        """
        mock_obj = Mock()
        for method_name, method_impl in methods.items():
            setattr(mock_obj, method_name, method_impl)
        return mock_obj
    
    @staticmethod
    def create_async_mock_with_methods(methods: Dict[str, Any]) -> AsyncMock:
        """
        Crée un AsyncMock avec des méthodes préconfigurées.
        
        Args:
            methods: Dictionnaire des méthodes à configurer
            
        Returns:
            L'objet AsyncMock configuré
        """
        mock_obj = AsyncMock()
        for method_name, method_impl in methods.items():
            setattr(mock_obj, method_name, method_impl)
        return mock_obj


class DependencyManager:
    """
    Gestionnaire pour les dépendances manquantes.
    
    Cette classe fournit des patterns pour gérer les modules manquants
    de manière standardisée dans les tests.
    """
    
    @staticmethod
    def safe_import(module_path: str, fallback_class: Optional[type] = None, fallback_value: Any = None):
        """
        Importe un module en toute sécurité avec fallback.
        
        Args:
            module_path: Chemin du module à importer
            fallback_class: Classe de fallback si l'import échoue
            fallback_value: Valeur de fallback si l'import échoue
            
        Returns:
            Le module importé ou le fallback
        """
        try:
            parts = module_path.split('.')
            module = __import__(parts[0])
            for part in parts[1:]:
                module = getattr(module, part)
            return module
        except (ImportError, AttributeError) as e:
            print(f"DEBUG: Erreur import {module_path}: {e}")
            if fallback_class is not None:
                return fallback_class
            return fallback_value
    
    @staticmethod
    def create_mock_for_missing_class(class_name: str, base_class: type = Mock):
        """
        Crée un mock pour une classe manquante.
        
        Args:
            class_name: Nom de la classe manquante
            base_class: Classe de base pour le mock
            
        Returns:
            Une classe mock configurée
        """
        class MockClass(base_class):
            def __init__(self, *args, **kwargs):
                super().__init__()
                self._class_name = class_name
                
            def __repr__(self):
                return f"Mock{class_name}"
        
        MockClass.__name__ = f"Mock{class_name}"
        return MockClass
    
    @staticmethod
    def patch_missing_module(module_path: str, mock_class: type):
        """
        Patch un module manquant avec un mock.
        
        Args:
            module_path: Chemin du module à patcher
            mock_class: Classe de mock à utiliser
        """
        patcher = patch(module_path, mock_class)
        patcher.start()
        return patcher


# Fixtures pytest
@pytest.fixture
def mock_exchange_status():
    """Fixture pour MockExchangeStatus."""
    return MockExchangeStatus


@pytest.fixture
def mock_order_status():
    """Fixture pour MockOrderStatus."""
    return MockOrderStatus


@pytest.fixture
def mock_order_type():
    """Fixture pour MockOrderType."""
    return MockOrderType


@pytest.fixture
def mock_order_side():
    """Fixture pour MockOrderSide."""
    return MockOrderSide


@pytest.fixture
def mock_simulator_config():
    """Fixture pour MockSimulatorConfig."""
    return MockSimulatorConfig()


@pytest.fixture
def mock_slippage_model():
    """Fixture pour MockSlippageModel."""
    return MockSlippageModel


@pytest.fixture
def mock_paper_trading_simulator():
    """Fixture pour MockPaperTradingSimulator."""
    return MockPaperTradingSimulator


@pytest.fixture
def mock_exchange_dispatcher():
    """Fixture pour MockExchangeDispatcher."""
    return MockExchangeDispatcher


@pytest.fixture
def mock_order_manager():
    """Fixture pour MockOrderManager."""
    return MockOrderManager


@pytest.fixture
def mock_helpers():
    """Fixture pour MockHelpers."""
    return MockHelpers


@pytest.fixture
def dependency_manager():
    """Fixture pour DependencyManager."""
    return DependencyManager


@pytest.fixture
def sample_order_data():
    """Fixture pour des données d'ordre de test."""
    return {
        'symbol': 'ETHUSDT',
        'side': 'buy',
        'order_type': 'market',
        'quantity': 0.1,
        'price': 2000.0
    }


@pytest.fixture
def sample_order_book():
    """Fixture pour un order book de test."""
    return {
        'bids': [(1999.9, 100.0), (1999.8, 50.0)],
        'asks': [(2000.1, 100.0), (2000.2, 50.0)]
    }


@pytest.fixture
def sample_market_data():
    """Fixture pour des données de marché de test."""
    return {
        'binance': {'price': 2000.0, 'volume': 100.0, 'spread': 0.1},
        'okx': {'price': 2001.0, 'volume': 80.0, 'spread': 0.15}
    }


# Configuration des imports pour les tests
def setup_test_imports():
    """
    Configure les imports pour les tests en utilisant les mocks.
    
    Cette fonction doit être appelée au début des fichiers de test
    pour remplacer les imports manquants par des mocks.
    """
    # Remplacer les imports manquants par des mocks
    globals().update({
        'PaperTradingSimulator': MockPaperTradingSimulator,
        'SimulatorConfig': MockSimulatorConfig,
        'SlippageModel': MockSlippageModel,
        'ExchangeStatus': MockExchangeStatus,
        'OrderStatus': MockOrderStatus,
        'OrderType': MockOrderType,
        'OrderSide': MockOrderSide,
        'ExchangeDispatcher': MockExchangeDispatcher,
        'OrderManager': MockOrderManager
    })


# Exportations pour un accès facile
__all__ = [
    'MockExchangeStatus',
    'MockOrderStatus', 
    'MockOrderType',
    'MockOrderSide',
    'MockSimulatorConfig',
    'MockSlippageModel',
    'MockPaperTradingSimulator',
    'MockExchangeDispatcher',
    'MockOrderManager',
    'MockHelpers',
    'DependencyManager',
    'setup_test_imports'
]