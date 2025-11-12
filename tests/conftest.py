"""
Fixtures communes pour les tests du projet Ares

Ce module contient les fixtures partagées utilisées par tous les tests.
"""

import pytest
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock
import json
import tempfile
import os
from typing import Dict, Any, List
from src.utils.tprint import tprint_logged, LogLevel

# Variable pour indiquer si tprint est disponible
TPRINT_AVAILABLE = True


@pytest.fixture(scope="session")
def event_loop():
    """Créer une boucle d'événements pour les tests asynchrones."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_market_data():
    """Fixture pour les données de marché mockées."""
    return {
        'symbol': 'ETHUSDT',
        'timestamp': datetime.now(),
        'open': 2000.0,
        'high': 2050.0,
        'low': 1950.0,
        'close': 2025.0,
        'volume': 1000.0,
        'bid': 2024.5,
        'ask': 2025.5,
        'spread': 1.0
    }


@pytest.fixture
def mock_order_data():
    """Fixture pour les données d'ordre mockées."""
    return {
        'symbol': 'ETHUSDT',
        'side': 'buy',
        'order_type': 'market',
        'quantity': 0.1,
        'price': 2000.0,
        'exchange': 'binance',
        'timestamp': datetime.now(),
        'time_in_force': 'GTC',
        'status': 'open'
    }


@pytest.fixture
def mock_trading_signal():
    """Fixture pour les signaux de trading mockés."""
    return {
        'symbol': 'ETHUSDT',
        'side': 'buy',
        'order_type': 'market',
        'quantity': 0.1,
        'price': 2000.0,
        'exchange': None,  # Sera déterminé automatiquement
        'timestamp': datetime.now(),
        'confidence': 0.85,
        'strategy': 'test_strategy',
        'signal_id': 'test_signal_123',
        'metadata': {
            'source': 'test',
            'version': '1.0'
        }
    }


@pytest.fixture
def mock_position_data():
    """Fixture pour les données de position mockées."""
    return {
        'symbol': 'ETHUSDT',
        'side': 'long',
        'quantity': 0.1,
        'entry_price': 2000.0,
        'current_price': 2025.0,
        'unrealized_pnl': 2.5,
        'realized_pnl': 0.0,
        'timestamp': datetime.now(),
        'exchange': 'binance',
        'fees': 0.2
    }


@pytest.fixture
def mock_portfolio_data():
    """Fixture pour les données de portefeuille mockées."""
    return {
        'total_value': 10000.0,
        'available_balance': 5000.0,
        'used_balance': 5000.0,
        'total_pnl': 100.0,
        'positions': [
            {
                'symbol': 'ETHUSDT',
                'side': 'long',
                'quantity': 0.1,
                'entry_price': 2000.0,
                'current_price': 2025.0,
                'unrealized_pnl': 2.5,
                'realized_pnl': 0.0
            },
            {
                'symbol': 'BTCUSDT',
                'side': 'short',
                'quantity': 0.05,
                'entry_price': 50000.0,
                'current_price': 49500.0,
                'unrealized_pnl': 2.5,
                'realized_pnl': 0.0
            }
        ],
        'timestamp': datetime.now()
    }


@pytest.fixture
def mock_exchange_config():
    """Fixture pour la configuration d'exchange mockée."""
    return {
        'binance': {
            'api_key': 'test_binance_key',
            'api_secret': 'test_binance_secret',
            'sandbox': True,
            'timeout': 30,
            'rate_limit': 10,
            'fees': {
                'maker': 0.001,
                'taker': 0.001
            }
        },
        'okx': {
            'api_key': 'test_okx_key',
            'api_secret': 'test_okx_secret',
            'passphrase': 'test_okx_passphrase',
            'sandbox': True,
            'timeout': 30,
            'rate_limit': 20,
            'fees': {
                'maker': 0.0008,
                'taker': 0.001
            }
        }
    }


@pytest.fixture
def mock_trading_config():
    """Fixture pour la configuration de trading mockée."""
    return {
        'max_position_size': 0.5,
        'max_daily_loss': 100.0,
        'max_positions_per_symbol': 1,
        'risk_management': {
            'stop_loss_pct': 0.02,
            'take_profit_pct': 0.05,
            'max_leverage': 3.0
        },
        'execution': {
            'default_exchange': 'binance',
            'slippage_tolerance': 0.001,
            'order_timeout': 60,
            'retry_attempts': 3
        }
    }


@pytest.fixture
def mock_database_connection():
    """Fixture pour une connexion de base de données mockée."""
    mock_conn = Mock()
    mock_conn.execute = Mock(return_value=Mock())
    mock_conn.commit = Mock()
    mock_conn.rollback = Mock()
    mock_conn.close = Mock()
    return mock_conn


@pytest.fixture
def mock_redis_connection():
    """Fixture pour une connexion Redis mockée."""
    mock_redis = Mock()
    mock_redis.get = Mock(return_value=None)
    mock_redis.set = Mock(return_value=True)
    mock_redis.delete = Mock(return_value=True)
    mock_redis.exists = Mock(return_value=False)
    mock_redis.expire = Mock(return_value=True)
    return mock_redis


@pytest.fixture
def mock_logger():
    """Fixture pour un logger mocké."""
    logger = Mock()
    logger.debug = Mock()
    logger.info = Mock()
    logger.warning = Mock()
    logger.error = Mock()
    logger.critical = Mock()
    return logger


@pytest.fixture
def sample_ohlcv_data():
    """Fixture pour des données OHLCV d'exemple."""
    dates = pd.date_range(start='2023-01-01', end='2023-01-10', freq='1H')
    np.random.seed(42)
    
    # Génération de données synthétiques
    close_prices = 2000 + np.cumsum(np.random.normal(0, 10, len(dates)))
    high_prices = close_prices + np.random.uniform(0, 20, len(dates))
    low_prices = close_prices - np.random.uniform(0, 20, len(dates))
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = close_prices[0]
    
    volumes = np.random.uniform(100, 1000, len(dates))
    
    return pd.DataFrame({
        'timestamp': dates,
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volumes
    })


@pytest.fixture
def sample_market_data_frame():
    """Fixture pour un DataFrame de données de marché."""
    dates = pd.date_range(start='2023-01-01', periods=100, freq='1H')
    np.random.seed(42)
    
    return pd.DataFrame({
        'timestamp': dates,
        'symbol': 'ETHUSDT',
        'open': 2000 + np.random.normal(0, 10, 100),
        'high': 2010 + np.random.normal(0, 10, 100),
        'low': 1990 + np.random.normal(0, 10, 100),
        'close': 2000 + np.random.normal(0, 10, 100),
        'volume': np.random.uniform(100, 1000, 100),
        'bid': 1999 + np.random.normal(0, 5, 100),
        'ask': 2001 + np.random.normal(0, 5, 100),
        'spread': np.random.uniform(0.5, 2.0, 100)
    })


@pytest.fixture
def temp_config_file():
    """Fixture pour un fichier de configuration temporaire."""
    config_data = {
        'test': {
            'value': 123,
            'nested': {
                'key': 'value'
            }
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config_data, f)
        temp_file = f.name
    
    yield temp_file
    
    # Nettoyage
    if os.path.exists(temp_file):
        os.unlink(temp_file)


@pytest.fixture
def temp_csv_file():
    """Fixture pour un fichier CSV temporaire."""
    data = {
        'timestamp': ['2023-01-01 00:00:00', '2023-01-01 01:00:00'],
        'open': [2000.0, 2010.0],
        'high': [2010.0, 2020.0],
        'low': [1990.0, 2000.0],
        'close': [2005.0, 2015.0],
        'volume': [100.0, 150.0]
    }
    
    df = pd.DataFrame(data)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        df.to_csv(f.name, index=False)
        temp_file = f.name
    
    yield temp_file
    
    # Nettoyage
    if os.path.exists(temp_file):
        os.unlink(temp_file)


@pytest.fixture
def mock_exchange_client():
    """Fixture pour un client d'exchange mocké."""
    client = Mock()
    
    # Méthodes de trading
    client.create_order = AsyncMock(return_value={
        'id': 'test_order_123',
        'symbol': 'ETHUSDT',
        'status': 'open',
        'filled': 0.0,
        'remaining': 0.1
    })
    
    client.cancel_order = AsyncMock(return_value={
        'id': 'test_order_123',
        'status': 'canceled'
    })
    
    client.fetch_order = AsyncMock(return_value={
        'id': 'test_order_123',
        'symbol': 'ETHUSDT',
        'status': 'closed',
        'filled': 0.1,
        'remaining': 0.0
    })
    
    client.fetch_balance = AsyncMock(return_value={
        'USDT': {'free': 10000.0, 'used': 5000.0, 'total': 15000.0},
        'ETH': {'free': 1.0, 'used': 0.5, 'total': 1.5}
    })
    
    client.fetch_ticker = AsyncMock(return_value={
        'symbol': 'ETHUSDT',
        'bid': 2024.5,
        'ask': 2025.5,
        'last': 2025.0,
        'baseVolume': 1000.0,
        'quoteVolume': 2025000.0
    })
    
    client.fetch_ohlcv = AsyncMock(return_value=[
        [1672531200000, 2000.0, 2010.0, 1990.0, 2005.0, 100.0],  # OHLCV
        [1672534800000, 2005.0, 2015.0, 1995.0, 2010.0, 150.0]
    ])
    
    return client


@pytest.fixture
def mock_websocket_connection():
    """Fixture pour une connexion WebSocket mockée."""
    ws = Mock()
    ws.connect = AsyncMock(return_value=True)
    ws.disconnect = AsyncMock(return_value=True)
    ws.send = AsyncMock(return_value=True)
    ws.recv = AsyncMock(return_value=json.dumps({
        'type': 'ticker',
        'symbol': 'ETHUSDT',
        'price': 2025.0,
        'timestamp': datetime.now().isoformat()
    }))
    return ws


@pytest.fixture
def mock_notification_service():
    """Fixture pour un service de notification mocké."""
    service = Mock()
    service.send_email = AsyncMock(return_value=True)
    service.send_sms = AsyncMock(return_value=True)
    service.send_webhook = AsyncMock(return_value=True)
    service.send_slack = AsyncMock(return_value=True)
    return service


@pytest.fixture
def mock_risk_manager():
    """Fixture pour un gestionnaire de risque mocké."""
    manager = Mock()
    manager.check_position_size = AsyncMock(return_value={'allowed': True, 'max_size': 0.5})
    manager.check_risk_limits = AsyncMock(return_value={'allowed': True, 'risk_score': 0.3})
    manager.calculate_stop_loss = Mock(return_value=1980.0)
    manager.calculate_take_profit = Mock(return_value=2100.0)
    manager.update_position_risk = AsyncMock(return_value=True)
    return manager


@pytest.fixture
def mock_performance_tracker():
    """Fixture pour un tracker de performance mocké."""
    tracker = Mock()
    tracker.record_trade = AsyncMock(return_value=True)
    tracker.calculate_metrics = AsyncMock(return_value={
        'total_return': 0.05,
        'sharpe_ratio': 1.2,
        'max_drawdown': 0.02,
        'win_rate': 0.6,
        'profit_factor': 1.5
    })
    tracker.get_daily_pnl = AsyncMock(return_value=100.0)
    tracker.get_monthly_pnl = AsyncMock(return_value=2000.0)
    return tracker


@pytest.fixture
def sample_strategy_signals():
    """Fixture pour des signaux de stratégie d'exemple."""
    return [
        {
            'timestamp': datetime.now() - timedelta(hours=3),
            'symbol': 'ETHUSDT',
            'signal': 'BUY',
            'confidence': 0.85,
            'price': 2000.0,
            'strategy': 'test_strategy'
        },
        {
            'timestamp': datetime.now() - timedelta(hours=2),
            'symbol': 'BTCUSDT',
            'signal': 'SELL',
            'confidence': 0.75,
            'price': 50000.0,
            'strategy': 'test_strategy'
        },
        {
            'timestamp': datetime.now() - timedelta(hours=1),
            'symbol': 'ETHUSDT',
            'signal': 'HOLD',
            'confidence': 0.60,
            'price': 2025.0,
            'strategy': 'test_strategy'
        }
    ]


@pytest.fixture
def sample_order_book():
    """Fixture pour un carnet d'ordres d'exemple."""
    return {
        'symbol': 'ETHUSDT',
        'bids': [
            [2024.5, 1.0],
            [2024.0, 2.0],
            [2023.5, 1.5]
        ],
        'asks': [
            [2025.5, 1.0],
            [2026.0, 2.0],
            [2026.5, 1.5]
        ],
        'timestamp': datetime.now()
    }


@pytest.fixture
def mock_time_series_data():
    """Fixture pour des données de séries temporelles."""
    dates = pd.date_range(start='2023-01-01', periods=252, freq='D')  # 1 an de données quotidiennes
    np.random.seed(42)
    
    # Simulation d'un prix avec tendance et volatilité
    returns = np.random.normal(0.0005, 0.02, 252)  # Retours journaliers
    prices = 100 * np.exp(np.cumsum(returns))  # Prix cumulés
    
    return pd.DataFrame({
        'date': dates,
        'price': prices,
        'returns': returns,
        'volume': np.random.uniform(1000000, 5000000, 252),
        'volatility': pd.Series(returns).rolling(20).std()
    })


# Fixtures pour les tests de performance
@pytest.fixture
def performance_test_data():
    """Fixture pour les données de tests de performance."""
    return {
        'small_dataset': list(range(100)),
        'medium_dataset': list(range(1000)),
        'large_dataset': list(range(10000)),
        'very_large_dataset': list(range(100000))
    }


# Fixtures pour les tests d'intégration
@pytest.fixture(scope="session")
def test_database():
    """Fixture pour une base de données de test."""
    # Créer une base de données SQLite en mémoire pour les tests
    import sqlite3
    
    conn = sqlite3.connect(':memory:')
    conn.row_factory = sqlite3.Row
    
    # Créer les tables de base
    conn.execute('''
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL,
            quantity REAL NOT NULL,
            price REAL NOT NULL,
            timestamp DATETIME NOT NULL,
            status TEXT NOT NULL
        )
    ''')
    
    conn.execute('''
        CREATE TABLE IF NOT EXISTS positions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL,
            quantity REAL NOT NULL,
            entry_price REAL NOT NULL,
            current_price REAL,
            unrealized_pnl REAL DEFAULT 0,
            realized_pnl REAL DEFAULT 0,
            timestamp DATETIME NOT NULL
        )
    ''')
    
    conn.commit()
    
    yield conn
    
    conn.close()


# Fixtures pour les tests de configuration
@pytest.fixture
def mock_config_loader():
    """Fixture pour un chargeur de configuration mocké."""
    loader = Mock()
    loader.load_config = Mock(return_value={
        'exchanges': {
            'binance': {
                'api_key': 'test_key',
                'api_secret': 'test_secret'
            }
        },
        'trading': {
            'max_position_size': 0.1,
            'risk_level': 'medium'
        }
    })
    loader.save_config = Mock(return_value=True)
    return loader


# Fixtures pour les tests de monitoring
@pytest.fixture
def mock_metrics_collector():
    """Fixture pour un collecteur de métriques mocké."""
    collector = Mock()
    collector.increment_counter = Mock()
    collector.set_gauge = Mock()
    collector.record_histogram = Mock()
    collector.record_timer = Mock()
    return collector


# Fixtures pour les tests de sécurité
@pytest.fixture
def mock_security_manager():
    """Fixture pour un gestionnaire de sécurité mocké."""
    manager = Mock()
    manager.authenticate = AsyncMock(return_value={'success': True, 'user_id': 'test_user'})
    manager.authorize = AsyncMock(return_value={'allowed': True})
    manager.encrypt = Mock(return_value='encrypted_data')
    manager.decrypt = Mock(return_value='decrypted_data')
    return manager


# Fixtures pour les tests de cache
@pytest.fixture
def mock_cache_client():
    """Fixture pour un client de cache mocké."""
    cache = Mock()
    cache.get = AsyncMock(return_value=None)
    cache.set = AsyncMock(return_value=True)
    cache.delete = AsyncMock(return_value=True)
    cache.exists = AsyncMock(return_value=False)
    cache.expire = AsyncMock(return_value=True)
    cache.flush_all = AsyncMock(return_value=True)
    return cache


# Helper functions pour les tests
@pytest.fixture
def create_test_order():
    """Helper pour créer des ordres de test."""
    def _create_order(symbol='ETHUSDT', side='buy', quantity=0.1, price=2000.0, order_type='market'):
        return {
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'price': price,
            'order_type': order_type,
            'timestamp': datetime.now(),
            'status': 'open'
        }
    return _create_order


@pytest.fixture
def create_test_position():
    """Helper pour créer des positions de test."""
    def _create_position(symbol='ETHUSDT', side='long', quantity=0.1, entry_price=2000.0):
        return {
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'entry_price': entry_price,
            'current_price': entry_price,
            'unrealized_pnl': 0.0,
            'realized_pnl': 0.0,
            'timestamp': datetime.now()
        }
    return _create_position


@pytest.fixture
def create_test_trade():
    """Helper pour créer des trades de test."""
    def _create_trade(symbol='ETHUSDT', side='buy', quantity=0.1, price=2000.0, fee=0.2):
        return {
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'price': price,
            'fee': fee,
            'timestamp': datetime.now(),
            'trade_id': f'trade_{datetime.now().timestamp()}'
        }
    return _create_trade


# Fixtures pour les tests d'échanges
@pytest.fixture
def mock_exchange_status():
    """Fixture pour les statuts d'exchange mockés."""
    from unittest.mock import Mock
    from exchanges.enums import ExchangeStatus
    
    status = Mock()
    status.ACTIVE = ExchangeStatus.ACTIVE
    status.DISABLED = ExchangeStatus.DISABLED
    status.INACTIVE = ExchangeStatus.INACTIVE
    status.MAINTENANCE = ExchangeStatus.MAINTENANCE
    status.ERROR = ExchangeStatus.ERROR
    return status


@pytest.fixture
def mock_order_status():
    """Fixture pour les statuts d'ordre mockés."""
    from unittest.mock import Mock
    from exchanges.enums import OrderStatus
    
    status = Mock()
    status.PENDING = OrderStatus.PENDING
    status.SUBMITTED = OrderStatus.SUBMITTED
    status.PARTIALLY_FILLED = OrderStatus.PARTIALLY_FILLED
    status.FILLED = OrderStatus.FILLED
    status.CANCELLED = OrderStatus.CANCELLED
    status.REJECTED = OrderStatus.REJECTED
    status.EXPIRED = OrderStatus.EXPIRED
    return status


@pytest.fixture
def mock_signal_status():
    """Fixture pour les statuts de signal mockés."""
    from unittest.mock import Mock
    from exchanges.enums import SignalStatus
    
    status = Mock()
    status.RECEIVED = SignalStatus.RECEIVED
    status.PROCESSED = SignalStatus.PROCESSED
    status.FAILED = SignalStatus.FAILED
    status.CANCELLED = SignalStatus.CANCELLED
    status.TIMEOUT = SignalStatus.TIMEOUT
    return status


@pytest.fixture
def mock_receiver_state():
    """Fixture pour les états du récepteur mockés."""
    from unittest.mock import Mock
    from exchanges.enums import ReceiverState
    
    state = Mock()
    state.STOPPED = ReceiverState.STOPPED
    state.STARTING = ReceiverState.STARTING
    state.ACTIVE = ReceiverState.ACTIVE
    state.STOPPING = ReceiverState.STOPPING
    state.ERROR = ReceiverState.ERROR
    return state


@pytest.fixture
def mock_dispatch_result():
    """Fixture pour les résultats de dispatch mockés."""
    from unittest.mock import Mock
    from exchanges.enums import DispatchResult
    
    result = Mock()
    result.SUCCESS = DispatchResult.SUCCESS
    result.FAILED = DispatchResult.FAILED
    result.RETRY = DispatchResult.RETRY
    result.TIMEOUT = DispatchResult.TIMEOUT
    return result


@pytest.fixture
def mock_trading_signal_class():
    """Fixture pour la classe TradingSignal mockée."""
    from unittest.mock import Mock
    from exchanges.enums import TradingSignal
    
    signal = Mock()
    signal.return_value = {
        'symbol': 'ETHUSDT',
        'side': 'buy',
        'order_type': 'market',
        'quantity': 0.1,
        'price': 2000.0,
        'exchange': None,
        'timestamp': datetime.now(),
        'confidence': 0.85,
        'strategy': 'test_strategy',
        'signal_id': 'test_signal_123',
        'metadata': {'source': 'test'}
    }
    return signal


@pytest.fixture
def mock_routed_order_class():
    """Fixture pour la classe RoutedOrder mockée."""
    from unittest.mock import Mock
    from exchanges.enums import RoutedOrder
    
    order = Mock()
    order.return_value = {
        'id': 'test_order_123',
        'exchange': 'binance',
        'symbol': 'ETHUSDT',
        'side': 'buy',
        'order_type': 'market',
        'quantity': 0.1,
        'price': 2000.0,
        'status': 'submitted',
        'exchange_order_id': 'binance_order_123',
        'timestamp': datetime.now(),
        'filled_quantity': 0.0,
        'average_price': 0.0,
        'fees': 0.0
    }
    return order


@pytest.fixture
def mock_exchange_registry():
    """Fixture pour un registre d'exchange mocké."""
    from unittest.mock import AsyncMock
    
    registry = AsyncMock()
    registry.get_exchange = AsyncMock(return_value=AsyncMock())
    registry.get_registered_exchanges = AsyncMock(return_value=['binance', 'okx'])
    return registry


@pytest.fixture
def mock_exchange_dispatcher():
    """Fixture pour un ExchangeDispatcher mocké avec attributs corrects."""
    from unittest.mock import AsyncMock, Mock
    from exchanges.enums import ExchangeStatus
    
    dispatcher = AsyncMock()
    # Configurer les attributs essentiels
    dispatcher._running = False
    dispatcher._monitoring_task = None
    dispatcher.exchange_status = {}
    dispatcher.dispatch_history = []
    
    # Configurer les méthodes asynchrones
    dispatcher.start = AsyncMock()
    dispatcher.stop = AsyncMock(return_value=False)
    dispatcher.dispatch_to_exchange = AsyncMock(return_value={
        'success': True,
        'order_id': 'test_order_123',
        'exchange': 'binance',
        'status': 'submitted',
        'timestamp': datetime.now()
    })
    dispatcher.dispatch_to_best_exchange = AsyncMock(return_value={
        'success': True,
        'order_id': 'test_order_123',
        'exchange': 'binance',
        'status': 'submitted'
    })
    dispatcher.dispatch_to_multiple_exchanges = AsyncMock(return_value={
        'success': True,
        'orders': [
            {
                'exchange': 'binance',
                'order_id': 'test_order_123',
                'quantity': 0.06,
                'allocation': 0.6
            },
            {
                'exchange': 'okx',
                'order_id': 'test_order_124',
                'quantity': 0.04,
                'allocation': 0.4
            }
        ]
    })
    dispatcher.get_best_exchange = AsyncMock(return_value='binance')
    dispatcher.get_exchange_status = AsyncMock(return_value={
        'success': True,
        'exchange': 'binance',
        'status': ExchangeStatus.ACTIVE,
        'last_check': datetime.now(),
        'latency': 50,
        'error_rate': 0.01
    })
    dispatcher.get_all_exchanges_status = AsyncMock(return_value={
        'success': True,
        'exchanges': {
            'binance': {
                'success': True,
                'exchange': 'binance',
                'status': ExchangeStatus.ACTIVE,
                'last_check': datetime.now(),
                'latency': 50,
                'error_rate': 0.01
            },
            'okx': {
                'success': True,
                'exchange': 'okx',
                'status': ExchangeStatus.ACTIVE,
                'last_check': datetime.now(),
                'latency': 60,
                'error_rate': 0.02
            }
        }
    })
    dispatcher.update_exchange_status = AsyncMock(return_value={
        'success': True,
        'exchange': 'binance',
        'status': ExchangeStatus.ACTIVE,
        'latency': 50,
        'error_rate': 0.01
    })
    dispatcher.disable_exchange = AsyncMock(return_value={
        'success': True,
        'exchange': 'binance',
        'status': ExchangeStatus.DISABLED,
        'reason': 'Maintenance'
    })
    dispatcher.enable_exchange = AsyncMock(return_value={
        'success': True,
        'exchange': 'binance',
        'status': ExchangeStatus.ACTIVE
    })
    dispatcher.get_dispatch_history = AsyncMock(return_value={
        'success': True,
        'history': [],
        'count': 0
    })
    dispatcher.get_statistics = AsyncMock(return_value={
        'success': True,
        'statistics': {
            'total_dispatches': 0,
            'successful_dispatches': 0,
            'failed_dispatches': 0,
            'by_exchange': {},
            'by_symbol': {},
            'by_side': {}
        }
    })
    dispatcher.check_exchange_health = AsyncMock(return_value={
        'success': True,
        'exchange': 'binance',
        'healthy': True,
        'latency': 50,
        'timestamp': datetime.now()
    })
    
    return dispatcher


@pytest.fixture
def mock_order_router():
    """Fixture pour un OrderRouter mocké avec attributs corrects."""
    from unittest.mock import AsyncMock, Mock
    from exchanges.enums import OrderStatus
    
    router = AsyncMock()
    # Configurer les attributs essentiels
    router._running = False
    router._monitoring_task = None
    router.routed_orders = {}
    router.active_orders = {}
    router.order_history = []
    
    # Configurer les méthodes asynchrones
    router.start = AsyncMock()
    router.stop = AsyncMock(return_value=False)
    router.route_order = AsyncMock(return_value={
        'success': True,
        'order_id': 'test_order_123',
        'exchange': 'binance',
        'status': OrderStatus.SUBMITTED,
        'exchange_order_id': 'binance_order_123',
        'timestamp': datetime.now()
    })
    router.cancel_order = AsyncMock(return_value={
        'success': True,
        'order_id': 'test_order_123',
        'status': OrderStatus.CANCELLED,
        'exchange': 'binance',
        'timestamp': datetime.now()
    })
    router.get_order_status = AsyncMock(return_value={
        'success': True,
        'order_id': 'test_order_123',
        'status': OrderStatus.FILLED,
        'exchange': 'binance',
        'symbol': 'ETHUSDT',
        'side': 'buy',
        'quantity': 0.1,
        'filled_quantity': 0.1,
        'average_price': 2000.0,
        'fees': 0.2,
        'timestamp': datetime.now()
    })
    router.get_active_orders = AsyncMock(return_value={
        'success': True,
        'orders': [],
        'count': 0
    })
    router.get_order_history = AsyncMock(return_value={
        'success': True,
        'orders': [],
        'count': 0
    })
    router.get_statistics = AsyncMock(return_value={
        'success': True,
        'statistics': {
            'total_routed': 0,
            'active_orders': 0,
            'successful_orders': 0,
            'failed_orders': 0,
            'success_rate': 0.0,
            'by_exchange': {},
            'by_symbol': {},
            'by_status': {}
        }
    })
    
    return router


@pytest.fixture
def mock_trading_receiver():
    """Fixture pour un TradingReceiver mocké avec attributs corrects."""
    from unittest.mock import AsyncMock, Mock
    from exchanges.enums import SignalStatus, ReceiverState
    
    receiver = AsyncMock()
    # Configurer les attributs essentiels
    receiver._running = False
    receiver._processing_task = None
    receiver.state = ReceiverState.STOPPED
    receiver.received_signals = []
    receiver.processed_signals = []
    receiver.active_signals = {}
    
    # Configurer les méthodes asynchrones
    receiver.start = AsyncMock()
    receiver.stop = AsyncMock(return_value=False)
    receiver.receive_trading_signal = AsyncMock(return_value={
        'success': True,
        'signal_id': 'test_signal_123',
        'status': SignalStatus.RECEIVED,
        'timestamp': datetime.now()
    })
    receiver.process_trading_signal = AsyncMock(return_value={
        'success': True,
        'signal_id': 'test_signal_123',
        'order_id': 'test_order_123',
        'exchange': 'binance',
        'status': SignalStatus.PROCESSED,
        'quantity': 0.1
    })
    receiver.receive_and_process_signal = AsyncMock(return_value={
        'success': True,
        'signal_id': 'test_signal_123',
        'order_id': 'test_order_123',
        'exchange': 'binance',
        'status': SignalStatus.PROCESSED
    })
    receiver.cancel_signal = AsyncMock(return_value={
        'success': True,
        'signal_id': 'test_signal_123',
        'order_id': 'test_order_123',
        'status': SignalStatus.CANCELLED
    })
    receiver.get_signal_status = AsyncMock(return_value={
        'success': True,
        'signal_id': 'test_signal_123',
        'status': SignalStatus.PROCESSED,
        'timestamp': datetime.now(),
        'order_id': 'test_order_123',
        'exchange': 'binance'
    })
    receiver.get_active_signals = AsyncMock(return_value={
        'success': True,
        'signals': [],
        'count': 0
    })
    receiver.get_signal_history = AsyncMock(return_value={
        'success': True,
        'signals': [],
        'count': 0
    })
    receiver.get_statistics = AsyncMock(return_value={
        'success': True,
        'statistics': {
            'total_signals': 0,
            'processed_signals': 0,
            'active_signals': 0,
            'failed_signals': 0,
            'cancelled_signals': 0,
            'success_rate': 0.0,
            'by_symbol': {},
            'by_status': {},
            'by_exchange': {}
        }
    })
    receiver._validate_signal = AsyncMock(return_value={'valid': True})
    receiver._check_risk_limits = AsyncMock(return_value={'allowed': True, 'reason': 'Risk limits passed'})
    receiver._calculate_position_size = AsyncMock(return_value={'quantity': 0.1})
    
    return receiver
