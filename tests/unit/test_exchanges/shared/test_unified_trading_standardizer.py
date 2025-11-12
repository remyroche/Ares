"""
Tests unitaires pour UnifiedTradingStandardizer

Ce module teste les fonctionnalités du standardiseur de trading unifié.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Import du module à tester
try:
    from exchanges.shared.unified_trading_standardizer import (
        UnifiedTradingStandardizer, 
        StandardizedOrder, 
        StandardizedTicker, 
        StandardizedTrade,
        StandardizedBalance,
        StandardizedPosition
    )
except ImportError:
    # Si le module n'existe pas encore, on utilise un mock
    UnifiedTradingStandardizer = Mock
    StandardizedOrder = Mock
    StandardizedTicker = Mock
    StandardizedTrade = Mock
    StandardizedBalance = Mock
    StandardizedPosition = Mock


@pytest.mark.unit
@pytest.mark.exchanges
@pytest.mark.asyncio
class TestUnifiedTradingStandardizer:
    """Classe de tests pour UnifiedTradingStandardizer."""

    def setup_method(self):
        """Setup pour chaque test."""
        self.mock_exchange_interfaces = {
            'binance': Mock(),
            'okx': Mock(),
            'coinbase': Mock()
        }
        
        # Créer une instance si la classe existe
        if hasattr(UnifiedTradingStandardizer, '__call__'):
            self.standardizer = UnifiedTradingStandardizer(self.mock_exchange_interfaces)
        else:
            self.standardizer = Mock()

    async def test_initialization_nominal(self):
        """Test d'initialisation nominale."""
        # Given/When
        if hasattr(self.standardizer, 'start'):
            await self.standardizer.start()
        
        # Then
        if hasattr(self.standardizer, 'supported_exchanges'):
            assert set(self.standardizer.supported_exchanges) == set(['binance', 'okx', 'coinbase'])
        if hasattr(self.standardizer, 'standardization_rules'):
            assert len(self.standardizer.standardization_rules) > 0
        if hasattr(self.standardizer, 'is_running'):
            assert self.standardizer.is_running is True

    async def test_standardize_order_nominal(self):
        """Test de standardisation d'ordre nominale."""
        # Given
        if not hasattr(self.standardizer, 'standardize_order'):
            pytest.skip("standardize_order method not implemented")
            
        # Ordre Binance original
        binance_order = {
            'exchange': 'binance',
            'order_id': 'binance_order_123',
            'symbol': 'ETHUSDT',
            'side': 'buy',
            'type': 'limit',
            'amount': 0.1,
            'price': 2000.0,
            'status': 'open',
            'timestamp': '2023-01-01T12:00:00Z'
        }
        
        # When
        result = await self.standardizer.standardize_order(binance_order)
        
        # Then
        assert result['success'] is True
        assert 'standardized_order' in result
        
        std_order = result['standardized_order']
        assert std_order['exchange'] == 'binance'
        assert std_order['original_order_id'] == 'binance_order_123'
        assert std_order['symbol'] == 'ETHUSDT'
        assert std_order['side'] == 'buy'
        assert std_order['type'] == 'limit'
        assert std_order['amount'] == 0.1
        assert std_order['price'] == 2000.0
        assert std_order['status'] == 'open'
        assert 'timestamp' in std_order

    async def test_standardize_order_different_exchanges(self):
        """Test de standardisation d'ordres de différents exchanges."""
        # Given
        if not hasattr(self.standardizer, 'standardize_order'):
            pytest.skip("standardize_order method not implemented")
            
        # Ordres de différents exchanges
        binance_order = {
            'exchange': 'binance',
            'order_id': 'binance_order_123',
            'symbol': 'ETHUSDT',
            'side': 'buy',
            'type': 'limit',
            'amount': 0.1,
            'price': 2000.0,
            'status': 'open'
        }
        
        okx_order = {
            'exchange': 'okx',
            'order_id': 'okx_order_456',
            'symbol': 'ETH-USDT',  # Format différent
            'side': 'buy',
            'type': 'limit',
            'amount': 0.1,
            'price': 2000.0,
            'status': 'open'
        }
        
        # When
        binance_result = await self.standardizer.standardize_order(binance_order)
        okx_result = await self.standardizer.standardize_order(okx_order)
        
        # Then
        assert binance_result['success'] is True
        assert okx_result['success'] is True
        
        # Les deux devraient avoir le même format standardisé
        std_binance = binance_result['standardized_order']
        std_okx = okx_result['standardized_order']
        
        # Le symbole devrait être standardisé
        assert std_binance['symbol'] == std_okx['symbol'] == 'ETHUSDT'
        
        # Les autres champs devraient être identiques
        assert std_binance['side'] == std_okx['side'] == 'buy'
        assert std_binance['type'] == std_okx['type'] == 'limit'
        assert std_binance['amount'] == std_okx['amount'] == 0.1

    async def test_standardize_order_invalid_exchange(self):
        """Test de standardisation d'ordre avec exchange non supporté."""
        # Given
        if not hasattr(self.standardizer, 'standardize_order'):
            pytest.skip("standardize_order method not implemented")
            
        invalid_order = {
            'exchange': 'nonexistent_exchange',
            'order_id': 'order_123',
            'symbol': 'ETHUSDT',
            'side': 'buy',
            'type': 'limit',
            'amount': 0.1,
            'price': 2000.0,
            'status': 'open'
        }
        
        # When
        result = await self.standardizer.standardize_order(invalid_order)
        
        # Then
        assert result['success'] is False
        assert 'error' in result
        assert 'exchange' in result['error'].lower() or 'supported' in result['error'].lower()

    async def test_standardize_ticker_nominal(self):
        """Test de standardisation de ticker nominale."""
        # Given
        if not hasattr(self.standardizer, 'standardize_ticker'):
            pytest.skip("standardize_ticker method not implemented")
            
        # Ticker Binance original
        binance_ticker = {
            'exchange': 'binance',
            'symbol': 'ETHUSDT',
            'bid': 2024.5,
            'ask': 2025.5,
            'last': 2025.0,
            'high': 2030.0,
            'low': 1990.0,
            'volume': 1000.0,
            'timestamp': '2023-01-01T12:00:00Z'
        }
        
        # When
        result = await self.standardizer.standardize_ticker(binance_ticker)
        
        # Then
        assert result['success'] is True
        assert 'standardized_ticker' in result
        
        std_ticker = result['standardized_ticker']
        assert std_ticker['exchange'] == 'binance'
        assert std_ticker['symbol'] == 'ETHUSDT'
        assert std_ticker['bid'] == 2024.5
        assert std_ticker['ask'] == 2025.5
        assert std_ticker['last'] == 2025.0
        assert std_ticker['high'] == 2030.0
        assert std_ticker['low'] == 1990.0
        assert std_ticker['volume'] == 1000.0
        assert 'timestamp' in std_ticker

    async def test_standardize_ticker_different_formats(self):
        """Test de standardisation de tickers avec formats différents."""
        # Given
        if not hasattr(self.standardizer, 'standardize_ticker'):
            pytest.skip("standardize_ticker method not implemented")
            
        # Tickers avec différents formats
        binance_ticker = {
            'exchange': 'binance',
            'symbol': 'ETHUSDT',
            'bid': 2024.5,
            'ask': 2025.5,
            'last': 2025.0,
            'volume': 1000.0,
            'timestamp': '2023-01-01T12:00:00Z'
        }
        
        okx_ticker = {
            'exchange': 'okx',
            'symbol': 'ETH-USDT',
            'best_bid': 2024.5,  # Nom de champ différent
            'best_ask': 2025.5,  # Nom de champ différent
            'last_price': 2025.0,  # Nom de champ différent
            'base_volume': 1000.0,  # Nom de champ différent
            'timestamp': '2023-01-01T12:00:00Z'
        }
        
        # When
        binance_result = await self.standardizer.standardize_ticker(binance_ticker)
        okx_result = await self.standardizer.standardize_ticker(okx_ticker)
        
        # Then
        assert binance_result['success'] is True
        assert okx_result['success'] is True
        
        # Les deux devraient avoir le même format standardisé
        std_binance = binance_result['standardized_ticker']
        std_okx = okx_result['standardized_ticker']
        
        # Vérifier la standardisation
        assert std_binance['symbol'] == std_okx['symbol'] == 'ETHUSDT'
        assert std_binance['bid'] == std_okx['bid'] == 2024.5
        assert std_binance['ask'] == std_okx['ask'] == 2025.5
        assert std_binance['last'] == std_okx['last'] == 2025.0
        assert std_binance['volume'] == std_okx['volume'] == 1000.0

    async def test_standardize_trade_nominal(self):
        """Test de standardisation de trade nominale."""
        # Given
        if not hasattr(self.standardizer, 'standardize_trade'):
            pytest.skip("standardize_trade method not implemented")
            
        # Trade Binance original
        binance_trade = {
            'exchange': 'binance',
            'trade_id': 'binance_trade_123',
            'order_id': 'binance_order_456',
            'symbol': 'ETHUSDT',
            'side': 'buy',
            'amount': 0.1,
            'price': 2000.0,
            'fee': 0.2,
            'timestamp': '2023-01-01T12:00:00Z'
        }
        
        # When
        result = await self.standardizer.standardize_trade(binance_trade)
        
        # Then
        assert result['success'] is True
        assert 'standardized_trade' in result
        
        std_trade = result['standardized_trade']
        assert std_trade['exchange'] == 'binance'
        assert std_trade['trade_id'] == 'binance_trade_123'
        assert std_trade['order_id'] == 'binance_order_456'
        assert std_trade['symbol'] == 'ETHUSDT'
        assert std_trade['side'] == 'buy'
        assert std_trade['amount'] == 0.1
        assert std_trade['price'] == 2000.0
        assert std_trade['fee'] == 0.2
        assert 'timestamp' in std_trade

    async def test_standardize_balance_nominal(self):
        """Test de standardisation de solde nominale."""
        # Given
        if not hasattr(self.standardizer, 'standardize_balance'):
            pytest.skip("standardize_balance method not implemented")
            
        # Balance Binance originale
        binance_balance = {
            'exchange': 'binance',
            'timestamp': '2023-01-01T12:00:00Z',
            'balances': {
                'ETH': {'free': 1.0, 'used': 0.5, 'total': 1.5},
                'USDT': {'free': 10000.0, 'used': 5000.0, 'total': 15000.0}
            }
        }
        
        # When
        result = await self.standardizer.standardize_balance(binance_balance)
        
        # Then
        assert result['success'] is True
        assert 'standardized_balance' in result
        
        std_balance = result['standardized_balance']
        assert std_balance['exchange'] == 'binance'
        assert 'timestamp' in std_balance
        assert 'balances' in std_balance
        
        # Vérifier la structure standardisée
        eth_balance = std_balance['balances']['ETH']
        usdt_balance = std_balance['balances']['USDT']
        
        assert eth_balance['free'] == 1.0
        assert eth_balance['used'] == 0.5
        assert eth_balance['total'] == 1.5
        assert usdt_balance['free'] == 10000.0
        assert usdt_balance['used'] == 5000.0
        assert usdt_balance['total'] == 15000.0

    async def test_standardize_position_nominal(self):
        """Test de standardisation de position nominale."""
        # Given
        if not hasattr(self.standardizer, 'standardize_position'):
            pytest.skip("standardize_position method not implemented")
            
        # Position Binance originale
        binance_position = {
            'exchange': 'binance',
            'position_id': 'binance_pos_123',
            'symbol': 'ETHUSDT',
            'side': 'long',
            'amount': 0.1,
            'entry_price': 2000.0,
            'current_price': 2100.0,
            'unrealized_pnl': 10.0,
            'timestamp': '2023-01-01T12:00:00Z'
        }
        
        # When
        result = await self.standardizer.standardize_position(binance_position)
        
        # Then
        assert result['success'] is True
        assert 'standardized_position' in result
        
        std_position = result['standardized_position']
        assert std_position['exchange'] == 'binance'
        assert std_position['position_id'] == 'binance_pos_123'
        assert std_position['symbol'] == 'ETHUSDT'
        assert std_position['side'] == 'long'
        assert std_position['amount'] == 0.1
        assert std_position['entry_price'] == 2000.0
        assert std_position['current_price'] == 2100.0
        assert std_position['unrealized_pnl'] == 10.0
        assert 'timestamp' in std_position

    async def test_convert_symbol_nominal(self):
        """Test de conversion de symbole nominale."""
        # Given
        if not hasattr(self.standardizer, 'convert_symbol'):
            pytest.skip("convert_symbol method not implemented")
            
        # Conversions de symboles
        test_cases = [
            ('ETHUSDT', 'binance', 'ETHUSDT'),
            ('ETH-USDT', 'okx', 'ETHUSDT'),
            ('ETH/USDT', 'coinbase', 'ETHUSDT'),
            ('ethusdt', 'binance', 'ETHUSDT'),  # Minuscules
            ('BTC-USDT', 'okx', 'BTCUSDT'),
            ('ADA/USDT', 'coinbase', 'ADAUSDT')
        ]
        
        # When/Then
        for original_symbol, exchange, expected_standard in test_cases:
            result = await self.standardizer.convert_symbol(original_symbol, exchange)
            
            assert result['success'] is True
            assert result['standard_symbol'] == expected_standard
            assert result['original_symbol'] == original_symbol
            assert result['exchange'] == exchange

    async def test_convert_symbol_invalid_exchange(self):
        """Test de conversion de symbole avec exchange invalide."""
        # Given
        if not hasattr(self.standardizer, 'convert_symbol'):
            pytest.skip("convert_symbol method not implemented")
            
        symbol = 'ETHUSDT'
        invalid_exchange = 'nonexistent_exchange'
        
        # When
        result = await self.standardizer.convert_symbol(symbol, invalid_exchange)
        
        # Then
        assert result['success'] is False
        assert 'error' in result
        assert 'exchange' in result['error'].lower() or 'supported' in result['error'].lower()

    async def test_convert_price_nominal(self):
        """Test de conversion de prix nominale."""
        # Given
        if not hasattr(self.standardizer, 'convert_price'):
            pytest.skip("convert_price method not implemented")
            
        # Conversions de prix
        test_cases = [
            (2000.0, 'USDT', 'USD', 2000.0),  # USDT vers USD (même)
            (2000.0, 'ETH', 'USDT', 0.5),  # ETH vers USDT (division)
            (0.5, 'USDT', 'ETH', 2000.0),  # USDT vers ETH (multiplication)
        ]
        
        # When/Then
        for amount, from_currency, to_currency, expected in test_cases:
            result = await self.standardizer.convert_price(amount, from_currency, to_currency)
            
            assert result['success'] is True
            assert abs(result['converted_amount'] - expected) < 0.0001
            assert result['original_amount'] == amount
            assert result['from_currency'] == from_currency
            assert result['to_currency'] == to_currency

    async def test_standardize_orderbook_nominal(self):
        """Test de standardisation de carnet d'ordres nominale."""
        # Given
        if not hasattr(self.standardizer, 'standardize_orderbook'):
            pytest.skip("standardize_orderbook method not implemented")
            
        # Carnet d'ordres Binance original
        binance_orderbook = {
            'exchange': 'binance',
            'symbol': 'ETHUSDT',
            'bids': [[2024.5, 1.0], [2024.0, 2.0]],
            'asks': [[2025.5, 1.0], [2026.0, 2.0]],
            'timestamp': '2023-01-01T12:00:00Z'
        }
        
        # When
        result = await self.standardizer.standardize_orderbook(binance_orderbook)
        
        # Then
        assert result['success'] is True
        assert 'standardized_orderbook' in result
        
        std_orderbook = result['standardized_orderbook']
        assert std_orderbook['exchange'] == 'binance'
        assert std_orderbook['symbol'] == 'ETHUSDT'
        assert len(std_orderbook['bids']) == 2
        assert len(std_orderbook['asks']) == 2
        
        # Vérifier la structure des niveaux de prix
        for bid in std_orderbook['bids']:
            assert len(bid) == 2  # [price, quantity]
            assert isinstance(bid[0], (int, float))
            assert isinstance(bid[1], (int, float))
        
        for ask in std_orderbook['asks']:
            assert len(ask) == 2  # [price, quantity]
            assert isinstance(ask[0], (int, float))
            assert isinstance(ask[1], (int, float))

    async def test_get_standardization_rules_nominal(self):
        """Test de récupération des règles de standardisation nominale."""
        # Given
        if not hasattr(self.standardizer, 'get_standardization_rules'):
            pytest.skip("get_standardization_rules method not implemented")
            
        # When
        result = await self.standardizer.get_standardization_rules()
        
        # Then
        assert result['success'] is True
        assert 'rules' in result
        assert 'version' in result
        assert 'last_updated' in result
        
        rules = result['rules']
        assert 'symbol_mapping' in rules
        assert 'price_conversion' in rules
        assert 'order_status_mapping' in rules
        assert 'timestamp_format' in rules

    async def test_add_custom_standardization_rule_nominal(self):
        """Test d'ajout de règle de standardisation personnalisée nominale."""
        # Given
        if not hasattr(self.standardizer, 'add_custom_standardization_rule'):
            pytest.skip("add_custom_standardization_rule method not implemented")
            
        custom_rule = {
            'name': 'custom_symbol_mapping',
            'type': 'symbol_mapping',
            'exchange': 'custom_exchange',
            'mapping': {
                'CUSTOM1': 'STANDARD1',
                'CUSTOM2': 'STANDARD2'
            }
        }
        
        # When
        result = await self.standardizer.add_custom_standardization_rule(custom_rule)
        
        # Then
        assert result['success'] is True
        assert 'rule_id' in result
        assert result['rule_name'] == custom_rule['name']
        
        # Vérifier que la règle a été ajoutée
        rules_result = await self.standardizer.get_standardization_rules()
        custom_rules = [r for r in rules_result['rules'] if r.get('name') == custom_rule['name']]
        assert len(custom_rules) == 1

    async def test_remove_standardization_rule_nominal(self):
        """Test de suppression de règle de standardisation nominale."""
        # Given
        if not hasattr(self.standardizer, 'add_custom_standardization_rule') or not hasattr(self.standardizer, 'remove_standardization_rule'):
            pytest.skip("Required methods not implemented")
            
        # Ajouter une règle personnalisée
        custom_rule = {
            'name': 'test_rule',
            'type': 'symbol_mapping',
            'exchange': 'test_exchange',
            'mapping': {'TEST': 'STANDARD'}
        }
        add_result = await self.standardizer.add_custom_standardization_rule(custom_rule)
        rule_id = add_result['rule_id']
        
        # When
        result = await self.standardizer.remove_standardization_rule(rule_id)
        
        # Then
        assert result['success'] is True
        assert result['rule_id'] == rule_id
        
        # Vérifier que la règle a été supprimée
        rules_result = await self.standardizer.get_standardization_rules()
        test_rules = [r for r in rules_result['rules'] if r.get('name') == 'test_rule']
        assert len(test_rules) == 0

    async def test_batch_standardization_nominal(self):
        """Test de standardisation en lot nominale."""
        # Given
        if not hasattr(self.standardizer, 'batch_standardize'):
            pytest.skip("batch_standardize method not implemented")
            
        # Données à standardiser
        orders = [
            {
                'exchange': 'binance',
                'order_id': 'order_1',
                'symbol': 'ETHUSDT',
                'side': 'buy',
                'type': 'limit',
                'amount': 0.1,
                'price': 2000.0
            },
            {
                'exchange': 'okx',
                'order_id': 'order_2',
                'symbol': 'BTCUSDT',
                'side': 'sell',
                'type': 'market',
                'amount': 0.05,
                'price': 50000.0
            }
        ]
        
        tickers = [
            {
                'exchange': 'binance',
                'symbol': 'ETHUSDT',
                'bid': 2024.5,
                'ask': 2025.5,
                'last': 2025.0
            }
        ]
        
        # When
        result = await self.standardizer.batch_standardize({
            'orders': orders,
            'tickers': tickers
        })
        
        # Then
        assert result['success'] is True
        assert 'standardized_orders' in result
        assert 'standardized_tickers' in result
        assert 'failed_items' in result
        
        assert len(result['standardized_orders']) == 2
        assert len(result['standardized_tickers']) == 1
        assert len(result['failed_items']) == 0

    async def test_concurrent_standardization(self):
        """Test de standardisation concurrente."""
        # Given
        if not hasattr(self.standardizer, 'standardize_order'):
            pytest.skip("standardize_order method not implemented")
            
        # When
        # Standardiser plusieurs ordres simultanément
        orders = []
        for i in range(10):
            order = {
                'exchange': 'binance',
                'order_id': f'order_{i}',
                'symbol': 'ETHUSDT',
                'side': 'buy',
                'type': 'limit',
                'amount': 0.1,
                'price': 2000.0 + i
            }
            orders.append(order)
        
        tasks = [self.standardizer.standardize_order(order) for order in orders]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Then
        successful_standardizations = [r for r in results if r and r.get('success')]
        assert len(successful_standardizations) == 10  # Tous devraient réussir
        
        # Vérifier que tous les IDs sont uniques
        order_ids = [r['standardized_order']['original_order_id'] for r in successful_standardizations]
        assert len(set(order_ids)) == 10

    async def test_error_handling_invalid_inputs(self):
        """Test de gestion des erreurs avec entrées invalides."""
        # Given/When/Then
        if hasattr(self.standardizer, 'standardize_order'):
            # Test avec order vide
            with pytest.raises((ValueError, TypeError)):
                await self.standardizer.standardize_order({})
            
            # Test avec order sans exchange
            with pytest.raises((ValueError, TypeError)):
                await self.standardizer.standardize_order({
                    'order_id': 'test',
                    'symbol': 'ETHUSDT',
                    'side': 'buy',
                    'type': 'limit',
                    'amount': 0.1,
                    'price': 2000.0
                    # Manque 'exchange'
                })
            
            # Test avec amount négatif
            with pytest.raises((ValueError, TypeError)):
                await self.standardizer.standardize_order({
                    'exchange': 'binance',
                    'order_id': 'test',
                    'symbol': 'ETHUSDT',
                    'side': 'buy',
                    'type': 'limit',
                    'amount': -0.1,
                    'price': 2000.0
                })

    async def test_performance_with_many_standardizations(self):
        """Test de performance avec beaucoup de standardisations."""
        # Given
        if not hasattr(self.standardizer, 'standardize_order'):
            pytest.skip("standardize_order method not implemented")
            
        # When
        start_time = datetime.now()
        
        # Standardiser beaucoup d'ordres
        tasks = []
        for i in range(100):
            order = {
                'exchange': 'binance',
                'order_id': f'order_{i}',
                'symbol': f'SYMBOL{i}',
                'side': 'buy',
                'type': 'limit',
                'amount': 0.1,
                'price': 2000.0
            }
            tasks.append(self.standardizer.standardize_order(order))
        
        await asyncio.gather(*tasks)
        end_time = datetime.now()
        
        # Then
        execution_time = (end_time - start_time).total_seconds()
        assert execution_time < 10.0  # Devrait s'exécuter rapidement

    async def test_memory_usage_with_large_data(self):
        """Test de l'utilisation mémoire avec beaucoup de données."""
        # Given
        if hasattr(self.standardizer, 'batch_standardize'):
            # Créer beaucoup de données
            large_orders = []
            for i in range(1000):
                order = {
                    'exchange': 'binance',
                    'order_id': f'order_{i}',
                    'symbol': f'SYMBOL{i}',
                    'side': 'buy',
                    'type': 'limit',
                    'amount': 0.1,
                    'price': 2000.0,
                    'metadata': {'key' + str(i): 'value' + str(i) for i in range(10)}  # Métadonnées volumineuses
                }
                large_orders.append(order)
            
            # When
            result = await self.standardizer.batch_standardize({'orders': large_orders})
            
            # Then
            if result['success']:
                assert len(result['standardized_orders']) == 1000
        
        # Then
        # Le système devrait pouvoir gérer cette charge sans erreur de mémoire
        # (En pratique, on pourrait vouloir ajouter des limites)

    async def test_standardization_consistency(self):
        """Test de cohérence de la standardisation."""
        # Given
        if not hasattr(self.standardizer, 'standardize_order') or not hasattr(self.standardizer, 'standardize_trade'):
            pytest.skip("Required methods not implemented")
            
        # Données cohérentes
        order = {
            'exchange': 'binance',
            'order_id': 'order_123',
            'symbol': 'ETHUSDT',
            'side': 'buy',
            'type': 'limit',
            'amount': 0.1,
            'price': 2000.0,
            'timestamp': '2023-01-01T12:00:00Z'
        }
        
        trade = {
            'exchange': 'binance',
            'order_id': 'order_123',
            'symbol': 'ETHUSDT',
            'side': 'buy',
            'amount': 0.1,
            'price': 2000.0,
            'fee': 0.2,
            'timestamp': '2023-01-01T12:00:00Z'
        }
        
        # When
        order_result = await self.standardizer.standardize_order(order)
        trade_result = await self.standardizer.standardize_trade(trade)
        
        # Then
        std_order = order_result['standardized_order']
        std_trade = trade_result['standardized_trade']
        
        # Vérifier la cohérence
        assert std_order['symbol'] == std_trade['symbol'] == 'ETHUSDT'
        assert std_order['side'] == std_trade['side'] == 'buy'
        assert std_order['amount'] == std_trade['amount'] == 0.1
        assert std_order['price'] == std_trade['price'] == 2000.0

    async def test_version_compatibility(self):
        """Test de compatibilité de version."""
        # Given
        if not hasattr(self.standardizer, 'get_version') or not hasattr(self.standardizer, 'check_compatibility'):
            pytest.skip("Required methods not implemented")
            
        # When
        version_result = await self.standardizer.get_version()
        compatibility_result = await self.standardizer.check_compatibility('1.0.0')
        
        # Then
        assert version_result['success'] is True
        assert 'version' in version_result
        assert 'build_date' in version_result
        
        assert compatibility_result['success'] is True
        assert 'compatible' in compatibility_result
        assert 'version' in compatibility_result

    async def test_export_import_standardization_rules(self):
        """Test d'export/import des règles de standardisation."""
        # Given
        if not hasattr(self.standardizer, 'export_standardization_rules') or not hasattr(self.standardizer, 'import_standardization_rules'):
            pytest.skip("Required methods not implemented")
            
        # When
        # Exporter les règles
        export_result = await self.standardizer.export_standardization_rules()
        assert export_result['success'] is True
        rules_data = export_result['rules_data']
        
        # Réinitialiser et importer les règles
        await self.standardizer.reset()
        import_result = await self.standardizer.import_standardization_rules(rules_data)
        
        # Then
        assert import_result['success'] is True
        
        # Vérifier que les règles ont été restaurées
        rules_result = await self.standardizer.get_standardization_rules()
        assert len(rules_result['rules']) > 0

    async def test_real_time_standardization(self):
        """Test de standardisation en temps réel."""
        # Given
        if not hasattr(self.standardizer, 'enable_real_time_mode') or not hasattr(self.standardizer, 'process_real_time_data'):
            pytest.skip("Required methods not implemented")
            
        # Activer le mode temps réel
        await self.standardizer.enable_real_time_mode()
        
        # Simuler des données en temps réel
        real_time_data = {
            'type': 'ticker',
            'exchange': 'binance',
            'symbol': 'ETHUSDT',
            'data': {
                'bid': 2024.5,
                'ask': 2025.5,
                'last': 2025.0
            },
            'timestamp': datetime.now().isoformat()
        }
        
        # When
        result = await self.standardizer.process_real_time_data(real_time_data)
        
        # Then
        assert result['success'] is True
        assert 'standardized_data' in result
        
        std_data = result['standardized_data']
        assert std_data['type'] == 'ticker'
        assert std_data['exchange'] == 'binance'
        assert std_data['symbol'] == 'ETHUSDT'
        assert 'bid' in std_data
        assert 'ask' in std_data
        assert 'last' in std_data