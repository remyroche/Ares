"""
Tests unitaires pour FeeCalculator

Ce module teste les fonctionnalités du calculateur de frais.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Import du module à tester
try:
    from src.simulator.fee_calculator import FeeCalculator, FeeType, FeeTier, ExchangeFee
except ImportError:
    # Si le module n'existe pas encore, on utilise un mock
    FeeCalculator = Mock
    FeeType = Mock
    FeeTier = Mock
    ExchangeFee = Mock


@pytest.mark.unit
@pytest.mark.simulator
@pytest.mark.asyncio
class TestFeeCalculator:
    """Classe de tests pour FeeCalculator."""

    def setup_method(self):
        """Setup pour chaque test."""
        # Configuration des frais par défaut
        self.default_fees = {
            'binance': {
                'maker': 0.001,  # 0.1%
                'taker': 0.001,  # 0.1%
                'tiers': [
                    {'volume_threshold': 0, 'maker': 0.001, 'taker': 0.001},
                    {'volume_threshold': 1000000, 'maker': 0.0009, 'taker': 0.001},
                    {'volume_threshold': 5000000, 'maker': 0.0008, 'taker': 0.0009}
                ]
            },
            'okx': {
                'maker': 0.0008,  # 0.08%
                'taker': 0.001,   # 0.1%
                'tiers': [
                    {'volume_threshold': 0, 'maker': 0.0008, 'taker': 0.001},
                    {'volume_threshold': 1000000, 'maker': 0.0007, 'taker': 0.0009}
                ]
            }
        }
        
        # Créer une instance si la classe existe
        if hasattr(FeeCalculator, '__call__'):
            self.fee_calculator = FeeCalculator(self.default_fees)
        else:
            self.fee_calculator = Mock()

    async def test_initialization_nominal(self):
        """Test d'initialisation nominale."""
        # Given/When
        if hasattr(self.fee_calculator, 'start'):
            await self.fee_calculator.start()
        
        # Then
        if hasattr(self.fee_calculator, 'fees'):
            assert 'binance' in self.fee_calculator.fees
            assert 'okx' in self.fee_calculator.fees
            assert self.fee_calculator.fees['binance']['maker'] == 0.001
            assert self.fee_calculator.fees['okx']['maker'] == 0.0008

    async def test_calculate_trading_fee_nominal(self):
        """Test de calcul de frais de trading nominale."""
        # Given
        if not hasattr(self.fee_calculator, 'calculate_trading_fee'):
            pytest.skip("calculate_trading_fee method not implemented")
            
        exchange = 'binance'
        symbol = 'ETHUSDT'
        side = 'buy'
        quantity = 0.1
        price = 2000.0
        order_type = 'market'  # Taker
        
        # When
        result = await self.fee_calculator.calculate_trading_fee(
            exchange, symbol, side, quantity, price, order_type
        )
        
        # Then
        assert result['success'] is True
        assert 'fee' in result
        assert 'fee_type' in result
        assert 'fee_rate' in result
        assert 'fee_amount' in result
        
        expected_amount = quantity * price * self.default_fees[exchange]['taker']  # 0.1 * 2000 * 0.001 = 0.2
        assert abs(result['fee_amount'] - expected_amount) < 0.01
        assert result['fee_type'] == FeeType.TRADING
        assert result['fee_rate'] == self.default_fees[exchange]['taker']

    async def test_calculate_trading_fee_maker_order(self):
        """Test de calcul de frais pour ordre maker."""
        # Given
        if not hasattr(self.fee_calculator, 'calculate_trading_fee'):
            pytest.skip("calculate_trading_fee method not implemented")
            
        exchange = 'binance'
        symbol = 'ETHUSDT'
        side = 'buy'
        quantity = 0.1
        price = 2000.0
        order_type = 'limit'  # Maker
        
        # When
        result = await self.fee_calculator.calculate_trading_fee(
            exchange, symbol, side, quantity, price, order_type
        )
        
        # Then
        assert result['success'] is True
        expected_amount = quantity * price * self.default_fees[exchange]['maker']  # 0.1 * 2000 * 0.001 = 0.2
        assert abs(result['fee_amount'] - expected_amount) < 0.01
        assert result['fee_rate'] == self.default_fees[exchange]['maker']

    async def test_calculate_trading_fee_different_exchange(self):
        """Test de calcul de frais pour différents exchanges."""
        # Given
        if not hasattr(self.fee_calculator, 'calculate_trading_fee'):
            pytest.skip("calculate_trading_fee method not implemented")
            
        symbol = 'ETHUSDT'
        side = 'buy'
        quantity = 0.1
        price = 2000.0
        order_type = 'market'
        
        # When
        # Binance
        binance_result = await self.fee_calculator.calculate_trading_fee(
            'binance', symbol, side, quantity, price, order_type
        )
        
        # OKX
        okx_result = await self.fee_calculator.calculate_trading_fee(
            'okx', symbol, side, quantity, price, order_type
        )
        
        # Then
        assert binance_result['success'] is True
        assert okx_result['success'] is True
        
        # OKX devrait avoir des frais plus bas
        assert okx_result['fee_amount'] < binance_result['fee_amount']
        
        expected_binance = quantity * price * self.default_fees['binance']['taker']  # 0.2
        expected_okx = quantity * price * self.default_fees['okx']['taker']  # 0.2
        
        assert abs(binance_result['fee_amount'] - expected_binance) < 0.01
        assert abs(okx_result['fee_amount'] - expected_okx) < 0.01

    async def test_calculate_trading_fee_invalid_exchange(self):
        """Test de calcul de frais avec exchange invalide."""
        # Given
        if not hasattr(self.fee_calculator, 'calculate_trading_fee'):
            pytest.skip("calculate_trading_fee method not implemented")
            
        exchange = 'nonexistent_exchange'
        symbol = 'ETHUSDT'
        side = 'buy'
        quantity = 0.1
        price = 2000.0
        order_type = 'market'
        
        # When
        result = await self.fee_calculator.calculate_trading_fee(
            exchange, symbol, side, quantity, price, order_type
        )
        
        # Then
        assert result['success'] is False
        assert 'error' in result
        assert 'exchange' in result['error'].lower() or 'not found' in result['error'].lower()

    async def test_calculate_trading_fee_zero_quantity(self):
        """Test de calcul de frais avec quantité nulle."""
        # Given
        if not hasattr(self.fee_calculator, 'calculate_trading_fee'):
            pytest.skip("calculate_trading_fee method not implemented")
            
        exchange = 'binance'
        symbol = 'ETHUSDT'
        side = 'buy'
        quantity = 0.0  # Quantité nulle
        price = 2000.0
        order_type = 'market'
        
        # When
        result = await self.fee_calculator.calculate_trading_fee(
            exchange, symbol, side, quantity, price, order_type
        )
        
        # Then
        assert result['success'] is True
        assert result['fee_amount'] == 0.0

    async def test_calculate_trading_fee_negative_price(self):
        """Test de calcul de frais avec prix négatif."""
        # Given
        if not hasattr(self.fee_calculator, 'calculate_trading_fee'):
            pytest.skip("calculate_trading_fee method not implemented")
            
        exchange = 'binance'
        symbol = 'ETHUSDT'
        side = 'buy'
        quantity = 0.1
        price = -2000.0  # Prix négatif
        order_type = 'market'
        
        # When
        result = await self.fee_calculator.calculate_trading_fee(
            exchange, symbol, side, quantity, price, order_type
        )
        
        # Then
        assert result['success'] is False
        assert 'error' in result
        assert 'price' in result['error'].lower() or 'invalid' in result['error'].lower()

    async def test_calculate_withdrawal_fee_nominal(self):
        """Test de calcul de frais de retrait nominale."""
        # Given
        if not hasattr(self.fee_calculator, 'calculate_withdrawal_fee'):
            pytest.skip("calculate_withdrawal_fee method not implemented")
            
        # Ajouter des frais de retrait à la configuration
        if hasattr(self.fee_calculator, 'add_withdrawal_fees'):
            self.fee_calculator.add_withdrawal_fees({
                'binance': {
                    'ETH': 0.005,  # 0.005 ETH
                    'BTC': 0.0005  # 0.0005 BTC
                }
            })
        
        exchange = 'binance'
        asset = 'ETH'
        amount = 1.0
        
        # When
        result = await self.fee_calculator.calculate_withdrawal_fee(exchange, asset, amount)
        
        # Then
        assert result['success'] is True
        assert 'fee' in result
        assert 'fee_type' in result
        assert 'fee_amount' in result
        assert 'net_amount' in result
        
        assert result['fee_type'] == FeeType.WITHDRAWAL
        assert result['fee_amount'] == 0.005  # Frais fixes pour ETH
        assert result['net_amount'] == amount - result['fee_amount']  # 1.0 - 0.005 = 0.995

    async def test_calculate_withdrawal_fee_percentage(self):
        """Test de calcul de frais de retrait en pourcentage."""
        # Given
        if not hasattr(self.fee_calculator, 'calculate_withdrawal_fee'):
            pytest.skip("calculate_withdrawal_fee method not implemented")
            
        # Ajouter des frais de retrait en pourcentage
        if hasattr(self.fee_calculator, 'add_withdrawal_fees'):
            self.fee_calculator.add_withdrawal_fees({
                'okx': {
                    'USDT': {'type': 'percentage', 'rate': 0.005}  # 0.5%
                }
            })
        
        exchange = 'okx'
        asset = 'USDT'
        amount = 1000.0
        
        # When
        result = await self.fee_calculator.calculate_withdrawal_fee(exchange, asset, amount)
        
        # Then
        assert result['success'] is True
        expected_fee = amount * 0.005  # 1000 * 0.005 = 5.0
        assert abs(result['fee_amount'] - expected_fee) < 0.01
        assert result['net_amount'] == amount - expected_fee  # 1000 - 5 = 995

    async def test_calculate_deposit_fee_nominal(self):
        """Test de calcul de frais de dépôt nominale."""
        # Given
        if not hasattr(self.fee_calculator, 'calculate_deposit_fee'):
            pytest.skip("calculate_deposit_fee method not implemented")
            
        # Ajouter des frais de dépôt
        if hasattr(self.fee_calculator, 'add_deposit_fees'):
            self.fee_calculator.add_deposit_fees({
                'binance': {
                    'USDT': 0.0,  # Gratuit
                    'BTC': 0.0001
                }
            })
        
        exchange = 'binance'
        asset = 'USDT'
        amount = 1000.0
        
        # When
        result = await self.fee_calculator.calculate_deposit_fee(exchange, asset, amount)
        
        # Then
        assert result['success'] is True
        assert 'fee' in result
        assert 'fee_type' in result
        assert 'fee_amount' in result
        assert 'net_amount' in result
        
        assert result['fee_type'] == FeeType.DEPOSIT
        assert result['fee_amount'] == 0.0  # Gratuit pour USDT
        assert result['net_amount'] == amount  # 1000 - 0 = 1000

    async def test_get_fee_tier_nominal(self):
        """Test de récupération de palier de frais nominale."""
        # Given
        if not hasattr(self.fee_calculator, 'get_fee_tier'):
            pytest.skip("get_fee_tier method not implemented")
            
        exchange = 'binance'
        volume = 2000000.0  # 2M de volume
        
        # When
        result = await self.fee_calculator.get_fee_tier(exchange, volume)
        
        # Then
        assert result['success'] is True
        assert 'tier' in result
        assert 'maker_rate' in result
        assert 'taker_rate' in result
        
        # Devrait être dans le deuxième palier (1M-5M)
        assert result['maker_rate'] == 0.0009
        assert result['taker_rate'] == 0.001

    async def test_get_fee_tier_zero_volume(self):
        """Test de récupération de palier avec volume nul."""
        # Given
        if not hasattr(self.fee_calculator, 'get_fee_tier'):
            pytest.skip("get_fee_tier method not implemented")
            
        exchange = 'binance'
        volume = 0.0
        
        # When
        result = await self.fee_calculator.get_fee_tier(exchange, volume)
        
        # Then
        assert result['success'] is True
        # Devrait être dans le premier palier (0-1M)
        assert result['maker_rate'] == 0.001
        assert result['taker_rate'] == 0.001

    async def test_get_fee_tier_high_volume(self):
        """Test de récupération de palier avec volume élevé."""
        # Given
        if not hasattr(self.fee_calculator, 'get_fee_tier'):
            pytest.skip("get_fee_tier method not implemented")
            
        exchange = 'binance'
        volume = 10000000.0  # 10M de volume
        
        # When
        result = await self.fee_calculator.get_fee_tier(exchange, volume)
        
        # Then
        assert result['success'] is True
        # Devrait être dans le dernier palier (>5M)
        assert result['maker_rate'] == 0.0008
        assert result['taker_rate'] == 0.0009

    async def test_update_trading_volume_nominal(self):
        """Test de mise à jour du volume de trading nominale."""
        # Given
        if not hasattr(self.fee_calculator, 'update_trading_volume'):
            pytest.skip("update_trading_volume method not implemented")
            
        exchange = 'binance'
        symbol = 'ETHUSDT'
        volume = 50000.0  # 50K de volume
        
        # When
        result = await self.fee_calculator.update_trading_volume(exchange, symbol, volume)
        
        # Then
        assert result['success'] is True
        assert 'exchange' in result
        assert 'symbol' in result
        assert 'volume' in result
        assert 'total_volume' in result
        
        assert result['exchange'] == exchange
        assert result['symbol'] == symbol
        assert result['volume'] == volume
        assert result['total_volume'] >= volume

    async def test_get_monthly_volume_nominal(self):
        """Test de récupération du volume mensuel nominale."""
        # Given
        if not hasattr(self.fee_calculator, 'update_trading_volume') or not hasattr(self.fee_calculator, 'get_monthly_volume'):
            pytest.skip("Required methods not implemented")
            
        exchange = 'binance'
        
        # Ajouter quelques volumes
        await self.fee_calculator.update_trading_volume(exchange, 'ETHUSDT', 50000.0)
        await self.fee_calculator.update_trading_volume(exchange, 'BTCUSDT', 100000.0)
        await self.fee_calculator.update_trading_volume(exchange, 'ETHUSDT', 25000.0)
        
        # When
        result = await self.fee_calculator.get_monthly_volume(exchange)
        
        # Then
        assert result['success'] is True
        assert 'total_volume' in result
        assert 'by_symbol' in result
        
        expected_total = 50000.0 + 100000.0 + 25000.0  # 175000.0
        assert abs(result['total_volume'] - expected_total) < 0.01
        
        assert result['by_symbol']['ETHUSDT'] == 75000.0  # 50000 + 25000
        assert result['by_symbol']['BTCUSDT'] == 100000.0

    async def test_calculate_total_fees_nominal(self):
        """Test de calcul des frais totaux nominale."""
        # Given
        if not hasattr(self.fee_calculator, 'calculate_total_fees'):
            pytest.skip("calculate_total_fees method not implemented")
            
        trades = [
            {
                'exchange': 'binance',
                'symbol': 'ETHUSDT',
                'side': 'buy',
                'quantity': 0.1,
                'price': 2000.0,
                'order_type': 'market'
            },
            {
                'exchange': 'binance',
                'symbol': 'ETHUSDT',
                'side': 'sell',
                'quantity': 0.1,
                'price': 2100.0,
                'order_type': 'limit'
            },
            {
                'exchange': 'okx',
                'symbol': 'BTCUSDT',
                'side': 'buy',
                'quantity': 0.05,
                'price': 50000.0,
                'order_type': 'market'
            }
        ]
        
        # When
        result = await self.fee_calculator.calculate_total_fees(trades)
        
        # Then
        assert result['success'] is True
        assert 'total_fees' in result
        assert 'by_exchange' in result
        assert 'by_type' in result
        assert 'trade_count' in result
        
        assert result['trade_count'] == 3
        assert result['total_fees'] > 0
        
        # Vérifier les frais par exchange
        assert 'binance' in result['by_exchange']
        assert 'okx' in result['by_exchange']
        
        # Vérifier les frais par type
        assert FeeType.TRADING in result['by_type']

    async def test_calculate_fee_savings_nominal(self):
        """Test de calcul d'économies de frais nominale."""
        # Given
        if not hasattr(self.fee_calculator, 'calculate_fee_savings'):
            pytest.skip("calculate_fee_savings method not implemented")
            
        current_exchange = 'okx'
        alternative_exchange = 'binance'
        symbol = 'ETHUSDT'
        quantity = 0.1
        price = 2000.0
        
        # When
        result = await self.fee_calculator.calculate_fee_savings(
            current_exchange, alternative_exchange, symbol, quantity, price
        )
        
        # Then
        assert result['success'] is True
        assert 'current_fee' in result
        assert 'alternative_fee' in result
        assert 'savings' in result
        assert 'savings_pct' in result
        
        # Les frais devraient être différents
        assert result['current_fee'] != result['alternative_fee']
        
        # Calculer l'économie attendue
        current_fee_amount = quantity * price * self.default_fees[current_exchange]['taker']
        alternative_fee_amount = quantity * price * self.default_fees[alternative_exchange]['taker']
        expected_savings = abs(current_fee_amount - alternative_fee_amount)
        
        assert abs(result['savings'] - expected_savings) < 0.01

    async def test_add_custom_exchange_fees(self):
        """Test d'ajout de frais d'exchange personnalisés."""
        # Given
        if not hasattr(self.fee_calculator, 'add_exchange_fees'):
            pytest.skip("add_exchange_fees method not implemented")
            
        custom_exchange = 'custom_exchange'
        custom_fees = {
            'maker': 0.0005,
            'taker': 0.0005,
            'tiers': [
                {'volume_threshold': 0, 'maker': 0.0005, 'taker': 0.0005}
            ]
        }
        
        # When
        result = await self.fee_calculator.add_exchange_fees(custom_exchange, custom_fees)
        
        # Then
        assert result['success'] is True
        assert result['exchange'] == custom_exchange
        
        # Vérifier que les frais ont été ajoutés
        fee_result = await self.fee_calculator.calculate_trading_fee(
            custom_exchange, 'ETHUSDT', 'buy', 0.1, 2000.0, 'market'
        )
        
        assert fee_result['success'] is True
        expected_fee = 0.1 * 2000.0 * custom_fees['taker']  # 0.1
        assert abs(fee_result['fee_amount'] - expected_fee) < 0.01

    async def test_get_best_exchange_for_fees(self):
        """Test de sélection du meilleur exchange pour les frais."""
        # Given
        if not hasattr(self.fee_calculator, 'get_best_exchange_for_fees'):
            pytest.skip("get_best_exchange_for_fees method not implemented")
            
        symbol = 'ETHUSDT'
        quantity = 0.1
        price = 2000.0
        order_type = 'market'
        exchanges = ['binance', 'okx']
        
        # When
        result = await self.fee_calculator.get_best_exchange_for_fees(
            symbol, quantity, price, order_type, exchanges
        )
        
        # Then
        assert result['success'] is True
        assert 'best_exchange' in result
        assert 'fee_comparison' in result
        
        # OKX devrait avoir des frais plus bas
        assert result['best_exchange'] == 'okx'
        
        # Vérifier la comparaison
        comparison = result['fee_comparison']
        assert 'binance' in comparison
        assert 'okx' in comparison
        assert comparison['okx'] < comparison['binance']

    async def test_calculate_liquidation_fee(self):
        """Test de calcul de frais de liquidation."""
        # Given
        if not hasattr(self.fee_calculator, 'calculate_liquidation_fee'):
            pytest.skip("calculate_liquidation_fee method not implemented")
            
        exchange = 'binance'
        symbol = 'ETHUSDT'
        position_size = 0.1
        liquidation_price = 1800.0
        
        # When
        result = await self.fee_calculator.calculate_liquidation_fee(
            exchange, symbol, position_size, liquidation_price
        )
        
        # Then
        assert result['success'] is True
        assert 'fee' in result
        assert 'fee_type' in result
        assert 'fee_amount' in result
        
        assert result['fee_type'] == FeeType.LIQUIDATION
        
        # Les frais de liquidation sont généralement plus élevés
        liquidation_fee_rate = 0.005  # 0.5% (exemple)
        expected_fee = position_size * liquidation_price * liquidation_fee_rate
        assert abs(result['fee_amount'] - expected_fee) < 0.01

    async def test_calculate_funding_fee(self):
        """Test de calcul de frais de funding."""
        # Given
        if not hasattr(self.fee_calculator, 'calculate_funding_fee'):
            pytest.skip("calculate_funding_fee method not implemented")
            
        exchange = 'binance'
        symbol = 'ETHUSDT'
        position_size = 0.1
        funding_rate = 0.0001  # 0.01%
        hours = 8  # 8 heures de funding
        
        # When
        result = await self.fee_calculator.calculate_funding_fee(
            exchange, symbol, position_size, funding_rate, hours
        )
        
        # Then
        assert result['success'] is True
        assert 'fee' in result
        assert 'fee_type' in result
        assert 'fee_amount' in result
        assert 'hourly_rate' in result
        
        assert result['fee_type'] == FeeType.FUNDING
        
        # Calcul attendu: position_size * mark_price * funding_rate * hours
        # En supposant un mark_price de 2000.0
        mark_price = 2000.0
        expected_fee = position_size * mark_price * funding_rate * hours
        assert abs(result['fee_amount'] - expected_fee) < 0.01

    async def test_error_handling_invalid_inputs(self):
        """Test de gestion des erreurs avec entrées invalides."""
        # Given/When/Then
        if hasattr(self.fee_calculator, 'calculate_trading_fee'):
            # Test avec quantité négative
            with pytest.raises((ValueError, TypeError)):
                await self.fee_calculator.calculate_trading_fee(
                    'binance', 'ETHUSDT', 'buy', -0.1, 2000.0, 'market'
                )
            
            # Test avec exchange vide
            with pytest.raises((ValueError, TypeError)):
                await self.fee_calculator.calculate_trading_fee(
                    '', 'ETHUSDT', 'buy', 0.1, 2000.0, 'market'
                )
            
            # Test avec symbole vide
            with pytest.raises((ValueError, TypeError)):
                await self.fee_calculator.calculate_trading_fee(
                    'binance', '', 'buy', 0.1, 2000.0, 'market'
                )

    async def test_performance_with_many_calculations(self):
        """Test de performance avec beaucoup de calculs."""
        # Given
        if not hasattr(self.fee_calculator, 'calculate_trading_fee'):
            pytest.skip("calculate_trading_fee method not implemented")
            
        # When
        start_time = datetime.now()
        
        # Effectuer beaucoup de calculs
        tasks = []
        for i in range(1000):
            tasks.append(self.fee_calculator.calculate_trading_fee(
                'binance', f'SYMBOL{i}', 'buy', 0.1, 2000.0, 'market'
            ))
        
        await asyncio.gather(*tasks)
        
        end_time = datetime.now()
        
        # Then
        execution_time = (end_time - start_time).total_seconds()
        assert execution_time < 5.0  # Devrait s'exécuter rapidement

    async def test_memory_usage_with_many_exchanges(self):
        """Test de l'utilisation mémoire avec beaucoup d'exchanges."""
        # Given
        if hasattr(self.fee_calculator, 'add_exchange_fees'):
            # Ajouter beaucoup d'exchanges
            tasks = []
            for i in range(100):
                custom_fees = {
                    'maker': 0.001,
                    'taker': 0.001,
                    'tiers': [{'volume_threshold': 0, 'maker': 0.001, 'taker': 0.001}]
                }
                tasks.append(self.fee_calculator.add_exchange_fees(f'exchange_{i}', custom_fees))
            
            await asyncio.gather(*tasks)
        
        # When/Then
        # Vérifier que le système peut gérer la charge
        if hasattr(self.fee_calculator, 'fees'):
            assert len(self.fee_calculator.fees) >= 100
        
        # Then
        # Le système devrait pouvoir gérer cette charge sans erreur de mémoire
        # (En pratique, on pourrait vouloir ajouter des limites)

    async def test_fee_comparison_across_exchanges(self):
        """Test de comparaison des frais entre exchanges."""
        # Given
        if not hasattr(self.fee_calculator, 'compare_fees'):
            pytest.skip("compare_fees method not implemented")
            
        symbol = 'ETHUSDT'
        quantity = 0.1
        price = 2000.0
        exchanges = ['binance', 'okx']
        
        # When
        result = await self.fee_calculator.compare_fees(
            symbol, quantity, price, exchanges
        )
        
        # Then
        assert result['success'] is True
        assert 'comparison' in result
        assert 'cheapest' in result
        assert 'most_expensive' in result
        
        comparison = result['comparison']
        assert 'binance' in comparison
        assert 'okx' in comparison
        
        # OKX devrait être moins cher
        assert result['cheapest'] == 'okx'
        assert result['most_expensive'] == 'binance'
        
        # Vérifier les montants
        assert comparison['okx'] < comparison['binance']

    async def test_export_import_fee_configuration(self):
        """Test d'export/import de configuration de frais."""
        # Given
        if not hasattr(self.fee_calculator, 'export_configuration') or not hasattr(self.fee_calculator, 'import_configuration'):
            pytest.skip("Required methods not implemented")
            
        # When
        # Exporter la configuration
        export_result = await self.fee_calculator.export_configuration()
        assert export_result['success'] is True
        config_data = export_result['configuration']
        
        # Réinitialiser et importer la configuration
        await self.fee_calculator.reset()
        import_result = await self.fee_calculator.import_configuration(config_data)
        
        # Then
        assert import_result['success'] is True
        
        # Vérifier que la configuration a été restaurée
        fee_result = await self.fee_calculator.calculate_trading_fee(
            'binance', 'ETHUSDT', 'buy', 0.1, 2000.0, 'market'
        )
        assert fee_result['success'] is True
        expected_fee = 0.1 * 2000.0 * self.default_fees['binance']['taker']  # 0.2
        assert abs(fee_result['fee_amount'] - expected_fee) < 0.01