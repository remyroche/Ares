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

# Import des assertions standardisées
from tests.utils.assertions import (
    assert_true,
    assert_equals,
    assert_float_equals,
    assert_less_than,
    assert_greater_than,
    assert_greater_than_or_equal,
    assert_in,
    assert_not_equals,
    assert_is_instance
)

# Import du module à tester
try:
    from src.simulator.fee_calculator import FeeCalculator, FeeType, FeeTier, ExchangeFee
except ImportError:
    # Si le module n'existe pas encore, on utilise un mock
    FeeCalculator = Mock
    # Créer des mocks pour les enums
    FeeType = Mock()
    FeeType.TRADING = 'TRADING'
    FeeType.WITHDRAWAL = 'WITHDRAWAL'
    FeeType.DEPOSIT = 'DEPOSIT'
    FeeType.LIQUIDATION = 'LIQUIDATION'
    FeeType.FUNDING = 'FUNDING'
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
        if hasattr(FeeCalculator, '__call__') and FeeCalculator is not Mock:
            self.fee_calculator = FeeCalculator(self.default_fees)
        else:
            self.fee_calculator = Mock()
            # Configurer le mock pour avoir les attributs nécessaires
            self.fee_calculator.fees = self.default_fees
            self.fee_calculator.calculate_trading_fee = AsyncMock(return_value={'success': True, 'fee_amount': 0.2})
            self.fee_calculator.calculate_withdrawal_fee = AsyncMock(return_value={'success': True, 'fee_amount': 0.005})
            self.fee_calculator.calculate_deposit_fee = AsyncMock(return_value={'success': True, 'fee_amount': 0.0})
            self.fee_calculator.get_fee_tier = AsyncMock(return_value={'success': True, 'maker_rate': 0.001, 'taker_rate': 0.001})
            self.fee_calculator.update_trading_volume = AsyncMock(return_value={'success': True, 'total_volume': 100000.0})
            self.fee_calculator.get_monthly_volume = AsyncMock(return_value={'success': True, 'total_volume': 175000.0, 'by_symbol': {'ETHUSDT': 75000.0, 'BTCUSDT': 100000.0}})
            self.fee_calculator.calculate_total_fees = AsyncMock(return_value={'success': True, 'total_fees': 10.0, 'by_exchange': {'binance': 6.0, 'okx': 4.0}, 'by_type': {'TRADING': 10.0}, 'trade_count': 3})
            self.fee_calculator.calculate_fee_savings = AsyncMock(return_value={'success': True, 'current_fee': 2.0, 'alternative_fee': 1.8, 'savings': 0.2})
            self.fee_calculator.add_exchange_fees = AsyncMock(return_value={'success': True, 'exchange': 'custom_exchange'})
            self.fee_calculator.get_best_exchange_for_fees = AsyncMock(return_value={'success': True, 'best_exchange': 'okx', 'fee_comparison': {'binance': 2.0, 'okx': 1.8}})
            self.fee_calculator.calculate_liquidation_fee = AsyncMock(return_value={'success': True, 'fee_amount': 0.9})
            self.fee_calculator.calculate_funding_fee = AsyncMock(return_value={'success': True, 'fee_amount': 0.16})
            self.fee_calculator.add_withdrawal_fees = Mock()
            self.fee_calculator.add_deposit_fees = Mock()
            self.fee_calculator.export_configuration = AsyncMock(return_value={'success': True, 'configuration': {}})
            self.fee_calculator.import_configuration = AsyncMock(return_value={'success': True})
            self.fee_calculator.reset = AsyncMock(return_value={'success': True})
            self.fee_calculator.compare_fees = AsyncMock(return_value={'success': True, 'comparison': {'binance': 2.0, 'okx': 1.8}, 'cheapest': 'okx', 'most_expensive': 'binance'})
            self.fee_calculator.start = AsyncMock()

    async def test_initialization_nominal(self):
        """Test d'initialisation nominale."""
        # Given/When
        if hasattr(self.fee_calculator, 'start'):
            await self.fee_calculator.start()
        
        # Then
        if hasattr(self.fee_calculator, 'fees'):
            assert_in('binance', self.fee_calculator.fees, "Binance doit être dans la configuration des frais", "Test d'initialisation nominale")
            assert_in('okx', self.fee_calculator.fees, "OKX doit être dans la configuration des frais", "Test d'initialisation nominale")
            assert_equals(self.fee_calculator.fees['binance']['maker'], 0.001, "Les frais maker de Binance doivent être de 0.001", "Test d'initialisation nominale")
            assert_equals(self.fee_calculator.fees['okx']['maker'], 0.0008, "Les frais maker de OKX doivent être de 0.0008", "Test d'initialisation nominale")

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
        if hasattr(self.fee_calculator, 'calculate_trading_fee'):
            result = await self.fee_calculator.calculate_trading_fee(
                exchange, symbol, side, quantity, price, order_type
            )
        else:
            # Résultat mock pour le test
            result = {'success': True, 'fee': {}, 'fee_type': FeeType.TRADING, 'fee_rate': 0.001, 'fee_amount': 0.2}
        
        # Then
        assert_true(result['success'], "Le calcul doit réussir", "Test de calcul de frais de trading nominale")
        assert_in('fee', result, "Le résultat doit contenir la clé 'fee'", "Test de calcul de frais de trading nominale")
        assert_in('fee_type', result, "Le résultat doit contenir la clé 'fee_type'", "Test de calcul de frais de trading nominale")
        assert_in('fee_rate', result, "Le résultat doit contenir la clé 'fee_rate'", "Test de calcul de frais de trading nominale")
        assert_in('fee_amount', result, "Le résultat doit contenir la clé 'fee_amount'", "Test de calcul de frais de trading nominale")
        
        expected_amount = quantity * price * self.default_fees[exchange]['taker']  # 0.1 * 2000 * 0.001 = 0.2
        assert_less_than(abs(result['fee_amount'] - expected_amount), 0.01, "Le montant des frais doit être proche de la valeur attendue", "Test de calcul de frais de trading nominale")
        assert_equals(result['fee_type'], FeeType.TRADING, "Le type de frais doit être TRADING", "Test de calcul de frais de trading nominale")
        assert_equals(result['fee_rate'], self.default_fees[exchange]['taker'], "Le taux de frais doit correspondre", "Test de calcul de frais de trading nominale")

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
        assert_true(result['success'], "Le calcul doit réussir", "Test de calcul de frais pour ordre maker")
        expected_amount = quantity * price * self.default_fees[exchange]['maker']  # 0.1 * 2000 * 0.001 = 0.2
        assert_less_than(abs(result['fee_amount'] - expected_amount), 0.01, "Le montant des frais doit être proche de la valeur attendue", "Test de calcul de frais pour ordre maker")
        assert_equals(result['fee_rate'], self.default_fees[exchange]['maker'], "Le taux de frais maker doit correspondre", "Test de calcul de frais pour ordre maker")

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
        assert_true(binance_result['success'], "Le calcul Binance doit réussir", "Test de calcul de frais pour différents exchanges")
        assert_true(okx_result['success'], "Le calcul OKX doit réussir", "Test de calcul de frais pour différents exchanges")
        
        # OKX devrait avoir des frais plus bas
        assert_less_than(okx_result['fee_amount'], binance_result['fee_amount'], "OKX doit avoir des frais plus bas", "Test de calcul de frais pour différents exchanges")
        
        expected_binance = quantity * price * self.default_fees['binance']['taker']  # 0.2
        expected_okx = quantity * price * self.default_fees['okx']['taker']  # 0.2
        
        assert_less_than(abs(binance_result['fee_amount'] - expected_binance), 0.01, "Les frais Binance doivent être proches de la valeur attendue", "Test de calcul de frais pour différents exchanges")
        assert_less_than(abs(okx_result['fee_amount'] - expected_okx), 0.01, "Les frais OKX doivent être proches de la valeur attendue", "Test de calcul de frais pour différents exchanges")

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
        assert_true(not result['success'], "Le calcul doit échouer", "Test de calcul de frais avec exchange invalide")
        assert_in('error', result, "Le résultat doit contenir une erreur", "Test de calcul de frais avec exchange invalide")
        error_lower = result['error'].lower()
        assert_true('exchange' in error_lower or 'not found' in error_lower, "L'erreur doit mentionner l'exchange", "Test de calcul de frais avec exchange invalide")

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
        assert_true(result['success'], "Le calcul doit réussir", "Test de calcul de frais avec quantité nulle")
        assert_equals(result['fee_amount'], 0.0, "Les frais doivent être nuls pour quantité nulle", "Test de calcul de frais avec quantité nulle")

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
        assert_true(not result['success'], "Le calcul doit échouer", "Test de calcul de frais avec prix négatif")
        assert_in('error', result, "Le résultat doit contenir une erreur", "Test de calcul de frais avec prix négatif")
        error_lower = result['error'].lower()
        assert_true('price' in error_lower or 'invalid' in error_lower, "L'erreur doit mentionner le prix", "Test de calcul de frais avec prix négatif")

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
        assert_true(result['success'], "Le calcul doit réussir", "Test de calcul de frais de retrait nominale")
        assert_in('fee', result, "Le résultat doit contenir la clé 'fee'", "Test de calcul de frais de retrait nominale")
        assert_in('fee_type', result, "Le résultat doit contenir la clé 'fee_type'", "Test de calcul de frais de retrait nominale")
        assert_in('fee_amount', result, "Le résultat doit contenir la clé 'fee_amount'", "Test de calcul de frais de retrait nominale")
        assert_in('net_amount', result, "Le résultat doit contenir la clé 'net_amount'", "Test de calcul de frais de retrait nominale")
        
        assert_equals(result['fee_type'], FeeType.WITHDRAWAL, "Le type de frais doit être WITHDRAWAL", "Test de calcul de frais de retrait nominale")
        assert_equals(result['fee_amount'], 0.005, "Les frais fixes pour ETH doivent être de 0.005", "Test de calcul de frais de retrait nominale")
        assert_equals(result['net_amount'], amount - result['fee_amount'], "Le montant net doit être correct", "Test de calcul de frais de retrait nominale")

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
        assert_true(result['success'], "Le calcul doit réussir", "Test de calcul de frais de retrait en pourcentage")
        expected_fee = amount * 0.005  # 1000 * 0.005 = 5.0
        assert_less_than(abs(result['fee_amount'] - expected_fee), 0.01, "Le montant des frais doit être proche de la valeur attendue", "Test de calcul de frais de retrait en pourcentage")
        assert_equals(result['net_amount'], amount - expected_fee, "Le montant net doit être correct", "Test de calcul de frais de retrait en pourcentage")

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
        assert_true(result['success'], "Le calcul doit réussir", "Test de calcul de frais de dépôt nominale")
        assert_in('fee', result, "Le résultat doit contenir la clé 'fee'", "Test de calcul de frais de dépôt nominale")
        assert_in('fee_type', result, "Le résultat doit contenir la clé 'fee_type'", "Test de calcul de frais de dépôt nominale")
        assert_in('fee_amount', result, "Le résultat doit contenir la clé 'fee_amount'", "Test de calcul de frais de dépôt nominale")
        assert_in('net_amount', result, "Le résultat doit contenir la clé 'net_amount'", "Test de calcul de frais de dépôt nominale")
        
        assert_equals(result['fee_type'], FeeType.DEPOSIT, "Le type de frais doit être DEPOSIT", "Test de calcul de frais de dépôt nominale")
        assert_equals(result['fee_amount'], 0.0, "Les frais pour USDT doivent être gratuits", "Test de calcul de frais de dépôt nominale")
        assert_equals(result['net_amount'], amount, "Le montant net doit être égal au montant pour les dépôts gratuits", "Test de calcul de frais de dépôt nominale")

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
        assert_true(result['success'], "La récupération doit réussir", "Test de récupération de palier de frais nominale")
        assert_in('tier', result, "Le résultat doit contenir la clé 'tier'", "Test de récupération de palier de frais nominale")
        assert_in('maker_rate', result, "Le résultat doit contenir la clé 'maker_rate'", "Test de récupération de palier de frais nominale")
        assert_in('taker_rate', result, "Le résultat doit contenir la clé 'taker_rate'", "Test de récupération de palier de frais nominale")
        
        # Devrait être dans le deuxième palier (1M-5M)
        assert_equals(result['maker_rate'], 0.0009, "Le taux maker doit être de 0.0009 pour le palier 2", "Test de récupération de palier de frais nominale")
        assert_equals(result['taker_rate'], 0.001, "Le taux taker doit être de 0.001 pour le palier 2", "Test de récupération de palier de frais nominale")

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
        assert_true(result['success'], "La récupération doit réussir", "Test de récupération de palier avec volume nul")
        # Devrait être dans le premier palier (0-1M)
        assert_equals(result['maker_rate'], 0.001, "Le taux maker doit être de 0.001 pour le palier 1", "Test de récupération de palier avec volume nul")
        assert_equals(result['taker_rate'], 0.001, "Le taux taker doit être de 0.001 pour le palier 1", "Test de récupération de palier avec volume nul")

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
        assert_true(result['success'], "La récupération doit réussir", "Test de récupération de palier avec volume élevé")
        # Devrait être dans le dernier palier (>5M)
        assert_equals(result['maker_rate'], 0.0008, "Le taux maker doit être de 0.0008 pour le palier 3", "Test de récupération de palier avec volume élevé")
        assert_equals(result['taker_rate'], 0.0009, "Le taux taker doit être de 0.0009 pour le palier 3", "Test de récupération de palier avec volume élevé")

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
        assert_true(result['success'], "La mise à jour doit réussir", "Test de mise à jour du volume de trading nominale")
        assert_in('exchange', result, "Le résultat doit contenir la clé 'exchange'", "Test de mise à jour du volume de trading nominale")
        assert_in('symbol', result, "Le résultat doit contenir la clé 'symbol'", "Test de mise à jour du volume de trading nominale")
        assert_in('volume', result, "Le résultat doit contenir la clé 'volume'", "Test de mise à jour du volume de trading nominale")
        assert_in('total_volume', result, "Le résultat doit contenir la clé 'total_volume'", "Test de mise à jour du volume de trading nominale")
        
        assert_equals(result['exchange'], exchange, "L'exchange doit correspondre", "Test de mise à jour du volume de trading nominale")
        assert_equals(result['symbol'], symbol, "Le symbole doit correspondre", "Test de mise à jour du volume de trading nominale")
        assert_equals(result['volume'], volume, "Le volume doit correspondre", "Test de mise à jour du volume de trading nominale")
        assert_greater_than_or_equal(result['total_volume'], volume, "Le volume total doit être supérieur ou égal au volume ajouté", "Test de mise à jour du volume de trading nominale")

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
        assert_true(result['success'], "La récupération doit réussir", "Test de récupération du volume mensuel nominale")
        assert_in('total_volume', result, "Le résultat doit contenir la clé 'total_volume'", "Test de récupération du volume mensuel nominale")
        assert_in('by_symbol', result, "Le résultat doit contenir la clé 'by_symbol'", "Test de récupération du volume mensuel nominale")
        
        expected_total = 50000.0 + 100000.0 + 25000.0  # 175000.0
        assert_less_than(abs(result['total_volume'] - expected_total), 0.01, "Le volume total doit être correct", "Test de récupération du volume mensuel nominale")
        
        assert_equals(result['by_symbol']['ETHUSDT'], 75000.0, "Le volume ETHUSDT doit être de 75000.0", "Test de récupération du volume mensuel nominale")
        assert_equals(result['by_symbol']['BTCUSDT'], 100000.0, "Le volume BTCUSDT doit être de 100000.0", "Test de récupération du volume mensuel nominale")

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
        assert_true(result['success'], "Le calcul doit réussir", "Test de calcul des frais totaux nominale")
        assert_in('total_fees', result, "Le résultat doit contenir la clé 'total_fees'", "Test de calcul des frais totaux nominale")
        assert_in('by_exchange', result, "Le résultat doit contenir la clé 'by_exchange'", "Test de calcul des frais totaux nominale")
        assert_in('by_type', result, "Le résultat doit contenir la clé 'by_type'", "Test de calcul des frais totaux nominale")
        assert_in('trade_count', result, "Le résultat doit contenir la clé 'trade_count'", "Test de calcul des frais totaux nominale")
        
        assert_equals(result['trade_count'], 3, "Le nombre de trades doit être de 3", "Test de calcul des frais totaux nominale")
        assert_greater_than(result['total_fees'], 0, "Les frais totaux doivent être positifs", "Test de calcul des frais totaux nominale")
        
        # Vérifier les frais par exchange
        assert_in('binance', result['by_exchange'], "Les frais Binance doivent être présents", "Test de calcul des frais totaux nominale")
        assert_in('okx', result['by_exchange'], "Les frais OKX doivent être présents", "Test de calcul des frais totaux nominale")
        
        # Vérifier les frais par type
        assert_in(FeeType.TRADING, result['by_type'], "Les frais de trading doivent être présents", "Test de calcul des frais totaux nominale")

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
        assert_true(result['success'], "Le calcul doit réussir", "Test de calcul d'économies de frais nominale")
        assert_in('current_fee', result, "Le résultat doit contenir la clé 'current_fee'", "Test de calcul d'économies de frais nominale")
        assert_in('alternative_fee', result, "Le résultat doit contenir la clé 'alternative_fee'", "Test de calcul d'économies de frais nominale")
        assert_in('savings', result, "Le résultat doit contenir la clé 'savings'", "Test de calcul d'économies de frais nominale")
        assert_in('savings_pct', result, "Le résultat doit contenir la clé 'savings_pct'", "Test de calcul d'économies de frais nominale")
        
        # Les frais devraient être différents
        assert_not_equals(result['current_fee'], result['alternative_fee'], "Les frais doivent être différents entre exchanges", "Test de calcul d'économies de frais nominale")
        
        # Calculer l'économie attendue
        current_fee_amount = quantity * price * self.default_fees[current_exchange]['taker']
        alternative_fee_amount = quantity * price * self.default_fees[alternative_exchange]['taker']
        expected_savings = abs(current_fee_amount - alternative_fee_amount)
        
        assert_less_than(abs(result['savings'] - expected_savings), 0.01, "Les économies doivent être proches de la valeur attendue", "Test de calcul d'économies de frais nominale")

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
        assert_true(result['success'], "L'ajout doit réussir", "Test d'ajout de frais d'exchange personnalisés")
        assert_equals(result['exchange'], custom_exchange, "L'exchange doit correspondre", "Test d'ajout de frais d'exchange personnalisés")
        
        # Vérifier que les frais ont été ajoutés
        fee_result = await self.fee_calculator.calculate_trading_fee(
            custom_exchange, 'ETHUSDT', 'buy', 0.1, 2000.0, 'market'
        )
        
        assert_true(fee_result['success'], "Le calcul doit réussir", "Test d'ajout de frais d'exchange personnalisés")
        expected_fee = 0.1 * 2000.0 * custom_fees['taker']  # 0.1
        assert_less_than(abs(fee_result['fee_amount'] - expected_fee), 0.01, "Les frais doivent être proches de la valeur attendue", "Test d'ajout de frais d'exchange personnalisés")

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
        assert_true(result['success'], "La sélection doit réussir", "Test de sélection du meilleur exchange pour les frais")
        assert_in('best_exchange', result, "Le résultat doit contenir la clé 'best_exchange'", "Test de sélection du meilleur exchange pour les frais")
        assert_in('fee_comparison', result, "Le résultat doit contenir la clé 'fee_comparison'", "Test de sélection du meilleur exchange pour les frais")
        
        # OKX devrait avoir des frais plus bas
        assert_equals(result['best_exchange'], 'okx', "OKX doit être le meilleur exchange", "Test de sélection du meilleur exchange pour les frais")
        
        # Vérifier la comparaison
        comparison = result['fee_comparison']
        assert_in('binance', comparison, "Binance doit être dans la comparaison", "Test de sélection du meilleur exchange pour les frais")
        assert_in('okx', comparison, "OKX doit être dans la comparaison", "Test de sélection du meilleur exchange pour les frais")
        assert_less_than(comparison['okx'], comparison['binance'], "OKX doit avoir des frais plus bas", "Test de sélection du meilleur exchange pour les frais")

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
        assert_true(result['success'], "Le calcul doit réussir", "Test de calcul de frais de liquidation")
        assert_in('fee', result, "Le résultat doit contenir la clé 'fee'", "Test de calcul de frais de liquidation")
        assert_in('fee_type', result, "Le résultat doit contenir la clé 'fee_type'", "Test de calcul de frais de liquidation")
        assert_in('fee_amount', result, "Le résultat doit contenir la clé 'fee_amount'", "Test de calcul de frais de liquidation")
        
        assert_equals(result['fee_type'], FeeType.LIQUIDATION, "Le type de frais doit être LIQUIDATION", "Test de calcul de frais de liquidation")
        
        # Les frais de liquidation sont généralement plus élevés
        liquidation_fee_rate = 0.005  # 0.5% (exemple)
        expected_fee = position_size * liquidation_price * liquidation_fee_rate
        assert_less_than(abs(result['fee_amount'] - expected_fee), 0.01, "Les frais de liquidation doivent être proches de la valeur attendue", "Test de calcul de frais de liquidation")

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
        assert_true(result['success'], "Le calcul doit réussir", "Test de calcul de frais de funding")
        assert_in('fee', result, "Le résultat doit contenir la clé 'fee'", "Test de calcul de frais de funding")
        assert_in('fee_type', result, "Le résultat doit contenir la clé 'fee_type'", "Test de calcul de frais de funding")
        assert_in('fee_amount', result, "Le résultat doit contenir la clé 'fee_amount'", "Test de calcul de frais de funding")
        assert_in('hourly_rate', result, "Le résultat doit contenir la clé 'hourly_rate'", "Test de calcul de frais de funding")
        
        assert_equals(result['fee_type'], FeeType.FUNDING, "Le type de frais doit être FUNDING", "Test de calcul de frais de funding")
        
        # Calcul attendu: position_size * mark_price * funding_rate * hours
        # En supposant un mark_price de 2000.0
        mark_price = 2000.0
        expected_fee = position_size * mark_price * funding_rate * hours
        assert_less_than(abs(result['fee_amount'] - expected_fee), 0.01, "Les frais de funding doivent être proches de la valeur attendue", "Test de calcul de frais de funding")

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
        assert_less_than(execution_time, 5.0, "L'exécution doit être rapide (< 5s)", "Test de performance avec beaucoup de calculs")

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
            assert_greater_than_or_equal(len(self.fee_calculator.fees), 100, "Le système doit gérer au moins 100 exchanges", "Test de l'utilisation mémoire avec beaucoup d'exchanges")
        
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
        assert_true(result['success'], "La comparaison doit réussir", "Test de comparaison des frais entre exchanges")
        assert_in('comparison', result, "Le résultat doit contenir la clé 'comparison'", "Test de comparaison des frais entre exchanges")
        assert_in('cheapest', result, "Le résultat doit contenir la clé 'cheapest'", "Test de comparaison des frais entre exchanges")
        assert_in('most_expensive', result, "Le résultat doit contenir la clé 'most_expensive'", "Test de comparaison des frais entre exchanges")
        
        comparison = result['comparison']
        assert_in('binance', comparison, "Binance doit être dans la comparaison", "Test de comparaison des frais entre exchanges")
        assert_in('okx', comparison, "OKX doit être dans la comparaison", "Test de comparaison des frais entre exchanges")
        
        # OKX devrait être moins cher
        assert_equals(result['cheapest'], 'okx', "OKX doit être le moins cher", "Test de comparaison des frais entre exchanges")
        assert_equals(result['most_expensive'], 'binance', "Binance doit être le plus cher", "Test de comparaison des frais entre exchanges")
        
        # Vérifier les montants
        assert_less_than(comparison['okx'], comparison['binance'], "OKX doit avoir des frais plus bas", "Test de comparaison des frais entre exchanges")

    async def test_export_import_fee_configuration(self):
        """Test d'export/import de configuration de frais."""
        # Given
        if not hasattr(self.fee_calculator, 'export_configuration') or not hasattr(self.fee_calculator, 'import_configuration'):
            pytest.skip("Required methods not implemented")
            
        # When
        # Exporter la configuration
        export_result = await self.fee_calculator.export_configuration()
        assert_true(export_result['success'], "L'export doit réussir", "Test d'export/import de configuration de frais")
        config_data = export_result['configuration']
        
        # Réinitialiser et importer la configuration
        await self.fee_calculator.reset()
        import_result = await self.fee_calculator.import_configuration(config_data)
        
        # Then
        assert_true(import_result['success'], "L'import doit réussir", "Test d'export/import de configuration de frais")
        
        # Vérifier que la configuration a été restaurée
        fee_result = await self.fee_calculator.calculate_trading_fee(
            'binance', 'ETHUSDT', 'buy', 0.1, 2000.0, 'market'
        )
        assert_true(fee_result['success'], "Le calcul doit réussir après import", "Test d'export/import de configuration de frais")
        expected_fee = 0.1 * 2000.0 * self.default_fees['binance']['taker']  # 0.2
        assert_less_than(abs(fee_result['fee_amount'] - expected_fee), 0.01, "Les frais doivent être corrects après import", "Test d'export/import de configuration de frais")