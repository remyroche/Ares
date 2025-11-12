"""
Trading Receiver pour le projet ARES.

Ce module gère la réception et le traitement des signaux de trading,
la validation des signaux et l'exécution des ordres.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from .enums import (
    SignalStatus, ReceiverState, TradingSignal
)


class TradingReceiver:
    """
    Récepteur de trading pour la gestion des signaux.
    
    Cette classe est responsable de:
    - Recevoir les signaux de trading
    - Valider les signaux
    - Exécuter les ordres correspondants
    - Suivre l'état des signaux
    - Gérer les risques
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialiser le récepteur de trading.
        
        Args:
            config: Configuration du récepteur
        """
        self.config = config
        self._running = False
        self._processing_task = None
        self.state = ReceiverState.STOPPED
        self.received_signals = []  # Signaux reçus
        self.processed_signals = []  # Signaux traités
        self.active_signals = {}  # {signal_id: signal_info}
        self.logger = logging.getLogger(__name__)
        
        # Initialiser les composants (simulation)
        self.order_router = None
        self.exchange_dispatcher = None
        
    async def start(self):
        """Démarrer le récepteur et la tâche de traitement."""
        if self._running:
            self.logger.warning("TradingReceiver is already running")
            return
            
        self._running = True
        self.state = ReceiverState.ACTIVE
        self._processing_task = asyncio.create_task(self._processing_loop())
        self.logger.info("TradingReceiver started")
        
    async def stop(self):
        """Arrêter le récepteur et la tâche de traitement."""
        if not self._running:
            return False
            
        self._running = False
        self.state = ReceiverState.STOPPING
        
        if self._processing_task:
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass
            self._processing_task = None
            
        self.state = ReceiverState.STOPPED
        self.logger.info("TradingReceiver stopped")
        return True
        
    async def receive_trading_signal(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recevoir un signal de trading.
        
        Args:
            signal: Signal de trading reçu
            
        Returns:
            Dict contenant le résultat de la réception
        """
        try:
            # Validation du signal
            validation_result = await self._validate_signal(signal)
            if not validation_result['valid']:
                return {
                    'success': False,
                    'error': validation_result['error'],
                    'signal': signal
                }
                
            # Validation des risques
            risk_result = await self._check_risk_limits(signal)
            if not risk_result['allowed']:
                return {
                    'success': False,
                    'error': f"Risk limit exceeded: {risk_result['reason']}",
                    'signal': signal
                }
                
            # Générer un ID de signal unique
            signal_id = signal.get('signal_id') or f"signal_{datetime.now().timestamp()}"
            
            # Créer l'objet de signal
            trading_signal = TradingSignal(
                symbol=signal['symbol'],
                side=signal['side'],
                order_type=signal['order_type'],
                quantity=signal['quantity'],
                price=signal.get('price', 0.0),
                exchange=signal.get('exchange'),
                timestamp=signal.get('timestamp', datetime.now()),
                confidence=signal.get('confidence', 0.0),
                strategy=signal.get('strategy'),
                signal_id=signal_id,
                metadata=signal.get('metadata', {})
            )
            
            # Ajouter aux signaux reçus
            signal_record = {
                'signal_id': signal_id,
                'signal': trading_signal,
                'status': SignalStatus.RECEIVED,
                'timestamp': datetime.now(),
                'raw_signal': signal
            }
            self.received_signals.append(signal_record)
            self.active_signals[signal_id] = signal_record
            
            self.logger.info(f"Signal {signal_id} received for {trading_signal.symbol}")
            
            return {
                'success': True,
                'signal_id': signal_id,
                'status': SignalStatus.RECEIVED,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Error receiving trading signal: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'signal': signal
            }
            
    async def process_trading_signal(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Traiter un signal de trading.
        
        Args:
            signal: Signal de trading à traiter
            
        Returns:
            Dict contenant le résultat du traitement
        """
        try:
            # D'abord recevoir le signal
            receive_result = await self.receive_trading_signal(signal)
            if not receive_result['success']:
                return receive_result
                
            signal_id = receive_result['signal_id']
            
            # Calculer la taille de position
            position_result = await self._calculate_position_size(signal)
            final_quantity = position_result['quantity']
            
            # Déterminer l'exchange cible
            target_exchange = signal.get('exchange')
            if not target_exchange:
                # Utiliser l'exchange par défaut ou le meilleur exchange
                target_exchange = self.config.get('primary_exchange', 'binance')
                
            # Router l'ordre
            if self.order_router:
                order_result = await self.order_router.route_order(
                    exchange=target_exchange,
                    symbol=signal['symbol'],
                    side=signal['side'],
                    order_type=signal['order_type'],
                    quantity=final_quantity,
                    price=signal.get('price')
                )
            else:
                # Simulation
                order_result = {
                    'success': True,
                    'order_id': f"order_{datetime.now().timestamp()}",
                    'exchange': target_exchange,
                    'status': 'submitted'
                }
                
            # Mettre à jour le statut du signal
            if order_result['success']:
                signal_record = self.active_signals.get(signal_id)
                if signal_record:
                    signal_record['status'] = SignalStatus.PROCESSED
                    signal_record['order_id'] = order_result['order_id']
                    signal_record['exchange'] = target_exchange
                    signal_record['processed_at'] = datetime.now()
                    
                # Ajouter aux signaux traités
                self.processed_signals.append(signal_record)
                
                # Retirer des signaux actifs
                if signal_id in self.active_signals:
                    del self.active_signals[signal_id]
                    
                self.logger.info(f"Signal {signal_id} processed, order {order_result['order_id']} created")
                
                return {
                    'success': True,
                    'signal_id': signal_id,
                    'order_id': order_result['order_id'],
                    'exchange': target_exchange,
                    'status': SignalStatus.PROCESSED,
                    'quantity': final_quantity
                }
            else:
                # Marquer comme échoué
                signal_record = self.active_signals.get(signal_id)
                if signal_record:
                    signal_record['status'] = SignalStatus.FAILED
                    signal_record['error'] = order_result.get('error', 'Unknown error')
                    signal_record['failed_at'] = datetime.now()
                    
                return {
                    'success': False,
                    'signal_id': signal_id,
                    'error': order_result.get('error', 'Order routing failed'),
                    'status': SignalStatus.FAILED
                }
                
        except Exception as e:
            self.logger.error(f"Error processing trading signal: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'signal': signal
            }
            
    async def receive_and_process_signal(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recevoir et traiter un signal en une seule étape.
        
        Args:
            signal: Signal de trading
            
        Returns:
            Dict contenant le résultat
        """
        return await self.process_trading_signal(signal)
        
    async def cancel_signal(self, signal_id: str) -> Dict[str, Any]:
        """
        Annuler un signal actif.
        
        Args:
            signal_id: ID du signal à annuler
            
        Returns:
            Dict contenant le résultat de l'annulation
        """
        try:
            if signal_id not in self.active_signals:
                return {
                    'success': False,
                    'error': f'Signal {signal_id} not found or not active',
                    'signal_id': signal_id
                }
                
            signal_record = self.active_signals[signal_id]
            
            # Si un ordre a été créé, l'annuler
            if 'order_id' in signal_record and self.order_router:
                cancel_result = await self.order_router.cancel_order(signal_record['order_id'])
                if not cancel_result['success']:
                    return {
                        'success': False,
                        'error': f"Failed to cancel order: {cancel_result['error']}",
                        'signal_id': signal_id
                    }
                    
            # Mettre à jour le statut du signal
            signal_record['status'] = SignalStatus.CANCELLED
            signal_record['cancelled_at'] = datetime.now()
            
            # Retirer des signaux actifs
            del self.active_signals[signal_id]
            
            self.logger.info(f"Signal {signal_id} cancelled")
            
            return {
                'success': True,
                'signal_id': signal_id,
                'order_id': signal_record.get('order_id'),
                'status': SignalStatus.CANCELLED
            }
            
        except Exception as e:
            self.logger.error(f"Error cancelling signal {signal_id}: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'signal_id': signal_id
            }
            
    async def get_signal_status(self, signal_id: str) -> Dict[str, Any]:
        """
        Récupérer le statut d'un signal.
        
        Args:
            signal_id: ID du signal
            
        Returns:
            Dict contenant le statut du signal
        """
        try:
            # Chercher dans les signaux actifs
            if signal_id in self.active_signals:
                signal_record = self.active_signals[signal_id]
                return {
                    'success': True,
                    'signal_id': signal_id,
                    'status': signal_record['status'],
                    'timestamp': signal_record['timestamp'],
                    'order_id': signal_record.get('order_id'),
                    'exchange': signal_record.get('exchange')
                }
                
            # Chercher dans les signaux traités
            for record in self.processed_signals:
                if record['signal_id'] == signal_id:
                    return {
                        'success': True,
                        'signal_id': signal_id,
                        'status': record['status'],
                        'timestamp': record['timestamp'],
                        'order_id': record.get('order_id'),
                        'exchange': record.get('exchange')
                    }
                    
            # Chercher dans les signaux reçus
            for record in self.received_signals:
                if record['signal_id'] == signal_id:
                    return {
                        'success': True,
                        'signal_id': signal_id,
                        'status': record['status'],
                        'timestamp': record['timestamp'],
                        'order_id': record.get('order_id'),
                        'exchange': record.get('exchange')
                    }
                    
            return {
                'success': False,
                'error': f'Signal {signal_id} not found',
                'signal_id': signal_id
            }
            
        except Exception as e:
            self.logger.error(f"Error getting signal status {signal_id}: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'signal_id': signal_id
            }
            
    async def get_active_signals(self,
                                symbol: str = None,
                                status: SignalStatus = None) -> Dict[str, Any]:
        """
        Récupérer les signaux actifs.
        
        Args:
            symbol: Filtrer par symbole (optionnel)
            status: Filtrer par statut (optionnel)
            
        Returns:
            Dict contenant les signaux actifs
        """
        try:
            signals = []
            for signal_id, signal_record in self.active_signals.items():
                signal_info = {
                    'signal_id': signal_id,
                    'symbol': signal_record['signal'].symbol,
                    'side': signal_record['signal'].side,
                    'order_type': signal_record['signal'].order_type,
                    'quantity': signal_record['signal'].quantity,
                    'price': signal_record['signal'].price,
                    'status': signal_record['status'],
                    'timestamp': signal_record['timestamp'],
                    'exchange': signal_record.get('exchange'),
                    'order_id': signal_record.get('order_id')
                }
                
                # Appliquer les filtres
                if symbol and signal_info['symbol'] != symbol:
                    continue
                if status and signal_info['status'] != status:
                    continue
                    
                signals.append(signal_info)
                
            return {
                'success': True,
                'signals': signals,
                'count': len(signals)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting active signals: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
            
    async def get_signal_history(self,
                                symbol: str = None,
                                status: SignalStatus = None,
                                limit: int = 100) -> Dict[str, Any]:
        """
        Récupérer l'historique des signaux.
        
        Args:
            symbol: Filtrer par symbole (optionnel)
            status: Filtrer par statut (optionnel)
            limit: Limite de résultats
            
        Returns:
            Dict contenant l'historique des signaux
        """
        try:
            # Combiner tous les signaux
            all_signals = self.received_signals + self.processed_signals
            
            # Appliquer les filtres
            filtered_signals = []
            for record in all_signals:
                signal_info = {
                    'signal_id': record['signal_id'],
                    'symbol': record['signal'].symbol,
                    'side': record['signal'].side,
                    'order_type': record['signal'].order_type,
                    'quantity': record['signal'].quantity,
                    'price': record['signal'].price,
                    'status': record['status'],
                    'timestamp': record['timestamp'],
                    'exchange': record.get('exchange'),
                    'order_id': record.get('order_id')
                }
                
                # Appliquer les filtres
                if symbol and signal_info['symbol'] != symbol:
                    continue
                if status and signal_info['status'] != status:
                    continue
                    
                filtered_signals.append(signal_info)
                
            # Trier par date (plus récent en premier)
            filtered_signals.sort(key=lambda x: x['timestamp'], reverse=True)
            
            # Limiter les résultats
            filtered_signals = filtered_signals[:limit]
            
            return {
                'success': True,
                'signals': filtered_signals,
                'count': len(filtered_signals)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting signal history: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
            
    async def get_statistics(self) -> Dict[str, Any]:
        """
        Récupérer les statistiques des signaux.
        
        Returns:
            Dict contenant les statistiques
        """
        try:
            total_signals = len(self.received_signals)
            processed_signals = len(self.processed_signals)
            active_signals = len(self.active_signals)
            
            # Compter par statut
            status_counts = {}
            for record in self.received_signals + self.processed_signals:
                status = record['status'].value
                status_counts[status] = status_counts.get(status, 0) + 1
                
            failed_signals = status_counts.get(SignalStatus.FAILED.value, 0)
            cancelled_signals = status_counts.get(SignalStatus.CANCELLED.value, 0)
            
            # Statistiques par symbole
            by_symbol = {}
            for record in self.received_signals + self.processed_signals:
                symbol = record['signal'].symbol
                if symbol not in by_symbol:
                    by_symbol[symbol] = 0
                by_symbol[symbol] += 1
                
            # Statistiques par statut
            by_status = status_counts
            
            # Statistiques par exchange
            by_exchange = {}
            for record in self.processed_signals:
                exchange = record.get('exchange', 'unknown')
                if exchange not in by_exchange:
                    by_exchange[exchange] = 0
                by_exchange[exchange] += 1
                
            return {
                'success': True,
                'statistics': {
                    'total_signals': total_signals,
                    'processed_signals': processed_signals,
                    'active_signals': active_signals,
                    'failed_signals': failed_signals,
                    'cancelled_signals': cancelled_signals,
                    'success_rate': processed_signals / total_signals if total_signals > 0 else 0,
                    'by_symbol': by_symbol,
                    'by_status': by_status,
                    'by_exchange': by_exchange
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error getting statistics: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
            
    async def _validate_signal(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Valider un signal de trading.
        
        Args:
            signal: Signal à valider
            
        Returns:
            Dict contenant le résultat de la validation
        """
        try:
            # Champs requis
            required_fields = ['symbol', 'side', 'order_type', 'quantity']
            for field in required_fields:
                if field not in signal or signal[field] is None:
                    return {
                        'valid': False,
                        'error': f'Missing required field: {field}'
                    }
                    
            # Validation des valeurs
            if signal['quantity'] <= 0:
                return {
                    'valid': False,
                    'error': 'Quantity must be positive'
                }
                
            if signal['side'] not in ['buy', 'sell']:
                return {
                    'valid': False,
                    'error': 'Side must be buy or sell'
                }
                
            if signal['order_type'] not in ['market', 'limit', 'stop', 'stop_limit']:
                return {
                    'valid': False,
                    'error': 'Invalid order type'
                }
                
            return {
                'valid': True
            }
            
        except Exception as e:
            return {
                'valid': False,
                'error': f'Validation error: {str(e)}'
            }
            
    async def _check_risk_limits(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Vérifier les limites de risque.
        
        Args:
            signal: Signal à vérifier
            
        Returns:
            Dict contenant le résultat de la vérification
        """
        try:
            # Simuler des limites de risque
            max_position_size = self.config.get('max_position_size', 1.0)
            max_daily_loss = self.config.get('max_daily_loss', 1000.0)
            
            # Vérifier la taille de position
            if signal['quantity'] > max_position_size:
                return {
                    'allowed': False,
                    'reason': f'Position size {signal["quantity"]} exceeds maximum {max_position_size}'
                }
                
            # Simuler une vérification de perte quotidienne
            # En pratique, cela impliquerait de calculer la perte réelle
            daily_loss = 0.0  # Simulation
            if daily_loss > max_daily_loss:
                return {
                    'allowed': False,
                    'reason': f'Daily loss {daily_loss} exceeds maximum {max_daily_loss}'
                }
                
            return {
                'allowed': True,
                'reason': 'Risk limits passed'
            }
            
        except Exception as e:
            return {
                'allowed': False,
                'reason': f'Risk check error: {str(e)}'
            }
            
    async def _calculate_position_size(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculer la taille de position optimale.
        
        Args:
            signal: Signal d'entrée
            
        Returns:
            Dict contenant la taille calculée
        """
        try:
            # Utiliser la quantité du signal ou calculer automatiquement
            base_quantity = signal.get('quantity', 0.1)
            
            # Appliquer des facteurs de réduction basés sur la confiance
            confidence = signal.get('confidence', 1.0)
            adjusted_quantity = base_quantity * confidence
            
            # Appliquer des limites
            max_quantity = self.config.get('max_position_size', 1.0)
            final_quantity = min(adjusted_quantity, max_quantity)
            
            return {
                'quantity': final_quantity,
                'base_quantity': base_quantity,
                'confidence_factor': confidence,
                'max_limit': max_quantity
            }
            
        except Exception as e:
            return {
                'quantity': 0.1,  # Valeur par défaut sécuritaire
                'error': str(e)
            }
            
    async def _processing_loop(self):
        """Boucle de traitement des signaux."""
        while self._running:
            try:
                # Traiter les signaux en attente
                # (Dans cette implémentation, les signaux sont traités immédiatement)
                
                # Nettoyer les anciens signaux
                cutoff_time = datetime.now() - timedelta(days=7)
                self.received_signals = [
                    s for s in self.received_signals 
                    if s['timestamp'] > cutoff_time
                ]
                self.processed_signals = [
                    s for s in self.processed_signals 
                    if s['timestamp'] > cutoff_time
                ]
                
                # Attendre avant la prochaine vérification
                await asyncio.sleep(10)  # Vérification toutes les 10 secondes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in processing loop: {str(e)}")
                await asyncio.sleep(5)  # Attendre avant de réessayer