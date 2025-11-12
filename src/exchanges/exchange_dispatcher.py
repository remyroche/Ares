"""
Exchange Dispatcher pour le projet ARES.

Ce module gère la distribution des ordres vers différents exchanges,
le monitoring de leur état et la sélection du meilleur exchange.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from .enums import (
    ExchangeStatus, OrderStatus, OrderType, OrderSide, DispatchResult
)


class ExchangeDispatcher:
    """
    Dispatcheur d'échanges pour la gestion multi-exchange.
    
    Cette classe est responsable de:
    - Router les ordres vers les exchanges appropriés
    - Monitorer l'état des exchanges
    - Sélectionner le meilleur exchange pour un ordre donné
    - Gérer les basculements en cas d'échec
    """
    
    def __init__(self, exchange_registry):
        """
        Initialiser le dispatcheur d'échanges.
        
        Args:
            exchange_registry: Registre des exchanges disponibles
        """
        self.exchange_registry = exchange_registry
        self._running = False
        self._monitoring_task = None
        self.exchange_status = {}  # {exchange_name: status_info}
        self.dispatch_history = []  # Historique des dispatches
        self.logger = logging.getLogger(__name__)
        
    async def start(self):
        """Démarrer le dispatcheur et la tâche de monitoring."""
        if self._running:
            self.logger.warning("ExchangeDispatcher is already running")
            return
            
        self._running = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        self.logger.info("ExchangeDispatcher started")
        
    async def stop(self):
        """Arrêter le dispatcheur et la tâche de monitoring."""
        if not self._running:
            return False
            
        self._running = False
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
            self._monitoring_task = None
            
        self.logger.info("ExchangeDispatcher stopped")
        return True
        
    async def dispatch_to_exchange(self, 
                                 exchange: str,
                                 symbol: str,
                                 side: str,
                                 order_type: str,
                                 quantity: float,
                                 price: float = None) -> Dict[str, Any]:
        """
        Dispatcher un ordre vers un exchange spécifique.
        
        Args:
            exchange: Nom de l'exchange cible
            symbol: Symbole de trading
            side: Côté de l'ordre (buy/sell)
            order_type: Type d'ordre
            quantity: Quantité
            price: Prix (pour les ordres limit)
            
        Returns:
            Dict contenant le résultat du dispatch
        """
        try:
            # Validation des entrées
            if not exchange or not symbol or not side or not order_type or quantity <= 0:
                raise ValueError("Invalid order parameters")
                
            # Récupérer l'exchange
            exchange_client = await self.exchange_registry.get_exchange(exchange)
            if not exchange_client:
                return {
                    'success': False,
                    'error': f'Exchange {exchange} not found',
                    'exchange': exchange
                }
                
            # Vérifier le statut de l'exchange
            status = await self.get_exchange_status(exchange)
            if not status['success'] or status['status'] != ExchangeStatus.ACTIVE:
                return {
                    'success': False,
                    'error': f'Exchange {exchange} is not active',
                    'exchange': exchange
                }
                
            # Créer et soumettre l'ordre
            order_data = {
                'symbol': symbol,
                'side': side,
                'type': order_type,
                'amount': quantity
            }
            
            if price is not None:
                order_data['price'] = price
                
            # Soumettre l'ordre (simulation)
            order_id = f"order_{datetime.now().timestamp()}"
            
            # Enregistrer le dispatch
            dispatch_record = {
                'timestamp': datetime.now(),
                'exchange': exchange,
                'symbol': symbol,
                'side': side,
                'order_type': order_type,
                'quantity': quantity,
                'price': price,
                'order_id': order_id,
                'success': True
            }
            self.dispatch_history.append(dispatch_record)
            
            return {
                'success': True,
                'order_id': order_id,
                'exchange': exchange,
                'status': OrderStatus.SUBMITTED,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Error dispatching order to {exchange}: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'exchange': exchange
            }
            
    async def dispatch_to_best_exchange(self,
                                     symbol: str,
                                     side: str,
                                     order_type: str,
                                     quantity: float) -> Dict[str, Any]:
        """
        Dispatcher un ordre vers le meilleur exchange disponible.
        
        Args:
            symbol: Symbole de trading
            side: Côté de l'ordre
            order_type: Type d'ordre
            quantity: Quantité
            
        Returns:
            Dict contenant le résultat du dispatch
        """
        try:
            # Sélectionner le meilleur exchange
            best_exchange = await self.get_best_exchange(symbol, side, order_type)
            if not best_exchange:
                return {
                    'success': False,
                    'error': 'No suitable exchange found'
                }
                
            # Dispatcher vers le meilleur exchange
            return await self.dispatch_to_exchange(
                best_exchange, symbol, side, order_type, quantity
            )
            
        except Exception as e:
            self.logger.error(f"Error dispatching to best exchange: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
            
    async def dispatch_to_multiple_exchanges(self,
                                          symbol: str,
                                          side: str,
                                          order_type: str,
                                          total_quantity: float,
                                          exchanges: List[str],
                                          allocation: Dict[str, float] = None) -> Dict[str, Any]:
        """
        Dispatcher un ordre vers plusieurs exchanges.
        
        Args:
            symbol: Symbole de trading
            side: Côté de l'ordre
            order_type: Type d'ordre
            total_quantity: Quantité totale
            exchanges: Liste des exchanges cibles
            allocation: Allocation par exchange (si None, allocation automatique)
            
        Returns:
            Dict contenant le résultat du dispatch
        """
        try:
            # Validation
            if not exchanges:
                raise ValueError("No exchanges specified")
                
            # Allocation automatique si non spécifiée
            if allocation is None:
                allocation = {exchange: 1.0/len(exchanges) for exchange in exchanges}
                
            # Vérifier que l'allocation totale = 1.0
            total_allocation = sum(allocation.values())
            if abs(total_allocation - 1.0) > 0.01:
                return {
                    'success': False,
                    'error': f'Invalid allocation: total = {total_allocation}, expected 1.0'
                }
                
            # Dispatcher vers chaque exchange
            orders = []
            for exchange in exchanges:
                if exchange not in allocation:
                    continue
                    
                quantity = total_quantity * allocation[exchange]
                result = await self.dispatch_to_exchange(
                    exchange, symbol, side, order_type, quantity
                )
                
                if result['success']:
                    orders.append({
                        'exchange': exchange,
                        'order_id': result['order_id'],
                        'quantity': quantity,
                        'allocation': allocation[exchange]
                    })
                    
            return {
                'success': len(orders) > 0,
                'orders': orders,
                'total_dispatched': sum(o['quantity'] for o in orders)
            }
            
        except Exception as e:
            self.logger.error(f"Error dispatching to multiple exchanges: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
            
    async def get_best_exchange(self,
                              symbol: str,
                              side: str,
                              order_type: str) -> Optional[str]:
        """
        Sélectionner le meilleur exchange pour un ordre donné.
        
        Args:
            symbol: Symbole de trading
            side: Côté de l'ordre
            order_type: Type d'ordre
            
        Returns:
            Nom du meilleur exchange ou None
        """
        try:
            # Récupérer les données de marché
            market_data = await self._get_market_data(symbol)
            
            # Filtrer les exchanges actifs
            active_exchanges = []
            for exchange, data in market_data.items():
                status = await self.get_exchange_status(exchange)
                if status['success'] and status['status'] == ExchangeStatus.ACTIVE:
                    active_exchanges.append((exchange, data))
                    
            if not active_exchanges:
                return None
                
            # Sélectionner basé sur le critère approprié
            if side == 'buy':
                # Pour les achats, choisir le prix le plus bas
                best_exchange = min(active_exchanges, key=lambda x: x[1]['price'])
            else:
                # Pour les ventes, choisir le prix le plus haut
                best_exchange = max(active_exchanges, key=lambda x: x[1]['price'])
                
            return best_exchange[0]
            
        except Exception as e:
            self.logger.error(f"Error selecting best exchange: {str(e)}")
            return None
            
    async def get_exchange_status(self, exchange: str) -> Dict[str, Any]:
        """
        Récupérer le statut d'un exchange.
        
        Args:
            exchange: Nom de l'exchange
            
        Returns:
            Dict contenant le statut
        """
        try:
            if exchange not in self.exchange_status:
                # Initialiser le statut
                self.exchange_status[exchange] = {
                    'status': ExchangeStatus.ACTIVE,
                    'last_check': datetime.now(),
                    'latency': 0,
                    'error_rate': 0.0
                }
                
            return {
                'success': True,
                'exchange': exchange,
                **self.exchange_status[exchange]
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'exchange': exchange
            }
            
    async def get_all_exchanges_status(self) -> Dict[str, Any]:
        """
        Récupérer le statut de tous les exchanges.
        
        Returns:
            Dict contenant le statut de tous les exchanges
        """
        try:
            exchanges = await self.exchange_registry.get_registered_exchanges()
            status_dict = {}
            
            for exchange in exchanges:
                status = await self.get_exchange_status(exchange)
                if status['success']:
                    status_dict[exchange] = status
                    
            return {
                'success': True,
                'exchanges': status_dict
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
            
    async def update_exchange_status(self,
                                  exchange: str,
                                  status: ExchangeStatus,
                                  latency: float = 0,
                                  error_rate: float = 0) -> Dict[str, Any]:
        """
        Mettre à jour le statut d'un exchange.
        
        Args:
            exchange: Nom de l'exchange
            status: Nouveau statut
            latency: Latence en ms
            error_rate: Taux d'erreur
            
        Returns:
            Dict contenant le résultat
        """
        try:
            self.exchange_status[exchange] = {
                'status': status,
                'last_check': datetime.now(),
                'latency': latency,
                'error_rate': error_rate
            }
            
            return {
                'success': True,
                'exchange': exchange,
                'status': status,
                'latency': latency,
                'error_rate': error_rate
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'exchange': exchange
            }
            
    async def disable_exchange(self, exchange: str, reason: str = None) -> Dict[str, Any]:
        """
        Désactiver un exchange.
        
        Args:
            exchange: Nom de l'exchange
            reason: Raison de la désactivation
            
        Returns:
            Dict contenant le résultat
        """
        return await self.update_exchange_status(
            exchange, ExchangeStatus.DISABLED
        )
        
    async def enable_exchange(self, exchange: str) -> Dict[str, Any]:
        """
        Activer un exchange.
        
        Args:
            exchange: Nom de l'exchange
            
        Returns:
            Dict contenant le résultat
        """
        return await self.update_exchange_status(
            exchange, ExchangeStatus.ACTIVE
        )
        
    async def get_dispatch_history(self,
                                 exchange: str = None,
                                 symbol: str = None,
                                 limit: int = 100) -> Dict[str, Any]:
        """
        Récupérer l'historique des dispatches.
        
        Args:
            exchange: Filtrer par exchange (optionnel)
            symbol: Filtrer par symbole (optionnel)
            limit: Limite de résultats
            
        Returns:
            Dict contenant l'historique
        """
        try:
            history = self.dispatch_history.copy()
            
            # Appliquer les filtres
            if exchange:
                history = [h for h in history if h['exchange'] == exchange]
            if symbol:
                history = [h for h in history if h['symbol'] == symbol]
                
            # Trier par date (plus récent en premier)
            history.sort(key=lambda x: x['timestamp'], reverse=True)
            
            # Limiter les résultats
            history = history[:limit]
            
            return {
                'success': True,
                'history': history,
                'count': len(history)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
            
    async def get_statistics(self) -> Dict[str, Any]:
        """
        Récupérer les statistiques de dispatch.
        
        Returns:
            Dict contenant les statistiques
        """
        try:
            total_dispatches = len(self.dispatch_history)
            successful_dispatches = len([h for h in self.dispatch_history if h['success']])
            failed_dispatches = total_dispatches - successful_dispatches
            
            # Statistiques par exchange
            by_exchange = {}
            for record in self.dispatch_history:
                exchange = record['exchange']
                if exchange not in by_exchange:
                    by_exchange[exchange] = {'total': 0, 'success': 0, 'failed': 0}
                by_exchange[exchange]['total'] += 1
                if record['success']:
                    by_exchange[exchange]['success'] += 1
                else:
                    by_exchange[exchange]['failed'] += 1
                    
            # Statistiques par symbole
            by_symbol = {}
            for record in self.dispatch_history:
                symbol = record['symbol']
                if symbol not in by_symbol:
                    by_symbol[symbol] = 0
                by_symbol[symbol] += 1
                
            # Statistiques par côté
            by_side = {}
            for record in self.dispatch_history:
                side = record['side']
                if side not in by_side:
                    by_side[side] = 0
                by_side[side] += 1
                
            return {
                'success': True,
                'statistics': {
                    'total_dispatches': total_dispatches,
                    'successful_dispatches': successful_dispatches,
                    'failed_dispatches': failed_dispatches,
                    'success_rate': successful_dispatches / total_dispatches if total_dispatches > 0 else 0,
                    'by_exchange': by_exchange,
                    'by_symbol': by_symbol,
                    'by_side': by_side
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
            
    async def check_exchange_health(self, exchange: str) -> Dict[str, Any]:
        """
        Vérifier la santé d'un exchange.
        
        Args:
            exchange: Nom de l'exchange
            
        Returns:
            Dict contenant le résultat du health check
        """
        try:
            exchange_client = await self.exchange_registry.get_exchange(exchange)
            if not exchange_client:
                return {
                    'success': False,
                    'error': f'Exchange {exchange} not found'
                }
                
            # Simuler un ping
            start_time = datetime.now()
            
            # Simuler une réponse
            latency = (datetime.now() - start_time).total_seconds() * 1000
            
            # Déterminer si l'exchange est sain
            healthy = latency < 1000  # Moins de 1 seconde de latence
            
            return {
                'success': True,
                'exchange': exchange,
                'healthy': healthy,
                'latency': latency,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            return {
                'success': True,
                'exchange': exchange,
                'healthy': False,
                'error': str(e),
                'timestamp': datetime.now()
            }
            
    async def _monitoring_loop(self):
        """Boucle de monitoring des exchanges."""
        while self._running:
            try:
                # Récupérer tous les exchanges
                exchanges = await self.exchange_registry.get_registered_exchanges()
                
                # Vérifier la santé de chaque exchange
                for exchange in exchanges:
                    health = await self.check_exchange_health(exchange)
                    
                    # Mettre à jour le statut
                    if health['success']:
                        status = ExchangeStatus.ACTIVE if health['healthy'] else ExchangeStatus.ERROR
                        await self.update_exchange_status(
                            exchange,
                            status,
                            health.get('latency', 0),
                            0.0 if health['healthy'] else 1.0
                        )
                        
                # Attendre avant la prochaine vérification
                await asyncio.sleep(30)  # Vérification toutes les 30 secondes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {str(e)}")
                await asyncio.sleep(5)  # Attendre avant de réessayer
                
    async def _get_market_data(self, symbol: str) -> Dict[str, Dict[str, Any]]:
        """
        Récupérer les données de marché pour un symbole.
        
        Args:
            symbol: Symbole de trading
            
        Returns:
            Dict contenant les données de marché par exchange
        """
        # Simulation de données de marché
        return {
            'binance': {
                'price': 2000.0,
                'volume': 100.0,
                'spread': 0.1
            },
            'okx': {
                'price': 2001.0,
                'volume': 80.0,
                'spread': 0.15
            }
        }