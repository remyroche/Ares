"""
Order Router pour le projet ARES.

Ce module gère le routage des ordres vers les exchanges,
le suivi de leur état et la gestion du cycle de vie.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from .enums import (
    OrderStatus, OrderType, OrderSide, RoutedOrder
)


class OrderRouter:
    """
    Routeur d'ordres pour la gestion multi-exchange.
    
    Cette classe est responsable de:
    - Router les ordres vers les exchanges appropriés
    - Suivre l'état des ordres
    - Gérer les annulations
    - Maintenir l'historique des ordres
    """
    
    def __init__(self, exchange_registry):
        """
        Initialiser le routeur d'ordres.
        
        Args:
            exchange_registry: Registre des exchanges disponibles
        """
        self.exchange_registry = exchange_registry
        self._running = False
        self._monitoring_task = None
        self.routed_orders = {}  # {order_id: RoutedOrder}
        self.active_orders = {}  # {order_id: RoutedOrder}
        self.order_history = []  # Historique des ordres
        self.logger = logging.getLogger(__name__)
        
    async def start(self):
        """Démarrer le routeur et la tâche de monitoring."""
        if self._running:
            self.logger.warning("OrderRouter is already running")
            return
            
        self._running = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        self.logger.info("OrderRouter started")
        
    async def stop(self):
        """Arrêter le routeur et la tâche de monitoring."""
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
            
        self.logger.info("OrderRouter stopped")
        return True
        
    async def route_order(self,
                         exchange: str,
                         symbol: str,
                         side: str,
                         order_type: str,
                         quantity: float,
                         price: Optional[float] = None) -> Dict[str, Any]:
        """
        Router un ordre vers un exchange.
        
        Args:
            exchange: Nom de l'exchange cible
            symbol: Symbole de trading
            side: Côté de l'ordre (buy/sell)
            order_type: Type d'ordre
            quantity: Quantité
            price: Prix (pour les ordres limit)
            
        Returns:
            Dict contenant le résultat du routage
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
                
            # Générer un ID d'ordre unique
            order_id = f"order_{datetime.now().timestamp()}_{exchange}_{symbol}"
            
            # Créer l'ordre routé
            routed_order = RoutedOrder(
                id=order_id,
                exchange=exchange,
                symbol=symbol,
                side=side,
                order_type=order_type,
                quantity=quantity,
                price=price or 0.0,
                status=OrderStatus.SUBMITTED,
                timestamp=datetime.now()
            )
            
            # Simuler la soumission à l'exchange
            exchange_order_id = f"{exchange}_order_{datetime.now().timestamp()}"
            routed_order.exchange_order_id = exchange_order_id
            
            # Ajouter aux ordres suivis
            self.routed_orders[order_id] = routed_order
            self.active_orders[order_id] = routed_order
            
            # Ajouter à l'historique
            self.order_history.append({
                'order_id': order_id,
                'exchange': exchange,
                'symbol': symbol,
                'side': side,
                'order_type': order_type,
                'quantity': quantity,
                'price': price,
                'status': OrderStatus.SUBMITTED,
                'timestamp': datetime.now()
            })
            
            self.logger.info(f"Order {order_id} routed to {exchange}")
            
            return {
                'success': True,
                'order_id': order_id,
                'exchange': exchange,
                'status': OrderStatus.SUBMITTED,
                'exchange_order_id': exchange_order_id,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Error routing order: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'exchange': exchange
            }
            
    async def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """
        Annuler un ordre.
        
        Args:
            order_id: ID de l'ordre à annuler
            
        Returns:
            Dict contenant le résultat de l'annulation
        """
        try:
            if order_id not in self.routed_orders:
                return {
                    'success': False,
                    'error': f'Order {order_id} not found',
                    'order_id': order_id
                }
                
            order = self.routed_orders[order_id]
            
            # Vérifier si l'ordre peut être annulé
            if order.status not in [OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED]:
                return {
                    'success': False,
                    'error': f'Order {order_id} cannot be cancelled (status: {order.status})',
                    'order_id': order_id
                }
                
            # Récupérer l'exchange client
            exchange_client = await self.exchange_registry.get_exchange(order.exchange)
            if not exchange_client:
                return {
                    'success': False,
                    'error': f'Exchange {order.exchange} not found',
                    'order_id': order_id
                }
                
            # Simuler l'annulation
            order.status = OrderStatus.CANCELLED
            
            # Retirer des ordres actifs
            if order_id in self.active_orders:
                del self.active_orders[order_id]
                
            # Mettre à jour l'historique
            history_entry = next(
                (h for h in self.order_history if h['order_id'] == order_id),
                None
            )
            if history_entry:
                history_entry['status'] = OrderStatus.CANCELLED
                history_entry['cancelled_at'] = datetime.now()
                
            self.logger.info(f"Order {order_id} cancelled")
            
            return {
                'success': True,
                'order_id': order_id,
                'status': OrderStatus.CANCELLED,
                'exchange': order.exchange,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Error cancelling order {order_id}: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'order_id': order_id
            }
            
    async def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """
        Récupérer le statut d'un ordre.
        
        Args:
            order_id: ID de l'ordre
            
        Returns:
            Dict contenant le statut de l'ordre
        """
        try:
            if order_id not in self.routed_orders:
                return {
                    'success': False,
                    'error': f'Order {order_id} not found',
                    'order_id': order_id
                }
                
            order = self.routed_orders[order_id]
            
            # Simuler une mise à jour de statut
            if order.status == OrderStatus.SUBMITTED:
                # Simuler que l'ordre est rempli après un certain temps
                time_elapsed = (datetime.now() - order.timestamp).total_seconds()
                if time_elapsed > 1:  # Après 1 seconde
                    order.status = OrderStatus.FILLED
                    order.filled_quantity = order.quantity
                    order.average_price = order.price if order.price > 0 else 2000.0  # Prix simulé
                    order.fees = order.quantity * 0.001  # 0.1% de frais
                    
                    # Retirer des ordres actifs
                    if order_id in self.active_orders:
                        del self.active_orders[order_id]
                        
            return {
                'success': True,
                'order_id': order_id,
                'status': order.status,
                'exchange': order.exchange,
                'symbol': order.symbol,
                'side': order.side,
                'quantity': order.quantity,
                'filled_quantity': order.filled_quantity,
                'average_price': order.average_price,
                'fees': order.fees,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting order status {order_id}: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'order_id': order_id
            }
            
    async def get_active_orders(self,
                               exchange: Optional[str] = None,
                               symbol: Optional[str] = None,
                               status: Optional[OrderStatus] = None) -> Dict[str, Any]:
        """
        Récupérer les ordres actifs.
        
        Args:
            exchange: Filtrer par exchange (optionnel)
            symbol: Filtrer par symbole (optionnel)
            status: Filtrer par statut (optionnel)
            
        Returns:
            Dict contenant les ordres actifs
        """
        try:
            orders = list(self.active_orders.values())
            
            # Appliquer les filtres
            if exchange:
                orders = [o for o in orders if o.exchange == exchange]
            if symbol:
                orders = [o for o in orders if o.symbol == symbol]
            if status:
                orders = [o for o in orders if o.status == status]
                
            # Convertir en dictionnaires
            orders_dict = []
            for order in orders:
                orders_dict.append({
                    'order_id': order.id,
                    'exchange': order.exchange,
                    'symbol': order.symbol,
                    'side': order.side,
                    'order_type': order.order_type,
                    'quantity': order.quantity,
                    'price': order.price,
                    'status': order.status,
                    'filled_quantity': order.filled_quantity,
                    'average_price': order.average_price,
                    'timestamp': order.timestamp
                })
                
            return {
                'success': True,
                'orders': orders_dict,
                'count': len(orders_dict)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting active orders: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
            
    async def get_order_history(self,
                               exchange: Optional[str] = None,
                               symbol: Optional[str] = None,
                               status: Optional[OrderStatus] = None,
                               limit: int = 100) -> Dict[str, Any]:
        """
        Récupérer l'historique des ordres.
        
        Args:
            exchange: Filtrer par exchange (optionnel)
            symbol: Filtrer par symbole (optionnel)
            status: Filtrer par statut (optionnel)
            limit: Limite de résultats
            
        Returns:
            Dict contenant l'historique des ordres
        """
        try:
            history = self.order_history.copy()
            
            # Appliquer les filtres
            if exchange:
                history = [h for h in history if h['exchange'] == exchange]
            if symbol:
                history = [h for h in history if h['symbol'] == symbol]
            if status:
                history = [h for h in history if h['status'] == status]
                
            # Trier par date (plus récent en premier)
            history.sort(key=lambda x: x['timestamp'], reverse=True)
            
            # Limiter les résultats
            history = history[:limit]
            
            return {
                'success': True,
                'orders': history,
                'count': len(history)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting order history: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
            
    async def get_statistics(self) -> Dict[str, Any]:
        """
        Récupérer les statistiques des ordres.
        
        Returns:
            Dict contenant les statistiques
        """
        try:
            total_routed = len(self.routed_orders)
            active_count = len(self.active_orders)
            
            # Compter par statut
            status_counts = {}
            for order in self.routed_orders.values():
                status = order.status.value
                status_counts[status] = status_counts.get(status, 0) + 1
                
            successful_orders = status_counts.get(OrderStatus.FILLED.value, 0)
            failed_orders = status_counts.get(OrderStatus.CANCELLED.value, 0) + \
                           status_counts.get(OrderStatus.REJECTED.value, 0)
            
            # Statistiques par exchange
            by_exchange = {}
            for order in self.routed_orders.values():
                exchange = order.exchange
                if exchange not in by_exchange:
                    by_exchange[exchange] = {'total': 0, 'success': 0, 'failed': 0}
                by_exchange[exchange]['total'] += 1
                if order.status == OrderStatus.FILLED:
                    by_exchange[exchange]['success'] += 1
                elif order.status in [OrderStatus.CANCELLED, OrderStatus.REJECTED]:
                    by_exchange[exchange]['failed'] += 1
                    
            # Statistiques par symbole
            by_symbol = {}
            for order in self.routed_orders.values():
                symbol = order.symbol
                if symbol not in by_symbol:
                    by_symbol[symbol] = 0
                by_symbol[symbol] += 1
                
            # Statistiques par statut
            by_status = status_counts
            
            return {
                'success': True,
                'statistics': {
                    'total_routed': total_routed,
                    'active_orders': active_count,
                    'successful_orders': successful_orders,
                    'failed_orders': failed_orders,
                    'success_rate': successful_orders / total_routed if total_routed > 0 else 0,
                    'by_exchange': by_exchange,
                    'by_symbol': by_symbol,
                    'by_status': by_status
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error getting statistics: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
            
    async def _monitoring_loop(self):
        """Boucle de monitoring des ordres."""
        while self._running:
            try:
                # Mettre à jour le statut des ordres actifs
                for order_id in list(self.active_orders.keys()):
                    await self.get_order_status(order_id)
                    
                # Nettoyer les anciens ordres de l'historique
                cutoff_time = datetime.now() - timedelta(days=7)
                self.order_history = [
                    h for h in self.order_history 
                    if h['timestamp'] > cutoff_time
                ]
                
                # Attendre avant la prochaine vérification
                await asyncio.sleep(5)  # Vérification toutes les 5 secondes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {str(e)}")
                await asyncio.sleep(5)  # Attendre avant de réessayer