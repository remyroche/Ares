"""
Subaccount Management Utilities

Handles subaccount creation, management, and operations across exchanges.
"""

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass
from enum import Enum

from src.utils.logger import system_logger


class SubaccountStatus(Enum):
    """Subaccount status enumeration"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    PENDING = "pending"


@dataclass
class SubaccountInfo:
    """Subaccount information structure"""
    subaccount_id: str
    subaccount_name: str
    parent_account_id: str
    status: SubaccountStatus
    created_at: datetime
    permissions: Set[str]
    api_keys: List[str]  # List of API key IDs
    balance: Dict[str, float]  # Currency balances
    trading_enabled: bool = True
    withdrawal_enabled: bool = False
    last_activity: Optional[datetime] = None


class SubaccountManager:
    """
    Manages subaccounts for exchanges that support them.
    """
    
    def __init__(self, exchange_name: str):
        self.exchange_name = exchange_name
        self.logger = system_logger.getChild(f"SubaccountManager.{exchange_name}")
        self.subaccounts: Dict[str, SubaccountInfo] = {}
        self.parent_account_id: Optional[str] = None
        
    def set_parent_account(self, account_id: str) -> None:
        """Set the parent account ID."""
        self.parent_account_id = account_id
        self.logger.info(f"Set parent account ID: {account_id}")
    
    async def create_subaccount(
        self,
        subaccount_name: str,
        create_function,
        permissions: Optional[Set[str]] = None
    ) -> Optional[SubaccountInfo]:
        """
        Create a new subaccount.
        
        Args:
            subaccount_name: Name for the subaccount
            create_function: Async function to create subaccount on exchange
            permissions: Set of permissions for the subaccount
            
        Returns:
            SubaccountInfo if successful, None otherwise
        """
        try:
            if not self.parent_account_id:
                self.logger.error("Parent account ID not set")
                return None
                
            # Call exchange-specific creation function
            result = await create_function(subaccount_name)
            if not result:
                self.logger.error(f"Failed to create subaccount {subaccount_name}")
                return None
                
            # Extract subaccount ID from result
            subaccount_id = result.get("subaccount_id") or result.get("id")
            if not subaccount_id:
                self.logger.error("No subaccount ID returned from exchange")
                return None
                
            # Create subaccount info
            subaccount_info = SubaccountInfo(
                subaccount_id=subaccount_id,
                subaccount_name=subaccount_name,
                parent_account_id=self.parent_account_id,
                status=SubaccountStatus.ACTIVE,
                created_at=datetime.now(),
                permissions=permissions or set(),
                api_keys=[],
                balance={}
            )
            
            # Store subaccount info
            self.subaccounts[subaccount_id] = subaccount_info
            
            self.logger.info(f"Created subaccount {subaccount_name} with ID {subaccount_id}")
            return subaccount_info
            
        except Exception as e:
            self.logger.error(f"Error creating subaccount {subaccount_name}: {e}")
            return None
    
    async def get_subaccount_info(
        self,
        subaccount_id: str,
        fetch_function
    ) -> Optional[SubaccountInfo]:
        """
        Get subaccount information from exchange.
        
        Args:
            subaccount_id: ID of the subaccount
            fetch_function: Async function to fetch subaccount info
            
        Returns:
            SubaccountInfo if successful, None otherwise
        """
        try:
            result = await fetch_function(subaccount_id)
            if not result:
                return None
                
            # Update or create subaccount info
            if subaccount_id in self.subaccounts:
                subaccount_info = self.subaccounts[subaccount_id]
                subaccount_info.status = SubaccountStatus(result.get("status", "active"))
                subaccount_info.trading_enabled = result.get("trading_enabled", True)
                subaccount_info.withdrawal_enabled = result.get("withdrawal_enabled", False)
                subaccount_info.last_activity = datetime.now()
            else:
                subaccount_info = SubaccountInfo(
                    subaccount_id=subaccount_id,
                    subaccount_name=result.get("name", ""),
                    parent_account_id=self.parent_account_id or "",
                    status=SubaccountStatus(result.get("status", "active")),
                    created_at=datetime.now(),
                    permissions=set(result.get("permissions", [])),
                    api_keys=[],
                    balance=result.get("balance", {}),
                    trading_enabled=result.get("trading_enabled", True),
                    withdrawal_enabled=result.get("withdrawal_enabled", False)
                )
                self.subaccounts[subaccount_id] = subaccount_info
            
            return subaccount_info
            
        except Exception as e:
            self.logger.error(f"Error fetching subaccount info for {subaccount_id}: {e}")
            return None
    
    async def list_subaccounts(self, list_function) -> List[SubaccountInfo]:
        """
        List all subaccounts.
        
        Args:
            list_function: Async function to list subaccounts from exchange
            
        Returns:
            List of SubaccountInfo objects
        """
        try:
            result = await list_function()
            if not result:
                return []
                
            subaccounts = []
            for subaccount_data in result:
                subaccount_id = subaccount_data.get("subaccount_id") or subaccount_data.get("id")
                if not subaccount_id:
                    continue
                    
                subaccount_info = SubaccountInfo(
                    subaccount_id=subaccount_id,
                    subaccount_name=subaccount_data.get("name", ""),
                    parent_account_id=self.parent_account_id or "",
                    status=SubaccountStatus(subaccount_data.get("status", "active")),
                    created_at=datetime.now(),
                    permissions=set(subaccount_data.get("permissions", [])),
                    api_keys=[],
                    balance=subaccount_data.get("balance", {}),
                    trading_enabled=subaccount_data.get("trading_enabled", True),
                    withdrawal_enabled=subaccount_data.get("withdrawal_enabled", False)
                )
                
                subaccounts.append(subaccount_info)
                self.subaccounts[subaccount_id] = subaccount_info
            
            self.logger.info(f"Listed {len(subaccounts)} subaccounts")
            return subaccounts
            
        except Exception as e:
            self.logger.error(f"Error listing subaccounts: {e}")
            return []
    
    async def update_subaccount_permissions(
        self,
        subaccount_id: str,
        permissions: Set[str],
        update_function
    ) -> bool:
        """
        Update subaccount permissions.
        
        Args:
            subaccount_id: ID of the subaccount
            permissions: New permissions set
            update_function: Async function to update permissions
            
        Returns:
            True if successful
        """
        try:
            result = await update_function(subaccount_id, permissions)
            if not result:
                return False
                
            if subaccount_id in self.subaccounts:
                self.subaccounts[subaccount_id].permissions = permissions
                
            self.logger.info(f"Updated permissions for subaccount {subaccount_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating subaccount permissions: {e}")
            return False
    
    async def get_subaccount_balance(
        self,
        subaccount_id: str,
        balance_function
    ) -> Optional[Dict[str, float]]:
        """
        Get subaccount balance.
        
        Args:
            subaccount_id: ID of the subaccount
            balance_function: Async function to fetch balance
            
        Returns:
            Balance dictionary if successful, None otherwise
        """
        try:
            result = await balance_function(subaccount_id)
            if not result:
                return None
                
            balance = result.get("balance", {})
            
            # Update stored balance
            if subaccount_id in self.subaccounts:
                self.subaccounts[subaccount_id].balance = balance
                
            return balance
            
        except Exception as e:
            self.logger.error(f"Error fetching subaccount balance: {e}")
            return None
    
    async def transfer_funds(
        self,
        from_account: str,
        to_account: str,
        currency: str,
        amount: float,
        transfer_function
    ) -> bool:
        """
        Transfer funds between accounts.
        
        Args:
            from_account: Source account ID
            to_account: Destination account ID
            currency: Currency to transfer
            amount: Amount to transfer
            transfer_function: Async function to perform transfer
            
        Returns:
            True if successful
        """
        try:
            result = await transfer_function(from_account, to_account, currency, amount)
            if not result:
                return False
                
            self.logger.info(f"Transferred {amount} {currency} from {from_account} to {to_account}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error transferring funds: {e}")
            return False
    
    def get_subaccount(self, subaccount_id: str) -> Optional[SubaccountInfo]:
        """Get subaccount info by ID."""
        return self.subaccounts.get(subaccount_id)
    
    def get_subaccounts_by_status(self, status: SubaccountStatus) -> List[SubaccountInfo]:
        """Get subaccounts by status."""
        return [
            subaccount for subaccount in self.subaccounts.values()
            if subaccount.status == status
        ]
    
    def get_active_subaccounts(self) -> List[SubaccountInfo]:
        """Get all active subaccounts."""
        return self.get_subaccounts_by_status(SubaccountStatus.ACTIVE)
    
    def add_api_key_to_subaccount(self, subaccount_id: str, api_key_id: str) -> bool:
        """Add an API key to a subaccount."""
        if subaccount_id not in self.subaccounts:
            self.logger.warning(f"Subaccount {subaccount_id} not found")
            return False
            
        if api_key_id not in self.subaccounts[subaccount_id].api_keys:
            self.subaccounts[subaccount_id].api_keys.append(api_key_id)
            self.logger.info(f"Added API key {api_key_id} to subaccount {subaccount_id}")
            
        return True
    
    def remove_api_key_from_subaccount(self, subaccount_id: str, api_key_id: str) -> bool:
        """Remove an API key from a subaccount."""
        if subaccount_id not in self.subaccounts:
            self.logger.warning(f"Subaccount {subaccount_id} not found")
            return False
            
        if api_key_id in self.subaccounts[subaccount_id].api_keys:
            self.subaccounts[subaccount_id].api_keys.remove(api_key_id)
            self.logger.info(f"Removed API key {api_key_id} from subaccount {subaccount_id}")
            
        return True
    
    def get_subaccount_statistics(self) -> Dict[str, Any]:
        """Get subaccount statistics."""
        total_subaccounts = len(self.subaccounts)
        active_subaccounts = len(self.get_active_subaccounts())
        
        status_counts = {}
        for subaccount in self.subaccounts.values():
            status = subaccount.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
        
        total_api_keys = sum(len(subaccount.api_keys) for subaccount in self.subaccounts.values())
        
        return {
            "total_subaccounts": total_subaccounts,
            "active_subaccounts": active_subaccounts,
            "inactive_subaccounts": total_subaccounts - active_subaccounts,
            "status_distribution": status_counts,
            "total_api_keys": total_api_keys,
            "parent_account_id": self.parent_account_id
        }
    
    def cleanup_inactive_subaccounts(self) -> int:
        """Remove inactive subaccounts from memory."""
        inactive_subaccounts = [
            subaccount_id for subaccount_id, subaccount in self.subaccounts.items()
            if subaccount.status == SubaccountStatus.INACTIVE
        ]
        
        for subaccount_id in inactive_subaccounts:
            del self.subaccounts[subaccount_id]
            
        if inactive_subaccounts:
            self.logger.info(f"Cleaned up {len(inactive_subaccounts)} inactive subaccounts")
            
        return len(inactive_subaccounts)