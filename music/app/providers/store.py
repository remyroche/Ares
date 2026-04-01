from abc import ABC, abstractmethod
from app.providers.base import StoreProductResult

class StoreProvider(ABC):
    @abstractmethod
    def create_product(self, payload: dict) -> StoreProductResult: pass
