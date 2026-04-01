from app.providers.store import StoreProvider
from app.providers.base import StoreProductResult
from app.config import settings

class GenericStoreAPIProvider(StoreProvider):
    def create_product(self, payload: dict) -> StoreProductResult:
        if settings.DRY_RUN:
            return StoreProductResult(success=True, product_id="dry-run-store", product_url="https://store/dryrun")

        return StoreProductResult(success=False, error_message="Generic store implementation pending endpoint details.")
