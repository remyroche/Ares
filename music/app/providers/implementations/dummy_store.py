from app.providers.base import StoreProductResult
from app.providers.store import StoreProvider, StoreProductResult
import uuid
import os
import json


class DummyStoreProvider(StoreProvider):
    def create_product(self, payload: dict) -> StoreProductResult:
        os.makedirs("exports/site", exist_ok=True)
        product_id = str(uuid.uuid4())
        with open(f"exports/site/product_{product_id}.json", "w") as f:
            json.dump(payload, f)
        return StoreProductResult(
            success=True,
            product_id=product_id,
            product_url=f"https://dummy.store/product/{product_id}",
        )
