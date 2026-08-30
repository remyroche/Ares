import hashlib


def generate_idempotency_key(*parts: str) -> str:
    key_material = "|".join(str(p) for p in parts)
    return hashlib.sha256(key_material.encode("utf-8")).hexdigest()
