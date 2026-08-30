from app.utils.hashing import generate_idempotency_key


def test_idempotency_key():
    key1 = generate_idempotency_key("123", "youtube", "longform")
    key2 = generate_idempotency_key("123", "youtube", "longform")
    key3 = generate_idempotency_key("123", "tiktok", "short")

    assert key1 == key2
    assert key1 != key3
