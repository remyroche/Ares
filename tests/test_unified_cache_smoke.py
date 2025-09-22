import asyncio
from src.utils.unified_cache import UnifiedCache


def test_unified_cache_sync_roundtrip():
    cache = UnifiedCache(namespace="test_sync", enable_disk=False)
    key = "k1"
    val = {"a": 1}
    assert cache.get(key) is None
    cache.set(key, val)
    assert cache.get(key) == val


def test_unified_cache_ttl_expiry():
    cache = UnifiedCache(namespace="test_ttl", enable_disk=False, default_ttl_seconds=0)
    key = "k2"
    cache.set(key, 123)
    assert cache.get(key) is None  # immediate expiry


def test_unified_cache_disk_persistence(tmp_path):
    cache = UnifiedCache(namespace="test_disk", cache_dir=str(tmp_path), enable_disk=True)
    key = "k3"
    cache.set(key, [1, 2, 3])
    # new instance same namespace and dir should load from disk
    cache2 = UnifiedCache(namespace="test_disk", cache_dir=str(tmp_path), enable_disk=True)
    assert cache2.get(key) == [1, 2, 3]


def test_unified_cache_async_helpers():
    async def run():
        cache = UnifiedCache(namespace="test_async", enable_disk=False)
        key = "k4"
        await cache.aset(key, "v4")
        return await cache.aget(key)
    assert asyncio.run(run()) == "v4"

