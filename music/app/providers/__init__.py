from app.config import settings

def get_music_provider():
    if settings.MUSIC_PROVIDER == "dummy" or settings.DEMO_MODE:
        from .implementations.dummy_music_generation import DummyMusicGenerationProvider
        return DummyMusicGenerationProvider()
    from .implementations.http_music_generation import HTTPMusicGenerationProvider
    return HTTPMusicGenerationProvider()

def get_image_provider():
    if settings.IMAGE_PROVIDER == "dummy" or settings.DEMO_MODE:
        from .implementations.dummy_image_generation import DummyImageGenerationProvider
        return DummyImageGenerationProvider()
    from .implementations.http_image_generation import HTTPImageGenerationProvider
    return HTTPImageGenerationProvider()

def get_youtube_provider():
    if not settings.YOUTUBE_ENABLED or settings.DEMO_MODE:
        from .implementations.dummy_youtube import DummyYouTubeProvider
        return DummyYouTubeProvider()
    from .implementations.youtube_api import YouTubeAPIProvider
    return YouTubeAPIProvider()

def get_tiktok_provider():
    if not settings.TIKTOK_ENABLED or settings.DEMO_MODE:
        from .implementations.dummy_tiktok import DummyTikTokProvider
        return DummyTikTokProvider()
    from .implementations.tiktok_api import TikTokAPIProvider
    return TikTokAPIProvider()

def get_instagram_provider():
    if not settings.INSTAGRAM_ENABLED or settings.DEMO_MODE:
        from .implementations.dummy_instagram import DummyInstagramProvider
        return DummyInstagramProvider()
    from .implementations.instagram_graph_api import InstagramGraphAPIProvider
    return InstagramGraphAPIProvider()

def get_cms_provider():
    if settings.CMS_PROVIDER == "dummy" or settings.DEMO_MODE:
        from .implementations.dummy_cms import DummyCMSProvider
        return DummyCMSProvider()
    from .implementations.wordpress_rest import WordpressRestProvider
    return WordpressRestProvider()

def get_store_provider():
    if settings.STORE_PROVIDER == "dummy" or settings.DEMO_MODE:
        from .implementations.dummy_store import DummyStoreProvider
        return DummyStoreProvider()
    from .implementations.generic_store_api import GenericStoreAPIProvider
    return GenericStoreAPIProvider()

def get_analytics_provider():
    from .implementations.dummy_analytics import DummyAnalyticsProvider
    return DummyAnalyticsProvider()
