from app.config import settings


def get_music_provider():
    if settings.MUSIC_PROVIDER == "dummy" or settings.DEMO_MODE:
        from .implementations.dummy_music_generation import DummyMusicGenerationProvider

        return DummyMusicGenerationProvider()
    raise NotImplementedError()


def get_image_provider():
    if settings.IMAGE_PROVIDER == "dummy" or settings.DEMO_MODE:
        from .implementations.dummy_image_generation import DummyImageGenerationProvider

        return DummyImageGenerationProvider()
    raise NotImplementedError()


def get_youtube_provider():
    if not settings.YOUTUBE_ENABLED or settings.DEMO_MODE:
        from .implementations.dummy_youtube import DummyYouTubeProvider

        return DummyYouTubeProvider()
    raise NotImplementedError()


def get_tiktok_provider():
    if not settings.TIKTOK_ENABLED or settings.DEMO_MODE:
        from .implementations.dummy_tiktok import DummyTikTokProvider

        return DummyTikTokProvider()
    raise NotImplementedError()


def get_instagram_provider():
    if not settings.INSTAGRAM_ENABLED or settings.DEMO_MODE:
        from .implementations.dummy_instagram import DummyInstagramProvider

        return DummyInstagramProvider()
    raise NotImplementedError()


def get_cms_provider():
    if settings.CMS_PROVIDER == "dummy" or settings.DEMO_MODE:
        from .implementations.dummy_cms import DummyCMSProvider

        return DummyCMSProvider()
    raise NotImplementedError()


def get_store_provider():
    if settings.STORE_PROVIDER == "dummy" or settings.DEMO_MODE:
        from .implementations.dummy_store import DummyStoreProvider

        return DummyStoreProvider()
    raise NotImplementedError()


def get_analytics_provider():
    from .implementations.dummy_analytics import DummyAnalyticsProvider

    return DummyAnalyticsProvider()
