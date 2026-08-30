from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    APP_NAME: str = "MusicFactory"
    APP_ENV: str = "development"
    APP_DEBUG: bool = True
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    INTERNAL_API_KEY: str = "dev_key"
    DATABASE_URL: str
    REDIS_URL: str

    AWS_ACCESS_KEY_ID: str = ""
    AWS_SECRET_ACCESS_KEY: str = ""
    AWS_REGION: str = "us-east-1"
    S3_BUCKET: str = "music-factory"
    S3_ENDPOINT_URL: str = ""
    STORAGE_PREFIX: str = "dev"

    MUSIC_PROVIDER: str = "dummy"
    MUSIC_PROVIDER_BASE_URL: str = ""
    MUSIC_PROVIDER_API_KEY: str = ""

    IMAGE_PROVIDER: str = "dummy"
    IMAGE_PROVIDER_BASE_URL: str = ""
    IMAGE_PROVIDER_API_KEY: str = ""

    YOUTUBE_ENABLED: bool = False
    YOUTUBE_CLIENT_ID: str = ""
    YOUTUBE_CLIENT_SECRET: str = ""
    YOUTUBE_REFRESH_TOKEN: str = ""
    YOUTUBE_CHANNEL_ID: str = ""
    YOUTUBE_DEFAULT_PLAYLIST_IDS: str = ""

    TIKTOK_ENABLED: bool = False
    TIKTOK_CLIENT_KEY: str = ""
    TIKTOK_CLIENT_SECRET: str = ""
    TIKTOK_ACCESS_TOKEN: str = ""

    INSTAGRAM_ENABLED: bool = False
    INSTAGRAM_ACCESS_TOKEN: str = ""
    INSTAGRAM_BUSINESS_ACCOUNT_ID: str = ""

    CMS_PROVIDER: str = "dummy"
    CMS_BASE_URL: str = ""
    CMS_API_KEY: str = ""
    STORE_PROVIDER: str = "dummy"
    STORE_BASE_URL: str = ""
    STORE_API_KEY: str = ""

    SCHEDULE_TIMEZONE: str = "Europe/Paris"
    DAILY_TRACK_COUNT: int = 3
    DAILY_YOUTUBE_PUBLISH_COUNT: int = 1
    DAILY_SHORTS_PER_TRACK: int = 3
    WEEKLY_COMPILATION_DAY: str = "sunday"
    WEEKLY_COMPILATION_HOUR: str = "04:00"

    BRAND_NAME: str = "Tokyo Rain Study"
    SERIES_NAME: str = "Rain Focus"
    DEFAULT_GENRE: str = "lofi ambient"
    DEFAULT_MOOD: str = "rainy night focus"
    DEFAULT_BPM: int = 78
    DEFAULT_TRACK_DURATION_SEC: int = 180

    QC_MIN_DURATION_SEC: int = 150
    QC_MAX_SILENCE_START_SEC: float = 1.5
    QC_MAX_SILENCE_END_SEC: float = 2.0
    QC_MIN_SCORE: float = 0.72

    LICENSE_PRICE_PERSONAL_EUR: float = 9.0
    LICENSE_PRICE_CREATOR_EUR: float = 19.0
    LICENSE_PRICE_COMMERCIAL_EUR: float = 49.0

    DRY_RUN: bool = True
    DEMO_MODE: bool = True


settings = Settings()
