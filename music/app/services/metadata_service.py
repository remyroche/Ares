from app.models import Track
from app.config import settings
from app.utils.slug import slugify


class MetadataService:
    @staticmethod
    def generate_track_metadata(track: Track, index: int) -> dict:
        title = f"{settings.BRAND_NAME} {index:03d} | LoFi Rain Focus Ambient"
        slug = slugify(title)

        hashtags = [
            "#lofi",
            "#focusmusic",
            "#studybeats",
            "#rainambience",
            "#ambientmusic",
        ]

        yt_desc = f"""{title}

Mood: {track.mood}
Genre: {track.genre}

Instrumental focus music featuring rain ambience background.
Perfect for studying, coding, or relaxing.

License this track for your own content at our site.
"""

        site_desc = f"""{title} is a {track.genre} track perfect for {track.mood}.

Licenses available:
- Personal License (EUR {settings.LICENSE_PRICE_PERSONAL_EUR}): For non-commercial personal use.
- Creator License (EUR {settings.LICENSE_PRICE_CREATOR_EUR}): For monetized social media content.
- Commercial License (EUR {settings.LICENSE_PRICE_COMMERCIAL_EUR}): For commercial projects and broadcast.
"""

        return {
            "title": title,
            "slug": slug,
            "youtube_description": yt_desc,
            "hashtags": hashtags,
            "site_description": site_desc,
            "product_name": title,
            "short_captions": [
                f"Relax with this rainy lofi beat 🌧️🎧 {hashtags[0]} {hashtags[1]}",
                f"Study vibes 📚☔ {hashtags[2]} {hashtags[3]}",
                f"Late night coding music 💻✨ {hashtags[0]} {hashtags[4]}",
            ],
        }
