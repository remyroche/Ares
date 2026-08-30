from app.config import settings

class LicensingService:
    @staticmethod
    def get_license_tiers():
        return [
            {
                "tier": "personal",
                "name": "Personal License",
                "price": settings.LICENSE_PRICE_PERSONAL_EUR,
                "currency": "EUR",
                "features": ["Non-commercial personal use", "Social media posts (unmonetized)"]
            },
            {
                "tier": "creator",
                "name": "Creator License",
                "price": settings.LICENSE_PRICE_CREATOR_EUR,
                "currency": "EUR",
                "features": ["Monetized social media content", "Podcasts", "Indie games"]
            },
            {
                "tier": "commercial",
                "name": "Commercial License",
                "price": settings.LICENSE_PRICE_COMMERCIAL_EUR,
                "currency": "EUR",
                "features": ["Commercial projects", "Broadcast TV", "Paid ads"]
            }
        ]
