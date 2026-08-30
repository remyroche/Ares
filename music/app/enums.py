import enum


class TrackStatus(str, enum.Enum):
    raw = "raw"
    approved = "approved"
    rejected = "rejected"
    published = "published"


class AssetType(str, enum.Enum):
    source_audio = "source_audio"
    master_audio = "master_audio"
    preview_audio = "preview_audio"
    loop_audio = "loop_audio"
    cover_image = "cover_image"
    youtube_video = "youtube_video"
    short_video = "short_video"
    compilation_audio = "compilation_audio"
    compilation_video = "compilation_video"
    metadata_json = "metadata_json"


class PublishChannel(str, enum.Enum):
    youtube = "youtube"
    tiktok = "tiktok"
    instagram = "instagram"
    site = "site"


class PublishContentType(str, enum.Enum):
    track_longform = "track_longform"
    track_short = "track_short"
    compilation = "compilation"
    product_page = "product_page"


class PublishJobStatus(str, enum.Enum):
    queued = "queued"
    processing = "processing"
    done = "done"
    failed = "failed"
    dead_letter = "dead_letter"


class CompilationStatus(str, enum.Enum):
    draft = "draft"
    rendered = "rendered"
    published = "published"
    failed = "failed"
