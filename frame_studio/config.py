from dataclasses import dataclass, field
from pathlib import Path
import os, secrets


def number(name, default):
    return int(os.getenv(name, default))


@dataclass
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(os.getenv('DATA_DIR', '/tmp/vos-frame-studio')))
    secret: str = field(default_factory=lambda: os.getenv('APP_SECRET') or secrets.token_hex(32))
    password: str = field(default_factory=lambda: os.getenv('APP_PASSWORD', ''))
    secure_cookies: bool = field(default_factory=lambda: os.getenv('COOKIE_SECURE', 'true' if os.getenv('RENDER') else 'false').lower() == 'true')
    video_bytes: int = field(default_factory=lambda: number('MAX_VIDEO_MB', 100) * 1024**2)
    zip_bytes: int = field(default_factory=lambda: number('MAX_ZIP_MB', 512) * 1024**2)
    max_seconds: int = field(default_factory=lambda: number('MAX_DURATION_SECONDS', 120))
    max_frames: int = field(default_factory=lambda: number('MAX_FRAMES', 6000))
    max_pixels: int = field(default_factory=lambda: number('MAX_PIXELS', 1920 * 1080))
    project_bytes: int = field(default_factory=lambda: number('MAX_PROJECT_MB', 1024) * 1024**2)
    total_bytes: int = field(default_factory=lambda: number('MAX_TOTAL_MB', 2048) * 1024**2)
    reserve_bytes: int = field(default_factory=lambda: number('MIN_FREE_MB', 128) * 1024**2)
    timeout: int = field(default_factory=lambda: number('JOB_TIMEOUT_SECONDS', 300))
    ttl: int = field(default_factory=lambda: number('PROJECT_TTL_SECONDS', 7200))
    queue_size: int = field(default_factory=lambda: number('MAX_QUEUED_JOBS', 4))
    ffmpeg: str = 'ffmpeg'
    ffprobe: str = 'ffprobe'
