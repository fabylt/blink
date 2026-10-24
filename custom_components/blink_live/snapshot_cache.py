"""Select the newest captured image independently of its delivery time or source."""

import asyncio
import base64
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from .blink_live_client.errors import BlinkError
from .blink_live_client.gateway import stop_process


def capture_time(value: str) -> datetime:
    stamp = datetime.fromisoformat(value)
    if stamp.tzinfo is None:
        raise ValueError("Snapshot capture time requires a timezone")
    return stamp.astimezone(UTC)


def thumbnail_time(reference, fallback: datetime) -> datetime:
    """Use Blink's timestamp when provided; otherwise use first observation."""
    reference = str(reference or "")
    stamp = reference if reference.isdecimal() else parse_qs(urlsplit(reference).query).get(
        "ts", [""]
    )[0]
    try:
        if stamp.isdecimal() and int(stamp) > 0:
            return datetime.fromtimestamp(int(stamp), UTC)
    except (ValueError, OverflowError, OSError):
        pass
    return fallback


async def video_snapshot(ffmpeg: str, path: Path) -> bytes:
    """Pick a representative image from the first three frames, without waking Blink."""
    process = await asyncio.create_subprocess_exec(
        ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error",
        "-protocol_whitelist", "file,pipe", "-i", str(path),
        "-vf", "thumbnail=3,scale=1280:720:force_original_aspect_ratio=decrease",
        "-frames:v", "1", "-threads", "1", "-f", "image2pipe", "-vcodec", "mjpeg", "pipe:1",
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
    )
    try:
        async with asyncio.timeout(20):
            image, _ = await process.communicate()
        if process.returncode or not image.startswith(b"\xff\xd8") or len(image) > 4 * 1024 * 1024:
            raise BlinkError("Could not derive a JPEG from archived video")
        return image
    finally:
        await stop_process(process)


class SnapshotCache:
    """Keep one Blink and one derived image per camera, exposing a shared selection."""

    def __init__(self, store, derived_enabled: bool):
        self.store = store
        self.derived_enabled = derived_enabled
        self.images: dict[str, bytes] = {}
        self.timestamps: dict[str, str] = {}
        self.sources: dict[str, str] = {}
        self.records: dict[str, dict] = {}
        self._lock = asyncio.Lock()

    async def load(self):
        self.records = await self.store.async_load() or {}
        for camera_key in self.records:
            self._select(camera_key)

    def _select(self, camera_key):
        records = self.records[camera_key]
        choices = [
            (capture_time(record["captured_at"]), source, record)
            for source, record in records.items()
            if source == "blink" or self.derived_enabled
        ]
        if not choices:
            return
        _, source, record = max(choices, key=lambda item: (item[0], item[1] == "blink"))
        self.images[camera_key] = base64.b64decode(record["image"])
        self.timestamps[camera_key] = record["cached_at"]
        self.sources[camera_key] = source

    async def update(self, camera_key, source, image, captured_at, reference=None):
        if source == "derived" and not self.derived_enabled:
            return
        async with self._lock:
            records = self.records.setdefault(camera_key, {})
            previous = records.get(source)
            if previous and capture_time(previous["captured_at"]) > captured_at:
                return
            records[source] = {
                "image": base64.b64encode(image).decode("ascii"),
                "captured_at": captured_at.isoformat(),
                "cached_at": datetime.now(UTC).isoformat(),
                "reference": reference,
            }
            self._select(camera_key)
            await self.store.async_save(self.records)
