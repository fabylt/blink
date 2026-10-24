"""Snapshot freshness is capture time, never archive/download completion time."""

import asyncio
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("homeassistant")

from custom_components.blink_live.blink_live_client import BlinkError
from custom_components.blink_live.snapshot_cache import (
    SnapshotCache,
    thumbnail_time,
    video_snapshot,
)


async def test_source_selection_persistence_and_disabling_downloads():
    store = AsyncMock()
    store.async_load.return_value = None
    cache = SnapshotCache(store, True)
    await cache.load()
    now = datetime(2026, 10, 24, 12, tzinfo=UTC)
    await cache.update("front", "blink", b"blink", now, "original")
    await cache.update("front", "derived", b"video", now + timedelta(seconds=10))
    assert cache.images["front"] == b"video"
    assert cache.sources["front"] == "derived"
    # A delayed download of an older clip cannot replace the newer derived snapshot.
    await cache.update("front", "derived", b"older video", now + timedelta(seconds=5))
    assert cache.images["front"] == b"video"
    # Re-fetching an older Blink thumbnail cannot supersede newer footage either.
    await cache.update("front", "blink", b"old blink", now - timedelta(seconds=5))
    assert cache.images["front"] == b"video"
    store.async_load.return_value = cache.records
    restored = SnapshotCache(store, True)
    await restored.load()
    assert restored.images == cache.images
    disabled = SnapshotCache(store, False)
    await disabled.load()
    assert disabled.images["front"] == b"blink"
    await disabled.update("front", "derived", b"ignored", now + timedelta(seconds=30))
    assert disabled.images["front"] == b"blink"
    await cache.update("back", "derived", b"back video", now)
    await cache.update("front", "blink", b"fresh blink", now + timedelta(seconds=20))
    assert cache.images == {"front": b"fresh blink", "back": b"back video"}


def test_blink_timestamp_and_unknown_reference():
    now = datetime(2026, 10, 24, 12, tzinfo=UTC)
    assert thumbnail_time(int(now.timestamp()), now - timedelta(days=1)) == now
    assert thumbnail_time(f"/thumb.jpg?ts={int(now.timestamp())}&ext=", now) == now
    assert thumbnail_time("/opaque/thumb", now) == now


@pytest.mark.parametrize("frames", [1, 5])
async def test_derive_jpeg_from_real_short_video(ffmpeg, tmp_path, frames):
    path = tmp_path / "clip.mp4"
    process = await asyncio.create_subprocess_exec(
        ffmpeg, "-nostdin", "-loglevel", "error", "-f", "lavfi",
        "-i", "color=c=red:s=64x48:r=10", "-frames:v", str(frames),
        "-c:v", "mpeg4", str(path),
    )
    assert await process.wait() == 0
    image = await video_snapshot(ffmpeg, path)
    assert image.startswith(b"\xff\xd8") and image.endswith(b"\xff\xd9")


async def test_bad_video_cannot_create_a_snapshot(ffmpeg, tmp_path):
    path = tmp_path / "bad.mp4"
    path.write_bytes(b"bad video")
    with pytest.raises(BlinkError):
        await video_snapshot(ffmpeg, path)
