import asyncio
import json
import time
from dataclasses import replace
from unittest.mock import AsyncMock

import aiohttp
import pytest
from aiohttp import web
from blink_live_client import BlinkClient, BlinkError
from blink_live_client.errors import ApiError, AuthError, StorageBusy


def state():
    return {
        "access_token": "test-token",
        "refresh_token": "test-refresh",
        "expires_at": time.time() + 3600,
        "account_id": 1,
        "tier": "prod",
    }


async def test_discovery_and_usb_manifest_pagination(server, device, module):
    calls = []

    async def handler(request):
        assert request.headers["Authorization"] == "Bearer test-token"
        calls.append((request.method, request.path_qs))
        if request.path.endswith("homescreen"):
            return web.json_response(
                {
                    "doorbells": [
                        {"id": 3, "network_id": 2, "serial": device.serial, "name": "Front Door"}
                    ],
                    "sync_modules": [{"id": 4, "network_id": 2, "local_storage_status": "active"}],
                }
            )
        if request.path.endswith("manifest/request"):
            return web.json_response({"id": 10})
        if request.path.endswith("commands/10"):
            return web.json_response({"complete": True})
        if "/media/10" in request.path:
            page = bool(request.query)
            return web.json_response(
                {
                    "manifest_id": 11,
                    "clips": [
                        {
                            "id": 6 if page else 5,
                            "camera_name": "FrontDoor",
                            "created_at": "2026-10-19T12:00:00Z",
                        }
                    ],
                    "pagination_key": None if page else 99,
                }
            )
        raise AssertionError(request.path)

    root = await server(handler)

    class LocalClient(BlinkClient):
        @property
        def root(self):
            return root

    async with aiohttp.ClientSession() as session:
        client = LocalClient(session, state())
        observations = []
        client.observer = lambda kind, data: observations.append((kind, data))
        await client.discover()
        clips = await client.clips(client.modules[4])
        assert len(clips) == 2
        assert clips[0].camera_key == device.key
        assert clips[0].created_at.endswith("+00:00")
        assert any("pagination_key=99" in path for _, path in calls)
        assert [kind for kind, _ in observations] == ["homescreen", "usb_manifest", "usb_manifest"]
        assert observations[-1][1]["module_id"] == 4
        assert observations[-1][1]["clips"][0]["id"] == 6


@pytest.mark.parametrize("content_length", [False, True])
async def test_usb_manifest_received_in_multiple_network_chunks(
    server, device, module, content_length
):
    payload = json.dumps({
        "manifest_id": 11,
        "clips": [{"id": 5, "camera_id": device.id, "created_at": "2026-10-19T12:00:00Z"}],
    }).encode()
    calls = []

    async def handler(request):
        calls.append((request.method, request.path))
        if request.path.endswith("manifest/request"):
            return web.json_response({"id": 10})
        if request.path.endswith("commands/10"):
            return web.json_response({"complete": True})
        assert request.path.endswith("media/10")
        response = web.StreamResponse(headers={"Content-Type": "application/json"})
        if content_length:
            response.content_length = len(payload)
        await response.prepare(request)
        await response.write(payload[:17])
        await asyncio.sleep(0.02)
        await response.write(payload[17:])
        return response

    root = await server(handler)

    class LocalClient(BlinkClient):
        @property
        def root(self):
            return root

    async with aiohttp.ClientSession() as session:
        client = LocalClient(session, state())
        client.devices = {device.key: device}
        clips = await client.clips(module)
        assert len(clips) == 1
        assert clips[0].id == "5"
        assert clips[0].camera_key == device.key
    assert sum(method == "POST" for method, _ in calls) == 1


@pytest.mark.parametrize(
    ("payload", "error"),
    [
        (b'{"value":"' + b"x" * 20 + b'"}', None),
        (b'{"value":"' + b"x" * 21 + b'"}', "Blink JSON response exceeds size limit"),
        (b'{"broken":', "Invalid Blink JSON response"),
    ],
)
async def test_fragmented_json_preserves_size_limit_and_validation(
    server, monkeypatch, payload, error
):
    monkeypatch.setattr("blink_live_client.client.MAX_JSON", 32)
    calls = 0

    async def handler(request):
        nonlocal calls
        calls += 1
        response = web.StreamResponse(headers={"Content-Type": "application/json"})
        await response.prepare(request)
        await response.write(payload[:5])
        await asyncio.sleep(0.02)
        await response.write(payload[5:])
        return response

    root = await server(handler)

    class LocalClient(BlinkClient):
        @property
        def root(self):
            return root

    async with aiohttp.ClientSession() as session:
        client = LocalClient(session, state())
        if error:
            with pytest.raises(BlinkError, match=error):
                await client.request("POST", "/command", {})
        else:
            assert await client.request("POST", "/command", {}) == {"value": "x" * 20}
    assert calls == 1


@pytest.mark.parametrize("busy", [{"code": 307}, ApiError(409), ApiError(423), ApiError(429, 7)])
async def test_usb_busy_retries_then_returns_result(busy, monkeypatch):
    sleep = AsyncMock()
    monkeypatch.setattr("blink_live_client.client.asyncio.sleep", sleep)
    async with aiohttp.ClientSession() as session:
        client = BlinkClient(session, state())
        client.request = AsyncMock(side_effect=[busy, busy, {"id": 10}])
        assert await client._storage_request("POST", "/usb/manifest", {}) == {"id": 10}
        assert client.request.await_count == 3
        expected = [7, 7] if isinstance(busy, ApiError) and busy.status == 429 else [2, 4]
        assert [call.args[0] for call in sleep.await_args_list] == expected


async def test_usb_busy_retries_are_bounded_and_cancellable(monkeypatch):
    sleep = AsyncMock()
    monkeypatch.setattr("blink_live_client.client.asyncio.sleep", sleep)
    async with aiohttp.ClientSession() as session:
        client = BlinkClient(session, state())
        client.request = AsyncMock(return_value={"code": 307})
        with pytest.raises(StorageBusy, match="storage is busy"):
            await client._storage_request("POST", "/usb/manifest", {})
        assert client.request.await_count == 3
        assert sleep.await_count == 2
        client.request.reset_mock()
        sleep.side_effect = asyncio.CancelledError
        with pytest.raises(asyncio.CancelledError):
            await client._storage_request("POST", "/usb/manifest", {})
        assert client.request.await_count == 1


async def test_usb_manifest_command_recovers_from_busy(module, monkeypatch):
    sleep = AsyncMock()
    monkeypatch.setattr("blink_live_client.client.asyncio.sleep", sleep)
    async with aiohttp.ClientSession() as session:
        client = BlinkClient(session, state())
        client.request = AsyncMock(side_effect=[
            {"code": 307}, {"id": 10}, {"complete": True},
            {"manifest_id": 11, "clips": []},
        ])
        assert await client.clips(module) == []
        assert [call.args[0] for call in client.request.await_args_list] == [
            "POST", "POST", "GET", "GET"
        ]
        sleep.assert_awaited_once_with(2)


async def test_usb_download_preparation_recovers_from_busy(module, clip, monkeypatch):
    sleep = AsyncMock()
    monkeypatch.setattr("blink_live_client.client.asyncio.sleep", sleep)
    async with aiohttp.ClientSession() as session:
        client = BlinkClient(session, state())
        client.modules = {module.id: module}
        client._clips = AsyncMock(return_value=[clip])
        client.wait_command = AsyncMock()
        client.request = AsyncMock(side_effect=[{"code": 307}, {"id": 12}])

        async def chunks(_path):
            yield b"video"

        client.resource_chunks = chunks
        assert b"".join([part async for part in client.clip_chunks(clip)]) == b"video"
        assert client.request.await_count == 2
        client._clips.assert_awaited_once()
        client.wait_command.assert_awaited_once_with(module.network_id, 12)
        sleep.assert_awaited_once_with(2)


@pytest.mark.parametrize("error", [ApiError(503), AuthError("expired"), BlinkError("timeout")])
async def test_usb_does_not_replay_ambiguous_or_auth_failed_commands(error):
    async with aiohttp.ClientSession() as session:
        client = BlinkClient(session, state())
        client.request = AsyncMock(side_effect=error)
        with pytest.raises(type(error)):
            await client._storage_request("POST", "/usb/manifest", {})
        assert client.request.await_count == 1


async def test_liveview_starts_without_waiting_for_command(device, monkeypatch):
    async with aiohttp.ClientSession() as session:
        client = BlinkClient(session, state())
        calls = []

        async def request(method, path, body):
            calls.append((method, path, body))
            return {"server": "immis://example.immedia-semi.com/live/abc?client_id=1", "id": 5}

        monkeypatch.setattr(client, "request", request)
        response = await client.liveview(device)
        assert response["id"] == 5
        assert len(calls) == 1
        assert "/v2/accounts/1/networks/2/doorbells/3/liveview" in calls[0][1]


@pytest.mark.parametrize(
    "url",
    [
        "https://evil.example/secret",
        "http://rest-prod.immedia-semi.com/secret",
        "https://rest-prod.immedia-semi.com.evil.example/x",
        "//127.0.0.1/x",
        "https://user@rest-prod.immedia-semi.com/x",
        "https://rest-prde.immedia-semi.com/x",
    ],
)
async def test_resource_credentials_stay_on_account_origin(url):
    async with aiohttp.ClientSession() as session:
        client = BlinkClient(session, state())
        with pytest.raises(BlinkError):
            client.resource_url(url)


async def test_post_command_not_replayed_after_503(server):
    calls = 0

    async def handler(request):
        nonlocal calls
        calls += 1
        return web.Response(status=503)

    root = await server(handler)

    class LocalClient(BlinkClient):
        @property
        def root(self):
            return root

    async with aiohttp.ClientSession() as session:
        with pytest.raises(ApiError):
            await LocalClient(session, state()).request("POST", "/liveview", {})
        assert calls == 1


async def test_ambiguous_names_not_misattributed(device, module):
    from dataclasses import replace

    async with aiohttp.ClientSession() as session:
        client = BlinkClient(session, state())
        other = replace(device, id=99, kind="camera")
        client.devices = {d.key: d for d in [device, other]}
        assert client._clip_device(module, {"camera_name": "FrontDoor"}) is None
        assert client._clip_device(module, {"camera_id": 3}) == device


async def test_device_name_lookup_is_case_insensitive_and_rejects_ambiguity(device):
    async with aiohttp.ClientSession() as session:
        client = BlinkClient(session, state())
        client.devices = {device.key: device}
        assert client.device("front door") == device
        assert client.device(device.key) == device
        other = replace(device, id=99, name="FRONT DOOR")
        client.devices[other.key] = other
        with pytest.raises(BlinkError, match="ambiguous"):
            client.device("front door")


async def test_usb_download_refreshes_manifest_and_serializes_module(module, clip):
    async with aiohttp.ClientSession() as session:
        client = BlinkClient(session, state())
        client.modules = {module.id: module}
        latest = replace(clip, manifest_id="999")
        client._clips = AsyncMock(return_value=[latest])
        client.wait_command = AsyncMock()
        active, peak = 0, 0
        paths = []

        async def request(method, path, body):
            paths.append(path)
            return {"id": 123}

        async def resource_chunks(path):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.01)
            yield b"mp4-content"
            active -= 1

        client.request = request
        client.resource_chunks = resource_chunks

        async def consume():
            return b"".join([chunk async for chunk in client.clip_chunks(clip)])

        assert await asyncio.gather(consume(), consume()) == [b"mp4-content", b"mp4-content"]
        assert peak == 1
        assert all("/manifest/999/clip/request/5" in path for path in paths)


async def test_usb_download_rejects_removed_clip(module, clip):
    async with aiohttp.ClientSession() as session:
        client = BlinkClient(session, state())
        client.modules = {module.id: module}
        client._clips = AsyncMock(return_value=[])
        with pytest.raises(BlinkError, match="no longer available"):
            await anext(client.clip_chunks(clip))
