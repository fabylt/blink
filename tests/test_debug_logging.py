import json
import logging

import aiohttp
from aiohttp import web
from blink_live_client import BlinkClient
from blink_live_client.debug import payload_summary, request_operation


def test_payload_summary_redacts_sensitive_fields_and_bounds_large_lists():
    payload = {
        "account_id": 123456, "email": "secret-email@example.test", "token": "secret-token",
        "server": "https://secret-host/path", "unexpected-private-field": "secret-value",
        "complete": True, "code": 307, "status": "pending", "polling_interval": 2,
        "clips": [{"id": 5, "camera_name": "private-camera", "size": 1024}] * 1000,
    }
    result = payload_summary(payload)
    serialized = json.dumps(result)
    assert "secret" not in serialized and "private" not in serialized
    assert "123456" not in serialized
    assert result["complete"] is True
    assert result["code"] == 307 and result["status"] == "pending"
    assert result["polling_interval"] == 2
    assert result["clips"]["count"] == 1000
    assert len(result["clips"]["sample"]) == 3
    assert result["clips"]["sample"][0]["size"] == 1024
    assert payload_summary({"status": "secret-text"})["status"] == "<redacted>"
    assert payload_summary({"size": 10**1000})["size"] == "<redacted>"


async def test_successful_request_debug_logs_redacted_payload(server, caplog):
    async def handler(_request):
        return web.json_response({
            "complete": True, "id": 99887766, "token": "secret-response-token",
            "message": "secret-vendor-message",
        })

    root = await server(handler)

    class LocalClient(BlinkClient):
        @property
        def root(self):
            return root

    async with aiohttp.ClientSession() as session:
        client = LocalClient(session, {"access_token": "secret-access-token", "expires_at": 1e12})
        with caplog.at_level(logging.DEBUG, logger="blink_live_client"):
            response = await client.request("GET", "/accounts/123456/networks/22/commands/3")
        assert response["id"] == 99887766
    assert "Blink request started: GET command status" in caplog.text
    assert "Blink response received: GET command status HTTP 200" in caplog.text
    assert '"complete": true' in caplog.text
    assert "<redacted>" in caplog.text
    for private in ("secret-", "123456", "99887766", "/accounts/"):
        assert private not in caplog.text


def test_request_labels_do_not_include_query_strings_or_identities():
    assert request_operation("/accounts/123/commands/4?token=secret") == "command status"
    assert request_operation("/private-route?token=secret") == "Blink API"
