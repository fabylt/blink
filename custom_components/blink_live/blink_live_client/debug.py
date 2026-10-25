"""Bounded protocol summaries for integration debug logs."""

_FIELDS = frozenset({
    "id", "account_id", "client_id", "network_id", "camera_id", "device_id", "manifest_id",
    "command_id", "code", "status", "complete", "polling_interval", "pagination_key", "size",
    "created_at", "clips", "media", "cameras", "doorbells", "owls", "sync_modules", "networks",
    "name", "camera_name", "serial", "thumbnail", "local_storage_status", "armed", "enabled",
    "online", "deleted", "message", "server", "url", "email", "token", "access_token",
    "refresh_token", "client_verification_required", "account_verification_required",
})
_STATUSES = frozenset({
    "complete", "done", "failed", "pending", "running", "busy", "active", "inactive",
    "online", "offline", "success",
})


def payload_summary(value, field="", depth=0):
    """Retain protocol flags/counts; hide identities, names, URLs and arbitrary text."""
    if depth >= 5:
        return "<nested>"
    if isinstance(value, dict):
        result = {
            key: payload_summary(item, key, depth + 1)
            for key, item in value.items() if key in _FIELDS
        }
        if extra := len(value) - len(result):
            result["other_field_count"] = extra
        return result
    if isinstance(value, list):
        return {
            "count": len(value),
            "sample": [payload_summary(item, field, depth + 1) for item in value[:3]],
        }
    if value is None:
        return None
    if isinstance(value, bool) and field in {
        "complete", "armed", "enabled", "online", "deleted", "client_verification_required",
        "account_verification_required",
    }:
        return value
    if (
        field in {"code", "size", "polling_interval"}
        and type(value) in (int, float) and -1e15 < value < 1e15
    ):
        return value
    if (
        field in {"status", "local_storage_status"}
        and isinstance(value, str) and value in _STATUSES
    ):
        return value
    return "<redacted>"


def request_operation(path: str) -> str:
    """Label internal REST routes without logging account IDs or query strings."""
    if "/local_storage/" in path:
        if "/clip/request/" in path:
            return "USB clip preparation"
        if path.endswith("/manifest/request"):
            return "USB manifest creation"
        return "USB manifest retrieval"
    if "/commands/" in path:
        return "command status"
    if path.endswith("/homescreen"):
        return "device discovery"
    if path.endswith("/users/info"):
        return "account verification"
    return "Blink API"
