"""
RFC 4122 UUID validation for optional OFDS identifiers (network, providers).
"""

from __future__ import annotations

import uuid


def _optional_uuid_invalid_message(label: str) -> str:
    return (
        f"{label} must be a valid UUID. "
        "Leave the field empty to auto-generate one."
    )


NETWORK_ID_INVALID_MESSAGE = _optional_uuid_invalid_message("Network ID")


def optional_rfc4122_uuid_validation_error(
    value: str | None, *, label: str
) -> str | None:
    """
    If value is empty or whitespace, return None (caller may auto-generate).

    If non-empty, return None when the value is an RFC 4122 UUID string,
    otherwise return a field-specific error message.
    """
    if value is None:
        return None
    s = value.strip()
    if not s:
        return None
    invalid = _optional_uuid_invalid_message(label)
    try:
        parsed = uuid.UUID(s)
    except ValueError:
        return invalid
    if parsed.variant != uuid.RFC_4122:
        return invalid
    return None


def network_id_validation_error(network_id: str | None) -> str | None:
    """Same rules as optional_rfc4122_uuid_validation_error for Network ID."""
    return optional_rfc4122_uuid_validation_error(
        network_id, label="Network ID"
    )
